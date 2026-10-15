"""tools/python_runner.py
Runs agent-written Python in a separate process with:
  * a hard timeout (default 5 s)
  * a memory ceiling (default 256 MB), checked while the process runs
  * an import allow-list and file access limited to the workspace folder
The restrictions inside the child live in _sandbox_bootstrap.py.

Note: this is a practical sandbox for a course project. Real isolation needs
a container or VM, which is worth mentioning in the report.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import psutil
from pydantic import BaseModel, Field

from tools.base_tool import BaseTool, ToolError

BOOTSTRAP = Path(__file__).with_name("_sandbox_bootstrap.py")

DEFAULT_ALLOWED_IMPORTS = [
    "math", "statistics", "json", "csv", "re", "random", "datetime", "time",
    "itertools", "functools", "collections", "string", "decimal", "fractions",
    "typing", "textwrap", "heapq", "bisect", "operator", "copy", "dataclasses",
    "unittest",
]


class PythonRunnerArgs(BaseModel):
    code: str = Field(description="Complete Python program. Use print() to show results.")


def _clip(text: str, limit: int, keep: str = "head") -> str:
    if len(text) <= limit:
        return text
    cut = len(text) - limit
    if keep == "tail":  # for tracebacks the last lines matter most
        return f"[...{cut} characters cut...]\n" + text[-limit:]
    return text[:limit] + f"\n[...{cut} characters cut...]"


class SandboxedPythonRunner(BaseTool):
    name = "python_runner"
    description = (
        "Runs Python code in a sandbox (5 second limit, no network, no os/subprocess). "
        "Allowed imports: math, statistics, json, csv, re, random, datetime, itertools, "
        "collections, unittest and a few more. Files can only be read/written in the "
        "workspace folder. Print the result you need."
    )
    args_schema = PythonRunnerArgs

    def __init__(
        self,
        workspace: str = "workspace",
        timeout_sec: float = 5.0,
        memory_limit_mb: int = 256,
        allowed_imports: Optional[List[str]] = None,
        max_output_chars: int = 4000,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.timeout_sec = timeout_sec
        self.memory_limit_mb = memory_limit_mb
        self.allowed_imports = list(allowed_imports or DEFAULT_ALLOWED_IMPORTS)
        self.max_output_chars = max_output_chars
        self.last_peak_mb = 0.0  # used by the benchmark

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _tree(pid: int) -> list:
        # On Windows a venv's python.exe starts the real interpreter as a
        # child process, so we must watch and kill the whole process tree.
        try:
            parent = psutil.Process(pid)
            return [parent] + parent.children(recursive=True)
        except psutil.Error:
            return []

    def _kill_tree(self, proc: subprocess.Popen) -> None:
        for p in reversed(self._tree(proc.pid)):
            try:
                p.kill()
            except psutil.Error:
                pass
        try:
            proc.wait(timeout=2)
        except Exception:
            pass

    def _watch(self, proc: subprocess.Popen) -> str:
        limit = self.memory_limit_mb * 1024 * 1024
        start, peak = time.monotonic(), 0
        status = "done"
        while proc.poll() is None:
            if time.monotonic() - start > self.timeout_sec:
                self._kill_tree(proc)
                status = "timeout"
                break
            used = 0
            for p in self._tree(proc.pid):
                try:
                    used += p.memory_info().rss
                except psutil.Error:
                    pass
            peak = max(peak, used)
            if used > limit:
                self._kill_tree(proc)
                status = "memory"
                break
            time.sleep(0.02)
        self.last_peak_mb = peak / (1024 * 1024)
        return status

    # -- main entry ------------------------------------------------------
    def execute(self, code: str) -> str:
        self.workspace.mkdir(parents=True, exist_ok=True)
        env = {"PYTHONIOENCODING": "utf-8"}
        for key in ("SYSTEMROOT", "PATH"):
            if key in os.environ:
                env[key] = os.environ[key]

        with tempfile.TemporaryDirectory(
            dir=self.workspace, prefix=".sandbox_", ignore_cleanup_errors=True
        ) as tmp:
            tmp = Path(tmp)
            code_file = tmp / "agent_code.py"
            code_file.write_text(code, encoding="utf-8")
            out_path, err_path = tmp / "stdout.txt", tmp / "stderr.txt"
            cmd = [
                sys.executable, "-I", str(BOOTSTRAP), str(code_file),
                str(self.workspace), json.dumps(self.allowed_imports),
            ]
            # Output goes to files, not pipes, so a chatty program cannot block.
            with open(out_path, "wb") as out_f, open(err_path, "wb") as err_f:
                proc = subprocess.Popen(
                    cmd, cwd=self.workspace, env=env, stdin=subprocess.DEVNULL,
                    stdout=out_f, stderr=err_f,
                )
                status = self._watch(proc)
            stdout = out_path.read_text(encoding="utf-8", errors="replace").strip()
            stderr = err_path.read_text(encoding="utf-8", errors="replace").strip()

        if status == "timeout":
            raise ToolError(f"execution timed out after {self.timeout_sec} seconds")
        if status == "memory":
            raise ToolError(f"memory limit of {self.memory_limit_mb} MB exceeded")
        if proc.returncode != 0:
            raise ToolError(
                f"runtime error (exit code {proc.returncode}):\n"
                + _clip(stderr, self.max_output_chars, keep="tail")
            )
        result = _clip(stdout, self.max_output_chars) if stdout else ""
        if stderr:  # e.g. the "OK" summary printed by unittest
            result += ("\n" if result else "") + _clip(stderr, 1000, keep="tail")
        return result or "[code ran successfully with no output]"
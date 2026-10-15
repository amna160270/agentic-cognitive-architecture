import time

import pytest

from tools.python_runner import SandboxedPythonRunner


@pytest.fixture
def runner(tmp_path):
    return SandboxedPythonRunner(workspace=str(tmp_path / "ws"), timeout_sec=3)


def test_print_works(runner):
    r = runner.run(code="print(sum(range(10)))")
    assert r.ok and r.output == "45"


def test_allowed_imports(runner):
    r = runner.run(code="import statistics\nprint(statistics.mean([1, 2, 3]))")
    assert r.ok and r.output == "2"


def test_no_output_message(runner):
    assert "no output" in runner.run(code="x = 1").output


def test_runtime_error_has_traceback(runner):
    r = runner.run(code="a = 1\nprint(a / 0)")
    assert not r.ok
    assert "ZeroDivisionError" in r.output and "agent_code" in r.output


def test_syntax_error(runner):
    r = runner.run(code="def broken(:\n    pass")
    assert not r.ok and "SyntaxError" in r.output


@pytest.mark.parametrize("module", ["os", "subprocess", "socket", "sys", "shutil", "ctypes", "importlib", "io"])
def test_blocked_imports(runner, module):
    r = runner.run(code=f"import {module}")
    assert not r.ok and "blocked by the sandbox" in r.output


def test_from_import_blocked(runner):
    r = runner.run(code="from os import system")
    assert not r.ok and "blocked" in r.output


def test_dunder_import_blocked(runner):
    r = runner.run(code="__import__('subprocess')")
    assert not r.ok and "blocked" in r.output


def test_eval_exec_input_removed(runner):
    for snippet in ("eval('1+1')", "exec('x=1')", "input()"):
        r = runner.run(code=snippet)
        assert not r.ok and "NameError" in r.output


def test_timeout_kills_infinite_loop(runner):
    start = time.monotonic()
    r = runner.run(code="while True:\n    pass")
    assert not r.ok and "timed out" in r.output
    assert time.monotonic() - start < 8


def test_memory_limit(tmp_path):
    runner = SandboxedPythonRunner(workspace=str(tmp_path / "ws"), memory_limit_mb=100, timeout_sec=10)
    r = runner.run(code="x = [0] * 100_000_000\nimport time\ntime.sleep(5)")
    assert not r.ok and "memory limit" in r.output


def test_files_inside_workspace_work(runner):
    code = "with open('a.txt', 'w') as f:\n    f.write('hi')\nprint(open('a.txt').read())"
    r = runner.run(code=code)
    assert r.ok and r.output == "hi"
    assert (runner.workspace / "a.txt").read_text() == "hi"


def test_files_outside_workspace_blocked(runner, tmp_path):
    r = runner.run(code="open('../outside.txt', 'w').write('x')")
    assert not r.ok and "PermissionError" in r.output
    assert not (tmp_path / "outside.txt").exists()


def test_long_output_is_truncated(runner):
    r = runner.run(code="print('x' * 20000)")
    assert r.ok and len(r.output) < 4200


def test_unittest_pass_and_fail(runner):
    base = (
        "import unittest\n"
        "class T(unittest.TestCase):\n"
        "    def test_a(self):\n"
        "        self.assertEqual(1 + 1, {expected})\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n"
    )
    assert runner.run(code=base.format(expected=2)).ok
    bad = runner.run(code=base.format(expected=3))
    assert not bad.ok and "FAILED" in bad.output


def test_missing_code_argument(runner):
    r = runner.run()
    assert not r.ok and "Invalid arguments" in r.output


def test_temp_files_cleaned_up(runner):
    runner.run(code="print(1)")
    leftovers = [p for p in runner.workspace.iterdir() if p.name.startswith(".sandbox_")]
    assert leftovers == []


@pytest.mark.parametrize("code", [
    "import random\nprint(random._os.getcwd())",
    "import collections\nprint(collections._sys)",
    "print(().__class__.__bases__)",
    "print(getattr(__builtins__, 'open'))",
])
def test_private_attribute_escapes_blocked(runner, code):
    r = runner.run(code=code)
    assert not r.ok
    assert "blocked by the sandbox" in r.output or "NameError" in r.output


def test_normal_class_code_still_works(runner):
    code = (
        "class A:\n"
        "    def __init__(self):\n"
        "        self.x = 1\n"
        "class B(A):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.y = 2\n"
        "print(B().x + B().y, B.__name__)"
    )
    r = runner.run(code=code)
    assert r.ok and r.output == "3 B"
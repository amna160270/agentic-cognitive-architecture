
"""Benchmark the local ReAct agent on a list of tasks."""

import json
import time
from pathlib import Path
from typing import Callable

from core.agent import ReActAgent


def run_benchmark(
    agent_factory: Callable[[], ReActAgent],
    tasks: list[str],
    output_file: str = "benchmark/results.json",
) -> dict:
    """Record task success and elapsed time, then save JSON results."""
    results = []

    for task in tasks:
        start = time.perf_counter()
        try:
            answer = agent_factory().run(task)
            elapsed = time.perf_counter() - start
            success = bool(answer.strip()) and not answer.startswith("Stopped after")
            results.append({
                "task": task,
                "success": success,
                "elapsed_seconds": round(elapsed, 3),
                "answer": answer,
            })
        except Exception as exc:
            elapsed = time.perf_counter() - start
            results.append({
                "task": task,
                "success": False,
                "elapsed_seconds": round(elapsed, 3),
                "error": f"{type(exc).__name__}: {exc}",
            })

    total = len(results)
    summary = {
        "total_tasks": total,
        "successful_tasks": sum(r["success"] for r in results),
        "success_rate": (
            sum(r["success"] for r in results) / total if total else 0.0
        ),
        "average_elapsed_seconds": (
            round(sum(r["elapsed_seconds"] for r in results) / total, 3)
            if total else 0.0
        ),
        "results": results,
    }

    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    print("Import successful. Create an agent_factory and task list to run benchmarks.")

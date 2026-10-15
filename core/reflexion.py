
"""Reflexion: learn from failed attempts and retry with feedback."""

from typing import Callable, Optional

from core.agent import ReActAgent
from core.local_llm import LocalLLMClient


class Reflexion:
    """Reflect on failed attempts and retry with lessons learned."""

    def __init__(
        self,
        llm: LocalLLMClient,
        max_retries: int = 2,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        self.llm = llm
        self.max_retries = max_retries
        self.lessons: list[str] = []

    def reflect(self, task: str, result: str) -> str:
        """Generate a short lesson from an unsuccessful attempt."""
        prompt = (
            "Analyze this agent attempt and give one concise lesson "
            "to improve the next attempt.\n"
            f"Task: {task}\nAttempt result: {result}\nLesson:"
        )
        response = self.llm.generate(
            prompt,
            system="You are a concise agent self-reflection assistant.",
        )
        lesson = response.text.strip()
        if lesson:
            self.lessons.append(lesson)
        return lesson

    def run(
        self,
        task: str,
        agent_factory: Callable[[str], ReActAgent],
        success_check: Optional[Callable[[str], bool]] = None,
    ) -> str:
        """Retry a task, adding reflection lessons after failed attempts."""
        check = success_check or (
            lambda answer: bool(answer.strip())
            and not answer.startswith("Stopped after")
        )

        result = ""
        for attempt in range(self.max_retries + 1):
            agent = agent_factory("\n".join(self.lessons))
            result = agent.run(task)

            if check(result):
                return result

            if attempt < self.max_retries:
                self.reflect(task, result)

        return result

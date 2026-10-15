
"""ReAct agent loop using a local LLM and registered tools."""

from typing import Optional

from core.local_llm import LocalLLMClient
from core.deliberation import (
    build_system_prompt,
    parse_react_output,
    ParseError,
)
from tools.base_tool import ToolRegistry


class ReActAgent:
    """Run a local LLM in a tool-using ReAct loop."""

    def __init__(
        self,
        llm: LocalLLMClient,
        tools: ToolRegistry,
        max_steps: int = 8,
        reflections: str = "",
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")

        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.reflections = reflections

    def run(self, task: str) -> str:
        """Execute a task through Thought, Action, and Observation steps."""
        messages = [
            {
                "role": "system",
                "content": build_system_prompt(
                    self.tools.describe(),
                    self.reflections,
                ),
            },
            {"role": "user", "content": task},
        ]

        for step in range(self.max_steps):
            result = self.llm.chat(messages)
            messages.append({"role": "assistant", "content": result.text})

            try:
                response = parse_react_output(result.text)
            except ParseError as exc:
                messages.append({
                    "role": "user",
                    "content": f"Observation: Invalid response. {exc}",
                })
                continue

            if response.is_final:
                return response.final_answer or "No final answer provided."

            if response.action is None:
                messages.append({
                    "role": "user",
                    "content": "Observation: No valid tool action provided.",
                })
                continue

            action = response.action
            tool_result = self.tools.execute(
                action.tool_name,
                action.tool_input,
            )

            messages.append({
                "role": "user",
                "content": f"Observation: {tool_result.output}",
            })

        return (
            f"Stopped after {self.max_steps} steps without a final answer."
        )

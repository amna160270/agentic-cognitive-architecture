
"""Short-term working memory with token-based context compaction."""

import re
from typing import Any, Callable, Dict, List, Optional

Message = Dict[str, Any]
Summarizer = Callable[[str], str]


def count_tokens(text: str) -> int:
    """Estimate token count without loading another model."""
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def _fallback_summarizer(text: str) -> str:
    """Provide a deterministic fallback summary."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "No earlier interaction details available."
    return "Earlier interaction highlights:\n" + "\n".join(lines[-12:])


def make_llm_summarizer(llm: Any) -> Summarizer:
    """Create a summarizer using the supplied local LLM client."""

    def summarize(text: str) -> str:
        result = llm.generate(
            prompt=(
                "Summarize these agent interactions as concise bullet points.\n"
                "Preserve important facts, decisions, and conclusions.\n"
                "Preserve tool names, inputs, outputs, errors, and numbers.\n"
                "Include thoughts, observations, and final answers.\n"
                "Do not invent details. Do not omit important information.\n\n"
                f"Interactions:\n{text}"
            ),
            system=(
                "You create accurate executive episodic summaries. "
                "Preserve all important information from the input."
            ),
        )
        summary = result.text.strip()
        return summary or _fallback_summarizer(text)

    return summarize


class WorkingMemory:
    """Maintain recent messages and compact older interaction steps."""

    def __init__(
        self,
        max_messages: int = 20,
        max_tokens: int = 2500,
        keep_last_steps: int = 3,
        summarizer: Optional[Summarizer] = None,
        token_counter: Callable[[str], int] = count_tokens,
        llm: Optional[Any] = None,
    ) -> None:
        if max_messages < 1:
            raise ValueError("max_messages must be at least 1")
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if keep_last_steps < 1:
            raise ValueError("keep_last_steps must be at least 1")
        if summarizer is not None and llm is not None:
            raise ValueError("Pass either summarizer or llm, not both")

        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.keep_last_steps = keep_last_steps
        self.summarizer = (
            summarizer
            if summarizer is not None
            else make_llm_summarizer(llm)
            if llm is not None
            else _fallback_summarizer
        )
        self.token_counter = token_counter
        self._messages: List[Message] = []
        self._next_step_id = 1

    def add(
        self,
        role: str,
        content: str,
        step: Optional[Any] = None,
        kind: Optional[str] = None,
    ) -> None:
        """Add a message; shared step IDs form one interaction step."""
        if not isinstance(role, str) or not role.strip():
            raise ValueError("role is required")
        if not isinstance(content, str):
            raise ValueError("content must be a string")

        message: Message = {"role": role, "content": content}

        if role.lower() != "system":
            if step is None:
                step = f"message-{self._next_step_id}"
                self._next_step_id += 1
            message["step"] = step

        if kind:
            message["kind"] = kind

        self._messages.append(message)
        self._compact_if_needed()
        self._enforce_message_limit()

    def _compact_if_needed(self) -> None:
        """Summarize older steps while preserving system and recent steps."""
        if self.token_counter(self.get_context()) <= self.max_tokens:
            return

        system_messages = [
            m for m in self._messages if m["role"].lower() == "system"
        ]
        existing_summaries = [
            m for m in self._messages
            if m.get("kind") == "executive_summary"
        ]
        regular_messages = [
            m for m in self._messages
            if m["role"].lower() != "system"
            and m.get("kind") != "executive_summary"
        ]

        groups: List[List[Message]] = []
        group_positions: Dict[Any, int] = {}

        for index, message in enumerate(regular_messages):
            step_id = message.get("step", f"unassigned-{index}")
            try:
                group_index = group_positions.get(step_id)
            except TypeError:
                step_id = repr(step_id)
                group_index = group_positions.get(step_id)

            if group_index is None:
                group_index = len(groups)
                groups.append([])
                group_positions[step_id] = group_index

            groups[group_index].append(message)

        if len(groups) <= self.keep_last_steps:
            return

        older_groups = groups[:-self.keep_last_steps]
        recent_groups = groups[-self.keep_last_steps:]

        old_messages = [
            message for group in older_groups for message in group
        ]
        summary_input_messages = existing_summaries + old_messages
        summary_input = "\n".join(
            f"{m['role']}: {m['content']}"
            for m in summary_input_messages
        )

        summary_text = self.summarizer(summary_input).strip()
        if not summary_text:
            summary_text = _fallback_summarizer(summary_input)

        summary_message: Message = {
            "role": "summary",
            "kind": "executive_summary",
            "content": summary_text,
        }
        recent_messages = [
            message for group in recent_groups for message in group
        ]

        self._messages = (
            system_messages + [summary_message] + recent_messages
        )

    def _enforce_message_limit(self) -> None:
        """Apply message-count limit without dropping system prompts."""
        system_messages = [
            m for m in self._messages if m["role"].lower() == "system"
        ]
        other_messages = [
            m for m in self._messages if m["role"].lower() != "system"
        ]

        allowed = max(0, self.max_messages - len(system_messages))
        if len(other_messages) > allowed:
            other_messages = other_messages[-allowed:] if allowed else []

        self._messages = system_messages + other_messages

    def get_messages(self) -> List[Message]:
        """Return copies of stored messages."""
        return [message.copy() for message in self._messages]

    def get_context(self) -> str:
        """Format memory as prompt-ready text."""
        return "\n".join(
            f"{m['role']}: {m['content']}" for m in self._messages
        )

    def clear(self) -> None:
        """Remove all messages and reset automatic step IDs."""
        self._messages.clear()
        self._next_step_id = 1

    def __len__(self) -> int:
        return len(self._messages)

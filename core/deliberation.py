"""core/deliberation.py
ReAct prompt + robust parser for small-model outputs (Task 1).

Small models often break the format a little (single quotes, trailing commas,
a missing closing brace, code fences). This parser tolerates those mistakes
and raises ParseError only when the output is really unusable.
"""
import ast
import json
import re
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, ValidationError


class ParseError(ValueError):
    """Raised when the model output cannot be turned into an action or answer."""


class AgentAction(BaseModel):
    thought: str = ""
    tool_name: str
    tool_input: Dict[str, Any] = Field(default_factory=dict)


class AgentResponse(BaseModel):
    is_final: bool
    thought: str = ""
    action: Optional[AgentAction] = None
    final_answer: Optional[str] = None


SYSTEM_PROMPT_TEMPLATE = """You are an autonomous agent powered by a local Small Language Model.
You work in a loop: Thought -> Action -> Observation, until you can answer.

Available Tools:
{tool_descriptions}

To use a tool, reply EXACTLY in this format and then STOP:
Thought: <why you need this tool>
Action: <exact tool name from the list>
Action Input: {{"param": "value"}}

When you have enough information, reply EXACTLY:
Thought: <final reasoning>
Final Answer: <your answer to the user>

Rules:
1. Action Input must be valid JSON on one line.
2. Never invent tool names.
3. Never write the Observation yourself; wait for it.
4. If an Observation shows an error, think about the cause before the next action.
"""


def build_system_prompt(tool_descriptions: str, reflections: str = "") -> str:
    prompt = SYSTEM_PROMPT_TEMPLATE.format(tool_descriptions=tool_descriptions)
    if reflections:
        prompt += "\nLessons from earlier failures:\n" + reflections + "\n"
    return prompt


def parse_error_feedback(err: Exception) -> str:
    """Message sent back to the model when its output could not be parsed."""
    return (
        f"Observation: Your last reply was not in the required format ({err}). "
        "Reply again using exactly 'Thought:', then 'Action:' and "
        "'Action Input:' (JSON), or 'Thought:' and 'Final Answer:'."
    )


def _strip_fences(text: str) -> str:
    return re.sub(r"```[a-zA-Z]*", "", text)


def _extract_balanced_json(text: str) -> str:
    """Return the first {...} block, handling nesting and quoted braces.
    If the model forgot closing braces, they are added."""
    start = text.find("{")
    if start == -1:
        raise ParseError("no '{' found after Action Input")
    depth, in_str, quote, escape = 0, False, "", False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                in_str = False
            continue
        if ch in "\"'":
            in_str, quote = True, ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:].rstrip() + "}" * max(depth, 1)


def _loads_lenient(blob: str) -> Dict[str, Any]:
    no_trailing_commas = re.sub(r",\s*([}\]])", r"\1", blob)
    for candidate in (blob, no_trailing_commas):
        try:
            obj = json.loads(candidate, strict=False)  # allows raw newlines in strings
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        try:
            obj = ast.literal_eval(candidate)  # single quotes, True/False/None
            if isinstance(obj, dict):
                return obj
        except (ValueError, SyntaxError):
            pass
    raise ParseError(f"malformed JSON in Action Input: {blob[:150]}")


def parse_react_output(raw_text: str) -> AgentResponse:
    """Turn raw model text into an AgentResponse (action or final answer)."""
    text = _strip_fences(raw_text)

    thought_m = re.search(
        r"Thought:\s*(.*?)(?=\s*Action:|\s*Final Answer:|\Z)", text, re.DOTALL
    )
    thought = thought_m.group(1).strip() if thought_m else ""

    # An Action wins over a Final Answer: the model cannot know the answer
    # before it has seen the tool's Observation.
    action_m = re.search(r"Action:\s*([A-Za-z_][A-Za-z0-9_]*)", text)
    if action_m:
        input_m = re.search(r"Action Input:\s*(.*)", text, re.DOTALL)
        if not input_m:
            raise ParseError("found 'Action:' but no 'Action Input:'")
        tool_input = _loads_lenient(_extract_balanced_json(input_m.group(1)))
        try:
            action = AgentAction(
                thought=thought, tool_name=action_m.group(1), tool_input=tool_input
            )
        except ValidationError as e:
            raise ParseError(f"invalid action: {e}")
        return AgentResponse(is_final=False, thought=thought, action=action)

    if "Final Answer:" in text:
        answer = text.split("Final Answer:", 1)[1].strip()
        if not answer:
            raise ParseError("empty Final Answer")
        return AgentResponse(is_final=True, thought=thought, final_answer=answer)

    raise ParseError("no 'Action:' or 'Final Answer:' found")
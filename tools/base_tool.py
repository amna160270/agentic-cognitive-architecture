"""tools/base_tool.py
MCP-style tool interface: every tool has a name, a description, a Pydantic
args_schema and an execute() method. The registry turns the schemas into the
text that is injected into the agent's system prompt (Task 2).
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Type

from pydantic import BaseModel, ValidationError


class ToolError(Exception):
    """Raised by a tool when it fails in an expected way (bad input, timeout...)."""


@dataclass
class ToolResult:
    """ok=False tells the Reflexion loop that the action failed."""
    ok: bool
    output: str


class BaseTool(ABC):
    name: str
    description: str
    args_schema: Type[BaseModel]

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Do the work. Raise ToolError on expected failures."""

    def run(self, **kwargs) -> ToolResult:
        """Validate arguments, call execute(), never raise."""
        try:
            args = self.args_schema(**kwargs)
        except ValidationError as e:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
                for err in e.errors()
            )
            return ToolResult(False, f"Invalid arguments for {self.name}: {problems}")
        try:
            return ToolResult(True, str(self.execute(**args.model_dump())))
        except ToolError as e:
            return ToolResult(False, f"{self.name} error: {e}")
        except Exception as e:  # a tool bug must not crash the agent loop
            return ToolResult(False, f"{self.name} crashed: {type(e).__name__}: {e}")

    def to_schema(self) -> Dict[str, Any]:
        """MCP-style tool description with a JSON schema for the inputs."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.args_schema.model_json_schema(),
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> BaseTool:
        if tool.name in self._tools:
            raise ValueError(f"tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
        return tool

    def names(self) -> List[str]:
        return list(self._tools)

    def schemas(self) -> List[Dict[str, Any]]:
        return [t.to_schema() for t in self._tools.values()]

    def describe(self) -> str:
        """Text block for the system prompt, generated from the schemas."""
        blocks = []
        for schema in self.schemas():
            props = schema["inputSchema"].get("properties", {})
            required = set(schema["inputSchema"].get("required", []))
            example = ", ".join(
                '"%s": <%s>' % (k, v.get("type", "any")) for k, v in props.items()
            )
            lines = [f"- {schema['name']}: {schema['description']}",
                     "  Action Input: {" + example + "}"]
            for k, v in props.items():
                opt = "" if k in required else ", optional"
                lines.append(f"    {k} ({v.get('type', 'any')}{opt}): {v.get('description', '')}")
            blocks.append("\n".join(lines))
        return "\n".join(blocks)

    def execute(self, name: str, args: Dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                False, f"Tool '{name}' not found. Available tools: {', '.join(self._tools)}"
            )
        return tool.run(**args)
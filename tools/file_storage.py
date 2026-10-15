"""tools/file_storage.py
Lets the agent save reports/code/CSV files, but only inside workspace/output.
"""
from pathlib import Path

from pydantic import BaseModel, Field

from tools.base_tool import BaseTool, ToolError

MAX_BYTES = 1_000_000


class FileStorageArgs(BaseModel):
    filename: str = Field(description="File name such as 'report.md' (no folders needed)")
    content: str = Field(description="Full text to write into the file")


class FileStorageTool(BaseTool):
    name = "file_storage"
    description = "Saves text (reports, code, CSV) to a file in the output folder."
    args_schema = FileStorageArgs

    def __init__(self, output_dir: str = "workspace/output") -> None:
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def execute(self, filename: str, content: str) -> str:
        if not filename.strip():
            raise ToolError("filename is empty")
        filename = filename.replace("\\", "/")  # same rules on Windows and Linux
        if filename.startswith("/") or ":" in filename:
            raise ToolError("absolute paths are not allowed")
        if ".." in filename.split("/"):
            raise ToolError("'..' is not allowed in the path")
        target = (self.output_dir / filename).resolve()
        if self.output_dir not in target.parents:
            raise ToolError("path escapes the output folder")
        data = content.encode("utf-8")
        if len(data) > MAX_BYTES:
            raise ToolError(f"content larger than {MAX_BYTES} bytes")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return f"Saved {len(data)} bytes to output/{target.relative_to(self.output_dir).as_posix()}"
from typing import Optional
from pydantic import BaseModel, Field


# ──────────────────────────
# File tools
# ──────────────────────────
class ReadFileInput(BaseModel):
    file_path: str = Field(..., description="Full path to the file to read")


class ReadFileOutput(BaseModel):
    content: str


class WriteFileInput(BaseModel):
    file_path: str = Field(..., description="Full path to the file to write")
    content: str = Field(..., description="Content that will be written to the file")


class WriteFileOutput(BaseModel):
    message: str


# ──────────────────────────
# Bash tool
# ──────────────────────────
class RunBashInput(BaseModel):
    command: str = Field(..., description="Bash command to execute")


class RunBashOutput(BaseModel):
    output: str


# ──────────────────────────
# Python-code tool
# ──────────────────────────
class RunCodeInput(BaseModel):
    code: str = Field(..., description="Python code snippet to execute")


class RunCodeOutput(BaseModel):
    output: str


# ──────────────────────────
# Scratch-pad tools
# ──────────────────────────
class WriteScratchpadInput(BaseModel):
    content: str = Field(..., description="Text to write to the scratch-pad")
    mode: Optional[str] = Field(
        "append",
        description="append (default) or overwrite the existing scratch-pad",
    )


class WriteScratchpadOutput(BaseModel):
    message: str


class ReadScratchpadInput(BaseModel):
    query: Optional[str] = Field(
        "", description="Ignored – may be empty or any string (‘read’, etc.)"
    )


class ReadScratchpadOutput(BaseModel):
    content: str
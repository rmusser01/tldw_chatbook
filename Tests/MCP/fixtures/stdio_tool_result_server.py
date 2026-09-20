"""Local stdio fixture: execute a tool result selected by a private JSON file."""

import json
import os
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class _Request(BaseModel):
    """Bounded fixture request shape; production protocol models stay separate."""

    model_config = ConfigDict(strict=True)
    jsonrpc: Literal["2.0"]
    method: str
    id: int | str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class _InitializeParams(BaseModel):
    """The fixture needs a valid version before echoing initialization."""

    model_config = ConfigDict(strict=True)
    protocolVersion: str


def _error(request_id: int | str | None, code: int, message: str) -> None:
    """Emit bounded JSON-RPC errors without reflecting malformed input."""
    print(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": code, "message": message},
            }
        ),
        flush=True,
    )


def main() -> None:
    """Serve tool results from a caller-owned fixture directory.

    Raises:
        ValueError: State or trace escapes the caller-owned fixture root.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from tldw_chatbook.Utils.input_validation import validate_json_size
    from tldw_chatbook.Utils.path_validation import (
        validate_canonical_directory,
        validate_path,
    )

    root = validate_canonical_directory(sys.argv[3])
    state = validate_path(sys.argv[1], root, redact_paths=True)
    trace = validate_path(sys.argv[2], root, redact_paths=True)
    for line in sys.stdin:
        try:
            if not validate_json_size(line):
                raise ValueError("Request too large")
            payload = json.loads(line)
        except (ValueError, RecursionError):
            _error(None, -32700, "Invalid JSON")
            continue
        try:
            request = _Request.model_validate(payload)
        except ValidationError:
            _error(None, -32600, "Invalid request")
            continue
        method = request.method
        if method == "initialize":
            try:
                _InitializeParams.model_validate(request.params)
            except ValidationError:
                if "id" in request.model_fields_set:
                    _error(request.id, -32602, "Invalid initialize parameters")
                continue
        with trace.open("a") as out:
            out.write(json.dumps({"pid": os.getpid(), "method": method}) + "\n")
        if "id" not in request.model_fields_set:
            continue
        if method == "initialize":
            result = {
                "protocolVersion": request.params["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "tool-result-fixture", "version": "1"},
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "review_echo",
                        "description": "A local tool result fixture.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"message": {"type": "string"}},
                            "required": ["message"],
                        },
                    }
                ]
            }
        elif method == "resources/list":
            result = {"resources": []}
        elif method == "prompts/list":
            result = {"prompts": []}
        elif method == "tools/call":
            result = json.loads(state.read_text())
        else:
            result = {}
        print(
            json.dumps({"jsonrpc": "2.0", "id": request.id, "result": result}),
            flush=True,
        )


if __name__ == "__main__":
    main()

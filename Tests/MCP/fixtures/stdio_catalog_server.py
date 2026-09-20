"""Local JSON-RPC fixture with a catalog controlled by its private state file."""

import json
import os
import sys
import time
from pathlib import Path


def main() -> None:
    """Serve bounded JSON-RPC discovery from isolated test-owned state files.

    Raises:
        ValueError: A state or trace path escapes the caller-owned fixture root.
        TimeoutError: Held initialization is not released within fifteen seconds.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from tldw_chatbook.Utils.path_validation import (
        validate_canonical_directory,
        validate_path,
    )

    root = validate_canonical_directory(sys.argv[3])
    state_path = validate_path(sys.argv[1], root, redact_paths=True)
    trace_path = validate_path(sys.argv[2], root, redact_paths=True)
    for line in sys.stdin:
        request = json.loads(line)
        method = request.get("method")
        with trace_path.open("a") as trace:
            trace.write(json.dumps({"pid": os.getpid(), "method": method}) + "\n")
        if "id" not in request:
            continue
        state = json.loads(state_path.read_text())
        deadline = time.monotonic() + 15
        while method == "initialize" and state.get("hold"):
            if time.monotonic() >= deadline:
                raise TimeoutError("Fixture initialization was not released")
            time.sleep(0.01)
            state = json.loads(state_path.read_text())
        if state.get("fail"):
            response = {"error": {"code": -32603, "message": "Fixture unavailable"}}
        else:
            version = state["version"]
            if method == "initialize":
                result = {
                    "protocolVersion": request["params"]["protocolVersion"],
                    "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
                    "serverInfo": {"name": "catalog-review-fixture", "version": "1"},
                }
            elif method == "tools/list":
                result = {
                    "tools": [
                        {
                            "name": f"{version}_tool",
                            "description": "Local review fixture",
                            "inputSchema": {"type": "object", "properties": {}},
                        }
                    ]
                }
            elif method == "resources/list":
                result = {
                    "resources": [
                        {
                            "uri": f"fixture://{version}",
                            "name": f"{version}_resource",
                        }
                    ]
                }
            elif method == "prompts/list":
                result = {"prompts": [{"name": f"{version}_prompt"}]}
            else:
                result = {}
            response = {"result": result}
        print(
            json.dumps({"jsonrpc": "2.0", "id": request["id"], **response}), flush=True
        )


if __name__ == "__main__":
    main()

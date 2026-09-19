"""Local JSON-RPC fixture with a catalog controlled by its private state file."""

import json
import os
import sys
import time
from pathlib import Path


def main():
    state_path, trace_path = map(Path, sys.argv[1:3])
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

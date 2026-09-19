"""Local stdio fixture: execute a tool result selected by a private JSON file."""

import json
import os
import sys
from pathlib import Path


def main():
    state, trace = map(Path, sys.argv[1:3])
    for line in sys.stdin:
        request = json.loads(line)
        method = request.get("method")
        with trace.open("a") as out:
            out.write(json.dumps({"pid": os.getpid(), "method": method}) + "\n")
        if "id" not in request:
            continue
        if method == "initialize":
            result = {
                "protocolVersion": request["params"]["protocolVersion"],
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
            json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}),
            flush=True,
        )


if __name__ == "__main__":
    main()

"""Local held MCP calls: progress precedes the final JSON-RPC response."""

import json
import sys
import threading
import time
from pathlib import Path

write_lock = threading.Lock()
root = Path(sys.argv[1])


def send(payload):
    with write_lock:
        print(json.dumps({"jsonrpc": "2.0", **payload}), flush=True)


def tool(request):
    params = request["params"]
    name = params["name"]
    token = params.get("_meta", {}).get("progressToken")
    if token is not None:
        send(
            {
                "method": "notifications/progress",
                "params": {"progressToken": token, "progress": 1, "message": name},
            }
        )
    while not (root / "release").exists():
        time.sleep(0.02)
    send(
        {
            "id": request["id"],
            "result": {"content": [{"type": "text", "text": name + " final"}]},
        }
    )
    if token is not None:
        send(
            {
                "method": "notifications/progress",
                "params": {"progressToken": token, "progress": 2, "message": "late"},
            }
        )


for line in sys.stdin:
    request = json.loads(line)
    if request.get("method") == "tools/call":
        threading.Thread(target=tool, args=(request,), daemon=True).start()

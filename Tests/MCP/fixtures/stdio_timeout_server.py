"""Local held MCP calls: slow tools outlive the wire deadline, fast ones do not."""

import json
import sys
import threading
import time


def send(payload):
    print(json.dumps({"jsonrpc": "2.0", **payload}), flush=True)


def tool(request):
    params = request["params"]
    name = params["name"]
    if name == "slow":
        # Sleep past the client's test deadline before replying.
        time.sleep(float(params.get("arguments", {}).get("seconds", 3)))
    send(
        {
            "id": request["id"],
            "result": {"content": [{"type": "text", "text": name + " final"}]},
        }
    )


for line in sys.stdin:
    request = json.loads(line)
    if request.get("method") == "tools/call":
        threading.Thread(target=tool, args=(request,), daemon=True).start()

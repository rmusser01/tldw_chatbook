#!/usr/bin/env python3
"""waitfor.py <socket> <text> [timeout_s] [--absent] -> polls capture-pane every 40 ms; prints elapsed seconds
from invocation until text appears (or disappears with --absent)."""
import subprocess, sys, time
sock, needle = sys.argv[1], sys.argv[2]
timeout = float(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else 30.0
absent = "--absent" in sys.argv
t0 = time.time()
while True:
    out = subprocess.run(["tmux", "-L", sock, "capture-pane", "-p"], capture_output=True, text=True).stdout
    hit = needle in out
    if hit != absent:
        print(f"{'gone' if absent else 'seen'} {needle!r} after {time.time()-t0:.2f}s at {time.strftime('%H:%M:%S')}")
        sys.exit(0)
    if time.time() - t0 > timeout:
        print(f"TIMEOUT {timeout}s waiting for {needle!r}"); sys.exit(1)
    time.sleep(0.04)

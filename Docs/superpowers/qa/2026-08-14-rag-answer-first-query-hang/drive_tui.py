"""Drive the TUI to the FIRST RAG Answer query and time it (TASK-15810 T1).

Every control is re-located in the SAME capture the click is computed from
(lessons-live-verification: layout-shift phantom "dead click"), and columns
are CHARACTER positions, never byte offsets.
"""

import os
import subprocess
import sys
import time

SOCK = os.environ.get("TMUX_SOCK", "rag15810")
PID = sys.argv[1]
TIMEOUT_S = float(sys.argv[2]) if len(sys.argv) > 2 else 420.0


def tmux(*args):
    return subprocess.run(
        ["tmux", "-L", SOCK, *args], capture_output=True, text=True
    ).stdout


def capture():
    return tmux("capture-pane", "-p").split("\n")


def find(label, offset=4):
    """Return (row, col) 1-based CHARACTER coords for `label`, or None."""
    for i, line in enumerate(capture(), 1):
        c = line.find(label)
        if c >= 0:
            return i, c + 1 + offset
    return None


def click(label, offset=4, settle=2.0):
    hit = find(label, offset)
    if hit is None:
        raise SystemExit(f"NOT FOUND on screen: {label!r}")
    row, col = hit
    tmux("send-keys", "-l", f"\x1b[<0;{col};{row}M")
    time.sleep(0.2)
    tmux("send-keys", "-l", f"\x1b[<0;{col};{row}m")
    time.sleep(settle)
    print(f"  click {label!r} at row={row} col={col}", flush=True)


def cpu():
    out = subprocess.run(
        ["ps", "-p", PID, "-o", "%cpu="], capture_output=True, text=True
    ).stdout.strip()
    return out or "?"


print("STEP: nav -> Library", flush=True)
click("⌃3 Library", offset=3, settle=4.0)
print("STEP: rail row -> Search / RAG", flush=True)
click("Search / RAG", offset=4, settle=3.0)
print("STEP: mode toggle -> RAG Answer", flush=True)
click("RAG Answer", offset=4, settle=3.0)
mode_line = [row for row in capture() if "mode:" in row]
print("  mode line:", mode_line[0].strip() if mode_line else "(none)", flush=True)

print("STEP: focus query input", flush=True)
click("Ask or search Library sources", offset=4, settle=1.0)
QUERY = "how do I use the command palette"
tmux("send-keys", "-l", QUERY)
time.sleep(1.0)

gate = [row for row in capture() if "Blocked" in row or "Unavailable" in row]
print("  run-gate lines before submit:", gate[:2], flush=True)

print(f"STEP: SUBMIT (Enter). timeout={TIMEOUT_S}s", flush=True)
t0 = time.time()
tmux("send-keys", "Enter")
first_evidence = None
last_status = None
while time.time() - t0 < TIMEOUT_S:
    time.sleep(1.0)
    el = time.time() - t0
    pane = "\n".join(capture())
    status = None
    for token in ("searching · ", "Answering", "results for '", "No matches", "Blocked"):
        if token in pane:
            status = token
            break
    if status != last_status:
        print(f"  t={el:6.1f}s cpu={cpu():>5}%  status-token={status!r}", flush=True)
        last_status = status
    elif int(el) % 10 == 0:
        print(f"  t={el:6.1f}s cpu={cpu():>5}%  status-token={status!r}", flush=True)
    if "results for '" in pane:
        first_evidence = el
        break

print("=" * 70, flush=True)
if first_evidence is None:
    print(f"VERDICT: NO EVIDENCE ROW within {TIMEOUT_S}s (timeout)", flush=True)
else:
    print(f"VERDICT: FIRST EVIDENCE ROW at t={first_evidence:.2f}s", flush=True)
print("final cpu:", cpu(), flush=True)
for line in capture():
    if any(
        t in line
        for t in ("results for '", "searching ·", "Evidence · top", "Answering", "Blocked")
    ):
        print("  |", line.strip()[:170], flush=True)

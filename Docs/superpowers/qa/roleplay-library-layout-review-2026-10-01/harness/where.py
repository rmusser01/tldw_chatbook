#!/usr/bin/env python3
"""Find text on a tmux pane. Usage: where.py <socket> <text> [--click [nth|L|B]]
Prints 'row col' (1-based, character columns) of each match; with --click clicks the nth (default 1) match's centre."""
import subprocess, sys
sock, needle = sys.argv[1], sys.argv[2]
lines = subprocess.run(["tmux", "-L", sock, "capture-pane", "-p"], capture_output=True, text=True).stdout.split("\n")
hits = []
for r, line in enumerate(lines, 1):
    start = 0
    while (i := line.find(needle, start)) != -1:
        hits.append((r, i + 1)); start = i + 1
for h in hits:
    print(h[0], h[1])
if "--click" in sys.argv:
    idx = sys.argv.index("--click")
    arg = sys.argv[idx + 1] if len(sys.argv) > idx + 1 else "1"
    if arg == "L" and hits:          # leftmost match (e.g. a rail entry vs a body duplicate)
        hits.sort(key=lambda h: (h[1], h[0])); nth = 1
    elif arg == "B" and hits:        # bottom-most match
        hits.sort(key=lambda h: (-h[0], h[1])); nth = 1
    elif arg in ("L", "B"):
        nth = 1
    else:
        nth = int(arg)
    if len(hits) >= nth:
        r, c = hits[nth - 1]; c += len(needle) // 2
        for s in ("M", "m"):
            subprocess.run(["tmux", "-L", sock, "send-keys", "-l", f"\x1b[<0;{c};{r}{s}"])
        print(f"clicked {c},{r}")
    else:
        print("NO MATCH to click", file=sys.stderr); sys.exit(1)

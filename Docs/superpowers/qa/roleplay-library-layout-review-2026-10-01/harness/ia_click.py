#!/usr/bin/env python3
"""ia_click.py <socket> <text> [minrow] [mincol] : click first match with row>=minrow and col>=mincol."""
import subprocess, sys
sock, needle = sys.argv[1], sys.argv[2]
minrow = int(sys.argv[3]) if len(sys.argv) > 3 else 4
mincol = int(sys.argv[4]) if len(sys.argv) > 4 else 1
lines = subprocess.run(["tmux","-L",sock,"capture-pane","-p"],capture_output=True,text=True).stdout.split("\n")
for r, line in enumerate(lines, 1):
    if r < minrow: continue
    i = line.find(needle, mincol-1)
    if i != -1:
        c = i + 1 + len(needle)//2
        for s in ("M","m"):
            subprocess.run(["tmux","-L",sock,"send-keys","-l",f"\x1b[<0;{c};{r}{s}"])
        print(f"clicked {c},{r}"); sys.exit(0)
print("NO MATCH", file=sys.stderr); sys.exit(1)

"""TASK-33003.6 review round 1: frame and divider contrast on destinations
outside Settings and Chat settings, from `capture-pane -e` dumps (other.sh).

Usage: measure_frames.py FILE...   Tallies every box-drawing cell below the
nav bar by (line colour, the colour under it) and prints the most common.
"""
import sys
from collections import Counter

from ansi_cells import ratio, rows

LINES = set("─│┌┐└┘╭╮╰╯├┤┬┴┼━┃")
for path in sys.argv[1:]:
    tally = Counter(
        (c[1], c[2])
        for r in rows(path)[3:]
        for c in r
        if c[0] in LINES and c[1] and c[2]
    )
    print(path)
    for (fg, bg), n in tally.most_common(4):
        print(f"  {n:5d} cells  line {fg} on {bg}: {ratio(fg, bg):.2f}:1")

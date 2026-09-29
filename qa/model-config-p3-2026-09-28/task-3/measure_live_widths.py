"""Measure painted Chat settings field widths from `tmux capture-pane -e` dumps.

Usage: measure_live_widths.py CAPTURE.ansi.txt [...]

At rest a field's fill can equal its section's background, so the width is
read from the FOCUSED field: its thick edge cell (█ or ▌) is followed by the
focus fill, which paints exactly the field's cells (an Input's selected value
and cursor paint their own colours inside it). The field is the span from the
edge to the last focus-fill cell before the row's own background (the cell
left of the edge) resumes. For each capture this
prints the focused field's label and its width, edge included (TASK-33003.3
AC#1 "edge included").
"""
import re
import sys


def parse(line):
    """Return [(char, bg)] for one captured line (truecolor or 256 SGR)."""
    bg = None
    cells = []
    for m in re.finditer(r"\x1b\[([0-9;:]*)m|([^\x1b])", line):
        if m.group(2) is not None:
            cells.append((m.group(2), bg))
            continue
        p = [int(x) if x else 0 for x in re.split("[;:]", m.group(1))] or [0]
        j = 0
        while j < len(p):
            v = p[j]
            if v == 0:
                bg = None
            elif v in (38, 48) and j + 1 < len(p) and p[j + 1] == 2:
                if v == 48:
                    bg = tuple(p[j + 2 : j + 5])
                j += 4
            elif v in (38, 48) and j + 1 < len(p) and p[j + 1] == 5:
                if v == 48:
                    bg = ("256", p[j + 2])
                j += 2
            elif v == 49:
                bg = None
            j += 1
    return cells


for path in sys.argv[1:]:
    found = None
    for line in open(path, encoding="utf-8").read().split("\n"):
        cells = parse(line)
        text = "".join(c for c, _ in cells)
        m = re.search(r"│  ?\s*([A-Z][A-Za-z ()%]+?)\s*[█▌]", text)
        if not m:
            continue
        edge = m.end() - 1
        rest, fill = cells[edge - 1][1], {cells[edge][1], cells[edge + 1][1]}
        stop = edge + 1
        while stop < len(cells) and cells[stop][1] != rest:
            stop += 1
        end = max(x for x in range(edge, stop) if cells[x][1] in fill) + 1
        found = (m.group(1).strip(), end - edge, text[edge + 1 : end].strip())
        break
    name = path.rsplit("/", 1)[-1]
    if found:
        print(f"{name:26} {found[0]:26} width={found[1]:3} value={found[2]!r}")
    else:
        print(f"{name:26} (no focused labelled field)")

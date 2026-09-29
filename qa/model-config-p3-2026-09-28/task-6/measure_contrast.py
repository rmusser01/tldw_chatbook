"""TASK-33003.6 AC#7: contrast of boundaries, focus cues, focused buttons and
list highlights, read from `tmux capture-pane -e` dumps made by drive.sh.

Usage: measure_contrast.py DIR PREFIX   (PREFIX = <tree>-<theme>-<cols>x<rows>)
Every ratio is WCAG contrast between two painted RGB colours.
"""
import sys
from pathlib import Path

from ansi_cells import ratio, rows

D, P = Path(sys.argv[1]), sys.argv[2]


def load(state):
    R = rows(D / f"{P}-{state}.ansi.txt")
    return R, ["".join(c[0] for c in r) for r in R]


def last_row(T, needle):
    return max(i for i, t in enumerate(T) if needle in t)


def painted(cell):
    """The colour a cell shows: the glyph's fg for a full block, else its bg."""
    return cell[1] if cell[0] in "█▌▐" else cell[2]


def out(name, a, b, what):
    print(f"  {name:<46} {ratio(a, b):5.2f}:1   {a} vs {b} ({what})")


# 1. Chat settings frame and field edges
R, T = load("modal-open")
y = T.index(next(t for t in T if "Conversation settings" in t)); x = T[y].index("Conversation settings")
fy = y - 1
while T[fy].rfind("╭", 0, x) < 0:
    fy -= 1
fx = T[fy].rfind("╭", 0, x)
print(f"{P}")
out("Chat settings frame vs modal panel", R[fy][fx][1], R[fy + 1][fx + 2][2], "╭ fg vs inside")
out("Chat settings frame vs backdrop", R[fy + 6][fx][1], R[fy + 6][fx - 1][2], "left │ fg vs the cell outside it")
for label in ("Endpoint", "Model "):
    ly = next(i for i, t in enumerate(T) if f"  {label}" in t and "│" in t[t.index(label):])
    ex = T[ly].index("│", T[ly].index(label))
    out(f"field edge ({label.strip()}) vs field fill", R[ly][ex][1], R[ly][ex + 1][2], "│ fg vs next cell")
    out(f"field edge ({label.strip()}) vs modal panel", R[ly][ex][1], R[fy + 1][fx + 2][2], "│ fg vs panel")

# 2. Chat settings Select overlay highlight
R, T = load("modal-select")
opts = {}
for name in ("Automatic", "Custom", "Select"):
    oy = next(i for i, t in enumerate(T) if f"│  {name}" in t)
    ox = T[oy].index(f"│  {name}") + 3
    opts[name] = R[oy][ox]
bar = next(c for c in opts.values() if "b" in c[3])
rest = next(c for c in opts.values() if "b" not in c[3] and c[2] != bar[2])
out("Select highlight bar vs other option", bar[2], rest[2], "fill vs fill")
out("Select highlight label vs bar", bar[1], bar[2], "text vs fill")

# 3. Buttons (focus must not lower contrast)
def button(state, label):
    R, T = load(state)
    by = last_row(T, label); bx = T[by].index(label)
    fill = R[by][bx][2]
    lo = bx
    while lo > 0 and R[by][lo - 1][2] == fill:
        lo -= 1
    hi = bx
    while hi + 1 < len(R[by]) and R[by][hi + 1][2] == fill:
        hi += 1
    return fill, R[by][lo - 1][2], R[by][hi + 1][2], "".join(sorted(R[by][bx][3]))

for rest_state, focus_state, label in (
    ("modal-button-rest", "modal-button-focus", "Use for this conversation"),
    ("pm-rest", "pm-button-focus", "Test Provider"),
):
    if not (D / f"{P}-{focus_state}.ansi.txt").exists():
        print(f"  {label}: focus capture missing"); continue
    f0, l0, r0, a0 = button(rest_state, label)
    f1, l1, r1, a1 = button(focus_state, label)
    print(f"  {label!r}: rest fill {f0} [{a0}] -> focused fill {f1} [{a1}]")
    out("   rest fill vs left surface", f0, l0, "")
    out("   focused fill vs left surface", f1, l1, "")
    out("   rest fill vs right surface", f0, r0, "")
    out("   focused fill vs right surface", f1, r1, "")

# 4. Settings rail focus cue (active row = Overview, inactive = Providers & Models)
Rr, Tr = load("rail-rest")
for state, label in (("rail-active-focus", "Overview (view)"), ("rail-inactive-focus", "Providers & Models")):
    R, T = load(state)
    ry = next(i for i, t in enumerate(T) if label in t)
    ex = T[ry].rfind("│", 0, T[ry].index(label)) + 2
    edge, fill = R[ry][ex], R[ry][ex + 1]
    was = Rr[ry][ex]
    print(f"  rail {label!r}: edge glyph {edge[0]!r} (rest {was[0]!r}); label attrs {''.join(sorted(R[ry][T[ry].index(label)][3]))}")
    out("   focus cue vs same cell unfocused", painted(edge), painted(was), "")
    out("   focus cue vs focused row fill", painted(edge), fill[2], "")

# 5. Providers & Models list highlight
R, T = load("pm-rest")
def list_cell(name):
    y = next(i for i, t in enumerate(T) if f"│ │ {name} " in t)
    return R[y][T[y].index(f"│ │ {name} ") + 4]
bar = next((list_cell(n) for n in ("llama.cpp",) ), None)
other = list_cell("KoboldCpp")
out("P&M provider highlight bar vs other row", bar[2], other[2], "fill vs fill")
out("P&M provider highlight label vs bar", bar[1], bar[2], "text vs fill")

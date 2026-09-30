"""Parent AC#13 (final head): the Settings rail focus cue's contrast, read from
live_final.sh's capture-pane -e dumps. Usage: measure_rail.py <cols>x<rows>
Same method as task-6/measure_contrast.py section 4."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "task-6"))
from ansi_cells import ratio, rows

D, size = Path(__file__).resolve().parent, sys.argv[1]


def load(state):
    R = rows(D / f"live-{state}-{size}.ansi.txt")
    return R, ["".join(c[0] for c in r) for r in R]


def painted(cell):
    return cell[1] if cell[0] in "█▌▐" else cell[2]


Rr, _ = load("rail-rest")
for state, label in (("rail-active-focus", "Overview (view)"), ("rail-inactive-focus", "Providers & Models")):
    R, T = load(state)
    ry = next(i for i, t in enumerate(T) if label in t)
    ex = T[ry].rfind("│", 0, T[ry].index(label)) + 2
    edge, was = R[ry][ex], Rr[ry][ex]
    print(f"{size} {label!r}: edge {edge[0]!r} (rest {was[0]!r}) "
          f"cue vs unfocused {ratio(painted(edge), painted(was)):.2f}:1, "
          f"cue vs row fill {ratio(painted(edge), R[ry][ex + 1][2]):.2f}:1")

"""Print painted cells of a tmux `capture-pane -e` dump.
Usage: cells.py FILE ROW COLSTART COLEND  (1-based row, 0-based cols)"""
import re, sys
def parse(line, state=None):
    """Cells of one line; ``state`` carries SGR across lines, as tmux emits
    a colour only when it changes (a line starts in the previous line's)."""
    fg, bg, attrs = state if state else (None, None, set())
    attrs = set(attrs); cells = []
    for m in re.finditer(r"\x1b\[([0-9;:]*)m|([^\x1b])", line):
        if m.group(2) is not None:
            cells.append((m.group(2), fg, bg, frozenset(attrs))); continue
        p = [int(x) if x else 0 for x in re.split("[;:]", m.group(1))] or [0]
        j = 0
        while j < len(p):
            v = p[j]
            if v == 0: fg = bg = None; attrs = set()
            elif v in (1, 4, 7): attrs.add({1: "b", 4: "u", 7: "rev"}[v])
            elif v == 22: attrs.discard("b")
            elif v == 24: attrs.discard("u")
            elif v == 27: attrs.discard("rev")
            elif v == 38 and p[j + 1] == 2: fg = tuple(p[j + 2:j + 5]); j += 4
            elif v == 48 and p[j + 1] == 2: bg = tuple(p[j + 2:j + 5]); j += 4
            elif v == 39: fg = None
            elif v == 49: bg = None
            j += 1
    parse.state = (fg, bg, attrs)
    return cells
def lum(c):
    def ch(v):
        v /= 255; return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = c; return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)
def ratio(a, b):
    hi, lo = sorted((lum(a), lum(b)), reverse=True); return (hi + 0.05) / (lo + 0.05)
def rows(path):
    out = []; parse.state = None
    for l in open(path, encoding="utf-8").read().split("\n"):
        out.append(parse(l, parse.state))
    return out
if __name__ == "__main__":
    R = rows(sys.argv[1]); r = int(sys.argv[2]); a, b = int(sys.argv[3]), int(sys.argv[4])
    for i, c in enumerate(R[r - 1][a:b], a):
        print(i, repr(c[0]), c[1], c[2], "".join(sorted(c[3])))

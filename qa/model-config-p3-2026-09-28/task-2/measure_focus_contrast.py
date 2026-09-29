"""Measure Chat settings field contrast from a `tmux capture-pane -e` dump.

Usage: measure_focus_contrast.py CAPTURE.ansi.txt [THEME_LABEL]

Reads the painted cells of two rows in the Model view with Advanced
generation expanded:
  * the focused Temperature row, whose edge cell is the thick focus edge `█`
  * the resting Top P row, whose edge cell is the one-column rest edge `│`
and prints WCAG contrast ratios for the rest edge, the focus edge and the
focus fill against the rest fill, plus the value text on each fill.
"""
import re
import sys


def _lum(rgb):
    def ch(v):
        v /= 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4

    r, g, b = rgb
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def ratio(a, b):
    hi, lo = sorted((_lum(a), _lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def parse(line):
    """Return [(char, fg, bg, attrs)] for one captured line (truecolor SGR)."""
    fg = bg = None
    attrs = set()
    cells = []
    for m in re.finditer(r"\x1b\[([0-9;:]*)m|([^\x1b])", line):
        if m.group(2) is not None:
            cells.append((m.group(2), fg, bg, frozenset(attrs)))
            continue
        p = [int(x) if x else 0 for x in re.split("[;:]", m.group(1))] or [0]
        j = 0
        while j < len(p):
            v = p[j]
            if v == 0:
                fg = bg = None
                attrs = set()
            elif v == 1:
                attrs.add("bold")
            elif v == 22:
                attrs.discard("bold")
            elif v == 38 and p[j + 1] == 2:
                fg = tuple(p[j + 2 : j + 5])
                j += 4
            elif v == 48 and p[j + 1] == 2:
                bg = tuple(p[j + 2 : j + 5])
                j += 4
            elif v == 39:
                fg = None
            elif v == 49:
                bg = None
            j += 1
    return cells


def field_row(lines, label, edge):
    """Cells of the row holding `label`, and the index of its edge glyph."""
    for line in lines:
        cells = parse(line)
        text = "".join(c[0] for c in cells)
        at = text.find(label)
        if at < 0:
            continue
        col = text.find(edge, at + len(label))
        if col >= 0:
            return cells, col
    raise SystemExit(f"row {label!r} with edge {edge!r} not found")


def main():
    path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else path
    lines = open(path, encoding="utf-8").read().split("\n")
    fcells, fcol = field_row(lines, "Temperature", "█")
    rcells, rcol = field_row(lines, "Top P", "│")
    f_edge, f_val = fcells[fcol], fcells[fcol + 2]
    r_edge, r_val = rcells[rcol], rcells[rcol + 2]
    surface = rcells[rcol - 2][2]  # label-column background left of the edge
    rest_fill, focus_fill = r_val[2], f_val[2]
    rows = [
        ("rest edge `│` fg vs its cell bg", ratio(r_edge[1], r_edge[2]), r_edge[1], r_edge[2]),
        ("rest edge fg vs rest fill", ratio(r_edge[1], rest_fill), r_edge[1], rest_fill),
        ("focus edge `█` fg vs surface", ratio(f_edge[1], surface), f_edge[1], surface),
        ("focus edge fg vs focus fill", ratio(f_edge[1], focus_fill), f_edge[1], focus_fill),
        ("focus fill vs rest fill", ratio(focus_fill, rest_fill), focus_fill, rest_fill),
        ("focused value text vs focus fill", ratio(f_val[1], focus_fill), f_val[1], focus_fill),
        ("resting value text vs rest fill", ratio(r_val[1], rest_fill), r_val[1], rest_fill),
    ]
    print(f"## {label}")
    print(f"focused value bold={'bold' in f_val[3]} resting value bold={'bold' in r_val[3]}")
    for name, value, a, b in rows:
        print(f"{name:34s} {value:5.2f}:1   {a} on {b}")


if __name__ == "__main__":
    main()

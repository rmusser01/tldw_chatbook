#!/usr/bin/env python3
"""Render an .ansi tmux capture to PNG. Usage: shot_png.py <in.ansi> <out.png>"""
import sys, pathlib
from rich.console import Console
from rich.text import Text
src, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
ansi = src.read_text()
lines = ansi.rstrip("\n").split("\n")
cols = max((Text.from_ansi(l).cell_len for l in lines), default=80)
con = Console(record=True, width=cols, color_system="truecolor", file=open("/dev/null", "w"), force_terminal=True)
for line in lines:
    con.print(Text.from_ansi(line), no_wrap=True, overflow="ignore", crop=True)
html = con.export_html(inline_styles=True, code_format=(
    "<html><head><meta charset='utf-8'><style>body{{margin:0;background:#151719}}"
    "pre{{margin:0;padding:6px;font-family:'Menlo','DejaVu Sans Mono',monospace;font-size:13px;line-height:1.18;"
    "color:#e0e0e0;background:#151719}} pre span{{display:inline-block;height:1.18em;vertical-align:top}}</style></head><body><pre>{code}</pre></body></html>"))
hp = out.with_suffix(".html"); hp.write_text(html)
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": int(cols * 8.0) + 20, "height": 400}, device_scale_factor=1)
    pg.goto(hp.resolve().as_uri())
    pg.screenshot(path=str(out), full_page=True)
    b.close()
hp.unlink()
print(out)

# Radio frame visual qualification

Production-CSS Console harness, owned disposable SQLite profile; 2026-10-23.
These are **presentation fixtures**, not evidence of a live speech provider.
`visual_probe.py` is the exact final capture probe (copy into Tests/UI for pytest
fixture discovery). The headless harness inherits NO_COLOR; the probe removes its
Monochrome/NoColor presentation filters before capture. Source artwork is the
packaged Samira neutral WebP. No image color transformation is applied.

Two batched inspection rounds completed. The first revealed the harness's
monochrome filter and a rasterizer font fallback. The final SVGs retain authored
colors (289 distinct colors at 200×72). PNG previews use CairoSVG with installed
`/opt/homebrew/lib/libcairo.2.dylib` and Menlo substituted for the SVG font name
only; the SVGs are the unmodified Textual output. No dependency was installed.

- 200×72: identity, recessed contain-fit portrait, transmitting status/meter and
  existing reaction controls are visible. Bounds stay fixed across speech states
  in the separate mounted tests.
- 120×36: portrait remains contained and the Character body's existing scrollbar
  provides access to lower status/controls. The probe scrolls the portrait into
  view; it does not claim all controls are simultaneously visible.
- 80×30: the existing narrow-shell policy collapses the Context rail; the Console
  composer and context-rail control remain present.

The once-only Impeccable detector returned `[]` for portrait_radio.py,
character_expression_avatar.py, portrait_performance_editor.py and source panel CSS.

# Library reader grip polish visual QA

Production-shaped evidence for TASK-30020. The capture mounts the real Library
Media reader with the same CSS stack as `TldwCli`, selects and settles a real
fixture item, focuses the Library grip, and verifies the final compositor paint.
Its config and application data stay under
`/private/tmp/tldw-chatbook-library-reader-grip-polish-qa`.

Run from the repository root:

```bash
.venv/bin/python Docs/superpowers/qa/library-reader-grip-polish-2026-09/capture_grips.py
```

The capture script uses Homebrew's local Cairo library when present to rasterize
Textual's SVG output; it does not install or download anything.

## Verified result

| Terminal | Effective panes | Library arrow rows | Items arrow row | Focus paint | Containment |
|---|---|---:|---:|---|---|
| 160×50 | Library + Items + Reader | 15, 27 of 43 | 21 | neutral background + accent endcaps | contained |
| 120×35 | Items + Reader | 9, 18 of 28 | 13 | neutral background + accent endcaps | contained |
| 100×30 | Items + Reader | 8, 16 of 25 | 12 | neutral background + accent endcaps | contained |
| 80×24 | Reader; both grips reachable | 6, 12 of 19 | 9 | neutral background + accent endcaps | contained |

The rendered pass confirms:

- the furthest-left Library grip forms the approved triangle with two arrows at
  approximately 35% and 65% and the Items arrow at the midpoint;
- the four-character arrows remain fully visible because focus uses only the
  top and bottom endcaps rather than an overpainting side outline;
- hover, focus, and pressed states retain the rest background with no stripe,
  reverse-video, tint, or filled-accent treatment;
- all visible children and the permanent Reader stay inside the shell at every
  required terminal size; and
- the existing production-shaped cross-reader resize suite passes for Media,
  Collections, Conversations, Notes, Prompts, and Skills.

The precise painted geometry and state checks are recorded in `geometry.json`.
PNG and SVG captures are retained for each size.

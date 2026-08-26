# Library Media Reader visual QA

Production-shaped captures for the NetNewsWire-style Library Media Reader.
`capture_reader.py` mounts the same CSS stack as `TldwCli`, uses deterministic
local media fixtures, loads a real Reader detail, checks both five-column
grips and Reader containment, and records the effective layout in
`geometry.json`. It forces `TLDW_CONFIG_PATH` to a scratch file under
`/private/tmp`; it never reads or writes the real user profile.

Run from the repository root:

```bash
.venv/bin/python Docs/superpowers/qa/library-media-reader-2026-08/capture_reader.py
```

Expected responsive progression:

- 160×50: Library + Items + Reader.
- 120×35: Items + Reader, with Library responsive-collapsed.
- 100×30: Items + Reader when the allocated shell permits it.
- 80×24: Reader only, with both grips still reachable.

Responsive results are effective session state only. They do not modify the
persisted open/collapsed preferences.

## Verified result

| Terminal | Shell | Effective panes | Reader | Result |
|---|---:|---|---:|---|
| 160×50 | 156 | Library 28 + Items 40 | 78 | contained |
| 120×35 | 116 | Items 40 | 66 | contained |
| 100×30 | 100 | Items 40 | 50 | contained |
| 80×24 | 80 | both collapsed, both grips visible | 70 | contained |

The visual pass caught two mounted-but-not-painted problems at 100×30:
Textual's default Button minimum pushed **More** and **Info** beyond Reader,
and Items retained **Selected · loading preview** after detail settlement.
The committed source-CSS minimum and the settled Items projection fix both;
the final PNG/SVG captures show every Reader action and **Loaded in Reader**.

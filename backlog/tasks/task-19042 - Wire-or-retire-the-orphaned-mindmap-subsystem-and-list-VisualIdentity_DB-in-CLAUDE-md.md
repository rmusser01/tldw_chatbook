---
id: TASK-19042
title: >-
  Wire-or-retire the orphaned mindmap subsystem (Tools/Mind_Map +
  MindmapViewer + write-only tables); list VisualIdentity_DB in CLAUDE.md
status: To Do
assignee: []
created_date: '2026-08-20 08:40'
labels:
  - cleanup
  - dead-code
  - docs
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The wave-2 close-out queue carried "mindmap viewer unmounted+stubbed;
`DB/Mindmap_DB.py` missing while CLAUDE.md claims it exists". Half of that is
already resolved on dev: TASK-15481 deleted `DB/Mindmap_DB.py` (it called a
nonexistent `self.get_connection()`), and commit `004872669` dropped the retired
DB modules from CLAUDE.md's Data Layer list. What remains at dev `1bf7f234e` is
the rest of the subsystem, verified orphaned:

- `Tools/Mind_Map/` — 7 modules, 2,803 LOC (model, renderer, integration,
  exporter, mermaid parser, jsoncanvas handler, an `anytree-demo.py` script) —
  imported by nothing in production except `UI/Widgets/MindmapViewer.py`.
- `UI/Widgets/MindmapViewer.py` — exported behind a guarded import in
  `UI/Widgets/__init__.py` but **never constructed anywhere** (whole-tree grep
  for `MindmapViewer(` hits only the class definition).
- Support plumbing with no production caller: `Utils/widget_helpers.py::
  alert_mindmap_not_available`, the `optional_deps.py` "mindmap" feature key
  (anytree).
- ChaChaNotes `mindmaps`/`mindmap_nodes` tables are write-only:
  `create_mindmap`/`add_mindmap_node` exist, zero read methods (16845's
  per-button evidence) — nothing written could ever be displayed.
- Tests keeping the corpse exercised: `Tests/test_jsoncanvas_handler.py`,
  `Tests/UI/test_mindmap_viewer_tooltips.py`.

(The Study screen's dead mindmap-pane buttons are TASK-19041's census; this
task is the subsystem behind them.)

CLAUDE.md residual: a fresh diff of the Data Layer DB list against
`ls tldw_chatbook/DB/` shows exactly one discrepancy — `VisualIdentity_DB.py`
(live since the v39 visual-identity schema) is absent from the list. Every
other listed module exists; no other live DB module is unlisted.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The mindmap subsystem is either reachable end-to-end (viewer mounted from a live screen with a real DB read path) or retired — code, guarded exports, optional-dep/helper plumbing, and its tests handled together with git-log provenance recorded; per the owner ruling, prefer durable retirement over speculative wiring
- [ ] #2 The write-only `mindmaps`/`mindmap_nodes` DB surface is resolved intentionally: gains a read path if wired, or its write-only accessors are retired (any table drop only via a proper schema migration)
- [ ] #3 CLAUDE.md's Data Layer DB list matches `ls tldw_chatbook/DB/` — `VisualIdentity_DB.py` listed, no other discrepancy
- [ ] #4 Targeted suites green; grep shows no remaining production references to retired names
<!-- AC:END -->

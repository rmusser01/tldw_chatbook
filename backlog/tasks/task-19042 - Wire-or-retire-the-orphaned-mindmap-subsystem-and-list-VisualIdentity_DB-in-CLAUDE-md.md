---
id: TASK-19042
title: >-
  Wire-or-retire the orphaned mindmap subsystem (Tools/Mind_Map +
  MindmapViewer + write-only tables); list VisualIdentity_DB in CLAUDE.md
status: Done
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
- [x] #1 The mindmap subsystem is either reachable end-to-end (viewer mounted from a live screen with a real DB read path) or retired — code, guarded exports, optional-dep/helper plumbing, and its tests handled together with git-log provenance recorded; per the owner ruling, prefer durable retirement over speculative wiring
- [x] #2 The write-only `mindmaps`/`mindmap_nodes` DB surface is resolved intentionally: gains a read path if wired, or its write-only accessors are retired (any table drop only via a proper schema migration)
- [x] #3 CLAUDE.md's Data Layer DB list matches `ls tldw_chatbook/DB/` — `VisualIdentity_DB.py` listed, no other discrepancy
- [x] #4 Targeted suites green; grep shows no remaining production references to retired names
<!-- AC:END -->

## Implementation Plan

1. Re-verify every orphan claim with fresh whole-tree greps at the branch base
   (dev `25500ad87`): importers of `Tools/Mind_Map`, constructors of
   `MindmapViewer`, callers of `alert_mindmap_not_available` /
   `check_mindmap_available`, consumers of the "mindmap"/"anytree"
   optional-dep keys, and readers of `create_mindmap`/`add_mindmap_node`.
2. Walk the dead-code graph from both ends (lessons-testing-evidence):
   outward (who imports the corpse) and downward (what the corpse uniquely
   imports — `anytree` joins the removal set; `SmartContentTree` and
   `chatbook_models.ContentType` have live consumers and stay).
3. Retire, per the owner's stability ruling: delete `Tools/Mind_Map/` (7
   modules), `UI/Widgets/MindmapViewer.py`, the guarded export in
   `UI/Widgets/__init__.py`, `alert_mindmap_not_available`, the optional-dep
   "mindmap" feature (registry entry, availability keys,
   `check_mindmap_available`, `AREA_VISUALIZATION` — sole user), the
   `mindmap = []` pyproject extra (the extras census test forces
   pyproject and `OPTIONAL_FEATURES` to move together), `anytree` from base
   deps/requirements, the write-only `create_mindmap`/`add_mindmap_node`
   accessors, and the two corpse tests.
4. Leave the `mindmaps`/`mindmap_nodes` TABLES (and their FTS mirror,
   triggers, index, and `sql_validation.py` entries) untouched — no schema
   version bump; tables stay dormant pending a future migration.
5. Update `Tests/ChaChaNotesDB/test_study_functionality.py` honestly: drop
   the accessor tests + `sample_mindmap` fixture; keep the table-census
   assertions (tables still exist).
6. Do NOT touch `UI/Study_Window.py` or Study tests (task-19041 owns them);
   verify by grep that nothing there imports a retired symbol.
7. CLAUDE.md: re-diff the Data Layer DB list against `ls tldw_chatbook/DB/`
   and fix every discrepancy found.
8. Record git-log provenance per piece in Implementation Notes; run targeted
   suites + a repo-wide `--collect-only -q` sweep; ruff on touched files.

## Implementation Notes

**Decision: RETIRED.** Re-verified every orphan claim with fresh whole-tree
greps at branch base dev `25500ad87` before deleting; all held. The
dead-code graph was measured from both ends: outward (nothing in production
imports `Tools/Mind_Map` except `MindmapViewer.py`, which is never
constructed) and downward (the corpse uniquely imported `anytree`, which
joined the removal set; `SmartContentTree` and
`Chatbooks.chatbook_models.ContentType` have live consumers and stay).

**Deleted (10 files, ~3,700 LOC):**
- `tldw_chatbook/Tools/Mind_Map/` — 7 modules, 2,803 LOC
- `tldw_chatbook/UI/Widgets/MindmapViewer.py` (583 LOC) and its guarded
  export / `MINDMAP_AVAILABLE` flag in `UI/Widgets/__init__.py`
- `Utils/widget_helpers.py::alert_mindmap_not_available` (zero callers)
- `Utils/optional_deps.py`: the "mindmap" `OPTIONAL_FEATURES` entry,
  `check_mindmap_available()` + its call in the check-all pass, the
  "mindmap"/"anytree" `DEPENDENCIES_AVAILABLE` keys, and
  `AREA_VISUALIZATION` (its sole user was the mindmap feature)
- `pyproject.toml`: the empty `mindmap = []` extra (the extras census test
  `test_optional_feature_metadata_covers_pyproject_extras` forces
  `OPTIONAL_FEATURES` and pyproject extras to move together) and `anytree`
  from base dependencies; `anytree` also dropped from `requirements.txt`
  and `requirements-test.txt` — the deleted code was its only importer
- `DB/ChaChaNotes_DB.py::create_mindmap` / `add_mindmap_node`
- Corpse tests: `Tests/test_jsoncanvas_handler.py`,
  `Tests/UI/test_mindmap_viewer_tooltips.py`, and the `TestMindmaps` class
  + `sample_mindmap` fixture in
  `Tests/ChaChaNotesDB/test_study_functionality.py`

**Accessor/table boundary (AC#2):** only the write-only accessors were
retired. The `mindmaps`/`mindmap_nodes` tables, `mindmap_nodes_fts`, their
triggers, the `idx_mindmap_nodes_mindmap_id` index, and the
`sql_validation.py` table entries remain untouched — no schema-version
bump (deliberate: schema-pinning tasks 19044/19045 are in flight). The
empty tables stay dormant pending a future migration; a NOTE comment at
the removal site in `ChaChaNotes_DB.py` says so.
`TestSchemaMigration`'s table/FTS census still asserts they exist.

**Git-log provenance (when each piece was orphaned):**
- `create_mindmap`/`add_mindmap_node` added 2025-07-30 (`19115f71a`) with
  no read counterpart — write-only from birth (16845's per-button evidence
  re-confirmed).
- `Tools/Mind_Map/`, `MindmapViewer.py`, `check_mindmap_available`, and
  their sole production consumer `UI/Mindmap_Viewer_Window.py` added
  2025-08-01 (`f7f1a8438`, routing touched in `8aba59766`);
  `alert_mindmap_not_available` added 2025-08-04 (`a780960a5`).
- 2026-07-25 `1b7be2213` (task-671) deleted the dead
  `Mindmap_Viewer_Window.py` — the only place that ever lazy-imported
  `MindmapViewer`/`MindmapIntegration` and called the availability check
  and alert helper. That commit is the orphaning event for the whole
  subsystem; nothing has referenced it in production since.
- Corpse tests: `test_jsoncanvas_handler.py` added 2025-08-06
  (`8653baa78`); `test_mindmap_viewer_tooltips.py` added 2026-05-02
  (`de8e58f3d`, #192).

**Deliberately untouched:** `UI/Study_Window.py` and Study tests
(task-19041 owns them; verified by grep they import no retired symbol —
`Study_Window.py:437` and `Tests/UI/test_study_screen.py:768` mention the
retired accessor names in comments/docstrings only).
`tldw_api/storage_schemas.py`'s "mindmap" literals stay — they mirror the
remote tldw server's generated-file storage enums (wire contract), not
this subsystem. Historical records (CHANGELOG, `Docs/Design/*` audits,
`Docs/Development/Mind-Map-Viewer-1.md` design exploration) stay. No
`Docs/User_Guide` page references the viewer (grep-verified), so no
user-guide update was needed.

**CLAUDE.md (AC#3):** re-diffed the Data Layer list against
`ls tldw_chatbook/DB/` — exactly one discrepancy, `VisualIdentity_DB.py`
unlisted; added it. Every other listed module exists; no other live
`*_DB*.py` module is unlisted.

**Verification:** targeted suites — `test_optional_deps.py` +
`test_study_functionality.py`: 91 passed, 1 failed;
`test_optional_import_deferral.py` + `test_subscriptions_dependency_gate.py`
+ `test_ingest_capabilities.py` + `test_css_class_coverage_contract.py`:
134 passed, 1 failed; `Tests/UI/test_study_screen.py`: 23 passed. Both
failures are PRE-EXISTING dev reds, proven by re-running each single test
in a throwaway worktree at origin/dev `25500ad87` with identical failure
sets: (1) `test_optional_feature_metadata_covers_pyproject_extras` — dev's
pyproject declares a `frontmatter` extra with no `OPTIONAL_FEATURES`
metadata entry; (2) `test_every_composed_class_is_styled_or_registered` —
six unstyled `console-*` tokens. Neither involves mindmap; routed to the
controller for filing. Repo-wide `pytest Tests/ --collect-only -q`: 51,450
tests collected, 0 collection errors, exit 0. `ruff check` + `format
--check` clean on all touched Python files. Final grep for
`MindmapViewer|Mind_Map|alert_mindmap_not_available|check_mindmap_available|
AREA_VISUALIZATION|create_mindmap|add_mindmap_node|anytree|MINDMAP_AVAILABLE`
across code/config: zero production references (only the boundary-owned
Study comment/docstring and this task's own removal notes).

**Fix round (controller-verified finding):** the deletions left
`Docs/security/production-diagnostic-inventory.json` stale — and contrary
to the review's note B, that JSON has a live test consumer
(`Tests/Architecture/test_persistent_diagnostic_inventory.py` runs
`scripts/check_persistent_diagnostic_inventory.py`, which rebuilds the
inventory from live code and byte-compares). Applied the established
surgical hand-edit playbook (16196/16835 precedent; 16846/19046 used it
too) rather than `--write`, which would have absorbed dev's pre-existing
unrelated drift: removed the 7 committed rows for deleted files (six
`Tools/Mind_Map/*.py` rows, all TASK-492, call_counts 4+1+1+2+9+2 = 19;
`UI/Widgets/MindmapViewer.py`, TASK-494, 2), updated the
`Utils/optional_deps.py` row to the rebuild's own values (call_count
66→64, digest `200520a8c1cc673d5653`→`0f7e2c6195b8b3b6c373`, recomputed
via the script's `_scan_file`/`diagnostic_digest` on live code — two
diagnostic calls died with `check_mindmap_available`), and adjusted the
summary: `owner_files` 503→496, `task_492_calls` 1228→1209 (−19),
`task_494_calls` 6991→6987 (−2 row, −2 count change). Sink topology
untouched (no deleted file had sinks). Invariants verified by a printing
probe reading the edited file: len(owners)=496=summary.owner_files;
sum(492)=1209, sum(494)=6987, len(topology)=6 — all matching their
summary fields; the committed file was confirmed to be in the script's
canonical encoding before editing, and the edit re-encoded identically.
Acceptance bar (19046 standard) met: the post-edit rebuild comparison
shows the branch's residual drift is EXACTLY dev's pre-existing drift —
missing `library_media_browse_controller.py` row (+2),
`Client_Media_DB_v2.py` 354→338, `library_screen.py` 110→109 (residual
494 delta −15 closes: −16−1+2), TASK-492 bucket now byte-exact
(1209==1209), every other top-level key SAME. The pin test stays red from
that pre-existing dev drift only; not this task's to fix.

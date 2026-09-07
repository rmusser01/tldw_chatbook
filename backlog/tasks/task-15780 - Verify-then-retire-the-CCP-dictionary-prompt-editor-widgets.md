---
id: TASK-15780
title: Verify-then-retire the CCP dictionary/prompt editor widgets
status: Done
assignee:
  - '@claude'
created_date: '2026-08-13 12:31'
labels:
  - cleanup
priority: low
---

## Description

Verify-then-retire candidate surfaced while reviewing task-15476 (input-latency
burn-down's picker-debounce task, which touched
`ccp_dictionary_editor_widget.py:730/:848` and `ccp_prompt_editor_widget.py:876`
for consistency without checking whether the widgets are actually reachable
in production). Confirmed by a repo-wide grep: `CCPDictionaryEditorWidget`
and `CCPPromptEditorWidget` are referenced only inside their own module and
`Widgets/CCP_Widgets/__init__.py`'s re-export — zero other production
importers anywhere in `tldw_chatbook/`. The package's own `__init__.py`
docstring says as much: "Surviving prompt/dictionary editor widgets. The
legacy CCP screen chrome (sidebar, character card/editor, conversation view,
persona card/editor) was retired in favor of the Personas workbench
(tldw_chatbook/Widgets/Persona_Widgets/)."

This is exactly the same shape as task-15481's dead-scheduler/dead-DB-module
sweep in the same programme: code that looks alive (and gets reflexively
touched by unrelated fixes, as task-15476 did) but is unreachable from any
production screen. Per the same standing preference task-15481 applied:
delete (with git-log provenance) or explicitly quarantine — do not leave a
loaded gun a future contributor might wire up without noticing it was
already retired.

## Acceptance Criteria

- [x] Re-verify at implementation time (not trusting this task's grep without
      re-checking) that `CCPDictionaryEditorWidget` and
      `CCPPromptEditorWidget` have zero production callers/importers outside
      their own module and `__init__.py`
- [x] If still dead: delete both widget modules (with git-log provenance
      recorded in the notes) and their now-orphaned test-only importers, or
      trim tests that also cover live code the same way task-15481 did for
      `Research_DB`/`Sync_Client`
- [x] If a live caller is found (contradicting the grep above): the task
      closes as "not dead," with the caller documented, and no deletion
      happens (N/A — no live caller was found; both widgets confirmed dead,
      see Implementation Notes)
- [x] `pytest --collect-only` over the whole tree has zero errors after the
      change; a final grep sweep for both class names returns no production
      hits

## Implementation Plan

1. Re-verify reachability at HEAD: repo-wide grep for `CCPDictionaryEditorWidget`/
   `CCPPromptEditorWidget`/`ccp_dictionary_editor_widget`/`ccp_prompt_editor_widget`
   across `.py`, tests, docs, config/packaging files; confirm no dynamic/
   reflective importer (`pkgutil`/`importlib` scans over `Widgets`); confirm the
   `screen_registry.py` `"ccp"` route aliases to `PersonasScreen`, not a
   composing CCP screen, so the historical composer (`ccp_screen.py`) is
   confirmed gone (git-log provenance).
2. Baseline `pytest --collect-only` over the whole tree (record count) and run
   `Tests/Architecture/test_persistent_diagnostic_inventory.py` to record its
   pre-existing pass/fail baseline (expected: pre-existing drift-related
   failures per task-16196/task-15743 lineage, unrelated to this change).
3. If confirmed dead: delete the two widget modules and the now-empty
   `Widgets/CCP_Widgets/` package (its `__init__.py` only re-exported these
   two), record git-log provenance in notes.
4. Hand-edit `Docs/security/production-diagnostic-inventory.json` per the
   task-16196 precedent: remove the two owner rows for the deleted files and
   decrement `summary.owner_files`/`summary.task_494_calls` by exactly their
   contribution, leaving all other (already-drifted) entries untouched.
5. Re-run `pytest --collect-only` over the whole tree (expect identical count
   to baseline, since no test files exercised these widgets) and re-run the
   diagnostic-inventory suite (expect identical pre-existing pass/fail split).
6. `ruff check`/`format` on any touched Python files; hand-edit this task file
   with ACs/notes/Done; commit locally (no push/PR/merge).

## Implementation Notes

Both widgets confirmed dead; deleted the whole `Widgets/CCP_Widgets/` package.

**Per-widget reachability (import-graph table):**

| Symbol / module | Importers outside its own module | Verdict |
|---|---|---|
| `CCPDictionaryEditorWidget` (`ccp_dictionary_editor_widget.py`) | none in `tldw_chatbook/` or `Tests/`; only `Widgets/CCP_Widgets/__init__.py`'s re-export | dead |
| `CCPPromptEditorWidget` (`ccp_prompt_editor_widget.py`) | none in `tldw_chatbook/` or `Tests/`; only `Widgets/CCP_Widgets/__init__.py`'s re-export | dead |
| `Widgets/CCP_Widgets` package itself | zero importers of the package outside itself (only 3 stray plain-text comments in `Persona_Widgets/personas_pane_messages.py` referencing the *already-deleted* `ccp_character_card_widget`/`ccp_character_editor_widget`, unrelated to these two files) | dead |

Verification performed (independent of task-15771's reactive-default touch,
which was a uniformity sweep, not liveness evidence):
- Repo-wide grep (`.py`, `Tests/`, docs, `pyproject.toml`/`MANIFEST.in`) for
  both class names and both module basenames: zero production hits, zero
  test-file hits (the historical test suite that once exercised these
  widgets, `Tests/Widgets/test_ccp_widgets.py`, was already deleted in commit
  `9594931c8` "refactor: retire legacy CCP screen, sidebar chrome, and
  orphaned widgets", 2026-06-11 — the same commit that deleted the composing
  `ccp_screen.py` (1808 lines) and its `.bak`, while explicitly *keeping*
  these two "surviving" widgets unreferenced).
- `tldw_chatbook/UI/Navigation/screen_registry.py`'s `"ccp"` route already
  aliases to `PersonasScreen` (`"ccp": ScreenRoute("ccp", "personas",
  "...personas_screen", "PersonasScreen")`), confirming no live screen
  composes the legacy CCP chrome — the composer these widgets needed is
  gone, not merely unreachable-by-nav.
- No dynamic/reflective importer: `grep -rn "pkgutil|iter_modules|
  walk_packages" tldw_chatbook/` has zero hits touching `Widgets`.
- Two independent prior confirmations found in-repo, both predating this
  task: `Docs/superpowers/specs/2026-07-13-roleplay-p1a-dictionaries-
  foundation-design.md` calls `ccp_dictionary_editor_widget.py` "broken/dead
  code that call[s] functions which do not exist" (P1a built the real
  Chat-Dictionaries UI as its replacement); `Docs/superpowers/qa/
  library-prompts-2026-07/README.md` independently calls
  `CCPPromptEditorWidget` a "dead pocket."

**Deleted:** `tldw_chatbook/Widgets/CCP_Widgets/` (all 3 files: `__init__.py`,
`ccp_dictionary_editor_widget.py` (947 lines), `ccp_prompt_editor_widget.py`
(900 lines)). No test files existed that exercised these widgets (they were
removed in `9594931c8`, so there were no orphaned test-only importers to
trim in this pass), and no other production code referenced the package.

**Preserved:** nothing needed preserving — both widgets and the whole
package were confirmed fully dead, not partially live.

**Diagnostic-inventory JSON handling** (following the task-16196 precedent
exactly, since regenerating the whole file via
`scripts/check_persistent_diagnostic_inventory.py --write` is known to pull
in large amounts of pre-existing unrelated drift — see task-1822/2768/3035/
3750/14651/15103/15600/15743): hand-removed the two
`Docs/security/production-diagnostic-inventory.json` owner rows for
`ccp_dictionary_editor_widget.py` (`call_count: 6`) and
`ccp_prompt_editor_widget.py` (`call_count: 10`), and decremented
`summary.owner_files` 490→488 and `summary.task_494_calls` 6925→6909 by
exactly their contribution (6+10=16). Verified post-edit: `len(owners) ==
488 == summary.owner_files`; `sum(TASK-494 call_count) == 6909 ==
summary.task_494_calls`; `sum(TASK-492 call_count) == 1185 ==
summary.task_492_calls` (unchanged, as expected). No other owner rows
touched.

Left `Docs/Development/ccp-refactoring-complete.md` untouched — it already
fully documented the deleted `ccp_screen.py` architecture (`CCPScreen`,
`CCPSidebarWidget`, `CCPCharacterCardWidget`, etc., all gone since
`9594931c8`) before this change; it is a stand-alone archival doc, not
linked from any docs index, and was equally stale either way. Out of scope
for a widget-deletion task. Other doc/backlog hits for the two class names
(`Docs/Design/2026-08-11-input-latency-audit.md`, the roleplay P1a spec, the
library-prompts QA README, closed task-1160/task-15476) are all historical
records describing past states and were left untouched per programme
convention (never rewrite closed task files or dated design docs).

**Tests:**
- `pytest --collect-only` over the whole tree: 48472 collected, 0 errors,
  identical before and after (no test files were lost — none existed for
  these widgets).
- `Tests/Architecture/test_persistent_diagnostic_inventory.py`: 2 failed, 63
  passed, both before and after — identical split.
  `test_production_diagnostic_inventory_and_sink_topology_are_unchanged` and
  `test_task_15743_final_rebase_diagnostics_are_metadata_only` were already
  red pre-change due to the same pre-existing, separately-tracked
  reconciliation drift documented in task-16196's notes (confirmed by
  running the suite against the unmodified base before touching anything);
  this change does not introduce, fix, or worsen that drift.
- `python -c "import tldw_chatbook.Widgets; import tldw_chatbook.app"`
  succeeds post-deletion (sanity import check).
- No Python files were added or modified (only deleted + one JSON edit), so
  there was nothing new for `ruff check`/`format` to lint.

**Files changed:**
- Deleted `tldw_chatbook/Widgets/CCP_Widgets/__init__.py`,
  `ccp_dictionary_editor_widget.py`, `ccp_prompt_editor_widget.py`
- `Docs/security/production-diagnostic-inventory.json` — removed the two
  dangling owner rows + summary counts (arithmetic verified)

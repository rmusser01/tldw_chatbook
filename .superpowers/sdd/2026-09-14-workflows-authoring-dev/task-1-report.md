# Task 1 implementation report

Status: DONE_WITH_CONCERNS for implementation handoff. Targeted tests are green;
Backlog remains In Progress pending coordinator code review and full DoD. Existing
whole-file static debt is not waived. Later final evidence supersedes earlier
checkpoint results recorded chronologically below.

## Scope and source

- Worktree: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/workflows-authoring-dev`
- Branch: `codex/workflows-authoring-dev`
- Reviewed source: `b34eda3d64`. Document/draft/expression/catalog code and tests reused selectively. Only `prompt_definition` was copied from test helpers. Runtime dataclasses and screen launch/admission APIs were omitted.
- Existing ADR-138 authoring boundary, ADR-125 SQLite factory and ADR-150 design tokens apply. No new ADR. Coordinator owns ADR/spec/plan, compatibility note and visual captures.
- WorkflowsDB contains only constructor, authoring transactions and close. The four source SQL migration files remain byte-identical. No runtime lock, schema5, PID/recovery infrastructure, Notes/provider graph, model/network call, or server write is introduced.

## Verification evidence

All commands below run from the designated worktree with the installed sibling interpreter. Test runs are targeted and sequential.

### Initial missing-module RED

```text
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Workflows/test_document_service.py Tests/Workflows/test_draft_session.py Tests/DB/test_workflows_authoring_storage.py -q --timeout=60 --tb=short --show-capture=no -p no:randomly

ERROR Tests/Workflows/test_document_service.py
ERROR Tests/Workflows/test_draft_session.py
ERROR Tests/DB/test_workflows_authoring_storage.py
ModuleNotFoundError: No module named 'tldw_chatbook.DB.Workflows_DB'
3 errors in 0.17s (exit 2)
```

### Authoring storage GREEN

```text
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Workflows/test_document_service.py Tests/Workflows/test_draft_session.py Tests/Workflows/test_expressions.py Tests/Workflows/test_catalog.py Tests/DB/test_workflows_authoring_storage.py -q --timeout=60 --tb=short --show-capture=no -p no:randomly

193 passed, 1 warning in 8.95s (exit 0)
```

Includes exact unfinished-buffer restart, immutable migration hashes, real foreign-process BEGIN IMMEDIATE exclusion in WAL and DELETE modes through ordinary failed construction and sibling connection close.

### Lifecycle/editor missing-module RED

```text
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Workflows/test_authoring.py Tests/UI/test_workflows_editor.py -q --timeout=60 --tb=short --show-capture=no -p no:randomly

ModuleNotFoundError: No module named 'tldw_chatbook.Workflows.authoring'
ModuleNotFoundError: No module named 'tldw_chatbook.UI.Workflows_Modules'
1 warning, 2 errors in 0.77s (exit 2)
```

### Lifecycle/exchange service GREEN

```text
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Workflows/test_authoring.py -k 'not real_app' -q --timeout=60 --tb=short --show-capture=no -p no:randomly

6 passed, 1 deselected, 1 warning in 0.94s (exit 0)
```

Includes lazy unused close, cancelled setup drain, failed-close buffer retention/retry, opaque metadata and stable step ID roundtrip, oversized import refusal and cancelled export drain.

### Editor smoke GREEN

An initial port-edit syntax error (unmatched closing parenthesis left while removing the execution-follow menu entry) caused collection failure; corrected before the following run.

```text
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_workflows_editor.py -k 'new_add_step or real_css_panes or neighbors or typing_keeps' -q --timeout=60 --tb=short --show-capture=no -p no:randomly

8 passed, 40 deselected, 1 warning in 14.39s (exit 0)
```

Production CSS frames at 160x48, 110x36, 60x20; actual painted controls, F6/Tab traversal, stable typing, New/Add Step/Discard.

## Baseline concerns

- Existing RequestsDependencyWarning (urllib3/chardet/charset_normalizer combination) appears during import.
- Pytest post-run cleanup reports old Kokoro temporary-directory cleanup failures. No host cleanup or dependency change performed.
- Coordinator independently recorded immutable-base Ruff debt: 25 in DB/private_sqlite.py and 14 in Tests/DB/test_private_sqlite_inventory.py. Shared-system cleanup and suppressions are out of scope. Additional touched-file baseline comparisons will be recorded below.

## Remaining verification / handoff

- Full targeted editor/lifecycle/quit run, affected destination/Console/navigation assertions, private-owner census, tokens/bundle, Ruff/format/diff checks, self-review and explicit-path commit.
- Coordinator capture/review and final DoD remain outstanding. No task Done transition is authorized at this checkpoint.

## Additional implementation and verification

### Implemented files and ownership

- `Workflows/{models,document_service,draft_session,expressions,catalog}.py`: reviewed authoring value types, lossless document operations, structural validation, immutable revision conflicts, durable invalid drafts and cancellation-safe draft transitions. Unused runtime dataclasses were omitted.
- `DB/Workflows_DB.py` and four `DB/migrations/workflows_v*_to_v*.sql`: ordinary serialized SQLite constructor/transactions/close; byte-identical migration compatibility. Added `workflows.local` to the existing registry and C58 to the private-owner census, with no centralized backup registration or helper change.
- `Workflows/authoring.py`: lazy app-loop owner, off-thread path/database setup, retained creation/exchange, shared close operation, failed-flush retry, bounded UTF-8 JSON import and exact saved-revision atomic private export.
- `app.py` / `config.py`: 38 additive app-hook lines and one local path accessor. Document/draft injections survive screen replacement; the existing navigation flush hook and app quit guard drain the owner. No runtime coordinator import, server/provider/Notes graph, execution API or lock file.
- `UI/Screens/workflows_screen.py` and six reviewed `UI/Workflows_Modules` files plus `console_context.py`: real library/navigator/overview/continuous form, history and recovery, explicit import/export pickers, existing-file replacement confirmation, disabled Run. Secondary Console adapter reads stay off-thread, while its snapshot lookup stays on the app thread. Only its labels/buttons refresh.
- `css/features/_workflows.tcss`, `css/build_css.py`, generated bundle: token-backed states and reviewed pane geometry. Impeccable's established-product-extension/craft floor preserved the Operate world and keyboard/painted-control requirements, with no design tournament or shared token changes.
- New copied/authoring/SQLite tests and file-to-note fixture, only `prompt_definition` from source helpers. Existing destination tests now load production styles for Workflows and drain the wrapped lazy owner; assertions retain Console routing/off-thread behavior and verify the approved authoring layout.
- `Docs/User_Guide/workflows.md`: controls, draft/revision distinction, failure recovery, exchange size/privacy, disabled execution, storage and bounded server compatibility caveat. Existing unrelated Console briefing instructions preserved and explicitly separated.

### Larger editor/app RED and UI corrections

```text
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_workflows_editor.py Tests/Workflows/test_authoring.py Tests/UI/test_app_quit_guard.py -q --timeout=60 --tb=short --show-capture=no -p no:randomly
3 failed, 68 passed, 1 warning in 61.45s (exit 1)
```

Failures: secondary Console rows starved raw repair at 60x20; shared `.form-textarea` overrode the restored six-row field; real-app harness used the wrong positional factory argument/issued navigation before startup. Console region was compacted; form sizing received proper specificity. The app factory argument is `configured_default`, now `"home"`.

```text
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_workflows_editor.py -k validate_uses -q --timeout=60 --tb=short --show-capture=no -p no:randomly
1 failed, 48 deselected, 1 warning in 1.81s (exit 1)
AttributeError: WorkflowsScreen has no attribute '_admission'

PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_workflows_editor.py Tests/Workflows/test_authoring.py -k 'validate_uses or reopened_invalid or reopen_ui_discovers or expanded_editor_escape or real_app' -q --timeout=60 --tb=short --show-capture=no -p no:randomly
1 failed, 3 passed, 52 deselected, 1 warning in 9.65s (exit 1)
```

The three UI cases passed after fixes. Remaining failure was the real-app initial-screen wait. The leftover admission reference was removed; Validate is local-only and cannot enable execution.

### Real app harness diagnosis (no production startup changes)

```text
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Workflows/test_authoring.py -k real_app -q --timeout=60 --tb=short -p no:randomly
1 failed, 6 deselected, 1 warning in 6.00s (exit 1)
AssertionError: app never finished pushing its initial screen
Captured log: Creating splash screen - duration: 7.0, card: random

PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Workflows/test_authoring.py -k real_app -q --timeout=60 --tb=short --show-capture=no -p no:randomly
1 failed, 6 deselected, 1 warning in 60.20s (exit 1)
Failed: Timeout (>60.0s) from pytest-timeout

PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Workflows/test_authoring.py -k real_app -q --timeout=20 --tb=short -p no:randomly
workflow test: mounted
workflow test: initial screen
workflow test: navigated
1 failed, 6 deselected, 1 warning in 20.19s (exit 1)
```

The initial helper waited ~3 seconds for a 7-second splash. Persisted splash-off only in isolated test config. Temporary phase diagnostics proved navigation completed and waiting for every app worker hung on unrelated long-lived work. Diagnostics removed; tests await only `workflows-*` workers. Model-catalog refresh is explicitly disabled in these full-app tests, matching existing UI fixture policy.

```text
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Workflows/test_authoring.py -k real_app -q --timeout=60 --tb=short --show-capture=no -p no:randomly
1 passed, 6 deselected, 1 warning in 5.49s (exit 0)
```

### Retained close failure RED/GREEN

```text
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Workflows/test_authoring.py -k close_waiters -q --timeout=60 --tb=short --show-capture=no -p no:randomly
2 failed, 7 deselected, 1 warning in 0.86s (exit 1)
Concurrent close waiters did not both receive DraftWriteFailed; a cancelled waiter left a failed close latched, preventing the next close from retrying.

PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Workflows/test_authoring.py -k 'not real_app' -q --timeout=60 --tb=short --show-capture=no -p no:randomly
8 passed, 1 deselected, 1 warning in 1.07s (exit 0)
```

Settlement/reset now belongs to the retained close task's callback, not individual waiters. Exact buffered bytes remain retryable after failure/cancellation.

### Real app failure and file-picker exchange RED/GREEN

```text
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Workflows/test_authoring.py -k real_app -q --timeout=60 --tb=short --show-capture=no -p no:randomly
1 failed, 3 passed, 8 deselected, 1 warning in 16.21s (exit 1)
Expected replacement confirmation after selecting an existing export target; the enhanced picker returned it without confirmation.

PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Workflows/test_authoring.py -k real_app -q --timeout=60 --tb=short --show-capture=no -p no:randomly
5 passed, 8 deselected, 1 warning in 20.52s (exit 0)
```

Added a Workflows-owned replacement confirmation after file selection. Both cancellation and replacement are tested with actual EnhancedFileOpen/EnhancedFileSave controls, real SQLite, saved-revision identity, opaque integer/envelope metadata and stable step ID. The other cases test actual app navigation and quit veto/retry, exact buffer restart, lazy first entry, and no informational success notification on write refusal.

### Existing Workflows destination/Console assertions

The identical command was run three times while adapting obsolete shell assertions and preserving functional contracts:

```text
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_destination_shells.py Tests/UI/test_destination_visual_parity_correction.py Tests/UI/test_console_live_work_handoffs.py -k workflows -q --timeout=60 --tb=short --show-capture=no -p no:randomly
18 failed, 1 passed, 276 deselected, 1 warning in 36.52s (exit 1)
3 failed, 16 passed, 276 deselected, 1 warning in 17.81s (exit 1)
19 passed, 276 deselected, 1 warning in 19.69s (exit 0)
```

The first run also caught a real Console regression: app-thread-only ScreenStateStore was read inside the background adapter worker, so follow items were absent. Restored the prior main-thread snapshot / off-thread adapter boundary. Remaining geometry failures came from the old harness omitting the app bundle after DEFAULT_CSS moved to token-backed source CSS; Workflows harness now loads the production styles. No other destinations were repaired.

### Comprehensive targeted GREEN before visual review fixes

```text
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Workflows Tests/DB/test_workflows_authoring_storage.py Tests/DB/test_private_sqlite_inventory.py Tests/UI/test_workflows_editor.py Tests/UI/test_app_quit_guard.py Tests/UI/test_design_token_governance.py Tests/UI/test_css_build_integrity.py Tests/UI/test_css_bundle_sync_guard.py -q --timeout=60 --tb=short --show-capture=no -p no:randomly
348 passed, 1 warning in 152.53s (exit 0)
```

Includes real SQLite foreign writers, cancellation/restart, 50 editor tests, independently expanded/restored sections, real-app file exchange/navigation/quit, private-owner census, app quit guards, token governance and CSS build/synchronization. The CSS integrity file includes two real-app stylesheet-loading checks, also passed; this was not a broad destination/UI sweep. Subsequent scoped visual-review evidence follows below.

## Visual review 1: two findings

Coordinator supplied `.superpowers/sdd/2026-09-14-workflows-authoring-dev/visual-review-1.md`, disposition **fix**. The receiving-code-review skill was used to verify both findings against actual status/focus code before implementation. Reviewer artifacts remain coordinator-owned.

```text
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_workflows_editor.py -k 'pinned_status or short_viewport_keeps' -q --timeout=60 --tb=short --show-capture=no -p no:randomly
4 failed, 50 deselected, 1 warning in 3.76s (exit 1)
Three widths rendered 'Saved locally' instead of distinguishing draft/revision; at 60x20 the painted frame omitted 'Prompt'.

PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_workflows_editor.py -k pinned_status -q --timeout=60 --tb=short --show-capture=no -p no:randomly
3 passed, 51 deselected, 1 warning in 3.98s (exit 0)

PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_workflows_editor.py -k 'short_viewport_keeps or pinned_status or real_css_panes or expanded_editor_escape or continuous_form_sections' -q --timeout=60 --tb=short --show-capture=no -p no:randomly
1 failed, 8 passed, 45 deselected, 1 warning in 14.24s (exit 1)
The new findings passed; the old Escape test hard-coded 6 rows at 60x20, now correctly 3. Its assertion was corrected to require restoration of the actual original height, preserving the behavior contract.
```

Status now derives from actual draft/base equality for durable states: `Saved revision · draft unchanged` versus `Draft stored · not a saved revision`; pending changes remain explicitly unstored. Export still selects only a saved revision. Short-screen form TextAreas use the existing control-height token and label-inclusive scrolling, including focus re-entry. Expanded fields retain the reviewed 18-row behavior. Pane order, actions, independent collapse and disabled Run are unchanged.

## Static baseline (not waived)

Immutable base: `c5711892ab84f597f693a43c99ab93c8f17fabca`. Whole-file Ruff counts match unchanged baseline debt exactly:

| File | Base / current findings |
| --- | ---: |
| `tldw_chatbook/app.py` | 485 / 485 |
| `tldw_chatbook/config.py` | 164 / 164 |
| `tldw_chatbook/DB/private_sqlite.py` | 25 / 25 |
| `Tests/DB/test_private_sqlite_inventory.py` | 14 / 14 |
| `Tests/UI/test_destination_shells.py` | 11 / 11 |
| `Tests/UI/test_destination_visual_parity_correction.py` | 11 / 11 |
| `Tests/UI/test_console_live_work_handoffs.py` | 1 / 1 |
| `tldw_chatbook/css/build_css.py` | 2 / 2 |

Baseline checked with `git show c5711892ab:<path> | /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check --output-format=json --stdin-filename <path> -` for each listed file; counts extracted from JSON. Total **713 existing findings**, including the coordinator's 39 shared SQLite findings. No baseline cleanup or suppressions. Two new app catch-all diagnostics were resolved by catching known authoring/storage error types instead.

Baseline formatter checks used `git show c5711892ab:<path> | /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff format --check --stdin-filename <path> -`. Five files already failed at base: app, config, destination_shells, destination_visual_parity_correction, build_css. The two shared SQLite files and console_live_work_handoffs were formatted at base. New/rewritten Workflows files are separately checked clean; touched new hunks in existing files are formatted without sweeping unrelated baseline regions.

## Self-review / concerns / final handoff

- New module APIs have no runtime ownership aliases. Migration hashes, real foreign-writer exclusion and private census tests passed; no raw DB pinning or helper machinery was copied.
- App owner retains setup, accepted create/import/export and close tasks. DraftSession retains accepted revision/recovery writes. Failed flush does not clear/replace the buffer or report success. Normal navigation and ordinary quit use the existing guards.
- Console context refresh updates only its own stable children. The snapshot owner is queried on its thread; potentially slow adapter reads remain in a worker. Console follow preserves target ID/route and never enables Run.
- Exchange is explicit, bounded to 16 MiB UTF-8 JSON, selected saved revisions only, owner-only atomic export. Imported files are subject to existing private-file hardening. Opaque content may contain secrets; warning and guide do not equate preservation with safety or local save with server validation.
- No full suite, installs, live-profile reads, network/model calls, push/merge/stash/reset, parent checkout change or parked worktree change. No subagents/reviewers spawned by implementation. Coordinator artifacts excluded from staging.
- Existing dependency warning, old Kokoro pytest temp cleanup noise, and unwaived Ruff/format baseline remain concerns. Not a claim that whole-file static analysis is clean.
- Coordinator review/recapture is required before task Done. Backlog remains In Progress. Final verification, recapture disposition and commit will be appended below.

### Visual fix pair GREEN and recapture handoff

The Escape assertion's first edit accidentally placed the `original_height`
assignment in another template-field test. The identical covering command then
reported `NameError: original_height` (1 failed, 8 passed, 45 deselected in
13.72s). Ruff also identified the unused/missing variable; the assignment was
moved to the correct test before this GREEN run:

```text
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_workflows_editor.py -k 'short_viewport_keeps or pinned_status or real_css_panes or expanded_editor_escape or continuous_form_sections' -q --timeout=60 --tb=short --show-capture=no -p no:randomly
9 passed, 45 deselected, 1 warning in 14.10s (exit 0)
```

Coordinator notified that both material fixes were ready and no app-boot tests
were running, for one four-frame recapture batch. Added focus re-entry coverage
to the narrow label test; both the Label and TextArea are hit-tested against the
actual compositor, not just queried as mounted widgets.

```text
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/build_css.py
CSS build complete
Total size: 436,054 characters
Widget defaults: 111 classes, 110,547 + 95,761 characters
Screen CSS: 11 classes, 2,769 + 15,216 characters
(exit 0; all 54 modules processed)

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check tldw_chatbook/Workflows tldw_chatbook/UI/Workflows_Modules tldw_chatbook/UI/Screens/workflows_screen.py tldw_chatbook/DB/Workflows_DB.py Tests/Workflows Tests/UI/test_workflows_editor.py Tests/DB/test_workflows_authoring_storage.py
All checks passed! (exit 0)

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff format --check tldw_chatbook/Workflows tldw_chatbook/UI/Workflows_Modules tldw_chatbook/UI/Screens/workflows_screen.py tldw_chatbook/DB/Workflows_DB.py Tests/Workflows Tests/UI/test_workflows_editor.py Tests/DB/test_workflows_authoring_storage.py Tests/UI/test_console_live_work_handoffs.py
26 files already formatted (exit 0)

git diff --check
(no output; exit 0)
```

### Self-review: rejected setup must not trap quit

```text
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Workflows/test_authoring.py -k failed_open -q --timeout=60 --tb=short --show-capture=no -p no:randomly
1 failed, 13 deselected, 1 warning in 0.85s (exit 1)
sqlite3.DatabaseError: file is not a database (re-raised by the pre-quit flush after setup had already refused)

PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Workflows/test_authoring.py -k 'not real_app' -q --timeout=60 --tb=short --show-capture=no -p no:randomly
9 passed, 5 deselected, 1 warning in 1.25s (exit 0)
```

The real invalid-SQLite fixture remains byte-identical. Flush now permits exit
only if setup failed before obtaining a database/draft owner; it still raises
when an owner exists and there may be edits to protect. No UI/capture change.
Self-review also removed redundant app imports/catch alternatives:
`DraftWriteFailed` already subclasses `RuntimeError` in the reviewed value types.

## Final selection attempt and second visual review

```text
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Workflows Tests/DB/test_workflows_authoring_storage.py Tests/DB/test_private_sqlite_inventory.py Tests/UI/test_workflows_editor.py Tests/UI/test_app_quit_guard.py Tests/UI/test_design_token_governance.py Tests/UI/test_css_build_integrity.py Tests/UI/test_css_bundle_sync_guard.py -q --timeout=60 --tb=short --show-capture=no -p no:randomly
3 failed, 350 passed, 1 warning in 181.80s (exit 1)
```

These are editor interaction failures, **not** the unrelated destination baseline:

- `test_raw_edits_reconcile_values_schema_and_summaries_without_synthetic_edits`: the focused raw editor's hit test addressed `(107, 49)` outside the 160x48 frame after form reconciliation.
- `test_real_ui_typing_during_save_recovery_and_second_save`: the recovery option helper queried `#workflow-dialog-choices` before a modal was present.
- `test_historical_inspection_keeps_dirty_current_draft_and_edit_guard`: overlapping overview reconstruction raised `DuplicateIds` for `workflow-overview-0`.

All three are being diagnosed before commit; later evidence below supersedes
this failed attempt. Initial live commentary counted two while the suite was
running; the completed output definitively contains three.

The retained visual reviewer accepted draft/revision labeling, but found that
the Prompt value was absent even from the raw 60x20 Textual SVG. The first visual
test checked the painted label and control, but not the text inside it; that
was insufficient evidence. Added populated-value assertions to that test and
a real `TldwCli` test using the existing isolated factory, actual navigation,
Workflows DB/lifecycle and production styles. Only unrelated startup is mocked.

```text
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_workflows_editor.py -k 'short_viewport' -q --timeout=60 --tb=short --show-capture=no -p no:randomly
2 failed, 53 deselected, 1 warning in 5.31s (exit 1)
Real-app computed layout: region=Region(x=2, y=11, width=55, height=4),
content=Region(x=4, y=13, width=51, height=0),
padding=Spacing(top=1, right=1, bottom=1, left=1), solid borders,
scroll=Offset(x=0, y=0). Prompt was painted; {{ prepare.text }} was not.
```

The inherited `.form-textarea` vertical padding and border consumed every content
row. The feature-local compact rule now uses zero vertical padding and the same
horizontal inset token. No pane, action, collapse, Run, shared token or other
world change. The CSS source was rebuilt, never the generated bundle hand-edited.

```text
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/build_css.py
CSS build complete; all 54 modules processed
Total size: 436,100 characters
Widget defaults: 111 classes, 110,547 + 95,761 characters
Screen CSS: 11 classes, 2,769 + 15,216 characters (exit 0)

PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_workflows_editor.py -k 'short_viewport or pinned_status or real_css_panes or expanded_editor_escape or continuous_form_sections' -q --timeout=60 --tb=short --show-capture=no -p no:randomly
10 passed, 45 deselected, 1 warning in 17.11s (exit 0)
```

Coordinator notified of recapture readiness; app-boot tests paused while the
second/final planned visual fix batch is captured. The updated real-app test
requires label, populated value and focus together, nonzero content height and
actual compositor hit tests. The original narrow test also covers focus re-entry.

### Interaction failures: diagnosis, focused RED/GREEN

Full test names from the failed comprehensive selection:

1. `Tests/UI/test_workflows_editor.py::test_raw_edits_reconcile_values_schema_and_summaries_without_synthetic_edits`
2. `Tests/UI/test_workflows_editor.py::test_real_ui_typing_during_save_recovery_and_second_save`
3. `Tests/UI/test_workflows_editor.py::test_historical_inspection_keeps_dirty_current_draft_and_edit_guard`

Diagnosis and changes restricted to the failing behavior:

- **Raw focus visibility:** form reconciliation restores a stored absolute scroll
  offset after layout, but `focus=False` previously did nothing to keep an
  already-focused persistent raw TextArea on screen. It can land below the
  viewport as the sibling form relayouts. After restoring the scroll, retain the
  existing focus/selection and scroll that focused descendant into view. No focus
  transfer is introduced.
- **Recovery modal timing:** the test waited for the control worker after an
  earlier `pilot.pause()`, then queried the modal immediately. `_open_choice()`
  queues native `push_screen`; worker completion is not modal mount completion.
  The shared test helper now gives Textual a `pilot.pause()` before querying the
  choices. This case passed in the focused rerun even before that correction;
  the original symptom was intermittent, and there is no claim of a durable
  recovery-write failure. No production recovery semantics were changed.
- **Duplicate overview:** the screen's after-refresh callback and control worker
  can concurrently enter awaited remove/mount operations. A deterministic new
  `test_overlapping_overview_refresh_keeps_one_complete_form` reproduces the
  duplicate IDs by overlapping two refreshes. One native `asyncio.Lock` now
  serializes only Workflows widget reconciliation; no runtime/storage lock,
  ownership protocol or external helper was added. Final form contents and IDs
  are asserted, not merely lack of exceptions.

```text
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_workflows_editor.py -k 'raw_edits_reconcile or real_ui_typing_during_save or historical_inspection_keeps or overlapping_overview' -q --timeout=60 --tb=short --show-capture=no -p no:randomly
2 failed, 2 passed, 52 deselected, 1 warning in 6.73s (exit 1)
Historical inspection: DuplicateIds workflow-overview-2.
Deterministic overlap: DuplicateIds workflow-overview-0.
Raw focus and recovery menu cases passed this focused pre-fix run.

PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_workflows_editor.py -k 'raw_edits_reconcile or real_ui_typing_during_save or historical_inspection_keeps or overlapping_overview' -q --timeout=60 --tb=short --show-capture=no -p no:randomly
4 passed, 52 deselected, 1 warning in 7.37s (exit 0)

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check tldw_chatbook/UI/Screens/workflows_screen.py tldw_chatbook/UI/Workflows_Modules/editor.py Tests/UI/test_workflows_editor.py
All checks passed! (exit 0)

git diff --check
(no output; exit 0)
```

Coordinator reported round-3 capture exit 0 and all four raw-SVG label/value
assertions passing. Retained UI reviewer final disposition: **ship**, explicitly
limited to the two scored findings (draft/revision status and 60x20 Prompt label,
value, caret/focus). No further visual polish requested. This is not a substitute
for the pending task code review.

### Final static analysis evidence

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check tldw_chatbook/Workflows tldw_chatbook/UI/Workflows_Modules tldw_chatbook/UI/Screens/workflows_screen.py tldw_chatbook/DB/Workflows_DB.py Tests/Workflows Tests/UI/test_workflows_editor.py Tests/DB/test_workflows_authoring_storage.py
All checks passed! (exit 0)

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff format --check tldw_chatbook/Workflows tldw_chatbook/UI/Workflows_Modules tldw_chatbook/UI/Screens/workflows_screen.py tldw_chatbook/DB/Workflows_DB.py Tests/Workflows Tests/UI/test_workflows_editor.py Tests/DB/test_workflows_authoring_storage.py Tests/UI/test_console_live_work_handoffs.py
26 files already formatted (exit 0)

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check tldw_chatbook/app.py tldw_chatbook/config.py tldw_chatbook/DB/private_sqlite.py Tests/DB/test_private_sqlite_inventory.py Tests/UI/test_destination_shells.py Tests/UI/test_destination_visual_parity_correction.py Tests/UI/test_console_live_work_handoffs.py tldw_chatbook/css/build_css.py --statistics
Found 713 errors. (exit 1; immutable-base counts tabulated above)
278 fixable with --fix; 22 additional hidden unsafe fixes were NOT applied.

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff format --check tldw_chatbook/app.py tldw_chatbook/config.py tldw_chatbook/DB/private_sqlite.py Tests/DB/test_private_sqlite_inventory.py Tests/UI/test_destination_shells.py Tests/UI/test_destination_visual_parity_correction.py Tests/UI/test_console_live_work_handoffs.py tldw_chatbook/css/build_css.py
5 files would be reformatted, 3 files already formatted (exit 1)
The same five immutable-base files fail, listed in the baseline section above.

git diff --check
(no output; exit 0)
```

The feature CSS rebuilt cleanly from source. No shared token changes, suppression
of baseline findings, broad formatter sweep, unrelated destination repair or
dependency installation was performed. A stale module docstring was corrected
to say the widgets have no execution authority. C58's explanatory paragraph stays
with its census row, outside the unrelated TASK-32160 descriptor section.

### Final comprehensive targeted GREEN

```text
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Workflows Tests/DB/test_workflows_authoring_storage.py Tests/DB/test_private_sqlite_inventory.py Tests/UI/test_workflows_editor.py Tests/UI/test_app_quit_guard.py Tests/UI/test_design_token_governance.py Tests/UI/test_css_build_integrity.py Tests/UI/test_css_bundle_sync_guard.py -q --timeout=60 --tb=short --show-capture=no -p no:randomly
355 passed, 1 warning in 187.51s (exit 0)
```

All three previously failing editor cases, deterministic overlap, actual-app
60x20 label/value/focus, storage foreign-writer/restart, cancellation and failed
flush, real navigation/quit/exchange, owner census, design tokens and CSS bundle
checks passed together. The only collected warning is the existing Requests
dependency warning; old Kokoro temporary-directory cleanup noise again followed
the successful pytest summary. No unrelated full UI/repository suite was run.

### Final affected destination/Console GREEN

```text
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_destination_shells.py Tests/UI/test_destination_visual_parity_correction.py Tests/UI/test_console_live_work_handoffs.py -k workflows -q --timeout=60 --tb=short --show-capture=no -p no:randomly
19 passed, 276 deselected, 1 warning in 28.00s (exit 0)
```

Combined final targeted result: **374 passed**, with 276 unrelated cases
deselected in the destination selection. App suites and coordinator captures
were sequential. Both final commands used the exact approved scope.

## Final self-review and handoff concerns

- Reviewed the additive app/config hooks, authoring lifecycle retention and
  failure paths, database factory/transaction boundary, immutable migration
  evidence, explicit file exchange, UI status/focus and final refresh race fix.
- Reviewed the 43 implementation-owned paths against the staging allowlist.
  Coordinator captures, QA scripts, ADR/spec/plan/compatibility and all other
  `.superpowers` files are excluded. This report is separately owned.
- No new failure remains in the final targeted selection. Whole-file static
  analysis remains non-green only at the separately measured immutable baseline:
  713 Ruff findings and five unformatted pre-existing files. No waiver or cleanup
  is implied. New/rewritten files pass Ruff and formatting; diff-check is clean.
- Existing Requests warning and old Kokoro post-test cleanup noise remain.
  No unrelated destination failures from the earlier broader baseline were
  addressed, and no broad sweep was run.
- The reviewer approved only the two material visual findings. Coordinator task
  code review and all DoD remain outstanding; the task is deliberately not Done.
- Source-only server compatibility evidence is coordinator-owned, not a live
  server claim. There is no publish/sync, centralized backup registration or
  workflow execution in this slice.
- Verification lesson from this task: mounted/hit-tested controls and visible
  labels do not prove their values paint; the real 60x20 TextArea had zero content
  rows because inherited padding consumed its height. The new real-app test
  checks the populated compositor output. No out-of-ownership lessons file was
  edited.

## Commits and final repository checks

Implementation commit:

`a1f47397eb1a7d62117707df914b63a8ad050998`

`feat(workflows): restore local authoring without execution`

43 explicit implementation-owned paths; 9,219 insertions and 497 deletions,
including the selectively reused reviewed source, tests and generated CSS.
The index's exact path list was compared against the 43-path allowlist before
commit. `git diff --cached --check` was clean. No Git escalation was needed.
The configured hooks directory contains sample hooks only; no model/reviewer
hook ran.

This report is committed separately under
`docs(workflows): record authoring verification and review handoff`, with no
production/test changes. Its commit hash is provided in the final handoff rather
than self-referenced inside its own content.

```text
git log -1 --format='%H%n%s'
a1f47397eb1a7d62117707df914b63a8ad050998
feat(workflows): restore local authoring without execution

git status --short
?? .impeccable/review/workflows-authoring-dev/
?? Docs/superpowers/qa/workflows-authoring-dev/
?? Docs/superpowers/specs/2026-09-14-workflows-authoring-compatibility.md
```

Those three untracked areas are coordinator-owned and were not staged or edited
by implementation. The report was ignored until explicitly staged by its exact
owned path. Parent and preserved worktree files were not modified. No push,
merge, stash or reset. No tests or capture process remain running at handoff.

## Task code-review fix round 1: database aliases and file exchange

Resumed from coordinator-only documentation commit
`b0ce08a093b49b7abb9652d94863e72c33a30111`; all its QA/parity/lesson files are
preserved. The review reported one Important finding and zero Critical findings.
The receiving-code-review and TDD skills were used; only the requested
authoring/storage/exchange scope was exercised, with no UI edits or recapture.

### Finding and confirmed cause

`WorkflowAuthoring._exchange_path` previously rejected only the lexical DB path.
The existing `open_private_binary` checks regular-file type, then calls `os.open`,
then rejects a multiply-linked inode via `fstat` and closes that raw descriptor.
That close can cancel SQLite's process-owned POSIX locks even though the import
failed. The shared helper was inspected read-only; its protocol was not changed.

`Tests/DB/test_workflows_authoring_storage.py::test_refused_alias_import_preserves_foreign_writer_exclusion`
holds a real ordinary draft transaction and probes writer exclusion through the
existing bounded stdlib-only subprocess helper. The cases use the actual main
DB, DELETE rollback journal, WAL, and WAL SHM files; no runtime aliases or new
helper subprocess design. In RED, the DELETE main-file and WAL SHM cases changed
from `blocked` to `acquired` after the refused import. Other cases remained
blocked, so the report does not claim every SQLite file carries writer locks.

`Tests/Workflows/test_authoring.py::test_exchange_refuses_database_alias_before_generic_file_io`
uses real SQLite DB/WAL/SHM/PERSIST-journal files and both import and export.
Hard links cover all four protected files; a trusted parent-directory alias of
a `.json`-named database covers equal device/inode with link count one. The
generic file boundary is observed while forwarding to the real implementation,
not replaced by a fabricated success/failure. Rejection by that boundary itself
is too late; all ten RED cases reached it.

### Exact RED evidence

```text
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/DB/test_workflows_authoring_storage.py -k refused_alias_import -q --timeout=60 --tb=short --show-capture=no -p no:randomly
2 failed, 3 passed, 6 deselected, 1 warning in 1.37s (exit 1)
FAILED test_refused_alias_import_preserves_foreign_writer_exclusion[DELETE-]
FAILED test_refused_alias_import_preserves_foreign_writer_exclusion[WAL--shm]
AssertionError: assert 'acquired' == 'blocked'

PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Workflows/test_authoring.py -k database_alias -q --timeout=60 --tb=short --show-capture=no -p no:randomly
10 failed, 14 deselected, 1 warning in 1.79s (exit 1)
All ten test_exchange_refuses_database_alias_before_generic_file_io cases:
AssertionError: Database aliases must be refused before generic file I/O
The boundary was called once instead of zero times.
```

### Minimal change and bounded limitation

- `authoring.py` now uses `lstat` before any generic file opening to reject
  non-regular/multiply-linked selected targets, and compares device/inode metadata
  with the active workflow DB and any existing `-journal`, `-wal`, `-shm` files.
  Missing selected targets remain valid for ordinary new-file export; other
  metadata errors propagate as refusal rather than bypassing checks.
- Export metadata inspection now runs off the UI thread, as import inspection
  already does. Ordinary generic private-file checks, hardening and atomic
  replacement remain unchanged after the feature preflight.
- **Limit:** this is a metadata-only preflight for aliases present at validation
  time, not a pathname lease or retained live-inode proof. It cannot prevent a
  concurrent actor from replacing/relinking a selected path after validation,
  or discover a detached live inode whose known database name was moved away.
  Existing generic post-open identity checks do not undo POSIX lock loss caused
  by opening/closing such a raced inode. Race-free cross-owner protection would
  require authority/lifetime coordination outside this approved tiny fix; no
  such protocol is introduced or claimed. Users should not move/relink selected
  paths or the live database during exchange. This limitation is also clarified
  in the guide.
- No shared SQLite/private-path changes, descriptor retention, new subprocess
  protocol, runtime lock, migration or UI change. Existing execution and ordinary
  transaction behavior are untouched. Coordinator artifacts remain untouched.

### Focused GREEN evidence

```text
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/DB/test_workflows_authoring_storage.py Tests/Workflows/test_authoring.py -k 'refused_alias_import or database_alias' -q --timeout=60 --tb=short --show-capture=no -p no:randomly
15 passed, 20 deselected, 1 warning in 1.91s (exit 0)

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check tldw_chatbook/Workflows/authoring.py Tests/Workflows/test_authoring.py Tests/DB/test_workflows_authoring_storage.py
All checks passed! (exit 0)
```

The previously failed foreign probe remains blocked until the transaction ends,
then acquires normally. Refused imports retain the selected draft and transaction
contents. Both exchange directions refuse all tested aliases before reaching
generic file I/O. The baseline 713 Ruff findings/five formatter files and existing
Requests/Kokoro noise remain recorded and unwaived; no cleanup was attempted.

### Existing exchange/lifecycle suite and off-loop regression

```text
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Workflows/test_authoring.py Tests/DB/test_workflows_authoring_storage.py -q --timeout=60 --tb=short --show-capture=no -p no:randomly
35 passed, 1 warning in 25.68s (exit 0)
```

The coordinator reiterated the binding requirement that export metadata checks
must stay off the app loop. The implementation already used the existing
`asyncio.to_thread(self._exchange_path, path)` seam. Added an assertion to the
existing ordinary exchange roundtrip that both import and export validation run
on a different thread while calling the real validator. Temporarily restored
only the old direct export call to verify that this regression genuinely fails:

```text
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Workflows/test_authoring.py -k exchange_keeps -q --timeout=60 --tb=short --show-capture=no -p no:randomly
1 failed, 23 deselected, 1 warning in 0.87s (exit 1)
AssertionError: Exchange metadata must not block the app loop
Failure at the export call; import metadata remained off-loop.
```

Restored the existing `to_thread` seam before final verification. No abstraction
or UI code was added.

### Final fix-round verification and self-review

```text
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Workflows/test_authoring.py Tests/DB/test_workflows_authoring_storage.py -q --timeout=60 --tb=short --show-capture=no -p no:randomly
35 passed, 1 warning in 22.80s (exit 0)

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check tldw_chatbook/Workflows/authoring.py Tests/Workflows/test_authoring.py Tests/DB/test_workflows_authoring_storage.py
All checks passed! (exit 0)

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff format --check tldw_chatbook/Workflows/authoring.py Tests/Workflows/test_authoring.py Tests/DB/test_workflows_authoring_storage.py
3 files already formatted (exit 0)

git diff --check
(no output; exit 0)
```

Self-review verified that the only production change is the feature-local
preflight plus the existing off-thread export seam. `stat`/`lstat` inspect
metadata without opening/closing database or sidecar descriptors. Both exchange
paths reach this guard; ordinary new/overwrite export and import still use the
existing private helpers after validation. Real writer exclusion, selected
draft preservation, roundtrip fields, failed writes, retained cancellation,
navigation/quit and actual exchange pickers pass the bounded suite. No 374-test
rerun, unrelated suite or visual capture was performed for this non-UI fix.

Fix commit subject: `fix(workflows): reject database aliases before file exchange`.
The full SHA is supplied in the final handoff rather than self-referenced in this
commit's report content. Six explicit owned paths: `Workflows/authoring.py`, its
authoring tests, the storage tests, this report, task notes and user-guide
clarification. Parent commit remains the coordinator's `b0ce08a093b49b7abb9652d94863e72c33a30111`;
no shared helpers or coordinator-owned documents are staged. Backlog remains
In Progress pending coordinator review of the fix and the existing unwaived DoD
concerns. No push/merge/stash/reset.

## Final static-gate fix — introduced import ordering only (2026-09-15)

Resumed from `3fd794c376b98f2dc3172ad9e18dfe24202ff5ea` on
`codex/workflows-authoring-dev`, in the designated authoring worktree only.
Read the current task, plan qualification section and ADR138 qualification,
plus the receiving-code-review and verification-before-completion skills.
The user-approved TASK-32601 gate now requires no new static debt, retaining
source-attributed baseline findings instead of requiring unrelated whole-file
cleanup. Coordinator attribution maps 711 diagnostics to immutable base
`77eb2601a63ba473318b8ec1e4edb53f8ac5899e`; the two unmatched findings were I001
in import blocks changed by this slice. That attribution is coordinator
evidence, not a new broad scan in this fix. Earlier 713/unwaived descriptions
above are historical; this qualification does not claim the full files are clean.

### Change and self-review

- Ruff's I001-only fixer sorted the imports in
  `Tests/UI/test_console_live_work_handoffs.py` and
  `Tests/UI/test_destination_visual_parity_correction.py`.
- Inspected the complete two-file diff: all changes are confined to imports;
  existing CSS comments remain. No test logic, new tests, suppression, broad
  formatter write, production code, runtime/storage/UI behavior or infrastructure
  changed. An ordering-only change did not require an invented test regression.
- This appendix is the only other owned edit. Coordinator plan/spec/ADR/task
  edits remain untouched and excluded from staging. Task status and final scoped
  gate disposition remain with the coordinator; this fix does not mark Done.

### Exact RED and GREEN static commands/results

All commands ran from
`/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/workflows-authoring-dev`.

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check --select I001 Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_destination_visual_parity_correction.py
I001 Import block is un-sorted or un-formatted
  Tests/UI/test_console_live_work_handoffs.py:3:1 (block through line 41)
I001 Import block is un-sorted or un-formatted
  Tests/UI/test_destination_visual_parity_correction.py:3:1 (block through line 70)
Found 2 errors.
[*] 2 fixable with the --fix option.
(exit 1; before the fix)

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check --select I001 --fix Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_destination_visual_parity_correction.py
Found 2 errors (2 fixed, 0 remaining).
(exit 0)

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check --select I001 Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_destination_visual_parity_correction.py
All checks passed!
(exit 0; also repeated before report/staging)

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_destination_visual_parity_correction.py
Found 10 errors.
[*] 2 fixable with the --fix option (2 hidden fixes can be enabled with the --unsafe-fixes option).
(exit 1)

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff format --check Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_destination_visual_parity_correction.py
unformatted: File would be reformatted
  Tests/UI/test_destination_visual_parity_correction.py:1467:23
1 file would be reformatted, 1 file already formatted
(exit 1; check only, no formatting applied)
```

The full Ruff result is zero findings in Console and ten retained baseline
findings in visual parity. Current locations: RUF012 at 85; RUF007 at 210 and
1210; S110 and BLE001 at 419; B009 at 715 and 740; ASYNC251 at 818, 823 and
828. The formatter leaves Console clean and reports two unrelated baseline
hunks in visual parity: wrapping the label tuple at 1467 and adding two blank
lines before the parametrization after current line 1516. Neither was changed.
Retaining these is the explicitly approved source-attributed qualification,
not a suppression or a claim that the full Ruff/formatter commands pass.

### Exact targeted interaction verification

```text
PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_destination_visual_parity_correction.py -k workflows -q --timeout=60 --tb=short --show-capture=no -p no:randomly
...............                                                          [100%]
15 passed, 164 deselected, 1 warning in 16.44s
(exit 0)

git diff --check -- Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_destination_visual_parity_correction.py
(no output; exit 0)
```

The test warning is the existing RequestsDependencyWarning for
urllib3/chardet/charset_normalizer versions. After the test summary, pytest also
reported two cleanup PytestWarnings for a pre-existing Kokoro garbage directory
(`test_kokoro_constructor_direct3` and its parent: `Errno 66 Directory not empty`).
No manual cleanup or unrelated tests were run. No capture, full suite, model or
network calls, dependencies or subagents were used.

Commit subject: `test(workflows): sort authoring handoff imports`. The full SHA
is supplied in the final handoff rather than self-referenced here. The commit is
limited to the two named test files and this appended report. Remaining concerns
are the explicitly retained baseline static debt and recorded test-environment
warnings, pending the coordinator's final qualification review.

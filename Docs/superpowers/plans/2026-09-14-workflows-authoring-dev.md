# Workflows Authoring-only Dev Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Execute the single authoring deliverable, then review. Steps use checkbox syntax.

**Goal:** Restore the approved usable Workflows editor on current dev without importing unfinished execution infrastructure.

**Architecture:** Reuse the reviewed editor-only UI and lossless document/draft services. A small authoring-only WorkflowsDB uses current dev's existing private SQLite connection factory; application lifecycle supplies and drains the document/draft services. Current Console follow remains a secondary existing behavior.

**Tech Stack:** Python >=3.12, Textual 8.x, SQLite, existing Pydantic and private-path utilities; no new dependencies.

**Spec:** Docs/superpowers/specs/2026-09-14-workflows-authoring-dev.md

ADR required: yes — amendment to existing ADR-138, not a new ADR number.
ADR path: backlog/decisions/138-portable-workflow-definitions-and-local-execution.md
Reason: record the user-approved stable-file exchange contract (2026-09-15); ADR-125 shared implementation and ADR-150 also govern unchanged.

## Global Constraints

- Work only in /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/workflows-authoring-dev on codex/workflows-authoring-dev.
- Base dev: 77eb2601a63ba473318b8ec1e4edb53f8ac5899e. UI source: b34eda3d64. Full source preserved: 8aa1987af9357655af9354610b878f247fd1e929.
- Sequential v1; branching v2; parallelism v3. New execution is unavailable in this slice.
- No new SQLite helper subsystem, runtime lock, schema v5, PID tracking, run/recovery engine, provider calls or server writes.
- Reuse current connect_private_sqlite; database setup/validation performs no parent raw-open/close of live DB or sidecars. Exchange retains metadata-only alias rejection under ADR-138's approved stable-file contract, not a race-free actual-open guarantee. Preserve existing four migrations byte-for-byte.
- Keep live database/sidecar names stable against external moves/replacements/relinks while open, and selected JSON files/containing paths stable during exchange. Normal SQLite-managed writes/sidecar lifecycle remain supported; no new file-I/O infrastructure.
- Preserve current dev navigation, Console follow, permissions and profile behavior. Do not copy old app.py, config.py, Console or shared SQLite files wholesale.
- Three panes at >=132 usable columns; navigator/editor at 96-131; editor and labeled selectors below 96. Verify actual 160x48, 110x36, 60x20 frames.
- Follow backlog/docs/design-language.md. Edit source CSS and rebuild; no direct generated CSS edits or new visual language.
- Targeted tests only; no dependency installs, live-profile access, network/model calls, full sweep, push or merge.
- Use /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python and /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff with PYTHONPATH=. from this worktree.
- Preserve and report baseline failures/debt; do not waive them or mark Done automatically.

### Task 1: Restore usable local workflow authoring

**Files:**

- Reuse from b34eda3d64: Workflows/{__init__,models,expressions,catalog,document_service,draft_session}.py; UI/Workflows_Modules/{__init__,controller,editor,library,navigator,reference_picker}.py; Tests/Workflows/{__init__,test_document_service,test_expressions,test_catalog,test_draft_session}.py; Tests/UI/test_workflows_editor.py; Tests/fixtures/workflows/file_to_note.json; only required prompt_definition fixture from Tests/Workflows/helpers.py.
- Create/adapt: tldw_chatbook/DB/Workflows_DB.py; the four existing DB/migrations/workflows_vN_to_vN+1.sql files; tldw_chatbook/Workflows/authoring.py for application-owned authoring setup/flush/close and explicit file exchange; Tests/Workflows/test_authoring.py; Tests/DB/test_workflows_authoring_storage.py.
- Adapt: tldw_chatbook/UI/Screens/workflows_screen.py using the reviewed source and current dev Console behavior. A same-package console_context.py may hold the existing Console-only display/refresh if needed to avoid conflating whole-screen recomposition with editor state.
- Additive hooks only: tldw_chatbook/app.py, config.py, and the existing navigation owner if required for actual draft flush/global F6. Read actual current hooks before choosing locations.
- Owner census: DB/private_sqlite.py; Tests/DB/test_private_sqlite_inventory.py; backlog/docs/sqlite-private-owner-inventory.md.
- CSS: source features/_workflows.tcss and build_css.py if a new module is required; generated tldw_cli_modular.tcss via build script.
- Affected old UI tests: test_destination_shells.py, test_destination_visual_parity_correction.py, test_console_live_work_handoffs.py, test_screen_navigation.py. Update only assertions made obsolete by the approved Workflows layout; retain equivalent navigation/Console contracts.
- Documentation: this task's notes and Docs/User_Guide/workflows.md. Coordinator owns ADR/spec/plan/parity notes and screenshot/review handoff files.

**Interfaces:**

- Consumes current private_sqlite.connect_private_sqlite(owner_id, path, **kwargs), existing path/picker services, BaseAppScreen and app quit/navigation hooks.
- Produces WorkflowsDB(path), transaction(write=True), close(), with no execution API.
- Reuses DocumentService(db) and DraftSession(documents, ...) signatures from the reviewed source; app.workflow_documents and app.workflow_drafts remain the UI injection points.
- Workflows/authoring.py provides a lazy app-owned authoring session: await open(), await flush(), await close(); concrete app hooks may use that one object rather than copy the old runtime coordinator. No setup until the user enters Workflows, and a failed flush must not erase its draft.

- [x] **Step 1: Prove the behavior is missing before porting.**
  Copy the reviewed tests first using apply_patch; their absent production imports establish the initial port RED. Add a behavior test for restart persistence and one for a foreign writer:
  ```python
  with first.transaction() as cursor:
      assert cursor.execute("SELECT 1").fetchone()[0] == 1
      # A foreign subprocess's BEGIN IMMEDIATE must still report SQLITE_BUSY
      # after another ordinary WorkflowsDB open/failed constructor.
  ```
  BEGIN IMMEDIATE in transaction() holds the real writer exclusion even for this read. Port the existing real subprocess probe mechanics from the parked regression file, but only constructor/ordinary-authoring paths; no runtime locks or extra helper operations.
  Run the document/draft tests and the new storage regression alone; record exact RED causes. Tests must name observable lost edits, lost isolation, unavailable controls or changed fields.

- [x] **Step 2: Port the authoring storage and services.**
  Use the source's normal transaction/migration loop and this connection seam:
  ```python
  self._connection = connect_private_sqlite(
      "workflows.local", path, isolation_level=None, check_same_thread=False
  )
  self._connection.row_factory = sqlite3.Row
  ```
  Keep normal RLock/BEGIN/commit/rollback/cursor cleanup. Remove the source constructor's raw file pinning and ALL WorkflowRuntimeLock/execution methods. Do not replace them with bespoke identity/helper protocols. Keep v1-v4 migrations immutable. Register only the workflow domain in the current shared registry and inventory.
  Reuse DocumentService and DraftSession including opaque-number preservation, revision conflict detection, generation checks and invalid-buffer recovery. Run their targeted tests and the storage regression.

- [x] **Step 3: Prove and implement real app composition/exchange.**
  Add actual lifecycle tests that enter Workflows, create/edit a definition, navigate away, quit, reopen the same private temporary DB, and read the draft/revision. Inject write refusal and prove the exact draft stays available; no success toast on failure.
  Add explicit JSON import/export tests using a temporary file with an opaque metadata field and a stable step ID; assert both survive import, edit of a different field, saved revision and export.
  Use the existing path/accessor and file picker APIs. Put potentially slow DB/file operations off the UI thread. Wire only authoring initialization/flush/close into app hooks; no runtime or Notes/provider graph. Keep existing current-screen and Console quit guards.
  Run Tests/Workflows/test_authoring.py and the relevant current navigation/quit tests.

- [x] **Step 4: Port and adapt the reviewed screen.**
  Port the editor-only source, its six local modules and real editor tests. Before UI edits read the Impeccable craft floor and repo design constitution; inherit the approved layout and dev visual world.
  Add tests for actual three-pane selection, independent collapses, validation without focus theft, hidden-pane traversal, import/export controls, and disabled Run with no runtime/lock module imported or lock file created.
  Keep current Console-follow controls functional in a secondary region; refresh only that region, not the authoring subtree. Use globals for F6 and current navigation guards. Reuse source width behavior and migrate touched styles onto existing tokens.
  Build source CSS with `python tldw_chatbook/css/build_css.py`; run editor tests and only the affected Workflows cases from the existing destination/Console/navigation files.

- [x] **Step 5: Verify and report the complete authoring slice.**
  Run one final targeted selection of Workflows, new DB authoring tests, private-owner inventory, relevant app/navigation/quit tests, and token/bundle checks. No whole Tests/UI or whole repository sweep.
  Check Ruff/format on changed files. Report existing-file baseline debt separately; do not suppress or waive it. Run git diff --check.
  Document source reuse, actual initialized files, no-execution boundary, size/privacy handling for exchange, test commands/results and any remaining failures. Coordinator performs actual capture/review handoff.
  Commit only explicit task-owned paths after checking the staged diff; do not push, merge, reset, stash or change either preserved branch.

## Implementation evidence

Implementation: `a1f47397eb1a7d62117707df914b63a8ad050998`.
Detailed report: `.superpowers/sdd/2026-09-14-workflows-authoring-dev/task-1-report.md`
(committed at `7eeed1efb13d9ad91b8c5e90a379c6ec1c91c58b`).
Final targeted selections: 355 passed plus 19 passed, 276 unrelated cases deselected.
New Workflows module/test lint and formatting and diff-check pass. Shared touched
files retain 713 baseline Ruff findings and five baseline formatter failures;
these are not waived. Completing the implementation steps does not mark the
Backlog task Done. Task code review and final branch review are separate gates.
Visual evidence and scoped correction verdict: Docs/superpowers/qa/workflows-authoring-dev/README.md.

## Approved contract amendment and review gate

Fix `9b13b49d51` adds metadata-only exchange admission; 35 covering tests pass.
On 2026-09-15 the user approved retaining picker exchange under the stable-file
assumption, recorded in ADR-138 and the spec. Retain existing checks; do not claim
the replacement race is technically fixed. This approval does not waive baseline
static debt or authorize infrastructure, execution, merge or push.

Coordinator follow-through: align the user guide and task with that contract,
request the original reviewer's finding disposition and final whole-branch review,
then record the evidence and remaining DoD gaps. Implementation checkboxes above
record completed work, not a Backlog Done transition.

# Retained Handoff Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this one bounded task with spec review before quality review. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refresh the existing Handoff label when source state changes, with no remount or eligibility-policy change.

**Architecture:** LibraryScreen retains workspace state and formats the summary. The existing LibraryRail.sync_state receives one optional string and updates its existing Static on the in-place path. All three current refresh callers supply it.

**Tech Stack:** Python, Textual, pytest/pytest-asyncio, existing native FD observer.

Spec: Docs/superpowers/specs/2026-09-10-pr2427-retained-handoff-summary-design.md.
Task: TASK-31932 step154.
ADR required: no new ADR.
ADR path: backlog/decisions/141-library-work-retirement-and-in-place-sync-followups.md and the accepted Library decomposition spec.
Reason: correct an omitted presentation write through existing owners, without another service boundary, lifetime, policy or dependency.

## Task 1: Prove the retained-label failure and patch the existing refresh

**File ownership**

- Modify tldw_chatbook/Widgets/Library/library_rail.py: optional presentation argument and exact retained Static update only.
- Modify tldw_chatbook/UI/Screens/library_screen.py: all three existing rail.sync_state callers; no unrelated Screen cleanup.
- Modify Tests/Widgets/Library/test_library_rail_workspace_action_sync.py: existing widget-level regression home and caller census if needed.
- Modify Tests/UI/test_library_entry_compose_once.py: mounted source-arrival regression using existing snapshot helpers.
- Keep Tests/UI/test_post_release_workspaces_library_depth.py assertions unchanged; its two failures must turn green.
- Main agent updates this plan, approved spec, TASK-31932 and reconciliation report after verification. Preserve the unrelated untracked September8 paydown plan.

- [x] Read the two production methods and their callers, existing widget-action test and entry `_apply_changed_snapshot` harness. Confirm three Screen calls in strict entry reconciliation, snapshot reconciliation, and lifecycle reconciliation.
- [x] Add a regression using the existing `widget_pilot`, `_make_shell`, and `LibraryRailPreferences`. Its body factory creates the real Handoff Static plus existing workspace action widgets and records how often it runs. Keep shell shape/preferences/query fixed; transition owner state from unavailable to mixed eligible/blocked and back. Assert exact `library_dim_label_text("Handoff", summary)` output, current button tooltip/class, same Static/Button identity, unchanged focus and a demonstrably nonzero scroll offset, and exactly one body-factory invocation. Use the existing formatter/state helpers, not a second policy implementation.
- [x] Add controls for omission (leave prior label intact), explicit empty string (update using the existing formatter), and absent optional summary Static (no failure/remount; existing action continues to refresh). Prefer parameterization over a new helper framework. Add concise Google-style docs and accurate test annotations.
- [x] Add a mounted entry regression: seed an empty source snapshot, open the Details section, retain rail/label/button identities and focus/scroll, then call existing `_apply_changed_snapshot` with a real stable-ID note. Await the existing reconciliation completion condition, not a longer sleep. Require changed summary equal to the current Screen formatter, current action state, and retained identities/focus/scroll. This test should fail before implementation because the existing label remains unavailable, independently of a missing new keyword argument.
- [x] Run new controls plus both unchanged original workspace failures against unchanged production and save terminal RED logs. The unit new-keyword TypeError is expected, but the mounted omitted-write failure must also be observed.

```sh
task_tmp=$(mktemp -d "$TMPDIR/pr2427-handoff-red.XXXXXX")
.venv/bin/python -I -m pytest Tests/Widgets/Library/test_library_rail_workspace_action_sync.py Tests/UI/test_library_entry_compose_once.py -k handoff -q --basetemp="$task_tmp/pytest"
```

- [x] Add `workspace_handoff_summary: str | None = None` after the existing optional action argument in `LibraryRail.sync_state`, and document it. In the existing in-place branch, beside the action update but outside the mandatory-Static exception handler, use exactly this behavior:

```python
if workspace_handoff_summary is not None:
    try:
        summary = self.query_one("#library-workspaces-handoff", Static)
    except NoMatches:
        pass  # Optional for standalone rails.
    else:
        summary.update(library_dim_label_text("Handoff", workspace_handoff_summary))
```

Do not call the body factory, change the existing shape/recompose decision, or store another summary cache. Leave optional absent rows harmless, while retaining the existing behavior for missing mandatory rows.

- [x] At all three Screen calls add:

```python
workspace_handoff_summary=self._workspace_handoff_summary_label(
    self._library_workspace_depth_state()
),
```

Do not refactor the already cached state reads or eligibility methods. Preserve generation/route guards and the lifecycle handler's existing refresh ordering. Verify the three keyword expressions by whole-file AST/caller census and independent review; add a small caller assertion in the existing test file if needed.

- [x] Rerun the new controls and both original failures, then complete the widget-action sync file. Confirm actual terminal results and scoped Ruff/formatting. No cap raises or broad formatting of existing large files.

```sh
task_tmp=$(mktemp -d "$TMPDIR/pr2427-handoff-green.XXXXXX")
.venv/bin/python -I -m pytest Tests/Widgets/Library/test_library_rail_workspace_action_sync.py Tests/UI/test_library_entry_compose_once.py -k handoff -q --basetemp="$task_tmp/pytest"
task_tmp=$(mktemp -d "$TMPDIR/pr2427-handoff-workspaces.XXXXXX")
.venv/bin/python -I -m pytest Tests/UI/test_post_release_workspaces_library_depth.py -q --basetemp="$task_tmp/pytest"
.venv/bin/python -I -m ruff check tldw_chatbook/Widgets/Library/library_rail.py Tests/Widgets/Library/test_library_rail_workspace_action_sync.py Tests/UI/test_library_entry_compose_once.py
git diff --check
```

- [x] Independent spec compliance review, then code-quality review. Address findings with the implementer; main commits after both reviews. Do not mark the whole Backlog task Done.
- [x] Once source is frozen, main runs complete affected owners under the existing observation-only native runner:

```sh
task_tmp=$(mktemp -d "$TMPDIR/pr2427-handoff-native.XXXXXX")
.venv/bin/python -I /private/tmp/pr2427-fd-identity.OTL9up/native_fd_identity.py "$task_tmp" Tests/Widgets/Library/test_library_rail_workspace_action_sync.py Tests/Widgets/Library/test_library_rail.py Tests/UI/test_post_release_workspaces_library_depth.py Tests/UI/test_library_entry_compose_once.py
```

Inspect the final native inventory and terminal exit code. The already measured pytest-asyncio kqueue/socketpair and intended faulthandler log are not application leaks; do not add cleanup hooks without new attribution. Run existing size/private-owner guards, report all residual failures, and verify preflight with the pinned local Mermaid source cache. Screen starts32715/31689 and13 original ceilings fail; this scoped correctness repair does not claim to pay down that separate debt.
- [x] Record RED/GREEN/native/review evidence in task/report and commit this bounded repair. Publish current PR progress without claiming merge readiness while other gates remain open.

## Verification checkpoint

Implementation and all specified regressions are complete; sequential spec and
quality reviews found no issues. Focused tests pass11; complete action/workspace
files pass20. Four complete native files finish135 passed/1 failed,3 warnings,
168.24s, with no SQLite/instance-lock retention. The one failure is an unchanged
Conversations fixture attempting to focus a loading-disabled row; the exact
pre-repair rail-sync control reproduces it, and a pre-release assertion proves
focus was never acquired. Do not mark that original case green or weaken its
oracle. Its separate fixture correction awaits approval. All seven preflight
checks pass; the13 existing size failures remain. Exact logs and native path
are in backlog/docs/pr-2427-rebase-reconciliation.md.

The follow-up fixture correction was explicitly approved and implemented in
TASK31932 step156. Its pre-release focus precondition fails on the old disabled
row, then passes with the existing enabled Search Input. Independent review
confirms all original race-result, selected-ID, owner and exact-focus assertions
remain. Complete four-file native rerun now passes136 tests,3 warnings,164.43s,
with zero SQLite/instance-lock retention. This closes the recorded fixture
failure, not the separate13 size-limit failures.

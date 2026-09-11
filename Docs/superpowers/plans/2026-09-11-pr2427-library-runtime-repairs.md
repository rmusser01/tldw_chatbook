# Library runtime repairs implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development for the bounded implementation task, followed by independent spec and quality reviews. Preserve the existing dirty test corrections.

**Goal:** Repair the four remaining original32 Library failure cases and qualify their affected files.
**Architecture:** Keep the existing Notes/Files projection boundaries and Library focus/footer owners. Add readiness checks and existing-target selection, not new coordinators or cleanup policy.
**Tech Stack:** Python3.12, installed Textual8, pytest-asyncio, real SQLite, native Darwin FD observation.

Spec: `Docs/superpowers/specs/2026-09-11-pr2427-library-runtime-repairs-design.md`, user approved on2026-09-11.
Task: TASK-31932. Worktree: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/pr2427-review-recovery`.
ADR required: no. ADR path: existing ADR086/141. Reason: restore existing contracts without ownership, service, persistence or policy changes.

## Task 1: bounded runtime repair, one implementation owner

Own only these runtime files:
- `tldw_chatbook/Widgets/Library/library_notes_canvas.py`
- `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py`
- `tldw_chatbook/UI/Screens/library_screen.py`
- `tldw_chatbook/UI/Library_Modules/library_notes_controller.py`

Tests belong in the existing relevant files, preserving their dirty corrections:
- `Tests/UI/test_library_shell.py`
- `Tests/UI/test_library_file_notes_workspace.py`
- Existing `Tests/Widgets/Library/test_library_notes_canvas.py` where a small mounted canvas control fits better.

Use one writer because these tests share the large Shell harness. Record each RED/GREEN separately and release all event gates in finalizers. Do not commit peers' documentation or the unrelated reader-paydown plan.

### A. Notes mount projection

- [x] Add a deterministic regression adapting `/private/tmp/pr2427_lifecycle_proof.py`: gate the exact work pane's editor `mount_all`, finish real backlinks while children are absent, then release mounting. Assert the gate was reached, latest state paints, editor-ready signaling and subsequent dirty edits work. Add a newer-state-before-release control; use the existing uniquely owned DB fixture/finalizer.
- [x] Run the new nodes; require the observed `NoMatches` path through `apply_session_state`, not an unrelated fixture failure.
- [x] Store the state before checking readiness, then reuse the established editor-child readiness boundary:

```python
self.presentation_state = state
self.compact = state.compact
if (
    self.mode != "editor"
    or not self.is_mounted
    or not self.query("#library-note-title")
    or any(not child.is_mounted for child in self.children)
):
    return
```

Existing post-compose wiring consumes the retained state. Confirm child composition order makes the title sentinel sufficient; if the controlled test demonstrates partial children, check the actual required subtree rather than catch errors. Preserve focused-field authority and controller guards.

Quality review reproduced partial children after the title became queryable.
The final predicate above uses existing direct-child mounted state: Textual
awaits each child's composed descendants before completing its Mount. Four
additional gates cover context keywords, mode controls, wide utilities and
delete actions, including no early authority write and latest-state replay.
- [x] Run the new nodes and the original `test_library_shell_pre_existing_note_emptied_out_still_saves_in_real_db`; require GREEN and no worker exception.

### B. Files teardown projection

- [x] Add a mounted regression holding the real `_build_runtime` result until the exact workspace's path-label descendant is removed but parent Unmount has not completed. Verify `is_running=False` with the other old guards still true. Release the runtime, await its actual completion, and release all teardown gates in finally.
- [x] Run RED: require the observed `_initialize -> _update_controls` missing-label error.
- [x] Extend the existing projection guard without changing acquisition/adoption/shutdown:

```python
if not self._active or not self.is_mounted or not self.is_running:
    return
```

Review sibling projections on this path; change an existing guard only if controlled evidence shows another stale projection. Do not abandon resources with an acquisition-site early return.
- [x] Run GREEN plus the original graduation/Files-switch test. Add exact owned shutdown and supported retained-workspace remount checks; do not use global cleanup.

### C. Notes footer publication

- [x] Run the corrected original navigator/create/sync/exit footer test RED. Add a focused control requiring retired token hidden/empty on Notes exit and retained contents for word/DB indicators.
- [x] At `_register_footer_shortcuts`, select `self._library_notes_footer_shortcuts()` instead of the generic selector. Keep its registration API and emergency-return updates unchanged.
- [x] In `_apply_library_notes_footer_context`, stop including the retired token selector in the ordinary ancillary visibility loop. Ensure it remains hidden/empty using the existing footer retirement contract, without exposing old values or changing word/DB responsive behavior.
- [x] Run GREEN for both existing Notes footer tests and the new control; include unrelated route/typing hints and footer replacement/pane visibility coverage. A temporary control with the old token-visible assertion is not qualification.

### D. Empty Conversations focus

- [x] Run corrected `test_background_recompose_restores_focus_on_an_empty_conversations_list` RED, starting with a current attached rail row and waiting for the real empty recovery page. Assert that entry itself does not navigate to Console.
- [x] Extend only the existing Conversations no-row fallback: prefer the attached/focusable `#library-conversations-filter` when present; otherwise try the attached/focusable `#library-conversations-empty-console`. Preserve existing retry/expiry and focus generation/user-intent checks. Other destinations keep their current fallback selection.
- [x] Run GREEN for the original node. Add/identify explicit controls for filtered-miss filter focus, nonempty row focus, newer user intent, and an unavailable target that still obeys the bounded retry/expiry path.

### Task 1 review and save

- [x] Self-review exact diff, runtime callers, owned-resource finalizers and assertion strength; run scoped fatal Ruff and whitespace checks.
- [x] Obtain independent spec-compliance review, then independent quality review; address findings in that order.
- [ ] Save the reviewed implementation with exact-path staging and `git -c gc.auto=0 commit`. Root owns documentation/plan status. No push until qualified evidence is recorded.

## Task 2: integrated qualification and checkpoint

- [x] Map all original32 IDs from `/private/tmp/pr2427-notes-dev-native-b.log`; account for renamed source-switch/handoff tests and updated parameter IDs. Require exactly32 selected original cases before adding new regressions.
- [ ] Run all32 and new regression nodes with frozen sources. All must pass; ungated success cannot replace the controlled lifecycle tests.
- [ ] Run complete `Tests/UI/test_library_file_notes_workspace.py`, `Tests/UI/test_library_shell.py`, and any touched Widgets test file. No repository-wide sweep. Preserve failed/pending output and do not rebase while running.
- [ ] Use the repository's existing deterministic pytest-shard partition for complete-file qualification: three independent native-observed processes, identical file arguments and options, `--num-shards=3` with `--shard-id=0/1/2`, no xdist. Collect an independent unsharded node inventory and compare it with the union of all observed `before_protocol` IDs: exact equality, no duplicate or omitted case, terminal outcomes for every shard. Report final descriptors separately per process. This independently reviewed strategy preserves complete coverage, not the historical whole-file single-process order; the original32-plus-races run above remains unsharded and ordered.
- [ ] Use the existing observer for each run:

```sh
task_report_dir=$(mktemp -d "${TMPDIR%/}/pr2427-runtime-verify.XXXXXX")
env -u NO_COLOR TERM=xterm-256color COLORTERM=truecolor \
  .venv/bin/python -I /private/tmp/pr2427-fd-identity.OTL9up/native_fd_identity.py \
  "$task_report_dir" EXACT_TEST_PATH_OR_NODE
```

Capture complete logs and terminal exit; inspect the report's final all-live identities for SQLite and instance locks. Never clear pytest's final traceback or force collection to produce a clean result. Gate timeouts fail loudly and test-owned resources finalize on failure/cancellation.
- [ ] Run fatal scoped Ruff, whitespace and existing derived-artifact checks appropriate to changed files. Preserve known baseline full-lint/size/preload/CSS debt separately; do not raise limits or regenerate unrelated output.
- [ ] Record evidence in TASK31932 and `backlog/docs/pr-2427-rebase-reconciliation.md`; update the quiet weekly automation with actual remaining state. Commit the checkpoint with exact staging after review.
- [ ] Only then resume fresh-dev integration and PR-review/normal protected merge under the standing request. These remain blocked by separately documented qualification issues, not silently included in completion of this repair batch.

No routine design reapproval is needed within the user-approved spec. Escalate only a materially different remedy or missing authority.

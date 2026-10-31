### Task 1: Preserve accepted visible handoffs and startup module budget

**Files:**
- Modify: tldw_chatbook/UI/Console_Modules/session.py
- Test: Tests/UI/test_console_runtime_ownership.py
- Inspect/modify as demonstrated necessary: tldw_chatbook/Chat/console_chat_start.py imports and importing owners; Tests/Performance/test_ui_ready_module_census.py (regression coverage only, preserve limits).
- Read: backlog/docs/design-language.md; ADR-097; Docs/superpowers/qa/2026-10-03-console-chat-starts-dev-integration/final-fix-scoped-review.md.

**Interfaces:**
- Consumes: _visible_agent_handoff_draft receipt, ComposerDraftSnapshot, capture_draft_for_send(), commit_captured_draft(), durable handoff incarnation/revision/state.
- Produces: Correct consumption of the original accepted visible prompt while retaining later authored drafts and replacement-widget state; startup count within unchanged budget.

- [ ] Reproduce the caret/selection-only defect in the mounted held-readiness native-start test. Extend the existing unchanged/replacement/same_text/switch_away matrix with caret-only and selection-only navigation. Assert actual accepted machine message, durable consumed handoff, empty accepted original composer, switch-away/back no resurrection, and later authored edits survive.
- [ ] Run the new cases against the current code and retain RED logs before implementation.
- [ ] Implement the smallest ownership comparison: preserve session/incarnation/revision and widget/generation/authored revision/content fencing while ignoring cursor/selection navigation. Do not clear unconditionally or use text equality alone.
- [ ] Reproduce the failed _ui_ready census. CI measured 1034 against limit1033 with console_chat_start among the new modules. Trace its import ownership and defer a genuine unnecessary boot edge; do not raise limits, refresh a failing snapshot, or delete checks.
- [ ] Run the expanded mounted matrix and directly affected handoff/composer tests, the primary-to-child shared closure regression, and affected startup/import ratchets. Capture exact commands, revisions, exits and warnings; use the shared .venv interpreter and task-private temporary profiles.
- [ ] Run fatal Ruff rules, the existing formatter ratchet for touched files, and git diff --check. Commit only source/test files and write the report with RED/GREEN evidence and self-review.

## External review and publication

- [ ] Independent task review of Task1, then whole-branch review of the rebased PR. Package diffs before dispatch; reviewers do not repeat completed test runs.
- [ ] Push with force-with-lease pinned to the verified old remote head; update the PR description and mark ready to enable reviews.
- [ ] Retrieve all PR issue/review comments and Qodo suggestions. Verify each actionable finding; dispatch concrete follow-up fixes with reproductions and covering tests. Reply and resolve addressed threads with commit/test evidence.
- [ ] Wait for current-head checks and Qodo completion. If future waiting is required, create a quiet thread heartbeat that continues review fixes and merge under the user's authorization.
- [ ] Fetch latest dev again; update/requalify if it advanced. Merge using normal repository gates and exact reviewed head, without admin bypass.
- [ ] Confirm merged state/SHA, close current Backlog criteria, archive this plan's evidence and remove only its disposable SDD directory after completion.

Observed covering-test qualification: compare the extra interaction-boot jobs-count failure on immutable f843ca811f01da6c39d903b6cd7328d68d50416f. If its expectation conflicts with current local+network refresh behavior, repair Tests/Packaging/test_console_interaction_boot_closure.py to assert that actual contract. Verify captured-send/cursor failures before repairing fixtures; keep authored edit assertions strict.

Ruling: Isolate the older skill-await snapshot test at the hook-review seam while retaining the real prompt dispatcher, runtime request and original draft assertions — current hook admission deliberately refuses a changed captured stash and has separate unchanged-owner controls — if wrong, combined skill and hook interaction coverage could be incomplete and need another regression.
Document this test boundary explicitly and qualify real hook unchanged/edit/cancel/session controls; do not change production hook semantics or weaken draft assertions.

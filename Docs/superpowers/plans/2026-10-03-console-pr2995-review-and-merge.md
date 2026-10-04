# Console PR 2995 review and merge implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development to implement this plan task by task. Steps use checkbox syntax for tracking.

**Goal:** Rebase PR #2995 on latest dev, repair the remaining composer defect and CI failures, address posted Qodo feedback, and merge the verified PR.

**Architecture:** Preserve ADR-211's durable acceptance and original allowance boundaries. Compare visible handoff ownership using authored draft identity rather than cursor navigation. Defer startup imports where the feature breaches ADR-097; retain current dev runtime contracts.

**Tech Stack:** Python 3.12, Textual 8.x, SQLite, pytest, GitHub CLI.

**Spec:** Docs/superpowers/specs/2026-10-02-console-chat-destinations-and-starts-design.md

ADR required: no
ADR path: backlog/decisions/211-console-chat-destinations-and-bounded-starts.md; backlog/decisions/097-boot-budget-ratchets.md
Reason: Restore approved draft ownership and startup ratchets; no new schema or authority policy.

## Global Constraints

- Both modes keep the user's current chat, active workspace, composer, and focus intact.
- New handoff drafts survive activation and persist edits/clears until accepted consumption or explicit discard.
- Use revision fencing for target interaction and cleanup.
- Opening text cannot execute composer slash commands or expand @ references.
- All members share finite counters, pause/review state, and the original deadline.
- Fresh chats inherit no source identity, prompt, bindings, staged inputs, or grants.
- Every genuine child creation still receives its own confirmation; trusted runtime identity overrides model claims.
- Startup budget constants remain unchanged. Preserve all current dev changes and historical QA bytes.
- Use the existing isolated worktree; no full repository test sweep, dependency installs, user profile edits, warning suppression, or new skip/xfail markers.

## Controller preparation

- [x] Incorporate latest dev81c7c94f48 backup-admission fix; verify reviewed feature source bytes unchanged and qualify affected backup/startup seams.

- [x] Record old PR head ce918c467898eb24feff216bced45b779bf310c6 and create /private/tmp/console-pr2995-merge/pre-rebase.bundle.
- [x] Fetch origin/dev at 01a2020981c6197e5cd9945e5287567ad977edfe and rebase the feature branch. Resolve the lessons append by keeping both original entries.
- [x] Verify historical QA blobs and resulting source integration; refresh scoped qualification.

### Task 1: Preserve accepted visible handoffs and startup module budget

**Files:**
- Modify: tldw_chatbook/UI/Console_Modules/session.py
- Test: Tests/UI/test_console_runtime_ownership.py
- Inspect/modify as demonstrated necessary: tldw_chatbook/Chat/console_chat_start.py imports and importing owners; Tests/Performance/test_ui_ready_module_census.py (regression coverage only, preserve limits).
- Test/repair after verified base comparison: Tests/Packaging/test_console_interaction_boot_closure.py and directly affected captured-send/cursor fixtures.
- Read: backlog/docs/design-language.md; ADR-097; Docs/superpowers/qa/2026-10-03-console-chat-starts-dev-integration/final-fix-scoped-review.md.

**Interfaces:**
- Consumes: _visible_agent_handoff_draft receipt, ComposerDraftSnapshot, capture_draft_for_send(), commit_captured_draft(), durable handoff incarnation/revision/state.
- Produces: Correct consumption of the original accepted visible prompt while retaining later authored drafts and replacement-widget state; startup count within unchanged budget.

- [x] Reproduce the caret/selection-only defect in the mounted held-readiness native-start test. Extend the existing unchanged/replacement/same_text/switch_away matrix with caret-only and selection-only navigation. Assert actual accepted machine message, durable consumed handoff, empty accepted original composer, switch-away/back no resurrection, and later authored edits survive.
- [x] Run the new cases against the current code and retain RED logs before implementation.
- [x] Implement the smallest ownership comparison: preserve session/incarnation/revision and widget/generation/authored revision/content fencing while ignoring cursor/selection navigation. Do not clear unconditionally or use text equality alone.
- [x] Reproduce the failed _ui_ready census. CI measured 1034 against limit1033 with console_chat_start among the new modules. Trace its import ownership and defer a genuine unnecessary boot edge; do not raise limits, refresh a failing snapshot, or delete checks.
- [x] Run the expanded mounted matrix and directly affected handoff/composer tests, the primary-to-child shared closure regression, and affected startup/import ratchets. Capture exact commands, revisions, exits and warnings; use the shared .venv interpreter and task-private temporary profiles.
- [x] Run fatal Ruff rules, the existing formatter ratchet for touched files, and git diff --check. Commit only source/test files and write the report with RED/GREEN evidence and self-review.

### Task 2: Repair final review close-ticket and child-tool documentation findings

**Files:**
- Modify: tldw_chatbook/Chat/console_chat_controller.py (finalize_session_close only).
- Modify: Docs/User_Guide/console/agent-runs-and-tools.md (Chat creation tools paragraph).
- Test: Tests/Chat/test_console_runtime_shutdown.py (existing meaningful controls).

**Interfaces:**
- Consumes: runtime-owned ConsoleSessionCloseTicket, current generation and remembered session grant.
- Produces: stale/mismatched/generation-refused tickets retain live grants; valid session close clears grants. Accurate child tool documentation.

- [x] Read final-branch-review.md. Reproduce existing rejected-close grant controls and the valid-ticket cleanup control on unchanged source before repair; preserve RED evidence.
- [x] Remove only the new prevalidation chat-create grant pop/comment at finalize_session_close. Preserve the existing validated post-ticket/generation grant cleanup and all runtime teardown behavior.
- [x] Correct the guide: child agents may fork a chat or create a same-workspace draft; child requests require fresh confirmation. Casual destinations and bounded starts remain primary-only. Verify current prepare/bridge contracts before wording.
- [x] Run complete Tests/Chat/test_console_runtime_shutdown.py with private profile and exact recorded command; no full suite/new markers/suppression. If baseline setup fails, diagnose/verify immutable base before repairing unrelated expectations.
- [x] Run fatal Ruff, formatter ratchet for the touched Python file against recorded BASE, and git diff --check. Commit only these two owned paths; report exact SHA, results and retained warnings.
- [x] Scoped independent re-review of these two findings and this fix diff; no second whole-branch review or repeated tests without a concrete doubt.

## External review and publication

- [x] Independent task review of Task1, then whole-branch review of the rebased PR. Package diffs before dispatch; reviewers do not repeat completed test runs.
- [ ] Push with force-with-lease pinned to the verified old remote head; update the PR description and mark ready to enable reviews.
- [ ] Retrieve all PR issue/review comments and Qodo suggestions. Verify each actionable finding; dispatch concrete follow-up fixes with reproductions and covering tests. Reply and resolve addressed threads with commit/test evidence.
- [ ] Wait for current-head checks and Qodo completion. If future waiting is required, create a quiet thread heartbeat that continues review fixes and merge under the user's authorization.
- [ ] Fetch latest dev again; update/requalify if it advanced. Merge using normal repository gates and exact reviewed head, without admin bypass.
- [ ] Confirm merged state/SHA, close current Backlog criteria, archive this plan's evidence and remove only its disposable SDD directory after completion.

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

### Task 3: Fix Qodo input validation, Redirect layout and transcript-note findings

**Files:**
- Modify: tldw_chatbook/Chat/console_agent_bridge.py (validate_new_chat_arguments wrapper/docstring).
- Modify: tldw_chatbook/Utils/input_validation.py (shared strict Pydantic chat-creation model/validation boundary).
- Modify: tldw_chatbook/Widgets/Console/console_composer_bar.py (_actions_row_width duplicate active-run width only).
- Modify: tldw_chatbook/Chat/console_chat_controller.py (build_transcript_note speaker label only).
- Test: Tests/Agents/test_agent_chat_create_tools.py; Tests/Chat/test_console_chat_create_integration.py; Tests/Chat/test_console_note_span_actions.py; Tests/UI/test_console_composer_run_controls.py; focused shared-validator regression owner.
- Read: backlog/docs/design-language.md and lessons-testing-evidence.md.

**Interfaces:**
- Consumes: existing strict string/default/length/mode/destination chat argument contract; shared agent-model constants; origin-bearing USER messages; single Redirect width reservation.
- Produces: one shared Pydantic validator at both existing creation callers, Google Args/Returns/Raises docs, unchanged literal prompt bytes and trusted authority separation; stable active/rest layout; saved note provenance labels.

- [x] Verify Qodo2/3/4/6 against actual source. Add RED controls for active/rest Redirect threshold with expected external geometry and agent/untrusted/human saved-note labels. Keep existing payload/default/type/limit controls.
- [x] Delegate public creation validation to a shared strict Pydantic model using existing agent-model constants. Preserve exact defaults, title trim-after-length check, literal prompt/instructions, optional routing, unknown-authority-field discard and non-coercion. Preserve stable public ValueError categories; no raw ValidationError/payload diagnostic leakage. Add Google doc sections.
- [x] Remove only duplicate active-run ten-cell budget. Existing Redirect reservation remains the single width source. Verify actual mounted run activation/deactivation without row/control shifts and room-fitting boundary; do not make the expected geometry depend on the buggy method.
- [x] Saved transcript notes label agent_chat_start USER rows Agent handoff and untrusted USER rows Unverified handoff; ordinary human USER/ASSISTANT and span/provenance behavior stay unchanged.
- [x] Run complete directly affected argument/integration/note/layout owners plus new shared-validator controls, private profiles and exact retained receipts. Diagnose actual failures against immutable BASE before expanding fixture repairs. No full suite/new markers/suppression/dependency installs.
- [x] Run fatal Ruff, formatter ratchet for touched Python paths against BASE and source whitespace. Commit only owned source/tests; report exact SHA, RED/GREEN, warnings and self-review.
- [x] Independent scoped spec/quality review; startup ratchets qualify shared-validation integration after runtime follow-up.

### Task 4: Settle uncertain starts and bound admission contention

**Files:**
- Modify: tldw_chatbook/Chat/console_chat_start.py (owned acceptance/commit drain/cleanup/logging/status constants).
- Modify: tldw_chatbook/DB/automatic_work.py (same-owner accepted uncertainty settlement using existing review state).
- Modify: tldw_chatbook/Chat/console_chat_controller.py (native-start owned durable-commit task only).
- Modify: tldw_chatbook/Chat/message_metadata.py (shared named handoff launch status values, no new module).
- Test: Tests/Chat/test_console_chat_start.py; directly affected automatic-work ledger and controller/owned-DB lifecycle controls; startup ratchets after combined fixes.
- Read: qodo-runtime-triage.md; ADR-211; lessons-testing-evidence.md; existing owned DB call contracts.

**Interfaces:**
- Consumes: original finite allowance root, exact attempt/runtime owner, dual-store receipt, loop-owned source/target/lifetime checks, operation-owned database connection and physically running commit task.
- Produces: accepted-unreceipted attempt leaves active state and requires explicit review in-process; root pauses atomically and stays charged; slot held until all owned physical DB/provider work drains; bounded lock-contention refusal at unchanged ledger source cutoff.

ADR required: no
ADR path: backlog/decisions/211-console-chat-destinations-and-bounded-starts.md (existing).
Reason: Implement existing accepted/uncertain charging and review contracts; use existing review state and same-owner settlement, no schema or new retry policy. Document runtime contention mitigation and retained I/O limit as implementation notes.

- [x] Verify Qodo1/5/7/8 against triage and real owners. Add RED real-ledger tests for accepted/unreceipted failure and canonical-root pause/charge/no replay, blocked physical commit cancellation retaining capacity until drain, misleading preaccept exception and hostile exception-text nonlogging, and actual SQLite contention event-loop behavior/timeout restoration.
- [x] Preserve final no-await source/target checks through ledger acceptance. Use the held operation-owned acceptance connection with scoped busy_timeout=0 restored in finally to refuse writer contention rather than freezing the loop. Do not simply offload acceptance or weaken its cutoff. Assign accepted custody before any subsequent fallible restoration. Distinguish proven preaccept lock refusal from accepted/uncertain commit; no accepted refund.
- [x] Add an owner-fenced atomic ledger settlement of accepted/unreceipted work to existing review_required state, pausing the canonical root with a fixed review reason and retaining the committed generation/deadline/counters. No automatic replay. Handle failed abort or uncertain accept outcomes conservatively if actual durable state already accepted; do not leave same-owner active accepted rows forever. If durable review settlement itself fails, retain a fail-closed runtime restriction on the affected shared allowance until verified reconciliation or owner recovery; launch metadata alone cannot block sibling work. Preserve healthy replacement-owner authority.
- [x] Own, shield and drain the exact native-start conversation-commit worker, including cancellation, before settling uncertain work or releasing automatic-primary capacity. Preserve ordinary manual/queue durable commit behavior and all source/target/incarnation fencing.
- [x] Replace shared handoff launch status raw runtime/metadata literals with named values/type vocabulary in already loaded message_metadata; no new eager module. Log unexpected failures with static phase and exception type only. Never log raw exception messages/tracebacks/prompts; do not use logger.exception. Distinguish unexpected preaccept start_failed from ordinary refusal and cancellation.
- [x] Run complete affected start/ledger lifecycle owners and focused controller commit cancellation controls, actual real-file lock/loop progress control, with private profiles. Controller qualifies combined startup/import budgets after the sequential CI repair so the final integrated tree is measured once. Retain precise cancellation/restore/error receipts and latency limits. No full suite/new markers/suppression/installs.
- [x] Fatal Ruff, formatter ratchet against recorded BASE and source whitespace; commit only owned source/tests; report exact SHA, RED/GREEN, warnings, connection-opening/fsync limits and self-review.
- [x] Independent task-scoped spec/quality review, including scoped recovery-race fix round1.
- [ ] Publish verified fixes, reply/resolve each initial Qodo thread and no-inline summary finding, then await current-head review/check completion.


### Task 5: Diagnose and repair current PR Fast Lane ownership failures

**Files:**
- Modify as diagnosed: Tests/UI/test_console_runtime_ownership.py and directly used mounted-app fixtures.
- Inspect/modify only for reproduced production defects: tldw_chatbook/app.py, app_navigation.py, UI/Screens/chat_screen.py, Chat/console_runtime.py and UI/Console_Modules/session.py.
- Inspect native commit/start stages using the reviewed Task4 source; carry a concrete uncovered durable-owner defect to the controller before expanding that source scope.
- Read: ci-ownership-triage.md, pr-initial-fastlane-job.log, ci-initial-merge-tree-receipt.json; relevant testing/live/design-language lessons.

**Interfaces:**
- Consumes: exact startup task, mounted screen/current-generation claim, attach/detach/reconciliation publication, active delivery polling and native receipt/readiness barrier.
- Produces: four reported failures have evidence-backed repairs, with original runtime/successor/poll/consumption assertions intact; a final integrated startup/CI qualification.

ADR required: no
ADR path: backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md (existing runtime/view ownership), backlog/decisions/097-boot-budget-ratchets.md, backlog/decisions/211-console-chat-destinations-and-bounded-starts.md.
Reason: restore existing runtime generation, startup and receipt contracts; a changed ownership boundary requires controller ADR review first.

- [x] Record post-Task4 BASE and four immutable7a CI failures. Run only those four nodes in original order with exact stage/claim diagnostics and private profiles. If isolated green, preserve four-file CI collection while selecting only failing nodes. No dependency installs or unexplained timeout increases; ordinary CI slowness is unproved.
- [x] Diagnose startup task/screen stack/current generation and reconciliation before repairing. Use deterministic interleaving to establish any competing-startup-screen cause; distinguish unsupported harness stacking from ordinary public single-Console navigation. For readiness timeout record task completion/outcome and controller/attempt stage provenance before deciding a repair. Retain causal RED or exact diagnostic evidence; no assertion weakening.
- [x] Apply the smallest verified production or fixture correction. Preserve shared runtime/store/controller/bridge identity, exact successor attachment, late old-screen refusal, actual active-delivery transcript polling and dual-fence handoff/composer consumption. No new retries, queue policy, lifetime owner, test skips or suppression.
- [x] Run complete affected ownership owner and the original four-file CI batch (ownership, viewless hooks, install-skill runtime tool, chat creation integration), targeted only. Retain unchanged inherited strictXFAIL/warnings and exact source/command/revision receipts. Independent task review must judge diagnoses and evidence, not only local pass status.
- [x] Fatal Ruff, BASE formatter ratchets and source whitespace; commit only diagnosed owned source/tests; exact SHA/report/self-review and independent scoped spec/quality review.
- [x] Handoff reviewed source to Task6 for latest-dev integration, owned-source/historical-QA checks and final startup/import qualification. No redundant broad whole-branch review.

### Task 6: Integrate formatter-heavy latest dev and qualify affected seams

**Files:**
- Merge/reconcile only demonstrated overlaps between reviewed feature and dev, beginning with latest-dev-structural-compare.json and its canonical diff artifacts.
- Preserve latest dev formatting, four immediate ChaChaNotes conversation writers, removed queue recovery branch, profile enrollment union, Anthropic settings/auth persistence and upstream library/security changes.
- Test only affected integration owners and final startup/import ratchets; do not repeat all unaffected feature owners or the entire repository.

**Interfaces:**
- Consumes: reviewed feature source e7cc533761, Task3/4/5 source manifests and latest dev ca2992cb10; earlier base81c7c94f48.
- Produces: branch descends from pinned latest dev, historical QA stays byte-identical, every non-overlap upstream path is retained, and owned behavior differs only by documented upstream contracts or separately reproduced corrective changes.

ADR required: no
ADR path: backlog/decisions/211-console-chat-destinations-and-bounded-starts.md; backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md; backlog/decisions/097-boot-budget-ratchets.md; existing upstream immediate-writer and provider-auth ADRs as identified.
Reason: restore and combine approved contracts and formatting; a new authority/storage/runtime policy requires controller scope review first.

- [ ] Record current BASE and pinned dev. Preserve the reviewed feature in a named local recovery ref/bundle before rebase. Fetch latest dev and report any additional functional delta before expanding qualification. Root metadata is committed before handoff; shared dirty checkout must remain untouched.
- [ ] Rebase with deliberate conflict resolution. Use uniform formatting/AST comparison where needed; preserve original feature logic and all upstream behavior. No blanket ours/theirs, waived guards, dropped tests, migrations renumbered without evidence, or undocumented semantic change. Keep historical QA bytes unchanged.
- [ ] Retain per-path proof: exact blobs for non-overlap upstream paths and historical QA; AST/canonical comparisons for every owned Python overlap, retaining literal values and annotations. Account for every non-equivalent node/function; inspect source-text guards independently where formatting matters. Source reviews remain bound to pre-rebase hashes with explicit mappings.
- [ ] Qualify native creation/commit and immediate-writer integration, actual queue/dispatch recovery owners, egress/profile overlap controls, Anthropic auth/provider persistence seams and final combined startup/import ratchets. Choose exact targeted owners after examining real upstream deltas and long stress markers; no full sweep/install/skip/suppression/budget increases. Do not duplicate all155 passing CI cases unless an actual source/fixture risk requires it; current-head external CI remains mandatory for native timeout.
- [ ] Fatal Ruff, current-dev formatter/source-text/derived guards and source whitespace as appropriate. Freeze source manifests, command/output/exit receipts and honest warnings. Commit only necessary integration repairs; no push/merge by worker.
- [ ] Report exact old/new SHAs, dev ancestry, conflict decisions, mappings, every qualification and limitation, self-review and proposed integration review surface. Root creates an immutable scoped integration package and obtains independent spec/quality review before publication; no duplicate broad whole-branch review.

## External review and publication

- [x] Independent task review of Task1, then whole-branch review of the rebased PR. Package diffs before dispatch; reviewers do not repeat completed test runs.
- [x] Push with force-with-lease pinned to the verified old remote head; update the PR description and mark ready to enable reviews.
- [ ] Retrieve all PR issue/review comments and Qodo suggestions. Verify each actionable finding; dispatch concrete follow-up fixes with reproductions and covering tests. Reply and resolve addressed threads with commit/test evidence.
- [ ] Wait for current-head checks and Qodo completion. If future waiting is required, create a quiet thread heartbeat that continues review fixes and merge under the user's authorization.
- [ ] Fetch latest dev again; update/requalify if it advanced. Merge using normal repository gates and exact reviewed head, without admin bypass.
- [ ] Confirm merged state/SHA, close current Backlog criteria, archive this plan's evidence and remove only its disposable SDD directory after completion.

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

- [ ] Record post-Task4 BASE and four immutable7a CI failures. Run only those four nodes in original order with exact stage/claim diagnostics and private profiles. If isolated green, preserve four-file CI collection while selecting only failing nodes. No dependency installs or unexplained timeout increases; ordinary CI slowness is unproved.
- [ ] Diagnose startup task/screen stack/current generation and reconciliation before repairing. Use deterministic interleaving to establish any competing-startup-screen cause; distinguish unsupported harness stacking from ordinary public single-Console navigation. For readiness timeout record task completion/outcome and controller/attempt stage provenance before deciding a repair. Retain causal RED or exact diagnostic evidence; no assertion weakening.
- [ ] Apply the smallest verified production or fixture correction. Preserve shared runtime/store/controller/bridge identity, exact successor attachment, late old-screen refusal, actual active-delivery transcript polling and dual-fence handoff/composer consumption. No new retries, queue policy, lifetime owner, test skips or suppression.
- [ ] Run complete affected ownership owner and the original four-file CI batch (ownership, viewless hooks, install-skill runtime tool, chat creation integration), targeted only. Retain unchanged inherited strictXFAIL/warnings and exact source/command/revision receipts. Independent task review must judge diagnoses and evidence, not only local pass status.
- [ ] Fatal Ruff, BASE formatter ratchets and source whitespace; commit only diagnosed owned source/tests; exact SHA/report/self-review and independent scoped spec/quality review.
- [ ] Controller incorporates latest dev after source reviews, verifies owned-source/historical-QA identity and qualifies upstream queue/dispatch recovery owners plus final combined startup/import ratchets. No redundant broad whole-branch review.

## External review and publication

- [x] Independent task review of Task1, then whole-branch review of the rebased PR. Package diffs before dispatch; reviewers do not repeat completed test runs.
- [x] Push with force-with-lease pinned to the verified old remote head; update the PR description and mark ready to enable reviews.
- [ ] Retrieve all PR issue/review comments and Qodo suggestions. Verify each actionable finding; dispatch concrete follow-up fixes with reproductions and covering tests. Reply and resolve addressed threads with commit/test evidence.
- [ ] Wait for current-head checks and Qodo completion. If future waiting is required, create a quiet thread heartbeat that continues review fixes and merge under the user's authorization.
- [ ] Fetch latest dev again; update/requalify if it advanced. Merge using normal repository gates and exact reviewed head, without admin bypass.
- [ ] Confirm merged state/SHA, close current Backlog criteria, archive this plan's evidence and remove only its disposable SDD directory after completion.

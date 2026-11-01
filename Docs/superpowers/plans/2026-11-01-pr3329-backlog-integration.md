# PR3329 integration plan

> Use superpowers:executing-plans for integration and superpowers:requesting-code-review before publication. Execution authorized after confirmed PR3328 merge 7b3b2371c3141e6168726102496db1d71181adaf; the parent grants the isolated worktree and serial test slot to the PR3329 implementer.

**Goal:** Integrate the named-workspace first-send diagnostics and actionable missing-trace-request recovery without losing newer accepted-turn or provider-contact guarantees.

**Architecture:** Retain the source PR's code-owned ConsoleSubmitBlocker, ConsoleSubmitResult.blocker, and existing trace preparation recovery. Rebase onto actual dev; preserve current continuation ownership and refusal-only replay proof. No new service or recovery path.

**Tech stack:** Existing Python, Textual, SQLite.

**Spec:** TASK-33621.23 in the immutable source snapshot; source review `../pr3329-initial-review.md`, resolved relative to the repository cwd, not this document's directory. Source 34ced0f64c4829955f0e76f18bcc36e923dee095, original base 57e330e735fff25515778e839f31ed85602c34f2. Read-only snapshot `../pr3329-source-review/metadata.json`, likewise relative to repository cwd, and its direct file paths.

## Global constraints

- Oldest eligible non-draft first; never merge drafts or mark them ready. PR3267/3297 excluded. Main checkout read-only. No release.
- Hosted CI and size limits waived by user; independent requesting-code-review replaces Qodo. Focused local tests only; serialize all tests/app imports/collection.
- Preserve current ADR094/230 custody and ADR097 trace/privacy/contact-proof behavior. No whole-controller rollback, security weakening, or replay after unknown provider contact.
- Parent owns Git refs, publication, exact leases, merge verification and protection restoration. Refresh live draft/head/auto/dev/threads before any publication.
- Existing source historical/live evidence remains attributed to its original author; do not claim it was reproduced.

## ADR disposition

ADR required: no.
ADR paths: backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md; backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md; backlog/decisions/230-console-send-preparation-and-io-ownership.md.
Reason: diagnostic cause tokens and recovery through the existing accepted-turn/trace owner implement existing contracts; no new persistence or authority boundary. Add this disposition to the existing task plan and notes during integration.

## Task 1: preserve and verify the reviewed branch

Files: source metadata lists 15 changed files. Production owners are Chat/console_chat_controller.py, console_chat_models.py, console_native_commit.py and console_send_diagnostics.py. Review source-added tests and Docs/User_Guide/logs.md. Keep current size budgets under the user's waiver. Regenerate Docs/security/production-diagnostic-inventory.json only after inspecting changed statements.

- [ ] Refresh live PR3329 OPEN/non-draft/head/branch/auto state and actual dev. If source moved, inspect its delta before relying on the source review. Disarm only this PR if armed before a necessary push/rebase.
- [ ] Parent prepares a clean isolated branch and rebases source commits on the actual previous merge. Resolve narrow conflicts preserving TraceReservationUnconfirmed re-pause, physical-contact projection, exact continuation owners, accepted Stop and citation readiness from earlier merges.
- [ ] Read final diff, original task and existing source review. Add the ADR disposition and accurate integration evidence notes. Do not claim TASK33621.141's diagnostic classification gap fixed; retain its follow-up unless integration introduces a new blocker.
- [ ] Run three source regression files: Tests/Chat/test_console_named_workspace_first_send.py, test_console_submit_blocker_code.py, test_console_trace_request_missing_pause.py. Expected 13 source cases, but collect actual receipt outcomes rather than assuming count. These cover visible card, no pre-trace provider entry, working Cancel/Retry/CaptureOff, failed rebuild retained card, cause logging and no draft body in diagnostics.
- [ ] Run Tests/Chat/test_console_trace_first_send_atomicity.py and test_console_trace_retry_recovery_proof.py. Preserve postcommit Retry/CaptureOff/Cancel/autonomous queue and unknown-contact no-replay controls.
- [ ] Run native commit nodes: test_controller_store_replacement_cannot_publish_or_dispatch_original_save_into_successor; test_cancelled_accepted_save_retains_recovery_if_terminal_write_fails; test_trace_failure_after_actual_commit_releases_owners_and_preserves_cancelled_draft in Tests/Chat/test_console_native_commit_integration.py.
- [ ] Run Tests/Chat/test_console_send_diagnostics.py and Tests/Architecture/test_persistent_diagnostic_inventory.py after inventory reconciliation.
- [ ] Run mounted Tests/UI/test_console_trace_capture_recovery_flow.py plus test_trace_recovery_action_is_consumed_before_double_press, test_trace_recovery_cancel_calls_terminal_controller_entrypoint, test_trace_recovery_refusal_reenables_all_actions, test_trace_refusal_copy_survives_transcript_sync_and_is_painted in Tests/UI/test_console_trace_call_recovery.py. Existing actual mounted fixtures remain required; do not substitute model-only assertions.
- [ ] For any integration regression, prove the actual causal failure, make a narrow fix and rerun affected controls. Use qualified private Windows fixture runner before config imports, with current base and all changed/test fixture inputs enrolled. Preserve receipt before/after/current hashes, commands, case results and any failure classifications.
- [ ] Inspect diagnostic --statements for changed owners against actual base; update inventory and verify canonical inventory and census contracts. Existing current-base census entries must remain. New UI entries require actual passing receipts; size waiver does not authorize deletions or budget changes.
- [ ] AST-parse touched Python and compare Ruff findings against actual base; fix introduced diagnostics. No full sweep.
- [ ] Parent commits a concrete candidate and dispatches independent whole-branch review of base..candidate, task/plan, source review, fresh evidence and declined follow-ups. Resolve Critical/Important findings and re-review fixes before publication.
- [ ] Parent rechecks remote exact state, uses reviewed-head exact-lease helper, verifies actual MERGED/merge commit and full protection restoration.

## Evidence limits and retained follow-ups

The new-workspace regression replaces _stream_assistant_response whole. It proves real workspace creation/controller/store behavior at that seam, not live gateway/notice/picker/paint. Source live evidence is not available for independent reproduction here. TASK33621.98 (declined notice), .140 (saved refusal copy), .141 (settlement ERROR/category), .142 (raw code/stale shelf) remain explicit follow-ups. No latency, hosted-CI or full-suite claim.

## Self-review

All three source acceptance criteria map to the new source files and existing mounted recovery coverage. Five integration risks are covered: missing request/no contact; retry rebuild failure; cancellation/physical commit; stale destination ownership; unknown prior transport contact. No new behavior is selected by this preparation plan.


---
id: TASK-37653
title: Qualify multi-model code review live through a completed verifier report
status: Done
assignee:
  - '@claude'
created_date: 2026-10-30 03:54
updated_date: '2026-11-02T23:59:00Z'
labels:
- agents
- code-review
priority: high
references:
- backlog/tasks/task-35582 - Implement-primary-agent-multi-model-code-review-recipe.md
documentation:
- Docs/superpowers/qa/2026-10-18-multi-model-code-review/macos-final60/report.md
- Docs/superpowers/qa/2026-10-18-multi-model-code-review/task37653-live60/report.md
- backlog/decisions/281-primary-agent-multi-model-code-review.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-35582 merged the multi-model code review recipe on offline and scripted evidence. Its bounded live Console/provider qualifications sealed and presented reviewer reports, but the fresh verifier never completed a live unified report: the last attempt (Docs/superpowers/qa/2026-10-18-multi-model-code-review/macos-final60/report.md) failed with primary review_refused report reads and a 'Provider continuation could not be persisted' error at the 600-second deadline, so the verifier never launched. This task carries TASK-35582's former live-qualification criterion so users can rely on the verification stage with real models.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The macos-final60 review_refused report reads and the continuation-persistence error are traced to an internal cause or reproduced offline, and any product defect found is fixed with a regression test.
- [x] #2 A bounded real Console/provider qualification uses two reviewer destinations and a fresh verifier, including a pre-authorized request under the default two-child ceiling, seeded defect/false positive/duplicate, failed reviewer, truncation, source drift, small-context and storage-overflow controls; live-model evidence is distinguished from scripted controls.
- [x] #3 The live run seals the verifier report and produces a unified report that expands every original finding exactly once.
- [x] #4 The owner approves the live request/child budget before the run, and actual spend is recorded with the evidence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
AC#1 (offline; AC#2-#4 need an owner-budgeted live run):
1. Rebuild the macos-final60 primary timeline from the retained packet (`runs.db` steps, `chat.db` continuation/sync rows, runner deadline code), read from scratch copies only.
2. Trace every path that fails a primary `code_review_read` before or after dispatch, and every path that fails continuation persistence, and keep only those consistent with the timeline.
3. Reproduce the consistent paths offline (real Console composition over fake physical HTTP; real instruction ledger with the real review provider).
4. Fix each product defect at its root with a RED->GREEN regression test; run the affected suites and preflight.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
**AC#1 (2026-10-31, branch claude/burn-37653 off dev 94a8a85f03).** Timeline rebuilt read-only from scratch copies of the retained packet (`/private/tmp/task35582-macos-final60/attempt-f_yrr5n4`: `runs.db` steps, `chat.db` continuation/sync/semantic-revision rows, `runner.py`). No live calls. Four findings: three defects fixed, one by design.

1. **Terminal "Provider continuation could not be persisted" = a cross-thread race (defect, fixed).** The deadline watchdog's `stop_active_run` fired at ~02:54:43.9 (it starts with `submit_draft`; 600 s later). The owner row shows `ToolCallExecuting` for `1d0e_0` committed at .760 (v42, checkpoint rev 40), then a non-semantic write at .922 (v43: no content/state/continuation change, no semantic revision) -- the Stop's terminal write -- then `tool_failed` at .929 and the error at .942; the checkpoint never left rev 40. `ConsoleChatStore.persist_provider_continuation_event` (agent worker) did not take the owner's generation lock that `mark_message_stopped`/`persist_selected_generation` (owner loop) hold, so the worker could read the durable version after the Stop's commit but before the Stop published the in-memory `provider_continuation_message_version`: `"Continuation version conflict; reload and retry."`, which `agent_runtime.persist_continuation` collapses into the generic error. A plain sequential Stop mid-tool persisted cleanly in every probe (single/multi-round, capture on/off, with the fence's status message) -- only the interleaving fails. Fix: the event decorator now holds `_generation_owner_scope(owner)` (an RLock already used by every other generation mutator). Regression `Tests/Chat/test_console_stop_mid_tool_continuation.py` (real Console controller/bridge/store over fake physical HTTP, custom-hosted chat_completions like the live primary) pauses the Stop between its commit and its version publication: RED `('error', ['Provider continuation could not be persisted; ...'])` with `Continuation version conflict`; GREEN `cancelled`, no error step.
2. **The round-9 pair (`call_e61b..._0/_1`) was never a review refusal (defect, fixed).** No `tool_proposed` event exists for it; the only runtime path that fails calls before proposal is the project-instruction deferral (`agent_runtime.py` `retry_with_context` branch), which writes the fixed "Deferred because project instructions were loaded" result; the next `ToolBatchReady` re-projection relabels it `review_refused`. With no AGENTS.md in the source and the root already resolved by the earlier git tools, the only trigger left is `CodeReviewToolProvider.path_targets` raising (refused arguments/reference), which the ledger maps to an `outside_instruction_scope` warning -- deferring the batch and delivering the real refusal a round later (round 10, `a09d`, the first individually traced failure). That cost ~55 s of the 600 s budget. The earlier v1-final60 deferral of four `code_review_read` calls (instruction-deferral-checkpoint.md) fits the same path; its arguments were not retained either. Fix: `path_targets` returns `()` for arguments it refuses (`invoke` refuses the same arguments and touches no path). RED: the ledger returned `retry_with_context` for a forged `document_ref`; GREEN: `proceed`, and the read returns `invalid_source_reference` in the same round. `test_public_authority_failures_do_not_produce_paths` now asserts `()` instead of a raise (same intent: no paths; invoke still refuses).
3. **Generic durable label (diagnostic defect, fixed).** `CodeReviewToolProvider.project_tool_record` mapped every failure to `review_refused`. It now keeps a refusal that is a snake_case code (`[a-z][a-z0-9_]{0,63}`, the closed ReviewRefused vocabulary, e.g. `invalid_report_reference`, `stage_grant_unavailable`); any other text (verdicts, hook messages, the deferral sentence) still projects as `review_refused`, so no body or path can leak. Regression in `Tests/Agents/test_code_review_isolation.py`.
4. **`1d0e_0`'s failure is by design**: `_signal_stop` fences the review session before marking the run stopped, so the in-flight read refused (the final pages all refuse `stage_grant_unavailable`).

**Unproved.** Round 9/10 arguments are omitted in the evidence, so their exact refusal code is inferred, not observed; the race is reproduced in its commit-before-publication order -- the read-then-commit order (optimistic-lock failure) is closed by the same lock but was not separately reproduced.

**For AC#2's live run.** Durable refusals now carry their code (runs.db tool_result, transcript `ERROR: <code>`, continuation results); a genuine instruction deferral still shows as a failed call with no `tool_proposed` event and a `review_refused` continuation result. A Stop mid-tool should now end `cancelled`; if "could not be persisted" recurs, it is a new cause -- have the runner wrap `persist_provider_continuation_event` (as the regression does) to keep the content-free exception text. V1 can only launch in a later automatic turn (default two-child ceiling, ADR-281 item 11), so the primary must end its first turn; in macos-final60 it kept reading reports for six rounds at 40-55 s each until the deadline, which is why no wake/reservation exists. Size the wall deadline for that turn boundary plus the verifier stage.

**Verification.** The three new regressions (plus the updated authority-failure test) are RED on the base product files and GREEN with the fixes (69 passed across the three files). Affected suites with the fixes: 26 review files 800 passed / 8 failed; 17 continuation/instruction files 507 passed / 71 failed. The same 39 failing test functions (79 cases) fail identically with the base product files swapped back in -- local environment and dev drift (`RecoveryRequired: raw_source_selection_changed`, "Hooks unavailable", semantic-mutation inventory rows for untouched modules, queued-custody), not this change. `test_module_size_ratchet.py`: 8 pre-existing failures, none in a touched file. `scripts/preflight.sh` passed; ruff clean on the changed code.

Files: `tldw_chatbook/Chat/console_chat_store.py` (decorator; size unchanged at the 22,338 ratchet row), `tldw_chatbook/Agents/code_review_tool_provider.py`, `Tests/Chat/test_console_stop_mid_tool_continuation.py` (new), `Tests/Agents/test_code_review_tool_provider.py`, `Tests/Agents/test_code_review_isolation.py`, `Docs/User_Guide/console/agent-runs-and-tools.md` (refused reads now show their code).

**Review round (2026-10-31, claude/burn-fixB off f58ff6e2a8).** Seven findings fixed; items 2 and 3 above are narrowed by this round.
- *Lock order.* The canonical order is generation owner scope -> `_first_persistence_lock` (`_generation_owner_scope`'s docstring; `mark_message_*`, `persist_message_if_needed`, `settle_message_thinking` and the continuation write all persist under an owner). `close_session` was the one inverter: first persistence, then the purge re-entering every owner. Close now claims every owner of the session (index order, parent before descendants) before first persistence; a static call-graph pass found no other first-persistence holder that reaches an owner scope. Regression `test_close_racing_a_continuation_write_cannot_deadlock`: RED both threads still alive after bounded 10 s joins; GREEN.
- *Seal outside the owner scope.* The managed finalizer (plugin `run_coroutine_threadsafe(...).result()`, no timeout) now runs before the scope is taken; the seal is a pure signature over (checkpoint, conversation, message, run), so the locked write gets it back only for identical arguments and refuses "changed while sealing" otherwise. RED: a Stop could not take the owner while the finalizer ran (`[False] == [True]`); GREEN, plus the refusal case.
- *path_targets.* Only argument-shaped refusals (`_ARGUMENT_REFUSALS`: invalid_source_cursor/path/reference/selection, source_cursor_conflict, source_outside_scope, source_path_unavailable, plus ValidationError) map to no path; authority/state refusals raise again so the ledger still defers. RED: five state cases did not raise.
- *Projection allowlist.* The shape regex is replaced by `_REVIEW_REFUSAL_CODES` (145 codes); a test pins it to every literal `ReviewRefused` reason in `Agents/code_review_*.py`. RED: `provider_not_ready` and a 64-hex digest were kept.
- Decorator renamed `_continuation_event_owner_scope`; User Guide sentence now covers every review tool failure.
- `console_chat_store.py` ratchet row 22,338 -> 22,410 (+72, dated comment). Preflight green. Failures in the touched suites are identical with the base product files swapped back in (Architecture: 47, environment/drift; 28 Console controller/preparation cases).

**Review round 3 (2026-10-31, claude/burn-fixB).**
- *Stop test flake.* Stop cancels the test's own `submit_draft` task while the agent worker runs on under `asyncio.shield` (`_run_maintenance_agent_call`) until it writes the terminal status, so the test could read `running`. It now awaits `controller._maintenance_background` first. RED, made deterministic by delaying the worker's `cancelled` terminal write 0.5 s: `('running', []) != ('cancelled', [])`. GREEN with the delay, and 24/24 plain runs (6 in parallel) plus 6/6 delayed.
- *Allowlist completeness.* `_REVIEW_REFUSAL_CODES` is now every `ReviewRefused` reason raised under tldw_chatbook/ (164; it was missing 18 Console-owned codes such as `review_session_fenced`). The pin test scans the whole package and checks both directions; the five codes routed through variables are pinned to the module that routes them. RED: 18 extra raised codes.
- *Refusals that cannot clear.* `unknown_review`, `review_expired` and `unsupported_source_type` now map to no path, so they no longer defer the whole batch; transient state/authority refusals still raise. The test gained a symlink case and asserts that invoke refuses each path-free case with an argument code. RED: expired, other_chat and symlink raised.
- *Stop persists the owner mid-seal.* The seal covers `persisted_message_id or message.id`, so a Stop that persisted the unpersisted assistant (DB-assigned id) during the unlocked seal made the recheck refuse. On that refusal (`_StaleContinuationSeal`) the write now leaves the scope, re-seals once over the fresh inputs and retries; a second change still refuses. RED: "Continuation owner changed while sealing"; GREEN: the row is written with a seal over the durable id (`sealed_for == [native, durable]`).
- *Close snapshot.* Close passes the owner ids it claimed to the purge, which refuses ("joined this chat while closing") rather than invalidate an owner whose scope it does not hold. RED: did not raise.
- Store ratchet row 22,410 -> 22,434 (+24, dated). Affected suites: 1002 passed. The Console store/controller suites fail only the 30 cases already known to fail on the base files.
**Review round 4 (2026-11-02).**
- *Close refusal removed.* Round 3's "message joined this chat while closing" refusal fired in reachable use and left the tab unclosable until restart. Close now claims late-joining owners without waiting, or waits for a held one outside first persistence, before anything irreversible; the purge never waits.
- *Dead retry removed.* The re-seal-once retry could never succeed, because managed owners keep native id == persisted id, so it is gone. The symlink case skips on Windows hosts without symlink rights.

**Live qualification (AC#2-#4, 2026-11-02, head 949ddbc452 on dev 88af9f9379).** Evidence: Docs/superpowers/qa/2026-10-18-multi-model-code-review/task37653-live60/report.md.
- *Approvals.* The owner approved each run (60 requests / 3 children, one invocation each, no retries, CI waived).
- *Run 1* spent 40 requests and 2 children. Both reviewers sealed, but the verifier never started: the gated harness predated PR #3286's presenter refactor, set `controller.app` directly instead of attaching through `wiring.attach_review_presenter`, and so never presented the originals (`original_reports_not_presented`).
- *Run 2* spent 50 requests and 3 children, with that one-line harness fix. Both originals were presented before the fresh Haiku verifier, V1 sealed, and the unified report expanded R2-F1..F3 once each:
  - the seeded empty-input regression: confirmed;
  - the seeded false string claim: refuted;
  - `bool` passing the int check: confirmed.
- *Coverage limits.* R1 (gpt-6-luna) found nothing in both runs. Duplicate grouping and the failed-reviewer, truncation, drift, small-context and storage-overflow controls come from the harness's scripted offline controls (passed at this head) and are reported separately from the live evidence.
- *Harness repairs.* macOS /tmp cleanup had deleted the raw history packets, so the runner records them as lost rather than refusing (spend caps and pins unchanged). A dev fixture-signature drift was absorbed.
- *Archives.* Under ~/.codex/artifacts/task-37653/, credential scan clean.
- *Totals.* 574 requests / 38 children after these runs.
- Final console_chat_store row, restated 2026-11-02 against dev 88af9f9379 (supersedes the numbers above): dev measures 22,400; this branch's net +122 plus the +2 lock-order comment gives 22,524.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->

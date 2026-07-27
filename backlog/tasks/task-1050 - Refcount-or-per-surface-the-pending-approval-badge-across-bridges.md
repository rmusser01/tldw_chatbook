---
id: TASK-1050
title: Refcount or per-surface the pending-approval badge across bridges
status: Done
assignee:
  - '@claude'
created_date: '2026-07-27 14:30'
updated_date: '2026-07-27 22:06'
labels:
  - console
  - approvals
  - concurrency
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The per-session pending-approval badge (`_pending_approvals`, toggled via `set_run_pending_approval`) is a single boolean shared across every approval surface a session can have outstanding at once -- MCP tool approvals, skill-install confirms, and skill-script confirms are three independent bridges that all write it. Each bridge's teardown (`request_mcp_approvals`'s `finally` around `console_chat_controller.py` ~:2100-2110, `request_skill_install_confirm`'s `finally` ~:2493-2517, `request_skill_script_confirm`'s `finally` ~:2687-2712) sets the badge to `False` for its own session_id independently of whether a sibling round from a DIFFERENT bridge is still outstanding for that same session. Whichever round finishes first clears the badge even though another one is still pending, so the fleet UI can show "nothing needs attention" for a session that in fact still has a live confirm waiting.

Compounding this, the skill-install and skill-script bridges' `finally` blocks already compute a `still_armed_same_session` guard (checking their OWN round map for a sibling round belonging to the same session) before deciding whether to clear the MOUNTED card -- but the session-keyed parked-payload pop (`_parked_skill_install_payloads.pop(session_id, None)` / `_parked_skill_script_payloads.pop(session_id, None)`) and the `set_run_pending_approval(session_id, False)` call right above it run UNCONDITIONALLY, ahead of that guard, with no equivalent check. The mounted-card clear is guarded; the badge clear and payload pop are not, even though they exist in the same teardown block for the same reason.

This is production-unreachable today: per-session tool dispatch is sequential, so a single session can only ever have one approval round of any kind outstanding at a time. It becomes a live correctness bug the moment same-session concurrent rounds are ever allowed to exist (e.g. an agent that can request an MCP approval and a skill-install confirm in the same turn, or two backgrounded runs sharing a session).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The pending-approval badge for a session stays set while ANY surface (MCP approval, skill-install confirm, skill-script confirm) has an outstanding round for that session, and only clears once none remain.
- [x] #2 The session-keyed parked-payload maps are only cleared for a bridge's own round, never in a way that discards a sibling surface's still-armed payload for the same session.
- [x] #3 A regression test exercises two concurrent same-session rounds from different bridges (or, if dispatch is truly sequential today, a direct unit test of the teardown logic) and proves the badge/payload survive the first round's resolution.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Audit every call site of `set_run_pending_approval` (controller bridges + ChatScreen._park_console_approval + direct test callers) to decide design (a) full-migration-with-round-id-signature vs (b) new add/discard_pending_round methods + deprecated boolean shim.
2. Change `_pending_approvals` from `set[str]` to `dict[str, set[str]]` (session id -> outstanding round ids); add `add_pending_round`/`discard_pending_round`/`has_pending_approval_round`; keep `set_run_pending_approval` as a deprecated shim backed by a reserved sentinel round id so it composes safely with real round ids.
3. Migrate all three bridges (request_mcp_approvals, request_skill_install_confirm, request_skill_script_confirm) arm/teardown to the round-keyed API; extend MCP's teardown with the still_armed_same_session guard the other two bridges already had (defect B parity across all three).
4. Guard each bridge's parked-payload pop with "last armed round for session OR stored payload still belongs to this round" so an earlier round's teardown never evicts a newer round's payload under the same session-keyed slot.
5. Guard ChatScreen._park_console_approval's redundant badge stamp with has_pending_approval_round so the deprecated shim never leaks a stale sentinel past a real round's resolution.
6. Rename _RunMarkerBearer -> RunMarkerBearer and test _FakeApp -> FakeApp (rider).
7. Add tests: same-session cross-bridge survival, same-session same-bridge payload-overwrite guard, idempotent add/discard, legacy-shim composition.
8. Run the two hard-rule pytest gates in the worktree venv; confirm the 2 known pre-existing failures are unchanged via git stash comparison.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replaced the single shared boolean `_pending_approvals` (set[str] of session ids, written by `set_run_pending_approval`) with round-keyed accounting: `_pending_approvals: dict[str, set[str]]` (session id -> outstanding round ids), via new `add_pending_round`/`discard_pending_round`/`has_pending_approval_round`. `run_marker_for`/`fleet_summary_counts` needed no code changes since dict key membership/iteration mirrors the old set semantics exactly.

Audited every `set_run_pending_approval` caller: all 6 in-controller bridge call sites (request_mcp_approvals x2, request_skill_install_confirm x2, request_skill_script_confirm x2) have a real round/request uuid4 id in scope and were migrated to the round-keyed API. `ChatScreen._park_console_approval` genuinely lacks one (its public contract `Callable[[str], None]` is wired directly to single-arg collectors in 3 existing tests, and one test drives it standalone with a payload carrying no round id) -- kept `set_run_pending_approval` as a deprecated shim backed by a reserved sentinel round id (composes safely alongside real ids in the same set), and guarded its one remaining call site with `has_pending_approval_round` so it never leaks a stale sentinel past a real round's own resolution.

Defect B fix (payload retention) applied to all three bridges: each `finally` now only pops its session-keyed parked-payload map when either this is the LAST armed round for the session (own bridge's `still_armed_same_session`) or the currently-stored payload is still THIS round's own (guards the "earlier round's teardown evicts the newer round's overwritten payload" case, since the map is keyed by session id alone, not by round id). MCP previously had no `still_armed_same_session` guard at all (unlike skill-install/skill-script, task-581/TASK-910) -- added it for parity, including gating the mounted-card clear, not just the payload pop.

Rider: `_RunMarkerBearer` -> `RunMarkerBearer` (conversation_browser_state.py), `_FakeApp` -> `FakeApp` (test_skill_install_concurrent_confirms.py); both single-use, no external importers.

Tests added (all additive, zero existing-test signature changes -- every pre-existing `in`/`not in controller._pending_approvals` assertion passed unchanged against the new dict type): 4 new in test_console_run_markers.py (same-session multi-round survival, idempotent add/discard x2, legacy-shim composition), 2 new in test_console_mcp_approval.py (cross-bridge MCP+skill-install same-session survival, same-bridge two-MCP-round payload-retention-under-overwrite), 1 new in test_skill_install_concurrent_confirms.py, 1 new in test_skill_script_concurrent_confirms.py (+ session-scoped _arm_for_session helper).

Verified in worktree venv (import path confirmed /private/tmp/tldw-approval-acct). Gate 1 (mcp_approval + skill_install_concurrent + skill_script_concurrent + run_markers + parallel_runs): 79 passed, 2 failed -- both are the named pre-existing failures (CSS-geometry batch-row test, mcp cancellation execution-log test), confirmed via git-stash comparison against the unmodified branch. Gate 2 (run_state_per_session + skill_install_confirm): 32 passed. ruff check clean on all 7 touched files.

Full design rationale, per-bridge diffs, and caller audit table: .superpowers/sdd/approval-accounting/task-1050-report.md
<!-- SECTION:NOTES:END -->

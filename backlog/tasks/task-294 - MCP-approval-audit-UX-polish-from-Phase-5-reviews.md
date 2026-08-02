---
id: TASK-294
title: MCP approval + audit UX polish from Phase 5 reviews
status: Done
assignee: []
created_date: '2026-07-17 19:18'
labels:
  - mcp
  - ux
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
sr-UX-adjacent polish deferred from PR #675: approval card shows no countdown/deadline hint (silently vanishes at 120s); 'Approve all' requires a second Submit click; the deny refusal copy 'blocked by MCP permissions (set to Off)' is reused for explicit user denials (misleading provenance); Audit 'When' column renders raw UTC ISO without tz conversion; test docstring says height<=3 while asserting <=4. Also fold the remaining P5 test-hygiene minors: hook test against spawn/find/load names, ChatTaskCards.sync_state batch branch direct test, misnamed collapse test, tool_naming dedupe docstring warning.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each item addressed or explicitly declined with reasons in this task
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Per-item disposition (2026-08-02):

1. **Countdown/deadline hint** -- DELIVERED EARLIER by TASK-1844 (PR #1192): "Auto-denies in M:SS" on the card, mounted-widget tested.
2. **"Approve all" requires a second Submit click** -- DECLINED, deliberately. The two-step (bulk-set, review, commit) is the safety pattern for a multi-row batch: since TASK-1861 each row is a separate per-target decision, and one-click bulk approval would remove the only moment the user sees what the bulk action selected before it fires. TASK-1845 moved the card AWAY from one-keystroke approval for exactly this reason, and the single-row case already has one-click fast buttons (F5). Collapsing the multi-row commit step reverses that direction.
3. **Deny copy misleading provenance** -- FIXED. `_apply_verdict`'s explicit user "Deny" now returns `USER_DENY_REFUSAL` ("tool call denied by the user"); `DENY_REFUSAL` ("blocked by MCP permissions (set to Off)") remains on the two paths where it is true (permanent deny state; no approval callback). Copy matches the builtin gate's and the review hook's user-denial wording, so a refusal reads the same at every layer. Reconciling the old pinned tests exposed a SECOND conflation my first fix would have inverted: `invoke()` defaulted a MISSING verdict to "deny" before `_apply_verdict` could distinguish it, so "nobody decided" would have read "denied by the user" -- the same provenance lie pointed the other way. Missing/unrecognized verdicts now fail closed as `UNRESOLVED_REFUSAL` ("tool call not approved (no decision recorded)"), blaming nobody. Qodo then caught the SAME principle one layer deeper: the transcript said "no decision recorded" while the AUDIT log still recorded `decision="denied"`. The unresolved branch now records `denied-unresolved` (mirroring the `denied-timeout` vocabulary), the audit Decision filter offers "Denied (no decision)", and the Outcome column buckets it as Blocked.
4. **Audit "When" column raw UTC** -- FIXED. `_format_when` converts tz-AWARE values via `.astimezone()` (viewer-local); naive values render unchanged (inventing a zone would be a different lie). The failing test demonstrated the bug numerically: 12:00 UTC displayed as 12:00 on a UTC-7 machine.
5. **height<=3 docstring vs <=4 assert** -- FIXED, stale twice over: the assert became <=6 in TASK-1846 (three stacked lines) while the docstring still said <=3. Docstring now states the current contract and why.
6. **Hook test against spawn/find/load names** -- ADDED. TASK-631's test proves those names are REFUSED with the kill switch on; the new `test_unclaimed_names_pass_through_the_hook_unreviewed_switch_off` pins the OFF half (no prompt, empty verdicts). Sabotage-verified: force-claiming unclaimed names fails it.
7. **ChatTaskCards.sync_state batch branch direct test** -- ADDED (`Tests/UI/test_chat_task_cards_sync.py`): payload -> two per-call-id rows visible, round_id rides through, cleared state hides the surface. Sabotage-verified: dropping the round_id fails it.
8. **Misnamed collapse test** -- RENAMED to `test_request_mcp_approvals_routes_one_decision_to_duplicate_names` with a docstring saying what it actually proves; nothing in it ever asserted payload collapsing, and since TASK-1861 visual collapsing is keyed per call id anyway.
9. **tool_naming dedupe docstring warning** -- ADDED: stability is ORDER-dependent, so a catalog rebuilt in a different order can swap a `_2`-suffixed name with its unsuffixed twin, silently re-pointing anything keyed by LLM name (session approvals, audit grouping).
<!-- SECTION:NOTES:END -->

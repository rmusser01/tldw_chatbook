# TASK-1050: Round-keyed pending-approval accounting across bridges

## Design decision: (a) vs (b), with caller audit

The task asked to choose between (a) keeping `set_run_pending_approval`'s
name with a round-id-aware signature (`set_run_pending_approval(session_id,
round_id, pending)`) and migrating every caller, or (b) adding
`add_pending_round`/`discard_pending_round` and keeping the old boolean
setter as a deprecated shim only for callers that genuinely lack a round id.

**Chosen: (b), after auditing every caller of `set_run_pending_approval`.**

Full call-site audit (`grep -rn set_run_pending_approval`):

| Call site | Has a round id in scope? |
|---|---|
| `request_mcp_approvals` arm (was line ~2160) | Yes -- `round_id` (uuid4, minted at round start) |
| `request_mcp_approvals` `finally` (was ~2245) | Yes -- same `round_id` |
| `request_skill_install_confirm` arm (was ~2646) | Yes -- `request_id` (uuid4) |
| `request_skill_install_confirm` `finally` (was ~2676) | Yes -- same `request_id` |
| `request_skill_script_confirm` arm (was ~2868) | Yes -- `request_id` (uuid4) |
| `request_skill_script_confirm` `finally` (was ~2896) | Yes -- same `request_id` |
| `ChatScreen._park_console_approval` (chat_screen.py ~15824) | **No** |
| `Tests/Chat/test_console_run_markers.py` (direct calls, 6 sites) | No -- exercises the boolean API on purpose, single-round scenarios |

All six in-controller bridge call sites have a real round/request id
available and were migrated to `add_pending_round`/`discard_pending_round`
(full migration, matching (a)'s spirit for every caller that can support
it).

`ChatScreen._park_console_approval` is the one production caller that
genuinely cannot: its public contract, `ConsoleChatController.
park_pending_approval: Callable[[str], None]`, takes only a session id.
Three existing tests wire it directly to a one-arg collector
(`controller.park_pending_approval = parked.append` in
`test_console_mcp_approval.py`, `test_console_skill_install_confirm.py`,
`test_console_skill_script_confirm.py`), and `Tests/UI/
test_console_parallel_runs.py::test_background_approval_parks_with_badge_and_single_toast`
calls `console._park_console_approval(background)` directly with a payload
seeded with **no round id at all**, then asserts the marker becomes
NEEDS_APPROVAL from that call alone. Changing `park_pending_approval`'s
arity to carry a round id would have forced "mechanical" signature changes
onto four passing tests outside this task's four gated files for no
behavioral gain -- the owning bridge (`request_mcp_approvals`/
`request_skill_install_confirm`/`request_skill_script_confirm`) already
registers the round's real id via `add_pending_round` *before* invoking the
park callback, so by the time `_park_console_approval` runs, the real round
is already tracked. Its own badge-stamp is redundant in the live path and
exists only to keep it usable as a standalone test seam.

**Design chosen for the shim:** `set_run_pending_approval(session_id,
pending)` is kept, marked deprecated, and internally represented as a
reserved sentinel round id (`_LEGACY_PENDING_APPROVAL_ROUND_ID`) added to /
discarded from the same per-session round-id set `add_pending_round`/
`discard_pending_round` use. This makes it compose safely: it can never
collide with a real bridge round's `uuid4()` id, and calling it while a
real round is registered only adds/removes its own sentinel, never
touching the real round's id.

That still leaves a latent hazard: if `_park_console_approval` always
called the shim, a real round's `discard_pending_round` at teardown would
leave the shim's own sentinel behind, leaking a stale badge past the
round's actual resolution. Fixed by adding `has_pending_approval_round
(session_id)` and guarding the shim call in `_park_console_approval`:
only stamp the shim when no round (real or sentinel) is registered yet.
In the live bridge path this guard is always true-guarded-off (the real
round already exists); in the standalone-test-seam path (no live round)
it falls through to the shim exactly as before. Pinned by
`test_legacy_boolean_shim_stacks_with_a_real_round_until_both_clear` in
`Tests/Chat/test_console_run_markers.py`.

## Defect A: round-keyed accounting

`_pending_approvals` changed from `set[str]` (session ids) to
`dict[str, set[str]]` (session id -> outstanding round ids). `run_marker_for`
and `fleet_summary_counts` needed **no code changes** -- `session_id in
dict` and `set(dict)` both operate on keys, identical to the old `set[str]`
semantics, so every pre-existing `in`/`not in controller._pending_approvals`
assertion in the test suite (7 sites across 5 files) keeps passing
unchanged.

New API: `add_pending_round(session_id, round_id)` / `discard_pending_round
(session_id, round_id)`, both lock-guarded under the existing
`_approval_state_lock`, both idempotent (set semantics -- double-add is a
no-op, double-discard is a safe no-op). A session reads NEEDS_APPROVAL iff
it has at least one round id in its set; `discard_pending_round` pops the
whole session key once its set empties, so no stale `{}` entry can ever
read as pending.

The terminal-run-state clear (`_set_run_state`, was `_pending_approvals.
discard(target)`) changed to `_pending_approvals.pop(target, None)` -- a
terminal run has no live approval left from ANY bridge, so it clears the
session's entire round set, not just one round's id (this is intentionally
different from a single bridge's own teardown, which only ever discards
its own round id).

## Defect B: payload retention guard, applied to all three bridges

All three bridges' `finally` blocks now guard the parked-payload pop with:
`if not still_armed_same_session or stored_payload_matches_this_round: pop`.

- `still_armed_same_session`: already computed by skill-install/skill-script
  (task-581/TASK-910); newly added to `request_mcp_approvals`, which had no
  such guard at all before this task (its payload pop AND its mounted-card
  clear were both unconditional).
- `stored_payload_matches_this_round`: new for all three -- reads the
  currently-stored payload under the session's key and compares its own
  `round_id`/`request_id` against this round's id, so an EARLIER round's
  teardown never blows away a NEWER round's payload that already overwrote
  the shared per-session slot (the `_parked_*_payloads` maps are keyed by
  session id alone, not by round id -- a structural single-slot limitation
  the task's fix shape works within rather than restructures).

Badge clearing was decoupled from payload clearing per the spec: each
bridge's `finally` now calls `discard_pending_round(session_id, round_id)`
unconditionally (this round's own id only), independent of whether the
payload was popped.

**MCP parity extension beyond the literal defect text:** the task's
"Apply symmetrically to all three bridges" instruction, plus the fact that
computing `still_armed_same_session` for MCP costs nothing extra, motivated
also gating MCP's mounted-card clear (`_marshal_pending_approval(None)`) on
`not still_armed_same_session` -- bringing it to parity with skill-install/
skill-script, which already had this guard. Without it, a second MCP round
for the same session resolving would blank the still-live sibling round's
visible card. Verified with
`test_two_mcp_rounds_for_the_same_session_the_earlier_ones_teardown_does_not_evict_the_newer_ones_payload`.

## Rider (mechanical, PascalCase)

- `tldw_chatbook/Workspaces/conversation_browser_state.py`: `class
  _RunMarkerBearer(Protocol)` -> `RunMarkerBearer`. Single internal use
  site (`_most_urgent_run_marker`'s type hint) updated; no external
  importers found (`grep -rn _RunMarkerBearer` across the repo returned
  only the definition + its one use, both fixed).
- `Tests/UI/test_skill_install_concurrent_confirms.py`: `class _FakeApp` ->
  `FakeApp`. Single use site (`ctrl.app = FakeApp()`) updated; no external
  importers.

## Per-bridge changes (file: `tldw_chatbook/Chat/console_chat_controller.py`)

- `request_mcp_approvals`: arm now calls `add_pending_round(session_id,
  round_id)` instead of `set_run_pending_approval(session_id, True)`.
  `finally` now computes `still_armed_same_session` (new), guards the
  `_parked_approval_payloads` pop with the OR-rule above (new), calls
  `discard_pending_round(session_id, round_id)` instead of `set_run_
  pending_approval(session_id, False)`, and gates the mounted-card clear on
  `not still_armed_same_session` (new).
- `request_skill_install_confirm`: same shape; arm uses `request_id`.
  `still_armed_same_session` already existed (task-581/TASK-910 pattern) --
  only the payload-pop OR-rule guard and the `discard_pending_round` swap
  are new.
- `request_skill_script_confirm`: identical shape/changes to skill-install.
- `_set_run_state`: terminal-transition clear changed from `.discard(target)`
  (set) to `.pop(target, None)` (dict), same semantic intent (drop
  everything for the session), adapted to the new type.

## Test evidence

Interpreter verified: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c "import tldw_chatbook; print(tldw_chatbook.__file__)"`
printed `/private/tmp/tldw-approval-acct/tldw_chatbook/__init__.py` (worktree,
not the main checkout) before any test run.

**Gate 1** (`Tests/UI/test_console_mcp_approval.py
Tests/UI/test_skill_install_concurrent_confirms.py
Tests/Chat/test_skill_script_concurrent_confirms.py
Tests/Chat/test_console_run_markers.py Tests/UI/test_console_parallel_runs.py`,
one foreground `pytest -q` call):

```
79 passed, 2 failed in 59.01s
```

The 2 failures are the two named pre-existing baseline failures:
- `test_batch_row_widgets_have_nonzero_geometry_and_do_not_overlap_under_bundled_css`
  (CSS-geometry batch-row test)
- `test_request_mcp_approvals_cancellation_records_denied_decision_to_execution_log`
  (mcp cancellation execution-log test)

Confirmed pre-existing and unrelated to this change: `git stash` (reverting
every change in this branch) then running just these two tests reproduced
the identical failures/assertions, then `git stash pop` restored the work.

**Gate 2** (`Tests/Chat/test_console_run_state_per_session.py
Tests/UI/test_console_skill_install_confirm.py`, one foreground call):

```
32 passed in 5.55s
```

**Additional sanity runs** (not part of the hard-rule gates, run for extra
confidence since they touch the same shared state):
- `Tests/Chat/test_console_skill_script_confirm.py`: 25 passed.
- `Tests/Workspaces/test_console_conversation_browser_state.py` (exercises
  `_most_urgent_run_marker`/`RunMarkerBearer`): 56 passed.

`ruff check` on all seven touched files: all checks passed.

## Touched assertions / mechanical updates

None of the four gated test files required signature changes to existing
tests -- every pre-existing assertion (including all `in`/`not in
controller._pending_approvals` membership checks) passed unchanged against
the new `dict[str, set[str]]` type, because dict `in`/`set(dict)` mirror
`set[str]` membership/iteration semantics exactly.

New tests added (all additive, no existing test modified):
- `Tests/Chat/test_console_run_markers.py`: 4 new tests (direct round-
  accounting seam tests: same-session multi-round survival, idempotent
  add/discard x2, legacy-shim composition).
- `Tests/UI/test_console_mcp_approval.py`: 2 new tests (cross-bridge
  MCP+skill-install same-session survival; same-bridge two-MCP-round
  payload-retention-under-overwrite).
- `Tests/UI/test_skill_install_concurrent_confirms.py`: 1 new test
  (same-session two-round badge/payload survival) + rider rename.
- `Tests/Chat/test_skill_script_concurrent_confirms.py`: 1 new test (same
  shape, skill-script bridge) + a session-scoped `_arm_for_session` helper
  (existing `_arm` intentionally stays session-agnostic for its own
  tests' cross-talk/teardown-isolation focus).

## Modified/added files

- `tldw_chatbook/Chat/console_chat_controller.py`
- `tldw_chatbook/UI/Screens/chat_screen.py`
- `tldw_chatbook/Workspaces/conversation_browser_state.py`
- `Tests/Chat/test_console_run_markers.py`
- `Tests/UI/test_console_mcp_approval.py`
- `Tests/UI/test_skill_install_concurrent_confirms.py`
- `Tests/Chat/test_skill_script_concurrent_confirms.py`

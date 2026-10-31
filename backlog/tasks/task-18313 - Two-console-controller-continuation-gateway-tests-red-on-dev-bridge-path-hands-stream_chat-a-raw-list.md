---
id: TASK-18313
title: >-
  Two Console controller continuation-gateway tests red on dev: the bridge path
  hands stream_chat a raw list (contract drift after TASK-16077's reconcile)
status: Done
assignee:
  - '[rmusser01]'
created_date: '2026-08-18 15:40'
labels:
  - testing
  - console
priority: medium
dependencies: []
---

## Description (the why)

Two pre-existing dev reds, attributed twice before filing (never this
branch's): PR 3b Task 5's landing report measured them failing identically at
untouched origin/dev `0e73851c4` in a throwaway worktree, and PR 3b Task 6
re-verified them failing at untouched dev `cf5db6f50` (clean tree):

- `Tests/Chat/test_console_chat_controller.py::test_controller_real_gateway_budgets_active_continuation_owner_atomically`
  — `AssertionError: assert ['old', 'old answer', 'current'] == ['current']`
  (the prepared payload carries the un-budgeted history rows the
  continuation sidecar was supposed to absorb).
- `Tests/Chat/test_console_chat_controller.py::test_controller_bridge_agent_service_bound_private_history_on_real_send`
  — `AttributeError: 'list' object has no attribute 'messages_payload'` at
  `gateway.prepared.messages_payload`.

Mechanism of the second (verified in the fixture): the shared
`ContinuationHistoryGateway.stream_chat` records `self.prepared = messages`
— whatever object it is handed. The direct-controller path hands it a
`PreparedProviderRequest` (test 1's attribute access succeeds); the
agent-bridge path hands it a raw message list. This is the same
"prepared requests as raw message lists" family TASK-16077 reconciled
(`347f20ca0`, 2026-08-13, whose close-out verified these exact assertions
green plus the 498-test Console agent/fleet gate on 2026-08-14) — so the
drift re-arrived on dev between 2026-08-14 and 2026-08-17 (inference from
the two dated measurements; not bisected).

Open question the fix must answer first: which side owns the contract — if
the production agent-bridge path genuinely dispatches raw lists where the
direct path dispatches `PreparedProviderRequest`, that is a production
contract divergence, not a fixture bug; if the bridge wraps correctly and
only the fake's recording seam drifted, it is test-only (16077's shape).

## Acceptance Criteria (the what)

- [x] Root cause identified with the commit that reintroduced the drift, and classified production vs test-only
- [x] Both named tests pass on dev exercising the current real contracts (no assertion weakened to pass)
- [x] If the bridge and direct paths genuinely dispatch different shapes to `stream_chat`, the divergence is either unified or pinned deliberately with the reason in-line

## Implementation Plan

1. Verify the premise at branch base `ecc0a531c8`: run both named tests; compare failure
   signature against the filed one (payload/budget assertions vs anything earlier).
2. Bisect the original drift: last-green `1f1439cb53` (2026-08-12 22:51) to the filed
   measurements; bisect the re-fix forward from the last-bad commit to current dev.
3. Classify the CURRENT red per the session taxonomy (fixed upstream / live defect /
   config-admission mask / hook-consent send-gate).
4. Apply the established fix for whatever class remains live at this tip; targeted
   runs only; no assertion changes.
5. Close out with classification, commands, results, ADR check.

## Implementation Notes

**Classification: the filed contract drift is FIXED UPSTREAM (same day this task
was filed); the red that remains at dev tip `ecc0a531c8` is the config-admission
mask class, fixed here by per-node enrollment.**

- **The drift was real and was a production defect.** Bisect (fresh 3.12 venv,
  `-p no:randomly`, both named tests):
  - Last green: `1f1439cb53` (2026-08-12 22:51, "preserve continuation budgeting
    policy") — 2 passed.
  - Bridge raw-list reintroduction: `8153579078` (2026-08-13 10:39, "close
    continuation sync review gaps") — first commit where
    `test_controller_bridge_agent_service_bound_private_history_on_real_send`
    fails (`AttributeError: 'list' object has no attribute 'messages_payload'`);
    the budget test still passed there. That commit added the adapter's
    continuation-sidecar plumbing whose `prepare_chat_request` wrap did not yet
    cover the bridge dispatch path.
  - Direct-path budget reintroduction: entered dev at merge `75c0cd3545`
    (2026-08-14 02:17, PR #1630 codex/uat-first-chat-remediation; branch rooted
    at `37b560bc34` "derive provider discovery routes safely", which forked
    before the 08-12 budget fix) — first FAILING first-parent commit; last
    passing first-parent: `ddc8316de9` (2026-08-13 23:08). PR #1630 touched
    `console_provider_gateway.py` (+95) among 94 files. Note this corrects the
    task's inferred window ("between 08-14 and 08-17") to 08-13/08-14, and
    TASK-16077's close-out claim does not hold at its own reconcile commit
    `347f20ca07` in a fresh environment (red there and at child `fcd3c67d961`).
  - **Re-fix: `58200d6f8d` (2026-08-18 16:55, "repair 16 pre-existing red tests,
    incl. broken continuation resume") — 2 passed; parent `8dd97f9e07`
    (08-18 15:52) fails with both filed signatures.** Production fix:
    `_continuation_restore_target_for_resolution` normalized `base_url` through
    `normalize_generic_endpoint_for_compare` (expanding it), so
    `validate_continuation_restore`'s byte-exact compare rejected every restore
    and the continuation budgeting/prepared-request paths degenerated. Carrying
    `api_base_url` verbatim fixed 11 red tests including both named here.
    **This landed 75 minutes after this task was filed (08-18 15:40)** — the
    task's premise expired same day.
- **The current red at `ecc0a531c8` is NOT the drift.** Both tests fail earlier
  at `assert result.accepted` with `visible_copy='Hooks unavailable; review or
  disable hooks before sending.'`: `submit_draft`'s hook admission gate calls
  `read_hooks_config_snapshot()`, which under the per-test sandbox redirect
  fails closed with `RecoveryRequired("raw_source_selection_changed")`
  (ADR-126 admission) and the gate's broad `except` converts that into a
  refusal. This is the config-admission mask class, not the hook-consent
  send-gate product behavior (TASK-33621.x): with the collection-time profile
  kept, the gate returns None and admits the send.
- **AC#3 answer: no divergence exists today.** The direct path and the agent
  bridge path both dispatch prepared requests — the current bridge wraps every
  dispatch branch (`trace_request_factory`, `continuation_groups`,
  thinking/continuation sidecars) through `prepare_chat_request` /
  `build_console_request` (`console_agent_bridge.py`, `_consume` in
  `_StreamingModelAdapter`), and both tests' `gateway.prepared.messages_payload`
  assertions pass unmodified.
- **Fix shipped:** `@pytest.mark.bootstrap_profile` on the two named tests
  (`Tests/Chat/test_console_chat_controller.py`), the TASK-32873 per-node
  pattern for mostly-sandbox suites whose send path reads the guarded config
  loader. No assertion, fixture or production change.
- **Evidence (targeted runs, worktree `.worktrees/test-console-reds`):**
  - Base `ecc0a531c8`: both tests `2 failed` ("Hooks unavailable" refusal).
  - Scratch-plugin probe (marker only, no edit): `2 passed`.
  - After in-repo markers: `2 passed, 12 warnings`.
  - A/B via `git checkout HEAD -- Tests/Chat/test_console_chat_controller.py`:
    base file `2 failed`; re-applied markers `2 passed`.
  - Combined `-k` slice with default random ordering: the two marked tests
    pass; unmasked neighbors
    (`test_controller_direct_replays_live_session_thinking_policy`,
    `test_provider_switch_ignores_unrelated_completed_continuation_history`,
    `test_controller_agent_replays_same_live_session_thinking_policy`) fail
    with the IDENTICAL `Hooks unavailable` mask signature — known red mass in
    this suite, mapped here for its owner (same enrollment class; not this
    task's ACs).
- **Modified files:** `Tests/Chat/test_console_chat_controller.py` (two
  markers + comments), this task file.
- ADR required: no — applied the existing per-node `bootstrap_profile`
  enrollment mechanism (ADR-126 admission machinery, TASK-32873 precedent) to
  two test nodes; no architectural decision made.

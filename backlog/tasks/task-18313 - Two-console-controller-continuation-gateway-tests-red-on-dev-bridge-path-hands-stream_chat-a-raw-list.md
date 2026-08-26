---
id: TASK-18313
title: >-
  Two Console controller continuation-gateway tests red on dev: the bridge path
  hands stream_chat a raw list (contract drift after TASK-16077's reconcile)
status: To Do
assignee: []
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

- [ ] Root cause identified with the commit that reintroduced the drift, and classified production vs test-only
- [ ] Both named tests pass on dev exercising the current real contracts (no assertion weakened to pass)
- [ ] If the bridge and direct paths genuinely dispatch different shapes to `stream_chat`, the divergence is either unified or pinned deliberately with the reason in-line

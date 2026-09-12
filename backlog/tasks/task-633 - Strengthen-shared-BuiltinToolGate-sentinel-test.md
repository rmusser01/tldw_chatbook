---
id: TASK-633
title: Strengthen the shared-BuiltinToolGate sentinel test with a call-counting factory
status: Done
assignee:
  - '@zcode'
created_date: '2026-07-25'
labels: [tests, tools, tech-debt]
dependencies: [TASK-545]
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`test_review_hook_and_run_reply_share_one_builtin_gate` (`Tests/Chat/test_console_agent_swap.py:1116`) is the regression test for TASK-545/P1's headline invariant: a single `BuiltinToolGate` instance must be shared between the run-level review hook and `BuiltinToolProvider`'s `run_reply`, so that a stamp set by the hook is actually the same object `invoke()` checks. It monkeypatches `build_builtin_gate` with `lambda service=None: sentinel` — a **constant-returning stub** that always hands back the exact same pre-built `_SentinelBuiltinGate()` instance no matter how many times it is called.

That stub cannot distinguish the invariant it claims to test ("the codebase calls `build_builtin_gate` once per run and threads the single result to both consumers") from a weaker, buggy alternative ("the codebase calls `build_builtin_gate` twice — once for the hook, once for `run_reply` — and both calls happen to return the same object only because the stub ignores its own call count and always returns the same sentinel"). A future refactor that regresses to building two separate gates (which would break real stamp-sharing in production, since the real `build_builtin_gate` builds a fresh `BuiltinToolGate` per call) would still pass this test unchanged, because the stub papers over the difference.

A call-counting factory — one that raises, or returns a distinguishable second sentinel, on any call beyond the first — would be strictly stronger: it fails if `build_builtin_gate` is ever invoked more than once for a single run, in addition to still proving both consumers observe the same instance.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] `test_review_hook_and_run_reply_share_one_builtin_gate` (or a new test alongside it) uses a factory that tracks/limits its call count, not a constant-returning lambda
- [x] The strengthened test fails if a hypothetical regression calls `build_builtin_gate` more than once for a single run (verified by temporarily reintroducing such a regression locally and confirming the test catches it, per this repo's TDD practice)
- [x] The existing assertions (the hook and `run_reply` both observe the identical gate instance) still pass and are preserved
<!-- AC:END -->

## Implementation Plan

1. Replace the constant-returning lambda in ``test_review_hook_and_run_reply_share_one_builtin_gate`` with a call-counting factory that raises on any call beyond the first, and assert exactly one call per run.
2. Mutation evidence per the AC: temporarily duplicate the production ``build_builtin_gate`` call in ``console_chat_controller`` (two builds per run), confirm the strengthened test fails with the factory's AssertionError, then revert the mutation.
3. Preserve every existing assertion (identity of the gate in both consumers, begin_turn threading).

ADR required: no
ADR path: N/A
Reason: Test-only strengthening; no production change.

## Implementation Notes

The sentinel test's stub is now ``_single_call_factory``: it records each ``build_builtin_gate`` invocation and raises ``AssertionError`` on a second call within one run, plus a direct ``assert len(factory_calls) == 1``. All original assertions are preserved unchanged (the ``run_reply`` kwarg IS the sentinel; the captured review hook's ``begin_turn`` threads to it).

Mutation evidence (AC#2): duplicating the production build (``console_chat_controller.py`` ~:14664, second ``build_builtin_gate`` call per run) made the strengthened test fail at the factory's guard -- the old constant lambda would have passed that mutation unchanged, which was exactly the blind spot. Mutation reverted immediately; production diff is empty.

Verification: the strengthened test passes on unmutated dev; the full ``test_console_agent_swap.py`` file shows one unrelated failure (``test_run_error_via_regenerate_preserves_original_answer_and_status``) that reproduces identically with this change stashed -- pre-existing. Ruff format clean.

Modified: ``Tests/Chat/test_console_agent_swap.py``.

# Phase 1.7 Narrow Core-Loop Proof

Date: 2026-05-05
Task: TASK-8.7
Branch: codex/product-maturity-phase1-7-core-loop
Workflow: Search/RAG result -> Console staged context

## What Was Verified

Phase 1.7 narrow core-loop proof verifies the remaining Phase 1 gate: a source-like Search/RAG result can reach Console as staged context with visible source authority and a draft prompt.

Verified path:

- Source producer: Search/RAG result payload.
- Console entry: app-owned `open_chat_with_handoff` route.
- Console state: ephemeral chat session with a visible staged-context card.
- Source authority: local.
- Recovery posture: the context is staged rather than auto-sent, and the user can review the draft before sending.

## Automated Evidence

Focused regression:

```bash
../../.venv/bin/python -m pytest Tests/UI/test_product_maturity_phase1_core_loop.py -q
```

Initial red run:

- The running-app staged-context assertion passed after the test pinned only the relevant chat-tab settings.
- The evidence assertion failed because Phase 1.7 evidence, tracker, README, and task closeout updates did not exist yet.

Final focused verification:

- `Tests/UI/test_product_maturity_phase1_core_loop.py -q`: 2 passed, 1 warning in 6.20s.
- Phase 1 product-maturity sweep including Phase 1.1-1.7 tests: 42 passed, 1 warning in 21.79s.
- Adjacent Search/RAG and Chat handoff regressions: 15 passed, 1 warning in 5.89s.

## Manual Walkthrough

The Textual pilot exercised the running app with chat tabs enabled, started from the shell, invoked the app-owned Search/RAG handoff path, navigated to Console, and verified the staged-context card.

Observed user-facing Console state:

- The card states `Context staged from RAG Search`.
- The card shows the result title and `Type: rag-result`.
- The card shows `Backend: local`.
- The card shows `Source: Local source`.
- The card includes the result score metadata.
- The composer is preloaded with the suggested prompt so the user can review before sending.

This is intentionally a narrow Phase 1 proof. It does not prove full grounded generation, artifact persistence, or Chatbook reopen. Those belong to Phase 2.

## Visual/Focus Notes

- The staged-context card is visible before the draft is sent.
- The user remains in control because context is not automatically submitted.
- Source authority is visible in the card rather than hidden in internal payload metadata.

## Defects Found

No P0/P1 defects found.

One test-harness issue was corrected before closeout: the initial test patch returned `True` for unrelated settings and corrupted splash/navigation setup. The final test pins only splash disablement and chat-tab enablement.

## Residual Risk

Phase 1 is now complete as a QA baseline. Remaining product-depth risks move to Phase 2:

- grounded answer generation with live provider/runtime configuration;
- save/update Artifact or Chatbook;
- reopen/resume from Artifacts or Home;
- missing-source and failed-generation recovery in the full loop.

## Exit Decision

Pass for Phase 1.7 and Phase 1. The app has verified first-run, navigation, keyboard/focus, visual/chrome, empty/error/setup, and narrow core-loop guardrails. Phase 2 can now focus on completing the full source/question -> grounded Console -> Artifact/Chatbook loop.

---
id: TASK-34350
title: >-
  Console: a send over the context budget should prompt to compact,
  auto-compact, or alert, instead of dead-ending
status: Done
assignee:
  - '@claude'
created_date: '2026-10-03 18:37'
updated_date: '2026-10-04 12:16'
labels:
  - console
  - context-compaction
  - follow-up-33940
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up to TASK-33940.5, which removed a stale 4,096-token fallback after it made the shipped default model refuse every send in a fresh profile. That fix removed the bad number but left the underlying policy unchanged: a send that does not fit is stopped, and the user is not given a way forward at the point where it stopped.

On dev `01a2020981` the Console's preflight (`_apply_conversation_memory_preflight` in `Chat/console_chat_controller.py`, deciding via `decide_compaction` in `Chat/console_context_compaction.py`) behaves like this:

- **Compaction mode Ask (the default).** When the conversation reaches the compaction threshold, the send is blocked with a failure row reading "Conversation context reached its compaction threshold. Review and approve compaction before sending again." Nothing on that row or the composer lets the user approve anything. The only Compact now control is in Conversation settings ▸ Context and memory.
- **Compaction mode Automatic.** The preflight compacts and then sends.
- **Compaction cannot help** (no complete older turns to summarise, or the response reservation plus safety margin plus mandatory input already exceed the window). The send is refused with "This request cannot fit the selected model. …" This is also what fires when the window is only an *estimate*: an unverified window is treated as hard fact, as the 4,096 fallback showed.

**Owner ruling (2026-10-03):** a send that exceeds the budget should **prompt the user to compact**, or **trigger an automatic compaction if that is enabled**, or **alert the user if the conversation is too large to compact**.

Compaction mode Off keeps its current behaviour; it is not part of this ruling.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 With compaction mode Ask, a send that reaches the compaction threshold shows a prompt at the send surface offering to compact and send, or cancel; choosing compact runs the same compaction as Compact now and then sends the held message without the user retyping it
- [x] #2 Cancelling that prompt sends nothing and compacts nothing, and the message goes back to the composer (the hold happens before anything is committed)
- [x] #3 With compaction mode Automatic, such a send compacts and sends without a prompt (current behaviour, pinned by a test)
- [x] #4 When compaction cannot make enough room, the user gets an alert that names the limiting reason and the specific setting to change (response maximum, mandatory context, or model context window); the send is refused before commit, so no dispatch-recovery panel buries the alert, and the message stays recoverable
- [x] #5 An estimated (unverified) context window takes the same prompt / automatic / alert route as a verified one, and the alert says when the window is an estimate and where to set the real value
- [x] #6 Tests drive the real controller through the prompt, automatic and alert outcomes, including one on an estimated window; the prompt and estimated-window tests fail on dev `01a2020981`
- [x] #7 The prompt and the alert are verified in the running app against a real provider, and `Docs/User_Guide/console/context-and-rag.md` describes the new behaviour
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pure copy module (console_context_budget_copy): alert per overflow cause (reservation fills window / mandatory context / nothing to compact) naming Max tokens and, for an estimated window, F4 Settings > Providers & Models; Ask-mode prompt copy. Unit tests first.
2. Preflight: classify the overflow cause from the real capacity and resolved policy; route the alert copy; keep Automatic unchanged (existing parametrized test pins it).
3. Ask mode: replace the dead-end block with an actionable prompt on the existing recovery-card pattern (Compact and send runs Compact now, then resends the held turn in place; Cancel leaves it unsent and resendable).
4. Route tests on the real preflight incl. estimated windows (private profiles).
5. Live-verify the prompt and the alert in the running app with a real provider; update Docs/User_Guide/console/context-and-rag.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Owner ruling 2026-10-03: a send over budget prompts to compact, auto-compacts when that is enabled, or alerts when compacting cannot make room. Scope agreed 2026-10-04 ("Real hold, core"): a proper policy hold now, and the rest of TASK-33621.4 (summarize preview, header/Inspect/chip warnings, /compact and palette, F1) stays on 33621.4.

Design: the hold happens before commit. In submit_draft, after the request is assembled and before any hook fires, _assess_context_compaction runs the real preflight as a side-effect-free probe (an assessment_sink hands back the decision, token numbers and alert copy). On Ask the preparation pauses with the new ConsolePreparationPauseKind.CONTEXT_COMPACTION (two new legal transitions: COMMITTING->PAUSED, PAUSED->READY). Nothing is persisted, nothing is marked Failed, and no dispatch checkpoint exists, so the 'Response accepted; waiting for dispatch.' panel of TASK-33621.4 cannot appear (33621.4 AC2/AC6 for the composer path). Actions: compact_and_send (Compact now, then resume the exact send; it never asks twice for one send), send_without_compacting (resume; a one-shot flag passed explicitly into the stream, because the durable path gives the stream no preparation id), and cancel (drop the held send; its text goes back to the composer). ConsoleRuntime treats a compaction hold as held, not refused, so no duplicate turn recovery is recorded. A held send's optimistic echo is skipped by _durable_context_snapshots, so Compact now and the compaction admission work while a send is held.

UI: a third mode on the existing pre-dispatch card (TraceCallRecoveryCallout), with neutral copy (numbers plus what each choice does, no Problem/Impact framing) and buttons Compact and send / Send without compacting / Cancel send. It adds zero new CSS: boot-CSS headroom was 139 bytes.

Alert: console_context_budget_copy names the cause (Max tokens plus margin fill the window; system prompt, tools and attached context exceed it; nothing to compact) and the setting to change, and says when the window is an estimate (F4 Settings > Providers & Models). One helper (_context_overflow_alert) serves the stream block and the probe, so they cannot disagree. Live verification showed the alert buried under the dispatch-recovery panel, so the cannot-fit refusal now also happens before commit (a capacity-only probe covers a durable chat's first message); the message stays in the runtime's 'Unsent turn needs attention' shelf with Restore. Sends that cannot show the card (queued, Retry, Regenerate, Buddy) keep the post-commit block, with actionable copy. The hold names an estimate only when the window actually sets the budget (found live: a custom 1,500 budget was wrongly called estimated).

Evidence:
- Tests/Chat/test_console_compaction_ask_hold.py (11; live-session harness plus runtime-owned controller; the hold was RED on dev as 'Accepted turn is retained for recovery.').
- Tests/Chat/test_console_context_budget_routes.py (8, private profiles).
- Tests/Chat/test_console_context_budget_copy.py (16).
- Tests/UI/test_console_trace_call_recovery.py (+3).
- Tests/UI/test_console_compaction_hold_flow.py (3; the real app, ChatScreen and runtime with only the adapter doubled; private profile per case; capture off by its real setting, because Capture's back-to-back trace timing intermittently paused earlier sends as TRACE_CALL under load, reproduced on neither side in 6 isolated probes; 9/9 green across three full-file runs; mutation-checked).
- Frozen matrices in test_console_turn_preparation.py updated; test_console_context_compaction.py's old copy assertion moved to the new alert.
- Live, real app + gpt-5.6-terra (shipped default) in an isolated profile with a custom 1,500-token budget: the hold card showed 1,413 of 1,500 tokens; Compact and send summarized ('Earlier turns summarized for context') and sent the held message once; Send without compacting sent it as is; Cancel put it back in the composer. With Max tokens 32,000 on the estimated 32,000 window, the alert named Max tokens and the estimate; after the before-commit fix there is no dispatch panel and Restore returned the message.
- Docs/User_Guide/console/context-and-rag.md: new 'When a chat reaches its context limit'.

Qodo round (PR #3003):
- The check now runs early (before skills/retrieval/hooks), and again before commit only if the request grew. A late hold resumes without re-appending notes or retrieval events and reuses its capture result.
- Send without compacting also holds when the policy turned Automatic.
- The check is skipped when compaction is Off. Measured probe cost: ~7 ms at 40 turns, 14-24 ms at 120 turns (~30k tokens), against a ~500 ms send.
- A cancelled or abandoned resumed hold drops its answered id.
- Not changed: an uncompacted over-ceiling send is windowed by the real preparation, as compaction Off is; this is pinned by a test.
- ADR-097 UI-ready census kept flat by importing the copy module lazily.
Tests added: hook context refused before commit (real RunHooksEngine, mutation-checked); late hold after hooks resumes without repeats (window measured with the real probe); Automatic-meanwhile; abandoned resume; over-ceiling windowing.

Regression (2026-10-04): 64 suites (every touched file plus the staged-evidence / queue-custody / dispatch-recovery / compaction / trace-recovery suites) compared against a clean worktree at the PR's dev base df2ba424de: 1,253 failures shared. Every difference was classified by isolated reruns. In test_console_automatic_library_preparation.py all 16 branch-only entries behave identically on both sides run one per process (10 hang on both, 6 fail on both under the local RecoveryRequired trap). With that trap bypassed by the sanctioned bootstrap profile, all 13 evidence / staged / lease / recovery tests in the file pass on both sides. Four other differences (two speculative-voice timing tests, one live-work handoff, one session-settings test) pass 3/3 individually on both sides, apart from one voice-test flake on each. Net: no branch-only regressions.
<!-- SECTION:NOTES:END -->

---
id: TASK-36610
title: >-
  Console: a send's start window can end early or never open, and its
  Inspector-visible path is unmeasured
status: In Progress
assignee:
  - '@claude'
created_date: '2026-10-24 07:50'
updated_date: '2026-10-31 09:10'
labels:
  - console
  - performance
  - ux-review-2026-09-29
dependencies:
  - TASK-33620.15.4
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-33620.15.4 proposes keeping three Console display reads on the UI loop during a send's start, from acceptance to the provider request and 3 s at most: the Library activity rows and the Inspector's registry and appearance reads. Off the loop, each read opened its own SQLite connection and delayed first sends by ~66 ms. That exception still needs the owner's approval. If the owner rejects it, this task is void.

Its review found gaps in that window. All of them cost time only; none shows wrong data:
- The provider-request mark is process-wide, so another session's provider request ends this send's start early, and its reads go off the loop for the rest of the start.
- Queued-prompt dispatch and automatic submissions never note acceptance, so their reads go off the loop during their start.
- If acceptance is ever noted after the provider request, only the 3 s limit ends the window.

The live measurement (2026-10-23, Anthropic claude-haiku-4-5, 160x45, dev 1381e70735 as the base) ran with the Inspector and Context rails collapsed. The Inspector-visible path, whose reads the window exists for, was not measured. First sends alone showed no detectable difference (16 per arm: paired median +1.1 ms, mean +9.7 ms, branch faster in 8/16). That is weaker than a proven "no later".
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A send's start ends at its own provider request, not at another session's
- [x] #2 Queued-prompt dispatches and automatic submissions open a send's start at their acceptance, or measured evidence shows they need not
- [x] #3 With the Inspector rail open, a send reaches the provider no later than on the base build (Enter to the provider request, live with a real provider, paired and interleaved, at least 30 sends per arm)
- [ ] #4 First sends alone reach the provider no later than on the base build (at least 30 first sends per arm, paired and interleaved, median and paired difference)
- [x] #5 Deterministic tests fail on the base build for #1 and #2
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Measure first on dev (live, probe, Inspector open): does the send-start window open at acceptance and close at the provider request; are display reads off the loop inside it.
2. Tests first: a mounted two-session test (another session's provider request must not end this send's start), a queued-prompt dispatch test, agent wake and goal iteration acceptance tests, and an agent chat-start test; each must fail on dev for the reason it pins.
3. Give each send its own start: a context-carried SendStart opened at the runtime's custody-task creation (every acceptance, any origin) and around an agent chat start's submission, ended by the provider request its own work makes, its task's end, or the 3 s limit; the Console reads the starts open on its store.
4. Remove the screen-level acceptance note and the process-wide provider-request mark.
5. Live, paired and interleaved against dev with the pf2probe kit (relay timing outside the app): Inspector open, >=30 sends per arm (AC#3); first sends, >=30 per arm (AC#4).
6. Rebase onto newest dev, gates by test id against dev, preflight, notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Each Console send now owns its start window, from its own acceptance to its own provider request (3 s at most). Another session's provider request no longer ends it. A queued prompt's dispatch, an agent wake, a goal iteration and an agent's chat start each open one. Live and paired against dev, sends with the Inspector open reach the provider no later (64 pairs: paired median -1.4 ms). First sends show no detectable difference (60 pairs: paired median +15.9 ms, 95% CI -7.6 to +38.4; mean -6.1 ms), which does not meet AC#4 as written.

**AC#4: open, for the owner's decision (2026-10-31).** AC#4 asks for first sends that reach the provider "no later than on the base build", judged by the median and the paired difference. Both lean the other way: the head's median is 1395.1 ms against 1377.9 ms on the base, and the paired median is +15.9 ms (95% bootstrap -7.6 to +38.4). Only the mean favours the head (-6.1 ms, -56.1 to +26.0). The tick was withdrawn after review, and the task stays In Progress. Running more sets would not settle it on its own: if the true difference is about zero, the interval narrows around zero but still spans it, so "no later" cannot be shown without a margin. The owner can choose one of two ways forward:
- Accept equivalence within the measured upper bound (about +38 ms on first sends) and reword AC#4 with a dated sign-off.
- Name a margin and have more interleaved first-send sets run until the paired interval falls inside it.

The mechanism evidence gives no code-path reason for a lean. In a probe set (6 first sends per arm, Inspector collapsed), the window opened about 261 ms after Enter in both arms and closed after the request. Neither arm started an off-loop display read in a send's start (0 of 6 each). UI-loop callback CPU from acceptance to the request was 357 ms on the base against 366 ms on the head. The app's stage log puts the remaining spread in the agent run start, which is worker-thread work this change does not touch. AC#3 stays ticked on the review's reading: its medians and its paired median favour the head, though its interval spans zero too (-17.3 to +18.7).

**Measured first (dev 7536ac3b87, live, probe, Inspector open, 2026-10-30).** The window opened at the first projection after acceptance (~255 ms after Enter) and closed at the first query after the provider request. It started no off-loop display reads in 4 of 4 sends at load 3-6. At load 11-49, every Enter-to-provider time ran past 3 s (5.1-6.4 s). The 3 s limit ended the window first, and the Library-activity and appearance re-reads started inside the send's start. The approved limit is unchanged here; this is recorded as context only.

**Changes**
- `Chat/console_send_start.py` (new): a `SendStart` per accepted send, carried in a context variable. Tasks and `asyncio.to_thread` workers created inside a send copy it. So the provider request that send's own work makes ends that start and no other (`note_provider_request`, called from `record_send_stage("provider_entry")`). Starts are kept per Console store in a `WeakKeyDictionary`.
- `ConsoleRuntime._create_custody_task` (both acceptance paths, every origin: the composer's sends, queued dispatch, wakes, goal iterations) opens the start as it creates the turn's task. The task's end also ends it. Net 0 lines in `console_runtime.py`, so its ratchet row is untouched.
- `ConsoleChatStartCoordinator._run`: an agent's chat start submits straight to the controller, so it opens its start around `submit_draft` (`submission_send_start`). The start ends with the submission. It opens at the submission, just before the coordinator's acceptance inside it.
- `config_sync.send_start_in_progress` reads the starts open on the screen's store. `note_send_accepted`, the screen's `_console_send_accepted_at` and the process-wide `provider_entered_at` are gone (wiring.py loses both calls).
- `Chat/console_first_send_modules.py`: the new module joins the post-UI-ready first-send pre-import. Without it, `test_console_first_send_preimport.py`'s harness-send guard sees it imported on the UI loop.
- `scripts/ui_pr_gate_census.txt`: the new test file joins the tail (7 cases, about 26 s at load 6).

**Tests** (`Tests/UI/test_console_send_start_window.py`, plain command, bootstrap profile):
- AC#1: two sessions, A's send and then B's are accepted and held just before their provider requests. A's real request must leave B's start open, and B's own request ends it. On dev it fails: "another session's provider request ended this send's start".
- AC#2: a queued prompt dispatched by the real queue after a run, an agent wake and a goal iteration through the runtime's own submitters, and an agent chat start on a real controller (screen-free). Each fails on dev with "... opened no send start".
- A pin that the real gateway's adapter entry, on its worker thread, ends the stock received send's start (passes on dev too), and the limit case. These are the quiet-gap file's three send-start tests, moved here and adapted. The workspace display-reads test opens its start the new way.
- Mutation checks on the head: making any provider request end every start fails the AC#1 test. Reverting the runtime's binding fails the queued, wake and goal tests. Dropping the chat-start binding fails that test.

**Live (AC#3, AC#4)**
- Set-up: real app, isolated profiles, null keyring, 160x45, Anthropic claude-haiku-4-5 through the review proxy. A timing relay on 127.0.0.1:18610 logged each POST's arrival, so Enter to the provider request was timed outside the app, with no probe.
- Rounds alternated base and head (ABBA). A round started only at load1 < 12. The driver waited for each send's request at the relay, then 8 s more.
- Base: dev 7536ac3b87. Head: ddf63754d2 for set A (c5b54f627d after the rebase); ff44878699 for sets B-E (the final code with the pre-import; 46afa9aa9e after the rebase).
- An earlier attempt at load ~100 was aborted and discarded (another agent's load generator).

| set | sends per arm | dev median | head median | paired median (95% bootstrap) | paired mean | head faster |
|---|---|---|---|---|---|---|
| A: Inspector open, 8 launches x 4 | 32 | 1077.5 | 1058.2 | +1.8 (-19.5..+30.0) | +66.5 | 15/32 |
| C: Inspector open, 8 launches x 4 | 32 | 1002.2 | 1032.8 | -8.1 (-34.6..+22.2) | -63.2 | 17/32 |
| **A+C (AC#3)** | **64** | 1063.6 | 1045.5 | **-1.4 (-17.3..+18.7)** | +1.6 | 32/64 |
| B: first sends, Inspector collapsed | 30 | 1391.6 | 1415.7 | +34.0 (-9.4..+59.1) | +16.3 | 12/30 |
| E: first sends, Inspector collapsed | 30 | 1348.2 | 1387.7 | -0.7 (-12.5..+30.2) | -28.5 | 15/30 |
| **B+E (AC#4)** | **60** | 1377.9 | 1395.1 | **+15.9 (-7.6..+38.4)** | -6.1 | 27/60 |
| all first sends (A, B, C, E) | 76 | 1393.6 | 1405.2 | +13.1 (-8.0..+31.8) | -26.8 | 35/76 |

All times are ms. Every send's request reached the relay, and every response was 200. Host load1 at Enter was 1.6-9.1.
- Set A's mean comes from one head launch (R3) whose last two sends took 2.4 and 2.3 s. Every stage in that launch ran about 2.4x slower (provider resolution 184 ms against 68), which points to the host.
- Set C's mean comes from a 3.35 s dev first send.
- First sends: B leaned to the head being slower (+34 ms). The independent repeat E did not (-0.7 ms, 15/30). Pooled, the paired median is +15.9 ms and the mean -6.1 ms, with both 95% intervals spanning zero: no detectable difference, which is not the "no later" AC#4 asks for. The probe set and the stage log are under "AC#4: open" above.
- The intervals are the archived report's (a seeded bootstrap, 4000 resamples). An earlier draft of these notes quoted intervals from a different bootstrap run.

**Gates on the rebased head (dev 804dc00231)**, one file per command, compared with a detached dev worktree by test id:
- `test_console_send_start_window.py`: 7 passed on the head. On dev 804dc00231 it fails 6 of 7, each for the reason it pins. The received-path pin passes on dev, and the limit case's module does not exist there.
- Identical results by test id on both: quiet gap (4 passed on the head, 7 on dev; 3 cases moved), workspace display reads 7, first-send preimport 10, Library activity off-loop 7, projection reads 5, received-intent feedback 7, send acknowledgement 9, runtime lifetime 35, received custody 24, goal dispatch 16, fleet wake 34, tab-sync source ownership 24, UI-ready module census 4, boot CSS budget 1, screen-size ratchet 4.
- `test_console_send_diagnostics.py`: the same 11 environmental `RecoveryRequired` setup errors and 1 failure on both.
- Module-size ratchet: the same 8 dev reds on both, none in a touched file. `console_runtime.py` stays at its row (6686), and `console_chat_controller.py` is untouched.
- Pre-rebase, on dev 7536ac3b87, these were identical by test id too: chat start 171, prompt-queue coordinator 38, Chat prompt queue 46, received-intent saved acceptance 8 and transitions 7, automatic wake dispatch 9, first-send atomicity 29, turn execution context 74, config-sync cadence 6, poll reconciliation 6, the headless-wake (11), dispatch-recovery round 3 (7) and durable-turn round 1 (1) reds, parallel runs (21 reds) and turn-navigation continuity (16 reds).
- `Tests/UI/test_console_prompt_queue.py` differs by test id only on two shelf cases (`..._fit_terminal[size2]`, `..._keeps_retry_on_screen[size2-False]`). Rerun three times each, dev failed them in 2 of 3 runs and the head in 1 of 3: a timing flake on both.
- Census checker (1164 entries, floor 1163) and the UI-lane shard and census CI tests pass. ruff is clean on the changed files; `console_chat_start.py` has dev's own F841. `./scripts/preflight.sh` passes.

**Review follow-ups (2026-10-31)**
- The chat-start test now also pins that the chat start's own provider request ends its start. Its stream notes a provider entry where the real gateway does, on the agent loop's thread, and reads the window either side: open before, closed after. Before this, making `note_provider_request` a no-op left that test green. Now it fails with `[(True, True)]`.
- `note_provider_request` now documents that a send's first adapter entry ends its start. That entry can be an auxiliary request (an automatic compaction's summary, a response rule's evaluation). The base build ended the start at the same request. Ending it at the main request instead is TASK-38461.
- Some send entry points still open no start, the same as on the base: resumed paused preparations, the buddy conversation, the mobile server, and the fleet-wake and goal-run controller fallbacks used when no runtime submitter is bound. These are TASK-38460.
- Declined: making `send_start_in_progress` read the screen's memoised runtime instead of `_console_chat_store`, which can resolve one. Every caller has already resolved the runtime. Both Library paths call `ensure_store()` before `live_reads()`. The workspace guard returns before the predicate unless the run poll is ticking, and the poll runs only while the screen's controller, a runtime handle, exists. So the predicate is never the first to resolve a runtime, and reading the private memo would add a new coupling.
- AC#3 pools two head builds: set A ran without the pre-import commit, and set C ran on the final code. The head-side window probe ran only with the Inspector collapsed, so with the Inspector open the head was measured for timing only.
- The plan commit landed after the first two fix commits. The history was left as it is.

Evidence: kept on the measuring host, outside the repo, at `~/.claude/projects/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/c28a6a75-4221-4434-989d-fbf24a73c96b/tools/evidence/bd5/P10/` (driver and relay logs, probe captures, REPORT.txt, and the kit that produced them). The diagnostic launch's requests went through another agent's relay on 8797, which had taken the port; see the new lesson in lessons-live-verification.md.
<!-- SECTION:NOTES:END -->

---
id: TASK-42345
title: >-
  A Console session reopened under the id of a closed session stays fenced until
  the app restarts
status: In Progress
assignee: []
created_date: '2026-11-22T16:37:00Z'
updated_date: '2026-11-24T19:54:53Z'
labels:
  - follow-up
  - console
  - bug
dependencies: []
references:
  - tldw_chatbook/Chat/console_goal_runs.py
  - tldw_chatbook/Chat/console_chat_store.py
  - tldw_chatbook/Chat/console_runtime.py
  - tldw_chatbook/Chat/console_chat_controller.py
  - tldw_chatbook/Web_Server/mobile_service.py
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the review of the PR #3493 port (TASK-34975, round 20). Read at 59cc99e61b (dev 8a55555798 plus the round-20 stack); line numbers are that tree's. Found by reading; not reproduced.

**Why it matters.** A user who closes a goal run's chat and then resumes the paused or ready goal gets a chat that refuses to send or close until the app restarts.

**Evidence.**
- A session's id is chosen in `ConsoleChatStore.create_session` (`console_chat_store.py:2319`): a fresh `uuid4()` unless the caller passes one (`:2410`). Saved-conversation hydration reaches it through `restore_persisted_session` (`:3130-3131`), passing on whatever `session_id` its caller gave.
- The goal-run resume passes the conversation id as the session id: `_resume` (`console_goal_runs.py:622-633`, paused or ready goals only) calls `restore_session`, which hydrates with `session_id=goal.conversation_id` (`:537-546`) whenever no open session already has that id (`:513-524`). Every reopen of the same goal therefore reuses one session id.
- A close records the session id in two sets that nothing clears after a successful close: the runtime's `_admission_fenced_sessions` (added at `console_runtime.py:6027`, discarded only when `begin_session_close` fails at `:6034`) and the controller's `_session_close_generations` (`console_chat_controller.py:14750`). `release_session_close_fences` (`:14952`) releases the fleet and wake fences but not these.
- So a session re-created under a closed id is refused: `_raise_if_disposed_or_session_fenced` (`console_runtime.py:2933-2938`) raises "Console session is closed." on admission, and `begin_session_close` refuses a second close (`console_chat_controller.py:14622-14625`). TASK-34975's port also makes the context display refuse such a session, retrying up to every 5 s.

The root cause is the id reuse. The latches are deliberate: a close that timed out keeps its generation latched so late callbacks cannot reach a later incarnation of the same chat (`console_runtime.py:6131-6139`).

**Mobile, for the record.** The mobile persisted-session restore (`mobile_service.py:851-861`) has the same id-reuse shape, but cannot hit this today: `MobileChatService._build_engine` (`:482-545`) builds its own controller with no `ConsoleRuntime` (ADR-234), and nothing under `Web_Server/` closes a session. The only caller of `ConsoleRuntime.close_session` is the Console UI (`UI/Console_Modules/session.py:3331`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After a clean close of a goal run's chat, resuming that paused or ready goal gives a session the user can send in and close
- [x] #2 Late callbacks from a closed session, including one whose close timed out, still cannot reach the reopened session
- [x] #3 A test closes a goal run's chat, resumes the goal, then sends and closes, and is shown red before the fix
- [ ] #4 The new test runs on a PR lane
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Close a goal chat through the runtime, resume it, and check that both automatic dispatch and another close target the new live session. Cover an old producer that outlives the bounded close drain.
2. Keep the durable goal conversation and ledger chain. Allocate a fresh live session ID after close and bind the next goal attempt to that exact ID. Reject an attempt retry from a different session.
3. Run the goal dispatch, attempt ledger, and close regressions in the required recovered-suites job.
4. Close the concurrent ledger test's worker-owned SQLite handles and rerun it with the recovered-suites storage-lease guard.

ADR required: no new ADR
ADR path: backlog/decisions/227-native-console-goal-runs.md
Reason: The fix preserves the durable goal and its existing authority and repairs a live session lifecycle bug.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Both close variants reproduced the closed-session refusal. The live-session fix passed those two cases and 11 ledger tests locally. The timeout case holds an old producer across hydration and observes its original close fence, then sends in the reopened chat and closes it. Required Linux CI also exercises public Resume on a real paused goal and verifies the saved conversation and the new live session in the next durable attempt. ADR-227 records the identity distinction.

The provider send and second close pass locally after an old producer outlives the original close, and the late producer still observes its exact admission fence.

The tightened regression uses one real runtime for close, public Resume, a native automatic iteration, an ordinary provider send, and the second close. Final Ubuntu qualification passed all 39 Goal and ledger cases at four workers. The same test with clean dev 9fa2bf1b1d production code failed in 6.17 seconds because the reopened live ID equaled the closed ID. [Linux job](https://github.com/rmusser01/tldw_chatbook/actions/runs/45475265787/job/138733698986).

The official recovered-suites job passed 2,022 tests and failed teardown because the Goal ledger test's two accept workers retained SQLite leases. Each accept worker now closes the handle it opened. The exact failure reproduced with `--storage-lease-guard` before that change. The complete Goal ledger file then passed all 11 cases with the guard enabled. [Initial PR lane](https://github.com/rmusser01/tldw_chatbook/actions/runs/45480195611/job/138754609306).
<!-- SECTION:NOTES:END -->

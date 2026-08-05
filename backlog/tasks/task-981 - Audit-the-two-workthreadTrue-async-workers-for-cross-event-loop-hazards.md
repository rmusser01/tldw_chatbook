---
id: TASK-981
title: Audit the two @work(thread=True) async workers for cross-event-loop hazards
status: Done
assignee:
  - '@Claude'
created_date: '2026-07-27 12:00'
updated_date: '2026-07-27 19:15'
labels:
  - ui
  - concurrency
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Noticed while sweeping `self.call_from_thread` misuse for TASK-929. Two workers combine `@work(thread=True)` with `async def`:

- `Widgets/Media_Creation/swarmui_widget.py:354` — `@work(exclusive=True, thread=True)` on `async def generate_image`
- `Widgets/multi_item_review_window.py:377` — `@work(thread=True)` on `async def _generate_analyses_worker`

**This is not a bug, and the task is not to "fix" it.** Textual supports the combination explicitly. `Worker._run_threaded` (`textual/worker.py:284-323`) checks `inspect.iscoroutinefunction(self._work)` and, when true, routes through `run_coroutine` → `run_awaitable` → `asyncio.run(do_work())`. The decorator only rejects the opposite mistake — a non-async function *without* `thread=True`.

**What it actually means, and why it is worth auditing.** The coroutine does not run on the application's event loop. It runs on a **brand-new event loop created by `asyncio.run()` inside the worker thread**. That is a sharp edge, because anything bound to the app's loop is now being touched from a different one:

- `asyncio` primitives created on the app loop — `Lock`, `Event`, `Queue`, `Semaphore` — are bound to that loop and misbehave when awaited from another.
- Long-lived library objects created on the app loop, notably an `httpx.AsyncClient`, carry loop-bound state; reusing one inside these workers is a genuine hazard.
- Any UI touch must go through `self.app.call_from_thread(...)` because this is a real thread. TASK-929 fixed exactly that in both of these files, which is what surfaced them.
- `asyncio.run()` also *closes* its loop on completion, so anything cached on it does not survive between invocations.

Audit both workers for those four patterns. If each only awaits objects it creates itself inside the worker, the combination is fine and should be left alone with a short comment recording why. If either shares a loop-bound object with the app, that is a real defect to fix — most cheaply by making the worker a plain `def` that owns its own `asyncio.run`, or by moving it off the thread pool entirely.

Worth checking whether any other `@work(thread=True)` in the tree decorates an `async def`; these two were found incidentally, not by an exhaustive search.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both workers are audited for awaiting app-loop-bound asyncio primitives or clients
- [x] #2 Any genuine cross-loop sharing is fixed; anything safe is left alone with a comment recording why
- [x] #3 The tree is searched for other `@work(thread=True)` on `async def` and each is judged the same way
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. AST-scan the tree for @work(thread=True) decorating async def (not regex) to confirm the full set: 6 hits, not the 2 named in the task body.
2. Confirm the real defect: CodeRepoCopyPasteWindow shares one GitHubAPIClient (cached httpx.AsyncClient) between app-loop handlers and two @work(thread=True) async workers, each running on its own asyncio.run()-created-and-closed loop.
3. Fix GitHubAPIClient.client/close to scope the cached client to the currently running loop, invalidating (and best-effort closing) a stale cross-loop client instead of reusing it.
4. Audit the remaining 5 workers individually for the same class of hazard (shared asyncio primitive/client) and for await-free async def that should become plain def; fix or leave with a recorded comment per case.
5. Write tests that drive the accessor from two different event loops and prove non-reuse across a closed loop, not just "a client is returned"; revert-check each fix and record actual failure text.
6. Run the targeted test files, self-review the diff, mark task Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
AST scan (not regex) found 6 @work(thread=True) on async def, not the 2 named in the task body: textual_scheduler_worker.start_scheduler, ccp_message_manager.load_conversation_messages, CodeRepoCopyPasteWindow._export_to_zip_worker + load_node_children, swarmui_widget.generate_image, multi_item_review_window._generate_analyses_worker.

Real defect (CodeRepoCopyPasteWindow): a single GitHubAPIClient instance is shared between app-loop async handlers (load_repository etc.) and the two @work(thread=True) workers, each of which runs on its own asyncio.run()-created-and-closed loop (Worker._run_threaded). GitHubAPIClient.client cached one httpx.AsyncClient forever, so it got built on whichever loop touched it first and was then reused from the other -- a real cross-loop hazard. Fixed at the source (Utils/github_api_client.py): the client property now tracks which running loop built the cached client (`_client_loop`) and invalidates it on a proven mismatch, handing the stale client off via `asyncio.run_coroutine_threadsafe(...)` for a graceful close if its owning loop is still alive, or simply dropping it if that loop is already closed. `close()` got the same loop-aware guard. Chose this (option 1) over per-worker manual open/close because it is the general, root-cause fix for every current and future caller of the shared client, and it self-heals correctly in both directions (app->worker and worker->app) without leaking indefinitely-growing cache entries. Backward compatible with existing tests that inject `_client` directly (only invalidates on a *known* mismatch, i.e. when the property itself set `_client_loop`).

Added Tests/Utils/test_github_api_client.py::TestGitHubAPIClientCrossEventLoop (4 tests) driving the accessor from two distinct event loops (including a real background-thread asyncio.run(), matching the actual worker shape) and asserting non-identity/non-reuse across a closed loop, plus that the app-loop-to-worker-loop handoff actually closes the stale client. Revert-checked: with the fix removed, all 4 fail with `AttributeError: 'GitHubAPIClient' object has no attribute '_client_loop'` (the mechanism itself, not an incidental mock).

Per-worker audit verdicts:
- CodeRepoCopyPasteWindow._export_to_zip_worker / load_node_children: genuinely await self.api_client (1 await each) -- stay async def; fixed via the shared-client change above. Recorded in-code.
- ccp_message_manager.load_conversation_messages: zero awaits (sync DB read + call_from_thread only) -- converted async def -> def. Updated Tests/UI/test_ccp_handlers.py accordingly (dropped `await`); revert-checked, fails with `AssertionError: expected call not found ... Actual: not called` + `RuntimeWarning: coroutine ... was never awaited` when reverted.
- swarmui_widget.generate_image: zero awaits, but worse than "wasteful" -- it internally builds a second event loop and calls `loop.run_until_complete(...)` while already running inside the asyncio.run() loop Textual created for it, which is a guaranteed `RuntimeError: Cannot run the event loop while another loop is running` on every real invocation (reproduced with a minimal repro script). Converted to plain `def`, matching the working pattern already used two methods above it in the same file (check_server_status, load_models). SwarmUIWidget is not mounted anywhere in the live app (dead code) which is why this was never caught.
- multi_item_review_window._generate_analyses_worker: genuinely awaits (asyncio.sleep, LLM call chain) and everything awaited is created fresh inside the call graph -- no shared loop-bound client/lock. Left as async def with a recorded comment. (Also dead code: MultiItemReviewWindow is never mounted, and the LLM path it references, app.llm_api_client/app.run_in_thread, does not exist on TldwCli -- out of scope to fix.)
- textual_scheduler_worker.start_scheduler: not a cross-loop hazard by design -- its while-loop keeps one worker-thread loop alive for the scheduler's whole lifetime, and none of its constituent objects cache a loop-bound asyncio primitive or httpx client (the two AsyncClient uses elsewhere in Subscriptions are already correctly `async with`-scoped per call). However, found and recorded a separate, unrelated, pre-existing bug: SubscriptionSchedulerWorker is a plain object, not a DOMNode, so `@work`'s own `assert isinstance(self, DOMNode)` fails the instant `start_scheduler()` is called (reproduced directly). Its only call site is therefore dead on arrival today. Left unfixed per the task's own guidance -- this module is already marked deprecated in favour of the ADR-019 unified Scheduling scheduler, and fixing a "can never start" bug in a component being migrated away from is out of scope for this audit. Recorded in-code.

Constraints honoured: no bare self.call_from_thread introduced (Tests/test_call_from_thread_guard.py passes), nothing new is awaited on call_from_thread, loguru calls use `{}` placeholders (only doc-comments changed, no new log call sites added).

Modified: tldw_chatbook/Utils/github_api_client.py, tldw_chatbook/UI/CodeRepoCopyPasteWindow.py (comments only), tldw_chatbook/Widgets/Media_Creation/swarmui_widget.py, tldw_chatbook/UI/CCP_Modules/ccp_message_manager.py, tldw_chatbook/Subscriptions/textual_scheduler_worker.py (comment only), tldw_chatbook/Widgets/multi_item_review_window.py (comment only), Tests/UI/test_ccp_handlers.py, Tests/Utils/test_github_api_client.py.

Tests run (foreground, targeted): 114 passed across test_code_repo_copy_paste_window.py, test_multi_item_review_window.py, test_ccp_handlers.py, test_legacy_entrypoints_retired.py, test_code_repo_integration.py, test_github_api_client.py, test_download_caps_wiring.py, test_scheduler_deprecation.py, test_swarmui_adapter.py, test_worker.py, test_call_from_thread_guard.py.
<!-- SECTION:NOTES:END -->

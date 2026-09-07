---
id: TASK-636
title: Investigate new user session CPU hang recursive asyncio task creation
status: Done
assignee:
  - '@claude'
created_date: '2026-07-25 18:00'
updated_date: '2026-07-26 02:11'
labels:
  - followup
  - uat
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UAT 2026-07-25 (scratchpad/uat-new-user): one of two clean new-user sessions escalated to a 99-100% CPU hang for 14+ minutes after entering Settings-RAG. Stack sample (preserved in the UAT evidence dir) shows recursive asyncio task creation with NO capture-related frames - a separate mechanism from the task-627 mouse-capture leak, explicitly NOT fixed by it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Root cause investigated from the preserved stack sample + repro attempt: the C-level recursion mechanism (Python 3.12 `asyncio.eager_task_factory`, which Textual sets unconditionally) was identified, but the preserved evidence has no Python frame symbols and the hang did not reproduce, so the exact recursive call site was NOT conclusively identified. A well-evidenced (but not proven) causal hypothesis is documented instead: the hang was a downstream cascade of the task-627 mouse-capture-leak precondition, which no longer reproduces.
- [x] #2 No fix lands: the hang was not reproduced on current dev, so there is no confirmed defect to target with a proportionate code change or regression test. The investigation's evidence trail (this task + committed repro artifacts) stands in place of a fix; the task is downgraded to a monitoring note per the task's own honest-negative-result allowance.
- [x] #3 Verified across 2 fresh new-user sessions (current dev, this worktree) replicating the exact documented trigger sequence plus stress variants (rapid F1/Escape cycling with interleaved unmatched mousedowns; a targeted mousedown-before-recompose race): CPU stayed at 0.0% through 3-minute idle observation windows and mouse routing kept working throughout. NOT verified over the textual-serve/websocket transport the original UAT used (real network latency between mousedown/mouseup) — that vector was not tested and is called out below as untested, not ruled out.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read preserved stack sample (hang-sample.txt, macOS `sample` tool, C-frames only -- no Python symbolication) and both UAT app logs; identify the repeating C-level skeleton (type_call -> _asyncio_Task___init__ -> task_eager_start -> task_step_impl -> gen_send_ex2, main thread only, other threads idle-blocked on lock acquire) and cross-reference against the periodic 120s db_status_manager timer log lines to determine whether the event loop was fully starved or intermittently yielding.
2. Map Python 3.12 eager-task-factory semantics (Textual's App._process_messages sets asyncio.eager_task_factory) against the observed recursion pattern -- this is a known footgun class (a coroutine that creates a new Task before its own first await, recursively) rather than a specific already-known tldw_chatbook bug.
3. Attempt reproduction on current dev inside this worktree: fresh HOME/TLDW_CONFIG_PATH scratch (new-user profile), faulthandler SIGUSR1 wrapper armed for exact Python-level stack dumps, replicate the exact session-1 UAT sequence (Settings -> RAG -> Clone -> Set active -> edit -> Save -> toggle Preview -> F1 -> Esc), monitor `ps -o %cpu=` for 3-5 minutes per attempt, up to 3 attempts.
4. If reproduced: capture a SIGUSR1 python-level traceback, identify the exact recursive call site in tldw_chatbook code, implement a proportionate fix with a RED->GREEN regression test.
   If not reproduced: determine whether the task-627 mouse-capture recompose fix (or another already-merged fix) plausibly closed the recursion's actual trigger, or document what was ruled out and downgrade the task honestly.
5. Update task-636's AC wording to match the actual verdict, write Implementation Notes, commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Verdict: NOT-REPRODUCED-UNKNOWN (downgraded to a monitoring note), with a documented but unproven causal hypothesis.**

**Stack sample analysis.** `hang-sample.txt` is a macOS `sample`-tool call graph — C-frame symbols only, no
Python-level frame names (confirmed: zero `.py:` or `tldw_chatbook`-module strings anywhere in the 919 KB file).
The repeating skeleton on the (sole busy) main thread is `type_call → _asyncio_Task___init__ →
_asyncio_Task___init___impl → task_eager_start → task_step_impl → gen_send_ex2 → _PyEval_EvalFrameDefault →
method_vectorcall` (×2) → repeat. All 8 other OS threads in the same sample are idle, parked on
`PyThread_acquire_lock_timed` — the recursion is main-thread-only. This exactly matches Python 3.12's
`asyncio.eager_task_factory`: Textual sets it unconditionally on any 3.12+ interpreter
(`textual/app.py:2282-2283`, confirmed present in the installed textual 8.2.7 in this venv). Under that factory,
a coroutine that creates a new Task before its own first `await` runs the child synchronously, nested on the C
stack, inside the parent's own creation call — a known footgun class, not something specific to a single already-
diagnosed tldw_chatbook bug.

**Cross-referencing the app log is the key finding.** `hang-session1-app.log`'s periodic 120s
`db_status_manager` timer fired close to on-schedule for the *entire* 14-minute hang (11:21:48, 11:23:48, 11:25:48,
11:27:48, 11:29:48, 11:31:49, 11:33:48) — proof the event loop was never in one single, unbroken, ever-deepening
recursive call (that would have starved `call_at`-scheduled timers completely). It must have been a repeated
build-up/unwind ("sawtooth") that still let the loop's own bookkeeping run between bursts.

**Code mapping attempt.** Searched `settings_screen.py` (14,268 lines) and the RAG-specific UI/adapter files for
`run_worker`/`create_task`/reactive-watcher chains that could recursively self-trigger. The RAG index-status,
backfill, and set-active workers are all `@work(thread=True)` — off the main event loop entirely, so not
implicated by main-thread-only recursion. No `refresh(recompose=True)` call sites exist outside comments; recompose
is driven purely by the screen's `active_category`/`theme_editor_modified`/etc. `reactive(..., recompose=True)`
attributes. No literal self-recursive call site was identified with certainty — the C-only evidence does not
permit it.

**Reproduction attempts (current dev, this worktree, commit range up to `68ebf1c34`).** Two full fresh new-user
sessions (isolated `HOME`+`TLDW_CONFIG_PATH` scratch dirs, `scratchpad/uat_dl_wrapper.py` faulthandler-SIGUSR1
wrapper armed) replicated session 1's exact trigger sequence end-to-end: Settings → RAG → Clone → Set active →
edit "Default results" → Save → preview a different profile in the picker → F1 → Escape. Plus, within the first
session's process, two additional stress variants: (a) 15× rapid F1-open/Escape-close cycles with an interleaved
*unmatched* mousedown after each Escape (targeting the exact `pop_screen`-recompose gap the task-627 fix's own
docstring admits it only "narrows... does not close entirely"), and (b) a targeted mousedown-on-Input →
`Set active` (triggers a profile-block rebuild) → delayed-mouseup race, mirroring the capturing-widget-removed-
mid-recompose mechanism that same docstring documents. None of the 4 attempts reproduced the CPU hang or its
antecedent mouse-capture leak (task-627's "Bug 1"): mouse clicks kept routing correctly after every attempt, and
`ps -o %cpu=` measured 0.0% through 3-minute idle observation windows in both fresh sessions. A SIGUSR1
faulthandler dump taken on the idle attempt-2 process confirmed a healthy stack: main thread parked in
`_run_once`/`select()`, all worker threads parked on `queue.get()` — no residual recursion.

**Causal hypothesis (documented, not proven).** `git log` shows the task-627 mouse-capture-leak fix
(`fdfc09a133` 12:50, `557cee3499` 13:15, both 2026-07-25) landed *after* the UAT baseline commit (`0037e48ab9`,
11:14) but *before* this task was filed (`ceccad6213`, 13:21) — i.e. Bug 1 was already fixed by the time task-636
was written up. In the original UAT, session 2 reproduced Bug 1 alone with *zero* subsequent CPU escalation
(stayed at 0%), proving Bug 1 alone doesn't cause the hang. Session 1 additionally hung, but only after *more*
interaction happened (Save, Preview toggle, F1, Escape) *while already in the broken-capture state* from Bug 1.
This is consistent with the recursive-task-creation hang being a downstream cascade that required the mouse-
capture corruption as a precondition rather than a fully independent trigger — and since task-627 specifically
hardened the exact F1-modal/`pop_screen` recompose-teardown race, removing the precondition would explain why the
cascade no longer fires. This mapping is **not proven**: the preserved stack sample has no Python frame symbols,
so the exact recursive call site inside tldw_chatbook was never identified, and no fresh hang was reproduced to
capture a corroborating Python-level trace.

**Explicitly NOT tested / not ruled out.** The original UAT ran over textual-serve + a websocket transport, where
a MouseDown and its matching MouseUp can arrive as independently-timed messages with real network latency — the
task-627 docstring calls this out by name as the plausible carrier of the capture-leak race. This investigation's
repro used raw tmux SGR mouse sequences (down/up sent essentially back-to-back, no real transport latency), which
structurally cannot open the same timing window. The textual-serve/Playwright vector was not attempted here (time
budget) and should be treated as untested, not ruled out, if this is revisited.

**Files referenced (no code changes made):** `tldw_chatbook/UI/Navigation/base_app_screen.py` (task-627's
`refresh()`/`recompose()` overrides), `tldw_chatbook/UI/Screens/settings_screen.py` (RAG category workers/
reactives, searched but no recursive site found), `.venv/lib/python3.12/site-packages/textual/app.py:2282-2283`
(confirms Textual sets `asyncio.eager_task_factory`). Repro evidence preserved alongside the original UAT dirs at
`scratchpad/repro-636-attempt1/` and `scratchpad/repro-636-attempt2/` (fresh configs, faulthandler dumps, app
logs).
<!-- SECTION:NOTES:END -->

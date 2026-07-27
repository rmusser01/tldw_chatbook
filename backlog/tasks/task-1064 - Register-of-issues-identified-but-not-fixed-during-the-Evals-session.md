---
id: TASK-1064
title: >-
  Register of issues identified but not fixed during the Evals/settings session
status: To Do
assignee: []
created_date: '2026-07-27 17:00'
labels:
  - tech-debt
  - follow-up
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Everything surfaced during the Evals rebuild and its follow-up work that was verified but deliberately left unfixed. Each item below is independently actionable and can be split into its own task; they are collected here so none is lost.

Findings that already have their own tasks are not repeated: TASK-1022 (ADR-019 rollback), TASK-1034/1035/1036 (Evals UAT).

---

### 1. `ConsoleProviderGateway` has the single-slot client-loop race that TASK-981 fixed elsewhere — **highest value here**

`Chat/console_provider_gateway.py::_active_http_client` keeps **one** `http_client` plus one `_client_loop`, and on a loop mismatch it swaps in a new client and schedules `aclose()` on the loop it replaced.

That is structurally the same defect Qodo found in `GitHubAPIClient` during PR #1009, where it was fixed by moving to a per-loop `WeakKeyDictionary` cache. The gateway is in better shape than that code was — the read-check-swap is atomic under `_client_lock` (PR #629), and the close future carries a done-callback — but the underlying hazard remains: **closing a client that another live loop may still be using.**

Its own docstrings state the exposure. `_active_http_client` says the gateway is shared between "readiness probes (awaited on the app's own event loop)" and "agent-runtime generation calls (bridged from a worker thread via a fresh `asyncio.run()` per turn)", and `_schedule_stale_client_close` concedes the previous loop "may still be running elsewhere (the app's main loop)". Those two can overlap, which is exactly the race condition.

The fix is the one already proven in `GitHubAPIClient`: key the cache by running loop so no loop ever closes another's client, prune closed loops to bound growth, and keep the existing lock and done-callback.

### 2. `Tests/UI/test_console_native_chat_flow.py` is red on `dev` — 18 failures

Observed repeatedly across several agents in this session and dismissed each time as "pre-existing and unrelated", which was true but meant nobody wrote it down. Measured on `origin/dev`: **18 failed, 192 passed** in ~5m24s.

At least two distinct causes:
- `AssertionError: assert 'llama_cpp' == 'local_llamacpp'` — a provider-name mismatch between test expectation and current behaviour.
- `AttributeError: 'ChatScreen' object has no attribute '_task_resume_state'` — reported by agents during the `call_from_thread` sweep.

Worth establishing whether the tests are stale (provider naming changed under them) or the product regressed. The file also takes over five minutes, which is why it is rarely run and why the rot went unnoticed.

### 3. Repository hygiene: 2,536 unreachable loose objects and a stale `gc.log`

`git count-objects -v` reports **2,536 loose objects, ~35 MB**, and `.git/gc.log` carries "There are too many unreachable loose objects; run 'git prune'". Automatic gc will not run again until that file is removed.

Deliberately not acted on during the session: this checkout hosts many concurrent worktrees and other sessions' in-flight work, and `git prune` removes unreachable objects — which is precisely what an agent's detached or uncommitted work may look like. It should be done when the repo is quiet and no other sessions are active, not opportunistically.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] `ConsoleProviderGateway` uses a per-loop client cache; a test proves two live loops never close each other's client
- [ ] The 18 failures in `test_console_native_chat_flow.py` are each classified as stale-test or product regression, and resolved
- [ ] Repo gc/prune is run at a quiet moment and `.git/gc.log` cleared
- [ ] Any item split into its own task is linked from here
<!-- AC:END -->

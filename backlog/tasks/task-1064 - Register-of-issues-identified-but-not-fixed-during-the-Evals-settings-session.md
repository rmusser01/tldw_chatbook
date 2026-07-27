---
id: TASK-1064
title: Register of issues identified but not fixed during the Evals/settings session
status: To Do
assignee: []
created_date: '2026-07-27 17:00'
updated_date: '2026-07-27 22:08'
labels:
  - tech-debt
  - follow-up
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Everything surfaced during the Evals rebuild and its follow-up work that was verified but deliberately left unfixed. Each item below is independently actionable and can be split into its own task; they are collected here so none is lost.

Findings that already have their own tasks are not repeated: TASK-1022 (ADR-019 rollback), and TASK-1034 / TASK-1076 / TASK-1036 (Evals UAT). All four are on `dev`. The UAT trio was filed as 1034/1035/1036; 1035 was renumbered to 1076 after colliding with another session's Done task of that id, so the reference above names the surviving ids.

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
- [ ] #1 `ConsoleProviderGateway` uses a per-loop client cache; a test proves two live loops never close each other's client
- [x] #2 The 18 failures in `test_console_native_chat_flow.py` are each classified as stale-test or product regression, and resolved
- [ ] #3 Repo gc/prune is run at a quiet moment and `.git/gc.log` cleared
- [ ] #4 Any item split into its own task is linked from here
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Triaged all 18 red tests in `Tests/UI/test_console_native_chat_flow.py`. All 18 are stale tests; no product regressions found. Full file: 210 passed in 4m32s (was 18 failed/192 passed in ~5m24s on dev).

Root causes (3 distinct, all test-only):

1. **`_task_resume_state` missing on bare screens (3 tests)** — `_bare_console_screen()` builds a `ChatScreen` via `__new__` (bypassing `__init__`) and manually seeds a few instance attributes, but never seeded `_task_resume_state` (a plain `TaskResumeState()` default, unconditionally set in `ChatScreen.__init__`). `_serialize_native_console_state` started reading it directly (no `getattr` fallback, unlike the file's own `_ensure_console_image_view` precedent for bare-screen safety). Fix: seed `screen._task_resume_state = TaskResumeState()` in the helper. Confirmed pre-fix error: `AttributeError: 'ChatScreen' object has no attribute '_task_resume_state'`.

2. **Provider-name mismatch, 1 test** — `test_console_provider_selection_reads_local_llamacpp_configured_model` set `app.chat_api_provider_value = "local_llamacpp"`, an attribute with zero readers anywhere in `tldw_chatbook/` (grepped confirmed) — dead since the legacy root-chat state removal (TASK-650). The live mechanism is `app_config["chat_defaults"]["provider"]`, read by `_effective_console_provider_model()`. Because nothing set it, provider fell through to the hardcoded `"llama_cpp"` fallback in `_build_console_provider_selection`. Confirmed: both `"llama_cpp"` and `"local_llamacpp"` are independently valid, actively-used provider keys throughout the codebase (`KEYLESS_PROVIDER_KEYS` lists both) — not an aliasing bug. Fixed by setting `app.app_config["chat_defaults"]` before mount, matching the working pattern used by `_configure_native_ready_console`. Confirmed pre-fix error: `AssertionError: assert 'llama_cpp' == 'local_llamacpp'`.

3. **`_select_llamacpp_console` test helper never selected a provider, 13 tests + 1 more with the same shape (14 total)** — the shared helper (used by 22 tests in this file, 29 more across 9 other files) set `console._console_control_provider`/`_console_control_model` post-mount and called `_sync_console_control_bar()`. Traced `_console_control_provider`'s only real consumer: `on_console_compact_provider_changed`'s docstring says it "mirror[s] native compact provider changes into Console-owned **labels**" — it is display-only and only feeds a *new* session's defaults via `_effective_console_provider_model()`. The screen's *already-existing* first session (created at mount, before the helper runs) keeps its mount-time snapshot (`provider=""`), so `_build_console_provider_selection` falls back to `"llama_cpp"` for the run recipe while the actual `session.settings.provider` readiness check sees `""` → `ConsoleSettingsReadiness(label="Unknown", native_send_supported=False)` → setup modal blocks the composer and the send never reaches the gateway. Confirmed by direct instrumentation (`_active_console_settings_readiness()` printed `label='Unknown'`, `provider=''`) and by comparing against tests that bypass the helper via direct `store.replace_session_settings(...)`, which do work. Fixed by having the helper additionally call `_replace_active_console_session_settings(replace(settings, provider="llama_cpp", model="test-model", base_url=None, source="user"))` — the same call the real Console Settings modal apply path (`chat_screen.py:1346`) and `_apply_detected_local_server` (`chat_screen.py:9143`) use. Confirmed pre-fix errors: `TimeoutError` (5 tests: composer stayed blocked so a stream/stop-control wait never resolved) and `AssertionError: Text not found: 'accepted'` / `'Assistant  hello'` / `'assistant return'` / `'llama.cpp stream failed'` (8 tests: send blocked, so gateway output text never appeared).
   `test_console_configured_model_reaches_gateway_when_ui_model_is_unset` hit the identical bug via a different post-mount no-op (`console._console_control_provider = "local_llamacpp"`) rather than the shared helper; fixed the same way by moving the provider into `app_config["chat_defaults"]` pre-mount.

Regression check: re-ran all 22 in-file callers of `_select_llamacpp_console` (13 previously-failing + 9 previously-passing) and all 29 tests in the 9 other files that import it (`test_console_stream_scrollback.py`, `test_console_keyboard_trust.py`, `test_console_stop_feedback.py`, `test_console_regenerate_feedback.py`, `test_console_rename_persistence.py`, `test_console_run_status_wiring.py`, `test_console_send_draft_snapshot.py`, `test_console_pending_attachment_stash.py`, `test_console_switch_draft_integrity.py`) — all pass, no regressions from making the helper actually select the provider.

Runtime: file dropped from ~5m24s to 4m32s (mostly the ~13 tests' failed `_wait_for_text` retry loops, 80 attempts x 0.05s each, no longer exhausting). Did not investigate further — task scope was triage/fix, not a performance rewrite. The remaining runtime is dominated by per-test full `TldwCli`/`ChatScreen` app mounts (~210 tests, most spinning up a full Textual pilot); no single obvious quick win beyond fixture reuse, which is a larger change.

Files changed: `Tests/UI/test_console_native_chat_flow.py` only (test-only; no `tldw_chatbook/` product code touched, since triage found no product regression).
<!-- SECTION:NOTES:END -->

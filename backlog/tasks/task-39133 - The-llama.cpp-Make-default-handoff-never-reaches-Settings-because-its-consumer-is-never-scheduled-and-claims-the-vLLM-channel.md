---
id: TASK-39133
title: "The llama.cpp \"Make default…\" handoff never reaches Settings, because its consumer is never scheduled and claims the vLLM channel"
status: To Do
created_date: 2026-11-07 02:03
labels:
- follow-up
- settings
- providers
- llamacpp
priority: medium
references:
- tldw_chatbook/UI/Screens/settings_screen.py
- tldw_chatbook/UI/LLM_Management/llamacpp_setup_view.py
- Tests/UI/test_llamacpp_consumers.py
dependencies:
- TASK-38975
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07. TASK-38975's notes (round-11 branch fu/r11-settings) list it as a follow-up. Line numbers are dev's.

- llama.cpp setup's "Make default…" stages a `LLAMACPP_DEFAULT` handoff and opens Settings ▸ Providers & Models (`llamacpp_setup_view.py:182`, `:395`, `stage_handoff` at `:689-713`). Nothing in Settings consumes it.
  - Settings' `on_mount` schedules only the vLLM consumer (`settings_screen.py:5064`).
  - `_consume_pending_llamacpp_default_intent` (`:14477`) has no production caller.
  - `git log -S` finds no commit that ever scheduled it.
- Called directly, `_consume_verified_default_intent` (`:14487`) uses only its `provider` argument. Its channel, intent type, owner and freshness are hard-coded to vLLM: it claims `HandoffChannel.VLLM_DEFAULT`, requires a `VllmDefaultIntent` and checks `_vllm_connection_owner` (`:14498-14516`).
- The recovery retries are vLLM-only too (`_retry_vllm_default_cleanup` `:14743`, `recover_vllm_default_handoff` `:14761`). This has been so since 9dab7d15b1 (2026-09-28, "wire llamacpp handoff consumers"), which added the parameterised wrapper but kept the vLLM literals in its body.
- A pure probe on dev, using the real `PendingHandoffStore` and a ready `LlamaCppConnectionOwner`, gives: consumer returns False; `LLAMACPP_DEFAULT` still pending.
- `Tests/UI/test_llamacpp_consumers.py::test_settings_stages_verified_llamacpp_without_persisting` fails at `draft is not None` under a private profile (TASK-38975's notes). Run plain, it fails locally with `RecoveryRequired` (TASK-38111) instead.
- No PR lane runs `test_llamacpp_consumers.py`, so the red went unnoticed.
- TASK-25907.23 (In Progress) describes this fix in its notes, but `git log -S` shows it never reached dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With a current llama.cpp default intent staged, opening Settings stages provider `llama_cpp` with the intent's model and endpoint, releases the intent, and persists nothing; `test_settings_stages_verified_llamacpp_without_persisting` is red on dev at `draft is not None` under a private profile, and green after
- [ ] #2 The llama.cpp consumer leaves a pending vLLM intent untouched, and the vLLM consumer leaves a pending llama.cpp intent untouched; both are pinned
- [ ] #3 A failed llama.cpp release can be retried and recovered on its own channel (`test_llamacpp_default_recovery_blocks_other_default_until_released` green)
- [ ] #4 The Settings nodes of `test_llamacpp_consumers.py` pass locally through run.sh and run on a PR lane (UI census tail entries of about 30 s each, with recent job times and three measured runs recorded; never the fast PR contract)
<!-- AC:END -->

---
id: TASK-39133
title: "The llama.cpp \"Make default…\" handoff never reaches Settings, because its consumer is never scheduled and claims the vLLM channel"
status: Done
assignee:
- '@claude'
created_date: 2026-11-07 02:03
updated_date: 2026-11-07 07:39
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
- [x] #1 With a current llama.cpp default intent staged, opening Settings stages provider `llama_cpp` with the intent's model and endpoint, releases the intent, and persists nothing; `test_settings_stages_verified_llamacpp_without_persisting` is red on dev at `draft is not None` under a private profile, and green after
- [x] #2 The llama.cpp consumer leaves a pending vLLM intent untouched, and the vLLM consumer leaves a pending llama.cpp intent untouched; both are pinned
- [x] #3 A failed llama.cpp release can be retried and recovered on its own channel (`test_llamacpp_default_recovery_blocks_other_default_until_released` green)
- [x] #4 The Settings nodes of `test_llamacpp_consumers.py` pass locally through run.sh and run on a PR lane (UI census tail entries of about 30 s each, with recent job times and three measured runs recorded; never the fast PR contract)
<!-- AC:END -->

## Implementation Plan

1. Red first: mark the Settings nodes of `Tests/UI/test_llamacpp_consumers.py` `@private_profile_test` (as the file's Console nodes already are) and run them on base; record `draft is not None` for AC#1.
2. Make `_consume_verified_default_intent` use its `channel`, `intent_type`, `provider`, `owner_attribute` and `current_intent` arguments (one consumer, no copy); its recovery gate retries whichever default channel holds a retained release (`_vllm_default_recovery()`), so either intent blocks the other until released.
3. Point `_retry_vllm_default_cleanup` and `recover_vllm_default_handoff` at the retained recovery's own channel.
4. Schedule `_consume_pending_llamacpp_default_intent` on mount beside the vLLM consumer.
5. Pin AC#2 (each consumer leaves the other channel pending) and run TASK-38975's held-echo test for the llama.cpp handoff too.
6. Lane the Settings nodes as UI census tail entries; record job times and three measured runs.

## Implementation Notes

**Fix** (`tldw_chatbook/UI/Screens/settings_screen.py`), following TASK-25907.23's notes ("schedule the already-verified llama consumer beside its sibling"):
- `on_mount` schedules `_consume_pending_llamacpp_default_intent` beside the vLLM consumer. The Settings route is not reusable, so every handoff mounts a fresh screen and the mount is the one boundary (TASK-38975's trace).
- `_consume_verified_default_intent` is the one consumer for both intents and now uses its arguments: it claims `channel`, requires an exact `intent_type`, reads `owner_attribute` and checks freshness with `current_intent`; the placeholder, credential, model-profile and config reads use `provider`. The staged-result copy names the provider ("Verified llama.cpp target staged…").
- Recovery: the consumer's gate, `_retry_vllm_default_cleanup` and `recover_vllm_default_handoff` retry the retained release on its own channel (`_vllm_default_recovery()`, which already read both channels). A retained release on either default channel blocks both consumers until it is released. The recovery status names the provider; the button reads "Retry provider handoff cleanup", the copy `test_llamacpp_default_recovery_blocks_other_default_until_released` already pinned.

**AC#1.** The file's Settings nodes now run under `@private_profile_test`, as its Console nodes already did (plain, they stop at `RecoveryRequired`, TASK-38111). Red on base (run.sh, `-n 4`, load 37): `test_settings_stages_verified_llamacpp_without_persisting` `assert None is not None` (`draft is not None`); the three `test_settings_acknowledges_llamacpp_even_when_fields_are_already_saved` cases and `test_llamacpp_settings_late_owner_expiry_compensates_before_ack` `assert False is True` (the consumer); `test_llamacpp_default_persists_only_after_settings_save` `assert True is False`. Green after.

**AC#2.** New `test_each_default_consumer_leaves_the_other_providers_intent_pending`: with a vLLM intent staged the llama.cpp consumer returns False and leaves it pending; with a llama.cpp intent staged the vLLM consumer does the same. Red on base: `assert True is False` (the llama.cpp consumer took the vLLM intent).

**AC#3.** `test_llamacpp_default_recovery_blocks_other_default_until_released`: red on base (`assert None is not None`, no llama.cpp recovery was retained), green after.

**TASK-38975's stale-echo guards.** `Tests/UI/test_settings_custom_id_fast_typing.py::test_a_late_mount_echo_keeps_the_handoffs_model_and_typing_still_stages` is parametrized over `vllm` and `llama_cpp`: the held model, endpoint and provider mount echoes delivered after the llama.cpp handoff leave its model and endpoint staged, and typing still stages. `[llama_cpp]` was red on base (`assert not True`: the intent was never consumed), green after; `[vllm]` green on both.

**Covering run** (run.sh, `-n 4`, load 35-41): the two files above plus the vLLM default-handoff tests of `test_console_provider_apply_defaults_flow.py` (eight `test_vllm_default_*` functions, plain, so run under a temporary private marker and then reverted): 45 passed.

**Lane (AC#4).** The 9 Settings nodes are UI census tail entries, one per node (with TASK-39135's 3 and TASK-39134's 1; `MINIMUM_FILES` 1809 -> 1822; they land one each on shards 18-30). Recent UI Fast Lane shards took 8-16 min of their 20 (runs 42321107301, 42319781834, 42310788537). The 13 new entries, run serially with the lane's flags (`--timeout=180 --durations=0`), 3 runs: 13 passed each, 153 s / 153 s / 201 s at load 56 / 32 / 32; per entry 8.1-20.3 s call in runs 1-2 and 9.2-30.2 s in run 3, the largest being the TASK-39134 sweep (19.0 / 20.3 / 30.2 s).

Files: `tldw_chatbook/UI/Screens/settings_screen.py`, `Tests/UI/test_llamacpp_consumers.py`, `Tests/UI/test_settings_custom_id_fast_typing.py`, `Tests/Architecture/test_module_size_ratchet.py` (+10, owner waiver), `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.

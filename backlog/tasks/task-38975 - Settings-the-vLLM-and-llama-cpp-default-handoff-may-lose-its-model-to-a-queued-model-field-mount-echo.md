---
id: TASK-38975
title: "Settings: the vLLM and llama.cpp default handoff may lose its model to a queued model-field mount echo"
status: Done
assignee:
- '@claude'
created_date: 2026-11-06 01:21
updated_date: 2026-11-07 01:38
labels:
- follow-up
- settings
- providers
priority: medium
references:
- tldw_chatbook/UI/Screens/settings_screen.py
- Tests/UI/test_llamacpp_snapshot_settings.py
dependencies:
- TASK-36530
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down. Checked on dev 29592ecc9b and on the TASK-36530 branch, 2026-11-05. Rewritten 2026-11-06, once the test flake this task first tracked had been traced.

**The test flake (traced; the fix belongs to TASK-36530).** `test_f9_save_preserves_new_edits_while_persistence_waits` failed about 1 in 240 runs with `_provider_draft()` None. Its first `await pilot.pause()` returns before the Providers & Models card has mounted, so the model Input's mount echo, an `Input.Changed` carrying the saved model, can arrive after the test's direct `_stage_provider_value("model", …)`. Then:
- `handle_model_value_changed` (`tldw_chatbook/UI/Screens/settings_screen.py:30515-30534`) re-stages the original model;
- the draft is now clean, so `_stage_provider_value` pops it (`:14260`).

The same race exists at base. The TASK-36530 branch fixes the test with one more `pilot.pause()`.

**The open product question.** The vLLM/llama.cpp default handoff, `_consume_verified_default_intent` (`:14487`), stages the provider, model and endpoint (`:14530-14533`). It then writes the model field quietly through `_set_model_field_value`, which suppresses `Input.Changed` (`:16508`), and writes the endpoint the same way (`:14534-14537`). On mount it is scheduled with `call_after_refresh(self._consume_pending_vllm_default_intent)` (`:5064`).

If a freshly composed card's model-field mount echo is still queued when the handoff runs, the echo arrives afterwards and `handle_model_value_changed` re-stages the saved model:
- its navigation-echo guard (`:30522-30528`) skips only when no provider draft exists, and the handoff has just created one;
- the provider and endpoint stay staged, so the draft is not popped, but the handoff's model is silently replaced.

The user would then save the new provider with the old model. This ordering is not traced yet.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The ordering is traced and recorded: whether a Providers & Models card's model-field mount echo can still be queued when _consume_verified_default_intent runs (on mount via call_after_refresh, and on a later in-session handoff), for both the vLLM and llama.cpp intents
- [x] #2 A test drives the real handoff with the mount echo delivered after it (forced if it cannot happen naturally); after the echo, the draft still holds the handoff's provider, model and endpoint and the model field shows the handoff's model; it is red on dev if the race is real
- [x] #3 If the race is real, the fix keeps a handoff-staged model against a stale mount echo while a user's later typing still re-stages the model; both are pinned
- [x] #4 If the race cannot happen, the notes record why with the evidence, the test from #2 stays as a pin, and the task closes
<!-- AC:END -->

## Implementation Plan

1. Trace, on a mounted Settings with a staged handoff, when the model field's mount echo reaches the screen relative to `_consume_verified_default_intent` (temporary probe, not committed), for vLLM; check whether the llama.cpp intent reaches that path at all, and whether any in-session path calls it.
2. Test: hold the real mount echo (`Input.post_message` wrapper) and deliver it after the real handoff; assert the draft and the field keep the handoff's model; then type a Custom ID and assert it stages. Red on dev first.
3. Fix in `handle_model_value_changed`: ignore an `Input.Changed` whose value is no longer the field's (a stale echo). A deterministic state check, no timer.

## Implementation Notes

The race is real in principle, though it was never seen in a natural run, so the fix applies (AC#3). AC#4's branch (the race cannot happen) did not hold. Its test stays as the pin AC#4 asks for, so AC#4 is ticked as covered by AC#3.

**AC#1, the trace.** A temporary probe (not committed; run.sh, private profile) wrapped `Input.Changed.__init__`, `SettingsScreen._on_message`, `_consume_verified_default_intent` and `_stage_provider_value`. It staged a vLLM default intent and mounted Settings the way production does: `apply_navigation_context` runs before the push or switch (`app_navigation.py:1043` before `:1077`), so the Providers & Models card is composed with the screen. In 40 of 40 runs at load 36-58, the model field's mount echo (`model-a`, the saved model) reached the screen 135-1018 ms before `_consume_verified_default_intent` started, and the run ended with the draft and the field both on `chatbook-vllm`. Textual does not guarantee that order, though. `MessagePump._pre_process` dispatches an Input's Compose and Mount directly, and the `Input.Changed` its constructor queued is handled only afterwards, in the Input's own loop. The parent's `mount_all` waits only for that Mount. So when Settings' `on_mount` queues the consume (`call_after_refresh`, screen hop then app hop), the echo can still be in the field's queue or partway up its bubble chain. Nothing orders the two.
- In-session handoff: none exists. The consume is scheduled only from `on_mount`, nothing else calls it, and the Settings route is not reusable (`screen_registry.py`), so every handoff gets a fresh screen.
- llama.cpp: the intent never reaches the staging on dev. `_consume_pending_llamacpp_default_intent` is never scheduled (only the vLLM consumer is, at `on_mount`). `_consume_verified_default_intent` also ignores its `channel`, `intent_type` and `owner` arguments: it claims `HandoffChannel.VLLM_DEFAULT` and requires a `VllmDefaultIntent` (since 9dab7d15b1). `test_settings_stages_verified_llamacpp_without_persisting`, run under a temporary `@private_profile_test` marker and then reverted, fails at `assert draft is not None`. This is listed as a proposed follow-up. The fix below is in the shared change handler, so it covers llama.cpp once that is wired.

**AC#2.** `test_a_late_mount_echo_keeps_the_handoffs_model_and_typing_still_stages` (`Tests/UI/test_settings_custom_id_fast_typing.py`, a whole-file UI census entry). It holds the field's real mount echo (an `Input.post_message` wrapper keeps the first `Input.Changed` of `#settings-model-value`), lets the real handoff consume on mount, and delivers the held echo afterwards. Red on dev: `('vllm', 'mod.../completions') == ('vllm', 'cha.../completions')`, i.e. the draft's model became `model-a` while the field showed `chatbook-vllm`.

**AC#3, the fix.** `handle_model_value_changed` returns, after `_sync_model_picker()`, when `event.value != event.input.value`. A stale echo names a value the field no longer holds. Any non-quiet writer posts its own Changed with the value it wrote, and a quiet writer stages for itself, so dropping the stale event loses nothing. This is a state check, not a timer. The same test then types a Custom ID through the picker (Tab to Custom ID, Enter, keys) and asserts the draft takes `typed-model`. Negative control: with the handler made to return always, that assertion fails (`'chatbook-vllm' == 'typed-model'`). Green after the fix: 2 passed (the file).
- Size ratchet: `settings_screen.py` 35654 -> 35657 under the owner waiver (`Tests/Architecture/test_module_size_ratchet.py`, dated 2026-11-06).
- Covering run (run.sh, `-n 4`, load 28-56, 37 files naming the model field, the picker or the handoff channels): 1262 passed, 106 failed. At `fu-base-r11`, 100 of the 106 fail (mostly `RecoveryRequired: raw_source_selection_changed` in plain UI tests). The other 6 passed serially both here and at base.

Files: `tldw_chatbook/UI/Screens/settings_screen.py`, `Tests/UI/test_settings_custom_id_fast_typing.py`, `Tests/Architecture/test_module_size_ratchet.py`.

Review r11g-rev1 I1: the endpoint field had the same race. A held endpoint mount echo (`http://127.0.0.1:9099`) delivered after the handoff overwrote the draft's endpoint while the field showed the handoff's.
- `handle_provider_endpoint_changed` now returns on `event.value != event.input.value`, after the suppress-queue branch.
- The provider Select has a late mount echo too: its held `Select.Changed` carried `llama_cpp` (the saved provider) while the Select already showed `vllm`. Handled, it re-applied llama.cpp and the draft emptied. `handle_provider_value_changed` returns on `event.value != event.select.value` as well.
- The held-echo test now holds all three echoes and checks the endpoint field. Red with the review head's sources: `assert None is not None` (the draft was gone). With the endpoint guard but no Select guard it was still `draft is None`, from the Select echo. Without the Select hold, the model and endpoint guards passed. With both guards the whole file passes (2 passed).
- Covering run (run.sh, `-n 4`): the Select guard broke `test_settings_provider_blank_select_value_is_not_treated_as_provider` (task-565). That test posts a synthetic `Select.Changed(provider, Select.NULL)` to a Select built with `allow_blank=False`. The guard now lets a `Select.NULL` event through, so task-565's pin holds unchanged, and a stale echo carrying a real provider is still skipped.

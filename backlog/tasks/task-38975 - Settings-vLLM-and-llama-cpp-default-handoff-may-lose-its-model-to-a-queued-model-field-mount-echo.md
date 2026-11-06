---
id: TASK-38975
title: "Settings vLLM and llama.cpp default handoff may lose its model to a queued model-field mount echo"
status: To Do
created_date: 2026-11-06 01:21
updated_date: 2026-11-06 01:37
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
- [ ] #1 The ordering is traced and recorded: whether a Providers & Models card's model-field mount echo can still be queued when _consume_verified_default_intent runs (on mount via call_after_refresh, and on a later in-session handoff), for both the vLLM and llama.cpp intents
- [ ] #2 A test drives the real handoff with the mount echo delivered after it (forced if it cannot happen naturally); after the echo, the draft still holds the handoff's provider, model and endpoint and the model field shows the handoff's model; it is red on dev if the race is real
- [ ] #3 If the race is real, the fix keeps a handoff-staged model against a stale mount echo while a user's later typing still re-stages the model; both are pinned
- [ ] #4 If the race cannot happen, the notes record why with the evidence, the test from #2 stays as a pin, and the task closes
<!-- AC:END -->

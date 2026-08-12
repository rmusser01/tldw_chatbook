---
id: task-15673
title: Settings navigation preselect applies the provider then silently drops the model
status: To Do
assignee: []
labels:
  - bug
  - settings
  - navigation
priority: high
---

## Description

Navigating to Settings ▸ Providers & Models with an explicit provider AND model
in the navigation context lands on the right provider but leaves the model field
showing the *previous* provider's model. The model in the context is staged and
then never applied, with no message to the user.

The cause is a self-inflicted race rather than a missing feature.
`apply_navigation_context` stages the provider and model and defers the write via
`call_after_refresh(self._apply_navigation_provider_context, ...)`. Applying the
provider is itself what marks the Providers & Models category dirty, and
`_apply_navigation_provider_context` opens with an unsaved-changes guard
(`settings_screen.py:14161`) whose purpose is to protect a user's real unsaved
draft. By the time the deferred call runs, the guard sees the dirtiness the
navigation just created and returns without writing the model.

This was invisible while the Console test harness booted with a near-empty config
(task-15270): with no saved provider values to differ from, the provider switch
did not register as dirty, so the deferred apply proceeded and the tests passed.
Fixing the harness exposed it.

Measured on `fix/console-test-config-honesty` in
`test_settings_navigation_context_can_preselect_provider_category_target`:
`dirty=False` before the call, context staged correctly
(`_navigation_model='meta-llama/test-model'`), then once the provider Select
reaches `huggingface`, `dirty=True` and the model field still reads `qwen`.

The fix must keep the guard's original purpose intact -- a user's genuinely
unsaved edits must still survive a navigation -- so it needs to distinguish
dirtiness the navigation itself just caused from dirtiness the user typed.
Suppressing the guard outright would reintroduce the draft-clobbering it exists
to prevent.

## Acceptance Criteria

- [ ] Navigating to Providers & Models with both a provider and a model applies both, with the model field showing the requested model
- [ ] A genuinely unsaved user draft in Providers & Models is still preserved when a navigation context arrives, and the model is not force-applied over it
- [ ] The two tests marked xfail(strict=True) for this defect in `Tests/UI/test_settings_configuration_hub.py` pass with the marker removed
- [ ] A test distinguishes the two dirtiness sources directly, so a future change that re-suppresses the model fails loudly rather than silently

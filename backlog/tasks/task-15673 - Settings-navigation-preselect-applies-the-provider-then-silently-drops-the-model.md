---
id: TASK-15673
title: Settings navigation preselect applies the provider then silently drops the model
status: Done
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

- [x] Navigating to Providers & Models with both a provider and a model applies both, with the model field showing the requested model
- [x] A genuinely unsaved user draft in Providers & Models is still preserved when a navigation context arrives, and the model is not force-applied over it
- [x] The two tests marked xfail(strict=True) for this defect in `Tests/UI/test_settings_configuration_hub.py` pass with the marker removed
- [x] A test distinguishes the two dirtiness sources directly, so a future change that re-suppresses the model fails loudly rather than silently

## Implementation Plan

1. Re-measure the dirt source with the echo class already fixed
2. Find the mount-time echo mechanism; verify it against Textual directly
3. Mirror the model handler's existing nav-echo tolerance

## Implementation Notes

Fixed together with task-15740 -- same root class, two mechanisms. The original
diagnosis ("applying the provider marks the category dirty") was right about
the effect but incomplete about the source: the dirt was not the provider
staging itself but (a) posted `Changed` echoes from programmatic repopulation
arriving after the `_syncing_*` flags dropped, and (b) the navigation
recompose's freshly mounted Inputs posting `Changed` for their compose-time
initial values -- verified against Textual with a minimal probe app; a mounted
`Input(value="x")` delivers `Changed("x")`. (b) is why fixing (a) alone left
this task's tests red. The guard in `_apply_navigation_provider_context` is
untouched -- a genuinely unsaved user draft still survives a navigation
(pinned by `test_settings_navigation_context_preserves_existing_provider_draft_values`);
what changed is that navigation no longer manufactures the dirt that tripped
it. Both xfail(strict=True) markers removed; the discriminating test lives in
task-15740's notes. See that task for the full mechanism write-up.

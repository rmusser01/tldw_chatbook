---
id: task-15610
title: Pressing Save in Providers & Models marks untouched fields dirty and refuses the save
status: To Do
assignee: []
labels:
  - bug
  - settings
  - high
priority: high
---

## Description

Change the provider in Settings ▸ Providers & Models and press Save. The save is
refused with **"Model context window must be a positive whole number."** —
an error about a field the user never touched — and nothing is persisted.

The act of pressing Save is what causes it. Measured on `origin/dev`:

| | draft `dirty_keys` |
|---|---|
| before the click | `['model', 'provider']` |
| after the click | `['credential_env_var', 'endpoint', 'model', **'model_context_window'**, 'provider']` |

Three fields the user never edited — `credential_env_var`, `endpoint` and
`model_context_window` — enter the draft as edited-to-empty when Save is
pressed. `model_context_window` then fails its guard
(`settings_screen.py:17231`), which requires a positive integer whenever the
field is dirty and not explicitly reset, and the guard `return`s before any
value is written. So a single unrelated blank field aborts the whole save.

The blanks come from repopulation: switching provider clears the per-provider
fields, and that clearing is indistinguishable from the user emptying them. The
loaded config has a real value the whole time (`model_context_window = 1047576`
in the failing run), so nothing is genuinely missing — the draft just believes
the user deleted it.

This is the same shape as **task-15510**: the app's own repopulation counts as a
user edit. Fixing them together is probably right, because both need the same
distinction between "the user changed this" and "we just rewrote this widget".

Found while triaging task-15512. Two of that task's red tests
(`test_settings_provider_category_saves_provider_defaults_without_sampling`,
`test_settings_provider_switch_does_not_save_stale_endpoint`) are this bug, not
stale contracts — they assert the save persists and it does not.

Note for whoever picks this up: the failure was masked. A malformed log call in
the same code path crashed the save worker under pytest, so the tests reported a
timeout rather than this validation refusal. That is fixed separately (see the
task-15512 branch); do not expect the old symptom.

## Acceptance Criteria

- [ ] Changing the provider and pressing Save persists the provider and model without an error about the context window
- [ ] A field the user has not edited is not entered into the draft as an empty edit when Save is pressed
- [ ] A genuinely user-emptied context window is still rejected with the existing message, so the guard is narrowed rather than removed
- [ ] The two task-15512 tests above pass without weakening what they assert
- [ ] A test distinguishes user-edit from repopulation directly, so a future change that re-blanks these fields fails loudly

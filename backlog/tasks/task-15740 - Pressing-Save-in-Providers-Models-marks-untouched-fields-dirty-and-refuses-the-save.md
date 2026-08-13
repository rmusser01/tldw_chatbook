---
id: TASK-15740
title: Pressing Save in Providers & Models marks untouched fields dirty and refuses the save
status: Done
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

This is the same shape as **task-15673**: the app's own repopulation counts as a
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

- [x] Changing the provider and pressing Save persists the provider and model without an error about the context window
- [x] A field the user has not edited is not entered into the draft as an empty edit when Save is pressed
- [x] A genuinely user-emptied context window is still rejected with the existing message, so the guard is narrowed rather than removed
- [x] The two task-15512 tests above pass without weakening what they assert
- [x] A test distinguishes user-edit from repopulation directly, so a future change that re-blanks these fields fails loudly

## Implementation Plan

1. Trace which handler stages the untouched fields during the Save click
2. Fix the class, not the site -- every programmatic widget rewrite
3. Mutation-check that the tests bite on the mechanism
4. Pin the user-edit / repopulation distinction directly

## Implementation Notes

The measured "pressing Save marks fields dirty" framing was slightly off: the
dirt was created by the PROVIDER SWITCH, not the click -- the click merely
pumped the message queue that delivered it. Two mechanisms, both fixed:

**1. `_syncing_*` flags cannot guard posted messages.** Every repopulation site
wrapped `widget.value = X` in a flag that a `finally` dropped before the
`Input.Changed`/`Select.Changed` message -- posted, not synchronous -- was
delivered. The handlers then staged the repopulated values (endpoint,
credential env var, context window, often "") against the OLD provider's
loaded originals: dirty-and-empty, and the empty context window tripped its
positive-integer guard which returns before writing anything. Fixed with
`with widget.prevent(<MessageType>):` at all eleven programmatic-assignment
sites (the codebase's own established idiom -- rail-label toggle, MCP modes,
speech catalog); the flags stay for the synchronous path. The api-key input's
value-matched suppress queue already handled async delivery correctly and is
untouched. The revert block's unflagged echoes stage loaded values against
loaded originals -- no-ops by construction -- and are deliberately left.

**2. A freshly MOUNTED Input posts Changed for its compose-time initial value**
(verified against Textual directly with a 12-line probe app). That is what
`prevent()` cannot reach and what kept task-15673 red after fix 1: a navigation
preselect recomposes the category with the nav provider's values, and the mount
echoes staged them as user edits. Fixed by giving the endpoint and credential
handlers the same nav-echo tolerance `handle_model_value_changed` has had all
along.

Mutation-checked both mechanisms: un-preventing the context-window sync and
dropping the endpoint nav-echo guard turns 3 tests red.

New tests: `test_settings_provider_repopulation_is_not_a_user_edit` (the app
path stays clean, a genuine edit still stages) and
`test_settings_user_emptied_context_window_still_refuses_the_save` (the guard
is narrowed, not removed: a deliberate clear still refuses with the existing
message and persists nothing).

Also fixes task-15673 (formerly 15510; renumbered on dev after an ID
collision): both its xfail(strict=True) tests now pass with the markers
removed.

Modified: `tldw_chatbook/UI/Screens/settings_screen.py`,
`Tests/UI/test_settings_configuration_hub.py`.

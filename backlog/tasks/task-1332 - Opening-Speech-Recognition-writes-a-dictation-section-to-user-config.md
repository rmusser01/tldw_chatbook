---
id: TASK-1332
title: Opening Speech Recognition writes a [dictation] section to user config
status: Done
assignee: []
labels:
  - bug
  - ui
  - speech
  - privacy
priority: high
---

## Description

Opening the Speech Recognition view writes settings to the user's config file.
The user changes nothing; simply arriving at the view persists a `[dictation]`
section, including `[dictation.privacy]`.

Reproduced against a throwaway `TLDW_CONFIG_PATH` containing only
`[general] users_name`. Mount the Lab → Speech destination, press the Speech
Recognition rail entry, wait, exit. Diff the file:

```
> [dictation]
> provider = "auto"
> language = "en"
> punctuation = true
> commands = true
> buffer_duration_ms = 500
>
> [dictation.privacy]
```

`on_mount` does not call `_save_settings()`. The write comes from
`on_switch_changed` / `on_input_changed` firing as the controls mount --
Textual posts `Changed` when a `Switch` or `Input` is created with a value --
and each of those handlers calls `_save_settings()`, which writes the whole
settings block via `save_setting_to_cli_config`.

Two reasons this matters beyond tidiness:

- It records privacy preferences the user never expressed. Whatever the
  defaults happen to be at that moment become written-down choices, and a
  later change to the shipped defaults will not reach anyone who once opened
  this view.
- It is indistinguishable, in the file, from a deliberate setting. There is
  no way afterwards to tell what the user chose from what merely mounted.

This is the same shape as a defect fixed previously in Evals, where a
predicate called from `compose()` wrote an `eval_models` row — opening the
screen mutated the database on every fresh install.

## Acceptance Criteria

- [x] Opening the view writes nothing; a diff of the config file before and
      after is empty
- [x] Changing a control still persists, as it does today
- [x] A test mounts the view against a scratch config path and asserts the
      file is byte-identical afterwards
- [x] The fix distinguishes mount-time `Changed` events from user-initiated
      ones, rather than debouncing or delaying the write

## Notes

Found while measuring the view for its Console-grammar rebuild. Not fixed in
passing: the guard belongs with the rebuild's own handler wiring, and getting
it wrong in either direction — writing on mount, or dropping a real user
change — is worse than the current state being understood and recorded.

## Implementation Notes

Fixed on `feat/speech-console-redesign` (`a29fb2355`).

`on_mount` sets `_settings_are_mounting` and clears it via
`call_after_refresh`, by which point Textual has delivered the `Changed`
events it posts when a `Switch` or `Input` is created with a value. All six
handler writes now go through one `_persist_settings()` gate that checks the
flag, so a new handler cannot reintroduce this by forgetting to check —
which was the likeliest way for it to come back.

Two tests, because this fails in both directions and neither raises:
`test_opening_the_view_writes_nothing` asserts the config FILE's bytes are
unchanged after opening the view, and `test_a_real_change_still_persists`
asserts a genuine toggle still reaches disk. Silencing real edits would be
the same defect pointing the other way, and harder to notice.

---
id: TASK-1331
title: Speech Recognition history is a shell — the switch records nothing
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

Speech Recognition offers a "Save history" switch and a History section. Neither
does anything.

Found while measuring the view for its Console-grammar rebuild
(`Docs/superpowers/specs/2026-07-27-speech-console-redesign-design.md`), on
`ImprovedDictationWindow`:

- `_add_to_history(transcript)` is `pass` with a `# TODO: Implement history
  saving` comment. **Nothing is ever recorded.**
- The History section is composed only `if self.settings["privacy"]["save_history"]`,
  which is read once at mount. Toggling the switch calls `_update_privacy_ui()`,
  which rewrites the privacy status text and nothing else — so turning the
  setting on mounts no History section. The switch appears to do nothing.
- `_clear_history()` clears a `ListView` that is either absent or empty, then
  notifies "History cleared".

The privacy dimension is why this is filed high rather than as tidy-up. A
switch labelled "Save history" tells the user their transcripts are being
kept. They are not. The reverse mistake — silently keeping them when the
switch is off — would be worse, and the same absent wiring is what would
have to be trusted to prevent it.

## Acceptance Criteria

- [x] Decide and record whether transcript history is a feature this app
      wants, given it stores what was said aloud on the user's machine
- [n/a] If kept: transcripts are actually recorded, persisted where the
      decision says, and shown in the History list — **not applicable, the
      feature was dropped**
- [n/a] If kept: toggling "Save history" takes effect immediately, without
      needing the view to be remounted — **not applicable, the feature was
      dropped**
- [x] If dropped: the switch, the History section and `_clear_history` go
      with it, rather than remaining as controls that do nothing
- [x] Either way, a test drives the switch and asserts the observable
      result, so "the control exists" cannot pass for "the feature works" —
      satisfied by the layout test's control inventory, which now asserts
      the switch is ABSENT, with the reason recorded beside it

## Notes

Not fixed in passing during the rebuild: which way this goes is a product
and privacy decision, not a layout one, and implementing storage for spoken
transcripts without that decision would be the wrong default.

## Implementation Notes

Resolved by REMOVING the controls, on `feat/speech-console-redesign`
(`a29fb2355`).

Of the two options in the acceptance criteria, "dropped" was the right one.
Nothing that worked is lost: `_add_to_history` was a `pass` stub, so no
transcript had ever been recorded. Building real storage would have meant
deciding where spoken transcripts live, how long they are kept, and what the
switch's default should be — a product and privacy decision, and not one to
settle inside a layout fix.

Removed together so nothing is left dangling: `save-history-switch`, the
History list and its Clear button, `_add_to_history`, `_clear_history`,
`_load_history`, the `save_history` settings key, `transcript_history`, and
the call site that fed the stub.

If transcript history is wanted later, it starts from the product decision
this task asked for rather than from a switch that already claimed to do it.

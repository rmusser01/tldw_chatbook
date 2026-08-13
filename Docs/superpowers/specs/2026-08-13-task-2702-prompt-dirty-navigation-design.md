# TASK-2702 — Prompt dirty-navigation feedback

## Context

Library's Prompt editor is explicit-save-only. When a Prompt has unsaved edits,
`LibraryScreen.flush_pending_work()` correctly returns `False` and keeps the
screen mounted, but app-level navigation only logs that refusal. The equivalent
Skill-editor refusal already emits a warning notification, so Prompt navigation
currently looks like a dead control.

## Design

Mirror the existing Skill dirty-exit contract at the Prompt editor boundary:

- add one fixed, content-free Prompt warning message and one Prompt-specific
  notification helper beside the existing Skill helper;
- render a `Discard changes` action in the existing Prompt editor action region;
- keep that action disabled, with a literal reason tooltip, until the editor is
  dirty, then enable it in place without recomposing the live fields;
- when pressed, discard the working copy and follow the same list-return/reset
  tail as a clean Back action; and
- when `flush_pending_work()` receives a `False` result from the Prompt save
  barrier, invoke the warning helper before returning the combined admission
  result.

The message is:

> Unsaved Prompt changes — Save or Discard changes first.

This names the blocked state and two recovery classes that are always truthful.
The concrete save label varies by artifact capability (`Save Prompt`, `Update
original`, or `Convert and save as new Prompt`), while `Discard changes` is the
stable recovery action in every dirty editor. In particular, a compatibility-only
artifact with no convertible System/User text can have every save action disabled;
Discard prevents that state from becoming a navigation dead end.

The existing navigation result, focus, editor scroll ownership, and other note/skill
barriers remain unchanged. A veto preserves the draft; only an explicit Discard
clears it. No persistent status, new reactive state, CSS, worker, confirmation
modal, or shared notification abstraction is introduced.

## Error and privacy behavior

The warning is emitted at `warning` severity through the existing app notification
seam. It contains no Prompt name, content, identifier, exception, or other dynamic
value. If the app notification seam is unavailable, navigation is still refused
and the draft remains intact, matching current fail-closed behavior.

## Verification

Strengthen the existing mounted Prompt `flush_pending_work()` regression to record
notifications and prove all of the following in one ordinary structured-Prompt
flow:

- the dirty Prompt still vetoes navigation;
- the fixed warning is emitted once at warning severity;
- the unsaved draft and dirty flag remain intact.

Add a mounted compatibility-only case with no convertible Prompt body. Edit its
metadata, prove its save/convert actions are unavailable but `Discard changes` is
enabled, press Discard, and prove the list returns without persisting the edit.
Also prove the Discard control is disabled with an explanatory tooltip on a clean
editor and re-disables after a successful save.

Keep the existing Skill veto regression green to prove the sibling behavior was not
changed. Run scoped Ruff, formatting, `py_compile`, diff checks, and the Impeccable
detector over the changed UI/test files.

## Architecture decision

ADR required: no
ADR path: N/A
Reason: this is a routine UX bug fix that applies an existing Library notification
pattern without changing state ownership, service contracts, persistence, security,
or long-lived application structure.

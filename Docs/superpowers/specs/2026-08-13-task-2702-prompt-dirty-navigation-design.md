# TASK-2702 — Prompt dirty-navigation feedback

## Context

Library's Prompt editor is explicit-save-only. When a Prompt has unsaved edits,
`LibraryScreen.flush_pending_work()` correctly returns `False` and keeps the
screen mounted, but app-level navigation only logs that refusal. The equivalent
Skill-editor refusal already emits a warning notification, so Prompt navigation
currently looks like a dead control.

## Design

Add one fixed, content-free Prompt warning message and one Prompt-specific
notification helper beside the existing Skill helper. When
`flush_pending_work()` receives a `False` result from the Prompt save barrier,
invoke that helper before returning the combined admission result.

The message is:

> Unsaved Prompt changes — Save before switching screens.

This names the blocked state and the recovery action that actually exists in the
current editor. It deliberately does not promise a Discard action, because the
Prompt editor currently exposes Save and a dirty-vetoed Back action but no general
Discard control.

The existing navigation result, draft state, focus, editor layout, and other
note/skill barriers remain unchanged. No persistent status, new reactive state,
CSS, worker, or shared notification abstraction is introduced.

## Error and privacy behavior

The warning is emitted at `warning` severity through the existing app notification
seam. It contains no Prompt name, content, identifier, exception, or other dynamic
value. If the app notification seam is unavailable, navigation is still refused
and the draft remains intact, matching current fail-closed behavior.

## Verification

Strengthen the existing mounted Prompt `flush_pending_work()` regression to record
notifications and prove all of the following in one flow:

- the dirty Prompt still vetoes navigation;
- the fixed warning is emitted once at warning severity;
- the unsaved draft and dirty flag remain intact.

Keep the existing Skill veto regression green to prove the sibling behavior was not
changed. Run scoped Ruff, formatting, `py_compile`, diff checks, and the Impeccable
detector over the changed UI/test files.

## Architecture decision

ADR required: no  
ADR path: N/A  
Reason: this is a routine UX bug fix that applies an existing Library notification
pattern without changing state ownership, service contracts, persistence, security,
or long-lived application structure.

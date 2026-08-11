# ADR-055: One reversibility rule for Library destructive actions

Status: Accepted
Date: 2026-08-11
Related Task: [TASK-14901 - One reversibility story across Library destructive actions](../tasks/task-14901%20-%20One-reversibility-story-across-Library-destructive-actions.md)
Supersedes: N/A

## Decision

Every Library-surface action that destroys something owes exactly one of four
patterns. The pattern is determined by two questions, so a future surface can
read this file and know what it owes without a new design discussion:

1. **Does the action destroy something persisted?** No → **Pattern D**
   (draft discard). Yes → question 2.
2. **Is the destruction recoverable at the store level (soft delete)?**
   Yes → **Pattern A** (receipt + Undo). No → **Pattern B** (permanence
   stated). A soft delete may instead qualify for **Pattern C** (silent GC)
   only if it meets *every* guard in Pattern C's definition.

### Pattern A — soft-deleting persisted user data: receipt + Undo

Destroying persisted user data always leaves a **receipt at the point of
action** naming what happened, with **Undo** (restores in place, through the
same service seam the delete used — never raw SQL) and **Dismiss**. The
receipt grammar is the media one (task-4022): `✓ deleted · N items` rendered
where the action completed, surviving until acted on or superseded by a newer
delete. Undo is the **at-point convenience**; the **durable recovery story is
a browsable Trash/restore surface** — task-4025 built it for media
(2026-08-11: the media list's "Trash" view), and its implementation
implements *this rule*, not a new one. Once a store's durable surface
exists, the confirm copy and the receipt both **name it** — media's receipt
reads `✓ deleted · N items · in Trash`, and both confirm copies say
"restore later from Trash". Restore itself is recovery, not destruction, so
it owes **no receipt** (a transient notice is feedback, not a receipt — it
carries no Undo). Until a store's durable
surface exists, the confirm copy must promise exactly what does exist (Undo
where implemented; honest "cannot be undone from Library" where nothing is) —
an honest gap plus a filed conformance task, never silence and never an
implied Trash. Restoring a trashed row is always an explicit operation
(task-4026's contract: resurrection requires an explicit decision —
`restore_trashed=True` / `restore_from_trash`, never a side effect).

Single-item and bulk variants of the same action are **one pattern, one
seam**: single media delete is one-item bulk, so it shares the bulk receipt
state, Undo coroutine, in-flight interlock, and worker group. Forking a
second undo path (or a second in-flight flag) for a new variant is a
regression — PR-1473 established that every mutator of the shared
list/count/receipt state must join one interlock.

### Pattern B — hard-deleting persisted user data: permanence stated

Where deletion is genuinely unrecoverable at the store level, the confirm
copy must **say the deletion is permanent** ("cannot be undone") and name
what is removed. No receipt is owed — a receipt whose Undo cannot exist would
be a lie; the confirmation carries the whole weight, so its copy must be
complete.

### Pattern C — the NAMED exception: blank-note GC (silent, no receipt)

Silent destruction is permitted **only** for a machine-created session
artifact the user never authored content into. The one instance is the
Blank-note GC (LIB-14 / task-2858 AC#5, guards refined by task-4021 and dev's
provenance fix), whose guard conditions are part of this rule — all must
hold, checked in `_flush_library_note_save`:

- the row is **this session's** Blank-note creation
  (`_library_note_session_blank_id == _selected_note_id`);
- no destructive operation is running or admitted for the note session (the
  GC never races an explicit delete/discard the user is managing);
- the title is effectively blank: empty, or still the literal `"Untitled"`
  seed **with the `_library_note_title_user_edited` provenance flag unset** —
  a user who deliberately typed "Untitled" is protected by provenance, never
  by string comparison;
- body and keywords are all blank (covers both "never touched" and "typed
  then emptied out");
- the delete is best-effort and version-checked: on any failure the row
  survives (the pre-existing behavior), and GC never blocks the exit it runs
  inside.

A surface that wants a new silent destruction must either satisfy every
analogous guard (never-touched, session-scoped, machine-created, nothing
user-authored) or take Pattern A/B. "The user probably doesn't care" does not
qualify.

### Pattern D — discarding unsaved drafts: confirm, no receipt

Discarding work that was never persisted destroys nothing in any store, so it
is **confirm-not-receipt**: an explicit act (a dirty-veto plus a deliberate
Discard affordance, or a modal that names the unsaved working copy), and no
receipt afterwards — there is no row an Undo could restore, and a receipt
would be noise. The confirm/affordance copy must make clear it is the
*unsaved* work being dropped.

## Inventory (audited 2026-08-11, task-14901)

| Surface | Store semantics | Confirm today | Recovery today | Pattern owed | Disposition |
| --- | --- | --- | --- | --- | --- |
| Media delete — bulk ("Delete selected") | Soft (`delete_media_item` → `mark_as_trash`) | Armed confirm + copy promising Undo | Receipt `✓ deleted · N items` + Undo/Dismiss (task-4022, PR-1473 interlock) | A | **Conforms** |
| Media delete — single (viewer "Delete") | Soft (same seam) | Armed confirm | Was silence; now the same receipt/Undo via the bulk seam | A | **Conforms (this task)** |
| Notes delete (editor "Delete") | Soft (`soft_delete_note`, version-checked, admission-coordinated) | Two-step confirm; copy honest ("cannot be undone from Library") | None in UI | A | Interim-honest; receipt+Undo filed as **task-15100** |
| Blank-note GC | Soft delete of the session blank | None (silent) | None | C | **Conforms** (all guards above) |
| Prompts/Recipes delete (modal) | Soft (`soft_delete_prompt`; ADR-049 version history retained) | Modal confirm; copy now states permanence (this task) | None in UI | A | Interim-honest; receipt+Undo filed as **task-15101** |
| Skills delete (editor confirm) | **Hard** (`shutil.rmtree` of the skill dir + index removal + script-grant revocation) | Inline confirm; copy names the directory + supporting-file count and says "cannot be undone" | None possible | B | **Conforms** |
| Collections delete ("Confirm delete") | Soft (`deleted_at`; members untouched) | Two-step; tooltip now states consequence + permanence (this task) | None in UI | A | Interim-honest; receipt+Undo filed as **task-15102** |
| Prompt draft / dirty edit discard | Nothing persisted destroyed (dirty veto on exit; modal names the unsaved working copy on delete-while-dirty) | Explicit | n/a | D | **Conforms** |
| Skill draft discard ("Discard changes") | Nothing persisted destroyed (button disabled until dirty) | Explicit | n/a | D | **Conforms** |
| Conversations | Not deletable from the Library surface | — | — | — | Out of scope |

## Context

The Library shipped three different reversibility stories on one screen
(task-14901, from task-4023's cross-task observation and the 2026-08-09
re-critique's heuristic #4 score of 1): blank-note GC destroyed silently,
bulk media delete left a receipt with Undo, and single media delete confirmed
then went silent — the same soft-delete seam, two different promises. The
audit above found the same divergence beyond media: notes, prompts, and
collections all soft-delete (recoverable rows) yet offer nothing after the
confirm, while skills hard-delete and already say so. What was missing was
not mechanism but a rule that names which pattern each surface owes, so new
surfaces (task-4025's Trash, future stores) implement the rule instead of
inventing a fourth story.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Toast-based undo instead of in-place receipts | Toasts vanish on their own schedule; the receipt row persists at the point of action until acted on, and the media canvases deliberately have no success toast today. |
| Heavier confirmation everywhere instead of receipts | Confirmation is not recovery — it front-loads anxiety without giving a way back; the re-critique scored exactly this shape down. |
| Receipting the blank-note GC too (no exceptions) | The GC's target is an artifact the user never knew existed (a committed-on-click blank row); a receipt would advertise internal bookkeeping and re-open LIB-14's noise. The exception is safe only because of its guards, so the guards are part of the rule. |
| Building notes/prompts/collections receipts in task-14901 | Each needs a store-level un-delete seam that does not exist yet (ChaChaNotes has `restore_conversation` but no note restore; `Prompts_DB` and the collections service have none) — structural work, filed as tasks 15100–15102 per the batch's scope-honesty constraint. |

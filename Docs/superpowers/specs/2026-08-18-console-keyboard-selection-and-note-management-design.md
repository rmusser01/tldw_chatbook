# Console keyboard selection (phase 5) + review-note management — Design

Date: 2026-08-18
Status: Approved (maintainer, via brainstorming Q&A; scope/model/surface each explicitly chosen)
Prior art: 2026-08-14 console-selection-annotations design (§42 phase 5 sketch), ADR-068 (+amendments), ADR-031, ADR-066, task-17169 (both-homes persistence)

## Scope decisions (maintainer, 2026-08-18)

1. **Phase 5 is single-row keyboard selection.** §42's "shift+j/k grows a
   row-range" is amended: multi-row selection stays out of scope, consistent
   with the shipped v1 domain ("selections are single-row only") and the
   anchor semantics the persistence layer is built on. Keyboard reaches the
   same states the mouse can, nothing more.
2. **Vim-style in-row character mode**, not whole-row select, not
   shift+arrows (terminal escape-code inconsistency; ADR-031 prefers single
   letters).
3. **Note management lives in a modal** opened from the ✎ marker (click) or
   `n` (keyboard), with per-note Edit and soft-Delete. Not inline marker
   chrome; not the trajectory screen.

## Part 1 — Keyboard selection mode

### Entry

- `s` on the j/k-selected message enters text-selection mode on that row.
  Eligibility is the mouse path's: plain, markdown, and diff rows qualify;
  protected/non-text rows toast a brief refusal and do not enter the mode.
  The row is scrolled visible on entry.
- `s` joins the transcript `BINDINGS` (free: verified against transcript,
  screen, and app bindings; c/e/r/o/v are the taken action letters).

### The mode drives the SAME SelectionManager

`_active_selection_row()` resolves the selected row through
`selection_manager.state.selection` (console_transcript.py:4796), so a
keyboard path that bypassed the manager would open a menu whose actions all
no-op. The manager's API is already input-agnostic — `begin_drag(row_key,
offset)` / `extend_drag(row_key, offset)` take pure offsets. Keyboard mode
therefore:

- `s` → `begin_drag(row_key, 0)` + extend to the first unit (char on plain
  rows, source line on markdown/diff rows — each row kind's existing
  granularity, including markdown's whole-line snap).
- Movement keys → `extend_drag(row_key, new_offset)`.
- `Enter` → the same finish path as mouse release: menu opens with
  `feedback_available` / `run_active` computed exactly as the mouse path
  computes them.

Mouse and keyboard converge on identical `TextSelection` state; every
downstream consumer (quote cap, side chat, feedback anchor, annotation
row_key) is byte-identical.

### Movement keys (mode-scoped, handled in `on_key` with `event.stop()` —
the transcript's established preempt-bindings pattern, see the jump-pill
enter interception)

- Plain rows: `h`/`l` ±1 character (crossing newlines naturally on
  multi-line content), `w`/`b` word forward/back, `0`/`$` start/end of the
  CURRENT line (vim semantics). Word boundaries implemented in the pure
  `console_selection.py` module (unit-testable, no Textual imports).
- Markdown and diff rows: `j`/`k` grow/shrink the selection by one source
  line (their existing line granularity). h/l/w/b/0/$ are inert on these
  rows.
- Selections have a 1-unit floor: shrinking at the floor no-ops. Esc is the
  way to reach "no selection".

### Exit and layering

- `Esc` in mode: exits the mode, clears the text selection, KEEPS the
  message selection. A second Esc clears message selection (the existing
  binding). Transient-surface-first, the same grammar as the modal rules.
- `Enter` in mode: finish + menu (mode interception preempts the
  toggle-message-selection binding).
- Mode state lives on `ConsoleTranscript`; anything that destroys the row
  (streaming replacement, prune, session switch) cancels the mode exactly
  as it cancels a mouse selection today.

### Menu anchoring without a mouse cell

Anchor from the row's laid-out region: x = region.x + indent, y = region
bottom + 1, `selection_top` = region top — feeding the existing
measured-clamp/above-row placement unchanged.

### Hints (ADR-031)

- The static footer/BINDINGS surface gains `s` ("Select text") from Part 1
  and `n` ("Notes") from Part 2 — nothing else.
- Mode keys are advertised by a one-line hint visible while the mode is
  active (the menu's no-run-hint pattern): `h/l chars · w/b words · 0/$
  line · Enter menu · Esc cancel` (line-mode variant on markdown/diff).
  Truthful-hints: the hint renders only in the mode, and lists exactly the
  keys the active row kind honors.

### Streaming

Keyboard selections inherit the shipped semantics untouched: plain rows
hold their last stable range; a markdown selection touching the last line
grows with the stream (recorded ADR-068 behavior).

### Implementation refinements (recorded during execution)

- Markdown rows store character ranges as-is (live-spike evolution), so
  char motions apply to plain AND markdown rows; only diff rows stay
  line-granular (j/k/o only).
- `o` swaps anchor and active end — without it a text-start anchor could
  never reach a mid-text span.
- Keyboard finish drains `consume_release_click()` + `consume_just_finished()`.
- In-mode consumption covers ALL printable chars + enter/up/down (up/down
  alias the selection-nav bindings); page keys/wheel fall through.
- Added scope (maintainer, mid-execution): a fourth base menu action,
  **Create note** — see the plan's Task 6 for the contract.

## Part 2 — Review-note management (the two riders)

### Entry points

- Clicking the ✎ marker opens the notes modal. This requires TWO changes:
  `console-transcript-annotations` joins `PROTECTED_CLICK_CLASSES` (today a
  marker click falls through to the message-selection toggle — a phase-4
  papercut this fixes), and the marker widget gains a click handler.
- `n` on the selected message opens the same modal when the message has
  notes; otherwise toasts "No review notes on this message." (`n` is free;
  joins BINDINGS alongside the other action letters.)

### The modal

`ConsoleReviewNotesModal` (ModalScreen, task-16211 safe-dismissal grammar):

- Opens with the message's annotations loaded off-thread
  (`get_transcript_annotations(conversation_id)` filtered to the anchor's
  persisted message id; annotation_id, comment, quote, timestamps).
- Per note: **Edit** — textarea prefilled with the comment; Save calls
  `upsert_transcript_annotation(..., annotation_id=...)` (the in-place
  update path that already exists and is already tested). The QUOTE is
  immutable — it is what the selection was, not part of the note.
- Per note: **Delete** — confirmation dialog (ADR-031 destructive rule),
  then `soft_delete_transcript_annotation`.
- On any change, the screen re-derives its annotation-preview map from the
  DB (the existing off-thread loader) so the marker updates or disappears
  on the next sync tick; deleting the last note removes the marker.

### What management NEVER touches

The sidecar `user_feedback` audit events are append-only history: editing
or deleting an annotation does not rewrite them, so the trajectory view
keeps the ORIGINAL comment. This divergence is by design and MUST be stated
in the user guide, or it reads as a bug.

No schema change in either part.

## Delivery

- Two tasks (IDs swept fresh at filing time — ≥17387 minimum, full
  remote+worktree sweep per lessons-backlog-hygiene), two sequential PRs
  off latest dev: phase 5 first, riders second.
- TDD throughout; the citation-sources SimpleNamespace fixture gets checked
  on every dev merge (three drift instances on the last branch).
- Spec §42 of the 2026-08-14 design gets an amendment note pointing here.
- User guide page `console/text-selection-and-feedback.md`: keyboard
  section + note-management section + the audit-divergence caveat.
- Live tmux verification for BOTH parts before merge (pilot events cannot
  prove real-terminal key handling — the phase-1 lesson; SGR/character-
  column recipes recorded in memory).
- Standing flow per maintainer: PR → Qodo loop → merge.

## Testing

- Pure: word-boundary/line-motion math in console_selection.py; manager
  state equivalence (keyboard-built vs mouse-built TextSelection).
- Widget: mode entry/exit/layering (Esc twice), per-row-kind movement,
  menu opens with correct gating and anchor, hint truthfulness per row
  kind, streaming-row cancellation.
- Modal: load/edit/delete round trips against real SQLite (unmocked — the
  task-222 rule), marker refresh including last-note removal, sidecar
  untouched after edit/delete (pinned).
- e2e: keyboard-only journey — j/k → s → extend → Enter → Comment → note →
  marker appears; then n → edit → marker text updates; delete → marker
  gone; trajectory event unchanged.

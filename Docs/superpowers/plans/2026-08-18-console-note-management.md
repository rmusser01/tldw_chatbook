# Console Review-Note Management (Riders) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Per-note Edit and soft-Delete for Console review notes, reached from the ✎ marker (click) or `n` (keyboard) via a notes modal — the spec's Part 2, completing task-17169's deferred Option-B management surface.

**Architecture:** A tiny `ConsoleAnnotationMarker(Static)` replaces the anonymous marker Static and posts a request message (fixing the phase-4 papercut: marker clicks currently toggle message selection). A new `ConsoleReviewNotesModal` lists the anchored message's notes with per-note Edit (prefilled TextArea → `upsert_transcript_annotation(annotation_id=...)`) and Delete (`ConfirmationDialog` → `soft_delete_transcript_annotation`). The screen fetches rows off-thread, pushes the modal in a worker, and forces the preview map to reload on change so the marker updates or disappears.

**Tech Stack:** Python 3.11+, Textual 8.x, pytest via `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <target> -q -p no:randomly`.

## Global Constraints

- Spec: `Docs/superpowers/specs/2026-08-18-console-keyboard-selection-and-note-management-design.md` Part 2. Binding clauses: modal follows the task-16211 safe-dismissal grammar; Delete gets a confirmation (ADR-031 destructive rule); the QUOTE is immutable (edit touches the comment only); **the sidecar `user_feedback` audit events are NEVER touched** — pin this with a test, and the user guide states the divergence.
- Branch `feat/console-note-management` (exists, tracks origin/dev). Every Bash command prefixed with ``; commits only via the branch-guard compound form (`[ "$(git branch --show-current)" = "feat/console-note-management" ] && git add ... && git commit ...`). NEVER pip install. NEVER bare `git stash` (shared stack across worktrees — use patch files).
- DB accessors already exist and are tested (`upsert_transcript_annotation` — upsert BY annotation id, `get_transcript_annotations(conversation_id)` → list[dict] with `annotation_id/row_key/message_id/quote_text/comment/created_at/updated_at`, `soft_delete_transcript_annotation(annotation_id) -> bool`).
- Screen state that exists: `_console_annotation_previews: dict[native_id, tuple[str,...]]`, `_console_annotation_loaded_conversation`, loader `_load_console_annotation_previews`, discovery `_sync_console_annotation_discovery` (reload trigger = set `_console_annotation_loaded_conversation = None`). Transcript holds `_annotation_previews` (same shape) via `set_annotation_previews`.
- File the backlog task FIRST (Task 1 Step 0) with a freshly swept ID (known floor 18315+; sweep every remote ref + worktree at filing time).

---

### Task 1: Clickable marker + `n` action

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py`
- Test: `Tests/UI/test_console_annotation_markers.py` (append)

**Interfaces:**
- Produces: `ConsoleReviewNotesRequested(Message)` with `anchor_message_id: str` (NATIVE id) — defined in `console_transcript.py` near the marker builder; `ConsoleAnnotationMarker(Static)` (same id/classes as today's Static: `console-annotations-{message.id}`, class `console-transcript-annotations`, plus stored `anchor_message_id`); transcript BINDINGS gains `("n", "open_review_notes", "Notes")` + `action_open_review_notes`.

- [ ] **Step 0:** File the backlog task (fresh sweep; title "Console review-note management modal"; ACs: marker click opens the notes modal instead of toggling message selection; `n` on a noted selected message opens it, un-noted toasts; per-note edit persists comment only; per-note delete confirms then soft-deletes; last-note delete removes the marker; sidecar events untouched (test-pinned); docs updated). Commit.
- [ ] **Step 1: Failing tests** (append; reuse the file's `_message` helper + transcript harness):

```python
@pytest.mark.asyncio
async def test_marker_click_requests_notes_not_message_toggle():
    app = ...  # transcript app with one message m1 + previews {"m1": ("note",)}
    # capture ConsoleReviewNotesRequested at app level (subclass handler)
    # click the marker widget's region center via pilot.click(f"#console-annotations-m1")
    # assert: one request with anchor_message_id == "m1"; selected_message_id UNCHANGED

@pytest.mark.asyncio
async def test_n_on_selected_noted_message_requests_notes():
    # selected_message_id = "m1", previews set, transcript focused, press "n"
    # assert one request

@pytest.mark.asyncio
async def test_n_without_notes_toasts_and_requests_nothing():
    # notify captured; press "n" on un-noted selection -> no request, one toast
```

- [ ] **Step 2: RED** (no `ConsoleAnnotationMarker` / no `n` action).
- [ ] **Step 3: Implement.** Marker widget:

```python
class ConsoleAnnotationMarker(Static):
    """Inline review-note marker; click opens the notes modal (Part 2).

    Phase 4 shipped this as an anonymous Static NOT in
    PROTECTED_CLICK_CLASSES, so clicking it toggled message selection --
    the papercut this widget closes.
    """

    def __init__(self, renderable, *, anchor_message_id: str, **kwargs) -> None:
        super().__init__(renderable, **kwargs)
        self.anchor_message_id = anchor_message_id

    def on_click(self, event: Click) -> None:
        event.stop()
        self.post_message(ConsoleReviewNotesRequested(self.anchor_message_id))
```

  Swap the `_build_row_widget` annotations branch to construct it (same id/classes/renderable). Add `"console-transcript-annotations"` to `PROTECTED_CLICK_CLASSES` (belt for the capture-reroute path). `action_open_review_notes`: if `self._annotation_previews.get(self.selected_message_id)` truthy → `self.post_message(ConsoleReviewNotesRequested(self.selected_message_id))`, else `self.notify("No review notes on this message.", severity="warning")`. `n` stays a plain BINDINGS entry — the phase-5 probe proved printable-key bindings fire when the transcript holds focus (the speculative on_key branch was reverted as unnecessary; do NOT add one).
- [ ] **Step 4: GREEN + neighbors** (`test_console_annotation_markers.py`, `test_console_citation_sources.py`).
- [ ] **Step 5: Ruff; commit.**

---

### Task 2: ConsoleReviewNotesModal

**Files:**
- Create: `tldw_chatbook/Widgets/Console/console_review_notes_modal.py`
- Test: `Tests/UI/test_console_review_notes_modal.py` (new)

**Interfaces:**
- Produces: `ConsoleReviewNotesModal(SafeModalDismissMixin, ModalScreen[bool])` — ctor `(notes: list[dict], on_edit: Callable[[str, str], bool], on_delete: Callable[[str], bool])` where notes are the DB dicts; `on_edit(annotation_id, new_comment) -> bool`, `on_delete(annotation_id) -> bool` are SYNC callables the SCREEN provides (screen wraps its own off-thread execution; the modal never imports DB code). Dismiss result: True if anything changed (screen reloads previews).
- Layout: one row per note — comment (multi-line Static), dim quote preview + created date, `Edit` and `Delete` buttons. Edit swaps the Static for a prefilled `TextArea` + Save/Cancel; Save calls `on_edit`, updates the row in place. Delete pushes `ConfirmationDialog` (`from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog`, `ModalScreen[bool]`, Escape = safe cancel) via `push_screen_wait` in a worker-safe way — the modal's own handler uses `self.app.push_screen` with a callback (NOT push_screen_wait; the modal is not in a worker). On confirmed delete: call `on_delete`, remove the row; when the last note goes, dismiss(True).
- Escape/backdrop/Close: dismiss(changed_so_far) — a mid-EDIT Escape first cancels the open editor (transient-surface-first), second Escape dismisses. Quote is rendered read-only, never editable.

- [ ] Steps: failing tests (rows render comments+quotes; edit round-trip calls on_edit with the id and new text and re-renders; edit-cancel restores; delete shows ConfirmationDialog, cancel = no on_delete call, confirm = called + row gone; last delete dismisses True; Escape layering) → RED → implement → GREEN → ruff → commit.

---

### Task 3: Screen wiring + unmocked round trips

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Test: `Tests/UI/test_console_annotation_markers.py` (append; reuse its real-DB harness pattern)

**Interfaces:**
- `@on(ConsoleReviewNotesRequested)` handler → worker (`group="console-review-notes"`, `exit_on_error=False`): resolve native→persisted id via the store's messages; `rows = await asyncio.to_thread(db.get_transcript_annotations, conversation_id)` filtered to that persisted id; empty → toast; else `changed = await self.app.push_screen_wait(ConsoleReviewNotesModal(rows, on_edit=..., on_delete=...))`; if changed → `self._console_annotation_loaded_conversation = None` (forces the existing discovery to reload on the next sync tick) + immediate `self._sync_console_annotation_discovery(store)`.
- `on_edit` wraps `db.upsert_transcript_annotation(conversation_id=..., row_key=row["row_key"], message_id=row["message_id"], quote_text=row["quote_text"], comment=new_comment, annotation_id=annotation_id)`; `on_delete` wraps `db.soft_delete_transcript_annotation`. Both run the DB call via `asyncio.to_thread`? NO — the modal calls them synchronously from UI handlers; keep them as thin sync calls (single-row indexed writes, the annotation-write precedent) and note it; never raise (log + return False).
- [ ] Tests (unmocked, real `CharactersRAGDB` via the marker file's harness): edit persists the new comment and ONLY `updated_at`/`comment` change (quote/row_key byte-identical); delete removes the row and, for the last note, the transcript marker disappears after the forced reload; **sidecar pin**: a `user_feedback` trajectory row written alongside the annotation is byte-identical after edit AND delete.
- [ ] RED → implement → GREEN (+ `test_console_selection_end_to_end.py -k create_note` unchanged) → ruff → commit.

---

### Task 4: Docs, live verification, wrap

- [ ] User guide `console/text-selection-and-feedback.md`: "Managing review notes" section (click the marker or press `n`; edit changes the note, the quote is fixed; delete confirms; **the trajectory ledger keeps the original feedback event — by design**, per ADR-068 amendment 4/5 lineage). Update the stamp.
- [ ] ADR-068: one-paragraph amendment noting the management surface shipped and the sidecar-immutability pin.
- [ ] Full sweep: marker + modal + selection e2e + keyboard files; ruff; stash-free baseline comparison for any failure.
- [ ] LIVE tmux verification (memory recipes; scratch profile, llama.cpp :9191): create a note via Comment, click the ✎ marker → modal, edit → marker text updates, delete → confirm → marker gone, trajectory viewer still shows the original feedback event. `n` path: select message, press n. Kill server, delete profile.
- [ ] Task Done + notes; push; PR against dev; STOP for maintainer instruction before merging (standing flow applies only when explicitly given).

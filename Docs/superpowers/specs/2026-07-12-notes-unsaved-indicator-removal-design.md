# Retire the unreachable notes unsaved-indicator CSS/watchers (task 173)

**Status:** Design approved (brainstorm — direction: remove), pending spec review.
**Backlog:** task-173 — "Retire or reconnect the unreachable notes unsaved-indicator CSS/watchers".
**Builds on:** the F2/T169 CSS sweeps that pruned the sibling legacy Notes-tab CSS but *kept* this family "since Python still names them" (per the in-file comment).

## Problem

A cluster of notes "unsaved / auto-save" indicator machinery survives only as dead code, left in place by earlier CSS sweeps because `app.py` still grep-referenced it:

- Two reactives on `TldwCli`: `notes_unsaved_changes: reactive[bool]` (`app.py:2280`) and `notes_auto_save_status: reactive[str]` (`app.py:2289`).
- Two watchers: `watch_notes_unsaved_changes` (`app.py:6623-6639`) and `watch_notes_auto_save_status` (`app.py:6641-6666`), which `query_one("#notes-unsaved-indicator")` and toggle `.has-unsaved`/`.auto-saving`/`.saved` classes.
- A CSS family `.unsaved-indicator{,.has-unsaved,.auto-saving,.saved}` authored in `css/features/_notes.tcss:15-40` and compiled into `css/tldw_cli_modular.tcss` (the file the app loads).

## Adjudication: remove (nothing live to reconnect)

The task asks to remove **or** reconnect. The evidence is one-sided for removal:

1. **`#notes-unsaved-indicator` is never composed** — the only references are the two watchers' `query_one`, which always raise `QueryError` (caught → no-op).
2. **The reactives are never mutated by any real code path** — grepping `notes_unsaved_changes`/`notes_auto_save_status` (excluding their declarations and the watchers) yields only self-references *inside* the two watchers. They never leave their `False`/`""` defaults, so the watchers never meaningfully fire.
3. **The Library notes editor already has its own save-state UI** — `library_notes_canvas.py` composes a `#library-note-save` button, a conflict banner ("This note changed elsewhere — Overwrite saves your text"), and a `#library-notes-sync-auto` auto-sync toggle. It does not use this indicator and does not need it.

"Reconnecting" would mean *building a new unsaved-indicator feature* (widget + save-state tracking) for the Library editor — net-new work that duplicates existing affordances. Therefore: remove the whole family together.

## Goal / Acceptance

- **AC1** — The unsaved-indicator CSS + watchers are removed together (the reactives, the watchers, and the CSS family), and **no dead grep-live references remain**.

## Components (all removals)

1. **`app.py` reactives** — delete `notes_unsaved_changes` (`:2280`) and `notes_auto_save_status` (`:2289`). Leave the sibling reactives in those blocks untouched (`notes_sort_by`, `notes_preview_mode`, `notes_auto_save_enabled`, `notes_auto_save_timer`, `notes_last_save_time`) — they are out of scope (the real Library auto-save may use them; a separate adjudication if ever needed).
2. **`app.py` watchers** — delete `watch_notes_unsaved_changes` (`:6623-6639`) and `watch_notes_auto_save_status` (`:6641-6666`) in full.
3. **`css/features/_notes.tcss`** — delete the "Auto-save status indicators" comment + the four `.unsaved-indicator*` rules (`:15-40`), leaving the surrounding Notes-feature comments intact.
4. **Regenerate `css/tldw_cli_modular.tcss`** — run `python tldw_chatbook/css/build_css.py`. A fresh build was verified to reproduce the committed modular file exactly except its `Generated:` timestamp, so the resulting diff is just the removed `.unsaved-indicator` block plus that one timestamp line.

## Data flow

None — this is pure dead-code removal. Nothing ever set the reactives, so removing them changes no runtime behavior (the watchers only ever no-op'd on `QueryError`).

## Error handling

N/A. Removing a reactive that had a `watch_` method is safe because nothing outside the removed watchers references it; Textual auto-wires `watch_<name>` only for reactives that still exist.

## Testing

Because this is a removal, the AC ("no dead grep-live references remain") is itself the test seam:

- **Grep-clean guard (RED→GREEN):** a unit test that scans `tldw_chatbook/**/*.py` and `tldw_chatbook/css/**/*.tcss` and asserts none of these dead identifiers appear: `notes_unsaved_changes`, `notes_auto_save_status`, `watch_notes_unsaved_changes`, `watch_notes_auto_save_status`, `notes-unsaved-indicator`, `unsaved-indicator`. RED before removal (all present), GREEN after. This directly encodes AC1 and guards against reintroduction.
- **Import smoke:** `python -c "import tldw_chatbook.app"` — proves removing the reactives/watchers didn't break the `TldwCli` class body.
- **App construct + CSS-parse smoke:** construct `TldwCli()` (or lean on any existing modular-CSS-parse test) to confirm `tldw_cli_modular.tcss` still parses after the rebuild. Removing unused CSS rules cannot break rendering (no widget referenced them), so this is a safety check, not a behavior assertion.
- **Build determinism:** after editing `_notes.tcss` and running `build_css.py`, `git diff` on `tldw_cli_modular.tcss` shows only the removed block + the `Generated:` timestamp line (no unrelated churn).

## Scope / non-goals

- Do **not** touch the Library notes editor's real save/conflict/auto-sync UI (`library_notes_canvas.py`) — it is the live replacement and is unaffected.
- Do **not** remove the sibling auto-save reactives (`notes_auto_save_enabled`/`notes_auto_save_timer`/`notes_last_save_time`) or the `auto_save_delay_ms` config — out of scope; they belong to a potentially-live auto-save path and would need their own adjudication.
- No new features, no widget changes, no reconnect.

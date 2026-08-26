# Console Text Selection, Selection Menu, Side Chat, and Annotations — Design

Date: 2026-08-14
Status: Approved design (pending spec review)
Inspirational source: plannotator (interaction patterns only; no dependency)

## Summary

Bring codex/plannotator-style review interactions to the Console screen:

1. **Character-level text selection** in the transcript (mouse-drag primary, keyboard fallback later), with a stacked options menu popping up at the selection: `Add to chat | More Details | Ask in Side Chat`.
2. **Ephemeral side chat** ("More Details" / "Ask in Side Chat"): an unsaved modal chat that answers questions about the selected text using a user-configured model and prompt template.
3. **Agent feedback + persisted annotations**: when the selection sits in agent/sub-agent plan/tool/diff output, the menu additionally offers `Request changes | LGTM | Comment`; the first two route structured feedback to the agent session, `Comment` also persists as a message/session-anchored annotation.

Not in scope (v1): multi-row selections, cross-session artifact browser, external/browser reviewer integration (plannotator itself), exporting annotations.

## Approach Chosen

**Per-row selection delegates.** Each transcript row widget owns its text; a `SelectionManager` on `ConsoleTranscript` coordinates drag hit-testing, asks rows to render highlights via a small protocol, and triggers the menu on release.

Rejected alternative: a transcript-wide virtual text buffer (plain start/end offsets into flattened text). Simpler selection math, but it maintains a shadow copy of every row's text that drifts during streaming/reflow, and markdown/diff rows need custom flatten/unflatten anyway — the drift problem gets worse, not better.

## 1. Selection System

New module: `tldw_chatbook/Widgets/Console/console_selection.py`.

- `SelectionManager` — owns the active selection as `(row_key, start_offset, end_offset)`; handles `MouseDrag`/`MouseMove`/`MouseUp` coordination across rows.
- `SelectableRow` protocol (implemented by `ConsoleMarkdownMessage`, `ConsoleTranscriptMessage`, `ConsoleToolDiffRow`, and other text rows):
  - `get_display_text() -> str` — the rendered plain text (see "Selection domain" below).
  - `apply_selection_highlight(start, end)` / `clear_selection_highlight()`.
  - `get_selection_text(start, end) -> str`.

Decisions:

- **Selection domain = rendered/displayed text, per row.** Offsets map to what is on screen; the same string is what gets quoted, sent to the side chat, or stored in an annotation. We do NOT map back to markdown source (headings, wrapping, code fences, rich markup make that fragile and it buys nothing for the user-facing actions).
- **Single-row only (v1).** Dragging across row boundaries clamps to the origin row. Multi-row selection is a possible future task.
- **Click vs. drag disambiguation is mandatory.** The manager arms "drag mode" after ≥1 cell of movement; rows suppress their existing `on_click` message-selection toggle whenever the manager reports an active or just-finished drag. Without this, sloppy clicks toggle message selection and clear selections simultaneously.
- **Streaming behavior.** Selections on actively-streaming rows clamp to the last stable text; if the row is replaced, the selection clears. Rows own their clamp logic (per-row delegate benefit). Row repaints during the 0.1–0.2s transcript sync tick must preserve highlight styles (refresh, not recompose, for selected rows).
- **Non-selectable rows.** Banners, action rows, scrollbars, and other `PROTECTED_CLICK_CLASSES`-style regions never start a selection.
- **Terminal interplay.** Plain drag reaches the app (Textual mouse reporting); shift+drag remains terminal-native copy. Verify with an early spike test in phase 1.
- **Selection size cap.** Actions quote at most a capped number of characters (e.g. 4,000; configurable constant) — larger selections are truncated with an ellipsis marker. Prevents blowing the composer draft, side-chat prompt, or stored annotation with a whole-file dump.
- **Keyboard fallback (phase 5).** *(Amended 2026-08-18: shipped SINGLE-ROW per the maintainer's scope decision — see the 2026-08-18 keyboard-selection design spec and ADR-068 amendment 5; the row-range wording below is superseded.)* shift+j/k grows a row-range over the existing j/k selection; entering a row activates character mode. Reuses the same `SelectionManager` and menu so all actions are shared. Keybindings must follow ADR-031 (htop-style single letters, no terminal-convention keys, footer hints truthful 1:1 — test-enforced).

### Selection Menu

`ConsoleSelectionMenu` — a small floating widget mounted in an overlay region of the transcript (same pattern as `ConsoleTranscriptJumpPill`, which already floats over the transcript), anchored at the mouse-release cell, offering a vertically stacked list:

- `Add to chat`
- `More Details`
- `Ask in Side Chat`
- On agent/sub-agent plan/tool/diff rows additionally: `Request changes` / `LGTM` / `Comment` (see §3).

Not a `ModalScreen`: modals are layer-centered and cannot anchor at a screen cell. The Escape/backdrop contract from task-16211 is implemented on the floating widget (Escape and click-outside dismiss; no side effects on dismiss). Dismiss also clears the selection highlight unless an action consumed it.

## 2. Menu Actions

- **Add to chat** — insert the quoted selection into the composer draft via a new public insert method on `ConsoleComposerBar`'s draft-segment API. Insert at the caret if the composer holds focus; otherwise append at the end as a quoted block (`> ` prefixed lines). Composer stays/becomes unfocused-preserving (do not steal focus).
- **More Details** — open `ConsoleSideChatModal` (a true `ModalScreen` per convention) and auto-send the customizable template, default `"Give me more details about: {selection}"`, to the side-chat model. Streaming reply display; ephemeral — nothing persisted to DB; reply buffer capped/truncated (e.g. last N chars / token cap) so long streams can't balloon memory.
- **Ask in Side Chat** — same modal; selection pre-quoted, prompt empty for freeform input.

### Settings

New console settings entries (canonical surface: `UI/Screens/settings_screen.py` / console settings modal; nothing in deprecated settings surfaces):

- `[console] sidechat_model` (provider + model identifier; defaults to the current session model if unset).
- `[console] sidechat_prompt_template` (default as above; `{selection}` placeholder).

### Side-chat execution isolation

The side chat runs in its own `run_worker(..., exclusive=False)` with its own conversation context — it must not cancel, block, or share streaming state with an active console agent run. It calls the provider through the standard `LLM_Calls` path with its own ephemeral message list (system prompt + `{selection}` exchange only; no console session context is attached, keeping it stateless and cheap).

## 3. Feedback and Annotations

On selections inside agent/sub-agent plan/tool/diff rows, the menu additionally offers:

- **Request changes** / **LGTM** — compose a structured feedback message (quoted selection + optional comment gathered via a small input modal) routed as the next user message to the agent session, reusing the existing composer send path.
- **Comment** — same structured message option, and additionally persists an annotation.

Rules:

- **No active run fallback:** when no agent run is active, `Request changes` and `LGTM` are disabled with a visible hint; `Comment` and the three selection actions remain available.
- **Persistence:** new table (schema v8 bump, migration added per conventions):

  ```sql
  transcript_annotations(
      annotation_id TEXT PRIMARY KEY,
      session_id TEXT NOT NULL,      -- console session
      row_key TEXT NOT NULL,         -- stable transcript row identity
      message_id TEXT NULL,          -- set when the row is a DB message
      quote_text TEXT NOT NULL,
      comment TEXT NOT NULL,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
  )
  ```

  Soft-delete per DB conventions. Render as an inline badge/marker on the anchored row with a viewer popover. The stable anchor is `(session_id, row_key)`; `message_id` is nullable because diff/tool rows are not DB messages.

  **`row_key` derivation must be deterministic across restarts.** A runtime widget/row identity regenerated on session reload would orphan annotations. `row_key` is derived from persisted data — e.g. `message:<db_message_id>` for message rows, `tool:<tool_call_id>` / `diff:<tool_call_id>:<hunk>` for tool/diff rows — never from Python object identity or mount order. Phase 4 starts with a spike that inventories the row kinds and confirms each has (or can cheaply gain) a durable key; any row kind without one is excluded from annotation (selectable but comment-persistence disabled with a hint).

## 4. Error Handling

- Selection on protected/non-text rows: no-op.
- Side-chat LLM failures: surfaced inline in the modal with retry; never crash the console session; modal remains escapable.
- Annotation DB write failures: transient notice; never block the feedback send.
- Feedback send failures ride the existing composer/send error paths.

## 5. Testing

- Unit: `SelectionManager` anchor/clamp/threshold math; `SelectableRow` protocol per widget type; annotation DB round-trip (in-memory SQLite); template rendering; composer insertion (caret and append modes).
- Widget tests: menu anchor/dismiss contract (Escape, click-outside), click-vs-drag suppression, footer-hint truthfulness (ADR-031 test), streaming clamp behavior.
- No live-LLM tests in CI; side-chat streaming is mocked.

## 6. ADR

ADR required: yes — new DB table/schema bump (annotation persistence) and a new cross-widget selection interface.

```text
ADR required: yes
ADR path: backlog/decisions/0NN-console-text-selection-and-annotations.md (number assigned at creation)
Reason: schema/storage change (annotations table, schema v8) plus a cross-module selection interface implemented by multiple transcript row widgets; references ADR-031 for keybinding additions.
```

The ADR is created before implementation begins (phase 1) and linked from the Backlog task, this spec, and the implementation plan.

## 7. Phasing

1. Selection manager + mouse drag + click/drag suppression + menu widget + `Add to chat` (includes the Textual mouse-reporting spike).
2. `ConsoleSideChatModal` + settings (model, template) + `More Details` / `Ask in Side Chat`.
3. Feedback actions (`Request changes` / `LGTM` / `Comment` routing, no-active-run fallback).
4. Persisted annotations + schema v8 bump + badge/popover UI.
5. Keyboard selection fallback (shift+j/k row ranges, in-row character mode).

Each phase is independently valuable and testable.

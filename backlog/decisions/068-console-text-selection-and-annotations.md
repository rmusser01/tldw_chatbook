# ADR-068: Console Text Selection and Annotations

Status: Accepted
Date: 2026-08-14
Related Task: TASK-16481
Supersedes: N/A

## Decision

Console transcript text selection uses per-row selection delegates with a
transcript-level `SelectionManager`. Each row widget owns its text and its
highlight rendering; the manager, a pure-logic object with no Textual
imports, coordinates drag begin/extend/finish across rows and reports the
active or finished selection. The selection domain is the row's displayed
text: offsets map to what is on screen, and the same string is what gets
quoted, sent to a side chat, or stored in an annotation. There is no mapping
back to markdown source.

Granularity is character-level on plain rows and line-level on markdown rows
in phase 1 — cell-to-offset mapping through Textual's `Markdown` renderer is
not stable enough for character-level, so markdown drags snap to whole
rendered source lines; later phases may tighten this. Selections are
single-row only (v1): drags crossing row boundaries clamp to the origin row.
Selections on actively-streaming rows clamp to the last stable text, and a
selection on a replaced row clears.

The selection action menu is a floating widget mounted in the transcript's
overlay region (the jump-pill pattern), not a `ModalScreen`. It is anchored
at the mouse-release cell, overlay-docked so its offset is
container-relative. Escape and click-outside dismiss it with no side
effects, per the task-16211 modal contract applied to the floating widget.

`Add to chat` inserts the quoted selection into the composer draft at the
caret via the composer's existing `_insert_literal_at_cursor` seam (the
caret always exists in the segment model, so an unfocused composer appends
at the end of the draft). Quotes are capped
(`SELECTION_QUOTE_CAP = 4000`, truncated with an explicit marker).

Persisted annotations are anchored by a deterministic `(session_id,
row_key)` pair derived from persisted data (for example
`message:<db_message_id>` or `tool:<tool_call_id>`), never from Python
object identity or mount order. The annotations table and its schema bump
are deferred to phase 4, which begins with a spike inventorying row kinds
for durable keys; any row kind without one is excluded from annotation
persistence.

No new keybindings are introduced in phase 1 (keyboard selection is
phase 5), so ADR-031 compliance is unchanged.

## Context

The Console transcript is a custom stack of row widgets — plain text rows
and markdown rows, plus banners, tool rows, and other non-text regions —
not a single text surface. Textual provides no character-level selection
over such a stack, so the Console currently has no way to select part of a
message. Review-style workflows (the plannotator/codex pattern) are
select-then-act: quote a span into the composer, ask about it, leave
structured feedback on it. Without a selection domain, none of those
interactions can be built on the transcript.

Because each row kind renders text differently (plain `Static`, `Markdown`,
diff fences), any selection system must let rows own both their text and
their highlight rendering while something above them coordinates the drag
across the stack.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Transcript-wide virtual text buffer (flat start/end offsets into flattened text) | Simpler selection math, but it maintains a shadow copy of every row's text that drifts during streaming and reflow, and markdown/diff rows need custom flatten/unflatten anyway — the drift problem gets worse, not better. |
| `ModalScreen` for the selection menu | Modals are layer-centered and cannot anchor at a screen cell; the menu must appear at the release cell next to the selection. |
| Keyboard-only selection | Not the codex/plannotator interaction; mouse-drag is primary. Keyboard fallback is deferred to phase 5 rather than rejected. |

## Consequences

- The selection core (`console_selection.py`) is pure logic with no Textual
  imports, unit-testable without widget mounts; row widgets and the
  transcript feed it events and render what it produces.
- Rows gain a small selection protocol (display text, highlight range,
  selection text); protected/non-text rows never start a selection.
- Click versus drag must be disambiguated: drag mode arms after movement,
  and rows suppress their existing `on_click` message-selection toggle
  during or just after a drag.
- The floating menu means no new modal layer and no focus steal; dismissal
  follows the established Escape/backdrop contract.
- Annotation persistence (schema bump, badge/popover UI) is explicitly
  phase 4; phases 1–3 ship selection, menu, side chat, and feedback routing
  without schema changes.
- Markdown rows are selectable at line granularity only until a later
  phase tightens cell-to-offset mapping; this is a recorded phase-1
  simplification, not a permanent ceiling. Implemented as decided in
  phase 1 (TASK-16481): `ConsoleMarkdownMessage` snaps selections outward
  to whole markdown source lines (`_snap_to_line_bounds`) and renders the
  highlight as a reverse-video `Static` strip below the Markdown widget
  (the Markdown renderer's block widgets are left untouched); pointer
  cells map to source lines by distributing the body-local row evenly
  across the source lines with a nearest-line clamp, because the
  `Markdown` widget does not expose per-source-line layout — the
  whole-line snap bounds the quoting error of that approximation. A
  streaming append that grows the markdown source re-snaps the stored
  line range, so a selection touching the last line intentionally GROWS
  with the stream, while plain rows hold their last stable range.
- Phase 2 (ephemeral side chat) landed: `More Details` / `Ask in Side
  Chat` push `ConsoleSideChatModal`, whose `ConsoleSideChatService` is
  gateway-only and persistence-free (`stream_chat` alone; the reply
  never leaves the modal), and whose worker runs in the
  non-exclusive `console-side-chat` group so it never cancels or
  blocks `console-run-{session_id}` session workers.
- Phase 3 (review feedback) landed: selections in agent output (tool
  and diff rows) offer `Request changes | LGTM | Comment` in the
  selection menu (run-gated with a visible hint when no run is
  active), and the composed feedback (action header + quoted
  selection + optional comment) routes as the next user message
  through the prompt-queue dispatch seam — it queues behind an active
  run and never touches the composer draft or `submit_draft` (which
  refuses during runs and would clobber in-progress typing). Feedback
  gating widened to assistant prose (ASSISTANT-role rows; USER rows still
  excluded) per product decision 2026-08-16. Phase 4
  adds annotation persistence.

## Links

- [Console Text Selection, Selection Menu, Side Chat, and Annotations — Design](../../Docs/superpowers/specs/2026-08-14-console-selection-annotations-design.md)
- [ADR-031: TUI Keybinding and Footer-Hint Conventions](031-tui-keybinding-and-footer-hint-conventions.md) — no new keybindings in phase 1
- [TASK-16481](../tasks/task-16481%20-%20Console-text-selection-phase-1.md)

## Amendments (2026-08-15, live-terminal spike)

The automated suites (pilot-driven, incl. a real-ChatScreen smoke test) all
passed while the feature was broken in real terminals. Three successive
user-run spike rounds found and fixed defects invisible to synthetic events:

1. **Real-terminal press events carry no widget.** Textual's screen
   forwarding dispatches MouseDown/MouseUp without setting `event.widget`
   (only the translated MouseMove path assigns one), so arming logic keyed
   on `event.control` no-oped in kitty/iTerm2/Terminal while pilot events
   stayed green. The arm now hit-tests from screen coordinates
   (`get_widget_at`), with `event.control` as the synthetic fallback.
2. **Markdown selections are character-level, not line-snapped.** Whole-line
   snapping made any partial drag on a one-line reply select the entire
   message. The cell-to-offset map already resolves character positions;
   ranges map verbatim (clamped), and the highlight strip spells the exact
   range out below the markdown body.
3. **Anchored floating widgets must use `position: absolute` +
   `absolute_offset`** (the tooltip mechanism, where region/paint/clip
   agree). `dock` + `styles.offset` paints translated while clipping and
   hit-testing at the un-translated dock slot -- one visible button, the
   rest obscured and unclickable. The menu mounts on the owning screen at
   the release cell, posts action/dismissal messages directly to the owning
   transcript, releases the transcript's tail-follow anchor for its
   lifetime, and supports keyboard navigation (first button focused,
   up/down cycle, Enter activates, Escape closes).
4. **Suppression of the synthesized drag-release Click** uses a dedicated
   one-shot token: Textual can synthesize the release Click late, after an
   intervening press consumed any shared flag.
5. **Idle Speak lives in the selected-message action row** (swaps to
   speak-stop while that message is speaking); the header hosts only
   active-playback lifecycle status.

### Amendment 3 (2026-08-16, live "black bar" spike): screen-mounted overlays must also carry `overlay: screen`

The phase-1 screen-mount mechanism (Amendment 2, item 3) had a hidden cost:
textual 8.2.8's vertical layout excludes `position: absolute` children from
sibling stacking but still feeds their height into the fr denominator, so the
menu silently subtracted its own height from `#screen-content`'s `1fr` --
with a selection menu open, the composer floated above dead rows between it
and the docked footer (the user-reported "black bar under the composer").
The fix adds `overlay: screen` to the menu's CSS: the style that removes an
overlay from the container's flow math entirely, while `position: absolute` +
`absolute_offset` continue to own anchoring, paint, and hit-testing. General
rule for this codebase: any widget mounted directly on a screen as an overlay
must set both `position: absolute` and `overlay: screen`. Regression test
`test_screen_mounted_menu_steals_no_flow_height` pins the 1fr budget against
a screen-shaped app. Attributed via a temporary F12 layout dump that listed
every direct screen child -- the screen-children view is what named the
consumer; the fixed-chain view could not.

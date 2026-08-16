# Trajectory View (Conversation Traces)

The trajectory view is a trace inspector for Console conversations: an
event ledger grouped by turn, with nested tool calls, per-record token and
timing facts, a brushable duration timeline, and live tail-follow while an
agent is working. It answers "what exactly happened in this conversation" —
every step, every tool call, every token — without scrolling the chat
transcript.

It is modeled on the trajectory screen in DeepSeek's
[deepseek-harness](https://github.com/deepseek-ai/deepseek-harness), adapted
to chatbook's Textual TUI.

## Opening it

With a conversation open in the **Console** (the chat screen), press
**`y`**. The footer there advertises `Y trajectory`. The view opens as an
overlay for the current conversation; `escape` closes it (pressing `escape`
while the search box is focused blurs the search first — press it twice).

The view is read-only: it never modifies the conversation.

## The ledger

The main table is one row per event, grouped into turns:

- **Turn headers** (`Turn 1 • 4 records`) start at each user message; `t`
  collapses or expands the focused turn.
- **Assistant steps** show token usage split into `In` (uncached input),
  `Cache` (cache read), and `Out` columns, plus `Start` / `Done` wall-clock
  timestamps.
- **Tool calls and results** are nested (indented `└` rows) under the
  assistant step that issued them.
- **Compaction markers** appear between turns when the context was
  compacted.
- Superseded regeneration **variants** are not separate rows; open the
  inspector on the turn's assistant record to list them.

Columns the data doesn't have are shown as `—`, never estimated: timing
facts are recorded only when they were actually observed.

### The inspector

Move the cursor to a row and press **`i`** (or `enter`) to open the
inspector pane: model and provider, the full token breakdown (input / cache
read / cache write / output), timing (step start → first token → completed,
with elapsed spans when both endpoints exist), superseded variants
(turn-level), and for tool records the full untruncated result. For
imported traces with redacted payloads, the inspector shows the redaction
marker instead of the payload.

### Search

`/` focuses the search box. Queries match record content **and** tool
payloads (so you can find a turn by a path a tool touched or text deep
inside a result). Matching collapses the ledger to hits; turn headers
survive while any child matches; clearing the box restores the view.

## The timeline

The strip above the ledger projects each timed record as a bar at its real
start/duration, colored by kind:

- **Drag** horizontally to brush a time range — the ledger filters to
  records active in that range (composed with the search filter). The
  caption shows the range and the number of active records.
- **Click** a bar to jump the ledger cursor to that record (clicking empty
  space clears the brush).
- **Wheel** zooms in/out centered on the mouse; when the timeline is
  focused, `[` / `]` zoom and `,` / `.` pan.
- The brush survives live refreshes while it still intersects the
  conversation's time span.

## Live tail-follow

With the conversation streaming, the view polls for new records and follows
the tail (auto-scrolling as the turn grows). Scrolling up suspends
following; **`f`** resumes it. Collapse state, search, and cursor position
are preserved across refreshes.

Long conversations open at the newest 500 records; **`e`** pages earlier
records in.

## Sharing traces: export and import

Trajectories can be exported to a single portable JSON file and opened on
another machine (or attached to a bug report):

- **Export** produces a versioned `tldw-trajectory` document containing
  everything the view renders. **Tool payloads are redacted by default** —
  only the tool name and short previews travel unless full payloads are
  explicitly opted in (they can contain file contents). The document-level
  `redacted` flag records the mode.
- **Import**: press **`o`** in the trajectory view to open a trace file. It
  renders as `Imported trace — <name> (read-only)`. Imported traces never
  write anything into your local database — messages, sidecar data, and
  sync state are untouched. Malformed files, unknown versions, and JSON
  errors are reported with actionable messages.

The export API lives in `tldw_chatbook/Chat/trajectory_export.py`
(`build_trajectory_export`, `write_trajectory_export`); import in
`Chat/trajectory_import.py`. The format contract is
[ADR-067](../../backlog/decisions/067-trajectory-export-format.md).

> **Note:** an export button in the UI is not wired yet (task-16813 landed
> the format and writer as a library). Import via `o` is wired.

## What older conversations show

Timing (start / first-token / completion) and tool records are captured as
they happen and stored in a local-only sidecar table
([ADR-066](../../backlog/decisions/066-console-trajectory-view-and-trace-metadata.md)).
Conversations from before this feature existed still open in the view —
grouped via a timestamp fallback — but their timing columns are blank, tool
steps are absent (tool output was session-only before), and the timeline
shows its `no timing data` placeholder. New turns get the full treatment.

## Key reference

| Key | Action |
| --- | --- |
| `y` (Console) | Open the trajectory view for the current conversation |
| `/` | Focus search |
| `t` | Collapse/expand the focused turn |
| `i` / `enter` | Toggle/open the inspector on the cursor row |
| `e` | Load earlier records |
| `f` | Resume tail-follow |
| `o` | Open an imported trace file (read-only) |
| `escape` | Close the view |

All bindings follow the repo-wide TUI keybinding conventions
([ADR-031](../../backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md));
the footer only advertises keys that work in the current context.

## Where the design lives

- Design spec: `Docs/superpowers/specs/2026-08-14-console-trajectory-view-design.md`
- Sidecar schema and view architecture: `backlog/decisions/066-console-trajectory-view-and-trace-metadata.md`
- Export format: `backlog/decisions/067-trajectory-export-format.md`

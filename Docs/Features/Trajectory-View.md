# Trace View

Trace is the Console's event inspector: an
event ledger grouped by turn, with nested tool calls, per-record token and
timing facts, a brushable duration timeline, and live tail-follow while an
agent is working. It answers "what exactly happened in this conversation" —
every step, every tool call, every token — without scrolling the chat
transcript.

It is modeled on the trace screen in DeepSeek's
[deepseek-harness](https://github.com/deepseek-ai/deepseek-harness), adapted
to chatbook's Textual TUI.

## Opening it

With a conversation open in the **Console** (the chat screen), press
**`y`**. The footer there advertises `Y trace`. Trace opens as an
overlay for the current conversation; `escape` closes it (pressing `escape`
while the search box is focused blurs the search first — press it twice).

The view is read-only: it never modifies the conversation.

The title uses the conversation's human-readable name. The raw conversation
identifier stays in the inspector metadata, where it is available for
diagnosis without dominating the first scan.

## Status and recovery

The line under the title makes the current mode explicit:

- `LIVE · FOLLOWING` follows new events; scrolling up changes this to
  `LIVE · PAUSED`, and `f` resumes.
- Imported files show `READ-ONLY SHARED TRACE` and never write local data.
- `LOADING`, `INCOMPLETE`, `NO TIMING`, `EMPTY`, and `NO MATCHES` explain why
  information is absent instead of leaving a blank ledger.
- Search, structured filters, and time ranges label **Shown**, **Matches**, and
  **Total** separately, so pagination is never mistaken for the complete
  result set; `x` clears all of them.
- A failed ledger render or live refresh shows `FAILED · … · r retry` without
  exposing payload or exception text. While the single retry worker runs, the
  state changes to `RETRYING` and another retry cannot be started.

## The responsive ledger

The main table is one row per event, grouped into turns:

- **Turn headers** (`Turn 1 • 4 records`) start at each user message; `t`
  collapses or expands the focused turn.
- At 60–99 columns, the primary `#`, `Event`, `Summary`, and `State` facts
  remain visible with no horizontal scrolling.
- At 100–119 columns, compact `Tokens` and `Duration` metrics appear.
- At 120 columns and wider, usage splits into `In`, `Cache`, and `Out`, with
  duration and observed start/completion clocks.
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

Move the cursor to a row and press **`i`** (or `enter`) to open and focus the
independently scrollable inspector pane: model and provider, the full token breakdown (input / cache
read / cache write / output), timing (step start → first token → completed,
with elapsed spans when both endpoints exist), superseded variants
(turn-level), and for tool records the full untruncated result. For
imported traces with redacted payloads, the inspector shows the redaction
marker instead of the payload. Other structured event payloads are shown as
JSON so retrieval, context, and agent facts are not dropped. `▼ more — scroll…` appears only while content
remains below the fold. Press **`d`** for a reversible full-pane detail view;
press `d` again to return to the ledger. Live refresh keeps the reading
position when the same stable event remains selected.

### Search

`/` focuses the search box. Queries match record content **and** tool
payloads (so you can find a turn by a path a tool touched or text deep
inside a result). Matching collapses the ledger to hits; turn headers
survive while any child matches. Press `x` to clear search, structured, and
timeline filters together. While the search box is focused, `x` remains
ordinary query text; press `escape`, then `x`, to clear filters.

### Structured filters

Filter by event kind, state, agent run, and provider; all selected dimensions,
search, and the timeline range combine with AND semantics. The Agent options
name actual primary/child run identities rather than model or user actors. At
100 columns and wider, the four native selectors and three explicit count rows
stay above the ledger. On narrower terminals, focus the compact filter row and
press **`enter`**, or press **`g`** anywhere, to edit the same state; its
one-line summary shows shown, matching, and total counts plus the number of
active filters. `x` is the single clear-all action for search, structured
filters, and the timeline range.

## The timeline

The strip above the ledger projects each timed record at its real
start/duration in named **Input**, **Model**, **Tools**, and **Agents** lanes.
Distinct glyphs remain recognizable in monochrome; color is secondary:
`◆` input / `◇` feedback, `━` model / `!` error, `▶` tool call / `◀` result,
and `●` agent run / `○` agent step. When an instantaneous child event shares
its boundary cell, `◉` preserves run and `⊙` preserves step meaning. Turn and
child-agent boundary marks show
grouping without implying a fabricated serial dependency:

- **Drag** horizontally to brush a time range — the ledger filters to
  records active in that range (composed with the search filter). The
  caption shows the range and the number of active records.
- **Click** a bar to jump the ledger cursor to that record. An accepted jump
  outside the active time range clears the range atomically; search or another
  structured filter can still reject the jump without changing selection.
  Clicking empty space clears the brush.
- **Wheel** zooms in/out centered on the mouse; when the timeline is
  focused, `[` / `]` zoom and `,` / `.` pan.
- With timeline focus, **`j` / `k`** select the next/previous timed event,
  **`enter`** jumps the ledger to it, and **`b`** starts or ends a range.
  `escape` clears an active range/anchor before it can close Trace.
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

Traces can be exported to a single portable JSON file and opened on
another machine (or attached to a bug report):

- **Export**: press **`w`** in Trace. The privacy preflight stays visible while
  you choose one of three profiles and a destination:
  - **Safe summary** keeps causal structure, state, and coarse timing while
    omitting payload bodies.
  - **Redacted diagnostic** is the recommended default. It keeps useful
    debugging context while governing paths, identifiers, and sensitive values.
  - **Full trace** keeps ordinary captured detail and requires a second explicit
    warning confirmation.
- The preflight reports event and field-level sensitive, redacted, omitted,
  truncated, and unavailable counts before any file is written. Credentials are
  prohibited in every profile. Existing files require a separate Replace
  confirmation.
- Export writes a versioned `tldw-trace` v2 JSON bundle with stable event
  identity/order, causal lineage, timing, missing-data reasons, privacy
  provenance, and a canonical SHA-256 digest.
- **Import**: press **`o`** in Trace and choose **Import**. Validation runs in
  the background, then opens `Shared trace — <name>` with
  `READ-ONLY SHARED TRACE · NOT SAVED`. Imported traces are ephemeral and never
  write messages, annotations, trajectory metadata, or agent-run data locally.
  The synthetic final `Trace import` event exposes the safe manifest, complete
  privacy inventory, digest verdict, and source-authenticity limitation.
- `DIGEST VALID` means the bundle content matches its included SHA-256 digest;
  it does **not** verify the sender. The screen therefore also states
  `SOURCE NOT AUTHENTICATED`. Malformed files, unknown versions, tampering, and
  credential-bearing bundles fail closed with actionable errors. Legacy v1
  `tldw-trajectory` files remain importable and say that no digest was provided.

The v2 export API lives in `tldw_chatbook/Chat/trajectory_export.py`
(`preflight_trace_export`, `build_trace_export`, `write_trajectory_export`);
version-dispatched import lives in `Chat/trajectory_import.py`. The current
collaboration contract is [ADR-080](../../backlog/decisions/080-trace-v2-exhaustive-event-projection-and-collaboration.md);
[ADR-067](../../backlog/decisions/067-trajectory-export-format.md) remains the
legacy v1 contract.

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
| `y` (Console) | Open Trace for the current conversation |
| `/` | Focus search |
| `t` | Collapse/expand the focused turn |
| `i` / `enter` | Toggle/open the inspector on the cursor row |
| `d` | Toggle full-pane detail while the inspector is open |
| `e` | Load earlier records |
| `f` | Resume tail-follow |
| `r` | Retry a failed render or live refresh |
| `g` (or `enter` on the compact filter row) | Open structured filters |
| `x` | Clear search, structured filters, and timeline range |
| `w` | Export with privacy preflight |
| `o` | Open an imported trace file (read-only) |
| `n` / `p` | Next / previous visible match |
| `j` / `k` | Next / previous error (or timed event with timeline focus) |
| `u` / `y` | Next / previous tool event |
| `v` / `b` | Next / previous feedback event |
| `a` / `s` | Next / previous child-agent event |
| `[` / `]` | Zoom out / in with timeline focus |
| `,` / `.` | Pan left / right with timeline focus |
| `escape` | Close the view |

All bindings follow the repo-wide TUI keybinding conventions
([ADR-031](../../backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md));
the footer only advertises keys that work in the current context.

## Where the design lives

- Design spec: `Docs/superpowers/specs/2026-08-14-console-trajectory-view-design.md`
- Sidecar schema and view architecture: `backlog/decisions/066-console-trajectory-view-and-trace-metadata.md`
- Collaboration export/import v2: `backlog/decisions/080-trace-v2-exhaustive-event-projection-and-collaboration.md`
- Legacy export format v1: `backlog/decisions/067-trajectory-export-format.md`

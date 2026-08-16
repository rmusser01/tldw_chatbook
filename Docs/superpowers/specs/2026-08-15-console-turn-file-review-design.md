# Console Turn File Review — Design

Date: 2026-08-15
Status: Approved in brainstorming (owner), pending spec review
Scope: V1 (read-only per-turn file cards). V1.5/V2 recorded as boundaries.

## Purpose

When an agent turn edits files, the Console transcript today shows only
generic tool call/result messages. The user cannot see, at a glance, which
files a turn touched, how much changed, or what the changes were. This
feature adds a per-turn **file card**: a summary header (`N files changed
+A −D`) over one-line per-file rows, each expandable into a scrollable
colored unified diff — the reference interaction is the stacked-card
pattern (Cursor/Claude-Code style) the owner supplied.

## Decisions (owner-ruled during brainstorming)

1. **Capture at the tool seam** (`fs_write` / `fs_edit` / `fs_patch`),
   with the record schema carrying `source: "tool" | "scan"` so a future
   workspace-scan mode can slot in without migration. MCP/external
   mutations are invisible in V1, by design.
2. **V1 is read-only.** No Undo, no Review button. Undo arrives with the
   V1.5 review screen (per-file, guarded by the stored after-digest).
3. **Persistence rides the existing step log.** No schema bump: records
   are carried by `AgentStep`/`ToolResult` model fields and persisted by
   the existing `AgentRuns_DB.append_steps`.
4. **The card mounts once, at turn end** (terminal run state). No
   live-updating card in V1 — deliberate, given the derived-state/refresh
   bug family (tasks 15673/15740/15511).
5. Same-file edits within a turn are shown as a **stacked sequence**
   (with `— edit 2 of 3 (fs_edit) —` separators), not a fabricated net
   diff: true nets need before/after content we deliberately do not
   store. Counts are summed and labeled `across N edits`.

## Architecture

### 1. Capture layer (pure + one seam)

New module `tldw_chatbook/Agents/file_change_capture.py`.

Choke point: `LocalToolProvider.invoke`, the single post-verdict
`spec.handler(args)` call (never-raises wrapper). Capture wraps exactly
that call, gated by a new declarative `LocalToolSpec.capture_file_changes:
bool` (set on `fs_write`, `fs_edit`, `fs_patch`) — future mutating tools
opt in visibly rather than by name list.

Per mutation, the capture helper:
- resolves the target path **with the same resolver the tool impls use**
  (a second resolver is the arranged-through-a-seam-production-doesn't-read
  bug family; a parity test pins this),
- reads content before the handler runs (missing file → `op: "create"`,
  `before_digest: None` — distinguishes create from emptying),
- on handler success reads content again, computes a unified diff
  (stdlib `difflib`) and exact `adds`/`dels`,
- emits `FileChangeRecord` dicts:

```
{path: str (workspace-relative ONLY — no $HOME-absolute paths in the DB,
       per the diagnostic-privacy programme),
 op: "create" | "modify",            # "delete" reserved, no tool does it
 tool: str, source: "tool",
 adds: int | None, dels: int | None, # None when diff not computed
 diff: str | None, truncated: bool,
 before_digest: str | None, after_digest: str}
```

Guards:
- **Binary/huge**: null bytes or content > 2 MB → `diff: None`,
  `adds/dels: None` (no fake counts), digests still recorded.
- **Per-record cap**: diff text ≤ ~64 KB / 400 lines, `truncated: True`
  past it; counts stay exact.
- **Per-run budget**: ~2 MB of stored diff text per run; past the budget,
  records keep counts/digests and drop diff text (`truncated: True`) —
  protects the steps JSON column and restore-parse cost.
- `fs_patch` is multi-file: one call emits a **list** of records. Its
  target paths are enumerated BEFORE the handler runs by parsing the diff
  argument with the existing `patch_tool_impls` parser (before-content
  must be read per path pre-run; no second diff parser is written).
- Failures capture nothing; a tool that reports failure after mutating is
  missed (accepted, documented). Content is read around the handler call,
  so a foreign writer in the gap yields a self-consistent record of what
  we observed (accepted race).

### 2. Model plumbing (pure)

- `ToolResult` gains `file_changes: list[dict] | None = None` (provider
  fills it).
- `AgentStep` gains the same optional field; the runtime forwards
  `result.file_changes` when building the step. Both are
  `dataclasses.asdict`-serialized today, so old rows construct with the
  default and new rows round-trip — no migration.
- Persistence is untouched: `append_steps` already stores the enriched
  dicts. A hard process crash mid-turn loses that turn's capture
  (accepted); FAILED/CANCELLED terminals persist normally.

### 3. Display state (pure)

`build_turn_file_card_state(steps) -> TurnFileCardState | None` in
`Chat/console_display_state.py` (repo convention). `None` when the run
touched no files → no card. State carries: per-file entries in
chronological order (path, op, counts or binary/large marker, edit
sequence with per-edit diffs), header totals, `outcome_note` derived from
the terminal state (COMPLETED → none; FAILED → "turn failed after these
edits"; CANCELLED → "stopped after these edits").

### 4. Widget

`ConsoleTurnFileCard` (`Widgets/Console/console_turn_file_card.py`),
**a `ChatMessage`-family widget** like `ToolCallMessage` — it rides the
transcript's existing mount/recompose machinery and the "never type-query
message rows" test convention.

- Header: `N files changed +A −D` (+ `· 1 binary/large` when counts are
  partial; + the outcome note), collapse/expand-all chevron
  (`resolve_glyph`). Header layout reserves space for V1.5's
  Undo/Review controls.
- Rows: always mounted, display-managed (the conditional-compose lesson);
  middle-elided path (via `_middle_elide_cells`, promoted from the
  file-notes workspace module to a shared util), per-file `+a −d`,
  chevron. Keyboard: rows focusable, Enter toggles. Counts/markers are
  text, never color-only.
- Expanded: unified diff as styled Rich text (`+` green, `−` red, hunk
  headers muted; ds tokens; theme-aware) in a capped scroll container
  (`max-height` + `overflow-y: auto`, the task-15110 pattern) with a
  `… diff capped` footer when truncated. Diff parse/style is lazy on
  first expand and cached.
- Structural sizing lives in the widget's `DEFAULT_CSS` (the bundle-less
  harness renders widget defaults; geometry measured without them is not
  measured); the app sheet adds theme polish only.

### 5. Mounting and restore

- **Live**: the run-terminal handler builds the state from the
  **in-memory `RunOutcome.steps`** the service hands it (verified: the
  completion path carries the full step list) — never a DB read-back,
  which could race `append_steps` — when a run reaches
  COMPLETED/FAILED/CANCELLED, and mounts one card into the run's
  **owning session** transcript (per-session, never "the active session"
  — the parallel-agents §2 single-slot family), filtered to **top-level
  runs only** (sub-agent terminals mount nothing in V1; the concrete
  top-level-vs-child discriminator is confirmed against the fleet's run
  linkage at planning time).
- **Restore**: transcript restore rebuilds from ChaChaNotes messages;
  the card's data lives in AgentRuns. Bridge: `agent_runs` rows carry
  `assistant_message_id` — restore issues **one batched query** for the
  conversation's runs, builds card states from their steps, and attaches
  cards to their owning messages.

### 6. Boundaries the implementation must not cross

- **The card is never persisted as a conversation message.** ChaChaNotes
  stays the truth for messages; AgentRuns stays the only source for
  cards. Mounting the card must not enter any message-save path (a
  dual-persistence card would drift from its own source on re-derive).
- **Conversation exports/copy-all exclude the card** — it is derived
  presentation, not content (`document_generator` and the transcript
  copy flows skip it).
- **Kill switch**: `[console] turn_file_cards`, default ON — the
  console-markdown precedent for transcript-rendering additions. OFF
  suppresses card *mounting* only; capture/persistence still run, so
  turning it back on retroactively shows past turns' cards.

## Testing

Verified at spec time (not deferred to planning): the provider holds
`workspace_root` and every impl funnels through the shared
`resolve_workspace_path()` — capture imports the same function.

- **Pure**: per-tool records (create/modify, multi-file patch), `None`
  before-digest on create, binary/size guard, per-record cap + per-run
  budget, builder outputs (no-card, outcome notes, multi-edit
  sequencing), and the **resolver-parity pin** (capture and impl resolve
  identically; mutation-tested).
- **Models**: `ToolResult`/`AgentStep` `file_changes` round-trip through
  `asdict` → `append_steps` → read-back.
- **Widget**: rendered-frame tests on the **real CSS stack** (bundle
  loaded — task-15110's lesson): rows render, expand shows styled lines,
  capped container scrolls internally, truncation footer, binary-row
  copy, keyboard toggle, ASCII glyph fallback. Queries by id/class only.
- **Integration**: send-integration pattern — a fake run driving
  `fs_write` + `fs_edit` doubles → card mounts at terminal in the owning
  session with correct rows; a restore test exercising the
  **assistant_message_id lookup** (not just the builder); a
  parallel-session test pinning the card lands in the non-active owning
  session.
- **Mutation checks**: delete the runtime→step forwarding → integration
  red; clear a spec's `capture_file_changes` flag → pure red.
- **Boundary tests**: kill switch OFF → no card mounts, capture still
  persists; a conversation export of a turn with a card contains no card
  content.

## Out of scope (backlog seeds)

- **V1.5**: `Review` button → dedicated screen (File-Notes-workspace
  structural model): all turn changes, annotations/feedback on specific
  hunks, per-file Undo (reverse-apply guarded by `after_digest` match).
- **V2**: sidebar multi-file review; `current` / `git commit` /
  `git push` / PR modes when the workspace is a git repo.
- Workspace-scan capture (`source: "scan"`) for MCP/external mutations.
- Sub-agent run aggregation into the parent card.
- Whole-turn Undo from the card header.

ADR required: no — additive UI + two optional model fields; no
cross-module contract changes.

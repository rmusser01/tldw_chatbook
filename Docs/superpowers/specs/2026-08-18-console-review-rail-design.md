# Console review rail + review comments (TASK-18060, V2 arc A)

**Status:** approved design, code-grounded (every anchor below read first-hand
at dev `f00acbd8b`, 2026-08-18)
**Task:** `backlog/tasks/task-18060 - Inspector-rail-multi-file-review-and-review-comments.md`
**Predecessors:** V1 card (`2026-08-15-console-turn-file-review-design.md`),
V1.5 annotate loop (`2026-08-17-console-turn-file-annotate-design.md`).
**Sibling:** TASK-16801 (narrowed by this arc's split to the git
`commit`/`push`/PR modes — arc B, not this spec).

## Goal

Review an agent conversation's file changes **across all turns** without
leaving Console: a "Changed files" section in the existing Inspector rail
lists the conversation's cross-turn latest state per file; selecting a file
opens the existing Review screen at that file; and the Review screen gains
plannotator-style commenting — a comment on a **specific diff line** or on
the **whole file** — feeding the same auto-attach delivery loop V1.5
shipped for hunk notes.

Owner rulings (2026-08-18): the section lives in the **Inspector rail**
(no new rail); line comments anchor to **diff lines** (immutable
snapshot-pinned diffs — anchors never drift), not current-file lines.

## Non-goals

- No git actions (`commit`/`push`/PR) — arc B / TASK-16801.
- No workspace-wide or cross-conversation aggregation — conversation-scoped,
  matching the provider identity everything else uses.
- No multi-pane simultaneous diff rendering — the single-file pane and its
  cap are a deliberate freeze guard (`change_review_screen.py:10-13`);
  "across several files at once" is satisfied by the rail's cross-turn list
  plus instant click-through (owner-approved shape).
- No hunk-note *creation* in the Review screen (stays on the card); no
  current-file/editor annotation mode.
- No new revert affordances; revert stays exactly where it is, behind its
  named-files confirm.

## Architecture

### 1. Cross-turn aggregation (pure + provider)

Nothing aggregates across runs today — `turns()` groups per run
(`change_review_screen.py:136`), `turn_for_run` is run-scoped, and the card
renders each run independently. New:

- **Pure assembly** in `Chat/console_display_state.py`:
  `conversation_file_summary(rows_with_files) -> list[ConversationFileEntry]`
  where the input is `[(snapshot_row, [ChangedFile])]` for the
  conversation's clean rows (tracking-error rows excluded, same filter as
  everywhere) and the output is latest-state per `(root, path)` — **the
  newest snapshot row covering a path wins** (rows arrive oldest-first via
  `ORDER BY cs.id`; last writer wins). A rename (`status R`) keys by its
  NEW path and supersedes the old path's entry.
  `ConversationFileEntry` (frozen): `root, path, label (multi-root
  prefixed, same convention as turn_file_entries), status, adds, dels,
  run_id, snapshot_id, note_count`. **Counts honesty**: `adds/dels` are
  the NEWEST covering turn's deltas for that file, not cumulative since
  conversation start — the section header says "latest turn deltas" so
  the numbers cannot be misread as totals.
  **Pruned rows**: `changed_files(row)` raises `ChangeTrackingError` when
  retention pruned a row's snapshots (the Review screen banners this,
  `_load_turn`'s per-row try/except) — the aggregation catches per row,
  skips it, and reports a `pruned_rows` count the section renders as a
  dim "history pruned for N turns" tail line rather than hiding it.
- **Provider method** on `AgentRunsChangeReviewProvider`:
  `conversation_changed_files() -> list[ConversationFileEntry]` — reads
  `change_snapshots_for_conversation`, calls `changed_files(row)` per clean
  row, joins per-file note counts (one query over `change_notes` for the
  conversation, grouped by `(root, path)`), and delegates assembly to the
  pure function. **This method does git subprocess work per row and is
  NEVER called on the UI thread** (see §2's cost model).

### 2. The rail section (cached-summary pattern, verbatim precedent)

The Inspector rail's hard invariant is documented at
`chat_screen.py:3732-3748`: the sync loop runs on a **0.2s timer while
streaming**, and rail compose/recompose paths read ONLY screen-held caches
— never a DB query. The dictionary/world-book summaries are the shipped
precedent, copied exactly:

- **Cache**: `ChatScreen._console_changed_files_summary:
  tuple[ConversationFileEntry, ...] | None`, plus per-row memo
  `_console_changed_files_row_cache: dict[int, list[ChangedFile]]` keyed by
  snapshot row id so a recompute only runs git for rows it has not seen
  (incremental: a new turn costs its own rows' git calls, not the whole
  history's). The memo and summary are **cleared on conversation switch**
  (CLAUDE.md performance rule: clear caches on context switch) — row ids
  are globally unique so this is hygiene, not correctness.
  The provider is acquired exactly like the card's factory — the V1
  `_console_change_review_provider()` recipe (bridge → conversation id →
  provider), called inside the worker, never at render time.
- **Guard**: `_last_console_changed_files_scope = (conversation_id,
  newest change_review_run_id present in the message store)`. Marker
  messages already carry `change_review_run_id`, so the guard costs no DB
  read at all (an O(messages) in-memory scan; derived where the message
  list is already being handled rather than re-scanned per idle tick). When the guard tuple changes, the screen dispatches ONE
  off-thread worker (`asyncio.to_thread`, exclusive group) that calls the
  provider's `conversation_changed_files()` and lands the cache via
  `call_from_thread`, then syncs the section in place by id.
  **Note-change invalidation** (self-review catch): the guard tuple only
  moves on new runs, so the `✎ N` badges would go stale on note
  save/delete. Every app-side note mutation path — the card's save/delete
  handlers and the Review screen's (via its dismissal callback) — also
  resets the guard to `None`, forcing one refresh on the next tick. The
  per-row git memo survives that reset (git content didn't change; only
  the notes join reruns).
- **Section widget**: `ConsoleChangedFilesSection` in
  `Widgets/Console/`, constructed with precomputed state per the rail
  convention (`right_rail.py:82-88`), mounted in
  `ConsoleInspectorRail.compose()` between the retrieval-Scope row and the
  run inspector, wrapped in `frame_console_region(..., variant="quiet")`.
  One row per file: status glyph, cell-elided path (`middle_elide_path`),
  `+A −D`, and a `✎ N` badge when `note_count > 0`. Header line: totals +
  file count. The list is **capped** (rail-section cap conventions,
  task-15110 family) with an honest "+N more — open Review" tail row.
  Rows are compact Buttons (`active_effect_duration = 0`); pressing one
  posts `ConsoleChangedFilesSection.FileSelected(run_id, snapshot_id,
  path)`; the section's handlers follow the card's absolute no-escape
  rule.
- **Screen handler**: opens the Review screen through the existing
  `_open_change_review` recipe with `initial_run_id=<the file's newest
  run>`, the new `initial_path=<path>`, and
  `initial_snapshot_id=<snapshot_id>` — `select_file` matches by path
  alone (`:705-714`), which is ambiguous when two windows of one run
  cover the same path; the extended selection prefers the leaf whose
  row id equals `initial_snapshot_id` and falls back to first-path-match
  (legacy callers pass no snapshot id and keep today's behavior).
- **Empty state**: the section renders nothing (height 0) when the
  conversation has no snapshot rows — the rail must not grow a permanent
  empty box.
- **Config**: `[console] changed_files_section` (default ON,
  presentation-only; OFF renders nothing and skips the recompute worker).

### 3. Review screen: `initial_path`, line cursor, comments

- **`initial_path` / `initial_snapshot_id`** join `initial_run_id` as
  **constructor state** — the post-push race is documented at
  `change_review_screen.py:414-427` (`call_after_refresh` fired before
  compose; NoMatches). `_load_turn` honors them via `select_file`
  (extended per §2's snapshot-aware matching) instead of
  `_focus_leaf(0)`, then clears them (turn switches revert to
  first-file).
- **Line cursor**: the diff pane stays ONE flat `Static`
  (`_render_diff`, `:827-861`); a cursor is screen state
  (`_cursor_line: int`, index over the **rendered** lines, which under the
  cap equal the full-diff line indices — the cap truncates the tail only).
  Rendering styles the cursor line's background; moving the cursor
  re-renders (≤2,000 appends — same cost class as the existing per-file
  render) and scrolls it into view. **Key routing hazard, named**: the
  diff `VerticalScroll` consumes up/down for scrolling, so the pane gets
  the card's proven `on_key`-reclaim treatment (`console_turn_file_card`'s
  Enter fix precedent — traced there against Textual's real dispatch):
  when the pane is focused, ONLY up/down/`c`/Escape are reclaimed —
  page-up/down/home/end keep native scrolling. Up/down move the cursor
  (scroll follows), `c` opens the line-comment input, and Escape returns
  focus to the tree — deliberately SHADOWING the screen's Esc-dismiss
  while the pane is focused (Esc-Esc = pane→tree→dismiss, an explicit UX
  decision, tested). `j`/`k` file navigation is untouched.
- **Comment creation**: `c` (pane focused, cursor on a diff line) opens an
  inline one-line `Input` under the pane (reusing the card's validation:
  strip, non-empty, ≤2,000 chars via `input_validation`); Enter saves
  off-thread through the provider, Escape cancels. A `Comment file`
  affordance (footer key `C` + a small button by the totals) records a
  file-level comment on the focused file the same way. Both anchor to the
  focused leaf's `(row, change)` — run_id, root, path, snapshot_id all in
  hand (`_leaves`, `:434-436`).
- **Note display**: the screen shows the focused file's existing notes —
  line comments as a `● comment` marker appended to their diff line (and
  the note text in a strip below the pane), file comments and the card's
  hunk notes in the same strip, each labeled by kind, `sent` when
  delivered. **Posture correction (self-review)**: this screen's diff
  load is SYNCHRONOUS on the UI thread today (`_render_diff` calls
  `provider.diff_text` directly, `:838`) — the notes read
  (`notes_for_run`, `:388`, a SQLite query) loads the same synchronous
  way, matching the screen's accepted posture; the off-thread discipline
  belongs to the RAIL (§2) and the comment WRITE paths, not to this
  screen's reads.
  Pending comments can be deleted from the strip (same
  `delete_change_note` rules as the card: pending only).

### 4. Anchors + schema (audit v11)

`change_notes` (v8, +`delivered_by_run_id` v9, +`snapshot_id` v10) gains,
by the same PRAGMA-guarded idempotent-ALTER convention:

- `anchor_kind TEXT NOT NULL DEFAULT 'hunk'` — `'hunk' | 'file' |
  'diff_line'`; every existing row reads as `'hunk'` truthfully.
- `diff_line_index INTEGER` — 0-based over the file's **full** diff text
  (consistent with `hunk_index` semantics; NULL except `diff_line`).
- `diff_line_text TEXT` — the anchored line, verbatim (NULL except
  `diff_line`); the quoted line makes delivery self-contained, the same
  retention posture as `hunk_excerpt`.

For `diff_line` rows, the hunk fields are ALSO populated (the hunk the
line falls in, via `split_unified_diff`) — a deliberate convergence: the
turn file card's existing hunk-note filter (hunk_index + hunk_header +
snapshot_id) therefore renders line comments under their hunk in the
card too, with no card changes (`file` rows' `-1`/`''` sentinels can
never match a real hunk, so they stay out of the card). Stated so a
reviewer reads it as intended, not accidental; `hunk_excerpt` for a line comment is the
line's own hunk excerpt. For `file` rows, `hunk_index = -1`,
`hunk_header = ''`, `hunk_excerpt = ''` — the formatters (§5) render kind
-aware and never show a dangling `@@` for them.
`add_change_note` gains `anchor_kind='hunk'`, `diff_line_index=None`,
`diff_line_text=None` keywords (defaults keep every existing caller
byte-compatible).

### 5. Delivery — the V1.5 loop, formatters kind-aware

Attach, exact-id stamping at completion, cap-with-holdover, disclosure
live + resume at the delivering run: **all unchanged** — the mechanics
never inspect the anchor. Only the two shared formatters in
`console_display_state.py` learn kinds:

- `render_diff_feedback_block`: a `file` note renders
  `### <path> — whole file   [run <short-id>]` with no fence; a
  `diff_line` note renders the hunk header line plus
  `> on line: <diff_line_text>` above the user's note, fenced excerpt as
  today.
- `format_diff_feedback_disclosure`: `<path> (whole file): "note"` /
  `<path> <hunk_header> line: "note"` — one line per note, stable format,
  shared verbatim by live and resume (the byte-parity contract from V1.5
  holds per kind).

## Known boundaries and risks (disclosed)

- **Aggregation cost**: `changed_files()` is a git subprocess per snapshot
  row. Mitigations are structural: off-thread only, per-row memo so only
  unseen rows pay, guard tuple so nothing recomputes on idle ticks. A
  degenerate conversation (hundreds of turns) pays once on first open,
  incrementally after.
- **Stale-across-revert**: a revert changes disk truth but not snapshot
  rows; the rail list (snapshot-derived) is unchanged by design — it
  describes what turns did, same as the card and screen. The screen
  already reloads after revert (`_report_outcomes`, `:812-814`).
- **Cursor-vs-cap**: lines beyond the display cap are unreachable by the
  cursor (they are not rendered); the cap's honest truncation line says
  so. Line comments on capped tails are not supported in this arc.
- **Inherited**: temp-conversation promotion id drift; superseded-run
  block-drop on resume; TOOL-row history participation — all exactly as
  documented in the V1.5 spec.

## Testing

- **Pure**: `conversation_file_summary` — latest-wins on overlapping
  turns, rename supersession, multi-root prefixing, note-count join;
  real `git diff` fixtures where hunks matter.
- **Provider**: `conversation_changed_files` on the REAL stack
  (tracker/shadow service/file-backed AgentRunsDB — the five-times-bitten
  fixture rule): two turns touching one file → one entry, newest counts.
- **Rail**: real CSS stack, id/class queries; section renders rows +
  badge + capped tail; empty-conversation renders nothing;
  `FileSelected` posts with the right `(run_id, path)`; the guard — a
  sync tick with an unchanged scope tuple performs no recompute (spy on
  the provider), a new marker message triggers exactly one.
- **Screen**: `initial_path` opens focused on that file (and the
  post-push race pattern is respected — ctor state, no
  `call_after_refresh` mutation); cursor moves render the styled line and
  never escape the pane's `on_key`; `c` saves a `diff_line` note with the
  right anchor (DB-read assertion), `C` a `file` note; notes strip shows
  all three kinds; pending-delete works, delivered shows `sent`.
- **Delivery**: kind-aware block + disclosure formats pinned byte-exact;
  a mixed batch (hunk + file + line) attaches, stamps, discloses, and
  resume-derives identically to live.
- **Guard discipline**: every behavior-changing test proven RED against
  pre-feature code; masks named in V1/V1.5 (auto-focus, raw `on_key`
  swallows, `Button.label` markup) actively checked where they apply.

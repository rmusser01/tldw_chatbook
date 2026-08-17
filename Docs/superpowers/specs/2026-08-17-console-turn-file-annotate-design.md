# Console turn file card V1.5 — annotate loop + Review affordance (TASK-16800)

**Status:** approved design, adversarially reviewed (amendments folded in)
**Date:** 2026-08-17
**Task:** `backlog/tasks/task-16800 - Turn-file-card-annotate-feedback-loop-and-Review-affordance.md`
**Predecessor:** `2026-08-15-console-turn-file-review-design.md` (V1, shipped as PR #1728
+ follow-ups #1731/#1734). This spec covers the V1.5 bucket that spec's
"Out of scope" section named: the annotate/feedback loop, the `Review`
affordance, and the two trimmed polish items (expand-all chevron,
middle-elided paths).

## Goal

Close the loop between "the agent shows me a diff" and "I tell it what to
change" without leaving the transcript: a user attaches short notes to
specific hunks of a turn's diff, and those notes automatically reach the
agent as labeled context on the user's next send — visibly, durably, and
with zero extra clicks (owner ruling 2026-08-16: **auto-attach +
disclosure**; no composer-draft path).

## Non-goals

- No mutation of the workspace from the card (revert stays exclusively on
  the Review screen behind its confirm — TASK-1845/TASK-1972 precedent).
- No editing of the agent's diff, no per-line commenting (hunks are the
  anchoring unit), no threaded replies to notes.
- No change to the V2 scope (sidebar multi-file review, git modes —
  TASK-16801).

## Architecture

Five pieces, in dependency order. Every anchor below was re-verified
against dev at `ed49499b8` (2026-08-17).

### 1. Persistence: `change_notes` in AgentRuns_DB (audit version 8)

AgentRuns_DB migrates by its own convention — idempotent
`CREATE TABLE IF NOT EXISTS` in the DDL block plus PRAGMA-guarded
idempotent ALTERs, with an **append-only audit** `schema_version` table
(`INSERT OR IGNORE ... VALUES (N)`; currently at 7 — see
`DB/AgentRuns_DB.py:335-345`; the `_CURRENT_SCHEMA_VERSION = 3` constant
at `:38` is not the live version gate). This feature adds the
`change_notes` DDL to the create block and appends audit version **8** —
self-contained, no ChaChaNotes migration. NOTE (v5 durability comment in
the DDL): this DB holds durable user-authored content, and notes extend
that — "clear run history" tooling must not treat it as disposable.

```sql
CREATE TABLE change_notes (
    id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL,
    root TEXT NOT NULL,
    path TEXT NOT NULL,
    hunk_index INTEGER NOT NULL,     -- 0-based, over the FULL diff (see §3)
    hunk_header TEXT NOT NULL,       -- the "@@ -a,b +c,d @@ …" line, verbatim
    hunk_excerpt TEXT NOT NULL,      -- ≤ 40 lines of the hunk body, captured at note time
    note TEXT NOT NULL,
    created_at TEXT NOT NULL,
    delivered_at TEXT                -- NULL = pending; set by the delivery seam (§4)
);
CREATE INDEX idx_change_notes_pending
    ON change_notes(run_id) WHERE delivered_at IS NULL;
```

No denormalized conversation id: `agent_runs` already carries
`conversation_id NOT NULL`, so `pending_notes_for_conversation` is a
JOIN through `change_notes.run_id = agent_runs.id`. One source of truth
— and the card never needs to learn the conversation id at insert time
(it only knows its `run_id`, which is exactly the key it writes).

- **Anchor** = `(run_id, root, path, hunk_index, hunk_header)`. Snapshot
  rows are immutable and `git diff -M <pinned-sha> <pinned-sha>` is
  deterministic, so the anchor is stable across resume.
- **`hunk_excerpt` is the retention safety net** (review finding #5):
  shadow-repo retention can prune a run's snapshots, after which the hunk
  can no longer be re-rendered. The excerpt (capped at 40 lines, elided
  with an honest `… N more lines` tail) makes both display-after-pruning
  and delivery self-contained. Captured once at note creation from the
  full diff text the card already has.
- API on `AgentRunsDB`: `add_change_note(...)`, `delete_change_note(id)`
  (allowed only while `delivered_at IS NULL`), `notes_for_run(run_id)`,
  `pending_notes_for_conversation(conversation_id)`,
  `mark_notes_delivered(ids, timestamp)`.
- All access goes through the existing thread-local connection pattern —
  the card reads/writes off-thread via `asyncio.to_thread`, same as its
  diff loads. Note text is length-bounded (2,000 chars) and validated via
  `input_validation` at the widget boundary.

### 2. Hunk segmentation (shared, pure)

A pure helper `split_unified_diff(text) -> list[Hunk]` where
`Hunk = (header: str, body_lines: list[str], file_prelude: str)` —
adapted from the proven `_HUNK_HEADER` regex + `_parse_hunk` in
`Tools/patch_tool_impls.py:58/:221` (reused, not reinvented; the patch
tool's own parser stays untouched). Lives beside the card's other pure
logic in `Chat/console_display_state.py`.

**Segmentation always runs on the FULL `provider.diff_text` output**
(review finding #2). The `diff_display_max_lines` cap
(`change_review_screen.py:93`) becomes a per-hunk *display* cap: every
hunk gets a block even when its body is elided, so hunks past the old cap
are still annotatable and indices never shift.

### 3. Widget: per-hunk blocks + note UI (the restructure)

Today an expanded row mounts ONE flat `Static` with the whole colored
diff. Per-hunk actions require restructuring the expand path (review
finding #1 — this is a real widget change, planned as its own task):

- Expanding a row mounts, inside the existing `VerticalScroll` diff body,
  one block per hunk: a colored hunk `Static` (header + capped body,
  reusing `_styled_diff`'s coloring) followed by a slim action row with
  `✎ note` (and the hunk's existing notes rendered beneath, each with a
  `✕` delete affordance while undelivered).
- `✎ note` opens an inline one-line `Input` under that hunk. Enter saves
  (off-thread insert, then the note renders in place); Escape cancels.
  Delete removes an undelivered note (off-thread). Delivered notes render
  with a `sent` marker and no delete — they are record.
- **Live-transcript safety** (review finding #6): the final-fix-wave
  `_update_row_widget` branch reuses the card in place when marker/run-id
  match, which must protect an open input across sync ticks — pinned by
  an explicit test (half-typed input survives a sync tick and a selection
  move). All new handlers follow the card's absolute rule: no exception
  escapes an `on_*` handler; every seam degrades logged.
- The diff cache changes shape from joined-string to segmented hunks
  (cache the `list[Hunk]` per row index; styling applied at mount).

### 4. Delivery: auto-attach + disclosure

- **Attach seam:** `ConsoleAgentBridge.run_reply` collects
  `pending_notes_for_conversation(...)` itself at the attach point — the
  bridge owns the AgentRuns DB, the conversation context, and the
  completion seam, so the whole feature stays inside it; the controller
  and `run_reply`'s signature are untouched. The rendered block is
  appended to the last `role=="user"` message of the bridge's own
  outbound copy, immediately after the `turn_bundle_block` append and by
  the same mechanism (bridge `:2576/:3343-3353`). Block format:

  ```
  ## Diff feedback from the user (on your earlier file changes)
  ### <path> — @@ -a,b +c,d @@   [run <short-id>]
  > <note text>
  <hunk_excerpt, fenced>
  ```

  The block is capped (16 KB total, oldest-first inclusion with an honest
  "… N more notes held for the next message" line) so a pile of notes
  cannot blow the context budget. **A note elided from the block is NOT
  attached and NOT stamped** — it stays pending and rides the following
  send; only feedback the model actually received is ever marked
  delivered.
- **Stamping happens at run completion, not attach** (review finding #3):
  the block is only ever in the outbound copy, so a run that dies before
  producing assistant output must leave the notes pending for the retry.
  **The attach step captures the exact note ids it included**, and that
  id list travels with the run; at completion, `mark_notes_delivered`
  stamps precisely that list — never "all pending for the conversation".
  This closes a real race: a user can annotate an *older* turn's card
  while a new run is already in flight, and those mid-run notes were
  never in the payload — a blanket stamp would silently swallow them.
  Stamping is called at the same run-completion point that invokes
  `_append_change_markers` (bridge `:4695`) — **beside** it, not inside
  it, so it happens even when the new turn changed no files and the
  marker seam's `if files:` gate emits nothing. Gated on the run having
  produced assistant output. Double-delivery is impossible by
  construction (pending query excludes stamped rows).
- **Disclosure row** (review finding #4 — resume amnesia): at the same
  completion seam, a TOOL-role transcript row records what was attached —
  content, not just a count:
  `📝 Diff feedback attached — a.py @@ -1,4 +1,6 @@: "use the cached
  value here"` (one line per note, same message family as the change
  markers). **Durability follows the marker precedent exactly**: the
  change markers are not persisted messages — they are emitted live and
  re-derived on resume (`resume_marker_messages`, bridge `:4536`) — so
  the disclosure row is likewise emitted live at completion and
  re-derived on resume from delivered `change_notes` rows (grouped by
  `delivered_at`, anchored after the delivering run's marker position).
  Everything needed for re-derivation is already in the table. The row
  carries **no** `change_review_run_id`, so it can never itself render as
  a turn file card. Whether TOOL rows re-enter rebuilt provider history
  is an inherited pipeline property (see Known boundaries).

### 5. Card affordances (bounded polish)

- **`Review` button** in the card header opens the existing Review screen
  scoped to the card's run — same opener recipe as `v`, plus a small
  addition: the screen accepts an initial `run_id` (today it opens at
  latest; `turn_for_run` at `change_review_screen.py:157` already gives
  the lookup). Keyboard `v` unchanged.
- **Expand/collapse-all chevron** in the header. Expand-all loads the
  not-yet-cached diffs **serialized in one worker** (not N concurrent git
  subprocesses), reusing the per-row cache.
- **Middle-elided paths** in row labels: elide middle path components to
  the row's width budget (keep first + last), recomputed on resize, full
  path preserved in the row's tooltip.

## Kill switch

`[console] turn_file_cards = false` keeps the plain marker byte-identical
(existing pinned test). No note UI exists in that mode, but
previously-created pending notes still deliver on the next send, with the
disclosure row — nothing silently vanishes (AC#5). The delivery seam is
controller/bridge-side and does not consult the presentation switch.

## Known boundaries (inherited, disclosed, not changed here)

- **Agent-path-only delivery:** `turn_bundle_block` is applied only
  inside `ConsoleAgentBridge.run_reply` (the wake feature's delivery-path
  decision record, `Chat/console_fleet_wake.py:18-40`, documents this).
  A send taken on the plain-provider path (agent runtime toggled off)
  attaches nothing — notes simply **stay pending** until the next agent
  turn, which is the turn that can act on file-change feedback anyway.
  Unlike the wake notice, none of that record's disqualifiers apply here:
  the notes are user-authored feedback riding the user's own real
  message, so the "reads as user input" and "no trailing user entry"
  concerns are moot.

- **Conversation-id drift:** temporary-conversation promotion can change
  a conversation's id while existing runs keep the old one. Notes share
  exactly the exposure `change_snapshots` already has; no new mitigation.
- **TOOL-row history participation:** the disclosure row's presence in
  rebuilt provider history follows whatever the pipeline does for the
  change markers today; this spec relies on it only as a human/export
  record.
- **`git diff -M` determinism:** deterministic for pinned trees in
  practice; the `hunk_excerpt` denormalization is the safety net if a git
  upgrade ever changes rename scoring.

## Testing

- **Pure:** `split_unified_diff` against real `git diff -M` output —
  multi-hunk, rename, binary, and the >cap case (segmentation on full
  text; display cap per hunk).
- **Persistence:** real file-backed `AgentRunsDB` (NOT `:memory:` —
  thread-affinity, V1 lesson): opening a pre-existing DB file creates
  `change_notes` and appends audit version 8 idempotently;
  add/delete/pending/mark-delivered; resume round-trip (new provider +
  card instance sees the notes).
- **Widget:** real CSS bundle, id/class queries only. Per-hunk blocks
  render with correct hunk count; note input opens/saves/cancels; the
  open-input-survives-sync-tick and selection-move test; no-escape
  degrade on injected DB failure; delivered notes lose the delete
  affordance.
- **Delivery:** injection test asserting the exact block lands on the
  last user message of the outbound copy and the stored message is
  unchanged; completion stamps `delivered_at` + emits the disclosure row
  with note content; a run failing before assistant output leaves notes
  pending; **the mid-run race** — a note created after attach, while the
  run is in flight, is NOT stamped at that run's completion and rides the
  next send; **cap elision keeps elided notes pending** (only notes in
  the block are stamped); **disclosure resume re-derivation** — a fresh
  session over the same DB re-derives the disclosure row with the same
  content, anchored after its run's marker.
- **Affordances:** Review button opens the screen at the card's run;
  expand-all mounts all diff bodies; elision keeps first+last components.
- **Kill switch:** OFF-path byte-parity unchanged; pending notes created
  before switching OFF still deliver.
- **Guard-test discipline:** every regression test proven RED against
  pre-fix/pre-feature code where a mask is plausible (V1 lesson:
  `run_test` auto-focus; fixture-invented shapes — real-provider fixtures
  reused from `Tests/UI/test_change_review_screen.py`).

# Console Review Rail + Review Comments Implementation Plan (TASK-18060)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A "Changed files" section in Console's Inspector rail lists the conversation's cross-turn latest state per file; selecting a file opens the Review screen focused on it; the Review screen gains diff-line and whole-file comments feeding the existing auto-attach delivery loop.

**Architecture:** Pure aggregation over existing snapshot rows (latest clean row per `(root, path)` wins); rail section on the cached-summary pattern (guard tuple + off-thread recompute — never DB/git on the sync tick); constructor-state click-through; line cursor over the existing flat diff Static with an `on_key` pane reclaim; `change_notes` anchor extension (audit v12 (renumbered from v11 at rebase: task-15669 concurrently minted v11 on dev): `anchor_kind`/`diff_line_index`/`diff_line_text`); kind-aware block/disclosure formatters with the delivery mechanics untouched.

**Tech Stack:** Python 3.11+, Textual 8.x, SQLite (AgentRuns_DB), pytest (venv-only: `VIRTUAL_ENV=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -p no:randomly -q`).

**Spec:** `Docs/superpowers/specs/2026-08-18-console-review-rail-design.md` — binding authority; read before any task. Line anchors verified at dev `f00acbd8b`.

## Global Constraints

- The rail's compose/recompose and the 0.2s sync tick NEVER touch the DB or git — cached-summary pattern only (`chat_screen.py:3732-3748` documents the invariant; the dictionary/world-book summaries are the precedent to copy).
- No exception may escape any `on_*` handler in new widgets/panes — every seam degrades logged (the card's absolute rule).
- All git/DB work for the rail runs in ONE off-thread worker (`exclusive=True` group); comment writes run off-thread; the Review screen's synchronous read posture is accepted and unchanged.
- Delivery mechanics (attach, exact-id stamping, cap-with-holdover, disclosure live + resume at the delivering run) are UNTOUCHED — only the two shared formatters learn anchor kinds, and live/resume disclosure output stays byte-identical per kind.
- Tests: file-backed `AgentRunsDB(tmp_path / "runs.db", client_id="t")`, never `:memory:`; real provider stack (tracker/shadow service) where provider behavior is under test — no fixture-invented shapes; UI tests on the real CSS stack (`_SCOPED, _SELF = build_css.screen_css_paths(...)` — scoped first, self last) with id/class queries; no `$ds-*` declarations in widget CSS.
- Every behavior-changing guard test proven RED against pre-feature code (known masks: `run_test` auto-focus; raw `on_key` swallowing; `Button.label` markup-parsing plain strings — build labels with `rich.text.Text` when a glyph fallback may contain brackets).
- Note text validation: strip, non-empty, ≤2,000 chars via `Utils/input_validation.validate_text_input` (the card's `_validate_note_text` is the template).
- Revert behavior, the diff display cap, and `[console] turn_file_cards` OFF byte-parity keep their existing pinned tests green.

---

### Task 1: `change_notes` anchor extension (audit v12) + note-counts query

**Files:**
- Modify: `tldw_chatbook/DB/AgentRuns_DB.py` (change_notes DDL comment block; migration block; `add_change_note`; new counts query)
- Modify: `tldw_chatbook/UI/Screens/change_review_screen.py` (provider `add_change_note` delegate gains the same kwargs)
- Test: `Tests/Chat/test_change_notes_db.py` (extend)

**Interfaces (Produces):**
```python
# AgentRunsDB.add_change_note gains (defaults keep every caller byte-compatible):
def add_change_note(self, *, run_id, root, path, hunk_index, hunk_header,
                    hunk_excerpt, note, snapshot_id=None,
                    anchor_kind="hunk", diff_line_index=None,
                    diff_line_text=None) -> int
def change_note_counts_for_conversation(self, conversation_id: str) -> dict[tuple[str, str], int]
    # {(root, path): count} over ALL the conversation's notes (JOIN agent_runs,
    # delivered and pending alike), parameterized, one query.
```

- [ ] **Step 1: Failing tests** — file-backed DB: v11 migration on a pre-v11 file (drop the three columns via raw sqlite3 the way the v9/v10 tests do; reopen; columns exist; `schema_version` contains 11; double-open idempotent); `add_change_note` defaults write `anchor_kind='hunk'` and NULL line fields (existing V1.5 callers unchanged — assert an old-signature call round-trips); a `diff_line` note round-trips index+text; a `file` note with `hunk_index=-1, hunk_header=''` round-trips; `change_note_counts_for_conversation` groups across two runs and two paths and excludes other conversations.
- [ ] **Step 2: Run to red.**
- [ ] **Step 3: Implement** — PRAGMA-guarded idempotent ALTERs (`anchor_kind TEXT NOT NULL DEFAULT 'hunk'`, `diff_line_index INTEGER`, `diff_line_text TEXT`) + `INSERT OR IGNORE INTO schema_version (version) VALUES (11)` per the file's convention; counts query mirrors `pending_notes_for_conversation`'s JOIN shape; Google docstrings with Args/Returns.
- [ ] **Step 4: Green** — plus `Tests/Chat/test_console_diff_feedback_delivery.py` and `Tests/UI/test_console_turn_file_card_notes.py` untouched-green (byte-compat proof).
- [ ] **Step 5: Commit** — `feat(console): change_notes anchor kinds + per-file counts (audit v12)`

---

### Task 2: Pure aggregation + provider method

**Files:**
- Modify: `tldw_chatbook/Chat/console_display_state.py` (beside `turn_file_entries`)
- Modify: `tldw_chatbook/UI/Screens/change_review_screen.py` (provider)
- Test: `Tests/Chat/test_console_conversation_files.py` (new)

**Interfaces (Produces):**
```python
@dataclass(frozen=True)
class ConversationFileEntry:
    root: str
    path: str
    label: str          # multi-root prefixed like turn_file_entries
    status: str
    adds: int
    dels: int
    run_id: str
    snapshot_id: int
    note_count: int

def conversation_file_summary(
    rows_with_files: Sequence[tuple[dict, Sequence[ChangedFile]]],
    note_counts: Mapping[tuple[str, str], int],
) -> list[ConversationFileEntry]
    # rows oldest-first; latest row covering a (root, path) wins; a rename
    # keys by NEW path and deletes the old path's entry; output ordered
    # newest-first by owning snapshot id, then path.

# Provider:
def conversation_changed_files(self) -> tuple[list[ConversationFileEntry], int]
    # (entries, pruned_rows). Clean rows only; per-row ChangeTrackingError
    # -> skipped + counted as pruned. NEVER call on the UI thread.
```

- [ ] **Step 1: Failing tests** — pure: latest-wins (same file in two rows → newest row's status/counts/run/snapshot); rename supersession (old-path entry deleted, new-path entry keyed R); delete-then-recreate shows A; multi-root label prefixing; note_count joined; ordering. Provider, REAL stack (tracker/shadow/file-backed DB, the `review_fixture` pattern): two turns touching one file → one entry with turn-2's identity; a retention-pruned row (reuse the existing pruned-row technique from `Tests/UI/test_change_review_screen.py`) → skipped, `pruned_rows == 1`, other rows still listed.
- [ ] **Step 2: Red.**
- [ ] **Step 3: Implement** (pure fn has no I/O; provider composes DB + git + counts + pure fn).
- [ ] **Step 4: Green** + `Tests/Chat/test_console_turn_file_entries.py` untouched.
- [ ] **Step 5: Commit** — `feat(console): cross-turn conversation file aggregation`

---

### Task 3: Review screen `initial_path`/`initial_snapshot_id` + snapshot-aware selection

**Files:**
- Modify: `tldw_chatbook/UI/Screens/change_review_screen.py` (`__init__` `:414-427`; `_load_turn` tail `:659-662`; `select_file` `:705-714`)
- Test: `Tests/UI/test_change_review_screen.py` (extend)

**Interfaces (Produces):** `ChangeReviewScreen(provider, initial_run_id=None, initial_path=None, initial_snapshot_id=None)`; `select_file(path, snapshot_id=None)` prefers the leaf whose row `id == snapshot_id`, falls back to first path match (legacy calls unchanged).

- [ ] **Step 1: Failing tests** — real-provider fixture: open with `initial_run_id` + `initial_path` → that file's diff is rendered (not the first leaf); with two same-path windows in one run, `initial_snapshot_id` picks the RIGHT leaf (assert the rendered diff content differs per window — reuse the two-window fixture family); unknown path falls back to first leaf; a later turn-switch reverts to first-file (the initials are cleared after first use). Constructor-state only — no `call_after_refresh` mutation (the `:414-427` race note is the law).
- [ ] **Step 2: Red** (the same-path disambiguation case must be red against path-only `select_file`).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Green** — whole `test_change_review_screen.py`.
- [ ] **Step 5: Commit** — `feat(console): snapshot-aware initial file selection for the Review screen`

---

### Task 4: `ConsoleChangedFilesSection` widget

**Files:**
- Create: `tldw_chatbook/Widgets/Console/console_changed_files_section.py` (+ export in `Widgets/Console/__init__.py` matching siblings)
- Test: `Tests/UI/test_console_changed_files_section.py` (new)

**Interfaces (Produces):**
```python
@dataclass(frozen=True)
class ConsoleChangedFilesState:
    entries: tuple[ConversationFileEntry, ...]
    pruned_rows: int = 0

class ConsoleChangedFilesSection(Vertical):
    class FileSelected(Message):
        def __init__(self, run_id: str, snapshot_id: int, path: str, root: str): ...
    def __init__(self, state: ConsoleChangedFilesState, *, id=None): ...
    def update_state(self, state: ConsoleChangedFilesState) -> None  # in-place resync
```

Rendering per spec §2: header `Changed files (N) · latest turn deltas +A −D`; one compact Button per entry (`active_effect_duration = 0`, label built as `rich.text.Text` — status glyph, `middle_elide_path` label, `+a −d`, `✎ n` badge when `note_count > 0`); list capped at 12 rows with an honest `+N more — open Review` tail Static; `pruned_rows > 0` renders a dim `history pruned for N turns` tail; empty entries + 0 pruned → the widget renders nothing (`display = False` / zero children). All handlers try/except-degrade; DEFAULT_CSS structural only.

- [ ] **Step 1: Failing tests** — real CSS stack host: rows render with badge/elision; press posts `FileSelected` with the entry's exact `(run_id, snapshot_id, path, root)`; cap at 12 with the tail counting the remainder; pruned tail; empty state renders nothing; `update_state` swaps rows in place (same widget instance).
- [ ] **Step 2: Red.**
- [ ] **Step 3-4: Implement; green** (+ the `$ds` contract suite `Tests/UI/test_non_obscuring_focus_contract.py` stays green).
- [ ] **Step 5: Commit** — `feat(console): changed-files rail section widget`

---

### Task 5: Screen wiring — cache, guard, worker, mount, click-through, invalidation

**Files:**
- Modify: `tldw_chatbook/UI/Console_Modules/right_rail.py` (accept + mount the section between the Scope row and `#console-run-inspector`)
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (cache/memo/guard fields near the dictionary-summary block `:3725-3760`; recompute worker; in-place `update_state` sync from `_sync_native_console_chat_ui`; `FileSelected` handler → `_open_change_review(run_id, initial_path=…, initial_snapshot_id=…)`; guard reset in the Review-screen dismissal callback and on conversation switch; config gate)
- Modify: `tldw_chatbook/Widgets/Console/console_turn_file_card.py` — NO code change expected; the card's note save/delete already round-trips through the SCREEN? Verify: if the card mutates notes without a screen hook, add a `NotesChanged` message posted after successful save/delete and handle it on the screen (guard reset). Whichever is true, document it in the report.
- Test: `Tests/UI/test_console_changed_files_wiring.py` (new)

Key mechanics (spec §2 verbatim): cache `_console_changed_files_summary`, per-row memo `_console_changed_files_row_cache` (cleared on conversation switch), guard `(conversation_id, newest change_review_run_id in the message store)`; guard change → ONE `asyncio.to_thread` worker (`exclusive=True`, group `console-changed-files`) building the provider via the `_console_change_review_provider()` recipe and calling `conversation_changed_files()`; `call_from_thread` lands cache + `update_state`. Config `[console] changed_files_section` default True: OFF renders nothing and never dispatches the worker.

- [ ] **Step 1: Failing tests** — mounted Console harness (copy the host from the factory/wiring tests of V1): with snapshot rows present, the section appears with the aggregated entries (worker awaited via pilot pauses); **the guard test**: with an unchanged scope, N sync ticks perform ZERO provider calls (spy/counter on the provider factory); a new marker message triggers exactly one recompute; note save/delete (or `NotesChanged`) resets the guard and refreshes the badge; conversation switch clears memo+summary; pressing a row opens `ChangeReviewScreen` with the right initials (assert on the pushed screen's ctor state); config OFF → no section, no worker calls.
- [ ] **Step 2: Red** (the zero-calls-on-idle-tick test must be red against a naive per-tick implementation only if one exists — otherwise it pins the invariant; state which in the report).
- [ ] **Step 3-4: Implement; green** — plus `Tests/UI/test_console_native_transcript.py -k "turn_file or selection"` and the factory byte-parity test untouched.
- [ ] **Step 5: Commit** — `feat(console): changed-files section wiring (cached summary, guarded recompute)`

---

### Task 6: Diff-pane line cursor + key reclaim

**Files:**
- Modify: `tldw_chatbook/UI/Screens/change_review_screen.py` (subclass the pane: `ChangeReviewDiffPane(VerticalScroll)` replacing the bare `VerticalScroll` in compose `:466`; cursor state + render in `_render_diff` `:827-861`)
- Test: `Tests/UI/test_change_review_screen.py` (extend)

Mechanics (spec §3): screen `_cursor_line: int` (reset per file); `_render_diff` styles the cursor line's background (`on grey37` via explicit style — content stays data, never markup) and scrolls it visible; pane `on_key` reclaims ONLY `up`/`down`/`c`/`escape` when the pane is focused (card `on_key` precedent — `event.stop()` + `prevent_default()`), page/home/end stay native; Escape focuses the tree (deliberately shadowing screen-dismiss while pane focused); `enter` (existing `action_focus_diff`) unchanged.

- [ ] **Step 1: Failing tests** — real-provider fixture: focus pane → down moves the styled cursor line (assert via `diff_pane_text`/renderable style spans on the cursor line) and the screen did NOT dismiss on Escape-in-pane (tree focused, screen alive; second Escape dismisses); page-down still scrolls natively; `j`/`k` still switch files; cursor index survives within a file and resets on file switch. RED-prove the escape-shadow test against a pane without the reclaim (temporary Edit-based neuter, restore, evidence in report).
- [ ] **Step 2-4: Red; implement; green** (whole screen suite).
- [ ] **Step 5: Commit** — `feat(console): line cursor + key reclaim in the review diff pane`

---

### Task 7: Comment creation + notes strip

**Files:**
- Modify: `tldw_chatbook/UI/Screens/change_review_screen.py` (`c`/`C` handling, inline Input, notes strip below the pane; provider calls off-thread)
- Test: `Tests/UI/test_change_review_screen.py` (extend)

Mechanics (spec §3/§4): `c` with pane focused + cursor on a diff line → inline `Input` (validated per the card's `_validate_note_text` template) → off-thread `provider.add_change_note(..., anchor_kind="diff_line", diff_line_index=<cursor>, diff_line_text=<line>, hunk_index/<header>/<excerpt> from `split_unified_diff` for the hunk containing the line, snapshot_id=<row id>)`; `C` (screen binding + a small button by the totals) → `anchor_kind="file"`, `hunk_index=-1`, `hunk_header=""`, `hunk_excerpt=""`. Notes strip: a Static/Vertical under the pane listing the focused file's notes (all kinds — filter `notes_for_run(run_id)` by root+path[+snapshot], synchronous read matching the screen's posture), each `kind · text · sent|pending`, pending rows with a `✕` delete (off-thread, pending-only, same rules as the card). Every handler try/except-degrade; the strip refreshes after save/delete and on file switch.

- [ ] **Step 1: Failing tests** — save a line comment → DB row has the exact anchor (kind/index/text/hunk fields/snapshot_id — DB-read assertion); `C` saves a file comment with the sentinels; the strip lists a pre-seeded hunk note + the new file + line comments with kind labels; pending `✕` deletes; a delivered note (stamp directly) shows `sent`, no delete; Escape cancels the input without a row; a provider whose `add_change_note` raises → no crash, warning logged (Edit-based RED-proof of the degrade, like the card's).
- [ ] **Step 2-4: Red; implement; green.**
- [ ] **Step 5: Commit** — `feat(console): diff-line and whole-file review comments`

---

### Task 8: Kind-aware delivery formatters + mixed-batch end-to-end

**Files:**
- Modify: `tldw_chatbook/Chat/console_display_state.py` (`render_diff_feedback_block` note-entry rendering; `format_diff_feedback_disclosure`)
- Test: `Tests/Chat/test_console_diff_hunks.py` + `Tests/Chat/test_console_diff_feedback_delivery.py` (extend)

Formats (spec §5, exact): `file` note → `### <path> — whole file   [run <short-id>]` + `> note` (no fence, no `@@`); `diff_line` note → the standard header line + `> on line: <diff_line_text>` above `> note` + fenced excerpt as today; hunk rendering byte-unchanged (existing exact-format tests must stay green unmodified). Disclosure: `<path> (whole file): "note"` / `<path> <hunk_header> line: "note"`; hunk format unchanged.

- [ ] **Step 1: Failing tests** — exact-format pins for both new kinds in block + disclosure; a MIXED pending batch (hunk + file + line) through the real bridge harness: attaches one block containing all three correctly rendered, stamps exact ids, discloses, and `resume_marker_messages` re-derives the disclosure byte-identical to live; cap behavior with a file note (empty excerpt) sane.
- [ ] **Step 2-4: Red; implement; green** — existing hunk-format tests untouched (byte-parity proof), full delivery + bridge suites.
- [ ] **Step 5: Commit** — `feat(console): kind-aware feedback block and disclosure rendering`

---

### Task 9: Docs + close-out

**Files:**
- Modify: `Docs/User_Guide/console/agent-runs-and-tools.md` (change-review section: the rail section, click-through, line/file comments, counts-honesty wording, pruned tail; stamp refresh)
- Modify: `backlog/tasks/task-18060 - Inspector-rail-multi-file-review-and-review-comments.md` (tick ACs only where genuinely satisfied; Implementation Notes; Done)
- Test: none new; full targeted sweep

- [ ] **Step 1: Verify every doc claim against shipped code BEFORE writing** (V1's false-failure-paragraph lesson; the reviewer will check claims against seams).
- [ ] **Step 2: Run the branch sweep** — `Tests/Chat/test_change_notes_db.py Tests/Chat/test_console_conversation_files.py Tests/Chat/test_console_diff_hunks.py Tests/Chat/test_console_diff_feedback_delivery.py Tests/UI/test_change_review_screen.py Tests/UI/test_console_changed_files_section.py Tests/UI/test_console_changed_files_wiring.py Tests/UI/test_console_turn_file_card_notes.py Tests/UI/test_console_turn_file_card.py Tests/UI/test_console_turn_file_card_factory.py Tests/Chat/test_console_agent_bridge.py` — all green, paste counts.
- [ ] **Step 3: Close out the task file; commit** — `docs(console): review-rail user guide + TASK-18060 close-out`

# Console Turn File Card V1.5 — Annotate Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Users attach notes to specific hunks of an agent turn's diff in the Console transcript; the notes auto-attach to the user's next agent send (with a visible disclosure) and are durably recorded.

**Architecture:** New `change_notes` table in AgentRuns_DB (idempotent DDL + audit version 8); pure hunk segmentation + block rendering in `console_display_state.py`; the card's expand path restructured into per-hunk blocks with an inline note input; delivery entirely inside `ConsoleAgentBridge.run_reply` via the `turn_bundle_block` mechanism, stamped at run completion by exact attached-note ids; disclosure rows emitted live and re-derived on resume (marker precedent); Review button / expand-all / middle-elide polish.

**Tech Stack:** Python 3.11+, Textual 8.x, SQLite (AgentRuns_DB), pytest (venv-only: `VIRTUAL_ENV=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -p no:randomly -q`).

**Spec:** `Docs/superpowers/specs/2026-08-17-console-turn-file-annotate-design.md` — the binding authority; read it before any task. Line anchors below were verified at dev `ed49499b8`.

## Global Constraints

- No exception may escape a Textual `on_*` handler in the card (an escaping exception exits the whole app) — every new handler seam degrades to a logged no-op, matching the card's existing pattern.
- All DB access from the widget goes through the provider and runs off the UI thread (`asyncio.to_thread`); tests use FILE-BACKED `AgentRunsDB(tmp_path / "runs.db", client_id="t")`, never `:memory:` (thread-affinity).
- UI tests run on the real CSS stack (`_SCOPED, _SELF = build_css.screen_css_paths(...)` + `tldw_cli_modular.tcss` — note the return order: scoped first, self last) and query by id/class, never message type.
- The generated CSS bundle is never hand-edited; if a `.tcss` source changes, regenerate via `python -m tldw_chatbook.css.build_css`.
- `[console] turn_file_cards = false` must keep the plain marker byte-identical (existing pinned test must stay green); delivery must work regardless of that switch.
- Note text: max 2,000 chars, validated at the widget boundary.
- Every regression/guard test where a mask is plausible must be shown RED against pre-feature code (checkout or targeted revert) before it counts as evidence.
- Widget-local CSS must not declare `$ds-*` variables (TASK-16811; the focus-contract suite enforces this).

---

### Task 1: `change_notes` persistence in AgentRuns_DB

**Files:**
- Modify: `tldw_chatbook/DB/AgentRuns_DB.py` (DDL block ~`:170-260`; migration/audit block ~`:300-350`; new API methods near `change_snapshots_for_run_review`)
- Test: `Tests/Chat/test_change_notes_db.py` (new)

**Interfaces (Produces):**
```python
def add_change_note(self, *, run_id: str, root: str, path: str,
                    hunk_index: int, hunk_header: str, hunk_excerpt: str,
                    note: str) -> int  # returns note id
def delete_change_note(self, note_id: int) -> bool  # False if delivered or missing
def notes_for_run(self, run_id: str) -> list[dict]  # oldest first, all columns
def pending_notes_for_conversation(self, conversation_id: str) -> list[dict]
    # JOIN change_notes.run_id = agent_runs.id WHERE ar.conversation_id = ?
    #   AND delivered_at IS NULL, ORDER BY cn.id
def mark_notes_delivered(self, note_ids: Sequence[int]) -> None
    # single UPDATE ... WHERE id IN (...) AND delivered_at IS NULL; timestamp _now_iso()
```

- [ ] **Step 1: Write failing tests** — file-backed DB; create two runs in one conversation via `create_run`; cover: add returns id and `notes_for_run` round-trips all fields; `pending_notes_for_conversation` sees notes from BOTH runs, oldest first, and excludes other conversations; `mark_notes_delivered([ids])` stamps only those ids (a note added after the list was captured stays pending — the mid-run race, spec §4); `delete_change_note` deletes pending, returns False for delivered; **migration test**: open a DB file created by the current code (no `change_notes`), reopen with the new code, table exists and `schema_version` contains 8; reopening twice is idempotent.
- [ ] **Step 2: Run tests, verify failures** — `Tests/Chat/test_change_notes_db.py` fails on missing table/methods.
- [ ] **Step 3: Implement** — add DDL to the create block (spec §1 schema verbatim: no `conversation_id` column; partial index `idx_change_notes_pending ON change_notes(run_id) WHERE delivered_at IS NULL`); append `INSERT OR IGNORE INTO schema_version (version) VALUES (8)` following the existing convention comment; implement the five methods with parameterized queries and the file's `_now_iso()`/connection patterns. Google-style docstrings with Args/Returns on all five.
- [ ] **Step 4: Run tests to green**, plus `Tests/Chat/test_change_turn_tracking.py` and `Tests/Workspaces/test_change_revert.py` (schema neighbors) to prove no regression.
- [ ] **Step 5: Commit** — `feat(console): change_notes table + API in AgentRuns_DB (audit v8)`

---

### Task 2: Pure helpers — hunk segmentation, excerpt, feedback block, disclosure text

**Files:**
- Modify: `tldw_chatbook/Chat/console_display_state.py` (beside `turn_file_entries`)
- Test: `Tests/Chat/test_console_diff_hunks.py` (new)

**Interfaces (Produces):**
```python
@dataclass(frozen=True)
class DiffHunk:
    header: str                 # the "@@ -a,b +c,d @@ …" line, verbatim
    body_lines: tuple[str, ...] # lines after the header, up to next header/prelude
    file_prelude: str           # "diff --git…/---/+++" lines (same for every hunk of the file)

def split_unified_diff(text: str) -> list[DiffHunk]
    # ALWAYS over the full diff text; a diff with no @@ (binary/rename-only)
    # yields one DiffHunk(header="", body_lines=<all lines>, file_prelude="")

def hunk_excerpt(hunk: DiffHunk, cap: int = 40) -> str
    # header + first `cap` body lines; "… N more lines" tail when elided

def render_diff_feedback_block(notes: Sequence[dict], *, cap_bytes: int = 16384) -> tuple[str, list[int]]
    # Spec §4 format ("## Diff feedback from the user…"); includes notes
    # oldest-first while the running utf-8 size stays under cap_bytes;
    # returns (block, included_ids). Excluded notes are NOT in the ids
    # list; when any are excluded the block ends with
    # "… N more notes held for the next message". Empty notes -> ("", []).

def format_diff_feedback_disclosure(notes: Sequence[dict]) -> str
    # "📝 Diff feedback attached — <path> <hunk_header>: "<note>"" one line
    # per note; shared verbatim by live emission (Task 5) and resume (Task 6).
```

- [ ] **Step 1: Failing tests** — segmentation against REAL `git diff -M` output captured in the test (build a tmp repo with two commits: multi-file, multi-hunk, a rename, a binary file): hunk count, headers verbatim, body reassembly (`prelude + header + body` per file equals the original diff for text files); no-@@ diff yields the single fallback hunk; `hunk_excerpt` cap + honest tail; `render_diff_feedback_block` includes/excludes at a small `cap_bytes` and returns exactly the included ids with the holdover line; disclosure text exact-format assertions. Adapt the hunk-header regex from `Tools/patch_tool_impls.py:58` (`_HUNK_HEADER`) — do not modify the patch tool.
- [ ] **Step 2: Verify failures.**
- [ ] **Step 3: Implement** (pure, no I/O; docstrings with Args/Returns).
- [ ] **Step 4: Green**, plus `Tests/Chat/test_console_turn_file_entries.py` untouched-green.
- [ ] **Step 5: Commit** — `feat(console): pure hunk segmentation + diff-feedback block rendering`

---

### Task 3: Card restructure — per-hunk blocks

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_turn_file_card.py` (`on_button_pressed` expand path; `_diff_cache` shape; `DEFAULT_CSS`)
- Test: `Tests/UI/test_console_turn_file_card.py` (extend)

**Interfaces:**
- Consumes: `split_unified_diff`, `hunk_excerpt` (Task 2); `provider.diff_text(row, path)` unchanged.
- Produces: expanded diff body = for each hunk, a `Static` (classes `console-turn-file-hunk`, colored via the existing `_styled_diff` applied per hunk: prelude only on the first hunk, then header + capped body) followed by a `Horizontal` action row (classes `console-turn-file-hunk-actions`) that Task 4 will populate — this task mounts it empty. `self._hunk_cache: dict[int, list[DiffHunk]]` replaces the joined-string `_diff_cache`.

- [ ] **Step 1: Failing tests** — expanding a row whose fake-provider diff has 3 hunks mounts exactly 3 `.console-turn-file-hunk` statics and 3 action rows inside that row's diff body; a diff longer than `diff_display_max_lines` still yields one block PER hunk with per-hunk elision (the "hunk past the old cap is still present" case — must be RED against the current flat-Static code); collapse/re-expand reuses the cache (provider `diff_text` called once — count calls on the fake); the existing expand/degrade/crash-regression tests stay green unmodified except where they queried the old flat `.console-turn-file-diff-text` (update those queries to the new classes, preserving their intent).
- [ ] **Step 2: Verify the new tests fail.**
- [ ] **Step 3: Implement** — segment the FULL `diff_text` in the off-thread `_read`; cache `list[DiffHunk]`; mount per-hunk blocks; per-hunk display cap = `max(1, diff_display_max_lines // max(1, len(hunks)))` floor-guarded, with the honest per-hunk elision line from `hunk_excerpt`'s convention; the whole expand body stays inside the existing single try/except-degrade. DEFAULT_CSS gains structural rules only (heights/margins — no `$ds-*` declarations).
- [ ] **Step 4: Green** — full `Tests/UI/test_console_turn_file_card.py` + `Tests/UI/test_console_turn_file_card_factory.py` (byte-parity OFF test untouched) + `Tests/UI/test_console_native_transcript.py` selection test.
- [ ] **Step 5: Commit** — `feat(console): per-hunk blocks in the turn file card`

---

### Task 4: Note UI on hunks

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_turn_file_card.py`; `tldw_chatbook/UI/Screens/change_review_screen.py` (provider note methods)
- Test: `Tests/UI/test_console_turn_file_card_notes.py` (new)

**Interfaces:**
- Provider (`AgentRunsChangeReviewProvider`) gains thin delegates: `add_change_note(...)`, `delete_change_note(note_id)`, `notes_for_run(run_id)` → the Task 1 DB API (Google docstrings; same duck-typed-optional posture the card already uses for `turn_for_run`).
- Card: each hunk action row gets `✎ note` (compact Button, class `console-turn-file-note-btn`, `active_effect_duration = 0`); pressing mounts a one-line `Input` (class `console-turn-file-note-input`, `max_length=2000`) under the hunk; Enter → off-thread `provider.add_change_note` with `hunk_excerpt(hunk)` captured now → on success the input is replaced by a note row (class `console-turn-file-note`, text + `✕` delete button while `delivered_at` is null; a `sent` marker and no delete once delivered); Escape unmounts the input. `_load_rows` also fetches `notes_for_run` off-thread and renders existing notes under their hunks on expand.

- [ ] **Step 1: Failing tests** — REAL provider stack (copy the fixture pattern from `Tests/UI/test_change_review_screen.py::review_fixture` — real `ChangeTurnTracker`/`ShadowRepoService`/file-backed `AgentRunsDB`/`AgentRunsChangeReviewProvider`): pressing `✎ note`, typing, Enter persists a row (assert via `notes_for_run` on the DB) anchored to the right `(hunk_index, hunk_header)` and renders `.console-turn-file-note`; Escape cancels without a row; `✕` deletes pending; a delivered note (stamp it directly in the DB) renders `sent` with no delete button; **resume round-trip**: a brand-new card instance over the same DB shows the note on expand; **live-safety**: with the card mounted inside a `ConsoleTranscript` host, an open input containing typed text survives a `set_messages` sync tick and a selection move (same-instance assertion, the V1 `_update_row_widget` reuse branch); **degrade**: provider whose `add_change_note` raises → press Enter → no crash, input stays, warning logged.
- [ ] **Step 2: Verify failures.**
- [ ] **Step 3: Implement** — all handlers inside try/except-degrade; note text stripped and length-checked via `input_validation` before insert.
- [ ] **Step 4: Green** — new file + Tasks 3's suites.
- [ ] **Step 5: Commit** — `feat(console): hunk note UI on the turn file card`

---

### Task 5: Delivery — attach, stamp-by-id, live disclosure

**Files:**
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py` (attach point `:3343-3353`; completion points that invoke `_append_change_markers` — call sites `:3480`/`:3905`)
- Test: `Tests/Chat/test_console_diff_feedback_delivery.py` (new)

**Interfaces:**
- Inside `run_reply`, immediately after the `turn_bundle_block` append: query `pending_notes_for_conversation(<the same conversation id the marker-append path uses>)` off the run's DB handle; `block, included_ids = render_diff_feedback_block(notes)`; if block truthy, append `f"\n\n{block}"` to the same last-user-message content (outbound copy only); retain `included_ids` on the run context object that reaches the completion seam.
- At completion, in the same place that calls `_append_change_markers` (BESIDE it, outside its `if files:` gating), and only when the run produced assistant output: `mark_notes_delivered(included_ids)` and append one TOOL-role disclosure message with `format_diff_feedback_disclosure(included_notes)` — a plain message with NO `change_review_run_id`.

- [ ] **Step 1: Failing tests** — drive `run_reply` with the existing bridge test harness/fakes for this file (find them: `grep -rn "run_reply" Tests/Chat/ | head`; reuse the established fake-service pattern): (a) pending notes → outbound copy's last user message ends with the block, the STORED message is unchanged, and `included_ids` are stamped after a successful run + disclosure row appended with note content; (b) a run erroring before assistant output → notes still pending, no disclosure; (c) mid-run race: add a new note AFTER `run_reply` captured its list (hook the fake service call) → that note remains pending after completion; (d) over-cap: only included ids stamped, holdover line present, excluded note delivers on a second `run_reply`; (e) no pending notes → payload byte-identical to before this feature (guard against unconditional mutation).
- [ ] **Step 2: Verify failures.**
- [ ] **Step 3: Implement** — every new bridge read/write wrapped so a notes failure NEVER breaks the reply itself (log + skip attach); stamping+disclosure inside the same protective posture the marker emission uses.
- [ ] **Step 4: Green** — new file + the bridge's existing change-marker test files (locate via `grep -rln "format_change_summary_marker\|_append_change_markers" Tests/`).
- [ ] **Step 5: Commit** — `feat(console): diff-feedback auto-attach with exact-id delivery stamping`

---

### Task 6: Disclosure resume re-derivation

**Files:**
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py` (`resume_marker_messages`, `:4536`)
- Test: extend `Tests/Chat/test_console_diff_feedback_delivery.py` + the existing resume-marker test file (locate via `grep -rln "resume_marker_messages" Tests/`)

- [ ] **Step 1: Failing tests** — a DB holding a run with snapshots AND delivered notes: `resume_marker_messages` yields the marker row AND, after it, a disclosure row whose text equals `format_diff_feedback_disclosure` over the delivered notes (grouped by `delivered_at` — two delivery batches yield two rows, in delivery order); pending notes yield NO disclosure; existing resume-marker tests stay green.
- [ ] **Step 2: Verify failures.**
- [ ] **Step 3: Implement** — same synthesized-message shape as live emission (TOOL role, no `change_review_run_id`).
- [ ] **Step 4: Green.**
- [ ] **Step 5: Commit** — `feat(console): re-derive diff-feedback disclosures on resume`

---

### Task 7: Affordances, docs, close-out

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_turn_file_card.py` (header row); `tldw_chatbook/UI/Screens/change_review_screen.py` (`initial_run_id`); `tldw_chatbook/UI/Screens/chat_screen.py` (handle the card's review-request message via the existing v-opener recipe); `tldw_chatbook/Chat/console_display_state.py` (`middle_elide_path`); `Docs/User_Guide/console/agent-runs-and-tools.md`; `backlog/tasks/task-16800 - *.md`
- Test: extend `Tests/UI/test_console_turn_file_card.py`; `Tests/UI/test_change_review_screen.py`

**Interfaces:**
- `ConsoleTurnFileCard.ReviewRequested(Message)` carrying `run_id`, posted by a header `Review` button; `ChatScreen` handles it by opening `ChangeReviewScreen` through the same recipe as the `v` action, passing `initial_run_id`; `ChangeReviewScreen.__init__(..., initial_run_id: str | None = None)` selects that turn on open when present (fall back to latest when absent/unknown).
- Header also gains an expand/collapse-all toggle button; expand-all loads uncached diffs SERIALIZED in one worker.
- `middle_elide_path(path: str, budget: int) -> str` — keeps first + last components, `…` in the middle, returns unchanged when it fits; row labels use it, recomputed on card resize, full path in the row Button's `tooltip`.

- [ ] **Step 1: Failing tests** — Review button posts `ReviewRequested(run_id)` (harness asserts the message, not the full screen push); `ChangeReviewScreen(initial_run_id=run2)` opens with run2's turn selected (real-provider fixture; unknown id falls back to latest); expand-all mounts every row's hunk blocks and collapse-all hides them; `middle_elide_path` unit cases (fits/loose/degenerate one-component); no destructive control exists on the card (assert no button with a revert/undo label/class — AC#4).
- [ ] **Step 2: Verify failures.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Green** — both files + factory byte-parity + a kill-switch delivery test (config OFF: plain marker byte-identical AND Task 5's attach test still delivers pending notes).
- [ ] **Step 5: Docs + close-out** — update the User Guide change-review section (annotate flow, auto-attach disclosure, Review button, expand-all, elision; verify every behavioral claim against the shipped code before writing it — V1's Task 4 lesson) + stamp; tick task-16800 ACs `- [x]` with Implementation Notes.
- [ ] **Step 6: Commit** — `feat(console): review affordance, expand-all, path elision + V1.5 docs`

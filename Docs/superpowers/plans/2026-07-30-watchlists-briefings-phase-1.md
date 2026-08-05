# Watchlists Briefings Phase 1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On-demand text briefings per watchlist — generated, stored, listed and readable in a new Artifacts section, with the queue-for-briefing affordance live.

**Architecture:** A pure selection module (id-watermark windows, three modes) feeds a generation service that makes one `chat_api_call` with a `content_kind`-aware prompt and writes honest statuses (`generating/complete/empty/failed/interrupted`). The Watchlists strip gains a seventh section rendering the artifact list, a briefing view, and Generate. Phase 1 is **preset-less**: `preset_id` NULL, app-default provider, built-in style.

**Tech Stack:** SQLite (additive column-presence idiom), `Chat_Functions.chat_api_call` (the real seam — `chat_with_provider` is stale doc), Textual ≥3.3.0, rich `Markdown(..., hyperlinks=False)`.

**Spec:** `Docs/superpowers/specs/2026-07-30-watchlists-briefings-design.md` — approved, two review rounds. Do not redesign it.

## Global Constraints

- Worktree `/private/tmp/tldw-briefings`, branch `docs/spec-2-watchlists-briefings`, off dev `6929e23cc`. Create the venv: `uv venv && uv pip install --python /private/tmp/tldw-briefings/.venv/bin/python -e ".[dev]"` (`uv venv` is pip-less).
- **pytest is the ONLY way to execute this repo's code** (`/private/tmp/tldw-briefings/.venv/bin/python -m pytest`; a bare `python -c` importing `tldw_chatbook` loads the user's live config). Never bare `pytest`, **never `-q`**. Never `git stash`. Never `git checkout --` to revert — use an editor. **Never any `git worktree` command.** Never write to `~/.config/tldw_cli/` or `~/.local/share/tldw_cli/`.
- New tables/columns: additive `CREATE TABLE IF NOT EXISTS` / column-presence `ALTER` only. **No data migration** — the `BEGIN IMMEDIATE` machinery is not needed; do not cargo-cult it.
- **No new `persist_event` events** — the ADR-029 amendment admits exactly six; observability is artifact status rows.
- The briefing body derives from REMOTE item content through an LLM — render it with `Markdown(..., hyperlinks=False)` (the TASK-1348 rule: prompt-injected `[text](url)` must not become a live hyperlink), and toasts near it use `markup=False`.
- Statuses are never silent: `empty` is a row, `failed` carries its error, a crash yields `interrupted` — all visible in the section.
- Every behavioural change carries a mutation check (revert with an editor, confirm RED, restore). New tests `pytest.mark.unit`. UI geometry asserts in the real-CSS harness (`ProductionCSSDestinationHarness`), on-screen placement not just `height>0`, with a styling mutation.
- Known pre-existing failures NOT yours: 2 tree-chevron; the moving TASK-1345 focus race (create-form/frequency files, passes in isolation); `Tests/DB` chat-image numpy; `Tests/UI/test_chat_shell_bar.py` collection error.

## File structure

| File | Responsibility |
|---|---|
| `tldw_chatbook/DB/Subscriptions_DB.py` | **Modify.** New tables, watchlist columns, queue-flag write, briefing CRUD queries. |
| `tldw_chatbook/Subscriptions/watchlist_normalizers.py` | **Modify.** Carry `queued_for_briefing` (the Phase D read-path lesson: the DB returns it; the normalizer must not drop it). |
| `tldw_chatbook/Subscriptions/briefing_selection.py` | **Create.** Pure selection: modes, watermark, caps. |
| `tldw_chatbook/Subscriptions/briefing_service.py` | **Create.** Generation pipeline, prompt builder, statuses, zombie recovery. |
| `tldw_chatbook/UI/Watchlists_Modules/artifacts_pane.py` | **Create.** The section: list + briefing view + Generate. |
| `tldw_chatbook/UI/Screens/watchlists_collections_screen.py` | **Modify.** Seventh strip section + routing arm (`:275` `_SECTION_DETAIL_TITLE`, `:1163-1228` routing), worker, queue handler. |
| `tldw_chatbook/UI/Watchlists_Modules/inspector_pane.py` + `items_pane.py` | **Modify.** Queue-for-briefing action + indicator. |
| `Tests/Subscriptions/test_briefing_selection.py`, `test_briefing_service.py`; `Tests/Watchlists/test_watchlists_artifacts_pane.py` | **Create.** |

---

### Task 1: DB — tables, watchlist fields, the queue flag's write and read paths

**Files:**
- Modify: `tldw_chatbook/DB/Subscriptions_DB.py` (schema block near `:655`; migration idiom region `:444+`; a write method near `mark_item_status:1500`)
- Modify: `tldw_chatbook/Subscriptions/watchlist_normalizers.py:230+` (`normalize_watchlist_item`)
- Test: `Tests/Subscriptions/test_briefing_selection.py` (create; `pytestmark = pytest.mark.unit`)

**Interfaces (produces — later tasks consume these verbatim):**
- Tables: `briefings(id, watchlist_id, status, error, covers_through_item_id, covers_from_ts, selection_mode, preset_id, model_used, body_markdown, item_count, featured_count, overflow_count, created_at, updated_at)`; `briefing_items(briefing_id, item_id, featured, PRIMARY KEY(briefing_id, item_id))`.
- Watchlist columns: `briefing_selection_mode TEXT DEFAULT 'auto_featured'`, `default_briefing_preset_id INTEGER` (NULL).
- `SubscriptionsDB.set_item_briefing_queued(item_id: int, queued: bool) -> None`
- `SubscriptionsDB.insert_briefing(watchlist_id, status='generating') -> int`; `update_briefing(briefing_id, **fields)`; `get_briefing(briefing_id) -> dict|None`; `list_briefings(watchlist_id) -> list[dict]` (newest first); `latest_completed_watermark(watchlist_id) -> int|None` (max `covers_through_item_id` over status IN ('complete','empty')).
- `normalize_watchlist_item` output gains `queued_for_briefing: bool`.

- [ ] **Step 1: failing tests**

```python
def test_briefings_tables_exist_with_watermark_column():
    db = SubscriptionsDB(":memory:", "test")
    cols = {r[1] for r in db.conn.execute("PRAGMA table_info(briefings)")}
    assert {"watchlist_id", "status", "covers_through_item_id", "body_markdown"} <= cols
    jcols = {r[1] for r in db.conn.execute("PRAGMA table_info(briefing_items)")}
    assert {"briefing_id", "item_id", "featured"} <= jcols
    wcols = {r[1] for r in db.conn.execute("PRAGMA table_info(watchlists)")}
    assert {"briefing_selection_mode", "default_briefing_preset_id"} <= wcols


def test_latest_completed_watermark_ignores_failed_and_interrupted():
    """THE coverage invariant's DB half: failure never advances the window."""
    db = SubscriptionsDB(":memory:", "test")
    w = db.create_watchlist(name="w")  # use the real creation API; check its name
    b1 = db.insert_briefing(w)
    db.update_briefing(b1, status="complete", covers_through_item_id=40)
    b2 = db.insert_briefing(w)
    db.update_briefing(b2, status="failed", covers_through_item_id=99, error="boom")
    b3 = db.insert_briefing(w)
    db.update_briefing(b3, status="empty", covers_through_item_id=55)
    assert db.latest_completed_watermark(w) == 55  # empty advances; failed never


def test_queue_flag_round_trips_through_the_normalizer():
    """Phase D's read-path lesson: the DB returns the flag; the normalizer
    must carry it, or every downstream consumer sees un-queued items."""
    # insert an item, set_item_briefing_queued(True), fetch via get_new_items,
    # normalize -> item["queued_for_briefing"] is True; set False -> False
```

(Write the real body for the third test; find the real watchlist-creation API by reading the DB class — do not guess it into existence.)

- [ ] **Step 2:** run → FAIL (no tables/methods).
- [ ] **Step 3:** implement. Schema in the CREATE block; `ALTER` via the column-presence idiom for `watchlists` (same pattern as `content_kind`, `:562-567`). Indexes: `briefings(watchlist_id, status)`, `briefing_items(item_id)`. CRUD methods use `self.transaction()`. `latest_completed_watermark` = `SELECT MAX(covers_through_item_id) FROM briefings WHERE watchlist_id=? AND status IN ('complete','empty')`.
- [ ] **Step 4:** run → PASS.
- [ ] **Step 5: mutations.** (a) include `'failed'` in the watermark statuses → the invariant test REDs. (b) drop the normalizer's `queued_for_briefing` line → round-trip test REDs.
- [ ] **Step 6:** commit `feat(briefings): tables, watchlist fields, queue flag read/write`.

---

### Task 2: selection — modes, watermark, caps

**Files:**
- Create: `tldw_chatbook/Subscriptions/briefing_selection.py`
- Test: `Tests/Subscriptions/test_briefing_selection.py` (extend)

**Interfaces:**
- Produces: `select_briefing_items(db, watchlist_id, *, mode: str, item_cap: int = 40, now: datetime | None = None) -> BriefingSelection` where `BriefingSelection` is a dataclass: `items: list[dict]` (normalized), `featured_ids: set[int]`, `overflow_count: int`, `covers_through_item_id: int | None` (max item id CONSIDERED — the new watermark; None when the window held nothing), `covers_from_ts: str`.
- Consumes: Task 1's watermark query; `watchlist_sources` for membership; `normalize_watchlist_item`.

**The rules, verbatim from the spec:**
- Window = items of the watchlist's current sources with `id > latest_completed_watermark`. First briefing (no watermark): last 7 days by `created_at`, item-capped. The watermark recorded is the max id **considered** (including overflow items — they were seen and reported as overflow, honestly), EXCEPT when the window held nothing: `None` means "do not advance".
- `auto`: window items only. `curated`: queued-and-not-covered-by-this-watchlist only (junction lookup), **window-exempt**. `auto_featured` (default): union; queued items window-exempt and `featured`.
- Overflow: newest items win the cap; the count of dropped ones returns as `overflow_count`. Featured items are never dropped by the cap (cap squeezes the auto side first); if queued items alone exceed the cap, keep the newest and count the rest as overflow too.

- [ ] **Step 1: failing tests** — real bodies, seeded through the real DB (insert subscriptions, `watchlist_sources` rows, items via `persist_subscription_item`):

```python
def test_watermark_window_excludes_a_late_added_sources_backlog():
    """The id watermark's free flood-fix: a source added after briefing 1 has
    historical items with ids below the watermark -- auto-excluded."""

def test_failed_briefing_does_not_advance_selection():
    """Generate window A -> complete at watermark X. Insert items. A failed
    briefing row with a HIGHER covers_through_item_id must not move the next
    selection: it still starts at X."""

def test_queued_items_bypass_the_window_in_both_modes():
    """A queued item OLDER than the watermark appears in curated AND
    auto_featured selections, marked featured in the latter; it does not
    appear in plain auto."""

def test_curated_excludes_items_this_watchlist_already_covered():
    """Junction rows for watchlist W exclude re-selection in W; the same item
    still selects for watchlist V (the global-queue-never-cleared rule)."""

def test_overflow_counts_dropped_items_and_features_survive_the_cap():
    """cap=3, five window items + two queued: both queued kept + newest auto
    item; overflow_count == 4... compute the exact numbers in the test and
    assert identities, not just counts."""
```

- [ ] **Step 2:** FAIL → **Step 3:** implement (one module-level function + dataclass; SQL: items joined through `watchlist_sources`; keep it a read-only module — junction WRITES belong to the service). → **Step 4:** PASS.
- [ ] **Step 5: mutations.** (a) watermark comparison `id >` → `created_at >` — the late-added-source test REDs. (b) drop the junction exclusion → the curated test REDs. (c) let the cap drop featured items → the overflow test REDs.
- [ ] **Step 6:** commit `feat(briefings): selection with id-watermark windows and three modes`.

---

### Task 3: the generation service

**Files:**
- Create: `tldw_chatbook/Subscriptions/briefing_service.py`
- Test: `Tests/Subscriptions/test_briefing_service.py` (create)

**Interfaces:**
- Produces:
  - `build_briefing_prompt(items: list[dict], featured_ids: set[int], overflow_count: int) -> tuple[str, str]` (system, user) — pure, tested directly.
  - `async generate_briefing(db, watchlist_id, *, chat=chat_api_call, provider: str | None = None, model: str | None = None) -> dict` (the finished briefing row).
  - `fail_interrupted_briefings(db, watchlist_id | None = None) -> int` — zombie recovery: any `generating` row (optionally scoped) becomes `failed` with error `"interrupted"`; returns count.
- Consumes: Task 1 CRUD, Task 2 selection, `Chat_Functions.chat_api_call` (`:698` — `api_endpoint`, `messages_payload`, `model`, `streaming=False`, `max_tokens`). Default provider: the app's configured default endpoint (read it the way the Chat tab does — find the accessor in `config.py` near `:5324`; do not hardcode `"openai"`).

**Pipeline (spec, condensed):** insert `generating` row → `fail_interrupted_briefings` is NOT called here (the UI calls it; the service assumes the guard) → select → if no items: status `empty`, watermark advanced iff selection returned one (it returns None for a truly empty window — then record `covers_through_item_id` as the PRIOR watermark, unchanged) → else build prompt → one `chat` call, `streaming=False` → on success: status `complete`, body, counts, junction rows (featured flag), watermark → on exception: status `failed`, error text, **no junction rows, watermark untouched**.

**Prompt is `content_kind`-aware:** article items contribute title/source/excerpt (per-item excerpt cap ~800 chars, stated); change items contribute the diff **labelled as a diff of the named page**. Featured items listed first under a "Queued by you" framing. Overflow stated in the prompt AND appended to the body by the service ("N more items arrived in this window and are not covered") so the note survives even if the model ignores instructions.

- [ ] **Step 1: failing tests** — fake ONLY the `chat` seam (an async/sync callable returning a canned markdown string, or raising):

```python
def test_generation_happy_path_writes_everything():
    # complete; body contains the canned text AND the overflow sentence when
    # overflow>0; junction has featured flags; watermark == selection's

def test_llm_failure_is_honest_and_loses_nothing():
    # chat raises -> status failed + error text; NO junction rows;
    # latest_completed_watermark unchanged; a second generate call re-selects
    # THE SAME items (assert identities) -- the spec's named invariant

def test_empty_window_is_a_row_not_an_absence():
    # no items -> status empty, item_count 0, no chat call (assert the fake
    # was never invoked), watermark unchanged

def test_prompt_labels_diffs_as_diffs():
    # build_briefing_prompt with one article + one change item: the change
    # item's section contains its diff and the words identifying it as a
    # page-change; the article's contains its excerpt; featured first

def test_interrupted_recovery_only_touches_generating_rows():
    # rows in every status; fail_interrupted_briefings -> only 'generating'
    # became failed/interrupted; returns the count
```

- [ ] **Step 2:** FAIL → **Step 3:** implement → **Step 4:** PASS.
- [ ] **Step 5: mutations.** (a) advance the watermark on failure → the invariant test REDs (this is the plan's most important mutation). (b) drop the service-side overflow append → happy-path REDs. (c) make recovery touch `complete` rows too → recovery test REDs.
- [ ] **Step 6:** commit `feat(briefings): generation service with honest statuses`.

---

### Task 4: the Artifacts section

**Files:**
- Create: `tldw_chatbook/UI/Watchlists_Modules/artifacts_pane.py`
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py` — `_SECTION_DETAIL_TITLE` (`:275`), the routing arms (`:1163-1228` — add an `artifacts` arm), a `run_worker` handler (own `group="wl-briefing"`, `exclusive=True` — the TASK-1362 lesson: never exclusive without a group), CSS feature file + regenerated bundle.
- Test: `Tests/Watchlists/test_watchlists_artifacts_pane.py` (create)

**Behaviour:**
- The strip gains **Artifacts** (7th section; full-width like Sources — the CONTENT gate keys on `active_section != "items"` and needs NO change, but assert it: opening Artifacts must not mount CONTENT).
- Pane: a DataTable of briefings (status, window, item/featured/overflow counts, created) + a detail area rendering the selected briefing's `body_markdown` via `Markdown(body, hyperlinks=False)` — the body derives from remote content through an LLM; injected `[text](url)` must render inert (Global Constraints).
- **Generate** button: refuses with a toast while a `generating` row exists for the watchlist (`markup=False` toasts); otherwise `run_worker` → `fail_interrupted_briefings(db, watchlist_id)` first (the zombie guard lives HERE, before the check) → `generate_briefing` → repaint the list only (no full-screen recompose — patch/refresh the pane, the Phase D recompose lesson).
- A `failed` row shows its error in the detail area; an `empty` row says the window was empty. Sections/tab keys: follow how `notifications` was added (grep its arm) — same message/`RowHighlighted` guards (`table.has_focus`) as the other panes.

- [ ] **Step 1: failing tests** (real-CSS harness for geometry; screen harness for wiring):
  1. Artifacts section exists in the strip; selecting it mounts the pane full-width; CONTENT stays unmounted.
  2. Generate → (faked chat seam at the service boundary) → a row appears with status `complete`; the detail renders the body; a hostile `[click](javascript:x)` in the body paints as literal text (render through a real Console/strips — the 1348 assertion pattern).
  3. A pre-seeded `generating` row: Generate refuses with the toast; after `fail_interrupted_briefings` the same click proceeds.
  4. Geometry: list + Generate + detail visible on-screen at 160×42 and 180×50 in the real-CSS harness; styling mutation (the pane's own CSS rule broken) REDs.
- [ ] **Step 2-4:** FAIL → implement → PASS. Regenerate the bundle via `build_css.py`; never hand-edit it.
- [ ] **Step 5: mutations.** (a) remove the zombie-recovery call → test 3's second half REDs. (b) `hyperlinks=False` dropped → the hostile-link test REDs.
- [ ] **Step 6:** commit.

---

### Task 5: queue-for-briefing affordance

**Files:**
- Modify: `tldw_chatbook/UI/Watchlists_Modules/inspector_pane.py` (a `QueueForBriefingRequested`-style message + button for ITEM selections, next to Ingest/Ignore — follow `SaveNoiseSelectorsRequested`'s exact shape), `items_pane.py` (a queued indicator in the row: a `Q` marker column), `watchlists_collections_screen.py` (handler → `db.set_item_briefing_queued`, in-place patch + `update_item_status_cell`-style repaint, **no recompose**, honest failure toast).
- Test: extend `Tests/UI/test_watchlists_inspector.py` + `Tests/Watchlists/`.

- [ ] **Step 1: failing tests:** press the real button → DB flag flips → the row indicator repaints in place (same `ItemsPane` instance — assert instance survival, the Phase D pattern); toggling off works; a failed write leaves the indicator unchanged with an error toast.
- [ ] **Step 2-4:** FAIL → implement → PASS.
- [ ] **Step 5: mutations.** (a) handler no-op → DB test REDs. (b) repaint dropped → indicator test REDs while the DB half stays green (layered).
- [ ] **Step 6:** commit.

---

### Task 6: close-out

- Full sweep: `Tests/Subscriptions/ Tests/Scheduling/ Tests/Watchlists/ Tests/UI/ -k watchlist` — expect only the documented baselines.
- File a backlog task for **phase 1 delivered** retrospectively? No — instead: run the cross-worktree ID scan, file ONE task for spec #2 tracking if none exists, with phase 1 checked `[x]` and phases 2-4 unchecked, so the board knows the programme state.
- Spec `Status: proposed` → `Status: phase 1 implemented; phases 2-4 pending`.
- Update `Docs/superpowers/specs/2026-07-25-watchlists-console-rebuild-design.md`? No — leave spec #1 alone; its Artifacts note now points at reality.
- Commit.

## Self-review

**Spec coverage (phase 1 scope):** tables+fields → T1; watermark/modes/caps/flood → T2; pipeline, `content_kind` prompts, overflow honesty, statuses, zombie rule → T3 (+T4 for where recovery is invoked); Artifacts section + reader + Generate → T4; queue affordance → T5; preset-less stated in header and T3 (provider/model params default to app config); no-new-persist-events, hyperlinks=False, markup=False toasts → Global Constraints + T4 tests. Phases 2-4 absent by design.

**Placeholders:** T1 step-1 third test and T2/T3 step-1 tests carry docstring-plus-contract rather than full bodies where they depend on the real watchlist-creation API — each such spot names the exact thing the implementer must read first (the Phase D convention: skeletal by design, full bodies mandatory, and the dispatch must say so).

**Type consistency:** `BriefingSelection` fields match T3's consumption; `covers_through_item_id: int|None` semantics identical in T1 query, T2 dataclass, T3 pipeline; `set_item_briefing_queued` name identical in T1/T5.

# Watchlists Briefings Phase 2a — Presets, Script Casting, Mode Picker, Citations

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A complete briefing can be cast into an N-speaker script from a user-defined preset; watchlists gain their selection-mode/default-preset writers; `[item N]` citations become navigable with honest pruned-item degradation.

**Architecture:** Mirrors phase 1 exactly — dumb DB layer (`Subscriptions_DB.py`), a pure-ish service module (`briefing_cast.py`, one faked seam: `chat_api_call`), UI as pane + screen wiring with all DB work through `asyncio.to_thread`. Scripts snapshot their roster at cast time so preset edits/deletes never orphan an artifact's meaning. **Audio is NOT in this plan** — plan-time adapter verification found the TTS layer needs a synthesis façade, a real stitching primitive, and a private-storage decision; that is phase 2b, its own plan.

**Tech Stack:** Python 3.11, Textual, SQLite, pytest (`.venv/bin/python -m pytest`, plain output).

## Global Constraints

- pytest is the ONLY python entry point for this repo's code. Never bare `python -c` importing `tldw_chatbook` (loads the user's live config).
- Never `git stash`; never `git checkout --`/`git restore` to revert (Edit-tool reverts only); never any `git worktree` command; never `-q` with pytest.
- Never hand-edit `tldw_chatbook/css/tldw_cli_modular.tcss` — edit `css/features/_watchlists.tcss` and regenerate via `cd tldw_chatbook/css && ../../.venv/bin/python build_css.py`.
- All DB calls from the screen go through `asyncio.to_thread` (phase-1 ruling, thrice-enforced). Workers: `group=` always set; guard flags claimed at dispatch, cleared in `finally`.
- Toasts whose body interpolates any value: `markup=False`. Model/item text never enters a markup-parsing surface unescaped; DataTable cells for such text use `rich.text.Text` objects.
- `Markdown(..., hyperlinks=False)` stays — citations are widget affordances, NOT markdown/OSC-8 links (both panes document this constraint).
- Exception logging is type-only (`type(exc).__name__`); never `logger.opt(exception=True)`; never log prompt/roster/body content.
- No new `persist_event` event names (ADR-029 admits exactly six).
- Every new test carries `pytest.mark.unit` (or the file's existing `pytestmark`); an unmarked test in `Tests/Watchlists` is collected by nothing.
- Every behavioural change gets a revert-confirm-RED-restore mutation check (Edit-tool reverts only). Green-under-mutation is acceptable only when documented non-load-bearing with a cross-reference to the test carrying the claim.
- Spec: `Docs/superpowers/specs/2026-07-30-watchlists-briefings-design.md` §"Casting and audio (phase 2)" (script half), §"UI (phases 1–2)", §Testing. Named invariant this plan owns: **unknown-speaker-fails-the-script-by-name** and **citation-to-pruned-item-degrades**.
- The briefing is NEVER touched by a script failure (spec §Error handling ethos).

## File Structure

- `tldw_chatbook/DB/Subscriptions_DB.py` — `briefing_presets` + `briefing_scripts` DDL, preset/script CRUD, `set_watchlist_briefing_settings`, `get_subscription_items_by_ids` (Task 1)
- `tldw_chatbook/Subscriptions/briefing_cast.py` — NEW: roster validation, cast prompt, strict turn parsing, `generate_script`, `fail_interrupted_scripts` (Task 2)
- `tldw_chatbook/Subscriptions/briefing_service.py` — `generate_briefing` gains `preset_id` plumbing (Task 2)
- `tldw_chatbook/UI/Watchlists_Modules/briefing_preset_modal.py` — NEW: preset manager modal (list + editor + delete) (Task 3)
- `tldw_chatbook/UI/Watchlists_Modules/artifacts_pane.py` — pickers, Cast button, scripts table, script detail, citations table (Tasks 4-6)
- `tldw_chatbook/UI/Screens/watchlists_collections_screen.py` — handlers + workers for all of the above (Tasks 3-6)
- Tests: `Tests/Subscriptions/test_briefing_presets_db.py`, `Tests/Subscriptions/test_briefing_cast.py`, `Tests/Watchlists/test_watchlists_briefing_presets_ui.py`, extensions to `Tests/Watchlists/test_watchlists_artifacts_pane.py`

Phase-1 sketch convention applies: where a step shows docstring-plus-contract instead of a full body, the full body is mandatory and the implementer must read the named precedent first.

---

### Task 1: DB foundation — presets, scripts, watchlist writers, item lookup

**Files:**
- Modify: `tldw_chatbook/DB/Subscriptions_DB.py` (DDL after the `briefing_items` block at ~`:731`; methods after `latest_completed_watermark` at ~`:1731`)
- Test: `Tests/Subscriptions/test_briefing_presets_db.py` (new)

**Interfaces — Produces (later tasks rely on these EXACT names):**
```python
insert_briefing_preset(self, name: str, *, roster_json: str, style_notes: str | None = None,
                       provider: str | None = None, model: str | None = None) -> int
update_briefing_preset(self, preset_id: int, **fields) -> None   # allowlist: name, roster_json,
                       # style_notes, provider, model, updated_at; unknown key -> ValueError
get_briefing_preset(self, preset_id: int) -> Optional[Dict[str, Any]]
list_briefing_presets(self) -> List[Dict[str, Any]]              # name ASC
delete_briefing_preset(self, preset_id: int) -> bool             # hard delete; scripts snapshot
insert_briefing_script(self, briefing_id: int, *, preset_id: int | None,
                       preset_name: str, roster_snapshot_json: str,
                       status: str = "generating") -> int
update_briefing_script(self, script_id: int, **fields) -> None   # allowlist: status, error,
                       # turns_json, model_used, updated_at
get_briefing_script(self, script_id: int) -> Optional[Dict[str, Any]]
list_briefing_scripts(self, briefing_id: int) -> List[Dict[str, Any]]   # newest first
set_watchlist_briefing_settings(self, watchlist_id: int, *,
                                selection_mode: str | None = None,
                                default_preset_id: object = _UNSET) -> None
get_subscription_items_by_ids(self, item_ids: Sequence[int]) -> Dict[int, Dict[str, Any]]
```

- [ ] **Step 1: Read the phase-1 shapes first.** `insert_briefing`/`update_briefing`/`get_briefing`/`list_briefings` (`Subscriptions_DB.py:1615-1731`) are the byte-for-byte precedent: `transaction()` on every operation (Qodo rule 1011851 — reads too), allowlist tuple + `sql_validation.validate_identifier` on `update_*` keys, Google docstrings.

- [ ] **Step 2: Write the failing tests.** Same harness as `test_briefing_selection.py` (real `SubscriptionsDB` on `tmp_path`, `WatchlistBundleService` for watchlists, `pytestmark = pytest.mark.unit`). Cases: preset round-trip incl. NULLs; update allowlist rejects unknown key by name; list ordered name ASC; delete returns False for missing id; script round-trip (`briefing_id` FK cascades on briefing delete — assert); `list_briefing_scripts` newest first; `update_briefing_script` allowlist; `set_watchlist_briefing_settings` writes `briefing_selection_mode` (reject a mode outside `('auto', 'curated', 'auto_featured')` with ValueError naming it) and writes/clears `default_briefing_preset_id` (`_UNSET` sentinel leaves it alone; `None` clears — test both); `get_subscription_items_by_ids` returns only existing rows keyed by id, empty input → `{}`, and bound-parameter count is chunked ≤ 500 per statement (the Qodo NOT-IN lesson — spy the connection like `test_briefing_selection.py`'s param-count test).

- [ ] **Step 3: Run tests, confirm they fail on missing attributes.**

- [ ] **Step 4: Implement.** DDL: `briefing_presets(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, style_notes TEXT, provider TEXT, model TEXT, roster_json TEXT NOT NULL, created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP)`; `briefing_scripts(id INTEGER PRIMARY KEY AUTOINCREMENT, briefing_id INTEGER NOT NULL REFERENCES briefings(id) ON DELETE CASCADE, preset_id INTEGER, preset_name TEXT NOT NULL, roster_snapshot_json TEXT NOT NULL, turns_json TEXT, status TEXT NOT NULL DEFAULT 'generating', error TEXT, model_used TEXT, created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP)` + `CREATE INDEX IF NOT EXISTS idx_briefing_scripts_briefing ON briefing_scripts(briefing_id, status)`. Additive `CREATE TABLE IF NOT EXISTS`, no `BEGIN IMMEDIATE` (spec forbids cargo-culting it). The mode tuple in `set_watchlist_briefing_settings` carries a greppable pact comment naming `briefing_selection.VALID_MODES` (the TASK-1393 ordering-pact convention — DB cannot import `Subscriptions/`).

- [ ] **Step 5: Run the new file + `Tests/Subscriptions/test_briefing_selection.py` (schema neighbours). Mutation checks:** (a) drop `validate_identifier` from `update_briefing_script` → allowlist test alone stays green, so the divergence test from the Qodo round is the precedent — add the same shape and confirm it REDs; restore. (b) invert the `_UNSET` sentinel handling → the leaves-it-alone test REDs; restore.

- [ ] **Step 6: Commit** `feat(briefings): presets and scripts tables, watchlist briefing settings writer`.

---

### Task 2: Cast service — roster validation, strict turn parsing, generate_script, preset plumbing

**Files:**
- Create: `tldw_chatbook/Subscriptions/briefing_cast.py`
- Modify: `tldw_chatbook/Subscriptions/briefing_service.py` (`generate_briefing:451`, `_default_provider:263`)
- Test: `Tests/Subscriptions/test_briefing_cast.py` (new), `Tests/Subscriptions/test_briefing_service.py` (preset plumbing cases)

**Interfaces — Consumes:** Task 1 CRUD verbatim; `briefing_service._invoke_chat:288` (copy its sync/async-seam shape, do not import the private).
**Interfaces — Produces:**
```python
VALID_STATUSES = ("generating", "complete", "failed")           # scripts have no 'empty'
class ScriptCastError(RuntimeError): ...                        # every failure names its cause
def validate_roster(roster: object) -> list[dict]               # returns normalized speakers;
    # each: {"name": str non-empty unique, "role_prompt": str, "character_card_id": int|None,
    #        "voice_profile_id": str|None}; raises ScriptCastError naming the defect
def dump_roster(roster: list[dict]) -> str                      # canonical JSON for storage
def load_roster(text: str) -> list[dict]                        # inverse; ScriptCastError on junk
def build_cast_prompt(body_markdown: str, roster: list[dict], style_notes: str | None,
                      character_texts: dict[str, str]) -> tuple[str, str]
def parse_script_turns(text: str, roster_names: set[str]) -> list[dict]
    # -> [{"speaker": str, "text": str}]; fence-strip + first-[...]-slice recovery, then STRICT:
    # non-array / non-object turn / missing keys -> ScriptCastError naming the parse defect;
    # a speaker not in roster_names -> ScriptCastError naming THE SPEAKER (named invariant)
async def generate_script(db, briefing_id: int, *, preset_id: int,
                          chat: Callable[..., Any] = chat_api_call,
                          load_character: Callable[[int], Optional[dict]] | None = None,
                          provider: str | None = None, model: str | None = None) -> dict
def fail_interrupted_scripts(db, briefing_id: int | None = None) -> int
```

- [ ] **Step 1: Read first:** `briefing_service.py` whole file (the error-boundary contract: try/except wraps the chat call and the parse ONLY; DB errors propagate — the screen's worker wraps); `Character_Chat/character_generation.py:218` `parse_whole_character_response` (the fence-strip/slice/named-error shape — this plan's parser targets an ARRAY, so the recovery slices `[`…`]`).

- [ ] **Step 2: Failing tests, the contract cases:** roster: duplicate speaker name fails naming it; empty roster fails; single-speaker roster is valid (spec: "a roster of one produces narration through the identical path — no special mode"). Parser: valid array round-trips; fenced ```json array recovers; prose-wrapped array recovers via slice; unknown speaker "Dave" fails with "Dave" in the message (**named invariant test: `test_an_unknown_speaker_fails_the_script_by_name`**); non-string text fails naming the turn index. `generate_script` (real DB, `_FakeChat`-style scripted chat): happy path writes `complete` + `turns_json` + `model_used` + roster snapshot; briefing not `complete` → refused with ScriptCastError BEFORE any row insert (empty/failed/generating briefings cannot cast); missing preset → ScriptCastError, no row; roster references card id 7, `load_character` returns None → **failed script row** with error naming card 7 (spec: "fails the cast at that point, naming the card"); `load_character=None` with a card-bearing roster → same failed-naming path; chat raises → failed row, error type-only, **briefing row byte-identical before/after** (assert full dict equality — the spec's briefing-never-touched rule); parse failure → failed row naming the defect; DB error (close the connection) propagates. `fail_interrupted_scripts` mirrors phase 1's `fail_interrupted_briefings:566`. Preset plumbing on `generate_briefing`: `preset_id=` param resolves provider/model (explicit args still win), appends `style_notes` to the system prompt, records `preset_id` on the briefing row; a missing preset id is recorded as None and generation proceeds on defaults (a deleted preset must not brick generation).

- [ ] **Step 3: Run, confirm RED.**  

- [ ] **Step 4: Implement.** `generate_script` order: load briefing (must be `complete`) → load preset → `validate_roster(load_roster(...))` → resolve character texts (each `character_card_id`: `load_character(card_id)`; None → fail the script naming the card id; contribution is the card's `personality` + `description` fields, truncated to 1000 chars each) → insert `generating` row with snapshot (snapshot embeds resolved `character_name` per speaker) → try: `_invoke_chat`-shaped call + `parse_script_turns` → `complete`; except `ScriptCastError`/chat exception → `failed` + capped `str(exc)[:1000]`, type-only log. `build_cast_prompt` system prompt: the roster with role prompts and character texts, style notes, the fixed output contract ("Respond with ONLY a JSON array of {\"speaker\", \"text\"} turns; speaker must be one of: <names>"). User prompt: the briefing body verbatim.

- [ ] **Step 5: Run all three test files. Mutation checks:** (a) parser accepts unknown speakers → the named-invariant test REDs; restore. (b) `generate_script` catches DB errors too (widen the except) → the propagation test REDs; restore. (c) drop the briefing-equality assertion's guard (make failure write `body_markdown=None` on the briefing) → the byte-identical test REDs; restore.

- [ ] **Step 6: Commit** `feat(briefings): cast service — strict turns, honest failures, preset plumbing`.

---

### Task 3: Preset manager modal

**Files:**
- Create: `tldw_chatbook/UI/Watchlists_Modules/briefing_preset_modal.py`
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py` (mount/dismiss wiring only)
- Modify: `tldw_chatbook/css/features/_watchlists.tcss` + regenerate bundle
- Test: `Tests/Watchlists/test_watchlists_briefing_presets_ui.py` (new)

**Interfaces — Consumes:** Task 1 preset CRUD; Task 2 `validate_roster`/`dump_roster`/`load_roster`/`ScriptCastError`.
**Interfaces — Produces:** `class BriefingPresetModal(ModalScreen[bool])` — constructor `(db, *, character_options: list[tuple[str, int]], voice_options: list[tuple[str, str]])`; dismisses `True` if anything changed (screen reloads preset lists on True). Message-free: the modal owns its DB writes (through `asyncio.to_thread`), the SCREEN owns none of them.

- [ ] **Step 1: Read first:** `UI/stts_profile_library.py` `TTSProfileEditorModal:280` + `TTSProfileDeleteModal:481` (the repo's modal-editor idiom: compose, validate-with-inline-error, dismiss protocol) and the phase-1 pane's compact-button/height-1 toolbar constraints (`artifacts_pane.py:147`).

- [ ] **Step 2: Failing tests** (real-CSS harness — `ProductionCSSDestinationHarness` per the three-way-vacuity lesson; geometry asserts on-screen placement + a styling mutation that REDs): modal lists presets name-ASC; create with two speakers persists exactly the roster entered (assert via `load_roster` on the DB row); duplicate speaker name shows the inline error and does NOT persist (assert row count unchanged — `validate_roster` is the gate, its message rendered `markup=False`-safe); delete asks confirmation and hard-deletes; editing preserves untouched fields; character Select offers the passed options and stores the card id; voice Input/Select stores `voice_profile_id` inert (2a records it; 2b consumes it — say so in a code comment). Speaker rows are dynamic: "Add speaker" appends a row, per-row "Remove" deletes it, one-row minimum enforced.

- [ ] **Step 3-4: Implement + green.** All writes `asyncio.to_thread`. Character/voice option lists are passed IN by the screen (the modal never queries other DBs): screen builds `character_options` via `asyncio.to_thread(app.chachanotes_db.list_character_cards)` → `[(name, id)]` when that DB is bound, else `[]` with the Select disabled + tooltip; `voice_options` likewise from `app._tts_profile_service.list_profiles` when bound, else disabled. A missing service degrades the FIELD, never the modal.

- [ ] **Step 5: Mutation:** remove the `validate_roster` call from save → the duplicate-name test REDs; restore. Styling mutation: drop the modal's width/height rules → the geometry test REDs; restore (regenerate bundle both times).

- [ ] **Step 6: Commit** `feat(briefings): preset manager modal`.

---

### Task 4: Toolbar pickers — selection mode, default preset, Presets… entry

**Files:**
- Modify: `tldw_chatbook/UI/Watchlists_Modules/artifacts_pane.py` (toolbar at `:147`; new reactives)
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py` (`_load_briefings:3078`, generate handler `:3188`)
- Test: extend `Tests/Watchlists/test_watchlists_artifacts_pane.py`

**Interfaces — Consumes:** Task 1 `set_watchlist_briefing_settings`, `list_briefing_presets`, watchlist row fields `briefing_selection_mode`/`default_briefing_preset_id`; Task 3 modal.
**Interfaces — Produces:** pane reactives `selection_mode: str`, `presets: list[dict]`, `default_preset_id: int | None` (all `recompose=True`); messages `BriefingModeChanged(mode: str)`, `BriefingDefaultPresetChanged(preset_id: int | None)`, `ManagePresetsRequested()`.

- [ ] **Step 1: Read the phase-1 wiring first:** `_load_briefings` (pushes pane reactives, never full-screen recompose), the Generate handler's dispatch-time guard claim, `handle_toggle_briefing_queue_requested` (the to_thread write + in-place patch shape).

- [ ] **Step 2: Failing tests:** two compact `Select`s + a `Presets…` `Button` render in the toolbar only when `can_generate` (watchlist scope); mode Select shows the watchlist's stored mode on load (seed `curated` via Task 1's writer, assert the Select's value — the read path pin); changing mode writes through `set_watchlist_briefing_settings` off-loop (thread-identity pin, the established pattern) and does NOT recompose the screen (instance-survival assertion); preset Select lists presets + "App default" (None); choosing one persists `default_briefing_preset_id`; **Generate now casts the die**: with a default preset stored, `generate_briefing` is invoked with `preset_id=<that id>` (assert on a captured fake); with none, `preset_id=None`. `Selection` mode reality test: set `curated` via the picker, generate, assert the briefing row's `selection_mode == "curated"` — the phase-1 deferral's dead branch is now REACHABLE end-to-end (this is the test that retires the deferral).
- [ ] **Step 3-4: Implement + green.** `Select.Changed` handlers on the pane post the typed messages; the SCREEN writes (to_thread) then patches its `_loaded_*` memory in place — no reload. Guard the Textual `Select` mount-fires-Changed trap (the Library lesson): ignore events whose value equals current state.
- [ ] **Step 5: Mutations:** (a) drop the screen's write call → the persistence test REDs while the Select still shows the new value (the honest split); (b) make Generate ignore the stored preset → the cast-the-die test REDs. Restore both.
- [ ] **Step 6: Commit** `feat(briefings): selection-mode and default-preset pickers — the deferred writers exist`.

---

### Task 5: Cast action + scripts in the Artifacts pane

**Files:**
- Modify: `tldw_chatbook/UI/Watchlists_Modules/artifacts_pane.py` (`_detail_renderable:228`, compose, new table)
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`
- Test: extend `Tests/Watchlists/test_watchlists_artifacts_pane.py`

**Interfaces — Consumes:** Task 2 `generate_script`/`fail_interrupted_scripts`/`load_roster`; Task 1 `list_briefing_scripts`.
**Interfaces — Produces:** pane reactives `scripts: list[dict]`, `selected_script: dict | None`; messages `CastScriptRequested()` (uses the pane's `default_preset_id`; Cast disabled with tooltip when None and no presets exist), `ScriptSelected(script: dict | None)`.

- [ ] **Step 1: Read first:** the whole `_generate_briefing` worker chain (`:3188-3405`) — dispatch-time guard claim, `_sweep_and_guard`, `finally`-clears, refusal toast copy discipline. The cast worker is its sibling: own flag `_cast_in_flight`, own `group="wl-cast"` (exclusive), claimed at dispatch, `fail_interrupted_scripts` swept via to_thread before refusing, all DB in `asyncio.to_thread`, `is_attached` before UI mutation.
- [ ] **Step 2: Failing tests** (fake chat at the screen's `generate_script` reference, everything below real — the phase-1 seam discipline): Cast on a complete briefing writes a script row and the scripts table shows preset name + status; script detail renders each turn as `Text` speaker-labelled lines (model text NEVER hits a markup parser — assert a turn containing `[bold red]x[/]` paints literally); Cast on a non-complete briefing refuses with a toast naming the status; second Cast while in flight refuses naming the running one; a crashed `generating` script row is failed `interrupted` on the next load (reuse the phase-1 zombie test shape INCLUDING the flag-at-call-time recorder lesson — the Generate-path sweep and the load-path sweep must be pinned separately or documented as one seam); a failed script renders its error string; the briefing detail is unchanged by a failed cast.
- [ ] **Step 3-4: Implement + green.** Scripts table sits under the briefing detail, loads with `_load_briefings` (same worker, same to_thread batch). Turn rendering caps at 200 turns with an honest "…N more turns" line (no silent truncation — spec ethos).
- [ ] **Step 5: Mutations:** (a) drop the guard claim at dispatch → the double-press test REDs; (b) render turns via markup-parsing path → the literal-paint test REDs; (c) sweep removed from cast dispatch → the recorder test REDs. Restore all.
- [ ] **Step 6: Commit** `feat(briefings): cast scripts from the Artifacts pane`.

---

### Task 6: Citations into the reader + pruned degradation

**Files:**
- Modify: `tldw_chatbook/Subscriptions/briefing_service.py` (add pure `extract_citation_ids(body_markdown: str) -> list[int]` — ordered, deduped, regex `\[item (\d+)\]`)
- Modify: `tldw_chatbook/UI/Watchlists_Modules/artifacts_pane.py` (citations DataTable under the detail)
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py` (`handle_citation_activated`; resolution inside `_load_briefings` via Task 1's `get_subscription_items_by_ids`)
- Test: extend `Tests/Watchlists/test_watchlists_artifacts_pane.py` + `Tests/Subscriptions/test_briefing_service.py` (extractor)

**Interfaces — Consumes:** `items_pane.select_and_reveal:318` (NOT `handle_item_selected` directly — its docstring warns the reactive goes stale), `screen.active_section`, Task 1 `get_subscription_items_by_ids`.
**Interfaces — Produces:** pane reactive `citations: list[dict]` (`{"item_id": int, "label": str, "available": bool}`); message `CitationActivated(item_id: int)`.

- [ ] **Step 1: Read first:** the reader chain notes — `ItemsPane.selected_item` → `ItemSelected` → `_select_entity` → `ContentPane.item` → `_mark_item_read_on_open:3456`. Design rulings (do not relitigate): a citation click is an OPEN — the mark-read side effect is intended, same as clicking the row; a cited item hidden by the active items filter still opens in the reader (`select_and_reveal` sets the reactive; the cursor no-op is acceptable); citation labels are `Text` objects (item titles are remote text).
- [ ] **Step 2: Failing tests:** extractor: ordered, deduped, ignores `[item x]`/`[item]`; a complete briefing's citations table lists each cited id with its title; **`test_a_citation_to_a_pruned_item_degrades`** (the named invariant): body cites an id that has no row → the citations table renders "item N — no longer available" as its label, `available=False`, activating it toasts (markup=False) and does NOT switch sections; activating an available citation switches `active_section` to `"items"` and the reader shows that item (assert `ContentPane.item["id"]`); the read-status write happens (assert the DB status flipped — pinning the ruling so a future "why did my item get marked read" has a test to find).
- [ ] **Step 3-4: Implement + green.** Resolution batch: one `get_subscription_items_by_ids` call per briefing selection, inside the existing load worker's to_thread hop.
- [ ] **Step 5: Mutations:** (a) resolution treats missing rows as available → the invariant test REDs; (b) activation skips the section switch → the reader test REDs. Restore both.
- [ ] **Step 6: Commit** `feat(briefings): citations navigate to the reader, pruned items degrade honestly`.

---

### Task 7: Close-out

- [ ] Full sweep: `Tests/Subscriptions/ Tests/Watchlists/ Tests/UI/test_watchlists_inspector.py Tests/UI/ -k watchlist` — only documented baselines may fail (tree-chevron ×2; the TASK-1345 focus flake re-run in isolation before classifying).
- [ ] `backlog/tasks/task-1540 …`: check the two deferral sub-bullets `[x]` (mode picker, citations — both now real), add a phase-2a line under AC #2 noting scripts shipped text-only and 2b (audio) remains; do NOT check AC #2 itself.
- [ ] Spec: under "Phase 1 delivery notes", add a "Phase 2a delivery notes" block — scripts/presets/mode-picker/citations shipped; audio split to 2b with the five adapter-reality findings one line each (stream-draining synthesize, legacy `OpenAISpeechRequest` path, dead `text_processing` chunking, no real stitcher, no private binary append/move helper).
- [ ] Cross-worktree ID scan (controller supplies IDs) → file ONE task: "Briefings phase 2b: audio synthesis, stitching, playback" carrying those five findings as its opening context.
- [ ] Commit `docs(briefings): phase 2a close-out`.

## Self-review

**Spec coverage (2a scope):** script pass contract (JSON turns, unknown-speaker-by-name, malformed-names-parse-error, briefing-untouched, deleted-card-names-card, roster-of-one) → T2; snapshots → T1 schema + T2 snapshot write; preset CRUD + picker → T3/T4; per-preset LLM + app-default fallback → T2 plumbing + T4 cast-the-die; mode picker deferral → T4; citations deferral + named invariant → T6; statuses-as-observability, no new persist events → constraints. Audio: explicitly out (2b), spec note in T7.

**Placeholders:** none — every step names its precedent file:line or carries the contract inline; the phase-1 skeletal convention is declared up front.

**Type consistency:** `roster_json`/`roster_snapshot_json` str at the DB seam, `list[dict]` in the service (`dump_roster`/`load_roster` the only converters); `preset_id: int | None` everywhere; message names match between pane (produces) and screen steps (consumes); `ScriptCastError` is the single service error type.

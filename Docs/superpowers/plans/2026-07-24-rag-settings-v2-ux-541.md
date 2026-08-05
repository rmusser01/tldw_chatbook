# RAG Settings Screen v2 UX Upgrades (task-541) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the six structural UX upgrades from the owner-approved senior design review of the SP3 RAG settings screen (task-541): honest toggles, pre-commit re-index confirmation, context-sensitive inspector guidance, a real manage-vs-edit split with preview-on-select, a first-run starter panel, and keyboard accelerators.

**Architecture:** All UI work extends the existing SP3 machinery in `settings_screen.py` pattern-for-pattern; the headless `settings_rag_profile_adapter.py` gains only small pure helpers. The context-sensitive inspector mirrors the Providers category's existing field-guidance machinery (`DescendantFocus` → `_active_settings_field_id` → impact-pane rows). Preview-on-select is a strictly READ-ONLY display state — the category-keyed framework draft remains bound to the ACTIVE profile only.

**Tech Stack:** Textual (`Checkbox` — repo precedent in `MCP_Modules/*`, `ChatbookCreationWindow`; `Collapsible.Toggled`; `DescendantFocus` hook already at `settings_screen.py:9363`), `ConfirmationDialog` (`Widgets/confirmation_dialog.py:20`), SP3 adapter seams, QA capture rig `Docs/superpowers/qa/rag-settings-sp3-2026-07/`.

## Global Constraints

- **Source of truth:** `backlog/tasks/task-541` (six ACs). The v1 quick wins already shipped (PR #829) — do not rework them, extend them.
- **Files:** `tldw_chatbook/UI/Screens/settings_screen.py`, `tldw_chatbook/UI/Screens/settings_rag_profile_adapter.py`, `Tests/UI/test_settings_rag_profile_{adapter,region}.py` (+`test_settings_configuration_hub.py` assertion updates only), CSS source tcss + bundle REGENERATED via `python tldw_chatbook/css/build_css.py` (never hand-edit the bundle). No RAG_Search engine changes.
- **Draft invariant (hard):** the framework draft (`_settings_drafts[LIBRARY_RAG]`) belongs to the ACTIVE profile only. Preview-on-select NEVER stages values, never creates a second draft, and per-field Changed handlers MUST NOT mark dirty while previewing.
- **Off-thread invariant:** anything touching `fetch_index_status`/profile IO runs in a `@work(thread=True, group=...)` worker with `self.app.call_from_thread` marshalling (Screen has no `call_from_thread`).
- **Keyboard:** screen-level keys in use: `s`/`r`/`t` (+`escape` in modals, `/` filter). New accelerators `a` (Set active), `c` (Clone…), `b` (Backfill) fire ONLY when the RAG category is active (guard inside the action); single-letter keys are naturally swallowed while an Input has focus (standard Textual dispatch — verify once in a test).
- **Copy rules (from the review):** no "fingerprint" in user copy; "results" terminology; warnings honest and specific.
- **Every task runs its tests in the FOREGROUND; never `run_in_background`.** Never `git add` `.superpowers/sdd/progress.md`. Stage explicit paths.
- End state: refreshed QA captures for the USER SCREEN GATE; user-gated PR; no merge without explicit approval.

## Plan-time facts (verified)
- Providers field guidance: `_refresh_provider_field_guidance()` called from sync (`:6276`); `_provider_field_guidance_rows()` switches on `self._active_settings_field_id` (`:6287-6470`); `@on(DescendantFocus)` sets it (`:9363`); impact pane renders at `_render_impact_pane` (`:8854`), pane id `#settings-impact-pane` (`:9178`).
- Profile Select id `#settings-library-rag-profile-select` (compose `:7858`; sync `:7642/:7976`); NO `Select.Changed` handler exists for it today.
- `ConfirmationDialog` two-way modal exists (`Widgets/confirmation_dialog.py:20`); `RagProfileSwitchConfirmModal` (3-way) + `RagProfileNameModal` precedents in `settings_screen.py`.
- Adapter seams available: `list_profiles_grouped()` (`{"builtin","user","active_id"}`), `active_profile_info()` (`{"id","name","read_only","description"}`), `activate_profile`, `clone_profile_as`, `index_change_pending(values)`, `fetch_index_status()` (`{"state","count","provenance"}`), `load_rag_defaults_from_active_profile()`.
- `Checkbox` precedent: `tldw_chatbook/UI/MCP_Modules/mcp_schema_form.py` etc.
- Save choke point: `action_settings_save_category` LIBRARY_RAG branch (captures+clears `_rag_profile_pending_activate` at top, validates, re-arms pending + dispatches `_settings_save_library_rag_worker(values, index_will_change)`).
- First-run detectability: `active_profile_info()["read_only"]` + `list_profiles_grouped()["user"] == []` + `fetch_index_status()["state"] == "absent"` (status async — the starter panel keys off the already-fetched status the row uses).

---

## File Structure
- `settings_rag_profile_adapter.py` — add pure helpers only: `get_profile_defaults(profile_id)` (read-only load of ANY profile for preview), `is_first_run_state(index_state)` (pure predicate over already-fetched inputs).
- `settings_screen.py` — all six UI items, extending existing renderers/handlers/sync.
- Tests: adapter tests for the helpers; region tests for every behavior (bare-instance + pilot patterns already established in `test_settings_rag_profile_region.py`).

---

### Task 1: Honest toggles + conditional rerank fields (AC #4)

**Files:** `settings_screen.py` (citations + reranking toggle rendering/handlers, rerank-field dimming), `Tests/UI/test_settings_rag_profile_region.py`.
**Interfaces produced:** citations + reranking render as `Checkbox(label=..., value=..., id=<existing ids>)`; rerank model/results Inputs get `disabled=True` + a `(enable reranking to edit)` tooltip/suffix when reranking is off, live-updated when the checkbox flips.

- [ ] **Step 1 (RED):** region tests: (a) with reranking disabled in loaded values, the two rerank Inputs compose disabled and re-enable after the checkbox is toggled on (pilot: toggle → sync → enabled); (b) citations/reranking are `Checkbox` widgets whose `value` mirrors loaded state; (c) toggling the checkbox stages the draft exactly like the old button did (reuse the existing dirty-marking assertions, retargeted).
- [ ] **Step 2:** replace the two state-labeled toggle Buttons with `Checkbox` (keep the SAME widget ids so `_library_rag_field_selector`/sync keep working — verify Checkbox honors `disabled=` for the read-only builtin lock); rewire their `@on(Button.Pressed)` handlers to `@on(Checkbox.Changed)` with identical value-aware dirty-marking. DIM-not-hide: rerank model/results get `disabled = not rerank_enabled or field_disabled` at compose AND in `_sync_library_rag_widgets`; append the explanatory suffix to their labels when dimmed for that reason (distinct from builtin read-only).
- [ ] **Step 3 (GREEN + regression):** `pytest Tests/UI/test_settings_rag_profile_region.py Tests/UI/test_settings_rag_profile_adapter.py -q`; `pytest Tests/UI/ -q -k "settings" 2>&1 | tail -3` (baseline ~7 pre-existing).
- [ ] **Step 4:** Commit `feat(settings): checkbox toggles + conditional rerank fields (541 AC4)`.

### Task 2: Pre-commit re-index confirmation (AC #2)

**Files:** `settings_screen.py` (save flow), `Tests/UI/test_settings_rag_profile_region.py`.
**Interfaces:** consumes `index_change_pending(values)`, `fetch_index_status()`; produces `_confirm_reindex_then_save(values)` flow.

- [ ] **Step 1 (RED):** region tests: (a) save with an index-changing draft while the (stubbed) status is `built` with `count=1234` → a confirm modal is pushed whose message contains `"1234"` and `"Backfill"`; Cancel → save worker NOT dispatched, draft intact, pending-activate NOT re-armed; Confirm → save worker dispatched with `index_will_change=True`; (b) save with index change but status `absent`/`empty` → NO modal (nothing to lose), worker dispatched directly (existing post-save warning suffices); (c) the save-then-switch flow (dirty prompt → Save) still completes: pending-activate survives a Confirm and is cleared on Cancel.
- [ ] **Step 2:** in the LIBRARY_RAG save branch, when `index_will_change` is True: dispatch a small `@work(thread=True, group="settings-rag-index-status")`-style fetch (reuse the existing status worker/cached last status if fresh — prefer the CACHED last-fetched status the row already holds to avoid a save-click latency; only fetch if none cached) and, when `state == "built"`, `push_screen` a `ConfirmationDialog` with: `This change re-points to a new EMPTY index — the current index ({count} vectors) stops being used and search returns nothing until you run Backfill. Save anyway?`; Confirm → proceed exactly as today (re-arm pending, dispatch worker); Cancel → clear pending-activate, keep draft. When state is not `built` (absent/empty/unknown/no cache) → proceed without the modal.
- [ ] **Step 3 (GREEN + regression)** as Task 1 Step 3.
- [ ] **Step 4:** Commit `feat(settings): pre-commit re-index confirmation with vector count (541 AC2)`.

### Task 3: Context-sensitive Scope Inspector (AC #3)

**Files:** `settings_screen.py`, `Tests/UI/test_settings_rag_profile_region.py`.
**Interfaces:** `_rag_field_guidance_rows() -> tuple[tuple[str, str], ...]` mirroring `_provider_field_guidance_rows`.

- [ ] **Step 1 (RED):** tests: focusing (simulate by setting `self._active_settings_field_id` + calling the refresh) a Reranking field yields guidance rows mentioning reranking; a Chunking field yields re-index/backfill guidance; no RAG field focused → the current static category guidance (fallback unchanged).
- [ ] **Step 2:** implement `_rag_field_guidance_rows()` keyed on `_active_settings_field_id` with one concise entry per GROUP (search / embedding / chunking / vector store / reranking / profile controls / index row) — map field-id prefixes to group guidance (e.g. any `settings-library-rag-chunk*` id → the chunking entry: what the fields mean + "changing these rebuilds the index — Backfill after"); wire it where the domain-category guidance is currently sourced for LIBRARY_RAG (the impact-pane render path), refreshed from the existing `@on(DescendantFocus)` hook (extend the hook's refresh call to cover the RAG category the way it covers Providers) AND from `Collapsible.Toggled` (`@on(Collapsible.Toggled)` → refresh, so expanding a group without focusing a field already switches context). Keep every string within the rail width (the SP3 fit lesson — no mid-sentence clipping at the QA viewport).
- [ ] **Step 3 (GREEN + regression)** as Task 1 Step 3, plus `pytest Tests/UI/test_settings_configuration_hub.py -q -k "inspector or guidance" 2>&1 | tail -3` (the guidance-coverage test at `settings_screen.py:797` comment — ensure it still passes).
- [ ] **Step 4:** Commit `feat(settings): context-sensitive RAG inspector guidance (541 AC3)`.

### Task 4: Manage-vs-edit split + read-only preview-on-select (AC #1)

**Files:** `settings_rag_profile_adapter.py` (+`get_profile_defaults`), `settings_screen.py`, both test files.
**Interfaces produced:** `get_profile_defaults(profile_id: str) -> SettingsLibraryRagDefaults | None` (pure read of ANY profile — reuse `load_rag_defaults_from_active_profile`'s mapping against an explicit profile); screen state `self._rag_preview_profile_id: str | None`.

- [ ] **Step 1 (RED, adapter):** `get_profile_defaults("<user id>")` round-trips distinctive values; unknown id → None. (Hermetic fixtures per the file's idiom.)
- [ ] **Step 2 (RED, region):** (a) `Select.Changed` to a non-active profile enters preview: editor fields display THAT profile's values, all disabled, and a preview banner names it (`Previewing 'X' (read-only) — press Set active to edit it`); NO draft is created (`_settings_drafts` unchanged) and field Changed events during preview do NOT stage; (b) selecting back the active profile exits preview and restores the active values WITH any pre-existing dirty draft re-applied (stage a draft first, browse away, browse back → staged value still shown + still dirty); (c) Set active from preview runs the existing flow (dirty prompt honored) and on success exits preview; (d) structural: the Profiles block renders inside its own titled container (`Profiles`) and the editor inside a container titled `Editing: <active name>` (preview flips that title to `Previewing: <selected name>`).
- [ ] **Step 3:** implement: new `@on(Select.Changed, "#settings-library-rag-profile-select")` handler setting `_rag_preview_profile_id` (None when selection == active) and calling a new `_sync_rag_editor_display()` that: in preview → writes the previewed profile's values into the widgets, disables all editor fields, sets the banner/title; in normal → delegates to the existing `_sync_library_rag_widgets` (draft-aware) + read-only lock. Guard EVERY `handle_library_rag_*_changed` with an early `if self._rag_preview_profile_id: return` (one shared helper). Wrap the two regions in titled containers (follow the repo's container/CSS conventions; new classes in source tcss + rebuild bundle). `_rag_after_set_active` and `_rag_after_profile_action` clear `_rag_preview_profile_id`.
- [ ] **Step 4 (GREEN + regression)** as Task 1 Step 3 + full `pytest Tests/UI/test_settings_configuration_hub.py -q 2>&1 | tail -3`.
- [ ] **Step 5:** Commit `feat(settings): manage-vs-edit split + read-only profile preview (541 AC1)`.

### Task 5: First-run starter panel (AC #5)

**Files:** `settings_rag_profile_adapter.py` (+`is_first_run_state`), `settings_screen.py`, both test files.
**Interfaces:** `is_first_run_state(info: dict, grouped: dict, index_state: str) -> bool` — pure: `info["read_only"] and not grouped["user"] and index_state == "absent"`.

- [ ] **Step 1 (RED, adapter):** truth-table tests for the predicate (all four combinations that must be False + the one True).
- [ ] **Step 2 (RED, region):** when the predicate holds (stubbed status absent + builtin active + no user profiles): a starter panel renders above the editor — copy: `Search already works on {active name}. Clone it to tune retrieval, or run Backfill to enable semantic results.` with two Buttons `Clone to tune…` (opens the existing clone name-modal seeded from the active builtin) and `Backfill now` (dispatches the existing backfill worker) — and the editor's Collapsible groups compose COLLAPSED (Search included) so the panel, not the disabled wall, is the first impression. When the predicate is False → no panel, groups per today's defaults. Panel disappears on the next sync after a clone or a backfill completes (state-driven, no dismissal persistence).
- [ ] **Step 3:** implement — the panel keys off the SAME cached index status the row uses (no extra fetch; while status is unknown/unfetched, no panel). Wire the two buttons to the existing handlers. CSS via source tcss + rebuild.
- [ ] **Step 4 (GREEN + regression)** as Task 1 Step 3.
- [ ] **Step 5:** Commit `feat(settings): RAG first-run starter panel (541 AC5)`.

### Task 6: Keyboard accelerators + captures + close-out (AC #6)

**Files:** `settings_screen.py` (BINDINGS + actions + RAG footer hint), test files; QA captures.

- [ ] **Step 1 (RED):** tests: with the RAG category active, actions `settings_rag_set_active` / `settings_rag_clone` / `settings_rag_backfill` route to the existing handlers (spies); with ANOTHER category active, the same actions no-op (guard verified); the RAG category's footer/hint line advertises `a set active | c clone | b backfill` alongside `s/r/t`.
- [ ] **Step 2:** add `("a", ...)`, `("c", ...)`, `("b", ...)` to the screen BINDINGS (show=False like s/r/t if that's the current style — match `:1146` exactly), each action guarded on the active category being LIBRARY_RAG; extend the RAG category hint text. One pilot test typing `a` while an editor Input has focus → NOT intercepted (Input swallows printable keys — assert the set-active spy was NOT called).
- [ ] **Step 3 (GREEN + full gates):** rag test files + `pytest Tests/UI/ -q -k "settings" 2>&1 | tail -3` + `pytest Tests/RAG/ -q 2>&1 | tail -2` + `python -c "import tldw_chatbook.UI.Screens.settings_screen"`.
- [ ] **Step 4: Refresh QA captures** with the rig (`DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib python Docs/superpowers/qa/rag-settings-sp3-2026-07/capture_rag_settings.py` → `svg_to_png.py`) — extend the script for the NEW states: (1) first-run starter panel; (2) preview-on-select (non-active profile selected, banner + disabled fields + `Previewing:` title); (3) pre-commit confirm modal (built index + index-changing draft); (4) checkbox toggles + dimmed rerank fields; (5) context-sensitive inspector showing reranking guidance while a rerank field is focused. Verify each capture visually before committing.
- [ ] **Step 5:** update task-541 (ACs → `[x]`, status → Done, Implementation Notes); commit `feat(settings): RAG keyboard accelerators + v2 UX captures (541 AC6)`.

---

## Self-Review

**1. Spec coverage:** AC1→T4, AC2→T2, AC3→T3, AC4→T1, AC5→T5, AC6→T6. Capture/screen-gate + close-out in T6. ✓
**2. Placeholder scan:** every step names exact behavior + copy + wiring; the two "match the repo convention" notes (container CSS classes, BINDINGS style) point at exact anchors (`:1146`, source tcss). ✓
**3. Type consistency:** `get_profile_defaults`, `is_first_run_state`, `_rag_preview_profile_id`, `_sync_rag_editor_display`, `_rag_field_guidance_rows`, action names `settings_rag_set_active/clone/backfill` — defined once, consumed as written. Draft/off-thread invariants restated in Global Constraints. ✓
**Ordering note:** T1-T3 are independent; T4 is the structural core (do after T1 so the toggles land inside the new editor container without rework — acceptable either order, T1 first chosen for smallest-first momentum); T5 sits on T4's layout; T6 last.

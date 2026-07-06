# Library Browse ▸ Media Canvas (L2a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Library rail's `Media` row from a screen-route into an in-Library **Browse ▸ Media canvas** — a list→preview split with a media-type filter in the canvas header — per spec §2 (L2), mirroring the shipped `Browse ▸ Conversations` canvas.

**Architecture:** Pure state first (`Library/library_media_state.py`), a `LibraryMediaCanvas` widget (`Widgets/Library/`), and screen wiring that adds a `media` canvas kind alongside `conversations`. Media records come from the existing local-source snapshot (the same `media_reading_scope_service.list_media_items` call the shell already makes for counts). The type filter is client-side over the fetched records (the service is paginate-only, no type filter — same fetched-set-scoped honesty as L1 conversation search). `[Open in Media]` hands off to the existing full Media viewer via `NavigateToScreen("media")`.

**Tech Stack:** Python 3.11+, Textual, pytest + pytest-asyncio, existing `LibraryHarness` UI harness.

**Spec:** `Docs/superpowers/specs/2026-07-04-home-library-redesign-design.md` §2 (Browse ▸ Media list→preview, media sub-types as a canvas-header filter). Reference implementation for every convention: the L1 conversations canvas on `dev` — `tldw_chatbook/Widgets/Library/library_conversations_canvas.py`, `tldw_chatbook/Library/library_conversations_state.py`, and the `conversations` branches of `tldw_chatbook/UI/Screens/library_screen.py`.

## Global Constraints

- Run tests with: `env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q <target> --tb=short`.
- The `timeout` shell command is not available.
- `tldw_chatbook/css/tldw_cli_modular.tcss` is GENERATED: edit `tldw_chatbook/css/components/_agentic_terminal.tcss`, run `./build_css.sh`, commit both.
- Row/canvas conventions (binding, from the L1 conversations canvas): list rows are two-line compact Buttons (`▸`/space marker, 20-char title truncation via the shared `_visible_row_title` helper, full title as tooltip, `id="library-media-row-{index}"`, class `library-media-row` + `-selected`); preview is a `Vertical` with a `Static` of preview lines + a `Horizontal(classes="ds-toolbar")` action row; loading/error states gate exactly as the conversations canvas does (`#library-canvas-loading` / `#library-canvas-error` when `not _library_loaded` / lookup error).
- Row secondary grammar exactly: `"{type} · {age}"` (or `"{type}"` when no age, or `"media"` when neither). Canvas header title exactly: `Media ({count})`. Type-filter default option label exactly: `All`.
- Canvas landing/empty copy for no media exactly: `No media in your Library yet. Ingest something to see it here.`
- Behavior contracts that MUST survive: `[Open in Media]` reaches the full Media viewer (`NavigateToScreen("media")`); every other Browse/Create/Ingest row and the conversations/collections/search canvases are unchanged; `NavigateToScreen` routing for the other screen-rows unchanged.
- Deferred (out of L2a scope; note honestly, do not fake): per-item `[Use in Chat]`, `[Export]`, and `[Run RAG on this]` are NOT added as canvas buttons — no clean app-level media→chat/RAG/export handoff exists yet; those actions remain reachable through the full Media viewer via `[Open in Media]`. The action row in L2a is `[Open in Media]` only.
- **Stage only files you changed (`git add <specific paths>`). NEVER `git add -A`, `git add .`, or `git commit -a`.** Never touch `.claude/settings.local.json`.
- Commit messages end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Library screen changes require live screenshot QA at viewport 2050x1240 (device_scale_factor 1) + explicit user approval before merge (Task 7). Seed media into the isolated HOME's media DB — reuse the L1 QA seeding recipe (ChaChaNotes lives under `$HOME/.local/share/tldw_cli/default_user/…`; the media DB path + client id follow the same pattern — verify at QA time).
- Work in the `claude/library-l2` branch (worktree `.claude/worktrees/library-l2`, based on `origin/dev`).

## File Structure

- Create `tldw_chatbook/Library/library_media_state.py` — pure media rows/type-filter/preview builder.
- Create `tldw_chatbook/Widgets/Library/library_media_canvas.py` — media list→preview widget with header type filter.
- Modify `tldw_chatbook/Library/library_shell_state.py` — `browse-media` row becomes `target_kind="canvas"`, `target_id="media"`; add `canvas_kind="media"` resolution.
- Modify `tldw_chatbook/UI/Screens/library_screen.py` — retain media records from the snapshot; `media` canvas branch in `compose_content`; type-filter + row-selection + `[Open in Media]` dispatch; loading/error gating.
- Modify `tldw_chatbook/css/components/_agentic_terminal.tcss` (+ regenerate).
- Tests: create `Tests/Library/test_library_media_state.py`; extend `Tests/UI/test_library_shell.py`; re-anchor the `browse-media`→screen-route assertions in `Tests/UI/test_library_shell.py` and `Tests/UI/test_destination_shells.py`.

---

### Task 1: Pure media canvas state

**Files:**
- Create: `tldw_chatbook/Library/library_media_state.py`
- Test: `Tests/Library/test_library_media_state.py`

**Interfaces:**
- Consumes: `format_console_relative_age` from `tldw_chatbook.Workspaces.conversation_browser_state`.
- Produces:

```python
@dataclass(frozen=True)
class LibraryMediaRow:
    media_id: str
    title: str
    media_type: str
    secondary: str          # "{type} · {age}" / "{type}" / "media"
    selected: bool = False

@dataclass(frozen=True)
class LibraryMediaCanvasState:
    rows: tuple[LibraryMediaRow, ...]
    type_options: tuple[str, ...]     # ("All", <distinct types, sorted, title-preserved>)
    active_type: str                  # "All" or a specific type
    status_copy: str                  # "" or "N of M · type: X"
    empty_copy: str                   # set when rows empty
    selected_id: str
    preview_lines: tuple[str, ...]
    count: int                        # total media in the fetched set (pre-type-filter)

def build_library_media_state(
    records: Sequence[Mapping[str, Any]],
    *,
    active_type: str = "All",
    selected_id: str = "",
    now: datetime | None = None,
    limit: int = 75,
) -> LibraryMediaCanvasState: ...
```

Rules (mirror `build_library_conversations_state`): media id from first present of `("id", "media_id", "uuid")` (str-cast, whitespace-stripped, skip records with no usable id); title from `record.get("title")` stripped, fallback `"Untitled media"`; type from first present of `("type", "media_type")` stripped, fallback `""`; updated timestamp from first present of `("last_modified", "ingestion_date", "date", "updated_at")`. `type_options` = `("All",) + tuple(sorted(distinct non-empty types))`. When `active_type != "All"`, filter rows to that type BEFORE limiting; `status_copy = f"{len(filtered)} of {count} · type: {active_type}"`. Sort by updated desc (missing last), truncate to `limit`. `selected_id` not in the filtered+limited rows → select first row; no rows → `selected_id=""` and `empty_copy` = the constant (type "All") or `f"No media of type '{active_type}'."`. `preview_lines` for the selected record: `(title, f"Type: {type or 'unknown'}", f"Updated: {age or 'unknown'}")`. Pure; no Textual imports; tolerate malformed records without raising.

- [ ] **Step 1: Write failing tests** — cover: rows + `type · age` secondary with fixed `now`; type_options enumeration + sort; type filter + status copy + no-match empty copy; selected-id fallback to first; preview lines; limit truncation; id/title/type key fallbacks; malformed-record tolerance (None / non-Mapping / missing-id skipped, one valid survives). Model the file on `Tests/Library/test_library_conversations_state.py`.
- [ ] **Step 2: Verify RED** (ImportError).
- [ ] **Step 3: Implement** per rules.
- [ ] **Step 4: Verify GREEN.**
- [ ] **Step 5: Commit** `feat(library): pure browse-media canvas state`.

---

### Task 2: Shell state — media becomes a canvas

**Files:**
- Modify: `tldw_chatbook/Library/library_shell_state.py`
- Test: `Tests/Library/test_library_shell_state.py`

**Interfaces:**
- Change the `browse-media` row: `target_kind="canvas"`, `target_id="media"` (was `"screen"`, `TAB_MEDIA`).
- `build_library_shell_state` selection resolution: a selected `browse-media` row → `canvas_kind="media"`, `canvas_target=""` (mirror the `conversations` case). Add module constant `LIBRARY_ROW_BROWSE_MEDIA = "browse-media"` and reuse it (consistent with the existing `LIBRARY_ROW_BROWSE_CONVERSATIONS`).

- [ ] **Step 1: Failing tests** — update `test_library_shell_state.py`: the browse-media row now `("canvas", "media")`; selecting `browse-media` yields `canvas_kind == "media"`. Keep the title assertion (`"Media"`).
- [ ] **Step 2: RED.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: GREEN** (`Tests/Library/`).
- [ ] **Step 5: Commit** `feat(library): media row selects an in-Library canvas`.

---

### Task 3: Retain media records + type-filter state in the screen

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Test: extend `Tests/UI/test_library_shell.py` (asserted via Task 5's canvas tests; this task is the data plumbing)

**Interfaces:**
- The snapshot already fetches media (`_list_local_source_snapshot` → `list_media` → response `{"items": [...], "pagination": {"total_items": N}}`). Retain the media **items** in `self._local_source_records["media"]` the same way conversations records are retained (find where conversations records are stored from the snapshot and store media items alongside). Add screen state: `self._library_media_type_filter: str = "All"`, `self._selected_media_id: str = ""`.
- Add `_build_library_media_state(self) -> LibraryMediaCanvasState` mirroring `_build_library_conversations_state`: reads `self._local_source_records.get("media", ())`, passes `active_type=self._library_media_type_filter`, `selected_id=self._selected_media_id`.

- [ ] **Step 1:** Read how the snapshot stores conversation records (`_apply_local_source_snapshot` / `_list_local_source_snapshot`) and store media items identically. Add the two state fields + the builder.
- [ ] **Step 2:** No standalone test here — Task 5 exercises it. Run `Tests/Library/` + existing `Tests/UI/test_library_shell.py` to confirm no regression from the plumbing.
- [ ] **Step 3: Commit** `feat(library): retain media records and type-filter state`.

---

### Task 4: LibraryMediaCanvas widget

**Files:**
- Create: `tldw_chatbook/Widgets/Library/library_media_canvas.py`

**Interfaces:** `LibraryMediaCanvas(Vertical)` — `__init__(self, canvas: LibraryMediaCanvasState, **kwargs)`, width `13fr`/min 40. Compose (mirror `LibraryConversationsCanvas`):
- Header `Horizontal(id="library-media-header")`: `Static(f"Media ({canvas.count})", id="library-media-title")` + a `Select` (`id="library-media-type-filter"`, options from `canvas.type_options`, value `canvas.active_type`, `allow_blank=False`).
- Status `Static(canvas.status_copy or canvas.empty_copy, id="library-media-status")`.
- List `Vertical(id="library-media-list")` of row Buttons: `id=f"library-media-row-{index}"`, classes `library-media-row` (+`-selected`), compact, label `f"{marker} {_visible_row_title(row.title)}\n    {row.secondary}"`, tooltip full title, attr `button.media_id`. (Copy `_visible_row_title` from `library_conversations_canvas` import or the shared helper it uses.)
- Preview `Vertical(id="library-media-preview")` (`display = bool(selected_id and preview_lines)`): `Static("\n".join(preview_lines), id="library-media-preview-lines")` + `Horizontal(classes="ds-toolbar")` with `Button("Open in Media", id="library-media-open", classes="library-canvas-action", compact=True)`.

No separate steps/commit — implemented and driven RED→GREEN by Task 5's pilot tests, landing in Task 5's commit (L1 precedent).

---

### Task 5: Screen wiring — media canvas branch, filter, selection, open

**Files:**
- Modify: `tldw_chatbook/UI/Screens/library_screen.py`
- Test: extend `Tests/UI/test_library_shell.py`

**Interfaces:** In `compose_content`, add a `media` branch mirroring `conversations`: when `shell.canvas_kind == "media"` → loading/error gating (reuse the same `not _library_loaded` / `_library_lookup_error` checks and `#library-canvas-loading` / `#library-canvas-error` ids), else build `_build_library_media_state()` (writing back `_selected_media_id = state.selected_id`) and yield `LibraryMediaCanvas(state, id="library-media-canvas")`. Handlers: `Select.Changed` on `#library-media-type-filter` → set `_library_media_type_filter`, `refresh(recompose=True)`; `Button.Pressed` on `.library-media-row` → set `_selected_media_id` from `button.media_id`, force `_library_selected_row_id="browse-media"`/`_active_mode` as conversations does, recompose; `Button.Pressed` on `#library-media-open` → `NavigateToScreen("media")`.

- [ ] **Step 1: Failing pilot tests** in `Tests/UI/test_library_shell.py` (seed media via `_seed_conversations(app, [...], media=[...])` — the harness already injects `StaticLibraryMediaScopeService`; media items are dicts with `id`/`title`/`type`/`last_modified`): (1) pressing `#library-row-browse-media` renders `#library-media-canvas` with `Media (N)` title, a `#library-media-type-filter` Select, media rows with `type · age` secondary, first row selected, preview + `#library-media-open`; (2) changing the type filter to a specific type narrows the list + status copy; (3) pressing a media row updates selection + preview; (4) pressing `#library-media-open` posts `NavigateToScreen("media")`; (5) before load (`_library_loaded` False) the media canvas shows `#library-canvas-loading`.
- [ ] **Step 2: Verify RED.**
- [ ] **Step 3: Implement** widget (Task 4) + screen wiring.
- [ ] **Step 4: Run** the new tests + all of `Tests/Library/` + existing `Tests/UI/test_library_shell.py`. The old browse-media test (pressing it posts `NavigateToScreen("media")`) is now wrong — re-anchor it to assert the media canvas renders (the media SCREEN route now lives on `#library-media-open`). Full green.
- [ ] **Step 5: Commit** `feat(library): browse-media list/preview canvas with type filter`.

---

### Task 6: CSS

**Files:** Modify `tldw_chatbook/css/components/_agentic_terminal.tcss`, regenerate.
- [ ] **Step 1: Failing presence test** (mirror the L1 stylesheet test): selectors `#library-media-header`, `#library-media-title`, `.library-media-row`, `.library-media-row-selected`, `#library-media-type-filter` present in component + generated CSS.
- [ ] **Step 2: RED**, then add rules — copy the `.library-conversation-row`/`-selected` blocks under the media selectors; style `#library-media-header` as a row (`height: auto`, the title `1fr`, the Select right-aligned/compact); `./build_css.sh`; GREEN.
- [ ] **Step 3: Commit** `style(library): browse-media canvas styles`.

---

### Task 7: Verification, screenshot QA, approval gate

- [ ] **Step 1:** Full affected run: `Tests/Library/`, `Tests/UI/test_library_shell.py`, `Tests/UI/test_destination_shells.py`, `Tests/UI/test_post_release_workspaces_library_depth.py`. All green (re-anchor any destination_shells media-route assertion the same way).
- [ ] **Step 2:** Live captures to `Docs/superpowers/qa/library-media-l2a-2026-07/` — textual-serve + playwright, 2050x1240, fresh isolated HOME seeded with real media rows of ≥2 types. Captures: (1) media canvas populated (list + `Media (N)` + type filter + preview + `[Open in Media]`); (2) type filter applied (narrowed list + status copy); (3) selection switch. Include populated states (empty states hide row defects — L1 lesson).
- [ ] **Step 3:** Write the QA README; present to the user for explicit approval before merge; commit QA artifacts.

## Self-review notes

- Spec coverage: media as a Browse canvas with list→preview (Tasks 1–5), media sub-types as a canvas-header filter not rail rows (Tasks 1, 4–5), Console browser row grammar (Task 4), landing/empty copy (Task 1), test + approval gate (Tasks 5, 7).
- Honest deviations from the spec mock: the `[Use in Chat]`/`[Export]`/`[Run RAG on this]` per-item actions are deferred (no clean handoff exists); `[Open in Media]` carries the deep actions via the full viewer. Type filter and count are scoped to the fetched media snapshot (service is paginate-only) — same fetched-set honesty as L1 conversation search; raising the fetch cap is a shared follow-up with L1's conversations cap.
- Type consistency: `canvas_kind == "media"`, `library-media-*` ids, and `_library_media_type_filter`/`_selected_media_id` names match across Tasks 2, 4, 5.

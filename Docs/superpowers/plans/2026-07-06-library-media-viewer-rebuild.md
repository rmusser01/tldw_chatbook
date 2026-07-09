# Library Browse ▸ Media — full in-canvas viewer (L2a rebuild) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Library media canvas's 3-line preview stub with a real in-canvas media **viewer** that has capability parity with `Widgets/Media/MediaViewerPanel` — full metadata (view + edit), the actual content/transcript (scrollable, searchable), reading highlights, read-it-later, and analysis view/edit — so Browse ▸ Media genuinely replaces the Media screen for viewing and managing media, rather than routing to it.

**Architecture:** The media canvas gains a canvas-level mode: **list** (the existing list + type filter) and **viewer** (the selected item's full detail). Selecting a media row switches to viewer mode and fetches the item detail async (`media_reading_scope_service.get_media_item`, the same offloaded-service pattern the local-source snapshot and collections load already use). Pure display-state modules shape the fetched detail; a `LibraryMediaViewer` widget renders it; `library_screen.py` orchestrates the fetch, mode, edit/delete/highlight/analysis handlers. All mutations go through existing async scope-service methods (`update_media_item`, `delete_media_item`, `*_reading_highlight`, `save_to/remove_from_read_it_later`, `save_analysis_version`).

**Tech Stack:** Python 3.11+, Textual, pytest + pytest-asyncio, existing `LibraryHarness` UI harness.

**Spec:** `Docs/superpowers/specs/2026-07-04-home-library-redesign-design.md` §2 (Browse ▸ Media, "preview is type-appropriate — `Widgets/Media/` viewer panel for media"; §6.4 REBUILD to the new design, capability parity). This plan supersedes the "light preview routes to Media screen" scoping of `2026-07-06-library-media-browse-l2a.md` (the list + type filter from that plan stay; only the preview stub is replaced). Reference for capability parity: `tldw_chatbook/Widgets/Media/media_viewer_panel.py` (Metadata/Content/Analysis tabs, actions). Reference for canvas/state/widget conventions: the shipped conversations + collections canvases in `Widgets/Library/` and `Library/`.

## Global Constraints

- Run tests with: `env HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q <target> --tb=short`.
- The `timeout` shell command is not available.
- `tldw_chatbook/css/tldw_cli_modular.tcss` is GENERATED: edit `tldw_chatbook/css/components/_agentic_terminal.tcss`, run `./build_css.sh`, commit both.
- **RENDERING RULE (binding, learned this session):** the Textual `Select` and any header child placed after a `1fr` sibling do NOT render in the deployed textual-serve TUI. Before building UI on any complex widget (`Markdown`, `TextArea`, `Collapsible`, `TabbedContent`, scroll containers), verify it renders in a live textual-serve capture (Task 1). Prefer stacked full-width elements over side-by-side horizontal layouts; prefer flat scrollable sections over `TabbedContent` unless a capture proves tabs render. Buttons, plain `Static`, and full-width rows render reliably.
- All scope-service methods are async and keyed by `media_id` keyword; call them through the existing `self._run_library_service_call(callable, ...)` offload wrapper (see how `_refresh_library_collections_snapshot` calls services). Service object: `getattr(self.app_instance, "media_reading_scope_service", None)`; guard for absence (degrade to a quiet "media viewer unavailable" line, Console convention).
- Optimistic-locking: `update_media_item` may require a version; read the current detail's version field and pass it, and handle a conflict by re-fetching and surfacing a quiet "changed elsewhere — reloaded" line (mirror the notes editor's version handling if present).
- Every mutation re-fetches the affected detail (or list) and recomposes; never mutate display state in place from a stale record.
- Behavior contracts that MUST survive: the list + type filter (title/type rows, cycling `type: <T> ▸` button, `Media (N)` header) are unchanged; `browse-conversations`/`collections`/`search`/other rows unchanged; the rail count sourcing unchanged.
- **Stage only files you changed (`git add <specific paths>`). NEVER `git add -A`, `git add .`, or `git commit -a`.** Never touch `.claude/settings.local.json`.
- Commit messages end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Library screen changes require live screenshot QA at 2050x1240 + explicit user approval before merge (Task 9). Seed real media into the isolated HOME's media DB (`default_user/tldw_chatbook_media_v2.db`, client_id `tldw_cli_local_instance_v1`, via `add_media_with_keywords` with real `content`) — reuse the session's `seed_media_qa.py`.
- Work in the `claude/library-l2` branch (worktree `.claude/worktrees/library-l2`).

## File Structure

- Create `tldw_chatbook/Library/library_media_viewer_state.py` — pure viewer display-state (metadata lines, content, keywords, analysis) + reading-highlights display-state.
- Create `tldw_chatbook/Widgets/Library/library_media_viewer.py` — the viewer widget (metadata section + content section + analysis section + actions + edit form + highlights), all stacked/scrollable.
- Modify `tldw_chatbook/UI/Screens/library_screen.py` — canvas list/viewer mode, detail fetch, edit/delete/highlight/read-later/analysis/use-in-chat handlers.
- Modify `tldw_chatbook/Widgets/Library/library_media_canvas.py` — render the viewer when in viewer mode (or the screen composes the viewer directly; decide in Task 2 — keep the list widget for list mode).
- Modify `tldw_chatbook/css/components/_agentic_terminal.tcss` (+ regenerate).
- Tests: create `Tests/Library/test_library_media_viewer_state.py`; extend `Tests/UI/test_library_shell.py`.

---

### Task 1: Detail fetch + viewer state + RENDERING SMOKE-CHECK

**Files:**
- Create: `tldw_chatbook/Library/library_media_viewer_state.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` (fetch + state fields only; no viewer UI yet)
- Test: `Tests/Library/test_library_media_viewer_state.py`
- Scratch: a throwaway textual-serve capture to prove the widgets render.

**Interfaces (produced):**
```python
@dataclass(frozen=True)
class LibraryMediaViewerState:
    media_id: str
    title: str
    metadata_lines: tuple[str, ...]   # ("Type: video", "Author: A", "URL: …", "Keywords: a, b", "Ingested: 2h")
    content: str                      # full content/transcript ("" if none)
    analysis: str                     # analysis_content ("" if none)
    has_content: bool
    has_analysis: bool
    version: int | None               # for optimistic update
    edit_fields: dict[str, str]       # {"title","author","url","keywords"} current values for the edit form

def build_library_media_viewer_state(detail: Mapping[str, Any] | None) -> LibraryMediaViewerState: ...
```
Field sourcing (from `get_media_item` detail = the Media row + `keywords` list + `content`): title from `title`; type from `type`/`media_type`; author `author`; url `url`; keywords `keywords` (list → `", ".join`); ingested age via `format_console_relative_age` over `ingestion_date`/`last_modified`; content `content`; analysis `analysis_content`; version `version`. Tolerate `None`/missing (blank lines, `has_*` False). Pure; no Textual imports.

**Screen plumbing:** add `self._library_media_view: str = "list"`, `self._library_media_detail: Mapping | None = None`, `self._library_media_detail_loading: bool = False`. Add `_refresh_library_media_detail(self, media_id)` — an async worker mirroring `_refresh_library_collections_snapshot`: calls `get_media_item(mode="local", media_id=..., include_content=True)` via `_run_library_service_call`, stores the result in `_library_media_detail`, sets loading False, recomposes. (Wired to selection in Task 2.)

- [ ] **Step 1: RENDERING SMOKE-CHECK FIRST.** Before writing state/tests, add a temporary media-viewer canvas branch that renders a `Static(long metadata)`, a scrollable `VerticalScroll` containing a `Static`/`Markdown` of a long content string, a `Collapsible`, a `TextArea`, and a `Button`. Serve via textual-serve (session `serve_media_qa.py` pattern), navigate to a selected media item, capture at 2050x1240, and CONFIRM each widget renders visibly. Record which of `Markdown` / `TextArea` / `Collapsible` / `TabbedContent` / `VerticalScroll` render. If any does not, note the fallback (plain `Static` + manual scroll) and use it in Task 2. Delete the temporary branch after. Put findings in the task report.
- [ ] **Step 2: Failing tests** for `build_library_media_viewer_state` — metadata line composition + order, keywords list-join, ingested age with fixed `now`, content/analysis presence flags, version passthrough, edit_fields, `None` detail → empty state. RED.
- [ ] **Step 3: Implement** the pure module + the screen fetch/worker + state fields.
- [ ] **Step 4: GREEN** (`Tests/Library/`); screen imports/constructs cleanly.
- [ ] **Step 5: Commit** `feat(library): media detail fetch and pure viewer state (+ render smoke-check)`.

---

### Task 2: Full viewer rendering (kills the stub)

**Files:**
- Create: `tldw_chatbook/Widgets/Library/library_media_viewer.py`
- Modify: `tldw_chatbook/UI/Screens/library_screen.py` (list/viewer mode + selection wiring), `tldw_chatbook/Widgets/Library/library_media_canvas.py` (list mode only)
- Test: extend `Tests/UI/test_library_shell.py`

**Interfaces:** `LibraryMediaViewer(Vertical)` — `__init__(self, viewer: LibraryMediaViewerState, **kwargs)`. Compose (using ONLY widgets Task 1 proved render): `‹ Back to list` Button (`id="library-media-back"`, class `library-canvas-action`); title `Static(id="library-media-viewer-title")`; metadata block `Static("\n".join(metadata_lines), id="library-media-viewer-meta")`; a content section header + a **scrollable** content region (`id="library-media-viewer-content"`) rendering `viewer.content` (empty-copy "No stored content." when blank); an analysis section (shown only when `has_analysis`); an actions row `Horizontal(classes="ds-toolbar")` with `Edit`/`Delete`/`Read it later`/`Use in Chat`/`Open in Media` Buttons (ids `library-media-edit`/`-delete`/`-read-later`/`-use-in-chat`/`-open`). (Edit form, highlights, analysis-edit, content-search are added in Tasks 3–7; leave the buttons present but their handlers land in their tasks — a button with no handler yet is fine as long as it does not error; OR add the buttons in their own tasks. Decide and note.)

**Screen wiring:** replace the media-row press behavior — selecting a `.library-media-row` sets `_selected_media_id`, `_library_media_view = "viewer"`, kicks `_refresh_library_media_detail(media_id)`, recomposes (viewer shows a "Loading media…" line until `_library_media_detail` arrives, mirroring the collections/loading pattern). `compose_content`'s media branch: when `_library_media_view == "viewer"` render `LibraryMediaViewer(build_library_media_viewer_state(self._library_media_detail))` (or the loading line); else render the list canvas (`LibraryMediaCanvas`). `#library-media-back` press → `_library_media_view = "list"`, recompose. `#library-media-open` → `NavigateToScreen("media")` (kept as a secondary escape hatch).

- [ ] **Step 1: Failing pilot tests** (seed media with real content via the harness media fake — extend `StaticLibraryMediaScopeService` to also serve `get_media_item` returning a detail dict with content/metadata; wire it in `_seed_conversations`): pressing a media row switches to the viewer showing the title, metadata lines (Type/Author/Keywords), and the content text; `#library-media-back` returns to the list; `#library-media-open` posts `NavigateToScreen("media")`. RED.
- [ ] **Step 2: Verify RED.**
- [ ] **Step 3: Implement** the viewer widget + screen mode wiring + the harness fake's `get_media_item`.
- [ ] **Step 4: GREEN**; run `Tests/Library/` + `Tests/UI/test_library_shell.py` + `test_destination_shells.py`. Re-anchor any media test that assumed the old stub preview.
- [ ] **Step 5: Commit** `feat(library): in-canvas media viewer replaces the preview stub`.

---

### Task 3: Metadata edit (Edit / Save / Cancel)

**Files:** Modify `library_media_viewer.py`, `library_screen.py`; extend the pilot tests.
- Edit form: `Edit` toggles `self._library_media_editing = True` → the viewer renders inputs (`#library-media-edit-title/-author/-url/-keywords`) prefilled from `viewer.edit_fields`, with `Save`/`Cancel`. `Save` → `_run_library_service_call(update_media_item, mode="local", media_id=…, title=…, author=…, url=…, keywords=[…], version=…)`, then re-fetch detail, exit edit mode, recompose; on version conflict re-fetch + quiet notice. `Cancel` → exit edit mode.
- [ ] TDD: failing pilot — press Edit, change the title input, press Save, assert `update_media_item` called with the new title and the viewer shows it; Cancel discards. RED → implement → GREEN → commit `feat(library): edit media metadata in the viewer`.

---

### Task 4: Delete (trash) with confirm

**Files:** Modify `library_media_viewer.py`, `library_screen.py`; extend tests.
- `Delete` → an inline confirm (`Delete` / `Cancel`, mirror the collections/delete confirm pattern if present) → `_run_library_service_call(delete_media_item, mode="local", media_id=…)` → drop the item from `_local_source_records["media"]` (or re-fetch the snapshot), set `_library_media_view = "list"`, recompose.
- [ ] TDD: failing pilot — Delete → confirm → `delete_media_item` called, viewer returns to list, item gone. RED → implement → GREEN → commit `feat(library): delete media from the viewer`.

---

### Task 5: Reading highlights

**Files:** Create highlights display-state in `library_media_viewer_state.py`; modify `library_media_viewer.py`, `library_screen.py`; extend tests.
- On entering the viewer, fetch `list_reading_highlights(mode="local", media_id=…)` (async, alongside the detail). Render a highlights section: each highlight (quote / note / color) as a row; an add form (`quote`/`note`/`color`) → `create_reading_highlight`; per-highlight edit/delete → `update_reading_highlight`/`delete_reading_highlight`. Re-fetch highlights after each mutation.
- [ ] TDD: pilot — highlights list renders; add creates one; delete removes one (assert the service calls + rendered rows). RED → implement → GREEN → commit `feat(library): reading highlights in the media viewer`.

---

### Task 6: Read-it-later + Analysis view/edit

**Files:** Modify `library_media_viewer.py`, `library_screen.py`, `library_media_viewer_state.py`; extend tests.
- Read-it-later: `Read it later` button reflects saved state; toggles via `save_to_read_it_later` / `remove_from_read_it_later`; re-fetch detail to reflect.
- Analysis: show `analysis` when present; an `Edit analysis` toggle → `TextArea` (if it rendered in Task 1; else a plain edit affordance) → Save via `save_analysis_version` / `overwrite_analysis_version`. Analysis (re)GENERATION (LLM) is explicitly OUT — note it as deferred.
- [ ] TDD: pilot — read-later toggle calls the right service and flips label; analysis edit saves via `save_analysis_version`. RED → implement → GREEN → commit `feat(library): read-it-later and analysis view/edit in the media viewer`.

---

### Task 7: In-content search

**Files:** Modify `library_media_viewer.py`, `library_screen.py`; extend tests.
- A search input over the content + `◀`/`▶` prev/next match; highlight/scroll to matches. Pure match logic (find offsets of a query in content) belongs in `library_media_viewer_state.py` (testable); the widget scrolls to them. If the content region could not be made scrollable/searchable with the widgets that render (Task 1 finding), degrade to a match-count + "open in Media for full search" and note it honestly.
- [ ] TDD: pure match-finding tests + a pilot that types a query and asserts the match count/status. RED → implement → GREEN → commit `feat(library): search within media content`.

---

### Task 8: Use-in-Chat handoff

**Files:** Modify `library_screen.py`; extend tests.
- Wire `#library-media-use-in-chat`: build a media→chat handoff payload (mirror `MediaViewerPanel.UseInChatRequested` / the conversation `_open_selected_conversation_handoff` shape) and post/route it so the media opens as Console context. If no clean app-level handler exists, add the minimal handoff (post a `NavigateToScreen("chat"/"console")` with a media-context payload the console screen already understands — verify the console's context intake; if genuinely absent, this task adds the smallest real handler rather than a fake, or is descoped with an explicit note and the button routes through `Open in Media`).
- [ ] TDD: pilot — Use-in-Chat produces the expected handoff (assert via the message/nav capture used by the conversation handoff test). RED → implement → GREEN → commit `feat(library): use media in chat from the viewer`.

---

### Task 9: CSS, whole-branch review, screenshot QA, approval gate

- [ ] **Step 1: CSS** for all new viewer selectors (mirror existing canvas-action/row styles; presence test in both css files); `./build_css.sh`; commit `style(library): media viewer styles`.
- [ ] **Step 2:** Full affected run green (`Tests/Library/`, `Tests/UI/test_library_shell.py`, `test_destination_shells.py`, `test_post_release_workspaces_library_depth.py`).
- [ ] **Step 3:** Whole-branch review (most capable model) over the full media viewer diff; batch-fix findings.
- [ ] **Step 4:** Live captures to `Docs/superpowers/qa/library-media-viewer-2026-07/` at 2050x1240 against real seeded media WITH content: (1) viewer populated (metadata + content + actions); (2) metadata edit; (3) a highlight added; (4) delete-confirm; (5) analysis/read-later. Present to the user for approval before the PR.

## Self-review notes

- Spec coverage: metadata view+edit (T2–T3), content view+search (T2, T7), delete (T4), highlights (T5), read-later + analysis (T6), use-in-chat (T8) → capability parity with MediaViewerPanel except analysis (re)generation (LLM, explicitly deferred) and ingestion (Library Ingest section, out of scope).
- Rendering risk front-loaded into Task 1's smoke-check so no task builds on an unrenderable widget.
- The list + type filter (prior L2a work) are untouched; this replaces only the preview stub. Layout: list-mode ↔ viewer-mode canvas toggle (Back to list), chosen over cramming list+viewer into one narrow column.

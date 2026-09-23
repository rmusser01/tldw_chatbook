# S20 — `UI/*.py` (top level only)

**Coverage:** files read in full: 9 | sampled (targeted regions + full structural/AST scan): 13 | mechanical only
(import census + pattern scans, no prose read): 11 (of 33).
Full: `__init__.py`, `focus_ownership.py`, `stable_command_palette.py`, `image_gen_command_provider.py`,
`tts_profile_recovery.py`, `server_chatbook_service_lease.py`, `character_display_text.py`,
`console_command_provider.py`, `MediaWindowV88.py`.

## Deletion table — the weighted question
Reachability resolved by **running** `resolve_screen_route()` from the registry, not by reading it. Importer counts
are AST-based (relative levels resolved, function-body imports included), production only.

| Module | lines | prod importers | reachable? | pinning tests | verdict |
|---|---:|---:|---|---|---|
| `Stats_Window.py` | 42 | **0** | no | **0** | **delete** |
| `MediaWindowV88.py` | 16 | **0** (alias re-export) | no | 1 test, 18 lines, asserts only `MediaWindowV88 is MediaWindow` | **delete** |
| `Chatbooks_Window.py` | 483 | **0** | no | 0 (one CSS census row) | **delete** |
| `CodeRepoCopyPasteWindow.py` | 1220 | **0** | no | 399-line test-only lifeline | **delete** |
| `SiteConfigSettings.py` | 599 | **0** | no | 225-line test-only lifeline | **delete** |
| `MediaWindow_v2.py` | 2562 | 2 (one dead, one = `media_screen.py`) | **no — route shadowed** | 12 test files incl. a **1,510-line** parity test | **delete (L)** — see P2 |
| `ChatbookCreationWindow.py` | 397 | 1 (`Tools_Settings_Window.py` only) | no | 2 | **delete** with TASK-32807.4 |
| `Outputs_Panel.py` | 594 | 1 (same) | no | 1 | **delete** with TASK-32807.4 |
| `Sharing_Panel.py` | 563 | 1 (same) | no | 1 | **delete** with TASK-32807.4 |
| `tools_settings_messages.py` | 40 | 1 (same) | no | 0 | **delete** with TASK-32807.4 |
| `Tools_Settings_Window.py` | 6928 | 1 (a 49-line screen that itself has 0 importers) | no | 6 | already TASK-32807.4 — **not re-filed** |
| `ChatbookTemplatesWindow.py` | 418 | 2 (one live) | yes | 2 | **keep** |
| `ChatbookExportManagementWindow.py` | 1213 | 1 (live) | yes | 4 | **keep** |
| `Chatbooks_Window_Improved.py` | 807 | 1 → route `chatbooks`, unaliased | yes | 5 | **keep** |
| `Study_/Research_/Writing_/Logs_/LLM_Management_/STTS_Window`, `stts_profile_library`, `stts_playground_catalog`, `Voice_Cloning_Window`, `Dictation_Window_Improved` | — | ≥1 live | yes (routes unaliased) | — | **keep** |
| `destination_recovery`, `character_display_text`, `console_command_provider`, `focus_ownership`, `tts_profile_recovery`, `server_chatbook_service_lease`, `stable_command_palette`, `image_gen_command_provider` | — | live, multi-importer | yes | — | **keep** |

```
$ .venv/bin/python -c "from …screen_registry import resolve_screen_route; …"
media           -> ('library', '…UI.Screens.library_screen')     # the ALIAS wins over _SCREEN_ROUTES["media"]
tools_settings  -> ('tools_settings', '…UI.Screens.mcp_screen')
chatbooks/study/writing/research/logs/stts/llm -> their own screens
```
`_lookup_route` applies `_SCREEN_ALIASES` **before** `_SCREEN_ROUTES`, so `_SCREEN_ROUTES["media"]` can never be
selected by any target string.

**Non-Python references that must move in the same commit as any delete:** `css/screen_css_scoped.tcss:11-70` +
`css/screen_css_self.tcss:11-13` (`SiteConfigSettings`), `css/components/_agentic_terminal.tcss:2158`
(`ChatbookCreationWindow`), `scripts/textual_await_dom_census.tsv:19,49-51`,
`Tests/UI/test_widget_css_consolidation.py:1130`, `Tests/UI/modal_wide_tier_registry.py`.
`Tests/UI/test_legacy_entrypoints_retired.py::RETIRED_MODULES` is the existing landing place for the new rows.

## Findings

### P1 [D1] — `LogsWindow.append_record` mutates the Textual DOM from whatever thread called `logger.*`, with no marshalling
- Where: `UI/Logs_Window.py:402-437` (DOM writes at `:429` `RichLog.write`, `:431`/`:433` via
  `_update_status_line`/`_update_filter_chips`, `:438-442` `_passes` → `query_one`). Caller:
  `app.py:12149-12167` (`PersistentLogHandler.emit`).
- Evidence: `PersistentLogHandler` is added to the **root** logger (`app.py:12186`).
  `rg -n "enqueue" tldw_chatbook/Logging_Config.py` → **no matches**, so the loguru→stdlib bridge runs
  synchronously on the emitting thread; every `@work(thread=True)` worker and `asyncio.to_thread` callee that logs
  enters `append_record` off the event loop. `rg -n "call_from_thread|call_later" UI/Logs_Window.py` → **0 hits in
  `append_record`** — and the file's *other* thread hop (`_persist_filter_state_off_loop:368-377`) is carefully
  documented, so the omission here is a gap, not a house convention. `app.py:12166-12167` wraps the call in
  `except Exception: pass`, so any resulting failure is **silent**.
- Why it matters: `query_one`/`RichLog.write`/`Static.update`/`Button.label=` on a non-main thread mutate widget
  state and enqueue messages on a non-thread-safe `asyncio.Queue` while the Logs screen is open — torn log output
  or a swallowed exception per record, **on exactly the screen a user opens when something is already going wrong.**
- Recommended correction: make `append_record` the marshalling seam — if
  `threading.current_thread() is not threading.main_thread()`, hand the body to `self.app.call_from_thread(...)`.
  Fix it in `Logs_Window` (one place), not in `emit` (which would have to know about the widget).
- Size: S · Confidence: **verified** (thread entry and absence of marshalling both traced; the runtime corruption
  itself is inferred)
- Already covered: none (`rg -l append_record backlog/tasks/` → only `task-19555`, about the Copy-all buffer).

### P1 [D1] — Voice Cloning ▸ Delete profile raises `AttributeError` before the confirm dialog can render
- Where: `UI/Voice_Cloning_Window.py:560-563`, inside `_delete_profile` (`:550`):
  ```python
  class ConfirmDialog(ModalScreen):
      def compose(self) -> ComposeResult:
          with ConfirmContainer(id="confirm-dialog"):
              yield Label(f"Delete profile '{self.selected_profile}'?")
  ```
  `self` here is the `ConfirmDialog`, **not** the enclosing `VoiceCloningWindow` — the outer instance is shadowed by
  the method's own parameter.
- Evidence: **lead-verified by reading `:552-568`.** `ModalScreen` has no `selected_profile`, so compose raises.
  Reachability: `stts_screen.py:61` registers a `voice-cloning` rail row; `STTS_Window.py:2055-2058` mounts
  `VoiceCloningWindow()` for it; the Delete button is composed at `:229-234` and dispatched at `:472-473`, `:723-726`.
  `rg "ConfirmDialog|confirm-dialog" Tests/` → only a CSS-tier registry row, **no behavioural test**.
- Why it matters: **the destructive action on the Speech ▸ Voice Cloning surface is 100% broken** — the modal cannot
  compose, so a profile can never be deleted, and the failure surfaces as a Textual compose error rather than a
  message.
- Recommended correction: bind the outer instance (or pass `selected_profile` into `__init__`); better, replace the
  method-local `ModalScreen` with `Widgets/confirmation_dialog.py`, which `ChatbookExportManagementWindow.py:1084`
  already uses. · Size: S · Confidence: **verified**

### P2 [D4] — The Chatbooks card always appends `"..."`, so a chatbook with no description renders as literally `"..."`; four sibling copies of the same idiom get it right
- Where: `UI/Chatbooks_Window_Improved.py:92-94` (grid card) and `:560-565` (list row). Correct siblings:
  `STTS_Window.py:567-568, 678-682, 841-845, 1428-1432` (all `x[:N] + "..." if len(x) > N else x`).
- Evidence: `_scan_chatbooks` (`:617-655`) sets `"description"` **only** inside the `manifest.json` branch. So a zip
  without a manifest → key absent → `.get("description","No description")` → renders **`No description...`**; a zip
  with a manifest and no description → `""` → renders **`...`**. The route is live.
- Size: S · Confidence: verified
- Already covered: **TASK-32808.3 (In Progress)** owns the consolidation but does not name these two sites — and
  **the defect here is a behaviour bug, not just a re-roll**, so adopting the helper fixes it. Say so in that task.

### P2 [D1] — `MediaWindow_v2.py` (2,562 lines) and `media_screen.py` are unreachable by construction, and the app's own pre-importer already treats them as such
- Evidence: `resolve_screen_route("media")` returns `library_screen`.
  `rg -n "media_screen|MediaScreen" tldw_chatbook/` → the registry entry, the lazy `UI/Screens/__init__.py` map, and
  docstrings only — **no constructor call anywhere**. Independently: `app.py:17814` computes
  `shadowed_route_ids = set(registered_screen_aliases())` and `:17822` skips those routes in the background
  pre-importer, with the comment *"a route no click can ever reach should not be attempted"* — **the app already
  classifies this route as dead.**
- Why it matters: 2,725 production lines and 12 test files (incl. a 1,510-line parity test) maintained for a surface
  no click reaches.
- Size: L · Confidence: verified
- Already covered: **none of TASK-32807's sub-tasks names the UI-root windows or `media_screen`.**

### P2 [D1] — Untrusted names are interpolated raw into `markup=True` `RichLog`s; the existing markup task cannot reach these sites because they call no escaper at all
- Where: `UI/STTS_Window.py:1435` (`log.write(f"[yellow]Generating preview for: {chapter.title}[/yellow]")`, log
  declared `markup=True` at `:434`) and `UI/Voice_Cloning_Window.py:647` (same shape with `test_profile`, log
  `markup=True` at `:263`).
- Evidence: `RichLog._make_renderable` does `Text.from_markup(content)` when `self.markup`. Then:
  ```
  Text.from_markup("[yellow]t: a[/b]b[/yellow]")   -> MarkupError: closing tag '[/b]' does not match any open tag
  Text.from_markup("[yellow]t: Bob [v2][/yellow]") -> renders "t: Bob "     # segment silently deleted
  ```
  `chapter.title` comes from chapter detection over imported text; `test_profile` is a user-named voice profile.
- Why it matters: at `STTS_Window:1435` the raise is caught by the enclosing `try/except` at `:1415` → **the preview
  never generates** and the user gets a cryptic toast. At `Voice_Cloning_Window:647` there is **no handler** and the
  caller is a bare `asyncio.create_task` → the exception goes to asyncio's default handler → **the Test-voice button
  silently does nothing.**
- Size: S · Confidence: **verified**
- Already covered: **TASK-32802.1 (In Progress) is insufficient for these two.** Its AC #2 is scoped to *"every site
  that currently calls `rich.markup.escape`"*; **these sites call no escaper, so a sweep driven by that AC will not
  find them.** Widen the AC to "every markup-ON sink that interpolates non-literal text", or add these rows.

### P2 [D1] — `Voice_Cloning_Window` drives eight user actions with bare `asyncio.create_task`, never cancelled at unmount, while the same file uses `@work` correctly for one path
- Where: `UI/Voice_Cloning_Window.py:706-717` (`_spawn_action`), consumers at `:719-739`; correct sibling at `:325`
  (`@work(exclusive=True, group="voice-cloning-load-profiles")`).
- Evidence: `_action_tasks` is defined (`:184`), added to (`:715`) and discarded on completion (`:716`); **there is
  no `on_unmount` in the file**, so nothing cancels a live task. Several spawned coroutines await a pushed screen
  and then touch the DOM (`_delete_profile:566` awaits `push_screen_wait` then calls `self.notify`/`_load_profiles`).
- Why it matters: navigating away while a picker or confirm dialog is open leaves a task that resumes against a
  detached widget; the resulting `NoMatches`/`NoScreen` lands in asyncio's default handler, **invisible**.
  `run_worker` solves both this and the GC problem the comment at `:709-712` was written for. · Size: S

### P3 [D3] — `Study_Window._configure_flashcards_lifecycle_controls` queries a widget it discards, computes a flag it discards, and writes the same property twice
- Where: `UI/Study_Window.py:898-912`. `git log -L 898,913:…` → commit `8a9ce5cf9c` (PR #707) turned
  `delete_deck_button = self.query_one(...)` into a bare call and `scope_enabled = bool(scope_checker())` into
  `bool(scope_checker())` — i.e. **the unused-variable warnings were silenced by deleting the assignment rather than
  adding the missing use.** The duplicated `display` line predates it. The gating is not lost
  (`UI/Study_Modules/flashcards_handler.py:374,383-388` does gate the button), but the discarded `query_one` is
  load-bearing as an undocumented presence check and the sibling `_configure_quizzes_lifecycle_controls:934-975`
  gates 14 widgets from the same inputs, so the asymmetry reads as a bug to every future reader. · Size: S

### P3 [D3] — Dead dispatch branch: `STTS_Window` pushes a `Widget` as a `Screen` for a button id that is never composed
- `UI/STTS_Window.py:2525-2528` — `self.app.push_screen(VoiceCloningWindow())`. `rg -n "view-voice-cloning-btn" .`
  → **exactly one hit, this branch**; no such Button exists anywhere. `VoiceCloningWindow` is a `Vertical`, and
  `push_screen` raises `TypeError` for a non-`Screen` — **a latent `TypeError` armed behind a dead id.** The live
  path (`:2055-2058`) is correct. · Size: S · Confidence: verified

### P3 [D4] — Non-adopters of `Widgets/status_line.py::set_status_line`, with a drifted missing-widget contract
- `UI/stts_profile_library.py:2759-2770` (full non-adopter) and `UI/Writing_Window.py:258-270` (**half**-adopter:
  calls the helper for `#writing-status` then hand-rolls the identical body for `#writing-source-status` at
  `:265-270` with a bare `except Exception: pass` — one line adopted, one re-rolled, **in the same method**).
- **The drift is a crash shape:** `stts_profile_library._set_status` guards with `if not self.is_mounted: return`
  then does four unguarded `query_one`s. That guard covers "this widget is unmounted" but **not** "that child is
  gone" — where the helper's `missing_ok=True` swallows the lookup error and returns `False`, this propagates
  `NoMatches` into the calling async page-load path (`:2369, 2396, 2424, 2443, 2447, 2451`). · Size: S
- Already covered: **task-32861 (To Do)** — these are the slice's non-adopters; the crash-shape drift is new detail.

## Candidate triage
**RETIRED — three of the lead's dispatch hypotheses, with evidence:**
- **`LLM_Management_Window.py` subprocess/survivable processes:**
  `rg -n "subprocess\|Popen\|os\.system\|terminate\|kill\(" …/LLM_Management_Window.py` → **zero matches.** The file
  never spawns or stops a process; that lives in `Event_Handlers/LLM_Management_Events/server_lifecycle.py`, which is
  TASK-32806.5's subject. The only slice subprocess sites are `ChatbookExportManagementWindow.py:1162-1176` (a
  fire-and-forget `open`/`xdg-open` launcher — `timeout=` is meaningless for a non-waited `Popen`, and it is
  documented) and `CodeRepoCopyPasteWindow.py:358` (no `timeout=`, but on a **0-importer** module → covered by the
  delete row).
- **`Research_Window.py` / `Writing_Window.py` hand-roll `_set_status`:** **both import `set_status_line`** and are
  adopters on this dev tip. `stts_profile_library.py` is the slice's only true non-adopter.
- `UI/Logs_Window.py:459 _compile_pattern` — known-deliberate; the memo cache at `:452-458` is real.

**RETIRED:** `dotted_section_setting` ×8 (`Dictation_Window_Improved.py:843-853`) — all the **2-arg** dotted form,
which `config.py:8508-8620` explicitly supports (TASK-1771) and walks the nested tree for 3-segment paths.
`query_one_in_timer_no_try` `stts_profile_library.py:3537` — `on_unmount` at `:2157` explicitly stops the timer.
`plain_readback` `Logs_Window.py:527,617` — both compare against labels this file itself set; no user text reaches
them. `inline_truncate` 4 of 5 (`STTS_Window`, all correctly conditional). `except_exception_pass` ×36 /
`except_exception_return` ×17 — 24 of the 53 are in dead modules; the live ones are deliberate UI-default guards
with `# noqa: BLE001` and a reason. `run_worker_coroutine` ×87 — **AST scan for `run_worker(..., exclusive=True)`
without `group=` over all 33 files → zero hits**, and `rg "execute(\|fetchall\|sqlite3"` over the live files →
**zero** (no direct DB access anywhere in this slice). `raw_1024x1024` ×6 — byte→MB display divisions (the
TASK-32808.1 cluster). `os_replace_no_atomic`/`tempfile_no_secure` ×4 — all `Tools_Settings_Window.py`, owned.
`strftime` ×12 — display-only or filename stamps. `legacy_markers` ×38 — 24 owned, rest descriptive.
`function_body_import` ×70 — mostly documented lazy-loads; **one exception folded into P1b**:
`Voice_Cloning_Window.py:556-558` imports already-imported Textual modules inside `_delete_profile` purely to define
a method-local `ModalScreen` subclass — **a fresh class object, and a fresh Textual stylesheet parse-cache slot, per
press.** `DUP_VERBATIM _maybe_await` ×3 (`MediaWindow_v2:223`, `Outputs_Panel:223`, `Sharing_Panel:176`) — retired,
and moot: **all three modules are on the delete list**. No call site in this slice does blocking I/O on the loop.
**One observation not filed:** `Chatbooks_Window_Improved.py:625-627` and `ChatbookExportManagementWindow.py:508`
both use `st_ctime` as "created", which on POSIX is the **inode-change** time — a `chmod`/move rewrites the card's
date.

## D4 observations for repo-wide Phase 3
**Helper exists, ignored:**
1. `Widgets/status_line.py::set_status_line` (17 importers) — non-adopters above; **`is_mounted`-guard vs
   `try/except` is not equivalent.** Owner task-32861.
2. `Widgets/confirmation_dialog.py` — `ChatbookExportManagementWindow.py:1084` imports it;
   `Voice_Cloning_Window.py:560` defines a method-local `ModalScreen` instead **and gets it wrong** (P1b).
3. Atomic-write helper — non-adopters: `Dictation_Window_Improved.py:997-999` and `:1035-1037`
   (`open(filename,"w")` + `f.write(self.transcript_text)`, unbounded, non-atomic, after a raw `mkdir`).
4. Byte-size formatter (TASK-32808.1) — add the inline `/(1024*1024)` re-rolls at
   `Chatbooks_Window_Improved.py:110`, `ChatbookExportManagementWindow.py:536,932,972`, `Chatbooks_Window.py:237`,
   `CodeRepoCopyPasteWindow.py:1137`. **Two of the six are in delete-list modules, so the cluster shrinks by 2 if the
   deletes land** — worth sequencing the deletes first.
5. Filename sanitizer (TASK-32808.2) — `ChatbookExportManagementWindow.py:839-851` hand-rolls one against a
   **server-supplied** `chatbook_name`. Correct as written on POSIX, but a trust-boundary sanitizer living in a UI
   module.

**No helper exists, N copies drifted:**
6. **Chatbook export-directory scanner.** `Chatbooks_Window_Improved.py:608-661` and
   `ChatbookExportManagementWindow.py:490-516` both glob+stat `get_private_chatbooks_dir()`. **Drift:** different
   dict keys (`size_mb` float vs `size` int; `created_at` ISO string vs `created` datetime), different sort keys
   (`st_ctime` vs `st_mtime`), and only the first reads `manifest.json`. A third copy lives in the dead
   `Chatbooks_Window.py:227-260`. **The same directory lists in two different orders across two screens of the same
   feature.** Home: `Chatbooks/database_paths.py` (already owns `get_private_chatbooks_dir`) or a new
   `Chatbooks/export_inventory.py`.
7. Truncate-with-ellipsis — 6 copies in slice, 4 correct, 2 unconditional. Feeds TASK-32808.3.
8. `_sentence`/trailing-period normaliser — `UI/destination_recovery.py:54` ≡ `Library/library_rag_state.py:649`,
   byte-identical. Four lines; only worth folding if a `Utils/` text home already exists.
9. **Note for the markup stream:** `UI/character_display_text.py` is the repo's designated sanitizer for untrusted
   character-card text (**18 production importers**) and deliberately handles control chars, surrogates and
   negative-width glyphs — **but not Textual markup brackets.** Every one of its 18 callers must therefore remember
   to escape separately. **If TASK-32802 lands a canonical escaper, this is the highest-leverage place to compose
   it in.**

## Left UNVERIFIED
| Claim | Why | Command |
|---|---|---|
| The `Logs_Window.append_record` cross-thread DOM write actually corrupts output / raises, rather than being formally unsafe | needs a live Textual loop plus a logging worker thread; the app must not be booted, and a race is nondeterministic | mount `LogsWindow` under `app.run_test()`, then `await asyncio.to_thread(lambda: [logging.getLogger("x").info("line %d", i) for i in range(2000)])` and assert `len(window._records) == 2000` and that the `RichLog` line count matches |
| Voice Cloning ▸ Delete reaches the `AttributeError` in a running app (vs being gated earlier by a disabled button) | `#delete-profile-btn` starts `disabled=True` and is enabled at `:435` only when a profile is selected; selection not exercised | a test that mounts the window, sets `selected_profile`, and calls `await window._delete_profile()` |
| Whether removing `_SCREEN_ROUTES["media"]` breaks anything beyond its 12 test files | traced every static reference, not the full test-collection graph | `pytest --collect-only -q 2>&1 \| grep -ci media_screen` and `rg -c "MediaWindow" Tests/` before/after a local removal |
| Whether `Voice_Cloning_Window`'s uncancelled tasks actually resume after unmount | needs a live screen switch mid-picker | mount under `app.run_test()`, call `action_delete_profile()`, `await pilot.pause()`, `await window.remove()`, assert cancelled |
| Whether any delete-list module is loaded by a plugin/entry-point mechanism outside the repo | only the repo tree was searched | `rg -n "entry_points\|importlib.import_module\|__import__" pyproject.toml tldw_chatbook/ \| rg -i "UI\."` |

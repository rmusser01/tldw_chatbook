# S22 + S23 — `UI/{Evals,LLM_Management,CCP_Modules,…}/`; `Widgets/{Media,Prompts,TTS,NewIngest,…}/`

**Coverage S22** (77 files / 29,039 lines): read in full **2** · sampled **22** · mechanical only **53**.
**Coverage S23** (39 files / 10,709 lines): read in full **4** · sampled **6** · mechanical only **29**.
Mechanical sweeps run over **all 116**: duplicate-def AST check (**0 hits**), mutable `reactive([])` defaults (**0**),
`query_one` in `try` non-body sections (**2**, both benign `except` fallbacks), `run_worker`/`@work` inventory (6+4),
timer inventory (13), threading/file-IO/subprocess inventory, an unescaped-markup-sink AST pass (**276 raw
candidates**, filtered by hand), and a production-importer + reachability walk from `app.py` + `screen_registry`
(including its string module paths).

## Findings

### P1 [D1] — Every flashcard row in Study renders with its queue-state badge deleted; the code writes it, Textual eats it
- Where: `UI/Study_Modules/flashcards_handler.py:665-666` —
  `label = f"{card.get('front','')} [{queue_state}]"` → `ListItem(Label(label))`.
- Evidence: `Content.from_markup("What is DNA? [new]").plain` → **`'What is DNA? '`**. Same for `[learning]`,
  `[review]`, `[suspended]`, `[unknown]` — **every value `Study_Interop/study_normalizers.py:126` produces.**
  `Label` defers parsing, **so `str(label)` still looks right in a test**; the deletion happens at render.
- Why it matters: the card list's only queue-state affordance is **invisible in the shipped UI, on every row,
  always.** Silent — no exception, no log.
- Recommended correction: build the badge outside markup (`Label(Content(...))` or `markup=False`), or
  `escape_markup(...)` the label. **The canonical helper is already imported two files over**
  (`UI/Evals/library_rail.py:76`). · Size: S · Confidence: **verified**
- Already covered: **no** — TASK-32802's five children name Console, Library, themes, file-notes and the shared
  confirmation dialog. **None names Study.**

### P1 [D1] — User-typed and wire-sourced text reaches markup-parsing sinks on 11 files across both slices; `[/…]` raises `MarkupError` inside `_compositor.reflow`, which nothing catches

| file:line | source of the text | sink |
|---|---|---|
| `UI/Study_Modules/flashcards_handler.py:594, 355` | deck names (user-typed) | `Select.set_options` |
| `UI/Study_Modules/quizzes_handler.py:388, 511` | quiz names | `Select.set_options` |
| `UI/Study_Modules/quizzes_handler.py:454, 267` | question text | `Label`, `Static.update` |
| `UI/LLM_Management/vllm_setup_view.py:1010-1011` | **model ids off the wire** from `/v1/models` | `Select.set_options` |
| `UI/LLM_Management/vllm_setup_view.py:1070-1073` | vLLM profile names | `Select.set_options` |
| `UI/Evals/skill_eval_panel.py:110, 115, 91` | skill/target names | `Select.set_options`, `Static.update` |
| `UI/Research_Workspace_Modules/quick_notes_section.py:237` | note titles | `Select.set_options` |
| `UI/Research_Workspace_Modules/sources_region.py:573` | folder names | `Select.set_options` |
| `UI/Research_Workspace_Modules/add_source_modal.py:400` | catalog item titles | `Select.set_options` |
| `UI/Research_Workspace_Modules/source_list.py:159`, `source_inspector.py:80` | source titles | `Static.update` |
| `Widgets/Media/media_navigation_panel.py:130` + `:189` | the DB `type` column (**free-form — no allowlist at insert**) | `Button(label)`, then `str(event.button.label)` reads the **mangled** name back and posts it as `MediaTypeSelectedEvent.display_name` |

- Evidence (Textual 8.2.8): `Button('[draft] video').label` → `' video'`; escaped first → `'[draft] video'`.
  `Button('Report [/] x')` → **`MarkupError` at construction.** A Select with a `[/]` option, `expanded = True` →
  **`MarkupError` raised inside `textual/_compositor.py:387 reflow` ← `screen.py:1352 _refresh_layout`.**
  `_is_admissible_model_id('model[/]x')` → **True** — ADR-114's boundary bans control chars, `\`, path shapes and
  `.gguf` but **allows brackets**, so a remote `/v1/models` body reaches the Select unfiltered.
- Why it matters: **a new and worse shape than 32802 documented.** `.2`/`.3` are compose()-time raises in one
  dialog, which a `try` around the mount can contain. **A raise during reflow is not confined to the widget that
  caused it.** Reachable by typing a deck name, a folder name, or pointing vLLM setup at an odd server.
- Recommended correction: `escape_markup` at each site — **the same call `UI/Evals/bench_editor.py:907-913` and
  `library_rail.py:179-196` already make in this slice, with comments naming exactly this hazard.**
- Size: M · Confidence: **verified** (mechanism + each site read)

### P1 [D4] — `Widgets/Home/home_rail.py` has a non-escaping same-named twin of the Library helper that *does* escape; TASK-32802.1's finding text says the opposite
- Where: `Widgets/Home/home_rail.py:21 _visible_row_title` (**truncate only**) → `:127`, `:170`
  `button.label = f"… {_visible_row_title(row.title)} …"`. Escaping twin:
  `Widgets/Library/library_rail.py:377` → `escape_markup(_truncate_row_title(title, budget))`.
- Evidence: **lead-verified — see `phase4-verification.md`.** `grep -c escape_markup` → **home_rail 0, library_rail
  4**; home_rail never imports `input_validation`. TASK-32802.1's text lists `home_rail.py:127,170` as *consumers of
  the shared escaper*.
- Why it matters: **the .1 fix will land in the Library helper, AC#1 will pass on Library rows, and Home rail titles
  stay unescaped.**
- Recommended correction: make `home_rail._visible_row_title` delegate to the Library helper — **it already takes a
  `budget` parameter.** · Size: S · Confidence: **verified**
- Already covered: **TASK-32802.1, but stated wrongly — report as a correction to that task.**

### P1 [D3] — `Widgets/NewIngest/` (5 modules, 1,240 lines) has zero production importers and is unreachable from `app.py` or the screen registry
- Evidence: AST import walk → the only non-test importers of each module are **its own four siblings**; external
  importers are six `Tests/` files. Reachability walk from `app.py` + `screen_registry` (including string module
  paths): **all five UNREACHABLE.** **The package says so itself:** `__init__.py:1` *"Legacy NewIngest compatibility
  exports."*, `SmartFileDropZone.py:1` *"Compatibility smart file drop zone for legacy NewIngest tests."*, and a
  plan doc: *"its `UnifiedProcessor`/`BackendIntegration` are mocked test shims and MUST NOT be adopted."*
  **It also carries test scaffolding in production code** — `CaptureSafePostMixin.__setattr__` intercepts
  `post_message` assignment purely so tests can capture messages, and `ImmediateButton` reimplements `Button.press`
  to skip Textual's animation.
  **Independent corroboration that it has never run:** `SmartFileDropZone._browse_files:232` calls
  `self.app.push_screen_wait(...)` **without `await`** from a sync handler; `push_screen_wait` is a coroutine
  function on 8.2.8, so `selected` is an un-awaited coroutine, always truthy, and `list(selected)` raises
  `TypeError`. · Size: S · Confidence: **verified**
- Pinning test: **the six test files ARE the keep-alive.**
- Already covered: **no** — TASK-32807.1 is scoped to *top-level* widget modules, and the 2026-09-17 census mentions
  `SmartFileDropZone.py` only in the byte-formatter duplication table, never in a deletion list.

### P2 [D1/D2] — The Evals snippet import reads a user-picked file with no size ceiling, on the event loop, while its sibling in this same slice caps at 2 MiB
- `UI/Evals/snippet_editor.py:634` `content = file_path.read_text(encoding="utf-8")` inside a `push_screen`
  callback (**the UI thread**). Bounded sibling: `UI/Chunking_Lab_Modules/sample_region.py:36-57 read_sample_file` —
  `SAMPLE_BYTES = 2 * 1024 * 1024`, reads `SAMPLE_BYTES + 1` and refuses. The snippet editor validates the path and
  handles `OSError`/`UnicodeDecodeError` **but never bounds the read**. · Size: S · Confidence: verified
- Already covered: **no** — TASK-32806.6 names the read tool, the directory tool, three editing tools, character-card
  import and trajectory import. Not this one.

### P2 [D3] — `UI/CCP_Modules/ccp_message_manager.py` (317 lines): `CCPMessageManager` is never constructed in production
- `grep -rn "CCPMessageManager("` → **one hit, a test.** `personas_screen.py:344-348` imports four sibling handlers,
  **not this one.**
- Why it matters: it carries a live-looking `@work(thread=True)` DB read with **no `group=`, no `exclusive=True` and
  no generation token** (`:67-105`), so two rapid conversation selections would race and the older loader could win.
  **That defect is inert only because nothing constructs the class — exactly the shape a future wiring commit
  revives.** · Size: S · Confidence: verified

### P2 [D3] — Three more unreferenced modules, each one census line short of the existing deletion tasks
- `UI/Workbench/route_inventory.py` (118) — production importers **0**; reached only through a lazy `_LAZY_EXPORTS`
  table; two prose references in `screen_registry.py:198,263` **point at the file but do not import it**.
- `UI/Widgets/config_search_widget.py` (228) — its only consumer is `UI/Tools_Settings_Window.py`, **which
  TASK-32807.4 is deleting**. Not named in .4's ACs, so **.4 will orphan it rather than remove it.**
- `Widgets/Coding_Widgets/repo_tree_widgets.py` (726) — single importer `UI/CodeRepoCopyPasteWindow.py`, itself
  **UNREACHABLE** (and on S20's delete list).
- `Widgets/Note_Widgets/note_creation_modal.py` (265) — single importer `Widgets/document_generation_modal.py`,
  which is on TASK-32807.1's **deferred** delete list. **Transitively dead; named nowhere.**
- Recommended correction: **fold these four rows into TASK-32807's census before .1/.4 land, so the deletions do not
  leave orphans.** · Size: S · Confidence: verified

### P2 [D3/D2] — `MediaViewerPanel` defines a `ModalScreen` subclass inside a worker body, so a new Screen class (with its own `DEFAULT_CSS`) is created on every Delete press
- `Widgets/Media/media_viewer_panel.py:2019-2100` — `class DeleteConfirmDialog(ModalScreen)` inside
  `@work(exclusive=True, …) async def _run_delete_confirmation`, plus function-body imports of `textual.widgets`,
  `textual.containers`, `textual.screen`.
- Why it matters: (a) Textual's `_MessagePumpMeta` snapshots `@on` handlers per class at class creation and
  `DEFAULT_CSS` is registered per type — **a fresh type per press means repeated registration and stylesheet growth
  for the session**; (b) the dialog has no `BINDINGS`, no escape handling and no `_perform_safe_cancel`, so it is
  **the only modal in these two slices outside the repo's `Widgets/modal_dismissal` convention**; (c) the result is
  read off the instance instead of `dismiss(value)`; (d) `textual.*` are not optional deps, so the per-call imports
  buy nothing. · Size: S · Confidence: verified

### P2 [D4] — The two Study controllers carry 7 byte-identical methods and 5 more that differ only in string-literal vs enum
- `UI/Study_Modules/flashcards_handler.py` ↔ `quizzes_handler.py`, 17 shared method names. **AST-normalised diff,
  not a text grep.** Identical: `_current_mode`, `_is_blank_select_value`, `_notify`, `_policy_action_allowed`,
  `_scope_state`, `_scope_type`, `_scope_type_value`. **Drifted only in the scope constant:** flashcards compares
  `!= 'workspace'`, quizzes compares `!= StudyScopeType.WORKSPACE.value`.
- Why it matters: **that drift is exactly the shape that turns into a real defect the day the enum value changes —
  one controller silently starts treating every workspace as global.** Today they are equal; nothing enforces it.
  · Size: M · Confidence: verified

### P3 [D2] — Three `re.compile` calls of constant patterns inside pydantic validator bodies
- `UI/CCP_Modules/ccp_validators.py:83, 101, 142`, recompiled per field validation **on the character-card import
  path**. (`:188`'s compile is of *user input* and is correct where it is.) · Size: S

### P3 [D3] — A Screen reaches into a controller module's private helper
- `UI/Screens/personas_screen.py:6961, 14600, 14632` call `ccp_character_handler._default_character_db()`. · Size: S

## Candidate triage
**CONFIRMED:** `plain_readback` `media_navigation_panel.py:189` (folded into the markup finding);
`re_compile_in_def` 3 of 4; `function_body_import` `media_viewer_panel.py:2032-2034`;
`legacy_markers` `Widgets/Evals/__init__.py:1` (a 1-line empty namespace, 0 importers).
**RETIRED:** `plain_readback` `prompt_block_editor.py:84 str(static.renderable)` — **symptom real, cause wrong.**
`Static` has no `renderable` on Textual 8.2.8, **but `tldw_chatbook/__init__.py:78-89
_install_textual_compatibility_shims` installs it as a property aliasing `Static.content`.** Reproduced end to end:
`_present_static(mounted_static, "new copy")` → OK. **Note for the lead: this package-level monkeypatch of a
third-party class is itself worth a D3 row** — it is why `str(x.renderable)` reads as valid in ~6 files.
`id_keyed_dict` `trajectory_timeline.py:184` — `TimelineModel.__init__:130-137` builds `_record_keys` from
`self._timed`, **which the same instance retains as a tuple**, so no key's object can be collected while the dict
lives. Not a recompose-lifecycle cache. `except_exception_pass` ×5 in `vllm_setup.py` — all child-process
reap/close cleanup carrying `# noqa: BLE001, S110 - cleanup never exposes child details`. `except_exception_pass`
×44 across Study/Media/Writing — **retired as a class**: every one read is a `query_one` miss around optional
chrome; **no swallowed DB write found in either slice.** `run_worker_coroutine` `add_source_modal.py:366` — an
`async def` given to `run_worker` with `group=` + `exclusive=True`. `function_body_import` ×10 — documented
circularity, exception-type imports, storage-layer laziness the boot-budget test pins. `raw_mkdir`
`vllm_profiles.py:712` — surrounded by `os.open(..., O_EXCL|O_NOFOLLOW|O_NONBLOCK)` and mode `0o600`.
`inline_truncate` ×3 — duplication only, TASK-32808.3 owns them. `strftime` ×3 — `trajectory_timeline.py:111`
documents local time as a deliberate match to the ledger columns; the others are display-only.
`DUP_VERBATIM` "CCP `_notify`/`_current_chat_id` byte-identical" — **retired**, AST diff shows both drifted
(docstring; `Optional[str]` vs `str | None`). **vLLM probe bypasses egress — retired**: the target is a user-typed
local endpoint (egress's own definition of a trusted origin), `follow_redirects=False`, the body is capped at
64 KiB, and the credential is never logged.
**`_maybe_await` D1 — not found.** The two call sites in these slices both take scope-service methods that are
`async def`.
**`_set_status`/task-32861 — zero definitions and zero `Widgets/status_line` importers in either slice.**

## D4 observations for repo-wide Phase 3
1. **`_visible_row_title` — two same-named helpers, one escapes and one does not.** Canonical home: the Library one,
   already `budget`-parameterised. **This drift is a D1 today and it invalidates part of TASK-32802.1's finding text.**
2. **Bounded vs unbounded user-file admission.** `sample_region.read_sample_file` (2 MiB ceiling, path validation,
   strict UTF-8, excerpt variant) vs `snippet_editor.py:634` (validated path, **no ceiling**). Both are "user picks
   a file through `FileOpen`, we read it". Home: promote `read_sample_file`'s shape into `Utils/file_handlers.py`.
3. **Study controller pair** — 7 byte-identical + 4 constant-drifted methods. Home: a `UI/Study_Modules/` mixin.
4. **CCP controller pair** — 10 shared method names, 1 identical, 9 drifted; 6 of those are same-shape CRUD pairs
   differing only in entity. **Lower priority than #3 because the drift is real, not cosmetic.**
5. **`_notify` — helper exists, four re-rolls ignore it.** `UI/Evals/notify_mixin.py` (48 lines) is the same idea,
   **already extracted and shared** by `LibraryRail`/`ResultsGrid`/`SnippetEditor` — its docstring says so — while
   the two CCP and two Study handlers each carry the identical `getattr(window,'notify') or getattr(app,'notify')`
   fallback.
6. **Byte-size formatter** — two of TASK-32808.1's 16 members (`SmartFileDropZone.py:97`,
   `repo_tree_widgets.py:201`) **live in modules this review finds dead. TASK-32807 and TASK-32808.1 should
   reconcile: deleting them shrinks the cluster by 2 with no adoption work.**
7. **Third-party class patched at package import.** `tldw_chatbook/__init__.py:85` adds `Static.renderable`, used by
   6 files. **Every one of those reads is invalid against stock Textual 8.2.8 and only works because of a 5-line
   shim in `__init__`.** Worth one D3 row and a comment at each consumer.

## Left UNVERIFIED
| Claim | Why | Command |
|---|---|---|
| A deck name containing `[/]` reaches `#deck-select` and crashes the live Study screen (**the Textual mechanism and the unescaped code path are proved; the end-to-end journey is not**) | app must not be run; **and every `Tests/UI/*` errors in setup at `app.py:1116 APP_CONFIG = load_settings()` — the ADR-126 `RecoveryRequired` gate — in this clean worktree** | `TLDW_CONFIG_PATH=<scratch> pytest Tests/UI/test_study_screen.py -q` after creating a deck named `Deck [/] x` |
| `media_type` values with brackets exist in a real database | `add_media_with_keywords` applies no allowlist to `type`, but not every ingest caller was traced | `rg -n "media_type=\|\"type\":" tldw_chatbook/Local_Ingestion/ tldw_chatbook/Library/library_ingest_*.py` |
| The vLLM external-model Select is reachable without a live server | requires running against a server | point the app at a stub `/v1/models` returning `{"data":[{"id":"model[/]x"}]}` |
| `UI/Widgets/SmartContentTree.py:261 node.label.plain` un-escapes user text | only consumer is `ChatbookCreationWizard.py:194`; what feeds `node.label` not traced | `rg -n "add_leaf\|add(\|set_label" …/SmartContentTree.py …/ChatbookCreationWizard.py` |
| Whether the six `NewIngest` keep-alive tests currently pass (relevant to deletion cost) | not run; **they are the justification for the modules' existence** | `pytest Tests/Widgets/test_new_ingest_end_to_end.py Tests/Widgets/test_unified_processor.py -q` |
| The 276-row unescaped-markup AST sweep beyond the 16 sites hand-classified | filtered by field name (`name`/`title`/…); **rows interpolating status enums and counts were dropped without individual tracing** | re-run the sweep script; raw output is 276 rows of `file<TAB>line<TAB>sink<TAB>expr` |

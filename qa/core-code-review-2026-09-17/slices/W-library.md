# W-library — tldw_chatbook/Widgets/Library/, 37 files, 40112 lines

Worktree: /Users/macbook-dev/Documents/GitHub/tldw-review (detached @ origin/dev d8fb4053f9)

## Coverage
| file | lines | status |
| --- | ---: | --- |
| library_file_notes_workspace.py | 8846 | sampled by symbol cluster, ~2,600 lines read (1-500, 1280-1500, 2280-2520, 2830-3010, 3130-3500, 3550-3620, 4760-4880, 5440-5620, 5760-6000, 6040-6120, 6250-6320, 6390-6440, 6540-6560, 6640-6940, 7040-7130, 7500-7960, 8040-8140, 8340-8400, 8800-8846) + full `ast` method census + pattern greps over all 8,846 |
| library_file_notes_git_panel.py | 4291 | read in full (delegated deep-read under the same evidence rules); headline timing probe independently re-run by me |
| library_notes_canvas.py | 3754 | read in full (delegated deep-read) |
| library_ingest_canvas.py | 2186 | read in full (delegated deep-read) |
| library_media_canvas.py | 2054 | sampled by symbol cluster (188-215, 590-810, 900-1020, 1130-1250, 1780-1990) + greps over all 2,054; speaker-rename flow read in full and probed |
| library_skills_canvas.py | 1998 | read in full (delegated deep-read) |
| library_prompts_canvas.py | 1857 | read in full (delegated deep-read); P0 probe independently re-run by me |
| library_notes_add_from_files_canvas.py | 1699 | read in full (delegated deep-read) |
| library_search_rag_panel.py | 1433 | read in full (delegated deep-read) |
| library_media_viewer.py | 1421 | read in full (delegated deep-read); rename flow also read by me |
| library_note_import_canvas.py | 1376 | read in full (delegated deep-read) |
| library_rail.py | 1338 | read in full (delegated deep-read); `_visible_row_title` probed by me |
| library_collections_capture_reader.py | 1072 | read in full (delegated deep-read) |
| library_conversation_reader.py | 775 | sampled (250-400, 620-700, 740-760) + pattern greps |
| library_media_trash_canvas.py | 731 | sampled (165-250, 420-460, 670-700) + pattern greps |
| library_entry_canvases.py | 577 | sampled (440-560) + pattern greps |
| library_media_content.py | 550 | sampled (280-300, 380-400, 430-450, 515-540) + pattern greps |
| library_media_raw_view.py | 513 | read in full-ish (60-215, 430-445, 500-513 read; remainder scanned) + probes |
| library_adaptive_reader_shell.py | 452 | sampled (120-220, 300-320, 400-420) + pattern greps |
| library_conversations_canvas.py | 420 | sampled (40-120, 290-345) + pattern greps |
| library_export_canvas.py | 289 | sampled (50-115, 140-260) + pattern greps |
| library_notes_sync_roots_canvas.py | 278 | sampled (240-278) + pattern greps |
| library_browse_reader_shell.py | 276 | mechanical only (greps: 0 query_one, 0 timers, 0 excepts) |
| library_canvas_sync.py | 247 | read in full |
| library_media_image_preview.py | 244 | read in full |
| prompt_delete_confirmation_modal.py | 221 | read in full (delegated deep-read) |
| __init__.py | 207 | read in full + import-time measured |
| library_skill_work_pane.py | 182 | read in full (delegated deep-read) |
| library_review_set_picker.py | 166 | sampled (95-166) + pattern greps |
| notes_recovery_dialog.py | 154 | read in full + two probes |
| library_note_folder_dialog.py | 148 | sampled (28-80, 100-148) + pattern greps |
| library_choice_strip.py | 134 | sampled (105-134) + pattern greps |
| library_note_work_pane.py | 63 | mechanical only (greps clean) |
| library_file_notes_events.py | 47 | mechanical only (greps clean) |
| library_prompt_work_pane.py | 46 | read in full (delegated deep-read) |
| library_emergency_return.py | 37 | mechanical only (greps clean) |
| library_character_return.py | 30 | mechanical only (greps clean) |

## Method and honesty note
40,112 lines is more than one context can hold, so the eight largest files (git panel, notes canvas, add-from-files, ingest, note-import, skills, prompts, search/RAG, rail, media viewer, collections reader, work panes, prompt-delete modal) were **read in full by delegated readers running under this same brief and the same rules of evidence**, each required to return `file:line` + a command and its output. I read the workspace, media canvas, canvas_sync, image preview, recovery dialog, `__init__` and the small files myself, and I own every mechanical sweep and the candidate table.

Probes I ran or re-ran personally: the media preview clobber, the File Notes poll pause, the recovery-dialog timer (both halves), the Textual 8 `Button` markup behaviour and `_visible_row_title`, the media row title at widget level, the config-read timing, the `ast` method census, the package import timing, the git-panel hot-path probe (re-run, numbers reproduced), the Prompts P0 Escape crash (re-run, reproduced). Probes I did **not** re-run — reported with their author's evidence quoted verbatim: the notes-canvas perf/width probes, the ingest template/focus/recompose-cost probes, the skills trust-header and work-pane-desync probes, the search/RAG bracket and detached-app probes. All probe scripts are in `<SCRATCH>/probe_*.py` and are re-runnable with the env.sh recipe.

Nothing in either checkout was modified.

## Findings
(ordered P0→P3, then D1→D4. Counts: P0 x1, P1 x11, P2 x17, P3 x16)

### P0 [D1] — Escape kills the app: the Prompts work pane's "More actions" flag survives a recompose into a branch that never composes the region it dereferences
- Where: `library_prompts_canvas.py:549-555 on_key` (`self.query_one("#library-prompt-more-actions-region")`, unguarded) and the same shape at `:527 _toggle_more_actions`; the flag is set at `:230`/`:525`, cleared only at `:512` and in `on_key`; `sync_state:323-357` recomposes without resetting it; the region is composed only by `_compose_editor`.
- Evidence: probe `<SCRATCH>/probe_skills_escape_import.py` (real keypress, real focus, production compose branch), **re-run by me**:
  `textual.css.query.NoMatches: No nodes match '#library-prompt-more-actions-region' on LibraryPromptWorkPane(id='library-prompt-work-pane')` raised from `library_prompts_canvas.py:554 in on_key` → `App._exception`.
  Production sequence (each step traced in the controller): prompt open in the work pane → press `#library-prompt-more-actions` (flag True) → press Import… on the Prompts list → `library_prompts_controller.py:1595-1604` sets `_library_prompts_import_open` and `_library_prompt_work_pane_kwargs` (`:985-987`) forces `mode="list"`, `import_open=True`, so only the import row composes and the flag is never in the kwargs (`grep -rn "more_actions_open" tldw_chatbook` → skills-side hits only) → the same handler focuses `#library-prompts-import-path` → **Escape** bubbles from that Input to `on_key`.
  Two neighbours in the same file (`_open_more_collections:534`, `_open_more_history:542`) already wrap their `query_one` in `try/except NoMatches`.
- Why it matters: an ordinary two-click sequence followed by Escape terminates the TUI — the defect class this repo already recorded in task-32639 ("a re-run flake was a real app-killing unguarded `query_one`").
- Recommended correction: root cause first — reset `self.more_actions_open = False` in `sync_state` before `refresh(recompose=True)` (this also fixes the quieter twin: `:1673` composes `more_region.display = self.more_actions_open`, so the menu silently reopens on the next prompt). Then guard `on_key`/`_toggle_more_actions` like their two neighbours. The skills sibling avoids the whole class by keeping the flag screen-owned (`library_screen.py:25651-25653`).
- Size: S · ADR: no · Confidence: verified (reproduced twice, independently)
- Pinning test: `Tests/UI/test_library_prompts_canvas.py:802 test_prompt_more_actions_is_inline_and_escape_restores_opener_focus` pins only the happy path (region mounted) — not a decision, a gap.
- Already covered: none

### P1 [D1] — Renaming a speaker in Library ▸ Media replaces the 3-line metadata preview pane with the entire transcript
- Where: `tldw_chatbook/Widgets/Library/library_media_canvas.py:688-696` (`_refresh_after_speaker_rename`); the pane it writes into is composed at `:1900-1904` from `canvas.preview_lines`, built at `tldw_chatbook/Library/library_media_state.py:1197-1205` and `:1573-1583` as exactly three metadata lines (title / `Type: …` / `Updated: …`).
- Evidence: probe `<SCRATCH>/probe_media_preview.py` (production-shaped `preview_lines`, real `MediaDatabase`, real canvas):
  `cd <wt> && source <SCRATCH>/env.sh && PYTHONPATH=<wt> $PY -m pytest <SCRATCH>/probe_media_preview.py -q -s`
  → BEFORE `'Meeting\nType: audio\nUpdated: today'`; AFTER `'[00:00:00] Alice: hello hello …\n[00:00:01] Speaker 2: world …'` — `AssertionError: PREVIEW PANE CLOBBERED BY FULL TRANSCRIPT`.
- Why it matters: after one inline rename the small "selected item" pane in the media list becomes the whole transcript (unbounded text in a fixed pane), and it never goes back until the next state push. The sibling reader (`library_media_viewer.py:870-899`) does the same job correctly, so this is the drifted copy.
- Recommended correction: `_refresh_after_speaker_rename` must not write `Media.content` into `#library-media-preview-lines`. The pane is metadata-only; the rename only needs the legend label patch (the second half of the method). If the canvas wants a content echo it has to ask the controller for a fresh `LibraryMediaCanvasState`.
- Size: S · ADR: no · Confidence: verified
- Pinning test: `Tests/UI/test_library_media_speaker_rename.py:121 test_speaker_legend_submit_renames_and_refreshes_preview` — it MASKS the bug: line 143 seeds `preview_lines=tuple(row["content"].splitlines())`, a shape no production builder ever produces, then asserts `"Alice" in preview`. It does not state the production behaviour as a requirement.
- Already covered: none

### P1 [D1] — Opening the File Notes "Review pairing" dialog silently stops the folder poll for the rest of the mount
- Where: `library_file_notes_workspace.py:6707-6709` (`_review_pairing` pauses `self._poll_timer` and no path resumes it); the only `resume()` in the 8,846-line file is `:8845`, inside `_refresh_pressed` (the manual Refresh button). `grep -n "_poll_timer" library_file_notes_workspace.py` → one `.pause()` (6709), one `.resume()` (8846), the rest are create/stop/None.
- Evidence: probe `<SCRATCH>/probe_poll_pause.py` (reuses the shipped `Tests/Backup_Recovery` subprocess harness, real replica + real workspace):
  `cd <wt> && source <SCRATCH>/env.sh && PYTHONPATH=<wt> $PY -m pytest <SCRATCH>/probe_poll_pause.py -q -s`
  → `POLL ACTIVE before review: True` / `POLL ACTIVE after dialog closed: False` / `POLL ACTIVE after pressing Refresh: True`.
- Why it matters: the workspace's background file monitoring (external edits, deletions, replica reconciliation) is dead from the moment the user opens the pairing review until they press Refresh or leave and re-enter the screen. Nothing tells them.
- Recommended correction: resume in a `finally:` in `_review_pairing` (guarded on `self._poll_timer is not None and self._active`), i.e. pair the pause with the scope that needed it.
- Size: S · ADR: no · Confidence: verified
- Pinning test: `Tests/Backup_Recovery/test_notes_recovery_controls.py` exercises exactly this flow and its `approve` branch presses Refresh at the end — which masks the paused timer rather than asserting it.
- Already covered: none

### P1 [D1] — The note editor's "where does this note live" row is elided to the LIST pane's width, not its own
- Where: `library_notes_canvas.py:1364 _effective_pane_width` (consumed at `:2760` compose and `:3650 _restate_note_location`); `:1387-1388` — `on_resize` returns before recording `_measured_width` for any mode but `"list"`. The work pane's `pane_width` is `reader_layout.items_width` (`library_notes_controller.py:3337-3338`), and `apply_pane_width` is only dispatched to `#library-notes-canvas` (`library_screen.py:6692`), never to `#library-note-work-pane`.
- Evidence: `<SCRATCH>/probe_notescanvas_live_width.py`, driving the repo's own harness from `Tests/UI/test_library_notes_w5_ideas.py` at its `WIDE = (235, 52)`:
  `work pane rendered width : 157` · `row rendered width : 156` · `wp.pane_width (contract) : 64` (the list pane) · `_effective_pane_width() : 64`.
  `RENDERED row: 'In a synced folder · …low-ups.md · file written 2026-09-15 01:06'` vs at the row's real width: `'In a synced folder · /var/folders/.../Obsidian/Vault/Project…atlas-follow-ups.md · file written …'`.
- Why it matters: the path is crushed past its own filename in a row 156 cells wide, and at a narrower split the `_NOTE_LOCATION_PATH_FLOOR` branch (`:265-267`) drops "file written …" entirely — a fact removed from a pane with room for it. Answering "where does this note live" is the row's whole purpose.
- Recommended correction: pass the row's own width (`self.size.width`, already 157) to `_restate_note_location`/`_compose_editor`, or let `on_resize` record `_measured_width` in every mode and stop preferring `pane_width` outside list mode.
- Size: S · ADR: no · Confidence: verified (measured against the shipped harness)
- Pinning test: `Tests/UI/test_library_notes_w5_ideas.py:300` asserts only `"Sam.md" in line` on a short tmp path; `:344` pins the pure function at hand-passed widths. Neither pins which width the editor supplies.
- Already covered: none

### P1 [D1] — The ingest chunking-template picker shows "None (manual settings)" while the form still submits a saved template
- Where: `library_ingest_canvas.py:1553-1560` (compose clamp) and `:2027-2036` (post-fetch restore).
- Evidence: `<SCRATCH>/probe_ingest_template_value.py` (canvas mounted with `form.type_options["generic"]["chunk_template"] = "big-words"` and a scope service returning it):
  `OPTIONS: ['', 'auto', 'big-words']` · `PICKER VALUE (what the user SEES): ''` · `FORM VALUE (what Start SUBMITS): 'big-words'` · `OptionValueChanged posted: []`.
  Mechanism: a fresh canvas has `_chunk_template_names = []`, so `available` at `:1556` lacks the saved name and `picker_value` is clamped to `""` — display only, the form is never corrected. `_fetch_chunk_templates` then reads `selected = picker.value` (`:2027`), now `""`, finds it in the new options and preserves it; the `else` branch at `:2033` pre-seeds `_reported_option_values` before `set_options`, so the resulting `Select.Changed` is swallowed as mount noise by `_handle_option_value_changed:2135`. Reachable on the shipped route: the canvas is rebuilt on every resolution to `ingest-media` (`library_screen.py:11898`, `:15585`) while `self._ingest_state.form` lives on the screen (`:3963`); `app.py:4275` reads `flat_opts["chunk_template"]`.
- Why it matters: the screen states one chunking policy and executes another, on the control whose entire job is to state it.
- Recommended correction: in `_fetch_chunk_templates`, prefer the form's stored value when it has become available — two lines in the existing preserve branch.
- Size: S · ADR: no · Confidence: verified
- Pinning test: `Tests/UI/test_library_ingest_template_picker.py` pins the default (`test_picker_default_is_none_manual_settings`) and the populate path, never a persisted value.
- Already covered: none

### P1 [D1] — A Search/RAG evidence row whose title carries a non-Rich-shaped bracket renders with that segment deleted
- Where: `library_search_rag_panel.py:1157-1165` (row title), `:1178-1182` (citation labels), `:1271-1276` (`results_count_line`), `:1318-1322` (empty-state copy) — all four `Static(...)` markup-ON; root at `Library/library_rag_state.py:388` (`_sanitize_display_text`'s terminal `escape_markup`), `:1138`, `:2062`.
- Evidence: `<SCRATCH>/probe_rail_title_markup2.py` / `probe_rail_widget_level.py`:
  `'[TODO] Q3 plan' -> render='1.  Q3 plan'` · `'[IMPORTANT]' -> render='1. '` · `'[ WIP ] thing' -> render='1.  thing'` · count line `"1 result for ' plan'."` for the query `[TODO] plan` · empty state `"No evidence matched ' plan'."` (the last two read from the widget's `.visual.plain`).
- Why it matters: same defect class as the row-title finding above, on the surface where the user is trying to identify which evidence matched.
- Recommended correction: same fix — `_escape_all_brackets` (`library_rag_state.py:520`) at these display sinks; the module's own comment already explains why `escape_markup` is not enough here.
- Size: M (shares the fix with the row-title P1) · ADR: no · Confidence: verified
- Pinning test: `Tests/Library/test_library_rag_state.py:591 test_result_row_display_snippet_bracketed_emphasis_stays_inert` pins the already-fixed snippet path; nothing pins titles/citations/count lines.
- Already covered: none

### P1 [D2] — 77% of the Session Git panel's arrow-key handler is a DOM existence probe guarding 0.03 ms of work
- Where: `library_file_notes_git_panel.py:1883` — `if any(not list(self.query(selector)) for selector in selectors): return` in `_sync_action_layout`; reached per keypress from `_row_highlighted` (`:3432` → `_update_actions`), three times per `set_mutating` (`:3038`), and on every resize (`:1862`).
- Evidence: `<SCRATCH>/probe_gitpanel_hot2.py` (panel mounted 120x40, 6 ready rows), **re-run by me**:
  `existence probe only (5x self.query): 1.404 ms` / `_visible_action_cells x5: 0.027 ms` / `_action_row_width x5: 0.004 ms` / `_sync_action_layout total: 1.431 ms`.
  First run (`probe_gitpanel_hot.py`): `_update_actions 1.841 ms`, of which `_sync_action_layout 1.452 ms`.
- Why it matters: `self.query(selector)` materialises a full-subtree `DOMQuery` five times to decide whether to do 0.03 ms of measuring; it runs on every Up/Down in the changed-files list.
- Recommended correction: delete the probe and wrap the `needs_stack` computation in `try: … except NoMatches: return` — `NoMatches` is already imported (`:18`) and already used this way at `:2782`.
- Size: S · ADR: no · Confidence: verified (measured twice, different processes)
- Pinning test: `Tests/UI/test_library_file_notes_git.py::test_action_controls_fit_from_visible_label_cells_and_recompute` (`:2323`) pins the RESULT, not the probe.
- Already covered: none

### P1 [D2] — Every Notes keystroke pays 462 µs for four whole-subtree `DOMQuery` truth-tests that `query_one` answers in 0.6 µs
- Where: `library_notes_canvas.py:3155, 3199, 3213-3215` (editor branch) and `:3122, 3129, 3132, 3139` (list branch) in `apply_compact_presentation`, called from `apply_session_state:3419`; same shape at `sync_state:1585, 1605`.
- Evidence: `<SCRATCH>/probe_notescanvas_perf.py` (real mounted editor canvas, 68 children, 35 KB body):
  `4x bool(query('#id')) 462.4 us` · `3x query_one('#id') 0.6 us` · `2x bool(query) in sync_state 230.2 us` · `apply_compact_presentation() 509.8 us` · `apply_session_state() 1002.3 us` · `len(BODY.split()) 80.3 us` (the scan this file's own design ruled out). Absent id: `bool(query) 119.7 us` vs `query_one 7.4 us`.
  Per-keystroke reachability traced: `library_notes_controller.py:3703/:3730` (`@on(Input.Changed, "#library-note-title")` / `@on(TextArea.Changed, "#library-note-body")`) → `_apply_library_note_presentation_state:1677` → `canvas.apply_session_state(...)`.
- Why it matters: 91% of `apply_compact_presentation`'s cost and 46% of every keystroke's, for existence tests — and `update_note_chrome_facts`' own docstring (`:3571-3579`) records the same anti-pattern being measured at 263.6 µs and fixed.
- Recommended correction: the sibling shape 40 lines over is right — `library_media_canvas.py:740` uses `try: self.query_one(...) except NoMatches: return`. Same swap for all seven sites plus the two in `sync_state`.
- Size: S · ADR: no · Confidence: verified (measured)
- Pinning test: none — `Tests/UI/test_library_honesty_accessibility.py:1272,1289` and `Tests/UI/test_library_multiselect_media.py:2930` drive `apply_compact_presentation` but assert labels, never lookup shape.
- Already covered: none

### P1 [D2] — The Notes editor re-renders the hidden Preview source on every keystroke, contradicting the comment two lines above it
- Where: `library_notes_canvas.py:3361` — `preview_source = render_preview_source(snapshot.body, title=snapshot.title)` computed before the `if show_preview and preview_body.source != preview_source:` gate; the comment at `:3347-3350` states the intent ("Keep the hidden Preview stale while typing … so edits cannot queue an unbounded hidden-render backlog").
- Evidence: `<SCRATCH>/probe_notescanvas_perf.py` → `render_preview_source(35KB) 177.9 us` (2.25× the 79.0 µs whole-body scan the design banned), `render_preview_source(350KB) 1859.5 us`. Three full-body passes (`_drop_leading_title_heading`'s `split("\n")`, `render_note_links`, `render_obsidian_callouts`).
- Why it matters: 1.86 ms per keystroke on a large note, in Edit mode, for output nobody can see.
- Recommended correction: move the call inside `if show_preview:`. One line.
- Size: S · ADR: no · Confidence: verified (measured)
- Pinning test: none — every test hit (`test_library_notes_w4_editor.py:433,440`, `w3_layout.py:195`, `w5_import_preview.py:85`) calls the pure function directly; none asserts it runs while hidden.
- Already covered: none

### P1 [D2] — The Skills trust header's `has_skills` predicate exists twice and the two copies disagree; the in-place path drops the recovery banner
- Where: `library_skills_canvas.py:1216-1219` (compose, includes the `trust_posture == "recovery_review"` disjunct) vs `:992-997` (`sync_state`'s `header_only` path, which does not).
- Evidence: `<SCRATCH>/probe_skills_trust_header_divergence.py`, identical inputs (`SkillsListState(rows=(), count=0, source_summary_fresh=False)`, posture `recovery_review`):
  `COMPOSE header: 1 action: 1` / `SYNC header: 0 action: 0` → `AssertionError: assert 0 == 1  # '#library-skills-trust-action'`.
- Why it matters: the `header_only` path exists precisely because "the posture read may settle after rows become interactive" (`:1213-1215`), so when the posture settles to `recovery_review` during a routine refresh the user loses the banner and the only list-level "Review restored skills" button until an unrelated input forces a full recompose.
- Recommended correction: extract the predicate beside `skill_trust_header_line` in `library_skills_state.py` and call it from both sites.
- Size: S · ADR: no · Confidence: verified (divergence); inferred (that this state pair occurs in a live session — settle by logging `(trust_posture, state.source_summary_fresh)` pairs in `_library_skills_canvas_kwargs` during a Skills refresh on a restored trust store)
- Pinning test: none (`grep -rn "recovery_review" Tests` → zero hits in the Skills canvas tests)
- Already covered: none

### P1 [D3] — Toggling one Notes-import review row drops keyboard focus entirely, and costs a full-page recompose measured at 120-175 ms
- Where: `library_note_import_canvas.py:597` (`sync_state` → `refresh(recompose=True)`) and `:664-667` (`_after_recompose`, no focus restore).
- Evidence: `<SCRATCH>/probe_ingest_review_focus.py` → `FOCUS BEFORE: note-import-action-item-6-skip` / `FOCUS AFTER: None` / `same-id button re-rendered: True`.
  Cost, `<SCRATCH>/probe_ingest_review_recompose_cost.py` (`app.run_test(size=(140,50))`, one item's action flipped per sample): 1 row/6 buttons → 69-94 ms; **25 rows/78 buttons → 119-174 ms** (25 is the real ceiling, `MAX_IMPORT_REVIEW_PAGE_SIZE`, `Library/library_note_import_state.py:39`); 50 rows → 250-314 ms. The ~70-90 ms floor is harness pump overhead, so the marginal row cost is ~50-80 ms per press at the ceiling.
  Shipped path: `ItemActionRequested` → `library_notes_controller.py:5534 set_item_action` → `_publish_library_snapshot` → `_sync_library_canvas(self, "notes")` (`:5036`) with **no** `then=`; the focus helper `_focus_library_note_import_control` (`:4971`) is wired at exactly one unrelated site (`:5071`).
- Why it matters: a keyboard user settling a 25-row review re-tabs from nowhere after every press, and waits >100 ms (the repo's own worker threshold) on the UI thread for it.
- Recommended correction: record `self.app.focused.id` before the recompose and re-focus it in `_after_recompose` (the canvas already carries `PostRecomposeCallback`; ids are stable — the probe shows the same-id button is re-rendered). Structurally: give the review rows their own render-from-state child, as the sibling ingest canvas already did (`LibraryIngestQueuePanel`, task-2042).
- Size: S (focus) / M (granularity) · ADR: no · Confidence: verified (harness-measured, not terminal-measured — stated)
- Pinning test: `Tests/UI/test_library_notes_w4_import_keyboard.py:321` goes review → Enter without ever toggling a row.
- Already covered: none

### P1 [D4a+D1] — Every Library list row title is escaped with `rich.markup.escape`, which does not cover `[TODO]`-shaped brackets; the repo's own full escaper is two packages away
- Where: the shared escaper `library_rail.py:367-377 _visible_row_title` = `escape_markup(_truncate_row_title(...))`, consumed by `library_media_canvas.py:204` (every media row), `library_conversations_canvas.py:317`, `library_media_trash_canvas.py:437`, and `Widgets/Home/home_rail.py:127,170`. Same narrow escaper at 27 further `escape_markup(...)` sites in `Widgets/Library/` that feed markup-ON surfaces — notably `library_notes_canvas.py:2132, 2324, 2373-2374, 2662` (note titles, folder labels), `library_skills_canvas.py:1284, 1318`, `library_prompts_canvas.py:697, 1007, 1008, 1023`, `library_entry_canvases.py:164`, `library_rail.py:1011, 1046`, `library_search_rag_panel.py:1387`.
- Evidence: `rich.markup.escape`'s pattern is `(\\*)(\[[a-z#/@][^[]*?])` — it only escapes a `[` followed by **lowercase**, `#`, `/` or `@`. Textual 8's tokenizer opens a tag on any unescaped `[`. Measured through the real helper:
  ```
  '[TODO] Q3 plan'  escaped='[TODO] Q3 plan'   rendered='▸  Q3 plan'
  '[IMPORTANT]'     escaped='[IMPORTANT]'      rendered='▸ '
  'Meeting [Call]'  escaped='Meeting [Call]'   rendered='▸ Meeting '
  '[ WIP ] thing'   escaped='[ WIP ] thing'    rendered='▸  thing'
  '[draft] notes'   escaped='\\[draft] notes'  rendered='▸ [draft] notes'   (lowercase IS covered)
  ```
  Widget level, real canvas (`<SCRATCH>/probe_media_title_markup.py`): `TITLE: '[TODO] Q3 plan'` → `RENDERED: '▸  Q3 plan\n    document · today'` → `AssertionError: row label lost the bracketed run`.
  The repo has already diagnosed this exact class and shipped the fix: `Library/library_rag_state.py:505-530` (`_escape_all_brackets`), whose own comment names the `[TODO]` shape — but it is applied only to `display_snippet` (`:1546`) and `library_rag_answer_display_text` (`:601`).
- Why it matters: `[TODO] …`, `[WIP] …`, `[2024-Q3] …` are ordinary titling conventions. A media item, conversation, note, prompt, skill or Home-rail row titled that way loses the bracketed run from its row, and a title that is only a bracketed word renders as a blank row — the user cannot see which item they are selecting.
- Recommended correction: promote `_escape_all_brackets` out of `Library/library_rag_state.py` into `tldw_chatbook/Utils/` and point `_visible_row_title` plus the 27 `escape_markup(...)` display sites at it. Canonical home: `Utils/` (a Library-private helper already has three packages of consumers). Leave `escape_markup` only where a Rich-tag-shaped escape is genuinely wanted.
- Size: M · ADR: no · Confidence: verified (helper-level and widget-level)
- Pinning test: none pins the broken shape. `Tests/Library/test_library_rag_state.py:2003 test_query_is_markup_escaped` and `:2032` only assert `[bold]…[/]` (tag-shaped, which `escape_markup` does cover); `Tests/UI/test_library_ingest_template_picker.py:143 test_picker_escapes_markup_in_labels` likewise asserts `"chapter [red] bold"`. Every existing test picks a lowercase tag name, so the gap is untested rather than decided.
- Already covered: none

### P2 [D1] — `NotesRecoveryDialog`'s 0.25 s interval never stops and its callback queries children unguarded
- Where: `Widgets/Library/notes_recovery_dialog.py:90` (`set_interval(0.25, self._check_current)`), `:92-101` (`_check_current` returns `False` intending to stop; `set_interval` ignores the return value), `:96-97` (`query_one` with no guard).
- Evidence: probe `<SCRATCH>/probe_recovery_timer.py` →
  mechanism: `MECHANISM: NoMatches No nodes match '#notes-recovery-approve' on NotesRecoveryDialog()` (children pruned, callback still queries);
  reachability: `RACE HITS: none in 40 close cycles` — Textual 8.2.8 stops the pump's timers in `MessagePump._close_messages` (`message_pump.py:533-535`) before the dismiss settles, so I could NOT reproduce a real crash. The remaining, certain defect is that once `current()` has gone False the interval keeps re-`update()`-ing `#notes-recovery-status` with the same string 4×/s for as long as the dialog stays open (a refresh per tick), because nothing stops the timer.
- Why it matters: a repaint every 250 ms behind a modal for an unbounded time, plus a query that is only safe by accident of Textual's teardown order.
- Recommended correction: keep the returned `Timer` and `.stop()` it in the two branches that return `False`; wrap the two `query_one` calls (or use `self.query(...)`) as every other timer body in this package does.
- Size: S · ADR: no · Confidence: verified (mechanism + no-stop), inferred (crash reachability — retired below)
- Pinning test: `Tests/Backup_Recovery/test_notes_recovery_controls.py` (`route`/`navigation` branches) asserts the "Selection changed" status appears; it does not assert the timer stops.
- Already covered: none

### P2 [D1+D4a] — The File Notes "Use <folder>" button deletes bracketed runs from the folder name, and its idempotence guard can then never match
- Where: `library_file_notes_workspace.py:2964` builds `label=f"Use {_folder_label(sync_folder)}"` (a real directory name) and `:2839-2841` does `if str(button.label) != label: button.label = label`. The file never imports `escape_markup`; the sibling canvases in the same package do (`library_entry_canvases.py:8,164,180`, `library_conversations_canvas.py:333`, and `library_rail.py:377 _visible_row_title` = `escape_markup(_truncate_row_title(...))`).
- Evidence: Textual 8.2.8 parses markup in Button labels — `Content.from_text` calls `Content.from_markup` (`textual/content.py:258`):
  `$PY -c` probe → `Content.from_text('Use [draft] notes').plain` = `'Use  notes'`, `Content('Use [draft] notes').plain` = `'Use [draft] notes'`; `Button('[archive]')` → `''`; `Button('x [/] y')` → **raises `textual.markup.MarkupError: auto closing tag ('[/]') has nothing to close`**.
  Faithful micro-repro of the two lines above (`<SCRATCH>` one-liner): `pass 0: label attr='Use [archive]' rendered='Use ' needs_write_next_time=True` … repeated for passes 1 and 2.
- Why it matters: a notes folder named `[archive]`, `[wip]` or `Notes [old]` is offered as "Use " with the name gone; and because the read-back never equals the written string, every `_update_root_surface()` re-assigns the reactive label (a refresh per call, including on the 3 s structural-wait tick).
- Recommended correction: `label=f"Use {escape_markup(_folder_label(sync_folder))}"` — the helper is already the package convention (`rich.markup.escape`, used 4 files over). The general fix for the class is to pass `Content(...)` instead of `str` to `Button`, but the one-line escape matches what this package already does.
- Size: S · ADR: no · Confidence: verified (mechanism + micro-repro); the widget-level manifestation follows from the two quoted lines
- Pinning test: none found (`grep -rn "Use {" Tests` and `rg file-notes-use-sync-folder Tests` → only presence/visibility assertions)
- Already covered: none

### P2 [D1] — Two `run_worker` calls in the git panel keep `exit_on_error=True` with an unguarded `query_one` as the coroutine's first statement
- Where: `library_file_notes_git_panel.py:2847-2866` (`_render_commit_review_notes`, first line queries `#file-notes-git-commit-included-notes`) and `:3215-3247` (`_render_rows`, queries `#file-notes-git-rows` OUTSIDE its `try`).
- Evidence: `run_worker`'s default is `exit_on_error: bool = True` (`textual/dom.py`); `grep -rn "exit_on_error=False" tldw_chatbook | wc -l` → **268** sites opt out repo-wide, these two do not. `_render_rows` is incidentally protected by a synchronous `query_one` at `:3242` before scheduling; `_replace_commit_review_notes` has no such pre-check.
- Why it matters: an uncaught `NoMatches` in a Textual worker with `exit_on_error=True` terminates the application, and this panel is explicitly retained across parent remounts (`library_file_notes_workspace.py:1422-1425`).
- Recommended correction: `exit_on_error=False` on both plus `if not self.is_mounted: return` at the top of each coroutine.
- Size: S · ADR: no · Confidence: inferred (the detach window's reachability was not reproduced; the defaults and the missing guard are verified)
- Pinning test: none
- Already covered: none

### P2 [D1] — The git panel's public render API raises `NoMatches` when detached, and the guarding convention is split between callee and caller
- Where: unguarded — `:1931 _fit_fixed_regions`, `:3073/:3086 set_last_action/clear_last_action`, `:3136/:3215 _clear_rows/_replace_rows`, `:3508 _settle_action_focus` (reads `self.screen`); guarded — `:1979` (`is_mounted`), `:2017` (`is_attached`). Callers guard at only 4 of ~25 sites (`library_file_notes_workspace.py:3623, 3692, 3715, 4125`; `:3559 mark_stale` is bare).
- Evidence: `<SCRATCH>/probe_gitpanel_unmounted.py` against an unmounted instance → `set_current_status: NoMatches`, `set_last_action: NoMatches`, `mark_stale: NoMatches`, `render_unavailable: NoMatches`, `render_untrusted: NoMatches`, `return_to_commit_list: NoMatches`, `return_to_push_list: NoMatches`; `clear_commit_availability: OK`, `clear_push_availability: OK`.
- Why it matters: two mounting predicates and two conventions across one API mean the safe set is undiscoverable; a late service callback landing during a Library recompose raises.
- Recommended correction: one rule — every public `render_*`/`set_*`/`clear_*`/`return_to_*` entry starts with `if not self.is_mounted: return`; then drop the four caller-side checks.
- Size: S · ADR: no · Confidence: verified (the raises), inferred (the live detach window)
- Pinning test: none
- Already covered: none

### P2 [D1] — The push-destination authorization dialog's Confirm bypasses the file's own double-dismiss guard
- Where: `library_file_notes_git_panel.py:4284-4286` uses `self.dismiss(True)` while every other exit in the file uses `dismiss_safe_once` (`:4180`, `:4271`, `:4275`).
- Evidence: `Widgets/modal_dismissal.py:270-282` — `dismiss_safe_once` refuses a second dismissal (`_safe_dismiss_committed`) and checks `host.app.screen is self`; Textual 8's `Screen.dismiss` has no idempotency guard (fires `_result_callbacks[-1](result)` then `pop_screen()` unconditionally). The authorize path therefore never sets `_safe_dismiss_committed` and skips the opener-focus restore at `:291-298`.
- Why it matters: this is the consent gate for first contact with a push destination; two `Button.Pressed` messages queued before the pop completes fire the authorization callback twice and pop the screen underneath.
- Recommended correction: `self.dismiss_safe_once(True)`.
- Size: S · ADR: no · Confidence: verified (the bypass), inferred (the double-press race)
- Pinning test: `Tests/UI/test_library_modal_dismissal.py::test_concrete_library_modal_public_positive_result_type` clicks ONCE and asserts one result — it pins the single-press result, not the guard.
- Already covered: none

### P2 [D1] — `_sync_disabled_action_presentation` reconstructs every button's base label from the rendered one
- Where: `library_file_notes_git_panel.py:3419-3430` — `label = str(button.label)`, `base = label.removeprefix(prefix)`, `if label != rendered_label: button.label = rendered_label`, looped over `self.query(Button)` (all 24 buttons).
- Evidence: `<SCRATCH>/probe_gitpanel_label.py` (Textual 8.2.8) → `before: Content('Ref  here', spans=[Span(4, 9, style='main')])` / `after: Content('○ Ref  here')` — the span is gone after the round trip. (Consistent with my own measurement that `Button.label` markup-parses a `str`: `Content.from_text` → `Content.from_markup`, `textual/content.py:258`.)
- Why it matters: the first styled label anyone gives a Session Git button is silently stripped on the next disabled-state sync, and the loop is not opt-in.
- Recommended correction: keep the plain base label on the button (attribute or id-keyed dict) — the same "stash the raw remainder" pattern the media/conversations canvases already use (`library_media_canvas.py:1872 button._library_row_label_rest`).
- Size: M · ADR: no · Confidence: verified (loss measured); latent today — no current label in the file contains `[`
- Pinning test: marker behaviour pinned at `Tests/UI/test_library_file_notes_workspace.py:8031, :8052`; the lossiness is not pinned.
- Already covered: none

### P2 [D1] — The Notes location row is overwritten with a factually false "In the Library database only — no file on disk" before the first `apply_session_state`
- Where: `library_notes_canvas.py:1047` (`_note_location = ("", "")`), `:3649-3652`, fired by `@on(Resize) _note_chrome_follows_width:3667`. Compose renders the row from `presentation_state.location_path` (`:2757`) while `_restate_note_location` reads the `_note_location` cache that only `apply_session_state` fills — two sources for one row.
- Evidence: `<SCRATCH>/probe_notescanvas_width2.py` → `A) _note_location right after mount: ('', '')` / `row text after mount+layout: 'In the Library database only — no file on disk'`; `B) after the controller's first state apply: row text: 'In a synced folder · /…/atlas-follow-ups.md'`.
- Why it matters: the row asserts the note has no file on disk when it does.
- Recommended correction: seed the cache in `_compose_editor` next to `:2756`. One line.
- Size: S · ADR: no · Confidence: verified (mechanism); inferred (duration in production — the open-a-note path recomposes and closes the window; the exposed path is the screen's direct construction at `library_screen.py:14970` with `mode="editor"`). Settling command: `$PY -m pytest Tests/UI/test_library_notes_w5_ideas.py -q` with `await pilot.resize_terminal(...)` inserted between shell build and `_open_first_note`.
- Pinning test: none · Already covered: none

### P2 [D2] — Every File Notes control repaint costs ~9.5 ms of config reads while no folder is linked
- Where: `library_file_notes_workspace.py:2861-2888 _configured_sync_folder` (two `get_cli_setting` calls in a `for` loop) called at `:2960` from `_update_root_surface`, which has 15 call sites plus a repeating `set_interval(STRUCTURAL_WAIT_PATIENCE_SECONDS=3.0, self._update_root_surface)` armed for every folder change (`:6909-6915`).
- Evidence: measured in the isolated env —
  `cd <wt> && source <SCRATCH>/env.sh && PYTHONPATH=<wt> $PY -c "<50-iteration timing loop over the exact two get_cli_setting calls>"` → `warm _configured_sync_folder config-read pair: median 9.45 ms, max 14.85 ms`.
  `_initialize` alone calls `_update_root_surface` twice (`:2506`, `:2510`) on top of the `on_mount` call at `:2364` → ~28 ms of config reads before the unlinked mode's first paint.
- Why it matters: the whole cost lands exactly in the state where the user is choosing a folder (the guard `None if self._root is not None` keeps it off the linked paths), and it repeats on a 3 s tick for the length of a folder change.
- Recommended correction: resolve the configured folder once per root change / per mount into an attribute and invalidate it where `_root` changes; nothing in `_update_root_surface` needs a fresh config read per repaint.
- Size: S · ADR: no · Confidence: verified (measured)
- Pinning test: none
- Already covered: none

### P2 [D2] — `_fetch_chunk_templates` is a coroutine worker, so its unbounded `SELECT *` runs on the event loop
- Where: `library_ingest_canvas.py:1960-1964` (`run_worker(self._fetch_chunk_templates(), group=…, exclusive=True)`), awaiting `list_templates(mode="local")` at `:1990`; fires from `on_show`, i.e. every entry to the Import rail path.
- Evidence: `<SCRATCH>/probe_ingest_worker_thread.py` → `EVENT LOOP THREAD: MainThread` / `DB READ RAN ON THREAD: MainThread` / `SAME THREAD: True`. Callee chain is synchronous SQLite: `RAG_Admin/rag_admin_scope_service.py:209` → the plain `def` at `RAG_Admin/local_rag_admin_service.py:221` (called before `_maybe_await` sees it) → `Chunking/chunking_interop_library.py:118-127` `conn.execute("SELECT * FROM ChunkingTemplates WHERE deleted = 0")` — whole table, no LIMIT.
- Why it matters: the one worker on this canvas gives no isolation at all, and the block scales with the template table.
- Recommended correction: hop the sync call (`asyncio.to_thread` inside the service) or make the method sync and run it with `thread=True`, marshalling back via `call_from_thread` — the shape `library_media_canvas.py:651` already uses.
- Size: S · ADR: no · Confidence: verified (thread identity); the block DURATION is unmeasured — settle by timing `get_all_templates` against a real `MediaDatabase` with N templates
- Pinning test: `Tests/UI/test_library_ingest_template_picker.py:159` pins only that the populate is off the mount path, not which thread it lands on.
- Already covered: none

### P2 [D2] — The re-chunk thread worker keeps `exit_on_error=True` and does work outside its `try`
- Where: `library_search_rag_panel.py:283` (`@work(thread=True, group=RECHUNK_WORKER_GROUP, exclusive=False)`), body `:309`, `:337`, `:343`, `:344-348`.
- Evidence: `textual._work_decorator` default is `exit_on_error: bool = True`, and `Worker._run` does `if self.exit_on_error: app._handle_exception(WorkerFailed(...))`. `<SCRATCH>/probe_rail_detached_app.py` → `thread .app while MOUNTED: A(...)` / `after remove: _parent = None` / `thread .app after REMOVE: RAISED NoActiveAppError`. The panel is not a resident canvas (`library_canvas_sync.py:424-428` lists only media/notes; `library_screen.py:11892` rebuilds it per route and the host `remove_children`s the old one), so navigating off Search/RAG mid-run detaches it. `format_rechunk_summary(summary)` at `:344` plus the two `call_from_thread`s at `:345-348` sit outside every `try`.
- Why it matters: today a mid-run navigation silently loses the summary line and the toast; a non-dict return from `rechunk_legacy_media` on a still-mounted panel is an uncaught worker exception, i.e. `app._handle_exception`. The identical shape one file over passes `exit_on_error=False` (`library_media_viewer.py:838`).
- Recommended correction: `exit_on_error=False` on the decorator and move `:344-348` inside the `try`/`else`.
- Size: S · ADR: no · Confidence: verified (mechanism + detach behaviour); inferred (the app-exit consequence)
- Pinning test: none (`grep -rn "exit_on_error" Tests/` → no hit on this worker)
- Already covered: none

### P2 [D2] — The Notes-import review's in-place fast path runs on snapshots that have no collision, throws, and full-recomposes anyway
- Where: `library_note_import_canvas.py:589-596` (`collision_only` omits `collision_kind`) and `:620-657` (`_sync_collision_controls`, `except Exception: refresh(recompose=True)`).
- Evidence: `<SCRATCH>/probe_ingest_sync_state.py::test_B` (review snapshot, `collision_kind=""`, same items/page, changed status line) → `recompose calls taken via the COLLISION except path: 1`. `_sync_collision_controls`'s first statement queries `#note-import-collision-heading`, composed only when `state.collision_kind` is truthy (`:894`).
- Why it matters: the outcome matches today, but `except Exception` around 30 lines of `query_one` means a renamed id or field silently degrades the path whose whole purpose (per `sync_state`'s docstring) is not replacing an actively edited input.
- Recommended correction: add `and bool(snapshot.collision_kind)` to `collision_only` at `:589`; narrow both `except Exception` to `except (NoMatches, QueryError)`.
- Size: S · ADR: no · Confidence: verified · Pinning test: none · Already covered: none

### P2 [D2] — A swallowed in-place patch permanently desyncs the Prompts/Skills work panes, because their change-detector reads the attribute the failed patch already wrote
- Where: `library_prompt_work_pane.py:42-46` and `library_skill_work_pane.py:176-182` (`all(getattr(self, key, object()) == value …)` → early return) against patchers that assign first and query second: `library_prompts_canvas.py:1746 sync_memberships`, `:476-482 sync_lifecycle_actions`, `library_skills_canvas.py:1086-1097`. Every caller swallows `NoMatches`/`QueryError` (`library_prompts_controller.py:1152-1157`, `:2879-2889`, `library_skills_controller.py:2349-2362`).
- Evidence: `<SCRATCH>/probe_skills_workpane_desync.py` → `summary widgets at compose: 0` / `patch raised (caller swallows this): NoMatches` / `pane.membership_state now: True` / `summary widgets after sync_state: 0` → `AssertionError: assert 0 == 1`. The next snapshot sees "unchanged" and skips the recompose; the Collections group never renders.
- Why it matters: the guard compares against INTENDED state, not RENDERED state, so any failed patch is unrecoverable instead of self-healing on the next snapshot. This is the pattern that manufactures the P1s above.
- Recommended correction: assign the attributes only after the `query_one` calls succeed (or keep a `_rendered_*` snapshot for the guard to compare against).
- Size: M · ADR: no · Confidence: verified (mechanism); inferred (that the membership variant fires in a live session)
- Pinning test: none · Already covered: none

### P2 [D3] — `LibraryFileNotesWorkspace` is a 8,846-line widget whose git half (94 methods, ~3.0k lines) duplicates the responsibility of the 4,291-line git panel beside it
- Where: `library_file_notes_workspace.py` (one class, `LibraryFileNotesWorkspace`, 280 body nodes / 305 defs) + `library_file_notes_git_panel.py` (4,291).
- Evidence: `ast` bucket of the class's own methods (command in `<SCRATCH>`, run under env.sh):
  `class LibraryFileNotesWorkspace: 280 body nodes, 7246 lines of methods (file 8846)` —
  `git 46/1428`, `push 30/927`, `commit 18/630`, `root 18/563`, `path 21/400`, `editor 12/311`, `reader 5/276`, `save 12/171`, `conflict 9/153`, `tree 7/127`, `folder 5/87`, `search 5/66`, `other 80/1975`.
  Responsibilities in one widget: service/replica lifecycle + root change under a deadline, the folder tree navigator with paging, the editor + autosave state machine, conflict resolution, search, git status/stage/commit/push review flows, the structural-wait patience surface, recovery pairing, and responsive layout.
- Why it matters: 41% of the method lines are the git workflow that already has its own 4.3k-line widget; every File Notes change pays the cost of reading a file where those concerns interleave.
- Recommended correction: the per-subsystem PR series in `backlog/docs/library-decomposition-recipe.md` §1 with the field-ownership script (§2); the seam is the `_git_*`/`_commit_*`/`_push_*`/`_stage_*` method set moving next to `library_file_notes_git_panel.py`. §17's size governance is the guard afterwards. Do not redesign — follow the recipe.
- Size: L · ADR: no (recipe is the settled shape) · Confidence: verified (counts)
- Pinning test: none (no size ratchet covers `Widgets/`; `scripts/` has no widget census — `ls scripts` shows only the four preflight checks)
- Already covered: none for this file (task-1378 / task-31202 are `settings_screen.py`)

### P2 [D3] — A policy-denied profile loses the legacy-chunk report AND the Re-chunk control, with no log line and no notice
- Where: `library_search_rag_panel.py:142-145` — `try: payload = await get_diagnostics(mode="local") / except Exception: return`.
- Evidence: `RAG_Admin/rag_admin_scope_service.py:312` calls `self._enforce_policy(self._admin_action_id(mode, "observe"))`, which raises `PolicyDeniedError` (`runtime_policy/types.py:116`); a shipped caller reaches this on every `on_show`. The sibling worker in the same file (`:328-334`) catches `PolicyDeniedError` explicitly and notifies the user. `_apply_legacy_chunk_report:177` drives `button.display` off the report, so a denial also hides the Re-chunk control.
- Why it matters: "nothing to re-chunk" and "you are not allowed to look" paint identically, and nothing is logged to tell them apart.
- Recommended correction: `except PolicyDeniedError: logger.debug(...)` and log the broad catch too. No UI change needed.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none · Already covered: none

### P2 [D3] — A canvas `sync_state` silently collapses every ingest queue outcome group the user opened
- Where: `library_ingest_canvas.py:514` (`self.expanded_groups: set[str] = set()`) and `:1288` (`sync_state` → `refresh(recompose=True)`).
- Evidence: `<SCRATCH>/probe_ingest_sync_state.py::test_A` → `BEFORE panel id: … expanded: {'failed:boom'}` / `AFTER panel id: … expanded: set()` / `SAME OBJECT: False`. `expanded_groups` is panel-local (writers: `library_ingest_controller.py:2841-2844`, `:2901`); nothing seeds it from render state, while the sibling disclosure flag `tooling_detail_expanded` DOES round-trip through state (`library_ingest_state.py:1364`, `:3442`). Reachable from any checkbox/select change (`library_screen.py:27896`) and the backend switch (`:27717`).
- Why it matters: "the expansion must survive" is already a pinned requirement one path over (`Tests/UI/test_library_crit9_import.py:480 test_dismissing_the_leading_member_keeps_the_group_expanded`); this is the same requirement failing at the canvas-recompose boundary.
- Recommended correction: seed from state the way `LibraryIngestPreflightSummary.__init__:311` already does — add an `expanded_queue_groups` field to the form/state and read it in `LibraryIngestQueuePanel.__init__`.
- Size: M · ADR: no · Confidence: verified
- Pinning test: the crit9 test pins the sibling path only. · Already covered: none

### P2 [D4b] — The Media speaker-rename flow exists twice and the canvas copy is missing all four hardenings the reader copy documents
- Where: `library_media_canvas.py:629-706` vs `library_media_viewer.py:806-899` (same four methods, same names, same message flow).
- Evidence: `sed -n '629,706p' library_media_canvas.py` / `sed -n '806,899p' library_media_viewer.py`. Drift, reader→canvas:
  1. reader captures `media_id` at submit time ("a selection change mid-rename must not retarget the write"); the canvas reads `self.speaker_rename_media_id` *inside the thread worker* (:659) and again in the repaint (:691) — a selection change between submit and worker start retargets the write to another media item;
  2. reader passes `exclusive=True` ("keeps two fast submits from piling up"); the canvas (:651-656) does not;
  3. reader does both post-rename DB reads on the worker thread; the canvas does `get_media_by_id` + `_meeting_speaker_legend_rows` on the UI thread inside `call_from_thread` (:691, :699) — two sqlite reads, one of them the whole content blob, on the event loop;
  4. reader maps `outcome.reason` through `_RENAME_REFUSAL_COPY`; the canvas shows the raw reason string (:684).
- Why it matters: the same user gesture behaves differently in the list and in the reader, and #1 can write a rename onto the wrong item.
- Recommended correction: one shared `_submit_speaker_rename(...)` helper next to `Library/meeting_speaker_rename.py` (which already owns the persistence half) taking `(db, media_id, cluster_id, name)` and returning `(outcome, content, rows)`; both widgets keep only their own repaint. Canonical home: `tldw_chatbook/Library/meeting_speaker_rename.py`.
- Size: M · ADR: no · Confidence: verified (drift), inferred (the #1 race window)
- Pinning test: `Tests/UI/test_library_media_speaker_rename.py` and `Tests/UI/test_library_media_viewer_speaker_rename.py` — two parallel suites, neither cross-checks the other's behaviour.
- Already covered: none

### P2 [D4b] — `_CHOICE_LABELS` is duplicated byte-for-byte across the canvas and its state module, and the two copies are compared against each other
- Where: `library_notes_add_from_files_canvas.py:63` and `Library/library_notes_lasting_sync_state.py:144`.
- Evidence: both dicts printed side by side → `IDENTICAL: True`. The coupling is live: the state module builds `row.selected_label` from ITS copy (`:278`, `:863`) while the canvas decides the "✓" tick and `is-selected` class from ITS copy (`library_notes_add_from_files_canvas.py:756`, `:1316`).
- Why it matters: a rename in one file silently stops the selected choice ever ticking.
- Recommended correction: export the state module's map (it already owns the enum) and delete the canvas copy; key `_CHOICE_SLUGS`/`_CHOICE_EFFECTS` off it too. Canonical home: `Library/library_notes_lasting_sync_state.py`.
- Size: S · ADR: no · Confidence: verified · Pinning test: none · Already covered: none

### P3 [D1] — The one seam every Library canvas' post-recompose follow-up runs through swallows its exception into a message-less DEBUG line
- Where: `library_canvas_sync.py:212-214` — `except Exception: logger.debug("Library post-recompose callback failed")` (no error text, no `exc_info`).
- Evidence: read; the same file's own docstring at `:86-90` names the blind spot ("`recompose` logs a failed callback at DEBUG only, so the intent would vanish without a trace") and works around it in `queue_default_after_recompose` with a `finally`.
- Why it matters: focus restores, scroll-offset restores and "land on Undo" receipts for every canvas in this package die here with nothing in the log that says why.
- Recommended correction: `logger.debug("Library post-recompose callback failed: {}", error)` at minimum (one-line change); WARNING is defensible since every caller is a UI intent the user asked for.
- Size: S · ADR: no · Confidence: verified (read)
- Pinning test: none
- Already covered: none

### P3 [D2] — Importing any one Library widget executes the whole package: 27 sibling modules for a 134-line leaf
- Where: `Widgets/Library/__init__.py` — 22 eager `from .x import ...` lines covering every widget in the package including the 8,846- and 4,291-line files.
- Evidence: `$PY -X importtime -c "import tldw_chatbook.Widgets.Library.library_choice_strip"` → `tldw_chatbook.Widgets.Library` cumulative **492 ms**, the requested leaf's own self-time **8 µs**; summed SELF time of the 28 `Widgets.Library` modules is **17 ms** (the rest is transitive deps the app may load anyway — I did not diff the closure, so 492 ms is NOT attributable to this `__init__`).
- Why it matters: 17 ms and 27 module objects are paid to reach any single widget; the honest cost is small, which is why this is P3 and not higher.
- Recommended correction: leave it unless a boot-budget snapshot shows the Library route on the critical path — the screen registry is already lazy (known-deliberate). Listed so the next person does not re-measure it.
- Size: S · ADR: no · Confidence: verified (measured, attribution stated)
- Pinning test: none
- Already covered: none

<!-- The findings below come from the deep read of library_file_notes_git_panel.py (4291 lines,
     read in full by a delegated reader under the same evidence rules); I re-ran the headline
     timing probe myself (numbers reproduced, noted inline). -->

### P3 [D2] — `_review_row_summary` is computed twice per Notes-import review row
- Where: `library_note_import_canvas.py:1107` and `:1112` (same call for the `Static` text and its tooltip) — 50 invocations per recompose at the 25-row ceiling, on the recompose path measured above.
- Recommended correction: one local. Size: S · Confidence: verified (read)

### P3 [D3] — Three copy-pasted deferred-focus helpers in the git panel, one of which lost its attachment guard
- Where: `:2340-2358 _focus_push_control_if_current` (has `if not self.is_attached: return`) vs `:2831-2845 _focus_commit_control_if_current` (no attachment guard, then `query_one`) vs `:3508-3513 _settle_action_focus` (no guard, reads `self.screen`). All three are entered from `call_after_refresh` (`:2334`, `:2827`, `:3396`).
- Evidence: the three bodies are otherwise identical (same `ancestors_with_self`/`display` walk, same `control.focus()`).
- Why it matters: `self.screen` raises `NoScreen` and `query_one` raises `NoMatches` on a detached widget; the guarded twin proves the case was considered for the push half only.
- Recommended correction: one `_focus_if_current(guard, selector)` carrying the `is_attached` check once.
- Size: S · ADR: no · Confidence: verified (divergence), inferred (that it fires)
- Pinning test: none · Already covered: none

### P3 [D3] — `_sync_commit_footer_layout` sizes the commit footer from hardcoded label literals that omit one of the labels it sizes for
- Where: `library_file_notes_git_panel.py:1892-1898` — `required = sum(cell_len(label) + 2 for label in ("Edit message", "Cancel commit", "Confirm commit"))`; its twin `_sync_push_footer_layout` (`:1900-1917`) measures the live buttons instead.
- Evidence: the `form` phase shows `#file-notes-git-commit-cancel` + `#file-notes-git-commit-review` (`_show_commit_phase`, `:2722-2726`) — "Review commit" is not in the tuple, and two of the three literals are not displayed in that phase but are charged to the width. The `+ 2` also hardcodes the CSS `padding: 0 1` instead of reading `button.styles.padding.width`.
- Recommended correction: delete the literal tuple; run the push version's body against `#file-notes-git-commit-footer`.
- Size: S · ADR: no · Confidence: verified
- Pinning test: `Tests/UI/test_library_file_notes_git.py::test_commit_footer_keeps_disclosure_edit_cancel_confirm_order_and_geometry` (`:1695`) · Already covered: none

### P3 [D3] — Two copies of "flatten + ellipsize an untrusted artifact name" in the same delete flow, with different limits
- Where: `library_prompts_canvas.py:88-93 _compact_receipt_name(value, limit=42)` and `prompt_delete_confirmation_modal.py:216-221 _display_name` with `_DISPLAY_NAME_LIMIT = 48`.
- Evidence: `grep -rn '" ".join(.*splitlines()' tldw_chatbook` → exactly these two (plus an unrelated `console_agent_bridge.py:1247`). They disagree on the cap (42/48) and on the empty case (`"Untitled"` / `""`), and both sit on the same journey — the modal previews the name, the list receipt (`:645-655`) echoes it seconds later.
- Recommended correction: one helper; canonical home `Library/library_prompts_state.py` (so the canvas need not import the modal).
- Size: S · Confidence: verified · Pinning test: `Tests/UI/test_prompt_delete_confirmation_modal.py` covers the modal copy only.

### P3 [D3] — `_compose_pager` is copied between the two near-sibling canvases and has already drifted three ways
- Where: `library_skills_canvas.py:1324-1379` vs `library_prompts_canvas.py:1053-1111`.
- Evidence: `diff -u` of the two ranges — both call the shared `library_pager_layout(pager)` and then diverge: skills yields the status line only if non-empty, prompts always yields it (spending a row in the 24-line case task-32354 was about); prompts additionally disables Previous/Next on `mutation_in_flight`, skills does not; chrome classes differ entirely (`library-source-pager*` vs `ds-toolbar`/`destination-purpose`).
- Recommended correction: one `compose_library_pager(pager, *, id_prefix, mutation_in_flight)` next to `library_pager_layout` in `Library/library_pager_state.py` — the rule already lives there, only the rendering was copied.
- Size: M · Confidence: verified (divergence); inferred (that the always-rendered empty Static costs a visible row)

### P3 [D3] — Near-miss id pair between two simultaneously mounted Skills widgets
- Where: `library_skills_canvas.py:1226` mounts `#library-skills-trust-region` (a `LibrarySkillsTrustHeader`) while `library_skill_work_pane.py:113` mounts `#library-skill-trust-region` (a plain `Vertical`) — one character apart, both live at once.
- Evidence: ids read directly; `sync_state:989` queries the first inside `except (NoMatches, QueryError): pass`, and `WrongType` IS a `QueryError`, so a typo degrades to a silent no-op.
- Recommended correction: rename one (`#library-skill-work-trust`). Size: S · Confidence: verified

### P3 [D3] — The three `str(button.label) != …` guards in the Notes canvas are redundant
- Where: `library_notes_canvas.py:3217`, `:3425`, `:3430` (three sibling sites at `:3134`, `:3151`, `:3201` already assign unconditionally).
- Evidence: `refresh() after IDENTICAL label assignment: 0 []` / `refresh() after REAL label change: 1 [{'repaint': True, …}]` — Textual's reactive already skips equal assignments, and `Content.__eq__` is exact.
- Recommended correction: drop the read-back; assign directly. Size: S · Confidence: verified

### P3 [D3] — Two spellings of the same `self.app.size.width` guard in the Notes canvas
- Where: `library_notes_canvas.py:1223-1226` catches bare `Exception`; `:3597-3605` catches only `NoActiveAppError` for the identical read, and its comment notes the divergence. Narrow the first. Size: S · Confidence: verified

### P3 [D4b] — Four Library canvases each re-roll the same two-line `sync_state`; a `reactive(recompose=True)` replaces all four
- Where: `library_export_canvas.py:101`, `library_search_rag_panel.py:82`, `library_conversations_canvas.py:51`, `library_notes_sync_roots_canvas.py:255` (also `library_media_canvas.py:596-625`, the same shape with more fields).
- Evidence: `sed` of all four → each is `self.<field> = <arg>` + `self.refresh(recompose=True)`.
- Why it matters: it is the smallest possible duplication, but it is also the single place a canvas could stop doing a whole-widget recompose for a one-field change, and today there is nowhere to make that decision once.
- Recommended correction: declare the state field as `reactive(..., recompose=True)` on each canvas and delete the method, or put the assignment+recompose in `PostRecomposeCallback` (the mixin every canvas already inherits) as `sync_state(self, state)`. Canonical home: `library_canvas_sync.py`.
- Size: S · ADR: no · Confidence: verified
- Pinning test: `Tests/UI/test_library_shell.py` calls `sync_state` directly in several places (a rename would have to follow).
- Already covered: adjacent to task-32089 (canvas_syncs shared dispatchers)

### P3 [D4b] — `_repository_path_for_display` is a fourth, weaker copy of the control-character display sanitizer
- Where: `library_file_notes_git_panel.py:274-298` vs `Notes/file_notes_session_owner.py:134 _sanitize_display_path` (a module this file ALREADY imports from, `:53`), `Workspaces/file_inspector.py:249`, `Notes/file_notes_git_service.py:1816`.
- Evidence: `grep -rn "0xDC80\|0xdc80" tldw_chatbook` → those four files. The panel's copy escapes only surrogates + `unicodedata.category in {"Cc","Cf","Cs"}` (`:285`); the session-owner copy also escapes U+2028/U+2029 (`Zl`/`Zp`) and `file_inspector.py` also escapes a bidi set.
- Why it matters: a repository path or git `disabled_reason` containing U+2028 renders unescaped here and escaped everywhere else, from the same source data.
- Recommended correction: export `_sanitize_display_path` from `file_notes_session_owner` and call it; keep only the `markup=` wrapper here. Canonical home: `Notes/file_notes_session_owner.py` (or `Utils/` if a third consumer appears).
- Size: S · ADR: no · Confidence: verified
- Pinning test: `Tests/UI/test_library_file_notes_git.py::test_trust_dialog_escapes_repository_path_controls_and_markup` (`:2283`) — check whether it asserts the weaker set before changing behaviour.
- Already covered: none

### P3 [D4b] — A third cell-aware middle-elide implementation, and `Utils/` still has none
- Where: `library_file_notes_git_panel.py:319 _grapheme_spans` + `:345 _middle_elide_cells`; the other cell-budgeted copies are `Chat/console_prompt_queue.py:127 _truncate_cells` and `Widgets/Console/console_composer_bar.py:2530`.
- Evidence: `grep -rn "def .*elide\|def .*truncate" tldw_chatbook | grep -i path` → `Utils/Utils.py:264 elide_path_middle` (CHARACTER-budgeted, basename-preserving), `Chat/console_display_state.py:1699 middle_elide_path`, `UI/Wizards/first_run_setup_state.py:607 middle_truncate_path`, `Widgets/Library/library_note_import_canvas.py:112 _elide_name_middle`; `grep -rn "split_graphemes" tldw_chatbook` → console_prompt_queue, console_composer_bar, this file.
- Why it matters: the character-budgeted `Utils` helper is NOT interchangeable with the cell-budgeted one (that is why this copy exists), so the missing helper keeps getting re-written.
- Recommended correction: promote `_grapheme_spans` + `_middle_elide_cells` to `tldw_chatbook/Utils/` and retire the two Console copies.
- Size: M · ADR: no · Confidence: verified · Pinning test: none · Already covered: none

### P3 [D4b] — `_widget_id` can collide and mount duplicate ids in the Collections capture reader
- Where: `library_collections_capture_reader.py:74-77`, consumed at `:209-211`, `:948`, `:953`, `:1004`, `:1009` — `re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-")[:48] or "item"`.
- Evidence: read only. Two source ids differing only past char 48, differing only in punctuation (`a.b` vs `a-b`), or both fully non-alphanumeric (both → `"item"`) produce the same Textual id → `DuplicateIds` on mount.
- Recommended correction: hash-suffix the fragment. Size: S
- Confidence: inferred. Settling command: `grep -rn "highlight_id\s*[:=]" tldw_chatbook/Library/collections_capture_models.py` to see whether these are server-opaque strings or generated UUIDs.

### P3 [D4b] — The Parakeet install label hard-codes a size that has a source of truth
- Where: `library_ingest_canvas.py:1601` — `"Install verified Parakeet v2 INT8 (630.6 MiB)…"`.
- Evidence: `sum(f.size_bytes for f in PARAKEET_V2_FILES)` = 661191781 = 630.6 MiB — correct today. `parakeet_v2_artifact.py:252` names `PARAKEET_V2_FILES` "the single source of truth … nothing here re-declares or copies a digest", and `Utils/Utils.py:729 _format_size_bytes` already formats it.
- Recommended correction: derive it. Size: S · Confidence: verified

### P3 [D4b] — `_toggle_label` has no production caller and is the 4th copy of a two-line glyph formatter
- Where: `library_ingest_canvas.py:1174-1177`.
- Evidence: an AST walk over `tldw_chatbook/**/*.py` for `Name`/`Attribute` nodes spelled `_toggle_label` returns 10 hits, none in or importing `library_ingest_canvas`; the only reference anywhere is `Tests/UI/test_library_crit9_grammar.py:102`, which imports it directly. Its body duplicates `library_note_import_canvas.py:63 _choice_label`, `library_search_rag_panel.py:507`, and `console_rag_settings_modal.py:109`.
- Recommended correction: delete it (and its function-body import at `:1176`); move `_choice_label`'s body next to the glyph constants in `Library/library_shell_state.py` and point the three live copies there.
- Size: S · Confidence: verified · Pinning test: `test_library_crit9_grammar.py:102-105` pins the output, which survives the move.

## Candidate dispositions
Every row in `<SCRATCH>/excerpts/W-library.md` is covered. "retired" always carries the evidence that retired it.

| candidate (file:line / group) | disposition |
| --- | --- |
| **dup_shape**, all 18 clusters (the `_cancel`/`_confirm`/`_close`/`_submit`/`_retry` button handlers across 49 files) | **retired** — these are Textual's `@on(Button.Pressed, "#id")` idiom: 2-3 line handlers that `event.stop()` then call one method. Extracting them adds indirection and removes the selector that makes them findable. The one cluster with real content (`_perform_safe_cancel`) is dispositioned separately below. |
| dup_shape `sync_state` ×4 (`library_export_canvas.py:101`, `library_search_rag_panel.py:82`, `library_conversations_canvas.py:51`, `library_notes_sync_roots_canvas.py:255`) | **confirmed** → P3 [D4b] above (a `reactive(recompose=True)` replaces all four). |
| dup_shape `_persist_*_location` ×3 (incl. `library_file_notes_workspace.py:6808`) | **retired** — each wraps `claim_browse_directory`/`remember_browse_directory` with a DIFFERENT context key (`"file_notes"/"browse"` vs the notes-sync and import keys); the shared helper is already the function they call. |
| dup_shape `_arm_autosave` / `_schedule_validation` ×3 (`library_file_notes_workspace.py:6275`) | **retired** — same *shape* (stop a timer, `set_timer`), different debounce constants and callbacks; there is no behaviour to share beyond `set_timer` itself. |
| dup_shape `_release_commit_editor_lease` (`library_file_notes_workspace.py:5289`) with the dictation/frame timer stoppers | **retired** — the three "copies" are unrelated resources (a commit-editor read-only lease vs two UI timers) that happen to share a null-check-and-clear shape. |
| dup_shape `_use_configured_sync_folder` (`:6758`) / `_review` / `handle_use_in_chat_button` | **retired** — a 3-line handler; no shared body. |
| dup_shape `_refit_path_surfaces` / `_sync_editor_action_layout` (`:3133`, `:3331`) | **retired** — both are width-fitting entry points on the same widget with different targets; the duplication that matters is the `_sync_action_layout` probe cost, filed as P1 [D2] against the git panel. |
| **dup_verbatim** `_perform_safe_cancel` ×5 (`library_file_notes_git_panel.py:4270`, `prompt_delete_confirmation_modal.py:197`, + 3 outside the slice) | **retired** — not duplication: `_perform_safe_cancel` is an overridable hook with a default in `Widgets/modal_dismissal.py:256`, and each override supplies that modal's own cancelled RESULT (`False`, `None`, `PromptDeleteDecision(False, fingerprint)`). The `del source` + one-line dismiss IS the contract. |
| dup_verbatim `_close_pressed` git_panel:4103 / `close_pressed` notes_recovery_dialog:149 | **retired** — two-line `event.stop(); self.action_close()`. |
| dup_verbatim, the remaining 6 clusters | **retired** — same reasoning as dup_shape; all are `event.stop()` + one call. |
| **except_exception_pass** `library_file_notes_workspace.py:6420`, `:6429` | **retired** — both `await asyncio.shield(task)` inside `flush_pending_work`; the save task's failure is surfaced by the state machine two lines later (`if self._save_state in {"conflict","error"}: return False`), not swallowed. |
| except_exception_pass `library_media_canvas.py:705` | **confirmed, low** — the legend-label refresh is best-effort and `noqa`-documented, but the `try` also covers a DB read (`_meeting_speaker_legend_rows`), so a DB failure is silent. Folded into the P2 [D4b] speaker-rename drift finding (the reader copy does this read on the worker thread). |
| except_exception_pass `library_media_image_preview.py:237` | **retired** — the documented graphics→mosaic fallback; the mosaic path runs unconditionally after it. |
| **except_exception_return** `library_ingest_canvas.py` ×2 (`:1965`, `:1996`) | **confirmed present, not P1 on their own** — folded into the P1 chunk-template finding (the silent non-population is what makes the display divergence stick). Minor separate defect: the coroutine at `:1961` is constructed before `run_worker`, so a scheduling failure leaks a never-awaited coroutine. |
| except_exception_return `library_note_import_canvas.py` ×2 (`:688`, `:715`) | **retired** — `_fit_source_summary` / `_update_overflow_hint` guard `query_one` for widgets that legitimately do not exist off the `select` phase; consequence is a wider elision or a missing scroll hint. Should be `except (NoMatches, QueryError)`. |
| except_exception_return `library_media_trash_canvas.py` ×2 (`:204`, `:232`) | **retired** — layout measurement helpers guarding `query_one`; same over-broad-catch nit, no data path. |
| except_exception_return `library_notes_canvas.py` ×2 (+1 the grep missed: `:1225`, `:1653`, `:1659`) | **retired** — `_authority_prefix` (presentation only) and `editor_has_focus` (returns `False` when there is no live app, i.e. nobody is typing). None on a persistence path. |
| except_exception_return `library_media_viewer.py` ×1, `library_search_rag_panel.py` ×2 | **`library_search_rag_panel.py:142-145` confirmed** → P2 [D3] (policy denial swallowed). The others retired as `query_one` guards. |
| **function_body_import** `library_file_notes_workspace.py` ×2 (`:6661`, `:6663`) | **retired** — both resolve: `$PY -c "from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDBError; from tldw_chatbook.Widgets.Library.notes_recovery_dialog import NotesRecoveryDialog"` → `function-body imports resolve: CharactersRAGDBError NotesRecoveryDialog`. |
| function_body_import `notes_recovery_dialog.py` ×2 | **retired** — same command, same modules. |
| function_body_import `library_ingest_canvas.py` ×2 | `:2082` (`Utils/install_clipboard`) **retired** — resolves, and the module pulls `subprocess`/`signal`, so deferring it is justified. `:1176` **confirmed** → folded into the P3 dead `_toggle_label` finding. |
| function_body_import `library_notes_canvas.py` ×3 (`:464`, `:2702`, `:3727`) | **retired** — all three resolve (`render_obsidian_callouts`, `front_matter_parser_factory`, `NOTE_TEMPLATES` with 9 entries); laziness justified in-comment by `Tests/Performance/test_screen_preimport_payload_budget.py`. |
| function_body_import `library_search_rag_panel.py` ×5 | **retired** — all five resolve (`main_navigation.NavigateToScreen`, `settings_config_models.SettingsCategoryId`, `ingestion_indexing.semantic_indexing_available`/`get_shared_rag_service`, `runtime_policy.types.PolicyDeniedError`). |
| function_body_import `library_media_image_preview.py` ×3, `library_rail.py` ×1 | **retired** — optional-dep deferrals (`PIL`, `textual_image`, `Utils.mosaic_render`) behind `optional_deps.check_dependency`, which is the documented pattern. |
| **get_cli_setting_hot** `library_file_notes_workspace.py:2879 _configured_sync_folder loop=True` | **confirmed** → P2 [D2] above (9.45 ms median measured for the two-call loop; reached from `_update_root_surface`'s 15 call sites and a 3 s timer while no folder is linked). |
| **legacy_markers**, all 12 files (14 in `library_search_rag_panel.py`, 14 in `library_collections_capture_reader.py`, …) | **retired — the whole group is a false positive.** `grep -c -E "TODO\|FIXME\|XXX\|HACK\|DEPRECATED"` over all 12 files → **0 each**. Every hit was the domain noun: "legacy-chunk report" / `rechunk_legacy_media` (live code reached from `compose()`) and "Legacy Collections" (the read-only recovery surface). No dead code mounted. |
| **plain_readback** `library_file_notes_workspace.py:2839` (`_show_root_row_button`) | **confirmed** → P2 [D1+D4a] above — the only site in the file whose label carries user text (a folder name), and the guard can never match because the read-back is markup-parsed. |
| plain_readback `library_file_notes_workspace.py:3356, 5467, 5515, 5529, 5602` | **retired** — all five read back app-owned literals ("Protect"/"Unprotect", "Reload…", "Save copy"/"Export exact copy") plus the `LIBRARY_DISABLED_ACTION_MARKER` prefix; `:3356` is a `cell_len` measurement. No user text, exact round trip. |
| plain_readback `library_file_notes_git_panel.py:1909, 1922` | **retired** — `cell_len(str(button.label))` measurements feeding a `set_class`; nothing branches on the text. |
| plain_readback `library_file_notes_git_panel.py:3423` | **confirmed** → P2 [D1] above (base label reconstructed from the rendered one; drops Rich spans — latent, no current label contains `[`). |
| plain_readback `library_notes_canvas.py:2423` | **retired** — `_tree_action_buttons:2509-2640` writes literals only; the read-back is a cell-width measurement. All 16 label/disabled combinations round-trip exactly. |
| plain_readback `library_notes_canvas.py:3217, 3425, 3430` | **retired as bugs** (labels are `back_cue_label(...)` literals) → **confirmed as P3 [D3]** redundancy: Textual's reactive already skips equal assignments (`refresh() after IDENTICAL label assignment: 0 []`). |
| plain_readback `library_media_canvas.py:1161, 1176` | **retired** — `str(type_filter.label)` / `str(sort_btn.label)` are passed to `_gate_mutation_action`/`_gate_stale_action` as the ACTION NAME for the disabled-marker helper; both labels come from `library_choice_label(...)`, app-owned. |
| plain_readback `library_media_canvas.py:1845` | **retired** — it is the comment block documenting why the raw remainder is stashed on the button instead of read back. |
| plain_readback `library_media_raw_view.py:84, 440` | **retired** — `Content.plain` used for selection/copy mapping, documented at `:78-86` (`Widget.get_selection` reads `Content.plain`, expand_tabs never mutates it). This file is the best-guarded in the slice. |
| plain_readback `library_media_image_preview.py:187` | **retired** — `mosaic.plain.splitlines()` to size an internally generated mosaic. |
| plain_readback `library_adaptive_reader_shell.py:210` | **retired** — `self.label.plain` is the grip's own arrow glyph. |
| plain_readback `library_conversations_canvas.py:314` | **retired** — the comment documenting the stash-the-raw-remainder pattern; the label itself is built through `_visible_row_title` (whose escaper gap is the separate P1). |
| **query_one_in_timer_no_try** `notes_recovery_dialog.py:96, 97` | **confirmed (mechanism) / retired (crash)** → P2 [D1] above. `NoMatches` reproduced against a pruned dialog; 40 close cycles produced no crash because Textual stops the pump's timers in `_close_messages` first. The certain defect is that the interval never stops. |
| **run_worker_coroutine** `library_ingest_canvas.py` ×1 | **confirmed** → P2 [D2] (sync `SELECT *` on `MainThread`). |
| run_worker_coroutine `library_file_notes_workspace.py` ×19 | **retired** — the file wraps every blocking call in `asyncio.to_thread` (28 sites: `service.open_file`, `save_file`, `scan`, `FileNotesService(...)`, `_build_runtime`, `_close_owned_replica`, …). The one raw `service.scan` (`:5857`, which blocks on `_service_lock`) is itself only reached through `asyncio.to_thread(self._scan_for_root, …)` (`:5983`). |
| run_worker_coroutine `library_file_notes_git_panel.py` ×2 | **confirmed** → P2 [D1] (`exit_on_error=True` + unguarded `query_one` as the first statement). |
| run_worker_coroutine `library_search_rag_panel.py` ×1 | **retired** — `_fetch_legacy_chunk_report` awaits `RAGAdminScopeService.get_template_diagnostics` → `_call_off_loop` (`rag_admin_scope_service.py:87-119`) → `asyncio.to_thread`, gated on `diagnostics_are_thread_safe()`. |
| run_worker_coroutine `notes_recovery_dialog.py` ×1 | **retired** — `_confirm` only awaits `push_screen_wait` and the caller-supplied `approve`, which the workspace already wraps in `asyncio.to_thread` (`library_file_notes_workspace.py:6688`). |
| **seed_name `_cancel`** `library_note_import_canvas.py:1359`, `library_note_folder_dialog.py:62`, `:141` | **retired** — `event.stop()` + `post_message`/`dismiss`; see dup_shape. |
| **seed_name `_normalize_mode`** `library_media_content.py:292` | **retired** — a validator that raises on an unknown mode and forces non-Markdown bodies to Raw; domain-specific, not a copy of any other `_normalize_*`. |
| **seed_name `_perform_safe_cancel`** `prompt_delete_confirmation_modal.py:197`, `library_file_notes_git_panel.py:4270` | **retired** as duplication (the mixin hook contract) — but the audit found the inverse defect on the SAME two classes: their affirmative paths (`library_file_notes_git_panel.py:4286`, `prompt_delete_confirmation_modal.py:213`) use raw `self.dismiss(...)`, filed as P2 [D1] and noted in Verified-fine. |
| **try_import_guard** `library_media_image_preview.py:225` (`build_media_image_widget`, `handlers=['Exception']`) | **retired** — guarded by `optional_deps.check_dependency("textual_image")` first and falls back to the mosaic renderer unconditionally; the broad catch also covers `fit_image_cell_size`, which is the documented intent. |

## Verified-fine
Things in this slice that look like smells and are not, with the evidence.

- **No sqlite, no file I/O, no subprocess anywhere in `Widgets/Library/`.** `grep -n "sqlite3\|\.execute(\|\.fetchall\|open(\|write_text\|read_text\|shutil\.\|subprocess\|Popen\|shell=True" *.py` → the only hits are two `from sqlite3 import Error` type imports, `Image.open(BytesIO(...))`, and `sync_open`/`_section_open` method names. Git work leaves as typed `Message` subclasses and reaches `asyncio.create_subprocess_exec` (direct argv, no shell) in `Notes/file_notes_git_service.py:2033` and `Notes/git_process_containment.py:189`. The widgets hand over no paths — only group ids and opaque binding keys — so `Utils/path_validation.py` is not bypassed here; the trust boundary is the controller/service layer. (The one path the workspace DOES validate, a config-sourced folder, goes through `validate_existing_absolute_directory`, `:2884`.)
- **Every `run_worker(exclusive=True)` in the slice passes `group=`.** `grep -n -A6 "run_worker(" *.py` over all 27 call sites.
- **`library_file_notes_workspace.py`'s threading discipline is right.** 28 `asyncio.to_thread` wrappers; the blocking `_service_lock` poll and the unbounded folder scan are both reached only from inside `to_thread`.
- **Tree labels are `Text(...)`, not markup.** `library_file_notes_workspace.py:3404, 3472, 3481, 3492` — a file named `[bold]literal.md` renders literally, and `Tests/Backup_Recovery/test_notes_recovery_controls.py` creates exactly that file and asserts it.
- **Tooltips are correctly escaped.** Textual's `Tooltip` is a `Static` created by the screen with `markup=True` (`textual/widgets/_tooltip.py`, `screen.py:1162`), so `escape_markup(row.title)` at `library_media_canvas.py:1879` / `library_media_trash_canvas.py:447` is right, not a stray backslash. (`escape_markup` still misses the `[TODO]` shape here — same P1.)
- **`library_media_raw_view.py`'s debounce timer is the model callback in this package**: `_fire_pending_reindex` checks `is_attached` (not `is_mounted`, with the reason recorded), and `:183-189` documents that `set_timer(0.0)` never fires in Textual 8.
- **No per-keystroke config read anywhere in the slice.** `grep -n "get_cli_setting\|load_settings"` over all 37 files → four hits, all in `library_file_notes_workspace.py`, none on a keystroke path.
- **This slice adds nothing to the `Utils/NotificationHelper.py` cluster.** `grep -n "notify(" *.py` → exactly four sites, all direct `self.app.notify(...)` (Textual's own API), no `_notify_*_warning` re-roll. `NotificationHelper` importer count re-confirmed at 1 (`Notifications/notification_dispatch_service.py:8`).
- **No widget in this slice is on the `get_active_review_set()` keypress chain** (UI-library P1 / UIM-library P2): `grep -rn "get_active_review_set" tldw_chatbook` → every call site is in `UI/Screens/library_screen.py` plus the service.
- **No `re.compile` in a function body** anywhere in the slice; the three `re.sub`/`re.search` calls use literal patterns (module cache).
- **Zero real legacy markers** in the whole slice (see the candidate table).
- **`library_canvas_sync.py`'s recompose ordering is sound** — it re-queues a follow-up when `_recompose_required` is re-armed, clears the callback unconditionally, and early-returns when detached or `_pruning`.
- **`str(Static.content)` / `.renderable` read-backs are exact.** Textual 8.2.8's `Static` has no `renderable`; `tldw_chatbook/__init__.py:86` installs a shim aliasing it to `content`, invoked by `Widgets/__init__.py:5`. Probed: `'Created 2026-06-30 [v3]' -> 'Created 2026-06-30 [v3]' equal=True`, and `"3 results for 'a \\[b]'"` round-trips with the backslash intact.
- **The prompt-delete flow identifies its target by id, not by a display string.** `prompt_delete_confirmation_modal.py` carries names for copy only; identity is the opaque `fingerprint`, re-validated against `_library_selected_row_id`, view, `select_mode`, `selection.generation`, `prompt_id` and `expected_version` (`library_screen.py:27192-27240`) before `PromptBatchTarget(entry.local_id, entry.expected_version)` is deleted. No label read-back in the gate.
- **The Notes canvases' deferred callbacks are guarded** — `_retain_tree_pager_focus:1108` checks `pager.is_attached`, `_focus_published_comparison:1356-1369` re-validates against a live query, `_focus_requested_conflict:1451` checks `focused.is_attached`; and `call_after_refresh` posts to the widget's OWN pump, which returns `False` once closed.
- **`library_media_viewer.py`'s three `refresh(recompose=True)` calls are all content changes**, not banner/loading flips; `sync_loading_state` is the in-place path and no caller bypasses it (`library_media_controller.py:3893-3901` excludes `loading`/`loading_message` from the compare and patches in place at `:3979`).

## Retired
Candidates raised during the audit and then retired, with the evidence.

- **`notes_recovery_dialog.py` timer crash (the flagged D1).** Mechanism reproduced (`NoMatches` when the callback runs after the children are pruned) but **not reachable**: 40 push/dismiss cycles with the querying branch active produced no app error, because `MessagePump._close_messages` (`message_pump.py:533-535`) stops the pump's timers before the dismiss settles. Downgraded to the never-stopping-interval defect (P2).
- **Media row titles / trash row titles losing markup at the `escape_markup` layer** — initially raised as "unescaped user title into a Button label"; retired as stated, because `_visible_row_title` does escape. The real defect is narrower and different: `escape_markup` only covers `[a-z#/@]`-initial brackets (P1 above).
- **`Static(title)` at `library_media_canvas.py:1018` and `Static(self._title)` at `library_note_folder_dialog.py:40/110`** — retired: the titles are `"Media (n)"` and the literals `"New folder"`/`"Rename folder"` (`library_screen.py:29723`, `:29746`).
- **Media row `secondary` not escaped** (`_media_row_label_rest:210`) — retired: `secondary` is `f"{media_type} · {age}"` from the state builder; `media_type` is a controlled vocabulary column, not free text.
- **`_perform_safe_cancel` ×5 "verbatim duplication"** — retired: it is the `SafeModalDismissMixin` hook contract (`Widgets/modal_dismissal.py:256`), each override returning its own cancelled result.
- **`Widgets/Library/__init__.py` eager imports as a boot cost** — measured and kept at P3: 27 modules and 17 ms of self-time; the 492 ms cumulative is transitive deps I did not attribute.
- **Search/RAG and Collections "14 legacy markers each"** — retired: zero TODO/FIXME/XXX/HACK/DEPRECATED in either file; the grep matched the domain nouns "legacy-chunk report" and "Legacy Collections", both live code.
- **`_restore_editor_mode_focus` unguarded `query_one` in `call_after_refresh`** (skills `:1048`, prompts `:415`) — probe drove focus → `set_editor_mode("basic")` → immediate recompose into list mode → 10 pumps: **no exception**; the `Callback` message drains before the idle-driven recompose, so the callback always sees the old tree, and every id it queries is composed unconditionally in editor mode.
- **`id=f"library-skill-row-{row.name}"` raising `BadIdentifier`** — retired: every write path normalizes through `_normalize_skill_name`, which raises on a bad name (`skills_schemas.py:55-62`), and there is no directory scan that could resurrect one.
- **Duplicate ids across simultaneously mounted Prompts/Skills canvases** — retired: the list canvases are fed `import_open=False` unconditionally (`library_prompts_controller.py:961`, `library_skills_controller.py:972`), and the one genuinely duplicated id is consumed with the plural `canvas.query(...)` (`library_prompts_controller.py:4245`).
- **`_sync_review`'s unguarded `query_one`s** (`library_notes_add_from_files_canvas.py:1319-1337`) — retired: `sync_state`'s fast path requires an unchanged `observation_token`, so the composed DOM always matches the snapshot's rows.
- **`_settle_commit_list_focus`'s stored `f"#{focused.id}"` going stale** — retired: the capture requires `focused.id is not None`, which excludes every dynamically mounted `_SessionGitListItem`.
- **`LibraryDetailsRow.content` shadowing `Static.content`** — retired: `Static.render()` reads the name-mangled `_Static__content`, which `__init__`/`update` assign directly, so the override cannot intercept them.

## Left UNVERIFIED
| claim | why not verified | literal command to run |
| --- | --- | --- |
| The `notes_recovery_dialog` timer can crash the app in a real session | 40 dismiss cycles under `run_test` produced no error; a real terminal's timing differs | `.claude/skills/verify/SKILL.md` recipe: `tmux -L verify new -d -s lib 'python3 -m tldw_chatbook.app'`, open Library ▸ Folder files ▸ Review pairing, change the folder in a second instance so `current()` goes False, then press Close repeatedly and `tmux -L verify capture-pane -p -S -200` |
| The git panel's detach window (`exit_on_error=True` workers, unguarded public render API) is reachable in a real session | needs a live service callback landing during a Library recompose | same tmux recipe: open Library ▸ Folder files ▸ session Git, start a status refresh, then navigate away with the rail before it returns; `capture-pane` for a traceback |
| `PushDestinationAuthorizationDialog`'s double-dismiss | needs two `Button.Pressed` in one frame | `await pilot.click("#file-notes-push-auth-confirm")` twice with no `pause()` between, then assert `len(app.results) == 1` and `app.screen is not modal_below` |
| The Skills `recovery_review` + `source_summary_fresh=False` pair occurring live | the divergence is proven; the state pair is not | instrument `_library_skills_canvas_kwargs` to log `(trust_posture, state.source_summary_fresh)` during a Skills refresh on a restored trust store |
| The Notes location row's false "database only" sentence persisting long enough to be seen | the mechanism is measured; the production window depends on which construction path runs | `$PY -m pytest Tests/UI/test_library_notes_w5_ideas.py -q` with `await pilot.resize_terminal(...)` inserted between shell build and `_open_first_note` |
| `_widget_id` collisions in the Collections capture reader | depends on whether highlight ids are opaque server strings | `grep -rn "highlight_id\s*[:=]" tldw_chatbook/Library/collections_capture_models.py` |
| The ingest `SELECT *` block duration | thread identity measured, duration not | seed a real `MediaDatabase` with N `ChunkingTemplates` rows and time `ChunkingInteropLibrary.get_all_templates` |
| That the always-rendered empty pager `Static` costs a visible row (prompts canvas) | not measured | `Static("").region.height` in an `app.run_test()` harness with the prompts canvas mounted |

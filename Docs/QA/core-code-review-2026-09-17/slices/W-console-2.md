# W-console-2 — `tldw_chatbook/Widgets/Console/` (second half by line count), 34 files, 33,727 lines

Worktree `/Users/macbook-dev/Documents/GitHub/tldw-review` @ d8fb4053f9. Probes under `<SCRATCH>/probe_*.py`, all run as
`cd <worktree> && source <SCRATCH>/env.sh && PYTHONPATH=<worktree> $PY <probe>`.

## Coverage
| file | lines | read in full / sampled (which ranges) / mechanical only |
|---|---:|---|
| console_transcript.py | 8333 | **read in full** (1-8333, 14 contiguous chunks) |
| console_settings_modal.py | 7601 | **read in full** (1-7602) |
| console_session_switcher_modal.py | 2591 | **read in full** (1-2592) |
| console_workspace_context.py | 2199 | **read in full** (1-2200) |
| console_turn_file_card.py | 1331 | **read in full** (1-1332) |
| console_workspace_tree.py | 1328 | **read in full** (1-1328) |
| console_workspace_files_modal.py | 1154 | **read in full** (1-1155) |
| console_status_chips.py | 945 | sampled 925-945 (its one `except_exception_pass` row) + mechanical sweeps |
| console_session_surface.py | 871 | sampled 220-235, 725-780 (all 4 `except_exception_pass` rows) + mechanical sweeps |
| console_terminal_workspace.py | 681 | mechanical sweeps only (timers, run_worker, recompose, imports, readback) |
| console_setup_modal.py | 569 | sampled 415-450, 510-535 (all 3 `except_exception_pass` rows) + mechanical sweeps |
| console_settings_summary.py | 487 | mechanical sweeps only (its `recompose=True` row at 308) |
| console_video_preview.py | 443 | sampled 280-345, 390-443 (its `set_interval` row at 295 and the callback chain) |
| console_workspace_switcher_modal.py | 438 | sampled 22-75, 150-270 (persona-suffix helper + the compose loop) |
| console_side_chat_modal.py | 419 | mechanical sweeps only (run_worker at 224 has a group) |
| trace_export_dialog.py | 379 | sampled 260-305 (its 2 function-body imports + 2 run_worker rows) |
| console_send_authority_summary.py | 378 | mechanical sweeps only |
| console_style_picker_modal.py | 361 | sampled 210-300 (its `set_timer` row at 236 + `_apply_filter`/`_render_results`) |
| console_workspace_action_menu.py | 355 | sampled 240-330 (its 2 run_worker rows) |
| prompt_variables_dialog.py | 305 | mechanical sweeps only |
| console_workspace_details.py | 304 | mechanical sweeps only (its `recompose=True` row at 67) |
| console_staged_context.py | 301 | sampled 240-301 (its function-body import + `recompose=True` rows) |
| console_video_card.py | 242 | mechanical sweeps only (its function-body import at 200) |
| console_video_capacity_modal.py | 235 | mechanical sweeps only |
| console_terminal_session_modal.py | 230 | mechanical sweeps only |
| console_speech_controls.py | 191 | mechanical sweeps only |
| console_task_panel.py | 178 | mechanical sweeps only |
| console_voice_preview.py | 175 | mechanical sweeps only |
| console_system_prompt_modal.py | 167 | mechanical sweeps only (run_worker at 150 has a group) |
| console_workbench_state.py | 153 | mechanical sweeps only |
| console_summarize_preview_modal.py | 149 | mechanical sweeps only |
| console_staged_evidence_strip.py | 109 | mechanical sweeps only (its `recompose=True` row at 109) |
| trace_export_profile_ui.py | 66 | mechanical sweeps only |
| console_terminal_messages.py | 59 | mechanical sweeps only |

"Mechanical sweeps" = every file in the slice was run through, and every hit inspected, for: `set_interval`/`set_timer`
(10 sites), `run_worker(` (16 sites, every one checked for `group=`), `refresh(recompose=True)` (10 sites),
`get_cli_setting`/`load_settings` (2 sites, both in console_transcript.py), function-body imports (12 real ones, all
import-probed), `.plain`/`str(...label)`/`str(...renderable)` read-back (10 sites), and the excerpt's `except Exception:
pass/return` rows. The seven files over 1,000 lines were read line by line.

---

## Findings

### P1 [D2] — `get_cli_setting("console","turn_file_cards")` runs once per turn-file-card row build at ~5 ms a call; 20 such rows turn a transcript row-build pass from 1.2 ms into 102 ms
- **Where:** `tldw_chatbook/Widgets/Console/console_transcript.py:7345` (`_build_message_widget`) and `:7700` (`_update_row_widget`); import at `:94`.
- **Evidence:**
  - `probe_gcs.py` (200 warm iterations) → `get_cli_setting warm ms median=4.8012 mean=4.8217 max=5.9183`
  - `probe_tfc.py` — a `ConsoleTranscript` with 20 USER + 20 TOOL rows, `_message_widgets()`, median of 8 passes:
    `with_card=False: 1.2 ms (81 widgets)` vs `with_card=True: 107.9 ms (81 widgets, 20 ConsoleTurnFileCard)`
  - `probe_tfc2.py` — same transcript, stubbing only the module-level name:
    `real get_cli_setting: 101.9 ms` → `get_cli_setting stubbed: 1.2 ms (calls per pass=20)`.
    **99 % of the pass is config I/O, exactly one read per card row.**
- **Why it matters:** `compose()` (`:3262-3270`) and `_reconcile_rows` build every row through this path, so a resumed agent session with N file-change turns pays N × ~5 ms on open. `:7700` additionally sits on the *selection* path — the card row's signature folds in `selected` (its own comment, `:7686-7695`) — so moving j/k selection onto or off a card row pays another ~5 ms per keypress.
- **Recommended correction:** the same class already has the cheap shape two methods away: `_assistant_markdown_enabled()` (`:4862`) and `_prune_watermarks()` (`:4854`) read `getattr(self.app, "app_config", None)` and hand it to a pure resolver. Mirror that, or (smallest diff that preserves the pinning test's patch point) call the module-level `get_cli_setting` **once per row-build pass** and thread the bool through `_build_row_widget`/`_update_row_widget`.
- **Size:** S · **ADR:** no · **Confidence: verified**
- **Pinning test:** `Tests/UI/test_console_turn_file_card_factory.py::test_summary_row_stays_plain_marker_when_disabled` patches `transcript_mod.get_cli_setting` (the module-level name at `:94`), so a per-pass memo of that same name keeps it green. `Tests/Chat/test_console_diff_feedback_delivery.py::test_kill_switch_off_does_not_prevent_note_delivery` patches `config_module.get_cli_setting` and never reaches this path.
- **Already covered:** none

### P1 [D2] — `ConsoleWorkspaceSwitcherModal.compose()` runs a synchronous sqlite SELECT (~2 ms) per workspace row, undoing the caller's deliberate off-thread fetch
- **Where:** `console_workspace_switcher_modal.py:199`, inside the `for index, workspace in enumerate(self._workspaces)` loop opened at `:183` **inside `compose()`**, calling `workspace_persona_label_suffix` (`:22-65`), which at `:42` calls `registry.get_workspace(...)` and at `:56` `personas.get_persona_profile(...)`.
- **Evidence:**
  - `tldw_chatbook/Workspaces/registry_service.py:658-674` — `get_workspace` is `with self.db.connection() as conn: conn.execute("SELECT * FROM workspace_records WHERE workspace_id = ?").fetchone()`. Plain synchronous sqlite.
  - `probe_ws.py`, real `WorkspaceDB` in a tmp dir, 200 warm iterations →
    `get_workspace warm ms median=2.0116 mean=2.0416 max=2.5979`
  - The caller already fetched the same records **off the loop**: `tldw_chatbook/UI/Console_Modules/workspace.py:4259-4263`
    `workspaces = tuple(await storage_call(registry_service, "list_workspaces", include_archived=True))`, and `storage_call`
    (`tldw_chatbook/Chat/conversation_archive_actions.py:50-69`) is `await asyncio.to_thread(call, ...)`. Those exact
    `WorkspaceRecord`s reach the modal at `workspace.py:4310-4314` — and `workspace_persona_label_suffix` already accepts
    them: `:45-46` `if record is None: record = workspace`.
- **Why it matters:** the modal's first paint blocks the event loop for N × ~2 ms of sqlite plus N persona lookups (10 workspaces ≈ 20 ms, 30 ≈ 60 ms) on the very screen the caller took care to keep off it. The re-read is also strictly redundant with data already in hand.
- **Recommended correction:** use the passed-in `WorkspaceRecord`'s `assistant_defaults` (the branch the code already has as a fallback), or resolve every suffix once, off-thread, in `_open()` beside `list_workspaces` and pass them in. No per-row service call in `compose()`.
- **Size:** S · **ADR:** no · **Confidence: verified** (per-call timing measured; the N-row multiply is a direct read of the loop)
- **Pinning test:** none — `grep -rn "workspace_persona_label_suffix" Tests/` returns nothing; only the module defines it.
- **Already covered:** none

### P2 [D1] — a raising filesystem service leaves the Workspace Files modal permanently on "Loading folder…", with no log line anywhere in the module
- **Where:** `console_workspace_files_modal.py:202-214` — `_OperationLane._run`'s `except Exception: outcome = None` (`:208-209`), after which `:213` refuses to publish and the caller's `status_copy="Loading folder…"` (`:606`) is never replaced. The module imports **no logger at all** (`grep -n logger console_workspace_files_modal.py` → no hits).
- **Evidence (reproduced):** `probe_wsfiles.py` — the real modal in the real `_Host` harness (`Tests/UI/test_console_workspace_files_modal.py`) with an inspector whose `list_directory` raises `PermissionError`:
  ```
  status_copy  = 'Loading folder…'
  tree content = 'Loading folder…'
  state.status = 'Loading folder…'
  lanes active = 0
  ```
  Stuck forever, zero lanes running, nothing logged.
- **Why it matters:** an unexpected raise is indistinguishable from "still loading" for the user and invisible to an operator. Why P2 and not P1: the real `LocalWorkspaceFileInspector.list_directory` (`Workspaces/file_inspector.py:290-370`) is heavily guarded and returns explicit `DirectoryStatus.FAILED` values, so I could not show a *shipped* path that raises — `os.close(root_fd)` at `:328`, `os.fstat(directory_fd)` at `:331` and `_directory_revision_from_stat` sit outside the `try` that starts at `:343`, which is the nearest thing to one.
- **Recommended correction:** log the exception (`logger.opt(exception=True).warning(...)`) and publish a failure status (`status_copy = "Folder is unavailable."`) instead of `outcome = None`. `_publish_directory` already has copy for `DirectoryStatus.FAILED` (`:663`).
- **Size:** S · **ADR:** no · **Confidence: verified** (consequence reproduced; trigger inferred)
- **Pinning test:** none for the raising branch; `Tests/UI/test_console_workspace_files_modal.py` exercises the returning-status branches.
- **Already covered:** none

### P2 [D3] — `console_transcript.py` (8,333 lines / one 5,365-line class) and `console_settings_modal.py` (7,601 / one 6,597-line class) are god modules that **no size ratchet governs**
- **Where / cluster map (from `ast`, `probe`-free):**
  - `console_transcript.py` — 58 module-level defs/classes, then `class ConsoleTranscript` at **2969-8333, 5,365 lines, 193 methods**. Responsibility clusters inside that one class: windowing/hydration (`3491-4197`), presentation setters (`4198-4738`), pruning (`4842-5058`), thinking/activity projection (`5085-5182`), keyboard text-selection mode (`5412-5623`), mouse drag-selection + floating menu (`5750-6460`), row planning (`6623-7030`), row build/reconcile (`7074-7742`), signatures/caching (`7744-7831`), action rows + overflow menu (`8124-8333`).
  - `console_settings_modal.py` — 18 module-level defs/classes, then `class ConsoleSettingsModal` at **1005-7601, 6,597 lines, 245 methods**: a ~970-line `compose()` (`1589-2593`), focus/scroll/layout (`2837-3253` + `3787-3828` + `4023-4139`), draft snapshot/restore (`2695-2836`), default-durability recovery (`3522-3772`), context-policy controls (`4267-4530` + `7238-7394`), provider/model rebase (`5047-5560`), model discovery + connection probes (`5560-6486`), generation test (`5928-6127`), readiness sync (`6488-6612`), provider/base-URL resolution (`6858-7208`).
- **Evidence:** `Tests/Architecture/test_screen_size_ratchet.py::_BUDGETS` holds exactly two rows — `UI/Screens/chat_screen.py (16966, 563)` and `UI/Screens/library_screen.py (33204, 1276)`. `Tests/Architecture/test_library_modules_size_ratchet.py` covers only `UI/Library_Modules/*_controller.py` (21 paths). Neither names anything under `Widgets/Console/`. `console_transcript.py` at 8,333 lines is larger than most of the 21 governed Library controllers.
- **Why it matters:** the recipe's §17 (`backlog/docs/library-decomposition-recipe.md:3309`) exists precisely because "the files we decompose INTO had no size governance at all". `Widgets/Console/` is the same gap, one package over, and already at screen scale.
- **Recommended correction:** do not redesign — follow `backlog/docs/library-decomposition-recipe.md` §17 option (a): a sibling `Tests/Architecture/test_console_widgets_size_ratchet.py` with exact per-file `_BUDGETS` rows discovered by glob over `Widgets/Console/*.py`, pinned at today's measured counts. Any actual split must follow §1 (per-subsystem PR series) and §2 (field-ownership script).
- **Size:** M for the ratchet (one new test file, no production change) · L for any split · **ADR:** no for the ratchet (§17 already rules the shape) · **Confidence: verified**
- **Pinning test:** the recommendation *is* the pinning test.
- **Already covered:** `task-1378` splits `settings_screen.py` and `task-31202` adds its ratchet row — the same pattern, a different file; neither covers `Widgets/Console/`.

### P2 [D3] — the switcher modal drives character activation on a bare `asyncio.create_task`, outside Textual's worker lifecycle, and never cancels it on unmount
- **Where:** `console_session_switcher_modal.py:2235` (`_begin_character_activation`) and `:2440` (`_recover_character_activation`). `on_unmount` (`:548-566`) stops both timers and sets the cancellation `Event` only in the `OPENING_CANCELLABLE` phase; it never touches `self._activation_task`.
- **Evidence:** `.venv/.../textual/widget.py:4849-4851` — `Widget._on_unmount` calls `self.workers.cancel_node(self)`, so every `run_worker` task on this screen is cancelled at unmount; a bare `asyncio.create_task` is not. The same class uses `run_worker(..., group=...)` for its four other async paths (`:650`, `:1326`, `:1877`, `:1907`).
- **Why it matters:** after the modal is popped, `_run_character_activation` keeps awaiting the activator and then drives the dead screen (`_show_activation_failure` → `_set_status`, or `dismiss_safe_once`); the modal cannot be collected until the adapter returns. Every reached call site is `try/except NoMatches`-guarded, so I could not produce a crash — this is lifetime/leak, not a crash.
- **Recommended correction:** `self.run_worker(self._run_character_activation(...), group="console-session-switcher-activation", exclusive=True)`, matching its own four siblings.
- **Size:** S · **ADR:** no · **Confidence: inferred**. Literal command to settle it: a probe that pushes the modal, calls `_begin_character_activation` with an activator blocking on an `asyncio.Event`, pops the screen, and asserts `modal._activation_task.cancelled()`.
- **Pinning test:** none
- **Already covered:** none

### P3 [D4] — `wrap_console_plain_text_uncapped` is a hand-copy of `wrap_console_conversation_title`'s wrap loop with the 2-line cap removed
- **Where:** `console_workspace_context.py:155-206` and `:209-265` — the `while remaining:` bodies are line-for-line identical apart from the `_TITLE_WRAP_MAX_LINES` branch and the blank-title fallback.
- **Evidence:** read only; the second function's own docstring asserts "this is strictly what it would compute with the cap lifted" — an invariant held by a comment, not by construction.
- **Why it matters:** two copies of a cell-aware greedy wrap that MUST agree, because `_conversation_browser_list_height` measures with one and `_compose_conversation_browser_row` renders with the other. TASK-1142 already shipped one clipping bug from exactly that disagreement (documented at `:1310-1339`).
- **Recommended correction:** one function taking `max_lines: int | None = None` and `blank_fallback: str | None = None`; keep both public names as thin wrappers so call sites and tests are untouched. Canonical home: stays in this module (no `Utils` wrap helper exists and the budget semantics are Console-row-specific).
- **Size:** S · **ADR:** no · **Confidence: verified (read)** · **Pinning test:** none named · **Already covered:** none

### P3 [D3] — `console_turn_file_card.py:1319` re-imports `rich.text.Text` inside `_styled_diff`, shadowing the module-level import at `:19`
`from rich.text import Text` is already at module scope (`:19`, used at `:716` and `:1282`). The function-body copy is a per-call `sys.modules` lookup plus a rebind for nothing — not an optional-dep guard, not a cycle break. Delete the line. **Size:** S · **Confidence: verified (read).**

### P3 [D3] — `console_transcript.py` reaches into another class's private attribute and its `_TranscriptRow.kind` Literal is missing a live value
- `:6287` `message = getattr(row, "_message", None)` in `_row_supports_selection_feedback` reads `ConsoleTranscriptMessage`/`ConsoleMarkdownMessage`'s private `_message`. Both classes are in this same module, so it is intra-module, but they already expose `message_id`; a `role` property (or a small `selection_feedback_eligible` predicate on the row classes) would remove the reach-through.
- `:1147-1163` — the `kind` `Literal[...]` lists 15 values and omits `"library-activity"`, which `_flat_transcript_rows` constructs at `:6736` and `_build_row_widget` handles at `:7307`. Runtime is unaffected (a `Literal` is not enforced); the annotation is simply wrong and a type check would flag it.
- **Size:** S each · **Confidence: verified (read)** · **Already covered:** none

### P3 [D4] — documented duplication, reported not recommended (per brief)
`console_transcript.py:639 _human_size` duplicates `Chat/attachment_core._format_size`, with the intent stated at `:640-644`. Reported as *documented* duplication only. This file is also one of the 16 byte-size-formatter sites in the already-tabled cluster.

---

## Candidate dispositions
| candidate (file:line / pattern) | verdict |
|---|---|
| `query_one_in_timer_no_try`: console_settings_modal.py:5400 `_apply_readiness_sync_debounced` | **retired** — `#console-settings-model-picker` is composed unconditionally (`compose()` 1769-1776, no branch); view switching only flips `.display` (`_show_settings_view` 3254-3301); no `refresh(recompose=True)` and no `remove_children` in the file; Textual stops a pump's timers on close (`message_pump.py:533-535`). Cannot raise `NoMatches`. I additionally swept **all 10** `set_interval`/`set_timer` sites in the slice (list in Verified-fine) — every one of them is either guarded or targets an unconditionally-composed id. |
| `plain_readback` × 10 (settings_modal 3204/3452, transcript 856/864/1018/2034/2096/2720, workspace_context 376, workspace_tree 714) | **all retired** — see Verified-fine for the per-site evidence. Textual 8.2.8 `Content.__str__` returns `self._text` (`content.py:179-180`), and no read-back in this slice is fed back into a markup-parsing surface. |
| `except_exception_pass` × 11 (session_surface 228/731/738/774, settings_modal 5680/6103, setup_modal 428/438/524, status_chips 937, workspace_switcher 63) | **retired** — every one is a display-only affordance (overflow hints, scroll scheduling, tooltip suffix, notify) with an in-code rationale; none is a data path. |
| `except_exception_return_per_file` × 8 (34 sites) | **1 confirmed** (`console_workspace_files_modal.py:208`, the P2 above); the rest retired — all guard `query_one`/`self.screen`/`self.app` on teardown races or degrade a render. Two settings-modal ones deserve a note but not a finding: `:4552` (`save_settings_to_cli_config` failure → `saved = False`) and `:4809` (live-commit failure) report to the user via `_set_validation_error`/status copy but log nothing. |
| `function_body_import_per_file` × 6 (12 real sites; the other grep hits are docstring text) | **all retired** — every target imports and every named attribute exists. `probe_fbimports.py` → 15/15 `OK`. |
| `run_worker_coroutine_per_file` × 8 | **all retired** — 16 `run_worker` calls in the slice, **every one has `group=`** (AST sweep printed zero "NO GROUP" rows). Coroutine workers only `await` (`asyncio.to_thread` for every blocking read); no sync sqlite inside one. |
| `dup_shape` × 19 / `dup_verbatim` × 19 | **retired as findings.** The Console-local pairs are 3-to-8-line `@on(Button.Pressed)` adapters (`_cancel`, `_close`, `_submit`, `_show_page`, `_detach`, `_move_highlight`, `_perform_safe_cancel`, `sync_state`, `_next_request_id`) whose bodies are `event.stop()` + one typed call. They are Textual's dispatch contract, not logic: each must exist under its own selector on its own class, and the real shared behaviour already lives in `Widgets/modal_dismissal.SafeModalDismissMixin` (which `_perform_safe_cancel` overrides by design). The one exception with real substance is `_human_size`/`_format_size` — reported above as documented duplication. |
| `seed_name__cancel` × 9 / `__perform_safe_cancel` × 8 / `__set_status` × 3 | **retired** — same reasoning; `_perform_safe_cancel` is the mixin's documented override hook (`Widgets/modal_dismissal.py:256`). |
| `inline_truncate`: console_transcript.py:1112 `[:199] + "…"` | **retired** — `_annotation_marker_content` caps a review-note preview line at 200 cells for an inline marker row with no scroll; the full note is in the notes modal (`:1106-1108`). Deliberate and documented. |
| `raw_1024x1024`: console_transcript.py:645, console_video_capacity_modal.py:25 | **retired** — `:645` is `_human_size`'s `1024 * 1024` MB threshold; `:25` is the same byte-unit constant. Not an image dimension. |
| `strftime`: console_transcript.py:5640 `%H:%M:%S` | **retired** — inside `_paint_debug_dump`, a probe that no-ops unless `TLDW_TRANSCRIPT_PAINT_LOG` is set (`:5630-5633`). A local wall-clock stamp in an operator-only debug file. |
| `try_import_guard` × 3 (transcript 5635/7636, workspace_switcher 33) | **retired** — `:7636` guards the genuinely optional `textual_image` and falls back to the Pixels row with a logged warning; `:5635` is the env-gated debug probe; `workspace_switcher:33` is display-only degradation. All import fine here (`probe_fbimports.py`). |
| `legacy_markers_per_file` × 9 | **not examined as a class** — these are `TASK-nnnn`/`Qodo #n` provenance comments, which in this package are load-bearing incident records, not dead-code markers. |

## Verified-fine
- **`console_settings_modal.py:2597` `set_interval(0.25, self._poll_subscription_readiness)` is not a hot-path cost.** `probe_settings_poll.py` (real modal in `Tests/UI/test_console_session_settings.ModalHarness`, 60 iterations, median/max/mean ms): `poll tick 0.104 / 0.123 / 0.107`; `_build_draft 0.037`; `_readiness_for_current_draft 0.057`; `subscription_readiness_revision 0.0000`; `_sync_readiness_display 0.309`. 0.104 ms every 250 ms ≈ 0.04 % of a core. The sibling's "idle 0.25 s poll at 30-49 ms/tick" is a different poll. No `get_cli_setting` is on this path.
- **The whole `plain_readback` class.** `Content.__str__` → `self._text` (`.venv/.../textual/content.py:179-180`). Per site: settings_modal `3204`/`3452` feed `len(...)` into `styles.width`; `3299` is a presence test; `4135` an emptiness test; `4155` reads a literally-composed label into a plain tooltip string. switcher `2174` wraps the result in `Text(...)` two lines later (`:2178`). workspace_tree `714`'s only consumer is `cell_len(...)` at `:1304`; the tooltip itself is `Text(data.raw_label)` at `:1302`. transcript `856`/`864`/`2720` read `.plain` off a `Content` this module assembled from literal `(text, style)` tuples (`Content.assemble` never markup-parses, so the round trip is exact); `2096` is the selection domain, built the same way.
- **The codebase already knows the double-escape trap and fixed it.** `console_workspace_context.py:358-385 _marker_aware_tooltip` escapes the **fully assembled** sentence exactly once and documents the `Content.from_markup("...[saved]...").plain` word-loss it was closing — the sibling P1's shape, caught here.
- **Unescaped `color` interpolated into Rich markup is safe.** `console_session_switcher_modal.py:83-109` (`f"[{color}]{escaped}[/] "` → `Text.from_markup` at `:1209`) and `console_workspace_context.py:1257` (`f"[{color}]{label}[/]"` → `Text.from_markup` at `:1260`). `Chat/console_appearance.py:120-128` raises unless `is_valid_console_appearance_color` (canonical `#rrggbb`), and the read path `parse_console_conversation_appearance` (`:131-144`) sanitises a malformed stored color to `None`. `UI/Console_Modules/workspace.py:2664-2667` is the only producer. Titles/metadata are separately `escape_markup`-ed.
- **All 16 `run_worker` calls in the slice carry `group=`.** AST sweep over every `run_worker(...)` call expression printed zero rows without a group.
- **All 12 function-body imports resolve**, attributes included (`probe_fbimports.py` → 15/15 `OK`, counting the stdlib ones).
- **The 10 `refresh(recompose=True)` sites are the legitimate case.** Each is equality- or signature-guarded on a tray/row widget whose child *count* varies with state: `console_workspace_context.py:741` behind the six-condition `_can_skip_recompose` + the `_ComposeReadView` read-set guard (`:745-829`), `:1034` behind width hysteresis (`_should_relabel_at_width`); `console_staged_context.py:287`, `console_settings_summary.py:308`, `console_staged_evidence_strip.py:109`, `console_workspace_details.py:67` all behind `if state == self.state: return`; `console_transcript.py:1603`/`1611` recompose only a ~3-widget header slot when its control set changes, `:2567` the empty panel.
- **Mutable class attributes on widgets.** `console_workspace_tree.py:272`/`:278` (`_tooltip_memo_key`, `_projection_memo`) and `console_workspace_context.py:592-604` (`_composed_row_signature`, `_composing_row_signature`, …) are class-level defaults, but every write is a *rebind* to a fresh immutable tuple/None, never an in-place mutation, and both files document why the class-level default is required (a watcher can fire inside `Tree.__init__` before instance attributes exist). `console_transcript.py`'s per-instance dicts are all set in `__init__`.
- **`id()`-keyed dicts in `console_transcript.py`** (`:4028-4030`, `:4990-4992`) are rebuilt from live children inside the same function call and never outlive it — not recompose-lifetime caches.
- **`console_workspace_tree.py:846-885 _move_node_preserving_identity`** uses Textual private shape, but is version-pinned to 8.2.8, validates six attributes, and raises on any mismatch — explicitly sanctioned by `backlog/decisions/083-console-edge-rails-and-workspace-tree-ownership.md`. **Out of scope by ADR 083.**
- **The transcript reconciler is genuinely incremental** (per the brief, not re-derived): `_reconcile_rows` (`:7074-7181`) diffs by row key + signature, syncs in place where it can, batches removals into one `remove_children`, and only `move_child`s rows not already at their index.
- **`console_turn_file_card.py` and `console_workspace_files_modal.py` do their blocking reads correctly** — every provider/filesystem call is inside `asyncio.to_thread` (`turn_file_card:435, 629, 1175, 1226`; `files_modal:205`), with generation/token fencing on publish.

## Retired
Beyond the candidate table above, these I raised and then retired with evidence:
1. **"The 0.25 s settings poll is the sibling's 30-49 ms/tick idle poll."** Symptom real (there *is* an unconditional 4 Hz poll that rebuilds a full draft), cause wrong — measured at 0.104 ms/tick. Retired.
2. **"`_poll_subscription_readiness` computes the expensive readiness before its cheap revision gate."** True as written (`:6496-6498`), but at 0.104 ms the reordering buys nothing. Retired as a finding, noted as a shape.
3. **"`_switcher_icon_prefix` / `_conversation_appearance_button` allow markup injection through an unvalidated colour."** Symptom plausible, cause wrong — validated at both the construction and the parse boundary. Retired (evidence in Verified-fine).
4. **"`console_workspace_tree.py:714 node.label.plain` un-escapes a user conversation title."** Labels are `Text(...)` objects, never markup-parsed, and the value feeds `cell_len` only. Retired.
5. **"`console_settings_modal.py:7251 int(custom_text)` is an unguarded `int()` on user input."** Both callers of `_build_context_policy_overrides` wrap it (`:2929-2934`, `:4496-4501`) in `except (ContextPolicyError, ValueError)`; `grep -n "_build_context_policy_overrides"` shows exactly those two plus the definition. Retired.
6. **"`_OperationLane` leaks bare `asyncio.create_task`s like the switcher does."** It does create them (`:200`), but `_teardown` (`:1118-1126`) closes all three lanes and `on_unmount` (`:1134-1145`) awaits it on every pop path. Retired — the switcher finding stands because it has no such teardown.

## Left UNVERIFIED
| claim | why not verified | literal command to run |
|---|---|---|
| The P1 workspace-switcher cost is *visible* as a pause on a real profile (how many workspaces a real user has) | requires running the app; forbidden by the brief | `tmux -L verify new-session -d -s c 'cd <worktree> && TLDW_CONFIG_PATH=<scratch profile> .venv/bin/python -m tldw_chatbook.app'` then open Console ▸ rail ▸ **Switch**, `tmux -L verify capture-pane -p -t c` (recipe: `.claude/skills/verify/SKILL.md`) |
| The P1 turn-file-card cost is visible as a hitch when resuming a long agent session | same | as above, resume a conversation with ≥20 file-change turns and watch the transcript's first paint |
| The switcher's bare `asyncio.create_task` actually survives a pop in practice | no probe written (P2, inferred) | a probe pushing `ConsoleSessionSwitcherModal` with a blocking `character_activate`, popping the screen, then asserting `modal._activation_task.cancelled()` |
| Whether `Workspaces/file_inspector.py`'s unguarded `os.close`/`os.fstat` (`:328`, `:331`) can raise on a shipped path, which would promote the Workspace Files finding to P1 | outside my slice | `cd <worktree> && source <SCRATCH>/env.sh && $PY -m pytest Tests/Workspaces/ -q` plus a read of `_open_root_descriptor`/`_open_target_descriptor` |

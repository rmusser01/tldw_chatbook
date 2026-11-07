# S19 — UI/Screens B (every remaining screen)

**Coverage:** files read in full: 35 | sampled: 8 | mechanical only: 25 (of 68).
Full includes `scheduling/schedules_workbench.py` (5433), `backup_restore_screen.py` (2595),
`scheduling/task_detail.py` (1857), `artifacts_screen.py` (1712), `meetings_screen.py` (1606), and 30 more.
Sampled: `workflows_screen.py` (1–700 of 1339), `home_screen.py` (330–929 of 1075), `skills_screen.py`
(~100–440 of 1338), `settings_endpoint_probe.py`, and 4 settings modules.
Mechanical sweeps run over the **whole** slice: `run_worker`/`@work(exclusive=)` without `group=`; mutable
`reactive([])`/`reactive({})` literals; `re.compile` in function bodies; post-`await` DOM lookups including the
`finally`/`except` blind spot; `.renderable`/`.plain` read-backs; dotted `get_cli_setting`.

## Findings

### P1 [D1] — `StatsScreen` crashes when a chat topic contains a non-ASCII word
- Where: `UI/Screens/stats_screen.py:590` (`Container(classes="topic-bar-fill", id=f"bar-{self.topic}")`), reached
  from `:935` (`TopicBar(topic.capitalize(), count, max_count)`); topic source `Stats/user_statistics.py:370-446`.
- Evidence: `textual/css/tokenize.py:32` → `IDENTIFIER = r"[a-zA-Z_\-][a-zA-Z0-9_\-]*"` (ASCII).
  `Container(id='bar-Über')` → `BadIdentifier`; `'Programming'` → OK. Topic words come from
  `re.findall(r"\b\w{4,}\b", content.lower())` filtered by `word.isalpha()` — **both Unicode-aware**:
  `re.findall(r'\b\w{4,}\b','über künstliche')` → `['über','künstliche']`, both `.isalpha()`. Any such word with
  `count > 5` becomes a topic (`user_statistics.py:444-446`).
- Why it matters: the whole Stats destination dies on mount for any user whose chats contain
  German/French/Spanish/Cyrillic/CJK words. `refresh_stats_display`'s `try/except` does **not** cover it, because
  the id is built inside `TopicBar.compose()`, which Textual runs asynchronously after `parent.mount(...)` returns.
- Recommended correction: gate the id the way this repo already gates the identical class —
  `Audio/meeting_session.py:367 is_widget_safe_cluster_id`, whose docstring says verbatim "a value with a space or a
  '#' would raise out of `compose()` and take the screen down". Either skip the bar or use `id=f"bar-{index}"`.
  Canonical home for the predicate: `Utils/` (it is currently reachable only via `Audio/`).
- Size: S · ADR: no · Confidence: verified (mechanism executed; reachability inferred from the topic-extraction code
  path, not executed end-to-end)
- Already covered: none (TASK-32800 crashes .1–.5 are all Done and none names Stats)

### P1 [D1] — The Settings/Console endpoint probe's chat path reaches the wire with the user's `base_url` **and** the resolved API key, bypassing `Utils/egress.py`; the TTS path in the same function does not
- Where: `UI/Screens/settings_endpoint_probe.py:607-641` (`probe_settings_endpoint` chat branch → `_request_models`),
  vs `:514-517` (`_probe_openai_tts_catalog` →
  `check_url_or_raise_async(..., trusted_origins=origin_set(endpoint.origin))`). Credential attach: `:294-295`
  `headers["Authorization"] = f"Bearer {api_key}"`.
- Evidence: `grep -n "check_url_or_raise_async" settings_endpoint_probe.py` → **one** hit, line 514 (TTS only).
  Callers do not compensate: `grep -rn "probe_settings_endpoint"` → `UI/Speech/speech_catalog_mixin.py:145`,
  `UI/Screens/chat_screen.py:3241`, `UI/Screens/settings_screen.py:15174`,
  `UI/Wizards/FirstRunSetupWizard.py:934`; none of the four contains `check_url_or_raise` / `egress`.
- Why it matters: the repo's own copy at `settings_screen.py:18018` states neighbouring fetches "go through the
  egress SSRF" check. Here a `base_url` of `http://169.254.169.254/...` or `http://127.0.0.1:<admin-port>/v1/models`
  is contacted **with the provider credential attached**. Request-level `follow_redirects=False` (`:302`) and the
  origin re-check at `:426` block redirect pivots, so the exposure is the *initial* URL only.
- Recommended correction: call `check_url_or_raise_async(resolution.models_url,
  trusted_origins=origin_set(origin_of(resolution.models_url)))` before building the client in the chat branch,
  matching the TTS branch exactly. Local providers already work under that shape because the TTS branch trusts the
  configured origin.
- Size: M (the `trusted_origins` argument must be chosen so loopback local providers still pass) · Confidence:
  verified (read + grep)
- Already covered: none — TASK-32806 .1–.8 name credentials, MCP permission files, calculator bounds, local servers,
  file size/atomicity, secrets-to-disk and image bombs; **not egress**.
- **P0 if a non-user-authored config path (imported profile, chatbook, restore) can set `base_url`** — see UNVERIFIED.

### P1 [D1] — `SchedulesWorkbench._run_sync`'s `finally` dereferences the DOM after an awaited network sync, in a worker with `exit_on_error=True`
- Where: `UI/Screens/scheduling/schedules_workbench.py:5429-5432`; worker dispatched at `:5393`
  `self.run_worker(self._run_sync, exclusive=True, group="schedules-sync-now")` (no `exit_on_error=False`).
- Evidence: the `await service.sync_now(owner_id)` is at `:5408`; `finally:` re-enables
  `#scheduling-owner-local` / `#scheduling-owner-server` with two **bare** `query_one` calls.
  `textual/widget.py:4849-4851` — `Widget._on_unmount` calls `self.workers.cancel_node(self)`;
  `textual/app.py:2869` — a non-installed screen is `await screen.remove()`d on switch (`"schedules"` is **not**
  `reusable=True` in `screen_registry.py`), so navigating away mid-sync cancels the worker and the `finally` runs
  against a torn-down tree. `textual/worker.py:375-384` — a non-`CancelledError` exception with `exit_on_error` True
  calls `app._handle_exception(WorkerFailed(...))` → panic.
- Why it matters: press `s`, navigate away while the server round-trip is in flight → `NoMatches` out of the
  `finally` **replaces** the `CancelledError` and takes the app down.
- Recommended correction: wrap the `finally` body in `try/except Exception`, or pass `exit_on_error=False` at
  `:5393`. The `try` at `:5400-5401` (pre-await) is fine as is.
- Size: S · Confidence: **inferred** (traced through Textual 8.2.8 source; not executed — see UNVERIFIED)

### P2 [D3] — `scripts/check_textual_worker_contract.py`'s W002 guard counts a `query_one` in a `finally:`/`except:` body as "guarded"; 59 repo-wide sites are invisible to both the census and the gate
- Where: `scripts/check_textual_worker_contract.py:216-224` — the parent walk breaks on the *first* `ast.Try`
  ancestor regardless of which section the node is in. **A `finally` body is not protected by that statement's own
  handlers.**
- Evidence: a script mirroring `collect_w002` but classifying the section and requiring an ancestor `Try.body`
  **with handlers** → `TOTAL_FALSE_GUARDED 59` repo-wide; 11 under `UI/Screens/` outside Tier-1, 5 in this slice:
  `scheduling/schedules_workbench.py::_run_sync` (`finally` — the P1 above), `chunking_lab_screen.py::_load` ×2
  (`except`), `workflows_screen.py::_initialize_authoring` ×2 (`except`).
  `scripts/textual_await_dom_census.tsv` has **no row** for `_run_sync`.
- Why it matters: the guard exists because exactly this defect class took the app down three times (its own header,
  TASK-32800.4). The `finally` shape is the most dangerous variant — it is the one that runs during cancellation.
- Recommended correction: in `collect_w002`, keep climbing past a `Try` whose descendant chain enters
  `finalbody`/`handlers`/`orelse`; only a `Try.body` with non-empty `handlers` counts as guarded. Re-pin the census
  (the 59 new rows are a baseline, not a gate failure).
- Size: S · Confidence: **verified** (script output)

### P2 [D1/D2] — `SchedulesWorkbench._settle_orphaned_transfers` runs a synchronous DB sweep on the event loop, while its docstring claims it was moved off it
- Where: `scheduling/schedules_workbench.py:4684-4699`: `async def _settle():
  service.sync_engine._settle_orphaned_transfer_mutations(target_owner)` dispatched via `self.run_worker(_settle,
  exclusive=True, group="schedules-orphan-sweep")` — **no `thread=True`, no `asyncio.to_thread`**.
- Evidence: `Scheduling/services/sync_engine.py:173 def _settle_orphaned_transfer_mutations(self, target_owner)` — a
  plain `def` doing `get_pending_mutations`/`set_transfer_state`. The docstring at `:4669-4676` claims "this used to
  run the local DB work inline and synchronously from `on_mount`, blocking mount on it. Deferred onto the same
  fire-and-forget worker path…". **Every other `service.db.*` read in this file uses `asyncio.to_thread`**
  (`:1358, 1421, 3317, 3621, 3998, 4350`) and the module calls that "the same 'local DB read, off-thread' discipline
  every `service.db.*` read here uses" (`:1305-1307`).
- Why it matters: `run_worker(coroutine)` is not a thread. Only the *ordering* moved; the sweep still blocks the loop
  on every mount of the Schedules screen, proportional to the pending-mutation count. **The comment actively
  misleads the next reader.**
- Size: S · Confidence: verified (read both sides)
- Already covered: adjacent to TASK-32804.12 (To Do) — **this specific site is not named there**, and the
  docstring's false claim is new information.

### P2 [D2] — `BackupRestoreScreen`'s 0.2 s poller re-`update()`s a `Static` with identical text 5×/s for the life of the screen, and never pauses while covered
- Where: `UI/Screens/backup_restore_screen.py:504` (`self._poller = self.set_interval(0.2,
  self._refresh_status)`), `:1690` (`self.query_one("#backup-status", Static).update(text)` — unconditional).
- Evidence: `textual/widgets/_static.py:85-95` — `update(content, *, layout: bool = True)` ends in
  `self.refresh(layout=layout)`; **the default lays out**. The screen has no `on_screen_suspend`, and it pushes
  `FileOpen`/`FileSave`/`SelectDirectory` over itself (`:2549, 2556, 2583`), so the poller ticks under an opaque
  screen — the exact TASK-23022 rule (`lessons-textual.md:253`). Once any operation exists, `_refresh_status` never
  early-returns again (`:1606` only guards `current is None`).
- Why it matters: 5 layout passes/second painting an unchanged string, plus an unsuppressed clock behind a modal.
  `SchedulesWorkbench` **in the same slice** already has both fixes: `_update_static_content` (`:624-628`, compares
  before updating) and `on_screen_suspend`/`on_screen_resume` timer pause (`:1174-1200`).
- Size: S · Confidence: verified
- Pinning test: `Tests/Architecture/test_progress_widget_clock_guard.py` guards progress widgets only, not screen
  pollers.

### P2 [D3] — A direct `await self.recompose()` bypasses `BaseAppScreen`'s focus-restore seam (the `refresh()` override), which is documented as "the ONE seam"
- Where in slice: `UI/Screens/workflows_screen.py:157`. Repo-wide: `UI/Library_Modules/
  library_unavailable_navigation.py:354,536`, `UI/Library_Modules/library_inspection_admission.py:403`,
  `UI/Screens/library_screen.py:11322,12403,21842,21939,22041`.
- Evidence: `base_app_screen.py:91-176` — focus identity is captured and re-posted **only** inside
  `refresh(..., recompose=True)` (`_focus_identity_for_recompose` at `:148`, restore queued at `:172`). The
  `recompose()` override at `:324-333` handles mouse capture but **never touches focus**.
  `textual/widget.py:1704-1716` — `Widget.recompose()` removes every child and restores no focus.
  `lessons-textual.md:494-517` (TASK-22281) documents the resulting app-wide keyboard soft-lock.
- Why it matters: the repo built one seam and then nine call sites route around it. In `workflows_screen.py` the
  impact is low (it runs from `on_mount`, before focus enters the screen); **the Library sites are Tier-1 and run
  from live handlers.**
- Recommended correction: move the focus capture/restore from `BaseAppScreen.refresh` into
  `BaseAppScreen.recompose` (where the mouse-capture release already is), so both entry points are covered by
  construction.
- Size: M · Confidence: verified

### P2 [D4] — `settings_image_gen_defaults.py` ↔ `settings_video_gen_defaults.py`: a whole parallel data layer, with behavioural drift in the coercion step
- Where: `settings_image_gen_defaults.py` (985) / `settings_video_gen_defaults.py` (439). Verbatim twins: `_spec_for`
  (`:168`/`:84`, DUP hash `bf8f30a89a72`), `canonical_backend_order` (`:371`/`:263`, hash `2d7c236cb311`). Shape
  twins: `FieldSpec`, `<X>GenBackendRow`, `<X>GenDraftValues`, `_GLOBAL_DRAFT_KEYS`, `diff_to_sections`
  (`:448`/`:308`, hash `603a79dcd4dd`), `_coerce_value`, `validate_draft`, `build_backend_rows`.
- Evidence: **Drift:** `_coerce_value` supports `int`/`float`/`origin` on the image side
  (`settings_image_gen_defaults.py:429-448`) and `int`/`bool` on the video side
  (`settings_video_gen_defaults.py:292-306`). Video's `FIELD_SCHEMA` declares
  `FieldSpec("allow_uploads", ..., "bool")` (`:70`); **the image copy has no bool branch at all.**
- Why it matters: this drift feeds *storage* — `diff_to_sections` writes the coerced value into `config.toml`.
  Adding a `bool` field to the image schema (or a `float`/`origin` field to video) silently persists a raw string
  where a typed value is expected, and `validate_draft` on the other side would not catch it.
- Recommended correction: one shared `UI/Screens/settings_gen_defaults_common.py` holding `FieldSpec`, `_spec_for`,
  `canonical_backend_order`, `_coerce_value` (union of all kinds), and the `diff_to_sections` skeleton parameterised
  by the section name and the global-key tuple. Both modules keep only their schema/labels/config adapters.
- Size: M · Confidence: verified
- Already covered: none — TASK-32808 `.1`-`.11` name byte-size, truncation, filename sanitizers, token estimator,
  boolean coercion, atomic write, scope-service, approval stamps, connection setup, verbatim copies, SQLite
  validator — **not this pair**.

### P2 [delete] — `UI/Screens/schedules_screen.py` (516 lines) is unrouted, has zero importers, and its stated reason for existing is false
- Where: the file's own docstring `:3-6`: "DEPRECATED … This screen is unrouted and retained only because existing
  tests still exercise it directly".
- Evidence: **importers = 0.** `grep -rn "schedules_screen" tldw_chatbook/` → only the file itself.
  `grep -rn "SchedulesScreen" tldw_chatbook/ Tests/` → **the class is never imported**; the three test hits are
  string literals (`Tests/test_application_state_ownership.py:222,1900` a path string in an AST inventory;
  `Tests/UI/test_consolidated_css_harness.py:293` a name in `_SPLIT_SHEET_OWNERS`;
  `Tests/QA/test_scheduling_css_tokens.py:69` a negative-selector list). **Reachability = none:**
  `screen_registry.py:121-125` maps route `"schedules"` → `scheduling.schedules_workbench.SchedulesWorkbench`, and
  `"schedules"` is not in `_SCREEN_ALIASES`.
- Why it matters: 516 lines of a second, divergent Schedules implementation that no test loads —
  `Tests/UI/test_schedules_terminology.py` sweeps only `schedules_workbench`/`task_detail`/`reminder_form`
  (`:227-229`), so this copy is not even held to the locked terminology.
- Recommended correction: delete the module; drop its row from `RECENT_WORK_SCREEN_PATHS` and the `"SchedulesScreen"`
  string from the two test inventories in the same commit.
- Size: M · Confidence: verified
- Already covered: none — TASK-32807 `.1` covers top-level *widget* modules, `.3` Event_Handlers, `.4` the
  Tools/Settings window, `.5` Chat modules, `.6` Utils/Widgets helpers. **No sub-task covers `UI/Screens/`.**

### P3 [D3] — `stats_screen.py`: stdlib `logging`, a dead expression in `compose()`, and a 3× redundant full rebuild
- Where: `:527, 548` (`import logging` / `logging.getLogger(__name__)` — the only screen in the slice not on loguru);
  `:585` (a percentage computed and discarded, recomputed at `:594`); `:756-772` (`watch_is_loading`,
  `watch_stats_data`, `watch_error_message` each `call_after_refresh(self.refresh_stats_display)`, and
  `_apply_statistics_result:644-652` sets all three in sequence → **three** queued `remove_children()` + full
  remounts for one result).
- Size: S · Confidence: verified

### P3 [D3] — `ChatbooksScreen`: three dead public methods and four write-only reactives (~45 of 91 lines)
- Where: `:317-320` (`current_chatbook`, `chatbook_list`, `is_editing`, `selected_chatbook_id`), `:348-384`
  (`create_new_chatbook`, `open_chatbook`, `delete_chatbook`).
- Evidence: `grep -rn "create_new_chatbook\|open_chatbook\|\.delete_chatbook(" tldw_chatbook/ Tests/` → zero hits on
  the screen (the `delete_chatbook` hits are all `service.`/`scope.` objects). `chatbook_list` is written at
  `:333`/`:346`/`:375` and read only by the dead methods.
- Why it matters: the screen keeps a shadow copy of the real list owned by `ChatbooksWindowImproved`;
  `delete_chatbook` mutates only the shadow, so if it were ever wired it would silently diverge. Separately,
  `on_screen_resume:344` calls `chatbook_window._refresh_chatbooks()` — a private method across a module boundary.
- Size: S · Confidence: verified

### P3 [D4] — `research_screen.py` / `writing_screen.py` are verbatim twins of the "screen wrapper with deferred restore_state" pattern
- Where: `research_screen.py:40-64` and `writing_screen.py:48-68` (`save_state`/`restore_state`), DUP_SHAPE hashes
  `6a29d29daff6` and `40e9993c628b`. Bodies differ only in the window class and the `#…-window` id.
- Recommended correction: a `HostedWindowScreen(BaseAppScreen)` mixin parameterised by `(window_id, window_type)`.
  Canonical home: `UI/Navigation/`.
- Size: S · Confidence: verified

### P3 [D3] — `task_detail.set_lifecycle_lock` parses its own output back out of a widget and drops it; plus a discarded `query_one`
- Where: `scheduling/task_detail.py:1750-1759` and `:1469`.
- Evidence: `grep -rn "scheduling-transfer-why" tldw_chatbook/` → only `task_detail.py` writes it (`:910` compose,
  `:1482` clear, `:1759` write). The comment at `:901-909` confirms the other writer (`set_transfer_reasons`) was
  retired. So `existing` is **always `[]`** after the filter — a read-modify-write preserving lines nothing can
  produce. `:1469` queries a Button and discards the result.
- Size: S · Confidence: verified
- Pinning test: `Tests/UI/test_schedules_transfer_actions.py:106,127` assert on `why.visual.plain`, not the round-trip

### P3 [D2] — `WorkbenchHostScreen._on_message` calls its `route_message` hook for **every** message, including `MouseMove`/`Update`/`Layout`
- Where: `scheduling/workbench_host_screen.py:90-100`; consumer `schedules_workbench.py:1943-1952` runs
  `isinstance(message, _PUSHED_DETAIL_MESSAGES)` against a **14-entry tuple**. The docstring acknowledges
  "deliberately unfiltered".
- Why it matters: a 14-way isinstance per mouse-move message while the pushed pane is open. Bounded, but it is
  per-input-event work on a host designed to relay ~14 message types.
- Size: S · Confidence: verified

### P3 [D2] — `SchedulesWorkbench` runs synchronous `service.db.*` reads on the event loop in five badge/overlay paths, against its own stated off-thread discipline
- Where: `:2484` (`_push_results_overlay` → `list_automation_results` + `count_automation_results` +
  `list_automation_definitions`), `:2544` (`_push_conflicts_overlay` → `get_conflicts`), `:4408-4414`
  (`_unread_result_ids` → `count_unread_results` + an **unbounded** `list_automation_results(limit=unread_total)`),
  `:4996`, `:5023`, `:4586`.
- Evidence: `load_tasks` (`:1358`) and every other `service.db.*` read in the file go through `asyncio.to_thread`,
  and `:1305-1307` calls that the file's discipline. `_refresh_results_surfaces` (`:5030-5047`) runs two of these
  after **every** results mutation.
- Size: M · Confidence: verified

### P3 [D3] — `MCPScreen.on_screen_suspend` mutates two private attributes of `MCPWorkbench`
- Where: `mcp_screen.py:822-823` (`self.workbench._mcp_recovery_busy` / `._mcp_recovery_token = None`).
  Recommended: a public `MCPWorkbench.invalidate_recovery_receipt()`. · Size: S · Confidence: verified

## Candidate triage
**RETIRED:** `query_one_in_timer_no_try` `backup_restore_screen:1690,1691` (unconditionally composed; poller stopped
in `on_unmount:509` — though the 5 Hz unconditional `update()` at the same lines **is** the P2 above) and
`skills_screen:247,248` (unconditionally composed; a `ModalScreen`'s interval dies with the dismiss);
`run_worker_coroutine` ×26 (every row is a coroutine object handed to `run_worker`, which Textual accepts; none runs
blocking sync work inside — **the one real instance was NOT in the candidate list**: `schedules_workbench.py:4684`);
`except_exception_pass` ×13 (all not-mounted-yet / diagnostics guards on display paths; none on a data-write path);
`except_exception_return` ×22 (the `settings_endpoint_probe` rows are the documented no-raise UI boundary; the rest
return last-good display state); `dotted_section_setting` `home_screen:805,820,823` — **`:823` is the 1-arg dotted
form** `get_cli_setting("home.rail_state")`, which is supported and returns the sub-dict; the broken shape is the
**2-arg** `get_cli_setting("a.b", key)`. `config.py:8447-8468` documents nested sections explicitly;
`plain_readback` `workflows_screen:371` (compared against ASCII literals; `.plain` cannot un-escape either);
`try_import_guard` ×7 (deliberate lazy imports with documented invariants; two pinned by
`Tests/Audio/test_meeting_import_safety.py`); `function_body_import` ×62 (sampled 18; all ADR-097 boot-census
deferrals or cycle breaks); `strftime` ×10 (all display strings; TASK-32803's scope is writers);
`legacy_markers` ×93 (partially — `schedules_screen.py:3` → the delete row; `tools_settings_screen.py` → already
TASK-32807.4; the ~80 `scheduling/*` rows are historical comments on live code).
**CONFIRMED:** DUP `ec135946f6e9` `_column_divider` ×4 (`skills_screen:791`, `schedules_screen:185`,
`acp_screen:105`, `settings_screen:11037`, byte-identical); DUP `6ff1d2c5de8b` `_format_clock` ×2
(`video_player_screen:67` ↔ `Widgets/Console/console_video_preview.py:65`, verbatim); the image↔video gen hashes
(with drift); the research↔writing state hashes; `_maybe_await` ×60 (one slice member).
**UNVERIFIED:** `raw_1024x1024` ×3 (`chunking_lab_screen:1108`, `model_browser_state:389,394` — mechanical-scan
files only); DUP `70afb437f576`/`2d8f77ac838f`/`ca16269a813d`/`ccefb30dabb7` (`automation_definition_form` ↔
`reminder_form`, both mechanical-scan only).

**Retired findings (symptom real, cause wrong) — reported per rule 6:**
- `scheduling/task_detail.py:1754 str(why.renderable)` → **not** an `AttributeError`. `Static` in Textual 8.2.8
  genuinely has no `renderable` (`hasattr(Static(''),'renderable')` → `False`), but
  `tldw_chatbook/__init__.py:78-89 _install_textual_compatibility_shims()` re-adds it as
  `property(lambda self: self.content, ...)`, invoked from `UI/__init__.py:5` and `Widgets/__init__.py:5` — so
  importing anything under `UI/` installs it. The *dead read-back* survives as a P3.
- `meetings_screen.py:1581` calling `self._owner._submit_on_ui_thread(...)` from inside `@work(thread=True)` →
  **correct, not a thread-safety bug**: `Audio/meeting_owner.py:1167-1168` — the method *is* the marshal
  (`return self._call_from_thread(self._submit_ingest, **kwargs)`). The name reads as a precondition; it is a promise.
- `artifacts_screen.py` whole-screen `refresh(recompose=True)` (9 sites) orphaning `app.focused` → **handled** by
  `BaseAppScreen.refresh` (`base_app_screen.py:91-176`, task-31946). The cost angle stays P3 at most.
- `run_worker(exclusive=True)` without `group=`, mutable `reactive([])` literals, and `re.compile` in function
  bodies → **zero occurrences** across all 68 slice files (AST sweep).

## D4 observations for repo-wide Phase 3
1. **`_maybe_await` — 60 verbatim copies**, one slice member (`provider_model_resolution.py:168`). No shared helper
   exists. Canonical home: `Utils/async_compat.py`. Byte-identical, so drift risk is currently nil.
2. **`_column_divider` ×4**, byte-identical. **Helper should exist and almost does**:
   `Widgets/destination_workbench.py` already owns `DestinationModeStrip` for this exact family.
3. **Image-gen / video-gen settings data layer** — the only cluster in this slice with *behavioural* drift reaching
   storage. Not a candidate for mechanical dedupe; needs the `_coerce_value` union decided first.
4. **"Skip an identical `Static.update()`"** — `SchedulesWorkbench._update_static_content` (`:624-628`) is the only
   implementation; `backup_restore_screen.py:1690`, `task_detail.py:1761-1764`, `stats_screen`,
   `meetings_screen._tick` all call `update()` unconditionally on repaint paths, and
   `meetings_screen._set_live_labels` (`:1078-1083`) and `left_rail.py:1735` re-roll the same idea differently.
   ~6 re-rolls in this slice alone. Canonical home: a `Widgets/` helper or a `Static` subclass.
5. **Textual-safe-identifier predicate** — `Audio/meeting_session.is_widget_safe_cluster_id` (`:367`) exists and is
   used **once**; `stats_screen.py:590` needed it and doesn't have it (P1). *Helper exists, wrong home, ignored.*
6. **`Static.renderable` compat shim** — `tldw_chatbook/__init__.py:78-89` monkeypatches it back onto Textual 8. Its
   getter returns `self.content` — the **source** object, not the rendered text. **25 sites repo-wide read through
   it**, and their correctness depends on the last `update()` having been passed a `str`. Worth a Phase-3 census:
   `str(widget.renderable)` on a site last updated with a `Text`/`Content`/Rich renderable silently yields `.plain`
   or a `<object at 0x…>` repr.

## Deletion-candidate rows
| Module | Importers (prod) | Reachable from shipped entry point? | Pinning tests | Row |
|---|---|---|---|---|
| `schedules_screen.py` (516) | **0** | No — `"schedules"` → `SchedulesWorkbench` | none import the class; 3 string-literal inventories | **delete** |
| `tools_settings_screen.py` (49) | 1 (lazy map) | No — `"tools_settings"` → `MCPScreen` | 2 test files push it | **keep** — owned by TASK-32807.4; CLAUDE.md says it stays untouched |
| `skills_screen.py` (1338) | 2 | No — `"skills"` → `"library"` | 12 test files; two modals reused by the Library skills editor | **keep** (`screen_registry.py:239-243` states this) |
| `media_screen.py` (163) | 1 | No — the `"media"` alias precedes the route in `_lookup_route` | 5 test files pin `save_state`/`restore_state` | **keep** (`screen_registry.py:266-278`) |
| `library_conversations_screen.py` (29) | 1 | Route `"conversation"` registered and not aliased, but the body is three `Static`s with no behaviour | 0 test files | **unknown** — check whether any nav surface emits `NavigateToScreen("conversation")` |
| `image_gen_demo_screen.py` (138) | 1 (command palette only) | Yes, via the palette | `Tests/Image_Generation/test_demo_screen.py` | **unknown** — self-declared "THROWAWAY … Replaced by the real chat card in Phase 2"; confirm Phase 2 shipped |

## Left UNVERIFIED
| Claim | Why | Command |
|---|---|---|
| `SchedulesWorkbench._run_sync`'s `finally` actually raises `NoMatches` (P1) | **could not execute: the ADR-126 recovery gate blocks this worktree's UI tests** (`Backup_Recovery/raw_participants.py:127: RecoveryRequired` during fixture setup). Traced through Textual source only. | in a recovered profile: `PYTHON=<venv> pytest Tests/UI/test_schedules_workbench.py -q` after adding a pilot test that starts `action_sync_now` against a slow `sync_now` double and pops the screen mid-await |
| `StatsScreen` crash end-to-end (P1) | `TopicBar.compose()` uses `with Horizontal(...)`, which needs an active app, so the out-of-app repro stops at `NoActiveAppError`. `Container(id='bar-Über')` → `BadIdentifier` **is** verified. | `pytest Tests/UI/ -q` with a harness App mounting `TopicBar('Über', 12, 12)`; or seed a chat message containing "über" ×6 and open Stats |
| Whether the missing egress check is reachable from a **non-user-authored** config — determines P1 vs **P0** | confirmed the four call sites pass a `base_url` read from config/Settings input; did not trace whether chatbook import / backup restore / profile import can write `api_settings.*.base_url` or `custom_endpoints.*` | `rg -n "base_url" tldw_chatbook/Chatbooks/ tldw_chatbook/Backup_Recovery/ tldw_chatbook/UI/Library_Modules/library_export_controller.py` |
| `raw_1024x1024` rows at `chunking_lab_screen:1108`, `model_browser_state:389,394` | mechanical-scan files only | `sed -n '1100,1115p' …/chunking_lab_screen.py; sed -n '383,400p' …/model_browser_state.py` |
| `automation_definition_form.py` ↔ `reminder_form.py` duplication extent (4 DUP hashes) | both mechanical-scan only | `pytest --collect-only -q Tests/UI/test_schedules_forms*.py` then diff the four named function pairs |
| The 25 mechanical-only files contain no D1 | swept for `run_worker` grouping, reactive defaults, `re.compile`, post-await DOM lookups (incl. the `finally`/`except` variant), `.renderable`/`.plain`, dotted config reads — all clean — but not read for logic errors | read them |

# S18 — UI/Screens A (the large screens)

**Coverage:** files read in full: 1 | read by symbol cluster: 6 | sampled: 2 | mechanical only: 2 (of 11).
Full: `model_external_view.py` (242). By symbol cluster (full AST method/line map built first, then the named
clusters read): `watchlists_collections_screen.py` (~1,200 of 14,319), `llm_screen.py` (~350 of 5,180),
`change_review_screen.py` (~450 of 4,967), `evals_screen.py` (~650 of 3,450), `trajectory_screen.py`
(~1,450 of 2,075), `research_workspace_screen.py` (~750 of 2,006). Sampled: `model_installed_view.py` (~150),
`settings_speech_tts.py` (~300 + full AST map). Mechanical only: `model_remote_view.py`, `model_curated_view.py`.
**The reviewer explicitly did not read 14k lines of `watchlists_collections_screen.py`; every claim below about it is
anchored to a line range actually opened.**

## Findings

### P1 [D1] — `evals_screen.py` still uses `rich.markup.escape`, which does not escape `[TODO]`-shaped brackets, so a bench named `[TODO] Q3 plan` loses that token from the Run button, its tooltip and the delete-confirmation dialog
- Where: `UI/Screens/evals_screen.py:38` (`from rich.markup import escape as escape_markup`), used at `:2536`
  (ConfirmationDialog message), `:3223`, `:3258`, `:3320` (Button label + tooltip).
- Evidence: **re-verified by the lead** — see `phase4-verification.md` "S18-P1a" for the runnable repro and the
  nine-file classification table. `rich.escape` → `'Run  Q3 plan'`; `Utils.input_validation.escape_markup` →
  `'Run [TODO] Q3 plan'`. #2717 **is** in this tree (`Widgets/Library/library_rail.py:8` imports the repo escaper,
  and TASK-32802.4 markers exist at 3 sites), so this is a missed site, not an unshipped task. Six of the nine
  remaining `rich.markup.escape` importers are on markup-ON surfaces.
- Why it matters: TASK-32802.1's AC#2 is "the correct escaper is used at every site that currently calls
  `rich.markup.escape` for a Textual surface"; it is not satisfied on dev. A user-named bench silently loses text in
  a **destructive confirmation dialog**.
- Size: S · ADR: no · Confidence: **verified**
- Already covered: **TASK-32802.1 — but insufficient as shipped.** The "cluster grew / sites missed" case.

### P1 [D1] — `change_review_screen.py` puts raw git stderr into `notify()` with markup ON: a repo path like `src/[id]/page.tsx` loses that segment, and a message containing `[/` crashes the app in the toast renderer
- Where: `UI/Screens/change_review_screen.py:2897` (`_land_commit_error`), `:3224` (`_land_push_error`),
  `:2884`/`:3211` (the refusal twins), `:2766`, `:3338`. Source: `:2856`
  `_land_on_ui(app, self._land_commit_error, token, str(exc))` inside the `_commit` thread worker; same at `:3159`.
  **36 dynamic `notify()` calls in this file carry no `markup=` argument.**
- Evidence:
  `Content.from_markup("fatal: pathspec 'src/[id]/page.tsx' did not match")` → `"fatal: pathspec 'src//page.tsx'
  did not match"`; `Content.from_markup('error: failed [/]')` → `RAISES MarkupError: auto closing tag ('[/]') has
  nothing to close`. `Toast.render()` (`textual/widgets/_toast.py:105-118`) calls `Content.from_markup(...)` when
  `notification.markup`, so the `MarkupError` is raised inside the render cycle. **The repo has already ruled on
  this exact class:** `evals_screen.py:1540-1548` says "unbalanced markup (a bare `[/]`) … takes down the whole app"
  (task-1476), and `evals_screen.py` now has **zero** unescaped dynamic notifies.
- Why it matters: a failed `git commit`/`git push` in Change Review is a routine outcome; a bracket in the path or
  error text either deletes information from the only message the user gets, or exits the app while a commit is
  half-done.
- Recommended correction: `markup=False` on the six sites that interpolate engine/exception text — the convention
  `watchlists_collections_screen.py` already follows. Same fix for `llm_screen.py:2055, 4692, 1571`, where `error`
  comes from `Model_Artifacts.acquisition`.
- Size: S · Confidence: **verified** for the renderer behaviour and the text's provenance; **inferred** that a real
  git error contains `[/` (a repo path with `[…]` is the likelier trigger and is the silent-deletion variant)
- Already covered: none. TASK-32802.1/.4/.5 all concern *wrong* or *double* escaping; these sites apply **no**
  escaping and pass no `markup=False`.

### P1 [D1] — Three watchlists write workers have no error handling at all and are dispatched with `exit_on_error` at its app-exiting default, in a file where every sibling write worker guards and toasts
- Where: `UI/Screens/watchlists_collections_screen.py:14213` → `_mark_all_read_worker` (`:14217-14252`, body has no
  `try`); `:8111` → `_mark_notification_read` (`:8117-8122`); `:8129` → `_dismiss_notification` (`:8135-8140`).
  The awaited calls are real sqlite writes: `NotificationsInboxController.mark_read`
  (`UI/Subscription_Modules/notifications_inbox_controller.py:129-142`) ends in
  `mark_method(int(notification_id), is_read=is_read)`; `_mark_all_read_worker` does
  `await self._controller.mark_all_read(...)` (a scoped bulk UPDATE) as its first statement.
- Evidence: AST sweep of every `run_worker` call in the slice, resolving the target method and checking for a
  full-body `try`: `watchlists_collections_screen.py: run_worker WITHOUT exit_on_error = 62` — the only
  `UI/Screens` file at 0/62. Of those, targets with **no `try` at all**: the three above plus
  `_open_kept_briefings_modal(9537)`, `_open_briefing_preset_manager(10481)`, `_present_cached_items_page(12911)`,
  `_recover_local_read(5565,5763)`, `_load_runs(5510)`. `textual/worker.py:374-384`:
  `except Exception … if self.exit_on_error: app._handle_exception(WorkerFailed(...))`. Contrast inside the same
  file: `_toggle_star_worker` (`:14008`), `_toggle_briefing_queue` (`:13198`), `_save_rule`, `_delete_source`,
  `_refresh_all_worker` all wrap and `notify`.
- Why it matters: pressing `a` (mark-all-read) on a locked or contended database **exits the app** rather than
  toasting. Secondarily, `mark_read`/`dismiss` return `False` on policy-denial or a missing store, and the screen's
  `if updated:` (`:8120`, `:8138`) then silently does nothing — the press has no visible effect and no message.
- Recommended correction: give the three bodies the `try/except → notify` shape the file's other ~30 write workers
  already use. Do **not** blanket-add `exit_on_error=False` — repo-wide the omission is the majority (265 vs 87), so
  the guard belongs in the body.
- Size: S · Confidence: **verified** (dispatch sites, bodies, and Textual source all read)
- Already covered: none. TASK-32800.4's `scripts/check_textual_worker_contract.py` gates W001 (sync target) and W002
  (post-await DOM) only — it does not see an unguarded worker body.

### P2 [D1] — `research_workspace_screen.py` omits `exit_on_error=False` on exactly the one worker whose body holds all 8 of the file's census-recorded post-await unguarded DOM lookups, plus an unbounded modal await in the middle of them
- Where: dispatch `:456-461` (`_start_catalog_refresh`); body `_refresh_workspace_catalog` `:463-467` →
  `_apply_catalog_state` `:469-565`. Inside: `:484` `query_one(ResearchQuickNotesSection)` (already post-await),
  then `:485-487` `async with self._quick_note_lock: await self._flush_quick_note_before_owner_switch()` — which can
  `await self.app.push_screen_wait(ResearchNoteSwitchRecoveryModal())` (`:722-731`) for **unbounded user time** —
  then `:490, 492, 493, 504, 508, 523, 527, 558` all unguarded `query_one`.
- Evidence: `grep -E "research_workspace_screen" scripts/textual_await_dom_census.tsv` → `::_apply_catalog_state 8`,
  and those rows are the **only** census rows for this entire slice. The file sets `exit_on_error=False` on 5 of its
  8 `run_worker` calls (all behind `_guard_*` wrappers); `:457`, `:1060`, `:1631` omit it. The route is **not**
  `reusable` (`UI/Navigation/screen_registry.py:205-209`), so navigation unmounts the screen. The only mid-body
  recheck is `controller.is_current_catalog_state(state)` — a controller-state check, never `self.is_mounted`.
- Why it matters: identical shape to the `mcp_inspector.py` P0 that `check_textual_worker_contract.py`'s own
  docstring records, on a path started from `on_mount` — "open the tab, immediately switch away" is realistic.
- Size: S · Confidence: **inferred** — census rows and dispatch asymmetry verified; the unmount-vs-resume race not
  reproduced (see UNVERIFIED)
- Already covered: the census rows exist under TASK-32800.4 but are explicitly "a baseline, not an endorsement … not
  been individually reviewed".

### P2 [D2] — The watchlists item-status write does a triple hop (thread → throwaway event loop → second thread) because the screen's own `to_thread` wrapper is now redundant with the service's `run_db_off_loop`, added nine days later; the wrapper's docstring states a premise that is no longer true
- Where: `UI/Screens/watchlists_collections_screen.py:13442-13509` (`_update_item_status_off_loop`), specifically
  `:13499-13508`: `def _invoke(): return asyncio.run(self._controller.update_item_status(...))` then
  `return await asyncio.to_thread(_invoke)`.
- Evidence: all four layers read — `WatchlistsBackendController.update_item_status`
  (`UI/Watchlists_Modules/watchlists_backend_controller.py:475-499`) → `WatchlistScopeService.update_item`
  (`Subscriptions/watchlist_scope_service.py:515-557`) → `LocalWatchlistsService.update_item`
  (`Subscriptions/local_watchlists_service.py:966-1010`), whose last line is
  `await run_db_off_loop(db, db.mark_item_status, row_id, normalized_status)` → `Subscriptions/db_offload.py:70-77`
  `await asyncio.to_thread(run_and_close)`. History:
  `git log --diff-filter=A -- Subscriptions/db_offload.py` → `668ac79791 2026-08-11`;
  `git log -S"_update_item_status_off_loop" | tail -1` → `2d4ae86506 2026-08-02`. The docstring at `:13446-13457`
  asserts "every layer of that call chain is an `async def` with no genuine `await` of its own" — **falsified nine
  days after it was written**.
- Why it matters: each `j`/`k` in the reader fires mark-read-on-open (`_mark_item_read_on_open`, `:12375`), so every
  keystroke in a reading run now costs one default-executor thread, one `asyncio.run` (new event loop **and** a new
  `ThreadPoolExecutor`, both torn down), a second thread, and a fresh sqlite connection opened and closed. The drain
  invariant the docstring defends is provided by `_drain_item_status`, not by this wrapper.
- Recommended correction: `return await self._controller.update_item_status(...)`; delete `_invoke`; correct the
  docstring. **Leave `_toggle_briefing_queue` (`:13223`) alone** — that one calls `db.set_item_briefing_queued`
  directly and its `to_thread` **is** load-bearing.
- Size: S · Confidence: **verified** by trace + git history
- Already covered: none — TASK-32804.12 is "remaining sync work on the loop"; this is the inverse, a redundant offload.

### P2 [D2] — The skill-eval run worker writes ~66 individually-committed sqlite rows in a tight loop on the event loop
- Where: `UI/Screens/evals_screen.py:2090 _run_skill_eval_worker`, dispatched at `:1949-1953` as a **coroutine**
  worker, not `thread=True`. The loop: `:2197-2199` and `:2200-2222`. Also synchronous on the same path:
  `load_skill_eval_bench` (`:2143`), `create_skill_eval_run` (`:2159`), `db.update_run` (`:2165, 2235, 2245`),
  `save_report` (`:2225`).
- Evidence: `Evals/skill_eval/storage.py:129-148 save_artifact` → `db.store_result`, which opens
  `with self.connection() as conn:` then `with conn:` — **one committed transaction per call**. Cell count:
  `Evals/skill_eval/models.py:108 deep_sim_total: int = 50`, plus 16 judge cells (`runner.py:48-56`).
- Why it matters: at the end of every deep run the UI freezes for ~66 serial sqlite commits. The same shape
  (smaller) is in `_run_bench_worker` and `_run_character_bench_worker`, so this is the file's convention.
- Recommended correction: wrap the persistence block in one `asyncio.to_thread`, or add a batched `store_results` to
  `EvalsDB` so the 50 sim cells commit once. Canonical home: `DB/Evals_DB.py`.
- Size: M · Confidence: verified by trace (not measured)
- Already covered: none of TASK-32804.1-.12 names Evals.

### P2 [D3] — `watchlists_collections_screen.py` (14,319 lines, one 392-method class) has no size-ratchet row anywhere and is absent from the decomposition-candidates record, while three *smaller* modules were budgeted by TASK-32809.2
- Where: `class WatchlistsCollectionsScreen` at `:636-14319`, 392 methods (AST). Also unbudgeted: `llm_screen.py`
  (5,180; `LLMScreen` 173 methods), `change_review_screen.py` (4,967; 91 methods + 3 modal classes + 17 module
  functions).
- Evidence: the ratchet files carry only `chat_screen 25363`, `library_screen 35855` (screens) and
  `app 21415`, `console_chat_controller 29367`, `console_chat_store 22344`, `personas_screen 16436`,
  `console_transcript 8353`, `console_settings_modal 7807`, `mcp_workbench 6760` (modules).
  `test_library_modules_size_ratchet.py`'s glob is `Library_Modules/*_controller.py` — no match.
  `backlog/docs/size-decomposition-candidates-2026-09-18.md` lists 9 modules, not this one, and asserts "Size
  ratchets now guard all of these."
- Why it matters: watchlists is larger than 3 of the 7 rows TASK-32809.2 added, and its class is roughly double the
  method count of the biggest budgeted non-`chat`/`library` class. Responsibility count in the one class: tree/scope
  navigation; watchlist CRUD; source CRUD + OPML; runs; notifications inbox; briefings
  (generate/keep/export/presets/cadence/schedules); scripts + cast; audio synthesis + playback + feed server; items
  reader (paging/snapshot/filter/search/status/star); rules + noise selectors; region-layout persistence +
  responsive relayout; keyboard actions — **12**. `llm_screen.py`: **7**.
- Recommended correction: add rows at the exact measured sizes (14319 / 5180 / 4967) and a cluster map to the
  decomposition-candidates doc. **No split proposed here** — `backlog/docs/library-decomposition-recipe.md`
  §1/§2/§17 governs any split, and §2's field-ownership script has to be run first.
- Size: S (rows) / L (any split) · Confidence: verified
- Already covered: **TASK-32809.2 (In Progress) — its AC#1 is satisfied for the modules it named; the named list is
  what is incomplete.** *(Second sighting; S06 found `tldw_api/client.py` missed the same way.)*

### P3 [D3] — `change_review_screen.py` imports `pathlib` 13 times and `Workspaces.git_workspace` 20 times from inside function bodies, with no circular-import reason
- Evidence: 42 function-body imports total. `Workspaces/git_workspace.py:24-38` imports only stdlib, loguru,
  `Utils.log_sanitizer` and `Workspaces.change_tracking` — nothing from `UI`. The screen already imports
  `Workspaces.change_tracking` at module level (`:57`). Neither `pathlib` nor `git_workspace` is at module level.
  `compose()` at `:4804` pays a `sys.modules` lookup per render.
- Size: S · Confidence: verified (no cycle)

### P3 [D1] — 9 of the 13 `except Exception: pass` blocks in `watchlists_collections_screen.py` wrap multi-statement bodies, so a failure *inside* a reactive push is swallowed with no log line; the file's own dominant idiom is `except NoMatches`
- Where: `:4832, 5524, 5584, 5704, 7915, 7979, 12939, 13005, 13748`. `:7915` has a 5-statement body
  (`sources_pane.sources = …`, `set_authoritative_source_ids`, `set_selected_source_ids`, `select_source_by_id`).
- Evidence: AST — 13 broad blocks, 9 with a >1-statement try body; `grep -c "except NoMatches"` → **61**.
- Why it matters: the try at `:7915` exists to survive an unmounted pane, but it also swallows any error raised by
  the four reactive assignments inside it (which run watchers), leaving the Sources table stale with nothing logged.
- Size: S · Confidence: verified

### P3 [D4] — Three copies of "is this process alive?" that disagree in the failure direction, giving a start/stop split-brain if `poll()` ever raises
- Where: `Event_Handlers/LLM_Management_Events/server_lifecycle.py:69 process_is_running` → `except Exception: return
  **True**`; `UI/Lab_Modules/lab_server_status.py:61 _is_running` → `return **False**`;
  `UI/Screens/llm_screen.py:2931 _vllm_process_liveness_proven` → `return **False**`.
- Evidence: `server_lifecycle.reserve_server_launch` (`:80-95`) refuses a launch when `process_is_running(process)` —
  so on a raising handle it says "already running" while `read_server_rows` paints the Lab row "stopped". The user
  sees "stopped", presses Start, and the reservation silently returns `None`.
- Why it matters: each site's fail-direction is individually documented and defensible; the *combination* is not.
- Recommended correction: one `process_is_alive(process, *, on_error: bool)` in `server_lifecycle.py` (the only
  non-UI home the other two already depend on), with each caller naming its fail-direction explicitly.
- Size: S · Confidence: verified for the drift; inferred for reachability

### P3 [D4] — `_safe_text` / `_safe_skill_text`: byte-identical sanitize-validate-fallback bodies that drift on the truncation cap (500 vs 1000)
- Where: `watchlists_collections_screen.py:1954-1960` (`max_length: int = 500`) and
  `skills_screen.py:433-441` (`max_length: int = 1000`). Both bodies are exactly `sanitize_string(...).strip()` →
  `validate_text_input(..., allow_html=False)` → `fallback`.
- Why it matters: the shared behaviour is "silently substitute a fallback when validation rejects the text" — a
  title becomes `"Untitled item"` with no log line — and the two surfaces disagree on at what length that happens.
- Recommended correction: one `safe_display_text(value, fallback="", *, max_length)` in
  `Utils/input_validation.py`, which already owns both primitives.
- Size: S · Confidence: verified
- Already covered: **task-32808.9 is scoped to a named list** (canonical-JSON digest, thaw helper, attachment setter,
  role mapper, identity helper, LIKE-escape, select-blank, thumbnail-mount) — this pair is not on it.

### P3 [D3] — `UI/Screens/settings_speech_tts.py` (2,408 lines) defines no `Screen` at all
- Evidence: AST — 13 dataclasses, 1 store class, 37 module functions, zero `Screen`/`Widget` subclass. Consumers are
  `settings_screen.py`, `Widgets/Settings_Widgets/speech_tts_settings_panel.py`,
  `Event_Handlers/STTS_Events/stts_events.py`, `UI/Speech/speech_catalog_mixin.py`,
  `UI/Wizards/first_run_voice_step_state.py`. Two functions dominate: `_validated_provider_values` (412 lines) and
  `load_global_speech_tts_state` (246).
- Why it matters: `UI/Screens/` is the directory `screen_registry.py` routes; a pure validation/state module there
  is an ownership mismatch, and it is why the module escaped **both** size ratchets (neither glob covers it).
- **On the credit side:** the module has **zero** logger calls and its fingerprint carries only
  `credential_present`/`credential_source`, never a key value (`:759-778`) — no secret exposure found.
- Size: M (9 importers) · Confidence: verified

## Candidate triage
**RETIRED:** `plain_readback` `watchlists:3284` (`_console_follow_copy` returns a `rich.text.Text` built with
`Text.from_markup(escape_markup(...))`; `str()` on both sides yields `.plain`, like-for-like);
`plain_readback` `change_review:4202` (an explicit test/observability seam);
`id_keyed_dict` `change_review:219` (rows are the caller's live list, consumed in the same expression);
`id_keyed_dict` `trajectory:546,601,614,629,1021` (`_record_keys` is rebuilt wholesale on every snapshot swap and
the records are held alive by `self._snapshot` — no stale-id window);
`run_worker_coroutine` ×80 (the coroutine form is correct — the watchlists chain does go off-loop via
`run_db_off_loop`; the 60-odd watchlists reads are fine. The two real problems are the *redundant* offload and the
*un-offloaded* Evals loop, both filed);
`except_exception_return` ×26 (sampled: config fallbacks, pre-mount guards, deliberate fail-closed liveness proofs);
`function_body_import` ×58 for `evals_screen` (7× `textual.css.query.QueryError`, the file's stated convention),
`model_remote_view`/`model_curated_view` (documented boundary deferrals), `llm_screen`/`trajectory` (optional/heavy);
`lock_and_execute` `watchlists` (no `threading.Lock` + raw SQL; the drain uses `_ItemStatusIntent` coalescing);
`raw_1024x1024` `model_installed_view:89` (`_MIN_LEGACY_MODEL_BYTES`, a file-size floor);
`strftime` ×3 (all display-only local-clock formatting; `_fmt_clock` returns `—` on failure);
`try_import_guard` `change_review:300,332,3403` (documented "a bad config never breaks review" fallbacks).
**CONFIRMED:** `except_exception_pass` — the 9 multi-statement watchlists sites (filed as a narrowness issue; the
trajectory ones retired as documented pre-mount guards); `function_body_import` — `change_review_screen.py`'s 33
no-cycle cases (filed); `DUP_VERBATIM _service_for_worker ×3` — **confirmed, low value**: all three are called
*from* `@work(thread=True)` bodies, so a shared helper buys nothing and the thread-safety question stays identical
(report, don't extract).
**UNVERIFIED:** `legacy_markers` ×21 (comment/identifier hits, not resolved — no budget); `DUP _perform_safe_cancel
×6`, `_confirm`/`_submit_* ×7`, `check_action ×2`, `_validate_note_text ×2`, `_response_records_and_count ×2`,
`apply_progress ×2`, `_external_key ×2` (mechanically confirmed as clones by hash, not traced for drift).

## D4 observations for repo-wide Phase 3
1. **`rich.markup.escape` survivors after #2717** — 9 importers, 3 deliberate markup-off-sink cases, **6 still the
   broken escaper on markup-ON surfaces**: `UI/Screens/evals_screen.py`, `home_screen.py`, `study_screen.py`,
   `Library/library_rag_state.py`, `Chat/console_provider_gateway.py`, `TTS/audio_cpp_supervisor.py`. Canonical home
   exists and is already adopted elsewhere. A repo-wide completion sweep for TASK-32802.1 AC#2.
2. **Unescaped `notify()` on external text — no helper exists.** AST count of `notify()` calls whose first argument is
   non-constant and which pass no `markup=`, in `UI/Screens/` alone: `change_review_screen` 36, `llm_screen` 16,
   `model_installed_view` 6, `trajectory_screen` 2, `research_workspace_screen` 2, `model_remote_view` 1 — versus
   `evals_screen` **0** and `watchlists_collections_screen` 3. **The two clean files got there by hand, site by site,
   with long comments.** A `notify_text(self, message, **kw)` that forces `markup=False`, or a lint rule, is the
   canonical fix; without one this class regenerates on every new handler.
3. **"is this process alive?" ×3 with opposite failure directions** — home: `server_lifecycle.process_is_running`.
4. **Sanitize-validate-fallback text helper ×2 with a 500/1000 cap drift** — home: `Utils/input_validation.py`.
   Worth a repo-wide census: the shape may exist beyond these two screens.
5. **`_service_for_worker` lazy-memoizer ×3** + `_registry_for_worker`. Not worth a shared helper; worth a repo-wide
   question about whether any of these views can dispatch two thread workers in *different* groups that both hit the
   memoizer.
6. **`_perform_safe_cancel` ×6 identical bodies** — `Widgets/cancel_confirmation_dialog.py:101`,
   `UI/Screens/change_review_screen.py:4436`, `Widgets/Library/library_file_notes_git_panel.py:4288`,
   `Widgets/ModelArtifacts/install_modal.py:127`, `Widgets/Console/console_auto_speak_consent.py:159`,
   `Widgets/Console/console_capture_policy_dialog.py:184`. **One of the six is *in*
   `Widgets/cancel_confirmation_dialog.py`** — a plausible canonical home already exists and the other five did not
   import it. *(Lead's note: the repo-wide `_perform_safe_cancel` cluster is 45 defs / 28 shapes and is a
   template-method override, not duplication — but this particular 6-member byte-identical subgroup is real. See
   `candidates/phase3-clusters.md`.)*

## Left UNVERIFIED
| Claim | Why | Command |
|---|---|---|
| A `NoMatches` from `_apply_catalog_state`'s post-await `query_one` reaches `app._handle_exception` rather than being pre-empted by `Widget._on_unmount`'s `workers.cancel_node` | requires racing an unmount against a scheduled worker resume; DO-NOT-RUN-THE-APP | a Pilot test: mount the screen with a `controller.refresh_workspace_catalog` awaiting a controllable `asyncio.Event`, `await pilot.app.switch_screen("home")`, set the event, `await pilot.pause()`, assert the app is still running |
| A real `git push`/`commit` failure string in this repo's engine contains `[/` (the crashing form, vs the merely-deleting `[x]` form) | verified the renderer and the text provenance, not the corpus of git messages | `grep -rn 'raise GitWorkspaceError\|raise CommitRefusedError\|raise PushRefusedError' tldw_chatbook/Workspaces/git_workspace.py` and read every message template for a bracket |
| The 50-cell `save_artifact` loop produces a user-visible freeze | traced, never measured | time 50 `store_result` calls against a file-backed `EvalsDB` and compare to the 16 ms frame budget |
| The TTS provider-test evidence store can report `REACHABLE` for a *replaced* API key | `build_provider_test_fingerprint` records only `credential_present`/`credential_source`; invalidation depends on `saved_revision`, whose bump condition lives outside this slice | `grep -n "_provider_configuration_revisions" tldw_chatbook/Widgets/Settings_Widgets/speech_tts_settings_panel.py` and read every write site |
| Whether the 21 `legacy_markers` rows name genuinely dead code | not examined — no budget | per file: `grep -n legacy <file>`, resolve each identifier, then `pytest --collect-only -q` plus an exact-module-path `rg` |
| Whether `_render_worker`'s thread (`trajectory_screen.py:1006`) can hit `KeyError` in `self._record_keys[id(record)]` when the main thread swaps `_turns` and `_record_keys` concurrently | the window exists (both are plain attribute reassignments with no lock) but only above `WORKER_THRESHOLD = 5000` records, and the worker's `except Exception` converts it to a failure card rather than a crash | `pytest Tests/UI/test_trajectory_screen.py -q` after adding a test that calls `_apply_live_snapshot` while a `_render_worker` is parked on a patched `_build_row_specs` |

# S21 — `UI/Wizards/` + `UI/Speech/` + `UI/Watchlists_Modules/`

**Coverage:** files read in full: 5 | sampled: 23 | mechanical only: 27 (of 55).
Mechanical = whole-slice AST sweeps for post-await DOM, `run_worker` arity/`exit_on_error`, **try-section
classification**, reactive mutable defaults, `re.compile`, dotted `get_cli_setting`, timers, `id()`-keyed dicts,
file I/O, secret logging, markup escaping.

## Findings

### P1 [D1] — The first-run wizard's "Test" button sends the user's API key to a host the config file can supply, with no `Utils/egress.py` check
- Where: `UI/Wizards/FirstRunSetupWizard.py:901-948` (`_probe_first_run_provider_connection`, call at `:934`),
  reached from `:3076` → `_run_probe:3107`; endpoint prefilled by `:2237 _initial_endpoint_for` →
  `Chat/console_provider_endpoints.py:63 first_configured_endpoint`.
- Evidence: **lead-verified and ruled — see `phase4-verification.md` "The endpoint-probe egress gap".**
  `settings_endpoint_probe.py` imports the egress helpers at `:44-46` and uses them **only** at `:514`, inside the
  TTS branch; the chat branch has none. `grep -rn "egress|check_url_or_raise" tldw_chatbook/UI/Wizards/` → **0
  hits** — the wizard does not compensate; it *adds* the credential
  (`httpx.AsyncClient(headers={"Authorization": f"Bearer {credential_value}"})` at `:922-924`).
  **Restore vector:** `Backup_Recovery/config_participants.py:385,412` lists `_write_raw_cli_config_unlocked` among
  its guarded writers, so a restored profile archive can set `api_settings.<p>.base_url`, and the user only has to
  press Test. **Chatbook import is NOT a vector** — `ChatbookImportWizard.py:79-83` scopes content to
  CONVERSATION/NOTE/CHARACTER/MEDIA/PROMPT/KEPT_BRIEFING, no config.
- **Lead's correction to this slice:** S21 counted **5** callers; there are **4**
  (`speech_catalog_mixin.py:145`, `chat_screen.py:3241`, `settings_screen.py:15174`, `FirstRunSetupWizard.py:934`) —
  the fifth was an import line. S19's original count was right. **S21's substantive contribution stands and is the
  more valuable half:** the restore vector, and the observation that `speech_catalog_mixin.py:145` passes
  `purpose=TTS_CATALOG` and is therefore on the **guarded** branch — an in-slice counter-example proving the fix
  belongs in the chat branch, not at call sites.
- Recommended correction: `check_url_or_raise_async(resolution.models_url, trusted_origins=origin_set(...))` in the
  chat branch, matching `:514`. **One guard in the shared function, not four at call sites.**
- Size: M · Confidence: verified (code + backup participant traced; not executed against a live restore)
- Pinning test: none — `Tests/UI/test_settings_endpoint_probe.py` contains **zero** `egress`/`check_url_or_raise`
  references.

### P1 [D1] — Two first-run workers exit the app on an unguarded post-await `query_one`, because they run at `exit_on_error`'s app-exiting default
- Where: `FirstRunSetupWizard.py:7116-7124` (`_apply_password_worker`, dispatched `:7110`) and `:7265-7417`
  (`_render_rows`, dispatched `:7261`; unguarded queries at `:7387`, `:7417`).
- Evidence: AST sweep found 23 post-await `query_one` sites outside any `try` body; cross-referenced against a
  `run_worker` audit, **these two are the only ones where `exit_on_error` is at its default AND there is no
  `is_mounted` guard** (`_run_voice_sample`'s four and `speech_*`'s six are all `exit_on_error=False`;
  `_run_probe:3119` carries the guard). `textual/worker.py:382-384` confirms app-exiting. Both cross a real
  suspension: `_apply_password_worker` → `run_in_executor(None, enable_config_encryption)`; `_render_rows` → three
  `run_in_executor` hops including a full `load_cli_config_and_ensure_existence(force_reload=True)`.
  **The step is not held open across that window** — `SetupWizardContainer`'s dismiss guard
  (`hold_provider_save_settlement()`, `:10335`) covers the **provider** save only, so Finish-later / Skip / Escape
  during the executor call unmounts the step. Note `enable_config_encryption` swallows and returns `False`, so
  **the raising call is the `query_one`, not the crypto.**
- Why it matters: a dismissal during password-apply or Summary load **takes the whole app down on the very first run.**
- Size: S · Confidence: inferred (traced end to end; not reproduced)
- Same shape, lower confidence (message-pump handler, not a worker — but an exception there also reaches
  `_handle_exception`): `watchlists_workbench.py:291,297,512,524` query the DOM after `await mounted.remove()` /
  `await body.mount(...)` with no guard.

### P1 [D1] — Voice-blend import/export reads and writes a user-chosen path with no `path_validation`, no size cap, no shape validation, and a non-atomic write — while the sibling export in the same package does all three
- Where: `UI/Speech/speech_settings_mixin.py:1068-1090` (`_handle_export_file`) and `:1106-1142`
  (`_handle_import_file`). Reachable: `VOICE_BLEND_ACTIONS` → `on_button_pressed:410-415` → `VoiceBlendsPane`,
  mounted at `UI/STTS_Window.py:2040`.
- Evidence: `open(import_path, "r")` + `json.load(f)` on a `FileOpen`-picked path, then
  `existing_blends.update(imported_blends)` → `write_kokoro_ui_blends(...)` — **no `validate_path_simple`, no byte
  cap, no type check on the parsed object before it reaches persisted storage.** Export is `open(..., "w")` +
  `json.dump` — non-atomic, while `Utils/atomic_file_ops.py:199 atomic_write_json` exists (TASK-32808.5 **Done**).
  **The drift is intra-file:** `speech_playback_mixin.py:1004-1025` imports `validate_filename` and
  `validate_path_simple` and applies both to *its* user-chosen destination.
  `grep -rn "path_validation" tldw_chatbook/UI/Speech/` → **only** those lines.
  **The destination writer is fine** (`TTS/voice_blend_paths.py:64` is atomic + private-path-verified) — the hole is
  entirely on the user-path side. Both handlers also surface the raw exception on screen and in logs, and an
  `OSError` carries the full filesystem path.
- Size: M · Confidence: verified
- Already covered: cluster members of TASK-32806.6 and .7 — **but the `path_validation` omission and the unvalidated
  `dict.update` into storage are neither**, and the in-package counter-example makes this a non-adopter rather than
  a missing helper.

### P2 [D3] — The Speech Playground's four mixins share one undeclared `self` namespace: 13 attributes are written by 2–3 mixins, and none owns any of them
*(the measurement the brief asked for, not a restatement)*

| mixin | lines | methods | `self.X` it assigns | attrs it **reads but never defines** |
|---|---:|---:|---:|---:|
| `SpeechCatalogMixin` | 1859 | 49 | 31 | 11 |
| `SpeechPlaybackMixin` | 1291 | 42 | 12 | 11 |
| `SpeechSettingsMixin` | 1194 | 31 | 6 | 4 |
| `SpeechProfileMixin` | 704 | 15 | 18 | 16 |
| `SpeechSynthesisMixin` | 659 | 11 | 5 | 14 |

- **5,707 mixin lines; 4,513 of them feed a 3,028-line pane — 60% of the Playground's behaviour lives outside the
  class.** Attributes written by ≥2 of the four co-mounted mixins: **13**, led by `_generation_operation_id`
  (**3 mixins, 9 write sites**), `_profile_save_suppressed` (3), `_provider_ids` (3),
  `_profile_effective_availability` (Catalog ×10 / Profile ×3), `_profile_preview_loading` (×5 / ×3),
  `_profile_voice_validation_token` (×3 / ×4), + 7 more.
- **Ownership is order-dependent by construction:** `speech_playground_pane.py:385-389` calls the four
  `init_*_state()` in a fixed order, and `init_profile_state` is called **again** at `:2734` on preset re-adopt,
  re-zeroing three fields that `speech_catalog_mixin` also writes from in-flight catalog replies.
- **Method-name collisions: 4 — and all four are honestly benign.** Each pairs `SpeechSettingsMixin` with one of the
  other four, and `SpeechSettingsMixin` never co-occurs with them. **Zero live MRO collisions among the four
  co-mounted mixins.** *(Stated rather than inflated — this is the honest answer to the brief's question.)* The one
  real MRO double-dispatch is deliberate and documented at `speech_settings_pane.py:1538-1540`.
- Recommended correction: **not a decomposition** — a single `SpeechPlaygroundState` dataclass owning the 13 shared
  fields, held as `self._state`. Mechanical, no behaviour change. · Size: M · Confidence: verified (AST)

### P2 [D3] — `FirstRunSetupWizard.py` (10,404 lines) is the largest module in the repo with no size-ratchet row, and the decomposition doc's own scope claim is false for it
- `_BUDGETS`' smallest row is `mcp_workbench.py: 6760`. **The wizard at 10,404 is larger than 3 of the 7 rows** and
  is absent. `backlog/docs/size-decomposition-candidates-2026-09-18.md:22-25` states *"Size ratchets now guard all
  of these"* and **does not list `UI/Wizards/` at all.**
- Obvious first extraction: `ProviderStep` (`:1148-3299`, ~2,150 lines of endpoint resolution / discovery / probe
  evidence that own no pixels outside `compose_step`). · Size: S (row) / L (split) · Confidence: verified
- Already covered: **shows TASK-32809.2 incomplete.** *(Fifth slice to find a missed god module.)*

### P2 [D4] — `middle_truncate_path` drifted from `Utils/Utils.elide_path_middle` and cuts the filename in half on the Summary screen
- `UI/Wizards/first_run_setup_state.py:607-630`, sole caller `FirstRunSetupWizard.py:7439`. Executed:
  ```
  p = ".../tldw_chatbook/config_for_my_work_profile.toml"
  middle_truncate_path(p, 40) -> '/Users/macbook-dev/…my_work_profile.toml'
  elide_path_middle(p, 40)    -> '/Users/m…config_for_my_work_profile.toml'
  ```
- Why it matters: **`elide_path_middle`'s docstring says it exists *because* a midpoint split "discards the very
  name the user just picked".** The wizard's Summary — the screen whose whole job is an honest read-back of what
  landed on disk — shows a half-eaten filename. · Size: S · Confidence: **verified (executed)**
- Already covered: TASK-32808.3 is about `text[:n] + "..."` **tail**-truncation; this is the *path* elider, a
  different canonical home and a **behavioural, not cosmetic**, drift.

### P2 [D4] — Five byte-identical `select_<X>_by_id` in `Watchlists_Modules`, with `table_selection.py` already established as the shared home
- `runs_pane.py:409`, `items_pane.py:523`, `article_list.py:949`, `rules_pane.py:261`, `sources_pane.py:1744` —
  all the identical 7-line linear scan. **`table_selection.py` is the package's shared selection module and is
  already imported by 6 of these panes** — the home exists and this shape ignores it. Companions: `displayed_items`
  ×2 (drifted only in docstring), `_restore_search_focus` ×2.
- Why it matters: 5 places where the id-coercion rule (`str(... or "")`) can drift, and the lessons file's "never
  carry a row INDEX across an await — capture IDENTITY" rule depends on that coercion being **one** rule. · Size: S

### P2 [D1/D3] — The playback progress loop is a raw `asyncio.create_task` outside Textual's worker registry, cancelled by `await asyncio.sleep(0.05)`, driving 3 `query_one` calls every 100 ms
- `UI/Speech/speech_playback_mixin.py:1153-1235`, started at `:770`/`:877`, cancelled at six sites.
- Evidence: not a `run_worker`, so `WorkerManager` will not cancel it on prune — cleanup depends entirely on six
  hand-written cancel sites. **Every cancel site is `task.cancel()` followed by
  `await asyncio.sleep(0.05)  # Small delay to ensure cancellation` rather than awaiting the task**, so a new timer
  can start while the old one is still in its `except CancelledError` → reset block, and **both write the same three
  widgets.** Steady state: **30 DOM queries/second for the duration of playback.** The loop's own
  `except Exception: logger.error(...)` means a mid-playback `NoMatches` shows up only as a log line and a silently
  frozen progress bar.
- Why it matters: **cancellation-by-sleep is a race, not a barrier.** · Size: M · Confidence: verified

### P3 [D1] — `_ProviderConnectionUiDraft`'s "memory-only" seal is incomplete: `pickle.dumps` emits the plaintext API key, where its sibling blocks it
- Executed: `copy.copy` **blocked**, `copy.deepcopy` **blocked**, `pickle.dumps` **ALLOWED, contains secret: True**;
  `repr` correctly redacted. `first_run_setup_state.py:106 ProviderCredentialDraft` defines `__copy__`,
  `__deepcopy__`, `__reduce__` **and** `__reduce_ex__`; `FirstRunSetupWizard.py:837` defines only the first two.
- **No live leak** — resume-draft persistence is JSON with a secret-key screen, not pickle. **The drift is the
  finding:** a defence-in-depth seal two classes implement differently, 1,000 lines apart, is one that will be
  assumed complete. · Size: S · Confidence: **verified (executed)**

### P3 [D1] — `RagStep` round-trips the embedding model id through a markup-parsing `RadioButton` label, and the resume path compares against it
- `FirstRunSetupWizard.py:4933` (`str(event.pressed.label)` → `selected_embedding_model` → `build_rag_commit` →
  **config**), `:4943`, `:8536`. Labels are built as `SetupRadioButton(model_id)` with **no `markup=False`**.
  Per `lessons-textual.md`, a bracketed segment is deleted from the read-back.
- **The same file already learned this twice:** `AppearanceStep` compares `_theme_name`/`_card_name` **riders**
  (`:8579`, `:8593`) *precisely because* labels are humanized (TASK-21149). **`RagStep` is the un-migrated one.**
  · Size: S · Confidence: verified (mechanism; no bracketed model id observed in the wild)

### P3 [D3] — 67 of 177 DOM guards in the slice use a bare `except Exception` where the slice's own dominant idiom is `except NoMatches`
- AST count: broad **67** vs `NoMatches` **110**. Distribution: `FirstRunSetupWizard.py` 36, `runs_pane.py` 7,
  `speech_settings_mixin.py` 6, `speech_playback_mixin.py` 5, `sources_pane.py` 4, +9.
  **Worst instance:** `runs_pane.py:453-458` wraps `table.clear()` + `table.add_row(...)` in
  `except Exception: pass`, so a failed row insert leaves a **silently partial run-items table.** · Size: S

### P3 [D4] — Briefing script-turn rendering and its three constants are duplicated inside one package, justified by a reason that does not apply
- `artifacts_pane.py:522-577` vs `kept_briefings_modal.py:111-224`. **The copy's docstring says "duplicated rather
  than imported, since it is a private helper of a sibling UI module" — but both files are in
  `tldw_chatbook/UI/Watchlists_Modules/`, so no package boundary is crossed.** Three constants now have to be kept
  in lockstep by comment. · Size: S · Reported as documented duplication **with the note that the stated
  justification is factually wrong.**

### P3 [D4] — `_cli_setting` / `_tts_service_factory` / `_is_valid_voice` are byte-identical across the Speech mixins
- Bodies identical; only `_tts_service_factory`'s docstrings differ (Catalog's carries the incident that motivated
  the seam). **The seam itself is deliberate (patchability) — only the triplication is the finding.** · Size: S

### P3 [D3] — Cross-package private import: `UI` reaches into `Scheduling`'s `_compute_next_run` from a per-row render path
- `sources_pane.py:1005`, called from `source_next_check_text` → `_source_row_cells` (per table row). The only
  `import _<name>` in the slice. The function-body placement is justified in situ on ADR-097 grounds and is correct;
  **the privateness is the finding.** · Size: S

### P3 [D2] — `load_region_layout()` writes config.toml as a side effect of a read
- `UI/Watchlists_Modules/region_layout_store.py:117-127`. Self-limiting (fires once per version bump), but it takes
  the config write lock synchronously from whatever called the load. · Size: S

## Candidate triage
**CONFIRMED:** the mixin shape (**measured**, see P2); `run_worker` with default `exit_on_error` + no `try` — 6
sites, 2 with a real post-await DOM query; `plain_readback` 2 of 5.
**RETIRED — several of the lead's dispatch hypotheses:**
- **`query_one` in a `finally:` body (the W002 blind spot): 0 instances in this slice.** Section-aware AST sweep: 8
  DOM calls in `except` bodies, **none in any `finally`**.
- **`rich.markup.escape`: 0 occurrences.** All 22 escape sites use `Utils/input_validation.escape_markup`; four sites
  document a *deliberate* non-escape into `Text` (correct — `Text.append` is not a markup parser).
- **Watchlists controllers carry the stale `thread → asyncio.run → thread` hop: retired, cause wrong.**
  `watchlists_backend_controller.py` is a pure async router; every routed method on the scope service is
  `async def`, so the `_maybe_await` argument is a coroutine, never blocking I/O. **The real controller-layer finding
  is different:** `briefing_preset_modal.py:371,567,582,622` and `kept_briefings_modal.py:492,511` call
  `asyncio.to_thread(self.db.<method>)` directly instead of `Subscriptions/db_offload.run_db_off_loop`, **whose
  docstring says the hop is unsafe for a `:memory:` SubscriptionsDB.** Production passes a file-backed DB, so **no
  live defect** — filed as a D4 non-adopter, not a P1.
- `run_worker(exclusive=True)` without `group=` → **0**; every one of the 21 exclusive dispatches names a group.
- `query_one` in `except asyncio.CancelledError` → **5 sites, real shape but contained** (4 are
  `exit_on_error=False`, the 5th carries `is_mounted`). Reported inside P1 as context, not as its own finding.
**RETIRED:** `tempfile_no_secure` (`NamedTemporaryFile` is 0600 and unlinked on four paths);
`get_cli_setting_hot` (7 cache-backed reads once per pane load; TASK-32804.8 already covered Speech settings);
`except_exception_pass`/`return` 67 rows (**all read**; idempotent UI-chrome recovery, rolled into the P3 breadth
finding with `runs_pane.py:457` as the one that hides real work); `function_body_import` ×107 (each of ~40 read
carries an ADR-097 rationale; `sources_pane.py:1005` re-filed on the **privateness**, not the placement);
`strftime` ×5 (`humane_time.py` explicitly documents "naive means UTC" and normalizes both sides);
`raw_1024x1024` ×2 (MB conversions); `try_import_guard` ×10; `legacy_markers` ×82 (**all** explanatory prose);
`open_backup_restore` 2-copy (a 4-line delegate — consolidating would be worse); `_button` 2-copy (the shared
definition already lives in `Library.library_shell_state` per its own docstring).
**`_maybe_await`: no qualifying call site.** Every argument at this copy's 20+ call sites is an `async def`.

## D4 observations for repo-wide Phase 3
1. **`table_selection.py` is an established but under-used package helper** — 5 + 2 + 2 + 2 re-rolls beside a module
   6 of those files already import from. **Canonical home exists; no new module needed.**
2. **`elide_path_middle` has a third, drifted implementation that violates the invariant its own docstring names.**
   Only **2 importers** of the canonical repo-wide — worth a census for other midpoint splitters.
3. **Two "memory-only credential" seals, two different dunder sets.** If the repo has more of this pattern, the whole
   family should share one base.
4. **`Subscriptions/db_offload.run_db_off_loop` has 58 call sites in 3 `Subscriptions`/`Scheduling` modules and 0 in
   `UI/`**, while `UI/Watchlists_Modules` makes 8 bare `asyncio.to_thread(db.…)` calls on the same DB. **The
   helper's own docstring names the condition that makes the bare form unsafe.**
5. **`_cli_setting`/`_tts_service_factory`** — the "hook so tests can patch it" shape likely recurs far beyond Speech.
6. **Three filename rules, none length-capped:** `Utils/text.sanitize_filename` (denylist), the
   `ChatbookCreationWizard` allowlist (`:670`, drops `.`/`(`/`,`), `chatbook_creator._safe_conversation_file_id`.
   **No downstream defect here** — the wizard passes `output_path` explicitly, so the two names cannot disagree.
   Feeds TASK-32808.2, and **the allowlist-vs-denylist split is the decision that task has to make.**
7. **Wizard twin of TASK-32804.3:** `FirstRunSetupWizard.py:1533` runs a **4 Hz** `set_interval` →
   `_current_provider_readiness()`, which shallow-copies `app_config` + `api_settings` and does 2 `query_one` calls
   per tick. Guarded to Anthropic-with-connection-visible, so bounded — **but it is the same shape, and the .3 fix
   should cover both.**

## Left UNVERIFIED
| Claim | Why | Command |
|---|---|---|
| Dismissing the wizard during `_apply_password_worker`'s executor hop actually raises `NoMatches` and exits the app | app must not be booted | `pytest Tests/Wizards/test_first_run_setup_wizard.py -q` after adding: drive to Protect, click set-password, dismiss mid-executor with a slow injected `_enable_encryption`, assert `app.return_code is None` — born-red before the fix |
| A restored profile archive can actually place an attacker-chosen `base_url` into the wizard's prefill | traced `config_participants.py` → `_write_raw_cli_config_unlocked` and `_initial_endpoint_for` → config; **did not execute a restore** — this is what decides P1 vs P0 | `pytest Tests/Backup/ -q` plus a case that restores an archive whose `config.toml` sets `api_settings.openai.base_url`, then asserts `ProviderStep._initial_endpoint_for("openai")` returns it |
| `_handle_import_file` parsing a large hostile JSON measurably stalls the pump | not measured; `test_no_blocking_io_on_message_pump.py` deliberately excludes small local FS ops | generate a 500k-key JSON, time `_handle_import_file` under a timing assertion |
| A bracketed embedding model id is reachable in practice | depends on what users put in `[embedding_config] models`; no shipped key contains `[` | `python -c "from textual.widgets import RadioButton; print(str(RadioButton('m[8b]').label))"` → expect `'m'` |
| No `finally:`-body `query_one` exists anywhere else in this slice | the sweep covers `ast.Try.finalbody` — **not** `contextlib` teardown or `__aexit__` bodies | re-run it extended to flag DOM calls inside any `__aexit__`/`@contextmanager` after-yield region |

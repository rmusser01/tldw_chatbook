# UI-personas — tldw_chatbook/UI/Screens/personas_screen.py, 16414 lines

## Coverage
| file | lines | read in full / sampled (which ranges) / mechanical only |
|---|---|---|
| tldw_chatbook/UI/Screens/personas_screen.py | 16414 | **read in full** (1–16414, in 15 sequential chunks). Mechanical over the whole file additionally: AST census of all 64 `run_worker` calls + 3 `@work` decorators (kwargs), class/def map (461 `PersonasScreen` methods measured with the ratchet test's own `_measure` units), 59 function-body imports (each module target `ls`-verified, 9 symbols grep-verified), all 180 `except Exception` sites (15 `→ return/pass` read individually), 3 timers, 0 `get_cli_setting`, `ruff --select E9,F63,F7,F82` clean |
| (context only, not reviewed) `UI/Persona_Modules/personas_preview_controller.py:333-380`, `Chat/provider_readiness.py:501-561`, `Character_Chat/visual_identity.py` (`cancel`, `cleanup_*`), `Backup_Recovery/dictionary_source_job.py:160-200`, `Backup_Recovery/storage_admission.py` (def list), `Tests/Architecture/test_screen_size_ratchet.py` (`_BUDGETS`, `_measure`) | — | sampled to settle specific claims below |

Environment used for every command: worktree `/Users/macbook-dev/Documents/GitHub/tldw-review` @ d8fb4053f9, `source <SCRATCH>/env.sh`, `$PY` = main-checkout venv 3.12.11 / Textual 8.2.8.

## Findings

### P2 [D2] — Three coroutine workers do their file writes / sqlite reads synchronously on the event loop while their sibling paths thread the same operation
- Where: `personas_screen.py:5688-5692` (`_dictionary_export_worker`: `exports_dir.mkdir`, `temp.write_text(body)`, `temp.replace(target)`); `:13537-13545` (`_export_expression_set`: `mkdir`, `temp.write_bytes(blob)` of a freshly built zip, `replace`); `:2109-2114` (`_apply_pending_character_conversation_link`, runs inside the `personas_initial_load` coroutine worker: `db.get_local_authority_id()` and `db.get_character_card_by_id(...)` are called inline while lines 2125/2130 of the same function use `asyncio.to_thread`). Contrast `_lore_export_worker:6755-6756`, which threads its `write_text`/`replace`.
- Evidence: read only (structural: no `await` between the calls; `run_worker(<coroutine>)` runs on the loop — AST census shows `thread=True` on 0/64 calls). `grep -n "def get_character_card_by_id" -A40 DB/ChaChaNotes_DB.py` → `SELECT * FROM character_cards WHERE id = ? AND deleted = 0` via `execute_query` (real sqlite). Stall duration not measured — see UNVERIFIED.
- Why it matters: while the zip/JSON write or the card read runs, the app processes no input or paint; on a slow disk or a large expression set this is a visible freeze, and the deep-link read sits on the mount path.
- Recommended correction: `await asyncio.to_thread(...)` around the mkdir+write+replace triple exactly as `_lore_export_worker` already does (one `_atomic_export_write` helper on the screen serves all three — see P3 D4 below); `to_thread` the two DB reads at 2110/2114.
- Size: S · ADR: no · Confidence: verified (structure) / stall magnitude inferred
- Pinning test: `Tests/UI/test_personas_dictionaries.py::test_export_json_writes_file_and_reports_path` asserts the file lands in `exports/`, not where the write runs — stays green after the fix.
- Already covered: none (task-1320 "Move screen mount IO off the App message pump", In Progress, covers the mount pump, not these worker bodies)

### P2 [D1] — Character-card import reads the user-picked file into memory with no byte cap; the two sibling importers in the same file cap at 10 MB
- Where: `personas_screen.py:13725-13726` (`_import_character_from_path`: `source.read_bytes()` unbounded, then the bytes go to `inspect_character_card_tts_attachment` and `import_character_card_with_outcome`). Siblings: `:14079` (world book, `stat().st_size > PERSONAS_WORLDBOOK_IMPORT_MAX_BYTES`) and `:14196` (dictionary, `PERSONAS_DICTIONARY_IMPORT_MAX_BYTES`) gate before reading; `:10520` avatar upload gates at 5 MB.
- Evidence: `grep -n "MAX_.*BYTES" Character_Chat/Character_Chat_Lib.py` → the lib caps decode **pixels** (`_MAX_CARD_DECODE_PIXELS = 50_000_000`, L148/1539) and history-export bytes (`_MAX_EXPORTED_HISTORY_FILE_BYTES`, L75) — nothing caps the card file's byte size; `grep -n "MAX\|len(" ccp_character_handler.py` → only field-length caps. The "Character Cards" picker filter (L581) accepts any `.png`/`.webp`/`.json`.
- Why it matters: a mispicked multi-GB `.png` is read whole into RAM on the `to_thread` worker before any validation runs; every other user-file import in this screen refuses first. Trust-boundary behaviour drifted across three siblings (also a D4b).
- Recommended correction: add `PERSONAS_CHARACTER_CARD_IMPORT_MAX_BYTES` (same 10 MB family) and the same `stat().st_size` gate before `read_bytes()` at 13725.
- Size: S · ADR: no · Confidence: inferred (memory consequence not reproduced)
- Pinning test: `Tests/UI/test_personas_workbench.py:3537` asserts `character_writes == [source.read_bytes()]` — pins the whole-file read as the contract; a size gate must precede it, and no test pins a card byte cap (grep `IMPORT_MAX_BYTES` in that file → only the avatar cap at :2507).
- Already covered: none (task-19558 "Security primitives … five seams" is Done and did not touch this read)

### P2 [D4b] — The "borrow-and-retire a native sqlite connection on the worker thread" block exists three times with behavioural drift; two copies live in this screen
- Where: `personas_screen.py:9186-9217` (`_persona_visual_thread.work`), `:12031-12057` (`_visual_identity_thread.work`), and `tldw_chatbook/Backup_Recovery/dictionary_source_job.py:172-195`. All three: capture `getattr(db._local, "conn", None)` before the call; in `finally`, if it was `None` → `db.close_connection()`, assert `db._local.conn is None` and `threading.current_thread() not in _repository_participant(db).retiring_threads`, else raise `<domain>_native_not_retired`.
- Evidence: `sed -n '160,200p' Backup_Recovery/dictionary_source_job.py` (third copy); `grep -rn "db\._local\b" tldw_chatbook | grep -v tldw_chatbook/DB/` → personas_screen is the only non-DB/non-Backup_Recovery module touching `_local.conn` (Actor_Packs/Persona_Visual read `transaction_depth` only); `grep -rn "_repository_participant(" tldw_chatbook | grep -v Backup_Recovery/` → personas_screen ×2 + `Sync_Interop/sync_state_repository.py:205`.
- Drift: dictionary_source_job raises `bootstrap.RecoveryRequired` (durable recovery path); both screen copies raise plain `RuntimeError` and hang recovery state on the exception object (`error.result` at 9216; `error.result`, `.source_error`, `.cleanup_candidate_relpath`, `._visual_identity_retirement_error` at 12053-12056), which the callers then pick apart (12059-12063, 9219-9232). The visual-identity copy additionally skips the retiring-threads check for `db.is_memory_db` (12050); the other two do not. That reaches storage (which retirement failures are retained for later cleanup) → also a D1-class drift.
- Recommended correction: one `retiring_native_borrow(db, *, domain)` context manager in `Backup_Recovery/participants.py` (it already owns `_repository_participant` and the third copy), returning a structured outcome instead of attribute-stuffed exceptions; the screen's two `work()` closures call it.
- Size: M · ADR: yes (`126-complete-local-backup-and-recovery.md` L317 "Existing native borrowers retain their actual owner-thread lifetime" governs the behaviour; consolidation needs no new decision) · Confidence: verified (copies read side by side)
- Pinning test: none found for the retirement block itself (`grep -rn "native_not_retired" Tests/` not run — see UNVERIFIED)
- Already covered: none

### P2 [D3] — personas_screen.py is a 16,414-line / 461-method god module with no row in the screen size ratchet
- Where: whole file; `Tests/Architecture/test_screen_size_ratchet.py:76-812` `_BUDGETS` has rows only for `chat_screen.py` (16966/563) and `library_screen.py` (33204/1276); `@pytest.mark.parametrize("rel_path", sorted(_BUDGETS))` (L938/977) so an absent row is simply never measured.
- Evidence: `python - <<EOF ast …` with the test's own `_measure` semantics → `lines 16414 PersonasScreen methods 461` (97% of chat_screen's line budget, 82% of its method budget). `__init__` alone sets ~90 instance attributes (L1321-1469).
- Responsibilities (line ranges): module constants + 12 frozen snapshot dataclasses + `_drain_async`/`_drain_to_thread` + 2 lifetime decorators (1-1067); compose/state round-trip/mount/demand-mounted center views per ADR-115 (1529-2035); character-conversation deep link per ADR-120 (2077-2223); runtime-backend switch + responsive rails (2225-2397); library paging/sort/search/tag + dictionary/lore row rendering (2398-2522, 3480-4386); **Character TTS controls** per ADR-028 (2523-3475, ~950 lines); mode switching + header copy (4387-4603); **Actor Pack export/import/create** per ADR-074 (4604-5131, 7512-7920); selection (5132-5540); **dictionaries** incl. character attach (5541-6485); **lore/world books** (6485-6932); saved conversations + Console handoff + preview delegation (6933-7389); create/edit/duplicate/toggle (7390-8302); **Persona shared visual identity** (8303-8919); **Persona Visual pack authoring** incl. `_persona_visual_thread` (8920-10205); character edit + visual-identity load + avatar upload (10206-10575); LLM-assisted character generation (10576-10723); avatar/expression thumbnails (10724-11003); **Character visual identity pack** incl. `_visual_identity_thread` (11004-12353); **expression slots** upload/generate/style (12355-13342); expression-set import/export (13343-13547); **import** (character + TTS commit, lore, dictionary) (13548-14288); **export** single/bulk/JSON/PNG (14289-14638); **delete** single/bulk (14639-15108); character save (15109-15432); policy rules + persona save (15433-15703); cancel (15704-15763); `_show_center` + aggregate draft snapshot + navigation veto (15764-16058); `_run_guarded`, key bindings, focus, footer sync (16060-16414).
- Why it matters: every one of the ~10 subsystems above shares one 90-attribute `__init__`, one `_io_dialog_active` flag and one message namespace; the file grew past the point where the repo ratchets its peers, and nothing stops it growing.
- Recommended correction: (1) add the row `"tldw_chatbook/UI/Screens/personas_screen.py": ("PersonasScreen", 16414, 461)` now (one-way ratchet, Size S); (2) any split follows `backlog/docs/library-decomposition-recipe.md` §1 per-subsystem PR series with §2 field-ownership script — the natural first peels are Character TTS (~950 lines, self-contained snapshot/authority family), Actor Packs (~1000), Persona Visual (~1300), Character visual identity (~1200) — each already has its own snapshot dataclasses at 671-826 (Size L, per recipe §17, not this review's design).
- Size: S (row) / L (split) · ADR: no for the row; the split is governed by the recipe (ADR 004/007/115 cover the workbench shape, not decomposition) · Confidence: verified
- Pinning test: `Tests/Architecture/test_screen_size_ratchet.py` (the row IS the pinning test once added)
- Already covered: none (task-1378 / task-31202 are settings_screen; task-118 extracted the preview controller and is Done)

### P3 [D3] — Private helpers reached across packages and across widget boundaries (14 distinct names, 40 sites)
- Where: `Backup_Recovery.storage_admission._Acquisition/_task_identity` (1014, 1026, 1044, 1046); `Backup_Recovery.participants._repository_participant` (9208, 12047); `persona_visual_participants._shape` (9184, 9191), `visual_identity_participants._shape` (12030, 12038), `chat_source_participants._validate` (9199); `PersonaVisualRepository._get_active_asset_storage_key` (9003); `config._CONFIG_CACHE` reached through `visual.sys.modules.get("tldw_chatbook.config")` (9140-9143); `db._local.conn` (9188, 9211, 12032, 12050); `ccp_character_handler._default_character_db` (6937, 14562, 14594); editor internals `_character_data`/`_mark_dirty`/`_user_touched`/`_run_validation`/`_input()`/`_area()`/`_set_avatar_status_from_record` (6074, 6225, 10843, 11835-11837, 12452-12458, 12677, 12740, 12796, 12997-12999); conversations-controller internals `self.conversations._conversation_query/_requested_conversation_id/_list_character_id/…` (2087-2190, 14 attributes snapshotted and restored by name).
- Evidence: `grep -nE "\b(storage|visual|chat|life|repository|config|db|editor|…)\._[a-zA-Z]" personas_screen.py` (output in working notes); `grep -rn "storage\._Acquisition\|storage\._task_identity" tldw_chatbook | grep -v Backup_Recovery/` → 6 further sites in `TTS/` (so `_Acquisition` is a de-facto public contract with 7 external users and no public name; `storage_admission.py` exports only `acquire_storage`, `admit_startup`, `StorageLease`).
- Why it matters: a rename inside `storage_admission`, `participants`, the editor widget or the conversations controller silently breaks this screen; mocked tests never catch it (the brief's stated failure mode).
- Recommended correction: promote `_Acquisition`/`_task_identity` to public names in `storage_admission` (7 external users); give `PersonasCharacterEditorWidget` a `generation_inputs()` accessor for name/description/personality and a `remove_avatar()` method (replaces 12452-12458); give `PersonasConversationsController` `snapshot_browse_state()/restore_browse_state()` (replaces 2087-2178).
- Size: S per seam · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none

### P3 [D4b] — Lazy image-generation import shims copied into four UI modules
- Where: `personas_screen.py:380-419` (5 wrappers: `get_image_generation_config`, `list_image_models_for_catalog`, `resolve_backend_reference_image_capability`, `build_request`, `run_generation`); `UI/Screens/settings_screen.py:464`; `UI/Console_Modules/image.py:47,55`; `Widgets/settings_image_gen_panel.py:70`.
- Evidence: `grep -rn "^def get_image_generation_config\|^def list_image_models_for_catalog\|…" tldw_chatbook | grep -v Image_Generation/` → the 4 modules above. Drift: personas/settings/panel wrappers forward `*args, **kwargs`; `Console_Modules/image.py:47` takes none.
- Why it matters: four places to keep in step when `Image_Generation.config` changes its signature; the excerpt's dup_verbatim rows are this.
- Recommended correction: a PEP 562 lazy `__getattr__` on `tldw_chatbook/Image_Generation/__init__.py` (the repo's own `Local_Ingestion` precedent, per the brief's known-deliberate list), then plain `from ...Image_Generation import get_image_generation_config` at the four sites.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none

### P3 [D4b] — Three exports-dir writers in one file drifted (stamp precision, threading, error surface)
- Where: `_dictionary_export_worker:5682-5699` (stamp `%Y%m%d-%H%M%S`, sync write, `detail.set_status`), `_export_expression_set:13531-13546` (stamp `%Y%m%d-%H%M%S-%f` — deliberate, documented 13532-13534; sync write; raises), `_lore_export_worker:6742-6767` (user path via `validate_path_simple`, `to_thread` write, `_notify`).
- Evidence: read side by side (lines above); the stamp drift is the excerpt's two `strftime` rows.
- Why it matters: the second-precision stamp can collide on two exports of the same dictionary in one second (the expression-set path fixed this for itself and documented it; the dictionary path still has it); threading drift is the P2 D2 finding.
- Recommended correction: one `_atomic_export_write(target: Path, payload: str | bytes) -> Path` on the screen (threaded temp+replace, `%f` stamp built by the caller), used by all three.
- Size: S · ADR: no · Confidence: verified
- Pinning test: `Tests/UI/test_personas_dictionaries.py::test_export_json_writes_file_and_reports_path` (glob-matches the filename; indifferent to stamp precision)
- Already covered: none

### P3 [D3] — Nine `_…_generation` fence counters, bumped inline at 27 sites; two have helpers, seven do not
- Where: helpers `_next_character_page_generation:3480`, `_advance_persona_buddy_session:4606`; inline bumps for `_character_editor_generation` (7443, 10222, 12467, 12527, 12564, 12588, 15343, 15719), `_actor_pack_generation`, `_actor_pack_portrait_generation`, `_persona_visual_generation`, `_character_tts_request_generation`, `_dictionary_lore_request_generation`, `_center_view_lifecycle_generation`.
- Evidence: `re.findall(r"self\._[a-z_]*generation \+= 1")` → 27 sites, 9 distinct counters.
- Why it matters: consistency only; `_reset_expression_generate_style`'s docstring (13331-13338) already has to enumerate which of the 8 `_character_editor_generation` bumps mean "new session" vs "invalidate render" — a helper per meaning would make that distinction structural.
- Recommended correction: none required; if touched, `_bump(name) -> int` or per-counter helpers.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none
- Already covered: none

### P3 [D3] — Post-attach editor re-sync swallows its DB read and editor lookup silently, leaving the editor's optimistic-lock version stale
- Where: `_sync_character_editor_dictionaries:6058-6073` and `_sync_character_editor_worldbooks:6209-6224` — `except Exception: return` twice each, no log.
- Evidence: read only. The attach itself (6009 / 6162) is already surfaced to the user; only the follow-up `editor.sync_attached_*(…, record.get("version"))` is lost.
- Why it matters: if the re-read fails, the open editor keeps the pre-attach `version`; the next Save hits `ConflictError` → "the character changed since it was loaded" with no cause visible in the log.
- Recommended correction: `logger.opt(exception=True).debug("Editor attachment re-sync failed …")` before each `return`.
- Size: S · ADR: no · Confidence: inferred (ConflictError chain read, not reproduced)
- Pinning test: none
- Already covered: none

## Candidate dispositions
| candidate (file:line pattern) | confirmed / retired (why) / unverified (check) |
|---|---|
| dup_shape `_handle_conversation_open_library/_resume/_continue_console` 7027-7039, `_handle_preview_reset/_open_console/_configure` 7365-7381 | retired — six 2-line `event.stop(); controller.x()` delegators; idiomatic Textual handler shape, no drift, delegation to task-118's extracted controller is the point |
| dup_shape `_next_character_page_generation` 3480 / `_advance_persona_buddy_session` 4606 | retired as a clone (two `+= 1; return` fences are intentional); folded into P3 D3 "nine counters, two helpers" |
| dup_shape `_handle_page_changed` 4108 | retired — `stop(); await delegate` |
| dup_shape `_reset_expression_generate_style` 13321 | retired — 2-line reset with a 17-line docstring explaining exactly which 3 of 8 bump sites call it; not a drifted copy |
| dup_verbatim `get_image_generation_config` 380 ≡ settings_screen:464 | confirmed → P3 D4b (4 modules, 8 shims) |
| dup_verbatim `list_image_models_for_catalog` 388 ≡ Console_Modules/image.py:55 | confirmed → same P3 D4b |
| except_exception_pass 2074 | retired — guards `self.notify` inside the initial-load error handler |
| except_exception_pass 4447 | retired — resets the search `Input.value` on mode switch; cosmetic |
| except_exception_pass 11780 | retired — awaits an already-cancelled generation task during shield/drain; the CancelledError is re-raised on 11782 |
| except_exception_pass 13831 | retired — same `Input.value` reset as 4447, post-import |
| except_exception_return ×11 (862, 2723, 2744, 3590, 6061, 6072, 6212, 6223, 9736, 14006, 15850) | 862 retired (error returned in `_DrainedTaskResult`); 2723/2744/9736 retired (fail-closed authority guards → False); 3590 retired (DTO `model_dump` shape); 14006 retired (cosmetic lorebook note); 15850 retired (teardown tolerance in `_show_center`); **6061/6072/6212/6223 confirmed → P3 D3** |
| function_body_import ×59 | confirmed present, **all targets exist** (26 distinct modules `ls`-verified, 9 symbols grep-verified, `textual_image.widget` importable); none dead. 3 are cycle/optional shims (`textual_image`, `contextlib`, `mimetypes`), the rest defer heavy modules (file picker, Image_Generation, buddy conversion, Backup_Recovery participants) — classified deliberate lazy-cost, not a finding. The `Backup_Recovery` ones import **private** names → P3 D3 |
| legacy_markers 13 (grep -i finds 8) | retired — comment references to the retired CCP route (`ccp_character_handler.handle_import`, "legacy CCP export route"); no `legacy` code path in this file |
| raw_1024x1024 492/494/495 | retired — three `MAX_BYTES` constants |
| raw_1024x1024 14082/14199 | retired — copy formatters `// (1024*1024)` for the user message |
| raw_mkdir 5688 | confirmed as part of P2 D2 (sync on loop); path is app-owned `get_user_data_dir()/exports`, not user input → no path_validation issue |
| raw_mkdir 13537 | confirmed as part of P2 D2; same |
| run_worker_coroutine 63 (AST: 64 + 3 `@work`) | confirmed count; **0 without `group=`, 0 `exclusive=True` without `group=`** (task-19559 holds); bodies with sync I/O on the loop → P2 D2 (3 sites); all other worker bodies use `asyncio.to_thread`/`_drain_to_thread` for DB/file work (read each) |
| strftime 5684 `%Y%m%d-%H%M%S` vs 13535 `…-%f` | confirmed drift, documented at 13532-13534 → P3 D4b (exports-dir writers) |
| try_import_guard 9522 `_preview_persona_visual_state` | retired — `except Exception` is the preview-op fallback to `set_preview_unavailable`, not an import guard |
| try_import_guard 10814 / 10880 / 11365 / 12420 (`textual_image.widget`) | retired — documented graphics→mosaic fallback (10886-10898); `textual_image` is installed in the venv |
| try_import_guard 15458 `_handle_policy_rules_changed` | retired — the in-try import shares the save's `except Exception` which logs + notifies "Policy rules save failed"; not swallowed |

## Verified-fine
- **4 Hz `set_interval(0.25, _poll_console_handoff_readiness)` (L1938)** — measured under env.sh: `get_provider_readiness("OpenAI"/"Anthropic", cfg, background_credentials=True)` = **0.010 ms/call**, Ollama 0.003 ms (N=200, warmed) → ≤0.04 ms/s of loop time; `_provider_send_block_reason` short-circuits to `None` unless a character/persona is selected and the action gate is open; the only `query_one` on the tick path (`_conversation_preview_is_open`) catches `QueryError`; guarded by `is_mounted and is_active`; stopped in `on_unmount` (2275). Pinned by `Tests/UI/test_personas_subscription_readiness.py:129-290`. **Not** a hot-path cost.
- Other timers: `set_timer` 4042 (debounce lambda → `_start_debounced_search_render`, no `query_one`, cancelled on unmount/mode switch) and 7007 (`_focus_conversations_list`, `query_one` wrapped in `QueryError`).
- `get_cli_setting`/`load_settings`: **0 calls** in this file (grep). `compose_content` (1529-1727) reads only `self.state` and module constants.
- All 64 `run_worker` + 3 `@work` carry `group=` (AST census) — task-19559 (Done) is not regressed here.
- Untrusted card/lorebook/dictionary reads: every user-picked path passes `Utils.path_validation.validate_path_simple(path, require_exists=True)` (13725 card, 13428 expression set, 14071 world book, 14188 dictionary, 10513 avatar); export writes pass `validate_path(target, base_directory=target.parent, redact_paths=True)` (14632) and PNG export defers to the lib's own base-directory check (14597-14609). Lorebook JSON goes through `normalize_world_book_import` (14103); dictionary JSON is type-checked (14224) then handed to the service. No URL fetch in this file → `Utils/egress` not applicable. Only gap is the card byte cap (P2 D1).
- `except Exception` census: 180 sites; the 15 `→ return/pass` all read (table above); none on a card-save or lorebook-import persistence path — every save/import/delete path logs with `opt(exception=True)` and notifies.
- `plain_readback`: no `.plain`, `str(label)` or `.renderable` read-backs in this file (grep) — the excerpt carried no rows and none apply.
- `candidate.cancel()` called synchronously on the loop (8666, 11506, 11560, 11566): in-memory only (`visual_identity.py` `cancel`: lock + `_cancelled` flag + `forget_cancelled`), no file I/O.
- `_sync_responsive_workbench` (2300-2363) on every `Resize`: ~15 `query_one` by id, all inside `try/except QueryError`; cheap.
- `ruff check --select E9,F63,F7,F82 personas_screen.py` → All checks passed.
- Recompose/`id()`-keyed caches: none — `_ready_center_views` is keyed by view name and validated with `is_mounted` (1964-1972); `_avatar_render_cache` keys are session-token strings. No mutable class attributes on `PersonasScreen` except `_WORKBENCH_FOCUS_TARGETS` (tuple), `_BULK_NOUNS` (dict, read-only use), `_MARKDOWN_LOSSY_FIELDS` (str).
- `on_mount` (1915-1949) is synchronous and defers the library read to a worker — task-1320's fix is in place here.

## Retired
- **"4 Hz poll is a hot-path cost"** — raised from the sibling `get_cli_setting` finding; retired by measurement (0.01 ms/call; `provider_readiness.py:501-561` takes the config mapping as an argument and reads no config/keyring per call).
- **"`_persona_visual_thread` / `_visual_identity_thread` close the shared DB connection from a worker thread"** as a D1 — retired as a defect: it is the ADR-126 native-borrower retirement contract (L317) and the same block ships in `Backup_Recovery/dictionary_source_job.py`; kept only as the D4b copy finding.
- **`except Exception: pass` ×4 as swallowed persistence errors** — retired; none touches persistence (table).
- **function-body imports of dead modules** — retired; all 59 resolve.
- **`storage._Acquisition` as a one-off private reach** — narrowed: 7 external users across TTS/ and this file → it is an unnamed public contract (P3 D3 recommends naming it), not a personas-only smell.

## Left UNVERIFIED
| claim | why not verified | literal command to run |
|---|---|---|
| P2 D2: exporting a large dictionary / expression set visibly stalls input for the write duration | needs the live app (do-not-run rule) | `cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook && tmux -L verify new -d -s v 'python3 -m tldw_chatbook.app'` per `.claude/skills/verify/SKILL.md`; seed a 50 MB expression set via `Tests/UI/test_personas_expression_slots.py` fixtures, open Roleplay ▸ Characters ▸ editor ▸ Export expression set, hold a key during the write and `tmux -L verify capture-pane -p` for dropped input |
| P2 D1: an oversized card file is read whole into memory before any check | memory effect not reproduced; only the code path was traced | `cd $WT && source <SCRATCH>/env.sh && PYTHONPATH=$WT $PY - <<'EOF'`<br>`from pathlib import Path; import resource, asyncio`<br>`p=Path("<SCRATCH>/big.png"); p.write_bytes(b"\x89PNG\r\n\x1a\n"+b"\0"*(300*1024*1024))`<br>`from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen  # then drive _import_character_from_path with a stub app whose _local_character_actions_allowed() is True and assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss grows by ≥300 MB`<br>`EOF` |
| P2 D4b: no test pins the `*_native_not_retired` retirement block | grep over Tests/ not run for that token | `cd $WT && grep -rn "native_not_retired\|retiring_threads" Tests/ \| head` |
| P3 D3 (6058-6073): a failed post-attach re-sync produces a spurious ConflictError on the next Save | chain read, not reproduced | `cd $WT && source <SCRATCH>/env.sh && $PY -m pytest Tests/UI/test_personas_dictionaries.py -q` after monkeypatching `chachanotes_db.get_character_card_by_id` to raise once inside `_sync_character_editor_dictionaries`, then pressing Save on the open editor and asserting the "changed since it was loaded" notify |

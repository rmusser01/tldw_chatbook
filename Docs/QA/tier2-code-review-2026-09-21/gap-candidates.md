# gap-candidates — tier-2 code review 2026-09-21

`✅` file it · `➖` file only for breadth · `❌` recommend against, with the design reason.
Grouped into **proposed batch tasks by canonical home, one task per PR**. **Nothing here has been filed.**

---

## Already owned — do NOT file these again

These tasks exist. Where this review has something to add, it is a **scope correction**, listed in the next section.

| Task | Status | Members this review confirmed |
|---|---|---|
| `TASK-32802.1`/`.4` markup escaping | In Progress / To Do | 6 files still on `rich.markup.escape` for markup-ON surfaces |
| `TASK-32803.5` adopt the timestamp helper | **Done** | 6 slices found non-adopters |
| `TASK-32804.12` remaining sync-work-on-the-loop | To Do | `Media/media_reading_scope_service.py` (117/138), `Notes/notes_scope_service.py` (11/14), `Character_Chat` + `Prompt_Management` scope services, `TTS/audiobook_generator`, `Scheduling/reminder_handler` |
| `TASK-32805.1` streaming handlers leak the response | In Progress | `tldw_api/` carries a **different** leak shape (see corrections) |
| `TASK-32806.5` local servers survive stop | In Progress | `TTS/audio_player.py` **P0**, `TTS/audio_cpp_supervisor.py`, `TTS/backends/chatterbox.py` |
| `TASK-32806.6` file reads/writes with no cap | In Progress | `Local_Ingestion` plaintext/HTML/MOBI, voice-blend import |
| `TASK-32806.8` image decompression bomb | In Progress | 3 new sites (see corrections) |
| `TASK-32807.*` delete dead code | 3 IP / 3 To Do | ~12 new clusters in packages none of the six sub-tasks names |
| `TASK-32808.1`/`.2`/`.3`/`.6`/`.9`/`.11` | mixed | measured members in the D4 table of `report.md` |
| `TASK-32809.2` size budgets | In Progress | **5 modules larger than existing rows are absent** |
| `TASK-32855` one strict-JSON parser | To Do | `tldw_api/` is a 4th family; the `Model_Artifacts/` drift is `parse_constant` |
| `TASK-32861` one StatusLine helper | To Do | non-adopters in S19, S20 |
| `TASK-32863` TTS convergence | To Do | **AC#1 already satisfied on dev**; AC#2 under-specified |

---

## ✅ BATCH 1 — Crashes and process safety *(one PR)*
**Canonical home:** the sites themselves. **Why one PR:** all are "the app dies or hangs", all small diffs.

1. **`TTS/audio_player.py:348,353,360`** — add `start_new_session=(os.name == "posix")` so the `killpg` at `:556`
   stops targeting the app's own process group. **P0.** Six in-repo precedents. `TTS/audio_cpp_supervisor.py:480`
   and `TTS/backends/chatterbox.py:284` get the same flag.
2. **`UI/Voice_Cloning_Window.py:560`** — bind the outer instance (or use `Widgets/confirmation_dialog.py`); the
   Delete-profile modal cannot compose today.
3. **`UI/Screens/stats_screen.py:590`** — gate the widget id; a non-ASCII chat topic kills the Stats screen at mount.
4. **`UI/Wizards/FirstRunSetupWizard.py:7110,7261`** — `exit_on_error=False` + an `is_mounted` recheck.
5. **`UI/Screens/scheduling/schedules_workbench.py:5393`** — same, for the `finally`-body `query_one`.
6. **`UI/Screens/watchlists_collections_screen.py:8111,8129,14213`** — give the three write workers the
   `try/except → notify` shape their ~30 siblings already use.
7. **`UI/Logs_Window.py:402`** — marshal `append_record` through `call_from_thread` when off the main thread.

## ✅ BATCH 2 — Data loss *(one PR)*
8. **`Library/library_rechunk_service.py:301`** — refuse a zero-row replacement instead of hard-DELETEing every
   chunk and reporting `"rechunked"`.
9. **`Library/collections_capture_repository.py:76`** — add `"interrupted"` to `_EXTRACTION_FAILURE_REASONS`;
   log the swallowed refusal at `collections_capture_service.py:333`.
10. **`Notes/server_notes_workspace_service.py:727`** — thread `expected_version` through, or drop the parameter.
11. **`TTS/backends/chatterbox_voice_manager.py:48`** — distinguish absent from unreadable; add the pre-write
    backup its Higgs sibling has.
12. **`TTS/backends/higgs_voice_manager.py:566`** — UTC backup names; select by parsed timestamp, not glob order.
13. **`Media/local_media_reading_service.py:5714,2792`** — stop the three silent export truncations.

## ✅ BATCH 3 — Security boundaries *(one PR; ADR-012 amendment cited, not re-decided)*
14. **`UI/Screens/settings_endpoint_probe.py`** — one `check_url_or_raise_async` in the chat branch, matching the
    TTS branch at `:514`. Closes all four callers at once.
15. **`Subscriptions/local_watchlists_service.py:2161`** — seed `trusted_origins` from the *subscription's* source,
    not the discovered `<loc>`; `URLMonitor._fetch_url_content` takes it fail-closed.
16. **`Web_Scraping/Confluence/confluence_auth.py:288`** — delete the branch predicate; fix the false-green test.
17. **`Web_Scraping/WebSearch_APIs.py`** — `allow_redirects=False` + a byte cap on the eight credentialed calls.
18. **`TTS/backends/{openai,elevenlabs}.py`** — route through `config.resolve_provider_api_key`; stored key over env,
    per ADR-012's 2026-09-19 amendment.
19. **`Evals/eval_runner.py:1855`, `Local_Ingestion/XML_Ingestion.py`, `Web_Scraping/Article_Extractor_Lib.py`,
    `Article_Scraper/crawler.py`** — `defusedxml` (a base dependency) and drop the matching `_KNOWN_UNHARDENED`
    entries in the same commit.

## ✅ BATCH 4 — Untrusted-input bounds *(one PR)*
20. **`Local_Ingestion/Image_Processing_Lib.py`**, **`Image_Generation/adapters/image_format_utils.py:140`**,
    **`Persona_Visual/importer.py:762`** — the three image ingresses without a pixel cap. The guard already exists
    in eight other modules; move `MAX_ASSET_DECODED_PIXELS` into `Persona_Visual/contracts.py`.
21. **`Local_Ingestion/local_file_ingestion.py:1488,1514`** + **`Book_Ingestion_Lib.py:2386`** — a
    `max_text_file_size_mb` symmetric with the audio/video caps that already exist.
22. **`TTS/backends/kokoro.py:482`** — compare the digest the code already computes; add a byte ceiling.
23. **`Character_Chat/Chat_Dictionary_Lib.py:174`** — call `world_info_regex.validate_regex_pattern` before
    `re.compile`. Fail-closed shape already in the sibling.
24. **subprocess timeouts** — `Media/local_media_reading_service.py:4680` (git clone),
    `STT/{parakeet_onnx,transcribe_cpp}.py`, `TTS/audio_service.py:412` (+ `stdin=DEVNULL`),
    `Audio/system_audio_tap.py:235`.

## ✅ BATCH 5 — `Utils/atomic_file_ops.py` hardening, then adoption *(one PR — order matters)*
25. **Harden first:** add the parent-directory fsync and the Darwin `F_FULLFSYNC` barrier, gated `private=True`
    for `O_EXCL|O_NOFOLLOW` + mode-at-open. **Three implementations in the repo are stronger than the shared one;
    adopting it as-is is a downgrade at those sites.**
26. **Then adopt:** `Notes/file_notes_service.py:576`, `Notes/template_store.py` (via
    `Backup_Recovery/raw_participants._replace` — **one fix closes both**), `Audio/voiceprint.py:235`,
    `Scheduling/scheduler_heartbeat.py:92`, `Video_Generation/video_store.py:683`,
    `LLM_Provider_Catalog/{models_dev_catalog,model_discovery_disk_cache}.py`,
    `Backup_Recovery/{raw_participants,mcp_source_participants}.py`.
27. `Petdex/review.py:170` and `Actor_Packs/publication.py:259` — lift `_fsync_parent` into the helper as part of 25.

## ✅ BATCH 6 — Guards *(one PR; the first three are REPAIRS to green, CI-required guards)*
28. **Fix `check_textual_worker_contract.py`'s W002 section test** — a `finally:`/`except:` body is not guarded by
    its own statement's handlers. Re-pin the 59 newly-visible sites as baseline, not as failures.
29. **Widen `check_timestamp_writers.py`** to the ADR-173 *contract*: add a kind for `.now(<any>).isoformat()` and
    for `strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"`. **Its census is currently empty and it reports OK while
    seven slices found writers.**
30. **Fix `check_canvas_mermaid_assets.py`** — 2 of its 6 "reproduced" outputs are copied out of the directory it
    then compares against. Either derive them from pinned inputs or move them to a *vendored, not derived*
    inventory with its own digests. Also bring `Canvas/static/canvas_shell.js` (51 KB, browser-side, holds the
    session cookie) under *some* integrity check.
31. Add `scripts/check_scope_service_contract.py` + a shrink-only exemption TSV (seed: 21 rows).
32. Add a ~20-line AST check for shadowed method definitions (`ruff F811` misses all 11).
33. **Add a carry-forward record for the local hand-edit in vendored `Chunking/engine/security_logger.py`** — the
    ADR-173 fix there reverts on the next `sync_chunking_engine.py` run, and the vendor test pins the commit sha,
    not per-file content.

## ✅ BATCH 7 — The scope-service scaffold *(ADR first, then one PR)*
34. **`TASK-32808.6`, re-scoped.** The ADR must settle: where the base lives; the dispatch predicate **and its
    fail-direction** (two documented rules disagree today); enum-vs-string mode **and the `mode=None` default**
    (25 SERVER / 19 LOCAL, with four `Interop`/`server_*` twin pairs inverted — **a base cannot hard-code it**);
    and the three runtime `TLDWAPIClient` importers (annotate `-> Any`). Fold in `_as_dict`, the gate funnel, and
    `run_finite_local_worker`. **~1,776 lines out, ~0 in.**

## ✅ BATCH 8 — Deletions *(one PR per package family)*
35. `UI/` root: `Stats_Window.py`, `MediaWindowV88.py`, `Chatbooks_Window.py`, `CodeRepoCopyPasteWindow.py`,
    `SiteConfigSettings.py` (+ the 4 `Tools_Settings_Window` satellites with TASK-32807.4).
34. `UI/Screens/schedules_screen.py` (516 lines, 0 importers, its stated reason for existing is false).
35. `MediaWindow_v2.py` + `media_screen.py` (2,725 lines; **the app's own pre-importer already skips the route**).
36. `Models/evaluation_state.py`, `Local_Inference/mlx_lm_inference_local.py`, `TTS/utils/` (512),
    `Audio/{dictation_metrics,console_dictation}.py`, `Prompt_Management/Prompt_Engineering.py` (**extract its
    metaprompt for task-474 first**), `Evals/{dataset_validator,ui_integration}.py`,
    `Local_Ingestion/API_Endpoint_Sample.py`, `Media_Creation/swarmui_client.py` + `image_generation_service.py`,
    `Character_Chat/ccv3_parser.py` + `handle_export_character`, 6 `*_Interop` modules (753 lines),
    `Backup_Recovery/credentials.py::restore_credential_values`,
    `Notes/notes_scope_service.py::_build_local_notes_graph`, `Notes/sync_paths.py`'s 5 dead methods.
37. **`Evals/` legacy run stack (~9,000 lines) — needs a ruling, not a PR.** Unreachable, and it carries a
    `subprocess` code-execution sandbox. ADR: `031-bounded-evaluation-and-tool-worker-execution.md`.
38. **`Widgets/Tamagotchi/` widget half (2,181 lines) — needs a product decision, not a cleanup.** The storage half
    is wired into backup/recovery and the private-SQLite allowlist.

## ➖ Breadth only
39. `Utils/Utils.py::ensure_directory_exists` — **delete** (0 importers across two reviews, 86 re-rolls).
40. The 67 broad-`except` DOM guards in S21 (dominant idiom is `except NoMatches`).
41. Diagnostics: the 71 `logger.*` calls whose kwargs reach no sink; `Notes/` 35-of-41 files with zero logging;
    `Persona_Buddy/controller.py` (8 swallows, 0 log calls in 1,444 lines).
42. `check_timestamp_writers.py` census is **empty while six slices found writers** — widen before declaring 32803 closed.

## ❌ Recommend against
43. **Do not consolidate `_perform_safe_cancel` (45 defs).** 28 distinct shapes; it is a template-method override
    against a shared base. Consolidation would be a regression.
44. **Do not consolidate `_initialize_schema` (16 defs / 2,120 LOC).** `@abstractmethod` on `DB/base_db.py:811`.
45. **Do not "fix" the 9 `_get_connection` overrides.** They call `super()`; the PRAGMA divergences are documented
    per-store decisions, one of which explicitly warns against copying it.
46. **Do not adopt `Utils/Utils.coerce_bool_flag` at the 11 `_coerce_bool` sites.** It stringifies ints, so `5` →
    `default` where the rail-state copies return `True`. TASK-32808.4's scoping was correct.
47. **Do not adopt `Utils/path_validation.safe_join_path` at `STT/executor_worker.py:503`.** The inline check is
    stricter (rejects Windows drive-relative `:`), and `validate_path`'s `allow_hidden=False` would break a GGUF
    under a dotted directory. Settle both properties first.
48. **Do not adopt `Utils/text.sanitize_filename` as the canonical sanitizer.** Measured, it is the **weakest** of
    the three — it passes `..`, NUL bytes and leading dashes. TASK-32808.2 needs **two** entry points (a validator
    that raises and a sanitizer that rewrites), not one.
49. **Do not apply TASK-32802.1's AC mechanically to `TTS/audio_cpp_supervisor.py:250`.** That sink is
    `markup=False`; swapping to the repo escaper **triples** the visible corruption. The fix is to **delete** the escape.
50. **Do not add a guard against a 48th `_maybe_await`.** The consolidation removes the reason to write one.

---

## Scope corrections to existing tasks (file as comments, not new tasks)
- **`TASK-32808.6`**: "the four helpers" → **six**; "~45 services" → **47**; "33 Interop + 12 core" → **31 + 16**;
  "~2,000 lines" → **1,776**; and **"the fix exists in exactly one of the services" → nine services thread, in five
  mutually incompatible shapes with two opposite fail-directions.**
- **`TASK-32809.2`**: add `tldw_api/client.py` (16,661 — repo rank 8, larger than a budgeted rank-9 file),
  `UI/Wizards/FirstRunSetupWizard.py` (10,404), `watchlists_collections_screen.py` (14,319),
  `TTS/profile_repository.py` (6,304), `TTS/TTS_Generation.py` (4,046),
  `Media/local_media_reading_service.py` (6,752). **The ratchet is 5-red at this SHA — 32809.1's re-pin is
  outstanding, and these rows should land in the same commit.**
- **`TASK-32806.8`**: scope names two files; three other image ingresses have no pixel cap.
- **`TASK-32808.5`** is marked Done but **the helper it drove adoption toward is the weakest of three
  implementations in the repo.** Re-open as "harden the helper", not "adopt the helper".
- **`TASK-32802.1`** AC#2 is keyed on *"sites that currently call `rich.markup.escape`"* — which structurally cannot
  find the sites that call **no** escaper (`UI/STTS_Window.py:1435`, `UI/Voice_Cloning_Window.py:647`,
  36 unescaped `notify()` calls in `change_review_screen.py`).
- **`TASK-32805.5`** + **ADR-175**: `tldw_api/` is a fourth strict-JSON boundary, in neither the adopter list nor
  the deliberate-exception list.
- **`TASK-32863`**: AC#1 is already satisfied on dev; AC#2 as written would **remove** Higgs's 300 s duration cap.

# S03 + S04 — TTS validation

## 1. P0 — `SimpleAudioPlayer.stop()` SIGKILLs the whole process group
- Verdict: CONFIRMED
- Site now: `tldw_chatbook/TTS/audio_player.py:556-558` (`os.killpg(os.getpgid(self._current.process.pid), signal.SIGKILL)`), children spawned at `:348,352,360` (mpv/mplayer/default `Popen` branches) — all unchanged line numbers.
- Proof: none of the 3 `subprocess.Popen(...)` calls pass `start_new_session=True`. `grep -n "start_new_session" tldw_chatbook/Agents/run_hooks.py tldw_chatbook/Skills_Interop/skill_script_runner.py tldw_chatbook/Notes/git_process_containment.py tldw_chatbook/Agents/agent_worktree_git.py tldw_chatbook/Tools/git_tool_impls.py tldw_chatbook/Web_Server/artifact_share.py` → 13 hits across those 6 files, confirming the repo's established pairing pattern that TTS's `audio_player.py` alone omits before its lone `killpg` call.

## 2. P1 — a transient read error on the Chatterbox voice-profile file destroys every profile; no backup
- Verdict: CONFIRMED
- Site now: `tldw_chatbook/TTS/backends/chatterbox_voice_manager.py:43-59` (`load_profiles`, `except Exception as e: ... self._profiles_cache = {}`), `:62-70` (`save_profiles`, no `_create_backup`).
- Proof: `grep -n "_create_backup\|def save_profiles\|def load_profiles" chatterbox_voice_manager.py higgs_voice_manager.py` → Higgs `save_profiles` (:100-105) calls `self._create_backup()`; Chatterbox's `save_profiles` (:62-70) has no such call anywhere in the file.

## 3. P1 — Kokoro model checksum computed then explicitly discarded; no size cap, no fsync
- Verdict: CONFIRMED
- Site now: `tldw_chatbook/TTS/backends/kokoro.py:482-488` (model) / `:546-547` (voices file) discard the computed digest with `logger.warning("Checksum verification skipped - no known checksum available")`; downloader `_kokoro_stream_download` at `:135-224`, `os.replace(partial, destination)` at `:209`.
- Proof: `grep -n "fsync" kokoro.py` → 0 hits. The download loop (`:184-191`) has no byte ceiling on `written`. `os.replace` at line 209 has no preceding `os.fsync(handle.fileno())` and no parent-directory fsync after.

## 4. P1 — audiobook generation runs the whole pipeline on the event loop; `ffmpeg` has no timeout/stdin guard
- Verdict: CONFIRMED
- Site now: `tldw_chatbook/TTS/audio_service.py:412` (`subprocess.run(cmd, capture_output=True, text=True)` inside `async def create_m4b_with_chapters` at `:321`) — unchanged line numbers.
- Proof: `grep -n "to_thread|run_in_executor" audio_service.py audiobook_generator.py` → 0 hits in either file. The `subprocess.run` call at :412 passes neither `timeout=` nor `stdin=subprocess.DEVNULL`.

## 5. P1 — OpenAI/ElevenLabs TTS backends read credentials env-first, contradicting ADR-012's 2026-09-19 amendment; gate is truthiness-only
- Verdict: CONFIRMED
- Site now: `tldw_chatbook/TTS/backends/openai.py:78-116` (env var `os.getenv("OPENAI_API_KEY")` checked first at :79, `api_settings.openai.api_key` only checked third at ~:96, after env and a raw config-dict lookup); gate `TTS/base_backends.py:129-133` `_validate_api_key`: `if not self.api_key: raise ValueError(...)`.
- Proof: `grep -rn "resolve_provider_api_key" tldw_chatbook/TTS/*.py tldw_chatbook/TTS/backends/*.py` → 0 hits, confirming the TTS backends never call the validity-checked resolver. `backlog/decisions/012-provider-credential-settings-boundary.md:100,124` confirms the amendment text: "the stored `api_key` outranks the environment variable it names."

## 6. P1 — Higgs voice-profile backups named with naive local time; DST fall-back can overwrite/misorder
- Verdict: CONFIRMED
- Site now: `tldw_chatbook/TTS/backends/higgs_voice_manager.py:566` (`datetime.now().strftime("%Y%m%d_%H%M%S")`), `:571-574` (lexical `sorted(self.backup_dir.glob(...))`, prune to last 10), `:589-594` (`backups[-1]` picked as most recent in `restore_from_backup`). Sibling naive sites: `:426`, `chatterbox_voice_manager.py:373`.
- Proof: `profile_repository.py:3771` correctly does `restored_at.astimezone(UTC).strftime(...)` — confirms the repo has the correct pattern elsewhere and Higgs/Chatterbox don't use it (exact line matches to review).

## 7. P1 — audio.cpp diagnostics are Rich-escaped into a `markup=False` RichLog (double-negative corruption)
- Verdict: CONFIRMED
- Site now: escape at `tldw_chatbook/TTS/audio_cpp_supervisor.py:250` (`escape_rich_markup`, imported at :18); sink at `UI/Speech/audio_cpp_runtime_card.py:603-611` (`RichLog(..., markup=False, id="audio-cpp-diagnostics-lines")`), written via `diagnostic_log.write(line)` at `:680-683`.
- Proof: direct read confirms the `RichLog` is constructed with `markup=False` and receives lines that already passed through `_sanitize` → `escape_rich_markup`, so any `[...]` in diagnostic text is escaped once but rendered literally (markup is off, so the escape backslash is never consumed).

## 8. P2 — `TTS/utils/` package (512 lines) is dead in production; its downloader can never succeed
- Verdict: CONFIRMED
- Site now: `TTS/utils/performance.py` (240 lines), `TTS/utils/download_models.py` (272 lines); `bytes_downloaded != expected_size` check at `:117` against rounded constant `311_000_000` at `:30` (unchanged from review).
- Proof: `grep -rn "get_performance_tracker\|PerformanceTracker" tldw_chatbook/ Tests/` excluding the defining file → 0 hits. `grep -rn "download_models" tldw_chatbook/ Tests/` excluding its own file → only 2 hits, both fencing/retirement tests (`test_recovery_entrypoints.py:27`, `test_task2062_3_legacy_downloader_retirement.py:106`), no production caller.

## 9. P2 — three reference-audio validators with three different bound sets; bounded validator unused by two
- Verdict: CONFIRMED
- Site now: `TTS/sample_audio_validation.py:21` (`MAX_PLAYABLE_AUDIO_BYTES = 8 * 1024 * 1024`); `voice_manager_base.py:171-201` `validate_audio_file` (extension allowlist only, no size/duration cap); `higgs_voice_manager.py:472-505` `_validate_audio_file` — the `SOUNDFILE_AVAILABLE` branch (common installed case) only checks `info["duration"] > 300`, no size cap; the 100MB size cap + extension allowlist live only in the `else` (soundfile-unavailable) branch at `:498-505`.
- Proof: direct read of both branches confirms the asymmetry exactly as described.

## 10. P2 — task-32863 AC#1 already satisfied on dev; task is stale as filed
- Verdict: CONFIRMED
- Site now: `ls tldw_chatbook/TTS/backends/ | grep -i isolated` → no match; `chatterbox_isolated.py` does not exist.
- Proof: `backlog task 32863 --plain` shows AC#1 (`chatterbox_isolated.py deleted with evidence of zero references`) still unchecked (`[ ]`) despite the file already being absent from the worktree — task file is stale on this AC.

## 11. P2 — the two largest TTS modules (`profile_repository.py`, `TTS_Generation.py`) have no size-ratchet row
- Verdict: CONFIRMED
- Site now: `Tests/Architecture/test_module_size_ratchet.py:40-47` `_BUDGETS` has no entry for either `TTS/profile_repository.py` (6304 lines) or `TTS/TTS_Generation.py` (4046 lines); smallest existing row is `mcp_workbench.py: 6760` (actual 6774 lines).
- Proof: `pytest Tests/Architecture/test_module_size_ratchet.py -q` → `5 failed, 9 passed` — exact match to the review's cited baseline (console_chat_controller, mcp_workbench, personas_screen, console_transcript, and app.py's slack test all still red), confirming the re-pin (TASK-32809.1) is still outstanding.

## 12. P2 — five intra-TTS duplication clusters with no shared home
- Verdict: CONFIRMED
- Site now: cluster 1 (`_sidecars_absent`) at `profile_migration_recovery.py:172`, `profile_migration_publication.py:179`, plus `DB/private_sqlite.py:2773 _profile_destination_sidecars_absent` and `_SIDECAR_SUFFIXES` at `:675`; `profile_migration_native.py:176,409` already imports other symbols from `private_sqlite`. Cluster 5 (the "freeze" family) confirmed present and semantically identical at all 5 cited sites (`console_turn_context.py:155 _freeze`, `playground_types.py:35 _freeze_option`, `preferences.py:21 _freeze_value`, `adapter_registry.py:92 _freeze_configuration_value`, `effective_settings.py:182 _freeze_value`) — all recursively wrap `Mapping`/`Sequence` into `MappingProxyType`/tuple with only naming differences.
- Proof: reads of all 5 "freeze" functions and the `_sidecars_absent`/`_SIDECAR_SUFFIXES` sites, as cited above.

## 13. P2 — neither TTS subprocess spawn uses `start_new_session`
- Verdict: CONFIRMED
- Site now: `TTS/audio_cpp_supervisor.py:480` (`asyncio.create_subprocess_exec`, no `start_new_session`... note: `asyncio.create_subprocess_exec` doesn't take that kwarg the same way as `Popen` but no `start_new_session=True`/process-group flag is passed either); `TTS/backends/chatterbox.py:284` similarly no flag.
- Proof: `grep -n "start_new_session" audio_cpp_supervisor.py backends/chatterbox.py` → 0 hits in either. `app.py:15088` calls `close_tts_resources()` (imported at :408) — confirms the "retired half" of the review's claim: the supervisor *is* stopped at shutdown, only the process-group flag is missing (prerequisite for the P0 fix, not an independent leak).

## 14. P2 — `_tts_provider_for_model`/`_tts_internal_model_id` are two more undocumented copies of a mapping with a real gap (`audio_cpp` unhandled)
- Verdict: CONFIRMED
- Site now: `Audio_Services_Interop/local_audio_services_service.py:184-190` and `Media/local_media_reading_service.py:5210-5216` — both copies' normalized-provider set is `{"kokoro","chatterbox","alltalk","higgs"}` plus an `elevenlabs` prefix check, falling through to `"openai"` otherwise.
- Proof: `TTS/provider_ids.py` `BUILT_IN_TTS_PROVIDER_IDS` includes `"audio_cpp"` as a shipped provider, but neither copy's normalized set contains it, so `_tts_provider_for_model("audio_cpp")` returns `"openai"` — confirmed unhandled.

## 15. P3 — audiobook FFMETADATA interpolation has no escaping
- Verdict: CONFIRMED
- Site now: `tldw_chatbook/TTS/audio_service.py:357-362` (`metadata_content += f"{key}={value}\n"` inside `";FFMETADATA1\n"` content, no escaping of `=`, `;`, `#`, `\`, or newline).
- Proof: direct read — no `.replace()`/escape call anywhere in that block before interpolation.

## 16. P3 — four strict-JSON decoders with different guarantees; artifact catalog is missing `parse_constant`
- Verdict: CONFIRMED
- Site now: `audio_cpp_contract.py:155,158` has `parse_constant=_reject_json_constant`; `audio_cpp_managed_config.py:269,272` has `parse_constant=_reject_non_json_constant`; `voice_bundle_codec.py:303-327` has both `parse_constant` and a depth cap; `audio_cpp_artifact_catalog.py:358-368` has `object_pairs_hook=reject_duplicate_keys` but **no** `parse_constant=` kwarg.
- Proof: `grep -n "parse_constant" <all 4 files>` → present in 3, absent in `audio_cpp_artifact_catalog.py` — matches the review's table exactly, including its own "not reachable" caveat (in-package manifest only, `_positive_integer` requires `type(value) is int`).

## 17. P3 — a `print()`-based demo CLI ships inside a production voice-manager module
- Verdict: CONFIRMED
- Site now: `tldw_chatbook/TTS/backends/higgs_voice_manager.py:631-648` (multiple `print(...)` calls); registered as an ADR-126 fencing route at `Tests/Architecture/test_recovery_entrypoints.py:28` — exact line match to review.
- Proof: direct read + grep, matches claim verbatim.

TOTALS: confirmed=17 fixed=0 wrong=0 demoted=0 promoted=0

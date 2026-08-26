---
id: TASK-3306
title: >-
  Expose remaining audio/video ingest tunables: time-range trim, URL cookies,
  recursive summary, adaptive/multi-level chunking
status: Done
assignee:
  - '@claude'
created_date: '2026-08-07 19:30'
updated_date: '2026-08-09 14:16'
labels:
  - library
  - ingest
  - parity
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred remainder of the 2026-08-07 options-parity audit (matrix in `.impeccable/critique/2026-08-07-media-ingest-ux-options-review.md`; owner scoped the high-value subset to task-3303). `process_audio_files`/`process_videos` accept, and the Library UI cannot reach: `start_time`/`end_time` trim, `use_cookies`/`cookies` for gated URL downloads, `summarize_recursively`, and the adaptive/multi-level chunking + `chunk_language` keys the pipeline reads from `chunk_options` but the app never populates. Also capped whisper model list (no large-v3/distil/turbo) and the permanently-closed `parakeet_defaults_enabled` promotion gate (no production caller).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each listed tunable is either exposed in the audio/video panel and wired to the processor call, or explicitly rejected in this task's notes with the reason recorded
- [x] #2 Any exposed option round-trips persisted defaults and has a wiring test against the real call signature
- [x] #3 Whisper model choices cover the models the routing layer actually accepts
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-verify each tunable against post-arc code (done: trim+cookies+recursive live in processors; adaptive/multi_level dead at _chunk_text; chunk_language metadata-only)
2. RED tests: capabilities shape, builder wiring, validator trim format, parse->processor wiring, video double-trim regression
3. Schema: start_time/end_time/cookies_file/summarize_recursively fields + whisper model list extended to the service catalog
4. Wire builder (_ingest_job_options) + parse_local_file_for_ingest (audio+video); fix _process_single_video double-trim; cookies=path-only, video branch only
5. Reject adaptive/multi_level/chunk_language with tripwire test
6. Mutation-check one wiring line + one gate; keep-green battery; docs + notes
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Per-tunable outcomes (each re-verified against post-arc code, not the parity audit):

EXPOSED — start_time/end_time ("Start at"/"Stop at", audio_video panel): text fields, format-gated at the shared validator seam (validate_ingest_option_value grew a trim-time branch: plain seconds or [HH:]MM:SS[.frac], the exact forms ffmpeg -ss/-to/-t and yt-dlp postprocessor args accept; blank = unbounded). Wired _ingest_job_options -> parse_local_file_for_ingest -> both processors. Found and fixed a latent DOUBLE-TRIM in the video path while wiring: _process_single_video extracts audio with the bounds applied, then delegated the same kwargs to _process_single_audio, whose own trim path re-cuts any local non-YouTube input — start=60s would shift the window to 120s. The bounds are now dropped after a successful extraction (kept when the input was already audio, where extraction never ran); regression-locked by test_video_extraction_trim_is_not_applied_twice.

EXPOSED — cookies as a FILE PATH ONLY ("Cookies file for gated URLs", cookies_file): raw cookie text REJECTED on security grounds — this options map persists with the job and echoes into [library.ingest_options.audio_video] in config.toml, and _process_single_video logs its kwargs at debug level, so a pasted cookie value would land in both. A cookiefile path is not a credential; yt-dlp consumes it natively (ydl_opts["cookiefile"]). Its presence IS use_cookies (no separate toggle to go stale). Wired to the VIDEO branch only: the audio downloader's cookies parameter has different semantics (json.loads -> raw Cookie header on a plain requests fetch — a path would raise JSONDecodeError and fail the job) and audio's YouTube path ignores cookies entirely; the hint says "video URLs only" and a wiring test pins that the audio branch never forwards it.

EXPOSED — summarize_recursively ("Recursive summary (map-reduce)", checkbox): live in the processor analysis tail (_analyze_content: per-chunk summaries + combine vs one direct call; governance test asserts 3 dispatches vs 1). Cross-group enabled_when on the generic Analyze toggle does NOT work (field_disabled_state resolves gates within cap.fields and per-group value maps only), so the dependency is hint-in-label per the 3303 convention; a stale True is inert because the processors only consult it when analysis actually runs. Legacy chunk_options["recursive_summary"] spelling kept as fallback.

EXPOSED — whisper model catalog: transcription_model select extended from 5 sizes to the 19-model faster-whisper catalog TranscriptionService.list_available_models declares (tiny..large incl. .en variants, large-v1/v2/v3, distil family, deepdml turbo + v3.5, nyrahealth CrisperWhisper), each with curated comma-free option_labels. The batch router passes an explicit faster-whisper model through untouched and the service hands it straight to WhisperModel, so nothing offered is rejected and nothing accepted was missing. parakeet_defaults_enabled untouched (out of scope); the "Auto (faster-whisper)" default label stays accurate.

REJECTED — adaptive + multi_level chunking + chunk_language: dead end-to-end on the audio/video path. process_audio_files accepts them, but _process_single_audio never reads use_adaptive_chunking/use_multi_level_chunking, and the only chunker (_chunk_text -> ChunkingService.chunk_text(content, chunk_size, chunk_overlap, method)) has no such parameters; chunk_language only lands in per-chunk metadata (never passed to the chunker) and defaults to the already-exposed transcription language. Output cannot vary with the input — exposing them would ship lying controls (the arc's core lesson). Tripwire test (TestAdaptiveChunkingStaysRejected) fails if the chunker ever grows these parameters, prompting a re-decision.

Files: tldw_chatbook/Library/ingest_capabilities.py (4 new audio_video fields + model catalog), tldw_chatbook/Library/library_ingest_state.py (trim validator branch), tldw_chatbook/app.py (_ingest_job_options audio_video branch), tldw_chatbook/Local_Ingestion/local_file_ingestion.py (audio+video processor calls), tldw_chatbook/Local_Ingestion/video_processing.py (double-trim fix), Docs/User_Guide/library/import-and-export.md (option list + 3306 stamp). Tests: 13 new wiring/governance tests (test_ingest_option_wiring.py), 7 new builder tests (test_submit_library_ingest_job.py), 2 validator tests (test_library_ingest_state.py), schema shape tests + pinned tuples (test_ingest_capabilities.py), config round-trip (test_library_screen.py). TDD: all wiring RED before implementation. Mutation checks: wiring line start_time->None caught; double-trim pop disabled caught; _TRIM_TIME_FIELDS emptied caught; all restored (grep-verified no markers). Battery: 488 passed / 0 failed across the six keep-green suites + test_library_screen.py. Known pre-existing failures NOT from this task (proven by neutralizing every 3306 edit in their dataflow and re-running): test_transcribe_cpp_ingestion.py::test_manual_library_job_reaches_fake_native_model_and_parent_writer and test_audio_model_dir_routing.py::test_audio_processor_passes_model_directory_to_transcription (provenance-kwargs drift; the latter never touches any 3306 file).
**xhigh review round (2026-08-09):** three confirmed defects in this task's
own new code, all fixed under TDD with READ reds.

1. **P0 — the cookies option destroyed the user's file.** Routing a
user-owned path into `download_video(cookies=...)` walked straight into a
cleanup `finally` that unlinked any `cookiefile` whose path merely *started
with* `tempfile.gettempdir()`. That heuristic had been safe only while the
key could hold nothing but a temp file the function itself wrote; a user who
exported cookies to `/tmp/cookies.txt` lost it on the first import (the
unlink failure swallowed into a debug log), and the second gated import then
failed with "Invalid cookie format". Ownership is now explicit:
`_resolve_cookiefile` returns `(cookiefile, owned_temp_path)` and only the
owned path is ever deleted. Mutation-checked: restoring the prefix heuristic
turns the two survival tests red.

2. **"Stop at" meant two different things.** `_extract_audio_from_video`
emitted `-ss` BEFORE `-i` (input seeking rebases output timestamps to zero)
and then `-to` as an OUTPUT option, so Start 0:30 / Stop 1:00 selected
0:30–1:30 on an .mp4 while the same pair selected 0:30–1:00 on an .mp3
(`_extract_time_range` puts `-ss` after `-i`, where `-to` is absolute).
Both paths now share one authority, `build_ffmpeg_trim_args` in
`audio_processing`, and "Stop at" is ABSOLUTE everywhere — what the label
promises. **Tradeoff, correctness first then speed:** absolute stop could
have been bought by moving `-ss` after `-i`, but output seeking decodes and
discards everything before the start (trimming the last minute of a 2-hour
file would decode 119 minutes). The fast pre-input seek is kept and the
absolute stop is converted to the duration it implies (`-t`), which is exact
under input seeking; an unparseable or inverted window falls back to
output-side seeking rather than silently reinterpreting the numbers.
Governance is on the constructed argv (ffmpeg and media files are not
guaranteed present): the test interprets both argvs with ffmpeg's own rules,
encoded independently of the builder.

3. **Cookies reached only the download.** The file-size probe immediately
before it constructed a bare `YoutubeDL({"quiet": True})`, and
`extract_metadata` ignored the `use_cookies`/`cookies` arguments it declares
— so an auth-gated URL, the option's only reason to exist, failed before the
cookied download ran. Both now take the same resolved cookiefile (and both
clean up only their own temp file).

4. **The path was never validated.** `cookies_file` was forwarded verbatim
and `download_video` degraded a non-existent path into a JSON parse attempt
logging only "Invalid cookie format". Validation now happens at the option
boundary (`app._resolve_ingest_cookies_file`: `path_validation.
validate_path_simple` + an is-file check), a rejected path yields
`use_cookies=False` plus a `cookies_problem` reason that travels
options → payload → the done row ("cookies ignored: …", the same channel
`analysis_skipped_reason` uses), and the downloader itself now raises
`VideoDownloadError` naming the missing file instead of continuing
un-authenticated. **Ownership note:** the canonical home for per-field
validation is `validate_ingest_option_value` in `library_ingest_state`
(owned by a concurrent agent this round), but existence is not a format
question — a path can be well-formed when typed and gone when the job is
claimed — so the check belongs at claim time regardless.

Files: `Local_Ingestion/audio_processing.py` (new `parse_media_timecode` /
`build_ffmpeg_trim_args`; `_extract_time_range` uses them),
`Local_Ingestion/video_processing.py` (cookie ownership + probe/metadata
threading + shared trim builder), `Local_Ingestion/local_file_ingestion.py`
(`cookies_problem` → payload), `app.py`
(`_resolve_ingest_cookies_file`, `_ingest_job_options`,
`_library_ingest_done_progress`), `Docs/User_Guide/library/
import-and-export.md`. Tests: new
`Tests/Local_Ingestion/test_video_download_cookies.py` (9), new
`TestAVTrimArgvSemantics` + a cookies-problem payload test in
`test_ingest_option_wiring.py`, 4 new/corrected builder tests in
`test_submit_library_ingest_job.py` (the old test asserted an invented
`/home/u/cookies.txt` travels — exactly the input now rejected).
<!-- SECTION:NOTES:END -->

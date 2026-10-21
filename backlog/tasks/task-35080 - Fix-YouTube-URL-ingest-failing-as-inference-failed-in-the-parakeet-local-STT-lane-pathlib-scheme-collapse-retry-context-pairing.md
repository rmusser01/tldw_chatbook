---
id: TASK-35080
title: Fix YouTube URL ingest failing as inference_failed in the parakeet local-STT
  lane (pathlib scheme collapse + retry context pairing)
status: Done
created_date: 2026-10-21 22:06
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Library ingest of a YouTube URL with transcription_provider=parakeet-onnx fails with the misleading `code=inference_failed` at phase=transcribing, seconds after submit. Root cause: the local-STT lane wraps the job source as FileAudioSource(Path(job.source_path)) (app_ingest_queue.py); Path() collapses `https://host/...` to `https:/host/...`. The mangled string still passes the URL sniff in parse_local_file_for_ingest but the SSRF/egress guard rejects it instantly (`Egress blocked (scheme) for <invalid-url>`); the STT worker classifies that as generic inference_failed, hiding the cause. Reproduced 2026-10-21: collapsed URL fails in 0.2s, uncollapsed URL transcribes the same video end-to-end.

Secondary bug exposed by retrying such a failure with faster-whisper: _ingest_job_options builds transcription_context with `retry_of_job_id` taken unconditionally from the job while `retry_of_attempt_id` is only set when failure provenance exists; a retry of a failure that stored no provenance (all parakeet inference_failed failures) carries retry_of_job_id alone, which STT/persistence validation rejects after transcription: "retry_of_job_id requires retry_of_attempt_id" (ingest jobs 13, 14, 18, 19, 23 in the local jobs DB).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] parse_local_file_for_ingest restores an https:// URL whose scheme separator was collapsed by pathlib, so the video arm receives the original URL and the egress guard no longer rejects it
- [x] URL jobs dispatched through the parakeet-onnx / transcribe-cpp local-STT lane no longer fail with code=inference_failed
- [x] A retry job whose source failure stored no attempt provenance no longer produces transcription_context with retry_of_job_id set without retry_of_attempt_id
- [x] Regression tests cover both fixes and targeted suites pass
- [x] Real end-to-end verification: a pathlib-mangled YouTube URL passes the egress check and reaches the video processor
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: regression test in Tests/Local_Ingestion/test_local_file_ingestion.py — parse_local_file_for_ingest(Path(youtube-url)) must hand the restored https:// URL to the video processor (stub pattern from test_video_options_are_routed_to_processor).
2. RED: regression test in Tests/App/test_submit_library_ingest_job.py — retry job with retry_of_job_id set but no retry_source_failure_provenance must build transcription_context with both retry fields None.
3. GREEN: restore collapsed scheme in parse_local_file_for_ingest at the raw_source boundary (shared pipeline owns URL-vs-file decision; no STT contract changes).
4. GREEN: gate retry_of_job_id on failed_attempt in _ingest_job_options (app_ingest_queue.py) so the provenance pair stays valid.
5. Targeted suites: the two touched test files.
6. Evidence: rerun the real end-to-end parakeet repro with the mangled URL; confirm download+transcription succeed.

ADR required: no
ADR path: N/A
Reason: routine bug fixes restoring documented behavior at existing boundaries; no contract, storage, or interface decision changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

### Approach

Two minimal fixes at the boundaries that own the broken invariants; no STT contract changes.

1. **Scheme restoration at the shared pipeline boundary** (`Local_Ingestion/local_file_ingestion.py`): the local-STT lane transports URL jobs as `FileAudioSource(Path(job.source_path))`; `str(Path("https://host/..."))` collapses `//` to `/`. The mangled string still passes `_is_http_url` but `urlparse` sees no hostname, so a YouTube watch URL misclassifies as "article" and the article arm's egress guard rejects it (`Egress blocked (scheme) for <invalid-url>`); the STT worker maps that to generic `inference_failed`. `parse_local_file_for_ingest` now restores the RFC 3986 authority separator (`_COLLAPSED_HTTP_SCHEME`, http(s) only) before classification and rebinds `file_path` so the audio/video arms receive the restored URL. Fixing it here (rather than relaxing the `FileAudioSource` Path contract) covers every present and future caller that wraps a URL in `Path`.

2. **Retry-context pairing** (`app_ingest_queue.py::_ingest_job_options`): `retry_of_job_id` was taken from the job unconditionally while `retry_of_attempt_id` required failure provenance; a retry of a local-STT failure (which stores no failed-attempt provenance) carried `retry_of_job_id` alone and died AFTER transcription in STT provenance validation ("retry_of_job_id requires retry_of_attempt_id"). Both fields are now gated on the failed attempt actually having provenance; such retries run as unlinked attempts instead of invalid pairs.

### Evidence (TDD)

- RED then GREEN, watched both ways:
  - `Tests/Local_Ingestion/test_local_file_ingestion.py::test_pathlib_collapsed_url_scheme_is_restored_for_video_processor` (fails pre-fix showing the collapsed URL reaching the article arm; passes post-fix),
  - `Tests/App/test_submit_library_ingest_job.py::TestIngestJobOptions::test_retry_without_failure_provenance_omits_retry_of_job_id` (fails pre-fix on `retry_of_job_id == 'ingest-job-22'`),
  - plus a characterization guard: `test_retry_with_failure_provenance_pairs_both_retry_ids`.
- Targeted suites before vs after the change (git stash diff): `test_submit_library_ingest_job.py` 38 pre-existing failures identical; `test_local_file_ingestion.py` 6 pre-existing failures identical (this dev checkout's local environment fails the config-bootstrap-dependent routing tests; the sets are byte-identical with and without this change). Zero new failures. `test_stt_failure_diagnostics.py` / `test_youtube_stt_selection.py` / `test_ingest_option_wiring.py`: identical 80-failure pre-existing set.
- Real end-to-end (2026-10-21, this machine): the exact collapsed source `https:/www.youtube.com/watch?v=jNQXAC9IVRw` through `parse_local_file_for_ingest` with the managed Parakeet v2 int8 runner — pre-fix: `PermanentIngestError` in 0.2s; post-fix: restored URL classified as video, yt-dlp download, Parakeet transcript, 4.3s total.

### Follow-up noted, not done here

- The local-STT lane still classifies any worker exception without `error_detail` as retryable `inference_failed`; an egress/pipeline failure surfacing with its real (permanent) message would have made this diagnosable in September. Contract-level change; left for a dedicated task.
- A lessons entry ("URLs must never travel through `Path()`; `str(Path(url))` silently collapses the scheme") could be added to backlog/docs if the maintainer wants a home for it.

### Modified files

- `tldw_chatbook/Local_Ingestion/local_file_ingestion.py` (restoration + rebind)
- `tldw_chatbook/app_ingest_queue.py` (retry-context gating)
- `Tests/Local_Ingestion/test_local_file_ingestion.py`, `Tests/App/test_submit_library_ingest_job.py` (regression tests)

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->

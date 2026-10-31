---
id: TASK-20973
title: >-
  Two public media seams let a caller supply the URL that vouches for itself
status: Done
assignee:
  - '[rmusser01]'
created_date: '2026-08-22'
labels:
  - security
  - egress
  - media
  - architecture
priority: medium
dependencies:
  - TASK-19556
---

## Description

Source: raised by **TASK-19556**'s reviewer while verifying the yt-dlp egress
guard. Re-verified at `684c6aba4`.

TASK-19556 put the app's egress policy in front of the two yt-dlp seams, as
`check_url_or_raise(url, trusted_origins=origin_set(url))`
(`Local_Ingestion/video_processing.py:85`). The URL is its own trusted origin.
That is a deliberate and correct choice *for a URL the user typed into the
ingest form*: `config.py`'s `[web_security]` contract permits an explicitly
configured URL to be private, because an intranet media server is a legitimate
source. The function's own docstring states this reasoning
(`video_processing.py:56-62`).

The reasoning holds only as long as every URL arriving at that check is
user-entered. Today it is — but by **wiring, not by invariant**. Two public,
test-covered methods accept caller-supplied URLs and forward them with no
provenance check of any kind:

- `LocalMediaReadingService.process_video(urls=…)`
  (`Media/local_media_reading_service.py:1217`)
- `MediaReadingScopeService.process_video(urls=…)`
  (`Media/media_reading_scope_service.py:2112`)

Neither has an in-app caller today — the only production reference to a
similarly named method is the unrelated private
`BackendIntegration._process_video` — so nothing is exposed right now. But
`trusted_origins=origin_set(url)` means a URL that reaches that seam from
anywhere else instructs the policy to trust its own host, which is precisely
what the policy exists to refuse for untrusted input. Wiring either seam to any
source that is not the ingest Input — a config file, an API payload, a feed, an
agent tool — silently converts a correct guard into no guard, with no test
failing and no reviewer prompted to look.

This is the "safe because of who happens to call it" shape rather than "safe by
construction". The fix wanted is a provenance decision at the boundary, not a
larger denylist.

## Acceptance Criteria

- [x] A URL reaching the yt-dlp egress check is trusted as its own origin only
      when its provenance actually establishes that trust; a caller-supplied URL
      of unknown provenance is not self-trusting
- [x] The two `process_video(urls=…)` seams either establish provenance
      explicitly or cannot reach the self-trusting path
- [x] The trust decision is expressed where it is made rather than inferred from
      the current caller set, so adding a caller cannot silently change it
- [x] A test proves that wiring a new, non-user-entered source into these seams
      does not grant self-trust — i.e. the guarantee survives a hypothetical
      future caller, which is the property that does not hold today
- [x] Existing user-entered ingest of a private or intranet media URL still
      works, and a test pins it, so the fix does not close a legitimate case
- [x] The residual limits TASK-19556 documented (yt-dlp's own redirect hops,
      extractor-discovered per-format URLs, resolve-then-connect TOCTOU) remain
      accurately stated and are not implied to be closed by this work

## Implementation Plan

ADR required: yes (decided during implementation; see below — the initial
plan draft said "no", reversed after checking `backlog/decisions/` and
finding NO egress ADR from the TASK-19556 era to link: the provenance
vocabulary is a new, durable, cross-module security contract with a
minting rule and two deliberate non-changes reviewers will question
again, which is exactly what an ADR is for).
ADR path: backlog/decisions/207-url-provenance-seeds-egress-self-trust.md
Reason: `UrlProvenance` is a public contract spanning `Utils/egress.py`,
the whole video chain, the two Media service seams, the parse worker and
the ingest queue; the enum's membership rule, the single minting point,
and the intentionally-unchanged audio arm / server backend are decisions
that outlive this task.

1. Add a `UrlProvenance` enum (`USER_ENTERED` / `UNKNOWN`) and a
   `trusted_origins_for(url, provenance)` helper to `Utils/egress.py` —
   the policy module already owns the "trust is seeded only at boundaries
   and threaded down" contract, so the provenance vocabulary lives beside
   it. `USER_ENTERED` → `origin_set(url)`; everything else → `frozenset()`.
2. Thread an explicit, keyword-only `url_provenance` parameter
   (default `UNKNOWN`, i.e. fail closed) down the whole video chain:
   `check_media_url_egress` → `download_video` / `extract_metadata` ←
   `_process_single_video` ← `LocalVideoProcessor.process_videos` ←
   `LocalMediaReadingService.process_video(urls=…)` ←
   `MediaReadingScopeService.process_video(urls=…)` (forwarded to the
   local backend only; the server backend fetches in another process our
   egress policy cannot reach, so no provenance claim crosses that
   boundary).
3. Mint `USER_ENTERED` where user entry is a fact, not an inference:
   - `app_ingest_queue._ingest_job_options` derives it from the job —
     `USER_ENTERED` for general Library-import submissions,
     `UNKNOWN` for `research_source_operation_id` jobs (agent-discovered
     catalog URLs, the live non-user-entered source that shares the parse
     pipeline today).
   - The minted enum rides the pickled parse-`options` dict (the
     documented transport across the spawn pool) and is translated back to
     the explicit parameter by `run_parse_job`, which pops it and refuses
     non-`UrlProvenance` values (a plain string cannot launder trust).
4. Write born-red tests first: a new
   `Tests/Local_Ingestion/test_video_url_provenance.py` proving (a) both
   public `process_video(urls=…)` seams do NOT self-trust a private-IP
   URL by default (red today: both reach yt-dlp), (b) the parse seam
   fails closed without provenance, (c) the queue mints per submission
   lineage, (d) `USER_ENTERED` still reaches yt-dlp for a private URL
   through the full ingest path (pins AC 5). Update the two TASK-19556
   "user-typed private URL" pins and the one exact-kwargs
   `FakeVideoProcessor` assertion to the new explicit parameter, with the
   reason inline.
5. Run the affected suites before/after (`Tests/Local_Ingestion/`,
   the three `Tests/Media/` reading-service files,
   `Tests/Utils/test_egress_adoption_census.py`), A/B any pre-existing
   reds against the clean base via `git checkout HEAD --` swaps. State
   TASK-19556's residual limits as unchanged in the new test module's
   docstring and in `check_media_url_egress`'s docstring.

## Notes

## Implementation Notes

### Approach

An explicit `UrlProvenance` enum (`USER_ENTERED` / `UNKNOWN`) plus a
`trusted_origins_for(url, provenance)` helper in `Utils/egress.py` — the
policy module already owns the "trust is seeded only at boundaries and
threaded down" contract, so the vocabulary lives beside it. A
keyword-only `url_provenance` parameter, defaulting to `UNKNOWN` (fail
closed), threads down the whole chain:

`check_media_url_egress` ← `download_video` / `extract_metadata` ←
`_process_single_video` ← `LocalVideoProcessor.process_videos` ←
`LocalMediaReadingService.process_video(urls=…)` ←
`MediaReadingScopeService.process_video(urls=…)` (forwarded to the LOCAL
backend only).

`USER_ENTERED` is minted in exactly one place:
`app_ingest_queue._ingest_job_options`, derived from the job's own
lineage — general Library-import submissions are `USER_ENTERED`;
research-source jobs (`research_source_operation_id`) are `UNKNOWN`
(agent-discovered catalog content sharing the parse pipeline today — a
live instance of the "future caller" this task was filed over, now
correctly refused for private targets). The enum rides the pickled
parse-`options` dict across the spawn pool; `run_parse_job` pops it and
hands it to `parse_local_file_for_ingest` as the explicit parameter,
honouring it ONLY as a genuine enum instance (a plain string cannot
launder trust). `parse_local_file_for_ingest` itself never mints; direct
programmatic callers default to `UNKNOWN`.

Deliberate non-changes, recorded in ADR-207: the audio arm
(`audio_processing.download_audio_file`'s own `origin_set(url)`,
TASK-19556's parity choice — same shape, different seams, out of scope)
and the server backend (its URLs are fetched by a remote process our
egress policy cannot reach; a provenance claim crossing a process
boundary is unverifiable, so it is not forwarded).

### Evidence (exact commands and results)

Born-red, at the clean base with only the new test file present (the
`UrlProvenance` import neutralized via a one-line shim so each case fails
on its own property rather than at collection):

`python -m pytest Tests/Local_Ingestion/test_video_url_provenance.py`
(17 tests) → **17 failed**. The two seam tests failed with the private
URL REACHING yt-dlp (the child log showed the download run all the way
into real ffmpeg on the stub's 8-byte file) — the exact property this
task exists to remove. The queue-minting tests failed with
`KeyError: 'url_provenance'`.

After implementation: `python -m pytest
Tests/Local_Ingestion/test_video_url_provenance.py` → **17 passed**
(includes the AC-4 seam proofs, the AC-5 user-entered end-to-end pin
(classify → metadata → download → transcript of a private URL), the
research-source minting pin, the string-cannot-launder pin, and an AST
mutation guard on the egress call site).

Targeted neighbourhood, with clean-base A/B (failure-name diff via
`git checkout HEAD --` swaps of the source files only):

- `Tests/Utils/test_egress_adoption_census.py` → 7 passed (its
  "rediscovery" fixture's literal un-fix strings updated to the new call
  shape, per that test's own instruction; the mutation still removes
  exactly the census-visible symbol).
- `Tests/Local_Ingestion/test_youtube_stt_selection.py` → 4 passed
  (extract_metadata stub now accepts kwargs).
- `Tests/Media/test_media_reading_scope_service.py` → 108 passed;
  `Tests/Media/test_media_reading_scope_service_off_loop.py` +
  `test_server_media_reading_service.py` → 97 passed.
- `Tests/Local_Ingestion/test_video_egress_guard.py` +
  `Tests/Media/test_local_media_reading_service.py` → 11 failed /
  85 passed **identical to clean base** — every one is the
  environmental `RecoveryRequired: raw_source_selection_changed`
  profile-selection trip this machine exhibits at base (verified by
  swapping source files back to HEAD and re-running; failure-name sets
  byte-identical).
- `Tests/App/test_submit_library_ingest_job.py` +
  `Tests/Local_Ingestion/test_ingest_option_wiring.py` → 118 failed /
  117 passed **identical to clean base** (same environmental trip).
- `Tests/Local_Ingestion/test_transcribe_cpp_ingestion.py` → failure set
  identical to base after its parse stub learned `**kwargs` (one test had
  pinned `run_parse_job`'s forwarding shape positionally).
- `Tests/Performance/test_boot_budget_ratchet_messages.py` +
  `Tests/Local_Ingestion/test_ingest_import_weight.py` → passed (the
  `Utils.egress` import in `app_ingest_queue` and
  `Media/local_media_reading_service` stays within the budget;
  `local_file_ingestion`'s provenance import is deliberately lazy,
  inside the function body, to respect its guarded import weight).

Known environmental caveat, stated per
`backlog/docs/lessons-testing-evidence.md`: this machine's profile state
makes config-reading tests under the default sandbox red AT BASE
(`RecoveryRequired`), so the new tests use the `@private_profile_test`
wrapper (`test_youtube_stt_selection.py`'s pattern for this exact chain)
to run immune to it; their pre-fix failures were verified to be the
property failures, not the environmental trip.

### Files changed

Source: `tldw_chatbook/Utils/egress.py` (enum + helper + contract note),
`tldw_chatbook/Local_Ingestion/video_processing.py` (decision point +
threading), `tldw_chatbook/Local_Ingestion/local_file_ingestion.py`
(parse-seam parameter, lazily normalized),
`tldw_chatbook/Local_Ingestion/ingest_parse_worker.py` (transport
translation + schema doc), `tldw_chatbook/app_ingest_queue.py` (the
mint), `tldw_chatbook/Media/local_media_reading_service.py` +
`tldw_chatbook/Media/media_reading_scope_service.py` (the two public
seams; local-only forwarding).
Tests: `Tests/Local_Ingestion/test_video_url_provenance.py` (new, 17
tests), `Tests/Local_Ingestion/test_video_egress_guard.py`,
`Tests/Local_Ingestion/test_youtube_stt_selection.py`,
`Tests/Local_Ingestion/test_transcribe_cpp_ingestion.py`,
`Tests/Media/test_local_media_reading_service.py`,
`Tests/Utils/test_egress_adoption_census.py` (updated with reasons
inline).
Docs: `backlog/decisions/207-url-provenance-seeds-egress-self-trust.md`.

### Residual limits (unchanged, restated)

This changes WHO may vouch for the entry URL, not what happens after
yt-dlp starts: yt-dlp's own redirect hops, extractor-discovered
per-format URLs, and the resolve-then-connect TOCTOU window remain
TASK-19556-documented residuals, stated in
`check_media_url_egress`'s docstring and the new test module's docstring.
The audio arm keeps the same self-trust shape and is out of scope here.

PR #2993 review addendum: that scope left research-source URLs classified
audio or article still self-trusting (Qodo, confirmed by an independent
review), so the audio and article arms now consume `url_provenance` too;
the Collections quick-capture extractor passes `USER_ENTERED` explicitly.
ADR-207 is updated to match. Pinned in
`Tests/Local_Ingestion/test_video_url_provenance.py`.

Filed medium with no live exposure, deliberately. The severity is not what an
attacker can do today — nothing reaches these seams — it is that the security
property is held by a fact nobody is watching, one wiring commit from being
false, and that the commit which makes it false will look entirely ordinary.

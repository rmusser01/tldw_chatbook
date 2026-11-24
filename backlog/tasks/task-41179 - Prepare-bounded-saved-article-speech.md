---
id: TASK-41179
title: Prepare bounded saved article speech
status: Done
assignee:
  - '@codex'
created_date: '2026-11-15'
updated_date: '2026-11-24T23:45:00Z'
labels:
  - watchlists
  - tts
dependencies:
  - TASK-41178
references:
  - https://github.com/rmusser01/tldw_chatbook/issues/2864
documentation:
  - backlog/decisions/367-watchlist-article-listening-and-controlled-playback.md
  - Docs/superpowers/specs/2026-11-15-watchlist-article-listening-design.md
  - Docs/superpowers/plans/2026-11-15-watchlist-article-listening.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Read the stored article body faithfully using a stable voice while bounding generation cost and memory.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Text, HTML and Markdown prepare inert complete speech text with link labels, excerpt identification and useful empty or over-limit failures.
- [x] #2 Long admitted bodies synthesize completely through existing TTS paths within explicit text, chunk and aggregate buffer bounds; partial output is never playable.
- [x] #3 Saved nonsecret voice settings and authorized dependency identities survive restart or default changes; changed dependencies pause without silent substitution.
- [x] #4 Valid retained audio requires no provider request and late or cancelled results cannot publish after source or authority retirement.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/367-watchlist-article-listening-and-controlled-playback.md
Reason: implement the accepted speech snapshot, reconstructible voice/reference authority, bounded native/legacy synthesis and source-owned publication contracts; the dated Task41179 clarification applies before code. No new runtime package or schema version is planned.
1. Trace current saved-body, HTML/Markdown, effective TTS admission/default-profile/reference and WAV/lease APIs at parent8ce8a67efa. Record the Task3→4 and Task4→5 interface decisions; keep Task41180 transport/UI out of scope. Existing briefing synthesis/audio-limit baseline is29passing cases.
2. Write and observe real REDs in new test_article_speech.py and test_article_listening_audio.py, with real SQLite/files/WAVs and only provider transport faked. Cover full visible prose/code, inert markup, excerpts, raw/prepared/queue/chunk limits, frozen UUID/options/dependency changes, cached zero-call reads, cancellation/source retirement and aggregate buffer copies.
3. Implement pure bounded text preparation and backward-compatible inert FrozenVoice extensions. Reuse existing parsers and effective/default-profile admission; reacquire live credentials/reference material and generations, retain stable nonsecret UUID/revision/digests, and refuse unbindable aliases without silently selecting new defaults.
4. Implement one constructible audio-preparation owner using authoritative stored entry data and the exact issued reservation/lease. Reuse the native/legacy effective response and splitter, normalize sequential chunks to qualified PCM, enforce64MiB across current/retained/converted/joined buffers before allocation, append the tail once, and publish only after final source/voice/permission checks. Retain physical work on cancellation and require explicit regeneration for lost audio.
5. Run targeted new tests and affected briefing, resolver/admission, model/recovery and audio-limit neighbors serially through normal guarded pytest. Verify one configured provider with private synthetic prose and no playback; record availability/receipt truthfully. Run changed-file Ruff/format and relevant layer/owner guards. No full suite without opt-in.
6. Complete independent spec/quality review, resolve blocking findings with covering regressions, update ADR/architecture/task notes and mark all four criteria only when verified. Publish a separate draft child of PR3766. Watch/fix PR3766 CI independently, preserving the qualification evidence tag and existing reviewed parents.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented complete bounded speech preparation and source-owned synthesis for saved articles. Pure text/HTML/Markdown conversion preserves visible prose and code, identifies excerpts, and enforces text/request limits. Frozen nonsecret voice/profile/reference identities survive default changes; unsupported effective selections pause explicitly. Valid attached or reusable WAVs are checked before TTS, including explicit retries.

Article preparation owns the exact reservation, physical response/conversion cleanup and final attachment decision. Existing native mutation custody orders source deletion against provider start; commit and cancellation have one reported winner. Sequential conversion accounts for retained copies within 64 MiB and publishes PCM16 stereo 48 kHz with one 12,000-frame tail. Complete streamed WAVs with both unknown-length markers normalize using bounded physical frames; ordinary truncation and partial frames reject before decoding.

ADR required: yes. Existing ADR: backlog/decisions/367-watchlist-article-listening-and-controlled-playback.md. Architecture: Docs/Architecture/speech.md. Plan: Docs/superpowers/plans/2026-11-15-watchlist-article-listening.md. Main source: Subscriptions/article_speech.py and article_listening_audio.py, with narrow store/asset, parser/splitter and TTS admission/legacy seams. Added targeted speech/audio tests and a concrete testing-evidence lesson. No schema change or runtime dependency. Independent spec/quality review and scoped re-reviews approved source head 80614fc183.

Validation: normal guarded Windows selection passed 257 integration cases; after the final WAV-only fix, all 101 affected speech/assembly/casting/audio-limit cases passed. Separate recovery selection passed 7 cases. Changed code passes Ruff/format/AST/whitespace checks, with previously recorded whole-file legacy lint/format exceptions unchanged. The uninstrumented configured OpenAI smoke exited 0 in 9.50s: published_and_reused, 103199 content frames at 48000 Hz, 460840 retained bytes including the 12000-frame tail. A second owner with no TTS service returned the same generation and digest. Isolated synthetic storage was removed after worker retirement. No playback occurred.

First delivery supports streaming OpenAI-compatible speech and unit-speed ElevenLabs. Native whole-response audio.cpp, Kokoro engine/fallback, native Pocket form and other unbound routes refuse new generation before startup; verified saved audio remains reusable. Windows reference materialization follows existing platform refusal. Five existing provider constructor/manager checks stopped at raw_source_selection_changed before generation; parent equivalence was not run, and these are disclosed verification limits. Task41180 transport/UI and final playback workflow remain separate.
<!-- SECTION:NOTES:END -->

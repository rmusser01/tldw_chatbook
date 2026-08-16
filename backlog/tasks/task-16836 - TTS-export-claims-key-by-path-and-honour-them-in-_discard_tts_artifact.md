---
id: TASK-16836
title: 'TTS export claims: key by path and honour them in _discard_tts_artifact'
status: To Do
assignee: []
created_date: '2026-08-16'
labels:
  - bug
  - concurrency
  - tts
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two findings from the TASK-16199 review (PR #1676), merged here because one redesign
covers both (F2+F3 of that review; both re-verified at dev `ee741cf10`):

1. **`_discard_tts_artifact` is the last secure-delete path that ignores export claims
entirely** (`Event_Handlers/TTS_Events/tts_events.py:3112-3138`). It zeroes-and-unlinks
with no `_exporting_audio_refcounts` check, and is reachable from the cache-replacement
path (`:3045`), cancelled-artifact cleanup (`:2946`), stale-console-completion discard
(`:3147`), and several playback-error paths (`:2089`, `:2113`, `:2141`, `:2186`,
`:2238`). The TASK-15471 claim invariant is honoured by exactly two call sites
(`_cleanup_audio_file` and the 16199 shutdown sweep) — contradicting 15471's "verified on
every path" claim. It also means `cleanup_tts_resources`' own
`_drain_retained_tts_artifact_work()` awaits (`:3909`, `:3978`) can let a retained
discard complete during shutdown and zero a claimed file.

2. **The claim is keyed by message id, and the id→path join breaks exactly when it is
needed** (review F3). The export copies a *path*; the sweep's retry-pass protection maps
claimed ids through the current `_audio_files` (`:3931-3935`) — but `_discard_tts_artifact`
deletes the id→path cache entry in the same lock hold that adds the path to the retry set
(`:3131-3134`), so a retry-set path is normally no longer `_audio_files[id]` for any
claimed id, and the skip cannot fire for precisely the evicted-entry case. Mutation-
confirmed untested: disabling the retry skip left all 26 TTS-improvement tests green.

Fix direction: record the claimed **source path** with the claim (refcount keyed by path,
or store the path beside the id), and have every secure-delete path —
`_discard_tts_artifact` included — consult it. Note the whole surface is latent today
(the export path has no production poster — see the companion wire-or-retire task filed
alongside this one), but it becomes live the moment export is wired.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 No secure-delete path in tts_events.py can destroy a file an export currently claims — `_discard_tts_artifact` included (forced-interleave test or probe as evidence)
- [ ] #2 The protection survives eviction of the id→path cache entry (the F3 case is pinned by a test that fails against the id-keyed join)
- [ ] #3 The existing 15471/16199 claim tests stay green
<!-- AC:END -->

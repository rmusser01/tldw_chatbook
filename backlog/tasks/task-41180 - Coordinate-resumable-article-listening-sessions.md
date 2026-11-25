---
id: TASK-41180
title: Coordinate resumable article listening sessions
status: Done
assignee:
  - '@codex'
created_date: '2026-11-15'
updated_date: '2026-11-25T02:33:00Z'
labels:
  - watchlists
  - tts
dependencies:
  - TASK-41179
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
Let a cited queue and an interrupting article preserve their places and return behavior through navigation, interruption and restart.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Validated cited items play once in first-citation order, preserving unavailable placeholders and the old queue when replacement cannot succeed.
- [x] #2 Single article interruption, replacement, completion, Skip, Stop and failure preserve independent bookmarks and the agreed automatic-return behavior.
- [x] #3 Startup restores unfinished sessions paused without synthesis; completed sessions replay only explicitly, and checkpoints bound crash repetition.
- [x] #4 Profile or runtime changes, obsolete callbacks, competing instances and surviving native work cannot overlap output or mutate newer sessions.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/367-watchlist-article-listening-and-controlled-playback.md
Reason: implement the existing accepted session, source authority, output custody, atomic checkpoint and manual restoration contracts. Reuse ADR367 and current private storage/output owners; no new schema, manifest, dependency or UI surface is planned.
1. Trace the Task41179 store/audio APIs, output observations/ownership, citation provenance paging, app composition/shutdown and profile/runtime transitions against parent fb5b12d3fe. Record interface conflicts and decisions before production edits.
2. Write discriminating state-machine tests with real SQLite/files and deterministic output: complete cited order beyond 50, unavailable placeholders, replacement preservation, both interruption roles/independent bookmarks, idempotency and all terminal/error/EOF cases. Observe RED under ordinary guarded pytest.
3. Implement ArticleListeningService intents and immutable snapshot using existing owners, revision/generation checks and one retained physical operation. Preserve blocked cleanup ownership and fence authority away/back transitions; no new transport UI.
4. Persist observed frame positions off-loop at most once per second and on transitions. Restore unfinished state paused without generation/output. Wire lazy app composition and drain listening before TTS/DB teardown; coordinate supported source/profile/runtime mutation paths.
5. Run targeted service/output/storage/lifecycle/runtime tests and changed-code static/layer checks serially with source frozen. Kill a private native probe between checkpoints and verify manual recovery of the same retained WAV; keep simulation separate from actual native evidence. No full sweep without opt-in.
6. Complete independent spec/quality review, resolve blocking findings with covering regressions, update architecture/ADR/plan/task evidence and publish a separate draft child of PR3767. Continue watching PR3766/3767 CI independently; preserve existing reviewed parents and the immutable qualification evidence tag.

Preflight rulings (2026-11-24): the current store resolves all bounded citations directly; require runtime_backend on play_brief as well as listen. Reuse a held native lease with finite per-operation storage admission, add CAS fresh-epoch adoption/all-entry replay reset and narrowly bound old-owner terminal cleanup. Introduce Windows job-at-suspended-birth and a fixed bounded predecessor record in the existing private lock companion, durable before ResumeThread; verify exact predecessor death plus the qualified 250 ms settle before new grant. Exclude valid device-local custody from backup and reject malformed records. Readiness must precede new synthesis. Project factual stored-feed provenance; excerpt classification is unknown without evidence. See the dated session/crash contract in ADR367 and the updated Task5 main-plan section. Preserve private snapshots, existing registered retire callback, source semantics and the subsequent transport UI scope.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the app-owned local listening coordinator under ADR-367. It validates full first-citation queues, preserves independent queue/foreground bookmarks and return intent, checkpoints observed frames, restores paused without synthesis, and exposes read-only brief action guidance. Original runtime/profile/source authority, exact physical holder and output retirement fence all mutations, selected commands, destructive transactions and late callbacks.

Windows players enter a private kill-on-close job at birth and seal a bounded device-local predecessor receipt before release. Checked predecessor death and 250 ms settlement precede successor admission. Backups exclude native custody; installed replacement checks the original destination. Uncertain preparation retains its exact attempt/holder. Failed listening teardown preserves dependent barriers and drains independent owners.

Independent task review and two scoped fix rounds approved source at 72e7a049fe. The seven initial findings and new empty-session cancellation race have discriminating RED/GREEN coverage. Final ownership selection: 10 passed in 20.96 s. Prior affected selection: 63 passed/one old observer assertion, followed by both corrected provider variants and SQLite shutdown barrier passing on identical production; shutdown ratchet checks passed. New/changed coordinator tests and formatter checks pass; legacy lint/whole-file format diagnostics and eight baseline broad failures remain explicitly documented, with no new lint diagnostics. No full sweep was run.

Separate native process probes restored the exact saved frame for active/suspended death and startup Stop; these are custody/progress evidence, distinct from existing Windows loopback qualification. Configured OpenAI smoke used one request to publish 101999 frames at 48 kHz/456040 bytes; reopened manual Resume reused it with zero provider accesses, no device admission and verified cleanup. Source hashes for each receipt are recorded. Native job/timing code stayed unchanged through review fixes.

Updated service/output tests, app composition/lifecycle, coordinated deletion/recovery, architecture, qualification evidence and ADR-367. Deferred minor: preserve a secondary listener shutdown diagnostic alongside an independent primary failure; final branch review will triage. Delivery is a separate draft child of PR3767; final controls and end-to-end issue acceptance remain separate work.
<!-- SECTION:NOTES:END -->

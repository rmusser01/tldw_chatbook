---
id: TASK-41177
title: Coordinate controlled audio output ownership
status: In Progress
assignee:
  - '@codex'
created_date: '2026-11-15'
updated_date: '2026-11-24T03:54:00Z'
labels:
  - watchlists
  - tts
dependencies:
  - TASK-41176
references:
  - https://github.com/rmusser01/tldw_chatbook/issues/2864
documentation:
  - backlog/decisions/366-watchlist-article-listening-and-controlled-playback.md
  - Docs/superpowers/specs/2026-11-15-watchlist-article-listening-design.md
  - Docs/superpowers/plans/2026-11-15-watchlist-article-listening.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make article listening coexist predictably with existing speech and media output, with reliable stop, position and completion control.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Windows controlled WAV playback restores a validated PCM16 stereo 48 kHz retained file with the qualified tail policy at its saved media position; unsupported platforms refuse controlled listening before synthesis.
- [ ] #2 All existing file, streaming PCM and media start paths serialize output ownership and wait for confirmed retirement before replacement output.
- [ ] #3 Late starts and old-surface cleanup cannot affect the new owner; failed termination retains a visible blocked owner.
- [ ] #4 Existing ordinary file playback and external-handoff behavior remain supported, with targeted regressions and native transfer evidence.
- [ ] #5 Exact logical-EOF Resume launches no player; Stop during padding stays an interruption and cannot publish natural completion or release a newer output owner.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/366-watchlist-article-listening-and-controlled-playback.md
Reason: existing accepted output ownership and Windows-first capability contract; no new architectural boundary.
1. Initial bounded increment: implement only AudioOutputOwnership and exact host-issued leases, disconnected from production callers. Use event-driven RED/GREEN tests for replacement ordering, revocation, same-label identities, stale releases, blocked retirement, cancellation custody and concurrent intents. Record the tested branch base.
2. Run targeted ownership tests, relevant import/architecture checks and Ruff; independent spec/quality review the increment. Keep the task In Progress and all whole-task AC open.
3. Next increment: implement the qualified Windows controlled WAV adapter with explicit retained artifact metadata, typed completion, fresh progress and exact EOF/padding boundaries. Rebase onto current dev and rerun targeted checks before production integration.
4. Integrate every actual output and exact retirement boundary, including ordinary/preview file and PCM, provider realtime, speculative duplex, Speech Lab/wizard samples, Watchlists/Library/artifact replay, media including seek, and quick/full exit. Failed physical cleanup retains custody. Preserve external handoff acknowledgement and existing platform support for ordinary playback.
5. Validate native Windows takeover and cross-surface races, configured adapter playback, shutdown and affected targeted regressions. Update architecture/docs and mark Done only after all task AC and required native evidence pass. No provider, queue, storage or UI feature work belongs to this task.
<!-- SECTION:PLAN:END -->

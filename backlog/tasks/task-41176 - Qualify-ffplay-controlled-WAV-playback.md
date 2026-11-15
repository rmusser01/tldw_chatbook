---
id: TASK-41176
title: Qualify ffplay controlled WAV playback
status: In Progress
assignee:
  - '@codex'
created_date: '2026-11-15'
updated_date: '2026-11-15T17:09:00Z'
labels:
  - watchlists
  - tts
dependencies: []
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
Establish whether an installed player can meet article listening resume, completion and stop promises before feature work depends on it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A reproducible synthetic-WAV qualification tool and deterministic parser/measurement tests record executable build and reject invalid or stale progress without retaining unbounded output.
- [ ] #2 Native seek, short-clip, final-content, startup-stop and playing-stop measurements are reported separately from process exit and from physical or listener confirmation.
- [ ] #3 Windows, macOS and Linux each have explicit pass, fail or unqualified results; the full gate passes only after supported native builds meet the 250 ms seek target and completion/stop contract.
- [ ] #4 Targeted tests, lint, independent review and a sanitized evidence report document exact run scope and any unresolved gate; no application runtime dependency or provider request is introduced.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/366-watchlist-article-listening-and-controlled-playback.md
Reason: qualify the provisional controlled-playback contract before dependent listening storage/UI.
1. Implement and red-green test bounded ffplay stats parsing and synthetic waveform measurement helpers under tools/evidence/watchlist_article_audio.
2. Run focused unit and static checks; isolate any capture-only dependency under the plan workspace without changing the app environment.
3. Exercise real installed ffplay against synthetic PCM WAVs on the Windows default-render endpoint, measuring seeks, final content, telemetry and stop/cleanup independently.
4. Record sanitized numerical evidence and limitations; retain unqualified macOS/Linux and physical/listener cells, then request independent review. Do not mark the gate complete or implement downstream feature work without its required evidence.
<!-- SECTION:PLAN:END -->

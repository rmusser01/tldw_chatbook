---
id: TASK-41176
title: Qualify ffplay controlled WAV playback
status: In Progress
assignee:
  - '@codex'
created_date: '2026-11-15'
updated_date: '2026-11-15T20:20:00Z'
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
  - Docs/Testing/watchlist-article-audio-qualification.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Establish whether an installed player can meet article listening resume, completion and stop promises before feature work depends on it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A reproducible synthetic-WAV qualification tool and deterministic parser/measurement tests record executable build and reject invalid or stale progress without retaining unbounded output.
- [x] #2 Native seek, short-clip, final-content, startup-stop and playing-stop measurements are reported separately from process exit and from physical or listener confirmation.
- [ ] #3 Windows, macOS and Linux each have explicit pass, fail or unqualified results; the full gate passes only after supported native builds meet the 250 ms seek target and completion/stop contract.
- [x] #4 Targeted tests, lint, independent review and a sanitized evidence report document exact run scope and any unresolved gate; no application runtime dependency or provider request is introduced.
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
5. User-authorized Windows follow-up: qualify a fixed 250 ms trailing-zero PCM tail with three repeated short-clip, near-EOF and resume matrices. Keep original content duration distinct from padded physical duration; test fresh Stop bookmarks against the identical retained WAV/digest, and preserve unpadded controls. Update ADR366 and evidence from measured results only. No dependent application feature work or non-Windows qualification is authorized by this follow-up.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented reproducible ffplay qualification tooling with bounded stats parsing, source-waveform matching, exact default-output stereo loopback, safe direct-executable resolution and owned child cleanup. No application feature code, provider requests or runtime dependency changes.

15 focused unit tests pass; Ruff check and format pass. Independent review's explicit Chocolatey shim bypass was fixed in fc311bf150 and scoped re-review found no new issues. Tooling commits: 797bd1284b and fc311bf150. Plan/ADR/tasks: 66228f1511.

Native Windows results: ten matrix runs, two paired tail controls and three QPC follow-ups, plus two separate telemetry preflights. Matched seeks were within 2.42 ms; six initial natural runs missed 30.4-73.1 ms of final content. Stop and reaping checks passed. A 250 ms trailing-silence control recovered the content tail but is not an adopted workaround; later unpadded output varied. macOS/Linux and acoustic/listener evidence remain unqualified.

ADR required: yes. ADR path: backlog/decisions/366-watchlist-article-listening-and-controlled-playback.md. Evidence: Docs/Testing/watchlist-article-audio-qualification.md. Task remains In Progress and downstream tasks have not started. Next: qualify a reliable output-drain strategy and complete the native platform matrix before feature implementation.

Windows trailing-silence follow-up (user-authorized): extended only the qualification runner with fixed 250 ms zero PCM, content/file geometry, a bounded three-round matrix and fresh content-frame bookmarks for same-WAV Stop/resume. Preserved original unpadded results and every censored/rejected padded attempt. No application, provider or runtime dependency changes.

Final tooling 3f6c1995b6 passes 38 targeted tests and Ruff. Independent review's capture-quality/backpressure diagnostics and exceptional-cleanup receipt-loss findings were addressed; scoped review found no new Important/Critical issue. Progressive receipts and aborting launches after uncertain retirement preserve failed evidence safely.

Final Windows matrix: 21 planned rows, 20 actual players, 13 matched natural passes and two matched Stops. Same-artifact saved resumes passed at frame 36,480 (0.76 s) and frame 35,520 (0.74 s). Maximum passing errors: seek 2.438 ms, tail 2.417 ms, bookmark 59.5 ms. Three natural captures and one Stop were unmatched; one natural case had a noisy baseline and two capture discontinuities; one dependent Resume was correctly skipped. All 20 players/capture workers were reaped. Fixed 250 ms padding is NOT qualified or adopted; do not average away the rejected rows.

ADR366 now records logical content versus physical padding identity, bounds and the provisional Windows decision. Evidence report and durable allowlisted 21-case JSON: Docs/Testing/watchlist-article-audio-qualification.md and Docs/Testing/evidence/watchlist-article-audio/windows-padding-250ms-buffer4800.json. Other platforms, acoustic output and the full gate remain unqualified. Task41176 remains In Progress with AC3 open; downstream tasks are untouched. Next: resolve capture/mix/timeline ambiguity before another unchanged acceptance matrix.
<!-- SECTION:NOTES:END -->

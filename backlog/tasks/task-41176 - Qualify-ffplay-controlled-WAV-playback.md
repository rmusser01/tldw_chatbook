---
id: TASK-41176
title: Qualify ffplay controlled WAV playback
status: Done
assignee:
  - '@codex'
created_date: '2026-11-15'
updated_date: '2026-11-24T03:54:00Z'
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
- [x] #3 The Windows-first gate passes only after the supported Windows profile meets the 250 ms seek target and completion/stop contract, including current-profile startup Stop and separate listener confirmation; Linux/macOS remain explicitly unqualified and unsupported until independent qualification.
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
6. Reproduce the Windows capture gaps, measure against native packet positions and timestamps, and rerun the same 21 planned cases with the same thresholds. Preserve earlier receipts and record the observed Windows result in ADR366 and the evidence report.
7. User-authorized listener/Linux follow-up: record a Windows listener confirmation, inspect the supplied native Linux host, isolate capture-only dependencies, and qualify exact hardware-output routing and measurement timing before repeating the same 21 cases plus startup Stop. Test and amend ADR366 before changing the measuring contract; retain original thresholds, no microphone or system audio reconfiguration, and no dependent feature implementation.
8. User-authorized Windows-first amendment: fix measured telemetry failure masking with RED/GREEN regressions, remove duplicate evidence keys without changing values, and update ADR366/spec/plan before dependent code. Complete three current-profile launch-relative startup-Stop controls with unchanged silence/timing/cleanup thresholds; preserve rejected attempts. Windows passes unlock TASK-41177 only; Linux/macOS remain unsupported and the portable gate stays open.
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

2026-11-23 packet-clock investigation and requested 21-case rerun

Reproduced two capture mechanisms. SoundCard 0.4.6 inserted 5,376 estimated zero frames in a device-free 112 ms wait. A native controlled empty poll added exactly 3,871 frames across matched waveform edges while device and source both advanced 21,296 frames. A separate control lost 10,560 native frames with a discontinuity. Older rejected rows lack enough provenance for individual attribution.

Commit b66e13d0ca adds WASAPI packet device positions, QPC and flags, independent drainage, bounded healthy quiet monitoring, native packet validation and EOF-aware Stop/resume eligibility. The same 21 cases and playback thresholds are retained. All 47 focused tests passed in 20.317 s. Ruff check and format passed. Independent code review found no actionable Important/Critical issue.

The packet rerun has 21 planned rows and 18 launched base players. Every prelaunch baseline contained active background audio at RMS 0.004150-0.018304 against the existing 0.0002 limit. Three dependent resumes were skipped. All resources retired and the native records showed zero later discontinuities and timestamp errors. A separate read-only session check confirmed an active render session. User pause request remains pending.

ADR366, the evidence report, and durable windows-wasapi-packet-21.json and windows-legacy-idle-control.json record causes and complete results. The 250 ms profile stays provisional. Task remains In Progress with AC3 open. A quiet endpoint is required for an eligible acceptance rerun.

2026-11-23 quiet-endpoint rerun: Windows endpoint qualification passed

The user paused audio, but the first retry still found active prelaunch sound at RMS 0.001626-0.009073. It preserved 18 rejected players and three skipped resumes, all resources reaped. Independent review confirmed that baseline samples end before player launch. Read-only session metering still showed nonzero output. The user then muted the active session.

The unchanged b66e13d0ca runner completed all 21 cases in three rounds with every threshold retained. Windows endpoint-loopback passed. All 18 natural playbacks, including three same-artifact saved resumes, and three playing Stops passed their applicable checks. Maximum absolute errors: seek 3.250 ms, natural tail 2.417 ms, bookmark alignment 69.542 ms and telemetry 25.492 ms. Maximum process Stop latency 10.003 ms and last detected signal after Stop 32.633 ms. Saved frames were 36,480, 35,520 and 36,000 at 48 kHz. Quiet RMS and alignment drift were zero, with no later discontinuity/timestamp-error flags. All 21 players/capture workers reaped.

ADR366 and the evidence report now link all three packet matrices, including windows-wasapi-packet-21-session-muted.json. The recorded Windows build/executable/default-render route/PCM16 stereo 48 kHz profile passes. Other platforms, other routes and acoustic/listener output remain unqualified; the portable feature gate stays open. Task41176 remains In Progress, AC3 open, and downstream tasks have not started. No application playback or dependency change.
2026-11-23 Windows listener and Linux native follow-up

Windows listener confirmation passed after the requested repeat. The second seven-case sequence passed its endpoint checks and the user confirmed audible playback, clean Stop and finished Resume. The missed first sequence and its one unqualified baseline remain preserved. This is human listening, with no instrumented acoustic recording.

Added capture-only Linux Pulse fragments in 0af764920d, with review fixes 96658a9e61 and ee3cc6ffee. Exact native sink/monitor identity, frame-index custody, finite PCM, bounded independent drainage and completion acknowledgement replace unsupported Windows identity/timestamp assumptions. Four review findings were addressed: missing end-boundary coverage, small-fragment live detection, same-name source replacement and independent player routing. Native startup then exposed an unassigned input misclassified as moved; the fix preserves unknown route coverage and explicitly unattributed waveform diagnostics. No consumer-index timing fence, invented timestamp or relaxed criterion was adopted. Independent scoped reviews found no open Critical/Important issue. One bounded capture-copy optimization remains deferred.

The final Linux matrix has 21 planned rows, 18 launched players and three prerequisite-skipped resumes. All 18 valid captures retained matched, unattributed source observations with observed quiet boundaries and zero drift. Fifteen natural observations had maximum source-start delta 2.438 ms and source-end delta 2.417 ms. These are diagnostics, not transport passes. Route history and absolute timing remain unqualified. Process/reaping observations passed, with maximum process Stop latency 7.245 ms. Separate final controls covered 500 ms natural playback, playing Stop and 20 ms startup Stop; initial controls and the native startup diagnosis are retained. The user has no connected Linux output and no macOS host, so neither receives listener or native passes from Windows.

All 68 focused tests passed on native Linux in 24.147 s. Windows passed the same 68 with process-local single-thread BLAS in 41.240 s after five unchanged 15 s CLI startup timeouts on the default run; cause is undetermined and no baseline attribution is claimed. Exact failures and validation are in Docs/Testing/evidence/watchlist-article-audio/listener-linux-verification.json. Ruff lint/format and numerical receipt checks passed. All native resources were reaped; the private Linux environment was removed after downloaded/member hashes verified.

ADR required: yes; existing ADR366 amended before measurement-contract changes. ADR366, the implementation plan, Docs/Testing/watchlist-article-audio-qualification.md and durable Linux/listener receipts record scope and remaining uncertainty. Task remains In Progress with AC3 open; dependent application tasks have not started. Portable qualification still requires Linux route/timing evidence, Linux audible confirmation and native macOS evidence.

2026-11-23 Windows-first amendment: user approved Windows-only controlled article listening; Linux/macOS remain unqualified and unsupported, with the portable gate open. ADR366, spec and plan updated before behavior changes. Commit 1128d94be2 preserves measured telemetry failures and separates launch-relative Stop from WAV padding. RED/GREEN regressions, 72 focused tests and Ruff passed; independent source review approved. Duplicate evidence keys were removed with parsed values unchanged. The original Windows 21-case receipt still passes the corrected grader. Three startup supplemental attempts are preserved: noisy-baseline rejection; two passes and one unqualified unmatched-PCM timing observation; diagnostic repeat with three passes and zero packet/prelaunch/post-Stop RMS. Actual Stop requests 20.079-22.168 ms after launch, maximum process retirement 4.507 ms, all nine player/capture pairs reaped. No threshold or matcher/grader change in the diagnostic repeat; earlier unmatched PCM source remains unknown. Windows-first release gate receipt binds matrix/listener/startup evidence by digest. Exact EOF, padding transitions and actual adapter/app-wide ownership remain TASK-41177/41180 work. ADR required: yes; existing backlog/decisions/366-watchlist-article-listening-and-controlled-playback.md. Evidence closeout review and final CLI transition pending.

Final Windows-first evidence audit approved: all gate digests, 9 raw-to-public control projections, exact executable/artifact identity, timing, quiet criteria and owned retirement independently verified. No open Critical/Important finding. Windows-first qualification is complete; the portable gate remains unqualified. This completion does not qualify the future application adapter or production ownership integration. TASK-41177 may now start under the amended scope.
<!-- SECTION:NOTES:END -->

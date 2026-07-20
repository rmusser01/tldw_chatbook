---
id: TASK-347
title: Reflect active generation in Console status surfaces instead of Ready - No active work
status: To Do
assignee: []
created_date: '2026-07-20 14:21'
labels: [console, ux]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
During an active run (both the silent thinking phase and while tokens were visibly streaming with a '[streaming]' tag in the transcript), the screen-level status chip still reads 'Ready', and the Inspector panel - the dedicated status surface - reads 'Status: Ready', 'Live work: No active work', 'Provider: ready' (j4-33 shows the assistant message mid-stream in the same frame). cell_attrs check confirmed the Ready chip is a plain non-bold gray chip that never changes during a run. The only truthful indicators are the amber Stop button and the per-message [streaming] suffix.

**Repro:** Send any long prompt -> while the reply streams ([streaming] visible in transcript), read the chip under the Console title and open the Inspector panel -> both claim Ready/No active work.

**Verifier note:** Confirmed in j4-33 (mid-[streaming] frame shows header chip 'Ready', Inspector 'Status: Ready', 'Live work: No active work', 'Provider: ready'). Code shows these are readiness/launch-context surfaces by construction (console_run_inspector.py:342-347 status line only knows Blocked/Needs-approval/Source-blocked/Ready; console_display_state.py:429 'Live work' comes from pending Library-RAG launch context, never chat runs; ConsoleControlState has no run-active field) — no ledger item settles that they stay 'Ready' during runs. Adjacent to (not covered by) the 2026-07-17 shell-chrome critique's 'Console triple readiness display' finding, which is about redundancy not run-state staleness; inspector-static-streaming-excerpt (task-280) covers only the selected-message excerpt row. Downgrade to P2: truthful run indicators exist (amber Stop, [streaming] suffix, tab dot) — the defect is contradiction, not absence.

**Source:** Console UX expert review 2026-07-20 (finding j4-status-surfaces-say-ready-during-run; P2, verdict NEW, high confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J4 streaming journey. Evidence: `j4-33-streaming3.png`, `j4-04d-gap-40s.png`, `j4-36-just-after-stop.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Status surfaces should agree with reality: the chip and Inspector 'Run/Live work' section should switch to a running state (e.g. 'Generating... 12s') for the duration of the run, and return to Ready on completion/stop
<!-- AC:END -->

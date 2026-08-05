---
id: TASK-338
title: Restore Streaming on-off visibility in the Console rail Model section
status: To Do
assignee: []
created_date: '2026-07-20 14:21'
labels: [console, ux, regression]
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The rail Model section lists Provider/Model/Temperature/Max tokens/System but not Streaming; the chips don't show it either. After toggling Streaming to Off and saving (verified applied: Inspector shows 'Streaming: off', 'Sampling: T 0.70'), nothing on the default screen reflects the change — it's only visible inside the modal or the collapsed-by-default Inspector.

**Repro:** 1. Rail > Configure; click the Streaming 'On' button (becomes 'Off'); Save. 2. Scan rail and chips: no streaming mention. 3. Expand Inspector (right edge): 'Streaming: off'.

**Verifier note:** True regression against the shipped-behavior ledger item model-section-compact, introduced AFTER the ledger by commit 0c26a8408 'feat(console): split model settings into labeled rows' (2026-07-18, on the PR #716 feature/console-screen-feedback branch): the old line2 built by build_console_model_section_lines joined (sampling, context, transport) where transport was 'Streaming: on/off'; the replacement labeled rows (chat_screen.py:7176-7252) carry only Provider/Model/Temperature/Max tokens — the streaming (and token-budget) readout was silently dropped from the default surface. Confidence medium only on intent: the rail redesign presumably passed the console-approval-gate, so the drop may have been implicitly accepted, but no settled-decision item supersedes the streaming-visible contract. P3 per reviewer: visibility polish, value still shown in modal/Inspector.

**Source:** Console UX expert review 2026-07-20 (finding j5-streaming-state-invisible; P3, verdict REGRESSION, medium confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J5 settings journey. Evidence: `j5-69-rail-after-streaming-save.png`, `j5-70-inspector-streaming.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every model setting the user can override per-session (including Streaming) is visible at a glance on the default Console surface — in the rail Model summary or via a non-default-state indicator
- [ ] #2 A regression test pins the restored behavior
<!-- AC:END -->

---
id: TASK-74
title: 'CDP QA: Console composer paste workflow from current dev'
status: Done
labels:
- qa
- console
- cdp
- ux
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify the current-dev Console composer paste workflow through actual textual-web/CDP use so the visible input, paste-collapse token, unfurl flow, and normal typing behavior are proven in the rendered app after the paste-threshold merge.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 CDP/browser-driven QA is run against a clean current-dev app instance, not a mockup, SVG, or static layout render.
- [x] #2 Actual screenshots are captured for Console baseline, visible typed input, collapsed large paste token, first-click Unfurl prompt, second-click expanded paste text, and normal typing over the threshold remaining literal.
- [x] #3 QA notes record exact commands, browser/CDP URL, app commit SHA, environment, and any failed or uncertain observations.
- [x] #4 The walkthrough verifies that only paste/insert chunks over the configured threshold collapse, while normal typing remains visible and literal.
- [x] #5 Any observed Console composer regression is either fixed in the same follow-up PR with focused regression coverage or logged as a separate Backlog task with severity and reproduction steps.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record the current dev SHA, branch, clean worktree state, runtime paths, and isolated QA profile paths before launching the app.
2. Launch the clean current-dev app through textual-web with PYTHONPATH pinned to this worktree and a temporary HOME/XDG profile with Console as the startup destination.
3. Drive the running Console through Playwright/CDP to capture actual rendered PNGs for baseline, visible typed input, collapsed large paste token, first-click Unfurl prompt, second-click expanded paste text, and normal typing over threshold remaining literal.
4. Document commands, browser URL, environment, screenshots, observations, and residual risks in durable QA notes under Docs/superpowers/qa/product-maturity/screen-qa/console/.
5. If any composer regression is observed, add a focused failing regression first, implement the smallest safe fix, rerun focused tests and CDP screenshots, then update task notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Completed an actual textual-web/CDP QA pass for the Console composer paste workflow from current `origin/dev` app code at `b1bcff71eeef67302b3dd3245c1a8685a09bddc5`. Launched the app from a clean worktree with `PYTHONPATH` pinned, an isolated HOME/XDG profile, Console as the startup destination, splash disabled, and paste collapse enabled at the default 50-character threshold. Captured the final evidence with standalone Playwright/Chromium as real PNG files after the browser-plugin capture path produced JPEG bytes under `.png` names. Verified baseline Console, visible short typing, a collapsed large paste token, first-click `Unfurl?`, second-click expanded pasted text, and true per-key normal typing over the threshold remaining literal. No product regression was found, so no code change or new regression was required; focused existing regression coverage was rerun and passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Actual CDP/browser QA evidence and notes are stored in `Docs/superpowers/qa/product-maturity/screen-qa/console/task74-2026-05-30-console-composer-paste-cdp-qa.md`, with final screenshots under `Docs/superpowers/qa/product-maturity/screen-qa/console/captures/`. Focused verification passed: 6 selected Console paste/unfurl/typing tests passed with one existing requests dependency warning.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] Current-dev SHA and clean worktree state are recorded before QA begins.
- [x] CDP/browser screenshots are attached or linked from durable QA notes.
- [x] The rendered app is verified for visual usability and functional behavior, not only selector presence or clickability.
- [x] Focused automated regression coverage is added for any bug fixed during the ticket.
- [x] Task notes identify residual risks and the next follow-up if the workflow is not fully usable.
<!-- DOD:END -->

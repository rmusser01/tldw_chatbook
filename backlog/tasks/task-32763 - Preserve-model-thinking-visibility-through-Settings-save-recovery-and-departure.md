---
id: TASK-32763
title: >-
  Preserve model-thinking visibility through Settings save recovery and
  departure
status: Done
assignee:
  - '@codex'
created_date: '2026-09-18 00:13'
updated_date: '2026-09-18 00:50'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the device-local model-thinking visibility choice consistent between the live Console, reopened Settings and the saved configuration when writes overlap, fail or outlive the Settings screen.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Latest visibility choice survives overlapping writes and actual Settings screen removal and recreation.
- [x] #2 Config reload resets the saved baseline so the next visibility change is persisted correctly.
- [x] #3 Failed writes restore the last saved visibility with truthful feedback and permit retry; post-replace cache failures preserve saved values and warning receipts.
- [x] #4 The canonical checkbox remains keyboard-operable and fully readable at compact and wide widths, and updates mounted thinking presentation without changing capture or replay policy.
- [x] #5 Targeted regression checks and isolated native dark/light journeys document real persistence, navigation, lifecycle and remaining scope.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce screen-departure and reload gaps through mounted production-style Settings and the real private config writer.
2. Reuse the existing app-owned Console toggle queue for model-thinking visibility, preserving immediate refresh and structured persistence outcomes; remove the obsolete screen-local queue.
3. Preserve existing visibility assertions at the config mutation boundary and cover failure, retry, superseded receipts, removed screens, reloads and compact keyboard paint.
4. Run affected targeted tests and static/artifact checks; verify native dark/light at wide and compact sizes with private local config saves and clean shutdown.
5. Complete independent review, documentation and task evidence, then save the increment to draft PR 2704.

ADR required: no
ADR path: backlog/decisions/090-console-thinking-block-ownership-and-replay.md
Reason: Bounded repair of the accepted device-local visibility behavior; capture, ownership, replay policy and storage schema do not change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Model-thinking visibility now reuses the app-owned Console toggle queue, preserving the newest value across overlapping writes, Settings removal/recreation and config reload. Removed the obsolete screen-local writer. App-owned inline receipts preserve failure, retry and cache warnings; concise recovery copy fits the padded compact row. Existing and new tests preserve presentation-only behavior and stored thinking. 81 distinct targeted cases pass (49 affected Settings/transcript, 32 governance); all seven derived-artifact guards pass; changed tests/runner pass Ruff, and the legacy Settings owner introduces no lint or format debt. Independent review is clear. Four final private native dark/light and wide/compact cells pass, with eight inspected captures, exact source hashes, clean shutdown, 11 healthy databases and unchanged default-file fingerprints. QA: Docs/superpowers/qa/2026-09-17-settings-thinking-visibility/README.md; guide and completion ledger updated. ADR required: no; implements existing backlog/decisions/090-console-thinking-block-ownership-and-replay.md with no capture, replay or schema change. No full suite or provider request was run. The broad feature review stays open and PR 2704 stays draft pending visual review and explicit merge approval.
<!-- SECTION:NOTES:END -->

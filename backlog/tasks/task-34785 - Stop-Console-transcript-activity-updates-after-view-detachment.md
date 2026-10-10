---
id: TASK-34785
title: Stop Console transcript activity updates after view detachment
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-10 20:50'
updated_date: '2026-10-10 21:05'
labels:
  - console
  - bug
  - ci
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_chatbook/pull/2882'
  - >-
    https://github.com/rmusser01/tldw_chatbook/actions/runs/38084008605/job/114306575704
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The required UI lane can crash while a pending question/approval journey removes and remounts its Console view. Ensure transcript reconciliation stops touching retired widgets so navigation and pending-decision recovery remain usable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A real Console view removal during an awaited activity update produces no MountError or further mounts into retired widgets.
- [ ] #2 Mounted activity updates and the existing pending question/approval sibling/remount journey retain their behavior.
- [ ] #3 A deterministic regression fails before repair; targeted checks, static analysis, preflight and independent review verify the fix.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/190-console-incremental-display-projections.md
Reason: repair stale DOM mutation within the existing mounted presentation lifecycle, without changing storage, authority or UI design.

1. Preserve the hosted failure and reproduce actual widget removal during an awaited activity update; compare the unchanged base implementation.
2. Reuse existing attachment/closing checks to stop the shared transcript mutation path after detachment; retain normal projection updates.
3. Verify the deterministic regression and affected activity/pending-projection modules, static checks and preflight; obtain independent review before publication.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Transcript mutation now checks actual attachment, closing/pruning state and expected parent ownership before and after nested activity/action/detail/adjunct awaits. Private Assistant/activity synchronization reports completion so retired work does not publish row signatures. The existing refresh lock also serializes Textual recompose; a refresh arriving during first composition queues one transcript-owned on_mount refresh, retaining changed activity after the screen caches its projection key. No style, storage, permission, runtime-owner or public caller contract changes. Existing ADR190 applies; no new ADR.

The exact hosted MountError is reproduced outside the repository on both immutable 732d1de8c29ede5c03d0a0263d4f9cacf29e0321 and base c2d6dcc3016708bec3473450016177b2ff602fb3. Their transcript source is byte-identical (SHA256 0f7e5c84964c5b6a2d683e71ff3e65d5859ff95ddc661b2a52104f66479b4779). Real recompose leaves the fresh turn attached while its activity stack is still awaiting composition. Separate real removal across lock, stale child removal and inter-mount boundaries produces inherited IndexError and stale signature publication. Raw probes: /private/tmp/pr2882-transcript-detach-probe.py; pr2882-transcript-composition-{head,base}.json; pr2882-transcript-detach-{head,base}.json.

Initial regression selection is NON-GREEN13FAIL: eleven genuine detach/mount/signature failures plus two app-context harness failures. Correcting only the composition harness preserves the actual Textual context and yields RED2 exact MountError. Initial standalone recovery_scope_uncertain and the first unchanged-signature probe oracle are retained separately. Final current-source selection PASS15 in3.68s: thirteen new deterministic lifecycle cases plus two existing real live-tool controls. Earlier GREEN15 and tightened lock-oracle GREEN2 are separate receipts, never summed. XML/logs: /private/tmp/pr2882-transcript-lifecycle-red.{xml,log}; pr2882-transcript-composition-red.{xml,log}; pr2882-transcript-lifecycle-final.{xml,log}. Existing cleanup warnings remain retained.

Scoped Ruff baseline27/current27/zeroNEW; fatal diagnostics zero and formatter all three files pass. Broader trace modules, the actual pending-projection journey, generated-artifact preflight and independent immutable review remain pending under root ownership. Task stays In Progress with acceptance criteria unchecked until that qualification completes.
<!-- SECTION:NOTES:END -->

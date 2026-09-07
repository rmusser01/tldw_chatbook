---
id: TASK-1979
title: 'Change review: Settings surface, per-workspace toggle, git-absent gating'
status: Done
assignee: []
created_date: '2026-08-02 21:00'
labels:
  - settings
  - change-review
  - workspaces
dependencies:
  - TASK-1971
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User control and honest availability: flat [change_review] config section (enabled, max_file_bytes, max_files, max_total_bytes, retention_days, diff_display_max_lines) with env overrides; a per-workspace toggle in Settings beside folder roots; feature-absent states with honest copy — no git binary ('Change review needs git — install git to enable'), no folder roots configured. Toggles take effect without restart (poke the live config tree — the app_config-captured-once trap).

Spec: `Docs/superpowers/specs/2026-08-02-agent-change-review-design.md`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Disabling per-workspace stops snapshots for that workspace's roots on the NEXT run without restart
- [x] #2 git absent -> Settings and card copy state the reason; runs behave exactly as with the feature off
- [x] #3 Every knob is read live from config with the documented env-var override
- [x] #4 Settings copy passes the monochrome/persistence-badge conventions of the Settings screen
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Gating choke point: `folder_binding_roots` (exists solely for tracking)
   returns () when the flat `[change_review] enabled` knob (env-overridable,
   read live per call) is off OR the workspace's toggle is off — AC#1 falls
   out with zero controller changes
2. Per-workspace toggle: `workspace_change_review` side table mirroring the
   workspace_rag_scopes pattern (absent row = enabled); registry get/set
3. Settings workspace card: Change review row — toggle button when git is
   present, the honest 'Change review needs git — install git to enable'
   copy when absent (monochrome, matching the pane's conventions)
4. TDD; sabotage first-try passes; regression before push
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Gating** lives at the ONE choke point: `folder_binding_roots` (which
exists solely as the tracker's root source) returns () when the flat
`[change_review] enabled` knob is off (env `TLDW_CHANGE_REVIEW_ENABLED`
beats config, read live per call — AC#3) or the workspace's toggle is off.
AC#1 (next-run effect, no restart) falls out with zero controller changes;
the registration hook's initial snapshot gates identically for free.

**Per-workspace toggle**: `workspace_change_review` side table mirroring
the `workspace_rag_scopes` pattern (absent row = enabled, opt-out; a
storage READ error also reads enabled — availability must not flip off on
a failed read); `LocalWorkspaceRegistryService.change_review_enabled` /
`set_change_review_enabled` (upsert, service clock seam).

**Settings surface**: the workspace card gains a "Change review (post-run
diffs)" section — state copy + Enable/Disable toggle button in the pane's
existing monochrome idiom (AC#4); with no git binary the row shows the
exact honest copy 'Change review needs git — install git to enable.' and
no dead toggle (AC#2; runs already behave feature-off via the bridge's
tracker=None gating from TASK-1971).

Tests: 3 gating round-trips (global knob, per-workspace isolation,
re-enable without restart) + 2 UI tests (toggle round-trip persisting to
the registry; git-absent copy — patched at the `ShadowRepoService.available`
seam, NOT `shutil.which`, which is a shared module object that broke the
file-notes git service mid-test). Settings section sabotage-verified
(blinded render failed both UI tests). 273 green.
<!-- SECTION:NOTES:END -->

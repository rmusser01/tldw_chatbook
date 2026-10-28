---
id: TASK-37174
title: Preserve full Windows workspace directory identities
status: Done
assignee:
  - '@codex'
created_date: '2026-10-28 17:07'
updated_date: '2026-10-28 17:22'
labels: []
dependencies: []
documentation:
  - Docs/superpowers/plans/2026-10-28-windows-workspace-identities.md
  - backlog/decisions/101-one-shot-pinned-workspace-tool-execution.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The existing real workspace folder-access test refuses an unchanged Windows directory as changed after dispatch. Python 3.12 can expose a 64-bit volume serial and a 128-bit file identifier, while the Windows workspace pin reader currently uses the older narrower metadata. Establish whether that representation mismatch is responsible and restore valid workspace access without weakening identity or reparse checks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A retained Windows directory handle has the same full volume and file identity as Python 3.12 stat metadata for that directory.
- [x] #2 Unchanged admitted Windows folders can be listed through the real workspace tool worker.
- [x] #3 Replaced directories, reparse points, unavailable identity metadata, and cleanup failures remain fail-closed.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no; ADR path: backlog/decisions/101-one-shot-pinned-workspace-tool-execution.md, backlog/decisions/069-console-project-instruction-local-state-and-preflight.md; Reason: compatibility correction preserving the existing retained-root identity contract. 1. Reproduce the metadata mismatch before implementation. 2. Use the matching Windows handle identity API and retain attributes/cleanup checks. 3. Run targeted root-pin and real folder-worker regressions and record any existing platform limits.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Independently reproduced and validated a native Windows workspace access defect on the older worktree base: the retained-handle reader used a truncated 32-bit volume serial while Python 3.12 reports the full 64-bit volume serial and can report 128-bit file identifiers. GetFileInformationByHandleEx(FileIdInfo) now reads both full values from the same retained no-follow handle. Existing reparse checks, before/after current-directory identity checks, missing-metadata refusals and cleanup remain intact. Regenerated the embedded remote worker through its canonical builder; no manual artifact edits.

Evidence: the new native handle/stat regression failed on the volume identity before the fix and passed afterward. The original real Console folder-listing regression then passed once its isolated worker was correctly bound to this worktree rather than the main checkout's editable install. The final integrated 67-test batch covers all folder-note/worker cases and root replacement/read/write confinement; ten final guard tests also passed. Source lint and formatting, generated-worker drift, a standard-library-only bundle import, and whitespace checks pass. Independent review found no actionable ABI, identity, reparse or cleanup defect.

No new ADR was needed: this corrects metadata representation under the existing ADR-101 retained-root contract and ADR-069 binding authority. Updated console-file-authority architecture, plan and lessons-testing-evidence with the actual incident and the isolated-interpreter import trap. Modified Tools/workspace_root_pin.py, generated remote_worker_bundle.py and the native root-pin regression. Existing Windows command-hook unsupported_platform failures were separately reproduced on HEAD and remain outside this identity compatibility fix.

Upstream reconciliation before PR: current dev already contains the equivalent implementation and broader native tests in commit 6fe22b80ba (2026-10-22). Retain that canonical implementation and its tests; this PR does not claim a new Windows repair. The original user report was from Fedora Linux, so this Windows-only finding is not its cause.
Post-rebase verification on current dev: 86 targeted tests passed (553.63s), including native/fenced reload recovery, typed preparation guards, skill gates, the upstream native root-pin controls, and isolated real folder-access/policy cases. Lint, changed-file formatting, generated-worker drift, task/ADR uniqueness, and patch whitespace checks pass. Fedora and live-model reproduction remain unconfirmed.
<!-- SECTION:NOTES:END -->

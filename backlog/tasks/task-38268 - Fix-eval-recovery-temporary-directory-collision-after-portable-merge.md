---
id: TASK-38268
title: Fix eval recovery temporary directory collision after portable merge
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-31T23:49:00Z'
updated_date: '2026-11-02'
labels: []
dependencies: []
---

## Description

Repair CI failures exposed after merging the Windows portable bundle and rebasing PR #3342. Keep recovery authority and runtime checks intact.

## Acceptance Criteria

- [x] #1 Recovery fixtures isolate temporary directories and native AppData without colliding with the launch-environment test.
- [ ] #2 Required CI passes on the approved Ubuntu 24.04 runner with its complete selectors and existing runtime budgets.
- [x] #3 Demonstrated runtime defects retain focused regression coverage; stale test doubles follow current interfaces.
- [x] #4 Generated build/test artifacts and unrelated refactors are removed before publication.
- [ ] #5 Current-head checks are reviewed before taking the PR out of draft.

## Implementation Plan

ADR required: yes (existing ADR-103 amendment).
ADR path: backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md
Reason: owner-approved explicit Ubuntu runner label. Existing recovery authority and UI/service contracts remain in force; ADR-097 records the explicitly approved 116-module first-send baseline; other boot/import limits remain enforced.

1. Preserve recovery fixture isolation and the approved runner pin.
2. Retain fixes tied to observed CI failures and their targeted regressions.
3. Retain cohesive refactors needed by the complexity/LoC gates; remove unrelated import changes, experimental diagnostics, and generated artifacts.
4. Run targeted checks, publish the reduced branch with an explicit lease, and review normal PR CI.

## Implementation Notes

- Recovery fixtures use separate temporary paths and private APPDATA/LOCALAPPDATA roots; native-path TOML is escaped correctly.
- Retained runtime repairs address stale selected-settings banners, pressed-send acknowledgement, closed-session draft switching, context-read retirement, and readiness waiters stranded by a closed session.
- Test repairs track current widget mounting, service/test-host interfaces, provider responses, and exact native file bytes. Windows Library search completion uses the approved 60-second window; stream/POSIX and the 3-second retirement guard keep their original limits.
- The app-owned shutdown helper preserves lifecycle patch points while removing the upward Utils dependency. Cohesive Stop, recovery, citation, startup, Notes ownership and composer-wrapping extractions are restored from the saved branch; current dev behavior stays intact. Budget-only refusal-type, avatar, Canvas and compaction import changes are removed.
- Corrected the overbroad cleanup: complexity/LoC refactors remain, their source pins tighten to the resulting measurements, and unrelated diagnostic experiments are removed. Remaining baseline corrections describe already merged dev growth. Runtime, CSS, privacy, recovery, and retirement checks remain enforced.
- Windows admission latency remains separately tracked in TASK-38268.1. No application performance improvement is claimed.
- Qualification is pending on the reduced current head; earlier green recovery jobs do not qualify a new revision.

Latest focused checks after restoring the refactors: 13 warm-up protocol tests pass; the cold expected-import check and the approved 116-module budget pass; 30 source-boundary, patch-target, TTS-finally and wrapping checks pass. The broader mixed batch was interrupted and is not passing evidence. Windows mounted controls remain unresolved at their existing provider-admission (15 seconds) and worker-wait (10 seconds, WorkerCancelled) boundaries; those limits were not raised. Earlier hosted recovery jobs passed on all three OSes (Windows on retry), and Library shard 19 passed unchanged on retry. Current-head CI must qualify this revision before it leaves draft.

Cleanup removed 1,093,147,035 bytes across 26,508 generated files, including the retained test environment, build copies, downloaded archives, reports, caches, and task-owned temporary profiles. A local Git backup retains the previous committed fixes.

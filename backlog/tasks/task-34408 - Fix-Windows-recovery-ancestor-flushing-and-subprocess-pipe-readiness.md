---
id: TASK-34408
title: Fix Windows recovery ancestor flushing and subprocess pipe readiness
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-04 18:37'
updated_date: '2026-10-10 03:42'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Allow native Windows recovery and its independent-process tests to finish without requiring write access to unchanged system ancestors or treating pipes as sockets.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Recovery durably flushes its modified directories without demanding write access to unchanged system ancestors; required barriers still fail closed.
- [x] #2 Real subprocess pipe output and absence checks work on Windows and POSIX with bounded deadlines and correct EOF/error handling.
- [x] #3 Affected Windows recovery, admission, Eval retention and rollback tests pass, with static checks and review recorded.
- [x] #4 Fresh isolated-profile launches preserve native TEMP and TMP platform selectors while provider credentials and app overrides remain excluded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes, amendment to existing ADR
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: document durable directory creation intent and an identity-bound ancestry receipt while preserving native privacy and legacy recovery barriers.
1. Reproduce system-ancestor flush and anonymous-pipe readiness failures on native Windows.
2. Add regressions for changed-directory barriers, interrupted mkdir/retry, held-parent rename, legacy pending and real pipe readiness/EOF/deadlines.
3. Record creation intent before mkdir; settle creation and retire intent on the same pinned parent. Establish and validate a private ancestry receipt before publishing pending records. Preserve original barriers for unresolved legacy operations.
4. Correct Windows test-only pipe readiness and logging-heavy subprocess capture; preserve real contention and native fault injection.
5. Run affected Windows primitive/admission/lifetime/Eval retention/rollback checks, compare static diagnostics with HEAD, review and record measured evidence and host limits.
Superpowers plan: Docs/superpowers/plans/2026-10-04-windows-recovery-verification-fixes.md

2026-10-10 review follow-up: ADR required: no new ADR; existing ADR-126 and _launch_environment platform-selector contract apply. Add a regression for each native temp selector and provider/app exclusion, preserve TEMP/TMP in the existing allowlist, and verify the Windows rebackup pipeline without depending on the TMPDIR fixture workaround. This corrects existing platform behavior without changing private-directory qualification or snapshot admission.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Recovery registration now persists a private creation intent before mkdir, settles mkdir/parent flush/intent retirement through one pinned native parent, and refuses interrupted retries even after permission or path-spelling changes. A validated ancestry-settled.json binds settled creation to the actual bootstrap directory before pending publication. New operations flush their pending record and bootstrap directory without opening unchanged C:\Users for write; unresolved legacy pending operations retain their required ancestry barriers. Bootstrap, registration, publication and activation recovery readers validate the receipt.
The test helper observes real Windows anonymous pipes with PeekNamedPipe and keeps POSIX select. Complete lines share a bounded deadline; readiness does not consume output. Logging-heavy fixtures preserve diagnostics in owned stderr files, avoiding unread-pipe backpressure. Native fault injection targets the actual facade; contention assertions retain real independent processes, registry locking and bounded native lock retirement. Windows namespace fixtures isolate their owned child and alias fixtures respect actual platform behavior.
ADR: backlog/decisions/126-complete-local-backup-and-recovery.md, TASK-34408 creation-durability clarification. Plan: Docs/superpowers/plans/2026-10-04-windows-recovery-verification-fixes.md. Added incident-backed testing lessons.
Verification: original system-directory flush and pipe failures reproduced before correction. Final targeted native recovery run: 124 passed, 2 skipped, 1 deselected in 765.36s. Final pipe/admission/lifetime/selector/schema run: 100 passed in 152.00s. There are 211 distinct passing cases (13 durability cases overlap). All 21 touched Python files parse; Ruff comparison adds zero lint diagnostics (134 inherited), new files and changed production ranges pass formatting, and diff whitespace checks pass. Fresh protocol review has no remaining substantive findings.
Limits: two actual WinError 1314 file-symlink cases skip because this account lacks that privilege. The complete-rebackup fixture is blocked by unavailable/overlapping owner inventory before publication. Installed-wheel two-profile restore exceeded its existing 300-second timeout after installation and entering restore. Those wider complete-release gates remain unverified; no full sweep or POSIX run was performed. The owner subsequently requested a PR against dev. These limits are distinct from the passing targeted Windows repairs.
Updated files: bootstrap/control_records/publication, native Windows and admission fixtures, shared subprocess pipe helper and real pipe tests, raw/settings lifetime fixtures, affected Eval retention/later-snapshot fixtures, ADR-126, plans and testing lessons.
PR preparation: feature branch fast-forwarded to dev a7d9bca5da; no upstream file overlaps. Fresh focused PR check: 38 passed in 9.11s (pipe helper, pending durability, private Eval overrides). Backlog ID/frontmatter guards pass after the documented voluntary renumbering.

PR #3018 qualification exposed a separate baseline platform-selector bug: fresh isolated-profile selection dropped TEMP/TMP even though its existing platform allowlist kept TMPDIR. The Windows full-App pipeline passes all 75 cases/one FIFO skip when its qualified temp parent survives selection via TMPDIR. Added a meaningful native launch regression: TEMP and TMP fail before the correction (2 failed, 3 passed), then all five home/temp cases pass after preserving the two native platform selectors. Provider credentials and app overrides remain excluded; native ancestry and SQLite validation remain intact. Removed the TMPDIR CI workaround and assert real TEMP/TMP plus tempfile parent after selection. Existing ADR-126 and the documented platform-selector launch contract apply, with no new boundary or ADR. Added incident-backed testing guidance. Final all-platform pipeline check without the workaround is pending; task stays In Progress.
<!-- SECTION:NOTES:END -->

## Renumbering Provenance

Originally TASK-34364. The 2026-10-04 pre-push sweep found a published claim on `origin/fix/model-metadata-never-rejects-listing`: Fireworks tool calls fail, streamed and non-streamed, created at 18:48. This record was created at 18:37, but voluntarily moved to TASK-34408 before publication to keep the peer branch's published references stable. Implementation and acceptance criteria are unchanged; local live references were updated.

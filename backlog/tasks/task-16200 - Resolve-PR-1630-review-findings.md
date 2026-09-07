---
id: TASK-16200
title: Resolve PR 1630 review findings
status: Done
assignee: []
created_date: '2026-08-14 06:18'
updated_date: '2026-08-14 09:14'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Close every actionable PR 1630 review finding without weakening the bounded TTS artifact validator or provider endpoint safety policy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Optional PyAV loading uses the centralized dependency seam.
- [x] #2 Playable audio paths pass shared validation while exact-file race checks remain enforced.
- [x] #3 Public audio validation API has complete Args and Returns documentation.
- [x] #4 Schemeless exact IPv6 loopback endpoints are accepted while remote schemeless endpoints remain blocked.
- [x] #5 Dotted provider display names canonicalize correctly for model probes.
- [x] #6 Focused regression and acceptance tests pass.
- [x] #7 Structured composer snapshots retain prompt-improvement fingerprint compatibility and reach the auxiliary provider exactly once.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the five resolved review fixes and their regression coverage.
2. Add a focused RED regression for structured composer snapshot fingerprint parity.
3. Update the prompt-improvement validator to include all producer-owned snapshot metadata.
4. Re-run focused prompt, first-run UAT, and PR review suites.
5. Resolve any remaining CI failures, refresh review threads, and merge only from the latest dev tip.

ADR required: no

ADR path: N/A

Reason: This is a bounded compatibility correction within an existing immutable snapshot contract; no ownership, storage, provider, security, or cross-module boundary decision changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Routed optional PyAV loading through the shared dependency seam and gated artifact paths through shared validation without weakening no-follow, bounded-read, or file-identity checks.
- Accepted exact schemeless IPv6 loopback endpoints, reused canonical provider aliases for discovery, and normalized continuation endpoint comparisons.
- Adapted rebased first-run, settings, todo, and Studio contracts to current `dev`; explicitly auto-sized the unframed Speech inspector so its controls remain reachable.
- Preserved structured composer segment metadata when copying and fingerprinting prompt-improvement snapshots, preventing valid collapsed-paste drafts from being rejected as stale before the auxiliary provider call.
- Updated two Console assertions to follow the approved single speech control in the stable message header rather than the superseded selected-row control.
- Hardened the mounted collapse test seed against delayed Textual geometry under full-suite load by waiting for the expected transcript overflow before asserting collapse behavior.
- Verified 1,344 acceptance tests with four expected PyAV skips, four Pocket TTS roleplay integrations, 140 auto-speak/first-run tests, and 865 review regressions. Compilation, diff checks, and scoped formatting passed; isolated mypy findings were outside changed lines.
- Re-ran the final combined prompt-improvement, native Console, composer undo, and composer collapse group: 1,195 tests passed. The dedicated speech-header suite passed 12 tests and the collapse module passed 75 tests.
<!-- SECTION:NOTES:END -->

## ADR Check

ADR required: no

ADR path: N/A

Reason: These are bounded correctness, compatibility, and UI layout fixes within existing dependency, endpoint, validation, and settings contracts.

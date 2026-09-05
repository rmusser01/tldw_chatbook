---
id: TASK-31685
title: Scope the Samira package inventory guard to its reviewed asset family
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 18:26'
updated_date: '2026-09-05 18:31'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Repair the stale Samira packaging assertion after the separately reviewed offline tokenizer inventory was added, while preserving explicit and bounded Samira inclusion.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Samira package-data guard retains exact reviewed patterns and rejects broader patterns that include Samira assets
- [x] #2 Separately covered tokenizer inventory does not cause a false Samira failure
- [x] #3 Targeted asset and tokenizer packaging tests plus static checks pass without production packaging changes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the exact stale package-data equality and inspect the separately tested tokenizer inventory.
2. Select every pattern explicitly under Samira or matching any real Samira asset, then retain exact equality against its four reviewed patterns; unrelated tokenizer data stays outside this ownership check.
3. Run Samira asset and tokenizer packaging checks and probe broad pattern rejection, then static review.
ADR required: no
ADR path: N/A
Reason: Test-only ownership correction preserving the existing explicit package-data contract and production inventory.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Scoped the exact four-pattern Samira equality to patterns explicitly under its path or matching any real Samira asset. Matching broad ancestor globs as well as prefixes prevents silent acceptance of assets/**/*; separately reviewed tokenizer entries are not Samira ownership. No production packaging changes.
Baseline failed on nine unrelated tokenizer entries. Full visual-identity asset and vendored tokenizer checks: 69 passed in 4.50s. Read-only negative probe rejected assets/*, assets/**/*, assets/characters/*, assets/characters/samira/* and unreviewed Samira *.exe. Ruff check, whole-file format and git diff --check pass; parent review found no blocking issue.
ADR required: no, test-only contract ownership correction. The in-test comment records why prefix-only filtering would weaken the bounded guard.
<!-- SECTION:NOTES:END -->

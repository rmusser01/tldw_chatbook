---
id: TASK-32478
title: Set owner default prompt for Improve My Prompt with recency anchor
status: Done
assignee: []
created_date: '2026-09-12 01:03'
updated_date: '2026-09-12 01:33'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the Console prompt-improvement default optimizer instructions (Auto/Review modes) with the owner-selected four-section prompt-engineering template (Situation/Task/Objective/Knowledge), keep the trusted-prefix/untrusted-JSON-tail security boundary per ADR-029, and add a final recency anchor so providers do not return schema-valid but lazy near-copies. Live-verified against DeepSeek: without the anchor, both deepseek-chat and deepseek-reasoner returned a valid envelope with a no-op rewrite; with the anchor, both produced the full four-section transformation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 trusted_optimizer_instructions for auto/review modes returns the owner template plus safety guards plus JSON envelope plus task anchor,Recipe mode instructions unchanged,Prompt_Management and Console prompts modal test suites pass,Live Improve My Prompt run against a real provider produces the four-section structure
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Replace Auto/Review optimizer instructions with owner four-section template, keeping ADR-029 trusted/untrusted boundary
2. Append safety guards + JSON envelope + final recency anchor
3. Live-verify against a real provider in an isolated scratch profile
4. PR #2626 against dev, rebase, address Qodo review, merge
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented and merged via PR #2626. Approach: owner template as _DEFAULT_REWRITE_TRUSTED_INSTRUCTIONS for auto/review (recipe unchanged); safety guards (no-answer, preservation, no-invention) + strict JSON envelope + recency anchor appended in that order. Live TUI verification (isolated scratch profile, DeepSeek) caught that the envelope-last ordering produced schema-valid no-op rewrites on both deepseek-chat and deepseek-reasoner; the recency anchor fixed it, replay-verified via curl. Also repaired 68 pre-existing dev test failures (service/fake gateway route-kwarg skew) and re-derived two context-preflight boundary tests after scaling the output budget 5x per Qodo review. ADR: linked ADR-029 (boundary preserved); no new ADR required. Files: prompt_improvement_prompts.py, prompt_improvement_service.py, test_prompt_improvement_service.py, lessons-live-verification.md. Tests: 769 passed on dev base.
<!-- SECTION:NOTES:END -->

---
id: TASK-34421
title: Batch legacy RAG-context store writes on chatbook import
status: In Progress
created_date: 2026-10-07 02:41
updated_date: 2026-10-07 09:25
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave 4 / F7: recovery-mode citation writes re-serialize and rewrite the whole cross-conversation JSON store once per message making imports quadratic
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Import of 500-cited-message chatbook triggers exactly one store write,Single-message recovery path unchanged,Store content identical to per-message writes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation.md Task 9 (T9)
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->

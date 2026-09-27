---
id: TASK-32917
title: 'Generic hosted provider engine Phase 1: registry + engine + Databricks'
status: In Progress
assignee:
  - '@Robert'
created_date: '2026-09-24 01:01'
updated_date: '2026-09-24 19:30'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Docs/superpowers/specs/2026-09-23-generic-hosted-provider-engine-and-presets-design.md Phase 1
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Databricks usable in Console (streaming + non-streaming) via AI Gateway (live-verified)
- [ ] #2 Engine contract tests parameterized over records green
- [ ] #3 Registry coverage parity tests green
- [ ] #4 No behavior change for existing providers
- [ ] #5 ADR-179 linked
- [ ] #6 Docs updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Plan: Docs/superpowers/plans/2026-09-23-generic-hosted-provider-engine-phase1.md
ADR: backlog/decisions/179-generic-hosted-provider-engine-and-preset-registry.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Phase 1 implementation complete (Tasks 1-14, 27 commits). LIVE GATE PENDING: maintainer has no Databricks instance available (2026-09-24). Run when workspace access exists: DATABRICKS_TOKEN+DATABRICKS_HOST .venv/bin/python -m pytest -q Tests/Chat/test_live_databricks_api.py -s — settles spec O-1 (listing route), captures response envelope for allowances, confirms native-tools flag. AC 'live-verified' intentionally unchecked; do not mark Done until it runs.
<!-- SECTION:NOTES:END -->

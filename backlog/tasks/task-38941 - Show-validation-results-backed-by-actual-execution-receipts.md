---
id: TASK-38941
title: Show validation results backed by actual execution receipts
status: To Do
created_date: 2026-11-05 16:09
labels:
- console
- validation
- agents
priority: medium
references:
- https://uxplanet.org/anthropic-just-shipped-claude-2-0-bbb46a9c1383
- https://code.claude.com/docs/en/plugins/mods/overview
- TASK-35582
documentation:
- Docs/Architecture/console.md
- Docs/Architecture/agent-runtime.md
- backlog/decisions/281-primary-agent-multi-model-code-review.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users distinguish checks that were proposed from checks that actually ran and understand what each result establishes about a generated change. Keep missing evidence and incomplete verification visible.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A bounded current-turn/run validation view shows proposed, running, completed, failed, cancelled, and unknown checks only to the extent existing host/tool receipts can establish those states.
- [ ] #2 An executed check links to its authoritative tool/run evidence and exposes the exact target revision/content identity and outcome where recorded; unsupported provenance remains unknown.
- [ ] #3 Assistant prose or a model-authored checklist alone cannot mark a check as executed or passed, and stale checks are not presented as verification of later changes.
- [ ] #4 Viewing the panel never executes commands, reruns tests, or broadens tool/file authority; existing explicit actions retain their usual review gates.
- [ ] #5 Errors, partial output, unavailable receipts, and session switches preserve truthful status without exposing unauthorized output.
- [ ] #6 Targeted and mounted tests prove the distinction between claimed and observed success and cover target drift, partial/cancelled checks, and missing receipts.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Implementation planning records ADR required: yes/no, ADR path, and reason; any required new or amended ADR is accepted and linked before implementation begins.
- [ ] #2 Applicable targeted automated checks, formatting/lint checks, and live UI or real-provider evidence are recorded before completion; full test sweeps require user opt-in.
- [ ] #3 Relevant user guidance and living Architecture documentation describe the delivered behavior and its limits.
<!-- DOD:END -->

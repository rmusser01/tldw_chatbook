---
id: TASK-38940
title: Show per-request context growth using existing token accounting
status: To Do
created_date: 2026-11-05 16:09
labels:
- console
- context
- diagnostics
priority: medium
references:
- https://uxplanet.org/anthropic-just-shipped-claude-2-0-bbb46a9c1383
documentation:
- Docs/Architecture/console.md
- Docs/Architecture/chat-pipeline.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Help users see why a long-running conversation is approaching its context limit and choose an appropriate response. Make recent context growth visible alongside the existing category breakdown.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The current-conversation context surface shows the last prepared request's total and attributable category changes against a comparable prior request, with snapshot identity/age and estimated counts labeled.
- [ ] #2 Request categories use the existing accounting boundaries; unknown attribution is retained and reconciles to totals rather than being invented or double-counted.
- [ ] #3 Model/tokenizer changes, compaction, branch changes, fresh sessions, and missing observations reset or label the comparison instead of showing a misleading growth delta.
- [ ] #4 Existing compaction and source/tool-management controls are reachable where applicable; observation itself neither compacts history nor disables components.
- [ ] #5 Presentation is bounded, does not add permanent high-frequency polling or unbounded history storage, and never mixes conversations or hidden instruction bodies.
- [ ] #6 Targeted accounting and mounted checks cover growing history, schema/attachment changes, compaction, unknown counts, and stale asynchronous results.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Implementation planning records ADR required: yes/no, ADR path, and reason; any required new or amended ADR is accepted and linked before implementation begins.
- [ ] #2 Applicable targeted automated checks, formatting/lint checks, and live UI or real-provider evidence are recorded before completion; full test sweeps require user opt-in.
- [ ] #3 Relevant user guidance and living Architecture documentation describe the delivered behavior and its limits.
<!-- DOD:END -->

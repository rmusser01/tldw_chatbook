---
id: TASK-34992
title: Engine-migrate or dedupe the moonshot/zai hosted-chat helper pairs
status: To Do
assignee: []
created_date: '2026-10-14 14:30'
labels: [refactor, providers]
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Residual from TASK-33131's evidence-based close: hosted_provider_engine.py (ADR-179) already provides the resolve/normalize family in ProviderRecord-parameterized form, yet moonshot.py and zai.py keep private same-named copies while being dual-registered (engine AND classic API_CALL_HANDLERS). Either migrate the classic handlers onto the engine family (deleting the private copies) or, at minimum, dedupe the mechanically-safe pairs: _normalize_tools (~6-line cosmetic drift), _resolve_api_key (per-provider data the engine already parameterizes), and _best_effort_close (3-line cosmetic drift across hosted_chat/qwencloud, also duplicated in qwencloud_streaming.py and provider_sessions.py). The four semantically-diverged pairs stay as-is unless the engine family grows hooks for their deltas.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The choice (engine migration vs mechanical dedupe) is recorded with evidence, and the chosen scope is implemented without behavior change.
- [ ] #2 Targeted hosted-chat/engine/adapter suites pass.
<!-- AC:END -->

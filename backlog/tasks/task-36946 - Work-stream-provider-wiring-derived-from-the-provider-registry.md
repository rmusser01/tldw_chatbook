---
id: TASK-36946
title: 'Work stream: provider wiring derived from the provider registry'
status: Done
created_date: 2026-10-26 01:09
updated_date: '2026-11-23 17:45'
labels:
- architecture-review-2026-10-25
- providers
priority: medium
documentation:
- Docs/QA/architecture-review-2026-10-25/report.md
- Docs/QA/architecture-review-2026-10-25/verification.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ADR-179 promises that a new engine preset costs one registry record plus one dispatch line. The newest preset needed 13 edit sites in 9 files, about 8 of them repeating `ProviderRecord` data; TASK-33510 and TASK-33511 were bugs of exactly this drift. Credential precedence is handled in TASK-36941.3.

Source: architecture review 2026-10-25, `Docs/QA/architecture-review-2026-10-25/report.md` §F11; claim-by-claim verification in `Docs/QA/architecture-review-2026-10-25/verification.md` (origin/dev 7a5e2513c5).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Adding an engine preset touches only its registry record and its config template section, demonstrated in the final subtask's notes
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Closed 2026-11-23. Three subtasks, all Done:
- TASK-36946.1: the provider lookup tables derive from the registry records.
- TASK-36946.2: the engine presets' dispatch entries (`API_CALL_HANDLERS`, `PROVIDER_PARAM_MAP`) derive from the registry.
- TASK-36946.3 (PR #3718, merged 2026-11-23 as `9418f179e3`): the last hand-kept per-preset lists now derive from the record: the continuation provider set, `KEY_URL_UNCONFIRMED`, the template's `[providers]` model lists (`seed_models`), the auto-refresh keys and the per-account host wording. The template byte pin leaves preset sections out (owner rulings 2026-11-22).

**AC #1** is shown in TASK-36946.3's notes. Two scratch presets, each touching only `provider_registry.py` and `Config_Files/default_config.toml`, passed the registry, setup-facts, readiness, dispatch and template-hash tests. The reviewer repeated this with two presets of their own.

AGENTS.md's "New LLM Provider" steps still told agents to write a `chat_with_<provider>()` and register it. They now point to the record-first recipe in CLAUDE.md.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Adding an engine preset now takes its `ProviderRecord` and its `[api_settings.<key>]` template section, and nothing else. The ADR-179 cost statement and CLAUDE.md say so.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->

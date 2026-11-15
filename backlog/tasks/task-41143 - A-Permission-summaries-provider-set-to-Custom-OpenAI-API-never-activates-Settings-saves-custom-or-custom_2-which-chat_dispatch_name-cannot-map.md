---
id: TASK-41143
title: "A Permission summaries provider set to Custom OpenAI API never activates: Settings saves custom or custom_2, which chat_dispatch_name cannot map"
status: To Do
created_date: 2026-11-15 06:30
labels:
- follow-up
- settings
- providers
- permissions
priority: low
references:
- tldw_chatbook/Library/ingest_analysis.py
- tldw_chatbook/Chat/permission_summary_service.py
- tldw_chatbook/UI/Settings_Modules/settings_field_rows.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top d4e8d51ca8 (endpoint stream b3c75c8234, on dev ecad404bd4); line numbers are that tree's. Amended in review r14cl-rev1.

**Why.** A user who picks Custom OpenAI API (or its second slot) as the Permission summaries provider gets no summaries and is not told why. Found by review r14ep-rev4 (declined there as pre-existing, #3).

**Evidence.**
- Settings' provider choices are readiness keys (`settings_screen.py:15423-15435`, `entry.readiness_key`), and `permission_summary_provider_options` (`settings_field_rows.py:765-794`) saves that value; the two custom slots' readiness keys are `custom` and `custom_2` (`provider_registry.py:1704`).
- `permission_summary_service.py:106-111` returns the inactive base config when `chat_dispatch_name(provider)` is None.
- `chat_dispatch_name` (`Library/ingest_analysis.py:61-91`) maps only `API_CALL_HANDLERS` keys and their normalized spellings. Through run.sh: `'custom'` and `'custom_2'` give None, while `'custom-openai-api'` and `'Custom OpenAI API'` give `custom-openai-api`.
- The same mapper serves Library ingest analysis (`ingest_analysis.py:177-184`: a ready `[analysis_defaults] provider` is refused as "not supported for ingest analysis" when it maps to None) and Dreams (`Dreams/story_service.py:241-256`: the provider falls back to `[chat_defaults] provider`, which is saved as a readiness key, and a None endpoint raises "Dreams provider/model unavailable").
- The mapping already exists: `Chat/console_provider_support.py:36-42`, `_READINESS_TO_EXECUTION_ALIASES`, maps `custom` -> `custom-openai-api` and `custom_2` -> `custom-openai-api-2` (through run.sh; the other three keys it holds, `local_llm`, `local_mlx_lm` and `mistralai`, already map the same way through `chat_dispatch_name`). Library RAG Answer reuses it (TASK-34000.21, `library_rag_answer_service.py:624`).

**Related open task.** TASK-34000.62 (To Do) is the same class of defect for briefings and the Reports demo consent, and its AC#3 asks for one shared alias table. This task covers other callers (Permission summaries, ingest analysis and Dreams, all through `chat_dispatch_name`), so it is not a subset of TASK-34000.62; both should land on the same table.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 chat_dispatch_name maps the custom and custom_2 readiness keys to their chat handlers (custom-openai-api, custom-openai-api-2) through the existing _READINESS_TO_EXECUTION_ALIASES table, with no second copy of it
- [ ] #2 A Permission summaries provider saved as custom or custom_2 activates when that slot is ready; a test covers both, red before the change
- [ ] #3 Library ingest analysis and Dreams resolve a custom or custom_2 provider to its chat handler, pinned by a test each
<!-- AC:END -->

---
id: TASK-41642
title: "Console skill mount discovery and trust preparation still decide trust by code identity"
status: To Do
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- console
- security
- architecture
priority: medium
references:
- tldw_chatbook/UI/Console_Modules/skill.py
- tldw_chatbook/Widgets/compact_model_bar.py
- Tests/Architecture/test_code_identity_ratchet.py
dependencies:
- TASK-41319
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** ADR-356 rule 4 says no authority comes from code identity. TASK-41319 (owner decision: the Send path only) removed the identity checks on the Console Send path; a wrapped or patched collaborator now records the same trust receipt as stock. These checks remain:
- the source-capsule definitions `_CONSOLE_SKILL_CONTEXT_SOURCE`, `_CONSOLE_SKILL_HELPER_SOURCE`, `_COMPACT_MODEL_CONFIG_SOURCE`, `_WIDGET_SOURCE` and `_CONSOLE_SKILL_CATALOG_SOURCE`;
- `_console_skill_source_current`, and the `_source_current`/`_function_current` helpers in `Widgets/compact_model_bar.py`;
- the identity checks in `UI/Console_Modules/skill.py` mount discovery (the `_CONSOLE_SKILL_ENTRY` identity, the capsules and `_DISCOVERY_SOURCE`);
- the controller-source check and the `workers`/`app`/`run_worker` class-method identity checks in `_capture_console_skill_trust_preparation`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Skill mount discovery and trust preparation decide from the app's registrations, not code identity
- [ ] #2 A wrapped collaborator gets the same result as stock, and a foreign one is still refused
- [ ] #3 The code-identity ratchet is lowered by the removed reads
<!-- AC:END -->

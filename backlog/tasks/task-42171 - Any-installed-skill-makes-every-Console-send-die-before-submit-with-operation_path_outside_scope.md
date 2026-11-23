---
id: TASK-42171
title: >-
  Any installed skill makes every Console send die before submit with
  operation_path_outside_scope
status: Done
assignee:
  - '@claude'
created_date: '2026-11-21 05:30'
labels:
  - console
  - skills
  - storage-admission
  - linux
priority: high
references:
  - https://github.com/rmusser01/tldw_chatbook/issues/3347
  - tldw_chatbook/Chat/console_configuration_preparation.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
GitHub issue #3347, reproduced again in a Linux UAT on dev 230205800c (2026-11-20). With any user-installed skill (here one imported from a workspace's `.SKILLS/` folder), every Console send ends about 1 s after Enter. The user sees nothing: no transcript row, no error, and the text stays in the composer. The log has only `Console runtime turn ended with exception_type=RecoveryRequired`; a diagnostic hook captured the reason, `operation_path_outside_scope`.

Each send's native configuration capture holds the Workspace DB's repository operation (`database.connection()`) for the whole capture. When no prepared skill maximum exists (the pre-capture fills one only for built-in-only catalogs), the capture reads each skill's trust status. `SkillTrustService.status_for_skill` takes its own storage operation on the skills and trust roots, and admission refuses a nested acquisition of a different path. The RAG profile `top_k` read is refused the same way, but it is caught and falls back to 5, so a user's profile `top_k` was silently ignored on every send.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 With a user-installed skill and a real trust service, a Console send's configuration capture completes and lists the skill
- [x] #2 The capture's RAG `top_k` is the active profile's value, not the fallback
- [x] #3 A test through the native owned capture with a real WorkspaceDB and SkillTrustService fails on the pre-fix code
- [x] #4 Verified in the running app on Linux: with an imported skill, a workspace chat sends and the agent edits a file
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Red test in Tests/Chat/test_console_configuration_preparation.py: runtime_case + WorkspaceDB-backed registry + LocalSkillsService with a real SkillTrustService and one imported skill, through capture_console_turn_configuration_owned.
2. In the owned capture, read the skill maximum and the RAG profile top_k before entering the ExitStack that holds the Workspace DB operation; pass the maximum as `_skill_context_maximum`.
3. Re-run the Linux UAT with the imported skill.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Fix (Chat/console_configuration_preparation.py, owned `capture()`):** the skill-context maximum and the RAG profile `top_k` are read before the ExitStack that holds the Workspace DB's repository operation. The maximum is passed as `_skill_context_maximum` (only when no planning, prepared or deferred skill context supplies one). Neither read needs that operation, and admission refuses a nested acquisition of a different path.

**Test:** `Tests/Chat/test_console_configuration_preparation.py::test_installed_skill_trust_read_runs_outside_the_workspace_operation` uses a real WorkspaceDB-backed registry, a real `SkillTrustService` with one imported skill, and `capture_console_turn_configuration_owned`. On base it fails with `RecoveryRequired: operation_path_outside_scope` from `_trust_fields_for_record -> status_for_skill`, the same chain as the live app. It also pins `rag_defaults["top_k"]` to the active profile's value (15), not the fallback of 5. A stub trust class could not reproduce it: the native capture qualifies only the real type and otherwise takes the older synchronous path.

**Live (Linux, Debian 12 container, 2026-11-20):** before the fix, 3 of 3 sends died with nothing shown (two workspace chats and one Default chat). After the fix, a fresh account imported `release-notes` from the bound project's `.SKILLS/`, sent in a new workspace chat, and the agent edited `notes.txt`. Timings: Enter to the instructions prompt 2.0 s; Proceed to the approval card 1.1 s; Allow to the file changing 0.32 s; final reply 0.9 s later. No `RecoveryRequired` in the log.
<!-- SECTION:NOTES:END -->

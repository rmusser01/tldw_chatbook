---
id: TASK-41119
title: "Speech stays off after every restore: no review approves tts.profile_store or tts.voices"
status: To Do
created_date: 2026-11-15 02:20
labels:
- follow-up
- backup
- speech
priority: medium
references:
- tldw_chatbook/app_speech.py
- tldw_chatbook/Backup_Recovery/activation.py
- tldw_chatbook/TTS/recovery.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** After any restore, speech start-up never runs again, and the user has no review that turns it back on. TASK-40711's notes name this consumer.

**Evidence.** `app_speech.py:986-994` admits speech start-up through `execution_scope(("config", "models.artifacts", "tts.profile_store", "tts.voices"))`. The approvers that exist call `ActivationStore.approve` for `skills` (`Skills_Interop/recovery_activation.py:459`), plugins (`Plugins/backup.py:502`), sync identity (`Sync_Interop/recovery.py:170`), MCP, Personal Context, Notes, `config` (`recovery_service.py:1199`), `models.artifacts` (`RAG_Search/model_recovery.py:330`) and RAG; none names a `tts.*` owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A restored generation's speech owners get a review (in Backup & Restore > Restored profiles or Speech settings) that approves them
- [ ] #2 After that review and the existing config and model reviews, speech start-up runs again; a test pins pending then approved
- [ ] #3 An approver-census pin allows exactly the new approver for these owners
<!-- AC:END -->

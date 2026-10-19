---
id: TASK-35553
title: Complete 34895's recorded-but-unperformed Local_Summarization_Lib deletion
status: To Do
assignee: []
created_date: '2026-10-19 09:15'
labels: [cleanup, providers]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found verifying TASK-34991's close (review round 1): TASK-34895's tranche 4 (commit 4898f20728) recorded Local_Summarization_Lib's diagnostic sites as outcome=deleted in the ledger re-freeze, but the module survives on dev — 2,321 lines, orphaned (nothing in tldw_chatbook/ imports it), with summarize_with_kobold and eight siblings still defined. Either complete the deletion (drop the module; verify the ledger's deleted-outcome records then match reality) or restore honest ledger outcomes for surviving sites. Also sweep the stragglers the same review found: the orphaned-module family's config-defaults keys (kobold_* sampling keys confirmed dead; llama/ooba/vllm/ollama temperature keys are LIVE via config_temp=True and must stay).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Local_Summarization_Lib is either deleted or its ledger outcomes match its on-disk state, with the choice recorded.
- [ ] #2 The config-defaults sweep distinguishes dead keys (removed) from live config_temp keys (kept), with evidence in notes.
- [ ] #3 Targeted LLM_Calls suites pass.
<!-- AC:END -->

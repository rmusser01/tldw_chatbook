---
id: TASK-35553
title: Complete 34895's recorded-but-unperformed Local_Summarization_Lib deletion
status: In Progress
assignee:
  - '@robert'
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
- [x] #1 Local_Summarization_Lib is either deleted or its ledger outcomes match its on-disk state, with the choice recorded.
- [x] #2 The config-defaults sweep distinguishes dead keys (removed) from live config_temp keys (kept), with evidence in notes.
- [x] #3 Targeted LLM_Calls suites pass.
<!-- AC:END -->


## Implementation Notes

**Deletion completed** (PR on chore/task-35553-lsl-deletion): Local_Summarization_Lib.py (2,321 lines) removed — the ledger's outcome=deleted records are now truthful; no fixture change needed (LIVE_SCAN_MODULES already excluded the module, MODULE_COUNTS stays the historical snapshot). The four dead kobold sampling keys removed from config.py's legacy section (kobold_temperature/top_p/top_k/max_tokens — zero consumers verified: the live chat handler reads api_settings.koboldcpp's different keys, and the summarizer spec lacks config_temp=True). The llama/ooba/vllm/ollama temperature keys KEPT — live via config_temp=True specs, verified against _resolve_summarizer_temp.

Test-file audit: six files mention the module in docstrings only (historical anecdotes, left intact); test_kobold_parameter_wiring.py was already updated by 34895's rounds to exercise the live path. Battery: privacy (ledger reconciliation green), kobold wiring, migration/parity/timeout, analyze, config suites — 146+228 passed; every failure (manifest-boundary ×3, migration ×6 RecoveryRequired flake, packaging wheel-snapshot ×1) reproduces identically on pristine origin/dev.

ADR: not required (completing a recorded deletion; no new decision).

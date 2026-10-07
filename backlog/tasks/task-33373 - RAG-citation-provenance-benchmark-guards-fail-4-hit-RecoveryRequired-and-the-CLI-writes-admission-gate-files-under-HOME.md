---
id: TASK-33373
title: 'RAG citation-provenance benchmark guards fail: 4 hit RecoveryRequired and
  the CLI writes admission gate files under HOME'
status: To Do
created_date: 2026-09-28 20:12
dependencies:
- TASK-33370
labels:
- testing
- rag
- backup-recovery
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Tests/Performance/test_rag_citation_provenance_benchmark.py has 5 failing guards on dev. Four hit RecoveryRequired('raw_source_selection_changed'); they select their own config, so the bootstrap_profile marker PERF-01 used does not help. test_cli_never_reads_or_writes_host_config_data_or_secrets fails because the benchmark CLI writes recovery-bootstrap admission gate files under HOME. That may be a real host-isolation defect in the CLI, not just a test problem. Found 2026-09-28 while verifying PERF-02 (PR #2887) and PERF-01 (PR #2888): the failures reproduce on the unchanged base commit 9cd9aad65f (dev 48019b1914 plus the audit docs commit), so they are pre-existing on dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The HOME write is explained, and either the CLI stops writing under the host HOME or the guard's contract is corrected with a documented reason
- [ ] #2 The 4 RecoveryRequired guards run their real assertions and pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->

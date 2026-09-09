---
id: TASK-32077
title: Design portable Workflows editor and local execution parity
status: Done
assignee:
  - '@codex'
created_date: '2026-09-08 19:17'
updated_date: '2026-09-08 21:15'
labels:
  - workflows
  - design
  - interop
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define the agreed three-pane Workflows experience, first-class local sequential execution, and server-compatible sharing and synchronization before implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design captures the approved three-pane layout, overview and focused step views, continuous collapsible forms, and v1/v2/v3 roadmap.
- [x] #2 Parity matrix separates inspected local building blocks from proposed and verified workflow adapter support.
- [x] #3 Portable definition, execution ownership, requirements mapping, sharing, revisions, and sync conflicts have concrete proposed contracts linked to current server evidence.
- [x] #4 User has reviewed the written design before implementation planning begins.
- [x] #5 The revised design defines executable input and resource binding rules, whole-definition import safety, exact timeout and retry mappings, and bounded run recovery.
- [x] #6 The revised design specifies durable invalid-draft recovery, result provenance, typed reference selection, and narrow-terminal focus transitions.
- [x] #7 Delivery starts with the local end-to-end workflow milestone while retaining the 21-step v1 minimum and an explicit paired-server v1 synchronization milestone.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Pin the latest tldw_server dev revision and inspect definition, adapter-discovery, editor, sharing, and sync contracts.
2. Map the server catalog to inspected Chatbook local services without claiming unimplemented adapter parity.
3. Write a proposed architecture decision, design specification, and complete catalog disposition matrix; make the approved UI and release boundaries explicit.
4. Check catalog coverage, document links, whitespace, and internal consistency, then present the draft for user review before implementation planning.

- ADR required: yes
- ADR path: backlog/decisions/138-portable-workflow-definitions-and-local-execution.md
- Reason: introduces local workflow persistence and execution, cross-client document identity, and a coordinated sync-domain contract.

## Tracking Notes

Created through the Backlog CLI. Its assigned ID 32051 was below the inspected all-remote/all-worktree maximum 32076, so only this newly created record was moved to 32077 before references were added. ADR 138 follows an inspected maximum of 137. Both allocations remain subject to revalidation before integration.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Completed the design and independent-review corrections against tldw_server dev 6cd2745f696af04668a61c20b84ab8a9e69ca5e4. The user approved the revised detailed contracts and numeric safety defaults on 2026-09-08 before implementation planning. ADR-138 is accepted for Chatbook architecture; server synchronization still requires a supporting counterpart contract. Preserved the full 130-name catalog, 21-step v1 target, explicit paired-server v1 sync milestone, v2 branching and v3 parallelism.

Prepared the [first local file-to-note implementation plan](../../Docs/superpowers/plans/2026-09-08-workflows-local-file-to-note.md), with dependency-ordered atomic Backlog work, red/green tests, exact interfaces, narrow-terminal QA and a real local-model completion gate. Source inspection additionally exposed the current Ollama wrapper's live-config timeout/retry/credential fallback; the plan requires a captured bounded request seam before admitting it. Workflow storage coverage explicitly does not claim the still-unimplemented complete-backup subsystem.

Modified only the design specification, parity matrix, ADR-138 and its index row, this design record, and planning/task artifacts. Document checks cover links, embedded JSON/Python syntax, existing source seams, catalog cardinality/dispositions and task dependency hygiene. No application/server code changed or runtime tests run in the design/planning work; runtime, lint/performance/security behavior and full adapter parity remain implementation gates, not claims from source inspection. This documentation-only task has no runtime test or licence/dependency change to qualify. Self-review completed; no new generalizable lesson beyond the existing Backlog allocation and live-verification guidance.
<!-- SECTION:NOTES:END -->

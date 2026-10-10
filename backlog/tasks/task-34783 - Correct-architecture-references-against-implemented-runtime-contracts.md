---
id: TASK-34783
title: Correct architecture references against implemented runtime contracts
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-10 19:27'
updated_date: '2026-10-10 21:39'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the new engineering reference accurate about live hooks, workflow execution, RAG provider costs, storage and module ownership.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Hooks documentation describes the wired v2 runtime and its permission/checkpoint requirements
- [x] #2 Workflow execution scope and reranking request counts match actual source
- [ ] #3 Schema, module paths and storage descriptions avoid known stale or inaccurate claims
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A; existing ADR-163/138/126 contracts apply. Reason: documentation correction preserves implemented behavior. 1. Verify reviewer findings against source. 2. Correct the affected references and AGENTS drift. 3. Check local links, preflight and independent review.

Stacked PR #3018 source-accuracy follow-up (2026-10-10): verify the Eval default/private override selectors, sparse YAML lifecycle and eval.definitions recovery owner; correct stale configuration, schema and persistence claims plus the database-path override sentence; validate relative links and source references before restoring AC #3 and Done.
ADR required: no
ADR path: backlog/decisions/220-evaluation-defaults-and-private-overrides.md; backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: copy-only alignment with the already implemented and reviewed configuration/recovery contracts.

Final source-accuracy review follow-up (2026-10-10): verify BaseAppScreen.refresh versus recompose and the independent ConsoleTranscript lifecycle; correct app-shell.md's conflation of refresh scheduling, mouse release and keyboard focus restoration. Recheck all 83 relative Architecture links, whitespace and the inherited source/doc correspondence; obtain independent immutable copy review before restoring AC #3 and Done. The existing successful carry preflight remains applicable to this Markdown-only correction.
ADR required: no
ADR path: backlog/decisions/190-console-incremental-display-projections.md
Reason: copy-only description of existing screen/widget presentation lifecycle, with no runtime, storage, security or caller contract change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Corrected live v2 hook ownership/effects/checkpoints, workflow session execution, uncached reranking operation counts and local-provider support; removed volatile schema numbers and inaccurate storage/module claims. Resolved AGENTS conflict with current tool-transcript ownership. Independent source review cleared the scoped changes after one local-provider wording correction; all37 local Markdown links resolve and derived-artifact preflight passes. No runtime behavior changes or new ADR; existingADR163/138/126 contracts remain. Documentation-only verification used source and link checks, without new runtime tests.

Stacked-source accuracy follow-up (2026-10-10, reviewed predecessors through fdb0db32364c778e047acead8cbad90a04458077): corrected the Eval reference to distinguish shipped defaults from sparse eval_overrides.yaml beside the effective config.toml, standalone inputs/full exports, raw YAML persistence and eval.definitions backup ownership. Genuine absence remains unused; invalid/unreadable override paths block complete backup, and retained legacy definitions stay inactive until authorized activation. Replaced stale Eval schema numbers with SCHEMA_VERSION, corrected word-bench HTTP ownership, and clarified explicit database path overrides. Verified against selectors, loader/save/recovery adapter, participant binding, EvalsDB and capture-client source under existing ADR-220/126. All83 relative Markdown links resolve and newly named source files exist; whitespace and Backlog file readability checks pass. Copy-only change; no runtime tests, full suite or new platform qualification. Root will run the final stacked preflight.

Final carry/source review (2026-10-10): rebased the four original docs commits from fdb0db32364c778e047acead8cbad90a04458077..673cc04ca70acdbc9688f51007e65cf7a2336e2e onto exact stack parent 590c18b2b2551f7871684854f982ba79f022bb0a. Pre-copy docs runtime head: 3810f8aab9a8c318003281f51f5e47b8e1b72570. Range-diff reports all four patches unchanged; all 25 owned paths retain their old blobs before this final copy correction. Runtime, tests and CI remain byte-identical to the new parent; inherited lessons-testing-evidence.md exactly matches its parent blob 7730bcd0b7256038f47960cd7075fdba53cf6f67. The preserved PR3040 follow-up branch stays at c08aa049ed2b7d50188e63a105d15e74de60c715.

Independent source review identified one existing app-shell.md sentence conflating refresh scheduling and focus restoration. Corrected it to name screen refresh(recompose=True) mouse release, BaseAppScreen.recompose focus capture/restoration for scheduled and direct awaits, and the independent child-widget lifecycle. Verified actual BaseAppScreen.refresh/recompose plus ConsoleTranscript.recompose/refresh_messages/on_mount; Eval selectors, sparse private override loading and eval.definitions recovery claims remain source-consistent. Existing ADR190 applies; no new ADR or runtime change.

All 83 relative Architecture links resolve before and after the sentence correction. Git whitespace check passes. The normal hash-pinned preflight on 3810f8aab9a8c318003281f51f5e47b8e1b72570 exits zero with all checks passed; its inherited UI census is 175/floor 175. The later Task34785 CI wiring commit b2d24076ca1af97b4a020aa7d90448da9420d547 is not an ancestor of this pinned parent, so no broader claim is made about that separate carry. This Markdown-only correction needs no runtime lint/formatter or test replay; Backlog readability is checked separately after task edits. Receipts: /private/tmp/pr3038-carry-checks.json; pr3038-carry-copy-links.json; pr3038-carry-preflight.log. Final immutable copy review remains pending before restoring AC #3 and Done.
<!-- SECTION:NOTES:END -->

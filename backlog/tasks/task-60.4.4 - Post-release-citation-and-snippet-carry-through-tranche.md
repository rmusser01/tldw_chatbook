---
id: TASK-60.4.4
title: Post-release citation and snippet carry-through tranche
status: In Progress
labels:
- post-release
- rag
- citations
- ux
priority: medium
parent_task_id: TASK-60.4
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Carry retrieved snippets and citations from Library/Search/RAG into Console answers, Artifacts, exported Chatbooks, and downstream saved work only after source authority is visible and testable end-to-end.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Citation and snippet carry-through scope references TASK-60.3 actual-use audit evidence.
- [ ] #2 Retrieved evidence keeps source identity, snippet text, and authority labels through Console responses and saved artifacts.
- [x] #3 Exported Chatbooks preserve citations/snippets in a user-readable and machine-checkable form.
- [ ] #4 QA verifies source-to-answer-to-artifact carry-through with actual app use before completion.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Open an epic tracking PR from `codex/citations-snippets-epic` to `dev` with the approved stacked-PR design.
2. Land the evidence/citation contract in a sub-PR targeting the epic branch.
3. Land Library/Search-RAG evidence bundle generation in a sub-PR targeting the epic branch.
4. Land Console evidence staging and blocked-state feedback in a sub-PR targeting the epic branch.
5. Land answer-level citation injection, response parsing, and validation in a sub-PR targeting the epic branch.
6. Land message persistence for evidence bundles and citation refs in a sub-PR targeting the epic branch.
7. Land Chatbook artifact/export evidence preservation in a sub-PR targeting the epic branch.
8. Close the epic with actual CDP/textual-web QA evidence, user-approved screenshots, focused automated tests, roadmap updates, and completed Definition of Done.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Epic setup started. Added `Docs/superpowers/specs/2026-05-23-citation-snippet-carry-through-epic-design.md` to define the stacked PR model, evidence/citation contracts, answer-level citation injection scope, persistence/export requirements, and QA gates. The implementation remains split across sub-PRs so each seam can be tested and reviewed independently.

Library/Search-RAG evidence bundle slice added. `build_library_rag_console_live_work_payload()` now attaches a serialized `EvidenceBundle` that preserves query, source identity, snippets, citation labels, local/server authority, workspace visibility, and active-context eligibility. Regression coverage verifies local and server Console handoff payloads, cross-workspace blocked evidence, and provenance-only source identity fallback before the later Console staging and answer-citation slices.

Console evidence staging slice added. Console staged context and the Run Inspector now summarize staged Library/Search-RAG evidence count, source authority, reference status, and snippets without exposing raw `evidence_bundle` metadata. Send recovery now blocks RAG-grounded Console sends when no eligible evidence is available, preserving the composer draft and showing native Console feedback. `ChatHandoffPayload.model_context_block()` formats staged evidence in a readable model-context section. Actual textual-web/CDP screenshot evidence was captured at `Docs/superpowers/qa/product-maturity/post-release-ux-hci/actual-screenshots/2026-05-23-console-evidence-staging.png` and approved by the user.

Answer-level citation injection slice added. Staged evidence context now includes explicit citation instructions, available/blocked evidence state, and stable `[S#]` source labels. Assistant responses are parsed for citation markers and validated against the staged evidence bundle on both streaming and non-streaming completion paths, attaching validated, unknown, uncited, blocked, stale, missing, or insufficient-evidence metadata to the generated message widget for the later persistence and export slices.

Console-saved Chatbook artifact preservation slice added. Saved assistant response metadata now carries bounded JSON-safe `citation_validation` and staged `evidence_bundle` payloads when present. Artifacts and Home resume paths expose compact citation/evidence summary fields for Console launch payloads so grounded saved answers retain validation status, cited labels, bundle identity, source count, and snippet count. Actual app QA remains follow-up work for this task.

PR review hardening added for the artifact preservation slice. Citation summary text now preserves falsy-but-valid values, caps evidence reference counting on resume paths, and is sanitized at Home/Artifacts payload boundaries before reaching Console live-work rendering.

Exported Chatbook ZIP preservation slice added. Conversation exports now retain JSON-safe message-level `citation_validation`, `evidence_bundle`, and citation payloads in the conversation JSON, emit a per-conversation Markdown citation/evidence report with readable snippets and source details, and advertise report paths plus citation/source/snippet counts through the manifest content-item metadata. Regression coverage verifies machine-checkable JSON preservation, readable snippet reports, and manifest metadata.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->

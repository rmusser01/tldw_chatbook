---
id: TASK-32026
title: Pilot a read-only bulk-reader agent with a reproducible comparison
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-08 04:45'
updated_date: '2026-09-08 06:30'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Evaluate whether a cheaper named reader can reduce the total cost of answering repository and long-transcript questions while preserving source evidence and answer quality. Start with an explicit preset and opt-in experiment before considering automatic routing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Settings can populate and save an editable bulk-reader definition without changing or overwriting existing definitions.
- [x] #2 The preset permits only workspace file discovery and reads and requests quoted findings with explicit gaps; its selected model reaches the Console provider boundary.
- [x] #3 A reproducible comparison exercises the existing reader runtime on nonsensitive repository and transcript fixtures, records both worker and main requests, and leaves missing usage or price unknown.
- [x] #4 Targeted runtime, UI, and comparison checks pass and documentation distinguishes a runnable experiment from measured model-quality or cost results.
- [ ] #5 A user-selected supported model pair has a recorded live comparison and manual assessment of evidence, quality, and total cost; failed or unavailable measurements remain explicit.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add an editable read-only bulk-reader preset and verify real named-agent model/tool behavior. 2. Add a scratch-corpus direct/delegated comparison with per-call accounting and manual quality review. 3. Run targeted checks and the chosen live model comparison. ADR required: no. ADR path: N/A. Reason: reuse existing named-agent/provider contracts. Plan: Docs/superpowers/plans/2026-09-07-bulk-reader-pilot.md; spec: Docs/superpowers/specs/2026-09-07-bulk-reader-pilot-design.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added an editable Bulk reader preset to the canonical Settings Agents form. Selecting it creates an unsaved form; existing CRUD and duplicate validation remain authoritative. Its four file tools are read-only and its evidence instructions are advisory.

A real-gateway reproducer showed Console discarded named-agent model overrides. The adapter now uses an immutable call-local resolution for request preparation, dispatch and usage, preserving parent continuation ownership.

Added a pinned four-case synthetic repository/transcript comparison through real AgentService, Console adapter, local tools and scratch SQLite. It records each main/worker call, requires named-worker content access, flags token-limited or incomplete results, and keeps missing usage/prices unknown. The live runner supports Moonshot/ZAI through their bounded timeout/retry seam; other preset providers are unchanged. Guide: Docs/Examples/agents/bulk-reader/README.md.

Worktree evidence: 19 focused preset/UI/adapter/regression tests passed, including production CSS at 120x40 and 70x40. After evaluator review fixes, 20 focused evaluator/preset/adapter tests passed. Changed-file lint, formatting, compilation and CLI help/refusal checks passed. One pre-existing requests dependency warning remains. Integration evidence: 37 targeted tests passed in the user checkout, covering the preset, Settings UI, evaluator, worker model override, provider continuation and concurrent subagents. Changed-file Ruff checks and new-file formatting passed. The final review found a test-only hardcoded interpreter path; replacing it with sys.executable passed the affected CLI subprocess test and Ruff/format checks in the integrated checkout. Scoped re-review confirmed all findings addressed with no new Critical/Important breakage. All 16 task files were integrated while preserving staged changes and concurrent edits; the initial patch round trip verified the original bytes for 15 files, and the testing lesson was appended without altering existing entries.

ADR required: no. ADR path: N/A. Existing named-agent/provider contracts are reused; no schema, dependency or automatic routing is introduced. Added an incident-based testing lesson about verifying model selection at the provider boundary.

Live validation remains pending: no user-selected Moonshot/ZAI main/worker model pair has been supplied, no billable call was made, and no real quality or savings claim is established. Acceptance criterion 5 remains open and status stays In Progress.
<!-- SECTION:NOTES:END -->

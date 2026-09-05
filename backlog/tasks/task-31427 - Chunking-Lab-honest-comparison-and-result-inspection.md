---
id: TASK-31427
title: Chunking Lab - honest comparison and result inspection
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-04 23:13'
updated_date: '2026-09-05 02:36'
labels:
  - chunking
  - chunking-lab
dependencies:
  - TASK-31421
  - TASK-31422
references:
  - backlog/decisions/118-chunking-lab-local-execution-and-recovery.md
documentation:
  - Docs/superpowers/specs/2026-09-04-chunking-lab-design.md
  - Docs/superpowers/plans/2026-09-04-chunking-lab.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Help users inspect what changed between captured results without implying unsupported alignment, comparable units, or retrieval quality. Covers spec section 7 and AC 3, 9-10, 14-15, 19, 22, 25. ADR required: yes; ADR path: backlog/decisions/118-chunking-lab-local-execution-and-recovery.md; reason: result interpretation and comparison contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Comparisons require matching sample, backend, engine, and execution versions but allow different methods and tokenizers; mismatches explain why and offer rerunning both.
- [x] #2 Common character and chunk measurements, explicit method budgets, and named token-count identities avoid incompatible deltas; elapsed time is labeled an observation and no quality score is invented.
- [x] #3 Configuration diffs show selected immutable result snapshots including ordered operations, captured defaults, classifier and metadata view, runtime differences, and newer-draft staleness.
- [x] #4 Chunk inspection is bounded and keyboard usable at 10000 chunks; linked source highlights and overlap measurements use verified spans only, with transformed-text inspection or unavailable explanations otherwise.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/118-chunking-lab-local-execution-and-recovery.md. Reason: implements approved truthful comparison and bounded result inspection. 1. Read Task7 brief/context, result/span contracts and established native UI design. 2. Write failing compatibility/statistics/diff and results-region tests. 3. Implement common-measurement summaries and captured config diffs without false quality rankings or guessed mappings. 4. Build bounded paged chunk inspection and selection/rerun events using theme/focus conventions. 5. Run targeted comparison/Pilot tests and bounded viewport inspections, scoped static checks, self-review and independent review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

Implemented captured-result compatibility, Unicode/word distributions, explicit method budgets, separately identified token measurements, verified overlap, and effective/authored JSON Pointer diffs in `Chunking/lab_comparison.py`. Different methods/assets remain experimental variables; incompatible results receive reasons without comparative deltas.

Added `UI/Chunking_Lab_Modules/results_region.py`: 100-row pages per candidate, one 8192-code-point paged literal inspector, off-loop summaries/diffs/selection preparation, source/transformed coordinates, runtime/metadata details, explicit Previous and Newer draft badges, and selection/rerun messages. View patches live under `session.view.results`; the owning screen supplies current-versus-previous records and persists navigation. No DB loads or recipe execution occur in the region. Regenerated only the changed `css/widget_defaults_self.tcss` using the existing builder.

Validation: 13 comparison and 11 mounted Textual cases pass (24 total), including 10000-row navigation, literal text tail access, restored selections without edit echoes, zero/failed inspection, and 80x24/120x40/160x50 geometry. Two bounded screenshot rounds confirmed the compact row fix, captured diff, and verified linked source highlights. Ruff lint/format and diff whitespace checks pass. Ten targeted CSS consolidation cases pass; the composed-class coverage and class-level allowlist nodes fail identically on the exact Task7 base archive, with unchanged offending Console/Library sources. These inherited failures remain outside scope; no full test sweep was run.

ADR check: direct implementation of [ADR-118](../decisions/118-chunking-lab-local-execution-and-recovery.md); no new ADR. No new lesson added: encountered stylesheet and repeated-text mapping traps are already documented. Status remains In Progress pending independent review and screen integration. Task notes were edited directly under the documented five-digit Backlog CLI workaround.

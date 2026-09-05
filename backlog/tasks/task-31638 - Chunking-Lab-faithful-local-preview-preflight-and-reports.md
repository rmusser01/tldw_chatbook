---
id: TASK-31638
title: Chunking Lab - faithful local preview preflight and reports
status: Done
assignee:
  - '@codex'
created_date: '2026-09-04 23:10'
updated_date: '2026-09-05 00:00'
labels:
  - chunking
  - chunking-lab
dependencies: []
references:
  - backlog/decisions/118-chunking-lab-local-execution-and-recovery.md
documentation:
  - Docs/superpowers/specs/2026-09-04-chunking-lab-design.md
  - Docs/superpowers/plans/2026-09-04-chunking-lab.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Enable faithful unsaved-template experiments on the completed ADR-078 runtime without reviving the retired pipeline or changing the server-parity validator. Implements Chunking Lab spec sections 5-7 and AC 6-9, 14, 18-19. ADR required: yes; ADR path: backlog/decisions/118-chunking-lab-local-execution-and-recovery.md; reason: local execution and structured result contract. Execution baseline is current dev with TASK-19801 through TASK-19806 completed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Run and Lab Save use a named local capability preflight: unknown executable fields, unavailable assets, legacy shapes, and implicit network or LLM work are refused while metadata and classifier selection rules survive.
- [x] #2 Unsaved preview and applying the same saved flat template share the existing runtime seam and produce equivalent full pre/chunk/post outputs; valid empty outputs never invoke default chunking.
- [x] #3 Structured results retain supported engine and operation metadata with authoritative fields protected; source alignment is exact and verified or explicitly unavailable, including repeated text and transformations.
- [x] #4 Real deterministic execution fixtures cover supported filtering, merging, context, dict-output behavior, and clear refusal of unsupported combinations; existing parity validation and vendor protections remain unchanged.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/118-chunking-lab-local-execution-and-recovery.md. Reason: faithful local execution and structured reports. Follow Task 1 of Docs/superpowers/plans/2026-09-04-chunking-lab.md: write failing capability and execution tests; implement immutable report models and separate Lab preflight; extend the shared non-vendored runtime seam; verify focused runtime/parity regressions; self-review and independent review before completion.
<!-- SECTION:PLAN:END -->

## Implementation Notes

Current task-level status: Done after independent review. Earlier In Progress and
pending-review statements below are preserved chronology. Final branch acceptance
and original non-green integration/platform/privacy qualifications are tracked in
[Chunking Lab verification](../../Docs/Chunking_Lab_Verification.md).

<!-- SECTION:NOTES:BEGIN -->
Implemented the named `prepare_recipe` admission seam for Run/Lab Save consumers,
canonical immutable authored/effective recipes and runtime identity, and shared
`execute_prepared`/legacy `apply_template` structured execution (ADR-118, preserving
ADR-078). Qualified methods are English words and fixed-size characters; all five
preprocessing and five text-postprocessing operations have explicit capability
defaults and field checks. Asset-dependent, hierarchical, LLM, legacy, unknown,
ignored, and known lossy settings are refused rather than rewritten. The pinned
word processor drops a word with `preserve_sentences=True`; a real characterization
fixture documents this and Lab specifically refuses that value.

Reports retain preprocessing metadata, engine metadata/non-text fields, each
contributor's transformation history, protected final counters, and only unique
verified exact source/transformed spans. Filtering retains verified survivors;
merging/context rewriting marks maps unavailable. Legitimate empty output stays
empty. Saved apply retains its signature, tolerant admission, and legacy flat
offset adapter; it now shares one chunk execution with preview instead of rerunning
the chunk stage for offset comparison. Vendor code and the parity validator are
unchanged. UI wiring and worker resource ceilings belong to the later Lab tasks.

Changed `Chunking/lab_models.py`, `Chunking/lab_preflight.py`,
`Chunking/template_runtime.py`, and the three focused Chunking test files.
Self-review completed; status remains In Progress pending independent review.

Verification from the isolated `codex/chunking-lab` worktree:

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chunking/test_lab_preflight.py Tests/Chunking/test_lab_execution.py Tests/Chunking/test_template_runtime.py Tests/RAG_Admin/test_template_validation.py -q
108 passed, 2 warnings in 4.71s

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chunking/test_chunking_templates.py Tests/Architecture/test_vendor_pin_consistency.py -q --tb=short
24 passed, 1 warning in 0.57s
```

New-file/test Ruff checks and formatting pass; modified runtime ranges are
formatted. Runtime lint passes with only its pre-existing UP006/UP007/UP035/
UP037/UP045/RUF022 debt excluded (baseline measured separately). `git diff --check`
passes. Warnings: pre-existing RequestsDependencyWarning and vendored
`datetime.utcnow()` deprecation reached by the new sanitation fixture. No optional
method is counted as passing evidence, no dependencies were changed, and no full
suite was run. Detailed RED/GREEN history and capability matrix are in the local
handoff `.superpowers/sdd/2026-09-04-chunking-lab/task-1-report.md`.

### Review fix round 1

Fixed the reviewed mixed-whitespace attribution defect: `one  two one two`
emits two `one two` chunks, but the first must not borrow the second occurrence's
exact span. The shared seam now conservatively marks word mappings unavailable
when whitespace normalization changes the processed document; no originating
engine windows are exposed to prove attribution in that case. It observes the
engine's resolved method, including legacy method overrides, and leaves actual
chunk output and fixed-size exact mappings intact. Added double-space, tab,
newline, saved-apply, and override regressions under the isolated pytest harness.
ADR-118's verified-or-unavailable policy applies; vendor/parity code is unchanged.

```text
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Chunking/test_lab_preflight.py Tests/Chunking/test_lab_execution.py Tests/Chunking/test_template_runtime.py Tests/RAG_Admin/test_template_validation.py -q
115 passed, 2 warnings in 3.93s
```

Ruff checks and modified-range formatting pass using the same documented
baseline runtime lint exclusions. `git diff --check` passes. Still In Progress
pending re-review; no unisolated application imports were used in this fix round.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

Formerly TASK-31421; moved to TASK-31638 during the user-approved
2026-09-05 pre-push bookkeeping correction. Upstream dev independently uses
31421–31424; the complete Lab chain moved together to preserve dependency
ordering without changing upstream tasks. Original creation dates, acceptance
and implementation history are retained. Historical commits and ignored review
artifacts retain the old IDs; current references use the new IDs. See
Docs/Chunking_Lab_Verification.md for the complete mapping and provenance.

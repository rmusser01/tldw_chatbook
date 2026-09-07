---
id: TASK-19901
title: 'Chunking auto-selection (sub-project #3): vendored planner, three-tier Auto, picker + re-chunk wiring'
status: Done
assignee: []
created_date: '2026-08-22'
updated_date: '2026-08-22'
labels:
  - chunking
dependencies: [TASK-19806]
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Sub-project #3 of the chunking parity program (single PR, stacked on #2/ADR-078's storage and ADR-073's vendoring pattern — no new ADR; the spec's §8 rulings are the long-form record): vendor `auto_planner.py` from the existing pin, decide auto-selection in exactly one module (`Chunking/auto_selection.py` — three tiers: strictly-positive classifier template → vendored planner plan → plain), ride the existing picker's `chunk_template` slot with the reserved name `"auto"` across all six ingest seams, persist `mode`/`auto_tier`/`rationale` per media, re-resolve on re-chunk (with the re-stamp carry fix so a tier flip leaves no stale template key), and close with the media-type vocabulary, reserved-name guard, docs/CHANGELOG, and final review.

Spec: `Docs/superpowers/specs/2026-08-22-chunking-auto-selection-design.md` (§7 ACs 1-16; §8's eight rulings incl. the three review-added; §0.2 upstream divergences). Plan: `Docs/superpowers/plans/2026-08-22-chunking-auto-selection.md` (five tasks, one PR).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `auto_planner.py` vendored from the existing pin (manifest+script move, 37→38 files, byte-faithful modulo rewrite rules, zero new shims) with sync contract tests updated, and exactly one module decides auto-selection (`Chunking/auto_selection.py`) — `TemplateClassifier`'s fencing test permits construction only there while `TemplateLearner`/`TemplateManager` stay fully fenced (spec ACs 1-2)
- [x] #2 Tier semantics pinned: tier 1 selects only strictly-positive scores tie-broken by `priority` (absent → 0) then listing order, excludes stored-invalid candidates, skips malformed blocks with a reason, and provably never auto-selects the six #2 built-ins; §0.1's corrected threshold semantics (no-block → never; block-with-absent-min_score → positive selects); tier 2 runs the vendored planner with `llm_available=False` and chatbook-real `semantic_available`, never running when tier 1 won (mutation-verified); planner parity fixtures byte-match the vendored planner (spec ACs 3-6)
- [x] #3 The picker offers "Auto" with "None (manual settings)" still the default and its output byte-identical; Auto reaches all six ingest seams via the existing `chunk_template` slot with sentinel `"auto"` and no seam-specific branching; `chunking_config` records `mode`/`auto_tier`/rationale with the `template` key only on template-tier wins — both #2 readers round-trip, no schema change (spec ACs 7-9)
- [x] #4 Re-chunk re-resolves a stored `mode:"auto"` (a classifier change flips the tier) with stored explicit template names behaving exactly as #2; config `default_template` never triggers auto and server mode is unaffected — and the Task-4-review carry is closed: `chunking_config` is re-stamped after an auto re-chunk so a tier flip leaves no stale template key (spec ACs 10-11)
- [x] #5 Vocabulary, reservation, defect, docs: the media-type mapping table covers every ingest media-type string and is enforced by the §6.9 vocabulary test; the name `"auto"` is reserved at create/rename with a named error (pinned), pre-existing `"auto"`-named rows flagged shadowed, never selected nor auto-shadowed; UPSTREAM_DEFECTS.md #16 filed (upstream auto/explicit paths apply only the hierarchical block and drop other stages); user guide gains Auto with the opt-in explanation and a re-verified stamp, CHANGELOG entry (spec ACs 15, 14, 16, 12)
- [x] #6 Close-out: targeted suites green (§6.8), import-weight guard green, no new core dependencies; the full-UI PR-checkbox run closed with all 34 failures attributed as pre-existing dev drift reproducing identically at the #2 tip; final review verdict READY — 16/16 ACs pass (spec AC 13)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Vendor `auto_planner.py` from the existing pin; unfence `TemplateClassifier` for `auto_selection` (plan Task 1)
2. Three-tier decision engine in `Chunking/auto_selection.py`; reserved `"auto"` name; defect #16 (plan Task 2)
3. Media-type vocabulary table + planner parity fixtures; un-skip the planner suite (plan Task 3)
4. Auto through the picker, resolution, persistence, and re-chunk (plan Task 4)
5. Re-stamp fix (stale template key), user guide + CHANGELOG, final review (plan Task 5)
<!-- SECTION:PLAN:END -->

## Implementation Notes

Approach: one PR stacked on #2 — planner vendored byte-faithfully from the pin, the entire decision in one new module, Auto riding the existing picker slot as a reserved sentinel, per-media `mode`/`auto_tier`/`rationale` persistence, re-chunk re-resolution, and the Task-4-review stale-template-key carry fixed via a re-stamp helper.

- Commits `272baabc7..30c269500` (8 commits, incl. tidy-up `623f90fe4` guard-test revert and vocabulary fix `b873ed381`); SDD tasks 1-5.
- Rulings: `semantic_available=True` documented default accepted (no enabled-state exists in `[embedding_config]`; the flag gates the LOCAL semantic strategy) — reconcile spec §4.2/AC-5 wording at sub-project #6; `set_document_config` rejected as the re-stamp writer (`ensure_ascii` default breaks the LIKE reader for non-ASCII names; no `version+1` aborts the sync trigger) — the helper mirrors the ingest seam instead; the §0.2 divergences are deliberate (winning template runs all three stages vs upstream's hierarchical-only extraction; reserved-name sentinel vs upstream's separate form flag; chatbook's tier composition).
- Final review READY (stronger than #2's: zero must-fix items), 16/16 ACs incl. two independent byte-level verifications; long-form record: spec §8 + `.superpowers/sdd/2026-08-22-chunking-auto-selection/progress.md`.
- Follow-up filed: TASK-19902 — nested-transaction hardening of the two-step re-stamp (final-review Minor).

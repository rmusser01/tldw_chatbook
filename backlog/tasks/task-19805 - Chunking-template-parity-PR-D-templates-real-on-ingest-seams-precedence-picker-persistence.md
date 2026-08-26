---
id: TASK-19805
title: 'Chunking template parity PR D: templates real on ingest — six seams, precedence, picker, persistence'
status: Done
assignee: []
created_date: '2026-08-21'
updated_date: '2026-08-21'
labels:
  - chunking
dependencies: [TASK-19804]
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR D (consumers) of the Chunking Template Parity sub-project (ADR-078): make templates actually do something — wire them into all six ingest seams (spec §9.2) with the §9.1 precedence fix (a resolved template's chunk-stage options beat builder defaults; only user-changed form values override), land the PR-B carry-forwards (AC-24 stored-invalid flag/refusal halves, the `resolve_template` deleted-filter), ship the ingest template picker, `chunking_config` persistence readable by both existing readers, and the `[chunking]` config section.

Spec: `Docs/superpowers/specs/2026-08-21-chunking-template-parity-design.md` (§9, §12 PR-D ACs 34-40 plus AC-24 halves). Plan: `Docs/superpowers/plans/2026-08-21-chunking-template-parity.md` (PR D, Tasks 10-11).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Ingest resolves picker/batch → config default → plain options and re-chunk resolves stored per-media → config default → plain options (§9.1); an unresolvable template name fails the ingest item with a named error and is skipped-and-counted by re-chunk — never a silent fallback (spec ACs 34, 37)
- [x] #2 Precedence holds and is proven by governance, not arrival: two different templates on one fixture produce different persisted chunk rows per media-type family, the "None" default is byte-identical to today's path, and template chunk-stage options beat builder defaults with only user-changed form values overriding (spec ACs 35-36)
- [x] #3 Persisted chunks carry `chunking_template`/`chunking_params` alongside `chunk_engine_version`, and `Media.chunking_config` is written in a shape both existing readers understand (`LIKE '%"template": …'` and `json_extract($.template)`) (spec AC 38); the PR-B carry-forwards landed here — stored-invalid listed-with-flag, apply-path refusal with named `InvalidTemplateError`, and the `resolve_template` deleted-filter (spec AC 24 halves)
- [x] #4 The picker lists DB templates, defaults to "None (manual settings)", escapes markup in labels, populates off the mount path, and is hidden in server mode; `[chunking]` ships in the config template/defaults with a test asserting the real loader emits the section (spec ACs 39-40)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Honor templates at the six ingest seams; precedence fix; AC-24 halves; `resolve_template` deleted-filter (plan Task 10)
2. Picker + `chunking_config` persistence + `[chunking]` config section; MIGRATION_GUIDE rewrite decision (plan Task 11)
<!-- SECTION:PLAN:END -->

## Implementation Notes

Approach: templates became real at every ingest surface — seam-by-seam option flow with the §9.1 precedence ruling, the PR-B deferred halves (stored-invalid flag, named apply-path refusal), the picker, dual-reader `chunking_config` persistence, and the `[chunking]` config tier.

- Commits `2e087aecf..78c0844ab` (PR-D marker `78c0844ab`); SDD tasks 10-11.
- Deviations-with-rulings: spec §13.1 and `.superpowers/sdd/2026-08-21-chunking-template-parity/progress.md` — user value == schema default → template wins that key (false-negative in the safest direction, inherent to the snapshot format; accepted); picker shows None for a deleted template's name on display while submit raises named `TemplateResolutionError` (§9.1/AC 37); MIGRATION_GUIDE rewritten to the dict-based contract here (carried from PR C).

---
id: TASK-38537
title: Preserve mix-in type across portable records and server capabilities
status: Done
created_date: 2026-11-01 19:30
dependencies:
- TASK-38536
labels:
- prompts
- console
references:
- backlog/decisions/286-console-prompt-mix-ins.md
documentation:
- Docs/superpowers/specs/2026-11-01-prompt-mix-ins-design.md
- Docs/superpowers/plans/2026-11-01-prompt-mix-ins-implementation.md
- backlog/decisions/286-console-prompt-mix-ins.md
assignee:
- '@codex'
updated_date: 2026-11-01 20:49
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users move mix-in definitions safely and use sources that explicitly support them without losing their identity or literal instructions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Chatbook record v2 preserves mix-in definitions and metadata while legacy and v1 Prompt/Recipe records retain their meanings; unsupported or malformed records fail before mutation.
- [x] #2 Mixin Markdown uses an explicit envelope that cannot be mistaken for a legacy ordinary Prompt, preserves braces/whitespace and never logs instruction bodies during import failure.
- [x] #3 Server brief/detail/write/version schemas and capability checks recognize supported mix-ins and clearly refuse older sources without local fallback, type flattening or auto-enabling imported entries.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing exact Chatbook v2 and distinct Markdown envelope regressions.
2. Implement strict version/type dispatch, preserving v1 and exact stored mix-in definition.
3. Extend known server schemas/capability normalization and verify refusal where revision-qualified reads are unavailable.
4. Run targeted codec/import/source/schema tests and scoped static checks; update export docs and self-review.
ADR required: no
ADR path: backlog/decisions/286-console-prompt-mix-ins.md; backlog/decisions/057-portable-chatbook-prompt-records.md
Reason: Implements approved portability/source boundaries.
Review round 1 (I1): reproduce BOM-prefixed valid/malformed Mix-in Markdown through the real parser and UTF-8 file import; normalize only the optional transport BOM consistently for detection/framing; run covering parser/import/codec checks, scoped static checks and self-review. ADR required: no; ADR path: backlog/decisions/286-console-prompt-mix-ins.md and 057-portable-chatbook-prompt-records.md; reason: privacy-boundary bug fix within the approved envelope contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented strict portable Mix-in record v2 and distinct single-line JSON Markdown framing while preserving existing Prompt/Recipe v1 and legacy behavior. Mix-ins keep exact stored definition text, literal body, metadata and placement; invalid records fail before mutation and do not become legacy Prompts.
Known server schemas now retain mixin identity, and source capabilities preserve actual advertised type/kind pairs. Added read-only mixin_snapshot_reads capability (local true, server false). Existing client get/version APIs cannot fulfill coherent UUID/revision-qualified capture; capture remains visibly unavailable with no invented endpoint or local fallback. Both server adapters and PromptScopeService refuse unsupported expected_version writes, including nested request_data. The local coherent UUID snapshot now includes current name for renamed records.
A real Mix-in archive regression found body text reused as a keyword leaking through the shared KeywordStore write path; those sinks now log fixed metadata/categories without changing storage/exception behavior. Updated the existing failure injection to target the live shared keyword owner. Added the supported bootstrap_profile marker to the Chatbook suite so real guarded constructors retain their private source.
Validation: 423 targeted tests passed; five inherited digit-sentinel privacy assertions were deselected after identical failure on exact BASE 72ec16000cb953c349ee2bf9a56b8ac7295b0ab1. Scoped Ruff has the same 515 diagnostics as exact BASE with zero introduced findings; formatter debt is identical in the two inherited owner files, and the other 13 changed Python files pass formatter checks. git diff --check passed. No full sweep or live server test.
Documentation: Docs/User_Guide/library/prompts.md and import-and-export.md explain v2/envelope/import and source limits. ADR required: no new ADR; implements backlog/decisions/286-console-prompt-mix-ins.md and 057-portable-chatbook-prompt-records.md. Self-reviewed version/type dispatch, pre-mutation validation, exact body preservation, downstream logging, policy/UUID/source refusal and conditional writes. Kept In Progress for independent review.
Review round 1 I1 fixed: parse_markdown_prompts_from_content now removes one optional leading transport BOM into an envelope-only view and uses that same view for discriminator and framing validation. The original legacy input and literal JSON contents are unchanged. Added eight real parser/UTF-8 file-import cases covering valid exact body/definition preservation (including a literal body BOM), bad JSON, missing fence and trailing legacy sections. Malformed imports return bounded record failures before DB mutation, with no raw snippets/body logs or legacy interpretation. TDD: eight expected RED failures; covering parser/import/codec run GREEN with 114 passed. Scoped Ruff matches exact BASE 4f33145297d61846ba5c3f8ee03dd926f4e3e479 (162 inherited findings, zero introduced); both files pass formatting and git diff --check. No full matrix, inherited failure rerun, live server or recovery-fence change. Self-reviewed single-prefix-only normalization, framing/decoder routing and real import error handling. Task remains In Progress for scoped re-review.
Independent review accepted source/portable contracts and identified BOM-prefixed Markdown falling into legacy interpretation/logging. Fixed in ffb0326c50f0894f924f887a3a07ea750e7ce67a with eight real parser/file-import regressions and 114 covering passes; scoped re-review approved I1 with no new breakage. Initial matrix had 423 passes and five exact-BASE archive digit-sentinel failures deselected; inherited Ruff/formatter debt remains documented. Actual server APIs cannot honor coherent snapshot reads or expected-version writes, so effective capabilities/refusal are explicit; no live server support claimed. No full suite, push or merge.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->

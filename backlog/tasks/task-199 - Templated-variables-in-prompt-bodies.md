---
id: TASK-199
title: Templated variables in prompt bodies
status: In Progress
assignee:
  - '@codex'
created_date: '2026-07-12 13:16'
updated_date: '2026-08-10 00:06'
labels:
  - ux
  - console
  - prompts
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred from the 2026-07-12 Library Prompts spec: compiled System/User prompt lanes may contain `{placeholders}` (for example `{date}`) filled at insert time. The Console and Library entry paths need one shared fill-in UI, literal-brace escaping, and stale-state guards.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A single shared parser recognizes case-sensitive `{name}` variables in active System and User lanes, preserves first-occurrence order, and collects one value that is reused at every occurrence without reparsing braces introduced by that value.
- [ ] #2 `{{` and `}}` express literal braces; invalid, unmatched, JSON/XML, adjacent, nested-looking, and triple-brace inputs follow the deterministic ADR-053 truth table without partial or repeated substitution.
- [ ] #3 Variable names follow `[A-Za-z_][A-Za-z0-9_]*`; a regex-valid name longer than 64 characters or a first occurrence beyond 64 unique valid variables produces an explicit validation state rather than truncation or literal fallback.
- [ ] #4 One shared Prompt Variables dialog serves exact `/prompt`, Console picker, and Library `Use in Console` flows; it shows each active variable once with lane usage, permits blank values, scrolls to 64 variables, and renders names/content literally.
- [ ] #5 A System lane is separately authorized by the exact checkbox copy in ADR-053, is off by default, recomputes active variables without losing mounted ephemeral values, and cannot be applied as an implicit or no-op side effect.
- [ ] #6 Apply renders selected lanes, `Use original placeholders` applies selected lanes unchanged, and Cancel mutates nothing; both application actions are disabled when no lane is active while Cancel remains available.
- [ ] #7 `/prompt` and picker capture the complete segment-aware composer snapshot at dispatch or picker opening before asynchronous work and replace all its draft segments only while the live snapshot still matches; Library uses only the app-owned sanitized target projection, refuses with bounded recovery when no prior Console target exists, and otherwise appends to the settled active draft captured at consumption. Stale composer, session, or System fingerprints apply nothing and report a bounded warning.
- [ ] #8 The memory-only typed application handoff is detached, owner-thread-only, latest-wins, one-shot, and expires at monotonic elapsed time greater than or equal to 120 seconds; an expired claim remains visible for one bounded warning and acknowledgement but is never requeued, while transient missing-composer retries are allowed only for a still-valid ready claim.
- [ ] #9 In-memory composer and authorized System changes are coordinated and reversible, while any later durable System persistence failure is reported honestly as a separate outcome rather than described as an atomic disk rollback.
- [ ] #10 Raw variable maps, individual values, source Prompt bodies, and rendered lane bodies are absent from representations, logs, and persisted defaults; log and refusal paths expose bounded metadata only.
- [ ] #11 Prompts with no recognized variables and no System lane retain the direct safe insertion path, System-only Prompts still require authorization, and Recipes remain non-executable until explicitly converted to an unsaved Prompt copy.
- [ ] #12 User documentation covers grammar and escaping examples, limits, blank values, System authorization, original-placeholder behavior, destination semantics, expiry, and non-persistence; automated and real-compositor verification covers all three entry paths at narrow and normal sizes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record ADR-053 for the shared variable grammar, privacy-safe application request, and guarded insertion semantics.
2. Implement the pure lexer/rendering and typed application contract with RED-first tests.
3. Build one shared Textual dialog for slash, picker, and Library destinations.
4. Extend the owner-thread pending handoff with typed expiry-safe prompt applications.
5. Integrate exact Console replacement and Library append/System opt-in flows.
6. Update guides, run focused/full verification and visual QA, complete independent review, then open and merge PR 4/6.

ADR required: yes
ADR path: backlog/decisions/053-prompt-variable-grammar-and-guarded-insertion.md
Reason: this establishes a durable cross-module grammar, privacy boundary, and guarded Console mutation contract.
<!-- SECTION:PLAN:END -->

---
id: TASK-32508
title: 'Preset fallback_models: user-configured fallback chains for routed sub-agents'
status: Done
assignee:
  - '@codex'
created_date: '2026-09-12 01:06'
updated_date: '2026-10-02 01:08'
labels:
  - agents
  - console
  - llm-routing
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up to TASK-32477 (agent provider routing, ADR-147). Let an AgentDefinition preset carry an ordered fallback_models list (provider/model entries) used when the primary target fails with a retryable provider error (rate-limit, overload, unavailable model, provider-reported timeout) BEFORE any tool activity. The chain is user-authored, so it does not violate the no-silent-fallback rule of ADR-147. Must define interaction with run budgeting, provider continuation, fleet admission, and the resolved-target snapshot; reference pi-subagents docs/models.md fallbackModels semantics as a starting point. Explicitly out of scope there: mid-run fallback after tool activity.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Preset schema gains validated fallback_models list
- [x] #2 Fallback triggers only on retryable pre-tool-activity provider failures
- [x] #3 Budget/continuation/snapshot interactions specified and tested
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/200-preset-pre-tool-fallback-targets.md (amends ADR-147; reuses ADR-110)
Reason: freezes explicit fallback routing and continuation state across storage/runtime boundaries.
1. Validate and persist preset fallback pairs; freeze run candidates and active index in schema v22 with exact recovery declarations.
2. Reuse FallbackRuntime for typed retryable failures before any proposed tool activity; rebuild each target call and persist switch before dispatch.
3. Edit explicit fallback pairs through the existing Settings preset form.
4. Verify targeted authoring, migration/recovery, adapter dispatch, tool boundary, continuation, cancellation and budget checks.

CI addendum 2026-10-01: preserve both multiline editors with a scoped shared token-backed class, regenerate the bundle, verify mounted paint at 120/70 columns and qualify the existing boot byte/token/bundle guards without changing any pin. ADR required: no; existing ADR-097/150/200 governs this mechanical consolidation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented ADR-200 on the existing fallback runtime: presets author up to eight validated provider/model pairs, with an empty list off. Settings edits the same stored pairs. Spawn freezes each candidate's own registry or effective configured/default URL, execution key (including custom engine selection), and resolved sampling parameters before admission. Active selection is persisted before dispatch; original resolved_* remains audit identity. The coordinator and optional Console observer publish the selected target after persistence. A retained child resumes its frozen active target without opening another preset chain.

Typed pre-tool policy covers existing transient failures and explicit model_not_found machine codes through owned HTTP mapping. Auth, arbitrary 400/404, configuration errors and diagnostic text remain terminal. Proposed/refused tool batches close this policy. Candidate closures reset model/provider/URL/params/reasoning/private continuation and protocol cache, preserving existing fleet slot/deadline/model/automatic-work ownership. Known provider error types survive the gateway's sanitized queue. The existing custom-hosted adapter receives a base URL rather than an already materialized /chat/completions route. Run context observation is once per reserved trace pair across candidate closures.

Storage landed as AgentRunsDB v22 with independently frozen fresh/migrated catalogs and the linear 21→22 recovery migration; the concurrent progress task extends the database to v23. Focused feature checks cover real HTTP adapter dispatch, same-provider alternate models, family edits, authoring, Settings round trip, selection-write failure, cancellation, model caps, conservative automatic accounting, refused tools and retained continuation. Targeted UI token and CSS bundle checks pass. Verification: all 45 focused fallback cases pass, including the independently frozen v21→current migration, 13 UI token/CSS bundle checks pass, and the post-review feature/routing/gateway/bridge selection passes 85 tests in 28.12s. Two existing profile-reading regression bodies now use supported per-node bootstrap markers; no external test plugin is needed. Exact v22 recovery qualification previously passed 17 targeted cases; the progress agent preserved those frozen declarations while extending the linear migration to v23. No full suite or native terminal UI qualification was run. Scoped new-file Ruff/formatter and fatal changed-file checks pass; unrelated baseline lint debt was not reformatted.

Independent review corrections: deleted raw registry targets now refuse before family credential lookup or readiness probes; a real gateway RED had emitted /health after deletion. Built-in candidate URLs are frozen through shared configured/default endpoint semantics; real HTTP fallback and persisted continuation keep the original URL after config edits. Primary snapshots preserve an exact matching parent's owned selected endpoint. Resumed fleet handles seed the active frozen config for display while DB audit columns remain original; the actual retained-child finish observer now sees the alternate on both segments. All three review defects were reproduced before fixing. The user guide includes bounded provider/model authoring, trigger/tool/budget rules, saved active-target continuation and current routed credential ownership. ADR-200 was clarified without changing policy.

ADR path: backlog/decisions/200-preset-pre-tool-fallback-targets.md. Added the observed same-wire/different-family incident to lessons-testing-evidence. Left In Progress for independent integration review.

Final disposition 2026-09-29: All three independent findings are fixed with RED→GREEN tests: deleted registry ownership, effective built-in endpoint freezing, and active resumed display. Author affected selection passed 85 cases; independent fallback/painted-target qualification passed 50 cases in 19.90s. Root primary-URL handoff selection passed nine cases. The formerly plugin-marked legacy gateway/bridge nodes now pass with committed per-node bootstrap_profile markers; the unchanged messaging/legacy selection passed 22 cases. No temporary plugin is needed. All acceptance criteria are checked; scoped tests, changed-code static checks, documentation and independent review are complete. Task is Done. Inherited source formatting debt is preserved; no full-suite/live-provider result is claimed. This disposition supersedes earlier pending-review notes.

October 1 second CI run exposed boot CSS 608118 B against 608090 B. Current dev bundle is 30 B smaller, leaving only 2 B of original headroom. Reopened for the existing Settings authoring acceptance criteria: replace duplicate textarea ID selectors with one scoped shared class, preserve exact token-backed geometry, rebuild from sources and verify actual mounted fields plus the unchanged byte ratchet. Existing ADR-097/150/200 applies; no new architectural decision.

October 1 final CI/Qodo follow-up: replaced the duplicate instruction/fallback textarea ID selectors with one scoped agents-area class applied only to those two fields; parameters remain unchanged. Regenerated CSS from sources, removing exactly 30 parsed bytes: 608088/608090 B, original pin unchanged. The existing production-CSS test paints actual fallback content at 120/70 columns. Exact budget, both mounted widths, fallback authoring, bundle and design-token selection passes 17 cases in 57.55s; independent exact budget/mounted rerun passes 3 cases. Completed ChatModelUnavailableError constructor annotations and model_unavailable_error optional return/Google docs; classification behavior unchanged. Peer/inventory/fallback selection passes 88 cases in 49.27s; independent real HTTP machine-code classification passes 6. Existing ADR-097/150/200; no new decision. Final 76-file changed-code static and diagnostic guards pass. Done; fresh remote CI remains a publication check.
<!-- SECTION:NOTES:END -->

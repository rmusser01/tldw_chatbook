---
id: TASK-32189
title: Add guided Web Search setup to F9 Settings
status: Done
assignee:
  - '@codex'
created_date: '2026-09-09 22:31'
updated_date: '2026-09-10 01:48'
labels: []
dependencies:
  - TASK-32188
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
First-time and experienced users need a discoverable search setup surface that separates the shared default from editing other providers and preserves their work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Web Search is discoverable from Core and settings search by provider names and common setup terms.
- [x] #2 Users can choose the shared default and configure all supported backend fields without raw TOML or accidental changes to the default.
- [x] #3 Staged edits, including masked credential replacements and clears, survive provider and category navigation; Save and Revert are explicit and truthful.
- [x] #4 Offline readiness and an explicit saved-settings search test show bounded secret-safe results and remain accurate after edits or navigation.
- [x] #5 Keyboard access and compact terminal layouts are verified alongside persistence, failure recovery, and independent design review.
- [x] #6 The shared backend metadata contracts and every public WebSearchSettings API document their fields, arguments, return values and draft/save/test state effects.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Extend the incumbent Settings layout with a Core Web Search category and provider-name search aliases.
2. Compose a modular staged editor: shared default, separate backend-to-configure selector, provider-specific masked fields and source/readiness copy, explicit Save/Revert and saved-settings test.
3. Keep draft ownership at the Settings screen lifetime; preserve changes across provider/category navigation and guard asynchronous save/test results against stale views.
4. Write focused persistence, navigation, keyboard, validation and worker-state tests. Inspect compact and wide rendered layouts in one batch and resolve material findings.
5. Run targeted checks and independent code/design reviews; update the user guide and review ledger.

ADR required: yes
ADR path: backlog/decisions/012-provider-credential-settings-boundary.md
Reason: guided credential and test boundaries extend the existing ADR; save behavior follows backlog/decisions/033-settings-commit-models-three-honestly-labeled.md.

Dependency: TASK-32188 supplies the shared backend catalog and current request resolution. Existing shared-default behavior is TASK-32193.

6. PR #2562 Qodo follow-up: document FieldSpec, BackendSpec, ProbeResult and every public WebSearchSettings API in Google style; verify formatting and the existing guided/search regression selection. Documentation only; existing ADR-012 and ADR-033 remain unchanged.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added Core Web Search with searchable provider terms, separate default/editor selectors, and the shared ten-backend field catalog. A small screen-lifetime controller owns staged values and async state; the Settings screen only wires category registration, Save/Revert and rendering. Saved credentials remain masked, blank replacement keeps a key, and Clear removes canonical and legacy local entries. Atomic saves preserve unrelated configuration and refuse concurrent edits to the same fields.

The explicit Test saved settings action sends the displayed sample query through real backend dispatch, using closed diagnostics and no synthesis. Local setup status stays separate from connection evidence. Dirty drafts disable tests; editing/navigation invalidates results; background completion cannot repaint another category. Compact layouts include Save/Revert and readable clean/saving explanations. Revert uses the existing confirmation dialog.

ADR: extended backlog/decisions/012-provider-credential-settings-boundary.md; follows ADR-033 staged commits and ADR-032 query/default boundaries. The earlier draft-test idea was refined to configure → Save → Test saved settings so the check exercises the same persisted configuration as ordinary searches. Incomplete configurations may be saved with setup blockers shown; no silent fallback. Existing Settings visual identity is retained.

Validation: combined targeted gate passed 448 tests, with three opt-in live-provider tests skipped. After final visual-review copy changes, all 12 Web Search UI/model tests passed again. Production-CSS mounted captures cover 120x35 and 80x24, at entry and test controls. Independent code review resolved all four findings; visual verdict resolved its two material fixes with disposition ship for those fixes. New-file Ruff/format, scoped static checks, diagnostic inventory and git diff --check pass. No full sweep or real provider requests.

Documentation: Docs/User_Guide/settings.md and console/agent-runs-and-tools.md describe first-time/power-user setup, environment precedence, lifecycle restrictions and exact save/test semantics. .impeccable/review/web-search contains compositor SVG/PNG evidence; .impeccable/surfaces/settings-web-search.md records the local surface contract. Added the stale-worker/shared-banner incident to backlog/docs/lessons-textual.md. Advanced Config raw draft preservation remains outside this guided-editor task. Implementation is included in PR #2562 against dev.

PR integration: moved onto dev 86a8054edb, preserving current TLS, profile/footer, privacy and config publication behavior. Final evidence and baseline limitations: Docs/superpowers/reviews/2026-09-09-search-settings-pr-integration.md (525 targeted cases passed across two runs, three live cases skipped).

Qodo review follow-up for PR #2562: added Google-style field documentation for FieldSpec, BackendSpec and ProbeResult, plus summaries and applicable arguments/results/state effects for all 17 public WebSearchSettings APIs and its constructor. AST inspection confirmed every public API has a docstring. Related guided/search/config regression gate passed 201 cases, with full Ruff and formatting on the modified modules. No search or guided-editor behavior changed. Existing ADR-012 and ADR-033 apply; independent review reported no remaining findings.
<!-- SECTION:NOTES:END -->

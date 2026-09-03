---
id: TASK-29229
title: Add canonical Tool Profiles management and first-bind review
status: Done
assignee:
  - '@codex'
created_date: '2026-09-02 20:12'
updated_date: '2026-09-02 21:30'
labels:
  - tool-packs
  - mcp
  - settings
  - ui
dependencies:
  - TASK-29228
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give users one canonical Settings surface to inspect, import, export, edit, remove, and safely bind local Tool policy profiles without conflating profile data with tool installation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Settings contains a canonical Tool Profiles panel listing default, valid local, imported, and workspace-managed profiles with truthful origin, lifecycle, provenance, reference counts, and disabled states; tombstones remain hidden.
- [x] #2 The panel is presentation-only and emits explicit import, export, edit-policy, bind, and remove requests; policy editing deep-links MCP Permissions to the exact captured profile.
- [x] #3 Import review shows identity, digests, source fallback, policy-difference counts, destination connectivity and mappings, unbound behavior, and that no tools are installed; only an explicit unbound-profile confirmation may commit.
- [x] #4 Import, export, and removal are worker-backed, context-captured, stale-safe, cancellable, and surface stable categorized outcomes; Windows publication_unsupported remains a separate truthful result.
- [x] #5 First bind reviews current policy and binding state, requires an exact confirmation token plus independent memory acknowledgement, invalidates stale review state, and saves through the registry without offering import-and-bind.
- [x] #6 Tool Profiles is searchable and navigable in canonical Settings, deprecated settings surfaces remain untouched, controls follow reserved-key and truthful-footer conventions, and supported narrow layouts remain usable.
- [x] #7 Focused Settings Tool Profiles, workspace defaults, category/search, narrow-layout, footer, service, and review tests plus scoped static/diff checks pass with no unresolved Critical or Important review findings.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add Tool Profiles to SettingsCategoryId, category summaries/groups, ownership/help/search contracts, and the canonical detail-pane branch; leave deprecated settings surfaces unchanged.
2. Write failing pure widget tests for immutable profile rows, truthful origin/lifecycle/provenance/reference/removal states, hidden tombstones, disabled actions, explicit messages, and no embedded policy editor.
3. Implement the modular ToolProfilesPanel as a presentation-only Textual widget with stable focusable actions, bounded plain-text detail, empty/unavailable states, and exact captured profile events.
4. Write failing import-review tests for producer/content identity, proposed id, fallback/posture and exact/changed/missing/pending/omitted counts, destination connectivity/mappings, unbound/no-install disclosure, review invalidation, and the single Import unbound profile commit action.
5. Implement the modular import review and first-bind modals, preserving untrusted strings as plain text and invalidating stale review state on profile, mapping, workspace, defaults, timeout, or failed save changes.
6. Wire Settings-owned exclusive workers for profile listing, import inspection/activation, export capture/publication, and removal with exact captured source/destination/revision identities, overwrite confirmation, cancellation, stable categorized errors, and truthful publication_unsupported handling.
7. Deep-link Edit permissions into MCP Permissions with the exact profile, and route workspace profile assignment through current review, confirmation-token exchange, independent read-write memory acknowledgement, and exact registry retry; never offer import-and-bind.
8. Add navigation, category sweep, search, narrow-layout, keyboard/focus, disabled-state, and truthful-footer coverage using the canonical Settings patterns and app-owned ToolPackService.
9. Run the prescribed focused Settings matrix plus ToolPack service and binding regressions, scoped Ruff/changed-range format/diff checks, Impeccable detector, self-review, and independent review before closeout.

ADR required: no new ADR
ADR path: backlog/decisions/107-portable-tool-use-packs.md
Reason: ADR-107 already defines the canonical Settings management surface, MCP-only policy editing, review-first unbound import, exact first-bind token boundary, removal semantics, stable failures, and the separate Windows publication claim.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the canonical Settings Tool Profiles category and modular presentation/review widgets. The panel lists current profile lifecycle, binding, provenance, posture, policy identity, and removal state; emits exact-context management requests; refreshes after returning from MCP; and keeps deprecated Settings surfaces unchanged.

Import, export, and removal now run through reviewed worker flows with captured identities, unbound-only activation, safe destination publication, cancellation checks, stable path-free outcomes, and a distinct Windows `publication_unsupported` result. MCP Permissions accepts only bounded deep links carrying the exact current policy digest. Workspace apply routes imported profiles through the current first-bind review/token exchange while retaining the separate `read_write` acknowledgement and preserving newer staged state across slow completions.

Review hardening added local-profile policy digests, stale export fencing, plain-text untrusted identifiers, private-error redaction, exact resume refresh, same-workspace completion-race protection, and 80-column action containment. ADR-107 remains the governing decision; no new ADR or schema change was required. User/security/performance documentation and whole-feature closeout remain in the existing Task 15 plan.

Verification: 69 focused Settings tests passed; 332 Tool Pack service, binding, and MCP Workbench tests passed; the exact deep-link follow-up passed 3 focused tests; scoped Ruff, new-file format, `git diff --check`, and the Impeccable detector passed. The repository-wide suite was not run, per project policy.
<!-- SECTION:NOTES:END -->

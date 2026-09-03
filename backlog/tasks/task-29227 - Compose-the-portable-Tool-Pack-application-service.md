---
id: TASK-29227
title: Compose the portable Tool Pack application service
status: Done
assignee:
  - '@codex'
created_date: '2026-09-02 14:00'
updated_date: '2026-09-03 02:20'
labels:
  - tool-packs
  - app-wiring
  - startup
  - receipts
dependencies:
  - TASK-29226
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expose the completed portable Tool Pack lifecycle behind one deferred app-owned service, attach its binding authority exactly once, and reconcile receipt storage without expanding startup-critical imports or leaking sensitive failure details.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One app-facing `ToolPackService` exposes separate export capture/publication, import inspection/unbound activation, first-bind review/confirmation, removal, and immutable non-tombstoned profile-listing operations without becoming a direct policy-rule editor.
- [x] #2 Profile listings report exact id, origin, lifecycle validity, binding state, active/archived references, compact policy counts, receipt health, and removal eligibility from current strict authority and workspace state; tombstones stay hidden while their ids remain reserved.
- [x] #3 Service composition shares one lifecycle coordinator across activation, binding, and removal; uses the sealed V1 inventory assembly and user-data receipt root; and exposes only stable bounded error categories without paths, credentials, commands, environment values, or archive/receipt excerpts.
- [x] #4 `app.py` composes exactly one complete service only after MCP and workspace prerequisites exist, attaches its binding guard exactly once, exposes a stable unavailable state on failure, and never attaches a partial guard or eagerly imports Tool Pack implementation modules on the startup-critical path.
- [x] #5 Receipt reconciliation runs only after UI readiness/first use, protects every authoritative referenced and live-owned receipt, honors orphan grace and authenticated regular-file checks, and remains bounded/fail-safe for fresh, symlinked, unknown, and corrupt entries.
- [x] #6 Focused service, startup-hygiene, and startup-performance tests plus scoped Ruff/format and diff checks pass; independent review has no unresolved Critical or Important findings.
- [x] #7 Public facade methods document their arguments, results, and stable errors; failed deferred composition can be retried on demand from Settings while preserving single-flight attachment.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing facade tests for split review/commit operations, immutable presentation listings, stable error privacy, and no rule-edit mutation surface.
2. Compose the existing export, publication, import, activation, binding, removal, inventory, and receipt owners behind one shared-lifecycle `ToolPackService`.
3. Add focused app-wiring tests for prerequisite ordering, exact guard attachment, unavailable/partial-failure behavior, user-data receipt placement, and deferred imports.
4. Add bounded post-ready receipt reconciliation using current profile receipt links plus in-flight activation ownership; prove only eligible expired orphans are reclaimed.
5. Run the prescribed targeted suites, scoped static/diff checks, self-review, and independent review before closeout.
6. Remediate independent-review blockers at their owning seams: add the workspace registry's fail-closed deferred guard, share export's runtime-floor projection with listings, and give receipt reconciliation an entry budget. This expands the initial six-file boundary only because app-only wrappers cannot close direct registry writes or bound the store-owned traversal.
7. PR review follow-up: add startup retry and facade-documentation regressions, reset the single-flight gate only after failure, trigger retry from Settings demand, and rerun focused startup/service tests.
8. CI performance follow-up: prove the global deferred-startup scheduler does not queue Tool Pack composition, move composition to first Tool Profiles use, and rerun the exact UI-latency guard plus focused startup/Settings/Tool Pack checks.

ADR required: no new ADR
ADR path: backlog/decisions/107-portable-tool-use-packs.md
Reason: ADR-107 already fixes the app/service ownership boundary, lazy composition, receipt authority and reconciliation, stable privacy-safe outcomes, and Settings-versus-MCP editing split.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the app-facing `ToolPackService` and immutable profile presentation over the existing export, publication, import, activation, binding, removal, inventory, and receipt owners. Composition shares one lifecycle coordinator and workspace-reference adapter, keeps Tool Pack implementation imports off the startup-critical path, and places receipts under the user-data root.

Independent review exposed five blockers. The final implementation installs a cheap fail-closed workspace guard during registry wiring, atomically activates the complete captured-registry guard before receipt recovery, unwraps the export review's immutable snapshot for publication, shares export's raw-shell Ask-floor projection with profile listings, and caps receipt reconciliation at 4,096 entries before any deletion. The original six-file boundary was expanded only to the registry, export projection, and receipt-store owners required for those fixes.

Focused controller verification passed 139 tests; the independent re-review passed 118 tests and reported no remaining findings. Scoped Ruff and diff checks passed. The only warning was the known environment `RequestsDependencyWarning`; `export.py` retains its exact pre-existing formatter baseline with no new formatting delta. ADR-107 remains controlling, no new ADR was required, and the work produced no new reusable lesson.

PR #2324 Qodo follow-up documents every split public facade operation with
Google-style arguments, results, and stable error behavior. Failed worker starts,
missing prerequisites, and composition failures now release the single-flight
gate; entering Tool Profiles retries composition on demand without replacing an
attached service. Final verification passed 399 Tool Pack tests and 49 focused
startup/Settings tests, with scoped Ruff/format and diff checks green.

The CI performance follow-up removed Tool Pack composition from the global
deferred-startup fleet and made Tool Profiles first use await the one shared
composition worker before refreshing its listing. Store-owned mutation result,
error, and policy-digest primitives now live in `MCP.permission_store`, removing
the core MCP-to-Tool-Pack import edge; the UI-ready guard explicitly keeps the
entire Tool Pack package absent and measures 970/972 resident modules. Final
focused verification passed 532 Tool Pack, MCP profile-authority,
startup/Settings, and exact UI-latency tests. Scoped Ruff, format, CSS bundle,
and diff checks passed. ADR-107 and ADR-097 remain controlling; no new ADR or
generalized lesson was required.
<!-- SECTION:NOTES:END -->

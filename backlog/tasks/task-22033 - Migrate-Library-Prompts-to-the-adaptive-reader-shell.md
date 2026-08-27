---
id: TASK-22033
title: Migrate Library Prompts to the adaptive reader shell
status: Done
assignee:
  - '@codex'
created_date: '2026-08-24 23:26'
updated_date: '2026-08-26 20:44'
labels:
  - library
  - ui
dependencies:
  - TASK-22032
references:
  - >-
    Docs/superpowers/specs/2026-08-24-library-destinations-adaptive-reader-design.md
  - backlog/decisions/086-library-adaptive-reader-shell.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move Prompts into the shared Library adaptive reader structure while preserving browse paging collections import history provenance validation optimistic updates and lifecycle behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Prompts list remains mounted beside a permanent work pane with independent list collapse and destination-specific geometry
- [x] #2 Basic is the default mode and Basic Advanced and Info operate on one lossless item-owned draft
- [x] #3 Saving from Basic preserves every Advanced-only field and validation can focus the owning mode
- [x] #4 Create import history collections provenance lifecycle and destructive actions remain reachable without unmounting the list
- [x] #5 Selection loading draft navigation stale workers conflicts deletion and retry follow the approved identity and recovery contracts
- [x] #6 Existing Prompt capability and backend ownership remain unchanged
- [x] #7 Automated browse editor hidden-field history geometry focus and capability tests pass with a representative live TUI walkthrough
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inventory Prompt capabilities and draft authority
2. Add one lossless reader projection
3. Split persistent list and work pane
4. Verify hidden fields, workflows, geometry, and focus

ADR required: yes
ADR path: backlog/decisions/086-library-adaptive-reader-shell.md
Reason: consumes the accepted Library structural boundary without changing Prompt authority.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Migrated Library Prompts into the shared adaptive reader with retained Items and Work panes, independent collapse geometry, and Basic as the default projection. Added one screen-owned lossless Prompt draft shared by Basic, Advanced, and Info; explicit validation ownership; selected-versus-loaded detail fencing; truthful read-only recovery; retained browse, bulk, import, history, collection, conflict, lifecycle, and delete/Undo authority without adding a parallel persistence path. Hardened editor-origin import and browse/detail retry behavior after independent review. Verified after rebasing onto origin/dev with 15 focused retained-reader tests, 340 Prompt canvas tests, 1,126 broader Prompt state/controller/widget/service tests, Ruff, compileall, diff checks, and the complete isolated production-CSS live matrix. Full-repository collection remains independently blocked on the existing unregistered filterwarnings marker baseline in Tests/Agents/test_mcp_tool_provider.py. ADR required: yes. ADR path: backlog/decisions/086-library-adaptive-reader-shell.md. Reason: directly implements the accepted long-lived Library adaptive-reader boundary; no new ADR was required.

Post-Qodo hardening: removed the unused test-only PromptReaderState module so LibraryScreen remains the sole mutable reader authority; transferred hidden-field preservation and validation-focus coverage to the mounted production reader; fixed Info fallback so only unavailable Basic routes to Advanced; routed invalid outer saves to the owning Advanced block control; and confined config, data, app-data, and all XDG evidence-driver paths through centralized validation. Verified with 17 mounted reader tests, 10 seam/authority tests, focused mode and isolation regressions, Ruff, compileall, diff checks, and a fresh complete isolated live matrix. ADR required: yes. ADR path: backlog/decisions/086-library-adaptive-reader-shell.md. Reason: hardens the existing accepted boundary without adding a new architectural decision.

CI derived-artifact review: inspected the exact persistent-diagnostic statement delta in tldw_chatbook/UI/Screens/library_screen.py (two new constant-message debug calls and one shortened existing info call retaining only the internal Prompt integer ID). Confirmed no new sink topology and no interpolation of user content, secrets, paths, or URLs; regenerated Docs/security/production-diagnostic-inventory.json and verified it reproduces at 538 owners, 1,241 TASK-492 calls, 7,359 TASK-494 calls, and 8 sink files.

Post-rebase verification: rebased onto origin/dev at c23113e2e00142d918bd64b56a53f755b07424ce and regenerated the production diagnostic inventory to resolve the generated-artifact conflict from source (537 owners, 1,244 TASK-492 calls, 7,362 TASK-494 calls, 8 sink files). Hardened two order-sensitive mounted-canvas tests to await actual widget/modal/notification settlement instead of fixed pauses. Verified the complete 341-test Prompt canvas file in exact order, with the remaining 314 targeted Prompt tests green in the immediately preceding run; also passed Ruff, compileall, diff checks, and diagnostic inventory reproduction. No new ADR required; this is verification hardening within ADR-086.

Final Qodo evidence hardening: split the isolated bootstrap from the mounted journey runner so configuration and path containment complete before third-party/application imports; validated every CLI journey selector through the shared input-validation boundary and a closed allow-list; always replaced scratch configs with a contained app-data path; deterministically closed every real Prompt database after the Textual host and on failures; documented every public journey; and strengthened Basic-mode preservation evidence to compare the complete structured definition rather than selected user-block fields. Added focused regressions for escaped XDG paths, malformed/unsupported selectors, hostile existing configs, and failure-path database cleanup, then regenerated the full production-CSS evidence matrix. No new ADR required; these changes harden the existing verification harness within ADR-086.

Latest-dev integration: reconciled dev's newer ordinary-route classifier and route/resize matrices with Prompts' adaptive-reader ownership. Prompts and New Prompt now bypass ordinary two-pane emergency geometry, while the ordinary test probes use Skills and the adaptive matrices explicitly cover Prompts. Added exception context to targeted Prompt canvas/work-pane recovery diagnostics. Verified the integration with 48 focused shared-shell and Prompt tests plus a fresh complete isolated production-CSS live matrix. No new ADR required; this is a rebase integration correction within ADR-086.
<!-- SECTION:NOTES:END -->

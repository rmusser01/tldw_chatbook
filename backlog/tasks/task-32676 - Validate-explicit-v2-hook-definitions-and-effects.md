---
id: TASK-32676
title: Validate explicit v2 hook definitions and effects
status: Done
assignee:
  - '@codex'
created_date: '2026-09-16 04:22'
updated_date: '2026-09-30 23:32'
labels:
  - plugins
  - implementation
  - hooks
dependencies:
  - TASK-32645
documentation:
  - Docs/superpowers/plans/2026-09-15-expanded-hooks.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make richer hook declarations inspectable and deterministic while preserving the legacy six-event configuration.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Legacy hooks.hook parsing and stdout behavior remain unchanged; hooks.handler and native v2 files validate the exact event/effect/matcher/template contracts.
- [x] #2 Required success, required nonempty context and dependency-scoped requirements stay distinct, and invalid controlling requirements never disappear through optional parsing.
- [x] #3 PreToolUse phase classification is exhaustive, including mixed transformer/deny, final guards, context-only handlers and effect-free required completion.
- [x] #4 Payload and result bounds reject whole invalid batches, reserved ownership fields cannot be supplied by output, and malformed teardown/approval/Stop requirements are refused.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/163-expanded-console-hook-runtime.md; backlog/decisions/197-console-hook-configuration-review.md; backlog/decisions/148-console-run-hooks.md
Reason: Integrate the existing accepted H1 implementation on current dev without changing runtime or consent ownership. The original reviewed implementation is preserved on codex/managed-plugins at cdeb1687b6.
1. Read TASK-32676, the expanded-hooks plan/spec, existing ADRs and the reviewed H1 implementation. Locate current loader and all consumers; preserve lossless legacy inventory, row switches and exact-definition consent.
2. Qualify current-dev baseline. Repair the existing real-config regression harness to use the already-bound private bootstrap profile rather than switching a live source. Record the observed baseline failure and a passing control.
3. Establish public-loader behavioral RED for valid v2 declarations, disabled required definitions and malformed required batches before integrating production code. Reuse the reviewed H1 schema/matcher/result tests and add current legacy-consent compatibility controls.
4. Reuse the reviewed H1 immutable models, bounded schema/raw decoders, matching/template and invalid-admission metadata. Extend the current inventory-backed loader with independent lazy v2 projection; do not replace current legacy validation or implement H2 execution here.
5. Carry the H1 contract clarifications into the existing spec/ADR. Keep default startup lazy, unknown config fields lossless, required/invalid state across master switch saves, and exact legacy grants unchanged.
6. Run the exact H1 schema/legacy/config files plus changed hook consent and Settings persistence neighbors. Check syntax, Ruff/formatter on new and authored ranges, whitespace and document links. Use targeted runs only; record actual limitations and reviewed source provenance.
7. Self-review the complete integration, check every AC, add current verification notes, mark Done via Backlog CLI and commit TASK-32676 independently before proceeding to the next task. Preserve the original implementation branch and unrelated checkouts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Integrated the reviewed H1 implementation from codex/managed-plugins (H1 checkpoint 7e99298cf2) onto current dev at 10b34ffdd5. Explicit v2 definitions, immutable models, structured matching, typed templates and bounded strict result/native-file decoding remain independent of Plugins. The current inventory-backed legacy loader keeps row enablement and exact-definition consent; v2 parsing is lazy and retains whole-batch failure scope without activating a valid-looking subset.

Reused the existing reviewed schema and loader regression tests. Real config/save tests now use the already-bound private bootstrap profile through the existing hook_file fixture, preserving recovery admission rather than switching a live config source. Baseline: one existing config test failed with raw_source_selection_changed and 81 tests passed. Public-loader RED: two expected missing-v2 assertion failures plus two passing controls. Integration qualification: 302 cases passed across the exact schema/legacy/config, consent, inventory and canonical Settings files in 50.69s; after fixture import cleanup the exact six metadata cases passed again. Counts overlap and are not aggregated. Eight authored Python files pass full Ruff/format; nine files parse; shared config.py retains the same 168 pre-existing Ruff diagnostics; whitespace passes. No new skips, full suite, provider generation, or cross-platform execution claim.

Existing ADR-163 and ADR-148 govern validation and legacy behavior; ADR-197 remains the consent owner. Applied only reviewed H1 contract clarifications to the existing spec/ADR and completed H1 plan tracking. Production v2 command/MCP execution and new event producers remain the following tasks. Current integration evidence is in Docs/superpowers/reviews/2026-09-30-expanded-hooks-integration.md. No new schema, dependency or permission owner.
<!-- SECTION:NOTES:END -->

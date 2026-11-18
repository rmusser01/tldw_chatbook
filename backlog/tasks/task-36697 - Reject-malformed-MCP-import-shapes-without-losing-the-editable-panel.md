---
id: TASK-36697
title: Reject malformed MCP import shapes without losing the editable panel
status: Done
created_date: 2026-10-24 17:34
labels:
- textual-patterns
- reliability
priority: medium
documentation:
- Docs/superpowers/qa/2026-10-24-textual-book-after-pr3228.md
assignee:
- '@codex'
updated_date: 2026-10-24 21:44
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The MCP JSON importer assumes args is iterable and env is a mapping. Valid JSON with malformed field containers escapes Preview validation, causing the error keepalive to retire the import panel and its editable draft.

### Evidence and scope

Verified after PR #3228 merged, against dev `96cae7eec00473bbfb5687ccfd0685eba2785add`. Real Preview of args:42 raises TypeError; env:[not-an-object] raises AttributeError. Valid args:[]/env:{} control passes. Installed keepalive guard keeps the host alive while the failing panel/editor detach. No verified whole-app production crash claim.

Book basis: Collateral concrete input-boundary defect found while tracing the book’s queued Preview/panel lifecycle model. Container validation is an existing application requirement, rather than a specific book design.

Affected owners: [mcp_import.py](../../tldw_chatbook/MCP/mcp_import.py), [mcp_profile_form.py](../../tldw_chatbook/UI/MCP_Modules/mcp_profile_form.py).

### Existing ownership and decisions

TASK-241 remains open for regex alignment, mutation remount, slot confirmation and lifecycle cancellation; its complete criteria do not cover malformed containers. TASK-26013 validates spawn-command shapes, TASK-14922 protocol/packaging. TASK-32533 and TASK-33621.13 keep handler errors alive while retiring failing pumps; they do not validate these JSON shapes.

Related tasks: [TASK-241](../tasks/task-241%20-%20MCP-Hub-engineering-follow-ups-from-Phase-2-3-ledgers.md), [TASK-26013](../tasks/task-26013%20-%20MCP-server-spawn-configuration-guard.md), [TASK-14922](../tasks/task-14922%20-%20Resolve-PR-1490-post-merge-review-findings.md), [TASK-32533](../tasks/task-32533%20-%20Library-Notes-the-app-exits-with-an-unhandled-InvalidSelectValueError-during-the-first-timer-edit-journey-and-the-crash-handler-records-no-site.md), [TASK-33621.13](../tasks/task-33621.13%20-%20Console-Choose-folder-for-project-instructions-freezes-the-whole-app-Esc-clicks-and-CtrlQ-ignored.md).

ADR required: no.
ADR paths: [161-component-pattern-library](../decisions/161-component-pattern-library.md), [240-mcp-destination-retirement-tools-hub-in-settings](../decisions/240-mcp-destination-retirement-tools-hub-in-settings.md).
Reason: Routine validation and recoverable errors within the existing importer and host; no new service or security policy.

[Post-merge book audit and evidence](../../Docs/superpowers/qa/2026-10-24-textual-book-after-pr3228.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Malformed args/env container shapes produce a clear import validation error without an uncaught handler exception.
- [x] #2 Rejected source text stays editable, Import remains disabled, and stale preview candidates cannot be submitted.
- [x] #3 Correcting the text and pressing Preview recovers normally; supported valid and omitted field shapes retain existing behavior.
- [x] #4 Parser and mounted Preview regressions cover both observed malformed shapes, a valid control and the installed keepalive semantics.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR paths: [ADR-161](../decisions/161-component-pattern-library.md) and [ADR-240](../decisions/240-mcp-destination-retirement-tools-hub-in-settings.md). Reason: malformed-container validation within the existing parser/ValueError Preview recovery contract; unchanged permission and spawn/secret policy.
1. Add parser controls for malformed args/env containers, including falsy wrong types, and valid/omitted/null supported inputs; add mounted valid-preview to rejection to correction controls under installed keepalive behavior.
2. Retain attributable parser/handler RED, then validate container shapes with field-specific ValueError before iteration. Preserve command screening, secret filtering and current supported coercion.
3. Verify rejected raw source remains editable, Import disabled and stale candidates cleared; correction/repreview recovers without retiring the panel. Preserve the reviewed file-read draft/retirement protections from TASK-36696.
4. Run new and directly relevant existing parser/spawn/Preview checks with actual private-child XML inspection and focused lint/format. No full suite or inherited-fixture repairs.
5. Obtain task-scoped Superpowers requesting-code-review, register parser/UI regressions on their existing required lanes, update architecture and implementation notes. Final hosted qualification follows all six tasks.
Master plan: Docs/superpowers/plans/2026-10-24-textual-lifecycle-and-import-followups.md (Task5).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The shared MCP JSON parser now rejects malformed non-null args/env containers with field-specific ValueError, including falsy wrong shapes. Omitted/null defaults, supported list/object item coercion, spawn screening and secret filtering retain behavior. The existing Preview ValueError path now handles these errors normally: raw text stays editable, stale candidates clear, Import disables, and correction/repreview recovers. No UI production code changed.

Verification:64 targeted parser/security/mounted cases passed, including20 new parser cases and4 real panel recovery variants. Parent and all4 private-child XMLs have zero failures/errors/skips; root and reviewer inspected actual reports. Real installed LifecycleMixin keepalive semantics were retained in a bounded native App host. Full-app exploratory RED is qualified separately; three unchanged legacy Preview bodies also passed in an untracked private-profile diagnostic. No full suite. Ruff passes; parser formatter delta is identical to inherited deferred-import wrapping debt. Dependency warnings remain attributed.

Superpowers task review approved with no blocking findings. All4 exact UI cases are registered (floor829), parser module added to PR Fast Lane and exact target contract; two focused gate checks and census validation pass. Final hosted qualification remains pending. ADR required:no; ADR-161/240 apply to existing validation/recovery ownership. Updated Docs/Architecture/mcp-hub.md. Source/tests: MCP/mcp_import.py, Tests/MCP/test_mcp_import_shapes.py and Tests/UI/test_mcp_import_validation_recovery.py.
Hosted qualification complete for PR #3243, code head087a75bab335485b2058f9a010c6f872c57ebc9b: https://github.com/rmusser01/tldw_chatbook/actions/runs/40132324399 (all25jobs success) and https://github.com/rmusser01/tldw_chatbook/actions/runs/40132324401 (Perf Guard success). Actual completed job logs prove all81 new UI selectors and20 parser cases executed, with17 successful UI shards and maximum986s, below1020s planning target/1200s cap. Pinned Canvas and all actual repository source guards pass on the hosted runner. No allowance, timeout, shard count or original census entry changed.

Superpowers requesting-code-review approved each task, the whole branch, and the single final fix re-review; M1/M2 are addressed with no remaining Critical/Important/Minor finding. All remote comment/review/thread pages were checked: no actionable finding or unresolved thread. All four acceptance criteria, implementation plan/notes, existing ADR links and relevant architecture/QA/incident documentation are complete. Native legacy/profile/Windows/static/size qualifications remain disclosed in Docs/superpowers/qa/2026-10-24-textual-followups-36693-36698.md; no full local suite or unrelated debt closure is claimed. This CLI closeout and verification-doc commit contains metadata only; its final PR head will receive fresh hosted CI before delivery.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
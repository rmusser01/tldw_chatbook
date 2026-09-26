---
id: TASK-32952
title: Expose character card creation and editing through MCP
status: Done
assignee:
  - '@codex'
created_date: '2026-09-25 16:16'
updated_date: '2026-09-26 06:41'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_chatbook/issues/2827'
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address GitHub issue #2827 by exposing existing validated local character authoring through both MCP runtimes with conflict detection and explicit write permissions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both MCP runtimes create and edit character cards through the existing service.
- [x] #2 Callers can read versions; stale updates and duplicate names fail without losing data.
- [x] #3 Invalid or unsupported fields fail explicitly; write receipts are bounded.
- [x] #4 Persistent writes honor permission floors, explicit grants, runtime policy, and the standalone kill switch.
- [x] #5 Targeted real-database, runtime, permission, and inventory tests pass; user documentation describes the contract.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/183-mcp-character-card-authoring.md
Reason: Adds public tools and defines conflict and permission behavior across two runtimes.

1. Pin missing authoring, optimistic conflict handling, and trusted write permission metadata with failing tests.
2. Add bounded text/JSON authoring delegates to LocalCharacterPersonaService, with roster versions.
3. Register both tools in each runtime and the existing runtime-policy action map.
4. Enforce code-owned mutation tags across catalog and by-key resolution; reuse the permission store for standalone dispatch.
5. Update inventory contracts and user documentation, run targeted checks, and review the resulting diff.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented both MCP authoring paths using LocalCharacterPersonaService, explicit
caller versions, roster versions, bounded receipts, and structured errors. Added
code-owned mutation tags and preserved their approval floor in both catalog and
by-key permission resolution; external clients use fresh operator grants and the
kill switch. Runtime-policy action mappings cover both operations.

ADR: [183-mcp-character-card-authoring.md](../decisions/183-mcp-character-card-authoring.md).
Updated three published inventory lists and the user guide, including their
pre-existing stale private-Library count (24 advertised versus 21 implemented).
An independent review found that encoded JSON null/scalar collection values could
report a no-op success. Four failing real-SQLite regressions demonstrated it;
normalizing with the shared schema and then enforcing array/object types closes it.

Validation: 283 targeted tests passed (including 25 authoring tests, gateway calls,
permission floors/revocation, runtime-policy denial, inventory/form contracts,
and three public protocol revisions). Evidence: /tmp/tldw-mcp-final.log and
/tmp/tldw-mcp-final.xml. The pinned optional mcp-unified dependency was installed
under /tmp only; the project environment was unchanged. New Python files pass Ruff;
all edited implementation files pass formatter checks. A HEAD comparison finds
zero added Ruff diagnostics against 111 pre-existing diagnostics in touched files.

A broader targeted run also exposed existing Watchlists documentation omissions,
a local-agent schema-count assertion (17 versus 20), and a private-Library wire
count assertion (24 versus 21). These unrelated failing checks were not weakened;
19 Watchlists/schema cases were deselected for the final scoped run. No full test
suite was run. Independent MCP review found no remaining issues after the JSON fix.

PR integration on current dev (0053a0a40f), 2026-09-25:

The PR is MCP-only. The capture and restored-dispatch fixes from the same investigation are already carried by PR #2841; character-agent tooling for #2827 is separate work. This change adds the remaining MCP entry points without duplicating those changes.

Preserved dev's newer permission-store parsing/read-failure guards and Library worker admission when resolving the port. Native recovery review exposed a further integration requirement: a raw to_thread write could outlive its waiting task without retaining the admitted MCP source set, and standalone permission grants alone could bypass invalidated recovery review. Both character write call sites now use the existing activation.in_worker helper. Six failing native cases reproduced this; eight controls/regressions now pass, including actual SQLite create/update, cancellation, timeout, full source ownership, and standalone refusal before effects. ADR-183 and the user guide describe the behavior.

Fresh verification on the isolated dev-based PR tree: 293 MCP cases passed, plus 8 native admission cases (301 total). Nine adjacent unchanged provider/Watchlists cases were separately attempted and fail in their existing per-test profile redirects with raw_source_selection_changed; excluded from the final feature run. Ten previously excluded documentation cases now pass on current dev. Relevant tests opt into the repository's existing bootstrap_profile fixture; production recovery checks remain active. No full suite or provider request ran.

Reviewed the sole added persistent diagnostic statement, logger.error("MCP character write failed."), which contains no runtime values. Regenerated the diagnostic inventory; its two already-stale summary aggregates now match its existing owner rows. Dropped unrelated formatting hunks from the port. New Python files pass Ruff/format; touched existing files add no Ruff diagnostics; scoped whitespace checks pass. Evidence: /tmp/tldw-mcp-pr-final.log, /tmp/tldw-mcp-pr-final.xml, /tmp/tldw-mcp-review-admission-red3.log, /tmp/tldw-mcp-review-admission-green.log, and /tmp/tldw-mcp-pr-adjacent.log.

Final PR preflight: all 10 derived-artifact and governance checks pass on the completed implementation (/tmp/tldw-mcp-pr-preflight-complete.log). Final self-review found no remaining issue.
<!-- SECTION:NOTES:END -->

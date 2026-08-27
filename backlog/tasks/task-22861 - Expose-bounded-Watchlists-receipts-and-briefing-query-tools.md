---
id: TASK-22861
title: Expose bounded Watchlists receipts and briefing query tools
status: Done
assignee:
  - '@codex'
created_date: '2026-08-27 04:14'
updated_date: '2026-08-27 19:26'
labels:
  - watchlists
  - console
  - mcp
  - briefings
dependencies:
  - TASK-22859
  - TASK-22860
references:
  - >-
    Docs/superpowers/specs/2026-08-26-console-driven-watchlists-workflow-uat-remediation-design.md
  - >-
    Docs/superpowers/plans/2026-08-27-watchlists-agent-boundary-and-provenance.md
  - backlog/decisions/032-local-agent-tool-permission-boundary.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let Console agents inspect sources, collections, durable operation receipts, and full briefing content with provenance while external MCP receives only explicitly approved metadata and receipts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Shared tools list bounded source, collection, briefing-receipt, and operational metadata using canonical IDs, stable order, filter-bound cursors, redacted URLs, and the established 30 KiB result ceiling.
- [x] #2 `watchlists_get_operation_status` accepts only exact `local:watchlist_run:<id>` or `local:briefing:<id>` receipts and returns their owning entity, timestamps, normalized state, retry/cancel capability, and inspection destination.
- [x] #3 `watchlists_get_briefing` is Console-only and returns bounded Markdown plus ordered selected/cited durable provenance, generated/untrusted labels, truncation metadata, and explicit legacy/missing-reference markers.
- [x] #4 External MCP can list bounded metadata/receipts but cannot receive article snippets, article bodies, briefing Markdown, or selected/cited provenance arrays, regardless of stored permission state.
- [x] #5 “Read latest briefing” deterministically resolves the newest completed receipt for one collection while reporting newer non-readable attempts as operational context.
- [x] #6 Server Watchlists mode and missing/old read-only storage return structured unsupported/feature-unavailable outcomes without initializing or mutating the local database.
- [x] #7 Service, provider, external-MCP, read-only-database, redaction, cursor, truncation, and documentation-contract tests cover every new query.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add RED database tests for source, collection, briefing receipt/provenance, operation overview/exact receipt, stable cursor ordering, newest readable completion, and explicit v2 read-only readiness.
2. Implement narrow parameterized SubscriptionsDB readers using immutable v2 provenance and stable filter-bound cursors.
3. Add bounded WatchlistsToolService list/get methods with exact arguments/canonical IDs, URL sanitation, structured outcomes, fixed briefing-body budget, bounded provenance, Unicode-safe valid JSON, legacy/missing markers, and a strict 30 KiB ceiling.
4. Register exact descriptors: shared metadata/receipt reads for Console and external MCP; article search/detail and full briefing/provenance Console-only with code-owned private-read effects.
5. Prove external persisted Allow cannot publish/resolve private reads; server mode and missing/old read-only storage remain structured unsupported/unavailable without mutation. Run complete task-targeted DB/service/provider/MCP/documentation tests, Ruff, diff checks, self-review, and independent review.

ADR required: yes
ADR path: backlog/decisions/032-local-agent-tool-permission-boundary.md
Reason: ADR-032 and its approved addendum own the synthetic principal, descriptor exposure, external read-only projection, and Console-only private briefing boundary implemented here.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented narrow v2-ready database readers and bounded Watchlists query
methods for source, collection, briefing receipt/content, and operation
metadata. Metadata lists use filter-bound keysets over fixed 96-character
casefold/raw name prefixes plus row ID, so deletion, rename, and hostile-sized
stored names cannot make a generated continuation unusable. All projections
are allowlisted, URLs are sanitized, receipt IDs are canonical, mixed timestamp
forms share `datetime(...), id` ordering, and complete rows remain below 30 KiB.
Full briefing reads reserve a fixed Markdown budget and expose independent,
followable selected/cited provenance streams with generated, untrusted,
truncated, legacy, and missing-reference labels.

The external MCP composition now constructs only descriptors classified for
external exposure, so persisted Allow cannot resolve Console-only article or
briefing-content handlers. The five metadata/receipt reads remain shared and
carry `PRIVATE_READ`; search, item detail, and full briefing content remain
Console-only. Historical-storage fixtures now use the literal v1 schema and
prove permanent non-retryable unavailability without byte mutation.

ADR required: yes
ADR path: backlog/decisions/032-local-agent-tool-permission-boundary.md
Reason: This implementation follows ADR-032's approved descriptor exposure and
external read-only projection; no additional architectural decision was made.

Three independent review rounds closed readiness-projection, mutable-anchor
cursor, provenance-continuation, cancellation-capability, timestamp-ordering,
external-copy, and hostile-name cursor findings. Fresh controller verification:
187 DB/service/read-only tests passed; 161 provider/MCP tests passed with two
optional `mcp-unified` skips; 10 documentation contracts passed; Ruff and diff
checks were clean. No live user profile or full-repository suite was used.
<!-- SECTION:NOTES:END -->

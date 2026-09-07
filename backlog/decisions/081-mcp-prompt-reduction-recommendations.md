# ADR-081: MCP Prompt-Reduction Recommendations

Status: Accepted
Date: 2026-08-01
Related Task: [backlog/tasks/task-21162 - Add-MCP-fewer-permission-prompts-recommendations.md](../tasks/task-21162%20-%20Add-MCP-fewer-permission-prompts-recommendations.md)
Supersedes: N/A

## Decision

Chatbook will add a local-only MCP prompt-reduction recommendation flow that analyzes the existing redacted MCP execution log and recommends reviewable tool-level permission changes through the existing MCP permission store APIs.

The v1 flow is MCP-only. It will not inspect shell or bash command history, will not add product telemetry or tracking, and will not introduce model-based automatic approval. The Console command `/fewer-permission-prompts` is the user-facing entry point for the report; persistent permission changes continue to go through `UnifiedMCPControlPlaneService.set_tool_state(..., "allow", tool=tool)` so the stored definition hash, rug-pull downgrade, and high-risk floor behavior remain authoritative.

## Context

Chatbook already owns a local MCP approval path for agent tool calls. Approved, allowed, denied, timeout, and downgrade decisions are written to a bounded local JSONL execution log. Persistent permission state lives in `mcp_permissions.json`, with precedence from tool override to server default to global default.

Users can already choose "Always allow" during a live approval prompt, but repeated "Approve once" or "Approve for session" decisions are hard to discover after the fact. A recommendation report can reduce repeated prompts without weakening the existing human review boundary.

The feature also needs to preserve Chatbook's local-first privacy posture. The existing MCP execution log is already local, bounded, and redacted. Recommendations can be derived from server key, tool name, decision, timestamp, and effective permission state; they do not require raw arguments, remote analytics, or telemetry.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Include bash and shell commands in v1 | Chatbook's durable approval evidence is currently MCP tool-oriented, not a general shell-command approval log. Adding shell allowlisting would require a separate audited executor and threat model. |
| Add model-based Auto Mode in v1 | Automatic approval changes the security boundary from human review to classifier judgment. That requires separate policy, failure-mode analysis, and user controls. |
| Recommend hand-edited JSON snippets | Manual edits can omit definition hashes, fight store normalization, or bypass future store migrations. Existing typed APIs already preserve the safety metadata. |
| Expand execution-log retention before recommendations | Longer retention may be useful later, but v1 can honestly operate over the bounded recent local log without new storage policy. |
| Put the feature in a new destination | Console is the live work surface and MCP owns permissions/audit. A Console command backed by MCP services fits the existing destination ownership model. |

## Consequences

The recommendation engine must be a pure, testable MCP service layer that consumes recent execution records, live `HubTool` catalog entries, and resolved `EffectiveToolState` values. It must only recommend tools with repeated agent-approved decisions that are still ask-gated and not downgraded by existing safety checks.

An `approved` record represents a prompted human approval. Later executions covered by the same in-memory session grant are recorded as `approved-session` and do not count as additional prompts. Recommendation application must use the same live catalog snapshot that produced the report so a definition that changes between catalog reads cannot inherit an older recommendation.

The report must surface excluded and empty states honestly: no log, no tools, no repeated approvals, already allowed, denied, definition changed, high-risk floor, stale/unavailable tool, or below threshold.

No telemetry, analytics, remote upload, or new tracking store is introduced. The feature may read the local MCP execution log and permission store; it must respect the existing argument-capture setting by not depending on arguments at all.

Future work can add richer MCP Audit/Permissions panels or a separately governed auto-approval mode, but those require new task scope and ADR review.

## Links

- [ADR-009: Local Skill Trust Boundary](009-local-skill-trust-boundary.md)
- [ADR-011: Chatbook Workbench UI System](011-chatbook-workbench-ui-system.md)
- [ADR-015: Complete the shell destination IA](015-shell-destination-ia.md)

# ADR-001: Adopt Backlog Decisions As Canonical ADRs

Status: Accepted
Date: 2026-06-02
Related Task: backlog/tasks/task-76 - Adopt-canonical-ADR-workflow.md
Supersedes: N/A

## Decision

Canonical Architecture Decision Records for `tldw_chatbook` live in `backlog/decisions/`, and significant architectural choices must be checked against this ADR workflow during planning and closeout.

## Context

The project has historical ADR-like material spread across development docs and feature-specific files. `AGENTS.md` already names `backlog/decisions/` as the project decision location, but that directory did not have a canonical ADR template, index, trigger rule, or workflow.

The project also uses Backlog.md tasks and Superpowers specs/plans to manage work. The ADR workflow needs to fit those existing systems rather than add a disconnected documentation lane.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Store canonical ADRs in `Docs/adr/` | This is a common convention, but it conflicts with the existing Backlog.md source-of-truth guidance that project decisions live in `backlog/decisions/`. |
| Require ADRs for every implementation task | This would create too much ceremony and make ADRs less useful. |
| Rely only on reviewer requests for ADRs | This would not reliably revive ADR usage or help agents decide when a decision needs durable context. |
| Add automated ADR validation immediately | The rule is new and should prove useful through checklist enforcement before adding brittle automation. |

## Consequences

New significant architectural decisions have a single canonical home. Backlog tasks and Superpowers plans must include an ADR check, but routine bug fixes and small implementation changes do not need ADRs.

Historical ADR-like docs remain useful context, but they are not canonical under the new immutability rules unless a future ADR explicitly imports or supersedes them.

Global Superpowers skills are not changed by this first rollout. The project can generalize this workflow later after using it on real tasks.

## Links

- Spec: `Docs/superpowers/specs/2026-06-02-adr-workflow-design.md`
- Plan: `Docs/superpowers/plans/2026-06-02-adr-workflow-implementation.md`
- ADR index: `backlog/decisions/README.md`
- Historical index: `backlog/decisions/historical-index.md`

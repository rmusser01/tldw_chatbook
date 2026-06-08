# Architecture Decision Records

Canonical ADRs for `tldw_chatbook` live in this directory.

ADRs explain why significant architectural decisions were made. Backlog tasks, Superpowers specs, and implementation plans explain what work is being done and how it is being executed.

## Index

| ADR | Status | Decision |
| --- | --- | --- |
| [ADR-001](001-adopt-backlog-decisions-as-canonical-adrs.md) | Accepted | Use `backlog/decisions/` as the canonical ADR location and require ADR checks for significant architectural choices. |
| [ADR-002](002-openai-compatible-model-discovery.md) | Accepted | Keep OpenAI-compatible model discovery local, manual, and scoped to configured providers with explicit user persistence. |
| [ADR-003](003-settings-library-rag-defaults.md) | Accepted | Keep Library/RAG Settings scoped to persisted global defaults while Library owns active search and Console owns staged context. |
| [ADR-004](004-settings-storage-defaults-restart-boundary.md) | Accepted | Keep Settings storage defaults persisted under `database` config while active storage handles remain restart-boundary owned. |
| [ADR-005](005-console-workspace-server-readiness.md) | Accepted | Keep Console workspace switching local-first while exposing honest server-readiness, handoff, runtime, and ACP task/run states behind adapter boundaries. |

## Historical Decision Material

Some older decision material exists outside this directory. See [historical-index.md](historical-index.md). Historical entries are context, not canonical ADRs under the current immutability rules.

## When An ADR Is Required

Create or link an ADR when a task makes or changes a significant architectural choice, including:

- Storage, schema, migrations, sync, conflict policy, or data ownership.
- Provider/runtime boundaries, adapters, service contracts, or cross-module interfaces.
- Security, privacy, encryption, authentication, permissions, or data exposure.
- Dependency, framework, tooling, packaging, or runtime policy choices.
- Long-lived UX/application structure choices, such as navigation model or screen ownership.
- Decisions that reject a plausible alternative future contributors may ask about again.

Do not require an ADR for routine bug fixes, small UI polish, copy-only changes, mechanical refactors that preserve existing boundaries, test-only changes, or direct implementation of an existing ADR.

## Workflow

1. Read relevant ADRs before implementation planning.
2. Add an ADR check to every implementation plan:
   - `ADR required: yes/no`
   - `ADR path: backlog/decisions/NNN-short-title.md or N/A`
   - `Reason: brief explanation`
3. If an ADR is required, create it before implementation starts.
4. Link ADRs from Backlog task plans, Superpowers plans, and implementation notes when relevant.
5. At closeout, record ADRs created, superseded, or followed.

## Immutability

Accepted ADRs are immutable except for typo fixes, link repairs, or metadata that does not alter the decision. If a decision changes, create a new ADR and mark the old one as `Superseded by ADR-NNN`.

## Naming

Use numeric filenames:

- `000-template.md`
- `001-short-title.md`
- `002-short-title.md`

Do not reuse numbers.

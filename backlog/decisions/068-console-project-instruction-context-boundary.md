# ADR-068: Treat repository instructions as ephemeral path-aware Console context

Status: Accepted
Date: 2026-08-20
Related Task: Implementation tasks will be created from the approved design during planning.
Supersedes: N/A

> A post-acceptance implementation-readiness review found that synchronized
> conversation metadata could not own device-local binding state, mutable
> binding IDs could not detect retargeting alone, and the existing
> security-review callback should not absorb optional context preparation.
> Proposed [ADR-069](069-console-project-instruction-local-state-and-preflight.md)
> will supersede this decision if accepted.

## Decision

Chatbook Console agents will support repository-authored `AGENTS.md` and
`AGENTS.override.md` as **untrusted, ephemeral user-level project context**.
The selected workspace folder binding is the discovery boundary. Instructions
compose from that root to the session working directory and activate lazily for
deeper local filesystem/git/patch targets before any affected tool batch can be
approved or executed.

`AGENTS.override.md` takes precedence over `AGENTS.md` in each directory, with
one effective non-empty file per directory and broad-to-specific composition.
V1 does not load global files, fallback filenames, or instructions from other
workspace bindings. It refuses symlinked instruction files and directory
symlink traversal.

The parent agent and subagents share a run-local activation ledger and budget,
while each model conversation tracks which activation revisions it has
actually received. Newly required guidance causes an atomic batch deferral,
content-free tool-result stubs, and a separate ephemeral provider-context
update. Automatically loaded repository contents never travel through tool
results or durable review records.

Chatbook's system prompt, workspace authorization, path confinement,
permission review, risk policy, and provider safety remain authoritative.
Automatically loaded instruction bodies are never persisted to conversations,
agent runs, steps, transcript markers, compaction summaries, exchange captures,
or logs. Here, exchange capture means durable historical capture; the explicit,
nonpersistent Next Send preview may display the exact rider. Explicit file
reads and model-authored quotations retain their normal persistence semantics;
automatic loading is not a data-loss-prevention system.

## Context

Workspace-bound Console agents can manipulate project files but currently lack
the repository conventions users already maintain for coding agents. Codex
defines the requested `AGENTS.md`/`AGENTS.override.md` hierarchy. Claude Code's
native `CLAUDE.md` mechanism demonstrates the value of loading nested guidance
when the agent works in a subdirectory.

Directly appending repository text to Chatbook's system prompt would grant it
the wrong trust level. Loading it through tool-review verdicts would be worse:
the current runtime persists tool results and full run logs, so instruction
bodies would become durable data. A model-only prompt convention would not
guarantee that guidance is loaded before an affected tool executes. The
decision therefore requires an explicit project-context layer and an atomic
preflight boundary.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Add repository text to the system prompt | Project-controlled text is untrusted and must not share Chatbook policy authority. |
| Ask the model to discover files with ordinary tools | Discovery would be optional and could happen after an affected action; behavior would vary by model. |
| Load only root-to-working-directory instructions | Misses the requested nested, path-specific behavior when tools enter deeper scopes. |
| Persist active instructions in the conversation or agent run | Risks stale guidance and stores potentially sensitive repository contents outside the project. |
| Return new instructions as tool-review failures | Existing tool results are durably logged, leaking the instruction bodies. |
| Apply the hierarchy to all workspace bindings at once | Removes the clear working-project boundary and creates ambiguous cross-root precedence. |
| Interpret MCP paths as local paths | MCP locations may be remote or provider-defined and cannot safely share local filesystem semantics. |
| Support global files, fallback names, and symlinks in v1 | Adds configuration and trust complexity without being required for repository compatibility. |

## Consequences

- Sessions need an explicit working-folder binding and working-directory
  relative path in addition to their workspace identity.
- New sessions enable the feature, while legacy sessions without explicit
  metadata remain disabled until the user opts in.
- Local path-aware tools gain a small inspection contract, and the agent loop
  gains a typed batch result that separates persistable protocol stubs from
  nonpersisted context updates.
- Parent/subagent concurrency must distinguish globally activated content from
  content delivered to an individual model conversation.
- Omitted, stale, invalid, or failed sources need terminal, per-chain-visible
  outcomes so fail-open retries cannot loop.
- Initial and nested instruction content each have a 32 KiB raw-content budget
  and must also fit the model's remaining token allowance.
- Compaction excludes automatic riders from summary input and rebuilds active
  guidance from the immutable run snapshot rather than rereading files.
- Missing or broken project guidance warns but does not grant or revoke tool
  authority. Invalid saved binding identity blocks silent retargeting and
  requires user recovery.
- Supporting another repository-memory convention or remote tool path model
  requires a later decision; it is not implied by this ADR.

## Links

- [Approved design spec](../../Docs/superpowers/specs/2026-08-20-agents-md-support-design.md)
- [ADR-005: Console workspace/server readiness](005-console-workspace-server-readiness.md)
- [ADR-028: Settings workspaces and folder roots](028-settings-workspaces-category-and-folder-roots.md)
- [ADR-032: Local agent tool permission boundary](032-local-agent-tool-permission-boundary.md)
- [Codex AGENTS.md guide](https://learn.chatgpt.com/docs/agent-configuration/agents-md)
- [Claude Code memory guide](https://code.claude.com/docs/en/memory)

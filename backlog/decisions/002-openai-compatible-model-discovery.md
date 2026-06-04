# ADR 002: OpenAI-Compatible Model Discovery

Status: Accepted
Date: 2026-06-04
Related Task: [backlog/tasks/task-78 - OpenAI-Compatible-Model-Discovery.md](../tasks/task-78%20-%20OpenAI-Compatible-Model-Discovery.md)
Supersedes: N/A

## Decision

Chatbook v1 model discovery is local, manual, and scoped to existing configured OpenAI-compatible providers. Discovery results are stored in a runtime cache first. Users must explicitly persist selected raw model IDs into the exact existing top-level `[providers].<provider-list-key>` entry. Settings owns discovery and persistence. Console consumes the same merged saved and discovered model list.

Provider identity and persistence must reuse Console provider normalization. If multiple top-level `[providers]` keys normalize to the same provider identity, persistence is refused until the user resolves the ambiguity.

## Context

Users need to discover models exposed by configured OpenAI-compatible endpoints without waiting for app or config updates. Existing Chatbook model selection is backed by the top-level `[providers]` config and Console readiness/execution provider identity logic.

Settings is the application configuration hub, while Console is the primary agentic control surface. Model discovery therefore needs to improve Settings and Console without creating a second registry that diverges from existing provider configuration.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Add a parallel provider registry for discovered models | A second registry would split provider identity, model ownership, and persistence semantics from the existing top-level `[providers]` config and Console readiness/execution normalization. Reusing the current provider list keeps v1 discovery additive instead of introducing a new source of truth. |
| Auto-save every discovered endpoint result | Discovery can return experimental, hidden, deprecated, or account-scoped model IDs. Auto-saving would mutate user configuration without consent and could make transient endpoint state look like durable application support. Explicit persistence keeps the boundary clear. |
| Integrate server keyring credentials or tldw_server catalog sync in v1 | Those paths require server trust, credential ownership, synchronization policy, and catalog contract decisions beyond local endpoint discovery. They are deferred so this ADR can preserve a local-only first step. |
| Add native Ollama or Kobold discovery in the same tranche | Native provider APIs have different endpoint shapes, model metadata, and capability semantics from OpenAI-compatible `/models` discovery. They are deferred to avoid weakening the OpenAI-compatible contract or overgeneralizing before the local workflow is proven. |
| Allow ad hoc endpoint discovery outside configured providers | It would blur provider setup with discovery and increase security/recovery complexity; v1 discovers only existing configured providers. |

## Consequences

Discovery does not create a new provider registry and does not auto-save endpoint results. Unknown discovered models remain usable with capability warnings. Native provider-specific discovery paths, server keyring credentials, and tldw_server catalog sync are deferred to later ADRs.

Implementations must preserve exact top-level `[providers]` key spelling/casing and write only selected raw endpoint model IDs. Settings and Console must share the same merged saved/discovered model list so users do not see contradictory model availability.

## Links

- [OpenAI-compatible model discovery PRD](../../Docs/superpowers/specs/2026-06-04-openai-compatible-model-discovery-prd-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-06-04-openai-compatible-model-discovery-implementation.md)
- [Backlog task TASK-78](../tasks/task-78%20-%20OpenAI-Compatible-Model-Discovery.md)

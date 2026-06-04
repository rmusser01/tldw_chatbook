# ADR 002: OpenAI-Compatible Model Discovery

## Status

Accepted

## Context

Users need to discover models exposed by configured OpenAI-compatible endpoints without waiting for app or config updates. Existing Chatbook model selection is backed by the top-level `[providers]` config and Console readiness/execution provider identity logic.

## Decision

Chatbook v1 model discovery is local, manual, and scoped to existing configured OpenAI-compatible providers. Discovery results are stored in a runtime cache first. Users must explicitly persist selected raw model IDs into the exact existing top-level `[providers].<provider-list-key>` entry. Settings owns discovery and persistence. Console consumes the same merged saved and discovered model list.

Provider identity and persistence must reuse Console provider normalization. If multiple top-level `[providers]` keys normalize to the same provider identity, persistence is refused until the user resolves the ambiguity.

## Consequences

Discovery does not create a new provider registry and does not auto-save endpoint results. Unknown discovered models remain usable with capability warnings. Native provider-specific discovery paths, server keyring credentials, and tldw_server catalog sync are deferred to later ADRs.

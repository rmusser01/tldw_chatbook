# ADR-149: Offer-first Canvas guide and inline skill

Status: Accepted — written spec approved by the user on 2026-09-10
Date: 2026-09-10
Related Task: [TASK-32312](../tasks/task-32312%20-%20Design-offer-first-Canvas-guidance-and-companion-skill.md)
Extends: [ADR-121](121-local-versioned-canvas-artifacts-and-browser-sandbox.md),
[ADR-124](124-canvas-mermaid-subset-and-immutable-runtime-profiles.md)
Preserves: [ADR-009](009-local-skill-trust-boundary.md)

## Context

Canvas already supplies conversation-owned HTML artifacts, constrained script
execution, an offline Mermaid subset, and create/read/update/list tools. The
assistant needs to recognize worthwhile Canvas opportunities without spending
tokens on an unsolicited artifact. The user selected a brief offer followed by
consent, implemented through model guidance rather than a host acceptance gate.

Detailed examples should be available after acceptance without filling every
request's context. Model-invoked Chatbook skills currently launch restricted child
runs without Canvas tool authority; explicit user skill mentions can expand inline
in the owning Console conversation. Neither artifact ownership nor child authority
should change merely to deliver authoring instructions.

## Decision

1. Put concise offer-first guidance in both existing Canvas discovery and loaded
   tool guidance. Proactive offers name the artifact and benefit; detailed guide
   loading and source generation wait for acceptance. Explicit creation requests
   and requested edits already authorize that work. Consent covers that artifact
   and bounded corrections, not unrelated artifacts or unrequested redesigns.
   Declines suppress repeated offers for the same idea. No response is not consent.
2. Add `canvas:canvas_guide`, exposed as `canvas_guide(topic)`, to the existing
   live, scoped Canvas provider. Its closed topic set is `basics`, `controls`,
   `mermaid`, and `repair`. It returns shipped Markdown documentation in a bounded
   JSON tool result, at most 12 KiB UTF-8 including the envelope. It accepts no
   filesystem path or generated source and performs no model, compiler, browser,
   selection, revision, or host action. Existing Canvas policy, reservations,
   scope validation, and the kill switch apply.
3. Keep packaged topic documents as the common authoring reference. Load them
   lazily with package resources, including in installed wheels. Keep detailed
   bodies out of discovery hints and tool schemas. Model results contain the
   requested guide; non-model record projections retain bounded topic/status/size
   metadata and do not emit Canvas artifact cards.
4. Ship an optional importable `$canvas` skill in the existing example-skill
   convention. Its short body uses `context: inline`, `user_invocable: true`, and
   `disable_model_invocation: true`; it directs the main assistant to the guide
   and existing artifact tools. It retains normal argument and mention handling,
   with no model override or tool-authority changes. It is not auto-installed or
   auto-trusted. Import/use stays within ADR-009, and an unavailable skill never
   becomes a reason for the guide to read or trust user files.
5. Product documentation and imported skills have distinct owners. The built-in
   guide reads only fixed packaged resources and works without an imported skill.
   The skill does not duplicate the detailed guides, and its explicit invocation
   does not turn an unspecified task into consent to generate an artifact.
6. Preserve exact-profile guidance and all renderer, storage, revision, and
   confirmation contracts. Guide availability does not admit a runtime profile.
   Distinguish staged/saved source from a working preview. Use received diagnostics
   for bounded repair; after one failed repair, explain the limitation and let the
   user decide on further work instead of generating repeated speculative versions.
7. Verify bounded guide behavior, policy/projections, installed-package resources,
   trusted inline invocation, real example compilation and browser interactions,
   and a small recorded model-behavior sample. Prompt-presence tests alone do not
   prove compliance or quantify token savings.

The [design spec](../../Docs/superpowers/specs/2026-09-10-canvas-guidance-skill-design.md)
defines the full behavior, error handling, content, and verification contract.
The [implementation plan](../../Docs/superpowers/plans/2026-09-10-canvas-guidance-skill-implementation.md)
maps the accepted decision to concrete provider, guide, skill, and verification work.

## Alternatives considered

- **Skill alone:** ordinary Canvas discovery would not carry the consent rule.
- **Model-invoked authoring skill:** adds a child model run without Canvas authority;
  widening that boundary is unnecessary for documentation.
- **Host-enforced proposal/acceptance:** adds UI and durable/transient state beyond
  the user's selected approach. Blocking create/update alone cannot recover tokens
  already spent generating source arguments.
- **Always inject a full manual:** incurs guide/example context cost before the user
  chooses Canvas and repeats instructions unnecessarily.
- **General skill-loading framework:** changes a cross-product runtime contract to
  solve a narrow Canvas documentation need.

## Consequences

One read-only public tool and a small packaged guide set are added. Tool discovery,
closed result projections, and wheel packaging need focused integration tests.
An imported skill can be customized without changing product guide ownership.
Runtime-profile guidance remains authoritative if an older skill gives stale advice.

Offer-first behavior remains a model instruction, not enforced acceptance or a
token-budget guarantee. The design adds no database migration, settings, dependency,
network access, renderer privilege, or general skill activation mechanism.

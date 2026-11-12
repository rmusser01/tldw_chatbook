# ADR-133: Question-directed Library reading experiment

Status: Accepted for the fixture experiment; product adoption pending evaluation
Date: 2026-09-07
Related task: [TASK-32029](../tasks/task-32029%20-%20Evaluate-question-directed-reading-of-selected-Library-sources.md)
Specification: [Library source-reader experiment](../../Docs/superpowers/specs/2026-09-07-library-source-reader-design.md)
Review: [Findings and resolutions](../docs/library-source-reader-design-review-2026-09-07.md)

## Decision

Evaluate a bounded explicit reader for selected local Library Media document
and transcript text before adding automatic routing. A single smaller-model
request receives host-assembled source packets and returns findings backed by
quotations. The host validates references and quotations before any finding
enters the main model's context. Exact matching establishes provenance location,
not semantic support; quality remains a separate evaluation.

Execute the reader through the existing sensitive `complete_auxiliary()`
gateway, not through named agent spawning. Its immutable request contains only
reader instructions and explicit question/source data. An empty named-agent
allowlist inherits tools and cannot enforce this boundary.

The worker uses a pinned explicit model at the already selected provider
endpoint and receives no tools or unrelated conversation context. Endpoint
identity alone does not prove downstream processor identity for model routers;
the initial experiment uses only nonsensitive fixtures in disposable databases.
Source access stays within the current Console direct-Library boundary. This
capability is distinct from ADR-030's lexical-only shared direct tools and is
not exported through MCP in this experiment. Do not silently change provider,
retrieval mode, or source scope on failure.

Reuse existing bounded Library reads, provider admission and lifecycle,
normalized usage, and canonical evidence ownership. Keep worker-submitted
source packets distinct from main-model-submitted evidence. A future product
integration must extend the canonical citation contract if required; it must
not invent a sidecar or label incomplete provenance as complete.

For the harness, reuse canonical evidence data types in memory and write exact
request manifests to a fixture-only experiment artifact. Do not add product
tool registration, settings, fleet integration, or durable citation writes yet.
Selection authority is a host-owned manifest; empty or invalid scope must fail
before invoking search helpers that could interpret a missing filter as global.

A timeout stops result delivery and further dispatch, not necessarily a
synchronous provider request already running in a thread. Retain ownership of
the outstanding request, account for late usage when observable, and record
unknown spend honestly. Do not add a general cancellation framework for this
experiment or promise hard cancellation that the current gateway cannot deliver.

The experiment compares direct reading, existing source-scoped retrieval, and
worker-assisted reading on a pinned corpus. Account for all requests and cache
effects and grade evidence quality as well as cost and latency. Require paired
measurements, record every failed/excluded attempt, and make unknown spend or
failed answers ineligible for a financial pass. Location-validated quotations
remain semantically unchecked. Product adoption is conditional on the
specification's quality-first criteria, including comparison against retrieval.

## Alternatives and consequences

- A named agent alone is suitable for a manual demonstration but cannot enforce
  the required source selection and evidence-result contract through instructions.
- Automatic interception adds policy and latency to ordinary reads before a
  benefit has been established for Chatbook's workloads.
- Cross-provider routing may offer savings but changes data destination and
  authentication ownership; it is outside this proposal.
- Summary-only output is smaller but prevents reliable evidence inspection.
- A separate research engine or snapshot store would duplicate existing runtime
  and citation ownership boundaries.

The experiment creates no storage migration and changes no shipped behavior.
Acceptance authorizes the specified experiment architecture. Automatic routing
and claims of preserved answer quality require the comparison evidence.

## Implementation qualification

The local harness is tracked in TASK-32029. The initial CLI supports explicitly
configured OpenAI-compatible main/reader calls. A source-scoped retrieval adapter
is implemented, but the CLI does not construct or attach a fixture vector index;
that arm remains unavailable. No three-way adoption decision is possible yet.

The harness records exact auxiliary gateway messages and parameters; provider
adapters can transform these before transmission. Local HTTP tests inspect the
actual wire independently. Experiment quote records remain ephemeral, use the
shared Library identity adapters, and do not pretend to be canonical persisted
Console citation traces. Final wire retention and canonical evidence integration
remain qualification gaps before product integration, as detailed in the
[local report](../docs/library-source-reader-local-qualification-2026-09-08.md).

## Related decisions

- [ADR-024: Canonical citation provenance](024-rag-citation-provenance-and-source-resolution.md)
- [ADR-029: Safe prompt improvement transactions](029-versioned-prompt-artifacts-and-safe-improvement-transactions.md)
- [ADR-030: Direct local Library tool boundary](030-local-library-agent-tool-boundary.md)
- [ADR-052: Console context and compaction policy](052-console-conversation-memory-and-compaction-policy.md)
- [ADR-130: Model-call lifeline client teardown](130-model-call-lifeline-client-teardown.md)
- [ADR-131: Durable agent budget accounting](131-durable-agent-budget-accounting.md)

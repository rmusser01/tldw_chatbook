# Library source-reader design review

Date: 2026-09-07
Scope: [experiment specification](../../Docs/superpowers/specs/2026-09-07-library-source-reader-design.md), [proposed ADR-133](../decisions/133-question-directed-library-reading-experiment.md), and [TASK-32029](../tasks/task-32029%20-%20Evaluate-question-directed-reading-of-selected-Library-sources.md).
Method: static code/contract inspection, six targeted existing tests, and design
revision. No model evaluation or application implementation was performed.
Findings below describe weaknesses
in the original proposal, not claims that an implemented reader is vulnerable.

## Findings and resolutions

### 1. Use the auxiliary gateway, not a named subagent — high priority

The original design relied broadly on existing agent machinery without choosing
a safe execution seam. [AgentDefinition](../../tldw_chatbook/Agents/agent_models.py)
explicitly treats an empty tool allowlist as inheritance. In
[agent_service.py](../../tldw_chatbook/Agents/agent_service.py),
`build_first_request_schema_plan` adds runtime schemas separately, and `spawn`
rejects combining a named agent with an explicit allowlist override.

The existing
[complete_auxiliary](../../tldw_chatbook/Chat/console_provider_gateway.py)
method already supplies a sensitive, non-streaming request without normal
history, tool dispatch, fallback copy, or persistence. Use it directly through
an injected gateway port. The revised spec excludes fleet run creation and
requires recording-server evidence that no tools or unrelated context leak
into the worker request. This both strengthens isolation and reduces scope.

### 2. The proposed hard timeout exceeded the current contract — high priority

The auxiliary gateway runs synchronous adapters via `asyncio.to_thread`.
Cancelling the await leaves the underlying adapter alive; the existing
`test_auxiliary_completion_cancellation_starts_no_second_call_and_resets` in
[gateway tests](../../Tests/Chat/test_console_provider_gateway.py) explicitly
releases a blocked adapter after cancellation. ADR-130 also limits guarantees
around non-cooperative cleanup. A result deadline cannot promise a remote
billing cutoff or guaranteed late usage.

The revised experiment allows one outstanding reader request, retains ownership
of a shielded call until it drains, stops further dispatch after cancellation
or timeout, discards late content, and reports missing spend as unknown. It
requires qualification of transport timeouts and owner cleanup and does not
build a general cancellation framework. Missing usage is a financial blocker.

### 3. An empty source filter can become an unrestricted query — high priority

The proposal said IDs were not authority but did not define who owned the
allowed selection. The harness now owns a frozen manifest, and validates the
entire request against it before body reads. Wrong-type, foreign, empty, or
stale selection is a refusal, never a signal to search all sources.

This distinction matters in the current code:
[build_semantic_allowlists](../../tldw_chatbook/Chat/rag_scope.py) returns `None`
for non-scoped states, leaving callers responsible for the empty-scope guard.
[RAGService.search](../../tldw_chatbook/RAG_Search/simplified/rag_service.py)
supports pre-ranking `metadata_allowlist`, while its similarly named
`filter_metadata` is a post-filter. The public
[Library RAG tool](../../tldw_chatbook/Agents/library_rag_tool_provider.py)
offers source types, not selected IDs. The revised baseline names the correct
service boundary and requires source revision/provenance parity.

### 4. Cheap failures could masquerade as savings — high priority

The original median cost and critical-error gate did not prevent empty answers,
rejected findings, or timeouts from making an arm appear cheap. It also did not
fully specify paired comparisons, cache order, retrieved-source preparation,
or reference facts fixed before evaluation.

The revised protocol fixes development/held-out cases, common eligibility,
essential facts, output allowances, reasoning settings, and interleaved arm
order. It reports failed attempts and total spend, task success, essential-fact
recall, paired cost ratios and latency deltas. No financial pass is possible
with a failed comparison attempt or unknown spend. Query embeddings and
reranking count; index construction and cache conditions are reported separately.
Multi-question savings require their own repeated-source scenario.

This also follows the incident documented in
[lessons-testing-evidence](lessons-testing-evidence.md), “A metric can be graded
on fallback content, and nothing in it says so”: good citation metrics once
hid failed summarization in this repository. The experiment records execution
path alongside answer scores and permits no silent fallback.

### 5. Reference validation and result limits were underspecified — medium priority

Unique quotations prove where text occurred, not that it supports the worker's
claim. The output now explicitly labels candidate claims as semantically
unchecked and supplies original excerpts for the main model's reasoning.
Adversarial tests include real but misleading quotations and instructions in
source text, not only made-up references.

The revised contract bounds packets, findings, references, quote lengths, and
serialized JSON and rejects extra/duplicate keys and malformed envelopes.
Locations use unchanged Unicode source spans; overlaps are deduplicated. Empty
source bodies, stalled pagination, and mid-read revision changes are explicit
errors. Byte and character budgets are checked separately, with complete
findings removed rather than JSON sliced. A short source is not rejected merely
because it cannot supply an arbitrarily long quote.

The 64-packet ceiling and packet size align with
[citation_trace_models.py](../../tldw_chatbook/Chat/citation_trace_models.py)'s
64-entry prompt sets and 64-KiB snapshot bound. This avoids deferring an obvious
size incompatibility to product integration.

### 6. Provider identity and accounting were overclaimed — medium priority

An unchanged endpoint can still route different model IDs to different upstream
processors. `complete_auxiliary()` constructs result provider/model labels
from the configured resolution, not an independently verified served identity.
The revised design pins explicit models and routing settings, records configured
versus observed identity separately, and limits v1 to nonsensitive fixtures.
Private-data destination qualification remains a product integration gate.

The direct llama.cpp auxiliary path currently returns text without normalized
`ProviderUsage`; malformed or cancelled responses can also lack usage. The
spec now forbids interpreting those cases as zero cost or proving dollar
savings. Capacity, thinking settings, wire overhead, and all request budgets
must be qualified before a paid comparison starts.

### 7. The experiment was drifting into product infrastructure — medium priority

Requiring durable multi-stage citation writes, Console tool wiring, and generic
lifecycle guarantees before measuring the idea would expand a small experiment
into several subsystems. The revised design uses existing evidence types in
memory, disposable Library databases, and a fixture-only request manifest.
It makes no changes to user conversations, fleet history, memory, settings,
or citation persistence. Console registration, source inspector integration,
and any versioned trace extension are later gates contingent on the report.

## What remains uncertain

- Whether the selected smaller model preserves exceptions, contradictions, and
  essential facts. Exact-quote tests cannot answer this.
- Whether total cost beats both direct reading and scoped retrieval under real
  cache behavior, especially on repeated questions over the same sources.
- Which configured models support useful structured output, bounded transport,
  and sufficiently complete usage reporting. Qualification may exclude some
  models; that is an explicit experiment limitation, not permission to bypass
  the gateway or choose a new provider.
- Whether the canonical citation trace needs a versioned extension for product
  integration. The fixture experiment does not claim this integration exists.

The revised design is suitable for implementation planning of the experiment.
These remaining questions are the experiment's purpose, not established results.

## Verification

Six selected tests in `Tests/Chat/test_console_provider_gateway.py` and
`Tests/Chat/test_sensitive_llm_logging.py` passed using the repository virtualenv
with `pytest -q --no-cov`. They cover immutable requests, tool-free one-shot
dispatch, cancellation with a blocked adapter, the direct llama.cpp auxiliary
path, disabled sensitive-request retries, and pinned endpoint behavior across
config changes. They use local test doubles; no paid provider call ran.

Result: **6 passed**, with one requests dependency-version warning. No full
suite ran. These results verify existing foundations, not an unimplemented
source reader. Relative document links, whitespace, placeholders, and Markdown
fence checks passed for the spec, ADR, draft, and this review.

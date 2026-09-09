# Question-directed Library reading experiment

Date: 2026-09-07
Status: Initial DeepSeek development run and a focused negative-evidence
revision probe completed; full comparison, held-out evaluation and human
grading remain pending.
Backlog: [TASK-32029](../../../backlog/tasks/task-32029%20-%20Evaluate-question-directed-reading-of-selected-Library-sources.md)
Decision: [ADR-133](../../../backlog/decisions/133-question-directed-library-reading-experiment.md)
Review: [Findings and resolutions](../../../backlog/docs/library-source-reader-design-review-2026-09-07.md)

## Purpose

Determine whether a smaller model can inspect selected Library documents and
transcripts, return useful evidence, and reduce the total cost of the main
model's answer without losing important facts or qualifications. The outcome
is an adopt, revise, or reject report. Automatic routing is a later decision.

The approved initial scope is Library documents and transcripts. The experiment
uses nonsensitive fixtures imported into disposable Library databases, not the
user's live Library. This does not authorize sending private content to a new
model or provider.

The motivating [Spotify report](https://engineering.atspotify.com/2026/9/portal-by-spotify-cut-my-claude-code-token-usage-by-90)
describes reducing main-model bulk-read tokens by delegating reading. Its
headline is not a total-cost or answer-quality guarantee for Chatbook.

## Alternatives

| Approach | Benefit | Limitation |
| --- | --- | --- |
| Named reader agent using existing tools | Smallest manual trial; model and instructions are configurable | Instructions alone do not enforce source scope, evidence format, or preservation of exact references |
| Explicit bounded source-reader tool, prototyped before product integration | Source scope and evidence checks are enforceable; comparison is reproducible | Needs a narrow source-assembly and result-validation contract |
| Automatically intercept large reads | Broad potential context savings | Adds policy, bypass, latency, and failure behavior before the benefit is established |

Recommend the explicit tool contract. First exercise it through an experimental
harness; integrate it into Console only if the report supports adoption. Reuse
production Library reads, request preparation, and the existing auxiliary
completion gateway. Do not register a product tool, add settings, create fleet
runs, or extend citation storage to conduct the experiment.

## Initial scope

- Local Library Media items whose stored `content` contains document text or a
  transcript. Media analysis summaries are not substituted for original text.
- A question and one to six explicit Media IDs selected from existing Library
  discovery or selection. No automatic collection expansion or whole-Library
  search by the reader.
- An explicitly configured worker model on the main model's existing provider
  endpoint. Pin provider, execution key, base URL, credential ownership, model,
  sampling/thinking settings, and advertised limits for each run. No automatic
  model aliases, model fallback, or new provider selection.
- A single bounded worker request with no tools and no recursive delegation.
- Findings and exact excerpts returned to the main model for its final answer.

Workspace files, Notes, live URLs, OCR/transcription, generated files, automatic
read interception, worker result caching, and new screens are outside this
experiment. The existing Media viewer remains the source-viewing destination.

## Existing contracts to preserve

1. **Library admission:** ADR-030 defines direct Library access and the
   direct-tools-versus-RAG setting. The reader cannot fetch full Media bodies
   when direct Library access is disabled. It must not be added to the shared
   18-tool lexical-only descriptor inventory: that inventory must not invoke
   an LLM. The proposed Console capability is separately composed and defaults
   off. No standalone MCP exposure is included.
2. **Source reads:** use `LocalLibraryToolService`'s bounded Media detail and
   revision-aware continuation behavior. A changed revision during pagination
   invalidates the assembled item. Do not query raw SQLite rows independently
   or include embeddings, binary fields, or local filesystem paths.
3. **Worker execution:** use `ConsoleProviderGateway.complete_auxiliary()` with
   an immutable `AuxiliaryCompletionRequest` and sensitive-content handling
   (ADR-029). It already excludes tools, chat history, streaming fallback copy,
   and normal persistence. A named `AgentDefinition` is not the execution seam:
   its empty allowlist means inherit, and runtime schemas are assembled
   separately from ordinary tool allowlists. Construct only the reader's
   system instruction and explicit question/source packet data; do not inherit
   conversation memory, skills, project instructions, or pending attachments.
4. **Evidence:** ADR-024 distinguishes source snapshots, actual submitted
   evidence, citation structure, semantic support, and current source access.
   Exact quote matching establishes location, not truth or entailment.
5. **Accounting:** ADR-131's budget tokens are not dollars. Use normalized
   `ProviderUsage` and the existing pricing catalog for cost comparisons.

## Proposed invocation and data flow

Conceptual tool name: `ask_library_sources`.

Input is a nonblank question, bounded at 2,000 Unicode codepoints, and one to
six distinct opaque Media IDs. The harness owns a frozen manifest of selected
IDs, expected revisions, and source authority, created outside model output.
Every requested ID must belong to that manifest and current permitted scope.
Validate the complete ID set before reading any body; a rejected or empty
intersection fails closed and never means unrestricted search. The future tool
must bind an equivalent host-owned selection instead of trusting model-supplied
IDs as authority. Model arguments cannot change the worker destination or limits.

```mermaid
flowchart LR
    A[Question and selected Media IDs] --> B[Validate access and source revisions]
    B --> C[Assemble bounded source packets]
    C --> D[One worker request without tools]
    D --> E[Validate findings and exact quotations]
    E --> F[Compact evidence in main-model context]
    F --> G[Main-model answer with source references]
    G --> H[Existing source inspector and Media viewer]
```

The host fetches source content directly for the worker; the main model does
not first read or repeat that content as spawn-task text. A general-purpose
subagent prompt is therefore insufficient on its own.

Use contiguous unchanged source spans, with host-generated packet IDs and
zero-based Unicode-codepoint positions. Prototype packets hold at most 4,000
codepoints with 200-codepoint overlap and at most 64 packets overall. Preserve
newlines and Unicode exactly. Deduplicate overlap matches by original source
span. These limits fit the existing 64-entry prompt evidence contract and
64-KiB per-snapshot bound; do not invent new chunking or normalization machinery.

Read source pages sequentially under the Library service's byte bounds and
check aggregate bytes before retaining each page. Verify cursor progress,
revision consistency, and complete text coverage. Empty/whitespace-only source
bodies and non-progressing continuations are explicit source errors. Recheck
selection/admission and expected revisions immediately before dispatch. The
result describes captured per-source revisions, not an atomic snapshot across
all items. Changes after dispatch preserve historical identity and trigger the
existing stale/revoked-access behavior at later use.

Prototype ceilings are six sources, 256 KiB of assembled UTF-8 source content,
24,000 worker input tokens including packet overlap, labels, instructions, and
provider serialization overhead, and 2,000 worker output tokens. Preflight must
account for the exact serialized request that will be sent; reject any prepared
request with dropped or transformed source units. Use current model-specific
context/output limits and reasoning reservation, with conservative accounting
where tokenization is estimated. Unknown capacity makes a model unqualified for
the comparison until an explicit tested cap is supplied. Never silently trim.
Oversized requests must select fewer/shorter sources; shortening the question
does not solve an oversized source corpus.

The 60-second limit is a result deadline, not a promise to stop remote billing.
The auxiliary gateway uses `asyncio.to_thread` for synchronous adapters;
cancelling the await does not kill that request. The harness owns at most one
outstanding reader request, stops issuing calls at timeout/cancellation, and
keeps an observable pending task until transport completion or process exit.
Use a shielded owned task to retain late usage where it is delivered, discard
late answer content, and close loop-owned clients only after their work drains
(ADR-130). A stuck call stops the experiment instead of accumulating workers.
Unobserved late spend remains unknown, never zero. Native-async cancellation and
configured adapter transport timeouts must be qualified separately. No new
thread supervisor or universal hard-cancellation framework is part of this work.

## Result contract

The worker returns exactly one JSON object with `findings`, at most 12 candidate
findings. Each has a nonblank `statement` of at most 400 codepoints and an
`evidence` array of one to three `{packet_id, quote}` references. Packet IDs are
host-generated strings of at most 32 codepoints; quotes are 1 to 800 codepoints.
Reject extra/duplicate keys, non-finite numbers, wrong types, excessive nesting,
and bodies above 32 KiB before schema validation. No Markdown-fence stripping
or partial-JSON repair. Ask for enough neighboring text to make a short quote
unambiguous without rejecting a short source solely for its length. The worker
does not choose source IDs, links, timestamps, final
citation labels, or access permissions.

The host validates schema and size, resolves every packet ID against this
invocation, and requires each nonempty quotation to match a unique contiguous
span within that packet. It computes source positions itself. Source titles,
revisions, locators, and citation labels come from trusted service results and
existing citation adapters. Transcript timestamps are included only when the
stored source already supplies a reliable mapping; they are never inferred
from text offsets or generated by the worker.

Reject a finding with an unknown packet, missing evidence, fabricated quote,
or ambiguous quote occurrence. Other valid findings may survive, but the result
must say it is partial and count rejections. If all findings fail validation,
return `invalid_worker_output`, not an empty successful answer. A valid empty
finding list is `no_evidence_found`, not proof that the selected sources contain
no answer. No automatic repair call is hidden in the first experiment.

The returned envelope separates selected/submitted source coverage, accepted,
rejected and omitted finding counts, and execution status. Accepted findings
carry `quote_location=validated` and `claim_support=unchecked`. The main prompt
calls them candidate claims and requires reasoning from original quotations,
including exceptions and contradictory evidence; it must not present the
worker's interpretation as verified. Complete submission is not complete recall.

Fit the entire envelope, including status and source metadata, against both
the Library serialized byte ceiling and the active tool-result character
ceiling; these are different units and cannot simply be compared numerically.
Reserve final-answer input/output space as well. Drop whole findings in stable
worker order with explicit omission counts, or return `result_budget_too_small`.
Never slice JSON or remove references from retained statements. The eventual
Console integration must prove its generic truncation path leaves this fitted
envelope intact; zero/unlimited runtime limits do not remove feature hard caps.

## Provenance and retention

For nonsensitive fixtures, store a versioned experiment manifest and the exact
worker/main wire payloads in an explicitly selected local output directory;
omit credentials and endpoint secrets. Reuse canonical evidence data types and
source identity adapters in memory. This is a disposable evaluation artifact,
not an answer-level citation store: no writes to the user's conversations,
memory, fleet history, or provenance database, and no private Library experiments
in v1. Ordinary logs carry bounded status/usage metadata under the existing
sensitive logging policy; they must not duplicate source text or model output.

The manifest records the relation from source revision to worker packet to
validated quote to main-model submission, plus retained and omitted findings.
Main-model evidence consists only of the exact excerpts actually sent. Do not
claim the main model saw all worker packets or write model-generated claims as
original source text. Product integration later requires canonical trace and
source-inspector support; any schema extension is a separate reviewed change.

The same API endpoint does not guarantee the same downstream processor behind
a router. The fixture-only experiment pins explicit model IDs, disables
automatic routing/fallback where controllable, and records configured identity
separately from actual served identity when available. Gateway result identity
alone is not proof: `complete_auxiliary()` labels results from the pinned
resolution. Private-data adoption requires destination-policy qualification.
Source and worker content remain untrusted; extraction is not an injection
defense or a provider zero-retention guarantee.

## Failure behavior

Reject wrong-type IDs, unavailable sources, denied access, changed revisions,
unsupported configuration, and excess input before the worker request. Failures
return concise structured reasons and useful next actions. Worker timeout,
cancellation, provider error, and malformed output are distinct from a successful
search that found no evidence. Never silently switch models, providers, or
access modes, and never silently retry by sending the full corpus to the main
model. Direct reading remains an explicit fallback within existing permissions.

## Experiment and decision criteria

Use four development cases and twelve held-out cases, with questions covering direct facts, cross-source
comparison, conflicting accounts, exceptions, absent answers, repeated quotations,
long transcripts, and source changes. Use nonsensitive fixtures for initial
qualification. Separate development cases from the held-out comparison cases;
freeze prompts, corpus revisions, model IDs, limits, and scoring rules before
running the held-out set. Include adversarial source instructions and authentic
but misleading quotations, not just fabricated citations. Freeze essential
facts, critical-error definitions, and acceptable no-answer behavior before
viewing outputs. Run two interleaved repetitions per held-out case; randomize
arm order to reduce load/cache-order bias, and keep each arm's model context
isolated from other arms and prior answers.

Compare three paths on the same source scope and main model:

| Path | Material supplied to the main model |
| --- | --- |
| Direct reading | All selected source text within the common context ceiling |
| Existing retrieval | Retrieved original passages from only those same source IDs |
| Worker-assisted reading | Host-validated findings and original quotations from the worker |

Use identical question text, main model, final-answer prompt, output allowance,
and reasoning settings; only the evidence envelope differs. Establish a common
eligible corpus before any outputs are observed. Log all size/capacity exclusions
and later errors; do not silently remove hard cases. Single-question cost is the
primary endpoint. Add a separately reported, fixed three-question-per-source-set
scenario before claiming multi-turn savings; repeated worker input also costs
tokens, while direct context may benefit from provider caching.

The retrieval baseline must filter to selected sources before candidate ranking
and top-k, using pinned indexed revisions. Filtering unrelated top-k results
after retrieval is not an equivalent baseline. Missing index support is an
explicit unavailable arm, not zero cost or a successful empty result; the
three-way experiment is incomplete until this baseline can be run faithfully.
Use `RAGService.search(metadata_allowlist=...)` and the existing scope-to-ID
adapters for semantic/hybrid profiles; `filter_metadata` is a post-filter and
is not sufficient. Stop on an empty effective scope before calling helpers that
return `None` for non-scoped states. The public `search_library_rag` tool only
accepts source types, so it is not the selected-ID baseline entry point. Match
source revisions and text provenance, not merely IDs. Freeze retrieval profile,
top-k and excerpt budget; count query embeddings and any reranker calls.

Before a paid run, require an operator-supplied request/token/spend ceiling and
validate the complete proposed run matrix against it. No live credentials are
needed for local contract tests. Allow no automatic retries or fallback arms;
record any observed adapter retry as a protocol deviation, with its spend.

Record input/output/cache buckets for every main and worker request, exact
request evidence, wall time, retries, errors, and direct-read fallbacks. Missing
provider usage or pricing is unknown. Show uncached and cache-enabled cases
separately where supported. For local models report tokens and latency without
equating absent API charges with zero compute cost. Report index-build cost
separately from warm-index query cost rather than charging one arm repeatedly
for setup. Preserve model-price provenance and reasoning-token accounting.
Some current auxiliary paths, including direct llama.cpp, return no normalized
usage; report this and disqualify dollar-savings conclusions for those runs.

Blind human review grades answer correctness, missing essential facts,
contradiction handling, and whether cited passages actually support claims.
Deterministic validation separately checks identity, exact quotes, bounds, and
source isolation. Do not substitute a new uncalibrated LLM judge for this review.

An error, invalid/empty answer to an answerable question, or result with no
usable findings remains a failed attempt. Report task-success rate and all
attempted spend; cheap failures cannot count as savings. Show paired per-case
cost ratios and latency deltas across repetitions, plus totals and the slowest
cases. Do not compare unpaired medians or hide failures by reporting only the
successful subset. Unknown usage makes the financial decision inconclusive.

Proposed gate to a limited opt-in pilot: zero source-scope escapes or fabricated
accepted quotations; no new critical evidence errors relative to direct reading
on any held-out case; no lower task-success rate or macro essential-fact recall;
median paired worker/direct total-cost ratio at most 0.80, and no more than 10 seconds
median paired latency increase. No financial pass is allowed if any comparison
attempt has failed or unknown spend. These small-sample thresholds are hypotheses,
not a statistical equivalence claim. Report failures and uncertainty even when
the gate passes. If retrieval meets the quality gate at lower cost, prefer it;
any reader advantage limited to particular question types needs a fresh held-out
set before defining an automatic routing rule.

## Qualification and delivery boundary

Targeted tests must cover real SQLite Media paging and revision changes,
production source/provider admission, worker requests without tools or unrelated
context, empty/foreign source scope, malformed results, Unicode/repeated quotes,
valid empty results, timeout/cancellation with a still-running adapter, output
fitting, logs without source canaries, and missing/late provider usage. Use real
SQLite and a request-recording local HTTP server through the real gateway to
prove request isolation, not only an injected adapter. The harness has no
production Console tool: catalog reachability, setting enforcement, UI dispatch,
and canonical citation persistence are explicit later integration gates.

No full suite is requested. The approved nonsensitive DeepSeek development run
is documented in the [live report](../../../backlog/docs/library-source-reader-deepseek-development-2026-09-08.md);
no private-data experiments have run. The draft was promoted to TASK-32029 and the fixture harness is
implemented under the [implementation plan](../plans/2026-09-07-library-source-reader.md).
Current qualification and remaining evaluation gaps are recorded in the
[local qualification report](../../../backlog/docs/library-source-reader-local-qualification-2026-09-08.md).
Shipping the Console tool, its Settings control, and citation-inspector wiring
is conditional on that decision and needs a separate atomic task.

ADR required: yes

ADR path: `backlog/decisions/133-question-directed-library-reading-experiment.md`

Reason: establishes a derived-evidence tool contract and provider, Library access,
and citation ownership boundaries; builds on ADR-024, ADR-029, ADR-030,
ADR-052, ADR-130, and ADR-131.

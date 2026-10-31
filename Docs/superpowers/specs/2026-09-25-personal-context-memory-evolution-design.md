# Personal Context memory evolution design

Date: 2026-09-25
Status: First-release implementation complete; pinned-dev integration under review. Later designs remain gated.
Foundation: TASK-25907
ADR required: yes
ADR path: backlog/decisions/182-personal-context-memory-evolution.md (Accepted)
Reason: Define memory ownership and first-release read surfaces across existing
profile, Notes, and conversation boundaries. ADR-102 continues to govern
canonical profile authority and encryption.

Tracker: [Memory roadmap](../../../backlog/docs/personal-context-memory-roadmap.md)
Decision: [ADR-182](../../../backlog/decisions/182-personal-context-memory-evolution.md)
Review: [Pre-implementation findings](../reviews/2026-09-25-personal-context-memory-preimplementation-review.md)

## Intent and approval boundary

The user wants to apply useful ideas from the supplied Muse diagram to the
existing memory/profiling system, and has requested a plan and tracker. The
goal is more trustworthy and useful memory: people can inspect what is
remembered, understand its effect on a response, retrieve relevant facts, and
eventually correct or forget information across its derivatives.

The user endorsed the direction, requested a technical review, and selected
native execution. This document incorporates that review and records later
design work. The offline baseline and first-release provenance, Next Send
selection, and local retrieval slices are implemented; the tracker links their
plans, native verification, and reviews. The pinned-dev integration is undergoing
qualification. Later schema, forgetting, disclosure, consolidation, and repair
designs do not authorize runtime activation or change existing permissions.

## Original inspected baseline and evidence

This table records the local checkout inspected for the original design, before
the first-release implementation. It is not a live audit of Muse or the companion
server, or a current implementation inventory. The tracker records subsequent
delivery; future implementation plans must recheck these owners before coding.

| Capability | Originally inspected owner and limit |
| --- | --- |
| Typed facts, record versions, controls, and source metadata | `packages/tldw_profile_core/src/tldw_profile_core/models.py`; references and hashes are not a complete versioned citation contract |
| User-authorized mutation and read views | `tldw_chatbook/Personal_Context/service.py`; Settings inspection and agent eligibility are separate concerns |
| Learning proposals and corrections | `tldw_chatbook/Personal_Context/proposal_service.py`; pending proposals stay outside context; direct update requires current-message evidence but substring presence does not prove entailment |
| Search/get tools | `tldw_chatbook/Agents/profile_tool_provider.py`; search currently matches a substring of serialized records |
| Context selection | `tldw_chatbook/Personal_Context/context_service.py`; workspace overrides, fixed priority groups, whole-record packing, 12 KiB and ten-percent token limits |
| User review | `tldw_chatbook/Widgets/Settings_Widgets/personal_context_panel.py` and `personal_context_review_modal.py`; existing controls are the canonical destination |
| Request use | `tldw_chatbook/Chat/console_chat_controller.py` and `console_agent_bridge.py`; a frozen snapshot is used for an agent run tree |
| Procedural memory | `tldw_chatbook/Notes/agent_lessons.py` and `Agents/agent_lesson_promotion.py`; ordinary Notes with review and independent verification requirements |
| Record deletion | `Personal_Context/repository.py`; older record and pending outbox bodies are retired, with a separate bounded Undo lifecycle |

Source IDs are opaque. A reference created on another device or database
cannot safely be opened against whichever local database is active. A span
hash cannot reconstruct the original quotation. A source marked as an
explicit statement is a recorded assertion about provenance, not a new
verification result.

## Alternatives and selected scope

1. **Inspect and improve existing memory first — recommended.** No new
   canonical fields or external infrastructure. It yields visible benefits
   while measuring the next problems.
2. **Evidence schema first.** It enables verified quotations sooner, but
   requires shared-core versioning, source authority, migration, retention,
   and companion-server coordination before user-visible delivery.
3. **Consolidation first.** It creates more memory before source support,
   disclosure, and forgetting are adequately defined. This is deferred.

The proposed first release uses option 1. Later work retains option 2 as a
separate design; option 3 depends on those safety and ownership contracts.

## First-release deliverables

### A. Synthetic baseline

Use synthetic fixtures only, through real PersonalContextService and tool
entry points with temporary encrypted repositories and in-memory key
protectors. Exercise automatic snapshot selection through its production
builder. Do not seed real user profiles or make a provider call.

The fixture matrix includes ordinary lexical queries, semantic paraphrases,
global/workspace exceptions, changed preferences, unsupported conclusions,
quoted instructions, absent evidence, archived/deleted/expired/private
records, and unrelated scopes. Label the desired eligible record IDs and the
expected evidence availability independently of the implementation.

Keep three labels separate: records the caller is authorized to see, records
relevant to the query, and records expected in a budgeted context snapshot
after workspace overrides and hard priorities. Search reports precision@K,
recall@K, and reciprocal rank over independently labelled relevant eligible
records, with K fixed in the fixture manifest. Empty relevant sets report
false-positive/empty-result behavior rather than a fabricated recall score.
Automatic context reports expected selection and budget/override behavior;
it is allowed to include standing constraints that do not match the query.

Freeze development and held-out regression cases before changing retrieval.
Report per-case and per-category results rather than only one average. Current
lexical misses are baseline observations, not failing feature claims disguised
by weakened assertions. Unauthorized disclosure is a separate hard failure,
even if aggregate recall improves. Successful authorized controls accompany
negative cases so an unrelated exception cannot pass the privacy checks.

Evidence-status checks describe stored metadata only. Until the provenance
projection exists, UI evidence status is explicitly unmeasured; the baseline
does not invent a second production explanation implementation. Unsupported
inferences and quoted instructions are labelled scenarios, not proof that a
no-provider harness evaluates semantic support or generated answers. Provider
answer evaluation is outside this first release. LongMemEval's temporal
updates and abstention categories are useful inspiration; no claim of
benchmark parity is made.

### B. Inspect existing provenance

Extend the canonical My Profile record detail and existing proposal review
with a read-only provenance section. Present the source type, recorded actor
and reason, creation/update times, current and parent version identifiers,
promotion origin where present, and whether source references were retained.
Distinguish pending proposal provenance from recorded user approval.

Use explicit evidence copy:

- **No source reference retained** when none exists.
- **Legacy source reference — quotation not verified** for current opaque
  references, even if a hash is present.
- **Record changed; reload details** when the selected version no longer
  matches the captured detail request.

Do not fabricate confidence for accepted records: V1 proposal confidence is
not a durable per-record confidence field. User approval or editing does not
prove that the original referenced text supports the final wording.

These fields are recorded provenance, not an audit history of the current
wording. Accept and accept-edited share approval metadata; ordinary Settings
edits preserve earlier provenance. Show **Edit history not recorded** rather
than infer whether an approval was edited, whether an agent claim was inferred,
or who authored the current words. Parent-version and promotion IDs do not
promise that earlier content still exists. Deleted records expose retained
tombstone metadata only; do not reconstruct retired content through Undo or
source lookup.

Existing source references may be disclosed as metadata to the user in
Settings, but they are not clickable source locators in this release. There
is no filesystem, cross-workspace, or network resolution. A later evidence
contract can add a verified source view without changing the meaning of these
legacy states. Render all imported or user-controlled metadata as bounded
literal text, without Rich/Markdown interpretation, terminal controls, or
automatic links.

UI consumes an immutable, purpose-specific projection produced through the
owning service. It does not inspect encrypted tables. Selection changes and
late worker results are fenced by profile generation, operational availability,
record/proposal identity, and version. Locking, replacing, or removing a profile
clears previously rendered details and rejects late results.
Inspection must not modify proposal acceptance, controls, or stored content.

### C. Explain selection in Next Send

Produce a transient selection explanation from the same candidate traversal
that builds the exact context snapshot. Keep the existing public snapshot
and injection contract compatible; do not perform a second independent
selection just to explain the first.

For candidates that already pass agent eligibility, explain:

- selected, with its existing priority group;
- suppressed by a keyed workspace override;
- omitted because the whole record exceeds the remaining byte or token
  budget.

Explain disabled or locked state only when a typed, content-free outcome from
the same authorized build establishes it; an empty snapshot alone does not.
Unknown build failures produce a generic unavailable state. Show empty or
insufficient-budget results without leaking hidden record counts. Priority
describes ordering, not an independent omission reason: a skipped oversized
record does not prevent a later smaller record from fitting. Archived,
expired, conflicted, user-only, and unrelated-scope
records do not appear as named rejected candidates. Their own lifecycle can
be inspected through the user-owned Settings surface instead.

The existing serialized snapshot has an `unsupported_records_present` flag
derived from an unscoped quarantine inventory. New diagnostic projections must
not copy that flag, infer scope from it, or expose quarantine counts. Its
pre-existing appearance in the model block is a compatibility/privacy gap for
the provider-disclosure design to resolve. Preserving the injection contract
does not prove the existing model block has no existence signals.

Next Send shows a disposable explanation of the preview revision, not a promise
about a later send after inputs change. Bind the snapshot and diagnostics to
one request-owned result: session/workspace, draft, resolved provider/model,
available budget after request reservations, profile generation/revisions,
and eligibility time. The existing snapshot `cache_key` omits several of those
inputs and must not be used alone to reuse diagnostic results. No extra
persistent digest of private prompt content is needed.

Rebuilding or invalidating the preview clears the old explanation before
replacement work begins. Recheck the captured owner and live availability
before publishing a worker result; drop late results after navigation, profile
lock/removal, record or authority change. An expiry boundary also invalidates
eligible diagnostics even when revisions are unchanged. The implementation
plan must name the actual invalidation events and bounded expiry mechanism;
do not add continuous repository-decryption polling. Parity is assessed with
identical request inputs and a fixed clock. Persisted chat,
model prompts, debug logs, exports, and Sync receive none of this diagnostic
content. Keep diagnostics outside `next_send_payload` and other generic
serialization/export paths; suppress their representation in debug output.
This restriction governs the new diagnostics, not a claim that existing
provider payload export already removes Personal Context. Existing run-tree
pinning is unchanged; mid-run revocation semantics
belong to the later forgetting/disclosure designs.

### D. Field-aware lexical recall

Replace searching serialized model JSON with matching the semantic key,
payload subject, and actual human-readable payload values. Use bounded,
Unicode-aware case folding and tokenization shared by search and the context
selector's relevance predicate. Specify punctuation behavior with fixtures,
including technical names and international text.

Only records with a positive content match enter search results. Rank them
deterministically by distinct matched query terms, subject matches, and exact
normalized phrase match, with stable identity tie-breaking. Matching any term
qualifies a candidate; matching more distinct terms ranks higher. Phrase
matching stays within an individual human-readable field, never across joined
fields. Queries with no usable terms or no matching records return no matches.
The executable plan must freeze normalization, punctuation handling, per-input
work bounds, and fixtures for C/C++/C#, .NET, accented text, CJK text, and
single-character terms before implementation. Do not promise language-aware
segmentation or semantic equivalence from Unicode tokenization alone.

This is lexical retrieval:
semantic equivalents such as "short answers" and "concise replies" may still
miss unless they share terms. Record that limitation rather than adding a
hand-maintained synonym dictionary or claiming semantic search.

Preserve existing search limits and live eligibility checks. Search may
return eligible global and workspace records for inspection; automatic
context continues to apply its own workspace override and hard priority
rules. Do not convert relevance into permission or allow recency to displace
constraints. No persistent plaintext index, embeddings, network call, or new
dependency is introduced.

Reuse the current authorized candidate view; do not add source resolution,
new cross-workspace reads, or extra full-store passes. The owning service
currently reads an export snapshot before scope filtering, so this release
does not claim to eliminate existing whole-repository reads. Measure retrieval
cost on bounded synthetic profiles and preserve worker ownership for slow work.

## UI and interaction constraints

Use the existing canonical Settings panel and proposal modal; no new Settings
destination or legacy parallel surface. Next Send uses its existing inspector
region. Provenance is a stacked read-only section using `$ds-space-stack`,
`$ds-space-section`, existing section labels and muted secondary text. Unknown
evidence is informational, not a destructive-action warning.

Interactive controls retain distinct rest, hover, focus, and disabled states
from ADR-150. Keyboard focus remains visible; footer hints name implemented
actions only. Reflow and scroll within existing containers at narrow terminal
sizes. Use token-backed classes; if a source stylesheet changes, rebuild the
generated CSS bundle. The design does not change existing token values.

## File ownership and verification map

Exact implementation interfaces are established in the executable plans after
design review. These are the existing owners each slice must extend:

| Slice | Production owners | Targeted evidence |
| --- | --- | --- |
| Baseline | Existing service, tool, and context entry points; synthetic fixture/runner under `Tests/Personal_Context/` | Offline reproducibility, independently labelled expected IDs, no provider calls |
| Provenance | `Personal_Context/service.py`, `Widgets/Settings_Widgets/personal_context_panel.py`, `personal_context_review_modal.py` | `Tests/Personal_Context/test_service.py`, `Tests/UI/test_settings_personal_context.py`, `Tests/UI/test_personal_context_review_modal.py`, mounted provenance cases |
| Selection explanation | `Personal_Context/context_service.py`, existing Console snapshot/Next Send owners | `Tests/Personal_Context/test_context_service.py`, `Tests/Chat/test_console_personal_context_snapshot.py`, mounted inspector parity cases |
| Retrieval | `Agents/profile_tool_provider.py`, `Personal_Context/context_service.py`, one shared local matching helper if needed | `Tests/Agents/test_profile_tool_provider.py`, context tests, synthetic recall comparison |

Maintain existing plaintext-canary and authority tests on affected paths.
Canonical compatibility fixtures must remain unchanged in the first release.
Run the design-token governance test if UI/CSS changes. Exercise actual
mounted controls and prepared requests, not just string-rendering helpers.
Operations exceeding 100 ms use existing worker patterns. Profile before
adding a cache; this work must not multiply repository reads on every send.

## Later design work

The roadmap separately tracks evidence/temporal contracts, cross-owner
forgetting, provider disclosure, consolidation, and feedback/repair tracking.
Their acceptance criteria define design deliverables, not implementation
completion. They must address the following boundaries:

- Source identity includes authority and version; quotes, attachments,
  hypotheses, and tool output are not direct user instructions.
- Validity dates, corrections, inference confidence, approval, and salience
  remain different concepts; newer text is not automatically more true.
- Forgetting governs derivatives, Undo, concurrent jobs, retained sources,
  source suppression, and remote acknowledgement. Existing exports outside
  managed custody cannot be promised erased.
- Device-only syncability does not imply local-model-only disclosure.
  Policies cover tools, fallbacks, child runs, interviews, embeddings, and
  maintenance as well as initial prompt injection.
- ADR-102's phrase "device_only records never leave Chatbook" is broader than
  the inspected request/tool filtering, which checks visibility rather than
  syncability. This is an unresolved policy/implementation discrepancy, not a
  new permission granted by this design. Provider-disclosure work must resolve
  that discrepancy explicitly; the first release cannot claim local-only
  model disclosure or complete end-to-end forgetting.
- Consolidation is opt-in and proposal-only, runs only on new eligible
  signal, respects budgets, and cannot approve itself or infer psychological
  traits silently.
- Feedback is evidence of a specific interaction. Approved preference changes
  use Personal Context; independently verified procedures use Agent Lessons.
  Narrative reflections do not establish truth or tool permission.

Completing design tasks is not sufficient to launch consolidation. Any later
implementation must depend on shipped and verified evidence, suppression,
disclosure, and recovery controls. Record those implementation dependencies
when their concrete tasks exist; do not point at future placeholder IDs.

## Definition of first-release success

A user can inspect what is actually known about a memory's provenance,
understand why an eligible record entered the next request, and retrieve
records more reliably on the declared lexical cases. Privacy, canonical
bytes, runtime authority, whole-record budgets, and existing mutation
behavior remain covered by targeted evidence. No unknown quotation is
presented as verified and no new background provider activity is enabled.

## Review record

- [x] Current owners and relevant ADRs inspected.
- [x] Existing TASK-25907 owner direction preserved.
- [x] First-release scope separated from later schema and lifecycle designs.
- [x] Self-review completed for ownership, unknown evidence, privacy, scope, and acceptance coverage.
- [x] User endorsed the direction and requested review before continuing.
- [x] Pre-implementation review corrections recorded in the design and tracker.
- [x] Executable plans for the approved first-release tasks are reviewed; linked from the [roadmap](../../../backlog/docs/personal-context-memory-roadmap.md).
- [x] Runtime implementation began under those approved plans; TASK-25907.1 through TASK-25907.4 are complete.

These historical milestones do not close the reopened pinned-dev integration
review or authorize any later design-only capability.

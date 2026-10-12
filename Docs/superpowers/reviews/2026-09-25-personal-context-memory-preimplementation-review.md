# Personal Context memory plan: pre-implementation review

Date: 2026-09-25
Scope: written design, ADR-182 and TASK-25907 tracker versus the local working
tree. No application code, real profiles, provider calls or background jobs
were changed. The checkout contains unrelated uncommitted work; this is a
source review, not qualification of the entire branch or companion server.

Design: [Memory evolution](../specs/2026-09-25-personal-context-memory-evolution-design.md)
Tracker: [Memory roadmap](../../../backlog/docs/personal-context-memory-roadmap.md)
Decision: [ADR-182](../../../backlog/decisions/182-personal-context-memory-evolution.md)

## Result

The bounded first release is ready for executable planning after the corrections
below. It retains the existing memory owners and requires no canonical schema
change. A separate read-only reviewer confirmed that the provenance and privacy
contradictions were corrected and found no material new contradiction in the
revised design and task criteria. This does not mean the two pre-existing
runtime privacy gaps are fixed.

The next delivery step is the offline baseline in TASK-25907.1, followed by
truthful provenance and selection inspection and measured lexical retrieval.
Provider-disclosure design remains high priority; richer citations, forgetting
and consolidation are separate contracts.

## Findings and dispositions

### 1. Important: provenance cannot reconstruct unrecorded history

The original acceptance criteria required distinct inferred/approved/edited
states that existing fields cannot reliably establish. Both unchanged and
edited proposal acceptance use the same approval provenance. Ordinary Settings
edits preserve earlier provenance, and resolving proposals removes their
original proposed content. A source reason can therefore predate the current
wording.

Evidence: `Personal_Context/proposal_service.py:326-359,425-431`,
`Personal_Context/service.py:1183-1195`,
`Personal_Context/repository.py:3305-3331`, and shared-core
`models.py:76-82`. Application paths in this report are relative to
`tldw_chatbook/`; the shared core is under
`packages/tldw_profile_core/src/tldw_profile_core/`.

**Corrected:** the design and TASK-25907.2 require truthful unknown edit/inference
history, recorded metadata labels, no historical-body reconstruction, and
literal rendering of imported text. Parent-version identifiers do not promise
a readable history. Current wording is never independently verified by an
approval flag.

### 2. Important: diagnostic freshness needs more than record revisions

`ProfileContextSnapshot.cache_key` contains profile generation/revisions and
scope, but excludes draft text, provider/model, budget and time. Expiry can
change eligibility without changing any revision. The current builder also
maps every exception to an empty snapshot, which cannot establish a specific
disabled/locked reason.

Evidence: `Personal_Context/context_service.py:71-77,92-102,110-124` and the
current Next Send load/assignment in
`Widgets/Console/console_conversation_inspector.py:1976-2012`.

**Corrected:** TASK-25907.3 requires one request-owned snapshot/explanation,
identical-input-and-clock parity, explicit availability/expiry invalidation,
late-result rejection and content-free typed outcomes. Old diagnostics clear
before replacement. The executable plan must identify actual invalidation
events and an expiry mechanism; no polling/decryption loop is implied.

### 3. Important: device-only policy and observed disclosure differ

ADR-102 says device-only records never leave Chatbook, while the inspected
context/tool candidate filters check agent visibility, lifecycle and expiry,
not sync mode. The context block removes controls while serializing payloads,
so that block alone cannot enforce a downstream per-record device-only policy.

Evidence: [ADR-102](../../../backlog/decisions/102-personal-context-profile-authority-sync-and-encryption.md),
line 82; `Personal_Context/context_service.py:111-124,224-235` and
`Agents/profile_tool_provider.py:330-350`.

**Documentation corrected; runtime gap remains open:** ADR-182 records the
discrepancy without granting new access or silently weakening ADR-102.
TASK-25907.7 must explicitly implement or amend the governing promise. The
first release does not claim local-model-only disclosure.

### 4. Important: existing unsupported-record metadata is unscoped

The service builds `unsupported_records_present` from the entire quarantine,
then the serializer places it in the model block, even when no ordinary record
is selected. Exact preservation of the existing snapshot cannot establish a
universal absence of inaccessible-record existence signals.

Evidence: `Personal_Context/service.py:1659-1665,1689`,
`Personal_Context/context_service.py:239-243,286-288`, and the existing expected
behavior in `Tests/Personal_Context/test_context_service.py:340` (test path is
repository-relative).

**Documentation corrected; runtime gap remains open:** new diagnostics must
not copy or reinterpret unqualified quarantine metadata. TASK-25907.7 owns
the compatibility/disclosure decision for the current injected flag. This
review does not characterize the whole existing model payload as leak-free.

### 5. Improvement: evaluate relevance independently of eligibility

The initial baseline did not define ranking denominators or distinguish search
relevance from automatic standing guidance. It also included optional answer
evaluation in an otherwise offline first slice. An authorized but irrelevant
record could appear to improve recall, while an empty result caused by a broken
caller could appear privacy-safe.

**Corrected:** TASK-25907.1 separates eligible records, relevant records and
expected context selection. It specifies fixed-K precision/recall and reciprocal
rank, false positives for empty relevant sets, frozen development/held-out cases,
per-category reporting and successful controls for negative checks. Generated
answers and semantic support remain unmeasured. Existing disclosure failures
must remain visible as failures; a baseline report is not proof of a clean
privacy audit.

### 6. Improvement: bound retrieval claims and explain actual packing

The search proposal left zero-match behavior ambiguous and the task's “hidden
global scan” wording obscured an existing repository-wide read. Priority itself
is not an omission reason: the packer skips a large record and can still include
a smaller later record.

Evidence: `Agents/profile_tool_provider.py:353-365`,
`Personal_Context/service.py:1560`, and
`Personal_Context/context_service.py:260-288`.

**Corrected:** TASK-25907.4 requires positive content matches, deterministic
multi-term ranking and same-field phrase matches. The executable plan must
freeze Unicode/punctuation rules and input bounds, including technical names
and single-character queries. No new cross-scope read or full-store pass is
introduced; existing read cost is measured rather than claimed absent.
TASK-25907.3 explains priority order and actual override/budget omissions.

### 7. Improvement: completed designs are not deployed safeguards

Consolidation depended on evidence, forgetting and disclosure design tasks.
Those dependencies alone would not make a later background job safe to enable.

**Corrected:** TASK-25907.8 and the roadmap require shipped and verified
evidence, suppression, disclosure and recovery controls before enabling an
implementation. Concrete implementation dependencies will be filed only when
their tasks exist. No job is enabled by this roadmap.

## Verification and limits

- Scoped validation passed for 14 documents/task files: 10 unique task IDs,
  59 child acceptance criteria, backward-only dependencies, 31 local Markdown
  links, exact CLI-written criteria, status boundaries, whitespace and conflict
  markers. The repository Backlog guard passed for this ten-task family.
- A fresh scan of active/completed/archived task buckets found no collision for
  this family. The global inventory still contains 38 unrelated duplicate-ID
  groups; the global guard is not claimed clean.
- All nine follow-up tasks remain To Do; TASK-25907 remains In Progress.
- No runtime tests, full-suite run, model evaluation, real-profile inspection
  or companion-server verification were performed for this documentation change.
- Mounted layout, conflict population, invalidation races, provider fallbacks
  and downstream egress enforcement remain implementation/qualification work.
- The proposed positioning ADR does not settle future schema or permission
  changes. No runtime privacy fix is claimed by updating a design task.

# Workspace project context: pre-implementation review

Date: 2026-10-20
Reviewed baseline: `e9fd86c2d7b18538ba1bd32b9e498aea9a9b9110`
Scope: approved design, ADR-240 and the three implementation plans against the
current repository source. Three read-only subagents independently reviewed
guidance/citations, index authority/lifecycle, and semantic adapters/generations.
The primary agent verified the reported code paths and reconciled corrections.

This is a design/plan review. No product implementation or runtime qualification
was performed. Findings below are addressed in the documents, with regression
evidence required during implementation; they are not claims of fixed software.

## Findings and disposition

### R1 [P1] Missing-marker repair could fabricate Library attribution

Canonical Console capture constructs a repair contract in
`Event_Handlers/Chat_Events/chat_rag_events.py:1801`. The current repair decision
requires markers whenever a contract exists (`Chat/citation_repair.py:189-190`),
including a correct workspace-only answer with unrelated staged Library evidence.
Formatter wording alone cannot prevent a repair request from adding `[S1]`.

**Correction:** G2 adds runtime-only `require_markers=True` to the immutable
contract and opts out only at canonical Console construction. Generic local
capture keeps its default; actual invalid markers remain invalid in initial and
repaired output. Test real direct/agent completion, no repair call for a file-only
answer, removal of `[S999]`, and unchanged exact capture identity/snapshots.

### R2 [P2] The handoff-helper test did not cover production delivery

`ChatHandoffPayload.model_context_block()` has no production consumer. Real
handoffs stage through `UI/Screens/chat_screen.py:16358-16452` and canonical
capture; source-only canonical formatting does not contain the strict helper
instruction (`RAG_Search/local_citation_capture.py:473-541`).

**Correction:** Remove the unused helper change from G2 and test actual handoff
staging, capture and dispatch. Any Console instruction goes outside the exact
captured source blocks.

### R3 [P2] Guidance preceded the last tool narrowing

`Chat/console_agent_bridge.py:5078` renders the note before automatic goals narrow
the config at `5138-5142`. A goal could remove every workspace reader while the
note still promises project exploration.

**Correction:** G1 renders/recomputes the note after all narrowing and qualifies
the final request with a goal that excludes readers.

### R4 [P1] Fresh reads did not serialize cross-process revocation

Registry mutation and cache publication use different databases. A revocation
could commit after an authority read and before a prepared batch publishes.
`Workspaces/registry_service.py:691-693` explicitly limits its existing mutation
counter to one process; SQLite transactions serialize only their own owner.
One app-owned writer also does not prevent another process claiming the same job.

**Correction:** K1/K3 add a durable registry authority revision/tombstone, a short
registry-then-store fence, and transactional active-job ownership/fencing. Test
two processes with the builder held after its authority read. K2 refreshes
authority/exclusions at every bounded request/read admission instead of keeping
initial exclusions for a long-lived scanner. Already admitted reads may finish;
their results still need current publication/disclosure admission. No fence is
held over worker joins, scans, network work or cleanup.

### R5 [P1] Search lacked the requesting run's exclusion floor

The proposed `search(scope, query)` knew only cache scope and current binding
exclusions. Existing file authority unions frozen/accumulated run exclusions
with live exclusions (`Tools/workspace_file_roots.py:845-849,910-921`); removing
an exclusion must not widen an already admitted run.

**Correction:** K4 receives code-issued per-call search authority and its admitted
executor. Both keyword and semantic selection apply that floor before top-k.
Test exclusion removal, fresh refresh, then old-run search. Semantic search
falls back to keyword when it cannot express the exact floor with bounded
backend filtering; do not allocate a tree-sized allowed-path list or post-filter
a global semantic top-k. Tool coverage counters must also be scoped or omitted;
unrestricted Settings totals cannot enter an older, narrower run.

### R6 [P2] Delayed management actions could undo newer choices

K5 fenced returned callbacks but K3's original mutators accepted only key/job ID.
An Enable queued before Clear could execute afterward and recreate enablement;
an old Clear could similarly remove a newly enabled index.

**Correction:** Require expected scope and, for job actions, job revision/state
before effects. Test both orderings through the service and mounted controls.
Semantic Disable also checks the binding epoch.

### R7 [P1] A semantic backend error looked like zero matches

`RAG_Search/simplified/vector_store.py:591-594` catches native query exceptions
and returns `[]`, so S2 could not produce its promised truthful failure state.

**Correction:** S1 adds opt-in strict query behavior with unchanged defaults.
S2 distinguishes successful zero hits from unavailable and uses keyword fallback.
Test failure and recovery through the actual collection query, not only a fake
service returning an error flag.

### R8 [P1] Semantic paging lacked a stable keyword source revision

Keyword Refresh mutates chunks without changing the authority epoch/processing
revision. The original S2 page contract had no manifest witness, so pages could
mix source memberships and still publish Ready.

**Correction:** K1 increments a manifest revision with membership publication.
S2 uses bounded keyset pages checked against one captured revision, carries it
through checkpoints, and compares it at final ready-pointer CAS. Test mutations
between pages and immediately before publication. Keep no SQLite read snapshot
open across embedding/network work.

### R9 [P1] Direct/native effects bypassed the proposed scope seam

Handling only `_call_scope` and `generation.query_scope` misses
`activation.native_worker`'s direct execution (`activation.py:251-262`) and lazy
native client/collection creation (`simplified/vector_store.py:305-310,342`).

**Correction:** S1 specifies one exact project carrier across decorated, direct,
lazy and transferred-native entrances. Test actual async embedding and initial
collection construction, copied/dead context, cancellation and real retirement.
Existing ancestor-liveness, identity, storage and native-borrower checks remain.

### R10 [P1] Ordinary local-model loading could still download

The wrapper sets offline loading only for a recovered binding
(`simplified/embeddings_wrapper.py:347-359`); the underlying ordinary HF config
defaults to `local_files_only=False` (`Embeddings/Embeddings_Lib.py:251`).

**Correction:** S1 specifies an explicit project-only offline loader option and
complete local snapshot requirement. Test an ordinary isolated profile with a
missing model and blocked network; retain existing callers' defaults.

### R11 [P1] Sanitizing Settings did not sanitize factory logs/errors

Raw custom endpoints are logged by the wrapper at
`simplified/embeddings_wrapper.py:570-573` and by the actual factory's success,
retry and failure paths at `Embeddings/Embeddings_Lib.py:556-610`.

**Correction:** Include the factory in S3's file scope. Use credential-free
destination labels and classified errors, covering actual controlled HTTP
success/retry/failure, captured logs and propagated exception/status text.

### R12 [P2] Shared-guard changes omitted existing recovery regressions

The original S1/S2 test lists omitted focused restored-query, native-lifetime
and model-loading controls despite changing their shared guard seams.

**Correction:** Add affected selections from
`Tests/Backup_Recovery/test_projection_generation.py`,
`test_rag_projection_lifetimes.py`, and `test_activation_local_embedding.py`.
Keep actual backend/lifetime qualification distinct from deterministic vectors.

## Delivery and verification

The agreed order remains [guidance](../../plans/2026-10-20-workspace-project-guidance.md),
[keyword indexing](../../plans/2026-10-20-workspace-project-keyword-index.md),
then [semantic indexing](../../plans/2026-10-20-workspace-project-semantic-index.md).
Saved Library choices and folder-bound cache ownership are unchanged. Guidance
does not wait for either index. The user selected subagents for execution.

[ADR-240](../../../../backlog/decisions/240-workspace-project-context-and-folder-bound-index.md)
and the [design](../../specs/2026-10-20-workspace-aware-project-context-design.md)
carry the corrected contracts. The pinned [Codex CLI source reference](../../specs/2026-10-20-codex-cli-project-context-reference.md)
remains a source-based comparison, not a claim that Codex supplies this durable
content/semantic indexing feature.

Review verification is source inspection, local-link/reference consistency and
diff/whitespace checks. Product tests, two-process races, 20k-item measurements
and real-model/backend journeys are planned evidence and remain to be executed.
All three reviewers confirmed their substantive findings were addressed in the
revised documents; final scanner callback/signature corrections were also applied.
Document checks cover seven documents, 25 local links and eight pinned source
anchors; the whitespace check passes. These are document checks only.

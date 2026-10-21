# ADR-240: Workspace project context and folder bound indexing

Status: Proposed
Date: 2026-10-20
Design: [Workspace aware project context](../../Docs/superpowers/specs/2026-10-20-workspace-aware-project-context-design.md)
Related Task: N/A (design requested before implementation task selection)
Related decisions: ADR-030 derived-index lifecycle, ADR-079 Console Library
authority, ADR-082 private scratch, ADR-101 pinned workers, ADR-102 run-admitted
roots, ADR-126 recovery, ADR-174 binding exclusions

## Context

Attaching a project folder grants file-tool access without staging its contents.
A natural project question can consequently receive an answer based on unrelated
Library excerpts when the model does not explore the folder. Existing workspace
notes explain root addressing but do not specify project exploration behavior.
One Console evidence formatter also supplies Library-only grounding instructions
that can compete with using file-tool evidence.

The user selected workspace-aware guidance while explicitly preserving saved
Library choices. They also requested optional indexing for thousands or 10k+
folders/items, choosing a folder-bound cache rather than retained Library copies.
This requires an ownership and revocation boundary distinct from Library ingest.

## Decision

### Workspace questions use capability aware guidance

Extend the existing model-facing workspace note after the final advertised-tool
filter. For project questions, guide the model to inspect relevant admitted
folders, README/manifests, and source as needed. Keep multiple-root selection and
scratch addressing explicit. The note grants no access and is never constructed
from the process-global active workspace in place of the owning run's authority.

Console grounding permits both Library references and workspace evidence with
distinct provenance. Dedicated Library answer generation retains its supplied-
evidence-only contract. An irrelevant Library bundle does not establish that
workspace files are unavailable.

No Library policy migration or automatic toggle occurs. Saved Never/Automatic,
Blocked/Allowed, manually staged sources, and scope intersection retain ADR-079
semantics, even when Automatic supplies unrelated evidence to a project turn.

### The optional index belongs to a local folder binding

Indexing is an explicit per-binding opt-in, disabled by default. A device-local
project-index service and separately versioned SQLite store own its manifest,
FTS, jobs/checkpoints, generations, and cleanup. Files remain authoritative;
cached text and vectors are derived. Root identity, locator fingerprint, local
data authority, binding identity, configuration, and index epoch scope readiness.
Enablement authorizes application indexing reads only within the live binding
and selected processing scope; it does not authorize assistant tool calls.

Project content is not persisted as Library Media/Notes, is absent from global
Library retrieval, and does not inherit Library access or item scopes. The cache
and its settings/consents are excluded from content sync/export. Rebuildable
cache data is excluded from ordinary backups and integrates with the existing
storage-admission, activation, maintenance, and borrower-retirement contracts.

Keyword FTS works without optional embedding dependencies. Semantic indexing is
a further opt-in using existing embedding/chunking/vector primitives with a
separate project namespace and lifecycle. Cloud embedding disclosure requires
index-specific destination consent; Library permission does not provide it.

### Discovery does not widen file authority

An optional read-only `fs_search_index` spec uses the existing local provider,
root aliases, permissions, review, budgets, and path-target preparation. It may
query only a compatible project generation for the selected run-admitted root.
Binding exclusions and authority filters apply before ranking and again before
disclosure. Validate each selected candidate against the live pinned file and
its content hash; return live excerpts rather than stale cache text. A denied
read, removed binding, or incompatible root cannot be bypassed through the cache.
The binding's effective `fs_read` permission is a narrowing floor for excerpt
delivery: Blocked refuses and Ask requires review even when index search is
otherwise Allowed.

Indexed text never becomes automatic project instructions. Search has the same
binding-root instruction scope as existing grep; exact follow-up file reads use
the current nested-instruction activation path. Library labels remain reserved
for their owning Library reference bundles.

### Builds are bounded jobs and revocation precedes cleanup

Incremental scanning uses bounded pages and queues with a durable frontier, not
a whole-tree list. A fixed-purpose contained scanner process lives for one job;
it reuses root-pinning/process-containment primitives and does not change the
one-shot agent file-worker protocol. The parent service owns transactional
publication and rechecks authority and epoch before each committed batch.

Keyword entries are eligible only after their manifest/text/FTS batch commits.
Semantic generations publish through an atomic compatible ready pointer. Partial
coverage, skipped processing-policy files, budget exhaustion, and interrupted
jobs are reported truthfully. Refresh and restart Resume are explicit in the
first release; no watcher dependency or scan-on-every-send is introduced.

Disable, Clear, detach, retarget, root replacement, and data-authority changes
revoke query/publication eligibility before retryable physical cleanup. New
exclusions narrow access immediately at the next query/publication boundary.
An old job cannot resurrect cleared or revoked data. Detachment cannot retract
tool output already disclosed into a conversation or provider request.

The first release indexes local UTF-8 project text. Remote/SSH indexing and
binary/media extraction require separate designs. Settings owns management;
Console shows status and a route to Settings. Library controls stay distinct.

## Alternatives considered

| Alternative | Reason not selected |
| --- | --- |
| Change saved Automatic/Allowed choices when a folder is attached | Conflicts with the user's explicit requirement and ADR-079 ownership. |
| Add a deterministic project-versus-Library intent router | Adds classification/state behavior before proving existing controls plus guidance insufficient. |
| Automatically ingest the project as Library Media | Gives retained Library records different lifetime and authority from their folder binding. |
| Reuse the shared Library RAG service as the project index | Couples Library source/activation contracts to live filesystem authority and risks cross-source retrieval. |
| Load every file or a complete directory tree into each prompt | Fails the requested scale and adds unnecessary context, I/O, and disclosure. |
| Add a resident filesystem daemon and mandatory semantic stack | Keyword indexing and explicit incremental jobs meet the initial need with fewer dependencies and lifecycle owners. |

## Consequences

Guidance can ship independently and works without an index. Saved Library choices
continue to affect pre-send retrieval; the model must evaluate relevance rather
than treating their presence as exclusive project evidence. Guidance improves
behavior but cannot guarantee every model's tool selection, so real-model
qualification is required.

Indexing adds a versioned storage owner, explicit job recovery, a tool contract,
and revocation tests. Its storage and semantic cost are bounded and optional.
Current-file validation adds query latency, but a fast stale cache cannot become
an alternate disclosure path after binding access changes.

Implementation task records and plans must link this ADR, preserve its proposed
status until design approval, and include storage/recovery qualification before
the index is enabled in production.

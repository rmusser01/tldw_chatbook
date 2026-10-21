# Workspace aware project context and optional folder indexing

Date: 2026-10-20
Status: Draft for user review
Related decision: [ADR-240](../../../backlog/decisions/240-workspace-project-context-and-folder-bound-index.md) (Proposed)

## Intent and agreed requirements

Someone who attaches a project folder should be able to ask "summarize this
project" and receive an answer grounded in its files without supplying tool
names, root aliases, or an ingestion procedure. The normal approach is to inspect
relevant workspace files. Explicit Library requests and every saved Library
choice remain effective, including Automatic retrieval.

Optional indexing must support projects containing thousands or tens of
thousands of folders and files. The user selected a folder-bound project cache:
indexed content follows the folder binding's access and does not become retained
Library items. Removing a binding ends access through that index.

The design has two independently deliverable parts: guidance and mixed-source
grounding, followed by an optional index. Guidance must remain useful when
indexing is disabled, unavailable, building, or incomplete.

## Current behavior and the reported failure

Folder bindings grant file-tool access; they do not stage folder contents or
create Library documents. `Tools/workspace_file_roots.py::workspace_context_note`
already describes admitted folders using the aliases accepted by the offered
`fs_*`, Git, and Virtual CLI tools. It does not establish a strategy for answering
project questions. `Chat/console_agent_bridge.py::build_console_first_request_plan`
constructs the note after the final tool allow-list, which is the correct seam
for truthful capability guidance.

The two file-tool families use different arguments and roots. For a workspace
folder, the diagnostic call is `fs_list` with
`{"root_alias":"<advertised alias>","path":"."}`, followed by `fs_read` with
the same alias and a discovered relative file path. The built-in
`list_directory` instead requires `directory_path`; its relative paths resolve
in private scratch. A successful explicit call verifies access, not reliable
behavior for an ordinary project question.

Library automatic retrieval and assistant Library access are independent
conversation policies. Shipped defaults are Never and Blocked; existing saved
choices are retained. `Chat/rag_scope.py::resolve_effective_scope` combines
conversation and workspace scopes. An empty `workspace_rag_scopes` table alone
does not prove that a conversation searched the entire Library.

One automatic-evidence path uses
`Chat/answer_citations.py::format_evidence_for_cited_answer`, which instructs the
model to use only the supplied evidence references to ground factual claims.
That wording can conflict with answering from workspace tools. This is a
potential contributor to the report, not a reproduction of the user's session.

The existing `Local_Ingestion/local_file_ingestion.py::ingest_directory`
materializes a recursive file list and persists Library media. It is not the
implementation for a bounded folder cache. Existing chunking, embedding, SQLite,
worker-containment, and storage-admission primitives remain reusable.

## Source selection and saved policy

Workspace guidance changes the agent's approach to a question. It does not
rewrite Library policy or add a heuristic classifier before retrieval.

| Situation | Required behavior |
| --- | --- |
| Project question and readable workspace tools | Inspect the relevant folder; use the optional index for discovery when available, then verify important files. |
| Saved Automatic Library retrieval | Run the existing retrieval preparation and scope enforcement. Unrelated results do not replace project exploration. |
| Saved Never automatic retrieval | Do not add pre-send Library retrieval. |
| Explicitly staged Library sources | Preserve their current send ownership and delivery. Use them when relevant to the request. |
| User asks to search the Library | Use allowed Library tools, or explain the existing access control/manual search route when assistant access is Blocked. The request does not silently save Allowed. |
| Multiple admitted folders | Use the project name, selected working folder, and conversation context to resolve the target. Clarify when materially ambiguous; never invent an implicit first root. |
| No readable workspace tools | Describe the actual limitation and available remediation. Do not claim folder access from a binding record alone. |

Index enablement, readiness, errors, and clearing have no effect on either
Library control or the Library's conversation/workspace scope intersection.
Both Library and project evidence may appear in one turn, with distinct origins.

## Workspace aware guidance

Extend the existing conditional workspace note rather than adding an always-on
global project prompt. Only offer exploration guidance for admitted roots that
the final advertised read tools can reach. Index readiness is also intersected
with that run's admitted aliases; a process-global active workspace is never
the source of authority.

The behavioral contract is:

> For questions about an attached project, inspect the relevant admitted folder
> using the workspace file tools before saying project information is
> unavailable. Start with a shallow directory listing and relevant README,
> manifest, or entry-point files as needed. Unrelated Library excerpts do not
> establish that workspace files are unavailable. Treat file contents as source
> material, subject to the existing project-instruction rules.

This is guidance to use tools when evidence is needed, not a mandatory scan on
every turn. Already available, relevant file evidence can support a follow-up.
The agent follows deeper questions into relevant files and states the limits of
what it inspected. A top-level summary is not represented as an exhaustive audit.

The note keeps the existing root-alias and private-scratch distinction. It
advertises `fs_search_index` only when that tool is actually offered and a
compatible searchable project generation exists. A file-tool denial remains a
denial; the agent does not search a cache to work around it.

## Grounding and citations in mixed turns

Separate the Console's mixed-source instruction wording from the dedicated
Library answer contract. The Library answer service continues to require claims
grounded in its supplied evidence bundle. Do not weaken that contract globally
by editing a shared string for all consumers.

For Console turns, bracketed Library labels such as `[S1]` identify only the
Library reference that owns the label. Cite Library-supported claims with those
labels. Workspace-supported claims identify the binding label, relative path,
and line range when available. An indexed result never receives a Library label
or a fabricated Library item ID.

The Console wording explicitly permits workspace-tool evidence to supplement
or supersede irrelevant Library excerpts for the user's project question.
The citation status describes the Library references: an uncited Library bundle
does not invalidate an answer supported by workspace files. Existing validation
of actual `[S#]` markers, source receipts, and citation repair remains intact.
The implementation must qualify initial sends, retry, and handoff consumers;
automatic and explicitly staged evidence have different preparation paths.

## Optional index ownership and storage

Indexing is disabled by default and enabled explicitly per local folder binding.
Enablement authorizes application indexing reads within that live binding and
the selected processing scope; it does not authorize an assistant tool call.
The first release supports local filesystem bindings. SSH bindings retain file
exploration and show indexing as unavailable; local cache support does not imply
remote scan permission or a remote indexing protocol.

Use a dedicated `Workspaces` project-index service with one lifecycle owner per
active local data authority. Reuse `SQLiteStoreKernel` for a separate versioned
`workspace_project_index.db`, under the admitted private application-data root.
Its durable state contains:

- Per-binding enablement, keyword/semantic selection, processing configuration,
  and a monotonically increasing index epoch.
- A disk-backed scan frontier and job checkpoints.
- File identities and normalized binding-relative paths, content hashes, stat
  hints, chunk/line mappings, source text, and keyword FTS rows.
- Generation identity, compatibility fingerprints, coverage counters, and
  pending cleanup state.

Keys include the local data authority, workspace and binding IDs, locator/root
identity, index epoch, and extractor/chunker version. Semantic generations also
include the resolved embedding configuration and destination identity. A binding
ID or matching pathname alone cannot attach an old cache to a new root.

The filesystem is authoritative. Project rows, chunks, and vectors are disposable
derived data. They are not Media/Notes records, do not enter global Library
search, and are not governed by Library RAG scopes. There is no cross-binding
content-hash deduplication that could leak cached content between authorities.

Integrate the database and optional vector storage into ADR-126's storage
inventory, admission, maintenance, activation, and borrower-retirement contracts
before opening production handles. The cache is rebuildable and excluded from
content sync, Chatbook export, and ordinary backups. A restore or data-authority
switch invalidates old generations rather than restoring their readiness.
Per-binding index settings and any cache-owned endpoint consent are device-local
and must not travel with synchronized workspace metadata.

## Index modes and discovery tool

Keyword indexing uses SQLite FTS5 over paths and bounded text chunks. It works
without the embeddings extra, model loading, or network requests. This is the
initial indexing mode and supports exact symbols, filenames, configuration keys,
and words in source or documentation.

Semantic indexing is an additional opt-in. Reuse the existing chunking and
embedding components and optional vector-store adapter, with a separately owned
project namespace and lifecycle. Do not instantiate or query the shared Library
RAG service, whose source and activation contracts are different. Missing
optional dependencies or embedding failures leave keyword search available.

An explicit semantic choice identifies the embedding provider/model and whether
file text will leave the device. A cloud embedding destination requires its own
indexing consent; Library Allowed or a chat-model choice is not that consent.
Changing that destination pauses semantic work pending a new choice. No project
analysis or chat-generation calls run during indexing.

Expose one optional read-only `fs_search_index` spec through the existing
`LocalToolProvider` and registry. It accepts `root_alias`, query, and a bounded
result limit; it searches only the selected admitted binding. Multiple roots
use the existing explicit-alias contract. Keyword and semantic candidates merge
only within that authority and compatible generation.

The tool uses the existing local permission principal and review, schema-hash,
read-only, budget, and project-instruction preparation seams. Excerpt delivery
also applies the binding's effective `fs_read` permission as a narrowing floor:
Blocked refuses delivery, and Ask still requires review even when index search
is otherwise Allowed. Its path-target semantics match a
binding-root search such as `fs_grep`; subsequent `fs_read` calls activate nested
instructions through the existing exact-target path. Indexed instruction-file
text is evidence, not automatically promoted guidance.

Results contain relative path, line locators, a bounded excerpt, origin, and
coverage/freshness state. Before returning a candidate's name or content,
revalidate the live binding, exclusions, file identity, and file content hash
through the pinned read boundary. Construct the excerpt from that validated live
read. Changed or missing files are omitted and queued for refresh; stored old
text is never returned as a substitute. Candidate selection is bounded and can
report partial results when too many candidates require refresh.

The index accelerates discovery. Important claims, edits, and project summaries
still use the normal live file tools to inspect relevant context. Search returns
at most 10 results and 16 KiB of serialized model-visible data per call; a smaller
run budget takes precedence.

## Scanning and large tree behavior

Build and refresh run in background work and never enumerate or read files on
the Textual event loop. A fixed-purpose scanner runs inside a contained process
for one indexing job, reusing root-pinning and process-tree primitives. It is not
an idle daemon or a generic command runner. The existing one-shot `fs_*` worker
contract remains unchanged.

The scanner incrementally uses `os.scandir`, a disk-backed directory frontier,
and bounded messages. The parent service owns the sole database writer and
applies backpressure. No `list(rglob(...))`, recursive Python call stack, complete
in-memory manifest, per-file UI row, or unbounded producer queue is permitted.
Checkpoints occur after committed pages. Restart can re-enumerate the interrupted
directory idempotently; an iterator offset is not durable proof of coverage.

Use pages of at most 256 discovered entries, at most two concurrent file reads,
and at most 4 MiB of extracted text queued for the writer. A file is eligible
for content indexing only when it is bounded UTF-8 text of at most 1 MiB.
Common source, configuration, manifest, and documentation files are included;
binary files, archives, generated outputs, dependencies, VCS internals, and
symlink/junction/reparse targets are skipped. Do not execute repository code,
install dependencies, perform OCR, or fetch remote resources.

Sensitive-path rules and exact binding exclusions are mandatory and evaluated
before descent or reads. Skipped excluded subtrees do not contribute names or
counts to agent-visible results. Default noise filters additionally skip
`node_modules`, `.venv`, `venv`, `__pycache__`, `dist`, `build`, and `.git`.
Noise filters are processing policy, not a new security mechanism or a complete
gitignore parser.

The initial ceilings are 250,000 discovered eligible entries and 1 GiB of total
cache-owned storage per binding, including staging and temporary/vector data.
Admission accounts for bytes before durable publication and retains cleanup
headroom. Reaching either ceiling pauses the job with explicit partial coverage
and a way to narrow included subfolders or file types. It never silently marks
the tree complete. A single enormous directory is subject to the same paging
and cancellation requirements as a deep tree.

These are implementation constants for the first release, not performance
claims. Their purpose is to cover the requested 10k+ scale with bounded memory,
disk, and context. Qualification must exercise both many directories and many
files in one directory.

## Refresh publication and revocation

The first release refreshes explicitly with Refresh index and can resume an
interrupted build. It does not add a filesystem-event dependency or perform a
full scan on every chat send. The UI reports the last completed scan, searchable
coverage, and changes encountered during candidate validation.

Stat hints avoid unnecessary processing within a refresh, but content hashes
establish chunk identity. Before/after read checks reject files changing during
capture. Renames are a new path plus removal of the old path; unchanged content
can reuse chunks only within the same binding authority. A completed traversal
retires unseen entries; a paused or failed traversal cannot classify unvisited
files as deleted.

Commit keyword rows and their manifest revisions transactionally in batches.
Only committed, individually compatible entries are searchable during a partial
build. Semantic work stages a separate compatible generation and atomically
publishes its ready pointer after validation; queries may retain the previous
compatible semantic generation during rebuilding, with live candidate validation.
Failed, cancelled, or configuration-incompatible semantic generations are never
queried. Embedding-profile changes retain keyword availability.

Removal, retargeting, root-identity replacement, Disable, Clear index, or data
authority changes revoke eligibility before cleanup. Every writer publication
checks the captured epoch and current authority, so an old worker cannot publish
after revocation. Each committed batch also rechecks binding exclusions; additions
take effect before the next publish or query, and removals admit work only in a
fresh indexing operation. Unknown current authority refuses delivery.

Queries apply authority and exclusion filters before ranking and top-k, then
validate the selected live candidates again. The current run's narrower admitted
roots and exclusions always win over cached index readiness. An index is not an
alternate route around a blocked file read or a missing binding.

Detachment and disabling stop pending work and make cached content immediately
unqueryable. Physical cleanup is asynchronous and retryable; failures cannot
restore query access. Clear index also removes consent/enablement and advances
the epoch, so nothing automatically rebuilds it. Shutdown drains issued writes
and retires owned handles; interrupted jobs need explicit Resume after restart.
Already disclosed tool output follows existing conversation retention; detaching
a folder cannot retract text previously sent to a model or saved in a transcript.

## User interaction

Settings remains the canonical workspace-management surface. Each local folder
binding gains an optional Project index section with mode, status, last scan,
coverage, approximate storage, and the actions appropriate to its state:
Enable/Create index, Refresh, Pause, Resume, Disable, and Clear index. A narrowing
control selects included subfolders and text file types before a large build.
Those processing inclusions cannot override binding exclusions or sensitive paths.
Changing them advances the processing revision. Narrowing removes affected
candidates immediately; widening requires a new Refresh before adding coverage.

The Console file inspector presents compact, read-only index status with a
management action that opens the binding's Settings section. It keeps Library
policy and project-index status distinct. No new global navigation destination
or mandatory setup wizard is introduced.

Visible state distinguishes Off, Building, Partial, Ready, Paused, Interrupted,
and Error. Keyword and semantic coverage are reported independently. "Ready"
means a scan completed under its stated policy, not a live guarantee that no
file has changed. Progress uses discovered counts until the total is known;
large trees do not show invented percentages. Failure copy identifies the next
action while preserving available keyword search and ordinary file tools.

Compose existing settings rows, status/callout, and action patterns under
ADR-150/161. Every visual value uses the existing design tokens; source styles
are rebuilt rather than editing the generated CSS bundle. Do not expose host
paths, excluded names, indexed bodies, or query text in generic diagnostics.

## Architecture boundaries and delivery

| Owner | Responsibility |
| --- | --- |
| Existing workspace note and first-request plan | Truthful capability-aware exploration guidance. |
| Console evidence formatting and presentation | Mixed Library/file grounding without changing dedicated Library answers. |
| Proposed project-index service in `Workspaces` | Index settings, scan jobs, generation state, query candidates, revocation, and cleanup. |
| Proposed `DB/Workspace_Project_Index_DB.py` | Versioned manifest, FTS, checkpoints, and transactional publication through the existing store kernel. |
| Fixed project scanner and existing pin/containment primitives | Bounded local discovery and file capture under the exact binding authority. |
| Existing local tool provider | Optional index search, normal approval and budget enforcement, and live-result validation. |
| Existing chunking/embedding/vector components | Optional semantic projection with isolated project ownership. |
| Settings and Console inspector | Management and read-only status; neither owns jobs or database authority. |

Deliver guidance and Console grounding first, with no storage migration and no
change to Library settings. Deliver keyword index contracts, storage, bounded
scan/recovery, and management as the next coherent increment. Semantic indexing
is a further optional increment sharing the same ownership and revocation
contract. Each implementation task must be atomic and independently verifiable;
the index does not block the guidance improvement.

ADR required: yes

ADR path: `backlog/decisions/240-workspace-project-context-and-folder-bound-index.md`

Reason: optional indexing introduces durable derived storage, background-worker
ownership, a file-tool contract, and source-disclosure/revocation rules. It
preserves ADR-079 Library choices and builds on ADR-101/102/174 file authority.
The guidance-only delivery implements this design without its own new boundary.

## Acceptance and verification

1. An unindexed project with an identifiable README/manifest can be summarized
   from a natural project question through the real Console, without a tool-name
   instruction. Repeat with unrelated Library evidence and saved Automatic.
2. Never/Automatic and Blocked/Allowed values are byte-for-byte preserved by
   folder attachment, guidance, indexing, restart, and index clearing. Explicit
   Library and mixed-source questions retain their existing scope and permissions.
3. Single/multiple roots, selected-root narrowing, read-only bindings, missing
   tools, denied reads, scratch-only sessions, and SSH-without-index all produce
   truthful behavior through native and fenced tool protocols.
4. Console answers can use verified file evidence alongside Library sources
   without inventing Library labels. Dedicated Library answer grounding and
   actual marker validation retain their existing strict behavior.
5. Keyword indexing and search work with embeddings dependencies absent and no
   network traffic. Semantic enablement identifies its destination; destination
   changes pause processing until explicitly accepted.
6. A corpus with at least 20,000 text files across 20,000 nested/sibling folders,
   plus a separate 20,000-file single directory, completes under the stated policy
   or reports an explicit limit. Search returns deeply placed known markers;
   directory listing truncation does not constrain indexing coverage.
7. Scanner and writer queues remain within the stated bounds; no tree-sized
   in-memory list or UI collection is created. The scale qualification records
   machine, corpus bytes, peak extra RSS, scan throughput, disk use, cancellation
   latency, and cached lexical-query p50/p95. Target cached lexical candidate
   lookup p95 is at most 500 ms on the recorded local reference machine; live
   validation and semantic/provider latency are reported separately.
8. Pause, cancellation, process interruption, disk-limit exhaustion, a file
   changing during read, and dependency failure preserve committed compatible
   state and truthful partial coverage. Resume is idempotent; it neither loses
   frontier entries nor duplicates file/chunk rows.
9. A delayed build/query race with exclusion addition, detach, retarget, Clear,
   root replacement, or data-authority switch cannot deliver revoked names or
   content or republish an old generation. Cleanup failure remains unqueryable.
10. Stale/deleted-file candidates never disclose cached old text. Failed or
    partial scans do not delete unvisited rows. Semantic rebuild failure retains
    only a compatible prior generation and independently available keyword search.
11. Storage admission, maintenance, restart, backup/restore, and actual worker
    retirement are exercised with real isolated SQLite owners. No project-cache
    payload enters Library media, global Library search, content sync, or export.
12. Targeted regressions cover the modified request-plan, citation, Library
    policy, local-provider, storage, and UI paths. Live Console qualification uses
    a real model and captures actual file calls and project-specific answers.
    Full-suite execution still requires the repository's explicit opt-in.

Update `Docs/Architecture/console-file-authority.md`, `rag.md`, and the Console
user guide with the corresponding delivery. Add a project-index architecture
document and index entry when that subsystem is implemented. Documentation and
tests must distinguish proposed budgets from measured results.

## Alternatives and exclusions

A prompt-only change without correcting the Console grounding conflict leaves
competing source instructions in place. A deterministic intent router adds a
new failure mode for ambiguous project/Library questions and cannot replace
honoring saved Automatic policy. Neither is the selected approach.

Automatic Library ingestion changes ownership and retention, duplicates project
content, and does not satisfy folder-bound access. Explicit Library import stays
available as its own workflow. Automatic whole-tree context injection exceeds
budgets at the requested scale; the selected index returns bounded discovery
results and live excerpts instead.

The initial index excludes remote indexing, filesystem watchers, repository
execution, generated project summaries on every refresh, a full language-server
or symbol graph, binary/media extraction, and cross-project search. These can be
designed separately if a demonstrated use case requires them.

# Console semantic trace reachability inventory

This is the schema-derived deletion contract for the reference-backed Console
trace ledger. ADR-097 is authoritative. The inventory describes ChaChaNotes
schema v63; every future trace owner or edge must update this file, the epoch
mutation matrix, and the adversarial GC fixtures in the same change.

## Roots

| Root | Durable evidence | Mark behavior |
|---|---|---|
| Persisted conversation | An attached `console_trace_owners` row whose conversation still exists | Mark the owner's root segment without an upper bound. A soft-deleted conversation remains attached and therefore remains a root. |
| Fork | An attached owner whose root segment has a parent boundary | Mark the child segment in full, then each ancestor only through the child's immutable inherited event boundary. |
| Open provider call | A call in `reserved`, `dispatch_started`, or `response_started` | Mark the call's segment lineage even if its conversation owner detached while provider work was live. |
| Migration | Any `console_trace_migration_state` row not `logical_complete` | Conservatively mark every trace row. Normalization and collection share the maintenance exclusion state. |
| Explicit retention | An unexpired `console_trace_retention_roots` row | Mark the named owner, call, revision, or artifact and its required closure. |

A detached owner is not itself a root. It is retained only when another root
reaches its shared segment/call history or an explicit retention row names it.
GC request rows, mark rows, segment-scope rows, the epoch singleton, and the
maintenance/migration singletons are control metadata and are never payload
roots.

## Graph edges

| Owner table | Outgoing reachability |
|---|---|
| `console_trace_owners` | `root_segment_id`; attached `conversation_id` is the external visibility root. |
| `console_trace_segments` | `parent_segment_id`, `inherited_through_sequence`, and `inherited_surface_head_id`. The sequence bounds ancestor events; it is not permission to mark the ancestor suffix. |
| `console_trace_calls` | `owner_id`, `segment_id`, `policy_id`, optional `surface_node_id`, and optional `request_header_id`. Lifecycle state decides whether the call is independently rooted. In a bounded ancestor, a call is retained when one of its events is inside the boundary; an eventless pre-dispatch call is retained only when its immutable `turn_id` is proven by an in-boundary turn event. |
| `console_trace_events` | `segment_id` plus optional call, surface node, surface replacement, request header, semantic revision, or artifact reference. Only events within a segment's marked boundary are reachable. |
| `console_trace_surface_nodes` | `segment_id`, predecessor node, and exactly one revision/artifact/omission reference. Marked nodes close recursively over predecessors. |
| `console_trace_surface_replacements` | Segment, predecessor head, start/end nodes, and replacement node. A replacement becomes reachable only through a marked event. |
| `console_trace_request_headers` | Header components; scalar route/model/generation fields contain no graph identity. |
| `console_trace_header_components` | Header and artifact. |
| `console_trace_semantic_revisions` | Predecessor revision and optional live message locator. The locator points outward to canonical Chat data; canonical message content is never copied into the revision row. |
| `console_trace_revision_bindings` | Revision, frozen policy, and optional artifact. A binding is retained only when both its revision and policy are reachable. |
| `console_trace_redaction_spans` | Frozen policy and exactly one revision or artifact source. Span rows are content-free but required to project that source safely. |
| `console_trace_response_links` | Call and exactly one revision or artifact response. |
| `console_trace_policies` | Optional PII ruleset revision identity. The identity is external frozen policy provenance, not a foreign key into semantic message revisions. |
| `console_trace_artifacts` | No outgoing graph edge; owns sanitized bytes. |
| `console_trace_retention_roots` | Generic retained owner/call/revision/artifact identity until `retain_until`. |

The migration, maintenance, graph-epoch, GC-run, GC-mark, and segment-scope
tables contain only control identities, counters, timestamps, and content-free
results. They do not retain trace payload independently.

## Reachability mutation and epoch matrix

Every row below advances `console_trace_graph_epoch` in the same SQLite
transaction when, and only when, it changes the reachable graph.

| Mutation | Epoch source |
|---|---|
| Attach/detach conversation owner | Repository advance; hard conversation deletion trigger performs detach and advance atomically. |
| Create inherited segment boundary | Repository advance. An unattached root segment has no edge and does not advance until ownership attaches. |
| Insert/reuse policy with a ruleset edge | Repository advance on first insertion with a ruleset identity. |
| Insert semantic revision with a live locator or predecessor | Repository advance; locator-free revision zero is not reachable until another edge names it. |
| Retire locator and append successor/replacement during message mutation | Coordinator performs one aggregate advance after the atomic mutation bundle. |
| Add revision binding, surface node, replacement, call reservation/binding, event, or response link | Repository advance after the durable write. Exact retries do not advance. |
| Add one or more redaction spans | Repository performs one aggregate advance after the span set is inserted. Exact reuse does not advance. |
| Cross provider-call open/terminal boundary | v63 `console_trace_calls_open_root_epoch` trigger. Transitions within the open set do not change roots. |
| Enter or leave migration-root state | v63 `console_trace_migration_root_epoch` trigger. |
| Add or remove an active explicit retention root | v63 retention insert/delete epoch triggers. Expired-root cleanup does not advance because the clock has already made that root inactive and the prior mark remains conservative. |

Direct graph deletion never constitutes a supported mutation API. All payload
tables retain fail-closed `BEFORE DELETE` guards. The registered SQLite callback
returns true only inside the private collector scope after the sweep transaction
has verified the exact lease ID, marked epoch, current epoch, and maintenance
state. A generic SQLite connection has no callback and therefore also fails
closed.

## Mark, sweep, and reclamation result

Mark persists opaque entity identities and bounded segment scopes under a
request ID and exact epoch. Sweep begins a new immediate transaction, changes
the held lease from `marking` to `sweeping`, rechecks the same epoch, and only
then opens the connection-local deletion grant. Any intervening owner, call,
migration, retention, or graph-edge change returns `stale_epoch` with zero
deletions. The caller may remark the same request safely.

Sweep deletes in foreign-key-safe leaf-to-root order: redaction spans, response
links, events, replacements, bindings, header components, calls, surface nodes,
headers, revisions, artifacts, policies, owners, then segments. Physical
`VACUUM`/compaction is deliberately outside this task and belongs to
TASK-23113.11.

`TraceGCResult` reports reclaimed rows, live/reclaimed sanitized-artifact bytes,
reclaimed/freelist pages and bytes, allocated pages and bytes, and WAL bytes as
separate measurements. Row counts cover the remaining metadata tables; the
logical byte counters deliberately measure the byte-owning artifact table and
do not pretend SQLite row overhead is content size. Replaying a completed
request returns the same stored content-free result. A crash before sweep leaves
the mark resumable; a crash during sweep rolls the whole sweep transaction back.

## Durable privacy owners and projections

The deletion oracle covers every trace payload table listed above, the
conversation/fork projection assembled from those rows, persisted GC result
JSON, GC marks/scopes, the WAL metric, and reopen behavior. GC result rows never
store captured text, artifact bytes, locators, request bodies, response bodies,
PII matches, or credentials. Logical reclamation does not claim secure erasure
from already allocated SQLite pages; automatic physical compaction is the
separate TASK-23113.11 boundary.

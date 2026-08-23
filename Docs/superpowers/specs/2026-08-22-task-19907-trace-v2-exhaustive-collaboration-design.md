# Trace v2: Exhaustive Events and Collaboration — Design

Date: 2026-08-22
Status: Approved
Tasks: TASK-19907, TASK-19908, TASK-19910, TASK-19911, TASK-19912, TASK-19913
Supersedes the product scope, not the retained compatibility guarantees, of the
2026-08-14 Console Trajectory View design.

## Job and audience

Trace is the Console's forensic debugging and collaboration surface. A developer,
operator, or reviewer opens it after or during a run to reconstruct every observable
event, find the first meaningful divergence, inspect the evidence, follow causal and
agent lineage, and share a privacy-governed copy.

The primary product job is **show every observable event in causal order**. Fast
failure diagnosis is the principal navigation optimization, not a reduction in event
coverage.

## Product language

- **Trace** is the canonical user-facing name and Console shortcut label.
- **Trajectory** remains an internal compatibility/projection term where renaming it
  would churn stable modules or the v1 file contract.
- Human-facing event labels use sentence case (`Tool call`, `Agent started`) rather
  than storage identifiers (`tool_call`).

## Exhaustive event contract

“Every event” means every transition observable by Chatbook at an owned runtime seam.
It does not mean fabricated provider-internal events or hidden chain-of-thought.

Required families:

1. Conversation: user, system, assistant, feedback, edit/regenerate, branch selection.
2. Model: request started, first token, response completed, retry, provider error,
   cancellation, and token/timing facts.
3. Tools: proposed, approval requested, approved/denied/revoked, execution started,
   succeeded, failed, timed out, and cancelled.
4. Context: project/system context attached, retrieval started/completed/failed,
   candidates selected, context injected, compaction started/completed/failed, and
   replacement ancestry.
5. Agents: primary/child run created, started, step, steering/handoff, completed,
   failed, cancelled, superseded, and resumed, with parent/child/source relationships.
6. Trace operations: import, export profile, redaction/truncation/omission markers,
   and incomplete capture diagnostics when they affect interpretation.

Every projected event exposes:

- stable `event_id` derived from its source owner, plus `conversation_id`;
- `kind`, `status`, and human label;
- conversation, turn, run, actor, parent, source, and replacement identifiers when
  known;
- immutable `source_seq` within its owner, distinct from display position, plus
  observed timestamps;
- safe summary and structured payload metadata;
- per-material-field state of `observed`, `not_available`, `redacted`, `truncated`,
  `omitted`, or `capture_failed`;
- a sensitivity classification used by export preflight.

## Ownership and projection

No second database duplicates all history. The pure projection reads the existing
local owners and normalizes them into one `TrajectorySnapshot`/Trace event stream:

- messages and `message_trajectory_metadata` for conversation/model/tool observations;
- `AgentRunsDB.agent_runs` and append-only `agent_run_steps` for agent steps and
  durable lineage;
- compaction and context repositories for context ancestry;
- citation/retrieval provenance for retrieval evidence;
- transcript feedback/annotation records for review events.

Agent steps are persisted incrementally at the existing `LoopDeps.on_step` seam so a
crash does not erase an otherwise observed run. A new injected UTC wall-clock seam is
separate from the runtime's monotonic budget clock. Each step uses its stable runtime
index as an idempotent database sequence; terminal persistence inserts any missing
indices and cannot duplicate successful incremental writes.

AgentRuns schema v14 adds nullable `spawn_event_id` to `agent_runs`. The parent spawn
step receives a stable event ID before dispatch; every child run, including fleet and
inline paths, stores it when created. This durably correlates parallel or identical
spawns after restart without treating process-local handle IDs as lineage.

Ordering is deterministic: causal parents precede descendants; within one owner its
durable sequence wins; unrelated concurrent events are ordered by observed time and
stable event ID. The UI may show concurrency but must never imply false causality.

Legacy conversations remain valid. Missing v2 sources produce explicit incomplete
metadata rather than fabricated events or a blocking migration/backfill.

## Interaction and layout

At rest Trace answers: what is this run, is it live/paused/imported/incomplete, what
happened, and where are the failures?

Hierarchy:

1. Title and state: `Trace · <conversation>` plus `LIVE · FOLLOWING`, `LIVE · PAUSED`,
   `READ-ONLY SHARED`, or `INCOMPLETE`.
2. Search and filters: kind, status, agent, provider, time, visible/total counts, clear.
3. Semantic timeline: Input, Model, Tools, Agents; Feedback appears with Input.
4. Responsive ledger: identity, event, summary, status, and progressively disclosed
   metrics.
5. Independently scrollable inspector: payload, timing, tokens, context, lineage,
   privacy, copy, wrap, and full-pane mode.

Responsive tiers:

- 60–99 columns: `# / Event / Summary / State`; metrics live in inspector.
- 100–119: add compact Tokens and Duration.
- 120+: full metrics where they fit without clipping.

The timeline collapses to an explanatory one-row state when no timing exists. Every
mouse action has a keyboard equivalent and contextual hint. Search/filter/timeline/
ledger/inspector selection share one selection model across live refreshes.

## Collaboration workflow

Export is user-initiated and starts with a preflight inventory. Profiles:

1. Safe summary: summaries, causal structure, status, coarse timing; payload bodies
   omitted.
2. Redacted diagnostic (default): diagnostic payloads with classified secrets,
   contents, paths, and identifiers redacted or previewed.
3. Full trace: explicit warning and confirmation; credentials remain forbidden.

Trace format v2 is a self-contained, schema-versioned JSON document preserving event
identity, order, lineage, timing, redaction provenance, missing-data reasons, and a
SHA-256 integrity digest over canonical JSON. The digest detects accidental/tampered
content but is not an authenticity signature.

Imports are visibly labeled `READ-ONLY SHARED TRACE`, remain ephemeral, and never
write conversations, messages, sidecars, agent runs, or trace events. Readers retain
v1 compatibility and reject unsupported versions or integrity failures with actionable
errors. The imported result carries the snapshot, manifest, integrity verdict, privacy
inventory, and ephemeral import operation so the screen does not discard collaboration
context.

## States and realistic ranges

- Empty: zero events or legacy data with a reason and next action.
- Typical: 10–500 events, 0–10 tool calls, 0–3 child agents.
- Large: 5,000+ events; worker projection, paging/virtualization, and stale-result guard.
- Live: following or explicitly paused by user navigation.
- Partial: capture failure, missing older metadata, redacted/truncated fields.
- Shared: safe-summary, redacted, or full profile; read-only.
- Failure: provider/tool/agent/export/import failure with retry or resolution guidance.

## Accessibility and performance constraints

- Full keyboard parity, visible focus, non-color event encoding, theme tokens.
- Every rendered field reachable at 60×18 and 80×24.
- No terminal-convention or reserved global shortcuts; ADR-031 footer governance holds.
- Projection remains pure; UI never queries databases directly.
- Trace capture is best-effort and never fails the run; capture failure is itself
  surfaced diagnostically when possible.
- No new dependency is required; use stdlib JSON/hashlib and existing Textual widgets.

## Scope and anti-goals

- No editing or replay from Trace in this programme.
- No hidden chain-of-thought persistence or export.
- No synchronized Trace database and no implicit network sharing.
- No broad class/file rename solely to replace the internal word trajectory.
- No second all-events database while existing owners can provide the contract.

## Delivery slices

- TASK-19907: event contract and ordered pure projection.
- TASK-19908: Console/model/tool/approval/context capture.
- TASK-19910: agent lifecycle and causal lineage capture.
- TASK-19911: responsive ledger, reachable inspector, explicit states, naming.
- TASK-19912: semantic timeline, filters, keyboard anomaly navigation.
- TASK-19913: collaboration export/import v2 and privacy preflight.

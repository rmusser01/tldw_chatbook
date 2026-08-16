# ADR-068: Local research execution engine

- **Status:** Accepted
- **Date:** 2026-08-15
- **Task:** task-16322 (Build the local research execution engine)
- **Related:** ADR-024 (RAG citation provenance — different domain, cited for
  contrast); `Docs/Development/research-sessions-runs-tranche-2.md`;
  task-255 (orphan research route removal); task-1356 (`web_deep_search`
  tool); task-16331 (deep-search citation verification); task-16323 (budget
  ledger, follows), task-16324 (iterative replanning, follows)

## Context

Since tranche 2 (2026-04), Chatbook has a full **run-lifecycle bookkeeping**
layer for research (`Research_Interop`: SQLite runs/events/artifacts, scope
routing, SSE observation) but **nothing executes a local run** —
`launch_run` writes a `running` row with `phase='local_planning'` that no
component ever advances. The only real research execution is the deep-search
pipeline (`Web_Scraping/WebSearch_APIs.py`: `generate_and_search` →
`analyze_and_aggregate`), exposed solely as the opt-in `web_deep_search`
agent tool and therefore disconnected from the run lifecycle. Meanwhile
task-255 removed the orphan research screen route, leaving
`Research_Window` unreachable.

tldw_server dev solves this with a phase machine (`app/core/Research/`:
drafting_plan → collecting → synthesizing → packaging, with checkpointed
autonomy variants) whose engines, artifacts, and events are the server
contract `ServerResearchService` already mirrors.

## Decision

1. **Add a UI-agnostic async engine** (`Research_Interop/local_research_engine.py`)
   that drives an existing local run through `planning → collecting →
   synthesizing → packaging → completed`, reusing the live pipeline
   functions — `generate_and_search` for collection and
   `analyze_and_aggregate` for synthesis — via **injectable callables**
   (`search_fn`, `analyze_fn`). The pipeline is never forked or re-implemented.
2. **The engine is not a storage writer.** All state, event, and artifact
   mutations go through `LocalResearchService` (single writer, versioned
   rows, append-only events). The service gains one generic
   `update_run_progress` transition; existing methods (`pause_run`,
   `resume_run`, `cancel_run`, `complete_run`, `fail_run`) remain the only
   terminal/control transitions.
3. **Pause/cancel are honored between phases** by polling `control_state`
   before each phase: paused runs are left in place (status stays
   non-terminal, an `engine_paused` event records the phase; a later resume
   re-invokes the engine, which restarts the run — phase-level resume is
   explicitly out of scope for v1), cancel requests resolve through
   `cancel_run`.
4. **Artifact contract mirrors the server's names** so bundles are
   shape-comparable across modes: `plan.json`, `collection_summary.json`,
   `sources.json`, `verification_summary.json` (fed by task-16331's
   `citation_verification` when the synthesis branch produced one),
   `report_v1.md`, and `bundle.json`.
5. **Re-register a Research destination** resolving task-255's deferred
   decision: a `ResearchScreen` hosting the existing `ResearchWindow`, under
   the `research` route id. Launching or resuming a **local** run from the
   window starts the engine in a Textual worker. Server-mode runs remain
   untouched (`ServerResearchService` contract is unchanged).
6. **`autonomy_mode` is recorded but not yet enforced locally** — v1 runs
   autonomously through all phases; local checkpoint review UI remains
   deferred (as in the tranche doc). Budget/limits enforcement follows in
   task-16323 against the same engine seams.

## Consequences

- A launched local run now actually executes and reaches a terminal state
  with inspectable artifacts and an event trail consumable by the existing
  stream endpoints.
- The engine's injectable seams are the attachment points for the budget
  ledger (task-16323), iterative replanning (task-16324), claims artifacts
  (task-16325), and academic lanes (task-16326) — each lands without
  reopening this contract.
- Draft runs created via the window start in `draft` and are normalized to
  `running` by the engine's first transition.
- Duplicate `research`-route startup configs resolve to the real screen
  again (behavior change vs. the task-255 library alias — intended).
- The pipeline's existing failure/degradation behavior (partial synthesis,
  fallback answers) flows through unchanged; the engine only classifies
  outcomes (completed vs failed).

## Alternatives considered

- **Embed the server's engine** — rejected: it is coupled to the server's
  job/worker SDK and DB layer; the client would inherit server contract
  drift and heavy deps for a single-user TUI.
- **Extend the `web_deep_search` tool to write runs** — rejected: wrong
  layer. The tool is an agent-facing capability with byte-capped text
  output; run lifecycle is a service concern with structured artifacts.
- **Delete `Research_Window` instead of wiring it** — rejected: it is the
  only observation surface for runs and events; the engine gives it a
  purpose. Revisit only if the workbench absorbs research UX later.
- **Fork the pipeline into the engine** — rejected: guarantees behavioral
  drift from the tool path (deadlines, robots, citation verification) the
  moment either side changes.

## Addendum (2026-08-15, task-16482): checkpoint enforcement activated

Clause 6's deferral is resolved: runs with `autonomy_mode="checkpointed"`
(the schema default) now pause at phase boundaries for review — a
`plan_review` checkpoint before any search spend and a `sources_review`
checkpoint before synthesis — parking in a non-terminal
`awaiting_<type>` control state with partial artifacts preserved.
Approval goes through `LocalResearchService.patch_and_approve_checkpoint`
with per-type patch validation (plan: `limits`; sources:
`pinned_source_ids`/`dropped_source_ids`/`recollect`, inventory-checked
and disjoint); an approved plan `limits` patch supersedes the run's
originals for budget enforcement, an approved sources patch drops the
named sources, and `recollect.enabled` loops the run back to collecting
for a fresh sources review (server parity). Outline review remains
unimplemented locally: the local engine has no separate outline phase
(its plan covers focus areas), so the server's third checkpoint type
does not map. The window's Approve Checkpoint action resolves the
latest pending local checkpoint and restarts the engine on approval.

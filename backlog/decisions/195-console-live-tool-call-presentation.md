# ADR-195: Console live tool-call presentation

Status: Accepted
Amended by: [ADR-210](210-console-region-ownership.md) (accepted 2026-10-01)
Date: 2026-09-27
Related Task: TASK-33095
Related: ADR-078, ADR-080, ADR-029, ADR-150

## Decision

Project complete primary tool calls into one stable, session-only Console TOOL
marker keyed by run and call identity. Reuse runtime lifecycle facts and structured
outcomes; never infer execution or approval from tool output text. Actual pending
approval rounds, rather than generic permission-review events, own the awaiting
approval state. The existing approval card remains the decision surface.

An optional best-effort runtime/service observer exposes lifecycle observations
to Console without changing legacy step callbacks, permissions, budgets, provider
history, or durable capture. Proposed-call arguments use the existing display
projection and travel only in the observer's ephemeral copy. Capture retains its
existing omitted-field policy. Provisional markers do not write trajectory rows;
terminal results retain the existing trajectory behavior exactly once. Live
visibility does not make previously uncaptured discovery tools eligible for
trajectory capture. Display observer failures cannot interrupt approval or run
teardown.

Show at most three wrapped result-preview rows with explicit omission information.
One disclosure contains arguments and the retained result, including existing
full-output and diff access. Keep mounted headers, expansion, selection, and scroll
stable during updates. Finish unresolved rows honestly when a run stops or fails;
an abandoned call does not claim that its underlying process was killed.

Existing raw-shell lifecycle presentation remains its own authority. A model
shell call retains its proposed row and Assistant disclosure when the shell
renderer takes ownership, including its existing live output details. User-owned
raw commands remain standalone. Historical runs keep their reconstruction path;
missing historical arguments or timing are not invented. No new SQLite schema,
tool-output retention, settings surface, or partial-output streaming is introduced.

## Alternatives

- Appending a new row for every event splits one call across the transcript and
  loses disclosure/focus identity.
- Polling durable trace rows adds database work and conflates capture availability
  with live presentation. Reusing observations keeps those responsibilities apart.
- Reusing the legacy step callback for lifecycle events changes step counts and
  existing consumers. A separate optional observer preserves that contract.
- Treating every approval-review event as a user wait mislabels automatically
  authorized calls. The pending approval host is the source of truth.

## Verification

Targeted runtime/bridge tests must observe rows before tool completion and match
same-name calls by ID. Mounted Console tests cover previews after wrapping,
keyboard disclosure, stable focus and body identity, and session isolation. Live
verification uses disposable state. Streaming partial results remains v2.

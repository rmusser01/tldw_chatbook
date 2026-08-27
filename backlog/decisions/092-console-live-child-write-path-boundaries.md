# ADR-092: Console live-child WRITE paths cross exact change boundaries

Status: Accepted
Date: 2026-08-26
Related Task: [TASK-15671](../tasks/task-15671%20-%20The-gitignore-force-add-carve-out-does-not-extend-to-a-survivor-window.md)
Related Spec: [Ignored survivor-write tracking](../../Docs/superpowers/specs/2026-08-26-task-15671-ignored-survivor-write-tracking-design.md)
Extends: ADR-089

## Decision

Console keeps a bounded, ephemeral projection of attributed sub-agent WRITE
paths per spawning turn. Pending fleet handles register that turn's state
before parent E returns; child scope and settle observations retain it only
while relevant. Exact B/E boundaries repeatedly consume references to these
states, so Git commits—not tool callback timing—determine which adjacent window
owns the content change.

Successor startup claims an open survivor window before starting B and carries
that window's retained child-state references into B's force paths. Final
settle may therefore remove a state from the live map without moving its pre-B
write into successor E. A closer uses the claimed handle's exact baseline; if
fresh close already started, B waits for it and begins from the resulting tip.
Concurrent closers share one completion event, ensuring successor E cannot
overtake close-time work.

Boundary snapshots may atomically force-add eligible recorded paths inside the
shadow repository lock. Supplied-SHA closure never rewrites that SHA, but may
prime the index so a path first available after B appears at successor E with
the existing concurrent-subagent disclosure.

The projection stores only owner identity, scope counts, and normalized path
strings. AgentRunsDB remains durable step authority. No file
content, step payload, schema row, filesystem watcher, or polling job is added.

## Context

ADR-089 requires turn-owned cards and exact abutting survivor/successor Git
boundaries. Fresh E already force-adds ignored primary WRITE paths, but survivor
closure had no equivalent input. Close-only run hydration cannot model a tool
call spanning E, a pending child whose thread starts late, an inherited child
crossing later turns, or an immutable successor B.

The attributed callback follows durable step observation and precedes WRITE
execution. Retaining its small path projection across relevant boundaries lets
each snapshot retry absent files without claiming that callback time equals
filesystem completion time.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| Read every child run only at close | Lacks execution timing and cannot repair an already-ignored successor B. |
| Partition by step-index watermark | Tool-call indexes precede execution and can lose an in-flight write. |
| Register only at child scope entry | A successfully launched fleet thread can remain pending past parent E. |
| Replace successor B after closure | Breaks exact abutting history and risks duplicate attribution. |
| Add a filesystem watcher | Converts bounded review into open-ended monitoring of detached work. |

## Consequences

- `ChangeTurnTracker.begin_turn` accepts optional recorded WRITE paths and
  baseline snapshots can force-add them atomically.
- Fresh E behavior and path eligibility remain unchanged; live child paths use
  the same rules at B and E.
- Supplied-SHA closure may stage paths for the next fresh snapshot but never
  changes the supplied SHA or its diff.
- Pending, inherited, and E-in-flight children remain visible through exact
  state references; children spawned by a later turn remain separate.
- Successor startup and survivor closure gain bounded event handoff. A timeout
  disables tracking instead of producing overlapping review history.
- Detached writes completing after the child lifecycle remain out of scope.

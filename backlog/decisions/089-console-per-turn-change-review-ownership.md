# ADR-089: Console per-turn change-review ownership

Status: Accepted
Date: 2026-08-25
Related Task: [TASK-22305](../tasks/task-22305%20-%20Simplify-Console-change-review-around-per-turn-cards.md)
Related Spec: [Console per-turn change review simplification](../../Docs/superpowers/specs/2026-08-25-console-turn-change-review-simplification-design.md)
Amends: ADR-077
Preserves: ADR-083

## Decision

Console file-change review is turn-owned. A changed agent turn presents one
transcript card with its summary, changed-file rows, bounded inline diffs,
notes, a direct guarded **Undo All** action, and a route to the full Change
Review screen. The full screen remains the authority for history, complete
diffs, per-file revert, comments, and git actions.

Inspector no longer owns a cross-turn Changed Files projection. Its
latest-per-path aggregation, polling guard, worker/cache, rail widget, config
switch, and rail-only supporting code are retired rather than hidden. This
amends ADR-077 only by removing Changed Files from the set of bounded Inspector
sections; every remaining bounded-section and scroll-ownership rule is
unchanged. ADR-083's edge ownership, pinned send authority, and Inspector group
hierarchy remain intact.

Card-level **Undo All** reuses the existing snapshot provider, edited-since
preflight, confirmation modal, active-run refusal, and per-path revert engine.
All git and filesystem reads/writes run off the Textual UI thread. Ordinary
multi-root turns are supported when each canonical root has one snapshot row.
A turn with multiple rows for the same root is refused inline before any disk
mutation and opened in Change Review, where its separate windows are visible.

Successful undo does not erase the historical card or snapshots. The action is
recorded through the existing `reverted` bookkeeping, the card marks the action
**Undone**, and Review history remains available.

## Context

The per-turn card and Change Review screen already cover the user goals of
discovering, inspecting, annotating, and reverting an agent turn's file changes.
The later Inspector section introduced a second, conversation-wide
latest-per-file model. Keeping it current required a store memo, screen-level
scope tracking, per-row git cache, background worker, note invalidation, rail
reconciliation, a config switch, and dedicated tests. The projection also
collapsed distinct snapshot windows, making its compact path/note view less
truthful than the turn-owned source.

The product still needs changed-file review; removing every surface would leave
users unable to inspect agent effects without leaving the app. The ownership
change removes the redundant projection while strengthening the already useful
turn surface.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| Retain and further optimize the Inspector projection | Optimization does not remove duplicate ownership or its window-collapsing semantics. |
| Keep the Inspector widget hidden behind its config switch | Leaves dead cache, worker, memo, docs, and maintenance surface in production. |
| Remove changed-file cards and keep only Inspector | Loses turn attribution and the natural post-turn review point. |
| Remove all inline undo and require Review | Does not provide the approved direct whole-turn recovery affordance. |
| Apply same-root windows in database order | Baseline order can make the result ambiguous or destructive; the card does not expose enough context to consent safely. |

## Consequences

- The common Console sync path performs no cross-turn file-history polling or
  aggregation.
- Inspector becomes smaller and more focused on next-send authority, sources,
  run state, and settings.
- Historical file review remains available per card and in Change Review.
- A user must open Review for conversation-wide history or ambiguous
  multi-window undo.
- The card gains a confirmation-gated destructive action and therefore owns
  explicit busy, refusal, cancellation, and outcome states.
- Removing the retired projection also removes its config compatibility key;
  an old `changed_files_section` value becomes inert.

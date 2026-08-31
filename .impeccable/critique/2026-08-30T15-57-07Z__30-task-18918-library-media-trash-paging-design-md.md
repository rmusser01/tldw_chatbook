---
target: TASK-18918 Library Media Trash paging design specification
total_score: 30
max_score: 40
na_heuristics:
p0_count: 0
p1_count: 4
timestamp: 2026-08-30T15-57-07Z
slug: 30-task-18918-library-media-trash-paging-design-md
---
# Design Health

| # | Heuristic | Score | Key issue |
|---|---|---:|---|
| 1 | Visibility of system status | 3 | Requested versus applied filter state needs fixed visible copy. |
| 2 | Match system / real world | 3 | Filtered total and local-only authority need explicit labels. |
| 3 | User control and freedom | 3 | Escape precedence is ambiguous while confirmation is open. |
| 4 | Consistency and standards | 3 | Pager-originated focus behavior is not pinned to the shipped convention. |
| 5 | Error prevention | 3 | Permanent-delete confirmation needs explicit mount/focus/activation fencing. |
| 6 | Recognition rather than recall | 3 | Users need to see which scope the retained rows represent. |
| 7 | Flexibility and efficiency | 3 | Page/filter completion focus outcomes are underspecified. |
| 8 | Aesthetic and minimalist design | 3 | 80x24 allocation priorities are outcomes rather than rules. |
| 9 | Error recovery | 4 | Committed mutation truthfulness and authoritative Retry are strong. |
| 10 | Help and documentation | 2 | Disabled reasons and filter-validation copy are not yet specified. |
| **Total** | | **30/40** | **Good, with P1 interaction/contract gaps** |

## Design Specificity Verdict

The design is authored for Chatbook's terminal-native Library: it preserves the
three-column reader, source-owned paging, dense keyboard workflow, local authority,
and truthful stale recovery. It is not a generic web Trash screen. The deterministic
detector returned zero findings, which is expected for a Markdown specification and
does not validate the runtime contracts.

## Overall Impression

The recovery model is unusually strong: exact pages, fail-closed envelopes, and
committed-but-unrefreshed mutation states are treated honestly. The main opportunity
is to make the interaction and service boundaries as exact as the data principles.

## What's Working

- Coherent count, rows, and complete facets prevent unreachable or falsely counted
  records.
- Restore and permanent delete cannot be retroactively reported as failed because a
  follow-up read failed.
- Trash remains an independent nested source, preserving normal Media scope and
  avoiding a generic controller.

## Priority Issues

### P1 — Keyboard and focus precedence is ambiguous

Escape is described as both confirmation Cancel and Trash Back, while page completion
may select/focus the first row even when the pager or filter should retain authority.
Add a precedence table for confirmation, page, filter, Retry, empty, mutation, and
Back outcomes.

### P1 — The exact local service and envelope are unnamed

The current legacy Trash seam returns raw IDs and separately read pagination. Name a
local-only exact scope seam and its payload keys, canonical `local:media:<id>` identity,
cardinality rules, and explicit prohibition on extending the server API.

### P1 — The immediate post-commit state is not formally stale

Removing one item locally makes a full page contain 19 rows, which cannot satisfy the
fresh-page invariant. Require the reducer to withdraw exact totals/boundaries and
enter stale/loading immediately after commit; only the authoritative reload returns
it to fresh.

### P1 — Permanent deletion needs stronger activation safety

A duplicate or truncated title is insufficient confirmation identity, and a repeated
Enter could leak from opener to confirm. Show the wrapped full title plus type/deleted
time, focus Cancel initially, consume opener activation, and require a later explicit
confirm activation. Pin the call to
`permanently_delete_media_item(mode="local", media_id=...)`, preserving its existing
physical cascade/FTS/no-sync behavior.

### P2 — Scope, return, and narrow-layout contracts need exact rules

Define fixed failed-filter copy and Retry target, a distinct stable-row/scroll Trash
return receipt, SQLite-safe page bounds, stored-type equality/facet sorting, explicit
NULL-last SQL order, and the 80x24 vertical priority/minimum list viewport. Clarify
that normal Media keeps scope/selection/focus but becomes stale after Restore rather
than receiving an unranked row insertion.

## Persona Red Flags

- **Keyboard power user:** Page changes may leave focus or selection on an unexpected
  target; disabled-action reasons are not yet guaranteed visible.
- **First-time recovery user:** `Trash (N)` can look like all records rather than
  matching records, and local authority is not visible.
- **Failure-recovery user:** Retry does not yet say whether it repeats the failed
  filter, reloads the prior applied page, or performs a clamp.

## Minor Observations

- State NULL ordering as SQL rather than merely requiring deterministic tests.
- Label counts as `N matching` under filters and consider a quiet `Local Trash`
  authority label.
- Reuse the shipped visible pager-disabled reason convention.

## Questions to Consider

- Should a restored record appear immediately in normal Media without an
  authoritative ranked read, or should that page become stale until refreshed?
- At 80x24, which information can collapse without hiding the irreversible
  consequence or recovery action?
- When a filter request fails, what exact line proves the rows still belong to the
  previous applied scope?

# Watchlists Read Aggregate Feed Selection Design

Status: Approved interactively; written specification review pending  
Date: 2026-08-25  
Extends: [Watchlists NetNewsWire Reader and Collapsible Rails Design](2026-08-23-watchlists-netnewswire-reader-collapsible-rails-design.md)  
Governed by: [ADR-042: Watchlists reader-first information architecture](../../../backlog/decisions/042-watchlists-reader-first-ia.md)

## Goal

Let a user select an individual feed directly beneath **All Sources**, **Unassigned**, or
**All Unread** in the Watchlists Navigation rail. This closes the gap where individual feeds
are visible only after the user creates and expands a watchlist.

The same feed may appear in several branches. Each appearance retains the branch's meaning, so
selecting a feed under All Unread shows only unread items from that feed while selecting the same
feed under All Sources shows its normal Reader population.

## Starting point

On current `origin/dev`:

- `WatchlistTree` renders All Sources, Unassigned, All Unread, Today, and Starred as leaf buttons.
- Only created watchlists are expandable and lazily render source children.
- `TreeScope(kind="source")` carries a source id and optional watchlist id, but cannot distinguish
  the same source selected beneath All Sources, Unassigned, or All Unread.
- `_items_scope_query()` can express source, unassigned, watchlist, and unread predicates
  independently.
- per-source total/unread counts are already loaded in bulk for tree badges.
- `_loaded_sources` is not a suitable aggregate-tree authority: it is capped at 100 rows and may
  not be populated while Read is active.
- ADR-042 already requires pending scope loads to commit navigation highlight, scope, rows, and
  Reader clearing atomically.

## User experience

### Shared Navigation structure

The Navigation rail remains structurally consistent throughout all Watchlists sub-screens:

```text
Smart Feeds
  All Unread   ▸  12
  Today            4
  ★ Starred         3

Sources
  All Sources  ▸  12
  Unassigned   ▸   2

Watchlists
  Research     ▸   6
  Security     ▸   1
```

All Sources, Unassigned, All Unread, and every created watchlist expand independently. Aggregate
feed branches do not disappear on management sub-screens; only their Read-specific item predicate
becomes relevant when Read is active.

Today and Starred remain leaf Smart Feeds in this task.

### Row interaction

Each expandable parent remains a two-target row:

- activating the caret expands or collapses that branch without changing Reader scope;
- activating the label requests the parent scope;
- activating an indented feed requests that feed in the visible parent's context.

Feed children show an unread badge only when their unread count is positive. Child rows are sorted
case-insensitively by display name, with source id as a deterministic tie-breaker.

The same source can render beneath multiple aggregate roots and multiple watchlists. Every child
has a root-qualified widget id, and only the child matching the committed contextual scope receives
active styling.

Keyboard focus stays on the control the user activated. Selecting a feed does not automatically
move focus into Feed Items; normal forward navigation moves there explicitly.

### Empty branches

An expanded empty aggregate branch renders one non-interactive row:

- All Unread: `No unread feeds`
- Unassigned: `No unassigned feeds`

All Sources renders the existing no-sources state when the library is empty.

## Contextual scope contract

The existing normalized Watchlists scope gains explicit source-parent context rather than adding a
parallel selection model. Conceptually, a source scope carries:

```text
kind: source
source_id: int
parent_context: all | unassigned | unread | watchlist
watchlist_id: int | None  # set only when parent_context is watchlist
```

The resulting Reader predicates are:

| Visible selection | Item predicate on Read |
| --- | --- |
| All Sources / Feed A | `source_id = A` |
| Unassigned / Feed A | `source_id = A AND unassigned_only` |
| All Unread / Feed A | `source_id = A AND status = new` |
| Watchlist X / Feed A | `source_id = A` with contextual membership `watchlist_id = X` |

The watchlist context remains necessary for breadcrumb identity and the existing
Remove-from-Watchlist action. That action is enabled only for `parent_context=watchlist`; a stale or
non-null watchlist id alone is not sufficient authority.

The contextual parent is part of item pagination, search, in-flight-load, and snapshot identity.
Two appearances of the same source therefore cannot share results fetched under different
predicates.

Breadcrumb and heading copy retain both levels, for example `All Unread / Feed A`.

## Expansion state

Root and watchlist expansion use separate typed state:

```text
expanded_root_kinds: frozenset[all | unassigned | unread]
expanded_watchlist_ids: frozenset[int]
```

This avoids sentinel ids and collisions between aggregate roots and real watchlists. Both values
are owned by `WatchlistsCollectionsScreen`, seeded into replacement tree factories, and survive
rail rebuilds, responsive parking, Article Focus, and sub-screen changes.

Expansion lifetime matches the existing watchlist-tree contract: it lasts for the screen session
but is not added to persisted pane-layout configuration in this task.

## Aggregate source data

The tree snapshot owns authoritative aggregate source rows; it does not reuse `_loaded_sources`.
Each refresh obtains complete bulk results for:

- all source rows;
- unassigned source rows;
- per-source total/unread counts;
- existing watchlists and watchlist counts.

All Unread children are the all-source rows whose bulk unread count is positive. The selected or
pending All Unread source may be presentation-pinned at zero as described below.

Created watchlist children retain their existing lazy membership load. No path performs one query
per source. Collapsed branches mount no child widgets. Aggregate results are cached only for the
current tree snapshot and invalidated together.

Synchronous database work that may exceed the UI budget runs off the event loop. Concurrent tree
refreshes use latest-generation-wins publication so an older snapshot cannot replace newer counts
or membership.

Virtualization and branch pagination are deferred until profiling demonstrates a real mount cost;
they are not introduced speculatively in this task.

## Atomic scope navigation

Feed selection follows ADR-042's pending/committed split:

```text
feed gesture
  → resolve contextual pending scope
  → keep committed highlight, heading, rows, selection, and Reader visible
  → load replacement Feed Items snapshot
  → if still newest, atomically commit scope + highlight + rows
  → clear selected item and show “Select a feed to display it here.”
```

Focus may remain on the attempted feed while it is pending, but focus styling must not masquerade
as committed active selection.

If loading fails, the previous scope, rows, selected article, and Reader remain active. Failure copy
names both the attempted and retained scopes, for example:

> Couldn't open Feed A under All Unread; still showing All Sources.

Rapid A → B → C selection discards late A and B results. Only C may commit.

Search text is preserved across a successful scope change and re-run against the replacement
scope. Pagination, row selection, previous results, and the previous Reader item reset only when
the replacement snapshot commits.

## All Unread semantics

All Unread contains only feeds whose unread count is positive, except for a bounded display pin.
Selecting a child forces the effective Feed Items status to Unread regardless of the user's parked
manual display filter.

One shared `effective_unread_scope` decision drives:

- item-query kwargs;
- removal of conflicting multi-status kwargs;
- item page and in-flight keys;
- status-filter presentation;
- contextual empty-state copy.

While the override is active, the status control displays **Unread**, is disabled, and explains in
its tooltip that All Unread always shows unread items. The prior manual filter is parked unchanged
and restored only after a different scope successfully commits. A failed pending navigation changes
neither the control nor the parked value.

Opening an unread item keeps the existing behavior: the item is marked read only after the write
succeeds, and **Mark unread** reverses it. The open item remains pinned in Feed Items and Reader even
though it no longer matches the unread predicate.

If that write reduces the selected feed's unread count to zero, its All Unread child remains visible
without a badge while the source scope is selected or pending. The display pin ends when:

- another scope commits;
- the All Unread branch is collapsed and later reopened;
- the app session ends; or
- Mark unread restores a positive count, making the pin unnecessary.

The pin changes presentation only. The query remains unread-only and never widens to all statuses.

## Reconciliation and refresh

Count and membership refreshes preserve expansion state, keyboard focus, committed scope, list
position, and Reader position.

Context invalidation follows explicit rules:

- a deleted selected source falls back to its nearest existing parent;
- a selected Unassigned child that becomes assigned falls back to Unassigned;
- an All Unread child reaching zero uses the bounded display pin instead of producing an invisible
  committed node;
- no scope may remain attached indefinitely to a node that is neither valid nor pinned.

A failed aggregate refresh may retain the last successful rows only for that exact branch and marks
them as potentially stale. It never substitutes rows from a different branch. With no usable
snapshot, the branch shows a non-interactive failure row and emits one concise notification per
failure episode.

Tree navigation remains navigation, not an implicit Inspector entity selection. Existing
management actions stay on their current entity-selection surfaces; this task does not couple a
Reader feed click to new management writes. The existing watchlist membership removal action is the
intentional exception already driven by contextual tree scope.

## Accessibility and copy

- Carets and labels remain independent focusable controls with action-specific tooltips.
- Expanded/collapsed and active state are exposed through the same widget state conventions as
  created watchlists.
- Feed names are escaped at the rendering boundary.
- Aggregate names use consistent title case: All Sources, Unassigned, All Unread.
- The empty Reader copy remains exactly `Select a feed to display it here.`
- No new keybinding is added; existing global and Read bindings remain unshadowed.

## Failure handling

- **Aggregate source load fails:** retain the exact branch's prior snapshot with stale indication,
  or show its failure row; do not clear unrelated branches.
- **Contextual item load fails:** keep the committed scope/list/Reader and restore active styling to
  it; offer Retry.
- **Superseded load completes:** discard it without notification.
- **Read/unread write fails:** preserve status, badge, branch membership, and Reader selection.
- **Source disappears or changes membership:** apply the reconciliation rules above after the
  authoritative snapshot commits.

## Verification

Automated coverage must prove:

1. All Sources, Unassigned, and All Unread expand and collapse independently from each other and
   from created watchlists.
2. Aggregate child populations are complete, correctly filtered, stably sorted, and use bulk
   loading without N+1 queries.
3. The same feed renders with unique root-qualified ids and only its exact contextual occurrence is
   active.
4. Caret activation changes expansion only; label and child activation request the correct scope.
5. Contextual source selections produce the predicates in this specification.
6. Parent context participates in pagination, search, and in-flight-load identity.
7. Pending, successful, failed, and superseded scope transitions obey ADR-042's atomic commit.
8. Search text persists while pagination, row selection, and Reader selection reset at successful
   commit.
9. All Unread forces and explains the temporary Unread filter without overwriting the parked manual
   filter.
10. Mark-read, Mark unread, zero-count display pinning, branch collapse, deletion, and Unassigned
    membership changes reconcile correctly.
11. Count refreshes and rail rebuilds preserve focus, expansion, scope, and Reader position.
12. Aggregate branches remain visible on every Watchlists sub-screen while Read-only predicates do
    not alter unrelated management data.

Run only the affected Watchlists tests plus modified-file Ruff and `git diff --check`, in accordance
with the workstream's established testing constraint.

## Scope boundary

In scope:

- contextual aggregate feed children;
- typed aggregate expansion state;
- authoritative bulk tree snapshots;
- contextual Reader queries and filter override;
- atomic navigation commit, recovery, and zero-unread pinning;
- focused Watchlists tests and documentation.

Out of scope:

- per-feed children beneath Today or Starred;
- tree virtualization or pagination without profiling evidence;
- persistence of tree-branch expansion across app restarts;
- database schema or server API changes;
- new feed-management actions;
- Media Library changes;
- an application-wide split-pane or navigation framework.

## ADR check

ADR required: no new ADR  
ADR path: `backlog/decisions/042-watchlists-reader-first-ia.md`  
Reason: ADR-042 already decides the reader-first Navigation hierarchy, normalized contextual scope,
atomic scope commits, and Watchlists-local ownership. This design is a direct, focused extension of
that accepted decision.


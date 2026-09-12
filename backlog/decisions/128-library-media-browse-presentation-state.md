# ADR-128: Library Media Browse presentation state

Status: Accepted (user approval, PR #2427)
Date: 2026-09-08
Related task: TASK-31932

## Context

LibraryMediaBrowseController mixes asynchronous page/facet orchestration with
pure presentation state and projections. It is 589 lines against its existing
371-line ceiling. The existing Media domain page types remain appropriate;
moving UI recovery state into those types would invert the domain/UI boundary.
The user approved a UI-local state separation without raising existing caps.

## Decision

Compose one `MediaBrowseState` from
`UI/Library_Modules/library_media_browse_state.py` as `controller.state`.
It owns requested/inflight/applied scope, retained records, freshness/loading
and recovery presentation, fault-episode fingerprints, and facet presentation.
Move the existing pure properties/methods and recovery-copy helpers there.
The state has no Screen, service, worker, event, callback, or persistence owner.

LibraryMediaBrowseController keeps both generation counters, admission checks,
worker groups, scheduling, service reads, cancellation, the bounded clamp loop,
and live view callbacks. Its admitted result paths write the state explicitly.
Do not invent another transition framework just to make the controller shorter.

Consumers access presentation through `.state`; callers of operations that
fence or schedule work continue using the controller. Mechanical retargeting
preserves writable fields, callback late binding, original ordering and tests.
No inheritance, generic attribute forwarding, compatibility field mirrors, or
cross-source pager abstraction is introduced.

ADR-067's bounded page, stale metadata, current-generation-only application,
mutation, and metadata-only diagnostics contracts are unchanged. Page/facet
failure priority, repeated-fault context, resume reset, one-clamp limit,
analysis-row projection, and known-committed mutation behavior are preserved.
No DOM, CSS, IDs, database schema, service signature, or persisted data changes.

## Alternatives

- Raise the controller ceiling: rejected by the approved unchanged-cap scope.
- Compress code/comments or move arbitrary line chunks: hides the ownership
  problem and makes behavior harder to inspect.
- Add inherited state/mixins or forwarding properties for every field: obscures
  ownership and creates duplicate interfaces with no invariant of their own.
- Move UI state into `Library/library_media_state.py`: gives domain types a UI
  recovery dependency and mixes presentation lifetime with service contracts.
- Extract a full transition API now: unnecessary new method boundaries when
  whole pure-method moves and explicit state writes suffice.

## Consequences and verification

One extra UI-local module and explicit consumer retargets are accepted. Keep
LibraryScreen off startup's eager path; verify actual import/payload budgets.
Tighten the existing controller pin to its measured lower count and give the
new state file an exact measured pin. No existing ceiling may increase.

Preserve existing behavioral assertions and add state independence/ownership
coverage before extraction. Run complete controller, retry/fault, paging,
mutation, mounted Media and Library shell tests, plus architecture and budget
guards. An extraction must not carry a behavioral repair; any new failure is
investigated and recorded separately.

Implementation plan:
`Docs/superpowers/plans/2026-09-08-pr2427-media-browse-state.md`.

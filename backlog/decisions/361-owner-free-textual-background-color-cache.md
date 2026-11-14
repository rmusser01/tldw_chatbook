# ADR-361: Owner-free native Textual background-color cache

Renumbered from ADR-227 under the oldest-date rule.

Status: Accepted — owner explicitly approved this bounded design in this chat
Date: 2026-10-21
Related task: TASK-31966

Placement update (ADR228): installer co-located at the existing lazy package
compatibility entry; original Utils import path remains an export. No new eager
Textual import or native cache/GC policy; warm module ceiling remains unchanged.

## Context

Pinned Textual8.2.8 `StylesCache.get_inner_outer` is a native 1,024-entry LRU
method whose calculation never reads its first (`cls`) argument. Its keys
nevertheless include each per-widget StylesCache instance, retaining that
instance's line cache after its widget otherwise retires. A disposable native
proof retains32 discarded owners; a color-only key retains0 with equal native
results, without collection. The runtime control remains NON-GREEN, so this
is ownership/allocation evidence, not a latency pass or unique GC attribution.

## Decision

At the existing UI/Widgets compatibility entry, install a repository-owned
adapter only for exact Textual8.2.8 and the verified native method/cache shape.
Validate version, source ownership, argument shape, unused first argument and
LRU parameters before any mutation. Unsupported or foreign methods retain
native behavior with a static warning.

Cache the ORIGINAL native calculation by its two immutable Color inputs only,
using standard-library `lru_cache(maxsize=1024)`. Keep LRU promotion/eviction,
native output and exceptions. This explicitly changes key ownership/deduplication,
not the capacity or line-cache policy. Existing instance line caches and custom
subclass overrides remain untouched. Installation is idempotent; do not clear
old caches or force retirement/collection. Adapter outputs own no widget/cache.

The cached native Textual styles are templates, never returned directly. A real cross-color-system RED showed
that these frozen styles still memoize mutable Rich ANSI state: sharing results
reused true-color escapes for256/standard renderers. Detached native styles keep
that memo private; color computation remains shared and bounded. Do not trade
appearance for fewer Style allocations.

Owner-approved refinement: every call still touches the global color-only LRU.
Each exact native StylesCache may reuse one current detached result pair while
its template tuple is identical. A different template, including recomputation
after global clear/eviction, replaces that single pair. No global owner reference
or reverse reference from results is added. Other owner types and subclasses
receive detached fields without private attributes; custom overrides stay native.
This restores same-owner native reuse without sharing mutable ANSI across owners.
The disposable control reduced14,194 pair allocations to791 but remained
NON-GREEN; that control is not a latency qualification or unique-cause claim.

No shared dependency edits, new dependency, collector thresholds/disable/freeze,
appearance, activation/focus/cancellation, resource boundary or timing-limit
changes. Existing ADR198/225/226 continue to govern those separate contracts.

## Alternatives and verification

Retaining owner keys preserves native behavior but demonstrably retains render
owners and repeats equal color allocations. Clearing caches periodically or on
unmount changes lifecycle/eviction and is rejected. Collector tuning, identity
divide eviction changes and inactive-layer hit-test changes are not authorized.

Native RED/GREEN weak-owner proof without collection, cross-owner deduplication,
same-owner pair reuse/clear invalidation, full LRU capacity/promotion/eviction,
old line caches, strict subclasses/overrides,
idempotence/mismatch controls, native color/render parity and collector invariance
precede affected tests/guards and a fresh unchanged original latency matrix.
Residual failures remain open.

Plan: [inline implementation](../../Docs/superpowers/plans/2026-10-21-task31966-owner-free-style-colors.md).

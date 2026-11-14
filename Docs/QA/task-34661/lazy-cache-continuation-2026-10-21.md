# Approved native render-cache allocation correction

The owner approved the bounded compatibility fix on 2026-10-21. See
[ADR226](../../../backlog/decisions/226-lazy-textual-strip-cache-allocation.md)
and its [inline plan](../../superpowers/plans/2026-10-21-task31966-lazy-strip-caches.md).
TASK31966 stays In Progress; allocation reduction is not latency clearance.

The existing UI/Widgets compatibility entry installs only against exact native
Textual 8.2.8 slots and constructor ownership. Unknown shapes retain native
behavior with a static warning. Exact-native Strips create each unchanged native
FIFO only on first use; all seven capacities, native eviction, values and render
methods remain unchanged. Existing caches survive installation; custom subclasses
use the original constructor. One initialization lock prevents competing first-use
owners. No installed dependency, appearance, collector policy, deadline, activation
authority, resource-retirement contract or other session was changed.

## Verification

- Original eager allocation RED: 9 failed / 5 passed in 0.79s.
- Added data-slot mismatch control RED: 1 failed / 15 passed in 0.96s;
  guarding all native slots addresses that unsupported-shape case.
- Lock-removal mutation RED: one failure in 0.31s exposes eight competing cache
  allocations. This is a mutation proof, not an original production failure.
- Custom subclass ownership RED: 1 failed / 18 passed in 1.11s; native-constructor
  fallback preserves the subclass's own assignment.
- Final native contracts: 19 passed in 1.03s. This includes all seven native FIFO
  capacities/eviction, pre-install values, idempotence, mismatch refusal, real
  threaded first use and isolated native-versus-adapted Unicode/ANSI/filter/style/
  selection parity. Collector thresholds, enablement and freeze count are unchanged
  across both normal compatibility entry points; no forced collection was used.
- Affected Console/renderer/boot-GC selection: 188 passed in 211.84s. After the
  final subclass fallback, native contracts plus setup, approval, countdown and
  original private geometry: 102 passed in 39.78s. Both processes exited zero.
- New module and CI tests: Ruff clean; both plus package initializer format clean.
  Existing unrelated inherited lint findings are not described as clean.
- All eleven artifact guards pass. The diagnostic inventory correctly rejected the one
  new static fallback warning before its reviewed regeneration. It interpolates
  no environment, user content, secrets or paths and creates no sink. The rebuilt
  inventory verifies 657 owners and unchanged 16 sink files.

An earlier failed patch left the old fourteen tests unchanged; its passing log
is not credited as RED. All raw failure, probe, parent and child logs remain LOCAL
under `/private/tmp/console-latency-continuation-fT3VrM`; only audited outcomes and
hashes are published. No failures or warnings are suppressed.

Next: commit locally, then new exact-head scale build, Keyword300 and the original
70-case UI matrix with the unchanged 50ms loop / 100ms busy-paint limits. No current
latency pass, native/Windows/participant qualification, full sweep, push or merge
is claimed here. Keep every residual finding open and continue measured fixes.

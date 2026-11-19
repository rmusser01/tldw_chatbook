# ADR-226: Lazy native Textual Strip cache allocation

Status: Accepted — owner explicitly approved the bounded compatibility fix
Date: 2026-10-21
Related task: TASK-31966

Placement update (ADR362): installer co-located at the existing lazy package
compatibility entry; original Utils import path remains an export. This removes
an extra warm-boot resident without changing native allocation/cache behavior.

## Context

The pinned Textual8.2.8 Strip constructor eagerly creates seven native FIFO
caches per rendered line. A disposable original-driver control at b8ecfdcae9
allocated32,412 FIFOs instead of roughly840,000 in other eager-cache controls;
only a minority of per-line caches were used. This reduces allocation and the
tracked render-cache graph, but does not prove unique GC causation or a latency
pass. All controls remain NON-GREEN; the original50ms/100ms limits still apply.

## Decision

Install an app-owned allocation adapter through the existing UI/Widgets Textual
compatibility entry point. Apply only to exact Textual8.2.8 with the verified
native slot and constructor shape. Validate everything before modifying the
class; unsupported/foreign shapes keep native behavior and emit a static warning.

New exact-native Strips initialize the same segments, cell length, render string, link ids
and cell count. Each of the seven cache slots starts empty and constructs the
original native FIFOCache on first access, once per Strip/slot. Capacities remain
4/16/16/4/4/4/4 in native constructor order. Native lookup, values, hits/misses,
FIFO eviction and every Strip operation remain unchanged. Preserve cache objects
already owned by Strips created before installation; installation is idempotent.
A bounded initialization lock protects first allocation, not rendering or FIFO
operations. No new render-result cache, capacity change or global retention pool.
Custom subclasses use the original constructor so subclass-owned slots keep their
native assignment and ownership semantics; they are not silently adapted.

All changes live in repository code and only affect its importing process.
Never edit the shared installed dependency. No new dependency, collector
threshold, disable, collection, freeze, appearance, timing-limit, activation,
resource-retirement or other-session change is authorized by this decision.

## Alternatives and limits

- Retain native eager allocation: safe fallback for unknown framework shapes,
  but leaves the measured unused-cache allocation in the verified version.
- Tune/disable GC, force collection or freeze later heaps: rejected; ADR198 and
  existing lifetime/qualification contracts remain unchanged.
- Opaque/frozen background, result pooling or altered cache eviction: outside
  this approved design; appearance, freshness and ownership must stay intact.
- Upgrade or patch the shared environment: rejected in this checkout. A future
  dependency bump must revisit/remove the adapter with native parity evidence.

## Verification and rollout

Native RED/GREEN allocation tests, all seven FIFO capacity/independence/eviction
checks, pre-existing Strip preservation, repeat-install and mismatch controls;
isolated original-versus-adapted segment/ANSI parity including Unicode, crop,
divide, filtering, styles and selection offsets. Then affected renderer/Console
and GC-contract tests, guards and fresh uninstrumented original matrix.
Object-count improvement is not completion. Keep every residual failure open.

Links: [inline plan](../../Docs/superpowers/plans/2026-10-21-task31966-lazy-strip-caches.md),
[audited isolation](../../Docs/QA/task-34661/allocation-continuation-2026-10-21.md),
[ADR198](198-gc-policy-freeze-boot-heap.md).

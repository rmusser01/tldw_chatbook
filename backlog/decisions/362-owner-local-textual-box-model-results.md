# ADR-362: Owner-local native Textual BoxModel result reuse

Renumbered from ADR-228 under the oldest-date rule.

Status: Accepted — explicitly approved by the owner in this chat
Date: 2026-10-21
Related task: TASK-31966

## Context

Native Textual8.2.8 Widget geometry keeps a16-entry keyed LRU. Different native
keys can produce equal immutable BoxModel/Fraction/Spacing values. Disposable
native control computed8,217results and reused5,164equal retained outputs, but
still failed five activation intervals. This is retained-allocation evidence,
not a latency pass, unique GC cause or permission to skip native geometry.

## Decision

Install one repository-owned adapter at the existing UI/Widgets compatibility
entry, only for verified Textual8.2.8/native Widget method and LRU shape. Unknown
version/foreign method/cache stays native. Installation is idempotent. Subclass
BoxModel method overrides, including calls through super, remain untouched.

Always call the original method, including every original keyed lookup,
promotion, miss, native geometry calculation, insertion and eviction. After a
new native miss, reuse the previous MRU value only when both results are exact
immutable native BoxModel/Fraction/Spacing values and equal. Replace only the
new current link's value; keys, links, counters, capacity and native side effects
stay unchanged. Validate the same cache and directly preceding MRU link remain
current; clear/replacement/reentrant changes fall back to the native result.
No extra keyed lookup, new owner field, global result map or owner reference.

The actual warm boot census exposed three extra resident adapter modules:
1036 against the unchanged1033 limit. Co-locate all three installers inside
the existing package compatibility entry with function-local dependency imports;
keep their original Utils import paths as compatibility exports. Existing Strip
and color executable AST remains identical apart from import scope/installation
flags; no eager Textual import is added to plain package import. Native tests
and original warm/cold guards return1033/1033 and681/686. Do not raise budgets.
Fingerprint validation reads source without populating global linecache; its
original source-retention RED is retained. No other cache is cleared or tuned.

The existing cache owns these values. Clear/replacement naturally ends reuse;
no separate memo survives it. Result identity may be shared between equal keys
of one native owner, never across owners. No shared dependency edit, new
dependency, appearance/geometry/activation/focus/cancellation/lifetime change,
collector disable/threshold/freeze/forced collection or timing-limit change.
ADR198/225/226/227 continue to govern their separate boundaries.

## Alternatives and verification

Global interning risks cross-owner retention; a separate per-owner memo adds
state and invalidation. Both are unnecessary because the native MRU already
owns the candidate. Skipping computation or changing LRU keys/capacity/eviction
changes native semantics and is rejected. Collector tuning is not authorized.

Native identity RED precedes implementation. Verify exact output/native side
effects, all misses and hits, LRU ordering/eviction/capacity, clear/replacement,
owners/subclasses/custom caches/errors/unsupported installation and idempotence.
Native geometry/Unicode paint, collector-invariance and affected shipped callers
precede static/artifact guards and fresh clean exact-head scale/Keyword/original
UI70 qualification. Residual latency and external qualification remain open.

Plan: [inline continuation](../../Docs/superpowers/plans/2026-10-21-task31966-owner-free-style-colors.md).

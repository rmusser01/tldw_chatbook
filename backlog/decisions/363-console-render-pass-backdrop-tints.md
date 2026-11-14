# ADR-363: Console render-pass backdrop tint reuse

Renumbered from ADR-229 under the oldest-date rule.

Status: Accepted — explicitly approved by the owner in this chat
Date: 2026-10-21
Related task: TASK-31966

## Context

On clean c79dc443, passive native render observation attributes 94 Rich renders
to the switcher's live BackgroundScreen (458.19ms total, 37.98ms maximum).
A disposable per-render color-pair control reduces 59,398 tint calculations to
4,284 over 133 live frames, but still fails four activation intervals. These
are allocation/call-path observations, not latency clearance or unique GC cause.
A separate scheduling-only 1ms-to-10ms control still fails six intervals; it is
not adopted. All original 50ms/100ms limits and failures remain evidence.

## Decision

Only the exact ConsoleSessionSwitcherModal may replace an exact native
BackgroundScreen returned by its unchanged super().render() with a private
app-owned subclass, inside its existing lazy module. Pin Textual8.2.8 and the
native source/method identities; unsupported versions/shapes/overrides remain
native. No global monkeypatch, new dependency or additional resident module.

Each live render still invokes the underlying compositor with current options.
Only that render iterator owns a memo of at most 128 pairs of immutable native
foreground/background colors to tint-only styles. Preserve native color
calculation. On overflow calculate normally without eviction or more entries.
No screen/widget/style owner is a memo key. No frame snapshot, later-frame
reuse, global or owner-local cache. The iterator's normal completion, close or
failure ends its memo lifetime.

Every segment still performs its native metadata/link cleanup and Style-add
LRU access, in order, even on a tint hit. Preserve controls, private ANSI state,
attributes, palettes, opacity, geometry, live content and exception behavior.
Transparent input and foreign types use native processing. No GC policy,
activation authority, focus, cancellation, teardown or timing-limit change.

## Alternatives and verification

Opaque backdrops or cached frame snapshots lose live underlying content and
change appearance. Global tint/style-owner caches add retention and invalidation.
Skipping native cleanup/addition cache touches changes their access/eviction.
These are rejected. This bounded allocation correction does not claim to solve
the remaining GC and non-GC scheduling work by itself.

First require a native allocation RED through the actual modal render boundary.
Then compare native segments, ANSI/palettes, metadata/links/controls, cache
touches/order, overflow, transparency, foreign types, live mutation/resize,
early close/errors, unsupported fallback and other modals. Run original affected
switcher/activation/staging/native phase/resource and startup guards, static
analysis and inline review. Safe prose/hashes only; raw diagnostics remain local.
Commit locally, then fresh exact-source scale/Keyword/original UI70 qualification.

Plan: [inline continuation](../../Docs/superpowers/plans/2026-10-21-task31966-owner-free-style-colors.md).

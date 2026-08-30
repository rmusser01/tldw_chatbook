# Library Media Return Settlement Design

**Date:** 2026-08-30
**Status:** Approved in chat; written-spec review pending
**Related task:** TASK-18918
**Decision:** [ADR-104](../../../backlog/decisions/104-library-media-return-settlement-boundary.md)

## Purpose

Normal Media must restore the retained page, semantic row, focus identity, and exact
list scroll after a Trash round trip or authoritative same-scope recompose. The current
flow carries the correct receipt but can apply it against a replacement scroll owner
before that owner has the Media adaptive presentation and final geometry. Textual then
truthfully clamps the desired `(0, 42)` to a transient or wrong-contract maximum such as
`(0, 33)`.

The repair must make restoration event-driven and deterministic. Scheduler turn counts,
arbitrary sleeps, framework-private layout signals, and recursive polling are not
readiness contracts.

## Evidence and corrected root cause

The established cross-reader test restores the correct Media row but intermittently
observes `(0, 33)` instead of the retained `(0, 42)`. The receipt remains intact through
capture, Trash Back, recompose, and focus arming.

Diagnostic runs established two separate prerequisites:

1. A newly composed `#library-shell-grid` can retain the legacy
   `library-notes-compact` class because the existing Media shell mount handler only
   synchronizes width. A same-size whole-screen recompose need not produce another
   screen resize, so the replacement Media owner can remain at virtual height 40 and
   `max_scroll_y == 33` for longer than the existing two-second focus arm.
2. Even with a corrected presentation, `call_after_refresh()` and `call_later()` are
   queue boundaries, not descendant-layout completion signals. One or two later turns
   sometimes pass and sometimes reproduce the same clamp.

Textual 8.2.8 posts a widget's non-bubbling `Resize` only after compositor reflow calls
`_size_updated(size, virtual_size, container_size)`, which installs all three geometry
values, updates scrollbars, and clamps current scroll. The current owner's own Resize is
therefore the public geometry boundary, but only after the application has projected the
correct Media presentation.

## Decision

Use a two-gate, event-driven return-settlement protocol owned by `LibraryScreen`:

1. **Presentation gate:** after the current Media adaptive shell mounts or reports its
   shell lifecycle message, reconcile the outer Library stage to the current adaptive
   Media presentation. The stage must have `library-adaptive-compact` and must not retain
   `library-notes-compact` when the Media adaptive reader is current. Reconciliation
   advances an application-owned presentation epoch and requests layout only when the
   projected class contract changes.
2. **Geometry gate:** the current Media row-scroll owner translates its own Textual
   `Resize` into an owner-identified application message carrying a monotonic geometry
   revision plus `size`, `virtual_size`, and `container_size`. `LibraryScreen` settles a
   retained return only from current-owner geometry for the required presentation epoch
   and only when the exact retained offset is representable.

No timer or callback count proves readiness. Real producer changes advance epochs; real
owner geometry events trigger attempts. Duplicate events are coalesced and idempotent.

## Components

### Media row-scroll owner

A small Media-owned `VerticalScroll` subclass remains a presentation widget. It:

- records its latest Resize-derived geometry and increments an owner-local revision;
- posts one owner-specific geometry message per distinct revision;
- does not own return receipts, focus policy, route state, timers, or page data; and
- emits no polling work.

The message must identify the concrete sender. A message from an old or detached owner
cannot settle a current request.

### Presentation reconciliation

The existing Media adaptive-shell lifecycle handler must run the same stage projection
that screen resize already uses before synchronizing shell width. Projection remains
equality-guarded: unchanged classes and geometry do not recompose, reload data, write
preferences, or create work.

Presentation epoch changes whenever an action can change the effective outer compact
contract or current Media owner, including adaptive-shell replacement and relevant
viewport/layout projection. The epoch is transient and screen-owned.

### Return-settlement request

An immutable request records:

- ABA-safe request identity;
- retained `(media_id, scroll_offset)` receipt;
- focus-intent, compose, Media lifecycle, and presentation generations;
- applied Media content/page revision and ordered stable IDs;
- route and Media sub-view identity;
- viewport/layout signature; and
- current shell, Items host, and row-scroll owner identities once attached.

The existing retained return receipt remains authoritative. Transient framework clamping
must never overwrite it.

## Settlement flow

1. Back or authoritative recompose arms one immutable settlement request from the
   retained receipt. A newer request invalidates the older object.
2. Mounting the replacement Media adaptive shell reconciles the presentation contract
   and advances or confirms the required presentation epoch.
3. The current row-scroll owner emits geometry after Textual reflow. If it emitted before
   the request was armed, its latest recorded revision may be evaluated immediately.
4. `LibraryScreen` verifies every authority fence and the live geometry payload.
5. For an unchanged content and layout revision, settlement is eligible only when the
   exact desired offset lies within the owner's current maxima.
6. In one synchronous commit, resolve the semantic row, focus it with
   `scroll_visible=False`, apply the unanimated exact offset, verify the resulting
   integer offset equals the receipt, and record `exact-settled` for that request and
   geometry revision.
7. Focus must not become the observable success boundary before exact scroll succeeds.
   Repeated current-owner messages after success are no-ops.

## Authority fences

Every attempt must reject when any of these differ from the request:

- request object identity;
- focus-intent, compose, Media lifecycle, or presentation generation;
- route, Media sub-view, Items-open state, or semantic target;
- current shell, Items host, row-scroll owner, attachment, or ancestry;
- retained receipt, applied scope, content revision, or ordered stable IDs;
- viewport/layout signature for an exact return; or
- live geometry versus the message payload.

User keyboard/mouse takeover, foreign focus, another Back request, route change,
content change, recompose, adaptive layout change, and unmount synchronously invalidate
the obsolete request. `is_mounted` alone is insufficient; current ancestry and owner
identity are required.

## Content or layout changes

Exact physical scroll is authoritative only while the content and viewport/layout
revision captured with the receipt remain unchanged.

If a controller/content revision proves that the list changed, or a viewport/layout
revision proves that the physical coordinate system changed, the new epoch may perform
one explicit semantic-row focus with an honest clamp from current geometry. It records
`clamped-after-revision`, never `exact-settled`. A timeout cannot be used to infer that
content shrank.

This preserves the TASK-18918 contract: Trash navigation and Restore do not replace or
rerank the retained normal-Media page. Restore only marks it stale pending an
authoritative normal-Media refresh.

## Bounded failure

Reuse the existing two-second list-entry focus deadline as a liveness bound, not as
readiness evidence.

If an unchanged request has no representable current-owner geometry by the deadline:

- record `layout-settlement-failed`;
- clear the ephemeral settlement request and timer;
- emit one metadata-only, privacy-safe warning; and
- perform at most one honest semantic-focus fallback against current geometry without
  reporting exact success.

No callback is re-enqueued after failure. Route change, cancellation, or unmount clears
the request without warning.

## Alternatives rejected

| Alternative | Reason rejected |
| --- | --- |
| Immediate focus and scroll after recompose | Applies the receipt against transient or wrong-contract geometry and clamps silently. |
| Presentation reconciliation alone | Removes one deterministic defect but does not make scheduler timing a layout boundary. |
| One, two, or N `call_later()` / `call_after_refresh()` turns | Repeated fresh-process trials proved these remain nondeterministic. |
| Poll until `max_scroll_y` grows | Creates a timing loop, confuses legitimate shrink with lateness, and has no producer authority. |
| Watch `virtual_size` only | Scroll maxima also depend on container size and current owner identity. |
| Use `screen_layout_refresh_signal` | It is private/semi-private and describes one screen reflow, not completion of the current application presentation epoch. |
| Change the expected scroll to `(0, 33)` | Hides the failure and violates the retained exact-return contract. |

## Testing

TDD retains the existing consumer-visible `(0, 42)` RED unchanged. New focused tests
must establish:

- same-size recompose reconciles adaptive Media presentation before settlement;
- focus is withheld until current-owner Resize-derived geometry is eligible;
- current geometry makes semantic focus and exact scroll observable together;
- duplicate geometry is idempotent;
- old owner, stale epoch, stale request, user takeover, route change, and unmount cannot
  settle;
- a proven content/layout revision clamps once and is labelled honestly;
- no geometry reaches one bounded failure with no further queued attempt; and
- no scheduler-turn count appears in production or tests.

The established exact test must pass in five consecutive fresh processes. Then run the
focused normal-Media Back/recompose/adaptive-layout tests, the production-shaped
cross-reader gate, TASK-18918 Trash renderer/lifecycle/mutation gates, and the live
160×50, 120×35, 100×30, and 80×24 walkthrough. A repository-wide test sweep remains
out of scope without explicit approval.

## Non-goals

- Generalizing settlement across all Library readers in this change.
- Altering Notes' existing restoration scheduler.
- Persisting presentation epochs or settlement outcomes.
- Changing Media pagination, filtering, selection, or stale-page ownership.
- Adding sleeps, polling workers, or new dependencies.

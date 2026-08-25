# Watchlists Responsive Layout Hysteresis Design

**Date:** 2026-08-25  
**Task:** TASK-22211  
**Status:** Approved for implementation planning  
**Related ADR:** [ADR-042: Watchlists reader-first information architecture](../../../backlog/decisions/042-watchlists-reader-first-ia.md)

## Problem

Watchlists derives its effective pane layout from bare width thresholds on every
Textual resize event. A one-column oscillation around a threshold can therefore
collapse and reopen a pane repeatedly. Each effective transition asks the workbench
to remove or mount an entire side-pane subtree, making a terminal resize drag or a
scrollbar-induced width fluctuation visibly unstable and unnecessarily expensive.

The Library adaptive reader already prevents the same failure with a four-column
reopening hysteresis band. Watchlists should use that established interaction policy
without changing the user's persisted manual pane preferences or the reader-first
layout defined by ADR-042.

## Goals

- Keep the effective Watchlists layout stable during one-column oscillation around
  every responsive collapse boundary.
- Use the Library reader's exact four-column hysteresis precedent.
- Keep explicit user actions immediate and predictable.
- Derive responsive state from a width that is not altered by child scrollbars.
- Avoid issuing workbench layout requests when the effective layout did not change.
- Cover the behavior with deterministic pure-state tests and a controller-level
  regression test for request/mount churn.

## Non-goals

- Changing pane widths, collapse order, breakpoint locations, or grip behavior.
- Changing Article Focus behavior or persisted manual layout semantics.
- Introducing time-based resize debouncing.
- Creating a shared application-wide split-pane abstraction.
- Modifying the parallel Media Library redesign.

## User-visible behavior

For each existing collapse threshold `T`:

- An open pane collapses when available width falls below `T`, exactly as it does
  today.
- A pane that was responsively collapsed remains collapsed until available width is
  at least `T + 4`.
- Once reopened, it remains open until width again falls below `T`.

For example, the all-open Read layout currently collapses Inspector below 145
columns. After this change, Inspector collapses at 144 and does not reopen at
145–148; it reopens at 149. After reopening, it remains open at 145 and collapses
again only at 144.

This rule applies at the Inspector, Navigation, and Feed Items boundaries on Read,
and at the mounted side-pane boundaries on management tabs. The established
responsive collapse priority remains Inspector, Navigation, then Feed Items.

## State model

`resolve_effective_layout` accepts an optional previous effective layout. Resolution
still begins from the persisted preferred layout and the current mounted pane set.
The previous layout is evidence only for whether a pane is reopening after a
responsive collapse; it never becomes persisted preference state.

The resolver first derives the nominal result using the existing minimum-width and
priority rules. For each pane that is nominally open but was collapsed in the
previous effective layout, reopening requires the nominal boundary plus
`LAYOUT_HYSTERESIS_WIDTH`, set to `4` to match Library. Previously open panes retain
today's collapse threshold, producing an asymmetric dead band rather than shifting
the breakpoint in both directions.

The first settled layout has no previous responsive state and therefore resolves
without hysteresis. This avoids treating the wide pre-mount fallback or an
unmeasured initial geometry as user-visible history.

## Intent boundaries

Hysteresis is used for passive resize recomputation only. The following explicit
transitions resolve without a previous layout and remain immediate:

- a manual grip or keyboard pane toggle;
- entering or leaving Article Focus;
- switching between Read and a management section;
- restoring or rolling back a failed manual layout request.

This separation prevents a prior Article Focus layout, in which every side pane is
collapsed, from delaying restoration when Article Focus exits. It also ensures that
an explicit request to open a pane is honored immediately and continues to receive
the existing temporary priority treatment.

When a manually prioritized pane can be released because all preferred panes fit,
the passive resize path uses the hysteresis-stabilized unprioritized result. The
priority target is therefore not cleared by a sub-hysteresis width fluctuation.

## Width authority

Responsive resolution uses the Watchlists screen's own settled allocation width as
the invariant authority. Child workbench content and container widths are excluded
because they may reflect descendant scrollbar allocation. Before the screen has a
positive settled size, resolution keeps the existing wide fallback so composition
does not collapse panes based on zero-width geometry.

The resolver's four-column dead band remains a second code-level defense. A future
one-column measurement fluctuation therefore cannot reintroduce pane churn even if
the width authority changes later.

## Workbench synchronization

After resolution, the controller compares the new effective layout with the current
effective layout. If they are equal, it does not allocate a new request token and
does not update the workbench's effective-layout request reactive. This prevents a
stable resize sequence from entering the asynchronous layout application path at
all.

When a real collapse or expansion occurs, the existing token fencing, focus
handoff, pane factory, mount/remove, rollback, and persistence behavior remains
unchanged.

## Testing strategy

Pure resolver tests will cover both directions at every representative boundary:

- collapse immediately below the threshold;
- remain collapsed through `T`, `T + 1`, `T + 2`, and `T + 3`;
- reopen at `T + 4`;
- remain open while shrinking back to `T`;
- collapse again below `T`;
- repeat one-column oscillations without changing the effective layout;
- reopen multiple previously collapsed panes in reverse collapse order without
  reopening them together before each pane's own buffered boundary;
- preserve manual collapse preferences, priority behavior, Article Focus, and
  management-tab mounted-pane rules.

A focused screen/controller test will feed passive resize widths around a boundary
after explicitly establishing the first settled layout, then assert that
sub-hysteresis changes do not allocate new workbench requests. This separates
initialization behavior from passive resize behavior. The test will also verify that
explicit manual and Article Focus transitions bypass hysteresis. Existing workbench
tests continue to prove that only actual effective layout changes mount or remove
region bodies.

Verification is limited to tests covering the changed Watchlists layout resolver,
screen recomputation path, and affected workbench contract, per the user's requested
test scope.

## Alternatives considered

### Time-based resize debounce

A debounce could reduce the number of transitions during a drag, but it would add
perceived latency, depend on timing-sensitive tests, and could still settle on the
wrong side of a scrollbar-induced width change.

### Quantized width buckets

Rounding widths would be simple but would move effective breakpoints and make their
relationship to pane minimum widths difficult to reason about.

### Previous-state hysteresis

Chosen. It is deterministic, pure-state testable, preserves existing collapse
thresholds, and matches the established Library reader behavior.

## ADR assessment

**ADR required:** no  
**ADR path:** `backlog/decisions/042-watchlists-reader-first-ia.md`  
**Reason:** This is a bounded stabilization of ADR-042's existing responsive pane
policy and follows an existing Library implementation precedent. It introduces no
new storage, ownership, cross-module interface, dependency, or long-lived UX
structure decision.

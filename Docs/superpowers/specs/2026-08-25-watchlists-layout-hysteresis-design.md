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

With every Read side pane preferred open and all three previously responsive-
collapsed, reopening proceeds one pane at a time in reverse collapse order:

- Feed Items reopens at 95 columns;
- Navigation reopens at 119 columns;
- Inspector reopens at 149 columns.

On management tabs, Navigation reopens at 82 columns and Inspector at 112. A jump
across several buffered boundaries may reopen several panes in one resolution, but
only panes whose own boundary has been crossed may reopen.

## State model

`resolve_effective_layout` accepts an optional previous effective layout. Resolution
still begins from the persisted preferred layout and the current mounted pane set.
The previous layout is evidence only for whether a pane is reopening after a
responsive collapse; it never becomes persisted preference state.

The resolver first derives the nominal result using the existing minimum-width and
priority rules. For each pane that is nominally open but was collapsed in the
previous effective layout, reopening requires the nominal boundary plus
`LAYOUT_HYSTERESIS_WIDTH`, set to `4` to match Library. Previously open panes retain
today’s collapse threshold, producing an asymmetric dead band rather than shifting
the breakpoint in both directions.

Nominal collapse is applied before hysteresis: the initial accepted-open set is the
intersection of panes open in the nominal result and panes open in the previous
effective layout. A pane the nominal result collapses therefore closes immediately
below its ordinary threshold, regardless of its previous state.

The remaining nominally open, previously collapsed reopen candidates are evaluated
deterministically in reverse collapse order. When a priority lease is active, this
means the reverse of the effective candidate order after moving the protected target
to collapse last. For each reopen candidate, calculate the required width with every
pane already accepted open plus that candidate. Accept the reopen only when the
available width is at least that required width plus four; otherwise leave the
candidate collapsed before evaluating the next one. Preferred-collapsed and
unmounted panes are never reopen candidates. This makes a multi-pane result
independent of set iteration order and prevents two panes from borrowing the same
width budget.

Responsive history begins at the first positive `self.size.width`. That first
positive allocation resolves without hysteresis and becomes the previous responsive
layout for later passive resize events. Pre-layout persisted state, the wide compose
fallback, and zero-width measurements are not responsive history.

## Intent boundaries

Every controller recomputation must explicitly state whether it is a passive resize;
there is no default value. Only a passive resize may supply the previous responsive
layout to the resolver. The following explicit transitions resolve without a
previous layout and remain immediate:

- a manual grip or keyboard pane toggle;
- entering or leaving Article Focus;
- switching between Read and a management section;
- restoring or rolling back a failed manual layout request.

This separation prevents a prior Article Focus layout, in which every side pane is
collapsed, from delaying restoration when Article Focus exits. It also ensures that
an explicit request to open a pane is honored immediately and continues to receive
the existing temporary priority treatment.

An explicit manual open creates a mode-local priority lease containing the target
pane and the originating mode (Read or management). Another manual open in that mode
replaces the target. Manually closing the target clears the lease. Article Focus
suspends the lease without clearing it, and leaving its originating mode parks it;
exiting Article Focus or returning to the originating mode resumes it. A lease from
Read therefore cannot be cleared merely because a management tab has fewer mounted
panes, and vice versa.

The lease clears only during a passive resize in its originating mode after the
hysteresis-stabilized unprioritized result proves that every preferred pane mounted
in that mode fits. A sub-hysteresis fluctuation cannot clear it. A failed manual
layout request restores the entire previous lease (target and originating mode), and
a failed section swap leaves a parked lease unchanged. This makes tab round-trips
and Article Focus entry/exit produce the same pane arrangement at the same width.

An explicit open inside a reopening dead band takes effect immediately. That pane is
then part of the previous open layout, so later passive events at the same width keep
it open; it collapses only when width falls below its ordinary unbuffered threshold.

## Width authority

Responsive resolution uses a positive `self.size.width` as the invariant authority.
Child workbench content and container widths are excluded because they may reflect
descendant scrollbar allocation. Before the screen has a positive width, composition
may keep the existing wide preferred-layout fallback, but controller recomputation
does not change effective responsive state, seed history, allocate a request token,
or contact the workbench.

If `self.size.width` transiently returns to zero after responsive history exists, the
controller retains the last positive responsive state and performs no recomputation.
The next positive measurement resolves against that retained history. A zero width
never reintroduces the compose fallback or becomes a collapse/expand boundary.

The resolver's four-column dead band remains a second code-level defense. A future
one-column measurement fluctuation therefore cannot reintroduce pane churn even if
the width authority changes later.

## Workbench synchronization

After resolution, the controller compares the new effective layout with its current
desired effective layout, including while an earlier asynchronous request is still
in flight. If they are equal, it does not allocate a new request token and does not
update the workbench's effective-layout request reactive. This prevents a stable
resize sequence from entering the asynchronous layout application path at all.

An apply rollback record is created only when an explicit action produced a real
workbench request token. An explicit preference change that does not change desired
DOM layout may still persist the changed preference, but it cannot attach rollback
state to the previous request's stale token. Failed real requests keep the existing
token fencing and restore preferred layout, Article Focus state, and the full
priority lease captured before the attempt.

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
- verify that an active priority lease reverses the priority-adjusted candidate order
  rather than the default collapse order;
- cover the Read reopening points 95, 119, and 149 and management reopening points
  82 and 112;
- ignore zero width before and after the first positive screen allocation without
  seeding history or issuing a request;
- preserve manual collapse preferences, priority behavior, Article Focus, and
  management-tab mounted-pane rules.

A focused screen/controller test will feed passive resize widths around a boundary
after explicitly establishing the first settled layout, then assert that
sub-hysteresis changes do not allocate new workbench requests. This separates
initialization behavior from passive resize behavior. The test will also verify that
explicit manual and Article Focus transitions bypass hysteresis; a manual open inside
the dead band survives the next same-width passive event; a priority lease parks and
resumes across Article Focus and mode changes; rollback restores the full lease; and
zero-width events are no-ops. Existing workbench tests continue to prove that only
actual effective layout changes mount or remove region bodies and will assert that
no-op widths do not move keyboard focus.

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

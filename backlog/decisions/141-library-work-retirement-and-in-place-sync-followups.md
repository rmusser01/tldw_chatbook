# ADR-141: Library Work retirement and in-place sync follow-ups

Status: Accepted for implementation; user approved the coordinated repair on 2026-09-09.

## Context

TASK-31932's phase-C integration exposed two lifecycle gaps. Removing a visible
outgoing Work subtree allowed Textual to paint a TextArea after its component
styles were released. Hiding before mount and retiring after refresh prevented
that crash, but a rapid same-ID return could mount before the old owner retired.
Separately, an in-place Notes bulk update queued no recompose; a widget-side
deferred callback drain could be invalidated while leaving an old callback parked.

## Decision

The existing browse shell owns retirement of its exact outgoing Work instances.
Incoming Work is always the fresh constructor result. It must not be replaced
by an old mounted Media viewer: construction consumes arrival/Find state and
builds current detail/edit/highlight inputs, while route adoption syncs Items.
Outgoing Work is hidden before awaits and ordinary retirement follows refresh.
A same-ID hidden predecessor must retire before the fresh owner mounts, with
exact-parent/current-owner guards making a later retirement callback harmless.
The implementation must prove compositor safety for rapid same-ID replacement;
if direct hidden-predecessor retirement is unsafe, use a refresh-completion
boundary in this same shell rather than a reuse cache or global cleanup.

Notes sync reports when a bulk presentation update preserves its editor children.
The existing canvas-sync coordinator chooses the follow-up owner using that
result: a retained Work pane must not receive a callback waiting for a recompose
that will not happen. The list canvas still recomposes and can own its exact
post-sync follow-up. Clear superseded Work callbacks at the coordinated in-place
boundary; preserve explicit follow-ups, current-user-focus vetoes and generation
guards. Remove the tentative widget helper that reads future callback state.

## Alternatives

- Immediate ordinary prune: rejected by three reproduced TextArea style crashes.
- Retain/reuse old Work: rejected because it discards fresh Media state/one-shots.
- Widget-side callback polling/draining: rejected because generations do not
  identify the exact queued callback and can leave orphaned follow-ups.
- New retirement registry or generalized sync scheduler: not needed for this
  bounded existing-owner repair; do not introduce without further evidence.

## Verification and consequences

Mounted controls must prove fresh object/current content, unique IDs, safe render
and final removal through rapid route cycles. Notes controls must prove unchanged
text, selection and undo, correct bulk read-only state, exact-once explicit
follow-ups, and no obsolete follow-up after a newer editor-owned sync. Preserve
existing resource, focus, route and size guards. No new storage, dependency,
service authority, user-facing layout or cleanup authority is introduced.

Related: TASK-31932; Docs/superpowers/specs/2026-09-01-library-screen-decomposition-design.md.

# ADR-365: Console transcript viewport ownership

Renumbered from ADR-231 under the oldest-date rule.

Status: Accepted for bounded retention
Date: 2026-11-13
Related task: TASK-31966

The Console has one active transcript. Today its message widgets belong to
whichever session last populated the shared viewport. An Active switch between
disjoint histories rebuilds every message row. A disposable Terminal control
kept two native viewports and eliminated all 50 direct row registrations on
each warm scale switch. It approximately halved Return-to-transcript time
in the matched scale sample. The full 12-trial control still exceeded 50 ms
on every trial. A warm trace reached a 133.94 ms loop gap with zero row
registrations and no overlapping generation-two collection.

The transcript surface owns the mounted viewport used by Console consumers.
Use its explicit viewport reference for reads, projection, focus and transient
voice state. Keep the existing ID for the single-view path and standalone
transcript consumers. A mounted production surface with no active viewport
has no transcript target. Keep exact session authority with the existing
store and Console activation path. The viewport is a disposable projection.

Hiding and disabling a viewport are insufficient to retire its input.
A mounted control queued an existing selection-quote message, hid and
disabled its sender, and changed the surface's current viewport before
returning to the loop. The original quote still reached the new draft.
The effect boundary must validate the originating activation before acting
on a queued message or focus callback. A check of the currently visible
viewport alone cannot distinguish two visits to the same viewport.

A follow-up can retain the active viewport and at most one previously viewed
viewport. Retention requires a bounded, settled view and the same live runtime
authority, session incarnation and conversation binding. The inactive viewport
must be excluded from layout and input, including queued actions and focus
restoration. It must release transient selection, menus and mouse capture.
Media, timers, workers and pending view callbacks must either be quiescent or
remain on the single-viewport path. Closing or rebinding a tab, switching
profile, replacing the surface and quitting must retire its view. Reproject
current messages and metadata before publishing a retained viewport. Publish
only for the current activation epoch and preserve the existing mount,
composer-focus, cancellation and exact-target readiness proofs.

The owner approved activation-scoped actions and one retained prior viewport.
The store's activation epoch invalidates an old visit as soon as it changes.
A published viewport also records its input cutoff on Textual's message clock.
An action keeps its origin through descendant dispatch, overflow-menu
callbacks and delivery to the screen. Deferred focus validates the same visit.
An inactive viewport has no actionable generation.

The first retained slot is for settled text windows. Scrollback or transient
input state, active decorations, media, tool workers, native action buttons
and rendered Markdown links use the ordinary projection path. These native
controls can turn an old click into a fresh message before it bubbles. The
surface owns the slot and serializes projection. It
validates the current store, session incarnation, binding and attachment
after awaited work, before exposing the viewport. Retention is local to the
mounted surface and has a 128-message and 128-row bound. The existing transcript
window sizes and rendering rules continue to determine what is shown.

Use the same Terminal journeys and 50 ms loop / 100 ms busy-paint limits.
A warm hit must reuse native widgets and current content. Exercise cold
opens, close, replacement, resizing, delayed input, cancellation and
ordinary quit.
Keep ADR-120 navigation, ADR-198 collector policy, ADR-210 region boundaries
and ADR-225 native resource ownership in force.

Evidence: [Terminal controls and mounted verification](../../Docs/QA/task-34661/stability-candidates-2026-11-11.md#active-transcript-retention-controls-on-2026-11-13).

Alternatives include changing mount batch size within the shared viewport,
which still rebuilds disjoint histories, and a virtualized row renderer,
which would need to reproduce native Markdown, focus, selection, scrollback
and media contracts. Retaining every open tab gives the widget graph the
lifetime of the open-tab list and has no independent memory bound.

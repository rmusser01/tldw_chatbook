# ADR-115: Demand-mount heavy Personas center views

Status: Accepted
Date: 2026-09-03
Related Task: TASK-31215
Extends: ADR-007

## Decision

The Personas destination will compose four stable, lightweight center-view slots
but will not construct or mount their heavy bodies during initial load:

- character editor;
- persona profile editor;
- dictionary detail/editor; and
- lore detail/editor.

`PersonasScreen` owns a single asynchronous first-use boundary that maps each
heavy view identity to its slot and widget factory. The boundary mounts at most
one instance of a requested view, coalesces concurrent requests for that view,
and keeps the mounted widget cached for the remainder of the screen lifetime.
A failed mount does not mark the view ready and a later user action may retry it.

The initial Characters card, library rail, preview, inspector, attachments, and
conversation surfaces remain eager because they form the usable default route.
Switching modes alone does not mount an editor or detail body. A body is requested
only by a workflow that needs it: character/persona create or edit, or
dictionary/lore selection, create, edit, or restore.

Existing public widget IDs remain on the heavy widget roots. The new slot IDs are
internal layout anchors. Existing workflow code may continue querying a heavy
widget only after its owning async entry point has crossed the first-use boundary.
The mount boundary applies view-specific retained state, such as the current
persona runtime source and character TTS presentation, before returning the widget
to the caller.

Restore and handoff inputs remain screen-owned while a view is absent. The owning
workflow first validates that its requested body is mounted and still belongs to
the current screen generation, then applies selection/editor state. Unmounting
invalidates the generation so late mount or hydration continuations cannot mutate
a detached or replacement screen.

## Context

TASK-2725 established that Roleplay/Personas latency is dominated by Textual
widget mounting and CSS application, not database I/O. It moved four heavy views
past first paint, reducing the visible navigation delay, but `_load_after_mount`
still mounts all four before restoring or selecting data. A current mounted census
found 574 descendants at settle and 458 effectively hidden. The four deferred
roots account for 350 hidden descendants: 140 in the character editor, 83 in the
persona editor, 67 in dictionary detail, and 60 in lore detail.

The same first-use strategy removed the apparent Lab > Models freeze in
TASK-31002. Personas has additional selection, restore, runtime-source, and
Console-handoff state, so its lifecycle must be explicit rather than implemented
as an uncoordinated set of missing-widget catches.

ADR-007 makes the top-level Personas route the durable owner of these workflows.
This decision stays within that destination while defining when its expensive
presentation bodies exist.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Keep mounting all four bodies immediately after first paint | It preserves every historical query assumption but still blocks the post-paint load worker on hundreds of hidden descendants and reproduces the freeze-like arrival. |
| Schedule the existing batch mount after an idle delay | It merely moves the same mount storm later, can interrupt the user's first interaction, and still constructs views the user may never open. |
| Add defensive `query_one` handling at every current call site | More than 80 queries would each encode lifecycle policy, making restore, retry, and teardown ordering inconsistent and difficult to review. |
| Recompose and discard a body whenever its mode changes | It lowers steady widget count but destroys unsaved editor state, focus, results, and worker presentation; repeated switching also repays the mount cost. |
| Use `textual.lazy.Lazy` without a screen-owned boundary | It does not define when restore and action handlers may safely query descendants and does not provide the required retry and generation-fencing behavior. |

## Consequences

- Initial Personas navigation pays only for the usable default route rather than
  every authoring surface.
- First use of a heavy workflow pays that view's one-time mount cost; later uses
  preserve widget identity and unsaved in-screen state.
- Async workflow entry points become the lifecycle boundary. Direct calls that
  bypass them are unsupported and should be caught by focused tests.
- Stable slots add four lightweight widgets and one internal mapping, but avoid
  layout-anchor churn and preserve the established center-view order.
- A mount failure is recoverable and isolated to the requested workflow; it does
  not crash the application or consume readiness state.
- No storage schema, provider/runtime contract, or user-visible information
  architecture changes.

## Verification Contract

The implementation must prove:

1. initial load settles with none of the four heavy bodies mounted;
2. first use mounts only the requested body and repeated/concurrent requests reuse
   the same widget;
3. all character, persona, dictionary, and lore entry paths cross the boundary
   before querying their heavy view;
4. restore/deep-link state is applied only after the required body is ready;
5. transient failure can retry and teardown invalidates late work; and
6. a production-CSS heartbeat records no event-loop stall above the repository's
   250 ms threshold while opening Personas.

## Links

- [TASK-31215](../tasks/task-31215%20-%20Personas-mount-heavy-center-views-on-first-use.md)
- [TASK-2725](../tasks/task-2725%20-%20Roleplay%20screen%20switch%20takes%202s%20where%20every%20other%20screen%20takes%20under%201s.md)
- [TASK-31002](../tasks/task-31002%20-%20Models-mount-only-the-active-provider-view.md)
- [Design specification](../../Docs/superpowers/specs/2026-09-03-personas-demand-mounted-center-views-design.md)

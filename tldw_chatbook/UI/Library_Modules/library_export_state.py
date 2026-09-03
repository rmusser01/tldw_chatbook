"""``LibraryExportState`` -- the Export subsystem's own fields.

State PR of the Export extraction series (wave-2 task 2 of
``.superpowers/sdd/2026-09-02-library-decomposition-wave2-cold-trio``; recipe:
``backlog/docs/library-decomposition-recipe.md``; conversations series is
the worked example this mirrors). Every field here was moved verbatim out
of ``LibraryScreen`` in ``tldw_chatbook/UI/Screens/library_screen.py`` --
same default, same type. ``library_screen.py`` originally kept every
original ``_library_export_<field>`` attribute name alive as a generated
getter/setter ``@property`` shim pointing at ``self._export_state.<field>``
(a sentinel-wrapped block right after the ``LibraryScreen`` class body).
The export cleanup PR (task 4) deleted that screen-side shim block
entirely once the subsystem's methods had all moved to
``LibraryExportController`` (task 3) and the screen's own remaining
references were retargeted to call through that controller instead. The
controller that took over the subsystem's methods carries its OWN
generated shim block in its place -- reading/writing through an injected
``export_state_accessor`` rather than a direct ``self._export_state``
attribute, since the controller does not hold the state object itself.
See the controller module's own shim-block comment for why that block is
permanent (not a cleanup-PR deletion target, unlike the one this class's
own state PR originally shared).

Every field uses the SAME ``_library_export_`` prefix -- unlike
Conversations, the ownership analysis found no field needing a plural
variant, so there is no ``EXPORT_PLURAL_STATE_FIELDS`` constant here (see
the task-2 report's ownership table for the per-field census that
established this).

One field, ``origin_row_id``, was never assigned in ``LibraryScreen.
__init__`` at all -- it was a plain class-level annotated default
(``_library_export_origin_row_id: str = ""``, task-4023 AC#7's "which
canvas's Export… action opened the Export canvas" flag), read by two
shell/plumbing methods (``_library_route_shortcuts_for_current_state``, the
footer/F1 shortcut projector; ``_select_library_rail_row_after_source_
admission``, the rail-switch clear) and written/cleared only by
Export-owned methods (``_open_library_export_canvas``,
``action_library_export_back``). Per the recipe §2 ownership script,
shell/plumbing-only non-subsystem consumers still move with the
subsystem; the class-level attribute is removed and this dataclass's own
default (``""``) supplies the identical value through the generated
property shim.

One field, ``form``, has a genuinely computed (not static-literal) default
in the original code -- ``self._default_library_export_form()``, a
``@staticmethod`` stamping today's export-name default. Per the recipe's
"computed defaults become constructor arguments" rule, ``__init__`` still
calls that method (at the exact position the removed ``self._library_
export_form = ...`` line occupied) and passes the result into the
``LibraryExportState(...)`` constructor call; this dataclass's own
``form`` default (an empty dict, via ``default_factory``) is therefore a
momentary placeholder, identical in spirit to the conversations state
object's three entangled-field placeholders, overwritten before anything
else in ``__init__`` reads it.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from ...Library.library_export_scope import ExportScope


@dataclass
class LibraryExportState:
    """Every field the Export subsystem exclusively owns."""

    # Export canvas state (F4 Task 2). ``_library_export_counts`` is
    # ``None`` until the counts worker lands a result for the current
    # scope (drives ``LibraryExportFormState.counts_loading`` --
    # deliberately not a separate boolean flag, so "loading" and "no
    # result yet" can never drift apart). ``_library_export_form`` is
    # a plain dict (not a dataclass, unlike the ingest form echo)
    # since Task 3 reads specific keys off it directly per the F4
    # plan's screen-attrs contract.
    scope: ExportScope = ExportScope(kind="everything")
    counts: dict[str, int] | None = None
    # Monotonic ownership for the counts request, separate from the export
    # execution token below.  Scope/route/generation can all repeat after a
    # leave -> return ABA visit, so none of them can identify the newest
    # counts worker on its own.
    counts_request_id: int = 0

    # Placeholder default only -- see module docstring: the original
    # `__init__` line's `self._default_library_export_form()` call still
    # runs, at the position of this field's removed assignment, and is
    # passed into the state constructor call directly.
    form: dict[str, Any] = field(default_factory=dict)

    # task-14902: True while the export quality chooser's direct-pick
    # strip renders below its (still-visible) opener button.
    quality_choices_visible: bool = False
    running: bool = False
    error: str = ""
    # Task 3: the running export's quiet status line ("Exporting…
    # (N items)"); no backing field existed after Task 2 (its report
    # flagged this as the natural next attr). Cleared alongside
    # ``_library_export_error`` on every canvas reset and on run
    # completion.
    status: str = ""
    # Task 3 review fix: a monotonic token identifying the CURRENT
    # export attempt. Bumped both when a new export starts
    # (``handle_library_export_submit``) and whenever the export
    # canvas's transient state is reset out from under an in-flight
    # run (``_reset_library_export_transient_state`` -- reachable via
    # any rail-row switch or "Export…" section action while a worker
    # is still executing on its own OS thread, which cannot be
    # preempted mid-``asyncio.run`` by ``Worker.cancel()``). The
    # worker captures the token at dispatch time and the completion
    # handlers compare it back against the live value before mutating
    # ``_library_export_running``/``_library_export_error``/
    # ``_library_export_status`` or touching the DOM -- an orphaned
    # run's late completion still notifies (the export genuinely
    # happened) but can never stomp whatever the user is now looking
    # at, mirroring ``_apply_library_export_counts``'s scope-mismatch
    # staleness guard for the sibling counts worker.
    run_id: int = 0
    # Task 4: the current run's cancellation signal. Created fresh at
    # every submit (``handle_library_export_submit``); the worker reads
    # ``event.is_set`` as the service's ``cancel_check``. Nothing sets
    # it yet in this task -- the Cancel button and navigate-away wiring
    # land in Task 5.
    cancel_event: threading.Event | None = None
    # task-2858 AC#3 (LIB-12): the last successful export's destination
    # + completion timestamp, for the durable "Last export: ..."
    # receipt. Deliberately NOT touched by
    # ``_reset_library_export_transient_state`` -- every OTHER export
    # field resets on every canvas entry (a fresh form each visit is
    # correct), but the receipt must survive leaving and re-entering
    # the canvas within the session. Also round-tripped through
    # ``save_state``/``restore_state`` so it survives a full navigate-
    # away-and-back to Library too (the "persist further" half of the
    # AC, via that already-existing seam).
    last_path: str = ""
    last_at: float | None = None

    #: task-4023 AC#7: which canvas's "Export…" action opened the Export
    #: canvas ("" = entered from the rail/deep link). Escape returns
    #: there; a plain rail switch clears it. Class-level default for the
    #: same restored-session reason as the other class-level route defaults.
    #
    # (The "class-level default" sentence above is carried verbatim from
    # its original site, ``LibraryScreen``'s own class body -- see module
    # docstring: this field alone was never a `__init__` assignment, only
    # that class-level annotation, so it has no matching entry in
    # `LibraryScreen.__init__`'s constructor call; this dataclass's own
    # default below supplies the same value the class-level attribute
    # used to, through the generated property shim.)
    origin_row_id: str = ""

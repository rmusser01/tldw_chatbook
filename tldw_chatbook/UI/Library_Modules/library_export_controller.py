"""Library Export canvas controller.

Controller PR of the Export extraction series (wave-2 task 3 of
``.superpowers/sdd/2026-09-02-library-decomposition-wave2-cold-trio``; export
series 2/3; recipe: ``backlog/docs/library-decomposition-recipe.md``;
conversations series -- ``library_conversations_controller.py`` -- is the
worked example this mirrors). Owns the Export canvas's own execution
mechanics that survive this task's two exclusion rounds (see below): scope
open/reset, the memory-DB server-mode/DB-handle helpers, the static
payload/service/message builders, the worker-completion marshal chain's
non-``@work`` links, the Escape-back action, and the form fields (name/
description/quality/destination). ``LibraryScreen`` keeps one-line
delegators under every one of these original names.

**Cluster derivation, round 1 -- ownership.** A mechanical ``ast`` scan of
``LibraryScreen`` for method names containing ``"export"`` (case-
insensitive) finds 51 methods (matches wave-2 task 2's own census exactly).
Naively moving all 51 would be wrong: the brief's "verify each candidate's
callers" instruction is the reason this docstring exists. Reading every one
of the 51 bodies shows the name match is frequently coincidental -- many are
a DIFFERENT subsystem's own "Export..." button handler (its guard reads that
subsystem's state, not Export's), which merely calls the shared
``_open_library_export_canvas`` opener. Those are that subsystem's cluster
material, not this one's, exactly the same "belongs to another subsystem"
exclusion the recipe's field-ownership census (§2) already codifies for
fields -- here applied to methods by reading what state each body actually
touches, not by name substring. **18 of the 51** stay exactly where they
already are, untouched by this PR:

- Notes (unmoved, real bodies -- Notes has no controller yet):
  ``_export_library_note``, ``_write_library_note_export_file``,
  ``handle_library_note_export_markdown``, ``handle_library_note_export_text``,
  ``handle_library_notes_export`` (``@on``), ``handle_library_notes_export_selected``
  (``@on``).
- Prompts (unmoved, real bodies -- Prompts has no controller yet):
  ``handle_library_prompt_export`` (``@on``), ``_export_library_prompt``,
  ``_write_library_prompt_export_file``, ``handle_library_prompts_export``
  (``@on``), ``handle_library_prompts_export_selected`` (``@on``).
- Media (unmoved, real bodies -- Media has no controller yet):
  ``handle_library_media_export`` (``@on``), ``handle_library_media_export_selected``
  (``@on``).
- Conversations (**already** moved or excluded by that series, tasks 8/9 --
  export makes no claim on either): ``handle_library_conversations_export``
  is already a one-line delegator to ``LibraryConversationsController``
  (task 8); ``handle_library_conversations_export_selected`` is one of task
  8's own 5 unbound-fake-self exclusions and stays a real, full-bodied
  ``LibraryScreen`` method for the SAME reason recorded in that controller's
  module docstring -- this task does not re-litigate either decision.
- Collections (unmoved, real bodies -- an entirely different mechanism, the
  legacy JSON recovery export, not the chatbook-zip Export canvas):
  ``choose_library_collection_legacy_recovery_export`` (``@on``),
  ``_export_library_collection_legacy_recovery``.
- Search/RAG (unmoved, real body -- does not touch any ``_library_export_*``
  state at all; it navigates to the Ingest Media rail row, a retired-feature
  stub): ``open_import_export_from_library_rag`` (``@on``).

**Round 2 -- a framework-decorator hazard.** Of the 33 remaining, 2 are
genuinely Export-owned but CANNOT move as methods onto this controller:
``_run_library_export_counts_worker`` and ``_run_library_export_worker``,
both decorated ``@work(thread=True, exclusive=True, group=...)``. Textual's
``work()`` decorator wraps the method in a closure that runs ``self =
args[0]; assert isinstance(self, DOMNode)`` at CALL time (see
``textual/_work_decorator.py``) -- a plain controller object is not a
``DOMNode``, so calling either through ``self.<name>(...)`` on THIS
controller would raise ``AssertionError`` synchronously, every time. Both
stay on ``LibraryScreen``, UNMOVED, decorator and body untouched (the screen
genuinely IS a ``DOMNode``, so nothing there needs to change). Call this
bypass shape **"framework-decorator self-type assertion"** -- a shape
alongside the recipe's §3 class-level-monkeypatch and §11's three (unbound
fake-self, instance-attribute monkeypatch, dynamic getattr/setattr
dispatch): the hazard here is not a test at all, it is the decorator's own
runtime contract, and it would have fired on EVERY call, not just under
test.

**Round 3 -- unbound fake-self, at a scale the conversations exemplar did
not see.** Of the 31 remaining after rounds 1-2, this task's own
verification battery (not static analysis) found **9 more** reached by
``LibraryScreen.<name>(fake, ...)`` calls where ``fake`` is a hand-built
``SimpleNamespace``/``Mock`` lacking ``_export_controller`` -- the SAME
bypass shape the recipe's §11 first catalogued for the conversations
exemplar (5 methods, one test file), but here spread across SIX different
test files (``Tests/UI/test_library_export_cancel.py``,
``test_library_export_progress_apply.py``, ``test_library_export_receipt.py``,
``test_library_shell.py``, and ``Tests/Library/test_library_export_
execution.py`` -- note the directory: this task's own §7 sweep evidence had
to be widened beyond ``Tests/UI`` to find the last four, since the recipe's
canonical sweep command scopes to ``Tests/UI -k "library"`` and
``Tests/Library/`` sits outside it entirely). All 9 stay real,full-bodied,
UNMOVED, on ``LibraryScreen``, with an inline comment at each site naming
its bypassing test:

- ``_apply_library_export_cancelled``, ``handle_library_export_cancel``
  (``test_library_export_cancel.py``, three unbound calls across two names).
- ``_apply_library_export_progress`` (``test_library_export_progress_
  apply.py``, three unbound calls).
- ``_apply_library_export_counts``, ``_build_library_export_state``
  (``test_library_export_receipt.py``) -- the second is reached only
  INDIRECTLY: the test does ``fake._build_library_export_state = (lambda:
  LibraryScreen._build_library_export_state(fake))`` so that
  ``_apply_library_export_counts``'s (also excluded) internal
  ``self._build_library_export_state()`` call, invoked with ``self=fake``,
  re-enters the REAL implementation instead of a delegator that would reach
  for a nonexistent ``fake._export_controller``. A moved body's own
  internal call chain does not shield it from this shape -- the entry point
  being unbound is what matters, not how deep the reference sits.
- ``_update_library_export_canvas_after_run`` (``test_library_export_
  receipt.py``, one direct unbound call; ALSO reached as a sibling
  dependency of ``_apply_library_export_failure``, which stays MOVED --
  see its own dependency list below).
- ``_start_library_export_counts_worker`` (``Tests/Library/test_library_
  export_execution.py::test_prompt_memory_database_forces_inline_count_
  resolution``).
- ``_start_library_export_worker`` (``Tests/Library/test_library_export_
  execution.py``, two tests).
- ``_apply_library_export_success`` (``test_library_shell.py::
  test_library_export_registry_failure_warns_it_wont_appear_in_artifacts``,
  a FOURTH, new-to-this-task bypass shape: ``screen = Mock()``, not a
  ``SimpleNamespace``. A ``Mock`` auto-creates any attribute access --
  including ``screen._export_controller`` -- as ANOTHER ``Mock``, so a
  delegator here does not raise; it silently "succeeds" while never running
  the real body, which is what actually breaks the test's assertions on
  ``screen._library_export_running``/etc. Call this shape **"silent Mock
  auto-attribution"** -- distinct from the ``AttributeError``-raising
  ``SimpleNamespace`` shape above precisely because it fails quietly rather
  than loudly; only running the actual test surfaced it, not a grep for
  ``AttributeError``.)

None of these 9 were caught by the free-name-resolution script, the
byte-for-byte body diff, or a static read of the moved bodies -- every one
was found only by running this task's own verification battery
(``-k "export and library"`` under ``Tests/UI``, THEN widened to
``Tests/Library/`` once the SimpleNamespace-unbound idiom's prevalence
there became apparent) and reading each failure's traceback. All 9 were
independently confirmed to PASS on a pristine ``git stash -u`` baseline
(i.e. genuine regressions this task's move introduced, not pre-existing
reds) before being excluded here.

**Net: 51 candidates -> 18 (other-subsystem) + 2 (framework-decorator) + 9
(unbound-fake-self) excluded = 22 move onto this controller.**

**Dynamic-dispatch census (recipe §3's "4th-bypass shape" forward note from
task 2's report), confirmed BEFORE moving anything:**
``_close_open_library_choice_strip`` (``library_screen.py``, shell/plumbing,
shared across media/prompts/skills/export) does
``setattr(self, visibility_attr, False)`` where ``visibility_attr`` can be
the literal string ``"_library_export_quality_choices_visible"`` -- looked
up from a dict keyed by canvas kind. This is NOT a hazard for this task:
``_close_open_library_choice_strip`` itself is shell-owned (calls across
FOUR subsystems, none of which own it) and stays on ``LibraryScreen``,
untouched by this move; the field it dynamically sets already has its
screen-facing property shim from task 2 (``LibraryScreen._library_export_
quality_choices_visible``, generated, unaffected by this PR). A grep for
every other ``getattr(self,``/``getattr(screen,``/``setattr(self,``/
``setattr(screen,`` call using an f-string or dict-literal argument
anywhere in ``library_screen.py`` and ``canvas_sync.py`` found exactly two
more dynamic-dispatch sites (``_library_rail_preferences()``'s
``f"{section_id}_open"`` lookup and ``_replace_library_reader_preference``'s
7-destination reader-preferences dict); neither's destination set includes
any Export name (Export has no rail-preferences/reader-pane concept), so
neither is a hazard for this cluster either. ``canvas_sync.py``'s own
``_sync_library_canvas`` dispatcher DOES have a literal ``kind == "export"``
branch (calling ``screen._build_library_export_state()``), but a repo-wide
grep for ``_sync_library_canvas(..., "export"...)`` call sites found NONE --
that branch is unreached by any caller, export-cluster or otherwise, so it
introduces no live coupling for this move to break.

**Byte-for-byte canon** (moved bodies never edited -- every name they
reference that is not this controller's own state is rebound under the SAME
name, per the two binding kinds; see ``ConsoleDictationController.__init__``,
``tldw_chatbook/UI/Console_Modules/dictation.py``, and
``LibraryConversationsController.__init__`` for the sibling worked
examples):

1. **Framework services** (``app_instance``, ``app``, ``call_after_refresh``,
   ``is_mounted``, ``query_one``, ``refresh``) are live-read from the screen
   via ``@property`` on every access -- never snapshotted.
2. **Everything else** the cluster depends on that is not its own state is a
   NAMED constructor dependency -- EXCEPT ``_safe_text``, bound the same
   class-level way the conversations controller's own ``_safe_text`` is
   (see that controller's module docstring for the full incident this shape
   resolves). This cluster's dependencies: (a) general Library-wide shell
   helpers the moved bodies call with explicit arguments
   (``_apply_library_open_item_surface``, ``_flush_library_note_save``,
   ``_set_library_destination_with_conversation_fence`` -- the SAME shared
   rail/destination-switch helper the conversations controller's own
   docstring documents as staying on the screen for every subsystem alike
   -- ``_sync_library_emergency_guard_presentation``, ``_close_open_library_
   choice_strip``, ``_focus_library_hub_entry``, ``_select_library_rail_row``,
   ``_focus_library_choice_strip_active``, ``_focus_library_control``); (b)
   shared shell state this cluster only READS (``_library_selected_row_id``
   -- the recipe's own canonical >=2-subsystems field, 226 refs, read-only
   in this cluster: no direct write site exists inside any of the 22 moved
   bodies, confirmed by an AST Store-context check before committing to a
   read-only accessor here -- and ``_library_prompts_mutation_in_flight``, a
   DIFFERENT subsystem's own state that ``_open_library_export_canvas``
   reads as a guard); and (c) the SEVEN screen-resident siblings excluded
   in round 3 above (``build_library_export_state``,
   ``start_library_export_counts_worker``, ``start_library_export_worker``,
   ``apply_library_export_success``, ``apply_library_export_cancelled``,
   ``update_library_export_canvas_after_run``, ``handle_library_export_
   cancel``), bound as named callables exactly like (a) despite their own
   bodies staying screen-resident, for the "unbound fake-self"/"silent Mock
   auto-attribution" reasons documented above. (Round 2's two ``@work``
   siblings need NO binding here at all: their only callers,
   ``_start_library_export_counts_worker``/``_start_library_export_worker``,
   are THEMSELVES excluded in round 3, so this controller never reaches for
   either name.)

This subsystem's OWN state (every ``_library_export_<field>`` name the moved
bodies reference) is exposed through generated properties reading
``self._export_state_accessor().<field>`` -- the same generator shape task 2
installed on ``LibraryScreen`` and the conversations controller installed on
itself, applied here. Export uses a single ``_library_export_`` prefix for
every field (task 2's report: no field needed a plural variant), so there is
no per-field prefix-selection logic in the generator loop, unlike
Conversations' plural/singular split.
"""
from __future__ import annotations

import asyncio
import dataclasses
import threading
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any, TYPE_CHECKING

from loguru import logger
from rich.markup import escape as escape_markup
from textual import on
from textual.css.query import NoMatches, QueryError
from textual.widgets import Button, Input, Static

from ...Chatbooks.chatbook_models import ContentType
from ...Library.library_export_scope import ExportScope, count_export_scope
from ...Library.library_export_state import (
    DEFAULT_MEDIA_QUALITY,
    MEDIA_QUALITY_OPTIONS,
    default_export_name,
    normalize_export_destination,
)
from ...Library.library_notes_session import NoteFlushOutcomeKind
from ...Library.library_shell_state import (
    LIBRARY_EXPORT_SERVER_DISABLED_TOOLTIP,
    LIBRARY_ROW_INGEST_EXPORT,
)
from ...Third_Party.textual_fspicker import FileSave
from ...Utils.path_validation import validate_path_simple
from ...Widgets.Library import LibraryExportCanvas
from .library_export_state import LibraryExportState

if TYPE_CHECKING:
    from ..Screens.library_screen import LibraryScreen


class LibraryExportController:
    """Owns the Export canvas's surviving cluster: scope open/reset,
    server-mode/DB-handle helpers, static payload/service/message builders,
    the non-``@work`` marshal links, Escape-back, and the form fields.

    Holds no state of its own beyond what it reads and writes through
    ``LibraryExportState`` (via the injected accessor) or the shared shell
    attributes bound below. ``LibraryScreen`` constructs exactly one of
    these, in ``__init__`` right after ``self._conversations_controller``,
    and keeps one-line delegators for every original name this cluster
    moved (22 of the naive 51 "export"-named candidates -- see the module
    docstring for where the other 29 went and why).
    """

    def __init__(
        self,
        screen: "LibraryScreen",
        *,
        export_state_accessor: Callable[[], LibraryExportState],
        # -- general Library-wide shell helpers, not moved (shared with
        # other subsystems; see module docstring group (a)).
        apply_open_item_surface: Callable[..., Awaitable[Any]],
        flush_note_save: Callable[[], Awaitable[Any]],
        set_library_destination_with_conversation_fence: Callable[[str], None],
        sync_library_emergency_guard_presentation: Callable[[], None],
        close_open_library_choice_strip: Callable[[], bool],
        focus_library_hub_entry: Callable[[], None],
        select_library_rail_row: Callable[..., Awaitable[Any]],
        focus_library_choice_strip_active: Callable[..., None],
        focus_library_control: Callable[..., None],
        # -- shared shell state this cluster only reads (see module
        # docstring group (b)).
        library_selected_row_id_accessor: Callable[[], str],
        library_prompts_mutation_in_flight_accessor: Callable[[], bool],
        # -- stays on LibraryScreen (see module docstring's round-3 "unbound
        # fake-self"/"silent Mock auto-attribution" exclusion notes) but a
        # moved body calls each internally, so each is bound like any other
        # general dependency (see module docstring group (c)).
        build_library_export_state: Callable[[], Any],
        start_library_export_counts_worker: Callable[[], None],
        start_library_export_worker: Callable[..., None],
        apply_library_export_success: Callable[..., None],
        apply_library_export_cancelled: Callable[[int], None],
        update_library_export_canvas_after_run: Callable[[], None],
        handle_library_export_cancel: Callable[[Any], None],
    ) -> None:
        """Build the controller and bind everything its moved bodies need.

        Every one of the 22 method bodies below is a byte-for-byte copy of
        the pre-extraction ``LibraryScreen`` method: no internal line was
        edited to retarget a call or an attribute. That is possible because
        this constructor binds every name those bodies reference that is
        not this controller's own state, under the SAME name the original
        method used. See the module docstring for the binding kinds this
        follows and for why 29 of the naive 51 "export"-named candidates
        are NOT here at all.

        Args:
            screen: The Library screen. Used ONLY for the six framework
                services below (``app_instance``, ``app``,
                ``call_after_refresh``, ``is_mounted``, ``query_one``,
                ``refresh``) -- this cluster owns no DOM of its own.
            export_state_accessor: Returns the live ``LibraryExportState``
                (``LibraryScreen._export_state``, task 2). Backs every
                generated ``_library_export_<field>`` property below.
            apply_open_item_surface: ``LibraryScreen.
                _apply_library_open_item_surface`` -- the shared per-click
                canvas-swap helper (rail selection + canvas-child swap,
                never a whole-screen rebuild) every section entry point
                uses; ``_open_library_export_canvas`` uses it to mount the
                fresh ``LibraryExportCanvas``.
            flush_note_save: ``LibraryScreen._flush_library_note_save`` --
                the shared dirty-note-flush guard called before ANY canvas
                switch (rail row or section action alike), not Notes-
                exclusive despite its name; ``_open_library_export_canvas``
                calls it before admitting the switch into Export.
            set_library_destination_with_conversation_fence: ``LibraryScreen.
                _set_library_destination_with_conversation_fence`` -- the
                SAME shared rail/destination-switch helper the
                conversations controller's own docstring documents as
                staying on the screen for every subsystem alike (not just
                Conversations, despite its name) -- moves
                ``_library_selected_row_id`` and invalidates whichever
                subsystem is being navigated away from.
            sync_library_emergency_guard_presentation: ``LibraryScreen.
                _sync_library_emergency_guard_presentation`` -- the shared
                emergency-return visibility sync; ``_apply_library_export_
                failure``/``handle_library_export_submit`` call it after
                their own state changes.
            close_open_library_choice_strip: ``LibraryScreen.
                _close_open_library_choice_strip`` -- the shared converged
                choice-strip closer (media type / prompts sort / skills
                sort / export quality); ``action_library_export_back``
                lets an open quality strip consume Escape first. See the
                module docstring's dynamic-dispatch census for why this
                staying unmoved is required, not just convenient.
            focus_library_hub_entry: ``LibraryScreen._focus_library_hub_entry``
                -- the shared hub-landing focus restorer, scheduled via
                ``call_after_refresh`` (passed bare, never called directly,
                by ``action_library_export_back``).
            select_library_rail_row: ``LibraryScreen._select_library_rail_row``
                -- the shared rail-row switch every Escape-back action uses
                to return to its origin (or the hub).
            focus_library_choice_strip_active: ``LibraryScreen.
                _focus_library_choice_strip_active`` -- the shared
                choice-strip-open focus mover; ``handle_library_export_
                quality`` uses it to land focus on the active quality
                choice.
            focus_library_control: ``LibraryScreen._focus_library_control``
                -- the shared plain-control focus mover; the quality
                chooser's close paths and the quality-choice pick both use
                it to return focus to the opener button.
            library_selected_row_id_accessor: Reads ``LibraryScreen.
                _library_selected_row_id`` -- the recipe's own canonical
                >=2-subsystems shared field (226 refs). Read-only in this
                cluster: confirmed by an AST Store-context check that no
                moved body writes it directly (writes happen only through
                ``set_library_destination_with_conversation_fence``/
                ``select_library_rail_row`` above), so no setter is bound.
            library_prompts_mutation_in_flight_accessor: Reads
                ``LibraryScreen._prompts_state.mutation_in_flight`` (the flat
                shim name this line used to give died with the prompts
                cleanup, wave-6 task 3) -- a DIFFERENT subsystem's (Prompts)
                own state; ``_open_library_export_canvas`` reads it as a guard
                blocking EVERY canvas switch, not just Export's.
            build_library_export_state: ``LibraryScreen.
                _build_library_export_state`` -- stays on the screen,
                UNMOVED (module docstring round 3: reached indirectly via
                an unbound-fake-self test wrapper). ``_open_library_export_
                canvas`` calls it internally.
            start_library_export_counts_worker: ``LibraryScreen.
                _start_library_export_counts_worker`` -- stays on the
                screen, UNMOVED (module docstring round 3: an unbound
                ``SimpleNamespace`` call in ``Tests/Library/test_library_
                export_execution.py``). ``_open_library_export_canvas``
                calls it internally.
            start_library_export_worker: ``LibraryScreen.
                _start_library_export_worker`` -- stays on the screen,
                UNMOVED (module docstring round 3: two unbound
                ``SimpleNamespace`` calls in the same file).
                ``handle_library_export_submit`` calls it (via
                ``call_after_refresh``) internally.
            apply_library_export_success: ``LibraryScreen.
                _apply_library_export_success`` -- stays on the screen,
                UNMOVED (module docstring round 3: the "silent Mock
                auto-attribution" shape). ``_marshal_library_export_
                success`` calls it internally.
            apply_library_export_cancelled: ``LibraryScreen.
                _apply_library_export_cancelled`` -- stays on the screen,
                UNMOVED (module docstring round 3: two unbound
                ``SimpleNamespace`` calls). ``_marshal_library_export_
                cancelled`` calls it internally.
            update_library_export_canvas_after_run: ``LibraryScreen.
                _update_library_export_canvas_after_run`` -- stays on the
                screen, UNMOVED (module docstring round 3: an unbound
                ``SimpleNamespace`` call). ``_apply_library_export_failure``
                calls it internally.
            handle_library_export_cancel: ``LibraryScreen.
                handle_library_export_cancel`` -- stays on the screen,
                UNMOVED (module docstring round 3: an unbound
                ``SimpleNamespace`` call). ``action_library_export_back``
                calls it internally when a run is in flight.
        """
        self._screen = screen
        self._export_state_accessor = export_state_accessor
        self._apply_open_item_surface_fn = apply_open_item_surface
        self._flush_note_save_fn = flush_note_save
        self._set_library_destination_with_conversation_fence_fn = (
            set_library_destination_with_conversation_fence
        )
        self._sync_library_emergency_guard_presentation_fn = (
            sync_library_emergency_guard_presentation
        )
        self._close_open_library_choice_strip_fn = close_open_library_choice_strip
        self._focus_library_hub_entry_fn = focus_library_hub_entry
        self._select_library_rail_row_fn = select_library_rail_row
        self._focus_library_choice_strip_active_fn = focus_library_choice_strip_active
        self._focus_library_control_fn = focus_library_control
        self._library_selected_row_id_accessor = library_selected_row_id_accessor
        self._library_prompts_mutation_in_flight_accessor = (
            library_prompts_mutation_in_flight_accessor
        )
        self._build_library_export_state_fn = build_library_export_state
        self._start_library_export_counts_worker_fn = start_library_export_counts_worker
        self._start_library_export_worker_fn = start_library_export_worker
        self._apply_library_export_success_fn = apply_library_export_success
        self._apply_library_export_cancelled_fn = apply_library_export_cancelled
        self._update_library_export_canvas_after_run_fn = (
            update_library_export_canvas_after_run
        )
        self._handle_library_export_cancel_fn = handle_library_export_cancel

    # -- framework services: live-read properties, never snapshotted -----

    @property
    def app_instance(self) -> Any:
        """This project's screen-level analogue of Textual's own ``self.app``,
        live-read from the screen. See ``__init__``'s docstring."""
        return self._screen.app_instance

    @property
    def app(self) -> Any:
        """``Screen.app``, live-read -- Textual's OWN app property. See
        ``__init__``'s docstring."""
        return self._screen.app

    @property
    def call_after_refresh(self) -> Any:
        """``Screen.call_after_refresh``, bound. See ``__init__``'s
        docstring."""
        return self._screen.call_after_refresh

    @property
    def is_mounted(self) -> bool:
        """``Screen.is_mounted``, live-read. See ``__init__``'s docstring."""
        return self._screen.is_mounted

    @property
    def query_one(self) -> Any:
        """``Screen.query_one``, bound. See ``__init__``'s docstring."""
        return self._screen.query_one

    @property
    def refresh(self) -> Any:
        """``Screen.refresh``, bound. See ``__init__``'s docstring."""
        return self._screen.refresh

    # -- named constructor dependencies -----------------------------------

    @property
    def _apply_library_open_item_surface(self) -> Any:
        """The injected ``apply_open_item_surface``. See ``__init__``'s
        docstring."""
        return self._apply_open_item_surface_fn

    @property
    def _flush_library_note_save(self) -> Any:
        """The injected ``flush_note_save``. See ``__init__``'s docstring."""
        return self._flush_note_save_fn

    @property
    def _set_library_destination_with_conversation_fence(self) -> Any:
        """The injected ``set_library_destination_with_conversation_fence``.
        See ``__init__``'s docstring."""
        return self._set_library_destination_with_conversation_fence_fn

    @property
    def _sync_library_emergency_guard_presentation(self) -> Any:
        """The injected ``sync_library_emergency_guard_presentation``. See
        ``__init__``'s docstring."""
        return self._sync_library_emergency_guard_presentation_fn

    @property
    def _close_open_library_choice_strip(self) -> Any:
        """The injected ``close_open_library_choice_strip``. See
        ``__init__``'s docstring."""
        return self._close_open_library_choice_strip_fn

    @property
    def _focus_library_hub_entry(self) -> Any:
        """The injected ``focus_library_hub_entry``. See ``__init__``'s
        docstring."""
        return self._focus_library_hub_entry_fn

    @property
    def _select_library_rail_row(self) -> Any:
        """The injected ``select_library_rail_row``. See ``__init__``'s
        docstring."""
        return self._select_library_rail_row_fn

    @property
    def _focus_library_choice_strip_active(self) -> Any:
        """The injected ``focus_library_choice_strip_active``. See
        ``__init__``'s docstring."""
        return self._focus_library_choice_strip_active_fn

    @property
    def _focus_library_control(self) -> Any:
        """The injected ``focus_library_control``. See ``__init__``'s
        docstring."""
        return self._focus_library_control_fn

    @property
    def _library_selected_row_id(self) -> str:
        """Calls the injected ``library_selected_row_id_accessor``.
        Read-only in this cluster (no setter -- see ``__init__``'s
        docstring)."""
        return self._library_selected_row_id_accessor()

    @property
    def _library_prompts_mutation_in_flight(self) -> bool:
        """Calls the injected ``library_prompts_mutation_in_flight_accessor``.
        Read-only in this cluster; owned by a DIFFERENT subsystem
        (Prompts). See ``__init__``'s docstring."""
        return self._library_prompts_mutation_in_flight_accessor()

    @property
    def _build_library_export_state(self) -> Any:
        """The injected ``build_library_export_state``. This name's own
        body stays on ``LibraryScreen`` (NOT moved -- see module
        docstring's round-3 exclusion notes); ``_open_library_export_
        canvas`` below still calls it via ``self.<name>(...)``. See
        ``__init__``'s docstring."""
        return self._build_library_export_state_fn

    @property
    def _start_library_export_counts_worker(self) -> Any:
        """The injected ``start_library_export_counts_worker``. This
        name's own body stays on ``LibraryScreen`` (NOT moved -- see
        module docstring's round-3 exclusion notes); ``_open_library_
        export_canvas`` below still calls it via ``self.<name>(...)``. See
        ``__init__``'s docstring."""
        return self._start_library_export_counts_worker_fn

    @property
    def _start_library_export_worker(self) -> Any:
        """The injected ``start_library_export_worker``. This name's own
        body stays on ``LibraryScreen`` (NOT moved -- see module
        docstring's round-3 exclusion notes); ``handle_library_export_
        submit`` below still calls it via ``self.<name>(...)``. See
        ``__init__``'s docstring."""
        return self._start_library_export_worker_fn

    @property
    def _apply_library_export_success(self) -> Any:
        """The injected ``apply_library_export_success``. This name's own
        body stays on ``LibraryScreen`` (NOT moved -- see module
        docstring's round-3 exclusion notes); ``_marshal_library_export_
        success`` below still calls it via ``self.<name>(...)``. See
        ``__init__``'s docstring."""
        return self._apply_library_export_success_fn

    @property
    def _apply_library_export_cancelled(self) -> Any:
        """The injected ``apply_library_export_cancelled``. This name's
        own body stays on ``LibraryScreen`` (NOT moved -- see module
        docstring's round-3 exclusion notes); ``_marshal_library_export_
        cancelled`` below still calls it via ``self.<name>(...)``. See
        ``__init__``'s docstring."""
        return self._apply_library_export_cancelled_fn

    @property
    def _update_library_export_canvas_after_run(self) -> Any:
        """The injected ``update_library_export_canvas_after_run``. This
        name's own body stays on ``LibraryScreen`` (NOT moved -- see
        module docstring's round-3 exclusion notes); ``_apply_library_
        export_failure`` below still calls it via ``self.<name>(...)``.
        See ``__init__``'s docstring."""
        return self._update_library_export_canvas_after_run_fn

    @property
    def handle_library_export_cancel(self) -> Any:
        """The injected ``handle_library_export_cancel``. This name's own
        body stays on ``LibraryScreen`` (NOT moved -- see module
        docstring's round-3 exclusion notes); ``action_library_export_
        back`` below still calls it via ``self.<name>(...)``. See
        ``__init__``'s docstring.

        Not underscore-prefixed (unlike the other injected screen-resident
        siblings above): the ORIGINAL body calls it as
        ``self.handle_library_export_cancel(None)`` (no leading underscore
        -- it is itself a public ``@on`` handler name), so this property
        must be exposed under that exact public name for the byte-for-byte
        canon to hold.
        """
        return self._handle_library_export_cancel_fn

    # `_safe_text` is deliberately NOT a property/named-constructor-
    # dependency here. It is bound as a single CLASS-level attribute from
    # `library_screen.py`, after both classes are fully defined:
    # `LibraryExportController._safe_text = staticmethod(LibraryScreen._safe_text)`.
    # See `LibraryConversationsController`'s module docstring (and its own
    # identical comment at this same spot) for the full incident this
    # binding shape resolves -- in short, a class-level assignment always
    # overwrites a same-named property, so an earlier property/constructor-
    # parameter pair would be silently dead code.

    # -- moved bodies (byte-for-byte; see module docstring) ---------------

    # ----- Export canvas -------------------------------------------------

    @staticmethod
    def _default_library_export_form() -> dict[str, Any]:
        """Build a fresh export form echo: today's stamped name, nothing else set."""
        return {
            "name": default_export_name(),
            "description": "",
            "quality": DEFAULT_MEDIA_QUALITY,
            "destination": "",
            "destination_exists": False,
        }

    def _reset_library_export_transient_state(
        self, scope: ExportScope | None = None
    ) -> None:
        """Clear the export canvas's scope/counts/form to defaults on entry.

        Called from both entry points into the export canvas -- the rail
        row's own ``_select_library_rail_row`` switch (always the default
        Everything ``scope``) and the browse-canvas "Export…" section
        actions (``_open_library_export_canvas``, their own pre-scoped
        ``ExportScope``) -- so neither a stale form from a previous Export
        visit nor a stale scope/counts pairing from a different section
        ever reappears. The name field re-stamps today's local date every
        time (mirrors the ingest form's own from-scratch reset), never
        carrying a previous visit's edited name forward.

        Also invalidates any export run still executing on its own OS
        thread (bumps ``_library_export_run_id``) -- navigating away mid-
        run resets ``running`` to ``False`` for THIS fresh visit, but the
        abandoned worker keeps running regardless (it cannot be preempted
        mid-``asyncio.run``); bumping the token here ensures that worker's
        eventual completion is recognized as stale and cannot stomp
        whatever the user is looking at by the time it lands. See
        ``_library_export_run_id``'s docstring in ``__init__``.

        Args:
            scope: The scope to open the canvas with; defaults to
                ``ExportScope(kind="everything")`` when omitted.
        """
        self._library_export_scope = scope or ExportScope(kind="everything")
        self._library_export_counts = None
        self._library_export_form = self._default_library_export_form()
        # task-14902: a fresh visit never inherits a half-open quality strip.
        self._library_export_quality_choices_visible = False
        self._library_export_running = False
        self._library_export_error = ""
        self._library_export_status = ""
        if self._library_export_cancel_event is not None:
            self._library_export_cancel_event.set()
        self._library_export_run_id += 1

    async def _open_library_export_canvas(self, scope: ExportScope) -> None:
        """Open the export canvas pre-scoped to a browse section's own filter.

        Wired to each browse canvas's "Export…" action (media/
        conversations/notes/Prompts) -- mirrors ``_select_library_rail_row``'s
        dirty-note-flush discipline for switching canvases, but only
        touches the export-specific state (the rail row's own switch
        already resets everything else on the way past); the caller's
        ``scope`` survives untouched (unlike a plain rail-row switch,
        which always resets to Everything).

        Args:
            scope: The section-specific scope to open the form with (e.g.
                ``ExportScope(kind="media", media_type=...)``).
        """
        if self._library_prompts_mutation_in_flight:
            return
        if self._library_export_is_server_mode():
            # The section "Export..." actions bypass the rail row's own
            # server-disabled gate, so re-check here (Qodo review): export
            # reads the LOCAL DBs, so running it while the Library is in
            # server runtime mode would package the wrong dataset.
            self.app_instance.notify(
                LIBRARY_EXPORT_SERVER_DISABLED_TOOLTIP, severity="warning"
            )
            return
        note_flush = await self._flush_library_note_save()
        if note_flush.kind is not NoteFlushOutcomeKind.PERMITTED:
            return
        # task-4023 AC#7: remember which canvas opened Export so Escape
        # (action_library_export_back) can return there -- "Export… from
        # within Media navigates away with no return path". Recorded
        # AFTER the flush admits the switch, BEFORE the row id moves.
        self._library_export_origin_row_id = self._library_selected_row_id
        self._set_library_destination_with_conversation_fence(LIBRARY_ROW_INGEST_EXPORT)
        self._reset_library_export_transient_state(scope)
        # task-21116: rail selection + canvas-child swap only, never a
        # whole-screen rebuild for a per-click section "Export…" action.
        await self._apply_library_open_item_surface(
            lambda: LibraryExportCanvas(
                self._build_library_export_state(),
                id="library-export-canvas",
            )
        )
        self._start_library_export_counts_worker()

    def _library_export_is_server_mode(self) -> bool:
        """True when the Library is in server runtime mode.

        Export packages LOCAL content only (it reads the local media /
        ChaChaNotes / Prompt DBs), so both the rail Export row and the section
        "Export..." actions must refuse to run in server mode.
        """
        runtime_policy = getattr(self.app_instance, "runtime_policy", None)
        runtime_state = runtime_policy.state if runtime_policy is not None else None
        active_source = str(
            getattr(runtime_state, "active_source", "local") or "local"
        ).lower()
        return active_source == "server"

    def _resolve_library_export_chachanotes_db(self) -> Any:
        """Return the ChaChaNotes DB handle for export counts.

        Mirrors ``_resolve_library_notes_sync_db``'s exact access path
        (prefer ``app_instance.chachanotes_db``, fall back to
        ``notes_service.db``) -- the same canonical DB-access path this
        screen already uses elsewhere, per the F4 brief's requirement that
        the counts worker reach the DB the same way the rest of the
        screen does.
        """
        notes_service = getattr(self.app_instance, "notes_service", None)
        return getattr(self.app_instance, "chachanotes_db", None) or getattr(
            notes_service, "db", None
        )

    @staticmethod
    def _compute_library_export_counts(
        scope: ExportScope,
        media_db: Any,
        chachanotes_db: Any,
        prompts_db: Any,
    ) -> dict[str, int]:
        """Run the full-query, uncapped counts for ``scope`` (never a rendered snapshot).

        A quiet-degrade failure (a missing DB seam, an unexpected DB
        error) reports all-zero counts rather than raising -- the export
        canvas simply shows "Nothing to export in this scope." rather
        than crashing the recompose; the failure is still logged.
        """
        try:
            return count_export_scope(scope, media_db, chachanotes_db, prompts_db)
        except Exception as exc:
            logger.warning(
                "Library export counts failed scope_kind={} category={}",
                scope.kind,
                type(exc).__name__,
            )
            return {"media": 0, "conversations": 0, "notes": 0, "prompts": 0}

    # ----- Export canvas: execution (Task 3) ------------------------------

    @on(Button.Pressed, "#library-export-submit")
    def handle_library_export_submit(self, event: Button.Pressed) -> None:
        """Validate and kick off the chatbook export worker.

        Re-validates on the UI thread (destination chosen, scope non-empty,
        not already running) rather than trusting the button's ``disabled``
        state alone. A second press while an export is already running is a
        guarded no-op here (``self._library_export_running``) -- on top of
        the button itself being disabled while running and the worker's own
        ``group="library_export"``/``exclusive=True`` single-flight, this is
        belt-and-suspenders against a stale/racing ``Pressed`` event.

        The transition INTO ``running`` is the one place this feature uses
        a full recompose rather than a targeted update (see
        ``_update_library_export_canvas_after_run``'s docstring for the
        reverse transition's targeted-update discipline): the user's last
        action was clicking this button, not typing, so nothing is
        mid-keystroke -- unlike the counts-landing case Task 2 fixed, or
        the run-completion case below, where the (long-running) wait window
        gives the user time to resume typing in the still-editable name/
        description fields. Worker dispatch runs after that refresh so an
        immediate completion always targets the newly mounted running canvas,
        never the outgoing form that the recompose is replacing.
        """
        event.stop()
        if self._library_export_running:
            return
        form = self._library_export_form
        destination = str(form.get("destination", "")).strip()
        counts = self._library_export_counts
        total = sum(counts.values()) if counts else 0
        if not destination or total <= 0:
            return
        if self._library_export_is_server_mode():
            # Defense in depth: the rail row and section actions already
            # gate on server mode, but re-check at submit in case the
            # runtime source flipped while the form was open (Qodo review).
            self.app_instance.notify(
                LIBRARY_EXPORT_SERVER_DISABLED_TOOLTIP, severity="warning"
            )
            return
        # Sanitize name/description at the UI boundary before they flow into
        # the export payload, chatbook manifest, and Artifacts registry
        # (Qodo review) -- bound length + strip unsafe content via the shared
        # input_validation helpers, mirroring the media-field path.
        name = self._safe_text(form.get("name", ""), "Chatbook", max_length=200)
        description = self._safe_text(form.get("description", ""), "", max_length=2000)
        media_quality = str(form.get("quality", DEFAULT_MEDIA_QUALITY))
        self._library_export_running = True
        self._library_export_error = ""
        self._library_export_status = f"Exporting… ({total} items)"
        self._library_export_run_id += 1
        run_id = self._library_export_run_id
        self._library_export_cancel_event = threading.Event()
        cancel_event = self._library_export_cancel_event
        self.refresh(recompose=True)
        self.call_after_refresh(self._sync_library_emergency_guard_presentation)
        self.call_after_refresh(
            self._start_library_export_worker,
            run_id=run_id,
            scope=self._library_export_scope,
            name=name,
            description=description,
            media_quality=media_quality,
            destination=destination,
            cancel_event=cancel_event,
        )

    @staticmethod
    def _build_library_export_payload(
        *,
        name: str,
        description: str,
        selections: Mapping[ContentType, list[str]],
        destination: str,
        media_quality: str,
    ) -> dict[str, Any]:
        """Build the ``local_chatbook_service.export_chatbook`` request payload.

        ``include_media`` is spec-critical (F4 plan Global Constraints):
        it MUST be ``True`` whenever ``ContentType.MEDIA`` is present in
        ``selections`` -- ``ChatbookCreator`` silently skips all media
        content otherwise, even when media ids ARE present in
        ``content_selections``. Since ``resolve_export_selections`` omits
        a ``ContentType`` key entirely when that source resolves zero ids
        (see its docstring), keying off simple membership is automatically
        correct for every scope, including an "everything" scope whose
        library happens to have no media at all.
        """
        return {
            "name": name,
            "description": description,
            "content_selections": dict(selections),
            "output_path": destination,
            "media_quality": media_quality,
            "include_media": ContentType.MEDIA in selections,
        }

    @staticmethod
    def _run_library_export_via_service(
        service: Any,
        payload: dict[str, Any],
        *,
        name: str,
        description: str,
        progress_callback=None,
        cancel_check=None,
    ) -> dict[str, Any]:
        """Execute one export through ``service``, synchronously: zip first, registry only on success.

        Runs both of ``service``'s async-signature/sync-body methods
        through ``asyncio.run`` -- they never touch the app's own event
        loop, so this is only ever safe to call from a genuine OS thread
        (never the UI thread, which already owns a running loop). Exposed
        as its own (non-``@work``) static method so tests can call it
        directly with a fake ``service`` and assert call ordering /
        the include_media invariant without booting a real thread.

        ``create_chatbook`` (the registry record) is attempted ONLY when
        ``export_chatbook`` reports ``success`` -- the F4 plan's Global
        Constraints' "zip first, registry record only on success". A
        registry-recording failure AFTER a successful zip does not flip
        the overall outcome to failure (the artifact genuinely exists on
        disk; only the bookkeeping failed) -- ``registry_recorded``
        reports that separately for callers/tests that care.

        Returns a plain dict: ``success``, ``message``, ``path``,
        ``dependency_info``, ``registry_recorded``.
        """
        try:
            export_result = asyncio.run(  # policy-exception: worker-thread loop
                service.export_chatbook(
                    payload,
                    progress_callback=progress_callback,
                    cancel_check=cancel_check,
                )
            )
        except Exception as exc:
            logger.opt(exception=True).warning("Library export service call failed.")
            return {
                "success": False,
                "message": f"Export failed: {exc}",
                "path": "",
                "dependency_info": {},
                "registry_recorded": False,
                "cancelled": False,
            }

        if not export_result.get("success"):
            return {
                "success": False,
                "message": str(export_result.get("message") or "Export failed."),
                "path": export_result.get("path") or payload.get("output_path", ""),
                "dependency_info": export_result.get("dependency_info") or {},
                "registry_recorded": False,
                "cancelled": bool(export_result.get("cancelled", False)),
            }

        output_path = export_result.get("path") or payload.get("output_path", "")
        dependency_info = export_result.get("dependency_info") or {}
        registry_recorded = False
        try:
            asyncio.run(  # policy-exception: worker-thread loop
                service.create_chatbook(
                    name=name,
                    description=description,
                    file_path=output_path,
                    tags=["library-export"],
                )
            )
            registry_recorded = True
        except Exception:
            logger.opt(exception=True).warning(
                f"Library export succeeded but registry recording failed for {output_path!r}."
            )

        return {
            "success": True,
            "message": export_result.get("message") or "",
            "path": output_path,
            "dependency_info": dependency_info,
            "registry_recorded": registry_recorded,
            "cancelled": False,
        }

    def _marshal_library_export_success(
        self,
        run_id: int,
        path: str,
        dependency_info: Any,
        registry_recorded: bool,
        message: str = "",
    ) -> None:
        """Marshal a successful run onto the UI thread (called from the worker)."""
        try:
            self.app.call_from_thread(
                self._apply_library_export_success,
                run_id,
                path,
                dependency_info,
                registry_recorded,
                message,
            )
        except Exception:
            # A shutdown/detach mid-marshal can raise RuntimeError OR
            # Textual's NoApp (which subclasses Exception, not RuntimeError)
            # -- either way the worker thread must not crash on teardown.
            pass

    def _marshal_library_export_failure(self, run_id: int, message: str) -> None:
        """Marshal a failed run onto the UI thread (called from the worker)."""
        try:
            self.app.call_from_thread(
                self._apply_library_export_failure, run_id, message
            )
        except Exception:
            # A shutdown/detach mid-marshal can raise RuntimeError OR
            # Textual's NoApp (which subclasses Exception, not RuntimeError)
            # -- either way the worker thread must not crash on teardown.
            pass

    def _marshal_library_export_cancelled(self, run_id: int) -> None:
        """Marshal a cancelled run onto the UI thread (called from the worker)."""
        try:
            self.app.call_from_thread(self._apply_library_export_cancelled, run_id)
        except Exception:
            # A shutdown/detach mid-marshal can raise RuntimeError OR
            # Textual's NoApp (which subclasses Exception, not RuntimeError)
            # -- either way the worker thread must not crash on teardown.
            pass

    @staticmethod
    def _build_library_export_success_message(
        path: Any, dependency_info: Any, creator_message: Any = ""
    ) -> str:
        """Build the success notification text.

        Three pieces, in order:

        1. The destination path (always present), ``escape_markup``'d:
           Textual notifications render Rich console markup, so a
           user-chosen path containing ``[...]`` (legal in filenames on
           any platform) would otherwise mis-render or raise in the
           markup parser.
        2. The creator's own ``outcome["message"]`` detail (task-158):
           ``ChatbookCreator.create_chatbook`` returns a message carrying
           its own counts (e.g. missing-dependency warnings) that was
           previously discarded entirely by the caller. Its redundant
           ``"Chatbook created successfully at <path>"`` prefix -- the
           path is already the primary notify line above -- is stripped
           so only the actual detail remains; an unrecognized message
           shape (e.g. a different service implementation) is kept
           verbatim rather than guessed at.
        3. The ``dependency_info.get("auto_included")`` count suffix (the
           character ids ``ChatbookCreator`` pulled in automatically as
           conversation dependencies) -- BUT only when the creator detail
           above does not already state it. ``create_chatbook`` already
           puts an ``"Auto-included N character dependencies"`` clause
           into its own message (that clause and ``auto_included`` derive
           from the same ``self.auto_included_characters`` state), so
           emitting the suffix on top of a detail that carries that clause
           would restate the identical fact twice. The suffix therefore
           only fires when the auto-included count would otherwise go
           unstated (e.g. an empty creator message, or a creator message
           whose only detail is a missing-dependency warning).
        """
        message = f"Exported bundle to {escape_markup(str(path))}"

        detail = str(creator_message or "").strip()
        known_prefix = f"Chatbook created successfully at {path}"
        if detail.startswith(known_prefix):
            detail = detail[len(known_prefix) :].strip(" .")
        if detail:
            message += f": {escape_markup(detail)}"

        auto_included = (
            dependency_info.get("auto_included")
            if isinstance(dependency_info, dict)
            else None
        )
        # De-dup: skip the suffix when the surfaced detail already states
        # the auto-included count (see point 3 above).
        if auto_included and "auto-included" not in detail.lower():
            try:
                count = len(auto_included)
            except TypeError:
                count = auto_included
            message += f" ({count} characters auto-included)"

        return message

    def _apply_library_export_failure(self, run_id: int, message: str) -> None:
        """UI-thread completion: render the escaped error, clear running, re-enable Export.

        See ``_apply_library_export_success``'s docstring for the
        ``run_id`` staleness guard -- a superseded run's failure is
        dropped silently here (no error line to render it into, since the
        canvas may now belong to a different scope/visit entirely) rather
        than notified, since surfacing a failure banner for a run the user
        has already navigated away from and possibly re-run successfully
        would be actively misleading.
        """
        if run_id != self._library_export_run_id:
            logger.info(
                f"Library export run {run_id} failed after being superseded "
                f"(current run {self._library_export_run_id}): {message}"
            )
            return
        self._library_export_running = False
        self._library_export_status = ""
        self._library_export_error = escape_markup(str(message))
        self._sync_library_emergency_guard_presentation()
        self._update_library_export_canvas_after_run()

    def _refresh_library_export_status_line(self) -> None:
        """Update only the #library-export-status-line widget (no recompose)."""
        if (
            not self.is_mounted
            or self._library_selected_row_id != LIBRARY_ROW_INGEST_EXPORT
        ):
            return
        try:
            widget = self.query_one("#library-export-status-line", Static)
            widget.update(self._library_export_status)
            widget.display = bool(self._library_export_status)
        except (NoMatches, QueryError):
            pass

    async def action_library_export_back(self) -> None:
        """Escape: leave the Export canvas (task-4023 AC#7).

        Returns to the canvas whose "Export…" action opened it (Media/
        Conversations/Notes/Prompts -- the "navigates away with no return path"
        finding), or to the hub landing when Export was entered from the
        rail. A running export keeps running; the canvas's own state
        (including the durable last-export receipt) survives exactly as a
        rail switch would leave it.

        task-14902: an open quality strip consumes the Escape first --
        cancelling a half-made pick must not eject the user from the form.
        """
        if self._library_selected_row_id != LIBRARY_ROW_INGEST_EXPORT:
            return
        if self._close_open_library_choice_strip():
            return
        if self._library_export_running:
            self.handle_library_export_cancel(None)
            return
        origin = self._library_export_origin_row_id
        self._library_export_origin_row_id = ""
        if origin:
            await self._select_library_rail_row(origin)
            return
        await self._select_library_rail_row("")
        if self.is_mounted:
            self.call_after_refresh(self._focus_library_hub_entry)

    # ----- Export canvas: form fields ------------------------------------

    @on(Input.Changed, "#library-export-name")
    def handle_library_export_name_changed(self, event: Input.Changed) -> None:
        """Track the export name text as the user types it (state only)."""
        event.stop()
        self._library_export_form["name"] = event.value

    @on(Input.Changed, "#library-export-description")
    def handle_library_export_description_changed(self, event: Input.Changed) -> None:
        """Track the export description text as the user types it (state only)."""
        event.stop()
        self._library_export_form["description"] = event.value

    @on(Button.Pressed, "#library-export-quality")
    def handle_library_export_quality(self, event: Button.Pressed) -> None:
        """Open or close the quality chooser's direct-pick strip.

        task-14902: the per-press thumbnail/compressed/original cycle
        retired -- the chooser opens a strip of all three values below the
        (still-visible) button, so a second press here also closes it.

        Args:
            event: Button press event emitted by the quality control.
        """
        event.stop()
        self._library_export_quality_choices_visible = (
            not self._library_export_quality_choices_visible
        )
        self.refresh(recompose=True)
        if self._library_export_quality_choices_visible:
            self.call_after_refresh(
                self._focus_library_choice_strip_active,
                ".library-export-quality-choice",
                str(self._library_export_form.get("quality", DEFAULT_MEDIA_QUALITY)),
            )
        else:
            self.call_after_refresh(
                self._focus_library_control, "#library-export-quality"
            )

    @on(Button.Pressed, ".library-export-quality-choice")
    def handle_library_export_quality_choice(self, event: Button.Pressed) -> None:
        """Apply the exact quality value carried by one strip choice.

        Args:
            event: Button press event emitted by a quality-strip option.
        """
        event.stop()
        requested = str(getattr(event.button, "choice_value", "") or "")
        self._library_export_quality_choices_visible = False
        if requested in MEDIA_QUALITY_OPTIONS:
            self._library_export_form["quality"] = requested
        self.refresh(recompose=True)
        self.call_after_refresh(self._focus_library_control, "#library-export-quality")

    @on(Button.Pressed, "#library-export-destination")
    def handle_library_export_choose_destination(self, event: Button.Pressed) -> None:
        """Push a ``FileSave`` dialog to pick the export's destination path.

        Mirrors ``_export_library_note``'s dialog flow: a sanitized
        default filename derived from the export name field, callback via
        ``call_after_refresh`` so the write-path runs after this handler
        returns. ``FileSave`` DOES have overwrite handling of its own
        (``can_overwrite: bool = True`` -- ``False`` blocks picking an
        existing file outright), but its default imposes no friction, and
        more importantly it can only ever judge the RAW picked path: the
        creator coerces the suffix to ``.zip``, so the path that must be
        confirmed for overwrite is the *normalized* one, which the dialog
        never sees. The form therefore owns overwrite confirmation of the
        normalized path (see ``_apply_library_export_destination``), and
        the dialog is deliberately left at its permissive default rather
        than ``can_overwrite=False`` (which would wrongly block picking
        ``report.zip`` even though the user is knowingly replacing it,
        while failing to block picking ``report`` when ``report.zip``
        exists).

        Args:
            event: Button press event emitted by the "Choose destination…"
                action.
        """
        event.stop()
        raw_name = str(self._library_export_form.get("name", "")).strip() or "bundle"
        safe_name = (
            "".join(
                char for char in raw_name if char.isalnum() or char in (" ", "-", "_")
            ).rstrip()
            or "bundle"
        )
        self.app.push_screen(
            FileSave(
                location=str(Path.home()),
                title="Choose Export Destination",
                default_file=f"{safe_name}.zip",
            ),
            callback=lambda path: self.call_after_refresh(
                self._apply_library_export_destination, path
            ),
        )

    def _apply_library_export_destination(self, selected_path: Path | None) -> None:
        """Validate, ``.zip``-normalize, and apply a ``FileSave``-picked destination.

        Runs the dialog-returned path through ``validate_path_simple``
        (same base-directory-free validator ``_write_library_note_export_file``
        uses for any user-chosen save path) BEFORE normalizing its suffix
        to ``.zip`` -- and normalizes BEFORE checking whether it already
        exists, so the overwrite line the form shows always names the
        actual path that will be written, never the raw picked one (the
        F4 design spec's explicit ordering: "normalized to .zip BEFORE any
        overwrite confirmation").

        Args:
            selected_path: The chosen destination, or ``None`` if the
                dialog was cancelled.
        """
        if not selected_path:
            return
        try:
            validated_path = validate_path_simple(selected_path, require_exists=False)
        except ValueError as exc:
            logger.warning(
                f"Rejected Library export destination {selected_path!r}: {exc}"
            )
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify(f"Rejected export destination: {exc}", severity="warning")
            return
        normalized_path = normalize_export_destination(validated_path)
        self._library_export_form["destination"] = str(normalized_path)
        self._library_export_form["destination_exists"] = normalized_path.exists()
        self.refresh(recompose=True)

# --- BEGIN generated export-controller-state shims ---
# Permanent, not a cleanup-PR deletion target -- same reasoning as
# `LibraryConversationsController`'s own identical block: the byte-for-byte
# canon (recipe §1) forbids editing a moved body, so the attribute names
# those bodies already use have to keep resolving through *something*.
# Exposes every `LibraryExportState` field under its original
# `_library_export_<field>` name on THIS controller, reading/writing
# through the injected `export_state_accessor` instead of a direct
# `self._export_state` attribute (this class has none) -- same generator
# shape task 2 installed on `LibraryScreen` and the conversations
# controller installed on itself, attached programmatically so the class
# body gains no `FunctionDef`s (the size ratchet counts those). Export uses
# a single `_library_export_` prefix for every field (task 2's report: no
# field needed a plural variant), so unlike Conversations there is no
# per-field prefix branch in this loop.
for _lec_field in dataclasses.fields(LibraryExportState):
    setattr(
        LibraryExportController,
        "_library_export_" + _lec_field.name,
        property(
            lambda self, _n=_lec_field.name: getattr(
                self._export_state_accessor(), _n
            ),
            lambda self, value, _n=_lec_field.name: setattr(
                self._export_state_accessor(), _n, value
            ),
        ),
    )
del _lec_field
# --- END generated export-controller-state shims ---

"""Library Conversations browse controller.

Controller PR of the Conversations extraction series (the decomposition
exemplar; ``backlog/docs/library-decomposition-recipe.md``,
``.superpowers/sdd/2026-09-01-library-decomposition-foundation`` task 8).
Owns the browse cluster moved verbatim out of ``LibraryScreen`` in
``tldw_chatbook/UI/Screens/library_screen.py``: every remaining
``conversation``-named method NOT already moved to
``LibraryConversationReaderController`` (task 7) -- list paging, row
selection/multiselect, export, filter, empty/retry states, and the
"Use in Console"/"Use as source" handoff -- plus the record-lookup and
label helpers those clusters share. ``LibraryScreen`` keeps one-line
delegators under every one of these original names.

**Ownership decision recorded per the recipe's brief:**
``_set_library_destination_with_conversation_fence`` stays on the screen,
UNMOVED. Despite its name, it is the shared rail/destination-switch helper
every subsystem's row-open and rail-switch dispatch calls (Notes, Media,
Ingest Export, and Conversations alike -- verified via
``grep -n "_set_library_destination_with_conversation_fence("
tldw_chatbook/UI/Screens/library_screen.py``: callers include the rail
switch dispatcher, the Ingest Export opener, and multiple row-open
dispatchers across subsystems, not just Conversations). Its only
Conversations-specific bit is invalidating the reader authority as a side
effect when navigating AWAY from Conversations -- the primary
responsibility (setting the shared, ≥2-subsystems ``_library_selected_row_id``)
is shell-wide. This matches the recipe's own explicit exclusion: "belongs
to another subsystem or shell".

Two names this task's mechanical re-enumeration caught that task 7's own
21-name cluster did NOT take (task 7's report already flags its own
``startswith("_conversation")`` shortcut as having missed two names; this
task's re-derivation, done withOUT that shortcut, additionally surfaces
these two, genuinely new, not task-7 misses): ``_library_conversation_focus_region``
and ``_library_conversation_escape_label``. Both concern the reader shell's
focus/Escape routing and were left on the screen by task 7 (not part of its
21-method list, not one of its 3 excluded label helpers either) -- since
they are Conversations-exclusive (gated on
``_library_selected_row_id == LIBRARY_ROW_BROWSE_CONVERSATIONS`` by every
caller) and not reader-cluster-owned by task 7's own accounting, they fall
to this cluster by the brief's "ALL remaining conversation-named methods"
default. Their callers (the help-footer shortcuts builder, ``check_action``,
``action_library_list_focus_rail``) are shell/plumbing, not Conversations
-- exactly the "still moves; shims keep them working" case the recipe's
§2 describes for non-subsystem shell callers.

**A second exclusion, found only by running this task's own verification
sweep (not by static analysis):** five names stay on ``LibraryScreen``,
UNMOVED, because ``Tests/UI/test_library_multiselect_conversations.py``
calls them directly on hand-built ``SimpleNamespace`` "fake screen"
objects -- e.g. ``LibraryScreen.handle_library_conversation_row(fake, ev)``
-- that provide only the exact free names the ORIGINAL bodies needed, not
a ``_conversations_controller`` attribute. This is a sibling failure mode
to the recipe's §3 monkeypatch-routing rule (same root cause -- a test
bypassing normal instance construction to intercept/inspect a method
directly -- same remedy: keep the whole name screen-routed), just reached
through unbound-class-method access instead of ``monkeypatch.setattr``.
The extraction-PR "never edit tests" rule left exactly one option:
``handle_library_conversation_row``, ``_library_conversation_loaded_preview_selected``,
``handle_library_conversations_export_selected``,
``handle_library_conversations_empty_console``, and
``handle_library_conversations_empty_clear_filter`` all stay real,
unmoved methods on ``LibraryScreen`` -- no delegator, body untouched.
Three OTHER methods that DO move
(``handle_library_conversations_select_toggle``/``_select_all``/
``_select_clear``) call ``_library_conversation_loaded_preview_selected``
internally, so it is bound here as one more named constructor dependency
(``library_conversation_loaded_preview_selected``) despite its own body
staying screen-resident -- see ``__init__``'s docstring for that
parameter. The other four have no internal callers elsewhere in this
cluster (verified by grep against the pre-move source) and need no
binding at all.

**A third exclusion, found only by this task's paired-baseline xdist
sweep (recipe §7) -- one method, a DIFFERENT bypass shape from the
``SimpleNamespace`` fakes above:** ``_selected_conversation_handoff_payload``
also stays on ``LibraryScreen``, UNMOVED.
``Tests/UI/test_post_release_workspaces_library_depth.py`` does
``screen._selected_conversation_handoff_payload = lambda: payload``
directly on a REAL, fully-constructed ``LibraryScreen`` instance (built
through a real harness, ``_conversations_controller`` and all) -- an
INSTANCE-attribute monkeypatch, not a class-level one and not an unbound-
method call. That only breaks because ``_open_selected_conversation_handoff``
(which DOES stay moved) calls ``self._selected_conversation_handoff_payload()``
internally: once both live on the controller, that internal call resolves
against the CONTROLLER instance, never seeing a patch applied to the
SCREEN instance -- a real regression this task's `-k "conversation and
library"`/paired-xdist-sweep evidence caught
(`test_single_item_handoff_gates_on_the_selected_row_not_the_aggregate`),
not the byte-for-byte diff, the free-name-resolution script, or the
characterization pins (none of which drive this call path with an
instance-level monkeypatch). Bound as one more named constructor
dependency (``selected_conversation_handoff_payload``), same shape as
``library_conversation_loaded_preview_selected`` above -- see
``__init__``'s docstring. This does NOT cascade to the three label
helpers below: their only caller is `_selected_conversation_handoff_payload`
itself, and since a `@classmethod` delegator does not depend on instance
construction, it does not matter which object (screen or controller)
ends up calling it.

The three label helpers task 7 excluded (``_conversation_message_count_label``,
``_conversation_workspace_label``, ``_conversation_updated_label``) join
this cluster anyway: their only caller,
``_selected_conversation_handoff_payload`` (the "Use in Console"
handoff-payload builder), is a browse/handoff concern, not reader-cluster
-- even though (see the THIRD exclusion below) that caller itself ends up
staying on ``LibraryScreen``, unmoved. A classmethod delegator does not
depend on ``_conversations_controller`` construction the way an instance
delegator does, so this move is safe regardless of which object the
caller runs on. Two of the three (``_conversation_workspace_label``,
``_conversation_updated_label``) call ``cls._safe_text(...)`` -- a general,
non-Conversations, ~36-call-site Library-wide staticmethod that stays on
``LibraryScreen`` (moving it is out of scope and would touch every other
subsystem). Byte-for-byte canon keeps these three as ``@classmethod``\\ s
(their screen delegators keep the ``@classmethod`` decorator too, per this
task's correction to task 7's minor -- see ``__init__``'s docstring), which
means ``cls`` is this controller class, not an instance -- the usual
per-instance constructor-dependency lambda (binding kind 2 below) has no
instance to attach to when ``cls`` is the class object itself. Resolved
with ONE class-level rebinding, done from ``library_screen.py`` after both
classes are fully defined (avoiding the circular import a controller-side
``from ..Screens.library_screen import LibraryScreen`` would create):
``LibraryConversationsController._safe_text = staticmethod(LibraryScreen._safe_text)``.
See ``library_screen.py``'s own trailing module-level code, next to the
generated state-shim block, for that one line.

Two binding kinds only (moved method bodies are never edited -- every name
they reference that is not this controller's own state is rebound under
the SAME name the body already used; see
``ConsoleDictationController.__init__``,
``tldw_chatbook/UI/Console_Modules/dictation.py``, the canonical worked
example this constructor's shape mirrors, and
``LibraryConversationReaderController`` -- task 7 -- for the sibling
controller in this same series):

1. **Framework services** (``run_worker``, ``query_one``, ``query``,
   ``call_after_refresh``, ``focused``, ``app`` -- Textual's own --,
   ``is_running``, ``refresh``, and this project's screen-level analogue of
   Textual's ``self.app`` -- ``app_instance``) are live-read from the
   screen via ``@property`` on every access -- never snapshotted. The last
   three (``app``, ``is_running``, ``refresh``) were not part of this
   cluster's original free-name inventory -- they surfaced only when
   ``_sync_library_conversation_canvas``'s moved body (byte-for-byte)
   turned out to forward ``self`` into the shared, multi-kind
   ``_sync_library_canvas(screen, kind, ...)`` dispatcher under the
   parameter name ``screen``: a real, runtime-only bug (rows never
   rendered despite the page loading fine) this task's characterization
   run caught that no static free-name-resolution pass could have, since
   ``self`` is never a "free name" needing an import -- see the task
   report's "a real bug found and fixed" section.
2. **Everything else** the cluster depends on that is not its own state is
   a NAMED constructor dependency. This cluster's dependencies fall into
   four groups: (a) general Library-wide screen helpers the moved bodies
   call with explicit arguments (``_acknowledge_library_destination_change``,
   ``_library_workspace_depth_state``, ``_open_library_export_canvas``,
   ``_open_library_item_by_id``, ``_run_library_service_call``,
   ``_safe_text``, ``_source_record_id``, ``_source_title``); (b) three
   cross-controller calls INTO the sibling reader controller (task 7) --
   ``_ensure_library_conversation_reader_selection``,
   ``_start_library_conversation_reader_selection``,
   ``_sync_library_conversation_reader`` -- bound by the screen as named
   callables per the "controllers never import each other" rule; (c)
   shared, ≥2-subsystems shell state this cluster both READS and WRITES
   (``_selected_conversation_id``, ``_pending_library_source_open``,
   ``_library_selected_row_id``, and -- per the ``_sync_library_canvas``
   forwarding above -- ``_library_canvas_resync_pending`` -- unlike task
   7's reader cluster, which only ever READS the first three, this browse
   cluster writes to all four, so each gets a getter AND a setter
   callable, not a read-only accessor); and (d) shared shell fields this
   cluster only reads (``_local_source_records``, the per-subsystem
   records cache other subsystems' background refresh replaces wholesale;
   ``_library_canvas_projection_depth``, the same ``_sync_library_canvas``
   forwarding's reentrancy guard).

This subsystem's OWN state (every ``_library_conversation*`` /
``_library_conversations_*`` name the moved bodies reference) is exposed
through generated properties reading
``self._conversations_state_accessor().<field>`` -- the same generator
shape Task 6 installed on ``LibraryScreen`` and Task 7 installed on
``LibraryConversationReaderController``, applied here to this controller.
The plural/singular prefix set (``CONVERSATIONS_PLURAL_STATE_FIELDS``) is
imported from ``library_conversations_state`` -- the dataclass's own
module -- rather than kept as a third local literal copy: task 7's fix
round 1 flagged the screen's and the reader controller's own independent
copies of this same two-name set as a concrete drift risk (a future field
added to one copy and not the other fails silently, as an
``AttributeError`` inside whichever moved body reaches for it first, under
the wrong prefix). This task promotes the set to one shared home instead of
adding a third copy; see that module's own docstring addendum.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Literal, TYPE_CHECKING

from loguru import logger
from textual import on
from textual.css.query import NoMatches, QueryError
from textual.widget import Widget
from textual.widgets import Button, Input

from ...Chat.chat_handoff_models import ChatHandoffPayload
from ...Library.library_conversation_reader_state import (
    LIBRARY_CONVERSATION_PAGE_SIZE,
    confirm_conversation_deleted,
    project_conversation_multiselect,
)
from ...Library.library_conversations_state import (
    build_library_conversations_state,
    validate_library_conversation_page,
)
from ...Library.library_export_scope import ExportScope
from ...Library.library_shell_state import LIBRARY_ROW_BROWSE_CONVERSATIONS
from ...Widgets.Library import LibraryAdaptiveReaderShell, LibraryConversationsCanvas
from ...Workspaces import library_item_context_handoff
from .canvas_sync import _apply_library_row_toggle, _sync_library_canvas
from .library_conversations_state import (
    CONVERSATIONS_PLURAL_STATE_FIELDS,
    LibraryConversationsState,
)

if TYPE_CHECKING:
    from ..Screens.library_screen import LibraryScreen


class LibraryConversationsController:
    """Owns the Conversations browse cluster: list/paging, selection,
    export, filter, and the Console handoff.

    Holds no state of its own beyond what it reads and writes through
    ``LibraryConversationsState`` (via the injected accessor) or the shared
    shell attributes bound below. ``LibraryScreen`` constructs exactly one
    of these, in ``__init__`` right after
    ``self._conversation_reader_controller`` (task 7), so this
    constructor's ``ensure_reader_selection``/``start_reader_selection``/
    ``sync_reader`` callables can close over an already-built reader
    controller, and keeps one-line delegators for every original name this
    cluster moved.
    """

    def __init__(
        self,
        screen: "LibraryScreen",
        *,
        conversations_state_accessor: Callable[[], LibraryConversationsState],
        # -- cross-controller calls into the sibling reader controller
        # (task 7) -- named callables the SCREEN wires, per "controllers
        # never import each other".
        ensure_reader_selection: Callable[[], None],
        start_reader_selection: Callable[[str], None],
        sync_reader: Callable[[], None],
        # -- shared, >=2-subsystems shell state this cluster reads AND
        # writes -- getter + setter, unlike the reader cluster's read-only
        # accessors for the same two names (this cluster mutates them).
        selected_conversation_id_accessor: Callable[[], str],
        set_selected_conversation_id: Callable[[str], None],
        pending_library_source_open_accessor: Callable[[], Any],
        set_pending_library_source_open: Callable[[Any], None],
        selected_row_id_accessor: Callable[[], str],
        set_selected_row_id: Callable[[str], None],
        # -- shared shell state this cluster only reads.
        local_source_records_accessor: Callable[[], Mapping[str, Any]],
        library_canvas_projection_depth_accessor: Callable[[], int],
        # -- shared shell state this cluster both reads and writes, through
        # the `_sync_library_canvas` dispatcher it forwards `self` into
        # (see the `app`/`is_running`/`refresh` properties' docstrings).
        library_canvas_resync_pending_accessor: Callable[[], bool],
        set_library_canvas_resync_pending: Callable[[bool], None],
        # -- general Library-wide screen helpers, not moved (shared with
        # other subsystems; see module docstring group (a)).
        acknowledge_destination_change: Callable[[], None],
        library_workspace_depth_state: Callable[..., Any],
        open_library_export_canvas: Callable[..., Awaitable[Any]],
        open_library_item_by_id: Callable[..., Awaitable[Any]],
        run_library_service_call: Callable[..., Awaitable[Any]],
        safe_text: Callable[..., str],
        source_record_id: Callable[[Mapping[str, Any]], str | None],
        source_title: Callable[[str, Mapping[str, Any]], str],
        # -- stays on LibraryScreen (see module docstring's "fake-self unit
        # test" exclusion note) but three moved multiselect handlers call
        # it internally, so it is bound like any other general dependency.
        library_conversation_loaded_preview_selected: Callable[[], bool | None],
        # -- stays on LibraryScreen (see module docstring's "instance-
        # attribute monkeypatch" exclusion note) but the moved
        # `_open_selected_conversation_handoff` calls it internally.
        selected_conversation_handoff_payload: Callable[[], ChatHandoffPayload | None],
    ) -> None:
        """Build the controller and bind everything its moved bodies need.

        Every one of the 40 method bodies below is a byte-for-byte copy of
        the pre-extraction ``LibraryScreen`` method: no internal line was
        edited to retarget a call or an attribute. That is possible
        because this constructor binds every name those bodies reference
        that is not this controller's own state, under the SAME name the
        original method used. See the module docstring for the binding
        kinds this follows.

        Args:
            screen: The Library screen. Used ONLY for the nine framework
                services below (``run_worker``, ``query_one``, ``query``,
                ``call_after_refresh``, ``focused``, ``app_instance``,
                ``app``, ``is_running``, ``refresh``) -- this cluster owns
                no DOM of its own.
            conversations_state_accessor: Returns the live
                ``LibraryConversationsState``
                (``LibraryScreen._conversations_state``) -- the SAME
                accessor the reader controller (task 7) already receives.
                Backs every generated ``_library_conversation*`` /
                ``_library_conversations_*`` property below.
            ensure_reader_selection: ``LibraryScreen.
                _ensure_library_conversation_reader_selection`` (task 7,
                reached through the reader controller) -- starts/confirms
                the permanent work pane's read for whatever row this
                cluster just made current. Called after a page load
                settles and after leaving multiselect.
            start_reader_selection: ``LibraryScreen.
                _start_library_conversation_reader_selection`` (task 7) --
                launches the fenced progressive reader for one explicitly
                clicked row. The brief's own worked example for this
                binding shape.
            sync_reader: ``LibraryScreen._sync_library_conversation_reader``
                (task 7) -- patches the mounted reader pane after this
                cluster's own state changes (multiselect toggle, page
                request start, page load settling).
            selected_conversation_id_accessor: Reads ``LibraryScreen.
                _selected_conversation_id`` -- a per-source "currently
                selected" field parallel to ``_selected_media_id``/
                ``_selected_note_id``, shared shell state despite its
                name. The reader cluster (task 7) only ever reads it; THIS
                cluster also writes it (row click, page-load reselection,
                list normalization) -- hence the paired setter below.
            set_selected_conversation_id: Writes
                ``LibraryScreen._selected_conversation_id`` back through
                the screen's own attribute, so shell code reading it
                directly (outside either controller) sees this cluster's
                writes immediately.
            pending_library_source_open_accessor: Reads ``LibraryScreen.
                _pending_library_source_open`` -- a deep-link intent tuple
                shared across every source type (media, conversations),
                one of the recipe's own canonical >=2-subsystems examples
                (29 refs). Both read (retry, list-normalization guard) and
                written (page-request start clears a stale intent) inside
                this cluster.
            set_pending_library_source_open: Writes ``LibraryScreen.
                _pending_library_source_open`` back through the screen.
            selected_row_id_accessor: Reads ``LibraryScreen.
                _library_selected_row_id`` -- the recipe's own canonical
                >=2-subsystems shared field (226 refs). The reader cluster
                (task 7) only reads it; THIS cluster also writes it (a row
                click reaffirms Conversations as the active destination)
                -- hence the paired setter below.
            set_selected_row_id: Writes ``LibraryScreen.
                _library_selected_row_id`` back through the screen.
            local_source_records_accessor: Reads ``LibraryScreen.
                _local_source_records`` (via the SAME
                ``getattr(self, "_local_source_records", {})`` defensive
                default the original body used, replicated one level down
                at the screen -- the moved body's own
                ``getattr(self, "_local_source_records", {})`` keeps
                working unedited because this controller exposes a real
                ``_local_source_records`` property under that name; see
                module docstring group (d)). Read-only in this cluster --
                the periodic background snapshot refresh (a DIFFERENT,
                screen-only, monkeypatch-routed method per the recipe's
                §3) owns every write site.
            library_canvas_projection_depth_accessor: Reads ``LibraryScreen.
                _library_canvas_projection_depth`` -- a targeted-canvas-
                projection reentrancy guard shared across EVERY Library
                canvas kind, not Conversations-owned. Needed only because
                ``_sync_library_conversation_canvas``'s moved body (byte-
                for-byte) forwards ``self`` into the shared
                ``_sync_library_canvas(screen, kind, ...)`` dispatcher under
                the parameter name ``screen`` -- a real bug this task's
                characterization run caught (rows never rendered despite
                the page loading fine; see the report's "a real bug found
                and fixed" section). Read-only in this cluster.
            library_canvas_resync_pending_accessor: Reads ``LibraryScreen.
                _library_canvas_resync_pending``. Same
                ``_sync_library_canvas``-forwarding reason as
                ``library_canvas_projection_depth_accessor`` above -- WRITTEN
                by that dispatcher (not by this cluster's own bodies
                directly), hence the paired setter below.
            set_library_canvas_resync_pending: Writes ``LibraryScreen.
                _library_canvas_resync_pending`` back through the screen.
            acknowledge_destination_change: ``LibraryScreen.
                _acknowledge_library_destination_change`` -- clears rail
                transition feedback; called before admitting an explicit
                row click, shared with every other subsystem's row click.
            library_workspace_depth_state: ``LibraryScreen.
                _library_workspace_depth_state`` -- the shared
                cross-subsystem workspace-eligibility cache the handoff
                gate reads before staging a payload.
            open_library_export_canvas: ``LibraryScreen.
                _open_library_export_canvas`` -- the shared export-canvas
                opener every browse section's "Export..." action (media/
                conversations/notes/prompts) calls.
            open_library_item_by_id: ``LibraryScreen.
                _open_library_item_by_id`` -- the shared per-result
                Search/RAG/deep-link "Open" route, used here only to
                retry one retained pending Conversation locator intent.
            run_library_service_call: ``LibraryScreen.
                _run_library_service_call`` -- the shared off-thread
                service-call wrapper every Library subsystem's list/detail
                fetch goes through. SAME binding shape as the reader
                controller's own (task 7).
            safe_text: ``LibraryScreen._safe_text`` -- the general,
                ~36-call-site Library-wide input-sanitizing staticmethod.
                Bound as an INSTANCE-level dependency for this cluster's
                three regular (non-classmethod) call sites; see the module
                docstring's class-level rebinding note for the two
                CLASSMETHOD call sites, which this constructor parameter
                does not cover (a classmethod dispatches via ``cls``, not
                an instance, so it cannot reach a per-instance injected
                callable).
            source_record_id: ``LibraryScreen._source_record_id`` -- the
                general, cross-subsystem record-id resolver (also used by
                Media, Notes, etc.), not itself Conversations-owned.
            source_title: ``LibraryScreen._source_title`` -- the general,
                cross-subsystem title resolver, same scope as
                ``source_record_id``.
            library_conversation_loaded_preview_selected: ``LibraryScreen.
                _library_conversation_loaded_preview_selected`` -- returns
                whether the retained transcript is in the checked-row set.
                NOT moved (see module docstring's "fake-self unit test"
                exclusion note: ``Tests/UI/test_library_multiselect_conversations.py``
                calls ``LibraryScreen._library_conversation_loaded_preview_selected``
                and ``LibraryScreen.handle_library_conversation_row`` directly
                on hand-built ``SimpleNamespace`` fakes that provide only the
                exact free names the pre-move bodies needed -- moving either
                one to a delegator broke both, an ``AttributeError`` this
                task's ``-k "conversation and library"`` sweep caught, not
                the byte-for-byte diff or the free-name-resolution script).
                Three OTHER moved bodies in this cluster
                (``handle_library_conversations_select_toggle``/``_select_all``/
                ``_select_clear``) still call it via ``self.<name>()``, so it
                is bound here exactly like ``source_record_id``/``source_title``
                above, even though its own body stays screen-resident.
            selected_conversation_handoff_payload: ``LibraryScreen.
                _selected_conversation_handoff_payload`` -- builds the "Use
                in Console" handoff payload. NOT moved (see module
                docstring's "instance-attribute monkeypatch" exclusion
                note: ``Tests/UI/test_post_release_workspaces_library_depth.py``
                does ``screen._selected_conversation_handoff_payload =
                lambda: payload`` directly on a REAL, fully-constructed
                screen instance, expecting ``_open_selected_conversation_handoff``
                -- which DOES stay moved -- to observe the patch on its
                next internal call. That only works if both resolve
                through the SAME object; moving just the payload builder
                broke it, a genuine regression this task's paired-baseline
                xdist sweep caught (`test_single_item_handoff_gates_on_the_selected_row_not_the_aggregate`),
                not the byte-for-byte diff, the free-name script, or the
                characterization pins). Bound here exactly like
                ``library_conversation_loaded_preview_selected`` above.
        """
        self._screen = screen
        self._conversations_state_accessor = conversations_state_accessor
        self._ensure_reader_selection_fn = ensure_reader_selection
        self._start_reader_selection_fn = start_reader_selection
        self._sync_reader_fn = sync_reader
        self._selected_conversation_id_accessor = selected_conversation_id_accessor
        self._set_selected_conversation_id_fn = set_selected_conversation_id
        self._pending_library_source_open_accessor = (
            pending_library_source_open_accessor
        )
        self._set_pending_library_source_open_fn = set_pending_library_source_open
        self._selected_row_id_accessor = selected_row_id_accessor
        self._set_selected_row_id_fn = set_selected_row_id
        self._local_source_records_accessor = local_source_records_accessor
        self._library_canvas_projection_depth_accessor = (
            library_canvas_projection_depth_accessor
        )
        self._library_canvas_resync_pending_accessor = (
            library_canvas_resync_pending_accessor
        )
        self._set_library_canvas_resync_pending_fn = (
            set_library_canvas_resync_pending
        )
        self._acknowledge_destination_change_fn = acknowledge_destination_change
        self._library_workspace_depth_state_fn = library_workspace_depth_state
        self._open_library_export_canvas_fn = open_library_export_canvas
        self._open_library_item_by_id_fn = open_library_item_by_id
        self._run_library_service_call_fn = run_library_service_call
        self._safe_text_fn = safe_text
        self._source_record_id_fn = source_record_id
        self._source_title_fn = source_title
        self._library_conversation_loaded_preview_selected_fn = (
            library_conversation_loaded_preview_selected
        )
        self._selected_conversation_handoff_payload_fn = (
            selected_conversation_handoff_payload
        )

    # -- framework services: live-read properties, never snapshotted -----

    @property
    def run_worker(self) -> Any:
        """``Screen.run_worker``, bound. See ``__init__``'s docstring."""
        return self._screen.run_worker

    @property
    def query_one(self) -> Any:
        """``Screen.query_one``, bound. See ``__init__``'s docstring."""
        return self._screen.query_one

    @property
    def query(self) -> Any:
        """``Screen.query``, bound. See ``__init__``'s docstring."""
        return self._screen.query

    @property
    def call_after_refresh(self) -> Any:
        """``Screen.call_after_refresh``, bound. See ``__init__``'s
        docstring."""
        return self._screen.call_after_refresh

    @property
    def focused(self) -> Widget | None:
        """``Screen.focused``, live-read. See ``__init__``'s docstring."""
        return self._screen.focused

    @property
    def app_instance(self) -> Any:
        """The running app instance, live-read from the screen.

        This project's screen-level analogue of Textual's own ``self.app``
        -- see ``__init__``'s docstring.
        """
        return self._screen.app_instance

    @property
    def app(self) -> Any:
        """``Screen.app``, live-read -- Textual's OWN app property (distinct
        from this project's ``app_instance`` above). Needed because
        ``_sync_library_conversation_canvas``'s moved body forwards ``self``
        to the shared, multi-kind ``_sync_library_canvas(screen, kind, ...)``
        dispatcher (``canvas_sync.py``) under the parameter name ``screen``
        -- a real bug this task's characterization run caught (see the
        report's "a real bug found and fixed" section): that dispatcher's
        `kind == "conversations"` branch, and its unconditional mouse-capture
        release, read several ``screen.*`` names this controller did not
        originally expose. See ``__init__``'s docstring."""
        return self._screen.app

    @property
    def is_running(self) -> bool:
        """``Screen.is_running``, live-read. Same ``_sync_library_canvas``
        dependency as ``app`` immediately above."""
        return self._screen.is_running

    @property
    def refresh(self) -> Any:
        """``Screen.refresh``, bound. Same ``_sync_library_canvas``
        dependency as ``app``/``is_running`` above -- reached only on its
        error-fallback path (the four DIRECT ``_sync_library_canvas(self,
        "conversations")`` call sites in this cluster's ``@on`` handlers
        default ``allow_screen_fallback`` to ``True``, unlike
        ``_sync_library_conversation_canvas``'s own explicit ``False``)."""
        return self._screen.refresh

    # -- named constructor dependencies -----------------------------------

    @property
    def _ensure_library_conversation_reader_selection(self) -> Any:
        """The injected ``ensure_reader_selection``. See ``__init__``'s
        docstring."""
        return self._ensure_reader_selection_fn

    @property
    def _start_library_conversation_reader_selection(self) -> Any:
        """The injected ``start_reader_selection``. See ``__init__``'s
        docstring."""
        return self._start_reader_selection_fn

    @property
    def _sync_library_conversation_reader(self) -> Any:
        """The injected ``sync_reader``. See ``__init__``'s docstring."""
        return self._sync_reader_fn

    @property
    def _selected_conversation_id(self) -> str:
        """Calls the injected ``selected_conversation_id_accessor``. See
        ``__init__``'s docstring."""
        return self._selected_conversation_id_accessor()

    @_selected_conversation_id.setter
    def _selected_conversation_id(self, value: str) -> None:
        """Calls the injected ``set_selected_conversation_id``. See
        ``__init__``'s docstring."""
        self._set_selected_conversation_id_fn(value)

    @property
    def _pending_library_source_open(self) -> Any:
        """Calls the injected ``pending_library_source_open_accessor``.
        See ``__init__``'s docstring."""
        return self._pending_library_source_open_accessor()

    @_pending_library_source_open.setter
    def _pending_library_source_open(self, value: Any) -> None:
        """Calls the injected ``set_pending_library_source_open``. See
        ``__init__``'s docstring."""
        self._set_pending_library_source_open_fn(value)

    @property
    def _library_selected_row_id(self) -> str:
        """Calls the injected ``selected_row_id_accessor``. See
        ``__init__``'s docstring."""
        return self._selected_row_id_accessor()

    @_library_selected_row_id.setter
    def _library_selected_row_id(self, value: str) -> None:
        """Calls the injected ``set_selected_row_id``. See ``__init__``'s
        docstring."""
        self._set_selected_row_id_fn(value)

    @property
    def _local_source_records(self) -> Any:
        """Calls the injected ``local_source_records_accessor``.
        Read-only here; see ``__init__``'s docstring."""
        return self._local_source_records_accessor()

    @property
    def _library_canvas_projection_depth(self) -> int:
        """Calls the injected ``library_canvas_projection_depth_accessor``.
        Read-only here (``_sync_library_canvas`` only ever reads it via
        ``getattr(screen, "_library_canvas_projection_depth", 0)``); shared
        shell state used by EVERY Library canvas kind's targeted sync, not
        Conversations-exclusive. See the ``app``/``is_running``/``refresh``
        properties above and ``__init__``'s docstring for why this cluster
        needs it at all."""
        return self._library_canvas_projection_depth_accessor()

    @property
    def _library_canvas_resync_pending(self) -> bool:
        """Calls the injected ``library_canvas_resync_pending_accessor``.
        See ``_library_canvas_projection_depth`` immediately above."""
        return self._library_canvas_resync_pending_accessor()

    @_library_canvas_resync_pending.setter
    def _library_canvas_resync_pending(self, value: bool) -> None:
        """Calls the injected ``set_library_canvas_resync_pending``. See
        ``_library_canvas_projection_depth`` above."""
        self._set_library_canvas_resync_pending_fn(value)

    @property
    def _acknowledge_library_destination_change(self) -> Any:
        """The injected ``acknowledge_destination_change``. See
        ``__init__``'s docstring."""
        return self._acknowledge_destination_change_fn

    @property
    def _library_workspace_depth_state(self) -> Any:
        """The injected ``library_workspace_depth_state``. See
        ``__init__``'s docstring."""
        return self._library_workspace_depth_state_fn

    @property
    def _open_library_export_canvas(self) -> Any:
        """The injected ``open_library_export_canvas``. See ``__init__``'s
        docstring."""
        return self._open_library_export_canvas_fn

    @property
    def _open_library_item_by_id(self) -> Any:
        """The injected ``open_library_item_by_id``. See ``__init__``'s
        docstring."""
        return self._open_library_item_by_id_fn

    @property
    def _run_library_service_call(self) -> Any:
        """The injected ``run_library_service_call``. See ``__init__``'s
        docstring."""
        return self._run_library_service_call_fn

    @property
    def _safe_text(self) -> Any:
        """The injected ``safe_text``. Instance-level uses only -- see the
        module docstring's class-level rebinding note for this cluster's
        two classmethod call sites."""
        return self._safe_text_fn

    @property
    def _source_record_id(self) -> Any:
        """The injected ``source_record_id``. See ``__init__``'s
        docstring."""
        return self._source_record_id_fn

    @property
    def _source_title(self) -> Any:
        """The injected ``source_title``. See ``__init__``'s docstring."""
        return self._source_title_fn

    @property
    def _library_conversation_loaded_preview_selected(self) -> Any:
        """The injected ``library_conversation_loaded_preview_selected``.
        This name's own body stays on ``LibraryScreen`` (NOT moved -- see
        module docstring); three OTHER moved bodies below still call it via
        ``self.<name>()``, so it is bound here like any other general
        dependency. See ``__init__``'s docstring."""
        return self._library_conversation_loaded_preview_selected_fn

    @property
    def _selected_conversation_handoff_payload(self) -> Any:
        """The injected ``selected_conversation_handoff_payload``. This
        name's own body stays on ``LibraryScreen`` (NOT moved -- see
        module docstring); the moved ``_open_selected_conversation_handoff``
        below still calls it via ``self.<name>()``, so it is bound here
        like any other general dependency. See ``__init__``'s docstring."""
        return self._selected_conversation_handoff_payload_fn

    # -- moved bodies (byte-for-byte; see module docstring) ---------------

    def _library_conversation_focus_region(self) -> str:
        """Return the retained Conversations role containing current focus."""
        focused = self.focused
        if focused is None:
            return ""
        try:
            shell = self.query_one(
                "#library-conversations-reader-shell", LibraryAdaptiveReaderShell
            )
        except (NoMatches, QueryError):
            return ""
        for name, pane in (
            ("library", shell.library),
            ("items", shell.items),
            ("work", shell.work),
        ):
            if focused is pane or pane in focused.ancestors:
                return name
        return ""

    def _library_conversation_escape_label(self) -> str:
        """Name the nearest visible prior role reached by Escape."""
        region = self._library_conversation_focus_region()
        layout = self._library_conversation_reader_layout
        if region == "work" and layout.items_open:
            return "focus Items"
        if region in {"work", "items"} and layout.library_open:
            return "focus Library"
        return ""

    def _adopt_library_conversation_state_selection(self, selected_id: str) -> None:
        """Adopt list normalization unless it would consume a pending deep link."""

        pending = self._pending_library_source_open
        if (
            pending is not None
            and pending[0] == "conversations"
            and self._selected_conversation_id == pending[1]
            and selected_id != pending[1]
        ):
            return
        retained_reader_id = self._library_conversation_reader_state.selected_id
        if (
            retained_reader_id
            and not self._library_conversation_reader_state.unavailable
            and selected_id != retained_reader_id
        ):
            return
        self._selected_conversation_id = selected_id

    def _carry_selected_conversation_into_snapshot(
        self,
        records: dict[str, tuple[Mapping[str, Any], ...]],
    ) -> dict[str, tuple[Mapping[str, Any], ...]]:
        """Preserve an out-of-page selected conversation across a snapshot replace.

        (C3) A wholesale ``_local_source_records`` replace -- the periodic
        background refresh, not a user action -- can silently drop the
        currently-open conversation if it fell off the loaded page (the
        conversations snapshot is capped, see
        ``LIBRARY_SOURCE_PAGE_SIZES["conversations"]``) or was fetched
        out-of-band via ``_open_library_item_by_id`` and prepended into the
        OLD records. Without this, the next recompose would silently reset
        the selection to the first row (``_ensure_selected_conversation_id``)
        even though the user never navigated away -- the same class of race
        ``_open_library_item_by_id`` already guards against for its own
        out-of-snapshot fetch, just triggered by a background refresh
        instead of a user click.

        Pure in-memory merge: reads the OLD ``self._local_source_records``
        (not yet replaced) and the INCOMING ``records``, and -- only when the
        selected id is present in the old snapshot but missing from the new
        one -- prepends the old record into the new conversations tuple so
        the selection survives the replace.

        Args:
            records: The incoming snapshot about to replace
                ``self._local_source_records``.

        Returns:
            ``records``, unchanged, or with the selected conversation's
            record prepended into its ``"conversations"`` tuple.
        """
        selected_id = getattr(self, "_selected_conversation_id", "")
        if not selected_id:
            return records
        old_conversations = getattr(self, "_local_source_records", {}).get(
            "conversations", ()
        )
        old_index_by_id = {
            self._conversation_record_id(record, index): record
            for index, record in enumerate(old_conversations)
        }
        carried_record = old_index_by_id.get(selected_id)
        if carried_record is None:
            # Not present in the old snapshot either -- nothing to carry.
            return records
        new_conversations = records.get("conversations", ())
        new_ids = {
            self._conversation_record_id(record, index)
            for index, record in enumerate(new_conversations)
        }
        if selected_id in new_ids:
            # Still present in the incoming snapshot -- no carry-over needed.
            return records
        merged = dict(records)
        merged["conversations"] = (carried_record, *new_conversations)[
            :LIBRARY_CONVERSATION_PAGE_SIZE
        ]
        return merged

    def _conversation_records(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self._library_conversation_page_records)

    def _conversation_record_id(self, record: Mapping[str, Any], index: int) -> str:
        return self._source_record_id(record) or f"conversation-{index + 1}"

    def _ensure_selected_conversation_id(self) -> str:
        records = self._conversation_records()
        record_ids = {
            self._conversation_record_id(record, index)
            for index, record in enumerate(records)
        }
        if self._selected_conversation_id in record_ids:
            return self._selected_conversation_id
        self._selected_conversation_id = (
            self._conversation_record_id(records[0], 0) if records else ""
        )
        return self._selected_conversation_id

    def _selected_conversation_record(self) -> tuple[int, Mapping[str, Any]] | None:
        selected_id = self._ensure_selected_conversation_id()
        if not selected_id:
            return None
        for index, record in enumerate(self._conversation_records()):
            if self._conversation_record_id(record, index) == selected_id:
                return index, record
        return None

    @classmethod
    def _conversation_message_count_label(cls, record: Mapping[str, Any]) -> str:
        for key in (
            "message_count",
            "messages_count",
            "messageCount",
            "message_total",
            "messages_total",
        ):
            value = record.get(key)
            if isinstance(value, int):
                return f"Messages: {value}"
            if isinstance(value, str) and value.strip().isdigit():
                return f"Messages: {value.strip()}"
        messages = record.get("messages")
        if isinstance(messages, Sequence) and not isinstance(
            messages, (str, bytes, bytearray)
        ):
            return f"Messages: {len(messages)}"
        return "Messages: unknown"

    @classmethod
    def _conversation_workspace_label(cls, record: Mapping[str, Any]) -> str:
        for key in ("workspace_name", "workspace_id", "workspace", "scope_id"):
            value = cls._safe_text(record.get(key), max_length=64)
            if value:
                return f"Workspace: {value}"
        return "Workspace: unassigned"

    @classmethod
    def _conversation_updated_label(cls, record: Mapping[str, Any]) -> str:
        for key in (
            "updated_at",
            "last_modified",
            "last_updated",
            "modified_at",
            "created_at",
        ):
            value = cls._safe_text(record.get(key), max_length=64)
            if value:
                return f"Updated: {value}"
        return "Updated: unknown"

    def _build_library_conversations_state(self):
        """Build the conversations canvas display state from local records."""
        state = build_library_conversations_state(
            self._conversation_records(),
            query=self._library_conversation_query,
            selected_id=self._selected_conversation_id,
            select_mode=self._library_conversations_select_mode,
            selected_ids=self._library_conversations_row_selection.ids,
            page=self._library_conversation_page,
            requested_page=self._library_conversation_requested_page,
            page_size=self._library_conversation_page_size,
            total_count=self._library_conversation_total,
            total_known=self._library_conversation_total_known,
            has_more=self._library_conversation_has_more,
            freshness=self._library_conversation_freshness,
            stale_copy=self._library_conversation_stale_copy,
            loading=self._library_conversation_loading,
            error_copy=self._library_conversation_error,
            selection_notice=self._library_conversation_selection_notice,
        )
        state = dataclasses.replace(
            state,
            query=self._library_conversation_requested_query,
            status_copy=(
                state.status_copy or self._library_conversation_selection_notice
            ),
        )
        retained_reader_id = self._library_conversation_reader_state.selected_id
        if (
            retained_reader_id
            and not self._library_conversation_reader_state.unavailable
            and all(row.conversation_id != retained_reader_id for row in state.rows)
        ):
            state = dataclasses.replace(
                state,
                selected_id="",
                preview_lines=(),
                rows=tuple(
                    dataclasses.replace(row, selected=False) for row in state.rows
                ),
            )
        if self._library_conversation_deleted_selection_id:
            state = dataclasses.replace(
                state,
                selected_id="",
                preview_lines=(),
                rows=tuple(
                    dataclasses.replace(row, selected=False) for row in state.rows
                ),
            )
        if self._library_conversations_select_mode:
            self._library_conversations_row_selection.reconcile(
                r.conversation_id for r in state.rows
            )
        return state

    def _sync_library_conversation_canvas(
        self, *, then: Callable[[], None] | None = None
    ) -> bool:
        """Sync Conversation state through its canvas-owned action gates."""

        return _sync_library_canvas(
            self,
            "conversations",
            then=then,
            allow_screen_fallback=False,
        )

    @staticmethod
    def _normalize_library_conversation_page(page: object) -> int:
        """Return a dispatch-safe one-based page, defaulting invalid input."""

        if isinstance(page, bool) or not isinstance(page, int) or page < 1:
            return 1
        max_page = (2**63 - 1) // LIBRARY_CONVERSATION_PAGE_SIZE + 1
        return page if page <= max_page else 1

    def _start_library_conversation_page_request(
        self,
        page: int,
        query: str,
        *,
        refocus_filter: bool = False,
        focus_after_apply: str = "",
    ) -> None:
        """Start one generation-guarded conversation page request."""
        self._pending_library_source_open = None
        requested_page = self._normalize_library_conversation_page(page)
        normalized_query, generation = self._prepare_library_conversation_page_request(
            query,
            page=requested_page,
            refocus_filter=refocus_filter,
            focus_after_apply=focus_after_apply,
        )
        self.run_worker(
            self._load_library_conversation_page(
                requested_page,
                normalized_query,
                generation,
            ),
            exclusive=True,
            group="library_conversation_page",
        )

    def _prepare_library_conversation_page_request(
        self,
        query: str,
        *,
        page: int = 1,
        refocus_filter: bool = False,
        focus_after_apply: str = "",
    ) -> tuple[str, int]:
        """Invalidate older requests and publish a coherent loading state."""
        self._library_conversation_request_generation += 1
        generation = self._library_conversation_request_generation
        normalized_query = self._safe_text(query, max_length=200)
        self._library_conversation_requested_page = (
            self._normalize_library_conversation_page(page)
        )
        self._library_conversation_requested_query = normalized_query
        if refocus_filter:
            focus_after_apply = "#library-conversations-filter"
        self._library_conversation_focus_after_apply = focus_after_apply
        self._library_conversation_loading = True
        self._library_conversation_error = ""
        had_selection = (
            self._library_conversations_select_mode
            or self._library_conversations_row_selection.count > 0
        )
        self._library_conversation_selection_notice = (
            "Selection cleared." if had_selection else ""
        )
        self._library_conversations_select_mode = False
        self._library_conversations_row_selection.clear()
        self._library_conversation_reader_state = project_conversation_multiselect(
            self._library_conversation_reader_state,
            active=False,
            selected_count=0,
            loaded_preview_selected=None,
        )
        self._sync_library_conversation_reader()
        self._sync_library_conversation_canvas()
        if refocus_filter:
            self._refocus_library_conversations_filter_after_sync()
        return normalized_query, generation

    def _library_conversation_page_needs_recovery(self) -> bool:
        """Return whether Conversation entry needs its requested scope refetched."""

        return not self._library_conversation_loading and bool(
            not self._library_conversation_page_loaded
            or self._library_conversation_freshness != "fresh"
            or self._library_conversation_error
        )

    def _finish_library_conversation_request_focus(self) -> None:
        """Restore captured pager/filter focus after Conversation recomposition."""

        requested = self._library_conversation_focus_after_apply
        self._library_conversation_focus_after_apply = ""
        if not requested:
            return
        selectors = [requested]
        if requested == "#library-conversations-next":
            selectors.extend(
                ("#library-conversations-previous", "#library-conversations-filter")
            )
        elif requested == "#library-conversations-previous":
            selectors.extend(
                ("#library-conversations-next", "#library-conversations-filter")
            )
        elif requested == "#library-conversations-retry":
            selectors.append("#library-conversations-filter")
        for selector in selectors:
            matches = self.query(selector)
            if not matches:
                continue
            target = matches.first()
            if getattr(target, "disabled", False):
                continue
            target.focus()
            return

    def _finish_library_conversation_page_apply(self) -> None:
        """Finish list focus and align the permanent work selection."""
        self._finish_library_conversation_request_focus()
        self._ensure_library_conversation_reader_selection()

    def _fail_library_conversation_request(
        self,
        requested_page: int,
        requested_query: str,
        generation: int,
        *,
        copy: str = "",
    ) -> None:
        """Retain last-good applied state and publish retryable failure copy."""

        if generation != self._library_conversation_request_generation:
            return
        self._library_conversation_loading = False
        if self._library_conversation_freshness == "stale":
            self._library_conversation_error = ""
            if copy:
                self._library_conversation_stale_copy = copy
        elif copy:
            self._library_conversation_error = copy
        elif self._library_conversation_freshness == "uninitialized":
            self._library_conversation_error = "Couldn't load conversations. Try again."
        elif requested_query != self._library_conversation_query:
            self._library_conversation_error = (
                "Filter wasn’t applied; showing previous results."
            )
        else:
            self._library_conversation_error = f"Couldn't load page {requested_page}."
        self._sync_library_conversation_canvas(
            then=self._finish_library_conversation_request_focus
        )

    @staticmethod
    def _conversation_out_of_range_total(
        result: object, *, requested_offset: int
    ) -> int | None:
        """Return the coherent total proving an ordinary page is out of range."""

        if not isinstance(result, Mapping):
            return None
        items = result.get("items")
        pagination = result.get("pagination")
        if not isinstance(items, list) or items:
            return None
        if not isinstance(pagination, Mapping):
            return None
        limit = pagination.get("limit")
        offset = pagination.get("offset")
        total = pagination.get("total")
        has_more = pagination.get("has_more")
        coordinates = (limit, offset, total)
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in coordinates
        ):
            return None
        if (
            limit != LIBRARY_CONVERSATION_PAGE_SIZE
            or offset != requested_offset
            or total < 0
            or requested_offset == 0
            or requested_offset < total
            or has_more is not False
        ):
            return None
        return total

    def _library_conversation_absence_fence_is_current(
        self,
        *,
        conversation_id: str,
        version: int | None,
        reader_generation: int,
        page_generation: int,
    ) -> bool:
        """Fence an exact-ID absence probe to its mounted reader epoch."""
        state = self._library_conversation_reader_state
        return (
            self._library_conversation_reader_mounted_authority
            and self._library_selected_row_id == LIBRARY_ROW_BROWSE_CONVERSATIONS
            and page_generation == self._library_conversation_request_generation
            and state.selected_id == conversation_id
            and state.selected_version == version
            and state.generation == reader_generation
        )

    async def _confirm_library_conversation_page_absence(
        self,
        *,
        conversation_id: str,
        version: int | None,
        reader_generation: int,
        page_generation: int,
    ) -> Literal["exists", "deleted", "unknown", "stale"]:
        """Confirm a missing page row through the existing exact-ID locator."""
        service = getattr(self.app_instance, "chat_conversation_scope_service", None)
        locate_page = getattr(service, "locate_conversation_page", None)
        if not callable(locate_page):
            return "unknown"
        try:
            located = await self._run_library_service_call(
                locate_page,
                conversation_id,
                mode="local",
                scope_type="all",
                limit=LIBRARY_CONVERSATION_PAGE_SIZE,
            )
        except Exception:
            logger.warning("Failed to confirm missing Library conversation.")
            located = Ellipsis
        if not self._library_conversation_absence_fence_is_current(
            conversation_id=conversation_id,
            version=version,
            reader_generation=reader_generation,
            page_generation=page_generation,
        ):
            return "stale"
        if located is None:
            return "deleted"
        if located is Ellipsis:
            return "unknown"
        try:
            self._validate_library_conversation_locator(located, conversation_id)
        except (TypeError, ValueError):
            return "unknown"
        return "exists"

    async def _load_library_conversation_page(
        self,
        page: int,
        query: str,
        generation: int,
        *,
        _clamp_attempted: bool = False,
    ) -> None:
        """Load a complete service-backed page, discarding stale results."""
        service = getattr(self.app_instance, "chat_conversation_scope_service", None)
        list_conversations = getattr(service, "list_conversations", None)
        if not callable(list_conversations):
            self._fail_library_conversation_request(page, query, generation)
            return

        requested_page = self._normalize_library_conversation_page(page)
        normalized_query = self._safe_text(query, max_length=200)
        requested_offset = (requested_page - 1) * LIBRARY_CONVERSATION_PAGE_SIZE
        try:
            result = await self._run_library_service_call(
                list_conversations,
                mode="local",
                scope_type="all",
                query=normalized_query or None,
                limit=LIBRARY_CONVERSATION_PAGE_SIZE,
                offset=requested_offset,
            )
        except Exception:
            if generation != self._library_conversation_request_generation:
                return
            logger.warning("Failed to load Library conversations page.")
            self._fail_library_conversation_request(
                requested_page, normalized_query, generation
            )
            return

        if generation != self._library_conversation_request_generation:
            return

        out_of_range_total = self._conversation_out_of_range_total(
            result,
            requested_offset=requested_offset,
        )
        if out_of_range_total is not None:
            page_count = max(
                1,
                (out_of_range_total + LIBRARY_CONVERSATION_PAGE_SIZE - 1)
                // LIBRARY_CONVERSATION_PAGE_SIZE,
            )
            if _clamp_attempted:
                self._library_conversation_loading = False
                self._library_conversation_freshness = "stale"
                self._library_conversation_total_known = False
                self._library_conversation_error = ""
                self._library_conversation_stale_copy = (
                    "Source changed again; try again."
                )
                self._sync_library_conversation_canvas(
                    then=self._finish_library_conversation_request_focus
                )
                return
            self._library_conversation_requested_page = page_count
            await self._load_library_conversation_page(
                page_count,
                normalized_query,
                generation,
                _clamp_attempted=True,
            )
            return

        try:
            validated = validate_library_conversation_page(
                result,
                requested_limit=LIBRARY_CONVERSATION_PAGE_SIZE,
                requested_offset=requested_offset,
            )
        except (TypeError, ValueError):
            self._fail_library_conversation_request(
                requested_page, normalized_query, generation
            )
            return

        if generation != self._library_conversation_request_generation:
            return

        reader_state = self._library_conversation_reader_state
        previous_ids = {
            str(record.get("id") or "")
            for record in self._library_conversation_page_records
        }
        refreshed_ids = {str(record.get("id") or "") for record in validated.items}
        selected_id = reader_state.selected_id or ""
        needs_exact_confirmation = (
            self._library_conversation_page_loaded
            and requested_page == self._library_conversation_page
            and normalized_query == self._library_conversation_query
            and selected_id in previous_ids
            and selected_id not in refreshed_ids
        )
        absence_result: Literal["exists", "deleted", "unknown", "stale"] | None = None
        if needs_exact_confirmation:
            absence_result = await self._confirm_library_conversation_page_absence(
                conversation_id=selected_id,
                version=reader_state.selected_version,
                reader_generation=reader_state.generation,
                page_generation=generation,
            )
            if absence_result == "stale":
                return
        if absence_result == "deleted":
            deleted_state = confirm_conversation_deleted(
                self._library_conversation_reader_state,
                selected_id,
                version=reader_state.selected_version,
                generation=reader_state.generation,
            )
            if deleted_state is not reader_state:
                self._library_conversation_reader_state = deleted_state
                self._library_conversation_deleted_selection_id = selected_id
                self._selected_conversation_id = ""
                self._sync_library_conversation_reader()

        self._library_conversation_page_records = validated.items
        self._library_conversation_page = requested_page
        self._library_conversation_total = validated.total
        self._library_conversation_total_known = True
        self._library_conversation_has_more = validated.has_more
        self._library_conversation_page_loaded = True
        self._library_conversation_query = normalized_query
        self._library_conversation_requested_page = requested_page
        self._library_conversation_requested_query = normalized_query
        self._library_conversation_freshness = (
            "stale" if absence_result == "unknown" else "fresh"
        )
        self._library_conversation_stale_copy = (
            "Conversation location could not be confirmed; try again."
            if absence_result == "unknown"
            else ""
        )
        self._library_conversation_loading = False
        self._library_conversation_error = ""
        self._adopt_library_conversation_state_selection(
            self._build_library_conversations_state().selected_id
        )
        self._sync_library_conversation_canvas(
            then=self._finish_library_conversation_page_apply
        )

    @on(Button.Pressed, "#library-conversations-select-toggle")
    def handle_library_conversations_select_toggle(self, event: Button.Pressed) -> None:
        """Enter/exit conversations select mode; clears the selection set (both on enter and exit)."""
        event.stop()
        if self._library_conversation_freshness != "fresh":
            return
        self._library_conversations_select_mode = (
            not self._library_conversations_select_mode
        )
        self._library_conversations_row_selection.clear()
        self._library_conversation_reader_state = project_conversation_multiselect(
            self._library_conversation_reader_state,
            active=self._library_conversations_select_mode,
            selected_count=0,
            loaded_preview_selected=(
                self._library_conversation_loaded_preview_selected()
                if self._library_conversations_select_mode
                else None
            ),
        )
        self._sync_library_conversation_reader()
        _sync_library_canvas(self, "conversations")
        if not self._library_conversations_select_mode:
            self._ensure_library_conversation_reader_selection()

    @on(Button.Pressed, "#library-conversations-select-all")
    def handle_library_conversations_select_all(self, event: Button.Pressed) -> None:
        """Select every conversation row currently rendered by the canvas."""
        event.stop()
        if self._library_conversation_freshness != "fresh":
            return
        rows = self._build_library_conversations_state().rows
        self._library_conversations_row_selection.select_all(
            r.conversation_id for r in rows
        )
        self._library_conversation_reader_state = project_conversation_multiselect(
            self._library_conversation_reader_state,
            active=True,
            selected_count=self._library_conversations_row_selection.count,
            loaded_preview_selected=self._library_conversation_loaded_preview_selected(),
        )
        self._sync_library_conversation_reader()
        _sync_library_canvas(self, "conversations")

    @on(Button.Pressed, "#library-conversations-select-clear")
    def handle_library_conversations_select_clear(self, event: Button.Pressed) -> None:
        """Clear the current conversations selection without leaving select mode."""
        event.stop()
        if self._library_conversation_freshness != "fresh":
            return
        self._library_conversations_row_selection.clear()
        self._library_conversation_reader_state = project_conversation_multiselect(
            self._library_conversation_reader_state,
            active=True,
            selected_count=0,
            loaded_preview_selected=self._library_conversation_loaded_preview_selected(),
        )
        self._sync_library_conversation_reader()
        _sync_library_canvas(self, "conversations")

    @on(Button.Pressed, "#library-conversations-export")
    async def handle_library_conversations_export(self, event: Button.Pressed) -> None:
        """Open the export canvas scoped to Conversations.

        Args:
            event: Button press event emitted by the conversations
                canvas's "Export…" action.
        """
        event.stop()
        if self._library_conversation_freshness != "fresh":
            return
        await self._open_library_export_canvas(ExportScope(kind="conversations"))

    @on(Input.Submitted, "#library-conversations-filter")
    def handle_library_conversations_filter_submitted(
        self, event: Input.Submitted
    ) -> None:
        """Search all conversations from the in-canvas filter box.

        Args:
            event: Input submit event emitted by the conversations canvas's
                filter box.
        """
        event.stop()
        query = self._safe_text(event.value, max_length=200)
        self._start_library_conversation_page_request(1, query, refocus_filter=True)

    @on(Button.Pressed, "#library-conversations-retry")
    def handle_library_conversations_retry(self, event: Button.Pressed) -> None:
        """Retry the most recently requested Conversation scope."""

        event.stop()
        if self._library_conversation_loading:
            return
        pending = self._pending_library_source_open
        if pending is not None and pending[0] == "conversations":
            self.run_worker(
                self._retry_pending_library_conversation_open(pending),
                exclusive=True,
                group="library_nav_open_source",
            )
            return
        self._start_library_conversation_page_request(
            self._library_conversation_requested_page,
            self._library_conversation_requested_query,
            focus_after_apply="#library-conversations-retry",
        )

    async def _retry_pending_library_conversation_open(
        self, pending: tuple[str, str]
    ) -> None:
        """Retry one retained Conversation locator intent."""

        if self._pending_library_source_open != pending:
            return
        await self._open_library_item_by_id(*pending)

    @on(Button.Pressed, "#library-conversations-previous")
    def handle_library_conversations_previous(self, event: Button.Pressed) -> None:
        """Load the preceding complete conversation page.

        Args:
            event: Button press emitted by the conversations pager.
        """
        event.stop()
        if (
            self._library_conversation_loading
            or self._library_conversation_freshness != "fresh"
            or self._library_conversation_page <= 1
        ):
            return
        self._start_library_conversation_page_request(
            self._library_conversation_page - 1,
            self._library_conversation_query,
            focus_after_apply="#library-conversations-previous",
        )

    @on(Button.Pressed, "#library-conversations-next")
    def handle_library_conversations_next(self, event: Button.Pressed) -> None:
        """Load the following complete conversation page.

        Args:
            event: Button press emitted by the conversations pager.
        """
        event.stop()
        if (
            self._library_conversation_loading
            or self._library_conversation_freshness != "fresh"
            or not self._library_conversation_has_more
        ):
            return
        self._start_library_conversation_page_request(
            self._library_conversation_page + 1,
            self._library_conversation_query,
            focus_after_apply="#library-conversations-next",
        )

    def _focus_library_conversations_filter(self) -> None:
        """Re-focus the conversations filter box after a submit-triggered recompose.

        Mirrors ``_focus_library_search_input``: the Submitted-driven
        recompose remounts a brand-new ``#library-conversations-filter``;
        without this, focus silently falls back to the screen after every
        filter submit.
        """
        try:
            self.query_one("#library-conversations-filter", Input).focus()
        except (NoMatches, QueryError):
            pass

    def _refocus_library_conversations_filter_after_sync(self) -> None:
        """Focus the remounted filter after its canvas-scoped recompose."""
        try:
            canvas = self.query_one(
                "#library-conversations-canvas", LibraryConversationsCanvas
            )
        except (NoMatches, QueryError):
            self.call_after_refresh(self._focus_library_conversations_filter)
            return
        canvas.call_after_refresh(self._focus_library_conversations_filter)

    def _notify_library_conversation_unavailable(self) -> None:
        """Preserve the existing deep-link warning for an unavailable target."""

        notify = getattr(self.app_instance, "notify", None)
        if callable(notify):
            notify("Conversation is unavailable.", severity="warning")

    @staticmethod
    def _validate_library_conversation_locator(
        response: object,
        conversation_id: str,
    ) -> tuple[tuple[Mapping[str, Any], ...], int, int, bool]:
        """Validate the bounded rank-derived Conversation locator envelope."""

        if not isinstance(response, Mapping):
            raise ValueError("Conversation locator response must be a mapping.")
        pagination = response.get("pagination")
        if not isinstance(pagination, Mapping):
            raise ValueError("Conversation locator pagination is required.")
        limit = pagination.get("limit")
        offset = pagination.get("offset")
        page = pagination.get("page")
        total = pagination.get("total")
        target_index = pagination.get("target_index")
        coordinates = (limit, offset, page, total, target_index)
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in coordinates
        ):
            raise ValueError("Conversation locator coordinates must be integers.")
        if limit != LIBRARY_CONVERSATION_PAGE_SIZE or target_index < 0:
            raise ValueError("Conversation locator coordinates are invalid.")
        expected_offset = (target_index // limit) * limit
        if (
            offset != expected_offset
            or page != expected_offset // limit + 1
            or target_index >= total
        ):
            raise ValueError("Conversation locator owner page is invalid.")
        validated = validate_library_conversation_page(
            response,
            requested_limit=limit,
            requested_offset=offset,
        )
        local_index = target_index - offset
        if (
            local_index < 0
            or local_index >= len(validated.items)
            or str(
                validated.items[local_index].get("id")
                or validated.items[local_index].get("conversation_id")
                or validated.items[local_index].get("uuid")
                or ""
            ).strip()
            != conversation_id
        ):
            raise ValueError("Conversation locator target position is invalid.")
        return validated.items, page, validated.total, validated.has_more

    def _open_selected_conversation_handoff(self) -> None:
        if self._library_conversation_freshness != "fresh":
            return
        if not self._library_conversation_reader_state.loaded_actions_eligible:
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify(
                    "Wait for the selected conversation to finish loading.",
                    severity="warning",
                )
            return
        workspace_state = self._library_workspace_depth_state()
        payload = self._selected_conversation_handoff_payload()
        notify = getattr(self.app_instance, "notify", None)
        if payload is None:
            if callable(notify):
                notify(
                    "Select a conversation before using it in Console.",
                    severity="warning",
                )
            return
        # Single-item action: gate on THIS conversation's own workspace
        # eligibility, not the aggregate blocked-count — one foreign item
        # elsewhere in the Library must not veto an eligible conversation
        # (TASK-15423). Bulk staging keeps the aggregate gate.
        item_eligible, item_reason = library_item_context_handoff(
            workspace_state,
            item_type=payload.item_type,
            item_id=str(payload.source_id or ""),
        )
        if not item_eligible:
            if callable(notify):
                notify(item_reason, severity="warning")
            return
        open_chat_with_handoff = getattr(
            self.app_instance, "open_chat_with_handoff", None
        )
        if not callable(open_chat_with_handoff):
            if callable(notify):
                notify(
                    "Console handoff is unavailable for Library Conversations.",
                    severity="warning",
                )
            return
        open_chat_with_handoff(payload, action_label="Use in Console")

    @on(Button.Pressed, "#library-conversation-open-console")
    def open_selected_conversation_in_console(self, event: Button.Pressed) -> None:
        event.stop()
        self._open_selected_conversation_handoff()

    @on(Button.Pressed, "#library-conversation-use-source")
    def use_selected_conversation_as_source(self, event: Button.Pressed) -> None:
        event.stop()
        self._open_selected_conversation_handoff()

# --- BEGIN generated conversations-state shims (delete wholesale at cleanup) ---
# task 8: exposes every `LibraryConversationsState` field under its original
# `_library_conversation*`/`_library_conversations_*` name on THIS
# controller too, reading/writing through the injected
# `conversations_state_accessor` instead of a direct
# `self._conversations_state` attribute (this class has none) -- same
# generator shape as the shim block `LibraryScreen` carries (task 6) and
# `LibraryConversationReaderController` carries (task 7), attached
# programmatically so the class body gains no `FunctionDef`s (the size
# ratchet counts those). `CONVERSATIONS_PLURAL_STATE_FIELDS` is imported
# from `library_conversations_state` -- the dataclass's own module -- so
# this is not a third independent literal copy of the plural/singular
# prefix split; see the module docstring's drift-risk note.
for _cc_field in dataclasses.fields(LibraryConversationsState):
    _cc_prefix = (
        "_library_conversations_"
        if _cc_field.name in CONVERSATIONS_PLURAL_STATE_FIELDS
        else "_library_conversation_"
    )
    setattr(
        LibraryConversationsController,
        _cc_prefix + _cc_field.name,
        property(
            lambda self, _n=_cc_field.name: getattr(
                self._conversations_state_accessor(), _n
            ),
            lambda self, value, _n=_cc_field.name: setattr(
                self._conversations_state_accessor(), _n, value
            ),
        ),
    )
del _cc_field, _cc_prefix
# --- END generated conversations-state shims ---

"""Library combined Search+RAG controller.

Controller PR of the combined Search+RAG extraction series (wave-3 task 3 of
``.superpowers/sdd/2026-09-03-library-decomposition-wave3-search-rag``;
search+RAG series 2/3; recipe: ``backlog/docs/library-decomposition-recipe.md``;
export/collections controllers -- ``library_export_controller.py``,
``library_collections_controller.py`` -- are the templates this mirrors
byte-for-byte in shape). Owns the entire top-search-bar + Search/RAG canvas
cluster: rail-search-box tracking/submit, query history (load/record/persist/
clear/rerun), mode/scope toggles, the retrieval run (everything up to and
including the ``@work`` boundary), the two-phase answer pipeline's non-worker
half (start/apply/chat-kwargs), evidence selection/open/console-handoff
(mouse AND keyboard paths), and every panel-region incremental-refresh
builder except the one lock-serialized top-level orchestrator excluded below.
``LibraryScreen`` keeps one-line delegators under every one of these
original names.

**Cluster derivation -- ownership.** Task 2 (state PR, this series) re-derived
the combined candidate set fresh (not carried over from wave-2): an ``ast``
scan of ``LibraryScreen`` for method names containing ``"search"`` (24 raw
matches) or ``"rag"`` (39 raw matches), unioned minus the 3-name overlap, is
**60** candidates. Reading each of the 60 bodies (not trusting the substring
match) finds **3 Prompts-owned** (``_flush_library_prompts_search``,
``_queue_library_prompts_search``, ``_stop_library_prompts_search_debounce``
-- an unrelated Prompts-search-box feature) and **7 Media-owned**
(``_focus_library_media_content_search_input``,
``_reset_library_media_search_on_mode_change``,
``handle_library_media_content_search_{next,prev,submitted}``,
``handle_library_media_trash_search_{changed,submitted}`` -- the Media
canvas's own content/trash search boxes), leaving **50** genuinely
combined-cluster candidates. This task re-ran that same census fresh at its
own execution time (recipe's "never trust a carried-over count" rule, §6)
and reconfirmed the identical 50.

**Single vs. split controller: single, reconfirmed at the METHOD level.**
Task 2's field-level census already found all 20 state fields consumed
inside one lock-serialized call graph rooted at
``_refresh_search_rag_panel_state_widgets``/``_library_rag_panel_state``.
This task re-verified the same conclusion independently, at the METHOD level
the task brief asked for, with an ``ast`` call-graph walk of all 50
candidates' bodies (every ``self.<name>(...)`` call where ``<name>`` is
itself another candidate): the two "search"-prefixed and "rag"-prefixed
naming families call directly into each other throughout, not just at one
seam -- ``handle_library_search_submitted``/``rerun_library_search_from_
history``/``submit_library_rag_query``/``run_library_rag_query`` (all
"search"/generic-named entry points) call ``_start_library_rag_query``
directly; ``_start_library_rag_query`` calls ``_record_library_search_
history`` (a "search"-named method) as an ordinary step in its own body;
``clear_library_search_history`` calls both ``_library_rag_panel_state`` and
``_refresh_library_rag_history_widget`` (both "rag"-named). There is no
subset of the 50 that only ever calls within itself -- any attempted split
would cut through calls that exist today, not around a natural seam. The
plan's own hypothetical clean seam ("rag-answer pipeline vs. search/history
surface") does not hold either: ``_sync_library_rag_scope_toggle_and_run_
gate_widgets`` (a scope/run-gate method) schedules
``_mirror_library_rag_scope_recovery`` as a worker, and the answer pipeline's
own ``_apply_library_rag_search_outcome`` calls ``_start_library_rag_answer``
unconditionally as one step of the SAME outcome-application sequence that
also updates retrieval state and history. **Decision: ONE combined
``LibraryRagSearchController``**, confirmed independently by two different
methods (Task 2's field census, this task's method call-graph) rather than
carried forward on the strength of either alone, per the task brief's own
instruction to re-verify rather than lean on the field-level finding.

**Exclusions -- 8 of the 50, not moved:**

1. **3 ``@work``-decorated methods** (export series' "framework-decorator
   self-type assertion" hazard -- Textual's ``@work`` closure asserts
   ``isinstance(self, DOMNode)`` at CALL time, which a plain controller
   instance would fail): ``_execute_library_rag_answer``
   (``@work(exclusive=True, group='library_rag_answer')``),
   ``_execute_library_rag_search``
   (``@work(exclusive=True, group='library_rag_search')``),
   ``_save_library_search_history`` (``@work(thread=True)``). Confirmed by
   reading every one of the 50 candidates' decorator lists -- no other
   candidate carries ``@work`` or any other self-type-asserting decorator.
2. **1 module-globals-coupling exclusion (recipe §3's SECOND documented
   bypass shape, distinct from instance-attribute monkeypatching) --
   found the hard way, by running the battery, not by the census below**:
   ``_load_library_search_history`` reads the bare name ``get_cli_setting``
   (a plain module-level import, not ``self.get_cli_setting``). Python
   resolves a free name against the DEFINING module's ``__globals__``,
   fixed at definition time -- moving this body to
   ``library_rag_search_controller.py`` (which carries its own independent
   ``from ...config import get_cli_setting``) would silently repoint every
   ``monkeypatch.setattr(library_screen_module, "get_cli_setting", ...)``
   test away from this method's own call, while leaving OTHER
   ``get_cli_setting`` call sites still on ``LibraryScreen`` (ingest
   options, rail-state) unaffected by the same patch -- exactly the
   "module-globals coupling" shape recipe §3 describes for
   ``_read_library_ingest_options_from_config``. Confirmed as a REAL
   regression, not a theoretical one: the full ``-k "(search or rag) and
   library"`` sweep (§ battery) failed
   ``test_library_shell_search_history_loads_from_cli_config_fallback``
   (``screen._library_search_history == ()`` instead of the expected
   fallback tuple) the one time this method was moved during this task's
   own execution -- a blanket per-test isolation fixture
   (``test_library_shell.py:1015``, "tests that want to exercise the
   CLI-config fallback itself re-patch ``library_screen_module.
   get_cli_setting`` after this fixture runs") depends on EVERY
   ``get_cli_setting`` call this subsystem makes resolving through
   ``library_screen_module``'s own globals, including this one. Fixed by
   reverting: ``_load_library_search_history`` stays a REAL, full-bodied
   ``LibraryScreen`` method, byte-for-byte as it always was -- not moved,
   not a delegator. Its only two callers are ``LibraryScreen.__init__``
   (the ``LibraryRagSearchState(history=...)`` computed default, task 2)
   and ``_record_library_search_history`` reads it only through the
   controller-owned STATE field it populates, never calls it directly. No
   named dependency binding was needed for this exclusion (no mover calls
   it).

   **A second, currently-latent instance of the SAME shape was found and
   deliberately NOT excluded for it**: ``cycle_library_rag_mode``/
   ``toggle_library_rag_scope_source`` forward ``self`` into the shared
   ``_sync_library_canvas(screen, kind, ...)`` dispatcher as a bare name
   (see binding-kind 1 above) -- also monkeypatched at the
   ``library_screen_module`` level in
   ``Tests/UI/test_library_entry_compose_once.py`` (4 sites, all
   ``Mock(wraps=...)``-style spies). Read each of the 4 patched tests'
   own bodies: none presses the RAG mode-toggle or scope-toggle buttons or
   asserts on a ``"search"``-kind sync call -- every one exercises
   ``landing``/``conversations``/``media`` kinds through screen-resident
   callers (``_apply_local_source_snapshot``, ``_reconcile_library_entry_
   state``), which still resolve ``_sync_library_canvas`` through
   ``library_screen_module``'s own globals correctly (those callers never
   moved). This is the IDENTICAL shape the conversations controller
   ALREADY shipped in wave-2 (its own ``_sync_library_conversation_
   canvas`` forwards ``self`` the same way) with no additional
   accommodation, and the same reasoning applies here: the risk is real in
   principle but unexercised by any current test, confirmed by the full
   3-root sweep (§ battery) finding zero failures attributable to it.
   Recorded here rather than silently accepted -- a future test that DOES
   press the RAG mode toggle while patching ``library_screen_module.
   _sync_library_canvas`` would hit this same bypass, and would need this
   paragraph's context to diagnose quickly.
3. **4 test-bypass exclusions -- the "instance-attribute monkeypatch"
   shape (conversations exemplar precedent, recipe §11 lesson 2), found by
   a repo-wide census (not just the 2 names Task 2's own characterization
   pass had already flagged) of every ``monkeypatch.setattr(screen,
   "<name>", ...)``/``monkeypatch.setattr(LibraryScreen, "<name>", ...)``/
   bare ``screen.<name> = ...`` site across ALL THREE test roots
   (``Tests/UI``, ``Tests/Library``, ``Tests/Live``) for all 50 candidate
   names**:

   - ``_refresh_search_rag_panel_state_widgets`` (Task 2's own forward
     note) -- patched as a full replacement in
     ``Tests/UI/test_product_maturity_gate16_library_search_rag.py:2905``
     (a ``fail_refresh`` that raises if reached, asserting the unmounted
     worker-completion path does NOT call it) and
     ``Tests/UI/test_library_shell.py:7310`` (a counting wrapper asserting
     it is called exactly twice for two rail-search keystrokes) and
     ``:28851`` (a ``Mock(wraps=...)`` asserting ``call_count == 0`` on an
     unrelated snapshot-driven path). Called internally by 5 other movers
     (``_apply_library_rag_answer``, ``_apply_library_rag_search_outcome``,
     ``_select_library_rag_result_by_index``, ``_start_library_rag_query``,
     ``update_library_rag_query``) -- if it moved, those same-controller
     calls would resolve ``self.<name>`` against the CONTROLLER, bypassing
     every one of the 3 SCREEN-instance patches above.
   - ``_patch_sibling_library_search_input`` (Task 2's own forward note)
     -- patched as a bounded wrapper in
     ``Tests/UI/test_library_shell.py:7307``, asserting the exact
     (selector, value) pairs it is called with when
     ``handle_library_search_changed``/``update_library_rag_query`` (both
     movers) fire. Same bypass shape.
   - ``_library_rag_panel_state`` (NEW finding, not in Task 2's narrower
     ``@on``-handler-only characterization scan) -- spied via
     ``_spy_panel_statuses`` in
     ``Tests/UI/test_product_maturity_gate16_library_search_rag.py:3014``
     (used by 2 tests, both asserting a transient ``"answering"`` status
     was actually observed mid-refresh, e.g. ``:3111``/``:3387``). This is
     the single most heavily-depended-on method in the whole cluster --
     called directly by 10 other movers
     (``_apply_library_rag_search_outcome``, ``_open_library_rag_result_
     by_index``, ``_select_library_rag_result_by_index``,
     ``_stage_library_rag_result_in_console``, ``_start_library_rag_
     answer``, ``_start_library_rag_query``, ``_sync_library_rag_scope_
     toggle_and_run_gate_widgets``, ``clear_library_search_history``, plus
     the 2 already-excluded ``_refresh_search_rag_panel_state_widgets``/
     ``_mirror_library_rag_scope_recovery``). Any of those becoming a
     controller-internal ``self._library_rag_panel_state()`` call would
     bypass a screen-instance patch on this name; several of them
     (``_start_library_rag_answer``, ``_apply_library_rag_search_
     outcome``) sit directly on the call path the ``"answering"``-status
     tests exercise (via ``run_library_rag_query`` -> ``_start_library_
     rag_query`` -> the ``@work``-excluded search/answer workers ->
     ``_apply_library_rag_search_outcome``/``_start_library_rag_answer``).
   - ``_mirror_library_rag_scope_recovery`` (NEW finding) -- spied via
     ``Mock(wraps=...)`` in ``Tests/UI/test_library_shell.py:29076``,
     asserting ``call_count == 0`` on a repeat steady-state snapshot (the
     change-gate's own no-op path). Its ONLY caller is
     ``_sync_library_rag_scope_toggle_and_run_gate_widgets`` (a mover),
     via ``self.run_worker(self._mirror_library_rag_scope_recovery(), ...)``
     -- if both moved to this controller, that call would resolve against
     the CONTROLLER's own method, bypassing the SCREEN-instance spy and
     making the ``call_count == 0`` assertion trivially (and silently)
     true regardless of whether the real logic still behaves correctly.

   Two additional monkeypatch sites the same repo-wide census found are
   confirmed SAFE and need no exclusion, because their only caller stays
   screen-resident (not a moved cluster member) and therefore always
   invokes them with ``self`` = the real, patched screen instance
   regardless of where the underlying implementation lives:
   ``_focused_library_rag_result_card_index`` (patched in
   ``Tests/UI/test_screen_navigation.py:2430``, but its only caller,
   ``LibraryScreen.check_action``, is not a cluster member) and
   ``_sync_library_rag_scope_toggle_and_run_gate_widgets`` itself (``Mock``
   -wrapped in ``test_library_shell.py:28843``, but its only external
   caller, ``_apply_local_source_snapshot``, is one of recipe §3's four
   permanently screen-routed names) -- MOVE normally.
   ``_apply_library_rag_search_outcome`` is also patched (instance- and
   class-level, ``test_library_shell.py:7407``/``:7556``) but every site
   uses a REAL, fully-constructed ``LibraryScreen`` (never an unbound fake),
   so an unbound class-level call (``real_apply(screen, request, outcome)``)
   still resolves correctly through this method's own screen-side
   delegator -- MOVES normally; its own internal calls to the two excluded
   builder methods above are now safe by construction (bound as named
   dependencies, §-canon below).

**42 of the 50 candidates move onto this controller.**

**Byte-for-byte canon** (moved bodies never edited -- every name they
reference that is not this controller's own state is rebound under the SAME
name, per the two binding kinds; see ``ConsoleDictationController.__init__``,
``tldw_chatbook/UI/Console_Modules/dictation.py``, and
``LibraryCollectionsController.__init__`` for the sibling worked examples):

1. **Framework services** (``app_instance``, ``app``, ``call_after_
   refresh``, ``focused``, ``is_mounted``, ``is_running``, ``query``,
   ``query_one``, ``refresh``, ``run_worker``) are live-read from the screen
   via ``@property`` on every access -- never snapshotted. ``app``/
   ``is_running``/``refresh`` exist ONLY because ``cycle_library_rag_mode``/
   ``toggle_library_rag_scope_source`` forward ``self`` verbatim into the
   shared, multi-subsystem ``_sync_library_canvas(screen, kind, ...)``
   dispatcher (``canvas_sync.py``) -- the SAME real bug/fix shape the
   conversations controller's own ``app``/``is_running``/``refresh``
   properties document (its own docstring: "a real bug this task's
   characterization run caught"). Read that dispatcher's actual body (not
   just its signature) before assuming a controller exposes everything it
   touches: for ``kind == "search"`` plus the function's own
   kind-independent top/exception-handler code, it reads/writes
   ``query_one``, ``is_running``, ``app.capture_mouse(None)``, ``refresh``,
   ``call_after_refresh``, the state field ``_library_rag_answer_render_
   key`` (already covered by the generated state shim below), the excluded
   ``_library_rag_panel_state`` (bound as a named dependency, item 2 group
   (d) below), and two SHARED shell fields this cluster does not otherwise
   touch at all -- ``_library_canvas_projection_depth``/``_library_canvas_
   resync_pending`` (group (b) below).
2. **Everything else** the cluster depends on that is not its own state is a
   NAMED constructor dependency: (a) six general Library-wide shell helpers
   a moved body calls with explicit arguments (``_active_library_rail``,
   ``_console_setup_would_block``, ``_open_library_item_by_id``,
   ``_safe_text``, ``_select_library_rail_row``, ``_trailing_index`` -- the
   last two are ``@staticmethod``s shared with/exclusively used by this
   cluster today but never censused as cluster-owned by the name-matching
   script above, so they stay screen-resident per the same "shell helper,
   not cluster member" treatment ``library_adaptive_reader_allocation_is_
   current`` got in the collections controller); (b) one piece of shared
   shell state this cluster only READS
   (``_library_selected_row_id``) and one pair this cluster reads AND
   writes only via the ``_sync_library_canvas`` forwarding described above
   (``_library_canvas_projection_depth`` read-only,
   ``_library_canvas_resync_pending`` get+set -- mirroring the
   conversations controller's own identical pair, group (d) there); (c) the
   3 ``@work``-decorated methods excluded above, reached via named
   late-binding callables exactly like the export series' own ``@work``
   exclusions (``_execute_library_rag_answer``, ``_execute_library_rag_
   search``, ``_save_library_search_history``); (d) the 4 test-bypass
   exclusions above, ALSO reached via named late-binding callables --
   ``_library_rag_panel_state``, ``_mirror_library_rag_scope_recovery``,
   ``_patch_sibling_library_search_input``,
   ``_refresh_search_rag_panel_state_widgets`` -- which is exactly why a
   test's ``monkeypatch.setattr(screen, "<name>", ...)`` keeps working
   after this move: each dependency is a ``lambda`` that re-reads
   ``screen.<name>`` on every invocation, at CALL time, not a value
   captured once at construction.

**Construction order -- the usual position.** `LibraryScreen.__init__`
builds `self._rag_search_controller` right after `self._collections_
controller`, matching every other controller in this file. An EARLIER
draft of this task tried building it before `self._rag_search_state`
instead, on the mistaken premise that `_load_library_search_history`
(called eagerly, one line below the state's own construction, to compute
`LibraryRagSearchState`'s computed `history=` default) needed the
controller to already exist. It does not: that exclusion (module-globals
coupling, above) means `_load_library_search_history` stays a REAL screen
method, never a delegator, so `LibraryScreen.__init__`'s eager call
resolves directly against `self` with no controller involved at all --
the standard construction position was always safe once that method was
excluded correctly.

This subsystem's OWN state (every ``_library_rag_<field>``/``_library_
search_history`` name the moved bodies reference) is exposed through
generated properties reading ``self._rag_search_state_accessor().<field>``
-- the same two-prefix generator shape ``LibraryScreen`` carries (task 2)
and the conversations controller's own ``_library_conversation*``/
``_library_conversations_*`` split mirrors exactly (recipe §11's
``CONVERSATIONS_PLURAL_STATE_FIELDS`` drift lesson, applied here from the
start): ``SEARCH_PREFIXED_STATE_FIELDS`` is imported from ``library_rag_
search_state`` -- the dataclass's own module, task 2's single authoritative
home for the one-field (``history``) prefix exception -- rather than
redefined as a second, independently-drifting literal copy.
"""
from __future__ import annotations

import dataclasses
from typing import Any, TYPE_CHECKING

from loguru import logger
from textual import on
from textual.containers import Vertical
from textual.css.query import NoMatches, QueryError
from textual.widgets import Button, Collapsible, Input, Static

from ...Library.library_rag_answer_service import (
    LibraryRagAnswer,
    library_rag_answer_provider_gate,
)
from ...Library.library_rag_service import LibraryRagSearchOutcome, LibraryRagSearchRequest
from ...Library.library_rag_state import (
    LIBRARY_RAG_QUERY_MAX_LENGTH,
    LIBRARY_RAG_SCOPE_TOGGLE_SOURCE_TYPES,
    LIBRARY_RAG_USE_IN_CONSOLE_LOCKED_NOTICE,
    LibraryRagPanelState,
    library_rag_scope_summary,
    update_search_history,
)
from ...Library.library_shell_state import LIBRARY_ROW_BROWSE_SEARCH, LIBRARY_ROW_INGEST_MEDIA
from ...Widgets.Library import (
    LibrarySearchRagPanel,
    library_rag_answer_children,
    library_rag_history_children,
    library_rag_query_quiet_text,
    library_rag_query_shows_full_recovery,
    library_rag_query_status_children,
    library_rag_results_body_children,
    library_rag_scope_recovery_children,
    library_rag_scope_shows_recovery,
    results_heading_text,
    scope_toggle_label,
)
from ..Views.RAGSearch.search_handoff import build_library_rag_console_live_work_payload
from .canvas_sync import _sync_library_canvas
from .library_rag_search_state import SEARCH_PREFIXED_STATE_FIELDS, LibraryRagSearchState
from .screen_constants import (
    LIBRARY_RAG_ANSWERABLE_RETRIEVAL_STATUSES,
    LIBRARY_RAG_RESULTS_STATIC_WIDGET_IDS,
)

if TYPE_CHECKING:
    from ..Screens.library_screen import LibraryScreen


class LibraryRagSearchController:
    """Owns the combined top-search-bar + Search/RAG canvas cluster (43 methods).

    Holds no state of its own beyond what it reads and writes through
    ``LibraryRagSearchState`` (via the injected accessor) and the shared
    shell/framework bindings below. ``LibraryScreen`` constructs exactly one
    of these, in ``__init__`` right after ``self._collections_controller``,
    and keeps one-line delegators for every original name this cluster
    moved (43 -- see the module docstring for the full derivation and the 7
    exclusions).
    """

    def __init__(
        self,
        screen: "LibraryScreen",
        *,
        rag_search_state_accessor,
        # -- general Library-wide shell helpers, not moved (shared with/
        # incidentally exclusive to this cluster today; see module
        # docstring group (a)).
        active_library_rail,
        console_setup_would_block,
        open_library_item_by_id,
        safe_text,
        select_library_rail_row,
        trailing_index,
        # -- shared shell state this cluster reads (group (b)); the
        # canvas-projection pair exists only for the `_sync_library_canvas`
        # forwarding described in the module docstring.
        library_selected_row_id_accessor,
        library_canvas_projection_depth_accessor,
        library_canvas_resync_pending_accessor,
        set_library_canvas_resync_pending,
        # -- the 3 `@work`-decorated methods excluded from this move
        # (group (c)).
        execute_library_rag_answer,
        execute_library_rag_search,
        save_library_search_history,
        # -- the 4 test-bypass exclusions (group (d)).
        library_rag_panel_state,
        mirror_library_rag_scope_recovery,
        patch_sibling_library_search_input,
        refresh_search_rag_panel_state_widgets,
    ) -> None:
        """Build the controller and bind everything its moved bodies need.

        Every one of the 43 method bodies below is a byte-for-byte copy of
        the pre-extraction ``LibraryScreen`` method: no internal line was
        edited to retarget a call or an attribute. That is possible because
        this constructor binds every name those bodies reference that is
        not this controller's own state, under the SAME name the original
        method used. See the module docstring for the binding kinds this
        follows.

        Args:
            screen: The Library screen. Used ONLY for the ten framework
                services below (``app_instance``, ``app``, ``call_after_
                refresh``, ``focused``, ``is_mounted``, ``is_running``,
                ``query``, ``query_one``, ``refresh``, ``run_worker``) --
                this cluster owns no DOM of its own.
            rag_search_state_accessor: Returns the live
                ``LibraryRagSearchState`` (``LibraryScreen._rag_search_
                state``, task 2). Backs every generated ``_library_rag_
                <field>``/``_library_search_history`` property below.
            active_library_rail: ``LibraryScreen._active_library_rail`` --
                used by ``_focus_library_search_input`` to re-find the
                mounted rail after a submit-triggered recompose.
            console_setup_would_block: ``LibraryScreen._console_setup_
                would_block`` -- used by ``_stage_library_rag_result_in_
                console`` to decide whether to show the pre-navigation
                Console-locked notice.
            open_library_item_by_id: ``LibraryScreen._open_library_item_by_
                id`` -- used by ``_open_library_rag_result_by_index`` to
                jump an evidence "Open" action straight to its detail
                surface.
            safe_text: ``LibraryScreen._safe_text`` -- used by ``_load_
                library_search_history``/``handle_library_search_
                submitted`` to sanitize history entries/the submitted
                query.
            select_library_rail_row: ``LibraryScreen._select_library_rail_
                row`` -- used by ``handle_library_search_submitted``/
                ``open_import_export_from_library_rag`` to drive the rail
                selection.
            trailing_index: ``LibraryScreen._trailing_index`` (a
                ``@staticmethod``) -- used by 4 evidence-row handlers to
                parse a button id's trailing index.
            library_selected_row_id_accessor: Reads ``LibraryScreen.
                _library_selected_row_id`` -- the recipe's own canonical
                >=2-subsystems shared field (226 refs). Read-only in this
                cluster: confirmed by an AST Store-context check that no
                moved body writes it directly, so no setter is bound.
            library_canvas_projection_depth_accessor: Reads ``LibraryScreen.
                _library_canvas_projection_depth`` -- NOT referenced by any
                moved body directly; needed only because ``cycle_library_
                rag_mode``/``toggle_library_rag_scope_source`` forward
                ``self`` verbatim into ``_sync_library_canvas``, whose own
                body reads this shared shell field via ``getattr(screen,
                ..., 0)`` at its own top/exception-handler code (mirrors the
                conversations controller's identical pair, for the
                identical reason).
            library_canvas_resync_pending_accessor /
            set_library_canvas_resync_pending: Read/write ``LibraryScreen.
                _library_canvas_resync_pending`` -- same ``_sync_library_
                canvas``-forwarding reason as the projection-depth pair
                immediately above; that dispatcher WRITES this field on its
                reentrancy-guard path, so both a getter and a setter are
                bound.
            execute_library_rag_answer: ``LibraryScreen._execute_library_
                rag_answer`` (``@work``-excluded) -- called from ``_start_
                library_rag_answer`` with one positional arg plus keyword
                arguments.
            execute_library_rag_search: ``LibraryScreen._execute_library_
                rag_search`` (``@work``-excluded) -- called from ``_start_
                library_rag_query`` with one positional arg.
            save_library_search_history: ``LibraryScreen._save_library_
                search_history`` (``@work``-excluded) -- called from
                ``_persist_library_search_history`` with one positional
                arg.
            library_rag_panel_state: ``LibraryScreen._library_rag_panel_
                state`` (test-bypass-excluded -- see module docstring) --
                the panel-state builder, called directly by 10 moved
                bodies.
            mirror_library_rag_scope_recovery: ``LibraryScreen._mirror_
                library_rag_scope_recovery`` (test-bypass-excluded) --
                called from ``_sync_library_rag_scope_toggle_and_run_gate_
                widgets`` via ``run_worker``.
            patch_sibling_library_search_input: ``LibraryScreen._patch_
                sibling_library_search_input`` (test-bypass-excluded) --
                called from ``handle_library_search_changed``/``update_
                library_rag_query``.
            refresh_search_rag_panel_state_widgets: ``LibraryScreen._
                refresh_search_rag_panel_state_widgets`` (test-bypass-
                excluded) -- the full-panel refresh orchestrator, called
                from 5 moved bodies.
        """
        self._screen = screen
        self._rag_search_state_accessor = rag_search_state_accessor
        self._active_library_rail_fn = active_library_rail
        self._console_setup_would_block_fn = console_setup_would_block
        self._open_library_item_by_id_fn = open_library_item_by_id
        self._safe_text_fn = safe_text
        self._select_library_rail_row_fn = select_library_rail_row
        self._trailing_index_fn = trailing_index
        self._library_selected_row_id_accessor = library_selected_row_id_accessor
        self._library_canvas_projection_depth_accessor = (
            library_canvas_projection_depth_accessor
        )
        self._library_canvas_resync_pending_accessor = (
            library_canvas_resync_pending_accessor
        )
        self._set_library_canvas_resync_pending_fn = set_library_canvas_resync_pending
        self._execute_library_rag_answer_fn = execute_library_rag_answer
        self._execute_library_rag_search_fn = execute_library_rag_search
        self._save_library_search_history_fn = save_library_search_history
        self._library_rag_panel_state_fn = library_rag_panel_state
        self._mirror_library_rag_scope_recovery_fn = mirror_library_rag_scope_recovery
        self._patch_sibling_library_search_input_fn = patch_sibling_library_search_input
        self._refresh_search_rag_panel_state_widgets_fn = (
            refresh_search_rag_panel_state_widgets
        )

    # -- framework services: live-read properties, never snapshotted -----

    @property
    def app_instance(self) -> Any:
        """This project's screen-level analogue of Textual's own ``self.app``,
        live-read from the screen. See ``__init__``'s docstring."""
        return self._screen.app_instance

    @property
    def app(self) -> Any:
        """``Screen.app``, live-read -- Textual's OWN app property (distinct
        from ``app_instance`` above). Needed ONLY for the ``_sync_library_
        canvas`` forwarding -- see ``__init__``'s docstring."""
        return self._screen.app

    @property
    def call_after_refresh(self) -> Any:
        """``Screen.call_after_refresh``, bound. See ``__init__``'s
        docstring."""
        return self._screen.call_after_refresh

    @property
    def focused(self) -> Any:
        """``Screen.focused``, live-read. See ``__init__``'s docstring."""
        return self._screen.focused

    @property
    def is_mounted(self) -> bool:
        """``Screen.is_mounted``, live-read. See ``__init__``'s docstring."""
        return self._screen.is_mounted

    @property
    def is_running(self) -> bool:
        """``Screen.is_running``, live-read. Needed ONLY for the
        ``_sync_library_canvas`` forwarding -- see ``__init__``'s
        docstring."""
        return self._screen.is_running

    @property
    def query(self) -> Any:
        """``DOMNode.query``, bound. See ``__init__``'s docstring."""
        return self._screen.query

    @property
    def query_one(self) -> Any:
        """``DOMNode.query_one``, bound. See ``__init__``'s docstring."""
        return self._screen.query_one

    @property
    def refresh(self) -> Any:
        """``Screen.refresh``, bound. Needed ONLY for the ``_sync_library_
        canvas`` forwarding's fallback path -- see ``__init__``'s
        docstring."""
        return self._screen.refresh

    @property
    def run_worker(self) -> Any:
        """``Screen.run_worker``, bound. See ``__init__``'s docstring."""
        return self._screen.run_worker

    # -- named constructor dependencies -----------------------------------

    @property
    def _active_library_rail(self) -> Any:
        """The injected ``active_library_rail``. See ``__init__``'s
        docstring."""
        return self._active_library_rail_fn

    @property
    def _console_setup_would_block(self) -> Any:
        """The injected ``console_setup_would_block``. See ``__init__``'s
        docstring."""
        return self._console_setup_would_block_fn

    @property
    def _open_library_item_by_id(self) -> Any:
        """The injected ``open_library_item_by_id``. See ``__init__``'s
        docstring."""
        return self._open_library_item_by_id_fn

    @property
    def _safe_text(self) -> Any:
        """The injected ``safe_text``. See ``__init__``'s docstring."""
        return self._safe_text_fn

    @property
    def _select_library_rail_row(self) -> Any:
        """The injected ``select_library_rail_row``. See ``__init__``'s
        docstring."""
        return self._select_library_rail_row_fn

    @property
    def _trailing_index(self) -> Any:
        """The injected ``trailing_index``. See ``__init__``'s docstring."""
        return self._trailing_index_fn

    @property
    def _library_selected_row_id(self) -> str:
        """Calls the injected ``library_selected_row_id_accessor``.
        Read-only in this cluster (no setter -- see ``__init__``'s
        docstring)."""
        return self._library_selected_row_id_accessor()

    @property
    def _library_canvas_projection_depth(self) -> int:
        """Calls the injected ``library_canvas_projection_depth_accessor``
        (``getattr(screen, "_library_canvas_projection_depth", 0)``); shared
        shell state this cluster reads only via the ``_sync_library_canvas``
        forwarding -- see ``__init__``'s docstring."""
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
    def _execute_library_rag_answer(self) -> Any:
        """The injected ``execute_library_rag_answer`` (``@work``-excluded).
        See ``__init__``'s docstring."""
        return self._execute_library_rag_answer_fn

    @property
    def _execute_library_rag_search(self) -> Any:
        """The injected ``execute_library_rag_search`` (``@work``-excluded).
        See ``__init__``'s docstring."""
        return self._execute_library_rag_search_fn

    @property
    def _save_library_search_history(self) -> Any:
        """The injected ``save_library_search_history`` (``@work``
        -excluded). See ``__init__``'s docstring."""
        return self._save_library_search_history_fn

    @property
    def _library_rag_panel_state(self) -> Any:
        """The injected ``library_rag_panel_state`` (test-bypass-excluded --
        see module docstring). See ``__init__``'s docstring."""
        return self._library_rag_panel_state_fn

    @property
    def _mirror_library_rag_scope_recovery(self) -> Any:
        """The injected ``mirror_library_rag_scope_recovery`` (test-bypass
        -excluded). See ``__init__``'s docstring."""
        return self._mirror_library_rag_scope_recovery_fn

    @property
    def _patch_sibling_library_search_input(self) -> Any:
        """The injected ``patch_sibling_library_search_input`` (test-bypass
        -excluded). See ``__init__``'s docstring."""
        return self._patch_sibling_library_search_input_fn

    @property
    def _refresh_search_rag_panel_state_widgets(self) -> Any:
        """The injected ``refresh_search_rag_panel_state_widgets``
        (test-bypass-excluded). See ``__init__``'s docstring."""
        return self._refresh_search_rag_panel_state_widgets_fn

    # -- moved bodies (byte-for-byte; see module docstring) ---------------

    def _record_library_search_history(self, query: str) -> None:
        """Update in-memory and persisted Library Search/RAG query history."""
        self._library_search_history = update_search_history(
            self._library_search_history, query
        )
        self._persist_library_search_history(list(self._library_search_history))

    def _persist_library_search_history(self, history_list: list[str]) -> None:
        """Write `history_list` into the in-memory config and to disk.

        Shared by `_record_library_search_history` (append a new query) and
        `clear_library_search_history` (D1: empty the list) so both funnel
        through one persistence path.
        """
        app_config = getattr(self.app_instance, "app_config", None)
        if isinstance(app_config, dict):
            library_config = app_config.get("library")
            if not isinstance(library_config, dict):
                library_config = {}
                app_config["library"] = library_config
            search_config = library_config.get("search")
            if not isinstance(search_config, dict):
                search_config = {}
                library_config["search"] = search_config
            search_config["history"] = history_list
        self._save_library_search_history(history_list)

    @on(Input.Changed, "#library-search-input")
    def handle_library_search_changed(self, event: Input.Changed) -> None:
        """Track rail-search text as the user types it (task-2016).

        Without this the screen's ``_library_rag_query`` echo only moved on
        SUBMIT, so every rail rebuild -- and the persisted shell state --
        re-seeded the box from the last submitted query: text the user typed
        (or deleted) without submitting resurrected on the next recompose or
        visit. The mount-echo ``Input.Changed`` Textual fires for the
        ``value=`` kwarg re-announces the same value, so storing it is
        idempotent.
        """
        event.stop()
        self._library_rag_query = event.value
        # task-4023 AC#6 (RC-08): mirror into the Search canvas's query box
        # when it is mounted, so the two visible inputs can never disagree
        # (the state is one; the widgets used to drift until a recompose).
        self._patch_sibling_library_search_input(
            "#library-rag-query-input", event.value
        )

    @on(Input.Submitted, "#library-search-input")
    async def handle_library_search_submitted(self, event: Input.Submitted) -> None:
        """Submit the rail-top query to the Search canvas (fast `search` mode).

        The rail search box is the single query truth for Library
        Search/RAG: submitting it seeds ``_library_rag_query``, selects the
        promoted Search canvas, and (for a non-blank query) runs it through
        the same exclusive-worker gate as the in-panel query box
        (``_start_library_rag_query``). A blank submit still lands on the
        Search canvas -- so a bare Enter always goes somewhere sensible --
        but never invokes the search service.

        Args:
            event: Input submit event emitted by the rail's search box.
        """
        event.stop()
        query = self._safe_text(event.value, max_length=LIBRARY_RAG_QUERY_MAX_LENGTH)
        self._library_rag_query = query
        self._library_rag_mode = "search"
        await self._select_library_rail_row(LIBRARY_ROW_BROWSE_SEARCH)
        if self._library_selected_row_id != LIBRARY_ROW_BROWSE_SEARCH:
            # A dirty note editor sitting in an unresolved save conflict
            # aborts the row switch (`_select_library_rail_row` returns
            # early without moving `_library_selected_row_id`) -- the rail
            # submit must not run a query against a canvas the user never
            # actually reached, and must not record a history entry for it.
            return
        if query.strip():
            await self._start_library_rag_query()
        self.call_after_refresh(self._focus_library_search_input)

    def _focus_library_search_input(self) -> None:
        """Re-focus the search box after a submit-triggered recompose.

        ``handle_library_search_submitted`` rebuilds the whole screen, which
        remounts a brand-new ``#library-search-input``; without this, focus
        silently falls back to the screen after every search.
        """
        rail = self._active_library_rail()
        if rail is None:
            return
        try:
            rail.query_one("#library-search-input", Input).focus()
        except (NoMatches, QueryError):
            pass

    def _library_rail_search_placeholder(self) -> str:
        """Placeholder for the rail search box.

        The rail box always feeds the Search canvas now (see
        ``handle_library_search_submitted``), never a source-specific
        filter, so the placeholder is unconditional regardless of which
        rail row is active.

        Returns:
            The rail search placeholder text.
        """
        return "Search Library…"

    @on(Input.Changed, "#library-rag-query-input")
    async def update_library_rag_query(self, event: Input.Changed) -> None:
        """Refresh the run gate/status region without rebuilding results/history.

        B5 (task-284): unsubmitted query text affects the run gate
        (enabled/disabled, quiet-line/recovery messaging) and the scope
        summary/recovery widgets and inspector -- all cheap, bounded
        refreshes -- but never the Evidence results list or Recent-searches
        history (search runs on Submitted). This used to call the full
        panel refresh unconditionally, which tears down and remounts
        ~100+ widgets (every result row + every history row) on every
        keystroke even though neither depends on the query text.

        Resets only the in-flight/service-reported status
        (`_reset_library_rag_in_flight_status`), not the landed results --
        unlike a full reset, this can't desync the (deliberately untouched)
        results widget from what `_library_rag_results` says it holds, so
        clicking an already-visible evidence row keeps working while the
        user types a new, not-yet-submitted query. Without resetting the
        status at all, a query typed while a prior search is still
        in-flight (or just failed) would leave the run gate stuck showing
        "Searching..."/disabled forever, since that stale request's own
        outcome gets discarded by `_apply_library_rag_search_outcome`'s
        query mismatch guard once it lands. `_start_library_rag_query`
        (Submit/Run) does its own full reset immediately before it
        replaces the results/history widgets.

        Args:
            event: The Input.Changed event carrying the query field's
                current text.
        """
        event.stop()
        if event.value == self._library_rag_query:
            return
        self._library_rag_query = event.value
        # task-4023 AC#6 (RC-08): one query truth at the WIDGET level too.
        # The STATE was already single-source, but the rail box only
        # re-seeds on recompose, so typing here left the mounted rail
        # widget visibly holding the older string (proven live: canvas
        # "terminals render" beside rail "terminals"). Patch the sibling
        # in place; its own Changed handler no-ops (value == state).
        self._patch_sibling_library_search_input("#library-search-input", event.value)
        self._reset_library_rag_in_flight_status()
        await self._refresh_search_rag_panel_state_widgets(
            include_results_and_history=False
        )

    @on(Input.Submitted, "#library-rag-query-input")
    async def submit_library_rag_query(self, event: Input.Submitted) -> None:
        """Run Library Search/RAG from the query field for keyboard-only users."""
        event.stop()
        await self._start_library_rag_query()

    def _reset_library_rag_retrieval_state(self) -> None:
        self._library_rag_results = ()
        self._library_rag_retrieval_status = ""
        self._library_rag_recovery_state = None
        self._library_rag_selected_result_id = ""
        self._library_rag_diagnostics = {}
        self._library_rag_searched_query = ""
        self._reset_library_rag_answer_state()

    def _reset_library_rag_answer_state(self) -> None:
        """Drop the answer and its staleness guards (PR-3 Task 4).

        Called wherever the results an answer was grounded in stop being the
        results on screen -- the mode toggle (via
        `_reset_library_rag_retrieval_state`) and the start of a new search.
        Clearing the two guard fields is what makes an in-flight answer's
        eventual arrival a no-op: `_apply_library_rag_answer` compares the
        request it was generated for against them, so an answer for a query
        (or mode) the panel has moved on from can never overwrite a newer
        one. The in-flight flag goes with them, so no reset path can leave a
        dangling "answering" status behind.
        """
        self._library_rag_answer = None
        self._library_rag_answer_query = ""
        self._library_rag_answer_mode = ""
        self._library_rag_answer_in_flight = False
        self._library_rag_answer_in_flight_provider = ""

    def _reset_library_rag_in_flight_status(self) -> None:
        """Un-stick the run gate without touching landed results (B5/task-284).

        Narrower than `_reset_library_rag_retrieval_state`: clears only the
        service-reported `retrieval_status`/`recovery_state` (the fields
        that can otherwise pin the run gate at "Searching..."/disabled or a
        stale failure/recovery message), leaving `_library_rag_results` and
        `_library_rag_selected_result_id` exactly as they are. Used by the
        query-edit path, which deliberately never touches the results/
        history widgets -- resetting those fields there without also
        rebuilding the widget would desync the two.

        The in-flight ANSWER flag (PR-3 Task 4) is cleared for the same
        reason its retrieval counterpart is: it too disables the Run button
        ("Answering…"), and a provider call has no bounded duration, so a
        user who starts typing a new query must not be locked out of running
        it until some model finishes. The landed `_library_rag_answer` itself
        is deliberately left alone -- it belongs to the results still on
        screen, exactly like them. The generation still running underneath
        keeps its guard fields, so its answer still applies when it lands
        (it answers the query those visible results were retrieved for).
        """
        self._library_rag_retrieval_status = ""
        self._library_rag_recovery_state = None
        self._library_rag_answer_in_flight = False
        self._library_rag_answer_in_flight_provider = ""

    @on(Button.Pressed, "#library-rag-run-query")
    async def run_library_rag_query(self, event: Button.Pressed) -> None:
        event.stop()
        await self._start_library_rag_query()

    @on(Button.Pressed, "#library-rag-open-import-export")
    async def open_import_export_from_library_rag(self, event: Button.Pressed) -> None:
        event.stop()
        # Drive the shell selection so the recomposed canvas resolves to the
        # Ingest canvas. The Import/Export row/mode this used to target is
        # retired -- the Ingest ▸ Import media canvas row is its only
        # surviving successor.
        await self._select_library_rail_row(LIBRARY_ROW_INGEST_MEDIA)

    @on(Button.Pressed, "#library-rag-mode-toggle")
    def cycle_library_rag_mode(self, event: Button.Pressed) -> None:
        """Cycle Library Search/RAG mode between keyword search and RAG answer."""
        event.stop()
        self._library_rag_mode = (
            "rag" if self._library_rag_mode == "search" else "search"
        )
        self._reset_library_rag_retrieval_state()
        _sync_library_canvas(self, "search")

    @on(Button.Pressed, ".library-rag-scope-toggle")
    def toggle_library_rag_scope_source(self, event: Button.Pressed) -> None:
        """Toggle one source type in/out of the Search/RAG retrieval scope (B2).

        Unlike the mode toggle, this does NOT reset in-flight retrieval or
        search history. It used to leave already-landed RESULTS visible too
        ("scope only affects the NEXT run") -- D4/task-5 fixed that:
        `LibraryRagPanelState.from_values` now filters already-landed rows
        against the current scope on every build, so a source toggled off
        hides its rows (and clears a selection pointing at one) in this
        exact recompose, not just the next run. Still a transition (like
        the mode toggle), so the canvas recomposes to pick up the new
        toggle labels, run-gate state, the now-filtered evidence list, and
        (if the scope is now empty) the A1 quiet line.
        """
        event.stop()
        button_id = event.button.id or ""
        source_type = button_id.removeprefix("library-rag-scope-toggle-")
        if source_type not in LIBRARY_RAG_SCOPE_TOGGLE_SOURCE_TYPES:
            return
        if source_type in self._library_rag_scope_deselected:
            self._library_rag_scope_deselected.discard(source_type)
        else:
            self._library_rag_scope_deselected.add(source_type)
        _sync_library_canvas(self, "search")

    @on(Collapsible.Toggled, "#library-rag-history")
    def sync_library_rag_history_collapsed(self, event: Collapsible.Toggled) -> None:
        """Track manual expand/collapse so recomposes preserve the user's choice.

        `Collapsible._watch_collapsed` posts this message whenever the
        `collapsed` reactive changes through the normal watcher path -- in
        practice the user clicking the title. The results-arrival
        force-collapse in `_refresh_library_rag_history_widget` deliberately
        BYPASSES the watcher (`set_reactive` + `_update_collapsed`, task-4023
        AC#6 / RC-08: the watcher's animated `scroll_visible` sailed past the
        Evidence region), so it never reaches this handler; that path keeps
        the field honest itself -- `_apply_library_rag_search_outcome` sets
        `_library_rag_history_collapsed` at the transition.
        """
        event.stop()
        self._library_rag_history_collapsed = event.collapsible.collapsed

    @on(Button.Pressed, "#library-rag-history-clear")
    async def clear_library_search_history(self, event: Button.Pressed) -> None:
        """Clear all Library Search/RAG query history, in memory and on disk (D1)."""
        event.stop()
        self._library_search_history = ()
        self._persist_library_search_history([])
        await self._refresh_library_rag_history_widget(self._library_rag_panel_state())

    @on(Button.Pressed, ".library-rag-history-row")
    async def rerun_library_search_from_history(self, event: Button.Pressed) -> None:
        """Re-run a prior Library Search/RAG query selected from history."""
        event.stop()
        index = self._trailing_index(event.button.id)
        if index is None or index >= len(self._library_search_history):
            return
        query = self._library_search_history[index]
        self._library_rag_query = query
        # Repopulate the visible query input too -- otherwise it keeps
        # whatever text (or blank) it held before the history row was
        # clicked, even though the run underneath used the history entry.
        # Set this before starting the run: `update_library_rag_query`'s
        # `Input.Changed` handler is a no-op once its value already equals
        # `_library_rag_query` (true here), so it can't clobber the new
        # run's "searching" status either way, but setting it first keeps
        # the widget and state in lockstep from the start of the run.
        try:
            self.query_one("#library-rag-query-input", Input).value = query
        except (NoMatches, QueryError):
            pass
        await self._start_library_rag_query()

    async def _start_library_rag_query(self) -> None:
        panel_state = self._library_rag_panel_state()
        run_action = panel_state.query_state.run_action
        if not run_action.enabled:
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify(run_action.disabled_reason, severity="warning")
            return

        request = LibraryRagSearchRequest(
            query=panel_state.query_state.query,
            source_types=panel_state.scope.selected_source_types,
            mode=panel_state.query_state.mode,
            top_k=panel_state.query_state.top_k,
            include_citations=panel_state.query_state.include_citations,
        )
        self._record_library_search_history(request.query)
        self._library_rag_results = ()
        self._library_rag_recovery_state = None
        self._library_rag_selected_result_id = ""
        self._library_rag_retrieval_status = "searching"
        self._library_rag_diagnostics = {}
        self._library_rag_searched_query = ""
        # PR-3 Task 4: a new search invalidates the previous answer AND any
        # generation still running for it -- clearing the guard fields is
        # what makes that older answer's arrival a discarded no-op rather
        # than a stale overwrite (`_apply_library_rag_answer`).
        self._reset_library_rag_answer_state()
        # The rail-top search box can invoke this mid-recompose -- it selects
        # the Search canvas via ``_select_library_rail_row`` and then runs the
        # query immediately after, before the scheduled recompose has mounted
        # ``#library-search-rag-panel``. The widget refresh is only attempted
        # when the panel is actually mounted; when it isn't, skipping is
        # non-fatal because the subsequent recompose renders the same state
        # (the status fields set above already carry it). This is an
        # explicit presence check, not a broad NoMatches/QueryError catch --
        # a prior version wrapped the whole refresh (results rows included)
        # in a blanket ``except (NoMatches, QueryError): pass``, which also
        # silently swallowed unrelated mid-rebuild query failures instead of
        # only tolerating the "panel not mounted yet" case it was meant for.
        if self.query("#library-search-rag-panel"):
            await self._refresh_search_rag_panel_state_widgets()
            # task-4023 AC#6 (RC-08): results used to land ~30 rows below
            # the fold behind the configuration region -- pressing Run left
            # the visible half of the canvas pixel-identical. Reveal the
            # Evidence region the moment a run starts (it already shows the
            # in-flight "Searching…" line), so the action visibly did
            # something at the point of action.
            self.call_after_refresh(self._reveal_library_rag_results)
        self._execute_library_rag_search(request)

    @on(Button.Pressed, ".library-rag-result-action")
    async def select_library_rag_result(self, event: Button.Pressed) -> None:
        """Select an evidence row for inspector review and Console handoff."""
        event.stop()
        result_index = self._trailing_index(event.button.id)
        await self._select_library_rag_result_by_index(result_index)

    async def _select_library_rag_result_by_index(
        self, result_index: int | None
    ) -> None:
        """Shared select-evidence implementation (Task 12).

        Used by the "Select evidence" button handler above AND by the
        focused-card Enter key path (`action_library_rag_result_card_select`)
        so both routes run the exact same selection logic -- no duplicated
        implementation between the mouse and keyboard paths.

        Resolves `result_index` against the CURRENT panel state's
        (scope-filtered, D4/task-5) `results`, not the screen's raw
        `_library_rag_results` -- the rendered cards' indices come from
        `library_rag_results_body_children`'s `enumerate(state.results)`,
        which is that same filtered tuple. Indexing the raw list instead
        would misalign as soon as any earlier row is scope-hidden: e.g.
        clicking the second VISIBLE card after an earlier source is
        toggled off would select whatever sits at raw position 1, not the
        row actually shown at that card.
        """
        rows = self._library_rag_panel_state().results
        if result_index is None or result_index >= len(rows):
            return
        self._library_rag_selected_result_id = rows[result_index].result_id
        await self._refresh_search_rag_panel_state_widgets()

    @on(Button.Pressed, ".library-rag-result-open")
    async def open_library_rag_result(self, event: Button.Pressed) -> None:
        """Open a Search/RAG evidence result straight to its Library detail surface."""
        event.stop()
        index = self._trailing_index(event.button.id)
        await self._open_library_rag_result_by_index(index)

    async def _open_library_rag_result_by_index(self, index: int | None) -> None:
        """Shared open-evidence implementation (Task 12).

        Used by the "Open" button handler above AND by the focused-card `o`
        key path (`action_library_rag_result_card_open`) so both routes run
        the exact same open logic -- no duplicated implementation between
        the mouse and keyboard paths.

        Resolves `index` against the CURRENT panel state's (scope-filtered,
        D4/task-5) `results` -- see `_select_library_rag_result_by_index`'s
        docstring for why the raw `_library_rag_results` list is the wrong
        source once scope filtering can remove earlier rows.
        """
        rows = self._library_rag_panel_state().results
        if index is None or not (0 <= index < len(rows)):
            return
        row = rows[index]
        await self._open_library_item_by_id(row.open_source_type, row.source_id)

    def _focused_library_rag_result_card_index(self) -> int | None:
        """Return the evidence index of the focused `.library-rag-result-card`.

        Task 12/RAG-36: Enter/`o`/the `u` fast path all gate on the
        CURRENTLY FOCUSED widget being one of the per-result cards (not
        just any Button.Pressed/global key) -- this is the single place
        that resolves "which card" via the same `_trailing_index` helper
        the button handlers already use on their own ids. Returns `None`
        when nothing is focused or the focused widget isn't a result card
        (e.g. the query Input, a Button, or nothing at all), which every
        caller treats as a no-op.
        """
        focused = self.focused
        if focused is None or not focused.id:
            return None
        if not focused.id.startswith("library-rag-result-card-"):
            return None
        return self._trailing_index(focused.id)

    async def action_library_rag_result_card_select(self) -> None:
        """Enter on a focused evidence card: select it (Task 12/RAG-36).

        Mirrors clicking the row's own "Select evidence" button -- routes
        through the identical `_select_library_rag_result_by_index` no
        matter which input method triggered it.
        """
        index = self._focused_library_rag_result_card_index()
        if index is None:
            return
        await self._select_library_rag_result_by_index(index)

    async def action_library_rag_result_card_open(self) -> None:
        """`o` on a focused evidence card: open it (Task 12/RAG-36).

        Mirrors clicking the row's own "Open" button -- routes through the
        identical `_open_library_rag_result_by_index` no matter which input
        method triggered it.
        """
        index = self._focused_library_rag_result_card_index()
        if index is None:
            return
        await self._open_library_rag_result_by_index(index)

    @on(Button.Pressed, "#library-rag-use-selected-in-console")
    def use_selected_library_rag_result_in_console(
        self,
        event: Button.Pressed,
    ) -> None:
        """Stage selected evidence from the center results lane."""
        self._use_library_rag_result_in_console(event)

    async def action_library_rag_use_in_console(self) -> None:
        """Keyboard shortcut for staging selected Search/RAG evidence in Console.

        Task 12/RAG-36 focused-card fast path: when a `.library-rag-result-
        card` currently holds keyboard focus, `u` selects THAT evidence
        (same as Enter would) and then stages it, in one keystroke --
        instead of requiring the user to Tab to the card, press Enter to
        select, then press `u` to stage. This does not change `u`'s
        meaning when no card is focused (still stages whatever evidence
        was already selected, unchanged from before this task); it only
        adds a shortcut for the case where the user is looking straight at
        the evidence they want staged but hasn't explicitly selected it
        yet -- the same "act on what's focused" idiom Enter/`o` use.
        """
        if self._library_selected_row_id != LIBRARY_ROW_BROWSE_SEARCH:
            return
        index = self._focused_library_rag_result_card_index()
        if index is not None:
            await self._select_library_rag_result_by_index(index)
        self._stage_library_rag_result_in_console()

    def _use_library_rag_result_in_console(self, event: Button.Pressed) -> None:
        """Shared implementation for inspector and results-lane handoff controls."""
        event.stop()
        self._stage_library_rag_result_in_console()

    def _stage_library_rag_result_in_console(self) -> None:
        """Stage the selected Search/RAG evidence result in Console."""
        panel_state = self._library_rag_panel_state()
        console_action = panel_state.use_in_console_action
        if not console_action.enabled or panel_state.selected_result is None:
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify(console_action.disabled_reason, severity="warning")
            return

        opener = getattr(self.app_instance, "open_console_for_live_work", None)
        if not callable(opener):
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify(
                    "Use in Console is unavailable for Library Search/RAG.",
                    severity="warning",
                )
            return

        # Task-2852 (b): Console's blocking first-run setup card visually
        # covers the whole workbench, so if we said nothing here the user's
        # selection would silently vanish into a locked "Get started"
        # screen. The handoff still proceeds -- the evidence really is
        # staged, and a matching receipt appears on the locked Console
        # surface (AC #2, `console_setup_staged_receipt`) -- this is just
        # the immediate, pre-navigation half of that receipt.
        if self._console_setup_would_block():
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify(
                    LIBRARY_RAG_USE_IN_CONSOLE_LOCKED_NOTICE,
                    severity="information",
                )

        opener(
            source="Library Search/RAG",
            title=panel_state.selected_result.title,
            payload=build_library_rag_console_live_work_payload(
                panel_state.selected_result,
                query=panel_state.query_state.query,
            ),
            status="staged",
            recovery="Review citations before sending.",
            action_label="Review evidence in Console",
        )

    async def _apply_library_rag_search_outcome(
        self,
        request: LibraryRagSearchRequest,
        outcome: LibraryRagSearchOutcome,
    ) -> None:
        """Resolve a completed Library Search/RAG worker's outcome into state.

        The state fields (results/status/recovery) always apply once the
        stale-query and stale-mode guards pass -- even if the user has since
        left the Search canvas (a different rail row, e.g. Media) -- so a
        dangling "searching" status can never survive: an outcome that lands
        while the user is elsewhere still resolves it, and re-entering the
        Search canvas composes from settled state instead of a stale
        in-flight line. Only the live widget refresh is skipped when the
        panel isn't mounted; there is nothing on screen to update.
        """
        if not self.is_mounted:
            return
        current_query = self._library_rag_panel_state().query_state.query
        if request.query != current_query:
            # Stale: a newer query has since replaced this one.
            return
        if request.mode != self._library_rag_mode:
            # Stale: the mode toggled mid-flight; this result belongs to
            # the mode the user has since left.
            return
        self._library_rag_results = outcome.results
        self._library_rag_retrieval_status = outcome.status
        self._library_rag_recovery_state = outcome.recovery_state
        self._library_rag_diagnostics = outcome.diagnostics
        # task-15 finding I3: `request.query` is what this outcome was
        # actually retrieved for -- already verified equal to the panel's
        # query at the top of this method (the stale-query guard above), so
        # it is safe to record as "the searched query" here.
        self._library_rag_searched_query = request.query
        self._library_rag_selected_result_id = ""
        # D1: the results-arrival transition is the ONLY place allowed to
        # force the `Recent searches` collapsible open/closed -- collapse it
        # once evidence lands (results take visual priority), expand it
        # when a search settles with nothing to show. Every other refresh
        # path leaves the user's manual expand/collapse alone.
        self._library_rag_history_collapsed = bool(self._library_rag_results)
        # Phase two. Deliberately before the mounted-panel check below: the
        # answer must be generated whether or not the user is looking at the
        # Search canvas right now (same reason the state fields above always
        # apply), and setting the in-flight flag here means the single
        # refresh below already paints "Answering…" instead of needing a
        # second one.
        self._start_library_rag_answer(request, outcome)
        if self._library_selected_row_id != LIBRARY_ROW_BROWSE_SEARCH or not self.query(
            "#library-search-rag-panel"
        ):
            return
        await self._refresh_search_rag_panel_state_widgets(force_history_collapse=True)
        # task-4023 AC#6 (RC-08): the landed evidence must be visible at
        # the point of action, not below the fold.
        self.call_after_refresh(self._reveal_library_rag_results)

    def _reveal_library_rag_results(self) -> None:
        """Scroll the Search/RAG panel so the Evidence region is on screen.

        Mirrors the prompt-history idiom (``scroll_to_widget(..., top=True)``)
        on the panel's own ``VerticalScroll``. Called after a run starts and
        after its results land; a missing panel (user navigated away
        mid-flight) is a silent no-op.
        """
        try:
            panel = self.query_one("#library-search-rag-panel", LibrarySearchRagPanel)
            heading = self.query_one("#library-rag-results-heading", Static)
        except (NoMatches, QueryError):
            return
        panel.scroll_to_widget(heading, animate=False, top=True)

    def _start_library_rag_answer(
        self,
        request: LibraryRagSearchRequest,
        outcome: LibraryRagSearchOutcome,
    ) -> None:
        """Kick off phase two -- generate an answer from what retrieval found.

        Only rag mode ever answers: keyword Search is a retrieval mode and
        never reaches a provider. Only a settled `ready`/`empty` retrieval
        does either -- a `blocked`/`failed` outcome already renders its own
        recovery copy, and an answer built on a retrieval that did not run
        would be a guess wearing an answer's clothes.

        The zero-row (`empty`) case still generates: `generate_library_rag_
        answer` answers it honestly ("Nothing in your library supports an
        answer to that.") WITHOUT calling a provider, and keeping one path
        here means that rule lives in exactly one place instead of being
        re-derived by the screen. What it does NOT do is raise the
        "answering" in-flight status: no provider call is made, so there is
        no in-flight window worth showing -- and showing one would swap the
        quiet no-match line for the idle "No evidence yet" line for a frame,
        which is precisely the state that line exists to replace.

        Args:
            request: The retrieval request this outcome answers.
            outcome: The settled retrieval outcome.
        """
        # A new retrieval invalidates whatever answer was on screen, whatever
        # happens next in this method.
        self._reset_library_rag_answer_state()
        if request.mode != "rag":
            return
        if outcome.status not in LIBRARY_RAG_ANSWERABLE_RETRIEVAL_STATUSES:
            return
        chat_kwargs = self._library_rag_answer_chat_kwargs()
        if chat_kwargs is None:
            return
        # The run gate (`_library_rag_panel_state`'s `provider_name=`) and
        # this call both go through `library_rag_answer_provider_gate()`
        # (PR-T2 Task 7 -- endpoint name AND resolvable credentials, the
        # same question Console's readiness check asks); a `None` provider
        # below means the gate would already have blocked `rag` mode. One
        # gate call resolves both halves, where this used to resolve the
        # endpoint twice (finding I1).
        answer_gate = library_rag_answer_provider_gate()
        provider, model = answer_gate.provider, answer_gate.model
        if provider is None:
            # The run gate blocks rag mode without a ready provider, so
            # this is unreachable through the UI; if it ever is reached,
            # saying nothing is honest -- that gate's copy already explains
            # why.
            logger.debug("Library RAG answer skipped: no provider configured.")
            return
        # Built from state that has just been applied, so the note describes
        # THESE results (`library_rag_coverage_note` derives it from the
        # outcome's diagnostics + rows). Read before the in-flight flag is
        # raised so the status overlay can't affect it.
        coverage_note = self._library_rag_panel_state().coverage_note
        self._library_rag_answer_query = request.query
        self._library_rag_answer_mode = request.mode
        self._library_rag_answer_in_flight = bool(outcome.results)
        # PR-3 Task 3: named for the in-flight "Asking <provider>..." line
        # -- set unconditionally alongside the flag above (never gated on
        # `outcome.results` itself) since the panel-state builder is what
        # decides whether to forward it, keyed off that same flag.
        self._library_rag_answer_in_flight_provider = provider
        self._execute_library_rag_answer(
            request,
            results=outcome.results,
            coverage_note=coverage_note,
            provider=provider,
            model=model,
            chat_kwargs=chat_kwargs,
        )

    def _library_rag_answer_chat_kwargs(self) -> dict[str, Any] | None:
        """The `chat=` seam override for `generate_library_rag_answer`.

        Three cases, and the distinction between the last two is the whole
        point: an app with no `library_rag_answer_chat` attribute at all
        (the shipping `TldwCli`) gets `{}` -- i.e. the service's own default,
        `chat_api_call` -- so production needs no wiring to answer. An app
        that DOES carry the attribute owns the decision: a callable is used
        as the chat seam, and a non-callable (`None`) disables generation
        entirely, which is how `Tests/UI/app_factory.py` keeps every pilot
        that never opted into a fake off the network.

        Returns:
            Keyword arguments to forward to `generate_library_rag_answer`,
            or `None` when generation is disabled for this app.
        """
        if not hasattr(self.app_instance, "library_rag_answer_chat"):
            return {}
        chat_seam = self.app_instance.library_rag_answer_chat
        return {"chat": chat_seam} if callable(chat_seam) else None

    async def _apply_library_rag_answer(
        self,
        request: LibraryRagSearchRequest,
        answer: LibraryRagAnswer,
    ) -> None:
        """Resolve a completed answer worker's outcome into state.

        The two guards mirror `_apply_library_rag_search_outcome`'s: an
        answer generated for a query the panel has moved past, or for a mode
        the user has since left, is discarded rather than applied. They
        compare against `_library_rag_answer_query`/`_library_rag_answer_mode`
        -- the fields recording what the CURRENT generation is for -- because
        every invalidating transition clears them (`_reset_library_rag_
        answer_state`), and unlike the live query box those fields do not
        move when the user merely types (an answer belongs to the results
        still on screen, and typing deliberately leaves those alone: B5).

        Discarding never leaves a dangling "answering" status: the same
        transitions that clear the guard fields clear the in-flight flag with
        them, so whatever superseded this answer already settled the panel.
        """
        if not self.is_mounted:
            return
        if request.query != self._library_rag_answer_query:
            # Stale: a newer search (or a reset) owns the panel now.
            return
        if request.mode != self._library_rag_answer_mode:
            # Stale: the mode toggled mid-flight; this answer belongs to the
            # mode the user has since left.
            return
        self._library_rag_answer = answer
        self._library_rag_answer_in_flight = False
        if self._library_selected_row_id != LIBRARY_ROW_BROWSE_SEARCH or not self.query(
            "#library-search-rag-panel"
        ):
            return
        # Results and history are untouched by an answer landing -- only the
        # answer region, the run gate and the query status line change.
        await self._refresh_search_rag_panel_state_widgets(
            include_results_and_history=False
        )

    def _sync_library_rag_scope_toggle_and_run_gate_widgets(self) -> None:
        """Refresh the scope-toggle counts and the Run gate in place, with
        NO `await` (RAG-27 fix-review).

        Called synchronously from `_apply_local_source_snapshot`'s
        in-place branch, which fires off the UI thread on every ingest
        done-count growth -- a moment with no coordination against the
        panel's four other refresh callers (`update_library_rag_query`,
        `_start_library_rag_query`, `select_library_rag_result`,
        `_apply_library_rag_search_outcome`), all of which `await
        self._refresh_search_rag_panel_state_widgets(...)` directly with
        no shared lock or exclusive worker group. That coroutine's real
        yield points (`await widget.remove()` / `await ...mount(...)` for
        the query-status callout and, when `include_results_and_history`
        is left True, results/history) make two concurrent invocations
        unsafe: an ingest snapshot landing mid-keystroke could interleave
        two remove/mount sequences on the same containers (double-remove
        or duplicate-id). Restricting the snapshot path to plain
        attribute writes -- `Button.label`/`.disabled`/`.tooltip`,
        `Static.update()` -- has no yield points at all, so it can never
        interleave with anything and needs no coordination.

        The query region's reserved quiet line IS synced here (F1), by the
        same yield-free `Static.update()` class of write as the scope
        summary below -- no remove/mount, so RAG-27's constraint holds.
        The original trade-off deferred that row along with the callout,
        on the reasoning that it carried only the run gate's *reason*
        text. That reasoning expired with PR-T2 Task 4: the same row now
        also carries the money disclosure (`library_rag_paid_mode_notice`,
        naming the provider Run would bill), and deferring it produced a
        silent paid state. A Library revisit past
        `LIBRARY_SNAPSHOT_CACHE_TTL_SECONDS` composes with all-zero counts
        -> no scope -> a blocked-but-QUIET gate (no callout, empty row);
        the snapshot then lands real counts and this method flips Run to
        enabled, leaving a runnable paid button above an empty row that
        never said a provider would be billed. Deriving the row's copy
        from `library_rag_query_quiet_text(panel_state)` -- the same
        builder `compose()` and the full refresh use, off the same state
        the run gate above is read from -- keeps the disclosure and the
        button they sit next to from ever disagreeing.

        Trade-off (narrowed): the blocked-callout/recovery block below the
        quiet line is still NOT refreshed here -- it is the part that
        requires remove/mount, and `#library-rag-query-controls`'
        `has-recovery` class with it. So a snapshot that lifts the
        quiet no-scope blocker only to land on a LOUD one (rag mode with
        no provider configured: "Select a provider/model...") still shows
        no callout until the next full refresh. That residue cannot spend
        money -- every blocker it covers leaves Run disabled, and the
        quiet line, derived from the same state, shows no paid notice
        while one is in force. Accepted narrowly for this snapshot-driven
        path only; every other caller above still runs the full
        `_refresh_search_rag_panel_state_widgets` and is unaffected.

        (task-2075 D5) The *scope* region's own recovery block --
        `#library-rag-source-scope`'s `has-recovery` class plus its
        `library_rag_scope_recovery_children` -- gets different treatment:
        it IS kept honest here, but change-gated rather than refreshed on
        every call. A cold boot lands on this canvas before the first
        compose (`_library_selected_row_id` is already
        `LIBRARY_ROW_BROWSE_SEARCH`), so `compose()` renders the recovery
        banner from the zero-count defaults every fresh screen starts
        with; the very next snapshot -- real counts, taken by this
        in-place branch precisely because the row is already Search --
        previously never told that banner counts had arrived, leaving a
        stale "No Library sources yet" beside populated, enabled toggles.
        Comparing `library_rag_scope_shows_recovery(...)` against
        `self._library_rag_scope_recovery_visible` (cached, `None` until
        the first call) means steady-state snapshots -- the overwhelming
        common case, and RAG-27's whole point -- see no change and take
        the same no-op, no-yield path as everything else in this method;
        only an actual flip schedules `_mirror_library_rag_scope_recovery`
        as a worker (see that method for why a worker rather than an
        inline remove/mount here).
        """
        if self._library_selected_row_id != LIBRARY_ROW_BROWSE_SEARCH or not self.query(
            "#library-search-rag-panel"
        ):
            return
        panel_state = self._library_rag_panel_state()

        try:
            run_button = self.query_one("#library-rag-run-query", Button)
        except (NoMatches, QueryError):
            return
        run_action = panel_state.query_state.run_action
        run_button.label = run_action.label
        run_button.disabled = not run_action.enabled
        run_button.tooltip = run_action.tooltip

        # (F1) Written in the SAME pass as the button above, off the SAME
        # `panel_state`, because this row carries rag mode's paid-mode
        # notice: enabling Run without refreshing it is exactly how a
        # revisit-past-the-snapshot-TTL produced a runnable paid button
        # with no disclosure on screen. Plain `Static.update()` -- no
        # yield point, so RAG-27's constraint (see the docstring) holds.
        try:
            self.query_one("#library-rag-query-quiet-line", Static).update(
                library_rag_query_quiet_text(panel_state)
            )
        except (NoMatches, QueryError):
            pass

        options_by_source_type = {
            option.source_type: option for option in panel_state.scope.options
        }
        for toggle in self.query(".library-rag-scope-toggle"):
            if not isinstance(toggle, Button) or toggle.id is None:
                continue
            source_type = toggle.id.removeprefix("library-rag-scope-toggle-")
            option = options_by_source_type.get(source_type)
            if option is None:
                continue
            toggle.label = scope_toggle_label(option)
            toggle.disabled = not option.available

        try:
            self.query_one("#library-rag-scope-summary", Static).update(
                self._library_rag_scope_summary(panel_state)
            )
        except (NoMatches, QueryError):
            pass

        shows_recovery = library_rag_scope_shows_recovery(panel_state.scope)
        if shows_recovery != self._library_rag_scope_recovery_visible:
            # Updated eagerly (before the worker even starts) so a burst of
            # snapshots landing faster than the worker can run only ever
            # schedules one mirror per actual flip -- a repeat call with the
            # SAME new value during that window already matches the cache
            # and takes the branch above instead.
            self._library_rag_scope_recovery_visible = shows_recovery
            self.run_worker(
                self._mirror_library_rag_scope_recovery(),
                exclusive=True,
                group="library_rag_scope_recovery_mirror",
            )

    async def _apply_library_rag_scope_recovery_block(
        self,
        scope_container: Vertical,
        panel_state: LibraryRagPanelState,
    ) -> None:
        """Remove/mount `#library-rag-source-scope`'s recovery children.

        Shared by `_refresh_search_rag_panel_state_widgets` (the full
        refresh every OTHER panel caller awaits directly) and
        `_mirror_library_rag_scope_recovery` (the change-gated snapshot
        path above) so the two can never render this block differently.
        """
        scope_container.set_class(
            library_rag_scope_shows_recovery(panel_state.scope), "has-recovery"
        )
        scope_recovery_widgets = list(self.query("#library-rag-scope-recovery"))
        import_buttons = list(self.query("#library-rag-open-import-export"))
        for widget in (*scope_recovery_widgets, *import_buttons):
            await widget.remove()
        for child in library_rag_scope_recovery_children(panel_state):
            await scope_container.mount(child)

    async def _refresh_library_rag_query_status_widgets(
        self,
        panel_state: LibraryRagPanelState,
    ) -> None:
        """Sync the Run button and the query region's conditional status block.

        The quiet line / callout+recovery block is torn down and rebuilt
        from `library_rag_query_status_children` on every call -- it is at
        most two `Static` widgets, so a full rebuild is cheap and (unlike
        hand-written incremental mount/update/remove logic) can never drift
        from what `compose()` renders on a fresh mount.
        """
        query_controls = self.query_one("#library-rag-query-controls", Vertical)
        query_controls.set_class(
            library_rag_query_shows_full_recovery(panel_state.query_state),
            "has-recovery",
        )

        run_action = panel_state.query_state.run_action
        run_button = self.query_one("#library-rag-run-query", Button)
        run_button.label = run_action.label
        run_button.disabled = not run_action.enabled
        run_button.tooltip = run_action.tooltip

        for widget_id in (
            "library-rag-query-quiet-line",
            "library-rag-query-blocked-callout",
            "library-rag-query-recovery",
        ):
            for widget in list(self.query(f"#{widget_id}")):
                await widget.remove()
        anchor = "#library-rag-query-input"
        for child in library_rag_query_status_children(panel_state):
            await query_controls.mount(child, after=anchor)
            anchor = f"#{child.id}"

    async def _refresh_library_rag_answer_widgets(
        self,
        panel_state: LibraryRagPanelState,
    ) -> None:
        """Rebuild the Answer region from `library_rag_answer_children` (Task 4).

        Torn down and rebuilt whole on every call, from the SAME builder
        `LibrarySearchRagPanel.compose()` uses -- the idiom
        `_refresh_library_rag_query_status_widgets` already follows for the
        query status block, and for the same reason: the region is a handful
        of `Static`s, so a full rebuild is cheap and (unlike hand-written
        incremental update logic) cannot drift from what a fresh mount
        renders.

        Mounted as a SIBLING, `before="#library-rag-results"`, matching
        `compose()`'s order exactly. It must never end up inside
        `#library-rag-results`: that container's own teardown loop
        (`_refresh_library_rag_results_widgets`) removes every child not in
        `LIBRARY_RAG_RESULTS_STATIC_WIDGET_IDS`, and would destroy the answer
        on every results refresh.

        Skipped entirely when nothing this region renders from has changed
        (`_library_rag_answer_render_key`). This refresh runs on EVERY panel
        refresh, including the per-keystroke one, and an answer only ever
        changes when generation settles -- without the check, typing in rag
        mode would tear down and remount the whole answer (up to
        `LIBRARY_RAG_ANSWER_DISPLAY_MAX_LENGTH` characters of `Static`) on
        each character, which is exactly the churn class task-284 removed
        for the results/history lists. Identity (`is`) is used for the
        answer: `LibraryRagAnswer` is frozen and replaced wholesale, so a
        new object always means new content, and this avoids deep-comparing
        an embedded evidence bundle.
        """
        panel_widgets = list(self.query("#library-search-rag-panel"))
        if not panel_widgets:
            return
        panel = panel_widgets[0]
        render_key = (
            panel_state.query_state.mode,
            panel_state.retrieval_status == "answering",
            panel_state.answer,
        )
        previous_key = self._library_rag_answer_render_key
        if (
            previous_key is not None
            and previous_key[0] == render_key[0]
            and previous_key[1] == render_key[1]
            and previous_key[2] is render_key[2]
        ):
            return
        for widget in list(self.query("#library-rag-answer")):
            await widget.remove()
        for child in library_rag_answer_children(panel_state):
            await panel.mount(child, before="#library-rag-results")
        self._library_rag_answer_render_key = render_key

    async def _refresh_library_rag_history_widget(
        self,
        panel_state: LibraryRagPanelState,
        *,
        force_collapsed: bool | None = None,
    ) -> None:
        """Rebuild the `Recent searches` collapsible content from state.

        Mutates the compose-time `Collapsible` in place (its `collapsed`
        reactive, then its `Contents` children) rather than replacing the
        whole widget -- two refreshes can be triggered back to back (the
        synchronous "searching" status refresh, then the search worker's
        own "outcome" refresh), and remove-then-mount of the same fixed ID
        from overlapping calls raises `DuplicateIds`. The lock serializes
        those calls so one full rebuild always finishes before the next
        starts.

        `force_collapsed` (D1) is `None` for every caller except the
        results-arrival transition in `_apply_library_rag_search_outcome`:
        `None` leaves the live widget's `collapsed` reactive exactly as the
        user left it; a `bool` applies the collapse below WITHOUT the
        watcher (task-4023 AC#6 / RC-08 -- see the inline comment). This is
        safe for full recomposes too (scope toggles, the mode toggle) -- not
        just in-place refreshes (query edits, evidence selection) -- because
        both writers keep the state field mirrored: a USER'S title click
        takes the watcher path and `sync_library_rag_history_collapsed`
        copies it into `_library_rag_history_collapsed`, while the
        force-collapse caller already set that same field at the transition,
        so `compose()` always rebuilds the `Collapsible` from the last
        choice instead of a stale field.
        """
        async with self._library_rag_history_refresh_lock:
            history_widgets = list(self.query("#library-rag-history"))
            if not history_widgets:
                return
            collapsible = history_widgets[0]
            if not isinstance(collapsible, Collapsible):
                return
            if force_collapsed is not None and (
                collapsible.collapsed != force_collapsed
            ):
                # task-4023 AC#6 (RC-08): assigning the reactive here used
                # to run Textual's ``Collapsible._watch_collapsed``, which
                # schedules an ANIMATED ``self.scroll_visible()`` -- at
                # results arrival that animation scrolled the panel to the
                # Recents strip at the BOTTOM, sailing past the Evidence
                # region (and overriding `_reveal_library_rag_results`,
                # found by spying ``panel.scroll_to``). Apply the visual
                # collapse without the watcher: the state field was already
                # set by the results-arrival transition, so skipping the
                # Collapsed/Expanded message loses nothing (the handler
                # only mirrors the value back into the same field). A
                # USER'S own title click still takes the normal watcher
                # path, scroll and all.
                collapsible.set_reactive(Collapsible.collapsed, force_collapsed)
                collapsible._update_collapsed(force_collapsed)
                collapsible.refresh(layout=True)
            try:
                contents = collapsible.query_one(Collapsible.Contents)
            except (NoMatches, QueryError):
                # Defensive, mirroring the two guards above: an "exclusive"
                # search worker can be cancelled mid-refresh by a newer one
                # (e.g. re-running a history entry while a prior query is
                # still settling), which can catch this specific
                # `Collapsible` instance between un/remounting its own
                # `Contents` child. The next refresh (there is always one --
                # every query/scope/selection change triggers one) picks up
                # the settled state; there is nothing to safely rebuild here.
                return
            for child in list(contents.children):
                await child.remove()
            for row in library_rag_history_children(panel_state):
                await contents.mount(row)

    async def _refresh_library_rag_results_widgets(
        self,
        panel_state: LibraryRagPanelState,
    ) -> None:
        """Rebuild the Evidence region body from `library_rag_results_body_children`.

        Shared with `LibrarySearchRagPanel.compose()` (C1): both build rows,
        the searching line, recovery copy, and the empty state from the
        same function, closing the compose-vs-refresh duplication that
        previously let the two paths drift apart. Each row is now ONE
        `.library-rag-result-card` (Task 12/RAG-36) instead of several flat
        sibling widgets, but that card is still just another direct child
        of `results_container` -- the remove/remount loop below (skip
        `LIBRARY_RAG_RESULTS_STATIC_WIDGET_IDS`, tear down everything else,
        remount from `library_rag_results_body_children`) needed no change
        to stay in lockstep with the new structure.

        What DOES need explicit handling: this remove/remount cycle
        destroys and recreates the card widget INSTANCES, including
        whichever one currently holds keyboard focus (e.g. the user just
        pressed Enter on a card, which calls this via
        `_select_library_rag_result_by_index`). Textual does not carry
        focus across a removed widget being replaced by a same-id
        successor, so without the save/restore below, every keyboard
        selection would silently drop focus back to nothing -- breaking
        the "Enter selects, then keep going" keyboard flow this task exists
        to add.
        """
        results_container = self.query_one("#library-rag-results", Vertical)
        self.query_one("#library-rag-results-heading", Static).update(
            results_heading_text(panel_state)
        )
        focused = self.focused
        focused_card_id = (
            focused.id
            if focused is not None
            and focused.id
            and focused.id.startswith("library-rag-result-card-")
            else None
        )
        for child in list(results_container.children):
            if child.id in LIBRARY_RAG_RESULTS_STATIC_WIDGET_IDS:
                continue
            await child.remove()
        for child in library_rag_results_body_children(panel_state):
            await results_container.mount(child)
        if focused_card_id is not None:
            try:
                self.query_one(f"#{focused_card_id}").focus()
            except (NoMatches, QueryError):
                # The just-focused card's result can legitimately be gone
                # after the rebuild (e.g. a re-run landed a shorter result
                # set) -- falling back to no focus is correct there, not a
                # bug to paper over.
                pass

    @staticmethod
    def _library_rag_scope_summary(panel_state: LibraryRagPanelState) -> str:
        return library_rag_scope_summary(panel_state.scope)


# --- BEGIN generated search+rag-state shims (permanent; byte-for-byte canon) ---
# task 3: exposes every `LibraryRagSearchState` field under its original
# `_library_rag_<field>`/`_library_search_<field>` name on THIS controller
# too, reading/writing through the injected `rag_search_state_accessor`
# instead of a direct `self._rag_search_state` attribute (this class has
# none) -- same generator shape the shim block `LibraryScreen` carries
# (task 2) and `LibraryConversationsController`/`LibraryCollectionsController`
# carry, attached programmatically so the class body gains no `FunctionDef`s
# (the size ratchet counts those). `SEARCH_PREFIXED_STATE_FIELDS` is
# imported from `library_rag_search_state` -- the dataclass's own module --
# so this is not a third independent literal copy of the one-field prefix
# exception; see that module's own docstring for the drift-risk note this
# avoids from the start.
for _rsc_field in dataclasses.fields(LibraryRagSearchState):
    _rsc_prefix = (
        "_library_search_"
        if _rsc_field.name in SEARCH_PREFIXED_STATE_FIELDS
        else "_library_rag_"
    )
    setattr(
        LibraryRagSearchController,
        _rsc_prefix + _rsc_field.name,
        property(
            lambda self, _n=_rsc_field.name: getattr(
                self._rag_search_state_accessor(), _n
            ),
            lambda self, value, _n=_rsc_field.name: setattr(
                self._rag_search_state_accessor(), _n, value
            ),
        ),
    )
del _rsc_field, _rsc_prefix
# --- END generated search+rag-state shims ---

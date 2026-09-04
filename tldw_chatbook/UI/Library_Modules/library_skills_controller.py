"""Library Skills canvas controller.

Controller PR of the Skills extraction series (wave-4 task 2 of
``.superpowers/sdd/2026-09-04-library-decomposition-wave4-skills``; skills
series 2/3; recipe: ``backlog/docs/library-decomposition-recipe.md``;
``library_rag_search_controller.py`` -- the newest, largest prior combined
series -- is the template this mirrors byte-for-byte in shape, since Skills
shares its two-prefix-plus-bare-name state shim and its "one giant
call-graph component" single-controller shape). Owns the entire Skills
canvas cluster: list/browse paging+sort+filter, the inline Import row, the
in-canvas skill detail/editor (open/save/delete/discard/conflict/dirty-
tracking/tool-picker/mode toggles), and the trust panel (setup/unlock/
review/approve/reset/script-grant). ``LibraryScreen`` keeps one-line
delegators under 70 of these original names -- skills cleanup (task 3)
deleted the other 16 as dead weight (zero external references beyond the
controller's own internal calls); see ``_SKILLS_CLUSTER_SCREEN_DELEGATOR_
PRUNED`` in ``Tests/Architecture/test_library_skills_wiring.py`` for the
list. Two existing,
already-extracted Skills modules -- ``library_skill_import_controller.py``
(``LibrarySkillImportCoordinator``) and ``library_skills_browse_
controller.py`` (``LibrarySkillsBrowseController``) -- are untouched;
``LibraryScreen`` still owns both live instances (``_library_skill_import_
coordinator``/``_library_skills_browse_controller``), now reached through
this controller via injected accessors, same capture-controller precedent
every prior series uses for its own held-collaborator field.

**Cluster derivation.** Wave-4 task 1's own census: an ``ast`` scan of
``LibraryScreen`` for method names containing ``"skill"`` (case-insensitive)
finds **133 raw ``FunctionDef`` matches, 127 unique names** (the 6-match gap
is SIX ``@property``/``@x.setter`` pairs -- each pair's getter def + setter
def = 2 raw matches but 1 unique name, so 6 names = 6 gap -- corrected here
from an earlier "three pairs" arithmetic error task 2's own report and this
docstring both originally carried despite listing all six names; task 2's
post-landing review fix round caught and fixed the report copy (§12c) but
missed this docstring copy, fixed now by task 3), all thin projections onto
``_library_skill_import_coordinator``). This task re-ran that census fresh
at its own execution time (recipe's "never trust a carried-over count"
rule, §6) and reconfirmed the identical 133/127. Reading every one of the
127 bodies (not the substring) finds **zero** false positives from another
subsystem (unlike collections' 3 Prompts-owned methods or search+RAG's 10)
-- every "skill"-named method genuinely belongs to this cluster or to the
6 import-coordinator delegator properties above. A companion ``ast`` sweep
for non-"skill"-named methods called EXCLUSIVELY by cluster members (the
conversations exemplar's own "startswith enumeration trap" in its method
form, recipe §11) found **zero** -- no hidden false negative.

**Single vs. split controller: single, decided by call-graph connected-
components analysis.** The wave-4 plan named a possible editor/trust vs.
browse/list seam as worth checking at this cluster's size (121 movable
candidates after the 6 delegator-property exclusion, nearly double
collections' 64). A union-find over every ``self.<name>(...)`` call among
the 121 candidates finds **one component of 107 members** plus 13 singleton/
pair components with no intra-cluster calls at all (reached only via
``@on``/other external dispatch, not evidence of a second cluster -- they
are isolated because they are leaf nodes, not because they belong
elsewhere). A finer, hand-labelled bucket pass (editor/trust/browse/detail/
import/other, heuristic only) shows the SAME thing search+RAG's own method-
level call graph showed: dense cross-calling in every direction --
editor<->trust (5), editor<->detail (15), trust<->detail (10), browse<->
editor (6) -- with no bucket pair showing zero edges. There is no subset of
the cluster that only ever calls within itself; the plan's own hypothetical
seam does not hold. **Decision: ONE combined ``LibrarySkillsController``**,
matching the plan's own "when unsure, one controller" default and the
search+RAG precedent's identical resolution at a comparable (50-candidate)
scale.

**Exclusions -- 41 of 127, not moved (86 move):**

1. **6 merely-delegate-to-existing-controller properties** (the plan's own
   named exclusion class): ``_library_skills_import_open``, ``_path``,
   ``_status``, ``_review_name``, ``_in_flight``, ``_generation`` -- each a
   ``@property``/``@x.setter`` pair whose ENTIRE body is
   ``self._library_skill_import_coordinator.snapshot.<x>`` /
   ``self._library_skill_import_coordinator.update(<x>=value)``. These stay
   screen-resident (already-extracted wiring, per the plan); this
   controller reaches them read-only where a mover needs to (below).
2. **27 unbound-fake-self test-bypass exclusions** (conversations exemplar
   precedent, recipe §11 lesson 1; Export's own 9-of-51 record is the prior
   high-water mark -- this series roughly triples it, confirming §12's own
   forward note that this shape "scales with how much a subsystem's test
   style favors unbound-``SimpleNamespace``/``Mock`` unit-style calls"). A
   repo-wide grep for ``LibraryScreen.<name>(`` across all four test roots,
   for every one of the 121 candidates, finds 27 names called with a bare
   ``SimpleNamespace``/hand-built fake (never a real, constructed
   ``LibraryScreen``) standing in for ``self``: ``_build_library_skills_
   state``, ``handle_library_skills_sort``, ``handle_library_skills_sort_
   choice``, ``handle_library_skills_filter``, ``handle_library_skill_row``,
   ``_call_library_skill_trust_service``, ``_approve_library_skill_trust``,
   ``handle_library_skill_delete``, ``handle_library_skill_delete_confirm``,
   ``handle_library_skill_delete_cancel``, ``handle_library_skill_trust_
   review``, ``_reset_library_skill_editor_state``, ``handle_library_
   skills_import_review``, ``handle_library_skills_import_browse_folder``,
   ``_library_skill_editor_active``, ``_library_skill_save_available``,
   ``_begin_library_skill_save``, ``action_library_skill_save``,
   ``_exit_library_skill_editor_guarded``, ``action_library_skill_back``,
   ``handle_library_skills_import_browse``, ``handle_library_skills_
   import_cancel``, ``handle_library_skills_trust_action``, ``_open_
   first_blocked_skill``, ``handle_library_skills_trust_reset_request``,
   ``_apply_library_skills_import_status`` (``Tests/UI/test_library_
   canvas_scoped_sync.py``), ``_present_library_skills_import_choice_if_
   needed`` (``Tests/Skills/test_skills_import.py`` -- the wave's own
   fourth-root trap, confirmed live here). Every one confirmed by reading
   the fixture construction at its call site (``SimpleNamespace(...)``/a
   ``_bind_editor_active``-style helper docstringed "Bind the real
   editor-active predicate onto a SimpleNamespace fake"), not inferred from
   the call shape alone.
3. **1 instance-attribute-monkeypatch exclusion** (conversations exemplar
   precedent, recipe §11 lesson 2): ``_request_library_skills_browse`` --
   ``Tests/UI/test_library_skills_canvas.py`` patches
   ``screen._request_library_skills_browse = lambda ...`` on one REAL,
   directly-constructed ``LibraryScreen`` instance, then calls
   ``screen._refresh_library_skills_after_committed_mutation()`` (a MOVER)
   expecting it to observe the patch. Once both live on this controller,
   the mover's own ``self._request_library_skills_browse(...)`` would
   resolve against the CONTROLLER's real copy, bypassing the SCREEN-instance
   patch. ``_call_library_skill_trust_service`` independently matches BOTH
   this shape (``Tests/UI/test_library_skills_reader.py:286`` patches it on
   a real, Pilot-mounted screen, expecting the mover ``_review_library_
   skill_trust`` to observe it) and shape 2 above (also called unbound with
   a fake) -- doubly confirmed, not a coincidence.
4. **1 module-globals-coupling exclusion** (recipe §3's second documented
   bypass shape, the search+RAG series' own ``_load_library_search_
   history``/``get_cli_setting`` precedent, reproduced here on a DIFFERENT
   free name): ``_persist_library_skill_editor_mode`` reads the bare name
   ``save_setting_to_cli_config`` (an ordinary module-level import in
   ``library_screen.py``, resolved against the DEFINING module's
   ``__globals__`` at call time). ``Tests/UI/test_library_skills_canvas.py``
   (the mode-toggle persistence test, ~line 1975) patches ``library_screen_
   module.save_setting_to_cli_config`` and presses the real editor-mode
   toggle button through a full Pilot session -- confirmed by reading the
   test, not assumed from the free-name census alone (the search+RAG
   precedent's own module-globals exclusion was found "the hard way, by
   running the battery"; this one was caught by the census first and the
   test read second, closing the loop the other way). Moving the body
   would silently repoint this test's patch away from the call the button
   press actually makes, exactly the search+RAG precedent's own failure
   mode. Its only mover-side caller, ``handle_library_skill_editor_mode``,
   reaches it through a named dependency below.
5. **1 new bypass shape this series adds to the recipe's own catalogue,
   found in TWO independent forms -- passing bare ``self`` as an
   IDENTITY-COMPARED argument, not merely as an attribute-lookup
   receiver.**

   - **Form A -- a framework API.** ``_refresh_library_skills_trust_
     posture`` calls ``self.workers.cancel_group(self, "library_skills_
     trust_posture")`` when the trust service is absent, to cancel a
     previously-scheduled posture-fetch worker. Textual's own
     ``WorkerManager.cancel_group`` (read from ``textual/worker_manager.
     py`` via ``inspect.getsource``, not assumed) filters ``worker.node ==
     node`` by IDENTITY -- unlike every other framework-service binding in
     this series (``app``/``query_one``/etc., all resolved by duck-typed
     ATTRIBUTE access, safe for a controller standing in for ``screen``),
     a controller instance can never equal the screen instance a worker
     was actually registered against (workers are always scheduled via
     the ``run_worker`` framework-service property, which forwards to
     ``self._screen.run_worker(...)`` -- the worker's own ``.node`` is
     therefore always the SCREEN). A verbatim move would make this
     cancellation a silent, permanent no-op.
   - **Form B -- a shared shell helper, and the first one that actually
     shipped broken, caught only by the battery, not the static census.**
     ``_library_screen_is_current(screen)`` (``screen_helpers.py``) reads
     ``current_screen = getattr(screen.app, "screen", screen); return
     current_screen is screen`` -- an identity check against the APP's
     real current screen. FOUR candidates call it as a bare ``self``
     forward: ``handle_library_skills_import``, ``handle_library_skills_
     import_path_changed``, ``handle_library_skills_import_retry``,
     ``_start_library_skills_import``. Moved verbatim, EVERY call would
     permanently evaluate to ``False`` (``real_screen is controller`` can
     never be true), silently no-opping the ENTIRE Skills import feature
     -- not a theoretical risk like Form A, but a REAL regression this
     task's own draft actually shipped once: the first draft moved all
     four, the wiring/ratchet battery went green (a same-name-forwarding
     regex and a byte-for-byte diff cannot see a semantic identity bug),
     and it was `Tests/UI/test_library_skills_reader.py::test_skills_
     mount_three_retained_roles_and_default_to_overview` and
     ``Tests/UI/test_library_per_click_recompose_t21116.py::test_skills_
     import_open_and_cancel_are_canvas_scoped`` -- both exercising a real
     Pilot press of the Import button -- that caught it, confirmed as a
     genuine regression (not flakiness) via a paired baseline: both PASS
     in under 2s combined against the pre-move tree and FAIL (each timing
     out its own 30s DOM-mount wait) against the four-moved draft.
     Reverted; all four stay screen-resident.
   - **Form C -- the SAME shape written inline instead of through the
     shared helper, found the SAME way (by running the Tests/Skills
     fourth-root suite, not by re-deriving from Form B's discovery).**
     ``_present_library_skills_import_snapshot`` guards its entire body
     -- the terminal-status DOM update AND the queued candidate-choice
     modal -- behind ``self.is_mounted and self.app.screen is self and
     ...``. This method is reached only externally, from the
     PRE-EXISTING, untouched ``LibrarySkillImportCoordinator._settle``
     (``library_skill_import_controller.py``), via ``getattr(runtime_
     app.screen, "_present_library_skills_import_snapshot", None)`` --
     always the REAL screen, never this controller -- so the screen's own
     delegator receives the call correctly, but its body (moved onto the
     controller) then compares ``self.app.screen is self`` with ``self``
     now the CONTROLLER: permanently ``False``. A first draft moved it;
     `Tests/Skills/test_skills_import.py` (7 of its own tests) and
     `Tests/Skills/test_skills_library_flow.py` (1 test) failed --
     confirmed via the SAME paired-baseline method as Form B (all 8 pass
     against the pre-move tree, fail against the with-this-method-moved
     draft). Reverted; stays screen-resident. Zero mover calls it
     internally (its only caller is the coordinator's external
     ``getattr``), so it needs no dependency binding at all.

   Neither form is a test-bypass in the usual sense (no test *monkeypatches*
   either function) -- all three are genuine behavior changes a pure move
   must not introduce (recipe §5's "pure moves must not change behaviour"
   in its sharpest form), excluded per the same conservative discipline as
   3-4 above. Six total exclusions from this one bypass shape:
   ``_refresh_library_skills_trust_posture`` (Form A; its three mover
   callers -- ``_setup_library_skill_trust``, ``_do_library_skill_trust_
   reset``, ``_unlock_library_skill_trust`` -- reach it through a named
   dependency below) plus the four Form-B names (``_start_library_
   skills_import`` has two mover callers, ``handle_library_skills_import_
   path_submitted``/``_run``, reached the same way; the other three are
   ``@on``-dispatched only, no mover ever calls them) plus the one Form-C
   name (``_present_library_skills_import_snapshot``, also zero mover
   callers). No other ``cancel_group`` call, ``_library_screen_is_
   current`` call, or bare-``self`` identity/equality/containment
   comparison exists anywhere in the FINAL 86-mover set -- confirmed by a
   repeat, method-scoped AST census (``ast.Compare`` with ``Is``/``IsNot``/
   ``Eq``/``NotEq``/``In``/``NotIn`` against a bare ``Name(id="self")`` on
   either side, plus every free-function and same-class-method call
   passing bare ``self`` as an argument), not assumed from these six
   instances.

**86 of the 127 candidates move onto this controller.**

**Byte-for-byte canon** (moved bodies never edited -- every name they
reference that is not this controller's own state is rebound under the SAME
name, per the two binding kinds; see ``ConsoleDictationController.__init__``
and ``LibraryRagSearchController.__init__`` for the sibling worked
examples):

1. **Framework services** (``app``, ``app_instance``, ``call_after_
   refresh``, ``focused``, ``is_mounted``, ``is_running``, ``query``,
   ``query_one``, ``refresh``, ``run_worker``) are live-read from the
   screen via ``@property`` on every access -- never snapshotted. ``is_
   running``/``app``/``refresh``/``call_after_refresh``/``query``/
   ``query_one`` exist partly because 15 movers forward ``self`` verbatim
   into the shared, multi-subsystem ``_sync_library_canvas(screen,
   "skills", ...)`` dispatcher (``canvas_sync.py``) -- the SAME real shape
   the RAG controller's own docstring documents for its own ``search``-kind
   forwarding. Read that dispatcher's actual ``kind == "skills"`` branch
   (not just its signature) before assuming a controller exposes
   everything it touches: it reads ``query_one``/``query`` for the two
   Skills widgets, calls two movers (``_library_skills_list_canvas_
   kwargs``, ``_library_skill_work_pane_kwargs``, both resolve fine on
   THIS controller), then falls into the function's own kind-independent
   top/exception-handler code, which reads/writes ``is_running``,
   ``app.capture_mouse(None)``, ``refresh``, ``call_after_refresh``, and
   two SHARED shell fields this cluster's own moved bodies never otherwise
   touch at all -- ``_library_canvas_projection_depth``/``_library_canvas_
   resync_pending`` (group (b) below, bound purely for this reason, exactly
   mirroring the RAG controller's own identical pair for the identical
   cause).

   **Fix round (post-landing review): ``focused`` was missing entirely --
   a SIXTH unbound-attribute-escape hazard, distinct in shape from
   exclusions 3e's bare-self-identity findings.** ``_sync_library_skills_
   browse_result`` reads ``focused = getattr(self, "focused", None)`` --
   a literal-name ``getattr`` with a default, exactly the shape the
   recipe's own census (a plain ``self.<attr>`` ``ast.Attribute`` walk)
   cannot see, because the attribute name never appears as a literal
   ``self.<name>`` expression. With no ``focused`` property bound, the
   call silently returns ``None`` on every invocation -- not an
   ``AttributeError``, not a wrong value, just quiet, permanent
   degradation of the two behaviors gated on it: the live-focus override
   (`if isinstance(focused, (Button, Input)) and ... startswith
   ("library-skills-"): focus_identity = live_focus_id`) and the live
   caret-position read for the filter input (a deliberate fix, commit
   ``8027e99f0``) both silently stop firing, so a committed-mutation
   refresh with ``focus_identity=None`` (the caller shape at this
   controller's own line ~1160, and at ``library_screen.py``'s
   ``restore_state``/``_select_library_rail_row_after_source_admission``
   call sites) can no longer recover focus from what the user was
   actually on. Fixed by adding the same live-read ``@property`` every
   other controller in this file already carries (``library_rag_search_
   controller.py``, ``library_conversations_controller.py``). A repeat,
   whole-file scan for EVERY ``getattr(self, "<literal-string>",
   default)`` call (not just ``ast.Attribute`` accesses) found exactly
   this one instance in the final 86-mover set -- confirmed, not assumed,
   the fix closes the whole class for this controller. See recipe §3's
   own updated bypass-shape catalogue for the general lesson.
2. **Everything else** the cluster depends on that is not its own state is a
   NAMED constructor dependency: (a) eight general Library-wide shell
   helpers a moved body calls with explicit arguments (``_run_library_
   service_call``, ``_sanitize_media_field``, ``_sanitize_note_content``,
   ``_refresh_local_source_snapshot`` -- one of recipe §3's four
   PERMANENTLY screen-routed monkeypatch names, reached here the same
   accessor-callable way every other subsystem already reaches it, never
   moved itself --, ``_library_entry_route_key``, ``_library_entry_
   reconcile_is_current``, ``_capture_library_entry_focus``, ``_restore_
   library_entry_focus``); (b) shared shell state this cluster reads
   (``_library_snapshot_state_generation``, ``_library_entry_reconcile_
   dirty``, ``_library_entry_reconcile_pending``, ``_library_canvas_
   projection_depth``) and reads+writes (``_library_canvas_resync_
   pending``, ``_library_selected_row_id`` -- written once, by ``_open_
   library_skill_editor_for_review``); (c) 2 WIRING accessor pairs for the
   pre-existing controllers this series deliberately leaves untouched
   (``_library_skill_import_coordinator``, ``_library_skills_browse_
   controller`` -- read+write, mirroring the collections controller's own
   ``_library_collections_capture_controller`` precedent exactly); (d) 5
   read-only accessors for the 6 merely-delegate import-coordinator
   properties (exclusion 1) that a MOVER reads (``_library_skills_import_
   in_flight``, ``_open``, ``_path``, ``_review_name``, ``_status``) plus
   one read+write pair for the sixth, which a mover WRITES (``_library_
   skills_import_generation``); (e) 10 named late-binding callables for
   the test-bypass/hazard exclusions (2-5 above) that a MOVER still calls
   internally -- ``_approve_library_skill_trust``, ``_begin_library_skill_
   save``, ``_build_library_skills_state``, ``_call_library_skill_trust_
   service``, ``_exit_library_skill_editor_guarded``, ``_persist_library_
   skill_editor_mode``, ``_refresh_library_skills_trust_posture``,
   ``_request_library_skills_browse``, ``_reset_library_skill_editor_
   state``, ``_start_library_skills_import`` -- each a ``lambda`` that
   re-reads ``screen.<name>`` on every invocation, at CALL time, not a
   value captured once at construction, which is exactly why a test's
   ``monkeypatch.setattr(screen, "<name>", ...)`` keeps working after this
   move. The remaining 4 exclusion-5/hazard names (``handle_library_
   skills_import``, ``_path_changed``, ``_retry``,
   ``_present_library_skills_import_snapshot``) have zero mover callers
   and need no binding at all.

**Construction order -- the usual position.** ``LibraryScreen.__init__``
builds ``self._skills_controller`` right after ``self._rag_search_
controller``, matching every other controller in this file.

This subsystem's OWN state (every ``_library_skill_<field>``/``_library_
skills_<field>``/``_selected_skill_name`` name the moved bodies reference)
is exposed through generated properties reading ``self._skills_state_
accessor().<field>`` -- the same generator shape task 1 installed on
``LibraryScreen`` (deleted at cleanup, task 3, once this controller's own
copy below makes the screen's copy dead) and the RAG controller's own
two-prefix split mirrors exactly: ``skill_state_shim_attr`` is imported
from ``library_skills_state`` -- the dataclass's own module, task 1's
single authoritative home for the three-way prefix mapping -- rather than
redefined as a second, independently-drifting literal copy.
"""
from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING

from loguru import logger
from textual import on
from textual.css.query import NoMatches, QueryError
from textual.widget import Widget
from textual.widgets import Button, Input, SelectionList, Static, TextArea

from ...Library.library_shell_state import LIBRARY_ROW_BROWSE_SKILLS
from ...Library.library_skills_state import (
    DEFAULT_SKILL_BROWSE_PAGE_SIZE,
    MAX_SKILL_BROWSE_PAGE,
    SkillBrowseResult,
    SkillBrowseScope,
    build_skill_editor_state,
    classify_skill_save_error,
    coerce_skill_reader_mode,
    compose_skill_markdown,
    reconcile_skill_allowed_tools,
    skill_allowed_tools_sequence,
    skill_invocation_copy,
    skill_review_identity_line,
)
from ...Utils.adaptive_reader_state import PANE_GRIP_WIDTH, resolve_adaptive_reader_layout
from ...config import coerce_bool_setting
from ...Widgets.Library import (
    LIBRARY_SKILLS_FILTER_ID,
    LIBRARY_SKILLS_PAGE_NEXT_ID,
    LIBRARY_SKILLS_PAGE_PREVIOUS_ID,
    LIBRARY_SKILLS_RETRY_ID,
    SKILL_DISCARD_TOOLTIP_CLEAN,
    SKILL_DISCARD_TOOLTIP_DIRTY,
    LibraryAdaptiveReaderShell,
    LibrarySkillWorkPane,
    LibrarySkillsListCanvas,
    next_skill_context,
    skill_context_toggle_label,
    skill_disable_model_label,
    skill_editor_warning_lines,
    skill_script_grant_line,
    skill_trust_approve_tooltip,
    skill_trust_panel_remediation_copy,
    skill_trust_review_enabled,
    skill_trust_review_preview,
    skill_trust_review_tooltip,
    skill_trust_state_line,
    skill_trust_unlock_enabled,
    skill_trust_unlock_tooltip,
    skill_user_invocable_label,
)
from .canvas_sync import _sync_library_canvas
from .library_skills_state import LibrarySkillsState, skill_state_shim_attr
from .screen_constants import (
    LIBRARY_SKILLS_IMPORT_WORKER_GROUP,
    LIBRARY_SKILLS_READER_PROFILE,
    LIBRARY_SKILL_DIRTY_VETO_COPY,
    LIBRARY_SKILL_SAVE_STATUS_COPY,
    LIBRARY_SKILL_TEXT_MAX_CHARS,
)
from .screen_helpers import _library_screen_is_current
from .screen_support_types import LibraryEntryReconcileResult
from ..Screens.skills_screen import SkillTrustBootstrapModal, SkillTrustPassphraseModal

if TYPE_CHECKING:
    from ..Screens.library_screen import LibraryScreen


class LibrarySkillsController:
    """Owns the Skills canvas cluster (86 methods).

    Holds no state of its own beyond what it reads and writes through
    ``LibrarySkillsState`` (via the injected accessor) and the shared
    shell/framework bindings below. ``LibraryScreen`` constructs exactly one
    of these, in ``__init__`` right after ``self._rag_search_controller``,
    and keeps one-line delegators for every original name this cluster
    moved (86 -- see the module docstring for the full derivation and the
    41 exclusions).
    """

    def __init__(
        self,
        screen: "LibraryScreen",
        *,
        skills_state_accessor,
        # -- wiring accessors for the two pre-existing, untouched Skills
        # controllers (group (c)).
        library_skill_import_coordinator_accessor,
        set_library_skill_import_coordinator,
        library_skills_browse_controller_accessor,
        set_library_skills_browse_controller,
        # -- general Library-wide shell helpers, not moved (group (a)).
        run_library_service_call,
        sanitize_media_field,
        sanitize_note_content,
        refresh_local_source_snapshot,
        library_entry_route_key,
        library_entry_reconcile_is_current,
        capture_library_entry_focus,
        restore_library_entry_focus,
        # -- shared shell state this cluster reads/writes (group (b)).
        library_selected_row_id_accessor,
        set_library_selected_row_id,
        library_snapshot_state_generation_accessor,
        library_entry_reconcile_dirty_accessor,
        library_entry_reconcile_pending_accessor,
        library_canvas_projection_depth_accessor,
        library_canvas_resync_pending_accessor,
        set_library_canvas_resync_pending,
        # -- read-only accessors for the 6 merely-delegate import-
        # coordinator properties (exclusion 1) a mover reads; one is
        # read+write since a mover WRITES it (group (d)).
        library_skills_import_open_accessor,
        library_skills_import_path_accessor,
        library_skills_import_status_accessor,
        library_skills_import_review_name_accessor,
        library_skills_import_in_flight_accessor,
        library_skills_import_generation_accessor,
        set_library_skills_import_generation,
        # -- 12 of the test-bypass/hazard exclusions (2-5 above) that a
        # MOVER still calls internally, each a late-binding callable
        # (group (e)); 3 more (`handle_library_skills_import`, `_path_
        # changed`, `_retry`) are ALSO excluded (hazard 5) but have zero
        # internal mover callers -- @on-dispatched only -- so need no
        # binding here at all.
        approve_library_skill_trust,
        begin_library_skill_save,
        build_library_skills_state,
        call_library_skill_trust_service,
        exit_library_skill_editor_guarded,
        persist_library_skill_editor_mode,
        refresh_library_skills_trust_posture,
        request_library_skills_browse,
        reset_library_skill_editor_state,
        start_library_skills_import,
    ) -> None:
        """Build the controller and bind everything its moved bodies need.

        Every one of the 86 method bodies below is a byte-for-byte copy of
        the pre-extraction ``LibraryScreen`` method: no internal line was
        edited to retarget a call or an attribute. That is possible because
        this constructor binds every name those bodies reference that is
        not this controller's own state, under the SAME name the original
        method used. See the module docstring for the binding kinds this
        follows and the full per-parameter derivation.

        Args:
            screen: The Library screen. Used ONLY for the ten framework
                services below (``app``, ``app_instance``, ``call_after_
                refresh``, ``focused``, ``is_mounted``, ``is_running``,
                ``query``, ``query_one``, ``refresh``, ``run_worker``) --
                this cluster owns no DOM of its own.
            skills_state_accessor: Returns the live ``LibrarySkillsState``
                (``LibraryScreen._skills_state``, task 1). Backs every
                generated ``_library_skill_<field>``/``_library_skills_
                <field>``/``_selected_skill_name`` property below.
        """
        self._screen = screen
        self._skills_state_accessor = skills_state_accessor
        self._library_skill_import_coordinator_accessor = (
            library_skill_import_coordinator_accessor
        )
        self._set_library_skill_import_coordinator_fn = (
            set_library_skill_import_coordinator
        )
        self._library_skills_browse_controller_accessor = (
            library_skills_browse_controller_accessor
        )
        self._set_library_skills_browse_controller_fn = (
            set_library_skills_browse_controller
        )
        self._run_library_service_call_fn = run_library_service_call
        self._sanitize_media_field_fn = sanitize_media_field
        self._sanitize_note_content_fn = sanitize_note_content
        self._refresh_local_source_snapshot_fn = refresh_local_source_snapshot
        self._library_entry_route_key_fn = library_entry_route_key
        self._library_entry_reconcile_is_current_fn = (
            library_entry_reconcile_is_current
        )
        self._capture_library_entry_focus_fn = capture_library_entry_focus
        self._restore_library_entry_focus_fn = restore_library_entry_focus
        self._library_selected_row_id_accessor = library_selected_row_id_accessor
        self._set_library_selected_row_id_fn = set_library_selected_row_id
        self._library_snapshot_state_generation_accessor = (
            library_snapshot_state_generation_accessor
        )
        self._library_entry_reconcile_dirty_accessor = (
            library_entry_reconcile_dirty_accessor
        )
        self._library_entry_reconcile_pending_accessor = (
            library_entry_reconcile_pending_accessor
        )
        self._library_canvas_projection_depth_accessor = (
            library_canvas_projection_depth_accessor
        )
        self._library_canvas_resync_pending_accessor = (
            library_canvas_resync_pending_accessor
        )
        self._set_library_canvas_resync_pending_fn = set_library_canvas_resync_pending
        self._library_skills_import_open_accessor = library_skills_import_open_accessor
        self._library_skills_import_path_accessor = library_skills_import_path_accessor
        self._library_skills_import_status_accessor = (
            library_skills_import_status_accessor
        )
        self._library_skills_import_review_name_accessor = (
            library_skills_import_review_name_accessor
        )
        self._library_skills_import_in_flight_accessor = (
            library_skills_import_in_flight_accessor
        )
        self._library_skills_import_generation_accessor = (
            library_skills_import_generation_accessor
        )
        self._set_library_skills_import_generation_fn = (
            set_library_skills_import_generation
        )
        self._approve_library_skill_trust_fn = approve_library_skill_trust
        self._begin_library_skill_save_fn = begin_library_skill_save
        self._build_library_skills_state_fn = build_library_skills_state
        self._call_library_skill_trust_service_fn = call_library_skill_trust_service
        self._exit_library_skill_editor_guarded_fn = exit_library_skill_editor_guarded
        self._persist_library_skill_editor_mode_fn = persist_library_skill_editor_mode
        self._refresh_library_skills_trust_posture_fn = (
            refresh_library_skills_trust_posture
        )
        self._request_library_skills_browse_fn = request_library_skills_browse
        self._reset_library_skill_editor_state_fn = reset_library_skill_editor_state
        self._start_library_skills_import_fn = start_library_skills_import

    # -- framework services: live-read properties, never snapshotted -----

    @property
    def app(self) -> Any:
        return self._screen.app

    @property
    def app_instance(self) -> Any:
        return self._screen.app_instance

    @property
    def call_after_refresh(self) -> Any:
        return self._screen.call_after_refresh

    @property
    def focused(self) -> Any:
        return self._screen.focused

    @property
    def is_mounted(self) -> bool:
        return self._screen.is_mounted

    @property
    def is_running(self) -> bool:
        return self._screen.is_running

    @property
    def query(self) -> Any:
        return self._screen.query

    @property
    def query_one(self) -> Any:
        return self._screen.query_one

    @property
    def refresh(self) -> Any:
        return self._screen.refresh

    @property
    def run_worker(self) -> Any:
        return self._screen.run_worker

    # -- shared shell state (group (b)) ------------------------------------

    @property
    def _library_selected_row_id(self) -> str:
        return self._library_selected_row_id_accessor()

    @_library_selected_row_id.setter
    def _library_selected_row_id(self, value: str) -> None:
        self._set_library_selected_row_id_fn(value)

    @property
    def _library_snapshot_state_generation(self) -> int:
        return self._library_snapshot_state_generation_accessor()

    @property
    def _library_entry_reconcile_dirty(self) -> bool:
        return self._library_entry_reconcile_dirty_accessor()

    @property
    def _library_entry_reconcile_pending(self) -> Any:
        return self._library_entry_reconcile_pending_accessor()

    @property
    def _library_canvas_projection_depth(self) -> int:
        return self._library_canvas_projection_depth_accessor()

    @property
    def _library_canvas_resync_pending(self) -> bool:
        return self._library_canvas_resync_pending_accessor()

    @_library_canvas_resync_pending.setter
    def _library_canvas_resync_pending(self, value: bool) -> None:
        self._set_library_canvas_resync_pending_fn(value)

    # -- wiring accessors (group (c)) --------------------------------------

    @property
    def _library_skill_import_coordinator(self) -> Any:
        return self._library_skill_import_coordinator_accessor()

    @_library_skill_import_coordinator.setter
    def _library_skill_import_coordinator(self, value: Any) -> None:
        self._set_library_skill_import_coordinator_fn(value)

    @property
    def _library_skills_browse_controller(self) -> Any:
        return self._library_skills_browse_controller_accessor()

    @_library_skills_browse_controller.setter
    def _library_skills_browse_controller(self, value: Any) -> None:
        self._set_library_skills_browse_controller_fn(value)

    # -- read accessors for the 6 merely-delegate import-coordinator
    # properties (exclusion 1); one read+write since a mover WRITES it
    # (group (d)) ------------------------------------------------------

    @property
    def _library_skills_import_open(self) -> bool:
        return self._library_skills_import_open_accessor()

    @property
    def _library_skills_import_path(self) -> str:
        return self._library_skills_import_path_accessor()

    @property
    def _library_skills_import_status(self) -> str:
        return self._library_skills_import_status_accessor()

    @property
    def _library_skills_import_review_name(self) -> str:
        return self._library_skills_import_review_name_accessor()

    @property
    def _library_skills_import_in_flight(self) -> bool:
        return self._library_skills_import_in_flight_accessor()

    @property
    def _library_skills_import_generation(self) -> int:
        return self._library_skills_import_generation_accessor()

    @_library_skills_import_generation.setter
    def _library_skills_import_generation(self, value: int) -> None:
        self._set_library_skills_import_generation_fn(value)

    # -- general Library-wide shell helpers (group (a)) --------------------

    @property
    def _run_library_service_call(self) -> Any:
        return self._run_library_service_call_fn

    @property
    def _sanitize_media_field(self) -> Any:
        return self._sanitize_media_field_fn

    @property
    def _sanitize_note_content(self) -> Any:
        return self._sanitize_note_content_fn

    @property
    def _refresh_local_source_snapshot(self) -> Any:
        return self._refresh_local_source_snapshot_fn

    @property
    def _library_entry_route_key(self) -> Any:
        return self._library_entry_route_key_fn

    @property
    def _library_entry_reconcile_is_current(self) -> Any:
        return self._library_entry_reconcile_is_current_fn

    @property
    def _capture_library_entry_focus(self) -> Any:
        return self._capture_library_entry_focus_fn

    @property
    def _restore_library_entry_focus(self) -> Any:
        return self._restore_library_entry_focus_fn

    # -- named late-binding callables for the test-bypass/hazard
    # exclusions (group (e)) ------------------------------------------

    @property
    def _approve_library_skill_trust(self) -> Any:
        return self._approve_library_skill_trust_fn

    @property
    def _begin_library_skill_save(self) -> Any:
        return self._begin_library_skill_save_fn

    @property
    def _build_library_skills_state(self) -> Any:
        return self._build_library_skills_state_fn

    @property
    def _call_library_skill_trust_service(self) -> Any:
        return self._call_library_skill_trust_service_fn

    @property
    def _exit_library_skill_editor_guarded(self) -> Any:
        return self._exit_library_skill_editor_guarded_fn

    @property
    def _persist_library_skill_editor_mode(self) -> Any:
        return self._persist_library_skill_editor_mode_fn

    @property
    def _refresh_library_skills_trust_posture(self) -> Any:
        return self._refresh_library_skills_trust_posture_fn

    @property
    def _request_library_skills_browse(self) -> Any:
        return self._request_library_skills_browse_fn

    @property
    def _reset_library_skill_editor_state(self) -> Any:
        return self._reset_library_skill_editor_state_fn

    @property
    def _start_library_skills_import(self) -> Any:
        return self._start_library_skills_import_fn

    # -- moved cluster methods (86), byte-for-byte, original file order ---
    def _sync_library_skills_reader_layout_from_shell(
        self,
        priority: Literal["library", "items"] | None = None,
    ) -> None:
        """Resolve the settled Skills shell and patch it in place."""
        try:
            shell = self.query_one(
                "#library-skills-reader-shell", LibraryAdaptiveReaderShell
            )
        except (NoMatches, QueryError):
            return
        width = shell.region.width
        if width <= 0:
            return
        previous = self._library_skills_reader_layout
        if (
            previous.reader_width == 0
            and previous.library_width == 0
            and previous.items_width == 0
        ):
            previous = None
        if (
            priority is None
            and self._library_skills_view == "list"
            and self._library_skills_reader_preferences.items_open
        ):
            items_priority_floor = (
                2 * PANE_GRIP_WIDTH
                + LIBRARY_SKILLS_READER_PROFILE.list_min_width
                + LIBRARY_SKILLS_READER_PROFILE.work_min_width
            )
            if width >= items_priority_floor:
                priority = "items"
        elif (
            priority is None
            and previous is not None
            and previous.priority_pane == "items"
        ):
            previous = dataclasses.replace(previous, priority_pane=None)
        layout = resolve_adaptive_reader_layout(
            width,
            self._library_skills_reader_preferences,
            LIBRARY_SKILLS_READER_PROFILE,
            previous=previous,
            priority=priority,
        )
        shell.sync_layout(layout)
        self._library_skills_reader_layout = layout

    def _mirror_library_skills_reader_preference(
        self,
        key: Literal["library_open", "items_open"],
        value: bool,
    ) -> None:
        """Mirror one optimistic Skills pane choice into app config."""
        app_config = getattr(self.app_instance, "app_config", None)
        if not isinstance(app_config, dict):
            return
        library_config = app_config.setdefault("library", {})
        if not isinstance(library_config, dict):
            library_config = {}
            app_config["library"] = library_config
        section_name = "reader" if key == "library_open" else "skills_reader"
        section = library_config.setdefault(section_name, {})
        if not isinstance(section, dict):
            section = {}
            library_config[section_name] = section
        section[key] = value

    @staticmethod
    def _restore_library_skills_scope(state: Mapping[str, Any]) -> SkillBrowseScope:
        """Return a dispatch-safe applied Skills scope from screen state."""
        saved = state.get("library_skills_scope")
        if isinstance(saved, SkillBrowseScope):
            raw = dataclasses.asdict(saved)
        elif type(saved) is dict:
            raw = saved
        else:
            raw = {
                "query": state.get("library_skills_filter", ""),
                "sort": state.get("library_skills_sort", "name"),
                "page": 1,
            }
        query = raw.get("query", "")
        page = raw.get("page", 1)
        if type(query) is not str:
            query = ""
        if type(page) is not int or not 1 <= page <= MAX_SKILL_BROWSE_PAGE:
            page = 1
        try:
            return SkillBrowseScope(
                query=query,
                sort=raw.get("sort", "name"),
                page=page,
                page_size=DEFAULT_SKILL_BROWSE_PAGE_SIZE,
            )
        except (TypeError, ValueError):
            return SkillBrowseScope()

    async def _skills_context_or_none(
        self, get_context: Any, **kwargs: Any
    ) -> Mapping[str, Any] | None:
        """Fetch the local skills context, degrading quietly on failure.

        Runs inside the same ``asyncio.gather`` as the notes/media/
        conversations/prompts fetch (see ``_list_local_source_snapshot``).
        Mirrors ``_prompts_count_or_none``: the seam is optional (guarded by
        ``callable(get_context)`` at the call site), so when it is missing
        this method is never invoked, and when it *is* present but raises,
        the failure is swallowed and ``None`` is returned -- the Skills
        rail row then renders uncounted with an empty context payload
        rather than surfacing an error or failing the whole snapshot
        fetch.

        Unlike prompts (whose rail count and exact browse result have
        separate owners), a single ``get_context`` call here supplies both:
        the count is derived from its
        ``available_skills``/``blocked_skills`` lengths by the caller, and
        the same payload is stashed for a future Skills canvas to render.

        Args:
            get_context: The bound ``skills_scope_service.get_context``
                callable to invoke.
            **kwargs: Forwarded to ``get_context`` (``mode``).

        Returns:
            The normalized ``get_context`` payload (``available_skills`` +
            ``blocked_skills``), or ``None`` if the call failed or returned
            something other than a ``Mapping``.
        """
        try:
            result = await self._run_library_service_call(
                get_context, isolate_in_worker=True, **kwargs
            )
        except Exception:
            logger.opt(exception=True).warning(
                "Failed to fetch local skills context; Skills row will show no count."
            )
            return None
        return result if isinstance(result, Mapping) else None

    def _build_library_skill_tool_catalog(self) -> tuple[str, ...]:
        """Return the existing builtin/local tool names a Skill may restrict."""
        try:
            from ...Agents.tool_catalog import BuiltinToolProvider

            names = [entry.name for entry in BuiltinToolProvider().list_catalog()]
            console_config = getattr(self.app_instance, "app_config", {}).get(
                "console", {}
            )
            local_enabled = bool(
                isinstance(console_config, Mapping)
                and coerce_bool_setting(
                    console_config.get("local_tools_enabled"), False
                )
            )
            if local_enabled:
                from ...Agents.local_tool_provider import LocalToolProvider

                configured_root = console_config.get("workspace_root")
                workspace_root = (
                    Path(str(configured_root)).expanduser()
                    if configured_root
                    else Path.cwd()
                )
                names.extend(
                    entry.name
                    for entry in LocalToolProvider(
                        workspace_root=workspace_root
                    ).list_catalog()
                )
        except Exception:
            logger.warning("Library Skill tool catalog is temporarily unavailable.")
            return ()
        return tuple(dict.fromkeys(names))

    def _library_skills_list_canvas_kwargs(self) -> dict[str, Any]:
        """Return list-only inputs for the retained Skills Items pane."""
        values = self._library_skills_canvas_kwargs(consume_scroll=False)
        values.update(
            state=self._build_library_skills_state(),
            mode="list",
            editor_state=None,
            warnings="",
            status="",
            conflict=False,
            active_review=None,
            confirming_delete=False,
            scroll_to_actions=False,
            skill_path="",
            import_open=False,
            import_path="",
            import_status="",
            import_review_name="",
            mutation_in_flight=False,
            more_actions_open=False,
            trust_details_open=False,
            script_access_granted=False,
        )
        return values

    def _library_skill_work_pane_kwargs(self) -> dict[str, Any]:
        """Return the active non-list projection for the Skills Work pane."""
        values = self._library_skills_canvas_kwargs()
        if self._library_skills_import_open:
            values["mode"] = "list"
        values["state"] = None
        values["sort_choices_visible"] = False
        values["reader_mode"] = self._library_skill_reader_mode
        return values

    def _library_skills_canvas_kwargs(
        self, *, consume_scroll: bool = True
    ) -> dict[str, Any]:
        """Return every compose input for the mounted Skills canvas."""
        values: dict[str, Any] = {
            "state": None,
            "sort_mode": self._library_skills_sort,
            "filter_value": self._library_skills_filter,
            "mode": "list",
            "trust_posture": self._library_skills_trust_posture,
            "confirming_reset": self._library_skill_trust_confirming_reset,
            "editor_state": None,
            "warnings": "",
            "status": "",
            "conflict": False,
            "active_review": None,
            "is_create": not self._selected_skill_name,
            "dirty": self._library_skill_dirty,
            "confirming_delete": self._library_skill_confirming_delete,
            "scroll_to_actions": False,
            "skill_path": "",
            "import_open": False,
            "import_path": "",
            "import_status": "",
            "import_review_name": "",
            "import_in_flight": False,
            "import_package_kind": "",
            "import_recovery_actions": (),
            "import_retryable": False,
            "sort_choices_visible": False,
            "editor_mode": self._library_skill_editor_mode,
            "tool_catalog": self._library_skill_tool_catalog,
            "tool_filter": self._library_skill_tool_filter,
            "mutation_in_flight": self._library_skill_mutation_in_flight,
            "more_actions_open": self._library_skill_more_actions_open,
            "trust_details_open": self._library_skill_trust_details_open,
            "script_access_granted": self._library_skill_script_grant,
            "detail_notice": "",
            "detail_retryable": False,
        }
        if self._library_skills_view == "editor":
            editor_state = self._library_skill_editor_state
            if editor_state is None:
                values["mode"] = "loading"
                values["detail_notice"] = (
                    self._library_skill_detail_error or "Loading skill…"
                )
                values["detail_retryable"] = self._library_skill_detail_retryable
            else:
                values.update(
                    {
                        "mode": "editor",
                        "editor_state": editor_state,
                        "warnings": "\n".join(
                            skill_editor_warning_lines(
                                live_name=editor_state.name,
                                trust_status=editor_state.trust_status,
                                trust_blocked=editor_state.trust_blocked,
                            )
                        ),
                        "status": self._library_skill_status,
                        "conflict": self._library_skill_conflict,
                        "active_review": self._library_skill_active_review,
                        "scroll_to_actions": (
                            self._consume_library_skill_scroll_pending()
                            if consume_scroll
                            else False
                        ),
                        "skill_path": self._library_skill_on_disk_path(),
                    }
                )
            return values

        values.update(
            {
                "state": self._build_library_skills_state(),
                "import_open": self._library_skills_import_open,
                "import_path": self._library_skills_import_path,
                "import_status": self._library_skills_import_status,
                "import_review_name": self._library_skills_import_review_name,
                "import_in_flight": self._library_skills_import_in_flight,
                "import_package_kind": (
                    self._library_skill_import_coordinator.snapshot.package_kind
                ),
                "import_recovery_actions": (
                    self._library_skill_import_coordinator.snapshot.recovery_actions
                ),
                "import_retryable": (
                    self._library_skill_import_coordinator.snapshot.retryable
                ),
                "sort_choices_visible": self._library_skills_sort_choices_visible,
            }
        )
        return values

    def _sync_library_skills_browse_result(
        self,
        result: SkillBrowseResult,
        focus_identity: str | None,
    ) -> LibraryEntryReconcileResult:
        """Project one accepted Skills generation and restore stable focus."""
        controller = self._library_skills_browse_controller
        if (
            result.request_token != controller.result.request_token
            or self._library_selected_row_id != LIBRARY_ROW_BROWSE_SKILLS
            or self._library_skills_view != "list"
        ):
            return LibraryEntryReconcileResult.SUPERSEDED

        self._library_skills_filter = controller.scope.query
        self._library_skills_sort = controller.scope.sort
        focused = getattr(self, "focused", None)
        live_focus_id = getattr(focused, "id", None)
        if (
            isinstance(focused, (Button, Input))
            and isinstance(live_focus_id, str)
            and live_focus_id.startswith("library-skills-")
        ):
            focus_identity = live_focus_id
        cursor_context = self._library_skills_filter_cursor_context
        cursor_position = (
            focused.cursor_position
            if isinstance(focused, Input) and live_focus_id == LIBRARY_SKILLS_FILTER_ID
            else cursor_context[1]
            if cursor_context is not None and cursor_context[0] == result.request_token
            else None
        )

        def restore_focus() -> None:
            if not focus_identity:
                return
            if result.status == "loading" and focus_identity in {
                LIBRARY_SKILLS_PAGE_PREVIOUS_ID,
                LIBRARY_SKILLS_PAGE_NEXT_ID,
                LIBRARY_SKILLS_RETRY_ID,
            }:
                return
            if focus_identity in {
                LIBRARY_SKILLS_PAGE_PREVIOUS_ID,
                LIBRARY_SKILLS_PAGE_NEXT_ID,
                LIBRARY_SKILLS_RETRY_ID,
            }:
                self._focus_library_skills_page_control(focus_identity)
                return
            try:
                target = self.query_one(f"#{focus_identity}", Widget)
            except (NoMatches, QueryError):
                return
            if not getattr(target, "disabled", False):
                target.focus()
                if cursor_position is not None and isinstance(target, Input):
                    target.cursor_position = cursor_position

        if _sync_library_canvas(
            self,
            "skills",
            then=restore_focus,
            allow_screen_fallback=False,
        ):
            return LibraryEntryReconcileResult.APPLIED
        return LibraryEntryReconcileResult.FAILED

    def _focus_library_skills_page_control(self, invoked: str) -> None:
        """Restore pager focus without landing on a disabled control."""
        opposite = {
            LIBRARY_SKILLS_PAGE_PREVIOUS_ID: LIBRARY_SKILLS_PAGE_NEXT_ID,
            LIBRARY_SKILLS_PAGE_NEXT_ID: LIBRARY_SKILLS_PAGE_PREVIOUS_ID,
            LIBRARY_SKILLS_RETRY_ID: LIBRARY_SKILLS_FILTER_ID,
        }[invoked]
        for control_id in (invoked, opposite, LIBRARY_SKILLS_FILTER_ID):
            try:
                control = self.query_one(f"#{control_id}", Widget)
            except (NoMatches, QueryError):
                continue
            if not getattr(control, "disabled", False):
                control.focus()
                return

    def _refresh_library_skills_after_committed_mutation(
        self,
        *,
        scope: SkillBrowseScope | None = None,
    ) -> None:
        """Invalidate exact totals after a durable Skill or trust change.

        Retained rows may remain visible, but they are explicitly stale and
        inert until the source-owned page refresh succeeds.
        """
        controller = self._library_skills_browse_controller
        refresh_scope = scope or controller.mutation_refresh_scope
        if controller.applied_result is not None:
            controller.retain_stale_items(
                controller.retained_items,
                stale_copy="Skills changed; refresh the page before acting.",
            )
        else:
            controller.invalidate(refresh_scope)
        if (
            self._library_selected_row_id == LIBRARY_ROW_BROWSE_SKILLS
            and self._library_skills_view == "list"
        ):
            self._request_library_skills_browse(refresh_scope)

    async def _load_library_skills_trust_posture(
        self, posture_fn
    ) -> LibraryEntryReconcileResult:
        """Await the off-thread ``trust_posture()`` call and apply the result.

        Args:
            posture_fn: The trust service's bound ``trust_posture`` method
                (captured by ``_refresh_library_skills_trust_posture`` so
                this never re-reads ``local_skill_trust_service`` itself --
                irrelevant here, but keeps this a pure "run this callable
                off-thread" step).
        """
        route_key = self._library_entry_route_key()
        generation = self._library_snapshot_state_generation
        try:
            posture = await asyncio.to_thread(posture_fn)
        except Exception:
            posture = ""
        self._library_skills_trust_posture = posture if isinstance(posture, str) else ""
        if (
            route_key != self._library_entry_route_key()
            or not self._library_entry_reconcile_is_current(generation, route_key)
            or self._library_selected_row_id != LIBRARY_ROW_BROWSE_SKILLS
            or self._library_skills_view != "list"
        ):
            return LibraryEntryReconcileResult.SUPERSEDED
        identity = self._capture_library_entry_focus()
        prior_callback: Callable[[], None] | None = None
        pending = (generation, route_key)
        if (
            self._library_entry_reconcile_dirty
            and self._library_entry_reconcile_pending == pending
        ):
            try:
                canvas = self.query_one(
                    "#library-skills-canvas", LibrarySkillsListCanvas
                )
            except (NoMatches, QueryError):
                canvas = None
            if canvas is not None:
                prior_callback = canvas._post_recompose_callback

        def finish_posture_sync() -> None:
            if prior_callback is not None:
                prior_callback()
            if identity is not None:
                self._restore_library_entry_focus(
                    identity,
                    generation=generation,
                    route_key=route_key,
                )

        follow_up = (
            finish_posture_sync
            if prior_callback is not None or identity is not None
            else None
        )
        if _sync_library_canvas(
            self,
            "skills",
            then=follow_up,
            allow_screen_fallback=False,
        ):
            return LibraryEntryReconcileResult.APPLIED
        return LibraryEntryReconcileResult.FAILED

    @on(Button.Pressed, f"#{LIBRARY_SKILLS_PAGE_PREVIOUS_ID}")
    def handle_library_skills_page_previous(self, event: Button.Pressed) -> None:
        """Request the preceding exact Skills page.

        Args:
            event: Button press event emitted by the Previous control.
        """
        event.stop()
        controller = self._library_skills_browse_controller
        applied = controller.applied_result
        if applied is None or applied.page <= 1 or controller.freshness != "fresh":
            return
        self._request_library_skills_browse(
            controller.scope_for_page(applied.page - 1),
            focus_identity=LIBRARY_SKILLS_PAGE_PREVIOUS_ID,
        )

    @on(Button.Pressed, f"#{LIBRARY_SKILLS_PAGE_NEXT_ID}")
    def handle_library_skills_page_next(self, event: Button.Pressed) -> None:
        """Request the following exact Skills page.

        Args:
            event: Button press event emitted by the Next control.
        """
        event.stop()
        controller = self._library_skills_browse_controller
        applied = controller.applied_result
        if (
            applied is None
            or applied.page >= applied.total_pages
            or controller.freshness != "fresh"
        ):
            return
        self._request_library_skills_browse(
            controller.scope_for_page(applied.page + 1),
            focus_identity=LIBRARY_SKILLS_PAGE_NEXT_ID,
        )

    @on(Button.Pressed, f"#{LIBRARY_SKILLS_RETRY_ID}")
    def handle_library_skills_retry(self, event: Button.Pressed) -> None:
        """Retry the last requested Skills page or stale-page refresh.

        Args:
            event: Button press event emitted by the Retry control.
        """
        event.stop()
        self._request_library_skills_browse(
            self._library_skills_browse_controller.scope,
            focus_identity=LIBRARY_SKILLS_RETRY_ID,
        )

    @on(Input.Submitted, "#library-skills-import-path")
    def handle_library_skills_import_path_submitted(
        self, event: Input.Submitted
    ) -> None:
        """Run the import when Enter is pressed in the Import row's path field.

        Args:
            event: Input submission event emitted by the Import row's
                path field.
        """
        event.stop()
        self._start_library_skills_import()

    @on(Button.Pressed, "#library-skills-import-run")
    def handle_library_skills_import_run(self, event: Button.Pressed) -> None:
        """Run the import when the Import row's "Import" action is pressed.

        Args:
            event: Button press event emitted by the Import row's
                "Import" action.
        """
        event.stop()
        self._start_library_skills_import()

    def _claim_library_skill_detail_generation(self) -> int:
        """Start one Skill detail request and return its settlement fence."""
        self._library_skill_detail_generation += 1
        self._library_skill_detail_loading = True
        self._library_skill_detail_error = ""
        self._library_skill_detail_retryable = False
        return self._library_skill_detail_generation

    def _invalidate_library_skill_detail_generation(self) -> None:
        """Refuse every pending Skill detail settlement."""
        self._library_skill_detail_generation += 1
        self._library_skill_detail_loading = False
        self._library_skill_detail_error = ""
        self._library_skill_detail_retryable = False

    def _library_skill_detail_request_is_current(
        self, *, skill_name: str, generation: int
    ) -> bool:
        """Return whether one detail outcome still owns the Skills Work pane."""
        return bool(
            generation == self._library_skill_detail_generation
            and skill_name == self._selected_skill_name
            and self._library_skills_view == "editor"
        )

    def _apply_library_skill_detail_failure(self, copy: str) -> None:
        """Keep the selected Skill in place and expose a scoped retry."""
        self._library_skill_detail_loading = False
        self._library_skill_detail_error = copy
        self._library_skill_detail_retryable = True
        if self.is_mounted:
            _sync_library_canvas(self, "skills")

    async def _refresh_library_skill_detail(
        self,
        skill_name: str,
        *,
        request_generation: int | None = None,
    ) -> None:
        """Fetch and store the full detail for a selected Library skill.

        Mirrors ``_refresh_library_prompt_detail``: offloads the (possibly
        blocking) ``get_skill`` service call via ``_run_library_service_call``
        and recomposes once the fetched detail (or a cleared state) has
        been stored.

        Args:
            skill_name: The Library skill name to fetch full detail for.
        """
        if request_generation is None:
            request_generation = self._claim_library_skill_detail_generation()
        service = getattr(self.app_instance, "skills_scope_service", None)
        get_skill = getattr(service, "get_skill", None)
        if not callable(get_skill):
            if self._library_skill_detail_request_is_current(
                skill_name=skill_name, generation=request_generation
            ):
                self._apply_library_skill_detail_failure(
                    "Couldn’t load the selected Skill. The local service is unavailable."
                )
            return
        failed = False
        try:
            detail = await self._run_library_service_call(
                get_skill,
                skill_name,
                mode="local",
                isolate_in_worker=True,
            )
        except Exception:
            logger.opt(exception=True).warning(
                f"Failed to load Library skill detail for {skill_name!r}."
            )
            detail = None
            failed = True
        # Discard out-of-order results: the same stale-race guard as
        # ``_refresh_library_prompt_detail``.
        if not self._library_skill_detail_request_is_current(
            skill_name=skill_name, generation=request_generation
        ):
            return
        if not isinstance(detail, Mapping):
            self._apply_library_skill_detail_failure(
                "Couldn’t load the selected Skill. Try again."
                if failed
                else "This Skill is no longer available. Refresh the list and try again."
            )
            return
        self._library_skill_detail_loading = False
        self._library_skill_detail_error = ""
        self._library_skill_detail_retryable = False
        self._apply_library_skill_detail(detail)

    def _apply_library_skill_detail(self, detail: Mapping[str, Any]) -> None:
        """Store a freshly-fetched skill detail and (re)render the editor.

        Shared by the initial open (``_refresh_library_skill_detail``) and
        a successful Save (whose response mapping is already a full detail
        -- see ``_save_library_skill``'s docstring for why no separate
        "refresh snapshot" fetch is needed there).

        Args:
            detail: A skill detail mapping shaped like ``get_skill``'s (or
                a save call's) response.
        """
        self._library_skill_detail = dict(detail)
        self._library_skill_detail_loading = False
        self._library_skill_detail_error = ""
        self._library_skill_detail_retryable = False
        self._library_skill_editor_state = build_skill_editor_state(
            self._library_skill_detail
        )
        self._library_skill_original_name = self._library_skill_editor_state.name
        self._library_skill_dirty = False
        self._library_skill_status = ""
        self._library_skill_conflict = False
        self._library_skill_active_review = None
        self._library_skill_script_grant = False
        self._library_skill_tool_catalog = self._build_library_skill_tool_catalog()
        self._library_skill_tool_filter = ""
        self._library_skill_tool_captured = skill_allowed_tools_sequence(
            self._library_skill_editor_state.allowed_tools_csv
        )
        self._library_skill_tool_picker_changed = False
        self._library_skill_more_actions_open = False
        self._library_skill_trust_details_open = False
        self._library_skill_mutation_in_flight = False
        self._library_skill_editor_armed = False
        if self.is_mounted:
            # Deterministic (task-15457 review I4b): arming is
            # dirty-tracking, not cosmetics -- a follow-up lost to the
            # canvas-pump race leaves the editor unarmed, so it rides the
            # canvas's own post-recompose hook rather than the screen's.
            _sync_library_canvas(self, "skills", then=self._arm_library_skill_editor)
        # Task 7: not part of ``get_skill``'s response, so it needs its own
        # off-thread fetch -- see ``_refresh_library_skill_script_grant``.
        self._refresh_library_skill_script_grant()

    def _arm_library_skill_editor(self) -> None:
        """Enable dirty-tracking once the skill editor's mount-time
        ``Input.Changed``/``TextArea.Changed`` (fired for the non-empty
        initial values) has already been delivered, so it is never mistaken
        for a real edit.
        """
        self._library_skill_editor_armed = True

    def _enter_library_skill_create_editor(self) -> None:
        """Open the in-canvas skill editor on a blank, not-yet-saved record.

        Entered via the Create rail's "New skill" row
        (``LIBRARY_ROW_CREATE_SKILL``, whose ``target_id`` is ``"skills"``
        -- the SAME canvas kind Browse > Skills targets), mirroring
        ``_enter_library_prompt_create_editor``'s "New prompt" row.

        ``_selected_skill_name`` stays ``""``: the sentinel
        ``_save_library_skill`` already reads (``is_create = not name``)
        to route its scope-service ``create_skill`` call instead of
        ``update_skill``, and the sentinel ``compose_content`` reads
        (``is_create=not self._selected_skill_name``) to keep the Name
        Input editable (an existing skill's Name Input is disabled --
        there is no rename primitive).

        ``_library_skill_editor_state`` is built directly from an empty
        mapping (``build_skill_editor_state({})``) rather than left
        ``None``: ``compose_content``'s skills-editor branch gates on
        ``_library_skill_editor_state is None`` to show a "Loading
        skill…" placeholder while the async detail fetch
        (``_refresh_library_skill_detail``) is in flight -- there is no
        fetch for a brand-new record, so leaving it ``None`` would show
        that placeholder forever.
        """
        self._selected_skill_name = ""
        self._library_skills_view = "editor"
        self._library_skill_reader_mode = "edit"
        self._library_skill_detail = {}
        self._invalidate_library_skill_detail_generation()
        self._library_skill_editor_state = build_skill_editor_state({})
        self._library_skill_original_name = ""
        self._library_skill_dirty = False
        self._library_skill_status = ""
        self._library_skill_conflict = False
        self._library_skill_active_review = None
        self._library_skill_script_grant = False
        self._library_skill_tool_catalog = self._build_library_skill_tool_catalog()
        self._library_skill_tool_filter = ""
        self._library_skill_tool_captured = ()
        self._library_skill_tool_picker_changed = False
        self._library_skill_more_actions_open = False
        self._library_skill_trust_details_open = False
        self._library_skill_mutation_in_flight = False
        self._library_skill_confirming_delete = False
        self._library_skill_scroll_pending = False
        self._library_skill_editor_armed = False

    def _consume_library_skill_scroll_pending(self) -> bool:
        """One-shot read of the create-save scroll-back flag (task-417).

        Returns:
            ``True`` exactly once per arm -- the first recompose after a
            create-save scrolls the action row into view, later ones don't.
        """
        pending = self._library_skill_scroll_pending
        self._library_skill_scroll_pending = False
        return pending

    def _library_skill_text_fields_match_state(self) -> bool:
        """True when every live text field equals the editor state's value.

        task-417: a recompose (e.g. the post-create snapshot refresh)
        remounts the editor's Inputs/TextArea, whose spurious mount-time
        ``Changed`` events can arrive AFTER any ``call_after_refresh``
        re-arm -- the armed-flag dance alone cannot win that race. A mount
        echo always carries the state's own values, so value equality is
        the reliable spurious-vs-real discriminator.
        """
        state = self._library_skill_editor_state
        if state is None:
            return False
        fields = self._read_library_skill_editor_fields()
        if fields is None:
            return False
        (
            raw_name,
            raw_description,
            raw_argument_hint,
            raw_allowed_tools_csv,
            raw_model,
            raw_body,
        ) = fields
        return (
            raw_name == (state.name or "")
            and raw_description == (state.description or "")
            and raw_argument_hint == (state.argument_hint or "")
            and raw_allowed_tools_csv == (state.allowed_tools_csv or "")
            and raw_model == (state.model or "")
            and raw_body == (state.body or "")
        )

    def _library_skill_on_disk_path(self) -> str:
        """The selected skill's on-disk directory, for remediation copy.

        Empty when the service or selection is unavailable (the copy
        helper falls back to generic wording).
        """
        service = getattr(self.app_instance, "local_skills_service", None)
        skills_dir = getattr(service, "skills_dir", None)
        name = self._selected_skill_name
        if skills_dir is None or not name:
            return ""
        try:
            return str(Path(skills_dir) / name)
        except Exception:
            return ""

    def _mark_library_skill_dirty(self, *, force: bool = False) -> None:
        """Record an in-progress skill edit.

        Ignored until ``_library_skill_editor_armed`` is set (see that
        flag's docstring), and -- unless ``force`` -- ignored when the live
        text fields still equal the editor state (a mount-time ``Changed``
        echo, task-417). The toggle/cycle buttons mutate the state BEFORE
        marking, so they pass ``force=True``. Unlike the notes editor,
        this never arms an autosave timer -- the skill editor is
        explicit-Save-only.
        """
        if not self._library_skill_editor_armed:
            return
        if not force and self._library_skill_text_fields_match_state():
            return
        self._library_skill_dirty = True
        # task-449: dirty-marking never recomposes, so the Discard button's
        # initial disabled render is patched live here.
        self._set_library_skill_discard_enabled(True)
        # task-417: a lingering "Saved." is stale the moment new edits
        # exist -- clear it alongside the dirty mark.
        if self._library_skill_status:
            self._update_library_skill_status_static("")

    def _read_library_skill_live_name(self) -> str:
        """Read the Name Input's current (possibly unsaved) value.

        Falls back to the editor state's own name when the Input isn't
        mounted (e.g. the conflict banner, which doesn't render the field
        Inputs).
        """
        try:
            return self.query_one("#library-skill-name", Input).value
        except (NoMatches, QueryError):
            state = self._library_skill_editor_state
            return state.name if state is not None else ""

    def _update_library_skill_warnings_static(self, *, name: str | None = None) -> None:
        """Targeted update of ``#library-skill-warnings``, no recompose.

        Args:
            name: The live Name field value to compute the shadow warning
                against. Defaults to the current editor state's name (used
                right after a trust action, where the Name field itself
                hasn't changed).
        """
        state = self._library_skill_editor_state
        if state is None:
            return
        if name is None:
            name = state.name
        try:
            warnings_static = self.query_one("#library-skill-warnings", Static)
        except (NoMatches, QueryError):
            return
        lines = skill_editor_warning_lines(
            live_name=name,
            trust_status=state.trust_status,
            trust_blocked=state.trust_blocked,
        )
        warnings_static.update("\n".join(lines))

    def _update_library_skill_status_static(self, text: str) -> None:
        """Targeted update of ``#library-skill-save-status``, no recompose.

        Args:
            text: The status copy to show (``""`` clears it).
        """
        self._library_skill_status = text
        try:
            status_static = self.query_one("#library-skill-save-status", Static)
        except (NoMatches, QueryError):
            return
        status_static.update(text)

    def _render_library_skill_trust_panel(self) -> None:
        """Targeted update of the trust panel's state/changed-files/buttons,
        no recompose -- called after every trust action and after a
        successful Save, so an in-progress (unsaved) edit elsewhere in the
        editor is never discarded by a full rebuild.
        """
        state = self._library_skill_editor_state
        if state is None:
            return
        try:
            self.query_one("#library-skill-trust-state", Static).update(
                skill_trust_state_line(state.trust_status, state.trust_changed_files)
            )
        except (NoMatches, QueryError):
            pass
        try:
            self.query_one("#library-skill-trust-remediation", Static).update(
                skill_trust_panel_remediation_copy(
                    state.trust_status, self._library_skill_on_disk_path()
                )
            )
        except (NoMatches, QueryError):
            pass
        try:
            self.query_one("#library-skill-trust-review-identity", Static).update(
                skill_review_identity_line(self._library_skill_active_review)
            )
        except (NoMatches, QueryError):
            pass
        try:
            self.query_one("#library-skill-trust-review-files", Static).update(
                ", ".join(
                    str(item)
                    for item in (
                        (self._library_skill_active_review or {}).get("changed_files")
                        or []
                    )
                )
            )
        except (NoMatches, QueryError):
            pass
        try:
            self.query_one("#library-skill-trust-review-content", Static).update(
                skill_trust_review_preview(self._library_skill_active_review)
            )
        except (NoMatches, QueryError):
            pass
        try:
            unlock_button = self.query_one("#library-skill-trust-unlock", Button)
            unlock_button.disabled = not skill_trust_unlock_enabled(state.trust_status)
            # F-018: reason/action tooltips flip in place with `disabled`.
            unlock_button.tooltip = skill_trust_unlock_tooltip(state.trust_status)
        except (NoMatches, QueryError):
            pass
        try:
            review_button = self.query_one("#library-skill-trust-review", Button)
            review_button.disabled = not skill_trust_review_enabled(
                state.trust_status, state.trust_blocked
            )
            review_button.tooltip = skill_trust_review_tooltip(
                state.trust_status, state.trust_blocked
            )
        except (NoMatches, QueryError):
            pass
        try:
            approve_button = self.query_one("#library-skill-trust-approve", Button)
            approve_button.disabled = self._library_skill_active_review is None
            approve_button.tooltip = skill_trust_approve_tooltip(
                self._library_skill_active_review is not None
            )
        except (NoMatches, QueryError):
            pass
        # Task 7 (skills-script-execution): reads the CACHED grant
        # (``_library_skill_script_grant``), never the trust service
        # directly -- ``script_execution_granted`` re-scans the skill's
        # on-disk directory to verify its fingerprint, which is blocking
        # file I/O this method (called from synchronous event handlers) must
        # not perform. ``_refresh_library_skill_script_grant`` is what keeps
        # the cache current, off-thread.
        try:
            self.query_one("#library-skill-script-grant", Static).update(
                skill_script_grant_line(self._library_skill_script_grant)
            )
            self.query_one(
                "#library-skill-script-grant-revoke", Button
            ).disabled = not self._library_skill_script_grant
        except (NoMatches, QueryError):
            pass

    @on(Input.Changed, "#library-skill-name")
    def handle_library_skill_name_changed(self, event: Input.Changed) -> None:
        """Mark the open skill dirty on a Name edit, and live-refresh the
        shadow-name warning (unconditionally, not gated by "armed" -- the
        warning is a plain live read, not a dirty-tracking concern).

        Args:
            event: Input change event emitted by the editor's Name field.
        """
        self._mark_library_skill_dirty()
        self._update_library_skill_warnings_static(name=event.value)

    @on(Input.Changed, "#library-skill-argument-hint")
    @on(Input.Changed, "#library-skill-allowed-tools")
    def handle_library_skill_input_changed(self, event: Input.Changed) -> None:
        """Mark the open skill dirty on a field edit.

        Args:
            event: Input change event emitted by one of the editor's
                single-line fields.
        """
        self._mark_library_skill_dirty()

    @on(Input.Changed, "#library-skill-description")
    def handle_library_skill_description_changed(self, event: Input.Changed) -> None:
        """Mark dirty on a Description edit and sync the derived-hint (review
        finding).

        The "No description set" hint (rendered only when the skill's
        description was auto-derived from its body) is compose-time-only, so
        without this it would linger beneath a now-populated field. It is
        meaningful only while the field is still empty.

        Args:
            event: Input change event emitted by the Description field.
        """
        self._mark_library_skill_dirty()
        self._sync_library_skill_description_hint(event.value)

    def _sync_library_skill_description_hint(self, live_value: str) -> None:
        """Show the derived-description hint only while the field is empty."""
        state = self._library_skill_editor_state
        should_show = bool(
            state is not None and state.description_derived and not live_value.strip()
        )
        for hint in self.query("#library-skill-description-hint"):
            if isinstance(hint, Static):
                hint.display = should_show

    @on(TextArea.Changed, "#library-skill-body")
    def handle_library_skill_body_changed(self, event: TextArea.Changed) -> None:
        """Mark the open skill dirty on a Body edit.

        Args:
            event: Text change event emitted by the editor's Body TextArea.
        """
        self._mark_library_skill_dirty()

    def _update_library_skill_toggle_buttons(self) -> None:
        """Targeted label update for the user-invocable/disable-model/context
        toggle Buttons, no recompose."""
        state = self._library_skill_editor_state
        if state is None:
            return
        try:
            self.query_one(
                "#library-skill-user-invocable", Button
            ).label = skill_user_invocable_label(state.user_invocable)
        except (NoMatches, QueryError):
            pass
        try:
            self.query_one("#library-skill-invocation-copy", Static).update(
                skill_invocation_copy(
                    state.user_invocable, state.disable_model_invocation
                )
            )
            self.query_one("#library-skill-argument-fields").display = bool(
                self._library_skill_editor_mode == "advanced" or state.user_invocable
            )
        except (NoMatches, QueryError):
            pass

    @on(Button.Pressed, "#library-skill-editor-mode")
    async def handle_library_skill_editor_mode(self, event: Button.Pressed) -> None:
        """Switch mounted Skill presentations and remember the display choice."""
        event.stop()
        self._snapshot_library_skill_live_fields()
        requested = (
            "basic" if self._library_skill_editor_mode == "advanced" else "advanced"
        )
        try:
            canvas = self.query_one("#library-skill-work-pane", LibrarySkillWorkPane)
        except NoMatches:
            return
        await canvas.set_editor_mode(requested)
        self._library_skill_editor_mode = requested
        library_config = self.app_instance.app_config.setdefault("library", {})
        if isinstance(library_config, dict):
            library_config["skill_editor_mode"] = requested
        self.run_worker(
            self._persist_library_skill_editor_mode(requested),
            group="library_skill_editor_mode",
            exclusive=True,
        )

    @on(Button.Pressed, "#library-skill-mode-overview")
    @on(Button.Pressed, "#library-skill-mode-edit")
    @on(Button.Pressed, "#library-skill-mode-trust")
    @on(Button.Pressed, "#library-skill-mode-files")
    def handle_library_skill_reader_mode(self, event: Button.Pressed) -> None:
        """Switch the selected Skill's explicit work-pane projection."""
        event.stop()
        requested = coerce_skill_reader_mode(event.button.id.rsplit("-", 1)[-1])
        if not self._selected_skill_name and requested != "edit":
            return
        if self._library_skill_reader_mode == "edit":
            self._snapshot_library_skill_live_fields()
        if requested == self._library_skill_reader_mode:
            return
        self._library_skill_reader_mode = requested
        _sync_library_canvas(self, "skills")

    @on(Button.Pressed, "#library-skill-detail-retry")
    def handle_library_skill_detail_retry(self, event: Button.Pressed) -> None:
        """Retry only the still-selected Skill detail with a fresh fence."""
        event.stop()
        name = self._selected_skill_name
        if (
            not name
            or self._library_skill_mutation_in_flight
            or not self._library_skill_detail_retryable
        ):
            return
        generation = self._claim_library_skill_detail_generation()
        self.run_worker(
            self._refresh_library_skill_detail(
                name,
                request_generation=generation,
            ),
            exclusive=True,
            group="library_skill_detail",
        )
        _sync_library_canvas(self, "skills")

    @on(Input.Changed, "#library-skill-tool-filter")
    def handle_library_skill_tool_filter(self, event: Input.Changed) -> None:
        """Filter picker rows without changing the Skill's allowlist content."""
        self._library_skill_tool_filter = event.value
        try:
            self.query_one(
                "#library-skill-work-pane", LibrarySkillWorkPane
            ).set_tool_filter(event.value)
        except (NoMatches, QueryError):
            return

    @on(SelectionList.SelectedChanged, "#library-skill-tool-picker")
    def handle_library_skill_tool_selection(
        self, event: SelectionList.SelectedChanged
    ) -> None:
        """Apply only a user-driven tool restriction edit to the canonical draft."""
        try:
            canvas = self.query_one("#library-skill-work-pane", LibrarySkillWorkPane)
        except (NoMatches, QueryError):
            return
        if canvas.rebuilding_tool_picker or not self._library_skill_editor_armed:
            return
        state = self._library_skill_editor_state
        if state is None:
            return
        visible = {selection.value for selection in event.selection_list.options}
        selected = set(skill_allowed_tools_sequence(state.allowed_tools_csv))
        selected.difference_update(visible)
        selected.update(str(value) for value in event.selection_list.selected)
        current_selected = set(skill_allowed_tools_sequence(state.allowed_tools_csv))
        if selected == current_selected:
            return
        reconciled = reconcile_skill_allowed_tools(
            self._library_skill_tool_captured,
            selected=tuple(selected),
            catalog_order=self._library_skill_tool_catalog,
            picker_changed=True,
        )
        self._library_skill_tool_picker_changed = True
        self._library_skill_editor_state = dataclasses.replace(
            state, allowed_tools_csv=", ".join(reconciled)
        )
        self._mark_library_skill_dirty(force=True)
        try:
            self.query_one(
                "#library-skill-disable-model", Button
            ).label = skill_disable_model_label(state.disable_model_invocation)
        except (NoMatches, QueryError):
            pass
        try:
            self.query_one(
                "#library-skill-context", Button
            ).label = skill_context_toggle_label(state.context)
        except (NoMatches, QueryError):
            pass

    @on(Button.Pressed, "#library-skill-user-invocable")
    def handle_library_skill_user_invocable_toggle(self, event: Button.Pressed) -> None:
        """Toggle the open skill's ``user_invocable`` flag.

        Args:
            event: Button press event emitted by the user-invocable toggle.
        """
        event.stop()
        state = self._library_skill_editor_state
        if state is None:
            return
        self._library_skill_editor_state = dataclasses.replace(
            state, user_invocable=not state.user_invocable
        )
        self._mark_library_skill_dirty(force=True)
        self._update_library_skill_toggle_buttons()

    @on(Button.Pressed, "#library-skill-disable-model")
    def handle_library_skill_disable_model_toggle(self, event: Button.Pressed) -> None:
        """Toggle the open skill's ``disable_model_invocation`` flag.

        Args:
            event: Button press event emitted by the disable-model toggle.
        """
        event.stop()
        state = self._library_skill_editor_state
        if state is None:
            return
        self._library_skill_editor_state = dataclasses.replace(
            state, disable_model_invocation=not state.disable_model_invocation
        )
        self._mark_library_skill_dirty(force=True)
        self._update_library_skill_toggle_buttons()

    @on(Button.Pressed, "#library-skill-context")
    def handle_library_skill_context_toggle(self, event: Button.Pressed) -> None:
        """Cycle the open skill's ``context`` field between ``inline``/``fork``.

        Args:
            event: Button press event emitted by the context cycler.
        """
        event.stop()
        state = self._library_skill_editor_state
        if state is None:
            return
        self._library_skill_editor_state = dataclasses.replace(
            state, context=next_skill_context(state.context)
        )
        self._mark_library_skill_dirty(force=True)
        self._update_library_skill_toggle_buttons()

    def _read_library_skill_editor_fields(
        self,
    ) -> tuple[str, str, str, str, str, str] | None:
        """Read the skill editor's current (possibly unsaved) field values.

        Returns:
            ``(name, description, argument_hint, allowed_tools_csv, model,
            body)`` read from the live widgets, or ``None`` if the editor
            isn't mounted.
        """
        try:
            name = self.query_one("#library-skill-name", Input).value
            description = self.query_one("#library-skill-description", Input).value
            argument_hint = self.query_one("#library-skill-argument-hint", Input).value
            body = self.query_one("#library-skill-body", TextArea).text
        except (NoMatches, QueryError):
            return None
        state = self._library_skill_editor_state
        if state is None:
            return None
        allowed_tools_csv = state.allowed_tools_csv
        model = state.model or ""
        return name, description, argument_hint, allowed_tools_csv, model, body

    @on(Button.Pressed, "#library-skill-save")
    def handle_library_skill_save(self, event: Button.Pressed) -> None:
        """Explicitly save the open skill (there is no autosave).

        Args:
            event: Button press event emitted by the editor's "Save" action.
        """
        event.stop()
        self._begin_library_skill_save()

    async def _run_library_skill_save(self) -> None:
        """Hold the shared editor interlock for one durable Skill save."""
        try:
            await self._save_library_skill()
        finally:
            self._library_skill_mutation_in_flight = False
            if self.is_mounted:
                self._sync_library_skill_lifecycle_actions()

    async def _save_library_skill(self) -> None:
        """Save the open Library skill's current editor text.

        Unlike the prompts editor (whose ``update_prompt_by_id`` has no
        caller-supplied expected-version parameter, forcing a manual
        pre-read staleness check), ``LocalSkillsService.update_skill``
        accepts ``expected_version`` directly and raises
        ``local_skill_version_conflict:...`` itself on a real mismatch --
        so this never needs its own pre-read; ``classify_skill_save_error``
        classifies whatever the real write call raises/returns.

        The create/update response mapping is already a full skill detail
        (``LocalSkillsService._response_for_record``'s shape, same as
        ``get_skill``'s), so a successful save's "refresh snapshot" is just
        rebuilding the editor state from THIS call's own result -- no
        second service round-trip needed. This is also how the
        save-marks-needs-review re-quarantine becomes visible without any
        special-casing: the write never passes ``trust_approved=True``, so
        a currently-trusted skill's post-save ``trust_status`` in the
        response is already ``quarantined_modified``.
        """
        if self._library_skills_view != "editor":
            return
        name = self._selected_skill_name
        is_create = not name
        base_state = self._library_skill_editor_state
        if base_state is None:
            return
        fields = self._read_library_skill_editor_fields()
        if fields is None:
            return
        (
            raw_name,
            raw_description,
            raw_argument_hint,
            raw_allowed_tools_csv,
            raw_model,
            raw_body,
        ) = fields

        live_name = self._sanitize_media_field(raw_name, max_length=64)
        description = self._sanitize_note_content(
            raw_description, max_length=LIBRARY_SKILL_TEXT_MAX_CHARS
        )
        argument_hint = self._sanitize_media_field(raw_argument_hint, max_length=500)
        allowed_tools_csv = self._sanitize_note_content(
            raw_allowed_tools_csv, max_length=LIBRARY_SKILL_TEXT_MAX_CHARS
        )
        model = self._sanitize_media_field(raw_model, max_length=128)
        body = self._sanitize_note_content(
            raw_body, max_length=LIBRARY_SKILL_TEXT_MAX_CHARS
        )

        if is_create:
            editor_name = live_name or base_state.name
        else:
            # Renaming an existing skill isn't supported -- the service has
            # no rename primitive, and ``update_skill`` writes under the
            # ORIGINAL directory name regardless of what the frontmatter's
            # ``name`` field says. The Name Input is disabled for existing
            # skills (see ``LibrarySkillsListCanvas._compose_editor``), but
            # this pins the persisted name defensively too: even if the
            # live value somehow diverged, the frontmatter written to disk
            # never does, so a save can never get marked
            # ``validation_status: "invalid"`` (name != parent directory
            # name) the way it silently did before this fix.
            editor_name = base_state.name

        write_state = dataclasses.replace(
            base_state,
            name=editor_name,
            description=description,
            argument_hint=argument_hint or None,
            allowed_tools_csv=allowed_tools_csv,
            model=model or None,
        )
        content = compose_skill_markdown(write_state, body=body)

        service = getattr(self.app_instance, "skills_scope_service", None)
        create_skill = getattr(service, "create_skill", None)
        update_skill = getattr(service, "update_skill", None)

        result: Any = None
        exc: Exception | None = None
        if is_create:
            if not callable(create_skill):
                return
            try:
                result = await self._run_library_service_call(
                    create_skill,
                    mode="local",
                    name=write_state.name,
                    content=content,
                    isolate_in_worker=True,
                )
            except Exception as caught:
                exc = caught
        else:
            if not callable(update_skill):
                return
            try:
                result = await self._run_library_service_call(
                    update_skill,
                    name,
                    mode="local",
                    content=content,
                    expected_version=base_state.version,
                    isolate_in_worker=True,
                )
            except Exception as caught:
                exc = caught

        # Discard out-of-order results, same stale-race guard as
        # ``_refresh_library_skill_detail``/``_save_library_prompt``'s
        # equivalent ``prompt_id != self._selected_prompt_id`` check --
        # applied uniformly for creates too (``name`` was already ``""``
        # at capture time when ``is_create``, so this still lets a
        # still-in-flight create through as long as nothing else got
        # selected meanwhile, but bails if a DIFFERENT skill's editor
        # opened while this create was in flight).
        if name != self._selected_skill_name or self._library_skills_view != "editor":
            return

        if exc is not None:
            logger.opt(exception=True).warning(
                f"Library skill save failed for {name!r}."
            )
            outcome = classify_skill_save_error(None, str(exc), exc)
        else:
            outcome = classify_skill_save_error(result, "", None)

        if outcome == "version-conflict":
            self._enter_library_skill_conflict()
            return
        if outcome != "ok":
            self._update_library_skill_status_static(
                LIBRARY_SKILL_SAVE_STATUS_COPY.get(
                    outcome, LIBRARY_SKILL_SAVE_STATUS_COPY["error"]
                )
            )
            return

        self._apply_library_skill_save_success(result, is_create=is_create)

    def _apply_library_skill_save_success(
        self, result: Any, *, is_create: bool
    ) -> None:
        """Apply a successful save's response: rebuild state, clear dirty,
        show "Saved.", and refresh the trust panel + warnings in place.

        Args:
            result: The create/update call's response mapping (a full
                skill detail).
            is_create: Whether this save created a brand-new skill (adopts
                the new name as ``_selected_skill_name`` and kicks a
                snapshot refresh so the rail badge/list pick up the new
                row).
        """
        if not isinstance(result, Mapping):
            self._update_library_skill_status_static(
                LIBRARY_SKILL_SAVE_STATUS_COPY["error"]
            )
            return
        # Deliberately NOT ``_apply_library_skill_detail`` (which recomposes
        # + re-arms): recomposing here would remount fresh Input/TextArea
        # widgets while the editor is still armed, and Textual's spurious
        # mount-time ``Changed`` event for a non-empty initial value would
        # immediately re-mark the just-saved skill dirty -- same discipline
        # ``_save_library_prompt``'s success tail documents.
        self._library_skill_detail = dict(result)
        self._library_skill_editor_state = build_skill_editor_state(
            self._library_skill_detail
        )
        self._library_skill_tool_captured = skill_allowed_tools_sequence(
            self._library_skill_editor_state.allowed_tools_csv
        )
        self._library_skill_tool_picker_changed = False
        self._library_skill_original_name = self._library_skill_editor_state.name
        self._library_skill_dirty = False
        # A content save changes the exact files/fingerprint that any prior
        # trust receipt identified. Never leave that receipt approvable.
        self._library_skill_active_review = None
        self._refresh_library_skills_after_committed_mutation()
        # task-449: this success tail deliberately never recomposes, so the
        # Discard button is re-disabled in place alongside the dirty clear.
        self._set_library_skill_discard_enabled(False)
        if is_create:
            self._selected_skill_name = self._library_skill_editor_state.name
            # task-417: the snapshot refresh below lands a recompose that
            # would reset the canvas scroll to the top, away from the Save
            # button the user just pressed -- arm the one-shot scroll-back.
            self._library_skill_scroll_pending = True
            # A brand-new skill changes the list's membership/count, so the
            # Skills rail badge and list must pick up the new row now --
            # fire-and-forget, mirrors ``_save_library_prompt``'s equivalent
            # post-create refresh.
            self._refresh_local_source_snapshot()
        self._update_library_skill_status_static(
            (
                "Saved. Review trust before using this Skill with the agent."
                if is_create
                else LIBRARY_SKILL_SAVE_STATUS_COPY["ok"]
            )
        )
        self._update_library_skill_warnings_static(
            name=self._read_library_skill_live_name()
        )
        self._render_library_skill_trust_panel()
        if is_create and self.is_mounted:
            # The first durable save changes the editor's structural truth:
            # it gains a real trust panel and saved-clean lifecycle. Replace
            # the canvas once at that commit point, never for display-mode or
            # ordinary lifecycle changes.
            self._library_skill_editor_armed = False
            _sync_library_canvas(
                self,
                "skills",
                then=self._arm_library_skill_editor,
            )
        # Task 7: saved content changes the skill's fingerprint, which
        # invalidates any prior standing script grant -- re-check off-thread
        # rather than let a just-invalidated grant keep showing as active.
        self._refresh_library_skill_script_grant()

    def _enter_library_skill_conflict(self) -> None:
        """Recompose into the save-conflict banner (Reload only -- see the
        brief's narrower scope vs. the prompts editor's Overwrite+Reload;
        ``update_skill``'s own ``expected_version`` guard is what raised
        this, so nothing here needs to preserve/replay the user's kept
        text the way the prompts conflict path does).
        """
        self._library_skill_conflict = True
        self._library_skill_status = ""
        if self.is_mounted:
            self._sync_library_skill_lifecycle_actions()

    @on(Button.Pressed, "#library-skill-conflict-reload")
    def handle_library_skill_conflict_reload(self, event: Button.Pressed) -> None:
        """Discard the conflicting edit and refetch the skill's fresh detail.

        Args:
            event: Button press event emitted by the conflict banner's
                "Reload" action.
        """
        event.stop()
        name = self._selected_skill_name
        if not name:
            return
        self._library_skill_conflict = False
        self.run_worker(
            self._refresh_library_skill_detail(name),
            exclusive=True,
            group="library_skill_detail",
        )

    async def _flush_library_skill_save(self) -> bool:
        """Veto leaving the skill editor while an edit is unsaved.

        Mirrors ``_flush_library_prompt_save`` exactly: the skill editor is
        explicit-Save-only, so this simply reports whether it is safe to
        proceed -- ``False`` whenever ``_library_skill_dirty`` is set.

        Returns:
            ``True`` when there is nothing unsaved (safe to proceed);
            ``False`` when a dirty edit must be resolved first.
        """
        return not self._library_skill_dirty

    def _notify_skill_dirty_veto(self) -> None:
        """Tell the user WHY a skill-editor exit was just vetoed (task-449).

        Every ``_flush_library_skill_save`` veto site calls this so a
        blocked Back / skill-row / rail-row click never reads as a dead
        button. Warning severity matches the trust-action toasts.
        """
        notify = getattr(self.app_instance, "notify", None)
        if callable(notify):
            notify(LIBRARY_SKILL_DIRTY_VETO_COPY, severity="warning")

    def _set_library_skill_discard_enabled(self, enabled: bool) -> None:
        """Patch the Discard button's disabled state in place (task-449).

        Dirty-marking and the save-success tail both deliberately avoid a
        recompose (see ``_apply_library_skill_save_success``), so the
        Discard button's initial ``disabled=not dirty`` render must be
        kept current by hand at those two transitions.
        """
        self._sync_library_skill_lifecycle_actions()
        for button in self.query("#library-skill-discard"):
            if isinstance(button, Button):
                button.disabled = not enabled
                # F-018: the reason/action tooltip flips in place with
                # `disabled` (this patcher exists precisely to avoid a
                # recompose, so the compose-time tooltip would go stale).
                button.tooltip = (
                    SKILL_DISCARD_TOOLTIP_DIRTY
                    if enabled
                    else SKILL_DISCARD_TOOLTIP_CLEAN
                )

    def _sync_library_skill_lifecycle_actions(self) -> bool:
        """Patch the mounted Skill action strip from screen-owned state."""
        try:
            canvas = self.query_one("#library-skill-work-pane", LibrarySkillWorkPane)
            canvas.sync_lifecycle_actions(
                dirty=self._library_skill_dirty,
                conflict=self._library_skill_conflict,
                confirming_delete=self._library_skill_confirming_delete,
                mutation_in_flight=self._library_skill_mutation_in_flight,
                more_actions_open=self._library_skill_more_actions_open,
                is_create=not self._selected_skill_name,
            )
        except (NoMatches, QueryError):
            return False
        return True

    def _focus_library_skill_name(self) -> None:
        """Focus the create editor's Name field (task-424)."""
        try:
            self.query_one("#library-skill-name", Input).focus()
        except (NoMatches, QueryError):
            pass

    @on(Button.Pressed, "#library-skill-back")
    async def handle_library_skill_back(self, event: Button.Pressed) -> None:
        """Return the Library skills canvas from the editor to its list view.

        Vetoed while dirty (see ``_flush_library_skill_save``) so Back
        never silently discards an unsaved edit.

        Args:
            event: Button press event emitted by the "‹ Back to list" action.
        """
        event.stop()
        await self._exit_library_skill_editor_guarded()

    @on(Button.Pressed, "#library-skill-cancel")
    def handle_library_skill_cancel(self, event: Button.Pressed) -> None:
        """Cancel a never-saved Skill draft and return to the list."""
        event.stop()
        if self._selected_skill_name or self._library_skill_mutation_in_flight:
            return
        self._reset_library_skill_editor_state()
        self._refresh_local_source_snapshot()
        _sync_library_canvas(self, "skills")

    @on(Button.Pressed, "#library-skill-more-actions")
    def handle_library_skill_more_actions(self, event: Button.Pressed) -> None:
        """Reveal saved-clean secondary actions without changing the draft."""
        event.stop()
        if self._library_skill_dirty or self._library_skill_mutation_in_flight:
            return
        self._snapshot_library_skill_live_fields()
        self._library_skill_more_actions_open = (
            not self._library_skill_more_actions_open
        )
        self._sync_library_skill_lifecycle_actions()
        try:
            self.query_one("#library-skill-more-actions", Button).focus()
        except (NoMatches, QueryError):
            pass

    @on(Button.Pressed, "#library-skill-trust-view-details")
    def handle_library_skill_trust_view_details(self, event: Button.Pressed) -> None:
        """Open the healthy trust detail region without changing editor mode."""
        event.stop()
        self._snapshot_library_skill_live_fields()
        self._library_skill_trust_details_open = True
        _sync_library_canvas(self, "skills")

    @on(Button.Pressed, "#library-skill-discard")
    def handle_library_skill_discard(self, event: Button.Pressed) -> None:
        """Leave the skill editor WITHOUT saving the dirty edit (task-449).

        The explicit counterpart to the dirty vetoes: Back and every row
        switch refuse to move while ``_library_skill_dirty`` is set, so
        this button is the one deliberate way out that drops the edit.
        Same exit tail as a clean Back (reset + snapshot + recompose); the
        button renders disabled until the editor is actually dirty, so a
        stray click on a clean editor can't discard anything.

        Args:
            event: Button press event emitted by the "Discard changes"
                action.
        """
        event.stop()
        if not self._library_skill_dirty:
            return
        self._reset_library_skill_editor_state()
        self._refresh_local_source_snapshot()
        _sync_library_canvas(self, "skills")

    def _snapshot_library_skill_live_fields(self) -> None:
        """Fold the editor's live (possibly unsaved) field values back into
        ``_library_skill_editor_state`` so a state-driven recompose renders
        exactly what the user had typed (task-415's confirm step).
        """
        state = self._library_skill_editor_state
        if state is None:
            return
        fields = self._read_library_skill_editor_fields()
        if fields is None:
            return
        (
            raw_name,
            raw_description,
            raw_argument_hint,
            raw_allowed_tools_csv,
            raw_model,
            raw_body,
        ) = fields
        self._library_skill_editor_state = dataclasses.replace(
            state,
            # Rename is unsupported; existing skills keep their name (the
            # Name Input is disabled there anyway), create mode keeps the
            # typed name.
            name=raw_name if not self._selected_skill_name else state.name,
            description=raw_description,
            # A typed description is no longer "derived from the body", so a
            # state-driven recompose must not re-show the "No description
            # set" hint next to the populated field (review finding). Stays
            # derived only while the field is still empty.
            description_derived=state.description_derived
            and not raw_description.strip(),
            argument_hint=raw_argument_hint,
            allowed_tools_csv=raw_allowed_tools_csv,
            model=raw_model,
            body=raw_body,
        )

    async def _run_library_skill_delete(
        self, skill_name: str, request_generation: int
    ) -> None:
        """Hold the shared editor interlock for one durable Skill delete."""
        try:
            await self._delete_library_skill(
                skill_name,
                request_generation=request_generation,
            )
        finally:
            if self._library_skill_detail_request_is_current(
                skill_name=skill_name,
                generation=request_generation,
            ):
                self._library_skill_mutation_in_flight = False
                if self.is_mounted:
                    self._sync_library_skill_lifecycle_actions()

    async def _delete_library_skill(
        self, skill_name: str, *, request_generation: int
    ) -> None:
        """Delete the selected Library skill, then return to the list view.

        Args:
            skill_name: The Library skill name to delete.
        """
        service = getattr(self.app_instance, "skills_scope_service", None)
        delete_skill = getattr(service, "delete_skill", None)
        if not callable(delete_skill):
            self._update_library_skill_status_static("Skill deletion is unavailable.")
            return
        state = self._library_skill_editor_state
        version = state.version if state is not None else None
        try:
            result = await self._run_library_service_call(
                delete_skill,
                skill_name,
                mode="local",
                expected_version=version,
                isolate_in_worker=True,
            )
        except Exception:
            logger.opt(exception=True).warning(
                f"Failed to delete Library skill {skill_name!r}."
            )
            if not self._library_skill_detail_request_is_current(
                skill_name=skill_name,
                generation=request_generation,
            ):
                return
            self._update_library_skill_status_static("Could not delete this skill.")
            return

        if not self._library_skill_detail_request_is_current(
            skill_name=skill_name,
            generation=request_generation,
        ):
            return

        deleted = (
            bool(result.get("deleted", True))
            if isinstance(result, Mapping)
            else bool(result)
        )
        if not deleted:
            self._update_library_skill_status_static(
                "This skill changed elsewhere — refresh and try again."
            )
            return

        self._reset_library_skill_editor_state()
        self._library_skills_filter = ""
        self._refresh_library_skills_after_committed_mutation(
            scope=dataclasses.replace(
                self._library_skills_browse_controller.mutation_refresh_scope,
                query="",
            )
        )
        self._refresh_local_source_snapshot()

    async def _request_library_skill_trust_passphrase(
        self,
        *,
        title: str | None = None,
        message: str | None = None,
    ) -> str | None:
        """Push the shared ``SkillTrustPassphraseModal`` and await a passphrase.

        Mirrors ``skills_screen.SkillsScreen._request_skill_trust_passphrase``
        (reused, not forked): never bootstraps from this editor, so
        ``confirm_bootstrap`` is always ``False``. task-418: ``title``/
        ``message`` let a caller present its real purpose (the approve
        flow) instead of "Unlock Local Skill Trust".
        """
        push_screen_wait = getattr(self.app, "push_screen_wait", None)
        if not callable(push_screen_wait):
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify(
                    "Local skill trust passphrase prompt is unavailable.",
                    severity="warning",
                )
            return None
        result = await push_screen_wait(
            SkillTrustPassphraseModal(
                confirm_bootstrap=False, title=title, message=message
            )
        )
        if isinstance(result, str) and result:
            return result
        return None

    async def _refresh_library_skill_trust_status(self) -> None:
        """Re-fetch the open skill's trust status and patch the trust panel
        in place (no recompose -- see ``_render_library_skill_trust_panel``).
        """
        name = self._selected_skill_name
        generation = self._library_skill_detail_generation
        state = self._library_skill_editor_state
        if not name or state is None:
            return
        result, ok = await self._call_library_skill_trust_service(
            "status_for_skill", name
        )
        if not ok or result is None:
            return
        if not self._library_skill_detail_request_is_current(
            skill_name=name,
            generation=generation,
        ):
            return
        self._library_skill_editor_state = dataclasses.replace(
            self._library_skill_editor_state,
            trust_status=result.trust_status,
            trust_blocked=result.trust_blocked,
            trust_changed_files=tuple(result.changed_files),
        )
        self._render_library_skill_trust_panel()
        self._update_library_skill_warnings_static(
            name=self._read_library_skill_live_name()
        )

    def _refresh_library_skill_script_grant(self) -> None:
        """Kick an off-thread read of the open skill's script-execution grant.

        Task 7 (skills-script-execution): mirrors
        ``_refresh_library_skills_trust_posture``'s shape rather than
        routing through ``_call_library_skill_trust_service`` -- that
        helper toasts a warning whenever the trust service is unavailable,
        which is correct for an explicit user action (Unlock/Review/
        Approve) but would be noisy fired silently on every skill open in
        a deployment with no local trust service wired (e.g. server mode).
        ``SkillTrustService.script_execution_granted`` re-scans the skill's
        on-disk directory to verify its fingerprint, so -- like
        ``trust_posture()`` -- it is NEVER called on the compose/event-loop
        thread, only from here via ``asyncio.to_thread`` in
        ``_load_library_skill_script_grant`` below. Called whenever the
        skill editor opens on an existing skill
        (``_apply_library_skill_detail``) and after a Save
        (``_apply_library_skill_save_success``), since editing content
        invalidates any prior grant.
        """
        name = self._selected_skill_name
        generation = self._library_skill_detail_generation
        trust_service = getattr(self.app_instance, "local_skill_trust_service", None)
        granted_fn = getattr(trust_service, "script_execution_granted", None)
        if not name or not callable(granted_fn):
            self._library_skill_script_grant = False
            return
        self.run_worker(
            self._load_library_skill_script_grant(name, generation, granted_fn),
            exclusive=True,
            group="library_skill_script_grant",
            exit_on_error=False,
        )

    async def _load_library_skill_script_grant(
        self,
        name: str,
        generation: int,
        granted_fn,
    ) -> None:
        """Await the off-thread grant lookup and patch the trust panel.

        Args:
            name: The skill name captured at kick-off time (by
                ``_refresh_library_skill_script_grant``), used to discard an
                out-of-order result if a different skill is open by the
                time this resolves.
            generation: The retained Work session captured at kick-off time.
            granted_fn: The trust service's bound
                ``script_execution_granted`` method (captured at kick-off
                so this never re-reads ``local_skill_trust_service``
                itself).
        """
        try:
            granted = await asyncio.to_thread(granted_fn, name)
        except Exception:
            granted = False
        if not self._library_skill_detail_request_is_current(
            skill_name=name,
            generation=generation,
        ):
            return
        self._library_skill_script_grant = bool(granted)

        # task-8 (skills-script-execution) fix: NOT a direct call. This
        # coroutine's own ``asyncio.to_thread`` round trip can resolve
        # before ``_apply_library_skill_detail``'s own ``refresh(recompose=
        # True)`` (posted moments earlier, on this same screen's message
        # queue) has actually remounted the editor -- a real trust service
        # doing real disk I/O usually loses that race, but there is no
        # guarantee, and a fast trust service (or a slow recompose under
        # load) can win it. A direct call here would then query widgets
        # that do not exist YET, silently no-op through this method's own
        # ``except (NoMatches, QueryError): pass`` guards, and never retry
        # -- leaving the panel stuck showing "not granted"/disabled forever
        # even though ``_library_skill_script_grant`` is correctly True in
        # memory. ``call_after_refresh`` was the original answer, and
        # task-15457 quietly invalidated it: the editor recompose it was
        # ordering against became CANVAS-scoped (`_sync_library_canvas`,
        # driven by the canvas's own message pump), which a SCREEN-level
        # ``call_after_refresh`` has no ordering against -- the render fired
        # before the canvas's children existed, swallowed ``NoMatches``, and
        # never retried (task-15790, measured: grant stored True, render ran,
        # button absent). Ride the same canvas post-recompose hook the
        # caller's arming follow-up already rides; it runs `then` only once
        # the canvas's new children are actually mounted.
        def _render_then_arm() -> None:
            self._render_library_skill_trust_panel()
            if not self._library_skill_editor_armed:
                self._arm_library_skill_editor()

        _sync_library_canvas(self, "skills", then=_render_then_arm)

    async def _request_library_skill_trust_bootstrap_passphrase(self) -> str | None:
        """Push the confirm-passphrase bootstrap modal and await a passphrase.

        Structural twin of ``_request_library_skill_trust_passphrase``: the
        only difference is which modal it drives -- this one CREATES a
        brand-new passphrase (twice-entry confirmed by
        ``SkillTrustBootstrapModal`` itself), the other unlocks an existing
        one.
        """
        push_screen_wait = getattr(self.app, "push_screen_wait", None)
        if not callable(push_screen_wait):
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify(
                    "Local skill trust passphrase prompt is unavailable.",
                    severity="warning",
                )
            return None
        result = await push_screen_wait(SkillTrustBootstrapModal())
        if isinstance(result, str) and result:
            return result
        return None

    def _begin_library_skill_trust_setup(self) -> None:
        self.run_worker(
            self._setup_library_skill_trust(),
            exclusive=True,
            group="library_skill_trust",
        )

    async def _setup_library_skill_trust(self) -> None:
        """Set up trust from the list header's "setup"/"resetup" action.

        Task 5 ambiguity resolution: "Set up" is reset-then-bootstrap ONLY
        when a stale manifest already exists (``trust_store.has_manifest()``
        -- the ``needs_resetup``/orphaned-manifest upgrade case). A truly
        fresh install (no manifest at all, ``needs_setup``) skips the reset
        entirely -- there is nothing to clear -- and goes straight to
        ``bootstrap_trust``, matching the editor's own first-run
        ``_bootstrap_library_skill_trust`` flow. Both the header's "setup"
        and "resetup" action ids route here (``handle_library_skills_trust_action``
        above); the manifest check itself is what decides whether a reset
        actually happens, not the action id.
        """
        service = getattr(self.app_instance, "local_skill_trust_service", None)
        if service is None:
            return
        passphrase = await self._request_library_skill_trust_bootstrap_passphrase()
        if passphrase is None:
            return
        if getattr(service, "trust_store", None) and service.trust_store.has_manifest():
            await self._call_library_skill_trust_service("reset_trust")
        _, ok = await self._call_library_skill_trust_service(
            "bootstrap_trust", passphrase
        )
        if ok:
            self._refresh_library_skills_after_committed_mutation()
            self._refresh_library_skills_trust_posture()
            self._refresh_local_source_snapshot()

    @on(Button.Pressed, "#library-skills-trust-reset-cancel")
    def handle_library_skills_trust_reset_cancel(self, event: Button.Pressed) -> None:
        """Back out of the Reset confirm row without resetting anything.

        Args:
            event: Button press event emitted by the confirm row's Cancel.
        """
        event.stop()
        self._library_skill_trust_confirming_reset = False
        if self.is_mounted:
            _sync_library_canvas(self, "skills")

    @on(Button.Pressed, "#library-skills-trust-reset-confirm")
    def handle_library_skills_trust_reset_confirm(self, event: Button.Pressed) -> None:
        """Run the confirmed destructive Reset (Task 5).

        Args:
            event: Button press event emitted by the confirm row's Reset.
        """
        event.stop()
        self._library_skill_trust_confirming_reset = False
        self.run_worker(
            self._do_library_skill_trust_reset(),
            exclusive=True,
            group="library_skill_trust",
        )

    async def _do_library_skill_trust_reset(self) -> None:
        """Wipe all local trust state, then refresh whichever view is open.

        Always refreshes the list header's posture (a stale ``locked``/
        ``needs_resetup`` header must never linger after a successful
        reset) and the shared local-source snapshot. When the editor's own
        ``quarantined_manifest_error`` trust panel is what triggered this
        (rather than the list header), also re-fetches the OPEN skill's own
        trust status -- mirrors the per-skill re-fetch
        ``_bootstrap_library_skill_trust`` already does after a bootstrap --
        so the panel doesn't keep showing "manifest cannot be verified"
        against a trust store that was just wiped.
        """
        _, ok = await self._call_library_skill_trust_service("reset_trust")
        if not ok:
            return
        self._refresh_library_skills_after_committed_mutation()
        self._refresh_library_skills_trust_posture()
        self._refresh_local_source_snapshot()
        name = self._selected_skill_name
        if (
            self._library_skills_view == "editor"
            and name
            and self._library_skill_editor_state is not None
        ):
            result, ok = await self._call_library_skill_trust_service(
                "status_for_skill", name
            )
            if (
                ok
                and result is not None
                and name == self._selected_skill_name
                and self._library_skills_view == "editor"
                and self._library_skill_editor_state is not None
            ):
                self._library_skill_editor_state = dataclasses.replace(
                    self._library_skill_editor_state,
                    trust_status=result.trust_status,
                    trust_blocked=result.trust_blocked,
                    trust_changed_files=tuple(result.changed_files),
                )
            self._library_skill_active_review = None
        if self.is_mounted:
            _sync_library_canvas(self, "skills")

    async def _open_library_skill_editor_for_review(self, skill_name: str) -> None:
        """Open ``skill_name``'s editor, the same steps a real row press runs.

        Deliberately NOT a refactor of ``handle_library_skill_row`` itself
        (that handler stays untouched) -- just the same flush-veto, reset,
        select, switch-to-editor-view, kick-the-off-thread-detail-fetch,
        recompose sequence, sourced from a name instead of a row Button's
        ``skill_name`` attribute.

        Args:
            skill_name: The blocked skill's name to open.
        """
        if not await self._flush_library_skill_save():
            self._notify_skill_dirty_veto()
            return
        self._reset_library_skill_editor_state()
        self._selected_skill_name = skill_name
        self._library_selected_row_id = LIBRARY_ROW_BROWSE_SKILLS
        self._library_skills_view = "editor"
        self._library_skill_reader_mode = "trust"
        self.run_worker(
            self._refresh_library_skill_detail(skill_name),
            exclusive=True,
            group="library_skill_detail",
        )
        _sync_library_canvas(self, "skills")

    @on(Button.Pressed, "#library-skill-trust-setup")
    def handle_library_skill_trust_setup(self, event: Button.Pressed) -> None:
        """Bootstrap local skill trust from the editor's first-run setup state.

        Only rendered while ``trust_status == "trust_uninitialized"`` (a
        brand-new, never-bootstrapped trust store) -- the Phase-1 gate fix
        for the finding that a fresh install had no live-UI path to create
        the trust passphrase at all.

        Args:
            event: Button press event emitted by the trust panel's "Set up
                skill trust" action.
        """
        event.stop()
        # task-417: any trust action supersedes a lingering "Saved.".
        self._update_library_skill_status_static("")
        self.run_worker(
            self._bootstrap_library_skill_trust(),
            exclusive=True,
            group="library_skill_trust",
        )

    async def _bootstrap_library_skill_trust(self) -> None:
        """Create the initial trust baseline via a confirm-passphrase modal.

        Unlike every other trust action here, ``bootstrap_trust`` is called
        directly (never preceded by ``unlock_with_passphrase`` -- it takes
        the new passphrase itself and derives+stores fresh keys). A full
        recompose follows a successful bootstrap, not the usual targeted
        ``_render_library_skill_trust_panel`` patch: the panel's layout
        itself changes shape here, from the first-run setup state to the
        normal Unlock/Review/Approve row, which a no-recompose patch can't
        produce since those buttons don't exist in the DOM yet.
        """
        if (
            self._library_skills_view != "editor"
            or self._library_skill_editor_state is None
        ):
            return
        name = self._selected_skill_name
        generation = self._library_skill_detail_generation
        passphrase = await self._request_library_skill_trust_bootstrap_passphrase()
        if passphrase is None:
            return
        _, ok = await self._call_library_skill_trust_service(
            "bootstrap_trust", passphrase
        )
        if not ok:
            return
        self._refresh_library_skills_after_committed_mutation()
        if (
            name
            and self._library_skill_detail_request_is_current(
                skill_name=name,
                generation=generation,
            )
            and self._library_skill_editor_state is not None
        ):
            result, status_ok = await self._call_library_skill_trust_service(
                "status_for_skill", name
            )
            if (
                status_ok
                and result is not None
                and self._library_skill_detail_request_is_current(
                    skill_name=name,
                    generation=generation,
                )
            ):
                self._library_skill_editor_state = dataclasses.replace(
                    self._library_skill_editor_state,
                    trust_status=result.trust_status,
                    trust_blocked=result.trust_blocked,
                    trust_changed_files=tuple(result.changed_files),
                )
        self._refresh_local_source_snapshot()
        if name and self._library_skill_detail_request_is_current(
            skill_name=name,
            generation=generation,
        ):
            self._library_skill_active_review = None
        if (
            name
            and self.is_mounted
            and self._library_skill_detail_request_is_current(
                skill_name=name,
                generation=generation,
            )
        ):
            # Disarm dirty-tracking before the recompose (mirrors
            # ``_apply_library_skill_detail``): remounting the Inputs with
            # their existing values still fires their initial
            # ``Input.Changed`` -- without this, still-armed dirty-tracking
            # would misread that as a real edit and wrongly mark the editor
            # dirty (vetoing the next Back/row-switch for no reason).
            self._library_skill_dirty = False
            self._library_skill_editor_armed = False
            _sync_library_canvas(
                self,
                "skills",
                then=self._arm_library_skill_editor,
            )

    @on(Button.Pressed, "#library-skill-trust-unlock")
    def handle_library_skill_trust_unlock(self, event: Button.Pressed) -> None:
        """Unlock local skill trust for this session via the passphrase modal.

        Args:
            event: Button press event emitted by the trust panel's "Unlock"
                action.
        """
        event.stop()
        # task-417: any trust action supersedes a lingering "Saved.".
        self._update_library_skill_status_static("")
        self.run_worker(
            self._unlock_library_skill_trust(),
            exclusive=True,
            group="library_skill_trust",
        )

    async def _unlock_library_skill_trust(self) -> None:
        """Unlock local skill trust for this session via the passphrase modal.

        Task 5: reused by BOTH the editor's own "Unlock" trust-panel action
        AND the list header's "unlock" action (posture ``locked``) -- this
        originally returned immediately unless already in the editor,
        which made the header's Unlock a silent no-op (browsing the list
        is exactly where ``locked`` posture is shown). Editor mode still
        gets the original no-recompose targeted panel patch
        (``_refresh_library_skill_trust_status``) so an in-progress unsaved
        edit elsewhere in the editor is never discarded by a full rebuild;
        list mode instead refreshes the header's posture AND the shared
        local-source snapshot -- the list rows' trust glyphs and the
        header's blocked-count both derive from that snapshot
        (``_build_library_skills_state``), so refreshing posture alone
        would leave every row's ``⚠``/``✓`` and the "N need review" count
        stale until some later snapshot refresh (matches the sibling
        ``_setup_library_skill_trust``/``_do_library_skill_trust_reset``
        handlers). The list view has no unsaved-edit state to lose.
        """
        editor_name = (
            self._selected_skill_name if self._library_skills_view == "editor" else None
        )
        generation = self._library_skill_detail_generation
        passphrase = await self._request_library_skill_trust_passphrase()
        if passphrase is None:
            return
        _, ok = await self._call_library_skill_trust_service(
            "unlock_with_passphrase", passphrase
        )
        if not ok:
            return
        self._refresh_library_skills_after_committed_mutation()
        if editor_name and self._library_skill_detail_request_is_current(
            skill_name=editor_name,
            generation=generation,
        ):
            await self._refresh_library_skill_trust_status()
        elif editor_name is None and self._library_skills_view != "editor":
            self._refresh_library_skills_trust_posture()
            self._refresh_local_source_snapshot()

    async def _review_library_skill_trust(self) -> None:
        if self._library_skills_view != "editor" or not self._selected_skill_name:
            return
        name = self._selected_skill_name
        generation = self._library_skill_detail_generation
        result, ok = await self._call_library_skill_trust_service(
            "capture_review", name
        )
        if not ok or not isinstance(result, Mapping) or not result.get("review_id"):
            return
        if not self._library_skill_detail_request_is_current(
            skill_name=name,
            generation=generation,
        ):
            return
        self._library_skill_active_review = dict(result)
        self._render_library_skill_trust_panel()

    @on(Button.Pressed, "#library-skill-trust-approve")
    def handle_library_skill_trust_approve(self, event: Button.Pressed) -> None:
        """Approve the captured trust review via the passphrase modal.

        Args:
            event: Button press event emitted by the trust panel's
                "Approve" action.
        """
        event.stop()
        # task-417: any trust action supersedes a lingering "Saved.".
        self._update_library_skill_status_static("")
        self.run_worker(
            self._approve_library_skill_trust(),
            exclusive=True,
            group="library_skill_trust",
        )

    @on(Button.Pressed, "#library-skill-script-grant-revoke")
    def handle_library_skill_script_grant_revoke(self, event: Button.Pressed) -> None:
        """Revoke the open skill's standing script-execution grant.

        Args:
            event: Button press event emitted by the trust panel's "Revoke
                script access" action.
        """
        event.stop()
        self.run_worker(
            self._revoke_library_skill_script_grant(),
            exclusive=True,
            group="library_skill_trust",
        )

    async def _revoke_library_skill_script_grant(self) -> None:
        """Drop the open skill's standing script grant, then patch the panel.

        Task 7 (skills-script-execution): the counterpart to whatever
        confirm-card action (Task 6) called ``grant_script_execution`` in
        the first place -- a grant the user cannot see or withdraw here
        would be a real hole. ``revoke_script_execution`` raises
        ``ValueError`` on a malformed skill name, which
        ``_call_library_skill_trust_service`` already catches and reports
        via the standard failure toast, so no extra guard is needed here.
        The outcome is applied directly (no re-fetch) since a successful
        revoke unambiguously means "not granted".
        """
        if self._library_skills_view != "editor" or not self._selected_skill_name:
            return
        name = self._selected_skill_name
        generation = self._library_skill_detail_generation
        _, ok = await self._call_library_skill_trust_service(
            "revoke_script_execution", name
        )
        if not ok:
            return
        if not self._library_skill_detail_request_is_current(
            skill_name=name,
            generation=generation,
        ):
            return
        self._library_skill_script_grant = False
        self._render_library_skill_trust_panel()


# --- BEGIN generated skills-state shims (permanent; byte-for-byte canon) ---
# task 2: exposes every `LibrarySkillsState` field under its original
# `_library_skill_<field>`/`_library_skills_<field>`/`_selected_skill_name`
# name on THIS controller too, reading/writing through the injected
# `skills_state_accessor` instead of a direct `self._skills_state`
# attribute (this class has none) -- same generator shape task 1 installed
# on `LibraryScreen` (deleted at cleanup, task 3, once this copy below
# makes the screen's copy dead) and `LibraryRagSearchController`/
# `LibraryConversationsController`/`LibraryCollectionsController` carry,
# attached programmatically so the class body gains no `FunctionDef`s (the
# size ratchet counts those). `skill_state_shim_attr` is imported from
# `library_skills_state` -- the dataclass's own module -- so this is not a
# second independent copy of the three-way prefix mapping; see that
# module's own docstring for the drift-risk note this avoids from the
# start.
for _lsc_field in dataclasses.fields(LibrarySkillsState):
    setattr(
        LibrarySkillsController,
        skill_state_shim_attr(_lsc_field.name),
        property(
            lambda self, _n=_lsc_field.name: getattr(
                self._skills_state_accessor(), _n
            ),
            lambda self, value, _n=_lsc_field.name: setattr(
                self._skills_state_accessor(), _n, value
            ),
        ),
    )
del _lsc_field
# --- END generated skills-state shims ---

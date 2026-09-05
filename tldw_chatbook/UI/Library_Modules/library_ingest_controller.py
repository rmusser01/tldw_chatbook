"""Library Ingest canvas controller.

Controller PR of the Ingest extraction series (wave-5 task 2 of
``.superpowers/sdd/2026-09-05-library-decomposition-wave5-ingest``; ingest
series 2/3; recipe: ``backlog/docs/library-decomposition-recipe.md``;
``library_skills_controller.py`` -- the newest, largest prior single-cluster
series -- is the template this mirrors byte-for-byte in shape). Owns the
Ingest canvas cluster: path/backend/browse form handling, pre-flight
debounce/cancel/apply, submit-form validation and two-press Start consent,
per-type option panels (fold state, group receipts, tooling-detail fold),
the job queue's row actions (open/retry/dismiss/view-on-server/choose-gguf/
retry-faster-whisper/cancel/force-stop -- job-id parsing only; see
exclusions below for the ones excluded), Clear-finished/expand-all/
collapse-all, and rail-width auto-collapse. ``LibraryScreen`` keeps
one-line delegators under every one of these 56 original names (55 plain
delegators + 1 class-forwarding staticmethod delegator).

**Cluster derivation.** Wave-5 task 1's own census: an ``ast`` scan of
``LibraryScreen`` for method names containing ``"ingest"`` (case-insensitive)
found **78 raw ``FunctionDef`` matches, 78 unique names** (no property/
setter-pair gap, unlike Skills' 133/127 or Conversations' own field-side
gap). This task re-ran that census fresh at its own execution time (recipe's
"never trust a carried-over count" rule, §6) and reconfirmed the identical
78. A reverse oddball scan (any NON-"ingest"-named method read by 2+
ingest-named methods, and any ingest-named method reaching a field outside
its own ``_library_ingest_`` prefix) found nothing beyond what task 1 already
excluded as shared shell state (``_library_external_submit_*``,
``_transcribe_cpp_configured``, ``_library_model_install_progress_*``).

**Single vs. split controller: single, decided by call-graph connected-
components analysis.** A ``self.<name>(...)`` internal call-graph among the
78 candidates (before exclusions) is ONE dense connected component -- hub
names like ``_update_library_ingest_dynamic_regions``,
``_build_library_ingest_state``, ``_library_ingest_registry``, and
``_disarm_library_ingest_start_confirm`` are each called from 3-8 different
sibling methods spanning path-entry, pre-flight, submit, and queue-row
concerns. There is no subset of the cluster that only ever calls within
itself. **Decision: ONE combined ``LibraryIngestController``**, matching the
skills/search+RAG precedent's identical resolution at a comparable scale.

**22 of the 78 candidates excluded, not moved (56 move):**

1. **4 `@work(thread=True)`-decorated methods -- the "framework-decorator
   self-type assertion" hazard (recipe §3, the export series' own
   precedent).** Textual's ``@work`` decorator asserts ``isinstance(self,
   DOMNode)`` at call time (read from ``textual/_work_decorator.py``, not
   assumed); a plain controller object is not a ``DOMNode``. ``_save_
   library_ingest_backend``, ``_persist_library_ingest_location``, ``_run_
   library_ingest_preflight``, and ``_save_library_ingest_options`` stay on
   ``LibraryScreen``, UNMOVED, decorator and body byte-for-byte untouched.
   Of these, only ``_persist_library_ingest_location`` and ``_run_library_
   ingest_preflight`` have a MOVER caller (``handle_library_ingest_browse``,
   ``_trigger_library_ingest_preflight``); the other two (``_save_library_
   ingest_backend``, called only by the ALSO-excluded ``handle_library_
   ingest_backend_switch``; ``_save_library_ingest_options``, called only by
   the ALSO-excluded ``_enqueue_library_ingest_snapshot``/``handle_library_
   ingest_option_reset``) need no binding at all.
2. **3 module-globals-coupling exclusions (recipe §3's oldest documented
   shape, and two close variants of it).**
   - ``_resolve_ingest_source`` reads the bare names ``validate_path_
     simple``/``validate_url`` (ordinary module-level imports in
     ``library_screen.py``, resolved against the DEFINING module's
     ``__globals__`` at call time). Found by the coordinator-mandated
     mechanical module-globals census (recipe §3's newest numbered shape,
     see that entry for the method), NOT by the original battery -- the
     first-draft move shipped GREEN because ``Tests/UI/test_library_
     shell.py::test_library_shell_ingest_canvas_invalid_path_notifies_and_
     submits_nothing`` patches ``tldw_chatbook.UI.Screens.library_screen.
     validate_path_simple`` with a stub that unconditionally raises, then
     presses ``#library-ingest-start`` with a NONEXISTENT tmp path -- the
     assertions (a warning fires, zero jobs submitted) pass whether the
     STUB fires or the REAL validator's own "file does not exist" check
     fires, so the moved body's silently-bypassed patch produced a
     GREEN-BUT-VACUOUS test, not a red one. Confirmed genuine (not
     assumed) by a probe: with the body still moved, patching ``library_
     screen.validate_path_simple`` to reject an EXISTING file left the
     controller's own separately-imported ``validate_path_simple`` binding
     unaffected -- the file passed real validation and no warning fired --
     then, after reverting the exclusion below, the SAME probe correctly
     produced the warning through both the direct screen call and the
     controller's forwarded call. Reverted to ``LibraryScreen``, UNMOVED,
     full-bodied; its only caller, ``_submit_library_ingest_form`` (a
     mover), reaches it through a named late-binding dependency below.
   - ``_remember_library_ingest_location`` reads the bare name ``save_
     setting_to_cli_config`` (an ordinary module-level import in
     ``library_screen.py``, resolved against the DEFINING module's
     ``__globals__`` at call time). ``Tests/UI/test_library_screen.py::
     test_ingest_browse_remembers_the_directory_of_the_picked_file`` patches
     ``tldw_chatbook.UI.Screens.library_screen.save_setting_to_cli_config``
     and calls the REAL, ``__init__``-constructed screen's method directly,
     expecting the internal free-name call to observe the patch -- moving
     the body would silently repoint this test's patch away from the call it
     actually makes. Its only caller, the ALSO-excluded (@work hazard)
     ``_persist_library_ingest_location``, needs no binding for it.
   - ``_load_library_ingest_options_from_config`` calls the bare name
     ``_library_ingest_options_for`` -- one of recipe §3's own permanently
     screen-routed trio (``_INGEST_OPTIONS_CACHE_ATTR``, ``_read_library_
     ingest_options_from_config``, ``_library_ingest_options_for``;
     ``library_screen.py:605-692``, confirmed untouched by task 1). Unlike
     the trio's own documented hazard (which is about the trio's INTERNAL
     resolution of each other), moving this METHOD (not the trio) is safe
     for test-patch-reach purposes -- ``_library_ingest_options_for``'s own
     body still resolves ``_read_library_ingest_options_from_config``/
     ``get_cli_setting`` through ITS OWN module's globals regardless of who
     calls it. The real blocker is CIRCULAR IMPORT: every controller import
     in ``library_screen.py`` (this one included) sits at the top of the
     file, before line 605 where the trio is defined -- a module-level
     ``from ..Screens.library_screen import _library_ingest_options_for`` in
     this controller module would try to import a name that does not exist
     yet on a module still mid-execution. Adding a deferred import INSIDE
     the moved method body would violate the byte-for-byte canon (editing
     the body). Excluded; its only caller is ``on_mount`` (a shell method,
     not a mover), so no binding is needed either.
3. **9 unbound-fake-self / `object.__new__`-bypass exclusions** (recipe §3's
   first documented shape, PLUS its own "seventh bypass shape" variant --
   task 1's own new finding -- manifesting here at the CONTROLLER level: a
   test builds a screen via ``object.__new__(LibraryScreen)``/``LibraryScreen.
   __new__(LibraryScreen)``, skipping ``__init__`` (so ``_ingest_controller``
   was never constructed either), then calls one of these methods UNBOUND or
   BOUND on that bare instance -- a delegator's ``self._ingest_controller.
   <name>(...)`` would raise ``AttributeError`` immediately). A repo-wide
   content grep (``LibraryScreen\\.<name>\\(`` unbound-call form, PLUS a
   second pass tracking every ``object.__new__(LibraryScreen)``/``LibraryScreen.
   __new__(LibraryScreen)`` assignment target and searching the SAME test
   function body for ``<var>.<name>(`` bound calls -- the unbound-only grep
   alone would have missed the bound-call sites) across ALL of ``Tests/``,
   not a keyword-filtered subset (the parakeet-lesson precedent, recipe §3's
   seventh-shape filter-blindness rule):
   - ``_do_submit_ingest`` (``Tests/UI/test_library_ingest_canvas.py`` x4,
     ``Tests/integration/test_library_ingest_flow.py`` x2, ALL via ``object.
     __new__``/``.LibraryScreen.__new__``-bypass screens; its own MOVER
     caller ``_submit_library_ingest_form`` reaches it via a named
     dependency).
   - ``_enqueue_library_ingest_snapshot`` (``Tests/UI/test_library_ingest_
     canvas.py``, unbound call on an ``object.__new__`` bypass; its only
     caller is the ALSO-excluded ``_do_submit_ingest``, so no binding
     needed).
   - ``_build_ingest_options_snapshot`` (``Tests/App/test_submit_library_
     ingest_job.py`` x5, BOUND calls -- ``screen._build_ingest_options_
     snapshot()`` -- on ``object.__new__`` bypass screens; its MOVER caller
     ``_current_library_ingest_start_consent`` reaches it via a named
     dependency).
   - ``_library_ingest_browse_location`` (``Tests/Utils/test_config_nested_
     settings.py::test_library_ingest_browse_location_audit_fix``, unbound
     call on a bare ``SimpleNamespace`` fake -- found OUTSIDE every
     ingest-named test root the recipe's own four canonical roots list,
     confirming the plan's "widen beyond the obvious roots" mandate; its
     MOVER caller ``handle_library_ingest_browse`` reaches it via a named
     dependency).
   - ``_run_debounced_library_ingest_preflight`` (``Tests/Library/test_
     ingest_preflight_egress.py::test_the_typing_debounce_forbids_probing_
     even_when_it_is_enabled``, unbound call on a bare ``SimpleNamespace``
     ``stand_in``; referenced -- not called -- by its MOVER caller ``handle_
     library_ingest_path_changed`` as a ``set_timer`` callback, reached via
     a named dependency).
   - ``handle_library_ingest_backend_switch`` (``Tests/UI/test_library_
     ingest_canvas.py``, unbound call on an ``object.__new__`` bypass screen;
     zero mover callers -- ``@on``-dispatched only).
   - ``handle_library_ingest_directory_browse`` (``Tests/UI/test_library_
     ingest_canvas.py``, unbound call on an ``object.__new__`` bypass screen;
     zero mover callers -- ``@on``-dispatched only; ALSO one of the plan's 5
     hard-precondition handlers, real ``.press()``-driven characterization
     pinned in the RED commit regardless of this exclusion, per the plan's
     own mandate that the pin's value (proving the message-dispatch path)
     does not depend on where the handler body ends up living).
   - ``handle_library_ingest_option_reset`` (``Tests/UI/test_library_ingest_
     canvas.py``, unbound call on an ``object.__new__`` bypass screen; zero
     mover callers -- ``@on``-dispatched only; ALSO one of the plan's 5
     hard-precondition handlers, same treatment as above).
   - ``handle_library_ingest_option_value_changed`` (``Tests/UI/test_
     library_canvas_scoped_sync.py::test_ingest_checkbox_routes_to_ingest_
     canvas_sync``, unbound call on a bare ``SimpleNamespace``; zero mover
     callers -- ``@on``-dispatched only).

   All 9 stay on ``LibraryScreen``, UNMOVED, full-bodied, decorator (where
   present) untouched -- the recipe's own established accommodation for this
   shape (leave the method real; a mover reaches it, where needed, through a
   named late-binding dependency that re-reads ``screen.<name>`` at call
   time, which is exactly why the SimpleNamespace/bypass fixtures above keep
   working unmodified after this move).
4. **6 instance-attribute-monkeypatch exclusions** (recipe §3's second
   documented bypass shape, the skills series' own ``_request_library_
   skills_browse`` precedent). Four seeded by the SAME shared fixture,
   ``Tests/UI/test_library_ingest_inline_consent.py``'s own ``_minimal_
   library_screen()`` (an ``object.__new__`` bypass shared by ~55 tests in
   that file); a fifth found independently, by a real Pilot test
   (``Tests/UI/test_library_shell.py::test_library_ingest_progress_action_
   change_recomposes_dynamic_regions``) patching an instance method on a
   REAL, mounted screen; a sixth found by a REAL, `__init__`-constructed
   screen fixture in a different file (``Tests/UI/test_library_screen.py::
   test_handle_library_ingest_open_wires_to_open_job_in_library``). Each
   expects a sibling mover to observe a ``screen.<name> = Mock(...)``/
   ``MagicMock(...)`` patch when it calls ``self.<name>(...)`` internally.
   Found by running the battery, not the static census -- five successive
   draft rounds each moved one more of these and watched a real test
   file's tests fail with a genuine ``AttributeError``/uncalled-mock
   chain, never a silent pass. All six stay on ``LibraryScreen``, UNMOVED,
   full-bodied; every mover caller reaches each through a named
   late-binding dependency below.
   - ``_build_library_ingest_state`` -- a heavily-shared, 13-caller helper
     (``_current_library_ingest_start_consent``, ``_submit_library_ingest_
     form``, ``_update_library_ingest_dynamic_regions``, ``action_library_
     ingest_back``, and the ``handle_library_ingest_{title,author,
     keywords}_changed``/``_path_changed``/``_path_submitted``/
     ``_clear_path``/``_expand_all``/``_collapse_all``/``_clear_finished``
     family).
   - ``_notify_library_ingest_warning`` -- 3 mover callers
     (``handle_library_ingest_cancel``, ``_submit_library_ingest_form``,
     ``_resolve_ingest_source``).
   - ``_update_library_ingest_gate`` -- 7 mover callers (``handle_library_
     ingest_{title,author,keywords}_changed``, ``_update_library_ingest_
     dynamic_regions``, ``_submit_library_ingest_form``, ``handle_library_
     ingest_path_changed``, ``action_library_ingest_back``).
   - ``_refresh_library_ingest_canvas_preserving_context`` -- 2 mover
     callers (``_update_library_ingest_dynamic_regions``, ``_restage_
     library_ingest_last_submission``).
   - ``_update_library_ingest_dynamic_regions`` -- a second heavily-shared,
     13-caller helper (``_apply_library_ingest_preflight_result``,
     ``_handle_library_ingest_progress_changed``, ``_handle_library_ingest_
     registry_changed``, ``_on_ingest_job_details``, ``_trigger_library_
     ingest_preflight``, and the ``handle_library_ingest_{clear_finished,
     clear_path,collapse_all,dismiss,expand_all,path_changed,retry,
     retry_faster_whisper}`` family). Itself the caller found broken by
     the real-Pilot test above -- pressing the real backend-switch button
     ended a job's local-STT progress tick, routing through the registry's
     progress listener (a mover) into this method (also a mover); patched
     directly on a REAL, mounted screen instance
     (``screen._update_library_ingest_dynamic_regions = Mock(wraps=...)``),
     expecting the listener to observe it.
   - ``_library_ingest_job_by_id`` -- 2 mover callers (``handle_library_
     ingest_open``, ``handle_library_ingest_view_on_server``); its own
     body also calls the ALSO-moved ``_library_ingest_registry``, reached
     the same accessor-callable way once excluded.

**56 of the 78 candidates move onto this controller** (25 ``@on`` handlers +
2 ``action_*`` methods + 1 ``@staticmethod`` + 28 plain).

**Byte-for-byte canon** (moved bodies never edited -- every name they
reference that is not this controller's own state is rebound under the SAME
name, per the two binding kinds; see ``LibrarySkillsController.__init__``
and ``ConsoleDictationController.__init__`` for the sibling worked
examples):

1. **Framework services** (``app``, ``app_instance``, ``call_after_
   refresh``, ``is_attached``, ``is_mounted``, ``is_running``, ``notify``,
   ``query``, ``query_one``, ``refresh``, ``register_footer_shortcuts``,
   ``run_worker``, ``set_focus``, ``set_timer``, ``size``) are live-read
   from the screen via ``@property`` on every access -- never snapshotted.
   ``is_running`` exists for the same reason the skills/RAG controllers'
   own docstrings document for theirs: ``_apply_library_ingest_backend_
   save`` forwards bare ``self`` into the shared, multi-subsystem ``_sync_
   library_canvas(screen, "ingest")`` dispatcher (``canvas_sync.py``) when
   a backend save lands while the Ingest canvas is showing -- found by the
   battery, not the static census (a first draft omitting it left ``self.
   is_running`` raising ``AttributeError`` on the controller, silently
   caught by the dispatcher's own outer ``except Exception`` and falling
   back to a full ``screen.refresh(recompose=True)`` -- exactly the
   whole-screen recompose ``Tests/UI/test_library_canvas_scoped_sync.py::
   test_ingest_backend_switch_recomposes_only_the_ingest_canvas`` asserts
   does NOT happen).

   **A related, deliberately UNFIXED finding from the same mechanical
   module-globals census (recipe §3's newest numbered shape):**
   ``_apply_library_ingest_backend_save`` ALSO reads a bare module global,
   ``_sync_library_canvas`` itself (the call just above) -- a plain
   function, not a method, imported fresh into this controller module
   (``from .canvas_sync import _sync_library_canvas``, the same shape
   every sibling controller already uses for its own ``kind=`` call). The
   census found every ``library_screen``-scoped patch target for this name
   across ``Tests/`` (7 files, ~20 sites, both the direct-attribute and the
   ``monkeypatch.setattr(module, "name", ...)``/fully-qualified-string
   patch shapes) and confirmed NONE is ingest-related -- all 7 files
   (``test_library_file_notes_workspace.py``, ``test_library_entry_
   compose_once.py``, ``test_library_note_import_flow.py``, ``test_
   library_review_round_t21116.py``, ``test_library_media_trash.py``,
   ``test_library_notes_folder_navigator.py``, ``Tests/Skills/test_
   skills_import.py``) patch it for notes/media/skills canvas syncs, zero
   for Ingest. **Verdict: KEEP as a mover.** Unlike ``_resolve_ingest_
   source`` above, there is no ACTIVE test whose assertions this coupling
   silently defeats -- excluding a whole method to guard against a
   theoretical, currently-unexercised collision would be over-conservative
   (and mechanically odd besides: ``_sync_library_canvas`` is a bare
   FUNCTION call, not `self.<name>`, so it cannot be late-bound as a
   named dependency without editing the body -- the only two
   accommodations available are "exclude the whole method" or "leave it,
   documented"). The identical bare-``_sync_library_canvas`` shape exists
   in every one of the five prior controllers (conversations, export,
   collections, search+RAG, skills) that call this same shared dispatcher
   -- this is a SYSTEMIC pattern, not an ingest-specific defect, and a
   cross-controller audit of all six is recorded as a follow-up, not
   fixed here (out of this task's own scope).

   One more name joins this group for a narrower reason:
   ``LIBRARY_INGEST_SHORTCUTS`` is a ``LibraryScreen`` CLASS attribute (a
   literal tuple, not an ``__init__`` field) that ``Tests/UI/test_library_
   ingest_keyboard.py`` reads directly off the SCREEN (``screen.LIBRARY_
   INGEST_SHORTCUTS``) -- deleting it from ``LibraryScreen`` would break
   that test, so it stays there permanently and this controller exposes it
   via the SAME live-read pass-through shape as every other framework
   service, rather than a second, independently-drifting literal copy.
2. **Everything else** the cluster depends on that is not its own state is a
   NAMED constructor dependency: (a) 13 general Library-wide shell helpers a
   moved body calls with explicit arguments (``_apply_library_notes_stage_
   visibility``, ``_focus_library_hub_entry``, ``_invalidate_library_
   external_submission``, ``_library_landing_attention_action``, ``_open_
   job_in_library``, ``_open_library_external_media_detail``, ``_open_
   transcribe_cpp_gguf_picker``, ``_refresh_local_source_snapshot`` -- one
   of recipe §3's four PERMANENTLY screen-routed monkeypatch names, reached
   here the same accessor-callable way every other subsystem already
   reaches it, never moved itself --, ``_safe_text``, ``_select_library_
   rail_row``, ``_server_binding_is_shipped_placeholder``, ``_sync_library_
   emergency_guard_presentation``, ``_sync_library_landing_lifecycle_
   presentation``); (b) 7 shared shell state accessors this cluster reads
   (``_library_selected_row_id``, ``_transcribe_cpp_configured``,
   ``_footer_shortcut_registration``, ``_library_canvas_projection_
   depth``) and reads+writes (``_library_rail_collapsed``, ``_library_
   landing_attention_signature``, ``_library_canvas_resync_pending`` --
   the last pair joining framework-service ``is_running`` above for the
   identical ``_sync_library_canvas`` forwarding reason, mirroring the
   skills/RAG controllers' own identical pair); (c) NONE -- no wiring
   accessor pair exists for this subsystem (task 1's own finding: "no
   field holds a live controller/coordinator instance"); (d) N/A -- no
   merely-delegate-to-existing-controller properties exist for Ingest
   (unlike Skills' import-coordinator precedent); (e) 13 named late-binding
   callables for the exclusions above that a MOVER still calls/references
   internally (``_build_ingest_options_snapshot``, ``_build_library_ingest_
   state``, ``_do_submit_ingest``, ``_library_ingest_browse_location``,
   ``_library_ingest_job_by_id``, ``_notify_library_ingest_warning``,
   ``_persist_library_ingest_location``, ``_refresh_library_ingest_canvas_
   preserving_context``, ``_resolve_ingest_source``, ``_run_debounced_
   library_ingest_preflight``, ``_run_library_ingest_preflight``,
   ``_update_library_ingest_dynamic_regions``, ``_update_library_ingest_
   gate``) -- each a ``lambda`` that
   re-reads ``screen.<name>`` on every invocation, at CALL time, not a
   value captured once at construction, which is exactly why every bypass
   fixture named above keeps working unmodified after this move.

**Class-level constants, one exception aside.** Three ``LibraryScreen``
class-body literals (``_RETRY_CONFIRM_DEAD_ZONE_SECONDS``, ``_START_
CONFIRM_DEAD_ZONE_SECONDS``, ``_CLEAR_FINISHED_DEAD_ZONE_SECONDS`` -- each
a bare ``0.3`` used by exactly one, now-moved, method and referenced
NOWHERE else in ``library_screen.py``, confirmed by a repo-wide grep) are
genuinely dead on the screen once their sole consumer moves, so they are
DELETED there and declared fresh on this controller -- the class-constant
analogue of a state-PR's field deletion for a zero-external-reference
field. ``LIBRARY_INGEST_SHORTCUTS`` is the one exception (see binding kind 1
above): it keeps its screen-side test dependency and is never deleted.

**Construction order -- the usual position.** ``LibraryScreen.__init__``
builds ``self._ingest_controller`` right after ``self._skills_controller``,
matching every other controller in this file.

This subsystem's OWN state (every ``_library_ingest_<field>`` name the
moved bodies reference -- all 20, single prefix, no plural variant, no
wiring exclusion, per task 1's own ``LibraryIngestState`` docstring) is
exposed through a generated property loop reading ``self._ingest_state_
accessor().<field>`` -- the same generator shape task 1 installed on
``LibraryScreen`` (deleted at cleanup, task 3, once this controller's own
copy makes the screen's copy dead), mirroring every single-prefix
controller precedent (``LibraryExportController``, ``LibraryCollections
Controller``) exactly.
"""
from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TYPE_CHECKING

from textual import on
from textual.containers import Vertical
from textual.css.query import NoMatches, QueryError
from textual.widgets import Button, Collapsible, Input, Static

from ...Library.ingest_analysis import resolve_ingest_analysis_provider
from ...Library.ingest_capabilities import capabilities_for_backend, get_capabilities
from ...Library.ingest_types import PreflightResult
from ...Library.library_ingest_jobs import (
    ACTIVE_INGEST_STATES,
    IngestJobState,
    LibraryIngestJob,
    build_active_ingest_consent_scope,
    count_duplicate_done_jobs,
    normalize_active_ingest_source,
)
from ...Library.library_ingest_state import (
    INGEST_UNAVAILABLE_COPY,
    LibraryIngestCanvasState,
    LibraryIngestFormState,
    active_ingest_start_confirm_line,
    build_ingest_forecast,
    format_ingest_progress_line,
    ingest_progress_action_signature,
    library_ingest_retry_available,
    library_ingest_retry_label,
    parse_keywords,
)
from ...Library.library_shell_state import LIBRARY_ROW_INGEST_MEDIA
from ...Third_Party.textual_fspicker import FileOpen
from ...Widgets.Library import (
    LibraryIngestCanvas,
    LibraryIngestPreflightSummary,
    LibraryIngestQueuePanel,
)
from ...Widgets.Library.library_ingest_canvas import (
    build_type_group_title,
    ingest_scope_label,
)
from .canvas_sync import _sync_library_canvas
from .library_ingest_state import LibraryIngestState
from .screen_constants import LIBRARY_INGEST_RAIL_COLLAPSE_BREAKPOINT
from .screen_helpers import _ingestible_file_filters
from .screen_support_types import _LibraryIngestStartConsent

if TYPE_CHECKING:
    from ..Screens.library_screen import LibraryScreen


class LibraryIngestController:
    """Owns the Ingest canvas cluster (56 methods).

    Holds no state of its own beyond what it reads and writes through
    ``LibraryIngestState`` (via the injected accessor) and the shared
    shell/framework bindings below. ``LibraryScreen`` constructs exactly one
    of these, in ``__init__`` right after ``self._skills_controller``, and
    keeps one-line delegators for every original name this cluster moved
    (56 -- see the module docstring for the full derivation and the 22
    exclusions).
    """

    # Genuinely dead on `LibraryScreen` once their sole consumer moved here
    # (repo-wide grep: each name appears nowhere else in library_screen.py) --
    # deleted there, declared fresh here. `LIBRARY_INGEST_SHORTCUTS` is NOT
    # duplicated this way; see the `LIBRARY_INGEST_SHORTCUTS` property below.
    _RETRY_CONFIRM_DEAD_ZONE_SECONDS = 0.3
    _START_CONFIRM_DEAD_ZONE_SECONDS = 0.3
    _CLEAR_FINISHED_DEAD_ZONE_SECONDS = 0.3

    def __init__(
        self,
        screen: "LibraryScreen",
        *,
        ingest_state_accessor,
        # -- general Library-wide shell helpers, not moved (group (a)).
        apply_library_notes_stage_visibility,
        focus_library_hub_entry,
        invalidate_library_external_submission,
        library_landing_attention_action,
        open_job_in_library,
        open_library_external_media_detail,
        open_transcribe_cpp_gguf_picker,
        refresh_local_source_snapshot,
        safe_text,
        select_library_rail_row,
        server_binding_is_shipped_placeholder,
        sync_library_emergency_guard_presentation,
        sync_library_landing_lifecycle_presentation,
        # -- shared shell state this cluster reads/writes (group (b)).
        library_selected_row_id_accessor,
        transcribe_cpp_configured_accessor,
        footer_shortcut_registration_accessor,
        library_rail_collapsed_accessor,
        set_library_rail_collapsed,
        library_landing_attention_signature_accessor,
        set_library_landing_attention_signature,
        library_canvas_projection_depth_accessor,
        library_canvas_resync_pending_accessor,
        set_library_canvas_resync_pending,
        # -- named late-binding callables for the test-bypass/hazard
        # exclusions (group (e)) that a MOVER still calls/references
        # internally.
        build_ingest_options_snapshot,
        build_library_ingest_state,
        do_submit_ingest,
        library_ingest_browse_location,
        library_ingest_job_by_id,
        notify_library_ingest_warning,
        persist_library_ingest_location,
        refresh_library_ingest_canvas_preserving_context,
        resolve_ingest_source,
        run_debounced_library_ingest_preflight,
        run_library_ingest_preflight,
        update_library_ingest_dynamic_regions,
        update_library_ingest_gate,
    ) -> None:
        """Build the controller and bind everything its moved bodies need.

        Every one of the 63 method bodies below is a byte-for-byte copy of
        the pre-extraction ``LibraryScreen`` method: no internal line was
        edited to retarget a call or an attribute. That is possible because
        this constructor binds every name those bodies reference that is
        not this controller's own state, under the SAME name the original
        method used. See the module docstring for the binding kinds this
        follows and the full per-parameter derivation.

        Args:
            screen: The Library screen. Used ONLY for the fifteen framework
                services below (``app``, ``app_instance``, ``call_after_
                refresh``, ``is_attached``, ``is_mounted``, ``is_running``,
                ``notify``, ``query``, ``query_one``, ``refresh``,
                ``register_footer_shortcuts``, ``run_worker``, ``set_focus``,
                ``set_timer``, ``size``) plus the screen-owned ``LIBRARY_
                INGEST_SHORTCUTS`` class constant -- this cluster owns no DOM
                of its own.
            ingest_state_accessor: Returns the live ``LibraryIngestState``
                (``LibraryScreen._ingest_state``, task 1). Backs every
                generated ``_library_ingest_<field>`` property below.
        """
        self._screen = screen
        self._ingest_state_accessor = ingest_state_accessor
        self._apply_library_notes_stage_visibility_fn = (
            apply_library_notes_stage_visibility
        )
        self._focus_library_hub_entry_fn = focus_library_hub_entry
        self._invalidate_library_external_submission_fn = (
            invalidate_library_external_submission
        )
        self._library_landing_attention_action_fn = library_landing_attention_action
        self._open_job_in_library_fn = open_job_in_library
        self._open_library_external_media_detail_fn = (
            open_library_external_media_detail
        )
        self._open_transcribe_cpp_gguf_picker_fn = open_transcribe_cpp_gguf_picker
        self._refresh_local_source_snapshot_fn = refresh_local_source_snapshot
        self._safe_text_fn = safe_text
        self._select_library_rail_row_fn = select_library_rail_row
        self._server_binding_is_shipped_placeholder_fn = (
            server_binding_is_shipped_placeholder
        )
        self._sync_library_emergency_guard_presentation_fn = (
            sync_library_emergency_guard_presentation
        )
        self._sync_library_landing_lifecycle_presentation_fn = (
            sync_library_landing_lifecycle_presentation
        )
        self._library_selected_row_id_accessor = library_selected_row_id_accessor
        self._transcribe_cpp_configured_accessor = transcribe_cpp_configured_accessor
        self._footer_shortcut_registration_accessor = (
            footer_shortcut_registration_accessor
        )
        self._library_rail_collapsed_accessor = library_rail_collapsed_accessor
        self._set_library_rail_collapsed_fn = set_library_rail_collapsed
        self._library_landing_attention_signature_accessor = (
            library_landing_attention_signature_accessor
        )
        self._set_library_landing_attention_signature_fn = (
            set_library_landing_attention_signature
        )
        self._library_canvas_projection_depth_accessor = (
            library_canvas_projection_depth_accessor
        )
        self._library_canvas_resync_pending_accessor = (
            library_canvas_resync_pending_accessor
        )
        self._set_library_canvas_resync_pending_fn = set_library_canvas_resync_pending
        self._build_ingest_options_snapshot_fn = build_ingest_options_snapshot
        self._build_library_ingest_state_fn = build_library_ingest_state
        self._do_submit_ingest_fn = do_submit_ingest
        self._library_ingest_browse_location_fn = library_ingest_browse_location
        self._library_ingest_job_by_id_fn = library_ingest_job_by_id
        self._notify_library_ingest_warning_fn = notify_library_ingest_warning
        self._persist_library_ingest_location_fn = persist_library_ingest_location
        self._refresh_library_ingest_canvas_preserving_context_fn = (
            refresh_library_ingest_canvas_preserving_context
        )
        self._resolve_ingest_source_fn = resolve_ingest_source
        self._run_debounced_library_ingest_preflight_fn = (
            run_debounced_library_ingest_preflight
        )
        self._run_library_ingest_preflight_fn = run_library_ingest_preflight
        self._update_library_ingest_dynamic_regions_fn = (
            update_library_ingest_dynamic_regions
        )
        self._update_library_ingest_gate_fn = update_library_ingest_gate

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
    def is_attached(self) -> bool:
        return self._screen.is_attached

    @property
    def is_mounted(self) -> bool:
        return self._screen.is_mounted

    @property
    def is_running(self) -> bool:
        return self._screen.is_running

    @property
    def notify(self) -> Any:
        return self._screen.notify

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
    def register_footer_shortcuts(self) -> Any:
        return self._screen.register_footer_shortcuts

    @property
    def run_worker(self) -> Any:
        return self._screen.run_worker

    @property
    def set_focus(self) -> Any:
        return self._screen.set_focus

    @property
    def set_timer(self) -> Any:
        return self._screen.set_timer

    @property
    def size(self) -> Any:
        return self._screen.size

    @property
    def LIBRARY_INGEST_SHORTCUTS(self) -> tuple[tuple[str, str], ...]:
        # `LibraryScreen`'s own class attribute -- kept there permanently
        # because `Tests/UI/test_library_ingest_keyboard.py` reads it
        # directly off the screen; forwarded live rather than duplicated as
        # a second literal.
        return self._screen.LIBRARY_INGEST_SHORTCUTS

    # -- shared shell state (group (b)) ------------------------------------

    @property
    def _library_selected_row_id(self) -> str:
        return self._library_selected_row_id_accessor()

    @property
    def _transcribe_cpp_configured(self) -> bool:
        return self._transcribe_cpp_configured_accessor()

    @property
    def _footer_shortcut_registration(self) -> Any:
        return self._footer_shortcut_registration_accessor()

    @property
    def _library_rail_collapsed(self) -> bool:
        return self._library_rail_collapsed_accessor()

    @_library_rail_collapsed.setter
    def _library_rail_collapsed(self, value: bool) -> None:
        self._set_library_rail_collapsed_fn(value)

    @property
    def _library_landing_attention_signature(self) -> Any:
        return self._library_landing_attention_signature_accessor()

    @_library_landing_attention_signature.setter
    def _library_landing_attention_signature(self, value: Any) -> None:
        self._set_library_landing_attention_signature_fn(value)

    @property
    def _library_canvas_projection_depth(self) -> int:
        return self._library_canvas_projection_depth_accessor()

    @property
    def _library_canvas_resync_pending(self) -> bool:
        return self._library_canvas_resync_pending_accessor()

    @_library_canvas_resync_pending.setter
    def _library_canvas_resync_pending(self, value: bool) -> None:
        self._set_library_canvas_resync_pending_fn(value)

    # -- general Library-wide shell helpers (group (a)) --------------------

    @property
    def _apply_library_notes_stage_visibility(self) -> Any:
        return self._apply_library_notes_stage_visibility_fn

    @property
    def _focus_library_hub_entry(self) -> Any:
        return self._focus_library_hub_entry_fn

    @property
    def _invalidate_library_external_submission(self) -> Any:
        return self._invalidate_library_external_submission_fn

    @property
    def _library_landing_attention_action(self) -> Any:
        return self._library_landing_attention_action_fn

    @property
    def _open_job_in_library(self) -> Any:
        return self._open_job_in_library_fn

    @property
    def _open_library_external_media_detail(self) -> Any:
        return self._open_library_external_media_detail_fn

    @property
    def _open_transcribe_cpp_gguf_picker(self) -> Any:
        return self._open_transcribe_cpp_gguf_picker_fn

    @property
    def _refresh_local_source_snapshot(self) -> Any:
        return self._refresh_local_source_snapshot_fn

    @property
    def _safe_text(self) -> Any:
        return self._safe_text_fn

    @property
    def _select_library_rail_row(self) -> Any:
        return self._select_library_rail_row_fn

    @property
    def _server_binding_is_shipped_placeholder(self) -> Any:
        return self._server_binding_is_shipped_placeholder_fn

    @property
    def _sync_library_emergency_guard_presentation(self) -> Any:
        return self._sync_library_emergency_guard_presentation_fn

    @property
    def _sync_library_landing_lifecycle_presentation(self) -> Any:
        return self._sync_library_landing_lifecycle_presentation_fn

    # -- named late-binding callables for the test-bypass/hazard
    # exclusions (group (e)) ------------------------------------------

    @property
    def _build_ingest_options_snapshot(self) -> Any:
        return self._build_ingest_options_snapshot_fn

    @property
    def _build_library_ingest_state(self) -> Any:
        return self._build_library_ingest_state_fn

    @property
    def _do_submit_ingest(self) -> Any:
        return self._do_submit_ingest_fn

    @property
    def _library_ingest_browse_location(self) -> Any:
        return self._library_ingest_browse_location_fn

    @property
    def _library_ingest_job_by_id(self) -> Any:
        return self._library_ingest_job_by_id_fn

    @property
    def _notify_library_ingest_warning(self) -> Any:
        return self._notify_library_ingest_warning_fn

    @property
    def _persist_library_ingest_location(self) -> Any:
        return self._persist_library_ingest_location_fn

    @property
    def _refresh_library_ingest_canvas_preserving_context(self) -> Any:
        return self._refresh_library_ingest_canvas_preserving_context_fn

    @property
    def _resolve_ingest_source(self) -> Any:
        return self._resolve_ingest_source_fn

    @property
    def _run_debounced_library_ingest_preflight(self) -> Any:
        return self._run_debounced_library_ingest_preflight_fn

    @property
    def _run_library_ingest_preflight(self) -> Any:
        return self._run_library_ingest_preflight_fn

    @property
    def _update_library_ingest_dynamic_regions(self) -> Any:
        return self._update_library_ingest_dynamic_regions_fn

    @property
    def _update_library_ingest_gate(self) -> Any:
        return self._update_library_ingest_gate_fn

    # -- moved cluster methods (56), byte-for-byte, original file order ---
    def _library_ingest_shortcuts_for_current_state(
        self,
    ) -> tuple[tuple[str, str], ...]:
        """Return prioritized Ingest hints, including Retry only when live."""
        shortcuts = list(self.LIBRARY_INGEST_SHORTCUTS)
        if getattr(self, "_library_ingest_start_consent", None) is not None:
            shortcuts[1] = ("esc", "cancel")
        registry = getattr(
            getattr(self, "app_instance", None), "library_ingest_jobs", None
        )
        jobs_fn = getattr(registry, "jobs", None)
        jobs = jobs_fn() if callable(jobs_fn) else ()
        if library_ingest_retry_available(
            jobs,
            last_submission_available=(
                getattr(self, "_library_ingest_last_submission", None) is not None
            ),
        ):
            shortcuts.insert(2, ("r", "retry"))
        return tuple(shortcuts)

    def _sync_library_ingest_rail_for_width(self, width: int) -> None:
        """Auto-collapse the rail only while narrow Ingest needs the space."""
        # The grid carries a wide min-width and can report its virtual width
        # while overflowing an 80-column terminal. The user experiences the
        # screen viewport, so use the smaller real allocation for this gate.
        viewport_width = self.size.width if self.is_mounted else 0
        if viewport_width > 0:
            width = min(width, viewport_width)
        ingest_active = self._library_selected_row_id == LIBRARY_ROW_INGEST_MEDIA
        should_collapse = (
            ingest_active and width < LIBRARY_INGEST_RAIL_COLLAPSE_BREAKPOINT
        )
        if should_collapse and not self._library_rail_collapsed:
            self._library_rail_collapsed = True
            self._library_ingest_auto_collapsed_rail = True
            self._apply_library_notes_stage_visibility()
        elif self._library_ingest_auto_collapsed_rail and (
            not ingest_active or not should_collapse
        ):
            self._library_rail_collapsed = False
            self._library_ingest_auto_collapsed_rail = False
            self._apply_library_notes_stage_visibility()

    def _sync_library_ingest_rail_from_shell(self) -> None:
        """Measure the settled shell, then apply the Ingest rail policy."""
        try:
            width = self.query_one("#library-shell-grid").region.width
        except (NoMatches, QueryError):
            return
        if width > 0:
            self._sync_library_ingest_rail_for_width(width)

    def _reset_library_ingest_transient_state(self) -> None:
        """Clear the ingest canvas's form to defaults on rail re-entry.

        Called on every ``_select_library_rail_row`` switch so a stale
        in-progress form from a previous Ingest visit never reappears when
        the user comes back to the canvas. The job queue
        itself is registry-owned and untouched by this reset -- only the
        local form echo resets. Any in-flight pre-flight worker is
        cancelled AND generation-fenced (cancellation is cooperative, so a
        finished worker's late result would otherwise still land on the
        fresh form -- task-2011).
        """
        self._invalidate_library_ingest_preflight()
        self._invalidate_library_external_submission()
        timer = self._library_ingest_path_debounce_timer
        if timer is not None:
            timer.stop()
            self._library_ingest_path_debounce_timer = None
        self._library_ingest_clear_finished_armed = False
        self._disarm_library_ingest_retry_confirm()
        self._library_ingest_expanded_details.clear()
        self._library_ingest_form = LibraryIngestFormState()

    def _pause_library_ingest_transient_ui(self) -> None:
        """Rail-switch hygiene WITHOUT wiping the staged form (task-2043).

        Stops the typing debounce (it must not fire while another canvas is
        showing), fences off any in-flight pre-flight worker (its late
        result would otherwise land while a DIFFERENT canvas is showing --
        the stored echo persists, only the worker is fenced), and disarms
        the two-press clear; the typed path, metadata, and pre-flight echo
        persist for the session so a multi-batch workflow survives a look
        at another rail row.
        """
        timer = self._library_ingest_path_debounce_timer
        if timer is not None:
            timer.stop()
            self._library_ingest_path_debounce_timer = None
        self._library_ingest_preflight_generation += 1
        self._cancel_library_ingest_preflight()
        self._invalidate_library_external_submission()
        self._library_ingest_clear_finished_armed = False
        # (task-3314) Same hygiene for the pending Start consent: changing
        # canvases mid-consent is not a "yes".
        self._disarm_library_ingest_start_confirm()

    def _library_ingest_registry(self) -> Any:
        """Return the app's ingest job registry, or ``None`` when absent."""
        return getattr(self.app_instance, "library_ingest_jobs", None)

    def _handle_library_ingest_registry_changed(self) -> None:
        """Registry listener: live-recompose the ingest canvas + poke the
        source snapshot when a job finishes (Task 5).

        Registered against ``self.app_instance.library_ingest_jobs`` in
        ``on_mount``, removed in ``on_unmount``. Per the registry's own
        contract (``LibraryIngestJobRegistry._notify_listeners``), this
        fires synchronously on the UI thread after every successful
        ``submit``/``mark_parsing``/``mark_writing``/``mark_done``/
        ``mark_failed``/``requeue`` -- from two different call shapes:

        - **Synchronously inside a message handler.** "Start import" and
          ordinary Library "Retry" actions mutate the registry (firing this
          listener) before their handler's trailing dynamic-region update.
          A Research-owned retry is scheduled through its durable operation
          owner and fires the listener when that owner persists its replacement.
        - **Marshaled from a background thread**, via ``call_from_thread``
          for ``mark_parsing``/``mark_writing`` (the F3 parse-pool
          coordinator, itself invoked from a pool callback thread) and
          ``mark_done``/``mark_failed`` (the writer's worker thread) --
          these land outside any message handler, as their own turn of the
          UI event loop.

        Both shapes are safe to handle with a plain, synchronous
        ``self.refresh(recompose=True)`` call (no ``call_after_refresh``
        indirection needed): ``Widget.refresh(recompose=True)`` never
        recomposes inline -- it only sets ``_recompose_required = True``
        and schedules the actual (async) ``_check_recompose`` via
        ``call_next``, which runs on a later turn of the event loop. That
        makes calling it redundant, or from inside another handler that
        will also call it, harmless: the flag is idempotent and the
        second scheduled check becomes a no-op once the first has already
        cleared it. (Verified by reading
        ``textual.widget.Widget.refresh``/``_check_recompose`` -- Textual
        8.2.7.)

        Behavior:

        - Recomposes the canvas ONLY when the ingest canvas is the
          currently selected rail row -- a job transition must never yank
          a user looking at a different canvas away from it.
        - Independently of the canvas recompose, pokes
          ``_refresh_local_source_snapshot()`` (which updates the rail's
          ``Media (N)`` count) whenever the registry's done-job count has
          grown since this screen last checked -- deduped via
          ``_library_ingest_last_done_count`` so a running/failed
          transition (or a second notification for the same completed
          job) never re-triggers the snapshot fetch. This fires
          regardless of which canvas is selected, since the rail is
          always visible.
        - A no-op when the screen isn't mounted -- belt-and-braces
          alongside ``on_unmount``'s removal (see that method's
          docstring for why removal can't simply happen earlier, e.g. on
          suspend). Note ``self.is_mounted`` never flips back to
          ``False`` after removal in this Textual version -- it only
          guards a callback that somehow fires before this screen's very
          first mount -- so ``on_unmount``'s ``remove_listener`` call is
          what actually prevents post-teardown notifications, not this
          guard.
        """
        if not self.is_mounted:
            return
        armed = getattr(self, "_library_ingest_start_consent", None)
        if armed is not None:
            submitted_source = self._library_ingest_form.path.strip()
            pending = self._current_library_ingest_start_consent(submitted_source)
            # Lifecycle tokens are deliberately absent from the
            # fingerprint. Only a different set of active jobs invalidates
            # what the user armed against on a registry tick.
            if (
                pending.active_job_ids != armed.active_job_ids
                and not self._authoritative_library_ingest_consent_is_current(
                    armed, pending
                )
            ):
                self._disarm_library_ingest_start_confirm()
        if self._library_selected_row_id == LIBRARY_ROW_INGEST_MEDIA:
            self._update_library_ingest_dynamic_regions()
            shortcuts = self._library_ingest_shortcuts_for_current_state()
            registration = ("library", tuple(shortcuts))
            if self._footer_shortcut_registration != registration:
                self.register_footer_shortcuts(source="library", shortcuts=shortcuts)
        registry = self._library_ingest_registry()
        counts_fn = getattr(registry, "counts", None)
        counts = counts_fn() if callable(counts_fn) else {}
        # (task-2015) Any registry mutation disarms a pending two-press
        # "Clear finished": the queue the user armed against just changed.
        self._library_ingest_clear_finished_armed = False
        # (task-2015) Batch-settle toast: when the active-job count crosses
        # 0 -> N a baseline of (done, failed) is captured; on N -> 0 one
        # summary toast reports the deltas -- the only above-the-fold
        # completion signal on a queue that renders below it.
        active_count = (
            counts.get("queued", 0)
            + counts.get("parsing", 0)
            + counts.get("writing", 0)
        )
        previous_active = self._library_ingest_last_active_count
        self._library_ingest_last_active_count = active_count
        done_now = counts.get("done", 0)
        failed_now = counts.get("failed", 0)
        skipped_now = counts.get("skipped", 0)
        # (task-2041) A dedup match reaches DONE without importing anything;
        # counting it as "imported" made the toast contradict the row copy.
        # Matches are recognised by the writer's progress-message prefix
        # (shared constant, so the two can never drift).
        # (task-2042 review) The registry's internal counter avoids a second
        # deep-copy ``jobs()`` snapshot per tick; the pure-function fallback
        # keeps test doubles working.
        matched_counter = getattr(registry, "count_duplicate_done", None)
        if callable(matched_counter):
            matched_now = matched_counter()
        else:
            jobs_fn = getattr(registry, "jobs", None)
            matched_now = count_duplicate_done_jobs(
                jobs_fn() if callable(jobs_fn) else ()
            )
        (
            baseline_done,
            baseline_failed,
            baseline_matched,
            baseline_skipped,
        ) = self._library_ingest_batch_baseline
        if (
            done_now < baseline_done
            or failed_now < baseline_failed
            or matched_now < baseline_matched
            or skipped_now < baseline_skipped
        ):
            # (task-2015 review) Clear/dismiss mid-batch shrinks DONE/FAILED
            # below the baseline; without re-anchoring, the settle deltas go
            # negative and completions after the clear vanish from the
            # toast.
            baseline_done = min(baseline_done, done_now)
            baseline_failed = min(baseline_failed, failed_now)
            baseline_matched = min(baseline_matched, matched_now)
            baseline_skipped = min(baseline_skipped, skipped_now)
            self._library_ingest_batch_baseline = (
                baseline_done,
                baseline_failed,
                baseline_matched,
                baseline_skipped,
            )
        if previous_active == 0 and active_count > 0:
            self._library_ingest_batch_baseline = (
                done_now,
                failed_now,
                matched_now,
                skipped_now,
            )
        elif previous_active > 0 and active_count == 0:
            (
                baseline_done,
                baseline_failed,
                baseline_matched,
                baseline_skipped,
            ) = self._library_ingest_batch_baseline
            matched = max(0, matched_now - baseline_matched)
            imported = (done_now - baseline_done) - matched
            failed = failed_now - baseline_failed
            skipped = max(0, skipped_now - baseline_skipped)
            if imported > 0 or matched > 0 or failed > 0 or skipped > 0:
                parts = []
                if imported > 0:
                    parts.append(f"{imported} imported")
                if matched > 0:
                    # (task-2837) The forecast says "will match"; the
                    # toast answers in the same word.
                    parts.append(f"{matched} matched")
                if skipped > 0:
                    parts.append(f"{skipped} skipped")
                if failed > 0:
                    parts.append(f"{failed} failed")
                notify = getattr(self.app_instance, "notify", None)
                if callable(notify):
                    # (task-2220) Skips are neutral and never warn on
                    # their own (failed == 0 -> information). Real failures
                    # with zero successes DO warn even when skips are also
                    # present -- something the user pointed at genuinely
                    # broke and nothing landed.
                    notify(
                        "Import finished — " + " · ".join(parts),
                        severity=(
                            "information"
                            if imported > 0 or matched > 0 or failed == 0
                            else "warning"
                        ),
                    )
        done_count = counts.get("done", 0)
        if done_count != self._library_ingest_last_done_count:
            grew = done_count > self._library_ingest_last_done_count
            self._library_ingest_last_done_count = done_count
            if grew:
                self._refresh_local_source_snapshot()
        attention = self._library_landing_attention_action()
        if attention != self._library_landing_attention_signature:
            self._library_landing_attention_signature = attention
            if not self._library_selected_row_id:
                self._sync_library_landing_lifecycle_presentation()

    def _handle_library_ingest_progress_changed(
        self,
        before: LibraryIngestJob,
        after: LibraryIngestJob,
    ) -> None:
        """Patch ordinary ingest telemetry without disturbing queue context.

        Progress updates cannot change lifecycle state, origin, result ids,
        error detail, or terminal actions. Local-STT cancellation is the one
        progress-owned structural exception: its action signature replaces
        Cancel with Force stop and therefore recomposes the dynamic regions.

        Args:
            before: Job snapshot immediately before the progress mutation.
            after: Job snapshot immediately after the progress mutation.
        """
        if (
            not self.is_attached
            or self._library_selected_row_id != LIBRARY_ROW_INGEST_MEDIA
        ):
            return
        if ingest_progress_action_signature(before) != ingest_progress_action_signature(
            after
        ):
            self._update_library_ingest_dynamic_regions()
            return
        try:
            progress_widget = self.query_one(
                f"#library-ingest-progress-{after.job_id}", Static
            )
        except (NoMatches, QueryError):
            return
        progress = after.progress
        if progress is None and after.state is IngestJobState.WRITING:
            progress = {"phase": "writing"}
        progress_widget.update(format_ingest_progress_line(progress, state=after.state))
        progress_widget.display = True

    def _restore_library_ingest_canvas_context(
        self,
        focused_id: str | None,
        cursor: int | None,
        scroll_y: float | None,
        stale_path_input: Input | None = None,
        stale_path_value: str | None = None,
    ) -> None:
        """Re-apply focus/cursor/scroll captured before a job-tick recompose.

        Scroll first, then focus with ``scroll_visible=False`` so restoring
        focus does not itself yank the scroll position. A vanished widget id
        (a finished job's row-action button) degrades silently (task-2010).

        Also rescues keystrokes that raced the recompose (task-3311's
        second half): the replaced path Input is the only holder of text
        typed between scheduling and applying the rebuild. Adoption is
        gated on the widget's value having CHANGED since capture, so a
        deliberate form rewrite under an untouched field (the retry
        re-stage) is never undone by a stale echo. The value is written to
        the live Input rather than to the form echo, so the ordinary
        ``Input.Changed`` seam runs the whole follow-through (gate, Clear
        button, intro lines, the pre-flight debounce) exactly once.

        Args:
            focused_id: Id of the widget focused before the recompose.
            cursor: That widget's cursor position, when it was an ``Input``.
            scroll_y: The canvas's scroll offset before the recompose.
            stale_path_input: The pre-recompose path ``Input`` object.
            stale_path_value: That widget's value at capture time.
        """
        if scroll_y is not None:
            try:
                canvas = self.query_one(LibraryIngestCanvas)
                canvas.scroll_to(y=scroll_y, animate=False, force=True)
            except (NoMatches, QueryError):
                pass
        raced_text: str | None = None
        if (
            stale_path_input is not None
            and stale_path_value is not None
            and stale_path_input.value != stale_path_value
        ):
            raced_text = stale_path_input.value
        if raced_text is not None:
            try:
                live_path = self.query_one("#library-ingest-path", Input)
            except (NoMatches, QueryError):
                live_path = None
            if live_path is not None and live_path.value != raced_text:
                live_path.value = raced_text
                live_path.cursor_position = len(raced_text)
                if focused_id == "library-ingest-path":
                    cursor = len(raced_text)
        if not focused_id:
            return
        try:
            widget = self.query_one(f"#{focused_id}")
        except (NoMatches, QueryError):
            return
        widget.focus(scroll_visible=False)
        if cursor is not None and isinstance(widget, Input):
            widget.cursor_position = min(cursor, len(widget.value))

    def _update_library_ingest_fold_hint(self) -> None:
        """Re-derive the canvas fold indicator after an in-place update.

        (task-3304, MI-08) The hint is canvas-owned, always-mounted,
        display-managed chrome (never conditionally composed); this hook
        exists because queue/summary recomposes change content height
        WITHOUT resizing the canvas, so the canvas's own ``on_resize``
        never fires for them.
        """
        try:
            canvas = self.query_one(LibraryIngestCanvas)
        except (NoMatches, QueryError):
            return
        canvas.sync_fold_hint()

    def _scroll_library_ingest_queue_into_view(self) -> None:
        """Bring the queue heading into view after a submit (MI-08).

        Every submit used to land its outcome rows below the fold: the
        acknowledgement existed but was invisible. ``top=True`` parks the
        heading at the top of the viewport so the freshly queued rows
        below it are what the user sees.
        """
        try:
            canvas = self.query_one(LibraryIngestCanvas)
            heading = canvas.query_one("#library-ingest-queue-heading", Static)
        except (NoMatches, QueryError):
            return
        canvas.scroll_to_widget(heading, animate=False, top=True)

    @on(Button.Pressed, "#library-ingest-top-button")
    async def _on_library_ingest_top_button(self, event: Button.Pressed) -> None:
        """Jump from the rail-top Ingest button to the Ingest media canvas."""
        event.stop()
        await self._select_library_rail_row(LIBRARY_ROW_INGEST_MEDIA)

    def _focus_library_ingest_path(self) -> None:
        """Focus the Ingest canvas's path field (task-3302 AC#1, MI-03).

        The Ingest entry-focus counterpart of ``_focus_library_skill_name``
        -- scheduled via ``call_after_refresh`` from the rail-row switch so
        typing immediately edits the path instead of feeding whatever
        widget happened to keep focus (the live walk: the rail search box,
        so a pasted path ran a Library search).
        """
        try:
            canvas = self.query_one(LibraryIngestCanvas)
            path_input = self.query_one("#library-ingest-path", Input)
        except (NoMatches, QueryError):
            return
        canvas.scroll_home(animate=False)
        self.set_focus(path_input, scroll_visible=False)

    async def action_library_ingest_back(self) -> None:
        """Escape: leave the Ingest canvas for the hub landing (task-3302).

        Gated by ``check_action`` to the Ingest row, mirroring the other
        six escape gates. Routes through the shared rail-row seam with the
        landing's empty row id, so every switch-hygiene reset (including
        ``_pause_library_ingest_transient_ui`` -- the form itself persists,
        task-2043) applies exactly as a rail press would.
        """
        if self._library_selected_row_id != LIBRARY_ROW_INGEST_MEDIA:
            return
        if self._library_ingest_start_consent is not None:
            # (task-3314 AC#4) Esc is the consent "no": drop the pending
            # two-press confirm and STAY on the canvas -- the user asked
            # to back out of the confirm, not out of the form. A second
            # Esc leaves for the hub as before.
            self._disarm_library_ingest_start_confirm()
            self._update_library_ingest_gate(self._build_library_ingest_state())
            return
        await self._select_library_rail_row("")
        if self.is_mounted:
            self.call_after_refresh(self._focus_library_hub_entry)

    def _adopt_library_ingest_path(self, new_path: str) -> str:
        """Set the staged path from a NON-typing source, invalidation included.

        (xhigh review + live-verify round) ``handle_library_ingest_path_
        changed`` is the only seam that disarms a pending Start consent on
        a path change, and it cannot see a programmatic set: writing
        ``form.path`` first makes the recomposed Input's re-announcement
        equal to the form's copy, which the handler's echo guard drops by
        design. Browse… did exactly that, so a consent armed against file
        A survived picking file B and B could be submitted under A's
        consent. Every non-typing path writer routes through here instead.

        Args:
            new_path: The path the picker (or any other non-typing source)
                chose, exactly as it should appear in the field.

        Returns:
            The previous staged path, for callers that need to know
            whether anything actually changed.
        """
        previous = self._library_ingest_form.path
        self._library_ingest_form.path = new_path
        if new_path != previous:
            # Same rule the typing seam applies: a different source means
            # a different blast radius, so any pending consent is stale --
            # both the Start consent and a pending destructive re-stage
            # (whose whole point is the path it would overwrite).
            self._disarm_library_ingest_start_confirm()
            self._disarm_library_ingest_retry_confirm()
        return previous

    @on(Input.Changed, "#library-ingest-path")
    async def handle_library_ingest_path_changed(self, event: Input.Changed) -> None:
        """Track the ingest path text as the user types it (state only).

        Also live-updates the Start button's disabled state, AND the
        blank-path quiet line (L3b AB wave, A4), via targeted DOM surgery
        (mirroring ``update_library_collection_name_input``) rather than a
        full canvas recompose, so typing never disturbs the Input's cursor
        position. The quiet line ``Static`` is always mounted by
        ``LibraryIngestCanvas.compose`` with a fixed one-row height, so
        this handler only updates its text in place -- never mounts or
        removes it -- keeping the Start button's position stable across
        gate-state changes (2026-07 UAT: mount/remove shifted the button
        ~2 rows on every valid/blank transition).

        Args:
            event: Input change event emitted by the path field.
        """
        event.stop()
        if event.value == self._library_ingest_form.path:
            # Textual re-announces an Input's ``value=`` when a recompose
            # remounts it. Treating that echo as a user edit re-armed the
            # debounce on every recompose (a perpetual pre-flight loop
            # while any path sat in the field) and would fence off a
            # just-started worker for no reason (task-2015 review; same
            # family as the canvas's ``_reported_option_values`` guard).
            return
        self._invalidate_library_external_submission()
        self._library_ingest_form.path = event.value
        # (task-3314) A genuine path edit invalidates the forecast a
        # pending Start consent was armed against; the trailing gate
        # update below re-renders the line out of its confirm state.
        self._disarm_library_ingest_start_confirm()
        # ...and it changes what a pending re-stage would discard, so that
        # consent is stale too (guarded: the label write is off the hot
        # typing path unless something is actually armed).
        if self._library_ingest_retry_confirm_armed:
            self._disarm_library_ingest_retry_confirm()
            self._update_library_ingest_retry_label()
        # (task-2015 review) Fence off any in-flight pre-flight the moment
        # the text genuinely changes: its result describes a path this
        # field no longer shows, and generation equality alone would
        # accept it during the debounce window.
        self._cancel_library_ingest_preflight()
        self._library_ingest_preflight_generation += 1
        # (task-2015) Feedback must not wait for blur: restart the debounce
        # timer on every edit; its fire runs the pre-flight for the text the
        # user has settled on. The blur/submit triggers still run
        # immediately -- this only ADDS the while-typing path.
        timer = self._library_ingest_path_debounce_timer
        if timer is not None:
            timer.stop()
            self._library_ingest_path_debounce_timer = None
        if event.value.strip():
            self._library_ingest_path_debounce_timer = self.set_timer(
                0.8, self._run_debounced_library_ingest_preflight
            )
        else:
            # A cleared field must not keep old errors or a summary parked
            # on screen with nothing staged (task-2015 review). Recompose
            # only when there IS pre-flight state to drop -- the plain
            # typed-then-deleted case keeps this handler's no-recompose
            # contract (cursor/widget identity preserved).
            had_preflight = (
                self._library_ingest_form.preflight is not None
                or self._library_ingest_form.preflight_checking
            )
            self._invalidate_library_ingest_preflight()
            if had_preflight:
                self._update_library_ingest_dynamic_regions()
        # (task-2016) The state model drops intro lines once a path exists,
        # but this handler avoids recomposing while typing -- hide/show them
        # in place so they track the field's content live.
        show_intros = (
            not event.value.strip() and self._library_ingest_form.preflight is None
        )
        for intro in self.query(".library-ingest-intro"):
            intro.display = show_intros
        # (task-2042) The Clear button is always mounted; only its
        # visibility tracks the field, so typing never changes the canvas's
        # widget structure.
        try:
            clear_button = self.query_one("#library-ingest-clear-path", Button)
        except (NoMatches, QueryError):
            pass
        else:
            clear_button.display = bool(event.value.strip())
        self._update_library_ingest_gate(self._build_library_ingest_state())

    @on(Input.Blurred, "#library-ingest-path")
    def handle_library_ingest_path_blurred(self, event: Input.Blurred) -> None:
        """Trigger pre-flight when the user leaves the path field.

        Blur deliberately does NOT disarm a pending Start consent
        (reversal of task-3314 AC#4's original reading; xhigh review +
        live-verify round). The original rule assumed the mouse flow
        always blurs BEFORE the arming press -- true only when the FIRST
        press is the click. Arm with Enter in the path field and the
        confirm copy says "Press Start again to import anyway"; the Start
        CLICK that instruction asks for blurs the path field on its way
        in, so the disarm fired between the gesture and the press handler
        and the second press merely RE-ARMED. Nothing could ever submit
        by the route the copy prescribes.

        A blur carries no information about the forecast: the staged
        path, its options, and its warnings are all unchanged by focus
        moving. Consent is invalidated by things that change what Start
        would DO -- a genuine path edit, an option edit, a fresh
        pre-flight with different warnings, pre-flight invalidation, a
        Browse… pick, a rail switch, or an explicit Esc -- and every one
        of those still disarms.
        """
        event.stop()
        path = self._library_ingest_form.path.strip()
        if path:
            self._trigger_library_ingest_preflight(path)

    @on(Input.Changed, "#library-ingest-title")
    def handle_library_ingest_title_changed(self, event: Input.Changed) -> None:
        """Track the ingest title text as the user types it (state only)."""
        event.stop()
        self._invalidate_library_external_submission()
        changed = event.value != self._library_ingest_form.title
        self._library_ingest_form.title = event.value
        if changed and self._library_ingest_start_consent is not None:
            self._disarm_library_ingest_start_confirm()
            self._update_library_ingest_gate(self._build_library_ingest_state())

    @on(Input.Changed, "#library-ingest-author")
    def handle_library_ingest_author_changed(self, event: Input.Changed) -> None:
        """Track the ingest author text as the user types it (state only)."""
        event.stop()
        self._invalidate_library_external_submission()
        changed = event.value != self._library_ingest_form.author
        self._library_ingest_form.author = event.value
        if changed and self._library_ingest_start_consent is not None:
            self._disarm_library_ingest_start_confirm()
            self._update_library_ingest_gate(self._build_library_ingest_state())

    @on(Input.Changed, "#library-ingest-keywords")
    def handle_library_ingest_keywords_changed(self, event: Input.Changed) -> None:
        """Track the ingest keywords text as the user types it (state only)."""
        event.stop()
        self._invalidate_library_external_submission()
        changed = event.value != self._library_ingest_form.keywords
        self._library_ingest_form.keywords = event.value
        if changed and self._library_ingest_start_consent is not None:
            self._disarm_library_ingest_start_confirm()
            self._update_library_ingest_gate(self._build_library_ingest_state())

    def _apply_library_ingest_backend_save(
        self,
        target: str,
        generation: int,
    ) -> None:
        """Reconcile one current preference completion on the UI thread."""

        if (
            generation != self._library_ingest_backend_generation
            or target != self._library_ingest_backend_target
        ):
            return
        self._library_ingest_backend_target = None
        if (
            self.is_mounted
            and self._library_selected_row_id == LIBRARY_ROW_INGEST_MEDIA
        ):
            _sync_library_canvas(self, "ingest")

    @on(Button.Pressed, "#library-ingest-clear-path")
    def handle_library_ingest_clear_path(self, event: Button.Pressed) -> None:
        """Empty the ingest path field in one press.

        Clearing a long path by hand meant selecting it first; there was no
        affordance for the most common correction on the screen.

        Args:
            event: Button press event emitted by the "Clear" action.
        """
        event.stop()
        self._invalidate_library_external_submission()
        self._invalidate_library_ingest_preflight()
        self._library_ingest_form.path = ""
        # (task-2100) In place: the whole-screen recompose this used to run
        # replaced every canvas widget mid-press -- the one handler the
        # task-2042 sweep missed. Clearing the Input's value directly lets
        # its Changed handler hide the button/show intros, and the updater
        # drops the stale pre-flight summary; focus moves to the path field
        # (the button is about to hide, and a fresh path is the next act).
        try:
            path_input = self.query_one("#library-ingest-path", Input)
        except (NoMatches, QueryError):
            path_input = None
        if path_input is not None:
            # This is a programmatic mirror of the form assignment above,
            # not a second user edit. Suppressing its deferred Changed
            # message prevents that stale empty event from landing after
            # the user's immediate first keystroke and erasing it.
            with path_input.prevent(Input.Changed):
                path_input.value = ""
            # (task-3311) SYNCHRONOUS focus, before the dynamic-region
            # update below. ``Widget.focus()`` defers through
            # ``app.call_later``, so with a pre-flight staged (type-group
            # set change -> the STRUCTURAL branch) the update's
            # ``_refresh_library_ingest_canvas_preserving_context`` still
            # saw the just-clicked Clear button as ``app.focused``,
            # captured THAT id, and after the full recompose tried to
            # restore focus onto the new Clear button -- hidden for an
            # empty path, so ``Screen.set_focus`` silently no-ops on the
            # non-focusable widget and focus stayed wherever the prune's
            # ``_reset_focus`` dropped it (live: the rail search box, or
            # nowhere -- a following "/" then ran the global focus-search
            # binding). Setting focus through the Screen API updates
            # ``screen.focused`` immediately, so the capture/restore
            # round-trips ``#library-ingest-path`` deterministically.
            self.set_focus(path_input, scroll_visible=False)
        # Clear is the one transition where the staged type-group set is
        # guaranteed to shrink. Hide those stale panels immediately and run
        # the ordinary in-place region update without a full canvas remount;
        # the next completed preflight may structurally rebuild for its new
        # group set. This removes the keystroke-loss window altogether.
        try:
            canvas = self.query_one(LibraryIngestCanvas)
        except (NoMatches, QueryError):
            canvas = None
        if canvas is not None:
            cleared_groups = set(self._build_library_ingest_state().type_groups)
            for panel in canvas.query(Collapsible):
                panel_id = panel.id or ""
                if panel_id.startswith("type-group-"):
                    group = panel_id[len("type-group-") :]
                    panel.display = group in cleared_groups
            for bulk in canvas.query(".library-ingest-options-bulk"):
                bulk.display = len(cleared_groups) > 1
        self._update_library_ingest_dynamic_regions(False)

    @on(Button.Pressed, "#library-ingest-retry-last")
    def handle_library_ingest_retry_last(self, event: Button.Pressed) -> None:
        """Re-stage the last submitted batch into the form (task-3313).

        Args:
            event: Button press event emitted by the "Retry this batch"
                action.
        """
        event.stop()
        self._restage_library_ingest_last_submission()

    def action_library_ingest_retry_last(self) -> None:
        """``r``: keyboard route to "Retry this batch" (task-3313 AC#2).

        Gated by ``check_action`` to the Ingest canvas while the affordance
        is offered; a printable key is only ever seen here when no
        Input/TextArea holds focus (text fields consume it first), so
        typing an ``r`` into the path field never re-stages.
        """
        self._restage_library_ingest_last_submission()

    def _disarm_library_ingest_retry_confirm(self) -> None:
        """Drop a pending destructive-re-stage consent (state only)."""
        self._library_ingest_retry_confirm_armed = False

    def _library_ingest_restage_discards_work(self) -> bool:
        """Whether re-staging would overwrite content the user entered.

        A re-stage is destructive only insofar as the form currently holds
        something DIFFERENT from what the snapshot would put back. Right
        after a submit it does not: path and title are cleared and the
        remaining metadata/options are the very values the snapshot
        carries, so re-staging replaces nothing and consent would be pure
        friction. Once the user has typed a new path, a new title, or
        flipped an option, the same press silently destroys that work --
        which is what the consent exists for.

        Returns:
            ``True`` when at least one non-empty form field (or option
            value) differs from what the last-submission snapshot would
            restore over it.
        """
        snapshot = self._library_ingest_last_submission
        if snapshot is None:
            return False
        form = self._library_ingest_form
        for current, staged in (
            (form.path, snapshot.source),
            (form.title, snapshot.title),
            (form.author, snapshot.author),
            (form.keywords, snapshot.keywords),
        ):
            # An empty field has nothing to lose; only entered text does.
            if current.strip() and current.strip() != str(staged).strip():
                return True
        for group, staged_values in snapshot.type_options.items():
            current_values = form.type_options.get(group, {})
            for name, staged_value in staged_values.items():
                if name in current_values and current_values[name] != staged_value:
                    return True
        return False

    def _update_library_ingest_retry_label(self) -> None:
        """Sync the retry affordance's label in place (never a recompose).

        The affordance is the pending consent's only surface -- the ``r``
        route has no gate line of its own -- so arming has to be visible
        without disturbing the form the user is mid-way through.
        """
        try:
            retry_button = self.query_one("#library-ingest-retry-last", Button)
        except (NoMatches, QueryError):
            return
        retry_button.label = library_ingest_retry_label(
            self._library_ingest_retry_confirm_armed
        )

    def _restage_library_ingest_last_submission(self) -> None:
        """Restore the last submission's source/options/metadata (task-3313).

        Destructive by construction: every form field is replaced from the
        snapshot with no undo. When that would discard work the user has
        entered since the submit, the repo's incumbent two-press consent
        applies (Clear-finished, task-2015/2160; the Start consent,
        task-3314) -- the FIRST press only arms, relabelling the
        affordance, and the second replaces the form. A form holding
        nothing the re-stage would discard skips consent and re-stages on
        one press. Both routes (button and the ``r`` accelerator) share
        this one path, so the key can never be the looser of the two.

        The old forecast must never be reused (AC#3): the stale pre-flight
        echo is invalidated first, then a FRESH pre-flight runs through
        the same trigger the typing path uses -- so tooling installed (or
        removed) since the last run changes the forecast and the gate.
        The context-preserving recompose re-renders the form widgets from
        the restored echo; focus lands in the path field, matching entry
        focus (the staged path is what the user acts on next).
        """
        snapshot = self._library_ingest_last_submission
        if snapshot is None:
            return
        if self._library_ingest_restage_discards_work():
            # ``getattr`` quiet-degrade, the convention several suites rely
            # on: they build this screen via ``object.__new__`` and seed
            # only the fields they exercise.
            if not getattr(self, "_library_ingest_retry_confirm_armed", False):
                self._library_ingest_retry_confirm_armed = True
                self._library_ingest_retry_confirm_armed_at = time.monotonic()
                self._update_library_ingest_retry_label()
                return
            if (
                time.monotonic() - self._library_ingest_retry_confirm_armed_at
                < self._RETRY_CONFIRM_DEAD_ZONE_SECONDS
            ):
                # A press inside the dead zone is the arming gesture
                # repeating, not consent -- ignore it (stays armed).
                return
        self._disarm_library_ingest_retry_confirm()
        form = self._library_ingest_form
        form.path = snapshot.source
        form.title = snapshot.title
        form.author = snapshot.author
        form.keywords = snapshot.keywords
        form.analyze = snapshot.analyze
        form.chunk = snapshot.chunk
        form.chunk_size = snapshot.chunk_size
        form.type_options = {
            group: dict(values) for group, values in snapshot.type_options.items()
        }
        self._invalidate_library_ingest_preflight()
        self._refresh_library_ingest_canvas_preserving_context()
        self._trigger_library_ingest_preflight(snapshot.source)
        self.call_after_refresh(self._focus_library_ingest_path)

    @on(Button.Pressed, "#ingest-preflight-choose")
    @on(Button.Pressed, "#library-ingest-browse")
    def handle_library_ingest_browse(self, event: Button.Pressed) -> None:
        """Push a ``FileOpen`` dialog to pick a local file to ingest.

        Mirrors the reviewed import dialog flow exactly (the
        working ``FileOpen`` reference, invoked the same simple
        ``title=``-only way). The callback writes the chosen path straight
        into the form and recomposes so the Input and the Start button's
        gate both reflect it immediately; validation still runs at Start
        so a path typed by hand (not picked via this dialog) is caught
        too.

        Args:
            event: Button press event emitted by the "Browse…" action.
        """
        event.stop()

        async def browse_callback(selected_path: Path | None) -> None:
            if selected_path is None:
                return
            self._invalidate_library_external_submission()
            self._persist_library_ingest_location(selected_path)
            self._adopt_library_ingest_path(str(selected_path))
            self.refresh(recompose=True)
            self._trigger_library_ingest_preflight(str(selected_path))

        self.app.push_screen(
            FileOpen(
                location=self._library_ingest_browse_location(),
                title="Import media",
                filters=_ingestible_file_filters(),
                # (task-2222 owner ruling) Folder import must be pickable,
                # not type-only: "Open" keeps descending, this returns the
                # folder on screen.
                offer_select_folder=True,
            ),
            browse_callback,
        )

    @on(LibraryIngestCanvas.OptionPanelToggled)
    def sync_library_ingest_type_group_expanded(
        self,
        event: LibraryIngestCanvas.OptionPanelToggled,
    ) -> None:
        """Track per-type panel expand/collapse so recomposes preserve the user's choice."""
        event.stop()
        if event.expanded:
            self._library_ingest_form.expanded_type_groups.add(event.group)
        else:
            self._library_ingest_form.expanded_type_groups.discard(event.group)

    @on(LibraryIngestCanvas.ToolingDetailToggled)
    def sync_library_ingest_tooling_detail_expanded(
        self,
        event: LibraryIngestCanvas.ToolingDetailToggled,
    ) -> None:
        """Track the "What's missing" fold so recomposes preserve it.

        (review round) The summary widget keeps its own expansion across
        the in-place ``refresh(recompose=True)`` a registry tick fires,
        but a STRUCTURAL recompose rebuilds the widget itself -- so the
        durable copy lives in the form echo, exactly like
        ``expanded_type_groups`` above.
        """
        event.stop()
        self._library_ingest_form.tooling_detail_expanded = bool(event.expanded)

    def _cancel_library_ingest_preflight(self) -> None:
        """Cancel any in-flight pre-flight worker, ignoring a finished one."""
        worker = self._library_ingest_preflight_worker
        if worker is None:
            return
        try:
            if not worker.is_finished:
                worker.cancel()
        except Exception:
            pass
        self._library_ingest_preflight_worker = None

    def _invalidate_library_ingest_preflight(self) -> None:
        """Drop the current pre-flight echo AND fence off in-flight workers.

        Worker cancellation is cooperative (``@work(thread=True)``), so a
        worker that has already finished analysing can still deliver its
        result after this runs. Bumping the generation makes
        ``_apply_library_ingest_preflight_result`` drop that late result
        instead of resurrecting the summary this method just cleared
        (task-2011: observed live as "Enter a file path to start." rendered
        together with "1 plain text file · 277 B").
        """
        self._library_ingest_preflight_generation += 1
        self._cancel_library_ingest_preflight()
        self._library_ingest_form.preflight = None
        self._library_ingest_form.preflight_checking = False
        # (task-3314) A consent armed against a forecast that no longer
        # exists must not survive it -- submit/Clear/reset all route here.
        self._disarm_library_ingest_start_confirm()

    def _trigger_library_ingest_preflight(
        self, path: str, *, allow_probe: bool = True
    ) -> None:
        """Start (or restart) the pre-flight worker for ``path``.

        No-op for empty paths so stray focus/blur/enter events never scan
        the current working directory.

        Args:
            path: The staged source path or URL.
            allow_probe: Whether a URL source may be probed over the
                network. ``False`` on the while-typing path (TASK-19556):
                a debounce fire is not the user asking to contact a host,
                so it never does -- even when the (default-off) probe has
                been opted in. The deliberate triggers -- blur, Enter,
                Browse…, the retry button -- leave this at ``True`` and let
                the config gate decide.
        """
        if not path.strip():
            self._library_ingest_form.preflight_checking = False
            return
        self._cancel_library_ingest_preflight()
        self._library_ingest_preflight_generation += 1
        generation = self._library_ingest_preflight_generation
        self._library_ingest_form.preflight_checking = True
        # (task-2042) In-place: only the summary child re-renders, so a
        # trigger landing mid-word can neither steal focus nor swallow a
        # click in flight.
        self._update_library_ingest_dynamic_regions()
        self._library_ingest_preflight_worker = self._run_library_ingest_preflight(
            path, generation, allow_probe
        )

    def _apply_library_ingest_preflight_result(
        self,
        result: PreflightResult,
        generation: int,
    ) -> None:
        """Merge a pre-flight result into the form echo and refresh.

        Drops results from a superseded generation: a clear/submit or a
        newer trigger bumped the counter after this worker started, so its
        result describes a path the form is no longer showing (task-2011).
        """
        if generation != self._library_ingest_preflight_generation:
            return
        self._library_ingest_form.preflight = result
        self._library_ingest_form.preflight_checking = False
        # Re-evaluate the complete request identity after a fresh result.
        # An identical refresh keeps consent; changed warnings or candidate
        # membership revoke it. This preview uses only the captured result.
        armed = getattr(self, "_library_ingest_start_consent", None)
        if armed is not None:
            submitted_source = self._library_ingest_form.path.strip()
            pending = self._current_library_ingest_start_consent(submitted_source)
            if pending.fingerprint != armed.fingerprint:
                self._disarm_library_ingest_start_confirm()
        # (task-2042) In-place for the same reason as the trigger: the
        # result can land while the user is typing or mid-click.
        self._update_library_ingest_dynamic_regions()

    @on(Button.Pressed, "#library-ingest-start")
    def handle_library_ingest_start(self, event: Button.Pressed) -> None:
        """Validate the form and submit a new Library ingest job.

        Args:
            event: Button press event emitted by the "Start import" action.
        """
        event.stop()
        self._submit_library_ingest_form()

    @on(Input.Submitted, "#library-ingest-path")
    def handle_library_ingest_path_submitted(self, event: Input.Submitted) -> None:
        """Submit the ingest form when Enter is pressed in the path field.

        Mirrors the Start import button exactly, but only when the Start
        gate is open (``start_enabled``) -- Enter on a blank path (or with
        the registry/DB unavailable) stays quiet instead of nagging, since
        the always-visible gate line already explains the blocker
        (2026-07 UAT: Enter in a valid path field previously did nothing).
        Pre-flight is also triggered so a final Enter in a non-submitting
        path still refreshes the summary.

        Args:
            event: Input submission event emitted by the path field.
        """
        event.stop()
        self._trigger_library_ingest_preflight(self._library_ingest_form.path)
        if not self._build_library_ingest_state().start_enabled:
            return
        self._submit_library_ingest_form()

    def _disarm_library_ingest_start_confirm(self) -> None:
        """Drop a pending two-press Start consent.

        Called wherever the request the consent was armed against changes:
        source, form/options, backend, active-job membership, warning set,
        pre-flight invalidation, rail reset, or Escape. Focus movement and
        an identical pre-flight refresh deliberately preserve it.
        """
        if getattr(self, "_library_ingest_start_consent", None) is None:
            return
        self._library_ingest_start_consent = None
        self._sync_library_emergency_guard_presentation()

    def _current_library_ingest_start_consent(
        self,
        submitted_source: str,
        gate_state: LibraryIngestCanvasState | None = None,
    ) -> _LibraryIngestStartConsent:
        """Snapshot the exact request and active membership awaiting consent.

        Candidate paths come only from the staged source or the already-captured
        pre-flight result. This method never scans a directory or touches the
        filesystem.
        """
        form = self._library_ingest_form
        resolve_backend = getattr(self.app_instance, "_resolve_ingest_backend", None)
        resolved_backend = resolve_backend() if callable(resolve_backend) else "local"
        backend = "server" if resolved_backend == "server" else "local"
        resolved_gate_state = gate_state or self._build_library_ingest_state()
        forecast = getattr(resolved_gate_state, "forecast", None)
        if forecast is None:
            forecast = build_ingest_forecast(
                form.preflight,
                targets_server=backend == "server",
            )
        tooling_affected_count = int(getattr(forecast, "consent_affected", 0) or 0)

        candidates: list[str] = []
        if form.preflight is not None:
            for paths in form.preflight.type_groups.values():
                candidates.extend(str(path) for path in paths)

        def normalized(source: str):
            try:
                return normalize_active_ingest_source(source, origin=backend)
            except (TypeError, ValueError, OSError):
                return None

        submitted_key = normalized(submitted_source)
        candidate_keys = {
            key
            for candidate in candidates
            if (key := normalized(candidate)) is not None
        }
        is_folder = bool(
            candidates
            and submitted_key is not None
            and submitted_key not in candidate_keys
        )
        preview_sources = candidates if is_folder else [submitted_source]
        registry = self._library_ingest_registry()
        find_matches = getattr(registry, "find_active_source_matches", None)
        matches = (
            find_matches(preview_sources, origin=backend)
            if callable(find_matches)
            else ()
        )
        active_job_ids = tuple(job.job_id for job in matches)
        matched_source_keys = {
            key for job in matches if (key := normalized(job.source_path)) is not None
        }
        request_payload = {
            "source": submitted_source,
            "backend": backend,
            "title": self._safe_text(form.title, max_length=300),
            "author": self._safe_text(form.author, max_length=200),
            "keywords": parse_keywords(form.keywords),
            "options": self._build_ingest_options_snapshot(),
            "warnings": form.preflight.warnings if form.preflight else [],
        }
        admission_scope = build_active_ingest_consent_scope(
            preview_sources,
            origin=backend,
            active_job_ids=active_job_ids,
            active_source_count=len(matched_source_keys),
        )
        request_fingerprint = json.dumps(
            request_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )
        fingerprint_payload = {
            **request_payload,
            "candidate_digest": admission_scope.candidate_digest,
            "candidate_count": admission_scope.candidate_count,
            "tooling_affected_count": tooling_affected_count,
            "active_job_ids": admission_scope.active_job_ids,
            "active_job_count": admission_scope.active_job_count,
            "active_job_ids_complete": admission_scope.active_job_ids_complete,
        }
        consent_context_payload = {
            key: value
            for key, value in fingerprint_payload.items()
            if key
            not in {
                "active_job_ids",
                "active_job_count",
                "active_job_ids_complete",
            }
        }
        consent_context_fingerprint = json.dumps(
            consent_context_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )
        fingerprint = json.dumps(
            fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )
        return _LibraryIngestStartConsent(
            fingerprint=fingerprint,
            admission_scope=admission_scope,
            tooling_affected_count=tooling_affected_count,
            is_folder=is_folder,
            request_fingerprint=request_fingerprint,
            consent_context_fingerprint=consent_context_fingerprint,
        )

    def _authoritative_library_ingest_consent_is_current(
        self,
        armed: _LibraryIngestStartConsent,
        pending: _LibraryIngestStartConsent,
    ) -> bool:
        """Validate only the bounded refusal IDs without re-scanning sources."""
        if (
            not armed.authoritative_refusal
            or armed.consent_context_fingerprint != pending.consent_context_fingerprint
        ):
            return False
        if armed.candidate_changed and not armed.active_job_ids:
            return True
        if pending.active_job_ids not in ((), armed.active_job_ids):
            return False
        registry = self._library_ingest_registry()
        get_job = getattr(registry, "get_job", None)
        if not callable(get_job):
            return False
        for job_id in armed.active_job_ids:
            job = get_job(job_id)
            if (
                job is None
                or job.superseded
                or job.dismissed
                or job.state not in ACTIVE_INGEST_STATES
            ):
                return False
        return bool(armed.active_job_ids)

    def _submit_library_ingest_form(self) -> None:
        """Validate the ingest form and submit a new Library ingest job.

        Shared by the Start import button and Enter in the path field. An
        invalid/missing path is a quiet warning notice, matching every
        other Library form failure path in this screen; a missing
        ``submit_library_ingest_job`` seam (registry absent) gets the same
        treatment. When pre-flight tooling warnings are present, consent
        is the inline two-press grammar (task-3314; the guardrail modal is
        retired): the FIRST press arms -- the gate line becomes "⚠ Press
        Start again to import anyway — N files may fail." via an in-place
        gate update -- and the SECOND press (outside the double-press dead
        zone) submits. Enter in the path field routes through this same
        method, so Enter,Enter carries identical semantics. On success,
        the path AND title fields clear (L3b AB wave, A1) -- title is
        per-file, so it must not silently reapply to the next file in a
        batch -- while author/keywords/advanced options persist, since
        those are batch metadata a user submitting several files in a row
        shouldn't have to retype for every submission.
        """
        form = self._library_ingest_form
        submitted_source = self._resolve_ingest_source(form.path.strip())
        if submitted_source is None:
            return
        submit = getattr(self.app_instance, "submit_library_ingest_job", None)
        if not callable(submit):
            self._notify_library_ingest_warning(INGEST_UNAVAILABLE_COPY)
            return
        # (task-14823) A selection the pre-flight already knows can import
        # nothing must not reach the queue: an empty folder used to leave
        # Start enabled, and the press manufactured "✗ failed · emptydir ·
        # No files to import were found in this folder." -- a permanent
        # failure receipt, in the tally and in Recent imports, for an
        # outcome that was predictable before the click. Enforced HERE,
        # not only on the Button's disabled flag, so no entry point
        # (Enter, accelerator, a future caller) can route around it.
        # (task-14911) The same flag now also covers a selection the
        # TARGETED SERVER refuses entirely -- a folder of images, which
        # this machine imports happily -- so this one refusal serves both
        # backends and the reason it quotes is the gate's own, whichever
        # of the two it is.
        gate_state = self._build_library_ingest_state()
        if gate_state.selection_has_nothing_importable is True:
            reason = gate_state.start_quiet_line
            if reason:
                self._notify_library_ingest_warning(reason)
            return
        pending = self._current_library_ingest_start_consent(
            submitted_source, gate_state
        )
        armed = self._library_ingest_start_consent
        if armed is not None and self._authoritative_library_ingest_consent_is_current(
            armed, pending
        ):
            pending = armed
        if pending.owed:
            if armed is None or armed.fingerprint != pending.fingerprint:
                self._library_ingest_start_consent = pending
                self._library_ingest_start_confirm_armed_at = time.monotonic()
                self._update_library_ingest_gate(self._build_library_ingest_state())
                self._sync_library_emergency_guard_presentation()
                return
            if (
                time.monotonic() - self._library_ingest_start_confirm_armed_at
                < self._START_CONFIRM_DEAD_ZONE_SECONDS
            ):
                return
            duplicate_consent = (
                armed.admission_scope
                if armed.allows_active_duplicate or armed.candidate_changed
                else None
            )
            confirmed_consent = armed if duplicate_consent is not None else None
            self._disarm_library_ingest_start_confirm()
        else:
            duplicate_consent = None
            confirmed_consent = None
            if armed is not None:
                self._disarm_library_ingest_start_confirm()
        self._do_submit_ingest(
            submitted_source,
            active_duplicate_consent=duplicate_consent,
            confirmed_consent=confirmed_consent,
        )

    @staticmethod
    def _ingest_job_id_from_button(button_id: str | None, prefix: str) -> str | None:
        """Parse a job id from a Library-ingest row-action button id.

        Row-action buttons (``library-ingest-open-{job_id}``/``-retry-``/
        ``-dismiss-``) are keyed by the registry-assigned ``job_id``, NOT
        by row index (PR #591 review, F1): the queue mutates
        asynchronously between a render and a click (runner completions,
        retry-supersede, new submissions), so re-deriving a fresh row
        snapshot and indexing into it at click time can silently resolve
        to a DIFFERENT job than the one the user actually pressed. A
        prefix-strip is exact regardless of how the queue has shifted
        since the button was rendered.

        Args:
            button_id: The pressed button's ``id``.
            prefix: The button-id prefix to strip (e.g.
                ``"library-ingest-open-"``).

        Returns:
            The job id, or ``None`` when ``button_id`` is missing or
            doesn't carry the expected prefix (defensive only -- every
            real row-action button always does).
        """
        if not button_id or not button_id.startswith(prefix):
            return None
        job_id = button_id[len(prefix) :]
        return job_id or None

    @on(Button.Pressed, ".library-ingest-open")
    async def handle_library_ingest_open(self, event: Button.Pressed) -> None:
        """Open a done ingest job's resulting media item in the Library viewer.

        Args:
            event: Button press event emitted by an "Open in Library" row action.
        """
        event.stop()
        job_id = self._ingest_job_id_from_button(
            event.button.id, "library-ingest-open-"
        )
        if job_id is None:
            return
        job = self._library_ingest_job_by_id(job_id)
        if job is None:
            return
        self._open_job_in_library(job)

    @on(Button.Pressed, ".library-ingest-view-server")
    def handle_library_ingest_view_on_server(self, event: Button.Pressed) -> None:
        """Open the item a finished SERVER ingest created, in the server view.

        Distinct from "Open in Library": that resolves a row in this machine's
        media DB, and a server ingest never wrote one. The server reports the id
        of the row it made, so the item is addressable -- against the server
        (task-700).

        Args:
            event: Button press event emitted by a "View on server" row action.
        """
        event.stop()
        job_id = self._ingest_job_id_from_button(
            event.button.id, "library-ingest-view-server-"
        )
        if job_id is None:
            return
        job = self._library_ingest_job_by_id(job_id)
        remote_media_id = getattr(job, "remote_media_id", None) if job else None
        if not remote_media_id:
            # The row only offers this action when the id is present, so this is
            # a stale-press guard rather than an expected path.
            self.notify("The server did not report which item it created.")
            return
        self.run_worker(self._open_library_external_media_detail(str(remote_media_id)))

    @on(Button.Pressed, ".library-ingest-retry")
    def handle_library_ingest_retry(self, event: Button.Pressed) -> None:
        """Requeue a failed ingest job.

        Args:
            event: Button press event emitted by a "Retry" row action.
        """
        event.stop()
        job_id = self._ingest_job_id_from_button(
            event.button.id, "library-ingest-retry-"
        )
        if job_id is None:
            return
        retry = getattr(self.app_instance, "retry_library_ingest_job", None)
        if callable(retry):
            # The shared app seam validates the exact job and chooses either
            # ordinary registry requeueing or the Research operation owner.
            # A stale or now-wrong-state id is a safe no-op.
            retry(job_id)
        # (task-2100) In place: the registry listener already updated the
        # queue; a trailing full recompose yanked the scroll off the queue.
        self._update_library_ingest_dynamic_regions()

    @on(Button.Pressed, ".library-ingest-choose-gguf")
    def handle_library_ingest_choose_gguf(self, event: Button.Pressed) -> None:
        """Choose a replacement GGUF and requeue the same manual provider."""
        event.stop()
        job_id = self._ingest_job_id_from_button(
            event.button.id, "library-ingest-choose-gguf-"
        )
        if job_id is not None:
            self._open_transcribe_cpp_gguf_picker(retry_job_id=job_id)

    @on(Button.Pressed, ".library-ingest-retry-faster-whisper")
    def handle_library_ingest_retry_faster_whisper(self, event: Button.Pressed) -> None:
        """Explicitly retry a direct-local failure with faster-whisper."""
        event.stop()
        job_id = self._ingest_job_id_from_button(
            event.button.id, "library-ingest-retry-faster-whisper-"
        )
        if job_id is None:
            return
        retry = getattr(
            self.app_instance, "retry_library_ingest_job_with_provider", None
        )
        if callable(retry):
            retry(job_id, "faster-whisper")
        # (task-2100) In place: the registry listener already updated
        # the queue; a trailing full recompose yanked the scroll off
        # the queue (stranding the armed clear confirm off-screen).
        self._update_library_ingest_dynamic_regions()

    @on(Button.Pressed, ".library-ingest-cancel")
    def handle_library_ingest_cancel(self, event: Button.Pressed) -> None:
        """Request cancellation of an in-flight local attempt or server batch.

        Local STT rows address their exact executor attempt. Server rows address
        their batch and remain asynchronous until polling records the outcome.

        A quiet no-op when the registry, the job, its batch id or the server
        seam is missing, matching every other seam-absent path in this screen.

        Args:
            event: Button press event emitted by a row's "Cancel" action.
        """
        event.stop()
        job_id = self._ingest_job_id_from_button(
            event.button.id, "library-ingest-cancel-"
        )
        if job_id is None:
            return
        registry = self._library_ingest_registry()
        get_job = getattr(registry, "get_job", None)
        job = get_job(job_id) if callable(get_job) else None
        if job is None:
            return
        if getattr(job, "origin", "local") == "local":
            request_cancel = getattr(
                self.app_instance,
                "cancel_local_ingest_job",
                None,
            )
            if callable(request_cancel):
                request_cancel(job_id)
                self.refresh(recompose=True)
            return
        if not getattr(job, "batch_id", None):
            return
        request_cancel = getattr(self.app_instance, "cancel_remote_ingest_batch", None)
        if not callable(request_cancel):
            self._notify_library_ingest_warning(
                "Cancelling a server import is unavailable in this runtime."
            )
            return
        request_cancel(job.batch_id)

    @on(Button.Pressed, ".library-ingest-force-stop")
    def handle_library_ingest_force_stop(self, event: Button.Pressed) -> None:
        """Force-stop one local STT attempt after cooperative cancellation."""

        event.stop()
        job_id = self._ingest_job_id_from_button(
            event.button.id,
            "library-ingest-force-stop-",
        )
        if job_id is None:
            return
        force_stop = getattr(
            self.app_instance,
            "force_stop_local_ingest_job",
            None,
        )
        if callable(force_stop):
            force_stop(job_id)
            self.refresh(recompose=True)

    @on(Button.Pressed, ".library-ingest-dismiss")
    def handle_library_ingest_dismiss(self, event: Button.Pressed) -> None:
        """Dismiss a failed ingest job row (L3b AB wave, B2).

        A thin wrapper over ``LibraryIngestJobRegistry.dismiss`` -- valid
        only for a ``FAILED`` row; a quiet no-op (mirrors every other
        Library seam-absent path in this screen) when the registry itself
        is unavailable. The registry's own listener
        (``_handle_library_ingest_registry_changed``) already recomposes
        on a successful dismiss; the trailing ``refresh(recompose=True)``
        here is redundant-but-harmless belt-and-braces, matching
        ``handle_library_ingest_retry``.

        Args:
            event: Button press event emitted by a "Dismiss" row action.
        """
        event.stop()
        job_id = self._ingest_job_id_from_button(
            event.button.id, "library-ingest-dismiss-"
        )
        if job_id is None:
            return
        registry = self._library_ingest_registry()
        dismiss = getattr(registry, "dismiss", None)
        if callable(dismiss):
            # Same id-based no-op safety as retry above -- ``dismiss`` only
            # ever acts on a currently-FAILED, not-yet-hidden job_id.
            dismissed_job = dismiss(job_id)
            # (task-2140) Dismiss was the one destructive act that erased
            # the failure from EVERY surface with zero friction -- the
            # dismissed record now survives in the Recent-ingests ledger,
            # marked as dismissed.
            if dismissed_job is not None:
                known = {job.job_id for job in self._library_ingest_recent_ledger}
                if dismissed_job.job_id not in known:
                    self._library_ingest_recent_ledger = [
                        dismissed_job,
                        *self._library_ingest_recent_ledger,
                    ][:10]
        self._library_ingest_expanded_details.discard(job_id)
        # (task-2100) In place: the registry listener already updated the
        # queue; a trailing full recompose here yanked the scroll off the
        # queue the user was working in.
        self._update_library_ingest_dynamic_regions()

    @on(Button.Pressed, ".library-ingest-details")
    def _on_ingest_job_details(self, event: Button.Pressed) -> None:
        """Toggle a failed row's inline error-detail lines (task-2043).

        Replaces the old auto-expiring notification: a toast gave the user
        ~4 unre-readable seconds with an uncopyable error. The lines render
        under the row via the in-place queue update, and the button flips
        Show/Hide.

        Args:
            event: Button press event emitted by a "Show details" row action.
        """
        event.stop()
        job_id = self._ingest_job_id_from_button(
            event.button.id, "library-ingest-details-"
        )
        if job_id is None:
            return
        if job_id in self._library_ingest_expanded_details:
            self._library_ingest_expanded_details.discard(job_id)
        else:
            self._library_ingest_expanded_details.add(job_id)
        self._update_library_ingest_dynamic_regions()

    @on(Button.Pressed, "#library-ingest-clear-finished")
    def handle_library_ingest_clear_finished(self, event: Button.Pressed) -> None:
        """Clear every done+failed ingest job in one shot (L3b AB wave, B2).

        A thin wrapper over ``LibraryIngestJobRegistry.clear_finished``; a
        quiet no-op when the registry itself is unavailable (matching
        ``handle_library_ingest_dismiss``/``handle_library_ingest_retry``).

        Args:
            event: Button press event emitted by the "Clear finished" action.
        """
        event.stop()
        # (task-2015) One unconfirmed press used to destroy every finished
        # row -- the only receipts an ingest leaves. First press arms (the
        # button label names what a second press removes); second press
        # clears. A registry mutation between the presses disarms (see
        # ``_handle_library_ingest_registry_changed``).
        if not self._library_ingest_clear_finished_armed:
            self._library_ingest_clear_finished_armed = True
            self._library_ingest_clear_finished_armed_at = time.monotonic()
            # (task-2160) Arming changes ONLY the button's label, in place.
            # Two rounds of scroll repair (2130's immediate scroll_visible,
            # 2140's call_after_refresh) both lost to the queue-panel
            # recompose yanking a tall queue's viewport to its top -- the
            # cure is to not disturb layout at all: no recompose, no
            # scroll, the confirm appears under the finger that armed it.
            try:
                armed_button = self.query_one("#library-ingest-clear-finished", Button)
            except (NoMatches, QueryError):
                self._update_library_ingest_dynamic_regions()
            else:
                armed_button.label = (
                    self._build_library_ingest_state().queue_clear_finished_label
                )
                # The label got longer; without a layout pass the
                # auto-width compact button keeps its old width and clips
                # the confirm copy (live-caught: "Press again to").
                armed_button.refresh(layout=True)
            return
        # (task-2160) Double-click protection: a press landing within the
        # dead zone of the arming press is the same gesture, not a
        # decision -- ignore it (stays armed).
        armed_at = getattr(self, "_library_ingest_clear_finished_armed_at", 0.0)
        if time.monotonic() - armed_at < self._CLEAR_FINISHED_DEAD_ZONE_SECONDS:
            return
        self._library_ingest_clear_finished_armed = False
        self._library_ingest_expanded_details.clear()
        registry = self._library_ingest_registry()
        # (task-2130) Snapshot the terminal jobs into the session ledger
        # BEFORE the removal -- Recent imports is the durable record.
        jobs_fn = getattr(registry, "jobs", None)
        if callable(jobs_fn):
            terminal = [
                job
                for job in jobs_fn()
                if job.state
                in (
                    IngestJobState.DONE,
                    IngestJobState.FAILED,
                    IngestJobState.SKIPPED,
                )
            ]
            known = {job.job_id for job in terminal}
            terminal.extend(
                job
                for job in self._library_ingest_recent_ledger
                if job.job_id not in known
            )
            self._library_ingest_recent_ledger = terminal[:10]
        clear_finished = getattr(registry, "clear_finished", None)
        if callable(clear_finished):
            clear_finished()
        # (task-2100) In place: the registry listener already updated
        # the queue; a trailing full recompose yanked the scroll off
        # the queue (stranding the armed clear confirm off-screen).
        self._update_library_ingest_dynamic_regions()

    @on(Button.Pressed, "#ingest-expand-all")
    def handle_library_ingest_expand_all(self, event: Button.Pressed) -> None:
        """Expand every per-type options panel."""
        event.stop()
        form = self._library_ingest_form
        state = self._build_library_ingest_state()
        form.expanded_type_groups.update(state.type_groups)
        # (task-2100 review) Panel collapsed state is set at compose time,
        # so the in-place updater alone leaves mounted panels shut -- write
        # `collapsed` on them directly (Textual's reactive handles the
        # reveal), keeping the press non-structural.
        self._set_library_ingest_panels_collapsed(state.type_groups, False)
        self._update_library_ingest_dynamic_regions()

    @on(Button.Pressed, "#ingest-collapse-all")
    def handle_library_ingest_collapse_all(self, event: Button.Pressed) -> None:
        """Collapse every per-type options panel."""
        event.stop()
        form = self._library_ingest_form
        state = self._build_library_ingest_state()
        form.expanded_type_groups.clear()
        self._set_library_ingest_panels_collapsed(state.type_groups, True)
        self._update_library_ingest_dynamic_regions()

    def _set_library_ingest_panels_collapsed(
        self, groups: Sequence[str], collapsed: bool
    ) -> None:
        for group in groups:
            try:
                panel = self.query_one(f"#type-group-{group}", Collapsible)
            except (NoMatches, QueryError):
                continue
            panel.collapsed = collapsed

    def _update_library_ingest_group_receipt(self, group: str) -> None:
        """Recompute one panel's title receipt from the ACTUAL option values."""
        cap = get_capabilities(group)
        resolve_backend = getattr(self.app_instance, "_resolve_ingest_backend", None)
        backend = resolve_backend() if callable(resolve_backend) else "local"
        visible_cap = capabilities_for_backend(cap, backend)
        values = dict(self._library_ingest_form.type_options.get(group, {}))
        if group == "generic":
            form = self._library_ingest_form
            values.setdefault("analyze", form.analyze)
            values["analyze"] = form.analyze
            values["chunk"] = form.chunk
            values["chunk_size"] = form.chunk_size
        try:
            panel = self.query_one(f"#type-group-{group}", Collapsible)
        except (NoMatches, QueryError):
            return
        # (task-3305, MI-16) One title builder for compose-time and this
        # in-place path: capped, empty-skipping, changed-values-first.
        panel.title = build_type_group_title(visible_cap, values)

# --- BEGIN generated ingest-controller-state shims ---
# Permanent, not a cleanup-PR deletion target -- same reasoning as
# `LibraryExportController`'s own identical block: the byte-for-byte canon
# (recipe §1) forbids editing a moved body, so the attribute names those
# bodies already use have to keep resolving through *something*. Exposes
# every `LibraryIngestState` field under its original `_library_ingest_
# <field>` name (single prefix, no plural variant -- see task 1's own
# `LibraryIngestState` module docstring).
for _lic_field in dataclasses.fields(LibraryIngestState):
    setattr(
        LibraryIngestController,
        "_library_ingest_" + _lic_field.name,
        property(
            lambda self, _n=_lic_field.name: getattr(
                self._ingest_state_accessor(), _n
            ),
            lambda self, value, _n=_lic_field.name: setattr(
                self._ingest_state_accessor(), _n, value
            ),
        ),
    )
del _lic_field
# --- END generated ingest-controller-state shims ---

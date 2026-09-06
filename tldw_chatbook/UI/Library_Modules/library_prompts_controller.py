"""Library Prompts canvas controller.

Controller PR of the Prompts extraction series (wave-6 task 2 of
``.superpowers/sdd/2026-09-05-library-decomposition-wave6-prompts``; prompts
series 2/3; recipe: ``backlog/docs/library-decomposition-recipe.md``;
``library_skills_controller.py`` -- the largest prior single-cluster move --
and ``library_ingest_controller.py`` -- the newest -- are the templates this
mirrors in shape). Owns the Prompts cluster: the list canvas's browse/
filter/sort/pager/selection surface, the Import row, the prompt detail load
and its failure/retry states, the editor (basic/advanced/info modes, block
editor, dirty tracking, save/discard/back), the version-history region, the
Prompt Collections membership surface, export/copy/duplicate, and the
delete/undo batch flow. ``LibraryScreen`` keeps one-line delegators under
**100 of these 139** original names; this series' own cleanup task (prompts
series 3/3) pruned the other 39, whose delegators had zero references
outside their own body anywhere in the repo -- see
``Tests/Architecture/test_library_prompts_wiring.py``'s own
``_PROMPTS_CLUSTER_SCREEN_DELEGATOR_PRUNED`` for the list. The 100 that
stay are 51 the recipe's transform whitelist keeps unconditionally (44
``@on`` + 1 ``action_*`` + the 6 ``on_<message>`` handlers Textual
dispatches BY NAME off ``Message.handler_name``, which a delete would
silently unhook) plus 49 with a genuine external caller.

**Cluster derivation.** An ``ast`` census of every ``LibraryScreen``
class-body method whose name contains ``"prompt"`` (case-insensitive), run
fresh at this task's own execution time (the recipe's "never trust a
carried-over count" rule, §6): **161 raw ``FunctionDef`` matches, 161 unique
names** -- no property/setter-pair gap, unlike Skills' own 133/127. Two
completeness checks ran alongside it, both returning nothing new:

- a reverse call-graph scan for any NON-"prompt"-named ``LibraryScreen``
  method whose callers are ALL prompt-named (the "bare-named cluster member"
  shape the conversations exemplar's own ``_conversation_records`` miss made
  the recipe warn about, §11's ``startswith`` lesson generalized to methods):
  **zero**;
- a decorator scan for any non-prompt-named method carrying a
  ``#library-prompt*``/``.library-prompt*`` ``@on`` selector, plus a sweep of
  every ``on_<message>`` naming-convention handler on the class: **zero**
  beyond the six ``on_prompt_block_editor_*`` handlers, which already contain
  "prompt" and are therefore already in the 161.

**Single vs. split controller: single, decided by connected-components
analysis, not by feel.** The plan explicitly leaves a two-controller split on
the table for a cluster this size ("split ONLY on a clean ownership seam;
when unsure, one controller"). Building the ``self.<name>`` reference graph
among all 161 candidates yields **one connected component of 145 names plus
16 isolated singletons** -- no second component of any size, and therefore no
seam to split on. The hypothetical editor/studio-vs-browse seam does not
hold: the editor's own exit path (``_exit_library_prompt_editor_guarded``)
drives the browse refetch, the detail loader
(``_refresh_library_prompt_detail``) is reached from both the row handler and
the conflict resolvers, and the delete/undo flow writes state both surfaces
read. **Decision: ONE combined ``LibraryPromptsController``**, matching the
skills/search+RAG/ingest precedent's identical resolution at comparable
scale.

**22 of the 161 candidates excluded, not moved (139 move):**

1. **2 screen-identity exclusions -- recipe §3's sixth bypass shape, Form C
   (the identity check INLINED rather than routed through
   ``_library_screen_is_current``).** ``_apply_library_prompts_import_status``
   and ``_run_library_prompts_import`` each carry
   ``self.app.screen is not self``. A moved body's ``self`` is the
   CONTROLLER, and ``real_screen is controller`` can never be true, so the
   guarded branch would silently take the "not current" path forever. This is
   the exact shape the skills series' own second draft shipped and had to
   revert after 8 ``Tests/Skills/`` tests failed; no accommodation exists
   (identity cannot be satisfied by a proxy), so both methods stay
   screen-resident, full-bodied, untouched. ``_run_library_prompts_import``
   additionally trips exclusion class 4 below (it reads the bare module
   global ``validate_path_simple``), so it is doubly disqualified.
2. **14 unbound-fake-self test-bypass exclusions** (recipe §3's first
   documented shape). A repo-wide content grep across ALL of ``Tests/`` --
   never a ``-k``-filtered subset, per the seventh shape's own
   filter-blindness rule -- found 19 direct ``LibraryScreen.<name>(fake, …)``
   call sites across 3 files covering 10 names
   (``Tests/UI/test_library_prompts_canvas.py`` ×14,
   ``Tests/UI/test_library_canvas_scoped_sync.py`` ×3,
   ``Tests/UI/test_library_choice_strips.py`` ×2). **FIVE MORE names, across
   three distinct indirection shapes, reach the same bypass in a way grep
   cannot see** -- each shape found by a separate mechanical census this
   controller PR ran precisely because the direct grep is not a completeness
   proof. (An earlier draft of this sentence said "Three MORE names",
   counting the three SHAPES below as if they were names; the bullets have
   always listed five names -- 1 + 2 + 2 -- and 10 + 5 = the 15 rows this
   class's own closing paragraph enumerates.)
   - ``_stage_library_prompt_for_console`` -- captured as a fake-harness
     **CLASS ATTRIBUTE**
     (``class _LibraryPromptHandlerHarness(SimpleNamespace): _stage_library_
     prompt_for_console = LibraryScreen._stage_library_prompt_for_console``,
     ``test_library_prompts_canvas.py:13343``), never called through a
     ``LibraryScreen.<name>(`` expression at all. Found by an AST census of
     NON-CALL attribute references to a cluster name.
   - ``handle_library_prompts_page_previous``/``handle_library_prompts_page_
     next`` -- placed as bare unbound functions in a ``@pytest.mark.
     parametrize`` tuple and invoked later as ``handler(fake, event)``
     (``test_library_prompts_canvas.py:2876-2877``, called at ``:2914``).
     Found by the same non-call-reference census.
   - ``handle_library_prompts_empty_clear_filter``/``handle_library_prompts_
     empty_all_prompts`` -- dispatched by **STRING NAME**,
     ``getattr(LibraryScreen, handler_name)(fake, …)``
     (``test_library_prompts_canvas.py:2044``, names supplied by the
     parametrize table at ``:2016``/``:2022``). Found by a census of every
     cluster method name appearing as a STRING LITERAL anywhere in
     ``Tests/`` -- the "quoted-string form" sweep the wave-5 round-2 string
     -loop incident added to this program's standing checklist, applied here
     to METHOD names rather than field names.

   All 14 stay on ``LibraryScreen``, UNMOVED, full-bodied -- the recipe's
   established accommodation for this shape (leave the method real; a mover
   reaches it, where needed, through a named late-binding dependency that
   re-reads ``screen.<name>`` at call time, which is exactly why every
   ``SimpleNamespace`` fixture above keeps working unmodified after this
   move). The 14: ``_apply_library_prompts_import_status`` (also class 1),
   ``_build_library_prompts_state``, ``_settle_library_prompt_delete``,
   ``_stage_library_prompt_for_console``,
   ``handle_library_prompt_insert_console``, ``handle_library_prompt_row``,
   ``handle_library_prompts_empty_new``,
   ``handle_library_prompts_empty_all_prompts``,
   ``handle_library_prompts_empty_clear_filter``,
   ``handle_library_prompts_filter``,
   ``handle_library_prompts_page_next``,
   ``handle_library_prompts_page_previous``,
   ``handle_library_prompts_sort``, ``handle_library_prompts_sort_choice``,
   ``on_prompt_block_editor_apply_requested`` -- 15 rows because
   ``_apply_library_prompts_import_status`` is counted under class 1, not
   here.
3. **3 instance-attribute-monkeypatch exclusions** (recipe §3's second
   documented bypass shape; the skills series' own
   ``_request_library_skills_browse`` precedent, whose Prompts analogue is
   one of these three). Each is patched on a REAL, ``__init__``-constructed
   ``LibraryScreen`` instance, with a sibling expected to observe the patch
   when it calls ``self.<name>(…)`` internally:
   - ``_flush_library_prompt_save`` --
     ``Tests/UI/test_screen_navigation.py:3239``/``:3251`` (bare attribute
     assignment) plus ``monkeypatch.setattr(screen, …)`` at
     ``test_screen_navigation.py:2021``,
     ``test_library_prompts_canvas.py:4038``/``:6777``.
   - ``_request_library_prompts_browse`` -- 7 ``monkeypatch.setattr(screen,
     …)``/assignment sites across ``test_screen_navigation.py``,
     ``test_library_prompts_canvas.py`` and ``test_library_shell.py``,
     several of them ``Mock(wraps=screen._request_library_prompts_browse)``
     recorders that must see the ORIGINAL bound method.
   - ``_reset_library_prompt_editor_state`` --
     ``test_screen_navigation.py:3252``.
   All three are called by movers (``_exit_library_prompt_editor_guarded``
   calls all three), which reach them through named late-binding dependencies
   below -- so the patches keep working after this move.
4. **2 module-globals-coupling exclusions** (recipe §3's oldest documented
   shape, restated as the eighth numbered shape's mechanical census). The
   census ran to completion: every one of the 82 bare module-global names the
   161 candidate bodies read, crossed against every ``library_screen``-scoped
   patch shape (direct-attribute, fully-qualified string, and the two-argument
   ``monkeypatch.setattr``/``patch.object`` form) across ALL of ``Tests/``,
   under every alias tests actually import the module as (``library_screen``,
   ``library_screen_module``, ``screen_module``, ``library_module`` -- the
   alias list was DERIVED, not assumed, per wave-5 task 3's own lesson).
   Seven names had hits; each hitting test was read:
   - ``validate_path_simple`` -> **ACTIVE**.
     ``test_library_prompts_canvas.py::test_library_prompt_write_export_file_
     rejects_invalid_path`` (``:10976``) patches
     ``library_screen_module.validate_path_simple`` with a stub that always
     raises, then calls ``screen._write_library_prompt_export_file(...)``
     with a perfectly VALID tmp path and asserts nothing was written. Moved,
     the controller's own import would win, the real validator would accept
     the path, the file WOULD be written and the assertion would fail --
     loudly, not vacuously. ``_write_library_prompt_export_file`` excluded;
     its one mover caller (``_export_library_prompt``) reaches it through a
     named dependency. (``_run_library_prompts_import`` reads the same name
     and is already excluded under class 1.)
   - ``save_setting_to_cli_config`` -> **ACTIVE**.
     ``test_library_prompt_mode_persistence_failure_keeps_live_mode_and_
     warns`` (``:689``) patches it to return ``False`` and awaits
     ``screen._persist_library_prompt_editor_mode("advanced")`` directly,
     asserting the warning notice; a second test (``:5585``) patches it with
     a thread-recording stub and clicks the real
     ``#library-prompt-mode-advanced`` button.
     ``_persist_library_prompt_editor_mode`` excluded; its one mover caller
     (``handle_library_prompt_editor_mode``) reaches it through a named
     dependency.
   - ``_sync_library_canvas`` -> **LATENT, kept** (**33 sites across 10
     files**; "site" per the ingest controller's own definition -- one match,
     one line, of the 3-shape pattern set, DEDUPLICATED by line number within
     a file, since a single line can match two shapes at once. Summing the
     per-shape match counts instead, without that dedup, gives 42: S1
     direct-attribute 20, S2 fully-qualified-string 9, S3 two-argument
     setattr/patch.object 13). Every site was read, and the LATENT verdict
     rests on ENCLOSING-TEST scope, not on filenames:
     - ``test_library_canvas_scoped_sync.py`` (3 sites) exercises
       ``handle_library_prompt_row`` and ``_apply_library_prompts_import_
       status`` -- both EXCLUDED, never moved, so their own bare
       ``_sync_library_canvas`` call still resolves through
       ``library_screen.py``'s globals regardless.
     - ``test_library_entry_compose_once.py`` (8 sites) **does invoke a
       MOVER** -- ``_sync_library_prompts_browse_result`` at ``:1014`` and
       ``:1044``. It is still LATENT, and this is the precise argument:
       ``monkeypatch`` is FUNCTION-scoped, and the file's four
       ``_sync_library_canvas`` patch pairs live in four other test
       functions (``test_source_worker_completion_during_resume_dispatch_
       reconciles_once``, ``test_snapshot_timeout_is_repaired_by_blocked_
       fresh_success``, ``test_queued_reconcile_supersedes_after_route_
       switch``, ``test_detached_queued_reconcile_completion_is_a_noop``)
       while the two mover invocations live in ``test_stale_prompt_token_
       cannot_project_after_route_switch``/``..._is_rejected_on_the_same_
       route``. Zero overlap, verified by mapping every census line and both
       invocation lines to their enclosing ``FunctionDef``: no patch is in
       effect while the mover runs. **A first draft of this entry claimed
       "only ONE test function mentions any mover name" -- false, and it
       would have hidden this file behind a wrong reason had the verdict
       gone the other way.**
     - The remaining eight files patch it for notes/media/skills canvas
       syncs and never touch a Prompts name at all.

     Zero of the 33 reach any of the SEVEN MOVERS that forward bare ``self``
     into this dispatcher. This is the same systemic bare-function shape
     every sibling controller already carries, and the same verdict the
     ingest series recorded for it.
   - ``LIBRARY_ROW_BROWSE_PROMPTS`` (2 sites),
     ``_LIBRARY_PROMPTS_IMPORT_WORKER_GROUP`` (1),
     ``resolve_adaptive_reader_layout`` (1) -> **LATENT**: all three are
     plain module-attribute READS (a test computing an expected value or
     naming a worker group), never a patch.
   - ``asyncio`` (1 site) -> **LATENT**:
     ``monkeypatch.setattr(library_screen_module.asyncio, "to_thread", …)``
     patches an attribute of the SHARED ``asyncio`` module object, which is
     the same object in every importer -- not a rebinding of a name in
     ``library_screen``'s own globals, so a move cannot bypass it.
5. **1 merely-delegate-to-existing-controller ``@property``** (the skills
   series' own named exclusion class, six of them there):
   ``_library_prompt_history_state``, whose ENTIRE body is
   ``return self._library_prompt_history_controller.state``. It stays
   screen-resident (already-extracted wiring), and this controller reaches it
   read-only below. Tests read it 64 times directly off the screen.

**139 of the 161 candidates move onto this controller** (44 ``@on`` handlers
+ 6 ``on_<message>`` naming-convention handlers + 1 ``action_*`` + 1
``@staticmethod`` + 87 plain).

**Byte-for-byte canon** (moved bodies never edited -- every name they
reference that is not this controller's own state is rebound under the SAME
name, per the two binding kinds; see ``LibraryIngestController.__init__`` and
``ConsoleDictationController.__init__`` for the sibling worked examples). The
binding surface below was derived MECHANICALLY, by walking all 139 moved
bodies for every ``self.<attr>`` load/store AND every
``getattr(self, "<literal>")`` call, then subtracting this controller's own
state fields and the movers themselves -- 42 names, every one of them pinned
in ``test_prompts_controller_binds_every_name_its_moved_bodies_use``:

1. **Framework services** (``app``, ``app_instance``, ``call_after_refresh``,
   ``focused``, ``is_mounted``, ``is_running``, ``query``, ``query_one``,
   ``refresh``, ``run_worker``, ``set_timer``, ``workers``) are live-read
   from the screen via ``@property`` on every access -- never snapshotted.
   Three of these are worth naming individually:
   - ``focused`` is here because four movers
     (``_sync_library_prompt_selection``, ``_library_prompts_focus_identity``,
     ``_capture_library_prompts_filter_cursor``,
     ``_sync_library_prompts_browse_result``) call
     ``getattr(self, "focused", None)``. That expression is invisible to a
     plain ``self.<attr>`` census and returns its DEFAULT silently when the
     name is unbound -- the exact unbound-attribute escape the skills series
     shipped and only caught in post-landing review (recipe §3). It is
     bound here from the start, and pinned by a wiring test rather than left
     to a reviewer.
   - ``is_running`` and ``app`` are reached through the shared
     ``_sync_library_canvas(self, "prompts", …)`` dispatcher, not by any body
     directly -- the same reason the skills/RAG/ingest controllers bind them.
     ELEVEN prompt-cluster methods forward bare ``self`` into that
     dispatcher, but only **7 of them are MOVERS**; the other 4
     (``_apply_library_prompts_import_status``, ``handle_library_prompt_
     row``, ``handle_library_prompts_sort``, ``handle_library_prompts_sort_
     choice``) are exclusions whose bodies never left ``library_screen.py``.
     Only the 7 movers' forwarding is this controller's problem.
   - ``workers`` is read once, by ``_library_prompt_write_worker_is_active``,
     inside its own ``try``/``except``: off the app tree the screen's
     ``workers`` raises, and the body already handles that. The forward
     preserves the raise rather than hiding it.
2. **Everything else** the cluster depends on that is not its own state is a
   NAMED constructor dependency: (a) 12 general Library-wide shell helpers a
   moved body calls with explicit arguments
   (``_arm_library_list_entry_focus``, ``_focus_library_control``,
   ``_library_entry_reconcile_is_current``, ``_library_entry_route_key``,
   ``_library_list_canvas_showing_list``,
   ``_library_note_keywords_from_input``, ``_open_library_export_canvas``,
   ``_refresh_local_source_snapshot`` -- one of recipe §3's four PERMANENTLY
   screen-routed monkeypatch names, reached here the same accessor-callable
   way every other subsystem already reaches it, never moved itself --,
   ``_run_library_service_call``, ``_safe_text``, ``_sanitize_media_field``,
   ``_sanitize_note_content``); (b) 4 shared shell state accessors this
   cluster READS and never writes (``_library_pending_list_entry_focus``,
   ``_library_selected_row_id``, ``_library_snapshot_state_generation``,
   ``_local_source_counts`` -- the last is a live ``dict`` two movers mutate
   IN PLACE through its getter, the ingest series'
   ``_library_ingest_analyze_outcomes`` precedent, so no setter is needed);
   (c) 3 wiring accessors for the prior-extracted prompt controller instances
   task 1 deliberately kept off ``LibraryPromptsState``
   (``_library_prompt_browse_controller`` -- 23 movers, the cluster's most
   referenced single name --, ``_library_prompt_collections_controller``,
   ``_library_prompt_history_controller``); (d) 1 read accessor for the
   merely-delegate ``@property`` exclusion (``_library_prompt_history_
   state``); (e) 10 named late-binding callables for the exclusions above
   that a MOVER still calls internally
   (``_apply_library_prompts_import_status``,
   ``_build_library_prompts_state``, ``_flush_library_prompt_save``,
   ``_persist_library_prompt_editor_mode``,
   ``_request_library_prompts_browse``,
   ``_reset_library_prompt_editor_state``, ``_run_library_prompts_import``,
   ``_settle_library_prompt_delete``, ``_stage_library_prompt_for_console``,
   ``_write_library_prompt_export_file``) -- each a ``lambda`` that re-reads
   ``screen.<name>`` on every invocation, at CALL time, not a value captured
   once at construction, which is exactly why every bypass fixture named
   above keeps working unmodified after this move.

**No class-level constants move.** ``LibraryScreen``'s own
``_PROMPTS_WORKBENCH_FOCUS_TARGETS`` is read only by a shell method
(``library_screen.py:9472``), never by a mover, so unlike the ingest series'
three dead-zone constants there is nothing to relocate or delete here.

**Construction order and import shape.** ``LibraryScreen.__init__`` builds
``self._prompts_controller`` right after ``self._ingest_controller``,
matching every other controller in that file, and the controller class is
imported **function-locally**, inside ``__init__``'s established lazy-import
block -- never at module level. That is a hard constraint of this wave, not a
style preference: ``library_screen.py`` is deliberately absent from the
``_ui_ready`` boot-budget snapshot
(``Tests/Performance/boot_budget_snapshots/ui_ready_modules.txt``), and both
``Tests/Packaging/test_library_preimport_closure.py`` and
``Tests/Performance/test_ui_ready_module_census.py`` enforce it.

This subsystem's OWN state (every flat prompt field name the moved bodies
reference -- all 43, across THREE prefix families, resolved by task 1's
single-source ``prompt_state_shim_attr()``) is exposed through a generated
property loop reading ``self._prompts_state_accessor().<field>`` -- the same
generator shape task 1 installed on ``LibraryScreen`` (deleted at cleanup,
task 3, once this controller's own copy makes the screen's copy dead),
mirroring the skills controller's three-prefix precedent exactly.
"""
from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING

from loguru import logger
from rich.markup import escape as escape_markup
from textual import on
from textual.css.query import NoMatches, QueryError
from textual.widget import Widget
from textual.widgets import Button, Checkbox, Input, Static, TextArea
from textual.worker import Worker

from ...DB.Prompts_DB import ConflictError as PromptConflictError
from ...Library.library_export_scope import ExportScope
from ...Library.library_prompts_state import (
    DEFAULT_PROMPT_BROWSE_PAGE_SIZE,
    PromptBrowseResult,
    PromptBrowseScope,
    PromptEditorState,
    PromptHistoryRestoreRequest,
    PromptHistoryState,
    PromptSelectionBasket,
    PromptSelectionEntry,
    build_prompt_editor_state,
    classify_prompt_save_error,
    prepare_prompt_artifact_save,
    prompt_basic_unavailable_reason,
    prompt_editor_meta_line,
    require_artifact_save_supported,
)
from ...Library.library_shell_state import (
    LIBRARY_ROW_BROWSE_PROMPTS,
    LIBRARY_ROW_CREATE_PROMPT,
    library_choice_label,
)
from ...Prompt_Management.prompt_artifact_codec import deserialize_definition
from ...Prompt_Management.prompt_artifact_models import ArtifactType
from ...Prompt_Management.prompt_batch_models import (
    PromptBatchDeleteResult,
    PromptBatchRestoreResult,
    PromptBatchTarget,
)
from ...Prompt_Management.prompt_markdown_export import render_prompt_markdown
from ...Third_Party.textual_fspicker import FileOpen, FileSave
from ...Utils.adaptive_reader_state import resolve_adaptive_reader_layout
from ...Widgets.Library import (
    PROMPT_DISCARD_TOOLTIP_BUSY,
    PROMPT_DISCARD_TOOLTIP_CLEAN,
    PROMPT_DISCARD_TOOLTIP_DIRTY,
    LibraryAdaptiveReaderShell,
    LibraryPromptWorkPane,
    LibraryPromptsListCanvas,
)
from ...Widgets.Library.prompt_delete_confirmation_modal import (
    PromptDeleteConfirmationModal,
    PromptDeleteItem,
    PromptDeleteRequest,
)
from ...Widgets.Prompts.prompt_block_editor import PromptBlockEditor
from ...Widgets.Prompts.prompt_block_editor_state import (
    PromptBlockEditorState,
    set_artifact_type,
)
from ...Widgets.confirmation_dialog import ConfirmationDialog
from .canvas_sync import _sync_library_canvas
from .library_prompts_state import LibraryPromptsState, prompt_state_shim_attr
from .prompt_history_region import LibraryPromptHistoryRegion
from .screen_constants import (
    _LIBRARY_PROMPT_WRITE_IN_PROGRESS_COPY,
    _LIBRARY_PROMPT_WRITE_WORKER_GROUPS,
    _LIBRARY_PROMPTS_IMPORT_WORKER_GROUP,
    _LIBRARY_PROMPTS_SEARCH_DEBOUNCE_SECONDS,
    LIBRARY_PROMPT_DIRTY_VETO_COPY,
    LIBRARY_PROMPT_SAVE_STATUS_COPY,
    LIBRARY_PROMPT_TEXT_MAX_CHARS,
    LIBRARY_PROMPTS_READER_PROFILE,
)
from .screen_support_types import LibraryEntryReconcileResult

if TYPE_CHECKING:
    from ..Screens.library_screen import LibraryScreen


class LibraryPromptsController:
    """Owns the Library Prompts cluster (139 methods).

    Holds no state of its own beyond what it reads and writes through
    ``LibraryPromptsState`` (via the injected accessor) and the shared
    shell/framework/wiring bindings below. ``LibraryScreen`` constructs
    exactly one of these, in ``__init__`` right after
    ``self._ingest_controller``, and keeps one-line delegators for 100 of
    the 139 original names this cluster moved (the prompts cleanup PR pruned
    the other 39 -- see the module docstring for the full derivation, the 22
    exclusions, and ``_PROMPTS_CLUSTER_SCREEN_DELEGATOR_PRUNED`` for the
    pruned list).
    """

    def __init__(
        self,
        screen: "LibraryScreen",
        *,
        prompts_state_accessor,
        # -- general Library-wide shell helpers, not moved (group (a)).
        arm_library_list_entry_focus,
        focus_library_control,
        library_entry_reconcile_is_current,
        library_entry_route_key,
        library_list_canvas_showing_list,
        library_note_keywords_from_input,
        open_library_export_canvas,
        refresh_local_source_snapshot,
        run_library_service_call,
        safe_text,
        sanitize_media_field,
        sanitize_note_content,
        # -- shared shell state this cluster READS (group (b)). All four are
        # getter-only: the mechanical binding census found ZERO stores to any
        # non-own-state name across all 139 moved bodies. `_local_source_
        # counts` is a live dict two movers mutate IN PLACE through the
        # getter (`_delete_library_prompts`, `_undo_library_prompt_delete`) --
        # the ingest controller's `_library_ingest_analyze_outcomes`
        # precedent, and the reason no setter is needed for it either.
        library_pending_list_entry_focus_accessor,
        library_selected_row_id_accessor,
        library_snapshot_state_generation_accessor,
        local_source_counts_accessor,
        # -- the 3 prior-extracted prompt WIRING controller instances task 1
        # deliberately kept off `LibraryPromptsState` (group (c)).
        library_prompt_browse_controller_accessor,
        library_prompt_collections_controller_accessor,
        library_prompt_history_controller_accessor,
        # -- the one merely-delegate-to-existing-controller `@property`
        # exclusion, read-only (group (d)).
        library_prompt_history_state_accessor,
        # -- named late-binding callables for the test-bypass/hazard
        # exclusions (group (e)) that a MOVER still calls internally.
        apply_library_prompts_import_status,
        build_library_prompts_state,
        flush_library_prompt_save,
        persist_library_prompt_editor_mode,
        request_library_prompts_browse,
        reset_library_prompt_editor_state,
        run_library_prompts_import,
        settle_library_prompt_delete,
        stage_library_prompt_for_console,
        write_library_prompt_export_file,
    ) -> None:
        """Build the controller and bind everything its moved bodies need.

        Every one of the 139 method bodies below is a byte-for-byte copy of
        the pre-extraction ``LibraryScreen`` method: no internal line was
        edited to retarget a call or an attribute. That is possible because
        this constructor binds every name those bodies reference that is not
        this controller's own state, under the SAME name the original method
        used. See the module docstring for the binding kinds this follows and
        the full per-parameter derivation.

        Args:
            screen: The Library screen. Used ONLY for the twelve framework
                services below (``app``, ``app_instance``,
                ``call_after_refresh``, ``focused``, ``is_mounted``,
                ``is_running``, ``query``, ``query_one``, ``refresh``,
                ``run_worker``, ``set_timer``, ``workers``) -- this cluster
                owns no DOM of its own.
            prompts_state_accessor: Returns the live ``LibraryPromptsState``
                (``LibraryScreen._prompts_state``, task 1). Backs every
                generated flat prompt-field property below.
        """
        self._screen = screen
        self._prompts_state_accessor = prompts_state_accessor
        self._arm_library_list_entry_focus_fn = arm_library_list_entry_focus
        self._focus_library_control_fn = focus_library_control
        self._library_entry_reconcile_is_current_fn = (
            library_entry_reconcile_is_current
        )
        self._library_entry_route_key_fn = library_entry_route_key
        self._library_list_canvas_showing_list_fn = library_list_canvas_showing_list
        self._library_note_keywords_from_input_fn = library_note_keywords_from_input
        self._open_library_export_canvas_fn = open_library_export_canvas
        self._refresh_local_source_snapshot_fn = refresh_local_source_snapshot
        self._run_library_service_call_fn = run_library_service_call
        self._safe_text_fn = safe_text
        self._sanitize_media_field_fn = sanitize_media_field
        self._sanitize_note_content_fn = sanitize_note_content
        self._library_pending_list_entry_focus_accessor = (
            library_pending_list_entry_focus_accessor
        )
        self._library_selected_row_id_accessor = library_selected_row_id_accessor
        self._library_snapshot_state_generation_accessor = (
            library_snapshot_state_generation_accessor
        )
        self._local_source_counts_accessor = local_source_counts_accessor
        self._library_prompt_browse_controller_accessor = (
            library_prompt_browse_controller_accessor
        )
        self._library_prompt_collections_controller_accessor = (
            library_prompt_collections_controller_accessor
        )
        self._library_prompt_history_controller_accessor = (
            library_prompt_history_controller_accessor
        )
        self._library_prompt_history_state_accessor = (
            library_prompt_history_state_accessor
        )
        self._apply_library_prompts_import_status_fn = (
            apply_library_prompts_import_status
        )
        self._build_library_prompts_state_fn = build_library_prompts_state
        self._flush_library_prompt_save_fn = flush_library_prompt_save
        self._persist_library_prompt_editor_mode_fn = (
            persist_library_prompt_editor_mode
        )
        self._request_library_prompts_browse_fn = request_library_prompts_browse
        self._reset_library_prompt_editor_state_fn = (
            reset_library_prompt_editor_state
        )
        self._run_library_prompts_import_fn = run_library_prompts_import
        self._settle_library_prompt_delete_fn = settle_library_prompt_delete
        self._stage_library_prompt_for_console_fn = stage_library_prompt_for_console
        self._write_library_prompt_export_file_fn = write_library_prompt_export_file

    # -- framework services: live-read properties, never snapshotted -------

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
        """Live-forward the screen's currently focused widget.

        Bound explicitly because four moved bodies reach it as
        ``getattr(self, "focused", None)`` -- an expression the recipe's own
        ``self.<attr>`` census cannot see, and one that returns its DEFAULT
        forever (no exception, no red test) when the name is unbound. The
        skills series shipped exactly that regression; see the module
        docstring's framework-services note.
        """
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

    @property
    def set_timer(self) -> Any:
        return self._screen.set_timer

    @property
    def workers(self) -> Any:
        """Live-forward the screen's worker manager.

        Deliberately NOT exception-guarded: off the app tree the screen's own
        ``workers`` raises, and this cluster's single reader
        (``_library_prompt_write_worker_is_active``) already wraps its access
        in ``try``/``except`` for exactly that case. Swallowing the raise here
        would change that body's behaviour.
        """
        return self._screen.workers

    # -- shared shell state, read-only (group (b)) -------------------------

    @property
    def _library_pending_list_entry_focus(self) -> bool:
        return self._library_pending_list_entry_focus_accessor()

    @property
    def _library_selected_row_id(self) -> str:
        return self._library_selected_row_id_accessor()

    @property
    def _library_snapshot_state_generation(self) -> int:
        return self._library_snapshot_state_generation_accessor()

    @property
    def _local_source_counts(self) -> dict[str, int]:
        """Calls the injected ``local_source_counts_accessor``.

        Getter only, deliberately: the two moved bodies that touch this map
        (``_delete_library_prompts``, ``_undo_library_prompt_delete``) assign
        into it by KEY, and a ``dict`` mutates in place through the getter --
        so no setter is needed and the moved lines stay byte-for-byte.
        """
        return self._local_source_counts_accessor()

    # -- prior-extracted prompt wiring controllers (group (c)) -------------

    @property
    def _library_prompt_browse_controller(self) -> Any:
        return self._library_prompt_browse_controller_accessor()

    @property
    def _library_prompt_collections_controller(self) -> Any:
        return self._library_prompt_collections_controller_accessor()

    @property
    def _library_prompt_history_controller(self) -> Any:
        return self._library_prompt_history_controller_accessor()

    # -- the merely-delegate-to-existing-controller property (group (d)) ---

    @property
    def _library_prompt_history_state(self) -> PromptHistoryState | None:
        """Calls the injected ``library_prompt_history_state_accessor``.

        Read-only, matching the screen-resident ``@property`` it forwards to
        (whose own entire body is
        ``self._library_prompt_history_controller.state``).
        """
        return self._library_prompt_history_state_accessor()

    # -- general Library-wide shell helpers (group (a)) --------------------

    @property
    def _arm_library_list_entry_focus(self) -> Any:
        return self._arm_library_list_entry_focus_fn

    @property
    def _focus_library_control(self) -> Any:
        return self._focus_library_control_fn

    @property
    def _library_entry_reconcile_is_current(self) -> Any:
        return self._library_entry_reconcile_is_current_fn

    @property
    def _library_entry_route_key(self) -> Any:
        return self._library_entry_route_key_fn

    @property
    def _library_list_canvas_showing_list(self) -> Any:
        return self._library_list_canvas_showing_list_fn

    @property
    def _library_note_keywords_from_input(self) -> Any:
        return self._library_note_keywords_from_input_fn

    @property
    def _open_library_export_canvas(self) -> Any:
        return self._open_library_export_canvas_fn

    @property
    def _refresh_local_source_snapshot(self) -> Any:
        return self._refresh_local_source_snapshot_fn

    @property
    def _run_library_service_call(self) -> Any:
        return self._run_library_service_call_fn

    @property
    def _safe_text(self) -> Any:
        return self._safe_text_fn

    @property
    def _sanitize_media_field(self) -> Any:
        return self._sanitize_media_field_fn

    @property
    def _sanitize_note_content(self) -> Any:
        return self._sanitize_note_content_fn

    # -- named late-binding callables for the exclusions (group (e)) -------

    @property
    def _apply_library_prompts_import_status(self) -> Any:
        return self._apply_library_prompts_import_status_fn

    @property
    def _build_library_prompts_state(self) -> Any:
        return self._build_library_prompts_state_fn

    @property
    def _flush_library_prompt_save(self) -> Any:
        return self._flush_library_prompt_save_fn

    @property
    def _persist_library_prompt_editor_mode(self) -> Any:
        return self._persist_library_prompt_editor_mode_fn

    @property
    def _request_library_prompts_browse(self) -> Any:
        return self._request_library_prompts_browse_fn

    @property
    def _reset_library_prompt_editor_state(self) -> Any:
        return self._reset_library_prompt_editor_state_fn

    @property
    def _run_library_prompts_import(self) -> Any:
        return self._run_library_prompts_import_fn

    @property
    def _settle_library_prompt_delete(self) -> Any:
        return self._settle_library_prompt_delete_fn

    @property
    def _stage_library_prompt_for_console(self) -> Any:
        return self._stage_library_prompt_for_console_fn

    @property
    def _write_library_prompt_export_file(self) -> Any:
        return self._write_library_prompt_export_file_fn

    # -- moved cluster methods (139), byte-for-byte, original file order ---
    def _sync_library_prompts_reader_layout_from_shell(
        self,
        priority: Literal["library", "items"] | None = None,
    ) -> None:
        """Resolve the settled Prompts shell and patch it in place."""
        try:
            shell = self.query_one(
                "#library-prompts-reader-shell", LibraryAdaptiveReaderShell
            )
        except (NoMatches, QueryError):
            return
        width = shell.region.width
        if width <= 0:
            return
        previous = self._library_prompts_reader_layout
        if (
            previous.reader_width == 0
            and previous.library_width == 0
            and previous.items_width == 0
        ):
            previous = None
        if priority is None and self._library_prompts_view == "list":
            priority = "items"
        elif (
            priority is None
            and previous is not None
            and previous.priority_pane == "items"
        ):
            previous = dataclasses.replace(previous, priority_pane=None)
        layout = resolve_adaptive_reader_layout(
            width,
            self._library_prompts_reader_preferences,
            LIBRARY_PROMPTS_READER_PROFILE,
            previous=previous,
            priority=priority,
        )
        shell.sync_layout(layout)
        self._library_prompts_reader_layout = layout

    def _mirror_library_prompts_reader_preference(
        self,
        key: Literal["library_open", "items_open"],
        value: bool,
    ) -> None:
        """Mirror one optimistic Prompts pane choice into app config."""
        app_config = getattr(self.app_instance, "app_config", None)
        if not isinstance(app_config, dict):
            return
        library_config = app_config.setdefault("library", {})
        if not isinstance(library_config, dict):
            library_config = {}
            app_config["library"] = library_config
        section_name = "reader" if key == "library_open" else "prompts_reader"
        section = library_config.setdefault(section_name, {})
        if not isinstance(section, dict):
            section = {}
            library_config[section_name] = section
        section[key] = value

    @staticmethod
    def _restore_library_prompts_scope(state: Mapping[str, Any]) -> PromptBrowseScope:
        """Return a dispatch-safe applied Prompt scope from screen state."""
        saved = state.get("library_prompts_scope")
        if isinstance(saved, PromptBrowseScope):
            raw = dataclasses.asdict(saved)
        elif type(saved) is dict:
            raw = saved
        else:
            legacy_sort = state.get("library_prompts_sort")
            legacy_query = state.get("library_prompts_filter")
            raw = {
                "query": legacy_query if type(legacy_query) is str else "",
                "sort_by": "name" if legacy_sort == "name" else "last_modified",
                "sort_order": "asc" if legacy_sort == "name" else "desc",
            }

        query = raw.get("query", "")
        if type(query) is not str:
            query = ""
        page = raw.get("page", 1)
        max_page = (2**63 - 1) // DEFAULT_PROMPT_BROWSE_PAGE_SIZE + 1
        if type(page) is not int or not 1 <= page <= max_page:
            page = 1
        try:
            return PromptBrowseScope(
                query=query,
                collection_id=raw.get("collection_id"),
                sort_by=raw.get("sort_by", "last_modified"),
                sort_order=raw.get("sort_order", "desc"),
                page=page,
                page_size=DEFAULT_PROMPT_BROWSE_PAGE_SIZE,
            )
        except (TypeError, ValueError):
            return PromptBrowseScope()

    def _library_prompt_editor_active(self) -> bool:
        """True while the in-canvas prompt editor is the live view (task-2856).

        Mirrors ``_library_skill_editor_active``: the Create flow keeps
        ``_library_selected_row_id == LIBRARY_ROW_CREATE_PROMPT`` while its
        editor is open (never reassigned to ``LIBRARY_ROW_BROWSE_PROMPTS``
        -- see ``_enter_library_prompt_create_editor``), so both row ids
        are checked here.
        """
        return (
            self._library_selected_row_id
            in (LIBRARY_ROW_BROWSE_PROMPTS, LIBRARY_ROW_CREATE_PROMPT)
            and getattr(self, "_library_prompts_view", "list") == "editor"
        )

    async def _prompts_count_or_none(
        self, count_prompts: Any, **kwargs: Any
    ) -> int | None:
        """Fetch the exact local prompts total, degrading quietly on failure.

        Runs inside the same ``asyncio.gather`` as the notes/media/
        conversations fetch (see ``_list_local_source_snapshot``). Mirrors
        ``_notes_true_count_or_none``: the count seam is optional (guarded
        by ``callable(count_prompts)`` at the call site), so when it is
        missing this method is never invoked, and when it *is* present but
        raises, the failure is swallowed and ``None`` is returned. A failed
        count has no fallback of its own, so it simply renders the Prompts
        rail row with no badge rather than failing the exact browse result.

        Args:
            count_prompts: The bound ``count_prompts`` callable to invoke.
            **kwargs: Forwarded to ``count_prompts`` (``mode``).

        Returns:
            The exact prompts count, or ``None`` if the call failed or
            returned something other than an ``int``.
        """
        try:
            result = await self._run_library_service_call(
                count_prompts, isolate_in_worker=True, **kwargs
            )
        except Exception:
            logger.opt(exception=True).warning(
                "Failed to fetch local prompts count; Prompts row will show no count."
            )
            return None
        return result if isinstance(result, int) else None

    def _library_prompt_basic_unavailable_reason(
        self,
        state: PromptEditorState,
        *,
        conflict: bool = False,
    ) -> str:
        """Return the forced-Advanced reason without changing the preference."""
        return prompt_basic_unavailable_reason(
            state,
            conflict=conflict,
            can_update_original=(
                state.prompt_id is None or self._library_prompt_can_update_original()
            ),
        )

    def _library_prompts_list_canvas_kwargs(self) -> dict[str, Any]:
        """Return list-only inputs for the retained Prompts Items pane."""
        controller = self._library_prompt_browse_controller
        requested_scope = controller.scope
        display_scope = controller.visible_result.scope
        return {
            "state": self._build_library_prompts_state(),
            "sort_mode": "name" if display_scope.sort_by == "name" else "newest",
            "filter_value": requested_scope.query,
            "browse_result": controller.visible_result,
            "pager": controller.pager,
            "mode": "list",
            "editor_state": None,
            "editor_mode": "basic",
            "basic_unavailable_reason": "",
            "conflict": False,
            "status": "",
            "show_open_existing": False,
            "import_open": False,
            "import_path": "",
            "import_status": "",
            "dirty": False,
            "can_update_original": False,
            "include_starter_content": False,
            "history_state": None,
            "history_current_compatible": True,
            "collection_label": self._library_prompt_collections_controller.collection_label(
                display_scope.collection_id
            ),
            "membership_state": None,
            "sort_choices_visible": self._library_prompts_sort_choices_visible,
            "delete_receipt": self._library_prompt_delete_receipt,
            "page_actions_disabled": controller.freshness == "stale",
            "mutation_in_flight": self._library_prompts_mutation_in_flight,
            "mutation_status": self._library_prompt_mutation_status,
            "write_in_flight": False,
            "bulk_read_only": False,
            "bulk_included": None,
            "identity_mismatch": False,
            "detail_notice": "",
            "detail_retryable": False,
        }

    def _library_prompt_work_pane_kwargs(self) -> dict[str, Any]:
        """Return active non-list content for the retained Prompt work pane."""
        values = self._library_prompts_canvas_kwargs()
        if self._library_prompts_import_open:
            values["mode"] = "list"
        values.update(
            state=None,
            browse_result=None,
            pager=None,
            sort_choices_visible=False,
            delete_receipt=None,
            page_actions_disabled=False,
            mutation_status="",
            bulk_read_only=(
                self._library_prompt_select_mode
                and self._library_prompts_view == "editor"
                and self._library_prompt_detail is not None
            ),
            bulk_included=(
                any(
                    entry.local_id == self._selected_prompt_id
                    for entry in self._library_prompt_selection.entries
                )
                if self._selected_prompt_id is not None
                else None
            ),
        )
        return values

    def _library_prompts_canvas_kwargs(self) -> dict[str, Any]:
        """Return every compose input for the mounted Prompts canvas."""
        membership_state = self._library_prompt_collections_controller.membership_state
        values: dict[str, Any] = {
            "state": None,
            "sort_mode": "newest",
            "filter_value": "",
            "browse_result": None,
            "pager": None,
            "mode": "list",
            "editor_state": None,
            "editor_mode": self._library_prompt_editor_mode,
            "basic_unavailable_reason": "",
            "conflict": False,
            "status": "",
            "show_open_existing": False,
            "import_open": self._library_prompts_import_open,
            "import_path": self._library_prompts_import_path,
            "import_status": self._library_prompts_import_status,
            "dirty": self._library_prompt_dirty,
            "can_update_original": False,
            "include_starter_content": (self._library_prompt_include_starter_content),
            "history_state": self._library_prompt_history_state,
            "history_current_compatible": self._library_prompt_block_state is not None,
            "collection_label": "All prompts",
            "membership_state": membership_state,
            "sort_choices_visible": False,
            "page_actions_disabled": False,
            "mutation_in_flight": self._library_prompts_mutation_in_flight,
            "write_in_flight": self._library_prompt_write_worker_is_active(),
            "bulk_read_only": False,
            "bulk_included": None,
            "identity_mismatch": False,
            "detail_notice": "",
            "detail_retryable": False,
        }
        if self._library_prompts_view == "editor":
            values["mode"] = "editor"
            if self._library_prompt_conflict_snapshot is not None:
                values["editor_state"] = self._current_library_prompt_editor_state(
                    self._library_prompt_conflict_snapshot
                )
                values["conflict"] = True
            elif self._library_prompt_detail is None:
                values["mode"] = "loading"
                values["detail_notice"] = (
                    self._library_prompt_detail_failure_notice()
                    if self._library_prompt_detail_error
                    else self._library_prompt_loading_notice()
                )
                values["detail_retryable"] = self._library_prompt_detail_retryable
            else:
                values["editor_state"] = self._current_library_prompt_editor_state()
                values["status"] = self._library_prompt_status
                values["show_open_existing"] = (
                    self._library_prompt_status
                    == LIBRARY_PROMPT_SAVE_STATUS_COPY["name-in-use"]
                )
                values["can_update_original"] = (
                    self._library_prompt_can_update_original()
                )
                values["identity_mismatch"] = (
                    self._selected_prompt_id != self._library_prompt_loaded_id
                )
                if values["identity_mismatch"]:
                    values["detail_notice"] = (
                        self._library_prompt_detail_failure_notice()
                        if self._library_prompt_detail_error
                        else self._library_prompt_loading_notice()
                    )
                    values["detail_retryable"] = self._library_prompt_detail_retryable
            if values["editor_state"] is not None:
                values["basic_unavailable_reason"] = (
                    self._library_prompt_basic_unavailable_reason(
                        values["editor_state"],
                        conflict=values["conflict"],
                    )
                )
            return values

        controller = self._library_prompt_browse_controller
        requested_scope = controller.scope
        display_scope = controller.visible_result.scope
        values.update(
            {
                "state": self._build_library_prompts_state(),
                "sort_mode": ("name" if display_scope.sort_by == "name" else "newest"),
                "filter_value": requested_scope.query,
                "browse_result": controller.visible_result,
                "pager": controller.pager,
                "page_actions_disabled": controller.freshness == "stale",
                "import_open": self._library_prompts_import_open,
                "import_path": self._library_prompts_import_path,
                "import_status": self._library_prompts_import_status,
                "collection_label": self._library_prompt_collections_controller.collection_label(
                    display_scope.collection_id
                ),
                "sort_choices_visible": self._library_prompts_sort_choices_visible,
            }
        )
        return values

    def _sync_library_prompt_selection(self, focus_identity: str | None) -> None:
        """Recompose only the Prompt canvas from the screen-owned basket."""
        if not self.query("#library-prompts-canvas"):
            return
        focused = getattr(self, "focused", None)
        filter_cursor = (
            focused.cursor_position
            if focus_identity == "library-prompts-filter" and isinstance(focused, Input)
            else None
        )

        def restore_focus() -> None:
            self._restore_library_prompts_focus(focus_identity, filter_cursor)

        _sync_library_canvas(self, "prompts", then=restore_focus)

    def _clear_library_prompt_selection(self, *, announce: bool) -> None:
        """End Prompt selection and optionally announce the bounded count."""
        count = len(self._library_prompt_selection.entries)
        changed = self._library_prompt_select_mode or count > 0
        if not changed:
            return
        self._library_prompt_select_mode = False
        self._library_prompt_selection = PromptSelectionBasket()
        if self.is_running:
            self._sync_library_prompt_selection(None)
        if announce and count:
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify(f"Selection discarded · {count} prompts")

    def _sync_library_prompt_memberships(self, state) -> None:
        """Patch only membership controls; preserve editor inputs and undo state."""
        if self._library_prompts_mutation_in_flight:
            return
        try:
            self.query_one(
                "#library-prompt-work-pane", LibraryPromptWorkPane
            ).sync_memberships(state)
        except (NoMatches, QueryError):
            pass

    def _reconcile_library_prompt_memberships(self) -> None:
        self._sync_library_prompt_memberships(
            self._library_prompt_collections_controller.membership_state
        )

    def _apply_library_prompt_collection(self, collection_id: int | None) -> None:
        self._request_library_prompts_browse(
            dataclasses.replace(
                self._library_prompt_browse_controller.scope,
                collection_id=collection_id,
                page=1,
            ),
            focus_identity="library-prompts-collection",
        )

    def _sync_library_prompt_collection_label(self) -> None:
        try:
            button = self.query_one("#library-prompts-collection", Button)
        except (NoMatches, QueryError):
            return
        label = self._library_prompt_collections_controller.collection_label(
            self._library_prompt_browse_controller.visible_result.scope.collection_id
        )
        # AC#5/task-14902: the in-place patcher must build the SAME chooser
        # label the canvas composes (recompose discipline).
        button.label = library_choice_label("collection", escape_markup(label))

    def _refresh_library_prompt_after_membership_apply(self) -> None:
        """Refresh retained Items and Library counts after membership Apply.

        The Prompt editor remains mounted in Work while the independent Items
        projection reloads its exact applied scope. Prompt Save state is never
        changed here.
        """
        if self._library_prompts_mutation_in_flight:
            return
        self._request_library_prompts_browse(
            self._library_prompt_browse_controller.mutation_refresh_scope,
            focus_identity="library-prompt-memberships-apply",
        )
        self._refresh_local_source_snapshot()

    def _load_library_prompt_memberships(self) -> None:
        self.run_worker(
            self._library_prompt_collections_controller.load_memberships(),
            exclusive=True,
            group="library_prompt_memberships_load",
        )
        self.call_after_refresh(self._reconcile_library_prompt_memberships)

    def _library_prompts_focus_identity(self) -> str | None:
        """Return the current Prompt canvas descendant's stable DOM id."""
        focused = getattr(self, "focused", None)
        focused_id = getattr(focused, "id", None)
        if (
            not isinstance(focused, Widget)
            or not focused.is_mounted
            or not isinstance(focused_id, str)
            or not focused_id
        ):
            return None
        owners = (
            *self.query("#library-prompts-canvas"),
            *self.query("#library-prompt-work-pane"),
        )
        return (
            focused_id if any(owner in focused.ancestors for owner in owners) else None
        )

    def _restore_library_prompts_focus(
        self,
        focus_identity: str | None,
        filter_cursor: int | None = None,
    ) -> None:
        """Restore a surviving Prompt control, else the stable sort control."""
        if not self._library_list_canvas_showing_list():
            return
        if focus_identity:
            try:
                target = self.query_one(f"#{focus_identity}", Widget)
            except (NoMatches, QueryError):
                target = None
            if target is not None and not getattr(target, "disabled", False):
                target.focus()
                if filter_cursor is not None and isinstance(target, Input):
                    target.call_after_refresh(
                        setattr,
                        target,
                        "cursor_position",
                        min(filter_cursor, len(target.value)),
                    )
                return
        try:
            pager_fallback_id = {
                "library-prompts-page-next": "#library-prompts-page-previous",
                "library-prompts-page-previous": "#library-prompts-page-next",
            }.get(focus_identity)
            fallback_id = pager_fallback_id or (
                "#library-prompts-selection-done"
                if self._library_prompt_select_mode
                else "#library-prompts-sort"
            )
            fallback = self.query_one(fallback_id, Button)
            if not fallback.disabled:
                fallback.focus()
            elif pager_fallback_id:
                self.query_one("#library-prompts-filter", Input).focus()
        except (NoMatches, QueryError):
            pass

    def _stop_library_prompts_search_debounce(self) -> None:
        """Cancel the pending search timer without changing its request scope."""
        timer = self._library_prompts_debounce_timer
        if timer is not None:
            timer.stop()
            self._library_prompts_debounce_timer = None

    def _capture_library_prompts_filter_cursor(
        self,
        request_token: int,
        focus_identity: str | None,
    ) -> None:
        """Bind the active filter caret to one exact browse request."""
        focused = getattr(self, "focused", None)
        self._library_prompts_filter_cursor_context = (
            (request_token, focused.cursor_position)
            if focus_identity == "library-prompts-filter" and isinstance(focused, Input)
            else None
        )

    def _sync_library_prompts_browse_result(
        self,
        result: PromptBrowseResult,
        focus_identity: str | None,
    ) -> LibraryEntryReconcileResult:
        """Project controller state and restore stable Prompt-list focus."""
        current = self._library_prompt_browse_controller.result
        route_key = self._library_entry_route_key()
        generation = self._library_snapshot_state_generation
        if (
            result.request_token != current.request_token
            or not self._library_entry_reconcile_is_current(generation, route_key)
            or self._library_selected_row_id
            not in (LIBRARY_ROW_BROWSE_PROMPTS, LIBRARY_ROW_CREATE_PROMPT)
        ):
            return LibraryEntryReconcileResult.SUPERSEDED

        live_focus_identity = self._library_prompts_focus_identity()
        live_focused = getattr(self, "focused", None)
        cursor_context = self._library_prompts_filter_cursor_context
        filter_cursor = (
            live_focused.cursor_position
            if live_focus_identity == "library-prompts-filter"
            and isinstance(live_focused, Input)
            else cursor_context[1]
            if live_focus_identity is None
            and cursor_context is not None
            and cursor_context[0] == result.request_token
            else None
        )
        restore_identity = live_focus_identity or focus_identity

        def restore_focus() -> None:
            if not self._library_entry_reconcile_is_current(generation, route_key):
                return
            if result.status == "loading" and focus_identity is not None:
                if focus_identity in {
                    "library-prompts-page-next",
                    "library-prompts-page-previous",
                }:
                    return
                try:
                    self.query_one(f"#{focus_identity}", Widget)
                except (NoMatches, QueryError):
                    return
            if result.status == "loading" or restore_identity is not None:
                self._restore_library_prompts_focus(
                    restore_identity,
                    filter_cursor,
                )
            elif not self._library_pending_list_entry_focus:
                self._restore_library_prompts_focus(None)

        if _sync_library_canvas(
            self,
            "prompts",
            then=restore_focus,
            allow_screen_fallback=False,
            sync_prompt_work=False,
        ):
            return LibraryEntryReconcileResult.APPLIED
        return LibraryEntryReconcileResult.FAILED

    def _queue_library_prompts_search(self, query: str) -> None:
        """Debounce one normalized search without blocking the UI loop."""
        if self._library_prompts_mutation_in_flight:
            return
        query = self._safe_text(query, max_length=200).strip()
        controller = self._library_prompt_browse_controller
        if query == controller.scope.query:
            return
        self._stop_library_prompts_search_debounce()
        scope = dataclasses.replace(controller.scope, query=query, page=1)
        token = controller.begin(scope)
        self._capture_library_prompts_filter_cursor(
            token,
            "library-prompts-filter",
        )

        def dispatch() -> None:
            self._library_prompts_debounce_timer = None
            controller.dispatch(
                scope,
                request_token=token,
                focus_identity="library-prompts-filter",
            )

        self._library_prompts_debounce_timer = self.set_timer(
            _LIBRARY_PROMPTS_SEARCH_DEBOUNCE_SECONDS,
            dispatch,
        )

    def _flush_library_prompts_search(self, query: str) -> None:
        """Flush Enter through the pending token without duplicating its call."""
        query = self._safe_text(query, max_length=200).strip()
        pending = self._library_prompts_debounce_timer is not None
        self._stop_library_prompts_search_debounce()
        controller = self._library_prompt_browse_controller
        current = controller.result
        if query != controller.scope.query:
            scope = dataclasses.replace(
                controller.scope,
                query=query,
                page=1,
            )
            token = controller.begin(scope)
        elif pending and current.status == "loading":
            scope = controller.scope
            token = current.request_token
        else:
            return
        self._capture_library_prompts_filter_cursor(
            token,
            "library-prompts-filter",
        )
        controller.dispatch(
            scope,
            request_token=token,
            focus_identity="library-prompts-filter",
        )

    def _invalidate_library_prompts_browse(self) -> None:
        """Cancel presentation timing and supersede the active browse token."""
        self._stop_library_prompts_search_debounce()
        self._library_prompts_filter_cursor_context = None
        self._library_prompt_browse_controller.invalidate()

    @on(Button.Pressed, "#library-prompts-select")
    async def handle_library_prompts_select(self, event: Button.Pressed) -> None:
        """Enter Prompt selection mode without changing the settled page."""
        event.stop()
        if (
            self._library_prompts_mutation_in_flight
            or self._library_prompt_browse_controller.freshness == "stale"
        ):
            return
        state = self._build_library_prompts_state()
        if (
            self._library_prompt_browse_controller.visible_result.status != "ready"
            or not state.rows
        ):
            return
        if not await self._flush_library_prompt_save():
            return
        self._library_prompt_select_mode = True
        self._sync_library_prompt_selection(None)

    @on(Button.Pressed, "#library-prompts-select-page")
    def handle_library_prompts_select_page(self, event: Button.Pressed) -> None:
        """Add every valid row from the currently settled Prompt page."""
        event.stop()
        if (
            self._library_prompts_mutation_in_flight
            or self._library_prompt_browse_controller.freshness == "stale"
        ):
            return
        result = self._library_prompt_browse_controller.visible_result
        if not self._library_prompt_select_mode or result.status != "ready":
            return
        state = self._build_library_prompts_state()
        try:
            page = tuple(
                PromptSelectionEntry(
                    row.prompt_id,
                    row.version,
                    row.name,
                    row.artifact_type,
                )
                for row in state.rows
            )
        except (TypeError, ValueError):
            return
        if not page:
            return
        selection = self._library_prompt_selection.select_page(page)
        if selection is self._library_prompt_selection:
            return
        focus_identity = self._library_prompts_focus_identity()
        self._library_prompt_selection = selection
        self._sync_library_prompt_selection(focus_identity)

    @on(Button.Pressed, "#library-prompts-clear-selection")
    def handle_library_prompts_clear_selection(self, event: Button.Pressed) -> None:
        """Clear the basket silently while remaining in selection mode."""
        event.stop()
        if self._library_prompts_mutation_in_flight:
            return
        selection = self._library_prompt_selection.clear()
        if selection is self._library_prompt_selection:
            return
        focus_identity = self._library_prompts_focus_identity()
        self._library_prompt_selection = selection
        self._sync_library_prompt_selection(focus_identity)

    @on(Button.Pressed, "#library-prompts-selection-done")
    def handle_library_prompts_selection_done(self, event: Button.Pressed) -> None:
        """Discard the basket and return to ordinary Prompt browsing."""
        event.stop()
        if self._library_prompts_mutation_in_flight:
            return
        self._clear_library_prompt_selection(announce=True)

    @on(Button.Pressed, "#library-prompts-collection")
    def handle_library_prompts_collection(self, event: Button.Pressed) -> None:
        event.stop()
        if self._library_prompts_mutation_in_flight:
            return
        self._library_prompt_collections_controller.open_manager("browse")

    @on(Button.Pressed, "#library-prompt-memberships-manage")
    def handle_library_prompt_memberships_manage(self, event: Button.Pressed) -> None:
        event.stop()
        if self._library_prompts_mutation_in_flight:
            return
        if self._library_prompt_collections_controller.membership_state.can_retry_load:
            self._load_library_prompt_memberships()
        else:
            self._library_prompt_collections_controller.open_manager("membership")

    @on(Button.Pressed, "#library-prompt-memberships-apply")
    def handle_library_prompt_memberships_apply(self, event: Button.Pressed) -> None:
        event.stop()
        if self._library_prompts_mutation_in_flight:
            return
        self.run_worker(
            self._await_library_prompt_durable_call(
                self._library_prompt_collections_controller.apply_memberships()
            ),
            exclusive=True,
            group="library_prompt_memberships_apply",
        )

    @on(Input.Changed, "#library-prompts-filter")
    def handle_library_prompts_filter_changed(self, event: Input.Changed) -> None:
        """Queue the service-backed Prompt search after a short debounce."""
        if self._library_prompts_mutation_in_flight:
            return
        self._queue_library_prompts_search(event.value)

    @on(Button.Pressed, "#library-prompts-retry")
    def handle_library_prompts_retry(self, event: Button.Pressed) -> None:
        """Retry the failed exact Prompt scope with a fresh request token."""
        event.stop()
        if self._library_prompts_mutation_in_flight:
            return
        self._stop_library_prompts_search_debounce()
        self._library_prompt_browse_controller.retry(
            focus_identity="library-prompts-retry"
        )

    @on(Button.Pressed, "#library-prompt-detail-retry")
    def handle_library_prompt_detail_retry(self, event: Button.Pressed) -> None:
        """Retry only the still-selected Prompt detail with a fresh fence."""
        event.stop()
        prompt_id = self._selected_prompt_id
        if (
            self._library_prompts_mutation_in_flight
            or not self._library_prompt_detail_retryable
            or type(prompt_id) is not int
        ):
            return
        generation, mutation_generation = self._claim_library_prompt_detail_generation()
        self.run_worker(
            self._refresh_library_prompt_detail(
                prompt_id,
                request_generation=generation,
                mutation_generation=mutation_generation,
            ),
            exclusive=True,
            group="library_prompt_detail",
        )
        _sync_library_canvas(self, "prompts")

    @on(Button.Pressed, "#library-prompts-import")
    async def handle_library_prompts_import(self, event: Button.Pressed) -> None:
        """Open the inline Import row below the prompts toolbar.

        Idempotent while already open -- pressing Import… again does not
        close it. Cancel is the only way to close the row once opened,
        avoiding a confusing double-duty toggle that would also hide a
        just-shown outcome line.

        Args:
            event: Button press event emitted by the "Import…" action.
        """
        event.stop()
        if self._library_prompts_mutation_in_flight:
            return
        if self._library_prompts_import_open:
            return
        if not await self._flush_library_prompt_save():
            self._notify_prompt_dirty_veto()
            return
        self._library_prompts_import_open = True
        self._library_prompts_import_path = ""
        self._library_prompts_import_status = ""
        # task-21116: canvas-scoped open (see the skills twin for the
        # focus-follow-up rationale).
        _sync_library_canvas(
            self,
            "prompts",
            then=lambda: self._focus_library_control("#library-prompts-import-path"),
        )

    @on(Button.Pressed, "#library-prompts-import-cancel")
    def handle_library_prompts_import_cancel(self, event: Button.Pressed) -> None:
        """Close the inline Import row, discarding any typed path/outcome.

        Args:
            event: Button press event emitted by the Import row's
                "Cancel" action.
        """
        event.stop()
        if self._library_prompts_mutation_in_flight:
            return
        self._library_prompts_import_open = False
        self._library_prompts_import_path = ""
        self._library_prompts_import_status = ""
        # task-21116: canvas-scoped close; focus returns to the Import…
        # opener the row folds back into.
        _sync_library_canvas(
            self,
            "prompts",
            then=lambda: self._focus_library_control("#library-prompts-import"),
        )

    @on(Button.Pressed, "#library-prompts-import-browse")
    def handle_library_prompts_import_browse(self, event: Button.Pressed) -> None:
        """Push a ``FileOpen`` dialog to pick a local file for prompt import
        (Task 8b D4).

        Mirrors ``handle_library_ingest_browse``'s dialog flow exactly. The
        shared ``FileOpen`` dialog (``Third_Party/textual_fspicker``) has no
        directory-selection mode, so this only covers the file case --
        importing a folder still requires typing its path into the Import
        row's path ``Input`` by hand.

        Args:
            event: Button press event emitted by the "Browse…" action.
        """
        event.stop()
        if self._library_prompts_mutation_in_flight:
            return

        async def browse_callback(selected_path: Path | None) -> None:
            if self._library_prompts_mutation_in_flight:
                return
            if selected_path is None:
                return
            self._library_prompts_import_path = str(selected_path)
            self.refresh(recompose=True)

        self.app.push_screen(
            FileOpen(title="Import Prompts (file)"),
            browse_callback,
        )

    @on(Input.Changed, "#library-prompts-import-path")
    def handle_library_prompts_import_path_changed(self, event: Input.Changed) -> None:
        """Track the Import row's path text as the user types it (state only).

        Args:
            event: Input change event emitted by the Import row's path field.
        """
        event.stop()
        if self._library_prompts_mutation_in_flight:
            return
        self._library_prompts_import_path = event.value

    @on(Input.Submitted, "#library-prompts-import-path")
    def handle_library_prompts_import_path_submitted(
        self, event: Input.Submitted
    ) -> None:
        """Run the import when Enter is pressed in the Import row's path field.

        Args:
            event: Input submission event emitted by the Import row's
                path field.
        """
        event.stop()
        if self._library_prompts_mutation_in_flight:
            return
        self._start_library_prompts_import()

    @on(Button.Pressed, "#library-prompts-import-run")
    def handle_library_prompts_import_run(self, event: Button.Pressed) -> None:
        """Run the import when the Import row's "Import" action is pressed.

        Args:
            event: Button press event emitted by the Import row's
                "Import" action.
        """
        event.stop()
        if self._library_prompts_mutation_in_flight:
            return
        self._start_library_prompts_import()

    def _start_library_prompts_import(self) -> Worker[None] | None:
        """Validate the Import row has a non-blank path, then run the import worker.

        Worker-executed (one app-owned slot) since it performs file IO plus
        one or more service calls per parsed prompt -- never inline on the UI
        thread. App ownership lets the batch finish if its initiating screen
        is unmounted. A blank path is a quiet inline status line, matching
        every other Library form's "nothing to do yet" gate (e.g.
        ``_submit_library_ingest_form``'s blank-path notice).
        """
        if self._library_prompts_mutation_in_flight:
            return None
        if self._library_prompts_view != "list":
            return None
        raw_path = self._library_prompts_import_path.strip()
        if not raw_path:
            self._apply_library_prompts_import_status(
                "Please enter a file or folder path."
            )
            return None
        for worker in self.app_instance.workers:
            if (
                worker.node is self.app_instance
                and worker.group == _LIBRARY_PROMPTS_IMPORT_WORKER_GROUP
                and not worker.is_finished
            ):
                self._apply_library_prompts_import_status(
                    "A prompt import is already in progress."
                )
                return worker
        return self.app_instance.run_worker(
            self._await_library_prompt_durable_call(
                self._run_library_prompts_import(raw_path)
            ),
            group=_LIBRARY_PROMPTS_IMPORT_WORKER_GROUP,
        )

    def _current_library_prompt_editor_state(
        self, base: PromptEditorState | None = None
    ) -> PromptEditorState:
        """Return metadata plus the screen-owned immutable block working copy."""
        if base is None:
            detail = (
                self._library_prompt_detail
                if isinstance(self._library_prompt_detail, Mapping)
                else {}
            )
            base = build_prompt_editor_state(
                detail, capabilities=self._library_prompt_capabilities
            )
        block_state = getattr(self, "_library_prompt_block_state", None)
        if block_state is None:
            return base
        return dataclasses.replace(
            base,
            artifact_type=block_state.artifact_type,
            block_editor_state=block_state,
            compiled_system_preview=block_state.compiled_system,
            compiled_user_preview=block_state.compiled_user,
            system_prompt=block_state.compiled_system,
            user_prompt=block_state.compiled_user,
            capabilities=self._library_prompt_capabilities,
        )

    def _library_prompt_can_update_original(self) -> bool:
        """Return whether the captured source can be conditionally replaced."""
        state = self._library_prompt_block_state
        if (
            state is None
            or self._selected_prompt_id is None
            or type(self._library_prompt_version) is not int
            or self._library_prompt_version < 1
            or not self._library_prompt_capabilities.conditional_update
        ):
            return False
        expected_kind = (
            "block_prompt" if state.artifact_type == "prompt" else "block_recipe"
        )
        return (
            state.definition.kind == expected_kind
            and (state.definition.schema_version, state.definition.kind)
            in self._library_prompt_capabilities.structured_kinds
            and state.artifact_type in self._library_prompt_capabilities.artifact_types
        )

    def _claim_library_prompt_detail_generation(self) -> tuple[int, int]:
        """Start one Prompt detail request and return its complete async fence."""
        self._library_prompt_detail_generation += 1
        self._library_prompt_detail_loading = True
        self._library_prompt_detail_error = ""
        self._library_prompt_detail_retryable = False
        return (
            self._library_prompt_detail_generation,
            self._library_prompt_mutation_generation,
        )

    def _invalidate_library_prompt_detail_generation(self) -> None:
        """Refuse every pending Prompt detail settlement."""
        self._library_prompt_detail_generation += 1
        self._library_prompt_detail_loading = False
        self._library_prompt_detail_error = ""
        self._library_prompt_detail_retryable = False

    def _library_prompt_loading_notice(self) -> str:
        """Describe selected/loaded Prompt identity without conflating them."""
        selected = self._library_prompt_detail_selected_name or (
            f"Prompt {self._selected_prompt_id}"
            if self._selected_prompt_id is not None
            else "prompt"
        )
        loaded = self._library_prompt_original_name.strip()
        if loaded and self._library_prompt_loaded_id != self._selected_prompt_id:
            return f"Loading “{selected}”… showing “{loaded}” until ready."
        return f"Loading “{selected}”…"

    def _library_prompt_detail_failure_notice(self) -> str:
        """Combine a scoped failure with truthful selected/loaded identity."""
        copy = self._library_prompt_detail_error
        loaded = self._library_prompt_original_name.strip()
        selected = self._library_prompt_detail_selected_name or (
            f"Prompt {self._selected_prompt_id}"
            if self._selected_prompt_id is not None
            else "the selected Prompt"
        )
        if loaded and self._library_prompt_loaded_id != self._selected_prompt_id:
            return (
                f"{copy} Still showing “{loaded}” while “{selected}” remains selected."
            )
        return copy

    def _library_prompt_detail_request_is_current(
        self,
        *,
        prompt_id: int,
        generation: int,
        mutation_generation: int,
        entry_route_key: tuple[object, ...] | None,
    ) -> bool:
        """Return whether one detail outcome still owns the Prompt work pane."""
        return bool(
            generation == self._library_prompt_detail_generation
            and mutation_generation == self._library_prompt_mutation_generation
            and prompt_id == self._selected_prompt_id
            and self._library_prompts_view == "editor"
            and (
                entry_route_key is None
                or entry_route_key == self._library_entry_route_key()
            )
        )

    def _apply_library_prompt_detail_failure(
        self,
        *,
        copy: str,
        retryable: bool,
        entry_origin: bool,
    ) -> LibraryEntryReconcileResult | None:
        """Keep loaded work truthful while exposing one scoped detail failure."""
        self._library_prompt_detail_loading = False
        self._library_prompt_detail_error = copy
        self._library_prompt_detail_retryable = retryable
        synced = False
        if self.is_mounted:
            synced = _sync_library_canvas(
                self,
                "prompts",
                allow_screen_fallback=not entry_origin,
            )
        if entry_origin:
            return (
                LibraryEntryReconcileResult.APPLIED
                if synced
                else LibraryEntryReconcileResult.FAILED
            )
        return None

    async def _refresh_library_prompt_detail(
        self,
        prompt_id: int,
        *,
        open_history: bool = False,
        expected_history_scope: tuple[str, int] | None = None,
        entry_origin: bool = False,
        request_generation: int | None = None,
        mutation_generation: int | None = None,
        expected_version: int | None = None,
    ) -> LibraryEntryReconcileResult | None:
        """Fetch and store the full detail for a selected Library prompt.

        Mirrors ``_refresh_library_note_detail``: offloads the (possibly
        blocking) ``get_prompt`` service call via ``_run_library_service_call``
        and recomposes once the fetched detail (or a cleared state) has
        been stored.

        Unlike notes' ``get_note_detail`` (which never carries keywords, so
        its private session port enriches the normalized load reply),
        the local backend's ``get_prompt`` seam is backed by
        ``PromptsDatabase.fetch_prompt_details``, which already joins
        keywords into the returned mapping -- no second enrichment call is
        needed here.

        Args:
            prompt_id: The Library prompt id to fetch full detail for.
            open_history: Whether freshly adopted history starts disclosed.
            expected_history_scope: Optional exact restore/editor session identity.
                Generic detail fetches omit this guard.
        """
        if request_generation is None or mutation_generation is None:
            request_generation, mutation_generation = (
                self._claim_library_prompt_detail_generation()
            )
        entry_route_key = self._library_entry_route_key() if entry_origin else None
        if expected_history_scope is not None and not (
            self._library_prompt_history_controller.matches_scope(
                prompt_uuid=expected_history_scope[0],
                scope_token=expected_history_scope[1],
            )
        ):
            return LibraryEntryReconcileResult.SUPERSEDED if entry_origin else None
        service = getattr(self.app_instance, "prompt_scope_service", None)
        get_prompt = getattr(service, "get_prompt", None)
        if not callable(get_prompt):
            if not self._library_prompt_detail_request_is_current(
                prompt_id=prompt_id,
                generation=request_generation,
                mutation_generation=mutation_generation,
                entry_route_key=entry_route_key,
            ):
                return LibraryEntryReconcileResult.SUPERSEDED if entry_origin else None
            return self._apply_library_prompt_detail_failure(
                copy="Couldn't load the selected Prompt. The local service is unavailable.",
                retryable=True,
                entry_origin=entry_origin,
            )
        failed = False
        try:
            detail = await self._run_library_service_call(
                get_prompt,
                mode="local",
                prompt_identifier=prompt_id,
                include_deleted=True,
                isolate_in_worker=True,
            )
        except Exception:
            logger.opt(exception=True).warning(
                f"Failed to load Library prompt detail for {prompt_id!r}."
            )
            detail = None
            failed = True
        if expected_history_scope is not None:
            state = self._library_prompt_history_controller.state
            if not (
                state is not None
                and self._library_prompt_history_controller.matches_scope(
                    prompt_uuid=expected_history_scope[0],
                    scope_token=expected_history_scope[1],
                )
            ):
                return LibraryEntryReconcileResult.SUPERSEDED if entry_origin else None
            # The disclosure may have changed while the detail call was awaited.
            # Adopt the live, still-matching scope state rather than the caller's
            # pre-await Boolean.
            open_history = state.is_open
        # Discard out-of-order results: the same stale-race guard as
        # ``_refresh_library_note_detail``.
        if not self._library_prompt_detail_request_is_current(
            prompt_id=prompt_id,
            generation=request_generation,
            mutation_generation=mutation_generation,
            entry_route_key=entry_route_key,
        ):
            return LibraryEntryReconcileResult.SUPERSEDED if entry_origin else None
        if not isinstance(detail, Mapping):
            if failed:
                return self._apply_library_prompt_detail_failure(
                    copy=(
                        "Couldn't load the selected Prompt. Check the local Library "
                        "and retry."
                    ),
                    retryable=True,
                    entry_origin=entry_origin,
                )
            logger.info(f"Library prompt {prompt_id!r} is no longer available.")
            self._request_library_prompts_browse(
                self._library_prompt_browse_controller.mutation_refresh_scope,
                focus_identity=None,
            )
            self._refresh_local_source_snapshot()
            return self._apply_library_prompt_detail_failure(
                copy="The selected Prompt is no longer available.",
                retryable=False,
                entry_origin=entry_origin,
            )
        detail_version = detail.get("version")
        if (
            expected_version is not None
            and type(detail_version) is int
            and detail_version != expected_version
        ):
            return self._apply_library_prompt_detail_failure(
                copy=(
                    "The selected Prompt changed since this Items page loaded. "
                    "Retry to load its current version."
                ),
                retryable=True,
                entry_origin=entry_origin,
            )
        self._library_prompt_detail_loading = False
        self._library_prompt_detail_error = ""
        self._library_prompt_detail_retryable = False
        self._adopt_library_prompt_persisted_detail(
            detail,
            open_history=open_history,
        )
        self._load_library_prompt_memberships()
        if self.is_mounted:
            # See the skills twin: arming must not be lost to the
            # canvas-pump race (task-15457 review I4b).
            synced = _sync_library_canvas(
                self,
                "prompts",
                then=self._arm_library_prompt_editor,
                allow_screen_fallback=not entry_origin,
            )
            if entry_origin:
                return (
                    LibraryEntryReconcileResult.APPLIED
                    if synced
                    else LibraryEntryReconcileResult.FAILED
                )
        return None

    def _adopt_library_prompt_persisted_detail(
        self,
        detail: Mapping[str, Any],
        *,
        status: str = "",
        open_history: bool | None = None,
    ) -> None:
        """Adopt one persisted Prompt identity and initialize its history scope."""
        if open_history is None:
            open_history = bool(
                self._library_prompt_history_state is not None
                and self._library_prompt_history_state.is_open
            )
        self._library_prompt_detail = dict(detail)
        editor_state = build_prompt_editor_state(
            self._library_prompt_detail,
            capabilities=self._library_prompt_capabilities,
        )
        prompt_id = self._library_prompt_detail.get("local_id")
        if type(prompt_id) is not int:
            prompt_id = self._library_prompt_detail.get("id")
        if type(prompt_id) is int:
            self._library_prompt_loaded_id = prompt_id
            self._selected_prompt_id = prompt_id
        self._library_prompt_block_state = editor_state.block_editor_state
        self._library_prompt_detached_structured = False
        self._library_prompt_original_name = editor_state.name
        self._library_prompt_version = editor_state.version
        self._library_prompt_dirty = False
        self._library_prompt_status = status
        self._library_prompt_conflict_snapshot = None
        self._library_prompt_include_starter_content = False
        self._library_prompt_editor_armed = False
        self._initialize_library_prompt_history(
            self._library_prompt_detail, open_history=open_history
        )

    def _detach_library_prompt_working_copy(self, detail: Mapping[str, Any]) -> None:
        """Detach a saved Prompt/Recipe identity before editing an unsaved copy."""
        detached = dict(detail)
        for source_field in (
            "id",
            "local_id",
            "server_id",
            "uuid",
            "source_id",
            "version",
            "created_at",
            "last_modified",
            "last_used_at",
        ):
            detached.pop(source_field, None)
        self._library_prompt_detail = detached
        self._selected_prompt_id = None
        self._library_prompt_loaded_id = None
        self._library_prompt_original_name = ""
        self._library_prompt_version = None
        self._library_prompt_conflict_snapshot = None
        self._invalidate_library_prompt_history()
        self._library_prompt_collections_controller.invalidate()

    def _arm_library_prompt_editor(self) -> None:
        """Enable dirty-tracking once the prompt editor is mounted."""
        if self._library_prompts_mutation_in_flight:
            return
        self._library_prompt_editor_armed = True

    def _invalidate_library_prompt_history(self) -> None:
        self._library_prompt_history_controller.invalidate()

    def _initialize_library_prompt_history(
        self, detail: Mapping[str, Any], *, open_history: bool = False
    ) -> None:
        self._library_prompt_history_controller.initialize(
            detail, open_history=open_history
        )

    def _sync_library_prompt_history_region(
        self, state: PromptHistoryState | None = None
    ) -> None:
        """Recompose the history sub-tree without remounting the editor shell."""
        target_state = (
            state
            if state is not None
            else self._library_prompt_history_controller.state
        )
        try:
            region = self.query_one(
                "#library-prompt-history-region", LibraryPromptHistoryRegion
            )
        except (NoMatches, QueryError):
            return
        region.sync_state(
            target_state,
            dirty=self._library_prompt_dirty,
            current_compatible=self._library_prompt_block_state is not None,
        )
        # A detail-load canvas recompose can overlap a fast count worker:
        # the targeted sync above may have reached the outgoing region after
        # the incoming canvas captured its initial ellipsis state. Reconcile
        # the currently mounted region once after refresh, reading current
        # controller/editor state so a stale callback can never paint an old
        # prompt into a new scope.
        if self.is_mounted:
            self.call_after_refresh(self._reconcile_library_prompt_history_region)

    def _reconcile_library_prompt_history_region(self) -> None:
        """Repair only a history-region/canvas-recompose overlap."""
        try:
            region = self.query_one(
                "#library-prompt-history-region", LibraryPromptHistoryRegion
            )
        except (NoMatches, QueryError):
            return
        desired = (
            self._library_prompt_history_controller.state,
            self._library_prompt_dirty,
            self._library_prompt_block_state is not None,
        )
        region.sync_state(
            desired[0],
            dirty=desired[1],
            current_compatible=desired[2],
        )

    @on(LibraryPromptHistoryRegion.Ready)
    def _on_library_prompt_history_region_ready(
        self, event: LibraryPromptHistoryRegion.Ready
    ) -> None:
        """Live-sync a region mounted after an overlapping controller publish."""
        event.stop()
        self._reconcile_library_prompt_history_region()

    def _request_library_prompt_history_count(self) -> None:
        self._library_prompt_history_controller.retry_count()

    def _request_library_prompt_history_page(self) -> None:
        self._library_prompt_history_controller.request_page()

    def _library_prompt_history_action_is_current(self, event: Any) -> bool:
        """Reject semantic actions emitted by an outgoing history region."""
        if self._library_prompts_mutation_in_flight:
            return False
        return self._library_prompt_history_controller.matches_scope(
            prompt_uuid=event.prompt_uuid,
            scope_token=event.scope_token,
        )

    @on(LibraryPromptHistoryRegion.DisclosureOpened)
    def handle_library_prompt_history_opened(
        self, event: LibraryPromptHistoryRegion.DisclosureOpened
    ) -> None:
        """Lazy-load the first retained page when the disclosure opens."""
        if not self._library_prompt_history_action_is_current(event):
            return
        state = self._library_prompt_history_state
        if state is not None and not state.is_open:
            self._request_library_prompt_history_page()

    @on(LibraryPromptHistoryRegion.DisclosureClosed)
    def handle_library_prompt_history_closed(
        self, event: LibraryPromptHistoryRegion.DisclosureClosed
    ) -> None:
        """Use the pure close reset when the disclosure collapses."""
        if not self._library_prompt_history_action_is_current(event):
            return
        self._library_prompt_history_controller.close()

    @on(LibraryPromptHistoryRegion.CountRetryRequested)
    def handle_library_prompt_history_retry_count(
        self, event: LibraryPromptHistoryRegion.CountRetryRequested
    ) -> None:
        if not self._library_prompt_history_action_is_current(event):
            return
        self._request_library_prompt_history_count()

    @on(LibraryPromptHistoryRegion.PageRequested)
    def handle_library_prompt_history_request_page(
        self, event: LibraryPromptHistoryRegion.PageRequested
    ) -> None:
        if not self._library_prompt_history_action_is_current(event):
            return
        self._request_library_prompt_history_page()

    @on(LibraryPromptHistoryRegion.RowSelected)
    def handle_library_prompt_history_row(
        self, event: LibraryPromptHistoryRegion.RowSelected
    ) -> None:
        """Select an already-loaded immutable preview with reducer guards."""
        if not self._library_prompt_history_action_is_current(event):
            return
        self._library_prompt_history_controller.select(
            change_id=event.change_id,
            source_version=event.source_version,
        )

    @on(LibraryPromptHistoryRegion.RestoreRequested)
    def handle_library_prompt_history_restore(
        self, event: LibraryPromptHistoryRegion.RestoreRequested
    ) -> None:
        """Confirm a gated restore without exposing retained Prompt bodies."""
        if not self._library_prompt_history_action_is_current(event):
            return
        state = self._library_prompt_history_state
        if state is None or self._library_prompt_block_state is None:
            return
        gate = self._library_prompt_history_controller.restore_gate(
            dirty=self._library_prompt_dirty
        )
        if (
            gate is None
            or not gate.enabled
            or gate.target is None
            or state.selected is None
        ):
            self._sync_library_prompt_history_region()
            return
        selected_type = state.selected.row.artifact_type
        current_type = self._library_prompt_block_state.artifact_type
        type_change = ""
        if {selected_type, current_type}.issubset({"prompt", "recipe"}) and (
            selected_type != current_type
        ):
            type_change = (
                f" This changes the artifact type from {current_type.title()} "
                f"to {selected_type.title()}."
            )
        target = gate.target
        modal = ConfirmationDialog(
            title="Restore retained version?",
            message=(
                f"Restore retained v{target.source_version} over current "
                f"v{target.expected_current_version}?{type_change} Confirming "
                "creates a new current version."
            ),
            confirm_label="Restore",
            cancel_label="Cancel",
        )
        self.app.push_screen(
            modal,
            lambda confirmed: self._confirm_library_prompt_history_restore(
                bool(confirmed),
                prompt_uuid=target.prompt_uuid,
                change_id=target.change_id,
                source_version=target.source_version,
                expected_current_version=target.expected_current_version,
            ),
        )

    @on(LibraryPromptHistoryRegion.ReloadRequested)
    def handle_library_prompt_history_reload(
        self, event: LibraryPromptHistoryRegion.ReloadRequested
    ) -> None:
        """Reset and reload the first retained page without a settled count fetch."""
        if not self._library_prompt_history_action_is_current(event):
            return
        self._library_prompt_history_controller.reload_page()

    def _confirm_library_prompt_history_restore(
        self,
        confirmed: bool,
        *,
        prompt_uuid: str,
        change_id: int,
        source_version: int,
        expected_current_version: int,
    ) -> None:
        """Revalidate the modal's captured target, then start one restore."""
        if self._library_prompts_mutation_in_flight or not confirmed:
            return
        request = self._library_prompt_history_controller.begin_restore(
            dirty=self._library_prompt_dirty,
            expected_target=(
                prompt_uuid,
                change_id,
                source_version,
                expected_current_version,
            ),
        )
        if request is None:
            return
        self.run_worker(
            self._await_library_prompt_durable_call(
                self._restore_library_prompt_history(request)
            ),
            exclusive=True,
            group="library_prompt_history_restore",
            name="library_prompt_history_restore",
        )

    async def _restore_library_prompt_history(
        self, request: PromptHistoryRestoreRequest
    ) -> None:
        """Conditionally restore through the scope service off the UI path."""
        if self._library_prompts_mutation_in_flight:
            return
        outcome = await self._library_prompt_history_controller.restore(request)
        if self._library_prompts_mutation_in_flight:
            return
        if outcome is None:
            return
        if outcome.kind == "conflict":
            editor = self._current_library_prompt_editor_state()
            self._enter_library_prompt_conflict(
                name=editor.name,
                author=editor.author,
                details=editor.details,
                system_prompt=editor.system_prompt,
                user_prompt=editor.user_prompt,
                keywords_text=editor.keywords_csv,
            )
            return
        if outcome.kind == "restored":
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify(outcome.message)
            prompt_id = self._selected_prompt_id
            if isinstance(prompt_id, int):
                await self._refresh_library_prompt_detail(
                    prompt_id,
                    expected_history_scope=(request.prompt_uuid, request.scope_token),
                )
            return

    def _enter_library_prompt_create_editor(self) -> None:
        """Open the in-canvas prompt editor on a blank, not-yet-saved record.

        Entered via the Create rail's "New prompt" row
        (``LIBRARY_ROW_CREATE_PROMPT``, whose ``target_id`` is ``"prompts"``
        -- the SAME canvas kind Browse > Prompts targets -- and via the
        Duplicate action (see ``handle_library_prompt_duplicate``, which
        pre-fills the blank record from the current prompt's fields after
        calling this).

        ``_selected_prompt_id`` stays ``None``: the sentinel
        ``_save_library_prompt`` reads to route its scope-service
        ``save_prompt`` call into the create path (``prompt_identifier=None``)
        instead of update, and the sentinel ``prompt_editor_meta_line`` reads
        to render "New prompt" instead of "Modified … · vN".
        ``_library_prompt_detail`` is set to ``{}`` (not ``None``) so the
        editor renders blank fields immediately -- ``None`` would instead
        show the "Loading prompt…" placeholder the browse-and-fetch path
        uses while ``_refresh_library_prompt_detail`` is in flight; there is
        nothing to fetch here.
        """
        if self._library_prompts_mutation_in_flight:
            return
        self._clear_library_prompt_selection(announce=True)
        self._invalidate_library_prompt_detail_generation()
        self._selected_prompt_id = None
        self._library_prompt_loaded_id = None
        self._library_prompts_view = "editor"
        self._library_prompt_detail = {}
        editor_state = build_prompt_editor_state(
            self._library_prompt_detail,
            capabilities=self._library_prompt_capabilities,
        )
        self._library_prompt_block_state = editor_state.block_editor_state
        self._library_prompt_detached_structured = False
        self._library_prompt_original_name = ""
        self._library_prompt_version = None
        self._library_prompt_dirty = False
        self._library_prompt_status = ""
        self._library_prompt_conflict_snapshot = None
        self._library_prompt_include_starter_content = False
        self._library_prompt_editor_armed = False
        self._invalidate_library_prompt_history()
        self._library_prompt_collections_controller.invalidate()

    def _mark_library_prompt_dirty(self) -> None:
        """Record an in-progress prompt edit.

        Ignored until ``_library_prompt_editor_armed`` is set and while the
        live fields still equal their backing state. Textual can deliver a
        recomposed field's mount-time ``Changed`` event after the post-refresh
        arm callback, so value equality—not callback timing—is the reliable
        discriminator. Unlike the notes editor, this never arms an autosave
        timer because the prompt editor is explicit-Save-only.

        Task 8c U6: the dirty flag was previously invisible until the
        ``flush_pending_work`` veto fired on nav-away. On the False->True
        transition, this patches ``#library-prompt-meta`` in place (via
        ``_update_library_prompt_meta_static``, the same targeted-Static
        pattern ``save-status`` already uses) so the unsaved marker appears
        immediately -- deliberately NOT a full ``self.refresh(recompose=True)``,
        which would remount the Input/TextArea fields on every keystroke and
        re-trigger their spurious mount-time ``Changed`` event. Guarded to
        the transition only (not every subsequent keystroke) since the
        flag/marker do not change again until Save or navigation.
        """
        if self._library_prompts_mutation_in_flight:
            return
        if not self._library_prompt_editor_armed:
            return
        if self._library_prompt_text_fields_match_state():
            return
        was_dirty = self._library_prompt_dirty
        self._library_prompt_dirty = True
        if not was_dirty:
            self._update_library_prompt_meta_static()
            self._sync_library_prompt_history_region()
            self._set_library_prompt_discard_enabled(True)

    @on(Input.Changed, "#library-prompt-name")
    @on(Input.Changed, "#library-prompt-author")
    @on(Input.Changed, "#library-prompt-details")
    @on(Input.Changed, "#library-prompt-keywords")
    def handle_library_prompt_input_changed(self, event: Input.Changed) -> None:
        """Mark the open prompt dirty on a field edit.

        Args:
            event: Input change event emitted by one of the editor's
                single-line fields.
        """
        self._mark_library_prompt_dirty()

    @on(TextArea.Changed, "#library-prompt-system")
    @on(TextArea.Changed, "#library-prompt-user")
    def handle_library_prompt_textarea_changed(self, event: TextArea.Changed) -> None:
        """Mark the open prompt dirty on a System/User prompt edit.

        Args:
            event: Text change event emitted by one of the editor's
                ``TextArea`` fields.
        """
        self._mark_library_prompt_dirty()

    @on(Button.Pressed, "#library-prompt-mode-basic")
    @on(Button.Pressed, "#library-prompt-mode-advanced")
    @on(Button.Pressed, "#library-prompt-mode-info")
    async def handle_library_prompt_editor_mode(self, event: Button.Pressed) -> None:
        """Switch the three mounted Prompt projections without replacing the draft."""
        event.stop()
        requested = {
            "library-prompt-mode-basic": "basic",
            "library-prompt-mode-advanced": "advanced",
            "library-prompt-mode-info": "info",
        }.get(event.button.id, "basic")
        state = self._current_library_prompt_editor_state()
        if requested == "basic" and self._library_prompt_basic_unavailable_reason(
            state,
            conflict=self._library_prompt_conflict_snapshot is not None,
        ):
            return
        fields = self._read_library_prompt_editor_fields()
        if fields is not None and isinstance(self._library_prompt_detail, Mapping):
            name, author, details, system, user, keywords = fields
            detail = dict(self._library_prompt_detail)
            detail.update(
                {
                    "name": name,
                    "author": author,
                    "details": details,
                    "system_prompt": system,
                    "user_prompt": user,
                    "keywords": keywords,
                }
            )
            self._library_prompt_detail = detail
        try:
            canvas = self.query_one("#library-prompt-work-pane", LibraryPromptWorkPane)
        except NoMatches:
            return
        await canvas.set_editor_mode(requested)
        self._library_prompt_editor_mode = requested
        library_config = self.app_instance.app_config.setdefault("library", {})
        if isinstance(library_config, dict):
            library_config["prompt_editor_mode"] = requested
        self.run_worker(
            self._persist_library_prompt_editor_mode(requested),
            group="library_prompt_editor_mode",
            exclusive=True,
        )

    def _capture_library_prompt_block_state(
        self, state: PromptBlockEditorState
    ) -> None:
        """Adopt one child-editor transition without recomposing its widgets."""
        if self._library_prompts_mutation_in_flight:
            return
        self._library_prompt_block_state = state
        if isinstance(self._library_prompt_detail, Mapping):
            detail = dict(self._library_prompt_detail)
            detail["artifact_type"] = state.artifact_type
            detail["system_prompt"] = state.compiled_system
            detail["user_prompt"] = state.compiled_user
            self._library_prompt_detail = detail
        was_dirty = self._library_prompt_dirty
        self._library_prompt_dirty = True
        if not was_dirty:
            self._update_library_prompt_meta_static()
            self._sync_library_prompt_history_region()
            self._set_library_prompt_discard_enabled(True)

    def on_prompt_block_editor_block_field_changed(
        self, event: PromptBlockEditor.BlockFieldChanged
    ) -> None:
        """Capture an incremental block edit after the canvas patches previews."""
        self._capture_library_prompt_block_state(event.state)

    def on_prompt_block_editor_block_action_requested(
        self, event: PromptBlockEditor.BlockActionRequested
    ) -> None:
        """Capture an incremental add/move/duplicate/delete transition."""
        self._capture_library_prompt_block_state(event.state)

    @on(Checkbox.Changed, "#library-prompt-recipe-starter")
    def handle_library_prompt_recipe_starter_changed(
        self, event: Checkbox.Changed
    ) -> None:
        """Persist the explicit Recipe starter-content choice for this draft."""
        if self._library_prompts_mutation_in_flight:
            return
        self._library_prompt_include_starter_content = bool(event.value)

    @on(Button.Pressed, "#library-prompt-convert")
    def handle_library_prompt_convert(self, event: Button.Pressed) -> None:
        """Convert valid compatibility lanes into a detached Prompt draft."""
        event.stop()
        if self._library_prompts_mutation_in_flight:
            return
        editor_state = self._current_library_prompt_editor_state()
        if not editor_state.can_convert_as_new:
            return
        converted = build_prompt_editor_state(
            {
                "artifact_type": "prompt",
                "system_prompt": editor_state.compiled_system_preview,
                "user_prompt": editor_state.compiled_user_preview,
            },
            capabilities=self._library_prompt_capabilities,
        ).block_editor_state
        if converted is None:
            return
        _draft, artifact_fields, converted = prepare_prompt_artifact_save(
            converted,
            artifact_type="prompt",
            include_recipe_starter_content=True,
            request_fields={},
        )
        detail = (
            dict(self._library_prompt_detail)
            if isinstance(self._library_prompt_detail, Mapping)
            else {}
        )
        detail.update(artifact_fields)
        self._detach_library_prompt_working_copy(detail)
        self._library_prompt_block_state = converted
        self._library_prompt_detached_structured = True
        self._library_prompt_dirty = True
        self._library_prompt_status = (
            "Compatibility text converted to an unsaved Prompt copy."
        )
        self._library_prompt_editor_armed = False
        if self.is_mounted:
            self.refresh(recompose=True)
            self.call_after_refresh(self._arm_library_prompt_editor)

    def on_prompt_block_editor_save_as_prompt_requested(
        self, event: PromptBlockEditor.SaveAsPromptRequested
    ) -> None:
        """Save the child working copy as a detached Prompt record."""
        event.stop()
        if self._library_prompts_mutation_in_flight:
            return
        self._library_prompt_block_state = event.state
        self.run_worker(
            self._await_library_prompt_save_call(
                self._save_library_prompt(
                    target_artifact_type="prompt", save_as_new=True
                )
            ),
            exclusive=True,
            group="library_prompt_save",
        )

    def on_prompt_block_editor_save_as_recipe_requested(
        self, event: PromptBlockEditor.SaveAsRecipeRequested
    ) -> None:
        """Save the child working copy as a detached Recipe record."""
        event.stop()
        if self._library_prompts_mutation_in_flight:
            return
        self._library_prompt_block_state = event.state
        self.run_worker(
            self._await_library_prompt_save_call(
                self._save_library_prompt(
                    target_artifact_type="recipe", save_as_new=True
                )
            ),
            exclusive=True,
            group="library_prompt_save",
        )

    def on_prompt_block_editor_update_original_requested(
        self, event: PromptBlockEditor.UpdateOriginalRequested
    ) -> None:
        """Conditionally update the captured Prompt/Recipe source version."""
        event.stop()
        if self._library_prompts_mutation_in_flight:
            return
        self._library_prompt_block_state = event.state
        self.run_worker(
            self._await_library_prompt_save_call(
                self._save_library_prompt(
                    target_artifact_type=event.state.artifact_type,
                    save_as_new=False,
                )
            ),
            exclusive=True,
            group="library_prompt_save",
        )

    async def on_prompt_block_editor_back_requested(
        self, event: PromptBlockEditor.BackRequested
    ) -> None:
        """Use the Library editor's existing dirty-aware back behavior."""
        event.stop()
        if self._library_prompts_mutation_in_flight:
            return
        if not await self._flush_library_prompt_save():
            return
        self._reset_library_prompt_editor_state()
        self._request_library_prompts_browse(
            self._library_prompt_browse_controller.mutation_refresh_scope,
            focus_identity=None,
        )
        self._refresh_local_source_snapshot()

    def _read_library_prompt_editor_fields(
        self,
    ) -> tuple[str, str, str, str, str, str] | None:
        """Read the prompt editor's current (possibly unsaved) field values.

        Returns:
            ``(name, author, details, system_prompt, user_prompt,
            keywords_text)`` read from the live widgets, or ``None`` if the
            editor isn't mounted.
        """
        try:
            name = self.query_one("#library-prompt-name", Input).value
            author = self.query_one("#library-prompt-author", Input).value
            details = self.query_one("#library-prompt-details", Input).value
            keywords_text = self.query_one("#library-prompt-keywords", Input).value
        except (NoMatches, QueryError):
            return None
        block_state = self._library_prompt_block_state
        if block_state is not None:
            system_prompt = block_state.compiled_system
            user_prompt = block_state.compiled_user
        else:
            # TASK-19602: the live legacy-lane TextAreas are the working
            # copy the user sees and edits, so non-empty live text outranks
            # the persisted detail. Structured/foreign artifacts mount
            # those lanes EMPTY (their truth is the STRUCTURE section) --
            # empty live lanes fall back to the detail's compatibility
            # text rather than blanking the copy/export.
            try:
                live_system = self.query_one("#library-prompt-system", TextArea).text
                live_user = self.query_one("#library-prompt-user", TextArea).text
            except (NoMatches, QueryError):
                live_system = live_user = None
            if live_system or live_user:
                system_prompt = live_system or ""
                user_prompt = live_user or ""
            elif isinstance(self._library_prompt_detail, Mapping):
                editor_state = build_prompt_editor_state(self._library_prompt_detail)
                system_prompt = editor_state.system_prompt
                user_prompt = editor_state.user_prompt
            else:
                return None
        return name, author, details, system_prompt, user_prompt, keywords_text

    def _library_prompt_text_fields_match_state(self) -> bool:
        """Return whether mounted prompt fields equal their backing detail."""
        detail = self._library_prompt_detail
        if not isinstance(detail, Mapping):
            return False
        try:
            name = self.query_one("#library-prompt-name", Input).value
            author = self.query_one("#library-prompt-author", Input).value
            details = self.query_one("#library-prompt-details", Input).value
            keywords = self.query_one("#library-prompt-keywords", Input).value
            basic_system = self.query_one("#library-prompt-system", TextArea).text
            basic_user = self.query_one("#library-prompt-user", TextArea).text
        except (NoMatches, QueryError):
            return False
        state = build_prompt_editor_state(detail)
        block_state = self._library_prompt_block_state
        if block_state is None:
            expected_system = ""
            expected_user = ""
        else:
            system_lane, user_lane = block_state.definition.lanes
            expected_system = (
                system_lane.blocks[0].content if len(system_lane.blocks) == 1 else ""
            )
            expected_user = (
                user_lane.blocks[0].content if len(user_lane.blocks) == 1 else ""
            )
        return (
            name,
            author,
            details,
            basic_system,
            basic_user,
            keywords,
        ) == (
            state.name,
            state.author,
            state.details,
            expected_system,
            expected_user,
            state.keywords_csv,
        )

    def _update_library_prompt_status_static(self, text: str) -> None:
        """Targeted update of ``#library-prompt-save-status``, no recompose.

        Args:
            text: The status copy to show (``""`` clears it).
        """
        self._library_prompt_status = text
        try:
            status_static = self.query_one("#library-prompt-save-status", Static)
        except (NoMatches, QueryError):
            return
        status_static.update(text)

    async def _sync_library_prompt_open_existing_button(self, *, show: bool) -> None:
        """Targeted mount/removal of ``#library-prompt-open-existing`` (Task
        8b D3), no recompose.

        ``_update_library_prompt_status_static`` (its sibling, called
        alongside this everywhere a save outcome is classified) never
        recomposes the editor either -- doing so here would rebuild the
        fields from the stale ``_library_prompt_detail`` mapping (the
        just-rejected name-in-use edit was never written there), silently
        discarding the user's in-progress text. Mirrors
        ``_refresh_collections_panel_action_state_widgets``'s targeted
        mount/remove pattern instead.

        Args:
            show: Whether the button should be present.
        """
        existing = list(self.query("#library-prompt-open-existing"))
        if show and not existing:
            try:
                status_static = self.query_one("#library-prompt-save-status", Static)
            except (NoMatches, QueryError):
                return
            parent = status_static.parent
            if parent is None:
                return
            await parent.mount(
                Button(
                    "Open existing",
                    id="library-prompt-open-existing",
                    classes="library-canvas-action",
                    compact=True,
                ),
                after=status_static,
            )
        elif not show and existing:
            for button in existing:
                await button.remove()

    async def _apply_library_prompt_save_outcome(
        self, outcome: str, *, name: str = ""
    ) -> None:
        """Set the save-status text for a classified outcome AND sync the
        D3 Open-existing affordance to match it, together (no recompose --
        see ``_sync_library_prompt_open_existing_button``'s docstring).

        Args:
            outcome: A ``classify_prompt_save_error`` return value.
            name: The attempted name that produced this outcome. Only
                meaningful (and only stashed) when ``outcome ==
                "name-in-use"`` -- captured here, at the moment the status
                is set, rather than re-derived later from the live Name
                field by ``_open_library_prompt_colliding_with_current_name``,
                which can have drifted if the user keeps typing after a
                failed Save without re-saving (Task 8b Fix wave 1 Minor).
        """
        self._library_prompt_name_in_use = name if outcome == "name-in-use" else ""
        self._update_library_prompt_status_static(
            LIBRARY_PROMPT_SAVE_STATUS_COPY.get(
                outcome, LIBRARY_PROMPT_SAVE_STATUS_COPY["error"]
            )
        )
        await self._sync_library_prompt_open_existing_button(
            show=outcome == "name-in-use"
        )

    def _update_library_prompt_meta_static(self) -> None:
        """Targeted update of ``#library-prompt-meta``, no recompose.

        Re-derives the meta line from ``_library_prompt_detail`` (the
        just-patched, post-save mirror) via the same pure
        ``prompt_editor_meta_line`` helper the editor's initial render
        uses, so a successful save's version bump -- or (Task 8c U6) a
        dirty-flag flip -- shows up without remounting the ``Input``/
        ``TextArea`` fields (which would re-arm-race the editor and risk
        the mount-time ``Changed`` event being mistaken for a fresh edit;
        see ``_mark_library_prompt_dirty``, this method's other caller).
        """
        if not isinstance(self._library_prompt_detail, Mapping):
            return
        try:
            meta_static = self.query_one("#library-prompt-meta", Static)
        except (NoMatches, QueryError):
            return
        meta_static.update(
            prompt_editor_meta_line(
                build_prompt_editor_state(self._library_prompt_detail),
                dirty=self._library_prompt_dirty,
            )
        )

    def _sync_library_prompt_save_action_widgets(self) -> None:
        """Patch save/update action truth after identity or version changes."""
        can_update = self._library_prompt_can_update_original()
        try:
            block_editor = self.query_one(
                "#library-prompt-block-editor", PromptBlockEditor
            )
        except (NoMatches, QueryError):
            block_editor = None
        if block_editor is not None:
            block_editor.set_update_original_available(can_update)

        try:
            outer_save = self.query_one("#library-prompt-save", Button)
        except (NoMatches, QueryError):
            return
        if self._selected_prompt_id is None:
            artifact_type = (
                self._library_prompt_block_state.artifact_type
                if self._library_prompt_block_state is not None
                else "prompt"
            )
            outer_save.label = f"Save {artifact_type}"
            outer_save.disabled = False
        else:
            outer_save.label = "Save changes"
            outer_save.disabled = not can_update
        try:
            canvas = self.query_one("#library-prompt-work-pane", LibraryPromptWorkPane)
            canvas.can_update_original = can_update
            canvas.sync_lifecycle_actions(
                dirty=self._library_prompt_dirty,
                conflict=self._library_prompt_conflict_snapshot is not None,
                mutation_in_flight=self._library_prompts_mutation_in_flight,
                write_in_flight=self._library_prompt_write_worker_is_active(),
            )
        except (NoMatches, QueryError):
            return

    def _set_library_prompt_discard_enabled(
        self, enabled: bool, *, write_in_flight: bool | None = None
    ) -> None:
        """Patch the Prompt Discard action without remounting live fields."""
        if write_in_flight is None:
            write_in_flight = self._library_prompt_write_worker_is_active()
        busy = self._library_prompts_mutation_in_flight or write_in_flight
        try:
            canvas = self.query_one("#library-prompt-work-pane", LibraryPromptWorkPane)
            canvas.sync_lifecycle_actions(
                dirty=enabled,
                conflict=self._library_prompt_conflict_snapshot is not None,
                mutation_in_flight=self._library_prompts_mutation_in_flight,
                write_in_flight=write_in_flight,
            )
        except (NoMatches, QueryError):
            pass
        for button in self.query("#library-prompt-discard"):
            if isinstance(button, Button):
                button.disabled = busy or not enabled
                button.tooltip = (
                    PROMPT_DISCARD_TOOLTIP_BUSY
                    if busy
                    else (
                        PROMPT_DISCARD_TOOLTIP_DIRTY
                        if enabled
                        else PROMPT_DISCARD_TOOLTIP_CLEAN
                    )
                )

    @on(Button.Pressed, "#library-prompt-save")
    def handle_library_prompt_save(self, event: Button.Pressed) -> None:
        """Explicitly save the open prompt, bypassing no debounce (there is
        none -- the prompt editor never autosaves).

        Args:
            event: Button press event emitted by the editor's "Save" action.
        """
        event.stop()
        if self._library_prompts_mutation_in_flight:
            return
        self.run_worker(
            self._await_library_prompt_save_call(self._save_library_prompt()),
            exclusive=True,
            group="library_prompt_save",
        )

    async def _save_library_prompt(
        self,
        *,
        target_artifact_type: ArtifactType | None = None,
        save_as_new: bool = False,
    ) -> None:
        """Save the open Library prompt's current editor text.

        The prompts DB's update seam (``update_prompt_by_id``, reached via
        ``PromptScopeService.save_prompt``) has no caller-supplied
        expected-version parameter of its own -- it always re-derives the
        version to bump from a fresh read inside its own transaction, so it
        cannot detect "this editor's cached version is stale" by itself.
        This method does that staleness check itself, via a fresh
        ``get_prompt`` read, BEFORE attempting the real write.

        Likewise, a rename to another prompt's name needs to distinguish
        "that name belongs to an active prompt" (name-in-use) from "that
        name belongs to a soft-deleted prompt" (soft-deleted-name) --
        outcomes the real ``update_prompt_by_id`` cannot cleanly
        distinguish either (both ultimately surface as the same
        ``ConflictError``/wrapped-``DatabaseError`` shape once the actual
        write is attempted, since ``Prompts.name`` is globally unique
        regardless of soft-delete state). So a rename is pre-checked by a
        name lookup too, before ever attempting the write.

        Every branch re-checks that the prompt this save was *for* is still
        selected (and the editor still showing) before mutating shared
        state, mirroring ``_save_library_note``'s stale-result guard.

        Task 8b D1: ``prompt_id is None`` (``_selected_prompt_id`` unset) is
        the create-flow sentinel -- set by
        ``_enter_library_prompt_create_editor``/``handle_library_prompt_duplicate``,
        never a stray/invalid state (a browsed prompt always has a real
        int id). ``save_prompt`` already routes ``prompt_identifier=None``
        to its own create path (``PromptScopeService.save_prompt``), so the
        actual write call below is unchanged between create and update --
        only the pre-checks (the version-staleness read has nothing to
        check for a not-yet-created prompt) and the post-write bookkeeping
        (adopting the freshly created id) differ, per ``is_create`` below.
        """
        if self._library_prompts_mutation_in_flight:
            return
        if self._library_prompts_view != "editor":
            return
        selected_prompt_id = self._selected_prompt_id
        prompt_id = None if save_as_new else selected_prompt_id
        is_create = prompt_id is None
        fields = self._read_library_prompt_editor_fields()
        if fields is None:
            return
        raw_name, raw_author, raw_details, raw_system, raw_user, raw_keywords_text = (
            fields
        )

        name = self._sanitize_media_field(raw_name, max_length=300)
        author = self._sanitize_media_field(raw_author, max_length=200)
        details = self._sanitize_note_content(
            raw_details, max_length=LIBRARY_PROMPT_TEXT_MAX_CHARS
        )
        keywords = self._library_note_keywords_from_input(raw_keywords_text)
        if not name:
            self._update_library_prompt_status_static(
                "Name is required; enter a Prompt name."
            )
            try:
                self.query_one("#library-prompt-name", Input).focus()
            except (NoMatches, QueryError):
                pass
            return

        block_state = getattr(self, "_library_prompt_block_state", None)
        prepared_state: PromptBlockEditorState | None = None
        if block_state is not None:
            artifact_type = target_artifact_type or block_state.artifact_type
            try:
                draft, save_fields, prepared_state = prepare_prompt_artifact_save(
                    block_state,
                    artifact_type=artifact_type,
                    include_recipe_starter_content=(
                        self._library_prompt_include_starter_content
                    ),
                    request_fields={
                        "name": name,
                        "author": author,
                        "details": details,
                        "keywords": keywords,
                        "expected_version": (
                            self._library_prompt_version if not is_create else None
                        ),
                    },
                )
                require_artifact_save_supported(
                    draft,
                    self._library_prompt_capabilities,
                    update_original=not is_create,
                    expected_version=(
                        self._library_prompt_version if not is_create else None
                    ),
                )
            except ValueError as exc:
                self._update_library_prompt_status_static(str(exc))
                if block_state.issues:
                    self._library_prompt_editor_mode = "advanced"
                    try:
                        work = self.query_one(
                            "#library-prompt-work-pane", LibraryPromptWorkPane
                        )
                        await work.set_editor_mode("advanced")
                        work.query_one(
                            "#library-prompt-block-editor", PromptBlockEditor
                        ).focus_first_error()
                    except (NoMatches, QueryError):
                        pass
                return
            system_prompt = draft.system_prompt
            user_prompt = draft.user_prompt
        else:
            save_fields = {
                "name": name,
                "author": author,
                "details": details,
                "system_prompt": self._sanitize_note_content(
                    raw_system, max_length=LIBRARY_PROMPT_TEXT_MAX_CHARS
                ),
                "user_prompt": self._sanitize_note_content(
                    raw_user, max_length=LIBRARY_PROMPT_TEXT_MAX_CHARS
                ),
                "keywords": keywords,
            }
            system_prompt = save_fields["system_prompt"]
            user_prompt = save_fields["user_prompt"]

        service = getattr(self.app_instance, "prompt_scope_service", None)
        get_prompt = getattr(service, "get_prompt", None)
        save_prompt = getattr(service, "save_prompt", None)
        if not callable(get_prompt) or not callable(save_prompt):
            return

        if name and name != self._library_prompt_original_name:
            try:
                candidate = await self._run_library_service_call(
                    get_prompt,
                    mode="local",
                    prompt_identifier=name,
                    include_deleted=True,
                    isolate_in_worker=True,
                )
            except Exception:
                candidate = None
            if (
                self._library_prompts_mutation_in_flight
                or selected_prompt_id != self._selected_prompt_id
                or self._library_prompts_view != "editor"
            ):
                return
            candidate_id = (
                candidate.get("local_id") if isinstance(candidate, Mapping) else None
            )
            if candidate_id is not None and candidate_id != prompt_id:
                if candidate.get("deleted"):
                    outcome = classify_prompt_save_error(
                        None, f"Prompt '{name}' exists but is soft-deleted.", None
                    )
                else:
                    outcome = classify_prompt_save_error(
                        None, f"Prompt '{name}' already exists.", None
                    )
                await self._apply_library_prompt_save_outcome(outcome, name=name)
                return

        if not is_create:
            # A not-yet-created prompt has no existing row to have gone
            # stale -- skip the pre-read entirely rather than calling
            # ``get_prompt(prompt_identifier=None)`` (which would only
            # raise, harmlessly swallowed by the ``except`` below, but is
            # wasted work with no real check to perform).
            try:
                fresh = await self._run_library_service_call(
                    get_prompt,
                    mode="local",
                    prompt_identifier=prompt_id,
                    include_deleted=True,
                    isolate_in_worker=True,
                )
            except Exception:
                fresh = None
            if (
                self._library_prompts_mutation_in_flight
                or selected_prompt_id != self._selected_prompt_id
                or self._library_prompts_view != "editor"
            ):
                return
            fresh_version = fresh.get("version") if isinstance(fresh, Mapping) else None
            if (
                fresh_version is not None
                and self._library_prompt_version is not None
                and fresh_version != self._library_prompt_version
            ):
                self._enter_library_prompt_conflict(
                    name=raw_name,
                    author=raw_author,
                    details=raw_details,
                    system_prompt=raw_system,
                    user_prompt=raw_user,
                    keywords_text=raw_keywords_text,
                )
                return

        try:
            result = await self._run_library_service_call(
                save_prompt,
                mode="local",
                prompt_identifier=prompt_id,
                **save_fields,
                isolate_in_worker=True,
            )
        except Exception as exc:
            logger.opt(exception=True).warning(
                f"Library prompt save failed for {prompt_id!r}."
            )
            if (
                self._library_prompts_mutation_in_flight
                or selected_prompt_id != self._selected_prompt_id
                or self._library_prompts_view != "editor"
            ):
                return
            outcome = classify_prompt_save_error(None, str(exc), exc)
            if outcome == "conflict":
                # A genuine race the pre-checks above could not see (e.g. a
                # second app instance or an external writer landing between
                # this save's pre-read and its real write) -- route into
                # the SAME conflict banner the pre-check staleness path
                # uses above, seeded from the same live (raw, unsanitized)
                # field values, rather than falling through to the generic
                # error status line.
                self._enter_library_prompt_conflict(
                    name=raw_name,
                    author=raw_author,
                    details=raw_details,
                    system_prompt=raw_system,
                    user_prompt=raw_user,
                    keywords_text=raw_keywords_text,
                )
                return
            await self._apply_library_prompt_save_outcome(outcome, name=name)
            return

        if (
            self._library_prompts_mutation_in_flight
            or selected_prompt_id != self._selected_prompt_id
            or self._library_prompts_view != "editor"
        ):
            return

        result_id = (
            result.get("local_id")
            if isinstance(result, Mapping)
            else (1 if result else None)
        )
        outcome = classify_prompt_save_error(result_id, "", None)
        if outcome != "ok":
            await self._apply_library_prompt_save_outcome(outcome)
            return

        if save_as_new and artifact_type == "recipe":
            # A Recipe save is an independent library artifact, never the
            # update identity of the active Prompt working copy. Keep the
            # source selection/version/block state and dirty semantics intact.
            self._refresh_local_source_snapshot()
            self._update_library_prompt_status_static("Recipe saved as a new artifact.")
            await self._sync_library_prompt_open_existing_button(show=False)
            return

        new_id = result_id if is_create else prompt_id
        self._selected_prompt_id = new_id
        self._library_prompt_loaded_id = new_id
        self._library_prompt_detail_selected_name = name
        self._library_prompt_detail_loading = False
        self._library_prompt_detail_error = ""
        self._library_prompt_detail_retryable = False
        version = result.get("version") if isinstance(result, Mapping) else None
        self._library_prompt_version = (
            version
            if version is not None
            else (1 if is_create else (self._library_prompt_version or 0) + 1)
        )
        patched_detail: dict[str, Any] = (
            dict(result)
            if is_create and isinstance(result, Mapping)
            else (
                dict(self._library_prompt_detail)
                if isinstance(self._library_prompt_detail, Mapping)
                else {}
            )
        )
        patched_detail["id"] = new_id
        patched_detail["name"] = name
        patched_detail["author"] = author
        patched_detail["details"] = details
        patched_detail["system_prompt"] = system_prompt
        patched_detail["user_prompt"] = user_prompt
        patched_detail["version"] = self._library_prompt_version
        if prepared_state is not None:
            patched_detail["artifact_type"] = prepared_state.artifact_type
            patched_detail["prompt_format"] = "structured"
            patched_detail["prompt_schema_version"] = 2
            patched_detail["prompt_definition"] = save_fields["prompt_definition"]
            self._library_prompt_block_state = prepared_state
        if isinstance(result, Mapping):
            if isinstance(result.get("uuid"), str):
                patched_detail["uuid"] = result["uuid"]
            if "keywords" in result:
                patched_detail["keywords"] = result["keywords"]
            if result.get("last_modified"):
                patched_detail["last_modified"] = result["last_modified"]
        elif keywords is not None:
            patched_detail["keywords"] = keywords
        history_was_open = bool(
            self._library_prompt_history_state is not None
            and self._library_prompt_history_state.is_open
        )
        if is_create:
            editor_was_armed = self._library_prompt_editor_armed
            self._adopt_library_prompt_persisted_detail(
                patched_detail,
                status=LIBRARY_PROMPT_SAVE_STATUS_COPY["ok"],
                open_history=history_was_open,
            )
            self._load_library_prompt_memberships()
            # This success patches the existing editor in place rather than
            # remounting it, so retain its prior dirty-tracking arm state.
            self._library_prompt_editor_armed = editor_was_armed
            self._sync_library_prompt_save_action_widgets()
            # Unlike a plain in-place field update (which defers the
            # broader snapshot refresh to when the editor is actually left
            # -- see the comment below), a brand-new prompt changes the
            # list's membership/count, so the Prompts rail badge and list
            # must pick up the new row now. Fire-and-forget, mirrors
            # ``_create_library_note``'s equivalent post-create refresh.
            self._refresh_local_source_snapshot()
        else:
            self._library_prompt_detail = patched_detail
            self._library_prompt_detached_structured = False
            self._library_prompt_original_name = name
            self._library_prompt_dirty = False
        self._set_library_prompt_discard_enabled(False)
        # Targeted updates only (no recompose): the fields already hold the
        # user's just-saved text, so nothing there needs to change -- only
        # the meta line's version and the status line need to reflect the
        # save. Mirrors ``_save_library_note``'s discipline (it "never
        # recomposes, so the TextArea/Input widget instances stay identical
        # across a save"): recomposing here would remount fresh Input/
        # TextArea widgets while the editor is still armed, and Textual's
        # spurious mount-time ``Changed`` event for a non-empty initial
        # value would immediately re-mark the just-saved prompt dirty.
        self._update_library_prompt_meta_static()
        # A prior attempt within this same editor session may have left the
        # D3 Open-existing button mounted (e.g. a name-in-use retry that
        # then succeeded with a different name) -- clear it alongside the
        # "Saved." status, same combined helper the failure branches above
        # use.
        await self._apply_library_prompt_save_outcome("ok")
        if not is_create:
            self._initialize_library_prompt_history(
                patched_detail, open_history=history_was_open
            )
        # The broader local-source snapshot (rail badge / list ordering) is
        # deliberately NOT refreshed here -- it would recompose the whole
        # canvas (see the comment above) while this editor is still open
        # and armed. It refreshes when the editor is actually left instead
        # (``handle_library_prompt_back``, delete), the same point notes'
        # save flow defers its own snapshot patch to.

    def _enter_library_prompt_conflict(
        self,
        *,
        name: str,
        author: str,
        details: str,
        system_prompt: str,
        user_prompt: str,
        keywords_text: str,
    ) -> None:
        """Recompose into the save-conflict banner, seeded from live text.

        Args:
            name: The editor's live Name field value at Save time.
            author: The editor's live Author field value at Save time.
            details: The editor's live Details field value at Save time.
            system_prompt: The editor's live System prompt field value.
            user_prompt: The editor's live User prompt field value.
            keywords_text: The editor's live Keywords field value.
        """
        self._library_prompt_conflict_snapshot = dataclasses.replace(
            self._current_library_prompt_editor_state(),
            prompt_id=self._selected_prompt_id,
            name=name,
            author=author,
            details=details,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            keywords_csv=keywords_text,
            version=self._library_prompt_version,
            created="",
            modified=(
                self._library_prompt_detail.get("last_modified", "")
                if isinstance(self._library_prompt_detail, Mapping)
                else ""
            ),
        )
        self._library_prompt_status = ""
        self._library_prompt_editor_armed = False
        if self.is_mounted:
            self.refresh(recompose=True)
            self.call_after_refresh(self._arm_library_prompt_editor)

    def _notify_prompt_dirty_veto(self) -> None:
        """Explain a dirty Prompt navigation veto without exposing content."""
        notify = getattr(self.app_instance, "notify", None)
        if callable(notify):
            notify(LIBRARY_PROMPT_DIRTY_VETO_COPY, severity="warning")

    def _apply_library_prompt_working_copy(
        self,
        *,
        state: PromptBlockEditorState,
        system_prompt: str | None,
        user_prompt: str | None,
    ) -> None:
        """Stage a Prompt, or detach a Recipe into an unsaved Prompt copy."""
        if self._library_prompts_mutation_in_flight:
            return
        notify = getattr(self.app_instance, "notify", None)
        if state.artifact_type == "recipe":
            prompt_state = set_artifact_type(state, "prompt")
            _draft, artifact_fields, prompt_state = prepare_prompt_artifact_save(
                prompt_state,
                artifact_type="prompt",
                include_recipe_starter_content=True,
                request_fields={},
            )
            self._library_prompt_block_state = prompt_state
            self._library_prompt_detached_structured = True
            self._library_prompt_dirty = True
            self._library_prompt_status = (
                "Recipe opened as an unsaved Prompt copy — review and save it "
                "before use."
            )
            detail = (
                dict(self._library_prompt_detail)
                if isinstance(self._library_prompt_detail, Mapping)
                else {}
            )
            detail.update(artifact_fields)
            self._detach_library_prompt_working_copy(detail)
            self._library_prompt_editor_armed = False
            if self.is_mounted:
                self.refresh(recompose=True)
                self.call_after_refresh(self._arm_library_prompt_editor)
            if callable(notify):
                notify(
                    "Recipe converted to an unsaved Prompt copy; nothing was applied.",
                    severity="information",
                )
            return

        self._stage_library_prompt_for_console(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    @on(Button.Pressed, "#library-prompt-discard")
    def handle_library_prompt_discard(self, event: Button.Pressed) -> None:
        """Leave the Prompt editor without persisting its working copy.

        Args:
            event: Button press event emitted by the editor's Discard action.
        """
        event.stop()
        if (
            self._library_prompts_mutation_in_flight
            or self._library_prompt_write_worker_is_active()
            or not self._library_prompt_dirty
        ):
            return
        prompt_id = self._selected_prompt_id
        focus_identity = (
            f"library-prompt-row-{prompt_id}"
            if type(prompt_id) is int and prompt_id > 0
            else None
        )
        self._reset_library_prompt_editor_state()
        self._request_library_prompts_browse(
            self._library_prompt_browse_controller.mutation_refresh_scope,
            focus_identity=focus_identity,
        )
        self._refresh_local_source_snapshot()
        self._arm_library_list_entry_focus()

    @on(Button.Pressed, "#library-prompt-back")
    async def handle_library_prompt_back(self, event: Button.Pressed) -> None:
        """Return the Library prompts canvas from the editor to its list view.

        Args:
            event: Button press event emitted by the "‹ Back to list" action.
        """
        event.stop()
        await self._exit_library_prompt_editor_guarded()

    async def _exit_library_prompt_editor_guarded(self) -> bool:
        """Shared Back exit: veto while dirty, else reset to list.

        Shared by the "‹ Back to list" button and the editor's Escape
        binding (``action_library_prompt_editor_back``, task-2856 AC2) --
        one seam, matching the skill/note editors' guarded-exit idiom.
        Vetoed while dirty (see ``_flush_library_prompt_save``) so Back
        never silently discards an unsaved edit.

        Also kicks the full local-source snapshot refetch (the notes Back
        handler's same pattern): any save made during this editor visit
        only patched ``_library_prompt_detail`` in place (see
        ``_save_library_prompt``, which deliberately skips a broader
        snapshot refresh while the editor is still open), so the list's
        ordering/rail badge are only guaranteed fresh once this refetch
        lands -- safe to fire now since the editor is no longer mounted to
        be spuriously re-dirtied by the recompose it eventually triggers.

        Returns:
            ``True`` when the editor was exited; ``False`` on a dirty veto.
        """
        if self._library_prompts_mutation_in_flight:
            return False
        if not await self._flush_library_prompt_save():
            return False
        self._reset_library_prompt_editor_state()
        self._request_library_prompts_browse(
            self._library_prompt_browse_controller.mutation_refresh_scope,
            focus_identity=None,
        )
        self._refresh_local_source_snapshot()
        # task-2856 AC1: every "back to list" exit re-focuses the list's
        # first row so Up/Down/Enter work immediately.
        self._arm_library_list_entry_focus()
        return True

    async def action_library_prompt_editor_back(self) -> None:
        """Escape: leave the prompt editor for its list, honoring the dirty
        guard (task-2856 AC2).

        ``check_action`` gates this to ``_library_prompt_editor_active()``,
        so it only ever fires while the prompt editor genuinely owns the
        Prompts canvas.
        """
        await self._exit_library_prompt_editor_guarded()

    @on(Button.Pressed, "#library-prompt-export")
    async def handle_library_prompt_export(self, event: Button.Pressed) -> None:
        """Export the open prompt as Markdown via a ``FileSave`` dialog.

        Args:
            event: Button press event emitted by the editor's "Export…" action.
        """
        event.stop()
        if self._library_prompts_mutation_in_flight:
            return
        await self._export_library_prompt()

    async def _export_library_prompt(self) -> None:
        """Push the Export dialog for the open Library prompt.

        Mirrors ``_export_library_note`` exactly (see that method's
        docstring for the full ``FileSave`` constructor-shape rationale):
        a ``FileSave`` prompt pre-filled with a sanitized default filename,
        whose callback renders and writes the export once a path is
        chosen. Reads the *live* editor widgets (via
        ``_read_library_prompt_editor_fields``), never the DB/detail
        mapping, so unlike Save there is nothing to flush first -- the
        export reflects exactly what's on screen, including unsaved edits.
        """
        if self._library_prompts_mutation_in_flight:
            return
        if self._library_prompts_view != "editor" or not self._selected_prompt_id:
            return
        if self._library_prompt_action_artifact_type() is None:
            self._notify_library_prompt_unsupported_artifact_type()
            return
        fields = self._read_library_prompt_editor_fields()
        if fields is None:
            return
        name, author, details, system_prompt, user_prompt, keywords_text = fields
        prompt_id = self._selected_prompt_id
        artifact_fields = self._library_prompt_markdown_artifact_fields(
            action="exporting"
        )
        if artifact_fields is None:
            return
        # Same inline sanitize-for-filename technique as
        # ``_export_library_note``'s ``safe_title`` -- alnum/space/-/_ only,
        # falling back to a generic name when that leaves nothing (e.g. a
        # prompt named entirely in punctuation/emoji).
        safe_name = (
            "".join(
                char
                for char in (name.strip() or "prompt")
                if char.isalnum() or char in (" ", "-", "_")
            ).rstrip()
            or "prompt"
        )
        default_filename = f"{safe_name}.md"
        await self.app.push_screen(
            FileSave(
                location=str(Path.home()),
                title="Export Prompt as Markdown",
                default_file=default_filename,
            ),
            callback=lambda path: self.call_after_refresh(
                self._write_library_prompt_export_file,
                path,
                name,
                author,
                details,
                system_prompt,
                user_prompt,
                keywords_text,
                prompt_id,
                artifact_fields,
            ),
        )

    @on(Button.Pressed, "#library-prompt-copy")
    def handle_library_prompt_copy(self, event: Button.Pressed) -> None:
        """Copy the live Prompt/Recipe working copy as canonical Markdown.

        Args:
            event: Button press emitted by the Prompt editor's Copy action.
        """
        event.stop()
        if self._library_prompts_mutation_in_flight:
            return
        if self._library_prompts_view != "editor":
            return
        if self._library_prompt_action_artifact_type() is None:
            self._notify_library_prompt_unsupported_artifact_type()
            return
        fields = self._read_library_prompt_editor_fields()
        if fields is None:
            return
        name, author, details, system_prompt, user_prompt, keywords_text = fields
        detail: dict[str, Any] = {
            "name": name,
            "author": author,
            "details": details,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "keywords": self._library_note_keywords_from_input(keywords_text) or [],
        }
        artifact_fields = self._library_prompt_markdown_artifact_fields(
            action="copying"
        )
        if artifact_fields is None:
            return
        detail.update(artifact_fields)

        notify = getattr(self.app_instance, "notify", None)
        copy_to_clipboard = getattr(self.app_instance, "copy_to_clipboard", None)
        if not callable(copy_to_clipboard):
            if callable(notify):
                notify(
                    "Clipboard copy is unavailable in this runtime.", severity="warning"
                )
            return
        try:
            copy_to_clipboard(render_prompt_markdown(detail))
        except Exception as exc:
            if callable(notify):
                notify(f"Error copying prompt: {type(exc).__name__}", severity="error")
            return
        if callable(notify):
            notify("Prompt copied to clipboard as markdown!", severity="information")

    def _library_prompt_artifact_fields(self) -> dict[str, Any]:
        """Return export/copy metadata for the live Prompt working copy.

        Supported stored v2 and explicitly detached structured block state is
        prepared canonically. Read-only compatibility records retain their raw
        metadata rather than being silently flattened to their compiled lanes.
        Legacy lane records stay in the established plain-Markdown form.
        """
        editor_state = self._current_library_prompt_editor_state()
        block_state = self._library_prompt_block_state
        if block_state is not None and (
            editor_state.definition_state == "supported_v2"
            or self._library_prompt_detached_structured
        ):
            _draft, artifact_payload, _prepared = prepare_prompt_artifact_save(
                block_state,
                artifact_type=block_state.artifact_type,
                include_recipe_starter_content=True,
                request_fields={},
            )
            return {
                key: artifact_payload[key]
                for key in (
                    "artifact_type",
                    "prompt_format",
                    "prompt_schema_version",
                    "prompt_definition",
                    "system_prompt",
                    "user_prompt",
                )
                if key in artifact_payload
            }
        if editor_state.definition_state == "legacy":
            return {}
        detail = self._library_prompt_detail
        if not isinstance(detail, Mapping):
            return {}
        return {
            key: detail[key]
            for key in (
                "artifact_type",
                "prompt_format",
                "prompt_schema_version",
                "prompt_definition",
            )
            if key in detail
        }

    def _library_prompt_markdown_artifact_fields(
        self, *, action: Literal["copying", "exporting"]
    ) -> dict[str, Any] | None:
        """Admit only working copies the Markdown grammar can preserve."""
        if self._library_prompt_legacy_recipe_requires_conversion():
            self._notify_library_prompt_legacy_recipe_requires_conversion()
            return None
        detail = self._library_prompt_detail
        if isinstance(detail, Mapping):
            has_modern_metadata = (
                detail.get("prompt_schema_version") is not None
                or detail.get("prompt_definition") is not None
            )
            if has_modern_metadata and detail.get("prompt_format") != "structured":
                self._notify_library_prompt_unrepresentable_markdown()
                return None
        try:
            artifact_fields = self._library_prompt_artifact_fields()
        except ValueError:
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify(
                    f"Fix block validation errors before {action}; "
                    "the structured artifact was not downgraded.",
                    severity="warning",
                )
            return None
        if not artifact_fields:
            return {}

        artifact_type = self._library_prompt_action_artifact_type()
        definition = deserialize_definition(artifact_fields.get("prompt_definition"))
        outer_schema = artifact_fields.get("prompt_schema_version")
        definition_schema = (
            definition.get("schema_version") if definition is not None else None
        )
        definition_kind = definition.get("kind") if definition is not None else None
        definition_kind_owner: ArtifactType | None = None
        if isinstance(definition_kind, str):
            if definition_kind.endswith("_prompt"):
                definition_kind_owner = "prompt"
            elif definition_kind.endswith("_recipe"):
                definition_kind_owner = "recipe"
        if (
            definition is not None
            and definition.get("definition_kind") == "single_text_recipe"
        ):
            definition_kind_owner = "recipe"

        representable = (
            artifact_type in {"prompt", "recipe"}
            and artifact_fields.get("artifact_type") == artifact_type
            and artifact_fields.get("prompt_format") == "structured"
            and definition is not None
            and type(outer_schema) is int
            and type(definition_schema) is int
            and outer_schema == definition_schema
            and (
                definition_kind_owner is None or definition_kind_owner == artifact_type
            )
        )
        if representable:
            return artifact_fields

        self._notify_library_prompt_unrepresentable_markdown()
        return None

    def _notify_library_prompt_unrepresentable_markdown(self) -> None:
        """Report metadata loss without exposing artifact content or errors."""
        notify = getattr(self.app_instance, "notify", None)
        if callable(notify):
            notify(
                "This structured artifact cannot be represented as Markdown "
                "without losing metadata. Use Convert and save as a new Prompt "
                "first.",
                severity="warning",
            )

    def _library_prompt_legacy_recipe_requires_conversion(self) -> bool:
        """Return whether a legacy Recipe would lose its type in this action."""
        if self._library_prompt_detached_structured:
            return False
        editor_state = self._current_library_prompt_editor_state()
        return (
            editor_state.definition_state == "legacy"
            and editor_state.artifact_type == "recipe"
        )

    def _notify_library_prompt_legacy_recipe_requires_conversion(self) -> None:
        """Direct a legacy Recipe to the explicit type-preserving conversion."""
        notify = getattr(self.app_instance, "notify", None)
        if callable(notify):
            notify(
                "This Recipe cannot use this action without losing its type. "
                "Convert and save as a new Prompt first.",
                severity="warning",
            )

    def _library_prompt_action_artifact_type(self) -> ArtifactType | None:
        """Return a supported explicit type, preserving missing legacy type as Prompt."""
        if self._library_prompt_detached_structured:
            block_state = self._library_prompt_block_state
            if block_state is None or block_state.artifact_type not in {
                "prompt",
                "recipe",
            }:
                return None
            return block_state.artifact_type
        detail = self._library_prompt_detail
        if isinstance(detail, Mapping) and "artifact_type" in detail:
            raw_type = detail["artifact_type"]
            if raw_type == "prompt":
                return "prompt"
            if raw_type == "recipe":
                return "recipe"
            return None
        artifact_type = self._current_library_prompt_editor_state().artifact_type
        if artifact_type not in {"prompt", "recipe"}:
            return None
        return artifact_type

    def _notify_library_prompt_unsupported_artifact_type(self) -> None:
        """Report an unsupported artifact without exposing its contents."""
        notify = getattr(self.app_instance, "notify", None)
        if callable(notify):
            notify("This artifact type is unsupported.", severity="warning")

    @on(Button.Pressed, "#library-prompt-duplicate")
    def handle_library_prompt_duplicate(self, event: Button.Pressed) -> None:
        """Open the editor on a NEW blank-id record pre-filled from the
        current prompt's fields (Task 8b U3).

        Reads the *live* editor widgets (never the DB/detail mapping) --
        same rationale as ``_export_library_prompt``: the duplicate should
        carry whatever is currently on screen, including unsaved edits, not
        revert to the last-saved text. The name becomes ``"<name> (copy)"``;
        the new record is dirty/unsaved by construction (unlike the D1
        blank-create entry, which starts clean). Reuses the D1 create path
        on Save (``_selected_prompt_id`` is ``None``, exactly like
        ``_enter_library_prompt_create_editor``'s sentinel).

        Args:
            event: Button press event emitted by the editor's "Duplicate" action.
        """
        event.stop()
        if self._library_prompts_mutation_in_flight:
            return
        if self._library_prompts_view != "editor":
            return
        if self._library_prompt_action_artifact_type() is None:
            self._notify_library_prompt_unsupported_artifact_type()
            return
        if self._library_prompt_legacy_recipe_requires_conversion():
            self._notify_library_prompt_legacy_recipe_requires_conversion()
            return
        fields = self._read_library_prompt_editor_fields()
        if fields is None:
            return
        name, author, details, system_prompt, user_prompt, keywords_text = fields
        editor_state = self._current_library_prompt_editor_state()
        block_state = self._library_prompt_block_state
        if (
            not self._library_prompt_detached_structured
            and editor_state.definition_state not in {"legacy", "supported_v2"}
        ):
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify(
                    "Convert this compatibility artifact and save it as a new "
                    "Prompt before duplicating.",
                    severity="warning",
                )
            return
        detached_structured = block_state is not None and (
            editor_state.definition_state == "supported_v2"
            or self._library_prompt_detached_structured
        )
        artifact_fields: dict[str, Any] = {}
        if detached_structured and block_state is not None:
            try:
                _draft, artifact_fields, block_state = prepare_prompt_artifact_save(
                    block_state,
                    artifact_type=block_state.artifact_type,
                    include_recipe_starter_content=True,
                    request_fields={},
                )
            except ValueError:
                # Keep the invalid live block state so the duplicate remains
                # editable; Copy/Save will surface its existing validation.
                artifact_fields = {"artifact_type": block_state.artifact_type}
        detached_detail = {
            "name": f"{name} (copy)",
            "author": author,
            "details": details,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            # A raw CSV string is a valid `keywords` input for
            # `build_prompt_editor_state`/`_csv_from_keywords` (it passes a
            # `str` through verbatim after stripping) -- preserves the live
            # Keywords field's exact text rather than round-tripping it
            # through a list.
            "keywords": keywords_text,
            **artifact_fields,
        }
        self._detach_library_prompt_working_copy(detached_detail)
        self._library_prompt_block_state = block_state
        self._library_prompt_detached_structured = detached_structured
        self._library_prompt_status = ""
        self._library_prompt_dirty = True
        self._library_prompt_editor_armed = False
        if self.is_mounted:
            self.refresh(recompose=True)
            self.call_after_refresh(self._arm_library_prompt_editor)

    @on(Button.Pressed, "#library-prompt-delete")
    def handle_library_prompt_delete(self, event: Button.Pressed) -> None:
        """Open a confirmation for the current Prompt/Recipe identity.

        Args:
            event: Button press emitted by the Prompt editor's Delete action.
        """
        event.stop()
        if self._library_prompts_view != "editor" or not self._selected_prompt_id:
            return
        if self._library_prompts_mutation_in_flight:
            return
        fields = self._read_library_prompt_editor_fields()
        artifact_type = self._library_prompt_action_artifact_type()
        version = self._library_prompt_version
        if (
            fields is None
            or artifact_type is None
            or not isinstance(version, int)
            or isinstance(version, bool)
        ):
            if artifact_type is None:
                self._notify_library_prompt_unsupported_artifact_type()
            return
        entry = PromptSelectionEntry(
            local_id=self._selected_prompt_id,
            expected_version=version,
            title=fields[0],
            artifact_type=artifact_type,
        )
        self._open_library_prompt_delete_confirmation(
            (entry,),
            dirty=self._library_prompt_dirty,
            selection_generation=None,
            editor_prompt_id=self._selected_prompt_id,
        )

    @on(Button.Pressed, "#library-prompts-delete-selected")
    def handle_library_prompts_delete_selected(self, event: Button.Pressed) -> None:
        """Confirm deletion of the immutable selected Prompt snapshot."""
        event.stop()
        if (
            self._library_prompts_mutation_in_flight
            or self._library_prompt_browse_controller.freshness == "stale"
            or self._library_selected_row_id != LIBRARY_ROW_BROWSE_PROMPTS
            or self._library_prompts_view != "list"
            or not self._library_prompt_select_mode
        ):
            return
        entries = self._library_prompt_selection.canonical_entries
        if not entries:
            return
        self._open_library_prompt_delete_confirmation(
            entries,
            dirty=False,
            selection_generation=self._library_prompt_selection.generation,
            editor_prompt_id=None,
        )

    def _open_library_prompt_delete_confirmation(
        self,
        entries: tuple[PromptSelectionEntry, ...],
        *,
        dirty: bool,
        selection_generation: int | None,
        editor_prompt_id: int | None,
    ) -> None:
        """Capture a strict snapshot and expose only an opaque modal token."""
        self._library_prompt_mutation_generation += 1
        fingerprint = str(self._library_prompt_mutation_generation)
        targets = tuple(
            PromptBatchTarget(entry.local_id, entry.expected_version)
            for entry in entries
        )
        self._library_prompt_delete_pending_fingerprint = fingerprint
        self._library_prompt_delete_pending_targets = targets
        self._library_prompt_delete_pending_entries = entries
        self._library_prompt_delete_pending_selection_generation = selection_generation
        self._library_prompt_delete_pending_editor_prompt_id = editor_prompt_id
        self.app.push_screen(
            PromptDeleteConfirmationModal(
                PromptDeleteRequest(
                    items=tuple(
                        PromptDeleteItem(entry.title, entry.artifact_type)
                        for entry in entries
                    ),
                    fingerprint=fingerprint,
                    dirty=dirty,
                )
            ),
            self._settle_library_prompt_delete,
        )

    def _library_prompt_delete_fingerprint(self) -> str | None:
        """Return the current opaque confirmation token, if any."""
        return self._library_prompt_delete_pending_fingerprint

    def _clear_library_prompt_delete_pending(self) -> None:
        """Discard confirmation-only authority without touching a receipt."""
        self._library_prompt_delete_pending_fingerprint = None
        self._library_prompt_delete_pending_targets = None
        self._library_prompt_delete_pending_entries = None
        self._library_prompt_delete_pending_selection_generation = None
        self._library_prompt_delete_pending_editor_prompt_id = None

    def _library_prompt_write_worker_is_active(self) -> bool:
        """Return whether an admitted Prompt writer has not settled yet.

        TASK-19602: the screen-owned worker manager is only reachable on a
        mounted screen (``self.workers`` resolves through the active-app
        context); headless callers (``_library_prompts_canvas_kwargs`` on
        an unmounted instance) still get the app-owned half of the scan.
        """
        managers = [self.app_instance.workers]
        try:
            managers.append(self.workers)
        except Exception:
            # An unmounted screen's worker walk surfaces NoActiveAppError
            # or its raw LookupError depending on entry point; neither
            # exists off the app tree.
            pass
        return any(
            worker.group in _LIBRARY_PROMPT_WRITE_WORKER_GROUPS
            and not worker.is_finished
            for manager in managers
            for worker in manager
        )

    async def _delete_library_prompts(
        self,
        targets: tuple[PromptBatchTarget, ...],
        *,
        selection_generation: int | None,
        editor_prompt_id: int | None,
        mutation_token: int,
        focus_identity: str | None = None,
    ) -> None:
        """Atomically delete one editor target or the selected batch."""
        selected_batch = selection_generation is not None
        committed = False
        try:
            service = getattr(self.app_instance, "prompt_scope_service", None)
            delete_prompts = getattr(service, "delete_prompts", None)
            if not callable(delete_prompts):
                if selected_batch:
                    self._notify_library_prompt_delete_failure(
                        "Bulk Prompt actions are unavailable."
                    )
                else:
                    self._update_library_prompt_status_static(
                        "Could not delete this prompt. Nothing was deleted."
                    )
                return
            try:
                result = await self._await_library_prompt_durable_call(
                    self._run_library_service_call(
                        delete_prompts,
                        mode="local",
                        targets=targets,
                        isolate_in_worker=True,
                    )
                )
            except PromptConflictError:
                if selected_batch:
                    self._notify_library_prompt_delete_failure(
                        "Selection changed; nothing was deleted. Clear all and "
                        "select the items again."
                    )
                else:
                    self._update_library_prompt_status_static(
                        "This prompt changed elsewhere — refresh and try again."
                    )
                return
            except Exception:
                if selected_batch:
                    self._notify_library_prompt_delete_failure(
                        "Could not delete the selected items. Nothing was deleted."
                    )
                else:
                    self._update_library_prompt_status_static(
                        "Could not delete this prompt. Nothing was deleted."
                    )
                return

            expected_ids = tuple(target.local_id for target in targets)
            if (
                type(result) is not PromptBatchDeleteResult
                or tuple(entry.local_id for entry in result.entries) != expected_ids
                or tuple(entry.tombstone_version for entry in result.entries)
                != tuple(target.expected_version + 1 for target in targets)
            ):
                if selected_batch:
                    self._notify_library_prompt_delete_failure(
                        "Could not delete the selected items. Nothing was deleted."
                    )
                else:
                    self._update_library_prompt_status_static(
                        "Could not delete this prompt. Nothing was deleted."
                    )
                return

            if not self._library_prompt_mutation_is_current(
                mutation_token,
                selection_generation=selection_generation,
                editor_prompt_id=editor_prompt_id,
            ):
                return
            self._library_prompt_delete_receipt = result
            self._library_prompt_mutation_status = ""
            self._local_source_counts["prompts"] = max(
                0, self._local_source_counts.get("prompts", 0) - len(targets)
            )
            if selected_batch:
                self._clear_library_prompt_selection(announce=False)
            else:
                self._reset_library_prompt_editor_state()
            controller = self._library_prompt_browse_controller
            deleted_ids = frozenset(expected_ids)
            controller.retain_stale_items(
                tuple(
                    item
                    for item in controller.retained_items
                    if item["local_id"] not in deleted_ids
                ),
                stale_copy="List may be out of date",
            )
            committed = True
            self._request_library_prompts_browse(
                controller.mutation_refresh_scope,
                focus_identity=focus_identity,
            )
            self._refresh_local_source_snapshot()
        finally:
            token = str(mutation_token)
            if self._library_prompt_delete_inflight_fingerprint == token:
                if not committed:
                    self._request_library_prompts_browse(
                        self._library_prompt_browse_controller.mutation_refresh_scope,
                        focus_identity=focus_identity,
                    )
                self._library_prompt_delete_inflight_fingerprint = None
                self._library_prompts_mutation_in_flight = False
                if self.is_mounted:
                    if self._library_prompts_view == "editor":
                        self._sync_library_prompt_mutation_presentation()
                    else:
                        self._library_prompt_mutation_disabled_states.clear()
                        self.refresh(recompose=True)
                        self.call_after_refresh(
                            self._restore_library_prompts_focus,
                            focus_identity or "library-prompts-select",
                        )
                else:
                    self._library_prompt_mutation_disabled_states.clear()

    async def _await_library_prompt_durable_call(self, awaitable: Any) -> Any:
        """Drain an admitted Prompt write even if its worker is cancelled."""
        task = asyncio.create_task(awaitable)
        while True:
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                if task.done():
                    return task.result()

    async def _await_library_prompt_save_call(self, awaitable: Any) -> Any:
        """Keep Discard interlocked for the full durable save lifetime.

        Args:
            awaitable: Admitted Prompt save operation to drain to settlement.

        Returns:
            The save operation's settled result.
        """
        self._set_library_prompt_discard_enabled(
            self._library_prompt_dirty, write_in_flight=True
        )
        try:
            return await self._await_library_prompt_durable_call(awaitable)
        finally:
            self._set_library_prompt_discard_enabled(
                self._library_prompt_dirty, write_in_flight=False
            )

    def _sync_library_prompt_mutation_presentation(self) -> None:
        """Project mutation ownership into the currently mounted Prompt canvas."""
        try:
            canvas = (
                self.query_one("#library-prompt-work-pane", LibraryPromptWorkPane)
                if self._library_prompts_view == "editor"
                else self.query_one("#library-prompts-canvas", LibraryPromptsListCanvas)
            )
        except (NoMatches, QueryError):
            if not self._library_prompts_mutation_in_flight:
                self._library_prompt_mutation_disabled_states.clear()
            return
        canvas.mutation_in_flight = self._library_prompts_mutation_in_flight
        canvas.mutation_status = self._library_prompt_mutation_status
        if canvas.mode != "editor":
            self._library_prompt_mutation_disabled_states.clear()
            canvas.refresh(recompose=True)
            return
        controls = list(canvas.query("Input, Checkbox, Button"))
        for selector in (
            "#library-prompt-block-editor",
            "#library-prompt-history-region",
        ):
            try:
                controls.append(canvas.query_one(selector, Widget))
            except (NoMatches, QueryError):
                pass
        if self._library_prompts_mutation_in_flight:
            for control in controls:
                self._library_prompt_mutation_disabled_states.setdefault(
                    control, control.disabled
                )
                control.disabled = True
        else:
            for (
                control,
                disabled,
            ) in self._library_prompt_mutation_disabled_states.items():
                if control.is_mounted:
                    control.disabled = disabled
            self._library_prompt_mutation_disabled_states.clear()

        progress = canvas.query("#library-prompts-mutation-progress")
        for indicator in progress:
            indicator.display = self._library_prompts_mutation_in_flight
        if self._library_prompts_mutation_in_flight and not progress:
            try:
                content = canvas.query_one("#library-prompt-editor-content", Widget)
                back = canvas.query_one("#library-prompt-back", Button)
            except (NoMatches, QueryError):
                return
            content.mount(
                Static(
                    "Updating selected items…",
                    id="library-prompts-mutation-progress",
                    classes="destination-purpose",
                    markup=False,
                ),
                before=back,
            )

    def _library_prompt_nearest_survivor_focus(
        self, targets: tuple[PromptBatchTarget, ...]
    ) -> str | None:
        """Choose the next page row after the focused deletion, else previous."""
        row_ids = tuple(
            row.prompt_id for row in self._build_library_prompts_state().rows
        )
        deleted_ids = {target.local_id for target in targets}
        if not row_ids or not deleted_ids.intersection(row_ids):
            return None
        focused = self._library_prompts_focus_identity()
        focused_id: int | None = None
        prefix = "library-prompt-row-"
        if isinstance(focused, str) and focused.startswith(prefix):
            try:
                focused_id = int(focused.removeprefix(prefix))
            except ValueError:
                focused_id = None
        anchor = (
            row_ids.index(focused_id)
            if focused_id in deleted_ids and focused_id in row_ids
            else min(
                index for index, row_id in enumerate(row_ids) if row_id in deleted_ids
            )
        )
        for index in range(anchor + 1, len(row_ids)):
            if row_ids[index] not in deleted_ids:
                return f"library-prompt-row-{row_ids[index]}"
        for index in range(anchor - 1, -1, -1):
            if row_ids[index] not in deleted_ids:
                return f"library-prompt-row-{row_ids[index]}"
        return None

    def _library_prompt_mutation_is_current(
        self,
        mutation_token: int,
        *,
        selection_generation: int | None,
        editor_prompt_id: int | None,
    ) -> bool:
        """Fail closed when an admitted mutation no longer owns its route."""
        if self._library_prompt_delete_inflight_fingerprint != str(mutation_token):
            return False
        if not self._library_prompts_mutation_in_flight:
            return False
        if self._library_selected_row_id != LIBRARY_ROW_BROWSE_PROMPTS:
            return False
        if selection_generation is not None:
            return (
                self._library_prompts_view == "list"
                and self._library_prompt_select_mode
                and self._library_prompt_selection.generation == selection_generation
            )
        return (
            self._library_prompts_view == "editor"
            and self._selected_prompt_id == editor_prompt_id
        )

    def _notify_library_prompt_delete_failure(self, message: str) -> None:
        """Show bounded aggregate failure copy without sensitive detail."""
        self._library_prompt_mutation_status = message
        notify = getattr(self.app_instance, "notify", None)
        if callable(notify):
            notify(message, severity="warning")

    def _refocus_library_prompt_delete_action(self) -> None:
        """Restore focus after a dismissed delete confirmation when possible."""
        try:
            self.query_one("#library-prompt-delete", Button).focus()
        except (NoMatches, QueryError):
            pass

    @on(Button.Pressed, "#library-prompts-delete-undo")
    def handle_library_prompt_delete_undo(self, event: Button.Pressed) -> None:
        """Start restoration for the Prompt/Recipe named by the receipt.

        Args:
            event: Press of the receipt's Undo button.

        Returns:
            None.
        """
        event.stop()
        if self._library_prompts_mutation_in_flight:
            return
        receipt = self._library_prompt_delete_receipt
        if receipt is None:
            return
        if type(receipt) is not PromptBatchDeleteResult:
            return
        if self._library_prompt_write_worker_is_active():
            self._notify_library_prompt_delete_failure(
                _LIBRARY_PROMPT_WRITE_IN_PROGRESS_COPY
            )
            self.refresh(recompose=True)
            self.call_after_refresh(
                self._restore_library_prompts_focus,
                "library-prompts-delete-undo",
            )
            return
        self._library_prompt_mutation_generation += 1
        mutation_token = self._library_prompt_mutation_generation
        controller = self._library_prompt_browse_controller
        controller.invalidate(controller.mutation_refresh_scope)
        self._library_prompts_mutation_in_flight = True
        self._library_prompt_delete_inflight_fingerprint = str(mutation_token)
        self._library_prompt_mutation_status = ""
        self._sync_library_prompt_mutation_presentation()
        self.run_worker(
            self._undo_library_prompt_delete(receipt, mutation_token),
            exclusive=True,
            group="library_prompt_mutation",
        )

    @on(Button.Pressed, "#library-prompts-delete-receipt-dismiss")
    def handle_library_prompt_delete_receipt_dismiss(
        self, event: Button.Pressed
    ) -> None:
        """Dismiss only the Prompt recovery receipt.

        Args:
            event: Press of the receipt's Dismiss button.

        Returns:
            None.
        """
        event.stop()
        if self._library_prompts_mutation_in_flight:
            return
        self._library_prompt_delete_receipt = None
        self.refresh(recompose=True)
        self.call_after_refresh(self._restore_library_prompts_focus, None)

    async def _undo_library_prompt_delete(
        self, receipt: PromptBatchDeleteResult, mutation_token: int
    ) -> None:
        """Atomically restore the complete typed deletion receipt."""
        restored = False
        committed = False
        try:
            service = getattr(self.app_instance, "prompt_scope_service", None)
            restore_prompts = getattr(service, "restore_deleted_prompts", None)
            if not callable(restore_prompts):
                raise TypeError("missing batch restore capability")
            result = await self._await_library_prompt_durable_call(
                self._run_library_service_call(
                    restore_prompts,
                    mode="local",
                    targets=receipt.targets,
                    isolate_in_worker=True,
                )
            )
            expected_ids = tuple(target.local_id for target in receipt.targets)
            if (
                type(result) is not PromptBatchRestoreResult
                or tuple(entry.local_id for entry in result.entries) != expected_ids
                or tuple(entry.restored_version for entry in result.entries)
                != tuple(target.expected_version + 1 for target in receipt.targets)
            ):
                raise TypeError("invalid batch restore result")
            if (
                self._library_prompt_delete_inflight_fingerprint != str(mutation_token)
                or not self._library_prompts_mutation_in_flight
            ):
                return
            if self._library_prompt_delete_receipt is receipt:
                self._library_prompt_delete_receipt = None
            self._library_prompt_mutation_status = ""
            self._local_source_counts["prompts"] = self._local_source_counts.get(
                "prompts", 0
            ) + len(receipt.entries)
            restored = True
            controller = self._library_prompt_browse_controller
            restored_versions = {
                entry.local_id: entry.restored_version for entry in result.entries
            }
            controller.retain_stale_items(
                tuple(
                    (
                        {**item, "version": restored_versions[item["local_id"]]}
                        if item["local_id"] in restored_versions
                        else item
                    )
                    for item in controller.retained_items
                ),
                stale_copy="List may be out of date",
            )
            committed = True
            self._request_library_prompts_browse(
                controller.mutation_refresh_scope,
                focus_identity=f"library-prompt-row-{receipt.entries[0].local_id}",
            )
            self._refresh_local_source_snapshot()
        except Exception:
            self._library_prompt_mutation_status = (
                "Could not restore the deleted items; Undo is still available."
            )
            notify = getattr(self.app_instance, "notify", None)
            if callable(notify):
                notify(
                    self._library_prompt_mutation_status,
                    severity="warning",
                )
        finally:
            if self._library_prompt_delete_inflight_fingerprint == str(mutation_token):
                if not committed:
                    self._request_library_prompts_browse(
                        self._library_prompt_browse_controller.mutation_refresh_scope,
                        focus_identity="library-prompts-delete-undo",
                    )
                self._library_prompt_delete_inflight_fingerprint = None
                self._library_prompts_mutation_in_flight = False
                if self.is_mounted:
                    self.refresh(recompose=True)
                    self.call_after_refresh(
                        self._restore_library_prompts_focus,
                        None if restored else "library-prompts-delete-undo",
                    )

    @on(Button.Pressed, "#library-prompt-open-existing")
    def handle_library_prompt_open_existing(self, event: Button.Pressed) -> None:
        """Discard the current unsaved edit and open the prompt whose name
        collided with it (Task 8b D3).

        Only rendered while the status line shows the name-in-use outcome
        (see ``compose_content``'s ``show_open_existing`` flag). Unlike
        every other "leave the editor" action, this deliberately does NOT
        go through ``_flush_library_prompt_save``'s dirty veto: a
        name-in-use status implies the very edit that triggered it IS the
        (still-unsaved) dirty state, so vetoing here would make the button
        permanently inert. This mirrors the conflict banner's Reload
        action, which also discards the local edit unconditionally.

        Args:
            event: Button press event emitted by the "Open existing" action.
        """
        event.stop()
        if self._library_prompts_mutation_in_flight:
            return
        if self._library_prompts_view != "editor":
            return
        self.run_worker(
            self._open_library_prompt_colliding_with_current_name(),
            exclusive=True,
            group="library_prompt_open_existing",
        )

    async def _open_library_prompt_colliding_with_current_name(self) -> None:
        """Resolve the name that triggered the name-in-use status to its
        colliding prompt and open it, replacing the current unsaved edit.

        Task 8b Fix wave 1 (Minor): resolves against
        ``_library_prompt_name_in_use`` -- the exact name captured when the
        status was set (see ``_apply_library_prompt_save_outcome`` and
        ``_return_to_library_prompt_create_draft``) -- rather than
        re-reading the editor's live Name field. The two can drift: this
        button (``show_open_existing``) stays mounted for as long as the
        status line reads name-in-use, but nothing clears that status if
        the user keeps typing in the Name field without re-saving, so a
        live re-read could resolve (or fail to resolve) against a name the
        user has since changed their mind about, not the one that actually
        collided. Falls back to the live field only if the captured name
        is unset, for robustness against any future caller that reaches
        this without going through the two capture points above.

        Args:
            None.
        """
        if self._library_prompts_mutation_in_flight:
            return
        name = self._library_prompt_name_in_use
        if not name:
            fields = self._read_library_prompt_editor_fields()
            if fields is None:
                return
            name = self._sanitize_media_field(fields[0], max_length=300)
        if not name:
            return
        service = getattr(self.app_instance, "prompt_scope_service", None)
        get_prompt = getattr(service, "get_prompt", None)
        if not callable(get_prompt):
            return
        try:
            candidate = await self._run_library_service_call(
                get_prompt,
                mode="local",
                prompt_identifier=name,
                include_deleted=False,
                isolate_in_worker=True,
            )
        except Exception:
            logger.opt(exception=True).warning(
                f"Failed to resolve the colliding Library prompt named {name!r}."
            )
            return
        if self._library_prompts_view != "editor":
            return
        candidate_id = (
            candidate.get("local_id") if isinstance(candidate, Mapping) else None
        )
        if candidate_id is None:
            return
        self._reset_library_prompt_editor_state()
        self._selected_prompt_id = candidate_id
        self._library_prompts_view = "editor"
        if self.is_mounted:
            self.refresh(recompose=True)
        self.run_worker(
            self._refresh_library_prompt_detail(candidate_id),
            exclusive=True,
            group="library_prompt_detail",
        )

    async def _resolve_library_prompt_conflict(self, *, overwrite: bool) -> None:
        """Resolve a shown save conflict via the Overwrite or Reload action.

        Both paths silently re-fetch the prompt's current server-side
        detail first (no "Loading…" placeholder -- the conflict UI stays
        put while this happens). Mirrors ``_resolve_library_note_conflict``.

        * ``overwrite=True``: take only the fresh ``version`` from that
          detail and re-save the user's *live* (kept) text with that
          version.
        * ``overwrite=False``: discard the local edits and recompose the
          editor from the freshly fetched detail.

        Either path falls back to the list view when the re-fetch
        discovers the prompt was deleted elsewhere entirely.

        Task 8b Fix wave 1: ``prompt_id is None`` here is the CREATE-flow
        sentinel (``_enter_library_prompt_create_editor``), not a "nothing
        to resolve" state -- a create's own write can raise a genuine
        ``ConflictError`` too (``_save_library_prompt``'s create-path
        write, racing another writer for the same name), which routes
        into this same conflict banner. That case has no existing row of
        its own to re-fetch a version from or overwrite, so it is
        delegated to ``_resolve_library_prompt_create_conflict`` instead
        of falling through this method's update-path body (which assumes
        a real, previously-persisted ``prompt_id`` throughout). Previously
        this method's guard (``if not prompt_id: return``) treated the
        create sentinel as a no-op for BOTH buttons, which also never
        cleared ``_library_prompt_dirty`` -- permanently trapping the user
        behind the conflict banner (``_flush_library_prompt_save`` vetoes
        Back/rail-row/prompt-row/app-tab navigation while dirty).

        Args:
            overwrite: ``True`` for Overwrite, ``False`` for Reload.
        """
        if self._library_prompts_mutation_in_flight:
            return
        snapshot = self._library_prompt_conflict_snapshot
        if snapshot is None:
            return
        prompt_id = self._selected_prompt_id
        if prompt_id is None:
            await self._resolve_library_prompt_create_conflict(
                overwrite=overwrite, snapshot=snapshot
            )
            return
        service = getattr(self.app_instance, "prompt_scope_service", None)
        get_prompt = getattr(service, "get_prompt", None)
        if not callable(get_prompt):
            return
        try:
            detail = await self._run_library_service_call(
                get_prompt,
                mode="local",
                prompt_identifier=prompt_id,
                include_deleted=True,
                isolate_in_worker=True,
            )
        except Exception:
            logger.opt(exception=True).warning(
                f"Failed to reload Library prompt {prompt_id!r} after a save conflict."
            )
            return
        if prompt_id != self._selected_prompt_id:
            return  # The user navigated away while the re-fetch was in flight.

        if not isinstance(detail, Mapping):
            logger.info(
                f"Library prompt {prompt_id!r} is no longer available; returning to list."
            )
            self._reset_library_prompt_editor_state()
            self._request_library_prompts_browse(
                self._library_prompt_browse_controller.mutation_refresh_scope,
                focus_identity=None,
            )
            self._refresh_local_source_snapshot()
            return

        if not overwrite:
            self._adopt_library_prompt_persisted_detail(
                detail,
                open_history=None,
            )
            if self.is_mounted:
                self.refresh(recompose=True)
                self.call_after_refresh(self._arm_library_prompt_editor)
            return

        fresh_version = build_prompt_editor_state(detail).version
        if fresh_version is None:
            return
        snapshot = self._library_prompt_conflict_snapshot
        name = self._sanitize_media_field(snapshot.name, max_length=300)
        author = self._sanitize_media_field(snapshot.author, max_length=200)
        details = self._sanitize_note_content(
            snapshot.details, max_length=LIBRARY_PROMPT_TEXT_MAX_CHARS
        )
        system_prompt = self._sanitize_note_content(
            snapshot.system_prompt, max_length=LIBRARY_PROMPT_TEXT_MAX_CHARS
        )
        user_prompt = self._sanitize_note_content(
            snapshot.user_prompt, max_length=LIBRARY_PROMPT_TEXT_MAX_CHARS
        )
        keywords = self._library_note_keywords_from_input(snapshot.keywords_csv)

        save_prompt = getattr(service, "save_prompt", None)
        if not callable(save_prompt):
            return
        try:
            result = await self._run_library_service_call(
                save_prompt,
                mode="local",
                prompt_identifier=prompt_id,
                name=name,
                author=author,
                details=details,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                keywords=keywords,
                isolate_in_worker=True,
            )
        except Exception:
            logger.opt(exception=True).warning(
                f"Failed to overwrite Library prompt {prompt_id!r} after a save conflict."
            )
            return
        if prompt_id != self._selected_prompt_id:
            return

        version = result.get("version") if isinstance(result, Mapping) else None
        persisted_version = version if version is not None else fresh_version + 1
        patched_detail: dict[str, Any] = dict(detail)
        if isinstance(result, Mapping):
            patched_detail.update(result)
        patched_detail["id"] = prompt_id
        patched_detail["name"] = name
        patched_detail["author"] = author
        patched_detail["details"] = details
        patched_detail["system_prompt"] = system_prompt
        patched_detail["user_prompt"] = user_prompt
        patched_detail["version"] = persisted_version
        if isinstance(result, Mapping) and "keywords" in result:
            patched_detail["keywords"] = result["keywords"]
        elif keywords is not None:
            patched_detail["keywords"] = keywords
        self._adopt_library_prompt_persisted_detail(
            patched_detail,
            status=LIBRARY_PROMPT_SAVE_STATUS_COPY["ok"],
        )
        # This recompose swaps the conflict banner's Overwrite/Reload
        # actions back for the normal action row (a real mode change,
        # unlike the plain Save success path above), so it also remounts
        # the Input/TextArea fields -- disarm first (mirroring every other
        # recompose in this editor) so their spurious mount-time `Changed`
        # is not mistaken for a fresh edit.
        if self.is_mounted:
            self.refresh(recompose=True)
            self.call_after_refresh(self._arm_library_prompt_editor)

    async def _resolve_library_prompt_create_conflict(
        self, *, overwrite: bool, snapshot: PromptEditorState
    ) -> None:
        """Resolve a save conflict raised by the CREATE flow's own write.

        Unlike ``_resolve_library_prompt_conflict``'s update-path handling
        (which re-fetches ITS row's fresh version to overwrite against, or
        to reload from), a create has no existing row of its own -- the
        ``ConflictError`` here means some OTHER prompt now holds the name
        the user typed (a genuine race ``_save_library_prompt``'s
        pre-check could not see; see that method's create-path ``except``
        branch). So there is nothing to re-fetch; recovery is built
        entirely from ``snapshot``, the conflict banner's kept text:

        * ``overwrite=True``: retries the create with the kept text,
          unchanged. A repeat "conflict" outcome re-shows this same
          banner (never a silent no-op); any other outcome (e.g.
          "name-in-use", "soft-deleted-name", or a generic error) returns
          to a plain, editable create draft with the kept text still in
          the fields and an honest status line -- the failed attempt
          remains open for the user to fix and re-save, exactly like a
          fresh create's own first-attempt failure.
        * ``overwrite=False``: abandons the kept text and returns to a
          fresh, blank create editor (mirrors
          ``_enter_library_prompt_create_editor``) -- the closest analog
          to Reload for a record that was never actually saved to reload
          FROM.

        Both paths clear ``_library_prompt_dirty``/the conflict snapshot,
        so ``_flush_library_prompt_save`` stops vetoing Back/rail-row/
        prompt-row/app-tab navigation -- the trap the finding described.

        Args:
            overwrite: ``True`` for Overwrite, ``False`` for Reload.
            snapshot: The conflict banner's kept editor state (the
                create attempt's live field values at Save time).
        """
        if self._library_prompts_mutation_in_flight:
            return
        if not overwrite:
            self._enter_library_prompt_create_editor()
            if self.is_mounted:
                self.refresh(recompose=True)
                self.call_after_refresh(self._arm_library_prompt_editor)
            return

        service = getattr(self.app_instance, "prompt_scope_service", None)
        save_prompt = getattr(service, "save_prompt", None)
        if not callable(save_prompt):
            return

        name = self._sanitize_media_field(snapshot.name, max_length=300)
        author = self._sanitize_media_field(snapshot.author, max_length=200)
        details = self._sanitize_note_content(
            snapshot.details, max_length=LIBRARY_PROMPT_TEXT_MAX_CHARS
        )
        system_prompt = self._sanitize_note_content(
            snapshot.system_prompt, max_length=LIBRARY_PROMPT_TEXT_MAX_CHARS
        )
        user_prompt = self._sanitize_note_content(
            snapshot.user_prompt, max_length=LIBRARY_PROMPT_TEXT_MAX_CHARS
        )
        keywords = self._library_note_keywords_from_input(snapshot.keywords_csv)

        try:
            result = await self._run_library_service_call(
                save_prompt,
                mode="local",
                prompt_identifier=None,
                name=name,
                author=author,
                details=details,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                keywords=keywords,
                isolate_in_worker=True,
            )
        except Exception as exc:
            logger.opt(exception=True).warning(
                "Library prompt create-retry failed after a save conflict."
            )
            if self._library_prompt_conflict_snapshot is not snapshot:
                return  # The user navigated away while the retry was in flight.
            outcome = classify_prompt_save_error(None, str(exc), exc)
            if outcome == "conflict":
                # Still colliding -- keep the banner up (same kept text)
                # rather than a silent no-op for the button just pressed.
                self._enter_library_prompt_conflict(
                    name=snapshot.name,
                    author=snapshot.author,
                    details=snapshot.details,
                    system_prompt=snapshot.system_prompt,
                    user_prompt=snapshot.user_prompt,
                    keywords_text=snapshot.keywords_csv,
                )
                return
            self._return_to_library_prompt_create_draft(snapshot, outcome)
            return

        if self._library_prompt_conflict_snapshot is not snapshot:
            return  # The user navigated away while the retry was in flight.

        result_id = (
            result.get("local_id")
            if isinstance(result, Mapping)
            else (1 if result else None)
        )
        outcome = classify_prompt_save_error(result_id, "", None)
        if outcome != "ok":
            self._return_to_library_prompt_create_draft(snapshot, outcome)
            return

        new_id = result_id
        version = result.get("version") if isinstance(result, Mapping) else None
        persisted_version = version if version is not None else 1
        patched_detail: dict[str, Any] = (
            dict(result) if isinstance(result, Mapping) else {}
        )
        patched_detail.update(
            {
                "id": new_id,
                "name": name,
                "author": author,
                "details": details,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "version": persisted_version,
            }
        )
        if isinstance(result, Mapping) and "keywords" in result:
            patched_detail["keywords"] = result["keywords"]
        elif keywords is not None:
            patched_detail["keywords"] = keywords
        self._adopt_library_prompt_persisted_detail(
            patched_detail,
            status=LIBRARY_PROMPT_SAVE_STATUS_COPY["ok"],
            open_history=False,
        )
        self._load_library_prompt_memberships()
        # Mirrors `_save_library_prompt`'s own create-success branch: a
        # brand-new prompt changes the list's membership/count, so the
        # Prompts rail badge/list must pick up the new row now.
        self._refresh_local_source_snapshot()
        if self.is_mounted:
            self.refresh(recompose=True)
            self.call_after_refresh(self._arm_library_prompt_editor)

    def _return_to_library_prompt_create_draft(
        self, snapshot: PromptEditorState, outcome: str
    ) -> None:
        """Return from the create-conflict banner to a plain, editable draft.

        Reached when an Overwrite retry (``_resolve_library_prompt_create_conflict``)
        fails with anything other than a repeat "conflict" -- keeps the
        user's kept text visible and editable (never silently discarded)
        with an honest status line, instead of leaving the conflict
        banner's buttons a dead end.

        Args:
            snapshot: The conflict banner's kept editor state.
            outcome: A ``classify_prompt_save_error`` return value (never
                ``"ok"`` -- callers only reach this on a failed retry).
        """
        detached_detail = {
            "name": snapshot.name,
            "author": snapshot.author,
            "details": snapshot.details,
            "system_prompt": snapshot.system_prompt,
            "user_prompt": snapshot.user_prompt,
            "keywords": snapshot.keywords_csv,
        }
        self._detach_library_prompt_working_copy(detached_detail)
        self._library_prompt_detached_structured = (
            snapshot.block_editor_state is not None
            and snapshot.definition_state == "supported_v2"
        )
        self._library_prompt_dirty = True
        self._library_prompt_status = LIBRARY_PROMPT_SAVE_STATUS_COPY.get(
            outcome, LIBRARY_PROMPT_SAVE_STATUS_COPY["error"]
        )
        # Task 8b Fix wave 1 (Minor): captured here too, same as
        # `_apply_library_prompt_save_outcome`, so "Open existing" (if this
        # outcome is "name-in-use") resolves against the name that actually
        # collided rather than whatever the Name field holds later.
        self._library_prompt_name_in_use = (
            snapshot.name if outcome == "name-in-use" else ""
        )
        self._library_prompt_editor_armed = False
        if self.is_mounted:
            self.refresh(recompose=True)
            self.call_after_refresh(self._arm_library_prompt_editor)

    @on(Button.Pressed, "#library-prompt-conflict-save-new")
    def handle_library_prompt_conflict_save_new(self, event: Button.Pressed) -> None:
        """Detach the kept conflict state and attempt a new-record save.

        Args:
            event: Button press event emitted by the conflict UI's
                "Save as new" action.
        """
        event.stop()
        if self._library_prompts_mutation_in_flight:
            return
        snapshot = self._library_prompt_conflict_snapshot
        if snapshot is None:
            return
        fields = self._read_library_prompt_editor_fields()
        if fields is None:
            return
        name, author, details, system_prompt, user_prompt, keywords_text = fields
        block_state = self._library_prompt_block_state or snapshot.block_editor_state
        artifact_type = block_state.artifact_type
        detached_detail = {
            "name": name,
            "author": author,
            "details": details,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "keywords": keywords_text,
            "artifact_type": artifact_type,
        }
        self._detach_library_prompt_working_copy(detached_detail)
        self._library_prompt_block_state = block_state
        self._library_prompt_detached_structured = (
            snapshot.definition_state == "supported_v2"
            or self._library_prompt_detached_structured
        )
        self._library_prompt_status = ""
        self._library_prompt_dirty = True
        self._library_prompt_editor_armed = False
        if not self.is_mounted:
            return
        self.refresh(recompose=True)
        self.call_after_refresh(
            lambda: self.run_worker(
                self._await_library_prompt_save_call(
                    self._save_library_prompt(
                        target_artifact_type=artifact_type,
                        save_as_new=True,
                    )
                ),
                exclusive=True,
                group="library_prompt_save",
            )
        )

    @on(Button.Pressed, "#library-prompt-conflict-reload")
    def handle_library_prompt_conflict_reload(self, event: Button.Pressed) -> None:
        """Resolve a shown save conflict by discarding local edits.

        Args:
            event: Button press event emitted by the conflict UI's
                "Reload" action.
        """
        event.stop()
        if self._library_prompts_mutation_in_flight:
            return
        self.run_worker(
            self._await_library_prompt_save_call(
                self._resolve_library_prompt_conflict(overwrite=False)
            ),
            exclusive=True,
            group="library_prompt_save",
        )

    @on(Button.Pressed, "#library-prompts-export")
    async def handle_library_prompts_export(self, event: Button.Pressed) -> None:
        """Open the export canvas scoped to active local Prompts.

        Args:
            event: Button press event emitted by the Prompt list's
                ``Export…`` action.
        """
        event.stop()
        if (
            self._library_prompts_mutation_in_flight
            or self._library_prompt_browse_controller.freshness == "stale"
        ):
            return
        await self._open_library_export_canvas(ExportScope(kind="prompts"))

    @on(Button.Pressed, "#library-prompts-export-selected")
    async def handle_library_prompts_export_selected(
        self, event: Button.Pressed
    ) -> None:
        """Open existing Export with the basket's canonical Prompt IDs."""
        event.stop()
        if (
            self._library_prompts_mutation_in_flight
            or self._library_prompt_browse_controller.freshness == "stale"
        ):
            return
        entries = self._library_prompt_selection.canonical_entries
        if not entries:
            return
        await self._open_library_export_canvas(
            ExportScope(
                kind="prompts",
                ids=tuple(str(entry.local_id) for entry in entries),
            )
        )

# --- BEGIN generated prompts-controller-state shims ---
# Permanent, not a cleanup-PR deletion target -- same reasoning as
# `LibraryIngestController`'s own identical block: the byte-for-byte canon
# (recipe §1) forbids editing a moved body, so the attribute names those
# bodies already use have to keep resolving through *something*. Exposes
# every `LibraryPromptsState` field under its original flat name via task
# 1's single-source `prompt_state_shim_attr()` -- THREE prefix families
# (`_library_prompt_` default, `_library_prompts_` plural, bare `_` for
# `selected_prompt_id`), the skills controller's own precedent, not the
# single-prefix string concatenation the export/collections/ingest
# controllers use.
for _lpc_field in dataclasses.fields(LibraryPromptsState):
    setattr(
        LibraryPromptsController,
        prompt_state_shim_attr(_lpc_field.name),
        property(
            lambda self, _n=_lpc_field.name: getattr(
                self._prompts_state_accessor(), _n
            ),
            lambda self, value, _n=_lpc_field.name: setattr(
                self._prompts_state_accessor(), _n, value
            ),
        ),
    )
del _lpc_field
# --- END generated prompts-controller-state shims ---

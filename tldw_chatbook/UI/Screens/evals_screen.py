"""Evaluations screen implementation.

The evaluation hub used to push a separate Textual ``Screen`` object as a
child inside a plain ``Container`` (``EvalsWindowV3.compose()`` yielding
``EvalNavigationScreen``). That is not a supported way to mount a
``Screen``: it mounts structurally (child widgets are still queryable) but
the compositor never gives it a laid-out region, so it renders with zero
size -- confirmed both by PR 1's before/after screen capture (header and
mode strip render, the body is empty) and by an isolated reproduction here
during Task 3 (a nested ``Screen``'s own descendants report
``region=Region(0, 0, 0, 0)`` despite existing in the DOM).

This screen replaces that architecture with the shared Lab frame
(``lab_frame.LabScreen``): the library rail, detail body and readiness
inspector are the frame's three regions, driven by selection state
(``EvalsSelection``, ``evals_state.py``) instead of a hand-rolled screen
stack. **No ``Screen`` subclass is mounted inside any region here.**

Detail and inspector content is swapped REGION BY REGION on selection change
(``select`` -> ``_swap_selection_regions``), tearing down and remounting plain
widgets (``Static``/``Button``/``Vertical``) -- never a ``Screen``. Until
task-15475 this was a screen-level ``refresh(recompose=True)``, which also
rebuilt the nav bar, footer, header row and mode strip -- 150-300 widgets by
the input-latency audit's count -- and is now reserved for whole-screen
rebuilds. Two consequences worth knowing before touching the swap: the rail's
ROWS are only rebuilt when a caller says they changed (``rail_dirty``), and
each region's frame-owned collapse header is NOT mode content, so a swap
removes only the children it put there (see ``_replace_region``).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional

#: The package css/ directory this screen's lazily-parsed sheet lives in.
_EVALS_SCREEN_CSS_DIR = Path(__file__).resolve().parent.parent.parent / "css"

from loguru import logger
from rich.markup import escape as escape_markup
from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.css.query import QueryError
from textual.widgets import Button, Static

from ..focus_ownership import focus_is_on_screen
from ...Chat.Chat_Functions import chat_api_call
from ...DB.Evals_DB import ConflictError, EvalsDB
from ...Evals.character_probe.cards import snapshot_cards
from ...Evals.character_probe.models import CharacterProbeConfig
from ...Evals.character_probe.runner import (
    CancelToken as CharacterCancelToken,
    ChatCallable,
    CharacterProbeRunner,
)
from ...Evals.character_probe.storage import (
    create_probe_run_group,
    is_probe_set,
    load_character_bench,
    load_probe_set,
    save_character_bench,
    save_conversations,
)
from ...Evals.word_bench.models import PreflightResult
from ...Evals.word_bench.models import Target as WordBenchTarget
from ...Evals.word_bench.runner import CancelToken, CaptureClientLike
from ...Evals.word_bench.storage import _unique_name, duplicate_bench
from ...Widgets.confirmation_dialog import ConfirmationDialog
from ..Evals import sample_bench
from ..Evals.bench_editor import BenchEditor, ClassicTaskDetail
from ..Evals.character_bench_editor import CharacterBenchEditor, ProbeSetDetail
from ..Evals.evals_state import EvalsSelection, EvalsViewModel, SelectionKind
from ..Evals.inspector import CharacterBenchEstimate, EvalsCellInspector, EvalsInspector
from ..Evals.library_rail import RAIL_SECTIONS, LibraryRail
from ..Evals.results_grid import ResultsGrid
from ..Evals.snippet_editor import SnippetEditor
from ..Workbench.workbench_state import WorkbenchHeaderState
from ..Lab_Modules.lab_rail_layout import LabRailLayout
from .lab_frame import LabScreen

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


#: Class on the collapse header the WORKBENCH composes as the first child of
#: `#lab-rail` and `#lab-inspector` (see `lab_workbench._region_collapse_header`).
#: Frame-owned chrome, not mode content -- a region swap must step around it.
_FRAME_REGION_HEADER_CLASS = "console-rail-header"


def _extract_chat_reply_text(response: Any) -> str:
    """The generated text out of one ``chat_api_call(streaming=False)`` reply.

    ``chat_api_call`` is a thin dispatcher: for a non-streaming call it
    returns whatever its provider handler returns verbatim, and llama.cpp's
    handler (``chat_with_llama`` -> ``_chat_with_openai_compatible_local_
    server``) returns the raw, already-JSON-decoded OpenAI-shaped response
    body -- never pre-extracted text. This is the one place that extraction
    happens for a character probe run.

    Per ``character_probe.models.ConversationTurn``'s own docstring, a
    reply that legitimately generated NO content (``message.content`` is
    ``""`` or ``None``) must become ``""``, not an error -- "the model said
    nothing" is a real, recordable observation this eval exists to surface.
    Anything else that does not have the expected shape (no ``choices``, a
    non-mapping ``message``, ...) raises instead: that is a genuine
    extraction failure, and ``CharacterProbeRunner._run_conversation``
    already catches it per-turn and records it as that conversation's own
    ``error`` -- silently degrading it to ``""`` here would misrepresent a
    real failure as an empty-but-successful reply.

    Args:
        response: Whatever ``chat_api_call`` returned.

    Returns:
        str: The generated text, or ``""`` for a legitimately contentless
        reply.

    Raises:
        ValueError: If ``response`` carries no usable
            ``choices[0].message.content`` at all.
    """
    if isinstance(response, str):
        return response
    if isinstance(response, Mapping):
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            message = first.get("message") if isinstance(first, Mapping) else None
            if isinstance(message, Mapping):
                content = message.get("content")
                if content is None or isinstance(content, str):
                    return content or ""
    raise ValueError(
        f"llama.cpp chat response had no usable message content: {response!r}"
    )


def _default_character_probe_chat_factory(_config: CharacterProbeConfig) -> ChatCallable:
    """Production ``ChatCallable``: a real call through the app's normal
    chat path (``Chat_Functions.chat_api_call``), per the design spec's own
    "Execution" section ("multi-turn messages in, text out, real sampler,
    no logprobs").

    **Hardcoded to ``api_endpoint="llama_cpp"``, deliberately narrow, not
    general** -- mirrors ``sample_bench.py``'s own "why the target
    resolution is narrow, not general" rationale for the identical reason:
    ``ChatCallable``'s shape (``chat_fn(messages, model, temperature,
    max_tokens, seed)``) carries no ``provider`` argument -- the runner
    calls this SAME callable for every target in the run, keyed only by
    ``model=``, so this factory must commit to one provider up front rather
    than branching per call on information it is never given. Every
    character-bench target reachable through this app's own UI today is a
    ``llama_cpp`` row: ``CharacterBenchEditor`` (Task 4) has no Add-target
    control, so ``target_ids`` is populated ONLY at bench creation, via
    ``sample_bench.resolve_unsteered_llama_cpp_target`` -- an llama.cpp-
    only resolution, like the one-click sample bench's own (see
    ``_on_new_character_bench_requested``'s own docstring for why it is a
    DIFFERENT function, not the same one). Inventing a
    per-provider dispatch here for targets this app can never actually
    create would be exactly the kind of fabrication ``sample_bench.py``'s
    own module docstring already rules out for the identical reason.

    ``chat_api_call``/``chat_with_llama`` resolve the endpoint URL, API
    key, and any config-level model default themselves, from the SAME live
    runtime config snapshot every other real chat call in this app reads
    (``get_runtime_config_snapshot()``) -- unlike ``sample_bench.py``'s raw
    ``WordBenchCaptureClient``, this factory needs no ``app_config``
    parameter of its own to thread through.

    Args:
        _config: The bench being run. Unused today (the callable it builds
            reads temperature/max_tokens/seed from its own per-call
            arguments, which ``CharacterProbeRunner`` already derives from
            this same config) -- kept as a parameter so the DI seam's
            shape (``Callable[[CharacterProbeConfig], ChatCallable]``,
            mirroring ``_sample_bench_client_factory``'s per-target
            callable) never has to change if a future caller needs it.

    Returns:
        ChatCallable: A plain, synchronous callable -- never a coroutine
        function -- matching ``character_probe.runner.ChatCallable``'s own
        contract. ``CharacterProbeRunner`` dispatches every call through
        ``asyncio.to_thread`` itself; this factory must never wrap it in
        anything ``async``.
    """

    def _chat(
        *,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        seed: Optional[int],
    ) -> str:
        response = chat_api_call(
            api_endpoint="llama_cpp",
            messages_payload=messages,
            model=model,
            temp=temperature,
            max_tokens=max_tokens,
            seed=seed,
            streaming=False,
        )
        return _extract_chat_reply_text(response)

    return _chat


class EvalsScreen(LabScreen):
    """Evals mode: library rail, detail body, readiness inspector -- on the Lab frame."""

    #: TASK-24459: evals-owned rules split out of ``features/_evals.tcss``,
    #: parsed on first visit instead of before first paint. GENERATED by
    #: ``css/build_css.py``. (`LabScreen` declares no ``CSS_PATH`` of its
    #: own, so this override drops nothing.)
    CSS_PATH = [
        str(
            _EVALS_SCREEN_CSS_DIR / "screen_feature_evals.tcss"
        )
    ]

    #: Both rails open on a first run. Unlike Models' server list or Speech's
    #: dependency detail, the Evals inspector is where target readiness is
    #: reported -- the reason to look at a bench before running it. Behind a
    #: collapsed handle it is content the user has to know to go find.
    LAB_FIRST_RUN_RAILS = LabRailLayout()

    def __init__(self, app_instance: "TldwCli", **kwargs):
        super().__init__(app_instance, "evals", **kwargs)
        self._view_model = EvalsViewModel(self._resolve_db(app_instance))
        #: The cross-database handle for character cards (``ChaChaNotes_
        #: DB``, a different database from the ``EvalsDB`` `_view_model`
        #: wraps -- see ``EvalsViewModel.character_cards``'s own
        #: docstring). Resolved ONCE here, like `_view_model`'s own
        #: ``EvalsDB`` handle just above, not re-read on every compose --
        #: this screen never opens it itself; `app.py`'s startup wiring
        #: (`self.chachanotes_db = ...`) already owns that. `None` when
        #: unavailable, degrading `EvalsViewModel.character_cards` to an
        #: empty picker rather than crashing the character-bench editor.
        self._chacha_db: Any = self._resolve_chacha_db(app_instance)
        self._selection = EvalsSelection()
        #: Preflight resolved once per selection, not once per pane.
        #: The frame calls compose_lab_rail/compose_lab_inspector during
        #: _populate_regions and build_lab_body later, from the deferred
        #: mount -- three separate calls where the old single
        #: compose_content() resolved it once and threaded it into both
        #: panes. Without this cache, adopting the frame would silently
        #: reintroduce the duplicate run-group snapshot read that I2
        #: fixed. Cleared wherever the selection changes.
        self._preflight_cache: dict[str, PreflightResult] | None = None
        #: task-15475: selection swaps are serialized by this lock and
        #: superseded by revision rather than cancelled, so a teardown is
        #: never interrupted half-done. `_selection_rail_dirty` accumulates
        #: across superseded calls -- see `select`.
        self._selection_swap_lock = asyncio.Lock()
        self._selection_swap_revision = 0
        self._selection_rail_dirty = False
        # Shared with LibraryRail by reference (see its own docstring) so
        # collapsed/expanded rail sections survive a selection-triggered
        # rebuild, which constructs a brand-new LibraryRail instance.
        self._rail_open_sections: dict[str, bool] = {
            section_id: True for section_id in RAIL_SECTIONS
        }
        #: DI seam for tests only -- overrides sample_bench.py's default
        #: (real ``WordBenchCaptureClient``) with a fake, mirroring
        #: ``WordBenchRunner``'s own client_factory parameter. ``None`` in
        #: production.
        self._sample_bench_client_factory: Optional[
            Callable[[WordBenchTarget], CaptureClientLike]
        ] = None
        #: True for the duration of one create-and-run flow. Guards against
        #: a second click starting a second worker once a run is genuinely
        #: in flight -- the button is also disabled live (see
        #: ``_set_sample_bench_running_ui``), but a disabled widget not yet
        #: re-rendered, or a message posted directly as this screen's own
        #: tests do, must not be able to race past it. For the tighter
        #: race -- two requests already queued before either dispatches --
        #: it is ``exclusive=True`` on the worker (below), not this flag,
        #: that actually protects: Textual cancels the second worker's Task
        #: before its first step, so the first worker's body (including its
        #: flag-set line) never runs.
        self._sample_bench_running: bool = False
        #: The active run's cooperative cancel token, or ``None`` when no
        #: run is in flight. NOTHING READS THIS TODAY (TASK-861 audited
        #: it): the running-guard above prevents the second-click race
        #: that would otherwise need it, and no Cancel affordance exists
        #: yet in this screen. Kept as a real, threaded seam rather than a
        #: decorative parameter, since ``WordBenchRunner.run`` already
        #: accepts one and a future PR wiring an actual Cancel button (PR
        #: 3c, per this program's own PR numbering) should not need a
        #: second plumbing pass to reach it.
        self._sample_bench_cancel_token: Optional[CancelToken] = None
        #: The selection snapshotted in ``_on_sample_bench_requested`` at
        #: PRESS time, before ``run_worker`` is even called -- same
        #: capture-outside-the-worker rationale as ``_bench_run_task_id``
        #: below (the selection can move before the scheduled worker's
        #: first line actually runs). Unlike a bench-run, a sample bench
        #: does not exist yet when this button is pressed, so there is no
        #: bench id to pin; what a completing worker must not yank the
        #: user away from is wherever they WERE, not a specific bench --
        #: see ``_selection_unmoved_since_launch`` (task-1482 Task 2).
        self._sample_bench_launch_selection: EvalsSelection = EvalsSelection()
        #: True for the duration of one run-existing-bench flow. Same
        #: double-guard rationale as ``_sample_bench_running`` (see that
        #: field's own comment above): this flag stops a second press once
        #: a worker has already set it, while ``exclusive=True`` on the
        #: worker (below) covers the tighter race of two presses already
        #: queued before either dispatches.
        self._bench_run_running: bool = False
        #: The bench (``eval_tasks``) id the in-flight run worker is
        #: running, resolved from the current selection at PRESS time (see
        #: ``_on_primary_action_pressed``) and never re-read from
        #: ``self._selection`` inside the worker -- the selection can move
        #: (another rail click) while the worker is still in flight.
        self._bench_run_task_id: Optional[str] = None
        #: The active run's cooperative cancel token, or ``None`` -- same
        #: no-current-caller status as ``_sample_bench_cancel_token`` above
        #: (no Cancel affordance exists in this screen yet): kept as a
        #: real, threaded seam rather than a decorative parameter.
        self._bench_run_cancel_token: Optional[CancelToken] = None
        #: task-1482 Task 7 fix round 1 (reviewer-found reentrancy): True
        #: from the moment ``_on_delete_bench_pressed`` dispatches
        #: ``_delete_bench_flow`` until ``_apply_bench_deletion`` finishes
        #: (confirmed, cancelled, or erroring out). See that handler's own
        #: docstring for why a synchronous flag -- not ``exclusive=True``
        #: on the worker, unlike the run-bench/sample-bench pattern this
        #: screen uses everywhere else -- is the correct guard here.
        self._bench_delete_pending: bool = False
        #: True for the duration of one character-bench run flow -- the
        #: character-probe sibling of ``_bench_run_running`` above, sharing
        #: the SAME physical ``#evals-primary-action`` button (a word bench
        #: and a character bench are never both selected at once, so the
        #: button itself is never ambiguous, but the WORKER guards must
        #: still cross-check every other kind: see ``_on_primary_action_
        #: pressed`` and ``_on_sample_bench_requested``, both extended to
        #: this THIRD flag for task-1691 phase 2 Task 6, mirroring the
        #: exact "PR #1113 review" cross-worker race this file's other two
        #: flags already document).
        self._character_bench_run_running: bool = False
        #: The character bench (``eval_tasks``) id the in-flight run worker
        #: is running -- resolved at PRESS time (see
        #: ``_on_primary_action_pressed``), never re-read from
        #: ``self._selection`` inside the worker, mirroring ``_bench_run_
        #: task_id`` exactly.
        self._character_bench_run_task_id: Optional[str] = None
        #: The active character-bench run's cooperative cancel token, or
        #: ``None``. Unlike ``_bench_run_cancel_token``/``_sample_bench_
        #: cancel_token`` above (real seams with no current reader -- no
        #: Cancel affordance exists in this screen for ANY bench type yet),
        #: this one IS read today, defensively, by nothing new this task
        #: adds; kept for the same future-Cancel-button reason those two
        #: document. A character-probe run's own cancellation contract
        #: (``character_probe.runner``'s module docstring) additionally
        #: differs from the word-bench one: cancelling stops SCHEDULING
        #: further turns/conversations but cannot abort a turn already
        #: in flight, since every provider call is dispatched through
        #: ``asyncio.to_thread``, which survives Task cancellation.
        self._character_bench_run_cancel_token: Optional[CharacterCancelToken] = None
        #: DI seam for tests only -- overrides the production chat callable
        #: (``_default_character_probe_chat_factory``, a plain synchronous
        #: ``def`` dispatched to a real llama.cpp endpoint via
        #: ``chat_api_call``) with a fake, mirroring ``_sample_bench_
        #: client_factory`` above seam-for-seam EXCEPT for shape: a word
        #: bench's client factory is called once PER TARGET (``Callable
        #: [[Target], CaptureClientLike]``, since ``WordBenchRunner`` opens
        #: one HTTP client per target); a character bench's factory is
        #: called ONCE for the whole run (``Callable[[CharacterProbeConfig],
        #: ChatCallable]``), because ``CharacterProbeRunner`` takes a single
        #: chat callable up front and reads only ``model=`` per call -- see
        #: ``_run_character_bench_worker``'s own docstring for why this
        #: means every target a character bench can ever run against is
        #: implicitly llama.cpp today. ``None`` in production.
        self._character_probe_chat_factory: Optional[
            Callable[[CharacterProbeConfig], ChatCallable]
        ] = None

    def _current_app_config(self) -> dict[str, Any]:
        """The app's loaded settings, read fresh on every recompose (not
        cached in ``__init__``) so a provider configured in Settings after
        this screen first mounted is picked up without a restart."""
        return dict(getattr(self.app_instance, "app_config", None) or {})

    @staticmethod
    def _resolve_db(app_instance: object) -> Optional[EvalsDB]:
        """Find the app's real ``EvalsDB``, or ``None`` if unavailable.

        ``app.py``'s ``_wire_evaluation_services`` already constructs
        ``app_instance.evaluation_orchestrator`` (an ``EvaluationOrchestrator``
        wrapping a real ``EvalsDB`` as ``.db``) at startup; this screen reads
        that existing wiring rather than opening a second database handle.
        ``evaluation_orchestrator`` is ``None`` when that wiring itself
        failed (caught and logged in ``_wire_evaluation_services``), so this
        degrades to ``None`` rather than raising -- ``EvalsViewModel``
        renders an empty (not broken) workbench in that case.
        """
        orchestrator = getattr(app_instance, "evaluation_orchestrator", None)
        return getattr(orchestrator, "db", None)

    @staticmethod
    def _resolve_chacha_db(app_instance: object) -> Any:
        """Find the app's real ``ChaChaNotes_DB`` handle, or ``None``.

        Character cards live in a different database from the one
        ``_resolve_db`` resolves above -- ``app.py``'s startup wiring
        assigns the real handle to ``app_instance.chachanotes_db``
        directly (unlike ``evaluation_orchestrator.db``, there is no
        intermediate wrapper object to unwrap here). Mirrors the exact
        ``getattr(self.app_instance, "chachanotes_db", None)`` convention
        already used elsewhere in this app (e.g. ``chat_screen.py``) for
        the same handle, rather than inventing a second lookup path.
        """
        return getattr(app_instance, "chachanotes_db", None)

    def select(  # noqa: A002
        self,
        *,
        kind: SelectionKind,
        id: Optional[str] = None,
        rail_dirty: bool = True,
    ) -> None:
        """Set the workbench's active selection and refresh dependent panes.

        Public, not just an internal message handler: it is the shell's own
        selection API. ``LibraryRail.EvalsSelectionChanged`` (posted on a
        rail row press) routes here via ``_on_library_selection_changed``
        below, but a caller may also drive selection directly.

        A plain (non-async) method: it only SCHEDULES the region swap, it
        does not await its completion -- callers that need the panes settled
        should ``await pilot.pause()`` afterward, exactly as they did when
        this scheduled a ``refresh(recompose=True)``.

        task-15475: it no longer recomposes the screen. A screen recompose
        rebuilt the nav bar, footer, header row, mode strip and the whole
        ``LabWorkbench`` -- 150-300 widgets by the input-latency audit's
        count -- to repaint two regions that read the selection.

        Args:
            kind: The selected object's kind (``SelectionKind`` --
                ``"none"``, ``"bench"``, ``"classic"``, ``"character_
                bench"``, ``"dataset"``, or ``"run_group"``).
            id: The selected object's id. Only meaningful for a non-
                ``"none"`` ``kind``; may be ``None`` (e.g. for ``kind=
                "none"``, or a caller clearing the selection).
            rail_dirty: Whether the rail's ROWS (not just which one is
                active) may have changed. Defaults to ``True``, the safe
                answer for every mutation caller -- a save, a finished run, a
                duplicate, a delete -- and for any caller outside this class.
                Only the rail-click path passes ``False``: the rail is what
                posted that selection, so its rows cannot have moved, and it
                just re-marks the active row in place.
        """
        self._selection = EvalsSelection(kind=kind, id=id)
        self._preflight_cache = None
        self._register_grid_shortcuts()
        if not self.is_mounted:
            return
        # Accumulated, never overwritten: if a rail-rebuilding selection is
        # superseded by a rail-click one before either swap runs, the rail
        # still has to be rebuilt.
        self._selection_rail_dirty = self._selection_rail_dirty or rail_dirty
        self._selection_swap_revision += 1
        self.run_worker(
            self._swap_selection_regions(
                revision=self._selection_swap_revision,
            ),
            group="evals-selection-regions",
            # NOT exclusive. An exclusive group cancels the in-flight swap,
            # and the cancellation can land INSIDE `remove_children` -- which
            # strands a region emptied but never refilled (the rail, if the
            # superseding swap is a rail-click one that does not rebuild it)
            # and skips the post-teardown mouse-capture sweep. A lock plus a
            # revision check gets the same "only the newest wins" without ever
            # interrupting a teardown: superseded swaps no-op BEFORE touching
            # anything. Mirrors `speech_playground_pane`'s reconcile-to-current
            # idiom in the same programme.
            exclusive=False,
            exit_on_error=False,
        )

    async def _swap_selection_regions(self, *, revision: int) -> None:
        """Rebuild the Lab regions that read the selection, and only those.

        The three regions are the frame's (``lab_frame.LabScreen``):
        ``#lab-rail``, ``#lab-body``, ``#lab-inspector``. Everything outside
        them -- the header row and its status chips (static for this mode),
        the mode strip, the nav bar and the footer -- is unaffected by a
        selection, so it now survives one.

        Removals are awaited before the replacements mount: Textual's
        ``remove`` is deferred, and every id here (``#evals-detail-pane``,
        ``#evals-inspector-pane``, the rail's row ids) is re-used by the
        replacement, so mounting early raises ``DuplicateIds``.

        The whole swap runs inside ``self.batch()`` -- the same lock-plus-
        ``batch_update`` ``Widget.recompose`` uses. Not a nicety: two
        separately-awaited region swaps each drove their own layout/repaint
        pass, and batching them cut the swap's own cost from a median of
        105 ms to 88 ms on the isolated harness.

        Serialized by a lock and superseded by revision, never cancelled --
        see ``select`` for why an exclusive worker group was the wrong tool.

        Focus is captured before the teardown and restored after IF it was
        inside a region this swap rebuilds and its id still resolves. Ids are
        stable across a rebuild (the rail's row ids, the detail pane's field
        ids), so this beats what the whole-screen recompose managed -- it left
        focus wherever Textual's ``_reset_focus`` happened to drop it, which
        on a rail rebuild is a section TOGGLE, one Space away from collapsing
        the section the user is working in.

        The restore is deferred and yields to the body: a freshly mounted body
        may claim focus ON PURPOSE (``ResultsGrid.on_mount`` focuses its
        DataTable so the ``l``/``b``/``s``/``e`` keys the footer advertises
        actually work). Anything focused inside ``#lab-body`` after the swap is
        that deliberate claim and is left alone; anything else is Textual's
        automatic reset, which the captured identity overrides.

        Args:
            revision: The ``select`` call this swap belongs to. A swap whose
                revision is no longer current returns without touching a
                single widget.

        Returns:
            None.
        """
        async with self._selection_swap_lock:
            if revision != self._selection_swap_revision or not self.is_mounted:
                # Superseded while queued (or the screen went away): the newest
                # swap owns the work, including this one's rail_dirty.
                return
            rail_dirty = self._selection_rail_dirty
            self._selection_rail_dirty = False
            focus_id = self._focused_id_in_swapped_regions(rail_dirty=rail_dirty)

            # Same protection the whole-screen recompose gave (task-627): a
            # widget about to be torn down -- the detail pane is full of
            # `Input`s -- must not be left holding the mouse capture, or every
            # click app-wide is silently swallowed from then on.
            self.release_mouse_capture_for_teardown()
            async with self.batch():
                if rail_dirty:
                    await self._replace_region(
                        "#lab-rail", list(self.compose_lab_rail())
                    )
                else:
                    try:
                        self.query_one(LibraryRail).apply_selection(self._selection)
                    except QueryError:
                        # No rail mounted (teardown race) -- the body/inspector
                        # swap below is still worth doing if they are there.
                        pass

                body = self.build_lab_body()
                await self._replace_region(
                    "#lab-body", [] if body is None else [body]
                )
                await self._replace_region(
                    "#lab-inspector", list(self.compose_lab_inspector())
                )
            # A MouseDown already queued on a child's pump can capture that
            # child DURING the removal drain, after the release above.
            self.sweep_stale_mouse_capture()
            if not self.is_mounted:
                return
            # Same notification the frame's own deferred body mount fires, so a
            # mode that re-wires itself against a fresh body keeps working.
            self.on_lab_body_ready()
            if focus_id:
                self.call_later(self._restore_selection_focus, focus_id)

    def _focused_id_in_swapped_regions(self, *, rail_dirty: bool) -> Optional[str]:
        """The focused widget's id, if this swap is about to destroy it.

        Args:
            rail_dirty: Whether ``#lab-rail`` is being rebuilt too.

        Returns:
            The id to restore afterwards, or None when focus is elsewhere,
            unidentifiable, or on a widget that survives.
        """
        focused = self.app.focused if self.is_running else None
        if not focus_is_on_screen(focused, self) or not focused.id:
            return None
        region_ids = ["lab-body", "lab-inspector"]
        if rail_dirty:
            region_ids.append("lab-rail")
        for ancestor in focused.ancestors_with_self:
            if getattr(ancestor, "id", None) in region_ids:
                return focused.id
        return None

    def _restore_selection_focus(self, focus_id: str) -> None:
        """Put focus back on ``focus_id`` unless the new body claimed it.

        Deferred (``call_later``) rather than immediate so it runs AFTER the
        focus callbacks a freshly mounted widget queued from its own
        ``on_mount`` -- which is what lets it see, and yield to, a deliberate
        claim like ``ResultsGrid``'s.
        """
        if not self.is_mounted:
            return
        focused = self.app.focused
        if focused is not None:
            for ancestor in focused.ancestors_with_self:
                if getattr(ancestor, "id", None) == "lab-body":
                    # A fresh body widget focused itself on purpose.
                    return
        try:
            self.query_one(f"#{focus_id}").focus()
        except QueryError:
            # The widget has no counterpart in the rebuilt region (a different
            # selection kind renders different controls) -- leave focus alone.
            pass

    async def _replace_region(self, region_id: str, content: list[Any]) -> None:
        """Swap one Lab region's MODE-OWNED children for ``content``.

        Mode-owned is the load-bearing word. ``#lab-rail`` and
        ``#lab-inspector`` are not empty containers the mode fills: the
        WORKBENCH composes a collapse header (title + collapse button) as
        each region's first child, because collapse is frame-owned, not a
        per-mode concern (``lab_workbench.LabWorkbench.compose``). That is
        precisely why ``LabScreen._populate_regions`` mounts mode content with
        ``mount_all``, which appends, and says so.

        A blanket ``remove_children()`` here destroyed those headers on the
        first selection -- permanently, since nothing recomposes the screen
        any more and the collapse buttons have no keyboard binding. Removing
        an explicit list of the non-header children keeps the frame's chrome
        and leaves it first, exactly where a fresh compose puts it.

        Args:
            region_id: ``#``-prefixed id of the frame region to refill.
            content: Widgets to mount after the existing mode content is gone.

        Returns:
            None.
        """
        try:
            region = self.query_one(region_id)
        except QueryError:
            logger.warning("Lab region {} missing; selection swap skipped.", region_id)
            return
        mode_owned = [
            child
            for child in region.children
            if not child.has_class(_FRAME_REGION_HEADER_CLASS)
            and not child.has_class("-textual-system")
        ]
        if mode_owned:
            await region.remove_children(mode_owned)
        if content and region.is_mounted:
            await region.mount_all(content)

    def _selection_unmoved_since_launch(
        self, launch_selection: EvalsSelection, bench_task_id: Optional[str]
    ) -> bool:
        """True when it is safe for a just-finished background worker
        (``_run_bench_worker``/``_create_sample_bench_worker``) to move the
        screen's selection to the run group it just produced.

        Two cases count as safe, matching what a user would read as "I'm
        still watching this run" rather than "I've moved on":

        1. ``self._selection`` is unchanged from ``launch_selection`` -- the
           selection captured at the moment the run/creation was started
           (``_bench_run_task_id`` for the bench-run worker, ``self.
           _sample_bench_launch_selection`` for the sample-bench worker,
           which has no pre-existing bench to pin against).
        2. The user has since navigated INTO one of ``bench_task_id``'s own
           run groups (e.g. clicked a still-"running" row in the rail while
           the run was in flight, per ``test_rail_run_row_shows_the_
           running_glyph_while_the_run_is_in_flight``) -- moving them to the
           freshly finished run group there is a refresh, not a yank.

        Any other selection means the user navigated somewhere unrelated
        while the worker was running -- once the bench editor holds
        unsaved form state (task-1482 Task 2's own motivation), forcing a
        recompose there would destroy it. The completing worker must
        degrade to a toast-only notification instead of calling
        ``select()`` (task-1482 Task 2).

        A THIRD, independent check overrides both branches above (task-1610):
        if the currently mounted detail pane holds a ``BenchEditor`` OR a
        ``CharacterBenchEditor`` whose ``is_dirty()`` is ``True``, this
        returns ``False`` regardless of selection identity -- a recompose
        would destroy that unsaved state even when the selection itself
        never moved. This is deliberately NOT limited to ``bench_task_id``'s
        own editor: the sample-bench worker's sharpest case is a user
        parked on some OTHER bench's editor (unrelated to the sample bench
        just created elsewhere) with unsaved edits -- ``self._selection``
        reads "unmoved" there (it never pointed at the sample bench to
        begin with), but the mounted editor is still real, unsaved, user
        state a recompose must not touch. Queried defensively (``QueryError``
        -> not dirty, nothing to protect): most selections never mount
        either editor at all, and the two are never BOTH mounted at once
        (mutually exclusive selection kinds), so checking both costs at
        most one real query.

        task-1691 phase 2 Task 6 review round 1 (Important finding): before
        this check covered ``CharacterBenchEditor`` too, editing a
        character bench's field without saving, then pressing Run (a
        SEPARATE button -- the editor stays mounted), let the completing
        worker's own ``select(kind="run_group", ...)`` silently discard
        the unsaved edit -- this is exactly the class of bug task-1610's
        original ``BenchEditor``-only check was built to prevent, just
        newly reachable because Task 6 is the first code to ever call
        ``select()`` after a character-bench run completes.
        """
        from textual.css.query import QueryError  # noqa: PLC0415 -- narrow, matches this module's other local imports

        try:
            editor: Any = self.query_one("#evals-bench-editor", BenchEditor)
        except QueryError:
            editor = None
        if editor is None:
            try:
                editor = self.query_one(
                    "#evals-character-bench-editor", CharacterBenchEditor
                )
            except QueryError:
                editor = None
        if editor is not None and editor.is_dirty():
            return False

        if self._selection == launch_selection:
            return True
        if bench_task_id and self._selection.kind == "run_group" and self._selection.id:
            group = self._view_model.run_group_by_id(self._selection.id)
            if group is not None and group.get("task_id") == bench_task_id:
                return True
        return False

    def _character_run_group(self, group: Optional[Mapping[str, Any]]) -> bool:
        """Whether a resolved run-group row (``EvalsViewModel.run_group_
        by_id``) belongs to a CHARACTER bench rather than a word bench.

        Shared by every place a ``"run_group"`` selection composes
        something bench-type-specific: ``_compose_detail_pane`` (a neutral
        placeholder instead of the word-bench-shaped ``ResultsGrid``) and
        ``_register_grid_shortcuts`` below (so lens/baseline/sort/export
        are never advertised in the footer for a run with no grid to act
        on) -- see this task's own "the two bench types never share a
        detail surface" constraint. Until task-1691 phase 2 Task 6 wired
        Run for a character bench, this was unreachable: no character-
        probe run group could ever exist to select.

        Args:
            group: A resolved row from ``run_group_by_id``, or ``None``
                for an unresolvable/missing selection.

        Returns:
            bool: ``False`` for ``None`` (nothing to classify) or a word
            bench's own run group; ``True`` only for a genuine
            character-probe run group.
        """
        if group is None:
            return False
        return self._view_model.character_bench_by_id(group.get("task_id")) is not None

    def _register_grid_shortcuts(self) -> None:
        """Advertises the results grid's `l`/`b`/`s`/`e` keys (see
        ``results_grid.ResultsGrid.BINDINGS``) through the shared
        ``ShortcutContext`` machinery only while a run group is selected --
        the only selection kind that mounts a ``ResultsGrid`` at all -- so
        the footer never advertises a grid shortcut with no grid on
        screen. `e` (export) is Task 2's addition -- Task 1 deliberately
        left it unbound and unadvertised so this task could claim it
        without a collision.

        Mirrors ``library_screen.py``'s ``_register_footer_shortcuts``: a
        static hint set, re-registered on every selection change rather
        than driven from inside the grid widget itself, since the grid
        does not know when it stops being the active selection (its own
        unmount does not fire a footer-clearing hook).

        task-1691 phase 2 Task 6: a CHARACTER-probe run group selection
        also clears these -- ``_compose_detail_pane`` renders a plain
        placeholder for that case, never ``ResultsGrid``, so advertising
        grid-only keys there would advertise controls with nothing
        mounted to act on. Classified via a single targeted
        ``list_runs(run_group_id=..., limit=1)`` row, deliberately NOT
        ``EvalsViewModel.run_group_by_id`` (which pivots every run in the
        database, via ``run_groups()``, just to answer this one
        selection's bench type) -- this method already runs on every
        selection change, so it must stay cheap even when nothing is
        actually selected in the "run_group" kind.
        """
        is_character_run = False
        if (
            self._selection.kind == "run_group"
            and self._selection.id
            and self._view_model.db is not None
        ):
            rows = self._view_model.db.list_runs(
                run_group_id=self._selection.id, limit=1
            )
            if rows:
                is_character_run = (
                    self._view_model.character_bench_by_id(rows[0].get("task_id"))
                    is not None
                )
        if self._selection.kind == "run_group" and self._selection.id and not is_character_run:
            self.register_footer_shortcuts(
                source="evals-grid",
                shortcuts=(
                    ("l", "lens"), ("b", "baseline"), ("s", "sort"), ("e", "export"),
                ),
            )
        else:
            self.clear_footer_shortcuts(source="evals-grid")

    @on(LibraryRail.EvalsSelectionChanged)
    def _on_library_selection_changed(
        self, event: LibraryRail.EvalsSelectionChanged
    ) -> None:
        event.stop()
        # task-15475: the RAIL knows whether it just mutated its own rows (an
        # import, a "+ New bench") or merely reported a row press; forward its
        # answer rather than assuming "came from the rail" means "unchanged".
        self.select(
            kind=event.selection.kind,
            id=event.selection.id,
            rail_dirty=event.rail_dirty,
        )

    @on(BenchEditor.Saved)
    def _on_bench_editor_saved(self, event: BenchEditor.Saved) -> None:
        """A successful `BenchEditor` Save re-selects the same bench --
        `select()`'s recompose reloads the form from what `save_bench`
        actually persisted (see `BenchEditor.Saved`'s own docstring for why
        that can differ from what was typed, e.g. `_clean_task_name`'s
        control-character strip), and refreshes the rail row and inspector
        alongside it for free, the same way any other selection change
        does."""
        event.stop()
        self.select(kind="bench", id=event.bench_id)

    @on(CharacterBenchEditor.Saved)
    def _on_character_bench_editor_saved(
        self, event: CharacterBenchEditor.Saved
    ) -> None:
        """Mirrors ``_on_bench_editor_saved`` exactly, for the character-
        bench editor's own ``Saved`` message: re-selecting reloads the
        form from what ``save_character_bench`` actually persisted and
        refreshes the rail row and inspector alongside it, the same way
        any other selection change does."""
        event.stop()
        self.select(kind="character_bench", id=event.bench_id)

    @on(LibraryRail.NewCharacterBenchRequested)
    def _on_new_character_bench_requested(
        self, event: LibraryRail.NewCharacterBenchRequested
    ) -> None:
        """Creates a draft character-probe bench bound to the newest probe
        set and selects it -- the character-bench mirror of
        ``LibraryRail._create_new_bench``. Handled here rather than
        in-widget in ``library_rail.py`` (see ``NewCharacterBenchRequested``'s
        own docstring): a plain DB write, exactly like ``_create_new_bench``
        -- no network call, so no worker.

        **Why a target is resolved here, unlike a draft WORD bench (whose
        ``target_ids`` starts empty and is filled in later via
        ``BenchEditor``'s Add-target picker).** The character-bench editor
        (task-1691 phase 2, Task 4) carries its target list through
        verbatim with no Add/Remove control of its own -- bench creation is
        the ONLY place ``target_ids`` is ever populated for this bench
        type. Leaving it empty here would ship a bench with no path to
        ever becoming runnable, so ``sample_bench.resolve_unsteered_llama_
        cpp_target`` is called with ``create=True``: reuse an existing
        UNSTEERED ``llama_cpp`` ``eval_models`` row if one exists, else
        mint a fresh (also unsteered) one from the configured endpoint if
        ``app_config`` names one -- still a plain DB write, no network call
        (like its sibling ``resolve_sample_target``, this never dials out;
        it only ever reads config and writes a row naming an endpoint).

        **Whole-branch review Critical 1 (fix round): this deliberately
        does NOT call ``resolve_sample_target`` -- the one-click sample
        bench's own resolver -- even though it once did.** That function
        reuses ``list_models(provider="llama_cpp")[0]`` -- the newest
        ``llama_cpp`` row, whatever it is -- with no regard for its
        ``config``, so it could just as easily hand back a row
        ``bench_editor.py``'s "+ New target" mini-form steered with a
        ``prefix`` or ``system_prompt``. Both are silently wrong for THIS
        caller: a ``prefix``-steered row makes every run attempt raise in
        ``targets.resolve_target`` (a probe is chat-shaped, with no slot
        for a literal prefix), permanently stranding the bench -- this
        editor has no way to change its target afterward -- and a
        ``system_prompt``-steered row has its steering composed ahead of
        the card's own system prompt by ``runner.py``, silently
        contaminating every probe conversation the bench exists to
        observe. See ``resolve_unsteered_llama_cpp_target``'s own
        docstring for the full account.

        If NEITHER an existing unsteered row nor a configured endpoint is
        available, ``target_ids`` stays empty and the created bench
        genuinely cannot be made runnable through this UI alone --
        ``config.py`` ships a default ``llama_cpp`` API URL, so this is the
        near-universal case in practice (mirrors ``resolve_sample_
        target``'s own "near-universal" note), but it is a real, reachable
        gap this task cannot close: this editor offers no way to add a
        target after the fact. The toast below names that state explicitly
        rather than claiming an unconditional success.

        ``character_ids`` starts empty (a draft has no characters picked
        yet -- that is the editor's job) and ``strict=False`` is required
        to construct a ``CharacterProbeConfig`` with it: the strict
        (default) path raises "needs at least one character" at
        construction, exactly the validation a genuine Save should keep
        (see ``CharacterProbeConfig``'s own docstring) but that a bare
        DRAFT must not be blocked by.
        """
        event.stop()
        db = self._view_model.db
        if db is None:
            self.app_instance.notify(
                "The evaluation service is unavailable.", severity="error"
            )
            return
        probe_sets = self._view_model.probe_sets()
        if not probe_sets:
            # Defensive only: `library_rail.py`'s own button is disabled
            # whenever this is true (see `_new_bench_actions`).
            self.app_instance.notify(
                "Import or create a probe set first.", severity="warning"
            )
            return
        # `probe_sets()` is `datasets()`'s own newest-first order (see
        # `EvalsViewModel.datasets`/`_create_new_bench`'s identical note),
        # filtered -- so the first entry IS "the newest probe set" with no
        # extra sort needed.
        probe_set = probe_sets[0]
        app_config = self._current_app_config()
        target = sample_bench.resolve_unsteered_llama_cpp_target(
            self._view_model, app_config, create=True
        )
        target_ids = (target["id"],) if target is not None else ()
        config = CharacterProbeConfig(
            name=_unique_name("Untitled character bench"),
            probe_set_id=str(probe_set.get("id")),
            character_ids=(),
            target_ids=target_ids,
            strict=False,
        )
        try:
            bench_id = save_character_bench(db, config)
        except Exception as exc:
            logger.opt(exception=True).warning("Could not create character bench.")
            # markup=False: `exc` can carry a name collision naming the
            # bench itself, the same free-text hazard `_create_new_bench`'s
            # own `_notify` call already guards against for word benches.
            self.app_instance.notify(
                f"Could not create character bench: {exc}",
                severity="error",
                markup=False,
            )
            return
        probe_set_name = str(probe_set.get("name") or "Untitled probe set")
        if target_ids:
            self.app_instance.notify(
                f"Character bench created against {probe_set_name}.",
                severity="information",
                markup=False,
            )
        else:
            # See this handler's own docstring on why this is a real,
            # reachable state and not merely defensive copy. Fix-round
            # correction: THIS bench, not just "a new one", must be named
            # as the thing that's stuck -- the previous wording only ever
            # suggested creating another bench and never told the user
            # this one cannot run and is safe to delete, which left a
            # one-click-reachable, never-undoable state with no recovery
            # instruction (review finding, this fix round). Deletion
            # itself is real as of this same round -- see
            # `_compose_inspector_pane`'s `"character_bench"` branch.
            self.app_instance.notify(
                "Character bench created, but it cannot be run: no "
                "llama.cpp target is configured. This bench cannot be "
                "made runnable after the fact -- delete it (see the "
                "Delete button below) and create a new one once a "
                "target is configured in Settings.",
                severity="warning",
                markup=False,
            )
        self.select(kind="character_bench", id=bench_id)

    @on(BenchEditor.CreateTargetRequested)
    async def _on_bench_create_target_requested(
        self, event: BenchEditor.CreateTargetRequested
    ) -> None:
        """Creates a real `eval_models` row for `bench_editor.py`'s
        "+ New target" mini-form -- ALWAYS rendered there (task-1611 T2),
        not only in the zero-`llama_cpp`-models state -- and stages it on
        the mounted `BenchEditor`. See that message class's own docstring
        for why `bench_editor.py` cannot make this call itself (the
        source-scan pin against the provider client/runner imports).

        Calls `EvalsDB.create_model` DIRECTLY (task-1611 T2) rather than
        `sample_bench.resolve_sample_target`, which this handler used
        exclusively before this task: that function reuses an already-
        registered `llama_cpp` row FIRST, before ever minting a new one --
        exactly wrong once this control's whole point is minting an
        ADDITIONAL, possibly differently-steered target even when one (or
        several) already exist. `configured_llama_cpp_url`/
        `configured_llama_cpp_model_id` (the same config-only, no-network
        reads `resolve_sample_target` itself uses internally) resolve the
        endpoint and model id instead.

        A blank/whitespace-only `event.name` auto-names via
        `storage._unique_name(sample_bench.BENCH_EDITOR_TARGET_NAME)` --
        the SAME base name/convention the old zero-models-only flow always
        used for its one auto-created row. A NON-blank name is used
        VERBATIM (never uniqued) so an intentional collision surfaces as
        the `ConflictError` it is, rather than being silently suffixed
        into a different row than the one the user asked to create.

        `event.prefix`/`event.system_prompt` are already mutually
        exclusive and already blank-normalized to `None` by
        `bench_editor.py`'s own `_on_create_target_pressed` (only one
        steering `Input` is ever mounted at a time) -- this handler only
        decides which non-`None` one becomes a `config` key; an empty
        `config` (`{}`) is what an unsteered target's row already gets
        everywhere else in this codebase (`EvalsDB.create_model`'s own
        `config or {}` default).

        A plain DB read/write, not a network call -- run inline, no
        worker, mirroring `BenchEditor._on_save_pressed`'s own synchronous
        `save_bench` call just one pane over.
        """
        event.stop()
        db = self._view_model.db
        if db is None:
            return
        app_config = self._current_app_config()
        if sample_bench.configured_llama_cpp_url(app_config) is None:
            self.app_instance.notify(
                "No llama.cpp server is configured; set one in Settings "
                "first.",
                severity="error",
                markup=False,
            )
            return
        model_id = sample_bench.configured_llama_cpp_model_id(app_config) or "default"
        typed_name = event.name.strip() if event.name else ""
        name = event.name if typed_name else _unique_name(sample_bench.BENCH_EDITOR_TARGET_NAME)
        config: dict[str, str] = {}
        if event.prefix:
            config["prefix"] = event.prefix
        if event.system_prompt:
            config["system_prompt"] = event.system_prompt
        try:
            new_id = db.create_model(
                name=name, provider="llama_cpp", model_id=model_id, config=config
            )
        except ConflictError as exc:
            self.app_instance.notify(str(exc), severity="error", markup=False)
            return
        model_row = db.get_model(new_id)
        if model_row is None:
            # Defensive only: create_model just returned this id.
            return
        from textual.css.query import QueryError  # noqa: PLC0415 -- narrow, matches this module's other local imports

        try:
            editor = self.query_one(BenchEditor)
        except QueryError:
            # Defensive only: this handler only ever runs from a press on
            # a button the mounted BenchEditor itself composed.
            return
        await editor.stage_target(model_row)

    @on(LibraryRail.SampleBenchRequested)
    def _on_sample_bench_requested(
        self, event: LibraryRail.SampleBenchRequested
    ) -> None:
        """Creates and runs the one-click sample bench (see
        ``sample_bench.py``). Real DB writes plus a real HTTP call (in
        production) -- run as a worker, never inline in a message handler,
        per CLAUDE.md's "Workers for operations >100ms" rule.

        Two guards cover two different race windows. If two requests are
        already queued before either dispatches, both see
        ``_sample_bench_running`` as ``False`` and both reach
        ``run_worker(exclusive=True, ...)``; it is ``exclusive=True`` that
        protects there, cancelling the second worker's Task before it takes
        its first step, so only one worker body (and one flag-set) ever
        runs. Once a worker IS running and has set the flag, THIS check is
        what stops a later request from calling ``run_worker`` again --
        without it, that call would cancel the already-running worker via
        the same ``exclusive`` group after it has done real work, abandoning
        its in-flight DB rows (see
        ``sample_bench._mark_orphaned_runs_cancelled`` for the cleanup that
        path needs).

        A THIRD check, ``_bench_run_running``, closes a different race: PR
        #1113 review (Qodo, seconding whole-branch review Note 6) found the
        sample-bench worker and the bench-run worker (``_run_bench_worker``,
        started from ``_on_primary_action_pressed``) were only ever guarded
        against THEMSELVES -- each lived in its own ``exclusive`` group, so
        neither worker's ``exclusive=True`` cancelled the other, and a press
        of one while the other was genuinely in flight started two REAL,
        overlapping runs (interleaved toasts, last-wins completion
        ``select()``). The recompose-time UI already disables both controls
        while EITHER flag is set (see ``_primary_action_state``'s and
        ``LibraryRail.sample_bench_running``'s own in-flight branches), but
        that alone does not stop a stale-render/queued-press race from
        reaching this handler -- this cross-check is the same belt this
        function's OTHER two guards already provide for the same-worker
        case, just against the other worker.

        The worker is handed as a CALLABLE, not a pre-built coroutine:
        ``exclusive=True`` cancels the superseded worker's Task before its
        first step, and a coroutine object constructed at the call site is
        then never awaited at all (``RuntimeWarning: coroutine ... was
        never awaited``). Textual only calls the callable when the worker
        actually starts, so in the very race this docstring describes no
        orphan coroutine is created.

        ``self._selection`` is also snapshotted into
        ``self._sample_bench_launch_selection`` HERE, before ``run_worker``
        is even called -- not re-read from ``self._selection`` inside the
        worker, mirroring ``_on_primary_action_pressed``'s own
        ``_bench_run_task_id`` capture and for the identical reason: the
        selection can move before the scheduled worker's first line
        actually runs. The completing worker reads this snapshot to decide
        whether it is still safe to move the selection to the new run
        group, or whether the user has navigated elsewhere and a recompose
        there would yank them (see ``_selection_unmoved_since_launch``,
        task-1482 Task 2).
        """
        event.stop()
        if (
            self._sample_bench_running
            or self._bench_run_running
            or self._character_bench_run_running
        ):
            return
        self._sample_bench_launch_selection = self._selection
        self.run_worker(
            self._create_sample_bench_worker,
            exclusive=True,
            group="evals-sample-bench",
        )

    async def _create_sample_bench_worker(self) -> None:
        """Creates and runs the one-click sample bench (see
        ``sample_bench.create_and_run_sample_bench``).

        On success, ``select(run_group)`` ONLY when
        ``_selection_unmoved_since_launch`` says the screen's current
        selection is still ``self._sample_bench_launch_selection`` (the
        selection snapshotted in ``_on_sample_bench_requested`` at press
        time) or has since moved into the freshly created bench's own run
        groups. Otherwise the run/creation is not lost -- it is still in
        the DB and the Runs section -- but a completing background worker
        must not force a recompose that would yank the user from wherever
        they navigated to mid-flight, e.g. into a half-edited bench editor
        form (task-1482 Task 2's own motivation).
        """
        app_config = self._current_app_config()
        cancel_token = CancelToken()
        self._sample_bench_running = True
        self._sample_bench_cancel_token = cancel_token
        self._set_sample_bench_running_ui()
        result = None
        try:
            result = await sample_bench.create_and_run_sample_bench(
                self._view_model,
                app_config,
                client_factory=self._sample_bench_client_factory,
                progress=self._on_sample_bench_progress,
                cancel_token=cancel_token,
            )
        except asyncio.CancelledError:
            # sample_bench.py's own except-and-re-raise already marked any
            # created run rows "cancelled" before this propagated here --
            # log and let it continue propagating; swallowing a
            # CancelledError is its own bug (Textual's worker bookkeeping
            # needs to observe the real cancellation).
            logger.info("Sample bench worker was cancelled.")
            raise
        except Exception as exc:
            # Type only: persistent exception diagnostics can serialize frame
            # locals, which here include app config and user-authored datasets.
            logger.warning(
                "Sample bench creation failed (exception_category={}).",
                type(exc).__name__,
            )
            # markup=False: `exc` can carry user-controlled text (e.g. a
            # dataset name derived from an imported filename stem) and
            # `notify()` defaults to markup=True -- unbalanced markup in
            # that text (a bare `[/]`) raises MarkupError inside the toast
            # renderer and crashes the whole app. See the identical fix on
            # `_run_bench_worker`'s two notify() calls below.
            self.app_instance.notify(
                f"Could not create the sample bench: {exc}",
                severity="error",
                markup=False,
            )
        finally:
            self._sample_bench_running = False
            self._sample_bench_cancel_token = None
            self._reset_sample_bench_running_ui()
        if result is not None:
            if self._selection_unmoved_since_launch(
                self._sample_bench_launch_selection, result.task_id
            ):
                self.app_instance.notify(
                    "Sample bench created and run.",
                    severity="information",
                    markup=False,
                )
                self.select(kind="run_group", id=result.run_group_id)
            else:
                # The user navigated elsewhere while the run was in flight
                # -- see `_selection_unmoved_since_launch`'s own docstring.
                # The bench and run group both still exist; only the
                # auto-navigate is skipped.
                self.app_instance.notify(
                    "Sample bench created and run — see the Runs section.",
                    severity="information",
                    markup=False,
                )

    def _on_sample_bench_progress(self, done: int, total: int) -> None:
        """``sample_bench.ProgressFn`` -- called synchronously from within
        ``WordBenchRunner.run``'s own coroutine (this worker's, not a
        separate OS thread), so mutating widgets directly here is safe,
        the same way ``_on_grid_cell_focused`` mutates the inspector
        directly rather than needing ``call_from_thread``."""
        self._set_sample_bench_running_ui(done=done, total=total)

    def _set_sample_bench_running_ui(self, *, done: int = 0, total: int = 0) -> None:
        """Disables the "Create sample bench" button and gives it a live
        running label for as long as a run is in flight -- see the class
        docstring note above on why a disabled-but-not-yet-rerendered
        button is not by itself a sufficient guard against a second click,
        only a visible signal that one is already running.
        """
        from textual.css.query import QueryError  # noqa: PLC0415 -- narrow, matches this module's other local imports

        try:
            button = self.query_one("#evals-create-sample-bench", Button)
        except QueryError:
            return
        button.disabled = True
        button.label = (
            f"Running sample bench… ({done}/{total})" if total else "Creating sample bench…"
        )

    def _reset_sample_bench_running_ui(self) -> None:
        """Restores the button after a run ends -- on BOTH the success and
        failure paths.

        TASK-1478 made "Create sample bench" a persistent control (rendered
        at the top of the Benches section regardless of whether any benches
        exist yet -- see ``library_rail.py``'s module docstring, "Creation
        affordances are not empty-only"), so the claim this docstring used
        to make -- that the success path's ``self.select(...)`` recompose
        "replaces this button with the bench's own row" -- is no longer
        true: a fresh ``LibraryRail`` recompose still renders the SAME
        button, at the same id, still needing to be un-disabled and
        re-labelled. The ``QueryError`` guard remains for the case the
        button genuinely isn't in the DOM at all (no configured provider,
        so the rail renders "Open Settings" instead)."""
        from textual.css.query import QueryError  # noqa: PLC0415 -- narrow, matches this module's other local imports

        try:
            button = self.query_one("#evals-create-sample-bench", Button)
        except QueryError:
            return
        button.disabled = False
        button.label = "Create sample bench"

    @on(ResultsGrid.CellFocused)
    def _on_grid_cell_focused(self, event: ResultsGrid.CellFocused) -> None:
        """Forwards a focused grid cell to the inspector pane's
        ``EvalsCellInspector`` -- a targeted ``show_cell()`` call against
        an already-mounted widget, never a screen recompose (see
        ``results_grid.py``'s module docstring for why that distinction
        matters on every arrow-key press)."""
        event.stop()
        from textual.css.query import QueryError  # noqa: PLC0415 -- narrow, matches _footer_status's own local import

        try:
            inspector = self.query_one(EvalsCellInspector)
        except QueryError:
            return
        inspector.show_cell(event)

    @on(Button.Pressed, "#evals-primary-action")
    def _on_primary_action_pressed(self, event: Button.Pressed) -> None:
        """Runs the selected bench -- a WORD bench via ``sample_bench.
        run_existing_bench``, or (task-1691 phase 2 Task 6) a CHARACTER
        bench via ``_run_character_bench_worker``. ``#evals-primary-action``
        is the one physical button both kinds share (``_compose_inspector_
        pane`` only ever composes it for whichever kind is currently
        selected, never both), so this single handler dispatches to
        whichever worker matches ``self._selection.kind`` at press time.

        Mirrors ``_on_sample_bench_requested``'s guard rationale exactly --
        see that method's own docstring for the full three-part
        explanation, repeated here only in brief. If two presses are
        already queued before either dispatches, both see the relevant
        running flag as ``False`` and both reach
        ``run_worker(exclusive=True, ...)``; it is ``exclusive=True`` that
        protects there, cancelling the second worker's Task before it takes
        its first step, so only one worker body (and one flag-set) ever
        runs. Once a worker IS running and has set its flag, THIS check is
        what stops a later press from calling ``run_worker`` again --
        without it, that call would cancel the already-running worker via
        the same ``exclusive`` group after it has done real work,
        abandoning its in-flight DB rows (see
        ``sample_bench._mark_orphaned_runs_cancelled`` for the cleanup that
        path needs). The other two flags close the cross-worker race PR
        #1113 review found: this button, ``#evals-create-sample-bench``,
        and (task-1691 phase 2) a character-bench run all live in separate
        ``exclusive`` groups, so without cross-checking every OTHER flag a
        press here while a DIFFERENT worker is in flight would start a
        second, genuinely overlapping run.

        The selected bench id is resolved and stored on the instance HERE,
        not re-read from ``self._selection`` inside the worker -- selection
        can move (another rail click) while the worker is in flight, and
        the worker must keep running the bench it was actually launched
        against.
        """
        event.stop()
        if (
            self._bench_run_running
            or self._sample_bench_running
            or self._character_bench_run_running
        ):
            return
        selection = self._selection
        if selection.kind == "bench" and selection.id:
            self._bench_run_task_id = selection.id
            self.run_worker(
                self._run_bench_worker,
                exclusive=True,
                group="evals-run-bench",
            )
            return
        if selection.kind == "character_bench" and selection.id:
            self._character_bench_run_task_id = selection.id
            self.run_worker(
                self._run_character_bench_worker,
                exclusive=True,
                group="evals-run-character-bench",
            )
            return
        # Defensive only: `_primary_action_state` keeps the button disabled
        # (so Textual never emits `Pressed` at all) for every selection
        # kind but a found, runnable bench of either type.

    async def _run_bench_worker(self) -> None:
        """Runs ``self._bench_run_task_id`` via
        ``sample_bench.run_existing_bench``. Mirrors
        ``_create_sample_bench_worker`` structure exactly -- see that
        method's own comments for the parts not re-explained here,
        including the "does not auto-select on completion once the user
        has navigated elsewhere" rule (``_selection_unmoved_since_launch``,
        task-1482 Task 2): here the launch selection to compare against is
        always ``EvalsSelection(kind="bench", id=task_id)``, since
        ``_on_primary_action_pressed`` only ever dispatches this worker
        for a selected bench.
        """
        app_config = self._current_app_config()
        task_id = self._bench_run_task_id
        cancel_token = CancelToken()
        self._bench_run_running = True
        self._bench_run_cancel_token = cancel_token
        self._set_bench_run_running_ui()
        result = None
        try:
            result = await sample_bench.run_existing_bench(
                self._view_model,
                app_config,
                task_id,
                client_factory=self._sample_bench_client_factory,
                progress=self._on_bench_run_progress,
                cancel_token=cancel_token,
            )
        except asyncio.CancelledError:
            # run_existing_bench's own except-and-re-raise already marked
            # any of this bench's still-"running" run rows "cancelled"
            # before this propagated here -- log and let it continue
            # propagating; swallowing a CancelledError is its own bug
            # (Textual's worker bookkeeping needs to observe the real
            # cancellation).
            logger.info("Bench run worker was cancelled.")
            raise
        except Exception as exc:
            # Type only: persistent exception diagnostics can serialize frame
            # locals, including the selected dataset id and current app config.
            logger.warning(
                "Bench run failed (exception_category={}).",
                type(exc).__name__,
            )
            # markup=False: `exc` can carry user-controlled text -- e.g.
            # `sample_bench._load_snippets` raises `RuntimeError(f"Dataset
            # {name!r} has no snippets to run.")`, and an imported dataset's
            # name defaults to the imported filename's stem, so a file named
            # `notes[/].txt` puts live markup straight into this string.
            # `notify()` defaults to markup=True; unbalanced markup (a bare
            # `[/]`) raises MarkupError inside the toast renderer and takes
            # down the whole app -- this path was unreachable before this
            # button was wired up (it was always disabled), so it is new
            # here.
            self.app_instance.notify(
                f"Could not run the bench: {exc}",
                severity="error",
                markup=False,
            )
        finally:
            self._bench_run_running = False
            self._bench_run_cancel_token = None
            self._reset_bench_run_running_ui()
        if result is not None:
            launch_selection = EvalsSelection(kind="bench", id=task_id)
            if self._selection_unmoved_since_launch(launch_selection, task_id):
                # markup=False for uniformity with the error toast above --
                # this string is static today, but pinning it keeps the
                # pair consistent if it ever starts interpolating the
                # bench name.
                self.app_instance.notify(
                    "Bench run finished.", severity="information", markup=False
                )
                self.select(kind="run_group", id=result.run_group_id)
            else:
                # The user navigated elsewhere while the run was in flight
                # -- see `_selection_unmoved_since_launch`'s own docstring.
                # The run group still exists; only the auto-navigate is
                # skipped.
                self.app_instance.notify(
                    "Bench run finished — see the Runs section.",
                    severity="information",
                    markup=False,
                )

    @staticmethod
    def _resolved_target_row(db: EvalsDB, target_id: str) -> Mapping[str, Any]:
        """One character-bench target's raw ``eval_models`` row.

        Mirrors ``sample_bench._resolve_targets``'s identical per-target
        lookup (the word-bench side of this exact concern) byte for byte:
        a ``target_id`` with no matching row (deleted after the bench was
        created or last saved) must raise, naming the id, rather than
        letting ``create_probe_run_group``/``CharacterProbeRunner.run``
        receive a hole in the target list -- both validate the ROWS they
        are given (via ``targets.resolve_targets``), but neither can tell
        "no row for this id" apart from "a caller silently dropped one".

        Args:
            db: The evals database handle.
            target_id: One of the bench's ``CharacterProbeConfig.
                target_ids``.

        Returns:
            The ``eval_models`` row, exactly as ``EvalsDB.get_model``
            returns it -- passed straight through to
            ``create_probe_run_group``/``CharacterProbeRunner.run``, which
            do their own validation (id/model_id presence, steering shape)
            via ``targets.resolve_targets``.

        Raises:
            RuntimeError: If no live row matches ``target_id``.
        """
        model = db.get_model(target_id)
        if model is None:
            raise RuntimeError(
                f"Target {target_id!r} could not be resolved — its "
                "eval_models row is missing or was deleted."
            )
        return model

    @staticmethod
    def _mark_character_run_ids(
        db: Optional[EvalsDB], run_ids: Mapping[str, str], status: str
    ) -> None:
        """Best-effort status stamp for every run
        ``_run_character_bench_worker`` created, mirroring ``sample_bench.
        _mark_orphaned_runs_cancelled``'s own ``EvalsDB.update_run_status``
        call and its "log and continue, never let a bookkeeping failure
        mask the real outcome" contract -- called immediately after
        ``create_probe_run_group`` returns (``status="running"``, whole-
        branch review fix round: the in-flight window between that call
        and ``save_conversations`` succeeding was the one outcome this
        worker's own remediation for every OTHER status had not yet
        covered), the success path (``status="completed"``), the
        ``CancelledError`` path (``status="cancelled"``), AND (review
        round 2 -- the general ``except Exception:`` branch is reachable
        with `run_ids` already populated too: ``factory(config)`` failing
        to build a chat callable, ``asyncio.Semaphore(config.
        concurrency)`` raising for a non-positive concurrency, or a plain
        DB I/O failure inside ``save_conversations`` itself are all
        ordinary exceptions, not cancellations) the general failure path
        (``status="failed"``) of that worker.

        Necessary because ``character_probe.storage``/``runner`` (Task 1's
        phase-1 engine) never call ``EvalsDB.update_run_status``
        themselves -- unlike ``WordBenchRunner.run``, which moves each run
        pending -> running -> completed/cancelled on its own. Left
        unstamped, ``EvalsViewModel.run_groups()``'s own pivot falls a
        "pending, nothing running/cancelled/failed" group through to
        "completed" (see that method's own docstring) -- true by
        coincidence for a genuinely successful run, but also true, and
        misleading, for one that never finished, regardless of WHY.

        ``"failed"`` and ``"cancelled"`` are not the same run-level fact
        (one was requested, one wasn't) but ``run_groups()``'s own pivot
        deliberately has no separate group-level "failed" bucket -- its
        ``_has_blocked`` check groups a run-level ``"cancelled"`` OR
        ``"failed"`` status into the identical group-level ``"cancelled"``
        label (see that method's own docstring). Stamping ``"failed"``
        here is therefore both the semantically honest run-level fact AND
        sufficient for the group to stop reading "completed" in the rail
        -- the property review round 1/2 both actually care about.

        Args:
            db: The evals database handle, or ``None`` -- nothing to
                stamp when this worker failed before ever resolving one
                (the earliest possible failure, before any run existed to
                mark).
            run_ids: ``target_id -> eval_runs id``, as returned by
                ``create_probe_run_group`` -- empty (``{}``) when this
                worker failed or was cancelled before that call ever ran,
                in which case there is nothing to stamp either.
            status: ``"running"``, ``"completed"``, ``"cancelled"``, or
                ``"failed"``.
        """
        if db is None or not run_ids:
            return
        for run_id in set(run_ids.values()):
            try:
                db.update_run_status(run_id, status)
            except Exception:
                logger.opt(exception=True).warning(
                    f"Could not mark character bench run {run_id!r} "
                    f"{status!r}."
                )

    async def _run_character_bench_worker(self) -> None:
        """Runs ``self._character_bench_run_task_id`` -- the character-probe
        sibling of ``_run_bench_worker`` just above (see that method's own
        docstring for the parts not re-explained here: the guard
        rationale, the ``_selection_unmoved_since_launch`` completion
        rule, and reusing ``_set_bench_run_running_ui``/``_reset_bench_
        run_running_ui``/``_on_bench_run_progress`` for the SAME physical
        ``#evals-primary-action`` button -- those three helpers are already
        bench-type-agnostic and need no character-bench variant).

        Unlike the word-bench path, there is no single ``sample_bench.
        run_existing_bench``-shaped engine entry point to call: Task 1's
        phase-1 engine exposes its steps directly (``load_character_
        bench``, ``load_probe_set``, ``cards.snapshot_cards``,
        ``storage.create_probe_run_group``, ``CharacterProbeRunner.run``,
        ``storage.save_conversations``), so this method IS the
        character-bench equivalent of ``run_existing_bench``, composed
        inline -- a second Textual-free orchestration module is outside
        this task's declared file list (``evals_screen.py``/
        ``inspector.py`` only).

        **The chat callable is SYNCHRONOUS and must never be awaited
        here.** ``CharacterProbeRunner`` already dispatches every call
        through ``asyncio.to_thread`` internally (see
        ``character_probe.runner``'s own module docstring) -- this method
        hands it a plain callable (from ``self._character_probe_chat_
        factory`` or the production default) and only ever ``await``s
        ``runner.run(...)`` itself, never the callable directly.

        **Cancellation**: ``to_thread`` survives ``Task`` cancellation, so
        a turn already dispatched to a worker thread always runs to
        completion and is recorded -- cancelling only stops SCHEDULING
        further turns/conversations (``character_probe.runner``'s own
        module docstring states this contract; it is not re-implemented
        here). A hard cancellation of THIS worker (Textual's
        ``exclusive=True`` superseding it) can still land before
        ``save_conversations`` ever runs, in which case the run group's
        ``eval_runs`` rows (already written by ``create_probe_run_group``,
        before the runner starts) persist with no results attached.
        Nothing in ``character_probe.storage``/``runner`` (Task 1's
        phase-1 engine) ever transitions ``eval_runs.status`` past its
        ``'pending'`` DB default itself -- unlike ``WordBenchRunner``,
        which moves each run pending -> running -> completed/cancelled on
        its own -- so this method stamps every status transition directly:
        ``"running"`` right after ``create_probe_run_group`` returns
        (whole-branch review fix round -- closes the one in-flight window
        this worker's own remediation for every other outcome had not yet
        covered), ``"completed"`` after ``save_conversations`` succeeds
        (review round 1 fix), ``"cancelled"`` in the ``except
        CancelledError`` branch below, via ``_mark_character_run_ids``
        (mirrors ``sample_bench._mark_orphaned_runs_cancelled``'s own
        ``EvalsDB.update_run_status`` call and its "log and continue,
        never let a bookkeeping failure
        mask the real outcome" contract). Left unstamped,
        ``EvalsViewModel.run_groups()``'s own pivot falls a "pending,
        nothing running/cancelled/failed" group through to "completed"
        (see that method's own docstring) -- true by coincidence for a
        genuinely finished run, but also true, and actively misleading,
        for one that was hard-cancelled with zero results.

        Every error ``cards.snapshot_cards``/``targets.resolve_targets``
        (via ``create_probe_run_group``/``CharacterProbeRunner.run``) can
        raise for an empty ``character_ids``/``target_ids`` -- reachable
        only for a hand-crafted or corrupted row, since
        ``_primary_action_state`` already blocks the button for both
        cases -- surfaces through the SAME broad ``except Exception``
        toast below as every other failure here, never silently
        swallowed.
        """
        task_id = self._character_bench_run_task_id
        cancel_token = CharacterCancelToken()
        self._character_bench_run_running = True
        self._character_bench_run_cancel_token = cancel_token
        self._set_bench_run_running_ui()
        group_id: Optional[str] = None
        db: Optional[EvalsDB] = None
        run_ids: dict[str, str] = {}
        try:
            db = self._view_model.db
            if db is None:
                raise RuntimeError("The evaluation service is unavailable.")
            config = load_character_bench(db, task_id)
            probe_set = load_probe_set(db, config.probe_set_id)
            if self._chacha_db is None:
                # Qodo review (task-1691 phase 2 fix wave): `__init__`
                # documents `self._chacha_db` as `Optional` (see
                # `_resolve_chacha_db`'s own docstring -- it degrades to
                # `None` when `app_instance.chachanotes_db` is absent), but
                # `snapshot_cards` unconditionally calls `chacha_db.get_
                # character_card_by_id(...)` with no `None` guard of its
                # own. Reachable for real: a bench SAVED earlier with
                # `character_ids` (so `_primary_action_state`'s "no
                # characters" gate does not fire and Run is enabled) then
                # RUN in a session where the character database never
                # wired up. Without this guard the first `snapshot_cards`
                # call raises a bare `AttributeError` ("'NoneType' object
                # has no attribute 'get_character_card_by_id'"), which the
                # broad `except Exception` below still catches and
                # reports -- but as that raw, unnamed attribute error
                # rather than a message that tells the user what is
                # actually missing. Same wording `_primary_action_state`'s
                # own new guard uses below, so both surfaces name the
                # identical cause.
                raise RuntimeError("The character card database is unavailable.")
            cards = snapshot_cards(self._chacha_db, list(config.character_ids))
            raw_targets = [
                self._resolved_target_row(db, target_id)
                for target_id in config.target_ids
            ]
            new_group_id, run_ids = create_probe_run_group(
                db, task_id, config, cards, probe_set, raw_targets
            )
            # Whole-branch review, deferred-minor-promoted-to-must-fix: the
            # run rows `create_probe_run_group` just wrote sit at their
            # `'pending'` DB default from here until `save_conversations`
            # succeeds below -- a real, observable window (the runner is
            # about to make one or more provider calls, which can each take
            # seconds). `_mark_character_run_ids`'s own docstring already
            # explains why an unstamped run reads as "completed" through
            # `run_groups()`'s pivot; the "completed"/"cancelled"/"failed"
            # terminal stamps this worker already applies below make a
            # lying-pending IN-FLIGHT group the one state this worker had
            # not yet corrected, inconsistent with its own remediation for
            # every OTHER outcome.
            self._mark_character_run_ids(db, run_ids, "running")
            factory = (
                self._character_probe_chat_factory
                or _default_character_probe_chat_factory
            )
            chat_fn = factory(config)
            runner = CharacterProbeRunner(chat_fn, cancel_token)
            conversations = await runner.run(
                cards,
                probe_set,
                raw_targets,
                config,
                progress=self._on_bench_run_progress,
            )
            save_conversations(db, new_group_id, run_ids, conversations)
            self._mark_character_run_ids(db, run_ids, "completed")
            group_id = new_group_id
        except asyncio.CancelledError:
            # A hard cancellation (Textual's `exclusive=True` superseding
            # this worker) is re-raised, never swallowed -- Textual's
            # worker bookkeeping needs to observe the real cancellation,
            # the same rule `_run_bench_worker`'s identical clause states.
            # `run_ids` is `{}` (its `__init__`-time default, never
            # reassigned) if cancellation landed before `create_probe_run_
            # group` ever ran -- `_mark_character_run_ids` no-ops on an
            # empty mapping, so this is safe to call unconditionally.
            self._mark_character_run_ids(db, run_ids, "cancelled")
            logger.info("Character bench run worker was cancelled.")
            raise
        except Exception as exc:
            # Review round 2 (Important finding): the window between
            # `create_probe_run_group` (populates `run_ids`) and
            # `save_conversations` completing is reachable by an ORDINARY
            # exception, not only cancellation -- `factory(config)`
            # failing to build a chat callable, `asyncio.Semaphore(config.
            # concurrency)` raising for a non-positive concurrency inside
            # `runner.run`, or a plain DB I/O failure inside `save_
            # conversations` itself (`db.get_run`/`update_run`/
            # `store_result`, all real writes). Left unstamped, this run
            # group's rows would stay `'pending'` forever, which `run_
            # groups()`'s own pivot (see `_mark_character_run_ids`'s own
            # docstring) falls through to "completed" -- the exact
            # falsehood the CancelledError branch above already guards
            # against. `"failed"` (not `"cancelled"`): this run was never
            # requested to stop, it genuinely errored -- `run_groups()`'s
            # pivot buckets a `"failed"` run-level status into the SAME
            # group-level `"cancelled"` label a `"cancelled"` run-level
            # status gets (there is no separate group-level "failed"
            # bucket; see that method's own `_has_blocked` check), so this
            # still reads truthfully as "not completed" in the rail even
            # though the run-level row itself records the more precise
            # reason. `run_ids` is `{}` if this exception fired before
            # `create_probe_run_group` ever ran -- `_mark_character_run_
            # ids` no-ops on an empty mapping, so this is safe
            # unconditionally, mirroring the CancelledError branch.
            self._mark_character_run_ids(db, run_ids, "failed")
            # Type only: persistent exception diagnostics can serialize
            # frame locals, which here include the bench's own config and
            # every snapshotted card's full text.
            logger.warning(
                "Character bench run failed (exception_category={}).",
                type(exc).__name__,
            )
            # markup=False: `exc` can carry user-controlled text -- a card
            # name, probe text, or bench name reaching this message via
            # `snapshot_cards`/`create_probe_run_group`'s own error
            # strings. Same hazard `_run_bench_worker`'s identical notify()
            # call documents.
            self.app_instance.notify(
                f"Could not run the bench: {exc}",
                severity="error",
                markup=False,
            )
        finally:
            self._character_bench_run_running = False
            self._character_bench_run_cancel_token = None
            self._reset_bench_run_running_ui()
        if group_id is not None:
            launch_selection = EvalsSelection(kind="character_bench", id=task_id)
            if self._selection_unmoved_since_launch(launch_selection, task_id):
                self.app_instance.notify(
                    "Bench run finished.", severity="information", markup=False
                )
                self.select(kind="run_group", id=group_id)
            else:
                # The user navigated elsewhere while the run was in flight
                # -- see `_selection_unmoved_since_launch`'s own docstring.
                # The run group still exists; only the auto-navigate is
                # skipped.
                self.app_instance.notify(
                    "Bench run finished — see the Runs section.",
                    severity="information",
                    markup=False,
                )

    def _on_bench_run_progress(self, done: int, total: int) -> None:
        """``sample_bench.ProgressFn`` -- called synchronously from within
        ``WordBenchRunner.run``'s own coroutine (this worker's, not a
        separate OS thread), so mutating the button directly here is safe,
        mirroring ``_on_sample_bench_progress``."""
        self._set_bench_run_running_ui(done=done, total=total)

    def _set_bench_run_running_ui(self, *, done: int = 0, total: int = 0) -> None:
        """Disables the primary-action button and gives it a live running
        label for as long as a run is in flight -- see
        ``_set_sample_bench_running_ui``'s own note on why a disabled-but-
        not-yet-rerendered button is only a visible signal, not by itself a
        sufficient guard against a second press."""
        from textual.css.query import QueryError  # noqa: PLC0415 -- narrow, matches this module's other local imports

        try:
            button = self.query_one("#evals-primary-action", Button)
        except QueryError:
            return
        button.disabled = True
        button.label = f"Running… ({done}/{total})" if total else "Running…"

    def _reset_bench_run_running_ui(self) -> None:
        """Restores the primary-action button after a run ends, from
        ``_primary_action_state()`` -- the current selection's own fresh
        label/disabled/tooltip, not a hardcoded constant, since (unlike
        ``_reset_sample_bench_running_ui``'s "Create sample bench") the
        ready-state label here is per-bench (``f"Run {name}"``). A no-op
        (via the same ``QueryError`` guard) on the success path, where
        ``self.select(...)`` immediately recomposes the inspector pane and
        replaces this button entirely -- this only matters on the failure
        path, where the SAME button instance survives and must not be left
        permanently disabled with a stale "Running…" label."""
        from textual.css.query import QueryError  # noqa: PLC0415 -- narrow, matches this module's other local imports

        try:
            button = self.query_one("#evals-primary-action", Button)
        except QueryError:
            return
        label, disabled, tooltip = self._primary_action_state()
        button.disabled = disabled
        button.label = label
        button.tooltip = tooltip

    def _bench_delete_disabled_reason(self, bench_id: Optional[str]) -> Optional[str]:
        """Why ``#evals-delete-bench`` should be disabled for ``bench_id``,
        or ``None`` when it's safe to delete.

        Gated ONLY on ``_bench_run_running``/``_character_bench_run_
        running`` for THIS bench -- unlike ``_primary_action_state``, which
        also blocks while the SAMPLE bench worker is running. That extra
        gate exists there because a completing sample-bench worker
        eventually selects a brand-new bench the primary action could
        otherwise race a second run against; the sample-bench worker never
        touches an *existing* bench id (it creates its own, not-yet-
        selected one) until it finishes, so it must not block deleting
        some OTHER, unrelated, already-selected bench here.
        """
        if bench_id and self._bench_run_running and self._bench_run_task_id == bench_id:
            return "A run of this bench is in flight."
        if (
            bench_id
            and self._character_bench_run_running
            and self._character_bench_run_task_id == bench_id
        ):
            return "A run of this bench is in flight."
        return None

    @on(Button.Pressed, "#evals-duplicate-bench")
    def _on_duplicate_bench_pressed(self, event: Button.Pressed) -> None:
        """Duplicates the selected bench via ``storage.duplicate_bench``
        (Task 3) -- a plain ``eval_tasks`` insert, never a network call, so
        this runs in-widget with no worker, mirroring ``library_rail.py``'s
        ``_create_new_bench``/``_create_new_dataset`` (the same "no worker
        for a bare DB write" convention).

        Catches broad ``Exception``, not ``duplicate_bench``'s own
        narrower ``RuntimeError`` (which it raises only for a missing/
        soft-deleted source) -- controller ruling from Task 3's review: a
        CORRUPT legacy bench (task-1132's lenient ``load_bench`` still
        loads it, but ``BenchConfig``/``save_bench`` downstream can raise
        their own native diagnostic exception for a shape ``load_bench``
        never normalised) must still toast here rather than crash this
        screen, matching every other DB-write handler in this file (see
        ``_run_bench_worker``'s own broad catch above).
        """
        event.stop()
        selection = self._selection
        if selection.kind != "bench" or not selection.id:
            # Defensive only: this button is composed only inside the
            # resolved-bench branch of `_compose_inspector_pane`.
            return
        db = self._view_model.db
        if db is None:
            self.app_instance.notify(
                "The evaluation service is unavailable.", severity="error"
            )
            return
        try:
            new_id = duplicate_bench(db, selection.id)
        except Exception as exc:
            logger.opt(exception=True).warning("Could not duplicate bench.")
            # markup=False: `exc` can carry the source bench's own
            # free-text name -- same hazard `_run_bench_worker`'s own
            # error toast documents.
            self.app_instance.notify(
                f"Could not duplicate the bench: {exc}",
                severity="error",
                markup=False,
            )
            return
        new_bench = self._view_model.bench_by_id(new_id)
        new_name = str(new_bench.get("name")) if new_bench else "the new bench"
        self.select(kind="bench", id=new_id)
        self.app_instance.notify(
            f"Duplicated as {new_name}.", severity="information", markup=False
        )

    @on(Button.Pressed, "#evals-delete-bench")
    def _on_delete_bench_pressed(self, event: Button.Pressed) -> None:
        """Starts the confirm-then-delete flow for the selected bench.

        Dispatches a worker: ``push_screen_wait`` raises ``NoActiveWorker``
        outside one (see ``ConsoleShellScreen.confirm_navigation``'s
        identical note in ``chat_screen.py``). The bench id and name are
        resolved here, before the worker's first line runs -- mirrors
        ``_on_primary_action_pressed``'s own capture-outside-the-worker
        rationale (the selection can move while the confirm dialog is
        still up).

        ``_bench_delete_pending`` guards a race review reproduced directly
        (screen stack depth 2 -> 4): two ``Button.Pressed`` messages queued
        with no intervening ``await`` both reach this synchronous handler
        before either's ``run_worker`` call has taken its first step, so
        without a check-and-set flag BOTH calls pass the (unrelated)
        in-flight-run guard above and each starts its own
        ``_delete_bench_flow`` worker -- pushing two ``ConfirmationDialog``s
        onto the screen stack.

        This is deliberately a plain flag, NOT ``exclusive=True`` on the
        worker below, unlike ``_on_primary_action_pressed``'s identical-
        looking double-press race (see that handler's own docstring for the
        contrast). There, ``exclusive=True`` is correct: Textual cancels a
        superseded worker's ``Task`` before its first step, so only one
        worker body -- and the DB write it performs -- ever runs. Here that
        would be actively wrong: ``_delete_bench_flow`` awaits
        ``self.app.push_screen_wait(...)``, which internally awaits
        ``asyncio.shield(future)`` -- shielding the WAIT itself from
        cancellation, not the widget it already pushed. Cancelling this
        worker's Task via an exclusive group after it has already pushed
        its ``ConfirmationDialog`` would tear down the coroutine waiting on
        that dialog's result while leaving the dialog itself mounted on the
        screen stack -- a user's Confirm/Cancel click would land on a
        dialog whose owning code no longer exists to act on it, a silent
        no-op indistinguishable from a hang. A synchronous flag, checked
        and set here before the FIRST worker is ever dispatched, avoids
        needing to cancel anything: the second queued press sees the flag
        already set and returns before calling ``run_worker`` at all, so
        only one worker -- and one dialog -- is ever created. Cleared in a
        ``finally`` inside ``_apply_bench_deletion`` (see that method's own
        docstring) once the flow fully resolves, whichever way it resolves.
        """
        event.stop()
        selection = self._selection
        # Fix round (review finding): a "character_bench" selection uses
        # this SAME button/handler/flow -- see `_compose_inspector_pane`'s
        # `"character_bench"` branch, which composes `#evals-delete-bench`
        # for it too, Delete-only (no Duplicate -- that engine call,
        # `duplicate_bench`, is word-bench-specific; see this fix round's
        # own report for why closing "can never be deleted" took priority
        # over adding duplication). `EvalsDB.delete_task` (called from
        # `_apply_bench_deletion` below) is a plain soft-delete by id --
        # it does not care what `config_data.bench_type` the row carries.
        if selection.kind not in ("bench", "character_bench") or not selection.id:
            return
        if self._bench_delete_disabled_reason(selection.id):
            # Defensive only: `_compose_inspector_pane` already disables
            # the button for this case, and a disabled Textual `Button`
            # never emits `Pressed`.
            return
        if self._bench_delete_pending:
            return
        self._bench_delete_pending = True
        bench = (
            self._view_model.character_bench_by_id(selection.id)
            if selection.kind == "character_bench"
            else self._view_model.bench_by_id(selection.id)
        )
        name = str(bench.get("name")) if bench else "Untitled bench"
        self.run_worker(
            self._delete_bench_flow(selection.id, name),
            group="evals-delete-bench",
        )

    async def _delete_bench_flow(self, task_id: str, name: str) -> None:
        """Confirms, then applies (via ``_apply_bench_deletion`` below)
        deleting ``task_id``.

        ``escape_markup(name)``: ``ConfirmationDialog.compose`` renders
        ``message`` through a plain ``Label`` (``markup`` left at its
        Textual-matching default of ``True``), so an unescaped bench name
        here would hit the same bare-``[/]``-crashes-the-app hazard
        ``_primary_action_state``'s own ``name`` computation documents.
        """
        confirmed = await self.app.push_screen_wait(
            ConfirmationDialog(
                title="Delete bench?",
                message=f'Delete "{escape_markup(name)}"? This can\'t be undone.',
                confirm_label="Delete bench",
                cancel_label="Cancel",
            )
        )
        self._apply_bench_deletion(bool(confirmed), task_id)

    def _apply_bench_deletion(self, confirmed: bool, task_id: str) -> None:
        """Applies the confirm dialog's own result.

        Public-shaped (a plain ``(confirmed, task_id)`` signature, not
        name-mangled) so tests call this directly with
        ``confirmed=True``/``False``, bypassing the modal (and the worker
        above) entirely -- mirrors ``snippet_editor.py``'s
        ``_handle_import_file_selected`` (the ``FileOpen`` dialog's own
        callback): driving a real modal in a test is expensive, and this
        is the one place the dialog's yes/no decision reaches code.

        The whole body runs inside a ``try/finally`` that clears
        ``_bench_delete_pending`` -- the single-flight guard
        ``_on_delete_bench_pressed`` sets before ever dispatching
        ``_delete_bench_flow`` (see that method's own docstring for the
        race this closes). Every return path here -- cancelled, no DB,
        delete failed, or genuinely completed -- is "the flow is over, a
        fresh press should be allowed again," so the reset lives in
        ``finally`` rather than being duplicated at each ``return``. Tests
        that call this method directly, bypassing ``_on_delete_bench_
        pressed`` entirely, harmlessly reset a flag that was never set.
        """
        try:
            if not confirmed:
                return
            db = self._view_model.db
            if db is None:
                self.app_instance.notify(
                    "The evaluation service is unavailable.", severity="error"
                )
                return
            try:
                db.delete_task(task_id)
            except Exception as exc:
                logger.opt(exception=True).warning("Could not delete bench.")
                self.app_instance.notify(
                    f"Could not delete the bench: {exc}",
                    severity="error",
                    markup=False,
                )
                return
            self.select(kind="none")
            # Provenance rule (task-1482 plan, "Delete vs runs"): deleting a
            # bench does not cascade its run history -- `EvalsDB.delete_task`
            # only soft-deletes the `eval_tasks` row; `list_runs`/`get_run`'s
            # own `JOIN eval_tasks` (unfiltered on `t.deleted_at`) still
            # resolves the runs, and `EvalsViewModel.run_groups()` reads
            # `list_runs()` directly, never `_all_tasks()` (which DOES filter
            # deleted tasks) -- so the Runs section keeps listing them, and
            # opening one still renders the grid. This toast is the only
            # place a user learns that on purpose.
            self.app_instance.notify(
                "Bench deleted. Its runs remain in the Runs section.",
                severity="information",
                markup=False,
            )
        finally:
            self._bench_delete_pending = False

    def lab_header_state(self) -> WorkbenchHeaderState:
        """Return the Evals destination header copy.

        Returns:
            Header state. The status is constant because nothing on this
            screen is a whole-destination readiness signal -- per-target
            readiness is the inspector's job, and a badge that never changes
            would only be decoration wearing a status label.
        """
        return WorkbenchHeaderState(
            title="Evals",
            subtitle="Run and review evaluation jobs.",
            status="ready",
        )

    def compose_lab_rail(self) -> ComposeResult:
        """Yield the library rail.

        A fresh ``LibraryRail`` per compose is deliberate: open/collapsed
        section state lives in ``self._rail_open_sections`` and is shared by
        reference, so it survives the instance being rebuilt.
        """
        yield LibraryRail(
            self._view_model,
            selection=self._selection,
            open_sections=self._rail_open_sections,
            app_config=self._current_app_config(),
            # Whole-branch review: gated on EITHER worker, not just the
            # sample-bench one -- "a run is in flight" is the condition
            # that makes starting a SECOND one (via this button) a stale-
            # button trap, regardless of which worker owns the first run.
            # See `_primary_action_state`'s own in-flight branch just
            # below for the identical rationale on the primary action.
            sample_bench_running=(
                self._sample_bench_running
                or self._bench_run_running
                or self._character_bench_run_running
            ),
            id="evals-library-pane",
        )

    def build_lab_body(self) -> Vertical:
        """Build the detail pane.

        Returns:
            A ``Vertical`` holding this selection's detail widgets. Built as
            a factory, not composed inline, because the frame mounts the body
            after first paint -- and a widget instance would not survive a
            ``recompose=True`` while a factory does.
        """
        return Vertical(
            *self._compose_detail_pane(self._preflight_for_selection()),
            id="evals-detail-pane",
        )

    def compose_lab_inspector(self) -> ComposeResult:
        """Yield the readiness inspector for the current selection.

        Wrapped in ``#evals-inspector-pane`` rather than yielded flat into
        the frame's region: that id is the inspector's stable selector and
        keeps the ``ds-inspector`` surface styling. The old
        ``destination-workbench-pane`` class is dropped -- sizing is the
        frame region's job now.
        """
        yield Vertical(
            *self._compose_inspector_pane(self._preflight_for_selection()),
            id="evals-inspector-pane",
            classes="ds-inspector",
        )

    def _preflight_for_selection(self) -> dict[str, PreflightResult]:
        """The current selection's readiness map, resolved once per selection.

        ``{}`` for every selection kind but ``"bench"`` -- no other kind's
        panes read it. Memoised because the body, rail and inspector are now
        composed by three separate frame hooks at three different times; see
        ``_preflight_cache``.

        Returns:
            Target id -> readiness, or an empty mapping.
        """
        if self._preflight_cache is not None:
            return self._preflight_cache
        selection = self._selection
        if selection.kind != "bench" or not selection.id:
            self._preflight_cache = {}
        else:
            self._preflight_cache = self._view_model.preflight_for_bench(selection.id)
        return self._preflight_cache

    def _compose_detail_pane(
        self, preflight: dict[str, PreflightResult]
    ) -> ComposeResult:
        selection = self._selection
        yield Static("Detail", classes="destination-section evals-pane-title")

        if selection.kind == "bench":
            bench = self._view_model.bench_by_id(selection.id) if selection.id else None
            if bench is None:
                yield Static(
                    "This bench could not be found; it may have been deleted.",
                    id="evals-detail-missing",
                )
                return
            yield BenchEditor(
                self._view_model, selection.id, preflight, id="evals-bench-editor"
            )
            return

        if selection.kind == "character_bench":
            bench = (
                self._view_model.character_bench_by_id(selection.id)
                if selection.id
                else None
            )
            if bench is None:
                yield Static(
                    "This bench could not be found; it may have been deleted.",
                    id="evals-detail-missing",
                )
                return
            # A genuinely SEPARATE widget from `BenchEditor` above -- word
            # benches and character-probe benches never share a detail
            # surface (see `character_bench_editor.py`'s own module
            # docstring). `self._chacha_db` (resolved once in `__init__`)
            # is threaded through here rather than this widget opening
            # `ChaChaNotes_DB` itself -- see that field's own comment.
            yield CharacterBenchEditor(
                self._view_model,
                selection.id,
                self._view_model.character_cards(self._chacha_db),
                id="evals-character-bench-editor",
            )
            return

        if selection.kind == "classic":
            task = (
                self._view_model.classic_task_by_id(selection.id)
                if selection.id
                else None
            )
            if task is None:
                yield Static(
                    "This task could not be found; it may have been deleted.",
                    id="evals-detail-missing",
                )
                return
            yield ClassicTaskDetail(self._view_model, task, id="evals-classic-detail")
            return

        if selection.kind == "dataset":
            dataset = (
                self._view_model.dataset_by_id(selection.id) if selection.id else None
            )
            if dataset is None:
                yield Static(
                    "This dataset could not be found; it may have been deleted.",
                    id="evals-detail-missing",
                )
                return
            if is_probe_set(dataset):
                # Whole-branch review Important 2: `SnippetEditor` is
                # word-bench shaped -- its Import control writes SNIPPET-
                # shaped samples into the selected dataset's own metadata,
                # which corrupts a probe set's `turns`-shaped samples on
                # the very next press (see `ProbeSetDetail`'s own module
                # docstring for the full failure chain). A probe-set
                # selection gets a read-only listing instead, with no
                # import/edit control of any kind.
                yield ProbeSetDetail(
                    self._view_model, dataset, id="evals-probeset-detail"
                )
                return
            yield SnippetEditor(
                self._view_model, selection.id, id="evals-snippet-editor"
            )
            return

        if selection.kind == "run_group":
            group = (
                self._view_model.run_group_by_id(selection.id)
                if selection.id
                else None
            )
            if group is None:
                yield Static(
                    "This run could not be found; it may have been deleted.",
                    id="evals-detail-missing",
                )
                return
            if self._character_run_group(group):
                # task-1691 phase 2 Task 6: before this task, a character
                # bench could never actually run, so this branch was
                # unreachable for that bench type. `ResultsGrid` is WORD-
                # BENCH shaped top to bottom (raw/chat mode, top-K,
                # snippets) -- its own snapshot-shape check would either
                # render a misleading "no snippets or no targets to
                # render" (a character-probe snapshot has no "snippets"
                # key at all) or, if that guard were ever loosened, leak
                # logprobs/top-K vocabulary into a bench type that carries
                # none. A neutral, honest placeholder instead: the
                # conversations from this run ARE saved (see the run
                # toast and this run's own `eval_results` rows via
                # `character_probe.storage.load_conversations`), only the
                # browsing UI for them is Phase 3's own deliverable, not
                # this one's (see the plan's "Not in Phase 2" list).
                yield Static(
                    "This run's conversations were saved. A review view "
                    "for character probe runs is not built yet.",
                    id="evals-detail-character-run-placeholder",
                    markup=False,
                )
                return
            # ResultsGrid renders its own header (bench name, prompt mode,
            # effective K, cell/failure counts) -- see results_grid.py's
            # _render_header -- so no separate name/count Statics are
            # yielded here; that would restate the same facts from a
            # SECOND, unsynchronized source (this pane reads `group` from
            # `EvalsViewModel.run_groups()`'s pivot, the grid reads its own
            # `load_grid` snapshot -- two reads of related but distinct
            # data that must not drift against each other in the UI).
            yield ResultsGrid(
                self._view_model, selection.id, id="evals-results-grid"
            )
            return

        yield Static(
            self._empty_detail_text(),
            id="evals-detail-empty",
            markup=False,
        )

    def _empty_detail_text(self) -> str:
        """Copy for the ``"none"``-selection Detail pane.

        TASK-1076: the old, single wording ("Select a bench, dataset, or
        run in the library rail...") is unactionable at the one moment it
        is guaranteed to show -- a first launch, where the rail has
        nothing to select at all. Distinguishes that genuinely-empty
        library (nothing in any of the three rail sections) from the more
        common "none" case -- a user who deleted their selection, or
        clicked empty rail padding, while real rows still exist -- where
        the original sentence is still the correct instruction.

        The emptiness check itself lives in
        ``EvalsViewModel.library_is_empty()``, not inline here: this
        method reruns on every selection change (``select()`` ->
        ``refresh(recompose=True)``), so a single, minimal-read helper
        matters more here than in a one-shot call site -- see that
        method's docstring for why it costs one task read (not two) and a
        1-row dataset existence check (not a 500-row page).
        """
        if self._view_model.library_is_empty():
            return (
                "Nothing here yet. Create a sample bench in the Catalog "
                "rail to get started — it builds a dataset and a run for "
                "you in one step."
            )
        return (
            "Select a bench, dataset, or run in the Catalog rail to see "
            "its detail here."
        )

    def _compose_inspector_pane(
        self, preflight: dict[str, PreflightResult]
    ) -> ComposeResult:
        yield Static("Inspector", classes="destination-section evals-pane-title")
        selection = self._selection

        if selection.kind == "bench":
            bench = (
                self._view_model.bench_by_id(selection.id) if selection.id else None
            )
            if bench is not None:
                yield EvalsInspector(
                    self._view_model,
                    selection.id,
                    preflight,
                    id="evals-inspector-bench",
                )
                # Duplicate/Delete are composed further down, AFTER
                # `#evals-primary-action` -- see the comment there (task-
                # 1482 Task 7 fix round 1) for why.

        if selection.kind == "character_bench":
            # task-1691 phase 2 Task 6: a genuinely SEPARATE widget from
            # `EvalsInspector` above, never that class reused with an
            # internal branch -- `EvalsInspector` renders logprobs/top-K/
            # canary vocabulary throughout (Readiness, per-target
            # continuations), and this bench type must never grow any of
            # it (see this task's own "no logprobs vocabulary anywhere in
            # character-probe UI" constraint). `CharacterBenchEstimate`
            # carries ONLY the Estimate section -- the one thing this
            # bench type's cost preview needs -- reusing the SAME
            # `#evals-inspector-estimate-calls` id the word-bench pane
            # uses so a caller that only wants "the estimate" can query
            # one selector regardless of which bench type is selected;
            # the two widgets are never mounted at once (mutually
            # exclusive selection kinds), so the shared id is never
            # ambiguous. Gated on a RESOLVED bench, mirroring the "bench"
            # branch above -- Duplicate/Delete are composed further down.
            character_bench_row = (
                self._view_model.character_bench_by_id(selection.id)
                if selection.id
                else None
            )
            if character_bench_row is not None:
                yield CharacterBenchEstimate(
                    self._view_model,
                    selection.id,
                    id="evals-inspector-character-bench",
                )

        if selection.kind == "classic":
            # Classic tasks are read-only in this workbench (see the design
            # spec's "Classic tasks" section and BenchEditor's
            # `ClassicTaskDetail`, which carries the deferral sentence) --
            # no run control is rendered here at all, not even a disabled
            # one; `_primary_action_state()` is never consulted for this
            # kind.
            return

        if selection.kind == "run_group":
            group = (
                self._view_model.run_group_by_id(selection.id)
                if selection.id
                else None
            )
            if group is not None and not self._character_run_group(group):
                # Focused-cell detail (full top-K + probe table), updated
                # by `_on_grid_cell_focused` as the grid's cell cursor
                # moves -- see that handler and results_grid.py's module
                # docstring for why this is a targeted `show_cell()` call,
                # never a recompose. The primary action button below still
                # renders (with its existing "already completed" reason,
                # unchanged from Task 3) beneath it.
                #
                # Excluded for a CHARACTER-probe run group (task-1691
                # phase 2 Task 6): `_compose_detail_pane` never mounts
                # `ResultsGrid` for that case (a plain placeholder instead
                # -- see its own comment), so no `CellFocused` event could
                # ever reach `_on_grid_cell_focused` to update this
                # widget; it would sit forever on its own placeholder text
                # ("...see its full top-K and probe table here"), both a
                # dead control and a leak of top-K vocabulary into a bench
                # type that carries none.
                yield EvalsCellInspector(id="evals-cell-inspector")

        label, disabled, tooltip = self._primary_action_state()
        if disabled and tooltip:
            # TASK-1076: a disabled Textual `Button` never emits `Pressed`
            # -- a click on it produces no toast, no inline message, no
            # state change, which is exactly the "silent no-op" UAT found.
            # `tooltip=` below is real (screen-reader/mouse-hover users
            # still get it) but it is the ONLY place the reason lived
            # before this, and a hover-only explanation is not reachable
            # from a keyboard-only session. Mirrors `EvalsInspector`'s own
            # readiness convention just above (and reachable through the
            # SAME `.ds-status-badge`/`evals-status-blocked` classes a
            # Blocked target row uses, in `_status_css_class`) rather than
            # inventing a second "why can't I do this" vocabulary: a
            # status badge naming the action, plus a callout stating the
            # reason -- always visible, never conditional on a mouse.
            yield Static(
                f"{label}: Blocked",
                id="evals-primary-action-status",
                classes="ds-status-badge evals-status-blocked",
                markup=False,
            )
            yield Static(
                tooltip,
                id="evals-primary-action-reason",
                classes="ds-recovery-callout",
                markup=False,
            )
        yield Button(
            label,
            id="evals-primary-action",
            disabled=disabled,
            tooltip=tooltip,
        )

        # task-1482 Task 7 fix round 1: composed AFTER `#evals-primary-
        # action`, not before it -- the design spec's inspector mock
        # orders these `[ Run bench ]` then `[ Duplicate ]` then
        # `[ Delete ]`, and the original Task 7 placement (right after
        # `EvalsInspector`, ahead of the primary action) inverted that.
        # Still gated on a RESOLVED bench (`bench is not None`, set in the
        # `selection.kind == "bench"` branch above): an unresolvable bench
        # id renders no `EvalsInspector` and, per this same guard, neither
        # of these buttons either -- there is nothing here to duplicate or
        # delete.
        if selection.kind == "bench" and bench is not None:
            yield Button("Duplicate", id="evals-duplicate-bench")
            yield from self._compose_delete_bench_button(selection.id)
        elif selection.kind == "character_bench":
            # Task 5 fix round (review finding): a character bench had NO
            # Duplicate/Delete affordance at all -- combined with the
            # residual no-resolvable-target dead end
            # (`_on_new_character_bench_requested`'s own docstring), a
            # bench created that way could never be deleted, fixed, or
            # hidden through the UI: permanent rail clutter with no
            # recovery path. Delete-only, deliberately: `duplicate_bench`
            # (`word_bench.storage`) loads/rebuilds through `BenchConfig`/
            # `save_bench`, which reject `CharacterProbeConfig`'s stored
            # shape outright -- a character-bench equivalent does not
            # exist yet, and inventing one is a bigger, separate change
            # than closing "can never be deleted" needs. `#evals-delete-
            # bench` is the SAME id/handler word benches use
            # (`_on_delete_bench_pressed` now accepts both kinds; see its
            # own updated comment) -- `EvalsDB.delete_task` is a plain
            # soft-delete by id and does not care about `bench_type`.
            # Reuses `character_bench_row`, resolved once already by the
            # `character_bench` branch above this function's `bench`/
            # `EvalsInspector` block -- both branches share ONE function
            # scope (this is a single generator, not two), so a second
            # `character_bench_by_id` read here would just repeat that
            # exact lookup.
            if character_bench_row is not None:
                yield from self._compose_delete_bench_button(selection.id)

    def _compose_delete_bench_button(self, bench_id: str) -> ComposeResult:
        """Yields ``#evals-delete-bench`` (plus its Blocked-reason status/
        callout, when blocked) for ``bench_id`` -- shared by the word-bench
        and character-bench branches of ``_compose_inspector_pane`` above,
        which differ only in whether Duplicate is ALSO offered alongside
        it (word bench only; see that method's own comment)."""
        delete_reason = self._bench_delete_disabled_reason(bench_id)
        if delete_reason:
            # Mirrors the primary action's own TASK-1076 convention above
            # (a status badge plus an always-visible callout, not a
            # hover-only tooltip -- see that block's comment for the
            # accessibility rationale). Not factored into one shared
            # helper with the primary action: the primary action's own
            # version also folds in the bench's NAME (this button's label
            # never changes).
            yield Static(
                "Delete: Blocked",
                id="evals-delete-bench-status",
                classes="ds-status-badge evals-status-blocked",
                markup=False,
            )
            yield Static(
                delete_reason,
                id="evals-delete-bench-reason",
                classes="ds-recovery-callout",
                markup=False,
            )
        yield Button(
            "Delete",
            id="evals-delete-bench",
            disabled=bool(delete_reason),
            tooltip=delete_reason,
        )

    def _primary_action_state(self) -> tuple[str, bool, str]:
        """Label, disabled, and tooltip-reason for the primary action button.

        A bare "Run bench" against an ambiguous or stale selection is how
        the old screen produced dead-end toasts (see the plan's design
        note) -- every branch here names the concrete object the action
        would run, or states a concrete reason it can't.

        The found-bench branch is the only one that ever enables the
        button -- every other branch (an unresolvable bench, a dataset, a
        completed run group, or no selection at all) stays disabled with
        its own stated reason, since none of those names an object this
        action can actually run. The in-flight branch below overrides ALL
        of those, found-bench included, whenever a run is genuinely in
        progress.
        """
        selection = self._selection

        if (
            self._bench_run_running
            or self._sample_bench_running
            or self._character_bench_run_running
        ):
            # Whole-branch review Important finding: this function used to
            # never consult either running-flag at all, so a rail click
            # during an in-flight run -- `EvalsScreen.select()` always
            # schedules `refresh(recompose=True)`, even for a same-bench
            # reselection -- recomposed the inspector into a FRESH,
            # ENABLED "Run <name>" button. A press there hits
            # `_on_primary_action_pressed`'s own `_bench_run_running`
            # guard and silently no-ops: the exact dead-end-toast/silent-
            # no-op anti-pattern this whole function's naming rule exists
            # to avoid, just reopened by a recompose instead of by a
            # missing press handler. Checked first, before every other
            # branch, so it wins regardless of what's currently selected --
            # including the found-bench branches just below, whose own
            # label this still borrows (escaped) so the button keeps
            # naming its object even while blocked. Extended for
            # task-1691 phase 2 Task 6: a character-bench selection
            # resolves its name via `character_bench_by_id`, since
            # `bench_by_id` only ever resolves WORD benches.
            if selection.kind == "bench" and selection.id:
                bench = self._view_model.bench_by_id(selection.id)
            elif selection.kind == "character_bench" and selection.id:
                bench = self._view_model.character_bench_by_id(selection.id)
            else:
                bench = None
            name = escape_markup(str(bench.get("name") or "Untitled bench")) if bench else None
            return (
                f"Run {name}" if name else "Run Bench",
                True,
                "A bench run is already in flight.",
            )

        if selection.kind == "bench":
            bench = (
                self._view_model.bench_by_id(selection.id) if selection.id else None
            )
            if bench is None:
                return (
                    "Run Bench",
                    True,
                    "The selected bench no longer exists; choose another "
                    "bench to run.",
                )
            # escape_markup: `name` is free-text and reaches TWO markup-
            # parsed surfaces from here -- this tooltip string (both
            # branches below), AND (via this same return value)
            # `Button(label=...)`'s construction in
            # `_compose_inspector_pane` plus the live `button.label = ...`/
            # `button.tooltip = ...` reassignment in
            # `_reset_bench_run_running_ui`. `Content.from_text`'s
            # markup=True default applies on EVERY assignment to a
            # Button's `.label`, not just construction (Textual's
            # `validate_label` reactive validator), so a bare `[/]` in a
            # bench name would raise `MarkupError` and crash the rail --
            # the same hazard class task-1476 fixed for bench-run toast
            # text, and library_rail.py's `_run_group_row_label` fixed for
            # run rows; this closes the last unescaped instance of it in
            # this file. Computed once here, ahead of the target-count
            # check below, so both the found-but-target-less and the
            # runnable branch can name the bench in their label.
            name = escape_markup(str(bench.get("name") or "Untitled bench"))
            # task-1482 fix round 1: a draft bench created via "+ New
            # bench" has `target_ids=()` until the bench editor (Task 6)
            # wires one on. Read straight from the already-loaded row's
            # `config_data` (no extra DB call -- `list_tasks`/`bench_by_id`
            # already parsed it) rather than `storage.load_bench`, which
            # this function has never otherwise needed. Without this
            # guard, pressing "Run" reached `run_existing_bench` with zero
            # targets, which "completed" an EMPTY run group -- the exact
            # dead-end-toast pattern this function's own naming rule
            # exists to prevent, just reopened one step further downstream
            # ("Bench run finished." followed by "This run could not be
            # found"). Wording matches the readiness panel's own "No
            # targets configured yet." (inspector.py/bench_editor.py) for
            # the same state, so the vocabulary stays consistent across
            # the two surfaces.
            target_ids = (bench.get("config_data") or {}).get("target_ids") or []
            if not target_ids:
                # task-1612: staging a target in the bench editor's Add
                # picker does NOT touch this row's persisted `target_ids`
                # -- only Save does (see `bench_editor.py`'s own module
                # docstring: staged targets are form state until Save
                # writes them via `save_bench`). Without naming Save here,
                # a user who has just staged one reads this tooltip as
                # stale or wrong, since it still says "no targets yet"
                # while one is visibly staged in the editor.
                return (
                    f"Run {name}",
                    True,
                    "This bench has no targets yet; add one in the bench "
                    "editor and Save.",
                )
            return (
                f"Run {name}",
                False,
                f"Runs {name} against its configured targets.",
            )

        if selection.kind == "character_bench":
            # A SEPARATE branch from "bench" above, never folded into it:
            # `bench_by_id` only ever resolves WORD benches (see its own
            # docstring), so a character-bench selection id would never
            # match there. `_compose_inspector_pane` composes no
            # `EvalsInspector` for this kind (see that method -- neither
            # of its `if` branches matches `"character_bench"`, so it
            # falls straight through to this function with no readiness
            # panel above it) -- deliberately: that panel's whole
            # vocabulary (top-K, logprobs, canary) belongs to the
            # word-bench world and would be a lie about what a character
            # probe measures.
            bench = (
                self._view_model.character_bench_by_id(selection.id)
                if selection.id
                else None
            )
            if bench is None:
                return (
                    "Run Bench",
                    True,
                    "The selected bench no longer exists; choose another "
                    "bench to run.",
                )
            name = escape_markup(str(bench.get("name") or "Untitled bench"))
            config_data = bench.get("config_data") or {}
            character_ids = config_data.get("character_ids") or []
            if not character_ids:
                # Reachable for every draft this program's own "+ New
                # character bench" creates (task-1691 phase 2, Task 5):
                # character_ids starts empty on purpose, since picking
                # characters is the editor's job, not the creation
                # button's. "card" (not just "character"): the editor's
                # own picker and section heading both use "character
                # card"/"Characters" -- matching that vocabulary here.
                return (
                    f"Run {name}",
                    True,
                    "This bench has no characters yet; pick at least one "
                    "character card in the editor.",
                )
            target_ids = config_data.get("target_ids") or []
            if not target_ids:
                # Deliberately NOT the word-bench branch's "add one in the
                # bench editor and Save" wording: the character-bench
                # editor (Task 4) has no Add-target control at all --
                # target_ids is set ONLY at creation time (see
                # `_on_new_character_bench_requested`'s own docstring).
                # Telling a user to do something this editor cannot do
                # would be a dead-end instruction dressed up as help; the
                # honest remedy is to recreate the bench once a target is
                # resolvable.
                return (
                    f"Run {name}",
                    True,
                    "This bench has no targets yet; configure a local "
                    "llama.cpp provider in Settings, then create a new "
                    "character bench.",
                )
            if self._chacha_db is None:
                # Qodo review (task-1691 phase 2 fix wave): characters and
                # targets are both present -- the only remaining reason
                # this bench cannot actually run is a card database that
                # never wired up (`_resolve_chacha_db` degrades to `None`
                # rather than raising -- see its own docstring). Checked
                # LAST, only once every other precondition already passed,
                # so this branch never shadows the "no characters"/"no
                # targets" messages just above with a less specific one --
                # this is the exact reachable state the finding names: a
                # bench saved earlier WITH characters, reopened in a
                # session where the character database is unavailable.
                # Without this guard the button would read as fully
                # runnable and only fail once `_run_character_bench_
                # worker`'s own matching guard caught it mid-run; named
                # with the identical wording that guard uses, so a user
                # sees the same cause whichever surface they hit first.
                return (
                    f"Run {name}",
                    True,
                    "The character card database is unavailable; this "
                    "bench cannot be run until it is.",
                )
            # task-1691 phase 2 Task 6: characters and targets are both
            # present, so this bench can actually run -- `_on_primary_
            # action_pressed` now dispatches `_run_character_bench_worker`
            # for `selection.kind == "character_bench"`. Wording mirrors
            # the word-bench ready branch's own tooltip exactly
            # ("Runs {name} against its configured targets.") for the same
            # naming-the-object convention this whole function follows.
            return (
                f"Run {name}",
                False,
                f"Runs {name} against its configured targets.",
            )

        # No "classic" branch: `_compose_inspector_pane` never calls this
        # function for a classic-task selection at all -- classic tasks
        # are read-only (see `ClassicTaskDetail`'s deferral sentence) and
        # get no run control, not even a disabled one.

        if selection.kind == "dataset":
            dataset = (
                self._view_model.dataset_by_id(selection.id) if selection.id else None
            )
            if dataset is not None and is_probe_set(dataset):
                # Whole-branch review Important 3 (fix round): a probe set
                # is bound to a bench via "+ New character bench", never
                # "+ New bench" (that button now deliberately filters
                # probe sets out -- see `library_rail._create_new_bench`'s
                # own updated docstring), and that control binds to the
                # NEWEST probe set, not necessarily the one selected here
                # -- unlike a word bench's "+ New bench", which DOES bind
                # to the currently-selected dataset. This branch must not
                # claim otherwise.
                return (
                    "Run Bench",
                    True,
                    "Datasets are run from within a bench; use + New "
                    "character bench in the Catalog rail to create one "
                    "(binds to the newest probe set).",
                )
            return (
                "Run Bench",
                True,
                # task-1482: names the concrete fix ("+ New bench" in the
                # Catalog rail creates a bench bound to THIS dataset)
                # instead of the old, more general "select a bench that
                # uses this dataset instead" -- which presupposed one
                # already existed, leaving a genuine dead end for a
                # dataset with no bench yet.
                "Datasets are run from within a bench; use + New bench in "
                "the Catalog rail to create one against this dataset.",
            )

        if selection.kind == "run_group":
            return (
                "Run Bench",
                True,
                "This run has already completed; select a bench to start a "
                "new run.",
            )

        return (
            "Run Bench",
            True,
            "Select a bench in the Catalog rail to run it.",
        )

    def save_state(self):
        """Save evals screen state."""
        return super().save_state()

    def restore_state(self, state):
        """Restore evals screen state."""
        super().restore_state(state)

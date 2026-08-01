"""Detail-pane content for a selected bench: the word bench editor, and
the classic (non-word-bench) task's read-only detail.

Mounted by ``evals_screen.py``'s ``_compose_detail_pane`` in place of the
inline ``Static`` fields it used to yield directly (Task 3's placeholder
bench/classic branches) -- see that module's own docstring for why no
``Screen`` subclass is mounted anywhere here.

Readiness renders from ``word_bench.storage.load_run_preflight``'s stored
``preflight`` mapping (via ``EvalsViewModel.preflight_for_bench``), never
recomputed here. That map is resolved ONCE per selection by
``evals_screen.py`` and passed into ``BenchEditor`` as a constructor
argument -- see ``BenchEditor.__init__`` -- rather than this widget calling
``preflight_for_bench`` itself, so a bench selection does not read the same
run-group snapshot twice (once for this pane, once for ``inspector.py``'s).
Neither this module nor ``inspector.py`` imports the HTTP capture client or
the runner that drives it -- a source-scan test in
``Tests/UI/test_evals_bench_editor.py`` pins that neither module can reach
a provider at all, not just that today's ``compose()`` happens not to
call one.

Task 5 (task-1482): ``BenchEditor`` becomes an editable form for name,
description, prompt mode, top-K, and probes. The dataset stays read-only
permanently -- ``save_bench`` has no ``dataset_id`` parameter, see its own
docstring. Editing is display-only until Save: no field posts or reacts
to a live ``Changed`` message, so there is no watcher to accidentally trip
on the Select-posts-Changed-on-mount trap this codebase has hit before.
Save reads every widget fresh, builds a ``BenchConfig``, and persists via
``save_bench``. On failure (``ValueError`` -- either this module's own
top-K parse, ``BenchConfig`` validation, or ``Evals_DB.InputError`` from a
blank/control-char name; or ``Evals_DB.ConflictError`` from a name
collision) the error renders in-place in ``#evals-bench-form-error`` and
NOTHING recomposes -- every other field keeps exactly what the user
typed. On success this widget posts ``Saved``; ``evals_screen.py`` handles
that by calling its own ``select(kind="bench", ...)``, which recomposes
from the freshly persisted row (picking up anything ``save_bench``'s own
cleaning -- e.g. ``_clean_task_name``'s control-character strip -- changed
from what was typed).

Task 6 (task-1482): targets become editable too, but through a SEPARATE
mechanism from every field above -- a staged ``self._staged_target_ids``
list, mutated in place by per-row ``Remove`` buttons and an Add picker
over ``EvalsViewModel.llama_targets()`` (``db.list_models(provider=
"llama_cpp")``), and read back verbatim by ``_on_save_pressed`` in place
of the "CURRENT stored ``target_ids``" this docstring used to describe
before this task. Add/Remove mutate ONLY ``#evals-bench-targets-section``
(``remove_children()`` + ``mount_all()``, built by
``_build_targets_section``) rather than a whole-widget recompose -- a
recompose would discard whatever the user has typed into Name/
Description/Top-K/Probes above, exactly the state loss the "display-only
until Save" paragraph above exists to avoid. A duplicate add (the target
id is already staged) is rejected inline with the exact text ``"Target
already on this bench."``, through the same ``#evals-bench-form-error``
callout Top-K/name failures use. ``stage_target()`` is the targeted
(non-recompose) call the screen makes once it has created a row this
module asked for via ``CreateTargetRequested`` -- see that message's own
docstring for why this module cannot create the row itself.

Task-1610: ``BenchEditor.is_dirty()`` reports whether the mounted form (the
five fields above, the staged target list, and -- task-1611 T2 fix round
1 -- the "+ New target" mini-form's own typed-but-not-yet-created Name/
steering text; task-1710 adds a sixth field, the per-cell continuation
opt-in checkbox) differs from ``self._loaded_config`` -- read by
``evals_screen.py``'s ``_selection_unmoved_since_launch`` so a run/sample-
bench worker completing while this editor holds unsaved edits degrades to
a toast instead of calling ``select()``, which would otherwise recompose
this whole widget and silently discard everything not yet Saved.

Task-1611 whole-branch review fix round (documented, not fixed -- judged
not to stay cleanly contained): a SUCCESSFUL Save is the one place this
protection does not reach. ``Saved`` triggers ``evals_screen.py``'s own
``select()``, which builds a genuinely NEW ``BenchEditor`` instance from
storage -- the mini-form's own ``self._pending_target_*`` has no path
from the old instance to the new one, so any typed-but-never-created
Name/steering text is silently gone the moment Save succeeds, even
though ``is_dirty()`` itself would have called that exact text worth
protecting one line earlier. Threading it through ``Saved`` -> the
screen -> the next ``BenchEditor``'s constructor was considered and
rejected: ``select()`` is this screen's GENERIC recompose entry point,
shared by every selection-kind change, not a Save-specific hook, and
correctly scoping carried state to only the bench just saved (never
leaking into a later, unrelated selection) is real surface area for a
strange, hard-to-notice bug in exchange for a minor convenience. This is
a deliberate boundary: Revert discarding unsaved state IS what "revert"
means; this is the one place a genuine SUCCESS does the same thing.

Task-1611 T2: a target's STEERING (``storage.model_steering`` -- a raw-
mode ``prefix`` or a chat-mode ``system_prompt``, read out of the target's
own ``eval_models.config``) is now a real, reachable, db-backed thing, not
merely a wired-but-dead seam (see ``_resolve_bench_targets``'s own
docstring). The "+ New target" mini-form (``_build_create_target_control``)
therefore renders ALWAYS in the targets section -- unlike Task 6's
``#evals-bench-create-target``, which rendered ONLY when zero ``llama_cpp``
rows existed: a bench author may want an ADDITIONAL, differently-steered
target even when one (or several) already exist, and steering is
IMMUTABLE per row (``model_steering``'s own docstring: no ``update_model``
exists, so a differently-steered variant is always a new row). Zero-models
keeps its old behavior as the degenerate case of this same, now-unified
control -- the Add picker simply has nothing to offer, per
``_build_target_add_control`` returning ``None``.

The mini-form is a Name ``Input`` (``#evals-target-name``, optional -- a
blank value auto-names on the screen side) plus ONE steering ``Input``,
picked by the CURRENT prompt mode: ``#evals-target-prefix`` for raw,
``#evals-target-system-prompt`` for chat -- never both, mirroring
``Target``/``model_steering``'s own one-field-per-mode contract. A prompt-
mode flip (``_on_prompt_mode_changed``) swaps which one is mounted via the
SAME targeted ``_refresh_targets_section`` rebuild Add/Remove/
``stage_target`` already use -- never a whole-widget recompose. Because
that rebuild tears down and rebuilds ``#evals-bench-targets-section``
wholesale, the mini-form's own typed Name/steering text would otherwise be
discarded on every Add/Remove/mode-flip; ``_capture_pending_target_form``/
``self._pending_target_*`` persist it across exactly those rebuilds, the
same state-loss concern Task 5's "display-only until Save" paragraph
raises for the OUTER fields, applied one level down. A successful create
(``stage_target``) resets that pending state back to blank -- a fresh
form for whatever gets created next, and no accidental same-name resubmit.

Target rows now also render a short steering suffix (`_build_target_row`)
-- `` · prefix: <preview>`` (whitespace made visible via the ␣ marker
convention, reusing ``snippet_editor.render_snippet_cell``) or
`` · system prompt set`` -- so a bench with multiple, differently-steered
variants of the same underlying model can actually be told apart at a
glance; an unsteered row's label is unchanged.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from rich.markup import escape as escape_markup
from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Checkbox, Input, Select, Static, TextArea

from ...DB.Evals_DB import ConflictError, EvalsDB
from ...Evals.word_bench.models import BenchConfig, PreflightResult, Target
from ...Evals.word_bench.storage import load_bench, model_steering, save_bench
from .evals_state import EvalsViewModel
from .snippet_editor import render_snippet_cell

#: Verbatim. The design spec's own classic-task copy
#: (`2026-07-25-evals-console-rebuild-design.md`, "Classic tasks" section) --
#: asserted byte-for-byte by
#: ``test_classic_task_detail_shows_run_history_and_deferral_sentence``.
#: Launching a classic task from this workbench is a deliberate scope
#: decision, not an omission still to be wired; do not reword this into a
#: promise of a future date.
CLASSIC_TASK_DEFERRAL_SENTENCE = "Running classic tasks is not available in this slice."

#: Verbatim. Task 5's own pinned error string for an unparseable or
#: sub-1 top-K value -- asserted exactly by
#: ``test_top_k_parse_failure_renders_the_pinned_callout``.
TOP_K_ERROR_TEXT = "Top-K must be a whole number of 1 or more."

#: Verbatim. The per-cell continuation opt-in's own ``Checkbox`` label
#: (task-1710) -- pinned by
#: ``test_capture_continuations_checkbox_reflects_the_loaded_config``.
#: States the cost plainly, in the label itself (task-1710's own
#: instruction: "the label/tooltip should say that plainly (e.g. that it
#: adds one request per cell)") rather than hiding it behind a hover-only
#: tooltip a keyboard-only user would never see -- the fuller nuance
#: (chat mode is free, this is stored with the run and shown for a
#: focused cell, off by default) lives in ``CAPTURE_CONTINUATIONS_
#: TOOLTIP`` for a mouse user, never as the ONLY place the cost is
#: stated. Kept to ONE short sentence, mounted ``compact=True`` (see
#: ``compose()`` below) -- this targets section's own docstring already
#: documents this pane's small, fixed vertical budget at a realistic
#: viewport (``_build_create_target_control``'s "maximally-compact
#: shape" paragraph); a bordered, multi-row checkbox plus a second,
#: separate cost sub-line was tried first and pushed the targets
#: section's own Add/Create controls off the bottom of a 160x45 terminal,
#: confirmed live, not merely reasoned about.
CAPTURE_CONTINUATIONS_LABEL = "Capture a continuation per cell (raw mode: +1 request/cell)"

#: Verbatim tooltip, the fuller cost/content explanation for a mouse user
#: hovering the checkbox -- never the ONLY place this cost is stated (see
#: ``CAPTURE_CONTINUATIONS_LABEL``'s own docstring: the label itself
#: already names the raw-mode cost). Mirrors ``dataset_static``'s own
#: tooltip precedent above (a fuller explanation of a field's own always-
#: visible claim) and this module's other verbatim-string convention.
CAPTURE_CONTINUATIONS_TOOLTIP = (
    "When on, every measured cell also captures a short sample of what "
    "the model says after this snippet, recorded with the run and shown "
    "for a focused cell. Raw mode issues one separate, extra request per "
    "cell (snippets × targets), roughly doubling this run's call count "
    "and time -- that request never perturbs the measured distribution. "
    "Chat mode salvages the continuation from the response already made, "
    "at no extra cost. Off by default so an existing bench's cost never "
    "changes without being explicitly turned on."
)

#: Verbatim. The raw-mode steering field's label in the "+ New target"
#: mini-form (task-1611 T2) -- pinned exactly by
#: ``test_steering_field_label_matches_the_current_prompt_mode``.
PREFIX_FIELD_LABEL = "Prefix (optional — leading whitespace preserved)"

#: Verbatim. The chat-mode steering field's label in the "+ New target"
#: mini-form (task-1611 T2) -- sibling of ``PREFIX_FIELD_LABEL`` above.
SYSTEM_PROMPT_FIELD_LABEL = "System prompt (optional)"

#: The row table's steering-preview cap (task-1611 T2): long enough to be
#: useful, short enough that a steered row never wraps onto a second line
#: in the table's own single-``Static``-per-row layout (see
#: ``_build_target_row``).
_STEERING_PREVIEW_MAX_LEN = 40


def _steering_preview_text(value: str) -> str:
    """A single-line, length-capped source string for a steering value's
    row-table preview, BEFORE it goes through ``render_snippet_cell``'s own
    ␣-marker convention for leading/trailing/interior-run whitespace.

    ``render_snippet_cell``'s own whitespace classifier only flags a RUN of
    2+ whitespace characters as anomalous (``snippet_editor.py``'s
    ``_INTERIOR_RUN_RE``) -- a single embedded ``"\\n"`` would slip through
    untouched and render as a literal line break inside the row's Static,
    breaking this row's single-line contract. Replaced with a visible "⏎"
    marker here instead of being silently dropped, so a steering value that
    happens to carry one (never possible to TYPE into the single-line
    ``Input`` this form uses, but not excluded for a row created some other
    way) is still an honest, if unusual, preview rather than a corrupted
    one.
    """
    single_line = value.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "⏎")
    if len(single_line) > _STEERING_PREVIEW_MAX_LEN:
        return single_line[:_STEERING_PREVIEW_MAX_LEN] + "…"
    return single_line


def _target_status_text(preflight: dict[str, Any], target_id: str) -> str:
    result = preflight.get(target_id)
    if result is None:
        # The bench has never run, or this target was added after the
        # last run -- there is no stored verdict to read, and rendering
        # one of Ready/Unavailable/Blocked here would be a claim no
        # preflight ever made.
        return "Not yet checked"
    return result.status_label


def _parse_probes_text(probes_text: str) -> tuple[str, ...]:
    """Splits a probes ``TextArea``'s raw text into one probe per line,
    dropping only ZERO-LENGTH lines -- a whitespace-only line (e.g. a lone
    ``" "``) is kept byte-exact. See ``BenchEditor._on_save_pressed``'s own
    inline comment for the full rationale (a user pressing Enter after the
    last probe, or leaving a blank line, produces a genuine zero-length
    line that ``BenchConfig`` would otherwise carry all the way through a
    run as a meaningless empty probe; "whitespace preserved exactly" is a
    claim about a token's CONTENT, so only the zero-length case is special).

    The ONE shared parse between Save (``_on_save_pressed``) and
    ``BenchEditor.is_dirty()`` -- both must agree on what counts as an
    edit, or a probes change Save treats as a no-op could still trip the
    dirty check, or the reverse.
    """
    return tuple(line for line in probes_text.split("\n") if line != "")


def _resolve_bench_targets(db: EvalsDB, target_ids: Sequence[str]) -> list[Target]:
    """Resolves ``target_ids`` to ``Target`` instances via their
    ``eval_models`` rows, for the save-time prompt-mode/target validation
    below -- a target id with no resolvable row (a deleted target, already
    rendered as unresolvable in the target table further down this same
    widget) is skipped rather than raising: it cannot be checked either
    way, and ``save_bench`` itself never rejects a bench for carrying one.
    A row whose OWN ``config`` is corrupt (``model_steering`` raises --
    e.g. hand-edited JSON with both ``prefix`` and ``system_prompt`` set)
    is skipped for the identical reason: this function's job is a best-
    effort mode check, not a data-integrity audit -- ``run_existing_bench``
    (``sample_bench.py``) already surfaces that same corruption as a hard
    error at the point it actually matters, RUN time.

    A thin, LOCAL mirror of ``sample_bench.py``'s own ``_resolve_targets``
    (same ``db.get_model`` lookup, same ``model_steering`` call) rather
    than an import of that private helper: ``sample_bench.py`` imports the
    runner and the HTTP client that drives it, both of which this module's
    own source-scan test pins it must never reach, even transitively
    through an import graph (see the module docstring's own "provider"
    mention above).

    Task-1611 T2: now sets ``prefix``/``system_prompt`` for real, via
    ``storage.model_steering`` -- no longer the wired-but-dead seam this
    docstring used to describe (``eval_models.config`` has carried
    steering since task-1611 T1; this function just did not read it yet).
    The save-time ``is_valid_for_mode`` check below is reachable through a
    genuine, db-backed steered target now, not only through the
    monkeypatched ``Target`` ``test_prompt_mode_switch_revalidates_
    targets_and_names_the_offending_target`` still exercises for a
    hand-built case.
    """
    targets: list[Target] = []
    for target_id in target_ids:
        model = db.get_model(target_id)
        if model is None:
            continue
        try:
            prefix, system_prompt = model_steering(model)
        except ValueError:
            continue
        targets.append(
            Target(
                id=model["id"],
                name=model["name"],
                provider=model["provider"],
                model_id=model["model_id"],
                prefix=prefix,
                system_prompt=system_prompt,
            )
        )
    return targets


class BenchEditor(Vertical):
    """Word bench editor: name, description, prompt mode, top-K, probes,
    and targets (Task 6: Add/Remove via a staged list) are all editable
    (Save/Revert); the dataset (name/sample count, resolved at render time)
    stays read-only permanently -- see the module docstring."""

    class Saved(Message, namespace="bench_editor"):
        """Posted after ``save_bench`` succeeds. Carries the bench's own
        ``eval_tasks`` id (unchanged across a save -- this is always an
        edit, never a create) so the handler can re-select it without
        this widget reaching into ``self.screen`` itself; see
        ``evals_screen.py``'s own handler, which recomposes via its public
        ``select()`` so the reopened form reads back whatever
        ``save_bench``'s own write path actually persisted (e.g.
        ``_clean_task_name``'s control-character strip), not merely what
        was typed.
        """

        def __init__(self, bench_id: str) -> None:
            super().__init__()
            self.bench_id = bench_id

    class CreateTargetRequested(Message, namespace="bench_editor"):
        """Posted when the "+ New target" ``#evals-bench-create-target``
        button is pressed -- task-1611 T2: ALWAYS rendered now, not only
        in the zero-``llama_cpp``-models state (see the module docstring's
        own T2 paragraph). Handled by ``evals_screen.py``, never here --
        creating the row is a real ``EvalsDB.create_model`` write, and
        this module's own source-scan test pins that its source text may
        never even NAME the provider capture client or the runner that
        drives it, even in a comment (see the module docstring's own
        careful phrasing above). The handler reaches the mounted editor
        via ``self.query_one(BenchEditor)`` (only one is ever mounted at a
        time, for the current selection) to call ``stage_target()`` on it
        once the row exists.

        Carries the create-target mini-form's own typed state, read fresh
        by ``_on_create_target_pressed`` at press time:

        Args:
            name: The Name ``Input``'s raw value, exactly as typed (never
                stripped here). ``evals_screen.py``'s handler treats a
                blank or whitespace-only value as "auto-name this", via
                ``storage._unique_name(sample_bench.
                BENCH_EDITOR_TARGET_NAME)`` -- the SAME base name/
                convention the old zero-models-only flow always used for
                its one auto-created row.
            prefix: The raw-mode steering ``Input``'s value, or ``None``
                if that Input was not the one mounted (chat mode) or was
                left blank. Passed through EXACTLY, no ``.strip()`` -- a
                raw-mode prefix's LEADING whitespace is meaningful (it is
                prepended literally to every snippet), so trimming it here
                would silently change what a run actually measures.
            system_prompt: The chat-mode steering ``Input``'s value, under
                the same blank-is-``None``/no-strip rules as ``prefix``
                above.

        At most one of ``prefix``/``system_prompt`` is ever non-``None`` --
        only ONE steering ``Input`` is ever mounted at a time (see
        ``_build_create_target_control``), so this is a construction
        invariant of ``_on_create_target_pressed``, not something this
        class itself checks.
        """

        def __init__(
            self,
            *,
            name: str = "",
            prefix: Optional[str] = None,
            system_prompt: Optional[str] = None,
        ) -> None:
            super().__init__()
            self.name = name
            self.prefix = prefix
            self.system_prompt = system_prompt

    def __init__(
        self,
        view_model: EvalsViewModel,
        bench_id: str,
        preflight: Optional[dict[str, PreflightResult]] = None,
        **kwargs: Any,
    ) -> None:
        """``preflight`` is the bench's readiness map, resolved ONCE by
        ``EvalsScreen`` per selection (see its ``_preflight_for_selection``)
        and passed in here rather than this widget calling
        ``EvalsViewModel.preflight_for_bench`` itself -- ``EvalsInspector``
        needs the identical map for the same selection, and each pane
        calling it independently read the bench's run-group snapshot twice
        on one render (see I2 in the PR 3a fix report). ``None`` (the
        default) falls back to resolving it locally, so a widget
        constructed directly -- as this module's own tests do -- still
        works without a caller threading the map through.
        """
        super().__init__(**kwargs)
        self._view_model = view_model
        self._bench_id = bench_id
        self._preflight = preflight
        #: The config `compose()` most recently loaded -- read back by the
        #: Save handler for the fields this widget does not stage directly
        #: (`dataset_id`, `concurrency`), which must round-trip verbatim.
        #: `None` only when `compose()` bailed out before reaching the form
        #: (no db, or an unreadable row) -- in which case no Save/Revert
        #: button exists for a press to ever reach this attribute through.
        self._loaded_config: Optional[BenchConfig] = None
        #: Task 6: the staged target id list -- FORM STATE like every
        #: other field on this widget (see the module docstring), mutated
        #: in place by the Add/Remove/`stage_target` handlers below and
        #: read back verbatim by `_on_save_pressed`. Reset to whatever
        #: `compose()` loads whenever the whole widget is rebuilt (a fresh
        #: selection, or Revert's re-select). Populated in `compose()`,
        #: not here, so a widget that never composes (no db, unreadable
        #: row) never has one of these that could go stale against nothing.
        self._staged_target_ids: list[str] = []
        #: The preflight map resolved by `compose()` (either the one
        #: passed in, or a freshly resolved one) -- cached so `_build_
        #: targets_section`'s later, targeted re-renders (Add/Remove/
        #: `stage_target`) never need to re-resolve it themselves; per-
        #: target readiness only ever changes on a fresh bench selection,
        #: never mid-edit.
        self._preflight_map: dict[str, PreflightResult] = {}
        #: Task-1611 T2: the prompt mode `_build_create_target_control`
        #: last built the "+ New target" mini-form's steering `Input` for
        #: -- the SOLE source of truth for which one (`#evals-target-
        #: prefix` vs `#evals-target-system-prompt`) is currently mounted.
        #: Set by `compose()` before its first build (never by querying
        #: `#evals-bench-prompt-mode` live -- that Select has not MOUNTED
        #: yet at that point, since `compose()` is still yielding its own
        #: widgets) and kept in sync by `_on_prompt_mode_changed` on every
        #: genuine flip afterward.
        self._last_prompt_mode: Optional[str] = None
        #: Task-1611 T2: the create-target mini-form's own typed Name/
        #: steering text, persisted across a targeted `#evals-bench-
        #: targets-section` rebuild (Add/Remove/a mode flip) that would
        #: otherwise tear down and silently discard it -- see
        #: `_capture_pending_target_form`. Only the ONE steering attribute
        #: matching `self._last_prompt_mode` is ever actually mounted at a
        #: given moment; the other is simply carried along unused until a
        #: flip brings it back. Reset to blank by `stage_target` once a
        #: create actually succeeds -- see that method's own docstring.
        self._pending_target_name: str = ""
        self._pending_target_prefix: str = ""
        self._pending_target_system_prompt: str = ""

    def is_dirty(self) -> bool:
        """True when the mounted form differs from ``self._loaded_config``
        -- i.e. there is unsaved state a recompose would destroy (task-1610:
        a background run/sample-bench worker completing must not force
        ``evals_screen.py``'s own ``select()`` while this is true -- see
        that module's ``_selection_unmoved_since_launch``, which queries
        this method defensively).

        Computed on demand by re-reading the same widgets
        ``_on_save_pressed`` reads (task-1710: now six, the five fields
        plus the per-cell continuation checkbox) and comparing each to
        what ``compose()`` loaded -- no field here posts a live
        ``Changed`` message (see the module docstring's "display-only
        until Save" paragraph), so there is no watcher to drive this
        reactively instead. Probes go through ``_parse_probes_text``, the
        exact same helper Save itself uses, so the two can never disagree
        about what counts as an edit. Target edits are staged directly
        onto ``self._staged_target_ids`` (Task 6's Add/Remove handlers)
        rather than read from a widget, so that list is compared to
        ``loaded.target_ids`` verbatim.

        Task-1611 T2 (fix round 1): the "+ New target" mini-form's own
        typed-but-not-yet-created Name/steering text ALSO counts -- a user
        can type a prefix, never press Create, and have a background
        worker complete mid-edit exactly as easily as they can edit the
        five fields above; nothing about ``stage_target``/``save_bench``
        having consumed nothing yet makes that text less real or less
        destroyable by a recompose. Read fresh via a direct query here
        (never ``self._pending_target_*`` for the CURRENTLY mounted
        steering ``Input`` -- that attribute is only ever refreshed by
        OTHER handlers right before a rebuild, see
        ``_capture_pending_target_form``'s own docstring, and this method
        is called at arbitrary times, not only right before one); the
        NON-mounted steering ``Input`` (only one of the two ever is, see
        ``_build_create_target_control``) falls back to
        ``self._pending_target_*`` instead, since a raw-mode prefix typed
        before a flip to chat is still real unsaved state even though its
        ``Input`` is not currently in the DOM to query.

        ``False`` when this widget never composed a form at all --
        ``self._loaded_config`` stays ``None`` in both of ``compose()``'s
        early-return branches (no db, or an unreadable bench row) -- there
        is no form to have edited. An unparseable Top-K value counts as
        dirty (the user typed SOMETHING different from the loaded int),
        matching Save's own treatment of that value as a real, if invalid,
        edit -- see ``_on_save_pressed``'s identical `int(...)` parse.

        Returns:
            bool: True when any form field, the staged target list, or the
            "+ New target" mini-form's own typed state differs from the
            loaded bench state (mini-form: differs from blank, since that
            state is never itself part of ``self._loaded_config``); False
            for a pristine form or when no form composed at all.
        """
        loaded = self._loaded_config
        if loaded is None:
            return False
        from textual.css.query import QueryError  # noqa: PLC0415 -- narrow, matches this module's other local imports

        try:
            name = self.query_one("#evals-bench-name", Input).value
            description = self.query_one("#evals-bench-description", Input).value
            prompt_mode = self.query_one("#evals-bench-prompt-mode", Select).value
            top_k_raw = self.query_one("#evals-bench-top-k", Input).value
            probes_text = self.query_one("#evals-bench-probes", TextArea).text
        except QueryError:
            # Defensive only: this widget always composes all five fields
            # together with `_loaded_config` (see compose()'s own early
            # returns above) -- treating an unreadable form as dirty is the
            # conservative direction if that invariant is ever broken (a
            # false positive here degrades a completing worker to a toast;
            # a false negative would let it destroy real unsaved state).
            return True

        if name != loaded.name:
            return True
        if description != loaded.description:
            return True
        if prompt_mode != loaded.prompt_mode:
            return True
        try:
            top_k = int(top_k_raw.strip())
        except ValueError:
            return True
        if top_k != loaded.top_k:
            return True
        if _parse_probes_text(probes_text) != tuple(loaded.probes):
            return True
        try:
            capture_continuations = self.query_one(
                "#evals-bench-capture-continuations", Checkbox
            ).value
        except QueryError:
            return True
        if capture_continuations != loaded.capture_continuations:
            return True
        if tuple(self._staged_target_ids) != tuple(loaded.target_ids):
            return True

        try:
            mini_form_name = self.query_one("#evals-target-name", Input).value
        except QueryError:
            # Defensive only, structurally unreachable (whole-branch
            # review): unlike the two steering `Input`s below, `#evals-
            # target-name` is unconditionally part of `_build_create_
            # target_control`'s own output in BOTH prompt modes -- there
            # is no mode in which the mini-form renders at all (which is
            # already guaranteed by this point, see this method's own
            # earlier `loaded is None` early return) without it. Kept for
            # the same reason the outer five-field `QueryError` handler
            # above is kept: the conservative direction if that invariant
            # is ever broken by a future change is treating an unreadable
            # field as dirty, not silently reading it as unchanged.
            mini_form_name = self._pending_target_name
        try:
            mini_form_prefix = self.query_one("#evals-target-prefix", Input).value
        except QueryError:
            mini_form_prefix = self._pending_target_prefix
        try:
            mini_form_system_prompt = self.query_one(
                "#evals-target-system-prompt", Input
            ).value
        except QueryError:
            mini_form_system_prompt = self._pending_target_system_prompt
        if mini_form_name or mini_form_prefix or mini_form_system_prompt:
            return True

        return False

    def compose(self) -> ComposeResult:
        db = self._view_model.db
        if db is None:
            yield Static(
                "The evaluation service is unavailable.",
                id="evals-bench-editor-unavailable",
            )
            return
        try:
            config = load_bench(db, self._bench_id)
        except Exception:
            yield Static(
                "This bench's configuration could not be read.",
                id="evals-bench-editor-error",
            )
            return
        self._loaded_config = config
        self._staged_target_ids = list(config.target_ids)

        yield Static("Name", classes="evals-bench-field-label")
        yield Input(value=config.name, id="evals-bench-name")

        yield Static("Description", classes="evals-bench-field-label")
        yield Input(value=config.description, id="evals-bench-description")

        dataset = self._view_model.dataset_by_id(config.dataset_id)
        dataset_name = dataset.get("name") if dataset else "(dataset not found)"
        sample_count = ((dataset or {}).get("metadata") or {}).get("sample_count")
        dataset_text = f"Dataset: {dataset_name}"
        if sample_count is not None:
            dataset_text += f" ({sample_count} snippets)"
        # markup=False: `dataset_name` is user-authored free text -- a bare
        # `[/]` would raise `MarkupError` the instant this Static lays out.
        # The dataset stays read-only permanently (a create-time-only
        # field -- `save_bench` has no `dataset_id` parameter on its edit
        # path, see that function's own docstring), so the tooltip states
        # that rather than merely omitting an edit control silently.
        dataset_static = Static(dataset_text, id="evals-detail-bench-dataset", markup=False)
        dataset_static.tooltip = (
            "The dataset is set when a bench is created and cannot be changed here."
        )
        yield dataset_static

        yield Static("Prompt mode", classes="evals-bench-field-label")
        yield Select(
            [("raw", "raw"), ("chat", "chat")],
            value=config.prompt_mode,
            id="evals-bench-prompt-mode",
            allow_blank=False,
        )

        yield Static("Top-K", classes="evals-bench-field-label")
        yield Input(value=str(config.top_k), id="evals-bench-top-k")

        yield Static(
            "Probes — one per line; leading and trailing spaces are "
            "significant and shown as ␣",
            classes="evals-bench-field-label",
        )
        if config.probes:
            with Vertical(id="evals-bench-probes-preview"):
                for index, probe in enumerate(config.probes):
                    yield Static(
                        render_snippet_cell(probe),
                        id=f"evals-bench-probe-preview-{index}",
                        classes="evals-bench-probe-preview-row",
                        markup=False,
                    )
        else:
            yield Static(
                "(no probes yet)", id="evals-bench-probes-preview-empty"
            )
        yield TextArea("\n".join(config.probes), id="evals-bench-probes")

        # task-1710: opt into a per-cell continuation. Part of the form
        # like every field above -- saved via Save, covered by
        # `is_dirty()` (see that method's own added check), and rebuilt
        # (not discarded) by the targeted `#evals-bench-targets-section`
        # rebuilds Add/Remove/a prompt-mode flip trigger, since it lives
        # OUTSIDE that container entirely (see `_refresh_targets_section`
        # -- it only ever tears down and rebuilds that one child).
        # `compact=True`: see `CAPTURE_CONTINUATIONS_LABEL`'s own
        # docstring -- a bordered, default `Checkbox` measured 2 rows here
        # (`ToggleButton`'s own DEFAULT_CSS border), real vertical budget
        # this pane's targets section, below, does not have to spare.
        yield Checkbox(
            CAPTURE_CONTINUATIONS_LABEL,
            value=config.capture_continuations,
            id="evals-bench-capture-continuations",
            tooltip=CAPTURE_CONTINUATIONS_TOOLTIP,
            compact=True,
        )

        # The `.ds-recovery-callout` CLASS is deliberately withheld here
        # and added only by `_show_form_error` on an actual failure -- not
        # just `display = False`. `EvalsInspector`'s own preflight callout
        # shares this exact class (see `#evals-inspector-bench .ds-
        # recovery-callout` in _evals.tcss), and `test_never_run_bench_
        # renders_unpreflighted_state` asserts a screen-WIDE `not screen.
        # query(".ds-recovery-callout")` for the clean-preflight case; an
        # always-classed-but-hidden Static here would still match that
        # query and fail an invariant this widget has nothing to do with.
        # An always-visible empty `.ds-recovery-callout` would ALSO be a
        # permanent bordered blank box (the class's own padding/border
        # render even with no text) -- a second, independent reason this
        # class is not applied until there is something to say.
        error_widget = Static("", id="evals-bench-form-error", markup=False)
        error_widget.display = False
        yield error_widget

        with Horizontal(id="evals-bench-form-actions", classes="ds-toolbar"):
            yield Button(
                "Save",
                id="evals-bench-save",
                classes="console-action-primary",
                tooltip="Save name, description, prompt mode, top-K, and probes.",
            )
            yield Button(
                "Revert",
                id="evals-bench-revert",
                classes="console-action-secondary",
                tooltip="Discard unsaved changes and reload this bench.",
            )

        preflight = (
            self._preflight
            if self._preflight is not None
            else self._view_model.preflight_for_bench(self._bench_id)
        )
        self._preflight_map = preflight
        # Task-1611 T2: the SOLE source of truth for which steering Input
        # `_build_create_target_control` mounts -- set here, BEFORE that
        # builder runs, never by querying `#evals-bench-prompt-mode` live:
        # that Select was only just yielded above and has not MOUNTED yet
        # (`compose()` is still executing), so `query_one` would raise.
        self._last_prompt_mode = config.prompt_mode
        yield Vertical(*self._build_targets_section(), id="evals-bench-targets-section")

    def _build_targets_section(self) -> list[Widget]:
        """Builds the whole "Targets (N)" slice -- a FIXED heading, then
        ONE shared scrollable body holding the row table (or the empty
        state), the Add picker (only when there is something to pick),
        and the always-rendered "+ New target" mini-form (task-1611 T2)
        -- as concrete widget INSTANCES rather than a `with Container():
        yield child`-composed generator.

        Task-1611 T2 fix round 1: the row table, the Add picker, and the
        create-target mini-form used to be THREE SEPARATE fixed siblings
        of the heading, each independently competing for this whole
        section's own small `1fr` share -- confirmed live that with
        enough targets to need it, the table's OWN box got squeezed down
        to a literal 1-row floor (the Add picker and mini-form, TOGETHER
        needing 2 more fixed rows, always won that competition first) --
        see `#evals-bench-targets-body`'s own CSS comment for the
        measurements. Wrapping all three in ONE shared scrollable
        `Vertical` (`#evals-bench-targets-body`, this method's own
        returned list's second element) instead means the row table is no
        longer capped by leftover space AFTER the other two are
        subtracted -- it is simply whatever comes FIRST in this ONE
        scrollable list, so it claims as much of the section's `1fr` share
        as there IS, and the Add picker / mini-form (lower priority than
        seeing your own already-staged targets) scroll into view after it
        instead of permanently starving it.

        That `with`-block pattern (used by every OTHER section of
        `compose()` above) only works while Textual's own compose
        machinery has an active `app._compose_stacks` frame open -- true
        during a real `compose()` call, NOT true when this same building
        logic needs to run again later from an event handler
        (`_on_add_target_pressed`, `_on_remove_target_pressed`,
        `_on_prompt_mode_changed`, `stage_target`). Building plain widget
        instances instead (passed as `*children` to each container's own
        constructor -- a fully supported `Widget.__init__` form, not a
        workaround) works identically in both places: `compose()` just
        yields the returned list's container, and `_refresh_targets_
        section` mounts the same builder's output directly into the
        already-live `#evals-bench-targets-section` container.
        """
        db = self._view_model.db
        preflight = self._preflight_map
        heading = Static(
            f"Targets ({len(self._staged_target_ids)})",
            id="evals-bench-targets-heading",
            classes="destination-section evals-pane-title",
        )
        body: list[Widget] = []
        if not self._staged_target_ids:
            body.append(
                Static("No targets configured yet.", id="evals-bench-targets-empty")
            )
        else:
            # Index-derived widget ids, not target_id-derived: `target_ids`
            # is user-editable data (see `BenchConfig`) with no uniqueness
            # or identifier-safety constraint enforced anywhere on write,
            # so a duplicate target id would otherwise collide and fail to
            # compose the whole pane. `target_id` itself is still used
            # below, just never as (or as part of) a widget id -- see
            # `snippet_editor.py`'s identical `_compose_row` fix for the
            # same principle applied to snippets.
            rows = [
                self._build_target_row(db, preflight, index, target_id)
                for index, target_id in enumerate(self._staged_target_ids)
            ]
            body.append(Vertical(*rows, id="evals-bench-target-table"))
        add_control = self._build_target_add_control()
        if add_control is not None:
            body.append(add_control)
        body.append(self._build_create_target_control())
        return [heading, Vertical(*body, id="evals-bench-targets-body")]

    @staticmethod
    def _build_target_row(
        db: Optional[EvalsDB],
        preflight: dict[str, Any],
        index: int,
        target_id: str,
    ) -> Widget:
        """One target row: its readiness ``Static`` (unchanged id/class
        from before Task 6, ``#evals-bench-target-{index}`` /
        ``evals-bench-target-row``) plus a per-row ``Remove`` button
        (``#evals-bench-target-remove-{index}``, following the row's own
        numbering).

        Task-1611 T2: a steered row's label gains a short suffix --
        `` · prefix: <preview>`` (the preview routed through
        ``render_snippet_cell``'s ␣-marker convention, same as every other
        user-authored snippet/probe text this workbench renders) or
        `` · system prompt set`` (no preview -- unlike a prefix, a system
        prompt is never literally concatenated into the measured text, so
        there is no whitespace-significance story to show). Reading a
        corrupt row's steering (``model_steering`` raising) degrades to
        the UNSUFFIXED label rather than crashing this render -- the same
        best-effort posture ``_resolve_bench_targets`` takes for the
        identical case at save time.
        """
        model = db.get_model(target_id) if db is not None else None
        status_text = _target_status_text(preflight, target_id)
        if model is None:
            # config_data.target_ids carries no foreign key (see the
            # design spec's "Run provenance" section) -- a deleted
            # eval_models row leaves a dangling reference here. Still
            # removable via the button below, just never resolvable to a
            # real name.
            label: Any = f"(deleted target {target_id}) — unresolvable"
        else:
            base = f"{model['name']} ({model['provider']}) — {status_text}"
            try:
                prefix, system_prompt = model_steering(model)
            except ValueError:
                prefix, system_prompt = None, None
            if prefix:
                # A `Text` object, never a plain string with `escape_
                # markup` -- `Text.append` treats every argument as
                # LITERAL content, so this is markup-safe by construction
                # regardless of what the prefix contains (the same
                # guarantee `render_snippet_cell` already relies on for
                # arbitrary snippet/probe text elsewhere in this widget).
                label = Text(base)
                label.append(" · prefix: ")
                label.append(render_snippet_cell(_steering_preview_text(prefix)))
            elif system_prompt:
                label = f"{base} · system prompt set"
            else:
                label = base
        return Horizontal(
            Static(
                label,
                id=f"evals-bench-target-{index}",
                classes="evals-bench-target-row",
                markup=False,
            ),
            Button(
                "Remove",
                id=f"evals-bench-target-remove-{index}",
                classes="evals-bench-target-remove console-action-secondary",
            ),
            classes="evals-bench-target-row-wrap",
        )

    def _build_target_add_control(self) -> Optional[Widget]:
        """The Add picker: a ``Select`` over ``EvalsViewModel.
        llama_targets()`` plus an ``Add`` button -- or ``None`` when no
        ``llama_cpp`` ``eval_models`` row exists anywhere in the db yet
        (there is nothing to pick from). Task-1611 T2: the zero-models
        ``#evals-bench-create-target`` button that used to live in this
        method's own empty branch now ALWAYS renders instead, from
        ``_build_create_target_control`` -- see the module docstring's T2
        paragraph.

        ``Select`` raises ``EmptySelectError`` when constructed with zero
        options and ``allow_blank=False`` (see its own docstring) --
        ``llama_targets()`` being empty is exactly when this method
        returns ``None`` instead of building one, so the two can never
        disagree and this never risks that error.
        """
        llama_targets = self._view_model.llama_targets()
        if not llama_targets:
            return None
        # escape_markup: `Select` options parse their label as markup on
        # render (the same `Content.from_markup` hazard this widget's
        # other user-authored strings already guard against, see the
        # module docstring's markup-hazard sweep) -- a model name is free
        # text a user typed, or imported, elsewhere in this app.
        options = [
            (escape_markup(f"{row['name']} ({row['model_id']})"), row["id"])
            for row in llama_targets
        ]
        return Horizontal(
            # compact=True: a bordered Select is 3 rows tall by default
            # (see Textual's own DEFAULT_CSS) against every other row in
            # this section's 1-row convention -- the same fix `sources_
            # pane.py`'s TASK-995 comment applies to a Select inside a
            # height-constrained strip, for the identical reason: without
            # it, this row alone could push the section's own fixed Add
            # control past `#evals-detail-pane`'s clip rectangle at a
            # realistic viewport (confirmed live via `test_every_pane_
            # descendant_stays_within_its_pane`).
            Select(options, allow_blank=False, compact=True, id="evals-bench-add-target"),
            Button(
                "Add",
                id="evals-bench-add-target-button",
                classes="console-action-secondary",
            ),
            id="evals-bench-add-target-row",
        )

    def _build_create_target_control(self) -> Widget:
        """The "+ New target" mini-form (task-1611 T2): a Name ``Input``,
        ONE steering ``Input`` picked by ``self._last_prompt_mode``, and
        the ``#evals-bench-create-target`` button, ALL on one shared row.
        ALWAYS rendered -- unlike the Add picker above, this has no
        zero-models gate: a bench author may want an ADDITIONAL,
        differently-steered target even when one (or several) already
        exist, and steering is immutable per row (see the module
        docstring's T2 paragraph).

        Both ``Input``s carry their descriptive text as a ``placeholder``
        (``PREFIX_FIELD_LABEL``/``SYSTEM_PROMPT_FIELD_LABEL`` for the
        steering one, mirroring the Name field's own "steered variant
        name" placeholder-not-label framing) rather than a separate
        ``Static`` label, and are ``compact=True`` (border-less, 1 row
        instead of the top-level fields' bordered 3) -- the SAME space-
        saving choice `_build_target_add_control`'s own Add picker already
        makes for its ``Select`` (see that method's own `compact=True`
        comment).

        Squeezed onto ONE row -- not, say, fields-above-button, which was
        this method's own first revision -- because this whole targets
        section has a small, FIXED budget inside `#evals-detail-pane` at a
        realistic viewport, confirmed live through TWO separate live
        failures while arriving at this shape: a first, taller draft (a
        margin, a "New target" heading, bordered Inputs, labels ABOVE each
        field) pushed `#evals-bench-create-target` below the SCREEN's own
        bottom edge entirely -- `Widget.region` proved it, not merely
        clipped within its pane -- and `pilot.click` raised `OutOfBounds`.
        A second, already-compacted draft (fields on one row, the button
        on a row below) still lost the SAME way once a real target already
        exists: the Add picker row that then also renders pushed the
        button's row down by exactly one more, landing it UNDER this
        app's own screen-wide footer bar -- geometrically in-bounds (no
        `OutOfBounds` this time) but genuinely unclickable, since
        `Screen.get_widget_at` resolves that point to the footer, not this
        button (confirmed via that exact call, not merely inferred from
        the region numbers).

        Reads ``self._last_prompt_mode`` (never queries ``#evals-bench-
        prompt-mode`` live) so this builds identically whether called from
        `compose()` (before that Select has mounted) or from a later
        targeted rebuild -- see `compose()`'s own comment for why.
        """
        prompt_mode = self._last_prompt_mode or "raw"
        if prompt_mode == "chat":
            steering_placeholder = SYSTEM_PROMPT_FIELD_LABEL
            steering_id = "evals-target-system-prompt"
            steering_value = self._pending_target_system_prompt
        else:
            steering_placeholder = PREFIX_FIELD_LABEL
            steering_id = "evals-target-prefix"
            steering_value = self._pending_target_prefix
        return Horizontal(
            Input(
                value=self._pending_target_name,
                placeholder="steered variant name",
                compact=True,
                id="evals-target-name",
            ),
            Input(
                value=steering_value,
                placeholder=steering_placeholder,
                compact=True,
                id=steering_id,
            ),
            Button(
                "+ New target",
                id="evals-bench-create-target",
                classes="console-action-secondary",
            ),
            id="evals-bench-create-target-form",
        )

    async def _refresh_targets_section(self) -> None:
        """Re-renders just ``#evals-bench-targets-section`` after a staged
        Add/Remove/``stage_target`` mutation -- see ``_build_targets_
        section``'s own docstring for why this is ``remove_children()`` +
        ``mount_all()``, never ``self.refresh(recompose=True)`` on the
        whole editor: a full recompose here would discard whatever the
        user has typed into Name/Description/Top-K/Probes, exactly the
        state loss the module docstring's Task 5 paragraph exists to
        avoid. Also clears any stale ``#evals-bench-form-error`` (e.g. a
        duplicate-add rejection) -- the state it complained about just
        changed.

        ``async`` (and its own callers below too) -- ``remove_children()``
        returns an awaitable that completes once Textual has actually torn
        the old rows down; without awaiting it, ``mount_all()`` runs
        against a DOM that STILL holds the old, same-id widgets (removal
        is scheduled, not immediate), and mounting id-colliding replacements
        into it raises ``DuplicateIds`` -- confirmed the hard way, not
        merely reasoned about, when this was first written as a bare
        `self.refresh`-free but still-synchronous method.
        """
        from textual.css.query import QueryError  # noqa: PLC0415 -- narrow, matches this module's other local imports

        try:
            section = self.query_one("#evals-bench-targets-section", Vertical)
        except QueryError:
            return
        await section.remove_children()
        await section.mount_all(self._build_targets_section())
        self._clear_form_error()

    def _capture_pending_target_form(self) -> None:
        """Stashes whatever is currently typed into the "+ New target"
        mini-form's Name/steering ``Input``s onto ``self._pending_target_
        *`` (task-1611 T2) -- called by every handler BELOW that is about
        to call ``_refresh_targets_section`` for a reason OTHER than a
        successful create (Add, Remove, a prompt-mode flip). Without this,
        any of those targeted rebuilds would silently discard whatever the
        user had started typing into this mini-form, exactly the state
        loss the module docstring's Task 5 paragraph already guards
        against for the OUTER Name/Description/Top-K/Probes fields, one
        level down.

        Only ONE steering ``Input`` is ever mounted at a time (see
        ``_build_create_target_control``), so at most one of the two
        steering ``query_one`` calls below finds a real widget -- the
        other's ``QueryError`` is expected, not a bug, and leaves that
        pending attribute exactly as it already was (e.g. a raw-mode
        prefix typed earlier survives a flip to chat and back).
        """
        from textual.css.query import QueryError  # noqa: PLC0415 -- narrow, matches this module's other local imports

        try:
            self._pending_target_name = self.query_one("#evals-target-name", Input).value
        except QueryError:
            pass
        try:
            self._pending_target_prefix = self.query_one(
                "#evals-target-prefix", Input
            ).value
        except QueryError:
            pass
        try:
            self._pending_target_system_prompt = self.query_one(
                "#evals-target-system-prompt", Input
            ).value
        except QueryError:
            pass

    def _reset_pending_target_form(self) -> None:
        """Clears the "+ New target" mini-form's persisted typed state
        (task-1611 T2) -- called by ``stage_target`` once a Create press
        actually succeeds, so the next rebuild shows a blank form rather
        than re-offering the just-submitted name (which would just collide
        on a second press) or steering text."""
        self._pending_target_name = ""
        self._pending_target_prefix = ""
        self._pending_target_system_prompt = ""

    async def stage_target(self, model_row: Mapping[str, Any]) -> None:
        """Stages a freshly created ``eval_models`` row as a bench target
        -- called by ``evals_screen.py``'s ``CreateTargetRequested``
        handler after IT creates the row (a real ``EvalsDB.create_model``
        write -- see that message's own docstring for why this module
        cannot make that call itself). A TARGETED call against the
        already-mounted editor instance, never a recompose -- see
        ``_build_targets_section``'s own docstring.

        Task-1611 T2: also resets the "+ New target" mini-form's own
        pending state (``_reset_pending_target_form``) -- a fresh, blank
        form for whatever the user creates next, rather than the just-
        submitted Name/steering text lingering to be accidentally
        resubmitted (which would just raise a ``ConflictError`` on the
        SAME name).
        """
        target_id = model_row.get("id") if isinstance(model_row, Mapping) else None
        if not target_id or target_id in self._staged_target_ids:
            # Defensive only: a freshly created row cannot already be
            # staged, but this mirrors the Add-picker's own duplicate
            # guard rather than assuming the caller never will pass one.
            return
        self._staged_target_ids.append(target_id)
        self._reset_pending_target_form()
        await self._refresh_targets_section()

    def _show_form_error(self, message: str) -> None:
        """Renders ``message`` in ``#evals-bench-form-error`` IN PLACE --
        never via ``self.refresh(recompose=True)``. A recompose here would
        rebuild every field from the last-saved ``BenchConfig``, discarding
        whatever the user had just typed -- exactly the state loss a failed
        Save must not cause (see the module docstring). ``add_class`` (not
        a class set at compose time) is what actually makes this callout
        `.ds-recovery-callout`-styled -- see this widget's own compose()
        comment for why that has to wait until there is a real error."""
        error_widget = self.query_one("#evals-bench-form-error", Static)
        error_widget.update(message)
        error_widget.add_class("ds-recovery-callout")
        error_widget.display = True

    def _clear_form_error(self) -> None:
        """Hides the shared inline error callout (``#evals-bench-form-
        error``) -- called by a target mutation's success path so a stale
        duplicate-target (or an earlier Top-K/name) error does not linger
        once the state it complained about has changed. Mirrors
        ``_show_form_error``'s own in-place-update contract: never a
        recompose."""
        from textual.css.query import QueryError  # noqa: PLC0415 -- narrow, matches this module's other local imports

        try:
            error_widget = self.query_one("#evals-bench-form-error", Static)
        except QueryError:
            return
        error_widget.update("")
        error_widget.remove_class("ds-recovery-callout")
        error_widget.display = False

    @on(Button.Pressed, "#evals-bench-add-target-button")
    async def _on_add_target_pressed(self, event: Button.Pressed) -> None:
        """Stages the picker's currently selected target -- form state
        only (see the module docstring); the actual `eval_tasks.
        config_data.target_ids` write happens at Save, same as every other
        field this widget edits."""
        event.stop()
        from textual.css.query import QueryError  # noqa: PLC0415 -- narrow, matches this module's other local imports

        try:
            picker = self.query_one("#evals-bench-add-target", Select)
        except QueryError:
            # Defensive only: this button is never composed without the
            # picker beside it (see `_build_target_add_control`) -- the
            # zero-models state means `_build_target_add_control` returned
            # `None` instead, with no picker/Add button at all.
            return
        target_id = picker.value
        if target_id is Select.BLANK or not isinstance(target_id, str):
            # Defensive only: `allow_blank=False` plus at least one option
            # (see `_build_target_add_control`'s own `EmptySelectError`
            # note) means Select auto-selects a real value the instant it
            # mounts.
            return
        if target_id in self._staged_target_ids:
            self._show_form_error("Target already on this bench.")
            return
        self._staged_target_ids.append(target_id)
        self._capture_pending_target_form()
        await self._refresh_targets_section()

    @on(Button.Pressed, ".evals-bench-target-remove")
    async def _on_remove_target_pressed(self, event: Button.Pressed) -> None:
        """Un-stages one target row by INDEX -- see `_build_target_row`'s
        own comment for why widget ids here are index-, not target_id-,
        derived. Removing the last target is allowed (a draft state --
        the zero-target Run gate and readiness copy both read SAVED state,
        never this widget's staged form state, see the module docstring)."""
        event.stop()
        button_id = event.button.id or ""
        prefix = "evals-bench-target-remove-"
        if not button_id.startswith(prefix):
            return
        try:
            index = int(button_id[len(prefix):])
        except ValueError:
            return
        if not 0 <= index < len(self._staged_target_ids):
            return
        del self._staged_target_ids[index]
        self._capture_pending_target_form()
        await self._refresh_targets_section()

    @on(Select.Changed, "#evals-bench-prompt-mode")
    async def _on_prompt_mode_changed(self, event: Select.Changed) -> None:
        """Swaps the "+ New target" mini-form's steering ``Input`` (task-
        1611 T2) via the SAME targeted ``_refresh_targets_section`` rebuild
        Add/Remove/``stage_target`` already use -- never a whole-widget
        recompose (see the module docstring's T2 paragraph).

        Guarded against the mount-time echo: a fresh ``Select`` constructed
        with a non-blank ``value=`` (this one always is -- see `compose()`)
        posts its own ``Changed`` the instant it mounts, carrying that SAME
        initial value -- not a real user flip (this codebase has hit this
        trap before, see the module docstring's own note). Comparing
        against ``self._last_prompt_mode`` (set in `compose()` before this
        Select even mounts) rather than a boolean "have I ever fired"
        flag means a GENUINE flip back to the value the form started with
        still refreshes correctly -- only the literal mount echo, whose
        value is by definition unchanged from what `_last_prompt_mode`
        already holds, is ever skipped.
        """
        event.stop()
        new_mode = event.value
        if new_mode == self._last_prompt_mode:
            return
        self._capture_pending_target_form()
        self._last_prompt_mode = new_mode
        await self._refresh_targets_section()

    @on(Button.Pressed, "#evals-bench-create-target")
    def _on_create_target_pressed(self, event: Button.Pressed) -> None:
        """Posts `CreateTargetRequested` with whatever is currently typed
        into the "+ New target" mini-form (task-1611 T2) -- Name (raw,
        un-stripped) and ONE steering value, picked by
        `self._last_prompt_mode` (never queried live from `#evals-bench-
        prompt-mode`, since only ONE of the two steering `Input`s is ever
        mounted at once -- see `_build_create_target_control`). A blank
        steering value normalizes to `None` here (not on the screen side)
        so `evals_screen.py`'s handler never has to distinguish "field
        left blank" from "field holds an explicit empty string" -- they
        are the same thing. Handled by `evals_screen.py`, never here --
        see that message class's own docstring for why this module cannot
        create the row itself."""
        event.stop()
        from textual.css.query import QueryError  # noqa: PLC0415 -- narrow, matches this module's other local imports

        try:
            name = self.query_one("#evals-target-name", Input).value
        except QueryError:
            name = ""
        prefix: Optional[str] = None
        system_prompt: Optional[str] = None
        if self._last_prompt_mode == "chat":
            try:
                value = self.query_one("#evals-target-system-prompt", Input).value
            except QueryError:
                value = ""
            system_prompt = value if value != "" else None
        else:
            try:
                value = self.query_one("#evals-target-prefix", Input).value
            except QueryError:
                value = ""
            prefix = value if value != "" else None
        self.post_message(
            self.CreateTargetRequested(name=name, prefix=prefix, system_prompt=system_prompt)
        )

    @on(Button.Pressed, "#evals-bench-save")
    def _on_save_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        db = self._view_model.db
        loaded = self._loaded_config
        if db is None or loaded is None:
            # Defensive only: this button is never composed unless both a
            # db and a successfully loaded config exist (see compose()'s
            # own early returns above).
            return

        name = self.query_one("#evals-bench-name", Input).value
        description = self.query_one("#evals-bench-description", Input).value
        prompt_mode = self.query_one("#evals-bench-prompt-mode", Select).value
        top_k_raw = self.query_one("#evals-bench-top-k", Input).value
        probes_text = self.query_one("#evals-bench-probes", TextArea).text
        capture_continuations = self.query_one(
            "#evals-bench-capture-continuations", Checkbox
        ).value

        try:
            top_k = int(top_k_raw.strip())
            if top_k < 1:
                raise ValueError("top_k below 1")
        except ValueError:
            self._show_form_error(TOP_K_ERROR_TEXT)
            return

        # One probe per line, whitespace preserved exactly -- see
        # `_parse_probes_text`'s own docstring for the full rationale
        # (also shared, verbatim, with `is_dirty()` below). Note this is a
        # real distinction from `compose()`'s own `"\n".join(config.
        # probes)`, which never appends a trailing newline of its own --
        # `TextArea.text` reflects exactly what the user TYPED, trailing
        # Enter-press included, and that is a different guarantee.
        probes = _parse_probes_text(probes_text)

        try:
            config = BenchConfig(
                name=name,
                description=description,
                prompt_mode=prompt_mode,
                top_k=top_k,
                dataset_id=loaded.dataset_id,
                # Task 6: the staged list (Add/Remove'd via this same
                # form, never yet persisted) replaces what used to be
                # `loaded.target_ids`'s verbatim carry-through -- see the
                # module docstring's Task 6 paragraph. `BenchConfig`'s own
                # `strict=True` default still rejects a duplicate here,
                # which the Add picker's own inline rejection (`_on_add_
                # target_pressed`) should already have made unreachable.
                target_ids=tuple(self._staged_target_ids),
                probes=probes,
                concurrency=loaded.concurrency,
                # task-1710 T1 review flag: this used to be the ONE
                # `BenchConfig` field with no passthrough here at all --
                # saving ANY existing bench through this editor silently
                # reset `capture_continuations` back to its dataclass
                # default (`False`), regardless of what was loaded or
                # what the checkbox above showed. Read fresh from the
                # checkbox (task-1710 T2's own opt-in control), exactly
                # like every other editable field above, not carried
                # verbatim from `loaded` the way `concurrency` still is
                # (that field has no UI control of its own yet).
                capture_continuations=capture_continuations,
            )
        except ValueError as exc:
            self._show_form_error(str(exc))
            return

        # Prompt-mode/target revalidation: see `_resolve_bench_targets`'s
        # own docstring -- task-1611 T2 made this reachable through a
        # real, db-backed steered target, not only the monkeypatched one
        # some of this module's own tests still use.
        resolved_targets = _resolve_bench_targets(db, config.target_ids)
        invalid_target = next(
            (t for t in resolved_targets if not t.is_valid_for_mode(config.prompt_mode)),
            None,
        )
        if invalid_target is not None:
            # Steering is IMMUTABLE per row (see `model_steering`'s own
            # docstring -- no `update_model` exists), so this target
            # cannot be "fixed" in place. Removal is the one NECESSARY
            # step -- creating an additional target via this same
            # section's "+ New target" mini-form, without also removing
            # this one, leaves the offending target still staged and this
            # exact error still blocking the NEXT Save (whole-branch
            # review, Minor: an earlier revision of this copy offered
            # "create a new target ... instead" as if it were an
            # alternative to removal, which does not unblock anything on
            # its own). A replacement is optional, phrased as such, and
            # deliberately names no specific steering -- an UNSTEERED
            # replacement target is just as valid for either mode as a
            # steered one (`Target.is_valid_for_mode`: raw only rejects a
            # `system_prompt`, chat only rejects a `prefix`; neither
            # requires the other field be set), so naming one over the
            # other here would over-prescribe.
            self._show_form_error(
                f"{invalid_target.name} is not valid for {config.prompt_mode} mode; "
                "steering cannot be edited on an existing target -- remove it "
                "from this bench (optionally replacing it with a new target "
                "instead)."
            )
            return

        try:
            save_bench(db, config, self._bench_id)
        except ConflictError:
            # `eval_tasks.name` collided with another task's name -- LIVE
            # OR soft-deleted (see `save_bench`'s own docstring): the
            # UNIQUE index on `eval_tasks.name` carries no `deleted_at`
            # exemption, so a deleted bench's name stays reserved forever.
            # `Evals_DB`'s raw message ("Task name already exists") uses
            # the DB's own vocabulary ("Task", not "bench") and says
            # nothing about the reservation trap -- a user who just
            # deleted a bench and reused its name would see this collision
            # with NO bench of that name visible anywhere in the library,
            # which reads as a lie without the explanation below (task-1612
            # copy polish; the earlier commit's own comment on this branch
            # already named this exact trap without fixing the copy).
            # Task-1612 pins this new copy exactly (see
            # test_renaming_to_a_taken_name_renders_the_conflict_callout);
            # the DB's raw message is deliberately never surfaced now.
            self._show_form_error(
                f'A bench named "{config.name}" already exists -- choose a '
                "different name. (Deleting a bench does not free its name: "
                "a deleted bench may still be holding it.)"
            )
            return
        except (ValueError, RuntimeError) as exc:
            # ValueError: BenchConfig re-validation inside save_bench (a
            # duplicate target_id -- unreachable here since the Add
            # picker's own inline rejection, `_on_add_target_pressed`,
            # already keeps `self._staged_target_ids` duplicate-free), or
            # `Evals_DB.InputError` (a ValueError subclass) from
            # `_clean_task_name` rejecting a blank/control-char-only name.
            # RuntimeError: `save_bench`'s update branch found no matching
            # row -- the bench was deleted (this process or another)
            # between this form loading it and this Save (PR #1138
            # review). Without this branch the exception propagated
            # uncaught out of this handler, crashing the worker, AND the
            # user would otherwise have seen nothing at all -- not even a
            # crash, if some caller ever swallowed it -- instead of the
            # honest "this bench is gone" this callout states.
            # Mutation check: dropping the `RuntimeError` half of this
            # tuple makes that Save failure raise straight out of this
            # handler instead of rendering the callout. `ConflictError` has
            # its own mutation check above -- dropping that `except`
            # clause entirely makes the matching Save failure raise
            # straight out of this handler too (it no longer widens
            # `except ValueError`, so it is no longer implicitly caught).
            self._show_form_error(str(exc))
            return

        # Whole-branch review, Minor (judged, documented, not fixed): a
        # successful Save discards the "+ New target" mini-form's own
        # typed-but-not-yet-created Name/steering text -- `Saved` triggers
        # `evals_screen.py`'s `select()`, a full recompose that builds a
        # BRAND NEW `BenchEditor` (see `Saved`'s own docstring: "always an
        # edit, never a create" refers to the BENCH, not this widget
        # instance), and `self._pending_target_*` resets to blank in that
        # fresh instance's `__init__` -- there is no path from here to it.
        # This contradicts `is_dirty()`'s own premise one specific way:
        # that state IS worth protecting from an involuntary recompose
        # (a completing background worker), yet a VOLUNTARY one the user
        # just triggered themselves (pressing Save) silently drops it
        # anyway. Considered threading the mini-form's pending state
        # through `Saved` -> `evals_screen.py` -> the next `BenchEditor`'s
        # constructor (mirroring how `save_bench`'s own cleaned name/
        # description already round-trip through this exact recompose);
        # rejected as not staying cleanly contained: `select()` is this
        # screen's GENERIC recompose entry point, shared by every
        # selection-kind change, not a Save-specific hook -- carrying
        # state through it correctly requires it be scoped to the EXACT
        # bench just saved and unconditionally cleared after one use, or
        # a stray value could leak into an unrelated LATER selection's
        # freshly-composed editor, a worse and stranger bug than today's
        # simple, well-understood loss. Documented here, and in the module
        # docstring's Task-1610 paragraph, as a deliberate boundary:
        # Revert discarding unsaved state is what "revert" MEANS: this is
        # the one place Save (a success, not a discard) does too.
        self.post_message(self.Saved(self._bench_id))

    @on(Button.Pressed, "#evals-bench-revert")
    def _on_revert_pressed(self, event: Button.Pressed) -> None:
        """Discards unsaved edits by re-selecting this same bench -- the
        screen's own `select()` recompose reloads every field from
        storage, which is what "revert" means here (there is no separate
        in-memory draft to roll back; the fields ARE the widgets)."""
        event.stop()
        screen_select = getattr(self.screen, "select", None)
        if callable(screen_select):
            screen_select(kind="bench", id=self._bench_id)


class ClassicTaskDetail(Vertical):
    """Read-only detail for a pre-existing (non-word-bench) eval_tasks row:
    its run history and the fixed deferral sentence. No run control is
    composed anywhere in this widget -- see ``evals_screen.py``'s
    ``_compose_inspector_pane``, which renders no button at all for a
    classic selection."""

    def __init__(self, view_model: EvalsViewModel, task: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._view_model = view_model
        # NOT `self._task` -- Textual's own `MessagePump` uses that name
        # for the widget's message-pump asyncio task and overwrites it
        # once mounting starts, silently clobbering a same-named instance
        # attribute set here in `__init__` (confirmed the hard way: this
        # was `self._task = task` originally, and `compose()` below saw an
        # `asyncio.Task`, not the row dict, by the time it ran).
        self._classic_task = task

    def compose(self) -> ComposeResult:
        task = self._classic_task
        # markup=False: the classic task's name is exactly as user-authored
        # as a word bench's (see `_classic_row_label` in library_rail.py,
        # fixed in the same task-1482 sweep) -- a bare `[/]` would raise
        # `MarkupError` the instant this Static lays out. `task_type` below
        # is not free text (a fixed, internally-controlled vocabulary), so
        # it needs no equivalent guard.
        yield Static(
            str(task.get("name") or "Untitled task"),
            id="evals-detail-classic-name",
            classes="evals-pane-heading",
            markup=False,
        )
        yield Static(
            f"Task type: {task.get('task_type', 'unknown')}",
            id="evals-detail-classic-type",
        )
        yield Static("Run history", classes="destination-section evals-pane-title")

        runs = self._view_model.runs_for_task(task["id"])
        if not runs:
            yield Static("No runs yet.", id="evals-detail-classic-runs-empty")
        else:
            for index, run in enumerate(runs):
                status = run.get("status", "unknown")
                created_at = run.get("created_at", "")
                model_name = run.get("model_name", "unknown")
                yield Static(
                    f"{created_at}  {model_name}  {status}",
                    id=f"evals-detail-classic-run-{index}",
                    classes="evals-classic-run-row",
                    markup=False,
                )

        yield Static(
            CLASSIC_TASK_DEFERRAL_SENTENCE, id="evals-detail-classic-deferral", markup=False
        )

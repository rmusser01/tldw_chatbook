"""Readiness inspector: per-target readiness, recovery callouts, and a
call/time estimate for a selected word bench.

Mounted by ``evals_screen.py``'s ``_compose_inspector_pane`` above the
existing ``#evals-primary-action`` button (unchanged from Task 3 -- this
widget only adds content, it does not own the run control). Readiness
renders from a ``preflight`` map (``word_bench.storage.load_run_preflight``,
via ``EvalsViewModel.preflight_for_bench``) resolved ONCE per selection by
``evals_screen.py`` and passed into ``__init__`` -- see this class's own
``__init__`` and ``bench_editor.py``'s identical parameter -- rather than
this widget resolving it itself, so a bench selection does not read the
same run-group snapshot twice. Never calls a provider -- see
``bench_editor.py``'s module docstring for the shared "never imports the
runner" guarantee both widgets carry.

Per the design contract, ``.ds-status-badge`` colour lives in app-tier CSS
(``css/features/_evals.tcss``), never in this widget's own CSS -- there is
no ``DEFAULT_CSS`` here at all, deliberately.

task-1691 Task 2: each target row also renders a captured continuation --
``PreflightResult.continuation`` (task-1691 Task 1), a short, best-effort
sample of what THIS target generates when the canary prompt is actually
continued (never a per-cell or per-snippet continuation -- see
``_CONTINUATION_LABEL``'s own comment for why the copy names the canary
prompt explicitly). Rendered as a sub-line directly under its target's own
badge row (``_continuation_static``), markup-safe and with whitespace made
visible via ``snippet_editor.render_snippet_cell``'s ␣ convention, reusing
that function rather than reinventing it -- the same convention
``bench_editor.py``'s steered-target rows already use. Absent or empty
(historical runs recorded before this field existed, or a failed capture
that degraded to ``""``) renders nothing extra: no empty label, no
dangling separator.

task-1710 T2: two more changes, the per-snippet sibling of task-1691's own
per-target continuation above --

- The "Estimate" call count now accounts for ``BenchConfig.
  capture_continuations`` (``_continuation_call_count``): a raw-mode run
  with the flag on issues one extra request per measured cell, roughly
  doubling the total; chat mode salvages the continuation off the
  measurement response already made and adds nothing. This is the ONE
  place a user sees this cost BEFORE pressing Run, not after.
- ``EvalsCellInspector`` (below) renders ``CellCapture.continuation`` --
  a captured continuation of THIS cell's own snippet, not the fixed
  canary prompt above -- beside the focused cell's top-K, under the same
  rules: ``markup=False``, ␣ whitespace marking via ``render_snippet_
  cell``, the ⏎ single-line guard, a bounded preview, and nothing
  rendered for an absent/empty continuation (a historical run, a flag-off
  run, or a failed capture -- see ``_cell_continuation_static``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from loguru import logger
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from ...config import LOCAL_PROVIDERS
from ...Evals.word_bench import analysis
from ...Evals.word_bench.models import BenchConfig, CellCapture, CellError, PreflightResult
from ...Evals.word_bench.storage import load_bench
from .evals_state import EvalsViewModel
from .results_grid import degenerate_canary_text, render_probe_reading, render_token
from .snippet_editor import render_snippet_cell

if TYPE_CHECKING:
    from .results_grid import ResultsGrid

#: Case-insensitive: `eval_models.provider` values in this codebase's own
#: fixtures and the design spec's examples are lowercase ("llama_cpp")
#: while `LOCAL_PROVIDERS`' keys mix case ("Ollama", "koboldcpp").
_LOCAL_PROVIDER_NAMES = {name.lower() for name in LOCAL_PROVIDERS}

#: There is no real timing data before a bench has ever run (nothing here
#: calls a provider -- see the module docstring). This is a rough,
#: explicitly-labelled placeholder used only to render a human-scale
#: duration, not a measurement.
_ASSUMED_SECONDS_PER_CALL = 0.4

#: State -> (problem, next action). Mirrors the design spec's Preflight
#: table (`2026-07-25-evals-console-rebuild-design.md`); an unlisted state
#: falls back to the target's own `detail` string plus a generic next step.
_BLOCKED_COPY: dict[str, tuple[str, str]] = {
    "unreachable": (
        "endpoint could not be reached.",
        "Check that the target's server is running and reachable, then retry.",
    ),
    "no_logprobs": (
        "responded but did not return logprobs.",
        "This provider cannot report logprobs for a word bench; choose a different target.",
    ),
    "mode_unsupported": (
        "does not support this bench's prompt mode.",
        "Switch the bench to the other prompt mode, or choose a different target.",
    ),
    "no_content_token": (
        "chat template only emitted control tokens within the capture window.",
        "This target cannot be measured in chat mode as configured.",
    ),
}


#: task-1691: this pane's OWN display bound on a captured continuation's
#: preview length -- deliberately independent of the engine's own storage
#: cap (``CONTINUATION_CHAR_CAP``, defined alongside the HTTP capture
#: seam). Never imported from there: this module's own source-scan test
#: pins that ``inspector.py`` may not even name that module in its source,
#: the same "never reaches the provider, not even transitively" guarantee
#: the module docstring describes.
_CONTINUATION_PREVIEW_MAX_LEN = 100

#: Verbatim, deliberately naming "canary prompt": a captured continuation is
#: a sample of what THIS target generates for the fixed canary prompt, not a
#: per-cell continuation of any snippet a bench actually measures and not a
#: claim about any snippet's own behaviour -- see the module docstring.
_CONTINUATION_LABEL = "Canary prompt continuation: "


def _is_local_provider(provider: str) -> bool:
    return provider.strip().lower() in _LOCAL_PROVIDER_NAMES


def _format_estimate_duration(total_seconds: float) -> str:
    minutes, seconds = divmod(max(0, int(round(total_seconds))), 60)
    return f"~{minutes:02d}:{seconds:02d}"


def _continuation_call_count(config: BenchConfig, measured_call_count: int) -> int:
    """The EXTRA calls ``BenchConfig.capture_continuations`` (task-1710)
    adds on top of ``measured_call_count`` (``sample_count * target_
    count``), so the "Estimate" section's call count is honest BEFORE a
    run starts, not only after (the whole point of task-1710's own
    "Estimate" acceptance criterion).

    Mirrors the engine's own, already-implemented cost story exactly
    (never reached from this module -- see the module docstring's own
    "never calls a provider" guarantee -- only mirrored in arithmetic)
    rather than re-deriving it independently:

    - Raw mode: the engine issues one genuinely separate HTTP request per
      measured cell for the continuation -- the flag roughly DOUBLES the
      run's call count, so this returns ``measured_call_count`` again.
    - Chat mode: the engine salvages the continuation off the measurement
      response already made -- costs nothing extra, so this returns ``0``
      even when the flag is on. A bench in chat mode must never have its
      estimate inflated for a cost it will not actually pay.
    - The flag off: ``0``, unconditionally -- today's request count is
      unchanged, per task-1710's own "with it off, request count per cell
      is unchanged from today" acceptance criterion.
    """
    if config.capture_continuations and config.prompt_mode == "raw":
        return measured_call_count
    return 0


def _status_css_class(result: Optional[PreflightResult]) -> str:
    if result is None:
        return "evals-status-unchecked"
    return {
        "Ready": "evals-status-ready",
        "Unavailable": "evals-status-unavailable",
        "Blocked": "evals-status-blocked",
    }.get(result.status_label, "evals-status-blocked")


def _recovery_callout_text(target_label: str, result: PreflightResult) -> str:
    """Owner/problem/next-action copy for a target whose preflight is not
    a clean pass -- covers the warned-but-Ready case (naming the target
    and what its canary produced) and every Blocked/Unavailable state
    (naming an owner, the problem, and a next action), per the design
    spec's Preflight table.
    """
    if result.is_warned:
        # TASK-1036: this sentence is shared with results_grid.py's run-
        # view callout via degenerate_canary_text -- see that function's
        # own docstring for why it lives there rather than being
        # duplicated here. A single-element list reproduces this bench
        # view's original wording byte for byte.
        return degenerate_canary_text([target_label])
    problem, next_action = _BLOCKED_COPY.get(
        result.state,
        (result.detail or "could not be confirmed ready.", "Review this target's configuration."),
    )
    return (
        f"Owner: {target_label}'s configured provider.\n"
        f"Problem: {target_label} {problem}\n"
        f"Next: {next_action}"
    )


def _continuation_preview_text(value: str) -> str:
    """Single-line, length-capped preview text for a captured continuation,
    BEFORE it goes through ``render_snippet_cell``'s ␣-marker convention --
    mirrors ``bench_editor.py``'s ``_steering_preview_text`` for the
    identical single-line/length-cap concerns. Unlike a steering prefix
    (typed into a single-line ``Input``, so an embedded newline there is
    only a theoretical possibility), a captured continuation is free-form
    generated text -- the motivating UAT's own payload
    (``'<|channel><|channel>thought\\n<channel|>The sky is **blue'``) is a
    real example of one. Every embedded newline is replaced with a visible
    "⏎" marker, exactly as ``bench_editor.py``'s row table already does for
    a steering value, so a continuation containing one still renders on
    ONE row rather than corrupting this pane's per-target row layout.
    """
    single_line = value.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "⏎")
    if len(single_line) > _CONTINUATION_PREVIEW_MAX_LEN:
        return single_line[:_CONTINUATION_PREVIEW_MAX_LEN] + "…"
    return single_line


def _continuation_static(index: int, continuation: str) -> Optional[Static]:
    """The captured-continuation sub-line for one target row, or ``None``
    when there is nothing to show. Absent/empty ``continuation`` (a
    historical run recorded before task-1691, or a capture that failed and
    degraded to ``""`` -- see ``PreflightResult.continuation``'s own
    docstring) must render NOTHING extra: no empty label, no dangling
    separator, so a bench mixing pre- and post-task-1691 runs never shows a
    row of blank sub-lines next to rows that have real ones.
    """
    if not continuation:
        return None
    label = Text(_CONTINUATION_LABEL)
    # A Rich `Text.append` call, never an f-string concatenated into a
    # plain `str` -- `render_snippet_cell` already returns a `Text` whose
    # every character is LITERAL content (see its own docstring), and
    # appending it into another `Text` preserves that guarantee end to
    # end. Combined with this Static's own `markup=False` below, a
    # continuation carrying a bare `[/]` (raw model output, never
    # sanitized) renders as four literal characters instead of crashing
    # the app on a stray Rich/Textual markup tag.
    label.append(render_snippet_cell(_continuation_preview_text(continuation)))
    return Static(
        label,
        id=f"evals-inspector-target-continuation-{index}",
        classes="evals-target-continuation",
        markup=False,
    )


#: task-1710 T2: the focused-cell continuation's own label -- unlike
#: ``_CONTINUATION_LABEL`` above, this one deliberately does NOT name
#: "canary prompt": ``EvalsCellInspector`` is already scoped to ONE
#: specific measured (snippet, target) cell (its own header line already
#: names the snippet and target -- see ``show_cell``), so "Continuation"
#: unambiguously means "what this cell's own snippet continues as" with
#: no risk of being misread as the fixed canary sample the readiness pane
#: shows for a different selection kind entirely.
_CELL_CONTINUATION_LABEL = "Continuation: "


def _cell_continuation_text(continuation: str) -> Optional[Text]:
    """The focused-cell continuation line's Rich ``Text``, or ``None`` for
    an absent/empty continuation -- a historical run recorded before
    task-1710, a run captured with ``BenchConfig.capture_continuations``
    off, or a capture that failed and degraded to ``""`` (see
    ``CellCapture.continuation``'s own docstring). ``None`` means "render
    nothing", not "render an empty line" -- see ``EvalsCellInspector.
    show_cell``, which hides the continuation widget entirely in that
    case.

    Deliberately reuses ``_continuation_preview_text``/``_CONTINUATION_
    PREVIEW_MAX_LEN`` -- task-1691's own preview bound -- rather than a
    second, independently-tuned cap: both are the same pane's own display
    bound on a captured continuation, and a per-cell continuation is no
    more or less likely to run long than a per-target one.
    """
    if not continuation:
        return None
    label = Text(_CELL_CONTINUATION_LABEL)
    # Same markup-safety argument as `_continuation_static` above: a
    # Rich `Text.append` call, never an f-string built into a plain
    # `str` -- `render_snippet_cell`'s output is LITERAL content by
    # construction, and appending it preserves that guarantee through to
    # the `Static(..., markup=False)` this label is ultimately handed to.
    label.append(render_snippet_cell(_continuation_preview_text(continuation)))
    return label


class EvalsInspector(Vertical):
    """Inspector-pane content for a selected word bench: a Readiness list
    (one row per target, with a recovery callout for anything short of a
    clean Ready) and an Estimate (call count, time, and cost for paid
    targets only)."""

    def __init__(
        self,
        view_model: EvalsViewModel,
        bench_id: str,
        preflight: Optional[dict[str, PreflightResult]] = None,
        **kwargs: Any,
    ) -> None:
        """``preflight`` is resolved ONCE by ``EvalsScreen`` per selection
        and threaded into both this widget and ``BenchEditor`` -- see
        ``BenchEditor.__init__``'s identical parameter for why (I2 in the
        PR 3a fix report: each pane calling
        ``EvalsViewModel.preflight_for_bench`` independently read the
        bench's run-group snapshot twice on one render). ``None`` falls
        back to resolving it locally, for a widget constructed directly.
        """
        super().__init__(**kwargs)
        self._view_model = view_model
        self._bench_id = bench_id
        self._preflight = preflight

    def compose(self) -> ComposeResult:
        db = self._view_model.db
        if db is None:
            return
        try:
            config = load_bench(db, self._bench_id)
        except Exception:
            # A bare `except Exception: return` here used to yield ZERO
            # widgets (this is a generator), leaving a blank inspector pane
            # with no message and no log line -- nothing to diagnose from
            # (TASK-861). `evals_screen.py`'s `_compose_inspector_pane`
            # only mounts this widget once `EvalsViewModel.bench_by_id`
            # already found the bench, so reaching this branch means
            # either a race (deleted between that read and this one) or an
            # unexpected failure below it (a locked database, a disk
            # error, or a corrupted `config_data` payload) -- see
            # `results_grid.py`'s `ResultsGrid.compose`, which hit the same
            # class of problem for `load_grid` and now logs + renders a
            # visible error state instead of guessing which case this is.
            # Deliberately `Exception`, not `BaseException`:
            # `asyncio.CancelledError` is a `BaseException` subclass and
            # must keep propagating, not be swallowed here.
            logger.opt(exception=True).error(
                f"Unexpected failure loading bench configuration for the "
                f"inspector pane, bench {self._bench_id!r}."
            )
            yield Static(
                "This bench's readiness could not be loaded because of an "
                "unexpected error; see the log for details.",
                id="evals-inspector-error",
                markup=False,
            )
            return

        preflight = (
            self._preflight
            if self._preflight is not None
            else self._view_model.preflight_for_bench(self._bench_id)
        )

        yield Static("Readiness", classes="destination-section evals-pane-title")
        if not config.target_ids:
            yield Static("No targets configured yet.", id="evals-inspector-readiness-empty")

        providers: list[str] = []
        # Index-derived widget ids, not target_id-derived -- same fix, same
        # reasoning as `bench_editor.py`'s identical target table: a
        # `BenchConfig.target_ids` duplicate must not collide two widget
        # ids and fail to compose this whole pane. `target_id` is still
        # used below for every lookup (`db.get_model`, `preflight.get`),
        # just never as part of a widget id.
        for index, target_id in enumerate(config.target_ids):
            model = db.get_model(target_id)
            target_label = model["name"] if model else f"(deleted target {target_id})"
            if model is not None:
                providers.append(str(model.get("provider") or ""))

            result = preflight.get(target_id)
            status_text = result.status_label if result is not None else "Not yet checked"
            yield Static(
                f"{target_label}: {status_text}",
                id=f"evals-inspector-target-{index}",
                classes=f"ds-status-badge {_status_css_class(result)}",
                markup=False,
            )
            # task-1691 Task 2: a sub-line directly under THIS target's own
            # badge, before any recovery callout below -- the captured
            # continuation is raw evidence a reader can weigh for
            # themselves; the callout (when present) is this pane's own
            # diagnosis of it. `_continuation_static` returns `None` (yields
            # nothing) for an absent/empty continuation, per its own
            # docstring.
            continuation_widget = _continuation_static(
                index, result.continuation if result is not None else ""
            )
            if continuation_widget is not None:
                yield continuation_widget
            # Only a clean pass (Ready, not warned) needs no callout -- see
            # the design spec's Preflight table: every other row states
            # "which endpoint / cannot report logprobs / switch mode /
            # out-of-distribution", never leaves a bare non-Ready badge.
            needs_callout = result is not None and (
                result.is_warned or result.status_label != "Ready"
            )
            if needs_callout:
                yield Static(
                    _recovery_callout_text(target_label, result),
                    id=f"evals-inspector-target-callout-{index}",
                    classes="ds-recovery-callout",
                    markup=False,
                )

        yield Static("Estimate", classes="destination-section evals-pane-title")
        dataset = self._view_model.dataset_by_id(config.dataset_id)
        sample_count = ((dataset or {}).get("metadata") or {}).get("sample_count") or 0
        target_count = len(config.target_ids)
        measured_call_count = sample_count * target_count
        # task-1710: with `capture_continuations` on, a raw-mode run also
        # issues one separate request per cell -- see
        # `_continuation_call_count`'s own docstring for why this is `0`
        # in chat mode (salvaged off the measurement response, free) and
        # always `0` with the flag off. Must be reflected here, BEFORE the
        # run starts -- this is the one place a user can see the cost of
        # turning the flag on before pressing Run.
        call_count = measured_call_count + _continuation_call_count(
            config, measured_call_count
        )
        duration = _format_estimate_duration(call_count * _ASSUMED_SECONDS_PER_CALL)
        yield Static(
            f"{call_count} calls · {duration}",
            id="evals-inspector-estimate-calls",
            markup=False,
        )

        unresolved_target_count = len(config.target_ids) - len(providers)
        if unresolved_target_count > 0:
            # A deleted target contributes no provider to `providers` at
            # all (see the loop above) -- a bench whose targets have ALL
            # been deleted used to fall through to the `else` below and
            # claim "local · no cost", which is a claim about money this
            # code has no basis for: the deleted target's provider (paid or
            # local) is simply unknown, not confirmed local. Any
            # unresolvable target makes the whole cost line unknown, not
            # just the deleted target's own row.
            cost_text = (
                "cost unknown — one or more targets could not be resolved"
            )
        elif providers and any(not _is_local_provider(provider) for provider in providers):
            cost_text = (
                "One or more targets are paid providers; this workbench does not "
                "estimate cost yet."
            )
        else:
            cost_text = "local · no cost"
        yield Static(cost_text, id="evals-inspector-estimate-cost", markup=False)


class EvalsCellInspector(Vertical):
    """Inspector-pane content for a ``"run_group"`` selection: a focused
    grid cell's full top-K and probe table.

    Mounted by ``evals_screen.py``'s ``_compose_inspector_pane`` alongside
    the (disabled, already-completed-run) primary action button. Updated by
    ``EvalsScreen._on_grid_cell_focused`` calling ``show_cell()`` directly
    against this already-mounted widget whenever ``results_grid.ResultsGrid``
    posts a ``CellFocused`` message -- NEVER by a screen-level
    ``refresh(recompose=True)``. Arrow-key movement in the grid fires one of
    these per keystroke; recomposing the screen on every keystroke would
    tear down and rebuild the grid's own ``DataTable``, losing cursor
    position and re-reading the run group from the database each time (see
    ``results_grid.py``'s module docstring).

    Renders no arithmetic of its own -- every value it prints is read
    directly off a ``CellCapture``/``ProbeReading`` the engine already
    computed, mirroring ``results_grid.py``'s own rule.

    task-1710 T2: also renders ``CellCapture.continuation`` (the per-cell
    sibling of ``PreflightResult.continuation``, task-1691's per-target
    canary continuation shown in the READINESS pane, an entirely
    different widget for an entirely different selection kind) as a
    dedicated sub-line BELOW the top-K/probe body -- see
    ``_cell_continuation_text`` for the rendering rules and
    ``#evals-cell-inspector-continuation``'s own compose()-time comment
    for why it is a permanent, always-composed widget rather than one
    mounted/removed per focus change.
    """

    def compose(self) -> ComposeResult:
        yield Static(
            "Focused cell", classes="destination-section evals-pane-title"
        )
        yield Static(
            "Focus a cell in the grid to see its full top-K and probe "
            "table here.",
            id="evals-cell-inspector-body",
            markup=False,
        )
        # task-1710 T2: always composed (never conditionally, unlike
        # `_continuation_static`'s per-target Optional[Static] above,
        # which `EvalsInspector.compose()` can afford since it rebuilds
        # from scratch on every bench selection) -- `show_cell()` below
        # is a TARGETED update against an already-mounted widget (see the
        # class docstring: never a recompose, so the grid's own cursor
        # position survives an arrow-key press), so there is no compose()
        # call between focus changes to conditionally yield or withhold
        # this widget from. Starts hidden (`display = False`) and empty;
        # `show_cell()` toggles both together on every focus change, the
        # same `display`-toggle idiom `bench_editor.py`'s `#evals-bench-
        # form-error` uses for an analogous "nothing to say yet" state.
        continuation_widget = Static(
            "",
            id="evals-cell-inspector-continuation",
            classes="evals-target-continuation",
            markup=False,
        )
        continuation_widget.display = False
        yield continuation_widget

    def show_cell(self, event: "ResultsGrid.CellFocused") -> None:
        """Renders one focused cell. ``event`` is a
        ``results_grid.ResultsGrid.CellFocused`` message -- typed as a
        string annotation (see the ``TYPE_CHECKING`` import above) so this
        module has no runtime dependency on ``results_grid.py``, only the
        other way around.
        """
        from textual.css.query import QueryError  # noqa: PLC0415 -- narrow, matches evals_screen.py's own local import

        try:
            body = self.query_one("#evals-cell-inspector-body", Static)
        except QueryError:
            return

        lines = [f"{event.snippet_text!r} × {event.target_name}"]
        cell = event.cell

        if cell is None:
            lines.append("")
            lines.append("Not yet run.")
        elif isinstance(cell, CellError):
            lines.append("")
            lines.append(f"Failed: {cell.reason}")
            if cell.detail:
                lines.append(cell.detail)
        else:
            lines.append("")
            lines.append(
                f"K requested {cell.k_requested} · K returned {cell.k_returned} · "
                f"canary {cell.canary}"
            )
            lines.append(f"Truncated mass: {cell.truncated_mass * 100:.1f}%")
            if event.delta is not None:
                lines.append("")
                lines.append(
                    f"Δ baseline: {event.delta.jsd:.2f}"
                    + (" !" if event.delta.is_bounded else "")
                )
                if event.delta.is_bounded:
                    # The "!" marker on the grid IS this sentence -- the
                    # grid's entire substitute for the "≥" PR 2's review
                    # disproved. This cell's OWN truncated mass (printed
                    # just above) is NOT what triggered it; the COMBINED
                    # mass across both compared cells is -- see
                    # analysis.combined_truncation's own docstring for why
                    # it can exceed the simple sum of the two cells' own
                    # truncated_mass at mixed K.
                    lines.append(
                        f"Combined truncated mass (this cell + baseline): "
                        f"{event.delta.combined_truncated_mass * 100:.1f}% "
                        f"-- above the "
                        f"{analysis.TRUNCATION_WARN_THRESHOLD * 100:.0f}% "
                        f"warn threshold, so this reading rests on a "
                        f"larger-than-usual amount of extrapolation."
                    )
            lines.append("")
            lines.append("Top-K:")
            for index, tok in enumerate(cell.top_k, start=1):
                lines.append(
                    f"  {index}. {render_token(tok.token)}  "
                    f"{tok.logprob:.2f}  {tok.prob * 100:.1f}%"
                )
            if event.probes:
                lines.append("")
                lines.append("Probes:")
                for probe in event.probes:
                    # The matched TokenProb rides on the reading itself
                    # (analysis.ProbeReading.matched) -- never re-derived
                    # here with a second copy of resolve_probe's match
                    # rule, which would silently keep the old rule if the
                    # engine's ever changed.
                    reading = analysis.resolve_probe(
                        cell, probe, ever_observed=event.ever_observed.get(probe, False)
                    )
                    lines.append(
                        f"  {render_token(probe)}: {render_probe_reading(reading)}"
                    )

        body.update("\n".join(lines))

        # task-1710 T2: the per-cell continuation sub-line -- only ever
        # present on a real `CellCapture` (never `None`/`CellError`,
        # neither of which has anything to continue), and only when one
        # was actually captured (`CellCapture.continuation` defaults to
        # ``""`` for a flag-off run, a historical run recorded before
        # task-1710, or a capture that failed and degraded to ``""``).
        continuation_text = (
            _cell_continuation_text(cell.continuation)
            if isinstance(cell, CellCapture)
            else None
        )
        try:
            continuation_widget = self.query_one(
                "#evals-cell-inspector-continuation", Static
            )
        except QueryError:
            # Defensive only: compose() always yields this widget
            # together with `#evals-cell-inspector-body` above (already
            # confirmed reachable by the earlier `body = ...` lookup this
            # far into the method) -- kept for the same reason that
            # earlier lookup's own `except QueryError: return` is kept.
            return
        if continuation_text is None:
            # `display = False` AND cleared content, not just hidden --
            # a stale continuation from a PREVIOUS focused cell must
            # never linger invisibly and reappear if some later change
            # ever flips `display` back on without also updating content.
            continuation_widget.display = False
            continuation_widget.update("")
        else:
            continuation_widget.update(continuation_text)
            continuation_widget.display = True

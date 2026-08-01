"""End-to-end: a captured continuation survives capture, persistence, and
render, together.

task-1691 Task 3's own closing E2E. Tasks 1-2 each built one half of this
seam and each covered it in isolation: T1 (``capture_client.py``) proved
``preflight`` captures a continuation and that ``storage._snapshot``/
``_preflight_from_snapshot`` round-trip it; T2 (``inspector.py``) proved
``EvalsInspector`` renders an already-constructed ``PreflightResult.
continuation`` correctly, via ``BenchConfig``s and ``PreflightResult``s
built directly in a fixture, never through a real run. Neither task's own
tests drive a REAL run through the screen's own worker
(``EvalsScreen._sample_bench_client_factory`` -> ``WordBenchRunner.run`` ->
``create_run_group(..., preflight=...)``) and then read the continuation
back off a FRESH bench selection -- a genuine DB round-trip through
``EvalsViewModel.preflight_for_bench``, not the in-memory ``PreflightResult``
the worker just built. That joint is this file's whole point.

Same harness and opening moves as ``test_evals_authoring_e2e.py``/
``test_evals_steering_e2e.py``: the rail's Import flow (a real temp file,
bypassing the FileOpen modal), "+ New bench", the zero-``llama_cpp``-models
"Create target" button, Save, Run, then ``EvalsScreen.select(kind="bench",
...)`` (never a stale selection) to read the readiness inspector back.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.Evals.word_bench.models import CellCapture, PreflightResult, Target, TokenProb
from tldw_chatbook.UI.Evals import inspector as inspector_module
from tldw_chatbook.UI.Evals.library_rail import LibraryRail
from tldw_chatbook.UI.Evals.snippet_editor import dataset_snippets
from tldw_chatbook.UI.Screens.evals_screen import EvalsScreen

from .test_evals_screen import EvalsHarness, _FakeAppInstance

_REALISTIC_SIZE = (160, 45)

#: Echoes the motivating UAT (task-1691's own description): a heavily
#: chat-tuned model, prompted in raw mode, emitting its own chat-template
#: scaffolding (multiple control tokens) BEFORE any legible content --
#: `'<|channel><|channel>thought\n<channel|>The sky is **blue'` is the
#: UAT's own real payload. A leading single space is prepended (a raw
#: completion's first generated token commonly IS a leading space -- the
#: model continuing "The sky is" with " <|channel>...") so this fixture
#: exercises BOTH markers `_continuation_static` applies: the "⏎" newline
#: guard (`inspector._continuation_preview_text`) AND the "␣" anomalous-
#: whitespace marker (`snippet_editor.render_snippet_cell`, reused as-is).
_UAT_CONTINUATION = " <|channel><|channel>thought\n<channel|>The sky is **blue"

#: The single-line, marked-up form `_UAT_CONTINUATION` renders as, once
#: the embedded newline becomes "⏎" and the leading space becomes "␣" --
#: computed once here rather than hand-transcribed twice in the test body.
_UAT_CONTINUATION_MARKED = "␣<|channel><|channel>thought⏎<channel|>The sky is **blue"


class _ContinuationCaptureClient:
    """A fake capture client whose ``preflight()`` returns a degenerate
    canary verdict (state ``"ok"``, canary ``"degenerate"``) carrying the
    UAT-shaped continuation above -- exactly the scenario task-1691 exists
    for: a warned target whose raw continuation makes the chat-template
    scaffolding legible as text rather than only as a distribution.
    ``capture()`` itself is unexercised by this test's own assertions (Run
    still needs it to complete without raising, one cell for the one
    imported snippet) so it returns a minimal, generic cell -- mirrors
    every sibling fake client in this suite (see
    ``test_evals_authoring_e2e.py``'s ``_TwoTargetFakeCaptureClient``,
    ``test_evals_steering_e2e.py``'s ``_SteeringAwareFakeCaptureClient``,
    ``test_evals_empty_states.py``'s ``_FakeCaptureClient``).
    """

    def __init__(self, target: Target) -> None:
        self._target = target

    async def preflight(self, target: Target, mode: str, top_k: int) -> PreflightResult:
        return PreflightResult(
            state="ok", k_returned=3, canary="degenerate",
            continuation=_UAT_CONTINUATION,
        )

    async def capture(
        self, snippet: str, target: Target, mode: str, top_k: int
    ) -> CellCapture:
        return CellCapture(
            prompt_mode=mode, k_requested=top_k, k_returned=1, content_offset=0,
            top_k=(TokenProb(token=" a", logprob=-0.3, token_id=1),),
            canary="unchecked", captured_at="2026-07-31T00:00:00Z",
        )


@pytest.fixture
def evals_db() -> EvalsDB:
    return EvalsDB(db_path=":memory:", client_id="test")


@pytest.fixture
def continuation_app(evals_db: EvalsDB) -> EvalsHarness:
    """A configured llama.cpp endpoint -- needed for the bench editor's
    zero-``llama_cpp``-models "Create target" mini-form
    (``evals_screen.py``'s own handler gates on ``sample_bench.
    configured_llama_cpp_url`` before it will write a row). Mirrors
    ``test_evals_authoring_e2e.py``'s own ``authoring_app``."""
    app_config = {"api_settings": {"llama_cpp": {"api_url": "http://localhost:8080"}}}
    return EvalsHarness(_FakeAppInstance(evals_db, app_config=app_config))


async def _wait_until(pilot, predicate, *, tries: int = 300, interval: float = 0.02) -> None:
    """Mirrors the sibling authoring/steering E2E files' own helper --
    polls until a background worker's completion becomes visible (a
    selection change), since ``run_worker`` schedules real async work that
    does not finish within a single ``pilot.pause()``."""
    for _ in range(tries):
        if predicate():
            return
        await pilot.pause(interval)
    raise AssertionError("condition never became true")


def _assert_continuation_row_renders(screen, index: int) -> None:
    """The readiness row's continuation sub-line, read LITERALLY through
    ``.visual.plain`` -- not ``.renderable``/``.content``: this codebase's
    own Textual compatibility shim (``tldw_chatbook/__init__.py``) defines
    ``Static.renderable`` as an alias for ``.content``, the RAW, unparsed
    constructor argument, so it reads back correctly regardless of whether
    ``markup=False`` actually got applied. ``.visual`` is the actual
    ``visualize(..., markup=self._render_markup)`` result ``Static.
    render()`` draws from -- only that path can catch a lost
    ``markup=False``. Mirrors ``test_evals_results_grid.py``'s and
    ``test_evals_bench_editor.py``'s identical rationale/pattern.
    """
    continuation = screen.query_one(
        f"#evals-inspector-target-continuation-{index}"
    )
    text = continuation.visual.plain
    assert text == inspector_module._CONTINUATION_LABEL + _UAT_CONTINUATION_MARKED, text
    # Belt: the two markers this assertion exists to prove, spelled out
    # individually so a future change narrowing either guard fails here
    # with a legible message rather than only failing the exact-match
    # assertion above.
    assert "␣" in text, "leading-whitespace marker missing"
    assert "⏎" in text, "embedded-newline guard missing"
    assert "\n" not in text, "a raw newline must never reach the rendered row"
    assert continuation.region.width > 0
    assert continuation.region.height > 0


@pytest.mark.asyncio
async def test_continuation_captured_persisted_and_rendered_end_to_end(
    continuation_app, evals_db, tmp_path
):
    """Import a dataset -> "+ New bench" -> create one target -> Save ->
    Run -> select the just-finished bench -> the readiness row for that
    target renders the UAT-shaped continuation, literally, with its "⏎"
    and "␣" markers -- then survives a reload of the persisted snapshot
    (select away to the run group and back to the bench).
    """
    import_path = tmp_path / "imported.txt"
    import_path.write_text("The sky is\n", encoding="utf-8")

    async with continuation_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        screen: EvalsScreen = pilot.app.screen
        # Shared by the sample-bench AND bench-run workers (see
        # evals_screen.py's own field docstring) -- this loop only ever
        # exercises the bench-run path below, but the seam is the same
        # one either worker reads.
        screen._sample_bench_client_factory = lambda t: _ContinuationCaptureClient(t)

        # -- Import a 1-snippet dataset via the rail's own Import flow
        # (bypasses the FileOpen modal -- established convention, see
        # library_rail.py's own `_handle_dataset_import_file_selected`
        # docstring).
        rail = screen.query_one(LibraryRail)
        rail._handle_dataset_import_file_selected(import_path)
        await pilot.pause()
        assert screen._selection.kind == "dataset"
        dataset_id = screen._selection.id
        assert len(dataset_snippets(evals_db.get_dataset(dataset_id))) == 1

        # -- "+ New bench" against the just-imported (and still selected)
        # dataset -- a draft bench with zero targets.
        await pilot.click("#evals-rail-new-bench")
        await pilot.pause()
        assert screen._selection.kind == "bench"
        bench_id = screen._selection.id

        # -- Target: the zero-`llama_cpp`-models "Create target" path. No
        # `llama_cpp` `eval_models` row exists anywhere yet, so the button
        # (never the Add picker) renders.
        assert evals_db.list_models(provider="llama_cpp") == []
        await pilot.click("#evals-bench-create-target")
        await pilot.pause()
        created = evals_db.list_models(provider="llama_cpp")
        assert len(created) == 1, "Create target must mint exactly one row"
        assert screen.query_one("#evals-bench-target-0")

        # -- Save persists the staged target and re-selects the bench.
        await pilot.click("#evals-bench-save")
        await pilot.pause()
        assert not screen.query_one("#evals-bench-form-error").display

        # -- Run. `_ContinuationCaptureClient.preflight()` above is what
        # actually captures the continuation, exactly the way a real
        # `WordBenchCaptureClient.preflight()` would (T1) -- this test
        # never calls it directly, only through the screen's own worker.
        await pilot.click("#evals-primary-action")
        await _wait_until(pilot, lambda: screen._selection.kind == "run_group")
        await pilot.pause()
        run_group_id = screen._selection.id

        # -- Select the bench: the joint under test. The readiness
        # inspector's continuation sub-line is built from a FRESH
        # `EvalsViewModel.preflight_for_bench` read (`EvalsScreen.select`
        # resets `_preflight_cache` -- see its own docstring) -- a real DB
        # round-trip through the run's persisted snapshot
        # (`word_bench.storage._snapshot`/`_preflight_from_snapshot`,
        # T1's own persistence seam), never the in-memory `PreflightResult`
        # the worker just built.
        screen.select(kind="bench", id=bench_id)
        await pilot.pause()
        _assert_continuation_row_renders(screen, 0)

        # -- Reload: select away (the run group) and back to the bench --
        # a second, independent DB round-trip through the SAME persisted
        # snapshot, proving this was never a one-shot coincidence of
        # whatever object happened to still be alive in memory right after
        # the run.
        screen.select(kind="run_group", id=run_group_id)
        await pilot.pause()
        screen.select(kind="bench", id=bench_id)
        await pilot.pause()
        _assert_continuation_row_renders(screen, 0)

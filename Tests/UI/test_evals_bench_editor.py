"""Bench editor and readiness inspector (PR 3a, Task 4).

Selecting a word bench renders its metadata and target table in the detail
pane, and per-target readiness plus a call/time estimate in the inspector.
Readiness always comes from the bench's most recent run snapshot
(``word_bench.storage.load_grid``'s ``preflight`` mapping) -- never
re-computed here, so these tests build that snapshot directly with
``create_run_group(..., preflight=...)`` rather than running anything.

Selecting a classic (non-word-bench) task renders a read-only detail with
run history and a fixed deferral sentence, and carries no run control.

Mirrors ``test_evals_screen.py``'s harness (bundled CSS, a fake
``app_instance`` exposing ``evaluation_orchestrator.db``) rather than
inventing a second one -- see that file's own module docstring for why the
real stylesheet is loaded.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from rich.markup import escape as escape_markup
from textual.app import App
from textual.widgets import Button, Checkbox, Input, Select, TextArea

import tldw_chatbook
from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.Evals.word_bench.models import BenchConfig, PreflightResult, Snippet, Target
from tldw_chatbook.Evals.word_bench.storage import (
    BENCH_TYPE,
    create_run_group,
    load_bench,
    save_bench,
)
from tldw_chatbook.UI.Evals import bench_editor as bench_editor_module
from tldw_chatbook.UI.Evals import inspector as inspector_module
from tldw_chatbook.UI.Evals import sample_bench
from tldw_chatbook.UI.Evals.bench_editor import (
    CAPTURE_CONTINUATIONS_LABEL,
    CLASSIC_TASK_DEFERRAL_SENTENCE,
    PREFIX_FIELD_LABEL,
    SYSTEM_PROMPT_FIELD_LABEL,
    TOP_K_ERROR_TEXT,
    BenchEditor,
)
from tldw_chatbook.UI.Evals.evals_state import EvalsViewModel
from tldw_chatbook.UI.Evals.inspector import EvalsInspector
from tldw_chatbook.UI.Screens.evals_screen import EvalsScreen, EvalsSelection

_BUNDLED_CSS_PATH = str(
    Path(tldw_chatbook.__file__).parent / "css" / "tldw_cli_modular.tcss"
)


class _FakeOrchestrator:
    def __init__(self, db: EvalsDB) -> None:
        self.db = db


class _FakeAppInstance:
    def __init__(self, db: EvalsDB, app_config: dict | None = None) -> None:
        self.evaluation_orchestrator = _FakeOrchestrator(db)
        self.notifications: list[tuple[str, str]] = []
        #: Read by EvalsScreen._current_app_config for the Task 6
        #: zero-models "Create target" flow (sample_bench.
        #: resolve_sample_target's own configured-endpoint gate) --
        #: mirrors test_evals_screen.py's own _FakeAppInstance.
        self.app_config: dict = app_config or {}

    def notify(self, message: str, *, severity: str = "information", **kwargs) -> None:
        self.notifications.append((message, severity))


class EvalsHarness(App):
    CSS_PATH = _BUNDLED_CSS_PATH

    def __init__(self, app_instance: _FakeAppInstance) -> None:
        super().__init__()
        self.app_instance = app_instance

    async def on_mount(self) -> None:
        await self.push_screen(EvalsScreen(self.app_instance))


@pytest.fixture
def evals_db() -> EvalsDB:
    return EvalsDB(db_path=":memory:", client_id="test")


@pytest.fixture
def evals_app(evals_db: EvalsDB) -> EvalsHarness:
    return EvalsHarness(_FakeAppInstance(evals_db))


@pytest.fixture
def evals_app_configured(evals_db: EvalsDB) -> EvalsHarness:
    """Mirrors ``test_evals_screen.py``'s own ``sample_bench_app``: a
    configured llama.cpp endpoint, needed by the "+ New target" create-
    target mini-form (task-1611 T2; Task 6 before it) -- ``evals_screen.
    py``'s handler gates on ``sample_bench.configured_llama_cpp_url``
    before writing a row, regardless of whether zero or several
    ``llama_cpp`` rows already exist."""
    app_config = {"api_settings": {"llama_cpp": {"api_url": "http://localhost:8080"}}}
    return EvalsHarness(_FakeAppInstance(evals_db, app_config=app_config))


def _make_model(db: EvalsDB, name: str, *, provider: str = "llama_cpp", model_id: str = "m") -> str:
    return db.create_model(name=name, provider=provider, model_id=model_id)


@pytest.fixture
def bench_with_mixed_readiness(evals_db: EvalsDB) -> tuple[str, dict[str, str]]:
    """One bench, one run group, three local targets covering the three
    readable labels plus the warned-but-Ready case -- see the design
    spec's Preflight table
    (``Docs/superpowers/specs/2026-07-25-evals-console-rebuild-design.md``).
    """
    ready_id = _make_model(evals_db, "ready-target")
    warned_id = _make_model(evals_db, "warned-target")
    blocked_id = _make_model(evals_db, "blocked-target")
    dataset_id = evals_db.create_dataset(
        name="loaded-nouns",
        format="custom",
        source_path="inline:loaded-nouns",
        metadata={"sample_count": 12},
    )
    config = BenchConfig(
        name="loaded-nouns v1",
        prompt_mode="raw",
        top_k=20,
        dataset_id=dataset_id,
        target_ids=(ready_id, warned_id, blocked_id),
        probes=(" Sure", " I"),
    )
    task_id = save_bench(evals_db, config)
    targets = [
        Target(id=ready_id, name="ready-target", provider="llama_cpp", model_id="m"),
        Target(id=warned_id, name="warned-target", provider="llama_cpp", model_id="m"),
        Target(id=blocked_id, name="blocked-target", provider="llama_cpp", model_id="m"),
    ]
    snippets = [Snippet(id="s1", text="The protestors were", group="neutral")]
    preflight = {
        ready_id: PreflightResult(state="ok", k_returned=20, canary="pass"),
        warned_id: PreflightResult(state="ok", k_returned=20, canary="degenerate"),
        blocked_id: PreflightResult(
            state="no_logprobs",
            k_returned=None,
            canary="unchecked",
            detail="provider does not return logprobs",
        ),
    }
    create_run_group(evals_db, task_id, config, targets, snippets, preflight=preflight)
    return task_id, {"ready": ready_id, "warned": warned_id, "blocked": blocked_id}


#: task-1691 Task 2: `bench_with_continuation_samples`'s stable index
#: mapping, mirroring `_TARGET_INDEX` above -- `config.target_ids` order is
#: what `inspector.py`'s index-derived widget ids follow.
_CONTINUATION_TARGET_INDEX = {
    "whitespace": 0,
    "hazard": 1,
    "newline": 2,
    "long": 3,
    "empty": 4,
}

#: Longer than `inspector._CONTINUATION_PREVIEW_MAX_LEN` (100) so the
#: truncation test below actually exercises the cap.
_LONG_CONTINUATION = "x" * 150


@pytest.fixture
def bench_with_continuation_samples(evals_db: EvalsDB) -> str:
    """One bench, five local targets covering task-1691 Task 2's rendering
    rules for `PreflightResult.continuation`: a continuation worth marking
    up for anomalous whitespace, a markup-hazard continuation (a bare
    `[/]`, the same Rich/Textual crash vector `bench_with_markup_hazard_
    text` covers for bench/dataset names), a continuation carrying an
    embedded newline (the motivating UAT's own scaffolding text), a
    continuation longer than the UI's own preview cap, and a target with
    no continuation at all (the historical-run/failed-capture default).
    Every `PreflightResult` here is a clean Ready pass (`state="ok"`,
    `canary="pass"`) so these tests read the continuation sub-line in
    isolation from the separate recovery-callout tests above.
    """
    whitespace_id = _make_model(evals_db, "whitespace-target")
    hazard_id = _make_model(evals_db, "hazard-target")
    newline_id = _make_model(evals_db, "newline-target")
    long_id = _make_model(evals_db, "long-target")
    empty_id = _make_model(evals_db, "empty-target")
    dataset_id = evals_db.create_dataset(
        name="continuation-samples",
        format="custom",
        source_path="inline:continuation-samples",
        metadata={"sample_count": 4},
    )
    config = BenchConfig(
        name="continuation samples v1",
        prompt_mode="raw",
        top_k=20,
        dataset_id=dataset_id,
        target_ids=(whitespace_id, hazard_id, newline_id, long_id, empty_id),
    )
    task_id = save_bench(evals_db, config)
    targets = [
        Target(id=whitespace_id, name="whitespace-target", provider="llama_cpp", model_id="m"),
        Target(id=hazard_id, name="hazard-target", provider="llama_cpp", model_id="m"),
        Target(id=newline_id, name="newline-target", provider="llama_cpp", model_id="m"),
        Target(id=long_id, name="long-target", provider="llama_cpp", model_id="m"),
        Target(id=empty_id, name="empty-target", provider="llama_cpp", model_id="m"),
    ]
    snippets = [Snippet(id="s1", text="The protestors were", group="neutral")]
    preflight = {
        whitespace_id: PreflightResult(
            state="ok",
            k_returned=20,
            canary="pass",
            continuation="  <|channel>thought  scaffolding",
        ),
        hazard_id: PreflightResult(
            state="ok",
            k_returned=20,
            canary="pass",
            continuation="[/]bold-looking output",
        ),
        newline_id: PreflightResult(
            state="ok",
            k_returned=20,
            canary="pass",
            continuation="<|channel><|channel>thought\n<channel|>The sky is **blue",
        ),
        long_id: PreflightResult(
            state="ok",
            k_returned=20,
            canary="pass",
            continuation=_LONG_CONTINUATION,
        ),
        empty_id: PreflightResult(state="ok", k_returned=20, canary="pass"),
    }
    create_run_group(evals_db, task_id, config, targets, snippets, preflight=preflight)
    return task_id


@pytest.fixture
def bench_with_markup_hazard_text(evals_db: EvalsDB) -> str:
    """task-1482 Task 1: every user-authored string ``BenchEditor`` renders
    as a plain ``Static`` -- name, description, dataset name, and a probe
    -- carries a bare ``[/]``, the same Rich/Textual markup hazard
    task-1476/TASK-1480 already fixed for the rail's own toast text and
    run-row labels (see ``library_rail.py``'s ``_run_group_row_label``).
    ``Static`` parses its argument as markup by default
    (``visualize()``'s own ``markup=True`` default, via ``Content.from_
    markup``), and that parsing happens lazily on first render/layout --
    not merely inside ``compose()`` -- so an unescaped hazard string
    crashes the WHOLE app the instant this bench is selected and laid
    out, confirmed directly against Textual (a bare-bracket ``Static``
    raises ``MarkupError`` out of the compositor's reflow during
    ``pilot.pause()``, not out of ``compose()`` itself).

    Bench/dataset names are machine-generated TODAY -- this hardens the
    rendering path ahead of the bench-authoring program that makes them
    user-typed.
    """
    target_id = _make_model(evals_db, "hazard-target")
    dataset_id = evals_db.create_dataset(
        name="dataset[/]name",
        format="custom",
        source_path="inline:dataset-hazard",
        metadata={"sample_count": 4},
    )
    config = BenchConfig(
        name="bench[/]name",
        description="description[/]text",
        prompt_mode="raw",
        top_k=20,
        dataset_id=dataset_id,
        target_ids=(target_id,),
        probes=("probe[/]text",),
    )
    return save_bench(evals_db, config)


#: `bench_with_mixed_readiness` builds `config.target_ids` in exactly this
#: order (`(ready_id, warned_id, blocked_id)`) -- the target table and
#: inspector rows are index-derived (see `bench_editor.py`/`inspector.py`'s
#: fix), so this is the stable mapping from the fixture's readable names to
#: the row a test should query.
_TARGET_INDEX = {"ready": 0, "warned": 1, "blocked": 2}


@pytest.fixture
def never_run_bench(evals_db: EvalsDB) -> str:
    """A bench with no run group at all -- there is no snapshot to read
    readiness from, so every target must render un-preflighted."""
    target_id = _make_model(evals_db, "solo-target")
    dataset_id = evals_db.create_dataset(
        name="probes-mix",
        format="custom",
        source_path="inline:probes-mix",
        metadata={"sample_count": 6},
    )
    config = BenchConfig(
        name="never-run bench",
        prompt_mode="raw",
        top_k=20,
        dataset_id=dataset_id,
        target_ids=(target_id,),
    )
    return save_bench(evals_db, config)


@pytest.fixture
def paid_target_bench(evals_db: EvalsDB) -> str:
    target_id = _make_model(
        evals_db, "gpt-target", provider="openai", model_id="gpt-3.5-turbo-instruct"
    )
    dataset_id = evals_db.create_dataset(
        name="openers-8",
        format="custom",
        source_path="inline:openers-8",
        metadata={"sample_count": 8},
    )
    config = BenchConfig(
        name="paid bench",
        prompt_mode="raw",
        top_k=20,
        dataset_id=dataset_id,
        target_ids=(target_id,),
    )
    return save_bench(evals_db, config)


@pytest.fixture
def bench_with_all_targets_deleted(evals_db: EvalsDB) -> str:
    """A bench whose only target id no longer resolves to a real
    ``eval_models`` row -- simulating every target having been deleted
    after the bench was configured (mirrors how ``bench_editor.py`` already
    renders a "deleted target" row for exactly this case)."""
    dataset_id = evals_db.create_dataset(
        name="orphaned-targets",
        format="custom",
        source_path="inline:orphaned-targets",
        metadata={"sample_count": 4},
    )
    config = BenchConfig(
        name="orphaned bench",
        prompt_mode="raw",
        top_k=20,
        dataset_id=dataset_id,
        target_ids=("target-that-no-longer-exists",),
    )
    return save_bench(evals_db, config)


@pytest.fixture
def bench_with_one_deleted_and_one_live_local_target(evals_db: EvalsDB) -> str:
    """One resolvable local target plus one deleted target -- "any target
    unresolvable" must still read as unknown cost, not "local · no cost"
    just because the resolvable target happens to be local."""
    live_id = _make_model(evals_db, "still-here")
    dataset_id = evals_db.create_dataset(
        name="mixed-targets",
        format="custom",
        source_path="inline:mixed-targets",
        metadata={"sample_count": 4},
    )
    config = BenchConfig(
        name="mixed bench",
        prompt_mode="raw",
        top_k=20,
        dataset_id=dataset_id,
        target_ids=(live_id, "target-that-no-longer-exists"),
    )
    return save_bench(evals_db, config)


@pytest.fixture
def classic_task_with_runs(evals_db: EvalsDB) -> str:
    """A pre-existing (non-word-bench) eval_tasks row with one completed
    run, mirroring the design spec's 'mmlu-subset' example."""
    dataset_id = evals_db.create_dataset(
        name="mmlu-500", format="custom", source_path="inline:mmlu-500"
    )
    task_id = evals_db.create_task(
        name="mmlu-subset",
        task_type="question_answer",
        config_format="custom",
        config_data={},
        dataset_id=dataset_id,
    )
    model_id = _make_model(evals_db, "gpt-4o-mini", provider="openai", model_id="gpt-4o-mini")
    run_id = evals_db.create_run(name="mmlu run", task_id=task_id, model_id=model_id)
    evals_db.update_run_status(run_id, "completed")
    return task_id


@pytest.fixture
def classic_task_with_markup_hazard_name(evals_db: EvalsDB) -> str:
    """task-1482 Task 1 fix round 1: ``ClassicTaskDetail.compose()``'s own
    heading Static (``#evals-detail-classic-name``) was missed by the
    original sweep -- it renders ``task.get("name")`` with no
    ``markup=False``, unlike the run rows and deferral sentence a few
    lines below it in the same method (both already protected). A classic
    task's name is exactly as user-authored as a word bench's (see
    ``_classic_row_label`` in ``library_rail.py``, fixed in the same
    original commit) -- this fixture carries the identical `[/]` hazard
    into the detail pane instead of the rail row.
    """
    dataset_id = evals_db.create_dataset(
        name="hazard-500", format="custom", source_path="inline:hazard-500"
    )
    return evals_db.create_task(
        name="classic[/]name",
        task_type="question_answer",
        config_format="custom",
        config_data={},
        dataset_id=dataset_id,
    )


def _target_status_text(screen, index: int) -> str:
    """Looks up an inspector target row by INDEX, not target id -- widget
    ids in ``inspector.py`` are index-derived (see its fix for the same
    duplicate-id-collision principle ``snippet_editor.py``'s rows follow),
    so a test must address a row the same way the widget itself does."""
    widget = screen.query_one(f"#evals-inspector-target-{index}")
    text = widget.renderable
    return text.plain if hasattr(text, "plain") else str(text)


def test_llama_targets_uses_the_documented_list_limit_not_list_models_default(
    evals_db: EvalsDB,
):
    """task-1612: ``EvalsViewModel.llama_targets()`` used to call
    ``db.list_models(provider="llama_cpp")`` with no ``limit=`` at all,
    silently falling back to ``EvalsDB.list_models``'s own default of 100
    -- unlike every other read on ``EvalsViewModel``
    (``_all_tasks``/``datasets``/``run_groups``/``runs_for_task``), which
    all pass the module's documented ``_LIST_LIMIT`` (500) so a busy
    install's older rows do not silently fall off. This creates more than
    100 ``llama_cpp`` models and asserts none are dropped -- before the
    fix, only the newest 100 came back."""
    for i in range(120):
        evals_db.create_model(
            name=f"llama-target-{i}", provider="llama_cpp", model_id=f"model-{i}"
        )

    view_model = EvalsViewModel(evals_db)
    targets = view_model.llama_targets()

    assert len(targets) == 120


# ---------------------------------------------------------------------------
# Detail pane: bench metadata + target table
# ---------------------------------------------------------------------------


#: A realistic detail-pane size -- matches test_evals_results_grid.py's own
#: standard (160x45), used everywhere in this file a painted-geometry
#: assertion needs a size closer to a real terminal than pytest-textual's
#: small default.
_REALISTIC_SIZE = (160, 45)


@pytest.mark.asyncio
async def test_bench_detail_pane_shows_metadata_and_target_table(
    evals_app, bench_with_mixed_readiness
):
    """Task 5: every editable field starts pre-filled from the loaded
    ``BenchConfig`` -- the read/write round-trip's read half. Save/Revert
    and every field, including the target table below them, must actually
    paint inside the detail pane at a realistic size, not merely exist in
    the DOM (see evals_screen.py's own module docstring on the hub's
    original zero-size-region defect)."""
    task_id, target_ids = bench_with_mixed_readiness
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app.screen

        name = screen.query_one("#evals-bench-name", Input)
        assert name.value == "loaded-nouns v1"

        description = screen.query_one("#evals-bench-description", Input)
        assert description.value == ""

        dataset_line = screen.query_one("#evals-detail-bench-dataset")
        assert "loaded-nouns" in str(dataset_line.renderable)
        # Dataset is read-only permanently (task-1482's own recorded spec
        # deviation) -- the tooltip is the only place that is stated, since
        # there is no edit control at all to disable instead.
        assert "cannot be changed" in str(dataset_line.tooltip)

        mode_select = screen.query_one("#evals-bench-prompt-mode", Select)
        assert mode_select.value == "raw"

        top_k_input = screen.query_one("#evals-bench-top-k", Input)
        assert top_k_input.value == "20"

        probes_area = screen.query_one("#evals-bench-probes", TextArea)
        # bench_with_mixed_readiness's own probes=(" Sure", " I") -- one
        # probe per line, whitespace preserved exactly (see the module
        # docstring's own whitespace-is-the-instrument rationale).
        assert probes_area.text == " Sure\n I"

        save_button = screen.query_one("#evals-bench-save")
        revert_button = screen.query_one("#evals-bench-revert")

        # Region, not just query_one success -- a widget can be present in
        # the DOM and occupy zero space (see evals_screen.py's own module
        # docstring on the hub's original defect).
        for field_widget in (
            name,
            description,
            dataset_line,
            mode_select,
            top_k_input,
            probes_area,
            save_button,
            revert_button,
        ):
            assert field_widget.region.width > 0, field_widget
            assert field_widget.region.height > 0, field_widget

        for index in range(len(target_ids)):
            row = screen.query_one(f"#evals-bench-target-{index}")
            assert row.region.width > 0
            assert row.region.height > 0


@pytest.mark.asyncio
async def test_probe_preview_renders_leading_space_markers(
    evals_app, bench_with_mixed_readiness
):
    """The read-only probe preview (above the editable TextArea) reuses
    snippet_editor's whitespace-marker convention: a probe's leading space
    is exactly as semantically loaded as a snippet's (see the module
    docstring), so it gets the identical visible ␣ treatment."""
    task_id, _ = bench_with_mixed_readiness
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app.screen

        first = screen.query_one("#evals-bench-probe-preview-0")
        assert first.renderable.plain == "␣Sure"
        assert first.region.width > 0
        assert first.region.height > 0

        second = screen.query_one("#evals-bench-probe-preview-1")
        assert second.renderable.plain == "␣I"


@pytest.mark.asyncio
async def test_bench_editor_fields_render_a_markup_hazard_literally(
    evals_app, bench_with_markup_hazard_text
):
    """task-1482 Task 1's original hazard sweep, updated for Task 5's field
    types. ``Input``/``TextArea`` never parse Rich markup at all (unlike a
    ``Static``, whose lazily-computed ``Content`` -- ``.visual`` -- raised
    ``MarkupError`` on a bare ``[/]`` at layout time before this task), so
    the crash risk for name/description/probe is gone; this instead proves
    the hazardous text round-trips into those widgets byte-for-byte rather
    than being silently dropped or mangled. The dataset name stays a
    ``Static`` (read-only, permanently -- see the module docstring) and
    keeps the original crash-must-not-happen assertion.
    """
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=bench_with_markup_hazard_text)
        await pilot.pause()
        assert pilot.app.is_running, "an unescaped hazard string crashed the app"
        screen = evals_app.screen

        name = screen.query_one("#evals-bench-name", Input)
        assert name.value == "bench[/]name"

        description = screen.query_one("#evals-bench-description", Input)
        assert description.value == "description[/]text"

        probes_area = screen.query_one("#evals-bench-probes", TextArea)
        assert probes_area.text == "probe[/]text"

        dataset_line = screen.query_one("#evals-detail-bench-dataset")
        assert "dataset[/]name" in dataset_line.visual.plain


# ---------------------------------------------------------------------------
# Requirement 1: readable status text, never colour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ready_target_renders_the_readable_label_ready(
    evals_app, bench_with_mixed_readiness
):
    task_id, target_ids = bench_with_mixed_readiness
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        status_text = _target_status_text(evals_app.screen, _TARGET_INDEX["ready"])
        assert "Ready" in status_text
        # No callout for a clean pass -- see the design spec's Preflight
        # table: only "—" (nothing) is prescribed for the sane case.
        assert not evals_app.screen.query(
            f"#evals-inspector-target-callout-{_TARGET_INDEX['ready']}"
        )


# ---------------------------------------------------------------------------
# Requirement 2: warned target is Ready + a callout naming target & canary
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_warned_target_renders_ready_plus_a_recovery_callout(
    evals_app, bench_with_mixed_readiness
):
    task_id, target_ids = bench_with_mixed_readiness
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app.screen

        status_text = _target_status_text(screen, _TARGET_INDEX["warned"])
        # The `or "Ready" in status_text` fallback made the strong
        # `endswith` clause inert (it always passes when the endswith
        # clause does, and would ALSO pass for a status_text that merely
        # mentions "Ready" somewhere without actually being the rendered
        # label) -- dropped.
        assert status_text.strip().endswith("Ready"), status_text
        assert "Blocked" not in status_text
        assert "Unavailable" not in status_text

        callout = screen.query_one(
            f"#evals-inspector-target-callout-{_TARGET_INDEX['warned']}"
        )
        assert "ds-recovery-callout" in callout.classes
        callout_text = str(callout.renderable)
        assert "warned-target" in callout_text
        assert "degenerate" in callout_text
        assert callout.region.width > 0
        assert callout.region.height > 0


# ---------------------------------------------------------------------------
# Requirement 3: Blocked renders owner/problem/next-action callout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_blocked_target_renders_owner_problem_and_next_action(
    evals_app, bench_with_mixed_readiness
):
    task_id, target_ids = bench_with_mixed_readiness
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app.screen

        status_text = _target_status_text(screen, _TARGET_INDEX["blocked"])
        assert "Blocked" in status_text

        callout = screen.query_one(
            f"#evals-inspector-target-callout-{_TARGET_INDEX['blocked']}"
        )
        assert "ds-recovery-callout" in callout.classes
        callout_text = str(callout.renderable)
        assert "blocked-target" in callout_text
        assert "Owner:" in callout_text
        assert "Problem:" in callout_text
        assert "Next:" in callout_text
        # Pinned requirement 3 -- prove this callout is genuinely rendered,
        # not merely present in the DOM (see the region-check rationale
        # above and in evals_screen.py's own module docstring).
        assert callout.region.width > 0
        assert callout.region.height > 0


# ---------------------------------------------------------------------------
# task-1691 Task 2: a captured continuation renders under its target row
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_continuation_renders_under_its_target_row_with_whitespace_markers(
    evals_app, bench_with_continuation_samples
):
    task_id = bench_with_continuation_samples
    index = _CONTINUATION_TARGET_INDEX["whitespace"]
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app.screen

        badge = screen.query_one(f"#evals-inspector-target-{index}")
        continuation = screen.query_one(
            f"#evals-inspector-target-continuation-{index}"
        )
        continuation_text = str(continuation.renderable)

        # Names the canary prompt, not a snippet/cell -- see the module
        # docstring's own copy rationale.
        assert continuation_text.startswith("Canary prompt continuation: ")
        # Leading run of 2 spaces AND the interior run of 2 spaces (`"
        # <|channel>thought  scaffolding"`) both become "␣␣" -- the same
        # marker convention `render_snippet_cell` already applies to
        # snippets/steering prefixes elsewhere in this workbench.
        assert continuation_text.count("␣␣") == 2
        assert "<|channel>thought" in continuation_text
        assert "scaffolding" in continuation_text
        # No raw, unmarked double space slipped through.
        assert "  " not in continuation_text

        # Painted -- this pane has a documented history of collapse/
        # clipping defects (see inspector.py's own module docstring and
        # _evals.tcss's #evals-inspector-bench comment); a new row must
        # genuinely paint, not merely exist in the DOM.
        assert badge.region.width > 0
        assert badge.region.height > 0
        assert continuation.region.width > 0
        assert continuation.region.height > 0


@pytest.mark.asyncio
async def test_markup_hazard_continuation_renders_literally_without_crashing(
    evals_app, bench_with_continuation_samples
):
    """Raw model output is never sanitized -- a captured continuation
    carrying a bare `[/]` must render as four literal characters, not crash
    the whole app the way an unescaped `Static(markup=True)` would (see
    `bench_with_markup_hazard_text`'s identical concern for bench/dataset
    names)."""
    task_id = bench_with_continuation_samples
    index = _CONTINUATION_TARGET_INDEX["hazard"]
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        assert pilot.app.is_running, "an unescaped hazard continuation crashed the app"
        screen = evals_app.screen

        continuation = screen.query_one(
            f"#evals-inspector-target-continuation-{index}"
        )
        continuation_text = str(continuation.renderable)
        assert "[/]bold-looking output" in continuation_text


@pytest.mark.asyncio
async def test_empty_continuation_renders_nothing_extra(
    evals_app, bench_with_continuation_samples
):
    """Absent/empty `continuation` (a historical run, or a failed capture
    that degraded to `""`) must render NOTHING extra -- no empty label, no
    dangling separator."""
    task_id = bench_with_continuation_samples
    index = _CONTINUATION_TARGET_INDEX["empty"]
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app.screen

        # The target's own badge still renders...
        badge = screen.query_one(f"#evals-inspector-target-{index}")
        assert "empty-target" in str(badge.renderable)
        # ...but no continuation sub-line exists for it at all.
        assert not screen.query(f"#evals-inspector-target-continuation-{index}")


@pytest.mark.asyncio
async def test_newline_bearing_continuation_stays_single_line(
    evals_app, bench_with_continuation_samples
):
    """A continuation carrying an embedded newline -- the motivating UAT's
    own scaffolding text -- must not blow up this row into more than one
    logical line; follows the same "⏎" guard convention `bench_editor.py`
    already uses for a target row's steering suffix."""
    task_id = bench_with_continuation_samples
    index = _CONTINUATION_TARGET_INDEX["newline"]
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app.screen

        continuation = screen.query_one(
            f"#evals-inspector-target-continuation-{index}"
        )
        continuation_text = str(continuation.renderable)
        assert "\n" not in continuation_text
        assert "⏎" in continuation_text
        assert "The sky is **blue" in continuation_text


@pytest.mark.asyncio
async def test_long_continuation_is_truncated_with_an_ellipsis(
    evals_app, bench_with_continuation_samples
):
    task_id = bench_with_continuation_samples
    index = _CONTINUATION_TARGET_INDEX["long"]
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app.screen

        continuation = screen.query_one(
            f"#evals-inspector-target-continuation-{index}"
        )
        continuation_text = str(continuation.renderable)
        assert continuation_text.endswith("…")
        expected = (
            inspector_module._CONTINUATION_LABEL
            + _LONG_CONTINUATION[: inspector_module._CONTINUATION_PREVIEW_MAX_LEN]
            + "…"
        )
        assert continuation_text == expected
        # Genuinely bounded, not merely ending with an ellipsis while still
        # carrying the full 150-character continuation ahead of it.
        assert len(continuation_text) < len(
            inspector_module._CONTINUATION_LABEL
        ) + len(_LONG_CONTINUATION)


@pytest.mark.asyncio
async def test_historical_preflight_without_a_continuation_still_renders_the_readiness_list(
    evals_app, bench_with_mixed_readiness
):
    """`bench_with_mixed_readiness`'s `PreflightResult`s were all built
    without a `continuation=` kwarg -- exactly how a run snapshot recorded
    before task-1691 loads back (`storage._preflight_from_snapshot`
    defaults a missing sub-key to `""`, per task-1691 Task 1's own report).
    Every target's readiness badge must still render, and none of them
    gets a continuation sub-line.
    """
    task_id, target_ids = bench_with_mixed_readiness
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app.screen

        for index in _TARGET_INDEX.values():
            badge = screen.query_one(f"#evals-inspector-target-{index}")
            assert badge.region.width > 0
            assert badge.region.height > 0
            assert not screen.query(
                f"#evals-inspector-target-continuation-{index}"
            )


# ---------------------------------------------------------------------------
# Un-preflighted state: a bench that has never run
# ---------------------------------------------------------------------------


def test_never_run_bench_renders_unpreflighted_state(evals_db, never_run_bench):
    """Exercise the production inspector and action-state functions directly."""
    inspector = EvalsInspector(
        EvalsViewModel(evals_db),
        never_run_bench,
        preflight={},
    )
    widgets = list(inspector.compose())
    targets = [
        widget
        for widget in widgets
        if widget.id == "evals-inspector-target-0"
    ]

    assert targets, "expected an un-preflighted status row"
    text = str(targets[0].renderable)
    assert "Not yet checked" in text, text
    assert "Ready" not in text
    assert "Blocked" not in text
    assert "Unavailable" not in text
    assert not any(
        widget.id == "evals-inspector-target-callout-0" for widget in widgets
    )

    screen = object.__new__(EvalsScreen)
    screen._view_model = EvalsViewModel(evals_db)
    screen._selection = EvalsSelection(kind="bench", id=never_run_bench)
    screen._bench_run_running = False
    screen._sample_bench_running = False

    label, disabled, tooltip = screen._primary_action_state()

    assert label == "Run never-run bench"
    assert disabled is False
    assert tooltip == "Runs never-run bench against its configured targets."


@pytest.mark.asyncio
async def test_preflight_is_resolved_once_per_bench_selection_not_twice(
    evals_app, bench_with_mixed_readiness, monkeypatch
):
    """I2: BenchEditor and EvalsInspector each used to call
    ``EvalsViewModel.preflight_for_bench`` independently from their own
    ``compose()``, so selecting a bench read (and, before the ``load_grid``
    -> ``load_run_preflight`` fix, fully paged) the bench's run-group
    snapshot twice on one render. ``evals_screen.py`` now resolves it once
    per selection and threads the same map into both widgets'
    constructors -- proven here by counting real calls through a
    monkeypatched wrapper, not by reading the source."""
    task_id, _ = bench_with_mixed_readiness

    call_count = 0
    original = EvalsViewModel.preflight_for_bench

    def _counting(self, bench_id):
        nonlocal call_count
        call_count += 1
        return original(self, bench_id)

    monkeypatch.setattr(EvalsViewModel, "preflight_for_bench", _counting)

    async with evals_app.run_test() as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()

    assert call_count == 1, f"expected exactly one preflight resolution, got {call_count}"


@pytest.mark.asyncio
async def test_bench_editor_and_inspector_never_import_the_runner_or_capture_client():
    """Preflighting on render would fire network calls on every selection
    change and could report a verdict no run ever used -- pin that neither
    new module can reach the provider at all, not just that today's
    compose() happens not to call it."""
    for module in (bench_editor_module, inspector_module):
        source = Path(module.__file__).read_text()
        assert "capture_client" not in source
        assert "WordBenchRunner" not in source
        assert "CaptureClientLike" not in source


# ---------------------------------------------------------------------------
# Requirement 5: estimate shows call count + time; cost only for paid
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_estimate_shows_call_count_and_time_and_no_cost_for_local_targets(
    evals_app, bench_with_mixed_readiness
):
    task_id, _ = bench_with_mixed_readiness
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app.screen

        calls_line = screen.query_one("#evals-inspector-estimate-calls")
        calls_text = str(calls_line.renderable)
        # dataset.metadata.sample_count=12 * 3 targets = 36 calls (see the
        # bench_with_mixed_readiness fixture: a 12-sample dataset and three
        # targets). Asserted as the exact rendered token -- a bare "3" in
        # "36" would also satisfy a substring check and could not tell a
        # correct formula from a materially wrong one. Anchored with
        # startswith rather than a bare "in" check so a stray extra digit
        # (e.g. a bug computing 136) could not slip past a substring match.
        assert calls_text.startswith("36 calls"), calls_text

        cost_line = screen.query_one("#evals-inspector-estimate-cost")
        cost_text = str(cost_line.renderable)
        assert "local" in cost_text
        assert "no cost" in cost_text

        for estimate_widget in (calls_line, cost_line):
            assert estimate_widget.region.width > 0
            assert estimate_widget.region.height > 0


@pytest.fixture
def bench_with_capture_continuations_on_raw(evals_db: EvalsDB) -> str:
    """task-1710 T2: a raw-mode bench with the opt-in ON -- 2 targets x a
    10-sample dataset, chosen so the arithmetic is easy to hand-verify:
    20 measured calls, +20 for the per-cell continuation (one genuinely
    separate raw-mode request per cell -- see ``capture_with_
    continuation``'s own docstring, task-1710 T1), 40 total."""
    target_a = _make_model(evals_db, "raw-target-a")
    target_b = _make_model(evals_db, "raw-target-b")
    dataset_id = evals_db.create_dataset(
        name="continuation-cost-raw",
        format="custom",
        source_path="inline:continuation-cost-raw",
        metadata={"sample_count": 10},
    )
    config = BenchConfig(
        name="continuation cost raw v1",
        prompt_mode="raw",
        top_k=20,
        dataset_id=dataset_id,
        target_ids=(target_a, target_b),
        capture_continuations=True,
    )
    return save_bench(evals_db, config)


@pytest.fixture
def bench_with_capture_continuations_off_raw(evals_db: EvalsDB) -> str:
    """The flag-off sibling of ``bench_with_capture_continuations_on_raw``
    -- same shape (2 targets x a 10-sample dataset), so the ONLY variable
    between the two fixtures' own Estimate call counts is the flag."""
    target_a = _make_model(evals_db, "raw-target-a")
    target_b = _make_model(evals_db, "raw-target-b")
    dataset_id = evals_db.create_dataset(
        name="continuation-cost-raw-off",
        format="custom",
        source_path="inline:continuation-cost-raw-off",
        metadata={"sample_count": 10},
    )
    config = BenchConfig(
        name="continuation cost raw off v1",
        prompt_mode="raw",
        top_k=20,
        dataset_id=dataset_id,
        target_ids=(target_a, target_b),
        capture_continuations=False,
    )
    return save_bench(evals_db, config)


@pytest.fixture
def bench_with_capture_continuations_on_chat(evals_db: EvalsDB) -> str:
    """task-1710 T2: a CHAT-mode bench with the opt-in ON -- same 2 x 10
    shape as the raw-mode fixtures above, but the flag must add NOTHING
    here: chat mode salvages the continuation off the measurement
    response already made (``_resolve_continuation``'s own docstring),
    so 20 calls stays 20, never 40. Proves the Estimate does not just
    blindly double whenever the flag is on."""
    target_a = _make_model(evals_db, "chat-target-a")
    target_b = _make_model(evals_db, "chat-target-b")
    dataset_id = evals_db.create_dataset(
        name="continuation-cost-chat",
        format="custom",
        source_path="inline:continuation-cost-chat",
        metadata={"sample_count": 10},
    )
    config = BenchConfig(
        name="continuation cost chat v1",
        prompt_mode="chat",
        top_k=20,
        dataset_id=dataset_id,
        target_ids=(target_a, target_b),
        capture_continuations=True,
    )
    return save_bench(evals_db, config)


@pytest.mark.asyncio
async def test_estimate_reflects_doubled_calls_when_capture_continuations_is_on_in_raw_mode(
    evals_app, bench_with_capture_continuations_on_raw
):
    """task-1710 AC: "with it on, the Estimate reflects the added calls
    before the run starts." Raw mode's per-cell continuation is one
    genuinely separate extra request per cell, so 2 targets x 10 samples
    (20 measured calls) becomes 40, not 20 -- pinned as the exact leading
    token, the same anchoring convention `test_estimate_shows_call_count_
    and_time_and_no_cost_for_local_targets` above uses, so a bug computing
    e.g. 120 could not slip past a bare substring check."""
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=bench_with_capture_continuations_on_raw)
        await pilot.pause()
        screen = evals_app.screen

        calls_line = screen.query_one("#evals-inspector-estimate-calls")
        calls_text = str(calls_line.renderable)
        assert calls_text.startswith("40 calls"), calls_text


@pytest.mark.asyncio
async def test_estimate_is_unchanged_when_capture_continuations_is_off_in_raw_mode(
    evals_app, bench_with_capture_continuations_off_raw
):
    """The flag-off sibling of the test above -- task-1710 AC: "with it
    off, request count per cell is unchanged from today." Same 2 x 10
    shape, flag off: 20 calls, not 40."""
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=bench_with_capture_continuations_off_raw)
        await pilot.pause()
        screen = evals_app.screen

        calls_line = screen.query_one("#evals-inspector-estimate-calls")
        calls_text = str(calls_line.renderable)
        assert calls_text.startswith("20 calls"), calls_text


@pytest.mark.asyncio
async def test_estimate_is_unchanged_when_capture_continuations_is_on_in_chat_mode(
    evals_app, bench_with_capture_continuations_on_chat
):
    """Chat mode salvages the continuation from the measurement response
    already made -- the flag must NOT double the estimate here, unlike
    raw mode. Without this test, a formula that always doubles whenever
    the flag is on (ignoring `prompt_mode` entirely) would still pass the
    raw-mode test above while silently lying to a chat-mode bench author
    about a cost they will never actually pay."""
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=bench_with_capture_continuations_on_chat)
        await pilot.pause()
        screen = evals_app.screen

        calls_line = screen.query_one("#evals-inspector-estimate-calls")
        calls_text = str(calls_line.renderable)
        assert calls_text.startswith("20 calls"), calls_text


@pytest.mark.asyncio
async def test_estimate_cost_line_is_not_no_cost_for_a_paid_target(
    evals_app, paid_target_bench
):
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=paid_target_bench)
        await pilot.pause()
        screen = evals_app.screen

        cost_line = screen.query_one("#evals-inspector-estimate-cost")
        cost_text = str(cost_line.renderable)
        assert "no cost" not in cost_text


@pytest.mark.asyncio
async def test_estimate_cost_is_unknown_when_all_targets_are_unresolvable(
    evals_app, bench_with_all_targets_deleted
):
    """A bench whose targets have all been deleted contributes no
    providers at all -- `providers` (built only from resolved targets) was
    empty, so the estimate fell through to `else: "local · no cost"` -- a
    wrong claim about money for a provider that was never actually
    resolved as local."""
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=bench_with_all_targets_deleted)
        await pilot.pause()
        screen = evals_app.screen

        cost_line = screen.query_one("#evals-inspector-estimate-cost")
        cost_text = str(cost_line.renderable)
        assert "no cost" not in cost_text
        assert "unknown" in cost_text
        # TASK-1481 fix-round-1: this line used ASCII "--" where the rest
        # of the Evals rail copy uses real em-dashes.
        assert " -- " not in cost_text
        assert "—" in cost_text


@pytest.mark.asyncio
async def test_estimate_cost_is_unknown_when_any_target_is_unresolvable(
    evals_app, bench_with_one_deleted_and_one_live_local_target
):
    """"Any target unresolvable" -- not "every target unresolvable" -- is
    the bar: one resolvable LOCAL target must not make the cost line read
    "local · no cost" while a sibling target's provider is genuinely
    unknown."""
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        evals_app.screen.select(
            kind="bench", id=bench_with_one_deleted_and_one_live_local_target
        )
        await pilot.pause()
        screen = evals_app.screen

        cost_line = screen.query_one("#evals-inspector-estimate-cost")
        cost_text = str(cost_line.renderable)
        assert "no cost" not in cost_text
        assert "unknown" in cost_text


# ---------------------------------------------------------------------------
# Requirement 4: classic tasks are read-only, no run control
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_classic_task_detail_shows_run_history_and_deferral_sentence(
    evals_app, classic_task_with_runs
):
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="classic", id=classic_task_with_runs)
        await pilot.pause()
        screen = evals_app.screen

        name = screen.query_one("#evals-detail-classic-name")
        assert "mmlu-subset" in str(name.renderable)

        run_row = screen.query_one("#evals-detail-classic-run-0")
        run_text = str(run_row.renderable)
        assert "gpt-4o-mini" in run_text
        assert "completed" in run_text

        deferral = screen.query_one("#evals-detail-classic-deferral")
        assert str(deferral.renderable).strip() == CLASSIC_TASK_DEFERRAL_SENTENCE

        # Pinned requirement 4 (classic tasks: read-only detail + run
        # history + deferral sentence) -- prove the whole detail is
        # genuinely rendered, not merely present in the DOM.
        for classic_widget in (name, run_row, deferral):
            assert classic_widget.region.width > 0
            assert classic_widget.region.height > 0


@pytest.mark.asyncio
async def test_classic_task_detail_heading_renders_a_markup_hazard_name_literally(
    evals_app, classic_task_with_markup_hazard_name
):
    """task-1482 Task 1 fix round 1. Selecting a classic task named with a
    bare `[/]` must not crash the app (the same lazy-parse-at-layout hazard
    ``test_bench_editor_statics_render_a_markup_hazard_literally`` pins for
    ``BenchEditor`` -- see that test's own docstring for why the crash
    surfaces at layout, not ``compose()``), and the heading Static's
    parsed plain text must round-trip the raw, unmangled name.
    """
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="classic", id=classic_task_with_markup_hazard_name)
        await pilot.pause()
        assert pilot.app.is_running, "an unescaped hazard name crashed the app"
        screen = evals_app.screen

        name = screen.query_one("#evals-detail-classic-name")
        assert name.visual.plain == "classic[/]name"


@pytest.mark.asyncio
async def test_classic_task_subgroup_is_reachable_by_clicking_its_rail_row(
    evals_app, classic_task_with_runs
):
    """I3: the design spec requires classic tasks to "appear in a labelled
    subgroup under Benches" -- the rail used to render only
    ``view_model.benches()``, filtering classic rows out entirely, so
    ``ClassicTaskDetail``, ``classic_tasks()``, and ``classic_task_by_id()``
    were all dead code reachable only by a test calling
    ``screen.select(kind="classic", ...)`` directly (see both existing
    classic-task tests above) -- never by a real user action. Driven here
    by an actual rail click instead."""
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        screen = evals_app.screen

        separators = screen.query(".evals-rail-classic-separator")
        assert list(separators), "no labelled classic subgroup rendered under Benches"

        await pilot.click("#evals-rail-row-benches-classic-0")
        await pilot.pause()

        name = screen.query_one("#evals-detail-classic-name")
        assert "mmlu-subset" in str(name.renderable)
        assert name.region.width > 0
        assert name.region.height > 0


@pytest.mark.asyncio
async def test_benches_section_header_counts_word_benches_and_classic_tasks_together(
    evals_app, bench_with_mixed_readiness, classic_task_with_runs
):
    """The design mockup's own worked example counts word benches and
    classic tasks together under one "BENCHES (N)" header (2 word benches +
    2 classic tasks -> "BENCHES (4)") -- the header names how many rows
    are under it, not how many word benches exist."""
    task_id, _ = bench_with_mixed_readiness
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        labels = [
            str(w.renderable)
            for w in evals_app.screen.query(".evals-rail-section-label")
        ]
        assert "Benches (2)" in " ".join(labels)


@pytest.mark.asyncio
async def test_classic_task_selection_has_no_run_control(evals_app, classic_task_with_runs):
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="classic", id=classic_task_with_runs)
        await pilot.pause()
        screen = evals_app.screen

        assert not screen.query("#evals-primary-action")
        inspector_pane = screen.query_one("#evals-inspector-pane")
        assert not inspector_pane.query("Button"), (
            "classic-task selection must render no run control at all in "
            "the inspector pane, not merely a disabled one"
        )


# ---------------------------------------------------------------------------
# Qodo #941 finding 2: duplicate target_ids must not crash composition
# ---------------------------------------------------------------------------


@pytest.fixture
def bench_with_duplicate_target_id(evals_db: EvalsDB) -> str:
    """An `eval_tasks` row whose `config_data.target_ids` names the SAME id
    twice -- a legacy bench saved before `BenchConfig.__post_init__` (or
    `create_run_group`) rejected that shape (task-1132's ancestor,
    `git rev` b73de3564).

    Written directly against `EvalsDB.create_task`, NOT through
    `BenchConfig`/`save_bench`: both now reject a duplicate unconditionally
    on the write path (`BenchConfig`'s default `strict=True`, and
    `save_bench`'s own independent check -- see task-1132), so a bench
    actually carrying this shape can now only arise from data written
    before that validation existed. `storage.load_bench` constructs its
    `BenchConfig` with `strict=False` specifically so this legacy shape can
    still be read back rather than becoming permanently unopenable (see
    that function's own docstring) -- this fixture exercises exactly that
    read path.

    Before the id-collision fix (Qodo #941 finding 2, unrelated to
    task-1132), `bench_editor.py`'s target table and `inspector.py`'s
    readiness list each derived a widget id straight from `target_id`
    (`evals-bench-target-<id>`, `evals-inspector-target-<id>`), so a
    duplicate collided at mount time and failed to compose the whole pane
    -- not just the duplicated row.

    Args:
        evals_db: In-memory EvalsDB fixture, written to directly (bypassing
            BenchConfig/save_bench) so it can carry a shape neither will
            construct anymore.

    Returns:
        The `eval_tasks` row id (`task_id`) of the bench carrying the
        duplicate `target_ids`.
    """
    target_id = _make_model(evals_db, "repeated-target")
    dataset_id = evals_db.create_dataset(
        name="dup-target-set",
        format="custom",
        source_path="inline:dup-target-set",
        metadata={"sample_count": 4},
    )
    return evals_db.create_task(
        name="duplicate-target bench",
        task_type="logprob",
        config_format="custom",
        config_data={
            "bench_type": BENCH_TYPE,
            "prompt_mode": "raw",
            "top_k": 20,
            "probes": [],
            "target_ids": [target_id, target_id],
            "concurrency": 1,
        },
        dataset_id=dataset_id,
    )


@pytest.mark.asyncio
async def test_bench_with_duplicate_target_id_composes_without_raising(
    evals_app, bench_with_duplicate_target_id
):
    """Composes the real workbench (not just the id-derivation helper) with
    a bench carrying a genuine duplicate target id, and asserts both the
    detail pane's target table and the inspector's readiness list render
    every row rather than raising out of compose().

    Args:
        evals_app: The Evals screen's app-under-test fixture.
        bench_with_duplicate_target_id: The `task_id` of the legacy bench
            fixture above, carrying a genuine duplicate `target_id`.
    """
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=bench_with_duplicate_target_id)
        await pilot.pause()
        screen = evals_app.screen

        target_rows = screen.query(".evals-bench-target-row")
        assert len(target_rows) == 2, "both duplicate-id target rows should compose"
        for row in target_rows:
            assert row.region.width > 0
            assert row.region.height > 0

        inspector_pane = screen.query_one("#evals-inspector-pane")
        readiness_rows = inspector_pane.query(".evals-status-unchecked")
        assert len(readiness_rows) == 2, "both duplicate-id readiness rows should compose"
        for row in readiness_rows:
            assert row.region.width > 0
            assert row.region.height > 0

        # Distinct, index-derived widget ids despite the shared target_id
        # underneath -- the actual regression check: a target_id-derived id
        # would have raised a MountError composing the second row, well
        # before either of these queries could even run.
        assert screen.query_one("#evals-bench-target-0")
        assert screen.query_one("#evals-bench-target-1")
        assert screen.query_one("#evals-inspector-target-0")
        assert screen.query_one("#evals-inspector-target-1")


# ---------------------------------------------------------------------------
# Task 5 (task-1482): BenchEditor becomes a form -- Save/Revert
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_persists_every_field_and_reselects_the_bench(
    evals_app, evals_db, bench_with_mixed_readiness
):
    """The write half of the field round-trip: every editable field, typed
    into and saved, lands in storage exactly as typed, `target_ids` pass
    through untouched (Task 5 does not edit targets), and a successful
    Save posts `Saved` -> the screen re-selects the same bench, recomposing
    the form from what was actually persisted."""
    task_id, target_ids = bench_with_mixed_readiness
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app.screen

        screen.query_one("#evals-bench-name", Input).value = "loaded-nouns v2"
        screen.query_one("#evals-bench-description", Input).value = "a new description"
        screen.query_one("#evals-bench-prompt-mode", Select).value = "chat"
        screen.query_one("#evals-bench-top-k", Input).value = "5"
        screen.query_one("#evals-bench-probes", TextArea).text = " Sure\n No way"

        await pilot.click("#evals-bench-save")
        await pilot.pause()

        # No error callout on a successful save. Scoped to the form's own
        # error Static, not a screen-wide `.ds-recovery-callout` query --
        # `bench_with_mixed_readiness`'s warned target legitimately renders
        # its OWN `.ds-recovery-callout` in the readiness inspector,
        # unrelated to this form's Save outcome.
        assert not screen.query_one("#evals-bench-form-error").display

        saved = load_bench(evals_db, task_id)
        assert saved.name == "loaded-nouns v2"
        assert saved.description == "a new description"
        assert saved.prompt_mode == "chat"
        assert saved.top_k == 5
        # Byte-exact round-trip, including the leading space on " Sure".
        assert saved.probes == (" Sure", " No way")
        assert set(saved.target_ids) == set(target_ids.values())

        assert screen._selection.kind == "bench"
        assert screen._selection.id == task_id
        assert screen.query_one("#evals-bench-name", Input).value == "loaded-nouns v2"
        assert screen.query_one("#evals-bench-prompt-mode", Select).value == "chat"
        assert screen.query_one("#evals-bench-probes", TextArea).text == " Sure\n No way"


@pytest.mark.asyncio
async def test_trailing_newline_after_the_last_probe_does_not_persist_an_empty_probe(
    evals_app, evals_db, bench_with_mixed_readiness
):
    """Fix round 1 (reviewer Important, bench_editor.py:362): a user who
    presses Enter after typing the last probe leaves a genuine zero-length
    line in `TextArea.text` -- `BenchConfig` accepts an empty-string probe
    happily, and `analysis.resolve_probe` would then carry a meaningless
    probe column all the way through a run. Only the trailing empty line
    must be dropped; the two real probes must still round-trip byte-exact."""
    task_id, _ = bench_with_mixed_readiness
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app.screen

        screen.query_one("#evals-bench-probes", TextArea).text = " Sure\n No way\n"
        await pilot.click("#evals-bench-save")
        await pilot.pause()

        assert not screen.query_one("#evals-bench-form-error").display
        saved = load_bench(evals_db, task_id)
        assert saved.probes == (" Sure", " No way")
        assert "" not in saved.probes


@pytest.mark.asyncio
async def test_empty_probes_textarea_saves_zero_probes(
    evals_app, evals_db, bench_with_mixed_readiness
):
    """An entirely empty TextArea means zero probes, not one empty-string
    probe."""
    task_id, _ = bench_with_mixed_readiness
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app.screen

        screen.query_one("#evals-bench-probes", TextArea).text = ""
        await pilot.click("#evals-bench-save")
        await pilot.pause()

        assert not screen.query_one("#evals-bench-form-error").display
        saved = load_bench(evals_db, task_id)
        assert saved.probes == ()


@pytest.mark.asyncio
async def test_whitespace_only_probe_line_is_kept_byte_exact(
    evals_app, evals_db, bench_with_mixed_readiness
):
    """A lone space is a legitimate exact token, not an empty one --
    "whitespace preserved exactly" is a claim about a token's CONTENT, and
    only a genuinely ZERO-LENGTH line is dropped (see the trailing-newline
    test above), never a whitespace-only one.

    Mutation check (pinned by the controller): changing this handler's
    filter to also drop whitespace-only lines (e.g. `if line.strip()`
    instead of `if line != ""`) makes this exact test fail -- confirmed by
    hand, then reverted; see task-5-report.md's "Fix round 1" section."""
    task_id, _ = bench_with_mixed_readiness
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app.screen

        screen.query_one("#evals-bench-probes", TextArea).text = " Sure\n \n No way"
        await pilot.click("#evals-bench-save")
        await pilot.pause()

        assert not screen.query_one("#evals-bench-form-error").display
        saved = load_bench(evals_db, task_id)
        assert saved.probes == (" Sure", " ", " No way")


@pytest.mark.asyncio
async def test_top_k_parse_failure_renders_the_pinned_callout_and_keeps_typed_state(
    evals_app, bench_with_mixed_readiness
):
    """Pinned exact error text, and the "no recompose on failure" contract:
    every OTHER field's unsaved edit survives a failed Save untouched, and
    the field widgets are the literal same instances -- proof this is an
    in-place callout update, not a rebuild from the last-saved config."""
    task_id, _ = bench_with_mixed_readiness
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app.screen

        name_input = screen.query_one("#evals-bench-name", Input)
        name_input.value = "renamed-but-not-saved"
        screen.query_one("#evals-bench-top-k", Input).value = "abc"
        await pilot.click("#evals-bench-save")
        await pilot.pause()

        callout = screen.query_one("#evals-bench-form-error")
        assert callout.display
        assert "ds-recovery-callout" in callout.classes
        assert str(callout.renderable) == TOP_K_ERROR_TEXT

        assert screen.query_one("#evals-bench-name", Input) is name_input
        assert name_input.value == "renamed-but-not-saved"


@pytest.mark.asyncio
async def test_blank_name_save_failure_renders_the_db_rejection_callout(
    evals_app, bench_with_mixed_readiness
):
    """A whitespace-only name passes `BenchConfig`'s own construction (it
    has no name check) but is rejected inside `save_bench` -> `Evals_DB.
    update_task` -> `_clean_task_name`, which raises `InputError` -- a
    `ValueError` subclass this handler's `except ValueError` branch must
    catch."""
    task_id, _ = bench_with_mixed_readiness
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app.screen

        screen.query_one("#evals-bench-name", Input).value = "   "
        await pilot.click("#evals-bench-save")
        await pilot.pause()

        callout = screen.query_one("#evals-bench-form-error")
        assert callout.display
        assert "cannot be empty" in str(callout.renderable)


@pytest.mark.asyncio
async def test_renaming_to_a_taken_name_renders_the_conflict_callout(
    evals_app, evals_db, bench_with_mixed_readiness
):
    """Mutation check (task-1612): dropping `_on_save_pressed`'s
    `except ConflictError` clause makes this test fail -- `Evals_DB.
    update_task` raises `ConflictError` (an `EvalsDBError`, NOT a
    `ValueError`), so it would no longer be caught by the sibling
    `except (ValueError, RuntimeError)` clause and would propagate
    uncaught out of this handler instead of rendering the callout below.

    task-1612: pins the bench-vocabulary copy exactly -- the DB's own
    raw message ("Task name already exists") says "Task", not "bench",
    and never mentions that `eval_tasks.name`'s UNIQUE index has no
    `deleted_at` exemption (a deleted bench's name stays reserved), so a
    prior substring-only assertion here would have tolerated either the
    raw DB copy or this replacement."""
    task_id, _ = bench_with_mixed_readiness
    other_dataset_id = evals_db.create_dataset(
        name="other-dataset", format="custom", source_path="inline:other-dataset"
    )
    save_bench(
        evals_db,
        BenchConfig(
            name="taken-name",
            prompt_mode="raw",
            top_k=5,
            dataset_id=other_dataset_id,
            target_ids=(),
        ),
    )

    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app.screen

        screen.query_one("#evals-bench-name", Input).value = "taken-name"
        await pilot.click("#evals-bench-save")
        await pilot.pause()

        callout = screen.query_one("#evals-bench-form-error")
        assert callout.display
        assert str(callout.renderable) == (
            'A bench named "taken-name" already exists -- choose a '
            "different name. (Deleting a bench does not free its name: "
            "a deleted bench may still be holding it.)"
        )

        # No recompose on failure: the Input still shows what was typed.
        assert screen.query_one("#evals-bench-name", Input).value == "taken-name"


@pytest.mark.asyncio
async def test_saving_a_bench_deleted_elsewhere_renders_an_error_not_a_false_success(
    evals_app, evals_db, bench_with_mixed_readiness
):
    """PR #1138 review (Bug, accepted): the bench was deleted -- e.g. by a
    second app instance -- between this editor loading it and Save being
    pressed. `save_bench`'s update branch now raises `RuntimeError` when
    `update_task` matched no row (see storage.py's own fix). This asserts
    the Save handler catches it, renders the exact message in the form
    callout, posts NO `Saved` message (no false "success"), and leaves the
    form's own state -- the typed, unsaved value -- untouched, exactly
    like every other Save failure path (no recompose)."""
    task_id, _ = bench_with_mixed_readiness
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app.screen

        name_input = screen.query_one("#evals-bench-name", Input)
        name_input.value = "renamed-but-bench-is-gone"

        # Simulates a second app instance deleting the bench in the window
        # between this editor loading it and Save being pressed.
        evals_db.delete_task(task_id)

        await pilot.click("#evals-bench-save")
        await pilot.pause()

        callout = screen.query_one("#evals-bench-form-error")
        assert callout.display
        assert str(callout.renderable) == (
            "This bench no longer exists; it may have been deleted elsewhere."
        )

        # No recompose: the SAME Input instance, still carrying the typed
        # (unsaved) value -- proves this was an in-place callout update,
        # not a rebuild from a (now-nonexistent) reload.
        assert screen.query_one("#evals-bench-name", Input) is name_input
        assert name_input.value == "renamed-but-bench-is-gone"

        # No Saved-adjacent side effect: the selection is untouched (no
        # `select()` call was ever triggered off a `Saved` message the
        # handler must not have posted).
        assert screen._selection.kind == "bench"
        assert screen._selection.id == task_id


@pytest.mark.asyncio
async def test_prompt_mode_switch_revalidates_targets_and_names_the_offending_target(
    evals_app, bench_with_mixed_readiness, monkeypatch
):
    """The prompt-mode/target revalidation seam, exercised via a hand-built
    `Target` (monkeypatching `_resolve_bench_targets` directly, rather than
    seeding a real steered `eval_models` row -- see
    `test_prompt_mode_switch_with_a_real_steered_target_blocks_save_with_
    reworded_copy` below for the task-1611 T2 real-row equivalent of this
    same check) carrying a `prefix` -- invalid the instant the mode
    switches to "chat" (`Target.is_valid_for_mode`: chat requires `prefix
    is None`)."""
    task_id, target_ids = bench_with_mixed_readiness

    def _fake_resolve_bench_targets(db, ids):
        return [
            Target(
                id=target_ids["ready"],
                name="ready-target",
                provider="llama_cpp",
                model_id="m",
                prefix="Continue the story:",
            )
        ]

    monkeypatch.setattr(
        bench_editor_module, "_resolve_bench_targets", _fake_resolve_bench_targets
    )

    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app.screen

        screen.query_one("#evals-bench-prompt-mode", Select).value = "chat"
        await pilot.click("#evals-bench-save")
        await pilot.pause()

        callout = screen.query_one("#evals-bench-form-error")
        assert callout.display
        text = str(callout.renderable)
        assert "ready-target" in text
        assert "chat" in text


@pytest.mark.asyncio
async def test_revert_discards_unsaved_edits_and_reloads_from_storage(
    evals_app, bench_with_mixed_readiness
):
    """Revert = re-selecting this same bench: the screen's own `select()`
    recompose reloads every field from storage, which is what "revert"
    means here (there is no separate in-memory draft -- the fields ARE the
    widgets)."""
    task_id, _ = bench_with_mixed_readiness
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app.screen

        screen.query_one("#evals-bench-name", Input).value = "typed-but-not-saved"
        await pilot.click("#evals-bench-revert")
        await pilot.pause()

        assert screen.query_one("#evals-bench-name", Input).value == "loaded-nouns v1"
        # Scoped to the form's own error Static -- see the identical note
        # in test_save_persists_every_field_and_reselects_the_bench.
        assert not screen.query_one("#evals-bench-form-error").display


# ---------------------------------------------------------------------------
# Task 6 (task-1482): targets become editable -- Add/Remove + create-target
# ---------------------------------------------------------------------------


@pytest.fixture
def bench_with_available_add_target(evals_db: EvalsDB) -> tuple[str, str, str]:
    """One bench with one target already wired, plus a SECOND `llama_cpp`
    `eval_models` row not yet on the bench -- exactly what the Add picker
    needs to offer a genuine, addable option.

    Returns:
        ``(task_id, existing_target_id, addable_target_id)``.
    """
    existing_id = _make_model(evals_db, "existing-target")
    addable_id = _make_model(evals_db, "extra-target", model_id="m2")
    dataset_id = evals_db.create_dataset(
        name="add-target-set",
        format="custom",
        source_path="inline:add-target-set",
        metadata={"sample_count": 4},
    )
    config = BenchConfig(
        name="add-target bench",
        prompt_mode="raw",
        top_k=20,
        dataset_id=dataset_id,
        target_ids=(existing_id,),
    )
    task_id = save_bench(evals_db, config)
    return task_id, existing_id, addable_id


@pytest.fixture
def bench_with_zero_llama_models(evals_db: EvalsDB) -> str:
    """A draft bench (no targets) with NO `llama_cpp` `eval_models` row
    anywhere in the db -- the zero-models "Create target" affordance's own
    gate."""
    dataset_id = evals_db.create_dataset(
        name="zero-models-set",
        format="custom",
        source_path="inline:zero-models-set",
        metadata={"sample_count": 4},
    )
    config = BenchConfig(
        name="zero-models bench",
        prompt_mode="raw",
        top_k=20,
        dataset_id=dataset_id,
        target_ids=(),
    )
    return save_bench(evals_db, config)


@pytest.fixture
def bench_with_markup_hazard_llama_model(evals_db: EvalsDB) -> tuple[str, str]:
    """A draft bench (no targets) plus one already-registered `llama_cpp`
    model whose name carries a bare `[/]` -- the Add picker's own option
    labels ("name (model_id)") must escape it, matching every other
    user-authored string this widget renders (see the module docstring's
    markup-hazard sweep)."""
    hazard_id = _make_model(evals_db, "loud[/]target", model_id="m")
    dataset_id = evals_db.create_dataset(
        name="hazard-picker-set",
        format="custom",
        source_path="inline:hazard-picker-set",
        metadata={"sample_count": 4},
    )
    config = BenchConfig(
        name="hazard-picker bench",
        prompt_mode="raw",
        top_k=20,
        dataset_id=dataset_id,
        target_ids=(),
    )
    task_id = save_bench(evals_db, config)
    return task_id, hazard_id


@pytest.mark.asyncio
async def test_add_from_picker_stages_a_row_and_save_persists_it(
    evals_app, evals_db, bench_with_available_add_target
):
    task_id, existing_id, addable_id = bench_with_available_add_target
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app.screen

        assert screen.query_one("#evals-bench-target-0")
        assert not screen.query("#evals-bench-target-1")

        select = screen.query_one("#evals-bench-add-target", Select)
        select.value = addable_id
        await pilot.click("#evals-bench-add-target-button")
        await pilot.pause()

        # The staged row renders immediately, before Save, with the
        # "never checked" status -- it was never part of the bench's last
        # run snapshot.
        new_row = screen.query_one("#evals-bench-target-1")
        row_text = str(new_row.renderable)
        assert "extra-target" in row_text
        assert "Not yet checked" in row_text
        assert new_row.region.width > 0
        assert new_row.region.height > 0

        await pilot.click("#evals-bench-save")
        await pilot.pause()

        assert not screen.query_one("#evals-bench-form-error").display
        saved = load_bench(evals_db, task_id)
        assert set(saved.target_ids) == {existing_id, addable_id}


@pytest.mark.asyncio
async def test_remove_target_row_and_save_persists_the_removal(
    evals_app, evals_db, bench_with_mixed_readiness
):
    task_id, target_ids = bench_with_mixed_readiness
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app.screen

        # Remove the "ready" target (index 0, per _TARGET_INDEX).
        await pilot.click("#evals-bench-target-remove-0")
        await pilot.pause()

        # Rows re-index: the remaining two targets now sit at 0/1, and
        # there is no longer a row at 2.
        assert screen.query_one("#evals-bench-target-0")
        assert screen.query_one("#evals-bench-target-1")
        assert not screen.query("#evals-bench-target-2")

        await pilot.click("#evals-bench-save")
        await pilot.pause()

        assert not screen.query_one("#evals-bench-form-error").display
        saved = load_bench(evals_db, task_id)
        assert set(saved.target_ids) == {target_ids["warned"], target_ids["blocked"]}


@pytest.mark.asyncio
async def test_duplicate_add_is_rejected_inline_with_the_pinned_text(
    evals_app, evals_db, bench_with_available_add_target
):
    task_id, existing_id, _addable_id = bench_with_available_add_target
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app.screen

        select = screen.query_one("#evals-bench-add-target", Select)
        select.value = existing_id
        await pilot.click("#evals-bench-add-target-button")
        await pilot.pause()

        callout = screen.query_one("#evals-bench-form-error")
        assert callout.display
        assert str(callout.renderable) == "Target already on this bench."

        # No second row was staged -- the rejected add left the bench's
        # target list untouched.
        assert not screen.query("#evals-bench-target-1")

        # And a save persists exactly what was there before the rejected
        # add, not a duplicate. task-1710: the per-cell continuation
        # checkbox added one row to this form, and the error callout
        # above already pushes this pane close to `#evals-bench-editor`'s
        # own documented scroll threshold at a realistic 160x45 viewport
        # (see that id's CSS comment) -- `scroll_visible` first, the same
        # convention this file's own inspector-pane geometry tests use,
        # rather than assume Save always fits one screenful below an
        # error callout.
        save_button = screen.query_one("#evals-bench-save", Button)
        save_button.scroll_visible(animate=False)
        await pilot.pause()
        await pilot.click("#evals-bench-save")
        await pilot.pause()
        saved = load_bench(evals_db, task_id)
        assert saved.target_ids == (existing_id,)


@pytest.mark.asyncio
async def test_zero_models_state_offers_create_target_and_stages_it(
    evals_app_configured, evals_db, bench_with_zero_llama_models
):
    task_id = bench_with_zero_llama_models
    assert evals_db.list_models(provider="llama_cpp") == []

    async with evals_app_configured.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app_configured.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app_configured.screen

        assert not screen.query("#evals-bench-add-target")
        create_button = screen.query_one("#evals-bench-create-target", Button)
        assert create_button.region.width > 0
        assert create_button.region.height > 0

        await pilot.click("#evals-bench-create-target")
        await pilot.pause()

        models = evals_db.list_models(provider="llama_cpp")
        assert len(models) == 1, "pressing Create target should create exactly one row"

        row = screen.query_one("#evals-bench-target-0")
        row_text = str(row.renderable)
        assert models[0]["name"] in row_text
        assert "Not yet checked" in row_text

        await pilot.click("#evals-bench-save")
        await pilot.pause()

        assert not screen.query_one("#evals-bench-form-error").display
        saved = load_bench(evals_db, task_id)
        assert saved.target_ids == (models[0]["id"],)


@pytest.mark.asyncio
async def test_create_target_with_no_llama_cpp_server_configured_notifies_and_creates_nothing(
    evals_app, evals_db, bench_with_zero_llama_models
):
    """task-1613: pins the zero-config error path for the "+ New target"
    mini-form's button (``#evals-bench-create-target`` -- task-1611 T2
    renamed its LABEL from "Create target from configured llama.cpp
    server" to "+ New target", but kept the id and the gate this test
    drives). Unlike every other create-target test in this module, this
    one uses the bare ``evals_app`` fixture (an empty ``app_config``, no
    llama_cpp URL set) rather than ``evals_app_configured`` -- exercising
    ``evals_screen.py``'s ``_on_bench_create_target_requested`` early
    return on ``sample_bench.configured_llama_cpp_url(app_config) is
    None``, BEFORE it ever reaches ``db.create_model``."""
    task_id = bench_with_zero_llama_models
    assert evals_db.list_models(provider="llama_cpp") == []

    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app.screen

        create_button = screen.query_one("#evals-bench-create-target", Button)
        assert create_button.region.width > 0

        await pilot.click("#evals-bench-create-target")
        await pilot.pause()

        assert evals_app.app_instance.notifications
        message, severity = evals_app.app_instance.notifications[-1]
        assert severity == "error"
        assert message == (
            "No llama.cpp server is configured; set one in Settings first."
        )

        assert evals_db.list_models(provider="llama_cpp") == [], (
            "the zero-config click must not create an eval_models row"
        )
        assert not screen.query("#evals-bench-target-0"), "nothing was staged"


@pytest.mark.asyncio
async def test_add_target_picker_option_labels_escape_markup_hazard_names(
    evals_app, bench_with_markup_hazard_llama_model
):
    task_id, hazard_id = bench_with_markup_hazard_llama_model
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        assert pilot.app.is_running, "an unescaped picker label crashed the app"
        screen = evals_app.screen

        select = screen.query_one("#evals-bench-add-target", Select)
        option_texts = [str(label) for label, _value in select._options]
        expected = escape_markup("loud[/]target (m)")
        assert any(text == expected for text in option_texts), option_texts


@pytest.fixture
def bench_with_two_saved_targets_only(evals_db: EvalsDB) -> tuple[str, dict[str, str]]:
    """Two targets, both already saved on the bench, and NO other
    `llama_cpp` `eval_models` row anywhere in the db -- "zero llama models
    beyond them", per the whole-branch pre-PR review's own scenario for
    the staged-edit-survival regression test below. Two (not one, not
    three) so the test can drive two SEPARATE staged Remove mutations in a
    row, each independently re-checked against the same unsaved Name/
    probe text.
    """
    first_id = _make_model(evals_db, "first-target", model_id="m1")
    second_id = _make_model(evals_db, "second-target", model_id="m2")
    dataset_id = evals_db.create_dataset(
        name="two-target-set",
        format="custom",
        source_path="inline:two-target-set",
        metadata={"sample_count": 4},
    )
    config = BenchConfig(
        name="two-target bench",
        prompt_mode="raw",
        top_k=20,
        dataset_id=dataset_id,
        target_ids=(first_id, second_id),
    )
    task_id = save_bench(evals_db, config)
    return task_id, {"first": first_id, "second": second_id}


@pytest.mark.asyncio
async def test_staged_target_edits_survive_unsaved_name_and_probe_text(
    evals_app, bench_with_two_saved_targets_only
):
    """Whole-branch pre-PR review: every OTHER Task 6 test drives a target
    mutation against a form nobody has typed into, so nothing previously
    proved the "targeted refresh, not a whole-widget recompose" contract
    `_refresh_targets_section`'s own docstring claims -- a future
    refactor to a bare `self.refresh(recompose=True)` there would pass
    every existing test while silently discarding a user's unsaved Name/
    Probes edit the instant they touched Add or Remove. This types into
    both, then drives TWO separate staged Remove mutations in a row,
    re-asserting survival after each -- not just the first, since a
    recompose-based regression could plausibly be masked by state that
    happens to survive exactly one refresh but not a second.
    """
    task_id, target_ids = bench_with_two_saved_targets_only
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app.screen

        name_input = screen.query_one("#evals-bench-name", Input)
        name_input.value = "typed-not-saved"
        probes_area = screen.query_one("#evals-bench-probes", TextArea)
        probes_area.text = "unsaved probe line"

        assert screen.query_one("#evals-bench-target-0")
        assert screen.query_one("#evals-bench-target-1")

        # First staged mutation: remove the first target row.
        await pilot.click("#evals-bench-target-remove-0")
        await pilot.pause()

        assert screen.query_one("#evals-bench-name", Input).value == "typed-not-saved"
        assert screen.query_one("#evals-bench-probes", TextArea).text == "unsaved probe line"
        # Also proves the SAME widget instances survived, not merely
        # matching values from a rebuilt pair -- a recompose would have
        # replaced both with fresh instances reading the last-SAVED
        # config, which happens to have an empty description/name-suffix
        # collision risk this identity check sidesteps entirely.
        assert screen.query_one("#evals-bench-name", Input) is name_input
        assert screen.query_one("#evals-bench-probes", TextArea) is probes_area
        assert screen.query_one("#evals-bench-target-0")
        assert not screen.query("#evals-bench-target-1")

        # Second staged mutation: remove the remaining target too (it is
        # now re-indexed to row 0 -- see `_build_target_row`'s own
        # index-derived-id comment).
        await pilot.click("#evals-bench-target-remove-0")
        await pilot.pause()

        assert screen.query_one("#evals-bench-name", Input).value == "typed-not-saved"
        assert screen.query_one("#evals-bench-probes", TextArea).text == "unsaved probe line"
        assert screen.query_one("#evals-bench-name", Input) is name_input
        assert screen.query_one("#evals-bench-probes", TextArea) is probes_area
        assert screen.query_one("#evals-bench-targets-empty")
        assert not screen.query(".evals-bench-target-row")

        # Neither mutation ever pressed Save -- both edits are still
        # genuinely unsaved, and no form error was ever triggered. Scoped
        # to the form's own error Static, mirroring the identical check in
        # test_save_persists_every_field_and_reselects_the_bench.
        assert not screen.query_one("#evals-bench-form-error").display


# ---------------------------------------------------------------------------
# task-1611 T2: the "+ New target" mini-form -- ALWAYS rendered (not only
# in the zero-`llama_cpp`-models state), with an optional Name and ONE
# mode-driven steering field, creating an ADDITIONAL, possibly
# differently-steered target.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_target_with_a_prefix_stages_a_row_and_save_persists_the_prefix(
    evals_app_configured, evals_db, bench_with_available_add_target
):
    """The headline T2 flow: the create-target mini-form coexists with the
    Add picker (a target already exists, so `llama_targets()` is
    non-empty -- unlike every Task 6 create-target test above, all of
    which use the zero-models fixture) and mints an ADDITIONAL, steered
    target. Asserts the prefix landed in the DB row's own `config` --
    `_resolve_bench_targets`/`model_steering` reading it back correctly is
    already covered by task-1611 T1's own storage tests; this only needs
    to prove the UI WROTE it.
    """
    task_id, existing_id, _addable_id = bench_with_available_add_target
    async with evals_app_configured.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app_configured.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app_configured.screen

        # Both affordances are present at once -- the whole point of T2.
        assert screen.query_one("#evals-bench-add-target", Select)
        assert screen.query_one("#evals-target-name", Input)
        assert screen.query_one("#evals-target-prefix", Input)
        assert not screen.query("#evals-target-system-prompt")

        screen.query_one("#evals-target-name", Input).value = "steered-raw"
        screen.query_one("#evals-target-prefix", Input).value = "Continue the story: "
        # task-1710: the per-cell continuation checkbox added one row to
        # this form, pushing the mini-form's own Create button right to
        # `#evals-bench-editor`'s own documented scroll threshold at a
        # realistic 160x45 viewport with BOTH the Add picker and the
        # mini-form present (see that id's CSS comment) -- `scroll_
        # visible` first rather than assume it still fits unscrolled;
        # `pilot.click` silently returns `False` (no exception) for a
        # widget whose computed screen position is not actually painted,
        # which is exactly what made this failure mode look like nothing
        # happened rather than an out-of-bounds error.
        create_button = screen.query_one("#evals-bench-create-target", Button)
        create_button.scroll_visible(animate=False)
        await pilot.pause()
        await pilot.click("#evals-bench-create-target")
        await pilot.pause()

        models = {m["name"]: m for m in evals_db.list_models(provider="llama_cpp")}
        assert "steered-raw" in models
        assert models["steered-raw"]["config"] == {"prefix": "Continue the story: "}

        new_row = screen.query_one("#evals-bench-target-1")
        row_text = str(new_row.renderable)
        assert "steered-raw" in row_text
        assert "prefix:" in row_text

        # The mini-form itself was reset after the successful create.
        assert screen.query_one("#evals-target-name", Input).value == ""
        assert screen.query_one("#evals-target-prefix", Input).value == ""

        save_button = screen.query_one("#evals-bench-save", Button)
        save_button.scroll_visible(animate=False)
        await pilot.pause()
        await pilot.click("#evals-bench-save")
        await pilot.pause()
        assert not screen.query_one("#evals-bench-form-error").display
        saved = load_bench(evals_db, task_id)
        assert set(saved.target_ids) == {existing_id, models["steered-raw"]["id"]}


@pytest.mark.asyncio
async def test_create_target_with_leading_whitespace_and_a_newline_in_the_prefix_persists_byte_exact(
    evals_app_configured, evals_db, bench_with_zero_llama_models
):
    """task-1611 T2 fix round 1 (Minor): the E2E test above only ever
    types a TRAILING space -- leading whitespace and an embedded newline
    are the sharper case `CreateTargetRequested`'s own docstring already
    promises ("passed through EXACTLY, no `.strip()`") but nothing UI-
    level had actually driven all the way to a real DB row. Types a
    LEADING ``"\\n"`` plus the rest of a prefix, creates, saves, and
    asserts the persisted ``eval_models`` row's ``config["prefix"]`` is
    byte-identical to what was typed."""
    task_id = bench_with_zero_llama_models  # raw mode
    prefix = "\nContinue: "
    async with evals_app_configured.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app_configured.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app_configured.screen

        screen.query_one("#evals-target-name", Input).value = "leading-ws-target"
        screen.query_one("#evals-target-prefix", Input).value = prefix
        await pilot.click("#evals-bench-create-target")
        await pilot.pause()
        await pilot.click("#evals-bench-save")
        await pilot.pause()
        assert not screen.query_one("#evals-bench-form-error").display

        model = next(
            m
            for m in evals_db.list_models(provider="llama_cpp")
            if m["name"] == "leading-ws-target"
        )
        assert model["config"]["prefix"] == prefix


@pytest.mark.asyncio
async def test_mode_flip_swaps_the_steering_field_and_preserves_typed_state(
    evals_app, bench_with_mixed_readiness
):
    """Flipping the bench's own prompt mode swaps which steering `Input`
    the "+ New target" mini-form shows, via the SAME targeted
    `#evals-bench-targets-section` rebuild Add/Remove already use -- never
    a whole-widget recompose. Proves survival for BOTH the OUTER fields
    (Name/Probes -- the pre-existing Task 5 guarantee) and the mini-form's
    OWN Name Input (new in T2, since it lives inside the section that
    actually gets rebuilt), then flips back and proves the raw-mode
    prefix text specifically survived the round trip, independent of the
    chat-mode text typed in between.
    """
    task_id, _ = bench_with_mixed_readiness  # raw mode
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app.screen

        assert screen.query_one("#evals-target-prefix", Input)
        assert not screen.query("#evals-target-system-prompt")

        screen.query_one("#evals-bench-name", Input).value = "typed-but-unsaved-name"
        screen.query_one("#evals-bench-probes", TextArea).text = "typed probe"
        screen.query_one("#evals-target-name", Input).value = "typed-target-name"
        screen.query_one("#evals-target-prefix", Input).value = "typed-prefix"

        screen.query_one("#evals-bench-prompt-mode", Select).value = "chat"
        await pilot.pause()

        assert not screen.query("#evals-target-prefix")
        assert screen.query_one("#evals-target-system-prompt", Input)

        # Outer fields, entirely outside the rebuilt section, are untouched.
        assert screen.query_one("#evals-bench-name", Input).value == "typed-but-unsaved-name"
        assert screen.query_one("#evals-bench-probes", TextArea).text == "typed probe"
        # The mini-form's own Name Input survives the swap too.
        assert screen.query_one("#evals-target-name", Input).value == "typed-target-name"
        # A fresh chat-mode field starts blank -- the raw-mode prefix text
        # does not leak into an unrelated field.
        assert screen.query_one("#evals-target-system-prompt", Input).value == ""

        screen.query_one("#evals-target-system-prompt", Input).value = "typed-system-prompt"

        # Flip back: the ORIGINAL raw-mode prefix text reappears, carried
        # independently of the chat-mode text just typed.
        screen.query_one("#evals-bench-prompt-mode", Select).value = "raw"
        await pilot.pause()

        assert screen.query_one("#evals-target-prefix", Input).value == "typed-prefix"
        assert screen.query_one("#evals-target-name", Input).value == "typed-target-name"


@pytest.mark.asyncio
async def test_blank_target_name_auto_names_uniquely_across_repeated_creates(
    evals_app_configured, evals_db, bench_with_zero_llama_models
):
    task_id = bench_with_zero_llama_models
    async with evals_app_configured.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app_configured.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app_configured.screen

        # task-1710: the per-cell continuation checkbox added one row to
        # this form; the SECOND create (after the first has already
        # staged a target row, pushing the mini-form's own Create button
        # one row further down) is what actually crosses `#evals-bench-
        # editor`'s own documented scroll threshold at this file's
        # realistic 160x45 viewport -- `scroll_visible` before each click
        # rather than assume either still fits unscrolled.
        create_button = screen.query_one("#evals-bench-create-target", Button)
        create_button.scroll_visible(animate=False)
        await pilot.pause()
        await pilot.click("#evals-bench-create-target")
        await pilot.pause()
        create_button = screen.query_one("#evals-bench-create-target", Button)
        create_button.scroll_visible(animate=False)
        await pilot.pause()
        await pilot.click("#evals-bench-create-target")
        await pilot.pause()

        models = evals_db.list_models(provider="llama_cpp")
        assert len(models) == 2, "two blank-name creates should mint two distinct rows"
        names = {m["name"] for m in models}
        assert len(names) == 2, f"auto-named rows collided: {names!r}"
        assert all(name.startswith(sample_bench.BENCH_EDITOR_TARGET_NAME) for name in names)
        assert screen.query_one("#evals-bench-target-0")
        assert screen.query_one("#evals-bench-target-1")


@pytest.mark.asyncio
async def test_create_target_with_blank_steering_field_stores_no_config_key(
    evals_app_configured, evals_db, bench_with_zero_llama_models
):
    task_id = bench_with_zero_llama_models  # raw mode
    async with evals_app_configured.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app_configured.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app_configured.screen

        screen.query_one("#evals-target-name", Input).value = "no-steering-target"
        # Prefix Input left blank.
        await pilot.click("#evals-bench-create-target")
        await pilot.pause()

        models = evals_db.list_models(provider="llama_cpp")
        assert len(models) == 1
        assert models[0]["name"] == "no-steering-target"
        assert models[0]["config"] == {}


@pytest.mark.asyncio
async def test_create_target_duplicate_name_notifies_and_stages_nothing(
    evals_app_configured, evals_db, bench_with_zero_llama_models
):
    """A NON-blank typed name is used VERBATIM (never uniqued, unlike a
    blank one) -- see `evals_screen.py`'s own handler docstring for why:
    an intentional collision must surface as the `ConflictError` it is.
    `evals_app_configured`'s own app_config sets no `model` key, so
    `configured_llama_cpp_model_id` resolves to `""` -> the `"default"`
    fallback -- matching the pre-seeded row's own `model_id` below so the
    (name, provider, model_id) triple genuinely collides.
    """
    task_id = bench_with_zero_llama_models
    evals_db.create_model(name="dup-target", provider="llama_cpp", model_id="default")

    async with evals_app_configured.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app_configured.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app_configured.screen

        screen.query_one("#evals-target-name", Input).value = "dup-target"
        # task-1710: the pre-seeded "dup-target" row above means
        # `llama_targets()` is non-empty, so the Add picker ALSO renders
        # alongside the mini-form (unlike the genuinely-zero-models
        # fixture other create-target tests use) -- combined with the
        # per-cell continuation checkbox's own added row, this pushes the
        # Create button past `#evals-bench-editor`'s own documented
        # scroll threshold at a realistic 160x45 viewport; `scroll_
        # visible` first rather than assume it still fits unscrolled.
        create_button = screen.query_one("#evals-bench-create-target", Button)
        create_button.scroll_visible(animate=False)
        await pilot.pause()
        await pilot.click("#evals-bench-create-target")
        await pilot.pause()

        assert len(evals_db.list_models(provider="llama_cpp")) == 1, (
            "the conflicting create must not have minted a second row"
        )
        assert not screen.query("#evals-bench-target-0"), "nothing was staged"
        assert evals_app_configured.app_instance.notifications
        message, severity = evals_app_configured.app_instance.notifications[-1]
        assert severity == "error"
        assert "already exists" in message

        # The typed name is left exactly as it was -- no recompose on a
        # failed create, mirroring every other Save-failure path's
        # "nothing typed is silently discarded" contract.
        assert screen.query_one("#evals-target-name", Input).value == "dup-target"


@pytest.mark.asyncio
async def test_steered_target_row_with_a_markup_hazard_prefix_does_not_crash(
    evals_app_configured, evals_db, bench_with_zero_llama_models
):
    task_id = bench_with_zero_llama_models
    async with evals_app_configured.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app_configured.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app_configured.screen

        screen.query_one("#evals-target-name", Input).value = "hazard-target"
        screen.query_one("#evals-target-prefix", Input).value = "loud [/] prefix"
        await pilot.click("#evals-bench-create-target")
        await pilot.pause()

        assert pilot.app.is_running, "an unescaped steering preview crashed the app"
        row = screen.query_one("#evals-bench-target-0")
        row_text = str(row.renderable)
        assert "hazard-target" in row_text
        assert "prefix:" in row_text


@pytest.mark.asyncio
async def test_prompt_mode_switch_with_a_real_steered_target_blocks_save_with_reworded_copy(
    evals_app_configured, evals_db, bench_with_zero_llama_models
):
    """The task-1611 T2 real-row equivalent of
    `test_prompt_mode_switch_revalidates_targets_and_names_the_offending_
    target` above (which still monkeypatches `_resolve_bench_targets` with
    a hand-built `Target`): creates a genuinely raw-only (prefixed)
    target through the UI, saves it onto the bench, then flips to chat
    mode and confirms `_resolve_bench_targets`'s new `model_steering`
    wiring catches it for real AND that the reworded copy (steering is
    immutable -- REMOVE the target, a new one is only an OPTIONAL
    replacement, never a nonexistent "change its settings" affordance,
    and never a claim about which steering the replacement needs --
    whole-branch review, Minor) actually renders.
    """
    task_id = bench_with_zero_llama_models  # raw mode
    async with evals_app_configured.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app_configured.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app_configured.screen

        screen.query_one("#evals-target-name", Input).value = "raw-only-target"
        screen.query_one("#evals-target-prefix", Input).value = "Continue: "
        await pilot.click("#evals-bench-create-target")
        await pilot.pause()
        await pilot.click("#evals-bench-save")
        await pilot.pause()
        assert not screen.query_one("#evals-bench-form-error").display

        screen.query_one("#evals-bench-prompt-mode", Select).value = "chat"
        await pilot.pause()
        await pilot.click("#evals-bench-save")
        await pilot.pause()

        callout = screen.query_one("#evals-bench-form-error")
        assert callout.display
        text = str(callout.renderable)
        assert "raw-only-target" in text
        assert "chat" in text
        assert "cannot be edited" in text
        # "remove it" is the NECESSARY step -- whole-branch review, Minor:
        # an earlier revision offered "create a new target instead" as if
        # a replacement alone unblocked Save, which it does not (the
        # offending target stays staged either way).
        assert "remove it" in text
        assert "optionally replacing it" in text
        # Never over-prescribes which steering a replacement needs -- an
        # unsteered target is equally valid for either mode.
        assert "with a prefix" not in text
        assert "with a system prompt" not in text


@pytest.mark.asyncio
async def test_steering_field_label_matches_the_current_prompt_mode(
    evals_app, bench_with_mixed_readiness
):
    """The steering field's descriptive text lives in its own
    ``placeholder`` (no separate ``Static`` label -- see
    ``_build_create_target_control``'s own docstring for why: this
    section's fixed, small vertical budget at a realistic viewport ruled
    out a persistent label row)."""
    task_id, _ = bench_with_mixed_readiness  # raw mode
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app.screen

        assert screen.query_one("#evals-target-prefix", Input).placeholder == (
            PREFIX_FIELD_LABEL
        )
        assert not screen.query("#evals-target-system-prompt")

        screen.query_one("#evals-bench-prompt-mode", Select).value = "chat"
        await pilot.pause()

        assert screen.query_one("#evals-target-system-prompt", Input).placeholder == (
            SYSTEM_PROMPT_FIELD_LABEL
        )
        assert not screen.query("#evals-target-prefix")


# ---------------------------------------------------------------------------
# task-1610: BenchEditor.is_dirty() -- read by evals_screen.py's
# `_selection_unmoved_since_launch` so a completing run/sample-bench worker
# never recomposes over unsaved form state.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_is_dirty_is_false_immediately_after_a_fresh_selection(
    evals_app, bench_with_mixed_readiness
):
    """A just-composed editor, untouched, reads clean -- the baseline every
    other test below flips away from."""
    task_id, _ = bench_with_mixed_readiness
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        editor = evals_app.screen.query_one(BenchEditor)
        assert editor.is_dirty() is False


@pytest.mark.asyncio
async def test_is_dirty_is_false_with_a_pristine_mini_form(
    evals_app, bench_with_available_add_target
):
    """task-1611 T2 fix round 1: the "+ New target" mini-form itself
    starts blank -- typing NOTHING into it must not be mistaken for an
    edit. Uses a bench with a pre-existing ``llama_cpp`` target (the Add
    picker AND the mini-form both render, task-1611 T2's whole point)
    rather than the zero-models fixture, so this exercises the mini-form
    coexisting with the picker, not merely the degenerate case."""
    task_id, _existing_id, _addable_id = bench_with_available_add_target
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app.screen
        assert screen.query_one("#evals-bench-add-target", Select)
        editor = screen.query_one(BenchEditor)
        assert editor.is_dirty() is False


@pytest.mark.asyncio
async def test_is_dirty_flips_true_on_typed_but_uncreated_mini_form_name(
    evals_app, bench_with_mixed_readiness
):
    """task-1611 T2 fix round 1: a Name typed into the "+ New target"
    mini-form but never submitted (Create never pressed) is real unsaved
    state -- exactly the loss a background worker's completion must not
    silently discard, one level down from the five top-level fields this
    method already protected before this fix."""
    task_id, _ = bench_with_mixed_readiness
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app.screen
        editor = screen.query_one(BenchEditor)
        assert editor.is_dirty() is False

        screen.query_one("#evals-target-name", Input).value = "typed-not-created"
        assert editor.is_dirty() is True


@pytest.mark.asyncio
async def test_is_dirty_flips_true_on_typed_but_uncreated_mini_form_steering(
    evals_app, bench_with_mixed_readiness
):
    """Same as the Name case above, for the steering ``Input`` -- exercised
    via the raw-mode ``#evals-target-prefix`` field (``bench_with_mixed_
    readiness`` is a raw-mode bench)."""
    task_id, _ = bench_with_mixed_readiness
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app.screen
        editor = screen.query_one(BenchEditor)
        assert editor.is_dirty() is False

        screen.query_one("#evals-target-prefix", Input).value = "Continue: "
        assert editor.is_dirty() is True


@pytest.mark.asyncio
async def test_is_dirty_stays_true_for_a_pending_steering_value_after_a_mode_flip(
    evals_app, bench_with_mixed_readiness
):
    """Genuinely exercises `is_dirty()`'s pending-steering FALLBACK branch
    in isolation (whole-branch review, Minor): an earlier version of this
    test flipped to chat and LEFT it flipped before asserting, so
    `prompt_mode != loaded.prompt_mode` alone already made `is_dirty()`
    True regardless of whether the pending-fallback branch worked at all
    -- reviewer proved, in an isolated clone, that stubbing
    `mini_form_prefix`/`mini_form_system_prompt` to `""` left this test
    (and all 115 others) green.

    This flips to chat, types into `#evals-target-system-prompt`, then
    flips BACK to raw -- `bench_with_mixed_readiness` loads in raw mode,
    and the `Select`'s CURRENT value is raw again too by the time
    `is_dirty()` is called, so the prompt-mode branch cannot fire and
    every other loaded field is untouched. Only the pending-steering
    fallback (`#evals-target-system-prompt` is not mounted in raw mode,
    so `is_dirty()` falls back to `self._pending_target_system_prompt`,
    stashed by `_capture_pending_target_form` during the flip back) can
    make this True -- the abandoned chat-mode text is exactly as real and
    exactly as destroyable by a recompose as it was before the flip."""
    task_id, _ = bench_with_mixed_readiness  # raw mode
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app.screen
        editor = screen.query_one(BenchEditor)
        prompt_mode = screen.query_one("#evals-bench-prompt-mode", Select)

        prompt_mode.value = "chat"
        await pilot.pause()
        screen.query_one("#evals-target-system-prompt", Input).value = (
            "typed-system-prompt"
        )

        prompt_mode.value = "raw"
        await pilot.pause()

        # The prompt-mode branch cannot be what makes this dirty: the
        # Select is back to matching `loaded.prompt_mode` exactly.
        assert prompt_mode.value == "raw"
        assert not screen.query("#evals-target-system-prompt")
        assert screen.query_one("#evals-target-prefix", Input)

        assert editor.is_dirty() is True


@pytest.mark.asyncio
async def test_is_dirty_flips_true_independently_for_each_scalar_field(
    evals_app, bench_with_mixed_readiness
):
    """Name/Description/Prompt-mode/Top-K each flip `is_dirty()` to True on
    their own -- exercised one field at a time against a freshly
    re-selected (clean) editor, so an earlier field's edit can never mask a
    later one's assertion. Also covers an UNPARSEABLE Top-K value: Save
    treats that as a real (if invalid) edit -- see `_on_save_pressed`'s own
    `int(...)` parse -- and `is_dirty()` must agree, not silently read it
    as unchanged."""
    task_id, _ = bench_with_mixed_readiness
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        screen = evals_app.screen

        screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen.query_one("#evals-bench-name", Input).value = "renamed"
        assert screen.query_one(BenchEditor).is_dirty() is True

        screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen.query_one("#evals-bench-description", Input).value = "new description"
        assert screen.query_one(BenchEditor).is_dirty() is True

        screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen.query_one("#evals-bench-prompt-mode", Select).value = "chat"
        assert screen.query_one(BenchEditor).is_dirty() is True

        screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen.query_one("#evals-bench-top-k", Input).value = "999"
        assert screen.query_one(BenchEditor).is_dirty() is True

        screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen.query_one("#evals-bench-top-k", Input).value = "not-a-number"
        assert screen.query_one(BenchEditor).is_dirty() is True


@pytest.mark.asyncio
async def test_is_dirty_is_false_when_probes_textarea_exactly_matches_loaded_probes(
    evals_app, bench_with_mixed_readiness
):
    """Setting the TextArea back to a byte-exact reconstruction of the
    loaded probes (`bench_with_mixed_readiness`'s own `(" Sure", " I")`,
    exactly what `compose()` itself renders as `"\\n".join(...)`) reads
    clean -- proves the comparison is content-based, not merely "has the
    widget's `.text` ever been assigned to"."""
    task_id, _ = bench_with_mixed_readiness
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app.screen
        editor = screen.query_one(BenchEditor)

        screen.query_one("#evals-bench-probes", TextArea).text = " Sure\n I"
        assert editor.is_dirty() is False


@pytest.mark.asyncio
async def test_is_dirty_flips_true_on_a_whitespace_only_probe_line_addition(
    evals_app, bench_with_mixed_readiness
):
    """A trailing whitespace-only line (a lone `" "`) is a real, distinct
    probe -- not a zero-length line `_parse_probes_text` would drop -- so
    adding one is a genuine edit `is_dirty()` must catch."""
    task_id, _ = bench_with_mixed_readiness
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app.screen
        editor = screen.query_one(BenchEditor)

        screen.query_one("#evals-bench-probes", TextArea).text = " Sure\n I\n "
        assert editor.is_dirty() is True


@pytest.mark.asyncio
async def test_is_dirty_is_false_when_a_trailing_empty_line_is_added(
    evals_app, bench_with_mixed_readiness
):
    """The zero-length-line filter `_parse_probes_text` shares with Save:
    a bare trailing Enter-press (a genuine zero-length line) parses to the
    SAME probe tuple as the loaded config, so it must not read as dirty --
    mirrors `test_trailing_newline_after_the_last_probe_does_not_persist_
    an_empty_probe`'s save-path assertion for the dirty check."""
    task_id, _ = bench_with_mixed_readiness
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app.screen
        editor = screen.query_one(BenchEditor)

        screen.query_one("#evals-bench-probes", TextArea).text = " Sure\n I\n"
        assert editor.is_dirty() is False


@pytest.mark.asyncio
async def test_is_dirty_flips_true_on_a_staged_target_add(
    evals_app, bench_with_available_add_target
):
    task_id, _existing_id, addable_id = bench_with_available_add_target
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app.screen
        editor = screen.query_one(BenchEditor)
        assert editor.is_dirty() is False

        select = screen.query_one("#evals-bench-add-target", Select)
        select.value = addable_id
        await pilot.click("#evals-bench-add-target-button")
        await pilot.pause()

        assert editor.is_dirty() is True


@pytest.mark.asyncio
async def test_is_dirty_flips_true_on_a_staged_target_remove(
    evals_app, bench_with_mixed_readiness
):
    task_id, _ = bench_with_mixed_readiness
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app.screen
        editor = screen.query_one(BenchEditor)
        assert editor.is_dirty() is False

        await pilot.click("#evals-bench-target-remove-0")
        await pilot.pause()

        assert editor.is_dirty() is True


@pytest.mark.asyncio
async def test_is_dirty_is_false_again_after_save_and_reload(
    evals_app, evals_db, bench_with_mixed_readiness
):
    """Save -> `Saved` -> the screen's own `select()` recomposes from the
    freshly persisted row: the new `BenchEditor` instance (a real rebuild,
    not the same one) reads clean again -- the round-trip this whole
    feature exists to protect: an in-flight worker completing AFTER a Save
    must still be free to auto-navigate."""
    task_id, _ = bench_with_mixed_readiness
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app.screen
        screen.query_one("#evals-bench-name", Input).value = "renamed-and-saved"

        assert screen.query_one(BenchEditor).is_dirty() is True

        await pilot.click("#evals-bench-save")
        await pilot.pause()

        editor = screen.query_one(BenchEditor)
        assert editor.is_dirty() is False


@pytest.mark.asyncio
async def test_save_discards_typed_but_uncreated_mini_form_text(
    evals_app, bench_with_mixed_readiness
):
    """Whole-branch review, Minor (judged, documented, NOT fixed -- see
    the module docstring's own paragraph, right after its Task-1610 one,
    for the full reasoning): a successful Save is the one place `is_
    dirty()`'s own "this text is worth protecting" premise does not hold
    -- `Saved` triggers a genuine recompose (`evals_screen.py`'s
    `select()`), which builds a brand-new `BenchEditor` whose `self.
    _pending_target_*` starts blank again, with no path from the old
    instance's typed-but-never-created mini-form text to it. Pins the
    CURRENT, deliberately-accepted behavior so a future change to it (in
    either direction -- fixing it, or a regression making it WORSE, e.g.
    losing the mini-form's state even on a Save FAILURE) is a conscious
    decision, not an accident nobody noticed."""
    task_id, _ = bench_with_mixed_readiness  # raw mode
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app.screen

        screen.query_one("#evals-target-name", Input).value = "typed-not-created"
        screen.query_one("#evals-target-prefix", Input).value = "typed-prefix"
        assert screen.query_one(BenchEditor).is_dirty() is True

        await pilot.click("#evals-bench-save")
        await pilot.pause()

        editor = screen.query_one(BenchEditor)
        assert screen.query_one("#evals-target-name", Input).value == ""
        assert screen.query_one("#evals-target-prefix", Input).value == ""
        assert editor.is_dirty() is False


@pytest.mark.asyncio
async def test_is_dirty_is_false_when_the_widget_never_composed_a_form(evals_app):
    """The two `compose()` early-return branches (no db, unreadable row)
    leave `_loaded_config` at `None` and never yield the Save button that
    would need `_on_save_pressed`'s widgets to exist -- `is_dirty()` must
    not raise `QueryError` reaching for widgets that were never composed,
    and must report clean (there is no form to have edited)."""
    editor = BenchEditor(
        EvalsViewModel(None), "does-not-exist", id="evals-bench-editor-standalone"
    )
    assert editor.is_dirty() is False


# ---------------------------------------------------------------------------
# task-1710 T2: the per-cell continuation opt-in checkbox itself -- reflects
# the loaded config, saves, flips `is_dirty()`, and survives a targeted
# targets-section rebuild. `bench_with_capture_continuations_on_raw` (above,
# in the Estimate section) already covers the checkbox=True load case, reused
# here rather than duplicated.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_capture_continuations_checkbox_reflects_the_loaded_config(
    evals_app, bench_with_mixed_readiness, bench_with_capture_continuations_on_raw
):
    """The checkbox's initial `value` (set in `compose()` from `config.
    capture_continuations`) must match whatever was actually persisted --
    both directions, off (`bench_with_mixed_readiness`, never set, so
    defaults `False`) and on (`bench_with_capture_continuations_on_raw`)."""
    task_id_off, _ = bench_with_mixed_readiness
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        screen = evals_app.screen

        screen.select(kind="bench", id=task_id_off)
        await pilot.pause()
        assert (
            screen.query_one("#evals-bench-capture-continuations", Checkbox).value
            is False
        )

        screen.select(kind="bench", id=bench_with_capture_continuations_on_raw)
        await pilot.pause()
        assert (
            screen.query_one("#evals-bench-capture-continuations", Checkbox).value
            is True
        )


def test_capture_continuations_label_states_the_per_cell_request_cost():
    """task-1710's own instruction: "labelled honestly ... the label/
    tooltip should say that plainly (e.g. that it adds one request per
    cell)." A pure string assertion against the pinned constant, no app
    needed -- mirrors this file's other verbatim-label pins (e.g.
    `PREFIX_FIELD_LABEL`)."""
    assert "request" in CAPTURE_CONTINUATIONS_LABEL
    assert "cell" in CAPTURE_CONTINUATIONS_LABEL


@pytest.mark.asyncio
async def test_save_without_touching_the_checkbox_preserves_capture_continuations_flag(
    evals_app, evals_db, bench_with_capture_continuations_on_raw
):
    """THE regression task-1710 T1's own report flagged by name:
    `_on_save_pressed` used to thread `concurrency=loaded.concurrency`
    through its `BenchConfig(...)` construction with no equivalent for
    `capture_continuations` at all, so saving ANY existing bench through
    this editor silently reset the flag back to its dataclass default
    (`False`) -- destroying a run's own recorded cost/content commitment
    on an UNRELATED edit that never touched this checkbox. Saves a
    `capture_continuations=True` bench after editing an unrelated field
    (Description) and WITHOUT touching the checkbox at all; the flag must
    still read `True` afterward."""
    task_id = bench_with_capture_continuations_on_raw
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app.screen

        assert (
            screen.query_one("#evals-bench-capture-continuations", Checkbox).value
            is True
        )
        screen.query_one("#evals-bench-description", Input).value = "edited elsewhere"

        await pilot.click("#evals-bench-save")
        await pilot.pause()

        assert not screen.query_one("#evals-bench-form-error").display
        saved = load_bench(evals_db, task_id)
        assert saved.capture_continuations is True
        assert saved.description == "edited elsewhere"


@pytest.mark.asyncio
async def test_save_with_capture_continuations_toggled_on_via_the_ui_persists_the_flip(
    evals_app, evals_db, bench_with_mixed_readiness
):
    """The opt-IN itself, end to end: a bench loaded with the flag off
    (`bench_with_mixed_readiness`'s default), the checkbox flipped on via
    the UI, Save -- the flag must persist as `True`, proving the checkbox
    is genuinely wired to `BenchConfig(...)`'s own construction, not just
    read back its own unchanged initial value (which the preservation
    test above alone could not rule out)."""
    task_id, _ = bench_with_mixed_readiness
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app.screen

        checkbox = screen.query_one("#evals-bench-capture-continuations", Checkbox)
        assert checkbox.value is False
        checkbox.value = True

        await pilot.click("#evals-bench-save")
        await pilot.pause()

        assert not screen.query_one("#evals-bench-form-error").display
        saved = load_bench(evals_db, task_id)
        assert saved.capture_continuations is True


@pytest.mark.asyncio
async def test_toggling_capture_continuations_checkbox_flips_is_dirty(
    evals_app, bench_with_mixed_readiness
):
    """Part of the form's dirty contract like every other field (task-1610)
    -- a background run/sample-bench worker completing while this is
    toggled-but-unsaved must degrade to a toast, not silently recompose
    and discard the flip (`evals_screen.py`'s `_selection_unmoved_since_
    launch`, which queries `is_dirty()`). Toggling on flips it true;
    toggling back to the loaded value (off) reads clean again -- proves
    the comparison is against `loaded.capture_continuations`, not merely
    "has this checkbox ever been touched"."""
    task_id, _ = bench_with_mixed_readiness
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app.screen
        editor = screen.query_one(BenchEditor)

        checkbox = screen.query_one("#evals-bench-capture-continuations", Checkbox)
        assert editor.is_dirty() is False

        checkbox.value = True
        assert editor.is_dirty() is True

        checkbox.value = False
        assert editor.is_dirty() is False


@pytest.mark.asyncio
async def test_capture_continuations_checkbox_survives_a_targeted_targets_section_rebuild(
    evals_app, bench_with_available_add_target
):
    """The checkbox lives OUTSIDE `#evals-bench-targets-section` (yielded
    earlier in `compose()`, right after the probes field) -- an Add-target
    press only ever tears down and rebuilds that one child
    (`_refresh_targets_section`'s own docstring), so a toggled-but-unsaved
    checkbox must survive it exactly like Name/Description/Top-K/Probes
    already do, not just happen to survive because no test ever checked."""
    task_id, existing_id, addable_id = bench_with_available_add_target
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app.screen

        checkbox = screen.query_one("#evals-bench-capture-continuations", Checkbox)
        checkbox.value = True

        select = screen.query_one("#evals-bench-add-target", Select)
        select.value = addable_id
        await pilot.click("#evals-bench-add-target-button")
        await pilot.pause()

        # The rebuild genuinely happened (a new row exists) -- otherwise
        # this test would trivially pass by never exercising the rebuild
        # at all.
        assert screen.query_one("#evals-bench-target-1")
        assert (
            screen.query_one("#evals-bench-capture-continuations", Checkbox).value
            is True
        )
        assert screen.query_one(BenchEditor).is_dirty() is True


@pytest.mark.asyncio
async def test_capture_continuations_checkbox_survives_a_prompt_mode_flip_rebuild(
    evals_app, bench_with_mixed_readiness
):
    """Review Minor: sibling of `test_capture_continuations_checkbox_
    survives_a_targeted_targets_section_rebuild` above, but for the OTHER
    trigger of the SAME targeted `_refresh_targets_section` rebuild -- a
    prompt-mode flip (`_on_prompt_mode_changed`). The checkbox lives
    outside `#evals-bench-targets-section` regardless of which handler
    triggers the rebuild, so this is mechanistically the identical
    guarantee the Add-target test above pins -- but nothing pinned THIS
    specific trigger directly until now. Also re-confirms, in the SAME
    flip, that a typed Name/Probes edit survives alongside the checkbox
    (already covered on its own by `test_mode_flip_swaps_the_steering_
    field_and_preserves_typed_state`; repeated here so this test stands
    alone rather than relying on a sibling test for that half of the
    claim)."""
    task_id, _ = bench_with_mixed_readiness  # raw mode
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app.screen

        checkbox = screen.query_one("#evals-bench-capture-continuations", Checkbox)
        checkbox.value = True
        screen.query_one("#evals-bench-name", Input).value = "typed-but-unsaved-name"
        screen.query_one("#evals-bench-probes", TextArea).text = "typed probe"

        screen.query_one("#evals-bench-prompt-mode", Select).value = "chat"
        await pilot.pause()

        # The rebuild genuinely happened (the steering field swapped) --
        # otherwise this test would trivially pass by never exercising
        # the rebuild at all.
        assert not screen.query("#evals-target-prefix")
        assert screen.query_one("#evals-target-system-prompt", Input)

        assert (
            screen.query_one("#evals-bench-capture-continuations", Checkbox).value
            is True
        )
        assert (
            screen.query_one("#evals-bench-name", Input).value
            == "typed-but-unsaved-name"
        )
        assert screen.query_one("#evals-bench-probes", TextArea).text == "typed probe"
        assert screen.query_one(BenchEditor).is_dirty() is True

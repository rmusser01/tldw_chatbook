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
from textual.app import App

import tldw_chatbook
from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.Evals.word_bench.models import BenchConfig, PreflightResult, Snippet, Target
from tldw_chatbook.Evals.word_bench.storage import BENCH_TYPE, create_run_group, save_bench
from tldw_chatbook.UI.Evals import bench_editor as bench_editor_module
from tldw_chatbook.UI.Evals import inspector as inspector_module
from tldw_chatbook.UI.Evals.bench_editor import CLASSIC_TASK_DEFERRAL_SENTENCE
from tldw_chatbook.UI.Evals.evals_state import EvalsViewModel
from tldw_chatbook.UI.Screens.evals_screen import EvalsScreen

_BUNDLED_CSS_PATH = str(
    Path(tldw_chatbook.__file__).parent / "css" / "tldw_cli_modular.tcss"
)


class _FakeOrchestrator:
    def __init__(self, db: EvalsDB) -> None:
        self.db = db


class _FakeAppInstance:
    def __init__(self, db: EvalsDB) -> None:
        self.evaluation_orchestrator = _FakeOrchestrator(db)
        self.notifications: list[tuple[str, str]] = []

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


def _target_status_text(screen, index: int) -> str:
    """Looks up an inspector target row by INDEX, not target id -- widget
    ids in ``inspector.py`` are index-derived (see its fix for the same
    duplicate-id-collision principle ``snippet_editor.py``'s rows follow),
    so a test must address a row the same way the widget itself does."""
    widget = screen.query_one(f"#evals-inspector-target-{index}")
    text = widget.renderable
    return text.plain if hasattr(text, "plain") else str(text)


# ---------------------------------------------------------------------------
# Detail pane: bench metadata + target table
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bench_detail_pane_shows_metadata_and_target_table(
    evals_app, bench_with_mixed_readiness
):
    task_id, target_ids = bench_with_mixed_readiness
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=task_id)
        await pilot.pause()
        screen = evals_app.screen

        name = screen.query_one("#evals-detail-bench-name")
        assert "loaded-nouns v1" in str(name.renderable)

        dataset_line = screen.query_one("#evals-detail-bench-dataset")
        assert "loaded-nouns" in str(dataset_line.renderable)

        mode_line = screen.query_one("#evals-detail-bench-prompt-mode")
        assert "raw" in str(mode_line.renderable)

        top_k_line = screen.query_one("#evals-detail-bench-top-k")
        assert "20" in str(top_k_line.renderable)

        probes_line = screen.query_one("#evals-detail-bench-probes")
        assert "Sure" in str(probes_line.renderable)

        # Region, not just query_one success -- a widget can be present in
        # the DOM and occupy zero space (see evals_screen.py's own module
        # docstring on the hub's original defect).
        for metadata_widget in (name, dataset_line, mode_line, top_k_line, probes_line):
            assert metadata_widget.region.width > 0
            assert metadata_widget.region.height > 0

        for index in range(len(target_ids)):
            row = screen.query_one(f"#evals-bench-target-{index}")
            assert row.region.width > 0
            assert row.region.height > 0


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
# Un-preflighted state: a bench that has never run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_never_run_bench_renders_unpreflighted_state(evals_app, never_run_bench):
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="bench", id=never_run_bench)
        await pilot.pause()
        screen = evals_app.screen

        targets = screen.query(".evals-status-unchecked")
        assert list(targets), "expected an un-preflighted status row"
        text = str(targets[0].renderable)
        # Positive assertion first -- the three negative checks below would
        # all pass for an empty (or any unrelated) label just as readily as
        # for the correct one; "Not yet checked" is the actual rendered
        # text (see inspector.py's `_target_status_text`/`status_text`
        # fallback for a `None` preflight result).
        assert "Not yet checked" in text, text
        assert "Ready" not in text
        assert "Blocked" not in text
        assert "Unavailable" not in text

        # No target-readiness recovery callout is warranted for "we haven't
        # checked yet". The separate primary-action callout remains visible
        # because Run is deliberately disabled until execution is wired.
        assert not screen.query("#evals-inspector-target-callout-0")
        assert screen.query_one("#evals-primary-action-reason")


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

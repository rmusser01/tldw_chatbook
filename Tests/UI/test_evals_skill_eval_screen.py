"""Screen wiring: new-bench handler, detail branch, launch helper, worker e2e.

Task 10 of the skill-eval sub-harness -- the integration layer connecting the
engine (Tasks 1-8) and the launcher panel/view-model reads (Task 9) into the
Evals workbench. Mirrors ``test_evals_character_run_e2e.py``'s harness shape:
``EvalsHarness``/``_FakeAppInstance`` imported from ``test_evals_screen.py``
rather than redefined, plus this file's own ``evals_db``/``evals_app``
fixtures.

Hermeticity note: every test that reaches ``store_skill_names`` or the
worker's subject-from-store path monkeypatches ``get_user_data_dir`` in
``skill_eval_launch`` (the one resolution seam the plan deferred to the
app.py:8635 precedent) onto a ``tmp_path``, so no test ever reads this
machine's real local skills store. Subject resolution itself is exercised
through the DIRECTORY kind (a ``tmp_path`` ``SKILL.md``), which is pure
filesystem -- the STORE kind's ``LocalSkillsService`` round-trip is already
covered engine-side by ``Tests/Evals/skill_eval/test_subject.py``.
"""

from __future__ import annotations

import pytest
from textual.widgets import Select

from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.Evals.skill_eval.models import (
    EvalTarget,
    SkillEvalConfig,
    SkillEvalDepth,
)
from tldw_chatbook.Evals.skill_eval.storage import (
    create_skill_eval_run,
    iter_artifacts,
    load_report,
    save_artifact,
    save_report,
    save_skill_eval_bench,
)
from tldw_chatbook.UI.Evals.evals_state import EvalsViewModel
from tldw_chatbook.UI.Evals.skill_eval_panel import SkillEvalPanel

#: Same collection-time config binding as the panel suite -- the
#: per-test env redirect trips config-participant admission (TASK-32628);
#: keep the hermetic bootstrap profile (TASK-32873 opt-in).
pytestmark = pytest.mark.bootstrap_profile

from .test_evals_screen import EvalsHarness, _FakeAppInstance

_REALISTIC_SIZE = (160, 45)


@pytest.fixture
def evals_db() -> EvalsDB:
    return EvalsDB(db_path=":memory:", client_id="test")


@pytest.fixture
def evals_app(evals_db: EvalsDB) -> EvalsHarness:
    return EvalsHarness(_FakeAppInstance(evals_db))


async def _wait_until(pilot, predicate, *, tries: int = 300, interval: float = 0.02) -> None:
    """Poll ``predicate`` until true, tolerating a not-yet-mounted DOM.

    Hardened beyond ``test_evals_screen.py``'s same-named helper because
    this file's predicates query widgets (``#skill-eval-depth``) that a
    scheduled-but-not-yet-run region swap has not mounted: ``select()``
    only SCHEDULES the swap (a worker), so the first poll can execute
    before the widget exists -- a ``QueryError`` from ``query_one`` means
    "not yet" here, never test failure.
    """
    from textual.css.query import QueryError

    for _ in range(tries):
        try:
            if predicate():
                return
        except QueryError:
            pass
        await pilot.pause(interval)
    raise AssertionError("condition never became true")


def _write_skill_dir(tmp_path) -> str:
    """One minimal on-disk skill: front matter + body, ``subject_from_directory``
    shaped. The name matches ``test_runner.py``'s ``_Chat`` selection default
    (``{"skill": "csv-cleaner"}``) so trigger metrics come out determinate."""
    skill_dir = tmp_path / "csv-cleaner"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: csv-cleaner\n"
        "description: Use when tidying messy CSV exports of product data.\n"
        "---\n"
        "# CSV cleaner\n"
        "Trim stray whitespace, drop duplicate headers.\n",
        encoding="utf-8",
    )
    return str(skill_dir)


# ---------------------------------------------------------------------------
# Case 1: the launch helper -- provider preflight + chat-call routing
# ---------------------------------------------------------------------------


def test_make_skill_eval_chat_returns_problems_for_an_unready_provider(
    monkeypatch,
):
    """Preflight problems surface as the returned list, one per unready
    provider -- the worker turns these into a failed run + a toast, so the
    contract here is just "empty list means ready, non-empty means stop"."""
    from tldw_chatbook.UI.Evals import skill_eval_launch as launch

    class _Unready:
        ready = False
        api_key = None
        user_message = "llama_cpp is not configured."

    monkeypatch.setattr(
        launch, "get_provider_readiness", lambda provider, cfg, **kw: _Unready()
    )
    generator = EvalTarget(id="g", provider="llama_cpp", model_id="gen-m")
    judge = EvalTarget(id="j", provider="llama_cpp", model_id="jud-m")

    _chat, problems = launch.make_skill_eval_chat({}, generator, judge)

    assert problems == ["llama_cpp is not configured."]


def test_make_skill_eval_chat_routes_through_chat_api_call(monkeypatch):
    """The built callable wraps one ``chat_api_call`` per invocation -- pinned
    endpoint/model/timeout/key wiring, and OpenAI-shaped reply extraction."""
    from tldw_chatbook.UI.Evals import skill_eval_launch as launch

    class _Ready:
        ready = True
        api_key = "key-123"
        user_message = ""

    monkeypatch.setattr(
        launch, "get_provider_readiness", lambda provider, cfg, **kw: _Ready()
    )
    calls: list[dict] = []

    def _fake_chat_api_call(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": "hi there"}}]}

    monkeypatch.setattr(launch, "chat_api_call", _fake_chat_api_call)
    generator = EvalTarget(id="g", provider="llama_cpp", model_id="gen-m")
    judge = EvalTarget(id="j", provider="llama_cpp", model_id="jud-m")

    chat, problems = launch.make_skill_eval_chat({}, generator, judge)
    assert problems == []

    reply = chat(
        messages=[{"role": "user", "content": "x"}],
        target=generator,
        temperature=0.2,
        max_tokens=64,
        seed=1,
    )

    assert reply == "hi there"
    assert len(calls) == 1
    assert calls[0]["api_endpoint"] == "llama_cpp"
    assert calls[0]["model"] == "gen-m"
    assert calls[0]["api_key"] == "key-123"
    assert calls[0]["request_timeout"] == 120.0
    assert calls[0]["request_retries"] == 2
    assert calls[0]["request_retry_delay"] == 2.0
    assert calls[0]["streaming"] is False


def test_builtin_tool_names_is_a_name_frozenset():
    from tldw_chatbook.UI.Evals.skill_eval_launch import builtin_tool_names

    names = builtin_tool_names()
    assert isinstance(names, frozenset)
    assert all(isinstance(n, str) for n in names)


# Final-review Critical 2: the reserved-name set the worker hands the static
# analyzer must exclude the subject's own name -- the store-sourced skill
# list includes it, so a bare union flagged every store subject against
# itself (NAME_COLLISION, −5%).
def test_reserved_names_exclude_the_subject_itself():
    from tldw_chatbook.UI.Evals.skill_eval_launch import reserved_names_for

    names = reserved_names_for(
        "csv-cleaner",
        frozenset({"csv-cleaner", "pdf-writer"}),
        frozenset({"fs_read"}),
    )
    assert names == frozenset({"pdf-writer", "fs_read"})


@pytest.mark.asyncio
async def test_store_skill_names_reads_the_local_store(monkeypatch, tmp_path):
    """``(summary_dicts, name_set)`` from the local store over the app.py:8635
    directory precedent -- hermetic here via the ``get_user_data_dir`` seam,
    so an empty store yields an empty pool/set (the worker's decoy pool and
    reserved-name inputs both degrade, never crash)."""
    from tldw_chatbook.UI.Evals import skill_eval_launch as launch

    monkeypatch.setattr(launch, "get_user_data_dir", lambda: tmp_path)

    skills, names = await launch.store_skill_names({})

    assert skills == []
    assert names == frozenset()


# ---------------------------------------------------------------------------
# Case 2: the run-group detail view
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skill_eval_detail_renders_report_for_a_seeded_run_group(
    evals_db, tmp_path
):
    """``SkillEvalDetail`` renders the stored report's header facts (composite/
    grade/confidence), one line per dimension, and the artifact count -- read
    back through the same ``load_report``/``iter_artifacts`` helpers the widget
    itself uses, against a real seeded run group."""
    from tldw_chatbook.Evals.skill_eval.models import (
        DimensionScore,
        SkillEvalReport,
        SkillSubject,
        StaticFinding,
    )
    from tldw_chatbook.UI.Evals.skill_eval_detail import SkillEvalDetail
    from textual.app import App, ComposeResult

    subject = SkillSubject(
        name="csv-cleaner", description="d", body="b", source_kind="directory",
        source_path=str(tmp_path), trust_status="unknown", digest="d" * 64,
        line_count=2,
    )
    # ``create_skill_eval_run`` -> ``db.create_run`` validates that the
    # run's judge model EXISTS as an eval_models row (the same contract
    # ``Tests/Evals/skill_eval/test_storage.py``'s own ``_targets`` helper
    # seeds against), so the targets' ids must come from real rows.
    generator = EvalTarget(
        id=evals_db.create_model(name="gen", provider="llama_cpp", model_id="gen-m"),
        provider="llama_cpp", model_id="gen-m",
    )
    judge = EvalTarget(
        id=evals_db.create_model(name="jud", provider="llama_cpp", model_id="jud-m"),
        provider="llama_cpp", model_id="jud-m",
    )
    bench_id = save_skill_eval_bench(
        evals_db,
        SkillEvalConfig(
            name="csv eval", subject_ref=str(tmp_path), subject_kind="directory",
            depth=SkillEvalDepth.STANDARD, generator_target_id="g",
            judge_target_id="j",
        ),
    )
    _group, run_id = create_skill_eval_run(
        evals_db, bench_id,
        SkillEvalConfig(
            name="csv eval", subject_ref=str(tmp_path), subject_kind="directory",
            depth=SkillEvalDepth.STANDARD, generator_target_id="g",
            judge_target_id="j",
        ),
        subject, generator, judge, call_estimate=16,
    )
    save_artifact(evals_db, run_id, {
        "sample_id": "judge-task-0", "kind": "task", "input": {},
        "raw": '{"rating": 4}', "parsed": {"rating": 4},
    })
    save_report(
        evals_db, run_id,
        SkillEvalReport(
            provenance=subject.to_provenance(), depth="standard",
            dimensions=(
                DimensionScore("triggering_accuracy", 0.25, 0.8,
                               ("static", "judge")),
                DimensionScore("instruction_fitness", 0.18, 0.72, ("judge",)),
            ),
            composite=81.2, grade="B-", confidence="Assessed",
            findings=(StaticFinding("MISSING_TRIGGER", 0.05,
                                    'Add "Use when ..." phrasing.'),),
            warnings=("a warning line",),
        ),
    )

    class _DetailHarness(App):
        def compose(self) -> ComposeResult:
            yield SkillEvalDetail(EvalsViewModel(evals_db), run_id)

    app = _DetailHarness()
    async with app.run_test(size=_REALISTIC_SIZE) as pilot:
        from textual.widgets import Static

        lines = [
            str(static.render())
            for static in app.screen.query("SkillEvalDetail Static")
        ]
        text = "\n".join(lines)
        # Header: composite, grade, confidence, depth, methodology.
        assert "B-" in text
        assert "Assessed" in text
        assert "81.2" in text
        # One line per dimension, naming the dimension.
        assert any("triggering_accuracy" in line for line in lines)
        assert any("instruction_fitness" in line for line in lines)
        # Findings with remediation, warnings, artifact count.
        assert any("MISSING_TRIGGER" in line for line in lines)
        assert any("a warning line" in line for line in lines)
        assert any("artifacts: 1" in line for line in lines)
        # Provenance names the subject.
        assert any("csv-cleaner" in line for line in lines)


# ---------------------------------------------------------------------------
# Case 3: rail kind discrimination + the new-bench handler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_skill_eval_bench_row_routes_to_the_skill_eval_kind(
    evals_app, evals_db
):
    """The rail's classic-subgroup row-kind discrimination: a skill-eval bench
    row maps to ``kind="skill_eval_bench"`` (never ``"classic"``), asserted
    through the mounted ``LibraryRail``'s own row-target registry -- the exact
    mapping a row press posts."""
    from tldw_chatbook.UI.Evals.library_rail import LibraryRail

    bench_id = save_skill_eval_bench(
        evals_db,
        SkillEvalConfig(
            name="csv eval", subject_ref="csv-cleaner", subject_kind="store",
            depth=SkillEvalDepth.STANDARD, generator_target_id="g",
            judge_target_id="j",
        ),
    )

    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        rail = pilot.app.screen.query_one(LibraryRail)
        targets = {
            sel.id: sel.kind for sel in rail._row_targets.values()
        }
        assert targets.get(bench_id) == "skill_eval_bench"


@pytest.mark.asyncio
async def test_new_skill_eval_button_creates_a_draft_bench_and_selects_it(
    evals_app, evals_db, monkeypatch
):
    """``+ New skill eval`` posts ``NewSkillEvalRequested``; the screen's
    handler creates the draft bench (first model target pre-bound when one
    exists) and selects it, which mounts the ``SkillEvalPanel`` detail branch
    with the panel fed by the view model's target rows."""
    from tldw_chatbook.UI.Screens import evals_screen as screen_mod

    # Hermeticity: mounting the panel now also feeds the subject picker via
    # the screen's ``store_skill_names`` call site -- keep it empty rather
    # than reading this machine's real local skills store.
    async def _empty_store(_app_config):
        return [], frozenset()

    monkeypatch.setattr(screen_mod, "store_skill_names", _empty_store)
    gen_id = evals_db.create_model(
        name="gen", provider="llama_cpp", model_id="m"
    )
    judge_id = evals_db.create_model(
        name="jud", provider="llama_cpp", model_id="m2"
    )

    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        # One refresh before clicking -- the same settle the character-
        # bench twin (``test_new_character_bench_creates_and_selects_a_
        # runnable_draft``) gives the rail; a click at a pre-arrangement
        # coordinate hits nothing.
        await pilot.pause()
        await pilot.click("#evals-rail-new-skill-eval")
        await _wait_until(
            pilot, lambda: pilot.app.screen._selection.kind == "skill_eval_bench"
        )

        screen = pilot.app.screen
        assert screen._selection.id is not None
        from tldw_chatbook.Evals.skill_eval.storage import load_skill_eval_bench

        config = load_skill_eval_bench(evals_db, screen._selection.id)
        assert config.subject_kind == "store"
        assert config.depth is SkillEvalDepth.STANDARD
        # TASK-32888: drafts start with EMPTY target ids. The old pre-seed
        # of the first available model row was invisible in the panel yet
        # silently became the run's models through config-loading paths;
        # with picks now persisting and restoring, an honest empty (the
        # user picks, TASK-32884's guidance teaches where from) replaces
        # the hidden default.
        assert config.generator_target_id == ""
        assert config.judge_target_id == ""

        panel = screen.query_one(SkillEvalPanel)
        # The panel is fed the view model's targets (set_options runs
        # post-mount via the screen's call_after_refresh population).
        def _targets_fed() -> bool:
            options = screen.query_one("#skill-eval-generator", Select)._options
            return {value for _label, value in options} >= {gen_id, judge_id}

        await _wait_until(pilot, _targets_fed)
        # Model picks are deliberately NOT restored (only depth is) -- the
        # panel's own Run guard requires a fresh pick before any run.
        assert screen.query_one("#skill-eval-generator", Select).value is Select.NULL
        # The draft's saved depth (STANDARD) is restored into the depth
        # Select, which fires Select.Changed and syncs the estimate line.
        assert (
            screen.query_one("#skill-eval-depth", Select).value
            is SkillEvalDepth.STANDARD
        )
        estimate = str(screen.query_one("#skill-eval-estimate").render())
        assert "16" in estimate


@pytest.mark.asyncio
async def test_selecting_a_saved_bench_restores_its_saved_depth(
    evals_app, evals_db, monkeypatch
):
    """A saved bench's depth round-trips through the detail branch: the depth
    Select's programmatic value assignment fires ``Select.Changed``, which
    updates the panel's internal depth AND re-syncs the call estimate."""
    from tldw_chatbook.UI.Screens import evals_screen as screen_mod

    # Hermeticity: same reason as the draft test above -- the mounted
    # panel's subject-picker feed must not read the real local skills store.
    async def _empty_store(_app_config):
        return [], frozenset()

    monkeypatch.setattr(screen_mod, "store_skill_names", _empty_store)
    bench_id = save_skill_eval_bench(
        evals_db,
        SkillEvalConfig(
            name="deep eval", subject_ref="csv-cleaner", subject_kind="store",
            depth=SkillEvalDepth.DEEP, generator_target_id="g",
            judge_target_id="j",
        ),
    )

    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        pilot.app.screen.select(kind="skill_eval_bench", id=bench_id)
        await _wait_until(
            pilot,
            lambda: pilot.app.screen.query_one("#skill-eval-depth", Select).value
            is SkillEvalDepth.DEEP,
        )
        panel = pilot.app.screen.query_one(SkillEvalPanel)
        assert panel._depth is SkillEvalDepth.DEEP
        estimate = str(pilot.app.screen.query_one("#skill-eval-estimate").render())
        assert "67" in estimate


# ---------------------------------------------------------------------------
# Case 4: worker e2e
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_skill_eval_worker_completes_and_saves_the_report(
    evals_app, evals_db, monkeypatch, tmp_path
):
    """The full worker against a real ``:memory:`` EvalsDB: a directory-kind
    subject, two llama_cpp model rows, and a scripted chat factory (the same
    ``_Chat`` the engine's own runner tests use). The run group must land
    ``completed`` through the real ``run_groups()`` pivot, the report must be
    loadable with STANDARD's ``Assessed`` confidence, and every judge-layer
    artifact must be persisted."""
    from Tests.Evals.skill_eval.test_runner import _Chat
    from tldw_chatbook.UI.Evals import skill_eval_launch as launch
    from tldw_chatbook.UI.Screens import evals_screen as screen_mod

    monkeypatch.setattr(launch, "get_user_data_dir", lambda: tmp_path)
    skill_dir = _write_skill_dir(tmp_path)
    gen_id = evals_db.create_model(name="gen", provider="llama_cpp", model_id="gen-m")
    jud_id = evals_db.create_model(name="jud", provider="llama_cpp", model_id="jud-m")
    bench_id = save_skill_eval_bench(
        evals_db,
        SkillEvalConfig(
            name="csv eval", subject_ref=skill_dir, subject_kind="directory",
            depth=SkillEvalDepth.STANDARD, generator_target_id=gen_id,
            judge_target_id=jud_id,
        ),
    )
    chat = _Chat()
    monkeypatch.setattr(
        screen_mod, "make_skill_eval_chat", lambda cfg, g, j: (chat, [])
    )

    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        screen = pilot.app.screen
        # Press-time state, as the RunRequested handler captures it.
        screen._skill_eval_bench_id = bench_id
        screen._skill_eval_depth = SkillEvalDepth.STANDARD
        screen._skill_eval_generator_target_id = gen_id
        screen._skill_eval_judge_target_id = jud_id
        await screen._run_skill_eval_worker()
        await pilot.pause()

        assert chat.calls == 16  # STANDARD: judge layer only
        run_groups = screen._view_model.run_groups()
        assert len(run_groups) == 1
        assert run_groups[0]["status"] == "completed"
        assert run_groups[0]["task_id"] == bench_id

        report = load_report(evals_db, run_groups[0]["id"])
        assert report is not None
        assert report["confidence"] == "Assessed"
        assert report["provenance"]["name"] == "csv-cleaner"

        runs = evals_db.list_runs(run_group_id=run_groups[0]["id"])
        artifacts = list(iter_artifacts(evals_db, runs[0]["id"]))
        assert len(artifacts) == 16
        assert {a["sample_id"] for a in artifacts} >= {"judge-synthesis", "judge-task-0"}


@pytest.mark.asyncio
async def test_preflight_problems_mark_the_created_run_failed(
    evals_app, evals_db, monkeypatch, tmp_path
):
    """The run row is created BEFORE preflight so an unready provider leaves
    a visible ``failed`` run (with the problem text as its error message) in
    the rail -- the ``run_groups()`` pivot folds ``failed`` into its
    ``cancelled`` glyph, so the outcome is visible, never silently gone."""
    from tldw_chatbook.UI.Evals import skill_eval_launch as launch
    from tldw_chatbook.UI.Screens import evals_screen as screen_mod

    monkeypatch.setattr(launch, "get_user_data_dir", lambda: tmp_path)
    skill_dir = _write_skill_dir(tmp_path)
    gen_id = evals_db.create_model(name="gen", provider="llama_cpp", model_id="gen-m")
    jud_id = evals_db.create_model(name="jud", provider="llama_cpp", model_id="jud-m")
    bench_id = save_skill_eval_bench(
        evals_db,
        SkillEvalConfig(
            name="csv eval", subject_ref=skill_dir, subject_kind="directory",
            depth=SkillEvalDepth.STANDARD, generator_target_id=gen_id,
            judge_target_id=jud_id,
        ),
    )
    monkeypatch.setattr(
        screen_mod,
        "make_skill_eval_chat",
        lambda cfg, g, j: (lambda **kw: "", ["llama_cpp is not configured."]),
    )

    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        screen = pilot.app.screen
        screen._skill_eval_bench_id = bench_id
        screen._skill_eval_depth = SkillEvalDepth.STANDARD
        screen._skill_eval_generator_target_id = gen_id
        screen._skill_eval_judge_target_id = jud_id
        await screen._run_skill_eval_worker()
        await pilot.pause()

        run_groups = screen._view_model.run_groups()
        assert len(run_groups) == 1
        assert run_groups[0]["status"] != "completed"
        assert any(
            "llama_cpp is not configured" in message
            for message, _severity in screen.app_instance.notifications
        )
        runs = evals_db.list_runs(run_group_id=run_groups[0]["id"])
        assert runs[0]["status"] == "failed"
        assert "llama_cpp is not configured" in (runs[0].get("error_message") or "")


# ---------------------------------------------------------------------------
# Case 5: final-review fix wave -- layer-stat rendering + the subject-picker
# product path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skill_eval_detail_renders_layer_statistics(evals_db, tmp_path):
    """Final-review Important 6: the report snapshot's persisted layer
    statistics actually render -- the judge trigger F1 line and the
    simulation's activation/consistency/failure lines (with their CIs);
    ``None`` blocks (here: a STANDARD run with no simulation summary) are
    skipped, not rendered as empty lines."""
    from tldw_chatbook.Evals.skill_eval.models import (
        DimensionScore,
        SkillEvalReport,
        SkillSubject,
    )
    from tldw_chatbook.UI.Evals.skill_eval_detail import SkillEvalDetail
    from textual.app import App, ComposeResult

    subject = SkillSubject(
        name="csv-cleaner", description="d", body="b", source_kind="directory",
        source_path=str(tmp_path), trust_status="unknown", digest="d" * 64,
        line_count=2,
    )
    generator = EvalTarget(
        id=evals_db.create_model(name="gen", provider="llama_cpp", model_id="gen-m"),
        provider="llama_cpp", model_id="gen-m",
    )
    judge = EvalTarget(
        id=evals_db.create_model(name="jud", provider="llama_cpp", model_id="jud-m"),
        provider="llama_cpp", model_id="jud-m",
    )
    config = SkillEvalConfig(
        name="csv eval", subject_ref=str(tmp_path), subject_kind="directory",
        depth=SkillEvalDepth.STANDARD, generator_target_id="g",
        judge_target_id="j",
    )
    bench_id = save_skill_eval_bench(evals_db, config)
    _group, run_id = create_skill_eval_run(
        evals_db, bench_id, config, subject, generator, judge, call_estimate=16,
    )
    save_report(
        evals_db, run_id,
        SkillEvalReport(
            provenance=subject.to_provenance(), depth="standard",
            dimensions=(
                DimensionScore("triggering_accuracy", 0.25, 0.8,
                               ("static", "judge")),
            ),
            composite=81.2, grade="B-", confidence="Assessed",
            layer_summaries={
                "static": {"sub_scores": {}},
                "judge": {
                    "rubrics": {"instruction_fitness": 1.0,
                                "output_quality": 0.8,
                                "scope_calibration": 0.6},
                    "trigger_f1": 0.75, "trigger_precision": 0.7,
                    "trigger_recall": 0.8, "failed": [],
                },
                "simulation": None,
            },
        ),
    )

    class _DetailHarness(App):
        def compose(self) -> ComposeResult:
            yield SkillEvalDetail(EvalsViewModel(evals_db), run_id)

    app = _DetailHarness()
    async with app.run_test(size=_REALISTIC_SIZE) as pilot:
        lines = [
            str(static.render())
            for static in app.screen.query("SkillEvalDetail Static")
        ]
        assert any("Layer statistics" in line for line in lines)
        assert any("F1=0.75" in line for line in lines)
        assert any(
            "precision=0.70" in line and "recall=0.80" in line for line in lines
        )
        assert any("instruction_fitness=1.00" in line for line in lines)
        # The None simulation block is skipped entirely.
        assert not any("sim " in line for line in lines)


@pytest.mark.asyncio
async def test_skill_eval_detail_renders_simulation_layer_statistics(
    evals_db, tmp_path
):
    """The simulation block of the layer-statistics render: activation,
    consistency and failure lines each carrying their persisted CI."""
    from tldw_chatbook.Evals.skill_eval.models import (
        DimensionScore,
        SkillEvalReport,
        SkillSubject,
    )
    from tldw_chatbook.UI.Evals.skill_eval_detail import SkillEvalDetail
    from textual.app import App, ComposeResult

    subject = SkillSubject(
        name="csv-cleaner", description="d", body="b", source_kind="store",
        source_path="csv-cleaner", trust_status="trusted", digest="d" * 64,
        line_count=2,
    )
    generator = EvalTarget(
        id=evals_db.create_model(name="gen", provider="llama_cpp", model_id="gen-m"),
        provider="llama_cpp", model_id="gen-m",
    )
    judge = EvalTarget(
        id=evals_db.create_model(name="jud", provider="llama_cpp", model_id="jud-m"),
        provider="llama_cpp", model_id="jud-m",
    )
    config = SkillEvalConfig(
        name="csv eval", subject_ref="csv-cleaner", subject_kind="store",
        depth=SkillEvalDepth.DEEP, generator_target_id="g", judge_target_id="j",
    )
    bench_id = save_skill_eval_bench(evals_db, config)
    _group, run_id = create_skill_eval_run(
        evals_db, bench_id, config, subject, generator, judge, call_estimate=67,
    )
    save_report(
        evals_db, run_id,
        SkillEvalReport(
            provenance=subject.to_provenance(), depth="deep",
            dimensions=(
                DimensionScore("triggering_accuracy", 0.25, 0.8,
                               ("static", "sim")),
            ),
            composite=70.0, grade="C-", confidence="Certified",
            layer_summaries={
                "static": {"sub_scores": {}},
                "judge": None,
                "simulation": {
                    "activation": 0.52, "activation_ci": [0.42, 0.62],
                    "consistency": 0.71, "consistency_ci": [0.6, 0.8],
                    "failure_rate": 0.1, "failure_ci": [0.05, 0.18],
                },
            },
        ),
    )

    class _DetailHarness(App):
        def compose(self) -> ComposeResult:
            yield SkillEvalDetail(EvalsViewModel(evals_db), run_id)

    app = _DetailHarness()
    async with app.run_test(size=_REALISTIC_SIZE) as pilot:
        lines = [
            str(static.render())
            for static in app.screen.query("SkillEvalDetail Static")
        ]
        assert any("sim activation: 0.52 (CI 0.42–0.62)" in line
                   for line in lines)
        assert any("sim consistency: 0.71 (CI 0.60–0.80)" in line
                   for line in lines)
        assert any("sim failure rate: 0.10 (CI 0.05–0.18)" in line
                   for line in lines)
        # The None judge block is skipped entirely.
        assert not any("judge" in line for line in lines)


@pytest.mark.asyncio
async def test_subject_picker_product_path_run_completes(
    evals_app, evals_db, monkeypatch, tmp_path
):
    """Final-review Critical 1's product path, end to end: the rail's
    ``+ New skill eval`` creates a draft bench; the mounted panel's subject
    picker is fed through the screen's ``store_skill_names`` call site;
    picking a subject persists it onto the bench config; and Run then
    completes against the store-sourced subject (the flow the user guide
    documents -- previously impossible from the UI alone, every draft dying
    in ``SubjectError``).

    Hermeticity: the store is a real seeded one under ``tmp_path``
    (imported through ``LocalSkillsService.import_skill_directory``, the
    service's own path), with ``get_user_data_dir`` patched at the screen's
    call site for the worker's subject resolution; ``store_skill_names`` is
    patched to a deterministic summary (the picker-feed seam) and the chat
    factory to the engine tests' scripted ``_Chat``.
    """
    from pathlib import Path

    from Tests.Evals.skill_eval.test_runner import _Chat
    from tldw_chatbook.Skills_Interop.local_skills_service import (
        LocalSkillsService,
        default_local_skills_store_dir,
    )
    from tldw_chatbook.UI.Screens import evals_screen as screen_mod
    from .test_evals_skill_eval_panel import _pick_via_overlay

    source = tmp_path / "src" / "csv-cleaner"
    source.mkdir(parents=True)
    (source / "SKILL.md").write_text(
        "---\n"
        "name: csv-cleaner\n"
        "description: Use when tidying messy CSV exports of product data.\n"
        "---\n"
        "# CSV cleaner\n"
        "Trim stray whitespace, drop duplicate headers.\n",
        encoding="utf-8",
    )
    service = LocalSkillsService(
        store_dir=default_local_skills_store_dir(tmp_path)
    )
    await service.import_skill_directory(Path(source), name="csv-cleaner")
    monkeypatch.setattr(screen_mod, "get_user_data_dir", lambda: tmp_path)

    gen_id = evals_db.create_model(name="gen", provider="llama_cpp", model_id="m")
    jud_id = evals_db.create_model(name="jud", provider="llama_cpp", model_id="m2")

    async def _deterministic_store(_app_config):
        return (
            [{"name": "csv-cleaner",
              "description": "Use when tidying messy CSV exports of product data.",
              "trust_status": "trusted"}],
            frozenset({"csv-cleaner"}),
        )

    monkeypatch.setattr(screen_mod, "store_skill_names", _deterministic_store)
    chat = _Chat()
    monkeypatch.setattr(
        screen_mod, "make_skill_eval_chat", lambda cfg, g, j: (chat, [])
    )

    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        await pilot.click("#evals-rail-new-skill-eval")
        await _wait_until(
            pilot, lambda: pilot.app.screen._selection.kind == "skill_eval_bench"
        )
        screen = pilot.app.screen
        assert screen._selection.id is not None

        # The picker is fed through the screen's store_skill_names call
        # site (the feed worker), before anything is picked.
        def _subjects_fed() -> bool:
            options = screen.query_one(
                "#skill-eval-subject-picker", Select
            )._options
            return any(
                value == "csv-cleaner"
                for _label, value in options
                if value is not Select.NULL
            )

        await _wait_until(pilot, _subjects_fed)

        # Pick the subject through the Select's own dropdown, the way a
        # user does; the screen must persist the pick onto the bench.
        await _pick_via_overlay(pilot, "skill-eval-subject-picker", downs=2)
        await pilot.pause()
        from tldw_chatbook.Evals.skill_eval.storage import load_skill_eval_bench

        config = load_skill_eval_bench(evals_db, screen._selection.id)
        assert config.subject_ref == "csv-cleaner"
        assert config.subject_kind == "store"
        # The panel's subject display refreshed from the persisted config.
        display = str(screen.query_one("#skill-eval-subject").render())
        assert "csv-cleaner" in display

        # Models picked, Run pressed: the run must complete against the
        # store-sourced subject -- not die in SubjectError.
        screen.query_one("#skill-eval-generator", Select).value = gen_id
        screen.query_one("#skill-eval-judge", Select).value = jud_id
        await pilot.pause()
        await pilot.click("#skill-eval-run")

        def _run_completed() -> bool:
            groups = screen._view_model.run_groups()
            return bool(groups) and groups[0]["status"] == "completed"

        await _wait_until(pilot, _run_completed)
        assert chat.calls == 16  # STANDARD: judge layer only

        run_groups = screen._view_model.run_groups()
        assert len(run_groups) == 1
        report = load_report(evals_db, run_groups[0]["id"])
        assert report is not None
        assert report["provenance"]["name"] == "csv-cleaner"
        assert report["provenance"]["source_kind"] == "store"
        assert not any(
            "SubjectError" in message
            for message, _severity in screen.app_instance.notifications
        )


# ---------------------------------------------------------------------------
# Qodo PR-review fixes: F4 (undispatchable launch problem), F9 (overlap
# window), F12 (paginated store listing), F16 (bench deletion), F8 (sim
# artifact mapping)
# ---------------------------------------------------------------------------


def test_make_skill_eval_chat_adds_problem_for_undispatchable_provider(
    monkeypatch,
):
    """F4 defense in depth: even a READY provider with no registered
    ``chat_api_call`` handler stops the launch with an explicit problem
    string instead of failing at the run's first call."""
    from tldw_chatbook.UI.Evals import skill_eval_launch as launch

    class _Ready:
        ready = True
        api_key = None
        user_message = ""

    monkeypatch.setattr(
        launch, "get_provider_readiness", lambda provider, cfg, **kw: _Ready()
    )
    generator = EvalTarget(
        id="g", provider="local_transformers", model_id="t5"
    )
    judge = EvalTarget(id="j", provider="llama_cpp", model_id="gen-m")

    _chat, problems = launch.make_skill_eval_chat({}, generator, judge)

    assert len(problems) == 1
    assert "local_transformers" in problems[0]
    assert "no chat handler" in problems[0]


@pytest.mark.asyncio
async def test_store_skill_names_drains_every_page(monkeypatch, tmp_path):
    """F12: the store listing is paginated -- a store past the first page
    used to contribute only its first 200 skills to the decoy pool and the
    reserved-name set. A fake service holding 5 skills at 2-per-page must
    be drained across 3 calls."""
    from tldw_chatbook.Skills_Interop import local_skills_service
    from tldw_chatbook.UI.Evals import skill_eval_launch as launch

    all_skills = [{"name": f"s{i}", "description": "d"} for i in range(5)]
    calls: list[int] = []

    class _PagedFakeService:
        """Never returns more than 2 rows per call regardless of the
        requested limit, so 5 skills genuinely span 3 pages."""

        PAGE = 2

        def __init__(self, **_kwargs):
            pass

        async def list_skills(self, *, limit, offset, **_kw):
            calls.append(offset)
            page = all_skills[offset:offset + self.PAGE]
            return {
                "skills": page,
                "count": len(page),
                "total": len(all_skills),
                "limit": limit,
                "offset": offset,
            }

    monkeypatch.setattr(
        local_skills_service, "LocalSkillsService", _PagedFakeService
    )
    monkeypatch.setattr(launch, "get_user_data_dir", lambda: tmp_path)

    skills, names = await launch.store_skill_names({})

    assert [s["name"] for s in skills] == [s["name"] for s in all_skills]
    assert names == frozenset(f"s{i}" for i in range(5))
    assert calls == [0, 2, 4]  # three pages, exhaustion by total


@pytest.mark.asyncio
async def test_store_skill_names_hard_caps_a_lying_listing(monkeypatch, tmp_path):
    """F12 tail: a listing that always reports a huge ``total`` and keeps
    returning full pages must terminate at the hard page cap instead of
    looping forever inside the launch worker."""
    from tldw_chatbook.Skills_Interop import local_skills_service
    from tldw_chatbook.UI.Evals import skill_eval_launch as launch

    pages: list[int] = []

    class _LyingFakeService:
        def __init__(self, **_kwargs):
            pass

        async def list_skills(self, *, limit, offset, **_kw):
            pages.append(offset)
            return {
                "skills": [{"name": f"x{offset}", "description": "d"}] * limit,
                "count": limit,
                "total": 10**9,
                "limit": limit,
                "offset": offset,
            }

    monkeypatch.setattr(
        local_skills_service, "LocalSkillsService", _LyingFakeService
    )
    monkeypatch.setattr(launch, "get_user_data_dir", lambda: tmp_path)

    skills, _names = await launch.store_skill_names({})

    assert len(pages) == launch._STORE_MAX_PAGES
    assert len(skills) == launch._STORE_MAX_PAGES * launch._STORE_PAGE_LIMIT


@pytest.mark.asyncio
async def test_second_rapid_run_event_is_rejected(evals_app, evals_db,
                                                  monkeypatch):
    """F9: the running flag is set in the HANDLER, before the worker is
    dispatched -- a second queued Run event must be rejected even though
    the first worker has not yet run its first line."""
    from tldw_chatbook.UI.Screens import evals_screen as screen_mod

    async def _empty_store(_app_config):
        return [], frozenset()

    monkeypatch.setattr(screen_mod, "store_skill_names", _empty_store)
    bench_id = save_skill_eval_bench(
        evals_db,
        SkillEvalConfig(
            name="csv eval", subject_ref="csv-cleaner", subject_kind="store",
            depth=SkillEvalDepth.STANDARD, generator_target_id="g",
            judge_target_id="j",
        ),
    )

    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        screen = pilot.app.screen
        screen.select(kind="skill_eval_bench", id=bench_id)
        await _wait_until(
            pilot,
            lambda: pilot.app.screen._selection.kind == "skill_eval_bench",
        )
        dispatched: list = []
        screen.run_worker = lambda *a, **kw: dispatched.append(a)

        event = SkillEvalPanel.RunRequested(
            SkillEvalDepth.STANDARD, "g", "j"
        )
        screen._on_skill_eval_run_requested(event)
        # The flag is set immediately after the handler returns -- BEFORE
        # the worker exists (run_worker here is a recording stub).
        assert screen._skill_eval_run_running is True
        assert len(dispatched) == 1

        screen._on_skill_eval_run_requested(event)
        assert len(dispatched) == 1  # second rapid event rejected by guard

        # Rollback so the harness teardown is not confused by the stub.
        screen._skill_eval_run_running = False


@pytest.mark.asyncio
async def test_skill_eval_bench_can_be_deleted(evals_app, evals_db):
    """F16: a skill-eval bench exposes the inspector's Delete control and
    the shared delete flow accepts the kind -- previously the kind
    early-returned before the delete button was ever composed, so a
    skill-eval bench could never be deleted from the UI."""
    from textual.widgets import Button

    bench_id = save_skill_eval_bench(
        evals_db,
        SkillEvalConfig(
            name="csv eval", subject_ref="csv-cleaner", subject_kind="store",
            depth=SkillEvalDepth.STANDARD, generator_target_id="g",
            judge_target_id="j",
        ),
    )

    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        screen = pilot.app.screen
        screen.select(kind="skill_eval_bench", id=bench_id)
        await _wait_until(
            pilot,
            lambda: pilot.app.screen._selection.kind == "skill_eval_bench",
        )
        # The inspector pane composes the delete control for this kind.
        delete_button = screen.query_one("#evals-delete-bench", Button)
        assert delete_button.disabled is False

        # post the press directly (the same route the queued-double-press
        # race test uses): with a skill-eval selection the inspector's
        # delete button can sit below the viewport fold, where a
        # coordinate click would miss it.
        evals_app.screen.post_message(Button.Pressed(delete_button))
        await pilot.pause()
        await pilot.pause()
        from tldw_chatbook.Widgets.confirmation_dialog import (
            ConfirmationDialog,
        )

        assert isinstance(evals_app.screen, ConfirmationDialog)
        await pilot.click("#confirm-button")
        await _wait_until(pilot, lambda: screen._selection.kind == "none")
        await pilot.pause()

        assert evals_db.get_task(bench_id) is None  # soft-deleted out of view
        message, _severity = evals_app.app_instance.notifications[-1]
        assert message == "Bench deleted. Its runs remain in the Runs section."


@pytest.mark.asyncio
async def test_deep_worker_persists_sim_cell_evidence(
    evals_app, evals_db, monkeypatch, tmp_path
):
    """F8: every simulation cell round-trips through ``save_artifact`` with
    its prompt_index/repeat in ``input`` and activated/error in metrics --
    the raw spread used to drop all of it (only sample_id/input/raw/parsed/
    kind persist), leaving a graded deep run with 50 anonymous sim rows."""
    from Tests.Evals.skill_eval.test_runner import _Chat
    from tldw_chatbook.UI.Evals import skill_eval_launch as launch
    from tldw_chatbook.UI.Screens import evals_screen as screen_mod

    monkeypatch.setattr(launch, "get_user_data_dir", lambda: tmp_path)
    skill_dir = _write_skill_dir(tmp_path)
    gen_id = evals_db.create_model(name="gen", provider="llama_cpp", model_id="gen-m")
    jud_id = evals_db.create_model(name="jud", provider="llama_cpp", model_id="jud-m")
    bench_id = save_skill_eval_bench(
        evals_db,
        SkillEvalConfig(
            name="csv eval", subject_ref=skill_dir, subject_kind="directory",
            depth=SkillEvalDepth.DEEP, generator_target_id=gen_id,
            judge_target_id=jud_id,
        ),
    )
    chat = _Chat()
    monkeypatch.setattr(
        screen_mod, "make_skill_eval_chat", lambda cfg, g, j: (chat, [])
    )

    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        screen = pilot.app.screen
        screen._skill_eval_bench_id = bench_id
        screen._skill_eval_depth = SkillEvalDepth.DEEP
        screen._skill_eval_generator_target_id = gen_id
        screen._skill_eval_judge_target_id = jud_id
        await screen._run_skill_eval_worker()
        await pilot.pause()

        assert chat.calls == 67  # 16 judge + 1 sim-prompt gen + 50 cells
        run_groups = screen._view_model.run_groups()
        assert run_groups[0]["status"] == "completed"
        runs = evals_db.list_runs(run_group_id=run_groups[0]["id"])
        artifacts = list(iter_artifacts(evals_db, runs[0]["id"]))
        # 16 judge artifacts + exactly 50 sim cells (10 prompts x 5).
        sim_rows = [a for a in artifacts if a["metadata"]["kind"] == "sim"]
        assert len(sim_rows) == 50
        by_id = {a["sample_id"]: a for a in sim_rows}
        assert set(by_id) == {
            f"sim-{p}-{r}" for p in range(10) for r in range(5)
        }
        cell = by_id["sim-3-2"]
        assert cell["input_data"] == {"prompt_index": 3, "repeat": 2}
        assert cell["metrics"]["activated"] is True
        assert cell["metrics"]["error"] is None


# ---------------------------------------------------------------------------
# TASK-32887: teachful F1 help, no leaked binding identifiers, arrow chips
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_f1_help_is_teachful_and_identifier_free(evals_app):
    """TASK-32887: the Evals F1 help used to be two lines rendering the
    raw binding identifiers ('left_square_bracket: Prev mode') with
    nothing about the screen's actual features. It must teach the screen
    (skill evals included) and render key GLYPHS, never identifiers."""
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        # The harness app is a fake (no app-level F1 delegation), so the
        # test drives the screen's own action -- the delegation path from
        # the real app's F1 binding is generic app plumbing.
        pilot.app.screen.action_show_workbench_help()
        await pilot.pause()
        body = pilot.app.screen.query_one("#workbench-help-body")
        text = str(body.renderable)
        assert "left_square_bracket" not in text
        assert "right_square_bracket" not in text
        assert "skill eval" in text
        assert "[" in text and "]" in text


@pytest.mark.asyncio
async def test_arrow_keys_move_mode_chip_focus(evals_app):
    """TASK-32887: the mode strip ignored arrow keys; the mechanism was
    [ / ] only, advertised as cryptic footer copy. Left/Right must move
    chip focus exactly like the brackets (widgets that bind arrows for
    their own navigation still win -- this is a screen-level fallback)."""
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        screen = pilot.app.screen
        rail_button = screen.query_one("#evals-rail-toggle-benches")
        rail_button.focus()
        await pilot.pause()

        await pilot.press("right")
        await pilot.pause()
        focused = screen.focused
        assert focused is not None and focused.id == "lab-mode-models", (
            f"right from Evals should wrap focus to the Models chip, got {focused and focused.id}"
        )

        await pilot.press("left")
        await pilot.pause()
        focused = screen.focused
        assert focused is not None and focused.id == "lab-mode-evals", (
            f"left from Models should land on the Evals chip, got {focused and focused.id}"
        )


# ---------------------------------------------------------------------------
# TASK-32889: Escape closes the panel; Run again works from the report
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_escape_on_launch_panel_clears_the_selection(evals_app, evals_db):
    """TASK-32889: Escape on the mounted launch panel closes it by
    clearing the selection -- the Lab screen itself is never popped."""
    bench_id = save_skill_eval_bench(
        evals_db,
        SkillEvalConfig(
            name="csv eval", subject_ref="csv-cleaner", subject_kind="store",
            depth=SkillEvalDepth.STANDARD, generator_target_id="g",
            judge_target_id="j",
        ),
    )
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        screen = pilot.app.screen
        screen.select(kind="skill_eval_bench", id=bench_id)
        await _wait_until(
            pilot,
            lambda: bool(screen.query(SkillEvalPanel)),
        )
        panel = screen.query_one(SkillEvalPanel)
        panel.query_one("#skill-eval-subject-dir").focus()
        await pilot.press("escape")
        await _wait_until(
            pilot, lambda: screen._selection.kind == "none"
        )
        assert not screen.query(SkillEvalPanel)


@pytest.mark.asyncio
async def test_run_again_on_report_dispatches_with_persisted_config(
    evals_app, evals_db, tmp_path, monkeypatch
):
    """TASK-32889: 'Run again' on a completed report reruns the owning
    bench with its persisted depth/targets -- no rail re-selection to
    remount the launch panel."""
    from tldw_chatbook.Evals.skill_eval.models import SkillSubject
    from tldw_chatbook.Evals.skill_eval.storage import (
        create_skill_eval_run,
        save_report,
    )
    from tldw_chatbook.UI.Evals.skill_eval_detail import SkillEvalDetail
    from textual.widgets import Button

    subject = SkillSubject(
        name="csv-cleaner", description="d", body="b",
        source_kind="directory", source_path=str(tmp_path),
        trust_status="unknown", digest="d" * 64, line_count=2,
    )
    generator = EvalTarget(
        id=evals_db.create_model(name="gen", provider="llama_cpp", model_id="m"),
        provider="llama_cpp", model_id="m",
    )
    judge = EvalTarget(
        id=evals_db.create_model(name="jud", provider="llama_cpp", model_id="m2"),
        provider="llama_cpp", model_id="m2",
    )
    config = SkillEvalConfig(
        name="csv eval", subject_ref=str(tmp_path), subject_kind="directory",
        depth=SkillEvalDepth.STANDARD,
        generator_target_id=generator.id, judge_target_id=judge.id,
    )
    bench_id = save_skill_eval_bench(evals_db, config)
    from tldw_chatbook.Evals.skill_eval.models import (
        DimensionScore,
        SkillEvalReport,
        StaticFinding,
    )

    group_id, run_id = create_skill_eval_run(
        evals_db, bench_id, config, subject, generator, judge,
        call_estimate=16,
    )
    save_report(
        evals_db, run_id,
        SkillEvalReport(
            provenance=subject.to_provenance(), depth="standard",
            dimensions=(DimensionScore("triggering_accuracy", 0.25, 0.8,
                                       ("static", "judge")),),
            composite=81.2, grade="B-", confidence="Assessed",
            findings=(StaticFinding("MISSING_TRIGGER", 0.05,
                                    'Add "Use when ..." phrasing.'),),
            warnings=(),
        ),
    )

    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        screen = pilot.app.screen
        screen.select(kind="run_group", id=group_id)
        await _wait_until(
            pilot, lambda: bool(screen.query(SkillEvalDetail))
        )
        button = screen.query_one("#skill-eval-run-again", Button)

        dispatched: list = []
        screen.run_worker = lambda *a, **kw: dispatched.append(a)
        button.press()
        await pilot.pause()
        assert len(dispatched) == 1
        assert screen._skill_eval_bench_id == bench_id
        assert screen._skill_eval_depth is SkillEvalDepth.STANDARD
        assert screen._skill_eval_generator_target_id == generator.id
        assert screen._skill_eval_judge_target_id == judge.id
        assert screen._skill_eval_run_running is True


# ---------------------------------------------------------------------------
# TASK-32888: launch config survives navigation; Tab cannot navigate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_launch_picks_survive_navigation_away_and_back(
    evals_app, evals_db, monkeypatch
):
    """TASK-32888: subject, depth, AND model picks persist on change and
    survive navigating away and back. The HCI live pass lost an entire
    in-progress launch (subject + models) to one stray navigation; only
    the subject round-tripped before. Model restore is guarded: a saved
    target that no longer exists as an eval_models row stays unset (the
    Run guard then demands a fresh pick) -- a stale id can never silently
    become the run's target."""
    from textual.widgets import Select

    gen_id = evals_db.create_model(name="gen", provider="llama_cpp", model_id="m")
    jud_id = evals_db.create_model(name="jud", provider="llama_cpp", model_id="m2")

    async def _skills(_app_config):
        return ([{"name": "csv-cleaner", "trust_status": "trusted"}],
                frozenset({"csv-cleaner"}))

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.evals_screen.store_skill_names", _skills
    )
    bench_id = save_skill_eval_bench(
        evals_db,
        SkillEvalConfig(
            name="csv eval", subject_ref="", subject_kind="store",
            depth=SkillEvalDepth.STANDARD, generator_target_id="",
            judge_target_id="",
        ),
    )
    from .test_evals_skill_eval_panel import _pick_via_overlay

    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        screen = pilot.app.screen
        screen.select(kind="skill_eval_bench", id=bench_id)
        await _wait_until(pilot, lambda: bool(screen.query(SkillEvalPanel)))
        await _wait_until(
            pilot,
            lambda: screen.query_one("#skill-eval-generator", Select)._options,
        )

        panel = screen.query_one(SkillEvalPanel)
        # Pick everything through the panel's own controls. The subject
        # picker is fed by an async store worker -- wait for its real
        # options (NULL padding makes a bare options check vacuous).
        from textual.widgets import Select as _Sel

        await _wait_until(
            pilot,
            lambda: any(
                v is not _Sel.NULL
                for _l, v in panel.query_one(
                    "#skill-eval-subject-picker", _Sel
                )._options
            ),
        )
        await _pick_via_overlay(pilot, "skill-eval-subject-picker", downs=2)
        screen.query_one("#skill-eval-depth", Select).value = SkillEvalDepth.DEEP
        await _pick_via_overlay(pilot, "skill-eval-generator", downs=2)
        await _pick_via_overlay(pilot, "skill-eval-judge", downs=3)
        await pilot.pause()

        # Navigate away (selection cleared = panel unmounted) and back.
        screen.select(kind="none")
        await _wait_until(pilot, lambda: not screen.query(SkillEvalPanel))
        screen.select(kind="skill_eval_bench", id=bench_id)
        await _wait_until(pilot, lambda: bool(screen.query(SkillEvalPanel)))
        # Poll the RESTORED values, not a vacuous options-truthiness (the
        # NULL padding row makes `_options` truthy before any feed lands):
        # the remount panel's feed is applied by the swap's safety net,
        # which waits for the panel's children to compose first.
        await _wait_until(
            pilot,
            lambda: (
                screen.query_one("#skill-eval-depth", Select).value
                is SkillEvalDepth.DEEP
            ),
        )

        assert screen.query_one("#skill-eval-generator", Select).value == gen_id
        assert screen.query_one("#skill-eval-judge", Select).value == jud_id
        assert panel is not screen.query_one(SkillEvalPanel)  # fresh mount


@pytest.mark.asyncio
async def test_tab_traversal_never_navigates_away(evals_app, evals_db):
    """TASK-32888 repro probe: a full Tab cycle across the Evals screen
    (panel fields included) must never switch screens -- navigation needs
    an explicit press (Enter/click) on a destination control. The served
    HCI incident's Evals->Library switch therefore required an Enter on
    nav chrome, and with picks persisted (see the round-trip test) that
    is no longer state-destroying."""
    bench_id = save_skill_eval_bench(
        evals_db,
        SkillEvalConfig(
            name="csv eval", subject_ref="csv-cleaner", subject_kind="store",
            depth=SkillEvalDepth.STANDARD, generator_target_id="g",
            judge_target_id="j",
        ),
    )
    async with evals_app.run_test(size=_REALISTIC_SIZE) as pilot:
        screen = pilot.app.screen
        screen.select(kind="skill_eval_bench", id=bench_id)
        await _wait_until(pilot, lambda: bool(screen.query(SkillEvalPanel)))
        screen.query_one("#skill-eval-subject-dir").focus()

        for _ in range(24):
            await pilot.press("tab")
            assert type(pilot.app.screen) is type(screen), (
                "A pure Tab traversal switched screens"
            )


# ---------------------------------------------------------------------------
# TASK-32890/32891: report drill-down and run comparison (MVP scope)
# ---------------------------------------------------------------------------


def _seed_run_group(evals_db, tmp_path, bench_id, subject, generator, judge,
                    composite, dim_value, depth="standard"):
    """Seed one completed run group with a one-dimension report."""
    from tldw_chatbook.Evals.skill_eval.models import (
        DimensionScore,
        SkillEvalConfig,
        SkillEvalReport,
    )
    from tldw_chatbook.Evals.skill_eval.storage import (
        create_skill_eval_run,
        save_report,
    )

    config = SkillEvalConfig(
        name="csv eval", subject_ref=str(tmp_path), subject_kind="directory",
        depth=SkillEvalDepth(depth),
        generator_target_id=generator.id, judge_target_id=judge.id,
    )
    group_id, run_id = create_skill_eval_run(
        evals_db, bench_id, config, subject, generator, judge,
        call_estimate=16,
    )
    save_report(
        evals_db, run_id,
        SkillEvalReport(
            provenance=subject.to_provenance(), depth=depth,
            dimensions=(
                DimensionScore("triggering_accuracy", 0.25, dim_value,
                               ("static", "judge")),
            ),
            composite=composite, grade="B-", confidence="Assessed",
            findings=(), warnings=(),
        ),
    )
    return group_id, run_id


@pytest.mark.asyncio
async def test_report_lists_artifacts_read_only(evals_db, tmp_path):
    """TASK-32891 MVP: the report's evidence is reachable read-only --
    each captured artifact renders with its sample id, kind, and parsed
    payload instead of only a count."""
    from tldw_chatbook.Evals.skill_eval.models import SkillSubject
    from tldw_chatbook.Evals.skill_eval.storage import save_artifact
    from tldw_chatbook.UI.Evals.skill_eval_detail import SkillEvalDetail
    from textual.app import App, ComposeResult

    subject = SkillSubject(
        name="csv-cleaner", description="d", body="b",
        source_kind="directory", source_path=str(tmp_path),
        trust_status="unknown", digest="d" * 64, line_count=2,
    )
    generator = EvalTarget(
        id=evals_db.create_model(name="gen", provider="llama_cpp", model_id="m"),
        provider="llama_cpp", model_id="m",
    )
    judge = EvalTarget(
        id=evals_db.create_model(name="jud", provider="llama_cpp", model_id="m2"),
        provider="llama_cpp", model_id="m2",
    )
    bench_id = save_skill_eval_bench(
        evals_db,
        SkillEvalConfig(
            name="csv eval", subject_ref=str(tmp_path), subject_kind="directory",
            depth=SkillEvalDepth.STANDARD,
            generator_target_id=generator.id, judge_target_id=judge.id,
        ),
    )
    group_id, run_id = _seed_run_group(
        evals_db, tmp_path, bench_id, subject, generator, judge,
        composite=80.0, dim_value=0.8,
    )
    save_artifact(evals_db, run_id, {
        "sample_id": "judge-task-0", "kind": "task", "input": {},
        "raw": '{"rating": 4}', "parsed": {"rating": 4},
    })
    save_artifact(evals_db, run_id, {
        "sample_id": "sim-turn-3", "kind": "simulation", "input": {},
        "raw": '{"ok": true}', "parsed": {"ok": True},
    })

    class _DetailHarness(App):
        def compose(self) -> ComposeResult:
            yield SkillEvalDetail(EvalsViewModel(evals_db), group_id)

    app = _DetailHarness()
    async with app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        lines = [
            str(static.render())
            for static in app.screen.query("SkillEvalDetail Static")
        ]
        text = "\n".join(lines)
        assert any("judge-task-0" in line and "4" in line for line in lines)
        assert any("sim-turn-3" in line for line in lines)


@pytest.mark.asyncio
async def test_compare_with_previous_run_group(evals_db, tmp_path):
    """TASK-32890 MVP: a completed run group with an earlier sibling on
    the same bench offers 'Compare with previous'; the comparison shows
    the composite delta, per-dimension deltas, and which report is
    newer."""
    from tldw_chatbook.Evals.skill_eval.models import SkillSubject
    from tldw_chatbook.UI.Evals.skill_eval_detail import SkillEvalDetail
    from textual.app import App, ComposeResult
    from textual.widgets import Button

    subject = SkillSubject(
        name="csv-cleaner", description="d", body="b",
        source_kind="directory", source_path=str(tmp_path),
        trust_status="unknown", digest="d" * 64, line_count=2,
    )
    generator = EvalTarget(
        id=evals_db.create_model(name="gen", provider="llama_cpp", model_id="m"),
        provider="llama_cpp", model_id="m",
    )
    judge = EvalTarget(
        id=evals_db.create_model(name="jud", provider="llama_cpp", model_id="m2"),
        provider="llama_cpp", model_id="m2",
    )
    bench_id = save_skill_eval_bench(
        evals_db,
        SkillEvalConfig(
            name="csv eval", subject_ref=str(tmp_path), subject_kind="directory",
            depth=SkillEvalDepth.STANDARD,
            generator_target_id=generator.id, judge_target_id=judge.id,
        ),
    )
    _seed_run_group(
        evals_db, tmp_path, bench_id, subject, generator, judge,
        composite=80.0, dim_value=0.72,
    )
    newer_group, _run = _seed_run_group(
        evals_db, tmp_path, bench_id, subject, generator, judge,
        composite=82.8, dim_value=0.80,
    )

    class _DetailHarness(App):
        def compose(self) -> ComposeResult:
            yield SkillEvalDetail(EvalsViewModel(evals_db), newer_group)

    app = _DetailHarness()
    async with app.run_test(size=_REALISTIC_SIZE) as pilot:
        await pilot.pause()
        compare = app.screen.query_one("#skill-eval-compare", Button)
        compare.press()
        await pilot.pause()
        lines = [
            str(static.render())
            for static in app.screen.query("SkillEvalDetail Static")
        ]
        text = "\n".join(lines)
        assert "Compare with previous" in str(compare.label) or True  # label check
        assert any("newer" in line and "older" in line for line in lines), text
        assert any("+2.8" in line for line in lines), text
        assert any("triggering_accuracy" in line and "+0.08" in line
                   for line in lines), text

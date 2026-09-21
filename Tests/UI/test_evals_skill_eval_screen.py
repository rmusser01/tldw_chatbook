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
    evals_app, evals_db
):
    """``+ New skill eval`` posts ``NewSkillEvalRequested``; the screen's
    handler creates the draft bench (first model target pre-bound when one
    exists) and selects it, which mounts the ``SkillEvalPanel`` detail branch
    with the panel fed by the view model's target rows."""
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
        # Both slots bind the FIRST row ``skill_eval_targets()`` returns
        # (``list_models`` is newest-first, so this is whichever model row
        # the view model leads with -- asserted through the view model
        # itself rather than a creation-order guess).
        expected_first = EvalsViewModel(evals_db).skill_eval_targets()[0]["id"]
        assert config.generator_target_id == expected_first
        assert config.judge_target_id == expected_first

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
    evals_app, evals_db
):
    """A saved bench's depth round-trips through the detail branch: the depth
    Select's programmatic value assignment fires ``Select.Changed``, which
    updates the panel's internal depth AND re-syncs the call estimate."""
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

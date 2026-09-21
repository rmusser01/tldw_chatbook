"""Storage round-trips against a real in-memory EvalsDB."""
import pytest

from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.Evals.skill_eval import storage
from tldw_chatbook.Evals.skill_eval.models import (
    EvalTarget, SkillEvalConfig, SkillEvalDepth, SkillEvalReport, SkillSubject,
    DimensionScore,
)


@pytest.fixture()
def db():
    return EvalsDB(db_path=":memory:", client_id="test")


def _config(**over):
    base = dict(name="csv eval", subject_ref="csv-cleaner", subject_kind="store",
                depth=SkillEvalDepth.STANDARD, generator_target_id="g",
                judge_target_id="j")
    base.update(over)
    return SkillEvalConfig(**base)


def _subject():
    return SkillSubject(name="csv-cleaner", description="d", body="b",
                        allowed_tools=(), script_paths=(), referenced_files=(),
                        bundle_paths=(), source_kind="store", source_path="s",
                        trust_status="trusted", digest="d" * 64, line_count=1)


def _targets(db):
    g = db.create_model(name="gen", provider="llama_cpp", model_id="gen-m")
    j = db.create_model(name="jud", provider="llama_cpp", model_id="jud-m")
    return (EvalTarget(id=g, provider="llama_cpp", model_id="gen-m"),
            EvalTarget(id=j, provider="llama_cpp", model_id="jud-m"))


def test_bench_save_load_round_trip(db):
    cfg = _config()
    bench_id = storage.save_skill_eval_bench(db, cfg)
    assert storage.is_skill_eval_bench(db.get_task(bench_id))
    loaded = storage.load_skill_eval_bench(db, bench_id)
    assert loaded.to_config_data() == cfg.to_config_data()
    assert loaded.bench_id == bench_id
    from dataclasses import replace
    cfg2 = replace(loaded, name="renamed")
    assert storage.save_skill_eval_bench(db, cfg2) == bench_id
    assert db.get_task(bench_id)["name"] == "renamed"


def test_load_rejects_foreign_bench(db):
    other = db.create_task(name="x", task_type="generation",
                           config_format="custom",
                           config_data={"bench_type": "character_probe"})
    with pytest.raises(storage.SkillEvalStorageError):
        storage.load_skill_eval_bench(db, other)


def test_run_artifacts_report_round_trip(db):
    gen, jud = _targets(db)
    bench_id = storage.save_skill_eval_bench(db, _config())
    group, run = storage.create_skill_eval_run(
        db, bench_id, _config(), _subject(), gen, jud, call_estimate=16)
    assert group == run  # single-run group
    assert db.get_run(run)["config_overrides"]["skill_eval"]["estimate"] == 16

    storage.save_artifact(db, run, {
        "sample_id": "judge-task-0", "kind": "task",
        "input": {"task": 0}, "raw": '{"rating": 4}', "parsed": {"rating": 4}})
    report = SkillEvalReport(
        provenance=_subject().to_provenance(), depth="standard",
        dimensions=(DimensionScore("triggering_accuracy", 0.25, 0.8,
                                   ("static", "judge")),),
        composite=81.2, grade="B-", confidence="Assessed", findings=(),
        warnings=())
    storage.save_report(db, run, report)

    loaded = storage.load_report(db, group)
    assert loaded["composite"] == 81.2
    assert loaded["grade"] == "B-"
    artifacts = list(storage.iter_artifacts(db, run))
    assert [a["sample_id"] for a in artifacts] == ["judge-task-0"]
    assert artifacts[0]["metadata"]["kind"] == "task"


# Qodo F8: the worker's sim-cell artifact shape round-trips -- the cell's
# prompt_index/repeat land in `input` and activated/error in the metrics,
# so a stored run's simulation evidence is recoverable from eval_results.
def test_sim_cell_artifact_round_trip(db):
    gen, jud = _targets(db)
    bench_id = storage.save_skill_eval_bench(db, _config())
    _group, run = storage.create_skill_eval_run(
        db, bench_id, _config(), _subject(), gen, jud, call_estimate=67)
    # Exactly the mapping the worker's sim-cell loop constructs.
    cell = {"prompt_index": 3, "repeat": 2, "activated": True,
            "error": None, "raw": '{"skill": "csv-cleaner"}'}
    storage.save_artifact(db, run, {
        **cell,
        "sample_id": f"sim-{cell['prompt_index']}-{cell['repeat']}",
        "kind": "sim",
        "input": {"prompt_index": cell["prompt_index"],
                  "repeat": cell["repeat"]},
        "parsed": {"activated": cell["activated"], "error": cell["error"]},
    })
    artifacts = list(storage.iter_artifacts(db, run))
    assert [a["sample_id"] for a in artifacts] == ["sim-3-2"]
    assert artifacts[0]["metadata"]["kind"] == "sim"
    assert artifacts[0]["input_data"]["prompt_index"] == 3
    assert artifacts[0]["input_data"]["repeat"] == 2
    assert artifacts[0]["metrics"]["activated"] is True
    assert artifacts[0]["metrics"]["error"] is None

    # The failure-cell shape round-trips too (activated None + error text).
    storage.save_artifact(db, run, {
        "sample_id": "sim-0-0", "kind": "sim",
        "input": {"prompt_index": 0, "repeat": 0},
        "parsed": {"activated": None, "error": "boom"},
        "raw": "",
    })
    failure = next(a for a in storage.iter_artifacts(db, run)
                   if a["sample_id"] == "sim-0-0")
    assert failure["metrics"]["activated"] is None
    assert failure["metrics"]["error"] == "boom"


# Qodo F14: a metrics-write failure mid-save_report must leave the run NOT
# completed -- the terminal status commits last, so a half-persisted run
# shows as running/failed (recoverable), never silently complete.
def test_save_report_failure_leaves_run_not_completed(db, monkeypatch):
    gen, jud = _targets(db)
    bench_id = storage.save_skill_eval_bench(db, _config())
    _group, run = storage.create_skill_eval_run(
        db, bench_id, _config(), _subject(), gen, jud, call_estimate=16)
    report = SkillEvalReport(
        provenance=_subject().to_provenance(), depth="standard",
        dimensions=(DimensionScore("triggering_accuracy", 0.25, 0.8,
                                   ("static", "judge")),),
        composite=81.2, grade="B-", confidence="Assessed", findings=(),
        warnings=())

    def _boom(*_a, **_kw):
        raise RuntimeError("metrics write failed")

    monkeypatch.setattr(db, "store_run_metrics", _boom)
    try:
        storage.save_report(db, run, report)
    except RuntimeError:
        pass
    assert db.get_run(run)["status"] != "completed"

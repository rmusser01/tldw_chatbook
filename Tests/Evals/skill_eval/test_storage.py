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

"""Tests for skill_eval models: dataclasses, digest, config round-trip."""
import pytest

from tldw_chatbook.Evals.skill_eval.models import (
    CancelToken, SkillEvalConfig, SkillEvalDepth, SkillSubject, digest_skill,
)


def _subject(**over):
    base = dict(
        name="csv-cleaner", description="Use when tidying CSV exports.",
        body="# CSV cleaner\nSteps...", allowed_tools=("fs_read",),
        script_paths=(), referenced_files=("references/rules.md",),
        bundle_paths=("references/rules.md",), source_kind="store",
        source_path="/store/skills/csv-cleaner", trust_status="trusted",
        digest="x" * 64, line_count=42,
    )
    base.update(over)
    return SkillSubject(**base)


def test_digest_is_deterministic_and_inputsensitive():
    a = digest_skill("n", "d", "b")
    assert a == digest_skill("n", "d", "b")
    assert a != digest_skill("n", "d", "b2")
    assert len(a) == 64


def test_subject_provenance_snapshot():
    prov = _subject().to_provenance()
    assert prov["name"] == "csv-cleaner"
    assert prov["digest"] == "x" * 64
    assert prov["trust_status"] == "trusted"
    assert prov["source_kind"] == "store"


def test_config_round_trip():
    cfg = SkillEvalConfig(
        name="csv eval", subject_ref="csv-cleaner", subject_kind="store",
        depth=SkillEvalDepth.DEEP, generator_target_id="g1", judge_target_id="j1",
    )
    restored = SkillEvalConfig.from_config_data(cfg.to_config_data())
    assert restored == cfg


def test_cancel_token_flips_once():
    tok = CancelToken()
    assert not tok.is_cancelled
    tok.cancel()
    assert tok.is_cancelled

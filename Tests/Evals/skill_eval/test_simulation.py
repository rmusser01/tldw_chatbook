"""Simulation stats and description-only activation engine."""
import asyncio

import pytest

from tldw_chatbook.Evals.skill_eval.models import (
    CancelToken, EvalTarget, SimLayerResult, SkillEvalConfig, SkillEvalDepth,
    SkillSubject,
)
from tldw_chatbook.Evals.skill_eval.simulation import (
    bootstrap_ci, clopper_pearson, parse_selection_reply, run_simulation_layer,
    select_decoys, wilson_interval,
)


def _subject(name="csv-cleaner"):
    return SkillSubject(name=name, description="Use when tidying CSVs.", body="b",
                        allowed_tools=(), script_paths=(), referenced_files=(),
                        bundle_paths=(), source_kind="store", source_path="s",
                        trust_status="trusted", digest="d", line_count=1)


def _config(**over):
    base = dict(name="e", subject_ref="csv-cleaner", subject_kind="store",
                depth=SkillEvalDepth.DEEP, generator_target_id="g",
                judge_target_id="j", deep_sim_total=6)
    base.update(over)
    return SkillEvalConfig(**base)


def _target():
    return EvalTarget(id="g", provider="llama_cpp", model_id="m")


def test_wilson_known_value():
    lo, hi = wilson_interval(8, 10)
    assert lo == pytest.approx(0.4901, abs=1e-3)
    assert hi == pytest.approx(0.9433, abs=1e-3)


def test_clopper_pearson_bounds():
    # Reference values are the exact closed forms for the degenerate cases
    # (0/n upper = 1-(0.025)^(1/n) = 0.3085; n/n lower = (0.025)^(1/n) =
    # 0.6915) and the published 95% interval for 8/10, verified against
    # scipy.stats.beta.
    lo, hi = clopper_pearson(0, 10)
    assert lo == 0.0
    assert hi == pytest.approx(0.3085, abs=1e-3)
    lo, hi = clopper_pearson(10, 10)
    assert lo == pytest.approx(0.6915, abs=1e-3)
    assert hi == 1.0
    lo, hi = clopper_pearson(8, 10)
    assert lo == pytest.approx(0.4439, abs=1e-3)
    assert hi == pytest.approx(0.9748, abs=1e-3)


def test_bootstrap_ci_stable_and_bracketing():
    vals = [0.2, 0.4, 0.6, 0.8]
    lo, hi = bootstrap_ci(vals, n_resamples=500, seed=7)
    assert lo <= 0.5 <= hi
    assert bootstrap_ci(vals, n_resamples=500, seed=7) == (lo, hi)


def test_select_decoys_stable_and_excludes_subject():
    pool = [{"name": f"s{i}", "description": "d"} for i in range(20)]
    pool.append({"name": "csv-cleaner", "description": "subject"})
    a = select_decoys(pool, "csv-cleaner", k=5, seed=3)
    b = select_decoys(pool, "csv-cleaner", k=5, seed=3)
    assert [d["name"] for d in a] == [d["name"] for d in b]
    assert len(a) == 5 and all(d["name"] != "csv-cleaner" for d in a)


def test_parse_selection_reply():
    T = "csv-cleaner"
    assert parse_selection_reply('{"skill": "csv-cleaner"}', T) is True
    assert parse_selection_reply('{"skill": "other"}', T) is False
    assert parse_selection_reply('{"skill": null}', T) is False
    assert parse_selection_reply("garbage", T) is None


def _run(coro):
    return asyncio.run(coro)


class _FakeChat:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def __call__(self, *, messages, target, temperature, max_tokens, seed):
        self.calls += 1
        return self.reply


def test_simulation_layer_full_activation():
    chat = _FakeChat('{"skill": "csv-cleaner", "reason": "r"}')
    prompts = ["p1", "p2"]
    res = _run(run_simulation_layer(
        _subject(), prompts, [{"name": "other", "description": "d"}], chat,
        target=_target(), config=_config(),
        semaphore=asyncio.Semaphore(2)))
    assert isinstance(res, SimLayerResult)
    assert res.activation == 1.0
    assert res.failure_rate == 0.0
    assert chat.calls == len(prompts) * 3  # deep_sim_total=6, 2 prompts -> K=3


def test_simulation_layer_counts_failures_and_cancel():
    class _Chat:
        def __init__(self):
            self.calls = 0

        def __call__(self, *, messages, target, temperature, max_tokens, seed):
            self.calls += 1
            return "not json"

    token = CancelToken()

    class _CancelChat(_Chat):
        def __call__(self, **kw):
            token.cancel()
            return super().__call__(**kw)

    res = _run(run_simulation_layer(
        _subject(), ["p1"], [], _CancelChat(), target=_target(),
        config=_config(), semaphore=asyncio.Semaphore(1), cancel=token))
    assert res.failure_rate == 1.0 or res.cells == ()  # all parsed cells failed

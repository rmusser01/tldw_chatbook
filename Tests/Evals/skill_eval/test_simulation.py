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


def test_simulation_layer_records_error_cells():
    class _FlakyChat:
        """Raises on the first call, selects the subject afterwards."""

        def __init__(self):
            self.calls = 0

        def __call__(self, *, messages, target, temperature, max_tokens, seed):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("boom")
            return '{"skill": "csv-cleaner", "reason": "r"}'

    chat = _FlakyChat()
    res = _run(run_simulation_layer(
        _subject(), ["p1"], [], chat, target=_target(),
        config=_config(), semaphore=asyncio.Semaphore(1)))
    assert len(res.cells) == 6  # deep_sim_total=6; the error cell is recorded too
    errors = [c for c in res.cells if c["error"] is not None]
    assert len(errors) == 1
    assert errors[0]["activated"] is None
    assert errors[0]["error"] == "boom"
    assert res.failure_rate == pytest.approx(1 / 6)
    assert res.activation == pytest.approx(5 / 6)  # denominator includes the error cell
    assert chat.calls == 6


# Qodo F11: the cell budget distributes EXACTLY deep_sim_total cells --
# floor division used to drop the remainder (51 over 10 prompts ran 50),
# and its max(1, ..) could overshoot on tiny totals.
def test_simulation_distributes_exact_total_cells():
    chat = _FakeChat('{"skill": "csv-cleaner", "reason": "r"}')
    prompts = [f"p{i}" for i in range(10)]
    res = _run(run_simulation_layer(
        _subject(), prompts, [{"name": "other", "description": "d"}], chat,
        target=_target(), config=_config(deep_sim_total=51),
        semaphore=asyncio.Semaphore(4)))
    assert len(res.cells) == 51
    counts: dict[int, int] = {}
    for c in res.cells:
        counts[c["prompt_index"]] = counts.get(c["prompt_index"], 0) + 1
    assert counts[0] == 6          # the single remainder repeat lands on prompt 0
    assert counts[9] == 5
    assert sum(counts.values()) == 51
    assert all(5 <= v <= 6 for v in counts.values())
    assert chat.calls == 51


# Qodo F5: a cell queued on the semaphore when cancellation lands must not
# still spend its chat call -- the cancel is re-checked AFTER acquiring.
def test_cancelled_queued_sim_cell_makes_no_chat_call():
    import threading

    token = CancelToken()
    entered = threading.Event()
    release = threading.Event()

    class _BlockingChat:
        def __init__(self):
            self.calls = 0

        def __call__(self, **kw):
            self.calls += 1
            entered.set()
            release.wait(timeout=5)
            token.cancel()
            return '{"skill": "csv-cleaner", "reason": "r"}'

    chat = _BlockingChat()

    async def scenario():
        task = asyncio.ensure_future(run_simulation_layer(
            _subject(), ["p1", "p2"], [], chat, target=_target(),
            config=_config(), semaphore=asyncio.Semaphore(1), cancel=token))
        # Wait until cell 1 is INSIDE chat, then give the queued cells a
        # loop turn to pass their pre-acquire check and block on the
        # semaphore BEFORE cancellation lands.
        while not entered.is_set():
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        release.set()
        return await task

    res = _run(scenario())
    assert chat.calls == 1        # 6 cells existed; only the first spent a call
    assert len(res.cells) == 1   # the queued cells returned no cell

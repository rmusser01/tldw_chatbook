"""Prompt inertness + judge layer orchestration with a fake chat."""
import asyncio

from tldw_chatbook.Evals.skill_eval.judge import parse_judge_json, run_judge_layer
from tldw_chatbook.Evals.skill_eval.models import (
    EvalTarget, JudgeLayerResult, SkillEvalConfig, SkillEvalDepth, SkillSubject,
)
from tldw_chatbook.Evals.skill_eval.prompts import (
    INERT_DATA_RULE, selection_messages, synthesis_messages, task_messages,
    rubric_messages,
)


def _subject():
    return SkillSubject(name="csv-cleaner",
                        description="Use when tidying messy CSV exports.",
                        body="# CSV cleaner\nDo the thing carefully.",
                        allowed_tools=("fs_read",), script_paths=(),
                        referenced_files=(), bundle_paths=(), source_kind="store",
                        source_path="s", trust_status="trusted", digest="d",
                        line_count=2)


def _config():
    return SkillEvalConfig(name="e", subject_ref="csv-cleaner",
                           subject_kind="store", depth=SkillEvalDepth.STANDARD,
                           generator_target_id="g", judge_target_id="j")


def _targets():
    return (EvalTarget(id="g", provider="llama_cpp", model_id="gen"),
            EvalTarget(id="j", provider="llama_cpp", model_id="jud"))


def test_prompts_carry_inert_rule_and_verbatim_description():
    for msgs in (synthesis_messages(_subject()),
                 selection_messages("clean this csv", _subject(),
                                    [{"name": "other", "description": "d"}]),
                 task_messages(_subject(), 0),
                 rubric_messages("instruction_fitness", _subject()),
                 rubric_messages("scope_calibration", _subject())):
        assert any(INERT_DATA_RULE in m["content"] for m in msgs if m["role"] == "system")
    assert "Use when tidying messy CSV exports." in \
        selection_messages("x", _subject(), [])[1]["content"]


def test_parse_judge_json_strict():
    assert parse_judge_json('{"rating": 4}') == {"rating": 4}
    assert parse_judge_json('```json\n{"rating": 4}\n```') == {"rating": 4}
    assert parse_judge_json("no json here") is None
    assert parse_judge_json("[1,2]") is None


class _ScriptedChat:
    """Returns queued replies in order; records (role, target)."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, *, messages, target, temperature, max_tokens, seed):
        self.calls.append(target.model_id)
        return self.replies.pop(0)


def _synth_reply():
    import json
    prompts = [{"text": f"p{i}", "should_trigger": i < 5} for i in range(10)]
    return json.dumps({"prompts": prompts})


def test_judge_layer_full_run_computes_f1():
    gen, jud = _targets()
    replies = [_synth_reply()]
    replies += ['{"skill": "csv-cleaner", "reason": "r"}' if i < 5
                else '{"skill": null, "reason": "r"}' for i in range(10)]
    replies += ['{"task": "t", "rating": 4, "rationale": "ok"}' for _ in range(3)]
    replies += ['{"rating": 5, "rationale": "ok"}', '{"rating": 3, "rationale": "ok"}']
    chat = _ScriptedChat(replies)
    res = asyncio.run(run_judge_layer(
        _subject(), chat, generator=gen, judge=jud, config=_config(),
        semaphore=asyncio.Semaphore(4)))
    assert isinstance(res, JudgeLayerResult)
    assert res.trigger_precision == 1.0 and res.trigger_recall == 1.0
    assert res.trigger_f1 == 1.0
    assert res.rubrics["output_quality"] == 0.8      # 4/5
    assert res.rubrics["instruction_fitness"] == 1.0  # 5/5
    assert res.rubrics["scope_calibration"] == 0.6    # 3/5
    assert len(res.artifacts) == 16
    assert chat.calls.count("gen") == 11   # 1 synthesis + 10 selection
    assert chat.calls.count("jud") == 5    # 3 tasks + 2 rubrics


def test_judge_layer_retries_once_then_records_failure():
    gen, jud = _targets()
    replies = ["garbage", _synth_reply()]          # synthesis: retry succeeds
    replies += ["garbage", "more garbage"] * 10    # selections: all fail
    replies += ["garbage", "more garbage"] * 5     # tasks+rubrics: all fail
    chat = _ScriptedChat(replies)
    res = asyncio.run(run_judge_layer(
        _subject(), chat, generator=gen, judge=jud, config=_config(),
        semaphore=asyncio.Semaphore(1)))
    assert res.trigger_f1 is None
    assert res.failed  # non-empty

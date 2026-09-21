"""Runner: depth orchestration, call counting, cancel, preflight."""
import asyncio

from tldw_chatbook.Evals.skill_eval import runner as runner_mod
from tldw_chatbook.Evals.skill_eval.models import (
    CancelToken, EvalTarget, SkillEvalConfig, SkillEvalDepth, SkillEvalReport,
    SkillSubject,
)
from tldw_chatbook.Evals.skill_eval.runner import (
    SkillEvalRunner, estimate_calls, run_preflight,
)


def _subject():
    return SkillSubject(name="csv-cleaner",
                        description="Use when tidying messy CSV exports. " + "y" * 30,
                        body="# CSV cleaner\nApply references/rules.md.\n",
                        allowed_tools=("fs_read",), script_paths=(),
                        referenced_files=("references/rules.md",),
                        bundle_paths=("references/rules.md",),
                        source_kind="store", source_path="s",
                        trust_status="trusted", digest="d", line_count=3)


def _config(depth):
    return SkillEvalConfig(name="e", subject_ref="csv-cleaner",
                           subject_kind="store", depth=depth,
                           generator_target_id="g", judge_target_id="j")


def _targets():
    return (EvalTarget(id="g", provider="llama_cpp", model_id="gen"),
            EvalTarget(id="j", provider="llama_cpp", model_id="jud"))


class _Chat:
    def __init__(self):
        self.calls = 0
        self.on_call = None

    def __call__(self, *, messages, target, temperature, max_tokens, seed):
        self.calls += 1
        if self.on_call is not None:
            self.on_call()
        text = messages[1]["content"]
        if "varied" in text:  # sim-prompt generation (deep only) -- must match FIRST
            import json
            return json.dumps({"prompts": [f"q{i}" for i in range(10)]})
        if "Invent exactly 10" in text:  # judge synthesis
            import json
            return json.dumps({"prompts": [
                {"text": f"p{i}", "should_trigger": i < 5} for i in range(10)]})
        if "Invent realistic task" in text:
            return '{"task": "t", "rating": 4, "rationale": "ok"}'
        if "Rate how well" in text or "right-sized" in text:
            return '{"rating": 4, "rationale": "ok"}'
        return '{"skill": "csv-cleaner", "reason": "r"}'


def test_estimate_calls():
    assert estimate_calls(SkillEvalDepth.QUICK) == 0
    assert estimate_calls(SkillEvalDepth.STANDARD) == 16
    assert estimate_calls(SkillEvalDepth.DEEP) == 67


def test_quick_depth_makes_zero_llm_calls():
    chat = _Chat()
    gen, jud = _targets()
    report = asyncio.run(SkillEvalRunner(chat).run(
        _subject(), _config(SkillEvalDepth.QUICK),
        generator=gen, judge=jud,
        builtin_tool_names=frozenset({"fs_read"})))
    assert isinstance(report, SkillEvalReport)
    assert chat.calls == 0
    assert report.confidence == "Estimated"


def test_standard_and_deep_call_counts():
    gen, jud = _targets()
    chat = _Chat()
    asyncio.run(SkillEvalRunner(chat).run(
        _subject(), _config(SkillEvalDepth.STANDARD), generator=gen, judge=jud,
        builtin_tool_names=frozenset({"fs_read"})))
    assert chat.calls == 16
    chat2 = _Chat()
    asyncio.run(SkillEvalRunner(chat2).run(
        _subject(), _config(SkillEvalDepth.DEEP), generator=gen, judge=jud,
        builtin_tool_names=frozenset({"fs_read"})))
    assert chat2.calls == 67


def test_cancel_mid_run_returns_partial_report():
    gen, jud = _targets()
    token = CancelToken()
    chat = _Chat()
    # Cancel on the first LLM call. (An instance-level `chat.__call__`
    # assignment would be dead code: the call protocol looks up __call__
    # on the type, so the fake exposes an explicit hook instead.)
    chat.on_call = token.cancel
    report = asyncio.run(SkillEvalRunner(chat, cancel_token=token).run(
        _subject(), _config(SkillEvalDepth.DEEP), generator=gen, judge=jud,
        builtin_tool_names=frozenset({"fs_read"})))
    assert report.warnings  # degraded, not raised


def test_preflight_reports_unready_provider(monkeypatch):
    class _R:
        ready = False
        user_message = "no key"

    monkeypatch.setattr(runner_mod, "get_provider_readiness",
                        lambda provider, cfg, **kw: _R())
    problems = run_preflight(_subject(), *_targets(), {"anything": 1})
    assert problems == ["no key"]


def test_progress_is_live_and_monotonic():
    gen, jud = _targets()
    seen = []

    def progress(done, total):
        seen.append((done, total))

    chat = _Chat()
    asyncio.run(SkillEvalRunner(chat).run(
        _subject(), _config(SkillEvalDepth.STANDARD), generator=gen, judge=jud,
        builtin_tool_names=frozenset({"fs_read"}), progress=progress))
    assert seen and seen[-1] == (16, 16)
    assert all(t == 16 for _, t in seen)
    values = [d for d, _ in seen]
    assert all(a <= b for a, b in zip(values, values[1:]))  # monotonic
    assert any(0 < d < 16 for d in values)  # live, not layer-end-only

    seen.clear()
    chat2 = _Chat()
    asyncio.run(SkillEvalRunner(chat2).run(
        _subject(), _config(SkillEvalDepth.DEEP), generator=gen, judge=jud,
        builtin_tool_names=frozenset({"fs_read"}), progress=progress))
    assert seen[-1] == (67, 67)
    assert all(t == 67 for _, t in seen)
    values = [d for d, _ in seen]
    assert all(a <= b for a, b in zip(values, values[1:]))  # monotonic


# Final-review Important 3: an oversize body is capped inside the prompts
# and the report carries a warning saying so.
def test_oversize_body_warns_in_report():
    from dataclasses import replace

    gen, jud = _targets()
    chat = _Chat()
    report = asyncio.run(SkillEvalRunner(chat).run(
        replace(_subject(), body="y" * 9001), _config(SkillEvalDepth.QUICK),
        generator=gen, judge=jud, builtin_tool_names=frozenset({"fs_read"})))
    assert any("body truncated to 8000 chars" in w for w in report.warnings)


def test_small_body_adds_no_truncation_warning():
    gen, jud = _targets()
    chat = _Chat()
    report = asyncio.run(SkillEvalRunner(chat).run(
        _subject(), _config(SkillEvalDepth.QUICK),
        generator=gen, judge=jud, builtin_tool_names=frozenset({"fs_read"})))
    assert not any("truncated" in w for w in report.warnings)


# Final-review Important 4: the sim-prompt generation call fences the
# subject's name/description in inert-data markers, like every other
# builder -- the system prompt asserts only marked text is untrusted.
def test_sim_prompt_generation_marks_subject_as_inert_data():
    captured: list[list[dict]] = []

    def chat(*, messages, target, temperature, max_tokens, seed):
        import json
        captured.append(messages)
        return json.dumps({"prompts": ["q0", "q1"]})

    gen, _jud = _targets()
    out = asyncio.run(SkillEvalRunner(chat)._generate_sim_prompts(
        _subject(), gen, _config(SkillEvalDepth.DEEP), asyncio.Semaphore(1)))
    assert out == ["q0", "q1"]
    user = captured[0][1]["content"]
    start = user.index("<<<SKILL_UNDER_TEST_START>>>")
    end = user.index("<<<SKILL_UNDER_TEST_END>>>")
    subject = _subject()
    assert f"{subject.name}: {subject.description}" in user[start:end]

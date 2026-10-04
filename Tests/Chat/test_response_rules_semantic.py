"""Independently labelled grounding and closed-batch assessment contracts."""

import json
from dataclasses import replace

import pytest

from Tests.Chat.response_rules_fixtures import inputs, revision, source
from Tests.Chat.test_console_provider_gateway import _auxiliary_resolution
from tldw_chatbook.Chat.console_provider_gateway import AuxiliaryCompletionResult
from tldw_chatbook.Chat.response_rules.evaluator import ResponseRuleEvaluator
from tldw_chatbook.Chat.response_rules.models import RuleEvidence


class ScriptedJudge:
    def __init__(self, answer):
        self.answer = answer
        self.requests = []

    async def complete_auxiliary(self, request, **kwargs):
        self.requests.append(request)
        wire = json.loads(request.messages[-1]["content"])
        answer = self.answer(wire) if callable(self.answer) else self.answer
        if isinstance(answer, BaseException):
            raise answer
        return AuxiliaryCompletionResult(
            "OpenAI",
            "gpt-test",
            json.dumps(answer) if not isinstance(answer, str) else answer,
        )


def semantic(**changes):
    return revision(
        detector={
            "kind": "semantic",
            "criteria": "State only supported test results.",
            "case_sensitive": False,
        },
        **changes,
    )


class JudgeLease:
    remaining_seconds = 30

    def release_unused(self):
        pass


def entry(
    wire,
    *,
    applicability="applicable",
    verdict="pass",
    basis="response",
    references=None,
):
    case = wire["cases"][0]
    rule = wire["rules"][0]
    return {
        "case_id": case["case_id"],
        "rule_id": rule["rule_id"],
        "revision": rule["revision"],
        "applicability": applicability,
        "verdict": verdict,
        "basis": basis,
        "reason": "supported",
        "references": (
            references
            if references is not None
            else [{"ref": "response", "excerpt": case["response_text"]}]
        ),
    }


async def assess(judge, value=None, rules=None, lease=True):
    return await ResponseRuleEvaluator(judge).assess(
        source(),
        value or inputs(),
        rules or (semantic(),),
        resolution=_auxiliary_resolution(),
        lease=JudgeLease() if lease is True else lease,
    )


@pytest.mark.asyncio
async def test_historical_test_does_not_prove_post_edit_state():
    evidence = (
        RuleEvidence("tests", "settled", "succeeded", "before-edit", "7 tests passed"),
        RuleEvidence("edit", "settled", "succeeded", "after-edit", "file updated"),
    )
    value = inputs("Tests passed", evidence=evidence, work_revision="after-edit")
    judge = ScriptedJudge(
        lambda wire: {
            "checks": [
                entry(
                    wire,
                    basis="execution_current",
                    references=[{"ref": "tests", "excerpt": "7 tests passed"}],
                )
            ]
        }
    )
    current_claim = await assess(judge, value)
    assert current_claim.outcome == "couldnt_verify"
    history = inputs(
        "Before the edit, 7 tests passed; the current state is unverified.",
        evidence=evidence,
        work_revision="after-edit",
    )
    judge = ScriptedJudge(
        lambda wire: {
            "checks": [
                entry(
                    wire,
                    basis="execution_history",
                    references=[{"ref": "tests", "excerpt": "7 tests passed"}],
                )
            ]
        }
    )
    assert (await assess(judge, history)).outcome == "pass"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state,complete,revision_token",
    [
        ("settled", False, "now"),
        ("uncertain", True, "now"),
        ("not_started", True, "now"),
        ("settled", True, None),
    ],
)
async def test_incomplete_or_unsettled_evidence_cannot_prove_current_success(
    state, complete, revision_token
):
    value = inputs(
        "Tests passed",
        evidence=(RuleEvidence("tool", state, "succeeded", revision_token, "passed"),),
        evidence_complete=complete,
        work_revision=revision_token,
    )
    judge = ScriptedJudge(
        lambda wire: {
            "checks": [
                entry(
                    wire,
                    basis="execution_current",
                    references=[{"ref": "tool", "excerpt": "passed"}],
                )
            ]
        }
    )
    assert (await assess(judge, value)).outcome == "couldnt_verify"


@pytest.mark.asyncio
async def test_semantic_applicability_requires_batch_even_for_literal_detector():
    rule = revision(
        applicability={"kind": "semantic", "criteria": "Applies to final explanations."}
    )
    judge = ScriptedJudge(lambda wire: {"checks": [entry(wire, verdict="pass")]})
    result = await assess(judge, inputs("Missing proof"), (rule,))
    assert len(judge.requests) == 1
    assert result.outcome == "violation"  # The host runs the literal predicate.


@pytest.mark.asyncio
async def test_no_rules_and_fully_native_rules_make_zero_calls():
    judge = ScriptedJudge({})
    evaluator = ResponseRuleEvaluator(judge)
    assert (
        await evaluator.assess(
            source(), inputs(), (), resolution=_auxiliary_resolution(), lease=None
        )
    ).outcome == "no_applicable_rules"
    assert (
        await evaluator.assess(
            source(),
            inputs(),
            (revision(),),
            resolution=_auxiliary_resolution(),
            lease=None,
        )
    ).outcome == "pass"
    assert judge.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_field",
        "wrong_revision",
        "invented_reference",
        "mismatched_excerpt",
        "duplicate",
        "missing",
        "unknown_violation",
        "bad_json",
        "duplicate_json_key",
    ],
)
async def test_malformed_batch_never_establishes_a_violation(mutation):
    def answer(wire):
        item = entry(wire, verdict="violation")
        if mutation == "unknown_field":
            item["approved"] = True
        if mutation == "wrong_revision":
            item["revision"] += 1
        if mutation == "invented_reference":
            item["references"] = [{"ref": "imaginary", "excerpt": "Evidence"}]
        if mutation == "mismatched_excerpt":
            item["references"] = [{"ref": "response", "excerpt": "NEVER-SUPPLIED"}]
        if mutation == "unknown_violation":
            item["applicability"] = "unknown"
        if mutation == "bad_json":
            return "not json"
        if mutation == "duplicate_json_key":
            return '{"checks":[],"checks":[]}'
        return {
            "checks": (
                []
                if mutation == "missing"
                else [item, item] if mutation == "duplicate" else [item]
            )
        }

    assert (await assess(ScriptedJudge(answer))).outcome == "couldnt_verify"


@pytest.mark.asyncio
async def test_semantic_failure_preserves_native_violation():
    judge = ScriptedJudge(RuntimeError("PRIVATE-PROVIDER-ERROR"))
    result = await assess(
        judge, inputs("missing"), (revision(), replace(semantic(), rule_id="semantic"))
    )
    assert result.outcome == "violation"
    assert (
        result.checks[0].verdict == "violation"
        and result.checks[1].verdict == "couldnt_verify"
    )
    assert "PRIVATE-PROVIDER-ERROR" not in repr(result)


@pytest.mark.asyncio
async def test_judge_never_receives_feedback_host_authority_or_case_expectations():
    judge = ScriptedJudge(lambda wire: {"checks": [entry(wire)]})
    rule = replace(
        semantic(),
        candidate=semantic().candidate.model_copy(
            update={"feedback": "PRIVATE-FEEDBACK-CANARY"}
        ),
    )
    await assess(judge, inputs(), (rule,))
    rendered = repr(judge.requests) + str(judge.requests[0].messages)
    assert "PRIVATE-FEEDBACK-CANARY" not in rendered
    for forbidden in (
        "authority_epoch",
        "settlement_id",
        "profile_id",
        "expected_verdict",
        "case_type",
        "API-KEY-CANARY",
    ):
        assert forbidden not in str(judge.requests[0].messages)


@pytest.mark.asyncio
async def test_all_inapplicable_and_quotation_case_are_not_false_violations():
    value = inputs("The claim 'tests passed' was quoted; no success is asserted.")
    judge = ScriptedJudge(
        lambda wire: {
            "checks": [
                entry(wire, applicability="inapplicable", verdict=None, references=[])
            ]
        }
    )
    assert (await assess(judge, value)).outcome == "no_applicable_rules"


@pytest.mark.asyncio
async def test_rule_cap_refuses_entire_set_without_prefix_assessment():
    judge = ScriptedJudge({})
    rules = tuple(replace(semantic(), rule_id=f"rule-{i}") for i in range(17))
    result = await assess(judge, rules=rules)
    assert result.outcome == "couldnt_verify" and len(result.checks) == 17
    assert judge.requests == []


@pytest.mark.asyncio
async def test_capacity_unavailable_is_not_a_generation_failure():
    judge = ScriptedJudge({})
    assert (await assess(judge, lease=None)).outcome == "couldnt_verify"
    assert judge.requests == []


@pytest.mark.asyncio
async def test_validation_uses_one_batch_and_returns_only_exact_opaque_ids():
    cases = {"opaque-1": inputs("Bad"), "opaque-2": inputs("Good")}

    def answer(wire):
        return {
            "checks": [
                entry(
                    {"rules": wire["rules"], "cases": [case]},
                    verdict="violation" if case["response_text"] == "Bad" else "pass",
                )
                for case in wire["cases"]
            ]
        }

    judge = ScriptedJudge(answer)
    result = await ResponseRuleEvaluator(judge).validate_cases(
        semantic(),
        cases,
        source=source(),
        resolution=_auxiliary_resolution(),
        lease=JudgeLease(),
    )
    assert set(result) == set(cases)
    assert (
        result["opaque-1"].verdict == "violation"
        and result["opaque-2"].verdict == "pass"
    )
    assert len(judge.requests) == 1


@pytest.mark.asyncio
async def test_input_overflow_releases_unused_capacity_without_judging():
    from time import monotonic

    from tldw_chatbook.Chat.response_rules.resources import RuleHelperPool

    pool = RuleHelperPool(
        usage_sink=lambda *_args: None, current=lambda _source: True, clock=monotonic
    )
    lease = pool.try_acquire(source(), purpose="checking", deadline=monotonic() + 30)
    judge = ScriptedJudge({})
    result = await assess(judge, inputs("x" * 70000), lease=lease)
    assert result.outcome == "couldnt_verify" and judge.requests == []
    assert pool.unsettled_count == 0

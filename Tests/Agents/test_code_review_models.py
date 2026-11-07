"""Review wire boundaries refuse malformed data without changing caller records."""

from copy import deepcopy

import pytest
from pydantic import TypeAdapter, ValidationError

from Tests.Agents.code_review_helpers import coverage_payload, finding_payload
from tldw_chatbook.Agents.agent_models import RunBudget
from tldw_chatbook.Agents.code_review_models import (
    CheckReceipt,
    DispositionRecord,
    FindingRecord,
    ReviewDestinationOverride,
    ReviewLimits,
    ReviewRefused,
    ReviewRunBinding,
    ReviewRunChoice,
    ReviewTarget,
    SourceQuery,
    SourceRange,
    canonical_record_bytes,
    minimum_continuations,
    parse_review_team,
    review_result_char_limit,
    validate_record_chunk,
)


def test_complete_finding_round_trips_without_mutation():
    payload = finding_payload()
    original = deepcopy(payload)
    (record,) = validate_record_chunk("findings", [payload])
    assert isinstance(record, FindingRecord)
    assert record.model_dump(mode="json") == original
    assert payload == original
    assert isinstance(record.locations, tuple)
    with pytest.raises(ValidationError):
        record.title = "rewrite"


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("schema_version", 1.0),
        ("unknown", "value"),
        ("severity", "info"),
        ("locations", []),
    ],
)
def test_invalid_finding_refuses_entire_chunk(field, value):
    invalid = finding_payload()
    invalid[field] = value
    records = [finding_payload(), invalid]
    before = deepcopy(records)
    with pytest.raises(ReviewRefused):
        validate_record_chunk("findings", records)
    assert records == before


@pytest.mark.parametrize(
    "start,end", [(0, 1), (4, 3), (True, 4), (1, False), (1.0, 4), (1, "4")]
)
def test_source_ranges_reject_non_integer_or_reversed_lines(start, end):
    with pytest.raises(ValidationError):
        SourceRange(start_line=start, end_line=end)
    payload = finding_payload()
    payload["locations"][0].update(start_line=start, end_line=end)
    with pytest.raises(ReviewRefused):
        validate_record_chunk("findings", [payload])


def test_duplicate_local_keys_refuse():
    with pytest.raises(ReviewRefused):
        validate_record_chunk("findings", [finding_payload(), finding_payload()])


def test_record_limit_counts_utf8_and_accepts_exact_boundary():
    payload = finding_payload()
    overhead = len(canonical_record_bytes(payload))
    payload["uncertainty"] += "x" * (8192 - overhead)
    assert len(canonical_record_bytes(payload)) == 8192
    validate_record_chunk("findings", [payload])
    payload["uncertainty"] = payload["uncertainty"][:-1] + "é"
    with pytest.raises(ReviewRefused):
        validate_record_chunk("findings", [payload])


def test_chunk_limit_accepts_eight_and_refuses_nine():
    records = []
    for index in range(9):
        record = finding_payload()
        record["local_key"] = str(index)
        records.append(record)
    assert len(validate_record_chunk("findings", records[:8])) == 8
    with pytest.raises(ReviewRefused):
        validate_record_chunk("findings", records)


def test_canonical_encoding_is_order_independent_and_real_utf8():
    assert canonical_record_bytes({"z": "é", "a": 1}) == b'{"a":1,"z":"\xc3\xa9"}'


@pytest.mark.parametrize(
    "destination",
    [
        {"provider": "openai"},
        {"model": "chosen"},
        {"provider": "openai", "model": "chosen"},
    ],
)
def test_run_destination_does_not_change_saved_team(destination):
    saved = {
        "reviewers": [{"preset_id": "stable-id", "focus": "security"}],
        "verifier_preset_id": "",
    }
    before = deepcopy(saved)
    team = parse_review_team(saved)
    run_choice = ReviewRunChoice(
        **team.reviewers[0].model_dump(), destination=destination
    )
    assert run_choice.destination.model_dump(exclude_none=True) == destination
    assert saved == before
    assert team.verifier_preset_id is None
    assert "destination" not in team.reviewers[0].model_dump()


@pytest.mark.parametrize(
    "destination",
    [{}, {"provider": ""}, {"model": " "}, {"provider": "a", "url": "secret"}],
)
def test_invalid_destination_refuses(destination):
    with pytest.raises(ValidationError):
        ReviewDestinationOverride(**destination)


def test_saved_choices_refuse_run_overrides_and_unsafe_or_oversized_focus():
    for choice in (
        {"preset_id": "x", "destination": {"model": "x"}},
        {"preset_id": "x", "focus": "x" * 10001},
        {"preset_id": "x", "focus": "<script>alert(1)</script>"},
    ):
        with pytest.raises(ReviewRefused):
            parse_review_team({"reviewers": [choice]})


def test_saved_empty_team_is_valid_but_launch_arithmetic_requires_two():
    assert parse_review_team({}).reviewers == ()
    for reviewers in (0, 1):
        with pytest.raises(ReviewRefused):
            minimum_continuations(reviewers, False, 2)


def test_default_child_arithmetic_does_not_change_run_budget():
    assert RunBudget().max_subagents == 2
    assert minimum_continuations(2, False, 2) == 0
    assert minimum_continuations(2, True, 2) == 1
    assert minimum_continuations(3, False, 2) == 1
    assert minimum_continuations(3, True, 2) == 1
    assert minimum_continuations(2, True, 1) == 1


@pytest.mark.parametrize(
    "args",
    [(True, False, 2), (2.0, False, 2), (2, 1, 2), (2, False, -1), (2, False, True)],
)
def test_arithmetic_rejects_malformed_counts(args):
    with pytest.raises(ReviewRefused):
        minimum_continuations(*args)


@pytest.mark.parametrize(
    "configured,expected",
    [
        (1024, 1024),
        (16000, 16000),
        (20000, 16000),
        (0, 16000),
        (-1, 16000),
    ],
)
def test_review_result_limit_retains_native_ceiling(configured, expected):
    assert review_result_char_limit(configured) == expected


@pytest.mark.parametrize(
    "query",
    [
        {"op": "list", "relative_dir": ".", "side": "base"},
        {"op": "search", "text": "parse", "relative_dir": "src", "side": "head"},
        {"op": "resolve", "relative_path": "src/main.py", "side": "worktree"},
    ],
)
def test_source_query_closed_union_accepts_captured_sides(query):
    assert TypeAdapter(SourceQuery).validate_python(query).model_dump() == query


@pytest.mark.parametrize(
    "query",
    [
        {"op": "shell", "side": "head"},
        {"op": "list", "relative_dir": ".", "side": "HEAD~2"},
        {"op": "list", "relative_dir": "../private", "side": "base"},
        {"op": "resolve", "relative_path": "/etc/passwd", "side": "base"},
        {"op": "list", "relative_dir": ".", "side": "base", "revision": "x"},
    ],
)
def test_source_query_refuses_arbitrary_operations_paths_and_revisions(query):
    with pytest.raises(ValidationError):
        TypeAdapter(SourceQuery).validate_python(query)


def test_target_union_refuses_unused_kind_fields_and_non_exact_revisions():
    adapter = TypeAdapter(ReviewTarget)
    common = {"binding_ref": "binding-1", "scope": ".", "context_paths": []}
    working = adapter.validate_python(
        {
            **common,
            "kind": "working_tree",
            "staged": True,
            "unstaged": False,
            "untracked_paths": [],
        }
    )
    assert working.context_paths == ()
    for extra in ({"base_commit": "a" * 40}, {"staged": 1}):
        with pytest.raises(ValidationError):
            adapter.validate_python({**working.model_dump(), **extra})
    for revision in ("HEAD", "a" * 39):
        with pytest.raises(ValidationError):
            adapter.validate_python(
                {
                    **common,
                    "kind": "commit_range",
                    "base_commit": revision,
                    "head_commit": "b" * 40,
                    "comparison": "direct",
                }
            )


def test_submission_kinds_validate_completion_and_actions_explicitly():
    assert (
        validate_record_chunk("coverage", [coverage_payload()])[0].empty_findings
        is False
    )
    assert len(validate_record_chunk("reviewer_complete", [coverage_payload()])) == 1
    assert validate_record_chunk("verifier_complete", []) == ()
    action = {"schema_version": 1, "proposal_id": "p1", "action": "claim"}
    assert validate_record_chunk("check_actions", [action])[0].action == "claim"
    for kind, records in (
        ("done", []),
        ("verifier_complete", [action]),
        ("findings", []),
        ("check_actions", [{**action, "action": "execute"}]),
        ("check_actions", [{**action, "command": "rm"}]),
    ):
        with pytest.raises(ReviewRefused):
            validate_record_chunk(kind, records)


def test_check_receipt_requires_native_receipt_only_for_execution():
    base = {
        "schema_version": 1,
        "receipt_id": "r",
        "proposal_id": "p",
        "review_id": "review",
        "verifier_attempt_id": "a",
        "target_digest": "a" * 64,
        "outcome": "declined",
        "evidence": "Not authorized",
    }
    assert CheckReceipt(**base).native_execution_receipt_id is None
    with pytest.raises(ValidationError):
        CheckReceipt(**{**base, "outcome": "executed"})
    with pytest.raises(ValidationError):
        CheckReceipt(**base, native_execution_receipt_id="native")
    assert (
        CheckReceipt(
            **{**base, "outcome": "executed"}, native_execution_receipt_id="native"
        ).outcome
        == "executed"
    )


def test_binding_feedback_mode_is_verifier_only_and_limits_are_frozen():
    base = {
        "review_id": "r",
        "slot_id": "s",
        "attempt_id": "a",
        "role": "reviewer",
        "target_digest": "a" * 64,
        "source_epoch": 0,
    }
    with pytest.raises(ValidationError):
        ReviewRunBinding(**base, check_feedback_mode="primary_handoff")
    assert (
        ReviewRunBinding(
            **{**base, "role": "verifier"}, check_feedback_mode="unavailable_inline"
        ).role
        == "verifier"
    )
    limits = ReviewLimits()
    assert limits.source_max_bytes == 64 * 1024 * 1024
    with pytest.raises((ValidationError, AttributeError)):
        limits.source_max_bytes = 1


def test_check_submission_accepts_only_one_proposal_or_action():
    proposal = {
        "schema_version": 1,
        "argv": ["python", "-V"],
        "intent": "Check runtime",
        "verification_unit_ids": ["unit-1"],
    }
    action = {"schema_version": 1, "proposal_id": "proposal-1", "action": "decline"}
    assert validate_record_chunk("check_proposals", [proposal])[0].argv == (
        "python",
        "-V",
    )
    for kind, record in (("check_proposals", proposal), ("check_actions", action)):
        with pytest.raises(ReviewRefused):
            validate_record_chunk(kind, [record, record])


@pytest.mark.parametrize(
    "field,value",
    [
        ("argv", []),
        ("argv", "python -V"),
        ("verification_unit_ids", []),
        ("intent", ""),
    ],
)
def test_invalid_check_proposals_refuse(field, value):
    proposal = {
        "schema_version": 1,
        "argv": ["python", "-V"],
        "intent": "Check runtime",
        "verification_unit_ids": ["unit-1"],
        field: value,
    }
    with pytest.raises(ReviewRefused):
        validate_record_chunk("check_proposals", [proposal])


def test_dispositions_use_opaque_units_and_captured_or_check_evidence():
    disposition = {
        "schema_version": 1,
        "verification_unit_id": "unit-1",
        "verdict": "unresolved",
        "reason": "Needs runtime check",
        "evidence": [{"kind": "check", "receipt_id": "receipt-1"}],
        "next_check": "Execute the proposed check",
    }
    record = validate_record_chunk("dispositions", [disposition])[0]
    assert record.evidence[0].receipt_id == "receipt-1"
    for change in (
        {"verdict": "agreed"},
        {"original_id": "R1-F1"},
        {
            "evidence": [
                {
                    "kind": "source",
                    "source_ref": "src-1",
                    "start_line": 1,
                    "end_line": 1,
                    "digest": "bad",
                }
            ]
        },
        {"evidence": [{"kind": "path", "path": "/tmp/check"}]},
    ):
        with pytest.raises(ReviewRefused):
            validate_record_chunk("dispositions", [{**disposition, **change}])


def test_coverage_and_completion_refuse_success_status_claims():
    for change in (
        {"status": "done"},
        {"empty_findings": 1},
        {"gaps": [{"reason": "unavailable", "host_path": "/secret"}]},
    ):
        with pytest.raises(ReviewRefused):
            validate_record_chunk(
                "reviewer_complete", [{**coverage_payload(), **change}]
            )


def test_non_json_or_non_record_chunks_refuse_stably():
    for records in (
        (finding_payload(),),
        [None],
        [{"bytes": b"binary"}],
        [{"nan": float("nan")}],
    ):
        with pytest.raises(ReviewRefused):
            validate_record_chunk("findings", records)


def test_valid_exact_targets_round_trip_and_verifier_focus_is_empty():
    from tldw_chatbook.Agents.code_review_models import ReviewVerifierChoice

    adapter = TypeAdapter(ReviewTarget)
    for target in (
        {
            "kind": "pr",
            "binding_ref": "b",
            "pr_ref": "pr-1",
            "base_commit": "a" * 40,
            "head_commit": "b" * 40,
            "comparison": "merge_base",
        },
        {
            "kind": "commit_range",
            "binding_ref": "b",
            "base_commit": "a" * 40,
            "head_commit": "b" * 40,
            "comparison": "direct",
        },
        {
            "kind": "patch",
            "binding_ref": "b",
            "patch_ref": "patch-1",
            "preimage_commit": "a" * 40,
        },
    ):
        assert adapter.validate_python(target).kind == target["kind"]
    assert ReviewVerifierChoice(preset_id="v").focus == ""
    with pytest.raises(ValidationError):
        ReviewVerifierChoice(preset_id="v", focus="bias")


def test_internal_limits_reject_malformed_caps():
    for value in (0, -1, True, 1.0):
        with pytest.raises(ValueError):
            ReviewLimits(source_max_bytes=value)


def test_saved_choices_cannot_smuggle_a_run_choice_instance():
    run = ReviewRunChoice(preset_id="x", destination={"model": "override"})
    with pytest.raises(ReviewRefused):
        parse_review_team({"reviewers": [run]})


def test_patch_preimage_may_be_captured_from_filesystem_without_git_revision():
    target = TypeAdapter(ReviewTarget).validate_python(
        {
            "kind": "patch",
            "binding_ref": "ssh-filesystem-binding",
            "patch_ref": "patch-1",
        }
    )
    assert target.preimage_commit is None


def test_fixture_limits_may_narrow_but_cannot_raise_production_caps():
    assert ReviewLimits(source_max_bytes=1024).source_max_bytes == 1024
    with pytest.raises(ValueError):
        ReviewLimits(source_max_bytes=64 * 1024 * 1024 + 1)


@pytest.mark.parametrize(
    "next_check", [{}, {"next_check": None}, {"next_check": ""}, {"next_check": " "}]
)
def test_unresolved_dispositions_require_a_nonempty_settling_check(next_check):
    payload = {
        "schema_version": 1,
        "verification_unit_id": "unit-1",
        "verdict": "unresolved",
        "reason": "Needs runtime check",
        "evidence": [],
        **next_check,
    }
    with pytest.raises(ValidationError):
        DispositionRecord(**payload)
    with pytest.raises(ReviewRefused):
        validate_record_chunk("dispositions", [payload])


@pytest.mark.parametrize("verdict", ["confirmed", "refuted"])
def test_settled_dispositions_may_omit_the_next_check(verdict):
    payload = {
        "schema_version": 1,
        "verification_unit_id": "unit-1",
        "verdict": verdict,
        "reason": "Captured source settles this claim",
        "evidence": [finding_payload()["evidence"][0]],
    }
    assert validate_record_chunk("dispositions", [payload])[0].next_check is None

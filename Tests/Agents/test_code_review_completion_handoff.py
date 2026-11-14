"""Coordinator readback survives verifier history reduction and false raw prose."""

import json
import re

import pytest

from Tests.Agents import code_review_helpers
from Tests.Agents.test_code_review_reports import finding
from Tests.Agents.test_code_review_verification import seed
from tldw_chatbook.Agents.code_review_history import project_verifier_history
from tldw_chatbook.Agents.run_context import CurrentRunActor, use_run_actor

review_case = code_review_helpers.review_case
source_case = code_review_helpers.source_case
pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.parametrize("stage", ["review", "verify"])
def test_primary_recipe_examples_match_advertised_role(review_case, stage):
    c = review_case
    with use_run_actor(c.actor):
        guide = c.primary.guide(stage)["text"]
        examples = list(
            re.finditer(r"(code_review_read|code_review_submit)\(\s*(\{)", guide)
        )
        assert examples
        for example in examples:
            name = example[1]
            args, _ = json.JSONDecoder().raw_decode(guide[example.start(2) :])
            schema = c.provider.load_schema("code_review:" + name).parameters
            field = "view" if name == "code_review_read" else "kind"
            assert args[field] in schema["properties"][field]["enum"], (name, args)


def test_primary_report_marks_pending_verification_unresolved(review_case):
    c = review_case
    prepared = seed(c, 1)
    with c.participant(prepared, role="verifier", slot="V1") as provider:
        service = provider.service
        service.verifier_input_tokens = 10000
        service.verifier_estimate = lambda text: len(text) // 4
        service.read(prepared.review_id, "batch")
        with use_run_actor(c.actor):
            page = c.primary.read(prepared.review_id, "report")
        assert page.records[0]["disposition"]["verdict"] == "unresolved"
        assert page.records[0]["disposition"]["reason"] == "verification_incomplete"
        assert page.records[0]["verifier_report_status"] == "pending"


def test_checkpoint_does_not_attribute_host_capacity_gaps_to_verifier(review_case):
    c = review_case
    prepared = c.prepare()
    with c.participant(prepared) as provider:
        provider.service.submit(
            prepared.review_id,
            "findings",
            [
                finding(c, prepared, "small"),
                finding(c, prepared, "large", impact="long impact " * 300),
            ],
            0,
            "findings",
        )
    with c.participant(prepared, role="verifier", slot="V1") as provider:
        service = provider.service
        page = service.next_verification_batch(
            prepared.review_id,
            remaining_input_tokens=10000,
            estimate=lambda text: len(text) // 4,
            max_result_chars=1400,
            cursor=None,
        )
        assert len(page.records) == 1
        unit = page.records[0]
        service.submit(
            prepared.review_id,
            "dispositions",
            [
                {
                    "schema_version": 1,
                    "verification_unit_id": unit["verification_unit_id"],
                    "verdict": "confirmed",
                    "reason": "captured evidence",
                    "evidence": unit["claim"]["evidence"],
                }
            ],
            0,
            "decisions",
        )
        if page.cursor:
            remainder = service.next_verification_batch(
                prepared.review_id,
                remaining_input_tokens=10000,
                estimate=lambda text: len(text) // 4,
                max_result_chars=1400,
                cursor=page.cursor,
            )
            assert not remainder.records and remainder.cursor is None
        checkpoint = service.verifier_checkpoint(prepared.review_id)
        assert checkpoint.disposition_counts == {
            "confirmed": 1,
            "refuted": 0,
            "unresolved": 0,
        }
    with use_run_actor(c.actor):
        dispositions = c.primary.expand_dispositions(prepared.review_id)
        assert sorted(row["disposition"]["verdict"] for row in dispositions) == [
            "confirmed",
            "unresolved",
        ]


def test_reviewer_retry_after_verification_keeps_new_original_explicitly_unverified(
    review_case,
):
    c = review_case
    prepared = seed(c, 1)
    with c.store.lease(prepared.review_id) as context:
        old = context.active_attempts["R1"]
    c.service(CurrentRunActor("subagent", "reviewerR1", "primary")).settle_review_run(
        "reviewerR1", terminal_status="done", receipt_id="reviewer-terminal"
    )
    with c.participant(prepared, role="verifier", slot="V1") as provider:
        service = provider.service
        service.verifier_input_tokens = 10000
        service.verifier_estimate = lambda text: len(text) // 4
        unit = service.read(prepared.review_id, "batch").records[0]
        decision = {
            "schema_version": 1,
            "verification_unit_id": unit["verification_unit_id"],
            "verdict": "confirmed",
            "reason": "captured evidence",
            "evidence": unit["claim"]["evidence"],
        }
        service.submit(prepared.review_id, "dispositions", [decision], 0, "decisions")
        service.submit(prepared.review_id, "verifier_complete", [], 1, "complete")
        service.capture_review_output(service.actor.run_id, "done", terminal=True)
        service.settle_review_run(
            service.actor.run_id, terminal_status="done", receipt_id="terminal"
        )
    actor = CurrentRunActor("subagent", "reviewer-retry", "primary")
    c.store.bind_attempt(actor.run_id, old.model_copy(update={"attempt_id": "retry"}))
    with use_run_actor(actor):
        c.service(actor).submit(
            prepared.review_id,
            "findings",
            [finding(c, prepared, "retry-new", trigger="new finding after retry")],
            0,
            "retry-findings",
        )
    with use_run_actor(c.actor):
        rows = c.primary.read(prepared.review_id, "report").records
    assert [row["original_id"] for row in rows] == ["R1-F1", "R1-F2"]
    assert (
        rows[0]["disposition"]["verification_unit_id"]
        == decision["verification_unit_id"]
    )
    assert rows[0]["disposition"]["verdict"] == "confirmed"
    assert rows[1]["gap"] == "original_not_in_verification"
    assert "disposition" not in rows[1]


@pytest.mark.parametrize("native", [True, False])
def test_own_verdict_counts_survive_reduction_and_primary_reads_sealed_decisions(
    review_case, native
):
    c = review_case
    prepared = seed(c, 3)
    with c.participant(prepared, role="verifier", slot="V1") as provider:
        service = provider.service
        service.verifier_input_tokens = 10000
        service.verifier_estimate = lambda text: len(text) // 4
        page = service.read(prepared.review_id, "batch")
        with use_run_actor(c.actor):
            c.primary.result_limit = 2000
            old_cursor = c.primary.read(prepared.review_id, "report").cursor
            assert old_cursor
            old_page = c.primary.read(prepared.review_id, "report", cursor=old_cursor)
        records = [
            {
                "schema_version": 1,
                "verification_unit_id": row["verification_unit_id"],
                "verdict": verdict,
                "reason": "captured evidence"
                if verdict != "unresolved"
                else "missing contract",
                "evidence": row["claim"]["evidence"] if verdict != "unresolved" else [],
                **(
                    {"next_check": "check the documented contract"}
                    if verdict == "unresolved"
                    else {}
                ),
            }
            for row, verdict in zip(
                page.records, ("confirmed", "refuted", "unresolved")
            )
        ]
        accepted = service.submit(
            prepared.review_id, "dispositions", records, 0, "decisions"
        )
        checkpoint = service.verifier_checkpoint(prepared.review_id)
        assert checkpoint.disposition_counts == {
            "confirmed": 1,
            "refuted": 1,
            "unresolved": 1,
        }
        payload = service.encode(accepted)
        reduced = project_verifier_history(
            [
                {"role": "user", "content": "Verify the claims"},
                {"role": "assistant", "content": "submit decisions"},
                {
                    "role": "tool" if native else "user",
                    "content": payload
                    if native
                    else "Tool result for code_review_submit: " + payload,
                },
            ],
            checkpoint,
            continuation_groups=(),
            native=native,
        )
        note = reduced[-1]["content"]
        state, _ = json.JSONDecoder().raw_decode(note.partition(": ")[2])
        assert state["disposition_counts"] == {
            "confirmed": 1,
            "refuted": 1,
            "unresolved": 1,
        }
        assert state["attempt_id"] == checkpoint.attempt_id
        assert state["target_digest"] == checkpoint.target_digest
        service.submit(prepared.review_id, "verifier_complete", [], 1, "complete")
        service.capture_review_output(
            service.actor.run_id, "I submitted no decisions", terminal=True
        )
        service.settle_review_run(
            service.actor.run_id, terminal_status="done", receipt_id="terminal"
        )

    with use_run_actor(c.actor):
        assert (
            c.primary.read(prepared.review_id, "report", cursor=old_cursor) == old_page
        )
        expected = {
            row["original_id"]: row["disposition"]
            for row in c.primary.expand_dispositions(prepared.review_id)
        }
        c.primary.result_limit = 2000  # Exercise complete-record pagination.
        seen, cursor = {}, None
        while True:
            page = c.primary.read(prepared.review_id, "report", cursor=cursor)
            for row in page.records:
                assert row["original_id"] not in seen
                assert row["verifier_report_status"] == "sealed"
                seen[row["original_id"]] = row["disposition"]
            cursor = page.cursor
            if cursor is None:
                break
        assert seen == expected

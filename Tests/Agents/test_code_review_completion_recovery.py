"""Offline recovery through the actual participant tool and history boundaries."""

import json

import pytest

from Tests.Agents import code_review_helpers
from Tests.Agents.test_code_review_reports import coverage, summary
from Tests.Agents.test_code_review_verification import seed
from tldw_chatbook.Agents.code_review_history import project_verifier_history
from tldw_chatbook.Agents.run_context import use_run_actor

review_case = code_review_helpers.review_case
source_case = code_review_helpers.source_case
pytestmark = pytest.mark.bootstrap_profile


def invoke(provider, name, review_id, **args):
    return provider.invoke(
        "code_review:code_review_" + name, {"review_id": review_id, **args}
    )


def test_coverage_feedback_and_same_id_repair_seal(review_case):
    c = review_case
    prepared = c.prepare()
    with c.participant(prepared) as provider:
        valid = coverage(c, prepared)
        bad = dict(
            valid,
            inspected_refs=[{"source_ref": valid["inspected_refs"][0]}],
            notes="PRIVATE-BODY",
        )
        refused = invoke(
            provider,
            "submit",
            prepared.review_id,
            kind="coverage",
            records=[bad],
            expected_revision=0,
            submission_id="coverage-repair",
        )
        assert not refused.ok and refused.error == "invalid_record"
        feedback = json.loads(refused.content)
        assert feedback["submission_id"] == "coverage-repair"
        assert feedback["recovery"] == "correct_same_submission_id"
        assert {tuple(item["field"]) for item in feedback["validation_errors"]} == {
            ("records", 0, "inspected_refs", 0),
            ("records", 0, "notes"),
        }
        assert "PRIVATE-BODY" not in refused.content
        assert (
            "same submission_id"
            in provider.load_schema("code_review:code_review_submit").description
        )
        # Correcting under another identity cannot retire the original rejection.
        replacement = invoke(
            provider,
            "submit",
            prepared.review_id,
            kind="coverage",
            records=[valid],
            expected_revision=0,
            submission_id="replacement",
        )
        assert replacement.ok
        blocked = invoke(
            provider,
            "submit",
            prepared.review_id,
            kind="reviewer_complete",
            records=[valid],
            expected_revision=1,
            submission_id="complete",
        )
        assert not blocked.ok and blocked.error == "unresolved_submission_rejections"
        assert (
            json.loads(blocked.content)["recovery"]
            == "repair_rejected_submissions_then_retry_same_completion_id"
        )
        with c.store.lease(prepared.review_id) as context:
            stream = context.streams[("R1", "attempt")]
            assert stream.revision == 1 and not stream.completion_requested
        repaired = invoke(
            provider,
            "submit",
            prepared.review_id,
            kind="coverage",
            records=[valid],
            expected_revision=1,
            submission_id="coverage-repair",
        )
        assert repaired.ok
        completed = invoke(
            provider,
            "submit",
            prepared.review_id,
            kind="reviewer_complete",
            records=[valid],
            expected_revision=2,
            submission_id="complete",
        )
        assert completed.ok
        service = provider.service
        service.capture_review_output(
            service.actor.run_id, "offline recovered review", terminal=True
        )
        service.settle_review_run(
            service.actor.run_id, terminal_status="done", receipt_id="offline-terminal"
        )
    slot = summary(c, prepared)["slots"][0]
    assert slot["report_status"] == "sealed" and not slot["rejected_submissions"]


@pytest.mark.parametrize("native", [True, False])
def test_cached_batch_after_history_reduction_recovers_live_checkpoint_and_seals(
    review_case, native
):
    c = review_case
    prepared = seed(c, 2)
    with c.participant(prepared, role="verifier", slot="V1") as provider:
        service = provider.service
        service.verifier_input_tokens = 10000
        service.verifier_estimate = lambda text: len(text) // 4
        first = invoke(provider, "read", prepared.review_id, view="batch")
        assert first.ok
        page = json.loads(first.content)
        records = [
            {
                "schema_version": 1,
                "verification_unit_id": row["verification_unit_id"],
                "verdict": "confirmed",
                "reason": "captured evidence",
                "evidence": row["claim"]["evidence"],
            }
            for row in page["records"]
        ]
        accepted = invoke(
            provider,
            "submit",
            prepared.review_id,
            kind="dispositions",
            records=records,
            expected_revision=0,
            submission_id="decisions",
        )
        assert accepted.ok
        checkpoint = service.verifier_checkpoint(prepared.review_id)
        call = {"role": "assistant", "content": "submit decisions"}
        feedback = {
            "role": "tool" if native else "user",
            "content": accepted.content
            if native
            else "Tool result for code_review_submit: " + accepted.content,
        }
        history = [
            {"role": "user", "content": "original task"},
            call,
            feedback,
            {"role": "assistant", "content": "continue"},
        ]
        reduced = project_verifier_history(
            history, checkpoint, continuation_groups=(), native=native
        )
        assert call not in reduced and feedback not in reduced
        assert any('view="checkpoint"' in row.get("content", "") for row in reduced)
        # Replay is immutable even though the durable protocol advanced.
        replay = invoke(provider, "read", prepared.review_id, view="batch")
        assert replay.ok and replay.content == first.content
        current = invoke(provider, "read", prepared.review_id, view="checkpoint")
        assert current.ok, current.error
        state = json.loads(current.content)
        assert state["submission_revision"] == 1
        assert state["records"][0]["pending_unit_ids"] == []
        assert state["records"][0]["batch_cursor"] is None
        assert state["records"][0]["all_units_disposed"] is True
        completed = invoke(
            provider,
            "submit",
            prepared.review_id,
            kind="verifier_complete",
            records=[],
            expected_revision=state["submission_revision"],
            submission_id="complete",
        )
        assert completed.ok
        with c.store.lease(prepared.review_id) as context:
            stream = context.streams[("V1", "attempt")]
            assert stream.revision == 2 and len(stream.verification.dispositions) == 2
        service.capture_review_output(
            service.actor.run_id, "offline recovered verification", terminal=True
        )
        service.settle_review_run(
            service.actor.run_id, terminal_status="done", receipt_id="offline-terminal"
        )
    with use_run_actor(c.actor):
        slots = [
            slot
            for row in c.primary.read(prepared.review_id, "summary").records
            for slot in row["slots"]
        ]
        assert (
            next(slot for slot in slots if slot["slot_id"] == "V1")["report_status"]
            == "sealed"
        )
        expanded = c.primary.expand_dispositions(prepared.review_id)
    assert len(expanded) == len({row["original_id"] for row in expanded}) == 2
    assert all(row["disposition"]["verdict"] == "confirmed" for row in expanded)


from Tests.Agents import test_code_review_dispatch

dispatch_case = test_code_review_dispatch.dispatch_case


@pytest.mark.parametrize("native", [True, False])
def test_rejection_recovery_feedback_reaches_actual_child_request(
    dispatch_case, native
):
    c = dispatch_case

    def malformed():
        valid = coverage(c, c.prepared)
        return c.reply(
            "code_review_submit",
            {
                "review_id": c.prepared.review_id,
                "kind": "coverage",
                "records": [
                    dict(
                        valid,
                        inspected_refs=[{"source_ref": valid["inspected_refs"][0]}],
                        notes="PRIVATE-BODY",
                    )
                ],
                "expected_revision": 0,
                "submission_id": "coverage-repair",
            },
        )

    c.run(
        native=native,
        inline=True,
        child_script=[
            lambda: c.reply("load_tools", {"ids": ["code_review:code_review_submit"]}),
            malformed,
            "Retain the rejection",
        ],
    )
    child = next(
        request for actor, request in reversed(c.calls) if actor.kind == "subagent"
    )
    feedback = [
        row["content"]
        for row in child["messages_payload"]
        if "invalid_record" in str(row.get("content", ""))
    ]
    assert feedback
    assert any(
        '"submission_id":"coverage-repair"' in content
        and '"recovery":"correct_same_submission_id"' in content
        for content in feedback
    )
    assert all("PRIVATE-BODY" not in content for content in feedback)


@pytest.mark.parametrize("role", ["primary", "reviewer"])
def test_checkpoint_read_does_not_widen_other_roles(review_case, role):
    from contextlib import nullcontext

    from tldw_chatbook.Agents.code_review_models import ReviewRefused

    c = review_case
    prepared = c.prepare()
    participant = (
        nullcontext(c.provider) if role == "primary" else c.participant(prepared)
    )
    with participant as provider, use_run_actor(provider.service.actor):
        assert (
            "checkpoint"
            not in provider.load_schema("code_review:code_review_read").parameters[
                "properties"
            ]["view"]["enum"]
        )
        with pytest.raises(ReviewRefused, match="role_denied"):
            provider.service.read(prepared.review_id, view="checkpoint")


def test_checkpoint_read_is_current_bounded_and_selection_free(review_case):
    from dataclasses import replace

    c = review_case
    prepared = seed(c, 2)
    with c.participant(prepared, role="verifier", slot="V1") as provider:
        current = invoke(provider, "read", prepared.review_id, view="checkpoint")
        assert current.ok
        assert json.loads(current.content)["records"][0]["all_units_disposed"] is False
        for selection in ({"document_ref": "unrelated"}, {"cursor": "unrelated"}):
            refused = invoke(
                provider, "read", prepared.review_id, view="checkpoint", **selection
            )
            assert not refused.ok and refused.error == "invalid_arguments"
        provider.service.result_limit = 50
        refused = invoke(provider, "read", prepared.review_id, view="checkpoint")
        assert (
            not refused.ok and refused.error == "review_response_capacity_unavailable"
        )
        provider.service.result_limit = 16000
        c.capacity = replace(c.capacity, remaining_input_tokens=1)
        refused = invoke(provider, "read", prepared.review_id, view="checkpoint")
        assert (
            not refused.ok and refused.error == "review_response_capacity_unavailable"
        )


def test_live_checkpoint_distinguishes_acknowledged_batch_from_all_units(review_case):
    c = review_case
    prepared = seed(c, 3)
    with c.participant(prepared, role="verifier", slot="V1", limit=1024) as provider:
        service = provider.service
        service.verifier_input_tokens = 10000
        service.verifier_estimate = lambda text: len(text) // 4
        first = invoke(provider, "read", prepared.review_id, view="batch")
        assert first.ok
        response, revision, seen = first, 0, set()
        while True:
            page = json.loads(response.content)
            ids = {row["verification_unit_id"] for row in page["records"]}
            assert ids and not seen.intersection(ids)
            seen.update(ids)
            records = [
                {
                    "schema_version": 1,
                    "verification_unit_id": row["verification_unit_id"],
                    "verdict": "confirmed",
                    "reason": "captured evidence",
                    "evidence": row["claim"]["evidence"],
                }
                for row in page["records"]
            ]
            accepted = invoke(
                provider,
                "submit",
                prepared.review_id,
                kind="dispositions",
                records=records,
                expected_revision=revision,
                submission_id=f"batch-{revision}",
            )
            assert accepted.ok
            revision += 1
            current = invoke(provider, "read", prepared.review_id, view="checkpoint")
            assert current.ok
            state = json.loads(current.content)
            assert state["submission_revision"] == revision
            assert state["records"][0]["pending_unit_ids"] == []
            assert state["records"][0]["batch_cursor"] == page["cursor"]
            assert state["records"][0]["all_units_disposed"] == (len(seen) == 3)
            if page["cursor"] is None:
                break
            response = invoke(
                provider,
                "read",
                prepared.review_id,
                view="batch",
                cursor=state["records"][0]["batch_cursor"],
            )
            assert response.ok
            pending = invoke(provider, "read", prepared.review_id, view="checkpoint")
            assert pending.ok
            assert set(
                json.loads(pending.content)["records"][0]["pending_unit_ids"]
            ) == {
                row["verification_unit_id"]
                for row in json.loads(response.content)["records"]
            }
        assert revision > 1 and len(seen) == 3
        assert (
            invoke(provider, "read", prepared.review_id, view="batch").content
            == first.content
        )
        assert invoke(
            provider,
            "submit",
            prepared.review_id,
            kind="verifier_complete",
            records=[],
            expected_revision=revision,
            submission_id="complete",
        ).ok
        with c.store.lease(prepared.review_id) as context:
            assert (
                len(context.streams[("V1", "attempt")].verification.dispositions) == 3
            )

"""Reviewer completion validates accepted findings before acknowledging intent."""

import json

import pytest

from Tests.Agents import code_review_helpers, test_code_review_dispatch
from Tests.Agents.test_code_review_completion_recovery import invoke
from Tests.Agents.test_code_review_reports import coverage, finding, summary

review_case = code_review_helpers.review_case
source_case = code_review_helpers.source_case
dispatch_case = test_code_review_dispatch.dispatch_case
pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.parametrize("has_findings", [True, False])
def test_inconsistent_completion_refuses_without_publishing_and_same_id_repairs(
    review_case, has_findings
):
    c = review_case
    prepared = c.prepare()
    with c.participant(prepared) as provider:
        service = provider.service
        revision = 0
        if has_findings:
            accepted = service.submit(
                prepared.review_id, "findings", [finding(c, prepared)], 0, "finding"
            )
            revision = accepted.submission_revision
        bad = coverage(c, prepared, empty=has_findings)
        # Intermediate coverage may precede findings; completion must be consistent.
        service.submit(prepared.review_id, "coverage", [bad], revision, "coverage")
        revision += 1
        with c.store.lease(prepared.review_id) as context:
            stream = context.streams[("R1", "attempt")]
            original_index = dict(context.original_index)
            previous_coverage = stream.coverage
            previous_manifest = context.manifest_path.read_bytes()
        refused = invoke(
            provider,
            "submit",
            prepared.review_id,
            kind="reviewer_complete",
            records=[bad],
            expected_revision=revision,
            submission_id="complete",
        )
        assert not refused.ok and refused.error == "invalid_record"
        feedback = json.loads(refused.content)
        assert feedback["submission_id"] == "complete"
        assert feedback["recovery"] == "correct_same_submission_id"
        assert feedback["validation_errors"] == [
            {
                "field": ["records", 0, "empty_findings"],
                "type": "findings_mismatch",
                "expected": not has_findings,
            }
        ]
        with c.store.lease(prepared.review_id) as context:
            assert stream.revision == revision and not stream.completion_requested
            assert stream.coverage == previous_coverage
            assert context.original_index == original_index
            assert context.manifest_path.read_bytes() == previous_manifest
            assert "complete" not in stream.submissions
        repaired_args = {
            "kind": "reviewer_complete",
            "records": [coverage(c, prepared, empty=not has_findings)],
            "expected_revision": revision,
            "submission_id": "complete",
        }
        repaired = invoke(provider, "submit", prepared.review_id, **repaired_args)
        assert repaired.ok
        assert json.loads(repaired.content)["submission_revision"] == revision + 1
        assert (
            invoke(provider, "submit", prepared.review_id, **repaired_args) == repaired
        )
        service.capture_review_output(service.actor.run_id, "Original", terminal=False)
        assert summary(c, prepared)["slots"][0]["report_status"] == "pending"
        service.settle_review_run(
            service.actor.run_id, terminal_status="done", receipt_id="native-terminal"
        )
        assert summary(c, prepared)["slots"][0]["report_status"] == "incomplete"
        service.capture_review_output(service.actor.run_id, "", terminal=True)
        with c.store.lease(prepared.review_id) as context:
            assert context.original_index == original_index
    slot = summary(c, prepared)["slots"][0]
    assert slot["report_status"] == "sealed" and not slot["rejected_submissions"]


@pytest.mark.parametrize("native", [True, False])
def test_completion_feedback_reaches_child_and_corrected_completion_seals(
    dispatch_case, native
):
    c = dispatch_case

    def submit(kind, records, revision, submission_id):
        return c.reply(
            "code_review_submit",
            {
                "review_id": c.prepared.review_id,
                "kind": kind,
                "records": records,
                "expected_revision": revision,
                "submission_id": submission_id,
            },
        )

    def repair():
        actor, request = c.calls[-1]
        assert actor.kind == "subagent"
        feedback = "\n".join(
            str(row.get("content", "")) for row in request["messages_payload"]
        )
        assert '"type":"findings_mismatch"' in feedback
        assert '"expected":false' in feedback
        assert '"recovery":"correct_same_submission_id"' in feedback
        return submit(
            "reviewer_complete", [coverage(c, c.prepared, empty=False)], 1, "complete"
        )

    c.run(
        native=native,
        inline=True,
        child_script=[
            lambda: c.reply("load_tools", {"ids": ["code_review:code_review_submit"]}),
            lambda: submit("findings", [finding(c, c.prepared)], 0, "finding"),
            lambda: submit(
                "reviewer_complete", [coverage(c, c.prepared)], 1, "complete"
            ),
            repair,
            "Original review complete",
        ],
    )
    slot = summary(c, c.prepared)["slots"][0]
    assert slot["report_status"] == "sealed" and not slot["rejected_submissions"]
    assert slot["submission_revision"] == 2


def test_later_findings_preserve_completion_replay_but_cannot_use_stale_seal(
    review_case,
):
    c = review_case
    prepared = c.prepare()
    with c.participant(prepared) as provider:
        service = provider.service
        records = [coverage(c, prepared)]
        completed = service.submit(
            prepared.review_id, "reviewer_complete", records, 0, "complete"
        )
        service.submit(
            prepared.review_id, "findings", [finding(c, prepared)], 1, "finding"
        )
        assert (
            service.submit(
                prepared.review_id, "reviewer_complete", records, 0, "complete"
            )
            == completed
        )
        service.capture_review_output(service.actor.run_id, "Original", terminal=True)
        service.settle_review_run(
            service.actor.run_id, terminal_status="done", receipt_id="native-terminal"
        )
    slot = summary(c, prepared)["slots"][0]
    assert slot["report_status"] == "incomplete" and not slot["seal_receipt"]
    assert slot["submission_revision"] == 2 and not slot["rejected_submissions"]

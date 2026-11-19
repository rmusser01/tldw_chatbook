"""Only a live host-bound child may omit redundant review identity."""

import json
from dataclasses import replace

import pytest

from Tests.Agents import code_review_helpers, test_code_review_conflict_recovery
from Tests.Agents.test_code_review_reports import coverage, finding, summary
from tldw_chatbook.Agents.code_review_tool_provider import (
    ReadArguments,
    SubmitArguments,
)
from tldw_chatbook.Agents.run_context import use_run_actor

review_case = code_review_helpers.review_case
source_case = code_review_helpers.source_case
accepted_case = test_code_review_conflict_recovery.accepted_case
pytestmark = pytest.mark.bootstrap_profile


def submission(c, prepared, kind="coverage", revision=0):
    return {
        "kind": kind,
        "records": [coverage(c, prepared, empty=False)],
        "expected_revision": revision,
        "submission_id": kind,
    }


def test_omitted_identity_preserves_accepted_findings_and_native_sealing(review_case):
    c = review_case
    prepared = c.prepare()
    with c.participant(prepared) as provider:
        service = provider.service
        accepted = service.submit(
            prepared.review_id, "findings", [finding(c, prepared)], 0, "finding"
        )
        with c.store.lease(prepared.review_id) as context:
            originals = dict(context.original_index)
            stream = context.streams[("R1", "attempt")]
            original_submission = stream.submissions["finding"]
        # Mirrors the failed wire envelope: coverage omits only review_id.
        args = submission(c, prepared, revision=1)
        result = provider.invoke("code_review:code_review_submit", args)
        assert result.ok, result.error
        assert "review_id" not in args  # Do not rewrite the captured model call.
        assert json.loads(result.content)["submission_revision"] == 2
        assert provider.invoke("code_review:code_review_submit", args) == result
        assert summary(c, prepared)["slots"][0]["seal_receipt"] is None
        done = provider.invoke(
            "code_review:code_review_submit",
            submission(c, prepared, "reviewer_complete", 2),
        )
        assert done.ok, done.error
        assert stream.submissions["finding"] == original_submission
        assert context.original_index == originals
        assert (
            service.submit(
                prepared.review_id, "findings", [finding(c, prepared)], 0, "finding"
            )
            == accepted
        )
        run_id = service.actor.run_id
        service.settle_review_run(run_id, terminal_status="done", receipt_id="terminal")
        assert summary(c, prepared)["slots"][0]["seal_receipt"] is None
        service.capture_review_output(run_id, "", terminal=True)
    slot = summary(c, prepared)["slots"][0]
    assert slot["report_status"] == "sealed"
    assert slot["submission_revision"] == 3 and not slot["rejected_submissions"]


@pytest.mark.parametrize("role,slot", [("reviewer", "R1"), ("verifier", "V1")])
def test_bound_read_uses_same_identity_for_instruction_targets(review_case, role, slot):
    c = review_case
    prepared = c.prepare()
    with c.participant(prepared, role, slot) as provider:
        args = {
            "view": "source",
            "source_query": {
                "op": "resolve",
                "relative_path": "main.py",
                "side": "worktree",
            },
        }
        targets = provider.path_targets("code_review:code_review_read", args)
        assert len(targets) == 1 and targets[0].path == c.file
        result = provider.invoke("code_review:code_review_read", args)
        assert result.ok, result.error
        ref = json.loads(result.content)["records"][0]["source_ref"]
        page = provider.invoke(
            "code_review:code_review_read", {"view": "source", "document_ref": ref}
        )
        assert page.ok and json.loads(page.content)["text"] == c.file.read_text()
        # Host identity does not bypass source permissions.
        provider.service.source_permission = lambda *args: "permission_denied"
        denied = provider.invoke("code_review:code_review_read", args)
        assert denied.error == "permission_denied" and not denied.content


@pytest.mark.parametrize("role,slot", [("reviewer", "R1"), ("verifier", "V1")])
def test_only_child_schemas_allow_omission_without_relaxing_wire_models(
    review_case, role, slot
):
    c = review_case
    prepared = c.prepare()
    with c.participant(prepared, role, slot) as provider:
        for name in ("code_review_read", "code_review_submit"):
            child = provider.load_schema("code_review:" + name).parameters
            assert "review_id" not in child["required"]
            assert child["properties"]["review_id"]["minLength"] == 1
            primary = c.provider.load_schema("code_review:" + name).parameters
            assert "review_id" in primary["required"]
    for model in (ReadArguments, SubmitArguments):
        assert "review_id" in model.model_json_schema()["required"]


@pytest.mark.parametrize("identity", [None, "", "unknown-review", 12])
def test_explicit_invalid_identity_is_never_replaced(review_case, identity):
    c = review_case
    prepared = c.prepare()
    with c.participant(prepared) as provider:
        result = provider.invoke(
            "code_review:code_review_submit",
            {**submission(c, prepared), "review_id": identity},
        )
        assert not result.ok
        assert result.error == (
            "unknown_review" if identity == "unknown-review" else "invalid_arguments"
        )
    slot = summary(c, prepared)["slots"][0]
    assert slot["submission_revision"] == 0 and slot["seal_receipt"] is None


@pytest.mark.parametrize("failure", ["actor", "unbound", "revoked", "expired", "stale"])
def test_missing_identity_cannot_restore_lost_child_authority(review_case, failure):
    c = review_case
    prepared = c.prepare()
    with c.participant(prepared) as provider:
        args = submission(c, prepared)
        actor = provider.service.actor
        if failure == "actor":
            actor = c.actor
        elif failure == "unbound":
            actor = replace(actor, run_id="never-bound")
            provider.service.actor = actor
        elif failure == "revoked":
            c.issuer.revoke_session(c.owner, "chat")
        elif failure == "expired":
            c.store.close()
        else:
            binding = c.store._tab.runs[actor.run_id]
            c.store.bind_attempt(
                "replacement", binding.model_copy(update={"attempt_id": "new"})
            )
        with use_run_actor(actor):
            result = provider.invoke("code_review:code_review_submit", args)
        assert not result.ok
        assert (
            result.error != "invalid_arguments"
        )  # Refuse authority, not a missing field.


def test_primary_omission_and_foreign_child_id_never_select_a_review(review_case):
    c = review_case
    prepared = c.prepare()
    other_actor = replace(c.actor, run_id="other-primary")
    other_service = c.service(actor=other_actor)
    with use_run_actor(other_actor):
        another = other_service.prepare(c.working_tree_target)
    with use_run_actor(c.actor):
        result = c.provider.invoke("code_review:code_review_read", {"view": "summary"})
        assert result.error == "invalid_arguments"
    with c.participant(prepared) as provider:
        result = provider.invoke(
            "code_review:code_review_submit",
            {**submission(c, prepared), "review_id": another.review_id},
        )
        assert result.error == "review_scope_mismatch"
        result = provider.invoke(
            "code_review:code_review_submit",
            {**submission(c, prepared), "kind": "verifier_complete", "records": []},
        )
        assert result.error == "role_kind_mismatch"


def test_other_invalid_fields_keep_rejection_and_accepted_records(review_case):
    c = review_case
    prepared = c.prepare()
    with c.participant(prepared) as provider:
        provider.service.submit(
            prepared.review_id, "findings", [finding(c, prepared)], 0, "finding"
        )
        args = submission(c, prepared, revision=1)
        refused = provider.invoke(
            "code_review:code_review_submit", {**args, "expected_revision": "invalid"}
        )
        assert refused.error == "invalid_arguments"
        slot = summary(c, prepared)["slots"][0]
        assert slot["submission_revision"] == 1 and slot["rejected_submissions"]
        refused = provider.invoke(
            "code_review:code_review_submit",
            submission(c, prepared, "reviewer_complete", 1),
        )
        assert refused.error == "unresolved_submission_rejections"
        assert provider.invoke("code_review:code_review_submit", args).ok
        assert provider.invoke(
            "code_review:code_review_submit",
            submission(c, prepared, "reviewer_complete", 2),
        ).ok


def test_verifier_omission_keeps_accepted_dispositions_and_checks_completion(
    accepted_case,
):
    _, _, provider, records = accepted_case
    args = {
        "kind": "dispositions",
        "records": records,
        "expected_revision": 0,
        "submission_id": "dispositions",
    }
    accepted = provider.invoke("code_review:code_review_submit", args)
    assert accepted.ok, accepted.error
    stream = test_code_review_conflict_recovery.stream_for(accepted_case)
    original = dict(stream.verification.dispositions)
    assert provider.invoke("code_review:code_review_submit", args) == accepted
    assert stream.revision == 1 and stream.verification.dispositions == original
    checkpoint = provider.invoke("code_review:code_review_read", {"view": "checkpoint"})
    assert checkpoint.ok
    assert json.loads(checkpoint.content)["records"][0]["all_units_disposed"]
    done = provider.invoke(
        "code_review:code_review_submit",
        {
            "kind": "verifier_complete",
            "records": [],
            "expected_revision": 1,
            "submission_id": "done",
        },
    )
    assert done.ok, done.error
    assert stream.verification.dispositions == original
    assert (
        test_code_review_conflict_recovery.verifier_summary(accepted_case)[
            "seal_receipt"
        ]
        is None
    )

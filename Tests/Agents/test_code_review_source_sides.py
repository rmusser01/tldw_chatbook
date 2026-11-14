"""Captured target sides constrain child schemas and refused-read recovery."""

import json
from dataclasses import replace

import pytest

from Tests.Agents import code_review_helpers
from Tests.Agents.test_code_review_completion_recovery import invoke
from tldw_chatbook.Agents.code_review_models import PatchTarget, ReviewRefused
from tldw_chatbook.Agents.run_context import use_run_actor

review_case = code_review_helpers.review_case
source_case = code_review_helpers.source_case
pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.parametrize("role,slot", [("reviewer", "R1"), ("verifier", "V1")])
@pytest.mark.parametrize(
    "kind,allowed,invalid",
    [
        ("working_tree", ["base", "staged", "worktree"], "head"),
        ("commit_range", ["base", "head"], "worktree"),
        ("patch", ["base", "head"], "staged"),
    ],
)
def test_child_schema_and_refusal_use_captured_target_sides(
    review_case, role, slot, kind, allowed, invalid
):
    c = review_case
    if kind == "commit_range":
        c.working_tree_target = c.committed_target
    elif kind == "patch":
        patch = b"--- a/main.py\n+++ b/main.py\n@@ -1 +1 @@\n-captured bytes\n+patched bytes\n"
        c.authority = replace(c.authority, resolve_patch=lambda ref: patch)
        c.working_tree_target = PatchTarget(
            kind="patch", binding_ref="binding", patch_ref="accepted"
        )
    c.primary = c.service()
    prepared = c.prepare()
    with c.participant(prepared, role=role, slot=slot) as provider:
        schema = provider.load_schema("code_review:code_review_read")
        for name in ("ListSourceQuery", "SearchSourceQuery", "ResolveSourceQuery"):
            assert (
                schema.parameters["$defs"][name]["properties"]["side"]["enum"]
                == allowed
            )
        refused = invoke(
            provider,
            "read",
            prepared.review_id,
            view="source",
            source_query={"op": "list", "relative_dir": ".", "side": invalid},
        )
        assert not refused.ok and refused.error == "invalid_source_side"
        feedback = json.loads(refused.content)
        assert feedback["allowed_sides"] == allowed
        assert "retry" not in feedback  # Never silently alias an ambiguous side.
        for side in feedback["allowed_sides"]:
            result = invoke(
                provider,
                "read",
                prepared.review_id,
                view="source",
                source_query={"op": "list", "relative_dir": ".", "side": side},
            )
            assert result.ok, result.error
    # The unscoped primary can manage multiple target kinds.
    with use_run_actor(c.actor):
        generic = c.provider.load_schema("code_review:code_review_read")
    assert generic.parameters["$defs"]["ListSourceQuery"]["properties"]["side"][
        "enum"
    ] == ["base", "head", "staged", "worktree"]


@pytest.mark.parametrize("failure", ["foreign", "permission", "capacity"])
def test_side_feedback_preserves_authority_and_capacity(review_case, failure):
    c = review_case
    prepared = c.prepare()
    with c.participant(prepared) as provider:
        review_id = prepared.review_id
        if failure == "foreign":
            review_id = "unknown-review"
        elif failure == "permission":
            provider.service.source_permission = lambda *args: "permission_denied"
        else:
            provider.service.result_limit = 32
        result = invoke(
            provider,
            "read",
            review_id,
            view="source",
            source_query={"op": "list", "relative_dir": ".", "side": "head"},
        )
        assert not result.ok and result.content == ""
        assert (
            result.error
            == {
                "foreign": "unknown_review",
                "permission": "permission_denied",
                "capacity": "invalid_source_side",
            }[failure]
        )


def test_expired_child_cannot_reuse_captured_side_schema(review_case):
    c = review_case
    prepared = c.prepare()
    with c.participant(prepared) as provider:
        provider.load_schema("code_review:code_review_read")
        c.store.close()
        with pytest.raises((ReviewRefused, KeyError)):
            provider.load_schema("code_review:code_review_read")

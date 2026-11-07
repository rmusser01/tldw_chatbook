"""Fixed workflow window bounds complement the native producer controls."""

import json

import pytest

from tldw_chatbook.DB.workflow_runs import WorkflowRunsStore


@pytest.mark.parametrize(
    "roots",
    [
        (),
        ("",),
        ("/same", "/same"),
        tuple(f"/root-{index}" for index in range(65)),
        ("/" + "x" * 4096,),
        ("/" + "é" * 2048,),
        tuple(f"/{index}" + "x" * 4090 for index in range(9)),
    ],
)
def test_window_roots_refuse_count_utf8_and_serialized_overflow(roots):
    with pytest.raises(ValueError, match="workflow_window_roots_bound"):
        WorkflowRunsStore._window_roots(roots)


def test_window_roots_preserve_exact_maximum_count_and_bytes():
    roots = tuple(f"/root-{index}" for index in range(64))
    assert json.loads(WorkflowRunsStore._window_roots(roots)) == list(roots)
    root = "/" + "x" * 4095
    assert json.loads(WorkflowRunsStore._window_roots((root,))) == [root]


@pytest.mark.bootstrap_profile
def test_direct_anchor_call_invokes_foreign_continuation_before_owner_refusal(tmp_path):
    """The unrestricted Python call is not an issuance boundary."""
    from contextlib import closing

    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    calls = []

    class ForeignAuthorization:
        _binding_state = ()

        def _check_continuation(self):
            calls.append("called")

    with (
        closing(AgentRunsDB(tmp_path / "runs.db")) as db,
        pytest.raises(ValueError, match="^workflow_window_original_owner_required$"),
    ):
        db.workflow_runs.anchor_change_window("unused", ForeignAuthorization(), None)
    assert calls == ["called"]

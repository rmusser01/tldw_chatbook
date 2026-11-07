"""Extraction does not leave native request bodies outside workflow source checks."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.bootstrap_profile


def test_native_receiver_modules_are_from_the_candidate_tree():
    from tldw_chatbook.Agents import agent_request_execution, agent_service
    from tldw_chatbook.Chat import (
        console_agent_reply,
        console_streaming_request,
        console_trace_maintenance_runtime,
        console_project_dispatch,
    )

    root = Path(__file__).resolve().parents[2]
    for module in (
        agent_service,
        agent_request_execution,
        console_agent_reply,
        console_streaming_request,
        console_trace_maintenance_runtime,
        console_project_dispatch,
    ):
        assert Path(module.__file__).resolve().is_relative_to(root / "tldw_chatbook")
        print(module.__name__, Path(module.__file__).resolve())


@pytest.mark.parametrize("name", ["make_call_model", "make_invoke_tool"])
@pytest.mark.parametrize("change", ["replacement", "body"])
def test_actual_request_receiver_replacement_is_refused(name, change, monkeypatch):
    from tldw_chatbook.Agents import agent_request_execution, agent_service

    service = object.__new__(agent_service.AgentService)
    agent_service._workflow_native_sources(service)
    original = getattr(agent_request_execution, name)

    def substituted(*args, **kwargs):
        raise AssertionError("substituted body must never run")

    if change == "replacement":
        monkeypatch.setattr(agent_request_execution, name, substituted)
    else:
        monkeypatch.setattr(original, "__code__", substituted.__code__)
    with pytest.raises(
        PermissionError, match="original extracted native receiver changed"
    ):
        agent_service._workflow_native_sources(service)

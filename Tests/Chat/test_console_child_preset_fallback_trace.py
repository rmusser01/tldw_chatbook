"""TASK-42556: a spawned child's preset fallback (ADR-200) under Capture On.

The real controller, store, agent bridge, gateway and trace factory run over
real SQLite; only the HTTP socket is a double (``_spawn_turn``). The parent
spawns the ``worker`` preset, whose own model is unavailable and whose
``fallback_models`` names FALLBACK_MODEL.
"""

import pytest

from Tests.Chat.test_console_spawn_subagent_trace_capture import (
    FALLBACK_MODEL,
    MODEL,
    _spawn_turn,
)
from tldw_chatbook.Chat.console_agent_preparation import PRESET_FALLBACK_NOT_CAPTURED

pytestmark = [
    pytest.mark.bootstrap_profile,
    pytest.mark.owned_http_client,
    pytest.mark.asyncio,
]


@pytest.mark.parametrize("capture", ["capture_on", "capture_off"])
async def test_a_spawned_childs_preset_fallback(tmp_path, monkeypatch, capture):
    """Capture Off falls back; Capture On skips the fallback visibly.

    ADR-337 records no call for a child's fallback (children keep one chain
    and segment). On dev 6e41919880 the child still switched: it saved the
    fallback as its active target, its call was reserved, the reservation
    was refused at dispatch binding and stayed `reserved`, and the child
    ended on "trace capture could not be saved". Now each alternate is a
    visible skip before any reservation or provider call, and the child
    ends on its own model's error. The parent's follow-up is sent either way.
    """
    facts = await _spawn_turn(
        tmp_path, monkeypatch, "Say child done.", preset_fallback=True, capture=capture
    )
    why = str(vars(facts))
    assert facts.result.accepted and facts.result.provider_started, why
    primary = [roles for kind, roles in facts.wire if kind == "primary"]
    assert primary == [["system", "user"], ["system", "user", "assistant", "tool"]], why
    child_models = [model for kind, model in facts.models if kind == "subagent"]
    switched = [s for _, s in facts.child_steps if s.startswith("Provider fallback:")]
    if capture == "capture_off":
        assert facts.runs == [("primary", "done"), ("subagent", "done")], why
        assert child_models == [MODEL, FALLBACK_MODEL], why
        assert len(switched) == 1 and facts.calls == [], why
        return
    assert facts.runs == [("primary", "done"), ("subagent", "error")], why
    assert child_models == [MODEL], why
    assert switched == [], why
    skipped = "Provider fallback skipped"
    skips = [s for _, s in facts.child_steps if s.startswith(skipped)]
    assert skips == [
        f"Provider fallback skipped: custom-ep:spawntrace/{FALLBACK_MODEL} "
        f"({PRESET_FALLBACK_NOT_CAPTURED})"
    ], why
    assert len(facts.child_errors) == 1, why
    assert "trace capture" not in facts.child_errors[0], why
    # Nothing is reserved for the fallback: the child's only call is its own.
    child_calls = [call[4:] for call in facts.owned_calls if not call[0]]
    assert child_calls == [("agent_first", "error")], why
    assert facts.boundary_errors == [] and facts.refusals == [], why

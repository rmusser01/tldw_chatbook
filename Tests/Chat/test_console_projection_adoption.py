"""One adoption retains checked values without changing the public codecs."""

from copy import deepcopy
from dataclasses import replace

import pytest

from tldw_chatbook.Chat import console_execution_projection_result as codec
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage, ConsoleMessageRole, ConsoleRunMarker, ConsoleRunState,
    ConsoleRunStatus,
)
from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession
from tldw_chatbook.Chat.console_control_dispatch import ConsoleProjection
from tldw_chatbook.Chat.console_execution_agent_rail_values import (
    ConsoleAgentRailProjection, encode_agent_rail_projection,
)
from tldw_chatbook.Chat.console_execution_composition_values import decode_session_composition
from tldw_chatbook.Chat.console_execution_context_summary_values import encode_context_summary
from tldw_chatbook.Chat.console_execution_projection_values import (
    _settings, encode_message_projection, encode_session_projection,
)
from tldw_chatbook.Chat.console_execution_readiness_values import encode_settings_readiness
from tldw_chatbook.Chat.console_execution_session_surface import decode_session_surface_projection
from tldw_chatbook.Chat.console_execution_workspace_values import (
    ConsoleWorkspaceSessionFacts, ConsoleWorkspaceSurface,
    decode_workspace_surface, encode_workspace_surface,
)
from tldw_chatbook.Chat.console_inspector_state import ConsoleInspectorSectionState
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings, ConsoleSettingsContextEstimate, ConsoleSettingsReadiness,
    ConsoleSettingsSummaryState,
)
from Tests.Chat.test_console_execution_composition_values import _encode


def projection_payload(session_count=1, message_count=2):
    """Generate inert rows using stock constructors/encoders, without a DB/App."""
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model")
    sessions = [ConsoleChatSession(id=f"session-{i}", settings=settings)
                for i in range(session_count)]
    current = sessions[-1]
    frame = ConsoleProjection(
        3, current.id, current.incarnation_id, 7,
        tuple(ConsoleChatMessage(ConsoleMessageRole.ASSISTANT, "answer " * 20,
                                 id=f"message-{i}") for i in range(message_count)),
        ConsoleRunState(ConsoleRunStatus.IDLE, "Ready"), None, 2,
    )
    payload = codec.encode_projection_payload(
        frame, messages=[encode_message_projection(row) for row in frame.messages],
    )
    readiness = ConsoleSettingsReadiness("Ready", "", True)
    payload["session_view"] = {
        "sessions": [encode_session_projection(row) for row in sessions],
        "active_session_id": current.id,
        "fingerprints": [{"session_id": row.id, "count": message_count,
                          "last_content_length": 140} for row in sessions],
        "pressed_inputs": {
            "draft": current.draft, "authored_token": None,
            "settings": _settings(settings, False), "settings_revision": 0,
            "attachment_revision": 0, "attachment_ids": [], "attachments": [],
            "one_shot_prefill": None, "prefill_revision": 0,
            "session_id": current.id, "incarnation_id": current.incarnation_id,
        },
        "readiness": {
            "effective_settings": _settings(settings, False),
            "state": encode_settings_readiness(readiness), "provider_label": "Local",
            "control_provider_label": "Local", "model": settings.model,
            "remaining_seconds": 0.8, "captured_at_monotonic": 100.0,
            "context_summary": encode_context_summary(
                ConsoleSettingsContextEstimate(32, 4096, "32 / 4096"),
                ConsoleSettingsSummaryState("model", "32 / 4096", "", "", readiness=readiness),
                compaction_pause_expected=False, compaction_forecast_unavailable=False,
            ),
            "agent_rail": encode_agent_rail_projection(ConsoleAgentRailProjection(
                ("Idle", "", ""), "", ConsoleInspectorSectionState((), ""), False, 0, False,
            )),
            "prompt_history": {"revision": 0, "entries": [], "loaded": True, "error": None},
        },
        "composition": _encode(current),
        "workspace": encode_workspace_surface(ConsoleWorkspaceSurface(
            None, True, tuple(ConsoleWorkspaceSessionFacts(
                row.id, "", None, None, None, 0, ConsoleRunMarker.NONE,
            ) for row in sessions),
        )),
        "session_surface": {
            "streaming_session_id": None,
            "run_markers": [{"session_id": row.id, "marker": "none"} for row in sessions],
            "queue_counts": [{"session_id": row.id, "count": 0} for row in sessions],
        },
    }
    return payload


def previous_adoption(payload, previous=None):
    """The original client order, through the unchanged public decoder APIs."""
    frame = codec.decode_projection_payload(payload, previous=previous)
    view = codec.decode_projection_view(payload, frame)
    body = payload["session_view"]
    workspace = decode_workspace_surface(body["workspace"])
    surface = decode_session_surface_projection(view, body["session_surface"])
    readiness = codec.decode_projection_readiness(payload, view)
    composition = (decode_session_composition(body["composition"], view.session(view.active_session_id))
                   if view.active_session_id is not None else None)
    return frame, view, workspace, surface, readiness, composition


def test_adoption_retains_original_values_and_exact_final_view():
    payload = projection_payload(3, 4)
    actual = codec.decode_projection_adoption(payload)
    assert actual == previous_adoption(payload)
    assert actual[3].view is actual[1]
    assert actual[1].projections[0] is actual[0]
    payload["session_view"]["sessions"][-1]["draft"] = "changed later"
    payload["session_view"]["workspace"]["sessions"].clear()
    payload["messages"][0]["content"] = "changed later"
    assert actual[1].session(actual[0].session_id).draft == ""
    assert len(actual[2].sessions) == 3
    assert actual[0].messages[0].content == "answer " * 20


def test_each_display_section_decodes_once_per_adoption(monkeypatch):
    from tldw_chatbook.Chat import console_execution_composition_values as composition
    from tldw_chatbook.Chat import console_execution_workspace_values as workspace
    from tldw_chatbook.Chat import console_execution_session_surface as surface
    from tldw_chatbook.Chat import console_execution_projection_values as values

    payload = projection_payload(3)
    counts = {}
    for owner, name in (
        (codec, "_readiness"), (composition, "decode_session_composition"),
        (workspace, "decode_workspace_surface"), (surface, "decode_session_surface_projection"),
        (values, "decode_session_projection"),
    ):
        original = getattr(owner, name)

        def counted(*args, _original=original, _name=name, **kwargs):
            counts[_name] = counts.get(_name, 0) + 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(owner, name, counted)
    codec.decode_projection_adoption(payload)
    assert counts == {
        "_readiness": 1, "decode_session_composition": 1,
        "decode_workspace_surface": 1, "decode_session_surface_projection": 1,
        "decode_session_projection": 3,
    }


@pytest.mark.parametrize("path,value", [
    (("marker_revision",), True),
    (("incarnation_id",), "different"),
    (("messages", 0, "sibling_count"), 0),
    (("session_view", "active_session_id"), "missing"),
    (("session_view", "sessions", 1, "id"), "session-0"),
    (("session_view", "fingerprints", 0, "count"), -1),
    (("session_view", "pressed_inputs", "draft"), "different"),
    (("session_view", "readiness", "remaining_seconds"), 2),
    (("session_view", "readiness", "effective_settings", "temperature"), 0.9),
    (("session_view", "readiness", "prompt_history", "entries"), [{"input": "x", "timestamp": "null"}]),
    (("session_view", "composition", "settings_revision"), 1),
    (("session_view", "workspace", "sessions"), []),
    (("session_view", "session_surface", "queue_counts"), []),
])
def test_original_malformed_section_error_and_precedence(path, value):
    payload = projection_payload(2)
    parent = payload
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    # A later, independently malformed value must not hide the first failure.
    payload["session_view"]["session_surface"]["streaming_session_id"] = "missing"
    with pytest.raises(Exception) as expected:
        previous_adoption(deepcopy(payload))
    with pytest.raises(type(expected.value)) as actual:
        codec.decode_projection_adoption(deepcopy(payload))
    assert actual.value.args == expected.value.args


def test_legacy_nullable_and_missing_views_keep_original_behavior():
    payload = projection_payload()
    body = payload["session_view"]
    body["active_session_id"] = None
    body["composition"] = None
    assert codec.decode_projection_adoption(payload) == previous_adoption(payload)
    body["composition"] = {}
    with pytest.raises(Exception) as expected:
        previous_adoption(payload)
    with pytest.raises(type(expected.value), match="composition_without_active_session"):
        codec.decode_projection_adoption(payload)
    for key in ("readiness", "composition", "workspace", "session_surface"):
        body.pop(key)
    frame = codec.decode_projection_payload(payload)
    assert codec.decode_projection_view(payload, frame).active_session_id is None
    for decode in (previous_adoption, codec.decode_projection_adoption):
        with pytest.raises(KeyError, match="workspace"):
            decode(payload)
    payload.pop("session_view")
    for decode in (previous_adoption, codec.decode_projection_adoption):
        with pytest.raises(Exception, match="projection_view_missing"):
            decode(payload)


def test_reference_reuse_requires_same_original_frame_and_marker():
    payload = projection_payload()
    first = codec.decode_projection_adoption(payload)[0]
    payload["messages"] = None
    actual = codec.decode_projection_adoption(payload, previous=first)
    assert actual == previous_adoption(payload, previous=first)
    assert actual[0].messages is first.messages
    for previous in (None, replace(first, marker_revision=3), replace(first, source_revision=4)):
        with pytest.raises(Exception, match="projection_snapshot_not_owned"):
            codec.decode_projection_adoption(payload, previous=previous)

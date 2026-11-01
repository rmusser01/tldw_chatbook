"""Focused unit controls for original Task35 discovery UI error policy.

These execute installed controller/binder bodies. Controlled source predicates
below isolate collaborator ordering; they do not qualify App ownership, native
preparation, source-reader completeness, or the original cold-mount scenario.
"""

import asyncio
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.asyncio
@pytest.mark.parametrize("changed_source", ("wiring", "helper", "prepared"))
async def test_preparation_return_refusal_precedes_get_context(
    monkeypatch, record_property, changed_source
):
    from tldw_chatbook.UI.Console_Modules import skill

    events = []
    completed = False
    wiring_source, helper_source, controller_source = object(), object(), object()
    source_reads = []

    def current():
        return True

    def source_current(source):
        assert source is wiring_source or source is helper_source
        role = "wiring" if source is wiring_source else "helper"
        source_reads.append((role, completed))
        return not (completed and changed_source == role)

    def prepared_current():
        source_reads.append(("prepared", completed))
        return not (completed and changed_source == "prepared")

    async def get_context(*, mode):
        events.append(("get_context", mode, completed))
        return {}

    service = SimpleNamespace(get_context=get_context)
    app = SimpleNamespace(skills_scope_service=service)
    prepared = (app, prepared_current)

    def capture(actual_app, actual_service, actual_current, actual_source):
        assert actual_app is app
        assert actual_service is service
        assert actual_current is current
        assert actual_source is controller_source
        events.append(("capture",))
        return prepared

    async def prepare(actual_prepared):
        nonlocal completed
        assert actual_prepared is prepared
        events.append(("prepare_enter",))
        await asyncio.sleep(0)  # Unit scheduling boundary, not a native hold.
        completed = True
        events.append(("prepare_return",))

    async def append_message(_message):
        raise AssertionError("This fetch control must not append a message")

    controller = skill.ConsoleSkillController(
        app_instance=app,
        append_native_console_system_message=append_message,
        sync_console_command_popup=lambda: None,
        task_resume_state=lambda: None,
        set_task_resume_state=lambda _state: None,
        current_chat_controller=lambda: None,
    )
    metadata = SimpleNamespace(_source_current=source_current)
    wiring = ModuleType("tldw_chatbook.app_service_wiring")
    wiring._CONSOLE_SKILL_WIRING_SOURCE = wiring_source
    wiring._CONSOLE_SKILL_HELPER_SOURCE = helper_source
    wiring._capture_console_skill_trust_preparation = capture
    wiring._prepare_console_skill_trust_service = prepare
    wiring._CONSOLE_SKILL_ENTRY = (capture, prepare)
    original_capture = skill._capture_console_skill_discovery
    original_wiring_present = "tldw_chatbook.app_service_wiring" in sys.modules
    original_wiring = sys.modules.get("tldw_chatbook.app_service_wiring")

    def capture_controller(actual_controller):
        assert actual_controller is controller
        return current, controller_source, metadata

    result = None
    escaped = None
    with monkeypatch.context() as patch:
        patch.setattr(skill, "_capture_console_skill_discovery", capture_controller)
        patch.setitem(sys.modules, "tldw_chatbook.app_service_wiring", wiring)
        try:
            result = await controller._fetch_console_skill_context()
        except Exception as error:
            escaped = error

    # Restoration precedes the causal count assertion, including on R1 RED.
    assert skill._capture_console_skill_discovery is original_capture
    assert ("tldw_chatbook.app_service_wiring" in sys.modules) is original_wiring_present
    assert sys.modules.get("tldw_chatbook.app_service_wiring") is original_wiring
    record_property("unit_events", json.dumps(events))
    record_property("unit_source_reads", json.dumps(source_reads))
    record_property("get_context_calls", sum(row[0] == "get_context" for row in events))
    record_property("escaped_exception_type", type(escaped).__name__ if escaped is not None else None)
    assert completed is True
    assert events[:3] == [("capture",), ("prepare_enter",), ("prepare_return",)]
    assert sum(row[0] == "capture" for row in events) == 1
    assert sum(row[0] == "prepare_enter" for row in events) == 1
    assert (changed_source, True) in source_reads
    assert not any(row[0] == "get_context" for row in events)
    assert escaped is None
    assert result == {}


def test_second_binding_does_not_replace_original_discovery_witness(
    monkeypatch, record_property
):
    from tldw_chatbook.UI.Console_Modules import skill, wiring
    from tldw_chatbook.UI.Screens import chat_screen
    from tldw_chatbook.Widgets import compact_model_bar

    view_type = chat_screen.ChatScreen
    source = skill._DISCOVERY_SOURCE
    bound_view = skill._DISCOVERY_VIEW
    original_popup = vars(view_type)["_sync_console_command_popup"]
    assert bound_view is view_type
    assert type(source) is tuple and len(source) == 6
    assert compact_model_bar._source_current(source) is True
    popup_rows = tuple(row for row in source[5] if row[1] == "_sync_console_command_popup")
    assert len(popup_rows) == 1
    assert popup_rows[0][2] is original_popup
    assert popup_rows[0][0]["_sync_console_command_popup"] is original_popup
    calls = []

    def customized_popup(self):
        calls.append(self)

    with monkeypatch.context() as patch:
        # Register TEST undo for the known R1 side effect. Product code must not
        # reset or replace the first witness to make this control pass.
        patch.setattr(skill, "_DISCOVERY_SOURCE", source)
        patch.setattr(skill, "_DISCOVERY_VIEW", bound_view)
        patch.setattr(view_type, "_sync_console_command_popup", customized_popup)
        assert compact_model_bar._source_current(source) is False
        skill.bind_console_skill_discovery(view_type, wiring.build_console_controllers)
        rebound_source = skill._DISCOVERY_SOURCE
        rebound_view = skill._DISCOVERY_VIEW
        original_still_refuses = compact_model_bar._source_current(source)
        rebound_current = compact_model_bar._source_current(rebound_source)

    # Both the popup slot and R1's replaced capsule are restored before RED.
    assert vars(view_type)["_sync_console_command_popup"] is original_popup
    assert skill._DISCOVERY_SOURCE is source
    assert skill._DISCOVERY_VIEW is bound_view
    assert compact_model_bar._source_current(source) is True
    record_property("same_original_witness", rebound_source is source)
    record_property("original_witness_current_while_customized", original_still_refuses)
    record_property("rebound_witness_current_while_customized", rebound_current)
    record_property("popup_calls", len(calls))
    assert calls == []
    assert original_still_refuses is False
    assert rebound_view is bound_view
    assert rebound_source is source
    assert rebound_current is False


@pytest.mark.asyncio
@pytest.mark.parametrize("policy_case", ("initial_stale", "prepare_error", "prepare_cancel"))
async def test_discovery_preparation_preserves_original_ui_error_policy(
    monkeypatch, record_property, policy_case
):
    from tldw_chatbook.UI.Console_Modules import skill

    events = []
    ordinary_error = LookupError("controlled-preparation-failure")
    cancellation = asyncio.CancelledError("controlled-preparation-cancellation")
    wiring_source, helper_source, controller_source = object(), object(), object()

    def current():
        events.append("current")
        return policy_case != "initial_stale"

    def source_current(source):
        assert source is wiring_source or source is helper_source
        events.append("source_current")
        return True

    def prepared_current():
        events.append("prepared_current")
        return True

    async def get_context(*, mode):
        events.append("get_context")
        assert mode == "local"
        return {}

    app = SimpleNamespace(skills_scope_service=SimpleNamespace(get_context=get_context))
    service = app.skills_scope_service
    prepared = (app, prepared_current)

    def capture(actual_app, actual_service, actual_current, actual_source):
        assert actual_app is app and actual_service is service
        assert actual_current is current and actual_source is controller_source
        events.append("capture")
        return prepared

    async def prepare(actual_prepared):
        assert actual_prepared is prepared
        events.append("prepare_enter")
        await asyncio.sleep(0)
        events.append("prepare_exception")
        if policy_case == "prepare_cancel":
            raise cancellation
        raise ordinary_error

    async def append_message(_message):
        raise AssertionError("The fetch control must not append a message")

    controller = skill.ConsoleSkillController(
        app_instance=app,
        append_native_console_system_message=append_message,
        sync_console_command_popup=lambda: None,
        task_resume_state=lambda: None,
        set_task_resume_state=lambda _state: None,
        current_chat_controller=lambda: None,
    )
    metadata = SimpleNamespace(_source_current=source_current)
    wiring = ModuleType("tldw_chatbook.app_service_wiring")
    wiring._CONSOLE_SKILL_WIRING_SOURCE = wiring_source
    wiring._CONSOLE_SKILL_HELPER_SOURCE = helper_source
    wiring._capture_console_skill_trust_preparation = capture
    wiring._prepare_console_skill_trust_service = prepare
    wiring._CONSOLE_SKILL_ENTRY = (capture, prepare)
    original_capture = skill._capture_console_skill_discovery
    original_wiring_present = "tldw_chatbook.app_service_wiring" in sys.modules
    original_wiring = sys.modules.get("tldw_chatbook.app_service_wiring")

    def capture_controller(actual_controller):
        assert actual_controller is controller
        events.append("capture_controller")
        return current, controller_source, metadata

    warning_errors = []

    def observe_log(message):
        record = message.record
        if (
            record["level"].name == "WARNING"
            and record["message"] == "Console skill context fetch failed."
        ):
            exception = record["exception"]
            warning_errors.append(exception.value if exception is not None else None)

    result = None
    escaped = None
    sink = skill.logger.add(observe_log, level="WARNING")
    try:
        with monkeypatch.context() as patch:
            patch.setattr(skill, "_capture_console_skill_discovery", capture_controller)
            patch.setitem(sys.modules, "tldw_chatbook.app_service_wiring", wiring)
            try:
                result = await controller._fetch_console_skill_context()
            except (Exception, asyncio.CancelledError) as error:
                escaped = error
    finally:
        skill.logger.remove(sink)

    # Cleanup is complete before either expected policy failure is asserted.
    assert skill._capture_console_skill_discovery is original_capture
    assert ("tldw_chatbook.app_service_wiring" in sys.modules) is original_wiring_present
    assert sys.modules.get("tldw_chatbook.app_service_wiring") is original_wiring
    record_property("unit_policy_case", policy_case)
    record_property("unit_events", json.dumps(events))
    record_property("escaped_exception_type", type(escaped).__name__ if escaped is not None else None)
    record_property("escaped_same_ordinary_error", escaped is ordinary_error)
    record_property("escaped_same_cancellation", escaped is cancellation)
    record_property("original_warning_count", len(warning_errors))
    record_property("original_warning_same_error", len(warning_errors) == 1 and warning_errors[0] is ordinary_error)
    assert events.count("capture_controller") == 1
    assert events.count("get_context") == 0
    if policy_case == "initial_stale":
        assert events == ["capture_controller", "current"]
        assert escaped is None
        assert result == {}
    elif policy_case == "prepare_error":
        assert events.count("capture") == events.count("prepare_enter") == events.count("prepare_exception") == 1
        assert escaped is None
        assert result == {}
        assert len(warning_errors) == 1 and warning_errors[0] is ordinary_error
    else:
        assert events.count("capture") == events.count("prepare_enter") == events.count("prepare_exception") == 1
        assert escaped is cancellation
        assert result is None
        assert warning_errors == []

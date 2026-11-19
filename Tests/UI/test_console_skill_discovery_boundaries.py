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
@pytest.mark.parametrize("changed_source", ("owner", "prepared"))
async def test_preparation_return_refusal_precedes_get_context(
    monkeypatch, record_property, changed_source
):
    # TASK-41642 (ADR-356 rule 4): the wiring/helper source capsules are gone;
    # the discovery owner (its view, app and runtime registrations) and the
    # preparation's own source decide after the preparation returns.
    from tldw_chatbook.UI.Console_Modules import skill

    events = []
    completed = False
    source_reads = []

    def current():
        source_reads.append(("owner", completed))
        return not (completed and changed_source == "owner")

    def prepared_current():
        source_reads.append(("prepared", completed))
        return not (completed and changed_source == "prepared")

    async def get_context(*, mode):
        events.append(("get_context", mode, completed))
        return {}

    service = SimpleNamespace(get_context=get_context)
    app = SimpleNamespace(skills_scope_service=service)
    prepared = (app, prepared_current)

    def capture(actual_app, actual_service, actual_current):
        assert actual_app is app
        assert actual_service is service
        assert actual_current is current
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
    wiring = ModuleType("tldw_chatbook.app_service_wiring")
    wiring._capture_console_skill_trust_preparation = capture
    wiring._prepare_console_skill_trust_service = prepare
    original_capture = skill._capture_console_skill_discovery
    original_wiring_present = "tldw_chatbook.app_service_wiring" in sys.modules
    original_wiring = sys.modules.get("tldw_chatbook.app_service_wiring")

    def capture_controller(actual_controller):
        assert actual_controller is controller
        return current

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


def test_second_binding_does_not_replace_the_bound_discovery_view(
    monkeypatch, record_property
):
    # TASK-41642 (ADR-356 rule 4): binding records no code, only the view type
    # and the context classes, and the first binding still wins. A popup
    # override on one view is refused at capture (the subprocess control in
    # test_console_skill_discovery_wrapped_collaborators.py, instance_view_popup).
    from tldw_chatbook.UI.Console_Modules import skill
    from tldw_chatbook.UI.Screens import chat_screen

    view_type = chat_screen.ChatScreen
    bound_view = skill._DISCOVERY_VIEW
    bound_classes = skill._CONTEXT_CLASSES
    assert bound_view is view_type
    assert type(bound_classes) is tuple and len(bound_classes) == 5

    class OtherView(view_type):
        pass

    with monkeypatch.context() as patch:
        # Register TEST undo in case product code wrongly rebinds.
        patch.setattr(skill, "_DISCOVERY_VIEW", bound_view)
        patch.setattr(skill, "_CONTEXT_CLASSES", bound_classes)
        skill.bind_console_skill_discovery(OtherView)
        rebound_view = skill._DISCOVERY_VIEW
        rebound_classes = skill._CONTEXT_CLASSES

    assert skill._DISCOVERY_VIEW is bound_view
    record_property("same_bound_view", rebound_view is bound_view)
    assert rebound_view is bound_view
    assert rebound_classes is bound_classes


@pytest.mark.asyncio
@pytest.mark.parametrize("policy_case", ("initial_stale", "prepare_error", "prepare_cancel"))
async def test_discovery_preparation_preserves_original_ui_error_policy(
    monkeypatch, record_property, policy_case
):
    from tldw_chatbook.UI.Console_Modules import skill

    events = []
    ordinary_error = LookupError("controlled-preparation-failure")
    cancellation = asyncio.CancelledError("controlled-preparation-cancellation")

    def current():
        events.append("current")
        return policy_case != "initial_stale"

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

    def capture(actual_app, actual_service, actual_current):
        assert actual_app is app and actual_service is service
        assert actual_current is current
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
    wiring = ModuleType("tldw_chatbook.app_service_wiring")
    wiring._capture_console_skill_trust_preparation = capture
    wiring._prepare_console_skill_trust_service = prepare
    original_capture = skill._capture_console_skill_discovery
    original_wiring_present = "tldw_chatbook.app_service_wiring" in sys.modules
    original_wiring = sys.modules.get("tldw_chatbook.app_service_wiring")

    def capture_controller(actual_controller):
        assert actual_controller is controller
        events.append("capture_controller")
        return current

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

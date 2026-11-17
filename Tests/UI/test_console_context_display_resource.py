"""A composed display resource survives transparent callback decoration."""

import asyncio

import pytest

from Tests.UI.test_console_received_intent_feedback import _received_console_case
from Tests.UI.test_console_shared_context_policy import _actual_calls
from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


async def test_owned_display_survives_transparent_context_callback(monkeypatch):
    """A wrapper changes no display owner, published value or live revisions."""
    async with _received_console_case(
        monkeypatch, "context-display-wrapper", durable=True
    ) as case:
        screen, controller, session_id = (
            case.console, case.controller, case.session.id
        )
        snapshot = spend.ConsoleContextReadSnapshot.for_screen(screen, max_age=1.0)
        assert await snapshot.warm(controller, session_id)
        expected = snapshot.value
        assert expected is not None
        original = screen._context_spend._context_inputs
        assert original(controller, session_id) is expected
        assert spend.console_transcript_context_inputs(
            screen, controller, session_id, refresh=False
        ) is expected, "The original composed reader did not publish current inputs"

        def transparent(*args, **kwargs):
            return original(*args, **kwargs)

        with monkeypatch.context() as changed:
            changed.setattr(screen._context_spend, "_context_inputs", transparent)
            tasks_before = asyncio.all_tasks()
            with _actual_calls(controller) as calls:
                actual = spend.console_transcript_context_inputs(
                    screen, controller, session_id, refresh=False
                )
            assert calls == {"config": 0, "live_policy": 0}
            assert asyncio.all_tasks() == tasks_before
            assert actual is expected, (
                "Transparent decoration revoked the same live display resource"
            )
        assert screen._context_spend._context_inputs is original
        assert snapshot.value is expected
        assert case.provider_calls == []


async def test_registered_display_keeps_actual_owner_and_current_publication(monkeypatch):
    """Copied bindings and changed display state cannot reuse a live publication."""
    from copy import copy

    from Tests.UI.console_controller_stubs import context_spend_for_test
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole

    async with _received_console_case(
        monkeypatch, "context-display-currentness", durable=True
    ) as case:
        screen, controller, session_id = case.console, case.controller, case.session.id
        snapshot = spend.ConsoleContextReadSnapshot.for_screen(screen, max_age=1.0)
        assert await snapshot.warm(controller, session_id)
        owner = screen._context_spend
        expected = owner._context_inputs(controller, session_id)
        assert expected is snapshot.value and expected is not None
        resource = owner._context_display_resource

        def current(selected_controller=controller, selected_session=session_id):
            return spend.console_transcript_context_inputs(
                screen, selected_controller, selected_session, refresh=False
            )

        assert current() is expected
        copied_owner, copied_resource = copy(owner), copy(resource)
        for candidate in (resource, copied_resource):
            with pytest.raises(ValueError, match="already_bound"):
                candidate.bind(copied_owner)
        with monkeypatch.context() as changed:
            changed.setattr(owner, "_context_display_resource", copied_resource)
            assert current() is None, "A copied resource inherited the original binding"
        with monkeypatch.context() as changed:
            changed.setattr(screen, "_context_spend", copied_owner)
            assert current() is None, "A copied controller inherited its source owner's binding"
            changed.setattr(copied_owner, "_context_display_resource", copied_resource)
            assert current() is None, "Copied owner/resource records recreated registration"
        assert current() is expected
        assert current(copy(controller)) is None
        assert current(selected_session="another-session") is None

        projection = snapshot.config_projection
        assert projection is not None
        mutations = (
            (screen, "_console_context_read_snapshot", copy(snapshot), "snapshot"),
            (snapshot, "config_projection", copy(projection), "config owner"),
            (snapshot, "value", None, "missing publication"),
            (projection, "value", None, "missing config publication"),
            (screen, "_console_attach_visit_generation",
             screen._console_attach_visit_generation + 1, "attachment visit"),
            (screen, "_closing", True, "closing screen"),
            (controller, "_disposed", True, "closed controller"),
        )
        for target, name, value, label in mutations:
            with monkeypatch.context() as changed:
                changed.setattr(target, name, value)
                tasks_before = asyncio.all_tasks()
                with _actual_calls(controller) as calls:
                    assert current() is None, label
                assert calls == {"config": 0, "live_policy": 0}, label
                assert asyncio.all_tasks() == tasks_before, label
            assert current() is expected, label

        original_read = snapshot.current_inputs

        def transparent_snapshot(*args, **kwargs):
            return original_read(*args, **kwargs)

        with monkeypatch.context() as changed:
            changed.setattr(snapshot, "current_inputs", transparent_snapshot)
            assert current() is expected, "A transparent resource wrapper changed ownership"

        # This established helper uses the actual public constructor without the
        # optional resource. Its caller-supplied callback must remain live.
        custom_value = (object(), object(), object())
        custom_calls = []

        def custom(selected_controller, selected_session):
            custom_calls.append((selected_controller, selected_session))
            return custom_value

        with monkeypatch.context() as changed:
            changed.delattr(screen, "_context_spend")
            unregistered = context_spend_for_test(screen)
            assert unregistered is not owner
            assert unregistered._context_display_resource is None
            changed.setattr(unregistered, "_context_inputs", custom)
            assert current() is None
            assert spend.console_transcript_context_inputs(
                screen, controller, session_id
            ) is custom_value
            assert custom_calls == [(controller, session_id)]
        assert screen._context_spend is owner
        assert current() is expected

        # Actual store mutation, not a fabricated publication/version token.
        store = controller.store
        other = store.create_session(activate=False, workspace_id=case.session.workspace_id)
        with monkeypatch.context() as changed:
            changed.setattr(store, "active_session_id", other.id)
            assert current() is None
        store.append_message(session_id, role=ConsoleMessageRole.USER, content="New context")
        tasks_before = asyncio.all_tasks()
        with _actual_calls(controller) as calls:
            assert current() is None, "Changed transcript revision reused its old display"
        assert calls == {"config": 0, "live_policy": 0}
        assert asyncio.all_tasks() == tasks_before
        assert snapshot.value is expected
        assert case.provider_calls == []

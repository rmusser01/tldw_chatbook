"""Workbench-to-app ownership fences for Actor Pack export operations."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from tldw_chatbook.Actor_Packs.controller import ActorPackExportOutcome
from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen


class _Controller:
    def __init__(self) -> None:
        self.started: list[object] = []
        self.cancelled: list[int] = []
        self.outcome = ActorPackExportOutcome(
            operation_id=1,
            error_category="actor_pack_export_cancelled",
        )

    def start_export(self, request: object) -> int:
        self.started.append(request)
        return 1

    async def wait(self, operation_id: int) -> ActorPackExportOutcome:
        assert operation_id == 1
        return self.outcome

    def cancel(self, operation_id: int) -> bool:
        self.cancelled.append(operation_id)
        return True


def _screen(controller: _Controller) -> PersonasScreen:
    screen = object.__new__(PersonasScreen)
    screen.app_instance = SimpleNamespace(actor_pack_export_controller=controller)
    screen.state = SimpleNamespace(
        runtime_source="local",
        selected_entity_kind="character",
        selected_entity_id="7",
    )
    screen._persona_buddy_session_generation = 4
    screen._actor_pack_export_operation = None
    screen._actor_pack_export_authority = None
    return screen


@pytest.mark.asyncio
async def test_late_result_applies_only_to_exact_screen_selection_authority() -> None:
    controller = _Controller()
    screen = _screen(controller)
    request = object()

    operation, authority = screen._start_actor_pack_export(request)
    current = await screen._wait_actor_pack_export(operation, authority)
    assert current is controller.outcome

    operation, authority = screen._start_actor_pack_export(request)
    screen._advance_persona_buddy_session()
    stale = await screen._wait_actor_pack_export(operation, authority)

    assert stale is None
    assert controller.started == [request, request]


def test_navigation_requests_cancellation_without_owning_worker() -> None:
    controller = _Controller()
    screen = _screen(controller)
    operation, _authority = screen._start_actor_pack_export(object())

    screen._cancel_actor_pack_export()

    assert controller.cancelled == [operation]
    assert screen._actor_pack_export_operation is None
    on_unmount = inspect.getsource(PersonasScreen.on_unmount)
    assert "_cancel_actor_pack_export" in on_unmount


def test_replacement_screen_cannot_claim_prior_screen_operation() -> None:
    controller = _Controller()
    first = _screen(controller)
    second = _screen(controller)
    operation, authority = first._start_actor_pack_export(object())

    assert second._actor_pack_export_authority_is_current(authority) is False
    assert first._actor_pack_export_authority_is_current(authority) is True
    assert operation == 1

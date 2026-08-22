"""Workbench-to-app ownership fences for Actor Pack export operations."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_chatbook.Actor_Packs.controller import ActorPackExportOutcome
from tldw_chatbook.Actor_Packs.export import ActorPackExportResult
from tldw_chatbook.Widgets.Persona_Widgets.personas_messages import (
    ActorPackExportRequested,
)
from tldw_chatbook.UI.Screens.personas_screen import (
    PersonasScreen,
    _actor_pack_export_filename,
    _normalize_actor_pack_destination,
)


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
    service = object()
    screen.app_instance = SimpleNamespace(
        actor_pack_export_controller=controller,
        actor_pack_export_service=service,
    )
    screen.state = SimpleNamespace(
        runtime_source="local",
        selected_entity_kind="character",
        selected_entity_id="7",
    )
    screen._persona_buddy_session_generation = 4
    screen._edit_mode = "view"
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


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        ("actor", "actor.tldw-actor-pack"),
        ("actor.tldw-actor-pack", "actor.tldw-actor-pack"),
        ("actor.TLDW-ACTOR-PACK", "actor.TLDW-ACTOR-PACK"),
    ),
)
def test_actor_pack_suffix_is_added_exactly_once(value: str, expected: str) -> None:
    assert _normalize_actor_pack_destination(value).name == expected
    assert _actor_pack_export_filename("  My/Actor []  ") == "MyActor.tldw-actor-pack"


class _EligibilityService:
    def __init__(self) -> None:
        self.calls = 0

    def capture_eligibility(self, *_args, **_kwargs):
        self.calls += 1
        return SimpleNamespace(actor_revision=4)


class _WorkflowController(_Controller):
    def __init__(self) -> None:
        super().__init__()
        self.requests: list[object] = []
        self.outcome = ActorPackExportOutcome(
            operation_id=1,
            result=ActorPackExportResult(
                archive_sha256="0" * 64,
                committed=True,
                durability="durable",
            ),
        )

    def create_request(self, **kwargs):
        request = SimpleNamespace(**kwargs)
        self.requests.append(request)
        return request


class _DialogApp:
    def __init__(self, responses: list[object], callback=None) -> None:
        self.responses = responses
        self.callback = callback
        self.screens: list[object] = []

    async def push_screen_wait(self, screen):
        self.screens.append(screen)
        if self.callback is not None:
            self.callback()
            self.callback = None
        return self.responses.pop(0)


class _DialogScreen:
    def __init__(self, target: Path, *, responses: list[object], callback=None):
        self.service = _EligibilityService()
        self.controller = _WorkflowController()
        self.app_instance = SimpleNamespace(
            actor_pack_export_service=self.service,
            actor_pack_export_controller=self.controller,
        )
        self.app = _DialogApp(responses, callback)
        self.state = SimpleNamespace(
            runtime_source="local",
            selected_entity_kind="character",
            selected_entity_id="7",
            selected_entity_name="Portable",
        )
        self._persona_buddy_session_generation = 4
        self._edit_mode = "view"
        self._io_dialog_active = True
        self._actor_pack_export_operation = None
        self._actor_pack_export_authority = None
        self.notifications: list[tuple[str, str]] = []
        self.target = target

    def _notify(self, message: str, severity: str) -> None:
        self.notifications.append((message, severity))

    def _actor_pack_export_preflight_is_current(self, authority):
        return PersonasScreen._actor_pack_export_preflight_is_current(self, authority)

    def _start_actor_pack_export(self, request, authority=None):
        return PersonasScreen._start_actor_pack_export(self, request, authority)

    def _actor_pack_export_authority_is_current(self, authority):
        return PersonasScreen._actor_pack_export_authority_is_current(self, authority)

    async def _wait_actor_pack_export(self, operation, authority):
        return await PersonasScreen._wait_actor_pack_export(self, operation, authority)

    def authority(self):
        return PersonasScreen._capture_actor_pack_export_ui_authority(
            self,
            source="local",
            actor_kind="character",
            local_actor_id="7",
            actor_revision=4,
        )


@pytest.mark.asyncio
async def test_overwrite_confirmation_precedes_destination_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "portable.tldw-actor-pack"
    target.write_bytes(b"old")
    screen = _DialogScreen(target, responses=[target, False])
    authority = screen.authority()
    assert authority is not None
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.personas_screen.capture_actor_pack_destination",
        lambda _destination: pytest.fail("destination captured before confirmation"),
    )

    await PersonasScreen._actor_pack_export_dialog_worker(screen, authority)

    assert len(screen.app.screens) == 2
    assert screen.controller.requests == []
    assert target.read_bytes() == b"old"


@pytest.mark.asyncio
async def test_selection_aba_during_picker_cannot_start_export(tmp_path: Path) -> None:
    target = tmp_path / "portable"
    screen = _DialogScreen(
        target,
        responses=[target],
        callback=lambda: setattr(screen, "_persona_buddy_session_generation", 6),
    )
    authority = screen.authority()
    assert authority is not None

    await PersonasScreen._actor_pack_export_dialog_worker(screen, authority)

    assert screen.service.calls == 0
    assert screen.controller.requests == []


@pytest.mark.asyncio
async def test_actor_revision_change_after_picker_cannot_start_export(
    tmp_path: Path,
) -> None:
    target = tmp_path / "portable"
    screen = _DialogScreen(target, responses=[target])
    screen.service.capture_eligibility = lambda *_args, **_kwargs: SimpleNamespace(
        actor_revision=5
    )
    authority = screen.authority()
    assert authority is not None

    await PersonasScreen._actor_pack_export_dialog_worker(screen, authority)

    assert screen.controller.requests == []


@pytest.mark.asyncio
async def test_duplicate_dialog_admission_is_ignored() -> None:
    class _DuplicateScreen:
        _io_dialog_active = True
        _actor_pack_export_operation = None

        def _capture_actor_pack_export_ui_authority(self, **_kwargs):
            pytest.fail("duplicate export reached authority capture")

    message = ActorPackExportRequested(
        actor_kind="character",
        source="local",
        local_actor_id="7",
        actor_revision=4,
    )

    await PersonasScreen._handle_actor_pack_export(_DuplicateScreen(), message)


@pytest.mark.asyncio
async def test_successful_dialog_normalizes_suffix_and_reports_only_basename(
    tmp_path: Path,
) -> None:
    target = tmp_path / "portable"
    screen = _DialogScreen(target, responses=[target])
    authority = screen.authority()
    assert authority is not None

    await PersonasScreen._actor_pack_export_dialog_worker(screen, authority)

    request = screen.controller.requests[0]
    assert request.destination.destination == target.with_name(
        "portable.tldw-actor-pack"
    )
    assert screen.notifications == [
        ("Exported portable.tldw-actor-pack.", "information")
    ]
    assert str(tmp_path) not in screen.notifications[0][0]

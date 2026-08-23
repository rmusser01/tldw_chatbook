from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_chatbook.Actor_Packs.import_controller import ActorPackImportOutcome
from tldw_chatbook.UI.Screens import personas_screen as screen_module
from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen


class _Review:
    allowed_actions = ("create_new", "create_copy", "update_existing")


class _Controller:
    def __init__(self) -> None:
        self.review = _Review()
        self.started: list[tuple[object, ...]] = []
        self.discarded: list[object] = []

    def create_request(self, path: Path) -> object:
        self.started.append(("request", path))
        return object()

    def start_inspection(self, request: object) -> int:
        self.started.append(("inspect", request))
        return 1

    def start_activation(self, review: object, action: str) -> int:
        self.started.append(("activate", review, action))
        return 2

    async def wait(self, operation: int) -> ActorPackImportOutcome:
        if operation == 1:
            return ActorPackImportOutcome(1, review=self.review)
        return ActorPackImportOutcome(
            2,
            result=SimpleNamespace(
                actor_kind="persona",
                local_actor_id="local-persona-imported",
                portable_uuid="123e4567-e89b-42d3-a456-426614174000",
            ),
        )

    def discard_review(self, review: object) -> bool:
        self.discarded.append(review)
        return True


class _Importer:
    def read_portrait_preview(self, review: object) -> object:
        assert isinstance(review, _Review)
        return object()


class _DialogApp:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.screens: list[object] = []

    async def push_screen_wait(self, screen: object) -> object:
        self.screens.append(screen)
        return self.responses.pop(0)


class _DialogScreen:
    def __init__(self, responses: list[object]) -> None:
        self.app = _DialogApp(responses)
        self.state = SimpleNamespace(active_mode="characters")
        self._actor_pack_import_operation = None
        self._actor_pack_import_review = None
        self._io_dialog_active = True
        self.notifications: list[tuple[str, str]] = []

    def _notify(self, message: str, severity: str) -> None:
        self.notifications.append((message, severity))


@pytest.mark.asyncio
async def test_picker_review_and_activation_use_one_controller_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "incoming.tldw-actor-pack"
    controller = _Controller()
    importer = _Importer()
    screen = _DialogScreen([archive, "create_new"])
    monkeypatch.setattr(screen_module, "ActorPackImportReview", _Review)
    monkeypatch.setattr(
        screen_module,
        "ActorPackImportReviewDialog",
        lambda review, preview: ("review", review, preview),
    )

    await PersonasScreen._actor_pack_import_dialog_worker(screen, controller, importer)

    assert controller.started[0] == ("request", archive.resolve())
    assert controller.started[-1] == ("activate", controller.review, "create_new")
    assert controller.discarded == []
    assert screen.notifications[-1] == ("Actor Pack activated.", "information")
    assert screen._io_dialog_active is False


@pytest.mark.asyncio
async def test_update_requires_second_confirmation_and_cancel_releases_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "incoming.tldw-actor-pack"
    controller = _Controller()
    importer = _Importer()
    screen = _DialogScreen([archive, "update_existing", False])
    monkeypatch.setattr(screen_module, "ActorPackImportReview", _Review)
    monkeypatch.setattr(
        screen_module,
        "ActorPackImportReviewDialog",
        lambda review, preview: ("review", review, preview),
    )

    await PersonasScreen._actor_pack_import_dialog_worker(screen, controller, importer)

    assert not any(item[0] == "activate" for item in controller.started)
    assert controller.discarded == [controller.review]
    assert len(screen.app.screens) == 3

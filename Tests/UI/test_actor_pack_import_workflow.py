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
                sections=("shared-visual-identity", "persona-runtime"),
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
        self.screen_stack: list[object] = []
        self.buddy_reconciliations = 0

    async def push_screen_wait(self, screen: object) -> object:
        self.screens.append(screen)
        return self.responses.pop(0)

    async def reconcile_persona_buddy_view(self) -> bool:
        self.buddy_reconciliations += 1
        return True


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

    async def _refresh_profile_rows_worker(self) -> None:
        return None

    async def _refresh_after_actor_pack_activation(
        self, result: object
    ) -> tuple[str, ...]:
        return await PersonasScreen._refresh_after_actor_pack_activation(self, result)


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


@pytest.mark.asyncio
async def test_committed_import_is_not_reported_failed_when_one_refresh_breaks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "incoming.tldw-actor-pack"
    controller = _Controller()
    importer = _Importer()
    screen = _DialogScreen([archive, "create_new"])
    screen.state.active_mode = "personas"
    invalidated: list[tuple[str, ...]] = []

    async def broken_profile_refresh() -> None:
        raise RuntimeError("presentation only")

    class _Session:
        async def invalidate_visual_identity_actor(
            self, actor_kind: str, actor_id: str
        ) -> None:
            invalidated.append(("shared", actor_kind, actor_id))

        async def invalidate_persona_visual_identity(self, persona_id: str) -> None:
            invalidated.append(("runtime", persona_id))

    screen._refresh_profile_rows_worker = broken_profile_refresh
    screen.app.screen_stack = [SimpleNamespace(_session=_Session())]
    monkeypatch.setattr(screen_module, "ActorPackImportReview", _Review)
    monkeypatch.setattr(
        screen_module,
        "ActorPackImportReviewDialog",
        lambda review, preview: ("review", review, preview),
    )

    await PersonasScreen._actor_pack_import_dialog_worker(screen, controller, importer)

    assert invalidated == [
        ("shared", "persona", "local-persona-imported"),
        ("runtime", "local-persona-imported"),
    ]
    assert screen.notifications[-1] == (
        "Actor Pack activated. Some views could not refresh.",
        "information",
    )
    assert screen.app.buddy_reconciliations == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("actor_kind", "actor_id"),
    [("character", "42"), ("persona", "local-persona-minimal")],
)
async def test_minimal_pack_activation_invalidates_portrait_consumers(
    actor_kind: str, actor_id: str
) -> None:
    screen = _DialogScreen([])
    screen.state.active_mode = "personas" if actor_kind == "character" else "characters"
    invalidated: list[tuple[str, str]] = []

    class _Session:
        async def invalidate_visual_identity_actor(
            self, committed_kind: str, committed_id: str
        ) -> None:
            invalidated.append((committed_kind, committed_id))

    screen.app.screen_stack = [SimpleNamespace(_session=_Session())]

    errors = await screen._refresh_after_actor_pack_activation(
        SimpleNamespace(actor_kind=actor_kind, local_actor_id=actor_id, sections=())
    )

    assert errors == ()
    assert invalidated == [(actor_kind, actor_id)]


@pytest.mark.asyncio
async def test_stale_activation_returns_to_a_fresh_review_without_repicking_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _StaleThenController(_Controller):
        def __init__(self) -> None:
            super().__init__()
            self._next_operation = 0
            self._activation_count = 0

        def start_inspection(self, request: object) -> int:
            self.started.append(("inspect", request))
            self.review = _Review()
            self._next_operation += 1
            return self._next_operation

        def start_activation(self, review: object, action: str) -> int:
            self.started.append(("activate", review, action))
            self._activation_count += 1
            self._next_operation += 1
            return self._next_operation

        async def wait(self, operation: int) -> ActorPackImportOutcome:
            if operation % 2:
                return ActorPackImportOutcome(operation, review=self.review)
            if self._activation_count == 1:
                return ActorPackImportOutcome(
                    operation,
                    error_category="actor_pack_import_review_stale",
                )
            return await super().wait(2)

    archive = tmp_path / "incoming.tldw-actor-pack"
    controller = _StaleThenController()
    screen = _DialogScreen([archive, "create_new", "create_new"])
    monkeypatch.setattr(screen_module, "ActorPackImportReview", _Review)
    monkeypatch.setattr(
        screen_module,
        "ActorPackImportReviewDialog",
        lambda review, preview: ("review", review, preview),
    )

    await PersonasScreen._actor_pack_import_dialog_worker(
        screen, controller, _Importer()
    )

    assert sum(item[0] == "request" for item in controller.started) == 2
    assert sum(item[0] == "inspect" for item in controller.started) == 2
    assert sum(item[0] == "activate" for item in controller.started) == 2
    assert len(controller.discarded) == 1
    assert screen.notifications[-1] == ("Actor Pack activated.", "information")

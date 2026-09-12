"""Focused Pilot coverage for the generated-video capacity choice modal."""

from __future__ import annotations

import asyncio
import inspect
import os
import shutil
import tempfile
import threading
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import cast

import pytest

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.widgets import Button, Static

from tldw_chatbook.Widgets.Console.console_video_capacity_modal import (
    CapacityAction,
    CapacityReason,
    ConsoleVideoCapacityModal,
)
from tldw_chatbook.Widgets.cancel_confirmation_dialog import (
    CancelConfirmationDialog,
)
from tldw_chatbook.Chat.console_generate_video import PendingVideoArtifact
from tldw_chatbook.UI.Console_Modules.video import ConsoleVideoController
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Video_Generation.video_metadata import VideoGenerationMetadata
from tldw_chatbook.Video_Generation.video_store import (
    VideoPublicationGate,
    VideoStore,
    VideoStoreSaveError,
)


class _ModalHost(ConsolidatedCSSApp):
    """Small real Textual host used to exercise modal interaction."""


def _button_label(button: Button) -> str:
    return button.label.plain


def _static_text(widget: Static) -> str:
    renderable = widget.renderable
    return getattr(renderable, "plain", str(renderable))


async def _mount_modal(
    app: _ModalHost,
    *,
    reason: str,
    results: list[CapacityAction],
) -> ConsoleVideoCapacityModal:
    modal = ConsoleVideoCapacityModal(
        reason=cast(CapacityReason, reason),
        size_bytes=3 * 1024 * 1024 + 512 * 1024,
        max_bytes=2 * 1024 * 1024,
    )
    await app.push_screen(modal, callback=results.append)
    return modal


@pytest.mark.asyncio
async def test_over_capacity_modal_has_exact_choices_and_safe_size_copy() -> None:
    app = _ModalHost()
    results: list[CapacityAction] = []
    async with app.run_test(size=(120, 40)) as pilot:
        modal = await _mount_modal(app, reason="over_capacity", results=results)
        await pilot.pause()

        assert _button_label(modal.query_one("#video-capacity-keep", Button)) == (
            "Keep here (remove other videos)"
        )
        assert _button_label(modal.query_one("#video-capacity-save", Button)) == (
            "Save to disk"
        )
        assert _button_label(modal.query_one("#video-capacity-discard", Button)) == (
            "Discard"
        )
        copy = " ".join(
            _static_text(widget)
            for widget in modal.query(
                "#video-capacity-summary, #video-capacity-guidance"
            )
            if isinstance(widget, Static)
        )
        assert "3.5 MiB" in copy
        assert "2.0 MiB" in copy
        assert "generated video" in copy.lower()


@pytest.mark.asyncio
async def test_store_failure_modal_has_exact_choices_without_capacity_claim() -> None:
    app = _ModalHost()
    results: list[CapacityAction] = []
    async with app.run_test(size=(120, 40)) as pilot:
        modal = await _mount_modal(app, reason="store_failure", results=results)
        await pilot.pause()

        assert _button_label(modal.query_one("#video-capacity-keep", Button)) == "Retry"
        assert _button_label(modal.query_one("#video-capacity-save", Button)) == (
            "Save to disk"
        )
        assert _button_label(modal.query_one("#video-capacity-discard", Button)) == (
            "Discard"
        )
        guidance = _static_text(
            modal.query_one("#video-capacity-guidance", Static)
        ).lower()
        assert "could not be stored" in guidance
        assert "exceeds" not in guidance


@pytest.mark.parametrize(
    ("reason", "button_id", "expected"),
    [
        ("over_capacity", "video-capacity-keep", "keep"),
        ("store_failure", "video-capacity-keep", "keep"),
        ("over_capacity", "video-capacity-save", "save_external"),
        ("store_failure", "video-capacity-discard", "discard"),
    ],
)
@pytest.mark.asyncio
async def test_modal_buttons_dismiss_with_typed_actions(
    reason: str,
    button_id: str,
    expected: CapacityAction,
) -> None:
    app = _ModalHost()
    results: list[CapacityAction] = []
    async with app.run_test(size=(120, 40)) as pilot:
        await _mount_modal(app, reason=reason, results=results)
        await pilot.pause()
        await pilot.click(f"#{button_id}")
        await pilot.pause()

    assert results == [expected]


@pytest.mark.parametrize(
    ("reason", "expected_focus", "expected_result"),
    [
        ("over_capacity", "video-capacity-save", "save_external"),
        ("store_failure", "video-capacity-keep", "keep"),
    ],
)
@pytest.mark.asyncio
async def test_modal_enter_uses_reason_specific_safe_default(
    reason: str,
    expected_focus: str,
    expected_result: CapacityAction,
) -> None:
    app = _ModalHost()
    results: list[CapacityAction] = []
    async with app.run_test(size=(120, 40)) as pilot:
        modal = await _mount_modal(app, reason=reason, results=results)
        await pilot.pause()

        assert modal.focused is not None
        assert modal.focused.id == expected_focus
        assert isinstance(modal.focused, Button)
        assert modal.focused.variant == "primary"
        other_id = (
            "video-capacity-keep"
            if expected_focus == "video-capacity-save"
            else "video-capacity-save"
        )
        assert modal.query_one(f"#{other_id}", Button).variant == "default"
        await pilot.press("enter")
        await pilot.pause()

    assert results == [expected_result]


def test_modal_rejects_unknown_reason_without_reflecting_private_value() -> None:
    private_reason = "PRIVATE-PATH:/Users/alice/generated.mp4"

    with pytest.raises(ValueError) as raised:
        ConsoleVideoCapacityModal(
            reason=cast(CapacityReason, private_reason),
            size_bytes=1,
            max_bytes=1,
        )

    assert private_reason not in str(raised.value)


@pytest.mark.asyncio
async def test_modal_copy_is_plain_and_contains_no_private_sentinels() -> None:
    app = _ModalHost()
    results: list[CapacityAction] = []
    async with app.run_test(size=(120, 40)) as pilot:
        modal = await _mount_modal(app, reason="store_failure", results=results)
        await pilot.pause()

        copy_widgets = [
            modal.query_one("#video-capacity-summary", Static),
            modal.query_one("#video-capacity-guidance", Static),
        ]
        rendered = " ".join(_static_text(widget) for widget in copy_widgets)
        assert all(widget._render_markup is False for widget in copy_widgets)
        for sentinel in (
            "/Users/private/generated.mp4",
            "PRIVATE-PATH",
            "make the person identifiable",
            "message-id-123",
            "Traceback",
        ):
            assert sentinel not in rendered


@pytest.mark.asyncio
async def test_modal_widgets_fit_inside_ninety_by_fourteen_screen() -> None:
    app = _ModalHost()
    results: list[CapacityAction] = []
    async with app.run_test(size=(90, 14)) as pilot:
        modal = await _mount_modal(app, reason="over_capacity", results=results)
        await pilot.pause()

        dialog = modal.query_one("#video-capacity-dialog")
        widgets = [
            dialog,
            modal.query_one("#video-capacity-summary", Static),
            modal.query_one("#video-capacity-guidance", Static),
        ]
        widgets.extend(
            modal.query_one(f"#{button_id}", Button)
            for button_id in (
                "video-capacity-keep",
                "video-capacity-save",
                "video-capacity-discard",
            )
        )
        for widget in widgets:
            assert widget.display
            assert widget.region.width > 0
            assert widget.region.height > 0
            assert 0 <= widget.region.x < widget.region.right <= modal.size.width
            assert 0 <= widget.region.y < widget.region.bottom <= modal.size.height

        for button in widgets[-3:]:
            assert isinstance(button, Button)
            assert dialog.region.x <= button.region.x
            assert button.region.right <= dialog.region.right
            assert button.region.width >= len(_button_label(button)) + 2


class _TrackingStream:
    def __init__(self, payload: bytes) -> None:
        self._stream = tempfile.TemporaryFile(mode="w+b")
        self._stream.write(payload)
        self._stream.seek(0)
        self.close_calls = 0

    @property
    def closed(self) -> bool:
        return self._stream.closed

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._stream.seek(offset, whence)

    def close(self) -> None:
        self.close_calls += 1
        self._stream.close()


def _artifact(
    payload: bytes = b"generated-video",
    *,
    reason: str = "over_capacity",
    message_id: str = "pending-message",
    extension: str = "mp4",
) -> PendingVideoArtifact:
    return PendingVideoArtifact(
        metadata=VideoGenerationMetadata(
            name="generated-clip",
            prompt="private generation prompt",
            backend="comfyui",
            **({"container": extension} if extension != "mp4" else {}),
        ),
        message_id=message_id,
        slug="generated-clip",
        extension=extension,
        size_bytes=len(payload),
        max_bytes=1024 * 1024,
        reason=cast(CapacityReason, reason),
        stream=_TrackingStream(payload),
    )


@pytest.mark.parametrize(
    ("reason", "request_source", "cancel_guard_with", "expected_focus"),
    [
        ("over_capacity", "escape", "continue", "video-capacity-save"),
        ("store_failure", "backdrop", "backdrop", "video-capacity-keep"),
    ],
)
@pytest.mark.asyncio
async def test_generic_capacity_dismissal_guards_real_staged_artifact(
    reason: str,
    request_source: str,
    cancel_guard_with: str,
    expected_focus: str,
) -> None:
    """Retain a staged artifact until discard is explicitly confirmed.

    Args:
        reason: Capacity outcome presented by the modal.
        request_source: Escape or backdrop dismissal source.
        cancel_guard_with: Action used to cancel the nested discard guard.
        expected_focus: Control that must regain focus after cancellation.
    """
    from Tests.UI.app_factory import _build_test_app

    class TrackingChatScreen(ChatScreen):
        capacity_results: list[object]

        def __init__(self, app_instance) -> None:
            super().__init__(app_instance)
            self.capacity_results = []

        async def _wait_for_console_screen_result(self, screen):
            result = await super()._wait_for_console_screen_result(screen)
            if isinstance(screen, ConsoleVideoCapacityModal):
                self.capacity_results.append(result)
            return result

    class Host(ConsolidatedCSSApp):
        def __init__(self, app_instance) -> None:
            super().__init__()
            self.app_instance = app_instance
            self.pending_handoffs = app_instance.pending_handoffs

        async def on_mount(self) -> None:
            await self.push_screen(TrackingChatScreen(self.app_instance))

    artifact = _artifact(reason=reason, message_id=f"guarded-{reason}")
    host = Host(_build_test_app())
    async with host.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = host.screen
        assert isinstance(screen, TrackingChatScreen)
        screen.app_instance = host
        worker = screen.run_worker(
            screen._video._resolve_generated_video_outcome(
                artifact,
                session_id="session",
                message_id=artifact.message_id,
            ),
            exclusive=False,
            exit_on_error=False,
        )
        await pilot.pause()
        capacity_modal = host.screen
        assert isinstance(capacity_modal, ConsoleVideoCapacityModal)
        other_focus = (
            "video-capacity-keep"
            if expected_focus == "video-capacity-save"
            else "video-capacity-save"
        )
        capacity_modal.query_one(f"#{other_focus}", Button).focus()
        await pilot.pause()
        assert capacity_modal.focused is not None
        assert capacity_modal.focused.id == other_focus

        if request_source == "escape":
            await pilot.press("escape")
        else:
            await pilot.click(offset=(0, 0))
        await pilot.pause()

        assert isinstance(host.screen, CancelConfirmationDialog)
        assert screen._video._owns_pending_console_video(artifact)
        assert not artifact.stream.closed

        await capacity_modal.request_safe_cancel(source="escape")
        await capacity_modal.request_safe_cancel(source="backdrop")
        await pilot.pause()
        guard_count = sum(
            isinstance(candidate, CancelConfirmationDialog)
            for candidate in host.screen_stack
        )
        assert guard_count == 1

        if cancel_guard_with == "continue":
            await pilot.click("#continue-btn")
        else:
            await pilot.click(offset=(0, 0))
        await pilot.pause()

        assert host.screen is capacity_modal
        assert capacity_modal.focused is not None
        assert capacity_modal.focused.id == expected_focus
        assert screen._video._owns_pending_console_video(artifact)
        assert not artifact.stream.closed

        await capacity_modal.request_safe_cancel(source="escape")
        await pilot.pause()
        guard = host.screen
        assert isinstance(guard, CancelConfirmationDialog)
        guard.dismiss(None)
        await pilot.pause()

        assert host.screen is capacity_modal
        assert screen._video._owns_pending_console_video(artifact)
        assert not artifact.stream.closed

        await capacity_modal.request_safe_cancel(source="backdrop")
        await pilot.pause()
        assert isinstance(host.screen, CancelConfirmationDialog)
        await pilot.click("#cancel-btn")
        await worker.wait()
        await pilot.pause()

        assert host.screen is screen
        assert screen.capacity_results == ["discard"]
        assert screen._video._pending_console_video_artifacts() == {}
        assert artifact.stream.closed
        assert cast(_TrackingStream, artifact.stream).close_calls == 1


class _OutcomeHarness:
    """Small owner that exercises the production resolver without a full app."""

    def __init__(self, *, actions: list[object], video_store: object) -> None:
        self.actions = list(actions)
        self.video_store = video_store
        self.appended: list[tuple] = []
        self.sync_count = 0
        self.waited_screens: list[object] = []
        self.opened: list[Path] = []
        self.notifications: list[tuple[str, str | None]] = []
        self.chat_store = SimpleNamespace(append_video_message=self._append)
        self.app_instance = SimpleNamespace(notify=self._notify)

    def _append(self, *args, **kwargs) -> None:
        self.appended.append((args, kwargs))

    def _notify(self, message: str, *, severity: str | None = None) -> None:
        self.notifications.append((message, severity))

    def _ensure_console_chat_store(self):
        return self.chat_store

    def _ensure_console_video_store(self):
        return self.video_store

    async def _sync_native_console_chat_ui(self) -> None:
        self.sync_count += 1

    async def _wait_for_console_screen_result(self, screen):
        self.waited_screens.append(screen)
        return self.actions.pop(0)

    def _open_video_with_os(self, path: Path) -> None:
        self.opened.append(path)

    async def _resolve_generated_video_outcome(self, *args, **kwargs):
        return await ConsoleVideoController._resolve_generated_video_outcome(
            self, *args, **kwargs
        )

    def _pending_console_video_artifacts(self):
        return ConsoleVideoController._pending_console_video_artifacts(self)

    def _owns_pending_console_video(self, artifact):
        return ConsoleVideoController._owns_pending_console_video(self, artifact)

    def _drain_pending_console_videos(self):
        return ConsoleVideoController._drain_pending_console_videos(self)

    def _retry_pending_console_video(self, artifact, **kwargs):
        return ConsoleVideoController._retry_pending_console_video(
            self, artifact, **kwargs
        )

    async def _save_pending_console_video_external(self, artifact):
        return await ConsoleVideoController._save_pending_console_video_external(
            self, artifact
        )

    def _begin_pending_console_video_operation(self, artifact):
        return ConsoleVideoController._begin_pending_console_video_operation(
            self, artifact
        )

    def _end_pending_console_video_operation(self, artifact):
        return ConsoleVideoController._end_pending_console_video_operation(
            self, artifact
        )

    async def _run_pending_console_video_operation(
        self, artifact, function, *args, **kwargs
    ):
        return await ConsoleVideoController._run_pending_console_video_operation(
            self, artifact, function, *args, **kwargs
        )

    def _close_pending_console_video(self, artifact):
        return ConsoleVideoController._close_pending_console_video(artifact)

    _external_video_target_identity = staticmethod(
        ConsoleVideoController._external_video_target_identity
    )
    _copy_pending_video_external = staticmethod(
        ConsoleVideoController._copy_pending_video_external
    )


@pytest.mark.asyncio
async def test_normal_result_appends_and_syncs_through_shared_resolver(
    tmp_path: Path,
) -> None:
    harness = _OutcomeHarness(actions=[], video_store=object())
    metadata = VideoGenerationMetadata(name="ready", prompt="prompt", backend="comfyui")

    await harness._resolve_generated_video_outcome(
        (metadata, tmp_path / "ready.mp4"),
        session_id="session",
        message_id="message",
    )

    assert harness.appended == [
        (
            ("session",),
            {"video_metadata": metadata, "persist": True, "message_id": "message"},
        )
    ]
    assert harness.sync_count == 1


@pytest.mark.asyncio
async def test_normal_commit_winning_before_unmount_persists_without_ui_sync(
    tmp_path: Path,
) -> None:
    harness = _OutcomeHarness(actions=[], video_store=object())
    metadata = VideoGenerationMetadata(
        name="committed", prompt="prompt", backend="comfyui"
    )
    harness._drain_pending_console_videos()

    await harness._resolve_generated_video_outcome(
        (metadata, tmp_path / "committed.mp4"),
        session_id="session",
        message_id="committed-message",
    )

    assert harness.appended == [
        (
            ("session",),
            {
                "video_metadata": metadata,
                "persist": True,
                "message_id": "committed-message",
            },
        )
    ]
    assert harness.sync_count == 0


@pytest.mark.asyncio
async def test_pending_discard_registers_then_closes_without_card() -> None:
    artifact = _artifact()
    harness = _OutcomeHarness(actions=["discard"], video_store=object())

    await harness._resolve_generated_video_outcome(
        artifact, session_id="session", message_id=artifact.message_id
    )

    assert harness.appended == []
    assert harness.sync_count == 0
    assert harness._pending_console_video_artifacts() == {}
    assert artifact.stream.closed
    assert cast(_TrackingStream, artifact.stream).close_calls == 1


@pytest.mark.asyncio
async def test_initial_dispatch_resolves_pending_discard_and_clears_bookkeeping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.UI.Console_Modules import video as video_controller_module
    from tldw_chatbook.Video_Generation import adapter_registry

    artifact = _artifact()
    harness = _OutcomeHarness(actions=["discard"], video_store=object())
    harness.chat_store.workspace_context = SimpleNamespace(
        active_workspace_id="workspace"
    )
    harness.chat_store.ensure_session = lambda **_kwargs: SimpleNamespace(id="session")
    harness._default_console_session_settings = lambda: object()
    harness._append_native_console_system_message = lambda *_args, **_kwargs: (
        _completed_async()
    )
    harness._console_composer_or_none = lambda: None
    harness._clear_console_composer_draft = lambda: None
    inflight: set[str] = set()
    cancels: dict = {}
    harness._console_videogen_inflight_sessions = lambda: inflight
    harness._console_videogen_cancel_events = lambda: cancels

    class Registry:
        @staticmethod
        def resolve_backend(_backend):
            return object()

    async def fake_to_thread(_function, **_kwargs):
        artifact.message_id = _kwargs["message_id"]
        return artifact

    monkeypatch.setattr(
        video_controller_module,
        "get_video_generation_config",
        lambda: SimpleNamespace(default_backend="comfyui", confirm_cost_estimate=False),
    )
    monkeypatch.setattr(adapter_registry, "get_registry", lambda: Registry())
    monkeypatch.setattr("asyncio.to_thread", fake_to_thread)

    await ConsoleVideoController._console_command_generate_video(
        harness, SimpleNamespace(args="a generated video")
    )

    assert harness.appended == []
    assert len(harness.waited_screens) == 1
    assert isinstance(harness.waited_screens[0], ConsoleVideoCapacityModal)
    assert harness._pending_console_video_artifacts() == {}
    assert artifact.stream.closed
    assert cast(_TrackingStream, artifact.stream).close_calls == 1
    assert inflight == set()
    assert cancels == {}


async def _completed_async() -> None:
    return None


@pytest.mark.asyncio
async def test_over_capacity_keep_adopts_real_store_as_sole_file_and_then_appends(
    tmp_path: Path,
) -> None:
    config = SimpleNamespace(max_store_mb=1, retention="ttl", retention_ttl_hours=24)
    store = VideoStore(root=tmp_path / "videos", config=config)
    old = store.save("old-message", "old", b"o" * (700 * 1024), extension="mp4")
    second_old = store.save(
        "second-old", "second", b"s" * (200 * 1024), extension="mp4"
    )
    assert isinstance(old, Path)
    assert isinstance(second_old, Path)
    old_messages = [
        SimpleNamespace(
            id="old-card",
            persisted_message_id="old-message",
            video_metadata=VideoGenerationMetadata(
                name="old", prompt="old", backend="comfyui"
            ),
        ),
        SimpleNamespace(
            id="second-card",
            persisted_message_id="second-old",
            video_metadata=VideoGenerationMetadata(
                name="second", prompt="second", backend="comfyui"
            ),
        ),
    ]
    spec_owner = SimpleNamespace(
        _ensure_console_video_store=lambda: store,
        _video_storage_message_id=ConsoleVideoController._video_storage_message_id,
    )
    before_specs = ConsoleVideoController._build_video_card_specs(
        spec_owner, old_messages
    )
    assert {spec.status for spec in before_specs.values()} == {"ready"}
    payload = b"n" * (1024 * 1024 + 17)
    artifact = _artifact(payload, extension="webm")
    harness = _OutcomeHarness(actions=["keep"], video_store=store)

    await harness._resolve_generated_video_outcome(
        artifact, session_id="session", message_id=artifact.message_id
    )

    stored = list(store.iter_stored())
    assert [(item.message_id, item.slug, item.size_bytes) for item in stored] == [
        (artifact.message_id, artifact.slug, len(payload))
    ]
    assert stored[0].path.suffix == ".webm"
    assert not old.exists()
    assert not second_old.exists()
    after_specs = ConsoleVideoController._build_video_card_specs(
        spec_owner, old_messages
    )
    assert {spec.status for spec in after_specs.values()} == {"expired"}
    assert len(harness.appended) == 1
    assert harness.sync_count == 1
    assert harness._pending_video_operation_cancels == {}
    assert artifact.stream.closed


@pytest.mark.asyncio
async def test_adoption_failure_reoffers_while_artifact_remains_readable() -> None:
    class FailingStore:
        def adopt_oversized(self, *_args, **_kwargs):
            raise VideoStoreSaveError("PRIVATE-PATH")

    artifact = _artifact()
    harness = _OutcomeHarness(actions=["keep", "discard"], video_store=FailingStore())
    original_wait = harness._wait_for_console_screen_result

    async def checking_wait(self, screen):
        if len(self.waited_screens) == 1:
            artifact.rewind()
            assert artifact.stream.read() == b"generated-video"
        return await original_wait(screen)

    harness._wait_for_console_screen_result = MethodType(checking_wait, harness)

    await harness._resolve_generated_video_outcome(
        artifact, session_id="session", message_id=artifact.message_id
    )

    assert len(harness.waited_screens) == 2
    assert harness.appended == []
    assert any("try again" in message.lower() for message, _ in harness.notifications)
    assert all("PRIVATE-PATH" not in message for message, _ in harness.notifications)


@pytest.mark.asyncio
async def test_managed_adoption_appends_only_after_store_resolves_target(
    tmp_path: Path,
) -> None:
    artifact = _artifact()
    published = tmp_path / "looks-present.mp4"
    published.write_bytes(b"generated-video")

    class UnresolvedStore:
        def __init__(self) -> None:
            self.resolve_calls = 0

        def adopt_oversized(self, *_args, **_kwargs):
            return published

        def resolve(self, *_args, **_kwargs):
            self.resolve_calls += 1
            return None

    store = UnresolvedStore()
    harness = _OutcomeHarness(actions=["keep", "discard"], video_store=store)

    await harness._resolve_generated_video_outcome(
        artifact, session_id="session", message_id=artifact.message_id
    )

    assert store.resolve_calls == 1
    assert harness.appended == []
    assert len(harness.waited_screens) == 2


@pytest.mark.asyncio
async def test_unmount_during_managed_resolve_persists_without_sync(
    tmp_path: Path,
) -> None:
    artifact = _artifact()
    published = tmp_path / "published.mp4"
    published.write_bytes(b"generated-video")
    harness: _OutcomeHarness

    class SlowResolveStore:
        def adopt_oversized(self, *_args, **_kwargs):
            return published

        def resolve(self, *_args, **_kwargs):
            harness._drain_pending_console_videos()
            return published

    harness = _OutcomeHarness(actions=["keep"], video_store=SlowResolveStore())

    await harness._resolve_generated_video_outcome(
        artifact, session_id="session", message_id=artifact.message_id
    )

    assert len(harness.appended) == 1
    assert harness.sync_count == 0
    assert artifact.stream.closed


@pytest.mark.asyncio
async def test_store_failure_retry_uses_ordinary_save_not_adoption(
    tmp_path: Path,
) -> None:
    artifact = _artifact(reason="store_failure", extension="webm")

    class RetryStore:
        capacity_bytes = 1024 * 1024

        def __init__(self) -> None:
            self.saved: list[tuple] = []
            self.adopted = False

        def save(
            self,
            message_id,
            slug,
            content,
            *,
            extension,
            publication_gate,
        ):
            assert isinstance(publication_gate, VideoPublicationGate)
            self.saved.append((message_id, slug, content, extension))
            with publication_gate.claim_publication() as active:
                assert active
                path = tmp_path / "retried.webm"
                path.write_bytes(content)
            return path

        def resolve(self, *_args, **_kwargs):
            return tmp_path / "retried.webm"

        def adopt_oversized(self, *_args, **_kwargs):
            self.adopted = True
            raise AssertionError("retry must not adopt")

    store = RetryStore()
    harness = _OutcomeHarness(actions=["keep"], video_store=store)

    await harness._resolve_generated_video_outcome(
        artifact, session_id="session", message_id=artifact.message_id
    )

    assert store.saved == [
        (artifact.message_id, artifact.slug, b"generated-video", "webm")
    ]
    assert not store.adopted
    assert len(harness.appended) == 1
    assert artifact.stream.closed


@pytest.mark.asyncio
async def test_repeated_store_retry_failure_reoffers_without_losing_payload() -> None:
    artifact = _artifact(reason="store_failure")

    class BusyStore:
        def __init__(self) -> None:
            self.attempts = 0

        def save(self, *_args, **_kwargs):
            self.attempts += 1
            raise VideoStoreSaveError("PRIVATE-PATH")

    store = BusyStore()
    harness = _OutcomeHarness(actions=["keep", "keep", "discard"], video_store=store)

    await harness._resolve_generated_video_outcome(
        artifact, session_id="session", message_id=artifact.message_id
    )

    assert store.attempts == 2
    assert len(harness.waited_screens) == 3
    assert harness.appended == []
    assert artifact.stream.closed
    assert all("PRIVATE-PATH" not in message for message, _ in harness.notifications)


def test_external_new_target_commit_never_clobbers_concurrent_creator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _artifact(b"new payload")
    target = tmp_path / "chosen.mp4"
    real_link = os.link

    def racing_link(source, destination, **kwargs):
        target.write_bytes(b"concurrent payload")
        return real_link(source, destination, **kwargs)

    monkeypatch.setattr(os, "link", racing_link)
    monkeypatch.setattr(os, "supports_dir_fd", set(os.supports_dir_fd) | {racing_link})
    monkeypatch.setattr(
        os,
        "supports_follow_symlinks",
        set(os.supports_follow_symlinks) | {racing_link},
    )

    result = ConsoleVideoController._copy_pending_video_external(artifact, target, None)

    assert result == "confirm"
    assert target.read_bytes() == b"concurrent payload"
    artifact.rewind()
    assert artifact.stream.read() == b"new payload"
    assert not list(tmp_path.glob(".*.tmp"))
    artifact.close()


def test_external_confirmed_replacement_writes_exact_bytes(tmp_path: Path) -> None:
    artifact = _artifact(b"exact replacement")
    target = tmp_path / "chosen.mp4"
    target.write_bytes(b"old")
    identity = ConsoleVideoController._external_video_target_identity(target)

    result = ConsoleVideoController._copy_pending_video_external(
        artifact, target, identity
    )

    assert result == "saved"
    assert target.read_bytes() == b"exact replacement"
    assert not list(tmp_path.glob(".*.tmp"))
    artifact.close()


def test_external_changed_confirmed_identity_requires_fresh_confirmation(
    tmp_path: Path,
) -> None:
    artifact = _artifact(b"new")
    target = tmp_path / "chosen.mp4"
    target.write_bytes(b"first")
    identity = ConsoleVideoController._external_video_target_identity(target)
    target.write_bytes(b"changed after confirmation")

    result = ConsoleVideoController._copy_pending_video_external(
        artifact, target, identity
    )

    assert result == "confirm"
    assert target.read_bytes() == b"changed after confirmation"
    artifact.close()


def test_external_commit_error_keeps_existing_destination_and_cleans_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _artifact(b"new")
    target = tmp_path / "chosen.mp4"
    target.write_bytes(b"old")
    identity = ConsoleVideoController._external_video_target_identity(target)

    def fail_replace(_source, _target, *, src_dir_fd=None, dst_dir_fd=None):
        raise OSError("commit failed")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError):
        ConsoleVideoController._copy_pending_video_external(artifact, target, identity)

    assert target.read_bytes() == b"old"
    artifact.rewind()
    assert artifact.stream.read() == b"new"
    assert not list(tmp_path.glob(".*.tmp"))
    artifact.close()


def test_external_copy_error_keeps_existing_destination_and_cleans_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _artifact(b"new")
    target = tmp_path / "chosen.mp4"
    target.write_bytes(b"old")
    identity = ConsoleVideoController._external_video_target_identity(target)

    def fail_copy(*_args, **_kwargs):
        raise OSError("copy failed")

    monkeypatch.setattr(shutil, "copyfileobj", fail_copy)

    with pytest.raises(OSError):
        ConsoleVideoController._copy_pending_video_external(artifact, target, identity)

    assert target.read_bytes() == b"old"
    artifact.rewind()
    assert artifact.stream.read() == b"new"
    assert not list(tmp_path.glob(".*.tmp"))
    artifact.close()


def test_external_parent_swap_fails_closed_and_cleans_pinned_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intended_parent = tmp_path / "intended"
    moved_parent = tmp_path / "moved-original"
    attacker_parent = tmp_path / "attacker"
    intended_parent.mkdir()
    attacker_parent.mkdir()
    target = intended_parent / "chosen.mp4"
    artifact = _artifact(b"safe bytes")
    original_check = ConsoleVideoController._external_video_precommit_check
    swapped = False

    def swap_parent_then_check(*args, **kwargs):
        nonlocal swapped
        if not swapped:
            intended_parent.rename(moved_parent)
            try:
                intended_parent.symlink_to(attacker_parent, target_is_directory=True)
            except OSError:
                pytest.skip("directory symlinks are unavailable")
            swapped = True
        return original_check(*args, **kwargs)

    monkeypatch.setattr(
        ConsoleVideoController,
        "_external_video_precommit_check",
        staticmethod(swap_parent_then_check),
    )

    with pytest.raises(OSError):
        ConsoleVideoController._copy_pending_video_external(artifact, target, None)

    assert swapped
    assert not (attacker_parent / "chosen.mp4").exists()
    assert not (moved_parent / "chosen.mp4").exists()
    assert not list(attacker_parent.glob(".*.tmp"))
    assert not list(moved_parent.glob(".*.tmp"))
    artifact.rewind()
    assert artifact.stream.read() == b"safe bytes"
    artifact.close()


@pytest.mark.asyncio
async def test_external_save_fails_closed_when_pinned_primitives_are_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _artifact(b"retained bytes")
    target = tmp_path / "unsupported.mp4"
    harness = _OutcomeHarness(
        actions=["save_external", target, "discard"], video_store=object()
    )
    opened_paths = []
    real_open = os.open

    def tracking_open(path, *args, **kwargs):
        opened_paths.append(path)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", tracking_open)
    monkeypatch.setattr(
        os,
        "supports_dir_fd",
        set(os.supports_dir_fd) - {os.link},
    )

    await harness._resolve_generated_video_outcome(
        artifact, session_id="session", message_id=artifact.message_id
    )

    assert opened_paths == []
    assert not target.exists()
    assert not list(tmp_path.glob(".*.tmp"))
    assert len(harness.waited_screens) == 3
    assert harness.appended == []
    assert any(severity == "error" for _, severity in harness.notifications)
    assert artifact.stream.closed


def test_external_fdopen_failure_closes_stage_fd_and_cleans_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _artifact(b"retained")
    target = tmp_path / "fdopen.mp4"
    real_open = os.open
    staged_fds: list[int] = []

    def tracking_open(path, *args, **kwargs):
        fd = real_open(path, *args, **kwargs)
        if kwargs.get("dir_fd") is not None:
            staged_fds.append(fd)
        return fd

    def fail_fdopen(fd, *_args, **_kwargs):
        assert fd in staged_fds
        raise OSError("PRIVATE-FDOPEN")

    monkeypatch.setattr(os, "open", tracking_open)
    monkeypatch.setattr(
        os, "supports_dir_fd", set(os.supports_dir_fd) | {tracking_open}
    )
    monkeypatch.setattr(os, "fdopen", fail_fdopen)

    try:
        with pytest.raises(OSError):
            ConsoleVideoController._copy_pending_video_external(artifact, target, None)
        assert staged_fds
        for fd in staged_fds:
            with pytest.raises(OSError):
                os.fstat(fd)
        assert not target.exists()
        assert not list(tmp_path.glob(".*.tmp"))
    finally:
        for fd in staged_fds:
            try:
                os.close(fd)
            except OSError:
                pass
        artifact.close()


@pytest.mark.parametrize("existing_destination", [False, True])
def test_external_first_stage_fstat_failure_closes_raw_fd_and_owned_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_destination: bool,
) -> None:
    artifact = _artifact(b"retained after initial fstat")
    target = tmp_path / "initial-fstat.mp4"
    original_destination = b"existing destination"
    confirmed_identity = None
    if existing_destination:
        target.write_bytes(original_destination)
        confirmed_identity = ConsoleVideoController._external_video_target_identity(
            target
        )
    real_open = os.open
    real_close = os.close
    real_fstat = os.fstat
    staged_fds: list[int] = []
    staged_closes: list[int] = []
    staged_fstat_calls: dict[int, int] = {}
    logged: list[str] = []

    def tracking_open(path, *args, **kwargs):
        fd = real_open(path, *args, **kwargs)
        if kwargs.get("dir_fd") is not None:
            staged_fds.append(fd)
        return fd

    def fail_first_stage_fstat(fd):
        if fd in staged_fds:
            call_count = staged_fstat_calls.get(fd, 0) + 1
            staged_fstat_calls[fd] = call_count
            if call_count == 1:
                raise OSError("PRIVATE-FIRST-STAGE-FSTAT")
        return real_fstat(fd)

    def tracking_close(fd):
        if fd in staged_fds:
            staged_closes.append(fd)
        return real_close(fd)

    monkeypatch.setattr(os, "open", tracking_open)
    monkeypatch.setattr(
        os, "supports_dir_fd", set(os.supports_dir_fd) | {tracking_open}
    )
    monkeypatch.setattr(os, "fstat", fail_first_stage_fstat)
    monkeypatch.setattr(os, "close", tracking_close)
    sink_id = __import__("loguru").logger.add(logged.append, format="{message}")
    try:
        with pytest.raises(OSError, match="PRIVATE-FIRST-STAGE-FSTAT"):
            ConsoleVideoController._copy_pending_video_external(
                artifact, target, confirmed_identity
            )
    finally:
        __import__("loguru").logger.remove(sink_id)

    assert len(staged_fds) == 1
    assert staged_fstat_calls == {staged_fds[0]: 2}
    assert staged_closes == staged_fds
    with pytest.raises(OSError):
        real_fstat(staged_fds[0])
    assert not list(tmp_path.glob(".*.tmp"))
    if existing_destination:
        assert target.read_bytes() == original_destination
    else:
        assert not target.exists()
    artifact.rewind()
    assert artifact.stream.read() == b"retained after initial fstat"
    assert all("PRIVATE-FIRST-STAGE-FSTAT" not in message for message in logged)
    artifact.close()


@pytest.mark.parametrize("existing_destination", [False, True])
def test_external_partial_copy_failure_removes_owned_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_destination: bool,
) -> None:
    artifact = _artifact(b"complete retained payload")
    target = tmp_path / "partial-copy.mp4"
    original_destination = b"existing destination"
    confirmed_identity = None
    if existing_destination:
        target.write_bytes(original_destination)
        confirmed_identity = ConsoleVideoController._external_video_target_identity(
            target
        )
    logged: list[str] = []

    def fail_after_partial_copy(_source, destination, *_args, **_kwargs):
        destination.write(b"partial")
        destination.flush()
        raise OSError("PRIVATE-PARTIAL-COPY")

    monkeypatch.setattr(shutil, "copyfileobj", fail_after_partial_copy)
    sink_id = __import__("loguru").logger.add(logged.append, format="{message}")
    try:
        with pytest.raises(OSError, match="PRIVATE-PARTIAL-COPY"):
            ConsoleVideoController._copy_pending_video_external(
                artifact, target, confirmed_identity
            )
    finally:
        __import__("loguru").logger.remove(sink_id)

    assert not list(tmp_path.glob(".*.tmp"))
    if existing_destination:
        assert target.read_bytes() == original_destination
    else:
        assert not target.exists()
    artifact.rewind()
    assert artifact.stream.read() == b"complete retained payload"
    assert all("PRIVATE-PARTIAL-COPY" not in message for message in logged)
    artifact.close()


@pytest.mark.parametrize("existing_destination", [False, True])
def test_external_post_write_fstat_failure_removes_owned_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_destination: bool,
) -> None:
    artifact = _artifact(b"retained after write")
    target = tmp_path / "post-write-fstat.mp4"
    original_destination = b"existing destination"
    confirmed_identity = None
    if existing_destination:
        target.write_bytes(original_destination)
        confirmed_identity = ConsoleVideoController._external_video_target_identity(
            target
        )
    real_open = os.open
    real_fstat = os.fstat
    staged_fds: list[int] = []
    staged_fstat_calls: dict[int, int] = {}
    logged: list[str] = []

    def tracking_open(path, *args, **kwargs):
        fd = real_open(path, *args, **kwargs)
        if kwargs.get("dir_fd") is not None:
            staged_fds.append(fd)
        return fd

    def fail_post_write_fstat(fd):
        if fd in staged_fds:
            call_count = staged_fstat_calls.get(fd, 0) + 1
            staged_fstat_calls[fd] = call_count
            if call_count == 2:
                raise OSError("PRIVATE-POST-WRITE-FSTAT")
        return real_fstat(fd)

    monkeypatch.setattr(os, "open", tracking_open)
    monkeypatch.setattr(
        os, "supports_dir_fd", set(os.supports_dir_fd) | {tracking_open}
    )
    monkeypatch.setattr(os, "fstat", fail_post_write_fstat)
    sink_id = __import__("loguru").logger.add(logged.append, format="{message}")
    try:
        with pytest.raises(OSError, match="PRIVATE-POST-WRITE-FSTAT"):
            ConsoleVideoController._copy_pending_video_external(
                artifact, target, confirmed_identity
            )
    finally:
        __import__("loguru").logger.remove(sink_id)

    assert staged_fstat_calls == {staged_fds[0]: 2}
    assert not list(tmp_path.glob(".*.tmp"))
    if existing_destination:
        assert target.read_bytes() == original_destination
    else:
        assert not target.exists()
    artifact.rewind()
    assert artifact.stream.read() == b"retained after write"
    assert all("PRIVATE-POST-WRITE-FSTAT" not in message for message in logged)
    artifact.close()


def test_external_parent_fd_close_failure_does_not_mask_saved_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _artifact(b"saved")
    target = tmp_path / "saved.mp4"
    parent_fds: list[int] = []
    logged: list[str] = []
    real_open = os.open
    real_close = os.close

    def tracking_open(path, *args, **kwargs):
        fd = real_open(path, *args, **kwargs)
        if Path(path) == tmp_path and kwargs.get("dir_fd") is None:
            parent_fds.append(fd)
        return fd

    def close_then_fail(fd):
        real_close(fd)
        if fd in parent_fds:
            raise OSError("PRIVATE-PARENT-CLOSE")

    monkeypatch.setattr(os, "open", tracking_open)
    monkeypatch.setattr(
        os, "supports_dir_fd", set(os.supports_dir_fd) | {tracking_open}
    )
    monkeypatch.setattr(os, "close", close_then_fail)
    sink_id = __import__("loguru").logger.add(logged.append, format="{message}")
    try:
        result = ConsoleVideoController._copy_pending_video_external(
            artifact, target, None
        )
    finally:
        __import__("loguru").logger.remove(sink_id)

    assert result == "saved"
    assert target.read_bytes() == b"saved"
    for fd in parent_fds:
        with pytest.raises(OSError):
            os.fstat(fd)
    assert any("external_parent_close" in message for message in logged)
    assert any("OSError" in message for message in logged)
    assert all("PRIVATE-PARENT-CLOSE" not in message for message in logged)
    artifact.close()


@pytest.mark.asyncio
async def test_unmount_immediately_before_external_commit_creates_no_path_or_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _artifact(b"external", message_id="active-external")
    target = tmp_path / "external.mp4"
    harness = _OutcomeHarness(actions=["save_external", target], video_store=object())
    original_check = ConsoleVideoController._external_video_precommit_check

    def drain_then_check(*args, **kwargs):
        harness._drain_pending_console_videos()
        return original_check(*args, **kwargs)

    monkeypatch.setattr(
        ConsoleVideoController,
        "_external_video_precommit_check",
        staticmethod(drain_then_check),
    )

    await harness._resolve_generated_video_outcome(
        artifact, session_id="session", message_id=artifact.message_id
    )

    assert not target.exists()
    assert harness.appended == []
    assert harness.opened == []
    assert artifact.stream.closed
    assert cast(_TrackingStream, artifact.stream).close_calls == 1


@pytest.mark.asyncio
async def test_external_picker_cancel_discards_without_card(tmp_path: Path) -> None:
    artifact = _artifact(extension="webm")
    harness = _OutcomeHarness(actions=["save_external", None], video_store=object())

    await harness._resolve_generated_video_outcome(
        artifact, session_id="session", message_id=artifact.message_id
    )

    picker = harness.waited_screens[1]
    assert picker.default_filename == "generated-clip.webm"
    assert harness.appended == []
    assert harness.opened == []
    assert artifact.stream.closed


@pytest.mark.parametrize(
    ("selected_name", "expected_name"),
    [("chosen", "chosen.webm"), ("chosen.webm", "chosen.webm")],
)
@pytest.mark.asyncio
async def test_external_picker_normalizes_only_missing_or_exact_webm_suffix(
    tmp_path: Path, selected_name: str, expected_name: str
) -> None:
    artifact = _artifact(b"webm bytes", extension="webm")
    selected = tmp_path / selected_name
    expected = tmp_path / expected_name
    harness = _OutcomeHarness(actions=["save_external", selected], video_store=object())

    await harness._resolve_generated_video_outcome(
        artifact, session_id="session", message_id=artifact.message_id
    )

    assert expected.read_bytes() == b"webm bytes"
    assert harness.opened == [expected]
    assert artifact.stream.closed


@pytest.mark.parametrize("bad_suffix", ["mp4", "mov", "WEBM", "txt"])
@pytest.mark.asyncio
async def test_external_picker_reprompts_for_any_nonmatching_suffix(
    tmp_path: Path, bad_suffix: str
) -> None:
    artifact = _artifact(b"webm bytes", extension="webm")
    bad_target = tmp_path / f"chosen.{bad_suffix}"
    good_target = tmp_path / "accepted.webm"
    harness = _OutcomeHarness(
        actions=["save_external", bad_target, good_target], video_store=object()
    )

    await harness._resolve_generated_video_outcome(
        artifact, session_id="session", message_id=artifact.message_id
    )

    pickers = [
        screen
        for screen in harness.waited_screens
        if screen.__class__.__name__ == "EnhancedFileSave"
    ]
    assert len(pickers) == 2
    assert not bad_target.exists()
    assert good_target.read_bytes() == b"webm bytes"
    assert any(
        "generated video format" in message for message, _ in harness.notifications
    )


def test_external_copy_rejects_mismatched_suffix_before_any_filesystem_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _artifact(b"webm bytes", extension="webm")
    parent = tmp_path / "absent-parent"
    target = parent / "chosen.mp4"
    actions: list[str] = []

    monkeypatch.setattr(
        ConsoleVideoController,
        "_require_external_video_pinned_capabilities",
        staticmethod(lambda: actions.append("capability")),
    )
    monkeypatch.setattr(
        ConsoleVideoController,
        "_external_video_parent_identity",
        staticmethod(lambda _parent: actions.append("target-inspection")),
    )
    real_mkdir = Path.mkdir

    def tracking_mkdir(self, *args, **kwargs):
        actions.append("mkdir")
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", tracking_mkdir)
    real_open = os.open

    def tracking_open(*args, **kwargs):
        actions.append("stage")
        return real_open(*args, **kwargs)

    monkeypatch.setattr(os, "open", tracking_open)
    try:
        with pytest.raises(ValueError, match="extension"):
            ConsoleVideoController._copy_pending_video_external(artifact, target, None)
    finally:
        artifact.close()

    assert actions == []
    assert not parent.exists()
    assert not target.exists()


def test_external_copy_rejects_dangerous_path_before_any_filesystem_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _artifact(b"webm bytes", extension="webm")
    target = Path(f"{tmp_path}/safe/../../chosen.webm")
    actions: list[str] = []

    monkeypatch.setattr(
        ConsoleVideoController,
        "_require_external_video_pinned_capabilities",
        staticmethod(lambda: actions.append("capability")),
    )
    monkeypatch.setattr(
        ConsoleVideoController,
        "_external_video_parent_identity",
        staticmethod(lambda _parent: actions.append("target-inspection")),
    )
    real_mkdir = Path.mkdir

    def tracking_mkdir(self, *args, **kwargs):
        actions.append("mkdir")
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", tracking_mkdir)
    try:
        with pytest.raises(ValueError, match="dangerous pattern"):
            ConsoleVideoController._copy_pending_video_external(artifact, target, None)
    finally:
        artifact.close()

    assert actions == []


@pytest.mark.asyncio
async def test_external_picker_validates_path_before_target_inspection(
    tmp_path: Path,
) -> None:
    artifact = _artifact(b"webm bytes", extension="webm")
    dangerous = Path(f"{tmp_path}/safe/../../chosen.webm")
    harness = _OutcomeHarness(
        actions=["save_external", dangerous, None], video_store=object()
    )
    inspected: list[Path] = []
    harness._external_video_target_identity = lambda target: inspected.append(target)

    await harness._resolve_generated_video_outcome(
        artifact, session_id="session", message_id=artifact.message_id
    )

    assert inspected == []
    assert any(
        "generated video format" in message for message, _ in harness.notifications
    )
    assert artifact.stream.closed


@pytest.mark.asyncio
async def test_external_success_opens_file_and_appends_no_card(tmp_path: Path) -> None:
    artifact = _artifact(b"external")
    target = tmp_path / "chosen.mp4"
    harness = _OutcomeHarness(actions=["save_external", target], video_store=object())

    await harness._resolve_generated_video_outcome(
        artifact, session_id="session", message_id=artifact.message_id
    )

    assert target.read_bytes() == b"external"
    assert harness.opened == [target]
    assert harness.appended == []
    assert artifact.stream.closed


@pytest.mark.asyncio
async def test_external_transient_sibling_unlink_failure_still_reports_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _artifact(b"committed external bytes")
    target = tmp_path / "transient-unlink.mp4"
    harness = _OutcomeHarness(
        actions=["save_external", target, "discard"], video_store=object()
    )
    real_unlink = os.unlink
    sibling_unlinks = 0
    logged: list[str] = []

    def fail_first_sibling_unlink(path, *args, **kwargs):
        nonlocal sibling_unlinks
        if str(path).startswith(".") and str(path).endswith(".tmp"):
            sibling_unlinks += 1
            if sibling_unlinks == 1:
                raise OSError("PRIVATE-TRANSIENT-UNLINK")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", fail_first_sibling_unlink)
    monkeypatch.setattr(
        os,
        "supports_dir_fd",
        set(os.supports_dir_fd) | {fail_first_sibling_unlink},
    )
    sink_id = __import__("loguru").logger.add(logged.append, format="{message}")
    try:
        await harness._resolve_generated_video_outcome(
            artifact, session_id="session", message_id=artifact.message_id
        )
    finally:
        __import__("loguru").logger.remove(sink_id)

    assert sibling_unlinks == 2
    assert target.read_bytes() == b"committed external bytes"
    assert not list(tmp_path.glob(".*.tmp"))
    assert len(harness.waited_screens) == 2
    assert harness.actions == ["discard"]
    assert harness.opened == [target]
    assert harness.appended == []
    assert harness.notifications == []
    assert artifact.stream.closed
    assert cast(_TrackingStream, artifact.stream).close_calls == 1
    assert all("external_stage_cleanup" not in message for message in logged)
    assert all("PRIVATE-TRANSIENT-UNLINK" not in message for message in logged)


@pytest.mark.asyncio
async def test_external_ultimate_sibling_unlink_failure_keeps_saved_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _artifact(b"committed despite cleanup")
    target = tmp_path / "ultimate-unlink.mp4"
    harness = _OutcomeHarness(
        actions=["save_external", target, "discard"], video_store=object()
    )
    real_unlink = os.unlink
    sibling_unlinks = 0
    logged: list[str] = []

    def fail_sibling_unlink(path, *args, **kwargs):
        nonlocal sibling_unlinks
        if str(path).startswith(".") and str(path).endswith(".tmp"):
            sibling_unlinks += 1
            raise OSError("PRIVATE-ULTIMATE-UNLINK")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", fail_sibling_unlink)
    monkeypatch.setattr(
        os,
        "supports_dir_fd",
        set(os.supports_dir_fd) | {fail_sibling_unlink},
    )
    sink_id = __import__("loguru").logger.add(logged.append, format="{message}")
    try:
        await harness._resolve_generated_video_outcome(
            artifact, session_id="session", message_id=artifact.message_id
        )
    finally:
        __import__("loguru").logger.remove(sink_id)

    assert sibling_unlinks == 2
    assert target.read_bytes() == b"committed despite cleanup"
    assert len(list(tmp_path.glob(".*.tmp"))) == 1
    assert len(harness.waited_screens) == 2
    assert harness.actions == ["discard"]
    assert harness.opened == [target]
    assert harness.appended == []
    assert harness.notifications == []
    assert artifact.stream.closed
    assert cast(_TrackingStream, artifact.stream).close_calls == 1
    cleanup_logs = [
        message for message in logged if "external_stage_cleanup" in message
    ]
    assert len(cleanup_logs) == 1
    assert "OSError" in cleanup_logs[0]
    assert all("PRIVATE-ULTIMATE-UNLINK" not in message for message in logged)


@pytest.mark.asyncio
async def test_managed_save_copy_failure_logs_and_notifies_without_private_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_path = tmp_path / "PRIVATE-SENTINEL-managed-video.mp4"
    private_error = f"PRIVATE-SENTINEL failure at {private_path}"
    notifications: list[tuple[str, str | None]] = []
    logged: list[str] = []
    message = SimpleNamespace(
        video_metadata=SimpleNamespace(name="private-video", container="mp4")
    )

    class Harness:
        app_instance = SimpleNamespace(
            notify=lambda text, *, severity=None: notifications.append((text, severity))
        )

        @staticmethod
        def _ensure_console_chat_store():
            return SimpleNamespace(get_message=lambda _message_id: message)

        @staticmethod
        def _ensure_console_video_store():
            return SimpleNamespace(resolve=lambda *_args, **_kwargs: private_path)

        @staticmethod
        def _video_storage_message_id(_message):
            return "managed-message"

    async def fail_copy_thread(*_args, **_kwargs):
        raise OSError(private_error)

    monkeypatch.setattr(asyncio, "to_thread", fail_copy_thread)
    sink_id = __import__("loguru").logger.add(logged.append, format="{message}")
    try:
        await ConsoleVideoController._save_console_video_copy(
            Harness(), "managed-message"
        )
    finally:
        __import__("loguru").logger.remove(sink_id)

    assert notifications == [("Could not save the video.", "error")]
    assert any("managed_copy" in entry for entry in logged)
    assert any("OSError" in entry for entry in logged)
    assert all("PRIVATE-SENTINEL" not in entry for entry in logged)
    assert all(str(private_path) not in entry for entry in logged)
    assert all("PRIVATE-SENTINEL" not in text for text, _ in notifications)
    assert all(str(private_path) not in text for text, _ in notifications)


@pytest.mark.asyncio
async def test_external_copy_failure_reoffers_choices_with_artifact_live(
    tmp_path: Path,
) -> None:
    artifact = _artifact(b"still available")
    target = tmp_path / "chosen.mp4"
    harness = _OutcomeHarness(
        actions=["save_external", target, "discard"], video_store=object()
    )

    def fail_copy(*_args, **_kwargs):
        raise OSError("PRIVATE-PATH")

    harness._copy_pending_video_external = fail_copy

    await harness._resolve_generated_video_outcome(
        artifact, session_id="session", message_id=artifact.message_id
    )

    assert len(harness.waited_screens) == 3
    assert harness.appended == []
    assert not target.exists()
    assert any("try again" in message.lower() for message, _ in harness.notifications)
    assert all("PRIVATE-PATH" not in message for message, _ in harness.notifications)


@pytest.mark.asyncio
async def test_unmount_during_external_copy_failure_has_no_late_notification(
    tmp_path: Path,
) -> None:
    artifact = _artifact(b"still available")
    target = tmp_path / "chosen.mp4"
    harness = _OutcomeHarness(actions=["save_external", target], video_store=object())

    def drain_then_fail(*_args, **_kwargs):
        harness._drain_pending_console_videos()
        raise OSError("PRIVATE-PATH")

    harness._copy_pending_video_external = drain_then_fail

    await harness._resolve_generated_video_outcome(
        artifact, session_id="session", message_id=artifact.message_id
    )

    assert harness.notifications == []
    assert harness.appended == []
    assert not target.exists()


@pytest.mark.asyncio
async def test_existing_external_target_requires_confirmation_and_decline_repicks(
    tmp_path: Path,
) -> None:
    artifact = _artifact(b"replacement")
    target = tmp_path / "existing.mp4"
    target.write_bytes(b"keep me")
    harness = _OutcomeHarness(
        actions=["save_external", target, False, None], video_store=object()
    )

    await harness._resolve_generated_video_outcome(
        artifact, session_id="session", message_id=artifact.message_id
    )

    assert target.read_bytes() == b"keep me"
    assert harness.appended == []
    assert harness.opened == []
    assert len(harness.waited_screens) == 4


@pytest.mark.asyncio
async def test_concurrent_external_creator_is_confirmed_and_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _artifact(b"ours")
    target = tmp_path / "raced.mp4"
    real_link = os.link

    def racing_link(source, destination, **kwargs):
        target.write_bytes(b"theirs")
        return real_link(source, destination, **kwargs)

    monkeypatch.setattr(os, "link", racing_link)
    monkeypatch.setattr(os, "supports_dir_fd", set(os.supports_dir_fd) | {racing_link})
    monkeypatch.setattr(
        os,
        "supports_follow_symlinks",
        set(os.supports_follow_symlinks) | {racing_link},
    )
    harness = _OutcomeHarness(
        actions=["save_external", target, False, None], video_store=object()
    )

    await harness._resolve_generated_video_outcome(
        artifact, session_id="session", message_id=artifact.message_id
    )

    assert target.read_bytes() == b"theirs"
    assert harness.appended == []
    assert harness.opened == []
    assert len(harness.waited_screens) == 4


@pytest.mark.asyncio
async def test_confirmed_target_disappearing_requires_fresh_confirmation_before_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _artifact(b"ours")
    target = tmp_path / "disappears.mp4"
    target.write_bytes(b"original")
    harness = _OutcomeHarness(
        actions=["save_external", target, True, False, None],
        video_store=object(),
    )
    original_wait = harness._wait_for_console_screen_result
    real_stat = os.stat
    identity_calls = 0

    def disappear_during_pre_replace(path, *args, **kwargs):
        nonlocal identity_calls
        if path == target.name and kwargs.get("dir_fd") is not None:
            identity_calls += 1
            target.unlink()
            raise FileNotFoundError(target)
        return real_stat(path, *args, **kwargs)

    async def checking_wait(self, screen):
        result = await original_wait(screen)
        if len(self.waited_screens) == 4:
            artifact.rewind()
            assert artifact.stream.read() == b"ours"
        return result

    monkeypatch.setattr(os, "stat", disappear_during_pre_replace)
    monkeypatch.setattr(
        os,
        "supports_dir_fd",
        set(os.supports_dir_fd) | {disappear_during_pre_replace},
    )
    monkeypatch.setattr(
        os,
        "supports_follow_symlinks",
        set(os.supports_follow_symlinks) | {disappear_during_pre_replace},
    )
    harness._wait_for_console_screen_result = MethodType(checking_wait, harness)

    await harness._resolve_generated_video_outcome(
        artifact, session_id="session", message_id=artifact.message_id
    )

    assert identity_calls == 1
    assert len(harness.waited_screens) == 5
    assert harness.appended == []
    assert harness.opened == []
    assert not target.exists()
    assert artifact.stream.closed


@pytest.mark.asyncio
async def test_external_open_failure_keeps_saved_file_and_reports_sanitized_notice(
    tmp_path: Path,
) -> None:
    artifact = _artifact(b"saved bytes")
    target = tmp_path / "saved.mp4"
    harness = _OutcomeHarness(actions=["save_external", target], video_store=object())

    def fail_open(_path: Path) -> None:
        raise OSError("PRIVATE-PATH")

    harness._open_video_with_os = fail_open

    await harness._resolve_generated_video_outcome(
        artifact, session_id="session", message_id=artifact.message_id
    )

    assert target.read_bytes() == b"saved bytes"
    assert harness.appended == []
    assert any(
        "saved" in message.lower() and "could not open" in message.lower()
        for message, _ in harness.notifications
    )
    assert all("PRIVATE-PATH" not in message for message, _ in harness.notifications)


@pytest.mark.asyncio
async def test_late_modal_completion_after_drain_is_noop() -> None:
    artifact = _artifact()
    harness = _OutcomeHarness(actions=[], video_store=object())

    async def draining_wait(self, screen):
        self.waited_screens.append(screen)
        self._drain_pending_console_videos()
        return "keep"

    harness._wait_for_console_screen_result = MethodType(draining_wait, harness)

    await harness._resolve_generated_video_outcome(
        artifact, session_id="session", message_id=artifact.message_id
    )

    assert harness.appended == []
    assert artifact.stream.closed
    assert cast(_TrackingStream, artifact.stream).close_calls == 1


@pytest.mark.asyncio
async def test_late_picker_completion_after_drain_is_noop(tmp_path: Path) -> None:
    artifact = _artifact()
    harness = _OutcomeHarness(actions=[], video_store=object())
    waits = 0

    async def draining_picker_wait(self, screen):
        nonlocal waits
        waits += 1
        self.waited_screens.append(screen)
        if waits == 1:
            return "save_external"
        self._drain_pending_console_videos()
        return tmp_path / "late.mp4"

    harness._wait_for_console_screen_result = MethodType(draining_picker_wait, harness)

    await harness._resolve_generated_video_outcome(
        artifact, session_id="session", message_id=artifact.message_id
    )

    assert not (tmp_path / "late.mp4").exists()
    assert harness.appended == []
    assert harness.opened == []
    assert cast(_TrackingStream, artifact.stream).close_calls == 1


@pytest.mark.parametrize("wait_surface", ["modal", "picker"])
@pytest.mark.asyncio
async def test_mounted_chat_screen_exit_drains_modal_and_picker_waiters(
    wait_surface: str,
) -> None:
    from Tests.UI.app_factory import _build_test_app
    from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileSave

    class Host(ConsolidatedCSSApp):
        def __init__(self, app_instance) -> None:
            super().__init__()
            self.app_instance = app_instance
            self.pending_handoffs = app_instance.pending_handoffs

        async def on_mount(self) -> None:
            await self.push_screen(ChatScreen(self.app_instance))

    artifact = _artifact(message_id=f"mounted-{wait_surface}")
    backing_app = _build_test_app()
    host = Host(backing_app)
    appended: list[tuple] = []
    async with host.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        screen = host.screen
        assert isinstance(screen, ChatScreen)
        screen.app_instance = host
        chat_store = screen._ensure_console_chat_store()
        original_append = chat_store.append_video_message

        def recording_append(*args, **kwargs):
            appended.append((args, kwargs))
            return original_append(*args, **kwargs)

        chat_store.append_video_message = recording_append
        screen._sync_native_console_chat_ui = _completed_async
        screen.run_worker(
            screen._video._resolve_generated_video_outcome(
                artifact,
                session_id="session",
                message_id=artifact.message_id,
            ),
            exclusive=False,
            exit_on_error=False,
        )
        await pilot.pause()
        assert isinstance(host.screen, ConsoleVideoCapacityModal)
        if wait_surface == "picker":
            host.screen.dismiss("save_external")
            await pilot.pause()
            assert isinstance(host.screen, EnhancedFileSave)
        host.exit()
        await pilot.pause()

    assert artifact.stream.closed
    assert cast(_TrackingStream, artifact.stream).close_calls == 1
    assert screen._pending_video_artifacts == {}
    assert appended == []


@pytest.mark.asyncio
async def test_screen_wait_helper_uses_nonexclusive_nonfatal_worker() -> None:
    recorded: dict = {}

    class Worker:
        async def wait(self):
            return "result"

    class AppStub:
        async def push_screen_wait(self, screen):
            return screen

    def run_worker(coro, **kwargs):
        coro.close()
        recorded.update(kwargs)
        return Worker()

    fake = SimpleNamespace(app_instance=AppStub(), run_worker=run_worker)

    result = await ChatScreen._wait_for_console_screen_result(fake, object())

    assert result == "result"
    assert recorded == {"exclusive": False, "exit_on_error": False}


def test_unmount_drain_atomically_closes_every_pending_stream_once() -> None:
    first = _artifact(message_id="one")
    second = _artifact(message_id="two")
    fake = SimpleNamespace(_pending_video_artifacts={"one": first, "two": second})
    fake._pending_console_video_artifacts = lambda: fake._pending_video_artifacts

    ConsoleVideoController._drain_pending_console_videos(fake)
    ConsoleVideoController._drain_pending_console_videos(fake)

    assert fake._pending_video_artifacts == {}
    assert not ConsoleVideoController._owns_pending_console_video(fake, first)
    assert not ConsoleVideoController._owns_pending_console_video(fake, second)
    assert cast(_TrackingStream, first.stream).close_calls == 1
    assert cast(_TrackingStream, second.stream).close_calls == 1


@pytest.mark.asyncio
async def test_pending_result_arriving_after_unmount_is_closed_without_modal() -> None:
    artifact = _artifact(message_id="late-result")
    harness = _OutcomeHarness(actions=[], video_store=object())
    harness._drain_pending_console_videos()

    await harness._resolve_generated_video_outcome(
        artifact, session_id="session", message_id=artifact.message_id
    )

    assert harness.waited_screens == []
    assert harness.appended == []
    assert harness._pending_console_video_artifacts() == {}
    assert artifact.stream.closed
    assert cast(_TrackingStream, artifact.stream).close_calls == 1


def test_pending_drain_sets_outstanding_generation_cancellation_signals() -> None:
    cancel = threading.Event()
    fake = SimpleNamespace(
        _pending_video_artifacts={},
        _console_videogen_cancels={"session": cancel},
    )

    ConsoleVideoController._drain_pending_console_videos(fake)

    assert cancel.is_set()


@pytest.mark.asyncio
async def test_same_message_collision_rejects_newcomer_without_closing_owner() -> None:
    owner = _artifact(message_id="collision")
    newcomer = _artifact(message_id="collision")
    harness = _OutcomeHarness(actions=[], video_store=object())
    harness._pending_console_video_artifacts()[owner.message_id] = owner
    assert harness._begin_pending_console_video_operation(owner) is not None

    try:
        await harness._resolve_generated_video_outcome(
            newcomer, session_id="session", message_id=newcomer.message_id
        )
    finally:
        harness._end_pending_console_video_operation(owner)

    assert not owner.stream.closed
    assert newcomer.stream.closed
    assert harness._pending_console_video_artifacts()[owner.message_id] is owner
    assert harness.waited_screens == []
    owner.close()


@pytest.mark.asyncio
async def test_unmount_defers_close_and_cancels_managed_precommit(
    tmp_path: Path,
) -> None:
    artifact = _artifact(b"managed", message_id="active-managed")
    published = tmp_path / "managed.mp4"
    started = threading.Event()
    release = threading.Event()
    received_gates = []

    class BlockingStore:
        def adopt_oversized(self, *_args, **kwargs):
            gate = kwargs.get("publication_gate")
            received_gates.append(gate)
            started.set()
            assert release.wait(5)
            with gate.claim_publication() as active:
                if not active:
                    raise VideoStoreSaveError("managed publication cancelled")
                published.write_bytes(b"managed")
            return published

        def resolve(self, *_args, **_kwargs):
            return published if published.exists() else None

    harness = _OutcomeHarness(actions=["keep"], video_store=BlockingStore())
    task = asyncio.create_task(
        harness._resolve_generated_video_outcome(
            artifact, session_id="session", message_id=artifact.message_id
        )
    )
    while not started.is_set():
        await asyncio.sleep(0)

    harness._drain_pending_console_videos()
    assert not artifact.stream.closed
    release.set()
    await task

    assert artifact.stream.closed
    assert cast(_TrackingStream, artifact.stream).close_calls == 1
    assert len(received_gates) == 1
    assert isinstance(received_gates[0], VideoPublicationGate)
    assert not published.exists()
    assert harness.appended == []


@pytest.mark.asyncio
async def test_pending_managed_commit_winning_before_unmount_persists_without_sync(
    tmp_path: Path,
) -> None:
    artifact = _artifact(b"committed", message_id="committed-pending")
    published = tmp_path / "committed.mp4"
    commit_claimed = threading.Event()
    release_commit = threading.Event()

    class CommitWinningStore:
        def adopt_oversized(self, *_args, **kwargs):
            gate = kwargs["publication_gate"]
            with gate.claim_publication() as active:
                assert active
                commit_claimed.set()
                assert release_commit.wait(5)
                published.write_bytes(b"committed")
            return published

        def resolve(self, *_args, **_kwargs):
            return published if published.exists() else None

    harness = _OutcomeHarness(actions=["keep"], video_store=CommitWinningStore())
    resolver = asyncio.create_task(
        harness._resolve_generated_video_outcome(
            artifact, session_id="session", message_id=artifact.message_id
        )
    )
    while not commit_claimed.is_set():
        await asyncio.sleep(0)

    drained = threading.Event()

    def drain():
        harness._drain_pending_console_videos()
        drained.set()

    teardown = threading.Thread(target=drain, daemon=True)
    teardown.start()
    while not getattr(harness, "_pending_video_artifacts_closed", False):
        await asyncio.sleep(0)
    assert not drained.is_set()
    release_commit.set()
    teardown.join(5)
    await resolver

    assert not teardown.is_alive()
    assert drained.is_set()
    assert published.read_bytes() == b"committed"
    assert len(harness.appended) == 1
    assert harness.appended[0][1]["message_id"] == artifact.message_id
    assert harness.sync_count == 0
    assert artifact.stream.closed


@pytest.mark.asyncio
async def test_resolver_cancellation_waits_for_external_copy_before_closing_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"cancelled external copy"
    artifact = _artifact(payload, message_id="cancel-external-copy")
    target = tmp_path / "cancelled-external.mp4"
    harness = _OutcomeHarness(actions=["save_external", target], video_store=object())
    copy_started = threading.Event()
    release_copy = threading.Event()
    copy_finished = threading.Event()
    observed_payloads: list[bytes | None] = []

    def blocking_copy(source, destination, *_args, **_kwargs):
        copy_started.set()
        assert release_copy.wait(5)
        try:
            if source.closed:
                observed_payloads.append(None)
                return
            copied = source.read()
            observed_payloads.append(copied)
            destination.write(copied)
        finally:
            copy_finished.set()

    monkeypatch.setattr(shutil, "copyfileobj", blocking_copy)
    resolver = asyncio.create_task(
        harness._resolve_generated_video_outcome(
            artifact, session_id="session", message_id=artifact.message_id
        )
    )
    while not copy_started.is_set():
        await asyncio.sleep(0)

    resolver.cancel()
    for _ in range(3):
        await asyncio.sleep(0)
    cancellation_waited = not resolver.done()
    stream_open_while_worker_active = not artifact.stream.closed
    release_copy.set()
    while not copy_finished.is_set():
        await asyncio.sleep(0)
    with pytest.raises(asyncio.CancelledError):
        await resolver

    assert cancellation_waited
    assert stream_open_while_worker_active
    assert observed_payloads == [payload]
    assert target.read_bytes() == payload
    assert harness.opened == []
    assert harness.appended == []
    assert harness.sync_count == 0
    assert artifact.stream.closed
    assert cast(_TrackingStream, artifact.stream).close_calls == 1


@pytest.mark.asyncio
async def test_resolver_cancellation_finalizes_managed_commit_before_stream_close(
    tmp_path: Path,
) -> None:
    payload = b"cancelled managed commit"
    artifact = _artifact(payload, message_id="cancel-managed-commit")
    managed_path = tmp_path / "managed.mp4"
    commit_finished = threading.Event()
    release_worker = threading.Event()
    worker_finished = threading.Event()
    observed_stream_open: list[bool] = []

    class CommitWinningStore:
        def adopt_oversized(self, *_args, **kwargs):
            stream = _args[2]
            stream.seek(0)
            copied = stream.read()
            stream.seek(0)
            gate = kwargs["publication_gate"]
            with gate.claim_publication() as active:
                assert active
                managed_path.write_bytes(copied)
            commit_finished.set()
            assert release_worker.wait(5)
            try:
                observed_stream_open.append(not stream.closed)
            finally:
                worker_finished.set()
            return managed_path

        def resolve(self, *_args, **_kwargs):
            return managed_path if managed_path.exists() else None

    harness = _OutcomeHarness(actions=["keep"], video_store=CommitWinningStore())
    resolver = asyncio.create_task(
        harness._resolve_generated_video_outcome(
            artifact, session_id="session", message_id=artifact.message_id
        )
    )
    while not commit_finished.is_set():
        await asyncio.sleep(0)

    resolver.cancel()
    for _ in range(3):
        await asyncio.sleep(0)
    cancellation_waited = not resolver.done()
    stream_open_while_worker_active = not artifact.stream.closed
    release_worker.set()
    while not worker_finished.is_set():
        await asyncio.sleep(0)
    with pytest.raises(asyncio.CancelledError):
        await resolver

    assert cancellation_waited
    assert stream_open_while_worker_active
    assert observed_stream_open == [True]
    assert managed_path.read_bytes() == payload
    assert len(harness.appended) == 1
    assert harness.appended[0][1]["message_id"] == artifact.message_id
    assert harness.sync_count == 0
    assert harness.opened == []
    assert artifact.stream.closed
    assert cast(_TrackingStream, artifact.stream).close_calls == 1


@pytest.mark.parametrize("caller", ["initial", "regenerate"])
@pytest.mark.asyncio
async def test_generation_cancellation_after_commit_persists_before_child_finishes(
    caller: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.UI.Console_Modules import video as video_controller_module
    from tldw_chatbook.Video_Generation import adapter_registry

    generated_meta = VideoGenerationMetadata(
        name="generated", prompt="generate me", backend="comfyui"
    )
    original_meta = VideoGenerationMetadata(
        name="original", prompt="generate me", backend="comfyui"
    )
    managed_path = tmp_path / f"{caller}-cancelled-after-commit.mp4"
    commit_finished = threading.Event()
    release_generation = threading.Event()
    generation_finished = threading.Event()
    captured_message_ids: list[str] = []
    inflight: set[str] = set()
    adapter_cancels: dict = {}
    harness = _OutcomeHarness(actions=[], video_store=object())
    harness._console_videogen_inflight_sessions = lambda: inflight
    harness._console_videogen_cancel_events = lambda: adapter_cancels
    harness._append_native_console_system_message = lambda *_args, **_kwargs: (
        _completed_async()
    )
    harness.app_instance = SimpleNamespace(notify=lambda *_args, **_kwargs: None)

    if caller == "initial":
        harness.chat_store.workspace_context = SimpleNamespace(
            active_workspace_id="workspace"
        )
        harness.chat_store.ensure_session = lambda **_kwargs: SimpleNamespace(
            id="session"
        )
        harness._default_console_session_settings = lambda: object()
        harness._console_composer_or_none = lambda: None
        harness._clear_console_composer_draft = lambda: None
    else:
        harness.chat_store.get_message = lambda _message_id: SimpleNamespace(
            video_metadata=original_meta
        )
        harness.chat_store.session_id_for_message = lambda _message_id: "session"

    class Registry:
        @staticmethod
        def resolve_backend(_backend):
            return object()

    def commit_then_block_generation(**kwargs):
        gate = kwargs["publication_gate"]
        captured_message_ids.append(kwargs["message_id"])
        with gate.claim_publication() as active:
            assert active
            managed_path.write_bytes(b"committed generation")
        commit_finished.set()
        assert release_generation.wait(5)
        generation_finished.set()
        return generated_meta, managed_path

    monkeypatch.setattr(
        video_controller_module,
        "get_video_generation_config",
        lambda: SimpleNamespace(default_backend="comfyui", confirm_cost_estimate=False),
    )
    monkeypatch.setattr(adapter_registry, "get_registry", lambda: Registry())
    monkeypatch.setattr(
        video_controller_module, "run_video_generation", commit_then_block_generation
    )

    if caller == "initial":
        operation = asyncio.create_task(
            ConsoleVideoController._console_command_generate_video(
                harness, SimpleNamespace(args="generate me")
            )
        )
    else:
        operation = asyncio.create_task(
            ConsoleVideoController._regenerate_console_video_message(
                harness, "old-message"
            )
        )
    while not commit_finished.is_set():
        await asyncio.sleep(0)

    harness._drain_pending_console_videos()
    operation.cancel()
    for _ in range(3):
        await asyncio.sleep(0)
    cancellation_waited = not operation.done()
    release_generation.set()
    while not generation_finished.is_set():
        await asyncio.sleep(0)
    with pytest.raises(asyncio.CancelledError):
        await operation

    assert cancellation_waited
    assert managed_path.read_bytes() == b"committed generation"
    assert len(captured_message_ids) == 1
    assert len(harness.appended) == 1
    assert harness.appended[0][1]["message_id"] == captured_message_ids[0]
    assert harness.appended[0][1]["persist"] is True
    assert harness.sync_count == 0
    assert inflight == set()
    assert adapter_cancels == {}


@pytest.mark.parametrize("caller", ["initial", "regenerate"])
@pytest.mark.asyncio
async def test_generation_cancellation_during_sync_does_not_wait_for_stale_ui(
    caller: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.UI.Console_Modules import video as video_controller_module
    from tldw_chatbook.Video_Generation import adapter_registry

    generated_meta = VideoGenerationMetadata(
        name="generated", prompt="generate me", backend="comfyui"
    )
    original_meta = VideoGenerationMetadata(
        name="original", prompt="generate me", backend="comfyui"
    )
    managed_path = tmp_path / f"{caller}-cancelled-during-sync.mp4"
    sync_started = asyncio.Event()
    release_sync = asyncio.Event()
    sync_completed = False
    inflight: set[str] = set()
    adapter_cancels: dict = {}
    harness = _OutcomeHarness(actions=[], video_store=object())
    harness._console_videogen_inflight_sessions = lambda: inflight
    harness._console_videogen_cancel_events = lambda: adapter_cancels
    harness._append_native_console_system_message = lambda *_args, **_kwargs: (
        _completed_async()
    )
    harness.app_instance = SimpleNamespace(notify=lambda *_args, **_kwargs: None)

    async def blocking_sync() -> None:
        nonlocal sync_completed
        sync_started.set()
        await release_sync.wait()
        sync_completed = True

    harness._sync_native_console_chat_ui = blocking_sync
    if caller == "initial":
        harness.chat_store.workspace_context = SimpleNamespace(
            active_workspace_id="workspace"
        )
        harness.chat_store.ensure_session = lambda **_kwargs: SimpleNamespace(
            id="session"
        )
        harness._default_console_session_settings = lambda: object()
        harness._console_composer_or_none = lambda: None
        harness._clear_console_composer_draft = lambda: None
    else:
        harness.chat_store.get_message = lambda _message_id: SimpleNamespace(
            video_metadata=original_meta
        )
        harness.chat_store.session_id_for_message = lambda _message_id: "session"

    class Registry:
        @staticmethod
        def resolve_backend(_backend):
            return object()

    def commit_generation(**kwargs):
        gate = kwargs["publication_gate"]
        with gate.claim_publication() as active:
            assert active
            managed_path.write_bytes(b"committed before sync")
        return generated_meta, managed_path

    monkeypatch.setattr(
        video_controller_module,
        "get_video_generation_config",
        lambda: SimpleNamespace(default_backend="comfyui", confirm_cost_estimate=False),
    )
    monkeypatch.setattr(adapter_registry, "get_registry", lambda: Registry())
    monkeypatch.setattr(
        video_controller_module, "run_video_generation", commit_generation
    )

    if caller == "initial":
        operation = asyncio.create_task(
            ConsoleVideoController._console_command_generate_video(
                harness, SimpleNamespace(args="generate me")
            )
        )
    else:
        operation = asyncio.create_task(
            ConsoleVideoController._regenerate_console_video_message(
                harness, "old-message"
            )
        )
    await sync_started.wait()

    harness._drain_pending_console_videos()
    operation.cancel()
    for _ in range(3):
        await asyncio.sleep(0)
    cancellation_finished_without_sync_release = operation.done()
    release_sync.set()
    with pytest.raises(asyncio.CancelledError):
        await operation
    await asyncio.sleep(0)

    assert cancellation_finished_without_sync_release
    assert managed_path.read_bytes() == b"committed before sync"
    assert len(harness.appended) == 1
    assert harness.appended[0][1]["persist"] is True
    assert harness.sync_count == 0
    assert not sync_completed
    assert inflight == set()
    assert adapter_cancels == {}


@pytest.mark.asyncio
async def test_generation_drain_during_durable_append_skips_outer_sync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated_meta = VideoGenerationMetadata(
        name="generated", prompt="generate me", backend="comfyui"
    )
    harness = _OutcomeHarness(actions=[], video_store=object())
    original_append = harness.chat_store.append_video_message

    def append_then_drain(*args, **kwargs):
        original_append(*args, **kwargs)
        harness._drain_pending_console_videos()

    harness.chat_store.append_video_message = append_then_drain

    async def fake_to_thread(_function, **_kwargs):
        return generated_meta, tmp_path / "durable-before-drain.mp4"

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    await ConsoleVideoController._run_console_video_generation_operation(
        harness,
        session_id="session",
        message_id="durable-before-drain",
        backend="comfyui",
    )

    assert len(harness.appended) == 1
    assert harness.appended[0][1]["persist"] is True
    assert harness.appended[0][1]["message_id"] == "durable-before-drain"
    assert harness.sync_count == 0


@pytest.mark.asyncio
async def test_initial_generation_cancellation_closes_late_pending_without_modal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.UI.Console_Modules import video as video_controller_module
    from tldw_chatbook.Video_Generation import adapter_registry

    worker_started = threading.Event()
    release_worker = threading.Event()
    worker_finished = threading.Event()
    pending: list[PendingVideoArtifact] = []
    inflight: set[str] = set()
    adapter_cancels: dict = {}
    harness = _OutcomeHarness(actions=[], video_store=object())
    harness._console_videogen_inflight_sessions = lambda: inflight
    harness._console_videogen_cancel_events = lambda: adapter_cancels
    harness._append_native_console_system_message = lambda *_args, **_kwargs: (
        _completed_async()
    )
    harness.app_instance = SimpleNamespace(notify=lambda *_args, **_kwargs: None)
    harness.chat_store.workspace_context = SimpleNamespace(
        active_workspace_id="workspace"
    )
    harness.chat_store.ensure_session = lambda **_kwargs: SimpleNamespace(id="session")
    harness._default_console_session_settings = lambda: object()
    harness._console_composer_or_none = lambda: None
    harness._clear_console_composer_draft = lambda: None

    class Registry:
        @staticmethod
        def resolve_backend(_backend):
            return object()

    def return_pending_after_cancel(**kwargs):
        worker_started.set()
        assert release_worker.wait(5)
        try:
            artifact = _artifact(
                reason="store_failure", message_id=kwargs["message_id"]
            )
            pending.append(artifact)
            return artifact
        finally:
            worker_finished.set()

    monkeypatch.setattr(
        video_controller_module,
        "get_video_generation_config",
        lambda: SimpleNamespace(default_backend="comfyui", confirm_cost_estimate=False),
    )
    monkeypatch.setattr(adapter_registry, "get_registry", lambda: Registry())
    monkeypatch.setattr(
        video_controller_module, "run_video_generation", return_pending_after_cancel
    )

    operation = asyncio.create_task(
        ConsoleVideoController._console_command_generate_video(
            harness, SimpleNamespace(args="generate me")
        )
    )
    while not worker_started.is_set():
        await asyncio.sleep(0)

    harness._drain_pending_console_videos()
    operation.cancel()
    for _ in range(3):
        await asyncio.sleep(0)
    cancellation_waited = not operation.done()
    release_worker.set()
    while not worker_finished.is_set():
        await asyncio.sleep(0)
    with pytest.raises(asyncio.CancelledError):
        await operation

    assert cancellation_waited
    assert len(pending) == 1
    assert pending[0].stream.closed
    assert cast(_TrackingStream, pending[0].stream).close_calls == 1
    assert harness.waited_screens == []
    assert harness.appended == []
    assert harness.sync_count == 0
    assert inflight == set()
    assert adapter_cancels == {}


@pytest.mark.asyncio
async def test_pending_cancellation_during_managed_resolve_persists_before_close(
    tmp_path: Path,
) -> None:
    payload = b"pending commit before blocked resolve"
    artifact = _artifact(payload, message_id="pending-blocked-resolve")
    managed_path = tmp_path / "pending-blocked-resolve.mp4"
    resolve_started = threading.Event()
    release_resolve = threading.Event()
    resolve_finished = threading.Event()

    class ResolveBlockingStore:
        def adopt_oversized(self, *_args, **kwargs):
            stream = _args[2]
            stream.seek(0)
            copied = stream.read()
            stream.seek(0)
            with kwargs["publication_gate"].claim_publication() as active:
                assert active
                managed_path.write_bytes(copied)
            return managed_path

        def resolve(self, *_args, **_kwargs):
            resolve_started.set()
            assert release_resolve.wait(5)
            resolve_finished.set()
            return managed_path

    harness = _OutcomeHarness(actions=["keep"], video_store=ResolveBlockingStore())
    resolver = asyncio.create_task(
        harness._resolve_generated_video_outcome(
            artifact, session_id="session", message_id=artifact.message_id
        )
    )
    while not resolve_started.is_set():
        await asyncio.sleep(0)

    harness._drain_pending_console_videos()
    resolver.cancel()
    for _ in range(3):
        await asyncio.sleep(0)
    cancellation_waited = not resolver.done()
    stream_open_during_resolve = not artifact.stream.closed
    release_resolve.set()
    while not resolve_finished.is_set():
        await asyncio.sleep(0)
    with pytest.raises(asyncio.CancelledError):
        await resolver

    assert cancellation_waited
    assert stream_open_during_resolve
    assert managed_path.read_bytes() == payload
    assert len(harness.appended) == 1
    assert harness.appended[0][1]["message_id"] == artifact.message_id
    assert harness.appended[0][1]["persist"] is True
    assert harness.sync_count == 0
    assert artifact.stream.closed
    assert cast(_TrackingStream, artifact.stream).close_calls == 1


@pytest.mark.asyncio
async def test_pending_cancellation_before_managed_commit_leaves_no_path_or_message(
    tmp_path: Path,
) -> None:
    payload = b"cancel before pending commit"
    artifact = _artifact(payload, message_id="pending-cancel-before-commit")
    managed_path = tmp_path / "must-not-publish.mp4"
    worker_started = threading.Event()
    release_worker = threading.Event()
    worker_finished = threading.Event()

    class CancelWinningStore:
        def adopt_oversized(self, *_args, **kwargs):
            worker_started.set()
            assert release_worker.wait(5)
            try:
                with kwargs["publication_gate"].claim_publication() as active:
                    if not active:
                        raise VideoStoreSaveError("publication cancelled")
                    managed_path.write_bytes(payload)
            finally:
                worker_finished.set()
            return managed_path

        def resolve(self, *_args, **_kwargs):
            return managed_path if managed_path.exists() else None

    harness = _OutcomeHarness(actions=["keep"], video_store=CancelWinningStore())
    resolver = asyncio.create_task(
        harness._resolve_generated_video_outcome(
            artifact, session_id="session", message_id=artifact.message_id
        )
    )
    while not worker_started.is_set():
        await asyncio.sleep(0)

    harness._drain_pending_console_videos()
    resolver.cancel()
    for _ in range(3):
        await asyncio.sleep(0)
    cancellation_waited = not resolver.done()
    assert cancellation_waited
    release_worker.set()
    while not worker_finished.is_set():
        await asyncio.sleep(0)
    with pytest.raises(asyncio.CancelledError):
        await resolver

    assert not managed_path.exists()
    assert harness.appended == []
    assert harness.sync_count == 0
    assert artifact.stream.closed
    assert cast(_TrackingStream, artifact.stream).close_calls == 1


def test_pending_drain_contains_close_failure_and_closes_remaining_artifacts() -> None:
    logged: list[str] = []

    class FailingArtifact:
        message_id = "failing"

        def close(self):
            raise RuntimeError("PRIVATE-CLOSE-SENTINEL")

    good = _artifact(message_id="good")
    fake = SimpleNamespace(
        _pending_video_artifacts={"failing": FailingArtifact(), "good": good},
        _console_videogen_cancels={},
    )
    sink_id = __import__("loguru").logger.add(logged.append, format="{message}")
    try:
        ConsoleVideoController._drain_pending_console_videos(fake)
    finally:
        __import__("loguru").logger.remove(sink_id)

    assert good.stream.closed
    assert any("artifact_close" in message for message in logged)
    assert any("RuntimeError" in message for message in logged)
    assert all("PRIVATE-CLOSE-SENTINEL" not in message for message in logged)


def test_pending_drain_contains_each_cancel_failure_and_finishes_cleanup() -> None:
    logged: list[str] = []
    adapter_cancelled = threading.Event()
    later_gate = VideoPublicationGate()
    artifact = _artifact(message_id="cleanup-survivor")

    class FailingAdapterCancel:
        def set(self):
            raise RuntimeError("PRIVATE-ADAPTER-CANCEL")

    class FailingPublicationCancel:
        def cancel(self):
            raise ValueError("PRIVATE-GATE-CANCEL")

    fake = SimpleNamespace(
        _pending_video_artifacts={artifact.message_id: artifact},
        _console_videogen_cancels={
            "failing": FailingAdapterCancel(),
            "later": adapter_cancelled,
        },
        _pending_video_operation_cancels={
            "failing": FailingPublicationCancel(),
            "later": later_gate,
        },
    )
    sink_id = __import__("loguru").logger.add(logged.append, format="{message}")
    try:
        ConsoleVideoController._drain_pending_console_videos(fake)
    finally:
        __import__("loguru").logger.remove(sink_id)

    assert adapter_cancelled.is_set()
    with later_gate.claim_publication() as active:
        assert not active
    assert fake._pending_video_artifacts == {}
    assert fake._pending_video_operation_cancels == {}
    assert artifact.stream.closed
    assert any("adapter_cancel" in message for message in logged)
    assert any("publication_cancel" in message for message in logged)
    assert any("RuntimeError" in message for message in logged)
    assert any("ValueError" in message for message in logged)
    assert all("PRIVATE-" not in message for message in logged)


def test_real_unmount_path_invokes_pending_artifact_drain() -> None:
    source = inspect.getsource(ChatScreen.on_unmount)

    assert "self._video._drain_pending_console_videos()" in source


@pytest.mark.asyncio
async def test_regenerate_normal_result_persists_then_syncs_through_shared_operation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original_meta = VideoGenerationMetadata(
        name="old", prompt="regenerate me", backend="comfyui", container="webm"
    )
    generated_meta = VideoGenerationMetadata(
        name="new", prompt="regenerate me", backend="comfyui"
    )
    harness = _OutcomeHarness(actions=[], video_store=object())
    harness.chat_store.get_message = lambda _message_id: SimpleNamespace(
        video_metadata=original_meta
    )
    harness.chat_store.session_id_for_message = lambda _message_id: "session"

    dispatched_kwargs = {}

    async def fake_to_thread(_function, **kwargs):
        dispatched_kwargs.update(kwargs)
        return generated_meta, tmp_path / "new.webm"

    monkeypatch.setattr("asyncio.to_thread", fake_to_thread)
    inflight: set[str] = set()
    cancels: dict = {}
    harness._console_videogen_inflight_sessions = lambda: inflight
    harness._console_videogen_cancel_events = lambda: cancels
    harness._append_native_console_system_message = lambda *_args, **_kwargs: (
        _completed_async()
    )
    harness.app_instance = SimpleNamespace(notify=lambda *_args, **_kwargs: None)

    await ConsoleVideoController._regenerate_console_video_message(
        harness, "old-message"
    )

    assert len(harness.appended) == 1
    assert harness.appended[0][1]["video_metadata"] is generated_meta
    assert harness.appended[0][1]["persist"] is True
    assert harness.appended[0][1]["message_id"]
    assert harness.sync_count == 1
    assert dispatched_kwargs["video_format"] == "webm"
    assert inflight == set()
    assert cancels == {}


@pytest.mark.asyncio
async def test_regenerate_pending_discard_closes_stage_and_clears_bookkeeping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_meta = VideoGenerationMetadata(
        name="old", prompt="regenerate me", backend="comfyui"
    )
    artifact = _artifact(message_id="regenerated-message")
    harness = _OutcomeHarness(actions=["discard"], video_store=object())
    harness.chat_store.get_message = lambda _message_id: SimpleNamespace(
        video_metadata=original_meta
    )
    harness.chat_store.session_id_for_message = lambda _message_id: "session"
    inflight: set[str] = set()
    cancels: dict = {}
    harness._console_videogen_inflight_sessions = lambda: inflight
    harness._console_videogen_cancel_events = lambda: cancels

    async def fake_to_thread(_function, **_kwargs):
        artifact.message_id = _kwargs["message_id"]
        return artifact

    monkeypatch.setattr("asyncio.to_thread", fake_to_thread)

    await ConsoleVideoController._regenerate_console_video_message(
        harness, "old-message"
    )

    assert harness.appended == []
    assert artifact.stream.closed
    assert inflight == set()
    assert cancels == {}


@pytest.mark.parametrize("caller", ["initial", "regenerate"])
@pytest.mark.parametrize("ordering", ["cancel_wins", "commit_wins", "generation_fails"])
@pytest.mark.asyncio
async def test_generation_publication_gate_linearizes_teardown_for_both_callers(
    caller: str,
    ordering: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.UI.Console_Modules import video as video_controller_module
    from tldw_chatbook.Video_Generation import adapter_registry

    generated_meta = VideoGenerationMetadata(
        name="generated", prompt="generate me", backend="comfyui"
    )
    original_meta = VideoGenerationMetadata(
        name="original", prompt="generate me", backend="comfyui"
    )
    managed_path = tmp_path / f"{caller}-{ordering}.mp4"
    pending: list[PendingVideoArtifact] = []
    captured_gates: list[VideoPublicationGate] = []
    inflight: set[str] = set()
    adapter_cancels: dict = {}
    harness = _OutcomeHarness(actions=[], video_store=object())
    harness._console_videogen_inflight_sessions = lambda: inflight
    harness._console_videogen_cancel_events = lambda: adapter_cancels
    harness._append_native_console_system_message = lambda *_args, **_kwargs: (
        _completed_async()
    )
    harness.app_instance = SimpleNamespace(notify=lambda *_args, **_kwargs: None)

    if caller == "initial":
        harness.chat_store.workspace_context = SimpleNamespace(
            active_workspace_id="workspace"
        )
        harness.chat_store.ensure_session = lambda **_kwargs: SimpleNamespace(
            id="session"
        )
        harness._default_console_session_settings = lambda: object()
        harness._console_composer_or_none = lambda: None
        harness._clear_console_composer_draft = lambda: None
    else:
        harness.chat_store.get_message = lambda _message_id: SimpleNamespace(
            video_metadata=original_meta
        )
        harness.chat_store.session_id_for_message = lambda _message_id: "session"

    class Registry:
        @staticmethod
        def resolve_backend(_backend):
            return object()

    async def fake_to_thread(_function, **kwargs):
        gate = kwargs["publication_gate"]
        message_id = kwargs["message_id"]
        captured_gates.append(gate)
        assert isinstance(gate, VideoPublicationGate)
        assert harness._pending_video_operation_cancels[message_id] is gate
        if ordering == "generation_fails":
            raise RuntimeError("PRIVATE-GENERATION-FAILURE")
        if ordering == "cancel_wins":
            harness._drain_pending_console_videos()
            with gate.claim_publication() as active:
                assert not active
            artifact = _artifact(message_id=message_id)
            pending.append(artifact)
            return artifact
        with gate.claim_publication() as active:
            assert active
            managed_path.write_bytes(b"committed")
        harness._drain_pending_console_videos()
        return generated_meta, managed_path

    monkeypatch.setattr(
        video_controller_module,
        "get_video_generation_config",
        lambda: SimpleNamespace(default_backend="comfyui", confirm_cost_estimate=False),
    )
    monkeypatch.setattr(adapter_registry, "get_registry", lambda: Registry())
    monkeypatch.setattr("asyncio.to_thread", fake_to_thread)

    if caller == "initial":
        await ConsoleVideoController._console_command_generate_video(
            harness, SimpleNamespace(args="generate me")
        )
    else:
        await ConsoleVideoController._regenerate_console_video_message(
            harness, "old-message"
        )

    assert len(captured_gates) == 1
    assert inflight == set()
    assert adapter_cancels == {}
    assert harness._pending_video_operation_cancels == {}
    if ordering == "cancel_wins":
        assert not managed_path.exists()
        assert harness.appended == []
        assert len(pending) == 1 and pending[0].stream.closed
    elif ordering == "commit_wins":
        assert managed_path.read_bytes() == b"committed"
        assert len(harness.appended) == 1
        assert harness.sync_count == 0
    else:
        assert not managed_path.exists()
        assert harness.appended == []

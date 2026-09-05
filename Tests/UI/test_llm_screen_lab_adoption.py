"""Models' adoption of the Lab frame, and its rail lift."""

from __future__ import annotations


import asyncio
import threading
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from textual.widgets import Button, Input, Select, Static

from tldw_chatbook.app import TldwCli
from tldw_chatbook.config import get_cli_setting as _real_get_cli_setting
from tldw_chatbook.Model_Artifacts.machine_memory import (
    AcceleratorMemoryObservation,
    AcceleratorSource,
    AcceleratorState,
    GIB,
    MachineMemorySnapshot,
    MemoryKind,
    ProbeReason,
    SystemMemoryState,
)
from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow
from tldw_chatbook.UI.Screens.llm_screen import LLMScreen
from Tests.UI.app_factory import _build_test_app

_MODELS_MOUNT_POLL_ATTEMPTS = 200
_MODELS_MOUNT_POLL_SECONDS = 0.01


@pytest.fixture(autouse=True)
def _deterministic_models_mount(monkeypatch):
    """Neutralise the splash race this file's press/pause sequences can hit.

    Same rationale as the identically named fixture in
    ``test_lab_frame_mode_keys.py``: ``SplashScreen`` starts a real 1.5s
    timer that can push a competing screen mid-test.

    Args:
        monkeypatch: pytest's monkeypatch fixture; reverts both patches
            automatically at the end of each test.
    """

    def fake_get_cli_setting(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        return _real_get_cli_setting(section, key, default)

    async def ollama_unavailable(_window):
        return False

    monkeypatch.setattr("tldw_chatbook.app.get_cli_setting", fake_get_cli_setting)
    monkeypatch.setattr(
        LLMManagementWindow,
        "_ollama_api_available",
        ollama_unavailable,
    )


async def _models_screen(pilot_app, *, populate_all: bool = True):
    """Mount Models with the legacy all-view fixture unless testing laziness."""

    screen = LLMScreen(pilot_app)
    await pilot_app.push_screen(screen)
    if populate_all:
        for _ in range(_MODELS_MOUNT_POLL_ATTEMPTS):
            windows = list(screen.query(LLMManagementWindow))
            if windows:
                break
            await asyncio.sleep(_MODELS_MOUNT_POLL_SECONDS)
        window = windows[0]
        for _ in range(_MODELS_MOUNT_POLL_ATTEMPTS):
            if all(
                list(window.query(f"#{view_id}"))
                for view_id in window.view_mapping.values()
            ):
                break
            await asyncio.sleep(_MODELS_MOUNT_POLL_SECONDS)
        for view_name in window.view_mapping:
            if view_name != "llama-cpp":
                await window._mount_deferred_views(view_name)
    return screen


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (140, 45)])
async def test_snapshot_extension_keeps_launcher_primary_controls_above_fold(size):
    app = _build_test_app()
    async with app.run_test(size=size) as pilot:
        screen = await _models_screen(app)
        await pilot.pause()
        pane = screen.query_one("#llm-view-llama-cpp")
        for selector in (
            "#llamacpp-start-server-button",
            "#llamacpp-stop-server-button",
        ):
            button = screen.query_one(selector, Button)
            _assert_painted_inside(app, button, pane)
        manager = screen.query_one("#llamacpp-snapshot-manager")
        assert manager.parent is pane
        assert (
            manager.region.y
            > screen.query_one("#llamacpp-start-server-button").region.bottom
        )


def _app():
    """Build the test app.

    ``_build_test_app`` constructs the real ``TldwCli``, so its class-level
    ``CSS_PATH`` is loaded during ``App.__init__`` and the production bundle
    applies to mounted compositor assertions.
    """
    return _build_test_app()


def _rail_rows(screen):
    return list(screen.query(".lab-rail-row").results(Button))


def _assert_painted_inside(app, widget, parent) -> None:
    """Assert real compositor visibility inside one owning region."""
    assert widget in app.screen._compositor.visible_widgets
    assert widget.is_on_screen
    assert widget.region.width > 0 and widget.region.height > 0
    bounds = parent.content_region
    assert widget.region.x >= bounds.x
    assert widget.region.right <= bounds.right
    assert widget.region.y >= bounds.y
    assert widget.region.bottom <= bounds.bottom


def _remote_text(remote) -> str:
    """Return the current Remote presentation text without markup rendering."""
    return "\n".join(str(item.renderable) for item in remote.query(Static))


def _machine_snapshot(
    *,
    total_gib: int = 32,
    available_gib: int | None = 10,
    system_state: SystemMemoryState = SystemMemoryState.OBSERVED,
    system_reason: ProbeReason | None = None,
    accelerator_state: AcceleratorState = AcceleratorState.NOT_OBSERVED,
    accelerator_reason: ProbeReason | None = None,
    device_count: int = 0,
) -> MachineMemorySnapshot:
    """Build one complete bounded probe result for screen lifecycle tests."""
    has_capacity = system_state in {
        SystemMemoryState.OBSERVED,
        SystemMemoryState.PARTIAL,
    }
    accelerators = tuple(
        AcceleratorMemoryObservation(
            vendor="nvidia",
            label=f"Production evidence GPU {index} with a bounded long label",
            total_bytes=(index + 1) * 8 * GIB,
            shared=False,
            source=AcceleratorSource.NVIDIA_SMI,
        )
        for index in range(device_count)
    )
    return MachineMemorySnapshot(
        platform="linux",
        architecture="x86_64",
        system_state=system_state,
        accelerator_state=(
            AcceleratorState.OBSERVED if accelerators else accelerator_state
        ),
        total_bytes=total_gib * GIB if has_capacity else None,
        available_bytes=(
            available_gib * GIB if has_capacity and available_gib is not None else None
        ),
        memory_kind=MemoryKind.SYSTEM if has_capacity else MemoryKind.UNKNOWN,
        accelerators=accelerators,
        system_reason=system_reason,
        accelerator_reason=None if accelerators else accelerator_reason,
    )


def _machine_screen() -> LLMScreen:
    """Build only the screen-owned memory lifecycle state, without mounting UI."""
    screen = LLMScreen.__new__(LLMScreen)
    screen._machine_memory_snapshot = None
    screen._machine_memory_observed_label = None
    screen._machine_memory_observed_monotonic = None
    screen._machine_memory_wall_clock = lambda: datetime(2032, 4, 5, 9, 41)
    screen._machine_memory_monotonic_clock = lambda: 8_765.25
    screen._machine_memory_generation = 0
    screen._machine_memory_worker = None
    screen._machine_memory_active = False
    screen._machine_memory_failure = None
    screen._hydrate_remote_machine_memory = MagicMock(return_value=False)
    return screen


def test_first_machine_memory_request_starts_one_screen_worker() -> None:
    """Removing the no-duplicate guard would start two probes for one resolution."""
    screen = _machine_screen()
    worker = object()
    screen._run_machine_memory_probe = MagicMock(return_value=worker)

    LLMScreen._request_remote_machine_memory(screen, force=False)
    LLMScreen._request_remote_machine_memory(screen, force=False)

    assert screen._machine_memory_generation == 1
    assert screen._machine_memory_active is True
    assert screen._machine_memory_worker is worker
    screen._run_machine_memory_probe.assert_called_once_with(1)


def test_active_machine_memory_request_hydrates_without_starting_another_probe() -> (
    None
):
    """A remounted RemoteView must receive retained facts during an active probe."""
    screen = _machine_screen()
    screen._machine_memory_snapshot = _machine_snapshot()
    screen._machine_memory_generation = 1
    screen._machine_memory_active = True
    screen._run_machine_memory_probe = MagicMock()

    LLMScreen._request_remote_machine_memory(screen, force=False)

    screen._hydrate_remote_machine_memory.assert_called_once_with()
    screen._run_machine_memory_probe.assert_not_called()


def test_forced_machine_memory_recheck_advances_generation() -> None:
    """Treating a forced recheck as a duplicate would leave stale facts forever."""
    screen = _machine_screen()
    screen._machine_memory_snapshot = _machine_snapshot()
    screen._run_machine_memory_probe = MagicMock(side_effect=[object(), object()])

    LLMScreen._request_remote_machine_memory(screen, force=True)
    LLMScreen._request_remote_machine_memory(screen, force=True)

    assert screen._machine_memory_generation == 2
    assert [item.args for item in screen._run_machine_memory_probe.call_args_list] == [
        (1,),
        (2,),
    ]


def test_stale_machine_memory_result_cannot_replace_newer_snapshot() -> None:
    """Dropping the generation fence would publish an older probe completion."""
    screen = _machine_screen()
    screen._machine_memory_generation = 2
    current = _machine_snapshot(total_gib=32)
    screen._machine_memory_snapshot = current

    LLMScreen._apply_machine_memory_result(screen, 1, _machine_snapshot(total_gib=64))

    assert screen._machine_memory_snapshot is current
    screen._hydrate_remote_machine_memory.assert_not_called()


def test_machine_memory_failed_recheck_retains_last_valid_ram() -> None:
    """Replacing accepted RAM with an unavailable refresh would erase useful facts."""
    screen = _machine_screen()
    current = _machine_snapshot(total_gib=32)
    screen._machine_memory_snapshot = current
    screen._machine_memory_observed_label = "09:41"
    screen._machine_memory_generation = 3

    LLMScreen._apply_machine_memory_result(
        screen,
        3,
        _machine_snapshot(
            system_state=SystemMemoryState.UNAVAILABLE,
            system_reason=ProbeReason.MEMORY_UNAVAILABLE,
            available_gib=None,
        ),
    )

    assert screen._machine_memory_snapshot is current
    assert screen._machine_memory_observed_label == "09:41"
    assert screen._machine_memory_failure is ProbeReason.MEMORY_UNAVAILABLE
    assert screen._machine_memory_active is False
    screen._hydrate_remote_machine_memory.assert_called_once_with()


def test_machine_memory_partial_valid_ram_replaces_previous_observation() -> None:
    """Rejecting valid partial RAM would keep an obsolete installed-memory total."""
    screen = _machine_screen()
    screen._machine_memory_snapshot = _machine_snapshot(total_gib=32)
    screen._machine_memory_generation = 4
    partial = _machine_snapshot(
        total_gib=64,
        available_gib=None,
        system_state=SystemMemoryState.PARTIAL,
        system_reason=ProbeReason.MEMORY_UNAVAILABLE,
    )

    LLMScreen._apply_machine_memory_result(screen, 4, partial)

    assert screen._machine_memory_snapshot is partial
    assert screen._machine_memory_failure is None
    assert screen._machine_memory_observed_label is not None


def test_machine_memory_accelerator_failure_does_not_discard_valid_ram() -> None:
    """Coupling accelerator and RAM status would hide a valid capacity estimate."""
    screen = _machine_screen()
    screen._machine_memory_generation = 1
    result = _machine_snapshot(
        accelerator_state=AcceleratorState.NOT_OBSERVED,
        accelerator_reason=ProbeReason.COMMAND_TIMEOUT,
    )

    LLMScreen._apply_machine_memory_result(screen, 1, result)

    assert screen._machine_memory_snapshot is result
    assert screen._machine_memory_failure is None


def test_machine_memory_completion_is_retained_during_remote_remount_gap() -> None:
    """A missing RemoteView at completion must not lose the accepted snapshot."""
    screen = _machine_screen()
    screen._machine_memory_generation = 1
    result = _machine_snapshot(total_gib=64)

    LLMScreen._apply_machine_memory_result(screen, 1, result)

    assert screen._machine_memory_snapshot is result
    screen._hydrate_remote_machine_memory.assert_called_once_with()


def test_deferred_remote_mount_hydrates_machine_memory_without_another_probe() -> None:
    """Recomposition may hydrate retained state but must not observe twice."""
    screen = _machine_screen()
    screen._machine_memory_snapshot = _machine_snapshot()
    screen._audio_cpp_model_request_claim = None
    screen._model_install_active = False
    screen._model_install_last_progress = None
    screen._model_install_kind = None
    screen._external_operation_status = ""
    screen._remote_runtime_handoff = None
    screen._model_install_presentation_pending = MagicMock(return_value=False)
    screen._hydrate_external_status = MagicMock()
    screen._replay_remote_runtime_handoff = MagicMock()
    screen._run_machine_memory_probe = MagicMock()

    LLMScreen._on_deferred_views_mounted(screen)

    screen._hydrate_remote_machine_memory.assert_called_once_with()
    screen._run_machine_memory_probe.assert_not_called()


def test_machine_memory_worker_returns_bounded_result_on_event_thread(
    monkeypatch,
) -> None:
    """Applying directly from the worker thread would violate Textual ownership."""
    screen = _machine_screen()
    result = _machine_snapshot()
    screen._machine_memory_probe_factory = MagicMock(return_value=result)
    app = MagicMock()
    monkeypatch.setattr(LLMScreen, "app", property(lambda _screen: app))

    LLMScreen._run_machine_memory_probe.__wrapped__(screen, 7)

    app.call_from_thread.assert_called_once_with(
        screen._apply_machine_memory_result,
        7,
        result,
    )


@pytest.mark.asyncio
async def test_injected_memory_clocks_survive_failed_refresh_and_real_recompose() -> (
    None
):
    """Global time or view-owned timestamps would drift or disappear on remount."""
    from tldw_chatbook.UI.Screens.model_remote_view import RemoteView

    observed_wall = datetime(2032, 4, 5, 9, 41, 37)
    observed_monotonic = 8_765.25
    app = _app()
    async with app.run_test(size=(80, 24)) as pilot:
        screen = LLMScreen(
            app,
            machine_memory_wall_clock=lambda: observed_wall,
            machine_memory_monotonic_clock=lambda: observed_monotonic,
        )
        await app.push_screen(screen)
        assert await _wait_for(lambda: bool(screen.query(LLMManagementWindow)), pilot)
        assert await _wait_for(
            lambda: screen.query_one(LLMManagementWindow).is_mounted, pilot
        )
        screen.query_one(LLMManagementWindow).active_view = "remote"
        assert await _wait_for(lambda: bool(screen.query(RemoteView)), pilot)
        accepted = _machine_snapshot(total_gib=32)
        screen._machine_memory_generation = 1

        screen._apply_machine_memory_result(1, accepted)

        assert screen._machine_memory_observed_label == "09:41"
        assert screen._machine_memory_observed_monotonic == observed_monotonic

        screen._machine_memory_generation = 2
        screen._apply_machine_memory_result(
            2,
            _machine_snapshot(
                system_state=SystemMemoryState.UNAVAILABLE,
                system_reason=ProbeReason.MEMORY_UNAVAILABLE,
                available_gib=None,
            ),
        )
        assert screen._machine_memory_snapshot is accepted
        assert screen._machine_memory_observed_label == "09:41"
        assert screen._machine_memory_observed_monotonic == observed_monotonic

        old_window = screen.query_one(LLMManagementWindow)
        old_remote = screen.query_one(RemoteView)
        await screen.recompose()
        assert await _wait_for(
            lambda: bool(screen.query(LLMManagementWindow))
            and screen.query_one(LLMManagementWindow) is not old_window
            and screen.query_one(LLMManagementWindow).is_mounted,
            pilot,
            attempts=500,
        )
        screen.query_one(LLMManagementWindow).active_view = "remote"
        assert await _wait_for(
            lambda: (
                bool(screen.query(RemoteView))
                and screen.query_one(RemoteView) is not old_remote
                and screen.query_one(RemoteView)._machine_snapshot is accepted
            ),
            pilot,
            attempts=500,
        )
        fresh_remote = screen.query_one(RemoteView)
        assert fresh_remote._machine_presentation.failure_line == (
            "Recheck failed · using memory observed at 09:41"
        )
        assert screen._machine_memory_observed_label == "09:41"
        assert screen._machine_memory_observed_monotonic == observed_monotonic


@pytest.mark.asyncio
async def test_audio_cpp_curated_provision_is_install_only(monkeypatch):
    from tldw_chatbook.Model_Artifacts import acquisition as acquisition_module
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef

    reference = ArtifactRef("audio-cpp-model", "a" * 40, "f16")
    calls = []

    class _Acquisition:
        def __init__(self, service) -> None:
            self.service = service

        async def provision(self, root, consent, registry, **kwargs):
            calls.append((root, kwargs))
            return root

    monkeypatch.setattr(acquisition_module, "ArtifactAcquisitionService", _Acquisition)
    screen = LLMScreen(MagicMock())
    screen._model_install_service = MagicMock()
    screen._model_install_registry = MagicMock()
    screen._model_install_registry.descriptor.return_value.consumer = "audio_cpp"
    screen._model_install_sources = {}
    report = MagicMock(root=reference)

    assert await screen._provision_curated(report) == reference
    assert calls[0][1]["activate"] is False


@pytest.mark.asyncio
async def test_ordinary_curated_provision_keeps_activation_default(monkeypatch):
    from tldw_chatbook.Model_Artifacts import acquisition as acquisition_module
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef

    reference = ArtifactRef("parakeet-v2", "a" * 40, "int8")
    calls = []

    class _Acquisition:
        def __init__(self, service) -> None:
            self.service = service

        async def provision(self, root, consent, registry, **kwargs):
            calls.append((root, kwargs))
            return root

    monkeypatch.setattr(acquisition_module, "ArtifactAcquisitionService", _Acquisition)
    screen = LLMScreen(MagicMock())
    screen._model_install_service = MagicMock()
    screen._model_install_registry = MagicMock()
    screen._model_install_registry.descriptor.return_value.consumer = "stt"
    screen._model_install_sources = {}
    report = MagicMock(root=reference)

    assert await screen._provision_curated(report) == reference
    assert "activate" not in calls[0][1]


def test_audio_cpp_terminal_result_is_one_time_and_skips_runtime_mutation(
    monkeypatch,
):
    from pathlib import Path

    from tldw_chatbook.Model_Artifacts.service import ArtifactRef
    from tldw_chatbook.UI.Navigation.audio_cpp_model_handoff import (
        AudioCppModelLibraryRequest,
        AudioCppModelLibraryResult,
    )
    from tldw_chatbook.UI.Navigation.pending_handoff_store import (
        HandoffChannel,
        PendingHandoffStore,
    )
    from tldw_chatbook.UI.Screens import llm_screen as module

    store = PendingHandoffStore()
    request = AudioCppModelLibraryRequest(token="request-token", draft_revision=4)
    store.stage(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST, request)
    claim = store.claim(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST)
    assert claim is not None
    app_instance = MagicMock(pending_handoffs=store)
    screen = module.LLMScreen(app_instance)
    screen._audio_cpp_model_request_claim = claim
    screen.notify = MagicMock()
    screen._deliver_curated = MagicMock()
    view = MagicMock()
    screen._curated_view = MagicMock(return_value=view)
    reference = ArtifactRef("audio-cpp-model", "a" * 40, "f16")
    screen._model_install_kind = "curated"
    screen._model_install_reference = reference
    screen._model_install_service = MagicMock()
    screen._model_install_registry = MagicMock()
    screen._model_install_sources = {}
    screen._model_install_pending_report = object()
    result = AudioCppModelLibraryResult(
        token=request.token,
        draft_revision=request.draft_revision,
        artifact_id=reference.artifact_id,
        revision=reference.revision,
        variant=reference.variant,
        canonical_root=str(Path("/managed/audio-cpp-model")),
    )

    module.LLMScreen._apply_audio_cpp_provision_result(screen, result, None)

    returned = store.claim(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT)
    assert returned is not None
    assert returned.value == result
    assert returned.value is not result
    assert store.claim(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST) is None
    assert store.acknowledge(claim) is False
    screen.notify.assert_called_once_with(
        "Installed — ready for review", severity="information"
    )
    app_instance._ensure_parakeet_source_service.assert_not_called()
    app_instance.start_server.assert_not_called()
    app_instance.set_default_model.assert_not_called()


def test_audio_cpp_standalone_terminal_is_installed_without_false_return() -> None:
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef
    from tldw_chatbook.UI.Navigation.pending_handoff_store import (
        HandoffChannel,
        PendingHandoffStore,
    )
    from tldw_chatbook.UI.Screens import llm_screen as module

    store = PendingHandoffStore()
    app_instance = MagicMock(pending_handoffs=store)
    screen = module.LLMScreen(app_instance)
    screen.notify = MagicMock()
    screen._deliver_curated = MagicMock()
    screen._curated_view = MagicMock(return_value=None)
    screen._model_install_kind = "curated"
    screen._model_install_reference = ArtifactRef("audio-cpp-model", "a" * 40, "f16")
    screen._model_install_service = MagicMock()
    screen._model_install_registry = MagicMock()
    screen._model_install_sources = {}
    screen._model_install_pending_report = object()

    module.LLMScreen._apply_audio_cpp_standalone_result(screen)

    assert store.claim(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT) is None
    screen.notify.assert_called_once_with("Installed", severity="information")
    app_instance._ensure_parakeet_source_service.assert_not_called()


def test_audio_cpp_terminal_rejects_a_foreign_result_without_staging() -> None:
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef
    from tldw_chatbook.UI.Navigation.audio_cpp_model_handoff import (
        AudioCppModelLibraryRequest,
        AudioCppModelLibraryResult,
    )
    from tldw_chatbook.UI.Navigation.pending_handoff_store import (
        HandoffChannel,
        PendingHandoffStore,
    )
    from tldw_chatbook.UI.Screens import llm_screen as module

    store = PendingHandoffStore()
    store.stage(
        HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST,
        AudioCppModelLibraryRequest("expected-token", 4),
    )
    claim = store.claim(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST)
    assert claim is not None
    screen = module.LLMScreen(MagicMock(pending_handoffs=store))
    screen._audio_cpp_model_request_claim = claim
    screen.notify = MagicMock()
    screen._deliver_curated = MagicMock()
    view = MagicMock()
    screen._curated_view = MagicMock(return_value=view)
    reference = ArtifactRef("audio-cpp-model", "a" * 40, "f16")
    screen._model_install_kind = "curated"
    screen._model_install_reference = reference
    screen._model_install_service = MagicMock()
    screen._model_install_registry = MagicMock()
    screen._model_install_sources = {}
    screen._model_install_pending_report = object()
    foreign = AudioCppModelLibraryResult(
        "foreign-token",
        4,
        reference.artifact_id,
        reference.revision,
        reference.variant,
        "/managed/audio-cpp-model",
    )

    module.LLMScreen._apply_audio_cpp_provision_result(screen, foreign, None)

    assert store.claim(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT) is None
    replay = store.claim(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST)
    assert replay is not None
    assert replay.value.token == "expected-token"
    screen.notify.assert_called_once_with(
        "Installed, but the Settings return expired. Reopen Guided Settings "
        "and choose this package again.",
        severity="error",
    )
    delivered = screen._deliver_curated.call_args.args[0]
    assert delivered.succeeded is True
    view.finish_install.assert_called_once_with(
        "Installed, but the Settings return expired. Reopen Guided Settings "
        "and choose this package again."
    )


def test_audio_cpp_owner_failure_keeps_private_details_out_of_logs(monkeypatch) -> None:
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef
    from tldw_chatbook.UI.Navigation.audio_cpp_model_handoff import (
        AudioCppModelInstallOperation,
    )
    from tldw_chatbook.UI.Screens import llm_screen as module

    fake_logger = MagicMock()
    monkeypatch.setattr(module, "logger", fake_logger)
    screen = module.LLMScreen(MagicMock())
    screen._model_install_reference = ArtifactRef("audio-cpp-model", "a" * 40, "f16")
    screen._audio_cpp_operation_expects_return = True
    screen._apply_audio_cpp_provision_result = MagicMock()
    operation = AudioCppModelInstallOperation(threading.Event())
    screen._audio_cpp_model_install_operation = operation
    monkeypatch.setattr(
        module.LLMScreen,
        "is_attached",
        property(lambda _self: True),
    )

    module.LLMScreen._audio_cpp_operation_settled(
        screen,
        operation,
        None,
        RuntimeError("PRIVATE-AUDIO-PATH-/secret/model"),
        False,
    )

    logged = " ".join(str(value) for value in fake_logger.error.call_args.args)
    assert "RuntimeError" in logged
    assert "PRIVATE-AUDIO-PATH" not in logged
    screen._apply_audio_cpp_provision_result.assert_called_once()
    assert "PRIVATE-AUDIO-PATH" not in str(
        screen._apply_audio_cpp_provision_result.call_args
    )


@pytest.mark.parametrize(
    ("failure", "expected_order"),
    (
        ("stage", ["stage", "release"]),
        ("ack", ["stage", "ack", "clear", "release"]),
    ),
)
def test_audio_cpp_result_stage_and_ack_fail_closed_without_orphan(
    monkeypatch,
    failure,
    expected_order,
) -> None:
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef
    from tldw_chatbook.UI.Navigation.audio_cpp_model_handoff import (
        AudioCppModelLibraryRequest,
        AudioCppModelLibraryResult,
    )
    from tldw_chatbook.UI.Navigation.pending_handoff_store import (
        HandoffChannel,
        HandoffValueError,
        PendingHandoffStore,
    )
    from tldw_chatbook.UI.Screens import llm_screen as module

    store = PendingHandoffStore()
    request = AudioCppModelLibraryRequest("atomic-request", 5)
    store.stage(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST, request)
    claim = store.claim(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST)
    assert claim is not None
    reference = ArtifactRef("audio-cpp-model", "a" * 40, "f16")
    result = AudioCppModelLibraryResult(
        request.token,
        request.draft_revision,
        reference.artifact_id,
        reference.revision,
        reference.variant,
        "/managed/audio-cpp-model",
    )
    order: list[str] = []
    real_stage = PendingHandoffStore.stage
    real_ack = PendingHandoffStore.acknowledge
    real_clear = PendingHandoffStore.clear_pending
    real_release = PendingHandoffStore.release

    def stage(self, channel, value):
        if channel is HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT:
            order.append("stage")
            if failure == "stage":
                raise HandoffValueError("injected stage failure")
        return real_stage(self, channel, value)

    def acknowledge(self, current):
        order.append("ack")
        return False if failure == "ack" else real_ack(self, current)

    def clear_pending(self, channel):
        order.append("clear")
        return real_clear(self, channel)

    def release(self, current):
        order.append("release")
        return real_release(self, current)

    monkeypatch.setattr(PendingHandoffStore, "stage", stage)
    monkeypatch.setattr(PendingHandoffStore, "acknowledge", acknowledge)
    monkeypatch.setattr(PendingHandoffStore, "clear_pending", clear_pending)
    monkeypatch.setattr(PendingHandoffStore, "release", release)
    screen = module.LLMScreen(MagicMock(pending_handoffs=store))
    screen._audio_cpp_model_request_claim = claim
    screen.notify = MagicMock()
    screen._deliver_curated = MagicMock()
    screen._curated_view = MagicMock(return_value=None)
    screen._model_install_kind = "curated"
    screen._model_install_reference = reference
    screen._model_install_service = MagicMock()
    screen._model_install_registry = MagicMock()
    screen._model_install_sources = {}
    screen._model_install_pending_report = object()

    module.LLMScreen._apply_audio_cpp_provision_result(screen, result, None)

    assert order == expected_order
    assert store.claim(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_RESULT) is None
    replay = store.claim(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST)
    assert replay is not None and replay.value == request


def test_detached_audio_cpp_failure_releases_request_and_settles_lifecycle():
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef
    from tldw_chatbook.UI.Navigation.audio_cpp_model_handoff import (
        AudioCppModelLibraryRequest,
    )
    from tldw_chatbook.UI.Navigation.pending_handoff_store import (
        HandoffChannel,
        PendingHandoffStore,
    )
    from tldw_chatbook.UI.Screens import llm_screen as module

    store = PendingHandoffStore()
    request = AudioCppModelLibraryRequest(token="request-token", draft_revision=4)
    store.stage(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST, request)
    claim = store.claim(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST)
    assert claim is not None
    screen = module.LLMScreen(MagicMock(pending_handoffs=store))
    screen._audio_cpp_model_request_claim = claim
    screen.notify = MagicMock()
    screen._deliver_curated = MagicMock()
    screen._curated_view = MagicMock(return_value=None)
    screen._model_install_kind = "curated"
    screen._model_install_reference = ArtifactRef("audio-cpp-model", "a" * 40, "f16")
    screen._model_install_service = MagicMock()
    screen._model_install_registry = MagicMock()
    screen._model_install_sources = {}
    screen._model_install_pending_report = object()

    module.LLMScreen._apply_audio_cpp_provision_result(
        screen, None, "Model installation was cancelled."
    )

    replay = store.claim(HandoffChannel.AUDIO_CPP_MODEL_LIBRARY_REQUEST)
    assert replay is not None
    assert replay.value == request
    assert screen._audio_cpp_model_request_claim is None
    assert screen._model_install_kind is None
    screen.notify.assert_called_once_with(
        "Model installation was cancelled.", severity="error"
    )


@pytest.mark.asyncio
async def test_all_provider_and_model_rows_live_in_the_rail():
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        await pilot.pause()
        await pilot.pause()
        keys = [row.lab_view_key for row in _rail_rows(screen)]
        assert keys == [
            "llama-cpp",
            "llamafile",
            "ollama",
            "vllm",
            "onnx",
            "transformers",
            "mlx-lm",
            "curated",
            "installed",
            "external",
            "remote",
        ]


@pytest.mark.asyncio
async def test_empty_models_recovery_routes_hold_at_80_columns(
    tmp_path,
    monkeypatch,
):
    """Downloader retirement leaves explicit, distinct, in-bounds recovery."""
    from tldw_chatbook.Model_Artifacts.service import ArtifactDiskUsage
    from tldw_chatbook.Model_Artifacts.remote_huggingface import (
        HuggingFaceRemoteAdapter,
    )
    from tldw_chatbook.UI.Screens.model_external_view import ExternalModelView
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    search = AsyncMock(return_value=())
    monkeypatch.setattr(HuggingFaceRemoteAdapter, "search", search)
    app = _app()
    app._parakeet_source_service = _FakeExternalSourceService()
    async with app.run_test(size=(80, 40)) as pilot:
        assert app.CSS_PATH == TldwCli.CSS_PATH
        screen = await _models_screen(app)
        assert await _wait_for(
            lambda: bool(screen.query("#installed-models-view")), pilot
        )

        rows = _rail_rows(screen)
        keys = [row.lab_view_key for row in rows]
        assert "download-models" not in keys
        assert not screen.query("#lab-models-row-download-models")
        rail = screen.query_one("#lab-rail")
        for row in rows:
            _assert_painted_inside(app, row, rail)

        window = screen.query_one(LLMManagementWindow)
        assert "download-models" not in window.view_mapping
        assert not window.query("#llm-view-download-models")

        installed = window.query_one("#installed-models-view", InstalledView)
        legacy_root = tmp_path / "empty-legacy-root"
        legacy_root.mkdir()
        managed_root = tmp_path / "managed-store"
        service = MagicMock()
        service.list_installed.return_value = ()
        service.disk_usage.return_value = ArtifactDiskUsage(0, 0, 64 * 1024 * 1024)
        service.artifacts_path = managed_root
        installed._service_factory = lambda: service
        installed._legacy_dir = legacy_root
        installed_row = next(row for row in rows if row.lab_view_key == "installed")
        installed_row.press()
        assert await _wait_for(lambda: installed._loaded, pilot)

        recovery = next(
            item
            for item in installed.query(Static)
            if str(item.renderable).startswith("No managed or legacy models found.")
        )
        import_button = installed.query_one("#installed-models-import-gguf", Button)
        installed_parent = window.query_one("#llm-view-installed")
        assert await _wait_for(
            lambda: import_button in app.screen._compositor.visible_widgets,
            pilot,
            attempts=500,
        )
        assert import_button in app.screen._compositor.visible_widgets
        assert import_button.is_on_screen
        assert import_button.region.right <= app.size.width
        assert import_button.region.bottom <= app.size.height
        _assert_painted_inside(app, recovery, installed_parent)
        assert import_button.can_focus

        for provider, view_key in (
            ("llamacpp", "llama-cpp"),
            ("llamafile", "llamafile"),
        ):
            next(row for row in rows if row.lab_view_key == view_key).press()
            await pilot.pause()
            mode = window.query_one(f"#{provider}-gguf-source-mode", Select)
            labels = tuple(str(label) for label, _value in mode._options)
            assert "External GGUF" in labels
            if mode.value != "external":
                mode.value = "external"
                await pilot.pause()
            external_region = window.query_one(f"#{provider}-gguf-external-region")
            external_region.scroll_visible()
            await pilot.pause()
            model_path = window.query_one(f"#{provider}-model-path", Input)
            browse = window.query_one(f"#{provider}-browse-model-button", Button)
            view = window.query_one(f"#llm-view-{view_key}")
            _assert_painted_inside(app, model_path, view)
            _assert_painted_inside(app, browse, view)
            source_copy = "\n".join(
                str(item.renderable) for item in external_region.query(Static)
            )
            assert "used in place" in source_copy
            assert "not imported, copied, deleted, or selected globally" in source_copy

        external_row = next(row for row in rows if row.lab_view_key == "external")
        assert str(external_row.label) == "External"
        external_row.press()
        await pilot.pause()
        external = window.query_one("#external-models-view", ExternalModelView)
        external_text = "\n".join(
            str(item.renderable) for item in external.query(Static)
        )
        assert "external Parakeet sources" in external_text
        assert "GGUF" not in external_text

        transformers_row = next(
            row for row in rows if row.lab_view_key == "transformers"
        )
        transformers_row.press()
        await pilot.pause()
        transformers_view = window.query_one("#llm-view-transformers")
        for selector in (
            "#transformers-models-dir-path",
            "#transformers-browse-models-dir-button",
            "#transformers-list-local-models-button",
        ):
            control = window.query_one(selector)
            control.scroll_visible()
            await pilot.pause()
            _assert_painted_inside(app, control, transformers_view)
        assert not window.query("#transformers-download-model-button")

        remote_row = next(row for row in rows if row.lab_view_key == "remote")
        remote_row.press()
        await pilot.pause()
        assert window.active_view == "remote"
        assert window.query_one("#remote-models-view")
        search.assert_not_awaited()

    expected = (
        "No managed or legacy models found. Use Import GGUF… for a managed copy, "
        "or choose External GGUF under Llama.cpp or Llamafile to use a file in place."
    )
    assert str(recovery.renderable) == expected
    assert str(tmp_path) not in str(recovery.renderable)


@pytest.mark.asyncio
async def test_remote_drill_down_install_action_stays_inside_real_models_body_at_80_columns():
    """The production body uses one complete pane at its measured width."""
    from tldw_chatbook.UI.Screens.model_remote_view import RemoteView

    app = _app()
    async with app.run_test(size=(80, 24)) as pilot:
        screen = await _models_screen(app)
        assert await _wait_for(lambda: bool(screen.query("#remote-models-view")), pilot)

        remote_row = next(
            row for row in _rail_rows(screen) if row.lab_view_key == "remote"
        )
        remote_row.press()
        assert await _wait_for(
            lambda: screen.query_one(LLMManagementWindow).active_view == "remote",
            pilot,
        )

        window = screen.query_one(LLMManagementWindow)
        remote = window.query_one("#remote-models-view", RemoteView)
        resolved = _resolved_remote_model()
        query = remote.query_one("#remote-model-query", Input)
        query.value = resolved.repository
        remote._resolve_generation = 1
        remote._apply_resolve_result(
            1,
            resolved.repository,
            resolved.repository,
            resolved,
            None,
        )
        remote._show_repository_detail()
        assert await _wait_for(
            lambda: (
                remote.has_class("-single-pane")
                and remote.query_one(".remote-detail-pane").display
                and bool(remote.query("#remote-variant-filter"))
                and bool(remote.query(".remote-candidate"))
            ),
            pilot,
        )

        parent = window.query_one("#llm-view-remote")
        results_pane = remote.query_one(".remote-results-pane")
        detail_pane = remote.query_one(".remote-detail-pane")
        variant_filter = remote.query_one("#remote-variant-filter", Input)
        variant_sort = remote.query_one("#remote-variant-sort", Select)
        candidate = remote.query_one(".remote-candidate", Button)
        selection = remote.query_one("#remote-model-selection", Static)
        install = remote.query_one("#remote-model-install", Button)

        assert remote.has_class("-single-pane")
        assert results_pane.display is False
        assert results_pane not in app.screen._compositor.visible_widgets
        assert detail_pane.display is True
        _assert_painted_inside(app, detail_pane, parent)
        back = remote.query_one("#remote-back-to-results", Button)
        _assert_painted_inside(app, back, parent)

        for control in (variant_filter, variant_sort, candidate):
            control.scroll_visible(
                animate=False,
                immediate=True,
                force=True,
                top=True,
            )
            assert await _wait_for(
                lambda control=control: (
                    control in app.screen._compositor.visible_widgets
                ),
                pilot,
            )
            _assert_painted_inside(app, control, parent)

        app.screen.set_focus(candidate)
        assert await _wait_for(lambda: app.focused is candidate, pilot)
        _assert_painted_inside(app, candidate, parent)
        await pilot.press("enter")
        assert await _wait_for(
            lambda: str(selection.renderable).startswith("Selected: model-q4.gguf"),
            pilot,
        )
        _assert_painted_inside(app, selection, parent)

        await pilot.press("tab")
        assert await _wait_for(lambda: app.focused is install, pilot)
        _assert_painted_inside(app, install, parent)


@pytest.mark.asyncio
async def test_remote_memory_scenarios_survive_recompose_at_80_columns():
    """Production rails, drill-down, refresh, and remount retain memory facts."""
    from dataclasses import replace

    from tldw_chatbook.Model_Artifacts.remote_huggingface import RemoteModelSummary
    from tldw_chatbook.UI.Screens.model_remote_view import RemoteView

    repository = (
        "publisher-with-long-name/model-with-an-even-longer-exact-repository-name"
    )
    filename = f"models/{'long-reviewed-variant-' * 7}Q4_K_M.gguf"
    resolved = _resolved_remote_model(
        repository,
        filename=filename,
        total_bytes=4 * GIB,
    )
    exact_resolved = replace(resolved, warnings=("exact-resolution-complete",))
    summary = RemoteModelSummary(
        repository=repository,
        private=False,
        gated="none",
        downloads=12_345,
        likes=678,
        last_modified="2026-08-01T00:00:00Z",
    )

    class _Adapter:
        def __init__(self) -> None:
            self.search_calls: list[str] = []
            self.resolve_calls: list[str] = []

        async def search(self, query: str, *, token=None):
            self.search_calls.append(query)
            return (summary,)

        async def resolve(self, requested: str, *, token=None):
            self.resolve_calls.append(requested)
            return resolved if len(self.resolve_calls) == 1 else exact_resolved

    class _Resolver:
        def resolve(self, _repository: str) -> None:
            return None

    accepted = _machine_snapshot(total_gib=32, available_gib=10, device_count=3)
    refreshed = _machine_snapshot(total_gib=32, available_gib=10, device_count=3)
    stale = _machine_snapshot(total_gib=64, available_gib=64)
    probe_starts = (threading.Event(), threading.Event())
    probe_releases = (threading.Event(), threading.Event())
    probe_results = (accepted, refreshed)
    probe_calls: list[int] = []

    def observe_memory() -> MachineMemorySnapshot:
        index = len(probe_calls)
        probe_calls.append(index)
        probe_starts[index].set()
        if not probe_releases[index].wait(10):
            raise RuntimeError("test-controlled memory probe was not released")
        return probe_results[index]

    async def assert_painted(control, parent, pilot, app) -> None:
        control.scroll_visible(
            animate=False,
            immediate=True,
            force=True,
            top=True,
        )
        assert await _wait_for(
            lambda: control in app.screen._compositor.visible_widgets,
            pilot,
        ), (
            control.id,
            control.region,
            control.display,
            getattr(control, "disabled", None),
            remote.query_one("#remote-model-details").scroll_offset,
            remote.query_one("#remote-model-details").content_region,
        )
        _assert_painted_inside(app, control, parent)

    async def assert_scroll_section_painted(
        section,
        viewport,
        parent,
        expected_widgets,
        pilot,
        app,
    ) -> None:
        """Prove a tall scroll section and its expected copy paint in slices."""

        def painted_intersection(widget) -> bool:
            clipped = widget.region.intersection(viewport.content_region)
            clipped = clipped.intersection(parent.content_region)
            return clipped.width > 0 and clipped.height > 0

        section.scroll_visible(
            animate=False,
            immediate=True,
            force=True,
            top=True,
        )
        assert await _wait_for(
            lambda: (
                section in app.screen._compositor.visible_widgets
                and painted_intersection(section)
            ),
            pilot,
        )
        assert section.region.x >= parent.content_region.x
        assert section.region.right <= parent.content_region.right

        for widget, expected_text in expected_widgets:
            widget.scroll_visible(
                animate=False,
                immediate=True,
                force=True,
                top=True,
            )
            assert await _wait_for(
                lambda widget=widget: (
                    widget in app.screen._compositor.visible_widgets
                    and painted_intersection(widget)
                ),
                pilot,
            )
            assert expected_text in str(widget.renderable)

    async def assert_exact_filename_painted(parent, pilot, app) -> None:
        """Read the current filename from painted compositor cells, not widget state."""
        from textual.strip import Strip

        viewport = remote.query_one("#remote-model-details")
        filename_widget: Static | None = None
        last_geometry: tuple[object, ...] = ()

        def painted_region(widget: Static):
            clipped = widget.content_region.intersection(viewport.content_region)
            return clipped.intersection(parent.content_region)

        def filename_is_painted() -> bool:
            nonlocal filename_widget, last_geometry
            current = remote.query_one(".remote-variant-filename", Static)
            if current is not filename_widget:
                filename_widget = current
            clipped = painted_region(current)
            visible = current in app.screen._compositor.visible_widgets
            last_geometry = (
                current.region,
                current.content_region,
                viewport.content_region,
                viewport.scroll_offset,
                clipped,
                visible,
            )
            if not visible or clipped.width <= 0 or clipped.height <= 0:
                current.scroll_visible(
                    animate=False,
                    immediate=True,
                    force=True,
                    top=True,
                )
                return False
            return True

        assert await _wait_for(
            filename_is_painted,
            pilot,
        ), last_geometry
        assert filename_widget is not None
        assert remote.query_one(".remote-variant-filename", Static) is filename_widget
        clipped = painted_region(filename_widget)
        assert clipped == filename_widget.content_region
        assert clipped.height > 1
        assert clipped.x >= viewport.content_region.x
        assert clipped.right <= viewport.content_region.right
        assert clipped.x >= parent.content_region.x
        assert clipped.right <= parent.content_region.right

        update = app.screen._compositor.render_full_update()
        painted_rows: list[str] = []
        for screen_y in range(clipped.y, clipped.bottom):
            line = Strip.join(update.strips[screen_y - update.region.y])
            painted_rows.append(
                line.crop(
                    clipped.x - update.region.x,
                    clipped.right - update.region.x,
                ).text.rstrip()
            )
        assert all(painted_rows)
        painted_filename = "".join(painted_rows)
        assert "…" not in painted_filename
        assert painted_filename == filename

    def current_candidate_ready(remote) -> bool:
        candidates = list(remote.query(".remote-candidate").results(Button))
        return (
            len(candidates) == 1
            and candidates[0].display
            and candidates[0].region.width > 0
            and candidates[0].region.height > 0
        )

    adapter = _Adapter()
    app = _app()
    try:
        async with app.run_test(size=(80, 24)) as pilot:
            assert app.CSS_PATH == TldwCli.CSS_PATH
            screen = await _models_screen(app)
            assert await _wait_for(lambda: bool(screen.query(RemoteView)), pilot)
            window = screen.query_one(LLMManagementWindow)
            remote = screen.query_one(RemoteView)
            remote._adapter_factory = lambda: adapter
            remote._credential_resolver_factory = _Resolver
            screen._machine_memory_probe_factory = observe_memory

            remote_row = next(
                row for row in _rail_rows(screen) if row.lab_view_key == "remote"
            )
            remote_row.press()
            assert await _wait_for(lambda: window.active_view == "remote", pilot)
            parent = window.query_one("#llm-view-remote")

            rail = screen.query_one("#lab-rail")
            rail_handle = screen.query_one("#lab-rail-handle")
            assert rail.display is True
            assert rail_handle.display is False
            assert await _wait_for(
                lambda: (
                    0 < remote.content_region.width < 72
                    and remote.has_class("-single-pane")
                ),
                pilot,
            )
            verified_rail_state_widths = {
                "expanded": remote.content_region.width,
            }
            screen.query_one("#lab-rail-collapse", Button).press()
            assert await _wait_for(
                lambda: (
                    not rail.display
                    and rail_handle.display
                    and verified_rail_state_widths["expanded"]
                    < remote.content_region.width
                    < 72
                    and remote.has_class("-single-pane")
                ),
                pilot,
            )
            verified_rail_state_widths["collapsed"] = remote.content_region.width

            query = remote.query_one("#remote-model-query", Input)
            query.value = "memory model"
            remote.query_one("#remote-model-search", Button).press()
            assert await _wait_for(lambda: bool(remote.query(".remote-result")), pilot)
            result = remote.query_one(".remote-result", Button)
            result.focus()
            assert await _wait_for(lambda: app.focused is result, pilot)
            result.press()
            assert await _wait_for(
                lambda: (
                    remote.query_one(".remote-detail-pane").display
                    and probe_starts[0].is_set()
                    and adapter.resolve_calls == [repository]
                ),
                pilot,
            )
            assert remote.query_one(".remote-results-pane").display is False
            assert "Machine memory: Checking local memory…" in _remote_text(remote)
            assert "Memory scenario: Checking local memory…" in _remote_text(remote)
            assert repository in _remote_text(remote)

            back = remote.query_one("#remote-back-to-results", Button)
            await assert_painted(back, parent, pilot, app)
            assert not rail.display
            assert (
                remote.content_region.width == verified_rail_state_widths["collapsed"]
            )
            back.press()
            assert await _wait_for(
                lambda: (
                    remote.query_one(".remote-results-pane").display
                    and app.focused is result
                ),
                pilot,
            )
            assert remote.query_one(".remote-result", Button) is result

            screen.query_one("#lab-rail-open", Button).press()
            assert await _wait_for(
                lambda: (
                    rail.display
                    and not rail_handle.display
                    and remote.content_region.width
                    == verified_rail_state_widths["expanded"]
                    and remote.content_region.width
                    < verified_rail_state_widths["collapsed"]
                    and remote.has_class("-single-pane")
                ),
                pilot,
            )

            query.value = repository
            remote.query_one("#remote-model-search", Button).press()
            assert await _wait_for(
                lambda: (
                    remote.query_one(".remote-detail-pane").display
                    and len(adapter.resolve_calls) == 2
                    and remote._resolved is exact_resolved
                    and "exact-resolution-complete" in _remote_text(remote)
                    and current_candidate_ready(remote)
                ),
                pilot,
            )
            assert adapter.search_calls == ["memory model"]
            assert adapter.resolve_calls == [repository, repository]
            assert probe_calls == [0]
            await assert_exact_filename_painted(parent, pilot, app)

            back = remote.query_one("#remote-back-to-results", Button)
            await assert_painted(back, parent, pilot, app)
            candidate = remote.query_one(".remote-candidate", Button)
            await assert_painted(candidate, parent, pilot, app)
            candidate.focus()
            assert await _wait_for(lambda: app.focused is candidate, pilot)
            probe_releases[0].set()
            assert await _wait_for(
                lambda: (
                    screen._machine_memory_snapshot is accepted
                    and "64K scenario within RAM budget" in _remote_text(remote)
                ),
                pilot,
            )
            assert remote.query_one(".remote-candidate", Button) is candidate
            assert app.focused is candidate
            assert "64K may need more free RAM now" in _remote_text(remote)
            assert "VRAM observed on 3 devices" in _remote_text(remote)

            panel = remote.query_one(".remote-machine-panel")
            toggle = remote.query_one("#remote-machine-details-toggle", Button)
            recheck = remote.query_one("#remote-machine-recheck", Button)
            model_details = remote.query_one("#remote-model-details")
            await assert_scroll_section_painted(
                panel,
                model_details,
                parent,
                (
                    (
                        remote.query_one("#remote-machine-headline", Static),
                        "Machine memory: 32.0 GiB RAM",
                    ),
                    (
                        remote.query_one("#remote-machine-evidence", Static),
                        "VRAM observed on 3 devices",
                    ),
                ),
                pilot,
                app,
            )
            for control in (toggle, candidate):
                await assert_painted(control, parent, pilot, app)
            candidate.focus()
            assert await _wait_for(lambda: app.focused is candidate, pilot)
            for _ in range(8):
                previous_focus = app.focused
                await pilot.press("shift+tab")
                assert await _wait_for(
                    lambda: app.focused is not previous_focus,
                    pilot,
                )
                if app.focused is recheck:
                    break
            assert app.focused is recheck
            assert recheck in app.screen._compositor.visible_widgets
            _assert_painted_inside(app, recheck, parent)

            exact_details = remote.query_one("#remote-machine-estimate-details", Static)
            assert exact_details.display is False
            toggle.press()
            assert await _wait_for(lambda: exact_details.display, pilot)
            assert all(
                device.label in str(exact_details.renderable)
                for device in accepted.accelerators
            )

            candidate.focus()
            await pilot.press("enter")
            selection = remote.query_one("#remote-model-selection", Static)
            install = remote.query_one("#remote-model-install", Button)
            assert await _wait_for(
                lambda: (
                    str(selection.renderable).startswith(f"Selected: {filename}")
                    and not install.disabled
                ),
                pilot,
            )
            await assert_painted(selection, parent, pilot, app)
            await assert_painted(install, parent, pilot, app)
            assert rail.display
            assert remote.content_region.width == verified_rail_state_widths["expanded"]
            verified_control_states = {"expanded"}

            screen.query_one("#lab-rail-collapse", Button).press()
            assert await _wait_for(
                lambda: (
                    not rail.display
                    and rail_handle.display
                    and screen.query_one(RemoteView) is remote
                    and remote.content_region.width
                    == verified_rail_state_widths["collapsed"]
                    and remote.content_region.width < 72
                    and remote.has_class("-single-pane")
                    and remote.query_one(".remote-candidate", Button) is candidate
                ),
                pilot,
            )

            collapsed_back = remote.query_one("#remote-back-to-results", Button)
            collapsed_panel = remote.query_one(".remote-machine-panel")
            collapsed_toggle = remote.query_one(
                "#remote-machine-details-toggle", Button
            )
            collapsed_recheck = remote.query_one("#remote-machine-recheck", Button)
            collapsed_candidate = remote.query_one(".remote-candidate", Button)
            collapsed_selection = remote.query_one("#remote-model-selection", Static)
            collapsed_install = remote.query_one("#remote-model-install", Button)
            collapsed_model_details = remote.query_one("#remote-model-details")
            assert collapsed_back is back
            assert collapsed_panel is panel
            assert collapsed_toggle is toggle
            assert collapsed_recheck is recheck
            assert collapsed_candidate is candidate
            assert collapsed_selection is selection
            assert collapsed_install is install

            await assert_exact_filename_painted(parent, pilot, app)

            await assert_scroll_section_painted(
                collapsed_panel,
                collapsed_model_details,
                parent,
                (
                    (
                        remote.query_one("#remote-machine-headline", Static),
                        "Machine memory: 32.0 GiB RAM",
                    ),
                    (
                        remote.query_one("#remote-machine-evidence", Static),
                        "VRAM observed on 3 devices",
                    ),
                ),
                pilot,
                app,
            )
            for control in (
                collapsed_back,
                collapsed_toggle,
                collapsed_recheck,
                collapsed_candidate,
                collapsed_selection,
                collapsed_install,
            ):
                await assert_painted(control, parent, pilot, app)
            assert str(collapsed_selection.renderable).startswith(
                f"Selected: {filename}"
            )
            assert collapsed_install.disabled is False
            verified_control_states.add("collapsed")

            initial_generation = screen._machine_memory_generation
            recheck.focus()
            await pilot.press("enter")
            assert await _wait_for(
                lambda: (
                    probe_starts[1].is_set()
                    and screen._machine_memory_generation == initial_generation + 1
                    and screen._machine_memory_active
                    and str(recheck.label) == "Checking…"
                    and recheck.disabled
                ),
                pilot,
            )
            screen._apply_machine_memory_result(initial_generation, stale)
            assert screen._machine_memory_snapshot is accepted
            assert screen._machine_memory_active is True
            assert "VRAM observed on 3 devices" in _remote_text(remote)

            probe_releases[1].set()
            assert await _wait_for(
                lambda: (
                    screen._machine_memory_snapshot is refreshed
                    and not screen._machine_memory_active
                    and not recheck.disabled
                ),
                pilot,
            )
            assert remote.query_one(".remote-candidate", Button) is candidate
            assert not install.disabled
            candidate.focus()
            await pilot.press("tab")
            assert await _wait_for(lambda: app.focused is install, pilot)
            await assert_painted(install, parent, pilot, app)

            old_remote = remote
            old_window = screen.query_one(LLMManagementWindow)
            await screen.recompose()
            assert await _wait_for(
                lambda: bool(screen.query(LLMManagementWindow))
                and screen.query_one(LLMManagementWindow) is not old_window,
                pilot,
                attempts=500,
            )
            screen.query_one(LLMManagementWindow).active_view = "remote"
            assert await _wait_for(
                lambda: (
                    bool(screen.query(RemoteView))
                    and screen.query_one(RemoteView) is not old_remote
                    and screen.query_one(RemoteView)._machine_snapshot is refreshed
                ),
                pilot,
                attempts=500,
            )
            fresh_remote = screen.query_one(RemoteView)
            assert fresh_remote._machine_presentation.action_disabled is False
            assert probe_calls == [0, 1]

            fresh_window = screen.query_one(LLMManagementWindow)
            fresh_remote._adapter_factory = lambda: adapter
            fresh_remote._credential_resolver_factory = _Resolver
            next(
                row for row in _rail_rows(screen) if row.lab_view_key == "remote"
            ).press()
            assert await _wait_for(
                lambda: (
                    fresh_window.active_view == "remote"
                    and fresh_remote.content_region.width
                    == verified_rail_state_widths["collapsed"]
                    and fresh_remote.content_region.width < 72
                    and fresh_remote.has_class("-single-pane")
                ),
                pilot,
            )
            fresh_remote.query_one("#remote-model-query", Input).value = repository
            fresh_remote.query_one("#remote-model-search", Button).press()
            assert await _wait_for(
                lambda: (
                    bool(fresh_remote.query(".remote-candidate"))
                    and "64K scenario within RAM budget" in _remote_text(fresh_remote)
                    and "VRAM observed on 3 devices" in _remote_text(fresh_remote)
                ),
                pilot,
            )
            assert probe_calls == [0, 1]
            assert set(verified_rail_state_widths) == {"expanded", "collapsed"}
            assert verified_rail_state_widths["expanded"] < 72
            assert (
                verified_rail_state_widths["expanded"]
                < verified_rail_state_widths["collapsed"]
                < 72
            )
            assert verified_control_states == {"expanded", "collapsed"}
    finally:
        for release in probe_releases:
            release.set()


@pytest.mark.asyncio
async def test_remote_completion_and_runtime_choice_fit_real_models_at_80_columns():
    """The complete adoption path must remain painted and keyboard-operable."""
    from tldw_chatbook.UI.Screens.model_remote_view import RemoteView
    from tldw_chatbook.Widgets.ModelArtifacts import ManagedGGUFRuntimeChoiceModal

    app = _app()
    async with app.run_test(size=(80, 24)) as pilot:
        screen = await _models_screen(app)
        assert await _wait_for(lambda: bool(screen.query("#remote-models-view")), pilot)

        remote_row = next(
            row for row in _rail_rows(screen) if row.lab_view_key == "remote"
        )
        remote_row.press()
        await pilot.pause()

        window = screen.query_one(LLMManagementWindow)
        remote = window.query_one("#remote-models-view", RemoteView)
        resolved = _resolved_remote_model()
        remote.query_one("#remote-model-query", Input).value = resolved.repository
        remote._resolve_generation = 1
        remote._apply_resolve_result(
            1,
            resolved.repository,
            resolved.repository,
            resolved,
            None,
        )
        remote._show_repository_detail()
        assert await _wait_for(
            lambda: (
                remote.query_one(".remote-detail-pane").display
                and bool(remote.query(".remote-candidate"))
            ),
            pilot,
        )
        remote.query_one(".remote-candidate", Button).press()
        await pilot.pause()

        reference = _remote_catalog().artifact.reference
        remote.finish_install(
            "Model downloaded and managed.",
            completed_reference=reference,
        )
        await pilot.pause()

        parent = window.query_one("#llm-view-remote")
        detail_pane = remote.query_one(".remote-detail-pane")
        open_installed = remote.query_one("#remote-model-open-installed", Button)
        configure = remote.query_one("#remote-model-configure-runtime", Button)
        assert await _wait_for(
            lambda: all(
                widget in app.screen._compositor.visible_widgets
                for widget in (detail_pane, open_installed, configure)
            ),
            pilot,
        )
        _assert_painted_inside(app, detail_pane, parent)
        _assert_painted_inside(app, open_installed, parent)
        _assert_painted_inside(app, configure, parent)
        assert open_installed.disabled is False
        assert configure.disabled is False
        assert (
            open_installed.region.bottom <= configure.region.y
            or open_installed.region.right <= configure.region.x
        )

        open_installed.focus()
        assert await _wait_for(lambda: app.focused is open_installed, pilot)
        _assert_painted_inside(app, open_installed, parent)
        await pilot.press("tab")
        assert await _wait_for(lambda: app.focused is configure, pilot)
        _assert_painted_inside(app, configure, parent)
        await pilot.press("enter")
        await pilot.pause()

        modal = app.screen
        assert isinstance(modal, ManagedGGUFRuntimeChoiceModal)
        dialog = modal.query_one(".managed-gguf-runtime-modal")
        llama_cpp = modal.query_one("#managed-gguf-runtime-llamacpp", Button)
        llamafile = modal.query_one("#managed-gguf-runtime-llamafile", Button)
        cancel = modal.query_one("#managed-gguf-runtime-cancel", Button)
        _assert_painted_inside(app, dialog, modal)
        for action in (llama_cpp, llamafile, cancel):
            _assert_painted_inside(app, action, dialog)
        assert app.focused is llama_cpp

        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is screen


@pytest.mark.asyncio
async def test_the_window_no_longer_carries_nav_buttons():
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        await pilot.pause()
        await pilot.pause()
        window = screen.query_one(LLMManagementWindow)
        assert not window.query(".llm-nav-button")


@pytest.mark.asyncio
async def test_the_rail_is_highlighted_on_arrival_before_any_press():
    """LLMManagementWindow.on_mount sets active_view itself, so a
    press-only implementation would leave the rail unhighlighted here."""
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        await pilot.pause()
        await pilot.pause()
        active = [r for r in _rail_rows(screen) if "is-active" in r.classes]
        assert len(active) == 1
        assert active[0].lab_view_key == "llama-cpp"


@pytest.mark.asyncio
async def test_pressing_a_rail_row_moves_both_the_body_and_the_highlight():
    """The highlight half fails SILENTLY -- query() returns empty rather than
    raising -- so a body-only assertion would pass with it dead."""
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        await pilot.pause()
        await pilot.pause()

        ollama = next(r for r in _rail_rows(screen) if r.lab_view_key == "ollama")
        ollama.press()
        await pilot.pause()

        window = screen.query_one(LLMManagementWindow)
        assert window.active_view == "ollama"
        assert "-active" in window.query_one("#llm-view-ollama").classes

        active = [r for r in _rail_rows(screen) if "is-active" in r.classes]
        assert len(active) == 1, "exactly one rail row must be highlighted"
        assert active[0].lab_view_key == "ollama"


@pytest.mark.asyncio
async def test_the_status_row_reports_running_servers():
    app = _app()
    app.llamacpp_server_process = None
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        await pilot.pause()
        await pilot.pause()
        chip = screen.query_one("#lab-status-chip-servers", Static)
        assert "Servers: none running" in str(chip.renderable)

        class _Alive:
            def poll(self):
                return None

        app.llamacpp_server_process = _Alive()
        screen.refresh_lab_status()
        await pilot.pause()
        assert "Servers: 1 running" in str(chip.renderable)


@pytest.mark.asyncio
async def test_model_install_progress_survives_switch_to_installed():
    """Curated progress remains visible in Installed and in the Lab status row.

    Delivers through ``LLMScreen._deliver_curated`` -- the screen's own
    entry point for a curated-install tick (TASK-1803: the screen owns the
    worker that would call this in production; ``CuratedView`` no longer
    posts ``InstallProgressed``/``InstallStatusChanged`` itself) -- rather
    than posting directly on ``CuratedView``, which nothing does any more.
    """
    from unittest.mock import MagicMock

    from tldw_chatbook.Model_Artifacts.acquisition import AcquisitionProgress
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView
    from tldw_chatbook.Widgets.ModelArtifacts import (
        InstallProgressed,
        InstallStatusChanged,
    )

    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app, populate_all=False)
        await pilot.pause()
        await pilot.pause()
        window = screen.query_one(LLMManagementWindow)
        window.active_view = "installed"
        assert await _wait_for(
            lambda: bool(window.query("#model-install-progress-phase")),
            pilot,
        )
        installed = window.query_one(InstalledView)
        installed.ensure_loaded = MagicMock()
        reference = ArtifactRef("parakeet-v2", "immutable-revision", "int8")
        progress = AcquisitionProgress(
            "fetch",
            reference,
            "encoder.onnx",
            512,
            1024,
        )

        screen._deliver_curated(InstallStatusChanged(reference, active=True))
        screen._deliver_curated(InstallProgressed(progress))
        await pilot.pause()

        installed_row = next(
            row for row in _rail_rows(screen) if row.lab_view_key == "installed"
        )
        installed_row.press()
        await pilot.pause()

        text = "\n".join(str(item.renderable) for item in installed.query(Static))
        chip = screen.query_one("#lab-status-chip-model-install", Static)
        assert "Downloading" in text
        assert "Model install: downloading" in str(chip.renderable)

        installed.ensure_loaded.reset_mock()
        screen._deliver_curated(
            InstallStatusChanged(reference, active=False, succeeded=True)
        )
        await pilot.pause()

        installed.ensure_loaded.assert_called_once_with(force=True)
        assert "Model install: idle" in str(chip.renderable)


@pytest.mark.asyncio
async def test_curated_install_progress_survives_a_screen_level_recompose(monkeypatch):
    """TASK-596 delta port / TASK-1803: a curated install must not go blank/stale.

    ``LabScreen.recompose()`` tears down and rebuilds the whole
    ``LLMManagementWindow`` -- ``CuratedView`` included -- which used to
    mean a curated install in progress lost its progress display for the
    rest of the run: the fresh ``CuratedView`` instance starts with no
    memory of the install, and (back when ``CuratedView`` owned its own
    preflight/provision worker) further progress ticks from the ORIGINAL
    instance's worker thread were posted to that now-closed instance and
    silently dropped, never reaching the fresh one either.

    TASK-1803 moved that worker to ``LLMScreen`` -- this screen owns the
    ``WorkerManager`` the download actually runs under, and a screen-level
    recompose never tears the *screen* down, only its body -- so there is
    no orphaned poster left to compensate for. This test exercises the
    real ``LLMScreen._provision_curated`` code path (not a simulation of
    it) against a stubbed ``ArtifactAcquisitionService`` so it controls
    exactly when a second progress tick fires relative to the recompose,
    then asserts both halves of the fix: the freshly (re)mounted view is
    hydrated with the last known progress (not blank), and a progress tick
    emitted AFTER the recompose -- delivered through this screen's own
    still-running worker, exactly as the real download would -- still
    reaches and updates the fresh view (not stale).

    Content-only, like this test: it cannot tell one render from three.
    See test_curated_install_progress_renders_exactly_once_per_tick below
    for the call-counting half of this fix (Review Important #1, fix
    round 1).

    Args:
        monkeypatch: pytest's monkeypatch fixture, used to stub the
            network-capable acquisition service so this test never
            performs real I/O; reverted automatically after the test.
    """
    import asyncio
    from unittest.mock import MagicMock

    import tldw_chatbook.Model_Artifacts.acquisition as acquisition_module
    from tldw_chatbook.Model_Artifacts.acquisition import AcquisitionProgress
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef
    from tldw_chatbook.UI.Screens.model_curated_view import CuratedView
    from tldw_chatbook.Widgets.ModelArtifacts.install_progress import (
        ModelInstallProgress,
    )

    reference = ArtifactRef("parakeet-v2", "immutable-revision", "int8")
    first_progress = AcquisitionProgress(
        "fetch", reference, "encoder.onnx", 100 * 1024 * 1024, 600 * 1024 * 1024
    )
    second_progress = AcquisitionProgress(
        "fetch", reference, "decoder.onnx", 400 * 1024 * 1024, 600 * 1024 * 1024
    )
    resume = asyncio.Event()

    class _FakeAcquisitionService:
        """Stands in for the real, network-capable acquisition service.

        Only ``.provision`` is exercised; it delivers one progress tick,
        waits for the test to force a screen-level recompose, then
        delivers a second tick -- all through the real ``progress``
        callback ``LLMScreen._provision_curated`` built, so
        ``_deliver_curated`` under test runs unmodified.
        """

        def __init__(self, _service) -> None:
            """Accept and discard the managed-store service the real
            constructor takes.

            Args:
                _service: The managed-store service (unused by the fake).
            """

        async def provision(self, root, consent, registry, *, sources, progress):
            """Deliver two progress ticks with the recompose in between.

            Args:
                root: The reference this closure is rooted at (unused; the
                    fake never inspects it beyond receiving it).
                consent: The granted consent object (unused).
                registry: The curated registry (unused).
                sources: File source map (unused).
                progress: The real ``deliver`` callback ``LLMScreen.
                    _provision_curated`` built; called synchronously,
                    twice, exactly as the real acquisition service would
                    call it from its own await points.

            Returns:
                A sentinel standing in for the real installed-path result;
                its value is never asserted on.
            """
            progress(first_progress)
            await resume.wait()
            progress(second_progress)
            return object()

    monkeypatch.setattr(
        acquisition_module, "ArtifactAcquisitionService", _FakeAcquisitionService
    )

    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app, populate_all=False)
        for _ in range(5):
            await pilot.pause()
        window = screen.query_one(LLMManagementWindow)
        window.active_view = "curated"
        assert await _wait_for(lambda: bool(window.query(CuratedView)), pilot)
        curated = window.query_one(CuratedView)

        # Mimics _confirm_curated_install's own setup (bypasses real
        # preflight/registry I/O) -- exercising _provision_curated itself
        # directly, on this test's own event loop rather than a real
        # background thread, so `resume` can pause it deterministically at
        # an exact point. State lives on the SCREEN now (TASK-1803), not
        # on the CuratedView instance -- it must survive the instance
        # being torn down below.
        # TASK-1914: _model_install_kind routes _model_install_progressed's
        # apply_progress call to the right view now that CuratedView and
        # RemoteView are both mounted at once -- set explicitly here since
        # this test drives _provision_curated directly rather than through
        # _curated_install_requested (which sets it in production).
        screen._model_install_kind = "curated"
        screen._model_install_reference = reference
        screen._model_install_service = MagicMock()
        screen._model_install_registry = MagicMock()
        screen._model_install_sources = {}
        fake_report = MagicMock(root=reference)

        provision_task = asyncio.create_task(screen._provision_curated(fake_report))
        await pilot.pause()
        await pilot.pause()

        def _progress_text(view: CuratedView) -> str:
            widget = view.query_one(
                "#curated-model-install-progress", ModelInstallProgress
            )
            detail = widget.query_one("#model-install-progress-detail", Static)
            return str(detail.renderable)

        assert "encoder.onnx" in _progress_text(curated)

        # A real screen-level recompose (LabScreen.recompose(), not
        # CuratedView's own internal refresh(recompose=True)) -- see
        # test_lab_frame.py::test_screen_level_recompose_repopulates_
        # rail_inspector_and_body for the same multi-pause shape this
        # mirrors.
        screen.refresh(recompose=True)
        await pilot.pause()
        await pilot.pause()
        await pilot.pause()
        await pilot.pause()
        await pilot.pause()

        fresh_window = screen.query_one(LLMManagementWindow)
        fresh_curated = fresh_window.query_one(CuratedView)
        assert fresh_curated is not curated, (
            "test setup bug: recompose did not actually replace CuratedView"
        )

        # Half 1 of the fix: hydration. The fresh instance was never told
        # about the install directly -- LLMScreen re-applied the last
        # known progress to it via _hydrate_model_install_progress.
        assert "encoder.onnx" in _progress_text(fresh_curated)

        # Half 2 of the fix: still updating. This tick is delivered by
        # THIS SCREEN's own still-running worker -- exactly what the real
        # download does after a mid-install recompose, since the worker
        # was never owned by the CuratedView instance the recompose tore
        # down in the first place. _deliver_curated posts at
        # self.llm_window, read fresh -- already the NEW window by this
        # point -- so this reaches fresh_curated with no fallback required.
        resume.set()
        await provision_task
        await pilot.pause()
        await pilot.pause()

        assert "decoder.onnx" in _progress_text(fresh_curated)


@pytest.mark.asyncio
async def test_curated_install_progress_after_recompose_still_mirrors_into_installed_view(
    monkeypatch,
):
    """PR #1185 automated review, Important #1 (fix round 2); TASK-1803.

    ``LLMManagementWindow`` (which owns the ``InstallProgressed``/
    ``InstallStatusChanged`` handlers that mirror progress and lifecycle
    into ``InstalledView``, see ``LLM_Management_Window.py``) sits BELOW
    the Screen. Before TASK-1803, ``CuratedView`` posted these messages
    itself and needed a durable fallback for when a screen-level recompose
    orphaned it; an earlier version of that fallback posted straight at
    the Screen, which -- since Textual only ever bubbles a message UP from
    wherever it is posted, never back down -- entered the tree above that
    mirroring node and silently never ran: Curated kept updating (the
    tests above only ever checked Curated), while Installed silently
    stopped receiving ticks/completion.

    TASK-1803 moved the worker to ``LLMScreen`` and made it always post at
    ``self.llm_window`` (``_deliver_curated``), read fresh so it already
    points at whichever ``LLMManagementWindow`` is currently mounted --
    which sits BELOW this screen by construction, so this can no longer
    regress the way the original fallback did. This test is the one the
    original review asked for: it checks the MIRRORING handler's own
    effect on ``InstalledView``, not the curated side, after a real
    recompose.

    Args:
        monkeypatch: pytest's monkeypatch fixture, used to stub the
            network-capable acquisition service so this test never
            performs real I/O; reverted automatically after the test.
    """
    import asyncio
    from unittest.mock import MagicMock

    import tldw_chatbook.Model_Artifacts.acquisition as acquisition_module
    from tldw_chatbook.Model_Artifacts.acquisition import AcquisitionProgress
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    reference = ArtifactRef("parakeet-v2", "immutable-revision", "int8")
    first_progress = AcquisitionProgress(
        "fetch", reference, "encoder.onnx", 100 * 1024 * 1024, 600 * 1024 * 1024
    )
    second_progress = AcquisitionProgress(
        "fetch", reference, "decoder.onnx", 400 * 1024 * 1024, 600 * 1024 * 1024
    )
    resume = asyncio.Event()

    class _FakeAcquisitionService:
        """Stands in for the real, network-capable acquisition service."""

        def __init__(self, _service) -> None:
            """Accept and discard the managed-store service the real
            constructor takes.

            Args:
                _service: The managed-store service (unused by the fake).
            """

        async def provision(self, root, consent, registry, *, sources, progress):
            """Deliver two progress ticks with the recompose in between.

            Args:
                root: The reference this closure is rooted at (unused).
                consent: The granted consent object (unused).
                registry: The curated registry (unused).
                sources: File source map (unused).
                progress: The real ``deliver`` callback ``LLMScreen.
                    _provision_curated`` built.

            Returns:
                A sentinel standing in for the real installed-path result;
                its value is never asserted on.
            """
            progress(first_progress)
            await resume.wait()
            progress(second_progress)
            return object()

    monkeypatch.setattr(
        acquisition_module, "ArtifactAcquisitionService", _FakeAcquisitionService
    )

    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        for _ in range(5):
            await pilot.pause()
        window = screen.query_one(LLMManagementWindow)
        installed = window.query_one(InstalledView)

        screen._model_install_reference = reference
        screen._model_install_service = MagicMock()
        screen._model_install_registry = MagicMock()
        screen._model_install_sources = {}
        fake_report = MagicMock(root=reference)

        provision_task = asyncio.create_task(screen._provision_curated(fake_report))
        await pilot.pause()
        await pilot.pause()

        # Sanity check on the normal (no-recompose) path: the FIRST tick
        # already reaches InstalledView's own mirroring, via the exact
        # bubble chain _deliver_curated's docstring describes
        # (LLMManagementWindow -> LLMScreen, posted at llm_window).
        assert installed._install_progress == first_progress
        assert installed._install_active is True

        screen.refresh(recompose=True)
        for _ in range(5):
            await pilot.pause()

        fresh_window = screen.query_one(LLMManagementWindow)
        fresh_installed = fresh_window.query_one(InstalledView)
        assert fresh_installed is not installed, (
            "test setup bug: recompose did not actually replace InstalledView"
        )

        # The tick under test: delivered by THIS SCREEN's own
        # still-running worker (never torn down by the recompose) --
        # exactly what the real download does after a mid-install
        # recompose. _deliver_curated posts at self.llm_window, read
        # fresh -- already the NEW window by this point -- so it reaches
        # LLMManagementWindow's mirroring handler with no fallback
        # required.
        resume.set()
        await provision_task
        for _ in range(3):
            await pilot.pause()

        assert fresh_installed._install_progress == second_progress, (
            "InstalledView's mirroring handler never observed the post-recompose tick"
        )
        assert fresh_installed._install_active is True


@pytest.mark.asyncio
async def test_deliver_curated_falls_back_to_the_screen_when_llm_window_is_stale_and_closed():
    """TASK-1803 review round 1 (Critical): ``self.llm_window`` is a plain
    attribute, not a live query. ``LabScreen.recompose()`` (``lab_frame.
    py``'s ``recompose()``, which calls the base ``Widget.recompose()``)
    tears down and closes the old ``LLMManagementWindow`` SYNCHRONOUSLY,
    but only the deferred ``_mount_lab_body`` (scheduled via
    ``call_after_refresh``) reassigns ``self.llm_window`` to the fresh
    instance. Between those two points, ``self.llm_window`` still refers
    to the closed widget -- ``post_message`` on a closed target returns
    ``False`` without raising, and the original ``_deliver_curated``
    ignored that return value, silently dropping the message: this
    screen's OWN ``@on(InstallProgressed)``/``@on(InstallStatusChanged)``
    handlers never fired either, so even the Lab status chip and
    ``_model_install_active`` got stuck -- a regression against the
    deleted ``CuratedView._deliver``, which used a live ``try``/``except
    NoMatches`` query specifically so a stale reference could never
    swallow a tick this way.

    Reproduces the exact gap without waiting for a real recompose to race
    it: removes the window exactly as ``recompose()``'s teardown does,
    confirms ``self.llm_window`` is still that same, now-closed instance
    (pinning the reproduction itself, not just the fix), then delivers a
    tick and asserts this screen's own state still updated -- proving the
    fallback-on-``False`` in ``_deliver_curated`` (not a query, and not
    the four-level chain the deleted code used) closes the gap.
    """
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef
    from tldw_chatbook.Widgets.ModelArtifacts import InstallStatusChanged

    reference = ArtifactRef("parakeet-v2", "immutable-revision", "int8")

    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        for _ in range(5):
            await pilot.pause()

        old_window = screen.llm_window
        assert old_window is not None

        # The synchronous half of a screen-level recompose: the old body
        # is removed/closed immediately. Deliberately NOT followed by
        # anything that would reassign screen.llm_window (the deferred
        # _mount_lab_body/build_lab_body pair a real recompose schedules
        # via call_after_refresh) -- that is exactly the gap under test.
        await old_window.remove()
        await pilot.pause()

        assert screen.llm_window is old_window, (
            "test setup bug: something already reassigned llm_window -- "
            "this must still be the stale, closed reference"
        )
        assert old_window._closed is True
        assert (
            old_window.post_message(InstallStatusChanged(reference, active=True))
            is False
        ), (
            "test setup bug: the removed window must already be closed, "
            "i.e. post_message on it must return False, for this to be "
            "the gap _deliver_curated needs to survive"
        )

        screen._deliver_curated(InstallStatusChanged(reference, active=True))
        await pilot.pause()

        assert screen._model_install_active is True, (
            "_deliver_curated silently dropped the message when "
            "self.llm_window was stale and closed -- this screen's own "
            "state never updated"
        )


@pytest.mark.asyncio
async def test_hydration_mirrors_a_tick_delivered_during_the_recompose_gap_into_installed_view():
    """TASK-1803 review round 2 (Important): the fallback in
    ``_deliver_curated`` keeps THIS screen's own state current when a
    tick lands in the teardown -> remount gap (see the test above), but
    posting on ``self`` never reaches ``LLMManagementWindow``'s mirroring
    handlers (``_managed_install_progressed``/``_managed_install_status_
    changed``) -- Textual only ever bubbles a message UP, never back down
    into a sibling/descendant, and the Screen is already above that node.
    Before this fix, ``InstalledView`` would show a stale "not
    installing" state for however long it took the next tick to arrive
    naturally -- this is the same mirroring gap PR #1185 fixed for
    ``CuratedView``, recurring one level deeper.

    This is the distinction the review asked to be pinned: not merely
    that the SCREEN's own state updated (the test above), but that
    ``InstalledView`` itself is brought up to date once the fresh window
    actually mounts.

    Reproduces the exact gap deterministically (removes the window
    exactly as recompose's teardown does -- see the test above --
    delivers a tick while it is closed, and confirms the OLD
    ``InstalledView`` never saw it, proving this is a real reproduction
    and not a no-op), then completes the deferred remount directly
    (``_mount_lab_body``, exactly what ``call_after_refresh`` would
    eventually call for a real recompose) and asserts the FRESH
    ``InstalledView`` reflects the delivered tick once hydration runs.
    """
    from tldw_chatbook.Model_Artifacts.acquisition import AcquisitionProgress
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView
    from tldw_chatbook.Widgets.ModelArtifacts import InstallProgressed

    reference = ArtifactRef("parakeet-v2", "immutable-revision", "int8")
    progress = AcquisitionProgress("fetch", reference, "encoder.onnx", 512, 1024)

    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        for _ in range(5):
            await pilot.pause()

        old_window = screen.llm_window
        assert old_window is not None
        old_installed = old_window.query_one(InstalledView)

        # The synchronous half of a screen-level recompose (see
        # test_deliver_curated_falls_back_to_the_screen_when_llm_window_
        # is_stale_and_closed above for why this deterministically
        # reproduces the gap without racing a real recompose).
        await old_window.remove()
        await pilot.pause()
        assert screen.llm_window is old_window, (
            "test setup bug: something already reassigned llm_window"
        )

        # The tick under test: delivered while the window is stale and
        # closed. Falls back to posting on self (TASK-1803 review round
        # 1), so this screen's own state updates -- but the OLD (about to
        # be discarded) InstalledView must NOT have received it, which is
        # exactly what makes this a genuine reproduction of the gap the
        # mirror needs to survive, not a no-op.
        screen._deliver_curated(InstallProgressed(progress))
        await pilot.pause()

        assert screen._model_install_active is True
        assert screen._model_install_last_progress == progress
        assert old_installed._install_progress is None, (
            "test setup bug: the OLD InstalledView must not have seen "
            "the tick, or this isn't reproducing the gap"
        )

        # Complete the deferred remount directly -- exactly what
        # call_after_refresh(self._mount_lab_body) would eventually call
        # for a real recompose.
        screen._mount_lab_body()
        for _ in range(5):
            await pilot.pause()

        fresh_window = screen.llm_window
        assert fresh_window is not old_window, (
            "test setup bug: _mount_lab_body did not actually replace the window"
        )
        fresh_installed = fresh_window.query_one(InstalledView)
        assert fresh_installed is not old_installed

        assert fresh_installed._install_active is True, (
            "InstalledView was not brought up to date by hydration after "
            "the remount -- it still shows the tick dropped in the "
            "teardown/remount gap as though the install were not active"
        )
        assert fresh_installed._install_progress == progress, (
            "InstalledView's progress was not hydrated from the tick "
            "delivered during the recompose gap"
        )


@pytest.mark.asyncio
async def test_curated_install_progress_renders_exactly_once_per_tick(monkeypatch):
    """TASK-596 delta port, fix round 1 (Review Important #1); TASK-1803.

    ``InstallProgressed`` bubbles by default -- nothing in this codebase
    ever calls ``event.stop()`` on it. Before TASK-1803, ``CuratedView``
    posted this message itself, which used to be handled by its own
    ``_install_progressed`` (rendering the widget), then bubble on,
    unstopped, through ``LLMManagementWindow`` (unrelated to this bug --
    it mirrors into ``InstalledView``, a different widget) up to
    ``LLMScreen``, whose own forwarding rendered the SAME, still-mounted
    ``CuratedView`` a second time via ``apply_progress`` -- three renders
    total for one event with an earlier, since-removed dual-delivery
    fallback added on top. TASK-1803 removed ``CuratedView``'s own
    posting and self-listening entirely: ``LLMScreen`` (via
    ``_deliver_curated``, posting at ``self.llm_window``) is now the ONLY
    originator of this message for a curated install, and
    ``_model_install_progressed`` is the only place that calls
    ``apply_progress``. This counts the actual number of calls, which
    content-only assertions (like the recompose tests above) cannot
    distinguish from two or three.

    TASK-1914 also wraps ``RemoteView.apply_progress`` with the same
    counting shim and asserts it is called ZERO times: ``LLMManagementWindow``
    composes every rail view eagerly, so ``RemoteView`` is mounted (just not
    visible) throughout this curated-only tick, and ``_active_install_view``
    (keyed by ``_model_install_kind``) is the only thing standing between
    "one view renders the tick" and "both views render it" now that a
    single ``_model_install_progressed`` handler serves both flows.

    Args:
        monkeypatch: pytest's monkeypatch fixture, used to wrap
            ``CuratedView.apply_progress``/``RemoteView.apply_progress``
            with call-counting shims; reverted automatically after the
            test.
    """
    from tldw_chatbook.Model_Artifacts.acquisition import AcquisitionProgress
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef
    from tldw_chatbook.UI.Screens.model_curated_view import CuratedView
    from tldw_chatbook.UI.Screens.model_remote_view import RemoteView
    from tldw_chatbook.Widgets.ModelArtifacts import InstallProgressed

    calls: list[AcquisitionProgress] = []
    original_apply_progress = CuratedView.apply_progress

    def counting_apply_progress(self, progress):
        calls.append(progress)
        return original_apply_progress(self, progress)

    monkeypatch.setattr(CuratedView, "apply_progress", counting_apply_progress)

    remote_calls: list[AcquisitionProgress] = []
    original_remote_apply_progress = RemoteView.apply_progress

    def counting_remote_apply_progress(self, progress):
        remote_calls.append(progress)
        return original_remote_apply_progress(self, progress)

    monkeypatch.setattr(RemoteView, "apply_progress", counting_remote_apply_progress)

    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        for _ in range(5):
            await pilot.pause()

        reference = ArtifactRef("parakeet-v2", "immutable-revision", "int8")
        progress = AcquisitionProgress("fetch", reference, "encoder.onnx", 1, 2)

        # TASK-1914: _model_install_kind now decides which view
        # _model_install_progressed forwards to (CuratedView and
        # RemoteView are both mounted at once) -- set explicitly since
        # this test posts directly rather than through
        # _curated_install_requested.
        screen._model_install_kind = "curated"

        # The production entry point for a live tick (TASK-1803):
        # LLMScreen's own worker calls exactly this. Bubbles through
        # LLMManagementWindow (mirrors into InstalledView, untouched by
        # this fix) up to LLMScreen, whose forwarding is the ONLY place
        # that calls apply_progress.
        screen._deliver_curated(InstallProgressed(progress))
        await pilot.pause()
        await pilot.pause()
        await pilot.pause()

    assert calls == [progress]
    assert len(calls) == 1, (
        f"expected exactly one apply_progress call for one progress tick, "
        f"got {len(calls)}"
    )
    assert remote_calls == [], (
        "a curated-install tick must never reach RemoteView.apply_progress "
        "-- _active_install_view routes by _model_install_kind precisely "
        "to prevent this"
    )


@pytest.mark.asyncio
async def test_curated_install_click_reaches_the_shared_consent_modal(monkeypatch):
    """A real Install click -- not a direct call to an internal method --
    posts ``CuratedView.InstallRequested``, which ``LLMScreen`` resolves
    (through a stubbed acquisition service, so this stays network-free)
    into the exact shared ``ModelInstallModal``.

    TASK-1803: this replaces ``test_model_curated_view.py``'s
    ``test_install_click_reaches_the_shared_consent_modal``, which used to
    assert this against ``CuratedView`` directly (it owned the worker that
    resolved the plan and pushed the modal itself). Now that ``LLMScreen``
    owns that worker, the equivalent end-to-end coverage belongs here,
    against a real, running ``LLMScreen``.

    Args:
        monkeypatch: pytest's monkeypatch fixture; stubs
            ``ArtifactAcquisitionService`` so preflight resolves without
            real network I/O, and stubs ``push_screen`` to capture its
            arguments without pushing a real screen.
    """
    from unittest.mock import MagicMock

    import tldw_chatbook.Model_Artifacts.acquisition as acquisition_module
    from tldw_chatbook.UI.Screens.model_curated_view import CuratedView
    from tldw_chatbook.Widgets.ModelArtifacts import ModelInstallModal

    class _FakeAcquisitionService:
        """Stands in for the real, network-capable acquisition service."""

        def __init__(self, _service) -> None:
            """Accept and discard the managed-store service the real
            constructor takes.

            Args:
                _service: The managed-store service (unused by the fake).
            """

        async def preflight(self, ref, _registry, *, sources):
            """Resolve a fake plan rooted at whatever reference was clicked.

            Args:
                ref: The reference LLMScreen asked to preflight.
                _registry: The curated registry (unused).
                sources: File source map (unused).

            Returns:
                A stand-in report whose ``.root`` is ``ref``, so
                ``LLMScreen``'s registry lookup for the modal's label
                resolves against the real curated registry.
            """
            report = MagicMock()
            report.root = ref
            return report

    monkeypatch.setattr(
        acquisition_module, "ArtifactAcquisitionService", _FakeAcquisitionService
    )

    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        monkeypatch.setattr(app, "push_screen", MagicMock())
        for _ in range(5):
            await pilot.pause()

        curated_row = next(
            row for row in _rail_rows(screen) if row.lab_view_key == "curated"
        )
        curated_row.press()
        await pilot.pause()

        window = screen.query_one(LLMManagementWindow)
        curated = window.query_one(CuratedView)

        async def _loaded() -> bool:
            return curated._loaded

        for _ in range(50):
            if await _loaded():
                break
            await pilot.pause()
        assert curated._loaded, "Curated never finished its catalog load"

        assert await _wait_for(
            lambda: any(
                not candidate.disabled
                and candidate in app.screen._compositor.visible_widgets
                for candidate in curated.query(".curated-install").results(Button)
            ),
            pilot,
            attempts=300,
        )
        button = next(
            candidate
            for candidate in curated.query(".curated-install").results(Button)
            if not candidate.disabled
            and candidate in app.screen._compositor.visible_widgets
        )
        await pilot.click(button)

        assert await _wait_for(lambda: app.push_screen.called, pilot, attempts=300), (
            "clicking Install never reached push_screen"
        )

        modal, callback = app.push_screen.call_args[0]
        assert isinstance(modal, ModelInstallModal)
        assert modal.report.root == button.reference
        assert callback == screen._confirm_curated_install
        assert screen._model_install_pending_report is modal.report


@pytest.mark.parametrize("operation", ("preflight", "installation"))
def test_curated_install_failures_log_bounded_error_type(operation, monkeypatch):
    """Worker diagnostics identify the failure type without collaborator data.

    TASK-1803: this used to run directly against ``CuratedView``'s own
    ``_preflight_model``/``_provision_model`` (formerly in
    ``test_model_installed_view.py``); the equivalent worker methods now
    live on ``LLMScreen``.

    Built via ``__new__`` (skipping ``__init__``) with ``app`` patched to
    a ``MagicMock`` at the class level, exactly like the pre-existing
    ``InstalledView``/``CuratedView`` versions of this test -- ``LLMScreen.
    __init__`` reads the real Lab rail-collapse config through
    ``load_rail_layout()``/``get_cli_setting()``, which this test must not
    touch, and a mocked ``app`` lets ``call_from_thread`` be inspected
    directly instead of raising (Textual refuses to run it from the app's
    own thread, which this synchronous test is).

    Args:
        operation: Which worker to exercise -- ``"preflight"`` drives
            ``_run_curated_preflight``, ``"installation"`` drives
            ``_run_curated_provision``.
        monkeypatch: pytest's monkeypatch fixture; patches ``LLMScreen.
            app`` and this module's ``logger``, both reverted afterward.
    """
    from unittest.mock import MagicMock

    from tldw_chatbook.Model_Artifacts.service import ArtifactRef
    from tldw_chatbook.UI.Screens import llm_screen as module

    reference = ArtifactRef("parakeet-v2", "immutable-revision", "int8")
    fake_app = MagicMock()
    fake_logger = MagicMock()
    fake_logger.opt.return_value = fake_logger
    monkeypatch.setattr(module.LLMScreen, "app", property(lambda self: fake_app))
    monkeypatch.setattr(module, "logger", fake_logger)

    screen = module.LLMScreen.__new__(module.LLMScreen)
    screen._model_install_reference = reference
    screen._model_install_service = MagicMock()
    screen._model_install_registry = MagicMock()
    screen._model_install_sources = {}
    screen._model_install_pending_report = None

    if operation == "preflight":

        async def fail_preflight(_reference):
            raise RuntimeError("PRIVATE-WORKER-DETAIL")

        screen._preflight_curated = fail_preflight
        module.LLMScreen._run_curated_preflight.__wrapped__(screen)
    else:
        report = MagicMock()
        report.root = reference
        screen._model_install_pending_report = report

        async def fail_provision(_report):
            raise RuntimeError("PRIVATE-WORKER-DETAIL")

        screen._provision_curated = fail_provision
        module.LLMScreen._run_curated_provision.__wrapped__(screen)

    operation_label = "preflight" if operation == "preflight" else "installation"
    fake_logger.error.assert_called_once_with(
        f"Curated model {operation_label} failed; error_type={{}}",
        "RuntimeError",
    )
    logged = " ".join(str(value) for value in fake_logger.error.call_args.args)
    assert "RuntimeError" in logged
    assert all(
        private not in logged
        for private in (
            reference.artifact_id,
            reference.revision,
            reference.variant,
            "PRIVATE-WORKER-DETAIL",
        )
    )
    if operation == "installation":
        fake_app._ensure_parakeet_source_service.assert_not_called()


def test_curated_preflight_failure_notifies_and_does_not_push_a_modal(monkeypatch):
    """The sibling success path is
    ``test_curated_install_click_reaches_the_shared_consent_modal`` above;
    this is its failure branch, adapted from ``test_model_curated_view.
    py``'s former ``test_preflight_failure_notifies_and_does_not_push_a_
    modal`` now that ``LLMScreen`` -- not ``CuratedView`` -- resolves the
    plan (TASK-1803). Built via ``__new__``, same rationale as the test
    above.

    Args:
        monkeypatch: pytest's monkeypatch fixture; patches ``LLMScreen.
            app`` (a read-only property with no setter, hence the
            class-level patch rather than plain instance assignment).
    """
    from unittest.mock import MagicMock

    from tldw_chatbook.Model_Artifacts.service import ArtifactRef
    from tldw_chatbook.UI.Screens import llm_screen as module

    fake_app = MagicMock()
    monkeypatch.setattr(module.LLMScreen, "app", property(lambda self: fake_app))

    screen = module.LLMScreen.__new__(module.LLMScreen)
    screen.notify = MagicMock()
    view = MagicMock()
    screen._curated_view = MagicMock(return_value=view)
    screen._model_install_worker = MagicMock()
    screen._model_install_reference = ArtifactRef("model-a", "a" * 40, "int8")
    screen._model_install_service = MagicMock()
    screen._model_install_registry = MagicMock()
    screen._model_install_sources = {}
    screen._model_install_pending_report = None

    module.LLMScreen._apply_curated_preflight_result(screen, None, "boom")

    screen.notify.assert_called_once_with("boom", severity="error")
    fake_app.push_screen.assert_not_called()
    assert screen._model_install_worker is None
    assert screen._model_install_reference is None
    assert screen._model_install_service is None
    assert screen._model_install_registry is None
    assert screen._model_install_sources is None
    view.cancel_pending_install.assert_called_once_with("boom")


def test_declining_the_consent_modal_does_not_start_the_install_worker():
    """Adapted from ``test_model_curated_view.py``'s former test of the
    same name -- ``LLMScreen`` now owns the decline path (TASK-1803).
    Built via ``__new__``, same rationale as the tests above.
    """
    from unittest.mock import MagicMock

    from tldw_chatbook.Model_Artifacts.service import ArtifactRef
    from tldw_chatbook.UI.Screens import llm_screen as module

    screen = module.LLMScreen.__new__(module.LLMScreen)
    view = MagicMock()
    screen._curated_view = MagicMock(return_value=view)
    screen._run_curated_provision = MagicMock()
    screen._model_install_worker = None
    screen._model_install_reference = ArtifactRef("model-a", "a" * 40, "int8")
    screen._model_install_service = MagicMock()
    screen._model_install_registry = MagicMock()
    screen._model_install_sources = {}
    screen._model_install_pending_report = object()

    module.LLMScreen._confirm_curated_install(screen, False)

    screen._run_curated_provision.assert_not_called()
    assert screen._model_install_reference is None
    assert screen._model_install_pending_report is None
    view.cancel_pending_install.assert_called_once_with()


def test_declining_insufficient_space_persists_exact_required_and_free_bytes():
    """A disabled consent plan leaves actionable byte-exact recovery behind."""
    from tldw_chatbook.Model_Artifacts.acquisition import PreflightReport
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef
    from tldw_chatbook.UI.Screens import llm_screen as module

    screen = module.LLMScreen.__new__(module.LLMScreen)
    view = MagicMock()
    screen._curated_view = MagicMock(return_value=view)
    screen._run_curated_provision = MagicMock()
    screen._model_install_worker = None
    screen._model_install_reference = ArtifactRef("model-a", "a" * 40, "int8")
    screen._model_install_service = MagicMock()
    screen._model_install_registry = MagicMock()
    screen._model_install_sources = {}
    screen._model_install_pending_report = PreflightReport(
        root=screen._model_install_reference,
        closure_fingerprint="f" * 64,
        entries=(),
        download_bytes=8_000_000,
        already_staged_bytes=0,
        staging_overhead_bytes=388_608,
        retained_bytes=0,
        destination=MagicMock(),
        free_bytes=2_097_152,
        required_bytes=8_388_608,
        sufficient_space=False,
        gating_errors=(),
    )

    module.LLMScreen._confirm_curated_install(screen, False)

    view.cancel_pending_install.assert_called_once_with(
        "Insufficient space — 8,388,608 bytes required; 2,097,152 bytes free. "
        "Free space, then select Retry install."
    )


@pytest.mark.asyncio
async def test_models_lab_insufficient_space_cancel_is_inline_and_never_provisions(
    tmp_path,
    monkeypatch,
):
    """The real 80x24 Lab modal returns to a focused byte-exact Retry action."""
    from tldw_chatbook.Model_Artifacts import acquisition as acquisition_module
    from tldw_chatbook.Model_Artifacts.acquisition import PreflightReport
    from tldw_chatbook.UI.Screens.model_curated_view import CuratedView
    from tldw_chatbook.Widgets.ModelArtifacts import ModelInstallModal

    required = 8_388_608
    free = 2_097_152
    provision_calls = []

    class Acquisition:
        def __init__(self, _service):
            pass

        async def preflight(self, reference, _registry, *, sources):
            assert sources
            return PreflightReport(
                root=reference,
                closure_fingerprint="f" * 64,
                entries=(),
                download_bytes=8_000_000,
                already_staged_bytes=0,
                staging_overhead_bytes=388_608,
                retained_bytes=0,
                destination=tmp_path / "managed",
                free_bytes=free,
                required_bytes=required,
                sufficient_space=False,
                gating_errors=(),
            )

        async def provision(self, *_args, **_kwargs):
            provision_calls.append(True)
            raise AssertionError("insufficient-space admission must not provision")

    monkeypatch.setattr(acquisition_module, "ArtifactAcquisitionService", Acquisition)
    app = _app()
    async with app.run_test(size=(80, 24)) as pilot:
        screen = await _models_screen(app)
        assert await _wait_for(
            lambda: bool(screen.query("#curated-models-view")), pilot
        )
        window = screen.query_one(LLMManagementWindow)
        curated = window.query_one("#curated-models-view", CuratedView)
        curated.set_consumer_filter("audio_cpp")
        window.active_view = "curated"
        assert await _wait_for(
            lambda: curated._loaded and bool(curated.query(".curated-install")),
            pilot,
        )
        install = curated.query_one(".curated-install", Button)
        install.press()
        assert await _wait_for(
            lambda: (
                isinstance(app.screen, ModelInstallModal)
                and bool(app.screen.query("#model-install-confirm"))
            ),
            pilot,
            attempts=1500,
        )
        modal = app.screen
        assert modal.query_one("#model-install-confirm", Button).disabled is True
        modal.query_one("#model-install-cancel", Button).press()
        assert await _wait_for(lambda: app.screen is screen, pilot)
        expected = (
            "Insufficient space — 8,388,608 bytes required; 2,097,152 bytes free. "
            "Free space, then select Retry install."
        )
        assert await _wait_for(
            lambda: (
                expected
                in "\n".join(str(item.renderable) for item in curated.query(Static))
            ),
            pilot,
        )
        retry = curated.query_one(".curated-install", Button)
        assert str(retry.label) == "Retry install…"
        assert await _wait_for(lambda: retry.has_focus, pilot)
        assert retry in app.screen._compositor.visible_widgets
        assert retry.region.right <= window.query_one("#llm-view-curated").region.right
        assert (
            retry.region.bottom <= window.query_one("#llm-view-curated").region.bottom
        )

    assert provision_calls == []


@pytest.mark.parametrize(
    ("detail", "retryable", "code_name", "expected"),
    (
        (
            "transport error fetching private/source/path",
            True,
            "SOURCE_UNAVAILABLE",
            "Pinned source unavailable — the app may be offline.",
        ),
        (
            "preverify checksum mismatch at private/source/path",
            False,
            "VERIFICATION_FAILED",
            "Package verification failed (size or SHA-256). No package was promoted.",
        ),
        (
            "PRIVATE local state detail /Users/example/model",
            True,
            "LOCAL_STATE",
            "Package install could not access local state. Select Retry install.",
        ),
        (
            "PRIVATE conflict at /Users/example/model",
            False,
            "LOCAL_STATE",
            "Package install found conflicting or invalid local state. Review or Repair the local model store before installing again.",
        ),
        (
            "PRIVATE blocked source https://secret.example/model",
            False,
            "SOURCE_BLOCKED",
            "Package install is blocked by local source-access policy. Review network policy, then select Retry install.",
        ),
    ),
)
def test_audio_cpp_transfer_failures_map_to_bounded_inline_recovery(
    detail, retryable, code_name, expected
):
    """Typed transfer failures select recovery without leaking collaborator text."""
    from tldw_chatbook.Model_Artifacts import acquisition as acquisition_module
    from tldw_chatbook.UI.Screens.model_browser_state import install_failure_message

    code_type = getattr(acquisition_module, "TransferFailureCode", None)
    assert code_type is not None
    message = install_failure_message(
        acquisition_module.TransferError(
            detail,
            retryable=retryable,
            code=getattr(code_type, code_name),
        ),
        model_label="audio.cpp package",
    )

    assert expected in message
    assert "private/source/path" not in message
    assert "PRIVATE" not in message
    assert "/Users" not in message
    assert "secret.example" not in message
    if code_name in {"LOCAL_STATE", "SOURCE_BLOCKED"}:
        assert "download" not in message.casefold()
    if code_name == "LOCAL_STATE" and not retryable:
        assert "Retry" not in message


def test_ambiguous_transfer_text_cannot_promote_a_typed_recovery_claim() -> None:
    """Unknown code stays generic even when private text contains size/offline words."""
    from tldw_chatbook.Model_Artifacts import acquisition as acquisition_module
    from tldw_chatbook.UI.Screens.model_browser_state import install_failure_message

    code_type = getattr(acquisition_module, "TransferFailureCode", None)
    assert code_type is not None
    marker = "PRIVATE source size endpoint is offline /Users/example/model"
    message = install_failure_message(
        acquisition_module.TransferError(
            marker,
            retryable=True,
            code=code_type.UNKNOWN,
        ),
        model_label="audio.cpp package",
    )

    assert message == "The download was interrupted. Retry Install to resume."
    assert "Pinned source unavailable" not in message
    assert "verification failed" not in message.casefold()
    assert "not promoted" not in message
    assert "PRIVATE" not in message
    assert "/Users" not in message


def test_curated_install_requested_refuses_a_second_concurrent_install():
    """TASK-1803 review round 1 (Important, gap #2): untested until now.

    ``_curated_install_requested``'s concurrency guard (``_install_in_
    progress()``, mirroring ``LibraryScreen.handle_parakeet_v2_install_
    requested``'s own worker-in-flight guard, generalized in TASK-1914
    fix round 2 to span the whole install lifecycle rather than only
    "a worker happens to be running") must refuse a second request while
    one install is still in progress -- not start a competing preflight
    worker, not touch the running install's own retained reference/
    service/registry/sources, and not touch its worker handle -- and must
    release only the freshly clicking ``CuratedView``'s own in-flight
    indicator (see the method's own docstring for why that view,
    specifically, is the one instance a screen-level recompose can hand a
    second chance to click Install while the original install is still in
    progress elsewhere).
    """
    from unittest.mock import MagicMock

    from tldw_chatbook.Model_Artifacts.service import ArtifactRef
    from tldw_chatbook.UI.Screens import llm_screen as module
    from tldw_chatbook.UI.Screens.model_curated_view import CuratedView

    running_worker = MagicMock()
    running_worker.is_finished = False
    running_reference = ArtifactRef("model-a", "a" * 40, "int8")
    running_service = MagicMock()
    running_registry = MagicMock()
    running_sources = {}

    screen = module.LLMScreen.__new__(module.LLMScreen)
    screen.notify = MagicMock()
    screen._run_curated_preflight = MagicMock()
    view = MagicMock()
    screen._curated_view = MagicMock(return_value=view)
    screen._model_install_kind = "curated"
    screen._model_install_worker = running_worker
    screen._model_install_reference = running_reference
    screen._model_install_service = running_service
    screen._model_install_registry = running_registry
    screen._model_install_sources = running_sources

    event = CuratedView.InstallRequested(
        ArtifactRef("model-b", "b" * 40, "int8"),
        service=MagicMock(),
        registry=MagicMock(),
        sources={},
    )
    event.stop = MagicMock()

    module.LLMScreen._curated_install_requested(screen, event)

    event.stop.assert_called_once_with()
    screen.notify.assert_called_once_with(
        "A curated model install is already running.",
        severity="information",
    )
    view.cancel_pending_install.assert_called_once_with()
    screen._run_curated_preflight.assert_not_called()
    # The running install's own retained state must survive untouched.
    assert screen._model_install_reference is running_reference
    assert screen._model_install_service is running_service
    assert screen._model_install_registry is running_registry
    assert screen._model_install_sources is running_sources
    assert screen._model_install_worker is running_worker


@pytest.mark.parametrize(
    ("reference", "service", "registry", "sources"),
    (
        (None, "service", "registry", {}),
        ("not-a-ref", "service", "registry", {}),
        (object(), "service", "registry", {}),
    ),
)
def test_curated_install_requested_refuses_an_invalid_payload_without_starting_a_worker(
    reference, service, registry, sources
):
    """TASK-1803 review round 2 (Critical, Finding 1): an invalid request
    must never be stored or acted on.

    ``_run_curated_preflight`` used to assume ``self._model_install_
    reference`` was always a valid ``ArtifactRef``: a missing or malformed
    reference reached ``reference.artifact_id`` inside that worker's own
    exception handler, raising a SECOND, unhandled exception that
    pre-empted ``_apply_curated_preflight_result`` entirely and stranded
    the retained install state with no path back to idle. The primary
    defense is here, before anything is ever stored: an invalid
    ``reference`` (parametrized: ``None``, a plain string, and an
    arbitrary object -- none are ``ArtifactRef``) notifies, releases the
    clicking view's own indicator via ``cancel_pending_install()``, clears
    every retained field via ``_clear_curated_install_state()``, and never
    starts a preflight worker.

    Args:
        reference: The (invalid) reference value under test.
        service: A stand-in for ``event.service`` (a valid, non-``None``
            placeholder here; only ``reference`` is exercised as invalid
            in this parametrization).
        registry: A stand-in for ``event.registry``, same rationale.
        sources: A stand-in for ``event.sources``, same rationale.
    """
    from tldw_chatbook.UI.Screens import llm_screen as module
    from tldw_chatbook.UI.Screens.model_curated_view import CuratedView

    screen = module.LLMScreen.__new__(module.LLMScreen)
    screen.notify = MagicMock()
    screen._run_curated_preflight = MagicMock()
    view = MagicMock()
    screen._curated_view = MagicMock(return_value=view)
    screen._model_install_kind = None
    screen._model_install_worker = None
    screen._model_install_reference = None
    screen._model_install_service = None
    screen._model_install_registry = None
    screen._model_install_sources = None
    screen._model_install_pending_report = None

    event = CuratedView.InstallRequested(
        reference,
        service=service,
        registry=registry,
        sources=sources,
    )
    event.stop = MagicMock()

    module.LLMScreen._curated_install_requested(screen, event)

    event.stop.assert_called_once_with()
    screen._run_curated_preflight.assert_not_called()
    screen.notify.assert_called_once_with(
        "Could not start the model install: invalid request.",
        severity="error",
    )
    view.cancel_pending_install.assert_called_once_with()
    assert screen._model_install_worker is None
    assert screen._model_install_reference is None
    assert screen._model_install_service is None
    assert screen._model_install_registry is None
    assert screen._model_install_sources is None
    assert screen._model_install_pending_report is None


@pytest.mark.parametrize(
    ("missing_field",),
    (("service",), ("registry",), ("sources",)),
)
def test_curated_install_requested_refuses_when_service_registry_or_sources_is_none(
    missing_field,
):
    """TASK-1803 review round 2 (Critical, Finding 1): the same validation
    covers ``event.service``/``event.registry``/``event.sources`` being
    ``None``, not only a malformed ``reference``.

    Args:
        missing_field: Which of the three payload fields to set to
            ``None`` for this parametrization; the other two stay valid.
    """
    from unittest.mock import MagicMock

    from tldw_chatbook.Model_Artifacts.service import ArtifactRef
    from tldw_chatbook.UI.Screens import llm_screen as module
    from tldw_chatbook.UI.Screens.model_curated_view import CuratedView

    fields = {"service": "service", "registry": "registry", "sources": {}}
    fields[missing_field] = None

    screen = module.LLMScreen.__new__(module.LLMScreen)
    screen.notify = MagicMock()
    screen._run_curated_preflight = MagicMock()
    view = MagicMock()
    screen._curated_view = MagicMock(return_value=view)
    screen._model_install_kind = None
    screen._model_install_worker = None
    screen._model_install_reference = None
    screen._model_install_service = None
    screen._model_install_registry = None
    screen._model_install_sources = None

    event = CuratedView.InstallRequested(
        ArtifactRef("model-a", "a" * 40, "int8"),
        **fields,
    )
    event.stop = MagicMock()

    module.LLMScreen._curated_install_requested(screen, event)

    screen._run_curated_preflight.assert_not_called()
    screen.notify.assert_called_once_with(
        "Could not start the model install: invalid request.",
        severity="error",
    )
    assert screen._model_install_reference is None


def test_run_curated_preflight_schedules_apply_result_when_reference_is_none(
    monkeypatch,
):
    """TASK-1803 review round 2 (Critical, Finding 1): defense-in-depth.

    ``_curated_install_requested`` already refuses to store an invalid
    reference, so this should be unreachable in practice -- but
    ``_run_curated_preflight`` must never trust that assumption blindly.
    If ``self._model_install_reference`` is somehow ``None`` when this
    worker runs, it must still schedule
    ``_apply_curated_preflight_result(None, <message>)`` directly rather
    than reaching the ``try`` block and risking an ``AttributeError`` on
    ``None.artifact_id`` that would pre-empt that call entirely.

    Args:
        monkeypatch: pytest's monkeypatch fixture; patches ``LLMScreen.
            app`` (a read-only property with no setter, hence the
            class-level patch rather than plain instance assignment).
    """
    from unittest.mock import MagicMock

    from tldw_chatbook.UI.Screens import llm_screen as module

    fake_app = MagicMock()
    monkeypatch.setattr(module.LLMScreen, "app", property(lambda self: fake_app))

    screen = module.LLMScreen.__new__(module.LLMScreen)
    screen._model_install_reference = None

    module.LLMScreen._run_curated_preflight.__wrapped__(screen)

    fake_app.call_from_thread.assert_called_once()
    args = fake_app.call_from_thread.call_args[0]
    assert args[0] == screen._apply_curated_preflight_result
    assert args[1] is None
    assert isinstance(args[2], str) and args[2]


def test_run_curated_preflight_except_clause_survives_a_malformed_reference(
    monkeypatch,
):
    """TASK-1803 review round 2 (Critical, Finding 1): the except clause
    itself must never raise a second exception.

    Forces ``_preflight_curated`` to fail with a malformed (non-
    ``ArtifactRef``) ``self._model_install_reference`` already in place --
    exactly the state a bug elsewhere could otherwise leave behind -- and
    asserts the exception handler still schedules
    ``_apply_curated_preflight_result(None, ...)`` exactly once without
    inspecting malformed artifact fields or logging collaborator details.

    Args:
        monkeypatch: pytest's monkeypatch fixture; patches ``LLMScreen.
            app`` (a read-only property with no setter) and this module's
            ``logger``.
    """
    from unittest.mock import MagicMock

    from tldw_chatbook.UI.Screens import llm_screen as module

    fake_app = MagicMock()
    fake_logger = MagicMock()
    fake_logger.opt.return_value = fake_logger
    monkeypatch.setattr(module.LLMScreen, "app", property(lambda self: fake_app))
    monkeypatch.setattr(module, "logger", fake_logger)

    screen = module.LLMScreen.__new__(module.LLMScreen)
    screen._model_install_reference = object()  # malformed: not an ArtifactRef

    async def fail_preflight(_reference):
        raise RuntimeError("PRIVATE-WORKER-DETAIL")

    screen._preflight_curated = fail_preflight

    # Must not itself raise (that is exactly the regression under test).
    module.LLMScreen._run_curated_preflight.__wrapped__(screen)

    fake_logger.error.assert_called_once_with(
        "Curated model preflight failed; error_type={}",
        "RuntimeError",
    )
    logged = " ".join(str(value) for value in fake_logger.error.call_args.args)
    assert "RuntimeError" in logged
    assert "unknown" not in logged
    assert "PRIVATE-WORKER-DETAIL" not in logged

    fake_app.call_from_thread.assert_called_once_with(
        screen._apply_curated_preflight_result,
        None,
        fake_app.call_from_thread.call_args[0][2],
    )
    assert fake_app.call_from_thread.call_args[0][1] is None


def test_failed_curated_provision_notifies_mirrors_and_resets_state(monkeypatch):
    """TASK-1803 review round 1 (Important, gap #1): untested until now.

    ``_apply_curated_provision_result`` is the sole path that ends a
    curated install, success or failure: it must notify the exact
    outcome, deliver the terminal ``InstallStatusChanged(active=False,
    ...)`` (so the Lab chip and ``InstalledView``'s mirror both learn the
    install finished -- losing this message is what leaves the chip stuck
    on "downloading" forever), reset every bit of retained install state
    so a later install starts clean, and tell the visible ``CuratedView``
    (if any) to reload. The deleted ``CuratedView._apply_provision_result``
    had a direct test (``test_curated_provision_completion_tolerates_
    recompose_gap``); its stated replacement
    (``test_finish_install_clears_the_indicator_and_reloads_despite_a_
    missing_progress_widget`` in ``test_model_curated_view.py``) only
    covers ``CuratedView.finish_install()``'s render half, not this
    method's notify/deliver/reset half.

    Args:
        error: The error string ``_run_curated_provision`` would pass on
            failure, or ``None`` on success.
        expected_severity: The ``notify()`` severity this error value
            maps to.
        expected_message: The ``notify()`` message this error value maps
            to.
        monkeypatch: pytest's monkeypatch fixture; patches ``LLMScreen.
            app`` (a read-only property with no setter).
    """
    from unittest.mock import MagicMock

    from tldw_chatbook.Local_Ingestion.parakeet_v2_artifact import (
        parakeet_reference,
    )
    from tldw_chatbook.Local_Ingestion.stt_batch_routing import PARAKEET_V2_MODEL
    from tldw_chatbook.UI.Screens import llm_screen as module

    fake_app = MagicMock()
    source_service = MagicMock()
    fake_app._ensure_parakeet_source_service.return_value = source_service
    monkeypatch.setattr(module.LLMScreen, "app", property(lambda self: fake_app))

    reference = parakeet_reference(PARAKEET_V2_MODEL, "int8")
    screen = module.LLMScreen.__new__(module.LLMScreen)
    screen.notify = MagicMock()
    screen._deliver_curated = MagicMock()
    view = MagicMock()
    screen._curated_view = MagicMock(return_value=view)
    screen._model_install_worker = MagicMock()
    screen._model_install_reference = reference
    screen._model_install_service = MagicMock()
    screen._model_install_registry = MagicMock()
    screen._model_install_sources = {}
    screen._model_install_pending_report = object()

    module.LLMScreen._apply_curated_provision_result(screen, "boom")

    screen.notify.assert_called_once_with("boom", severity="error")
    assert screen._model_install_worker is None
    assert screen._model_install_pending_report is None
    assert screen._model_install_reference is None
    assert screen._model_install_service is None
    assert screen._model_install_registry is None
    assert screen._model_install_sources is None

    screen._deliver_curated.assert_called_once()
    delivered = screen._deliver_curated.call_args[0][0]
    assert isinstance(delivered, module.InstallStatusChanged)
    assert delivered.reference == reference
    assert delivered.active is False
    assert delivered.succeeded is False

    view.finish_install.assert_called_once_with("boom")
    source_service.prefer_managed.assert_not_called()


@pytest.mark.parametrize("preference_write_fails", (False, True))
@pytest.mark.asyncio
async def test_successful_curated_install_persists_managed_preference_off_loop(
    preference_write_fails: bool,
):
    import threading

    from tldw_chatbook.Local_Ingestion.parakeet_v2_artifact import (
        parakeet_reference,
    )
    from tldw_chatbook.Local_Ingestion.stt_batch_routing import PARAKEET_V2_MODEL

    service = _FakeExternalSourceService()
    if preference_write_fails:

        def fail_preference_write(_key, *, cancelled=lambda: False):
            service.prefer_threads.append(threading.get_ident())
            raise RuntimeError("preference write failed")

        service.prefer_managed = fail_preference_write
    app = _app()
    app._parakeet_source_service = service
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        assert await _wait_for(
            lambda: bool(screen.query("#external-models-view")), pilot
        )
        screen.notify = MagicMock()
        screen._deliver_curated = MagicMock()
        reference = parakeet_reference(PARAKEET_V2_MODEL, "int8")
        screen._model_install_kind = "curated"
        screen._model_install_reference = reference
        screen._model_install_service = MagicMock()
        screen._model_install_registry = MagicMock()
        screen._model_install_sources = {}
        screen._model_install_pending_report = object()

        async def provision_succeeds(_report):
            return reference

        screen._provision_curated = provision_succeeds

        screen._model_install_worker = screen._run_curated_provision()

        assert await _wait_for(lambda: screen._model_install_kind is None, pilot)
        assert service.prefer_threads[0] != threading.get_ident()
        assert screen._model_install_pending_report is None
        delivered = screen._deliver_curated.call_args.args[0]
        assert delivered.succeeded is True
        assert delivered.active is False
        expected_message = (
            "Model installed, but the managed source preference could not be saved."
            if preference_write_fails
            else "Model installed and activated."
        )
        screen.notify.assert_called_once_with(
            expected_message,
            severity="error" if preference_write_fails else "information",
        )
        assert bool(service.preferred) is not preference_write_fails


@pytest.mark.asyncio
async def test_successful_curated_preference_survives_immediate_screen_unmount():
    import threading

    from textual.screen import Screen

    from tldw_chatbook.Local_Ingestion.parakeet_v2_artifact import (
        parakeet_reference,
    )
    from tldw_chatbook.Local_Ingestion.stt_batch_routing import PARAKEET_V2_MODEL
    from tldw_chatbook.STT.parakeet_sources import ParakeetSourceKey

    service = _FakeExternalSourceService()
    app = _app()
    app._parakeet_source_service = service
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        assert await _wait_for(
            lambda: bool(screen.query("#external-models-view")), pilot
        )
        reference = parakeet_reference(PARAKEET_V2_MODEL, "int8")
        screen._model_install_kind = "curated"
        screen._model_install_reference = reference
        screen._model_install_service = MagicMock()
        screen._model_install_registry = MagicMock()
        screen._model_install_sources = {}
        screen._model_install_pending_report = object()
        screen.notify = MagicMock()
        screen._deliver_curated = MagicMock()
        terminal_delivery_started = threading.Event()
        release_terminal_delivery = threading.Event()
        real_call_from_thread = app.call_from_thread

        async def provision_succeeds(_report):
            return reference

        screen._provision_curated = provision_succeeds

        def block_terminal_delivery(callback, *args, **kwargs):
            callback_function = getattr(callback, "__func__", None)
            if callback_function in {
                LLMScreen._apply_curated_provision_result,
                LLMScreen._apply_curated_preference_result,
            }:
                terminal_delivery_started.set()
                assert release_terminal_delivery.wait(3)
            return real_call_from_thread(callback, *args, **kwargs)

        app.call_from_thread = block_terminal_delivery

        screen._model_install_worker = screen._run_curated_provision()
        assert await _wait_for(terminal_delivery_started.is_set, pilot)
        await app.switch_screen(Screen())
        await pilot.pause()
        assert screen not in app.screen_stack
        assert screen.is_attached is False
        release_terminal_delivery.set()

        assert await _wait_for(lambda: bool(service.preferred), pilot)
        assert service.preferred == [ParakeetSourceKey.V2_INT8]
        assert service.prefer_threads[0] != threading.get_ident()
        await pilot.pause()
        screen.notify.assert_not_called()
        screen._deliver_curated.assert_not_called()


@pytest.mark.asyncio
async def test_the_inspector_rows_refresh_alongside_the_status_chip():
    """Regression test: `refresh_lab_status` used to update only the chip.

    Live evidence: the chip read "Servers: 1 running" while the inspector
    row beside it still read "stopped" -- `refresh_lab_status` mutated only
    `#lab-status-chip-*`, never the per-server rows `compose_lab_inspector`
    composed. Both must agree after the same refresh, on the same poll.
    """
    app = _app()
    app.llamacpp_server_process = None
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        await pilot.pause()
        await pilot.pause()
        chip = screen.query_one("#lab-status-chip-servers", Static)
        row = screen.query_one("#lab-inspector-server-llama-cpp", Static)
        assert "Servers: none running" in str(chip.renderable)
        assert "stopped" in str(row.renderable)

        class _Alive:
            def poll(self):
                return None

        app.llamacpp_server_process = _Alive()
        screen.refresh_lab_status()
        await pilot.pause()

        assert "Servers: 1 running" in str(chip.renderable)
        assert "running" in str(row.renderable)
        assert "stopped" not in str(row.renderable)


@pytest.mark.asyncio
async def test_the_initial_view_is_marked_active_on_arrival_with_no_press():
    """Regression test for the blank-body-on-arrival bug.

    ``LLMManagementWindow`` now mounts from ``call_after_refresh`` (Models'
    body costs 488-787 ms to compose), which changed *when* the window
    mounts relative to ``active_view``'s reactive default-value watcher.
    ``_initialize_view`` used to just assign
    ``self.active_view = "llama-cpp"`` -- the reactive's own default -- and
    Textual skips a watcher when a value is set to one already equal to the
    current value, so no view was ever marked ``-active`` and the body
    rendered blank.

    This must assert the ARRIVAL state without pressing any rail row: a
    press assigns a genuinely new value, which does fire the watcher and
    would mask the bug entirely (as every other test in this file does,
    intentionally or not).
    """
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        await pilot.pause()
        await pilot.pause()
        window = screen.query_one(LLMManagementWindow)

        active_views = [v for v in window.query(".llm-view") if "-active" in v.classes]
        assert len(active_views) == 1, "exactly one .llm-view must carry -active"
        assert active_views[0].id == "llm-view-llama-cpp"


@pytest.mark.asyncio
async def test_surviving_model_rails_trigger_no_unprompted_http_or_search(monkeypatch):
    """Traversing the mounted Models rail stays idle until an explicit action."""
    import httpx

    from tldw_chatbook.Model_Artifacts.remote_huggingface import (
        HuggingFaceRemoteAdapter,
    )
    from tldw_chatbook.UI.Screens.model_remote_view import RemoteView

    http_calls: list[tuple[str, str]] = []
    search_calls: list[str] = []
    original_search = HuggingFaceRemoteAdapter.search

    async def counted_send(self, request, *args, **kwargs):
        http_calls.append((request.method, str(request.url)))
        return httpx.Response(200, json=[], request=request)

    async def counted_search(self, query, *, token=None):
        search_calls.append(query)
        return await original_search(self, query, token=token)

    monkeypatch.setattr(httpx.AsyncClient, "send", counted_send)
    monkeypatch.setattr(HuggingFaceRemoteAdapter, "search", counted_search)

    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        for _ in range(6):
            await pilot.pause()

        for row in _rail_rows(screen):
            row.press()
            await pilot.pause()

        assert http_calls == [], f"rail traversal issued HTTP: {http_calls}"
        assert search_calls == [], f"rail traversal searched Remote: {search_calls}"

        window = screen.query_one(LLMManagementWindow)
        remote = window.query_one("#remote-models-view", RemoteView)
        remote.query_one("#remote-model-query", Input).value = "quantized model"
        await pilot.click("#remote-model-search")
        for _ in range(50):
            if http_calls:
                break
            await pilot.pause()

        expected = (
            "GET",
            "https://huggingface.co/api/models?search=quantized+model&limit=50",
        )
        assert search_calls == ["quantized model"]
        assert http_calls == [expected]


# ---------------------------------------------------------------------------
# TASK-1914: RemoteView's preflight/provision workers, moved here from
# model_remote_view.py mirroring TASK-1803's move of CuratedView's. Small
# local helpers below build a one-item remote catalog/report without any
# real network or filesystem I/O, reused across the tests that follow.
# ---------------------------------------------------------------------------


def _resolved_remote_model(
    repository: str = "owner/repository",
    *,
    license_id: str = "apache-2.0",
    filename: str = "model-q4.gguf",
    total_bytes: int = 1024,
):
    from tldw_chatbook.Model_Artifacts.remote_huggingface import (
        RemoteGGUFCandidate,
        RemoteGGUFFile,
        ResolvedRemoteModel,
    )

    commit = "a" * 40
    digest = "b" * 64
    candidate = RemoteGGUFCandidate(
        label=f"{repository} · {filename}",
        files=(RemoteGGUFFile(filename, total_bytes, digest),),
        total_bytes=total_bytes,
    )
    return ResolvedRemoteModel(
        repository=repository,
        commit=commit,
        license_id=license_id,
        review_url=f"https://huggingface.co/{repository}/tree/{commit}",
        candidates=(candidate,),
        total_candidate_count=1,
        warnings=(),
    )


def _remote_catalog(*, license_id: str = "apache-2.0"):
    from tldw_chatbook.Model_Artifacts.remote_huggingface import build_remote_catalog

    resolved = _resolved_remote_model(license_id=license_id)
    return build_remote_catalog(resolved, resolved.candidates[0])


def _remote_report_for(catalog, destination):
    from tldw_chatbook.Model_Artifacts.acquisition import (
        ArtifactPreflightEntry,
        PreflightReport,
    )
    from tldw_chatbook.Model_Artifacts.service import ProvenanceClass

    descriptor = catalog.artifact
    return PreflightReport(
        root=descriptor.reference,
        closure_fingerprint="f" * 64,
        entries=(
            ArtifactPreflightEntry(
                ref=descriptor.reference,
                source_url=descriptor.source_url,
                repository=descriptor.upstream_repository,
                revision=descriptor.upstream_revision,
                license_id=descriptor.license_id,
                license_url=descriptor.license_url,
                precision=descriptor.precision,
                total_bytes=descriptor.expected_installed_bytes,
                file_count=len(descriptor.files),
                already_installed=False,
                provenance=(ProvenanceClass.LOCAL_INTEGRITY_RECORDED,),
            ),
        ),
        download_bytes=descriptor.expected_installed_bytes,
        already_staged_bytes=0,
        staging_overhead_bytes=128,
        retained_bytes=0,
        destination=destination,
        free_bytes=4096,
        required_bytes=descriptor.expected_installed_bytes + 128,
        sufficient_space=True,
        gating_errors=(),
    )


@pytest.mark.asyncio
async def test_remote_install_progress_survives_a_screen_level_recompose(monkeypatch):
    """TASK-1914 mirror of ``test_curated_install_progress_survives_a_screen_
    level_recompose``, for the remote flow: exercises the real ``LLMScreen.
    _provision_remote`` code path against a stubbed
    ``ArtifactAcquisitionService`` so this test controls exactly when a
    second progress tick fires relative to a real screen-level recompose,
    then asserts both halves of the fix -- the freshly (re)mounted
    ``RemoteView`` is hydrated with the last known progress, and a tick
    emitted AFTER the recompose (delivered through this screen's own
    still-running worker) still reaches the fresh view.
    """
    import asyncio
    from unittest.mock import MagicMock

    import tldw_chatbook.Model_Artifacts.acquisition as acquisition_module
    from tldw_chatbook.Model_Artifacts.acquisition import AcquisitionProgress
    from tldw_chatbook.UI.Screens.model_remote_view import RemoteView
    from tldw_chatbook.Widgets.ModelArtifacts.install_progress import (
        ModelInstallProgress,
    )

    resolved = _resolved_remote_model()
    catalog = _remote_catalog()
    candidate = resolved.candidates[0]
    reference = catalog.artifact.reference
    first_progress = AcquisitionProgress(
        "fetch", reference, "model-part-1.gguf", 100, 1024
    )
    second_progress = AcquisitionProgress(
        "fetch", reference, "model-part-2.gguf", 400, 1024
    )
    resume = asyncio.Event()

    class _FakeAcquisitionService:
        """Stands in for the real, network-capable acquisition service."""

        def __init__(self, _service, *, credential_resolver=None) -> None:
            """Accept and discard the managed-store service/resolver the
            real constructor takes.

            Args:
                _service: The managed-store service (unused by the fake).
                credential_resolver: The credential resolver (unused).
            """

        async def provision(
            self, root, consent, catalog, *, sources, progress, activate
        ):
            """Deliver two progress ticks with the recompose in between.

            Args:
                root: The reference this closure is rooted at (unused).
                consent: The granted consent object (unused).
                catalog: The remote catalog (unused).
                sources: File source map (unused).
                progress: The real ``deliver`` callback ``LLMScreen.
                    _provision_remote`` built.
                activate: Must be ``False`` for the remote flow (asserted
                    by the sibling ``test_provision_remote_never_activates``
                    test; not re-asserted here).

            Returns:
                A sentinel standing in for the real installed-path result.
            """
            progress(first_progress)
            await resume.wait()
            progress(second_progress)
            return object()

    monkeypatch.setattr(
        acquisition_module, "ArtifactAcquisitionService", _FakeAcquisitionService
    )

    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app, populate_all=False)
        for _ in range(5):
            await pilot.pause()
        window = screen.query_one(LLMManagementWindow)
        window.active_view = "remote"
        assert await _wait_for(lambda: bool(window.query(RemoteView)), pilot)
        remote = window.query_one(RemoteView)

        # Match the selected-model context the real RemoteView posts with
        # InstallRequested before LLMScreen takes ownership of the worker.
        remote._resolved = resolved
        remote._selected_repository = resolved.repository
        remote._selected_candidate = candidate
        remote._operation_reference = reference
        remote._refresh_with_status("Preparing the managed install plan…")
        await pilot.pause()

        # State lives on the SCREEN now (TASK-1914), not on the RemoteView
        # instance -- it must survive the instance being torn down below.
        screen._model_install_kind = "remote"
        screen._model_install_reference = reference
        screen._model_install_service = MagicMock()
        screen._model_install_catalog = catalog
        screen._model_install_candidate = candidate
        screen._model_install_credential_resolver = MagicMock()
        fake_report = MagicMock(root=reference)

        provision_task = asyncio.create_task(
            screen._provision_remote(fake_report, catalog)
        )
        await pilot.pause()
        await pilot.pause()

        def _progress_text(view: RemoteView) -> str:
            widget = view.query_one(
                "#remote-model-install-progress", ModelInstallProgress
            )
            detail = widget.query_one("#model-install-progress-detail", Static)
            return str(detail.renderable)

        assert "model-part-1.gguf" in _progress_text(remote)

        screen.refresh(recompose=True)
        for _ in range(5):
            await pilot.pause()

        fresh_window = screen.query_one(LLMManagementWindow)
        fresh_remote = fresh_window.query_one(RemoteView)
        assert fresh_remote is not remote, (
            "test setup bug: recompose did not actually replace RemoteView"
        )

        # Half 1 of the fix: hydration.
        assert "model-part-1.gguf" in _progress_text(fresh_remote)
        fresh_text = "\n".join(
            str(item.renderable) for item in fresh_remote.query(Static)
        )
        assert resolved.repository in fresh_text
        assert candidate.files[0].upstream_path in fresh_text
        assert fresh_remote.query_one("#remote-model-install", Button).disabled

        # Half 2 of the fix: still updating, via this screen's own
        # still-running worker -- never owned by the RemoteView instance
        # the recompose tore down.
        resume.set()
        await provision_task
        await pilot.pause()
        await pilot.pause()

        assert "model-part-2.gguf" in _progress_text(fresh_remote)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("lifecycle_phase", "expected_status"),
    (
        ("preflight", "Preparing the managed install plan…"),
        ("pending-consent", "Awaiting review; no download has started."),
    ),
)
async def test_remote_context_survives_recompose_before_the_first_progress_tick(
    lifecycle_phase: str,
    expected_status: str,
    monkeypatch,
):
    """Remote context stays truthful throughout preflight and consent.

    This catches gating remount hydration on ``_model_install_active``: that
    flag is false until consent/progress, while ``_model_install_kind`` owns
    the full accepted-request lifecycle.
    """
    from unittest.mock import MagicMock

    from tldw_chatbook.UI.Screens.model_remote_view import RemoteView

    resolved = _resolved_remote_model()
    catalog = _remote_catalog()
    candidate = resolved.candidates[0]
    reference = catalog.artifact.reference
    # This test owns the recompose timing; suppress the unrelated managed-GGUF
    # startup read so its thread callback cannot target the deliberately removed
    # old window. Inventory behavior is covered by its own adoption tests.
    monkeypatch.setattr(
        LLMManagementWindow,
        "_refresh_managed_gguf_inventory",
        lambda _self: None,
    )
    app = _app()

    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        assert await _wait_for(lambda: bool(screen.query(RemoteView)), pilot)

        screen._model_install_kind = "remote"
        screen._model_install_reference = reference
        screen._model_install_service = MagicMock()
        screen._model_install_catalog = catalog
        screen._model_install_candidate = candidate
        screen._model_install_credential_resolver = MagicMock()
        screen._model_install_active = False
        screen._model_install_worker = (
            MagicMock() if lifecycle_phase == "preflight" else None
        )
        screen._model_install_pending_report = (
            None if lifecycle_phase == "preflight" else MagicMock(root=reference)
        )

        old_remote = screen.query_one(RemoteView)
        screen.refresh(recompose=True)
        assert await _wait_for(
            lambda: (
                bool(screen.query(RemoteView))
                and screen.query_one(RemoteView) is not old_remote
                and all(
                    marker
                    in "\n".join(
                        str(item.renderable)
                        for item in screen.query_one(RemoteView).query(Static)
                    )
                    for marker in (
                        resolved.repository,
                        candidate.files[0].upstream_path,
                    )
                )
            ),
            pilot,
        )
        fresh_remote = screen.query_one(RemoteView)
        detail_text = "\n".join(
            str(item.renderable) for item in fresh_remote.query(Static)
        )

        assert resolved.repository in detail_text
        assert candidate.files[0].upstream_path in detail_text
        assert fresh_remote.query_one("#remote-model-install", Button).disabled
        assert fresh_remote.query_one("#remote-model-search", Button).disabled
        assert (
            str(fresh_remote.query_one("#remote-model-status", Static).renderable)
            == expected_status
        )


@pytest.mark.asyncio
async def test_restored_remote_context_updates_phase_copy_without_another_recompose(
    monkeypatch,
):
    """An identical retained context still accepts later lifecycle copy."""
    from tldw_chatbook.UI.Screens.model_remote_view import RemoteView

    monkeypatch.setattr(
        LLMManagementWindow,
        "_refresh_managed_gguf_inventory",
        lambda _self: None,
    )
    catalog = _remote_catalog()
    candidate = _resolved_remote_model().candidates[0]
    app = _app()

    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        assert await _wait_for(lambda: bool(screen.query(RemoteView)), pilot)
        remote = screen.query_one(RemoteView)

        for expected_status in (
            "Preparing the managed install plan…",
            "Awaiting review; no download has started.",
            "Installing the selected GGUF variant…",
        ):
            assert remote.restore_install_context(
                catalog,
                candidate,
                status_message=expected_status,
            )
            await pilot.pause()
            assert (
                str(remote.query_one("#remote-model-status", Static).renderable)
                == expected_status
            )


@pytest.mark.asyncio
async def test_remote_install_progress_after_recompose_still_mirrors_into_installed_view(
    monkeypatch,
):
    """TASK-1914 mirror of ``test_curated_install_progress_after_recompose_
    still_mirrors_into_installed_view``: checks the MIRRORING handler's own
    effect on ``InstalledView`` for a remote install, after a real
    screen-level recompose.
    """
    import asyncio
    from unittest.mock import MagicMock

    import tldw_chatbook.Model_Artifacts.acquisition as acquisition_module
    from tldw_chatbook.Model_Artifacts.acquisition import AcquisitionProgress
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    catalog = _remote_catalog()
    reference = catalog.artifact.reference
    first_progress = AcquisitionProgress(
        "fetch", reference, "model-part-1.gguf", 100, 1024
    )
    second_progress = AcquisitionProgress(
        "fetch", reference, "model-part-2.gguf", 400, 1024
    )
    resume = asyncio.Event()

    class _FakeAcquisitionService:
        def __init__(self, _service, *, credential_resolver=None) -> None:
            """See the sibling recompose test above for this fake's rationale."""

        async def provision(
            self, root, consent, catalog, *, sources, progress, activate
        ):
            progress(first_progress)
            await resume.wait()
            progress(second_progress)
            return object()

    monkeypatch.setattr(
        acquisition_module, "ArtifactAcquisitionService", _FakeAcquisitionService
    )

    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        for _ in range(5):
            await pilot.pause()
        window = screen.query_one(LLMManagementWindow)
        installed = window.query_one(InstalledView)

        screen._model_install_kind = "remote"
        screen._model_install_reference = reference
        screen._model_install_service = MagicMock()
        screen._model_install_catalog = catalog
        screen._model_install_credential_resolver = MagicMock()
        fake_report = MagicMock(root=reference)

        provision_task = asyncio.create_task(
            screen._provision_remote(fake_report, catalog)
        )
        await pilot.pause()
        await pilot.pause()

        assert installed._install_progress == first_progress
        assert installed._install_active is True

        screen.refresh(recompose=True)
        for _ in range(5):
            await pilot.pause()

        fresh_window = screen.query_one(LLMManagementWindow)
        fresh_installed = fresh_window.query_one(InstalledView)
        assert fresh_installed is not installed, (
            "test setup bug: recompose did not actually replace InstalledView"
        )

        resume.set()
        await provision_task
        for _ in range(3):
            await pilot.pause()

        assert fresh_installed._install_progress == second_progress, (
            "InstalledView's mirroring handler never observed the post-recompose tick"
        )
        assert fresh_installed._install_active is True


@pytest.mark.asyncio
async def test_remote_install_progress_renders_exactly_once_per_tick_and_never_reaches_curated_view(
    monkeypatch,
):
    """TASK-1914: the mirror image of ``test_curated_install_progress_
    renders_exactly_once_per_tick``'s own new assertion -- a remote-install
    tick must render exactly once, on ``RemoteView``, and never reach the
    unrelated, currently-idle ``CuratedView`` (both are mounted at once;
    only ``_model_install_kind``-based routing in ``_active_install_view``
    prevents this).
    """
    from tldw_chatbook.Model_Artifacts.acquisition import AcquisitionProgress
    from tldw_chatbook.UI.Screens.model_curated_view import CuratedView
    from tldw_chatbook.UI.Screens.model_remote_view import RemoteView
    from tldw_chatbook.Widgets.ModelArtifacts import InstallProgressed

    remote_calls: list[AcquisitionProgress] = []
    original_remote_apply_progress = RemoteView.apply_progress

    def counting_remote_apply_progress(self, progress):
        remote_calls.append(progress)
        return original_remote_apply_progress(self, progress)

    monkeypatch.setattr(RemoteView, "apply_progress", counting_remote_apply_progress)

    curated_calls: list[AcquisitionProgress] = []
    original_curated_apply_progress = CuratedView.apply_progress

    def counting_curated_apply_progress(self, progress):
        curated_calls.append(progress)
        return original_curated_apply_progress(self, progress)

    monkeypatch.setattr(CuratedView, "apply_progress", counting_curated_apply_progress)

    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        for _ in range(5):
            await pilot.pause()

        catalog = _remote_catalog()
        reference = catalog.artifact.reference
        progress = AcquisitionProgress("fetch", reference, "model-q4.gguf", 1, 2)

        screen._model_install_kind = "remote"
        screen._deliver_curated(InstallProgressed(progress))
        await pilot.pause()
        await pilot.pause()
        await pilot.pause()

    assert remote_calls == [progress]
    assert len(remote_calls) == 1, (
        f"expected exactly one apply_progress call for one progress tick, "
        f"got {len(remote_calls)}"
    )
    assert curated_calls == [], (
        "a remote-install tick must never reach CuratedView.apply_progress "
        "-- _active_install_view routes by _model_install_kind precisely "
        "to prevent this"
    )


@pytest.mark.asyncio
async def test_remote_install_click_reaches_the_shared_consent_modal(monkeypatch):
    """Real candidate selection plus the contextual install action
    posts ``RemoteView.InstallRequested``, which ``LLMScreen`` resolves
    (through a stubbed acquisition service, so this stays network-free)
    into the exact shared ``ModelInstallModal``. Mirrors ``test_curated_
    install_click_reaches_the_shared_consent_modal``.
    """
    from unittest.mock import MagicMock

    import tldw_chatbook.Model_Artifacts.acquisition as acquisition_module
    from tldw_chatbook.Model_Artifacts.remote_huggingface import (
        HuggingFaceRemoteAdapter,
        build_remote_catalog,
    )
    from tldw_chatbook.UI.Screens.model_remote_view import RemoteView
    from tldw_chatbook.Widgets.ModelArtifacts import ModelInstallModal
    from textual.widgets import Input

    resolved = _resolved_remote_model()

    async def fake_resolve(self, repository, *, token=None):
        return resolved

    class _FakeAcquisitionService:
        def __init__(self, _service, *, credential_resolver=None) -> None:
            """Accept and discard the managed-store service/resolver."""

        async def preflight(self, ref, _catalog, *, sources):
            """Resolve a fake plan rooted at whatever reference was clicked."""
            report = MagicMock()
            report.root = ref
            return report

    monkeypatch.setattr(HuggingFaceRemoteAdapter, "resolve", fake_resolve)
    monkeypatch.setattr(
        acquisition_module, "ArtifactAcquisitionService", _FakeAcquisitionService
    )

    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        monkeypatch.setattr(app, "push_screen", MagicMock())
        for _ in range(5):
            await pilot.pause()

        remote_row = next(
            row for row in _rail_rows(screen) if row.lab_view_key == "remote"
        )
        remote_row.press()
        await pilot.pause()

        window = screen.query_one(LLMManagementWindow)
        remote = window.query_one(RemoteView)

        remote.query_one("#remote-model-query", Input).value = resolved.repository
        await pilot.click("#remote-model-search")
        for _ in range(50):
            if remote._resolved is not None:
                break
            await pilot.pause()
        assert remote._resolved is not None, "Remote view never finished resolving"

        candidate = remote.query_one(".remote-candidate")
        candidate.press()
        await pilot.pause()
        assert app.push_screen.called is False

        install = remote.query_one("#remote-model-install")
        install.press()
        await pilot.pause()
        await pilot.pause()

        for _ in range(20):
            if app.push_screen.called:
                break
            await pilot.pause()
        assert app.push_screen.called, "install action never reached push_screen"

        modal, callback = app.push_screen.call_args[0]
        assert isinstance(modal, ModelInstallModal)
        expected_catalog = build_remote_catalog(resolved, resolved.candidates[0])
        assert modal.report.root == expected_catalog.artifact.reference
        assert callback == screen._confirm_remote_install
        assert screen._model_install_pending_report is modal.report


@pytest.mark.parametrize(
    ("license_id", "expected_acknowledgment"),
    (
        (
            "NOASSERTION",
            "No license was declared. I reviewed the source and want to continue.",
        ),
        ("apache-2.0", None),
    ),
)
def test_apply_remote_preflight_result_requires_acknowledgment_only_for_unknown_license(
    license_id, expected_acknowledgment, tmp_path, monkeypatch
):
    """TASK-1914 mirror of ``RemoteView``'s former ``test_preflight_modal_
    requires_acknowledgment_only_for_unknown_license``, now driving
    ``LLMScreen._apply_remote_preflight_result`` directly -- this is where
    that logic lives now.
    """
    from unittest.mock import MagicMock

    from tldw_chatbook.UI.Screens import llm_screen as module
    from tldw_chatbook.Widgets.ModelArtifacts import ModelInstallModal

    catalog = _remote_catalog(license_id=license_id)
    candidate = _resolved_remote_model(license_id=license_id).candidates[0]
    report = _remote_report_for(catalog, tmp_path / "managed")
    fake_app = MagicMock()
    monkeypatch.setattr(module.LLMScreen, "app", property(lambda self: fake_app))

    screen = module.LLMScreen.__new__(module.LLMScreen)
    screen._remote_view = MagicMock(return_value=None)
    screen._model_install_worker = MagicMock()
    screen._model_install_catalog = catalog
    screen._model_install_candidate = candidate

    module.LLMScreen._apply_remote_preflight_result(screen, report, None)

    modal, callback = fake_app.push_screen.call_args.args
    assert isinstance(modal, ModelInstallModal)
    assert modal.required_acknowledgment == expected_acknowledgment
    # The source map is keyed by the MANAGED path (e.g. "model.gguf" for a
    # single non-sharded file, see remote_huggingface._managed_paths), not
    # the upstream filename -- read it back off the built catalog rather
    # than hardcoding that internal naming convention here.
    managed_path = catalog.artifact.files[0].path
    assert modal.selected_file_details == (
        (
            "model-q4.gguf",
            1024,
            "b" * 64,
            catalog.sources[catalog.artifact.reference][managed_path],
        ),
    )
    assert callback == screen._confirm_remote_install
    assert screen._model_install_pending_report is report


def test_remote_phase_copy_tracks_preflight_consent_and_active_transitions(
    tmp_path, monkeypatch
):
    """Mounted Remote detail follows the host-owned install lifecycle."""
    from unittest.mock import MagicMock, call

    from tldw_chatbook.UI.Screens import llm_screen as module

    catalog = _remote_catalog()
    candidate = _resolved_remote_model().candidates[0]
    report = _remote_report_for(catalog, tmp_path / "managed")
    fake_app = MagicMock()
    monkeypatch.setattr(module.LLMScreen, "app", property(lambda self: fake_app))

    view = MagicMock()
    view.is_mounted = True
    screen = module.LLMScreen.__new__(module.LLMScreen)
    screen._remote_view = MagicMock(return_value=view)
    screen.refresh_lab_status = MagicMock()
    screen._model_install_worker = MagicMock()
    screen._model_install_catalog = catalog
    screen._model_install_candidate = candidate
    screen._model_install_pending_report = None
    screen._model_install_active = False
    screen._model_install_succeeded = None
    screen._model_install_phase = None
    screen._model_install_kind = "remote"

    module.LLMScreen._apply_remote_preflight_result(screen, report, None)
    module.LLMScreen._model_install_status_changed(
        screen,
        module.InstallStatusChanged(catalog.artifact.reference, active=True),
    )

    assert view.restore_install_context.call_args_list == [
        call(
            catalog,
            candidate,
            status_message="Awaiting review; no download has started.",
        ),
        call(
            catalog,
            candidate,
            status_message="Installing the selected GGUF variant…",
        ),
    ]


def test_remote_terminal_action_literals_have_one_named_definition_each():
    """Terminal action values stay centralized instead of becoming magic strings."""
    import ast
    from collections import Counter
    import inspect

    from tldw_chatbook.UI.Screens import llm_screen as module

    tree = ast.parse(inspect.getsource(module))
    action_literals = Counter(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and node.value in {"finish", "cancel"}
    )

    assert action_literals == Counter({"finish": 1, "cancel": 1})
    assert module._REMOTE_INSTALL_TERMINAL_FINISH == "finish"
    assert module._REMOTE_INSTALL_TERMINAL_CANCEL == "cancel"


@pytest.mark.parametrize("operation", ("preflight", "installation"))
def test_remote_install_failures_log_exact_context(operation, monkeypatch, tmp_path):
    """Worker diagnostics classify remote failures without logging exception
    details, mirroring ``test_curated_install_failures_log_bounded_error_type``'s
    shape but pinning the remote-specific ``error_type=``/
    ``retryable=`` format (unchanged from ``RemoteView``'s own pre-TASK-1914
    worker methods).
    """
    from unittest.mock import MagicMock

    from tldw_chatbook.Model_Artifacts.acquisition import TransferError
    from tldw_chatbook.UI.Screens import llm_screen as module

    marker = "PRIVATE-REMOTE-INSTALL-DETAIL"
    catalog = _remote_catalog()
    fake_app = MagicMock()
    fake_logger = MagicMock()
    monkeypatch.setattr(module.LLMScreen, "app", property(lambda self: fake_app))
    monkeypatch.setattr(module, "logger", fake_logger)

    screen = module.LLMScreen.__new__(module.LLMScreen)
    screen._model_install_catalog = catalog

    if operation == "preflight":

        async def fail_preflight(_catalog):
            raise TransferError(marker, retryable=True)

        screen._preflight_remote = fail_preflight
        module.LLMScreen._run_remote_preflight.__wrapped__(screen)
    else:
        report = _remote_report_for(catalog, tmp_path / "managed")
        screen._model_install_pending_report = report

        async def fail_provision(_report, _catalog):
            raise TransferError(marker, retryable=True)

        screen._provision_remote = fail_provision
        module.LLMScreen._run_remote_provision.__wrapped__(screen)

    logged = " ".join(str(value) for value in fake_logger.error.call_args.args)
    assert "TransferError" in logged
    assert "retryable" in logged.casefold()
    assert "True" in logged
    assert marker not in logged
    assert marker not in str(fake_app.call_from_thread.call_args)


def test_remote_preflight_failure_notifies_and_does_not_push_a_modal(monkeypatch):
    """Sibling failure branch of ``test_remote_install_click_reaches_the_
    shared_consent_modal``, adapted from ``RemoteView``'s former direct
    coverage of ``_apply_preflight_result``'s failure path -- ``LLMScreen``
    resolves it now.
    """
    from unittest.mock import MagicMock

    from tldw_chatbook.UI.Screens import llm_screen as module

    fake_app = MagicMock()
    monkeypatch.setattr(module.LLMScreen, "app", property(lambda self: fake_app))

    catalog = _remote_catalog()
    screen = module.LLMScreen.__new__(module.LLMScreen)
    screen.notify = MagicMock()
    view = MagicMock()
    screen._remote_view = MagicMock(return_value=view)
    screen._model_install_worker = MagicMock()
    screen._model_install_reference = catalog.artifact.reference
    screen._model_install_service = MagicMock()
    screen._model_install_catalog = catalog
    screen._model_install_candidate = None
    screen._model_install_credential_resolver = MagicMock()
    screen._model_install_pending_report = None
    screen._model_install_kind = "remote"

    module.LLMScreen._apply_remote_preflight_result(screen, None, "boom")

    screen.notify.assert_called_once_with("boom", severity="error")
    fake_app.push_screen.assert_not_called()
    assert screen._model_install_worker is None
    assert screen._model_install_reference is None
    assert screen._model_install_service is None
    assert screen._model_install_catalog is None
    assert screen._model_install_candidate is None
    assert screen._model_install_credential_resolver is None
    assert screen._model_install_kind is None
    view.cancel_pending_install.assert_called_once_with("boom")


def test_declining_the_remote_consent_modal_does_not_start_the_install_worker():
    """Mirrors ``test_declining_the_consent_modal_does_not_start_the_
    install_worker`` for the remote flow."""
    from unittest.mock import MagicMock

    from tldw_chatbook.UI.Screens import llm_screen as module

    catalog = _remote_catalog()
    screen = module.LLMScreen.__new__(module.LLMScreen)
    view = MagicMock()
    screen._remote_view = MagicMock(return_value=view)
    screen._run_remote_provision = MagicMock()
    screen._model_install_worker = None
    screen._model_install_reference = catalog.artifact.reference
    screen._model_install_service = MagicMock()
    screen._model_install_catalog = catalog
    screen._model_install_candidate = None
    screen._model_install_credential_resolver = MagicMock()
    screen._model_install_pending_report = object()
    screen._model_install_kind = "remote"

    module.LLMScreen._confirm_remote_install(screen, False)

    screen._run_remote_provision.assert_not_called()
    assert screen._model_install_reference is None
    assert screen._model_install_pending_report is None
    assert screen._model_install_kind is None
    view.cancel_pending_install.assert_called_once_with(None)


@pytest.mark.parametrize(
    ("first_kind", "second_kind", "phase"),
    (
        ("curated", "remote", "worker"),
        ("remote", "curated", "worker"),
        ("curated", "curated", "pending_consent"),
        ("remote", "remote", "pending_consent"),
        ("curated", "remote", "pending_consent"),
        ("remote", "curated", "pending_consent"),
    ),
)
def test_a_second_concurrent_install_is_refused_regardless_of_kind_or_phase(
    first_kind, second_kind, phase
):
    """TASK-1914 fix round 2: curated and remote installs share ONE
    screen-level concurrency guard, ``_install_in_progress()`` (checking
    ``_model_install_kind``), not two independent ones and not a check on
    the worker handle -- documented in ``_install_in_progress``'s own
    docstring: the managed store's own ``ArtifactAcquisitionService.
    provision`` already serializes concurrent installs behind one
    in-process lease regardless of which view started them, so tracking
    two locks would only mean paying for a wasted preflight before the
    second one blocked anyway.

    Parametrized over both the kind pairing (same-flow and cross-flow)
    AND the phase the first install is in:

    - ``"worker"``: a preflight/provision thread is actually running
      (``_model_install_worker`` set, not finished) -- the case the
      original (pre-fix-round-2) guard already covered.
    - ``"pending_consent"``: the PR #1245 automated-review finding this
      round fixes. Preflight has already succeeded and the shared consent
      modal is up, awaiting the user's decision -- ``_model_install_
      worker`` is ``None`` here (see ``_apply_curated_preflight_result``/
      ``_apply_remote_preflight_result``, which reset it to ``None``
      before pushing the modal), which is exactly the window a
      worker-handle-only guard could not see. ``_model_install_kind``
      stays set through this window (cleared only in the terminal
      apply-provision-result/clear-state paths), so the fixed guard still
      refuses here.

    Every combination asserts the same contract: the second request is
    refused, notified, only the second (freshly clicking) view's own
    in-flight indicator is released, and the FIRST install's own
    ``_model_install_kind``/reference/pending-report survive byte-
    identical -- not merely equal, but the exact same retained objects,
    proving the second request never touched them before being refused.
    """
    from unittest.mock import MagicMock

    from tldw_chatbook.Model_Artifacts.service import ArtifactRef
    from tldw_chatbook.UI.Screens import llm_screen as module
    from tldw_chatbook.UI.Screens.model_curated_view import CuratedView
    from tldw_chatbook.UI.Screens.model_remote_view import RemoteView

    running_reference = ArtifactRef("model-a", "a" * 40, "int8")
    running_report = object()
    if phase == "worker":
        running_worker = MagicMock()
        running_worker.is_finished = False
    else:
        # The exact state _apply_curated_preflight_result/_apply_remote_
        # preflight_result leave behind the moment the shared consent
        # modal is pushed: no worker running, but the install is very
        # much still in progress.
        running_worker = None

    screen = module.LLMScreen.__new__(module.LLMScreen)
    screen.notify = MagicMock()
    screen._run_curated_preflight = MagicMock()
    screen._run_remote_preflight = MagicMock()
    curated_view = MagicMock()
    remote_view = MagicMock()
    screen._curated_view = MagicMock(return_value=curated_view)
    screen._remote_view = MagicMock(return_value=remote_view)
    screen._model_install_kind = first_kind
    screen._model_install_worker = running_worker
    screen._model_install_reference = running_reference
    screen._model_install_service = MagicMock()
    screen._model_install_registry = MagicMock() if first_kind == "curated" else None
    screen._model_install_sources = {} if first_kind == "curated" else None
    screen._model_install_catalog = (
        _remote_catalog() if first_kind == "remote" else None
    )
    screen._model_install_candidate = None
    screen._model_install_credential_resolver = (
        MagicMock() if first_kind == "remote" else None
    )
    screen._model_install_pending_report = running_report

    if second_kind == "curated":
        event = CuratedView.InstallRequested(
            ArtifactRef("model-b", "b" * 40, "int8"),
            service=MagicMock(),
            registry=MagicMock(),
            sources={},
        )
        event.stop = MagicMock()
        module.LLMScreen._curated_install_requested(screen, event)
        released_view = curated_view
        screen._run_curated_preflight.assert_not_called()
    else:
        second_catalog = _remote_catalog(license_id="mit")
        second_candidate = _resolved_remote_model(license_id="mit").candidates[0]
        event = RemoteView.InstallRequested(
            second_catalog,
            second_candidate,
            service=MagicMock(),
            credential_resolver=MagicMock(),
        )
        event.stop = MagicMock()
        module.LLMScreen._remote_install_requested(screen, event)
        released_view = remote_view
        screen._run_remote_preflight.assert_not_called()

    event.stop.assert_called_once_with()
    screen.notify.assert_called_once()
    released_view.cancel_pending_install.assert_called_once_with()
    # The in-progress install's own retained state must survive untouched
    # -- byte-identical (`is`), not merely equal, proving the second
    # request never overwrote it before being refused.
    assert screen._model_install_kind == first_kind
    assert screen._model_install_reference is running_reference
    assert screen._model_install_pending_report is running_report
    assert screen._model_install_worker is running_worker


@pytest.mark.parametrize(
    ("catalog", "candidate", "service", "credential_resolver"),
    (
        (None, "candidate", "service", "resolver"),
        ("not-a-catalog", "candidate", "service", "resolver"),
        (object(), "candidate", "service", "resolver"),
    ),
)
def test_remote_install_requested_refuses_an_invalid_catalog_without_starting_a_worker(
    catalog, candidate, service, credential_resolver
):
    """TASK-1914, applying TASK-1803 review round 2's lesson from the
    start: an invalid ``event.catalog`` must never be stored or acted on.
    """
    from unittest.mock import MagicMock

    from tldw_chatbook.UI.Screens import llm_screen as module
    from tldw_chatbook.UI.Screens.model_remote_view import RemoteView

    screen = module.LLMScreen.__new__(module.LLMScreen)
    screen.notify = MagicMock()
    screen._run_remote_preflight = MagicMock()
    view = MagicMock()
    screen._remote_view = MagicMock(return_value=view)
    screen._model_install_worker = None
    screen._model_install_reference = None
    screen._model_install_service = None
    screen._model_install_catalog = None
    screen._model_install_candidate = None
    screen._model_install_credential_resolver = None
    screen._model_install_pending_report = None
    screen._model_install_kind = None

    event = RemoteView.InstallRequested(
        catalog,
        candidate,
        service=service,
        credential_resolver=credential_resolver,
    )
    event.stop = MagicMock()

    module.LLMScreen._remote_install_requested(screen, event)

    event.stop.assert_called_once_with()
    screen._run_remote_preflight.assert_not_called()
    screen.notify.assert_called_once_with(
        "Could not start the model install: invalid request.",
        severity="error",
    )
    # _clear_remote_install_state's default message=None reaches
    # RemoteView.cancel_pending_install as an explicit None (unlike
    # CuratedView.cancel_pending_install, which takes no argument at all).
    view.cancel_pending_install.assert_called_once_with(None)
    assert screen._model_install_worker is None
    assert screen._model_install_reference is None
    assert screen._model_install_kind is None


@pytest.mark.parametrize(
    ("missing_field",),
    (("candidate",), ("service",), ("credential_resolver",)),
)
def test_remote_install_requested_refuses_when_candidate_service_or_resolver_is_missing(
    missing_field,
):
    """Same validation as above, covering ``event.candidate``/``event.
    service``/``event.credential_resolver`` being missing/invalid, not
    only a malformed ``event.catalog``.
    """
    from unittest.mock import MagicMock

    from tldw_chatbook.UI.Screens import llm_screen as module
    from tldw_chatbook.UI.Screens.model_remote_view import RemoteView

    catalog = _remote_catalog()
    fields = {
        "candidate": _resolved_remote_model().candidates[0],
        "service": "service",
        "credential_resolver": "resolver",
    }
    fields[missing_field] = None

    screen = module.LLMScreen.__new__(module.LLMScreen)
    screen.notify = MagicMock()
    screen._run_remote_preflight = MagicMock()
    view = MagicMock()
    screen._remote_view = MagicMock(return_value=view)
    screen._model_install_kind = None
    screen._model_install_worker = None
    screen._model_install_reference = None
    screen._model_install_service = None
    screen._model_install_catalog = None
    screen._model_install_candidate = None
    screen._model_install_credential_resolver = None

    event = RemoteView.InstallRequested(catalog, **fields)
    event.stop = MagicMock()

    module.LLMScreen._remote_install_requested(screen, event)

    screen._run_remote_preflight.assert_not_called()
    screen.notify.assert_called_once_with(
        "Could not start the model install: invalid request.",
        severity="error",
    )
    assert screen._model_install_reference is None


def test_run_remote_preflight_schedules_apply_result_when_catalog_is_none(monkeypatch):
    """Defense-in-depth mirror of ``test_run_curated_preflight_schedules_
    apply_result_when_reference_is_none``: ``_run_remote_preflight`` must
    never trust that ``_curated_install_requested``'s sibling validation
    always ran first.
    """
    from unittest.mock import MagicMock

    from tldw_chatbook.UI.Screens import llm_screen as module

    fake_app = MagicMock()
    monkeypatch.setattr(module.LLMScreen, "app", property(lambda self: fake_app))

    screen = module.LLMScreen.__new__(module.LLMScreen)
    screen._model_install_catalog = None

    module.LLMScreen._run_remote_preflight.__wrapped__(screen)

    fake_app.call_from_thread.assert_called_once()
    args = fake_app.call_from_thread.call_args[0]
    assert args[0] == screen._apply_remote_preflight_result
    assert args[1] is None
    assert isinstance(args[2], str) and args[2]


@pytest.mark.parametrize(
    ("error", "expected_severity", "expected_message"),
    (
        (
            None,
            "information",
            "Model downloaded and managed. Runtime compatibility has not "
            "been verified.",
        ),
        ("boom", "error", "boom"),
    ),
)
def test_apply_remote_provision_result_notifies_mirrors_and_resets_state(
    error, expected_severity, expected_message, monkeypatch
):
    """Mirrors ``test_apply_curated_provision_result_notifies_mirrors_and_
    resets_state`` for the remote flow -- including the different success
    copy (no activation) and that ``RemoteView.finish_install`` receives
    the exact same message text as ``notify``.
    """
    from unittest.mock import MagicMock

    from tldw_chatbook.UI.Screens import llm_screen as module

    fake_app = MagicMock()
    monkeypatch.setattr(module.LLMScreen, "app", property(lambda self: fake_app))

    catalog = _remote_catalog()
    reference = catalog.artifact.reference
    screen = module.LLMScreen.__new__(module.LLMScreen)
    screen.notify = MagicMock()
    screen._deliver_curated = MagicMock()
    view = MagicMock()
    screen._remote_view = MagicMock(return_value=view)
    screen._model_install_worker = MagicMock()
    screen._model_install_reference = reference
    screen._model_install_service = MagicMock()
    screen._model_install_catalog = catalog
    screen._model_install_candidate = _resolved_remote_model().candidates[0]
    screen._model_install_credential_resolver = MagicMock()
    screen._model_install_pending_report = object()
    screen._model_install_kind = "remote"

    module.LLMScreen._apply_remote_provision_result(screen, error)

    screen.notify.assert_called_once_with(expected_message, severity=expected_severity)
    assert screen._model_install_worker is None
    assert screen._model_install_pending_report is None
    assert screen._model_install_reference is None
    assert screen._model_install_service is None
    assert screen._model_install_catalog is None
    assert screen._model_install_candidate is None
    assert screen._model_install_credential_resolver is None
    assert screen._model_install_kind is None

    screen._deliver_curated.assert_called_once()
    delivered = screen._deliver_curated.call_args[0][0]
    assert isinstance(delivered, module.InstallStatusChanged)
    assert delivered.reference == reference
    assert delivered.active is False
    assert delivered.succeeded == (error is None)

    if error is None:
        view.finish_install.assert_called_once_with(
            expected_message,
            completed_reference=reference,
        )
    else:
        view.finish_install.assert_called_once_with(expected_message)


@pytest.mark.parametrize(
    ("terminal_path", "error", "expected_method", "expected_message"),
    (
        (
            "provision",
            None,
            "finish_install",
            "Model downloaded and managed. Runtime compatibility has not been verified.",
        ),
        ("provision", "download failed", "finish_install", "download failed"),
        ("preflight", "plan failed", "cancel_pending_install", "plan failed"),
    ),
)
def test_remote_terminal_outcome_crosses_the_recompose_gap(
    terminal_path,
    error,
    expected_method,
    expected_message,
    monkeypatch,
):
    """A terminal result is retained until a mounted RemoteView consumes it.

    This deterministically models the teardown/remount gap by returning no
    view at the terminal call, then a fresh mounted view at hydration time.
    Clearing the only catalog/candidate copy before that second step makes
    the observable outcome call disappear and fails this test.
    """
    from unittest.mock import MagicMock

    from tldw_chatbook.UI.Screens import llm_screen as module

    fake_app = MagicMock()
    monkeypatch.setattr(module.LLMScreen, "app", property(lambda self: fake_app))
    catalog = _remote_catalog()
    candidate = _resolved_remote_model().candidates[0]
    screen = module.LLMScreen.__new__(module.LLMScreen)
    screen.notify = MagicMock()
    screen._deliver_curated = MagicMock()
    screen._remote_view = MagicMock(return_value=None)
    screen._installed_view = MagicMock(return_value=None)
    screen._model_install_worker = MagicMock()
    screen._model_install_reference = catalog.artifact.reference
    screen._model_install_service = MagicMock()
    screen._model_install_catalog = catalog
    screen._model_install_candidate = candidate
    screen._model_install_credential_resolver = MagicMock()
    screen._model_install_pending_report = object()
    screen._model_install_kind = "remote"
    screen._model_install_active = False
    screen._model_install_last_progress = None

    if terminal_path == "provision":
        module.LLMScreen._apply_remote_provision_result(screen, error)
    else:
        module.LLMScreen._clear_remote_install_state(screen, error)

    fresh_view = MagicMock()
    fresh_view.is_mounted = True
    fresh_view.restore_install_context.return_value = True
    screen._remote_view.return_value = fresh_view
    module.LLMScreen._hydrate_model_install_progress(screen)

    fresh_view.restore_install_context.assert_called_once_with(catalog, candidate)
    outcome = getattr(fresh_view, expected_method)
    if terminal_path == "provision" and error is None:
        outcome.assert_called_once_with(
            expected_message,
            completed_reference=catalog.artifact.reference,
        )
    else:
        outcome.assert_called_once_with(expected_message)


def test_successful_remote_completion_survives_later_recomposes(monkeypatch):
    """A consumed success remains durable until Remote starts new discovery."""
    from unittest.mock import MagicMock

    from tldw_chatbook.UI.Screens import llm_screen as module

    fake_app = MagicMock()
    monkeypatch.setattr(module.LLMScreen, "app", property(lambda self: fake_app))
    catalog = _remote_catalog()
    candidate = _resolved_remote_model().candidates[0]
    reference = catalog.artifact.reference
    screen = module.LLMScreen.__new__(module.LLMScreen)
    screen.notify = MagicMock()
    screen._deliver_curated = MagicMock()
    screen._remote_view = MagicMock(return_value=None)
    screen._installed_view = MagicMock(return_value=None)
    screen._model_install_worker = MagicMock()
    screen._model_install_reference = reference
    screen._model_install_service = MagicMock()
    screen._model_install_catalog = catalog
    screen._model_install_candidate = candidate
    screen._model_install_credential_resolver = MagicMock()
    screen._model_install_pending_report = object()
    screen._model_install_kind = "remote"
    screen._model_install_active = False
    screen._model_install_last_progress = None
    screen._remote_install_terminal_catalog = None
    screen._remote_install_terminal_candidate = None
    screen._remote_install_terminal_action = None
    screen._remote_install_terminal_message = None
    screen._remote_install_completed_catalog = None
    screen._remote_install_completed_candidate = None
    screen._remote_install_completed_reference = None
    screen._remote_install_completed_message = None

    module.LLMScreen._apply_remote_provision_result(screen, None)

    first_view = MagicMock(is_mounted=True)
    first_view.restore_install_context.return_value = True
    screen._remote_view.return_value = first_view
    module.LLMScreen._hydrate_model_install_progress(screen)

    second_view = MagicMock(is_mounted=True)
    second_view.restore_install_context.return_value = True
    screen._remote_view.return_value = second_view
    module.LLMScreen._hydrate_model_install_progress(screen)

    for view in (first_view, second_view):
        view.restore_install_context.assert_called_once_with(catalog, candidate)
        view.finish_install.assert_called_once_with(
            "Model downloaded and managed. Runtime compatibility has not been verified.",
            completed_reference=reference,
        )


def test_new_remote_discovery_clears_durable_completion_identity():
    """A new query supersedes the prior completed model at screen scope."""
    from tldw_chatbook.UI.Screens import llm_screen as module
    from tldw_chatbook.UI.Screens.model_remote_view import RemoteView

    screen = module.LLMScreen.__new__(module.LLMScreen)
    screen._remote_install_completed_catalog = object()
    screen._remote_install_completed_candidate = object()
    screen._remote_install_completed_reference = object()
    screen._remote_install_completed_message = "done"

    screen._remote_discovery_started(RemoteView.DiscoveryStarted("new model"))

    assert screen._remote_install_completed_catalog is None
    assert screen._remote_install_completed_candidate is None
    assert screen._remote_install_completed_reference is None
    assert screen._remote_install_completed_message is None


def test_open_installed_switches_and_reveals_exact_reference_without_activation():
    """The Remote completion action is navigation, never implicit activation."""
    from unittest.mock import MagicMock

    from tldw_chatbook.Model_Artifacts.service import ArtifactRef
    from tldw_chatbook.UI.Screens import llm_screen as module
    from tldw_chatbook.UI.Screens.model_remote_view import RemoteView

    reference = ArtifactRef("remote-gguf", "a" * 40, "q4_k_m")
    screen = module.LLMScreen.__new__(module.LLMScreen)
    screen.llm_window = MagicMock()
    installed = MagicMock()
    screen._installed_view = MagicMock(return_value=installed)
    screen.call_after_refresh = MagicMock(
        side_effect=lambda callback, *args: callback(*args)
    )
    event = RemoteView.OpenInstalledRequested(reference)

    screen._remote_open_installed_requested(event)

    assert screen.llm_window.active_view == "installed"
    screen.call_after_refresh.assert_called_once_with(
        installed.reveal_reference,
        reference,
    )
    installed.reveal_reference.assert_called_once_with(reference)
    assert not any(call[0] == "activate" for call in installed.method_calls)


@pytest.mark.asyncio
async def test_open_installed_preserves_reference_until_first_lazy_mount(
    monkeypatch,
):
    """First-use Installed navigation must replay the exact requested root."""

    from tldw_chatbook.Model_Artifacts.service import ArtifactRef
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView
    from tldw_chatbook.UI.Screens.model_remote_view import RemoteView

    reference = ArtifactRef("remote-gguf", "a" * 40, "q4_k_m")
    revealed = []
    original_reveal = InstalledView.reveal_reference

    def capture_reveal(self, requested):
        revealed.append(requested)
        return original_reveal(self, requested)

    monkeypatch.setattr(InstalledView, "reveal_reference", capture_reveal)
    app = _app()
    async with app.run_test(size=(80, 24)) as pilot:
        screen = await _models_screen(app, populate_all=False)
        for _ in range(8):
            await pilot.pause()
        window = screen.query_one(LLMManagementWindow)
        assert not list(window.query("#installed-models-view"))

        screen._remote_open_installed_requested(
            RemoteView.OpenInstalledRequested(reference)
        )
        for _ in range(8):
            await pilot.pause()

        assert revealed == [reference]
        assert window.active_view == "installed"
        assert window.query_one("#installed-models-view", InstalledView).is_mounted


@pytest.mark.asyncio
async def test_open_installed_exact_row_focus_wins_real_window_switch(
    tmp_path,
):
    """The handoff focus must run after Installed's standard focus restore."""
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef
    from tldw_chatbook.UI.Screens.model_browser_state import InventoryRow
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView
    from tldw_chatbook.UI.Screens.model_remote_view import RemoteView

    reference = ArtifactRef("remote-gguf", "a" * 40, "q4_k_m")
    row = InventoryRow(
        path=tmp_path / reference.artifact_id,
        reference=reference,
        model_label=reference.artifact_id,
        revision=reference.revision,
        precision=reference.variant,
        dependencies=(),
        ready=True,
        active=False,
        activation_allowed=True,
        is_broken=False,
        is_unmanaged=False,
        provenance="Integrity verified",
        action_hint="Ready",
        error=None,
        size_bytes=1024,
        installed_store_bytes=1024,
        staging_store_bytes=0,
        free_bytes=4096,
    )

    app = _app()
    async with app.run_test(size=(80, 24)) as pilot:
        screen = await _models_screen(app)
        assert await _wait_for(
            lambda: bool(screen.query("#installed-models-view")),
            pilot,
        )
        window = screen.query_one(LLMManagementWindow)
        installed = window.query_one("#installed-models-view", InstalledView)
        installed._loaded = True
        installed._rows = (row,)
        installed.refresh(recompose=True)
        assert await _wait_for(
            lambda: bool(installed.query(".installed-model-row")),
            pilot,
        )
        window._model_library_focus_ids["installed"] = "installed-models-refresh"

        screen._remote_open_installed_requested(
            RemoteView.OpenInstalledRequested(reference)
        )
        for _ in range(4):
            await pilot.pause()

        focused = app.focused
        assert window.active_view == "installed"
        assert focused is not None
        assert focused.has_class("model-activate")
        assert any(
            getattr(ancestor, "reference", None) == reference
            for ancestor in focused.ancestors_with_self
        )


def test_configure_runtime_request_opens_choice_and_preserves_exact_reference(
    monkeypatch,
):
    """The host owns provider choice while Remote contributes only identity."""
    from unittest.mock import MagicMock

    from tldw_chatbook.Model_Artifacts.service import ArtifactRef
    from tldw_chatbook.UI.Screens import llm_screen as module
    from tldw_chatbook.UI.Screens.model_remote_view import RemoteView
    from tldw_chatbook.Widgets.ModelArtifacts import ManagedGGUFRuntimeChoiceModal

    fake_app = MagicMock()
    monkeypatch.setattr(module.LLMScreen, "app", property(lambda self: fake_app))
    reference = ArtifactRef("remote-gguf", "a" * 40, "q4_k_m")
    screen = module.LLMScreen.__new__(module.LLMScreen)
    screen.llm_window = MagicMock()
    screen.llm_window.configure_managed_gguf.return_value = True
    event = RemoteView.ConfigureRuntimeRequested(reference)

    screen._remote_configure_runtime_requested(event)

    modal, callback = fake_app.push_screen.call_args.args
    assert isinstance(modal, ManagedGGUFRuntimeChoiceModal)
    callback("llamacpp")
    screen.llm_window.configure_managed_gguf.assert_called_once_with(
        "llamacpp",
        reference,
    )


def test_runtime_refresh_rejection_clears_screen_owned_handoff():
    """A synchronous lifecycle refusal cannot survive as pending intent."""
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef

    reference = ArtifactRef("remote-gguf", "a" * 40, "q4_k_m")
    screen = LLMScreen.__new__(LLMScreen)
    screen.llm_window = MagicMock()
    screen.llm_window.configure_managed_gguf.return_value = False
    screen._remote_runtime_handoff = None
    screen.notify = MagicMock()

    screen._remote_runtime_selected(reference, "llamacpp")

    assert screen._remote_runtime_handoff is None
    screen.notify.assert_called_once_with(
        "Stop the active Llama.cpp or Llamafile server, then configure this "
        "managed model again.",
        severity="warning",
    )


@pytest.mark.asyncio
async def test_pending_runtime_handoff_replays_into_recomposed_models_window(
    monkeypatch,
):
    """A fresh Models body must retain a still-resolving exact handoff."""
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef

    reference = ArtifactRef("remote-gguf", "a" * 40, "q4_k_m")
    accepted: list[tuple[LLMManagementWindow, str, ArtifactRef]] = []

    def configure(
        window: LLMManagementWindow,
        provider: str,
        received: ArtifactRef,
    ) -> bool:
        accepted.append((window, provider, received))
        return True

    monkeypatch.setattr(LLMManagementWindow, "configure_managed_gguf", configure)

    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        assert await _wait_for(lambda: bool(screen.query("#remote-models-view")), pilot)
        first_window = screen.query_one(LLMManagementWindow)

        screen._remote_runtime_selected(reference, "llamacpp")
        assert accepted == [(first_window, "llamacpp", reference)]

        screen.refresh(recompose=True)
        assert await _wait_for(
            lambda: (
                screen.llm_window is not first_window
                and screen.llm_window is not None
                and screen.llm_window.is_attached
            ),
            pilot,
        )
        replacement = screen.llm_window
        assert replacement is not None
        assert await _wait_for(lambda: len(accepted) == 2, pilot)

        assert accepted == [
            (first_window, "llamacpp", reference),
            (replacement, "llamacpp", reference),
        ]


def test_runtime_handoff_clears_only_after_matching_window_resolution():
    """Detached-window results cannot clear a replacement window's intent."""
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef

    reference = ArtifactRef("remote-gguf", "a" * 40, "q4_k_m")
    stale = ArtifactRef("other-gguf", "b" * 40, "q8_0")
    screen = LLMScreen.__new__(LLMScreen)
    screen._remote_runtime_handoff = ("llamacpp", reference)
    screen.notify = MagicMock()

    screen._managed_gguf_handoff_resolved(
        LLMManagementWindow.ManagedGGUFHandoffResolved(
            "llamacpp",
            stale,
            succeeded=True,
        )
    )
    assert screen._remote_runtime_handoff == ("llamacpp", reference)

    screen._managed_gguf_handoff_resolved(
        LLMManagementWindow.ManagedGGUFHandoffResolved(
            "llamacpp",
            reference,
            succeeded=True,
        )
    )
    assert screen._remote_runtime_handoff is None
    screen.notify.assert_not_called()


def test_runtime_handoff_failure_surfaces_inventory_specific_recovery():
    """A resolved inventory failure clears intent with actionable copy."""
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef

    reference = ArtifactRef("remote-gguf", "a" * 40, "q4_k_m")
    screen = LLMScreen.__new__(LLMScreen)
    screen._remote_runtime_handoff = ("llamafile", reference)
    screen.notify = MagicMock()

    screen._managed_gguf_handoff_resolved(
        LLMManagementWindow.ManagedGGUFHandoffResolved(
            "llamafile",
            reference,
            succeeded=False,
            reason="inventory-error",
        )
    )

    assert screen._remote_runtime_handoff is None
    screen.notify.assert_called_once_with(
        "Managed models could not be loaded. Refresh Installed models, then try again.",
        severity="warning",
    )


@pytest.mark.asyncio
async def test_preflight_remote_receives_exact_catalog_sources_and_credential_resolver(
    tmp_path, monkeypatch
):
    """Mirrors ``RemoteView``'s former ``test_preflight_receives_exact_
    catalog_sources_and_fresh_resolver``, now against ``LLMScreen.
    _preflight_remote``.
    """
    from tldw_chatbook.UI.Screens import llm_screen as module

    catalog = _remote_catalog()
    report = _remote_report_for(catalog, tmp_path / "managed")
    core = object()
    resolver = object()
    captured: dict[str, object] = {}

    class _Acquisition:
        def __init__(self, received_core, *, credential_resolver) -> None:
            captured["core"] = received_core
            captured["resolver"] = credential_resolver

        async def preflight(self, root, received_catalog, *, sources):
            captured["preflight"] = (root, received_catalog, sources)
            return report

    import tldw_chatbook.Model_Artifacts.acquisition as acquisition_module

    monkeypatch.setattr(acquisition_module, "ArtifactAcquisitionService", _Acquisition)

    screen = module.LLMScreen.__new__(module.LLMScreen)
    screen._model_install_service = core
    screen._model_install_credential_resolver = resolver

    actual = await screen._preflight_remote(catalog)

    assert actual is report
    assert captured == {
        "core": core,
        "resolver": resolver,
        "preflight": (
            catalog.artifact.reference,
            catalog,
            catalog.sources,
        ),
    }


@pytest.mark.asyncio
async def test_provision_remote_reuses_exact_preflight_values_without_activation(
    tmp_path, monkeypatch
):
    """Mirrors ``RemoteView``'s former ``test_provision_reuses_exact_
    preflight_values_without_activation``, now against ``LLMScreen.
    _provision_remote``: any catalog/source substitution or activation
    would violate reviewed consent.
    """
    from unittest.mock import MagicMock

    import tldw_chatbook.Model_Artifacts.acquisition as acquisition_module
    from tldw_chatbook.UI.Screens import llm_screen as module

    catalog = _remote_catalog()
    report = _remote_report_for(catalog, tmp_path / "managed")
    core = object()
    resolver = object()
    captured: dict[str, object] = {}

    class _Acquisition:
        def __init__(self, received_core, *, credential_resolver) -> None:
            captured["core"] = received_core
            captured["resolver"] = credential_resolver

        async def provision(
            self,
            root,
            consent,
            received_catalog,
            *,
            sources,
            progress,
            activate,
        ):
            captured["provision"] = (
                root,
                consent,
                received_catalog,
                sources,
                progress,
                activate,
            )

    monkeypatch.setattr(acquisition_module, "ArtifactAcquisitionService", _Acquisition)

    screen = module.LLMScreen.__new__(module.LLMScreen)
    screen._model_install_service = core
    screen._model_install_credential_resolver = resolver
    screen._deliver_curated = MagicMock()

    await screen._provision_remote(report, catalog)

    root, consent, actual_catalog, sources, progress, activate = captured["provision"]
    assert captured["core"] is core
    assert captured["resolver"] is resolver
    assert root == report.root
    assert consent == report.grant()
    assert actual_catalog is catalog
    assert sources is catalog.sources
    assert callable(progress)
    assert activate is False


class _FakeExternalSourceService:
    """Complete fake for the exact production service surface Lab consumes."""

    def __init__(
        self,
        *,
        records=None,
        block_verification: bool = False,
        block_plan: bool = False,
        block_copy: bool = False,
        block_prefer: bool = False,
    ) -> None:
        import threading

        self._records = dict(records or {})
        self.block_verification = block_verification
        self.block_plan = block_plan
        self.block_copy = block_copy
        self.block_prefer = block_prefer
        self.progress_seen = threading.Event()
        self.release_verification = threading.Event()
        self.plan_started = threading.Event()
        self.release_plan = threading.Event()
        self.plan_returned = threading.Event()
        self.plan_returned_at = 0.0
        self.copy_started = threading.Event()
        self.copy_cancelled = threading.Event()
        self.release_copy = threading.Event()
        self.copy_continued = threading.Event()
        self.prefer_started = threading.Event()
        self.release_prefer = threading.Event()
        self.vad_ready = True
        self.prepare_threads: list[int] = []
        self.commit_threads: list[int] = []
        self.stop_threads: list[int] = []
        self.prefer_threads: list[int] = []
        self.prepare_calls = []
        self.commit_attempts = []
        self.committed = []
        self.stopped = []
        self.preferred = []
        self.released_scopes = []
        self.copy_plans = []
        self.copied = []

    def records(self):
        return dict(self._records)

    def close(self) -> None:
        pass

    def may_delete(self, _reference):
        return None

    def on_root_activated(self, _reference):
        return None

    def prefer_managed(self, key, *, cancelled=lambda: False):
        import threading

        self.prefer_started.set()
        if self.block_prefer:
            self.release_prefer.wait(3)
        if cancelled():
            return
        self.prefer_threads.append(threading.get_ident())
        self.preferred.append(key)

    def release_scope(self, scope_id):
        self.released_scopes.append(scope_id)

    def prepare_external(
        self,
        key,
        directory,
        *,
        owner=None,
        cancelled=lambda: False,
        progress=None,
    ):
        import threading
        from types import SimpleNamespace

        from tldw_chatbook.Local_Ingestion.parakeet_v2_artifact import (
            parakeet_reference,
        )

        self.prepare_threads.append(threading.get_ident())
        self.prepare_calls.append((key, directory, owner))
        if progress is not None:
            progress(4, 8)
        self.progress_seen.set()
        if self.block_verification:
            self.release_verification.wait(3)
        if cancelled():
            raise RuntimeError("cancelled")
        verified = SimpleNamespace(
            directory=directory,
            reference=parakeet_reference(key.model_id, key.precision),
        )
        return SimpleNamespace(key=key, verified=verified)

    def commit_external(self, prepared, *, cancelled=lambda: False):
        import threading

        from tldw_chatbook.STT.parakeet_sources import (
            ParakeetSourceError,
            ParakeetSourceErrorCode,
        )

        self.commit_threads.append(threading.get_ident())
        self.commit_attempts.append(prepared)
        if cancelled():
            return
        if not self.vad_ready:
            raise ParakeetSourceError(ParakeetSourceErrorCode.VAD_UNAVAILABLE)
        self.committed.append(prepared)

    def stop_using_external(self, key, *, cancelled=lambda: False):
        import threading

        if cancelled():
            return
        self.stop_threads.append(threading.get_ident())
        self.stopped.append(key)

    def plan_managed_copy(self, verified):
        import time

        from tldw_chatbook.STT.parakeet_sources import ManagedCopyPlan

        self.copy_plans.append(verified)
        if self.block_plan:
            self.plan_started.set()
            self.release_plan.wait(3)
            self.plan_returned_at = time.monotonic()
            self.plan_returned.set()
        return ManagedCopyPlan(
            reference=verified.reference,
            additional_bytes=1024,
            destination=verified.directory.parent / "managed-store",
            free_bytes=4096,
            already_installed=False,
        )

    def copy_into_managed(self, verified, consent, *, cancelled=lambda: False):
        if self.block_copy:
            self.copy_started.set()
            while not cancelled() and not self.release_copy.is_set():
                self.release_copy.wait(0.01)
            if cancelled():
                self.copy_cancelled.set()
                raise RuntimeError("cancelled")
            self.copy_continued.set()
        self.copied.append((verified, consent))
        return verified.reference


@pytest.mark.parametrize(
    ("code", "message", "is_error"),
    (
        (
            "MISSING",
            "Required model files are missing. Choose a complete model directory.",
            True,
        ),
        (
            "IRREGULAR",
            "Model files must be regular files without links. Choose a safe model directory.",
            True,
        ),
        (
            "CHANGED",
            "Model files changed during verification. Wait for file changes to finish, then retry.",
            True,
        ),
        (
            "CORRUPT",
            "Model files do not match the curated model. Choose an unmodified model directory.",
            True,
        ),
        (
            "UNSUPPORTED",
            "This curated model does not support an external directory.",
            True,
        ),
        (
            "CANCELLED",
            "Verification cancelled. The prior source is unchanged.",
            False,
        ),
    ),
)
def test_external_verification_codes_have_distinct_path_free_recovery_copy(
    tmp_path,
    monkeypatch,
    code,
    message,
    is_error,
):
    from types import SimpleNamespace

    from tldw_chatbook.STT.parakeet_external import (
        ExternalParakeetErrorCode,
        ExternalParakeetVerificationError,
    )
    from tldw_chatbook.STT.parakeet_sources import ParakeetSourceKey
    from tldw_chatbook.UI.Screens import llm_screen as module

    selected = (tmp_path / "private-model-root").absolute()
    service = MagicMock()
    service.prepare_external.side_effect = ExternalParakeetVerificationError(
        ExternalParakeetErrorCode[code]
    )
    fake_app = MagicMock()
    fake_app._ensure_parakeet_source_service.return_value = service
    monkeypatch.setattr(module.LLMScreen, "app", property(lambda _self: fake_app))
    monkeypatch.setattr(
        module,
        "get_current_worker",
        lambda: SimpleNamespace(is_cancelled=False),
    )
    fake_logger = MagicMock()
    monkeypatch.setattr(module, "logger", fake_logger)
    screen = module.LLMScreen.__new__(module.LLMScreen)
    token = (1, id(screen))

    module.LLMScreen._verify_external_source.__wrapped__(
        screen,
        token,
        ParakeetSourceKey.V2_INT8,
        selected,
        "commit",
    )

    callback = fake_app.call_from_thread.call_args.args
    assert callback[:4] == (
        screen._apply_external_verification_result,
        token,
        "commit",
        None,
    )
    assert callback[4:] == (message, is_error)
    assert str(selected) not in message
    if code == "CANCELLED":
        fake_logger.warning.assert_not_called()
    else:
        fake_logger.warning.assert_called_once()


def test_external_verification_cancellation_is_information_not_error():
    screen = LLMScreen.__new__(LLMScreen)
    screen._external_selection_worker = object()
    screen._owns_external_token = MagicMock(return_value=True)
    screen._release_external_scope = MagicMock()
    screen._set_external_status = MagicMock()
    screen.notify = MagicMock()
    token = (1, id(screen))
    message = "Verification cancelled. The prior source is unchanged."

    screen._apply_external_verification_result(
        token,
        "commit",
        None,
        message,
        False,
    )

    screen._release_external_scope.assert_called_once_with(token)
    screen._set_external_status.assert_called_once_with(
        message,
        error=False,
        active=False,
    )
    screen.notify.assert_called_once_with(message, severity="information")


async def _wait_for(condition, pilot, *, attempts: int = 120) -> bool:
    deadline = time.monotonic() + max(30.0, attempts * 0.02)
    while time.monotonic() < deadline:
        if condition():
            return True
        await pilot.pause(0.01)
    return condition()


@pytest.mark.asyncio
async def test_external_rail_mounts_through_the_existing_deferred_view_pattern():
    from tldw_chatbook.UI.Screens.model_external_view import ExternalModelView

    app = _app()
    app._parakeet_source_service = _FakeExternalSourceService()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app, populate_all=False)
        await pilot.pause()
        await pilot.pause()
        window = screen.query_one(LLMManagementWindow)
        window.active_view = "external"
        assert await _wait_for(
            lambda: bool(screen.query("#external-models-view")), pilot
        )
        assert window.query_one("#external-models-view", ExternalModelView)

        external_row = next(
            row for row in _rail_rows(screen) if row.lab_view_key == "external"
        )
        external_row.press()
        await pilot.pause()
        assert window.active_view == "external"
        assert "-active" in window.query_one("#llm-view-external").classes


@pytest.mark.asyncio
async def test_real_picker_verifies_off_loop_reports_bytes_and_commits_after_success(
    tmp_path, monkeypatch
):
    import threading

    from tldw_chatbook.Local_Ingestion.parakeet_v2_artifact import (
        parakeet_reference,
    )
    from tldw_chatbook.Local_Ingestion.stt_batch_routing import PARAKEET_V2_MODEL
    from tldw_chatbook.Third_Party.textual_fspicker import SelectDirectory
    from tldw_chatbook.Third_Party.textual_fspicker.parts import DirectoryNavigation
    from tldw_chatbook.UI.Screens.model_curated_view import CuratedView
    from tldw_chatbook.UI.Screens.model_external_view import ExternalModelView
    from tldw_chatbook.Utils import optional_deps

    monkeypatch.setattr(optional_deps, "parakeet_onnx_deps_installed", lambda: False)

    service = _FakeExternalSourceService(block_verification=True)
    app = _app()
    app._parakeet_source_service = service
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        screen.notify = MagicMock()
        assert await _wait_for(
            lambda: bool(screen.query("#external-models-view")), pilot
        )
        window = screen.query_one(LLMManagementWindow)
        curated_row = next(
            row for row in _rail_rows(screen) if row.lab_view_key == "curated"
        )
        curated_row.press()
        await pilot.pause()
        assert window.active_view == "curated"
        reference = parakeet_reference(PARAKEET_V2_MODEL, "int8")
        screen.post_message(CuratedView.UseFromDiskRequested(reference))
        assert await _wait_for(lambda: isinstance(app.screen, SelectDirectory), pilot)

        picker = app.screen
        picker.query_one(DirectoryNavigation).location = tmp_path
        picker.query_one("#select", Button).press()
        assert await _wait_for(service.progress_seen.is_set, pilot)

        external = screen.query_one(ExternalModelView)
        assert window.active_view == "external"
        assert "-active" in window.query_one("#llm-view-external").classes
        active_rail = [row for row in _rail_rows(screen) if "is-active" in row.classes]
        assert [row.lab_view_key for row in active_rail] == ["external"]
        status = external.query_one("#external-model-operation-status", Static)
        assert "4 / 8 bytes" in str(status.renderable)
        assert status.region.width > 0 and status.region.height > 0
        assert service.prepare_threads == [service.prepare_threads[0]]
        assert service.prepare_threads[0] != threading.get_ident()

        service.release_verification.set()
        assert await _wait_for(
            lambda: (
                len(service.committed) == 1
                and bool(statuses := screen.query("#external-model-operation-status"))
                and str(statuses.first(Static).renderable) == "Runtime required"
            ),
            pilot,
        )
        external = screen.query_one(ExternalModelView)
        assert service.commit_threads[0] != threading.get_ident()
        owner = service.prepare_calls[0][2]
        assert owner[0] == "scope"
        assert str(tmp_path) not in owner[1]
        assert service.released_scopes == [owner[1]]
        status = external.query_one("#external-model-operation-status", Static)
        assert str(status.renderable) == "Runtime required"
        assert all(
            str(tmp_path) not in str(call) for call in screen.notify.call_args_list
        )

        await screen.recompose()
        assert await _wait_for(
            lambda: (
                bool(screen.query("#external-model-operation-status"))
                and str(
                    screen.query_one(
                        "#external-model-operation-status", Static
                    ).renderable
                )
                == "Runtime required"
            ),
            pilot,
            attempts=1500,
        )
        status = screen.query_one("#external-model-operation-status", Static)
        assert str(status.renderable) == "Runtime required"


@pytest.mark.asyncio
async def test_stale_picker_result_and_cancel_leave_the_prior_source_unchanged(
    tmp_path, monkeypatch
):
    from unittest.mock import MagicMock

    from tldw_chatbook.Local_Ingestion.parakeet_v2_artifact import (
        parakeet_reference,
    )
    from tldw_chatbook.Local_Ingestion.stt_batch_routing import (
        PARAKEET_V2_MODEL,
        PARAKEET_V3_MODEL,
    )

    service = _FakeExternalSourceService()
    app = _app()
    app._parakeet_source_service = service
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        assert await _wait_for(
            lambda: bool(screen.query("#external-models-view")), pilot
        )
        window = screen.query_one(LLMManagementWindow)
        window.active_view = "curated"
        await pilot.pause()
        pushes = MagicMock()
        monkeypatch.setattr(app, "push_screen", pushes)

        screen._begin_external_selection(parakeet_reference(PARAKEET_V2_MODEL, "int8"))
        stale_callback = pushes.call_args.args[1]
        screen._begin_external_selection(parakeet_reference(PARAKEET_V3_MODEL, "f32"))
        current_callback = pushes.call_args.args[1]

        stale_callback(tmp_path)
        current_callback(None)
        await pilot.pause()

        assert service.prepare_calls == []
        assert service.commit_attempts == []
        assert window.active_view == "curated"


@pytest.mark.asyncio
async def test_replacing_external_verification_releases_its_path_free_scope(
    tmp_path,
):
    from tldw_chatbook.STT.parakeet_sources import ParakeetSourceKey

    service = _FakeExternalSourceService(block_verification=True)
    app = _app()
    app._parakeet_source_service = service
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        assert await _wait_for(
            lambda: bool(screen.query("#external-models-view")), pilot
        )
        token = screen._next_external_token()
        screen._external_directory_selected(
            token,
            ParakeetSourceKey.V2_INT8,
            tmp_path,
        )
        assert await _wait_for(service.progress_seen.is_set, pilot)
        owner = service.prepare_calls[0][2]

        screen._next_external_token()
        service.release_verification.set()
        await pilot.pause()

        assert owner[0] == "scope"
        assert str(tmp_path) not in owner[1]
        assert service.released_scopes == [owner[1]]


@pytest.mark.asyncio
async def test_stop_action_becomes_physical_cancel_during_external_work_at_80_columns(
    tmp_path,
):
    from tldw_chatbook.STT.parakeet_sources import (
        ParakeetSourceKey,
        ParakeetSourcePreference,
        ParakeetSourceRecord,
    )
    from tldw_chatbook.UI.Screens.model_external_view import ExternalModelView

    key = ParakeetSourceKey.V2_INT8
    root = (tmp_path / "configured").absolute()
    prior = ParakeetSourceRecord(
        model_id=key.model_id,
        precision=key.precision,
        directory=root,
        preferred_source=ParakeetSourcePreference.EXTERNAL,
    )
    service = _FakeExternalSourceService(
        records={key: prior},
        block_verification=True,
    )
    app = _app()
    app._parakeet_source_service = service
    async with app.run_test(size=(80, 24)) as pilot:
        screen = await _models_screen(app)
        assert await _wait_for(
            lambda: bool(screen.query("#external-models-view")), pilot
        )
        external_row = next(
            row for row in _rail_rows(screen) if row.lab_view_key == "external"
        )
        external_row.press()
        await pilot.pause()

        await pilot.click(f"#external-model-copy-{key.value}")
        assert await _wait_for(service.progress_seen.is_set, pilot)
        cancel = screen.query_one(f"#external-model-stop-{key.value}", Button)
        painted = "".join(
            cancel.render_line(line).text for line in range(cancel.region.height)
        )
        parent = screen.query_one("#llm-view-external")
        assert "Cancel operation" in painted
        assert cancel.region.width > 0 and cancel.region.height > 0
        assert parent.region.x <= cancel.region.x
        assert (
            cancel.region.x + cancel.region.width
            <= parent.region.x + parent.region.width
        )
        cancel.focus()
        await pilot.pause()
        assert app.focused is cancel

        await pilot.click(f"#external-model-stop-{key.value}")
        assert await _wait_for(
            lambda: "cancelled" in screen._external_operation_status.casefold(), pilot
        )
        cancelled_status = screen._external_operation_status
        service.release_verification.set()
        await pilot.pause()
        await pilot.pause()

        restored = screen.query_one(f"#external-model-stop-{key.value}", Button)
        assert "Stop using" in str(restored.label)
        assert screen._external_operation_status == cancelled_status
        assert service.records()[key] == prior
        assert service.stopped == []
        assert service.commit_attempts == []
        assert service.copy_plans == []
        assert screen.query_one(ExternalModelView).is_mounted


@pytest.mark.asyncio
async def test_first_use_shows_one_physical_cancel_and_returns_to_empty_idle(
    tmp_path,
):
    from tldw_chatbook.STT.parakeet_sources import ParakeetSourceKey

    service = _FakeExternalSourceService(block_verification=True)
    app = _app()
    app._parakeet_source_service = service
    async with app.run_test(size=(80, 24)) as pilot:
        screen = await _models_screen(app)
        assert await _wait_for(
            lambda: bool(screen.query("#external-models-view")), pilot
        )
        token = screen._next_external_token()
        screen._external_directory_selected(
            token,
            ParakeetSourceKey.V2_INT8,
            tmp_path,
        )
        assert await _wait_for(service.progress_seen.is_set, pilot)
        await pilot.pause()

        cancel_buttons = screen.query("#external-model-cancel-operation")
        assert len(cancel_buttons) == 1
        cancel = cancel_buttons.first(Button)
        parent = screen.query_one("#llm-view-external")
        assert str(cancel.label) == "Cancel operation"
        assert cancel.region.width > 0 and cancel.region.height > 0
        assert parent.region.x <= cancel.region.x
        assert (
            cancel.region.x + cancel.region.width
            <= parent.region.x + parent.region.width
        )
        cancel.focus()
        await pilot.pause()
        assert app.focused is cancel

        await pilot.click("#external-model-cancel-operation")
        assert await _wait_for(
            lambda: "cancelled" in screen._external_operation_status.casefold(), pilot
        )
        service.release_verification.set()
        await pilot.pause()
        await pilot.pause()

        assert service.records() == {}
        assert service.commit_attempts == []
        assert len(screen.query("#external-model-cancel-operation")) == 0


@pytest.mark.asyncio
async def test_failed_external_commit_releases_scope_and_preserves_prior_state(
    tmp_path,
):
    from tldw_chatbook.STT.parakeet_sources import ParakeetSourceKey

    service = _FakeExternalSourceService()

    def fail_commit(prepared, *, cancelled=lambda: False):
        if cancelled():
            return
        service.commit_attempts.append(prepared)
        raise RuntimeError("private commit detail")

    service.commit_external = fail_commit
    app = _app()
    app._parakeet_source_service = service
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        screen.notify = MagicMock()
        assert await _wait_for(
            lambda: bool(screen.query("#external-models-view")), pilot
        )
        token = screen._next_external_token()
        screen._external_directory_selected(
            token,
            ParakeetSourceKey.V2_INT8,
            tmp_path,
        )
        assert await _wait_for(lambda: bool(service.commit_attempts), pilot)
        assert await _wait_for(lambda: bool(service.released_scopes), pilot)

        owner = service.prepare_calls[0][2]
        assert service.committed == []
        assert service.released_scopes == [owner[1]]
        assert "prior source is unchanged" in screen._external_operation_status


@pytest.mark.asyncio
async def test_replacement_keeps_commit_scope_until_point_of_no_return_finishes(
    tmp_path,
):
    import threading

    from tldw_chatbook.STT.parakeet_sources import ParakeetSourceKey

    started = threading.Event()
    release = threading.Event()
    timeline: list[str] = []
    service = _FakeExternalSourceService()
    real_release_scope = service.release_scope

    def blocked_commit(prepared, *, cancelled=lambda: False):
        service.commit_attempts.append(prepared)
        started.set()
        assert release.wait(3)
        service.committed.append(prepared)
        timeline.append("promoted")

    def release_scope(scope_id):
        timeline.append("released")
        real_release_scope(scope_id)

    service.commit_external = blocked_commit
    service.release_scope = release_scope
    app = _app()
    app._parakeet_source_service = service
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        assert await _wait_for(
            lambda: bool(screen.query("#external-models-view")), pilot
        )
        old_token = screen._next_external_token()
        screen._external_directory_selected(
            old_token,
            ParakeetSourceKey.V2_INT8,
            tmp_path,
        )
        assert await _wait_for(started.is_set, pilot)
        old_owner = service.prepare_calls[0][2][1]

        screen._next_external_token()
        screen._set_external_status("Verifying newer model files…", active=True)
        await pilot.pause()

        assert service.released_scopes == []
        release.set()
        assert await _wait_for(lambda: old_owner in service.released_scopes, pilot)

        assert timeline == ["promoted", "released"]
        assert service.committed == [service.commit_attempts[0]]
        assert screen._external_operation_status == "Verifying newer model files…"


@pytest.mark.asyncio
async def test_replaced_commit_releases_scope_even_when_worker_never_enters_service():
    from types import SimpleNamespace

    service = _FakeExternalSourceService()
    app = _app()
    app._parakeet_source_service = service
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        assert await _wait_for(
            lambda: bool(screen.query("#external-models-view")), pilot
        )
        stale = screen._next_external_token()
        stale_scope = screen._external_scope_id
        screen._external_commit_tokens.add(stale)
        screen._next_external_token()

        screen._run_external_commit(stale, SimpleNamespace())
        released = await _wait_for(
            lambda: stale_scope in service.released_scopes,
            pilot,
        )

        assert released is True
        assert service.commit_attempts == []


@pytest.mark.asyncio
async def test_external_workers_keep_paths_out_of_descriptions_and_logs(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from loguru import logger as loguru_logger
    from textual.worker import WorkerState

    from tldw_chatbook.Local_Ingestion import parakeet_v2_artifact as artifact
    from tldw_chatbook.STT.parakeet_sources import ParakeetSourceKey
    from tldw_chatbook.UI.Screens.model_external_view import ExternalModelView

    selected = (tmp_path / "worker-description-must-not-leak").absolute()
    prepared = SimpleNamespace(
        directory=selected,
        verified=SimpleNamespace(directory=selected, reference=object()),
    )
    report = SimpleNamespace(destination=selected)
    consent = SimpleNamespace(destination=selected)

    async def fake_preflight(**_kwargs):
        return report

    async def fake_provision(_report, **_kwargs):
        return selected

    monkeypatch.setattr(artifact, "run_parakeet_vad_preflight", fake_preflight)
    monkeypatch.setattr(artifact, "run_parakeet_vad_provision", fake_provision)
    service = _FakeExternalSourceService(block_verification=True)
    app = _app()
    app._parakeet_source_service = service
    messages: list[str] = []

    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        assert await _wait_for(
            lambda: bool(screen.query("#external-models-view")), pilot
        )
        sink_id = loguru_logger.add(
            lambda message: messages.append(str(message)),
            level="DEBUG",
            format="{message}",
        )
        try:
            token = screen._next_external_token()
            screen._external_directory_selected(
                token,
                ParakeetSourceKey.V2_INT8,
                selected,
            )
            verify_worker = screen._external_selection_worker
            assert verify_worker is not None
            stale_token = (token[0] - 1, token[1])
            workers = (
                verify_worker,
                screen._run_external_commit(stale_token, prepared),
                screen._run_external_vad_preflight(stale_token, prepared),
                screen._run_external_vad_provision(
                    stale_token,
                    prepared,
                    report,
                ),
                screen._run_external_copy(stale_token, prepared, consent),
                screen._run_external_stop(
                    stale_token,
                    ParakeetSourceKey.V2_INT8,
                ),
            )
            assert await _wait_for(service.progress_seen.is_set, pilot)
            await pilot.pause()
            assert verify_worker.state is WorkerState.RUNNING

            assert [worker.description for worker in workers] == [
                "Verify external Parakeet source",
                "Save external Parakeet source",
                "Check managed VAD dependency",
                "Install managed VAD dependency",
                "Copy external Parakeet source",
                "Stop using external Parakeet source",
            ]
            log_output = "\n".join(messages)
            assert str(selected) not in log_output
            status = screen.query_one(ExternalModelView).query_one(
                "#external-model-operation-status", Static
            )
            assert "Verifying model files" in str(status.renderable)
            assert status.region.width > 0 and status.region.height > 0
        finally:
            service.release_verification.set()
            loguru_logger.remove(sink_id)


@pytest.mark.asyncio
async def test_unmount_cancels_vad_provision_before_it_can_continue(
    monkeypatch,
):
    import asyncio
    import threading
    from types import SimpleNamespace

    from tldw_chatbook.Local_Ingestion import parakeet_v2_artifact as artifact

    started = threading.Event()
    cancelled = threading.Event()
    release = threading.Event()
    continued = threading.Event()

    async def fake_provision(_report, **_kwargs):
        started.set()
        try:
            while not release.is_set():
                await asyncio.sleep(0.005)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        continued.set()

    monkeypatch.setattr(artifact, "run_parakeet_vad_provision", fake_provision)
    app = _app()
    app._parakeet_source_service = _FakeExternalSourceService()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        assert await _wait_for(
            lambda: bool(screen.query("#external-models-view")), pilot
        )
        token = screen._next_external_token()
        prepared = SimpleNamespace()
        report = SimpleNamespace()
        screen._external_selection_worker = screen._run_external_vad_provision(
            token,
            prepared,
            report,
        )
        assert await _wait_for(started.is_set, pilot)

        await app.pop_screen()
        was_cancelled = await _wait_for(cancelled.is_set, pilot)
        release.set()

        assert was_cancelled is True
        assert continued.is_set() is False


@pytest.mark.asyncio
async def test_replacing_external_copy_cooperatively_stops_its_side_effect():
    from types import SimpleNamespace

    service = _FakeExternalSourceService(block_copy=True)
    app = _app()
    app._parakeet_source_service = service
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        assert await _wait_for(
            lambda: bool(screen.query("#external-models-view")), pilot
        )
        token = screen._next_external_token()
        prepared = SimpleNamespace(verified=SimpleNamespace())
        consent = SimpleNamespace()
        screen._external_selection_worker = screen._run_external_copy(
            token,
            prepared,
            consent,
        )
        assert await _wait_for(service.copy_started.is_set, pilot)

        screen._next_external_token()

        was_cancelled = await _wait_for(service.copy_cancelled.is_set, pilot)
        service.release_copy.set()

        assert was_cancelled is True
        assert service.copy_continued.is_set() is False
        assert service.copied == []


@pytest.mark.asyncio
async def test_external_copy_planning_keeps_cancel_responsive_off_loop(tmp_path):
    import asyncio
    import threading
    import time

    from tldw_chatbook.Local_Ingestion.stt_batch_routing import PARAKEET_V2_MODEL
    from tldw_chatbook.STT.parakeet_sources import (
        ParakeetSourceKey,
        ParakeetSourcePreference,
        ParakeetSourceRecord,
    )
    from tldw_chatbook.UI.Screens.model_external_view import ExternalModelView
    from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

    key = ParakeetSourceKey.V2_INT8
    root = tmp_path / "external-root"
    root.mkdir()
    service = _FakeExternalSourceService(
        records={
            key: ParakeetSourceRecord(
                model_id=PARAKEET_V2_MODEL,
                precision="int8",
                directory=root,
                preferred_source=ParakeetSourcePreference.EXTERNAL,
            )
        },
        block_plan=True,
    )
    app = _app()
    app._parakeet_source_service = service

    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        assert await _wait_for(
            lambda: bool(screen.query("#external-models-view")), pilot
        )
        ui_thread = threading.get_ident()
        owns_external_token = screen._owns_external_token

        def ui_owned_token(candidate):
            assert threading.get_ident() == ui_thread
            return owns_external_token(candidate)

        screen._owns_external_token = ui_owned_token

        async def cancel_during_plan() -> float:
            started = await asyncio.to_thread(service.plan_started.wait, 2)
            assert started is True
            screen.post_message(ExternalModelView.CancelRequested())
            assert await _wait_for(
                lambda: "cancelled" in screen._external_operation_status.casefold(),
                pilot,
            )
            return time.monotonic()

        release = threading.Timer(0.5, service.release_plan.set)
        release.start()
        try:
            cancel_task = asyncio.create_task(cancel_during_plan())
            screen.post_message(ExternalModelView.CopyRequested(key))
            cancelled_at = await cancel_task
            assert await asyncio.to_thread(service.plan_returned.wait, 2)
        finally:
            service.release_plan.set()
            release.cancel()

        assert cancelled_at < service.plan_returned_at
        assert not isinstance(app.screen, ConfirmationDialog)
        assert service.copied == []


@pytest.mark.asyncio
async def test_missing_vad_shows_vad_only_consent_and_commits_only_after_provision(
    tmp_path, monkeypatch
):
    from dataclasses import replace
    from types import SimpleNamespace

    from tldw_chatbook.Local_Ingestion import parakeet_v2_artifact as artifact
    from tldw_chatbook.Local_Ingestion.parakeet_v2_artifact import (
        parakeet_reference,
        parakeet_vad_descriptor,
        parakeet_vad_reference,
    )
    from tldw_chatbook.Local_Ingestion.stt_batch_routing import PARAKEET_V2_MODEL
    from tldw_chatbook.Model_Artifacts.acquisition import (
        ArtifactPreflightEntry,
        PreflightReport,
    )
    from tldw_chatbook.Third_Party.textual_fspicker import SelectDirectory
    from tldw_chatbook.Third_Party.textual_fspicker.parts import DirectoryNavigation
    from tldw_chatbook.UI.Screens.model_curated_view import CuratedView
    from tldw_chatbook.Widgets.ModelArtifacts import ModelInstallModal

    service = _FakeExternalSourceService()
    service.vad_ready = False
    descriptor = parakeet_vad_descriptor()
    report = PreflightReport(
        root=parakeet_vad_reference(),
        closure_fingerprint="f" * 64,
        entries=(
            ArtifactPreflightEntry(
                ref=descriptor.reference,
                source_url=descriptor.source_url,
                repository=descriptor.upstream_repository,
                revision=descriptor.upstream_revision,
                license_id=descriptor.license_id,
                license_url=descriptor.license_url,
                precision=descriptor.precision,
                total_bytes=descriptor.expected_installed_bytes,
                file_count=len(descriptor.files),
                already_installed=False,
                provenance=descriptor.provenance,
            ),
        ),
        download_bytes=descriptor.expected_installed_bytes,
        already_staged_bytes=0,
        staging_overhead_bytes=0,
        retained_bytes=0,
        destination=tmp_path / "managed-vad",
        free_bytes=descriptor.expected_installed_bytes + 1,
        required_bytes=descriptor.expected_installed_bytes,
        sufficient_space=True,
        gating_errors=(),
    )
    import asyncio

    from tldw_chatbook.Model_Artifacts.acquisition import AcquisitionProgress

    calls = []
    allow_provision = asyncio.Event()

    async def fake_preflight(**_kwargs):
        calls.append("preflight")
        return report

    async def fake_provision(received, *, progress, **_kwargs):
        calls.append(("provision", received))
        progress(
            AcquisitionProgress(
                "fetch",
                parakeet_vad_reference(),
                "silero_vad.onnx",
                4,
                8,
            )
        )
        await allow_provision.wait()
        service.vad_ready = True
        return tmp_path / "managed-vad"

    monkeypatch.setattr(artifact, "run_parakeet_vad_preflight", fake_preflight)
    monkeypatch.setattr(artifact, "run_parakeet_vad_provision", fake_provision)
    app = _app()
    app._parakeet_source_service = service

    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        assert await _wait_for(
            lambda: bool(screen.query("#external-models-view")), pilot
        )
        reference = parakeet_reference(PARAKEET_V2_MODEL, "int8")
        bad_report = replace(
            report,
            entries=(
                replace(
                    report.entries[0],
                    source_url=(
                        "https://huggingface.co/nvidia/"
                        "parakeet-tdt-0.6b-v2/resolve/main/model.onnx"
                    ),
                ),
            ),
        )
        bad_token = screen._next_external_token()
        bad_scope = screen._external_scope_id
        screen._apply_external_vad_preflight_result(
            bad_token,
            SimpleNamespace(),
            bad_report,
            None,
        )
        await pilot.pause()
        assert not isinstance(app.screen, ModelInstallModal)
        assert service.released_scopes == [bad_scope]

        screen.post_message(CuratedView.UseFromDiskRequested(reference))
        assert await _wait_for(lambda: isinstance(app.screen, SelectDirectory), pilot)
        picker = app.screen
        picker.query_one(DirectoryNavigation).location = tmp_path
        picker.query_one("#select", Button).press()
        assert await _wait_for(
            lambda: (
                isinstance(app.screen, ModelInstallModal)
                and bool(app.screen.query("#model-install-cancel"))
            ),
            pilot,
        )

        modal = app.screen
        assert await _wait_for(
            lambda: bool(modal.query("#model-install-cancel")), pilot
        )
        assert modal.report.root == parakeet_vad_reference()
        assert {entry.ref for entry in modal.report.entries} == {
            parakeet_vad_reference()
        }
        plan_text = "\n".join(str(item.renderable) for item in modal.query(Static))
        assert "parakeet-tdt" not in plan_text.lower()
        assert "nvidia/parakeet" not in plan_text.lower()
        modal.query_one("#model-install-cancel", Button).press()
        await pilot.pause()
        assert service.committed == []
        first_owner = service.prepare_calls[0][2]
        assert service.released_scopes == [bad_scope, first_owner[1]]

        screen.post_message(CuratedView.UseFromDiskRequested(reference))
        assert await _wait_for(lambda: isinstance(app.screen, SelectDirectory), pilot)
        picker = app.screen
        picker.query_one(DirectoryNavigation).location = tmp_path
        picker.query_one("#select", Button).press()
        assert await _wait_for(
            lambda: (
                isinstance(app.screen, ModelInstallModal)
                and bool(app.screen.query("#model-install-confirm"))
            ),
            pilot,
        )
        app.screen.query_one("#model-install-confirm", Button).press()
        assert await _wait_for(
            lambda: "4 / 8 bytes" in screen._external_operation_status,
            pilot,
        )
        assert screen._external_operation_status.startswith(
            "Installing managed VAD dependency"
        )
        allow_provision.set()
        assert await _wait_for(lambda: len(service.committed) == 1, pilot)
        second_owner = service.prepare_calls[1][2]
        assert service.released_scopes == [
            bad_scope,
            first_owner[1],
            second_owner[1],
        ]

    assert calls.count("preflight") == 2
    assert calls[-1] == ("provision", report)


@pytest.mark.asyncio
async def test_external_copy_uses_task6_plan_and_stop_uses_the_shared_service(
    tmp_path,
):
    from tldw_chatbook.Local_Ingestion.stt_batch_routing import PARAKEET_V2_MODEL
    from tldw_chatbook.STT.parakeet_sources import (
        ParakeetSourceKey,
        ParakeetSourcePreference,
        ParakeetSourceRecord,
    )
    from tldw_chatbook.UI.Screens.model_external_view import ExternalModelView
    from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

    root = tmp_path / "external-root"
    root.mkdir()
    key = ParakeetSourceKey.V2_INT8
    service = _FakeExternalSourceService(
        records={
            key: ParakeetSourceRecord(
                model_id=PARAKEET_V2_MODEL,
                precision="int8",
                directory=root,
                preferred_source=ParakeetSourcePreference.EXTERNAL,
            )
        }
    )
    app = _app()
    app._parakeet_source_service = service

    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        screen.notify = MagicMock()
        assert await _wait_for(
            lambda: bool(screen.query("#external-models-view")), pilot
        )
        screen.post_message(ExternalModelView.CopyRequested(key))
        assert await _wait_for(
            lambda: isinstance(app.screen, ConfirmationDialog), pilot
        )
        dialog = app.screen
        assert str(root) not in dialog.message
        assert "1.0 KiB" in dialog.message
        assert await _wait_for(lambda: bool(dialog.query("#confirm-button")), pilot)
        dialog.query_one("#confirm-button", Button).press()
        assert await _wait_for(lambda: len(service.copied) == 1, pilot)

        screen.post_message(ExternalModelView.StopRequested(key))
        assert await _wait_for(lambda: service.stopped == [key], pilot)
        assert service.stopped == [key]
        import threading

        assert service.stop_threads[0] != threading.get_ident()
        assert all(str(root) not in str(call) for call in screen.notify.call_args_list)


def _task6_install_request(kind: str):
    """Build one valid request for the real curated/remote screen handlers."""
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef
    from tldw_chatbook.UI.Screens.model_curated_view import CuratedView
    from tldw_chatbook.UI.Screens.model_remote_view import RemoteView

    if kind == "curated":
        return CuratedView.InstallRequested(
            ArtifactRef("task6-curated", "c" * 40, "int8"),
            service=MagicMock(),
            registry=MagicMock(),
            sources={},
        )
    catalog = _remote_catalog()
    return RemoteView.InstallRequested(
        catalog,
        _resolved_remote_model().candidates[0],
        service=MagicMock(),
        credential_resolver=MagicMock(),
    )


async def _task6_mounted_host(app, pilot):
    """Return the real mounted screen/window/InstalledView host chain."""
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    screen = await _models_screen(app)
    assert await _wait_for(
        lambda: bool(screen.query("#installed-models-view")),
        pilot,
    )
    window = screen.query_one(LLMManagementWindow)
    installed = window.query_one("#installed-models-view", InstalledView)
    return screen, window, installed


def _task6_capture_screens(app):
    pushed = []
    app.push_screen = MagicMock(
        side_effect=lambda modal, callback=None: pushed.append((modal, callback))
    )
    return pushed


def _task6_stub_preflights(screen):
    screen._run_curated_preflight = MagicMock(return_value=MagicMock())
    screen._run_remote_preflight = MagicMock(return_value=MagicMock())


def _task6_show_unmanaged_row(installed, source):
    """Render one real external-GGUF row without starting inventory I/O."""
    from tldw_chatbook.Model_Artifacts.service import ArtifactDiskUsage
    from tldw_chatbook.UI.Screens.model_browser_state import (
        UnmanagedRow,
        inventory_rows,
    )

    installed._loaded = True
    installed._rows = inventory_rows(
        (),
        ArtifactDiskUsage(0, 0, 64 * 1024 * 1024),
        (UnmanagedRow(source, source.stat().st_size),),
    )
    installed.refresh(recompose=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "button_selector"),
    (
        ("curated", "#installed-models-import-gguf"),
        ("remote", ".model-import"),
    ),
)
async def test_local_selection_lane_refuses_host_installs_until_declined(
    tmp_path,
    monkeypatch,
    kind,
    button_selector,
):
    """Header and row imports own the host before either consent starts."""
    from tldw_chatbook.UI.Screens import llm_screen as llm_screen_module

    source = tmp_path / "private-outside.gguf"
    source.write_bytes(b"gguf")
    app = _app()
    pushed = []
    fake_logger = MagicMock()
    monkeypatch.setattr(llm_screen_module, "logger", fake_logger)

    async with app.run_test(size=(120, 40)) as pilot:
        screen, window, installed = await _task6_mounted_host(app, pilot)
        _task6_show_unmanaged_row(installed, source)
        window.active_view = "installed"
        await pilot.pause()
        pushed = _task6_capture_screens(app)
        screen.notify = MagicMock()
        _task6_stub_preflights(screen)

        installed.query_one(button_selector, Button).press()
        await pilot.pause()
        assert len(pushed) == 1
        picker_callback = pushed[-1][1]
        picker_callback(source)
        await pilot.pause()
        assert len(pushed) == 2
        consent_callback = pushed[-1][1]
        assert installed._import_selecting is screen._local_gguf_import_active is True

        screen.post_message(_task6_install_request(kind))
        await pilot.pause()

        runner = getattr(screen, f"_run_{kind}_preflight")
        runner.assert_not_called()
        assert screen._model_install_kind is None
        assert installed._import_selecting is True
        assert installed._pending_import_path == source
        consent_callback(False)
        await pilot.pause()
        assert installed._import_selecting is False
        assert installed._pending_import_path is None
        assert screen._local_gguf_import_active is False

        screen.post_message(_task6_install_request(kind))
        await pilot.pause()
        runner.assert_called_once_with()
        assert screen._model_install_kind == kind
        assert str(source) not in str(screen.notify.call_args_list)
        assert str(source) not in str(fake_logger.mock_calls)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "phase"),
    (("curated", "preflight"), ("remote", "pending-consent")),
)
async def test_host_install_ownership_refuses_local_picker(
    tmp_path,
    kind,
    phase,
):
    """Curated preflight and remote consent both block physical Import."""
    source = tmp_path / "private-never-selected.gguf"
    source.write_bytes(b"gguf")
    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen, window, installed = await _task6_mounted_host(app, pilot)
        _task6_show_unmanaged_row(installed, source)
        window.active_view = "installed"
        await pilot.pause()
        pushed = _task6_capture_screens(app)
        installed.notify = MagicMock()
        _task6_stub_preflights(screen)
        request = _task6_install_request(kind)
        screen.post_message(request)
        await pilot.pause()
        assert screen._model_install_kind == kind

        if phase == "pending-consent":
            report = _remote_report_for(request.catalog, tmp_path / "managed")
            screen._apply_remote_preflight_result(report, None)
            assert len(pushed) == 1

        pushed_before_import = len(pushed)
        installed.query_one("#installed-models-import-gguf", Button).press()
        await pilot.pause()

        assert len(pushed) == pushed_before_import
        assert installed._import_selecting is False
        assert installed._pending_import_path is None
        installed.notify.assert_called_once()
        assert str(source) not in str(installed.notify.call_args_list)


class _Task6HostImportService:
    def __init__(self, root, *, fail=False):
        from tldw_chatbook.Model_Artifacts.service import ArtifactRef

        self.artifacts_path = root / "managed" / "artifacts"
        self.reference = ArtifactRef("task6-local", "d" * 40, "filetype-7")
        self.fail = fail
        self.entered = threading.Event()
        self.release = threading.Event()
        self.activation_calls = 0

    def import_local_gguf(self, source_file, *, cancelled, progress):
        from tldw_chatbook.Model_Artifacts.service import LocalGGUFImportResult

        self.entered.set()
        assert self.release.wait(timeout=3.0)
        if self.fail:
            raise RuntimeError("private failure")
        return LocalGGUFImportResult(self.reference, False)

    def activate(self, root_reference):
        self.activation_calls += 1
        return root_reference

    def list_installed(self):
        return ()

    def disk_usage(self):
        from tldw_chatbook.Model_Artifacts.service import ArtifactDiskUsage

        return ArtifactDiskUsage(0, 0, 64 * 1024 * 1024)


# Windows Proactor event-loop setup owns an internal loopback socket pair.
@pytest.mark.allow_network
@pytest.mark.asyncio
@pytest.mark.parametrize("fail", (False, True), ids=("success", "failure"))
async def test_local_import_terminal_releases_host_ownership(tmp_path, fail):
    """Success and failure free the host only after the worker settles."""
    source = tmp_path / "outside.gguf"
    source.write_bytes(b"gguf")
    service = _Task6HostImportService(tmp_path, fail=fail)
    app = _app()

    async with app.run_test(size=(120, 40)) as pilot:
        screen, _window, installed = await _task6_mounted_host(app, pilot)
        installed._service_factory = lambda: service
        installed._service = None
        screen._run_curated_preflight = MagicMock(return_value=MagicMock())

        installed._begin_import(source)
        assert await _wait_for(service.entered.is_set, pilot)
        try:
            assert screen._local_gguf_import_active is True
        finally:
            service.release.set()
        assert await _wait_for(lambda: not installed._import_active, pilot)
        assert screen._local_gguf_import_active is False

        screen.post_message(_task6_install_request("curated"))
        await pilot.pause()
        screen._run_curated_preflight.assert_called_once_with()


@pytest.mark.asyncio
async def test_model_library_view_switch_restores_keyboard_focus_at_80x24(
    monkeypatch,
):
    """A hidden model-library pane must not retain the live keyboard focus."""
    from tldw_chatbook.UI.Screens.model_curated_view import CuratedView
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    monkeypatch.setattr(CuratedView, "ensure_loaded", lambda self: None)
    monkeypatch.setattr(InstalledView, "ensure_loaded", lambda self: None)
    app = _app()

    async with app.run_test(size=(80, 24)) as pilot:
        screen = await _models_screen(app, populate_all=False)
        await pilot.pause()
        await pilot.pause()
        window = screen.query_one(LLMManagementWindow)
        window.active_view = "installed"
        assert await _wait_for(
            lambda: bool(screen.query("#installed-models-repair")),
            pilot,
        )

        window.active_view = "installed"
        await pilot.pause()
        repair = window.query_one("#installed-models-repair", Button)
        repair.focus()
        await pilot.pause()
        assert app.focused is repair

        window.active_view = "curated"
        await pilot.pause()
        assert app.focused is window.query_one("#curated-models-refresh", Button)

        window.active_view = "installed"
        await pilot.pause()
        assert app.focused is window.query_one("#installed-models-repair", Button)


@pytest.mark.asyncio
async def test_real_model_library_projects_exact_settings_and_running_evidence(
    tmp_path,
    monkeypatch,
):
    """Only exact saved/draft and applied-supervisor evidence promotes axes."""
    from tldw_chatbook.Model_Artifacts.curated_registry import CuratedRegistry
    from dataclasses import replace

    import tldw_chatbook.TTS.audio_cpp_recipes as recipes_module
    from textual.widgets._collapsible import CollapsibleTitle

    from tldw_chatbook.Model_Artifacts.service import (
        ArtifactDiskUsage,
        InstalledArtifact,
        ModelArtifactService,
    )
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import audio_cpp_curated_entries
    from tldw_chatbook.TTS.audio_cpp_artifact_dependencies import (
        AudioCppArtifactRemovalEvidence,
        AudioCppModelLibraryObservationSnapshot,
    )
    from tldw_chatbook.UI.Screens.model_curated_view import CuratedView
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    descriptor, sources = audio_cpp_curated_entries()[0]
    recipe = next(
        item
        for item in recipes_module.AUDIO_CPP_RECIPE_REGISTRY.recipes
        if descriptor.reference.artifact_id in item.model_library_artifact_ids
    )
    signal = recipe.required_files[0]
    companions = tuple(
        replace(signal, relative_path=f"companions/review-file-{index:02d}.json")
        for index in range(12)
    )
    monkeypatch.setattr(
        recipes_module,
        "AUDIO_CPP_RECIPE_REGISTRY",
        type(
            "Registry",
            (),
            {
                "recipes": (
                    replace(recipe, required_files=recipe.required_files + companions),
                )
            },
        )(),
    )
    descriptor = replace(
        descriptor,
        model_id="audio-cpp/" + "very-long-reviewed-model-" * 5,
        model_family="very-long-family-" * 5,
    )
    registry = CuratedRegistry()
    registry.register(descriptor, sources=sources)

    observation_calls = []

    async def evidence(exact_references):
        observation_calls.append(exact_references)
        assert exact_references == (descriptor.reference,)
        return AudioCppModelLibraryObservationSnapshot(
            (
                AudioCppArtifactRemovalEvidence(
                    descriptor.reference,
                    settings_consumers=(
                        ("saved", "Guided Settings", "saved-package"),
                        ("draft", "Unsaved Guided Settings", "draft-package"),
                    ),
                    live_runtime_ids=("process-generation-7",),
                ),
            )
        )

    app = _app()
    monkeypatch.setattr(app, "_audio_cpp_model_library_observation_snapshot", evidence)
    async with app.run_test(size=(80, 24)) as pilot:
        screen = await _models_screen(app)
        assert await _wait_for(
            lambda: bool(screen.query("#curated-models-view")),
            pilot,
        )
        window = screen.query_one(LLMManagementWindow)
        curated = window.query_one("#curated-models-view", CuratedView)
        curated._service_factory = lambda: ModelArtifactService(tmp_path / "store")
        curated._registry_factory = lambda: registry
        curated._service = None
        curated._registry = None
        curated.set_consumer_filter("audio_cpp")
        window.active_view = "curated"
        assert await _wait_for(
            lambda: (
                "Configured: Saved Settings + detached draft"
                in "\n".join(str(item.renderable) for item in curated.query(Static))
            ),
            pilot,
        )
        text = "\n".join(str(item.renderable) for item in curated.query(Static))
        assert "Running: Applied supervisor generation is active" in text
        assert "Configured: Unknown" not in text
        assert "Running: Unknown" not in text
        assert observation_calls == [(descriptor.reference,)]

        pane = window.query_one("#llm-view-curated")
        assert pane.region.width < 80
        refresh = curated.query_one("#curated-models-refresh", Button)
        refresh.focus()
        await pilot.press("tab")
        disclosure_title = curated.query_one(CollapsibleTitle)
        assert disclosure_title.has_focus
        await pilot.press("enter", "tab")
        install = curated.query_one(".curated-install", Button)
        assert install.has_focus
        assert install in app.screen._compositor.visible_widgets
        assert install.region.right <= pane.region.right
        assert install.region.bottom <= pane.region.bottom

        window.active_view = "installed"
        await pilot.pause()
        window.active_view = "curated"
        await pilot.pause()
        assert curated.query_one(".curated-install", Button).has_focus
        assert await _wait_for(lambda: len(observation_calls) == 2, pilot)
        from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
            STTSProviderConfigurationChanged,
        )

        app.handle_stts_provider_configuration_changed(
            STTSProviderConfigurationChanged("audio_cpp", 7)
        )
        assert await _wait_for(lambda: len(observation_calls) == 3, pilot)

        class InstalledService:
            def list_installed(self):
                return (
                    InstalledArtifact(
                        path=tmp_path / "managed",
                        descriptor=descriptor,
                        ready=False,
                        active=False,
                        error=None,
                    ),
                )

            def disk_usage(self):
                return ArtifactDiskUsage(1, 0, 64 * 1024 * 1024)

        installed_view = window.query_one("#installed-models-view", InstalledView)
        installed_view._service_factory = InstalledService
        installed_view._service = None
        window.active_view = "installed"
        installed_view.ensure_loaded(force=True)
        assert await _wait_for(
            lambda: bool(installed_view.query(CollapsibleTitle)), pilot
        )
        installed_title = installed_view.query_one(CollapsibleTitle)
        installed_title.focus()
        await pilot.press("enter", "tab")
        delete = installed_view.query_one(".model-delete", Button)
        assert delete.has_focus
        assert delete in app.screen._compositor.visible_widgets
        assert (
            delete.region.right <= window.query_one("#llm-view-installed").region.right
        )
        window.active_view = "curated"
        await pilot.pause()
        window.active_view = "installed"
        await pilot.pause()
        assert installed_view.query_one(".model-delete", Button).has_focus


# Windows Proactor event-loop setup owns an internal loopback socket pair.
@pytest.mark.allow_network
@pytest.mark.asyncio
async def test_selection_unmount_releases_host_ownership(tmp_path):
    """A picker-only lane has no worker and releases immediately on teardown."""
    app = _app()

    async with app.run_test(size=(120, 40)) as pilot:
        screen, _window, installed = await _task6_mounted_host(app, pilot)
        app.push_screen = MagicMock()
        screen._run_curated_preflight = MagicMock(return_value=MagicMock())
        installed.query_one("#installed-models-import-gguf", Button).press()
        await pilot.pause()
        assert screen._local_gguf_import_active is True

        await installed.remove()
        assert screen._local_gguf_import_active is False
        screen.post_message(_task6_install_request("curated"))
        await pilot.pause()
        screen._run_curated_preflight.assert_called_once_with()


# Windows Proactor event-loop setup owns an internal loopback socket pair.
@pytest.mark.allow_network
@pytest.mark.asyncio
async def test_active_import_unmount_keeps_host_owned_until_worker_stops(tmp_path):
    """A remounted window cannot steal ownership from a detached live worker."""
    source = tmp_path / "outside.gguf"
    source.write_bytes(b"gguf")
    service = _Task6HostImportService(tmp_path)
    app = _app()

    async with app.run_test(size=(120, 40)) as pilot:
        screen, _window, installed = await _task6_mounted_host(app, pilot)
        installed._service_factory = lambda: service
        installed._service = None
        screen.notify = MagicMock()
        screen._run_curated_preflight = MagicMock(return_value=MagicMock())
        worker_stopped = MagicMock(wraps=installed._import_worker_stopped)
        installed._import_worker_stopped = worker_stopped
        installed._begin_import(source)
        assert await _wait_for(service.entered.is_set, pilot)
        try:
            assert screen._local_gguf_import_active is True
        except BaseException:
            service.release.set()
            raise

        try:
            await installed.remove()
            assert screen._local_gguf_import_active is True
            screen.post_message(_task6_install_request("curated"))
            await pilot.pause()
            screen._run_curated_preflight.assert_not_called()
        finally:
            service.release.set()
        assert await _wait_for(lambda: bool(worker_stopped.call_args_list), pilot)
        assert await _wait_for(
            lambda: screen._local_gguf_import_active is False,
            pilot,
        )
        assert service.activation_calls == 0
        screen.post_message(_task6_install_request("curated"))
        await pilot.pause()
        screen._run_curated_preflight.assert_called_once_with()
        assert str(source) not in str(screen.notify.call_args_list)


# Windows Proactor event-loop setup owns an internal loopback socket pair.
@pytest.mark.allow_network
@pytest.mark.asyncio
async def test_queued_import_unmount_releases_host_before_thread_body(
    tmp_path,
    monkeypatch,
):
    """Cancellation before executor entry releases ownership without mutation."""
    import asyncio

    from textual.worker import Worker

    source = tmp_path / "outside.gguf"
    source.write_bytes(b"gguf")
    service = _Task6HostImportService(tmp_path)
    app = _app()

    async with app.run_test(size=(120, 40)) as pilot:
        screen, _window, installed = await _task6_mounted_host(app, pilot)
        installed._service_factory = lambda: service
        installed._service = None
        screen._run_curated_preflight = MagicMock(return_value=MagicMock())
        worker_queued = asyncio.Event()
        release_executor = asyncio.Event()
        original_run_threaded = Worker._run_threaded

        async def hold_before_executor(worker):
            worker_queued.set()
            await release_executor.wait()
            return await original_run_threaded(worker)

        monkeypatch.setattr(Worker, "_run_threaded", hold_before_executor)
        installed._begin_import(source)
        assert await _wait_for(worker_queued.is_set, pilot)
        assert screen._local_gguf_import_active is True

        await installed.remove()
        await pilot.pause()

        assert screen._local_gguf_import_active is False
        assert service.entered.is_set() is False
        assert service.activation_calls == 0
        screen.post_message(_task6_install_request("curated"))
        await pilot.pause()
        screen._run_curated_preflight.assert_called_once_with()

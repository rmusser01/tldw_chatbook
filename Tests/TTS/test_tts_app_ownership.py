from __future__ import annotations

import ast
import asyncio
import builtins
import gc
import threading
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock
from uuid import UUID

import pytest

import tldw_chatbook.app as app_module
import tldw_chatbook.TTS as tts_package
from Tests.TTS.adapter_fakes import FakeAdapterFactory, provider_spec
from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Event_Handlers.STTS_Events.stts_events import (
    STTSEventHandler,
    STTSPlaygroundGenerateEvent,
    STTSProviderConfigurationChanged,
    STTSSettingsSaveEvent,
)
from tldw_chatbook.TTS import (
    ProfileRepositoryState,
    STTSPlaygroundRequest,
)
from tldw_chatbook.TTS.adapter_registry import TTSAdapterRegistry
from tldw_chatbook.TTS.adapter_types import ProgressSink, TTSProgress
from tldw_chatbook.TTS.preferences import TTSPreferencesSnapshot
from tldw_chatbook.TTS.profile_repository import TTSProfileRepository
from tldw_chatbook.TTS.TTS_Generation import (
    TTSService,
    TTSSettingsPersistenceOutcome,
    TTSSettingsPublicationTicket,
    get_tts_service,
    reset_tts_service_binding,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _playground_event(
    *,
    response_format: str = "wav",
) -> STTSPlaygroundGenerateEvent:
    return STTSPlaygroundGenerateEvent(
        STTSPlaygroundRequest(
            operation_id="ownership-test-operation",
            provider_id="openai",
            model_id="tts-1",
            text="hello",
            voice_id="alloy",
            response_format=response_format,
            speed=1.0,
        )
    )


class FakeOwnedService:
    def __init__(self) -> None:
        self.close_calls = 0
        self.wait_closed_calls = 0

    async def close(self) -> None:
        self.close_calls += 1

    async def wait_closed(self) -> None:
        self.wait_closed_calls += 1


@pytest.fixture(autouse=True)
def isolated_tts_binding() -> Iterator[None]:
    reset_tts_service_binding()
    yield
    reset_tts_service_binding()


def _method_node(
    path: Path,
    class_name: str,
    method_name: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    )


def _self_method_calls(node: ast.AST, method_name: str) -> list[ast.Call]:
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "self"
        and call.func.attr == method_name
    ]


def _isolate_constructor_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        app_module,
        "get_tts_profiles_db_path",
        lambda: tmp_path / "tts_profiles.sqlite",
        raising=False,
    )
    monkeypatch.setattr(
        "tldw_chatbook.app.get_library_collections_db_path",
        lambda: tmp_path / "library_collections.sqlite",
    )
    monkeypatch.setattr(
        "tldw_chatbook.app.get_library_ingest_jobs_db_path",
        lambda: tmp_path / "library_ingest_jobs.sqlite",
    )
    monkeypatch.setattr(
        "tldw_chatbook.app.get_scheduled_tasks_db_path",
        lambda: tmp_path / "scheduled_tasks.sqlite",
    )


def test_tts_package_exports_profile_repository_owner() -> None:
    assert tts_package.TTSProfileRepository is TTSProfileRepository


def test_app_constructs_one_closed_pure_profile_repository(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "missing-parent" / "profiles.sqlite"
    repositories: list[TTSProfileRepository] = []

    def build_repository(path: Path) -> TTSProfileRepository:
        repository = TTSProfileRepository(path)
        repositories.append(repository)
        return repository

    monkeypatch.setattr(
        app_module,
        "get_tts_profiles_db_path",
        Mock(return_value=database_path),
        raising=False,
    )
    monkeypatch.setattr(
        app_module,
        "TTSProfileRepository",
        build_repository,
        raising=False,
    )
    _isolate_constructor_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(
        app_module,
        "get_tts_profiles_db_path",
        Mock(return_value=database_path),
        raising=False,
    )

    app = _build_test_app()

    assert repositories == [app._tts_profile_repository]
    assert app._tts_profile_repository.state is ProfileRepositoryState.CLOSED
    assert app._tts_profile_repository.generation == 0
    assert app._tts_profile_repository.terminal is False
    assert app._tts_profile_repository._executor is None
    assert app._tts_profile_repository._connection is None
    assert app._tts_profile_repository._lease is None
    assert app._tts_profile_repository_open_task is None
    assert app._tts_profile_repository_close_task is None
    assert not database_path.parent.exists()


@pytest.mark.asyncio
async def test_profile_repository_ensure_joins_one_lazy_open_idempotently() -> None:
    open_started = asyncio.Event()
    allow_open = asyncio.Event()

    class BlockingRepository:
        state = ProfileRepositoryState.CLOSED
        open_calls = 0

        async def open(self) -> None:
            self.open_calls += 1
            open_started.set()
            await allow_open.wait()
            self.state = ProfileRepositoryState.OPEN

    repository = BlockingRepository()
    owner = SimpleNamespace(
        _tts_profile_repository=repository,
        _tts_profile_repository_open_task=None,
        _tts_profile_repository_close_task=None,
        loguru_logger=Mock(),
    )

    first = asyncio.create_task(TldwCli._ensure_tts_profile_repository(owner))
    second = asyncio.create_task(TldwCli._ensure_tts_profile_repository(owner))
    await open_started.wait()
    first.cancel("one waiter stopped")
    await asyncio.sleep(0)

    assert second.done() is False
    allow_open.set()

    with pytest.raises(asyncio.CancelledError, match="one waiter stopped"):
        await first
    assert await second is repository
    assert await TldwCli._ensure_tts_profile_repository(owner) is repository
    assert repository.open_calls == 1


@pytest.mark.asyncio
async def test_profile_repository_open_failure_is_safe_and_nonfatal() -> None:
    secret = "/private/profile/path/never-log.sqlite"

    class FailingRepository:
        state = ProfileRepositoryState.CLOSED

        async def open(self) -> None:
            self.state = ProfileRepositoryState.UNAVAILABLE
            raise RuntimeError(f"could not open {secret}")

    repository = FailingRepository()
    owner = SimpleNamespace(
        _tts_profile_repository=repository,
        _tts_profile_repository_open_task=None,
        _tts_profile_repository_close_task=None,
        loguru_logger=Mock(),
    )

    result = await TldwCli._ensure_tts_profile_repository(owner)

    assert result is None
    assert repository.state is ProfileRepositoryState.UNAVAILABLE
    warning_copy = repr(owner.loguru_logger.warning.call_args_list)
    assert "phase=open" in warning_copy
    assert "RuntimeError" in warning_copy
    assert "operation_failed" in warning_copy
    assert secret not in warning_copy


class _OpenControlFlow(BaseException):
    """Test-only signal that follows the BaseException control-flow path."""


@pytest.mark.parametrize(
    "failure_type",
    (RuntimeError, _OpenControlFlow),
    ids=("ordinary-error", "control-flow"),
)
@pytest.mark.asyncio
async def test_cancelled_sole_open_waiter_settles_retained_task_without_disclosure(
    failure_type: type[BaseException],
) -> None:
    secret = "/private/profile/path/detached-open.sqlite"
    open_started = asyncio.Event()
    allow_open = asyncio.Event()

    class FailingRepository:
        state = ProfileRepositoryState.CLOSED

        async def open(self) -> None:
            open_started.set()
            await allow_open.wait()
            self.state = ProfileRepositoryState.UNAVAILABLE
            raise failure_type(f"could not open {secret}")

    repository = FailingRepository()
    owner: Any = SimpleNamespace(
        _tts_profile_repository=repository,
        _tts_profile_repository_open_task=None,
        _tts_profile_repository_close_task=None,
        loguru_logger=Mock(),
    )
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    unhandled_contexts: list[dict[str, Any]] = []
    loop.set_exception_handler(
        lambda _loop, context: unhandled_contexts.append(context)
    )
    try:
        waiter = asyncio.create_task(TldwCli._ensure_tts_profile_repository(owner))
        await open_started.wait()
        retained_task = owner._tts_profile_repository_open_task
        assert retained_task is not None

        waiter.cancel("sole open waiter stopped")
        with pytest.raises(asyncio.CancelledError, match="sole open waiter stopped"):
            await waiter

        allow_open.set()
        while not retained_task.done():
            await asyncio.sleep(0)
        await asyncio.sleep(0)

        marker_cleared = owner._tts_profile_repository_open_task is None
        if not marker_cleared:
            owner._tts_profile_repository_open_task = None
        del waiter
        del retained_task
        gc.collect()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
    finally:
        allow_open.set()
        loop.set_exception_handler(previous_handler)

    warning_copy = repr(owner.loguru_logger.warning.call_args_list)
    context_copy = repr(unhandled_contexts)
    assert marker_cleared is True
    assert unhandled_contexts == []
    assert secret not in warning_copy
    assert secret not in context_copy


@pytest.mark.asyncio
async def test_profile_repository_ensure_rejects_publication_after_close_admission() -> (
    None
):
    open_started = asyncio.Event()
    allow_open = asyncio.Event()
    close_started = asyncio.Event()
    allow_close = asyncio.Event()

    class RacingRepository:
        state = ProfileRepositoryState.CLOSED
        open_calls = 0
        close_calls = 0

        async def open(self) -> None:
            self.open_calls += 1
            open_started.set()
            await allow_open.wait()
            self.state = ProfileRepositoryState.OPEN

        async def close(self) -> None:
            self.close_calls += 1
            close_started.set()
            await allow_close.wait()
            self.state = ProfileRepositoryState.CLOSED

    repository = RacingRepository()
    owner: Any = SimpleNamespace(
        _tts_profile_repository=repository,
        _tts_profile_repository_open_task=None,
        _tts_profile_repository_close_task=None,
        loguru_logger=Mock(),
    )

    ensure_task = asyncio.create_task(TldwCli._ensure_tts_profile_repository(owner))
    await open_started.wait()
    close_waiter = asyncio.create_task(TldwCli._close_tts_profile_repository(owner))
    await close_started.wait()
    assert owner._tts_profile_repository_close_task is not None

    allow_open.set()
    first_result = await ensure_task
    second_result = await TldwCli._ensure_tts_profile_repository(owner)
    allow_close.set()
    await close_waiter

    assert first_result is None
    assert second_result is None
    assert repository.open_calls == 1
    assert repository.close_calls == 1


@pytest.mark.asyncio
async def test_profile_repository_close_is_shared_idempotent_and_cancellation_safe() -> (
    None
):
    close_started = asyncio.Event()
    allow_close = asyncio.Event()

    class BlockingRepository:
        close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1
            close_started.set()
            await allow_close.wait()

    repository = BlockingRepository()
    owner = SimpleNamespace(
        _tts_profile_repository=repository,
        _tts_profile_repository_open_task=None,
        _tts_profile_repository_close_task=None,
        loguru_logger=Mock(),
    )
    first = asyncio.create_task(TldwCli._close_tts_profile_repository(owner))
    second = asyncio.create_task(TldwCli._close_tts_profile_repository(owner))
    await close_started.wait()
    first.cancel("app shutdown cancelled")
    await asyncio.sleep(0)

    assert first.done() is False
    assert second.done() is False
    allow_close.set()

    with pytest.raises(asyncio.CancelledError, match="app shutdown cancelled"):
        await first
    await second
    await TldwCli._close_tts_profile_repository(owner)
    assert repository.close_calls == 1


@pytest.mark.asyncio
async def test_owned_tts_cleanup_runs_both_and_preserves_first_failure() -> None:
    calls: list[str] = []
    profile_error = RuntimeError("profile path must stay private")
    service_error = RuntimeError("service secret must stay private")

    async def close_profile() -> None:
        calls.append("profile")
        raise profile_error

    async def close_service() -> None:
        calls.append("service")
        raise service_error

    owner = SimpleNamespace(
        _close_tts_profile_repository=close_profile,
        _close_tts_service=close_service,
        loguru_logger=Mock(),
    )

    with pytest.raises(RuntimeError) as caught:
        await TldwCli._close_owned_tts_resources(owner)

    assert caught.value is profile_error
    assert calls == ["profile", "service"]
    assert any("cleanup" in note.lower() for note in profile_error.__notes__)
    warning_copy = repr(owner.loguru_logger.warning.call_args_list)
    assert "profile path must stay private" not in warning_copy
    assert "service secret must stay private" not in warning_copy


@pytest.mark.asyncio
async def test_owned_tts_cleanup_preserves_cancellation_and_still_closes_service() -> (
    None
):
    calls: list[str] = []
    cancellation = asyncio.CancelledError("shutdown interrupted")

    async def close_profile() -> None:
        calls.append("profile")
        raise cancellation

    async def close_service() -> None:
        calls.append("service")
        raise RuntimeError("secondary service failure")

    owner = SimpleNamespace(
        _close_tts_profile_repository=close_profile,
        _close_tts_service=close_service,
        loguru_logger=Mock(),
    )

    with pytest.raises(asyncio.CancelledError) as caught:
        await TldwCli._close_owned_tts_resources(owner)

    assert caught.value is cancellation
    assert calls == ["profile", "service"]
    assert any("cleanup" in note.lower() for note in cancellation.__notes__)


@pytest.mark.parametrize(
    "control_flow_type",
    (KeyboardInterrupt, SystemExit),
)
@pytest.mark.asyncio
async def test_owned_tts_cleanup_prefers_later_control_flow_over_ordinary_failure(
    control_flow_type: type[BaseException],
) -> None:
    calls: list[str] = []
    ordinary_error = RuntimeError("ordinary profile failure")
    control_flow = control_flow_type("service control flow")

    async def close_profile() -> None:
        calls.append("profile")
        raise ordinary_error

    async def close_service() -> None:
        calls.append("service")
        raise control_flow

    owner: Any = SimpleNamespace(
        _close_tts_profile_repository=close_profile,
        _close_tts_service=close_service,
        loguru_logger=Mock(),
    )

    with pytest.raises(control_flow_type) as caught:
        await TldwCli._close_owned_tts_resources(owner)

    assert caught.value is control_flow
    assert calls == ["profile", "service"]


@pytest.mark.asyncio
async def test_owned_tts_cleanup_preserves_earliest_control_flow_signal() -> None:
    calls: list[str] = []
    first_control_flow = KeyboardInterrupt("profile control flow")
    later_cancellation = asyncio.CancelledError("later cancellation")

    async def close_profile() -> None:
        calls.append("profile")
        raise first_control_flow

    async def close_service() -> None:
        calls.append("service")
        raise later_cancellation

    owner: Any = SimpleNamespace(
        _close_tts_profile_repository=close_profile,
        _close_tts_service=close_service,
        loguru_logger=Mock(),
    )

    with pytest.raises(KeyboardInterrupt) as caught:
        await TldwCli._close_owned_tts_resources(owner)

    assert caught.value is first_control_flow
    assert calls == ["profile", "service"]


def test_only_application_constructs_profile_repository() -> None:
    constructor_calls: list[Path] = []
    package_root = REPO_ROOT / "tldw_chatbook"
    for path in package_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if any(
            isinstance(call, ast.Call)
            and (
                isinstance(call.func, ast.Name)
                and call.func.id == "TTSProfileRepository"
                or isinstance(call.func, ast.Attribute)
                and call.func.attr == "TTSProfileRepository"
            )
            for call in ast.walk(tree)
        ):
            constructor_calls.append(path)

    assert constructor_calls == [REPO_ROOT / "tldw_chatbook/app.py"]


def test_app_construction_defers_profile_service() -> None:
    constructor = _method_node(
        REPO_ROOT / "tldw_chatbook/app.py",
        "TldwCli",
        "__init__",
    )
    deferred_assignments = [
        node
        for node in ast.walk(constructor)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Attribute)
        and isinstance(node.target.value, ast.Name)
        and node.target.value.id == "self"
        and node.target.attr == "_tts_profile_service"
    ]
    profile_service_calls = [
        node
        for node in ast.walk(constructor)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "TTSProfileService"
    ]

    assert len(deferred_assignments) == 1
    assert ast.unparse(deferred_assignments[0].annotation) == (
        "TTSProfileService | None"
    )
    assert isinstance(deferred_assignments[0].value, ast.Constant)
    assert deferred_assignments[0].value.value is None
    assert profile_service_calls == []


@pytest.mark.asyncio
async def test_profile_service_concurrent_first_use_joins_one_open_and_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    open_started = asyncio.Event()
    allow_open = asyncio.Event()

    class BlockingRepository:
        state = ProfileRepositoryState.CLOSED
        open_calls = 0

        async def open(self) -> None:
            self.open_calls += 1
            open_started.set()
            await allow_open.wait()
            self.state = ProfileRepositoryState.OPEN

    repository = BlockingRepository()
    tts_service = object()
    profile_service = object()
    coordinator = object()
    constructions: list[tuple[object, object, object]] = []

    def build_profile_service(
        repository_dependency: object,
        tts_dependency: object,
        *,
        artifact_lease_coordinator: object,
    ) -> object:
        assert repository.state is ProfileRepositoryState.OPEN
        constructions.append(
            (repository_dependency, tts_dependency, artifact_lease_coordinator)
        )
        return profile_service

    monkeypatch.setattr(
        app_module,
        "TTSProfileService",
        build_profile_service,
        raising=False,
    )
    owner: Any = SimpleNamespace(
        _tts_profile_repository=repository,
        _tts_profile_repository_open_task=None,
        _tts_profile_repository_close_task=None,
        tts_service=tts_service,
        loguru_logger=Mock(),
    )

    async def ensure_repository() -> object | None:
        return await TldwCli._ensure_tts_profile_repository(owner)

    owner._ensure_tts_profile_repository = ensure_repository
    owner._ensure_audio_cpp_artifact_lease_coordinator = lambda: coordinator

    first = asyncio.create_task(TldwCli._ensure_tts_profile_service(owner))
    second = asyncio.create_task(TldwCli._ensure_tts_profile_service(owner))
    await open_started.wait()
    first.cancel("one profile-service waiter stopped")

    with pytest.raises(
        asyncio.CancelledError,
        match="one profile-service waiter stopped",
    ):
        await first
    assert second.done() is False
    assert constructions == []

    allow_open.set()

    assert await second is profile_service
    assert await TldwCli._ensure_tts_profile_service(owner) is profile_service
    assert repository.open_calls == 1
    assert constructions == [(repository, tts_service, coordinator)]
    assert owner._tts_profile_service is profile_service


@pytest.mark.asyncio
async def test_removal_evidence_uses_bounded_profile_snapshot_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef
    from tldw_chatbook.TTS.profile_errors import ProfileRepositoryError

    class Profiles:
        async def bounded_profile_assignment_snapshot(self):
            raise ProfileRepositoryError("unavailable")

        def __getattr__(self, name: str):
            if name in {"list_profiles", "get_profile", "assignment_count"}:
                raise AssertionError("app must use the service-owned bounded snapshot")
            raise AttributeError(name)

    class Registry:
        async def provider_configuration_snapshot(self, _provider_id: str):
            return SimpleNamespace(
                staged_config=None,
                applied_config={},
                staged_generation=0,
            )

    profile_service = Profiles()
    owner = SimpleNamespace(
        app_config={},
        screen=SimpleNamespace(),
        tts_service=SimpleNamespace(registry=Registry()),
    )

    async def ensure_profiles() -> Profiles:
        return profile_service

    owner._ensure_tts_profile_service = ensure_profiles
    owner._audio_cpp_removal_settings_inputs = lambda: (
        app_module.AudioCppSettingsConfig(),
        None,
        TTSPreferencesSnapshot(
            provider_id="openai",
            model_mode="exact",
            model_id="tts-1",
            voice_mode="exact",
            voice_id="alloy",
            response_format="mp3",
            speed=1.0,
        ),
        None,
    )
    monkeypatch.setattr(
        app_module,
        "project_audio_cpp_artifact_removal_evidence",
        lambda *_args, **_kwargs: object(),
    )

    with pytest.raises(ProfileRepositoryError, match="unavailable"):
        await TldwCli._audio_cpp_artifact_removal_evidence(
            owner,
            ArtifactRef("audio-cpp-model", "a" * 40, "q8_0"),
        )


@pytest.mark.asyncio
async def test_model_library_bulk_observation_collects_shared_evidence_once() -> None:
    """Many exact refs share one Settings/profile/runtime collection generation."""
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef

    calls = {"settings": 0, "profiles": 0, "runtime": 0}

    class Profiles:
        async def bounded_profile_assignment_snapshot(self):
            calls["profiles"] += 1
            return ()

    class Registry:
        async def provider_configuration_snapshot(self, provider_id: str):
            calls["runtime"] += 1
            assert provider_id == "audio_cpp"
            return SimpleNamespace(
                staged_config=None,
                applied_config={},
                staged_generation=0,
            )

    def settings_inputs():
        calls["settings"] += 1
        return (
            app_module.AudioCppSettingsConfig(),
            None,
            TTSPreferencesSnapshot(
                provider_id="openai",
                model_mode="exact",
                model_id="tts-1",
                voice_mode="exact",
                voice_id="alloy",
                response_format="mp3",
                speed=1.0,
            ),
            None,
        )

    profiles = Profiles()

    async def ensure_profiles():
        return profiles

    owner = SimpleNamespace(
        _audio_cpp_removal_settings_inputs=settings_inputs,
        _ensure_tts_profile_service=ensure_profiles,
        tts_service=SimpleNamespace(
            registry=Registry(),
            _audio_cpp_supervisor=None,
        ),
    )
    references = tuple(
        ArtifactRef(f"audio-cpp-model-{index}", chr(97 + index) * 40, "q8_0")
        for index in range(3)
    )
    collect = getattr(TldwCli, "_audio_cpp_model_library_observation_snapshot", None)
    assert callable(collect)

    snapshot = await collect(owner, references)

    assert type(snapshot).__name__ == "AudioCppModelLibraryObservationSnapshot"
    assert tuple(item.reference for item in snapshot.observations) == references
    assert calls == {"settings": 1, "profiles": 1, "runtime": 1}
    with pytest.raises((AttributeError, TypeError)):
        snapshot.observations = ()


def test_removal_settings_inputs_prefer_exact_detached_typed_draft() -> None:
    """Detached Settings state wins over a stale mounted fallback snapshot."""

    from dataclasses import replace

    from Tests.UI.test_settings_speech_tts_panel import _audio_cpp_state
    from tldw_chatbook.UI.Navigation.screen_state_store import (
        RuntimeIdentity,
        ScreenStateStore,
    )
    from tldw_chatbook.Widgets.Settings_Widgets.speech_tts_settings_panel import (
        SpeechTTSPanelDraftSnapshot,
        _RealtimeSettingsDraft,
    )

    realtime = _RealtimeSettingsDraft(
        False,
        "openai",
        "gpt-realtime",
        "",
        "30",
        "auto",
        "semantic_vad",
        "0.5",
        "500",
    )

    def snapshot(model_id: str) -> SpeechTTSPanelDraftSnapshot:
        state = _audio_cpp_state(model_mode="exact", model_id=model_id)
        original = _audio_cpp_state()
        return SpeechTTSPanelDraftSnapshot(
            state=state,
            original_state=original,
            realtime_draft=realtime,
            realtime_original=replace(realtime),
            configure_provider="audio_cpp",
            draft_revision=1,
        )

    identity = RuntimeIdentity("local")
    store = ScreenStateStore()
    store.save(
        "settings",
        {"speech_tts_panel_draft": snapshot("stored-exact-model")},
        identity,
    )
    owner = SimpleNamespace(
        app_config={},
        screen_state_store=store,
        screen=SimpleNamespace(
            _speech_tts_draft_snapshot=snapshot("mounted-stale-model")
        ),
        _current_runtime_identity=lambda: identity,
    )

    _saved, draft, _saved_preferences, draft_preferences = (
        TldwCli._audio_cpp_removal_settings_inputs(owner)
    )

    assert draft is not None
    assert draft_preferences is not None
    assert draft_preferences.provider_id == "audio_cpp"
    assert draft_preferences.model_mode == "exact"
    assert draft_preferences.model_id == "stored-exact-model"


def test_removal_settings_inputs_treat_missing_stored_draft_as_no_draft() -> None:
    """An authoritative Settings snapshot without Speech/TTS has no draft."""

    from Tests.UI.test_settings_speech_tts_panel import _audio_cpp_state
    from tldw_chatbook.UI.Navigation.screen_state_store import (
        RuntimeIdentity,
        ScreenStateStore,
    )

    identity = RuntimeIdentity("local")
    store = ScreenStateStore()
    store.save("settings", {"active_category": "theme"}, identity)
    owner = SimpleNamespace(
        app_config={},
        screen_state_store=store,
        screen=SimpleNamespace(
            _speech_tts_draft_snapshot=SimpleNamespace(state=_audio_cpp_state())
        ),
        _current_runtime_identity=lambda: identity,
    )

    _saved, draft, _saved_preferences, draft_preferences = (
        TldwCli._audio_cpp_removal_settings_inputs(owner)
    )

    assert draft is None
    assert draft_preferences is None


@pytest.mark.asyncio
async def test_app_lifecycle_shutdown_drains_all_owners_in_authority_order() -> None:
    calls: list[str] = []

    class Coordinator:
        async def shutdown(self) -> None:
            calls.append("coordinator")

    class InstallOwner:
        async def shutdown(self) -> None:
            calls.append("install")

    async def image_shutdown() -> None:
        calls.append("image")

    async def console_runtime_shutdown() -> None:
        # TASK-18609: production drains the Console runtime between the
        # image-edit and file-notes owners (app.py
        # `_shutdown_app_owned_lifecycles`); the double drifted when the
        # call was added and the test died on the missing attribute instead
        # of asserting the ordering it exists for.
        calls.append("console-runtime")

    async def notes_sync_shutdown() -> None:
        calls.append("notes-sync")

    async def actor_pack_import_shutdown() -> None:
        calls.append("actor-pack-import")

    async def actor_pack_export_shutdown() -> None:
        calls.append("actor-pack-export")

    async def persona_buddy_shutdown() -> None:
        calls.append("persona-buddy")

    class ChangeReviewOwner:
        def shutdown(self, *, timeout: float) -> None:
            assert timeout == 1.0
            calls.append("change-review")

    async def notes_shutdown() -> None:
        calls.append("notes")

    owner = SimpleNamespace(
        _audio_cpp_artifact_lease_coordinator=Coordinator(),
        audio_cpp_model_install_owner=InstallOwner(),
        _shutdown_notes_sync_runtime=notes_sync_shutdown,
        _shutdown_actor_pack_import=actor_pack_import_shutdown,
        _shutdown_actor_pack_export=actor_pack_export_shutdown,
        _shutdown_console_image_edits=image_shutdown,
        _shutdown_console_runtime=console_runtime_shutdown,
        _shutdown_persona_buddy=persona_buddy_shutdown,
        change_review_consent_service=ChangeReviewOwner(),
        _shutdown_file_notes_session_owner=notes_shutdown,
    )

    await TldwCli._shutdown_app_owned_lifecycles(owner)

    assert calls == [
        "notes-sync",
        "actor-pack-import",
        "actor-pack-export",
        "console-runtime",
        "change-review",
        "persona-buddy",
        "coordinator",
        "install",
        "image",
        "notes",
    ]


@pytest.mark.asyncio
async def test_profile_service_store_open_failure_leaves_ordinary_tts_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tts_service = FakeOwnedService()

    async def unavailable_repository() -> None:
        return None

    constructor = Mock(
        side_effect=AssertionError(
            "profile service must not be constructed without its repository"
        )
    )
    monkeypatch.setattr(
        app_module,
        "TTSProfileService",
        constructor,
        raising=False,
    )
    owner: Any = SimpleNamespace(
        _ensure_tts_profile_repository=unavailable_repository,
        tts_service=tts_service,
        _tts_binding_active=True,
    )

    result = await TldwCli._ensure_tts_profile_service(owner)

    assert result is None
    assert getattr(owner, "_tts_profile_service", None) is None
    assert owner.tts_service is tts_service
    assert owner._tts_binding_active is True
    assert tts_service.close_calls == 0
    assert tts_service.wait_closed_calls == 0
    constructor.assert_not_called()


def test_profile_service_owns_only_existing_app_dependencies() -> None:
    class FocusedRepository:
        generation = 1

        async def list_profiles(self, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError("not used")

        async def create_profile(self, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError("not used")

        async def create_profile_with_reference(
            self,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            raise AssertionError("not used")

        async def update_profile(self, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError("not used")

        async def delete_profile(self, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError("not used")

        async def assignment_count(self, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError("not used")

        async def get_assigned_profile(
            self,
            character_ref: tts_package.CharacterRef,
        ) -> Any:
            raise AssertionError(character_ref)

        async def get_profile(self, profile_id: UUID) -> Any:
            raise AssertionError(profile_id)

        async def set_assignment(
            self,
            character_ref: tts_package.CharacterRef,
            profile_id: UUID,
            *,
            expected_generation: int,
            expected_profile_revision: int,
            expected_current_profile_id: UUID | None,
        ) -> tts_package.ProfileStoreResult[tts_package.CharacterTTSAssignment]:
            del (
                character_ref,
                profile_id,
                expected_generation,
                expected_profile_revision,
                expected_current_profile_id,
            )
            raise AssertionError("not used")

        async def remove_assignment(
            self,
            character_ref: tts_package.CharacterRef,
            *,
            expected_generation: int,
            expected_profile_id: UUID,
        ) -> tts_package.ProfileStoreResult[None]:
            del character_ref, expected_generation, expected_profile_id
            raise AssertionError("not used")

        async def get_reference(
            self,
            profile_id: UUID,
            *,
            expected_revision: int,
            expected_generation: int,
        ) -> Any:
            del profile_id, expected_revision, expected_generation
            raise AssertionError("not used")

    class FocusedTTSService:
        def configuration_revision(self, provider_id: str) -> int:
            raise AssertionError(provider_id)

        async def get_native_capability_snapshot(
            self,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            raise AssertionError("not used")

        async def require_current_configuration_revision(
            self,
            *args: Any,
            **kwargs: Any,
        ) -> None:
            raise AssertionError("not used")

        async def audio_cpp_guided_dependency_snapshot(
            self,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            raise AssertionError("not used")

    repository = FocusedRepository()
    tts_service = FocusedTTSService()

    profile_service = tts_package.TTSProfileService(repository, tts_service)

    assert profile_service._repository is repository
    assert profile_service._tts_service is tts_service
    assert isinstance(profile_service._consumer_mutation_lock, asyncio.Lock)
    assert set(vars(profile_service)) == {
        "_repository",
        "_tts_service",
        "_consumer_mutation_lock",
        "_sample_evidence",
        "_sample_evidence_lock",
        "_sample_evidence_lifecycle",
        "_sample_evidence_epoch",
    }
    for resource_name in (
        "_close_task",
        "_adapter",
        "_registry",
        "_executor",
        "_connection",
        "close",
        "shutdown",
        "wait_closed",
    ):
        assert not hasattr(profile_service, resource_name)
    assert not hasattr(TldwCli, "_close_tts_profile_service")


def test_app_constructs_one_tts_service(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service = FakeOwnedService()
    builder = Mock(return_value=service)
    monkeypatch.setattr("tldw_chatbook.app.build_default_tts_service", builder)
    _isolate_constructor_paths(monkeypatch, tmp_path)

    app = _build_test_app()

    assert app.tts_service is service
    assert app._tts_binding_active is False
    builder.assert_called_once_with(app.app_config)


def test_app_construction_keeps_audio_cpp_import_and_all_adapters_lazy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    factory = FakeAdapterFactory("openai")

    def provider_specs(
        config: Mapping[str, Any],
    ) -> tuple[Any, ...]:
        del config
        return (provider_spec("openai", factory),)

    monkeypatch.setattr(
        "tldw_chatbook.TTS.adapter_bootstrap.legacy_provider_specs",
        provider_specs,
    )
    real_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: Mapping[str, Any] | None = None,
        locals: Mapping[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "tldw_chatbook.TTS.adapters.audio_cpp":
            raise AssertionError(
                "app construction must not import the audio.cpp adapter"
            )
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    _isolate_constructor_paths(monkeypatch, tmp_path)

    app = _build_test_app()

    assert tuple(
        descriptor.provider_id for descriptor in app.tts_service.provider_descriptors()
    ) == ("audio_cpp", "openai")
    assert app.tts_service.preferences_snapshot() == (
        TTSPreferencesSnapshot.from_settings(app.app_config)
    )
    assert factory.calls == 0


@pytest.mark.asyncio
async def test_app_binding_and_close_are_explicit_and_idempotent() -> None:
    service = FakeOwnedService()
    owner = SimpleNamespace(tts_service=service, _tts_binding_active=False)

    TldwCli._bind_tts_service(owner)
    TldwCli._bind_tts_service(owner)
    assert await get_tts_service() is service

    await TldwCli._close_tts_service(owner)
    await TldwCli._close_tts_service(owner)

    assert service.close_calls == 1
    assert service.wait_closed_calls == 1
    assert owner._tts_binding_active is False
    with pytest.raises(RuntimeError, match="not bound"):
        await get_tts_service()


@pytest.mark.asyncio
async def test_stts_initialization_only_retrieves_the_bound_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = object()
    get_service = AsyncMock(return_value=service)
    monkeypatch.setattr(
        "tldw_chatbook.Event_Handlers.STTS_Events.stts_events.get_tts_service",
        get_service,
    )
    monkeypatch.setattr(
        "tldw_chatbook.config.load_cli_config_and_ensure_existence",
        lambda: pytest.fail("initialization must not rebuild compatibility config"),
    )
    handler = STTSEventHandler(SimpleNamespace())

    await handler.initialize_stts()

    get_service.assert_awaited_once_with()
    assert handler._stts_service is service


def test_app_forwards_provider_configuration_change_to_stts_handler() -> None:
    callback = Mock()
    owner = SimpleNamespace(
        _stts_handler=SimpleNamespace(
            on_stts_provider_configuration_changed=callback,
        )
    )
    event = STTSProviderConfigurationChanged("audio_cpp", 2)

    TldwCli.handle_stts_provider_configuration_changed(owner, event)

    callback.assert_called_once_with(event)


def test_existing_mount_binds_before_screen_work() -> None:
    method = _method_node(REPO_ROOT / "tldw_chatbook/app.py", "TldwCli", "on_mount")
    bind_calls = _self_method_calls(method, "_bind_tts_service")
    restore_calls = _self_method_calls(
        method, "_restore_ingest_jobs_and_schedule_research_sources"
    )

    assert len(bind_calls) == 1
    assert len(restore_calls) == 1
    assert bind_calls[0].lineno < restore_calls[0].lineno


def test_unmount_closes_owned_tts_resources_from_outer_finally() -> None:
    method = _method_node(REPO_ROOT / "tldw_chatbook/app.py", "TldwCli", "on_unmount")
    close_calls = _self_method_calls(method, "_close_owned_tts_resources")

    assert len(close_calls) == 1
    enclosing_cleanup = next(
        statement
        for statement in method.body
        if isinstance(statement, ast.Try)
        and any(
            close_calls[0] in ast.walk(finally_statement)
            for finally_statement in statement.finalbody
        )
    )
    assert any(
        _self_method_calls(statement, "_disconnect_local_mcp_client")
        for statement in enclosing_cleanup.body
    )
    parent_by_node = {
        child: parent
        for parent in ast.walk(enclosing_cleanup)
        for child in ast.iter_child_nodes(parent)
    }
    ancestor = parent_by_node[close_calls[0]]
    while ancestor is not enclosing_cleanup:
        assert not isinstance(ancestor, ast.If)
        ancestor = parent_by_node[ancestor]

    owner_close = _method_node(
        REPO_ROOT / "tldw_chatbook/app.py",
        "TldwCli",
        "_close_owned_tts_resources",
    )
    portability_calls = _self_method_calls(
        owner_close, "_close_tts_voice_bundle_service"
    )
    repository_calls = _self_method_calls(owner_close, "_close_tts_profile_repository")
    assert len(portability_calls) == 1
    assert len(repository_calls) == 1
    assert portability_calls[0].lineno < repository_calls[0].lineno
    assert len(_self_method_calls(owner_close, "_close_tts_profile_repository")) == 1
    assert len(_self_method_calls(owner_close, "_close_tts_service")) == 1


@pytest.mark.asyncio
async def test_voice_bundle_service_is_lazy_singleton_and_closes_before_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed: list[tuple[Path, object, object]] = []

    class PortabilityOwner(FakeOwnedService):
        pass

    portability = PortabilityOwner()
    repository = object()

    @asynccontextmanager
    async def mutation_fence():
        yield

    profile_service = SimpleNamespace(consumer_mutation_fence=mutation_fence)
    tts_service = object()

    coordinator = object()

    def build(
        root: Path,
        repo: object,
        dependency: object,
        *,
        profile_mutation_fence: object,
        artifact_lease_coordinator: object,
    ) -> object:
        assert artifact_lease_coordinator is coordinator
        assert profile_mutation_fence is mutation_fence
        constructed.append((root, repo, dependency))
        return portability

    # TASK-21108: `app.py` imports the portability class function-locally
    # inside `_ensure_tts_voice_bundle_service` (the 1,857-line module is off
    # the boot path now), so the substitution has to land on the defining
    # module -- there is no `app_module.TTSVoiceBundlePortabilityService` to
    # patch, and patching one would not be read by the deferred import.
    import tldw_chatbook.TTS.voice_bundle_service as voice_bundle_module

    monkeypatch.setattr(
        voice_bundle_module, "TTSVoiceBundlePortabilityService", build
    )
    monkeypatch.setattr(app_module, "get_user_data_dir", lambda: tmp_path)
    owner = SimpleNamespace(
        _tts_voice_bundle_service=None,
        _tts_voice_bundle_service_close_task=None,
        _tts_profile_repository=repository,
        tts_service=tts_service,
    )

    async def ensure_profile_service() -> object:
        return profile_service

    owner._ensure_tts_profile_service = ensure_profile_service
    owner._ensure_audio_cpp_artifact_lease_coordinator = lambda: coordinator

    first = await TldwCli._ensure_tts_voice_bundle_service(owner)
    second = await TldwCli._ensure_tts_voice_bundle_service(owner)
    assert first is portability
    assert second is portability
    assert constructed == [
        (tmp_path / "tts_voice_bundle_portability", repository, tts_service)
    ]

    await TldwCli._close_tts_voice_bundle_service(owner)
    assert portability.close_calls == 1
    assert portability.wait_closed_calls == 1


@pytest.mark.asyncio
async def test_composite_shutdown_joins_each_owner_in_authority_order() -> None:
    events: list[str] = []
    portability_joined = asyncio.Event()
    repository_joined = asyncio.Event()

    async def close_portability() -> None:
        events.append("portability:start")
        await asyncio.sleep(0)
        portability_joined.set()
        events.append("portability:joined")

    async def close_repository() -> None:
        assert portability_joined.is_set()
        events.append("repository:start")
        await asyncio.sleep(0)
        repository_joined.set()
        events.append("repository:joined")

    async def close_service() -> None:
        assert repository_joined.is_set()
        events.append("service:start")
        await asyncio.sleep(0)
        events.append("service:joined")

    owner = SimpleNamespace(
        _close_tts_voice_bundle_service=close_portability,
        _close_tts_profile_repository=close_repository,
        _close_tts_service=close_service,
        loguru_logger=Mock(),
    )

    await TldwCli._close_owned_tts_resources(owner)

    assert events == [
        "portability:start",
        "portability:joined",
        "repository:start",
        "repository:joined",
        "service:start",
        "service:joined",
    ]


def test_application_and_stts_do_not_reach_through_to_backend_manager() -> None:
    paths = (
        REPO_ROOT / "tldw_chatbook/app.py",
        REPO_ROOT / "tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py",
    )
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        accesses = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr == "backend_manager"
        ]
        assert accesses == [], f"{path} reaches through to backend_manager"


class CapturingStreamService:
    def __init__(self) -> None:
        self.progress_sink: ProgressSink | None = None

    async def generate_audio_stream(
        self,
        request: object,
        internal_model_id: str,
        progress_sink: ProgressSink | None = None,
    ) -> AsyncIterator[bytes]:
        del request, internal_model_id
        self.progress_sink = progress_sink
        assert progress_sink is not None
        await progress_sink(
            TTSProgress(
                status="Generating",
                fraction=0.5,
                processed=1,
                total=2,
            )
        )
        yield b"RIFF"


@pytest.mark.asyncio
async def test_stts_forwards_typed_progress_sink_to_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = SimpleNamespace(notify=Mock())
    handler = STTSEventHandler(app=app)
    service = CapturingStreamService()
    handler._stts_service = service
    created_tasks: list[asyncio.Task[Any]] = []
    create_task = asyncio.create_task

    def capture_task(coro: Any, **kwargs: Any) -> asyncio.Task[Any]:
        task = create_task(coro, **kwargs)
        created_tasks.append(task)
        return task

    monkeypatch.setattr(
        "tldw_chatbook.Event_Handlers.STTS_Events.stts_events.asyncio.create_task",
        capture_task,
    )
    status_text = SimpleNamespace(update=Mock())
    progress_bar = SimpleNamespace(update=Mock())
    generation_log = SimpleNamespace(write=Mock())
    status_container = SimpleNamespace(remove_class=Mock(), add_class=Mock())
    audio_status = SimpleNamespace(update=Mock())
    generate_button = SimpleNamespace(disabled=True)
    play_button = SimpleNamespace(disabled=True)
    export_button = SimpleNamespace(disabled=True)
    widgets = {
        "#generation-status-container": status_container,
        "#generation-progress": progress_bar,
        "#generation-status-text": status_text,
        "#tts-generation-log": generation_log,
        "#audio-play-btn": play_button,
        "#audio-export-btn": export_button,
        "#audio-player-status": audio_status,
        "#tts-generate-btn": generate_button,
    }
    scheduled: list[object] = []

    def query_one(selector: str, widget_type: object = None) -> object:
        del widget_type
        return widgets[selector]

    def call_from_thread(callback: object, *args: object) -> None:
        scheduled.append(callback)
        assert callable(callback)
        callback(*args)

    playground = SimpleNamespace(
        query_one=query_one,
        call_from_thread=call_from_thread,
    )
    app.query_one = Mock(return_value=playground)
    event = _playground_event()

    await handler.handle_playground_generate(event)

    assert service.progress_sink is not None
    assert scheduled
    status_text.update.assert_called_with("Generating")
    progress_bar.update.assert_any_call(progress=50.0)
    generation_log.write.assert_any_call("[dim]Processed 1/2 item(s)[/dim]")
    assert created_tasks == []
    assert handler._current_audio_file is not None
    audio_file = handler._current_audio_file

    await handler.cleanup_tts_resources()
    await handler.cleanup_tts_resources()

    assert not audio_file.exists()
    assert handler._current_audio_file is None


@pytest.mark.asyncio
async def test_stts_playground_generation_stays_in_the_owned_event_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_nested_worker(coro: Any, *, exclusive: bool) -> None:
        del exclusive
        coro.close()
        raise AssertionError("nested worker used")

    playground = SimpleNamespace(run_worker=Mock(side_effect=reject_nested_worker))
    app = SimpleNamespace(query_one=Mock(return_value=playground), notify=Mock())
    handler = STTSEventHandler(app=app)
    handler._stts_service = object()
    generation = AsyncMock()
    monkeypatch.setattr(handler, "_generate_tts_worker", generation)
    event = _playground_event()

    await handler.handle_playground_generate(event)

    generation.assert_awaited_once_with(event)
    app.query_one.assert_not_called()
    playground.run_worker.assert_not_called()


@pytest.mark.asyncio
async def test_stts_cleanup_cancels_and_joins_tracked_event_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = STTSEventHandler(app=SimpleNamespace(notify=Mock()))
    started = asyncio.Event()
    finished = asyncio.Event()
    created_tasks: list[asyncio.Task[Any]] = []
    create_task = asyncio.create_task

    def capture_task(coro: Any, **kwargs: Any) -> asyncio.Task[Any]:
        task = create_task(coro, **kwargs)
        created_tasks.append(task)
        return task

    monkeypatch.setattr(
        "tldw_chatbook.Event_Handlers.STTS_Events.stts_events.asyncio.create_task",
        capture_task,
    )

    async def block_until_cancelled(event: STTSSettingsSaveEvent) -> None:
        del event
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            finished.set()

    monkeypatch.setattr(handler, "handle_settings_save", block_until_cancelled)
    handler.on_stts_settings_save_event(STTSSettingsSaveEvent({}))
    await started.wait()

    try:
        tracked_task = next(iter(handler._active_tasks))
        await handler.cleanup_tts_resources()
        await handler.cleanup_tts_resources()

        assert tracked_task.cancelled()
        assert finished.is_set()
        assert handler._active_tasks == set()
    finally:
        for task in created_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*created_tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_stts_event_cancellation_does_not_cancel_service_owned_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook import config as config_module

    registry = TTSAdapterRegistry(
        specs=(provider_spec("audio_cpp", FakeAdapterFactory("audio_cpp")),),
        aliases={},
    )
    service = TTSService(registry)
    app = SimpleNamespace(notify=Mock(), post_message=Mock())
    handler = STTSEventHandler(app=app)
    handler._stts_service = service
    persistence_started = threading.Event()
    release_persistence = threading.Event()
    captured_ticket: TTSSettingsPublicationTicket | None = None
    begin_publication = service.begin_preferences_publication
    preferences = TTSPreferencesSnapshot(
        provider_id="audio_cpp",
        model_mode="exact",
        model_id="Model/Retained",
        voice_mode="server_default",
        voice_id=None,
        response_format="wav",
        speed=1.0,
    )

    def persist_settings(*_args: object, **_kwargs: object) -> object:
        persistence_started.set()
        release_persistence.wait()
        return SimpleNamespace(
            file_replaced=True,
            caches_reloaded=True,
            failure_phase=None,
        )

    def capture_publication(*args: Any, **kwargs: Any) -> TTSSettingsPublicationTicket:
        nonlocal captured_ticket
        captured_ticket = begin_publication(*args, **kwargs)
        return captured_ticket

    monkeypatch.setattr(config_module, "settings", {})
    monkeypatch.setattr(
        config_module,
        "apply_settings_mutation_to_cli_config",
        persist_settings,
    )
    monkeypatch.setattr(service, "begin_preferences_publication", capture_publication)

    handler.on_stts_settings_save_event(
        STTSSettingsSaveEvent({}, preferences=preferences)
    )
    try:
        for _ in range(100):
            if persistence_started.is_set():
                break
            await asyncio.sleep(0.01)
        assert persistence_started.is_set()
        assert captured_ticket is not None

        await handler.cleanup_tts_resources()

        assert captured_ticket.completion.cancelled() is False
        assert captured_ticket.completion.done() is False

        release_persistence.set()
        publication = await asyncio.shield(captured_ticket.completion)

        assert publication.persistence == TTSSettingsPersistenceOutcome(
            file_replaced=True,
            caches_reloaded=True,
            failure_phase=None,
        )
        assert publication.published is True
        assert service.preferences_snapshot() == preferences
    finally:
        release_persistence.set()
        await service.close()
        await service.wait_closed()


@pytest.mark.asyncio
async def test_stts_conversion_cancellation_tracks_and_deletes_partial_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OneChunkService:
        async def generate_audio_stream(
            self,
            request: object,
            internal_model_id: str,
            progress_sink: ProgressSink | None = None,
        ) -> AsyncIterator[bytes]:
            del request, internal_model_id, progress_sink
            yield b"RIFF"

    conversion_started = asyncio.Event()
    process_terminated = asyncio.Event()
    process_waited = asyncio.Event()
    output_path: Path | None = None

    class FakeProcess:
        returncode: int | None = None

        async def communicate(self) -> tuple[bytes, bytes]:
            assert output_path is not None
            output_path.write_bytes(b"partial")
            conversion_started.set()
            await asyncio.Event().wait()
            return b"", b""

        def terminate(self) -> None:
            self.returncode = 0
            process_terminated.set()
            raise ProcessLookupError

        async def wait(self) -> int:
            process_waited.set()
            return self.returncode or 0

    async def create_process(*command: object, **kwargs: object) -> FakeProcess:
        del kwargs
        nonlocal output_path
        output_path = Path(str(command[-1]))
        return FakeProcess()

    monkeypatch.setattr(
        "tldw_chatbook.Event_Handlers.STTS_Events.stts_events.asyncio.create_subprocess_exec",
        create_process,
    )
    handler = STTSEventHandler(app=SimpleNamespace(notify=Mock()))
    handler._stts_service = OneChunkService()
    event = _playground_event(response_format="mp3")
    generation = asyncio.create_task(handler._generate_tts_worker(event))

    try:
        await conversion_started.wait()
        assert output_path is not None
        owned_before_cancellation = output_path in handler._playground_audio_files
        generation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await generation

        await handler.cleanup_tts_resources()

        assert owned_before_cancellation
        assert process_terminated.is_set()
        assert process_waited.is_set()
        assert not output_path.exists()
        assert handler._playground_operation_files == {}
    finally:
        if not generation.done():
            generation.cancel()
            await asyncio.gather(generation, return_exceptions=True)
        if output_path is not None:
            output_path.unlink(missing_ok=True)
        for path in handler._playground_audio_files:
            path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_cancelled_stts_cleanup_finishes_deleting_owned_audio(
    tmp_path: Path,
) -> None:
    handler = STTSEventHandler(app=SimpleNamespace(notify=Mock()))
    owned_audio = tmp_path / "playground.wav"
    owned_audio.write_bytes(b"temporary")
    handler._playground_audio_files.add(owned_audio)
    task_started = asyncio.Event()
    task_cancelling = asyncio.Event()
    release_task = asyncio.Event()

    async def active_handler_task() -> None:
        task_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            task_cancelling.set()
            await release_task.wait()

    handler._start_event_task(active_handler_task())
    await task_started.wait()
    cleanup = asyncio.create_task(handler.cleanup_tts_resources())

    try:
        await task_cancelling.wait()
        cleanup.cancel()
        await asyncio.sleep(0)
        release_task.set()

        with pytest.raises(asyncio.CancelledError):
            await cleanup

        assert not owned_audio.exists()
        assert handler._playground_audio_files == set()
        assert handler._active_tasks == set()
    finally:
        release_task.set()
        if not cleanup.done():
            cleanup.cancel()
            await asyncio.gather(cleanup, return_exceptions=True)
        for task in tuple(handler._active_tasks):
            task.cancel()
        await asyncio.gather(*handler._active_tasks, return_exceptions=True)
        owned_audio.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_stts_cleanup_does_not_wait_for_its_calling_event_task() -> None:
    handler = STTSEventHandler(app=SimpleNamespace(notify=Mock()))
    cleanup_returned = asyncio.Event()

    async def cleanup_from_event_task() -> None:
        await handler.cleanup_tts_resources()
        cleanup_returned.set()

    handler._start_event_task(cleanup_from_event_task())

    await asyncio.wait_for(cleanup_returned.wait(), timeout=1)
    await asyncio.sleep(0)

    assert handler._cleanup_task is not None
    assert handler._cleanup_task.done()
    assert handler._active_tasks == set()


@pytest.mark.asyncio
async def test_stts_cleanup_seals_handler_against_late_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = STTSEventHandler(app=SimpleNamespace(notify=Mock()))
    handler._stts_service = object()
    generate = AsyncMock()
    monkeypatch.setattr(handler, "_generate_tts_worker", generate)
    event = _playground_event()

    await handler.cleanup_tts_resources()
    await handler.handle_playground_generate(event)

    generate.assert_not_awaited()
    assert handler._active_tasks == set()


@pytest.mark.asyncio
async def test_stts_cleanup_preserves_persistent_audiobook_output(
    tmp_path: Path,
) -> None:
    handler = STTSEventHandler(app=SimpleNamespace(notify=Mock()))
    temporary = tmp_path / "playground.wav"
    persistent = tmp_path / "audiobook.wav"
    temporary.write_bytes(b"temporary")
    persistent.write_bytes(b"persistent")
    handler._playground_audio_files.add(temporary)
    handler._current_audio_file = persistent

    await handler.cleanup_tts_resources()
    await handler.cleanup_tts_resources()

    assert not temporary.exists()
    assert persistent.read_bytes() == b"persistent"
    assert handler._current_audio_file == persistent

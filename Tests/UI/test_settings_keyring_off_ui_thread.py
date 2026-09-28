"""TASK-32926: keyring reads on a cache miss must never run on the UI thread.

On Linux an OS-keyring read is a SecretService D-Bus round trip that can block
for seconds, or indefinitely on an unlock prompt. Each fake below stands in for
such a read: it blocks on a ``threading.Event`` the test controls and records
whether it was called on the main thread (where Textual's event loop runs in
these tests). A UI path that still read the keyring synchronously would either
hang the pilot or record a main-thread call.
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from Tests.private_profile import private_profile_test
from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
    _visible_text,
)
from Tests.UI.test_settings_configuration_hub import _select_settings_category
from tldw_chatbook.runtime_policy.server_event_scope import (
    event_principal_id_from_active_context,
)
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen

_RELEASE_TIMEOUT = 10.0


class _BlockingRead:
    """A keyring-shaped read that blocks until released; records its threads."""

    def __init__(self, value: object) -> None:
        self.value = value
        self.release = threading.Event()
        self.main_thread_calls = 0
        self.worker_thread_calls = 0

    def __call__(self, *_args: object, **_kwargs: object) -> object:
        if threading.current_thread() is threading.main_thread():
            self.main_thread_calls += 1
        else:
            self.worker_thread_calls += 1
        self.release.wait(_RELEASE_TIMEOUT)
        return self.value


class _BlockingSkillTrustService:
    keyring_convenience_enabled = True
    reduced_rollback_protection = False

    def __init__(self) -> None:
        self.read = _BlockingRead("trusted")

    def overall_status(self) -> str:
        return self.read()


async def _wait_until(pilot, predicate, message: str, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await pilot.pause(0.02)
    raise AssertionError(message)


# --- Settings > Privacy & Security ------------------------------------------


def test_privacy_posture_can_defer_the_skill_trust_status_read():
    class RaisingService:
        keyring_convenience_enabled = False
        reduced_rollback_protection = False

        def overall_status(self):
            raise AssertionError("posture read the keyring-backed trust status")

    screen = SettingsScreen(
        SimpleNamespace(app_config={}, local_skill_trust_service=RaisingService())
    )

    posture = screen._settings_privacy_posture(read_skill_trust=False)

    assert posture.skill_trust_enabled is True
    assert posture.skill_trust_status == "checking"


@pytest.mark.asyncio
@private_profile_test
async def test_privacy_category_renders_pending_trust_then_fills_from_worker(
    request,
):
    app = _build_test_app()
    service = _BlockingSkillTrustService()
    app.local_skill_trust_service = service
    host = DestinationHarness(app, "settings")

    try:
        async with host.run_test(size=(180, 50)) as pilot:
            screen = _active_destination_screen(host)
            await _select_settings_category(
                screen, pilot, "privacy-security", expected_text="Config encryption"
            )
            # The category rendered while the keyring read is still blocked.
            assert "Skill trust: checking" in _visible_text(screen)

            service.read.release.set()
            await _wait_until(
                pilot,
                lambda: "Skill trust: trusted" in _visible_text(screen),
                "trust posture never filled in from the worker",
            )
            assert service.read.main_thread_calls == 0
            assert service.read.worker_thread_calls >= 1
    finally:
        service.read.release.set()


# --- Settings sync scope -----------------------------------------------------


def _server_mode_app(context_read: _BlockingRead, **extra: object) -> SimpleNamespace:
    return SimpleNamespace(
        app_config={},
        runtime_policy=SimpleNamespace(
            state=SimpleNamespace(active_source="server", active_server_id="srv-1")
        ),
        server_context_provider=SimpleNamespace(get_active_context=context_read),
        **extra,
    )


def _server_context() -> SimpleNamespace:
    return SimpleNamespace(auth_token="opaque-token", credential_source="keyring")


async def _assert_loop_stays_live_while_blocked(task, read: _BlockingRead) -> None:
    """The loop keeps turning while the blocked read holds a worker thread."""
    deadline = time.monotonic() + 5.0
    while read.worker_thread_calls == 0 and read.main_thread_calls == 0:
        assert time.monotonic() < deadline, "the scope read never started"
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.05)  # would never return if the loop were blocked
    assert not task.done()
    assert read.main_thread_calls == 0


@pytest.mark.asyncio
async def test_manual_sync_run_resolves_principal_scope_off_the_ui_thread():
    read = _BlockingRead(_server_context())
    calls: list[dict[str, object]] = []

    class Control:
        async def run_once(self, **kwargs):
            calls.append(kwargs)
            raise RuntimeError("stop after capturing the scope")

    screen = SettingsScreen(
        _server_mode_app(read, manual_sync_control_service=Control())
    )
    screen._apply_manual_sync_rows = lambda rows: None

    task = asyncio.ensure_future(screen._run_manual_sync_once())
    try:
        await _assert_loop_stays_live_while_blocked(task, read)
    finally:
        read.release.set()
    await asyncio.wait_for(task, 5.0)

    assert calls == [
        {
            "server_profile_id": "srv-1",
            "authenticated_principal_id": event_principal_id_from_active_context(
                _server_context()
            ),
            "workspace_scope": None,
        }
    ]


@pytest.mark.asyncio
async def test_notes_adoption_resolves_principal_scope_off_the_ui_thread():
    read = _BlockingRead(_server_context())
    calls: list[dict[str, object]] = []

    class Control:
        def resolve_notes_organization_adoption(self, **kwargs):
            calls.append(kwargs)
            return False

    screen = SettingsScreen(
        _server_mode_app(read, manual_sync_control_service=Control())
    )
    screen._apply_manual_sync_rows = lambda rows: None

    task = asyncio.ensure_future(
        screen._resolve_notes_adoption("merge", "review-1", None)
    )
    try:
        await _assert_loop_stays_live_while_blocked(task, read)
    finally:
        read.release.set()
    await asyncio.wait_for(task, 5.0)

    assert len(calls) == 1
    assert calls[0]["server_profile_id"] == "srv-1"
    assert calls[0]["authenticated_principal_id"] == (
        event_principal_id_from_active_context(_server_context())
    )
    assert calls[0]["review_id"] == "review-1"


# --- Library server-scope resolution ----------------------------------------


def _library_screen(app_instance: SimpleNamespace):
    from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

    return LibraryScreen(app_instance)


def test_library_admission_key_sync_half_never_reads_the_auth_context():
    read = _BlockingRead(_server_context())
    read.release.set()  # a regression must fail the assertion, not hang
    screen = _library_screen(_server_mode_app(read))

    key = screen._library_onboarding_admission_key()

    assert read.main_thread_calls == 0
    assert key[2:5] == ("server", "srv-1", None)


@pytest.mark.asyncio
async def test_library_admission_key_resolves_principal_off_the_ui_thread():
    read = _BlockingRead(_server_context())
    screen = _library_screen(_server_mode_app(read))

    task = asyncio.ensure_future(screen._resolve_library_onboarding_admission_key())
    try:
        await _assert_loop_stays_live_while_blocked(task, read)
    finally:
        read.release.set()
    key = await asyncio.wait_for(task, 5.0)

    assert key[2:5] == (
        "server",
        "srv-1",
        event_principal_id_from_active_context(_server_context()),
    )


# --- Settings > Image Gen / Video Gen panels ---------------------------------


class _BlockingLoader(_BlockingRead):
    """Wraps a real config loader (which resolves secrets via the keyring)."""

    def __init__(self, real) -> None:
        super().__init__(None)
        self.real = real

    def __call__(self, *args: object, **kwargs: object) -> object:
        super().__call__()
        return self.real(*args, **kwargs)

    @property
    def calls(self) -> int:
        return self.main_thread_calls + self.worker_thread_calls


@pytest.mark.asyncio
async def test_gen_panels_without_preloaded_config_render_a_pending_state():
    from textual.app import App

    import tldw_chatbook.Widgets.settings_image_gen_panel as image_panel_module
    import tldw_chatbook.Widgets.settings_video_gen_panel as video_panel_module

    # The panels no longer hold a loader at all: config arrives from the
    # screen's off-thread load, so compose cannot reach the keyring.
    assert not hasattr(image_panel_module, "get_image_generation_config")
    assert not hasattr(video_panel_module, "get_video_generation_config")

    class Host(App):
        def compose(self):
            yield image_panel_module.ImageGenSettingsPanel(id="image")
            yield video_panel_module.VideoGenSettingsPanel(id="video")

    async with Host().run_test() as pilot:
        await pilot.pause()
        assert pilot.app.query("#settings-imagegen-loading")
        assert pilot.app.query("#settings-videogen-loading")


async def _release_and_wait(pilot, loader: _BlockingLoader, predicate, message: str):
    loader.release.set()
    await _wait_until(pilot, predicate, message)
    loader.release.clear()


@pytest.mark.asyncio
@private_profile_test
async def test_image_gen_panel_loads_config_off_the_ui_thread(request, monkeypatch):
    import tldw_chatbook.UI.Screens.settings_screen as settings_screen_module
    from tldw_chatbook.UI.Screens.settings_image_gen_defaults import (
        ImageGenProbeResult,
    )

    loader = _BlockingLoader(settings_screen_module.get_image_generation_config)
    monkeypatch.setattr(settings_screen_module, "get_image_generation_config", loader)
    probes: list[dict[str, str]] = []

    def fake_probe(backend_id, form_values, secret):
        probes.append(dict(form_values))
        return ImageGenProbeResult(ok=True, badge="Reachable")

    monkeypatch.setattr(settings_screen_module, "image_gen_probe_backend", fake_probe)
    after_clear = _BlockingLoader(
        settings_screen_module.image_gen_key_source_after_clear
    )
    monkeypatch.setattr(
        settings_screen_module, "image_gen_key_source_after_clear", after_clear
    )
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    def loaded() -> bool:
        return bool(screen.query("#settings-imagegen-default_backend"))

    try:
        async with host.run_test(size=(190, 55)) as pilot:
            screen = _active_destination_screen(host)
            # Open: the category renders a pending panel while config loads.
            await _select_settings_category(
                screen, pilot, "image_generation", selector="#settings-imagegen-loading"
            )
            assert not loaded()
            await _release_and_wait(pilot, loader, loaded, "Image Gen never loaded")

            # Test: the probe's effective-value fallbacks load off-thread.
            before = loader.calls
            screen._handle_image_gen_test("swarmui")
            await pilot.pause(0.05)
            assert loader.calls == before + 1 and not probes
            await _release_and_wait(
                pilot, loader, lambda: bool(probes), "probe never ran"
            )
            assert probes[0]["base_url"]  # untouched field fell back to effective

            # Clear: the after-Clear key source is resolved off-thread too.
            screen._handle_image_gen_clear("openrouter", "api_key")
            await pilot.pause(0.05)

            def key_source_text() -> str:
                return str(
                    screen.query_one(
                        "#settings-imagegen-key-source-openrouter"
                    ).render()
                )

            assert "checking" in key_source_text()
            await _release_and_wait(
                pilot,
                after_clear,
                lambda: "checking" not in key_source_text(),
                "after-Clear key source never resolved",
            )
            assert after_clear.main_thread_calls == 0

            # Revert: returns while the reload is still blocked.
            before = loader.calls
            await asyncio.wait_for(screen._handle_image_gen_revert(), 2.0)
            await _wait_until(pilot, lambda: loader.calls == before + 1, "no reload")
            await _release_and_wait(
                pilot, loader, loaded, "Image Gen never reloaded after revert"
            )

            # Save: the post-save reload also runs off the UI thread.
            before = loader.calls
            screen._handle_image_gen_save()
            await _wait_until(pilot, lambda: loader.calls == before + 1, "no reload")
            await pilot.pause(0.05)
            await _release_and_wait(
                pilot,
                loader,
                lambda: "Image Gen defaults saved." in _visible_text(screen),
                "save never finished",
            )
            assert loader.main_thread_calls == 0
    finally:
        loader.release.set()


@pytest.mark.asyncio
@private_profile_test
async def test_video_gen_panel_loads_config_off_the_ui_thread(request, monkeypatch):
    import tldw_chatbook.UI.Screens.settings_screen as settings_screen_module

    loader = _BlockingLoader(settings_screen_module.get_video_generation_config)
    monkeypatch.setattr(settings_screen_module, "get_video_generation_config", loader)
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    def loaded() -> bool:
        return bool(screen.query("#settings-videogen-default_backend"))

    try:
        async with host.run_test(size=(190, 55)) as pilot:
            screen = _active_destination_screen(host)
            await _select_settings_category(
                screen, pilot, "video_generation", selector="#settings-videogen-loading"
            )
            assert not loaded()
            await _release_and_wait(pilot, loader, loaded, "Video Gen never loaded")

            before = loader.calls
            await asyncio.wait_for(screen._handle_video_gen_revert(), 2.0)
            await _wait_until(pilot, lambda: loader.calls == before + 1, "no reload")
            await _release_and_wait(
                pilot, loader, loaded, "Video Gen never reloaded after revert"
            )

            before = loader.calls
            screen._handle_video_gen_save()
            await _wait_until(pilot, lambda: loader.calls == before + 1, "no reload")
            await pilot.pause(0.05)
            await _release_and_wait(
                pilot,
                loader,
                lambda: "Video Gen defaults saved." in _visible_text(screen),
                "save never finished",
            )
            assert loader.main_thread_calls == 0
    finally:
        loader.release.set()


# --- Stale callbacks and load failures (Qodo review on #2831) ---------------


def _screen_that_must_not_touch_the_dom() -> SettingsScreen:
    screen = SettingsScreen(SimpleNamespace(app_config={}))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a stale callback touched the DOM")

    screen.query_one = forbidden
    screen._set_static_text = forbidden
    return screen


@pytest.mark.asyncio
async def test_superseded_gen_panel_loads_are_dropped():
    screen = _screen_that_must_not_touch_the_dom()
    stale = object()
    screen._image_gen_load_token = object()
    screen._video_gen_load_token = object()

    await screen._apply_image_gen_panel_config(None, {}, None, stale)
    await screen._apply_video_gen_panel_config(None, {}, None, stale)
    screen._show_image_gen_load_error(stale)
    screen._show_video_gen_load_error(stale)


def test_superseded_clear_key_source_results_are_dropped():
    screen = _screen_that_must_not_touch_the_dom()
    stale = object()
    screen._image_gen_key_source_tokens = {"openrouter": object()}
    screen._video_gen_key_source_tokens = {}

    screen._apply_image_gen_key_source("openrouter", "api_key", "keyring", stale)
    screen._apply_video_gen_key_source("minimax", "keyring", stale)


def test_superseded_skill_trust_results_are_dropped():
    screen = _screen_that_must_not_touch_the_dom()
    screen._skill_trust_token = object()

    screen._apply_skill_trust_status({"trust_status": "trusted"}, object())


@pytest.mark.asyncio
@private_profile_test
async def test_gen_panel_config_load_failure_is_reported_not_fatal(
    request, monkeypatch
):
    import tldw_chatbook.UI.Screens.settings_screen as settings_screen_module

    def broken(*_args, **_kwargs):
        raise RuntimeError("config exploded")

    monkeypatch.setattr(settings_screen_module, "get_image_generation_config", broken)
    monkeypatch.setattr(settings_screen_module, "get_video_generation_config", broken)
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        screen = _active_destination_screen(host)
        for category, label in (
            ("image_generation", "Image Gen"),
            ("video_generation", "Video Gen"),
        ):
            await _select_settings_category(screen, pilot, category)
            await _wait_until(
                pilot,
                lambda label=label: f"{label} settings could not be loaded"
                in _visible_text(screen),
                f"{label} load failure was not reported",
            )
        # The worker raise would exit the app (exit_on_error) with this set.
        assert app._exception is None, app._exception
        assert app.return_code is None

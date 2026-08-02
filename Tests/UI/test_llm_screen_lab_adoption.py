"""Models' adoption of the Lab frame, and its rail lift."""

from __future__ import annotations


import pytest
from textual.widgets import Button, Static

from tldw_chatbook.config import get_cli_setting as _real_get_cli_setting
from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow
from tldw_chatbook.UI.Screens.llm_screen import LLMScreen
from Tests.UI.app_factory import _build_test_app


@pytest.fixture(autouse=True)
def _deterministic_models_mount(monkeypatch):
    """Neutralise the splash race and live network call this file's
    press/pause sequences can hit. Same rationale as the identically named
    fixture in ``test_lab_frame_mode_keys.py``: ``SplashScreen`` starts a
    real 1.5s timer that can push a competing screen mid-test, and
    The ``HuggingFaceAPI.search_models`` stub that used to sit here is gone:
    the browse now waits for the Download Models view to be activated
    (task-887), so mounting Models reaches no network at all.

    Args:
        monkeypatch: pytest's monkeypatch fixture; reverts both patches
            automatically at the end of each test.
    """

    def fake_get_cli_setting(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        return _real_get_cli_setting(section, key, default)

    monkeypatch.setattr("tldw_chatbook.app.get_cli_setting", fake_get_cli_setting)


async def _models_screen(pilot_app):
    screen = LLMScreen(pilot_app)
    await pilot_app.push_screen(screen)
    return screen


def _app():
    """Build the test app.

    No CSS bundle: every assertion here is behavioural (class membership,
    reactive values, chip text), not rendered styling. Rail-row styling is
    asserted in test_lab_workbench.py against a class-level CSS_PATH -- a
    post-construction `app.CSS_PATH = ...` would silently do nothing, since
    App.__init__ reads CSS_PATH once at construction.
    """
    return _build_test_app()


def _rail_rows(screen):
    return list(screen.query(".lab-rail-row").results(Button))


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
            "remote",
            "download-models",
        ]


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
    """Curated progress remains visible in Installed and in the Lab status row."""
    from unittest.mock import MagicMock

    from tldw_chatbook.Model_Artifacts.acquisition import AcquisitionProgress
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef
    from tldw_chatbook.UI.Screens.model_curated_view import CuratedView
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView
    from tldw_chatbook.Widgets.ModelArtifacts import (
        InstallProgressed,
        InstallStatusChanged,
    )

    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        await pilot.pause()
        await pilot.pause()
        window = screen.query_one(LLMManagementWindow)
        curated = window.query_one(CuratedView)
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

        curated.post_message(InstallStatusChanged(reference, active=True))
        curated.post_message(InstallProgressed(progress))
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
        curated.post_message(
            InstallStatusChanged(reference, active=False, succeeded=True)
        )
        await pilot.pause()

        installed.ensure_loaded.assert_called_once_with(force=True)
        assert "Model install: idle" in str(chip.renderable)


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
async def test_mounting_models_reaches_no_network_until_the_view_is_opened(monkeypatch):
    """Opening Models must not call huggingface.co (task-887).

    `ModelSearchWidget` used to `call_after_refresh(self._initial_browse)`
    from `on_mount`, and it lives inside `llm-view-download-models`, which
    `LLMManagementWindow.compose()` builds eagerly -- so every visit to this
    screen fired a live request for users who never open Download Models.

    Counting calls is the oracle. Asserting the results list is empty would
    pass whether the request was skipped or merely returned nothing.
    """
    from tldw_chatbook.LLM_Calls.huggingface_api import HuggingFaceAPI

    calls: list[int] = []

    async def counted(self, *args, **kwargs):
        calls.append(1)
        return []

    monkeypatch.setattr(HuggingFaceAPI, "search_models", counted)

    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        await pilot.pause()
        await pilot.pause()
        assert calls == [], "mounting Models reached the network"

        window = screen.llm_window
        assert window is not None
        window.active_view = "download-models"
        await pilot.pause()
        await pilot.pause()
        assert len(calls) == 1, "opening Download Models did not browse"

        window.active_view = "llama-cpp"
        await pilot.pause()
        window.active_view = "download-models"
        await pilot.pause()
        await pilot.pause()
        assert len(calls) == 1, "re-opening the view browsed again"


@pytest.mark.asyncio
async def test_pressing_remote_still_waits_for_explicit_search(monkeypatch):
    """Remote activation itself must remain metadata-I/O free."""
    from tldw_chatbook.Model_Artifacts.remote_huggingface import (
        HuggingFaceRemoteAdapter,
    )
    from tldw_chatbook.UI.Screens.model_remote_view import RemoteView

    calls: list[str] = []

    async def counted_search(self, query, *, token=None):
        calls.append("search")
        return ()

    async def counted_resolve(self, repository, *, token=None):
        calls.append("resolve")
        raise AssertionError("Remote resolve ran before Search")

    monkeypatch.setattr(HuggingFaceRemoteAdapter, "search", counted_search)
    monkeypatch.setattr(HuggingFaceRemoteAdapter, "resolve", counted_resolve)

    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        await pilot.pause()
        await pilot.pause()
        remote_row = next(
            row for row in _rail_rows(screen) if row.lab_view_key == "remote"
        )

        remote_row.press()
        await pilot.pause()
        await pilot.pause()

        window = screen.query_one(LLMManagementWindow)
        assert window.active_view == "remote"
        assert window.query_one("#remote-models-view", RemoteView)
        assert calls == []

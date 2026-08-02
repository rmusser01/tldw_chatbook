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
        screen = await _models_screen(app)
        await pilot.pause()
        await pilot.pause()
        window = screen.query_one(LLMManagementWindow)
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
        screen = await _models_screen(app)
        for _ in range(5):
            await pilot.pause()
        window = screen.query_one(LLMManagementWindow)
        curated = window.query_one(CuratedView)

        # Mimics _confirm_curated_install's own setup (bypasses real
        # preflight/registry I/O) -- exercising _provision_curated itself
        # directly, on this test's own event loop rather than a real
        # background thread, so `resume` can pause it deterministically at
        # an exact point. State lives on the SCREEN now (TASK-1803), not
        # on the CuratedView instance -- it must survive the instance
        # being torn down below.
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
        # known progress to it via _hydrate_curated_progress.
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
            "InstalledView's mirroring handler never observed the "
            "post-recompose tick"
        )
        assert fresh_installed._install_active is True


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

    Args:
        monkeypatch: pytest's monkeypatch fixture, used to wrap
            ``CuratedView.apply_progress`` with a call-counting shim;
            reverted automatically after the test.
    """
    from tldw_chatbook.Model_Artifacts.acquisition import AcquisitionProgress
    from tldw_chatbook.Model_Artifacts.service import ArtifactRef
    from tldw_chatbook.UI.Screens.model_curated_view import CuratedView
    from tldw_chatbook.Widgets.ModelArtifacts import InstallProgressed

    calls: list[AcquisitionProgress] = []
    original_apply_progress = CuratedView.apply_progress

    def counting_apply_progress(self, progress):
        calls.append(progress)
        return original_apply_progress(self, progress)

    monkeypatch.setattr(CuratedView, "apply_progress", counting_apply_progress)

    app = _app()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = await _models_screen(app)
        for _ in range(5):
            await pilot.pause()

        reference = ArtifactRef("parakeet-v2", "immutable-revision", "int8")
        progress = AcquisitionProgress("fetch", reference, "encoder.onnx", 1, 2)

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

        button = next(iter(curated.query(".curated-install").results(Button)))
        await pilot.click(button)
        await pilot.pause()
        await pilot.pause()

        for _ in range(20):
            if app.push_screen.called:
                break
            await pilot.pause()
        assert app.push_screen.called, "clicking Install never reached push_screen"

        modal, callback = app.push_screen.call_args[0]
        assert isinstance(modal, ModelInstallModal)
        assert modal.report.root == button.reference
        assert callback == screen._confirm_curated_install
        assert screen._model_install_pending_report is modal.report


@pytest.mark.parametrize("operation", ("preflight", "installation"))
def test_curated_install_failures_log_exact_artifact_context(operation, monkeypatch):
    """Worker diagnostics identify the safe immutable artifact reference.

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

    logged = " ".join(str(value) for value in fake_logger.error.call_args.args)
    assert reference.artifact_id in logged
    assert reference.revision in logged
    assert reference.variant in logged


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
    view.cancel_pending_install.assert_called_once_with()


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

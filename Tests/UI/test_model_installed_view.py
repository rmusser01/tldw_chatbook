"""Focused tests for the managed-model Installed view."""

import asyncio
import inspect
import threading
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from textual.app import App, ComposeResult

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.css.query import NoMatches
from textual.screen import Screen
from textual.widgets import Button, Static

from tldw_chatbook.app import TldwCli
from tldw_chatbook.Model_Artifacts.service import (
    ArtifactDiskUsage,
    ArtifactRef,
    InstalledArtifact,
    LocalGGUFImportProgress,
    LocalGGUFImportResult,
    ModelArtifactService,
)

_MODEL_IMPORT_SELECTOR = ".model-import"


class _InstalledApp(ConsolidatedCSSApp):
    def __init__(self, view) -> None:
        self.view = view
        super().__init__()

    def compose(self) -> ComposeResult:
        yield self.view


class _StyledInstalledApp(_InstalledApp):
    """Installed-view harness using the exact production stylesheet bundle."""

    CSS_PATH = TldwCli.CSS_PATH


def _fake_never_cancelled() -> bool:
    return False


def _fake_ignore_progress(_event: LocalGGUFImportProgress) -> None:
    return None


class _ImportServiceFake:
    """Discriminating import seam with the production service's public shape."""

    def __init__(
        self,
        import_impl: Callable[
            [
                Path,
                Callable[[], bool],
                Callable[[LocalGGUFImportProgress], None],
            ],
            LocalGGUFImportResult,
        ],
        *,
        installed: tuple[InstalledArtifact, ...] = (),
        activation_error: BaseException | None = None,
        activation_impl: Callable[[ArtifactRef], ArtifactRef] | None = None,
    ) -> None:
        self.import_impl = import_impl
        self.installed = installed
        self.activation_error = activation_error
        self.activation_impl = activation_impl
        self.import_sources: list[Path] = []
        self.activation_calls: list[ArtifactRef] = []
        self.import_entered = threading.Event()
        self.activation_finished = threading.Event()
        self.inventory_reads = 0

    def import_local_gguf(
        self,
        source_file: Path,
        *,
        cancelled: Callable[[], bool] = _fake_never_cancelled,
        progress: Callable[[LocalGGUFImportProgress], None] = _fake_ignore_progress,
    ) -> LocalGGUFImportResult:
        self.import_sources.append(source_file)
        self.import_entered.set()
        return self.import_impl(source_file, cancelled, progress)

    def activate(self, root_reference: ArtifactRef) -> ArtifactRef:
        self.activation_calls.append(root_reference)
        try:
            if self.activation_error is not None:
                raise self.activation_error
            if self.activation_impl is not None:
                return self.activation_impl(root_reference)
            return root_reference
        finally:
            self.activation_finished.set()

    def list_installed(self) -> tuple[InstalledArtifact, ...]:
        self.inventory_reads += 1
        return self.installed

    def disk_usage(self) -> ArtifactDiskUsage:
        return ArtifactDiskUsage(0, 0, 64 * 1024 * 1024)


def _signature_contract(callable_object) -> tuple[tuple[str, object, bool], ...]:
    """Normalize the public parameter contract without comparing function objects."""
    return tuple(
        (
            name,
            parameter.kind,
            parameter.default is inspect.Parameter.empty,
        )
        for name, parameter in inspect.signature(callable_object).parameters.items()
    )


def test_import_service_fake_matches_real_public_signatures() -> None:
    """The fake cannot silently bless a wrong positional or keyword call shape."""
    assert _signature_contract(_ImportServiceFake.import_local_gguf) == (
        ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD, True),
        ("source_file", inspect.Parameter.POSITIONAL_OR_KEYWORD, True),
        ("cancelled", inspect.Parameter.KEYWORD_ONLY, False),
        ("progress", inspect.Parameter.KEYWORD_ONLY, False),
    )
    assert _signature_contract(
        _ImportServiceFake.import_local_gguf
    ) == _signature_contract(ModelArtifactService.import_local_gguf)
    assert _signature_contract(_ImportServiceFake.activate) == _signature_contract(
        ModelArtifactService.activate
    )


def _import_reference(character: str = "a") -> ArtifactRef:
    return ArtifactRef(
        f"local-gguf-{character * 16}",
        f"sha256-{character * 64}",
        "filetype-7",
    )


def _unmanaged_inventory(source: Path):
    from tldw_chatbook.UI.Screens.model_browser_state import (
        UnmanagedRow,
        inventory_rows,
    )

    return inventory_rows(
        (),
        ArtifactDiskUsage(0, 0, 64 * 1024 * 1024),
        (UnmanagedRow(source, source.stat().st_size),),
    )


async def _wait_until(pilot, predicate, *, timeout_seconds: float = 10.0) -> None:
    """Pump Textual until a cross-thread observation becomes true.

    Args:
        pilot: Mounted Textual test pilot used to process deferred work.
        predicate: Zero-argument condition that signals completion.
        timeout_seconds: Maximum wall-clock time to wait.

    Raises:
        AssertionError: If the condition does not settle before the timeout.
    """
    try:
        async with asyncio.timeout(timeout_seconds):
            while not predicate():
                await pilot.pause()
    except TimeoutError:
        assert predicate()


def _rendered_static_text(view) -> str:
    return "\n".join(str(widget.renderable) for widget in view.query(Static))


def _painted_screen_text(app: App) -> str:
    """Return text emitted by the real screen compositor."""
    return "".join(
        segment.text
        for strip in app.screen._compositor.render_strips()
        for segment in strip
    )


def _painted_style_of_text(app: App, region, needle: str):
    strips = app.screen._compositor.render_strips()
    for y in range(region.y, region.bottom):
        cursor = 0
        for segment in strips[y]:
            end = cursor + segment.cell_length
            if (
                max(cursor, region.x) < min(end, region.right)
                and needle in segment.text
            ):
                return segment.style
            cursor = end
    return None


def _contrast(first, second) -> float:
    def luminance(color) -> float:
        triplet = color.get_truecolor()

        def channel(value: int) -> float:
            value /= 255
            return (
                value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
            )

        return sum(
            weight * channel(value)
            for weight, value in zip(
                (0.2126, 0.7152, 0.0722),
                (triplet.red, triplet.green, triplet.blue),
                strict=True,
            )
        )

    lighter, darker = sorted((luminance(first), luminance(second)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


@pytest.mark.asyncio
async def test_installed_view_performs_no_io_at_compose_time(tmp_path: Path) -> None:
    """Eagerly mounted model views stay idle until their rail row is selected."""
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    service_factory = MagicMock()
    view = InstalledView(service_factory=service_factory, legacy_dir=tmp_path)
    app = _InstalledApp(view)
    async with app.run_test() as pilot:
        await pilot.pause()

    service_factory.assert_not_called()


@pytest.mark.asyncio
async def test_models_host_lazily_wires_parakeet_activation_and_deletion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Models host binds its app service on the Textual UI thread."""
    import threading

    from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow
    from tldw_chatbook.UI.Screens.llm_screen import LLMScreen
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView
    from tldw_chatbook.config import get_cli_setting as real_get_cli_setting
    from Tests.UI.app_factory import _build_test_app

    root = ArtifactRef("parakeet-v2", "immutable-revision", "int8")
    vad = ArtifactRef("silero-vad", "immutable-revision", "f32")
    ui_thread = threading.get_ident()
    lifecycle_threads: list[tuple[str, int]] = []
    recycled: list[tuple[str, str, str]] = []

    class _Source:
        def __init__(self) -> None:
            self.activated: list[ArtifactRef] = []
            self.deletion_checks: list[ArtifactRef] = []

        def release_scopes_except(self, scopes: set[str]) -> None:
            assert scopes == set()
            lifecycle_threads.append(("release", threading.get_ident()))

        def records(self) -> dict[object, object]:
            return {}

        def close(self) -> None:
            return None

        def on_root_activated(self, reference: ArtifactRef) -> None:
            self.activated.append(reference)

        def may_delete(self, reference: ArtifactRef) -> str:
            self.deletion_checks.append(reference)
            return "Managed dependency is in use."

    source = _Source()

    class _Core:
        def activate(self, reference: ArtifactRef) -> ArtifactRef:
            return reference

    def no_splash(section, key=None, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        return real_get_cli_setting(section, key, default)

    monkeypatch.setattr("tldw_chatbook.app.get_cli_setting", no_splash)
    monkeypatch.setattr(
        LLMManagementWindow,
        "_ollama_api_available",
        lambda _window: False,
    )
    app = _build_test_app()

    class ExecutorDouble:
        def recycle_idle_managed_reference(
            self,
            reference: tuple[str, str, str],
        ) -> bool:
            recycled.append(reference)
            return True

        def close(self) -> None:
            return None

    app._local_stt_executor = ExecutorDouble()
    monkeypatch.setattr(
        app,
        "_create_local_stt_executor",
        MagicMock(side_effect=AssertionError("Models must not create an executor")),
    )

    def create_source_service() -> _Source:
        lifecycle_threads.append(("construct", threading.get_ident()))
        return source

    def add_listener(_listener) -> None:
        lifecycle_threads.append(("listener", threading.get_ident()))

    def read_jobs() -> tuple[object, ...]:
        lifecycle_threads.append(("read", threading.get_ident()))
        return ()

    monkeypatch.setattr(app, "_create_parakeet_source_service", create_source_service)
    monkeypatch.setattr(app.library_ingest_jobs, "add_listener", add_listener)
    monkeypatch.setattr(app.library_ingest_jobs, "jobs", read_jobs)
    async with app.run_test(size=(235, 52)) as pilot:
        await pilot.pause()
        screen = LLMScreen(app)
        await app.push_screen(screen)
        for _ in range(6):
            await pilot.pause()

        screen.query_one(LLMManagementWindow)
        view = screen.query_one(InstalledView)
        assert [name for name, _thread in lifecycle_threads] == [
            "construct",
            "listener",
            "read",
            "release",
        ]
        assert all(thread == ui_thread for _name, thread in lifecycle_threads)

        ensure_after_mount = MagicMock(
            side_effect=AssertionError("activation must use the bound source service")
        )
        monkeypatch.setattr(
            app,
            "_ensure_parakeet_source_service",
            ensure_after_mount,
        )

        view._service_factory = _Core
        view._legacy_dir = tmp_path
        view._apply_lifecycle_result = MagicMock()
        await view._activate_model(root).wait()
        view._apply_lifecycle_result.assert_called_once_with("activate", None)
        assert source.activated == [root]
        ensure_after_mount.assert_not_called()

        assert view._may_delete(vad) == "Managed dependency is in use."
        assert source.deletion_checks == [vad]
        assert view._recycle_idle(vad) is True
        assert recycled == [("silero-vad", "immutable-revision", "f32")]
        ensure_after_mount.assert_not_called()


def test_unmanaged_scan_is_bounded_and_labels_supported_model_files(
    tmp_path: Path,
) -> None:
    """Legacy GGUF/ONNX files stay visible without an unbounded result set."""
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    for index in range(3):
        (tmp_path / f"model-{index}.gguf").write_bytes(b"x" * (1024 * 1024 + 1))
    (tmp_path / "ignore.txt").write_text("not a model")

    rows = InstalledView.scan_unmanaged(tmp_path, limit=2)

    assert len(rows) == 2
    assert all(row.path.suffix == ".gguf" for row in rows)


def test_unmanaged_scan_validates_root_before_walking(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Configured legacy roots pass the shared path-safety boundary first."""
    from tldw_chatbook.UI.Screens import model_installed_view as module

    walk = MagicMock()
    monkeypatch.setattr(module.os, "walk", walk)

    with pytest.raises(ValueError, match="dangerous pattern"):
        module.InstalledView.scan_unmanaged(tmp_path / "../..")

    walk.assert_not_called()


def test_scan_unmanaged_excludes_managed_artifacts_root(tmp_path: Path) -> None:
    """The managed subtree is pruned without hiding other outside GGUF files."""
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    legacy_root = tmp_path / "legacy"
    store = ModelArtifactService(legacy_root / "managed")
    managed_payload = store.artifacts_path / "local" / "model.gguf"
    managed_payload.parent.mkdir(parents=True)
    managed_payload.write_bytes(b"x" * 1_048_577)
    external_payload = legacy_root / "outside.gguf"
    external_payload.write_bytes(b"y" * 1_048_577)

    rows = InstalledView.scan_unmanaged(
        legacy_root,
        excluded_root=store.artifacts_path,
    )
    paths = {row.path for row in rows}

    assert managed_payload not in paths
    assert external_payload in paths


@pytest.mark.asyncio
async def test_header_and_unmanaged_row_open_real_gguf_picker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Header and outside-row actions use one real GGUF-only picker contract."""
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView
    from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen
    from tldw_chatbook.Widgets.ModelArtifacts import LocalGGUFImportConsentModal

    source = tmp_path / ("outside-model-" * 5 + ".gguf")
    source.write_bytes(b"x" * 1_048_577)
    service = _ImportServiceFake(
        lambda _source, _cancelled, _progress: LocalGGUFImportResult(
            _import_reference(),
            False,
        )
    )
    view = InstalledView(service_factory=lambda: service, legacy_dir=tmp_path)
    view._loaded = True
    view._rows = _unmanaged_inventory(source)
    app = _StyledInstalledApp(view)
    pushed: list[tuple[Screen, Callable]] = []

    async with app.run_test(size=(80, 24)) as pilot:
        monkeypatch.setattr(
            app,
            "push_screen",
            lambda screen, callback=None: pushed.append((screen, callback)),
        )
        header = view.query_one("#installed-models-import-gguf", Button)
        row_action = view.query_one(_MODEL_IMPORT_SELECTOR, Button)
        for action in (header, row_action):
            assert action in app.screen._compositor.visible_widgets
            assert action.region.right <= app.size.width
            assert action.region.bottom <= app.size.height

        header.focus()
        await pilot.press("enter")
        await pilot.pause()
        picker, picked = pushed.pop(0)
        assert isinstance(picker, EnhancedFileOpen)
        assert picker.filters is not None
        assert picker.filters[0](Path("model.gguf")) is True
        assert picker.filters[0](Path("model.GGUF")) is True
        assert picker.filters[0](Path("model.bin")) is False
        assert len(pushed) == 0

        picked(None)
        assert len(pushed) == 0
        await pilot.pause()
        view._header_import_pressed()
        _picker, picked = pushed.pop(0)
        picked(source)
        consent, decided = pushed.pop(0)
        assert isinstance(consent, LocalGGUFImportConsentModal)
        assert consent.source == source
        decided(False)
        await pilot.pause()

        row_action = view.query_one(_MODEL_IMPORT_SELECTOR, Button)
        row_action.focus()
        await pilot.press("enter")
        await pilot.pause()
        picker, picked = pushed.pop(0)
        assert isinstance(picker, EnhancedFileOpen)
        assert picker.filters is not None
        assert picker.filters[0](Path("replacement.gguf")) is True
        assert picker.filters[0](Path("replacement.onnx")) is False
        picked(source)
        consent, decided = pushed.pop(0)
        assert isinstance(consent, LocalGGUFImportConsentModal)
        assert consent.source == source
        decided(False)

    assert service.import_sources == []
    assert service.activation_calls == []


@pytest.mark.asyncio
async def test_declined_consent_performs_no_service_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Declining the transient copy consent leaves the outside row untouched."""
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView
    from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen
    from tldw_chatbook.Widgets.ModelArtifacts import LocalGGUFImportConsentModal

    source = tmp_path / "outside.gguf"
    source.write_bytes(b"x" * 1_048_577)
    service = _ImportServiceFake(
        lambda _source, _cancelled, _progress: LocalGGUFImportResult(
            _import_reference(),
            False,
        )
    )
    view = InstalledView(service_factory=lambda: service, legacy_dir=tmp_path)
    view._loaded = True
    view._rows = _unmanaged_inventory(source)
    app = _InstalledApp(view)
    pushed: list[tuple[Screen, Callable]] = []

    async with app.run_test() as pilot:
        monkeypatch.setattr(
            app,
            "push_screen",
            lambda screen, callback=None: pushed.append((screen, callback)),
        )
        await pilot.click(_MODEL_IMPORT_SELECTOR)
        await pilot.pause()
        picker, picked = pushed.pop()
        assert isinstance(picker, EnhancedFileOpen)
        picked(source)
        consent, decided = pushed.pop()
        assert isinstance(consent, LocalGGUFImportConsentModal)
        decided(False)
        await pilot.pause()

        assert view._pending_import_path is None
        assert len(view.query(_MODEL_IMPORT_SELECTOR)) == 1
        assert source.name in _rendered_static_text(view)

    assert service.import_sources == []
    assert service.activation_calls == []


@pytest.mark.asyncio
async def test_picker_reserves_lane_and_blocks_second_selection_and_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Picker ownership disables every competing Installed-view mutation."""
    from Tests.Model_Artifacts.test_acquisition_types import make_descriptor
    from tldw_chatbook.UI.Screens.model_browser_state import inventory_rows
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    source = tmp_path / "outside.gguf"
    source.write_bytes(b"x" * 1_048_577)
    reference = _import_reference()
    managed = InstalledArtifact(
        path=tmp_path / "managed-model",
        descriptor=replace(
            make_descriptor(),
            reference=reference,
            precision=reference.variant,
        ),
        ready=True,
        active=False,
        error=None,
    )
    service = _ImportServiceFake(
        lambda _source, _cancelled, _progress: LocalGGUFImportResult(
            reference,
            False,
        )
    )
    view = InstalledView(service_factory=lambda: service, legacy_dir=tmp_path)
    view._loaded = True
    view._rows = inventory_rows(
        (managed,),
        ArtifactDiskUsage(0, 0, 64 * 1024 * 1024),
        (),
    ) + _unmanaged_inventory(source)
    app = _StyledInstalledApp(view)
    pushed: list[tuple[Screen, Callable]] = []

    async with app.run_test(size=(80, 24)) as pilot:
        monkeypatch.setattr(
            app,
            "push_screen",
            lambda screen, callback=None: pushed.append((screen, callback)),
        )
        await pilot.click("#installed-models-import-gguf")
        await pilot.pause()

        assert getattr(view, "_import_selecting", False) is True
        assert len(pushed) == 1
        for selector in (
            "#installed-models-refresh",
            "#installed-models-repair",
            "#installed-models-import-gguf",
            _MODEL_IMPORT_SELECTOR,
            ".model-activate",
            ".model-delete",
        ):
            assert view.query_one(selector, Button).disabled is True

        view._header_import_pressed()
        assert len(pushed) == 1


@pytest.mark.asyncio
async def test_consent_fails_closed_if_install_owns_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Consent cannot cross an app-level install that claimed the store later."""
    from tldw_chatbook.UI.Screens import model_installed_view as module

    InstalledView = module.InstalledView

    source = tmp_path / "PRIVATE-SELECTION-RACE.gguf"
    source.write_bytes(b"x" * 1_048_577)
    service = _ImportServiceFake(
        lambda _source, _cancelled, _progress: LocalGGUFImportResult(
            _import_reference(),
            False,
        )
    )
    view = InstalledView(service_factory=lambda: service, legacy_dir=tmp_path)
    view._loaded = True
    view._rows = _unmanaged_inventory(source)
    app = _InstalledApp(view)
    pushed: list[tuple[Screen, Callable]] = []
    logs: list[str] = []
    sink_id = module.logger.add(lambda message: logs.append(str(message)))

    try:
        async with app.run_test() as pilot:
            monkeypatch.setattr(
                app,
                "push_screen",
                lambda screen, callback=None: pushed.append((screen, callback)),
            )
            await pilot.click("#installed-models-import-gguf")
            picker, picked = pushed.pop()
            assert isinstance(picker, Screen)
            picked(source)
            _consent, decided = pushed.pop()

            view.set_install_state(None, active=True)
            await pilot.pause()
            decided(True)
            await pilot.pause()

            rendered = _rendered_static_text(view)
            notifications = " ".join(
                notification.message for notification in app._notifications
            )
            assert service.import_sources == []
            assert service.activation_calls == []
            assert view._pending_import_path is None
            assert getattr(view, "_import_selecting", False) is False
            assert str(source) not in rendered
            assert str(source) not in notifications
            assert str(source) not in "".join(logs)
    finally:
        module.logger.remove(sink_id)


@pytest.mark.asyncio
async def test_decline_releases_selection_lane_and_restores_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ordinary decline restores actions and leaves the outside row available."""
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    source = tmp_path / "outside.gguf"
    source.write_bytes(b"x" * 1_048_577)
    service = _ImportServiceFake(
        lambda _source, _cancelled, _progress: LocalGGUFImportResult(
            _import_reference(),
            False,
        )
    )
    view = InstalledView(service_factory=lambda: service, legacy_dir=tmp_path)
    view._loaded = True
    view._rows = _unmanaged_inventory(source)
    app = _InstalledApp(view)
    pushed: list[tuple[Screen, Callable]] = []

    async with app.run_test() as pilot:
        monkeypatch.setattr(
            app,
            "push_screen",
            lambda screen, callback=None: pushed.append((screen, callback)),
        )
        await pilot.click("#installed-models-import-gguf")
        _picker, picked = pushed.pop()
        picked(source)
        _consent, decided = pushed.pop()
        decided(False)
        await pilot.pause()

        assert getattr(view, "_import_selecting", False) is False
        assert view._pending_import_path is None
        assert view.query_one("#installed-models-refresh", Button).disabled is False
        assert view.query_one("#installed-models-repair", Button).disabled is False
        assert view.query_one("#installed-models-import-gguf", Button).disabled is False
        assert view.query_one(_MODEL_IMPORT_SELECTOR, Button).disabled is False
        assert source.name in _rendered_static_text(view)

        view._header_import_pressed()
        assert len(pushed) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("consent_open", (False, True))
async def test_unmount_invalidates_reserved_selection_lane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    consent_open: bool,
) -> None:
    """Detached picker and consent callbacks cannot regain import ownership."""
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    source = tmp_path / "outside.gguf"
    source.write_bytes(b"x" * 1_048_577)
    service = _ImportServiceFake(
        lambda _source, _cancelled, _progress: LocalGGUFImportResult(
            _import_reference(),
            False,
        )
    )
    view = InstalledView(service_factory=lambda: service, legacy_dir=tmp_path)
    app = _InstalledApp(view)
    pushed: list[tuple[Screen, Callable]] = []

    async with app.run_test() as pilot:
        monkeypatch.setattr(
            app,
            "push_screen",
            lambda screen, callback=None: pushed.append((screen, callback)),
        )
        view._open_import_picker()
        assert getattr(view, "_import_selecting", False) is True
        _picker, picked = pushed.pop()
        callback = picked
        callback_result = source
        if consent_open:
            picked(source)
            _consent, callback = pushed.pop()
            callback_result = True

        generation = view._import_generation
        await view.remove()
        callback(callback_result)
        await pilot.pause()

        assert getattr(view, "_import_selecting", False) is False
        assert view._pending_import_path is None
        assert view._import_generation > generation
        assert service.import_sources == []


@pytest.mark.asyncio
async def test_activation_controls_emit_intents_and_refuse_pending_reentry() -> None:
    """Controls post exact refs and disable both mutations while pending."""
    from tldw_chatbook.Widgets.ModelArtifacts.activation_controls import (
        ActivationRequested,
        DeletionRequested,
        ModelActivationControls,
    )

    reference = ArtifactRef("parakeet-v2", "immutable-revision", "int8")

    class _ControlsApp(ConsolidatedCSSApp):
        def __init__(self) -> None:
            self.messages = []
            super().__init__()

        def compose(self) -> ComposeResult:
            yield ModelActivationControls(reference, active=False, ready=True)

        def on_activation_requested(self, event: ActivationRequested) -> None:
            self.messages.append(event)

        def on_deletion_requested(self, event: DeletionRequested) -> None:
            self.messages.append(event)

    app = _ControlsApp()
    async with app.run_test() as pilot:
        controls = app.query_one(ModelActivationControls)
        await pilot.click(".model-activate")
        await pilot.pause()
        assert isinstance(app.messages[0], ActivationRequested)
        assert app.messages[0].reference == reference

        controls.set_pending(True)
        await pilot.pause()
        assert app.query_one(".model-activate", Button).disabled is True
        assert app.query_one(".model-delete", Button).disabled is True


@pytest.mark.asyncio
async def test_unassigned_controls_omit_activate_and_keep_delete_available() -> None:
    """An unassigned inventory row can be deleted but cannot request activation.

    This fails if disabling activation also removes Delete, or if controls
    render an activation affordance when policy disallows it.
    """
    from tldw_chatbook.Widgets.ModelArtifacts.activation_controls import (
        DeletionRequested,
        ModelActivationControls,
    )

    reference = ArtifactRef("remote-gguf", "immutable-revision", "q4_k_m")

    class _ControlsApp(ConsolidatedCSSApp):
        def __init__(self) -> None:
            self.messages = []
            super().__init__()

        def compose(self) -> ComposeResult:
            yield ModelActivationControls(
                reference,
                active=False,
                ready=True,
                allow_activation=False,
            )

        def on_deletion_requested(self, event: DeletionRequested) -> None:
            self.messages.append(event)

    app = _ControlsApp()
    async with app.run_test() as pilot:
        controls = app.query_one(ModelActivationControls)
        assert len(app.query(".model-activate")) == 0
        assert app.query_one(".model-delete", Button).disabled is False

        controls.set_pending(True)
        await pilot.pause()
        assert app.query_one(".model-delete", Button).disabled is True

        controls.set_pending(False)
        await pilot.click(".model-delete")
        await pilot.pause()

    assert len(app.messages) == 1
    assert isinstance(app.messages[0], DeletionRequested)
    assert app.messages[0].reference == reference


@pytest.mark.asyncio
async def test_mounted_dependency_row_labels_ownership_and_omits_activate(
    tmp_path: Path,
) -> None:
    """The Installed surface exposes dependency ownership without root actions."""
    from dataclasses import replace

    from textual.widgets import Static

    from Tests.Model_Artifacts.test_acquisition_types import make_descriptor
    from tldw_chatbook.Model_Artifacts.service import ArtifactRole, InstalledArtifact
    from tldw_chatbook.UI.Screens.model_browser_state import inventory_rows
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    descriptor = replace(make_descriptor(), role=ArtifactRole.DEPENDENCY)
    rows = inventory_rows(
        (
            InstalledArtifact(
                path=tmp_path / "dependency",
                descriptor=descriptor,
                ready=False,
                active=False,
                error=None,
            ),
        ),
        None,
        (),
    )
    view = InstalledView(service_factory=MagicMock(), legacy_dir=tmp_path)
    app = _InstalledApp(view)

    async with app.run_test() as pilot:
        view._apply_inventory(rows, None, None)
        await pilot.pause()
        text = "\n".join(str(item.renderable) for item in view.query(Static))

        assert "Managed dependency" in text
        assert len(view.query(".model-activate")) == 0
        assert view.query_one(".model-delete", Button).disabled is False


def test_installed_view_refuses_a_second_lifecycle_operation() -> None:
    """Activation/deletion cannot re-enter while hashing or leasing is pending."""
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    view = InstalledView(service_factory=MagicMock(), legacy_dir=Path("/tmp/models"))
    view._operation_reference = ArtifactRef("parakeet-v2", "rev1", "int8")
    view._activate_model = MagicMock()

    view._request_activation(ArtifactRef("parakeet-v2", "rev2", "f32"))

    view._activate_model.assert_not_called()


def test_activation_changes_source_preference_only_after_core_success(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The preference callback follows the real activation boundary."""
    from tldw_chatbook.UI.Screens import model_installed_view as module

    reference = ArtifactRef("parakeet-v2", "immutable-revision", "int8")
    events: list[str] = []

    class _Service:
        def activate(self, activated: ArtifactRef) -> ArtifactRef:
            assert activated == reference
            events.append("activate")
            return activated

    fake_app = MagicMock()
    monkeypatch.setattr(module.InstalledView, "app", property(lambda self: fake_app))
    view = module.InstalledView(
        service_factory=_Service,
        legacy_dir=tmp_path,
        on_root_activated=lambda activated: events.append(
            "prefer-managed" if activated == reference else "wrong-reference"
        ),
    )

    module.InstalledView._activate_model.__wrapped__(view, reference)

    assert events == ["activate", "prefer-managed"]
    fake_app.call_from_thread.assert_called_once_with(
        view._apply_lifecycle_result,
        "activate",
        None,
    )


def test_failed_activation_does_not_change_source_preference(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """A failed core activation leaves the exact source preference untouched."""
    from tldw_chatbook.UI.Screens import model_installed_view as module

    reference = ArtifactRef("parakeet-v2", "immutable-revision", "int8")
    preferred: list[ArtifactRef] = []
    service = MagicMock()
    service.activate.side_effect = RuntimeError("private activation detail")
    fake_app = MagicMock()
    monkeypatch.setattr(module.InstalledView, "app", property(lambda self: fake_app))
    view = module.InstalledView(
        service_factory=lambda: service,
        legacy_dir=tmp_path,
        on_root_activated=preferred.append,
    )

    module.InstalledView._activate_model.__wrapped__(view, reference)

    assert preferred == []
    fake_app.call_from_thread.assert_called_once()
    assert fake_app.call_from_thread.call_args.args[1] == "activate"
    assert fake_app.call_from_thread.call_args.args[2] is not None


def test_lease_blocked_deletion_message_is_specific_and_sanitized() -> None:
    """An active lease is named without surfacing raw internal exception text."""
    from tldw_chatbook.Model_Artifacts.service import ArtifactInUseError
    from tldw_chatbook.UI.Screens.model_installed_view import lifecycle_failure_message

    marker = "RAW-LEASE-DETAIL"
    message = lifecycle_failure_message(
        ArtifactInUseError(marker),
        operation="delete",
    )

    assert "in use" in message
    assert marker not in message


def test_repair_summary_reports_every_reconciliation_outcome(tmp_path: Path) -> None:
    """Repair copy names state/staging cleanup and corruption without paths."""
    from tldw_chatbook.Model_Artifacts.service import ReconcileReport
    from tldw_chatbook.UI.Screens.model_installed_view import (
        reconcile_result_message,
    )

    marker = "PRIVATE-MODEL-PATH"
    report = ReconcileReport(
        readiness_created=2,
        state_removed=3,
        corrupt_artifacts=(tmp_path / marker,),
        staging_entries=(tmp_path / "staged-a", tmp_path / "staged-b"),
        staging_removed=(),
    )

    message = reconcile_result_message(report)

    assert "2 readiness" in message
    assert "3 stale state" in message
    assert "2 staging entries observed" in message
    assert "0 staging entries removed" in message
    assert "1 corrupt model" in message
    assert marker not in message


@pytest.mark.parametrize(
    ("error", "expected"),
    (
        ("path", "read safely"),
        ("parse", "valid GGUF"),
        ("bounds", "valid GGUF"),
        ("version", "supported GGUF"),
        ("integrity", "integrity verification"),
        ("busy", "busy"),
        ("generic", "could not be imported"),
    ),
)
def test_local_import_failure_message_uses_stable_path_free_taxonomy(
    error: str,
    expected: str,
) -> None:
    """Every service/admission category maps to fixed recovery copy."""
    from tldw_chatbook.Model_Artifacts.gguf_admission import (
        GGUFBoundsError,
        GGUFParseError,
        GGUFPathError,
        GGUFVersionError,
    )
    from tldw_chatbook.Model_Artifacts.leases import ArtifactLeaseTimeoutError
    from tldw_chatbook.Model_Artifacts.service import ArtifactIntegrityError
    from tldw_chatbook.UI.Screens.model_installed_view import (
        local_import_failure_message,
    )

    marker = "PRIVATE-ERROR-PATH"
    exceptions = {
        "path": GGUFPathError(marker),
        "parse": GGUFParseError(marker),
        "bounds": GGUFBoundsError(marker),
        "version": GGUFVersionError(marker),
        "integrity": ArtifactIntegrityError(marker),
        "busy": ArtifactLeaseTimeoutError(marker),
        "generic": RuntimeError(marker),
    }

    message = local_import_failure_message(exceptions[error])

    assert expected in message
    assert marker not in message


@pytest.mark.parametrize(
    ("worker_name", "service_method", "worker_args", "log_context"),
    (
        ("_load_inventory", "list_installed", (), ("legacy", "configured")),
        (
            "_activate_model",
            "activate",
            (ArtifactRef("parakeet-v2", "rev", "int8"),),
            ("parakeet-v2", "rev", "int8"),
        ),
        (
            "_delete_model",
            "delete",
            (ArtifactRef("parakeet-v2", "rev", "int8"),),
            (),
        ),
        ("_repair_store", "reconcile", (), ("store", "shared")),
    ),
)
def test_installed_worker_failures_are_logged_and_sanitized(
    monkeypatch,
    tmp_path: Path,
    worker_name: str,
    service_method: str,
    worker_args: tuple,
    log_context: tuple[str, ...],
) -> None:
    """Every background failure retains diagnostics without exposing them in UI."""
    from tldw_chatbook.UI.Screens import model_installed_view as module

    marker = "PRIVATE-WORKER-DETAIL"
    service = MagicMock()
    getattr(service, service_method).side_effect = RuntimeError(marker)
    fake_app = MagicMock()
    fake_logger = MagicMock()
    fake_logger.opt.return_value = fake_logger
    monkeypatch.setattr(module.InstalledView, "app", property(lambda self: fake_app))
    monkeypatch.setattr(module, "logger", fake_logger)
    view = module.InstalledView(service_factory=lambda: service, legacy_dir=tmp_path)

    worker = getattr(module.InstalledView, worker_name).__wrapped__
    worker(view, *worker_args)

    if worker_name == "_delete_model":
        fake_logger.opt.assert_not_called()
    else:
        fake_logger.opt.assert_called_once_with(exception=True)
    fake_logger.error.assert_called_once()
    logged = " ".join(
        str(value) for value in fake_logger.error.call_args.args
    ).casefold()
    assert all(value in logged for value in log_context)
    assert marker not in str(fake_app.call_from_thread.call_args)


def test_deletion_requires_confirmation_before_starting_worker(monkeypatch) -> None:
    """The destructive service call starts only after explicit confirmation."""
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView
    from tldw_chatbook.Widgets.ModelArtifacts.activation_controls import (
        DeletionRequested,
    )
    from tldw_chatbook.Widgets.delete_confirmation_dialog import (
        DeleteConfirmationDialog,
    )

    reference = ArtifactRef("parakeet-v2", "immutable-revision", "int8")
    fake_app = MagicMock()
    monkeypatch.setattr(InstalledView, "app", property(lambda self: fake_app))
    view = InstalledView(service_factory=MagicMock(), legacy_dir=Path("/tmp/models"))
    view.refresh = MagicMock()
    view._delete_model = MagicMock()

    view._deletion_requested(DeletionRequested(reference))

    view._delete_model.assert_not_called()
    dialog, callback = fake_app.push_screen.call_args[0]
    assert isinstance(dialog, DeleteConfirmationDialog)
    callback(True)
    view._delete_model.assert_called_once_with(reference)


def test_audio_cpp_dependency_confirmation_names_keep_consumers_acknowledgement(
    monkeypatch,
) -> None:
    from tldw_chatbook.TTS.audio_cpp_artifact_dependencies import (
        AudioCppArtifactRemovalPreview,
    )
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    reference = ArtifactRef("audio-cpp-model", "a" * 40, "q8_0")
    preview = AudioCppArtifactRemovalPreview(
        reference=reference,
        fingerprint="b" * 64,
        settings_labels=("Guided Settings",),
        profile_labels=("Narrator",),
        assignment_count=2,
        clone_reference_count=1,
        staged_or_live=False,
        generic_lease_blocked=False,
    )
    fake_app = MagicMock()
    monkeypatch.setattr(InstalledView, "app", property(lambda self: fake_app))
    view = InstalledView(service_factory=MagicMock(), legacy_dir=Path("/tmp/models"))

    view._show_delete_confirmation(reference, preview)

    dialog, _callback = fake_app.push_screen.call_args.args
    assert dialog.confirm_label == "Remove package; keep consumers unavailable"
    assert "remain unchanged and become unavailable" in dialog.additional_warning
    assert "/" not in dialog.additional_warning


@pytest.mark.asyncio
async def test_audio_cpp_removal_worker_revalidates_without_self_probe_then_commits(
    monkeypatch,
) -> None:
    from tldw_chatbook.TTS.audio_cpp_artifact_dependencies import (
        AudioCppArtifactLeaseCoordinator,
        AudioCppArtifactRemovalEvidence,
        build_audio_cpp_artifact_removal_preview,
    )
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    reference = ArtifactRef("audio-cpp-model", "a" * 40, "q8_0")
    evidence = AudioCppArtifactRemovalEvidence(
        reference,
        settings_consumers=(("saved", "Guided Settings", "package-1"),),
    )
    fingerprint = build_audio_cpp_artifact_removal_preview(evidence).fingerprint
    events: list[str] = []

    class Authority:
        def commit(self) -> None:
            events.append("commit")

        def close(self) -> None:
            events.append("close")

    class Service:
        def acquire_removal_authority(self, exact: ArtifactRef) -> Authority:
            assert exact == reference
            events.append("authority")
            return Authority()

        def probe_removal_availability(self, _exact: ArtifactRef):
            raise AssertionError("authority revalidation must not self-probe")

    async def collect(exact: ArtifactRef):
        assert exact == reference
        events.append("evidence")
        return evidence

    fake_app = MagicMock()
    fake_app._audio_cpp_artifact_removal_evidence = collect
    coordinator = AudioCppArtifactLeaseCoordinator(
        Service(),
        saved_settings_snapshot=lambda: (),
        catalog_entries=lambda: (),
    )
    fake_app._ensure_audio_cpp_artifact_lease_coordinator = lambda: coordinator
    monkeypatch.setattr(InstalledView, "app", property(lambda self: fake_app))
    view = InstalledView(service_factory=MagicMock(), legacy_dir=Path("/tmp/models"))
    view._apply_lifecycle_result = MagicMock()

    await InstalledView._delete_audio_cpp_model.__wrapped__(
        view,
        reference,
        fingerprint,
    )

    assert events == ["authority", "evidence", "commit", "close"]
    view._apply_lifecycle_result.assert_called_once_with("delete", None)


@pytest.mark.asyncio
async def test_audio_cpp_removal_cancellation_waits_for_commit_before_cleanup(
    monkeypatch,
) -> None:
    from tldw_chatbook.TTS.audio_cpp_artifact_dependencies import (
        AudioCppArtifactLeaseCoordinator,
        AudioCppArtifactRemovalEvidence,
        build_audio_cpp_artifact_removal_preview,
    )
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    reference = ArtifactRef("audio-cpp-model", "a" * 40, "q8_0")
    evidence = AudioCppArtifactRemovalEvidence(reference)
    fingerprint = build_audio_cpp_artifact_removal_preview(evidence).fingerprint
    commit_entered = threading.Event()
    release_commit = threading.Event()
    events: list[str] = []

    class Authority:
        def commit(self) -> None:
            events.append("commit-start")
            commit_entered.set()
            assert release_commit.wait(timeout=3.0)
            events.append("commit-end")

        def close(self) -> None:
            events.append("close")

    class Service:
        def acquire_removal_authority(self, _exact: ArtifactRef) -> Authority:
            return Authority()

    async def collect(_exact: ArtifactRef):
        return evidence

    fake_app = MagicMock()
    fake_app._audio_cpp_artifact_removal_evidence = collect
    coordinator = AudioCppArtifactLeaseCoordinator(
        Service(),
        saved_settings_snapshot=lambda: (),
        catalog_entries=lambda: (),
    )
    fake_app._ensure_audio_cpp_artifact_lease_coordinator = lambda: coordinator
    monkeypatch.setattr(InstalledView, "app", property(lambda self: fake_app))
    view = InstalledView(service_factory=MagicMock(), legacy_dir=Path("/tmp/models"))
    view._apply_lifecycle_result = MagicMock()

    task = asyncio.create_task(
        InstalledView._delete_audio_cpp_model.__wrapped__(
            view,
            reference,
            fingerprint,
        )
    )
    assert await asyncio.to_thread(commit_entered.wait, 1.0)
    task.cancel()
    await asyncio.sleep(0.05)
    assert events == ["commit-start"]
    release_commit.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert events == ["commit-start", "commit-end", "close"]


def test_audio_cpp_removal_cleanup_ownership_is_not_widget_local() -> None:
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    view = InstalledView(service_factory=MagicMock(), legacy_dir=Path("/tmp/models"))

    assert not hasattr(view, "_removal_cleanup_authorities")


def test_audio_cpp_delete_failure_on_app_loop_is_bounded_and_clears_lane(
    monkeypatch,
) -> None:
    import tldw_chatbook.UI.Screens.model_installed_view as installed_module
    from tldw_chatbook.Model_Artifacts.service import ArtifactStateError
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    canary = "PRIVATE /Users/person/models/model.gguf"
    try:
        raise OSError(canary)
    except OSError as cause:
        failure = ArtifactStateError("artifact deletion I/O failed")
        failure.__cause__ = cause
    logged: list[object] = []

    class BoundedLogger:
        def opt(self, **kwargs):
            logged.append(kwargs)
            return self

        def error(self, *args, **kwargs):
            logged.extend((args, kwargs))

        def warning(self, *args, **kwargs):
            logged.extend((args, kwargs))

    class AppLoop:
        def call_from_thread(self, *_args, **_kwargs):
            raise RuntimeError("call_from_thread used from app loop")

    reference = ArtifactRef("audio-cpp-model", "a" * 40, "q8_0")
    monkeypatch.setattr(installed_module, "logger", BoundedLogger())
    monkeypatch.setattr(InstalledView, "app", property(lambda self: AppLoop()))
    view = InstalledView(service_factory=MagicMock(), legacy_dir=Path("/tmp/models"))
    view._apply_lifecycle_result = MagicMock()

    view._finish_delete_failure_on_loop(reference, failure)

    view._apply_lifecycle_result.assert_called_once()
    rendered = repr((logged, view._apply_lifecycle_result.call_args))
    assert canary not in rendered
    assert "exception': True" not in rendered


@pytest.mark.parametrize("hostile_code", ["PRIVATE-TOKEN-123", "short-secret"])
def test_audio_cpp_delete_failure_never_logs_untrusted_exception_code(
    monkeypatch,
    hostile_code: str,
) -> None:
    import tldw_chatbook.UI.Screens.model_installed_view as installed_module
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    class CollaboratorFailure(RuntimeError):
        code = hostile_code

    logged: list[object] = []

    class BoundedLogger:
        def error(self, *args, **kwargs):
            logged.extend((args, kwargs))

        def warning(self, *args, **kwargs):
            logged.extend((args, kwargs))

    monkeypatch.setattr(installed_module, "logger", BoundedLogger())
    reference = ArtifactRef("audio-cpp-model", "a" * 40, "q8_0")

    InstalledView._bounded_delete_failure(reference, CollaboratorFailure("private"))

    assert hostile_code not in repr(logged)
    assert "operation_failed" in repr(logged)


def test_audio_cpp_delete_failure_does_not_read_hostile_exception_code(
    monkeypatch,
) -> None:
    import tldw_chatbook.UI.Screens.model_installed_view as installed_module
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    class CollaboratorFailure(RuntimeError):
        @property
        def code(self) -> str:
            raise AssertionError("PRIVATE hostile property was evaluated")

    logged: list[object] = []

    class BoundedLogger:
        def error(self, *args, **kwargs):
            logged.extend((args, kwargs))

        def warning(self, *args, **kwargs):
            logged.extend((args, kwargs))

    monkeypatch.setattr(installed_module, "logger", BoundedLogger())
    reference = ArtifactRef("audio-cpp-model", "a" * 40, "q8_0")

    InstalledView._bounded_delete_failure(reference, CollaboratorFailure("private"))

    assert "operation_failed" in repr(logged)
    assert "PRIVATE" not in repr(logged)


def test_deletion_guard_blocks_before_and_after_confirmation(monkeypatch) -> None:
    """A dependency that becomes required cannot enter the delete worker."""
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView
    from tldw_chatbook.Widgets.ModelArtifacts.activation_controls import (
        DeletionRequested,
    )

    reference = ArtifactRef("silero-vad", "immutable-revision", "f32")
    decisions = iter((None, "Managed dependency is required by an external source."))
    fake_app = MagicMock()
    monkeypatch.setattr(InstalledView, "app", property(lambda self: fake_app))
    view = InstalledView(
        service_factory=MagicMock(),
        legacy_dir=Path("/tmp/models"),
        may_delete=lambda _reference: next(decisions),
    )
    view.refresh = MagicMock()
    view._delete_model = MagicMock()

    view._deletion_requested(DeletionRequested(reference))
    dialog, callback = fake_app.push_screen.call_args.args
    callback(True)

    view._delete_model.assert_not_called()
    fake_app.notify.assert_called_once()
    assert fake_app.notify.call_args.args == (
        "Managed dependency is required by an external source.",
    )
    assert fake_app.notify.call_args.kwargs["severity"] == "warning"


def _direct_delete_view(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    service: object,
    recycle_idle,
    may_delete,
):
    from tldw_chatbook.UI.Screens import model_installed_view as module

    fake_app = MagicMock()
    fake_app.call_from_thread.side_effect = lambda callback, *args: callback(*args)
    monkeypatch.setattr(module.InstalledView, "app", property(lambda self: fake_app))
    view = module.InstalledView(
        service_factory=lambda: service,
        legacy_dir=tmp_path,
        recycle_idle=recycle_idle,
        may_delete=may_delete,
    )
    view.ensure_loaded = MagicMock()
    view.refresh = MagicMock()
    reference = ArtifactRef("parakeet-v2", "immutable-revision", "int8")
    view._operation_reference = reference
    view._operation_name = "delete"
    return module, view, fake_app, reference


def test_delete_recycles_idle_owner_and_retries_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A confirmed delete retires one idle owner before one service retry."""
    from tldw_chatbook.Model_Artifacts.service import ArtifactInUseError

    events: list[str] = []

    class _Service:
        def delete(self, _reference: ArtifactRef) -> None:
            attempt = sum(event.startswith("delete-") for event in events) + 1
            events.append(f"delete-{attempt}")
            if attempt == 1:
                raise ArtifactInUseError("private lease")

    def recycle(_reference: ArtifactRef) -> bool:
        events.append("recycle")
        return True

    def policy(_reference: ArtifactRef) -> None:
        events.append("policy-recheck")
        return None

    module, view, fake_app, reference = _direct_delete_view(
        monkeypatch,
        tmp_path,
        service=_Service(),
        recycle_idle=recycle,
        may_delete=policy,
    )

    module.InstalledView._delete_model.__wrapped__(view, reference)

    assert events == ["delete-1", "recycle", "policy-recheck", "delete-2"]
    assert fake_app.notify.call_args.args[0] == "Model delete completed."
    assert fake_app.notify.call_args.kwargs["severity"] == "information"


def test_delete_recycles_idle_owner_refusal_stays_blocked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A busy or unrelated resident never authorizes a delete retry."""
    from tldw_chatbook.Model_Artifacts.service import ArtifactInUseError

    service = MagicMock()
    service.delete.side_effect = ArtifactInUseError("private lease")
    recycle = MagicMock(return_value=False)
    policy = MagicMock(return_value=None)
    module, view, fake_app, reference = _direct_delete_view(
        monkeypatch,
        tmp_path,
        service=service,
        recycle_idle=recycle,
        may_delete=policy,
    )

    module.InstalledView._delete_model.__wrapped__(view, reference)

    service.delete.assert_called_once_with(reference)
    recycle.assert_called_once_with(reference)
    policy.assert_not_called()
    assert "in use" in fake_app.notify.call_args.args[0]
    assert fake_app.notify.call_args.kwargs["severity"] == "error"


def test_delete_rechecks_policy_after_idle_owner_recycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A newly required dependency remains installed after idle retirement."""
    from tldw_chatbook.Model_Artifacts.service import ArtifactInUseError

    service = MagicMock()
    service.delete.side_effect = ArtifactInUseError("private lease")
    blocker = "Managed dependency is required by an external source."
    module, view, fake_app, reference = _direct_delete_view(
        monkeypatch,
        tmp_path,
        service=service,
        recycle_idle=lambda _reference: True,
        may_delete=lambda _reference: blocker,
    )

    module.InstalledView._delete_model.__wrapped__(view, reference)

    service.delete.assert_called_once_with(reference)
    assert fake_app.notify.call_args.args[0] == blocker
    assert fake_app.notify.call_args.kwargs["severity"] == "warning"


def test_delete_retries_once_when_new_lease_wins(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A second lease conflict is final and never starts another recycle."""
    from tldw_chatbook.Model_Artifacts.service import ArtifactInUseError
    from tldw_chatbook.UI.Screens import model_installed_view as module

    marker = "PRIVATE-LEASE-PATH"
    service = MagicMock()
    service.delete.side_effect = (
        ArtifactInUseError(marker),
        ArtifactInUseError(marker),
        None,
    )
    recycle = MagicMock(return_value=True)
    logs: list[str] = []
    sink_id = module.logger.add(lambda message: logs.append(str(message)))
    _module, view, fake_app, reference = _direct_delete_view(
        monkeypatch,
        tmp_path,
        service=service,
        recycle_idle=recycle,
        may_delete=lambda _reference: None,
    )

    try:
        module.InstalledView._delete_model.__wrapped__(view, reference)
    finally:
        module.logger.remove(sink_id)

    assert service.delete.call_count == 2
    recycle.assert_called_once_with(reference)
    assert marker not in "".join(logs)
    assert marker not in str(fake_app.notify.mock_calls)


def test_delete_recycle_callback_failure_is_path_private(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Unexpected recycle failures retain no callback detail in logs or copy."""
    from tldw_chatbook.Model_Artifacts.service import ArtifactInUseError
    from tldw_chatbook.UI.Screens import model_installed_view as module

    marker = "PRIVATE-CALLBACK-PATH"
    service = MagicMock()
    service.delete.side_effect = ArtifactInUseError("private lease")
    logs: list[str] = []
    sink_id = module.logger.add(lambda message: logs.append(str(message)))

    def failed_recycle(_reference: ArtifactRef) -> bool:
        raise RuntimeError(marker)

    _module, view, fake_app, reference = _direct_delete_view(
        monkeypatch,
        tmp_path,
        service=service,
        recycle_idle=failed_recycle,
        may_delete=lambda _reference: None,
    )

    try:
        module.InstalledView._delete_model.__wrapped__(view, reference)
    finally:
        module.logger.remove(sink_id)

    assert service.delete.call_count == 1
    assert "RuntimeError" in "".join(logs)
    assert marker not in "".join(logs)
    assert marker not in str(fake_app.notify.mock_calls)


@pytest.mark.asyncio
async def test_mounted_idle_recycle_state_is_distinct_and_path_private(
    tmp_path: Path,
) -> None:
    """The matching row paints checking and retrying while controls stay pending."""
    from tldw_chatbook.UI.Screens.model_browser_state import InventoryRow
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    reference = ArtifactRef("parakeet-v2", "immutable-revision", "int8")
    recycle_entered = threading.Event()
    recycle_release = threading.Event()
    retry_entered = threading.Event()
    retry_release = threading.Event()
    policy_threads: list[int] = []

    class _Service:
        def __init__(self) -> None:
            self.calls = 0

        def delete(self, _reference: ArtifactRef) -> None:
            from tldw_chatbook.Model_Artifacts.service import ArtifactInUseError

            self.calls += 1
            if self.calls == 1:
                raise ArtifactInUseError("PRIVATE-LEASE-PATH")
            retry_entered.set()
            assert retry_release.wait(timeout=2.0)

    def recycle(_reference: ArtifactRef) -> bool:
        recycle_entered.set()
        assert recycle_release.wait(timeout=2.0)
        return True

    def policy(_reference: ArtifactRef) -> None:
        policy_threads.append(threading.get_ident())
        return None

    row = InventoryRow(
        path=tmp_path / "managed",
        reference=reference,
        model_label="Parakeet v2",
        revision=reference.revision,
        precision="INT8",
        dependencies=(),
        ready=True,
        active=False,
        activation_allowed=True,
        is_broken=False,
        is_unmanaged=False,
        provenance="Managed",
        action_hint="Ready",
        error=None,
        size_bytes=1024,
        installed_store_bytes=1024,
        staging_store_bytes=0,
        free_bytes=4096,
    )
    view = InstalledView(
        service_factory=_Service,
        legacy_dir=tmp_path,
        recycle_idle=recycle,
        may_delete=policy,
    )
    view._loaded = True
    view._rows = (row,)
    app = _InstalledApp(view)

    async with app.run_test(size=(80, 24)) as pilot:

        async def wait_for(event: threading.Event) -> None:
            for _ in range(50):
                if event.is_set():
                    return
                await pilot.pause()
            assert event.is_set()

        view._operation_reference = reference
        view._operation_name = "delete"
        view.refresh(recompose=True)
        worker = view._delete_model(reference)
        await wait_for(recycle_entered)
        await pilot.pause()
        checking = "\n".join(str(item.renderable) for item in view.query("Static"))
        assert "Checking for an idle model to unload…" in checking
        assert "PRIVATE-LEASE-PATH" not in checking

        recycle_release.set()
        await wait_for(retry_entered)
        await pilot.pause()
        retrying = "\n".join(str(item.renderable) for item in view.query("Static"))
        assert "Idle model unloaded; retrying deletion…" in retrying
        assert "PRIVATE-LEASE-PATH" not in retrying
        assert policy_threads == [app._thread_id]

        retry_release.set()
        await worker.wait()


@pytest.mark.asyncio
async def test_empty_inventory_still_reports_managed_and_staging_space(
    tmp_path: Path,
) -> None:
    """Disk totals do not disappear merely because no manifest row exists."""
    from tldw_chatbook.Model_Artifacts.service import ArtifactDiskUsage
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    view = InstalledView(service_factory=MagicMock(), legacy_dir=tmp_path)
    app = _InstalledApp(view)
    async with app.run_test() as pilot:
        view._apply_inventory(
            (),
            ArtifactDiskUsage(
                installed_bytes=0,
                staging_bytes=2 * 1024 * 1024,
                free_bytes=4 * 1024 * 1024,
            ),
            None,
        )
        await pilot.pause()
        text = "\n".join(str(item.renderable) for item in view.query("Static"))

    # All disk totals render through the single shared format_mib formatter
    # (TASK-596 delta port), which always renders MiB -- 2 MiB and 4 MiB
    # chosen so the expected strings are distinct and hand-verifiable.
    assert "2.0 MiB staging" in text
    assert "4.0 MiB free" in text


@pytest.mark.asyncio
async def test_empty_inventory_paints_path_private_recovery_and_real_import_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty managed/legacy inventory keeps both honest GGUF routes usable."""
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView
    from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen

    service = MagicMock()
    service.list_installed.return_value = ()
    service.disk_usage.return_value = ArtifactDiskUsage(0, 0, 64 * 1024 * 1024)
    service.artifacts_path = tmp_path / "managed-store"
    view = InstalledView(service_factory=lambda: service, legacy_dir=tmp_path)
    app = _StyledInstalledApp(view)
    pushed: list[tuple[Screen, Callable | None]] = []

    async with app.run_test(size=(80, 24)) as pilot:
        assert app.CSS_PATH == TldwCli.CSS_PATH
        view.ensure_loaded()
        await _wait_until(pilot, lambda: view._loaded)

        button = view.query_one("#installed-models-import-gguf", Button)
        assert button.can_focus
        assert button in app.screen._compositor.visible_widgets
        for bounds in (button.parent.content_region, view.content_region):
            assert button.region.x >= bounds.x
            assert button.region.right <= bounds.right
            assert button.region.y >= bounds.y
            assert button.region.bottom <= bounds.bottom

        monkeypatch.setattr(
            app,
            "push_screen",
            lambda screen, callback=None: pushed.append((screen, callback)),
        )
        app.screen.set_focus(button)
        await pilot.pause()
        assert button.has_focus
        await pilot.press("enter")
        await pilot.pause()
        picker, callback = pushed.pop()
        assert isinstance(picker, EnhancedFileOpen)
        assert callback is not None

        recovery = next(
            item
            for item in view.query(Static)
            if str(item.renderable).startswith("No managed or legacy models found.")
        )
        assert recovery in app.screen._compositor.visible_widgets
        painted = _painted_screen_text(app)
        assert "No managed or legacy models found." in painted
        assert "Import GGUF…" in painted
        assert "External GGUF" in painted

    expected = (
        "No managed or legacy models found. Use Import GGUF… for a managed copy, "
        "or choose External GGUF under Llama.cpp or Llamafile to use a file in place."
    )
    assert str(recovery.renderable) == expected
    assert str(tmp_path) not in str(recovery.renderable)


@pytest.mark.asyncio
async def test_mounted_install_progress_updates_without_recomposing_inventory(
    tmp_path: Path,
) -> None:
    """Frequent byte events mutate the progress widget, not every model row."""
    from tldw_chatbook.Model_Artifacts.acquisition import AcquisitionProgress
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    reference = ArtifactRef("parakeet-v2", "immutable-revision", "int8")
    progress = AcquisitionProgress(
        "fetch",
        reference,
        "encoder.onnx",
        512,
        1024,
    )
    view = InstalledView(service_factory=MagicMock(), legacy_dir=tmp_path)
    app = _InstalledApp(view)
    async with app.run_test() as pilot:
        view.set_install_state(None, active=True)
        await pilot.pause()
        view.refresh = MagicMock()
        view.set_install_state(progress, active=True)
        await pilot.pause()
        text = "\n".join(str(item.renderable) for item in view.query("Static"))

    view.refresh.assert_not_called()
    assert "Downloading" in text
    assert "encoder.onnx" in text


# Windows Proactor event-loop setup owns an internal loopback socket pair.
@pytest.mark.allow_network
@pytest.mark.asyncio
async def test_tldwcli_css_finish_slice_restores_terminal_import_focus(
    tmp_path: Path,
) -> None:
    """The production CSS paints consent and terminal success restores focus."""
    from tldw_chatbook.UI.Screens import model_installed_view as module
    from tldw_chatbook.Widgets.ModelArtifacts import LocalGGUFImportConsentModal

    source_dir = tmp_path.joinpath(*(["long-private-directory"] * 12))
    source_dir.mkdir(parents=True)
    source = source_dir / "selected-local-model.gguf"
    source.write_bytes(b"x" * 1_048_577)
    reference = _import_reference()
    preference_callback = MagicMock()
    service = _ImportServiceFake(
        lambda _source, _cancelled, _progress: LocalGGUFImportResult(
            reference,
            False,
        )
    )
    view = module.InstalledView(
        service_factory=lambda: service,
        legacy_dir=tmp_path,
        on_root_activated=preference_callback,
    )
    view._loaded = True
    view._rows = _unmanaged_inventory(source)
    app = _StyledInstalledApp(view)
    logs: list[str] = []
    sink_id = module.logger.add(lambda message: logs.append(str(message)))

    try:
        async with app.run_test(size=(80, 24)) as pilot:
            assert app.CSS_PATH == TldwCli.CSS_PATH
            await app.push_screen(
                LocalGGUFImportConsentModal(source, source.stat().st_size)
            )
            await pilot.pause()
            cancel = app.screen.query_one("#local-gguf-import-cancel", Button)
            confirm = app.screen.query_one("#local-gguf-import-confirm", Button)
            painted = _painted_screen_text(app)
            for button, label in ((cancel, "Cancel"), (confirm, "Import")):
                assert button in app.screen._compositor.visible_widgets
                assert button.region.right <= app.size.width
                assert button.region.bottom <= app.size.height
                assert label in painted
            for fact in (
                source.name,
                "managed copy",
                "original stays in place",
                "License and runtime compatibility are not verified",
            ):
                assert fact in painted

            await pilot.press("escape")
            await pilot.pause()
            view._begin_import(source)
            await _wait_until(pilot, lambda: service.activation_finished.is_set())
            await _wait_until(pilot, lambda: service.inventory_reads >= 1)
            await _wait_until(
                pilot,
                lambda: (
                    view.query_one("#installed-models-import-gguf", Button).has_focus
                ),
            )

            assert "Imported and ready" in _rendered_static_text(view)
            assert str(source) not in " ".join(
                notification.message for notification in app._notifications
            )
            assert all(
                str(source) not in str(getattr(worker, "description", ""))
                for worker in app.workers
            )
    finally:
        module.logger.remove(sink_id)

    preference_callback.assert_not_called()
    assert str(source) not in "".join(logs)


# Windows Proactor event-loop setup owns an internal loopback socket pair.
@pytest.mark.allow_network
@pytest.mark.asyncio
async def test_import_progress_updates_without_replacing_focused_cancel(
    tmp_path: Path,
) -> None:
    """Byte events update the mounted progress widget and preserve Cancel focus."""
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    source = tmp_path / "outside.gguf"
    source.write_bytes(b"x" * 1_048_577)
    emit_second = threading.Event()
    second_progress = threading.Event()
    release = threading.Event()
    reference = _import_reference()

    def import_impl(_source, _cancelled, progress):
        progress(LocalGGUFImportProgress("copy", "model.gguf", 1_048_576, 4_194_304))
        assert emit_second.wait(timeout=3.0)
        progress(LocalGGUFImportProgress("copy", "model.gguf", 2_097_152, 4_194_304))
        second_progress.set()
        assert release.wait(timeout=3.0)
        return LocalGGUFImportResult(reference, False)

    service = _ImportServiceFake(import_impl)
    view = InstalledView(service_factory=lambda: service, legacy_dir=tmp_path)
    view._loaded = True
    view._rows = _unmanaged_inventory(source)
    app = _StyledInstalledApp(view)

    async with app.run_test(size=(80, 24)) as pilot:
        view._begin_import(source)
        await _wait_until(pilot, lambda: service.import_entered.is_set())
        cancel = view.query_one("#installed-gguf-import-cancel", Button)
        app.screen.set_focus(cancel)
        emit_second.set()
        await _wait_until(pilot, lambda: second_progress.is_set())
        await pilot.pause()

        assert view.query_one("#installed-gguf-import-cancel", Button) is cancel
        assert cancel.has_focus is True
        painted = _rendered_static_text(view)
        assert "2.0 MiB / 4.0 MiB" in painted

        release.set()
        await _wait_until(pilot, lambda: service.activation_finished.is_set())


# Windows Proactor event-loop setup owns an internal loopback socket pair.
@pytest.mark.allow_network
@pytest.mark.asyncio
async def test_physical_cancel_sets_service_probe_and_preserves_source(
    tmp_path: Path,
) -> None:
    """Enter on the mounted Cancel reaches the service probe, never the source."""
    from tldw_chatbook.Model_Artifacts.service import ArtifactStateError
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    source = tmp_path / "outside.gguf"
    original = b"user-owned" * 131_073
    source.write_bytes(original)
    before = source.stat()
    probe_now = threading.Event()
    observed_cancel = threading.Event()
    release_cancel = threading.Event()
    probe_values: list[bool] = []

    def import_impl(_source, cancelled, _progress):
        assert probe_now.wait(timeout=3.0)
        probe_values.append(cancelled())
        observed_cancel.set()
        assert release_cancel.wait(timeout=3.0)
        raise ArtifactStateError("PRIVATE cancellation detail")

    service = _ImportServiceFake(import_impl)
    view = InstalledView(service_factory=lambda: service, legacy_dir=tmp_path)
    view._loaded = True
    view._rows = _unmanaged_inventory(source)
    app = _StyledInstalledApp(view)

    async with app.run_test(size=(80, 24)) as pilot:
        view._begin_import(source)
        await _wait_until(pilot, lambda: service.import_entered.is_set())
        cancel = view.query_one("#installed-gguf-import-cancel", Button)
        app.screen.set_focus(cancel)
        await pilot.press("enter")
        probe_now.set()
        await _wait_until(pilot, lambda: observed_cancel.is_set())

        try:
            assert probe_values == [True]
            assert view.query_one("#installed-gguf-import-cancel", Button) is cancel
            assert cancel.disabled is True
            assert "Cancelling import…" in _rendered_static_text(view)
        finally:
            release_cancel.set()
        await _wait_until(pilot, lambda: not view._import_active)
        await _wait_until(pilot, lambda: len(view.query(_MODEL_IMPORT_SELECTOR)) == 1)

        assert view._import_cancel_event is not None
        assert view._import_cancel_event.is_set() is True
        assert len(view.query(_MODEL_IMPORT_SELECTOR)) == 1

    assert source.read_bytes() == original
    assert source.stat().st_mtime_ns == before.st_mtime_ns


# Windows Proactor event-loop setup owns an internal loopback socket pair.
@pytest.mark.allow_network
@pytest.mark.asyncio
async def test_attached_queued_cancel_settles_without_entering_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancel before thread entry settles the mounted import lane safely."""
    import asyncio

    from textual.worker import Worker

    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    source = tmp_path / "PRIVATE-QUEUED-CANCEL.gguf"
    source.write_bytes(b"user-owned")
    service = _ImportServiceFake(
        lambda _source, _cancelled, _progress: LocalGGUFImportResult(
            _import_reference(),
            False,
        )
    )
    lane_changes: list[bool] = []
    view = InstalledView(
        service_factory=lambda: service,
        legacy_dir=tmp_path,
        on_import_lane_changed=lane_changes.append,
    )
    view._loaded = True
    view._rows = _unmanaged_inventory(source)
    app = _StyledInstalledApp(view)

    async with app.run_test(size=(80, 24)) as pilot:
        worker_queued = asyncio.Event()
        release_executor = asyncio.Event()
        original_run_threaded = Worker._run_threaded

        async def hold_before_executor(worker):
            worker_queued.set()
            await release_executor.wait()
            return await original_run_threaded(worker)

        monkeypatch.setattr(Worker, "_run_threaded", hold_before_executor)
        view._begin_import(source)
        await _wait_until(pilot, worker_queued.is_set)
        worker = next(
            worker for worker in app.workers if worker.group == "installed_gguf_import"
        )
        cancel = view.query_one("#installed-gguf-import-cancel", Button)
        app.screen.set_focus(cancel)
        try:
            await pilot.press("enter")
            assert cancel.disabled is True
            assert "Cancelling import…" in _rendered_static_text(view)
        finally:
            release_executor.set()
        await worker.wait()
        await _wait_until(pilot, lambda: not view._import_active)
        await _wait_until(
            pilot,
            lambda: len(view.query("#installed-gguf-import-retry")) == 1,
        )
        retry = view.query_one("#installed-gguf-import-retry", Button)
        await _wait_until(
            pilot,
            lambda: retry.has_focus,
        )

        rendered = _rendered_static_text(view)
        notices = " ".join(item.message for item in app._notifications)
        assert "Import cancelled" in rendered
        assert str(source) not in rendered
        assert str(source) not in notices
        assert service.import_sources == []
        assert service.activation_calls == []
        assert lane_changes == [True, False]
        for selector in (
            "#installed-models-refresh",
            "#installed-models-repair",
            "#installed-models-import-gguf",
            _MODEL_IMPORT_SELECTOR,
        ):
            assert view.query_one(selector, Button).disabled is False


# Windows Proactor event-loop setup owns an internal loopback socket pair.
@pytest.mark.allow_network
@pytest.mark.asyncio
async def test_finalizing_disables_cancel_before_promotion(tmp_path: Path) -> None:
    """The synchronous Finalizing callback closes cancellation before commit."""
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    source = tmp_path / "outside.gguf"
    source.write_bytes(b"x" * 1_048_577)
    finalizing_callback_returned = threading.Event()
    promotion_gate = threading.Event()
    reference = _import_reference()

    def import_impl(_source, _cancelled, progress):
        progress(LocalGGUFImportProgress("finalize", None, 1_048_577, 1_048_577))
        finalizing_callback_returned.set()
        assert promotion_gate.wait(timeout=3.0)
        return LocalGGUFImportResult(reference, False)

    service = _ImportServiceFake(import_impl)
    view = InstalledView(service_factory=lambda: service, legacy_dir=tmp_path)
    view._loaded = True
    view._rows = _unmanaged_inventory(source)
    app = _StyledInstalledApp(view)

    async with app.run_test(size=(80, 24)) as pilot:
        view._begin_import(source)
        await _wait_until(pilot, lambda: finalizing_callback_returned.is_set())
        cancel = view.query_one("#installed-gguf-import-cancel", Button)
        assert cancel.disabled is True
        assert "Finalizing managed model" in _rendered_static_text(view)
        assert promotion_gate.is_set() is False

        promotion_gate.set()
        await _wait_until(pilot, lambda: service.activation_finished.is_set())


@pytest.mark.asyncio
async def test_converged_import_finalizes_before_blocking_activation(
    tmp_path: Path,
) -> None:
    """Already-installed convergence closes Cancel before activation starts."""
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    source = tmp_path / "outside.gguf"
    source.write_bytes(b"x" * 1_048_577)
    reference = _import_reference()
    activation_entered = threading.Event()
    release_activation = threading.Event()

    def activate_impl(activated: ArtifactRef) -> ArtifactRef:
        activation_entered.set()
        assert release_activation.wait(timeout=3.0)
        return activated

    service = _ImportServiceFake(
        lambda _source, _cancelled, _progress: LocalGGUFImportResult(
            reference,
            True,
        ),
        activation_impl=activate_impl,
    )
    view = InstalledView(service_factory=lambda: service, legacy_dir=tmp_path)
    view._loaded = True
    view._rows = _unmanaged_inventory(source)
    app = _StyledInstalledApp(view)

    async with app.run_test(size=(80, 24)) as pilot:
        view._begin_import(source)
        await _wait_until(pilot, lambda: activation_entered.is_set())
        cancel = view.query_one("#installed-gguf-import-cancel", Button)

        try:
            assert cancel.disabled is True
            assert "Finalizing managed model" in _rendered_static_text(view)
            assert view._import_cancel_event is not None
            assert view._import_cancel_event.is_set() is False

            app.screen.set_focus(cancel)
            await pilot.press("enter")
            await pilot.pause()
            assert view._import_cancel_event.is_set() is False
            assert "Cancelling import" not in _rendered_static_text(view)
        finally:
            release_activation.set()

        await _wait_until(pilot, lambda: service.activation_finished.is_set())
        await _wait_until(pilot, lambda: not view._import_active)


@pytest.mark.asyncio
async def test_import_success_activates_but_does_not_change_source_preference(
    tmp_path: Path,
) -> None:
    """Import activates the exact ref without selecting it for any runtime."""
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    source = tmp_path / "outside.gguf"
    source.write_bytes(b"x" * 1_048_577)
    reference = _import_reference()
    preference_callback = MagicMock()
    service = _ImportServiceFake(
        lambda _source, _cancelled, _progress: LocalGGUFImportResult(
            reference,
            False,
        )
    )
    view = InstalledView(
        service_factory=lambda: service,
        legacy_dir=tmp_path,
        on_root_activated=preference_callback,
    )
    view._loaded = True
    view._rows = _unmanaged_inventory(source)
    app = _InstalledApp(view)

    async with app.run_test() as pilot:
        view._begin_import(source)
        await _wait_until(pilot, lambda: service.activation_finished.is_set())
        await _wait_until(pilot, lambda: not view._import_active)

        assert service.activation_calls == [reference]
        assert service.inventory_reads >= 1
        assert view._pending_import_path is None
        assert "Imported and ready" in _rendered_static_text(view)

    preference_callback.assert_not_called()


# Windows Proactor event-loop setup owns an internal loopback socket pair.
@pytest.mark.allow_network
@pytest.mark.asyncio
async def test_activation_failure_keeps_installed_row_and_offers_activate(
    tmp_path: Path,
) -> None:
    """A promoted artifact survives readiness failure with exact Activate recovery."""
    from Tests.Model_Artifacts.test_acquisition_types import make_descriptor
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    source = tmp_path / "outside.gguf"
    source.write_bytes(b"x" * 1_048_577)
    reference = _import_reference()
    descriptor = replace(
        make_descriptor(),
        reference=reference,
        model_id="Imported local model",
        precision=reference.variant,
    )
    installed = InstalledArtifact(
        path=tmp_path / "managed-copy",
        descriptor=descriptor,
        ready=False,
        active=False,
        error=None,
    )
    service = _ImportServiceFake(
        lambda _source, _cancelled, _progress: LocalGGUFImportResult(
            reference,
            False,
        ),
        installed=(installed,),
        activation_error=RuntimeError("PRIVATE activation detail"),
    )
    view = InstalledView(service_factory=lambda: service, legacy_dir=tmp_path)
    view._loaded = True
    view._rows = _unmanaged_inventory(source)
    app = _InstalledApp(view)

    async with app.run_test() as pilot:
        view._begin_import(source)
        await _wait_until(pilot, lambda: service.activation_finished.is_set())
        await _wait_until(pilot, lambda: service.inventory_reads >= 1)
        await _wait_until(
            pilot,
            lambda: "Installed — activation required" in _rendered_static_text(view),
        )
        await _wait_until(pilot, lambda: len(view.query(".model-activate")) == 1)

        activate = view.query_one(".model-activate", Button)
        assert activate.disabled is False
        assert view._pending_import_path is None
        assert source.name in _rendered_static_text(view)


@pytest.mark.asyncio
async def test_stale_import_callback_cannot_replace_newer_status(
    tmp_path: Path,
) -> None:
    """Generation N cannot settle the visible lane after N+1 takes ownership."""
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    view = InstalledView(service_factory=MagicMock(), legacy_dir=tmp_path)
    app = _InstalledApp(view)

    async with app.run_test() as pilot:
        view._import_generation = 2
        view._import_active = True
        view._import_status = "Newer import is running"
        view.refresh(recompose=True)
        await pilot.pause()

        view._apply_import_success(
            1,
            LocalGGUFImportResult(_import_reference(), False),
        )
        await pilot.pause()

        assert view._import_generation == 2
        assert view._import_active is True
        assert view._import_status == "Newer import is running"
        assert "Newer import is running" in _rendered_static_text(view)


# Windows Proactor event-loop setup owns an internal loopback socket pair.
@pytest.mark.allow_network
@pytest.mark.asyncio
async def test_import_failure_logs_only_stable_category_and_never_selected_path(
    tmp_path: Path,
) -> None:
    """Import failures retain a type/category while suppressing exception paths."""
    from tldw_chatbook.UI.Screens import model_installed_view as module

    source = tmp_path / "PRIVATE-SENTINEL-MODEL.gguf"
    source.write_bytes(b"x" * 1_048_577)

    def import_impl(_source, _cancelled, _progress):
        raise RuntimeError(f"failed at {source}")

    service = _ImportServiceFake(import_impl)
    view = module.InstalledView(service_factory=lambda: service, legacy_dir=tmp_path)
    view._loaded = True
    view._rows = _unmanaged_inventory(source)
    app = _InstalledApp(view)
    logs: list[str] = []
    sink_id = module.logger.add(lambda message: logs.append(str(message)))

    try:
        async with app.run_test() as pilot:
            view._begin_import(source)
            await _wait_until(pilot, lambda: not view._import_active)
            notifications = " ".join(
                notification.message for notification in app._notifications
            )
            rendered = _rendered_static_text(view)
    finally:
        module.logger.remove(sink_id)

    combined_logs = "".join(logs)
    assert "phase=import" in combined_logs
    assert "error_type=RuntimeError" in combined_logs
    assert str(source) not in combined_logs
    assert str(source) not in notifications
    assert str(source) not in rendered


# Windows Proactor event-loop setup owns an internal loopback socket pair.
@pytest.mark.allow_network
@pytest.mark.asyncio
async def test_real_import_lease_timeout_offers_busy_retry_without_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real service timeout reaches exact path-private mounted recovery."""
    from Tests.Model_Artifacts.gguf_test_helpers import make_gguf
    from tldw_chatbook.Model_Artifacts import service as service_module
    from tldw_chatbook.UI.Screens import model_installed_view as module

    source = tmp_path / "PRIVATE-BUSY-SOURCE.gguf"
    payload = make_gguf(architecture="llama", name="Busy", file_type=7)
    source.write_bytes(payload)
    before = source.stat()
    service = service_module.ModelArtifactService(tmp_path / "store")
    raw_lock_detail = f"PRIVATE-LOCK-DETAIL for {source}"

    def time_out_lease(_lease) -> None:
        raise service_module.ArtifactLeaseTimeoutError(raw_lock_detail)

    monkeypatch.setattr(
        service_module._leases.ArtifactOperationLease,
        "acquire",
        time_out_lease,
    )
    lane_changes: list[bool] = []
    view = module.InstalledView(
        service_factory=lambda: service,
        legacy_dir=tmp_path,
        on_import_lane_changed=lane_changes.append,
    )
    view._loaded = True
    view._rows = _unmanaged_inventory(source)
    app = _StyledInstalledApp(view)
    logs: list[str] = []
    sink_id = module.logger.add(lambda message: logs.append(str(message)))

    try:
        async with app.run_test(size=(80, 24)) as pilot:
            view._begin_import(source)
            await _wait_until(pilot, lambda: not view._import_active)
            await _wait_until(
                pilot,
                lambda: len(view.query("#installed-gguf-import-retry")) == 1,
            )

            expected = "The managed model store is busy. Retry shortly."
            rendered = _rendered_static_text(view)
            notices = [item.message for item in app._notifications]
            assert expected in rendered
            assert expected in notices
            assert (
                view.query_one("#installed-gguf-import-retry", Button).disabled is False
            )
            assert (
                view.query_one("#installed-gguf-import-choose", Button).disabled
                is False
            )
            assert view._import_lane_owned is False
            assert lane_changes == [True, False]
            for selector in (
                "#installed-models-refresh",
                "#installed-models-repair",
                "#installed-models-import-gguf",
                _MODEL_IMPORT_SELECTOR,
            ):
                assert view.query_one(selector, Button).disabled is False
            combined_ui = rendered + " ".join(notices) + "".join(logs)
            assert str(source) not in combined_ui
            assert raw_lock_detail not in combined_ui
    finally:
        module.logger.remove(sink_id)

    assert source.read_bytes() == payload
    assert source.stat().st_mtime_ns == before.st_mtime_ns
    assert tuple(service.artifacts_path.rglob("manifest.json")) == ()
    assert tuple(service.staging_path.iterdir()) == ()


@pytest.mark.asyncio
async def test_cancelled_and_failed_import_offer_retry_and_choose_another(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both recoveries stay path-private; Retry alone retains the selection."""
    from tldw_chatbook.Model_Artifacts.service import ArtifactStateError
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView
    from tldw_chatbook.Widgets.enhanced_file_picker import EnhancedFileOpen

    source = tmp_path / "PRIVATE-RETRY-SENTINEL.gguf"
    source.write_bytes(b"x" * 1_048_577)
    attempts = 0
    cancellation_seen = threading.Event()

    def import_impl(_source, cancelled, _progress):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            while not cancelled():
                cancellation_seen.wait(timeout=0.01)
            cancellation_seen.set()
            raise ArtifactStateError("private cancellation detail")
        raise RuntimeError(f"private failure at {source}")

    service = _ImportServiceFake(import_impl)
    view = InstalledView(service_factory=lambda: service, legacy_dir=tmp_path)
    view._loaded = True
    view._rows = _unmanaged_inventory(source)
    app = _InstalledApp(view)
    pushed: list[tuple[Screen, Callable]] = []

    async with app.run_test(size=(80, 24)) as pilot:
        monkeypatch.setattr(
            app,
            "push_screen",
            lambda screen, callback=None: pushed.append((screen, callback)),
        )
        view._begin_import(source)
        await _wait_until(pilot, lambda: service.import_entered.is_set())
        cancel = view.query_one("#installed-gguf-import-cancel", Button)
        app.screen.set_focus(cancel)
        await pilot.press("enter")
        await _wait_until(pilot, lambda: cancellation_seen.is_set())
        await _wait_until(pilot, lambda: not view._import_active)

        assert "Import cancelled" in _rendered_static_text(view)
        assert view.query_one("#installed-gguf-import-retry", Button)
        assert view.query_one("#installed-gguf-import-choose", Button)
        assert str(source) not in _rendered_static_text(view)

        retry = view.query_one("#installed-gguf-import-retry", Button)
        app.screen.set_focus(retry)
        await pilot.press("enter")
        await _wait_until(pilot, lambda: len(service.import_sources) == 2)
        await _wait_until(pilot, lambda: not view._import_active)

        assert "could not be imported" in _rendered_static_text(view)
        assert service.import_sources == [source, source]
        assert str(source) not in _rendered_static_text(view)

        choose = view.query_one("#installed-gguf-import-choose", Button)
        app.screen.set_focus(choose)
        await pilot.press("enter")
        await pilot.pause()
        picker, _callback = pushed.pop()
        assert isinstance(picker, EnhancedFileOpen)
        assert view._pending_import_path is None
        assert str(source) not in _rendered_static_text(view)


# Windows Proactor event-loop setup owns an internal loopback socket pair.
@pytest.mark.allow_network
@pytest.mark.asyncio
async def test_import_lane_disables_every_lifecycle_action_at_80_columns(
    tmp_path: Path,
) -> None:
    """One active import owns all managed mutations while controls remain visible."""
    from Tests.Model_Artifacts.test_acquisition_types import make_descriptor
    from tldw_chatbook.UI.Screens.model_browser_state import inventory_rows
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    source = tmp_path / "outside.gguf"
    source.write_bytes(b"x" * 1_048_577)
    reference = _import_reference()
    descriptor = replace(
        make_descriptor(),
        reference=reference,
        precision=reference.variant,
    )
    managed = InstalledArtifact(
        path=tmp_path / "managed-model",
        descriptor=descriptor,
        ready=True,
        active=False,
        error=None,
    )
    release = threading.Event()

    def import_impl(_source, _cancelled, _progress):
        assert release.wait(timeout=3.0)
        return LocalGGUFImportResult(reference, True)

    service = _ImportServiceFake(import_impl)
    view = InstalledView(service_factory=lambda: service, legacy_dir=tmp_path)
    view._loaded = True
    view._rows = inventory_rows(
        (managed,),
        ArtifactDiskUsage(0, 0, 64 * 1024 * 1024),
        (),
    ) + _unmanaged_inventory(source)
    app = _StyledInstalledApp(view)

    async with app.run_test(size=(80, 24)) as pilot:
        view._begin_import(source)
        await _wait_until(pilot, lambda: service.import_entered.is_set())

        selectors = (
            "#installed-models-refresh",
            "#installed-models-repair",
            "#installed-models-import-gguf",
            _MODEL_IMPORT_SELECTOR,
            ".model-activate",
            ".model-delete",
        )
        for selector in selectors:
            button = view.query_one(selector, Button)
            assert button.disabled is True
            try:
                button.query_ancestor(".installed-model-row").scroll_visible(
                    animate=False,
                )
            except NoMatches:
                button.scroll_visible(animate=False)
            await pilot.pause()
            assert button in app.screen._compositor.visible_widgets
            assert button.region.right <= app.size.width
            assert button.region.bottom <= app.size.height
        assert (
            len(
                [
                    worker
                    for worker in app.workers
                    if worker.group == "installed_gguf_import"
                ]
            )
            == 1
        )

        release.set()
        await _wait_until(pilot, lambda: service.activation_finished.is_set())


@pytest.mark.asyncio
async def test_unmount_cancels_import_and_forgets_selected_path(tmp_path: Path) -> None:
    """Unmount invalidates the lane without touching detached widgets."""
    from tldw_chatbook.Model_Artifacts.service import ArtifactStateError
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    source = tmp_path / "outside.gguf"
    source.write_bytes(b"x" * 1_048_577)
    cancel_seen = threading.Event()

    def import_impl(_source, cancelled, _progress):
        while not cancelled():
            cancel_seen.wait(timeout=0.01)
        cancel_seen.set()
        raise ArtifactStateError("cancelled after unmount")

    service = _ImportServiceFake(import_impl)
    view = InstalledView(service_factory=lambda: service, legacy_dir=tmp_path)
    app = _InstalledApp(view)

    async with app.run_test() as pilot:
        view._begin_import(source)
        await _wait_until(pilot, lambda: service.import_entered.is_set())
        generation = view._import_generation
        await view.remove()
        await _wait_until(pilot, lambda: cancel_seen.is_set())

        assert view._pending_import_path is None
        assert view._import_generation > generation


def test_forced_refresh_queues_behind_an_inflight_inventory_load(
    tmp_path: Path,
) -> None:
    """A lifecycle completion cannot lose its mandatory post-operation refresh."""
    from tldw_chatbook.Model_Artifacts.service import ArtifactDiskUsage
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    view = InstalledView(service_factory=MagicMock(), legacy_dir=tmp_path)
    view._loading = True
    view._load_inventory = MagicMock()
    view.refresh = MagicMock()

    view.ensure_loaded(force=True)
    view._apply_inventory(
        (),
        ArtifactDiskUsage(installed_bytes=0, staging_bytes=0, free_bytes=4096),
        None,
    )

    view._load_inventory.assert_called_once_with()
    assert view._loading is True


@pytest.mark.asyncio
async def test_curated_view_performs_no_io_at_compose_time(tmp_path: Path) -> None:
    """Curated is also eagerly mounted but remains idle until selected."""
    from tldw_chatbook.UI.Screens.model_curated_view import CuratedView

    service_factory = MagicMock()
    registry_factory = MagicMock()
    view = CuratedView(
        service_factory=service_factory,
        registry_factory=registry_factory,
    )
    app = _InstalledApp(view)
    async with app.run_test() as pilot:
        await pilot.pause()

    service_factory.assert_not_called()
    registry_factory.assert_not_called()


@pytest.mark.asyncio
async def test_installed_audio_cpp_row_separates_package_truth_at_80x24(
    tmp_path: Path,
) -> None:
    """Installed package state never impersonates configured or running state."""
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import audio_cpp_curated_entries
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    descriptor, _sources = audio_cpp_curated_entries()[0]
    installed = InstalledArtifact(
        path=tmp_path / "managed-package",
        descriptor=descriptor,
        ready=True,
        active=True,
        error=None,
    )

    class Service:
        def list_installed(self):
            return (installed,)

        def disk_usage(self):
            return ArtifactDiskUsage(
                descriptor.expected_installed_bytes,
                0,
                64 * 1024 * 1024,
            )

    view = InstalledView(service_factory=Service, legacy_dir=tmp_path)
    app = _StyledInstalledApp(view)

    async with app.run_test(size=(80, 24)) as pilot:
        view.ensure_loaded()
        await _wait_until(pilot, lambda: view._loaded)
        text = _rendered_static_text(view)
        for fact in (
            "Available: Complete pinned source recorded; live reachability not checked",
            "Installed package: Local record found",
            "Integrity: Not checked this session",
            "Recipe: audio-cpp-",
            "installed scan not checked",
            "Compatibility:",
            "Configured: Unknown — Settings state was not checked",
            "Running: Unknown — supervisor state was not checked",
            "Speech tasks:",
            "Required package files:",
            "Pinned source:",
            "Manifest authority: Pinned sizes and SHA-256 digests recorded",
            "Package size:",
            "Model package only — audiocpp_server is not included",
        ):
            assert fact in text
        assert "Active" not in text
        assert "Integrity verified" not in text
        assert "Integrity: Verified" not in text
        assert "Recipe: Matched" not in text
        assert not view.query(".model-activate")
        delete = view.query_one(".model-delete", Button)
        delete.focus()
        await pilot.pause()
        assert delete in app.screen._compositor.visible_widgets
        assert delete.region.right <= view.region.right
        assert delete.region.bottom <= view.region.bottom


@pytest.mark.asyncio
async def test_corrupt_audio_cpp_row_never_promotes_ready_or_integrity_truth(
    tmp_path: Path,
) -> None:
    """A stale readiness bit cannot outweigh current inventory failure evidence."""
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import audio_cpp_curated_entries
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    descriptor, _sources = audio_cpp_curated_entries()[0]
    installed = InstalledArtifact(
        path=tmp_path / "managed-package",
        descriptor=descriptor,
        ready=False,
        active=True,
        error="PRIVATE checksum failure detail",
    )

    class Service:
        def list_installed(self):
            return (installed,)

        def disk_usage(self):
            return ArtifactDiskUsage(1, 0, 64 * 1024 * 1024)

    view = InstalledView(service_factory=Service, legacy_dir=tmp_path)
    app = _StyledInstalledApp(view)
    async with app.run_test(size=(80, 24)) as pilot:
        view.ensure_loaded()
        await _wait_until(pilot, lambda: view._loaded)
        text = _rendered_static_text(view)

    assert "Integrity: Unknown — package record needs Repair" in text
    assert "Integrity verified" not in text
    assert "Integrity: Verified" not in text
    assert "Ready" not in text
    assert "Active" not in text
    assert "Recipe: Matched" not in text
    assert "PRIVATE" not in text


@pytest.mark.asyncio
async def test_install_only_audio_cpp_root_without_readiness_is_not_a_repair_error(
    tmp_path: Path,
) -> None:
    """The real activate=False state is installed, inactive, and unverified."""
    from Tests.Model_Artifacts.test_acquisition_types import (
        DictCatalog,
        make_descriptor,
    )
    from tldw_chatbook.Model_Artifacts.acquisition import ArtifactAcquisitionService
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    reference = ArtifactRef("audio-cpp-install-only", "a" * 40, "int8")
    descriptor = replace(
        make_descriptor(reference, files_body=b"audio-package"),
        consumer="audio_cpp",
        precision=reference.variant,
    )
    service = ModelArtifactService(tmp_path / "store")
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.onnx").write_bytes(b"audio-package")
    service.install(descriptor, source)
    acquisition = ArtifactAcquisitionService(
        service,
        free_bytes_probe=lambda _path: 1_000_000_000,
    )
    catalog = DictCatalog({reference: descriptor})
    report = await acquisition.preflight(reference, catalog)
    await acquisition.provision(reference, report.grant(), catalog, activate=False)
    installed = service.list_installed()[0]
    assert installed.ready is False
    assert installed.active is False
    assert installed.error is None

    view = InstalledView(service_factory=lambda: service, legacy_dir=tmp_path)
    app = _StyledInstalledApp(view)
    async with app.run_test(size=(80, 24)) as pilot:
        view.ensure_loaded()
        await _wait_until(pilot, lambda: view._loaded)
        text = _rendered_static_text(view)

    assert "Integrity: Not checked this session" in text
    assert "package record needs Repair" not in text
    assert "Active" not in text


@pytest.mark.parametrize("mismatch", ("revision", "repository"))
@pytest.mark.asyncio
async def test_installed_audio_cpp_mismatch_is_review_required_when_mounted(
    tmp_path: Path,
    mismatch: str,
) -> None:
    """Installed rows do not promote drifted descriptors to canonical truth."""
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import audio_cpp_curated_entries
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    descriptor, _sources = audio_cpp_curated_entries()[0]
    mismatched = (
        replace(
            descriptor,
            reference=ArtifactRef(
                descriptor.reference.artifact_id,
                "0" * 40,
                descriptor.reference.variant,
            ),
        )
        if mismatch == "revision"
        else replace(descriptor, upstream_repository="attacker/repository")
    )
    installed = InstalledArtifact(
        path=tmp_path / "managed-package",
        descriptor=mismatched,
        ready=False,
        active=False,
        error=None,
    )
    service = MagicMock()
    service.list_installed.return_value = (installed,)
    service.disk_usage.return_value = ArtifactDiskUsage(1, 0, 64 * 1024 * 1024)
    view = InstalledView(service_factory=lambda: service, legacy_dir=tmp_path)
    app = _StyledInstalledApp(view)

    async with app.run_test(size=(80, 24)) as pilot:
        view.ensure_loaded()
        await _wait_until(pilot, lambda: view._loaded)
        text = _rendered_static_text(view)

    assert "Available: Unknown" in text
    assert "review required" in text
    assert "Recipe: Unknown" in text
    assert "Pinned source: Unknown" in text
    assert "Complete pinned source recorded" not in text
    assert "exact manifest mapping recorded" not in text
    assert "attacker/repository" not in text


@pytest.mark.parametrize(
    "theme",
    ("textual-dark", "textual-light", "tokyo-night", "monokai", "dracula"),
)
@pytest.mark.asyncio
async def test_disabled_audio_cpp_delete_has_adjacent_reason_and_contrast(
    tmp_path: Path,
    theme: str,
) -> None:
    """A disabled destructive action remains readable without pointer help."""
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import audio_cpp_curated_entries
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    descriptor, _sources = audio_cpp_curated_entries()[0]
    installed = InstalledArtifact(
        path=tmp_path / "managed-package",
        descriptor=descriptor,
        ready=True,
        active=False,
        error=None,
    )
    service = MagicMock()
    service.list_installed.return_value = (installed,)
    service.disk_usage.return_value = ArtifactDiskUsage(1, 0, 64 * 1024 * 1024)
    view = InstalledView(service_factory=lambda: service, legacy_dir=tmp_path)
    view._install_active = True
    app = _StyledInstalledApp(view)
    app.theme = theme

    async with app.run_test(size=(80, 24)) as pilot:
        view.ensure_loaded()
        await _wait_until(pilot, lambda: view._loaded)
        delete = view.query_one(".model-delete", Button)
        delete.scroll_visible(animate=False, immediate=True, force=True)
        await pilot.pause()
        assert delete.disabled is True
        assert (
            "Delete unavailable — another model package operation is in progress."
            in _rendered_static_text(view)
        )
        painted = _painted_style_of_text(app, delete.region, "Delete")
        assert painted is not None
        assert painted.color is not None and painted.bgcolor is not None
        ratio = _contrast(painted.color, painted.bgcolor)
        assert ratio >= 3.0, f"Delete is {ratio:.2f}:1 under {theme}"


@pytest.mark.parametrize(
    "control_id",
    ("installed-models-refresh", "installed-models-repair"),
)
@pytest.mark.asyncio
async def test_installed_header_action_restores_focus_after_recompose(
    tmp_path: Path,
    control_id: str,
) -> None:
    """Refresh and Repair return focus to the semantic invoking action."""
    from tldw_chatbook.Model_Artifacts.service import ReconcileReport
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    service = MagicMock()
    service.list_installed.return_value = ()
    service.disk_usage.return_value = ArtifactDiskUsage(0, 0, 64 * 1024 * 1024)
    service.reconcile.return_value = ReconcileReport(0, 0, (), (), ())
    view = InstalledView(service_factory=lambda: service, legacy_dir=tmp_path)
    app = _StyledInstalledApp(view)
    async with app.run_test(size=(80, 24)) as pilot:
        view.ensure_loaded()
        await _wait_until(pilot, lambda: view._loaded)
        control = view.query_one(f"#{control_id}", Button)
        control.focus()
        await pilot.press("enter")
        await _wait_until(
            pilot,
            lambda: (
                not view._loading
                and view._operation_name is None
                and view.query_one(f"#{control_id}", Button).has_focus
            ),
        )


@pytest.mark.asyncio
async def test_blocked_audio_cpp_removal_paints_recovery_and_restores_delete_focus(
    tmp_path: Path,
) -> None:
    """A live generation blocks removal inline and returns focus to retry."""
    from tldw_chatbook.Model_Artifacts.service import ArtifactRemovalAvailability
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import audio_cpp_curated_entries
    from tldw_chatbook.TTS.audio_cpp_artifact_dependencies import (
        AudioCppArtifactRemovalEvidence,
    )
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    descriptor, _sources = audio_cpp_curated_entries()[0]
    reference = descriptor.reference
    installed = InstalledArtifact(
        path=tmp_path / "managed-package",
        descriptor=descriptor,
        ready=True,
        active=False,
        error=None,
    )
    evidence = AudioCppArtifactRemovalEvidence(
        reference,
        staged_runtime_ids=("staged-generation",),
    )

    class Service:
        def list_installed(self):
            return (installed,)

        def disk_usage(self):
            return ArtifactDiskUsage(1, 0, 64 * 1024 * 1024)

    class Coordinator:
        async def probe_removal_availability(self, exact):
            assert exact == reference
            return ArtifactRemovalAvailability.AVAILABLE

    view = InstalledView(service_factory=Service, legacy_dir=tmp_path)

    class BlockedApp(_StyledInstalledApp):
        async def _audio_cpp_artifact_removal_evidence(self, exact):
            assert exact == reference
            return evidence

        def _ensure_audio_cpp_artifact_lease_coordinator(self):
            return Coordinator()

    app = BlockedApp(view)

    async with app.run_test(size=(80, 24)) as pilot:
        view.ensure_loaded()
        await _wait_until(pilot, lambda: view._loaded)
        view.query_one(".model-delete", Button).press()
        await _wait_until(
            pilot,
            lambda: "Package in use" in _rendered_static_text(view),
        )
        status = view.query_one("#installed-lifecycle-status", Static)
        assert "Shut down or discard active work, then review removal again." in str(
            status.renderable
        )
        delete = view.query_one(".model-delete", Button)
        assert delete.has_focus
        assert delete in app.screen._compositor.visible_widgets


@pytest.mark.asyncio
async def test_delayed_bulk_observation_restores_installed_delete_focus(
    tmp_path: Path,
) -> None:
    """Evidence refresh preserves the focused id-less Delete button in place."""
    import asyncio
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import audio_cpp_curated_entries
    from tldw_chatbook.TTS.audio_cpp_artifact_dependencies import (
        AudioCppArtifactRemovalEvidence,
        AudioCppModelLibraryObservationSnapshot,
    )
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    descriptor, _sources = audio_cpp_curated_entries()[0]
    installed = InstalledArtifact(
        path=tmp_path / "managed-package",
        descriptor=descriptor,
        ready=False,
        active=False,
        error=None,
    )
    service = MagicMock()
    service.list_installed.return_value = (installed,)
    service.disk_usage.return_value = ArtifactDiskUsage(1, 0, 64 * 1024 * 1024)
    release = asyncio.Event()
    calls: list[tuple[ArtifactRef, ...]] = []

    async def observe(references):
        calls.append(references)
        await release.wait()
        return AudioCppModelLibraryObservationSnapshot(
            (
                AudioCppArtifactRemovalEvidence(
                    descriptor.reference,
                    settings_consumers=(("saved", "Guided Settings", "package"),),
                ),
            )
        )

    view = InstalledView(
        service_factory=lambda: service,
        legacy_dir=tmp_path,
        observation_provider=observe,
    )
    app = _StyledInstalledApp(view)
    async with app.run_test(size=(80, 24)) as pilot:
        view.ensure_loaded()
        await _wait_until(pilot, lambda: view._loaded and bool(calls))
        delete = view.query_one(".model-delete", Button)
        delete.focus()
        release.set()
        await _wait_until(
            pilot,
            lambda: "Configured: Saved Settings" in _rendered_static_text(view),
        )
        assert delete is app.focused
        assert view.query_one(".model-delete", Button).has_focus

    assert calls == [(descriptor.reference,)]


@pytest.mark.asyncio
async def test_delete_press_during_observation_refresh_reviews_once(
    tmp_path: Path,
) -> None:
    """Refreshing evidence cannot unmount Delete before its intent is delivered."""
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import audio_cpp_curated_entries
    from tldw_chatbook.TTS.audio_cpp_artifact_dependencies import (
        AudioCppArtifactRemovalEvidence,
        AudioCppModelLibraryObservationSnapshot,
    )
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    descriptor, _sources = audio_cpp_curated_entries()[0]
    installed = InstalledArtifact(
        path=tmp_path / "managed-package",
        descriptor=descriptor,
        ready=False,
        active=False,
        error=None,
    )
    service = MagicMock()
    service.list_installed.return_value = (installed,)
    service.disk_usage.return_value = ArtifactDiskUsage(1, 0, 64 * 1024 * 1024)
    refresh_entered = asyncio.Event()
    release_refresh = asyncio.Event()
    calls = 0

    async def observe(references):
        nonlocal calls
        calls += 1
        if calls == 2:
            refresh_entered.set()
            await release_refresh.wait()
        return AudioCppModelLibraryObservationSnapshot(
            (AudioCppArtifactRemovalEvidence(references[0]),)
        )

    view = InstalledView(
        service_factory=lambda: service,
        legacy_dir=tmp_path,
        observation_provider=observe,
    )
    view._review_audio_cpp_deletion = MagicMock()
    app = _StyledInstalledApp(view)
    async with app.run_test(size=(80, 24)) as pilot:
        view.ensure_loaded()
        await _wait_until(pilot, lambda: calls == 1)
        delete = view.query_one(".model-delete", Button)
        assert delete.disabled is False

        view.refresh_observations()
        assert delete.is_attached
        delete.press()

        await _wait_until(pilot, refresh_entered.is_set)
        await _wait_until(
            pilot,
            lambda: view._review_audio_cpp_deletion.call_count == 1,
        )
        view._review_audio_cpp_deletion.assert_called_once_with(descriptor.reference)
        release_refresh.set()


@pytest.mark.asyncio
async def test_back_to_back_installed_refresh_starts_only_the_latest_generation(
    tmp_path: Path,
) -> None:
    """A deferred old start cannot cancel the newer installed observation."""
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import audio_cpp_curated_entries
    from tldw_chatbook.TTS.audio_cpp_artifact_dependencies import (
        AudioCppArtifactRemovalEvidence,
        AudioCppModelLibraryObservationSnapshot,
    )
    from tldw_chatbook.UI.Screens.model_installed_view import InstalledView

    descriptor, _sources = audio_cpp_curated_entries()[0]
    installed = InstalledArtifact(
        path=tmp_path / "managed-package",
        descriptor=descriptor,
        ready=False,
        active=False,
        error=None,
    )
    service = MagicMock()
    service.list_installed.return_value = (installed,)
    service.disk_usage.return_value = ArtifactDiskUsage(1, 0, 64 * 1024 * 1024)
    calls = 0

    async def observe(references):
        nonlocal calls
        calls += 1
        return AudioCppModelLibraryObservationSnapshot(
            (
                AudioCppArtifactRemovalEvidence(
                    references[0],
                    settings_consumers=(
                        (("saved", "Guided Settings", "package"),) if calls == 1 else ()
                    ),
                ),
            )
        )

    view = InstalledView(
        service_factory=lambda: service,
        legacy_dir=tmp_path,
        observation_provider=observe,
    )
    app = _StyledInstalledApp(view)
    async with app.run_test(size=(80, 24)) as pilot:
        view.ensure_loaded()
        await _wait_until(
            pilot,
            lambda: "Configured: Saved Settings" in _rendered_static_text(view),
        )
        view.query_one(".model-delete", Button).focus()
        await pilot.pause()
        assert view.query_one(".model-delete", Button).has_focus
        view.refresh_observations()
        view.refresh_observations()
        await _wait_until(
            pilot,
            lambda: (
                "Configured: Not configured — exact Settings state checked"
                in _rendered_static_text(view)
            ),
        )
        assert view.query_one(".model-delete", Button).has_focus

    assert calls == 2


def test_curated_progress_tolerates_recompose_gap() -> None:
    """A progress event is retained while its widget is temporarily absent.

    ``apply_progress`` (called only by the host screen, ``LLMScreen`` --
    see its own docstring for why ``CuratedView`` no longer renders
    itself in response to a bubbled ``InstallProgressed`` -- TASK-596
    delta port fix round 1) shares this exact tolerance with the
    self-listening handler it replaced.
    """
    from tldw_chatbook.Model_Artifacts.acquisition import AcquisitionProgress
    from tldw_chatbook.UI.Screens.model_curated_view import CuratedView

    progress = AcquisitionProgress(
        "fetch",
        ArtifactRef("parakeet-v2", "immutable-revision", "int8"),
        "encoder.onnx",
        1,
        2,
    )
    view = CuratedView(service_factory=MagicMock(), registry_factory=MagicMock())
    view.query_one = MagicMock(side_effect=NoMatches)
    view.refresh = MagicMock()

    view.apply_progress(progress)

    assert view._progress is progress
    view.refresh.assert_called_once_with(recompose=True)


def test_models_rail_lists_surviving_destinations_without_a_downloader() -> None:
    """The Models rail keeps recovery destinations, not the retired browser."""
    from tldw_chatbook.UI.Screens.llm_screen import MODELS_RAIL_SECTIONS

    models_section = dict(MODELS_RAIL_SECTIONS)["Models"]
    keys = [key for key, _label in models_section]
    assert keys == ["curated", "installed", "external", "remote"]

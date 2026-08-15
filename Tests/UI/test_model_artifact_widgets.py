"""Focused Pilot tests for shared managed-model controls."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from textual import events, on

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.app import App, ComposeResult
from textual.widgets import Button, Checkbox, Static

from tldw_chatbook.Model_Artifacts.acquisition import (
    ArtifactPreflightEntry,
    PreflightReport,
)
from tldw_chatbook.Model_Artifacts.service import ArtifactRef, ProvenanceClass
from tldw_chatbook.Widgets.ModelArtifacts import LocalGGUFImportRequested


_BUNDLED_CSS = (
    Path(__file__).resolve().parents[2]
    / "tldw_chatbook"
    / "css"
    / "tldw_cli_modular.tcss"
)


def _report(
    destination: Path,
    *,
    sufficient_space: bool = True,
    gating_errors: tuple[str, ...] = (),
    repository: str = "publisher/parakeet-v2",
    license_id: str = "CC-BY-4.0",
    revision: str = "immutable-revision",
) -> PreflightReport:
    reference = ArtifactRef("parakeet-v2", "immutable-revision", "int8")
    return PreflightReport(
        root=reference,
        closure_fingerprint="f" * 64,
        entries=(
            ArtifactPreflightEntry(
                ref=reference,
                source_url="https://example.test/model",
                repository=repository,
                revision=revision,
                license_id=license_id,
                license_url="https://example.test/license",
                precision="int8",
                total_bytes=1024,
                file_count=4,
                already_installed=False,
                provenance=(ProvenanceClass.CHATBOOK_CURATED,),
            ),
        ),
        download_bytes=1024,
        already_staged_bytes=64,
        staging_overhead_bytes=128,
        retained_bytes=0,
        destination=destination,
        free_bytes=4096 if sufficient_space else 1,
        required_bytes=1152,
        sufficient_space=sufficient_space,
        gating_errors=gating_errors,
    )


class _PanelApp(ConsolidatedCSSApp):
    def __init__(
        self,
        report: PreflightReport,
        *,
        selected_file_details: tuple[tuple[str, int, str, str], ...] = (),
    ) -> None:
        self.report = report
        self.selected_file_details = selected_file_details
        super().__init__()

    def compose(self) -> ComposeResult:
        from tldw_chatbook.Widgets.ModelArtifacts import ModelPlanPanel

        yield ModelPlanPanel(
            self.report,
            model_label="Parakeet v2",
            selected_file_details=self.selected_file_details,
        )


class _ModalApp(ConsolidatedCSSApp):
    def compose(self) -> ComposeResult:
        return []


class _StyledModalApp(_ModalApp):
    """Modal harness using the exact stylesheet loaded by the production app."""

    CSS_PATH = _BUNDLED_CSS


class _ProgressApp(ConsolidatedCSSApp):
    def __init__(self, initial=None) -> None:
        self.initial = initial
        super().__init__()

    def compose(self) -> ComposeResult:
        from tldw_chatbook.Widgets.ModelArtifacts import ModelInstallProgress

        yield ModelInstallProgress(self.initial)


def test_progress_widget_idle_bar_is_hidden_before_mount() -> None:
    """The idle bar must be safe before Textual finishes mounting its children."""
    from textual.widgets import ProgressBar

    from tldw_chatbook.Widgets.ModelArtifacts import ModelInstallProgress

    widget = ModelInstallProgress()
    bar = next(item for item in widget.compose() if isinstance(item, ProgressBar))

    assert bar.display is False


@pytest.mark.asyncio
async def test_plan_panel_renders_every_consent_field(tmp_path: Path) -> None:
    """Closure, source, license, bytes, staging, and disk result are visible."""
    report = _report(tmp_path / "managed")
    app = _PanelApp(report)
    async with app.run_test() as pilot:
        await pilot.pause()
        text = "\n".join(str(item.renderable) for item in app.query(Static))

    for expected in (
        "publisher/parakeet-v2",
        "immutable-revision",
        "License: CC-BY-4.0",
        "Source review page: https://example.test/license",
        "int8",
        "4 files",
        str(tmp_path / "managed"),
        "Staging",
        "Enough free space",
        "Every declared file is checked against pinned sizes and SHA-256 digests "
        "before installation completes.",
    ):
        assert expected in text


@pytest.mark.asyncio
async def test_plan_panel_labels_noassertion_as_unknown_and_separates_review_url(
    tmp_path: Path,
) -> None:
    """Missing license metadata cannot be mistaken for a declared license.

    This fails if the panel shows the sentinel as a license or combines the
    source-review page with that license text.
    """
    report = _report(tmp_path / "managed", license_id="NOASSERTION")
    app = _PanelApp(report)
    async with app.run_test() as pilot:
        await pilot.pause()
        text = "\n".join(str(item.renderable) for item in app.query(Static))

    assert "License: Unknown / not declared" in text
    assert "Source review page: https://example.test/license" in text
    assert "License: NOASSERTION" not in text


@pytest.mark.asyncio
async def test_plan_panel_renders_selected_file_details_as_plain_bounded_text(
    tmp_path: Path,
) -> None:
    """Selected upstream identity and integrity values must be visible before consent."""
    digest = "a" * 64
    source = (
        "https://huggingface.co/owner/repository/resolve/"
        + ("b" * 40)
        + "/nested/model%20%5Bq4%5D.gguf"
    )
    app = _PanelApp(
        _report(tmp_path / "managed"),
        selected_file_details=(("nested/model [q4].gguf", 1_234_567, digest, source),),
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.query_one(".model-plan-panel", Static)
        text = str(panel.renderable)

    assert "Selected upstream files:" in text
    assert "Path: nested/model [q4].gguf" in text
    assert "Bytes: 1234567" in text
    assert f"SHA-256: {digest}" in text
    assert f"Pinned source URL: {source}" in text
    assert panel._render_markup is False


@pytest.mark.asyncio
async def test_install_is_disabled_when_report_is_not_grantable(
    tmp_path: Path,
) -> None:
    """The consent plan is a hard gate, not a warning shown after failure."""
    from tldw_chatbook.Widgets.ModelArtifacts import ModelInstallModal

    report = _report(
        tmp_path / "managed",
        sufficient_space=False,
        gating_errors=("Credential required",),
    )
    app = _ModalApp()
    async with app.run_test() as pilot:
        await app.push_screen(ModelInstallModal(report, model_label="Parakeet v2"))
        await pilot.pause()
        confirm = app.screen.query_one("#model-install-confirm", Button)
        text = "\n".join(str(item.renderable) for item in app.screen.query(Static))
        assert confirm.disabled is True
        assert "Not enough free space" in text
        assert "Credential required" in text


@pytest.mark.asyncio
async def test_confirm_and_cancel_return_decisions(tmp_path: Path) -> None:
    """The shared modal returns a decision and starts no acquisition work."""
    from tldw_chatbook.Widgets.ModelArtifacts import ModelInstallModal

    report = _report(tmp_path / "managed")
    app = _ModalApp()
    decisions: list[bool] = []
    async with app.run_test() as pilot:
        await app.push_screen(
            ModelInstallModal(report, model_label="Parakeet v2"),
            decisions.append,
        )
        await pilot.pause()
        assert app.screen.query_one("#model-install-confirm", Button).disabled is False
        await pilot.click("#model-install-confirm")
        await pilot.pause()
        assert decisions == [True]

        await app.push_screen(
            ModelInstallModal(report, model_label="Parakeet v2"),
            decisions.append,
        )
        await pilot.pause()
        await pilot.click("#model-install-cancel")
        await pilot.pause()
        assert decisions == [True, False]


@pytest.mark.parametrize("source", ["visible", "escape", "backdrop"])
@pytest.mark.asyncio
async def test_model_install_library_modal_contract_exact_negative_once(
    tmp_path: Path,
    source: str,
) -> None:
    from tldw_chatbook.Widgets.ModelArtifacts import ModelInstallModal

    app = _ModalApp()
    results: list[bool] = []
    modal = ModelInstallModal(
        _report(tmp_path / "managed", license_id="NOASSERTION"),
        model_label="Parakeet v2",
        required_acknowledgment="I reviewed the source.",
    )
    async with app.run_test(size=(100, 36)) as pilot:
        await app.push_screen(modal, callback=results.append)
        await pilot.pause()
        assert modal.query_one(".model-install-modal")

        if source == "visible":
            await pilot.click("#model-install-cancel")
        elif source == "escape":
            await pilot.press("escape")
        else:
            await pilot.click(offset=(0, 0))
        await pilot.pause()

    assert len(results) == 1
    assert results[0] is False
    assert modal._acknowledged is False


@pytest.mark.asyncio
async def test_model_install_library_modal_contract_inside_and_non_primary_stay_open(
    tmp_path: Path,
) -> None:
    from tldw_chatbook.Widgets.ModelArtifacts import ModelInstallModal

    app = _ModalApp()
    results: list[bool] = []
    modal = ModelInstallModal(_report(tmp_path / "managed"), model_label="Parakeet v2")
    async with app.run_test(size=(100, 36)) as pilot:
        await app.push_screen(modal, callback=results.append)
        await pilot.pause()
        await pilot.click(".model-plan-panel")
        event = events.Click(
            modal,
            x=0,
            y=0,
            delta_x=0,
            delta_y=0,
            button=3,
            shift=False,
            meta=False,
            ctrl=False,
            screen_x=0,
            screen_y=0,
        )
        await modal._dispatch_message(event)
        await pilot.pause()

        assert app.screen is modal
        assert results == []


@pytest.mark.asyncio
async def test_model_install_library_modal_contract_positive_is_exact_true(
    tmp_path: Path,
) -> None:
    from tldw_chatbook.Widgets.ModelArtifacts import ModelInstallModal

    app = _ModalApp()
    results: list[bool] = []
    modal = ModelInstallModal(_report(tmp_path / "managed"), model_label="Parakeet v2")
    async with app.run_test(size=(100, 36)) as pilot:
        await app.push_screen(modal, callback=results.append)
        await pilot.pause()
        await pilot.click("#model-install-confirm")
        await pilot.pause()

    assert len(results) == 1
    assert results[0] is True
    assert type(results[0]) is bool


@pytest.mark.asyncio
async def test_model_install_repeated_input_dismisses_once(tmp_path: Path) -> None:
    from tldw_chatbook.Widgets.ModelArtifacts import ModelInstallModal

    app = _ModalApp()
    results: list[bool] = []
    modal = ModelInstallModal(_report(tmp_path / "managed"), model_label="Parakeet v2")
    async with app.run_test(size=(100, 36)) as pilot:
        await app.push_screen(modal, callback=results.append)
        await pilot.pause()
        await pilot.press("escape", "escape")
        await pilot.pause()

    assert results == [False]


@pytest.mark.asyncio
async def test_install_acknowledgment_gates_only_callers_that_require_it(
    tmp_path: Path,
) -> None:
    """An unknown-license acknowledgment unlocks the shared Install control.

    This fails if the required checkbox does not gate consent, or if the modal
    fails to render the acknowledgment supplied by a caller.
    """
    from tldw_chatbook.Widgets.ModelArtifacts import ModelInstallModal

    report = _report(tmp_path / "managed", license_id="NOASSERTION")
    app = _ModalApp()
    async with app.run_test() as pilot:
        await app.push_screen(
            ModelInstallModal(
                report,
                model_label="Parakeet v2",
                required_acknowledgment=(
                    "No license was declared. I reviewed the source and want to continue."
                ),
            )
        )
        await pilot.pause()
        confirm = app.screen.query_one("#model-install-confirm", Button)
        checkbox = app.screen.query_one(Checkbox)
        assert checkbox.value is False
        assert confirm.disabled is True

        await pilot.click(Checkbox)
        await pilot.pause()

        assert checkbox.value is True
        assert confirm.disabled is False


@pytest.mark.asyncio
async def test_progress_widget_shows_each_phase_and_byte_detail() -> None:
    """Progress names all four phases and limits byte copy to byte phases."""
    from tldw_chatbook.Model_Artifacts.acquisition import AcquisitionProgress
    from tldw_chatbook.Widgets.ModelArtifacts import ModelInstallProgress

    reference = ArtifactRef("parakeet-v2", "immutable-revision", "int8")
    events = (
        AcquisitionProgress("fetch", reference, "encoder.onnx", 512, 1024),
        AcquisitionProgress("pre-verify", reference, "encoder.onnx", 1024, 1024),
        AcquisitionProgress("verify-install", reference, None, 0, 0),
        AcquisitionProgress("activate", reference, None, 0, 0),
    )
    expected_labels = (
        "Downloading",
        "Checking",
        "Verifying and installing",
        "Activating",
    )
    app = _ProgressApp()
    async with app.run_test() as pilot:
        widget = app.query_one(ModelInstallProgress)
        for index, (event, expected) in enumerate(zip(events, expected_labels)):
            widget.update_progress(event)
            await pilot.pause()
            text = "\n".join(str(item.renderable) for item in widget.query(Static))
            assert expected in text
            if index < 2:
                assert "encoder.onnx" in text
                assert "/" in text
            else:
                assert "encoder.onnx" not in text
                assert "/" not in text


def test_progress_callback_posts_a_message_without_touching_widgets() -> None:
    """The worker-thread callback only crosses the boundary with a message."""
    from textual.message import Message

    from tldw_chatbook.Model_Artifacts.acquisition import AcquisitionProgress
    from tldw_chatbook.Widgets.ModelArtifacts import (
        InstallProgressed,
        make_progress_callback,
    )

    received: list[Message] = []
    callback = make_progress_callback(received.append)
    event = AcquisitionProgress(
        "fetch",
        ArtifactRef("parakeet-v2", "immutable-revision", "int8"),
        "encoder.onnx",
        1,
        2,
    )

    callback(event)

    assert len(received) == 1
    assert isinstance(received[0], InstallProgressed)
    assert received[0].progress is event


@pytest.mark.asyncio
async def test_progress_widget_restores_the_latest_event_after_recompose() -> None:
    """A host recompose does not erase an in-flight install's status."""
    from tldw_chatbook.Model_Artifacts.acquisition import AcquisitionProgress

    event = AcquisitionProgress(
        "fetch",
        ArtifactRef("parakeet-v2", "immutable-revision", "int8"),
        "encoder.onnx",
        256,
        1024,
    )
    app = _ProgressApp(event)
    async with app.run_test() as pilot:
        await pilot.pause()
        text = "\n".join(str(item.renderable) for item in app.query(Static))

    assert "Downloading" in text
    assert "encoder.onnx" in text


@pytest.mark.asyncio
async def test_plan_panel_survives_bracket_bearing_repository_and_license_text(
    tmp_path: Path,
) -> None:
    """Repository ids, license strings, and revisions can contain square
    brackets (TASK-596 delta port); Rich would otherwise parse them as
    markup and eat them. ModelPlanPanel renders the whole plan as one
    ``Static(..., markup=False)`` -- this pins that the raw bracket text
    actually reaches the screen, not just that construction doesn't raise.

    Args:
        tmp_path: pytest fixture; used only as the plan's destination path.
    """
    report = _report(
        tmp_path / "managed",
        repository="org/model[experimental]",
        license_id="Custom[v1]",
        revision="rev[abc]",
    )
    app = _PanelApp(report)
    async with app.run_test() as pilot:
        await pilot.pause()
        text = "\n".join(str(item.renderable) for item in app.query(Static))

    assert "org/model[experimental]" in text
    assert "Custom[v1]" in text
    assert "rev[abc]" in text


@pytest.mark.asyncio
async def test_progress_callback_marshals_across_threads_not_direct_mutation() -> None:
    """The provision() callback runs on a worker thread; every host screen
    (CuratedView, InstalledView, LibraryScreen, LLMScreen) wires it as
    ``post_message -> InstallProgressed -> update_progress`` rather than
    calling ``update_progress`` directly off-thread. This proves that
    shared contract end-to-end with a REAL ``threading.Thread``, not just
    that ``make_progress_callback`` builds a callable that posts a message
    (already covered above) -- it additionally proves the widget update is
    only ever QUEUED by the worker thread, never applied by it.

    Deterministic, not a race: ``app.run_test()`` drives this app's entire
    message loop as an ``asyncio`` Task on the one event loop that also
    runs this test coroutine. ``thread.join()`` blocks this coroutine
    (never yielding to that loop) until the worker thread finishes, so
    nothing the worker thread scheduled onto the loop -- including
    ``post_message``'s ``call_soon_threadsafe`` hop for a foreign-thread
    caller -- can be processed until ``join()`` returns and this test
    reaches its next ``await``. So the instant ``join()`` returns, the
    posted update is guaranteed to still be sitting unprocessed: a
    callback that mutated the widget directly instead of posting would
    already show the new text right there, with no ``await`` involved.
    """
    import threading

    from textual import on

    from tldw_chatbook.Model_Artifacts.acquisition import AcquisitionProgress
    from tldw_chatbook.Widgets.ModelArtifacts import (
        InstallProgressed,
        ModelInstallProgress,
        make_progress_callback,
    )

    class _HostApp(ConsolidatedCSSApp):
        """Mirrors every real host's own wiring, not CuratedView's specifically."""

        def compose(self) -> ComposeResult:
            yield ModelInstallProgress(id="progress")

        @on(InstallProgressed)
        def _forward(self, event: InstallProgressed) -> None:
            self.query_one(ModelInstallProgress).update_progress(event.progress)

    reference = ArtifactRef("parakeet-v2", "immutable-revision", "int8")
    event = AcquisitionProgress(
        "fetch", reference, "encoder.onnx", 1_048_576, 2_097_152
    )

    app = _HostApp()
    async with app.run_test() as pilot:
        widget = app.query_one(ModelInstallProgress)
        await pilot.pause()
        detail = widget.query_one("#model-install-progress-detail", Static)
        text_before = str(detail.renderable)
        assert "encoder.onnx" not in text_before

        callback = make_progress_callback(app.post_message)
        errors: list[BaseException] = []

        def _invoke_off_thread() -> None:
            try:
                callback(event)
            except (
                BaseException
            ) as exc:  # pragma: no cover - fails the assertions below
                errors.append(exc)

        thread = threading.Thread(target=_invoke_off_thread)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert not errors

        # See this test's own docstring for why this is deterministic, not
        # a race: the widget must NOT have been updated yet.
        assert str(detail.renderable) == text_before

        await pilot.pause()
        assert "encoder.onnx" in str(detail.renderable)


@pytest.mark.asyncio
async def test_unready_root_activation_is_keyboard_reachable() -> None:
    """A valid installed root can be activated before readiness exists."""
    from tldw_chatbook.Widgets.ModelArtifacts.activation_controls import (
        ActivationRequested,
        ModelActivationControls,
    )

    reference = ArtifactRef("parakeet-v2", "immutable-revision", "int8")

    class _ActivationApp(ConsolidatedCSSApp):
        def __init__(self) -> None:
            self.requested: list[ArtifactRef] = []
            super().__init__()

        def compose(self) -> ComposeResult:
            yield ModelActivationControls(
                reference,
                active=False,
                ready=False,
                allow_activation=True,
            )

        def on_activation_requested(self, event: ActivationRequested) -> None:
            self.requested.append(event.reference)

    app = _ActivationApp()
    async with app.run_test() as pilot:
        activate = app.query_one(".model-activate", Button)
        assert activate.disabled is False
        activate.focus()
        await pilot.press("enter")
        await pilot.pause()

    assert app.requested == [reference]


@pytest.mark.asyncio
async def test_default_unready_controls_keep_activation_visible_but_disabled() -> None:
    """Legacy callers retain corrupt-model recovery controls without activation."""
    from tldw_chatbook.Widgets.ModelArtifacts.activation_controls import (
        ActivationRequested,
        ModelActivationControls,
    )

    reference = ArtifactRef("parakeet-v2", "immutable-revision", "int8")

    class _RecoveryApp(ConsolidatedCSSApp):
        def __init__(self) -> None:
            self.requested: list[ArtifactRef] = []
            super().__init__()

        def compose(self) -> ComposeResult:
            yield ModelActivationControls(reference, active=False, ready=False)

        def on_activation_requested(self, event: ActivationRequested) -> None:
            self.requested.append(event.reference)

    app = _RecoveryApp()
    async with app.run_test() as pilot:
        controls = app.query_one(ModelActivationControls)
        activate = app.query_one(".model-activate", Button)
        assert activate.disabled is True

        controls.set_pending(True)
        controls.set_pending(False)
        await pilot.pause()
        assert activate.disabled is True

        activate.focus()
        await pilot.press("enter")
        await pilot.pause()

    assert app.requested == []


class _ImportControlApp(ConsolidatedCSSApp):
    """Capture local-import requests through Textual's real message route."""

    def __init__(self, source: Path, *, pending: bool = False) -> None:
        self.source = source
        self.pending = pending
        self.received: list[Path] = []
        super().__init__()

    def compose(self) -> ComposeResult:
        from tldw_chatbook.Widgets.ModelArtifacts import LocalGGUFImportControls

        yield LocalGGUFImportControls(self.source, pending=self.pending)

    @on(LocalGGUFImportRequested)
    def _capture_import_request(self, event: LocalGGUFImportRequested) -> None:
        self.received.append(event.path)


class _StyledImportControlApp(_ImportControlApp):
    """Import-control harness using the production stylesheet bundle."""

    CSS_PATH = _BUNDLED_CSS


def _painted_text(app: App) -> str:
    """Return the text actually emitted by the screen compositor."""
    return "".join(
        segment.text
        for strip in app.screen._compositor.render_strips()
        for segment in strip
    )


def _painted_region_text(app: App, widget: Static) -> str:
    """Return ASCII text the compositor paints inside one widget's region."""
    lines: list[str] = []
    for y in range(widget.region.y, widget.region.bottom):
        cursor = 0
        parts: list[str] = []
        for segment in app.screen._compositor.render_strips()[y]:
            next_cursor = cursor + segment.cell_length
            start = max(widget.region.x, cursor)
            end = min(widget.region.right, next_cursor)
            if start < end:
                parts.append(segment.text[start - cursor : end - cursor])
            cursor = next_cursor
        lines.append("".join(parts))
    return "".join(lines)


def _relative_luminance(color) -> float:
    """Return WCAG relative luminance for a compositor-painted Rich colour."""
    triplet = color.get_truecolor()

    def channel(value: int) -> float:
        srgb = value / 255
        return srgb / 12.92 if srgb <= 0.04045 else ((srgb + 0.055) / 1.055) ** 2.4

    return (
        0.2126 * channel(triplet.red)
        + 0.7152 * channel(triplet.green)
        + 0.0722 * channel(triplet.blue)
    )


def _contrast(first, second) -> float:
    """Return the WCAG contrast ratio of two compositor-painted colours."""
    lighter, darker = sorted(
        (_relative_luminance(first), _relative_luminance(second)), reverse=True
    )
    return (lighter + 0.05) / (darker + 0.05)


def _painted_foreground_and_background(
    app: App, widget: Button
) -> tuple[object, object]:
    """Return the painted colours for the first visible button-label glyph."""
    for y in range(widget.region.y, widget.region.bottom):
        cursor = 0
        for segment in app.screen._compositor.render_strips()[y]:
            next_cursor = cursor + segment.cell_length
            overlaps = cursor < widget.region.right and next_cursor > widget.region.x
            if overlaps and segment.text.strip() and segment.style is not None:
                foreground = segment.style.color
                background = segment.style.bgcolor
                if foreground is not None and background is not None:
                    return foreground, background
            cursor = next_cursor
    raise AssertionError(f"no painted glyph colours inside {widget.region!r}")


@pytest.mark.asyncio
async def test_unmanaged_import_control_posts_the_exact_selected_path(
    tmp_path: Path,
) -> None:
    """The reusable row control sends intent only, preserving the selected Path."""
    source = tmp_path / "outside.gguf"
    app = _ImportControlApp(source)

    async with app.run_test() as pilot:
        await pilot.click(".model-import")
        await pilot.pause()

    assert app.received == [source]


@pytest.mark.asyncio
async def test_pending_unmanaged_import_control_is_disabled_and_posts_no_intent(
    tmp_path: Path,
) -> None:
    """A pending import cannot be started a second time through the row action."""
    app = _ImportControlApp(tmp_path / "outside.gguf", pending=True)

    async with app.run_test() as pilot:
        control = app.query_one(".model-import", Button)
        assert control.disabled is True
        control.focus()
        await pilot.press("enter")
        await pilot.pause()

    assert app.received == []


def test_pending_import_handler_does_not_post_when_dispatched_directly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The pending guard remains necessary even if an event bypasses disabled UI."""
    from tldw_chatbook.Widgets.ModelArtifacts import LocalGGUFImportControls

    control = LocalGGUFImportControls(tmp_path / "outside.gguf", pending=True)
    posted: list[object] = []
    stopped: list[bool] = []
    monkeypatch.setattr(control, "post_message", posted.append)

    control.on_button_pressed(SimpleNamespace(stop=lambda: stopped.append(True)))

    assert stopped == [True]
    assert posted == []


@pytest.mark.asyncio
async def test_pending_import_action_has_three_to_one_painted_contrast() -> None:
    """The production stylesheet keeps a pending Import label legible."""
    app = _StyledImportControlApp(Path("/private/model.gguf"), pending=True)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        button = app.query_one(".model-import", Button)
        foreground, background = _painted_foreground_and_background(app, button)

    assert _contrast(foreground, background) >= 3.0


@pytest.mark.asyncio
async def test_local_import_modal_states_copy_original_and_compatibility_truth(
    tmp_path: Path,
) -> None:
    """Consent says exactly what local import will and will not establish."""
    from tldw_chatbook.Widgets.ModelArtifacts import LocalGGUFImportConsentModal

    source = tmp_path / "user [private].gguf"
    app = _ModalApp()
    async with app.run_test() as pilot:
        await app.push_screen(LocalGGUFImportConsentModal(source, 4_194_304))
        await pilot.pause()
        statics = list(app.screen.query(Static))
        text = "\n".join(str(widget.renderable) for widget in statics)

    assert source.name in text
    assert str(source) in text
    assert "4.0 MiB" in text
    assert "managed copy" in text
    assert "original stays in place" in text
    assert "License and runtime compatibility are not verified" in text
    assert all(widget._render_markup is False for widget in statics)


@pytest.mark.asyncio
async def test_local_import_modal_confirm_cancel_and_escape_return_booleans(
    tmp_path: Path,
) -> None:
    """Every consent exit returns a boolean decision without starting import work."""
    from tldw_chatbook.Widgets.ModelArtifacts import LocalGGUFImportConsentModal

    source = tmp_path / "outside.gguf"
    app = _ModalApp()
    decisions: list[bool] = []
    async with app.run_test() as pilot:
        await app.push_screen(LocalGGUFImportConsentModal(source, 1), decisions.append)
        await pilot.pause()
        confirm = app.screen.query_one("#local-gguf-import-confirm", Button)
        cancel = app.screen.query_one("#local-gguf-import-cancel", Button)
        await pilot.press("tab")
        assert app.focused is cancel
        await pilot.press("tab")
        assert app.focused is confirm
        await pilot.press("shift+tab")
        assert app.focused is cancel
        await pilot.press("tab")
        assert app.focused is confirm
        await pilot.press("enter")
        await pilot.pause()

        await app.push_screen(LocalGGUFImportConsentModal(source, 1), decisions.append)
        await pilot.pause()
        await pilot.press("tab")
        assert app.focused is app.screen.query_one("#local-gguf-import-cancel", Button)
        await pilot.press("enter")
        await pilot.pause()

        await app.push_screen(LocalGGUFImportConsentModal(source, 1), decisions.append)
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

    assert decisions == [True, False, False]


@pytest.mark.asyncio
async def test_long_path_consent_keeps_facts_and_actions_painted_at_80_columns() -> (
    None
):
    """The 80x24 production modal scrolls facts without clipping its actions."""
    from tldw_chatbook.Widgets.ModelArtifacts import LocalGGUFImportConsentModal

    source = Path("/").joinpath(*(["very-long-directory-name"] * 24), "model.gguf")
    assert len(str(source)) >= 500
    app = _StyledModalApp()
    async with app.run_test(size=(80, 24)) as pilot:
        await app.push_screen(LocalGGUFImportConsentModal(source, 4_194_304))
        await pilot.pause()
        cancel = app.screen.query_one("#local-gguf-import-cancel", Button)
        confirm = app.screen.query_one("#local-gguf-import-confirm", Button)
        painted = _painted_text(app)

        for button, label in ((cancel, "Cancel"), (confirm, "Import")):
            assert button in app.screen._compositor.visible_widgets
            assert 0 <= button.region.x
            assert button.region.right <= app.size.width
            assert 0 <= button.region.y
            assert button.region.bottom <= app.size.height
            assert label in painted

        for protected_fact in (
            "managed copy",
            "original stays in place",
            "License and runtime compatibility are not verified",
        ):
            assert protected_fact in painted
        source_path = app.screen.query(Static)[1]
        assert _painted_region_text(app, source_path).rstrip() == str(source)

        await pilot.press("tab")
        assert app.focused is cancel
        await pilot.press("tab")
        assert app.focused is confirm


@pytest.mark.asyncio
async def test_progress_widget_reuses_byte_bar_for_local_copy_only() -> None:
    """Local import shares the stable display; only copy is a byte phase."""
    from tldw_chatbook.Model_Artifacts import LocalGGUFImportProgress
    from tldw_chatbook.Widgets.ModelArtifacts import ModelInstallProgress

    events = (
        LocalGGUFImportProgress("copy", "outside.gguf", 512, 1024),
        LocalGGUFImportProgress("inspect", None, 0, 0),
        LocalGGUFImportProgress("verify", None, 0, 0),
        LocalGGUFImportProgress("finalize", None, 0, 0),
    )
    expected_labels = (
        "Copying model into Chatbook",
        "Checking GGUF structure",
        "Verifying managed copy",
        "Finalizing managed model",
    )
    app = _ProgressApp()
    async with app.run_test() as pilot:
        widget = app.query_one(ModelInstallProgress)
        bar = widget.query_one("#model-install-progress-bar")
        for index, (event, expected_label) in enumerate(zip(events, expected_labels)):
            widget.update_progress(event)
            await pilot.pause()
            text = "\n".join(str(item.renderable) for item in widget.query(Static))
            assert expected_label in text
            assert bar.display is (index == 0)
            if index == 0:
                assert "outside.gguf" in text
                assert "/" in text
            else:
                assert "outside.gguf" not in text
                assert "/" not in text

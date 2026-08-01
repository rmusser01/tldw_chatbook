"""Focused Pilot tests for shared managed-model controls."""

from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Checkbox, Static

from tldw_chatbook.Model_Artifacts.acquisition import (
    ArtifactPreflightEntry,
    PreflightReport,
)
from tldw_chatbook.Model_Artifacts.service import ArtifactRef, ProvenanceClass


def _report(
    destination: Path,
    *,
    sufficient_space: bool = True,
    gating_errors: tuple[str, ...] = (),
    license_id: str = "CC-BY-4.0",
) -> PreflightReport:
    reference = ArtifactRef("parakeet-v2", "immutable-revision", "int8")
    return PreflightReport(
        root=reference,
        closure_fingerprint="f" * 64,
        entries=(
            ArtifactPreflightEntry(
                ref=reference,
                source_url="https://example.test/model",
                repository="publisher/parakeet-v2",
                revision="immutable-revision",
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


class _PanelApp(App):
    def __init__(self, report: PreflightReport) -> None:
        self.report = report
        super().__init__()

    def compose(self) -> ComposeResult:
        from tldw_chatbook.Widgets.ModelArtifacts import ModelPlanPanel

        yield ModelPlanPanel(self.report, model_label="Parakeet v2")


class _ModalApp(App):
    def compose(self) -> ComposeResult:
        return []


class _ProgressApp(App):
    def __init__(self, initial=None) -> None:
        self.initial = initial
        super().__init__()

    def compose(self) -> ComposeResult:
        from tldw_chatbook.Widgets.ModelArtifacts import ModelInstallProgress

        yield ModelInstallProgress(self.initial)


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

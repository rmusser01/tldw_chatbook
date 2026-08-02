"""Dedicated coverage for the Curated view (TASK-596 delta port).

``CuratedView`` already has a handful of tests scattered in
``Tests/UI/test_model_installed_view.py`` (no-I/O-at-compose, the
preflight-result-opens-the-modal path, and two recompose-gap tolerance
tests). This file adds the coverage that was still missing: that
``ensure_loaded`` actually performs the load and renders rows once
triggered (not just that it stays idle before that), that an installed
reference is marked as such rather than offered a redundant Install, that
no user-visible string contains "artifact", that a real Install click
reaches the shared consent modal, and two error/decline paths
(``_apply_preflight_result``'s failure branch and ``_confirm_install``'s
decline branch) that were not exercised anywhere else.

Adapted from the reference implementation's ``feat/model-artifact-browser``
branch, NOT copied: that branch's ``CuratedView`` takes ``service_root=``/
``registry=`` directly, posts an ``InstallRequested`` message, and never
calls ``preflight()``/``provision()`` itself -- ``LLMScreen`` owns those
workers there. dev's ``CuratedView`` (this branch) takes
``service_factory=``/``registry_factory=`` lazy factories and owns its own
preflight/provision workers directly, so every test here drives dev's
actual methods (``ensure_loaded``, ``_install_pressed``, ``_preflight_model``,
``_confirm_install``, ``_apply_preflight_result``) instead of the reference's
``activate()``/``InstallRequested``/screen-owned-worker shape. Tests that
only make sense against that other shape (the "no preflight/provision call
anywhere in this module" AST check, the "only one @work method" AST check,
and every ``LLMScreen``-owned-worker test) are dropped -- dev's module
genuinely does call preflight/provision itself, and asserting otherwise
would just be testing a design dev does not have.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

import tldw_chatbook.Model_Artifacts.acquisition as acquisition_module
from tldw_chatbook.Model_Artifacts import (
    ArtifactDescriptor,
    ArtifactFile,
    ArtifactFormat,
    ArtifactRef,
    ArtifactRole,
    ModelArtifactService,
    ProvenanceClass,
)
from tldw_chatbook.Model_Artifacts.acquisition import (
    ArtifactPreflightEntry,
    PreflightReport,
)
from tldw_chatbook.Model_Artifacts.curated_registry import CuratedRegistry
from tldw_chatbook.UI.Screens.model_curated_view import CuratedView
from tldw_chatbook.Widgets.ModelArtifacts import ModelInstallModal


class _ViewApp(App):
    def __init__(self, view: CuratedView) -> None:
        self.view = view
        super().__init__()

    def compose(self) -> ComposeResult:
        yield self.view


async def _wait_until(condition, *, pilot, attempts: int = 100) -> bool:
    for _ in range(attempts):
        if condition():
            return True
        await pilot.pause()
    return condition()


def _artifact_file(content: bytes, path: str = "model.bin") -> ArtifactFile:
    return ArtifactFile(path, len(content), hashlib.sha256(content).hexdigest())


def _descriptor(reference: ArtifactRef, content: bytes = b"payload") -> ArtifactDescriptor:
    files = (_artifact_file(content),)
    return ArtifactDescriptor(
        reference=reference,
        model_id=f"example/{reference.artifact_id}",
        role=ArtifactRole.ROOT,
        format=ArtifactFormat.GGUF,
        consumer="stt",
        model_family=reference.artifact_id,
        upstream_repository=f"example/{reference.artifact_id}",
        upstream_revision="main",
        source_url="https://example.test/model.bin",
        precision=reference.variant,
        license_id="mit",
        license_url="https://example.test/license",
        usage_notice="Review the upstream model card before use.",
        runtime_name="onnx-asr",
        runtime_version_constraint="==0.12.0",
        supported_os=("linux", "macos", "windows"),
        supported_architectures=("x86-64", "arm64"),
        provenance=(ProvenanceClass.CHATBOOK_CURATED, ProvenanceClass.LOCAL_INTEGRITY_RECORDED),
        files=files,
        dependencies=(),
        expected_installed_bytes=sum(f.size_bytes for f in files),
    )


def _registry_with(*descriptors: ArtifactDescriptor) -> CuratedRegistry:
    registry = CuratedRegistry()
    for descriptor in descriptors:
        registry.register(
            descriptor,
            sources={file.path: f"https://example.test/{file.path}" for file in descriptor.files},
        )
    return registry


def _report(reference: ArtifactRef, *, destination: Path) -> PreflightReport:
    entry = ArtifactPreflightEntry(
        ref=reference,
        source_url=f"https://example.test/{reference.artifact_id}/model.bin",
        repository=f"example/{reference.artifact_id}",
        revision=reference.revision,
        license_id="mit",
        license_url="https://example.test/license",
        precision=reference.variant,
        total_bytes=100_000,
        file_count=1,
        already_installed=False,
        provenance=(ProvenanceClass.CHATBOOK_CURATED,),
    )
    return PreflightReport(
        root=reference,
        closure_fingerprint="f" * 64,
        entries=(entry,),
        download_bytes=100_000,
        already_staged_bytes=0,
        staging_overhead_bytes=0,
        retained_bytes=0,
        destination=destination,
        free_bytes=10**12,
        required_bytes=200_000,
        sufficient_space=True,
        gating_errors=(),
    )


def _all_text(app: App) -> str:
    return "\n".join(str(static.renderable) for static in app.screen.query(Static))


def _install_buttons(app: App) -> list[Button]:
    return list(app.screen.query(".curated-install").results(Button))


# ---------------------------------------------------------------------------
# ensure_loaded actually performs the load and renders rows once triggered
# (test_curated_view_performs_no_io_at_compose_time in
# test_model_installed_view.py already pins the "stays idle before that"
# half; this is the other half).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_loaded_triggers_the_catalog_load_and_marks_installed_rows(
    tmp_path: Path,
) -> None:
    """An installed reference is marked as such, never offered a redundant Install.

    Args:
        tmp_path: pytest fixture; the managed store root and the on-disk
            source directory the installed descriptor is installed from.
    """
    installed_ref = ArtifactRef("model-a", "a" * 40, "int8")
    not_installed_ref = ArtifactRef("model-b", "b" * 40, "int8")
    installed_descriptor = _descriptor(installed_ref, b"payload-a")
    not_installed_descriptor = _descriptor(not_installed_ref, b"payload-b")
    registry = _registry_with(installed_descriptor, not_installed_descriptor)

    service = ModelArtifactService(tmp_path / "store")
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.bin").write_bytes(b"payload-a")
    service.install(installed_descriptor, source)

    view = CuratedView(
        service_factory=lambda: service,
        registry_factory=lambda: registry,
    )
    app = _ViewApp(view)
    async with app.run_test() as pilot:
        view.ensure_loaded()
        loaded = await _wait_until(lambda: view._loaded, pilot=pilot)
        assert loaded

        text = _all_text(app)
        assert "example/model-a" in text
        assert "example/model-b" in text

        buttons = _install_buttons(app)
        assert len(buttons) == 2
        by_reference = {button.reference: button for button in buttons}
        assert str(by_reference[installed_ref].label) == "Installed"
        assert by_reference[installed_ref].disabled is True
        assert str(by_reference[not_installed_ref].label) == "Review and install…"
        assert by_reference[not_installed_ref].disabled is False


@pytest.mark.asyncio
async def test_ensure_loaded_without_force_does_not_rerun_the_expensive_reads(
    tmp_path: Path, monkeypatch
) -> None:
    """A second ``ensure_loaded()`` without ``force=True`` must not repeat
    the ``list_installed()`` read; ``force=True`` must.

    Args:
        tmp_path: pytest fixture; the managed store root.
        monkeypatch: pytest fixture; wraps ``ModelArtifactService.
            list_installed`` to record each real invocation.
    """
    calls: list[str] = []
    original_list_installed = ModelArtifactService.list_installed

    def _tracked(self):
        calls.append("list_installed")
        return original_list_installed(self)

    monkeypatch.setattr(ModelArtifactService, "list_installed", _tracked)
    registry = _registry_with(_descriptor(ArtifactRef("model-a", "a" * 40, "int8")))
    service = ModelArtifactService(tmp_path / "store")

    view = CuratedView(service_factory=lambda: service, registry_factory=lambda: registry)
    app = _ViewApp(view)
    async with app.run_test() as pilot:
        view.ensure_loaded()
        loaded = await _wait_until(lambda: view._loaded, pilot=pilot)
        assert loaded
        assert calls.count("list_installed") == 1

        view.ensure_loaded()
        for _ in range(10):
            await pilot.pause()
        assert calls.count("list_installed") == 1, (
            "ensure_loaded() without force=True re-ran list_installed(); "
            "the _loaded guard is not preventing a repeat load"
        )

        view.ensure_loaded(force=True)
        refreshed = await _wait_until(
            lambda: calls.count("list_installed") == 2, pilot=pilot
        )
        assert refreshed


@pytest.mark.asyncio
async def test_no_user_visible_string_contains_artifact(tmp_path: Path) -> None:
    """No rendered Static text says "artifact" -- the UI says "model" throughout.

    Args:
        tmp_path: pytest fixture; the managed store root.
    """
    registry = _registry_with(_descriptor(ArtifactRef("model-a", "a" * 40, "int8")))
    service = ModelArtifactService(tmp_path / "store")
    view = CuratedView(service_factory=lambda: service, registry_factory=lambda: registry)
    app = _ViewApp(view)
    async with app.run_test() as pilot:
        view.ensure_loaded()
        loaded = await _wait_until(lambda: view._loaded, pilot=pilot)
        assert loaded

        assert "artifact" not in _all_text(app).lower()


# ---------------------------------------------------------------------------
# Install-request flow through to the consent modal.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_install_click_reaches_the_shared_consent_modal(
    tmp_path: Path, monkeypatch
) -> None:
    """A real Install click -- not a direct call to an internal method --
    resolves a plan (through a stubbed acquisition service, so this stays
    network-free) and pushes the exact shared ``ModelInstallModal``.

    Args:
        tmp_path: pytest fixture; the managed store root and the report's
            destination path.
        monkeypatch: pytest fixture; stubs ``ArtifactAcquisitionService``
            so preflight resolves without real network I/O.
    """
    reference = ArtifactRef("model-a", "a" * 40, "int8")
    descriptor = _descriptor(reference)
    registry = _registry_with(descriptor)
    service = ModelArtifactService(tmp_path / "store")
    report = _report(reference, destination=tmp_path / "dest")

    class _FakeAcquisitionService:
        def __init__(self, _service) -> None:
            pass

        async def preflight(self, ref, _registry, *, sources):
            assert ref == reference
            return report

    monkeypatch.setattr(
        acquisition_module, "ArtifactAcquisitionService", _FakeAcquisitionService
    )

    view = CuratedView(service_factory=lambda: service, registry_factory=lambda: registry)

    app = _ViewApp(view)
    async with app.run_test() as pilot:
        # Patched on the real, running app instance (not the CuratedView
        # class): this test needs the view's own `self.app` to resolve
        # normally so the real @work threaded worker and pilot.click()
        # both function -- only push_screen itself is stubbed, to capture
        # its arguments without actually pushing a screen.
        monkeypatch.setattr(app, "push_screen", MagicMock())

        view.ensure_loaded()
        loaded = await _wait_until(lambda: view._loaded, pilot=pilot)
        assert loaded

        button = _install_buttons(app)[0]
        await pilot.click(button)
        await pilot.pause()

        pushed = await _wait_until(lambda: app.push_screen.called, pilot=pilot)
        assert pushed, "clicking Install never reached push_screen"

        modal, callback = app.push_screen.call_args[0]
        assert isinstance(modal, ModelInstallModal)
        assert modal.report is report
        assert modal.model_label == descriptor.model_id
        assert callback == view._confirm_install
        assert view._pending_report is report


# ---------------------------------------------------------------------------
# Error / decline paths not otherwise exercised.
# ---------------------------------------------------------------------------


def test_preflight_failure_notifies_and_does_not_push_a_modal(monkeypatch) -> None:
    """The sibling success path is test_curated_preflight_result_opens_the_
    shared_modal in test_model_installed_view.py; this is its failure branch.

    Args:
        monkeypatch: pytest fixture; replaces the read-only ``app`` property
            on the ``CuratedView`` class with a ``MagicMock`` for this bare,
            unmounted view.
    """
    fake_app = MagicMock()
    # Class-level property patch (Screen/Widget.app is read-only) -- safe
    # here because this view is never mounted inside a real App/run_test(),
    # unlike test_install_click_reaches_the_shared_consent_modal above.
    monkeypatch.setattr(CuratedView, "app", property(lambda self: fake_app))
    view = CuratedView(service_factory=MagicMock(), registry_factory=MagicMock())
    reference = ArtifactRef("model-a", "a" * 40, "int8")
    view._operation_reference = reference
    view.notify = MagicMock()
    view.refresh = MagicMock()

    view._apply_preflight_result(None, "boom")

    assert view._operation_reference is None
    view.notify.assert_called_once_with("boom", severity="error")
    fake_app.push_screen.assert_not_called()
    view.refresh.assert_called_once_with(recompose=True)


def test_declining_the_consent_modal_does_not_start_the_install_worker() -> None:
    view = CuratedView(service_factory=MagicMock(), registry_factory=MagicMock())
    reference = ArtifactRef("model-a", "a" * 40, "int8")
    view._operation_reference = reference
    view._pending_report = object()
    view._provision_model = MagicMock()
    view.refresh = MagicMock()

    view._confirm_install(False)

    assert view._pending_report is None
    assert view._operation_reference is None
    view._provision_model.assert_not_called()
    view.refresh.assert_called_once_with(recompose=True)

"""Dedicated coverage for the Curated view (TASK-596 delta port; TASK-1803).

``CuratedView`` also has a handful of tests scattered in
``Tests/UI/test_model_installed_view.py`` (no-I/O-at-compose and one
recompose-gap tolerance test for ``apply_progress``). This file adds the
coverage that was still missing: that ``ensure_loaded`` actually performs
the load and renders rows once triggered (not just that it stays idle
before that), that an installed reference is marked as such rather than
offered a redundant Install, that no user-visible string contains
"artifact", and that a real Install click posts the exact intent message
the host screen needs.

TASK-1803 moved this view's preflight/provision workers to ``LLMScreen``,
mirroring how the reference implementation's ``feat/model-artifact-
browser`` branch always shaped ``CuratedView`` (``service_root=``/
``registry=``, posting ``InstallRequested``, never calling ``preflight()``/
``provision()`` itself) -- except this branch keeps its established
``service_factory=``/``registry_factory=`` lazy-factory constructor
instead of adopting that branch's ``service_root=``/``registry=`` shape,
since ``Tests/UI/test_model_installed_view.py`` and
``test_llm_screen_lab_adoption.py`` already depend on it. Tests that used
to drive this view's own ``_preflight_model``/``_confirm_install``/
``_apply_preflight_result`` directly (the plan-resolution and consent-
modal-push coverage, and the decline/failure paths that used to run
against them) moved to ``test_llm_screen_lab_adoption.py``, against
``LLMScreen``, which now owns that logic; what belongs here instead is
``CuratedView``'s own render-only contract: it posts
``InstallRequested`` and, once told the outcome, calls
``cancel_pending_install()``/``finish_install()``/``apply_progress()``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from textual import on
from textual.app import App, ComposeResult
from textual.css.query import NoMatches
from textual.widgets import Button, Static

from tldw_chatbook.Model_Artifacts import (
    ArtifactDescriptor,
    ArtifactFile,
    ArtifactFormat,
    ArtifactRef,
    ArtifactRole,
    ModelArtifactService,
    ProvenanceClass,
)
from tldw_chatbook.Model_Artifacts.curated_registry import CuratedRegistry
from tldw_chatbook.UI.Screens.model_curated_view import CuratedView


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
async def test_dependency_descriptors_do_not_render_as_standalone_models(
    tmp_path: Path,
) -> None:
    root = _descriptor(ArtifactRef("model-root", "a" * 40, "int8"))
    dependency = ArtifactDescriptor(
        **{
            **_descriptor(ArtifactRef("model-vad", "b" * 40, "f32")).__dict__,
            "role": ArtifactRole.DEPENDENCY,
        }
    )
    registry = _registry_with(root, dependency)
    view = CuratedView(
        service_factory=lambda: ModelArtifactService(tmp_path / "store"),
        registry_factory=lambda: registry,
    )
    app = _ViewApp(view)

    async with app.run_test() as pilot:
        view.ensure_loaded()
        assert await _wait_until(lambda: view._loaded, pilot=pilot)
        assert "example/model-root" in _all_text(app)
        assert "example/model-vad" not in _all_text(app)
        assert len(_install_buttons(app)) == 1


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
# Install-request flow: this view posts the intent and stops.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_install_click_posts_install_requested_with_the_resolved_service_and_registry(
    tmp_path: Path,
) -> None:
    """A real Install click -- not a direct call to an internal method --
    posts ``CuratedView.InstallRequested`` carrying the exact reference,
    service, registry, and source map the host screen needs to resolve a
    plan itself (TASK-1803: this view no longer performs that resolution;
    ``LLMScreen`` does). See ``test_llm_screen_lab_adoption.py``'s
    ``test_curated_install_click_reaches_the_shared_consent_modal`` for
    the end-to-end coverage of what happens once ``LLMScreen`` receives
    this message.

    Args:
        tmp_path: pytest fixture; the managed store root.
    """
    reference = ArtifactRef("model-a", "a" * 40, "int8")
    descriptor = _descriptor(reference)
    registry = _registry_with(descriptor)
    service = ModelArtifactService(tmp_path / "store")

    view = CuratedView(service_factory=lambda: service, registry_factory=lambda: registry)

    class _CapturingApp(App):
        def __init__(self, view: CuratedView) -> None:
            self.view = view
            self.requests: list[CuratedView.InstallRequested] = []
            super().__init__()

        def compose(self) -> ComposeResult:
            yield self.view

        @on(CuratedView.InstallRequested)
        def _capture(self, event: CuratedView.InstallRequested) -> None:
            self.requests.append(event)

    app = _CapturingApp(view)
    async with app.run_test() as pilot:
        view.ensure_loaded()
        loaded = await _wait_until(lambda: view._loaded, pilot=pilot)
        assert loaded

        button = _install_buttons(app)[0]
        await pilot.click(button)
        await pilot.pause()

        assert len(app.requests) == 1
        event = app.requests[0]
        assert event.reference == reference
        assert event.service is service
        assert event.registry is registry
        assert event.sources == {reference: registry.sources(reference)}

        # The clicked row's own button re-disables immediately (the
        # long-standing "cannot double-click Install" contract, unrelated
        # to whether LLMScreen has even received the message yet) -- via
        # a full recompose, so re-query rather than reuse the pre-click
        # Button instance, which the recompose already detached.
        refreshed_button = _install_buttons(app)[0]
        assert refreshed_button.disabled is True


# ---------------------------------------------------------------------------
# Render-only outcomes: cancel_pending_install() / finish_install().
#
# TASK-1803: the host screen (LLMScreen) is the only caller of either --
# after a preflight failure or an explicit consent-modal decline
# (cancel_pending_install(), no reload), or once provisioning completes,
# successfully or not (finish_install(), always reloads). Both replace
# this view's former _apply_preflight_result failure branch and
# _confirm_install decline branch, now that LLMScreen owns preflight/
# provision and only tells this view the outcome.
# ---------------------------------------------------------------------------


def test_cancel_pending_install_clears_the_indicator_without_reloading() -> None:
    """A preflight failure or a decline never started an install, so this
    must not reload the catalog -- unlike ``finish_install()`` below."""
    view = CuratedView(service_factory=MagicMock(), registry_factory=MagicMock())
    view._operation_reference = ArtifactRef("model-a", "a" * 40, "int8")
    view.refresh = MagicMock()
    view.ensure_loaded = MagicMock()

    view.cancel_pending_install()

    assert view._operation_reference is None
    view.refresh.assert_called_once_with(recompose=True)
    view.ensure_loaded.assert_not_called()


def test_finish_install_clears_the_indicator_and_reloads_despite_a_missing_progress_widget() -> None:
    """``finish_install()`` always reloads (a just-installed row must stop
    offering a redundant Install), and tolerates the progress widget being
    momentarily unfindable mid-recompose -- the same tolerance
    ``apply_progress`` documents for the same underlying reason."""
    view = CuratedView(service_factory=MagicMock(), registry_factory=MagicMock())
    view._operation_reference = ArtifactRef("model-a", "a" * 40, "int8")
    view._progress = object()
    view.query_one = MagicMock(side_effect=NoMatches)
    view.ensure_loaded = MagicMock()

    view.finish_install()

    assert view._operation_reference is None
    assert view._progress is None
    view.ensure_loaded.assert_called_once_with(force=True)


# ---------------------------------------------------------------------------
# Module-scope import boundary (TASK-1914 fix round 1).
#
# Not previously covered by a dedicated test for this module: the review
# that added the AST-based check for model_remote_view.py confirmed
# CuratedView has the identical gap (nothing here enforced "acquisition
# only inside functions" beyond code review), and the STT/Model_Artifacts
# subprocess import-recording suite (test_credentials_and_boundaries.py)
# does not cover this module either -- its script only ever imports
# STT/transcription/registry/store modules.
# ---------------------------------------------------------------------------


def test_curated_view_does_not_import_acquisition_at_module_scope() -> None:
    """``CuratedView`` posts intents; only ``LLMScreen``'s worker methods
    import ``Model_Artifacts.acquisition``.

    Reuses ``test_model_remote_view.py``'s AST-based ``module_scope_
    forbidden_acquisition_imports`` -- both modules are held to the
    identical rule, so one AST walker implementation backs both tests
    rather than two independent (and independently stale-able) substring
    scans.
    """
    import inspect

    from tldw_chatbook.UI.Screens import model_curated_view as module
    from Tests.UI.test_model_remote_view import (
        module_scope_forbidden_acquisition_imports,
    )

    source = inspect.getsource(module)
    assert "class CuratedView(Widget):" in source
    findings = module_scope_forbidden_acquisition_imports(source)
    assert findings == [], (
        f"model_curated_view.py imports acquisition/fetch at module scope: {findings}"
    )


@pytest.mark.asyncio
async def test_every_exact_parakeet_root_posts_use_from_disk_with_its_catalog_ref(
    tmp_path: Path,
) -> None:
    """The disk action carries the immutable catalog ref, never parsed row copy."""
    from tldw_chatbook.Local_Ingestion.parakeet_v2_artifact import (
        PARAKEET_PRECISIONS,
        parakeet_descriptor,
        parakeet_reference,
        parakeet_source_map,
        parakeet_vad_descriptor,
        parakeet_vad_reference,
    )
    from tldw_chatbook.Local_Ingestion.stt_batch_routing import (
        PARAKEET_V2_MODEL,
        PARAKEET_V3_MODEL,
    )

    registry = CuratedRegistry()
    source_map = parakeet_source_map()
    expected = []
    for model in (PARAKEET_V2_MODEL, PARAKEET_V3_MODEL):
        for precision in PARAKEET_PRECISIONS:
            reference = parakeet_reference(model, precision)
            expected.append(reference)
            registry.register(
                parakeet_descriptor(model, precision),
                sources=source_map[reference],
            )
    vad_reference = parakeet_vad_reference()
    registry.register(
        parakeet_vad_descriptor(),
        sources=source_map[vad_reference],
    )
    view = CuratedView(
        service_factory=lambda: ModelArtifactService(tmp_path / "store"),
        registry_factory=lambda: registry,
    )

    class _CapturingApp(App[None]):
        def __init__(self) -> None:
            self.requests: list[CuratedView.UseFromDiskRequested] = []
            super().__init__()

        def compose(self) -> ComposeResult:
            yield view

        @on(CuratedView.UseFromDiskRequested)
        def _capture(self, event: CuratedView.UseFromDiskRequested) -> None:
            self.requests.append(event)

    app = _CapturingApp()
    async with app.run_test(size=(100, 40)) as pilot:
        view.ensure_loaded()
        assert await _wait_until(lambda: view._loaded, pilot=pilot)
        buttons = list(app.query(".curated-use-from-disk").results(Button))
        assert [button.reference for button in buttons] == expected
        for button in buttons:
            button.press()
            await pilot.pause()

        assert [request.reference for request in app.requests] == expected
        assert all(button.reference != vad_reference for button in buttons)


@pytest.mark.asyncio
async def test_non_parakeet_root_has_no_use_from_disk_action(tmp_path: Path) -> None:
    descriptor = _descriptor(ArtifactRef("ordinary-model", "a" * 40, "int8"))
    view = CuratedView(
        service_factory=lambda: ModelArtifactService(tmp_path / "store"),
        registry_factory=lambda: _registry_with(descriptor),
    )
    app = _ViewApp(view)

    async with app.run_test() as pilot:
        view.ensure_loaded()
        assert await _wait_until(lambda: view._loaded, pilot=pilot)
        assert not app.query(".curated-use-from-disk")

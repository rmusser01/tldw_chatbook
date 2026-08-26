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
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from textual import on

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.app import App, ComposeResult
from textual.css.query import NoMatches
from textual.widgets import Button, Collapsible, Static
from textual.widgets._collapsible import CollapsibleTitle

from tldw_chatbook.app import TldwCli
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


@pytest.mark.parametrize(
    "function_name",
    [
        "model_library_focus_locator",
        "restore_model_library_focus",
        "project_audio_cpp_observation",
        "clear_audio_cpp_observation",
        "audio_cpp_package_projection",
    ],
)
def test_public_audio_cpp_projection_helpers_document_arguments_and_returns(
    function_name: str,
) -> None:
    from tldw_chatbook.UI.Screens import model_curated_view as module

    docstring = getattr(module, function_name).__doc__ or ""

    assert "Args:" in docstring
    assert "Returns:" in docstring


class _ViewApp(ConsolidatedCSSApp):
    def __init__(self, view: CuratedView) -> None:
        self.view = view
        super().__init__()

    def compose(self) -> ComposeResult:
        yield self.view


class _StyledViewApp(_ViewApp):
    """Curated harness using the exact production stylesheet bundle."""

    CSS_PATH = TldwCli.CSS_PATH


async def _wait_until(condition, *, pilot, attempts: int = 100) -> bool:
    for _ in range(attempts):
        if condition():
            return True
        await pilot.pause()
    return condition()


def _artifact_file(content: bytes, path: str = "model.bin") -> ArtifactFile:
    return ArtifactFile(path, len(content), hashlib.sha256(content).hexdigest())


def _descriptor(
    reference: ArtifactRef,
    content: bytes = b"payload",
    *,
    consumer: str = "stt",
) -> ArtifactDescriptor:
    files = (_artifact_file(content),)
    return ArtifactDescriptor(
        reference=reference,
        model_id=f"example/{reference.artifact_id}",
        role=ArtifactRole.ROOT,
        format=ArtifactFormat.GGUF,
        consumer=consumer,
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
        provenance=(
            ProvenanceClass.CHATBOOK_CURATED,
            ProvenanceClass.LOCAL_INTEGRITY_RECORDED,
        ),
        files=files,
        dependencies=(),
        expected_installed_bytes=sum(f.size_bytes for f in files),
    )


def _registry_with(*descriptors: ArtifactDescriptor) -> CuratedRegistry:
    registry = CuratedRegistry()
    for descriptor in descriptors:
        registry.register(
            descriptor,
            sources={
                file.path: f"https://example.test/{file.path}"
                for file in descriptor.files
            },
        )
    return registry


def _all_text(app: App) -> str:
    return "\n".join(str(static.renderable) for static in app.screen.query(Static))


def _install_buttons(app: App) -> list[Button]:
    return list(app.screen.query(".curated-install").results(Button))


def _painted_style_of_text(app: App, region, needle: str):
    """Return the compositor style carrying one visible label glyph."""
    strips = app.screen._compositor.render_strips()
    for y in range(region.y, region.bottom):
        cursor = 0
        for segment in strips[y]:
            next_cursor = cursor + segment.cell_length
            start = max(cursor, region.x)
            end = min(next_cursor, region.right)
            if start < end and needle in segment.text:
                return segment.style
            cursor = next_cursor
    return None


def _relative_luminance(color) -> float:
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
    lighter, darker = sorted(
        (_relative_luminance(first), _relative_luminance(second)), reverse=True
    )
    return (lighter + 0.05) / (darker + 0.05)


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

    view = CuratedView(
        service_factory=lambda: service, registry_factory=lambda: registry
    )
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
    view = CuratedView(
        service_factory=lambda: service, registry_factory=lambda: registry
    )
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

    view = CuratedView(
        service_factory=lambda: service, registry_factory=lambda: registry
    )

    class _CapturingApp(ConsolidatedCSSApp):
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


@pytest.mark.asyncio
async def test_install_press_during_observation_refresh_is_not_swallowed(
    tmp_path: Path,
) -> None:
    """Refreshing evidence cannot unmount an enabled Install before delivery."""
    import asyncio

    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import (
        audio_cpp_curated_entries,
    )
    from tldw_chatbook.TTS.audio_cpp_artifact_dependencies import (
        AudioCppArtifactRemovalEvidence,
        AudioCppModelLibraryObservationSnapshot,
    )

    descriptor, sources = audio_cpp_curated_entries()[0]
    registry = CuratedRegistry()
    registry.register(descriptor, sources=sources)
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

    view = CuratedView(
        service_factory=lambda: ModelArtifactService(tmp_path / "store"),
        registry_factory=lambda: registry,
        observation_provider=observe,
    )
    view.set_consumer_filter("audio_cpp")

    class _CapturingApp(_StyledViewApp):
        def __init__(self) -> None:
            self.requests: list[CuratedView.InstallRequested] = []
            super().__init__(view)

        @on(CuratedView.InstallRequested)
        def _capture(self, event: CuratedView.InstallRequested) -> None:
            self.requests.append(event)

    app = _CapturingApp()
    async with app.run_test(size=(80, 24)) as pilot:
        view.ensure_loaded()
        assert await _wait_until(lambda: calls == 1, pilot=pilot)
        install = view.query_one(".curated-install", Button)
        assert install.disabled is False

        view.refresh_observations()
        assert install.is_attached
        install.press()

        assert await _wait_until(refresh_entered.is_set, pilot=pilot)
        assert await _wait_until(lambda: len(app.requests) == 1, pilot=pilot)
        assert app.requests[0].reference == descriptor.reference
        release_refresh.set()


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


def test_finish_install_clears_the_indicator_and_reloads_despite_a_missing_progress_widget() -> (
    None
):
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal",
    ("cancel", "finish"),
)
async def test_curated_terminal_failure_persists_bounded_inline_recovery(
    tmp_path: Path,
    terminal: str,
) -> None:
    """Host-terminal failures remain visible after the operation recomposes."""
    reference = ArtifactRef("model-a", "a" * 40, "int8")
    descriptor = _descriptor(reference)
    view = CuratedView(
        service_factory=lambda: ModelArtifactService(tmp_path / "store"),
        registry_factory=lambda: _registry_with(descriptor),
    )
    app = _StyledViewApp(view)
    recovery = (
        "Pinned source unavailable — the app may be offline. "
        "Select Retry install when connectivity returns."
        if terminal == "cancel"
        else "Package verification failed (size or SHA-256). No package was "
        "promoted. Select Retry install."
    )

    async with app.run_test(size=(80, 24)) as pilot:
        view.ensure_loaded()
        assert await _wait_until(lambda: view._loaded, pilot=pilot)
        view._operation_reference = reference
        if terminal == "cancel":
            view.cancel_pending_install(recovery)
        else:
            view.finish_install(recovery)
        assert await _wait_until(
            lambda: (
                recovery in _all_text(app)
                and view.query_one(".curated-install", Button).has_focus
            ),
            pilot=pilot,
        )
        retry = view.query_one(".curated-install", Button)
        assert str(retry.label) == "Retry install…"
        assert retry.has_focus
        assert retry in app.screen._compositor.visible_widgets


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

    class _CapturingApp(ConsolidatedCSSApp):
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
        view._operation_reference = expected[0]
        view.refresh(recompose=True)
        await pilot.pause()
        use_from_disk = view.query_one(".curated-use-from-disk", Button)
        assert use_from_disk.disabled is True
        assert use_from_disk.tooltip == (
            "Use from disk unavailable — another model package operation is in progress."
        )
        assert (
            "Use from disk unavailable — another model package operation is in progress."
            in _all_text(app)
        )


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


@pytest.mark.asyncio
async def test_pending_curated_actions_expose_adjacent_reasons_to_tab_only_user_at_80x24(
    tmp_path: Path,
) -> None:
    """Tab skips inert actions while both reasons remain painted beside them."""
    from tldw_chatbook.Local_Ingestion.parakeet_v2_artifact import (
        parakeet_descriptor,
        parakeet_reference,
        parakeet_source_map,
    )
    from tldw_chatbook.Local_Ingestion.stt_batch_routing import PARAKEET_V2_MODEL

    reference = parakeet_reference(PARAKEET_V2_MODEL, "int8")
    registry = CuratedRegistry()
    registry.register(
        parakeet_descriptor(PARAKEET_V2_MODEL, "int8"),
        sources=parakeet_source_map()[reference],
    )
    view = CuratedView(
        service_factory=lambda: ModelArtifactService(tmp_path / "store"),
        registry_factory=lambda: registry,
    )
    view._operation_reference = reference
    app = _StyledViewApp(view)

    async with app.run_test(size=(80, 24)) as pilot:
        view.ensure_loaded()
        assert await _wait_until(lambda: view._loaded, pilot=pilot)
        refresh = view.query_one("#curated-models-refresh", Button)
        refresh.focus()
        await pilot.press("tab")

        assert app.focused is refresh
        assert view.query_one(".curated-install", Button).disabled is True
        assert view.query_one(".curated-use-from-disk", Button).disabled is True
        reasons = list(view.query(".curated-disabled-reason").results(Static))
        assert {str(reason.renderable) for reason in reasons} == {
            "Install unavailable — another model package operation is in progress.",
            "Use from disk unavailable — another model package operation is in progress.",
        }
        assert all(
            reason in app.screen._compositor.visible_widgets for reason in reasons
        )
        assert all(reason.region.right <= view.region.right for reason in reasons)
        assert all(reason.region.bottom <= view.region.bottom for reason in reasons)


@pytest.mark.asyncio
async def test_audio_cpp_mode_filters_catalog_and_exposes_joined_recipe_facts(
    tmp_path: Path,
) -> None:
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import audio_cpp_curated_entries

    audio_descriptor, audio_sources = audio_cpp_curated_entries()[0]
    ordinary = _descriptor(ArtifactRef("ordinary-model", "a" * 40, "int8"))
    registry = _registry_with(ordinary)
    registry.register(audio_descriptor, sources=audio_sources)
    view = CuratedView(
        service_factory=lambda: ModelArtifactService(tmp_path / "store"),
        registry_factory=lambda: registry,
    )
    view.set_consumer_filter("audio_cpp")
    app = _ViewApp(view)

    async with app.run_test(size=(120, 44)) as pilot:
        view.ensure_loaded()
        assert await _wait_until(lambda: view._loaded, pilot=pilot)
        text = _all_text(app)
        assert ordinary.model_id not in text
        assert audio_descriptor.model_id in text
        assert audio_descriptor.model_family in text
        assert (
            "Available: Complete pinned source recorded; live reachability not checked"
            in text
        )
        assert "Integrity: Not checked — package is not installed" in text
        assert "Recipe: audio-cpp-" in text
        assert "installed scan not checked" in text
        assert "Compatibility:" in text
        assert "Configured: Unknown — Settings state was not checked" in text
        assert "Running: Unknown — supervisor state was not checked" in text
        assert "Speech tasks:" in text
        assert "Required package files:" in text
        assert "Pinned source:" in text
        assert "Manifest authority: Pinned sizes and SHA-256 digests recorded" in text
        assert "Package size:" in text
        assert view.query_one(".audio-cpp-companions", Collapsible)
        assert "audiocpp_server is not included" in text
        assert "Available: Downloadable" not in text
        assert "Integrity: Verified" not in text
        assert "Recipe: Matched" not in text


@pytest.mark.parametrize("mismatch", ("revision", "repository"))
def test_audio_cpp_projection_rejects_mismatched_static_catalog_facts(
    mismatch: str,
) -> None:
    """Persisted descriptor drift cannot inherit canonical package claims."""
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import audio_cpp_curated_entries
    from tldw_chatbook.UI.Screens.model_curated_view import (
        audio_cpp_package_projection,
    )

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

    projection = audio_cpp_package_projection(mismatched)

    assert projection is not None
    assert projection.availability.startswith("Unknown")
    assert projection.recipe.startswith("Unknown")
    assert projection.pinned_source.startswith("Unknown")
    assert "attacker/repository" not in projection.pinned_source


@pytest.mark.parametrize("mismatch", ("revision", "repository"))
@pytest.mark.asyncio
async def test_curated_audio_cpp_mismatch_is_review_required_when_mounted(
    tmp_path: Path,
    mismatch: str,
) -> None:
    """Curated rows render mismatch truth without canonical source claims."""
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import audio_cpp_curated_entries

    descriptor, sources = audio_cpp_curated_entries()[0]
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
    registry = CuratedRegistry()
    registry.register(mismatched, sources=sources)
    view = CuratedView(
        service_factory=lambda: ModelArtifactService(tmp_path / "store"),
        registry_factory=lambda: registry,
    )
    view.set_consumer_filter("audio_cpp")
    app = _StyledViewApp(view)

    async with app.run_test(size=(80, 24)) as pilot:
        view.ensure_loaded()
        assert await _wait_until(lambda: view._loaded, pilot=pilot)
        text = _all_text(app)

    assert "Available: Unknown" in text
    assert "review required" in text
    assert "Recipe: Unknown" in text
    assert "Pinned source: Unknown" in text
    assert "Complete pinned source recorded" not in text
    assert "exact manifest mapping recorded" not in text
    assert "attacker/repository" not in text


@pytest.mark.asyncio
async def test_installed_audio_cpp_row_uses_shared_install_message_for_return(
    tmp_path: Path,
) -> None:
    descriptor = _descriptor(
        ArtifactRef("audio-cpp-model", "a" * 40, "f16"),
        b"audio-package",
        consumer="audio_cpp",
    )
    registry = _registry_with(descriptor)
    service = ModelArtifactService(tmp_path / "store")
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.bin").write_bytes(b"audio-package")
    service.install(descriptor, source)
    view = CuratedView(
        service_factory=lambda: service,
        registry_factory=lambda: registry,
    )
    view.set_consumer_filter("audio_cpp", allow_installed_return=True)

    class _CapturingApp(App[None]):
        def __init__(self) -> None:
            self.requests: list[CuratedView.InstallRequested] = []
            super().__init__()

        def compose(self) -> ComposeResult:
            yield view

        @on(CuratedView.InstallRequested)
        def _capture(self, event: CuratedView.InstallRequested) -> None:
            self.requests.append(event)

    app = _CapturingApp()
    async with app.run_test() as pilot:
        view.ensure_loaded()
        assert await _wait_until(lambda: view._loaded, pilot=pilot)
        button = _install_buttons(app)[0]
        assert str(button.label) == "Use installed package"
        assert button.disabled is False
        button.press()
        await pilot.pause()

    assert len(app.requests) == 1
    assert app.requests[0].reference == descriptor.reference
    assert app.requests[0].already_installed is True


@pytest.mark.asyncio
async def test_installed_audio_cpp_row_outside_handoff_is_not_a_return_action(
    tmp_path: Path,
) -> None:
    descriptor = _descriptor(
        ArtifactRef("audio-cpp-model", "a" * 40, "f16"),
        b"audio-package",
        consumer="audio_cpp",
    )
    registry = _registry_with(descriptor)
    service = ModelArtifactService(tmp_path / "store")
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.bin").write_bytes(b"audio-package")
    service.install(descriptor, source)
    view = CuratedView(
        service_factory=lambda: service,
        registry_factory=lambda: registry,
    )
    app = _ViewApp(view)

    async with app.run_test() as pilot:
        view.ensure_loaded()
        assert await _wait_until(lambda: view._loaded, pilot=pilot)
        button = _install_buttons(app)[0]
        assert str(button.label) == "Installed"
        assert button.disabled is True


@pytest.mark.asyncio
async def test_audio_cpp_row_is_truthful_expandable_and_keyboard_reachable_at_80x24(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One dense row keeps six lifecycle dimensions and its action reachable."""
    import tldw_chatbook.TTS.audio_cpp_recipes as recipes_module
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import audio_cpp_curated_entries

    descriptor, sources = audio_cpp_curated_entries()[0]
    recipe = next(
        item
        for item in recipes_module.AUDIO_CPP_RECIPE_REGISTRY.recipes
        if descriptor.reference.artifact_id in item.model_library_artifact_ids
    )
    signal = recipe.required_files[0]
    companion_paths = tuple(
        f"companions/voice-assets/reviewed-file-{index:02d}.json" for index in range(12)
    )
    recipe = replace(
        recipe,
        display_name=("Extremely long reviewed audio.cpp family and model name " * 3),
        required_files=recipe.required_files
        + tuple(replace(signal, relative_path=path) for path in companion_paths),
    )
    monkeypatch.setattr(
        recipes_module,
        "AUDIO_CPP_RECIPE_REGISTRY",
        SimpleNamespace(recipes=(recipe,)),
    )
    descriptor = replace(
        descriptor,
        model_id=("audio-cpp/very-long-reviewed-model-name-" * 5),
        model_family=("long-family-name-" * 5),
    )
    registry = _registry_with()
    registry.register(descriptor, sources=sources)
    view = CuratedView(
        service_factory=lambda: ModelArtifactService(tmp_path / "store"),
        registry_factory=lambda: registry,
    )
    view.set_consumer_filter("audio_cpp")
    app = _StyledViewApp(view)

    async with app.run_test(size=(80, 24)) as pilot:
        view.ensure_loaded()
        assert await _wait_until(lambda: view._loaded, pilot=pilot)
        text = _all_text(app)
        for fact in (
            descriptor.model_id,
            "Available: Complete pinned source recorded; live reachability not checked",
            "Integrity: Not checked — package is not installed",
            f"Recipe: {recipe.recipe_id}@{recipe.recipe_revision}",
            "installed scan not checked",
            "Compatibility:",
            "Configured: Unknown — Settings state was not checked",
            "Running: Unknown — supervisor state was not checked",
            "Speech tasks: TTS, CLONE",
            "Required package files:",
            "Pinned source:",
            "Manifest authority: Pinned sizes and SHA-256 digests recorded",
            "Package size:",
            "Model package only — audiocpp_server is not included",
        ):
            assert fact in text
        assert "Active" not in text
        assert "Available: Downloadable" not in text
        assert "Integrity: Verified" not in text
        assert "Recipe: Matched" not in text

        class _ComposeMustNotReadRecipes:
            @property
            def recipes(self):
                raise AssertionError("compose read the mutable recipe registry")

        monkeypatch.setattr(
            recipes_module,
            "AUDIO_CPP_RECIPE_REGISTRY",
            _ComposeMustNotReadRecipes(),
        )
        view.refresh(recompose=True)
        await pilot.pause()
        assert f"Recipe: {recipe.recipe_id}@{recipe.recipe_revision}" in _all_text(app)

        disclosure = view.query_one(".audio-cpp-companions", Collapsible)
        assert str(disclosure.title) == "Companion files (12)"
        assert disclosure.collapsed is True
        refresh = view.query_one("#curated-models-refresh", Button)
        app.screen.set_focus(refresh)
        await pilot.press("tab")
        title = disclosure.query_one(CollapsibleTitle)
        assert title.has_focus
        await pilot.press("enter")
        assert disclosure.collapsed is False
        assert all(path in _all_text(app) for path in companion_paths)

        await pilot.press("tab")
        install = view.query_one(".curated-install", Button)
        assert install.has_focus
        assert install in app.screen._compositor.visible_widgets
        assert install.region.right <= view.region.right
        assert install.region.bottom <= view.region.bottom


@pytest.mark.parametrize(
    "theme",
    ("textual-dark", "textual-light", "tokyo-night", "monokai", "dracula"),
)
@pytest.mark.asyncio
async def test_disabled_installed_audio_cpp_action_has_reason_and_three_to_one_contrast(
    tmp_path: Path,
    theme: str,
) -> None:
    """Installed is readable and explains why it is inert in every target theme."""
    descriptor = _descriptor(
        ArtifactRef("audio-cpp-model", "a" * 40, "f16"),
        b"audio-package",
        consumer="audio_cpp",
    )
    registry = _registry_with(descriptor)
    service = ModelArtifactService(tmp_path / "store")
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.bin").write_bytes(b"audio-package")
    service.install(descriptor, source)
    view = CuratedView(
        service_factory=lambda: service, registry_factory=lambda: registry
    )
    app = _StyledViewApp(view)
    app.theme = theme

    async with app.run_test(size=(80, 24)) as pilot:
        view.ensure_loaded()
        assert await _wait_until(lambda: view._loaded, pilot=pilot)
        button = view.query_one(".curated-install", Button)
        assert button.disabled is True
        assert button.styles.opacity == 1.0
        assert button.tooltip == (
            "Already installed — open Guided Settings to review this package."
        )
        assert "Installed — open Guided Settings to review this package." in _all_text(
            app
        )
        button.scroll_visible(animate=False, immediate=True, force=True)
        await pilot.pause()
        painted = _painted_style_of_text(app, button.region, "Installed")
        assert painted is not None
        assert painted.color is not None and painted.bgcolor is not None
        ratio = _contrast(painted.color, painted.bgcolor)
        assert ratio >= 3.0, f"Installed is {ratio:.2f}:1 under {theme}"


@pytest.mark.asyncio
async def test_refresh_restores_focus_after_curated_worker_recompose(
    tmp_path: Path,
) -> None:
    """Keyboard refresh returns focus to the newly mounted Refresh control."""
    descriptor = _descriptor(ArtifactRef("model-a", "a" * 40, "int8"))
    view = CuratedView(
        service_factory=lambda: ModelArtifactService(tmp_path / "store"),
        registry_factory=lambda: _registry_with(descriptor),
    )
    app = _StyledViewApp(view)

    async with app.run_test(size=(80, 24)) as pilot:
        view.ensure_loaded()
        assert await _wait_until(lambda: view._loaded, pilot=pilot)
        refresh = view.query_one("#curated-models-refresh", Button)
        app.screen.set_focus(refresh)
        await pilot.press("enter")
        assert await _wait_until(
            lambda: (
                not view._loading
                and view.query_one("#curated-models-refresh", Button).has_focus
            ),
            pilot=pilot,
        )


def test_default_only_evidence_does_not_claim_guided_settings_membership() -> None:
    """A global default is not evidence that the package is in Guided Settings."""
    from tldw_chatbook.TTS.audio_cpp_artifact_dependencies import (
        AudioCppArtifactRemovalEvidence,
    )
    from tldw_chatbook.UI.Screens.model_curated_view import (
        AudioCppPackageProjection,
        project_audio_cpp_observation,
    )

    reference = ArtifactRef("audio-cpp-model", "a" * 40, "f16")
    projection = AudioCppPackageProjection(
        recipe="recipe",
        compatibility="compatibility",
        availability="availability",
        companion_paths=(),
        speech_tasks="TTS",
        required_files="model.bin",
        pinned_source="source",
        manifest_authority="authority",
        package_size="1 byte",
    )
    evidence = AudioCppArtifactRemovalEvidence(
        reference,
        settings_consumers=(
            ("saved-default", "Saved global TTS default", "model"),
            ("draft-default", "Unsaved global TTS default", "model"),
        ),
    )

    observed = project_audio_cpp_observation(projection, reference, evidence)

    assert observed.configured == "Not configured — exact Settings state checked"
    assert "Saved Settings" not in observed.configured
    assert "draft" not in observed.configured


@pytest.mark.asyncio
async def test_all_audio_cpp_rows_share_one_bulk_observation_call(
    tmp_path: Path,
) -> None:
    """The catalog's many rows must not each trigger a full app evidence snapshot."""
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import audio_cpp_curated_entries
    from tldw_chatbook.TTS.audio_cpp_artifact_dependencies import (
        AudioCppModelLibraryObservationSnapshot,
    )

    registry = CuratedRegistry()
    for descriptor, sources in audio_cpp_curated_entries():
        registry.register(descriptor, sources=sources)
    calls = []

    async def observe(references):
        calls.append(references)
        return AudioCppModelLibraryObservationSnapshot(())

    view = CuratedView(
        service_factory=lambda: ModelArtifactService(tmp_path / "store"),
        registry_factory=lambda: registry,
        observation_provider=observe,
    )
    view.set_consumer_filter("audio_cpp")
    app = _ViewApp(view)
    async with app.run_test() as pilot:
        view.ensure_loaded()
        assert await _wait_until(lambda: view._loaded and bool(calls), pilot=pilot)
        await pilot.pause()

    assert len(registry.list()) >= 40
    assert calls == [(tuple(item.reference for item in registry.list()))]


@pytest.mark.parametrize("role", ("install", "disclosure"))
@pytest.mark.asyncio
async def test_delayed_bulk_observation_restores_curated_semantic_focus(
    tmp_path: Path,
    role: str,
) -> None:
    """Evidence refresh preserves the focused id-less row control in place."""
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import audio_cpp_curated_entries
    from tldw_chatbook.TTS.audio_cpp_artifact_dependencies import (
        AudioCppArtifactRemovalEvidence,
        AudioCppModelLibraryObservationSnapshot,
    )

    descriptor, sources = audio_cpp_curated_entries()[0]
    registry = CuratedRegistry()
    registry.register(descriptor, sources=sources)
    release = __import__("asyncio").Event()
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

    view = CuratedView(
        service_factory=lambda: ModelArtifactService(tmp_path / "store"),
        registry_factory=lambda: registry,
        observation_provider=observe,
    )
    view.set_consumer_filter("audio_cpp")
    app = _StyledViewApp(view)
    async with app.run_test(size=(80, 24)) as pilot:
        view.ensure_loaded()
        assert await _wait_until(lambda: view._loaded and bool(calls), pilot=pilot)
        target = (
            view.query_one(".curated-install", Button)
            if role == "install"
            else view.query_one(CollapsibleTitle)
        )
        target.focus()
        release.set()
        assert await _wait_until(
            lambda: "Configured: Saved Settings" in _all_text(app), pilot=pilot
        )
        assert target is app.focused
        assert (
            view.query_one(".curated-install", Button).has_focus
            if role == "install"
            else view.query_one(CollapsibleTitle).has_focus
        )

    assert calls == [(descriptor.reference,)]


@pytest.mark.asyncio
async def test_newer_bulk_observation_wins_over_delayed_old_generation(
    tmp_path: Path,
) -> None:
    """A cancelled old observer cannot overwrite the newer exact-ref projection."""
    import asyncio

    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import audio_cpp_curated_entries
    from tldw_chatbook.TTS.audio_cpp_artifact_dependencies import (
        AudioCppArtifactRemovalEvidence,
        AudioCppModelLibraryObservationSnapshot,
    )

    descriptor, sources = audio_cpp_curated_entries()[0]
    registry = CuratedRegistry()
    registry.register(descriptor, sources=sources)
    old_release = asyncio.Event()
    old_entered = asyncio.Event()
    calls = 0

    async def observe(references):
        nonlocal calls
        calls += 1
        assert references == (descriptor.reference,)
        if calls == 1:
            old_entered.set()
            try:
                await old_release.wait()
            except asyncio.CancelledError:
                await old_release.wait()
            consumers = (("saved", "Guided Settings", "old-package"),)
        else:
            consumers = ()
        return AudioCppModelLibraryObservationSnapshot(
            (
                AudioCppArtifactRemovalEvidence(
                    descriptor.reference,
                    settings_consumers=consumers,
                ),
            )
        )

    view = CuratedView(
        service_factory=lambda: ModelArtifactService(tmp_path / "store"),
        registry_factory=lambda: registry,
        observation_provider=observe,
    )
    view.set_consumer_filter("audio_cpp")
    app = _StyledViewApp(view)
    async with app.run_test(size=(80, 24)) as pilot:
        view.ensure_loaded()
        assert await _wait_until(old_entered.is_set, pilot=pilot)
        view.refresh_observations()
        assert await _wait_until(
            lambda: (
                calls == 2
                and "Configured: Not configured — exact Settings state checked"
                in _all_text(app)
            ),
            pilot=pilot,
        )
        old_release.set()
        await pilot.pause()
        await pilot.pause()
        assert "Configured: Saved Settings" not in _all_text(app)


@pytest.mark.asyncio
async def test_observation_refresh_clears_stale_affirmation_while_pending(
    tmp_path: Path,
) -> None:
    """A new generation cannot display the prior generation as current evidence."""
    import asyncio

    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import audio_cpp_curated_entries
    from tldw_chatbook.TTS.audio_cpp_artifact_dependencies import (
        AudioCppArtifactRemovalEvidence,
        AudioCppModelLibraryObservationSnapshot,
    )

    descriptor, sources = audio_cpp_curated_entries()[0]
    registry = CuratedRegistry()
    registry.register(descriptor, sources=sources)
    release_refresh = asyncio.Event()
    calls = 0

    async def observe(references):
        nonlocal calls
        calls += 1
        if calls == 2:
            await release_refresh.wait()
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

    view = CuratedView(
        service_factory=lambda: ModelArtifactService(tmp_path / "store"),
        registry_factory=lambda: registry,
        observation_provider=observe,
    )
    view.set_consumer_filter("audio_cpp")
    app = _StyledViewApp(view)
    async with app.run_test(size=(80, 24)) as pilot:
        view.ensure_loaded()
        assert await _wait_until(
            lambda: "Configured: Saved Settings" in _all_text(app), pilot=pilot
        )
        view.refresh_observations()
        await pilot.pause()
        text = _all_text(app)
        assert "Configured: Saved Settings" not in text
        assert "Configured: Unknown — Settings state was not checked" in text
        release_refresh.set()


@pytest.mark.asyncio
async def test_back_to_back_curated_refresh_starts_only_the_latest_generation(
    tmp_path: Path,
) -> None:
    """A deferred old start cannot cancel the newer observation worker."""
    from tldw_chatbook.TTS.audio_cpp_artifact_catalog import audio_cpp_curated_entries
    from tldw_chatbook.TTS.audio_cpp_artifact_dependencies import (
        AudioCppArtifactRemovalEvidence,
        AudioCppModelLibraryObservationSnapshot,
    )

    descriptor, sources = audio_cpp_curated_entries()[0]
    registry = CuratedRegistry()
    registry.register(descriptor, sources=sources)
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

    view = CuratedView(
        service_factory=lambda: ModelArtifactService(tmp_path / "store"),
        registry_factory=lambda: registry,
        observation_provider=observe,
    )
    view.set_consumer_filter("audio_cpp")
    app = _StyledViewApp(view)
    async with app.run_test(size=(80, 24)) as pilot:
        view.ensure_loaded()
        assert await _wait_until(
            lambda: "Configured: Saved Settings" in _all_text(app), pilot=pilot
        )
        view.query_one(".curated-install", Button).focus()
        await pilot.pause()
        assert view.query_one(".curated-install", Button).has_focus
        view.refresh_observations()
        view.refresh_observations()
        assert await _wait_until(
            lambda: (
                "Configured: Not configured — exact Settings state checked"
                in _all_text(app)
            ),
            pilot=pilot,
        )
        assert view.query_one(".curated-install", Button).has_focus

    assert calls == 2

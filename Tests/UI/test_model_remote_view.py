"""Focused behavior tests for lazy managed Remote model discovery."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from threading import Event
from unittest.mock import MagicMock

import httpx
import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Static

from tldw_chatbook.Model_Artifacts.remote_huggingface import (
    HuggingFaceRemoteAdapter,
    RemoteDiscoveryError,
    RemoteGGUFCandidate,
    RemoteGGUFFile,
    RemoteModelSummary,
    ResolvedRemoteModel,
    build_remote_catalog,
)
from tldw_chatbook.Model_Artifacts.acquisition import (
    AcquisitionProgress,
    ArtifactPreflightEntry,
    PreflightReport,
)
from tldw_chatbook.Model_Artifacts.service import ProvenanceClass


_COMMIT = "a" * 40
_DIGEST = "b" * 64


class _Resolver:
    def __init__(self, calls: list[str], token: str | None = "configured-token") -> None:
        self.calls = calls
        self.token = token

    def resolve(self, repository: str) -> str | None:
        self.calls.append(repository)
        return self.token


class _Adapter:
    def __init__(
        self,
        *,
        search_result: tuple[RemoteModelSummary, ...] = (),
        resolved: ResolvedRemoteModel | None = None,
    ) -> None:
        self.search_result = search_result
        self.resolved = resolved or _resolved()
        self.search_calls: list[tuple[str, str | None]] = []
        self.resolve_calls: list[tuple[str, str | None]] = []

    async def search(
        self,
        query: str,
        *,
        token: str | None = None,
    ) -> tuple[RemoteModelSummary, ...]:
        self.search_calls.append((query, token))
        return self.search_result

    async def resolve(
        self,
        repository: str,
        *,
        token: str | None = None,
    ) -> ResolvedRemoteModel:
        self.resolve_calls.append((repository, token))
        return self.resolved


def _summary(repository: str = "owner/repository") -> RemoteModelSummary:
    return RemoteModelSummary(
        repository=repository,
        private=False,
        gated="none",
        downloads=12,
        likes=3,
        last_modified="2026-08-01T00:00:00Z",
    )


def _candidate(label: str = "owner/repository · model-q4.gguf") -> RemoteGGUFCandidate:
    return RemoteGGUFCandidate(
        label=label,
        files=(RemoteGGUFFile("model-q4.gguf", 1024, _DIGEST),),
        total_bytes=1024,
    )


def _resolved(
    repository: str = "owner/repository",
    *,
    license_id: str = "apache-2.0",
    warnings: tuple[str, ...] = (),
) -> ResolvedRemoteModel:
    return ResolvedRemoteModel(
        repository=repository,
        commit=_COMMIT,
        license_id=license_id,
        review_url=f"https://huggingface.co/{repository}/tree/{_COMMIT}",
        candidates=(_candidate(f"{repository} · model-q4.gguf"),),
        total_candidate_count=1,
        warnings=warnings,
    )


def _catalog(*, license_id: str = "apache-2.0"):
    resolved = _resolved(license_id=license_id)
    return build_remote_catalog(resolved, resolved.candidates[0])


def _report_for(catalog, destination: Path) -> PreflightReport:
    descriptor = catalog.artifact
    return PreflightReport(
        root=descriptor.reference,
        closure_fingerprint="f" * 64,
        entries=(
            ArtifactPreflightEntry(
                ref=descriptor.reference,
                source_url=descriptor.source_url,
                repository=descriptor.upstream_repository,
                revision=descriptor.upstream_revision,
                license_id=descriptor.license_id,
                license_url=descriptor.license_url,
                precision=descriptor.precision,
                total_bytes=descriptor.expected_installed_bytes,
                file_count=len(descriptor.files),
                already_installed=False,
                provenance=(ProvenanceClass.LOCAL_INTEGRITY_RECORDED,),
            ),
        ),
        download_bytes=descriptor.expected_installed_bytes,
        already_staged_bytes=0,
        staging_overhead_bytes=128,
        retained_bytes=0,
        destination=destination,
        free_bytes=4096,
        required_bytes=descriptor.expected_installed_bytes + 128,
        sufficient_space=True,
        gating_errors=(),
    )


class _RemoteApp(App):
    def __init__(self, view) -> None:
        self.view = view
        self.install_statuses: list[object] = []
        super().__init__()

    def compose(self) -> ComposeResult:
        yield self.view

    def on_install_status_changed(self, event) -> None:
        self.install_statuses.append(event)


def _view(
    *,
    adapter_factory: Callable[[], object],
    resolver_factory: Callable[[], object],
    service_factory: Callable[[], object] = MagicMock,
):
    from tldw_chatbook.UI.Screens.model_remote_view import RemoteView

    return RemoteView(
        adapter_factory=adapter_factory,
        credential_resolver_factory=resolver_factory,
        service_factory=service_factory,
    )


async def _submit(app: _RemoteApp, pilot, query: str) -> None:
    app.view.query_one("#remote-model-query", Input).value = query
    await pilot.click("#remote-model-search")
    await app.workers.wait_for_complete()
    await pilot.pause()


def _text(view) -> str:
    return "\n".join(str(item.renderable) for item in view.query(Static))


@pytest.mark.asyncio
async def test_compose_and_mount_create_no_remote_dependencies_or_io() -> None:
    """An eager parent mount cannot instantiate any I/O-bearing dependency."""
    adapter_factory = MagicMock()
    resolver_factory = MagicMock()
    service_factory = MagicMock()
    view = _view(
        adapter_factory=adapter_factory,
        resolver_factory=resolver_factory,
        service_factory=service_factory,
    )
    app = _RemoteApp(view)

    async with app.run_test() as pilot:
        await pilot.pause()

    adapter_factory.assert_not_called()
    resolver_factory.assert_not_called()
    service_factory.assert_not_called()


@pytest.mark.asyncio
async def test_exact_repository_submission_resolves_without_searching() -> None:
    """Changing exact-ID classification must not add an unnecessary search request."""
    adapter = _Adapter(resolved=_resolved())
    resolver_calls: list[str] = []
    view = _view(
        adapter_factory=lambda: adapter,
        resolver_factory=lambda: _Resolver(resolver_calls),
    )
    app = _RemoteApp(view)

    async with app.run_test() as pilot:
        await _submit(app, pilot, "owner/repository")
        rendered = _text(view)

    assert adapter.search_calls == []
    assert adapter.resolve_calls == [("owner/repository", "configured-token")]
    assert resolver_calls == ["owner/repository"]
    assert "owner/repository · model-q4.gguf" in rendered


@pytest.mark.asyncio
async def test_free_text_search_resolves_only_after_result_selection() -> None:
    """Free text must remain a search and selection must resolve that exact result."""
    adapter = _Adapter(search_result=(_summary(),), resolved=_resolved())
    resolver_calls: list[str] = []
    view = _view(
        adapter_factory=lambda: adapter,
        resolver_factory=lambda: _Resolver(resolver_calls),
    )
    app = _RemoteApp(view)

    async with app.run_test() as pilot:
        await _submit(app, pilot, "quantized model")
        assert adapter.search_calls == [("quantized model", "configured-token")]
        assert adapter.resolve_calls == []
        assert "owner/repository" in _text(view)

        await pilot.click(".remote-result")
        await app.workers.wait_for_complete()
        await pilot.pause()
        rendered = _text(view)

    assert adapter.resolve_calls == [("owner/repository", "configured-token")]
    assert resolver_calls == ["quantized model", "owner/repository"]
    assert "Runtime compatibility has not been verified." in rendered
    assert "Local integrity recorded" in rendered


@pytest.mark.asyncio
async def test_stale_search_and_resolve_completions_cannot_replace_newer_results() -> None:
    """Removing either generation check must let an older completion overwrite state."""
    view = _view(adapter_factory=MagicMock(), resolver_factory=MagicMock())
    app = _RemoteApp(view)
    older_summary = _summary("old/result")
    newer_summary = _summary("new/result")
    older_resolved = _resolved("old/result")
    newer_resolved = _resolved("new/result")

    async with app.run_test() as pilot:
        query = view.query_one("#remote-model-query", Input)
        query.value = "new query"
        view._search_generation = 2
        view._apply_search_result(1, "old query", (older_summary,), None)
        view._apply_search_result(2, "new query", (newer_summary,), None)
        await pilot.pause()
        assert "new/result" in _text(view)
        assert "old/result" not in _text(view)

        view._resolve_generation = 4
        view._apply_resolve_result(
            3,
            "old/result",
            "old query",
            older_resolved,
            None,
        )
        view._apply_resolve_result(
            4,
            "new/result",
            "new query",
            newer_resolved,
            None,
        )
        await pilot.pause()
        rendered = _text(view)

    assert "new/result · model-q4.gguf" in rendered
    assert "old/result · model-q4.gguf" not in rendered


@pytest.mark.asyncio
async def test_same_generation_resolve_rejects_a_different_repository_response() -> None:
    """An adapter identity mismatch must never expose an installable candidate."""
    adapter = _Adapter(resolved=_resolved("other/repository"))
    view = _view(
        adapter_factory=lambda: adapter,
        resolver_factory=lambda: _Resolver([]),
    )
    app = _RemoteApp(view)

    async with app.run_test() as pilot:
        await _submit(app, pilot, "owner/repository")
        rendered = _text(view)
        candidate_buttons = list(view.query(".remote-candidate").results(Button))
        search_disabled = view.query_one("#remote-model-search", Button).disabled

    assert "other/repository · model-q4.gguf" not in rendered
    assert candidate_buttons == []
    assert view._selected_catalog is None
    assert search_disabled is False


@pytest.mark.asyncio
async def test_same_generation_resolve_rejects_when_repository_input_changes() -> None:
    """Input drift during one request must not make its completion installable."""
    started = Event()
    release = Event()

    class _WaitingAdapter(_Adapter):
        async def resolve(
            self,
            repository: str,
            *,
            token: str | None = None,
        ) -> ResolvedRemoteModel:
            self.resolve_calls.append((repository, token))
            started.set()
            release.wait(timeout=2)
            return self.resolved

    adapter = _WaitingAdapter(resolved=_resolved("owner/repository"))
    view = _view(
        adapter_factory=lambda: adapter,
        resolver_factory=lambda: _Resolver([]),
    )
    app = _RemoteApp(view)

    async with app.run_test() as pilot:
        query = view.query_one("#remote-model-query", Input)
        query.value = "owner/repository"
        await pilot.click("#remote-model-search")
        for _attempt in range(20):
            if started.is_set():
                break
            await pilot.pause()
        assert started.is_set(), "resolve worker did not reach the adapter"

        query.value = "new/repository"
        release.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        rendered = _text(view)
        candidate_buttons = list(view.query(".remote-candidate").results(Button))
        search_disabled = view.query_one("#remote-model-search", Button).disabled

    assert "owner/repository · model-q4.gguf" not in rendered
    assert candidate_buttons == []
    assert view._selected_catalog is None
    assert search_disabled is False


@pytest.mark.asyncio
async def test_resolution_mismatch_removes_stale_rendered_candidate_controls() -> None:
    """Cleared retained state must also remove previously mounted candidates."""
    view = _view(adapter_factory=MagicMock(), resolver_factory=MagicMock())
    app = _RemoteApp(view)
    old_resolved = _resolved("old/repository")
    new_resolved = _resolved("new/repository")

    async with app.run_test() as pilot:
        query = view.query_one("#remote-model-query", Input)
        query.value = "old/repository"
        view._resolve_generation = 1
        view._apply_resolve_result(
            1,
            "old/repository",
            "old/repository",
            old_resolved,
            None,
        )
        await pilot.pause()
        assert list(view.query(".remote-candidate").results(Button))

        view._resolve_remote = MagicMock()
        query.value = "new/repository"
        view._search_submitted()
        assert "Inspecting repository" in _text(view)

        query.value = "changed/repository"
        view._apply_resolve_result(
            2,
            "new/repository",
            "new/repository",
            new_resolved,
            None,
        )
        await pilot.pause()

        stale_controls = list(
            view.query(".remote-result, .remote-candidate").results(Button)
        )
        status = str(view.query_one("#remote-model-status", Static).renderable)
        search_disabled = view.query_one("#remote-model-search", Button).disabled
        query_disabled = query.disabled

    assert stale_controls == []
    assert view._selected_catalog is None
    assert view._operation_reference is None
    assert "Inspecting repository" not in status
    assert "Press Search" in status
    assert search_disabled is False
    assert query_disabled is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected"),
    (
        (
            RemoteDiscoveryError("authentication_required"),
            "Configure or verify Hugging Face access, then Retry.",
        ),
        (RemoteDiscoveryError("rate_limited", retryable=True), "Retry."),
        (RemoteDiscoveryError("network_error", retryable=True), "Retry."),
        (
            RemoteDiscoveryError("response_too_large"),
            "cannot be safely inspected",
        ),
        (
            RemoteDiscoveryError(
                "no_eligible_gguf",
                details=("owner/repository · model missing 00002",),
            ),
            "LFS-backed with size and SHA-256 metadata",
        ),
    ),
)
async def test_discovery_errors_render_sanitized_retry_guidance(
    error: RemoteDiscoveryError,
    expected: str,
) -> None:
    """Raw upstream details must never displace bounded recovery guidance."""
    view = _view(adapter_factory=MagicMock(), resolver_factory=MagicMock())
    app = _RemoteApp(view)

    async with app.run_test() as pilot:
        view.query_one("#remote-model-query", Input).value = "owner/repository"
        view._resolve_generation = 1
        view._apply_resolve_result(
            1,
            "owner/repository",
            "owner/repository",
            None,
            error,
        )
        await pilot.pause()
        rendered = _text(view)

    assert expected in rendered
    assert repr(error) not in rendered


@pytest.mark.asyncio
async def test_no_eligible_error_renders_bounded_incomplete_shard_details() -> None:
    """Validated missing-shard recovery details must follow the generic LFS rule."""
    detail = "owner/repository · model-q4 missing 00002 00004"
    error = RemoteDiscoveryError("no_eligible_gguf", details=(detail,))
    view = _view(adapter_factory=MagicMock(), resolver_factory=MagicMock())
    app = _RemoteApp(view)

    async with app.run_test() as pilot:
        view.query_one("#remote-model-query", Input).value = "owner/repository"
        view._resolve_generation = 1
        view._apply_resolve_result(
            1,
            "owner/repository",
            "owner/repository",
            None,
            error,
        )
        await pilot.pause()
        status = view.query_one("#remote-model-status", Static)
        rendered = str(status.renderable)

    generic = "Files must be LFS-backed with size and SHA-256 metadata."
    assert generic in rendered
    assert detail in rendered
    assert rendered.index(generic) < rendered.index(detail)
    assert status._render_markup is False


@pytest.mark.asyncio
async def test_oversized_lfs_size_recovers_without_rendering_a_candidate() -> None:
    """Hostile declared sizes must be rejected before the Remote view formats them."""
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "sha": _COMMIT,
                "siblings": [
                    {
                        "rfilename": "huge.gguf",
                        "lfs": {"size": 2**63, "sha256": _DIGEST},
                    }
                ],
                "cardData": None,
            },
        )

    adapter = HuggingFaceRemoteAdapter(
        client_factory=lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        )
    )
    view = _view(
        adapter_factory=lambda: adapter,
        resolver_factory=lambda: _Resolver([]),
    )
    app = _RemoteApp(view)

    async with app.run_test() as pilot:
        await _submit(app, pilot, "owner/repository")
        rendered = _text(view)
        candidates = list(view.query(".remote-candidate").results(Button))

    assert "No eligible GGUF files were found" in rendered
    assert str(2**63) not in rendered
    assert candidates == []


@pytest.mark.asyncio
async def test_incomplete_shard_warnings_render_bounded_candidate_and_indexes() -> None:
    """Dropping resolution warnings must hide the actionable missing-shard evidence."""
    warning = "owner/repository · model-q4 missing 00002 00004"
    resolved = _resolved(warnings=(warning,))
    view = _view(adapter_factory=MagicMock(), resolver_factory=MagicMock())
    app = _RemoteApp(view)

    async with app.run_test() as pilot:
        view.query_one("#remote-model-query", Input).value = "owner/repository"
        view._resolve_generation = 1
        view._apply_resolve_result(
            1,
            "owner/repository",
            "owner/repository",
            resolved,
            None,
        )
        await pilot.pause()
        rendered = _text(view)

    assert warning in rendered


@pytest.mark.asyncio
async def test_candidate_cap_discloses_deterministic_first_hundred() -> None:
    """A truncated candidate list must disclose its deterministic upstream order."""
    candidates = tuple(
        RemoteGGUFCandidate(
            label=f"owner/repository · {index:03d}.gguf",
            files=(RemoteGGUFFile(f"{index:03d}.gguf", 1, _DIGEST),),
            total_bytes=1,
        )
        for index in range(100)
    )
    resolved = ResolvedRemoteModel(
        repository="owner/repository",
        commit=_COMMIT,
        license_id="apache-2.0",
        review_url=f"https://huggingface.co/owner/repository/tree/{_COMMIT}",
        candidates=candidates,
        total_candidate_count=137,
        warnings=(),
    )
    view = _view(adapter_factory=MagicMock(), resolver_factory=MagicMock())
    app = _RemoteApp(view)

    async with app.run_test() as pilot:
        view._resolved = resolved
        view._refresh_with_status("Select one GGUF candidate.")
        await pilot.pause()
        rendered = _text(view)

    assert "First 100 of 137, sorted by upstream path" in rendered


@pytest.mark.asyncio
async def test_candidate_selection_freezes_catalog_and_disables_all_replacement_controls() -> None:
    """A pending plan must remain bound to the selected candidate."""
    resolved = _resolved()
    view = _view(adapter_factory=MagicMock(), resolver_factory=MagicMock())
    app = _RemoteApp(view)
    view._preflight_model = MagicMock()

    async with app.run_test() as pilot:
        view.query_one("#remote-model-query", Input).value = "owner/repository"
        view._resolve_generation = 1
        view._apply_resolve_result(
            1,
            "owner/repository",
            "owner/repository",
            resolved,
            None,
        )
        await pilot.pause()
        await pilot.click(".remote-candidate")
        await pilot.pause()

        expected = build_remote_catalog(resolved, resolved.candidates[0])
        assert view._selected_catalog == expected
        assert view._operation_reference == expected.artifact.reference
        assert view.query_one("#remote-model-query", Input).disabled is True
        assert view.query_one("#remote-model-search", Button).disabled is True
        assert view.query_one(".remote-candidate", Button).disabled is True


@pytest.mark.asyncio
async def test_stale_candidate_press_is_rejected_at_the_ui_boundary() -> None:
    """A queued button event from an old resolution must not start preflight."""
    old_resolved = _resolved("old/repository")
    current_resolved = _resolved("current/repository")
    view = _view(adapter_factory=MagicMock(), resolver_factory=MagicMock())
    view._preflight_model = MagicMock()
    app = _RemoteApp(view)

    async with app.run_test() as pilot:
        view._resolved = old_resolved
        view._refresh_with_status("Old resolution")
        await pilot.pause()
        stale_button = view.query_one(".remote-candidate", Button)

        view._resolved = current_resolved
        view._refresh_with_status("Current resolution")
        await pilot.pause()
        view._candidate_pressed(Button.Pressed(stale_button))
        await pilot.pause()

    view._preflight_model.assert_not_called()
    assert view._selected_catalog is None
    assert view._operation_reference is None


@pytest.mark.asyncio
async def test_preflight_receives_exact_catalog_sources_and_fresh_resolver(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Changing the catalog, source map, or credential seam must fail this boundary."""
    from tldw_chatbook.UI.Screens import model_remote_view as module

    catalog = _catalog()
    report = _report_for(catalog, tmp_path / "managed")
    core = object()
    resolver = object()
    captured: dict[str, object] = {}

    class _Acquisition:
        def __init__(self, received_core, *, credential_resolver) -> None:
            captured["core"] = received_core
            captured["resolver"] = credential_resolver

        async def preflight(self, root, received_catalog, *, sources):
            captured["preflight"] = (root, received_catalog, sources)
            return report

    monkeypatch.setattr(module, "ArtifactAcquisitionService", _Acquisition)
    resolver_factory = MagicMock(return_value=resolver)
    view = _view(
        adapter_factory=MagicMock(),
        resolver_factory=resolver_factory,
        service_factory=lambda: core,
    )

    actual = await view._preflight(catalog)

    assert actual is report
    assert captured == {
        "core": core,
        "resolver": resolver,
        "preflight": (
            catalog.artifact.reference,
            catalog,
            catalog.sources,
        ),
    }
    resolver_factory.assert_called_once_with()


@pytest.mark.asyncio
async def test_default_preflight_uses_env_config_credential_resolver(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The production path must not silently fall back to anonymous acquisition."""
    from tldw_chatbook.Model_Artifacts.acquisition import EnvConfigCredentialResolver
    from tldw_chatbook.UI.Screens import model_remote_view as module

    catalog = _catalog()
    report = _report_for(catalog, tmp_path / "managed")
    captured: list[object] = []

    class _Acquisition:
        def __init__(self, _core, *, credential_resolver) -> None:
            captured.append(credential_resolver)

        async def preflight(self, _root, _catalog, *, sources):
            return report

    monkeypatch.setattr(module, "ArtifactAcquisitionService", _Acquisition)
    view = module.RemoteView(service_factory=lambda: object())

    await view._preflight(catalog)

    assert len(captured) == 1
    assert isinstance(captured[0], EnvConfigCredentialResolver)


@pytest.mark.parametrize(
    ("license_id", "expected_acknowledgment"),
    (
        (
            "NOASSERTION",
            "No license was declared. I reviewed the source and want to continue.",
        ),
        ("apache-2.0", None),
    ),
)
def test_preflight_modal_requires_acknowledgment_only_for_unknown_license(
    license_id: str,
    expected_acknowledgment: str | None,
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Known licenses must not be gated and unknown licenses must be explicit."""
    from tldw_chatbook.UI.Screens import model_remote_view as module
    from tldw_chatbook.Widgets.ModelArtifacts import ModelInstallModal

    resolved = _resolved(license_id=license_id)
    candidate = resolved.candidates[0]
    catalog = build_remote_catalog(resolved, candidate)
    report = _report_for(catalog, tmp_path / "managed")
    fake_app = MagicMock()
    monkeypatch.setattr(module.RemoteView, "app", property(lambda self: fake_app))
    view = module.RemoteView()
    view._selected_catalog = catalog
    view._operation_reference = report.root

    view._apply_preflight_result(report, None, candidate)

    modal, callback = fake_app.push_screen.call_args.args
    assert isinstance(modal, ModelInstallModal)
    assert modal.required_acknowledgment == expected_acknowledgment
    assert modal.selected_file_details == (
        (
            "model-q4.gguf",
            1024,
            _DIGEST,
            f"https://huggingface.co/owner/repository/resolve/{_COMMIT}/model-q4.gguf",
        ),
    )
    assert callback == view._confirm_install


@pytest.mark.asyncio
async def test_provision_reuses_exact_preflight_values_without_activation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Any catalog/source substitution or activation would violate reviewed consent."""
    from tldw_chatbook.UI.Screens import model_remote_view as module

    catalog = _catalog()
    report = _report_for(catalog, tmp_path / "managed")
    core = object()
    resolver = object()
    captured: dict[str, object] = {}

    class _Acquisition:
        def __init__(self, received_core, *, credential_resolver) -> None:
            captured["core"] = received_core
            captured["resolver"] = credential_resolver

        async def provision(
            self,
            root,
            consent,
            received_catalog,
            *,
            sources,
            progress,
            activate,
        ):
            captured["provision"] = (
                root,
                consent,
                received_catalog,
                sources,
                progress,
                activate,
            )

    monkeypatch.setattr(module, "ArtifactAcquisitionService", _Acquisition)
    resolver_factory = MagicMock(return_value=resolver)
    view = _view(
        adapter_factory=MagicMock(),
        resolver_factory=resolver_factory,
        service_factory=lambda: core,
    )
    view.post_message = MagicMock()

    await view._provision(report, catalog)

    root, consent, actual_catalog, sources, progress, activate = captured["provision"]
    assert captured["core"] is core
    assert captured["resolver"] is resolver
    assert root == report.root
    assert consent == report.grant()
    assert actual_catalog is catalog
    assert sources is catalog.sources
    assert callable(progress)
    assert activate is False
    resolver_factory.assert_called_once_with()


@pytest.mark.asyncio
async def test_controls_stay_disabled_through_consent_and_reenable_after_failure(
    tmp_path: Path,
) -> None:
    """Consent must not create a window where Search can replace the plan."""
    catalog = _catalog()
    report = _report_for(catalog, tmp_path / "managed")
    view = _view(adapter_factory=MagicMock(), resolver_factory=MagicMock())
    app = _RemoteApp(view)
    view._preflight_model = MagicMock()
    view._provision_model = MagicMock()

    async with app.run_test() as pilot:
        view._resolved = _resolved()
        view._refresh_with_status("Select one GGUF candidate.")
        await pilot.pause()
        await pilot.click(".remote-candidate")
        await pilot.pause()
        view._apply_preflight_result(report, None)
        await pilot.pause()

        assert view.query_one("#remote-model-search", Button).disabled is True
        assert view.query_one(".remote-candidate", Button).disabled is True

        await pilot.click("#model-install-confirm")
        await pilot.pause()
        assert view.query_one("#remote-model-search", Button).disabled is True
        assert view.query_one(".remote-candidate", Button).disabled is True

        view._apply_provision_result("Managed download failed. Retry.")
        await pilot.pause()

        assert view.query_one("#remote-model-search", Button).disabled is False
        assert view.query_one(".remote-candidate", Button).disabled is False


@pytest.mark.asyncio
async def test_progress_and_completion_publish_shared_lifecycle_and_success_copy(
    tmp_path: Path,
) -> None:
    """Removing progress/completion messages must desynchronize Installed and Lab."""
    from tldw_chatbook.Widgets.ModelArtifacts import (
        InstallProgressed,
        InstallStatusChanged,
    )

    catalog = _catalog()
    report = _report_for(catalog, tmp_path / "managed")
    view = _view(adapter_factory=MagicMock(), resolver_factory=MagicMock())
    app = _RemoteApp(view)
    notifications: list[tuple[str, str]] = []
    view._selected_catalog = catalog
    view._pending_report = report
    view._operation_reference = report.root
    view._provision_model = MagicMock()

    async with app.run_test() as pilot:
        view.notify = lambda message, *, severity: notifications.append(
            (message, severity)
        )
        view._confirm_install(True)
        progress = AcquisitionProgress("fetch", report.root, "model.gguf", 1, 2)
        view._install_progressed(InstallProgressed(progress))
        await pilot.pause()
        assert "Downloading" in _text(view)

        view._apply_provision_result(None)
        await pilot.pause()

    statuses = [
        message
        for message in app.install_statuses
        if isinstance(message, InstallStatusChanged)
    ]
    assert [(item.active, item.succeeded) for item in statuses] == [
        (True, None),
        (False, True),
    ]
    assert notifications == [
        (
            "Model downloaded and managed. Runtime compatibility has not been verified.",
            "information",
        )
    ]
    assert view._pending_report is None
    assert view._operation_reference is None
    assert view._selected_catalog is None

from __future__ import annotations

from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Any

import pytest
from textual.widgets import Button, Input, Select, Static

from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.Event_Handlers.LLM_Management_Events.gguf_source_modes import (
    GGUFSourceMode,
    GGUFSourceSelection,
    ManagedGGUFChoice,
)
from tldw_chatbook.Event_Handlers.LLM_Management_Events.server_lifecycle import (
    ServerLaunchClaim,
    current_server_claim,
    release_server_claim,
    reserve_server_launch,
)
from tldw_chatbook.Model_Artifacts import ArtifactRef
from tldw_chatbook.UI import LLM_Management_Window as window_module
from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow
from tldw_chatbook.UI.Screens import llm_screen as llm_screen_module
from tldw_chatbook.UI.Screens.llm_screen import LLMScreen
from tldw_chatbook.app import TldwCli
from tldw_chatbook.config import get_cli_setting as _real_get_cli_setting


REF_A = ArtifactRef("managed-a", "a" * 40, "q4_k_m")
REF_B = ArtifactRef("managed-b", "b" * 40, "q8_0")
PRIVATE_MANAGED_PATH = "/private/chatbook-model-store/managed-a/model.gguf"
EXTERNAL_AUTHORITY = "Outside Chatbook · integrity unknown"
EXTERNAL_CONTRACT = (
    "This file is used in place and is not imported, copied, deleted, or "
    "selected globally."
)


class _InventoryService:
    def __init__(self, result: tuple[object, ...] = ()) -> None:
        self.result = result
        self.thread_ids: list[int] = []
        self.error: BaseException | None = None

    def list_installed(self) -> tuple[object, ...]:
        self.thread_ids.append(threading.get_ident())
        if self.error is not None:
            raise self.error
        return self.result


class _BlockingInventoryService(_InventoryService):
    def __init__(self, result: tuple[object, ...] = ()) -> None:
        super().__init__(result)
        self.started = threading.Event()
        self.release = threading.Event()

    def list_installed(self) -> tuple[object, ...]:
        self.thread_ids.append(threading.get_ident())
        self.started.set()
        self.release.wait(timeout=3)
        return self.result


@pytest.fixture(autouse=True)
def _deterministic_models_mount(monkeypatch: pytest.MonkeyPatch) -> None:
    async def probe_down(host: str = "127.0.0.1", port: int = 11434) -> bool:
        return False

    def no_splash(section: str, key: str | None = None, default: Any = None) -> Any:
        if section == "splash_screen" and key == "enabled":
            return False
        return _real_get_cli_setting(section, key, default)

    monkeypatch.setattr(llm_screen_module, "_probe_local_server", probe_down)
    monkeypatch.setattr("tldw_chatbook.app.get_cli_setting", no_splash)


def _rendered_text(root: Any) -> str:
    rendered: list[str] = []
    for widget in root.query("*"):
        try:
            rendered.append(str(widget.render()))
        except Exception:
            continue
    return "\n".join(rendered)


async def _mount_models(
    monkeypatch: pytest.MonkeyPatch,
    *,
    size: tuple[int, int] = (120, 40),
    choices: tuple[ManagedGGUFChoice, ...] = (),
    service: _InventoryService | None = None,
):
    inventory_service = service or _InventoryService()
    monkeypatch.setattr(window_module, "managed_service", lambda: inventory_service)
    monkeypatch.setattr(
        window_module,
        "managed_gguf_choices",
        lambda _installed: choices,
    )
    app = _build_test_app()
    assert app.CSS_PATH == TldwCli.CSS_PATH
    context = app.run_test(size=size)
    pilot = await context.__aenter__()
    screen = LLMScreen(app)
    await app.push_screen(screen)
    for _ in range(4):
        await pilot.pause()
    window = screen.query_one(LLMManagementWindow)
    return app, pilot, context, screen, window, inventory_service


async def _close_context(context: Any) -> None:
    await context.__aexit__(None, None, None)


def _select_values(select: Select) -> tuple[object, ...]:
    return tuple(value for _label, value in select._options if value is not Select.NULL)


@pytest.mark.asyncio
async def test_source_matrix_and_legacy_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, pilot, context, _screen, window, _service = await _mount_models(monkeypatch)
    try:
        llama_mode = window.query_one("#llamacpp-gguf-source-mode", Select)
        llamafile_mode = window.query_one("#llamafile-gguf-source-mode", Select)

        assert _select_values(llama_mode) == ("managed", "external")
        assert _select_values(llamafile_mode) == (
            "embedded",
            "managed",
            "external",
        )
        assert llama_mode.value == "external"
        assert llamafile_mode.value == "embedded"
        assert window.gguf_source_snapshot("llamacpp") == GGUFSourceSelection(
            GGUFSourceMode.EXTERNAL
        )
        assert window.gguf_source_snapshot("llamafile") == GGUFSourceSelection(
            GGUFSourceMode.EMBEDDED
        )

        legacy = "/outside/legacy.gguf"
        window.query_one("#llamafile-model-path", Input).value = legacy
        await pilot.pause()
        snapshot = window.gguf_source_snapshot("llamafile")
        assert snapshot.mode is GGUFSourceMode.EXTERNAL
        assert snapshot.external_path == Path(legacy)
        assert window.query_one("#llamafile-model-path", Input).value == legacy
    finally:
        await _close_context(context)


@pytest.mark.asyncio
async def test_switching_modes_preserves_inactive_exact_selections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    choices = (
        ManagedGGUFChoice(REF_A, "Model A · Q4_K_M · 4.0 GiB · Managed"),
        ManagedGGUFChoice(REF_B, "Model B · Q8_0 · 8.0 GiB · Managed"),
    )
    _app, pilot, context, _screen, window, _service = await _mount_models(
        monkeypatch,
        choices=choices,
    )
    try:
        path = window.query_one("#llamacpp-model-path", Input)
        path.value = "/outside/preserved.gguf"
        managed = window.query_one("#llamacpp-gguf-managed-select", Select)
        mode = window.query_one("#llamacpp-gguf-source-mode", Select)

        mode.value = "managed"
        managed.value = REF_B
        await pilot.pause()
        mode.value = "external"
        await pilot.pause()
        mode.value = "managed"
        await pilot.pause()

        snapshot = window.gguf_source_snapshot("llamacpp")
        assert snapshot.managed_ref == REF_B
        assert snapshot.external_path == Path("/outside/preserved.gguf")
        assert managed.value == REF_B
    finally:
        await _close_context(context)


@pytest.mark.asyncio
async def test_managed_selector_uses_exact_refs_and_path_free_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    choices = (
        ManagedGGUFChoice(
            REF_A,
            "Model A · Q4_K_M · 4096.0 MiB · Managed · integrity verified",
        ),
    )
    _app, _pilot, context, _screen, window, _service = await _mount_models(
        monkeypatch,
        choices=choices,
    )
    try:
        for provider in ("llamacpp", "llamafile"):
            select = window.query_one(f"#{provider}-gguf-managed-select", Select)
            assert _select_values(select) == (REF_A,)
            assert select.value == REF_A
        text = _rendered_text(window)
        assert "Model A" in text
        assert "Q4_K_M" in text
        assert "4096.0 MiB" in text
        assert "integrity verified" in text
        assert PRIVATE_MANAGED_PATH not in text
    finally:
        await _close_context(context)


@pytest.mark.asyncio
async def test_inventory_runs_off_loop_and_stale_results_are_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _InventoryService((SimpleNamespace(path=PRIVATE_MANAGED_PATH),))
    choices = (ManagedGGUFChoice(REF_A, "Model A · Q4_K_M · 4 MiB · Managed"),)
    main_thread = threading.get_ident()
    _app, pilot, context, _screen, window, _service = await _mount_models(
        monkeypatch,
        choices=choices,
        service=service,
    )
    try:
        for _ in range(20):
            if service.thread_ids:
                break
            await pilot.pause(0.02)
        assert service.thread_ids
        assert all(thread_id != main_thread for thread_id in service.thread_ids)

        current = window._managed_gguf_inventory_generation
        stale = ManagedGGUFChoice(REF_B, "Stale · Q8_0 · 8 MiB · Managed")
        window._apply_managed_gguf_inventory(current - 1, (stale,), None)
        assert REF_B not in _select_values(
            window.query_one("#llamacpp-gguf-managed-select", Select)
        )
        assert PRIVATE_MANAGED_PATH not in _rendered_text(window)
    finally:
        await _close_context(context)


@pytest.mark.asyncio
async def test_inventory_completion_cannot_write_through_replaced_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = (ManagedGGUFChoice(REF_A, "Current · Q4_K_M · 4 MiB · Managed"),)
    app, pilot, context, screen, window, _service = await _mount_models(
        monkeypatch,
        choices=initial,
    )
    try:
        generation = window._managed_gguf_inventory_generation
        screen.refresh(recompose=True)
        for _ in range(8):
            await pilot.pause()
            if screen.llm_window is not None and screen.llm_window is not window:
                break
        replacement = screen.query_one(LLMManagementWindow)
        assert replacement is not window
        assert not window.is_attached

        stale = (ManagedGGUFChoice(REF_B, "Stale · Q8_0 · 8 MiB · Managed"),)
        window._apply_managed_gguf_inventory(generation, stale, None)

        assert REF_B not in _select_values(
            replacement.query_one("#llamacpp-gguf-managed-select", Select)
        )
        assert current_server_claim(app, "llamacpp") is None
    finally:
        await _close_context(context)


@pytest.mark.asyncio
async def test_inventory_started_before_launch_cannot_overwrite_fenced_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _BlockingInventoryService((SimpleNamespace(path=PRIVATE_MANAGED_PATH),))
    choices = (ManagedGGUFChoice(REF_A, "Late · Q4_K_M · 4 MiB · Managed"),)
    app, pilot, context, _screen, window, _service = await _mount_models(
        monkeypatch,
        choices=choices,
        service=service,
    )
    try:
        for _ in range(20):
            if service.started.is_set():
                break
            await pilot.pause(0.02)
        assert service.started.is_set()
        assert window.gguf_source_snapshot("llamacpp").managed_ref is None

        claim = reserve_server_launch(app, "llamacpp", authority="External GGUF")
        assert claim is not None
        window._sync_process_controls("llamacpp")
        service.release.set()
        for _ in range(20):
            await pilot.pause(0.02)

        assert window.gguf_source_snapshot("llamacpp").managed_ref is None
        assert REF_A not in _select_values(
            window.query_one("#llamacpp-gguf-managed-select", Select)
        )
        assert "External GGUF" in str(
            window.query_one("#llamacpp-gguf-source-status", Static).render()
        )
    finally:
        service.release.set()
        await _close_context(context)


@pytest.mark.asyncio
async def test_inventory_failure_disables_only_managed_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _InventoryService()
    service.error = RuntimeError(PRIVATE_MANAGED_PATH)
    _app, pilot, context, _screen, window, _service = await _mount_models(
        monkeypatch,
        service=service,
    )
    try:
        for _ in range(20):
            managed = window.query_one("#llamacpp-gguf-managed-select", Select)
            if managed.disabled:
                break
            await pilot.pause(0.02)
        assert managed.disabled
        assert not window.query_one("#llamacpp-gguf-source-mode", Select).disabled
        assert not window.query_one("#llamacpp-model-path", Input).disabled
        assert not window.query_one("#llamacpp-browse-model-button", Button).disabled
        assert not window.query_one("#llamacpp-gguf-refresh-button", Button).disabled
        assert PRIVATE_MANAGED_PATH not in _rendered_text(window)
    finally:
        await _close_context(context)


@pytest.mark.asyncio
async def test_accepted_start_immediately_fences_every_source_control(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    executable = tmp_path / "llama-server"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    model = tmp_path / "outside.gguf"
    model.write_bytes(b"worker-validates-this-off-loop")
    app, pilot, context, _screen, window, _service = await _mount_models(monkeypatch)
    try:
        window.query_one("#llamacpp-exec-path", Input).value = str(executable)
        window.query_one("#llamacpp-model-path", Input).value = str(model)
        await pilot.pause()
        workers: list[tuple[object, dict[str, object]]] = []
        monkeypatch.setattr(
            app,
            "run_worker",
            lambda work, **kwargs: workers.append((work, kwargs)),
        )

        start = window.query_one("#llamacpp-start-server-button", Button)
        await window.on_button_pressed(Button.Pressed(start))

        assert len(workers) == 1
        assert current_server_claim(app, "llamacpp") is not None
        assert start.disabled
        assert not window.query_one("#llamacpp-stop-server-button", Button).disabled
        for control_id in (
            "llamacpp-gguf-source-mode",
            "llamacpp-gguf-managed-select",
            "llamacpp-gguf-refresh-button",
            "llamacpp-model-path",
            "llamacpp-browse-model-button",
            "llamacpp-exec-path",
            "llamacpp-browse-exec-button",
            "llamacpp-detect-exec-button",
        ):
            assert window.query_one(f"#{control_id}").disabled, control_id
    finally:
        await _close_context(context)


@pytest.mark.asyncio
async def test_claim_authority_survives_screen_recompose_and_not_window_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, pilot, context, screen, window, _service = await _mount_models(monkeypatch)
    try:
        claim = reserve_server_launch(app, "llamacpp", authority="Managed GGUF")
        assert claim is not None
        window.query_one("#llamacpp-gguf-source-mode", Select).value = "external"
        window._sync_process_controls("llamacpp")
        assert "Managed GGUF" in str(
            window.query_one("#llamacpp-gguf-source-status", Static).render()
        )

        screen.refresh(recompose=True)
        for _ in range(8):
            await pilot.pause()
            if screen.llm_window is not None and screen.llm_window is not window:
                break
        replacement = screen.query_one(LLMManagementWindow)
        assert replacement is not window
        assert "Managed GGUF" in str(
            replacement.query_one("#llamacpp-gguf-source-status", Static).render()
        )
        assert replacement.query_one("#llamacpp-gguf-source-mode", Select).disabled
    finally:
        await _close_context(context)


@pytest.mark.asyncio
async def test_status_updates_preserve_stop_identity_and_restore_focus_on_death(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, pilot, context, _screen, window, _service = await _mount_models(monkeypatch)
    try:
        claim = reserve_server_launch(app, "llamacpp", authority="External GGUF")
        assert claim is not None
        window._sync_process_controls("llamacpp")
        stop = window.query_one("#llamacpp-stop-server-button", Button)
        stop.focus()
        await pilot.pause()

        window._handle_server_process_state_change("llamacpp")
        window._handle_server_process_state_change("llamacpp")
        assert window.query_one("#llamacpp-stop-server-button", Button) is stop
        assert stop.has_focus

        assert release_server_claim(app, "llamacpp", claim)
        window._handle_server_process_state_change("llamacpp")
        await pilot.pause()
        assert not window.query_one("#llamacpp-start-server-button", Button).disabled
        assert window.query_one("#llamacpp-start-server-button", Button).has_focus
    finally:
        await _close_context(context)


@pytest.mark.asyncio
async def test_physical_stop_cancels_only_current_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, pilot, context, _screen, window, _service = await _mount_models(monkeypatch)
    try:
        stale = ServerLaunchClaim("llamacpp", authority="stale")
        current = reserve_server_launch(app, "llamacpp", authority="External GGUF")
        assert current is not None
        window._sync_process_controls("llamacpp")
        stop = window.query_one("#llamacpp-stop-server-button", Button)

        stop.focus()
        await pilot.press("enter")
        await pilot.pause()

        assert current.cancel_event.is_set()
        assert not stale.cancel_event.is_set()
    finally:
        await _close_context(context)


@pytest.mark.asyncio
async def test_external_copy_keyboard_geometry_and_unrelated_views_stay_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    long_choice = ManagedGGUFChoice(
        REF_A,
        "A very long managed model name that must not push actions out · "
        "Q4_K_M · 16384.0 MiB · Managed · local integrity recorded",
    )
    app, pilot, context, _screen, window, _service = await _mount_models(
        monkeypatch,
        size=(80, 24),
        choices=(long_choice,),
    )
    try:
        mode = window.query_one("#llamacpp-gguf-source-mode", Select)
        managed = window.query_one("#llamacpp-gguf-managed-select", Select)
        browse = window.query_one("#llamacpp-browse-model-button", Button)
        start = window.query_one("#llamacpp-start-server-button", Button)
        stop = window.query_one("#llamacpp-stop-server-button", Button)
        view = window.query_one("#llm-view-llama-cpp")

        for control in (mode, browse, start):
            control.scroll_visible(animate=False)
            await pilot.pause()
            assert control.region.width > 0 and control.region.height > 0
            assert control.region.x >= view.content_region.x
            assert control.region.right <= view.content_region.right

        mode.focus()
        await pilot.press("enter", "up", "enter")
        await pilot.pause()
        assert mode.value == "managed"
        managed.scroll_visible(animate=False)
        await pilot.pause()
        assert managed.region.right <= view.content_region.right
        assert (
            window.query_one("#llamacpp-gguf-refresh-button", Button).region.right
            <= view.content_region.right
        )

        mode.value = "external"
        await pilot.pause()
        external_region = window.query_one("#llamacpp-gguf-external-region")
        external_region.query_one(".gguf-source-copy", Static).scroll_visible(
            animate=False
        )
        await pilot.pause()
        text = _rendered_text(window)
        svg = app.export_screenshot(simplify=True)
        assert EXTERNAL_AUTHORITY in text
        assert EXTERNAL_CONTRACT in text
        assert all(token in svg for token in ("Outside", "Chatbook", "This", "used"))
        assert PRIVATE_MANAGED_PATH not in text + svg
        assert "<svg" in svg and "</svg>" in svg

        vllm = window.query_one("#vllm-model-path", Input)
        mlx = window.query_one("#mlx-model-path", Input)
        vllm.value, mlx.value = "org/vllm", "org/mlx"
        mode.value = "managed"
        await pilot.pause()
        assert (vllm.value, mlx.value) == ("org/vllm", "org/mlx")
        assert not vllm.disabled and not mlx.disabled

        claim = reserve_server_launch(app, "llamacpp", authority="Managed GGUF")
        assert claim is not None
        window._sync_process_controls("llamacpp")
        stop.scroll_visible(animate=False)
        stop.focus()
        await pilot.pause()
        assert stop.has_focus and stop.region.right <= view.content_region.right
    finally:
        await _close_context(context)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "view_name"),
    (("llamacpp", "llama-cpp"), ("llamafile", "llamafile")),
)
async def test_supported_width_keyboard_reaches_each_provider_source_and_actions(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    view_name: str,
) -> None:
    choices = (ManagedGGUFChoice(REF_A, "Model A · Q4_K_M · 4 MiB · Managed"),)
    app, pilot, context, _screen, window, _service = await _mount_models(
        monkeypatch,
        size=(80, 24),
        choices=choices,
    )
    try:
        if window.active_view != view_name:
            window.active_view = view_name
            await pilot.pause()
        assert window.active_view == view_name
        view = window.query_one(f"#llm-view-{view_name}")
        mode = window.query_one(f"#{provider}-gguf-source-mode", Select)
        start = window.query_one(f"#{provider}-start-server-button", Button)
        stop = window.query_one(f"#{provider}-stop-server-button", Button)

        mode.scroll_visible(animate=False)
        mode.focus()
        await pilot.pause()
        await pilot.press("enter")
        if provider == "llamacpp":
            await pilot.press("up")
        else:
            await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause()
        assert mode.value == "managed"

        await pilot.press("tab")
        await pilot.pause()
        managed = window.query_one(f"#{provider}-gguf-managed-select", Select)
        assert managed.has_focus
        for widget in (mode, managed, start):
            widget.scroll_visible(animate=False)
            await pilot.pause()
            assert widget.region.width > 0 and widget.region.height > 0
            assert widget.region.x >= view.content_region.x
            assert widget.region.right <= view.content_region.right
        claim = reserve_server_launch(app, provider, authority="Managed GGUF")
        assert claim is not None
        window._sync_process_controls(provider)
        stop.scroll_visible(animate=False)
        await pilot.pause()
        assert stop.has_focus
        assert stop.region.width > 0 and stop.region.height > 0
        assert stop.region.x >= view.content_region.x
        assert stop.region.right <= view.content_region.right
    finally:
        await _close_context(context)

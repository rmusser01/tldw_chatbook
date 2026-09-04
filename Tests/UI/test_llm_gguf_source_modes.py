from __future__ import annotations

import asyncio
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Any

import pytest
from textual.widgets import Button, Input, Select, Static
from textual.widgets._select import SelectOverlay

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
    mount_llamafile: bool = True,
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
    await _settle_pilot_until(
        pilot,
        lambda: (
            window.active_view == "llama-cpp"
            and len(window.query("#llm-view-remote")) == 1
        ),
        message="deferred model views did not finish mounting",
    )
    # Most of this module compares both GGUF providers. Under the lazy-once
    # lifecycle, visit llamafile once so those comparison tests retain their
    # original two-pane fixture while the dedicated deferral tests cover the
    # true first-arrival shape.
    if mount_llamafile:
        window.active_view = "llamafile"
        await _settle_pilot_until(
            pilot,
            lambda: len(window.query("#llamafile-gguf-source-mode")) == 1,
            message="llamafile pane did not populate on first selection",
        )
        window.active_view = "llama-cpp"
        await pilot.pause()
    return app, pilot, context, screen, window, inventory_service


async def _close_context(context: Any) -> None:
    await context.__aexit__(None, None, None)


def _select_values(select: Select) -> tuple[object, ...]:
    return tuple(value for _label, value in select._options if value is not Select.NULL)


def _assert_painted_inside(app: Any, widget: Any, view: Any) -> None:
    """Assert real compositor visibility inside both parent and view bounds."""

    assert widget in app.screen._compositor.visible_widgets
    assert widget.is_on_screen
    assert widget.region.width > 0 and widget.region.height > 0
    assert widget.parent is not None
    for bounds in (widget.parent.content_region, view.content_region):
        assert widget.region.x >= bounds.x
        assert widget.region.right <= bounds.right
        assert widget.region.y >= bounds.y
        assert widget.region.bottom <= bounds.bottom


def _assert_overlay_painted_on_screen(app: Any, overlay: SelectOverlay) -> None:
    """Assert an expanded Select overlay is compositor-painted inside the screen."""

    assert overlay in app.screen._compositor.visible_widgets
    assert overlay.is_on_screen
    assert overlay.region.width > 0 and overlay.region.height > 0
    bounds = app.screen.content_region
    assert overlay.region.x >= bounds.x
    assert overlay.region.right <= bounds.right
    assert overlay.region.y >= bounds.y
    assert overlay.region.bottom <= bounds.bottom
    strips = app.screen._compositor.render_strips()
    painted_rows = set()
    for y in range(overlay.region.y, overlay.region.bottom):
        for x in range(overlay.region.x, overlay.region.right):
            try:
                owner, _region = app.screen.get_widget_at(x, y)
            except Exception:
                continue
            if owner is not overlay and overlay not in owner.ancestors:
                continue
            if any(
                segment.text and segment.text[0].isalnum()
                for segment in strips[y].crop(x, x + 1)
            ):
                painted_rows.add(y)
    assert len(painted_rows) >= overlay.option_count


def _relative_luminance(rgb: tuple[int, int, int]) -> float:
    channels = tuple(
        value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
        for value in (channel / 255 for channel in rgb)
    )
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def _contrast_ratio(
    foreground: tuple[int, int, int], background: tuple[int, int, int]
) -> float:
    lighter, darker = sorted(
        (_relative_luminance(foreground), _relative_luminance(background)),
        reverse=True,
    )
    return (lighter + 0.05) / (darker + 0.05)


def _painted_text_contrast(
    app: Any,
    widget: Any,
) -> tuple[float, frozenset[tuple[tuple[int, int, int], tuple[int, int, int]]]]:
    """Measure final compositor colors for alphanumeric glyphs owned by a control."""

    strips = app.screen._compositor.render_strips()
    painted: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
    for y in range(widget.region.y, widget.region.bottom):
        for x in range(widget.region.x, widget.region.right):
            try:
                owner, _region = app.screen.get_widget_at(x, y)
            except Exception:
                continue
            if owner is not widget and widget not in owner.ancestors:
                continue
            cell = strips[y].crop(x, x + 1)
            for segment in cell:
                if not segment.text or not segment.text[0].isalnum():
                    continue
                style = segment.style
                if style is None or style.color is None or style.bgcolor is None:
                    continue
                foreground = tuple(style.color.get_truecolor())
                background = tuple(style.bgcolor.get_truecolor())
                painted.append((foreground, background))
    assert painted, f"no compositor-painted text colors for {widget.id}"
    return min(_contrast_ratio(*colors) for colors in painted), frozenset(painted)


async def _press_until_focus(
    pilot: Any,
    target: Any,
    *,
    key: str = "tab",
    limit: int = 80,
) -> tuple[str | None, ...]:
    """Traverse real keyboard focus until target, returning every visited id."""

    visited: list[str | None] = []
    for _ in range(limit):
        await pilot.press(key)
        await pilot.pause()
        focused = pilot.app.focused
        visited.append(focused.id if focused is not None else None)
        if target.has_focus:
            return tuple(visited)
    raise AssertionError(f"{key} did not reach {target.id}; visited={visited}")


async def _settle_pilot_until(
    pilot: Any,
    predicate: Any,
    *,
    message: str,
) -> None:
    """Pump bounded Pilot cycles until deferred Textual work reaches a condition."""

    try:
        async with asyncio.timeout(10):
            while True:
                await pilot.pause()
                if predicate():
                    return
    except TimeoutError:
        raise AssertionError(message) from None


@pytest.mark.asyncio
async def test_pilot_settle_waits_for_deferred_refresh_cycles() -> None:
    class DelayedPilot:
        cycles = 0

        async def pause(self) -> None:
            self.cycles += 1

    pilot = DelayedPilot()
    await _settle_pilot_until(
        pilot,
        lambda: pilot.cycles == 21,
        message="deferred Textual work did not settle",
    )
    assert pilot.cycles == 21


@pytest.mark.asyncio
async def test_pilot_settle_reports_a_clear_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class YieldingPilot:
        async def pause(self) -> None:
            await asyncio.sleep(0)

    real_timeout = asyncio.timeout
    monkeypatch.setattr(asyncio, "timeout", lambda _seconds: real_timeout(0.001))
    with pytest.raises(AssertionError, match="condition did not settle"):
        await _settle_pilot_until(
            YieldingPilot(),
            lambda: False,
            message="condition did not settle",
        )


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
async def test_external_mode_requires_a_path_before_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _app, pilot, context, _screen, window, _service = await _mount_models(monkeypatch)
    try:
        model_path = window.query_one("#llamacpp-model-path", Input)
        start = window.query_one("#llamacpp-start-server-button", Button)
        status = window.query_one("#llamacpp-gguf-source-status", Static)

        assert window.gguf_source_snapshot("llamacpp").mode is GGUFSourceMode.EXTERNAL
        assert start.disabled
        assert str(status.render()) == "Choose an external GGUF file to enable Start."

        model_path.value = "/outside/model.gguf"
        await pilot.pause()
        assert not start.disabled
        assert window.gguf_source_snapshot("llamacpp") == GGUFSourceSelection(
            GGUFSourceMode.EXTERNAL,
            external_path=Path("/outside/model.gguf"),
        )

        model_path.value = ""
        await pilot.pause()
        assert start.disabled
        assert str(status.render()) == "Choose an external GGUF file to enable Start."
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
async def test_configure_managed_gguf_opens_runtime_and_preselects_exact_ref(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remote adoption configures source state without starting or activating."""
    choices = (
        ManagedGGUFChoice(REF_A, "Model A · Q4_K_M · 4 MiB · Managed"),
        ManagedGGUFChoice(REF_B, "Model B · Q8_0 · 8 MiB · Managed"),
    )
    app, pilot, context, _screen, window, _service = await _mount_models(
        monkeypatch,
        choices=choices,
    )
    try:
        accepted = window.configure_managed_gguf("llamafile", REF_B)
        await pilot.pause()

        selection = window.gguf_source_snapshot("llamafile")
        assert accepted is True
        assert window.active_view == "llamafile"
        assert selection.mode is GGUFSourceMode.MANAGED
        assert selection.managed_ref == REF_B
        assert (
            window.query_one("#llamafile-gguf-source-mode", Select).value == "managed"
        )
        assert window.query_one("#llamafile-gguf-managed-select", Select).value == REF_B
        assert current_server_claim(app, "llamafile") is None
    finally:
        await _close_context(context)


@pytest.mark.asyncio
async def test_configure_managed_gguf_waits_for_first_runtime_mount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exact handoff survives the target runtime's first lazy mount."""

    choices = (
        ManagedGGUFChoice(REF_A, "Model A · Q4_K_M · 4 MiB · Managed"),
        ManagedGGUFChoice(REF_B, "Model B · Q8_0 · 8 MiB · Managed"),
    )
    app, pilot, context, _screen, window, _service = await _mount_models(
        monkeypatch,
        choices=choices,
        mount_llamafile=False,
    )
    try:
        assert not list(window.query("#llamafile-gguf-source-mode"))

        assert window.configure_managed_gguf("llamafile", REF_B) is True
        await _settle_pilot_until(
            pilot,
            lambda: (
                len(window.query("#llamafile-gguf-managed-select")) == 1
                and window.gguf_source_snapshot("llamafile").managed_ref == REF_B
            ),
            message="managed GGUF handoff did not survive first mount",
        )

        assert window.active_view == "llamafile"
        assert (
            window.query_one("#llamafile-gguf-source-mode", Select).value == "managed"
        )
        assert window.query_one("#llamafile-gguf-managed-select", Select).value == REF_B
        assert current_server_claim(app, "llamafile") is None
    finally:
        await _close_context(context)


@pytest.mark.asyncio
async def test_configure_managed_gguf_waits_for_fresh_exact_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A just-downloaded ref is selected only after inventory proves it exists."""
    choice_a = ManagedGGUFChoice(REF_A, "Model A · Q4_K_M · 4 MiB · Managed")
    choice_b = ManagedGGUFChoice(REF_B, "Model B · Q8_0 · 8 MiB · Managed")
    _app, pilot, context, _screen, window, _service = await _mount_models(
        monkeypatch,
        choices=(choice_a,),
    )
    try:
        monkeypatch.setattr(
            window_module,
            "managed_gguf_choices",
            lambda _installed: (choice_a, choice_b),
        )
        generation = window._managed_gguf_inventory_generation

        assert window.configure_managed_gguf("llamacpp", REF_B) is True
        await _settle_pilot_until(
            pilot,
            lambda: (
                window._managed_gguf_inventory_generation > generation
                and window.query_one("#llamacpp-gguf-managed-select", Select).value
                == REF_B
            ),
            message="fresh exact managed GGUF was not selected",
        )

        selection = window.gguf_source_snapshot("llamacpp")
        assert selection.mode is GGUFSourceMode.MANAGED
        assert selection.managed_ref == REF_B
    finally:
        await _close_context(context)


@pytest.mark.asyncio
async def test_configure_managed_gguf_rejects_if_server_starts_before_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refresh guard race cannot strand an accepted runtime handoff."""
    choice_a = ManagedGGUFChoice(REF_A, "Model A · Q4_K_M · 4 MiB · Managed")
    _app, _pilot, context, _screen, window, _service = await _mount_models(
        monkeypatch,
        choices=(choice_a,),
    )
    try:
        states = iter((False, False, True))
        monkeypatch.setattr(window, "_server_active", lambda _provider: next(states))
        generation = window._managed_gguf_inventory_generation

        assert window.configure_managed_gguf("llamacpp", REF_B) is False
        assert window._pending_managed_gguf_handoff is None
        assert window._managed_gguf_inventory_generation == generation
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
async def test_refresh_removing_selected_ref_blocks_stale_managed_launch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    choice_a = ManagedGGUFChoice(REF_A, "Model A · Q4_K_M · 4 MiB · Managed")
    choice_b = ManagedGGUFChoice(REF_B, "Model B · Q8_0 · 8 MiB · Managed")
    app, pilot, context, _screen, window, _service = await _mount_models(
        monkeypatch,
        choices=(choice_a,),
    )
    try:
        mode = window.query_one("#llamacpp-gguf-source-mode", Select)
        managed = window.query_one("#llamacpp-gguf-managed-select", Select)
        start = window.query_one("#llamacpp-start-server-button", Button)
        mode.value = "managed"
        await pilot.pause()
        assert managed.value == REF_A
        assert not start.disabled

        monkeypatch.setattr(
            window_module,
            "managed_gguf_choices",
            lambda _installed: (choice_b,),
        )
        generation = window._managed_gguf_inventory_generation
        refresh = window.query_one("#llamacpp-gguf-refresh-button", Button)
        await window.on_button_pressed(Button.Pressed(refresh))
        for _ in range(20):
            await pilot.pause(0.02)
            if (
                window._managed_gguf_inventory_generation > generation
                and _select_values(managed) == (REF_B,)
            ):
                break

        assert _select_values(managed) == (REF_B,)
        assert managed.value is Select.NULL
        with pytest.raises(
            ValueError,
            match="^managed GGUF selection unavailable$",
        ):
            window.gguf_source_snapshot("llamacpp")
        assert start.disabled
        recovery = str(
            window.query_one("#llamacpp-gguf-source-status", Static).render()
        )
        assert recovery == (
            "Selected managed GGUF is unavailable. "
            "Choose another managed model or refresh."
        )
        assert PRIVATE_MANAGED_PATH not in recovery

        executable = tmp_path / "llama-server"
        executable.write_text("#!/bin/sh\n", encoding="utf-8")
        window.query_one("#llamacpp-exec-path", Input).value = str(executable)
        workers: list[object] = []
        monkeypatch.setattr(
            app,
            "run_worker",
            lambda work, **_kwargs: workers.append(work),
        )
        start.scroll_visible(animate=False)
        await pilot.pause()
        await pilot.click(start)
        assert current_server_claim(app, "llamacpp") is None
        assert workers == []

        managed.value = REF_B
        await pilot.pause()
        assert window.gguf_source_snapshot("llamacpp").managed_ref == REF_B
        assert not start.disabled
    finally:
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
async def test_inventory_failure_after_selection_blocks_stale_managed_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    choice = ManagedGGUFChoice(REF_A, "Model A · Q4_K_M · 4 MiB · Managed")
    service = _InventoryService()
    app, pilot, context, _screen, window, _service = await _mount_models(
        monkeypatch,
        choices=(choice,),
        service=service,
    )
    try:
        modes = {
            provider: window.query_one(f"#{provider}-gguf-source-mode", Select)
            for provider in ("llamacpp", "llamafile")
        }
        managed = {
            provider: window.query_one(f"#{provider}-gguf-managed-select", Select)
            for provider in modes
        }
        starts = {
            provider: window.query_one(f"#{provider}-start-server-button", Button)
            for provider in modes
        }
        for provider in modes:
            modes[provider].value = "managed"
        await pilot.pause()
        assert all(select.value == REF_A for select in managed.values())
        assert all(not button.disabled for button in starts.values())

        service.error = RuntimeError(PRIVATE_MANAGED_PATH)
        generation = window._managed_gguf_inventory_generation
        refresh = window.query_one("#llamacpp-gguf-refresh-button", Button)
        await window.on_button_pressed(Button.Pressed(refresh))
        for _ in range(20):
            await pilot.pause(0.02)
            if (
                window._managed_gguf_inventory_generation > generation
                and window._managed_gguf_inventory_error
            ):
                break

        for provider in modes:
            assert managed[provider].value is Select.NULL
            assert managed[provider].disabled
            assert starts[provider].disabled
            with pytest.raises(
                ValueError,
                match="^managed GGUF inventory unavailable$",
            ):
                window.gguf_source_snapshot(provider)
            recovery = str(
                window.query_one(f"#{provider}-gguf-source-status", Static).render()
            )
            assert recovery == (
                "Managed GGUF inventory unavailable. Refresh managed models to retry."
            )
            assert PRIVATE_MANAGED_PATH not in recovery

        workers: list[object] = []
        real_run_worker = app.run_worker
        monkeypatch.setattr(
            app,
            "run_worker",
            lambda work, **_kwargs: workers.append(work),
        )
        for provider in modes:
            await window.on_button_pressed(Button.Pressed(starts[provider]))
            assert current_server_claim(app, provider) is None
        assert workers == []

        window.query_one("#llamacpp-model-path", Input).value = "/outside/model.gguf"
        modes["llamacpp"].value = "external"
        modes["llamafile"].value = "embedded"
        await pilot.pause()
        assert not starts["llamacpp"].disabled
        assert not starts["llamafile"].disabled
        assert window.gguf_source_snapshot("llamacpp").mode is GGUFSourceMode.EXTERNAL
        assert window.gguf_source_snapshot("llamafile").mode is GGUFSourceMode.EMBEDDED

        monkeypatch.setattr(app, "run_worker", real_run_worker)
        for provider in modes:
            modes[provider].value = "managed"
        service.error = None
        generation = window._managed_gguf_inventory_generation
        await window.on_button_pressed(Button.Pressed(refresh))
        for _ in range(20):
            await pilot.pause(0.02)
            if (
                window._managed_gguf_inventory_generation > generation
                and not window._managed_gguf_inventory_error
                and all(select.value == REF_A for select in managed.values())
            ):
                break

        assert all(select.value == REF_A for select in managed.values())
        assert all(not select.disabled for select in managed.values())
        assert all(not button.disabled for button in starts.values())
        assert all(
            window.gguf_source_snapshot(provider).managed_ref == REF_A
            for provider in modes
        )
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

        deferred_skipped = asyncio.Event()

        async def skip_deferred_finish(_replacement: LLMManagementWindow) -> None:
            deferred_skipped.set()

        monkeypatch.setattr(
            LLMManagementWindow,
            "_finish_deferred_mount",
            skip_deferred_finish,
        )
        screen.refresh(recompose=True)
        await _settle_pilot_until(
            pilot,
            deferred_skipped.is_set,
            message="screen recompose did not reach deferred mount",
        )
        replacement = screen.query_one(LLMManagementWindow)
        assert replacement is not window
        replacement.active_view = "llama-cpp"
        status = replacement.query_one("#llamacpp-gguf-source-status", Static)
        status.scroll_visible(animate=False)
        await pilot.pause()
        assert status in app.screen._compositor.visible_widgets
        assert str(status.render()) == "Pending authority: Managed GGUF"
        first_frame = app.export_screenshot(simplify=True)
        assert "Managed" in first_frame and "GGUF" in first_frame
        for control_id in (
            "llamacpp-start-server-button",
            "llamacpp-gguf-source-mode",
            "llamacpp-gguf-managed-select",
            "llamacpp-gguf-refresh-button",
            "llamacpp-model-path",
            "llamacpp-browse-model-button",
            "llamacpp-exec-path",
            "llamacpp-browse-exec-button",
            "llamacpp-detect-exec-button",
        ):
            assert replacement.query_one(f"#{control_id}").disabled, control_id
        assert not replacement.query_one(
            "#llamacpp-stop-server-button", Button
        ).disabled
    finally:
        await _close_context(context)


@pytest.mark.asyncio
async def test_status_updates_preserve_stop_identity_and_restore_focus_on_death(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, pilot, context, _screen, window, _service = await _mount_models(monkeypatch)
    try:
        window.query_one("#llamacpp-model-path", Input).value = "/outside/model.gguf"
        await pilot.pause()
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
@pytest.mark.parametrize(
    ("provider", "view_name", "focus_id"),
    (
        ("vllm", "vllm", "vllm-model-path"),
        ("mlx", "mlx-lm", "mlx-model-path"),
    ),
)
async def test_non_gguf_lifecycle_sync_preserves_existing_focus(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    view_name: str,
    focus_id: str,
) -> None:
    app, pilot, context, _screen, window, _service = await _mount_models(monkeypatch)
    try:
        window.active_view = view_name
        await pilot.pause()
        focused = window.query_one(f"#{focus_id}", Input)
        focused.value = f"{provider}-value-byte-for-byte"
        input_values = {
            widget.id: widget.value
            for widget in window.query_one(f"#llm-view-{view_name}").query(Input)
        }
        focused.scroll_visible(animate=False)
        focused.focus()
        await pilot.pause()
        assert focused.has_focus

        claim = reserve_server_launch(app, provider, authority="Local process")
        assert claim is not None
        window._sync_process_controls(provider)
        await pilot.pause()
        assert focused.has_focus
        assert {
            widget.id: widget.value
            for widget in window.query_one(f"#llm-view-{view_name}").query(Input)
        } == input_values

        assert release_server_claim(app, provider, claim)
        window._sync_process_controls(provider)
        await pilot.pause()
        assert focused.has_focus
        assert {
            widget.id: widget.value
            for widget in window.query_one(f"#llm-view-{view_name}").query(Input)
        } == input_values
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
@pytest.mark.parametrize("theme", ("textual-dark", "high_contrast_yellow_black"))
async def test_disabled_gguf_controls_keep_live_compositor_contrast(
    monkeypatch: pytest.MonkeyPatch,
    theme: str,
) -> None:
    choices = (
        ManagedGGUFChoice(
            REF_A,
            "Managed readability model · Q4_K_M · 4.0 GiB · Managed",
        ),
    )
    app, pilot, context, _screen, window, _service = await _mount_models(
        monkeypatch,
        size=(80, 24),
        choices=choices,
    )
    try:
        app.theme = theme
        await pilot.pause()
        for provider, view_name in (
            ("llamacpp", "llama-cpp"),
            ("llamafile", "llamafile"),
        ):
            window.active_view = view_name
            await pilot.pause()
            stop = window.query_one(f"#{provider}-stop-server-button", Button)
            stop.scroll_visible(animate=False)
            await pilot.pause()
            idle_stop_ratio, idle_stop_colors = _painted_text_contrast(app, stop)
            assert idle_stop_ratio >= 3.0, (theme, provider, stop.id, idle_stop_ratio)

            for source_mode in ("external", "managed"):
                mode = window.query_one(f"#{provider}-gguf-source-mode", Select)
                mode.value = source_mode
                if source_mode == "external":
                    window.query_one(
                        f"#{provider}-model-path", Input
                    ).value = "/outside/readable-model.gguf"
                    active_source_controls = (
                        window.query_one(f"#{provider}-model-path", Input),
                        window.query_one(f"#{provider}-browse-model-button", Button),
                    )
                else:
                    active_source_controls = (
                        window.query_one(f"#{provider}-gguf-managed-select", Select),
                        window.query_one(f"#{provider}-gguf-refresh-button", Button),
                    )
                controls = (
                    mode,
                    *active_source_controls,
                    window.query_one(f"#{provider}-start-server-button", Button),
                )
                await pilot.pause()
                enabled_colors = {}
                for control in controls:
                    control.scroll_visible(animate=False)
                    await pilot.pause()
                    _ratio, enabled_colors[control.id] = _painted_text_contrast(
                        app, control
                    )

                claim = reserve_server_launch(
                    app,
                    provider,
                    authority=f"{source_mode.title()} GGUF",
                )
                assert claim is not None
                window._sync_process_controls(provider)
                await pilot.pause()
                status = str(
                    window.query_one(f"#{provider}-gguf-source-status", Static).render()
                )
                assert "Pending authority:" in status
                for control in controls:
                    assert control.disabled, control.id
                    control.scroll_visible(animate=False)
                    await pilot.pause()
                    ratio, disabled_colors = _painted_text_contrast(app, control)
                    assert ratio >= 3.0, (theme, provider, control.id, ratio)
                    assert disabled_colors != enabled_colors[control.id], (
                        theme,
                        provider,
                        control.id,
                    )

                stop.scroll_visible(animate=False)
                await pilot.pause()
                running_stop_ratio, running_stop_colors = _painted_text_contrast(
                    app, stop
                )
                assert running_stop_ratio >= 3.0
                assert running_stop_colors != idle_stop_colors
                assert release_server_claim(app, provider, claim)
                window._sync_process_controls(provider)
                await pilot.pause()
    finally:
        await _close_context(context)


@pytest.mark.asyncio
async def test_external_copy_keyboard_geometry_and_unrelated_views_stay_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def load_fixture_profiles(screen: LLMScreen) -> None:
        screen._accept_vllm_profiles(screen._vllm_profiles)

    monkeypatch.setattr(LLMScreen, "_load_vllm_profiles", load_fixture_profiles)
    long_choice = ManagedGGUFChoice(
        REF_A,
        "A very long managed model name that must not push actions out · "
        "Q4_K_M · 16384.0 MiB · Managed · local integrity recorded",
    )
    app, pilot, context, screen, window, _service = await _mount_models(
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
        copy = external_region.query_one(".gguf-source-copy", Static)
        copy.scroll_visible(animate=False)
        await _settle_pilot_until(
            pilot,
            lambda: all(
                token in app.export_screenshot(simplify=True)
                for token in ("Outside", "Chatbook", "This", "used")
            ),
            message="external GGUF copy did not reach the compositor",
        )
        text = _rendered_text(window)
        svg = app.export_screenshot(simplify=True)
        assert EXTERNAL_AUTHORITY in text
        assert EXTERNAL_CONTRACT in text
        assert all(token in svg for token in ("Outside", "Chatbook", "This", "used"))
        assert PRIVATE_MANAGED_PATH not in text + svg
        assert "<svg" in svg and "</svg>" in svg

        window.active_view = "vllm"
        await _settle_pilot_until(
            pilot,
            lambda: (
                len(window.query("#vllm-hf-model")) == 1
                and screen._vllm_profiles_loaded
            ),
            message="vLLM pane did not finish profile hydration",
        )
        vllm = window.query_one("#vllm-hf-model", Input)
        window.active_view = "mlx-lm"
        await _settle_pilot_until(
            pilot,
            lambda: len(window.query("#mlx-model-path")) == 1,
            message="MLX pane did not populate on first selection",
        )
        mlx = window.query_one("#mlx-model-path", Input)
        vllm.value = "org/vllm"
        await pilot.pause()
        mlx.value = "org/mlx"
        await pilot.pause()
        window.active_view = "llama-cpp"
        await pilot.pause()
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
    tmp_path: Path,
    provider: str,
    view_name: str,
) -> None:
    choices = (
        ManagedGGUFChoice(
            REF_A,
            "LongInventoryMarker-model-name-that-must-not-push-actions-outside "
            "· Q4_K_M · 16384.0 MiB · Managed · local integrity recorded",
        ),
    )
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
        managed = window.query_one(f"#{provider}-gguf-managed-select", Select)
        refresh = window.query_one(f"#{provider}-gguf-refresh-button", Button)
        start = window.query_one(f"#{provider}-start-server-button", Button)
        stop = window.query_one(f"#{provider}-stop-server-button", Button)
        executable = tmp_path / f"{provider}-server"
        executable.touch()
        window.query_one(f"#{provider}-exec-path", Input).value = str(executable)
        await _settle_pilot_until(
            pilot,
            lambda: _select_values(managed) == (REF_A,) and not managed.disabled,
            message=f"{provider} managed inventory did not settle",
        )

        mode.scroll_visible(animate=False)
        mode.focus()
        await _settle_pilot_until(
            pilot,
            lambda: (
                mode in app.screen._compositor.visible_widgets
                and mode.region.y >= view.content_region.y
                and mode.region.bottom <= view.content_region.bottom
            ),
            message=f"{provider} source mode did not settle inside its view",
        )
        _assert_painted_inside(app, mode, view)
        await pilot.press("enter")
        await pilot.pause()
        overlay = mode.query_one(SelectOverlay)
        assert mode.expanded and overlay.has_focus
        _assert_overlay_painted_on_screen(app, overlay)
        overlay_svg = app.export_screenshot(simplify=True)
        assert "Managed" in overlay_svg and "GGUF" in overlay_svg
        assert "External" in overlay_svg
        await pilot.press("home")
        if provider == "llamafile":
            await pilot.press("down")
        target_index = 0 if provider == "llamacpp" else 1
        await _settle_pilot_until(
            pilot,
            lambda: overlay.highlighted == target_index,
            message=f"{provider} managed option did not receive keyboard highlight",
        )
        await pilot.press("enter")
        await _settle_pilot_until(
            pilot,
            lambda: (
                mode.value == "managed"
                and managed in app.screen._compositor.visible_widgets
                and not managed.disabled
            ),
            message=f"{provider} managed source controls did not settle",
        )

        await pilot.press("tab")
        await _settle_pilot_until(
            pilot,
            lambda: (
                managed.has_focus
                and managed in app.screen._compositor.visible_widgets
                and managed.region.y >= view.content_region.y
                and managed.region.bottom <= view.content_region.bottom
            ),
            message=f"{provider} managed selector did not settle inside its view",
        )
        assert managed.has_focus
        assert not managed.expanded
        _assert_painted_inside(app, managed, view)
        await pilot.press("enter")
        await pilot.pause()
        managed_overlay = managed.query_one(SelectOverlay)
        assert managed.expanded and managed_overlay.has_focus
        _assert_overlay_painted_on_screen(app, managed_overlay)
        assert "LongInvent" in app.export_screenshot(simplify=True)
        await pilot.press("enter")
        await pilot.pause()
        assert managed.has_focus and not managed.expanded
        await pilot.press("tab")
        await pilot.pause()
        assert refresh.has_focus
        _assert_painted_inside(app, refresh, view)
        managed_svg = app.export_screenshot(simplify=True)
        assert "LongInvent" in managed_svg
        assert "oryMarker-" in managed_svg
        assert "Refresh" in managed_svg

        await pilot.press("shift+tab")
        await pilot.pause()
        assert managed.has_focus
        await pilot.press("tab")
        await pilot.pause()
        assert refresh.has_focus
        generation = window._managed_gguf_inventory_generation
        await pilot.press("enter")
        for _ in range(10):
            await pilot.pause()
            if window._managed_gguf_inventory_generation > generation:
                break
        assert window._managed_gguf_inventory_generation == generation + 1

        await pilot.press("shift+tab", "shift+tab")
        await pilot.pause()
        assert mode.has_focus
        await pilot.press("enter", "down", "enter")
        await pilot.pause()
        assert mode.value == "external"
        await pilot.press("tab")
        await pilot.pause()
        external_path = window.query_one(f"#{provider}-model-path", Input)
        assert external_path.has_focus
        await pilot.press("tab")
        await pilot.pause()
        browse = window.query_one(f"#{provider}-browse-model-button", Button)
        assert browse.has_focus
        _assert_painted_inside(app, browse, view)
        pushed_screens: list[tuple[Any, Any]] = []

        async def capture_push_screen(screen: Any, callback: Any = None) -> None:
            pushed_screens.append((screen, callback))

        monkeypatch.setattr(app, "push_screen", capture_push_screen)
        await pilot.press("enter")
        await pilot.pause()
        assert len(pushed_screens) == 1
        assert pushed_screens[0][1] is not None
        copy = window.query_one(f"#{provider}-gguf-external-region").query_one(
            ".gguf-source-copy", Static
        )
        copy.scroll_visible(animate=False)
        await _settle_pilot_until(
            pilot,
            lambda: (
                copy in app.screen._compositor.visible_widgets
                and copy.region.y >= view.content_region.y
                and copy.region.bottom <= view.content_region.bottom
            ),
            message=f"{provider} external copy did not settle inside its view",
        )
        _assert_painted_inside(app, copy, view)
        external_svg = app.export_screenshot(simplify=True)
        assert all(
            token in external_svg
            for token in ("Browse", "Outside", "Chatbook", "This", "used")
        )

        await pilot.press("shift+tab", "shift+tab")
        await pilot.pause()
        assert mode.has_focus
        await pilot.press("enter", "up", "enter")
        await pilot.pause()
        assert mode.value == "managed"
        await pilot.press("tab", "tab")
        await pilot.pause()
        assert refresh.has_focus
        traversed = await _press_until_focus(pilot, start)
        assert traversed[-1] == start.id
        await _settle_pilot_until(
            pilot,
            lambda: (
                start in app.screen._compositor.visible_widgets
                and start.region.y >= view.content_region.y
                and start.region.bottom <= view.content_region.bottom
            ),
            message=f"{provider} Start did not settle inside its view",
        )
        _assert_painted_inside(app, start, view)

        gated_workers: list[tuple[Any, dict[str, Any]]] = []

        def hold_worker(worker: Any, **kwargs: Any) -> SimpleNamespace:
            gated_workers.append((worker, kwargs))
            return SimpleNamespace()

        monkeypatch.setattr(app, "run_worker", hold_worker)
        stale = ServerLaunchClaim(provider, authority="stale")
        await pilot.press("enter")
        await pilot.pause()

        claim = current_server_claim(app, provider)
        assert claim is not None and claim is not stale
        assert len(gated_workers) == 1
        assert claim in gated_workers[0][0].args
        assert start.disabled and not stop.disabled
        assert stop.has_focus
        _assert_painted_inside(app, stop, view)
        stop_identity = stop
        stop_svg = app.export_screenshot(simplify=True)
        assert "Stop" in stop_svg
        status = window.query_one(f"#{provider}-gguf-source-status", Static)
        status.scroll_visible(animate=False)
        await _settle_pilot_until(
            pilot,
            lambda: (
                status in app.screen._compositor.visible_widgets
                and status.region.y >= view.content_region.y
                and status.region.bottom <= view.content_region.bottom
            ),
            message=f"{provider} source status did not settle inside its view",
        )
        _assert_painted_inside(app, status, view)
        assert "Pending" in app.export_screenshot(simplify=True)

        window._handle_server_process_state_change(provider)
        window._handle_server_process_state_change(provider)
        await pilot.pause()
        assert (
            window.query_one(f"#{provider}-stop-server-button", Button) is stop_identity
        )
        assert stop_identity.has_focus

        await pilot.press("enter")
        await pilot.pause()
        assert claim.cancel_event.is_set()
        assert not stale.cancel_event.is_set()
        assert current_server_claim(app, provider) is claim

        assert release_server_claim(app, provider, claim)
        window._handle_server_process_state_change(provider)
        await pilot.pause()
        assert start.has_focus and not start.disabled
    finally:
        await _close_context(context)

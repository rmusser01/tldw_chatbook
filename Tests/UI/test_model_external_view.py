"""Lab Models coverage for configured user-owned Parakeet sources."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual import on

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

from tldw_chatbook.Local_Ingestion.stt_batch_routing import PARAKEET_V2_MODEL
from tldw_chatbook.STT.parakeet_sources import (
    ParakeetSourceKey,
    ParakeetSourcePreference,
    ParakeetSourceRecord,
)
from tldw_chatbook.UI.Screens.model_external_view import ExternalModelView


_BUNDLED_STYLESHEET = (
    Path(__file__).resolve().parents[2] / "tldw_chatbook/css/tldw_cli_modular.tcss"
)


class _SourceService:
    def __init__(self, records):
        self._records = dict(records)

    def records(self):
        return dict(self._records)


class _ViewApp(ConsolidatedCSSApp):
    CSS_PATH = str(_BUNDLED_STYLESHEET)

    def __init__(self, service: _SourceService, *, runtime_ready: bool = True) -> None:
        self.requests: list[object] = []
        self.view = ExternalModelView(
            service,
            runtime_ready=lambda: runtime_ready,
            id="external-models-view",
        )
        super().__init__()

    def compose(self) -> ComposeResult:
        yield self.view

    @on(ExternalModelView.ChangeRequested)
    @on(ExternalModelView.StopRequested)
    @on(ExternalModelView.CopyRequested)
    @on(ExternalModelView.CancelRequested)
    def _capture(self, event) -> None:
        self.requests.append(event)


def _text(app: App) -> str:
    return "\n".join(str(item.renderable) for item in app.query(Static))


@pytest.mark.asyncio
async def test_external_view_lists_only_records_with_user_owned_directories(
    tmp_path: Path,
) -> None:
    v2_root = (tmp_path / "v2-int8").absolute()
    remembered_root = (tmp_path / "v2-f32").absolute()
    service = _SourceService(
        {
            ParakeetSourceKey.V2_INT8: ParakeetSourceRecord(
                model_id=PARAKEET_V2_MODEL,
                precision="int8",
                directory=v2_root,
                preferred_source=ParakeetSourcePreference.EXTERNAL,
            ),
            ParakeetSourceKey.V2_F32: ParakeetSourceRecord(
                model_id=PARAKEET_V2_MODEL,
                precision="f32",
                directory=remembered_root,
                preferred_source=ParakeetSourcePreference.MANAGED,
            ),
            ParakeetSourceKey.V3_INT8: ParakeetSourceRecord(
                model_id=ParakeetSourceKey.V3_INT8.model_id,
                precision="int8",
                preferred_source=ParakeetSourcePreference.MANAGED,
            ),
        }
    )
    app = _ViewApp(service)

    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        text = _text(app)

        assert "External source · descriptor verified" in text
        assert str(v2_root) in text
        assert str(remembered_root) in text
        assert "Parakeet v3" not in text
        assert len(app.query(".external-model-row")) == 2
        assert len(app.query(".external-model-path")) == 2


@pytest.mark.asyncio
async def test_fresh_production_css_view_persists_runtime_required(
    tmp_path: Path,
) -> None:
    """A reopened External view derives runtime readiness from the runtime."""

    service = _SourceService(
        {
            ParakeetSourceKey.V2_INT8: ParakeetSourceRecord(
                model_id=PARAKEET_V2_MODEL,
                precision="int8",
                directory=(tmp_path / "model").absolute(),
                preferred_source=ParakeetSourcePreference.EXTERNAL,
            )
        }
    )

    for _ in range(2):
        app = _ViewApp(service, runtime_ready=False)
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            status = app.query_one(".external-model-status", Static)
            assert str(status.renderable) == "Runtime required"
            assert status.region.width > 0 and status.region.height > 0


@pytest.mark.asyncio
async def test_external_view_emits_exact_change_stop_and_copy_actions(
    tmp_path: Path,
) -> None:
    service = _SourceService(
        {
            ParakeetSourceKey.V2_INT8: ParakeetSourceRecord(
                model_id=PARAKEET_V2_MODEL,
                precision="int8",
                directory=(tmp_path / "model").absolute(),
                preferred_source=ParakeetSourcePreference.EXTERNAL,
            )
        }
    )
    app = _ViewApp(service)

    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        for action in ("change", "stop", "copy"):
            app.query_one(f"#external-model-{action}-v2_int8", Button).press()
            await pilot.pause()

    assert [type(event) for event in app.requests] == [
        ExternalModelView.ChangeRequested,
        ExternalModelView.StopRequested,
        ExternalModelView.CopyRequested,
    ]
    assert all(event.key is ParakeetSourceKey.V2_INT8 for event in app.requests)


@pytest.mark.asyncio
async def test_external_view_keeps_actions_and_status_reachable_at_80_columns(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.LLM_Management_Window import LLMManagementWindow
    from tldw_chatbook.UI.Screens.llm_screen import LLMScreen
    from Tests.UI.app_factory import _build_test_app

    selected = (tmp_path / ("long-model-directory-" * 4)).absolute()
    service = _SourceService(
        {
            ParakeetSourceKey.V2_INT8: ParakeetSourceRecord(
                model_id=PARAKEET_V2_MODEL,
                precision="int8",
                directory=selected,
                preferred_source=ParakeetSourcePreference.EXTERNAL,
            )
        }
    )
    service.close = lambda: None
    service.may_delete = lambda _reference: None
    service.on_root_activated = lambda _reference: None
    app = _build_test_app()
    assert app.CSS_PATH == TldwCli.CSS_PATH
    app._parakeet_source_service = service
    monkeypatch.setattr(
        "tldw_chatbook.app.get_cli_setting",
        lambda section, key=None, default=None: (
            False if (section, key) == ("splash_screen", "enabled") else default
        ),
    )
    monkeypatch.setattr(
        LLMManagementWindow,
        "_ollama_api_available",
        lambda _window: False,
    )

    async with app.run_test(size=(80, 24)) as pilot:
        screen = LLMScreen(app)
        await app.push_screen(screen)
        for _ in range(120):
            if screen.query("#external-models-view"):
                break
            await pilot.pause()
        external_row = next(
            row
            for row in screen.query(".lab-rail-row").results(Button)
            if row.lab_view_key == "external"
        )
        external_row.press()
        await pilot.pause()

        parent = screen.query_one("#llm-view-external")
        actions = [
            screen.query_one(f"#external-model-{action}-v2_int8", Button)
            for action in ("change", "stop", "copy")
        ]
        assert parent.region.width > 0 and parent.region.height > 0
        for button in actions:
            assert button.region.width > 0 and button.region.height > 0
            assert parent.region.x <= button.region.x
            assert button.region.x + button.region.width <= (
                parent.region.x + parent.region.width
            )
            assert parent.region.y <= button.region.y
            assert button.region.y + button.region.height <= (
                parent.region.y + parent.region.height
            )
        copy = actions[-1]
        painted_copy = "".join(
            copy.render_line(y).text for y in range(copy.region.height)
        )
        assert "Copy into managed store…" in painted_copy
        actions[0].focus()
        await pilot.press("tab")
        assert app.focused is actions[1]
        await pilot.press("tab")
        assert app.focused is copy
        assert str(selected) in str(
            screen.query_one(".external-model-path", Static).renderable
        )


@pytest.mark.asyncio
async def test_external_operation_error_is_path_safe_and_does_not_replace_edit_path(
    tmp_path: Path,
) -> None:
    selected = (tmp_path / "private-model-root").absolute()
    service = _SourceService(
        {
            ParakeetSourceKey.V2_INT8: ParakeetSourceRecord(
                model_id=PARAKEET_V2_MODEL,
                precision="int8",
                directory=selected,
                preferred_source=ParakeetSourcePreference.EXTERNAL,
            )
        }
    )
    app = _ViewApp(service)

    async with app.run_test(size=(80, 24)) as pilot:
        app.view.apply_operation_status(
            "The selected model could not be verified. Choose the directory again.",
            error=True,
        )
        await pilot.pause()

        status = str(
            app.query_one("#external-model-operation-status", Static).renderable
        )
        assert str(selected) not in status
        assert str(selected) in str(
            app.query_one(".external-model-path", Static).renderable
        )


@pytest.mark.parametrize("configured", (False, True))
@pytest.mark.asyncio
async def test_external_progress_preserves_physical_cancel_focus_at_80_columns(
    tmp_path: Path,
    configured: bool,
) -> None:
    """Progress-only updates retain the mounted cancellation control and focus."""

    key = ParakeetSourceKey.V2_INT8
    records = (
        {
            key: ParakeetSourceRecord(
                model_id=PARAKEET_V2_MODEL,
                precision="int8",
                directory=(tmp_path / "model").absolute(),
                preferred_source=ParakeetSourcePreference.EXTERNAL,
            )
        }
        if configured
        else {}
    )
    app = _ViewApp(_SourceService(records))
    selector = (
        f"#external-model-stop-{key.value}"
        if configured
        else "#external-model-cancel-operation"
    )

    async with app.run_test(size=(80, 24)) as pilot:
        app.view.apply_operation_status("Verifying model files…", active=True)
        await pilot.pause()
        cancel = app.query_one(selector, Button)
        cancel.focus()
        await pilot.pause()
        assert app.focused is cancel

        for status in (
            "Verifying model files · 1 / 4 bytes",
            "Verifying model files · 2 / 4 bytes",
            "Verifying model files · 4 / 4 bytes",
        ):
            app.view.apply_operation_status(status, active=True)
            await pilot.pause()
            assert app.query_one(selector, Button) is cancel
            assert app.focused is cancel
            assert cancel.region.width > 0 and cancel.region.height > 0

        await pilot.click(selector)
        await pilot.pause()

    expected = (
        ExternalModelView.StopRequested
        if configured
        else ExternalModelView.CancelRequested
    )
    assert [type(request) for request in app.requests] == [expected]

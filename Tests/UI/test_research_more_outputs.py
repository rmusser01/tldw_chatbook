import pytest
from textual.widgets import Button, Input, Select, Static

from Tests.UI.library_quit_guard_support import _settings_without_splash, _until
from Tests.UI.test_research_chat_region import build_research_app, open_chat
from Tests.UI.test_research_studio_pane import prepare_studio
from Tests.app_module_patches import patch_app_global
from tldw_chatbook.Research_Workspace.action_registry import PLANNED_OUTPUTS
from tldw_chatbook.Research_Workspace.studio_models import ResearchWorkProductKind
from tldw_chatbook.Research_Workspace.work_product_templates import (
    TEXT_WORK_PRODUCT_TEMPLATES,
)
from tldw_chatbook.UI.Research_Workspace_Modules.studio_history import history_key

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


async def test_more_outputs_has_one_command_per_action_and_no_focusable_planned_items(
    tmp_path, monkeypatch
):
    app, _, _ = build_research_app(tmp_path, monkeypatch)
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=(160, 40)) as pilot:
            screen, _ = await open_chat(app, pilot)
            await prepare_studio(screen, pilot, monkeypatch, "summary")
            action = screen.query_one("#research-studio-more-outputs", Button)
            action.focus()
            await pilot.press("enter")
            await _until(pilot, lambda: app.screen is not screen, "More outputs")
            modal = app.screen
            labels = {
                str(row.render()) for row in modal.query(".research-more-heading")
            }
            assert labels == {"Learn", "Analyze", "Present", "Work products", "Planned"}
            assert not modal.query(".research-studio-create")
            assert len(modal.query(".research-more-orientation")) == 5
            assert len(modal.query(".research-more-action")) == 10
            planned = list(modal.query(".research-more-planned"))
            assert [str(row.render()) for row in planned] == [
                row.label for row in PLANNED_OUTPUTS
            ]
            assert all(isinstance(row, Static) and not row.can_focus for row in planned)
            mind_map = modal.query_one("#research-more-mind-map", Button)
            assert mind_map.disabled
            detail = modal.query_one("#research-more-detail-mind-map", Static)
            assert "Local Mind Map" in str(detail.render())
            assert "Choose a Server workspace" in str(detail.render())
            for width, height in (
                (160, 40),
                (120, 30),
                (100, 30),
                (84, 24),
                (80, 24),
                (60, 20),
            ):
                await pilot.resize_terminal(width, height)
                brief = modal.query_one("#research-more-executive-brief", Button)
                assert not brief.disabled
                brief.focus()
                await pilot.pause()
                assert (
                    app.focused is brief
                    and 0 <= brief.region.y < brief.region.bottom <= height
                )
                close = modal.query_one("#research-more-close", Button)
                close.focus()
                await pilot.pause()
                assert app.focused is close and close.region.bottom <= height
            await pilot.press("escape")
            await _until(pilot, lambda: app.screen is screen, "returned to Studio")


async def test_editing_processing_route_invalidates_previous_availability(
    tmp_path, monkeypatch
):
    app, _, _ = build_research_app(tmp_path, monkeypatch)
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=(160, 40)) as pilot:
            screen, _ = await open_chat(app, pilot)
            _, _, summary = await prepare_studio(screen, pilot, monkeypatch, "summary")
            model = screen.query_one("#research-studio-model", Input)
            model.value = "changed-model"
            await _until(pilot, lambda: summary.disabled, "old route disabled")


async def test_more_outputs_dispatches_work_product_to_canonical_artifact(
    tmp_path, monkeypatch
):
    app, _, provider = build_research_app(tmp_path, monkeypatch)
    kind = ResearchWorkProductKind.EXECUTIVE_BRIEF
    provider.answer = "\n\n".join(
        f"## {label}\nObservation [S1]."
        for label in TEXT_WORK_PRODUCT_TEMPLATES[kind].sections
    )
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=(60, 20)) as pilot:
            screen, _ = await open_chat(app, pilot)
            screen.query_one("#research-pane-mode-studio", Button).press()
            port, _, _ = await prepare_studio(screen, pilot, monkeypatch, "summary")
            screen.query_one("#research-studio-more-outputs", Button).press()
            await _until(pilot, lambda: app.screen is not screen, "More outputs dialog")
            more = app.screen
            button = more.query_one("#research-more-executive-brief", Button)
            button.focus()
            await pilot.press("enter")
            await _until(
                pilot,
                lambda: bool(list(app.screen.query("#research-studio-generate"))),
                "brief options",
            )
            app.screen.query_one(
                "#research-studio-title", Input
            ).value = "Observatory decision brief"
            app.screen.query_one("#research-studio-generate", Button).press()
            status = screen.query_one("#research-studio-status", Static)
            await _until(
                pilot,
                lambda: "Saved to Local Artifacts" in str(status.render()),
                "canonical owner receipt",
            )
            (output,) = (await port.list_outputs(screen.controller.selected_ref)).items
            assert output.output_kind is kind
            artifact = await port.resolve_output(output)
            assert artifact["name"] == "Observatory decision brief"
            history = screen.query_one("#research-studio-history", Select)
            await _until(
                pilot,
                lambda: history.value == history_key(output),
                "selected output history",
            )

"""Exercise the real Studio and Runs navigation with a native Local owner."""

import pytest
from textual.widgets import Button, Input, Static

from Tests.app_module_patches import patch_app_global
from Tests.UI.app_factory import swap_app_aliases
from Tests.UI.library_quit_guard_support import _settings_without_splash, _until
from Tests.UI.test_research_chat_region import build_research_app, open_chat
from Tests.UI.test_research_studio_pane import prepare_studio
from tldw_chatbook.Research_Interop.local_research_service import LocalResearchService
from tldw_chatbook.Research_Interop.research_scope_service import ResearchScopeService
from tldw_chatbook.UI.Research_Window import ResearchWindow
from tldw_chatbook.UI.Screens.research_screen import ResearchScreen

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


async def test_deep_research_launch_returns_to_exact_workspace_before_import(
    tmp_path, monkeypatch
):
    app, _, _ = build_research_app(tmp_path, monkeypatch)
    owner = LocalResearchService(tmp_path / "research.sqlite")
    scope = ResearchScopeService(local_service=owner, server_service=None)
    swap_app_aliases(app, local_research_service=owner, research_scope_service=scope)
    starts = []
    monkeypatch.setattr(
        ResearchWindow,
        "_start_local_engine",
        lambda _window, run_id: starts.append(run_id),
    )
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=(84, 24)) as pilot:
            screen, _ = await open_chat(app, pilot)
            screen.query_one("#research-pane-mode-studio", Button).press()
            port, _, _ = await prepare_studio(screen, pilot, monkeypatch, "report")
            origin = screen.controller.selected_ref
            launch = screen.query_one("#research-studio-deep-research", Button)
            await _until(pilot, lambda: not launch.disabled, "Deep Research available")
            launch.press()
            await _until(
                pilot,
                lambda: bool(app.screen.query("#research-launch-query")),
                "launch question",
            )
            app.screen.query_one(
                "#research-launch-query", Input
            ).value = "What did the observatory record?"
            app.screen.query_one("#research-launch-confirm", Button).press()
            await _until(
                pilot,
                lambda: isinstance(app.screen, ResearchScreen) and starts,
                "native Runs detail selected",
            )
            run_id = starts[0]
            run = owner.get_run(run_id)
            assert (
                run["chat_handoff"]["research_workspace"]["origin"]["workspace_id"]
                == origin.workspace_id
            )
            assert not (await port.list_outputs(origin)).items
            owner.save_artifact(
                run_id,
                artifact_name="bundle.json",
                content_type="application/json",
                content={
                    "query": run["query"],
                    "report_markdown": "# Reviewed report\nNative findings.",
                    "source_count": 2,
                },
            )
            owner.complete_run(run_id)
            runs = app.screen
            runs.query_one("#research-refresh-runs", Button).press()
            returning = runs.query_one("#research-return-to-workspace", Button)
            await _until(pilot, lambda: returning.display, "return action")
            returning.press()
            await _until(
                pilot,
                lambda: bool(app.screen.query("#research-report-preview")),
                "Report preview",
            )
            assert "Native findings" in str(
                app.screen.query_one("#research-report-preview", Static).render()
            )
            assert not (await port.list_outputs(origin)).items
            app.screen.query_one("#research-report-save", Button).press()
            await _until(
                pilot,
                lambda: (
                    bool(app.screen.query("#research-studio-status"))
                    and "Saved a draft Report"
                    in str(
                        app.screen.query_one("#research-studio-status", Static).render()
                    )
                ),
                "draft saved in origin",
            )
            outputs = (await port.list_outputs(origin)).items
            assert len(outputs) == 1
            record = await port.resolve_output(outputs[0])
            assert record["metadata"]["review_status"] == "draft"
            assert record["metadata"]["research_run_ref"]["run_id"] == run_id

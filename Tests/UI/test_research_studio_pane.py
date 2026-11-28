"""Exercise Studio at the mounted Research destination."""

import asyncio
from dataclasses import replace

import pytest
from textual.widgets import Button, Input, Select, Static

from Tests.app_module_patches import patch_app_global
from Tests.Research_Workspace.test_studio_models import SOURCE
from Tests.UI.library_quit_guard_support import _settings_without_splash, _until
from Tests.UI.test_research_chat_region import build_research_app, open_chat
from tldw_chatbook.Research_Workspace.contracts import WorkspaceDataSource
from tldw_chatbook.Research_Workspace.studio_generation import StudioEvidence
from tldw_chatbook.Research_Workspace.studio_models import PRIMARY_RESEARCH_OUTPUT_KINDS
from tldw_chatbook.UI.Research_Workspace_Modules.studio_history import history_key


pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


async def prepare_studio(screen, pilot, monkeypatch, card):
    port = screen.controller.port_for_data_source(WorkspaceDataSource.LOCAL)
    sources = [replace(SOURCE, ref=screen.controller.selected_ref)]

    async def capture(ref):
        if ref != sources[0].ref:
            return ()
        return tuple(sources)

    async def evidence(request):
        return (
            StudioEvidence(request.retrieval_sources[0], "Observatory", "Readings"),
        )

    monkeypatch.setattr(port, "capture_chat_sources", capture)
    monkeypatch.setattr(port._studio.generator, "current_sources", capture)
    monkeypatch.setattr(port._studio.generator, "read_evidence", evidence)
    screen.query_one("#research-studio-provider", Input).value = "fixture"
    screen.query_one("#research-studio-model", Input).value = "model"
    screen.query_one("#research-studio-check-route", Button).press()
    create = screen.query_one(f"#research-studio-create-{card}", Button)
    await _until(pilot, lambda: not create.disabled, f"{card} available")
    return port, sources, create


async def start_output(app, screen, pilot, button, title):
    button.focus()
    await pilot.press("enter")
    await _until(pilot, lambda: app.screen is not screen, "Studio options")
    app.screen.query_one("#research-studio-title", Input).value = title
    app.screen.query_one("#research-studio-generate", Button).press()


async def test_studio_shows_all_primary_cards_and_canonical_owner_recovery(
    tmp_path, monkeypatch
):
    app, _, _ = build_research_app(tmp_path, monkeypatch)
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=(160, 40)) as pilot:
            screen, _ = await open_chat(app, pilot)
            pane = screen.query_one("#research-studio-pane")
            assert len(pane.query(".research-studio-card")) == 5
            await _until(
                pilot,
                lambda: (
                    "Local Artifacts"
                    in str(
                        pane.query_one(
                            "#research-studio-availability-report", Static
                        ).render()
                    )
                ),
                "canonical owner availability",
            )
            for kind in PRIMARY_RESEARCH_OUTPUT_KINDS:
                key = kind.value.replace("_", "-")
                assert pane.query_one(f"#research-studio-create-{key}", Button).disabled
            assert "Local Artifacts" in str(
                pane.query_one("#research-studio-availability-report", Static).render()
            )
            assert "Local Study decks" in str(
                pane.query_one(
                    "#research-studio-availability-flashcards", Static
                ).render()
            )
            assert (
                pane.query_one("#research-studio-history", Select).value is Select.NULL
            )


async def test_studio_shows_compare_gate_and_saves_to_canonical_history(
    tmp_path, monkeypatch
):
    app, _, provider = build_research_app(tmp_path, monkeypatch)
    provider.answer = "The observatory's result [S1]."
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=(160, 40)) as pilot:
            screen, _ = await open_chat(app, pilot)
            port, _, report = await prepare_studio(screen, pilot, monkeypatch, "report")
            compare = screen.query_one(
                "#research-studio-availability-compare-sources", Static
            )
            assert "two" in str(compare.render()).lower() or "2" in str(
                compare.render()
            )
            assert screen.query_one(
                "#research-studio-create-compare-sources", Button
            ).disabled
            await start_output(app, screen, pilot, report, "Observatory report")
            status = screen.query_one("#research-studio-status", Static)
            await _until(
                pilot,
                lambda: "Saved to Local Artifacts" in str(status.render()),
                "Saved receipt",
            )
            page = await port.list_outputs(screen.controller.selected_ref)
            assert len(page.items) == 1
            artifact = await port.resolve_output(page.items[0])
            assert artifact["name"] == "Observatory report"
            history = screen.query_one("#research-studio-history", Select)
            await _until(
                pilot,
                lambda: history.value == history_key(page.items[0]),
                "canonical output selected",
            )
            for width, height in (
                (160, 40),
                (120, 30),
                (100, 30),
                (84, 24),
                (80, 24),
                (60, 20),
            ):
                await pilot.resize_terminal(width, height)
                if not screen.query_one("#research-studio-pane").display:
                    screen.query_one("#research-pane-mode-studio", Button).press()
                    await pilot.pause()
                history.focus()
                await pilot.pause()
                assert app.focused is history
                assert history.value == history_key(page.items[0])
                assert history.region.y >= 0 and history.region.bottom <= height


async def test_studio_blocks_workspace_switch_during_generation_then_retries(
    tmp_path, monkeypatch
):
    app, _, provider = build_research_app(tmp_path, monkeypatch)
    app.workspace_registry_service.create_workspace(
        workspace_id="z-project", name="Z project"
    )
    provider.hold = True
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=(60, 20)) as pilot:
            screen, _ = await open_chat(app, pilot)
            screen.query_one("#research-pane-mode-studio", Button).press()
            port, _, create = await prepare_studio(
                screen, pilot, monkeypatch, "summary"
            )
            await start_output(app, screen, pilot, create, "Narrow summary")
            await asyncio.wait_for(provider.started.wait(), 8)
            running_ref = screen.controller.selected_ref
            picker = screen.query_one("#research-workspace-picker", Select)
            other = replace(running_ref, workspace_id="z-project")
            picker.value = other
            status = screen.query_one("#research-studio-status", Static)
            await _until(
                pilot,
                lambda: "Stop the active" in str(status.render()),
                "blocked workspace switch",
            )
            assert screen.controller.selected_ref == running_ref
            stop = screen.query_one("#research-studio-stop", Button)
            stop.focus()
            await pilot.press("enter")
            retry = screen.query_one("#research-studio-retry", Button)
            await _until(pilot, lambda: not retry.disabled, "retry available")
            assert not (await port.list_outputs(running_ref)).items
            provider.hold = False
            retry.focus()
            await pilot.press("enter")
            await _until(
                pilot,
                lambda: "Saved to Local Artifacts" in str(status.render()),
                "retried output saved",
            )
            assert len((await port.list_outputs(running_ref)).items) == 1

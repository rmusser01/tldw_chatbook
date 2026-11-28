from uuid import UUID

import pytest
from textual.app import App
from textual.widgets import Button, Input, Select, Static

from Tests.UI.library_quit_guard_support import _settings_without_splash, _until
from Tests.UI.test_research_chat_region import build_research_app, open_chat
from Tests.UI.test_research_studio_pane import prepare_studio
from Tests.app_module_patches import patch_app_global
from tldw_chatbook.Research_Workspace.action_registry import PLANNED_OUTPUTS
from tldw_chatbook.Research_Workspace.chat_models import ResearchSourceVersionRef
from tldw_chatbook.Research_Workspace.contracts import (
    ProcessingRoute,
    QualifiedWorkspaceRef,
    ResearchCapability,
)
from tldw_chatbook.Research_Workspace.controller import ResearchWorkspaceController
from tldw_chatbook.Research_Workspace.server_studio import ServerStudioAdapter
from tldw_chatbook.Research_Workspace.studio_models import ResearchWorkProductKind
from tldw_chatbook.Research_Workspace.studio_service import ResearchStudioService
from tldw_chatbook.Research_Workspace.work_product_templates import (
    TEXT_WORK_PRODUCT_TEMPLATES,
)
from tldw_chatbook.UI.Research_Workspace_Modules.studio_history import history_key
from tldw_chatbook.UI.Research_Workspace_Modules.studio_region import (
    ResearchStudioRegion,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


class _NativeServerPort:
    ref = QualifiedWorkspaceRef("server", "observatory", "server-a", "jwt-sub:1")
    route = ProcessingRoute("server", "server", "provider", "model")

    def __init__(self, source_count):
        self.sources = tuple(
            ResearchSourceVersionRef(
                self.ref, f"source-{index}", str(index), "media", 1, 1, ("fts",)
            )
            for index in range(1, source_count + 1)
        )

    async def studio_capabilities(self, workspace):
        assert workspace == self.ref
        ready = ResearchCapability(True, "available", "Ready", "server")
        return {"save_workspace_artifact": ready, "generate_grounded_artifact": ready}

    async def capture_chat_sources(self, workspace):
        assert workspace == self.ref
        return self.sources

    async def resolve_chat_route(self, workspace, *_):
        assert workspace == self.ref
        return self.route

    async def native_studio_capabilities(self, workspace, processing, selected):
        assert (workspace, processing, selected) == (self.ref, self.route, self.sources)
        verified = ResearchCapability(
            True,
            "available",
            "Ready",
            "server",
            capability_revision="workspace-api:reachable:authenticated",
        )
        result = dict.fromkeys(
            (
                "generate_data_table",
                "get_data_table",
                "generate_literature_matrix",
                "generate_corpus_gap_finder",
                "create_audio_speech_job",
                "list_audio_speech_job_artifacts",
            ),
            verified,
        )
        if len(self.sources) == 1:
            result.update(
                generate_presentation_from_media=verified, get_presentation=verified
            )
        return result


async def _mount_and_open_more_outputs(app, pilot, port):
    ref = port.ref
    controller = ResearchWorkspaceController({ref.data_source: port})
    controller.select_workspace(ref)
    studio = ResearchStudioRegion(controller=controller)
    await app.mount(studio)
    studio.set_workspace(ref)
    studio.query_one("#research-studio-check-route", Button).press()
    route_status = studio.query_one("#research-studio-route", Static)
    await _until(
        pilot, lambda: "provider / model" in str(route_status.render()), "Server route"
    )
    studio.query_one("#research-studio-more-outputs", Button).press()
    await _until(
        pilot, lambda: bool(app.screen.query("#research-more-slides")), "More outputs"
    )
    return studio


@pytest.mark.parametrize(
    ("source_count", "enabled"),
    (
        (1, ("data-table", "slides", "audio-summary")),
        (2, ("data-table", "literature-matrix", "corpus-gap-finder", "audio-summary")),
    ),
)
async def test_more_outputs_enables_native_server_owners_at_the_probed_revision(
    source_count, enabled
):
    app = App()
    async with app.run_test(size=(120, 32)) as pilot:
        await _mount_and_open_more_outputs(app, pilot, _NativeServerPort(source_count))
        for kind in enabled:
            action = app.screen.query_one(f"#research-more-{kind}", Button)
            assert not action.disabled, kind

        app.screen.query_one("#research-more-data-table", Button).press()
        await _until(
            pilot,
            lambda: bool(app.screen.query("#research-studio-max-rows")),
            "Data Table options",
        )


@pytest.mark.parametrize("initial_status", ("queued", "cancelled"))
async def test_terminal_native_table_cancellation_requires_a_new_generation(
    initial_status,
):
    port = _NativeServerPort(1)
    status = initial_status
    job_uuid, table_uuid = str(UUID(int=54)), str(UUID(int=60))
    launched = []
    cancelled = []

    class TableServer:
        async def _request(self, method, path, *, headers, json_data=None):
            nonlocal status
            assert headers == {"X-TLDW-Expected-User-ID": "1"}
            if method == "POST":
                assert path == "/api/v1/workspaces/observatory/research-data-tables"
                launched.append(json_data["operation_id"])
                return {
                    "job_id": 54,
                    "job_uuid": job_uuid,
                    "status": status,
                    "table": {
                        "uuid": table_uuid,
                        "name": "Data Table",
                        "prompt": "Extract",
                        "status": "generating",
                        "row_count": 0,
                        "version": 1,
                        "workspace_tag": "workspace:observatory",
                    },
                }
            assert path == "/api/v1/data-tables/jobs/54"
            if method == "DELETE":
                cancelled.append(path)
                status = "cancelled"
                return {"success": True, "job_id": 54, "status": status}
            assert method == "GET"
            return {
                "id": 54,
                "uuid": job_uuid,
                "status": status,
                "job_type": "data_table_generate",
                "owner_user_id": "1",
                "created_at": "2026-11-24T19:00:00Z",
                "started_at": None,
                "completed_at": None,
                "cancelled_at": None,
                "cancellation_reason": None,
                "progress_percent": 0,
                "progress_message": "queued",
                "result": None,
                "error_message": None,
                "table_uuid": table_uuid,
            }

    server = TableServer()
    service = ResearchStudioService(None, ServerStudioAdapter(lambda _: server))
    port.generate_output = service.generate
    app = App()
    async with app.run_test(size=(120, 32)) as pilot:
        studio = await _mount_and_open_more_outputs(app, pilot, port)
        app.screen.query_one("#research-more-data-table", Button).press()
        await _until(
            pilot,
            lambda: bool(app.screen.query("#research-studio-generate")),
            "Data Table options",
        )
        app.screen.query_one("#research-studio-generate", Button).press()
        await _until(pilot, lambda: bool(launched), "native table admission")
        if initial_status == "queued":
            studio.query_one("#research-studio-stop", Button).press()
        await _until(pilot, lambda: not studio.is_busy, "settled native generation")
        assert len(cancelled) == (1 if initial_status == "queued" else 0)
        assert len(launched) == 1
        assert studio.query_one("#research-studio-retry", Button).disabled
        assert "Start a new generation" in str(
            studio.query_one("#research-studio-status", Static).render()
        )


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


async def test_audio_options_preview_the_selected_voice_without_generating(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Research_Workspace.studio_models import StudioSpeechChoice
    from tldw_chatbook.UI.Research_Workspace_Modules import studio_dialogs

    first = StudioSpeechChoice("audio_cpp", "reader", "voice-a", "wav", True)
    second = StudioSpeechChoice("audio_cpp", "reader", "voice-b", "wav", True)
    calls = []

    async def preview(choice, speed):
        calls.append((choice, speed))
        return b"audio sample"

    async def play(data, format):
        calls.append((data, format))

    monkeypatch.setattr(studio_dialogs, "play_voice_preview", play, raising=False)
    app, _, _ = build_research_app(tmp_path, monkeypatch)
    with patch_app_global("get_cli_setting", side_effect=_settings_without_splash):
        async with app.run_test(size=(60, 20)) as pilot:
            await open_chat(app, pilot)
            modal = studio_dialogs.ResearchStudioOptions(
                studio_dialogs.ResearchOutputKind.AUDIO_SUMMARY, "Local TTS history",
                "On device", 2, speech_choices=(first, second), speech_preview=preview,
            )
            await app.push_screen(modal)
            modal.query_one("#research-studio-speech", Select).value = 1
            modal.query_one("#research-studio-speech-speed", Input).value = "1.2"
            button = modal.query_one("#research-studio-voice-preview", Button)
            button.focus()
            await pilot.press("enter")
            await _until(pilot, lambda: len(calls) == 2, "selected voice preview")
            assert calls == [(second, 1.2), (b"audio sample", "wav")]
            assert app.screen is modal and app.focused is button


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

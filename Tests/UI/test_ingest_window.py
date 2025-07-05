# test_ingest_window.py
#
# Imports
import pytest
import pytest_asyncio
from pytest_mock import MockerFixture  # For mocking
from pathlib import Path
#
# Third-party Libraries
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Select, Checkbox, TextArea, RadioSet, RadioButton, Collapsible, ListView, \
    ListItem, Markdown, LoadingIndicator, Label, Static
from textual.containers import Container, VerticalScroll, Horizontal, Vertical
from textual.pilot import Pilot
from textual.css.query import QueryError
#
# Local Imports
from tldw_chatbook.app import TldwCli  # The main app
from tldw_chatbook.UI.Ingest_Window import IngestWindow, MEDIA_TYPES  # Import MEDIA_TYPES
from tldw_chatbook.tldw_api.schemas import ProcessVideoRequest, ProcessAudioRequest, ProcessPDFRequest, \
    ProcessEbookRequest, ProcessDocumentRequest, ProcessXMLRequest, ProcessMediaWikiRequest
#
#
########################################################################################################################
#
# Fixtures and Helper Functions

# Helper to get the IngestWindow instance from the app
async def get_ingest_window(pilot: Pilot) -> IngestWindow:
    ingest_window_query = pilot.app.query(IngestWindow)
    assert len(ingest_window_query) > 0, "IngestWindow not found"
    return ingest_window_query.first()


@pytest_asyncio.fixture
async def app_pilot() -> Pilot:
    app = TldwCli()
    async with app.run_test() as pilot:
        # Ensure the Ingest tab is active. Default is Chat.
        # Switching tabs is handled by app.py's on_button_pressed for tab buttons.
        # We need to find the Ingest tab button and click it.
        # Assuming tab IDs are like "tab-ingest"
        try:
            await pilot.click("#tab-ingest")
        except QueryError:
            # Fallback if direct ID click isn't working as expected in test setup
            # This might indicate an issue with tab IDs or pilot interaction timing
            all_buttons = pilot.app.query(Button)
            ingest_tab_button = None
            for btn in all_buttons:
                if btn.id == "tab-ingest":
                    ingest_tab_button = btn
                    break
            assert ingest_tab_button is not None, "Ingest tab button not found"
            await pilot.click(ingest_tab_button)

        # Verify IngestWindow is present and active
        ingest_window = await get_ingest_window(pilot)
        assert ingest_window is not None
        assert ingest_window.display is True, "IngestWindow is not visible after switching to Ingest tab"
        # Also check the app's current_tab reactive variable
        assert pilot.app.current_tab == "ingest", "App's current_tab is not set to 'ingest'"
        yield pilot


# Test Class
class TestIngestWindowTLDWAPI:

    @pytest.mark.asyncio
    async def test_initial_tldw_api_nav_buttons_and_views(self, app_pilot: Pilot):
        ingest_window = await get_ingest_window(app_pilot)
        # The IngestWindow itself is a container, nav buttons are direct children of its "ingest-nav-pane"
        nav_pane = ingest_window.query_one("#ingest-nav-pane")

        for mt in MEDIA_TYPES:
            # Handle the special case where mediawiki_dump becomes just mediawiki in the UI
            ui_mt = "mediawiki" if mt == "mediawiki_dump" else mt
            nav_button_id = f"ingest-nav-api-{ui_mt}"  # IDs don't have #
            view_id = f"ingest-view-api-{ui_mt}"

            # Check navigation button exists
            nav_button = nav_pane.query_one(f"#{nav_button_id}", Button)
            assert nav_button is not None, f"Navigation button {nav_button_id} not found"
            expected_label_part = mt.replace('_', ' ').title()
            if mt == "mediawiki_dump":
                expected_label_part = "MediaWiki Dump"
            elif mt == "pdf":
                expected_label_part = "PDF"
            elif mt == "xml":
                expected_label_part = "XML"
            assert expected_label_part in str(nav_button.label), f"Label for {nav_button_id} incorrect"

            # Check view area exists
            view_area = ingest_window.query_one(f"#{view_id}")
            assert view_area is not None, f"View area {view_id} not found"

            # Check initial visibility based on app's active ingest view
            # This assumes that after switching to Ingest tab, a default sub-view *within* Ingest is activated.
            # If `ingest_active_view` is set (e.g. to "ingest-view-prompts" by default), then
            # all tldw-api views should be hidden.
            active_ingest_view_on_app = app_pilot.app.ingest_active_view
            if view_id != active_ingest_view_on_app:
                assert view_area.display is False, f"{view_id} should be hidden if not the active ingest view ('{active_ingest_view_on_app}')"
            else:
                assert view_area.display is True, f"{view_id} should be visible as it's the active ingest view ('{active_ingest_view_on_app}')"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("media_type", MEDIA_TYPES)
    async def test_tldw_api_navigation_and_view_display(self, app_pilot: Pilot, media_type: str):
        ingest_window = await get_ingest_window(app_pilot)
        # Handle the special case where mediawiki_dump becomes just mediawiki in the UI
        ui_media_type = "mediawiki" if media_type == "mediawiki_dump" else media_type
        nav_button_id = f"ingest-nav-api-{ui_media_type}"
        target_view_id = f"ingest-view-api-{ui_media_type}"

        # Click the navigation button - use offset to click on visible portion
        try:
            await app_pilot.click(f"#{nav_button_id}")
        except Exception:
            # If OutOfBounds, try clicking with offset
            nav_button = ingest_window.query_one(f"#{nav_button_id}", Button)
            await app_pilot.click(nav_button, offset=(5, 5))
        await app_pilot.pause()  # Allow watchers to update display properties

        # Verify target view is visible
        target_view_area = ingest_window.query_one(f"#{target_view_id}")
        assert target_view_area.display is True, f"{target_view_id} should be visible after clicking {nav_button_id}"
        assert app_pilot.app.ingest_active_view == target_view_id, f"App's active ingest view should be {target_view_id}"

        # Verify other TLDW API views are hidden
        for other_mt in MEDIA_TYPES:
            if other_mt != media_type:
                ui_other_mt = "mediawiki" if other_mt == "mediawiki_dump" else other_mt
                other_view_id = f"ingest-view-api-{ui_other_mt}"
                other_view_area = ingest_window.query_one(f"#{other_view_id}")
                assert other_view_area.display is False, f"{other_view_id} should be hidden when {target_view_id} is active"

        # Verify common form elements exist with dynamic IDs
        common_endpoint_input = target_view_area.query_one(f"#tldw-api-endpoint-url-{media_type}", Input)
        assert common_endpoint_input is not None

        common_submit_button = target_view_area.query_one(f"#tldw-api-submit-{media_type}", Button)
        assert common_submit_button is not None

        # Verify media-specific options container and its widgets
        if media_type == "video":
            opts_container = target_view_area.query_one("#tldw-api-video-options")
            assert opts_container.display is True
            widget = opts_container.query_one(f"#tldw-api-video-transcription-model-{media_type}", Input)
            assert widget is not None
        elif media_type == "audio":
            opts_container = target_view_area.query_one("#tldw-api-audio-options")
            assert opts_container.display is True
            widget = opts_container.query_one(f"#tldw-api-audio-transcription-model-{media_type}", Input)
            assert widget is not None
        elif media_type == "pdf":
            opts_container = target_view_area.query_one("#tldw-api-pdf-options")
            assert opts_container.display is True
            widget = opts_container.query_one(f"#tldw-api-pdf-engine-{media_type}", Select)
            assert widget is not None
        elif media_type == "ebook":
            opts_container = target_view_area.query_one("#tldw-api-ebook-options")
            assert opts_container.display is True
            widget = opts_container.query_one(f"#tldw-api-ebook-extraction-method-{media_type}", Select)
            assert widget is not None
        elif media_type == "document":  # Has minimal specific options currently
            opts_container = target_view_area.query_one("#tldw-api-document-options")
            assert opts_container.display is True
            # Example: find the label if one exists
            try:
                label = opts_container.query_one(Label)  # Assuming there's at least one label
                assert label is not None
            except QueryError:  # If no labels, this is fine for doc
                pass
        elif media_type == "xml":
            opts_container = target_view_area.query_one("#tldw-api-xml-options")
            assert opts_container.display is True
            widget = opts_container.query_one(f"#tldw-api-xml-auto-summarize-{media_type}", Checkbox)
            assert widget is not None
        elif media_type == "mediawiki_dump":
            opts_container = target_view_area.query_one("#tldw-api-mediawiki-options")
            assert opts_container.display is True
            widget = opts_container.query_one(f"#tldw-api-mediawiki-wiki-name-{media_type}", Input)
            assert widget is not None

    @pytest.mark.asyncio
    async def test_tldw_api_video_submission_data_collection(self, app_pilot: Pilot, mocker: MockerFixture):
        media_type = "video"
        ingest_window = await get_ingest_window(app_pilot)

        # Navigate to video tab by clicking its nav button
        ui_media_type = "mediawiki" if media_type == "mediawiki_dump" else media_type
        nav_button_id = f"ingest-nav-api-{ui_media_type}"
        
        # Click the navigation button - use offset to click on visible portion
        try:
            await app_pilot.click(f"#{nav_button_id}")
        except Exception:
            # If OutOfBounds, try clicking with offset
            nav_button = ingest_window.query_one(f"#{nav_button_id}", Button)
            await app_pilot.click(nav_button, offset=(5, 5))
        await app_pilot.pause()  # Allow UI to update

        target_view_id = f"ingest-view-api-{ui_media_type}"
        target_view_area = ingest_window.query_one(f"#{target_view_id}")
        assert target_view_area.display is True, "Video view area not displayed after click"

        # Mock the API client and its methods
        mock_api_client_instance = mocker.MagicMock()
        # Make process_video an async mock
        mock_process_video = mocker.AsyncMock(return_value=mocker.MagicMock())
        mock_api_client_instance.process_video = mock_process_video
        mock_api_client_instance.close = mocker.AsyncMock()

        mocker.patch("tldw_chatbook.Event_Handlers.ingest_events.TLDWAPIClient", return_value=mock_api_client_instance)

        # Set form values
        endpoint_url_input = target_view_area.query_one(f"#tldw-api-endpoint-url-{media_type}", Input)
        urls_textarea = target_view_area.query_one(f"#tldw-api-urls-{media_type}", TextArea)
        video_trans_model_input = target_view_area.query_one(f"#tldw-api-video-transcription-model-{media_type}", Input)
        auth_method_select = target_view_area.query_one(f"#tldw-api-auth-method-{media_type}", Select)

        endpoint_url_input.value = "http://fakeapi.com"
        urls_textarea.text = "http://example.com/video.mp4"
        video_trans_model_input.value = "test_video_model"
        auth_method_select.value = "config_token"

        app_pilot.app.app_config = {"tldw_api": {"auth_token_config": "fake_token"}}

        submit_button_id = f"tldw-api-submit-{media_type}"
        await app_pilot.click(f"#{submit_button_id}")
        await app_pilot.pause(delay=0.5)

        mock_process_video.assert_called_once()
        call_args = mock_process_video.call_args[0]

        assert len(call_args) >= 1, "process_video not called with request_model"
        request_model_arg = call_args[0]

        assert isinstance(request_model_arg, ProcessVideoRequest)
        assert request_model_arg.urls == ["http://example.com/video.mp4"]
        assert request_model_arg.transcription_model == "test_video_model"
        assert request_model_arg.api_key == "fake_token"

        # Example for local_file_paths if it's the second argument
        if len(call_args) > 1:
            local_files_arg = call_args[1]
            assert local_files_arg == [], "local_files_arg was not empty"
        else:
            # This case implies process_video might not have received local_file_paths,
            # which could be an issue if it's expected. For now, let's assume it's optional.
            pass

# tldw_chatbook/Event_Handlers/tldw_api_events.py
#
# TLDW API form handling and submission event handlers
#
# Imports
import json
from os import getenv
from typing import TYPE_CHECKING, Optional, List, Any, Dict, Union

# 3rd-party Libraries
from loguru import logger
from textual.widgets import Select, Input, TextArea, Checkbox, Label, Button, LoadingIndicator
from textual.css.query import QueryError
from textual.containers import Container

# Local Imports
from ..Constants import ALL_TLDW_API_OPTION_CONTAINERS
from ..UI.Ingest_Window import IngestWindow
from ..config import get_cli_setting
from ..tldw_api import (
    TLDWAPIClient, ProcessVideoRequest, ProcessAudioRequest,
    APIConnectionError, APIRequestError, APIResponseError, AuthenticationError,
    MediaItemProcessResult, ProcessedMediaWikiPage, BatchMediaProcessResponse,
    ProcessPDFRequest, ProcessEbookRequest, ProcessDocumentRequest,
    ProcessXMLRequest, ProcessMediaWikiRequest, ProcessPlaintextRequest,
    BatchProcessXMLResponse
)

if TYPE_CHECKING:
    from ..app import TldwCli

# --- TLDW API Form Handlers ---
async def handle_tldw_api_auth_method_changed(app: 'TldwCli', event: Union[Select.Changed, str]) -> None:
    event_value = str(event.value) if isinstance(event, Select.Changed) else event
    logger.debug(f"TLDW API Auth method changed to: {event_value}")
    try:
        custom_token_input = app.query_one("#tldw-api-custom-token", Input)
        custom_token_label = app.query_one("#tldw-api-custom-token-label", Label)
        if event_value == "custom_token":
            custom_token_input.display = True
            custom_token_label.display = True
            custom_token_input.focus()
        else:
            custom_token_input.display = False
            custom_token_label.display = False
    except QueryError as e:
        logger.error(f"UI component not found for TLDW API auth method change: {e}")

async def handle_tldw_api_media_type_changed(app: 'TldwCli', event: Union[Select.Changed, str]) -> None:
    """Shows/hides media type specific option containers."""
    event_value = str(event.value) if isinstance(event, Select.Changed) else event
    logger.debug(f"TLDW API Media Type changed to: {event_value}")
    try:
        # Hide all specific option containers first
        for container_id in ALL_TLDW_API_OPTION_CONTAINERS:
            try:
                app.query_one(f"#{container_id}", Container).display = False
            except QueryError:
                pass  # Container might not exist if not composed yet, or for all types

        # Show the relevant one
        target_container_id = f"tldw-api-{event_value.lower().replace('_', '-')}-options"
        if event_value:  # If a type is selected
            try:
                container_to_show = app.query_one(f"#{target_container_id}", Container)
                container_to_show.display = True
                logger.info(f"Displaying options container: {target_container_id}")
            except QueryError:
                logger.warning(f"Options container #{target_container_id} not found for media type {event_value}.")

    except QueryError as e:
        logger.error(f"UI component not found for TLDW API media type change: {e}")
    except Exception as ex:
        logger.error(f"Unexpected error handling media type change: {ex}", exc_info=True)

def _collect_common_form_data(app: 'TldwCli', media_type: str) -> Dict[str, Any]:
    """Collects common data fields from the TLDW API form for a given media_type."""
    data = {}
    # Keep track of which field was being processed for better error messages
    # The f-string will be used in the actual query_one call.
    current_field_template_for_error = "Unknown Field-{media_type}"

    # Get the IngestWindow instance to access selected_local_files
    try:
        ingest_window = app.query_one(IngestWindow)
    except QueryError:
        logger.error("Could not find IngestWindow instance to retrieve selected files.")
        # Decide how to handle this: raise error, return empty, or notify.
        # For now, let's log and proceed, which means local_files might be empty.
        # A more robust solution might involve ensuring IngestWindow is always available.
        ingest_window = None  # Or handle error more strictly

    try:
        current_field_template_for_error = f"#tldw-api-urls-{media_type}"
        data["urls"] = [url.strip() for url in app.query_one(f"#tldw-api-urls-{media_type}", TextArea).text.splitlines() if url.strip()]
        logger.debug(f"Collected URLs for {media_type}: {data['urls']}")

        # Try to get local files from the individual window's selected_local_files
        data["local_files"] = []
        # First try the old way with IngestWindow
        if ingest_window and hasattr(ingest_window, 'selected_local_files') and media_type in ingest_window.selected_local_files:
            # Convert Path objects to strings as expected by the API client processing functions
            data["local_files"] = [str(p) for p in ingest_window.selected_local_files[media_type]]
        else:
            # Try to find the specific media type window
            try:
                # Map media type to window class
                window_map = {
                    "video": "IngestTldwApiVideoWindow",
                    "audio": "IngestTldwApiAudioWindow", 
                    "pdf": "IngestTldwApiPdfWindow",
                    "ebook": "IngestTldwApiEbookWindow",
                    "document": "IngestTldwApiDocumentWindow",
                    "xml": "IngestTldwApiXmlWindow",
                    "mediawiki_dump": "IngestTldwApiMediaWikiWindow"
                }
                if media_type in window_map:
                    from ..Widgets.Media_Ingest import IngestTldwApiPdfWindow
                    from ..Widgets.Media_Ingest import IngestTldwApiVideoWindow
                    from ..Widgets.Media_Ingest import IngestTldwApiDocumentWindow
                    from ..Widgets.Media_Ingest import IngestTldwApiXmlWindow
                    from ..Widgets.Media_Ingest import IngestTldwApiMediaWikiWindow
                    from ..Widgets.Media_Ingest import IngestTldwApiAudioWindow
                    from ..Widgets.Media_Ingest import IngestTldwApiEbookWindow
                    window_classes = {
                        "video": IngestTldwApiVideoWindow,
                        "audio": IngestTldwApiAudioWindow,
                        "pdf": IngestTldwApiPdfWindow,
                        "ebook": IngestTldwApiEbookWindow,
                        "document": IngestTldwApiDocumentWindow,
                        "xml": IngestTldwApiXmlWindow,
                        "mediawiki_dump": IngestTldwApiMediaWikiWindow
                    }
                    window_class = window_classes.get(media_type)
                    if window_class:
                        media_window = app.query_one(window_class)
                        if hasattr(media_window, 'selected_local_files'):
                            data["local_files"] = [str(p) for p in media_window.selected_local_files]
            except:
                # If we can't find the window, just use empty list
                pass
        
        logger.debug(f"Collected local files for {media_type}: {data['local_files']}")
        
        # Double-check that URLs and local files are separate
        if data['urls'] and data['local_files']:
            logger.warning(f"Both URLs and local files present for {media_type}. URLs: {data['urls']}, Files: {data['local_files']}")

        current_field_template_for_error = f"#tldw-api-title-{media_type}"
        data["title"] = app.query_one(f"#tldw-api-title-{media_type}", Input).value or None

        current_field_template_for_error = f"#tldw-api-author-{media_type}"
        data["author"] = app.query_one(f"#tldw-api-author-{media_type}", Input).value or None

        current_field_template_for_error = f"#tldw-api-keywords-{media_type}"
        data["keywords_str"] = app.query_one(f"#tldw-api-keywords-{media_type}", TextArea).text

        current_field_template_for_error = f"#tldw-api-custom-prompt-{media_type}"
        data["custom_prompt"] = app.query_one(f"#tldw-api-custom-prompt-{media_type}", TextArea).text or None

        current_field_template_for_error = f"#tldw-api-system-prompt-{media_type}"
        data["system_prompt"] = app.query_one(f"#tldw-api-system-prompt-{media_type}", TextArea).text or None

        current_field_template_for_error = f"#tldw-api-perform-analysis-{media_type}"
        data["perform_analysis"] = app.query_one(f"#tldw-api-perform-analysis-{media_type}", Checkbox).value

        current_field_template_for_error = f"#tldw-api-overwrite-db-{media_type}"
        data["overwrite_existing_db"] = app.query_one(f"#tldw-api-overwrite-db-{media_type}", Checkbox).value

        current_field_template_for_error = f"#tldw-api-perform-chunking-{media_type}"
        data["perform_chunking"] = app.query_one(f"#tldw-api-perform-chunking-{media_type}", Checkbox).value

        current_field_template_for_error = f"#tldw-api-chunk-method-{media_type}"
        data["chunk_method"] = app.query_one(f"#tldw-api-chunk-method-{media_type}", Select).value

        current_field_template_for_error = f"#tldw-api-max-chunk-size-{media_type}"
        max_chunk_size_val = app.query_one(f"#tldw-api-max-chunk-size-{media_type}", Input).value
        try:
            data["max_chunk_size"] = int(max_chunk_size_val) if max_chunk_size_val else None
        except ValueError:
            logger.warning(f"Invalid max_chunk_size value for {media_type}: '{max_chunk_size_val}'. Using default.")
            data["max_chunk_size"] = None

        current_field_template_for_error = f"#tldw-api-chunk-overlap-{media_type}"
        chunk_overlap_val = app.query_one(f"#tldw-api-chunk-overlap-{media_type}", Input).value
        try:
            data["chunk_overlap"] = int(chunk_overlap_val) if chunk_overlap_val else None
        except ValueError:
            logger.warning(f"Invalid chunk_overlap value for {media_type}: '{chunk_overlap_val}'. Using default.")
            data["chunk_overlap"] = None

        # Parse keywords
        if data["keywords_str"]:
            data["keywords"] = [k.strip() for k in data["keywords_str"].split(",") if k.strip()]
        else:
            data["keywords"] = []

        # Clean up fields
        del data["keywords_str"]

        return data

    except QueryError as e:
        logger.error(f"UI field not found: {current_field_template_for_error}. Exception: {e}")
        raise ValueError(f"Required field not found: {current_field_template_for_error}. Check UI composition for media type: {media_type}")
    except Exception as e:
        logger.error(f"Unexpected error collecting common data for {media_type}: {e}", exc_info=True)
        raise

def _collect_video_specific_data(app: 'TldwCli', common_data: Dict[str, Any], media_type: str) -> ProcessVideoRequest:
    """Collects video-specific data from the form."""
    try:
        diarize = app.query_one(f"#tldw-api-diarize-{media_type}", Checkbox).value
        transcription_model = app.query_one(f"#tldw-api-transcription-model-{media_type}", Select).value
        if transcription_model == Select.BLANK:
            transcription_model = None
        
        # Get start and end time fields
        start_time = app.query_one(f"#tldw-api-video-start-time-{media_type}", Input).value or None
        end_time = app.query_one(f"#tldw-api-video-end-time-{media_type}", Input).value or None

        return ProcessVideoRequest(
            urls=common_data.get("urls"),
            custom_prompt=common_data.get("custom_prompt"),
            system_prompt=common_data.get("system_prompt"),
            keywords=common_data.get("keywords"),
            perform_analysis=common_data.get("perform_analysis", False),
            overwrite_existing_db=common_data.get("overwrite_existing_db", False),
            perform_chunking=common_data.get("perform_chunking", False),
            chunk_method=common_data.get("chunk_method", "words"),
            max_chunk_size=common_data.get("max_chunk_size"),
            chunk_overlap=common_data.get("chunk_overlap"),
            title=common_data.get("title"),
            author=common_data.get("author"),
            diarize=diarize,
            transcription_model=transcription_model,
            start_time=start_time,
            end_time=end_time
        )
    except QueryError as e:
        logger.error(f"Video-specific UI field not found: {e}")
        raise ValueError(f"Video-specific field error: {str(e)}")

def _collect_audio_specific_data(app: 'TldwCli', common_data: Dict[str, Any], media_type: str) -> ProcessAudioRequest:
    """Collects audio-specific data from the form."""
    try:
        diarize = app.query_one(f"#tldw-api-diarize-{media_type}", Checkbox).value
        transcription_model = app.query_one(f"#tldw-api-transcription-model-{media_type}", Select).value
        if transcription_model == Select.BLANK:
            transcription_model = None
        
        # Get start and end time fields
        start_time = app.query_one(f"#tldw-api-audio-start-time-{media_type}", Input).value or None
        end_time = app.query_one(f"#tldw-api-audio-end-time-{media_type}", Input).value or None

        return ProcessAudioRequest(
            urls=common_data.get("urls"),
            custom_prompt=common_data.get("custom_prompt"),
            system_prompt=common_data.get("system_prompt"),
            keywords=common_data.get("keywords"),
            perform_analysis=common_data.get("perform_analysis", False),
            overwrite_existing_db=common_data.get("overwrite_existing_db", False),
            perform_chunking=common_data.get("perform_chunking", False),
            chunk_method=common_data.get("chunk_method", "words"),
            max_chunk_size=common_data.get("max_chunk_size"),
            chunk_overlap=common_data.get("chunk_overlap"),
            title=common_data.get("title"),
            author=common_data.get("author"),
            diarize=diarize,
            transcription_model=transcription_model,
            start_time=start_time,
            end_time=end_time
        )
    except QueryError as e:
        logger.error(f"Audio-specific UI field not found: {e}")
        raise ValueError(f"Audio-specific field error: {str(e)}")

def _collect_pdf_specific_data(app: 'TldwCli', common_data: Dict[str, Any], media_type: str) -> ProcessPDFRequest:
    """Collects PDF-specific data from the form."""
    return ProcessPDFRequest(
        urls=common_data.get("urls"),
        custom_prompt=common_data.get("custom_prompt"),
        system_prompt=common_data.get("system_prompt"),
        keywords=common_data.get("keywords"),
        perform_analysis=common_data.get("perform_analysis", False),
        overwrite_existing_db=common_data.get("overwrite_existing_db", False),
        perform_chunking=common_data.get("perform_chunking", False),
        chunk_method=common_data.get("chunk_method", "words"),
        max_chunk_size=common_data.get("max_chunk_size"),
        chunk_overlap=common_data.get("chunk_overlap"),
        title=common_data.get("title"),
        author=common_data.get("author")
    )

def _collect_ebook_specific_data(app: 'TldwCli', common_data: Dict[str, Any], media_type: str) -> ProcessEbookRequest:
    """Collects e-book specific data from the form."""
    return ProcessEbookRequest(
        urls=common_data.get("urls"),
        custom_prompt=common_data.get("custom_prompt"),
        system_prompt=common_data.get("system_prompt"),
        keywords=common_data.get("keywords"),
        perform_analysis=common_data.get("perform_analysis", False),
        overwrite_existing_db=common_data.get("overwrite_existing_db", False),
        perform_chunking=common_data.get("perform_chunking", False),
        chunk_method=common_data.get("chunk_method", "words"),
        max_chunk_size=common_data.get("max_chunk_size"),
        chunk_overlap=common_data.get("chunk_overlap"),
        title=common_data.get("title"),
        author=common_data.get("author")
    )

def _collect_document_specific_data(app: 'TldwCli', common_data: Dict[str, Any], media_type: str) -> ProcessDocumentRequest:
    """Collects document-specific data from the form."""
    return ProcessDocumentRequest(
        urls=common_data.get("urls"),
        custom_prompt=common_data.get("custom_prompt"),
        system_prompt=common_data.get("system_prompt"),
        keywords=common_data.get("keywords"),
        perform_analysis=common_data.get("perform_analysis", False),
        overwrite_existing_db=common_data.get("overwrite_existing_db", False),
        perform_chunking=common_data.get("perform_chunking", False),
        chunk_method=common_data.get("chunk_method", "words"),
        max_chunk_size=common_data.get("max_chunk_size"),
        chunk_overlap=common_data.get("chunk_overlap"),
        title=common_data.get("title"),
        author=common_data.get("author")
    )

def _collect_plaintext_specific_data(app: 'TldwCli', common_data: Dict[str, Any], media_type: str) -> ProcessPlaintextRequest:
    """Collects plaintext-specific data from the form."""
    return ProcessPlaintextRequest(
        urls=common_data.get("urls"),
        custom_prompt=common_data.get("custom_prompt"),
        system_prompt=common_data.get("system_prompt"),
        keywords=common_data.get("keywords"),
        perform_analysis=common_data.get("perform_analysis", False),
        overwrite_existing_db=common_data.get("overwrite_existing_db", False),
        perform_chunking=common_data.get("perform_chunking", False),
        chunk_method=common_data.get("chunk_method", "words"),
        max_chunk_size=common_data.get("max_chunk_size"),
        chunk_overlap=common_data.get("chunk_overlap"),
        title=common_data.get("title"),
        author=common_data.get("author")
    )

def _collect_xml_specific_data(app: 'TldwCli', common_api_data: Dict[str, Any], media_type: str) -> ProcessXMLRequest:
    """Collects XML-specific data from the form."""
    try:
        # Parse per_segment_summary
        per_segment_summary_input = app.query_one(f"#tldw-api-per-segment-summary-{media_type}", Input)
        per_segment_summary_str = per_segment_summary_input.value.strip()
        per_segment_summary = False  # Default value
        if per_segment_summary_str.lower() in ["true", "1", "yes", "y"]:
            per_segment_summary = True
            logger.debug(f"per_segment_summary: True")
        elif per_segment_summary_str.lower() in ["false", "0", "no", "n", ""]:
            per_segment_summary = False
            logger.debug(f"per_segment_summary: False (explicit or blank)")
        else:
            logger.warning(f"Invalid per_segment_summary value: '{per_segment_summary_str}'. Using default (False).")

        return ProcessXMLRequest(
            # Note: URLs might be empty/None for XML, should use local_file_paths instead
            custom_prompt=common_api_data.get("custom_prompt"),
            system_prompt=common_api_data.get("system_prompt"),
            keywords=common_api_data.get("keywords"),
            title=common_api_data.get("title"),
            author=common_api_data.get("author"),
            per_segment_summary=per_segment_summary
        )
    except QueryError as e:
        logger.error(f"XML-specific UI field not found: {e}")
        raise ValueError(f"XML-specific field error: {str(e)}")

def _collect_mediawiki_specific_data(app: 'TldwCli', common_api_data: Dict[str, Any], media_type: str) -> ProcessMediaWikiRequest:
    """Collects MediaWiki-specific data from the form."""
    try:
        namespaces_val = app.query_one(f"#tldw-api-namespaces-{media_type}", Input).value.strip()
        namespaces_list = None
        if namespaces_val:
            try:
                namespaces_list = [int(ns.strip()) for ns in namespaces_val.split(",") if ns.strip()]
            except ValueError:
                logger.warning(f"Invalid namespaces value for {media_type}: '{namespaces_val}'. Using default.")
                namespaces_list = None

        return ProcessMediaWikiRequest(
            # Note: URLs might be empty/None for MediaWiki dumps
            custom_prompt=common_api_data.get("custom_prompt"),
            system_prompt=common_api_data.get("system_prompt"),
            keywords=common_api_data.get("keywords"),
            title=common_api_data.get("title"),
            author=common_api_data.get("author"),
            namespaces=namespaces_list  # Will handle None gracefully
        )
    except QueryError as e:
        logger.error(f"MediaWiki-specific UI field not found: {e}")
        raise ValueError(f"MediaWiki-specific field error: {str(e)}")

async def handle_tldw_api_submit_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    if not event.button.id:
        logger.error("Submit button pressed but has no ID. Cannot determine media_type.")
        app.notify("Critical error: Submit button has no ID.", severity="error")
        return

    logger.info(f"TLDW API Submit button pressed: {event.button.id}")

    selected_media_type = event.button.id.replace("tldw-api-submit-", "")
    logger.info(f"Extracted media_type: {selected_media_type} from button ID.")

    app.notify(f"Processing {selected_media_type} request via tldw API...")

    try:
        loading_indicator = app.query_one(f"#tldw-api-loading-indicator-{selected_media_type}", LoadingIndicator)
        status_area = app.query_one(f"#tldw-api-status-area-{selected_media_type}", TextArea)
        submit_button = event.button  # This is already the correct button
        endpoint_url_input = app.query_one(f"#tldw-api-endpoint-url-{selected_media_type}", Input)
        auth_method_select = app.query_one(f"#tldw-api-auth-method-{selected_media_type}", Select)
    except QueryError as e:
        logger.error(f"Critical UI component missing for media_type '{selected_media_type}': {e}")
        app.notify(f"Error: UI component missing for {selected_media_type}: {e.widget.id if hasattr(e, 'widget') and e.widget else 'Unknown'}. Cannot proceed.", severity="error")
        return

    endpoint_url = endpoint_url_input.value.strip()
    auth_method = str(auth_method_select.value)  # Ensure it's a string

    # --- Input Validation ---
    if not endpoint_url:
        app.notify("API Endpoint URL is required.", severity="error")
        endpoint_url_input.focus()
        # No need to revert UI state as it hasn't been changed yet
        return

    if not (endpoint_url.startswith("http://") or endpoint_url.startswith("https://")):
        app.notify("API Endpoint URL must start with http:// or https://.", severity="error")
        endpoint_url_input.focus()
        # No need to revert UI state
        return

    if auth_method == str(Select.BLANK):
        app.notify("Please select an Authentication Method.", severity="error")
        auth_method_select.focus()
        return

    # --- Set UI to Loading State ---
    loading_indicator.display = True
    status_area.clear()
    status_area.load_text("Validating inputs and preparing request...")
    status_area.display = True
    submit_button.disabled = True
    # app.notify is already called at the start of the function

    def _reset_ui():
        """Return the widgets to their idle state after a hard failure."""
        loading_indicator.display = False
        submit_button.disabled = False
        status_area.load_text("Submission halted.")

    # --- Get Auth Token (after basic validations pass) ---
    auth_token: Optional[str] = None
    try:
        if auth_method == "custom_token":
            custom_token_input = app.query_one(f"#tldw-api-custom-token-{selected_media_type}", Input)
            auth_token = custom_token_input.value.strip()
            if not auth_token:
                app.notify("Custom Auth Token is required for selected method.", severity="error")
                custom_token_input.focus()
                # Revert UI loading state
                loading_indicator.display = False
                submit_button.disabled = False
                status_area.load_text("Custom token required. Submission halted.")
                return

        elif auth_method == "config_token":
            # 1.  Look in the active config, then in the environment.
            auth_token = (
                    get_cli_setting("tldw_api", "auth_token")  # ~/.config/tldw_cli/config.toml
                    or get_cli_setting("tldw_api", "api_key")  # Alternative config key
                    or getenv("TDLW_AUTH_TOKEN")  # optional override
                    or getenv("TLDW_API_KEY")  # Alternative env var
            )

            # 2. Abort early if we still have nothing.
            if not auth_token:
                msg = (
                    "Auth token not found — add it to the [tldw_api] section as "
                    "`auth_token = \"<your token>\"` or `api_key = \"<your key>\"`, "
                    "or export TDLW_AUTH_TOKEN or TLDW_API_KEY."
                )
                logger.error(msg)
                app.notify(msg, severity="error")
                _reset_ui()
                return
    except QueryError as e:
        logger.error(f"UI component not found for TLDW API auth token for {selected_media_type}: {e}")
        app.notify(f"Error: Missing UI field for auth for {selected_media_type}: {e.widget.id if hasattr(e, 'widget') and e.widget else 'Unknown'}", severity="error")
        loading_indicator.display = False
        submit_button.disabled = False
        status_area.load_text("Error accessing auth fields. Submission halted.")
        return

    status_area.load_text("Collecting form data and building request...")
    request_model: Optional[Any] = None
    local_file_paths: Optional[List[str]] = None
    try:
        common_data = _collect_common_form_data(app, selected_media_type)  # Pass selected_media_type
        local_file_paths = common_data.pop("local_files", [])

        if selected_media_type == "video":
            request_model = _collect_video_specific_data(app, common_data, selected_media_type)
            logger.debug(f"Video request model URLs: {request_model.urls}")
        elif selected_media_type == "audio":
            request_model = _collect_audio_specific_data(app, common_data, selected_media_type)
        elif selected_media_type == "pdf":
            request_model = _collect_pdf_specific_data(app, common_data, selected_media_type)
        elif selected_media_type == "ebook":
            request_model = _collect_ebook_specific_data(app, common_data, selected_media_type)
        elif selected_media_type == "document":
            request_model = _collect_document_specific_data(app, common_data, selected_media_type)
        elif selected_media_type == "plaintext":
            request_model = _collect_plaintext_specific_data(app, common_data, selected_media_type)
        elif selected_media_type == "xml":
            request_model = _collect_xml_specific_data(app, common_data, selected_media_type)
        elif selected_media_type == "mediawiki_dump":
            request_model = _collect_mediawiki_specific_data(app, common_data, selected_media_type)
        else:
            app.notify(f"Media type '{selected_media_type}' not yet supported by this client form.", severity="warning")
            loading_indicator.display = False
            submit_button.disabled = False
            status_area.load_text("Unsupported media type selected. Submission halted.")
            return
    except (QueryError, ValueError) as e:
        logger.error(f"Error collecting form data for {selected_media_type}: {e}", exc_info=True)
        app.notify(f"Error in form data for {selected_media_type}: {str(e)[:100]}. Please check fields.", severity="error")
        loading_indicator.display = False
        submit_button.disabled = False
        status_area.load_text(f"Error processing form data: {str(e)[:100]}. Submission halted.")
        return
    except Exception as e:
        logger.error(f"Unexpected error preparing request model for TLDW API ({selected_media_type}): {e}", exc_info=True)
        app.notify("Error: Could not prepare data for API request.", severity="error")
        loading_indicator.display = False
        submit_button.disabled = False
        status_area.load_text("Unexpected error preparing request. Submission halted.")
        return

    if not request_model:
        app.notify("Failed to create request model.", severity="error")
        loading_indicator.display = False
        submit_button.disabled = False
        status_area.load_text("Internal error: Failed to create request model. Submission halted.")
        return

    # URL/Local file validation (adjust for XML/MediaWiki which primarily use local_file_paths)
    if not getattr(request_model, 'urls', None) and not local_file_paths:
        # This check might be specific to certain request models, adjust if necessary
        # For XML and MediaWiki, local_file_paths is primary and urls might not exist on model
        is_xml_or_mediawiki = selected_media_type in ["xml", "mediawiki_dump"]
        if not is_xml_or_mediawiki or (is_xml_or_mediawiki and not local_file_paths):
            app.notify("Please provide at least one URL or one local file path.", severity="warning")
            try:
                app.query_one(f"#tldw-api-urls-{selected_media_type}", TextArea).focus()
            except QueryError:
                pass
            loading_indicator.display = False
            submit_button.disabled = False
            status_area.load_text("Missing URL or local file. Submission halted.")
            return

    status_area.load_text("Connecting to TLDW API and sending request...")
    # Determine if auth_token is a Bearer token or API key based on auth_method
    if auth_method == "custom_token":
        # Custom token is treated as Bearer token
        api_client = TLDWAPIClient(base_url=endpoint_url)
        api_client.bearer_token = auth_token
    else:
        # Config token is treated as API key
        api_client = TLDWAPIClient(base_url=endpoint_url, token=auth_token)
    overwrite_db = common_data.get("overwrite_existing_db", False)  # From common_data

    # Worker and callbacks remain largely the same but need to use the correct UI element IDs for this tab
    # The on_worker_success and on_worker_failure need to know which loading_indicator/submit_button/status_area to update.
    # This is implicitly handled as they are queried again using the selected_media_type.

    async def process_media_worker():  # This worker is fine
        nonlocal request_model
        try:
            if selected_media_type == "video":
                logger.debug(f"Processing video with URLs: {getattr(request_model, 'urls', None)}")
                logger.debug(f"Processing video with local_file_paths: {local_file_paths}")
                return await api_client.process_video(request_model, local_file_paths)
            elif selected_media_type == "audio":
                return await api_client.process_audio(request_model, local_file_paths)
            elif selected_media_type == "pdf":
                return await api_client.process_pdf(request_model, local_file_paths)
            elif selected_media_type == "ebook":
                return await api_client.process_ebook(request_model, local_file_paths)
            elif selected_media_type == "document":
                return await api_client.process_document(request_model, local_file_paths)
            elif selected_media_type == "plaintext":
                return await api_client.process_plaintext(request_model, local_file_paths)
            elif selected_media_type == "xml":
                if not local_file_paths:
                    raise ValueError("XML processing requires a local file path.")
                return await api_client.process_xml(request_model, local_file_paths[0])
            elif selected_media_type == "mediawiki_dump":
                if not local_file_paths:
                    raise ValueError("MediaWiki processing requires a local file path.")
                # For streaming, the worker should yield, not return directly.
                # This example shows how to initiate and collect, actual handling of stream in on_success would differ.
                results = []
                async for item in api_client.process_mediawiki_dump(request_model, local_file_paths[0]):
                    results.append(item)
                return results
            else:
                raise NotImplementedError(f"Client-side processing for {selected_media_type} not implemented.")
        finally:
            await api_client.close()

    def on_worker_success(response_data: Any):
        # Query the specific UI elements for this tab
        try:
            current_loading_indicator = app.query_one(f"#tldw-api-loading-indicator-{selected_media_type}", LoadingIndicator)
            current_loading_indicator.display = False
            # current_submit_button = app.query_one(f"#tldw-api-submit-{selected_media_type}", Button) # Button instance is already event.button
            submit_button.disabled = False  # submit_button is already defined from event.button
        except QueryError as e_ui:
            logger.error(f"UI component not found in on_worker_success for {selected_media_type}: {e_ui}")

        app.notify(f"TLDW API request for {selected_media_type} successful. Processing results...", timeout=2)
        logger.info(f"TLDW API Response for {selected_media_type}: {response_data}")

        try:
            current_status_area = app.query_one(f"#tldw-api-status-area-{selected_media_type}", TextArea)
            current_status_area.clear()
        except QueryError:
            logger.error(f"Could not find status_area for {selected_media_type} in on_worker_success.")
            return  # Cannot display results

        if not app.media_db:
            logger.error("Media_DB_v2 not initialized. Cannot ingest API results.")
            app.notify("Error: Local media database not available.", severity="error")
            current_status_area.load_text("## Error\n\nLocal media database not available.")
            return

        processed_count = 0
        error_count = 0
        successful_ingestions_details = []  # To store details of successful items

        # Handle different response types
        results_to_ingest: List[MediaItemProcessResult] = []
        if isinstance(response_data, BatchMediaProcessResponse):
            results_to_ingest = response_data.results
        elif isinstance(response_data, BatchProcessXMLResponse):
            # Convert ProcessXMLResponseItem to MediaItemProcessResult
            for xml_item in response_data.results:
                results_to_ingest.append(MediaItemProcessResult(
                    status=xml_item.status,
                    input_ref=xml_item.input_ref,
                    processing_source=xml_item.input_ref,
                    media_type="xml",
                    metadata={"title": xml_item.title, "author": xml_item.author, "keywords": xml_item.keywords},
                    content=xml_item.content,
                    summary=xml_item.summary,
                    segments=xml_item.segments,
                    error=xml_item.error
                ))
        elif isinstance(response_data, dict) and "results" in response_data:
            if "processed_count" in response_data:
                raw_results = response_data.get("results", [])
                for item_dict in raw_results:
                    # Try to coerce into MediaItemProcessResult, might need specific mapping for XML
                    # For now, assume XML result items can be mostly mapped.
                    results_to_ingest.append(MediaItemProcessResult(**item_dict))

        elif isinstance(response_data, list) and all(isinstance(item, ProcessedMediaWikiPage) for item in response_data):
            # MediaWiki dump (if collected into a list by worker)
            for mw_page in response_data:
                if mw_page.status == "Error":
                    error_count += 1
                    logger.error(f"MediaWiki page '{mw_page.title}' processing error: {mw_page.error_message}")
                    continue
                # Adapt ProcessedMediaWikiPage to MediaItemProcessResult structure for ingestion
                results_to_ingest.append(MediaItemProcessResult(
                    status="Success",  # Assume success if no error status
                    input_ref=mw_page.input_ref or mw_page.title,
                    processing_source=mw_page.title,  # or another identifier
                    media_type="mediawiki_article",  # or "mediawiki_page"
                    metadata={"title": mw_page.title, "page_id": mw_page.page_id, "namespace": mw_page.namespace, "revision_id": mw_page.revision_id, "timestamp": mw_page.timestamp},
                    content=mw_page.content,
                    chunks=[{"text": chunk.get("text", ""), "metadata": chunk.get("metadata", {})} for chunk in mw_page.chunks] if hasattr(mw_page, 'chunks') and mw_page.chunks else None,
                ))
        else:
            logger.error(f"Unexpected TLDW API response data type for {selected_media_type}: {type(response_data)}.")
            current_status_area.load_text(f"## API Request Processed\n\nUnexpected response format. Raw response logged.")
            current_status_area.display = True
            app.notify("Error: Received unexpected data format from API.", severity="error")
            return
        # Add elif for XML if it returns a single ProcessXMLResponseItem or similar

        for item_result in results_to_ingest:
            if item_result.status == "Success":
                media_id_ingested = None  # For storing the ID if ingestion is successful
                try:
                    # Prepare data for add_media_with_keywords
                    # Keywords: API response might not have 'keywords'. Use originally submitted ones if available.
                    # For simplicity, let's assume API response's metadata *might* have keywords.
                    api_keywords = item_result.metadata.get("keywords", []) if item_result.metadata else []
                    if isinstance(api_keywords, str):  # If server returns comma-sep string
                        api_keywords = [k.strip() for k in api_keywords.split(',') if k.strip()]

                    # Chunks for UnvectorizedMediaChunks
                    # Ensure chunks are in the format: [{'text': str, 'start_char': int, ...}, ...]
                    unvectorized_chunks_to_save = None
                    if item_result.chunks:
                        unvectorized_chunks_to_save = []
                        for idx, chunk_item in enumerate(item_result.chunks):
                            if isinstance(chunk_item, dict) and "text" in chunk_item:
                                unvectorized_chunks_to_save.append({
                                    "text": chunk_item.get("text"),
                                    "start_char": chunk_item.get("metadata", {}).get("start_char"),  # Assuming metadata structure
                                    "end_char": chunk_item.get("metadata", {}).get("end_char"),
                                    "chunk_type": chunk_item.get("metadata", {}).get("type", selected_media_type),
                                    "metadata": chunk_item.get("metadata", {})  # Store full chunk metadata
                                })
                            elif isinstance(chunk_item, str):  # If chunks are just strings
                                unvectorized_chunks_to_save.append({"text": chunk_item})
                            else:
                                logger.warning(f"Skipping malformed chunk item: {chunk_item}")

                    media_id, media_uuid, msg = app.media_db.add_media_with_keywords(
                        url=item_result.input_ref,  # Original URL/filename
                        title=item_result.metadata.get("title", item_result.input_ref) if item_result.metadata else item_result.input_ref,
                        media_type=item_result.media_type,
                        content=item_result.content or item_result.transcript,  # Use transcript if content is empty
                        keywords=api_keywords or (request_model.keywords if hasattr(request_model, "keywords") else []),  # Fallback to request
                        prompt=request_model.custom_prompt if hasattr(request_model, "custom_prompt") else None,  # From original request
                        analysis_content=item_result.analysis or item_result.summary,
                        transcription_model=item_result.analysis_details.get("transcription_model") if item_result.analysis_details else (request_model.transcription_model if hasattr(request_model, "transcription_model") else None),
                        author=item_result.metadata.get("author") if item_result.metadata else (request_model.author if hasattr(request_model, "author") else None),
                        # ingestion_date: use current time,
                        overwrite=overwrite_db,  # Use the specific DB overwrite flag
                        chunks=unvectorized_chunks_to_save  # Pass prepared chunks
                    )
                    if media_id:
                        logger.info(f"Successfully ingested '{item_result.input_ref}' into local DB for {selected_media_type}. Media ID: {media_id}. Message: {msg}")
                        processed_count += 1
                        media_id_ingested = media_id  # Store the ID
                    else:
                        logger.error(f"Failed to ingest '{item_result.input_ref}' into local DB for {selected_media_type}. Message: {msg}")
                        error_count += 1
                except Exception as e_ingest:
                    logger.error(f"Error ingesting item '{item_result.input_ref}' for {selected_media_type} into local DB: {e_ingest}", exc_info=True)
                    error_count += 1

                if media_id_ingested:  # Only add to details if successfully ingested
                    successful_ingestions_details.append({
                        "input_ref": item_result.input_ref,
                        "title": item_result.metadata.get("title", "N/A") if item_result.metadata else "N/A",
                        "media_type": item_result.media_type,
                        "db_id": media_id_ingested
                    })
            else:
                logger.error(f"API processing error for '{item_result.input_ref}' ({selected_media_type}): {item_result.error}")
                error_count += 1

        summary_parts = [f"## TLDW API Request Successful ({selected_media_type.title()})\n\n"]
        # ... (rest of summary construction similar to before) ...
        if processed_count == 0 and error_count == 0 and not results_to_ingest:
            summary_parts.append("API request successful, but no items were provided or found for processing.\n")
        elif processed_count == 0 and error_count > 0:
            summary_parts.append(f"API request successful, but no new items were ingested due to errors.\n")
            summary_parts.append(f"- Successfully processed items by API: {processed_count}\n")  # This might be confusing if API said success but ingest failed
            summary_parts.append(f"- Items with errors during API processing or local ingestion: {error_count}\n")
        else:
            summary_parts.append(f"- Successfully processed and ingested items: {processed_count}\n")
            summary_parts.append(f"- Items with errors during API processing or local ingestion: {error_count}\n\n")

        if error_count > 0:
            summary_parts.append("**Please check the application logs for details on any errors.**\n\n")

        if successful_ingestions_details:
            if len(successful_ingestions_details) <= 5:
                summary_parts.append("### Successfully Ingested Items:\n")
                for detail in successful_ingestions_details:
                    title_str = f" (Title: {detail['title']})" if detail['title'] != 'N/A' else ""
                    summary_parts.append(f"- **Input:** `{detail['input_ref']}`{title_str}\n")  # Use backticks for input ref
                    summary_parts.append(f"  - **Type:** {detail['media_type']}, **DB ID:** {detail['db_id']}\n")
            else:
                summary_parts.append(f"Details for {len(successful_ingestions_details)} successfully ingested items are available in the logs.\n")
        elif processed_count > 0:  # Processed but no details (should not happen if logic is correct)
            summary_parts.append("Successfully processed items, but details are unavailable.\n")

        current_status_area.load_text("".join(summary_parts))
        current_status_area.display = True
        current_status_area.scroll_home(animate=False)

        notify_msg = f"{selected_media_type.title()} Ingestion: {processed_count} done, {error_count} errors."
        app.notify(notify_msg, severity="information" if error_count == 0 and processed_count > 0 else "warning", timeout=6)

    def on_worker_failure(error: Exception):
        try:
            current_loading_indicator = app.query_one(f"#tldw-api-loading-indicator-{selected_media_type}", LoadingIndicator)
            current_loading_indicator.display = False
            # current_submit_button = app.query_one(f"#tldw-api-submit-{selected_media_type}", Button)
            submit_button.disabled = False  # submit_button is already defined from event.button
        except QueryError as e_ui:
            logger.error(f"UI component not found in on_worker_failure for {selected_media_type}: {e_ui}")

        logger.error(f"TLDW API request worker failed for {selected_media_type}: {error}", exc_info=True)

        error_message_parts = [f"## API Request Failed! ({selected_media_type.title()})\n\n"]
        brief_notify_message = f"{selected_media_type.title()} API Request Failed."
        if isinstance(error, APIResponseError):
            error_type = "API Error"
            error_message_parts.append(f"**Type:** API Error\n**Status Code:** {error.status_code}\n**Message:** `{str(error)}`\n")
            if error.detail:
                error_message_parts.append(f"**Details:**\n```\n{error.detail}\n```\n")
            brief_notify_message = f"{selected_media_type.title()} API Error {error.status_code}: {str(error)[:50]}"
            if error.response_data:
                try:
                    # Try to pretty-print if it's JSON, otherwise just str
                    response_data_str = json.dumps(error.response_data, indent=2)
                except (TypeError, ValueError):
                    response_data_str = str(error.response_data)
                error_message_parts.append(f"**Response Data:**\n```json\n{response_data_str}\n```\n")
            brief_notify_message = f"API Error {error.status_code}: {str(error)[:100]}"
        elif isinstance(error, AuthenticationError):
            error_type = "Authentication Error"
            error_message_parts.append(f"**Type:** {error_type}\n")
            error_message_parts.append(f"**Message:** `{str(error)}`\n")
            brief_notify_message = f"Auth Error: {str(error)[:100]}"
        elif isinstance(error, APIConnectionError):
            error_type = "Connection Error"
            error_message_parts.append(f"**Type:** {error_type}\n")
            error_message_parts.append(f"**Message:** `{str(error)}`\n")
            brief_notify_message = f"Connection Error: {str(error)[:100]}"
        elif isinstance(error, APIRequestError):
            error_type = "API Request Error"
            error_message_parts.append(f"**Type:** {error_type}\n")
            error_message_parts.append(f"**Message:** `{str(error)}`\n")
            brief_notify_message = f"Request Error: {str(error)[:100]}"
        else:
            error_type = "General Error"
            error_message_parts.append(f"**Type:** {error_type}\n")
            error_message_parts.append(f"**Message:** `{str(error)}`\n")
            brief_notify_message = f"Processing failed: {str(error)[:100]}"

        try:
            current_status_area = app.query_one(f"#tldw-api-status-area-{selected_media_type}", TextArea)
            current_status_area.clear()
            current_status_area.load_text("".join(error_message_parts))
            current_status_area.display = True
            current_status_area.scroll_home(animate=False)
        except QueryError:
            logger.error(f"Could not find status_area for {selected_media_type} to display error.")
            app.notify(f"Critical: Status area for {selected_media_type} not found. Error: {brief_notify_message}", severity="error", timeout=10)
            return

        app.notify(brief_notify_message, severity="error", timeout=8)

    # STORE THE CONTEXT
    app._last_tldw_api_request_context = {
        "request_model": request_model,
        "overwrite_db": overwrite_db,
    }

    app.run_worker(
        process_media_worker,
        name=f"tldw_api_processing_{selected_media_type}",  # Unique worker name per tab
        group="api_calls",
        description=f"Processing {selected_media_type} media via TLDW API",
        exit_on_error=False
    )
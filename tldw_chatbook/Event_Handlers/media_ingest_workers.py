# tldw_chatbook/Event_Handlers/media_ingest_workers.py
#
# Worker event handlers for media ingestion
#
# Imports
from typing import TYPE_CHECKING, List

# 3rd-party Libraries
from loguru import logger
from textual.widgets import Button, TextArea, LoadingIndicator
from textual.css.query import QueryError
from textual.worker import Worker

# Local Imports
from ..tldw_api import (
    APIConnectionError, APIResponseError, 
    MediaItemProcessResult, ProcessedMediaWikiPage, BatchMediaProcessResponse,
    BatchProcessXMLResponse
)

if TYPE_CHECKING:
    from ..app import TldwCli

async def handle_tldw_api_worker_failure(app: 'TldwCli', event: 'Worker.StateChanged'):
    """Handles the failure of a TLDW API worker and updates the UI."""
    worker_name = event.worker.name or ""
    media_type = worker_name.replace("tldw_api_processing_", "")
    error = event.worker.error

    logger.error(f"TLDW API request worker failed for {media_type}: {error}", exc_info=True)

    try:
        loading_indicator = app.query_one(f"#tldw-api-loading-indicator-{media_type}", LoadingIndicator)
        submit_button = app.query_one(f"#tldw-api-submit-{media_type}", Button)
        status_area = app.query_one(f"#tldw-api-status-area-{media_type}", TextArea)

        loading_indicator.display = False
        submit_button.disabled = False
    except QueryError as e_ui:
        logger.error(f"UI component not found in on_worker_failure for {media_type}: {e_ui}")
        return

    error_message_parts = [f"## API Request Failed! ({media_type.title()})\n\n"]
    brief_notify_message = f"{media_type.title()} API Request Failed."

    # This logic is copied from your original local on_worker_failure function
    if isinstance(error, APIConnectionError):
        error_type = "Connection Error"
        error_message_parts.append(f"**Type:** {error_type}\n")
        error_message_parts.append(f"**Message:** `{str(error)}`\n")
        brief_notify_message = f"Connection Error: {str(error)[:100]}"
    elif isinstance(error, APIResponseError):
        error_type = "API Error"
        error_message_parts.append(f"**Type:** API Error\n**Status Code:** {error.status_code}\n**Message:** `{str(error)}`\n")
        brief_notify_message = f"API Error {error.status_code}: {str(error)[:100]}"
    # ... add other specific error types from your original function if needed ...
    else:
        error_type = "General Error"
        error_message_parts.append(f"**Type:** {type(error).__name__}\n")
        error_message_parts.append(f"**Message:** `{str(error)}`\n")
        brief_notify_message = f"Processing failed: {str(error)[:100]}"

    status_area.clear()
    status_area.load_text("".join(error_message_parts))
    status_area.display = True
    app.notify(brief_notify_message, severity="error", timeout=8)


async def handle_tldw_api_worker_success(app: 'TldwCli', event: 'Worker.StateChanged'):
    """Handles the success of a TLDW API worker and ingests the results."""
    worker_name = event.worker.name or ""
    media_type = worker_name.replace("tldw_api_processing_", "")
    response_data = event.worker.result

    logger.info(f"TLDW API worker for {media_type} succeeded. Processing results.")

    try:
        # Reset UI state (disable loading, enable button)
        app.query_one(f"#tldw-api-loading-indicator-{media_type}", LoadingIndicator).display = False
        app.query_one(f"#tldw-api-submit-{media_type}", Button).disabled = False
        status_area = app.query_one(f"#tldw-api-status-area-{media_type}", TextArea)
        status_area.clear()

    except QueryError as e_ui:
        logger.error(f"UI component not found in on_worker_success for {media_type}: {e_ui}")
        return
    except Exception as e:
        logger.error(f"Error resetting UI state in worker success handler: {e}", exc_info=True)
        return

    # --- Pre-flight Checks and Context Retrieval ---
    if not app.media_db:
        logger.error("Media_DB_v2 not initialized. Cannot ingest API results.")
        app.notify("Error: Local media database not available.", severity="error")
        status_area.load_text("## Error\n\nLocal media database is not available. Cannot save results.")
        return

    # Retrieve the context we saved before starting the worker
    request_context = getattr(app, "_last_tldw_api_request_context", {})
    request_model = request_context.get("request_model")
    overwrite_db = request_context.get("overwrite_db", False)

    if not request_model:
        logger.error("Could not retrieve request_model from app context. Cannot properly ingest results.")
        status_area.load_text("## Internal Error\n\nCould not retrieve original request context. Ingestion aborted.")
        return

    # --- Data Processing and Ingestion ---
    processed_count = 0
    error_count = 0
    successful_ingestions_details = []
    results_to_ingest: List[MediaItemProcessResult] = []

    # Normalize different response types into a single list of MediaItemProcessResult
    if isinstance(response_data, BatchMediaProcessResponse):
        results_to_ingest = response_data.results
    elif isinstance(response_data, list) and all(isinstance(item, ProcessedMediaWikiPage) for item in response_data):
        for mw_page in response_data:
            if mw_page.status == "Error":
                error_count += 1
                logger.error(f"MediaWiki page '{mw_page.title}' processing error: {mw_page.error_message}")
                continue
            # Adapt ProcessedMediaWikiPage to the common result structure
            results_to_ingest.append(MediaItemProcessResult(
                status="Success",
                input_ref=mw_page.input_ref or mw_page.title,
                processing_source=mw_page.title,
                media_type="mediawiki_page",
                metadata={"title": mw_page.title, "page_id": mw_page.page_id, "namespace": mw_page.namespace},
                content=mw_page.content,
                chunks=[{"text": chunk.get("text", ""), "metadata": chunk.get("metadata", {})} for chunk in mw_page.chunks] if mw_page.chunks else None,
            ))
    elif isinstance(response_data, BatchProcessXMLResponse):
        for xml_item in response_data.results:
            if xml_item.status == "Error":
                error_count += 1
                continue
            results_to_ingest.append(MediaItemProcessResult(
                status="Success", input_ref=xml_item.input_ref, media_type="xml",
                metadata={"title": xml_item.title, "author": xml_item.author, "keywords": xml_item.keywords},
                content=xml_item.content, analysis=xml_item.summary,
            ))
    else:
        logger.error(f"Unexpected TLDW API response data type for {media_type}: {type(response_data)}.")
        status_area.load_text(f"## API Request Processed\n\nUnexpected response format. Raw response logged.")
        app.notify("Error: Received unexpected data format from API.", severity="error")
        return

    # --- Ingestion Loop ---
    for item_result in results_to_ingest:
        if item_result.status == "Success":
            try:
                # Prepare chunks for database insertion
                unvectorized_chunks_to_save = []
                if item_result.chunks:
                    for chunk_item in item_result.chunks:
                        if isinstance(chunk_item, dict) and "text" in chunk_item:
                            unvectorized_chunks_to_save.append({
                                "text": chunk_item.get("text"), "metadata": chunk_item.get("metadata", {})
                            })
                        elif isinstance(chunk_item, str):
                            unvectorized_chunks_to_save.append({"text": chunk_item, "metadata": {}})

                # Call the DB function with data from both the API response and original request
                media_id, _, msg = app.media_db.add_media_with_keywords(
                    url=item_result.input_ref,
                    title=item_result.metadata.get("title", item_result.input_ref),
                    media_type=item_result.media_type,
                    content=item_result.content or item_result.transcript,
                    keywords=item_result.metadata.get("keywords", []) or request_model.keywords,
                    prompt=request_model.custom_prompt,
                    analysis_content=item_result.analysis or item_result.summary,
                    author=item_result.metadata.get("author") or request_model.author,
                    overwrite=overwrite_db,
                    chunks=unvectorized_chunks_to_save
                )

                if media_id:
                    logger.info(f"Successfully ingested '{item_result.input_ref}' into local DB. Media ID: {media_id}. Msg: {msg}")
                    processed_count += 1
                    successful_ingestions_details.append({
                        "input_ref": item_result.input_ref,
                        "title": item_result.metadata.get("title", "N/A"),
                        "media_type": item_result.media_type,
                        "db_id": media_id
                    })
                else:
                    logger.error(f"Failed to ingest '{item_result.input_ref}' into local DB. Message: {msg}")
                    error_count += 1

            except Exception as e_ingest:
                logger.error(f"Error ingesting item '{item_result.input_ref}' into local DB: {e_ingest}", exc_info=True)
                error_count += 1
        else:
            logger.error(f"API processing error for '{item_result.input_ref}': {item_result.error}")
            error_count += 1

    # --- Build and Display Summary ---
    summary_parts = [f"## TLDW API Request Successful ({media_type.title()})\n\n"]
    if not results_to_ingest and error_count == 0:
        summary_parts.append("API request successful, but no items were provided or found for processing.\n")
    else:
        summary_parts.append(f"- **Successfully Processed & Ingested:** {processed_count}\n")
        summary_parts.append(f"- **Errors (API or DB):** {error_count}\n\n")

    if error_count > 0:
        summary_parts.append("**Please check the application logs for details on any errors.**\n\n")

    if successful_ingestions_details:
        summary_parts.append("### Ingested Items:\n")
        for detail in successful_ingestions_details[:10]:  # Show max 10 details
            title_str = f" (Title: `{detail['title']}`)" if detail['title'] != 'N/A' else ""
            summary_parts.append(f"- **Input:** `{detail['input_ref']}`{title_str}\n")
            summary_parts.append(f"  - **Type:** {detail['media_type']}, **DB ID:** {detail['db_id']}\n")
        if len(successful_ingestions_details) > 10:
            summary_parts.append(f"\n...and {len(successful_ingestions_details) - 10} more items.")

    status_area.load_text("".join(summary_parts))
    status_area.display = True
    status_area.scroll_home(animate=False)

    notify_msg = f"{media_type.title()} Ingestion: {processed_count} done, {error_count} errors."
    app.notify(notify_msg, severity="information" if error_count == 0 else "warning", timeout=7)
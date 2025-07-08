# tldw_chatbook/Event_Handlers/prompt_ingest_events.py
#
# Prompt ingestion event handlers and related functions
#
# Imports
from pathlib import Path
from typing import TYPE_CHECKING, Optional, List, Any, Dict, Callable, Union

# 3rd-party Libraries
from loguru import logger
from textual.widgets import Button, Label, ListItem, ListView, Static, Markdown, TextArea, Select
from textual.css.query import QueryError
from textual.containers import VerticalScroll

# Local Imports
from ..Prompt_Management.Prompts_Interop import (
    parse_yaml_prompts_from_content, parse_json_prompts_from_content,
    parse_markdown_prompts_from_content, parse_txt_prompts_from_content,
    is_initialized as prompts_db_initialized,
    import_prompts_from_files, _get_file_type as _get_prompt_file_type
)
from ..Third_Party.textual_fspicker import FileOpen
from .ingest_utils import (
    PROMPT_FILE_FILTERS,
    MAX_PROMPT_PREVIEWS,
    _truncate_text
)

if TYPE_CHECKING:
    from ..app import TldwCli

# --- Prompt Preview Functions ---
async def _update_prompt_preview_display(app: 'TldwCli') -> None:
    """Updates the prompt preview area in the UI."""
    try:
        preview_area = app.query_one("#ingest-prompts-preview-area", VerticalScroll)
        await preview_area.remove_children()

        if not app.parsed_prompts_for_preview:
            await preview_area.mount(
                Static("Select files to see a preview, or no prompts found.", id="ingest-prompts-preview-placeholder"))
            return

        num_to_display = len(app.parsed_prompts_for_preview)
        prompts_to_show = app.parsed_prompts_for_preview[:MAX_PROMPT_PREVIEWS]

        for idx, prompt_data in enumerate(prompts_to_show):
            name = prompt_data.get("name", f"Unnamed Prompt {idx + 1}")
            author = prompt_data.get("author", "N/A")
            details = _truncate_text(prompt_data.get("details"), 150)
            system_prompt = _truncate_text(prompt_data.get("system_prompt"), 200)
            user_prompt = _truncate_text(prompt_data.get("user_prompt"), 200)
            keywords_list = prompt_data.get("keywords", [])
            keywords = ", ".join(keywords_list) if keywords_list else "N/A"

            md_content = f"""### {name}
**Author:** {author}
**Keywords:** {keywords}

**Details:**
```text
{details}
```

**System Prompt:**
```text
{system_prompt}
```

**User Prompt:**
```text
{user_prompt}
```
---
"""
            await preview_area.mount(Markdown(md_content, classes="prompt-preview-item"))

        if num_to_display > MAX_PROMPT_PREVIEWS:
            await preview_area.mount(
                Static(f"...and {num_to_display - MAX_PROMPT_PREVIEWS} more prompts loaded (not shown)."))

    except QueryError as e:
        logger.error(f"UI component not found for prompt preview update: {e}")
        app.notify("Error updating prompt preview UI.", severity="error")
    except Exception as e:
        logger.error(f"Unexpected error updating prompt preview: {e}", exc_info=True)
        app.notify("Unexpected error during preview update.", severity="error")


def _parse_single_prompt_file_for_preview(file_path: Path, app_ref: 'TldwCli') -> List[Dict[str, Any]]:
    """Parses a single prompt file and returns a list of prompt data dicts."""
    file_type = _get_prompt_file_type(file_path)  # Use helper from interop
    if not file_type:
        logger.warning(f"Unsupported file type for preview: {file_path}")
        return [{"name": f"Error: Unsupported type {file_path.name}",
                 "details": "File type not recognized for prompt import."}]

    parser_map: Dict[str, Callable[[str], List[Dict[str, Any]]]] = {
        "json": parse_json_prompts_from_content,
        "yaml": parse_yaml_prompts_from_content,
        "markdown": parse_markdown_prompts_from_content,
        "txt": parse_txt_prompts_from_content,
    }
    parser = parser_map.get(file_type)
    if not parser:
        logger.error(f"No parser found for file type {file_type} (preview)")
        return [
            {"name": f"Error: No parser for {file_path.name}", "details": f"File type '{file_type}' has no parser."}]

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        parsed = parser(content)
        if not parsed:  # If parser returns empty list (e.g. empty file or no valid prompts)
            logger.info(f"No prompts found in {file_path.name} by parser for preview.")
            # Not necessarily an error, could be an empty file.
            # Return an empty list, or a specific message if preferred.
            return []
        return parsed
    except RuntimeError as e:
        logger.error(f"Parser dependency missing for {file_path}: {e}")
        app_ref.notify(f"Cannot preview {file_path.name}: Required library missing ({e}).", severity="error", timeout=7)
        return [{"name": f"Error processing {file_path.name}", "details": str(e)}]
    except ValueError as e:
        logger.error(f"Failed to parse {file_path} for preview: {e}")
        app_ref.notify(f"Error parsing {file_path.name}: Invalid format.", severity="warning", timeout=7)
        return [{"name": f"Error parsing {file_path.name}", "details": str(e)}]
    except Exception as e:
        logger.error(f"Unexpected error reading/parsing {file_path} for preview: {e}", exc_info=True)
        app_ref.notify(f"Error reading {file_path.name}.", severity="error", timeout=7)
        return [{"name": f"Error reading {file_path.name}", "details": str(e)}]


async def _handle_prompt_file_selected_callback(app: 'TldwCli', selected_path: Optional[Path]) -> None:
    """
    Callback function executed after the FileOpen dialog for prompt selection returns.
    """
    if selected_path:
        logger.info(f"Prompt file selected via dialog: {selected_path}")
        if selected_path in app.selected_prompt_files_for_import:
            app.notify(f"File '{selected_path.name}' is already in the selection.", severity="warning")
            return

        app.selected_prompt_files_for_import.append(selected_path)
        app.last_prompt_import_dir = selected_path.parent

        try:
            list_view = app.query_one("#ingest-prompts-selected-files-list", ListView)

            placeholder_exists = False
            if list_view.children:  # Check if there are any children
                first_child = list_view.children[0]
                # Ensure the first child is a ListItem and it has children (the Label)
                if isinstance(first_child, ListItem) and first_child.children:
                    first_label_of_first_item = first_child.children[0]
                    if isinstance(first_label_of_first_item, Label):
                        # Convert Label's renderable (Rich Text) to plain string for comparison
                        if str(first_label_of_first_item.renderable).strip() == "No files selected.":
                            placeholder_exists = True

            if placeholder_exists:
                await list_view.clear()
                logger.debug("Cleared 'No files selected.' placeholder from prompt list.")

            await list_view.append(ListItem(Label(str(selected_path))))
            logger.debug(f"Appended '{selected_path}' to prompt list view.")

        except QueryError:
            logger.error("Could not find #ingest-prompts-selected-files-list ListView to update.")
        except Exception as e_lv:
            logger.error(f"Error updating prompt list view: {e_lv}", exc_info=True)

        # Parse this file and add to overall preview list
        parsed_prompts_from_file = _parse_single_prompt_file_for_preview(selected_path, app)
        app.parsed_prompts_for_preview.extend(parsed_prompts_from_file)

        await _update_prompt_preview_display(app)  # Update the preview display
    else:
        logger.info("Prompt file selection cancelled by user.")
        app.notify("File selection cancelled.")


# --- Prompt Ingest Handlers ---
async def handle_ingest_prompts_select_file_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    logger.debug("Select Prompt File(s) button pressed. Opening file dialog.")
    current_dir = app.last_prompt_import_dir or Path(".")
    await app.push_screen(
        FileOpen(
            location=str(current_dir),
            title="Select Prompt File (.md, .json, .yaml, .txt)",
            filters=PROMPT_FILE_FILTERS
        ),
        lambda path: app.call_after_refresh(lambda: _handle_prompt_file_selected_callback(app, path))
        # path type here is Optional[Path]
    )


async def handle_ingest_prompts_clear_files_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles the 'Clear Selection' button press for prompt import."""
    logger.info("Clearing selected prompt files and preview.")
    app.selected_prompt_files_for_import.clear()
    app.parsed_prompts_for_preview.clear()

    try:
        selected_list_view = app.query_one("#ingest-prompts-selected-files-list", ListView)
        await selected_list_view.clear()
        await selected_list_view.append(ListItem(Label("No files selected.")))

        preview_area = app.query_one("#ingest-prompts-preview-area", VerticalScroll)
        await preview_area.remove_children()
        await preview_area.mount(Static("Select files to see a preview.", id="ingest-prompts-preview-placeholder"))

        status_area = app.query_one("#prompt-import-status-area", TextArea)
        status_area.clear()
        app.notify("Selection and preview cleared.")
    except QueryError as e:
        logger.error(f"UI component not found for clearing prompt selection: {e}")
        app.notify("Error clearing UI.", severity="error")


async def handle_ingest_prompts_import_now_button_pressed(app: 'TldwCli', event: Button.Pressed) -> None:
    """Handles the 'Import Selected Files Now' button press."""
    logger.info("Import Selected Prompt Files Now button pressed.")

    if not app.selected_prompt_files_for_import:
        app.notify("No prompt files selected to import.", severity="warning")
        return

    if not prompts_db_initialized():
        msg = "Prompts database is not initialized. Cannot import."
        app.notify(msg, severity="error", timeout=7)
        logger.error(msg + " Aborting import.")
        return

    try:
        status_area = app.query_one("#prompt-import-status-area", TextArea)
        status_area.text = ""
        status_area.text = "Starting import process...\n"
    except QueryError:
        logger.error("Could not find #prompt-import-status-area TextArea.")
        app.notify("Status display area not found.", severity="error")
        return

    app.notify("Importing prompts... This may take a moment.")

    # Import these for worker-related functionality
    import tldw_chatbook.Event_Handlers.conv_char_events as ccp_handlers
    from .Chat_Events import chat_events as chat_handlers

    # The worker function itself remains the same
    async def import_worker_target():  # Renamed to avoid confusion with Worker class
        logger.info("--- import_worker_target (Prompts) RUNNING ---")
        try:
            results = import_prompts_from_files(app.selected_prompt_files_for_import)
            logger.info(f"--- import_worker_target (Prompts) FINISHED, results count: {len(results)} ---")
            return results  # Return the results
        except Exception as e_worker:
            logger.error(f"Exception inside import_worker_target (Prompts): {e_worker}", exc_info=True)
            # To signal an error to the worker system, you should re-raise the exception
            # or return a specific error indicator if you want to handle it differently
            # in on_worker_state_changed. For now, re-raising is simpler.
            raise e_worker

    # Define the functions that will handle success and failure,
    # these will be called by your app's on_worker_state_changed handler.
    # We pass the worker_name to identify which worker completed.

    def process_prompt_import_success(results: List[Dict[str, Any]], worker_name: str):
        if worker_name != "prompt_import_worker":  # Ensure this is for the correct worker
            return

        logger.info(f"--- process_prompt_import_success CALLED for worker: {worker_name} ---")
        logger.debug(f"Import results received: {results}")

        log_text_parts = ["Import process finished.\n\nResults:\n"]
        successful_imports = 0
        failed_imports = 0

        if not results:
            log_text_parts.append("No results returned from import worker.\n")
            logger.warning("process_prompt_import_success: Received empty results list.")
        else:
            for res_idx, res in enumerate(results):
                logger.debug(f"Processing result item {res_idx}: {res}")
                status = res.get("status", "unknown")
                file_path_str = res.get("file_path", "N/A")
                prompt_name = res.get("prompt_name", "N/A")
                message = res.get("message", "")

                log_text_parts.append(f"File: {Path(file_path_str).name}\n")
                if prompt_name and prompt_name != "N/A":
                    log_text_parts.append(f"  Prompt: '{prompt_name}'\n")
                log_text_parts.append(f"  Status: {status.upper()}\n")
                if message:
                    log_text_parts.append(f"  Message: {message}\n")
                log_text_parts.append("-" * 30 + "\n")

                if status == "success":
                    successful_imports += 1
                else:
                    failed_imports += 1

        summary = f"\nSummary: {successful_imports} prompts imported successfully, {failed_imports} failed."
        log_text_parts.append(summary)
        final_log_text_to_display = "".join(log_text_parts)
        logger.debug(f"Final text for status_area:\n{final_log_text_to_display}")

        try:
            status_area_cb = app.query_one("#prompt-import-status-area", TextArea)
            logger.info("Successfully queried #prompt-import-status-area in process_prompt_import_success.")
            status_area_cb.load_text(final_log_text_to_display)
            logger.info("Called load_text on #prompt-import-status-area.")
            status_area_cb.refresh(layout=True)
            logger.info("Called refresh() on status_area_cb.")
        except QueryError:
            logger.error("Failed to find #prompt-import-status-area in process_prompt_import_success.")
        except Exception as e_load_text:
            logger.error(f"Error during status_area_cb.load_text in process_prompt_import_success: {e_load_text}",
                         exc_info=True)

        app.notify(f"Prompt import finished. Success: {successful_imports}, Failed: {failed_imports}", timeout=8)
        logger.info(f"Prompt import summary: {summary.strip()}")

        app.call_later(ccp_handlers.populate_ccp_prompts_list_view, app)
        app.call_later(chat_handlers.handle_chat_sidebar_prompt_search_changed, app, "")
        logger.info("--- process_prompt_import_success FINISHED ---")

    def process_prompt_import_failure(error: Exception, worker_name: str):
        if worker_name != "prompt_import_worker":
            return

        logger.error(f"--- process_prompt_import_failure CALLED for worker {worker_name}: {error} ---", exc_info=True)
        try:
            status_area_cb_fail = app.query_one("#prompt-import-status-area", TextArea)
            current_text = status_area_cb_fail.text
            status_area_cb_fail.load_text(
                current_text + f"\nImport process failed critically: {error}\nCheck logs for details.\n")
        except QueryError:
            logger.error("Failed to find #prompt-import-status-area in process_prompt_import_failure.")
        app.notify(f"Prompt import failed: {str(error)[:100]}", severity="error", timeout=10)

    # Store these handlers on the app instance temporarily or pass them via a different mechanism
    # For simplicity here, we'll assume app.py's on_worker_state_changed will call them.
    # A more robust way is to make these methods of a class or use a dispatch dictionary in app.py.
    app.prompt_import_success_handler = process_prompt_import_success
    app.prompt_import_failure_handler = process_prompt_import_failure

    # Run the worker
    app.run_worker(
        import_worker_target,  # The async callable
        name="prompt_import_worker",  # Crucial for identifying the worker later
        group="file_operations",
        description="Importing selected prompt files."
        # No on_success or on_failure here
    )
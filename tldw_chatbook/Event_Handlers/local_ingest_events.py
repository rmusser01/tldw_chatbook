# tldw_chatbook/Event_Handlers/local_ingest_events.py
#
# Local file processing event handlers
#
# Imports
from typing import TYPE_CHECKING

# 3rd-party Libraries
from loguru import logger
from textual.css.query import QueryError

# Local Imports
from ..UI.Ingest_Window import IngestWindow

if TYPE_CHECKING:
    from ..app import TldwCli

async def handle_ingest_local_web_button_pressed(app: 'TldwCli', event: 'Button.Pressed') -> None:
    """Handle local web article button presses by delegating to IngestWindow methods."""
    button_id = event.button.id
    try:
        ingest_window = app.query_one("#ingest-window", IngestWindow)
        
        if button_id == "ingest-local-web-clear-urls":
            await ingest_window._handle_clear_urls()
        elif button_id == "ingest-local-web-import-urls":
            await ingest_window._handle_import_urls_from_file()
        elif button_id == "ingest-local-web-remove-duplicates":
            await ingest_window._handle_remove_duplicate_urls()
        elif button_id == "ingest-local-web-process":
            await ingest_window.handle_local_web_article_process()
        elif button_id == "ingest-local-web-stop":
            await ingest_window._handle_stop_web_scraping()
        elif button_id == "ingest-local-web-retry":
            await ingest_window._handle_retry_failed_urls()
    except QueryError:
        logger.error(f"Could not find IngestWindow to handle button {button_id}")
    except Exception as e:
        logger.error(f"Error handling local web button {button_id}: {e}")
        app.notify(f"Error: {str(e)}", severity="error")

async def handle_local_pdf_ebook_submit_button_pressed(app: 'TldwCli', event: 'Button.Pressed') -> None:
    """Handle local PDF and ebook processing button presses."""
    button_id = event.button.id
    media_type = "pdf" if button_id == "local-submit-pdf" else "ebook"
    
    try:
        ingest_window = app.query_one("#ingest-window", IngestWindow)
        
        if media_type == "pdf":
            await ingest_window.handle_local_pdf_process()
        else:  # ebook
            await ingest_window.handle_local_ebook_process()
            
    except QueryError:
        logger.error(f"Could not find IngestWindow to handle button {button_id}")
    except Exception as e:
        logger.error(f"Error handling local {media_type} button: {e}")
        app.notify(f"Error: {str(e)}", severity="error")

async def handle_local_audio_video_submit_button_pressed(app: 'TldwCli', event: 'Button.Pressed') -> None:
    """Handle local audio and video processing button presses."""
    button_id = event.button.id
    media_type = "audio" if button_id == "local-submit-audio" else "video"
    
    try:
        ingest_window = app.query_one("#ingest-window", IngestWindow)
        
        if media_type == "audio":
            await ingest_window.handle_local_audio_process()
        else:  # video
            await ingest_window.handle_local_video_process()
            
    except QueryError:
        logger.error(f"Could not find IngestWindow to handle button {button_id}")
    except Exception as e:
        logger.error(f"Error handling local {media_type} button: {e}")
        app.notify(f"Error: {str(e)}", severity="error")

async def handle_local_document_submit_button_pressed(app: 'TldwCli', event: 'Button.Pressed') -> None:
    """Handle local document processing button presses."""
    try:
        ingest_window = app.query_one("#ingest-window", IngestWindow)
        await ingest_window.handle_local_document_process()
            
    except QueryError:
        logger.error(f"Could not find IngestWindow to handle document button")
    except Exception as e:
        logger.error(f"Error handling local document button: {e}")
        app.notify(f"Error: {str(e)}", severity="error")

async def handle_local_plaintext_submit_button_pressed(app: 'TldwCli', event: 'Button.Pressed') -> None:
    """Handle local plaintext processing button presses."""
    try:
        ingest_window = app.query_one("#ingest-window", IngestWindow)
        await ingest_window.handle_local_plaintext_process()
            
    except QueryError:
        logger.error(f"Could not find IngestWindow to handle plaintext button")
    except Exception as e:
        logger.error(f"Error handling local plaintext button: {e}")
        app.notify(f"Error: {str(e)}", severity="error")
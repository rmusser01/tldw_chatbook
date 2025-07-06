# Logging_Config.py
# Description: Configuration for logging
#
# Imports
import asyncio
import logging
import sys
import traceback
#
# 3rd-Party Imports
from loguru import logger as loguru_logger
from textual.app import App
from textual.css.query import QueryError
from textual.logging import TextualHandler
from textual.widgets import RichLog
#
# Local Imports
from tldw_chatbook.config import get_cli_log_file_path, get_cli_setting
#
########################################################################################################################
#
# Functions:



# --- Custom Logging Handler ---
class RichLogHandler(logging.Handler):
    def __init__(self, rich_log_widget: RichLog):
        super().__init__()
        self.rich_log_widget = rich_log_widget
        self.log_queue = asyncio.Queue()
        self.formatter = logging.Formatter(
            "{asctime} [{levelname:<8}] {name}:{lineno:<4} : {message}",
            style="{", datefmt="%Y-%m-%d %H:%M:%S"
        )
        self.setFormatter(self.formatter)
        self._queue_processor_task = None

    def start_processor(self, app: App):  # Keep 'app' param for context if needed elsewhere, but don't use for run_task
        """Starts the log queue processing task using the widget's run_task."""
        if not self._queue_processor_task or self._queue_processor_task.done():
            try:
                # Get the currently running event loop
                loop = asyncio.get_running_loop()
                # Create the task using the standard asyncio function
                self._queue_processor_task = loop.create_task(
                    self._process_log_queue(),
                    name="RichLogProcessor"
                )
                logging.debug("RichLog queue processor task started via asyncio.create_task.")
            except RuntimeError as e:
                # Handle cases where the loop might not be running (shouldn't happen if called from on_mount)
                logging.error(f"Failed to get running loop to start log processor: {e}")
            except Exception as e:
                logging.error(f"Failed to start log processor task: {e}", exc_info=True)

    async def stop_processor(self):
        """Signals the queue processor task to stop and waits for it."""
        # This cancellation logic works for tasks created with asyncio.create_task
        if self._queue_processor_task and not self._queue_processor_task.done():
            logging.debug("Attempting to stop RichLog queue processor task...")
            self._queue_processor_task.cancel()
            try:
                # Wait for the task to acknowledge cancellation
                await self._queue_processor_task
            except asyncio.CancelledError:
                logging.debug("RichLog queue processor task cancelled successfully.")
            except Exception as e:
                # Log errors during cancellation itself
                logging.error(f"Error occurred while awaiting cancelled log processor task: {e}", exc_info=True)
            finally:
                self._queue_processor_task = None  # Ensure it's cleared

    async def _process_log_queue(self):
        """Coroutine to process logs from the queue and write to the widget."""
        while True:
            try:
                message = await self.log_queue.get()
                if self.rich_log_widget.is_mounted and self.rich_log_widget.app:
                    self.rich_log_widget.write(message)
                self.log_queue.task_done()
            except asyncio.CancelledError:
                logging.debug("RichLog queue processor task received cancellation.")
                # Process any remaining items? Might be risky if app is shutting down.
                # while not self.log_queue.empty():
                #    try: message = self.log_queue.get_nowait(); # process...
                #    except asyncio.QueueEmpty: break
                break  # Exit the loop on cancellation
            except Exception as e:
                print(f"!!! CRITICAL ERROR in RichLog processor: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                # Avoid continuous loop on error, maybe sleep?
                await asyncio.sleep(1)

    def emit(self, record: logging.LogRecord):
        """Format the record and put it onto the async queue."""
        try:
            message = self.format(record)
            # Use call_soon_threadsafe if emit might be called from non-asyncio threads (workers)
            # For workers started with thread=True, this is necessary.
            try:
                # Try to get the current event loop
                loop = asyncio.get_running_loop()
                # Use call_soon_threadsafe to add to queue from any thread
                loop.call_soon_threadsafe(self.log_queue.put_nowait, message)
            except RuntimeError:
                # No event loop running, fallback to direct logging for warnings and above
                if record.levelno >= logging.WARNING: 
                    print(f"LOG_FALLBACK: {message}", file=sys.stderr)
            except Exception as e:
                # Don't re-raise to avoid breaking the logging system
                if record.levelno >= logging.WARNING:
                    print(f"LOG_FALLBACK: {message} (Error: {e})", file=sys.stderr)
        except Exception:
            # Last resort - print to stderr to avoid losing critical messages
            print(f"!!!!!!!! ERROR within RichLogHandler.emit !!!!!!!!!!", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


def configure_application_logging(app_instance):
    """Sets up all logging handlers, including Loguru integration."""
    # FIXME - LOGGING MAY BRING BACK BLINKING
    temp_handler = logging.StreamHandler(sys.stdout)
    temp_handler.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(temp_handler)
    # This first logging.info will go to the stderr handler from the initial basicConfig
    logging.info("--- _setup_logging START (from Logging_Config.py) ---")
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # --- BEGIN LOGURU MANAGEMENT (Your existing code is mostly fine here) ---
    try:
        loguru_logger.remove()  # Good: removes Loguru's default stderr sink
        logging.info("Loguru: All pre-existing sinks removed.")

        def sink_to_standard_logging(message):
            # ... (your existing sink_to_standard_logging function)
            record = message.record
            level_mapping = {
                "TRACE": logging.DEBUG, "DEBUG": logging.DEBUG, "INFO": logging.INFO,
                "SUCCESS": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR,
                "CRITICAL": logging.CRITICAL,
            }
            std_level = level_mapping.get(record["level"].name, logging.INFO)
            std_logger = logging.getLogger(record["name"])
            if record["exception"]:
                std_logger.log(std_level, record["message"], exc_info=record["exception"])
            else:
                std_logger.log(std_level, record["message"])

        loguru_logger.add(
            sink_to_standard_logging,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            level="TRACE"
        )
        # This log message will also currently go to the initial basicConfig stderr handler
        logging.info("Loguru: Configured to forward its messages to standard Python logging system.")
    except Exception as e:
        # This log message will also currently go to the initial basicConfig stderr handler
        logging.error(f"Loguru: Error during Loguru reconfiguration: {e}", exc_info=True)
    # --- END LOGURU MANAGEMENT ---

    # --- CONFIGURE STANDARD PYTHON LOGGING ROOT LOGGER ---
    root_logger = logging.getLogger()

    # !!! IMPORTANT FIX: Remove all existing handlers from the root logger !!!
    # This will get rid of the StreamHandler (to stderr) added by the initial
    # global logging.basicConfig() call.
    initial_handlers_removed_count = 0
    for handler in root_logger.handlers[:]:  # Iterate over a copy
        root_logger.removeHandler(handler)
        if hasattr(handler, 'close') and callable(handler.close):
            try:
                handler.close()
            except Exception:
                pass  # Ignore errors during close of old handlers
        initial_handlers_removed_count += 1

    # Log this removal using Loguru, as standard logging has no handlers yet.
    # This message will go to Loguru's sink (which forwards to std logging,
    # but std logging has no handlers yet, so it might hit Python's "last resort" stderr).
    # Or, better, print to stderr just for this one-off setup message if needed, then rely on proper handlers.
    if initial_handlers_removed_count > 0:
        # Using print here because logging state is actively being changed.
        # This should be one of the last messages to hit raw stderr if setup is correct.
        print(
            f"INFO: _setup_logging: Removed {initial_handlers_removed_count} pre-existing handler(s) from root logger.",
            file=sys.stderr)

    # Now that root_logger is clean, set its overall level.
    # This level acts as a filter before messages reach any of its handlers.
    initial_log_level_str = app_instance.app_config.get("general", {}).get("log_level", "INFO").upper()
    initial_log_level = getattr(logging, initial_log_level_str, logging.INFO)
    root_logger.setLevel(initial_log_level)
    # (A temporary print to confirm, as logging to root_logger now might go to "last resort" until a handler is added)
    print(f"INFO: _setup_logging: Root logger level set to {logging.getLevelName(root_logger.level)}",
          file=sys.stderr)

    # --- Add TextualHandler (to standard logging) ---
    # (Your existing TextualHandler setup code is fine)
    # Ensure it's added AFTER clearing old handlers and setting root level.
    # ...
    has_textual_handler = any(isinstance(h, TextualHandler) for h in root_logger.handlers)
    if not has_textual_handler:
        textual_console_handler = TextualHandler()
        textual_console_handler.setLevel(initial_log_level)  # Respects app_config
        console_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)-8s] %(name)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        textual_console_handler.setFormatter(console_formatter)
        root_logger.addHandler(textual_console_handler)
        # Now, logging.info should go to Textual's dev console (and other handlers added below)
        logging.info(
            f"Standard Logging: Added TextualHandler (Level: {logging.getLevelName(textual_console_handler.level)}).")
    else:
        logging.info("Standard Logging: TextualHandler already exists.")

    # Test Loguru message again. It should now go to TextualHandler (and others).
    loguru_logger.info(
        "Loguru Test: This message from Loguru should now appear in Textual dev console (and other configured handlers).")

    # --- Setup RichLog Handler (to standard logging) ---
    # (Your existing RichLogHandler setup code is fine, ensure it's added AFTER clearing)
    # ...
    try:
        log_display_widget = app_instance.query_one("#app-log-display", RichLog)
        # Check if it's already added by a previous call (should not happen if _setup_logging is called once)
        if not any(isinstance(h, RichLogHandler) and h.rich_log_widget is log_display_widget for h in
                   root_logger.handlers):
            if not app_instance._rich_log_handler:  # Create if it doesn't exist
                app_instance._rich_log_handler = RichLogHandler(log_display_widget)
            # Configure and add
            rich_log_handler_level_str = app_instance.app_config.get("logging", {}).get("rich_log_level", "DEBUG").upper()
            rich_log_handler_level = getattr(logging, rich_log_handler_level_str, logging.DEBUG)
            app_instance._rich_log_handler.setLevel(rich_log_handler_level)
            root_logger.addHandler(app_instance._rich_log_handler)
            logging.info(
                f"Standard Logging: Added RichLogHandler (Level: {logging.getLevelName(app_instance._rich_log_handler.level)}).")
        else:
            logging.info("Standard Logging: RichLogHandler already exists and is added.")
    except QueryError:
        logging.error("!!! ERROR: Failed to find #app-log-display widget for RichLogHandler setup.")
        app_instance._rich_log_handler = None
    except Exception as e:
        logging.error(f"!!! ERROR setting up RichLogHandler: {e}", exc_info=True)
        app_instance._rich_log_handler = None

    # --- Setup File Logging (to standard logging) ---
    # (Your existing FileHandler setup code is fine, ensure it's added AFTER clearing)
    # ... (your existing code to add file_handler to root_logger) ...
    try:
        log_file_path = get_cli_log_file_path()
        log_dir = log_file_path.parent
        log_dir.mkdir(parents=True, exist_ok=True)

        has_file_handler = any(
            isinstance(h, logging.handlers.RotatingFileHandler) and h.baseFilename == str(log_file_path) for h in
            root_logger.handlers)

        if not has_file_handler:
            max_bytes_default = 10485760
            backup_count_default = 5
            file_log_level_default_str = "INFO"
            max_bytes = int(get_cli_setting("logging", "log_max_bytes", max_bytes_default))
            backup_count = int(get_cli_setting("logging", "log_backup_count", backup_count_default))
            file_log_level_str = get_cli_setting("logging", "file_log_level", file_log_level_default_str).upper()
            file_log_level = getattr(logging, file_log_level_str, logging.INFO)

            file_handler = logging.handlers.RotatingFileHandler(
                log_file_path, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8'
            )
            file_handler.setLevel(file_log_level)
            file_formatter = logging.Formatter(
                "%(asctime)s [%(levelname)-8s] %(name)s:%(lineno)d - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
            logging.info(
                f"Standard Logging: Added RotatingFileHandler (File: '{log_file_path}', Level: {logging.getLevelName(file_log_level)}).")
        else:
            logging.info("Standard Logging: RotatingFileHandler already exists for this file path.")
    except Exception as e:
        logging.warning(f"!!! ERROR setting up file logging: {e}", exc_info=True)

    # Re-evaluate lowest level for standard logging root logger
    # (Your existing logic for this is fine)
    all_std_handlers = root_logger.handlers
    if all_std_handlers:
        handler_levels = [h.level for h in all_std_handlers if h.level > 0]
        if handler_levels:
            lowest_effective_level = min(handler_levels)
            current_root_level = root_logger.level
            # Only adjust root logger level if it's currently *less* verbose (higher numeric value)
            # than the most verbose handler.
            if current_root_level > lowest_effective_level:
                logging.info(
                    f"Standard Logging: Adjusting root logger level from {logging.getLevelName(current_root_level)} to {logging.getLevelName(lowest_effective_level)} to match most verbose handler.")
                root_logger.setLevel(lowest_effective_level)
        logging.info(f"Standard Logging: Final Root logger level is: {logging.getLevelName(root_logger.level)}")
    else:
        logging.warning("Standard Logging: No handlers found on root logger after setup!")

    logging.info("Logging setup complete.")
    logging.info("--- _setup_logging END ---")




#
# End of Logging_Config.py
########################################################################################################################

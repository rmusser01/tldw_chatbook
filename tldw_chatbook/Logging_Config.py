# Logging_Config.py
# Description: Configuration for logging
#
# Imports
import asyncio
import logging
import sys
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path

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
from tldw_chatbook.Utils.private_paths import (
    PrivatePathError,
    lexical_path,
    open_private_binary,
    open_private_text_append_stream,
    secure_private_directory,
)
from tldw_chatbook.Utils.persistent_diagnostics import (
    PersistentDiagnosticFilter,
    persist_event,
)
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
            style="{",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.setFormatter(self.formatter)
        self._queue_processor_task = None
        self._closed = False

    def start_processor(
        self, app: App
    ):  # Keep 'app' param for context if needed elsewhere, but don't use for run_task
        """Starts the log queue processing task using the widget's run_task."""
        if not self._queue_processor_task or self._queue_processor_task.done():
            try:
                # Get the currently running event loop
                loop = asyncio.get_running_loop()
                # Check if the loop is closed before creating task
                if loop.is_closed():
                    logging.warning(
                        "Cannot start RichLog processor: event loop is closed"
                    )
                    return

                # Create the task using the standard asyncio function
                self._queue_processor_task = loop.create_task(
                    self._process_log_queue(), name="RichLogProcessor"
                )
                logging.debug(
                    "RichLog queue processor task started via asyncio.create_task."
                )
            except RuntimeError as e:
                # Handle cases where the loop might not be running (shouldn't happen if called from on_mount)
                logging.error(f"Failed to get running loop to start log processor: {e}")
            except Exception as e:
                logging.error(f"Failed to start log processor task: {e}", exc_info=True)

    async def stop_processor(self):
        """Signals the queue processor task to stop and waits for it."""
        self._closed = True
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
                logging.error(
                    f"Error occurred while awaiting cancelled log processor task: {e}",
                    exc_info=True,
                )
            finally:
                self._queue_processor_task = None  # Ensure it's cleared

    def close(self):
        """Close the handler and mark it as closed."""
        self._closed = True
        super().close()

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
                # Process any remaining items in the queue before exiting
                try:
                    while not self.log_queue.empty():
                        try:
                            message = self.log_queue.get_nowait()
                            if (
                                self.rich_log_widget.is_mounted
                                and self.rich_log_widget.app
                            ):
                                self.rich_log_widget.write(message)
                            self.log_queue.task_done()
                        except asyncio.QueueEmpty:
                            break
                except Exception:
                    pass  # Ignore errors during cleanup
                break  # Exit the loop on cancellation
            except RuntimeError as e:
                if "Event loop is closed" in str(e):
                    # Event loop was closed, exit gracefully
                    logging.debug("RichLog processor exiting due to closed event loop")
                    break
                else:
                    print(
                        f"!!! RUNTIME ERROR in RichLog processor: {e}", file=sys.stderr
                    )
                    traceback.print_exc(file=sys.stderr)
                    await asyncio.sleep(1)
            except Exception as e:
                print(f"!!! CRITICAL ERROR in RichLog processor: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                # Avoid continuous loop on error, maybe sleep?
                try:
                    await asyncio.sleep(1)
                except Exception:
                    # If we can't even sleep, the loop is probably closed
                    break

    def emit(self, record: logging.LogRecord):
        """Format the record and put it onto the async queue."""
        try:
            # Check if the processor task is still running
            if self._queue_processor_task and self._queue_processor_task.done():
                # Task is done, don't try to emit
                return

            message = self.format(record)
            # Use call_soon_threadsafe if emit might be called from non-asyncio threads (workers)
            # For workers started with thread=True, this is necessary.
            try:
                # Try to get the current event loop
                loop = asyncio.get_running_loop()
                # Check if the loop is closed before trying to use it
                if not loop.is_closed():
                    # Use call_soon_threadsafe to add to queue from any thread
                    loop.call_soon_threadsafe(self.log_queue.put_nowait, message)
                elif record.levelno >= logging.WARNING:
                    print(f"LOG_FALLBACK (loop closed): {message}", file=sys.stderr)
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
            print(
                "!!!!!!!! ERROR within RichLogHandler.emit !!!!!!!!!!", file=sys.stderr
            )
            traceback.print_exc(file=sys.stderr)


class PrivateRotatingFileHandler(RotatingFileHandler):
    """A rotating handler whose files stay inside the private-path boundary."""

    def __init__(
        self,
        filename: str | Path,
        mode: str = "a",
        maxBytes: int = 0,
        backupCount: int = 0,
        encoding: str | None = None,
        delay: bool = False,
        errors: str | None = None,
    ) -> None:
        selected = lexical_path(filename)
        self._private_parent = selected.parent
        self._configured_backup_count = backupCount
        secure_private_directory(
            self._private_parent,
            create=True,
            application_owned=True,
        )
        self._harden_existing_generations(selected)
        super().__init__(
            selected,
            mode=mode,
            maxBytes=maxBytes,
            backupCount=backupCount,
            encoding=encoding,
            delay=delay,
            errors=errors,
        )
        try:
            self._harden_existing_generations(selected)
        except BaseException:
            self.close()
            raise

    def _generation_paths(self, active: Path) -> list[Path]:
        return [
            active,
            *(
                active.with_name(f"{active.name}.{index}")
                for index in range(1, self._configured_backup_count + 1)
            ),
        ]

    def _harden_existing_generations(self, active: Path | None = None) -> None:
        selected = active or lexical_path(self.baseFilename)
        for generation in self._generation_paths(selected):
            try:
                generation.lstat()
            except FileNotFoundError:
                continue
            with open_private_binary(generation):
                pass

    def _open(self):
        self._harden_existing_generations()
        return open_private_text_append_stream(
            self.baseFilename,
            application_owned_directory=self._private_parent,
            encoding=self.encoding or "utf-8",
            errors=self.errors,
        )

    def doRollover(self) -> None:
        """Rotate only after every existing generation passes private checks."""

        self._harden_existing_generations()
        super().doRollover()
        self._harden_existing_generations()


def _configure_private_file_logging(root_logger: logging.Logger) -> bool:
    """Install the private file sink, leaving existing handlers on failure."""

    try:
        log_file_path = get_cli_log_file_path()
        existing_handler = next(
            (
                handler
                for handler in root_logger.handlers
                if isinstance(handler, PrivateRotatingFileHandler)
                and handler.baseFilename == str(log_file_path)
            ),
            None,
        )
        if existing_handler is not None:
            if not any(
                isinstance(item, PersistentDiagnosticFilter)
                for item in existing_handler.filters
            ):
                existing_handler.addFilter(PersistentDiagnosticFilter())
            root_logger.info("Private rotating file logging is already installed.")
            return True

        max_bytes = int(get_cli_setting("logging", "log_max_bytes", 10485760))
        backup_count = int(get_cli_setting("logging", "log_backup_count", 5))
        file_log_level_name = str(
            get_cli_setting("logging", "file_log_level", "INFO")
        ).upper()
        file_log_level = getattr(logging, file_log_level_name, logging.INFO)
        file_handler = PrivateRotatingFileHandler(
            log_file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(file_log_level)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)-8s] %(name)s:%(lineno)d - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        file_handler.addFilter(PersistentDiagnosticFilter())
        root_logger.addHandler(file_handler)
        root_logger.info(
            "Private rotating file logging installed at level %s.",
            logging.getLevelName(file_log_level),
        )
        installed_level = file_log_level
    except PrivatePathError as exc:
        root_logger.warning(
            "File logging disabled: unsafe persistent target (%s).",
            exc.result.status.value,
        )
        return False
    except Exception as exc:
        root_logger.warning(
            "File logging disabled: persistent sink setup failed (%s).",
            type(exc).__name__,
        )
        return False

    # TASK-1240. Written the moment the sink is live, so an empty file means
    # "the sink did not install" rather than "nothing has happened yet". This
    # function swallows install failures (it warns and returns False), so
    # without this line those two states are indistinguishable.
    #
    # Emitted OUTSIDE the try above (M7): that try's handlers report "install
    # failed" and return False, so a future failure originating in this call --
    # after the handler is built, filtered and attached -- would misreport a
    # working sink as a broken one.
    #
    # Emitted above BOTH gates in front of it, not at INFO (I3). A record must
    # clear the *logger's* effective level before `logging` will even build it,
    # and then the *handler's* level. The two fail at opposite ends of the
    # configured range, and the install line has to survive both:
    #
    #   - Handler gate. `file_log_level` is user-configurable and the shipped
    #     `config.py` comment offers WARNING/ERROR/CRITICAL. At any of those an
    #     INFO install line is dropped by the very handler it is meant to prove
    #     installed.
    #   - Logger gate. `configure_application_logging` only lowers the root
    #     logger to match the most verbose handler *after* calling this
    #     function, so at this moment root still sits at `general.log_level`.
    #     With `file_log_level = "DEBUG"` and `general.log_level = "INFO"`, a
    #     DEBUG install line is discarded by the root logger before the handler
    #     ever sees it.
    #
    # Either way the user sends a zero-byte log -- which the paragraph above
    # tells a maintainer to read as "the sink did not install". `max()` of the
    # two clears whichever is higher, so the line survives every combination.
    #
    # Corollary, worth stating because it looks like a bug otherwise: the
    # *other* events in this design are still level-gated normally, so under
    # `file_log_level = "WARNING"` a log containing only this line means
    # "installed, and everything below WARNING was filtered" -- not "nothing
    # happened".
    emit_level = max(installed_level, root_logger.getEffectiveLevel())
    try:
        persist_event(
            "logging",
            "persistent_sink_installed",
            level=emit_level,
            status="ok",
        )
    except Exception:
        # Diagnostics must never be the reason the sink reports failure.
        pass
    return True


def _forward_loguru_to_standard(message) -> None:
    """Forward a Loguru record while preserving its original ownership.

    The source path is attached for the persistent handler's admission filter.
    Non-persistent handlers continue to receive the original message and
    exception details.
    """

    record = message.record
    level_mapping = {
        "TRACE": logging.DEBUG,
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "SUCCESS": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    std_level = level_mapping.get(record["level"].name, logging.INFO)
    std_logger = logging.getLogger(record["name"])
    extra = {"_tldw_source_path": str(record["file"].path)}
    if record["exception"]:
        std_logger.log(
            std_level,
            record["message"],
            exc_info=record["exception"],
            extra=extra,
        )
    else:
        std_logger.log(std_level, record["message"], extra=extra)


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

        loguru_logger.add(
            _forward_loguru_to_standard,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            level="TRACE",
        )
        # This log message will also currently go to the initial basicConfig stderr handler
        logging.info(
            "Loguru: Configured to forward its messages to standard Python logging system."
        )
    except Exception as e:
        # This log message will also currently go to the initial basicConfig stderr handler
        logging.error(
            f"Loguru: Error during Loguru reconfiguration: {e}", exc_info=True
        )
    # --- END LOGURU MANAGEMENT ---

    # --- CONFIGURE STANDARD PYTHON LOGGING ROOT LOGGER ---
    root_logger = logging.getLogger()

    # !!! IMPORTANT FIX: Remove all existing handlers from the root logger !!!
    # This will get rid of the StreamHandler (to stderr) added by the initial
    # global logging.basicConfig() call.
    initial_handlers_removed_count = 0
    for handler in root_logger.handlers[:]:  # Iterate over a copy
        root_logger.removeHandler(handler)
        if hasattr(handler, "close") and callable(handler.close):
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
            file=sys.stderr,
        )

    # Now that root_logger is clean, set its overall level.
    # This level acts as a filter before messages reach any of its handlers.
    initial_log_level_str = (
        app_instance.app_config.get("general", {}).get("log_level", "INFO").upper()
    )
    initial_log_level = getattr(logging, initial_log_level_str, logging.INFO)
    root_logger.setLevel(initial_log_level)
    # (A temporary print to confirm, as logging to root_logger now might go to "last resort" until a handler is added)
    print(
        f"INFO: _setup_logging: Root logger level set to {logging.getLevelName(root_logger.level)}",
        file=sys.stderr,
    )

    # --- Add TextualHandler (to standard logging) ---
    # (Your existing TextualHandler setup code is fine)
    # Ensure it's added AFTER clearing old handlers and setting root level.
    # ...
    has_textual_handler = any(
        isinstance(h, TextualHandler) for h in root_logger.handlers
    )
    if not has_textual_handler:
        textual_console_handler = TextualHandler()
        textual_console_handler.setLevel(initial_log_level)  # Respects app_config
        console_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)-8s] %(name)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        textual_console_handler.setFormatter(console_formatter)
        root_logger.addHandler(textual_console_handler)
        # Now, logging.info should go to Textual's dev console (and other handlers added below)
        logging.info(
            f"Standard Logging: Added TextualHandler (Level: {logging.getLevelName(textual_console_handler.level)})."
        )
    else:
        logging.info("Standard Logging: TextualHandler already exists.")

    # Test Loguru message again. It should now go to TextualHandler (and others).
    loguru_logger.info(
        "Loguru Test: This message from Loguru should now appear in Textual dev console (and other configured handlers)."
    )

    # --- Setup RichLog Handler (to standard logging) ---
    # (Your existing RichLogHandler setup code is fine, ensure it's added AFTER clearing)
    # ...
    try:
        log_display_widget = app_instance.query_one("#app-log-display", RichLog)
        # Check if it's already added by a previous call (should not happen if _setup_logging is called once)
        if not any(
            isinstance(h, RichLogHandler) and h.rich_log_widget is log_display_widget
            for h in root_logger.handlers
        ):
            if not app_instance._rich_log_handler:  # Create if it doesn't exist
                app_instance._rich_log_handler = RichLogHandler(log_display_widget)
            # Configure and add
            rich_log_handler_level_str = (
                app_instance.app_config.get("logging", {})
                .get("rich_log_level", "DEBUG")
                .upper()
            )
            rich_log_handler_level = getattr(
                logging, rich_log_handler_level_str, logging.DEBUG
            )
            app_instance._rich_log_handler.setLevel(rich_log_handler_level)
            root_logger.addHandler(app_instance._rich_log_handler)
            logging.info(
                f"Standard Logging: Added RichLogHandler (Level: {logging.getLevelName(app_instance._rich_log_handler.level)})."
            )
        else:
            logging.info(
                "Standard Logging: RichLogHandler already exists and is added."
            )
    except QueryError:
        # The legacy Logs window widget (#app-log-display) does not exist in the
        # master-shell UI, so skipping the RichLogHandler here is expected on every boot.
        logging.debug(
            "RichLogHandler setup skipped: #app-log-display widget not present (legacy Logs window)."
        )
        app_instance._rich_log_handler = None
    except Exception as e:
        logging.error(f"!!! ERROR setting up RichLogHandler: {e}", exc_info=True)
        app_instance._rich_log_handler = None

    # File logging is isolated so an unsafe target cannot remove terminal/UI sinks.
    _configure_private_file_logging(root_logger)

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
                    f"Standard Logging: Adjusting root logger level from {logging.getLevelName(current_root_level)} to {logging.getLevelName(lowest_effective_level)} to match most verbose handler."
                )
                root_logger.setLevel(lowest_effective_level)
        logging.info(
            f"Standard Logging: Final Root logger level is: {logging.getLevelName(root_logger.level)}"
        )
    else:
        logging.warning(
            "Standard Logging: No handlers found on root logger after setup!"
        )

    logging.info("Logging setup complete.")
    logging.info("--- _setup_logging END ---")


#
# End of Logging_Config.py
########################################################################################################################

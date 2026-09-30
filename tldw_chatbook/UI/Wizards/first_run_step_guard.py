"""Keep the first-run wizard usable when a step fails (TASK-33621.14).

Review finding G3-01 (2026-09-29): on Quick setup ▸ Provider, highlighting any
of 26 hosted-preset rows raised ``ValueError('Provider is not supported.')``.
The app-level keep-alive handler kept the screen open, but Textual had already
stopped the Provider step's message loop, so the step detached. That left a
blank body, a focused widget that no longer existed (a dead keyboard), and a
Next that queried the detached step and quit the app. Two guards live here, so
``FirstRunSetupWizard.py``, which is already over its size ratchet, does not
grow:

* ``first_run_provider_catalog`` is the Provider step's list: the Settings
  picker's set (``settings_provider_catalog``), which holds only providers
  setup persistence owns, so a listed row can always be selected and saved.
* ``WizardErrorGuard`` and ``contain_advance_error`` handle any other error.
  A step handler that raises, or a Next whose commit raises, leaves the step
  on screen with an error line above the navigation. Keyboard focus stays
  alive, and Back and Exit setup keep working.

Both error guards follow the production app's policy
(``app_lifecycle._handle_exception``). They act only when the UI would
otherwise be kept alive, which excludes headless ``run_test`` unless a test
opts in, so the suite keeps its exception signal.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

STEP_HANDLER_ERROR_COPY = (
    "Something went wrong on this step. Try again or choose something else; "
    "Back and Esc (exit setup) still work. Details are in the log file."
)
ADVANCE_ERROR_COPY = (
    "Something went wrong on this step, so setup stayed here. Retry with Next, "
    "go Back, or press Esc to exit setup. Details are in the log file."
)


def first_run_provider_catalog() -> tuple[Any, ...]:
    """Return the Provider step's entries: exactly the Settings picker's set.

    Returns:
        ``settings_provider_catalog()``: the providers provider setup can
        persist, which Settings ▸ Providers & Models offers too.
    """
    from tldw_chatbook.Chat.console_session_settings import settings_provider_catalog

    return settings_provider_catalog()


def contains_wizard_errors(node: Any) -> bool:
    """Whether a wizard error should be contained instead of raised.

    Args:
        node: Any mounted wizard widget.

    Returns:
        The app's ``_keep_screen_alive_on_handler_error`` choice, which
        defaults to on outside headless runs.
    """
    try:
        app = node.app
    except Exception:  # noqa: BLE001 - an unmounted node has no app to keep alive.
        return False
    return bool(
        getattr(app, "_keep_screen_alive_on_handler_error", not app.is_headless)
    )


def _raise_site(error: BaseException) -> tuple[str, str, int]:
    """Return the innermost frame as identifiers only: module, function, line."""
    site = ("", "", 0)
    trace = error.__traceback__
    while trace is not None:
        frame = trace.tb_frame
        site = (
            str(frame.f_globals.get("__name__", "")),
            frame.f_code.co_name,
            trace.tb_lineno,
        )
        trace = trace.tb_next
    return site


def report_contained_error(
    node: Any, category: str, error: Exception, copy: str
) -> None:
    """Log a contained error and put its step back in the user's hands.

    Logs the error type and raising frame only. The message and traceback
    values can hold what the user typed (an API key), so they are never
    logged. The current step shows ``copy`` on the pinned error line, and
    keyboard focus is re-anchored if the error orphaned it.

    Args:
        node: The step or container whose work raised.
        category: ``"handler"`` or ``"advance"``, for the log line.
        error: The contained exception.
        copy: The user-facing explanation.
    """
    container = node if hasattr(node, "steps") else getattr(node, "wizard", None)
    step = node
    try:
        step = container.steps[container.current_step]
    except Exception:  # noqa: BLE001 - fall back to the node that raised.
        pass
    step_id = getattr(getattr(step, "config", None), "id", "") or "unknown"
    module, function, line = _raise_site(error)
    logger.error(
        "First-run setup error contained (category={}, step={}, error_type={}, "
        "raise_module={}, raise_function={}, raise_line={})",
        category,
        step_id,
        type(error).__name__,
        module,
        function,
        line,
    )
    recovery = (
        ("show_step_error", (copy,)),
        ("_heal_orphaned_focus", ()),
    )
    for name, args in recovery:
        action = getattr(step, name, None)
        if not callable(action):
            continue
        try:
            action(*args)
        except Exception:  # noqa: BLE001 - recovery must not raise again.
            logger.warning("First-run setup error recovery skipped (action={})", name)


class WizardErrorGuard:
    """Mixin for wizard widgets: a raising handler must not kill the widget.

    Textual breaks a widget's message loop after one of its handlers raises,
    and the widget then drops out of the DOM. Catching the error inside the
    widget's own dispatch keeps the loop, the widget and its focus alive.
    """

    async def _on_message(self, message: Any) -> None:
        try:
            await super()._on_message(message)  # type: ignore[misc]
        except Exception as error:
            if not contains_wizard_errors(self):
                raise
            report_contained_error(self, "handler", error, STEP_HANDLER_ERROR_COPY)


def contain_advance_error(container: Any, error: Exception) -> bool:
    """Report a Next that raised, if the app keeps its UI alive.

    ``_advance`` runs as an ``exit_on_error`` worker, so an exception that
    escapes it exits the whole app.

    Args:
        container: The ``SetupWizardContainer`` whose advance raised.
        error: The exception.

    Returns:
        True when the error was contained and reported; False when the caller
        must re-raise it (headless runs, so the suite keeps its signal).
    """
    if not contains_wizard_errors(container):
        return False
    report_contained_error(container, "advance", error, ADVANCE_ERROR_COPY)
    return True

"""Keep the first-run wizard usable when a step fails (TASK-33621.14).

Review finding G3-01 (2026-09-29): on Quick setup ▸ Provider, highlighting a
row setup persistence did not own (13 Cloud rows and Custom Hosted at review
time) raised ``ValueError('Provider is not supported.')``.
The app-level keep-alive handler kept the screen open, but Textual had already
stopped the Provider step's message loop, so the step detached. That left a
blank body, a focused widget that no longer existed (a dead keyboard), and a
Next that queried the detached step and quit the app. Two guards live here, so
``FirstRunSetupWizard.py``, which is already over its size ratchet, does not
grow:

* ``first_run_provider_catalog`` is the Provider step's list: the Settings
  picker's set (``settings_provider_catalog``) rather than the whole handler
  catalog, so execution-only ``custom-hosted`` is not offered. Setup
  persistence owns every key in that set (engine presets by registry
  derivation, TASK-33510), and the tests sweep it, so a listed row can always
  be selected and saved.
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

import logging
from typing import Any

from loguru import logger

#: The ``persist_event`` component for first-run setup's contained errors.
_DIAGNOSTICS_COMPONENT = "first_run"
#: Container attribute holding each ``(step, category, type, site)`` already
#: reported, so a failure that repeats (a 250 ms interval) logs once.
_REPORTED_ATTR = "_first_run_contained_error_sites"
_NO_FRAME = ("", "", 0)


def error_copy(category: str, *, back: bool) -> str:
    """Return the pinned-line explanation for a contained error.

    Args:
        category: ``"advance"`` for a Next that raised, else a step handler.
        back: Whether the step has a ← Back (the first step does not).

    Returns:
        User-facing copy that names only controls that work.
    """
    if category == "advance":
        options = "Retry with Next, go Back, or" if back else "Retry with Next or"
        return (
            "Something went wrong on this step, so setup stayed here. "
            f"{options} press Esc to exit setup. The error was logged."
        )
    keys = (
        "Back and Esc (exit setup) still work"
        if back
        else "Esc (exit setup) still works"
    )
    return (
        "Something went wrong on this step. Try again or choose something "
        f"else; {keys}. The error was logged."
    )


def first_run_provider_catalog() -> tuple[Any, ...]:
    """Return the Provider step's entries: exactly the Settings picker's set.

    Returns:
        ``settings_provider_catalog()``, which Settings ▸ Providers & Models
        offers too; provider setup can persist every entry.
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


def _raise_sites(
    error: BaseException,
) -> tuple[tuple[str, str, int], tuple[str, str, int]]:
    """Return the innermost frame and the innermost Chatbook frame.

    Identifiers only (module, function, line), the same selection
    ``app_lifecycle._handle_exception`` persists: the innermost frame is often
    Textual's own (``query_one`` for ``NoMatches``), so the deepest
    ``tldw_chatbook.`` frame is the one that names the failing call site.
    Walked here rather than imported, because ``app_lifecycle`` pulls in the
    whole application.
    """
    frames: list[tuple[str, str, int]] = []
    trace = error.__traceback__
    while trace is not None:
        frame = trace.tb_frame
        name = str(frame.f_globals.get("__name__", ""))
        frames.append((name, frame.f_code.co_name, trace.tb_lineno))
        trace = trace.tb_next
    if not frames:
        return _NO_FRAME, _NO_FRAME
    own = (frame for frame in reversed(frames) if frame[0].startswith("tldw_chatbook."))
    return frames[-1], next(own, frames[-1])


def _wizard_steps(node: Any) -> tuple[Any, Any, Any]:
    """Return ``(container, step that raised, step on screen)`` for ``node``."""
    container = node if hasattr(node, "steps") else getattr(node, "wizard", None)
    try:
        current = container.steps[container.current_step]
    except Exception:  # noqa: BLE001 - no container, or no step on screen yet.
        current = None
    return container, (current if node is container else node), current


def _back_available(container: Any) -> bool:
    """Whether the step on screen has a ← Back, read from its track position.

    The button's own state is no guide mid-Next: ``_advance`` disables Back
    until it settles, which is exactly when an advance error is reported.
    """
    position = getattr(container, "_active_position", None)
    try:
        return callable(position) and position(container.current_step) > 0
    except Exception:  # noqa: BLE001 - unknown position: do not promise Back.
        return False


def _persist(
    node: Any, category: str, step_id: str, error: Exception, sites: Any
) -> None:
    """Record the contained error on the persistent diagnostics channel."""
    (raise_mod, raise_fn, raise_line), (site_mod, site_fn, site_line) = sites
    frame_fields: dict[str, object] = {}
    if raise_mod:
        frame_fields = {
            "raise_module": raise_mod,
            "raise_function": raise_fn,
            "raise_line": raise_line,
            "site_module": site_mod,
            "site_function": site_fn,
            "site_line": site_line,
        }
    try:
        from tldw_chatbook.Utils.persistent_diagnostics import persist_event

        persist_event(
            _DIAGNOSTICS_COMPONENT,
            "contained_exception",
            level=logging.ERROR,
            operation=category,
            phase=step_id,
            exception_type=type(error).__name__,
            widget_type=type(node).__name__,
            **frame_fields,
        )
    except Exception:  # noqa: BLE001 - diagnostics must never fail the recovery.
        pass


def report_contained_error(node: Any, category: str, error: Exception) -> None:
    """Log a contained error and put its step back in the user's hands.

    Records the error type and the raising frames only. The message and
    traceback values can hold what the user typed (an API key), so they are
    never logged. The log names the step whose work raised; only the step on
    screen shows the pinned error line (a hidden step's timer must not put
    "went wrong" over a healthy step). A repeat of the same failure logs once.
    Keyboard focus is re-anchored if the error orphaned it.

    Args:
        node: The step or container whose work raised.
        category: ``"handler"`` or ``"advance"``, for the log and the copy.
        error: The contained exception.
    """
    container, raising, current = _wizard_steps(node)
    step_id = getattr(getattr(raising, "config", None), "id", "") or "unknown"
    sites = _raise_sites(error)
    key = (step_id, category, type(error).__name__, sites)
    owner = container if container is not None else node
    reported = getattr(owner, _REPORTED_ATTR, None)
    if reported is None:
        reported = set()
        setattr(owner, _REPORTED_ATTR, reported)
    (raise_mod, raise_fn, raise_line), (site_mod, site_fn, site_line) = sites
    if key in reported:
        logger.debug(
            "First-run setup error contained again (category={}, step={}, "
            "error_type={})",
            category,
            step_id,
            type(error).__name__,
        )
    else:
        reported.add(key)
        logger.error(
            "First-run setup error contained (category={}, step={}, error_type={}, "
            "raise_module={}, raise_function={}, raise_line={}, site_module={}, "
            "site_function={}, site_line={})",
            category,
            step_id,
            type(error).__name__,
            raise_mod,
            raise_fn,
            raise_line,
            site_mod,
            site_fn,
            site_line,
        )
        _persist(node, category, step_id, error, sites)
    recovery: list[tuple[Any, str, tuple[Any, ...]]] = []
    if raising is current:
        copy = error_copy(category, back=_back_available(container))
        recovery.append((current, "show_step_error", (copy,)))
    recovery.append((current, "_heal_orphaned_focus", ()))
    for step, name, args in recovery:
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
            report_contained_error(self, "handler", error)


def contain_advance_error(container: Any, error: Exception) -> bool:
    """Report a Next that raised, if the app keeps its UI alive.

    ``_advance`` runs as an ``exit_on_error`` worker, so an exception that
    escapes it exits the whole app. Its ``finally`` then re-syncs the nav bar
    (``_set_advancing(False)`` runs ``update_progress``), so a Next that
    failed part-way through a step change still shows the right position.

    Args:
        container: The ``SetupWizardContainer`` whose advance raised.
        error: The exception.

    Returns:
        True when the error was contained and reported; False when the caller
        must re-raise it (headless runs, so the suite keeps its signal).
    """
    if not contains_wizard_errors(container):
        return False
    report_contained_error(container, "advance", error)
    return True

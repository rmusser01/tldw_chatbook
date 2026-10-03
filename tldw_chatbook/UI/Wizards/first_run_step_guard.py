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
* ``WizardErrorGuard`` and ``report_advance_error`` handle any other error.
  A step handler that raises, a key binding (Ctrl+B, Ctrl+N, Enter, Esc)
  whose action raises, or a Next whose commit raises, leaves the step on
  screen with an error line above the navigation. The nav bar is re-synced
  to the step on screen, keyboard focus stays alive, and Back and Exit setup
  keep working. ``provider_switch`` names the provider that stays selected
  when picking another one fails, and clears that line once a pick succeeds.

The handler and key-binding guards follow the production app's policy
(``app_lifecycle._handle_exception``). They act only when the UI would
otherwise be kept alive, which excludes headless ``run_test`` unless a test
opts in, so the suite keeps its exception signal.

Background work is different (TASK-34100.1). A Textual worker started with
its default ``exit_on_error=True`` takes the whole app down whatever the
app's keep-alive policy says. ``run_wizard_worker`` and ``@wizard_work`` are
the one way first-run code starts a worker. Both always pass
``exit_on_error=False`` and report an error that escapes the work on the
pinned status strip, in headless runs too.
``Tests/Architecture/test_wizard_lifecycle_guards.py`` pins that no first-run
module calls ``run_worker`` or ``@work`` directly.
"""

from __future__ import annotations

import functools
import inspect
import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Iterator

from loguru import logger
from textual.actions import SkipAction

if TYPE_CHECKING:
    from textual.worker import Worker

#: The ``persist_event`` component for first-run setup's contained errors.
_DIAGNOSTICS_COMPONENT = "first_run"
#: Container attribute holding each ``(step, category, type, site)`` already
#: reported, so a failure that repeats (a 250 ms interval) logs once.
_REPORTED_ATTR = "_first_run_contained_error_sites"
#: Container attribute holding the error line a contained error put up, so a
#: later success clears that line and never a different step message.
_SHOWN_ATTR = "_first_run_contained_error_copy"
#: Marks an action method ``WizardErrorGuard`` already wrapped.
_CONTAINED_ACTION_ATTR = "_first_run_contained_action"
#: A key-bound action whose failure is reported as a Next that stayed put:
#: ``action_next`` only launches the advance worker, so nothing moved yet.
_ADVANCE_ACTIONS = frozenset({"action_next"})
#: Container attribute holding the widget that had focus when a fence
#: disabled the nav bar, so lifting the fence can give focus back.
_FENCED_FOCUS_ATTR = "_first_run_fenced_focus"
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


def switch_error_copy(target: str, kept: str, *, back: bool) -> str:
    """Return the pinned-line copy for a provider switch that failed.

    The list's highlight has already moved to ``target`` (arrow keys move it
    before the switch runs), so the line must say which provider is really
    selected: that is the one Next saves, with the key typed for it.

    Args:
        target: Display name of the provider the user tried to pick.
        kept: Display name of the provider still selected, or ``""``.
        back: Whether the step has a ← Back.

    Returns:
        User-facing copy that names the selected provider and working keys.
    """
    keys = (
        "Back and Esc (exit setup) still work"
        if back
        else "Esc (exit setup) still works"
    )
    stays = f" — {kept} is still selected" if kept else ""
    return (
        f"Couldn't switch to {target}{stays}. Choose another provider or try "
        f"again; {keys}. The error was logged."
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


def _release_hidden_focus(step: Any) -> None:
    """Drop focus that sits inside a hidden subtree, so the heal re-anchors it.

    A failure part-way through ``show_step`` (the entered step's ``on_show``)
    hides the old step before the step-change focus fix runs. The focused
    widget is still attached and its own ``display`` is still on, so the
    step's heal sees nothing wrong until Textual drops the focus at the next
    layout, and from then on no wizard binding resolves.
    """
    try:
        app = step.app
        focused = app.focused
        if focused is None or step.screen is not app.screen:
            return
        if all(getattr(node, "display", True) for node in focused.ancestors_with_self):
            return
        app.screen.set_focus(None)
    except Exception:  # noqa: BLE001 - no app or screen: nothing to release.
        return


def report_contained_error(
    node: Any,
    category: str,
    error: Exception,
    *,
    copy: str | None = None,
    step: Any = None,
) -> None:
    """Log a contained error and put its step back in the user's hands.

    Records the error type and the raising frames only. The message and
    traceback values can hold what the user typed (an API key), so they are
    never logged. The log names the step whose work raised; only the step on
    screen shows the pinned error line (a hidden step's timer must not put
    "went wrong" over a healthy step). A repeat of the same failure logs once.
    The nav bar is re-synced to the step on screen (a failure part-way
    through a step change leaves it counting the old one), and keyboard focus
    is re-anchored if the error orphaned it.

    Args:
        node: The step or container whose work raised.
        category: ``"handler"`` or ``"advance"``, for the log and the copy.
        error: The contained exception.
        copy: The error line to show instead of the category's own.
        step: The step whose work raised when the wizard has since moved off
            it (a Next that failed after changing step). The log names it;
            the step on screen shows the handler copy, never "stayed here".
    """
    container, raising, current = _wizard_steps(node)
    moved = step is not None and step is not current
    if step is not None:
        raising = step
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
    recovery: list[tuple[str, Callable[[], Any] | None]] = []
    if raising is current or moved:
        shown = copy or error_copy(
            "handler" if moved else category, back=_back_available(container)
        )
        if container is not None:
            setattr(container, _SHOWN_ATTR, shown)
        recovery.append(("show_step_error", _bound(current, "show_step_error", shown)))
    # What ``show_step`` runs after the step change, in its order.
    for name in ("validate_step", "_sync_exit_controls", "_sync_action_controls"):
        recovery.append((name, _bound(container, name)))
    recovery.append(("release_hidden_focus", lambda: _release_hidden_focus(current)))
    recovery.append(("_heal_orphaned_focus", _bound(current, "_heal_orphaned_focus")))
    for name, action in recovery:
        if action is None:
            continue
        try:
            action()
        except Exception:  # noqa: BLE001 - recovery must not raise again.
            logger.warning("First-run setup error recovery skipped (action={})", name)


def _bound(target: Any, name: str, *args: Any) -> Callable[[], Any] | None:
    """Return ``target.name(*args)`` as a thunk, or None if it has no such method."""
    method = getattr(target, name, None) if target is not None else None
    return functools.partial(method, *args) if callable(method) else None


def _display_name(provider_key: str) -> str:
    """Return a provider's display name, or the key if the catalog has none."""
    try:
        from tldw_chatbook.Chat.provider_catalog import provider_display_name

        return provider_display_name(provider_key) or provider_key
    except Exception:  # noqa: BLE001 - a missing name must not fail the report.
        return provider_key


def clear_contained_error(node: Any) -> None:
    """Take down the error line a contained error put up, now that work succeeded.

    Only that line: a probe failure or a refused Next shown since stays up.

    Args:
        node: A mounted wizard step or the container.
    """
    container, _raising, _current = _wizard_steps(node)
    shown = getattr(container, _SHOWN_ATTR, "") if container is not None else ""
    if not shown:
        return
    setattr(container, _SHOWN_ATTR, "")
    try:
        strip = container.screen.query_one("#setup-step-error-pinned")
        if getattr(strip, "content", None) == shown:
            container._clear_pinned_step_error()
    except Exception:  # noqa: BLE001 - no strip: nothing to clear.
        return


@contextmanager
def provider_switch(step: Any, provider_key: str) -> Iterator[None]:
    """Contain a provider pick that fails, and say which provider stays selected.

    Arrow keys move the list's highlight before the pick runs, so after a
    failure the highlighted row is not the selected provider. The highlight
    stays where the user put it (snapping it back would trap the arrow keys
    above a failing row); the error line names the provider Next will save
    instead. A pick that fails after selecting its provider (model discovery
    raised) leaves highlight and selection agreeing, so its line names none.
    A failed pick that leaves nothing selected also stops Next falling back
    to the highlighted row: Next saves no provider, as with no pick. A pick
    that succeeds, or a return to the selected row, takes that line down.

    Args:
        step: The Provider step.
        provider_key: The provider being picked.

    Yields:
        None; the ``with`` body runs the pick.
    """
    kept = getattr(step, "selected_provider_key", "") or ""
    try:
        yield
    except Exception as error:
        if not contains_wizard_errors(step):
            raise
        selected = getattr(step, "selected_provider_key", "") or ""
        if not selected:
            # ``_effective_provider_key`` would hand Next the failed row.
            step._provider_choice_interacted = False
        container, _raising, _current = _wizard_steps(step)
        copy = None  # it got as far as selecting: the plain handler line
        if selected == kept:
            copy = switch_error_copy(
                _display_name(provider_key),
                _display_name(kept) if kept else "",
                back=_back_available(container),
            )
        report_contained_error(step, "handler", error, copy=copy)
        return
    clear_contained_error(step)


def _contained_action(method: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a key-bound action so its failure is reported, not fatal."""
    category = "advance" if method.__name__ in _ADVANCE_ACTIONS else "handler"

    def _contains(node: Any, error: Exception) -> bool:
        # SkipAction is Textual's "not handled here" signal, not a failure.
        return not isinstance(error, SkipAction) and contains_wizard_errors(node)

    if inspect.iscoroutinefunction(method):

        @functools.wraps(method)
        async def run_async(self: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                return await method(self, *args, **kwargs)
            except Exception as error:
                if not _contains(self, error):
                    raise
                report_contained_error(self, category, error)
                return None

        wrapper: Callable[..., Any] = run_async
    else:

        @functools.wraps(method)
        def run(self: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                return method(self, *args, **kwargs)
            except Exception as error:
                if not _contains(self, error):
                    raise
                report_contained_error(self, category, error)
                return None

        wrapper = run
    setattr(wrapper, _CONTAINED_ACTION_ATTR, True)
    return wrapper


class WizardErrorGuard:
    """Mixin for wizard widgets: a raising handler must not kill the widget.

    Textual breaks a widget's message loop after one of its handlers raises,
    and the widget then drops out of the DOM. Catching the error inside the
    widget's own dispatch keeps the loop, the widget and its focus alive.

    Key bindings need their own cover. Textual runs a non-priority binding's
    action (the container's Ctrl+B, Ctrl+N, Enter and Esc) from the App's
    message loop, which never passes through this widget's ``_on_message``,
    and the app refuses to keep its own loop alive: a failure a click on
    Back survives would quit the app from the keyboard. So every
    ``action_*`` / ``_action_*`` method a guarded class defines is wrapped
    when the class is created, and reports its failure the same way.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for name, member in list(vars(cls).items()):
            if (
                name.startswith(("action_", "_action_"))
                and inspect.isfunction(member)
                and not getattr(member, _CONTAINED_ACTION_ATTR, False)
            ):
                setattr(cls, name, _contained_action(member))

    async def _on_message(self, message: Any) -> None:
        try:
            await super()._on_message(message)  # type: ignore[misc]
        except Exception as error:
            if not contains_wizard_errors(self):
                raise
            report_contained_error(self, "handler", error)


def hold_fenced_focus(container: Any) -> None:
    """Remember who has focus before a fence disables the nav bar.

    Textual blurs a focused widget the moment it is disabled
    (``Widget.watch_disabled``). With Back, Next and Exit all disabled, the
    blur leaves focus at None. A Next that moves on gets focus back from
    ``show_step``. One that stays on its step (the commit refused, or the
    checkpoint failed) got nothing back, so Enter, Ctrl+N and Ctrl+B went
    dead under copy that said "Retry with Next" (TASK-34100.1 review).

    Args:
        container: The ``SetupWizardContainer`` about to fence its nav bar.
    """
    try:
        focused = container.screen.focused
    except Exception:  # noqa: BLE001 - not on a screen yet: nothing to hold.
        return
    if focused is not None:
        setattr(container, _FENCED_FOCUS_ATTR, focused)


def restore_fenced_focus(container: Any) -> None:
    """Give focus back once the fence lifts, unless something else took it.

    Runs after the callbacks already queued, because ``Widget.focus`` defers
    too: a ``show_step`` that just moved to a new step has queued its own
    focus, which must win. Focus goes back to the held widget if it can take
    it again, else the step on screen re-anchors it.

    Args:
        container: The ``SetupWizardContainer`` whose fence just lifted.
    """
    held = getattr(container, _FENCED_FOCUS_ATTR, None)
    if held is None:
        return
    setattr(container, _FENCED_FOCUS_ATTR, None)

    def restore() -> None:
        try:
            screen = container.screen
            if screen.focused is not None or not container.is_attached:
                return
            if (
                held.is_attached
                and held.focusable
                and held.screen is screen
                and all(getattr(node, "display", True) for node in held.ancestors_with_self)
            ):
                screen.set_focus(held)
                return
            heal = _bound(container.steps[container.current_step], "_heal_orphaned_focus")
        except Exception:  # noqa: BLE001 - the wizard is gone: nothing to restore.
            return
        if heal is not None:
            heal()

    try:
        container.app.call_later(restore)
    except Exception:  # noqa: BLE001 - no running app: nothing to restore.
        return


def report_advance_error(
    container: Any, error: Exception, started_at: int | None = None
) -> None:
    """Report a Next that raised, and keep the wizard where the user can act.

    ``_advance`` catches its own errors and reports them here, so a failing
    Next never escapes its worker. Its ``finally`` then re-syncs the nav bar
    (``_set_advancing(False)`` runs ``update_progress``), so a Next that
    failed part-way through a step change still shows the right position.
    Only a Next that failed before leaving its step says "setup stayed here".
    One that failed after (the next step's ``on_show``) is logged against the
    step it committed, and shows neutral copy on the step now on screen.

    TASK-34100.1: this used to apply only when the app keeps its UI alive and
    re-raised otherwise, out of a worker started with ``exit_on_error=True``.
    In a headless run that exited the app, and it was the one advance path
    that could. Background work now never exits the app (see the module
    docstring).

    Args:
        container: The ``SetupWizardContainer`` whose advance raised.
        error: The exception.
        started_at: ``current_step`` when the advance began.
    """
    committing = None
    if started_at is not None and started_at != container.current_step:
        try:
            committing = container.steps[started_at]
        except Exception:  # noqa: BLE001 - unknown step: attribute to the container.
            committing = None
    report_contained_error(container, "advance", error, step=committing)


def _report_target(node: Any) -> Any:
    """The wizard node a worker error is reported against.

    Steps and the container carry their own wizard context (``wizard`` and
    ``steps``). The setup screen does not, so its errors are reported against
    the container it hosts.
    """
    if hasattr(node, "steps") or hasattr(node, "wizard"):
        return node
    try:
        return next(iter(node.query(".wizard-container")), node)
    except Exception:  # noqa: BLE001 - no DOM to search: report on the node.
        return node


def _report_worker_error(node: Any, error: BaseException, category: str) -> None:
    """Log a worker's escaped error and show it on the pinned strip.

    A node that has already left the DOM (the user moved on, or dismissed
    setup) only logs: there is no strip of its own left to show.
    """
    try:
        attached = bool(node.is_attached)
    except Exception:  # noqa: BLE001 - no app: treat as detached.
        attached = False
    if not attached or not isinstance(error, Exception):
        logger.error(
            "First-run setup worker failed after its step left "
            "(category={}, error_type={})",
            category,
            type(error).__name__,
        )
        return
    try:
        report_contained_error(_report_target(node), category, error)
    except Exception:  # noqa: BLE001 - reporting must never raise again.
        logger.warning(
            "First-run setup worker error report skipped (error_type={})",
            type(error).__name__,
        )


def run_wizard_worker(
    node: Any,
    work: Callable[[], Any] | Awaitable[Any],
    *,
    group: str,
    exclusive: bool = False,
    thread: bool = False,
    name: str = "",
    description: str = "",
    category: str = "handler",
) -> "Worker[Any]":
    """Start first-run background work that can never exit the app.

    The one ``run_worker`` call site for first-run setup (TASK-34100.1). The
    worker always runs with ``exit_on_error=False``. An error that escapes
    ``work`` is logged by type and site only, never by message (it can hold
    a typed key), and shown on the pinned status strip of the step on
    screen. Then it is re-raised, so the worker still ends in
    ``WorkerState.ERROR`` and ``Worker.wait()`` still raises. Cancellation is
    not an error, and passes through.

    Args:
        node: The step, container or setup screen that owns the work. Its
            removal cancels the worker, as with ``node.run_worker``.
        work: A coroutine, an async callable, or (with ``thread=True``) a
            plain callable.
        group: The worker group (exclusive workers cancel their group).
        exclusive: Cancel the group's other workers first. Defaults to
            False, as Textual's ``run_worker`` and ``@work`` do.
        thread: Run ``work`` on a thread (a plain callable).
        name: The worker's name; defaults to the callable's name.
        description: The worker's debug description. It is not built from
            arguments, so a password argument never reaches it.
        category: ``"advance"`` for a Next, else ``"handler"``; picks the
            pinned-strip copy.

    Returns:
        The started ``Worker``.
    """
    if thread:
        if not callable(work):
            raise TypeError("a thread wizard worker needs a plain callable")

        def run_on_thread() -> Any:
            try:
                return work()
            except Exception as error:
                try:
                    node.app.call_from_thread(
                        _report_worker_error, node, error, category
                    )
                except Exception:  # noqa: BLE001 - app gone: nothing to show.
                    logger.error(
                        "First-run setup worker failed (error_type={})",
                        type(error).__name__,
                    )
                raise

        target: Any = run_on_thread
    else:

        async def run_on_loop() -> Any:
            try:
                result = work if inspect.isawaitable(work) else work()
                if inspect.isawaitable(result):
                    result = await result
                return result
            except Exception as error:
                _report_worker_error(node, error, category)
                raise

        target = run_on_loop
    label = name or getattr(work, "__name__", "") or getattr(
        getattr(work, "func", None), "__name__", ""
    )
    return node.run_worker(
        target,
        name=label,
        group=group,
        description=description or label,
        exclusive=exclusive,
        thread=thread,
        exit_on_error=False,
    )


def wizard_work(
    *,
    group: str,
    exclusive: bool = False,
    thread: bool = False,
    description: str = "",
) -> Callable[[Callable[..., Any]], Callable[..., "Worker[Any]"]]:
    """``@work`` for first-run setup: every call starts a ``run_wizard_worker``.

    Mirrors Textual's decorator: calling the method starts a worker and
    returns it, and ``__wrapped__`` still reaches the original body. The
    worker never exits the app (see ``run_wizard_worker``).

    Args:
        group: The worker group.
        exclusive: Cancel the group's other workers first.
        thread: Run the method on a thread (a plain ``def``).
        description: The worker's debug description; defaults to the name.

    Returns:
        The decorator.

    Raises:
        TypeError: A non-thread worker that is not ``async def``, the same
            mistake Textual's own ``WorkerDeclarationError`` refuses.
    """

    def decorator(method: Callable[..., Any]) -> Callable[..., "Worker[Any]"]:
        if not thread and not inspect.iscoroutinefunction(method):
            raise TypeError(
                f"{method.__name__} must be async def, or use thread=True"
            )

        @functools.wraps(method)
        def launch(self: Any, *args: Any, **kwargs: Any) -> "Worker[Any]":
            return run_wizard_worker(
                self,
                functools.partial(method, self, *args, **kwargs),
                group=group,
                exclusive=exclusive,
                thread=thread,
                name=method.__name__,
                description=description or method.__name__,
            )

        return launch

    return decorator

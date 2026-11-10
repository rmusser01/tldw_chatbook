"""Give a test's Console controller the host production would build (TASK-36947.4).

``ConsoleChatController`` no longer holds the app: it receives a
``ConsoleControllerHost`` of service members and narrow callables. Tests used to
assign ``controller.app = fake``; they now call ``bind_app(controller, fake)``,
which builds the host over ``fake`` with the production builder
(``console_controller_host``) and installs it as construction would have.

Test apps rarely publish ``app.services``, so members come from the fake's
alias attributes, exactly the names the controller's old ``getattr`` reads used.

A bound host is rebuilt from the fake on every read. Production builds it once,
because a real app's services and methods never change after composition; a
test fake is patched after binding (``app.notify = spy``), and the old
``controller.app`` reads saw that, so the bound host must too. ``bound_app``
returns the fake for assertions.
"""

from __future__ import annotations

import weakref
from typing import Any

from tldw_chatbook.Chat.console_controller_host import (
    MEMBER_SOURCES,
    NO_HOST,
    ConsoleControllerHost,
    console_controller_host,
)

#: Members a test fake supplies through its alias attributes.
MEMBER_NAMES = (*MEMBER_SOURCES, "raw_cli_runtime")

_BOUND: "weakref.WeakKeyDictionary[Any, Any]" = weakref.WeakKeyDictionary()


def host_for_app(app: Any) -> ConsoleControllerHost | None:
    """Build the host production would build over ``app``; ``None`` for no app.

    Args:
        app: A fake app-like object (or ``None``).

    Returns:
        The host, with members read from ``app``'s alias attributes.
    """
    if app is None:
        return None
    return console_controller_host(
        app, **{name: getattr(app, name, None) for name in MEMBER_NAMES}
    )


class _LiveHost:
    """A host read through ``host_for_app(app)`` on every attribute access."""

    __slots__ = ("_fake_app",)

    def __init__(self, app: Any) -> None:
        self._fake_app = app

    def __getattr__(self, name: str) -> Any:
        return getattr(host_for_app(self._fake_app), name)


def bind_app(controller: Any, app: Any) -> Any:
    """Install the host ``controller`` would have been constructed with.

    Args:
        controller: A ``ConsoleChatController`` (or a controller double).
        app: The fake app the test used to assign to ``controller.app``.

    Returns:
        ``app``, so ``app = bind_app(controller, SimpleNamespace(...))`` reads.
    """
    controller._host = _LiveHost(app) if app is not None else NO_HOST
    if not isinstance(getattr(type(controller), "host", None), property):
        controller.host = controller._host  # a double: mirror the read-only property
    try:
        _BOUND[controller] = app
    except TypeError:  # a controller double that cannot be weakly referenced
        pass
    return app


class _MarksHost:
    """The controller's host with the fleet marks taken over another app."""

    __slots__ = ("_base", "set_fleet_unseen_completion", "clear_fleet_unseen_completion")

    def __init__(self, base: Any, app: Any) -> None:
        self._base = base
        marks = console_controller_host(app)
        self.set_fleet_unseen_completion = marks.set_fleet_unseen_completion
        self.clear_fleet_unseen_completion = marks.clear_fleet_unseen_completion

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)


def bind_fleet_marks(controller: Any, app: Any) -> None:
    """Give the fleet wake ``app``'s durable marks, as ``wire(app=app)`` did.

    Before TASK-36947.4 a test could hand the wake a marks app different
    from the controller's own (or give a host-less controller one); the
    wake now reads the marks from the controller's host.
    """
    controller._host = _MarksHost(controller._host, app)
    if not isinstance(getattr(type(controller), "host", None), property):
        controller.host = controller._host


def bind_runtime_app(controller: Any, runtime: Any) -> None:
    """Bind the runtime's app, as ``set_chat_controller`` did before TASK-36947.4.

    Production builds the host when the runtime constructs the controller; a
    test that hands the runtime a controller it built itself binds here first.
    A runtime without an app leaves the controller as it was.
    """
    app = getattr(runtime, "_app", None)
    if app is not None:
        bind_app(controller, app)


def update_app(controller: Any, **attributes: Any) -> None:
    """Set attributes on the controller's app, as ``controller.app.<name> = ...`` did.

    The app is the bound fake, else the owning runtime's app (which the old
    ``controller.app`` was); a runtime-built host is then re-bound live so the
    patch reaches the controller.

    Raises:
        AssertionError: ``controller`` has neither a bound fake nor a runtime app.
    """
    app = bound_app(controller)
    if app is None:
        app = getattr(getattr(controller, "_hooks_v2_runtime", None), "_app", None)
        assert app is not None, "bind_app(controller, app) first"
        bind_app(controller, app)
    for name, value in attributes.items():
        setattr(app, name, value)


def bound_app(controller: Any) -> Any:
    """Return the fake app last bound to ``controller`` (``None`` if none)."""
    try:
        return _BOUND.get(controller)
    except TypeError:
        return None

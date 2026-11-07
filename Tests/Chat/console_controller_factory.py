"""Give a test's Console controller the host production would build (TASK-36947.4).

``ConsoleChatController`` no longer holds the app: it receives a
``ConsoleControllerHost`` of service members and narrow callables. Tests used to
assign ``controller.app = fake``; they now call ``bind_app(controller, fake)``,
which builds the host over ``fake`` with the production builder
(``console_controller_host``) and installs it as construction would have.

Test apps rarely publish ``app.services``, so members come from the fake's
alias attributes, exactly the names the controller's old ``getattr`` reads used.
The host is built once, like production's: set every member and capability on
the fake BEFORE binding. ``bound_app`` returns the fake for assertions.
"""

from __future__ import annotations

import weakref
from typing import Any

from tldw_chatbook.Chat.console_controller_host import (
    MEMBER_NAMES,
    ConsoleControllerHost,
    console_controller_host,
)

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


def bind_app(controller: Any, app: Any) -> Any:
    """Install the host ``controller`` would have been constructed with.

    Args:
        controller: A ``ConsoleChatController`` (or a controller double).
        app: The fake app the test used to assign to ``controller.app``.

    Returns:
        ``app``, so ``app = bind_app(controller, SimpleNamespace(...))`` reads.
    """
    from tldw_chatbook.Chat.console_chat_controller import _NO_HOST

    host = host_for_app(app)
    controller._host = host if host is not None else _NO_HOST
    try:
        _BOUND[controller] = app
    except TypeError:  # a controller double that cannot be weakly referenced
        pass
    return app


def update_app(controller: Any, **attributes: Any) -> None:
    """Set attributes on the bound fake app, then rebuild the host from it.

    A test that patched ``controller.app.<name>`` after binding would otherwise
    patch a fake the controller no longer reads.

    Raises:
        AssertionError: ``controller`` has no bound fake app.
    """
    app = bound_app(controller)
    assert app is not None, "bind_app(controller, app) first"
    for name, value in attributes.items():
        setattr(app, name, value)
    bind_app(controller, app)


def bound_app(controller: Any) -> Any:
    """Return the fake app last bound to ``controller`` (``None`` if none)."""
    try:
        return _BOUND.get(controller)
    except TypeError:
        return None

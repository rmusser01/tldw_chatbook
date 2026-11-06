"""A served-canvas gateway stand-in that cannot drift from ``CanvasGateway``.

TASK-38992: hand-written stand-ins in ``Tests/Web_Server`` kept only the
methods ``serve.py`` called when they were written. When ``serve.py`` began
calling ``revoke_browser_session`` on unbind (df22692d70), one failed far
from its cause with ``AttributeError`` on a ``SimpleNamespace``.
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable
from functools import cache
from pathlib import Path
from unittest.mock import NonCallableMagicMock, create_autospec

from tldw_chatbook.Canvas.gateway import CanvasGateway
from tldw_chatbook.Web_Server import serve


@cache
def served_gateway_calls() -> frozenset[str]:
    """Every attribute ``serve.py`` reads on ``self._served_canvas_gateway``."""
    tree = ast.parse(Path(serve.__file__).read_text(encoding="utf-8"))
    return frozenset(
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "_served_canvas_gateway"
    )


def served_gateway_double(**methods: Callable[..., object]) -> NonCallableMagicMock:
    """Return a ``CanvasGateway`` stand-in with the real interface.

    Every gateway method exists with its real signature. The ones given in
    ``methods`` run as that method; any other raises when called, naming
    itself. Each call is recorded for ``assert_called_*``.

    Args:
        methods: Gateway method name to the callable it runs (sync, or async
            for an async method).

    Returns:
        The stand-in.

    Raises:
        AttributeError: ``serve.py`` calls, or ``methods`` names, a method
            ``CanvasGateway`` lacks; the message names it.
    """
    names = {
        name
        for name, member in inspect.getmembers(CanvasGateway, inspect.isfunction)
        if not name.startswith("_")
    }
    unknown = sorted((set(methods) | served_gateway_calls()) - names)
    if unknown:
        raise AttributeError(f"CanvasGateway has no method(s): {unknown}")
    gateway = create_autospec(CanvasGateway, instance=True, spec_set=True)
    for name in names:
        getattr(gateway, name).side_effect = methods.get(name) or _unexpected(name)
    return gateway


def _unexpected(name: str) -> Callable[..., object]:
    def call(*_args: object, **_kwargs: object) -> object:
        raise AssertionError(f"unexpected served-canvas gateway call: {name}")

    return call

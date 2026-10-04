"""Signature probes for optional keyword arguments."""

import inspect
from typing import Any


def accepts_keyword(func: Any, name: str) -> bool:
    """Report whether ``func`` can be called with the ``name`` keyword.

    Asked up front instead of calling and catching ``TypeError``: that pattern
    cannot tell "this callable has no such parameter" from "a ``TypeError`` was
    raised inside it", so a genuine bug downstream reads as a missing feature
    and degrades silently. That is exactly how the remote ingest poller shipped
    asking for an ``offset`` the client did not yet accept, and paginated
    nothing for it (task-684.2).

    A callable whose signature cannot be read (a C builtin, an exotic mock) is
    reported as accepting the keyword, so real services are not downgraded by
    an unreadable signature; ``**kwargs`` counts as accepting it.
    """
    try:
        parameters = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return True
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return True
    parameter = parameters.get(name)
    return parameter is not None and parameter.kind in {
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    }

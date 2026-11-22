"""Cached handler lookup and O(1) callback arity for Textual's message pump.

Every message a widget processes goes through ``MessagePump._get_dispatch_methods``
at least twice: once for the message and once for the ``Idle`` event the pump
dispatches when its queue empties. Upstream walks the widget's whole MRO each
time and makes three ``cls.__dict__`` lookups per class, although the answer
only changes when a class is created or patched. A ``call_later`` callback
additionally runs ``inspect.signature`` on every invocation, because the
``functools.partial`` it is wrapped in is new each time and upstream's
``_param_count`` cache lives on that partial.

TASK-40608 measured this on the Console (558 mounted widgets, Windows 10,
2026-11-21): one ``Pilot._wait_for_screen`` pass, which posts one callback to
every widget, cost 34.5 ms of UI-thread CPU. In an isolated 553-widget app the
same pass costs 26.3 ms upstream and 22.0 ms with the cached lookup; the
arity cache removes one ``inspect.signature`` call per widget on top of that.
The same per-message cost is paid by every timer tick, refresh and app message
in normal use.

Both replacements keep upstream as the definition of behaviour:

* ``count_parameters`` only changes the ``partial`` case, and only when the
  wrapped callable is a plain Python function object. Upstream's own
  ``inspect.signature`` count for that function is then kept on the function,
  as upstream already does for callables that are not partials, so the
  hundreds of partials one barrier pass makes around the same function share
  one count. Everything else calls upstream.
* ``_get_dispatch_methods`` replays, per (pump class, message class, handler
  name), the list of classes that upstream would yield from. The cached plan is
  dropped whenever a handler attribute is set or deleted on any message-pump
  class, and it re-checks the three lookups for every class in the MRO that is
  not a message-pump class (plain mixins), so a patched handler is always seen.

``Tests/Performance/test_textual_dispatch_fastpath.py`` pins the upstream
source both replacements were derived from and compares their results with
upstream's, so a Textual upgrade fails there instead of changing dispatch.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from types import FunctionType
from typing import Any

import textual._callback as _callback
from textual.message import Message
from textual.message_pump import MessagePump, _MessagePumpMeta

__all__ = ["install_dispatch_fastpath", "is_installed", "upstream"]

#: Marks the patched callables so installation is idempotent and detectable.
_MARKER = "_tldw_dispatch_fastpath"

#: Attribute holding upstream's ``inspect.signature`` parameter count on a
#: plain function used as a partial's target.
_TOTAL = "_tldw_signature_parameters"

#: (pump class, message class, handler name) -> plan; see ``_build_plan``.
_plans: dict[tuple[type, type, str], tuple] = {}
#: Handler names that have been asked for, with and without the ``_`` prefix.
_watched_names: set[str] = {"_decorated_handlers"}
#: The upstream callables this module replaced, for the fidelity tests.
_upstream: dict[str, Any] = {}


def _build_plan(pump_class: type, message_class: type, method_name: str) -> tuple:
    """Record what upstream's MRO walk finds for this dispatch.

    Returns:
        ``(entries, mixins)``. ``entries`` holds, in MRO order, one
        ``(cls, decorated handlers, convention handler)`` per class that can
        yield. ``mixins`` holds, for each class that is not a message-pump
        class, its namespace and the three objects the walk read from it, so a
        later call can tell whether one was patched.
    """
    message_mro = [
        _type for _type in message_class.__mro__ if issubclass(_type, Message)
    ]
    private_name = f"_{method_name}"
    entries = []
    mixins = []
    for cls in pump_class.__mro__:
        namespace = cls.__dict__
        decorated = namespace.get("_decorated_handlers")
        private = namespace.get(private_name)
        public = namespace.get(method_name)
        if cls is not object and not isinstance(cls, _MessagePumpMeta):
            mixins.append((namespace, private_name, decorated, private, public))
        handlers: list = []
        if decorated:
            for message_type in message_mro:
                handlers.extend(decorated.get(message_type, []))
        method = private or public
        if method is not None and getattr(method, "_textual_on", None):
            method = None
        if handlers or method is not None:
            entries.append((cls, tuple(handlers), method))
    return tuple(entries), tuple(mixins)


def _fast_get_dispatch_methods(
    self: MessagePump, method_name: str, message: Message
) -> Iterable[tuple[type, Any]]:
    """``MessagePump._get_dispatch_methods`` from a cached, validated plan."""
    key = (self.__class__, message.__class__, method_name)
    plan = _plans.get(key)
    if plan is not None:
        for namespace, private_name, decorated, private, public in plan[1]:
            if (
                namespace.get("_decorated_handlers") is not decorated
                or namespace.get(private_name) is not private
                or namespace.get(method_name) is not public
            ):
                plan = None
                break
    if plan is None:
        _watched_names.add(method_name)
        _watched_names.add(f"_{method_name}")
        plan = _plans[key] = _build_plan(*key)
    entries = plan[0]
    if not entries:
        return

    from textual._on import OnNoWidget
    from textual.css.match import match
    from textual.widget import Widget

    methods_dispatched: set = set()
    for cls, handlers, method in entries:
        if message._no_default_action:
            break
        for handler, selectors in handlers:
            if handler in methods_dispatched:
                continue
            if not selectors:
                yield cls, handler.__get__(self, cls)
                methods_dispatched.add(handler)
            else:
                if not message._sender:
                    continue
                for attribute, selector in selectors.items():
                    node = getattr(message, attribute)
                    if node is None:
                        break
                    if not isinstance(node, Widget):
                        raise OnNoWidget(
                            f"on decorator can't match against {attribute!r} as it is not a widget."
                        )
                    if not match(selector, node):
                        break
                else:
                    yield cls, handler.__get__(self, cls)
                    methods_dispatched.add(handler)
        if method is not None:
            yield cls, method.__get__(self, cls)


def _forgetting(original: Any) -> Any:
    """Wrap ``type.__setattr__``/``__delattr__`` to drop plans a patch may change."""

    def changed(cls: type, name: str, *value: Any) -> None:
        if _plans and name in _watched_names:
            _plans.clear()
        return original(cls, name, *value)

    setattr(changed, _MARKER, True)
    return changed


def install_dispatch_fastpath() -> bool:
    """Install both fast paths. Idempotent.

    Returns:
        ``True`` if this call installed them, ``False`` if an earlier call did.
    """
    upstream_dispatch = MessagePump._get_dispatch_methods
    if getattr(upstream_dispatch, _MARKER, False):
        return False
    upstream_count = _callback.count_parameters
    upstream_total = _callback._count_parameters

    def count_parameters(func: Any) -> int:
        try:
            return func._param_count
        except AttributeError:
            pass
        if type(func) is partial and type(func.func) is FunctionType:
            target = func.func
            total = target.__dict__.get(_TOTAL)
            if total is None:
                total = upstream_total(target)
                target.__dict__[_TOTAL] = total
            param_count = total - (len(func.args) + len(func.keywords))
            try:
                func._param_count = param_count
            except TypeError:
                pass
            return param_count
        return upstream_count(func)

    setattr(count_parameters, _MARKER, True)
    _upstream["count_parameters"] = upstream_count
    _callback.count_parameters = count_parameters

    setattr(_fast_get_dispatch_methods, _MARKER, True)
    _upstream["get_dispatch_methods"] = upstream_dispatch
    MessagePump._get_dispatch_methods = _fast_get_dispatch_methods  # type: ignore[method-assign]
    _MessagePumpMeta.__setattr__ = _forgetting(type.__setattr__)  # type: ignore[method-assign]
    _MessagePumpMeta.__delattr__ = _forgetting(type.__delattr__)  # type: ignore[method-assign]
    return True


def upstream(name: str) -> Any:
    """The upstream callable a fast path replaced (``"count_parameters"`` or
    ``"get_dispatch_methods"``), for tests that compare the two."""
    return _upstream[name]


def is_installed() -> bool:
    """Whether the fast paths are currently installed."""
    return bool(getattr(MessagePump._get_dispatch_methods, _MARKER, False))

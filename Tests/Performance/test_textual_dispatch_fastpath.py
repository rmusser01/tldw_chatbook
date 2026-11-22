"""Guards for the message-pump fast paths (``Utils/textual_dispatch_fastpath.py``).

TASK-40608: every message a widget processes looked its handlers up by walking
the widget's MRO (twice, counting the ``Idle`` event that follows), and every
``call_later`` callback ran ``inspect.signature``. On the 558-widget Console one
``Pilot._wait_for_screen`` pass cost 34.5 ms of UI-thread CPU. The fast paths
cache the lookup and keep the callback's arity on the function it wraps.

Two things have to stay true for that to be safe, and each has tests here:

1. **Dispatch must not change.** The cached lookup and the arity are compared
   with upstream's own results over handlers found by naming convention, by
   ``@on`` (with and without selectors), on plain mixins, across message
   subclasses and with ``prevent_default``; and a handler patched onto a class
   after the first dispatch is seen by the next one.
2. **Upstream must still be the code the fast paths were derived from.** The
   source of the replaced functions is pinned, so a Textual upgrade fails here
   and the fast paths are re-reviewed instead of silently changing dispatch.
"""

from __future__ import annotations

import functools
import hashlib
import inspect

import pytest
import textual._callback as textual_callback
from textual import on
from textual.app import App, ComposeResult
from textual.message import Message
from textual.message_pump import MessagePump, _MessagePumpMeta
from textual.widgets import Button, Static

from tldw_chatbook.Utils import textual_dispatch_fastpath as fastpath

fastpath.install_dispatch_fastpath()

_FAST_DISPATCH = MessagePump._get_dispatch_methods
_UPSTREAM_DISPATCH = fastpath.upstream("get_dispatch_methods")
_FAST_COUNT = textual_callback.count_parameters
_UPSTREAM_COUNT = fastpath.upstream("count_parameters")

#: sha256 of ``inspect.getsource`` for Textual 8.2.8.
UPSTREAM_SOURCE = {
    "MessagePump._get_dispatch_methods": (
        _UPSTREAM_DISPATCH,
        "42cf0aeddecf314f6e7557e1aee0003f91b0aec2a8309a5c9595e8a98f94e073",
    ),
    "count_parameters": (
        _UPSTREAM_COUNT,
        "1cfb1fbbe594186f829b7f7114f79d55ac504c809fc7291b1632016c8d0b3d18",
    ),
    "_count_parameters": (
        textual_callback._count_parameters,
        "645475a994d72349717663a6f72456b3b8eb540dca0ebffcf28cde8bf988b9e9",
    ),
    "Widget._check_refresh": (
        fastpath.upstream("check_refresh"),
        "ad1c8196be9747bff64e8e37a934439fcfd8160ed02c5127660989428d5711a6",
    ),
    "_MessagePumpMeta.__new__": (
        _MessagePumpMeta.__new__,
        "c261e963e7ae4079096492e203b2abdab876c6f2996aff1920d8b7bd88f5b442",
    ),
}


def test_the_fast_paths_are_installed_once():
    assert fastpath.is_installed()
    assert fastpath.install_dispatch_fastpath() is False
    assert MessagePump._get_dispatch_methods is _FAST_DISPATCH


@pytest.mark.parametrize("name", sorted(UPSTREAM_SOURCE))
def test_upstream_is_still_the_source_the_fast_paths_were_derived_from(name):
    """A Textual upgrade that changes dispatch or arity must be re-reviewed.

    ``_MessagePumpMeta.__new__`` is pinned because the cached plan relies on
    ``_decorated_handlers`` being built there, once, when a class is created.
    """
    function, expected = UPSTREAM_SOURCE[name]
    actual = hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
    assert actual == expected, (
        f"{name} changed upstream; review tldw_chatbook/Utils/"
        "textual_dispatch_fastpath.py against the new source, then update this pin"
    )


# --- callback arity ---------------------------------------------------------


class _Callbacks:
    def none(self):
        return None

    def one(self, a):
        return a

    def defaults(self, a, b=1, *, c=2):
        return a, b, c

    def star(self, *args, **kwargs):
        return args, kwargs

    def only_star(*args):  # noqa: N805 - binding removes no parameter
        return args

    async def coroutine(self, a, b):
        return a, b


def _plain(a, b, /, c, *, d=None):
    return a, b, c, d


def _wrapped_two(a, b):
    return a, b


@functools.wraps(_wrapped_two)
def _wrapper(*args, **kwargs):
    return _wrapped_two(*args, **kwargs)


class _CallableObject:
    def __call__(self, a, b, c):
        return a, b, c


def _callables():
    """Fresh callables each time: both implementations cache on the object."""
    owner = _Callbacks()
    return {
        "function": lambda: _plain,
        "lambda-none": lambda: (lambda: None),
        "lambda-two": lambda: (lambda a, b: None),
        "bound-none": lambda: owner.none,
        "bound-one": lambda: owner.one,
        "bound-defaults": lambda: owner.defaults,
        "bound-star": lambda: owner.star,
        "bound-only-star": lambda: owner.only_star,
        "bound-coroutine": lambda: owner.coroutine,
        "partial-function": lambda: functools.partial(_plain, 1),
        "partial-function-two": lambda: functools.partial(_plain, 1, 2),
        "partial-keyword": lambda: functools.partial(_plain, d=4),
        "partial-lambda": lambda: functools.partial(lambda a, b: None),
        "partial-bound": lambda: functools.partial(owner.one),
        "partial-bound-arg": lambda: functools.partial(owner.defaults, 1),
        "partial-bound-star": lambda: functools.partial(owner.star, 1, 2),
        "partial-bound-only-star": lambda: functools.partial(owner.only_star, 1),
        "partial-of-partial": lambda: functools.partial(functools.partial(_plain, 1), 2),
        "partial-wrapper": lambda: functools.partial(_wrapper),
        "partial-builtin": lambda: functools.partial(divmod, 7),
        "partial-callable-object": lambda: functools.partial(_CallableObject(), 1),
        "partial-class": lambda: functools.partial(_CallableObject),
        "wrapper": lambda: _wrapper,
        "callable-object": lambda: _CallableObject(),
    }


def _count_or_error(count, target):
    try:
        return count(target)
    except Exception as error:  # noqa: BLE001 - the error type is the result
        return type(error)


def _forget_counts(target):
    """Both implementations cache on the callable; clear so each computes."""
    for cached in (target, getattr(target, "func", None), getattr(target, "__func__", None)):
        for name in ("_param_count", fastpath._TOTAL):
            try:
                delattr(cached, name)
            except (AttributeError, TypeError):
                pass


@pytest.mark.parametrize("name", sorted(_callables()))
def test_callback_arity_matches_upstream(name):
    make = _callables()[name]
    target = make()
    _forget_counts(target)
    expected = _count_or_error(_UPSTREAM_COUNT, target)
    fresh = make()
    _forget_counts(fresh)
    assert _count_or_error(_FAST_COUNT, fresh) == expected
    again = make()  # a new partial around a function that now carries its count
    try:
        del again._param_count
    except (AttributeError, TypeError):
        pass
    assert _count_or_error(_FAST_COUNT, again) == expected


def test_partials_around_one_function_share_one_signature_call(monkeypatch):
    calls = []
    count = textual_callback._count_parameters
    monkeypatch.setattr(
        textual_callback, "signature", lambda *a, **k: calls.append(a) or inspect.signature(*a, **k)
    )
    assert count is fastpath.upstream("count_parameters").__globals__["_count_parameters"]

    def decrement(counter, step=1):
        return None

    assert [_FAST_COUNT(functools.partial(decrement)) for _ in range(50)] == [2] * 50
    assert _FAST_COUNT(functools.partial(decrement, 1)) == 1
    assert len(calls) == 1, "every partial paid inspect.signature again"


# --- handler lookup ----------------------------------------------------------


class Ping(Message):
    pass


class LoudPing(Ping):
    pass


class _PlainMixin:
    """Not a message pump class: its namespace is re-checked on every call."""

    def on_ping(self, message):
        return "mixin"


class Base(Static):
    def on_ping(self, message):
        return "base"

    def _on_loud_ping(self, message):
        return "base-private"

    @on(Ping)
    def decorated_base(self, message):
        return "decorated-base"


class Child(_PlainMixin, Base):
    def _on_ping(self, message):
        return "child-private"

    def on_ping(self, message):  # shadowed by the private name, as upstream
        return "child-public"

    @on(LoudPing)
    def decorated_child(self, message):
        return "decorated-child"

    @on(Button.Pressed, "#go")
    def decorated_selector(self, message):
        return "selector"


class Preventing(Child):
    def on_ping(self, message):
        message.prevent_default()
        return "preventing"


def _dispatch(function, pump, name, message):
    return [(cls, method.__func__, method.__self__) for cls, method in function(pump, name, message)]


def _both(pump, name, make_message):
    expected = _dispatch(_UPSTREAM_DISPATCH, pump, name, make_message())
    actual = _dispatch(_FAST_DISPATCH, pump, name, make_message())
    again = _dispatch(_FAST_DISPATCH, pump, name, make_message())  # from the cached plan
    assert actual == expected
    assert again == expected
    return expected


@pytest.mark.parametrize("pump_class", [Static, Base, Child, Preventing, Button])
@pytest.mark.parametrize(
    ("name", "message_class"),
    [
        ("on_ping", Ping),
        ("on_ping", LoudPing),
        ("on_loud_ping", LoudPing),
        ("on_idle", Ping),
        ("on_callback", Ping),
        ("on_missing", Ping),
    ],
)
def test_handler_lookup_matches_upstream(pump_class, name, message_class):
    pump = pump_class("x")
    found = _both(pump, name, message_class)
    if pump_class is Child and name == "on_ping" and message_class is Ping:
        names = [function.__name__ for _, function, _ in found]
        assert names == ["_on_ping", "on_ping", "decorated_base", "on_ping"]


def test_prevent_default_stops_later_classes_as_upstream():
    pump = Preventing("x")

    def consume(function):
        message = Ping()
        seen = []
        for cls, method in function(pump, "on_ping", message):
            seen.append((cls.__name__, method.__name__))
            method(message)
        return seen

    expected = consume(_UPSTREAM_DISPATCH)
    assert consume(_FAST_DISPATCH) == expected
    assert expected == [("Preventing", "on_ping")]


def test_selector_handlers_match_upstream():
    pump = Child("x")
    for button_id, sender in (("go", True), ("stop", True), ("go", False)):
        button = Button("b", id=button_id)

        def make():
            message = Button.Pressed(button)
            message._sender = button if sender else None
            return message

        found = _both(pump, "on_button_pressed", make)
        names = [function.__name__ for _, function, _ in found]
        assert ("decorated_selector" in names) == (button_id == "go" and sender)


def test_a_handler_patched_onto_a_pump_class_is_seen(monkeypatch):
    pump = Child("x")
    before = _both(pump, "on_ping", Ping)

    def patched(self, message):
        return "patched"

    monkeypatch.setattr(Base, "on_ping", patched)
    after = _both(pump, "on_ping", Ping)
    assert patched in [function for _, function, _ in after]
    assert after != before
    monkeypatch.undo()
    assert _both(pump, "on_ping", Ping) == before


def test_a_handler_patched_onto_a_plain_mixin_is_seen(monkeypatch):
    pump = Child("x")
    before = _both(pump, "on_ping", Ping)

    def patched(self, message):
        return "patched"

    monkeypatch.setattr(_PlainMixin, "on_ping", patched)
    after = _both(pump, "on_ping", Ping)
    assert patched in [function for _, function, _ in after]
    monkeypatch.delattr(_PlainMixin, "on_ping")
    removed = _both(pump, "on_ping", Ping)
    assert "_PlainMixin" not in [cls.__name__ for cls, _, _ in removed]
    monkeypatch.undo()
    assert _both(pump, "on_ping", Ping) == before


def test_a_handler_added_and_deleted_on_a_pump_class_is_seen():
    pump = Base("x")

    def by_convention():
        # ``@on(Ping)`` handlers match whatever the handler name is.
        return [f for _, f, _ in _both(pump, "on_extra", Ping) if f.__name__ == "on_extra"]

    assert by_convention() == []

    def on_extra(self, message):
        return "extra"

    Base.on_extra = on_extra
    try:
        assert by_convention() == [on_extra]
    finally:
        del Base.on_extra
    assert by_convention() == []


# --- idle refresh check -----------------------------------------------------


class _Flags:
    """Stands in for a widget: upstream is replaced by a recorder."""

    _layout_required = _repaint_required = False
    _scroll_required = _refresh_styles_required = False


@pytest.mark.parametrize(
    "flag",
    [None, "_layout_required", "_repaint_required", "_scroll_required", "_refresh_styles_required"],
)
def test_the_idle_check_calls_upstream_exactly_when_a_refresh_flag_is_set(monkeypatch, flag):
    from textual.widget import Widget

    calls = []
    fast = Widget._check_refresh
    cell = next(
        cell for cell in fast.__closure__ if cell.cell_contents is fastpath.upstream("check_refresh")
    )
    monkeypatch.setattr(cell, "cell_contents", calls.append)
    widget = _Flags()
    if flag is not None:
        setattr(widget, flag, True)
    fast(widget)
    assert calls == ([] if flag is None else [widget])


def test_upstream_idle_check_reads_only_the_four_flags():
    """The early exit is only right while upstream acts on these flags alone."""
    source = inspect.getsource(fastpath.upstream("check_refresh"))
    acted_on = sorted(
        line.strip()[len("if self.") : -1]
        for line in source.splitlines()
        if line.strip().startswith("if self._") and line.strip().endswith("_required:")
    )
    assert acted_on == [
        "_layout_required",
        "_refresh_styles_required",
        "_repaint_required",
        "_scroll_required",
    ]


# --- the running app ----------------------------------------------------------


class _DispatchApp(App):
    def __init__(self):
        super().__init__()
        self.events = []

    def compose(self) -> ComposeResult:
        yield Child("child", id="child")
        yield Button("Go", id="go")
        for index in range(40):
            yield Static(str(index))

    @on(Button.Pressed, "#go")
    def pressed(self, message):
        self.events.append("pressed")

    def on_ping(self, message):
        self.events.append("ping")


@pytest.mark.asyncio
async def test_messages_callbacks_and_the_pilot_barrier_still_work():
    app = _DispatchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        called = []
        for widget in app.screen.walk_children():
            widget.call_later(called.append, widget)
        app.post_message(Ping())
        await pilot.click("#go")
        await pilot.pause()
        assert len(called) == len(list(app.screen.walk_children()))
        assert app.events.count("ping") == 1
        assert app.events.count("pressed") == 1
        assert await pilot._wait_for_screen()
        # A refresh requested after the fast idle check is still painted.
        child = app.query_one("#child")
        child.update("changed")
        await pilot.pause()
        assert not child._repaint_required and not child._layout_required

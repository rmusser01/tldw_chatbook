"""Characterisation of the Console controller wiring (decomposition wave 4).

Written BEFORE wave 4 task 1 moved the six `Console*Controller(...)`
constructions out of `ChatScreen.__init__` and into
`UI/Console_Modules/wiring.py`. Every assertion here is *behavioural* -- it
pins what the wiring DOES, never where the source lines live -- so this file
must stay byte-identical across that move. If the extraction needs one of
these to change, the extraction changed behaviour: stop and treat it as a
finding.

Three properties are pinned, each the thing a "pure move" is most likely to
break silently:

1. **Presence, class, and relative construction order.** `vars(screen)`
   preserves assignment order in CPython, so the six controller slots' order
   in the instance dict is a direct observation of the order `__init__` (now
   `build_console_controllers`) built them in. Only the six's order relative
   to each other is pinned -- unrelated attributes around them are free to
   move.
2. **Dependency identity.** Every named dependency is wired as a
   late-binding lambda closing over the screen, so calling the stored
   callable must return the *identical* object the screen's own method
   returns. "is not None" would pass against a lambda rewired to a different
   (but similarly-shaped) source; `is` would not.
3. **Late binding, including across controllers.** Replacing the target on
   the screen *instance* after construction must be observed by the stored
   callable -- that is the binding rule's whole point (see
   `ConsoleDictationController.__init__`'s docstring, the canonical
   statement of it). The cross-controller cases additionally prove the
   sibling is resolved at CALL time, which is why construction order among
   the six cannot matter.
"""

from types import SimpleNamespace

import pytest

from Tests.UI.test_destination_shells import _build_test_app
from tldw_chatbook.UI.Console_Modules.dictation import ConsoleDictationController
from tldw_chatbook.UI.Console_Modules.hands_free import ConsoleHandsFreeController
from tldw_chatbook.UI.Console_Modules.message import ConsoleMessageController
from tldw_chatbook.UI.Console_Modules.prompts import ConsolePromptsController
from tldw_chatbook.UI.Console_Modules.retrieval import ConsoleRetrievalController
from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController
from tldw_chatbook.UI.Console_Modules.skill import ConsoleSkillController
from tldw_chatbook.UI.Console_Modules.workspace import ConsoleWorkspaceController
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

#: (screen attribute, controller class), in the order the wiring builds them.
_EXPECTED_SLOTS: list[tuple[str, type]] = [
    ("_workspace", ConsoleWorkspaceController),
    ("_session", ConsoleSessionController),
    ("_dictation", ConsoleDictationController),
    ("_hands_free", ConsoleHandsFreeController),
    ("_message", ConsoleMessageController),
    ("_prompts", ConsolePromptsController),
]

#: Every controller takes `chat_store_accessor=lambda: self._ensure_console_
#: chat_store()`; it is stored as `_chat_store_accessor`. The one dependency
#: shared by all six, so it is the cheapest uniform identity probe.
_SHARED_ACCESSOR = "_chat_store_accessor"


def _unmounted_console() -> ChatScreen:
    """A real, unmounted `ChatScreen`.

    The wiring under test runs entirely in `__init__`, so no pilot/mount is
    needed -- this mirrors the `ChatScreen(app)` idiom already used across
    `test_console_internals_decomposition.py`,
    `test_console_message_controller.py` and friends.
    """
    return ChatScreen(_build_test_app())


def test_all_six_controllers_are_constructed_with_the_right_classes():
    screen = _unmounted_console()

    for attr, cls in _EXPECTED_SLOTS:
        controller = getattr(screen, attr, None)
        assert controller is not None, f"{attr} was never wired"
        assert isinstance(controller, cls), (
            f"{attr} is {type(controller).__name__}, expected {cls.__name__}"
        )


def test_retrieval_controller_is_constructed_with_late_bound_screen_edges():
    """Wave 6 wires one retrieval owner without freezing screen state."""
    screen = _unmounted_console()

    assert isinstance(screen._retrieval, ConsoleRetrievalController)
    sentinel = object()
    screen._current_console_rail_conversation_id = lambda: sentinel
    assert screen._retrieval._current_conversation_id() is sentinel


def test_skill_controller_is_constructed_with_late_bound_screen_edges():
    """Wave 6 wires `_skill` without changing the original six-slot contract."""
    screen = _unmounted_console()

    assert isinstance(screen._skill, ConsoleSkillController)
    sentinel = object()
    screen._task_resume_state = sentinel
    assert screen._skill._task_resume_state() is sentinel

    calls: list[str] = []
    screen._sync_console_command_popup = lambda: calls.append("replacement")
    screen._skill._sync_console_command_popup()
    assert calls == ["replacement"]


def test_controllers_are_built_in_the_documented_order():
    """The six slots appear in `vars(screen)` in build order.

    Order is load-bearing documentation (every cross-controller lambda
    resolves its sibling at call time precisely so it *need not* be
    load-bearing behaviour) -- pin it so a reshuffle is a deliberate,
    visible act rather than a silent side effect of moving the block.
    """
    screen = _unmounted_console()

    names = [attr for attr, _ in _EXPECTED_SLOTS]
    observed = [key for key in vars(screen) if key in set(names)]

    assert observed == names, f"controller build order changed: {observed}"


def test_every_controller_holds_the_same_app_instance_as_the_screen():
    screen = _unmounted_console()

    for attr, _ in _EXPECTED_SLOTS:
        controller = getattr(screen, attr)
        assert controller.app_instance is screen.app_instance, (
            f"{attr}.app_instance is not the screen's app_instance"
        )


def test_shared_chat_store_accessor_resolves_to_the_screens_own_store():
    """Identity, not truthiness: all six must reach the SAME store object."""
    screen = _unmounted_console()

    expected = screen._ensure_console_chat_store()
    assert expected is not None, "screen produced no chat store to compare against"

    for attr, _ in _EXPECTED_SLOTS:
        accessor = getattr(getattr(screen, attr), _SHARED_ACCESSOR)
        assert accessor() is expected, (
            f"{attr}.{_SHARED_ACCESSOR}() is not the screen's chat store"
        )


def test_shared_chat_store_accessor_is_late_bound():
    """An instance-level replacement made AFTER construction must be seen."""
    screen = _unmounted_console()
    sentinel = object()
    screen._ensure_console_chat_store = lambda: sentinel

    for attr, _ in _EXPECTED_SLOTS:
        accessor = getattr(getattr(screen, attr), _SHARED_ACCESSOR)
        assert accessor() is sentinel, (
            f"{attr}.{_SHARED_ACCESSOR} froze the constructor-time method"
        )


@pytest.mark.parametrize(
    "attr,accessor_name",
    [
        ("_session", "_composer_accessor"),
        ("_dictation", "_composer_accessor"),
        ("_hands_free", "_composer_accessor"),
        ("_prompts", "_composer_accessor"),
    ],
)
def test_composer_accessor_is_late_bound(attr, accessor_name):
    screen = _unmounted_console()
    sentinel = object()
    screen._console_composer_or_none = lambda: sentinel

    accessor = getattr(getattr(screen, attr), accessor_name)
    assert accessor() is sentinel, f"{attr}.{accessor_name} froze at construction"


@pytest.mark.parametrize(
    "attr,accessor_name",
    [
        ("_workspace", "_current_chat_store_accessor"),
        ("_session", "_current_chat_store_accessor"),
        ("_message", "_current_chat_store_accessor"),
    ],
)
def test_current_chat_store_accessor_reads_the_live_attribute(attr, accessor_name):
    """`lambda: self._console_chat_store` -- a bare attribute read, late."""
    screen = _unmounted_console()
    sentinel = object()
    screen._console_chat_store = sentinel

    accessor = getattr(getattr(screen, attr), accessor_name)
    assert accessor() is sentinel, f"{attr}.{accessor_name} snapshotted the value"


def test_workspace_resolves_the_session_sibling_at_call_time():
    """`_workspace` is built BEFORE `_session`; the lambda must still work.

    Swapping `screen._session` wholesale after construction and seeing the
    workspace accessor follow is the proof that the sibling is looked up per
    call -- the property that makes the six constructions order-independent.
    """
    screen = _unmounted_console()
    sentinel = object()
    screen._session = SimpleNamespace(
        _current_console_conversation_id=lambda: sentinel
    )

    assert screen._workspace._current_conversation_id_accessor() is sentinel


def test_dictation_resolves_the_hands_free_sibling_at_call_time():
    """`_dictation` is built BEFORE `_hands_free` -- same proof, other way."""
    screen = _unmounted_console()
    sentinel = object()
    screen._hands_free = SimpleNamespace(_console_hands_free=sentinel)

    assert screen._dictation._hands_free_session_accessor() is sentinel


def test_dictation_session_late_binds_the_app_owned_service_factory():
    """The session must follow app factory replacement after screen wiring."""
    screen = _unmounted_console()
    first = object()
    second = object()
    calls: list[tuple[object, dict]] = []

    def first_factory(**kwargs):
        calls.append((first, kwargs))
        return first

    def second_factory(**kwargs):
        calls.append((second, kwargs))
        return second

    screen.app_instance._create_console_dictation_service = first_factory
    session = screen._dictation._create_console_dictation_session()
    assert session._build_service(language="en") is first

    screen.app_instance._create_console_dictation_service = second_factory
    assert session._build_service(language="fr") is second
    assert calls == [
        (
            first,
            {
                "language": "en",
                "max_buffer_bytes": 1_920_000,
            },
        ),
        (
            second,
            {
                "language": "fr",
                "max_buffer_bytes": 1_920_000,
            },
        ),
    ]


def test_hands_free_reads_dictation_state_through_the_screen_at_call_time():
    screen = _unmounted_console()
    sentinel = object()
    screen._dictation = SimpleNamespace(_console_dictation_state=sentinel)

    assert screen._hands_free._dictation_state_accessor() is sentinel


def test_prompts_resolves_the_session_sibling_at_call_time():
    screen = _unmounted_console()
    sentinel = object()
    screen._session = SimpleNamespace(
        _ensure_active_console_session_settings=lambda: sentinel
    )

    assert (
        screen._prompts._ensure_active_console_session_settings_fn() is sentinel
    )

"""``copy_console_message`` is ``dataclasses.replace(message)``, only faster.

TASK-33628.5.2: the store snapshots every active-path message several times
per post-action Console sync, and ``replace`` re-runs the ~50-field
``__init__`` for each one (about 0.24 s per sync at 3,000 messages). The
fast copy is exact only while ``ConsoleChatMessage`` has no ``__post_init__``
and no ``init=False`` field; these tests fail the day either changes.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
    ConsoleVariant,
    ConsoleVariantSet,
    copy_console_message,
)


def _message() -> ConsoleChatMessage:
    return ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="body",
        persisted_message_id="p1",
        parent_message_id="p0",
        sibling_index=1,
        sibling_count=3,
        variants=ConsoleVariantSet(
            turn_id="t1",
            variants=[ConsoleVariant(content="a"), ConsoleVariant(content="b")],
            selected_index=1,
        ),
        image_data=b"\x89PNG",
        status="streaming",
        live_activity="read_file",
    )


def test_the_copy_equals_replace_and_is_a_new_object() -> None:
    message = _message()

    copied = copy_console_message(message)

    assert copied == replace(message)
    assert copied is not message
    assert type(copied) is ConsoleChatMessage
    # Shallow, exactly like replace: field values are shared, not copied.
    assert copied.variants is message.variants
    assert copied.image_data is message.image_data


def test_the_copy_holds_the_fields_and_nothing_else() -> None:
    message = _message()
    message.__dict__["_not_a_field"] = object()

    copied = copy_console_message(message)

    assert set(copied.__dict__) == {f.name for f in fields(ConsoleChatMessage)}
    copied.content = "changed"
    assert message.content == "body"


def test_console_chat_message_keeps_the_shape_the_fast_copy_needs() -> None:
    assert not hasattr(ConsoleChatMessage, "__post_init__")
    assert all(f.init for f in fields(ConsoleChatMessage))


def test_a_class_with_post_init_goes_through_replace() -> None:
    @dataclass
    class Checked(ConsoleChatMessage):
        def __post_init__(self) -> None:
            self.content = self.content.strip()

    message = Checked(role=ConsoleMessageRole.USER, content=" padded ")
    message.content = " again "

    copied = copy_console_message(message)

    assert type(copied) is Checked
    assert copied.content == "again"

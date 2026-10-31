"""A rendered terminal receipt is acknowledged once, not on every refresh (TASK-33620.15.1).

Measured on the base build 46c3959526 (live, Anthropic haiku, 160x45): during
a run the transcript refreshes on most 0.2 s sync ticks, and every refresh
acknowledged every rendered terminal receipt again -- one DELETE transaction
per earlier turn of the chat, on the UI loop, through the database's storage
admission. After TASK-33620.15.1's other cuts it was three quarters of the
transcript sync's main-thread time, and it grows with the chat.

A receipt whose acknowledgement already completed is skipped while the
runtime's last durable read of the marks does not list it. A mark that the
runtime learns of later (written after an earlier, no-op acknowledgement) is
acknowledged on the next render, and with no successful mark read yet every
render acknowledges, as before.
"""

from __future__ import annotations

from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

RECEIPT_A = "11111111-1111-4111-8111-111111111111"
RECEIPT_B = "22222222-2222-4222-8222-222222222222"


class _Marks:
    def __init__(self, pairs=()) -> None:
        self.pairs = list(pairs)
        self.calls = 0
        self.read_error: Exception | None = None

    def list_console_unseen_marks(self):
        if self.read_error is not None:
            raise self.read_error
        return tuple(self.pairs)

    def acknowledge_console_unseen(self, conversation_id, receipt_id) -> bool:
        self.calls += 1
        pair = (conversation_id, receipt_id)
        if pair not in self.pairs:
            return False
        self.pairs.remove(pair)
        return True


class _App:
    def __init__(self, marks: _Marks) -> None:
        self.conversation_local_marks_service = marks
        self.console_attention_updates: list[bool] = []
        self.chachanotes_db = None

    def set_console_attention_projection(self, value: bool) -> None:
        self.console_attention_updates.append(value)


RENDERED = (("conv-a", RECEIPT_A), ("conv-a", RECEIPT_B))


def test_a_rendered_receipt_is_acknowledged_once_until_its_mark_is_known():
    marks = _Marks((("conv-a", RECEIPT_A),))
    runtime = ConsoleRuntime(_App(marks))
    runtime.recompute_console_attention()

    assert runtime.acknowledge_rendered_terminal_receipts(RENDERED) == (RECEIPT_A,)
    first = marks.calls
    for _ in range(5):
        assert runtime.acknowledge_rendered_terminal_receipts(RENDERED) == ()
    assert marks.calls == first, (
        f"{marks.calls - first} repeat acknowledgements in 5 refreshes"
    )

    # B's mark lands after its no-op acknowledgement; once the runtime has
    # read it, the next render acknowledges it.
    marks.pairs.append(("conv-a", RECEIPT_B))
    runtime.recompute_console_attention()
    assert runtime.acknowledge_rendered_terminal_receipts(RENDERED) == (RECEIPT_B,)
    assert marks.pairs == []


def test_without_a_successful_mark_read_every_render_acknowledges():
    marks = _Marks()
    marks.read_error = RuntimeError("marks unavailable")
    runtime = ConsoleRuntime(_App(marks))
    runtime.recompute_console_attention()

    runtime.acknowledge_rendered_terminal_receipts(RENDERED)
    runtime.acknowledge_rendered_terminal_receipts(RENDERED)
    assert marks.calls == 4


def test_a_failed_acknowledgement_is_retried_on_the_next_render():
    marks = _Marks((("conv-a", RECEIPT_A),))
    runtime = ConsoleRuntime(_App(marks))
    runtime.recompute_console_attention()
    real = marks.acknowledge_console_unseen
    failures = [1]

    def flaky(conversation_id, receipt_id):
        if failures[0]:
            failures[0] -= 1
            marks.calls += 1
            raise RuntimeError("database busy")
        return real(conversation_id, receipt_id)

    marks.acknowledge_console_unseen = flaky
    assert runtime.acknowledge_rendered_terminal_receipts(RENDERED[:1]) == ()
    assert runtime.acknowledge_rendered_terminal_receipts(RENDERED[:1]) == (RECEIPT_A,)

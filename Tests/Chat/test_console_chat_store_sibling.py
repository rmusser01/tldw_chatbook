"""``create_sibling`` primitive tests (Phase A, Task 5).

Pins the regenerate-as-sibling primitive: forking a new node alongside an
existing one (sharing its native parent) rather than beneath it, then making
that new node the active leaf so the visible transcript follows the new
branch. Also exercises the "swipe back" recovery path via ``set_active_leaf``
that the mid-conversation case depends on.
"""

from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole


def _s():
    st = ConsoleChatStore()
    ses = st.create_session(title="t")
    st.active_session_id = ses.id
    return st, ses.id


def test_create_sibling_of_last_assistant_makes_two_children():
    st, sid = _s()
    u = st.append_message(sid, role=ConsoleMessageRole.USER, content="q")
    a = st.append_message(sid, role=ConsoleMessageRole.ASSISTANT, content="ans-1")
    a2 = st.create_sibling(a.id, role=ConsoleMessageRole.ASSISTANT, content="ans-2")
    sibs, idx, count = st.siblings_at(a2.id)
    assert count == 2 and idx == 1
    assert st.active_leaf(sid) == a2.id
    assert st.active_path_message_ids(sid) == [u.id, a2.id]  # tail is the new sibling


def test_create_sibling_midconversation_truncates_visible_tail():
    st, sid = _s()
    u1 = st.append_message(sid, role=ConsoleMessageRole.USER, content="q1")
    a1 = st.append_message(sid, role=ConsoleMessageRole.ASSISTANT, content="a1")
    u2 = st.append_message(sid, role=ConsoleMessageRole.USER, content="q2")
    a2 = st.append_message(sid, role=ConsoleMessageRole.ASSISTANT, content="a2")
    # regenerate a1 (mid-conversation) -> new sibling under u1, tail (u2,a2) drops off-path
    a1b = st.create_sibling(a1.id, role=ConsoleMessageRole.ASSISTANT, content="a1-alt")
    assert st.active_path_message_ids(sid) == [u1.id, a1b.id]
    # swiping back to a1 restores the old tail
    st.set_active_leaf(sid, a2.id)
    assert st.active_path_message_ids(sid) == [u1.id, a1.id, u2.id, a2.id]

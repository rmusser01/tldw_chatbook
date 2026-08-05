from types import SimpleNamespace

from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.Utils.token_counter import estimate_tokens


def test_estimate_tokens_uses_shared_estimator_no_split():
    # Call the unbound method with a minimal self; it must delegate to estimate_tokens,
    # never a .split() word count.
    self_stub = SimpleNamespace()
    text = "def f(x): return x*x  # a code-ish draft with symbols"
    result = ChatScreen._estimate_tokens(self_stub, {"draft": text})
    assert result == estimate_tokens(text, "", "")
    # A word-count would be far lower than the chars-based estimate.
    assert result != len(text.split())


def test_estimate_tokens_none_for_empty_draft():
    self_stub = SimpleNamespace()
    assert ChatScreen._estimate_tokens(self_stub, {"draft": ""}) is None

"""Resume navigation keeps the shared exact-identity contract."""

import pytest

from tldw_chatbook.UI.Navigation.pending_handoff_store import (
    ConsoleConversationResumeIntent,
)
from tldw_chatbook.Utils.input_validation import validate_conversation_resume_id


@pytest.mark.parametrize(
    "value", [None, 1, True, b"chat", "", " ", " chat", "chat ", "x" * 257]
)
def test_resume_identity_rejects_invalid_boundary_values(value):
    with pytest.raises(ValueError):
        validate_conversation_resume_id(value)
    with pytest.raises(ValueError):
        ConsoleConversationResumeIntent(value)


@pytest.mark.parametrize("value", ["chat-1", "x" * 256, "exact Unicode é"])
def test_resume_identity_preserves_valid_spelling(value):
    assert validate_conversation_resume_id(value) == value
    assert ConsoleConversationResumeIntent(value).conversation_id == value

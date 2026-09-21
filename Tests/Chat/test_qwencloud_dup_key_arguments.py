"""QwenCloud non-streaming tool-call arguments reject duplicate JSON keys.

task-32805.5: the streaming path (`qwencloud_streaming._strict_json_loads`)
and the continuation checkpoint already reject duplicate keys; the
non-streaming `_validate_tool_calls` must agree, so the same provider's
function-call arguments are not accepted on one path and refused on the other.
These arguments drive tool execution, so a repeated key is ambiguity to reject.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Chat.Chat_Deps import ChatBadRequestError
from tldw_chatbook.LLM_Calls.qwencloud import _validate_tool_calls

pytestmark = pytest.mark.unit


def _call(arguments: str) -> list[dict[str, str]]:
    return _validate_tool_calls(
        [{"id": "call_1", "type": "function",
          "function": {"name": "do_thing", "arguments": arguments}}],
        seen_call_ids=set(),
    )


def test_well_formed_arguments_are_accepted():
    validated = _call('{"path": "a", "mode": "r"}')
    assert validated == [
        {"call_id": "call_1", "name": "do_thing",
         "arguments": '{"path": "a", "mode": "r"}'}
    ]


def test_duplicate_key_arguments_are_rejected():
    # last-wins would have surfaced mode="w"; the duplicate must be refused
    with pytest.raises(ChatBadRequestError):
        _call('{"path": "a", "mode": "r", "mode": "w"}')


def test_non_finite_constant_still_rejected():
    with pytest.raises(ChatBadRequestError):
        _call('{"temperature": NaN}')

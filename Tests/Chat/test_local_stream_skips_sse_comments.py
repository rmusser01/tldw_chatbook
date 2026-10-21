"""SSE comment frames must never surface as streamed content.

Local providers all stream through one generator in
`LLM_Calls/LLM_API_Calls_Local.py::_chat_with_openai_compatible_local_server`,
which passes raw lines through. Servers may legally send SSE comment frames --
lines whose first character is ':' (WHATWG SSE section 3) -- as heartbeats
during long generations; strata, for one, sends ": keep-alive" between chunks
while prefilling 240k-token prompts. The generator forwarded those lines
verbatim, so consumers rendered ": keep-alive" as assistant message text and
tool-call turns degraded around them. This pins the spec behavior: comment
frames are dropped, data frames and the sentinel pass through untouched.

The handler under test is driven the same way
`test_chat_functions.py::test_chat_with_llama_posts_to_v1_chat_completions_regardless_of_suffix`
drives it: the runtime config snapshot and the egress session are monkeypatched
so no user config or network is touched.
"""

from __future__ import annotations

from unittest.mock import Mock

from tldw_chatbook.LLM_Calls import LLM_API_Calls_Local
from tldw_chatbook.LLM_Calls.LLM_API_Calls_Local import (
    _chat_with_openai_compatible_local_server,
)
from tldw_chatbook.config import RuntimeConfigSnapshot


class _FakeStreamResponse:
    status_code = 200
    text = "{}"

    def __init__(self, lines):
        self._lines = lines

    def raise_for_status(self):
        return None

    def json(self):
        return {}

    def iter_lines(self, decode_unicode=False):
        for line in self._lines:
            yield line if decode_unicode else line.encode("utf-8")

    def close(self):
        return None


class _CapturedSession:
    """requests.Session stand-in that records the request and replays lines."""

    def __init__(self, response):
        self._response = response
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_exc_info):
        self.close()
        return False

    def mount(self, *_args, **_kwargs):
        return None

    def close(self):
        self.closed = True

    def post(self, url, *, headers=None, json=None, stream=False, **_kwargs):
        return self._response


def _collect_stream(lines, monkeypatch):
    monkeypatch.setattr(
        LLM_API_Calls_Local,
        "get_runtime_config_snapshot",
        lambda: RuntimeConfigSnapshot(generation=0, values={"api_settings": {}}),
    )
    monkeypatch.setattr(
        LLM_API_Calls_Local,
        "create_default_session",
        lambda: _CapturedSession(_FakeStreamResponse(lines)),
    )
    result = _chat_with_openai_compatible_local_server(
        api_base_url="http://server/v1",
        model_name="m",
        input_data=[{"role": "user", "content": "hi"}],
        api_key=None,
        streaming=True,
    )
    return [chunk for chunk in result]


def test_sse_comment_frames_are_dropped(monkeypatch):
    lines = [
        'data: {"id": "1", "choices": [{"delta": {"content": "Hi"}}]}',
        ": keep-alive",
        "",
        ": keep-alive",
        'data: {"id": "1", "choices": [{"delta": {"content": "!"}}]}',
    ]
    chunks = _collect_stream(lines, monkeypatch)
    joined = "".join(chunks)
    assert "keep-alive" not in joined
    assert joined.count('"delta"') == 2
    assert '"Hi"' in joined and '"!"' in joined
    assert chunks[-1] == "data: [DONE]\n\n"


def test_leading_colon_line_is_comment_not_content(monkeypatch):
    lines = [
        ":heartbeat",
        'data: {"ok": true}',
    ]
    chunks = _collect_stream(lines, monkeypatch)
    assert ":heartbeat" not in "".join(chunks)
    assert 'data: {"ok": true}\n\n' in chunks

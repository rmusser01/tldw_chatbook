"""Local SSE decoding must preserve UTF-8 regardless of HTTP charset hints."""

from __future__ import annotations

import io
import json

import pytest
import requests
from requests.adapters import HTTPAdapter

from tldw_chatbook.LLM_Calls import LLM_API_Calls_Local as local


class _FragmentedResponse(io.BytesIO):
    """In-memory HTTP body with transport-sized reads and real byte storage."""

    status = 200
    reason = "OK"
    _original_response = None

    def __init__(self, body: bytes, content_type: str, fragment_size: int) -> None:
        super().__init__(body)
        self.headers = {"Content-Type": content_type}
        self._fragment_size = fragment_size

    def read(self, size: int = -1) -> bytes:
        return super().read(
            self._fragment_size if size < 0 else min(size, self._fragment_size)
        )

    def release_conn(self) -> None:
        self.close()


@pytest.mark.parametrize(
    "content_type",
    [
        "text/event-stream",
        "text/event-stream; charset=utf-8",
        "text/event-stream; charset=iso-8859-1",
    ],
    ids=["missing-charset", "utf8-charset", "misleading-charset"],
)
@pytest.mark.parametrize("fragment_size", [1, 8192], ids=["one-byte", "whole-body"])
@pytest.mark.parametrize("stop_early", [False, True], ids=["exhaust", "stop"])
@pytest.mark.parametrize(
    ("reply", "ensure_ascii"),
    [
        ("Disk is tight — café → 日本語 🙂", False),
        ("Disk is tight — café → 日本語 🙂", True),
        ("Plain ASCII response.", False),
    ],
    ids=["unicode", "escaped-unicode", "ascii"],
)
def test_local_sse_preserves_text_and_closes_response(
    monkeypatch: pytest.MonkeyPatch,
    content_type: str,
    fragment_size: int,
    stop_early: bool,
    reply: str,
    ensure_ascii: bool,
) -> None:
    wire = (
        "data: "
        + json.dumps(
            {"choices": [{"delta": {"content": reply}}]}, ensure_ascii=ensure_ascii
        )
        + "\n\n"
    ).encode("utf-8")
    raw = _FragmentedResponse(wire, content_type, fragment_size)

    class FixtureAdapter(HTTPAdapter):
        def send(
            self, request: requests.PreparedRequest, **kwargs: object
        ) -> requests.Response:
            # Requests itself infers the encoding from these headers.
            return self.build_response(request, raw)

    with requests.Session() as session:
        session.trust_env = False
        session.mount("http://", FixtureAdapter())
        monkeypatch.setattr(
            local, "create_pooled_default_session", lambda **kwargs: session
        )
        stream = local._chat_with_openai_compatible_local_server(
            api_base_url="http://local.test",
            model_name="test-model",
            input_data=[{"role": "user", "content": "Hello"}],
            streaming=True,
        )
        if stop_early:
            chunks = [next(stream)]
            stream.close()
        else:
            chunks = list(stream)
            assert chunks.pop() == "data: [DONE]\n\n"

    content = "".join(
        json.loads(chunk.removeprefix("data:"))["choices"][0]["delta"]["content"]
        for chunk in chunks
        if chunk  # SSE control lines yield empty checkpoints for Stop handling.
    )
    assert content == reply
    assert raw.closed

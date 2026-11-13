"""OpenAI embedding responses must be mapped by their ``index`` field.

TASK-38590: OpenAI-compatible endpoints (vLLM / LM Studio via ``base_url``)
may return the ``data`` items of an embeddings response out of order.  The
parser used the list order, silently handing chunk ``i`` the vector of chunk
``j``; the ADR-223 persistent cache then stored the wrong vector under the
correct text hash (the cache keys ``result[i]`` by ``sha256(texts[i])``).

Hermetic by construction: ``requests.Session`` is stubbed, so no network is
touched.  The stubbing mirrors ``Tests/RAG/test_embeddings_lib_logging.py``.
"""

from __future__ import annotations

import pytest

import tldw_chatbook.Embeddings.Embeddings_Lib as embeddings_lib


class _StubResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def _install_session(monkeypatch: pytest.MonkeyPatch, payload: dict) -> None:
    """Serve ``payload`` as the JSON body of every POST from a stubbed session."""

    class _StubSession:
        def __init__(self) -> None:
            self.headers: dict = {}

        def post(self, url, *, json=None, timeout=None, allow_redirects=True):
            return _StubResponse(payload)

        def close(self) -> None:
            pass

    monkeypatch.setattr(embeddings_lib.requests, "Session", _StubSession)


def _cfg() -> embeddings_lib.OpenAICfg:
    return embeddings_lib.OpenAICfg(
        provider="openai",
        model_name_or_path="text-embedding-3-small",
        dimension=8,
        base_url="http://127.0.0.1:1/v1",  # never reached: session is stubbed
    )


@pytest.fixture
def build_embedder(monkeypatch, request):
    def build(payload):
        _install_session(monkeypatch, payload)
        embed, close = embeddings_lib._openai_embedder(_cfg())
        request.addfinalizer(close)
        return embed

    return build


def _vec(i: int) -> list[float]:
    """A vector that is unique and recognizable per request index."""
    return [float(i)] * 8


def test_out_of_order_response_maps_vectors_by_index(build_embedder):
    """A shuffled ``data`` list still yields vector ``i`` for text ``i``."""
    data = [
        {"index": 2, "embedding": _vec(2)},
        {"index": 0, "embedding": _vec(0)},
        {"index": 1, "embedding": _vec(1)},
    ]
    embed = build_embedder({"data": data})
    texts = ["alpha chunk", "beta chunk", "gamma chunk"]

    arr = embed(texts)
    assert arr.shape == (3, 8)
    # Each text must get the vector whose ``index`` matches its position in
    # the request, not the position of the item in the response list.
    for i in range(3):
        assert arr[i].tolist() == pytest.approx(_vec(i), abs=1e-7)


def test_out_of_order_response_maps_vectors_by_index_as_list(build_embedder):
    """The ``as_list=True`` path keeps the same index-based mapping."""
    data = [
        {"index": 1, "embedding": _vec(1)},
        {"index": 0, "embedding": _vec(0)},
    ]
    embed = build_embedder({"data": data})
    result = embed(["first text", "second text"], as_list=True)

    assert len(result) == 2
    for i in range(2):
        assert result[i] == pytest.approx(_vec(i), abs=1e-7)


def test_missing_index_field_falls_back_to_list_order(build_embedder):
    """Lenient boundary: without an ``index`` field, list order is kept."""
    data = [
        {"embedding": _vec(0)},
        {"embedding": _vec(1)},
    ]
    embed = build_embedder({"data": data})
    arr = embed(["text zero", "text one"])

    for i in range(2):
        assert arr[i].tolist() == pytest.approx(_vec(i), abs=1e-7)


def test_malformed_index_field_falls_back_to_list_order(build_embedder):
    """A non-sortable ``index`` (mixed types) is tolerated via list order."""
    data = [
        {"index": None, "embedding": _vec(0)},
        {"index": "1", "embedding": _vec(1)},
    ]
    embed = build_embedder({"data": data})
    arr = embed(["text zero", "text one"])

    for i in range(2):
        assert arr[i].tolist() == pytest.approx(_vec(i), abs=1e-7)

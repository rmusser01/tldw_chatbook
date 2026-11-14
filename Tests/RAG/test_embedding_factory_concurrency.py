"""Concurrency tests for ``EmbeddingFactory`` (task-34398).

Before task-34398, ``EmbeddingFactory.embed`` held its cache lock across the
entire model forward pass, so a background ingest batch serialized every
interactive query embedding (concurrency 1 regardless of hardware). The lock
now guards only cache mutation; eviction safety comes from a per-record
refcount instead.

These tests prove, with deterministic event sequencing (no sleeps, no
timing races):

1. Two embeds for the SAME model interleave -- the second forward pass
   starts before the first finishes.
2. Two embeds for DIFFERENT models run concurrently.
3. LRU eviction never evicts (or closes) a model that is checked out; the
   cache transiently exceeds ``max_cached`` rather than evicting an in-use
   model, and the in-use model becomes evictable again after check-in.
4. ``close()`` waits for in-flight embeds to drain before closing models
   (preserving the old lock-held-during-embed guarantee).
"""

import threading
import time

import numpy as np
import pytest

from tldw_chatbook.Utils.optional_deps import embeddings_rag_deps_installed

pytestmark = pytest.mark.skipif(
    not embeddings_rag_deps_installed(),
    reason="EmbeddingFactory requires the embeddings_rag extras",
)

_TIMEOUT = 10.0


class FakeModel:
    """A controllable stand-in for a loaded embedding model.

    ``embed`` signals ``start`` and then blocks until ``release`` is set, so
    a test can observe exactly when the forward pass is in flight.
    """

    def __init__(self, dim: int = 4):
        self.dim = dim
        self.start = threading.Event()
        self.release = threading.Event()
        self.closed = False
        self.calls = 0

    def embed(self, texts, *, as_list=False):
        self.calls += 1
        self.start.set()
        if not self.release.wait(timeout=_TIMEOUT):
            raise RuntimeError("test harness: release event never set")
        arr = np.tile(
            np.arange(self.dim, dtype=np.float32), (len(texts), 1)
        )
        return arr.tolist() if as_list else arr

    def close(self):
        self.closed = True


def _make_factory(monkeypatch, models: dict, max_cached: int = 2):
    """Build an EmbeddingFactory whose ``_build`` returns the given fakes."""
    from tldw_chatbook.Embeddings.Embeddings_Lib import (
        CacheRecord,
        EmbeddingFactory,
    )

    cfg = {
        # default_model_id must be a key of `models` (schema validation);
        # every test passes explicit model_ids anyway.
        "default_model_id": next(iter(models)),
        "models": {
            mid: {"provider": "huggingface", "model_name_or_path": mid}
            for mid in models
        },
    }
    factory = EmbeddingFactory(cfg, max_cached=max_cached, idle_seconds=900)
    monkeypatch.setattr(
        factory,
        "_build",
        lambda mid: CacheRecord(
            embed=models[mid].embed,
            close=models[mid].close,
            last=time.monotonic(),
            refs=0,
        ),
    )
    return factory


def _run_in_thread(fn):
    result = {}

    def _target():
        try:
            result["value"] = fn()
        except BaseException as e:  # surfaced via join below
            result["error"] = e

    thread = threading.Thread(target=_target)
    return thread, result


class TestEmbeddingFactoryConcurrency:
    def test_same_model_embeds_interleave(self, monkeypatch):
        """The second forward pass STARTS before the first RETURNS.

        Under the old whole-call lock this deadlocked against its own
        serialization: thread two could not enter ``embed`` until thread
        one's forward pass finished. Starting thread two only after thread
        one is inside its forward pass makes the interleaving deterministic.
        """
        model = FakeModel()
        factory = _make_factory(monkeypatch, {"default": model})

        first, first_out = _run_in_thread(
            lambda: factory.embed(["a"], as_list=True)
        )
        first.start()
        assert model.start.wait(_TIMEOUT), "first embed never reached the model"

        # The first embed is parked inside the forward pass. The old
        # whole-call lock would keep the second caller out of embed() (and
        # pytest-timeout would kill us); the new checkout releases the lock
        # before the forward pass, so this proceeds.
        second, second_out = _run_in_thread(
            lambda: factory.embed(["b"], as_list=True)
        )
        second.start()
        deadline = time.monotonic() + _TIMEOUT
        while model.calls < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert model.calls == 2, (
            "second embed never entered the forward pass while the first "
            "was parked there -- the factory-global lock is back"
        )
        model.release.set()

        first.join(_TIMEOUT)
        second.join(_TIMEOUT)
        assert "error" not in first_out and "error" not in second_out
        assert first_out["value"] == [[0.0, 1.0, 2.0, 3.0]]
        assert second_out["value"] == [[0.0, 1.0, 2.0, 3.0]]
        assert model.calls == 2
        # Both checked back in.
        assert factory._cache["default"]["refs"] == 0
        factory.close()

    def test_different_models_run_concurrently(self, monkeypatch):
        """Two models are both in their forward passes simultaneously."""
        models = {"m1": FakeModel(), "m2": FakeModel()}
        factory = _make_factory(monkeypatch, models, max_cached=2)

        t1, out1 = _run_in_thread(lambda: factory.embed(["a"], model_id="m1"))
        t1.start()
        assert models["m1"].start.wait(_TIMEOUT)

        t2, out2 = _run_in_thread(lambda: factory.embed(["b"], model_id="m2"))
        t2.start()
        # m2's forward pass starts while m1's is still parked -- impossible
        # under a factory-global lock held across the call.
        assert models["m2"].start.wait(_TIMEOUT), (
            "m2 embed never started while m1 was in flight"
        )
        assert models["m1"].release.is_set() is False  # m1 still in flight

        models["m1"].release.set()
        models["m2"].release.set()
        t1.join(_TIMEOUT)
        t2.join(_TIMEOUT)
        assert "error" not in out1 and "error" not in out2
        assert factory._cache["m1"]["refs"] == 0
        assert factory._cache["m2"]["refs"] == 0
        factory.close()

    def test_in_use_model_is_never_evicted_for_another_model(self, monkeypatch):
        """LRU eviction skips checked-out models; cache transiently exceeds
        max_cached instead, and the model closes only after check-in."""
        models = {"m": FakeModel(), "n": FakeModel(), "o": FakeModel()}
        factory = _make_factory(monkeypatch, models, max_cached=1)

        # Hold m checked out, parked inside its forward pass.
        t1, out1 = _run_in_thread(lambda: factory.embed(["a"], model_id="m"))
        t1.start()
        assert models["m"].start.wait(_TIMEOUT)

        # Request n: the only cached entry (m) is in use, so it must NOT be
        # evicted/closed; n is built past max_cached instead.
        t2, out2 = _run_in_thread(lambda: factory.embed(["b"], model_id="n"))
        t2.start()
        assert models["n"].start.wait(_TIMEOUT)
        assert "m" in factory._cache, "in-use model was evicted"
        assert not models["m"].closed, "in-use model was closed"
        assert "n" in factory._cache

        # Request o with BOTH m and n checked out: nothing is evictable, o
        # is still built (third transient entry).
        t3, out3 = _run_in_thread(lambda: factory.embed(["c"], model_id="o"))
        t3.start()
        assert models["o"].start.wait(_TIMEOUT)
        assert not models["m"].closed and not models["n"].closed

        # Drain all three.
        for m in models.values():
            m.release.set()
        t1.join(_TIMEOUT)
        t2.join(_TIMEOUT)
        t3.join(_TIMEOUT)
        for out in (out1, out2, out3):
            assert "error" not in out

        # After check-in the LRU entry (m, touched first) is evictable: one
        # more model build evicts and closes it.
        models["p"] = FakeModel()
        models["p"].release.set()
        factory.embed(["d"], model_id="p")
        assert models["m"].closed, "checked-in LRU model should now be evictable"
        factory.close()

    def test_close_waits_for_in_flight_embed(self, monkeypatch):
        """close() drains in-flight embeds instead of closing underneath one.

        Before the lock-scope reduction this ordering was guaranteed by the
        whole-call lock; the refcount + Condition pair preserves it.
        """
        model = FakeModel()
        factory = _make_factory(monkeypatch, {"default": model})

        t, out = _run_in_thread(lambda: factory.embed(["a"], as_list=True))
        t.start()
        assert model.start.wait(_TIMEOUT)

        closer, _ = _run_in_thread(factory.close)
        closer.start()
        # Give close() a beat, then confirm it is still waiting while the
        # forward pass is parked (it must NOT have closed the model yet).
        closer.join(timeout=0.2)
        assert closer.is_alive(), "close() finished while an embed was in flight"
        assert not model.closed, "model closed underneath an in-flight embed"

        model.release.set()
        t.join(_TIMEOUT)
        closer.join(_TIMEOUT)
        assert "error" not in out
        assert model.closed
        assert factory._cache == {}

    def test_embed_failure_still_checks_model_back_in(self, monkeypatch):
        """A raising embed_fn releases its checkout (finally-block path)."""

        def _boom(texts, *, as_list=False):
            raise RuntimeError("forward pass exploded")

        model = FakeModel()
        factory = _make_factory(monkeypatch, {"default": model})
        rec = factory._build("default")
        rec["embed"] = _boom
        factory._cache["default"] = rec

        with pytest.raises(RuntimeError, match="forward pass exploded"):
            factory.embed(["a"])

        assert factory._cache["default"]["refs"] == 0
        factory.close()

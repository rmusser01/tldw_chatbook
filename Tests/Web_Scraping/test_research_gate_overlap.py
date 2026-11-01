"""B16: research gate -- relevance stays sequential; scrape+summarize overlap
under a semaphore(3) with order-preserving, error-isolated slots.

Contract pinned here (mirrors the serial loop's observable behavior):

* the relevance LLM is consulted strictly in result order (it feeds spend
  decisions and the ``gate_rejected`` fallback order);
* scrape+summarize for relevant results may overlap, bounded by 3 in flight;
* the returned dict preserves the serial loop's key order (result/idx order)
  and content on the no-failure fixture;
* a failing slot (scrape or summarize) degrades exactly like the serial
  loop's per-result handlers -- fallback content, siblings unaffected.
"""

import asyncio
import threading
import time

import pytest

from tldw_chatbook.Web_Scraping import WebSearch_APIs as ws


class StageRecorder:
    """Tracks peak in-flight concurrency across the scrape/summarize fakes
    and the order of relevance-eval calls."""

    def __init__(self):
        self._lock = threading.Lock()
        self._live = 0
        self.peak_scrape_summarize_concurrency = 0
        self.relevance_call_order: list[int] = []

    def enter(self):
        with self._lock:
            self._live += 1
            self.peak_scrape_summarize_concurrency = max(
                self.peak_scrape_summarize_concurrency, self._live
            )

    def exit(self):
        with self._lock:
            self._live -= 1


@pytest.fixture()
def research_gate_factory(monkeypatch):
    def _factory(n_results: int, n_relevant: int, per_stage_delay: float = 0.0,
                 fail_summarize_for: set[int] | None = None,
                 fail_scrape: bool = False):
        recorder = StageRecorder()
        fail_summarize_for = fail_summarize_for or set()

        results = [
            {
                "id": str(i),
                "url": f"https://example.com/{i}",
                "title": f"Title {i}",
                "content": f"snippet body {i}",
            }
            for i in range(n_results)
        ]

        async def fake_scrape_article(url, *args, **kwargs):
            recorder.enter()
            try:
                if fail_scrape:
                    raise RuntimeError("scrape refused")
                if per_stage_delay:
                    await asyncio.sleep(per_stage_delay)
                idx = int(url.rsplit("/", 1)[1])
                return {"content": f"scraped {idx}", "title": f"Title {idx}"}
            finally:
                recorder.exit()

        def fake_chat_api_call(*args, **kwargs):
            messages = kwargs.get("messages_payload") or []
            body = messages[0]["content"] if messages else ""
            # The snippet text is embedded in the eval prompt.
            idx = int(body.split("snippet body ")[1].split()[0])
            recorder.relevance_call_order.append(idx)
            verdict = "True" if idx < n_relevant else "False"
            return (
                f"Selected Answer: {verdict}\n"
                f"Reasoning: verdict for {idx}"
            )

        def fake_analyze(*args, **kwargs):
            # The gate offloads the SYNC analyze via asyncio.to_thread, so
            # this fake is sync; blocking here still overlaps across slots.
            recorder.enter()
            try:
                content = str(kwargs.get("input_data", ""))
                if per_stage_delay:
                    time.sleep(per_stage_delay)
                idx = _idx_from_content(content)
                if idx in fail_summarize_for:
                    raise RuntimeError(f"summarize boom {idx}")
                return f"summary {idx}"
            finally:
                recorder.exit()

        monkeypatch.setattr(ws, "chat_api_call", fake_chat_api_call)
        monkeypatch.setattr(ws, "chat_reply_text", lambda raw: raw)
        monkeypatch.setattr(ws, "scrape_article", fake_scrape_article)
        # Overlap tests use synthetic URLs; DNS policy has its own tests.
        monkeypatch.setattr(ws, "is_public_http_url", lambda url: True)
        monkeypatch.setattr(summ_module(), "analyze", fake_analyze)
        # Hermetic prompt rendering: the real resolver reads the guarded
        # config loader (raw-participant admission), which this loop-structure
        # test neither needs nor exercises.
        monkeypatch.setattr(
            ws,
            "render_internal_prompt",
            # The real template interpolates the content kwarg into the
            # prompt; keep that so the eval transport sees the snippet text.
            lambda name, **kw: "prompt:"
            + name
            + " "
            + " ".join(f"{k}={v}" for k, v in kw.items()),
        )
        # Keep the deliberate rate-limit sleeps from slowing the tests down;
        # their presence/position is unchanged in production code.
        monkeypatch.setattr(ws.random, "uniform", lambda a, b: 0.0)

        class Gate:
            @staticmethod
            async def run(question: str, *, cancel_event=None):
                return await ws.search_result_relevance(
                    results,
                    question,
                    ["sub-q"],
                    "openai",
                    cancel_event=cancel_event,
                )

            @staticmethod
            def results_order_preserved(returned) -> bool:
                keys = list(returned.keys())
                return keys == sorted(keys, key=int)

        return Gate(), recorder

    return _factory


def summ_module():
    from tldw_chatbook.LLM_Calls import Summarization_General_Lib

    return Summarization_General_Lib


def _idx_from_content(content: str) -> int:
    """Recover the fixture result index from scraped text OR the snippet
    fallback block (which carries ``URL: https://example.com/<idx>``)."""
    if "scraped " in content:
        return int(content.split("scraped ")[-1].split()[0])
    if "example.com/" in content:
        tail = content.rsplit("example.com/", 1)[1]
        digits = ""
        for ch in tail:
            if ch.isdigit():
                digits += ch
            elif digits:
                break
        return int(digits)
    raise ValueError(f"no index recoverable from: {content[:80]!r}")


@pytest.mark.asyncio
async def test_relevant_results_processed_concurrently(research_gate_factory):
    gate, recorder = research_gate_factory(n_results=6, n_relevant=4, per_stage_delay=0.1)
    start = time.perf_counter()
    result = await gate.run("test query")
    elapsed = time.perf_counter() - start

    # Positive overlap evidence: 4 relevant slots x 0.1 s stages cannot
    # overlap at all in the serial loop (peak == 1).
    assert recorder.peak_scrape_summarize_concurrency >= 2, (
        "scrape+summarize stages are fully serial -- no overlap"
    )
    assert recorder.peak_scrape_summarize_concurrency <= 3
    # Serial bound would be 4 slots x 2 stages x 0.1 s ~= 0.8 s; under
    # semaphore(3) the 4 slots finish in ~2 waves ~= 0.4 s.
    assert elapsed < 0.65, f"stages appear serial: {elapsed:.2f}s"
    assert recorder.relevance_call_order == list(range(6)), (
        "relevance gate must stay in result order"
    )
    assert gate.results_order_preserved(result)
    # Content equivalence with the serial loop on the no-failure fixture.
    assert list(result.keys()) == ["0", "1", "2", "3"]
    for i in range(4):
        entry = result[str(i)]
        assert entry["content"] == f"summary {i}"
        assert entry["original_content"] == f"scraped {i}"
        assert entry["reasoning"] == f"verdict for {i}"
        assert entry["url"] == f"https://example.com/{i}"
        assert entry["title"] == f"Title {i}"


@pytest.mark.asyncio
async def test_summarize_failure_degrades_slot_not_siblings(research_gate_factory):
    gate, recorder = research_gate_factory(
        n_results=4, n_relevant=3, per_stage_delay=0.0, fail_summarize_for={1}
    )
    result = await gate.run("q")

    assert list(result.keys()) == ["0", "1", "2"]
    # Slot 1 falls back to the (scraped) source content, exactly like the
    # serial loop's summarize-failure path.
    assert result["1"]["content"] == "scraped 1"
    assert result["1"]["original_content"] == "scraped 1"
    # Siblings are unaffected.
    assert result["0"]["content"] == "summary 0"
    assert result["2"]["content"] == "summary 2"


@pytest.mark.asyncio
async def test_all_irrelevant_falls_back_to_gate_rejected_snippets(
    research_gate_factory,
):
    gate, recorder = research_gate_factory(n_results=3, n_relevant=0)
    result = await gate.run("q")

    assert recorder.relevance_call_order == [0, 1, 2]
    assert set(result.keys()) == {"0", "1", "2"}
    for i in range(3):
        entry = result[str(i)]
        assert entry["gate_unverified"] is True
        # Fallback content is the gate's formatted snippet block.
        assert f"snippet body {i}" in entry["content"]
        assert entry["reasoning"] == "gate fallback: evidence not relevance-verified"


@pytest.mark.asyncio
async def test_scrape_failure_keeps_snippet_fallback(research_gate_factory):
    """A scrape failure inside a slot falls back to the search snippet,
    matching the serial loop's scrape-failure handler."""
    gate, recorder = research_gate_factory(n_results=2, n_relevant=2, fail_scrape=True)
    result = await gate.run("q")

    assert list(result.keys()) == ["0", "1"]
    for i in range(2):
        entry = result[str(i)]
        assert entry["content"] == f"summary {i}"  # summarize ran on the snippet
        # original_content is the fallback block built from the search result.
        assert f"snippet body {i}" in entry["original_content"]


@pytest.mark.asyncio
async def test_cancel_event_stops_queued_scrape_summarize_slots(
    research_gate_factory, monkeypatch
):
    gate, _recorder = research_gate_factory(n_results=6, n_relevant=6)
    cancel_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    scraped = []
    summarized = []
    lock = threading.Lock()

    async def cancel():
        cancel_event.set()

    async def scrape(url):
        idx = int(url.rsplit("/", 1)[1])
        scraped.append(idx)
        return {"content": f"scraped {idx}"}

    def summarize(**kwargs):
        idx = _idx_from_content(kwargs["input_data"])
        with lock:
            summarized.append(idx)
            first = len(summarized) == 1
        if first:
            asyncio.run_coroutine_threadsafe(cancel(), loop).result(timeout=2)
        return f"summary {idx}"

    monkeypatch.setattr(ws, "is_public_http_url", lambda url: True)
    monkeypatch.setattr(ws, "scrape_article", scrape)
    monkeypatch.setattr(summ_module(), "analyze", summarize)
    result = await gate.run("q", cancel_event=cancel_event)

    assert cancel_event.is_set()
    assert summarized
    assert all(idx < 3 for idx in scraped), scraped
    assert all(idx < 3 for idx in summarized), summarized
    assert all(int(key) < 3 for key in result), result


@pytest.mark.asyncio
async def test_cancel_during_gate_skips_deferred_scrape_and_summary(
    research_gate_factory, monkeypatch
):
    gate, _recorder = research_gate_factory(n_results=3, n_relevant=3)
    cancel_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    calls = []
    scrapes = []
    summaries = []

    async def cancel():
        cancel_event.set()

    def judge(**kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            asyncio.run_coroutine_threadsafe(cancel(), loop).result(timeout=2)
            return "Selected Answer: False\nReasoning: cancelled"
        return "Selected Answer: True\nReasoning: relevant"

    async def scrape(url):
        scrapes.append(url)
        return {"content": "scraped"}

    def summarize(**kwargs):
        summaries.append(kwargs)
        return "summary"

    monkeypatch.setattr(ws, "chat_api_call", judge)
    monkeypatch.setattr(ws, "is_public_http_url", lambda url: True)
    monkeypatch.setattr(ws, "scrape_article", scrape)
    monkeypatch.setattr(summ_module(), "analyze", summarize)
    result = await gate.run("q", cancel_event=cancel_event)

    assert len(calls) == 2
    assert not scrapes
    assert not summaries
    assert result == {}

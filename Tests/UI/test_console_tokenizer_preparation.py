"""Cold tokenizers share the Console display's finite read lifetime."""

import asyncio
import threading

import pytest

from Tests.UI.test_console_readiness_config_projection import _read_result, _screen
from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend

pytestmark = pytest.mark.bootstrap_profile


class HeldPreparation:
    def __init__(self, model, provider):
        self.model, self.provider = model, provider
        self.entered, self.release = threading.Event(), threading.Event()
        self.thread = None
        self.current = True
        self.success = True

    def __call__(self):
        self.thread = threading.get_ident()
        self.entered.set()
        assert self.release.wait(5)
        return self.success


def setup_projection(monkeypatch):
    screen, owner, identity, tasks, published = _screen(monkeypatch)
    requests = []

    def capture(model, provider):
        request = HeldPreparation(model, provider)
        requests.append(request)
        return request

    monkeypatch.setattr(
        spend._tokenizer_display, "capture_tokenizer_preparation", capture
    )
    monkeypatch.setattr(
        spend._tokenizer_display, "preparation_current", lambda r: r.current
    )
    projection = spend.ConsoleReadinessConfigProjection(
        screen, read_current=lambda: _read_result({"ready": True})
    )
    return projection, requests, owner, identity, tasks, published


def prepared(projection):
    return {key: slot.prepared for key, slot in projection._tokenizers.items()}


@pytest.mark.asyncio
async def test_tokenizer_preparation_is_coalesced_off_loop(monkeypatch):
    projection, requests, _owner, _identity, tasks, published = setup_projection(
        monkeypatch
    )
    try:
        assert not projection.tokenizer_ready("model", "provider")
        request = requests[0]
        assert await asyncio.to_thread(request.entered.wait, 3)
        assert request.thread != threading.get_ident()
        for _ in range(3):
            assert not projection.tokenizer_ready("model", "provider")
            await asyncio.sleep(0)
        assert len(requests) == len(projection._reads) == 1
        request.release.set()
        await asyncio.gather(*tasks)
        assert projection.tokenizer_ready("model", "provider")
        await asyncio.gather(*tasks)
        assert published
        assert not projection._reads
    finally:
        for request in requests:
            request.release.set()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["model", "source", "settings", "close"])
async def test_changed_owner_cannot_publish_tokenizer_readiness(monkeypatch, change):
    projection, requests, owner, identity, tasks, _published = setup_projection(
        monkeypatch
    )
    try:
        assert not projection.tokenizer_ready("first", "provider")
        request = requests[0]
        assert await asyncio.to_thread(request.entered.wait, 3)
        if change == "model":
            assert not projection.tokenizer_ready("second", "provider")
        elif change == "source":
            identity[0] += 1
        elif change == "settings":
            owner.revision += 1
        else:
            projection.screen._closing = True
        request.release.set()
        await asyncio.gather(*tasks)
        if change == "model":
            # Another model is another slot, not another owner: the read in
            # flight prepares only the slot it captured.
            assert prepared(projection) == {
                ("first", "provider"): True,
                ("second", "provider"): False,
            }
        else:
            assert prepared(projection) == {("first", "provider"): False}
    finally:
        for request in requests:
            request.release.set()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_tokenizer_cancel_waits_for_physical_retirement(monkeypatch):
    projection, requests, _owner, _identity, tasks, _published = setup_projection(
        monkeypatch
    )
    try:
        assert not projection.tokenizer_ready("model", "provider")
        request = requests[0]
        assert await asyncio.to_thread(request.entered.wait, 3)
        tasks[0].cancel()
        await asyncio.sleep(0.02)
        tasks[0].cancel()
        await asyncio.sleep(0.02)
        assert projection._reads and not tasks[0].done()
        request.release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        assert not projection._reads
        assert not any(prepared(projection).values())
    finally:
        for request in requests:
            request.release.set()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_failed_backend_stays_unavailable_without_a_refresh_loop(monkeypatch):
    projection, requests, _owner, _identity, tasks, _published = setup_projection(
        monkeypatch
    )
    assert not projection.tokenizer_ready("model", "provider")
    request = requests[0]
    request.success = False
    request.release.set()
    await asyncio.gather(*tasks)
    await asyncio.gather(*tasks)
    before = len(tasks)
    assert not projection.tokenizer_ready("model", "provider")
    assert len(tasks) == before


@pytest.mark.asyncio
async def test_custom_backend_keeps_direct_path(monkeypatch):
    projection, _requests, _owner, _identity, tasks, _published = setup_projection(
        monkeypatch
    )
    monkeypatch.setattr(
        spend._tokenizer_display, "capture_tokenizer_preparation", lambda *_: None
    )
    assert projection.tokenizer_ready("custom", "provider")
    assert not tasks


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("key", "labels"),
    [
        ("openai", ("OpenAI", "openai", " OpenAI ")),
        ("local_llm", ("local-llm", "Local LLM")),
    ],
)
async def test_provider_label_and_config_key_share_one_preparation(
    monkeypatch, key, labels
):
    """TASK-38703: the cost chip passes the provider's display label ("OpenAI")
    and the context estimate its config key ("openai"). Read as two owners,
    each sync recaptured the other's request and refreshed again, so an idle
    Console never stopped refreshing."""
    projection, requests, _owner, _identity, tasks, _published = setup_projection(
        monkeypatch
    )
    assert not projection.tokenizer_ready("model", key)
    requests[0].release.set()
    await asyncio.gather(*tasks)
    await asyncio.gather(*tasks)
    before = len(tasks)
    for provider in labels:
        assert projection.tokenizer_ready("model", provider)
    assert len(requests) == 1
    assert len(tasks) == before


@pytest.mark.asyncio
async def test_differing_models_keep_their_own_preparations(monkeypatch):
    """TASK-38703 review round: the cost chip asks for the selection's model
    (``explicit_model or configured_model``) and the context estimates for the
    session's ``settings.model``. One slot recaptured on each ask, so when the
    two differ every sync re-read config and published another sync."""
    projection, requests, _owner, _identity, tasks, published = setup_projection(
        monkeypatch
    )

    def sync():
        return [
            projection.tokenizer_ready("chip-model", "OpenAI"),
            projection.tokenizer_ready("session-model", "openai"),
        ]

    try:
        assert sync() == [False, False]
        for _ in range(2):
            for request in requests:
                request.release.set()
            await asyncio.gather(*tasks)
        settled = (len(requests), len(tasks), len(published))
        for _ in range(3):
            assert sync() == [True, True]
            await asyncio.gather(*tasks)
        assert (len(requests), len(tasks), len(published)) == settled
    finally:
        for request in requests:
            request.release.set()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_tokenizer_slots_stay_bounded(monkeypatch):
    projection, requests, _owner, _identity, tasks, _published = setup_projection(
        monkeypatch
    )
    try:
        for index in range(spend.TOKENIZER_SLOTS + 3):
            projection.tokenizer_ready(f"model-{index}", "provider")
        assert [model for model, _provider in projection._tokenizers] == [
            f"model-{index}" for index in range(3, spend.TOKENIZER_SLOTS + 3)
        ]
    finally:
        for request in requests:
            request.release.set()
        await asyncio.gather(*tasks, return_exceptions=True)

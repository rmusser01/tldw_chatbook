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
        assert projection._tokenizer_prepared is None
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
        assert projection._tokenizer_prepared is None
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

"""Focused tests for the served Console's shared startup deadline."""

from __future__ import annotations

import asyncio

import pytest
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

from Tests.Canvas.browser import canvas_live_harness as served_flow

pytestmark = pytest.mark.asyncio


class _PageProbe:
    def __init__(self) -> None:
        self.selectors: list[str] = []

    def locator(self, selector: str) -> str:
        self.selectors.append(selector)
        return selector


class _ExpectationProbe:
    def __init__(self, locator: str, calls: list[tuple[str, str, float]]) -> None:
        self._locator = locator
        self._calls = calls

    async def to_have_class(self, _pattern, *, timeout: float) -> None:
        self._calls.append(("first output", self._locator, timeout))

    async def wait_for_text(self, _text: str, *, timeout_ms: float) -> None:
        self._calls.append(("Composer", self._locator, timeout_ms))

    async def to_have_attribute(self, name: str, value: str, *, timeout: float) -> None:
        assert (name, value) == ("aria-busy", "false")
        self._calls.append(("terminal input", self._locator, timeout))


async def test_shared_deadline_passes_initial_and_consumed_literal_budgets(
    monkeypatch,
) -> None:
    calls: list[tuple[str, str, float]] = []
    clock = iter((100.0, 100.0, 101.25, 102.5))
    monkeypatch.setattr(served_flow, "monotonic", lambda: next(clock))
    monkeypatch.setattr(
        served_flow,
        "expect",
        lambda locator: _ExpectationProbe(locator, calls),
    )
    monkeypatch.setattr(
        served_flow,
        "terminal_observer",
        lambda _page: _ExpectationProbe("#terminal", calls),
    )

    page = _PageProbe()
    await served_flow._wait_for_initial_console_ready(page)

    assert page.selectors == ["body", "#terminal"]
    assert calls == [
        ("first output", "body", 45_000.0),
        ("Composer", "#terminal", 43_750.0),
        ("terminal input", "#terminal", 42_500.0),
    ]


async def test_expired_deadline_refuses_to_dispatch_terminal_assertion(
    monkeypatch,
) -> None:
    calls: list[tuple[str, str, float]] = []
    clock = iter((10.0, 10.0, 55.0))
    monkeypatch.setattr(served_flow, "monotonic", lambda: next(clock))
    monkeypatch.setattr(
        served_flow,
        "expect",
        lambda locator: _ExpectationProbe(locator, calls),
    )

    page = _PageProbe()
    with pytest.raises(PlaywrightTimeoutError, match="terminal Composer"):
        await served_flow._wait_for_initial_console_ready(page)

    assert page.selectors == ["body"]
    assert calls == [("first output", "body", 45_000.0)]


async def test_first_output_assertion_failure_is_labeled_and_not_retried(
    monkeypatch,
) -> None:
    calls: list[str] = []

    class _FailingExpectation:
        async def to_have_class(self, _pattern, *, timeout: float) -> None:
            assert timeout == 45_000.0
            calls.append("first output")
            raise AssertionError("framework detail")

    monkeypatch.setattr(served_flow, "monotonic", lambda: 20.0)
    monkeypatch.setattr(served_flow, "expect", lambda _locator: _FailingExpectation())

    with pytest.raises(AssertionError, match="framework detail") as failure:
        await served_flow._wait_for_initial_console_ready(_PageProbe())

    assert calls == ["first output"]
    assert failure.value.__notes__ == ["initial Console readiness stage: first output"]


async def test_cancellation_propagates_without_dispatching_terminal_assertion(
    monkeypatch,
) -> None:
    calls: list[str] = []

    class _CancelledExpectation:
        async def to_have_class(self, _pattern, *, timeout: float) -> None:
            assert timeout == 45_000.0
            calls.append("first output")
            raise asyncio.CancelledError

    monkeypatch.setattr(served_flow, "monotonic", lambda: 30.0)
    monkeypatch.setattr(served_flow, "expect", lambda _locator: _CancelledExpectation())

    with pytest.raises(asyncio.CancelledError):
        await served_flow._wait_for_initial_console_ready(_PageProbe())

    assert calls == ["first output"]


async def test_startup_refuses_missing_first_output_or_terminal_content() -> None:
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            headless=True,
            executable_path=served_flow.chromium_executable(playwright.chromium),
        )
        try:
            missing_first_output = await browser.new_page()
            await missing_first_output.set_content(
                "<!doctype html><body><div id='terminal'>Composer</div></body>"
            )
            with pytest.raises(AssertionError) as first_output_failure:
                await served_flow._wait_for_initial_console_ready(
                    missing_first_output, timeout_ms=100
                )
            assert first_output_failure.value.__notes__ == [
                "initial Console readiness stage: first output"
            ]

            missing_composer = await browser.new_page()
            await missing_composer.set_content(
                "<!doctype html><body class='first-byte'><div id='terminal'></div></body>"
            )
            with pytest.raises(AssertionError) as composer_failure:
                await served_flow._wait_for_initial_console_ready(
                    missing_composer, timeout_ms=100
                )
            assert composer_failure.value.__notes__ == [
                "initial Console readiness stage: terminal Composer"
            ]
        finally:
            await browser.close()


async def test_painted_composer_does_not_finish_while_input_is_fenced(monkeypatch):
    calls = []
    ready = asyncio.Event()

    class Pending(_ExpectationProbe):
        async def to_have_attribute(self, name, value, *, timeout):
            await super().to_have_attribute(name, value, timeout=timeout)
            await ready.wait()

    monkeypatch.setattr(served_flow, "monotonic", lambda: 100.0)
    monkeypatch.setattr(served_flow, "expect", lambda locator: Pending(locator, calls))
    monkeypatch.setattr(
        served_flow,
        "terminal_observer",
        lambda _page: _ExpectationProbe("#terminal", calls),
    )
    task = asyncio.create_task(
        served_flow._wait_for_initial_console_ready(_PageProbe())
    )
    try:
        await asyncio.sleep(0)
        assert not task.done(), (
            "Rendered Composer returned before published input readiness"
        )
        assert calls == [
            ("first output", "body", 45_000.0),
            ("Composer", "#terminal", 45_000.0),
            ("terminal input", "#terminal", 45_000.0),
        ]
    finally:
        ready.set()
        outcome = await asyncio.gather(task, return_exceptions=True)
    assert outcome == [None]


async def test_exhausted_shared_deadline_refuses_input_readiness_dispatch(monkeypatch):
    calls = []
    clock = iter((10.0, 10.0, 54.0, 55.0))
    monkeypatch.setattr(served_flow, "monotonic", lambda: next(clock))
    monkeypatch.setattr(
        served_flow, "expect", lambda locator: _ExpectationProbe(locator, calls)
    )
    monkeypatch.setattr(
        served_flow,
        "terminal_observer",
        lambda _page: _ExpectationProbe("#terminal", calls),
    )
    page = _PageProbe()
    with pytest.raises(PlaywrightTimeoutError, match="terminal input"):
        await served_flow._wait_for_initial_console_ready(page)
    assert page.selectors == ["body"]
    assert calls == [
        ("first output", "body", 45_000.0),
        ("Composer", "#terminal", 1_000.0),
    ]


@pytest.mark.parametrize("cancelled", [False, True])
async def test_input_readiness_failure_is_not_retried_or_replaced(
    monkeypatch, cancelled
):
    calls = []
    failure = (
        asyncio.CancelledError()
        if cancelled
        else AssertionError("readiness framework detail")
    )

    class Failing(_ExpectationProbe):
        async def to_have_attribute(self, name, value, *, timeout):
            await super().to_have_attribute(name, value, timeout=timeout)
            raise failure

    monkeypatch.setattr(served_flow, "monotonic", lambda: 100.0)
    monkeypatch.setattr(served_flow, "expect", lambda locator: Failing(locator, calls))
    monkeypatch.setattr(
        served_flow,
        "terminal_observer",
        lambda _page: _ExpectationProbe("#terminal", calls),
    )
    with pytest.raises(type(failure)) as caught:
        await served_flow._wait_for_initial_console_ready(_PageProbe())
    assert caught.value is failure
    assert calls == [
        ("first output", "body", 45_000.0),
        ("Composer", "#terminal", 45_000.0),
        ("terminal input", "#terminal", 45_000.0),
    ]
    if not cancelled:
        assert failure.__notes__ == ["initial Console readiness stage: terminal input"]

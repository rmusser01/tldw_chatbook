"""Focused tests for the served Console's shared startup deadline."""

from __future__ import annotations

import asyncio

import pytest
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

from Tests.Canvas.browser import test_canvas_served_flow as served_flow

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

    async def to_contain_text(self, _text: str, *, timeout: float) -> None:
        self._calls.append(("Composer", self._locator, timeout))


async def test_shared_deadline_passes_initial_and_consumed_literal_budgets(
    monkeypatch,
) -> None:
    calls: list[tuple[str, str, float]] = []
    clock = iter((100.0, 100.0, 101.25))
    monkeypatch.setattr(served_flow, "monotonic", lambda: next(clock))
    monkeypatch.setattr(
        served_flow,
        "expect",
        lambda locator: _ExpectationProbe(locator, calls),
    )

    page = _PageProbe()
    await served_flow._wait_for_initial_console_ready(page)

    assert page.selectors == ["body", "#terminal"]
    assert calls == [
        ("first output", "body", 45_000.0),
        ("Composer", "#terminal", 43_750.0),
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


async def test_real_dom_requires_first_output_and_terminal_composer() -> None:
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            headless=True,
            executable_path=served_flow._chromium_executable(playwright.chromium),
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

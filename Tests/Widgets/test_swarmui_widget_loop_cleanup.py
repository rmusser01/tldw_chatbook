# test_swarmui_widget_loop_cleanup.py
"""
Regression tests for TASK-981 Finding 4: ``SwarmUIWidget`` bridges its
async ``ImageGenerationService`` from plain ``def`` thread workers via a
manual ``asyncio.new_event_loop()`` / ``run_until_complete()`` pattern
(``check_server_status``, ``load_models``, ``generate_image``). None of
the three closed the loop they created -- every call leaked its
selector/epoll resources. The fix wraps the loop lifetime in
``try``/``finally`` so the loop is deterministically closed (and the
thread's event loop binding cleared) on every exit path, success or
error.

These tests invoke each worker's thread-body directly, bypassing the
``@work(thread=True)`` dispatch (same idiom used by
``Tests/UI/test_settings_rag_profile_region.py``:
``getattr(worker, "__wrapped__", worker)``), against a lightweight fake
``self`` so no real Textual App/mount is needed. ``asyncio.new_event_loop``
is monkeypatched to hand back a real, inspectable loop so ``is_closed()``
can be asserted precisely.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

from tldw_chatbook.Widgets.Media_Creation.swarmui_widget import SwarmUIWidget


def _wrapped(name: str):
    """Unwrap a ``@work``-decorated method to its raw thread-body function."""
    worker = SwarmUIWidget.__dict__[name]
    return getattr(worker, "__wrapped__", worker)


class _FakeApp:
    """Stand-in for ``Widget.app``: runs ``call_from_thread`` synchronously,
    which is close enough for these thread-body tests (no real worker
    thread involved) and matches the synchronous-marshal semantics of the
    real ``call_from_thread``."""

    def call_from_thread(self, func, *args, **kwargs):
        return func(*args, **kwargs)


def _make_widget(service) -> SimpleNamespace:
    return SimpleNamespace(
        service=service,
        app=_FakeApp(),
        is_generating=False,
        is_loading=False,
        server_status="unknown",
        current_models=[],
        last_result=None,
        current_image=None,
        query_one=MagicMock(side_effect=AssertionError("unexpected query_one() call")),
        post_message=MagicMock(),
        show_status_message=MagicMock(),
        update_status_indicator=MagicMock(),
        update_model_selector=MagicMock(),
        show_generating_ui=MagicMock(),
        hide_generating_ui=MagicMock(),
        show_image_preview=MagicMock(),
    )


def _query_one_router(overrides: dict):
    def query_one(selector, _expected_type=None):
        try:
            return overrides[selector]
        except KeyError:
            raise AssertionError(f"unexpected query_one({selector!r})")

    return query_one


class TestCheckServerStatusLoopCleanup:
    def test_closes_the_loop_it_creates(self, monkeypatch):
        service = SimpleNamespace(initialize=_async_return(True))
        widget = _make_widget(service)

        real_loop = asyncio.new_event_loop()
        monkeypatch.setattr(asyncio, "new_event_loop", lambda: real_loop)
        try:
            _wrapped("check_server_status")(widget)
            assert real_loop.is_closed() is True, (
                "check_server_status leaked its event loop -- "
                "asyncio.new_event_loop() was never closed"
            )
        finally:
            if not real_loop.is_closed():
                real_loop.close()

        assert widget.server_status == "online"

    def test_closes_the_loop_even_when_initialize_raises(self, monkeypatch):
        service = SimpleNamespace(initialize=_async_raise(RuntimeError("boom")))
        widget = _make_widget(service)

        real_loop = asyncio.new_event_loop()
        monkeypatch.setattr(asyncio, "new_event_loop", lambda: real_loop)
        try:
            _wrapped("check_server_status")(widget)
            assert real_loop.is_closed() is True, (
                "check_server_status did not close its loop on the "
                "exception path -- the close must be deterministic "
                "(try/finally), not only on success"
            )
        finally:
            if not real_loop.is_closed():
                real_loop.close()

        assert widget.server_status == "offline"


class TestLoadModelsLoopCleanup:
    def test_closes_the_loop_it_creates(self, monkeypatch):
        service = SimpleNamespace(
            get_available_models=_async_return([{"name": "model-a"}])
        )
        widget = _make_widget(service)

        real_loop = asyncio.new_event_loop()
        monkeypatch.setattr(asyncio, "new_event_loop", lambda: real_loop)
        try:
            _wrapped("load_models")(widget)
            assert real_loop.is_closed() is True, (
                "load_models leaked its event loop -- "
                "asyncio.new_event_loop() was never closed"
            )
        finally:
            if not real_loop.is_closed():
                real_loop.close()

        assert widget.current_models == ["model-a"]


class TestGenerateImageLoopCleanup:
    def _widget_with_prompt(self, service, prompt="a cat"):
        widget = _make_widget(service)
        widget.query_one = _query_one_router(
            {
                "#prompt-input": SimpleNamespace(text=prompt),
                "#negative-prompt-input": SimpleNamespace(text=""),
                "#size-select": SimpleNamespace(value="512x512"),
                "#model-select": SimpleNamespace(value="default"),
                "#steps-input": SimpleNamespace(value="20"),
                "#cfg-input": SimpleNamespace(value="7.0"),
                "#seed-input": SimpleNamespace(value="-1"),
            }
        )
        return widget

    def test_closes_the_loop_on_success(self, monkeypatch):
        result = SimpleNamespace(
            success=True,
            images=["/tmp/out.png"],
            generation_time=1.23,
            error=None,
        )
        service = SimpleNamespace(generate_custom=_async_return(result))
        widget = self._widget_with_prompt(service)

        real_loop = asyncio.new_event_loop()
        monkeypatch.setattr(asyncio, "new_event_loop", lambda: real_loop)
        try:
            _wrapped("generate_image")(widget)
            assert real_loop.is_closed() is True, (
                "generate_image leaked its event loop on the success path "
                "-- asyncio.new_event_loop() was never closed"
            )
        finally:
            if not real_loop.is_closed():
                real_loop.close()

        assert widget.current_image == "/tmp/out.png"

    def test_closes_the_loop_when_generate_custom_raises(self, monkeypatch):
        service = SimpleNamespace(
            generate_custom=_async_raise(RuntimeError("network exploded"))
        )
        widget = self._widget_with_prompt(service)

        real_loop = asyncio.new_event_loop()
        monkeypatch.setattr(asyncio, "new_event_loop", lambda: real_loop)
        try:
            _wrapped("generate_image")(widget)
            assert real_loop.is_closed() is True, (
                "generate_image did not close its loop when "
                "service.generate_custom() raised -- the close must be "
                "deterministic (try/finally), not only on success"
            )
        finally:
            if not real_loop.is_closed():
                real_loop.close()

    def test_never_creates_a_loop_when_prompt_is_empty(self, monkeypatch):
        """The early-return-on-empty-prompt path never reaches
        ``asyncio.new_event_loop()`` at all -- confirm the fix's ``loop is
        not None`` finally-guard doesn't blow up when no loop was ever
        created."""
        service = SimpleNamespace(generate_custom=_async_raise(AssertionError(
            "generate_custom must not be called for an empty prompt"
        )))
        widget = self._widget_with_prompt(service, prompt="   ")

        calls = {"count": 0}
        real_new_event_loop = asyncio.new_event_loop

        def counting_new_event_loop():
            calls["count"] += 1
            return real_new_event_loop()

        monkeypatch.setattr(asyncio, "new_event_loop", counting_new_event_loop)

        _wrapped("generate_image")(widget)

        assert calls["count"] == 0
        widget.show_status_message.assert_called_once_with(
            "Please enter a prompt", "error"
        )


def _async_return(value):
    async def _coro(*_args, **_kwargs):
        return value

    return _coro


def _async_raise(exc: BaseException):
    async def _coro(*_args, **_kwargs):
        raise exc

    return _coro

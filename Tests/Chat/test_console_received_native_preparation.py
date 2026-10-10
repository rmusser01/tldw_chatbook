"""Finite native preparation remains retained through repeated cancellation."""

from __future__ import annotations

import asyncio
import contextlib
import sys
import threading
from functools import partial

import pytest

from Tests.Chat.test_console_library_policy_coordinator import _holder
from Tests.UI.test_console_hook_refresh_lifetime import _until
from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
from tldw_chatbook.Chat.console_library_policy_coordinator import (
    ConsoleLibraryPolicyCoordinator,
)
from tldw_chatbook.Chat.console_library_policy_repository import (
    ConsoleLibraryPolicyRepository,
)
from tldw_chatbook.Chat.console_preparation_reads import run_preparation_read
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


class _HeldOriginal:
    def __init__(self, callback, owner=None):
        self.code = callback.__code__
        self.owner = owner
        self.entered = threading.Event()
        self.release = threading.Event()
        self.returned = threading.Event()
        self.worker = None
        self.release_timed_out = False

    def observe(self, frame, event, arg):
        if frame.f_code is not self.code:
            return
        if self.owner is not None and frame.f_locals.get("self") is not self.owner:
            return
        if event == "call" and not self.entered.is_set():
            self.worker = threading.current_thread()
            self.entered.set()
            self.release_timed_out = not self.release.wait(8)
        elif event == "return":
            self.returned.set()

    @contextlib.contextmanager
    def installed(self):
        previous, previous_threads = sys.getprofile(), threading.getprofile()
        threading.setprofile_all_threads(self.observe)
        try:
            yield
        finally:
            self.release.set()
            threading.setprofile_all_threads(previous_threads)
            sys.setprofile(previous)


@pytest.mark.parametrize("kind", ["library-policy", "provider-readiness"])
async def test_injected_native_runner_retires_original_callback_before_cancellation(
    kind, tmp_path
):
    reads = set()
    creator = object()
    runner = partial(
        run_preparation_read,
        creator=creator,
        session_id="received-native",
        reads=reads,
        require_current=lambda: None,
    )
    db = gateway = None
    if kind == "library-policy":
        db = CharactersRAGDB(tmp_path / "policy.sqlite", client_id="received-native")
        conversation = db.add_conversation({"title": "received native policy"})
        repository = ConsoleLibraryPolicyRepository(db)
        coordinator = ConsoleLibraryPolicyCoordinator(repository)
        coordinator.register_holder(
            "received-native", conversation, _holder(allowed=False)
        )
        probe = _HeldOriginal(ConsoleLibraryPolicyRepository.read, repository)

        def operation():
            return coordinator.capture_for_execution(
                "received-native", _run_native=runner
            )
    else:
        from tldw_chatbook.Chat.provider_readiness import get_provider_readiness

        gateway = ConsoleProviderGateway(
            config_provider=lambda: {
                "api_settings": {"openai": {"api_key_env_var": "OPENAI_API_KEY"}}
            },
            environ={"OPENAI_API_KEY": "test-native-key"},
        )
        probe = _HeldOriginal(get_provider_readiness)

        def operation():
            return gateway.resolve_for_send(
                ConsoleProviderSelection(provider="openai", explicit_model="gpt-4.1"),
                _run_native=runner,
            )

    task = None
    try:
        with probe.installed():
            try:
                task = asyncio.create_task(operation())
                assert await _until(lambda: probe.entered.is_set() or task.done(), 5)
                if task.done():
                    task.result()
                assert probe.entered.is_set() and reads
                assert probe.worker is not threading.current_thread()
                task.cancel()
                await asyncio.sleep(0)
                task.cancel()
                await asyncio.sleep(0)
                assert (
                    not task.done()
                ), "cancelled waiter abandoned original native work"
                assert reads and not probe.returned.is_set()
            finally:
                probe.release.set()
                if task is not None:
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
            assert probe.returned.is_set() and not probe.release_timed_out
            assert not reads
    finally:
        probe.release.set()
        if gateway is not None:
            await gateway.aclose()
        if db is not None:
            db.close()


async def test_late_custom_gateway_inner_keeps_original_call_shape(monkeypatch):
    gateway = ConsoleProviderGateway(config_provider=lambda: {}, environ={})
    selection = ConsoleProviderSelection(provider="unavailable-test-provider")
    expected = await gateway.resolve_for_send(selection)
    seen = []

    async def native(_callback):
        raise AssertionError("custom resolver must retain its own scheduling")

    async def custom(selected):
        seen.append(selected)
        return expected

    pending = gateway.resolve_for_send(selection, _run_native=native)
    monkeypatch.setattr(gateway, "_resolve_for_send_unclassified", custom)
    try:
        result = await pending
        assert seen == [selection]
        assert result.visible_copy == expected.visible_copy
    finally:
        await gateway.aclose()


@pytest.mark.parametrize("target", ["_read_current_binding", "_run_repository_call"])
async def test_late_custom_library_reader_keeps_original_call_shape(
    monkeypatch, target
):
    from types import SimpleNamespace
    from tldw_chatbook.Chat.console_library_policy import normalize_policy_read

    expected = normalize_policy_read(None)
    repository = SimpleNamespace(db=None, read=lambda _conversation: expected)
    coordinator = ConsoleLibraryPolicyCoordinator(repository)
    coordinator.register_holder(
        "received-native", "conversation", _holder(allowed=False)
    )
    seen = []

    async def native(_callback):
        raise AssertionError("custom reader must retain its own scheduling")

    async def custom(*args):
        seen.append(args)
        return expected

    pending = coordinator.capture_for_execution("received-native", _run_native=native)
    monkeypatch.setattr(coordinator, target, custom)
    result = await pending
    expected_args = (
        ("received-native",)
        if target == "_read_current_binding"
        else (repository.read, "conversation")
    )
    assert seen == [expected_args]
    assert result == expected.snapshot

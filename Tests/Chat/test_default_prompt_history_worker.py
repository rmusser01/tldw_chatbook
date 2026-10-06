"""Original prompt-history resolver actor controls; no source replacement."""

import asyncio
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from Tests.Backup_Recovery.config_test_support import install_config_source
from Tests.Backup_Recovery.test_bootstrap import local_scope as local_scope  # noqa: PLC0414
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Chat import prompt_history
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime


@pytest.mark.asyncio
async def test_default_runtime_history_resolves_and_qualifies_on_its_worker(
    local_scope,  # noqa: F811 - exported fixture dependency
    monkeypatch,
):
    """The real guarded resolver runs, but never on the shared event loop."""
    root, configuration, data, _authority = local_scope
    configuration.write_text(
        '[general]\nusers_name = "data"\n[paths]\ndata_dir = '
        + json.dumps(data.parent.as_posix())
        + "\n",
        encoding="utf-8",
    )
    install_config_source(monkeypatch)
    resolver = prompt_history.default_prompt_history_path
    resolver_code = resolver.__code__
    main_thread = threading.current_thread()
    calls = []

    def observe(frame, event, _argument):
        if event == "call" and frame.f_code is resolver_code:
            calls.append((threading.current_thread(), frame.f_back.f_code.co_name))

    loop = asyncio.get_running_loop()
    previous_executor = loop._default_executor
    executor = ThreadPoolExecutor(max_workers=1)
    previous_main_profile = sys.getprofile()
    previous_thread_profile = threading.getprofile()
    runtime = ConsoleRuntime(SimpleNamespace(), canvas_enabled_reader=lambda: False)
    try:
        loop.set_default_executor(executor)
        sys.setprofile(observe)
        threading.setprofile(observe)
        history = runtime.ensure_prompt_history()
        assert runtime.ensure_prompt_history() is history
        await history.load()
        assert history._loaded, history.persistence_error
        assert await history.append("worker-owned default history")
        assert history.path == data / "prompt_history.jsonl"
        assert (await history.get_entry(-1))["input"] == "worker-owned default history"
        assert history.size == 1
        assert prompt_history.default_prompt_history_path is resolver
        assert resolver.__code__ is resolver_code
        assert calls, "the original default resolver was never entered"
        assert all(actor is not main_thread for actor, _caller in calls), calls
    finally:
        sys.setprofile(previous_main_profile)
        threading.setprofile(previous_thread_profile)
        await runtime.dispose()
        executor.shutdown(wait=True)
        loop._default_executor = previous_executor
        startup = storage._startups.pop((os.getpid(), str(root)), None)
        if startup is not None:
            startup.close()


@pytest.mark.asyncio
async def test_custom_history_factory_keeps_synchronous_creator_abi():
    """The existing factory remains synchronous and is called exactly once."""
    creator = threading.current_thread()
    calls = []
    sentinel = object()

    def factory():
        calls.append(threading.current_thread())
        return sentinel

    runtime = ConsoleRuntime(
        SimpleNamespace(console_prompt_history_factory=factory),
        canvas_enabled_reader=lambda: False,
    )
    try:
        assert runtime.ensure_prompt_history() is sentinel
        assert runtime.ensure_prompt_history() is sentinel
        assert calls == [creator]
    finally:
        await runtime.dispose()

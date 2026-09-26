"""Opt-in qualification with an existing local binary and GGUF; no downloads."""

from __future__ import annotations

import asyncio
import os
import socket
import threading
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest


@pytest.mark.asyncio
@pytest.mark.loopback_network
@pytest.mark.skipif(
    not os.environ.get("CHATBOOK_TEST_LLAMA_SERVER")
    or not os.environ.get("CHATBOOK_TEST_LLAMA_MODEL"),
    reason="Existing binary and model must be explicitly selected",
)
async def test_existing_llama_runtime_launch_verify_infer_and_stop():
    from tldw_chatbook.Event_Handlers.LLM_Management_Events import (
        server_lifecycle as lifecycle,
    )
    from tldw_chatbook.Event_Handlers.LLM_Management_Events.gguf_source_modes import (
        GGUFSourceMode,
        GGUFSourceSelection,
    )
    from tldw_chatbook.Event_Handlers.LLM_Management_Events.llm_management_events import (
        run_llamacpp_server_worker,
    )
    from tldw_chatbook.LLM_Management.llamacpp_connection import (
        LlamaCppConnectionOwner,
        probe_llamacpp_target,
    )
    from tldw_chatbook.LLM_Management.llamacpp_diagnostics import DiagnosticSink

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = str(listener.getsockname()[1])
    app = SimpleNamespace(
        _llm_server_lifecycle_lock=threading.RLock(),
        _llm_server_launch_claims={},
        llamacpp_server_process=None,
        screen_stack=[],
        notify=lambda *a, **k: None,
    )
    app.call_from_thread = lambda callback, *args: callback(*args)
    claim = lifecycle.reserve_server_launch(app, "llamacpp")
    claim._diagnostics = DiagnosticSink()
    selection = GGUFSourceSelection(
        mode=GGUFSourceMode.EXTERNAL,
        external_path=Path(os.environ["CHATBOOK_TEST_LLAMA_MODEL"]),
    )
    work = asyncio.create_task(
        asyncio.to_thread(
            run_llamacpp_server_worker,
            app,
            os.environ["CHATBOOK_TEST_LLAMA_SERVER"],
            "127.0.0.1",
            port,
            (
                "--ctx-size",
                "512",
                "--gpu-layers",
                "0",
                "--threads",
                "2",
                "--parallel",
                "1",
                "--batch-size",
                "64",
            ),
            selection,
            claim,
        )
    )
    owner = LlamaCppConnectionOwner()
    request = owner.begin(
        f"http://127.0.0.1:{port}",
        runtime_owner="lab_process",
        live_check=lambda: lifecycle.snapshot_claim_is_live(app, claim),
    )
    try:
        async with asyncio.timeout(120):
            while True:
                if work.done():
                    pytest.fail(
                        f"Runtime exited before readiness: {claim._diagnostics.snapshot()} / {work.result()}"
                    )
                result = await probe_llamacpp_target(request, timeout_seconds=2)
                owner.accept(result)
                if result.code == "ready":
                    break
                await asyncio.sleep(0.5)
        target = owner.snapshot().target
        assert target is not None and target.model_id == "chatbook-llamacpp"
        async with httpx.AsyncClient(trust_env=False, timeout=60) as client:
            response = await client.post(
                f"{target.base_url}/v1/chat/completions",
                json={
                    "model": target.model_id,
                    "messages": [{"role": "user", "content": "Say hello."}],
                    "max_tokens": 2,
                    "stream": False,
                },
            )
            assert response.status_code == 200
            assert response.json()["choices"]
    finally:
        await lifecycle.stop_server_process(
            app, "llamacpp", "smoke test", expected_claim=claim
        )
        await asyncio.wait_for(work, 15)
    assert app.llamacpp_server_process is None
    assert owner.snapshot().target is None

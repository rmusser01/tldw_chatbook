from __future__ import annotations

import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from tldw_chatbook.Event_Handlers.LLM_Management_Events.server_lifecycle import (
    reserve_server_launch,
    run_server_subprocess,
)
from tldw_chatbook.LLM_Management.llamacpp_diagnostics import (
    DiagnosticPump,
    DiagnosticSink,
)


def test_fixed_categories_are_bounded_and_never_retain_private_payload():
    sink = DiagnosticSink()
    for _ in range(300):
        sink.feed(b"PRIVATE_SECRET /private/model.gguf failed to allocate memory\n")
        sink.feed(b"PRIVATE_SECRET unknown argument --private\n")
    entries = sink.snapshot()
    assert len(entries) == 128
    assert any("memory" in entry.lower() for entry in entries)
    assert "PRIVATE" not in repr(vars(sink)) + repr(entries)
    assert "/private" not in repr(entries)


def test_real_child_drains_both_pipes_without_newlines():
    sink = DiagnosticSink()
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os; [(os.write(1,b'PRIVATE'*8192),os.write(2,b'SECRET'*8192)) for _ in range(80)]; os.write(2,b'\\nunknown argument --secret\\n')",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    with DiagnosticPump(process, sink) as pump:
        assert pump.wait(timeout=10) == 0
    assert process.stdout.closed and process.stderr.closed
    assert any("option" in entry.lower() for entry in sink.snapshot())
    assert "SECRET" not in repr(sink.snapshot())


def test_real_lifecycle_exit_releases_exact_resource_and_retains_safe_diagnostics():
    app = SimpleNamespace(
        _llm_server_lifecycle_lock=threading.RLock(),
        _llm_server_launch_claims={},
        llamacpp_server_process=None,
        screen_stack=[],
    )
    app.call_from_thread = lambda callback, *args: callback(*args)
    claim = reserve_server_launch(app, "llamacpp")
    sink = DiagnosticSink()
    result = run_server_subprocess(
        app,
        "llamacpp",
        [
            sys.executable,
            "-c",
            "import sys; print('out of memory PRIVATE',file=sys.stderr); sys.exit(3)",
        ],
        claim,
        subprocess,
        diagnostics=sink,
    )
    assert "code=3" in result
    assert app.llamacpp_server_process is None
    assert not app._llm_server_launch_claims
    assert any("memory" in entry.lower() for entry in sink.snapshot())
    assert any("3" in entry for entry in sink.snapshot())


def test_split_failure_marker_is_classified_without_retaining_tail():
    sink = DiagnosticSink()
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os,time; os.write(2,b'PRIVATE failed to allo'); time.sleep(.1); os.write(2,b'cate memory SECRET')",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    with DiagnosticPump(process, sink) as pump:
        assert pump.wait(timeout=5) == 0
    assert any("memory" in entry.lower() for entry in sink.snapshot())
    assert not pump._tails


@pytest.mark.asyncio
async def test_real_cancel_keeps_lease_until_exact_process_dies():
    import asyncio

    from tldw_chatbook.Event_Handlers.LLM_Management_Events import (
        server_lifecycle as lifecycle,
    )

    published = threading.Event()
    closed = []
    app = SimpleNamespace(
        _llm_server_lifecycle_lock=threading.RLock(),
        _llm_server_launch_claims={},
        llamacpp_server_process=None,
        screen_stack=[],
        notify=lambda *a, **k: None,
    )

    def callback(function, *args):
        result = function(*args)
        if function is lifecycle.publish_server_process:
            published.set()
        return result

    app.call_from_thread = callback
    claim = reserve_server_launch(app, "llamacpp")
    lifecycle.attach_server_claim_resource(
        app, "llamacpp", claim, SimpleNamespace(close=lambda: closed.append(True))
    )
    sink = DiagnosticSink()
    work = asyncio.create_task(
        asyncio.to_thread(
            run_server_subprocess,
            app,
            "llamacpp",
            [sys.executable, "-c", "import time; time.sleep(30)"],
            claim,
            subprocess,
            diagnostics=sink,
        )
    )
    assert await asyncio.to_thread(published.wait, 5)
    assert not closed
    assert await lifecycle.stop_server_process(
        app, "llamacpp", "test", expected_claim=claim
    )
    await asyncio.wait_for(work, 5)
    assert closed == [True]
    assert app.llamacpp_server_process is None

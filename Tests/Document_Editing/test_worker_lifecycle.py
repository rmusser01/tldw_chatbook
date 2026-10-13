"""OS-process lifecycle controls with real positive engine checks."""

import asyncio
import os
import time
from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_successful_child_then_real_stall_is_killed(
    request_for, tmp_path, monkeypatch
):
    from tldw_chatbook.Document_Editing import worker as module

    worker = module.DocumentWorker()
    assert (await worker.check(request_for("{}"))).format_allowed
    script = tmp_path / "stall.py"
    script.write_text("import time\ntime.sleep(60)\n")
    monkeypatch.setattr(module, "WORKER_PATH", script)
    call = asyncio.create_task(worker.check(request_for("{}")))
    while worker._process is None:
        await asyncio.sleep(0.01)
    process = worker._process
    start = time.monotonic()
    result = await asyncio.wait_for(call, 5)
    assert result.state == "resource_limited" and not result.complete
    assert 1.5 < time.monotonic() - start < 4
    assert process.returncode is not None
    assert worker._process is None
    await worker.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "script",
    [
        "raise SystemExit(7)",
        'import sys; sys.stdout.buffer.write(b"bad!"); sys.stdout.flush()',
    ],
)
async def test_crash_or_malformed_reply_is_safe(
    request_for, tmp_path, monkeypatch, script
):
    from tldw_chatbook.Document_Editing import worker as module

    path = tmp_path / "broken.py"
    path.write_text(script)
    worker = module.DocumentWorker()
    assert (await worker.check(request_for("{}"))).complete
    monkeypatch.setattr(module, "WORKER_PATH", path)
    result = await worker.check(request_for("PRIVATE_SOURCE_MARKER"))
    assert result.state == "failed" and not result.complete
    assert "PRIVATE_SOURCE_MARKER" not in repr(result)
    assert worker._process is None
    await worker.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["cancel", "close"])
async def test_close_or_cancel_reaps_actual_child(
    request_for, tmp_path, monkeypatch, action
):
    from tldw_chatbook.Document_Editing import worker as module

    path = tmp_path / "stall.py"
    path.write_text("import time\ntime.sleep(60)\n")
    monkeypatch.setattr(module, "WORKER_PATH", path)
    worker = module.DocumentWorker()
    call = asyncio.create_task(worker.check(request_for("{}")))
    while worker._process is None:
        await asyncio.sleep(0.01)
    process = worker._process
    if action == "cancel":
        call.cancel()
    else:
        await worker.close()
    with pytest.raises(asyncio.CancelledError):
        await call
    try:
        assert process.returncode is not None and worker._process is None
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
        await worker.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text,overrides,state",
    [
        ("x" * (1024 * 1024 + 1), {}, "resource_limited"),
        ("\ud800", {}, "unsupported"),
        ("{}", {"is_excerpt": True}, "unsupported"),
    ],
)
async def test_reject_before_dispatch(request_for, monkeypatch, text, overrides, state):
    from tldw_chatbook.Document_Editing import worker as module

    monkeypatch.setattr(module, "WORKER_PATH", Path("/nonexistent/e4-must-not-launch"))
    worker = module.DocumentWorker()
    result = await worker.check(request_for(text, **overrides))
    assert result.state == state and not result.complete
    assert worker._process is None
    await worker.close()


@pytest.mark.asyncio
async def test_real_format_depth_aliases_and_diagnostic_cap(request_for):
    from tldw_chatbook.Document_Editing.worker import DocumentWorker

    worker = DocumentWorker()
    result = await worker.format(request_for('{"n":9007199254740993,"x":-0}'))
    assert result.state == "formatted" and "9007199254740993" in result.replacement
    assert "-0" in result.replacement
    depth = await worker.check(request_for("[" * 101 + "0" + "]" * 101))
    assert depth.state == "resource_limited" and not depth.complete
    aliases = await worker.check(
        request_for(
            "a: &a [1]\nb: [" + ",".join(["*a"] * 5000) + "]\n", language="yaml"
        )
    )
    assert aliases.complete and aliases.format_allowed
    capped = await worker.check(request_for("\n" * 101, language="jsonl"))
    assert len(capped.diagnostics) == 100 and capped.truncated and not capped.complete
    await worker.close()


@pytest.mark.asyncio
async def test_child_has_no_parent_secret_app_import_or_extra_fd(
    request_for, tmp_path, monkeypatch
):
    from tldw_chatbook.Document_Editing import worker as module

    original = module.WORKER_PATH
    metadata = tmp_path / "child-metadata.json"
    extra = os.open(tmp_path / "private-parent-only", os.O_CREAT | os.O_RDWR, 0o600)
    os.set_inheritable(extra, True)
    script = tmp_path / "isolated.py"
    script.write_text(f"""import json, os, pathlib, runpy, sys
assert 'E4_SYNTHETIC_SECRET' not in os.environ
try:
    os.fstat({extra})
except OSError:
    pass
else:
    raise AssertionError('Inherited descriptor')
try:
    runpy.run_path({str(original)!r}, run_name='__main__')
finally:
    pathlib.Path({str(metadata)!r}).write_text(json.dumps({{
        'app_imported': 'tldw_chatbook.app' in sys.modules,
        'config_imported': 'tldw_chatbook.config' in sys.modules,
        'package_imported': 'tldw_chatbook' in sys.modules,
        'module_count': len(sys.modules),
        'environment_keys': sorted(os.environ),
    }}))
""")
    monkeypatch.setenv("E4_SYNTHETIC_SECRET", "synthetic-only")
    monkeypatch.setattr(module, "WORKER_PATH", script)
    worker = module.DocumentWorker()
    try:
        result = await worker.check(request_for("{}"))
        assert result.complete and result.format_allowed
        import json

        evidence = json.loads(metadata.read_text())
        assert not any(
            evidence[name]
            for name in ("app_imported", "config_imported", "package_imported")
        )
        assert evidence["module_count"] < 200
    finally:
        os.close(extra)
        await worker.close()


@pytest.mark.asyncio
async def test_closed_worker_and_failed_launch_remain_safe(request_for, monkeypatch):
    from tldw_chatbook.Document_Editing import worker as module

    monkeypatch.setattr(module, "WORKER_PATH", Path("/missing/e4/worker.py"))
    worker = module.DocumentWorker()
    assert (await worker.check(request_for("{}"))).state == "failed"
    await worker.close()
    assert (await worker.check(request_for("{}"))).state == "failed"


def test_wire_refuses_wrong_identity_types_bounds_and_false_completion(request_for):
    from dataclasses import asdict

    from tldw_chatbook.Document_Editing.engine import check_document
    from tldw_chatbook.Document_Editing.worker_process import decode_result

    request = request_for("{}")
    base = asdict(check_document(request))
    base["diagnostics"] = []
    for override in [
        {"key": {}},
        {"key": {**base["key"], "draft_revision": True}},
        {"key": {**base["key"], "request_id": 1.0}},
        {"complete": 1},
        {"diagnostics": [{}] * 101},
        {"truncated": True},
        {"state": "other"},
        {
            "diagnostics": [
                {
                    "code": "syntax",
                    "severity": "error",
                    "message": "PRIVATE_SOURCE",
                    "start": 0,
                    "end": 3,
                }
            ]
        },
    ]:
        with pytest.raises((ValueError, TypeError)):
            decode_result(request, "check", {**base, **override})
    safe = decode_result(
        request,
        "check",
        {
            **base,
            "format_allowed": False,
            "diagnostics": [
                {
                    "code": "syntax",
                    "severity": "error",
                    "message": "PRIVATE_SOURCE",
                    "start": 0,
                    "end": 1,
                }
            ],
        },
    )
    assert "PRIVATE_SOURCE" not in repr(safe)


@pytest.mark.asyncio
async def test_repeated_retire_during_termination_still_reaps(
    request_for, tmp_path, monkeypatch
):
    from tldw_chatbook.Document_Editing import worker as module

    script = tmp_path / "ignore-term.py"
    ready = tmp_path / "ready"
    script.write_text(
        f"import signal,time,pathlib\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\npathlib.Path({str(ready)!r}).touch()\ntime.sleep(60)\n"
    )
    monkeypatch.setattr(module, "WORKER_PATH", script)
    worker = module.DocumentWorker()
    lease = await worker.acquire(lambda result: None)
    lease.submit(request_for("{}"), immediate=True)
    async with asyncio.timeout(2):
        while not ready.exists():
            await asyncio.sleep(0.01)
    process = worker._process
    lease.retire()
    await asyncio.sleep(0.02)
    await lease.close()
    try:
        assert process.returncode is not None and worker._process is None
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
        await worker.close()


@pytest.mark.asyncio
async def test_size_admission_accounts_for_json_wire_escaping(request_for):
    from tldw_chatbook.Document_Editing.worker import DocumentWorker

    worker = DocumentWorker()
    result = await worker.check(request_for("\x00" * (1024 * 1024)))
    assert result.state == "checked" and result.complete
    assert result.diagnostics[0].code == "syntax"
    await worker.close()


@pytest.mark.asyncio
async def test_close_during_pending_launch_reaps_before_return(
    request_for, tmp_path, monkeypatch
):
    from tldw_chatbook.Document_Editing import worker as module

    script = tmp_path / "pending-launch.py"
    script.write_text("import time\ntime.sleep(60)\n")
    monkeypatch.setattr(module, "WORKER_PATH", script)
    create = asyncio.create_subprocess_exec
    launched, release = asyncio.Event(), asyncio.Event()
    processes = []

    async def delayed_launch(*args, **kwargs):
        process = await create(*args, **kwargs)
        processes.append(process)
        launched.set()
        await release.wait()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_launch)
    owner = module.DocumentWorker()
    pending = asyncio.create_task(owner.check(request_for("{}")))
    await asyncio.wait_for(launched.wait(), 2)
    closing = asyncio.create_task(owner.close())
    await asyncio.sleep(0)
    assert not closing.done()
    release.set()
    await asyncio.wait_for(closing, 2)
    assert processes[0].returncode is not None
    assert owner._process is None
    with pytest.raises(asyncio.CancelledError):
        await pending


@pytest.mark.asyncio
async def test_exited_child_with_inherited_pipe_is_closed_before_replacement(
    request_for, tmp_path, monkeypatch
):
    from tldw_chatbook.Document_Editing import worker as module

    original = module.WORKER_PATH
    descendant_pid = tmp_path / "descendant.pid"
    script = tmp_path / "pipe-holder.py"
    script.write_text(
        f"import pathlib,subprocess,sys\np=subprocess.Popen([sys.executable,'-I','-c','import time; time.sleep(5)'])\npathlib.Path({str(descendant_pid)!r}).write_text(str(p.pid))\n"
    )
    monkeypatch.setattr(module, "WORKER_PATH", script)
    owner = module.DocumentWorker()
    started = time.monotonic()
    try:
        result = await asyncio.wait_for(owner.check(request_for("{}")), 4)
        assert result.state == "resource_limited"
        assert time.monotonic() - started < 3
        assert owner._process is None
        monkeypatch.setattr(module, "WORKER_PATH", original)
        assert (await owner.check(request_for("{}"))).format_allowed
    finally:
        if descendant_pid.exists():
            try:
                os.kill(int(descendant_pid.read_text()), 9)
            except ProcessLookupError:
                pass
        await owner.close()

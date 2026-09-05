"""Real fresh-process preview ownership and resource boundaries."""

import asyncio
import importlib
import json
import os
import subprocess
import sys
import time

import pytest

from tldw_chatbook.Chunking.lab_state import (
    capture_batch,
    edit_json,
    new_session,
    replace_sample,
)


def api():
    from importlib.util import find_spec

    assert find_spec("tldw_chatbook.Chunking.lab_runner"), (
        "Bounded preview runner is missing"
    )
    return importlib.import_module("tldw_chatbook.Chunking.lab_runner")


def request_for(text="one two three", body=None):
    session = replace_sample(new_session("test-profile"), text, {"kind": "paste"})
    candidate_id = next(iter(session.candidates))
    if body is not None:
        session = edit_json(session, candidate_id, json.dumps(body))
    return capture_batch(session, (candidate_id,))[0]


@pytest.mark.asyncio
async def test_sample_limit_reports_failure_without_clipping():
    module = api()
    request = request_for()
    runner = module.LocalPreviewRunner(module.PreviewLimits(sample_bytes=3))
    result = await runner.run(request)
    assert result.status == "limited"
    assert result.request.sample.text == "one two three"
    assert result.report is None
    await runner.close()


@pytest.mark.asyncio
async def test_real_worker_returns_exact_input_and_complete_report():
    module = api()
    runner = module.LocalPreviewRunner(module.PreviewLimits())
    request = request_for()
    result = await runner.run(request)
    assert result.status == "completed", result.error
    assert result.request == request
    assert [chunk["text"] for chunk in result.report.chunks] == ["one two three"]
    await runner.close()


def noncooperative(monkeypatch, tmp_path, module):
    pidfile = tmp_path / "pid"
    code = (
        "import os,signal,time,pathlib; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid())); "
        "time.sleep(100)"
    )
    monkeypatch.setattr(module, "_worker_command", lambda: [sys.executable, "-c", code])
    return pidfile


async def wait_pid(path):
    async with asyncio.timeout(10):
        while not path.exists():
            await asyncio.sleep(0.01)
    return int(path.read_text())


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["timeout", "cancel", "close", "task_cancel"])
async def test_noncooperative_child_is_killed_and_reaped(monkeypatch, tmp_path, action):
    module = api()
    pidfile = noncooperative(monkeypatch, tmp_path, module)
    runner = module.LocalPreviewRunner(
        module.PreviewLimits(wall_seconds=0.7 if action == "timeout" else 30)
    )
    task = asyncio.create_task(runner.run(request_for()))
    pid = await wait_pid(pidfile)
    start = time.monotonic()
    if action == "task_cancel":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        if action != "timeout":
            await getattr(runner, action)()
        result = await task
        assert result.status == ("limited" if action == "timeout" else "canceled")
    assert time.monotonic() - start < 4
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    await runner.close()


@pytest.mark.asyncio
async def test_concurrent_run_rejected_until_physical_child_stops(
    monkeypatch, tmp_path
):
    module = api()
    pidfile = noncooperative(monkeypatch, tmp_path, module)
    runner = module.LocalPreviewRunner(module.PreviewLimits())
    first = asyncio.create_task(runner.run(request_for()))
    await wait_pid(pidfile)
    with pytest.raises(RuntimeError):
        await runner.run(request_for())
    await runner.cancel()
    assert (await first).status == "canceled"
    await runner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body,text,code",
    [
        (
            {
                "chunking": {
                    "method": "fixed_size",
                    "config": {"max_size": 1, "overlap": 0},
                }
            },
            "a" * 10001,
            "chunk_limit",
        ),
        (
            {
                "chunking": {
                    "method": "fixed_size",
                    "config": {"max_size": 10000, "overlap": 9999},
                }
            },
            "a" * 15000,
            "intermediate_limit",
        ),
        (
            {
                "chunking": {"method": "words"},
                "postprocessing": [
                    {
                        "operation": "format_chunks",
                        "config": {"template": "{chunk}" * 10000},
                    }
                ],
            },
            "a" * 10000,
            "intermediate_limit",
        ),
        (
            {
                "chunking": {
                    "method": "fixed_size",
                    "config": {"max_size": 1, "overlap": 0},
                },
                "postprocessing": [
                    {"operation": "add_metadata", "config": {"prefix": "x" * 10000}}
                ],
            },
            "a" * 5000,
            "intermediate_limit",
        ),
    ],
    ids=["chunk-count", "tiny-step", "repeated-format", "long-prefix"],
)
async def test_amplification_refused_before_launch(monkeypatch, body, text, code):
    module = api()
    monkeypatch.setattr(
        module,
        "_worker_command",
        lambda: pytest.fail("Rejected recipe launched a child"),
    )
    runner = module.LocalPreviewRunner(module.PreviewLimits())
    result = await runner.run(request_for(text, body))
    assert result.status == "limited"
    assert result.error["code"] == code
    await runner.close()


@pytest.mark.asyncio
async def test_serialized_limit_includes_captured_request():
    module = api()
    request = request_for("z" * 5000)
    runner = module.LocalPreviewRunner(module.PreviewLimits(result_bytes=6000))
    result = await runner.run(request)
    assert result.status == "limited"
    assert result.report is None
    assert result.error["code"] == "result_limit"
    await runner.close()


@pytest.mark.parametrize(
    "body",
    [
        {
            "chunking": {"method": "words"},
            "preprocessing": [{"operation": "detect_language"}] * 17,
        },
        {"chunking": {"method": "words"}, "metadata": {"large": "x" * 2097152}},
    ],
)
def test_shared_preflight_refuses_static_resource_excess(body):
    from tldw_chatbook.Chunking.lab_preflight import (
        PreviewUnsupportedError,
        current_local_runtime,
        prepare_recipe,
    )

    with pytest.raises(PreviewUnsupportedError, match="resource"):
        prepare_recipe(body, runtime=current_local_runtime())


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["oversize", "bad_json", "wrong_member", "exit"])
async def test_worker_protocol_refuses_untrusted_or_incomplete_results(
    monkeypatch, mode
):
    module = api()
    request = request_for()
    if mode == "oversize":
        code = "import sys,struct; sys.stdout.buffer.write(struct.pack('!Q',33554433)); sys.stdout.buffer.flush()"
    elif mode == "bad_json":
        code = "import sys,struct; sys.stdout.buffer.write(struct.pack('!Q',1)+b'{'); sys.stdout.buffer.flush()"
    elif mode == "wrong_member":
        wrong = module.terminal_result(
            request.model_copy(update={"run_id": "not-member"}), "failed", "test"
        )
        payload = module._encoded(wrong)
        code = f"import sys,struct; p={payload!r}; sys.stdout.buffer.write(struct.pack('!Q',len(p))+p); sys.stdout.buffer.flush()"
    else:
        code = "raise SystemExit(2)"
    monkeypatch.setattr(module, "_worker_command", lambda: [sys.executable, "-c", code])
    runner = module.LocalPreviewRunner(module.PreviewLimits(wall_seconds=3))
    result = await runner.run(request)
    assert result.status == ("limited" if mode == "oversize" else "failed")
    assert result.request == request and result.report is None
    await runner.close()


@pytest.mark.asyncio
async def test_changed_runtime_refuses_without_substituting_new_assets():
    module = api()
    request = request_for()
    request = request.model_copy(
        update={
            "recipe": request.recipe.model_copy(
                update={
                    "runtime": request.recipe.runtime.model_copy(
                        update={"execution_version": "unavailable"}
                    )
                }
            )
        }
    )
    runner = module.LocalPreviewRunner(module.PreviewLimits())
    result = await runner.run(request)
    assert result.status == "failed"
    assert result.request.recipe.runtime.execution_version == "unavailable"
    await runner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pattern,text,code",
    [
        ("()", "x" * 10000, "section_limit"),
        (r"(?=(.{10000}))", "x" * 18000, "intermediate_limit"),
    ],
    ids=["too-many-matches", "overlapping-captures"],
)
async def test_extract_prescan_bounds_matches_and_cumulative_capture_bytes(
    pattern, text, code
):
    module = api()
    request = request_for(
        text,
        {
            "chunking": {"method": "words"},
            "preprocessing": [
                {"operation": "extract_sections", "config": {"pattern": pattern}}
            ],
        },
    )
    runner = module.LocalPreviewRunner(module.PreviewLimits())
    result = await runner.run(request)
    assert result.status == "limited", result.error
    assert result.error["code"] == code
    await runner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case",
    ["sample", "tokens", "format", "overlap", "merge", "sections", "stage-count"],
)
async def test_admitted_resource_stress_records_real_child_peak_rss(
    case, record_property
):
    module = api()
    body = {
        "chunking": {"method": "fixed_size", "config": {"max_size": 4000, "overlap": 0}}
    }
    text = "x" * 2097152
    if case == "tokens":
        text = "x " * 165000
        body = {"chunking": {"method": "words"}}
    elif case == "format":
        text = "x" * 200000
        body["postprocessing"] = [
            {"operation": "format_chunks", "config": {"template": "{chunk}" * 160}}
        ]
    elif case == "overlap":
        text = "abcdefghij" * 32
        body["chunking"]["config"]["max_size"] = 10
        body["postprocessing"] = [
            {"operation": "add_overlap", "config": {"size": 1000}}
        ] * 8
    elif case == "merge":
        text = "x" * 300000
        body["postprocessing"] = [
            {
                "operation": "merge_small",
                "config": {"min_size": 500000, "separator": "|"},
            }
        ]
    elif case == "sections":
        text = "x" * 3000
        body["preprocessing"] = [
            {"operation": "extract_sections", "config": {"pattern": r"(?=(.{1000}))"}}
        ]
    elif case == "stage-count":
        text = "x" * 3200
        body["postprocessing"] = [
            {"operation": "add_metadata", "config": {"prefix": "x" * 10000}}
        ] * 16
    runner = module.LocalPreviewRunner(module.PreviewLimits())
    result = await runner.run(request_for(text, body))
    assert result.status == "completed", (case, result.error)
    resources = result.report.diagnostics[-1]
    record_property(case, resources)
    print(case, json.dumps(resources, sort_keys=True))
    assert 0 < resources["estimated_working_bytes"] <= 33554432
    assert 0 < resources["peak_rss_bytes"] < 1073741824
    assert resources["applied_limits"].get("RLIMIT_CPU") == 61
    if sys.platform.startswith("linux"):
        assert resources["applied_limits"].get("RLIMIT_AS") == 1073741824
    await runner.close()


def test_fresh_mounted_textual_first_launch_with_negative_stderr_fd(tmp_path):
    api()
    code = """
import asyncio, io, sys
from textual.app import App
from tldw_chatbook.Chunking.lab_runner import LocalPreviewRunner, PreviewLimits
from tldw_chatbook.Chunking.lab_state import new_session, replace_sample, capture_batch
class Captured(io.StringIO):
    def fileno(self): return -1
class Harness(App):
    async def on_mount(self):
        async def launch():
            old = sys.stderr
            sys.stderr = Captured()
            try:
                session = replace_sample(new_session("isolated"), "first mounted preview", {"kind":"paste"})
                runner = LocalPreviewRunner(PreviewLimits())
                result = await runner.run(capture_batch(session, tuple(session.candidates))[0])
                await runner.close()
                assert result.status == "completed", result.error
                assert result.report.chunks[0]["text"] == "first mounted preview"
                self.exit(True)
            finally:
                sys.stderr = old
        self.run_worker(launch(), exclusive=True)
async def main():
    app = Harness()
    async with app.run_test(size=(80,24)) as pilot:
        async with asyncio.timeout(10):
            while app.return_value is None: await asyncio.sleep(.01)
    assert app.return_value is True
asyncio.run(main())
print("fresh-mounted-ok")
"""
    env = {
        **os.environ,
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "TLDW_CONFIG_PATH": str(tmp_path / "config.toml"),
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=20,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "fresh-mounted-ok" in result.stdout


@pytest.mark.asyncio
async def test_peak_rss_comes_from_reaped_child_not_early_self_report(monkeypatch):
    module = api()
    from tldw_chatbook.Chunking.lab_models import ExecutionReport, RunResult

    request = request_for()
    result = RunResult(
        request=request,
        status="completed",
        report=ExecutionReport(
            chunks=(),
            transformed_text="",
            diagnostics=({"kind": "resources", "peak_rss_bytes": 1},),
        ),
        started_at="start",
        finished_at="end",
        elapsed_ms=0,
        error=None,
    )
    payload = module._encoded(result)
    code = f"import sys,struct; p={payload!r}; b=bytearray(150000000); del b; sys.stdout.buffer.write(struct.pack('!Q',len(p))+p); sys.stdout.buffer.flush()"
    monkeypatch.setattr(module, "_worker_command", lambda: [sys.executable, "-c", code])
    runner = module.LocalPreviewRunner(module.PreviewLimits())
    received = await runner.run(request)
    assert received.status == "completed"
    assert received.report.diagnostics[-1]["peak_rss_bytes"] >= 150000000
    await runner.close()


@pytest.mark.asyncio
async def test_delayed_kill_reap_keeps_ownership_until_wait_succeeds(
    monkeypatch, tmp_path
):
    module = api()
    pidfile = noncooperative(monkeypatch, tmp_path, module)
    original = module.subprocess.Popen
    delayed = []

    def spawn(*args, **kwargs):
        process = original(*args, **kwargs)
        original_kill, original_wait = process.kill, process.wait
        killed = False

        def kill():
            nonlocal killed
            killed = True
            return original_kill()

        def wait(*args, **kwargs):
            if killed and not delayed:
                delayed.append(True)
                raise subprocess.TimeoutExpired("reap", 0.5)
            return original_wait(*args, **kwargs)

        process.kill, process.wait = kill, wait
        return process

    monkeypatch.setattr(module.subprocess, "Popen", spawn)
    runner = module.LocalPreviewRunner(module.PreviewLimits())
    task = asyncio.create_task(runner.run(request_for()))
    pid = await wait_pid(pidfile)
    await runner.cancel()
    assert (await task).status == "canceled"
    assert delayed
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    await runner.close()


@pytest.mark.asyncio
async def test_normalizer_temporary_replacement_is_counted_before_execution(
    monkeypatch,
):
    module = api()
    request = request_for(
        "small sample",
        {
            "chunking": {"method": "words"},
            "preprocessing": [
                {
                    "operation": "normalize_whitespace",
                    "config": {"max_line_breaks": 33554432},
                }
            ],
        },
    )
    monkeypatch.setattr(
        module,
        "_worker_command",
        lambda: pytest.fail("Oversized temporary allocation must not launch"),
    )
    runner = module.LocalPreviewRunner(module.PreviewLimits())
    result = await runner.run(request)
    assert result.status == "limited" and result.error["code"] == "intermediate_limit"
    await runner.close()


@pytest.mark.asyncio
async def test_expired_admission_deadline_does_not_start_a_child(monkeypatch):
    module = api()
    request = request_for()
    monkeypatch.setattr(
        module,
        "_worker_command",
        lambda: pytest.fail("Admission already used the time budget"),
    )
    runner = module.LocalPreviewRunner(module.PreviewLimits(wall_seconds=0.0000001))
    result = await runner.run(request)
    assert result.status == "limited" and result.error["code"] == "time_limit"
    await runner.close()

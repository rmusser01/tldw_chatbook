"""Optional config timing must preserve real Python execution and privacy."""

import json
import sys
import threading
from contextlib import contextmanager
from types import FunctionType, SimpleNamespace

import pytest

from Tests.Backup_Recovery import startup_timing_diagnostic as diagnostic


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_test_progress_tracks_suspended_code_without_retaining_values(cancelled):
    import asyncio

    release = asyncio.Event()
    marker = "private-await-value-must-not-appear"

    async def selected_test(value):
        await release.wait()
        assert observer.snapshot()["active"][0].get("suspended_at") is None
        return value

    observer = diagnostic.ConfigTimings({selected_test.__code__: "test_case"})
    observer.start()
    task = asyncio.create_task(selected_test(marker))
    try:
        await asyncio.sleep(0)
        record = observer.snapshot()
        coordinate = record["active"][0]["suspended_at"]
        assert set(coordinate) == {"file", "function", "line"}
        assert coordinate["function"] == "selected_test"
        assert coordinate["line"] == selected_test.__code__.co_firstlineno + 1
        assert marker not in json.dumps(record)
        if cancelled:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            release.set()
            assert await task == marker
        assert not observer.snapshot()["active"]
    finally:
        release.set()
        if not task.done():
            await task
        observer.close()
    assert sys.monitoring.get_local_events(observer.tool_id, selected_test.__code__) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_handled_await_exception_clears_suspension_before_handler_runs(cancelled):
    import asyncio

    future = asyncio.get_running_loop().create_future()
    marker = object()
    async def selected_test():
        try:
            await future
        except (ValueError, asyncio.CancelledError):
            assert observer.snapshot()["active"][0].get("suspended_at") is None
        return marker
    observer = diagnostic.ConfigTimings({selected_test.__code__: "test_case"})
    observer.start()
    task = asyncio.create_task(selected_test())
    try:
        await asyncio.sleep(0)
        assert "suspended_at" in observer.snapshot()["active"][0]
        if cancelled:
            task.cancel()
        else:
            future.set_exception(ValueError("private-awaited-error"))
        assert await task is marker
        assert not observer.snapshot()["active"]
    finally:
        if not task.done():
            task.cancel()
            await task
        observer.close()


def test_progress_selection_only_observes_exact_collected_test_and_fixed_helpers(tmp_path):
    source = tmp_path.resolve()
    expected = source / "Tests/UI/test_llm_gguf_source_modes.py"
    def located(name, path):
        return FunctionType((lambda: None).__code__.replace(co_name=name, co_filename=str(path)), {})
    selected = located("test_keyboard", expected)
    selected.__globals__.update({
        "_mount_models": located("_mount_models", expected),
        "_settle_pilot_until": located("_settle_pilot_until", expected),
        "_close_context": located("_close_context", expected),
        "not_selected": located("not_selected", expected),
    })
    plugin = diagnostic.ConfigTimingPlugin(source)
    plugin.pytest_collection_modifyitems([SimpleNamespace(obj=selected)])
    assert set(plugin.timings.codes.values()) == {"test_case", "test_mount", "test_settle", "test_close"}
    foreign = located("foreign_test", source / "another_test.py")
    plugin.pytest_collection_modifyitems([SimpleNamespace(obj=foreign)])
    assert len(plugin.timings.codes) == 4


@pytest.mark.parametrize("label", ["config_lock", "config_operation"])
def test_config_timing_records_lock_wait_body_and_retirement(label):
    lock = threading.RLock()

    @contextmanager
    def locked():
        with lock:
            yield token

    token = object()
    observer = diagnostic.ConfigTimings({locked.__wrapped__.__code__: label})
    observer.start()
    try:
        with locked() as result:
            assert result is token
            current = observer.snapshot()
            assert current["active"][0]["phase"] == "body_and_release"
        final = observer.snapshot()
        assert not final["active"]
        record = final["completed"][0]
        assert record["function"] == label
        assert record["outcome"] == "returned"
        assert record["acquisition_seconds"] >= 0
        assert record["body_and_release_seconds"] >= 0
        assert record["thread_cpu_seconds"] >= 0
    finally:
        observer.close()
    assert sys.monitoring.get_tool(observer.tool_id) is None
    assert sys.monitoring.get_local_events(observer.tool_id, locked.__wrapped__.__code__) == 0


def test_active_operation_snapshot_records_only_bounded_live_code_coordinates():
    entered, release = threading.Event(), threading.Event()
    marker = "private-local-value-must-not-appear"

    def operation(private_value):
        entered.set()
        assert release.wait(3)
        return private_value

    observer = diagnostic.ConfigTimings({operation.__code__: "config_operation"})
    observer.start()
    worker = threading.Thread(target=operation, args=(marker,))
    try:
        worker.start()
        assert entered.wait(3)
        snapshot = observer.snapshot()
        active = snapshot["active"][0]
        assert active["thread"] == worker.native_id
        assert 1 <= len(active["frames"]) <= 25
        assert any(row["function"] == "operation" for row in active["frames"])
        assert all(set(row) == {"file", "function", "line"} for row in active["frames"])
        assert marker not in json.dumps(snapshot)
    finally:
        release.set()
        worker.join(3)
        observer.close()
    assert not worker.is_alive()


@pytest.mark.parametrize("operation_source", ["selected", "foreign", "absent"])
def test_config_operation_selection_checks_source_without_importing_it(
    tmp_path, monkeypatch, operation_source
):
    source = tmp_path.resolve()
    module = SimpleNamespace()
    names = (
        "_config_write_lock", "_apply_literal_settings_transaction_locked",
        "_publish_runtime_config_unlocked", "_load_settings_uncached", "get_user_data_dir",
    )
    def located_function(name, path):
        code = (lambda: None).__code__.replace(co_name=name, co_filename=str(path))
        return FunctionType(code, {})

    for name in names:
        setattr(module, name, located_function(name, source / "tldw_chatbook/config.py"))
    if operation_source != "absent":
        path = source / "tldw_chatbook/Backup_Recovery/config_participants.py"
        if operation_source == "foreign":
            path = tmp_path / "foreign.py"
        module._config_participants = SimpleNamespace(operation=located_function("operation", path))
    monkeypatch.setitem(sys.modules, "tldw_chatbook.config", module)
    plugin = diagnostic.ConfigTimingPlugin(source)
    try:
        plugin.pytest_collection_finish()
        if operation_source == "foreign":
            assert plugin.timings.errors == ["ValueError"]
            assert not plugin.timings.owned
        else:
            assert not plugin.timings.errors
            assert plugin.timings.owned
            assert ("config_operation" in plugin.timings.codes.values()) == (operation_source == "selected")
    finally:
        plugin.timings.close()


@pytest.mark.parametrize("raises", [False, True])
def test_config_timing_preserves_original_worker_return_or_exception(raises):
    token = object()
    error = ValueError("private-exception-must-not-appear")
    results = []

    def transaction(private_value):
        if raises:
            raise error
        return private_value

    def run():
        try:
            results.append(transaction(token))
        except ValueError as caught:
            results.append(caught)

    observer = diagnostic.ConfigTimings({transaction.__code__: "transaction"})
    observer.start()
    try:
        worker = threading.Thread(target=run)
        worker.start()
        worker.join(3)
        assert not worker.is_alive()
        assert results == [error if raises else token]
        record = observer.snapshot()
        assert not record["active"]
        assert record["completed"][0]["outcome"] == ("raised" if raises else "returned")
        assert "private-exception" not in json.dumps(record)
        assert str(token) not in json.dumps(record)
    finally:
        observer.close()


def test_config_timing_bounds_records_and_coexists_with_cprofile():
    import cProfile

    def transaction():
        return "private-return-value"

    observer = diagnostic.ConfigTimings({transaction.__code__: "transaction"})
    profiler = cProfile.Profile()
    observer.start()
    try:
        profiler.enable()
        try:
            for _ in range(100):
                assert transaction() == "private-return-value"
        finally:
            profiler.disable()
        record = observer.snapshot()
        assert len(record["completed"]) == 16
        assert not record["active"]
        assert record["counts"] == {"transaction": 100}
        assert set(record["slowest"]) == {"transaction"}
        assert "private-return-value" not in json.dumps(record)
        assert any(row.code is transaction.__code__ and row.callcount == 100 for row in profiler.getstats())
    finally:
        observer.close()


def test_config_timing_callback_failure_does_not_replace_product_result(monkeypatch):
    def transaction():
        return token

    def failed_clock():
        raise RuntimeError("private-observer-error")

    token = object()
    observer = diagnostic.ConfigTimings({transaction.__code__: "transaction"})
    observer.start()
    try:
        monkeypatch.setattr(observer, "clock", failed_clock)
        assert transaction() is token
        record = observer.snapshot()
        assert record["errors"]
        assert set(record["errors"]) == {"RuntimeError"}
        assert "private-observer-error" not in json.dumps(record)
    finally:
        observer.close()


def test_config_timing_does_not_claim_another_tools_slot():
    observer = diagnostic.ConfigTimings({})
    sys.monitoring.use_tool_id(observer.tool_id, "existing-test-profiler")
    try:
        observer.start()
        observer.close()
        assert sys.monitoring.get_tool(observer.tool_id) == "existing-test-profiler"
        assert observer.snapshot()["errors"] == ["ValueError"]
    finally:
        sys.monitoring.free_tool_id(observer.tool_id)


def test_config_timing_retires_a_lock_when_the_body_raises():
    @contextmanager
    def locked():
        yield

    error = RuntimeError("original body failure")
    observer = diagnostic.ConfigTimings({locked.__wrapped__.__code__: "config_lock"})
    observer.start()
    try:
        with pytest.raises(RuntimeError) as caught, locked():
            raise error
        assert caught.value is error
        record = observer.snapshot()
        assert not record["active"]
        assert record["completed"][0]["outcome"] == "raised"
        assert record["completed"][0]["body_and_release_seconds"] >= 0
    finally:
        observer.close()


def test_config_timing_cleanup_failure_keeps_ownership_for_retry(monkeypatch):
    observer = diagnostic.ConfigTimings({})
    observer.start()
    original = sys.monitoring.set_events

    def failed_cleanup(*args):
        raise RuntimeError("private cleanup failure")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(sys.monitoring, "set_events", failed_cleanup)
            observer.close()
        assert observer.owned
        assert observer.snapshot()["errors"] == ["RuntimeError"]
        assert sys.monitoring.get_tool(observer.tool_id) == "backup-startup-config"
        assert sys.monitoring.set_events is original
    finally:
        observer.close()
    assert sys.monitoring.get_tool(observer.tool_id) is None

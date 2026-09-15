"""Optional config timing must preserve real Python execution and privacy."""

import json
import sys
import threading
from contextlib import contextmanager

import pytest

from Tests.Backup_Recovery import startup_timing_diagnostic as diagnostic


def test_config_timing_records_lock_wait_body_and_retirement():
    lock = threading.RLock()

    @contextmanager
    def locked():
        with lock:
            yield token

    token = object()
    observer = diagnostic.ConfigTimings({locked.__wrapped__.__code__: "config_lock"})
    observer.start()
    try:
        with locked() as result:
            assert result is token
            current = observer.snapshot()
            assert current["active"][0]["phase"] == "body_and_release"
        final = observer.snapshot()
        assert not final["active"]
        record = final["completed"][0]
        assert record["function"] == "config_lock"
        assert record["outcome"] == "returned"
        assert record["acquisition_seconds"] >= 0
        assert record["body_and_release_seconds"] >= 0
        assert record["thread_cpu_seconds"] >= 0
    finally:
        observer.close()
    assert sys.monitoring.get_tool(observer.tool_id) is None
    assert sys.monitoring.get_local_events(observer.tool_id, locked.__wrapped__.__code__) == 0


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

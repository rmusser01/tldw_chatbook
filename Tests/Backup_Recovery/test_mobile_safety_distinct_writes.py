"""R144 local mechanics; mocked upstream prefix is never source authority."""

from pathlib import Path
from threading import Event

import pytest

from Tests.Backup_Recovery.test_mobile_safety_parent_groups import local_members
from tldw_chatbook.Backup_Recovery import admission, bootstrap
from tldw_chatbook.Backup_Recovery import mobile_builtin_preservation as preservation
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.limits import _TraceReadLimit
from tldw_chatbook.Backup_Recovery.mobile_preview_budget import path_cost
from tldw_chatbook.Chat.console_trace_read_budget import _TraceReadBudget


@pytest.mark.parametrize(
    "mode",
    [
        "repeat_alias_fresh",
        "overlap",
        "set_limit",
        "lookup_limit",
        "add_limit",
        "resolve_limit",
        "cancel_before",
        "cancel_after_overlap",
    ],
)
def test_local_distinct_write_mechanics(tmp_path, monkeypatch, mode):
    calls, buckets = [], []
    budget, cancel = _TraceReadBudget(), Event()
    holds = len(storage._failed_capture_holds)
    native_overlap = bootstrap._overlap
    native_container = _TraceReadBudget.container
    native_reserve = _TraceReadBudget.reserve
    native_price = admission._price_paths
    native_retire = storage._CaptureFileDescriptors.retire
    with local_members(tmp_path, monkeypatch) as case:
        first, second = tmp_path / "outside-first", tmp_path / "outside-second"
        first.write_bytes(b"first")
        second.write_bytes(b"second")
        alias = tmp_path / "lexical-alias"
        alias.symlink_to(first)
        assert alias != first and alias.samefile(first)
        writes = (
            ("a", first),
            ("b", alias),
            ("c", Path(str(first))),
            ("d", second),
            ("e", alias),
            ("f", second),
        )
        if mode == "overlap":
            writes = (("a", first), ("inside", case.files[0]), ("later", second))
        case.plan.restore = writes
        case.plan.retire = ()
        state = {
            "after_source": False,
            "containers": 0,
            "set_priced": False,
            "injected": False,
        }
        source_members = preservation._mobile_source_members

        def source(*args, **kwargs):
            result = source_members(*args, **kwargs)
            state.update(after_source=True, containers=0, set_priced=False)
            return result

        def overlap(left, right):
            result = native_overlap(left, right)
            assert left == case.root
            calls.append(right)
            if mode == "cancel_after_overlap" and len(calls) == 1:
                cancel.set()
            return result

        def container(actual, count=0):
            result = native_container(actual, count)
            if actual is budget and state["after_source"]:
                state["containers"] += 1
                index = state["containers"]
                if index == 3 and count == 0:
                    state["set_priced"] = True
                if (mode == "set_limit" and index == 2) or (
                    mode == "lookup_limit" and index == 3
                ):
                    native_reserve(actual, actual._remaining)
                    state["injected"] = True
            return result

        def reserve(actual, amount):
            result = native_reserve(actual, amount)
            if (
                mode == "add_limit"
                and actual is budget
                and state["set_priced"]
                and not state["injected"]
                and amount == path_cost(first)
            ):
                native_reserve(actual, actual._remaining)
                state["injected"] = True
            return result

        def price(paths, actual, **kwargs):
            if (
                mode == "resolve_limit"
                and actual is budget
                and paths == (case.root, first)
                and not state["injected"]
            ):
                native_reserve(actual, actual._remaining)
                state["injected"] = True
            return native_price(paths, actual, **kwargs)

        def retire(custody):
            registered = custody in case.session._scopes
            result = native_retire(custody)
            if registered:
                assert (
                    not custody.fds
                    and not custody.directory_iterators
                    and not custody.close_failed
                )
                buckets.append(custody)
            return result

        monkeypatch.setattr(preservation, "_mobile_source_members", source)
        monkeypatch.setattr(bootstrap, "_overlap", overlap)
        monkeypatch.setattr(_TraceReadBudget, "container", container)
        monkeypatch.setattr(_TraceReadBudget, "reserve", reserve)
        monkeypatch.setattr(admission, "_price_paths", price)
        monkeypatch.setattr(storage._CaptureFileDescriptors, "retire", retire)

        def invoke():
            return preservation._mobile_safety_members(
                case.plan,
                case.journal,
                case.parent,
                case.prepared,
                case.receipt,
                case.session,
                case.archive,
                case.original,
                budget,
                cancel,
            )

        if mode == "repeat_alias_fresh":
            for index in range(2):
                state["after_source"] = False
                assert invoke() == case.members
                assert calls == [first, alias, second] * (index + 1)
            assert case.plan.restore is writes
            assert len(buckets) == 2
        elif mode == "overlap":
            with pytest.raises(ValueError, match="mobile_safety_source_not_preserved"):
                invoke()
            assert calls == [first, case.files[0]]
        elif mode.startswith("cancel"):
            if mode == "cancel_before":
                cancel.set()
            with pytest.raises(InterruptedError):
                invoke()
            assert calls == ([] if mode == "cancel_before" else [first, alias, second])
            if mode == "cancel_before":
                assert budget._remaining == 16 * 1024**2 and not buckets
        else:
            with pytest.raises(
                _TraceReadLimit, match="mobile_trace_input_limit"
            ) as raised:
                invoke()
            frames = []
            tb = raised.value.__traceback__
            while tb is not None:
                if tb.tb_frame.f_code.co_name == "_mobile_safety_members":
                    frames.append(tb.tb_frame.f_locals)
                tb = tb.tb_next
            assert len(frames) == 1 and not calls and state["injected"]
            frame = frames[0]
            if mode == "set_limit":
                assert "seen_writes" not in frame
            elif mode in {"lookup_limit", "add_limit"}:
                assert frame["seen_writes"] == set()
            else:
                assert frame["seen_writes"] == {first}
            assert budget._remaining == 0
        assert not case.session._scopes
        if mode != "cancel_before":
            assert buckets
    assert not case.session._active
    assert len(storage._failed_capture_holds) == holds

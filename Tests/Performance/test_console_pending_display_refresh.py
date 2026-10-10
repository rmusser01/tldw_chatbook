"""Real expired display proof plus original finite-reader pending ownership."""

import ast
import asyncio
from collections import Counter
import json
from pathlib import Path
import threading
import time

import pytest

from Tests.Performance.console_storage_unit_observer import OriginalStorageUnitObserver
from Tests.private_profile import private_profile_test

pytestmark = pytest.mark.bootstrap_profile


def _reader_lines(reader):
    """Select original executable assignment lines, never an injected reader."""
    tree = ast.parse(Path(reader.__code__.co_filename).read_text(encoding="utf-8"))
    candidates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "read_current"
        and node.lineno == reader.__code__.co_firstlineno
    ]
    assert len(candidates) == 1
    body = candidates[0]
    admitted = [
        node
        for node in ast.walk(body)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "after"
            for target in node.targets
        )
    ]
    retired = [
        node
        for node in body.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "proof"
            for target in node.targets
        )
        and isinstance(node.value, ast.Constant)
        and node.value.value is None
    ]
    assert len(admitted) == len(retired) == 1
    with_node = next(
        node
        for node in body.body
        if isinstance(node, ast.With)
        and any(admitted[0] is child for child in ast.walk(node))
    )
    assert with_node.end_lineno < retired[0].lineno
    executable = {line for _, _, line in reader.__code__.co_lines() if line is not None}
    assert admitted[0].lineno in executable and retired[0].lineno in executable
    return {"admitted": admitted[0].lineno, "retired": retired[0].lineno}


class PendingReadWitness(OriginalStorageUnitObserver):
    """Add one local LINE gate to the existing original-code census mechanics."""

    def __init__(self, reader, edge, operation_current):
        self.units = Counter()
        self.main = threading.current_thread()
        self.count_ui = False
        super().__init__(self.units, lambda: self.count_ui, self._bump_ui)
        self.reader, self.reader_code = reader, reader.__code__
        self.edge, self.line = edge, _reader_lines(reader)[edge]
        self.operation_current = operation_current
        self.entered, self.release, self.reader_returned = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        self.worker_thread = None
        self.operation_was_registered = None
        self.gate_count = 0
        self.extra_registered = False
        self.line_callback = self._line
        self.closed_receipt = None

    def _bump_ui(self, unit):
        if self.count_ui and threading.current_thread() is self.main:
            self.units[unit] += 1

    def install_reader(self, participants, storage, helper_class):
        try:
            self.install(participants, storage, helper_class)
            self.codes[self._pin(self.reader)] = "finite_readers"
            previous = self.monitor.register_callback(
                self.tool, self.monitor.events.LINE, self.line_callback
            )
            if previous is not None:
                self.monitor.register_callback(
                    self.tool, self.monitor.events.LINE, previous
                )
                raise RuntimeError("line_callback_not_unowned")
            self.extra_registered = True
            self.monitor.set_local_events(
                self.tool,
                self.reader_code,
                self.monitor.events.PY_START
                | self.monitor.events.PY_RETURN
                | self.monitor.events.LINE,
            )
            assert self.monitor.get_events(self.tool) == 0
        except BaseException:
            self.release.set()
            self.close()
            raise

    def _line(self, code, line):
        if not self.active or code is not self.reader_code or line != self.line:
            return
        try:
            frame = self._frame(code)
            assert frame.f_globals is self.reader.__globals__
            del frame
            assert threading.current_thread() is not self.main
            assert self.gate_count == 0
            self.gate_count += 1
            self.worker_thread = threading.current_thread()
            self.operation_was_registered = self.operation_current()
            assert self.operation_was_registered is (self.edge == "admitted")
            self.entered.set()
            assert self.release.wait(10), "original_reader_gate_release_timeout"
        except BaseException as error:
            self.invalid.append("reader_line:" + type(error).__name__)
            self.entered.set()

    def _return(self, code, offset, value):
        super()._return(code, offset, value)
        if self.active and code is self.reader_code:
            if threading.current_thread() is not self.worker_thread:
                self.invalid.append("reader_return_actor_changed")
            self.reader_returned.set()

    def close(self):
        if self.closed_receipt is not None:
            return self.closed_receipt
        self.release.set()
        if self.tool is not None and self.extra_registered:
            try:
                self.monitor.set_local_events(self.tool, self.reader_code, 0)
            except BaseException as error:
                self.invalid.append("reader_local_retirement:" + type(error).__name__)
            try:
                previous = self.monitor.register_callback(
                    self.tool, self.monitor.events.LINE, None
                )
                if previous is not self.line_callback:
                    self.invalid.append("reader_line_owner_changed")
            except BaseException as error:
                self.invalid.append(
                    "reader_callback_retirement:" + type(error).__name__
                )
            self.extra_registered = False
        self.closed_receipt = super().close()
        return self.closed_receipt


async def _entered(witness):
    deadline = time.monotonic() + 5
    while not witness.entered.is_set() and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    assert witness.entered.is_set(), "original_reader_did_not_reach_selected_line"
    assert not witness.invalid


def _census(storage, raw):
    with storage._lock:
        return {
            "ordinary": len(storage._live_leases - set(storage._startups.values())),
            "pending": len(storage._pending_acquisitions),
            "core": len(storage._operations),
            "raw": len(storage._raw_operations),
            "retiring": len(storage._retiring_holds),
            "raw_states": len(raw._states),
            "startups": len(storage._startups),
        }


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize("kind", ["pending", "queued", "direct", "cancel"])
async def test_original_expired_display_pending_refresh(tmp_path, request, kind):
    # Imports occur only after the original private-profile child has selected its profile.
    from Tests.UI import test_console_checked_display_scope as fixture
    from tldw_chatbook import config
    from tldw_chatbook.Backup_Recovery import config_participants as participants
    from tldw_chatbook.Backup_Recovery import raw_participants as raw
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.DB.private_sqlite_process import HelperLease
    from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend

    database, _store, _controller, screen, tasks = fixture._screen(tmp_path)
    witness = None
    receipt = {}
    retries, rendered, rendered_with_live_owner = [], [], []
    try:
        projection = spend.ConsoleReadinessConfigProjection.for_screen(screen)
        assert projection.run(lambda: None) is False
        await asyncio.gather(*tasks)
        assert projection._display_proof in spend._checked_display_proofs
        assert projection in spend._standard_readiness_projections
        assert projection.max_age == 1
        original_mapping, original_proof = projection.value, projection._display_proof
        assert not projection.pending and projection._settled.is_set()
        initial_status = []
        assert (
            projection.run(
                lambda: initial_status.append(spend._checked_display_status(projection))
            )
            is True
        )
        assert initial_status == [
            True
        ], "original issued proof failed its existing fences"
        await asyncio.sleep(
            1.02
        )  # Genuine original clock; neither at nor TTL is mutated.
        assert time.monotonic() - original_proof.at >= projection.max_age

        def operation_current():
            with storage._lock:
                active = getattr(raw._local, "operation", None)
                return (
                    active is not None
                    and active in raw._states
                    and active in storage._raw_operations
                )

        witness = PendingReadWitness(
            projection.read_current,
            "admitted" if kind == "cancel" else "retired",
            operation_current,
        )
        witness.install_reader(participants, storage, HelperLease)
        # Pin the actual original factory/body/wrapper and checked-display functions.
        for function in (
            spend.ConsoleReadinessConfigProjection.__dict__["for_screen"].__func__,
            projection.run.__func__,
            projection._refresh.__func__,
            projection._key.__func__,
            spend._checked_display_status,
            spend._run_checked_display_sync,
            spend.run_console_config_sync,
            fixture._screen,
            fixture._real_screen,
        ):
            witness._pin(function)
        witness.slots.extend(
            [
                (
                    spend.ConsoleReadinessConfigProjection,
                    "for_screen",
                    spend.ConsoleReadinessConfigProjection.__dict__["for_screen"],
                ),
                (projection, "read_current", projection.read_current),
                (spend, "_checked_display_status", spend._checked_display_status),
                (spend, "run_console_config_sync", spend.run_console_config_sync),
                (fixture, "_screen", fixture._screen),
                (config, "load_settings", config.load_settings),
            ]
        )

        def render():
            rendered.append(screen._provider_readiness_app_config() is original_mapping)
            rendered_with_live_owner.append(
                getattr(raw._local, "operation", None) is not None
            )

        def checked_sync():
            return spend.run_console_config_sync(
                render,
                maintenance_paused=False,
                request_retry=lambda: retries.append(True),
                checked_projection=projection,
            )

        if kind == "direct":
            # The original expired NONpending contract deliberately enters fresh native scope.
            witness.count_ui = True
            assert checked_sync() is True
            witness.count_ui = False
            assert not projection.pending and not witness.entered.is_set()
        else:
            before = len(tasks)
            assert projection.run(lambda: None) is True
            owner = tasks[before]
            assert projection.pending and not projection._settled.is_set()
            if kind == "queued":
                # The actual original refresh Task is scheduled but cannot have
                # started in this same synchronous tick under the default factory.
                assert asyncio.get_running_loop().get_task_factory() is None
                assert projection._read_request is None
                assert not witness.entered.is_set() and not owner.done()
                witness.count_ui = True
                receipt["display_result"] = projection.run(checked_sync)
                witness.count_ui = False
                assert projection.pending and not projection._settled.is_set()
                assert projection._read_request is None and not owner.done()
                assert projection.value is original_mapping
                assert projection._display_proof is original_proof
                receipt["genuine_queued_refresh_before_reader_start"] = True
            await _entered(witness)
            assert projection.read_current is witness.reader
            assert not owner.done() and not witness.reader_returned.is_set()
            if kind == "pending":
                witness.count_ui = True
                receipt["display_result"] = projection.run(checked_sync)
                witness.count_ui = False
                assert projection.pending and not projection._settled.is_set()
                assert projection.value is original_mapping
                assert projection._display_proof is original_proof
            elif kind == "cancel":
                for _ in range(2):
                    owner.cancel()
                    await asyncio.sleep(0)
                    assert not owner.done() and projection.pending
                    assert not projection._settled.is_set()
                    assert not witness.reader_returned.is_set()
                receipt["cancel_retained_pending"] = True
            witness.release.set()
            if kind == "cancel":
                with pytest.raises(asyncio.CancelledError):
                    await owner
            else:
                await owner
                assert projection.value == original_mapping
                assert projection._display_proof is not original_proof
                assert projection._display_proof in spend._checked_display_proofs
                assert projection.at == projection._display_proof.at
                receipt["fresh_equal_mapping_can_render"] = projection.run(checked_sync)
                receipt["equal_refresh_added_tasks"] = len(tasks) - before - 1
            assert witness.reader_returned.is_set()
            assert not projection.pending and projection._settled.is_set()
            receipt["worker_return_before_settled"] = True
    finally:
        if witness is not None:
            witness.release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()
        if witness is not None:
            receipt["observer"] = witness.close()
            receipt["main_units"] = dict(witness.units)
            receipt["held_original_reader"] = witness.gate_count == 1
            receipt["held_operation_registered"] = witness.operation_was_registered
            receipt["reader_returned"] = witness.reader_returned.is_set()
        receipt["census"] = _census(storage, raw)
        receipt["render_count"] = len(rendered)
        receipt["rendered_with_live_owner"] = rendered_with_live_owner
        receipt["retry_count"] = len(retries)
        request.node.user_properties.append(
            ("pending_refresh_receipt", json.dumps(receipt, sort_keys=True))
        )

    assert receipt["observer"]["complete"], receipt
    assert receipt["observer"]["global_events"] == 0, receipt
    assert receipt["observer"]["hooks_retired_before_inactive"], receipt
    assert all(
        value == 0 for key, value in receipt["census"].items() if key != "startups"
    ), receipt
    if kind == "direct":
        assert witness.units["config_admissions"] == 1, receipt
        assert rendered_with_live_owner == [True], receipt
    elif kind in {"pending", "queued"}:
        assert receipt["held_original_reader"] and receipt["reader_returned"], receipt
        # Intended genuine RED: current original expiry path opens an enclosing UI scope.
        assert witness.units["config_admissions"] == 0, receipt
        assert receipt["display_result"] is False, receipt
        assert retries == [True], receipt
        assert receipt["fresh_equal_mapping_can_render"] is True, receipt
        assert rendered_with_live_owner == [False], receipt
        if kind == "queued":
            assert receipt["genuine_queued_refresh_before_reader_start"], receipt
    else:
        assert receipt["cancel_retained_pending"] and not rendered, receipt

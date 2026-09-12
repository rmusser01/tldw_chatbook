"""Resource qualification for the persistent Terminal implementation."""

from __future__ import annotations

import asyncio
import gc
import os
from pathlib import Path
from statistics import quantiles
import time

import psutil
import pytest

from tldw_chatbook.Terminal.contracts import (
    AdmissionGate,
    BackendIdentity,
    CleanupAttempt,
    CleanupProof,
    MAX_COLUMNS,
    MAX_IO_CHUNK_BYTES,
    MAX_PENDING_OUTPUT_BYTES,
    MAX_ROWS,
    MAX_SCROLLBACK_BYTES,
    MAX_SCROLLBACK_LINES,
    MAX_SESSION_RECORDS,
    TerminalLaunchRequest,
)
from tldw_chatbook.Terminal.io_actors import TerminalOutputActor
from tldw_chatbook.Terminal.screen_model import TerminalScreenModel
from tldw_chatbook.Terminal.session_manager import TerminalSessionManager


MAX_FOUR_SESSION_RSS_BYTES = 256 * 1024 * 1024
RSS_QUIESCENCE_SECONDS = 5.0


class _InertBackend:
    """Admit manager sessions without starting a shell or runtime bridge."""

    def start(
        self, request: TerminalLaunchRequest, admission: AdmissionGate
    ) -> BackendIdentity:
        del request
        assert admission.admitted is True
        return BackendIdentity(session_id=admission.token)

    def write(self, data: bytes) -> None:
        del data

    def resize(self, columns: int, rows: int) -> None:
        del columns, rows

    def request_priority_close(self) -> None:
        return None

    def cleanup(self, attempt: CleanupAttempt) -> CleanupProof:
        del attempt
        return CleanupProof(True, True, True)

    def finalize_shutdown(self) -> None:
        return None


def test_four_maximum_sessions_report_parent_rss_without_user_shells(
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> None:
    line = b"x" * MAX_COLUMNS
    payload = b"\r\n".join((line,) * (MAX_SCROLLBACK_LINES + MAX_ROWS))
    models: list[TerminalScreenModel] = []
    session_ids: list[str] = []

    def make_model(columns: int, rows: int) -> TerminalScreenModel:
        model = TerminalScreenModel(columns=columns, rows=rows)
        models.append(model)
        return model

    gc.collect()
    process = psutil.Process(os.getpid())
    baseline_rss = process.memory_info().rss
    terminal = TerminalSessionManager(
        read_permitted=lambda: True,
        backend_factory=_InertBackend,
        screen_model_factory=make_model,
    )
    assert terminal.arm(acknowledge_disclosure=True).armed is True

    try:
        for index in range(MAX_SESSION_RECORDS):
            created = terminal.create_session(
                TerminalLaunchRequest(
                    name=f"qualification-{index + 1}",
                    shell="default",
                    start_directory=str(tmp_path),
                    columns=MAX_COLUMNS,
                    rows=MAX_ROWS,
                )
            )
            assert created.admitted is True
            assert created.projection is not None
            session_ids.append(created.projection.session_id)

        assert len(terminal.projections()) == MAX_SESSION_RECORDS
        assert terminal.managed_process_inventory_for_tests() == ()
        for model in models:
            for offset in range(0, len(payload), MAX_IO_CHUNK_BYTES):
                model.feed(payload[offset : offset + MAX_IO_CHUNK_BYTES])

            snapshot = model.snapshot()
            assert snapshot.in_alternate is False
            assert len(snapshot.lines) == MAX_ROWS
            assert len(snapshot.scrollback) == MAX_SCROLLBACK_LINES
            assert snapshot.scrollback_bytes <= MAX_SCROLLBACK_BYTES
            assert all(
                retained.column_width == MAX_COLUMNS for retained in snapshot.scrollback
            )

        time.sleep(RSS_QUIESCENCE_SECONDS)
        rss_delta = max(0, process.memory_info().rss - baseline_rss)
        request.node.user_properties.append(
            ("terminal_four_session_rss_delta_bytes", rss_delta)
        )
        if os.environ.get("TLDW_TERMINAL_QUALIFICATION_HOST") == "1":
            assert rss_delta <= MAX_FOUR_SESSION_RSS_BYTES
    finally:
        terminal.disarm()
        for session_id in session_ids:
            assert terminal.wait_for_cleanup(session_id, timeout_seconds=1.0)
        terminal.finalize_shutdown()


@pytest.mark.asyncio
async def test_ten_second_ansi_flood_keeps_actors_bounded_and_reports_latency(
    request: pytest.FixtureRequest,
) -> None:
    actor = TerminalOutputActor()
    model = TerminalScreenModel(columns=80, rows=24)
    loop = asyncio.get_running_loop()
    duration = 10.0
    stop_at = loop.time() + duration
    sentinel_lateness: list[float] = []
    payload = (b"\x1b[32mterminal-flood\x1b[0m\r\n" * 2_048)[:MAX_IO_CHUNK_BYTES]
    assert actor.offer_output(payload).accepted is True

    async def sentinel() -> None:
        target = loop.time() + 0.1
        while target < stop_at:
            await asyncio.sleep(max(0.0, target - loop.time()))
            sentinel_lateness.append(max(0.0, loop.time() - target))
            target += 0.1

    sentinel_task = asyncio.create_task(sentinel())
    while loop.time() < stop_at:
        if actor.next_read_size >= len(payload):
            assert actor.offer_output(payload).accepted is True
        actor.process_parser_turn(model.feed, visible=False)
        assert actor.pending_bytes <= MAX_PENDING_OUTPUT_BYTES
        await asyncio.sleep(0)
    await sentinel_task

    assert len(sentinel_lateness) >= 90
    p95 = quantiles(sentinel_lateness, n=100, method="inclusive")[94]
    request.node.user_properties.append(("terminal_ansi_flood_p95_ms", p95 * 1_000))
    if os.environ.get("TLDW_TERMINAL_QUALIFICATION_HOST") == "1":
        assert p95 < 0.1

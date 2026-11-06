"""A run start with nothing to reuse composes without a catalog fingerprint (TASK-33620.15.38).

Measured live on origin/dev e8eeae7db9 (Anthropic haiku via the review proxy,
160x45, a fresh profile, timing-only hook with zero-wrap markers): every
Console send took ``run_catalog_fingerprint`` and then composed from the
stores; none reused or kept a catalog. A live send freezes its definition
ceiling at composition (``mcp_definition_capture == "composition"``), and such
a run never reuses, so the fingerprint decided nothing; on a first send there
is nothing kept to compare it with either. At load1 ~120 that check held run
start's UI loop 80-200 ms (task notes).

Such a run start now takes an anchor instead -- the store files' stamps, the
admission epoch and the selected sources, before the composition reads --
and keeps its composition pending. The next run start's own fingerprint
proves it or refuses it. Each test below first checks the first run start
took no fingerprint (red on origin/dev, which takes one before every
composition), then pins one way a change between the anchored composition
and the next run start must be refused: the next run composes from the
stores and offers what they hold now, never the pending catalog. Negative
controls (run by hand, task notes): dropping the refusal each test names
from ``_proven`` turns that test red.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from Tests.UI.test_console_mcp_catalog_reuse import (
    _Console,
    _ids,
    _recomposed,
    _reused,
    _run_start,
    _with_console,
    trust_new_files as trust_new_files,  # noqa: PLC0414 - autouse fixture
)

pytestmark = pytest.mark.bootstrap_profile


class Fingerprints:
    """Count the run starts' catalog fingerprints on the real service."""

    def __init__(self, service, monkeypatch) -> None:
        self.calls = 0
        real = service.run_catalog_fingerprint

        async def counted(*args, **kwargs):
            self.calls += 1
            return await real(*args, **kwargs)

        monkeypatch.setattr(service, "run_catalog_fingerprint", counted)


def _tools():
    from tldw_chatbook.Chat import console_run_start_tools as tools

    return tools


def _pending() -> bool:
    entry = getattr(_tools(), "_entry", None)
    return entry is not None and getattr(entry, "anchor", None) is not None


def _every_file_settled(monkeypatch) -> None:
    """Every store file reads as settled, so only the stamp check can refuse."""
    monkeypatch.setattr(_tools(), "time", SimpleNamespace(time_ns=lambda: 2**62))


async def _cold_run_start(c: _Console, fingerprints: Fingerprints):
    """A run start with nothing kept: it composes, with no fingerprint."""
    _tools().forget()
    before = fingerprints.calls
    provider, *_ = await _run_start(c.console)
    assert fingerprints.calls == before, (
        "a run start with nothing to reuse took a catalog fingerprint"
    )
    return provider


@pytest.mark.asyncio
async def test_a_first_run_start_takes_no_fingerprint_and_the_next_one_proves_it(
    monkeypatch,
):
    """The next run start's fingerprint proves the composition, then it is kept."""

    async def check(c: _Console) -> None:
        fingerprints = Fingerprints(c.service, monkeypatch)
        first = await _cold_run_start(c, fingerprints)
        assert first is not None, "the harness offers built-in MCP tools"
        assert _pending(), "the composition was not kept pending"
        second = await _reused(c.console, c.reads)
        assert fingerprints.calls == 1
        assert not _pending(), "the proof did not keep the composition"
        assert _ids(second) == _ids(first)
        await _reused(c.console, c.reads)
        assert fingerprints.calls == 2

    await _with_console(check)


@pytest.mark.asyncio
async def test_a_store_write_during_the_anchored_composition_is_refused(monkeypatch):
    """The pending catalog is never proven by bytes put back after it read.

    A tool is denied while the anchored composition reads the permission
    states, and the file's anchored bytes are then put back. Every file reads
    as settled, so only the anchored stamp can tell: had the next fingerprint
    proven the composition by its content, the next run would offer the
    denied state's catalog for bytes that allow the tool.
    """
    _every_file_settled(monkeypatch)

    async def check(c: _Console) -> None:
        fingerprints = Fingerprints(c.service, monkeypatch)
        first = await _cold_run_start(c, fingerprints)
        names = list(first._entry_by_llm_name.items())
        (_n1, (one, _s1)), (llm_two, (two, _s2)) = names[0], names[1]
        c.set_state(one, "ask")
        path = c.service.permission_store.path
        before = path.read_bytes()
        real = c.service.effective_tool_states
        fired = []

        def deny_meanwhile(tools, **kwargs):
            if not fired:
                fired.append(True)
                c.set_state(two, "deny")
            return real(tools, **kwargs)

        monkeypatch.setattr(c.service, "effective_tool_states", deny_meanwhile)
        during = await _cold_run_start(c, fingerprints)
        assert fired and llm_two not in _ids(during)
        assert _pending(), "precondition: the composition was kept pending"
        monkeypatch.setattr(c.service, "effective_tool_states", real)
        path.write_bytes(before)
        after = await _recomposed(c.console, c.reads)
        assert llm_two in _ids(after)

    await _with_console(check)


@pytest.mark.asyncio
async def test_an_mcp_server_added_after_the_anchored_composition_is_offered(
    monkeypatch,
):
    """A server saved between the two run starts refuses the pending catalog."""
    from tldw_chatbook.MCP.local_store import LocalExternalMCPProfile

    _every_file_settled(monkeypatch)

    async def check(c: _Console) -> None:
        fingerprints = Fingerprints(c.service, monkeypatch)
        await _cold_run_start(c, fingerprints)
        assert _pending(), "precondition: the composition was kept pending"
        local = c.service.local_service
        local.store.save_profile(
            LocalExternalMCPProfile.from_input_dict(
                {"profile_id": "reusecheck", "command": "echo", "args": ["hi"]}
            )
        )
        local.store.save_discovery_snapshot(
            "reusecheck",
            {"tools": [{"name": "ping", "description": "Ping the server."}]},
        )
        added = await _recomposed(c.console, c.reads)
        assert any("reusecheck" in name for name in _ids(added)), _ids(added)
        await _reused(c.console, c.reads)

    await _with_console(check)


@pytest.mark.asyncio
async def test_a_manifest_provider_selected_after_the_anchored_composition_is_refused(
    monkeypatch,
):
    """The fingerprint names the manifest; the anchor pins the provider it named.

    Store files, stamps and epoch are all unchanged here: only the selected
    manifest provider differs, and the pending catalog holds the built-in
    tools the default provider described.
    """
    _every_file_settled(monkeypatch)

    async def check(c: _Console) -> None:
        fingerprints = Fingerprints(c.service, monkeypatch)
        first = await _cold_run_start(c, fingerprints)
        assert _ids(first), "the harness offers built-in MCP tools"
        assert _pending(), "precondition: the composition was kept pending"
        local = c.service.local_service
        monkeypatch.setattr(local, "manifest_provider", lambda: {"tools": []})
        after = await _recomposed(c.console, c.reads)
        assert not set(_ids(first)) & set(_ids(after)), _ids(after)

    await _with_console(check)


@pytest.mark.asyncio
async def test_a_store_over_another_generation_file_is_refused(monkeypatch, tmp_path):
    """Another store selected after the anchored composition refuses it.

    The newly selected store reads an older, settled generation file with a
    server the composition never saw. The pending catalog's anchor names the
    store it was composed from; the next run composes from the one selected.
    """
    from tldw_chatbook.MCP.local_store import LocalExternalMCPProfile, LocalMCPStore

    _every_file_settled(monkeypatch)

    async def check(c: _Console) -> None:
        fingerprints = Fingerprints(c.service, monkeypatch)
        local = c.service.local_service
        older = LocalMCPStore(tmp_path / "local_mcp_store.json")
        older.save_profile(
            LocalExternalMCPProfile.from_input_dict(
                {"profile_id": "olderserver", "command": "echo", "args": ["hi"]}
            )
        )
        older.save_discovery_snapshot(
            "olderserver",
            {"tools": [{"name": "ping", "description": "Ping the server."}]},
        )
        first = await _cold_run_start(c, fingerprints)
        assert not any("olderserver" in name for name in _ids(first))
        assert _pending(), "precondition: the composition was kept pending"
        selected = local.store
        local.store = older
        try:
            after = await _recomposed(c.console, c.reads)
        finally:
            local.store = selected
        assert any("olderserver" in name for name in _ids(after)), _ids(after)

    await _with_console(check)


@pytest.mark.asyncio
async def test_an_admission_write_after_the_anchored_composition_is_refused(
    monkeypatch,
):
    """A recovery or admission record written meanwhile refuses the pending catalog."""
    from tldw_chatbook.Backup_Recovery import bootstrap

    _every_file_settled(monkeypatch)

    async def check(c: _Console) -> None:
        fingerprints = Fingerprints(c.service, monkeypatch)
        await _cold_run_start(c, fingerprints)
        assert _pending(), "precondition: the composition was kept pending"
        bootstrap.advance_admission_epoch()
        await _recomposed(c.console, c.reads)
        await _reused(c.console, c.reads)

    await _with_console(check)


@pytest.mark.asyncio
async def test_a_just_written_store_is_not_anchored(monkeypatch):
    """A file changed under ``SETTLE_NS`` before the anchor is read again next run."""
    import os

    tools = _tools()
    settle = 2_000_000_000
    monkeypatch.setattr(tools, "SETTLE_NS", settle)
    clock = {"ns": 0}
    monkeypatch.setattr(tools, "time", SimpleNamespace(time_ns=lambda: clock["ns"]))

    def after_last_store_change(c: _Console, delay_ns: int) -> None:
        paths = (c.service.permission_store.path, c.service.local_service.store.path)
        changed = [os.stat(path).st_ctime_ns for path in paths if os.path.exists(path)]
        clock["ns"] = max(changed, default=0) + delay_ns

    async def check(c: _Console) -> None:
        fingerprints = Fingerprints(c.service, monkeypatch)
        first = await _run_start(c.console)
        _name, (tool, _state) = next(iter(first[0]._entry_by_llm_name.items()))
        c.set_state(tool, "ask")
        after_last_store_change(c, settle // 2)
        await _cold_run_start(c, fingerprints)
        assert not _pending(), "a composition over an unsettled file was kept"
        await _recomposed(c.console, c.reads)
        after_last_store_change(c, 2 * settle)
        await _cold_run_start(c, fingerprints)
        assert _pending()
        await _reused(c.console, c.reads)

    await _with_console(check)


@pytest.mark.asyncio
async def test_a_custom_manifest_provider_keeps_the_fingerprint_before_composing(
    monkeypatch,
):
    """Guard (green on origin/dev too): only stock sources are anchored.

    A manifest callback can change what it returns between the composition
    and a later fingerprint, so a run over one keeps the fingerprint before
    its composition, as before.
    """

    async def check(c: _Console) -> None:
        fingerprints = Fingerprints(c.service, monkeypatch)
        manifest = c.service.local_service.manifest_provider
        monkeypatch.setattr(
            c.service.local_service, "manifest_provider", lambda: manifest()
        )
        _tools().forget()
        await _run_start(c.console)
        assert fingerprints.calls == 1
        await _reused(c.console, c.reads)

    await _with_console(check)

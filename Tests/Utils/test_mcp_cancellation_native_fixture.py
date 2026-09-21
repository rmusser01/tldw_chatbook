"""Native cancellation evidence must preserve pre-existing private profile data."""

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from tldw_chatbook.MCP.local_store import LocalExternalMCPProfile, LocalMCPStore

RUNNER = (
    Path(__file__).resolve().parents[2]
    / "Docs/superpowers/qa/2026-09-18-mcp-lifecycle-cancellation/current-dev/native_check.py"
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind",
    [
        "cancellation",
        "refresh",
        "tool-errors",
        "server-actions-alpha",
        "server-actions-beta",
    ],
)
@pytest.mark.parametrize("existing", ["cleanup-demo", "other", None])
async def test_fixture_preserves_existing_profile_and_runtime(
    tmp_path: Path, existing: str | None, kind: str
) -> None:
    """Reject an occupied fixture ID without altering its persisted data.

    Args:
        tmp_path: Isolated directory for the real MCP store.
        existing: Existing profile ID, or None for an empty store.
        kind: Native journey whose fixture admission is being checked.
    """
    fixture_id = {
        "cancellation": "cleanup-demo",
        "refresh": "wire-review",
        "tool-errors": "execution-review",
        "server-actions-alpha": "alpha",
        "server-actions-beta": "beta",
    }[kind]
    if existing == "cleanup-demo":
        existing = fixture_id
    runner = RUNNER
    if kind.startswith("server-actions"):
        runner = (
            Path(__file__).resolve().parents[2]
            / "Docs/superpowers/qa/2026-09-18-mcp-server-actions/current-dev/native_check.py"
        )
    elif kind != "cancellation":
        directory = "mcp-connection-refresh" if kind == "refresh" else "mcp-tool-errors"
        runner = (
            Path(__file__).resolve().parents[2]
            / f"Docs/superpowers/qa/2026-09-18-{directory}/current-dev/native_check.py"
        )
    arguments = () if kind == "cancellation" else (runner.parents[5], tmp_path)
    if kind.startswith("server-actions"):
        arguments = (fixture_id,)
    store_path = tmp_path / "mcp.json"
    store = LocalMCPStore(store_path)
    if existing is not None:
        store.save_profile(
            LocalExternalMCPProfile(
                profile_id=existing, command="python", args=("-m", "owned.server")
            )
        )
        store.save_profile_runtime_state(existing, {"ok": True, "last_action": "test"})
    before = store_path.read_bytes() if store_path.exists() else None
    original = store.get_profile(existing) if existing else None
    original_runtime = store.get_profile_runtime_state(existing) if existing else None

    async def save(payload: dict[str, Any]) -> dict[str, Any]:
        profile = LocalExternalMCPProfile.from_input_dict(payload)
        return store.save_profile(profile).to_input_dict()

    service = SimpleNamespace(
        local_service=SimpleNamespace(store=store),
        save_local_profile=AsyncMock(side_effect=save),
    )
    create = runpy.run_path(str(runner))["_save_fixture_profile"]
    if existing == fixture_id:
        with pytest.raises(ValueError, match="already exists"):
            await create(service, *arguments)
        service.save_local_profile.assert_not_awaited()
        assert store_path.read_bytes() == before
    else:
        assert await create(service, *arguments) == fixture_id
        service.save_local_profile.assert_awaited_once()
        assert store.get_profile(fixture_id).command == (
            "/usr/bin/false"
            if kind == "cancellation" or kind.startswith("server-actions")
            else sys.executable
        )
        if existing:
            assert store.get_profile(existing) == original
            assert store.get_profile_runtime_state(existing) == original_runtime

"""An unreadable permission file must not turn the kill switch off.

TASK-32806.3. `_load_locked` caught `OSError` in the same clause as the
JSON parse errors, so a transient read failure was treated as corruption:
the live `mcp_permissions.json` was renamed to `.bak` and the store
resolved from fresh permissive defaults. Measured before the fix, on a file
whose policy said kill switch ON and global default `off`:

    kill_switch True -> False
    global_default 'off' -> 'ask'
    live file renamed away, .bak created

and the next mutator would have written that reset back permanently. The
method's own docstring already promised the opposite: "uncertain native
persistence failures propagate without resetting policy".

A permission bit is the cheapest way to produce a real read error, so these
tests use one. They skip as root, where it does not apply.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tldw_chatbook.MCP.permission_store import (
    SCHEMA_VERSION,
    MCPPermissionStore,
)


pytestmark = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="chmod 000 does not deny root, so the read error cannot be produced",
)

#: A policy that is the OPPOSITE of the permissive first-run default, so a
#: reset to those defaults is unambiguous rather than coincidental.
RESTRICTIVE = {
    "schema_version": SCHEMA_VERSION,
    "kill_switch": True,
    "profiles": {"default": {"global_default": "off", "servers": {}}},
}


@pytest.fixture()
def store(tmp_path: Path):
    path = tmp_path / "mcp_permissions.json"
    path.write_text(json.dumps(RESTRICTIVE), encoding="utf-8")
    created = MCPPermissionStore(path)
    yield created, path
    # Leave the tree deletable whatever the test did to the mode bits.
    if path.exists():
        os.chmod(path, 0o600)


def _make_unreadable(path: Path) -> None:
    os.chmod(path, 0o000)
    with pytest.raises(OSError):
        path.read_text(encoding="utf-8")


def test_a_read_error_does_not_resolve_from_permissive_defaults(store):
    created, path = store
    _make_unreadable(path)
    with pytest.raises(OSError):
        created.load()


def test_a_read_error_leaves_the_live_file_alone(store):
    created, path = store
    _make_unreadable(path)
    with pytest.raises(OSError):
        created.load()
    assert path.exists(), "the live policy file was renamed away on a read error"
    assert not path.with_suffix(".json.bak").exists(), (
        "a read error produced a corruption backup"
    )


def test_policy_survives_a_transient_read_error(store):
    """The whole point: the failure passes and the user's policy is still there."""
    created, path = store
    _make_unreadable(path)
    with pytest.raises(OSError):
        created.load()
    os.chmod(path, 0o600)

    recovered = created.load()
    assert recovered["kill_switch"] is True
    assert recovered["profiles"]["default"]["global_default"] == "off"


def test_display_getters_fail_closed_rather_than_permissive(store):
    """The read-only getters must not raise into a screen, or lie permissively."""
    created, path = store
    _make_unreadable(path)
    assert created.get_global_default() == "off"


def test_genuine_corruption_still_backs_up_and_resets(store):
    """The behaviour that was correct stays correct: bytes arrived, not JSON."""
    created, path = store
    path.write_text("{ this is not json", encoding="utf-8")

    payload = created.load()

    assert path.with_suffix(".json.bak").exists()
    assert payload["profiles"]["default"]["global_default"] == "ask"
    assert payload["kill_switch"] is False


def test_named_profile_also_fails_closed_on_unreadable(store):
    """Qodo #3: a NAMED profile must also deny on an unreadable store, not fall
    through to the permissive DEFAULT_GLOBAL ('ask') because it is absent from
    the fallback payload."""
    created, path = store
    _make_unreadable(path)
    assert created.get_global_default(profile_id="team-alpha") == "off"
    assert created.get_global_default(profile_id="default") == "off"


def test_builtin_gate_denies_when_the_store_is_unreadable(store):
    """Qodo #1: an unreadable policy store must not let an untagged built-in
    (calculator/date-time) resolve from the built-in allow floor, and the
    kill-switch read must block rather than report 'disabled'."""
    from tldw_chatbook.Agents.builtin_tool_gate import BuiltinToolGate
    from tldw_chatbook.Tools.tool_executor import CalculatorTool

    created, path = store

    class _FakeService:
        def __init__(self, s):
            self.permission_store = s

        def get_kill_switch(self):
            return self.permission_store.get_kill_switch()

        def is_session_approved(self, *_a, **_k):
            return False

    gate = BuiltinToolGate(_FakeService(created))
    _make_unreadable(path)

    assert gate._kill_switch() is True, "unreadable store must block via kill switch"
    assert gate.resolve(CalculatorTool()).state == "deny", (
        "an untagged built-in resolved from the allow floor on an unreadable store"
    )

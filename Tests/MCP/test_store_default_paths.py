"""TASK-855: the three MCP store modules' no-argument defaults must derive
from ``config.get_user_data_dir()`` -- not a hardcoded ``~/.config/tldw_cli``
literal the app never actually uses.

Every real construction site (``app.py``) always passes an explicit
``get_user_data_dir() / <name>`` path, so this covers a *latent* gap: the
permission store and execution log are both derived from a
``LocalMCPStore``'s own ``.path`` (see
``MCP.unified_control_plane_service``'s ``permission_store``/
``execution_log`` properties, ``Path(store.path).with_name(...)``). A store
built with no explicit path anywhere -- in a test, or a future call site --
used to silently place both outside ``Utils.sensitive_paths``' denylist
coverage.
"""

from __future__ import annotations

from pathlib import Path

from tldw_chatbook.MCP.local_store import LocalMCPStore
from tldw_chatbook.MCP.server_target_store import ConfiguredServerTargetStore
from tldw_chatbook.MCP.unified_context_store import UnifiedMCPContextStore


def test_local_mcp_store_default_path_derives_from_get_user_data_dir():
    from tldw_chatbook.config import get_user_data_dir

    store = LocalMCPStore()

    assert store.path == get_user_data_dir() / "local_mcp_store.json"


def test_unified_mcp_context_store_default_path_derives_from_get_user_data_dir():
    from tldw_chatbook.config import get_user_data_dir

    store = UnifiedMCPContextStore()

    assert store.path == get_user_data_dir() / "unified_mcp_context.json"


def test_configured_server_target_store_default_path_derives_from_get_user_data_dir():
    from tldw_chatbook.config import get_user_data_dir

    store = ConfiguredServerTargetStore()

    assert store.path == get_user_data_dir() / "mcp_server_targets.json"


def test_local_mcp_store_default_tracks_a_retargeted_profile(monkeypatch, tmp_path):
    """The default must resolve at construction time via the live accessor.

    TASK-855's point is that the default is not a module-level constant
    baked from whichever profile happened to be active at import. Retarget
    the accessor itself and the next construction must follow it -- a
    baked constant would keep pointing at the old location.

    Note this patches ``get_user_data_dir`` rather than setting
    ``TLDW_CONFIG_PATH``: the data directory is resolved from
    ``[paths] data_dir`` (or the platform default), NOT from where the
    config file happens to live, so retargeting the config path would not
    move it and would prove nothing.
    """
    import tldw_chatbook.config as config_module

    retargeted = tmp_path / "profile-two"
    retargeted.mkdir(parents=True, exist_ok=True)
    # Patch the accessor at its source: the helper imports it inside the
    # function body, so a module-attribute patch on the store module would
    # silently not apply and the test would pass for the wrong reason.
    monkeypatch.setattr(config_module, "get_user_data_dir", lambda: retargeted)

    store = LocalMCPStore()

    assert store.path == retargeted / "local_mcp_store.json"


def test_local_mcp_store_default_is_covered_by_the_sensitive_path_denylist():
    """The whole point of TASK-855: once the default derives from
    ``get_user_data_dir()``, it (and its permission-store/execution-log
    companions derived from it) fall under the denylist's ``user_data_dir``
    file rule -- unlike the old ``~/.config/tldw_cli`` literal default."""
    from tldw_chatbook.Utils.sensitive_paths import is_sensitive_path

    store = LocalMCPStore()

    assert is_sensitive_path(store.path)
    assert is_sensitive_path(Path(store.path).with_name("mcp_permissions.json"))
    assert is_sensitive_path(Path(store.path).with_name("mcp_execution_log.jsonl"))


def test_explicit_path_construction_sites_are_unaffected(tmp_path):
    """AC #3: every real construction site (app.py) always passes an
    explicit path -- this must keep resolving to exactly the path passed,
    unaffected by the default's derivation."""
    explicit_path = tmp_path / "local_mcp_store.json"

    store = LocalMCPStore(explicit_path)

    assert store.path == explicit_path

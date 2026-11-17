"""Default-profile startup parity without app/UI or a timed journey."""

import json
import sys

import pytest

from Tests.private_profile import private_profile_test
from Tests.MCP.test_permission_process_routing import owners  # noqa: F401

pytestmark = pytest.mark.skipif(
    sys.platform != "win32", reason="Native Windows permission startup"
)


@pytest.mark.asyncio
@private_profile_test
async def test_default_profile_permission_startup(request, tmp_path):
    from tldw_chatbook import config
    from tldw_chatbook.MCP import permission_process as process
    from tldw_chatbook.MCP.permission_store import MCPPermissionStore

    assert not config.get_cli_setting("paths", "data_dir", None)
    assert not config.get_cli_setting("Paths", "data_dir", None)
    source = MCPPermissionStore(config.get_user_data_dir() / "mcp_permissions.json")
    source.set_global_default("ask")
    assert source.get_global_default() == "ask"
    broker = process.select(source)
    assert broker.process is not None and broker.unresolved is None
    assert "pywintypes" in broker.origins
    assert "_win32sysloader" in broker.origins
    broker.close()
    assert broker.retired and broker.process.returncode == 0
    (tmp_path / "default-profile-facts.json").write_text(json.dumps(broker.facts))

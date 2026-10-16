"""Ordinary Console setup must not load plugin package-management machinery."""

import json

from Tests.Performance.test_app_import_weight import _run_isolated_python


def test_plugin_facade_defers_package_management_until_use(tmp_path):
    """A metadata facade must not import inspection or open its storage worker."""
    result = _run_isolated_python(
        tmp_path,
        """
import asyncio
import json
import sys
from pathlib import Path

from tldw_chatbook.Plugins.service import PluginService

root = Path.home() / "plugin-profile"
service = PluginService(root, workspace_lookup=lambda _: None)
try:
    print(json.dumps({
        "loaded_management": sorted(name for name in (
            "tldw_chatbook.Plugins.coordinator",
            "tldw_chatbook.Plugins.inspection",
            "tldw_chatbook.Plugins.schemas",
        ) if name in sys.modules),
        "worker_started": service._thread is not None,
        "storage_created": root.exists(),
    }))
finally:
    asyncio.run(service.aclose())
""",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    observed = json.loads(result.stdout.strip().splitlines()[-1])
    assert observed == {
        "loaded_management": [],
        "worker_started": False,
        "storage_created": False,
    }

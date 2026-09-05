"""ADR-097: route discovery must not load unvisited Settings/Scheduling features."""

from __future__ import annotations

import pytest

from Tests.Packaging.test_chat_persistence_import_closure import _run_isolated_python


@pytest.mark.parametrize(
    ("route_module", "deferred", "first_use"),
    [
        (
            "tldw_chatbook.UI.Screens.settings_screen",
            (
                "tldw_chatbook.UI.Screens.settings_rag_profile_adapter",
                "tldw_chatbook.RAG_Search.simplified.enhanced_rag_service_v2",
                "tldw_chatbook.Tool_Packs.service",
                "tldw_chatbook.Widgets.Settings_Widgets.tool_pack_import_review",
            ),
            """
from tldw_chatbook.UI.Screens import settings_rag_profile_adapter as adapter
sentinel = object()
adapter.active_profile_info = lambda: sentinel
assert screen.active_profile_info() is sentinel
from tldw_chatbook.Widgets.Settings_Widgets.tool_profiles_panel import ToolProfilesPanel
assert screen.ToolProfilesPanel is ToolProfilesPanel
""",
        ),
        (
            "tldw_chatbook.UI.Screens.scheduling.schedules_workbench",
            (
                "tldw_chatbook.UI.Screens.scheduling.task_detail",
                "tldw_chatbook.UI.Screens.scheduling.definition_detail",
                "tldw_chatbook.UI.Screens.scheduling.definition_audit_view",
                "tldw_chatbook.UI.Screens.scheduling.forms.automation_definition_form",
                "tldw_chatbook.UI.Screens.scheduling.forms.reminder_form",
            ),
            """
form = screen.ReminderForm
from tldw_chatbook.UI.Screens.scheduling.forms.reminder_form import ReminderForm
assert form is ReminderForm
assert screen.ReminderForm is form
try:
    getattr(screen, '_task31660_missing_export')
except AttributeError:
    pass
else:
    raise AssertionError('Unknown lazy export must raise AttributeError')
""",
        ),
    ],
    ids=["settings", "scheduling"],
)
def test_route_preimport_defers_feature_implementations(
    tmp_path, route_module, deferred, first_use
):
    result = _run_isolated_python(
        tmp_path,
        f"""
import importlib
import sys
import tldw_chatbook.app
import tldw_chatbook.UI.Screens.chat_screen
screen = importlib.import_module({route_module!r})
assert screen.__name__ in sys.modules
resident = [name for name in {deferred!r} if name in sys.modules]
assert not resident, 'Unvisited feature payload loaded during route pre-import: ' + repr(resident)
{first_use}
print('FIRST_USE_CLOSURE_OK')
""",
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-4000:]
    assert "FIRST_USE_CLOSURE_OK" in result.stdout

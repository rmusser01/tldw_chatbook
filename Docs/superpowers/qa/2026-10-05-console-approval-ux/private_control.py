"""Private Windows control: protect original owner home before redirecting it."""

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

repo = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(repo))
assert Path(importlib.util.find_spec("Tests").origin).is_relative_to(repo)
assert Path(importlib.util.find_spec("tldw_chatbook").origin).is_relative_to(repo)
from Tests import real_profile_guard  # noqa: E402 - assert origins first

real_profile_guard.install()
original_home = real_profile_guard._real_home()
root = Path(
    tempfile.mkdtemp(prefix="approval-control-private-", dir=repo / ".superpowers")
)
for relative in ("home", "config", "data", "temp"):
    (root / relative).mkdir(exist_ok=True)
for name, relative in {
    "HOME": "home",
    "USERPROFILE": "home",
    "XDG_CONFIG_HOME": "config",
    "XDG_DATA_HOME": "data",
    "TLDW_TEST_CONFIG_ROOT": ".",
    "TLDW_CONFIG_PATH": "config/config.toml",
    "TEMP": "temp",
    "TMP": "temp",
}.items():
    os.environ[name] = str(root / relative)
os.environ.update(
    PYTHONUTF8="1",
    PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
    HF_HUB_OFFLINE="1",
    PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
)
os.environ.pop("TLDW_TEST_CONFIG_ROOT_OWNER", None)
assert original_home != root / "home"
print(
    json.dumps(
        {
            "interpreter": sys.executable,
            "python": sys.version.split()[0],
            "private_root": str(root),
            "module_origins_verified": True,
            "original_home_protected_before_redirection": True,
        }
    ),
    flush=True,
)
from tldw_chatbook.Backup_Recovery import bootstrap  # noqa: E402 - isolate first

assert bootstrap.effective_config_path().is_relative_to(root)
assert bootstrap.default_bootstrap_root().is_relative_to(root)
admission = bootstrap.startup_permission(
    bootstrap.effective_config_path(), bootstrap.default_bootstrap_root()
)
print(json.dumps({"startup_permission": admission}), flush=True)
if not admission[0]:
    raise SystemExit(1)
import pytest  # noqa: E402 - isolate before collection

target = (
    sys.argv[1] if len(sys.argv) > 1 else "Tests/UI/test_approval_action_ownership.py"
)
assert target in {
    "Tests/MCP/test_redaction_value_shapes.py",
    "Tests/MCP/test_redaction.py",
    "Tests/Chat/test_console_interrupt_host_wiring.py",
    "Tests/Performance/test_app_startup_performance.py",
    "Tests/Scripts/test_regen_approval_card_svg.py",
    "Tests/Chat/test_console_approval_scope_journeys.py",
    "Tests/UI/test_console_approval_ux_journeys.py",
    "Tests/Chat/test_console_raw_shell_progress.py",
    "Tests/Chat/test_console_viewless_hooks.py",
    "Tests/UI/test_console_runtime_ownership.py",
    "Tests/Architecture/test_screen_size_ratchet.py",
    "Tests/Chat/test_console_activity_presentation.py",
    "Tests/MCP/test_control_plane_permissions.py",
    "Tests/Agents/test_approval_observation.py",
    "Tests/Chat/test_console_approval_feedback.py",
    "Tests/UI/test_approval_feedback.py",
    "Tests/Chat/test_console_interrupt_rounds.py",
    "Tests/Chat/test_console_tool_activity.py",
    "Tests/Chat/test_console_raw_shell_revocation.py",
    "Tests/Agents/test_trace_approval_capture.py",
    "Tests/Agents/test_approval_denial_reasons.py",
    "Tests/UI/test_approval_argument_budget.py",
    "Tests/UI/test_approval_row_information_budget.py",
    "Tests/UI/test_approval_details.py",
    "Tests/UI/test_chat_approval_card.py",
    "Tests/UI/test_console_approval_compact_layout.py",
    "Tests/UI/test_approval_batch_geometry.py",
    "Tests/UI/test_console_approval_first_open_render.py",
    "Tests/UI/test_design_token_governance.py",
    "Tests/UI/test_component_pattern_governance.py",
    "Tests/UI/test_css_bundle_sync_guard.py",
    "Tests/UI/test_approval_interaction.py",
    "Tests/Chat/test_approval_presentation.py",
    "Tests/Chat/test_approval_payload_summary.py",
    "Tests/Agents/test_mcp_tool_provider.py",
    "Tests/Agents/test_local_tool_provider.py",
    "Tests/Agents/test_builtin_tool_gate.py",
    "Tests/Agents/test_raw_shell_tool_provider.py",
    "Tests/Chat/test_console_virtual_cli_approval.py",
    "Tests/UI/test_approval_action_ownership.py",
    "Tests/Benchmarks/test_console_approval_latency_measurement.py",
    "Tests/Benchmarks/test_console_approval_private_control.py",
}
selection = sys.argv[2:]
assert not selection or (len(selection) == 2 and selection[0] == "-k")
raise SystemExit(
    pytest.main(
        [
            target,
            "-q",
            "-p",
            "pytest_asyncio.plugin",
            "-p",
            "pytest_timeout",
            "--basetemp=" + str(root / "pytest"),
            *selection,
        ]
    )
)

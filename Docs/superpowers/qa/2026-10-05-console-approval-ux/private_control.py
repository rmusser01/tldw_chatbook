"""Private Windows control: protect original owner home before redirecting it."""

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

repo = Path(__file__).resolve().parents[4]
core = repo / "packages/tldw_profile_core/src"
sys.path.insert(0, str(repo))
sys.path.insert(1, str(core))
assert (
    Path(importlib.util.find_spec("Tests").origin).resolve()
    == (repo / "Tests/__init__.py").resolve()
)
assert (
    Path(importlib.util.find_spec("tldw_chatbook").origin).resolve()
    == (repo / "tldw_chatbook/__init__.py").resolve()
)
assert (
    Path(importlib.util.find_spec("tldw_profile_core").origin).resolve()
    == (core / "tldw_profile_core/__init__.py").resolve()
)
from Tests import real_profile_guard  # noqa: E402 - Verify origins first.

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
from tldw_chatbook.Backup_Recovery import bootstrap  # noqa: E402 - Isolate first.

assert bootstrap.effective_config_path().is_relative_to(root)
assert bootstrap.default_bootstrap_root().is_relative_to(root)
admission = bootstrap.startup_permission(
    bootstrap.effective_config_path(), bootstrap.default_bootstrap_root()
)
print(json.dumps({"startup_permission": admission}), flush=True)
if not admission[0]:
    raise SystemExit(1)
import pytest  # noqa: E402 - Isolate before collection.

target = (
    sys.argv[1] if len(sys.argv) > 1 else "Tests/UI/test_approval_action_ownership.py"
)
assert target in {
    "Tests/Audio/test_dictation_speech_resumed.py",
    "Tests/Audio/test_dictation_recorder_factory.py",
    "Tests/Audio/test_dictation_privacy_allowlist.py",
    "Tests/Audio/test_dictation_capture_release.py",
    "Tests/Audio/test_dictation_stop_join.py",
    "Tests/Performance/test_screen_preimport_payload_budget.py::test_preimport_pass_payload_stays_within_budget",
    "Tests/Audio/test_dictation_meter_import_contract.py",
    "Tests/Tools/test_remote_worker_bundle.py",
    "Tests/Tools/test_file_tool_sandbox.py::test_read_file_refuses_chroma_vector_store_file_even_when_sandbox_root_contains_it",
    "Tests/Tools/test_local_tool_sensitive_paths.py::test_both_file_tool_families_refuse_the_same_denylisted_paths",
    "Tests/RAG/test_config_profiles.py::test_default_rag_profiles_dir_keeps_live_userdata_accessor",

    "Tests/RAG/test_state_paths.py",
    "Tests/Utils/test_sensitive_paths.py",
    "Tests/RAG/simplified/test_vector_store_selection.py",
    "Tests/RAG/simplified/test_chroma_persist_directory.py",
    "Tests/Tools/test_remote_worker_bundle.py::test_cli_regeneration_writes_canonical_utf8_lf_bytes",
    "Tests/Tools/test_remote_worker_bundle.py::test_bundle_rebuild_matches_committed_artifact",
    "Tests/Tools/test_remote_executor_loopback.py::test_committed_bundle_stamp_matches_artifact_prefix",
    "Tests/Tools/test_remote_executor_loopback.py::test_stamp_covers_the_executable_entry_guard",
    "Tests/Tools/test_remote_executor_loopback.py::test_ping_bundle_sha256_matches_the_artifact_stamp",
    "Tests/Tools/test_remote_executor_loopback.py::test_executor_ping_returns_verified_payload",
    "Tests/Tools/test_remote_executor_loopback.py::test_executor_ping_rejects_a_bundle_mismatch",
    "Tests/Tools/test_remote_executor_loopback.py::test_fs_list_long_roundtrip_retains_exclusions_and_plain_output",
    "Tests/Performance/test_app_import_weight.py::test_app_import_own_module_count_stays_at_the_post_diet_size",
    "Tests/Performance/test_ui_ready_module_census.py::test_ui_ready_module_census_stays_at_the_pinned_size",
    "Tests/QA/test_project_chat_uat.py::test_project_chat_uat[wrong-family]",
    "Tests/UI/test_approval_details_privacy.py",
    "Tests/Chat/test_approval_display_controls.py",
    "Tests/UI/test_approval_review_repairs.py",
    "Tests/Web_Server/test_served_relay_publisher.py",
    "Tests/Chat/test_console_relay_approval_integration.py",
    "Tests/CI/test_console_approval_critical_gate.py",
    "Tests/CI/test_ui_pr_gate_census_checker.py",
    "Tests/CI/test_ci_queue_pressure_contract.py",
    "Tests/CI/test_derived_artifacts_workflow.py",
    "Tests/CI/test_console_p0_regression_gate.py",
    "Tests/UI/test_console_response_rules.py",
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
    "Tests/RAG/test_metadata_import_boundary.py",
    "Tests/Chat/test_console_configuration_preparation.py::test_received_profile_depth_capture_uses_lightweight_metadata",
    "Tests/RAG/test_active_config_resolution.py",
    "Tests/RAG/test_config_profiles.py",
    "Tests/RAG/test_config_profiles_import_order.py",
    "Tests/RAG/test_rag_search_facade.py",
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

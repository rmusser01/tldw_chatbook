"""Isolated genuine App coordinator journeys; no fixture-issued authority."""

import asyncio
import importlib
import os
import sys
from pathlib import Path

from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner

os.environ["TLDW_AGENTS_WORKFLOW_RUNS_ENABLED"] = "true"
os.environ["TLDW_AGENTS_AUTOWAKE_ENABLED"] = (
    "true"
    if sys.argv[1] in {"mixed-wake", "planning-wake", "planning-integrity"}
    else "false"
)
os.environ["OPENAI_API_KEY"] = "private-native-workflow-test"
network_guard.install()
real_profile_guard.install()

routes = {
    "consent-unknown": "test_workflow_native_return",
    "wave-release-receiver": "test_workflow_native_return",
    "configuration-entry": "test_workflow_native_return",
    "normal-terminal-config-exit": "test_workflow_native_return",
    "terminal-config-exit": "test_workflow_native_return",
    "finite-preentry": "test_workflow_native_return",
    "finite-planning": "test_workflow_native_return",
    "finite-scope": "test_workflow_native_return",
    "live-bind": "test_workflow_native_return",
    "callback-baseexception": "test_workflow_native_return",
    "wave-capture": "test_workflow_native_return",
    "native-return": "test_workflow_native_return",
    "storage-source": "test_workflow_native_return",
    "dispatch": "test_console_workflow_dispatch",
    "planning-pending": "test_console_workflow_dispatch",
    "hook-retained": "test_console_runtime_shutdown",
    "scheduling": "test_console_workflow_scheduling",
    "fleet-setup": "test_console_workflow_scheduling",
    "mixed-wake": "test_console_workflow_scheduling",
    "planning-wake": "test_console_workflow_scheduling",
    "planning-integrity": "test_console_workflow_scheduling",
    "q1-release": "test_workflow_settlement_delivery",
    "activity": "test_workflow_activity_admission",
    **dict.fromkeys(
        (
            "capacity-notification",
            "capacity-accounting",
            "capacity-lost-wake",
            "capacity-stop",
            "capacity-fenced",
        ),
        "test_workflow_activity_admission",
    ),
    "settlement": "test_workflow_settlement_delivery",
    "unknown-usage": "test_workflow_settlement_delivery",
    "routing": "test_console_workflow_scheduling",
    "next-phase": "test_console_workflow_dispatch",
    "phase-dependency": "test_console_workflow_dispatch",
    "repair": "test_console_workflow_dispatch",
    **dict.fromkeys(
        ("stale-initial", "partial-resume"), "test_console_workflow_dispatch"
    ),
    "pending-acceptance": "test_workflow_settlement_delivery",
    "pending-wrapper": "test_workflow_settlement_delivery",
    **dict.fromkeys(
        (
            "metadata-positive",
            "metadata-late-failure",
            "metadata-external",
            "metadata-coherent",
            "metadata-equal-counters",
            "metadata-helper-held",
            "metadata-helper-before",
            "metadata-superseded",
            "metadata-direct-close",
        ),
        "test_workflow_settlement_delivery",
    ),
    "endpoint": "test_console_workflow_scheduling",
    "direct-planner": "test_console_workflow_scheduling",
    **dict.fromkeys(
        ("pending-shutdown", "pending-prepared", "pending-no-reader"),
        "test_workflow_settlement_delivery",
    ),
    **dict.fromkeys(
        ("child-race", "fleet-race", "inner-wait-exit", "outer-wait-exit"),
        "test_console_workflow_scheduling",
    ),
    **dict.fromkeys(
        (
            "control-positive",
            "control-driver-cancel",
            "control-unknown",
            "control-association",
        ),
        "test_workflow_settlement_delivery",
    ),
}
with user_fixture_default_owner():
    module = importlib.import_module(
        "Tests.Agents.test_workflow_launch_freshness"
        if sys.argv[1] == "excluded-prepared"
        else "Tests.Chat." + routes[sys.argv[1]]
    )
    method = {
        "consent-unknown": "consent_unknown_return_journey",
        "wave-release-receiver": "wave_release_receiver_failure_journey",
        "configuration-entry": "configuration_entry_failure_journey",
        "normal-terminal-config-exit": "normal_terminal_configuration_exit_journey",
        "terminal-config-exit": "terminal_configuration_exit_journey",
        "finite-preentry": "finite_preentry_failure_journey",
        "finite-planning": "finite_planning_scope_journey",
        "finite-scope": "finite_scope_failure_journey",
        "live-bind": "live_bind_source_journey",
        "callback-baseexception": "callback_baseexception_journey",
        "wave-capture": "wave_capture_close_journey",
        "excluded-prepared": "excluded_prepared_path_journey",
        "native-return": "native_return_journey",
        "storage-source": "storage_source_journey",
        "unknown-usage": "unknown_usage_journey",
        "routing": "routing_journey",
        "next-phase": "next_phase_journey",
        "phase-dependency": "next_phase_journey",
        "planning-pending": "pending_planning_journey",
        "hook-retained": "retained_hook_journey",
        "fleet-setup": "foreign_survivor_setup_journey",
        "mixed-wake": "mixed_wake_journey",
        "planning-wake": "planning_wake_journey",
        "planning-integrity": "planning_release_integrity_journey",
        "q1-release": "execution_release_retention_journey",
        "pending-acceptance": "pending_acceptance_journey",
        "pending-wrapper": "pending_wrapper_failure_journey",
    }.get(sys.argv[1], "journey")
    journey = getattr(module, method)
    if sys.argv[1] == "repair":
        journey = module.repair_journey
    elif sys.argv[1].startswith("control-"):
        journey = module.planning_control_return_journey
    elif sys.argv[1] == "endpoint":
        journey = module.endpoint_journey
    elif sys.argv[1] in {
        "child-race",
        "fleet-race",
        "inner-wait-exit",
        "outer-wait-exit",
    }:
        journey = module.capacity_race_journey
    elif sys.argv[1] == "direct-planner":
        journey = module.direct_planner_journey
    elif sys.argv[1] == "pending-shutdown":
        journey = module.pending_shutdown_journey
    elif sys.argv[1] in {"pending-prepared", "pending-no-reader"}:
        journey = module.incomplete_acceptance_journey
    elif sys.argv[1] in {"stale-initial", "partial-resume"}:
        journey = module.refusal_and_remaining_journey
    elif sys.argv[1] in {
        "capacity-notification",
        "capacity-accounting",
        "capacity-lost-wake",
        "capacity-stop",
        "capacity-fenced",
    }:
        journey = module.native_capacity_notification_journey
    elif sys.argv[1] == "metadata-superseded":
        journey = module.superseded_metadata_journey
    elif sys.argv[1] == "metadata-helper-before":
        journey = module.replaced_metadata_before_call_journey
    elif sys.argv[1] == "metadata-coherent":
        journey = module.coherent_metadata_journey
    elif sys.argv[1] == "metadata-equal-counters":
        journey = module.equal_counter_metadata_journey
    elif sys.argv[1] == "metadata-external":
        journey = module.external_metadata_journey
    elif sys.argv[1] in {
        "metadata-positive",
        "metadata-late-failure",
        "metadata-helper-held",
        "metadata-direct-close",
    }:
        journey = module.final_metadata_journey
    asyncio.run(journey(Path(os.environ["XDG_DATA_HOME"]).parent, None))

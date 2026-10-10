"""TASK-33621.27: the Console review's P0 regression tests gate every PR.

The 2026-09-29 Console review shipped its P0 fixes with regression tests that
no pull-request lane ran. The UI Fast Lane runs only
``scripts/ui_pr_gate_census.txt``; the PR Fast Lane runs only the targets its
two pytest steps name; ``Tests/Architecture`` and most of ``Tests/Chat`` run in
no PR lane at all. So a P0 could regress and merge green -- and one did drift
unseen: the TASK-33621.1 rejection-copy tests went red when TASK-34100.5
reworded the provider failure copy, and nothing noticed. (They now assert the
facts the user must see -- provider, 400, the refused tool -- and are gated.)

This file pins where each P0's regression tests run. Removing one from its
lane, or moving a ``bootstrap_profile`` test into a lane whose process it would
poison (TASK-32873), fails the PR Fast Lane here, naming the P0. A whole file
is gated where it is fast (about 30 s locally or less); a slow file is gated by
the node ids that pin the P0 itself.

Measured at TASK-33621.27 (2026-10-10). Left out on purpose, so a reader does
not re-add them blind:

* ``test_console_tray_rebuild_focus.py`` pins a focus restore, not the Save
  .md crash, and measured 218 s for 30 tests under load.

Times above and in the lanes' comments were measured on a host at load
average 35-60, where the gated reference file
``Tests/UI/test_console_send_acknowledgement.py`` (about 10 s on an idle
laptop, under 6 s in CI) took 17 s.
"""

from __future__ import annotations

import importlib.util
import shlex
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = PROJECT_ROOT / ".github" / "workflows" / "derived-artifacts.yml"
CENSUS_CHECKER = PROJECT_ROOT / "scripts" / "check_ui_pr_gate_census.py"

_UI = "Tests/UI/"
_CHOOSE = "Tests/UI/test_console_project_instruction_choose_folder.py::"
_COMPOSER = "Tests/UI/test_console_composer_run_controls.py::"
_LIVE = "Tests/Chat/test_console_compaction_live_session.py::"
_HOOK = "Tests/UI/test_console_hook_review_send_freeze.py::"
_WIZARD = "Tests/Wizards/test_first_run_provider_catalog.py::"
_BLOCKED = "Tests/UI/test_console_blocked_send_recovery.py::"
_SWEEP = "Tests/UI/test_console_row_menu_action_sweep.py::"

#: P0 task -> the gated regression tests (a file, or one test's node id).
P0_REGRESSION_TESTS: dict[str, tuple[str, ...]] = {
    # Default sends to OpenAI/Anthropic 400'd on built-in tool schemas.
    "TASK-33621.1": (
        "Tests/Agents/test_provider_tool_schema_conformance.py",
        # The default send reaches the model; a tool rejection names the
        # tool, not the model; the 400 is logged redacted (5 s, whole file).
        "Tests/Chat/test_console_tool_definition_rejection.py",
    ),
    # Chats with a system prompt, a character or an image refused every send.
    "TASK-33621.2": (
        "Tests/Chat/test_console_capture_send_triggers.py",
        "Tests/Chat/test_console_trace_row_sources.py",
        # Every trigger is pinned at the controller level above; the mounted
        # file is ~22 s a test, so one trigger and the Retry recovery here.
        _BLOCKED + "test_each_trigger_chat_sends_and_renders_the_reply"
        "[system-prompt-before-first-send]",
        _BLOCKED + "test_provenance_failure_recovery_actions_work[retry]",
    ),
    # Automatic compaction never succeeded live and re-billed every send.
    "TASK-33621.3": (
        "Tests/Chat/test_console_compaction_failure.py",
        # The failure_reason column and its repository contract (12 s).
        "Tests/DB/test_chachanotes_v74_auxiliary_failure_reason.py",
        _LIVE + "test_live_automatic_compaction_commits_memory_and_the_send_replies",
        _LIVE + "test_live_compact_now_succeeds_after_real_durable_sends",
        _LIVE + "test_live_failed_compaction_records_reason_and_discloses_spend",
        _LIVE + "test_live_failed_compact_now_says_nothing_changed",
        _LIVE + "test_live_failure_is_not_rebilled_until_the_policy_changes",
    ),
    # Copy as > Save .md... on a conversation row ended the app.
    "TASK-33621.12": (
        "Tests/Console/test_console_markdown_export.py",
        "Tests/UI/test_console_conversation_action_menu.py"
        "::test_row_menu_save_md_writes_the_file_and_keeps_the_app_running",
        "Tests/UI/test_console_conversation_action_menu.py"
        "::test_save_prompt_closes_without_writing_and_restores_focus[escape]",
        # The full sweep (29 live screens) measured 289 s under load.
        _SWEEP + "test_the_sweep_covers_every_declared_action_constant",
        _SWEEP + "test_conversation_row_action_never_ends_the_app[save-markdown]",
    ),
    # Choose folder froze the app; a handler error left Ctrl+Q dead.
    "TASK-33621.13": (
        _CHOOSE + "test_choose_folder_opens_picker_and_applies_the_chosen_binding[choose]",
        _CHOOSE + "test_cancelling_the_picker_returns_to_a_responsive_inspector[escape]",
        "Tests/UI/test_app_keep_alive_dead_screen.py"
        "::test_ctrl_q_quits_after_a_screen_handler_error_while_a_modal_is_up",
    ),
    # The first-run wizard blanked the Provider step and Next quit the app.
    "TASK-33621.14": (
        "Tests/Architecture/test_wizard_lifecycle_guards.py",
        _WIZARD + "test_wizard_provider_list_is_the_settings_picker_set",
        _WIZARD + "test_arrowing_through_every_listed_row_keeps_the_step_live",
        _WIZARD + "test_a_raising_step_handler_keeps_the_step_and_the_keyboard",
        _WIZARD + "test_next_after_a_step_error_never_exits_the_app",
    ),
    # Closing a session tab silently did nothing.
    "TASK-33621.15": (
        "Tests/UI/test_console_session_tab_close.py",
        "Tests/Architecture/test_console_controllers_define_their_self_attributes.py",
    ),
    # A Send needing hook review froze the app (Enter) or the Console.
    "TASK-33621.28": (
        _HOOK + "test_the_app_pump_answers_ctrl_q_while_the_enter_review_is_open",
        _HOOK + "test_a_declined_send_review_settles_and_keeps_the_draft",
        _HOOK + "test_allow_all_resumes_the_captured_send_exactly_once",
    ),
    # Stop was clipped out of the composer row, with no other route.
    "TASK-33625.1": (
        _COMPOSER + "test_running_stop_is_painted_whole_inside_the_action_row",
        _COMPOSER + "test_tab_reaches_a_visibly_focused_stop_and_its_key_stops_the_run",
        _COMPOSER + "test_stop_key_is_advertised_only_while_running_and_stops_the_run",
        _COMPOSER + "test_slash_stop_stops_the_viewed_tabs_run_and_clears_itself",
        _COMPOSER + "test_palette_stop_command_stops_the_viewed_tabs_run",
        _COMPOSER + "test_stop_routes_are_registered_and_documented_in_f1",
        # TASK-33622.2, same PR: Enter on a focused composer button runs it.
        _COMPOSER + "test_enter_on_each_idle_composer_button_runs_its_own_action",
    ),
}

_CASES = [
    pytest.param(task, target, id=f"{task}:{target.rsplit('/', 1)[-1]}")
    for task, targets in P0_REGRESSION_TESTS.items()
    for target in targets
]


def _census_module():
    spec = importlib.util.spec_from_file_location("p0_census", CENSUS_CHECKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _step_targets(name: str) -> tuple[str, ...]:
    jobs = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]
    step = next(s for s in jobs["pr-fast-lane"]["steps"] if s.get("name") == name)
    tokens = shlex.split(step["run"].replace("\\\n", " "))
    return tuple(token for token in tokens if token.startswith("Tests"))


def _lanes() -> dict[str, tuple[str, ...]]:
    census = _census_module()
    return {
        "UI Fast Lane census": tuple(census.read_census(census.CENSUS_PATH)),
        "PR Fast Lane (sandboxed)": _step_targets("Run fast PR contract"),
        "PR Fast Lane (admission-sensitive)": _step_targets(
            "Run admission-sensitive suites"
        ),
    }


def _selected_by(target: str, lane_targets: tuple[str, ...]) -> bool:
    """A lane runs `target` if it lists it, its file, or a directory above it."""
    file_part = target.split("::", 1)[0]
    return any(
        target == listed
        or file_part == listed
        or file_part.startswith(listed.rstrip("/") + "/")
        for listed in lane_targets
    )


def _uses_bootstrap_profile(target: str) -> bool:
    try:
        source = (PROJECT_ROOT / target.split("::", 1)[0]).read_text(encoding="utf-8")
    except OSError:  # reported by test_every_p0_target_names_a_test_that_exists
        return False
    return "pytest.mark.bootstrap_profile" in source


_BOOTSTRAP_CASES = [
    case for case in _CASES if _uses_bootstrap_profile(case.values[1])
]


@pytest.mark.parametrize(("task", "target"), _CASES)
def test_every_p0_regression_test_runs_in_a_required_lane(task, target):
    lanes = _lanes()
    gated_in = [name for name, targets in lanes.items() if _selected_by(target, targets)]
    assert gated_in, (
        f"{task}'s regression test {target} runs in no pull-request lane. "
        "Put it back in scripts/ui_pr_gate_census.txt or the PR Fast Lane step "
        "it came from (bootstrap_profile tests: the admission-sensitive step)."
    )
    assert len(gated_in) == 1, f"{target} runs in more than one lane: {gated_in}"


@pytest.mark.parametrize(("task", "target"), _BOOTSTRAP_CASES)
def test_bootstrap_profile_p0_tests_run_only_in_the_admission_sensitive_step(
    task, target
):
    """TASK-32873: a bootstrap_profile suite's enrollment poisons the sandboxed
    suites that share its process, so it may run only in its own invocation."""
    lanes = _lanes()
    assert _selected_by(target, lanes["PR Fast Lane (admission-sensitive)"]), (
        f"{task}: {target} is bootstrap_profile; gate it in the admission-"
        "sensitive step, not the census or the sandboxed step."
    )


@pytest.mark.parametrize(("task", "target"), _CASES)
def test_every_p0_target_names_a_test_that_exists(task, target):
    """A renamed test makes pytest refuse the whole step with 'not found'."""
    file_part, _, node = target.partition("::")
    path = PROJECT_ROOT / file_part
    assert path.is_file(), f"{task}: {file_part} is gone"
    if node:
        assert _census_module().defines_test(path, node), (
            f"{task}: {file_part} no longer defines {node}"
        )


def test_every_p0_named_by_the_review_has_gated_tests():
    """AC#1 of TASK-33621.27 names these five; none may drop out of the table."""
    for task in (
        "TASK-33621.1",
        "TASK-33621.12",
        "TASK-33621.13",
        "TASK-33621.15",
        "TASK-33625.1",
    ):
        assert P0_REGRESSION_TESTS.get(task), f"{task} has no gated regression test"

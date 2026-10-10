"""TASK-33621.27: the Console review's P0 regression tests gate every PR.

The 2026-09-29 Console review shipped its P0 fixes with regression tests that
no pull-request lane ran. The UI Fast Lane runs only
``scripts/ui_pr_gate_census.txt``; the PR Fast Lane runs only the targets its
two pytest steps name; ``Tests/Architecture`` and most of ``Tests/Chat`` run in
no PR lane at all. So a P0 could regress and merge green -- and two did drift
unseen: the TASK-33621.1 rejection-copy tests went red when TASK-34100.5
reworded the provider failure copy, and the TASK-33621.3 v74 test pinned a
schema version later migrations moved past. Both now assert facts, not
literals, and are gated.

The P0 tests run in the ``console-p0-gate`` job (``Console P0 regression
gate``, aggregated by the required check) and, for the mounted private-profile
ones, in the UI census. This file pins where each one runs. Removing one from
its lane, running one in two lanes, or putting a ``bootstrap_profile`` one in a
sandboxed invocation fails the PR Fast Lane here, naming the P0. A whole file
is gated where it is fast (about 30 s locally or less); a slow file is gated by
the node ids that pin the P0 itself.

Left out on purpose (measured at TASK-33621.27, 2026-10-10), so a reader does
not re-add them blind: ``test_console_tray_rebuild_focus.py`` pins a focus
restore, not the Save .md crash, and measured 218 s for 30 tests under load.

Times in the lanes' comments were measured on a host at load average 35-60,
where the gated reference file ``Tests/UI/test_console_send_acknowledgement.py``
(about 10 s on an idle laptop, under 6 s in CI) took 17 s.
"""

from __future__ import annotations

import ast
import importlib.util
import shlex
from functools import lru_cache
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = PROJECT_ROOT / ".github" / "workflows" / "derived-artifacts.yml"
CENSUS_CHECKER = PROJECT_ROOT / "scripts" / "check_ui_pr_gate_census.py"
CONFTEST = PROJECT_ROOT / "Tests" / "conftest.py"

_CHOOSE = "Tests/UI/test_console_project_instruction_choose_folder.py::"
_COMPOSER = "Tests/UI/test_console_composer_run_controls.py::"
_LIVE = "Tests/Chat/test_console_compaction_live_session.py::"
_HOOK = "Tests/UI/test_console_hook_review_send_freeze.py::"
_WIZARD = "Tests/Wizards/test_first_run_provider_catalog.py::"
_BLOCKED = "Tests/UI/test_console_blocked_send_recovery.py::"
_SWEEP = "Tests/UI/test_console_row_menu_action_sweep.py::"
_MENU = "Tests/UI/test_console_conversation_action_menu.py::"
_KEEP = "Tests/UI/test_app_keep_alive_dead_screen.py::"

#: P0 task -> the gated regression tests (a file, or one test's node id).
P0_REGRESSION_TESTS: dict[str, tuple[str, ...]] = {
    # Default sends to OpenAI/Anthropic 400'd on built-in tool schemas.
    "TASK-33621.1": (
        "Tests/Agents/test_provider_tool_schema_conformance.py",
        # The default send reaches the model; a tool rejection names the
        # tool, not the model; the 400 is logged redacted (5 s, whole file).
        "Tests/Chat/test_console_tool_definition_rejection.py",
        # AC#2: the either/or rule moved from the schema to the handler.
        "Tests/Agents/test_local_tool_provider.py"
        "::test_todo_update_delete_rule_is_enforced_by_the_handler_not_the_schema",
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
        _MENU + "test_row_menu_save_md_writes_the_file_and_keeps_the_app_running",
        _MENU + "test_save_prompt_closes_without_writing_and_restores_focus[escape]",
        _MENU + "test_save_prompt_closes_without_writing_and_restores_focus[cancel]",
        # AC#3: a save that cannot complete names the problem, the app lives.
        _MENU + "test_unwritable_save_path_shows_an_error_and_keeps_the_app_running",
        # AC#5: every conversation-row action id against a live ChatScreen
        # (its ids are computed from the menu models, so the whole test).
        _SWEEP + "test_the_sweep_covers_every_declared_action_constant",
        _SWEEP + "test_conversation_row_action_never_ends_the_app",
    ),
    # Choose folder froze the app; a handler error left Ctrl+Q dead.
    "TASK-33621.13": (
        _CHOOSE + "test_choose_folder_opens_picker_and_applies_the_chosen_binding[choose]",
        _CHOOSE + "test_choose_folder_opens_picker_and_applies_the_chosen_binding[enable]",
        _CHOOSE + "test_cancelling_the_picker_returns_to_a_responsive_inspector[escape]",
        _KEEP + "test_ctrl_q_quits_after_a_screen_handler_error_while_a_modal_is_up",
        _KEEP + "test_a_screen_handler_error_leaves_only_live_screens_on_the_stack",
        # The keep-alive's in-process contract (each well under a second).
        _KEEP + "test_a_dead_content_screen_over_only_the_placeholder_takes_the_loud_exit",
        _KEEP + "test_a_recovery_that_raises_is_logged_before_the_loud_exit",
        _KEEP + "test_retiring_a_dead_screen_resumes_a_worker_awaiting_a_screen_above_it",
        _KEEP + "test_pump_loop_ended_matches_what_textual_does_to_the_pump",
        _KEEP + "test_a_retired_screen_is_torn_down_however_its_loop_ended",
        _KEEP + "test_a_dead_screen_whose_unmount_raises_is_still_dropped",
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
        # AC#4: the disabled-reason copy names the queue state (static).
        _COMPOSER + "test_disabled_reason_names_the_queue_state_not_send_or_setup",
        _COMPOSER + "test_queue_state_labels_match_the_prompt_queue_presentation",
        _COMPOSER + "test_palette_lists_stop_and_the_composer_menu_actions_class_safe",
        # TASK-33622.2, same PR: Enter on a focused composer button runs it.
        _COMPOSER + "test_enter_on_each_idle_composer_button_runs_its_own_action",
    ),
}

_CASES = [
    pytest.param(task, target, id=f"{task}:{target.rsplit('/', 1)[-1]}")
    for task, targets in P0_REGRESSION_TESTS.items()
    for target in targets
]

#: The lanes that are not one sandboxed pytest invocation shared with
#: unrelated suites. The census already runs bootstrap_profile files green
#: (test_console_send_acknowledgement.py, the Chat settings files), and the
#: admission-sensitive steps exist for them (TASK-32873).
_BOOTSTRAP_OK = (
    "UI Fast Lane census",
    "PR Fast Lane (admission-sensitive)",
    "Console P0 gate (admission-sensitive)",
)


@lru_cache(maxsize=1)
def _census_module():
    spec = importlib.util.spec_from_file_location("p0_census", CENSUS_CHECKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _step_targets(job: str, name: str) -> tuple[str, ...]:
    jobs = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]
    step = next(s for s in jobs[job]["steps"] if s.get("name") == name)
    tokens = shlex.split(step["run"].replace("\\\n", " "))
    return tuple(token for token in tokens if token.startswith("Tests"))


def _lanes() -> dict[str, tuple[str, ...]]:
    census = _census_module()
    return {
        "UI Fast Lane census": tuple(census.read_census(census.CENSUS_PATH)),
        "PR Fast Lane (sandboxed)": _step_targets("pr-fast-lane", "Run fast PR contract"),
        "PR Fast Lane (admission-sensitive)": _step_targets(
            "pr-fast-lane", "Run admission-sensitive suites"
        ),
        "Console P0 gate (sandboxed)": _step_targets(
            "console-p0-gate", "Run the sandboxed Console P0 regression tests"
        ),
        "Console P0 gate (admission-sensitive)": _step_targets(
            "console-p0-gate", "Run the admission-sensitive Console P0 regression tests"
        ),
    }


def _file(target: str) -> str:
    return target.split("::", 1)[0].rstrip("/")


def _under(path: str, directory: str) -> bool:
    return path.startswith(directory.rstrip("/") + "/")


def _selected_by(target: str, lane_targets: tuple[str, ...]) -> bool:
    """A lane runs `target` if it lists it, its file, or a directory above it."""
    return any(
        target == listed or _file(target) == listed or _under(_file(target), listed)
        for listed in lane_targets
    )


def _overlaps(target: str, lane_targets: tuple[str, ...]) -> bool:
    """A lane runs some of `target`'s tests: either direction of containment.

    Unlike `_selected_by`, a whole-file `target` also overlaps a lane that
    lists one of its node ids (that node would run twice), and a node id
    overlaps a lane listing its file or a directory above it.
    """
    for listed in lane_targets:
        if listed == target or _selected_by(target, (listed,)):
            return True
        if "::" not in target and (_file(listed) == target or _under(_file(listed), target)):
            return True
    return False


def _conftest_bootstrap_filenames() -> frozenset[str]:
    """File names ``Tests/conftest.py`` keeps on the bootstrap profile.

    Read from its ``keep_bootstrap_profile`` expression: every string set it
    compares ``request.node.path.name`` against. A file in that set behaves
    like a ``bootstrap_profile``-marked one without carrying the marker.
    """
    tree = ast.parse(CONFTEST.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "keep_bootstrap_profile"
            for t in node.targets
        )):
            continue
        for compare in ast.walk(node.value):
            if not isinstance(compare, ast.Compare):
                continue
            left = ast.unparse(compare.left)
            if left != "request.node.path.name":
                continue
            for comparator in compare.comparators:
                if isinstance(comparator, ast.Set):
                    names.update(
                        element.value
                        for element in comparator.elts
                        if isinstance(element, ast.Constant)
                        and isinstance(element.value, str)
                    )
    return frozenset(names)


def _marks_bootstrap(node: ast.AST) -> bool:
    return any(
        isinstance(sub, ast.Attribute)
        and sub.attr == "bootstrap_profile"
        and ast.unparse(sub.value) in {"pytest.mark", "mark"}
        for sub in ast.walk(node)
    )


def _uses_bootstrap_profile(target: str) -> bool:
    """Whether any test `target` selects runs on the bootstrap profile.

    The conftest's real rule: the ``bootstrap_profile`` marker (module
    ``pytestmark``, a class or the function's decorators -- read by AST, so a
    comment naming it does not count) or a file name in its bootstrap set.
    """
    path = PROJECT_ROOT / _file(target)
    if path.name in _conftest_bootstrap_filenames():
        return True
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return False  # reported by test_every_p0_target_resolves
    module_marked = any(
        isinstance(item, (ast.Assign, ast.AnnAssign))
        and "pytestmark" in ast.unparse(item.targets[0] if isinstance(item, ast.Assign) else item.target)
        and _marks_bootstrap(item.value)
        for item in tree.body
        if getattr(item, "value", None) is not None
    )
    if module_marked:
        return True
    names = target.split("::")[1:]
    if names:
        names[-1] = names[-1].split("[", 1)[0]
    body = tree.body
    for name in names:  # a node id: only its own class/function decorators
        match = next(
            (item for item in body if getattr(item, "name", None) == name), None
        )
        if match is None:
            return False
        if any(_marks_bootstrap(d) for d in getattr(match, "decorator_list", [])):
            return True
        body = getattr(match, "body", [])
    if names:
        return False
    return any(  # a whole file: any marked function or class in it
        _marks_bootstrap(decorator)
        for item in ast.walk(tree)
        for decorator in getattr(item, "decorator_list", [])
    )


_BOOTSTRAP_CASES = [case for case in _CASES if _uses_bootstrap_profile(case.values[1])]


@pytest.mark.parametrize(("task", "target"), _CASES)
def test_every_p0_regression_test_runs_in_exactly_one_required_lane(task, target):
    lanes = _lanes()
    assert any(_selected_by(target, targets) for targets in lanes.values()), (
        f"{task}'s regression test {target} runs in no pull-request lane. "
        "Put it back in scripts/ui_pr_gate_census.txt or the Console P0 "
        "regression gate's step it came from."
    )
    overlapping = [name for name, targets in lanes.items() if _overlaps(target, targets)]
    assert len(overlapping) == 1, f"{target} runs in more than one lane: {overlapping}"


@pytest.mark.parametrize(("task", "target"), _BOOTSTRAP_CASES)
def test_bootstrap_profile_p0_tests_stay_out_of_sandboxed_invocations(task, target):
    """TASK-32873: a bootstrap_profile suite's enrollment poisons the sandboxed
    suites that share its pytest process, so a P0 one runs in an
    admission-sensitive step or in the census (which already runs
    bootstrap-marked files green), never in a sandboxed step."""
    lanes = _lanes()
    found = [name for name, targets in lanes.items() if _selected_by(target, targets)]
    assert found and set(found) <= set(_BOOTSTRAP_OK), (
        f"{task}: {target} is bootstrap_profile but runs in {found}; gate it in "
        "an admission-sensitive step."
    )


@pytest.mark.parametrize(("task", "target"), _CASES)
def test_every_p0_target_resolves(task, target):
    """A renamed test or parametrize id makes pytest exit 4: nothing runs."""
    file_part, _, node = target.partition("::")
    path = PROJECT_ROOT / file_part
    assert path.is_file(), f"{task}: {file_part} is gone"
    if node:
        reason = _census_module().resolve_node(path, node)
        assert reason is None, f"{task}: {target}: {reason}"


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


def test_overlap_check_sees_both_directions():
    """Review finding: a whole-file P0 in one lane plus a node id of it in
    another used to read as "one lane"."""
    whole = "Tests/Chat/test_x.py"
    node = "Tests/Chat/test_x.py::test_y"
    assert _overlaps(whole, (node,))
    assert _overlaps(node, (whole,))
    assert _overlaps(node, ("Tests/Chat",))
    assert not _overlaps(node, ("Tests/Chat/test_x.py::test_z",))


def test_bootstrap_detection_uses_the_conftest_rule_not_a_substring():
    """Review finding: the substring search missed the conftest's filename set
    and could match a comment."""
    assert "test_mcp_workbench.py" in _conftest_bootstrap_filenames()
    assert _uses_bootstrap_profile("Tests/UI/test_mcp_workbench.py")
    # This file names the marker in comments and strings, and is not marked.
    assert not _uses_bootstrap_profile("Tests/CI/test_console_p0_regression_gate.py")
    # Node-level marker: only the marked node of a mixed file is bootstrap.
    assert _uses_bootstrap_profile("Tests/Chat/test_console_compaction_failure.py")

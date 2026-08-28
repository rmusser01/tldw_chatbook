"""Source-inspected evidence for the amended Console Wave 6 closeout."""

from __future__ import annotations

import ast
import subprocess
from collections import Counter
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCREEN_RELATIVE_PATH = "tldw_chatbook/UI/Screens/chat_screen.py"
_SCREEN_PATH = _REPO_ROOT / _SCREEN_RELATIVE_PATH
_RATCHET_PATH = _REPO_ROOT / "Tests/Architecture/test_screen_size_ratchet.py"
_GIT_TIMEOUT_SECONDS = 10

POST_IMAGE_BASE = "8d806b71d9c5ae7ed333ccb42780f6b2ea68acd0"
FINAL_WAVE6_DELIVERY_BASE = "87791f85533d883341a6b52489660c9e1a67223d"
AMENDMENT_BASE = "d20dd733b72148818f4491943136edfa68494c68"
POST_IMAGE_COUNTS = (22_172, 712)
FINAL_WAVE6_DELIVERY_COUNTS = (19_863, 630)
AMENDMENT_COUNTS = (19_884, 632)
IMMUTABLE_BUDGETS = (17_727, 593)
FINAL_CLOSEOUT_COUNTS = (17_037, 565)

# Each tuple is (task base, delivered revision, base counts, delivered counts).
TASK_DELIVERIES = (
    (
        "2e115cf042ae440d447c14a69fe1da069a2ab0cd",
        "49d47c2e13f2d925a401f72ba79e849bd97889ef",
        (22_260, 712),
        (20_943, 681),
    ),
    (
        "2ff12ac50b0d7a73599f34e796ca9e933f40a4e8",
        "520b1ec127c6137ea96a89fc66b68fd502ae3533",
        (20_943, 681),
        (19_920, 660),
    ),
    (
        "6ffd74c591b443d2e271559f250b2792386493b3",
        "fc1b21f2612da44f409253923b897fc3a6f40bc4",
        (21_791, 701),
        (20_593, 668),
    ),
    (
        "ab8eb1e7d37d7e20a40e9b3da12392f7b7922db0",
        "f4c45fc14a47d79ae86c0c58bd97af0a759a6f87",
        (20_587, 668),
        (20_202, 655),
    ),
    (
        "fdc6ad663135c27db1ff9aa550ab3cb45672cab2",
        "cbb2be574c25040eaf4b51be32fce444c939c46e",
        (20_526, 658),
        (20_349, 652),
    ),
    (
        "c352b407841b4ee774f33c915eeb86723c17033b",
        "f2e2749939d689bd79c2363d6e64f0ba920ec857",
        (20_486, 656),
        (19_995, 640),
    ),
    (
        "a2c09daa81935dbc59cf487b96128a729d843382",
        "043e0415798273f0a20a100f4c924cdcb5e68916",
        (20_250, 641),
        (19_906, 633),
    ),
    (
        "527152ad3e7d1f12dc828e5dba941867cdddf902",
        "73a43c71f59becfcafc717cd0db1aeb9a58bf10e",
        (19_906, 633),
        (19_883, 631),
    ),
)

REALTIME_METHODS = frozenset(
    {
        "_append_console_realtime_row",
        "_begin_console_realtime_reply_audio",
        "_build_console_realtime_callbacks",
        "_build_console_realtime_session",
        "_build_console_realtime_sink",
        "_close_console_realtime_resources",
        "_close_console_realtime_session",
        "_connect_console_realtime",
        "_console_realtime_adopt_transcript",
        "_console_realtime_api_key",
        "_console_realtime_begin_reconnect",
        "_console_realtime_connect_failed",
        "_console_realtime_exit_loop",
        "_console_realtime_exit_message",
        "_console_realtime_failure_token",
        "_console_realtime_fallback_to_pipeline",
        "_console_realtime_instructions",
        "_console_realtime_marshal",
        "_console_realtime_mode_changed",
        "_console_realtime_playback_finished",
        "_console_realtime_played_ms",
        "_console_realtime_row_metadata",
        "_console_realtime_seed_items",
        "_console_realtime_seed_text",
        "_console_realtime_silence_speech",
        "_end_console_realtime_reply_audio",
        "_enter_console_realtime_loop",
        "_finish_console_realtime_reply_row",
        "_handle_console_realtime_intent",
        "_mark_console_realtime_transcript_empty",
        "_note_console_realtime_audio_unavailable",
        "_on_console_realtime_audio_delta",
        "_on_console_realtime_closed",
        "_on_console_realtime_error",
        "_on_console_realtime_first_audio",
        "_on_console_realtime_frames",
        "_on_console_realtime_input_transcript",
        "_on_console_realtime_output_transcript_delta",
        "_on_console_realtime_ready",
        "_on_console_realtime_reply_done",
        "_on_console_realtime_reply_started",
        "_on_console_realtime_sink_event",
        "_on_console_realtime_speech_started",
        "_on_console_realtime_transcription_usage",
        "_on_console_realtime_turn_committed",
        "_on_console_realtime_usage",
        "_persist_console_realtime_event",
        "_pump_console_realtime_audio",
        "_release_console_realtime_state",
        "_repaint_console_realtime_chip",
        "_sanitize_console_realtime_failure",
        "_send_console_realtime_text_turn",
        "_set_console_realtime_transcript_status",
        "_start_console_realtime_connect",
        "_start_console_realtime_tap",
        "_teardown_console_realtime_loop",
        "_tick_console_realtime",
    }
)

REVIEW_SELECTION_METHODS = frozenset(
    {
        "_build_console_changed_files_state",
        "_console_change_review_provider",
        "_console_change_review_run_id",
        "_console_change_review_workspace_roots",
        "_console_changed_files_scope",
        "_console_changed_files_section_enabled",
        "_console_review_notes_flow",
        "_console_selection_feedback_flow",
        "_console_selection_quote_requested",
        "_create_console_selection_note",
        "_dismiss_console_selection_menus_outside_transcript",
        "_dispatch_console_changed_files_worker",
        "_land_console_changed_files",
        "_land_console_changed_files_empty",
        "_load_console_annotation_previews",
        "_on_console_change_review_dismissed",
        "_open_change_review",
        "_record_console_feedback_event",
        "_sync_console_annotation_discovery",
        "_sync_console_changed_files_if_scope_changed",
        "_sync_console_changed_files_section",
        "action_open_trajectory_view",
        "handle_console_changed_files_selected",
        "on_console_review_notes_requested",
        "on_console_selection_feedback_requested",
        "on_console_selection_note_requested",
    }
)

REALTIME_DELEGATE_METHODS: frozenset[str] = frozenset()
REALTIME_STAY_METHODS = frozenset(
    {
        "_repaint_console_realtime_chip",
    }
)
REALTIME_MOVE_METHODS = (
    REALTIME_METHODS - REALTIME_DELEGATE_METHODS - REALTIME_STAY_METHODS
)

REVIEW_SELECTION_ON_DELEGATE_METHODS = frozenset(
    {
        "handle_console_changed_files_selected",
        "on_console_selection_feedback_requested",
        "on_console_selection_note_requested",
    }
)
REVIEW_SELECTION_ACTION_DELEGATE_METHODS = frozenset({"action_open_trajectory_view"})
REVIEW_SELECTION_DELEGATE_METHODS = (
    REVIEW_SELECTION_ON_DELEGATE_METHODS | REVIEW_SELECTION_ACTION_DELEGATE_METHODS
)
REVIEW_SELECTION_STAY_METHODS = frozenset(
    {
        "_console_change_review_run_id",
        "_console_review_notes_flow",
        "_console_selection_quote_requested",
        "_dismiss_console_selection_menus_outside_transcript",
        "_open_change_review",
        "_sync_console_changed_files_section",
        "on_console_review_notes_requested",
    }
)
REVIEW_SELECTION_MOVE_METHODS = (
    REVIEW_SELECTION_METHODS
    - REVIEW_SELECTION_DELEGATE_METHODS
    - REVIEW_SELECTION_STAY_METHODS
)

MAX_DELEGATE_LINES = 5

REALTIME_INTERNAL_ONLY_METHODS = frozenset(
    {
        "_begin_console_realtime_reply_audio",
        "_console_realtime_begin_reconnect",
        "_start_console_realtime_connect",
        "_teardown_console_realtime_loop",
    }
)


def _source_file_at_revision(revision: str, path: str) -> str:
    """Read ``path`` at ``revision`` with a bounded git subprocess."""
    return subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT_SECONDS,
    ).stdout


def _source_at_revision(revision: str) -> str:
    """Read the frozen screen source at ``revision``."""
    return _source_file_at_revision(revision, _SCREEN_RELATIVE_PATH)


def _files_containing_at_revision(revision: str, token: str) -> set[str]:
    """Return production paths containing ``token`` at ``revision``."""
    output = subprocess.run(
        ["git", "grep", "--name-only", "-F", token, revision, "--", "tldw_chatbook"],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT_SECONDS,
    ).stdout
    return {line.split(":", 1)[1] for line in output.splitlines()}


def _source_references_method(source: str, method_name: str) -> bool:
    """Return whether executable syntax references ``method_name`` externally."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and node.attr == method_name:
            return True
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"getattr", "hasattr"}
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value == method_name
        ):
            return True
    return False


def _external_method_reference_paths(revision: str, method_name: str) -> set[str]:
    """Return non-screen production paths that reference ``method_name``."""
    candidate_paths = _files_containing_at_revision(revision, method_name)
    candidate_paths.discard(_SCREEN_RELATIVE_PATH)
    return {
        path
        for path in candidate_paths
        if _source_references_method(
            _source_file_at_revision(revision, path), method_name
        )
    }


def _screen_class(source: str) -> ast.ClassDef:
    """Return the sole top-level ``ChatScreen`` class in ``source``."""
    return next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "ChatScreen"
    )


def _method_nodes(
    owner: ast.ClassDef,
) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Return every direct method definition, including property setters."""
    return [
        node
        for node in owner.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _methods(owner: ast.ClassDef) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    """Return direct methods by name for uniquely named candidates."""
    return {node.name: node for node in _method_nodes(owner)}


def _counts(source: str) -> tuple[int, int]:
    """Return physical lines and direct ``ChatScreen`` methods."""
    return len(source.splitlines()), len(_method_nodes(_screen_class(source)))


def _span(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Return a definition's physical span without decorator lines."""
    return node.end_lineno - node.lineno + 1


def _binding_actions(owner: ast.ClassDef) -> set[str]:
    """Return Textual action names declared in direct ``BINDINGS`` assignments."""
    return {
        call.args[1].value
        for assignment in owner.body
        if isinstance(assignment, (ast.Assign, ast.AnnAssign))
        for target in (
            assignment.targets
            if isinstance(assignment, ast.Assign)
            else [assignment.target]
        )
        if isinstance(target, ast.Name) and target.id == "BINDINGS"
        for call in ast.walk(assignment.value)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "Binding"
        and len(call.args) >= 2
        and isinstance(call.args[1], ast.Constant)
        and isinstance(call.args[1].value, str)
    }


def _require_candidates(
    methods: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
    expected: frozenset[str],
) -> None:
    """Fail when an exact characterized candidate is absent."""
    missing = expected - methods.keys()
    assert not missing, f"characterized methods missing: {sorted(missing)}"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("callback = screen._target", True),
        ('callback = getattr(screen, "_target")', True),
        ('available = hasattr(screen, "_target")', True),
        ('"""Documentation mentioning _target."""', False),
        ("screen._other()", False),
    ],
)
def test_source_method_reference_detection(source: str, expected: bool) -> None:
    """Detect executable external references without matching documentation."""
    assert _source_references_method(source, "_target") is expected


@pytest.mark.unit
def test_closeout_evidence_explains_the_remaining_absolute_deficit() -> None:
    """Separate completed extraction gains from concurrent screen growth."""
    assert _counts(_source_at_revision(POST_IMAGE_BASE)) == POST_IMAGE_COUNTS
    assert (
        _counts(_source_at_revision(FINAL_WAVE6_DELIVERY_BASE))
        == FINAL_WAVE6_DELIVERY_COUNTS
    )
    assert _counts(_source_at_revision(AMENDMENT_BASE)) == AMENDMENT_COUNTS

    task_line_reduction = 0
    task_method_reduction = 0
    for base, delivered, base_counts, delivered_counts in TASK_DELIVERIES:
        assert _counts(_source_at_revision(base)) == base_counts
        assert _counts(_source_at_revision(delivered)) == delivered_counts
        task_line_reduction += base_counts[0] - delivered_counts[0]
        task_method_reduction += base_counts[1] - delivered_counts[1]

    assert (task_line_reduction, task_method_reduction) == (4_958, 130)
    expected_without_concurrent_growth = (
        POST_IMAGE_COUNTS[0] - task_line_reduction,
        POST_IMAGE_COUNTS[1] - task_method_reduction,
    )
    concurrent_growth = (
        AMENDMENT_COUNTS[0] - expected_without_concurrent_growth[0],
        AMENDMENT_COUNTS[1] - expected_without_concurrent_growth[1],
    )
    assert expected_without_concurrent_growth == (17_214, 582)
    assert concurrent_growth == (2_670, 50)
    assert (
        AMENDMENT_COUNTS[0] - IMMUTABLE_BUDGETS[0],
        AMENDMENT_COUNTS[1] - IMMUTABLE_BUDGETS[1],
    ) == (2_157, 39)

    live_counts = _counts(_SCREEN_PATH.read_text(encoding="utf-8"))
    assert live_counts == FINAL_CLOSEOUT_COUNTS

    ratchet_tree = ast.parse(_RATCHET_PATH.read_text(encoding="utf-8"))
    budget_assignment = next(
        node
        for node in ratchet_tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "_BUDGETS"
    )
    assert budget_assignment.value is not None
    budgets = ast.literal_eval(budget_assignment.value)
    assert budgets[_SCREEN_RELATIVE_PATH] == ("ChatScreen", *FINAL_CLOSEOUT_COUNTS)


@pytest.mark.unit
def test_closeout_candidates_have_exact_non_vacuous_margin() -> None:
    """Lock both candidate families and their conservative closeout margin."""
    source = _source_at_revision(AMENDMENT_BASE)
    owner = _screen_class(source)
    nodes = _method_nodes(owner)
    name_counts = Counter(node.name for node in nodes)
    methods = _methods(owner)

    assert len(REALTIME_METHODS) == 57
    assert len(REVIEW_SELECTION_METHODS) == 26
    assert (
        len(REALTIME_MOVE_METHODS),
        len(REALTIME_DELEGATE_METHODS),
        len(REALTIME_STAY_METHODS),
    ) == (56, 0, 1)
    assert (
        len(REVIEW_SELECTION_MOVE_METHODS),
        len(REVIEW_SELECTION_DELEGATE_METHODS),
        len(REVIEW_SELECTION_STAY_METHODS),
    ) == (15, 4, 7)
    assert REALTIME_METHODS.isdisjoint(REVIEW_SELECTION_METHODS)
    assert REALTIME_DELEGATE_METHODS.isdisjoint(REALTIME_STAY_METHODS)
    assert (
        REALTIME_MOVE_METHODS | REALTIME_DELEGATE_METHODS | REALTIME_STAY_METHODS
    ) == REALTIME_METHODS
    assert REVIEW_SELECTION_DELEGATE_METHODS.isdisjoint(REVIEW_SELECTION_STAY_METHODS)
    assert (
        REVIEW_SELECTION_MOVE_METHODS
        | REVIEW_SELECTION_DELEGATE_METHODS
        | REVIEW_SELECTION_STAY_METHODS
    ) == REVIEW_SELECTION_METHODS
    assert "_build_console_provider_selection_uncached" not in REVIEW_SELECTION_METHODS
    assert {
        "_console_review_notes_flow",
        "on_console_review_notes_requested",
    } <= REVIEW_SELECTION_STAY_METHODS
    assert REALTIME_INTERNAL_ONLY_METHODS <= REALTIME_MOVE_METHODS
    assert {
        "_console_realtime_marshal",
        "_enter_console_realtime_loop",
        "_release_console_realtime_state",
        "_tick_console_realtime",
    } <= REALTIME_MOVE_METHODS
    assert {
        "_on_console_change_review_dismissed",
        "_console_selection_feedback_flow",
        "_sync_console_annotation_discovery",
    } <= REVIEW_SELECTION_MOVE_METHODS
    _require_candidates(methods, REALTIME_METHODS)
    _require_candidates(methods, REVIEW_SELECTION_METHODS)
    assert all(name_counts[name] == 1 for name in REALTIME_METHODS)
    assert all(name_counts[name] == 1 for name in REVIEW_SELECTION_METHODS)
    assert all(
        not methods[name].decorator_list for name in REALTIME_INTERNAL_ONLY_METHODS
    )
    assert all(
        not _external_method_reference_paths(AMENDMENT_BASE, name)
        for name in REALTIME_INTERNAL_ONLY_METHODS
    )
    assert {
        name: tuple(ast.unparse(item) for item in methods[name].decorator_list)
        for name in REVIEW_SELECTION_ON_DELEGATE_METHODS
    } == {
        "handle_console_changed_files_selected": (
            "on(ConsoleChangedFilesSection.FileSelected)",
        ),
        "on_console_selection_feedback_requested": (
            "on(ConsoleSelectionFeedbackRequested)",
        ),
        "on_console_selection_note_requested": ("on(ConsoleSelectionNoteRequested)",),
    }
    assert REVIEW_SELECTION_ACTION_DELEGATE_METHODS == {"action_open_trajectory_view"}
    assert "open_trajectory_view" in _binding_actions(owner)

    realtime_lines = sum(_span(methods[name]) for name in REALTIME_METHODS)
    review_lines = sum(_span(methods[name]) for name in REVIEW_SELECTION_METHODS)
    assert (realtime_lines, review_lines) == (1_997, 1_114)

    realtime_stay_lines = sum(_span(methods[name]) for name in REALTIME_STAY_METHODS)
    review_stay_lines = sum(
        _span(methods[name]) for name in REVIEW_SELECTION_STAY_METHODS
    )
    realtime_residue_lines = realtime_stay_lines + (
        len(REALTIME_DELEGATE_METHODS) * MAX_DELEGATE_LINES
    )
    review_residue_lines = review_stay_lines + (
        len(REVIEW_SELECTION_DELEGATE_METHODS) * MAX_DELEGATE_LINES
    )
    assert (realtime_stay_lines, review_stay_lines) == (19, 438)
    assert (realtime_residue_lines, review_residue_lines) == (19, 458)

    projected_lines = (
        realtime_lines + review_lines - realtime_residue_lines - review_residue_lines
    )
    projected_methods = len(REALTIME_MOVE_METHODS) + len(REVIEW_SELECTION_MOVE_METHODS)
    assert (projected_lines, projected_methods) == (2_634, 71)
    assert projected_lines > AMENDMENT_COUNTS[0] - IMMUTABLE_BUDGETS[0]
    assert projected_methods > AMENDMENT_COUNTS[1] - IMMUTABLE_BUDGETS[1]

    incomplete = dict(methods)
    incomplete.pop(next(iter(REALTIME_METHODS)))
    with pytest.raises(AssertionError, match="characterized methods missing"):
        _require_candidates(incomplete, REALTIME_METHODS)

"""Source-inspected ownership contract for Console review and selection policy."""

from __future__ import annotations

import ast
import subprocess
from collections import Counter
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CHAT_SCREEN_PATH = REPO_ROOT / "tldw_chatbook/UI/Screens/chat_screen.py"
REVIEW_PATH = REPO_ROOT / "tldw_chatbook/UI/Console_Modules/review_selection.py"
WIRING_PATH = REPO_ROOT / "tldw_chatbook/UI/Console_Modules/wiring.py"
RATCHET_PATH = REPO_ROOT / "Tests/Architecture/test_screen_size_ratchet.py"
SCREEN_RELATIVE_PATH = "tldw_chatbook/UI/Screens/chat_screen.py"
REVIEW_RELATIVE_PATH = "tldw_chatbook/UI/Console_Modules/review_selection.py"
TASK_BASE = "c6218918d1e70c1938f7e11df592d0c70ca60383"
GIT_TIMEOUT_SECONDS = 10
TASK_BASE_COUNTS = (17_624, 539)
FAMILY_LINES = 850
MOVE_LINES = 280
STAY_LINES = 426
MINIMUM_LINE_REDUCTION = 409
MINIMUM_METHOD_REDUCTION = 7
PROJECTED_LINE_CEILING = 17_215
PROJECTED_METHOD_CEILING = 532
MAX_DELEGATE_LINES = 5

MOVE_METHODS = frozenset(
    {
        "_console_change_review_provider",
        "_console_change_review_workspace_roots",
        "_console_selection_feedback_flow",
        "_create_console_selection_note",
        "_load_console_annotation_previews",
        "_record_console_feedback_event",
        "_sync_console_annotation_discovery",
    }
)
DELEGATE_METHODS = frozenset(
    {
        "action_open_trajectory_view",
        "on_console_selection_feedback_requested",
        "on_console_selection_note_requested",
    }
)
STAY_METHODS = frozenset(
    {
        "_console_change_review_run_id",
        "_console_review_notes_flow",
        "_console_selection_quote_requested",
        "_dismiss_console_selection_menus_outside_transcript",
        "_open_change_review",
        "on_console_review_notes_requested",
    }
)
FAMILY_METHODS = MOVE_METHODS | DELEGATE_METHODS | STAY_METHODS
STATE_SLOTS = {
    "_console_annotation_loaded_conversation": (
        "_review_selection",
        "annotation_loaded_conversation",
    ),
    "_console_annotation_previews": ("_review_selection", "annotation_previews"),
    "_console_selection_feedback_inflight": (
        "_review_selection",
        "selection_feedback_inflight",
    ),
}
CONTROLLER_DEPENDENCIES = frozenset(
    {
        "store_accessor",
        "agent_conversation_id_accessor",
        "change_review_provider_accessor",
        "run_active_accessor",
        "run_active_for_root",
        "workspace_roots_accessor",
        "agent_runs_db_accessor",
        "capture_policy_bindings_accessor",
        "native_messages_accessor",
        "run_worker",
        "show_feedback_comment",
        "dispatch_prompt",
        "marshal_to_ui",
        "present_trajectory",
        "notify",
    }
)

FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


def _tree(path: Path) -> ast.Module:
    """Parse a required source file."""
    assert path.is_file(), f"required production module is missing: {path}"
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _class_node(path: Path, class_name: str) -> ast.ClassDef:
    """Return one top-level class by exact name."""
    classes = [
        node
        for node in _tree(path).body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    assert len(classes) == 1
    return classes[0]


def _class_from_source(source: str, class_name: str) -> ast.ClassDef:
    """Return one top-level class from source text."""
    classes = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    assert len(classes) == 1
    return classes[0]


def _method_nodes(owner: ast.ClassDef) -> list[FunctionNode]:
    """Return every direct method, retaining duplicates."""
    return [
        node
        for node in owner.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _methods(owner: ast.ClassDef) -> dict[str, FunctionNode]:
    """Return direct methods by name."""
    return {node.name: node for node in _method_nodes(owner)}


def _method_counts(owner: ast.ClassDef) -> Counter[str]:
    """Count every direct method definition."""
    return Counter(node.name for node in _method_nodes(owner))


def _span(node: FunctionNode) -> int:
    """Return physical definition span without decorators."""
    return node.end_lineno - node.lineno + 1


def _counts(source: str) -> tuple[int, int]:
    """Return physical source lines and unique direct ChatScreen method names."""
    owner = _class_from_source(source, "ChatScreen")
    return len(source.splitlines()), len(_methods(owner))


def _source_at_revision(revision: str, relative_path: str) -> str:
    """Read a repository file at one revision with a bounded subprocess."""
    return subprocess.run(
        ["git", "show", f"{revision}:{relative_path}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT_SECONDS,
    ).stdout


def _optional_source_at_revision(revision: str, relative_path: str) -> str | None:
    """Read an optional revision path, distinguishing absence from Git failure."""
    try:
        return _source_at_revision(revision, relative_path)
    except subprocess.CalledProcessError as error:
        if (
            "does not exist" in error.stderr
            or "exists on disk, but not in" in error.stderr
        ):
            return None
        raise


def _call_name(call: ast.Call) -> str | None:
    """Return a call's terminal name."""
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _class_bindings(
    owner: ast.ClassDef, names: set[str]
) -> dict[str, list[ast.expr | None]]:
    """Return class-body assignment values for exact names."""
    found = {name: [] for name in names}
    for node in owner.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id in found:
                found[target.id].append(value)
    return found


def _state_bindings(owner: ast.ClassDef) -> dict[str, tuple[str, str]]:
    """Return exact compatibility descriptor targets."""
    bindings = _class_bindings(owner, set(STATE_SLOTS))
    assert all(len(values) == 1 for values in bindings.values())
    result: dict[str, tuple[str, str]] = {}
    for name, (value,) in bindings.items():
        assert isinstance(value, ast.Call)
        assert _call_name(value) == "_ControllerState"
        assert len(value.args) == 2
        assert all(isinstance(argument, ast.Constant) for argument in value.args)
        result[name] = tuple(argument.value for argument in value.args)  # type: ignore[misc]
    return result


def _self_writes(owner: ast.ClassDef, names: set[str]) -> set[str]:
    """Return named self attributes assigned by direct screen methods."""
    found: set[str] = set()
    for method in _method_nodes(owner):
        for node in ast.walk(method):
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                targets = [node.target]
            else:
                continue
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and target.attr in names
                ):
                    found.add(target.attr)
    return found


def _descriptor_type() -> type:
    """Execute only the shared descriptor class for runtime forwarding tests."""
    screen_tree = _tree(CHAT_SCREEN_PATH)
    descriptor = next(
        node
        for node in screen_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_ControllerState"
    )
    namespace: dict[str, object] = {"__builtins__": __builtins__}
    module = ast.Module(body=[descriptor], type_ignores=[])
    exec(
        compile(ast.fix_missing_locations(module), str(CHAT_SCREEN_PATH), "exec"),
        namespace,
    )
    descriptor_type = namespace["_ControllerState"]
    assert isinstance(descriptor_type, type)
    return descriptor_type


def _assert_descriptor_runtime() -> None:
    """Prove unwired access is loud and wired access has no shadow storage."""
    descriptor_type = _descriptor_type()

    class Host:
        pass

    for name, (owner_name, state_name) in STATE_SLOTS.items():
        setattr(Host, name, descriptor_type(owner_name, state_name))
    host = Host()
    for name in STATE_SLOTS:
        with pytest.raises(RuntimeError, match="controller not wired"):
            getattr(host, name)
        with pytest.raises(RuntimeError, match="controller not wired"):
            setattr(host, name, object())
        assert name not in vars(host)

    class Review:
        annotation_loaded_conversation = None
        annotation_previews: dict[str, tuple[str, ...]] = {}
        selection_feedback_inflight = False

    review = Review()
    host._review_selection = review
    preview_map = {"m1": ("note",)}
    host._console_annotation_loaded_conversation = "conversation"
    host._console_annotation_previews = preview_map
    host._console_selection_feedback_inflight = True
    assert review.annotation_loaded_conversation == "conversation"
    assert review.annotation_previews is preview_map
    assert review.selection_feedback_inflight is True
    assert not (set(vars(host)) & set(STATE_SLOTS))


def _binding_actions(owner: ast.ClassDef) -> set[str]:
    """Return Textual action names declared in direct BINDINGS assignments."""
    return {
        call.args[1].value
        for assignment in owner.body
        if isinstance(assignment, (ast.Assign, ast.AnnAssign))
        if assignment.value is not None
        for call in ast.walk(assignment.value)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "Binding"
        and len(call.args) >= 2
        and isinstance(call.args[1], ast.Constant)
        and isinstance(call.args[1].value, str)
    }


def _review_calls(method: FunctionNode) -> list[ast.Call]:
    """Return calls rooted directly at self._review_selection."""
    calls: list[ast.Call] = []
    for node in ast.walk(method):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        root = node.func.value
        if (
            isinstance(root, ast.Attribute)
            and isinstance(root.value, ast.Name)
            and root.value.id == "self"
            and root.attr == "_review_selection"
        ):
            calls.append(node)
    return calls


def _assert_final_ownership(screen_source: str, review_source: str) -> None:
    """Assert final method ownership in an arbitrary complete revision."""
    screen = _class_from_source(screen_source, "ChatScreen")
    controller = _class_from_source(review_source, "ConsoleReviewSelectionController")
    screen_counts = _method_counts(screen)
    controller_counts = _method_counts(controller)
    assert all(controller_counts[name] == 1 for name in MOVE_METHODS)
    assert all(screen_counts[name] == 0 for name in MOVE_METHODS)
    assert all(screen_counts[name] == 1 for name in STAY_METHODS | DELEGATE_METHODS)
    assert all(controller_counts[name] == 0 for name in STAY_METHODS | DELEGATE_METHODS)


@pytest.mark.unit
def test_review_selection_controller_path_and_exact_method_ownership() -> None:
    """The dedicated controller owns seven methods and the six stays do not move."""
    controller = _class_node(REVIEW_PATH, "ConsoleReviewSelectionController")
    screen = _class_node(CHAT_SCREEN_PATH, "ChatScreen")
    screen_counts = _method_counts(screen)
    controller_counts = _method_counts(controller)

    assert all(controller_counts[name] == 1 for name in MOVE_METHODS)
    assert all(screen_counts[name] == 0 for name in MOVE_METHODS)
    assert all(screen_counts[name] == 1 for name in STAY_METHODS)
    assert all(controller_counts[name] == 0 for name in STAY_METHODS)
    assert all(screen_counts[name] == 1 for name in DELEGATE_METHODS)
    assert all(controller_counts[name] == 0 for name in DELEGATE_METHODS)


@pytest.mark.unit
def test_review_selection_delegates_are_complete_and_bounded() -> None:
    """Framework bindings stop their event when needed and perform one handoff."""
    screen = _class_node(CHAT_SCREEN_PATH, "ChatScreen")
    methods = _methods(screen)
    assert "open_trajectory_view" in _binding_actions(screen)
    assert {
        name: tuple(ast.unparse(item) for item in methods[name].decorator_list)
        for name in DELEGATE_METHODS - {"action_open_trajectory_view"}
    } == {
        "on_console_selection_feedback_requested": (
            "on(ConsoleSelectionFeedbackRequested)",
        ),
        "on_console_selection_note_requested": ("on(ConsoleSelectionNoteRequested)",),
    }
    for name in DELEGATE_METHODS:
        method = methods[name]
        assert _span(method) <= MAX_DELEGATE_LINES, (name, _span(method))
        assert len(_review_calls(method)) == 1
        assert not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"run_worker", "to_thread", "dispatch"}
            for node in ast.walk(method)
        )


@pytest.mark.unit
def test_trajectory_adapter_moves_without_eager_screen_import() -> None:
    """Trajectory service reads move with the owner while presentation stays lazy."""
    screen_tree = _tree(CHAT_SCREEN_PATH)
    review_tree = _tree(REVIEW_PATH)
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_build_trajectory_snapshot"
        for node in screen_tree.body
    )
    helpers = [
        node
        for node in review_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_build_trajectory_snapshot"
    ]
    assert len(helpers) == 1
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        and "trajectory_screen" in ast.unparse(node)
        for node in review_tree.body
    )


@pytest.mark.unit
def test_review_selection_module_has_no_dom_sibling_or_authority_bypass() -> None:
    """The owner cannot recover ambient screen, sibling, Git, SQL, or DOM authority."""
    tree = _tree(REVIEW_PATH)
    controller = _class_node(REVIEW_PATH, "ConsoleReviewSelectionController")
    assert not controller.bases
    source = REVIEW_PATH.read_text(encoding="utf-8")
    assert "ChatScreen" not in source
    assert "trajectory_screen" not in source
    assert "subprocess" not in source
    assert ".execute(" not in source
    assert not any(
        isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith("textual")
        for node in tree.body
    )
    forbidden = {
        "__getattr__",
        "__getattribute__",
        "query",
        "query_one",
        "focus",
        "push_screen",
        "_agent",
        "_prompt_queue",
        "_session",
        "_console_chat_controller",
    }
    assert not {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in forbidden
    }


@pytest.mark.unit
def test_review_selection_state_uses_fail_loud_descriptors_without_shadowing() -> None:
    """Compatibility state is read/write, fail-loud, and owned only once."""
    screen = _class_node(CHAT_SCREEN_PATH, "ChatScreen")
    assert _state_bindings(screen) == STATE_SLOTS
    assert not _self_writes(screen, set(STATE_SLOTS))
    _assert_descriptor_runtime()


@pytest.mark.unit
def test_build_console_controllers_is_the_single_explicit_constructor() -> None:
    """Wiring constructs one controller from exact keyword-only callables."""
    wiring_tree = _tree(WIRING_PATH)
    builder = next(
        node
        for node in wiring_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_console_controllers"
    )
    assignments = [
        node
        for node in ast.walk(builder)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "screen"
            and target.attr == "_review_selection"
            for target in node.targets
        )
    ]
    assert len(assignments) == 1
    constructor = assignments[0].value
    assert isinstance(constructor, ast.Call)
    assert _call_name(constructor) == "ConsoleReviewSelectionController"
    assert not constructor.args
    assert all(keyword.arg is not None for keyword in constructor.keywords)
    assert {keyword.arg for keyword in constructor.keywords} == CONTROLLER_DEPENDENCIES

    controller = _class_node(REVIEW_PATH, "ConsoleReviewSelectionController")
    initializer = _methods(controller)["__init__"]
    assert [argument.arg for argument in initializer.args.args] == ["self"]
    assert initializer.args.vararg is None
    assert initializer.args.kwarg is None
    dependencies = {argument.arg: argument for argument in initializer.args.kwonlyargs}
    assert set(dependencies) == CONTROLLER_DEPENDENCIES
    assert not any("controller" in name or "screen" in name for name in dependencies)
    assert all(
        argument.annotation is not None
        and "Callable" in ast.unparse(argument.annotation)
        for argument in dependencies.values()
    )

    constructors: list[Path] = []
    for path in sorted((REPO_ROOT / "tldw_chatbook").rglob("*.py")):
        if "ConsoleReviewSelectionController" not in path.read_text(encoding="utf-8"):
            continue
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.Call) and _call_name(node) == (
                "ConsoleReviewSelectionController"
            ):
                constructors.append(path)
    assert constructors == [WIRING_PATH]


@pytest.mark.unit
def test_chat_screen_remains_within_reviewed_projection_without_ratchet_raise() -> None:
    """The extraction earns the conservative screen reduction itself."""
    source = CHAT_SCREEN_PATH.read_text(encoding="utf-8")
    assert _counts(source) <= (PROJECTED_LINE_CEILING, PROJECTED_METHOD_CEILING)
    ratchet_tree = ast.parse(RATCHET_PATH.read_text(encoding="utf-8"))
    assignment = next(
        node
        for node in ratchet_tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "_BUDGETS"
    )
    assert assignment.value is not None
    budgets = ast.literal_eval(assignment.value)
    assert budgets[SCREEN_RELATIVE_PATH] == ("ChatScreen", 17_727, 593)


@pytest.mark.unit
def test_frozen_task_base_family_and_projection_match_review() -> None:
    """The written 7/3/6 arithmetic remains reproducible at the exact base."""
    source = _source_at_revision(TASK_BASE, SCREEN_RELATIVE_PATH)
    owner = _class_from_source(source, "ChatScreen")
    methods = _methods(owner)
    assert _counts(source) == TASK_BASE_COUNTS
    assert all(_method_counts(owner)[name] == 1 for name in FAMILY_METHODS)
    assert sum(_span(methods[name]) for name in FAMILY_METHODS) == FAMILY_LINES
    assert sum(_span(methods[name]) for name in MOVE_METHODS) == MOVE_LINES
    assert sum(_span(methods[name]) for name in STAY_METHODS) == STAY_LINES
    residue = STAY_LINES + len(DELEGATE_METHODS) * MAX_DELEGATE_LINES
    assert FAMILY_LINES - residue == MINIMUM_LINE_REDUCTION
    assert len(MOVE_METHODS) == MINIMUM_METHOD_REDUCTION
    assert (
        TASK_BASE_COUNTS[0] - MINIMUM_LINE_REDUCTION,
        TASK_BASE_COUNTS[1] - MINIMUM_METHOD_REDUCTION,
    ) == (PROJECTED_LINE_CEILING, PROJECTED_METHOD_CEILING)


@pytest.mark.unit
def test_origin_dev_review_selection_family_still_matches_review() -> None:
    """Latest dev is a complete reviewed pre- or post-extraction state."""
    screen_source = _source_at_revision("origin/dev", SCREEN_RELATIVE_PATH)
    review_source = _optional_source_at_revision("origin/dev", REVIEW_RELATIVE_PATH)
    screen = _class_from_source(screen_source, "ChatScreen")
    counts = _method_counts(screen)
    if all(counts[name] == 1 for name in MOVE_METHODS):
        methods = _methods(screen)
        assert all(counts[name] == 1 for name in FAMILY_METHODS)
        assert sum(_span(methods[name]) for name in FAMILY_METHODS) == FAMILY_LINES
        assert sum(_span(methods[name]) for name in MOVE_METHODS) == MOVE_LINES
        assert sum(_span(methods[name]) for name in STAY_METHODS) == STAY_LINES
        projected = (
            _counts(screen_source)[0] - MINIMUM_LINE_REDUCTION,
            _counts(screen_source)[1] - MINIMUM_METHOD_REDUCTION,
        )
        assert projected <= (PROJECTED_LINE_CEILING, PROJECTED_METHOD_CEILING)
        return
    assert review_source is not None
    _assert_final_ownership(screen_source, review_source)

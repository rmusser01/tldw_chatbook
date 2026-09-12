"""Source-inspected ownership contract for the Console realtime controller."""

from __future__ import annotations

import ast
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from Tests.Architecture.test_console_wave6_closeout_inventory import (
    REALTIME_DELEGATE_METHODS,
    REALTIME_METHODS,
    REALTIME_MOVE_METHODS,
    REALTIME_STAY_METHODS,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CHAT_SCREEN_PATH = REPO_ROOT / "tldw_chatbook/UI/Screens/chat_screen.py"
REALTIME_PATH = REPO_ROOT / "tldw_chatbook/UI/Console_Modules/realtime.py"
WIRING_PATH = REPO_ROOT / "tldw_chatbook/UI/Console_Modules/wiring.py"
SCREEN_RELATIVE_PATH = "tldw_chatbook/UI/Screens/chat_screen.py"
GIT_TIMEOUT_SECONDS = 10
EXTRACTION_LINES = 1_978
EXTRACTION_METHODS = 56
PROJECTED_LINE_CEILING = 18_076
PROJECTED_METHOD_CEILING = 577
# Delivery rebase 794ae11521 added 56 unrelated ChatScreen lines after the
# reviewed base. This allowance applies only to the still-pre-extraction base;
# the implemented screen remains held to PROJECTED_LINE_CEILING below.
REVIEWED_DELIVERY_BASE_LINE_DRIFT = 56
REALTIME_CONTROLLER_DEPENDENCIES = frozenset(
    {
        "ensure_session_settings",
        "chat_store_accessor",
        "runtime_accessor",
        "dictation_state_accessor",
        "request_dictation_stop",
        "pipeline_blocker",
        "enter_pipeline_loop",
        "recorder_factory_accessor",
        "provider_session_factory_accessor",
        "sink_factory_accessor",
        "notify",
        "ui_thread_id_accessor",
        "event_loop_accessor",
        "set_interval",
        "run_worker",
        "defer_native_sync",
        "repaint_chip",
        "restore_voice_chip",
    }
)
REALTIME_CONTROLLER_LAMBDA_TEMPLATES = {
    "ensure_session_settings": "lambda: screen._session._ensure_active_console_session_settings()",
    "chat_store_accessor": "lambda: screen._ensure_console_chat_store()",
    "runtime_accessor": "lambda: screen._console_runtime()",
    "dictation_state_accessor": "lambda: screen._console_dictation_state",
    "request_dictation_stop": "lambda: screen._dictation._request_console_dictation_stop()",
    "pipeline_blocker": "lambda: screen._hands_free._console_pipeline_hands_free_blocker()",
    "enter_pipeline_loop": "lambda capture_live: screen._hands_free._enter_console_hands_free_pipeline_loop(capture_live=capture_live)",
    "recorder_factory_accessor": 'lambda: getattr(screen.app_instance, "console_realtime_recorder_factory", None)',
    "provider_session_factory_accessor": 'lambda: getattr(screen.app_instance, "console_realtime_session_factory", None)',
    "sink_factory_accessor": 'lambda: getattr(screen.app_instance, "console_realtime_sink_factory", None)',
    "notify": "lambda *args, **kwargs: screen.app_instance.notify(*args, **kwargs)",
    "ui_thread_id_accessor": "lambda: screen.app_instance._thread_id",
    "event_loop_accessor": 'lambda: getattr(screen.app_instance, "_loop", None)',
    "set_interval": "lambda *args, **kwargs: screen.set_interval(*args, **kwargs)",
    "run_worker": 'lambda *args, **kwargs: screen.run_worker(*args, group=kwargs.pop("group"), **kwargs)',
    "defer_native_sync": "lambda: screen.call_later(screen._sync_native_console_chat_ui)",
    "repaint_chip": "lambda: screen._repaint_console_realtime_chip()",
    "restore_voice_chip": "lambda: screen._restore_console_voice_chip()",
}


FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


def _tree(path: Path) -> ast.Module:
    """Parse a production file, reporting a missing path as a test failure."""
    assert path.is_file(), f"required production module is missing: {path}"
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _class_node(path: Path, class_name: str) -> ast.ClassDef:
    """Return a top-level class by exact name."""
    tree = _tree(path)
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    assert len(classes) == 1, (
        f"{path.relative_to(REPO_ROOT)} must define one direct {class_name} class"
    )
    return classes[0]


def _direct_methods(owner: ast.ClassDef) -> dict[str, FunctionNode]:
    """Return direct methods, excluding nested functions and classes."""
    return {node.name: node for node in _direct_method_nodes(owner)}


def _direct_method_nodes(owner: ast.ClassDef) -> list[FunctionNode]:
    """Return every direct method node, preserving duplicate definitions."""
    return [
        node
        for node in owner.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _direct_method_counts(owner: ast.ClassDef) -> Counter[str]:
    """Count every direct method node, including duplicate definitions."""
    return Counter(node.name for node in _direct_method_nodes(owner))


def _definition_span(node: FunctionNode) -> int:
    """Return a method's physical span, excluding decorator lines."""
    return node.end_lineno - node.lineno + 1


def _self_writes(method: FunctionNode, names: set[str]) -> set[str]:
    """Find direct writes to named ``self`` attributes in a method."""
    writes: set[str] = set()
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
                writes.add(target.attr)
    return writes


def _class_body_bindings(
    owner: ast.ClassDef, names: set[str]
) -> dict[str, list[ast.expr | None]]:
    """Return every class-body assignment value for each named binding."""
    bindings = {name: [] for name in names}
    for node in owner.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id in names:
                bindings[target.id].append(value)
    return bindings


def _module_level_chat_screen_facades(tree: ast.Module, names: set[str]) -> set[str]:
    """Find module-level ChatScreen aliases and setattr facades."""
    found: set[str] = set()

    def targets(node: ast.Assign | ast.AnnAssign | ast.AugAssign) -> list[ast.expr]:
        if isinstance(node, ast.Assign):
            return node.targets
        return [node.target]

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            return

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return

        def visit_Assign(self, node: ast.Assign) -> None:
            self._check_targets(node)
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            self._check_targets(node)
            self.generic_visit(node)

        def visit_AugAssign(self, node: ast.AugAssign) -> None:
            self._check_targets(node)
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            if (
                _call_terminal_name(node) == "setattr"
                and node.args
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "ChatScreen"
            ):
                found.add("setattr(ChatScreen)")
            self.generic_visit(node)

        def _check_targets(
            self, node: ast.Assign | ast.AnnAssign | ast.AugAssign
        ) -> None:
            for target in targets(node):
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "ChatScreen"
                    and target.attr in names
                ):
                    found.add(target.attr)

    Visitor().visit(tree)
    return found


def _controller_state_slot_values(
    owner: ast.ClassDef, names: set[str]
) -> dict[str, tuple[str, str]]:
    """Return exact ``_ControllerState`` values after binding-count checks."""
    bindings = _class_body_bindings(owner, names)
    assert all(len(values) == 1 for values in bindings.values()), (
        "realtime compatibility descriptors must each have exactly one class binding: "
        f"{ {name: len(values) for name, values in bindings.items()} }"
    )
    slots: dict[str, tuple[str, str]] = {}
    for name, values in bindings.items():
        value = values[0]
        assert (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "_ControllerState"
            and len(value.args) == 2
            and all(isinstance(argument, ast.Constant) for argument in value.args)
        ), f"{name} must bind directly to _ControllerState(...)"
        slots[name] = tuple(argument.value for argument in value.args)  # type: ignore[misc]
    return slots


def _controller_state_class() -> ast.ClassDef:
    """Return the shared descriptor class without importing production modules."""
    tree = _tree(CHAT_SCREEN_PATH)
    descriptor = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "_ControllerState"
        ),
        None,
    )
    assert descriptor is not None, "ChatScreen must define _ControllerState"
    return descriptor


def _controller_state_descriptor_type() -> type:
    """Execute only ``_ControllerState`` in an isolated namespace."""
    module = ast.Module(body=[_controller_state_class()], type_ignores=[])
    namespace: dict[str, object] = {"__builtins__": __builtins__}
    exec(
        compile(ast.fix_missing_locations(module), str(CHAT_SCREEN_PATH), "exec"),
        namespace,
    )
    descriptor_type = namespace.get("_ControllerState")
    assert isinstance(descriptor_type, type)
    return descriptor_type


def _assert_controller_state_descriptor_runtime() -> None:
    """Prove unwired descriptors fail loudly and wired descriptors forward state."""
    descriptor_type = _controller_state_descriptor_type()

    class Host:
        pass

    setattr(Host, "_console_realtime", descriptor_type("_realtime", "session"))
    setattr(
        Host,
        "_console_realtime_close_worker",
        descriptor_type("_realtime", "close_worker"),
    )
    host = Host()
    for name in ("_console_realtime", "_console_realtime_close_worker"):
        with pytest.raises(RuntimeError, match="controller not wired"):
            getattr(host, name)
        with pytest.raises(RuntimeError, match="controller not wired"):
            setattr(host, name, object())
    assert "_console_realtime" not in vars(host)
    assert "_console_realtime_close_worker" not in vars(host)

    class Realtime:
        pass

    realtime = Realtime()
    realtime.session = object()
    realtime.close_worker = object()
    host._realtime = realtime
    assert host._console_realtime is realtime.session
    assert host._console_realtime_close_worker is realtime.close_worker

    replacement_session = object()
    replacement_worker = object()
    host._console_realtime = replacement_session
    host._console_realtime_close_worker = replacement_worker
    assert realtime.session is replacement_session
    assert realtime.close_worker is replacement_worker
    assert "_console_realtime" not in vars(host)
    assert "_console_realtime_close_worker" not in vars(host)


def _call_terminal_name(call: ast.Call) -> str | None:
    """Return a called function's terminal name for AST boundary checks."""
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _forbidden_realtime_module_references(tree: ast.Module) -> set[str]:
    """Collect forbidden DOM, screen, and sibling reachbacks from the whole module."""
    siblings = {"_dictation", "_hands_free", "_session"}
    dom_names = {
        "DOM",
        "children",
        "compose",
        "focus",
        "get_screen",
        "mount",
        "push_screen",
    }
    found: set[str] = set()
    for node in ast.walk(tree):
        identifier = None
        if isinstance(node, ast.Name):
            identifier = node.id
        elif isinstance(node, ast.Attribute):
            identifier = node.attr
        if identifier is None:
            continue
        lowered = identifier.lower()
        if (
            identifier == "ChatScreen"
            or identifier in siblings
            or identifier in dom_names
            or "screen" in lowered
            or any(token in lowered for token in ("modal", "query"))
        ):
            found.add(identifier)
    return found


def _source_at_revision(revision: str, relative_path: str) -> str:
    """Read an arbitrary repository path at a revision with a bounded subprocess."""
    return subprocess.run(
        ["git", "show", f"{revision}:{relative_path}"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=GIT_TIMEOUT_SECONDS,
    ).stdout


def _optional_source_at_revision(revision: str, relative_path: str) -> str | None:
    """Read a revision path, returning ``None`` only when that path is absent."""
    try:
        return _source_at_revision(revision, relative_path)
    except subprocess.CalledProcessError as error:
        if (
            "does not exist" in error.stderr
            or "exists on disk, but not in" in error.stderr
        ):
            return None
        raise


def _class_from_source(source: str, class_name: str) -> ast.ClassDef:
    """Return a named top-level class from source text."""
    classes = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    assert len(classes) == 1, f"source must define one direct {class_name} class"
    return classes[0]


def _assert_final_origin_dev_ownership(
    screen_source: str, realtime_source: str
) -> None:
    """Assert the exact extracted 56/0/1 ownership when origin/dev has it."""
    screen = _class_from_source(screen_source, "ChatScreen")
    realtime_tree = ast.parse(realtime_source)
    controllers = [
        node
        for node in realtime_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ConsoleRealtimeController"
    ]
    assert len(controllers) == 1
    controller = controllers[0]
    screen_counts = _direct_method_counts(screen)
    controller_counts = _direct_method_counts(controller)
    assert all(controller_counts[name] == 1 for name in REALTIME_MOVE_METHODS)
    assert all(screen_counts[name] == 0 for name in REALTIME_MOVE_METHODS)
    assert all(screen_counts[name] == 1 for name in REALTIME_STAY_METHODS)
    assert all(controller_counts[name] == 0 for name in REALTIME_STAY_METHODS)
    assert all(screen_counts[name] == 0 for name in REALTIME_DELEGATE_METHODS)
    assert all(controller_counts[name] == 0 for name in REALTIME_DELEGATE_METHODS)


@pytest.mark.unit
def test_realtime_controller_path_and_exact_method_ownership() -> None:
    """The dedicated controller owns exactly the reviewed move set."""
    controller = _class_node(REALTIME_PATH, "ConsoleRealtimeController")
    screen = _class_node(CHAT_SCREEN_PATH, "ChatScreen")
    screen_counts = _direct_method_counts(screen)
    controller_counts = _direct_method_counts(controller)

    assert all(controller_counts[name] == 1 for name in REALTIME_MOVE_METHODS), (
        "controller is missing reviewed realtime methods: "
        f"{sorted(name for name in REALTIME_MOVE_METHODS if controller_counts[name] != 1)}"
    )
    assert all(screen_counts[name] == 0 for name in REALTIME_MOVE_METHODS), (
        "moved realtime methods remain directly on ChatScreen: "
        f"{sorted(name for name in REALTIME_MOVE_METHODS if screen_counts[name])}"
    )
    assert REALTIME_STAY_METHODS == {"_repaint_console_realtime_chip"}
    assert all(screen_counts[name] == 1 for name in REALTIME_STAY_METHODS)
    assert all(controller_counts[name] == 0 for name in REALTIME_STAY_METHODS)
    assert all(screen_counts[name] == 0 for name in REALTIME_DELEGATE_METHODS)
    assert all(controller_counts[name] == 0 for name in REALTIME_DELEGATE_METHODS)


@pytest.mark.unit
def test_chat_screen_has_no_realtime_delegate_or_dynamic_facade() -> None:
    """ChatScreen exposes no same-name delegate, facade, or realtime mixin."""
    screen_tree = _tree(CHAT_SCREEN_PATH)
    screen = _class_node(CHAT_SCREEN_PATH, "ChatScreen")
    methods = _direct_methods(screen)
    counts = _direct_method_counts(screen)
    forbidden_bindings = set(REALTIME_MOVE_METHODS) | {
        "__getattr__",
        "__getattribute__",
    }

    assert all(
        not values
        for values in _class_body_bindings(screen, forbidden_bindings).values()
    ), (
        "same-name callable realtime delegates remain on ChatScreen: "
        f"{sorted(name for name, values in _class_body_bindings(screen, forbidden_bindings).items() if values)}"
    )
    assert all(counts[name] == 0 for name in REALTIME_MOVE_METHODS)
    assert all(counts[name] == 0 for name in REALTIME_DELEGATE_METHODS)
    assert all(counts[name] == 1 for name in REALTIME_STAY_METHODS)
    assert "__getattr__" not in methods
    assert "__getattribute__" not in methods
    assert not _module_level_chat_screen_facades(screen_tree, forbidden_bindings)
    assert not any(
        any(token in ast.unparse(base).lower() for token in ("realtime", "mixin"))
        for base in screen.bases
    ), "ChatScreen must not gain a realtime mixin base"


@pytest.mark.unit
def test_realtime_controller_has_no_dom_or_sibling_controller_boundary_bypass() -> None:
    """Realtime orchestration is non-DOM and cannot reach sibling controllers."""
    realtime_tree = _tree(REALTIME_PATH)
    controller = _class_node(REALTIME_PATH, "ConsoleRealtimeController")

    assert not controller.bases, "realtime ownership must not be supplied by a mixin"
    assert not _forbidden_realtime_module_references(realtime_tree)


@pytest.mark.unit
def test_realtime_state_uses_fail_loud_controller_descriptors_without_shadowing() -> (
    None
):
    """Compatibility attributes forward to controller state and never shadow it."""
    screen = _class_node(CHAT_SCREEN_PATH, "ChatScreen")
    methods = _direct_methods(screen)
    names = {"_console_realtime", "_console_realtime_close_worker"}

    assert _controller_state_slot_values(screen, names) == {
        "_console_realtime": ("_realtime", "session"),
        "_console_realtime_close_worker": ("_realtime", "close_worker"),
    }
    assert not any(_self_writes(method, names) for method in methods.values())
    _assert_controller_state_descriptor_runtime()


@pytest.mark.unit
def test_build_console_controllers_owns_the_single_realtime_construction() -> None:
    """The existing controller builder performs the sole realtime construction."""
    wiring_tree = _tree(WIRING_PATH)
    builder = next(
        node
        for node in wiring_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_console_controllers"
    )

    realtime_assignments = [
        node
        for node in ast.walk(builder)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "screen"
            and target.attr == "_realtime"
            for target in node.targets
        )
    ]
    assert len(realtime_assignments) == 1
    assignment = realtime_assignments[0]
    assert isinstance(assignment.value, ast.Call)
    assert _call_terminal_name(assignment.value) == "ConsoleRealtimeController"
    constructor = assignment.value
    assert not constructor.args, "realtime controller dependencies must be keyword-only"

    controller = _class_node(REALTIME_PATH, "ConsoleRealtimeController")
    initializer = next(
        (node for node in _direct_method_nodes(controller) if node.name == "__init__"),
        None,
    )
    assert initializer is not None, "controller must define an explicit __init__"
    assert not initializer.args.posonlyargs
    assert [argument.arg for argument in initializer.args.args] == ["self"]
    assert initializer.args.vararg is None
    assert initializer.args.kwarg is None
    assert initializer.args.kwonlyargs, (
        "controller dependencies must be explicit keyword-only parameters"
    )
    dependencies = {argument.arg: argument for argument in initializer.args.kwonlyargs}
    assert set(dependencies) == REALTIME_CONTROLLER_DEPENDENCIES
    assert not any("screen" in name.lower() for name in dependencies), (
        "controller dependencies must not be named after an ambient screen"
    )
    for name, argument in dependencies.items():
        annotation = argument.annotation
        assert annotation is not None and "Callable" in ast.unparse(annotation), (
            f"controller dependency {name} must have a Callable annotation"
        )

    keyword_names = [keyword.arg for keyword in constructor.keywords]
    assert None not in keyword_names, "controller wiring must use named dependencies"
    assert len(keyword_names) == len(dependencies)
    assert set(keyword_names) == set(dependencies)
    for keyword in constructor.keywords:
        assert keyword.arg is not None
        assert isinstance(keyword.value, ast.Lambda), (
            f"controller dependency {keyword.arg} must be wired by lambda"
        )
        expected = ast.parse(
            REALTIME_CONTROLLER_LAMBDA_TEMPLATES[keyword.arg], mode="eval"
        ).body
        assert isinstance(expected, ast.Lambda)
        assert ast.dump(keyword.value, include_attributes=False) == ast.dump(
            expected, include_attributes=False
        ), f"unexpected lambda template for {keyword.arg}"

    constructors: list[tuple[Path, str]] = []
    for path in sorted((REPO_ROOT / "tldw_chatbook").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "ConsoleRealtimeController" not in source:
            continue
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _call_terminal_name(node) == "ConsoleRealtimeController":
                function = next(
                    (
                        parent
                        for parent in ast.walk(tree)
                        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and any(child is node for child in ast.walk(parent))
                    ),
                    None,
                )
                constructors.append((path, function.name if function else "<module>"))
    assert constructors == [(WIRING_PATH, "build_console_controllers")]


@pytest.mark.unit
def test_chat_screen_remains_within_the_reviewed_projection_ceiling() -> None:
    """The extraction leaves the screen at or below the reviewed size budget."""
    source = CHAT_SCREEN_PATH.read_text(encoding="utf-8")
    screen = _class_node(CHAT_SCREEN_PATH, "ChatScreen")
    assert len(source.splitlines()) <= PROJECTED_LINE_CEILING
    assert len(_direct_method_nodes(screen)) <= PROJECTED_METHOD_CEILING


@pytest.mark.unit
def test_origin_dev_realtime_family_and_projection_still_match_review() -> None:
    """The live origin/dev base is either exact pre- or post-extraction state."""
    source = _source_at_revision("origin/dev", SCREEN_RELATIVE_PATH)
    realtime_source = _optional_source_at_revision(
        "origin/dev", "tldw_chatbook/UI/Console_Modules/realtime.py"
    )
    if realtime_source is not None:
        _assert_final_origin_dev_ownership(source, realtime_source)
        return

    screen = _class_from_source(source, "ChatScreen")
    method_nodes = _direct_method_nodes(screen)
    methods = _direct_methods(screen)
    realtime_methods = {name for name in methods if "console_realtime" in name}

    assert realtime_methods == REALTIME_METHODS
    assert (
        len(REALTIME_MOVE_METHODS),
        len(REALTIME_DELEGATE_METHODS),
        len(REALTIME_STAY_METHODS),
    ) == (56, 0, 1)
    assert sum(_definition_span(methods[name]) for name in REALTIME_METHODS) == 1_997
    assert sum(_definition_span(methods[name]) for name in REALTIME_STAY_METHODS) == 19

    projected = (
        len(source.splitlines()) - EXTRACTION_LINES,
        len(method_nodes) - EXTRACTION_METHODS,
    )
    assert (
        projected[0] <= PROJECTED_LINE_CEILING + REVIEWED_DELIVERY_BASE_LINE_DRIFT
        and projected[1] <= PROJECTED_METHOD_CEILING
    ), (
        "origin/dev projection drifted beyond the reviewed delivery amendment: "
        f"{projected} > "
        f"{(PROJECTED_LINE_CEILING + REVIEWED_DELIVERY_BASE_LINE_DRIFT, PROJECTED_METHOD_CEILING)}"
    )
    current_source = CHAT_SCREEN_PATH.read_text(encoding="utf-8")
    current_screen = _class_node(CHAT_SCREEN_PATH, "ChatScreen")
    assert len(current_source.splitlines()) <= PROJECTED_LINE_CEILING
    assert len(_direct_method_nodes(current_screen)) <= PROJECTED_METHOD_CEILING

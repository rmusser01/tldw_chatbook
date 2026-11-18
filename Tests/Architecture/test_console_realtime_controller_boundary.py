"""Source-inspected ownership contract for the Console realtime controller."""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import pytest

from Tests.Architecture.test_screen_size_ratchet import _BUDGETS as _SCREEN_SIZE_BUDGETS


#: The reviewed realtime family. TASK-39055 moved these sets here from the
#: wave-6 closeout inventory, which this file was the last consumer of.
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

REALTIME_DELEGATE_METHODS: frozenset[str] = frozenset()
REALTIME_STAY_METHODS = frozenset(
    {
        "_repaint_console_realtime_chip",
    }
)
REALTIME_MOVE_METHODS = (
    REALTIME_METHODS - REALTIME_DELEGATE_METHODS - REALTIME_STAY_METHODS
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CHAT_SCREEN_PATH = REPO_ROOT / "tldw_chatbook/UI/Screens/chat_screen.py"
REALTIME_PATH = REPO_ROOT / "tldw_chatbook/UI/Console_Modules/realtime.py"
WIRING_PATH = REPO_ROOT / "tldw_chatbook/UI/Console_Modules/wiring.py"
#: 2026-11-06, TASK-38988 (owner size waiver): the live ChatScreen ceiling is
#: the screen-size ratchet's own ChatScreen row (24,616 / 741 on that day),
#: read from that file rather than re-pinned here. A pin at the measured
#: 24,604 would go red unseen whenever the row is raised by a dated waiver
#: (review r10a-rev1). TASK-39055 retired the frozen-revision projection.
_, LIVE_SCREEN_LINE_CEILING, LIVE_SCREEN_METHOD_CEILING = _SCREEN_SIZE_BUDGETS[
    "tldw_chatbook/UI/Screens/chat_screen.py"
]
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
    "request_dictation_stop": "lambda: screen._request_console_dictation_stop()",
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
    assert len(source.splitlines()) <= LIVE_SCREEN_LINE_CEILING
    assert len(_direct_method_nodes(screen)) <= LIVE_SCREEN_METHOD_CEILING

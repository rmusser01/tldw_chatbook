from __future__ import annotations

import ast
from pathlib import Path
import warnings

import pytest

from tldw_chatbook.runtime_policy.bootstrap import RuntimePolicyContext
from tldw_chatbook.runtime_policy.source_state import RuntimeSourceStateStore
from tldw_chatbook.runtime_policy.types import RuntimeSourceState
from tldw_chatbook.state import (
    AppState,
    ChatState,
    NavigationState,
    NotesState,
    UIState,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_ROOT = PROJECT_ROOT / "tldw_chatbook"
APP_PATH = PRODUCTION_ROOT / "app.py"
BOOTSTRAP_PATH = PRODUCTION_ROOT / "runtime_policy" / "bootstrap.py"
SOURCE_STATE_PATH = PRODUCTION_ROOT / "runtime_policy" / "source_state.py"
SCHEDULES_WORKBENCH_PATH = (
    PRODUCTION_ROOT / "UI" / "Screens" / "scheduling" / "schedules_workbench.py"
)
PROJECTION_NAMES = (
    "current_runtime_backend",
    "runtime_backend",
    "active_server_id",
)
PROJECTION_SNAPSHOT = "_runtime_policy_projection_snapshot"
PROJECTION_PUBLISHER = "_publish_runtime_policy_projection"
PROJECTION_BOUNDARY = "_apply_runtime_policy_to_app"
PRIVATE_CONTEXT_STORE = "__runtime_policy_state_store"
PRIVATE_CONTEXT_CALLBACK = "__runtime_policy_projection_callback"
RUNTIME_POLICY_LOADER = "load_runtime_policy_for_app"


def _parse(path: Path) -> ast.Module:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _chain(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _chain(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _constant_dynamic_name(node: ast.AST) -> str | None:
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"getattr", "setattr", "delattr"}
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    ):
        return node.args[1].value
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    ):
        return node.slice.value
    return None


class _NamedOccurrenceCollector(ast.NodeVisitor):
    """Collect only syntax fields that can bind or reference one exact name."""

    def __init__(self, path: Path, target: str) -> None:
        self.path = path
        self.target = target
        self.scopes: list[str] = []
        self.occurrences: list[tuple[str, str, tuple[str, ...], int]] = []

    def _record(self, kind: str, lineno: int) -> None:
        self.occurrences.append(
            (
                str(self.path.relative_to(PROJECT_ROOT)),
                kind,
                tuple(self.scopes),
                lineno,
            )
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if node.name == self.target:
            self._record("class_definition", node.lineno)
        self.scopes.append(node.name)
        self.generic_visit(node)
        self.scopes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name == self.target:
            self._record("function_definition", node.lineno)
        self.scopes.append(node.name)
        self.generic_visit(node)
        self.scopes.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if node.name == self.target:
            self._record("async_function_definition", node.lineno)
        self.scopes.append(node.name)
        self.generic_visit(node)
        self.scopes.pop()

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == self.target:
            self._record(f"name_{type(node.ctx).__name__.lower()}", node.lineno)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr == self.target:
            self._record(
                f"attribute_{type(node.ctx).__name__.lower()}",
                node.lineno,
            )
        self.generic_visit(node)

    def visit_alias(self, node: ast.alias) -> None:
        if node.name.rsplit(".", 1)[-1] == self.target:
            self._record("import_name", node.lineno)
        if node.asname == self.target:
            self._record("import_alias", node.lineno)

    def visit_arg(self, node: ast.arg) -> None:
        if node.arg == self.target:
            self._record("argument", node.lineno)
        self.generic_visit(node)

    def visit_keyword(self, node: ast.keyword) -> None:
        if node.arg == self.target:
            self._record("keyword", node.lineno)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if _constant_dynamic_name(node) == self.target:
            self._record("dynamic_name", node.lineno)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if _constant_dynamic_name(node) == self.target:
            self._record("subscript_name", node.lineno)
        self.generic_visit(node)


def _occurrences(
    path: Path, target: str
) -> list[tuple[str, str, tuple[str, ...], int]]:
    collector = _NamedOccurrenceCollector(path, target)
    collector.visit(_parse(path))
    return collector.occurrences


def _production_occurrences(
    target: str,
) -> list[tuple[str, str, tuple[str, ...], int]]:
    found: list[tuple[str, str, tuple[str, ...], int]] = []
    for path in sorted(PRODUCTION_ROOT.rglob("*.py")):
        found.extend(_occurrences(path, target))
    return found


def _class_definition(path: Path, name: str) -> ast.ClassDef:
    return next(
        node
        for node in _parse(path).body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def _method_definition(
    class_node: ast.ClassDef,
    name: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    return next(
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )


def _top_level_function(path: Path, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in _parse(path).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_tldw_cli_neither_imports_nor_instantiates_app_state() -> None:
    tree = _parse(APP_PATH)
    imported: list[str] = []
    constructed: list[int] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.extend(
                alias.name for alias in node.names if alias.name == "AppState"
            )
        elif isinstance(node, ast.Import):
            imported.extend(
                alias.name for alias in node.names if alias.name.endswith(".AppState")
            )
        elif isinstance(node, ast.Call) and _chain(node.func).endswith("AppState"):
            constructed.append(node.lineno)

    assert imported == []
    assert constructed == []


def test_tldw_cli_runtime_projections_are_getter_only_properties() -> None:
    from tldw_chatbook.app import TldwCli

    for name in PROJECTION_NAMES:
        descriptor = vars(TldwCli).get(name)
        assert isinstance(descriptor, property), name
        assert descriptor.fset is None, name


def test_runtime_projection_snapshot_has_exact_read_write_shape() -> None:
    relative_app = str(APP_PATH.relative_to(PROJECT_ROOT))
    observed = [
        (path, kind, scopes)
        for path, kind, scopes, _line in _production_occurrences(PROJECTION_SNAPSHOT)
    ]

    assert observed == [
        (relative_app, "name_store", ("TldwCli",)),
        (
            relative_app,
            "attribute_load",
            ("TldwCli", "current_runtime_backend"),
        ),
        (relative_app, "attribute_load", ("TldwCli", "runtime_backend")),
        (relative_app, "attribute_load", ("TldwCli", "active_server_id")),
        (
            relative_app,
            "attribute_store",
            ("TldwCli", PROJECTION_PUBLISHER),
        ),
    ]

    app_class = _class_definition(APP_PATH, "TldwCli")
    default_assignment = next(
        node
        for node in app_class.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == PROJECTION_SNAPSHOT
    )
    assert ast.literal_eval(default_assignment.value) == ("local", None)


def test_projection_boundary_and_publisher_have_exact_reference_allowlists() -> None:
    relative_app = str(APP_PATH.relative_to(PROJECT_ROOT))
    relative_bootstrap = str(BOOTSTRAP_PATH.relative_to(PROJECT_ROOT))

    boundary = [
        (path, kind, scopes)
        for path, kind, scopes, _line in _production_occurrences(PROJECTION_BOUNDARY)
    ]
    assert boundary == [
        (
            relative_bootstrap,
            "name_load",
            ("load_runtime_policy_for_app",),
        ),
        (relative_bootstrap, "function_definition", ()),
    ]

    publisher = [
        (path, kind, scopes)
        for path, kind, scopes, _line in _production_occurrences(PROJECTION_PUBLISHER)
    ]
    assert publisher == [
        (
            relative_app,
            "function_definition",
            ("TldwCli",),
        ),
        (
            relative_bootstrap,
            "dynamic_name",
            (PROJECTION_BOUNDARY,),
        ),
    ]


def test_projection_boundary_uses_only_the_private_publisher() -> None:
    boundary = _top_level_function(BOOTSTRAP_PATH, PROJECTION_BOUNDARY)
    dynamic_names = [
        _constant_dynamic_name(node)
        for node in ast.walk(boundary)
        if _constant_dynamic_name(node) is not None
    ]
    attribute_writes = [
        node.attr
        for node in ast.walk(boundary)
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store)
    ]

    assert dynamic_names == [PROJECTION_PUBLISHER]
    assert attribute_writes == []
    assert not set(dynamic_names).intersection(PROJECTION_NAMES)


def test_tldw_cli_contains_no_public_projection_assignments() -> None:
    app_class = _class_definition(APP_PATH, "TldwCli")
    writes = [
        (node.attr, node.lineno)
        for node in ast.walk(app_class)
        if isinstance(node, ast.Attribute)
        and isinstance(node.ctx, (ast.Store, ast.Del))
        and node.attr in PROJECTION_NAMES
    ]
    dynamic_writes = [
        (_constant_dynamic_name(node), node.lineno)
        for node in ast.walk(app_class)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"setattr", "delattr"}
        and _constant_dynamic_name(node) in PROJECTION_NAMES
    ]

    assert writes == []
    assert dynamic_writes == []


def test_projection_boundary_never_accesses_app_state() -> None:
    boundary = _top_level_function(BOOTSTRAP_PATH, PROJECTION_BOUNDARY)
    accesses: list[int] = []
    for node in ast.walk(boundary):
        if isinstance(node, ast.Attribute) and node.attr == "app_state":
            accesses.append(node.lineno)
        elif _constant_dynamic_name(node) == "app_state":
            accesses.append(node.lineno)

    assert accesses == []


def test_runtime_policy_context_has_no_public_mutation_or_persistence_escape(
    tmp_path: Path,
) -> None:
    context = RuntimePolicyContext(
        RuntimeSourceState(),
        RuntimeSourceStateStore(tmp_path / "runtime-policy.json"),
    )
    alias = context

    assert isinstance(RuntimePolicyContext.state, property)
    assert RuntimePolicyContext.state.fset is None
    assert not hasattr(context, "persist")
    assert not hasattr(context, "store")
    with pytest.raises(AttributeError):
        alias.state = RuntimeSourceState(active_source="server")


def test_runtime_policy_context_private_fields_have_exact_structural_shape() -> None:
    relative_bootstrap = str(BOOTSTRAP_PATH.relative_to(PROJECT_ROOT))
    context_class = _class_definition(BOOTSTRAP_PATH, "RuntimePolicyContext")
    slots_assignment = next(
        node
        for node in context_class.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "__slots__"
    )
    assert ast.literal_eval(slots_assignment.value) == (
        "_owner_thread_id",
        "_snapshot",
        PRIVATE_CONTEXT_CALLBACK,
        PRIVATE_CONTEXT_STORE,
    )

    store_occurrences = [
        (path, kind, scopes)
        for path, kind, scopes, _line in _production_occurrences(PRIVATE_CONTEXT_STORE)
    ]
    assert store_occurrences == [
        (
            relative_bootstrap,
            "attribute_store",
            ("RuntimePolicyContext", "__init__"),
        ),
        (
            relative_bootstrap,
            "attribute_load",
            ("RuntimePolicyContext", "commit_state"),
        ),
    ]

    callback_occurrences = [
        (path, kind, scopes)
        for path, kind, scopes, _line in _production_occurrences(
            PRIVATE_CONTEXT_CALLBACK
        )
    ]
    assert callback_occurrences == [
        (
            relative_bootstrap,
            "attribute_store",
            ("RuntimePolicyContext", "__init__"),
        ),
        (
            relative_bootstrap,
            "attribute_load",
            ("RuntimePolicyContext", "commit_state"),
        ),
        (
            relative_bootstrap,
            "attribute_load",
            ("RuntimePolicyContext", "commit_state"),
        ),
    ]

    assert (
        _production_occurrences("_RuntimePolicyContext__runtime_policy_state_store")
        == []
    )
    assert (
        _production_occurrences(
            "_RuntimePolicyContext__runtime_policy_projection_callback"
        )
        == []
    )


def test_runtime_policy_context_commit_has_one_immediate_private_store_save() -> None:
    context_class = _class_definition(BOOTSTRAP_PATH, "RuntimePolicyContext")
    commit = _method_definition(context_class, "commit_state")
    direct_saves = [
        statement
        for statement in commit.body
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Attribute)
        and statement.value.func.attr == "save"
        and isinstance(statement.value.func.value, ast.Attribute)
        and statement.value.func.value.attr == PRIVATE_CONTEXT_STORE
    ]

    assert len(direct_saves) == 1
    assert isinstance(direct_saves[0].value.args[0], ast.Name)
    assert direct_saves[0].value.args[0].id == "candidate"


def test_runtime_policy_preparation_precedes_single_direct_app_attachment() -> None:
    loader = _top_level_function(BOOTSTRAP_PATH, "load_runtime_policy_for_app")
    prepare_indices = [
        index
        for index, statement in enumerate(loader.body)
        if isinstance(statement, ast.Assign)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "_prepare_runtime_policy_context"
    ]
    attach_indices = [
        index
        for index, statement in enumerate(loader.body)
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "app"
            and target.attr == "runtime_policy"
            for target in statement.targets
        )
    ]

    assert len(prepare_indices) == 1
    assert len(attach_indices) == 1
    assert prepare_indices[0] < attach_indices[0]


def test_tldw_cli_constructor_invokes_runtime_loader_as_standalone_expression() -> None:
    app_class = _class_definition(APP_PATH, "TldwCli")
    constructor = _method_definition(app_class, "__init__")
    standalone_calls = [
        statement
        for statement in constructor.body
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "load_runtime_policy_for_app"
    ]
    assigned_calls = [
        statement
        for statement in constructor.body
        if isinstance(statement, (ast.Assign, ast.AnnAssign))
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "load_runtime_policy_for_app"
    ]

    assert len(standalone_calls) == 1
    assert assigned_calls == []


def test_runtime_policy_loader_has_exact_production_reference_allowlist() -> None:
    relative_app = str(APP_PATH.relative_to(PROJECT_ROOT))
    relative_bootstrap = str(BOOTSTRAP_PATH.relative_to(PROJECT_ROOT))
    observed = sorted(
        (path, kind, scopes)
        for path, kind, scopes, _line in _production_occurrences(RUNTIME_POLICY_LOADER)
    )

    assert observed == sorted(
        [
            (relative_app, "import_name", ()),
            (
                relative_app,
                "name_load",
                ("TldwCli", "__init__"),
            ),
            (
                relative_bootstrap,
                "function_definition",
                (),
            ),
            (
                relative_bootstrap,
                "name_load",
                ("ensure_runtime_policy_for_app",),
            ),
        ]
    )


def test_schedules_calls_authoritative_runtime_source_with_context_and_config() -> None:
    schedules_calls = [
        node
        for node in ast.walk(_parse(SCHEDULES_WORKBENCH_PATH))
        if isinstance(node, ast.Call)
        and _chain(node.func).endswith("set_authoritative_runtime_source")
    ]
    assert len(schedules_calls) == 1
    schedules_call = schedules_calls[0]
    assert _chain(schedules_call.args[0]) == "self.app_instance.runtime_policy"
    assert {
        keyword.arg: _chain(keyword.value)
        for keyword in schedules_call.keywords
        if keyword.arg is not None
    }["app_config"] == "self.app_instance.app_config"

    production_calls = [
        (path, node)
        for path in sorted(PRODUCTION_ROOT.rglob("*.py"))
        for node in ast.walk(_parse(path))
        if isinstance(node, ast.Call)
        and _chain(node.func).endswith("set_authoritative_runtime_source")
    ]
    assert production_calls
    for path, call in production_calls:
        assert call.args, path
        first_argument = _chain(call.args[0])
        assert first_argument.endswith(".runtime_policy"), (
            path,
            first_argument,
        )
        app_config_keywords = [
            keyword for keyword in call.keywords if keyword.arg == "app_config"
        ]
        assert len(app_config_keywords) == 1, path


def test_runtime_source_state_store_references_are_confined_to_owner_modules() -> None:
    observed_paths = {
        PROJECT_ROOT / path
        for path, _kind, _scopes, _line in _production_occurrences(
            "RuntimeSourceStateStore"
        )
    }

    assert observed_paths == {BOOTSTRAP_PATH, SOURCE_STATE_PATH}


def test_legacy_state_exports_remain_serialization_compatible() -> None:
    assert all(
        state_type is not None
        for state_type in (
            AppState,
            ChatState,
            NavigationState,
            NotesState,
            UIState,
        )
    )
    original = AppState(
        runtime_source=RuntimeSourceState(
            active_source="server",
            active_server_id="server-compatible",
            server_configured=True,
        )
    )
    payload = original.to_dict()

    assert AppState.from_dict(payload).to_dict() == payload


def test_legacy_state_docs_describe_compatibility_not_live_authority() -> None:
    app_state_source = (PRODUCTION_ROOT / "state" / "app_state.py").read_text(
        encoding="utf-8"
    )
    package_source = (PRODUCTION_ROOT / "state" / "__init__.py").read_text(
        encoding="utf-8"
    )
    rendered = f"{app_state_source}\n{package_source}".lower()

    assert "single source of truth" not in rendered
    assert "centralized state" not in rendered
    assert "compatibility" in rendered

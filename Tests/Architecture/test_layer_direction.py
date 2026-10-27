"""No new upward import between layers (TASK-36942.3).

The layer map is in ``Docs/Architecture/layer-map.md``: kernel < DB < domain <
presentation, and a module may import only from its own layer or a lower one.
The one exception: DB may import domain model modules (``models`` or
``*_models``), never domain services.

This scan is stdlib AST only. Its import resolution copies
``Docs/QA/architecture-review-2026-10-25/scripts/graph.py`` and its layers
copy ``layers.py`` there, so its counts agree with the review's record.
Top-level and function-level imports both count; ``if TYPE_CHECKING:`` imports
are out of scope.

``layer_direction_allowlist.tsv`` freezes the upward imports that existed when
this test landed, one ``src_module<TAB>dst_module<TAB>kind`` row each, where
kind is ``top`` or ``lazy`` (inside a function). It may only shrink.
"""

from __future__ import annotations

import ast
import functools
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "tldw_chatbook"
ALLOWLIST_NAME = "Tests/Architecture/layer_direction_allowlist.tsv"
ALLOWLIST = REPO_ROOT / ALLOWLIST_NAME
LAYER_MAP = "Docs/Architecture/layer-map.md"

LAYERS = ("kernel", "DB", "domain", "presentation")
KERNEL = frozenset(
    {
        "<root>",
        "Utils",
        "Logging_Config.py",
        "Constants.py",
        "config.py",
        "Third_Party",
        "Metrics",
        "state",
        "runtime_policy",
    }
)
PRESENTATION = frozenset(
    {"UI", "Widgets", "Event_Handlers", "css", "app(shell)", "cli.py", "Web_Server"}
)


def _area(module: str, parents: frozenset[str]) -> str:
    """The package (or root-level module) a module belongs to, as layers.py names it."""
    parts = module.split(".")
    if len(parts) == 1:
        return "<root>"
    if len(parts) == 2 and module not in parents:
        name = parts[1]
        return (
            "app(shell)" if name == "app" or name.startswith("app_") else name + ".py"
        )
    return parts[1]


def _rank(area: str) -> int:
    if area in PRESENTATION:
        return 3
    if area == "DB":
        return 1
    if area in KERNEL:
        return 0
    return 2


def _is_model_module(module: str) -> bool:
    name = module.rsplit(".", 1)[-1]
    return name == "models" or name.endswith("_models")


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _imports(tree: ast.Module) -> list[tuple[ast.Import | ast.ImportFrom, str]]:
    """Every import statement with its kind: top, lazy or typing."""
    found: list[tuple[ast.Import | ast.ImportFrom, str]] = []

    def visit(node: ast.AST, kind: str) -> None:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            found.append((node, kind))
            return
        if isinstance(node, ast.If) and _is_type_checking(node.test):
            for stmt in node.body:
                visit(stmt, "typing")
            for stmt in node.orelse:
                visit(stmt, kind)
            return
        if kind == "top" and isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
        ):
            kind = "lazy"
        for child in ast.iter_child_nodes(node):
            visit(child, kind)

    visit(tree, "top")
    return found


@functools.cache
def _upward_imports() -> dict[tuple[str, str, str], str]:
    """Each executable upward (src, dst, kind) edge, mapped to where it first occurs."""
    modules: dict[str, Path] = {}
    for path in sorted((REPO_ROOT / PACKAGE).rglob("*.py")):
        if "__pycache__" not in path.parts:
            name = ".".join(path.relative_to(REPO_ROOT).with_suffix("").parts)
            modules[name.removesuffix(".__init__")] = path
    parents = frozenset(
        ".".join(parts[:i])
        for parts in (name.split(".") for name in modules)
        for i in range(1, len(parts))
    )

    def resolve(name: str) -> str | None:
        """The longest prefix of ``name`` that is a module (None outside the package)."""
        parts = name.split(".")
        while parts:
            if ".".join(parts) in modules:
                return ".".join(parts)
            parts.pop()
        return None

    edges: dict[tuple[str, str, str], str] = {}
    for src, path in modules.items():
        src_rank = _rank(_area(src, parents))
        if src_rank == 3:  # nothing sits above presentation
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node, kind in _imports(tree):
            if kind == "typing":
                continue
            if isinstance(node, ast.Import):
                targets = [resolve(alias.name) for alias in node.names]
            else:
                if node.level:
                    package = src.split(".")
                    if path.name != "__init__.py":
                        package = package[:-1]
                    package = package[: len(package) - (node.level - 1)]
                    base = ".".join(package + ([node.module] if node.module else []))
                else:
                    base = node.module or ""
                targets = [resolve(f"{base}.{alias.name}") for alias in node.names]
            for dst in targets:
                if dst is None:
                    continue
                dst_rank = _rank(_area(dst, parents))
                if dst_rank <= src_rank:
                    continue
                if src_rank == 1 and dst_rank == 2 and _is_model_module(dst):
                    continue  # DB may import domain model modules
                edges.setdefault(
                    (src, dst, kind),
                    f"{path.relative_to(REPO_ROOT).as_posix()}:{node.lineno}  "
                    f"{src} -> {dst}  ({LAYERS[src_rank]} -> {LAYERS[dst_rank]}, {kind})",
                )
    return edges


def _allowlist() -> set[tuple[str, str, str]]:
    rows: set[tuple[str, str, str]] = set()
    lines = ALLOWLIST.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, 1):
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        assert len(fields) == 3, (
            f"{ALLOWLIST_NAME}:{number}: expected src_module<TAB>dst_module<TAB>kind, "
            f"got {line!r}"
        )
        rows.add((fields[0], fields[1], fields[2]))
    return rows


def test_no_new_upward_import() -> None:
    edges = _upward_imports()
    new = sorted(set(edges) - _allowlist())
    assert not new, (
        f"{len(new)} new upward import(s) break the layer map in {LAYER_MAP}:\n"
        + "\n".join(f"  {edges[row]}" for row in new)
        + "\nFix the direction: move the code that needs the higher layer up into "
        "that layer, or have the higher layer pass it in (a parameter, callback or "
        "Protocol). An import inside a function counts too, so moving one there does "
        "not fix it. If an import truly belongs, explain why in the PR and get the "
        f"reviewer's agreement first. Never just append a row to {ALLOWLIST_NAME} to "
        "make this test pass."
    )


def test_allowlist_has_no_stale_rows() -> None:
    stale = sorted(_allowlist() - set(_upward_imports()))
    assert not stale, (
        f"{len(stale)} row(s) in {ALLOWLIST_NAME} no longer match "
        "an upward import. The allowlist only shrinks: delete these rows.\n"
        + "\n".join("  " + "\t".join(row) for row in stale)
    )

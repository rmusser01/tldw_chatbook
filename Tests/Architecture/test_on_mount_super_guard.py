"""Repo-wide guard: no ``BaseAppScreen`` subclass calls ``super().on_mount()``.

Textual's message-pump dispatcher walks the *whole* MRO for one Mount event
and separately invokes every class's own ``on_mount`` -- see
``message_pump.py::_get_dispatch_methods``: it iterates
``self.__class__.__mro__`` and, for each class that defines ``on_mount`` in
its own ``__dict__``, yields (and later awaits) that class's handler
independently. A subclass that ALSO calls ``super().on_mount()`` inline
therefore runs the parent's handler a second time -- once nested inside the
subclass's own call, and once more via the dispatcher's separate walk.

TASK-2610 hit this the hard way: ``LabFrameScreen.on_mount`` mounts rail
widgets, and the old ``STTSScreen.on_mount`` called ``super().on_mount()``,
double-mounting the rail and crashing every visit to Lab > Speech with
``DuplicateIds``. TASK-2710 removed the ~20 other call sites over
``BaseAppScreen`` that were harmless only because ``BaseAppScreen.on_mount``
is a single log line (see its docstring at
``tldw_chatbook/UI/Navigation/base_app_screen.py`` for the contract this
guard enforces). This test is what keeps the pattern from coming back.

Scope matches TASK-2710's audit: every current ``super().on_mount()`` call
site in the repo lives under ``UI/Screens``, ``UI/Wizards``, or ``Widgets``
(``Third_Party/`` is vendored and excluded, per the task's AC#1).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "tldw_chatbook"
SCANNED_SUBDIRS = ("UI/Screens", "UI/Wizards", "Widgets")

BASE_APP_SCREEN = "BaseAppScreen"
WIZARD_CONTAINER = "WizardContainer"


def _scanned_source_paths(package_root: Path = PACKAGE_ROOT) -> list[Path]:
    paths: list[Path] = []
    for subdir in SCANNED_SUBDIRS:
        paths.extend(sorted((package_root / subdir).rglob("*.py")))
    return paths


def _base_name(base: ast.expr) -> str | None:
    """The trailing identifier of a base-class expression.

    Handles the shapes this codebase actually uses: a plain name
    (``BaseAppScreen``), a dotted attribute (``nav.BaseAppScreen``), and a
    subscripted generic base (``ModalScreen[str | None]``).
    """
    if isinstance(base, ast.Name):
        return base.id
    if isinstance(base, ast.Attribute):
        return base.attr
    if isinstance(base, ast.Subscript):
        return _base_name(base.value)
    return None


def _calls_super_on_mount(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True if ``function``'s body contains a ``super().on_mount()`` call."""
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "on_mount"
            and isinstance(func.value, ast.Call)
            and isinstance(func.value.func, ast.Name)
            and func.value.func.id == "super"
        ):
            return True
    return False


def _own_on_mount(
    class_node: ast.ClassDef,
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """The ``on_mount`` method defined directly in ``class_node``'s body, if any."""
    for item in class_node.body:
        if (
            isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name == "on_mount"
        ):
            return item
    return None


class _ClassRecord:
    __slots__ = ("name", "bases", "path", "node")

    def __init__(
        self, name: str, bases: set[str], path: Path, node: ast.ClassDef
    ) -> None:
        self.name = name
        self.bases = bases
        self.path = path
        self.node = node


def _parse_source(source_path: Path) -> ast.Module:
    return ast.parse(source_path.read_text(encoding="utf-8"))


def _collect_class_records(source_paths: list[Path]) -> list[_ClassRecord]:
    records: list[_ClassRecord] = []
    for path in source_paths:
        tree = _parse_source(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = {
                    name
                    for name in (_base_name(base) for base in node.bases)
                    if name is not None
                }
                records.append(_ClassRecord(node.name, bases, path, node))
    return records


def _transitive_subclass_names(
    records: list[_ClassRecord], root: str
) -> set[str]:
    """Names of every class in ``records`` that (transitively) derives from ``root``.

    This is a name-based closure, not a real import-resolved MRO: it matches
    the established style of this repo's other AST census guards (e.g.
    ``Tests/DB/test_private_sqlite_inventory.py``), and is correct here
    because every ``BaseAppScreen`` subclass in this codebase spells its base
    with the bare identifier ``BaseAppScreen`` (verified by the mutation test
    below, which would catch a silent scope narrowing).
    """
    known = {root}
    changed = True
    while changed:
        changed = False
        for record in records:
            if record.name in known:
                continue
            if record.bases & known:
                known.add(record.name)
                changed = True
    return known


def _find_super_on_mount_violations(
    records: list[_ClassRecord], root: str = BASE_APP_SCREEN
) -> list[tuple[str, str, int]]:
    """``(module, class_name, lineno)`` for every offending class."""
    known = _transitive_subclass_names(records, root)
    violations: list[tuple[str, str, int]] = []
    for record in records:
        if record.name == root or record.name not in known:
            continue
        on_mount = _own_on_mount(record.node)
        if on_mount is None:
            continue
        if _calls_super_on_mount(on_mount):
            try:
                module = record.path.relative_to(PROJECT_ROOT).as_posix()
            except ValueError:
                module = record.path.as_posix()
            violations.append((module, record.name, on_mount.lineno))
    return violations


def test_no_baseappscreen_subclass_calls_super_on_mount() -> None:
    records = _collect_class_records(_scanned_source_paths())
    violations = _find_super_on_mount_violations(records)

    assert violations == [], (
        "super().on_mount() over BaseAppScreen runs the parent handler "
        "twice via Textual's whole-MRO dispatch (TASK-2610/TASK-2710) -- "
        "found: " + ", ".join(f"{module}:{cls}:{line}" for module, cls, line in violations)
    )


def test_no_wizardcontainer_subclass_calls_super_on_mount() -> None:
    """The one real bug this audit found, guarded the same way.

    ``SetupWizardContainer`` (over ``WizardContainer``, `UI/Wizards/
    BaseWizard.py` + `FirstRunSetupWizard.py`) is not a ``BaseAppScreen``
    subclass, so the guard above never covers it -- but it was the one site
    in this audit where ``super().on_mount()`` was genuinely double-running
    real work (`show_step(0)`, its on_hide/on_show pair, and a validation
    timer) rather than a harmless log line. TASK-2710 fixed it with
    `WizardContainer._post_mount_hook()`. This guards that fix the same way
    as the ``BaseAppScreen`` guard above, over the ``WizardContainer`` tree,
    so a future ``on_mount`` + ``super().on_mount()`` override cannot creep
    back in undetected.
    """
    records = _collect_class_records(_scanned_source_paths())
    violations = _find_super_on_mount_violations(records, root=WIZARD_CONTAINER)

    assert violations == [], (
        "super().on_mount() over WizardContainer double-runs show_step(0) "
        "(TASK-2710) -- found: "
        + ", ".join(f"{module}:{cls}:{line}" for module, cls, line in violations)
    )


def test_scan_actually_finds_baseappscreen_subclasses() -> None:
    """Sanity check the closure isn't accidentally empty (e.g. a path typo)."""
    records = _collect_class_records(_scanned_source_paths())
    known = _transitive_subclass_names(records, BASE_APP_SCREEN)

    # 27 screens derive from BaseAppScreen as of TASK-2710; this is a floor,
    # not an exact match, so new screens don't need to update this test.
    assert len(known) >= 20
    assert "ChatScreen" in known
    assert "SettingsScreen" in known


def test_guard_detects_a_synthetic_super_on_mount_violation(tmp_path: Path) -> None:
    """Proves the detector actually fires, per this repo's mutation-test convention."""
    package_root = tmp_path / "tldw_chatbook"
    screens_dir = package_root / "UI" / "Screens"
    screens_dir.mkdir(parents=True)

    (screens_dir / "base.py").write_text(
        "\n".join(
            (
                "class BaseAppScreen:",
                "    def on_mount(self) -> None:",
                "        pass",
                "",
            )
        ),
        encoding="utf-8",
    )
    (screens_dir / "clean_screen.py").write_text(
        "\n".join(
            (
                "from .base import BaseAppScreen",
                "",
                "class CleanScreen(BaseAppScreen):",
                "    def on_mount(self) -> None:",
                "        self._load()",
                "",
            )
        ),
        encoding="utf-8",
    )
    (screens_dir / "offending_screen.py").write_text(
        "\n".join(
            (
                "from .base import BaseAppScreen",
                "",
                "class OffendingScreen(BaseAppScreen):",
                "    def on_mount(self) -> None:",
                "        super().on_mount()",
                "        self._load()",
                "",
                # An intermediate subclass, to prove the closure is transitive.
                "class GrandchildScreen(OffendingScreen):",
                "    def on_mount(self) -> None:",
                "        super().on_mount()",
                "",
                # A class unrelated to BaseAppScreen must never be flagged.
                "class Unrelated:",
                "    def on_mount(self) -> None:",
                "        super().on_mount()",
                "",
            )
        ),
        encoding="utf-8",
    )

    records = _collect_class_records(_scanned_source_paths(package_root))
    violations = _find_super_on_mount_violations(records)

    flagged = {cls for _module, cls, _line in violations}
    assert flagged == {"OffendingScreen", "GrandchildScreen"}
    assert "CleanScreen" not in flagged
    assert "Unrelated" not in flagged


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

"""Ctrl+S is bound once in Library, as the editor Save, and nowhere new.

TASK-34000.40 (N-33). The owner ruled (ADR-031 refinement 2026-10-03) that
Ctrl+S saves in every editor: it performs the focused editor's visible commit,
flushes an autosaving editor, says why at the editor when it refuses, and is
unbound outside editors. Library had four editors with four save models and a
skill-only binding; it now has ONE binding, ``ctrl+s -> library_editor_save``,
gated by ``LibraryScreen._library_editor_save_target`` and dispatched by
``action_library_editor_save`` over ``_EDITOR_SAVE_TARGETS``.

This contract is read from the AST of ``tldw_chatbook/`` -- no app import, so
it runs in the fast PR lane:

(i) every ``ctrl+s`` binding in the tree is Library's editor Save or on the
    allowlist below, which TASK-34000.43 (Ctrl+S outside Library) shrinks;
(ii) ``LibraryScreen.BINDINGS`` holds exactly one ``ctrl+s``, not priority;
(iii) ``_EDITOR_SAVE_TARGETS`` names six editors and the dispatcher names
    each one.

A binding is a ``Binding(...)`` call, or a tuple inside a list/tuple assigned
to a name ending in ``BINDINGS``. A footer chip such as ``("ctrl+s", "save")``
is neither, so it is not counted.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2] / "tldw_chatbook"
LIBRARY_SCREEN = ROOT / "UI" / "Screens" / "library_screen.py"

#: The one Library binding (ruling: one binding, one action, one gate).
LIBRARY_EDITOR_SAVE = ("UI/Screens/library_screen.py", "library_editor_save")

#: Ctrl+S bindings outside Library, each owned by TASK-34000.43 (conform the
#: rest of the app to the amended ADR-031). Shrink it; never grow it.
ALLOWLIST = {
    # TASK-34000.43: Roleplay's editor Save (R41) -- an editor, but its gate
    # and chip are still Roleplay's own; .43 brings it under the one rule.
    ("UI/Screens/personas_screen.py", "personas_save"),
    # TASK-34000.43: the folder picker's "Select this folder" -- a modal, not
    # an editor, so it breaks rule (d); .43 moves it off Ctrl+S.
    ("Third_Party/textual_fspicker/base_dialog.py", "select_current_folder"),
    # TASK-34000.43: the prompt block editor opens its "Save…" menu; it is
    # embedded in the Library prompt editor's Advanced mode, where it wins
    # the focus chain, so Library's target excludes its subtree.
    ("Widgets/Prompts/prompt_block_editor.py", "save_menu"),
}


def _keys(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {part.strip() for part in node.value.split(",")}
    return set()


def _action(node: ast.AST) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return "?"


def _is_binding_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
    return name == "Binding"


def _binding_lists(tree: ast.AST):
    """Every list/tuple literal assigned to a ``...BINDINGS`` name."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        names = [
            target.id if isinstance(target, ast.Name) else getattr(target, "attr", "")
            for target in targets
        ]
        if any(name.endswith("BINDINGS") for name in names) and isinstance(
            value, (ast.List, ast.Tuple)
        ):
            yield value


def ctrl_s_bindings(source: str) -> list[tuple[str, int, bool]]:
    """``(action, line, priority)`` for each ``ctrl+s`` binding in ``source``."""
    tree = ast.parse(source)
    found: dict[int, tuple[str, int, bool]] = {}
    for node in ast.walk(tree):
        if _is_binding_call(node) and node.args and "ctrl+s" in _keys(node.args[0]):
            action = _action(node.args[1]) if len(node.args) > 1 else "?"
            for keyword in node.keywords:
                if keyword.arg == "action":
                    action = _action(keyword.value)
            priority = any(
                keyword.arg == "priority"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in node.keywords
            )
            found[id(node)] = (action, node.lineno, priority)
    for listing in _binding_lists(tree):
        for element in listing.elts:
            if (
                isinstance(element, ast.Tuple)
                and len(element.elts) >= 2
                and "ctrl+s" in _keys(element.elts[0])
            ):
                found[id(element)] = (_action(element.elts[1]), element.lineno, False)
    return sorted(found.values(), key=lambda row: row[1])


def _tree_bindings() -> list[tuple[str, str, int]]:
    rows = []
    for path in sorted(ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "ctrl+s" not in source:
            continue
        relative = path.relative_to(ROOT).as_posix()
        for action, line, _priority in ctrl_s_bindings(source):
            rows.append((relative, action, line))
    return rows


def _class_bindings(source: str, class_name: str) -> list[ast.AST]:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for statement in node.body:
                if isinstance(statement, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "BINDINGS"
                    for target in statement.targets
                ):
                    return list(statement.value.elts)
    raise AssertionError(f"{class_name}.BINDINGS not found")


def test_every_ctrl_s_binding_is_the_library_editor_save_or_allowlisted() -> None:
    rows = _tree_bindings()
    unexpected = [
        row for row in rows if (row[0], row[1]) not in ALLOWLIST | {LIBRARY_EDITOR_SAVE}
    ]
    assert not unexpected, (
        "a new ctrl+s binding: ADR-031 (refinement 2026-10-03) binds ctrl+s only "
        f"as an editor's Save -- route it through that rule: {unexpected}"
    )
    seen = {(path, action) for path, action, _line in rows}
    assert LIBRARY_EDITOR_SAVE in seen, rows
    # A stale allowlist row means .43 fixed a site; drop the row with it.
    assert ALLOWLIST <= seen, sorted(ALLOWLIST - seen)


def test_library_screen_binds_ctrl_s_exactly_once_as_the_editor_save() -> None:
    source = LIBRARY_SCREEN.read_text(encoding="utf-8")
    entries = _class_bindings(source, "LibraryScreen")
    ctrl_s = []
    for entry in entries:
        if _is_binding_call(entry) and entry.args and "ctrl+s" in _keys(entry.args[0]):
            ctrl_s.append(entry)
        elif isinstance(entry, ast.Tuple) and entry.elts and "ctrl+s" in _keys(
            entry.elts[0]
        ):
            ctrl_s.append(entry)
    assert len(ctrl_s) == 1, [ast.unparse(entry) for entry in ctrl_s]
    module_rows = ctrl_s_bindings(source)
    assert [(action, priority) for action, _line, priority in module_rows] == [
        ("library_editor_save", False)
    ], module_rows


def test_every_editor_save_target_is_dispatched() -> None:
    tree = ast.parse(LIBRARY_SCREEN.read_text(encoding="utf-8"))
    targets: tuple[str, ...] | None = None
    dispatcher: ast.AsyncFunctionDef | ast.FunctionDef | None = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_EDITOR_SAVE_TARGETS"
            for target in node.targets
        ):
            targets = tuple(ast.literal_eval(node.value))
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "action_library_editor_save"
        ):
            dispatcher = node
    assert targets is not None, "_EDITOR_SAVE_TARGETS is gone from library_screen.py"
    assert len(targets) == 6 and len(set(targets)) == 6, targets
    assert dispatcher is not None, "action_library_editor_save is gone"
    named = {
        node.value
        for node in ast.walk(dispatcher)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert set(targets) <= named, sorted(set(targets) - named)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        # A screen-level tuple binding and a Binding call are both caught.
        (
            "class S:\n    BINDINGS = [('ctrl+s', 'save', 'Save')]\n",
            [("save", 2, False)],
        ),
        (
            "X_BINDINGS = [Binding('ctrl+x,ctrl+s', 'go', 'Go', priority=True)]\n",
            [("go", 1, True)],
        ),
        # A footer chip is not a binding.
        ("CHIPS = (('ctrl+s', 'save now'),)\n", []),
        ("BINDINGS = [('ctrl+n', 'new', 'New')]\n", []),
    ],
)
def test_the_scan_catches_bindings_and_ignores_footer_chips(source, expected) -> None:
    """Negative controls: the scan is what the three tests above rely on."""
    assert ctrl_s_bindings(source) == expected

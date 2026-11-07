"""Restore only TASK-35613 targets, preserving later Console return fixes."""

import ast
from pathlib import Path

import presentation_baseline as original

BASE_REVISION = "f152dfc4a6c4fbf2fcd277543757a46a0d2874e2"
METHODS = {
    "_active_console_provider_model_display",
    "_active_console_settings_readiness",
}
original.BASE_REVISION = BASE_REVISION
original.MODULES = {
    "tldw_chatbook.UI.Screens.chat_screen",
    "tldw_chatbook.Utils.textual_css_fastpath",
}
OriginalLoader = original.OriginalLoader


def restore_derivations(current: str, before: str) -> str:
    """Replace the two method definitions and assert all other AST stays intact."""

    def chat_class(source):
        tree = ast.parse(source)
        owner = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ChatScreen"
        )
        return tree, owner

    _, previous = chat_class(before)
    bodies = {
        node.name: before.splitlines(keepends=True)[node.lineno - 1 : node.end_lineno]
        for node in previous.body
        if isinstance(node, ast.FunctionDef) and node.name in METHODS
    }
    assert set(bodies) == METHODS
    tree, owner = chat_class(current)
    targets = [
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name in METHODS
    ]
    assert len(targets) == len(METHODS) and all(not n.decorator_list for n in targets)
    lines = current.splitlines(keepends=True)
    for node in sorted(targets, key=lambda n: n.lineno, reverse=True):
        lines[node.lineno - 1 : node.end_lineno] = bodies[node.name]
    restored = "".join(lines)
    other_tree, other_owner = chat_class(restored)
    for item in (owner, other_owner):
        item.body = [
            node
            for node in item.body
            if not (isinstance(node, ast.FunctionDef) and node.name in METHODS)
        ]
    assert ast.dump(tree) == ast.dump(other_tree), "control changed unrelated code"
    return restored


class ScopedLoader(OriginalLoader):
    def exec_module(self, module):
        if module.__name__ == "tldw_chatbook.UI.Screens.chat_screen":
            self.source = restore_derivations(
                Path(self.filename).read_text(encoding="utf-8"),
                self.source.decode("utf-8"),
            )
        super().exec_module(module)


original.OriginalLoader = ScopedLoader

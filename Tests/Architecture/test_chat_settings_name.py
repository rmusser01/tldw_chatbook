"""No user-visible string calls Chat settings "Conversation settings" (TASK-33006.5).

The modal was renamed Chat settings (spec §3, AC#9). This scans every string
literal in the package, f-string pieces included, for the old name in any
case. Log calls (``logger.*``, ``logging.*``) and the messages of raised
exceptions may keep it, as AC#9 allows; docstrings are scanned too, so the
old name does not survive in the code that describes the modal.
"""

from __future__ import annotations

import ast
import re
import warnings
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / "tldw_chatbook"
OLD_NAME = re.compile(r"conversation\s+settings", re.IGNORECASE)


def _exempt_nodes(tree: ast.AST) -> set[int]:
    """Return the ids of every node inside a log call or a raised exception."""
    roots: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and node.exc is not None:
            roots.append(node.exc)
        elif isinstance(node, ast.Call):
            base = node.func
            while isinstance(base, (ast.Attribute, ast.Call)):
                base = base.value if isinstance(base, ast.Attribute) else base.func
            if isinstance(base, ast.Name) and base.id in {"logger", "logging"}:
                roots.append(node)
    return {id(child) for root in roots for child in ast.walk(root)}


def old_name_strings(source: str) -> list[tuple[int, str]]:
    """Return ``(line, text)`` for each string literal naming the old modal.

    Args:
        source: Python source code.

    Returns:
        The offending literals outside log calls and raised exceptions.
    """
    with warnings.catch_warnings():  # another module's invalid escapes
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(source)
    exempt = _exempt_nodes(tree)
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in exempt
        and OLD_NAME.search(node.value)
    ]


def test_no_user_visible_string_names_conversation_settings() -> None:
    """AC#9: notices, buttons, help and palette copy say Chat settings."""
    offenders = [
        f"{path.relative_to(PACKAGE.parent)}:{line}: {text!r}"
        for path in sorted(PACKAGE.rglob("*.py"))
        for line, text in old_name_strings(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, "\n".join(offenders)


def test_the_guard_flags_copy_and_spares_logs_and_exceptions() -> None:
    """The scan catches a notice and spares the two exempt kinds."""
    source = (
        'self.notify("Open Conversation settings again.")\n'
        'label = f"Return to {where} conversation  settings"\n'
        'logger.debug("Conversation settings return restore failed")\n'
        'logger.opt(lazy=True).error("Conversation settings broke")\n'
        'raise ValueError("Conversation settings return intent is invalid")\n'
    )
    assert [line for line, _text in old_name_strings(source)] == [1, 2]

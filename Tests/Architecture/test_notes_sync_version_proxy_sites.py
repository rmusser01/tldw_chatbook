"""A binding's version is never compared against a note's version.

TASK-34000.49. A binding's ``note_version`` is the note's version at the last
baseline commit. Comparing it against the NOTE's version -- the live
``note.version``, or ``operation.expected_note_version``, which is the live
version captured at admission -- made every version-only move (a delete and
restore, a keywords-only save, a title-only edit) refuse the next
``update_note`` as ``stale_observation`` for good, and (fix round 1) made an
interrupted one unrecoverable at ``reconstruct_request``, because the
digest-only reconciler never re-bases a binding whose content did not change.
The content baseline (``_note_matches_baseline``), the journaled binding
(``_binding_matches_reviewed``) and the exact fresh re-observe
(``note != request.note``) carry "the note is still what we synced"; a
binding version is compared only against the journal-recorded copy of the
binding itself (``reviewed.get("note_version")``).

The scan is derived from the executor's AST, not from line numbers. An
operand is a BINDING version if anything in its subtree names ``note_version``
(an attribute, a ``getattr`` / ``.get`` / subscript key, or a local name bound
from one in the same function); it is a NOTE version if anything in its
subtree names ``version`` or ``expected_note_version`` the same way. Any
``Compare`` -- plain, reversed, chained, ``in`` / ``not in``, through ``int()``
/ ``str()`` / arithmetic / tuple wrappers -- with a BINDING operand and a NOTE
operand is a violation. The census test lists the comparisons that remain by
enclosing function so a new site of either shape fails loudly with its name,
and the negative controls feed the scan the refactor shapes a reviewer found
could slip past a narrower matcher.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_EXECUTOR = (
    Path(__file__).resolve().parents[2]
    / "tldw_chatbook"
    / "Notes"
    / "notes_sync_executor.py"
)

pytestmark = pytest.mark.unit

#: Names that make an operand a BINDING version, however it is reached.
_BINDING_VERSION_NAMES = frozenset({"note_version"})
#: Names that make an operand a NOTE version. ``expected_note_version`` is the
#: live note's version captured at admission -- a note fact, never a binding
#: fact -- which is why the attribute name is matched exactly here and not
#: as a ``note_version`` suffix.
_NOTE_VERSION_NAMES = frozenset({"version", "expected_note_version"})

_FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


class _Scope:
    """One function's simple local bindings (``name = <expr>``, walrus)."""

    def __init__(self, function: ast.AST) -> None:
        self.bound: dict[str, ast.AST] = {}
        for node in ast.walk(function):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        self.bound[target.id] = node.value
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name) and node.value is not None:
                    self.bound[node.target.id] = node.value
            elif isinstance(node, ast.NamedExpr):
                if isinstance(node.target, ast.Name):
                    self.bound[node.target.id] = node.value

    def mentions(
        self, node: ast.AST, names: frozenset[str], seen: set[str] | None = None
    ) -> bool:
        """Whether ``node``'s subtree names one of ``names`` in any form."""

        seen = set() if seen is None else seen
        for sub in ast.walk(node):
            if isinstance(sub, ast.Attribute) and sub.attr in names:
                return True
            if isinstance(sub, ast.Constant) and sub.value in names:
                return True
            if isinstance(sub, ast.Name) and sub.id in self.bound:
                if sub.id in seen:
                    continue
                seen.add(sub.id)
                if self.mentions(self.bound[sub.id], names, seen):
                    return True
        return False


def _version_comparisons(source: str) -> list[tuple[str, int, str, bool]]:
    """Every Compare with a BINDING-version operand.

    Returns ``(enclosing function, line, source text, against_note_version)``.
    """

    tree = ast.parse(source)
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    scopes: dict[ast.AST, _Scope] = {}
    found: list[tuple[str, int, str, bool]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        owner: ast.AST = node
        while owner in parents and not isinstance(owner, _FunctionNode):
            owner = parents[owner]
        enclosing = owner.name if isinstance(owner, _FunctionNode) else "<module>"
        scope = scopes.setdefault(owner, _Scope(owner))
        operands = [node.left, *node.comparators]
        binding_operands = [
            operand
            for operand in operands
            if scope.mentions(operand, _BINDING_VERSION_NAMES)
        ]
        if not binding_operands:
            continue
        against_note = any(
            scope.mentions(operand, _NOTE_VERSION_NAMES) for operand in operands
        )
        found.append((enclosing, node.lineno, ast.unparse(node), against_note))
    return found


def _executor_source() -> str:
    return _EXECUTOR.read_text(encoding="utf-8")


def _flagged(source: str) -> list[tuple[str, str]]:
    return [
        (function, text)
        for function, _line, text, against_note in _version_comparisons(source)
        if against_note
    ]


def test_no_binding_version_is_compared_against_a_note_version() -> None:
    live = [
        (function, line, text)
        for function, line, text, against_note in _version_comparisons(
            _executor_source()
        )
        if against_note
    ]
    assert live == [], (
        "A binding's note_version is compared against a note's version at "
        f"{live}. The binding's version is the version at the last baseline "
        "commit and is only ever compared against the journal-recorded copy of "
        "the binding; a version-only move (delete+restore, keywords-only save, "
        "title-only edit) would wedge every later update_note, or make an "
        "interrupted one unrecoverable (TASK-34000.49). Guard content with "
        "_note_matches_baseline, _binding_matches_reviewed or the fresh "
        "re-observe."
    )


def test_the_surviving_comparisons_are_the_journal_side_census() -> None:
    """A new binding-version comparison of any shape fails here with its name."""

    census = sorted(
        {
            function
            for function, _line, _text, _against_note in _version_comparisons(
                _executor_source()
            )
        }
    )
    assert census == ["_binding_matches_reviewed"], (
        f"binding-version comparison sites are now {census}. The only "
        "permitted shape compares the binding against the journal-recorded "
        "copy of the binding (reviewed.get('note_version')). Add one elsewhere "
        "and the first test says why not."
    )


def test_the_rule_is_recorded_on_the_content_baseline_helper() -> None:
    source = _executor_source()
    helper = source[source.index("def _note_matches_baseline(") :]
    # Whitespace-normalized: the docstring wraps at 79 columns.
    helper = " ".join(helper[: helper.index("\n\n\n")].split())
    assert "never a precondition against the live note" in helper
    # Fix round 1: the admitted version is a note fact; the docstring must not
    # list it as something a binding version may be compared against.
    assert "journal-recorded copy of the binding" in helper
    assert "operation.expected_note_version" not in helper


def _function(body: str) -> str:
    return "def _probe(self, binding, note, request, operation, x, a, b):\n" + "".join(
        "    " + line + "\n" for line in body.strip("\n").splitlines()
    )


_MUST_BE_FLAGGED = (
    pytest.param("return binding.note_version != note.version", id="plain"),
    pytest.param("return note.version != binding.note_version", id="reversed"),
    pytest.param(
        "return binding.note_version != request.note.version", id="request-note"
    ),
    pytest.param(
        "return self._store.get_binding(x).note_version != note.version",
        id="inline-store-lookup",
    )
    ,
    pytest.param(
        'return getattr(binding, "note_version") != note.version', id="getattr-binding"
    ),
    pytest.param(
        'return binding.note_version != getattr(note, "version")', id="getattr-note"
    ),
    pytest.param(
        "bv = binding.note_version\nreturn bv != note.version", id="hoisted-binding"
    ),
    pytest.param(
        "v = note.version\nreturn binding.note_version != v", id="hoisted-note"
    ),
    pytest.param("return int(binding.note_version) != note.version", id="int-wrap"),
    pytest.param(
        "return str(binding.note_version) != str(note.version)", id="str-wrap"
    ),
    pytest.param(
        "return (binding.note_version, a) != (note.version, b)", id="tuple"
    ),
    pytest.param("return binding.note_version + 1 != note.version", id="binop"),
    pytest.param(
        "return binding.note_version not in {note.version}", id="not-in-set"
    ),
    pytest.param(
        "return (bv := binding.note_version) != request.note.version", id="walrus"
    ),
    pytest.param("return a < binding.note_version <= note.version", id="chained"),
    pytest.param("return self._binding.note_version != note.version", id="self-attr"),
    pytest.param("return x.record.note_version != note.version", id="deep-chain"),
    pytest.param(
        'return operation.expected_note_version != x.get("note_version")',
        id="dropped-reconstruct-clause",
    ),
    pytest.param(
        'return x["note_version"] != note.version', id="subscript-journal-binding"
    ),
    pytest.param(
        "return binding.note_version != operation.expected_note_version",
        id="binding-vs-admitted-version",
    ),
)


@pytest.mark.parametrize("body", _MUST_BE_FLAGGED)
def test_negative_control_every_refactor_shape_is_flagged(body: str) -> None:
    flagged = _flagged(_function(body))
    assert [function for function, _text in flagged] == ["_probe"], (
        f"not flagged: {body!r}"
    )


_MUST_STAY_PERMITTED = (
    pytest.param(
        'return binding.note_version != x.get("note_version")',
        True,
        id="binding-vs-journaled-binding",
    ),
    pytest.param(
        "return operation.expected_note_version != note.version",
        False,
        id="admitted-version-vs-live-note",
    ),
    pytest.param(
        "return note.version != operation.expected_note_version + 1",
        False,
        id="post-write-version",
    ),
    pytest.param(
        'return operation.expected_note_version != x.get("reviewed_note_version")',
        False,
        id="two-journal-facts-of-the-note",
    ),
    pytest.param(
        'return self._encoded_binding(binding) != x.get("current_binding")',
        False,
        id="whole-binding-vs-journal",
    ),
)


@pytest.mark.parametrize(("body", "in_census"), _MUST_STAY_PERMITTED)
def test_negative_control_the_permitted_shapes_are_not_flagged(
    body: str, in_census: bool
) -> None:
    found = _version_comparisons(_function(body))
    assert _flagged(_function(body)) == []
    assert [function for function, _l, _t, _a in found] == (
        ["_probe"] if in_census else []
    )


def test_negative_control_the_live_executor_source_is_what_the_scan_reads() -> None:
    """Re-add the dropped clauses to the REAL file text: the scan must go red.

    Guards against the scan silently reading an empty or unrelated file and
    passing for the wrong reason.
    """

    source = _executor_source()
    assert "class NotesSyncExecutor" in source
    mutated = source + (
        "\n\n"
        "def _reinstated_precondition(self, request, binding):\n"
        "    if binding.note_version != request.note.version:\n"
        '        raise RuntimeError("stale_observation")\n'
        "\n\n"
        "def _reinstated_reconstruction(self, operation, reviewed_binding):\n"
        '    if operation.expected_note_version != reviewed_binding.get("note_version"):\n'
        '        raise RuntimeError("recovery_authority_changed")\n'
    )
    assert [function for function, _text in _flagged(mutated)] == [
        "_reinstated_precondition",
        "_reinstated_reconstruction",
    ]

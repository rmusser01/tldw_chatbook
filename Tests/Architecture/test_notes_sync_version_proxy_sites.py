"""``binding.note_version`` is never a precondition against the live note.

TASK-34000.49. A binding's ``note_version`` is the note's version at the last
baseline commit. Comparing it against the LIVE note's ``version`` made every
version-only move (a delete and restore, a keywords-only save, a title-only
edit) refuse the next ``update_note`` as ``stale_observation`` for good,
because the digest-only reconciler never re-bases a binding whose content did
not change. The content baseline (``_note_matches_baseline``) and the exact
fresh re-observe (``note != request.note``) carry "the note is still what we
synced"; ``binding.note_version`` is compared only against journal-recorded
binding facts (``reviewed.get("note_version")``, ``operation.expected_note_version``).

This scan is derived from the executor's AST, not from line numbers: any
``Compare`` with a ``<name>.note_version`` operand on one side and a bare
``.version`` attribute (``note.version``, ``request.note.version``,
``current_note.version`` ...) on the other is a live-note comparison and
fails the first test. The second test is the census of the comparisons that
remain, by enclosing function, so a new site of either shape fails loudly
with its name. The third is the negative control: a snippet that re-adds the
dropped clause must be flagged.
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


def _attribute_chain(node: ast.AST) -> tuple[str, ...] | None:
    """``request.note.version`` -> ("request", "note", "version"); else None."""

    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return tuple(reversed(parts))


def _is_binding_version(node: ast.AST) -> bool:
    chain = _attribute_chain(node)
    return chain is not None and len(chain) == 2 and chain[1] == "note_version"


def _is_live_note_version(node: ast.AST) -> bool:
    """A bare ``.version`` attribute: the live note, never a journal fact.

    Journal-side operands look different on purpose: ``operation.
    expected_note_version`` is a different attribute and ``reviewed.get(
    "note_version")`` is a call.
    """

    chain = _attribute_chain(node)
    return chain is not None and len(chain) >= 2 and chain[-1] == "version"


def _binding_version_comparisons(source: str) -> list[tuple[str, int, str, bool]]:
    """Every Compare with a ``<name>.note_version`` operand.

    Returns ``(enclosing function, line, source text, against_live_note)``.
    """

    tree = ast.parse(source)
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    found: list[tuple[str, int, str, bool]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        if not any(_is_binding_version(operand) for operand in operands):
            continue
        against_live = any(
            _is_live_note_version(operand)
            for operand in operands
            if not _is_binding_version(operand)
        )
        owner: ast.AST = node
        while owner in parents and not isinstance(
            owner, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            owner = parents[owner]
        enclosing = (
            owner.name
            if isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef))
            else "<module>"
        )
        found.append((enclosing, node.lineno, ast.unparse(node), against_live))
    return found


def _executor_source() -> str:
    return _EXECUTOR.read_text(encoding="utf-8")


def test_no_binding_version_comparison_against_the_live_note() -> None:
    live = [
        (function, line, text)
        for function, line, text, against_live in _binding_version_comparisons(
            _executor_source()
        )
        if against_live
    ]
    assert live == [], (
        "binding.note_version is compared against the live note's version at "
        f"{live}. The binding's version is the version at the last baseline "
        "commit and is only ever compared against journal-recorded binding "
        "facts; a version-only move (delete+restore, keywords-only save, "
        "title-only edit) would wedge every later update_note (TASK-34000.49). "
        "Guard content with _note_matches_baseline or the fresh re-observe."
    )


def test_the_surviving_comparisons_are_the_journal_side_census() -> None:
    """A new ``note_version`` comparison of any shape fails here with its name."""

    census = sorted(
        {
            function
            for function, _line, _text, _against_live in (
                _binding_version_comparisons(_executor_source())
            )
        }
    )
    assert census == ["_binding_matches_reviewed"], (
        f"binding.note_version comparison sites are now {census}. The only "
        "permitted shape compares the binding against the journal-recorded "
        "binding (reviewed.get('note_version')); the operation-side checks "
        "(operation.expected_note_version, reconstruct_request) never read the "
        "binding's version. Add a live-note precondition elsewhere and the "
        "first test says why not."
    )


def test_the_rule_is_recorded_on_the_content_baseline_helper() -> None:
    source = _executor_source()
    helper = source[source.index("def _note_matches_baseline(") :]
    helper = helper[: helper.index("\n\n\n")]
    assert "never a precondition against the live note" in helper


_REINSTATED_CLAUSE = '''
async def _validate_initial(self, request):
    binding = self._require_owner_identity(request)
    if request.action_kind is NotesSyncActionKind.UPDATE_NOTE:
        if (
            binding.note_version != request.note.version
            or not _note_matches_baseline(
                request.note, binding.content_digest, binding.serialization
            )
        ):
            raise RuntimeError("stale_observation")
'''

_JOURNAL_SIDE_ONLY = '''
def _binding_matches_reviewed(binding, metadata):
    reviewed = metadata.get("binding")
    return not (
        binding.note_version != reviewed.get("note_version")
        or operation.expected_note_version != reviewed_binding.get("note_version")
    )

def _proven_post_write_baseline(self, note, operation):
    if note.version != operation.expected_note_version + 1:
        raise RuntimeError("postcondition_failed")
'''


def test_negative_control_the_dropped_clause_is_flagged_when_reinstated() -> None:
    """The scan must catch the exact clause this task removed."""

    flagged = [
        (function, text)
        for function, _line, text, against_live in _binding_version_comparisons(
            _REINSTATED_CLAUSE
        )
        if against_live
    ]
    assert flagged == [
        ("_validate_initial", "binding.note_version != request.note.version")
    ]


def test_negative_control_the_journal_side_shapes_are_not_flagged() -> None:
    """The permitted shapes stay permitted: only the binding/journal pair counts."""

    found = _binding_version_comparisons(_JOURNAL_SIDE_ONLY)
    assert [(function, against_live) for function, _l, _t, against_live in found] == [
        ("_binding_matches_reviewed", False)
    ]


def test_negative_control_the_live_executor_source_is_what_the_scan_reads() -> None:
    """Re-insert the clause into the REAL file text: the scan must go red.

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
    )
    live = [
        function
        for function, _line, _text, against_live in _binding_version_comparisons(
            mutated
        )
        if against_live
    ]
    assert live == ["_reinstated_precondition"]

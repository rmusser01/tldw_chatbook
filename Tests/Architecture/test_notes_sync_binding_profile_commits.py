"""Every file-profile comparison and binding commit goes through the newline rule.

TASK-34000.48. A file observation whose logical text has no ``"\\n"`` carries
no evidence about its line-ending convention; ``_parse_supported_text`` still
has to report ``lf`` or ``crlf``, so it says ``lf``. Comparing that raw
observed profile with a recorded one (the binding's, the journal's, the
reviewed snapshot's, a create's candidate) refused a correct write of a
one-line note into a CRLF file -- first in ``PosixNotesSyncFilesystem.replace``
's own post-write check, then at every executor site that compares the same
two profiles -- and committing it would have flipped the file's recorded
convention on its next multi-line write. The one rule is
``notes_sync_filesystem.proven_profile(observed, text, recorded)``: a
newline-free observation inherits the recorded ``newline`` and nothing else;
the executor wraps it as ``_bound_file_serialization(file, recorded)``.

The scan is derived from the AST of the three sync modules, not from line
numbers. A RAW observed profile is a ``_file_serialization(...)`` call, an
``<x>.observation.serialization`` attribute, or a local name bound (directly
or through a conditional) to one in the same function. A PROVEN profile is a
``proven_profile(...)`` / ``_bound_file_serialization(...)`` call or a local
bound to one. A ``Compare`` whose operands mention a raw profile and no proven
one is a violation, whatever the shape (reversed, hoisted, tuple-wrapped,
``in`` a set, chained). A ``serialization=`` / ``baseline_serialization=``
keyword on a binding commit (``replace(binding, ...)``,
``NotesSyncBindingRecord(...)``, ``BindingObservation(...)``) whose value
mentions a raw profile and no proven one is a violation too. The census test
lists the surviving sites by enclosing function so a new one of either shape
fails loudly with its name, and the negative controls feed the scan each
refactor shape plus the base's own ``_file_holds_note`` body.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_NOTES = Path(__file__).resolve().parents[2] / "tldw_chatbook" / "Notes"
_MODULES = {
    "executor": _NOTES / "notes_sync_executor.py",
    "runtime": _NOTES / "notes_sync_runtime.py",
    "filesystem": _NOTES / "notes_sync_filesystem.py",
}

pytestmark = pytest.mark.unit

_PROVEN_CALLS = frozenset({"proven_profile", "_bound_file_serialization"})
_RAW_CALLS = frozenset({"_file_serialization"})
_COMMIT_CALLS = frozenset({"replace", "NotesSyncBindingRecord", "BindingObservation"})
_COMMIT_KEYWORDS = frozenset({"serialization", "baseline_serialization"})
#: The helpers themselves are where a raw profile is turned into a proven
#: one; their bodies are not comparison or commit sites.
_RULE_HOLDERS = frozenset(
    {"proven_profile", "_bound_file_serialization", "_file_serialization"}
)

_FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _is_raw_node(node: ast.AST) -> bool:
    if isinstance(node, ast.Call) and _call_name(node) in _RAW_CALLS:
        return True
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "serialization"
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "observation"
    )


def _is_proven_node(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and _call_name(node) in _PROVEN_CALLS


class _Scope:
    """One function's simple local bindings (``name = <expr>``, walrus).

    A name carries the VALUE KIND of what it is bound to -- raw or proven,
    looked up through conditionals, containers and further names -- not
    everything beneath that expression. (``candidate = NotesSyncBindingRecord(
    ..., serialization=file.observation.serialization)`` makes ``candidate`` a
    binding, not a profile; a later ``relative_path in claimed_paths`` must
    not be read as a profile comparison because ``relative_path`` was derived
    from an earlier, unrelated ``candidate``.)
    """

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

    def kinds(self, node: ast.AST, seen: set[str] | None = None) -> set[str]:
        """The profile kinds (``raw``/``proven``) the VALUE of ``node`` can be."""

        seen = set() if seen is None else seen
        if _is_raw_node(node):
            return {"raw"}
        if _is_proven_node(node):
            return {"proven"}
        if isinstance(node, ast.IfExp):
            return self.kinds(node.body, seen) | self.kinds(node.orelse, seen)
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            found: set[str] = set()
            for element in node.elts:
                found |= self.kinds(element, seen)
            return found
        if isinstance(node, ast.BinOp):
            return self.kinds(node.left, seen) | self.kinds(node.right, seen)
        if isinstance(node, ast.NamedExpr):
            return self.kinds(node.value, seen)
        if isinstance(node, ast.Name) and node.id in self.bound and node.id not in seen:
            seen.add(node.id)
            return self.kinds(self.bound[node.id], seen)
        return set()

    def mentions(self, node: ast.AST, kind: str) -> bool:
        """Whether any node in ``node``'s subtree has value kind ``kind``."""

        return any(kind in self.kinds(sub) for sub in ast.walk(node))


def _sites(source: str) -> list[tuple[str, str, int, str, bool]]:
    """Every profile comparison and binding-commit keyword in ``source``.

    Returns ``(kind, enclosing function, line, source text, violation)`` where
    kind is ``"compare"`` or ``"commit"``.
    """

    tree = ast.parse(source)
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    scopes: dict[ast.AST, _Scope] = {}
    found: list[tuple[str, str, int, str, bool]] = []

    def owner_of(node: ast.AST) -> tuple[str, _Scope]:
        owner: ast.AST = node
        while owner in parents and not isinstance(owner, _FunctionNode):
            owner = parents[owner]
        name = owner.name if isinstance(owner, _FunctionNode) else "<module>"
        return name, scopes.setdefault(owner, _Scope(owner))

    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            enclosing, scope = owner_of(node)
            if enclosing in _RULE_HOLDERS:
                continue
            operands = [node.left, *node.comparators]
            raw = any(scope.mentions(operand, "raw") for operand in operands)
            proven = any(scope.mentions(operand, "proven") for operand in operands)
            if raw or proven:
                found.append(
                    ("compare", enclosing, node.lineno, ast.unparse(node), raw and not proven)
                )
        elif isinstance(node, ast.Call) and _call_name(node) in _COMMIT_CALLS:
            enclosing, scope = owner_of(node)
            for keyword in node.keywords:
                if keyword.arg not in _COMMIT_KEYWORDS:
                    continue
                raw = scope.mentions(keyword.value, "raw")
                proven = scope.mentions(keyword.value, "proven")
                found.append(
                    (
                        "commit",
                        enclosing,
                        keyword.value.lineno,
                        f"{keyword.arg}={ast.unparse(keyword.value)}",
                        raw and not proven,
                    )
                )
    return found


def _source(module: str) -> str:
    return _MODULES[module].read_text(encoding="utf-8")


def _violations(source: str) -> list[tuple[str, str, str]]:
    return [
        (kind, function, text)
        for kind, function, _line, text, violation in _sites(source)
        if violation
    ]


@pytest.mark.parametrize("module", sorted(_MODULES))
def test_no_raw_observed_profile_is_compared_or_committed(module: str) -> None:
    live = [
        (kind, function, line, text)
        for kind, function, line, text, violation in _sites(_source(module))
        if violation
    ]
    assert live == [], (
        f"{module}: a file's observed profile is compared or committed raw at "
        f"{live}. A text with no line ending is reported 'lf' whatever the "
        "file's convention; compare and commit through proven_profile / "
        "_bound_file_serialization so the recorded newline is inherited "
        "(TASK-34000.48)."
    )


#: The surviving sites, by module and enclosing function. Every one was read:
#: each compares a proven profile with the recorded one it was proven against,
#: or commits a proven profile. A new site of either shape lands here first.
_CENSUS = {
    "executor": {
        "compare": [
            "_admit_undo",
            "_binding_matches_current",
            "_classify_move_restore",
            "_classify_restore",
            "_file_holds_note",
            "_proven_post_write_baseline",
            "_proven_written_note_text",
            "_require_move_restore_owner",
            "_require_new_desired",
            "_require_restore_owner",
            "_require_undo_restored",
            "_validate_initial",
            "_verify_undo_opposite",
            "reconstruct_request",
        ],
        "commit": [
            "_advance",
            "_advance_move_restore",
            "_advance_new",
            "_advance_restore",
            # The undo commit restores the journal-decoded profile: no
            # observation is involved, so nothing to prove.
            "_advance_undo",
            "_commit_keep_both_binding",
            "_proven_post_write_baseline",
        ],
    },
    "runtime": {"compare": [], "commit": ["observe_root"]},
    "filesystem": {
        "compare": ["create", "move", "replace"],
        "commit": [],
    },
}


@pytest.mark.parametrize("module", sorted(_MODULES))
def test_the_surviving_sites_are_the_census(module: str) -> None:
    """A new profile comparison or binding commit of any shape fails here with its name."""

    found = _sites(_source(module))
    census = {
        kind: sorted({function for k, function, _l, _t, _v in found if k == kind})
        for kind in ("compare", "commit")
    }
    assert census == _CENSUS[module], (
        f"{module}: profile comparison/commit sites are now {census}. Read the "
        "new site: it must prove the observed profile against the recorded one "
        "it is compared with or committed over, then add it here."
    )


def test_the_rule_is_recorded_on_the_helper() -> None:
    source = _source("filesystem")
    helper = source[source.index("def proven_profile(") :]
    helper = " ".join(helper[: helper.index("\n\n\n")].split())
    assert "inherits the recorded" in helper
    assert "never" in helper and "mode" in helper and "utf8_bom" in helper


def _function(body: str) -> str:
    return (
        "def _probe(self, binding, note, request, file, reviewed, x, a, b):\n"
        + "".join("    " + line + "\n" for line in body.strip("\n").splitlines())
    )


_MUST_BE_FLAGGED = (
    pytest.param(
        "return _file_serialization(file) != binding.serialization", id="call-plain"
    ),
    pytest.param(
        "return binding.serialization != _file_serialization(file)", id="call-reversed"
    ),
    pytest.param(
        "return file.observation.serialization == reviewed.observation.serialization",
        id="attribute-both-sides",
    ),
    pytest.param(
        "return file.observation.serialization != request.candidate_serialization",
        id="attribute-vs-candidate",
    ),
    pytest.param(
        "p = file.observation.serialization\nreturn p == binding.serialization",
        id="hoisted",
    ),
    pytest.param(
        "p = _file_serialization(file) if x else binding.serialization\n"
        "return p == binding.serialization",
        id="hoisted-through-conditional",
    ),
    pytest.param(
        "return (_file_serialization(file), a) == (binding.serialization, b)",
        id="tuple",
    ),
    pytest.param(
        "return _file_serialization(file) in {binding.serialization}", id="in-set"
    ),
    pytest.param(
        "return a < _file_serialization(file).mode <= b", id="chained-field"
    ),
    pytest.param(
        "return self._store.get_binding(x).serialization != _file_serialization(file)",
        id="inline-store-lookup",
    ),
    pytest.param(
        "return (p := file.observation.serialization) != binding.serialization",
        id="walrus",
    ),
    pytest.param(
        "return replace(binding, serialization=_file_serialization(file))",
        id="commit-replace",
    ),
    pytest.param(
        "return NotesSyncBindingRecord(serialization=file.observation.serialization)",
        id="commit-record",
    ),
    pytest.param(
        "return BindingObservation(baseline_serialization=file.observation.serialization)",
        id="commit-observation-baseline",
    ),
    pytest.param(
        "p = _file_serialization(file)\nreturn replace(binding, serialization=p)",
        id="commit-hoisted",
    ),
)


@pytest.mark.parametrize("body", _MUST_BE_FLAGGED)
def test_negative_control_every_refactor_shape_is_flagged(body: str) -> None:
    flagged = _violations(_function(body))
    assert [function for _kind, function, _text in flagged] == ["_probe"], (
        f"not flagged: {body!r}"
    )


_MUST_STAY_PERMITTED = (
    pytest.param(
        "return _bound_file_serialization(file, binding.serialization) != binding.serialization",
        True,
        id="proven-vs-recorded",
    ),
    pytest.param(
        "p = proven_profile(file.observation.serialization, file.text, reviewed)\n"
        "return file.raw_bytes == _represented_bytes(note.content, p)",
        True,
        id="bytes-under-a-proven-profile",
    ),
    pytest.param(
        "return replace(binding, serialization=_bound_file_serialization(file, binding.serialization))",
        True,
        id="commit-proven",
    ),
    pytest.param(
        "return replace(binding, serialization=self._decoded_binding_serialization(x))",
        True,
        id="commit-journal-profile",
    ),
    pytest.param(
        "return NotesSyncBindingRecord(serialization=request.candidate_serialization)",
        True,
        id="commit-candidate-profile",
    ),
    pytest.param(
        "return binding.serialization != self._decoded_binding_serialization(x)",
        False,
        id="binding-vs-journal",
    ),
    pytest.param(
        "return file == request.file", False, id="whole-snapshot-equality"
    ),
    pytest.param(
        "return binding.serialization.newline != x.get('newline')",
        False,
        id="binding-field-vs-journal-field",
    ),
)


@pytest.mark.parametrize(("body", "in_census"), _MUST_STAY_PERMITTED)
def test_negative_control_the_permitted_shapes_are_not_flagged(
    body: str, in_census: bool
) -> None:
    found = _sites(_function(body))
    assert _violations(_function(body)) == []
    assert [function for _k, function, _l, _t, _v in found] == (
        ["_probe"] if in_census else []
    )


def test_negative_control_the_live_sources_are_what_the_scan_reads() -> None:
    """Append the base's own raw sites to the REAL file text: the scan goes red.

    Guards against the scan silently reading an empty or unrelated file and
    passing for the wrong reason. The appended bodies are the two shapes dev
    shipped: ``_file_holds_note``'s hoisted reviewed profile and
    ``_proven_post_write_baseline``'s raw compare-then-commit.
    """

    source = _source("executor")
    assert "class NotesSyncExecutor" in source
    mutated = source + (
        "\n\n"
        "def _reinstated_file_holds_note(file, note, reviewed):\n"
        "    profile = reviewed.observation.serialization\n"
        "    return (\n"
        "        file.observation.serialization == profile\n"
        "        and file.raw_bytes == _represented_bytes(note.content, profile)\n"
        "    )\n"
        "\n\n"
        "def _reinstated_post_write_baseline(file, binding):\n"
        "    if file.observation.serialization != binding.serialization:\n"
        "        return None\n"
        "    return replace(binding, serialization=file.observation.serialization)\n"
    )
    assert sorted((kind, function) for kind, function, _text in _violations(mutated)) == [
        ("commit", "_reinstated_post_write_baseline"),
        ("compare", "_reinstated_file_holds_note"),
        ("compare", "_reinstated_file_holds_note"),
        ("compare", "_reinstated_post_write_baseline"),
    ]
    filesystem = _source("filesystem") + (
        "\n\n"
        "def _reinstated_post_write_check(observed, payload, profile):\n"
        "    return observed.raw_bytes != payload or observed.observation.serialization != profile\n"
    )
    assert [(kind, function) for kind, function, _text in _violations(filesystem)] == [
        ("compare", "_reinstated_post_write_check"),
    ]

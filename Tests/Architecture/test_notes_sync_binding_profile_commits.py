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
mentions a raw profile and no proven one is a violation too.

Fix round 1 (review Important 1): a helper call is PROVEN only when its
recorded argument is a real record. ``None``, a raw profile of the SAME
observation (``_bound_file_serialization(file, _file_serialization(file))``)
or the call's own first operand make the call RAW -- proving an observation
against itself proves nothing (``observe_root``'s discovered-candidate site
is the one legitimate ``None`` and is allowlisted by name). A compare that
mixes a proven call with a raw operand is allowed only when that raw operand
IS the call's recorded argument (``_bound_file_serialization(x, R) == R``);
``_bound_file_serialization(file, binding.serialization) != request.file
.observation.serialization`` is flagged. And ``_file_holds_note`` must be
called with its ``recorded`` argument. The census test
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


def _is_proven_call(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and _call_name(node) in _PROVEN_CALLS


def _raw_base(node: ast.AST) -> ast.AST | None:
    """The observation a raw profile expression was read from."""

    if isinstance(node, ast.Call) and _call_name(node) in _RAW_CALLS:
        return node.args[0] if node.args else None
    if _is_raw_node(node):
        return node.value.value  # type: ignore[attr-defined]
    return None


def _recorded_argument(call: ast.Call) -> ast.AST | None:
    """The ``recorded`` argument of a helper call, however it is passed."""

    for keyword in call.keywords:
        if keyword.arg == "recorded":
            return keyword.value
    index = 1 if _call_name(call) == "_bound_file_serialization" else 2
    return call.args[index] if len(call.args) > index else None


def _first_operand(call: ast.Call) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg in {"snapshot", "observed"}:
            return keyword.value
    return call.args[0] if call.args else None


#: Functions in which a helper call with ``recorded=None`` is legitimate: a
#: discovered file has no recorded convention yet, so its own observation IS
#: the record (``notes_sync_runtime.observe_root``'s candidate site).
_UNRECORDED_ALLOWED = frozenset({"observe_root"})


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

    def __init__(self, function: ast.AST, name: str) -> None:
        self.name = name
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

    def resolve(self, node: ast.AST) -> ast.AST:
        """Follow a local name to the expression it is bound to."""

        seen: set[str] = set()
        while isinstance(node, ast.Name) and node.id in self.bound and node.id not in seen:
            seen.add(node.id)
            node = self.bound[node.id]
        return node

    def helper_proves(self, call: ast.Call) -> bool:
        """Whether a helper call proves against a REAL record.

        ``None`` (outside the allowlisted function), a raw profile of the
        call's own observation, or the call's own first operand prove
        nothing: the observation is compared with itself.
        """

        recorded = _recorded_argument(call)
        if recorded is None:
            return False
        if isinstance(recorded, ast.Constant) and recorded.value is None:
            return self.name in _UNRECORDED_ALLOWED
        resolved = self.resolve(recorded)
        if isinstance(resolved, ast.Constant) and resolved.value is None:
            return self.name in _UNRECORDED_ALLOWED
        first = _first_operand(call)
        if first is None:
            return False
        first_text = ast.unparse(self.resolve(first))
        first_base = _raw_base(self.resolve(first))
        if ast.unparse(resolved) == first_text:
            return False
        base = _raw_base(resolved)
        if base is not None:
            own = first_base if first_base is not None else self.resolve(first)
            if ast.unparse(base) == ast.unparse(own):
                return False
        return True

    def kinds(self, node: ast.AST, seen: set[str] | None = None) -> set[str]:
        """The profile kinds (``raw``/``proven``) the VALUE of ``node`` can be."""

        seen = set() if seen is None else seen
        if _is_raw_node(node):
            return {"raw"}
        if _is_proven_call(node):
            return {"proven"} if self.helper_proves(node) else {"raw"}
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

    def raw_outside_helpers(self, node: ast.AST) -> bool:
        """Whether ``node`` reads a raw profile anywhere a helper is not proving it.

        A proving call is opaque: the raw observation and the record it is
        proven against are its arguments, not operands of the comparison.
        """

        if _is_proven_call(node) and self.helper_proves(node):
            return False
        if "raw" in self.kinds(node):
            return True
        return any(self.raw_outside_helpers(child) for child in ast.iter_child_nodes(node))

    def proven_calls(self, node: ast.AST) -> list[ast.Call]:
        """Every proving helper call reachable from ``node``, through names."""

        calls: list[ast.Call] = []
        for sub in ast.walk(node):
            candidate = self.resolve(sub) if isinstance(sub, ast.Name) else sub
            if _is_proven_call(candidate) and self.helper_proves(candidate):
                calls.append(candidate)
        return calls

    def compare_violates(self, operands: list[ast.AST]) -> bool:
        """A raw operand is tolerated only as a proven call's own record."""

        raw_operands = [operand for operand in operands if self.raw_outside_helpers(operand)]
        if not raw_operands:
            return False
        calls = [call for operand in operands for call in self.proven_calls(operand)]
        if not calls:
            return True
        records: set[str] = set()
        for call in calls:
            recorded = _recorded_argument(call)
            if recorded is not None:
                records.add(ast.unparse(recorded))
                records.add(ast.unparse(self.resolve(recorded)))
        return any(
            ast.unparse(operand) not in records
            and ast.unparse(self.resolve(operand)) not in records
            for operand in raw_operands
        )


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
        return name, scopes.setdefault(owner, _Scope(owner, name))

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
                    (
                        "compare",
                        enclosing,
                        node.lineno,
                        ast.unparse(node),
                        scope.compare_violates(operands),
                    )
                )
        elif isinstance(node, ast.Call) and _call_name(node) == "_file_holds_note":
            enclosing, _scope = owner_of(node)
            carries_record = len(node.args) >= 4 or any(
                keyword.arg == "recorded" for keyword in node.keywords
            )
            found.append(
                ("call", enclosing, node.lineno, ast.unparse(node), not carries_record)
            )
        elif isinstance(node, ast.Call) and _call_name(node) in _COMMIT_CALLS:
            enclosing, scope = owner_of(node)
            for keyword in node.keywords:
                if keyword.arg not in _COMMIT_KEYWORDS:
                    continue
                raw = scope.raw_outside_helpers(keyword.value)
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
        "call": ["_classify", "_classify_restore"],
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
    "runtime": {"call": [], "compare": [], "commit": ["observe_root"]},
    "filesystem": {
        "call": [],
        "compare": ["_requested_write_profile", "create", "move", "replace"],
        "commit": [],
    },
}


@pytest.mark.parametrize("module", sorted(_MODULES))
def test_the_surviving_sites_are_the_census(module: str) -> None:
    """A new profile comparison or binding commit of any shape fails here with its name."""

    found = _sites(_source(module))
    census = {
        kind: sorted({function for k, function, _l, _t, _v in found if k == kind})
        for kind in ("call", "compare", "commit")
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
    # Fix round 1 (review Important 1): the six probed shapes.
    pytest.param(
        "return binding.serialization != _bound_file_serialization(file, _file_serialization(file))",
        id="prove-against-itself",
    ),
    pytest.param(
        "return proven_profile(file.observation.serialization, file.text, None) != binding.serialization",
        id="proven-against-nothing-in-a-compare",
    ),
    pytest.param(
        "return replace(binding, serialization=_bound_file_serialization(file, None))",
        id="commit-proven-against-nothing",
    ),
    pytest.param(
        "return _bound_file_serialization(file, binding.serialization) != request.file.observation.serialization",
        id="proven-current-vs-a-different-raw-reviewed",
    ),
    pytest.param(
        "return self._file_holds_note(file, note, reviewed)",
        id="holds-note-without-recorded",
    ),
    pytest.param(
        "return binding.serialization != file.observation.serialization",
        id="binding-vs-observation-attribute",
    ),
    pytest.param(
        "return proven_profile(file.observation.serialization, file.text, file.observation.serialization) != binding.serialization",
        id="prove-against-own-first-operand",
    ),
    pytest.param(
        "r = _file_serialization(file)\nreturn _bound_file_serialization(file, r) == r",
        id="prove-against-itself-hoisted",
    ),
    pytest.param(
        "r = None\nreturn _bound_file_serialization(file, r) == binding.serialization",
        id="proven-against-hoisted-none",
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
        "reviewed = _file_serialization(request.file)\n"
        "return _bound_file_serialization(file, reviewed) == reviewed",
        True,
        id="proven-against-the-reviewed-observation",
    ),
    pytest.param(
        "return self._file_holds_note(file, note, reviewed, binding.serialization)",
        True,
        id="holds-note-with-recorded",
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

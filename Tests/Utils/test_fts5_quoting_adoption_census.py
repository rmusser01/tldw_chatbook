"""TASK-19558: the FTS5 quoting primitive cannot be re-spelled, and a
computed-but-unbound "safe" term cannot come back.

This repo already owned `Utils/fts5_match_forms.quote_fts5_token` -- the ONE
correct FTS5 string-literal escape -- and, at this task's branch base, had
**six** other spellings of the same idea scattered across the DB and UI
layers. Four of them were correct. Two omitted the embedded-quote doubling
entirely, so any search containing a `"` either raised
`OperationalError('unterminated string')` or escaped the literal into a live
column filter. Worse, three `ChaChaNotes_DB` search methods computed a
`safe_search_term` and then bound the RAW one: protection that read as
protection in code review and reached no query at all.

Fixing the six sites is not the durable outcome -- the next search method is.
These two censuses are that outcome.

**Census 1 (re-spelling).** Doubling a `"` into `""` has exactly one purpose
in this codebase's data layer: escaping a double quote inside a quoted
string literal. So `<expr>.replace('"', '""')` anywhere outside the two
modules that own an escape IS a hand-rolled re-spelling, and this census
fails on it. There is one genuine non-FTS5 owner, `DB/sql_validation.py`'s
`escape_identifier`, which escapes a SQL *identifier* (`"table name"` in
DDL) -- a different language layer with the same doubling rule -- and it is
allowlisted by exact (module, function) name rather than by module, so a
second, FTS5-shaped helper cannot be smuggled into that file.

**Census 2 (dead store).** A local whose name marks it as the sanitized form
of user input (`safe_*`, `quoted_*`, `escaped_*`) and whose every READ is
inside a logging call is, by construction, not protecting anything: the
value reaches the diagnostics and nothing else. That is the exact shape of
the three defects this task fixed, and running this census against the base
revision of `ChaChaNotes_DB.py` independently rediscovers all three and
nothing else (`test_dead_store_census_rediscovers_the_three_base_defects`).

**Why AST, not regex over source text.** Following
`Tests/Utils/test_egress_adoption_census.py` (PR #1967's review): a regex
over raw source is bypassable in both directions -- a docstring *mentioning*
the primitive launders an unguarded module (false green), and a comment
*containing* the offending literal flags a clean one (false red). Comments
and docstrings produce no `Call`/`Name` nodes, so both bypasses are
structurally closed here; both directions are proven below.

**Census 3 (the module-level net).** A module that binds a value to a
`... MATCH ?` query must import from `Utils/fts5_match_forms`. Seven modules
do so today and all seven import it. This is the only one of the three that
can catch the *broken* spelling in a new file -- see the limits below for why
that matters more than it sounds.

**What these cannot express, stated rather than left to be rediscovered.**

*Census 1 does not catch the spelling that actually caused this task.* It
fires on `.replace('"', '""')` -- the CORRECT escape, hand-rolled. The two
sites that were genuinely broken wrote `f'"{term}"'` with no doubling at
all, and that expression contains nothing for this census to match. So
census 1 prevents a seventh copy of the RIGHT rule, not a third copy of the
WRONG one. A repo-wide detector for the bare `f'"{x}"'` shape was tried and
rejected on measurement: it is indistinguishable from SQL identifier quoting
(`f'DROP TRIGGER IF EXISTS "{name}"'`) and from ordinary UI copy
(`f'Imported "{skill_name}" ...'`), producing four false positives and zero
true ones on the current tree. Census 3 is the structural answer available:
a NEW module that hand-rolls either spelling reds because it binds `MATCH ?`
without importing the primitive. A new seam added *inside* one of the seven
modules that already import it is NOT covered -- for those, the behavioural
sweep in `Tests/DB/test_fts5_quoting_search_seams.py` is the net, and it only
covers seams enumerated in it.

*Census 2 is defeated by this repo's own house logging style.* The
logging-root walk resolves `logger.error(...)` and `logging.error(...)`, but
`logger.opt(exception=True).error(...)` -- which is used widely here -- puts
an `ast.Call` (`logger.opt(...)`) where the walk expects a `Name`, so the
call is not recognised as logging and a dead store read only inside one
would be reported as "used". The walk is deliberately left simple rather
than made to chase arbitrary attribute chains, but the consequence is real:
census 2 catches the shape as it was written in `ChaChaNotes_DB`, not every
shape it could take.

*Both are single-file and syntactic.* Census 1 cannot see an escape built by
string concatenation in a loop, by `str.translate`, or through a helper in
another module; census 2 cannot see a dead store whose reads are routed
through a local alias or an f-string assigned to another variable first;
census 3 sees imports, not use. None of them can prove the bound value is the
*right* one -- only that the sanitized one is not obviously discarded. The
behavioural tests in `Tests/DB/test_fts5_quoting_search_seams.py` are what
prove the actual matching semantics.
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "tldw_chatbook"

#: The ONE module allowed to implement the FTS5 string-literal escape.
FTS5_PRIMITIVE = "tldw_chatbook/Utils/fts5_match_forms.py"

#: (module, function) pairs allowed to double a `"` for a NON-FTS5 reason.
#: Keyed on the function too, so an FTS5 helper cannot be parked in the same
#: module and inherit the exemption.
QUOTE_DOUBLING_EXEMPTIONS = frozenset(
    {
        # SQL identifier quoting: `"my table"` in DDL/DML. Same doubling
        # rule, different language layer, and it is not interchangeable with
        # the FTS5 escape (its output is an identifier, not a MATCH term).
        ("tldw_chatbook/DB/sql_validation.py", "escape_identifier"),
    }
)

#: Locals whose NAME claims they are the sanitized form of user input.
_SANITIZED_LOCAL = re.compile(r"^(safe|quoted|escaped)_")

#: Call roots treated as "this is a logging call".
_LOGGING_ROOTS = frozenset({"logger", "logging", "log"})


def _python_sources() -> list[Path]:
    return sorted(PACKAGE_ROOT.rglob("*.py"))


def _rel(path: Path) -> str:
    return str(path.relative_to(PACKAGE_ROOT.parent))


def _is_quote_doubling_call(node: ast.AST) -> bool:
    """True for `<expr>.replace('"', '""')` -- the escape, spelled out."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "replace"
        and len(node.args) == 2
        and all(
            isinstance(arg, ast.Constant) and isinstance(arg.value, str)
            for arg in node.args
        )
        and node.args[0].value == '"'
        and node.args[1].value == '""'
    )


def _enclosing_function_name(tree: ast.AST, target: ast.AST) -> str:
    """Innermost def containing `target`, or "<module>"."""
    best = "<module>"
    best_lineno = -1
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        end = getattr(node, "end_lineno", None) or node.lineno
        if node.lineno <= target.lineno <= end and node.lineno > best_lineno:
            best, best_lineno = node.name, node.lineno
    return best


def quote_doubling_sites(source: str, relative_path: str) -> list[tuple[str, str, int]]:
    """Every `.replace('"', '""')` in `source`, as (path, function, line)."""
    tree = ast.parse(source)
    return [
        (relative_path, _enclosing_function_name(tree, node), node.lineno)
        for node in ast.walk(tree)
        if _is_quote_doubling_call(node)
    ]


def _logged_node_ids(function: ast.AST) -> set[int]:
    """ids of every node inside a `logger.*(...)`/`logging.*(...)` call."""
    logged: set[int] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        root = node.func
        while isinstance(root, ast.Attribute):
            root = root.value
        if isinstance(root, ast.Name) and root.id in _LOGGING_ROOTS:
            for descendant in ast.walk(node):
                logged.add(id(descendant))
    return logged


def dead_store_sites(source: str, relative_path: str) -> list[tuple[str, str, str, str]]:
    """Sanitized-looking locals that are never read outside a log call.

    Returns (path, function, variable, reason) rows.
    """
    tree = ast.parse(source)
    findings: list[tuple[str, str, str, str]] = []
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        assigned: set[str] = set()
        for node in ast.walk(function):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and _SANITIZED_LOCAL.match(
                        target.id
                    ):
                        assigned.add(target.id)
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and _SANITIZED_LOCAL.match(node.target.id)
            ):
                assigned.add(node.target.id)
        if not assigned:
            continue
        logged = _logged_node_ids(function)
        for name in sorted(assigned):
            reads = [
                node
                for node in ast.walk(function)
                if isinstance(node, ast.Name)
                and node.id == name
                and isinstance(node.ctx, ast.Load)
            ]
            if not reads:
                findings.append((relative_path, function.name, name, "never read"))
            elif all(id(node) in logged for node in reads):
                findings.append(
                    (relative_path, function.name, name, "read only inside logging")
                )
    return findings


# ---------------------------------------------------------------------------
# Census 1: nobody re-spells the escape.
# ---------------------------------------------------------------------------


def test_no_module_outside_the_primitive_hand_rolls_the_fts5_escape() -> None:
    offenders: list[tuple[str, str, int]] = []
    for path in _python_sources():
        relative = _rel(path)
        try:
            sites = quote_doubling_sites(path.read_text(encoding="utf-8"), relative)
        except SyntaxError:  # pragma: no cover - repo is expected to parse
            continue
        for site in sites:
            if relative == FTS5_PRIMITIVE:
                continue
            if (site[0], site[1]) in QUOTE_DOUBLING_EXEMPTIONS:
                continue
            offenders.append(site)
    assert offenders == [], (
        "These sites double a '\"' outside "
        f"{FTS5_PRIMITIVE}. That is the FTS5 string-literal escape; import "
        "quote_fts5_token/quote_fts5_phrase/quote_fts5_prefix instead of "
        f"writing a seventh spelling of it: {offenders}"
    )


def test_the_primitive_actually_doubles_embedded_quotes() -> None:
    """The census is only worth anything if the one survivor is correct."""
    from tldw_chatbook.Utils.fts5_match_forms import (
        quote_fts5_phrase,
        quote_fts5_prefix,
        quote_fts5_token,
    )

    assert quote_fts5_token('foo"bar') == '"foo""bar"'
    assert quote_fts5_phrase('alpha" OR title:"beta') == '"alpha"" OR title:""beta"'
    assert quote_fts5_prefix('foo"bar') == '"foo""bar"*'
    # Phrase and token are the same escape, deliberately -- not two
    # implementations that happen to agree today.
    assert quote_fts5_phrase is quote_fts5_token


def test_census_flags_a_reintroduced_hand_rolled_escape() -> None:
    """Bite-proof: the exact shape this task removed fails the census."""
    reintroduced = (
        "def search(term):\n"
        "    escaped = term.replace('\"', '\"\"')\n"
        "    return f'\"{escaped}\"*'\n"
    )
    sites = quote_doubling_sites(reintroduced, "tldw_chatbook/DB/Fake_DB.py")
    assert sites == [("tldw_chatbook/DB/Fake_DB.py", "search", 2)]


def test_a_comment_or_docstring_mentioning_the_escape_is_not_flagged() -> None:
    """False-red direction: prose about the escape is not the escape."""
    prose = (
        'def search(term):\n'
        '    """Historically this called term.replace(\'"\', \'""\') inline."""\n'
        "    # do not write term.replace('\"', '\"\"') here\n"
        "    return quote_fts5_phrase(term)\n"
    )
    assert quote_doubling_sites(prose, "tldw_chatbook/DB/Fake_DB.py") == []


def test_the_sql_identifier_exemption_is_scoped_to_its_own_function() -> None:
    """False-green direction: the exemption does not cover its whole module.

    An FTS5 helper parked inside `sql_validation.py` would be a new
    spelling, so the allowlist is keyed on (module, function).
    """
    smuggled = (
        "def escape_identifier(identifier):\n"
        "    return '\"' + identifier.replace('\"', '\"\"') + '\"'\n"
        "\n"
        "def quote_fts_term(term):\n"
        "    return '\"' + term.replace('\"', '\"\"') + '\"'\n"
    )
    sites = quote_doubling_sites(smuggled, "tldw_chatbook/DB/sql_validation.py")
    unexempt = [
        site for site in sites if (site[0], site[1]) not in QUOTE_DOUBLING_EXEMPTIONS
    ]
    assert unexempt == [("tldw_chatbook/DB/sql_validation.py", "quote_fts_term", 5)]


# ---------------------------------------------------------------------------
# Census 2: no `safe_*` dead stores.
# ---------------------------------------------------------------------------


def test_no_sanitized_local_is_computed_and_then_only_logged() -> None:
    offenders: list[tuple[str, str, str, str]] = []
    for path in _python_sources():
        try:
            offenders.extend(
                dead_store_sites(path.read_text(encoding="utf-8"), _rel(path))
            )
        except SyntaxError:  # pragma: no cover
            continue
    assert offenders == [], (
        "These locals name themselves as the sanitized form of user input "
        "but are never read outside a logging call -- i.e. the sanitized "
        "value reaches the diagnostics and nothing else, while the raw one "
        f"reaches the query: {offenders}"
    )


def test_dead_store_census_flags_the_shape_it_was_built_for() -> None:
    """Bite-proof, on the literal shape found in `search_character_cards`."""
    reintroduced = (
        "def search_character_cards(self, search_term, limit=10):\n"
        "    safe_search_term = quote_fts5_phrase(search_term)\n"
        "    try:\n"
        "        return self.execute_query(QUERY, (search_term, limit))\n"
        "    except Exception as e:\n"
        "        logger.error(f\"Error searching for '{safe_search_term}': {e}\")\n"
        "        raise\n"
    )
    assert dead_store_sites(reintroduced, "tldw_chatbook/DB/Fake_DB.py") == [
        (
            "tldw_chatbook/DB/Fake_DB.py",
            "search_character_cards",
            "safe_search_term",
            "read only inside logging",
        )
    ]


def test_dead_store_census_does_not_flag_a_term_that_reaches_the_query() -> None:
    """False-red direction: logging it TOO is fine; only logging is not."""
    fixed = (
        "def search_character_cards(self, search_term, limit=10):\n"
        "    safe_search_term = quote_fts5_phrase(search_term)\n"
        "    try:\n"
        "        return self.execute_query(QUERY, (safe_search_term, limit))\n"
        "    except Exception as e:\n"
        "        logger.error(f\"Error searching for '{safe_search_term}': {e}\")\n"
        "        raise\n"
    )
    assert dead_store_sites(fixed, "tldw_chatbook/DB/Fake_DB.py") == []


#: The commit this task branched from, and the last one containing the three
#: dead stores. Pinned rather than resolved through `git merge-base HEAD
#: origin/dev`: merge-base MOVES on every rebase, and after the fix lands on
#: dev it resolves to a commit where the defects are already gone -- the
#: check would then fail for the wrong reason, and the obvious "repair" is
#: to delete it. A historical commit id is a fixed fact; this one is a
#: reachable ancestor of every branch that will ever carry this test.
DEFECT_BASE_REVISION = "72a82bc56"


def test_dead_store_census_rediscovers_the_three_base_defects() -> None:
    """The census finds the real defects in the real pre-fix file.

    Read from git rather than restated inline, so this is a claim about the
    shipped code that was actually there, not about a paraphrase of it.

    **Fails closed.** An earlier version `pytest.skip`ped when the object
    could not be read, which meant a shallow clone or a checkout without the
    `origin/dev` ref turned the one test that proves the census DETECTS
    anything into a silent pass -- the same "green because it did not run"
    shape this programme keeps finding. If git cannot answer, that is a
    failure of this check, not an excuse for it.
    """
    repo_root = PACKAGE_ROOT.parent
    try:
        source = subprocess.run(
            [
                "git",
                "show",
                f"{DEFECT_BASE_REVISION}:tldw_chatbook/DB/ChaChaNotes_DB.py",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        raise AssertionError(
            "cannot read "
            f"{DEFECT_BASE_REVISION}:tldw_chatbook/DB/ChaChaNotes_DB.py, so "
            "the dead-store census is UNPROVEN -- unshallow the clone and "
            f"re-run rather than treating this as skippable: {exc}"
        ) from exc

    findings = dead_store_sites(source, "tldw_chatbook/DB/ChaChaNotes_DB.py")
    assert [(function, name) for _path, function, name, _why in findings] == [
        ("search_character_cards", "safe_search_term"),
        ("search_conversations_by_title", "safe_search_term"),
        ("search_messages_by_content", "safe_search_term"),
    ], findings
    assert all(why == "read only inside logging" for *_rest, why in findings)


# ---------------------------------------------------------------------------
# Census 3: a module that binds to `MATCH ?` imports the primitive.
# ---------------------------------------------------------------------------

_MATCH_PARAM = re.compile(r"\bMATCH\s*\?", re.IGNORECASE)


def _docstring_node_ids(tree: ast.AST) -> set[int]:
    """ids of every docstring Constant in `tree`.

    Docstrings are the one kind of prose that DOES produce an `ast.Constant`
    -- unlike comments, which produce no node at all -- so a census that
    reads string constants has to exclude them explicitly or a module whose
    docstring merely explains `MATCH ?` looks like a search seam.
    """
    ids: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            ids.add(id(first.value))
    return ids


def binds_fts_match(source: str) -> bool:
    """Whether a module contains a `... MATCH ?` SQL string.

    Read off string CONSTANTS in the AST rather than off the raw text, with
    docstrings excluded (see `_docstring_node_ids`): a comment mentioning
    `MATCH ?` produces no node at all, and a docstring mentioning it is
    filtered, so neither can make a module that never queries FTS5 look like
    a search seam.
    """
    tree = ast.parse(source)
    docstrings = _docstring_node_ids(tree)
    return any(
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
        and _MATCH_PARAM.search(node.value)
        for node in ast.walk(tree)
    )


def imports_the_primitive(source: str) -> bool:
    tree = ast.parse(source)
    return any(
        isinstance(node, ast.ImportFrom)
        and (node.module or "").endswith("fts5_match_forms")
        for node in ast.walk(tree)
    )


def test_every_module_that_binds_a_match_parameter_imports_the_primitive() -> None:
    """The only one of the three censuses that a NEW file cannot slip past.

    Censuses 1 and 2 both look for a specific wrong SHAPE. This one looks
    for the situation instead: if a module runs an FTS5 MATCH with a bound
    parameter, it is building a MATCH expression, and there is exactly one
    place in this repo allowed to build one.
    """
    offenders: list[str] = []
    for path in _python_sources():
        source = path.read_text(encoding="utf-8", errors="replace")
        try:
            if binds_fts_match(source) and not imports_the_primitive(source):
                offenders.append(_rel(path))
        except SyntaxError:  # pragma: no cover
            continue
    assert offenders == [], (
        "These modules bind a parameter to an FTS5 MATCH without importing "
        f"Utils/fts5_match_forms: {offenders} -- build the expression with "
        "build_and_match_query / build_phrase_match_query rather than by hand."
    )


def test_census_three_flags_a_new_seam_that_hand_rolls_either_spelling() -> None:
    """Bite-proof, on BOTH spellings -- including the broken one census 1
    cannot see."""
    correct_but_hand_rolled = (
        "def search(db, term):\n"
        "    q = '\"' + term.replace('\"', '\"\"') + '\"'\n"
        "    return db.execute('SELECT 1 FROM t_fts WHERE t_fts MATCH ?', (q,))\n"
    )
    broken = (
        "def search(db, term):\n"
        "    q = f'\"{term}\"'\n"
        "    return db.execute('SELECT 1 FROM t_fts WHERE t_fts MATCH ?', (q,))\n"
    )
    for source in (correct_but_hand_rolled, broken):
        assert binds_fts_match(source)
        assert not imports_the_primitive(source)
    # And the point of the pair: census 1 sees only the first one.
    assert quote_doubling_sites(correct_but_hand_rolled, "x.py")
    assert quote_doubling_sites(broken, "x.py") == []


def test_census_three_is_not_triggered_by_prose_about_match() -> None:
    """False-red direction: a docstring naming `MATCH ?` is not a seam."""
    prose = (
        'def helper():\n'
        '    """Historically this built the `t_fts MATCH ?` expression."""\n'
        "    return None\n"
    )
    assert not binds_fts_match(prose)

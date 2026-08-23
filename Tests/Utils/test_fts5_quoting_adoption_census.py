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

**What this cannot express, stated rather than left to be rediscovered.**
Both censuses are single-file and syntactic. Census 1 cannot see an escape
built by string concatenation in a loop, by `str.translate`, or through a
helper in another module; census 2 cannot see a dead store whose reads are
routed through a local alias or an f-string assigned to another variable
first. Neither can prove the bound value is the *right* one -- only that the
sanitized one is not obviously discarded. The behavioural tests in
`Tests/DB/test_fts5_quoting_search_seams.py` are what prove the actual
matching semantics; these two only close the "silently re-spelled / silently
discarded" routes back in.
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import pytest

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


def test_dead_store_census_rediscovers_the_three_base_defects() -> None:
    """The census finds the real defects in the real pre-fix file.

    Read from git rather than restated inline, so this is a claim about the
    shipped code that was actually there, not about a paraphrase of it. The
    base revision is this branch's merge base with `origin/dev`; if the
    object is unavailable (a shallow clone, a pruned reflog) the test skips
    rather than passing vacuously.
    """
    repo_root = PACKAGE_ROOT.parent
    base = "72a82bc56"
    try:
        source = subprocess.run(
            ["git", "show", f"{base}:tldw_chatbook/DB/ChaChaNotes_DB.py"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        pytest.skip(f"base revision {base} unavailable: {exc}")

    findings = dead_store_sites(source, "tldw_chatbook/DB/ChaChaNotes_DB.py")
    assert [(function, name) for _path, function, name, _why in findings] == [
        ("search_character_cards", "safe_search_term"),
        ("search_conversations_by_title", "safe_search_term"),
        ("search_messages_by_content", "safe_search_term"),
    ], findings
    assert all(why == "read only inside logging" for *_rest, why in findings)

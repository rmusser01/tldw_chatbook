# test_default_timeout_session_guard.py
# Description: Guard against bare `requests.Session()` / timeout-less `requests` calls (task-19830)
"""
task-19830: a plain ``requests`` call with no ``timeout=`` waits forever on a
half-open connection. On the LLM paths that means a chat or summarization
request that never returns and cannot be cancelled -- several also run on the
event loop, so the whole TUI stops. ``requests`` has no per-session default
timeout, so every call site has to remember one individually.

``tldw_chatbook.Utils.egress.create_default_session()`` fixes this: it returns
a ``requests.Session`` subclass (``DefaultTimeoutSession``) whose ``request()``
fills in a config-driven ``(connect, read)`` timeout only when the caller
omitted one -- an explicit ``timeout=`` always wins.

This guard walks the AST of every file under ``tldw_chatbook/`` that imports
``requests`` (module-level or a lazy in-function import -- several files here
do that deliberately, so a plain substring/import-at-top check would miss
them) and flags:

* every ``requests.Session()`` construction (should be
  ``create_default_session()`` instead), and
* every ``get``/``post``/``put``/``delete``/``patch``/``request`` call made
  either directly on the ``requests`` module or on a variable/``self``
  attribute the file itself assigned from ``requests.Session()``, that
  supplies neither a ``timeout=`` keyword nor (for ``.request()``, whose
  positional signature has room for one) a positional argument in that slot.

``LLM_Calls/`` -- the user-facing hazard this task targets -- must be
completely clean: any finding there fails the guard. Every other file that
still has findings is named explicitly in ``EXEMPT_FILES`` below (task-19830
converts ``LLM_Calls/`` only; the rest is deliberately deferred, one
subsystem at a time, to keep each conversion independently reviewable). A
file that no longer has any findings, or that has been deleted, is a STALE
exemption and fails the guard too -- the whole point of naming files here is
that the remaining work stays visible, not implied.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "tldw_chatbook"

#: The six ``requests`` methods this guard checks for a timeout. `head`/
#: `options` are deliberately excluded -- nothing in this codebase uses them
#: for a live network call with a body large enough to hang meaningfully, and
#: the original AST sweep that sized this task didn't count them either.
_REQUEST_METHODS = frozenset({"get", "post", "put", "delete", "patch", "request"})

#: ``requests.Session.request``'s full positional signature is ``method, url,
#: params, data, headers, cookies, files, auth, timeout, ...`` -- ``timeout``
#: is the 9th positional argument, index 8 in an AST ``Call`` node's ``args``
#: (which, unlike a hand-written override's ``*args``, still includes
#: ``method``/``url`` as elements 0 and 1). A ``.request()`` call with 9 or
#: more positional arguments therefore supplied an explicit timeout.
#: ``get``/``post``/``put``/``delete``/``patch`` only take ``url``
#: positionally (everything else is ``**kwargs``), so no entry is needed for
#: them.
_REQUEST_POSITIONAL_TIMEOUT_INDEX = {"request": 8}

#: Files that still have a bare ``requests.Session()`` and/or a timeout-less
#: request call, deliberately not converted in this pass (task-19830 scopes
#: the conversion to ``LLM_Calls/`` only -- see the module docstring). Keep
#: this list exactly in sync with reality: an entry with zero findings, or
#: naming a file that no longer exists, fails the guard (see
#: ``test_exempt_files_are_current`` below) precisely so a fixed or deleted
#: file can't quietly keep "remaining work" on the books.
EXEMPT_FILES: frozenset[str] = frozenset(
    {
        # Embeddings, Character_Chat, Local_Inference, TTS and Web_Scraping
        # are real hang-forever hazards too (same audit that sized this
        # task), but converting five unrelated subsystems in the same diff
        # as the LLM_Calls fix is a regression risk this task isn't taking.
        "Embeddings/Embeddings_Lib.py",
        "Local_Inference/ollama_model_mgmt.py",
        # (task-19560 shipped the Kokoro download timeouts while this branch
        # was in review -- `TTS/backends/kokoro.py` used to be exempted here
        # and is now clean, so the entry was removed. This guard's
        # stale-exemption check is what caught that, which is the point of
        # naming files individually instead of skipping a whole directory.)
        "Web_Scraping/Confluence/confluence_auth.py",
        "Web_Scraping/WebSearch_APIs.py",
        # Utils/egress.py's OWN internals (`guarded_fetch_requests`'s
        # `session or requests.Session()` default) are explicitly out of
        # scope -- this is the module that HOUSES the factory being added
        # here, and its existing `timeout: float = 30.0` parameter is a
        # deliberate, already-reviewed default, not an oversight.
        "Utils/egress.py",
    }
)


def _qualified(node: ast.AST) -> str:
    """Render a call target as a dotted string (``requests.post``, ``x.y.get``)."""
    if isinstance(node, ast.Attribute):
        base = _qualified(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


@dataclass(frozen=True)
class Finding:
    path: str  # relative to PACKAGE_ROOT's parent (e.g. "LLM_Calls/foo.py")
    lineno: int
    kind: str  # "bare-session" | "timeoutless-call"
    detail: str

    def __str__(self) -> str:  # pragma: no cover - human-readable only
        return f"{self.path}:{self.lineno} [{self.kind}] {self.detail}"


def _requests_import_aliases(tree: ast.AST) -> set[str]:
    """Local binding name(s) for the ``requests`` module, found anywhere in
    the file -- module scope OR inside a function. Several files here do a
    lazy ``import requests`` on purpose (keeps it out of the hot import path
    for code that doesn't need it), so this must not be a module-scope-only
    check.
    """
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "requests":
                    aliases.add(alias.asname or alias.name)
    return aliases


def _scan_source(source: str, *, relative_path: str) -> list[Finding]:
    """Return every bare-Session/timeout-less finding in one file's source."""
    tree = ast.parse(source, filename=relative_path)
    aliases = _requests_import_aliases(tree)
    if not aliases:
        return []

    session_ctor_names = {f"{a}.Session" for a in aliases} | {
        f"{a}.sessions.Session" for a in aliases
    }

    def is_session_ctor(node: ast.AST) -> bool:
        return isinstance(node, ast.Call) and _qualified(node.func) in session_ctor_names

    # Pass 1: find every name/self-attribute this file binds to a fresh
    # `requests.Session()` -- assignment, `with ... as`, or an annotated
    # parameter -- so pass 2 can recognise `session.post(...)` as a
    # requests call even though `session` isn't the module itself.
    session_vars: set[str] = set()
    self_session_attrs: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and is_session_ctor(node.value):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    session_vars.add(tgt.id)
                elif (
                    isinstance(tgt, ast.Attribute)
                    and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == "self"
                ):
                    self_session_attrs.add(tgt.attr)
        elif (
            isinstance(node, ast.AnnAssign)
            and node.value is not None
            and is_session_ctor(node.value)
        ):
            if isinstance(node.target, ast.Name):
                session_vars.add(node.target.id)
            elif (
                isinstance(node.target, ast.Attribute)
                and isinstance(node.target.value, ast.Name)
                and node.target.value.id == "self"
            ):
                self_session_attrs.add(node.target.attr)
        elif isinstance(node, ast.With):
            for item in node.items:
                if is_session_ctor(item.context_expr) and isinstance(
                    item.optional_vars, ast.Name
                ):
                    session_vars.add(item.optional_vars.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in list(node.args.args) + list(node.args.kwonlyargs):
                if arg.annotation is not None and _qualified(
                    arg.annotation
                ) in {f"{a}.Session" for a in aliases}:
                    session_vars.add(arg.arg)

    def has_timeout(call: ast.Call, leaf: str) -> bool:
        # A `**kwargs` expansion deliberately does NOT count. It MIGHT carry
        # a timeout, and the AST cannot tell -- but a guard that assumes the
        # safe case whenever it can't see is a guard that passes on exactly
        # the call sites hardest to eyeball. `requests.post(url, **opts)` is
        # the shape most likely to hide a missing timeout, not least likely.
        # A site that really does forward one through `**kwargs` states it
        # explicitly (`timeout=opts.get("timeout", ...)`) and stays green.
        for kw in call.keywords:
            if kw.arg == "timeout":
                return True
        idx = _REQUEST_POSITIONAL_TIMEOUT_INDEX.get(leaf)
        return idx is not None and len(call.args) > idx

    findings: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if is_session_ctor(node):
            findings.append(
                Finding(relative_path, node.lineno, "bare-session", _qualified(node.func))
            )
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        leaf = node.func.attr
        if leaf not in _REQUEST_METHODS:
            continue
        base = node.func.value
        is_requests_target = False
        if isinstance(base, ast.Name):
            if base.id in aliases or base.id in session_vars:
                is_requests_target = True
        elif (
            isinstance(base, ast.Attribute)
            and isinstance(base.value, ast.Name)
            and base.value.id == "self"
            and base.attr in self_session_attrs
        ):
            is_requests_target = True
        if is_requests_target and not has_timeout(node, leaf):
            findings.append(
                Finding(
                    relative_path,
                    node.lineno,
                    "timeoutless-call",
                    f"{_qualified(node.func)}(",
                )
            )
    return findings


@lru_cache(maxsize=1)
def _scan_package() -> dict[str, list[Finding]]:
    """Every finding under ``tldw_chatbook/``, keyed by path relative to it.

    Cached: three tests in this module each need the whole picture, and
    re-parsing every file in the package three times cost ~9s of a suite
    run for three identical answers (14.2s -> 5.0s for this module). Safe
    because nothing here mutates the tree between tests -- the package is
    read-only input to a pure function of its own source.

    Returns:
        ``{relative_path: [Finding, ...]}`` for files with at least one
        finding. Callers must treat the value as read-only; it is shared.
    """
    by_file: dict[str, list[Finding]] = {}
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        source = path.read_text(encoding="utf-8")
        findings = _scan_source(source, relative_path=relative)
        if findings:
            by_file[relative] = findings
    return by_file


def test_llm_calls_package_has_no_bare_session_or_timeoutless_call() -> None:
    """AC #2: LLM_Calls/ is the user-facing hazard this task closes -- it must
    be completely clean, with zero exemptions."""
    by_file = _scan_package()
    offenders = {
        path: findings for path, findings in by_file.items() if path.startswith("LLM_Calls/")
    }
    assert offenders == {}, "\n".join(
        str(f) for findings in offenders.values() for f in findings
    )


def test_covered_area_has_no_unexempted_bare_session_or_timeoutless_call() -> None:
    """AC #3: a NEW bare Session()/timeout-less call anywhere in the package,
    outside the explicit exemption list, fails this guard and names the
    file+line -- this is what keeps the LLM_Calls fix from silently
    regressing, and keeps a new hazard from appearing anywhere else while it
    isn't yet a tracked exemption."""
    by_file = _scan_package()
    unexempted = {
        path: findings for path, findings in by_file.items() if path not in EXEMPT_FILES
    }
    assert unexempted == {}, "\n".join(
        str(f) for findings in unexempted.values() for f in findings
    )


def test_exempt_files_are_current() -> None:
    """AC #4: the exemption set must reflect reality, not accumulate rot.

    Two ways an entry goes stale: the file was deleted (nothing left to
    convert -- delete the entry too), or the file was already converted and
    now has zero findings (leaving the entry implies work that no longer
    exists). Either way the fix is the same: remove the entry as part of
    whatever change made it stale -- a deliberate edit to this file, not an
    automatic prune, so the removal shows up in review.
    """
    by_file = _scan_package()
    deleted = sorted(
        relative for relative in EXEMPT_FILES if not (PACKAGE_ROOT / relative).exists()
    )
    already_clean = sorted(
        relative
        for relative in EXEMPT_FILES
        if (PACKAGE_ROOT / relative).exists() and relative not in by_file
    )
    assert not deleted, f"EXEMPT_FILES names deleted file(s): {deleted}"
    assert not already_clean, (
        f"EXEMPT_FILES names already-clean file(s), remove from the set: {already_clean}"
    )


# ---------------------------------------------------------------------------
# The scanner itself must not silently pass -- prove it catches known-bad
# shapes directly (belt) in addition to the red-proof done by hand against a
# real repo file (suspenders, see task-19830's report).
# ---------------------------------------------------------------------------


def test_scanner_flags_bare_session_construction() -> None:
    source = "import requests\n\ndef f():\n    return requests.Session()\n"
    findings = _scan_source(source, relative_path="synthetic.py")
    assert [f.kind for f in findings] == ["bare-session"]


def test_scanner_flags_module_level_timeoutless_call() -> None:
    source = "import requests\n\ndef f():\n    return requests.post('https://x', json={})\n"
    findings = _scan_source(source, relative_path="synthetic.py")
    assert [f.kind for f in findings] == ["timeoutless-call"]


def test_scanner_flags_timeoutless_call_on_a_tracked_session_variable() -> None:
    source = (
        "import requests\n\n"
        "def f():\n"
        "    session = requests.Session()\n"
        "    return session.post('https://x', json={})\n"
    )
    findings = _scan_source(source, relative_path="synthetic.py")
    assert {f.kind for f in findings} == {"bare-session", "timeoutless-call"}


def test_scanner_accepts_explicit_keyword_timeout() -> None:
    source = (
        "import requests\n\n"
        "def f():\n"
        "    return requests.post('https://x', json={}, timeout=30)\n"
    )
    assert _scan_source(source, relative_path="synthetic.py") == []


def test_scanner_accepts_explicit_timeout_none() -> None:
    """``timeout=None`` is a deliberate 'no timeout' -- not the same thing as
    omitting the keyword, and must not be flagged."""
    source = (
        "import requests\n\n"
        "def f():\n"
        "    return requests.post('https://x', json={}, timeout=None)\n"
    )
    assert _scan_source(source, relative_path="synthetic.py") == []


def test_scanner_accepts_positional_timeout_on_session_request() -> None:
    source = (
        "import requests\n\n"
        "def f():\n"
        "    session = requests.Session()\n"
        "    return session.request(\n"
        "        'GET', 'https://x', None, None, None, None, None, None, 7\n"
        "    )\n"
    )
    findings = _scan_source(source, relative_path="synthetic.py")
    # The Session() construction itself still needs converting; the
    # `.request()` call, with its explicit positional timeout, must not be.
    assert [f.kind for f in findings] == ["bare-session"]


def test_scanner_ignores_files_that_do_not_import_requests() -> None:
    """A dict's ``.get()`` (``payload.get("x")``) must never be mistaken for
    a ``requests`` call -- this is what makes the file-level ``requests``
    import gate load-bearing rather than decorative."""
    source = "def f(payload):\n    return payload.get('id')\n"
    assert _scan_source(source, relative_path="synthetic.py") == []


def test_scanner_ignores_lazy_in_function_import_target_of_confusion() -> None:
    """A lazy, function-local ``import requests`` must still be recognised --
    several real files in this codebase import it that way on purpose."""
    source = (
        "def f():\n"
        "    import requests\n"
        "    return requests.Session()\n"
    )
    findings = _scan_source(source, relative_path="synthetic.py")
    assert [f.kind for f in findings] == ["bare-session"]


def test_scanner_flags_kwargs_expansion_without_an_explicit_timeout() -> None:
    """`requests.post(url, **opts)` must be flagged, not assumed safe.

    The AST cannot see whether `opts` carries a timeout. Treating the
    unknowable case as safe would blind the guard to precisely the call
    shape where a missing timeout is hardest to spot by eye, so the guard
    resolves the ambiguity the other way and makes the call site say so.
    """
    source = "import requests\nrequests.post(url, **opts)\n"
    findings = _scan_source(source, relative_path="synthetic.py")
    assert len(findings) == 1, findings
    assert "timeout" in str(findings[0]).lower()


def test_scanner_accepts_a_timeout_forwarded_explicitly_beside_kwargs() -> None:
    """The escape hatch has to actually work, or the rule above is a wall.

    A site that genuinely forwards a caller-supplied timeout stays green by
    naming it -- `**kwargs` alongside an explicit `timeout=` is fine.
    """
    source = (
        "import requests\n"
        "requests.post(url, timeout=opts.get('timeout', 30), **opts)\n"
    )
    assert _scan_source(source, relative_path="synthetic.py") == []

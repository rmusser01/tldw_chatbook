"""The test suite never reaches the developer's real OS credential store (TASK-19570 A).

Background: `TldwCli.__init__` wires a server context provider unconditionally --
`app.py:_wire_server_context_provider` -> `build_default_server_credential_store()` ->
`keyring.get_keyring()` (`runtime_policy/server_credentials.py`). There is no test seam
on that path: `Tests/UI/app_factory.py` contains no `keyring` string at all, and
`PYTHON_KEYRING_BACKEND` appeared in zero `conftest.py` files -- its only occurrences
were subprocess env dicts in `Tests/Packaging/`, which isolate a *spawned* process and
not the test session that spawns it.

So every one of the ~620 files that mounts an app was reading the host's credential
subsystem. On macOS a first read can raise a Keychain consent dialog or block on a
locked keychain, and under `timeout_method="thread"` a blocked test kills the entire
run rather than just itself.

A guard that can be removed silently is no guard, so these tests pin the mechanism
(the ambient backend), the seam (what production resolves to), and the end state (what
a constructed app actually holds).
"""

from __future__ import annotations

import ast
import functools
import importlib.util
import os
import re
import tempfile
import textwrap
from pathlib import Path
from unittest import mock

import pytest

from Tests.private_profile import private_profile_test
from tldw_chatbook.runtime_policy.server_credentials import (
    CredentialStoreUnavailable,
    build_default_server_credential_store,
)

NULL_BACKEND = "keyring.backends.null.Keyring"


def test_the_ambient_keyring_backend_is_the_null_backend() -> None:
    """The mechanism: the conftest bootstrap pins the backend before any app import."""
    assert os.environ.get("PYTHON_KEYRING_BACKEND") == NULL_BACKEND

    import keyring

    backend = keyring.get_keyring()
    qualified = f"{type(backend).__module__}.{type(backend).__name__}"
    assert qualified == NULL_BACKEND, (
        f"tests are running against the {qualified!r} keyring backend; on a developer "
        "machine that is the real OS credential store"
    )


def test_the_production_credential_seam_resolves_to_no_os_store() -> None:
    """The seam: the exact call `TldwCli.__init__` makes finds nothing secure.

    `_resolve_secure_keyring_backend` admits only backends whose module is on its
    secure list, so the null backend is correctly refused rather than silently
    accepted as a place to read and write secrets.
    """
    with pytest.raises(CredentialStoreUnavailable):
        build_default_server_credential_store()


@pytest.mark.asyncio
@private_profile_test
def test_a_constructed_app_records_the_store_as_unavailable(request) -> None:
    """The end state: app construction takes the documented fallback, not the OS.

    This is the assertion that would have caught the original gap -- it exercises
    `_wire_server_context_provider` itself rather than the env var it depends on.

    Since TASK-21111(b) the store is resolved lazily, so reading either attribute
    below is what triggers `build_default_server_credential_store()`. That is the
    point of the test, not a hole in it: the fallback decision still has to come
    out the same. `Tests/App/test_startup_init_hygiene.py` pins the complementary
    property -- that construction alone reaches no keyring at all.

    It runs in a private profile (TASK-42480): app construction reads config,
    which the per-test profile redirect refuses with
    `RecoveryRequired: raw_source_selection_changed`. The child inherits this
    process's keyring backend and its own conftest pins the same one.
    """
    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()

    assert app.server_credential_store_unavailable_reason is not None, (
        "app construction found a usable OS credential store during tests -- the "
        "keyring sandbox is not in effect and tests can read/write real credentials"
    )
    assert (
        type(app.server_credential_store).__name__ == "UnavailableServerCredentialStore"
    )


# --- Every environment that moves HOME pins the null backend (TASK-42479) ---
#
# A child or script that moves HOME and the config paths and then imports the
# app must also pin the backend: without the conftest, keyring resolves the
# OS store (the login Keychain on macOS). Under pytest, a copy of the running
# environment, or the running environment itself, already carries the
# conftest's pin, so only test files that copy or edit it are excused. A fresh
# environment handed to a Python child drops the pin too, whatever keys it
# sets (TASK-42520).

_REPO_ROOT = Path(__file__).resolve().parents[1]
_KEYRING_KEY = "PYTHON_KEYRING_BACKEND"
#: `.github/` and `Packaging/` hold Python spawn sites too (TASK-42520). `Docs/` QA
#: harnesses are archived and left out, though some are unsafe to re-run by hand:
#: `Docs/superpowers/qa/2026-08-14-rag-answer-first-query-hang/reproduce.sh` moves
#: HOME without the pin, so it could hit the TASK-32344 Keychain authorization hang.
_CENSUS_TREES = ("Tests", "Helper_Scripts", "scripts", ".github", "Packaging")
#: HOME together with one of these moves the app's profile.
_PROFILE_KEYS = frozenset(
    {
        "TLDW_CONFIG_PATH",
        "TLDW_TEST_MODE",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_CACHE_HOME",
        "XDG_STATE_HOME",
    }
)
_MENTIONS_HOME = re.compile(r"""["']HOME["']|\bHOME\s*=""")
_PASSES_ENV = re.compile(r"\benv\s*=")
#: Exact (path, scope) pairs excused from the pin, each with its reason. A stale
#: pair fails the census.
_EXEMPT = {
    ("Tests/Canvas/mermaid_probe.py", "run_mermaid_case"): "a Node VM child",
    (
        "Tests/Notes/test_file_notes_git_push_service.py",
        "test_network_environment_is_allowlist_with_chatbook_controls",
    ): "an ambient Git environment handed to the push service",
    (
        "Tests/Notes/test_file_notes_git_push_service.py",
        "test_network_context_proved_git_exec_path_dispatches_pinned_targets",
    ): "an ambient Git environment handed to the push service",
    (
        "Tests/Notes/test_file_notes_git_push_service.py",
        "test_unknown_scope_home_fallback_helper_blocks_when_xdg_is_set",
    ): "an ambient Git environment handed to the push service",
    (
        "Tests/Notes/test_file_notes_git_push_service.py",
        "test_unknown_scope_xdg_global_config_is_admitted_and_fingerprinted",
    ): "an ambient Git environment handed to the push service",
    (
        "Tests/Notes/test_file_notes_git_push_service.py",
        "test_exact_candidate_tree_lfs_ignores_external_attribute_sources",
    ): "an ambient Git environment handed to the push service",
    (
        "Tests/Notes/test_file_notes_git_push_transport.py",
        "_fixture_git_environment",
    ): "a Git-only child environment",
    ("Tests/private_profile.py", "wrapped"): (
        "copies the pytest process's pinned environment; forcing the key would "
        "override the TLDW_TEST_REAL_KEYRING opt-out"
    ),
    (
        "Tests/Packaging/test_windows_portable.py",
        "test_portable_environment_drops_host_profile_and_secrets",
    ): "a host environment handed to the launcher, which the test asserts pins null",
    (
        "Tests/Performance/test_rag_citation_provenance_benchmark.py",
        "test_benchmark_host_state_context_isolates_and_restores_environment",
    ): (
        "seeds an in-process host environment with the fail backend (it reaches no "
        "store) to prove the benchmark context replaces it"
    ),
    (
        "Tests/Live/test_library_adaptive_reader_closeout.py",
        "test_child_environment_redirects_every_writable_owner_before_import",
    ): (
        "the child runs pytest on a temporary file that imports no app code; the "
        "runner's allowlist drops the pin"
    ),
}
#: An empty or broken scan must not pass. Measured on 2026-11-24 (TASK-42520):
#: 179 profile moves (176 in the first three trees, 2 in `.github/`, 1 in
#: `Packaging/`) and 272 Python spawns passing ``env=``.
_MIN_PROFILE_MOVES = 120
_MIN_PYTHON_SPAWNS = 200


def _is_environ(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "environ"


def _own_nodes(scope: ast.AST):
    """The scope's nodes, not those of the functions or classes it defines."""
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            stack.extend(ast.iter_child_nodes(node))


class _Env:
    """What one scope does to one environment mapping."""

    def __init__(self, copy: bool = False, unknown: bool = False) -> None:
        self.keys: set[str] = set()
        self.pin: str | None = None  # "null", or "other" for any other value
        self.popped = False
        self.copy = copy  # a copy of the running environment, or os.environ itself
        self.table = False  # only a table of values copied into another mapping
        self.unknown = unknown  # starts from a mapping this scope cannot read
        self.helper: str | None = None  # starts from this module function's result

    def derived(self) -> _Env:
        env = _Env(self.copy, self.unknown)
        env.keys, env.pin, env.popped = set(self.keys), self.pin, self.popped
        env.helper = self.helper
        return env

    def unpin(self) -> None:
        """A ``pop``/``del`` of the key: undoes any pin, and any inherited one."""
        self.pin = None
        self.popped = True

    def take(self, table: _Env) -> None:
        """Copy a table's entries in (a ``.items()`` loop or comprehension)."""
        self.keys |= table.keys
        self.pin = table.pin or self.pin
        table.table = True


def _profile_moves(source: str, *, pytest_file: bool) -> list[tuple[str, int, bool]]:
    """(scope, line, pinned) for each environment a scope moves HOME in."""
    return _scan(source, pytest_file=pytest_file)[0]


def _python_spawns(source: str) -> list[tuple[str, int, bool | None]]:
    """(scope, line, pinned) for each mapping handed as ``env=`` to a Python child.

    TASK-42520 (G1): a fresh environment drops the conftest's pin whatever keys
    it sets. With no HOME, the child even reaches the real home
    (``Path.home()`` falls back to the password database). So every mapping
    passed as ``env=`` to a call that starts Python must pin the null backend,
    or be a copy of the running environment (``{**os.environ}``,
    ``dict(os.environ)``, ``os.environ.copy()``, ``os.environ``, ``None``),
    which starts the child with this process's backend. A ``pop``/``del`` of
    the key undoes a copy. The call starts Python when its argv's first
    element is ``sys.executable`` (also through ``str``/``os.fspath``/
    ``Path(...).resolve()``, or a name bound to it in the scope or module) or
    a ``python*`` string, or when ``-m`` is among its leading options. A ``-c``
    child whose source pins the null backend in-process
    (``keyring.set_keyring(Keyring())`` with ``Keyring`` from
    ``keyring.backends.null``) before any ``tldw_chatbook`` import is
    pinned too (G5).

    ``pinned`` is None for a mapping the scan cannot read: a parameter,
    another module's helper, a method's result. A same-module function's
    result is judged by what that function returns. Limits: the verdict is
    the mapping's state at the end of its scope, not at the call.
    """
    return _scan(source, pytest_file=False)[1]


def _python_source(node: ast.AST | None) -> str | None:
    """A ``-c`` argument's source text; an f-string's fields become ``None``."""
    if isinstance(node, ast.Call) and node.args:  # textwrap.dedent(...)
        if getattr(node.func, "attr", None) == "dedent":
            node = node.args[0]
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            part.value if isinstance(part, ast.Constant) else "None"
            for part in node.values
        )
    return None


def _pins_in_process(source: str | None) -> bool:
    """The child source calls ``keyring.set_keyring(<null Keyring>())`` at its
    top level before any top-level import of ``tldw_chatbook``."""
    if source is None:
        return False
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return False
    null_names = set()  # names bound to the null Keyring class right now
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = (node.module or "") if isinstance(node, ast.ImportFrom) else None
            for alias in node.names:
                if (module or alias.name).split(".")[0] == "tldw_chatbook":
                    return False
                bound = alias.asname or alias.name.split(".")[0]
                if module == "keyring.backends.null" and alias.name == "Keyring":
                    null_names.add(bound)
                else:
                    null_names.discard(bound)
            continue
        if isinstance(node, ast.Assign):
            null_names -= {ast.unparse(target) for target in node.targets}
        call = node.value if isinstance(node, ast.Expr) else None
        if (
            isinstance(call, ast.Call)
            and ast.unparse(call.func) == "keyring.set_keyring"
            and len(call.args) == 1
            and isinstance(call.args[0], ast.Call)
            and not call.args[0].args
            and (
                ast.unparse(call.args[0].func) == NULL_BACKEND
                or ast.unparse(call.args[0].func) in null_names
            )
        ):
            return True
    return False


def _scan(
    source: str, *, pytest_file: bool
) -> tuple[list[tuple[str, int, bool]], list[tuple[str, int, bool | None]]]:
    """The HOME moves and the Python spawns of one source.

    Each mapping is tracked on its own: a name (``env = {...}``, then
    ``env.update(...)``, ``env["K"] = ...``, ``dict(env, K=...)``), ``os.environ``
    itself, or an unnamed literal. It is pinned only by the backend key set to
    the constant null backend (or a module constant holding it), never by a
    ``pop``/``del`` of it (a later re-pin still counts). In a pytest file, a copy of the running environment
    (``{**os.environ}``, ``dict(os.environ)``, ``os.environ.copy()``) or
    ``os.environ`` itself already carries the conftest's pin. An allowlist or
    denylist built from ``os.environ.items()`` is a fresh environment. A table
    copied in by a ``.items()`` comprehension or loop (``x[key] = ...``,
    ``monkeypatch.setenv``) counts for the mapping it fills, not on its own.

    Limits: one scope at a time, so HOME and the profile keys set in different
    helpers are not seen together; a mapping passed to another function to be
    filled in is not followed; and a path table that is itself passed on (as
    ``env=``) is still treated as only a table (the one live case is the Git
    environment in ``Tests/Notes/test_file_notes_git_integration.py``).
    """
    tree = ast.parse(source)
    constants = {
        target.id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        for target in node.targets
        if isinstance(target, ast.Name)
    }

    module_values = {
        target.id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }

    def is_null(value: ast.AST) -> bool:
        if isinstance(value, ast.Name):
            return constants.get(value.id) == NULL_BACKEND
        return isinstance(value, ast.Constant) and value.value == NULL_BACKEND

    def resolve(node: ast.AST, values: dict[str, ast.AST]) -> ast.AST:
        if isinstance(node, ast.Name):
            return values.get(node.id) or module_values.get(node.id) or node
        return node

    def is_python(node: ast.AST, values: dict[str, ast.AST]) -> bool:
        for _ in range(4):  # str(Path(sys.executable).resolve()) and the like
            node = resolve(node, values)
            if not isinstance(node, ast.Call):
                break
            func = node.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name in ("resolve", "absolute") and isinstance(func, ast.Attribute):
                node = func.value
            elif name in ("str", "fspath", "Path") and len(node.args) == 1:
                node = node.args[0]
            else:
                break
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return Path(node.value).name.startswith("python")
        return ast.unparse(node) == "sys.executable"

    def python_argv(call: ast.Call, values: dict[str, ast.AST]) -> list | None:
        """The argv of the Python child ``call`` starts, or None."""
        args = (
            call.args[1:]
            if getattr(call.func, "attr", None) == "to_thread"
            else call.args
        )
        if not args:
            return None
        first = resolve(args[0], values)
        argv = list(first.elts) if isinstance(first, (ast.List, ast.Tuple)) else args
        if not argv:
            return None
        return argv if "-m" in flags(argv) or is_python(argv[0], values) else None

    def flags(argv: list) -> list[str]:
        """The constant options that follow the program, up to the first operand."""
        found = []
        for item in argv[1:]:
            if not (
                isinstance(item, ast.Constant)
                and isinstance(item.value, str)
                and item.value.startswith("-")
            ):
                break
            found.append(item.value)
        return found

    scopes = [tree] + [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    moves = []
    spawns: list[tuple[str, int, _Env | None, bool]] = []
    returns: dict[str, list[_Env | None]] = {}
    for scope in scopes:
        nodes = list(_own_nodes(scope))
        envs: dict[object, _Env] = {}
        handled: set[int] = set()
        values: dict[str, ast.AST] = {}

        def env_of(target: ast.AST) -> _Env:
            key = ast.unparse(target)
            if key not in envs:
                environ = _is_environ(target)
                envs[key] = _Env(copy=environ, unknown=not environ)
            return envs[key]

        def table_of(iterable: ast.AST) -> _Env | None:
            """The mapping a ``.items()`` loop reads, when this scope built it."""
            func = getattr(iterable, "func", None)
            if not isinstance(func, ast.Attribute) or func.attr != "items":
                return None
            if isinstance(func.value, ast.Dict):
                table = _Env()
                fill(table, func.value)
                return table
            if _is_environ(func.value):
                return None
            return envs.get(ast.unparse(func.value))

        def put(env: _Env, key: str, value: ast.AST) -> None:
            env.keys.add(key)
            if key == _KEYRING_KEY:
                env.pin = "null" if is_null(value) else "other"

        def fill(env: _Env, node: ast.AST) -> None:
            """Apply a dict literal or a ``dict(...)``/``update(...)`` call."""
            handled.add(id(node))
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if key is None:  # ``**base``: its provenance and pin, not its keys
                        base = (
                            _Env(copy=True)
                            if _is_environ(value)
                            else envs.get(ast.unparse(value), _Env(unknown=True))
                        )
                        env.copy = env.copy or base.copy
                        env.unknown = env.unknown or base.unknown
                        env.helper = env.helper or base.helper
                        env.popped = env.popped or base.popped
                        env.pin = base.pin or env.pin
                    elif isinstance(key, ast.Constant) and isinstance(key.value, str):
                        put(env, key.value, value)
            elif isinstance(node, ast.Call):
                for arg in node.args:
                    if isinstance(arg, ast.Dict):
                        fill(env, arg)
                    elif isinstance(arg, ast.DictComp):
                        table = table_of(arg.generators[0].iter)
                        if table is not None:
                            env.take(table)
                for keyword in node.keywords:
                    if keyword.arg:
                        put(env, keyword.arg, keyword.value)

        def result_env(value: ast.AST) -> _Env:
            """A mapping that does not copy one: a comprehension is fresh, a
            same-module function's result is that function's, the rest unknown."""
            env = _Env(unknown=not isinstance(value, ast.DictComp))
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
                env.unknown, env.helper = False, value.func.id
            return env

        def source_env(value: ast.AST) -> _Env | None:
            """The environment a new mapping starts from, if it copies one."""
            if isinstance(value, ast.Dict):
                return _Env()
            if isinstance(value, ast.Call):
                func = value.func
                if isinstance(func, ast.Name) and func.id == "dict":
                    base = value.args[0] if value.args else None
                    if base is None or isinstance(base, ast.Dict):
                        return _Env()
                    if _is_environ(base):
                        return _Env(copy=True)
                    known = envs.get(ast.unparse(base))
                    return known.derived() if known else _Env(unknown=True)
                if isinstance(func, ast.Attribute) and func.attr == "copy":
                    if _is_environ(func.value):
                        return _Env(copy=True)
            return None

        # Assignments first, in source order, so a later update finds its env.
        for node in sorted(
            (
                n
                for n in nodes
                if isinstance(
                    n, (ast.Assign, ast.AnnAssign, ast.Call, ast.Delete, ast.For)
                )
            ),
            key=lambda n: (n.lineno, n.col_offset),
        ):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                target = (
                    node.targets[0] if isinstance(node, ast.Assign) else node.target
                )
                if node.value is None:
                    continue
                if isinstance(target, ast.Subscript):
                    if isinstance(target.slice, ast.Constant) and isinstance(
                        target.slice.value, str
                    ):
                        put(env_of(target.value), target.slice.value, node.value)
                    continue
                if isinstance(target, (ast.Name, ast.Attribute)):
                    # Rebinding a name starts a new mapping: a filtered copy
                    # or a helper's result is not the copy it replaced.
                    if isinstance(target, ast.Name):
                        values[target.id] = node.value
                    start = source_env(node.value)
                    if start is not None:  # before binding: ``e = {**e, ...}``
                        fill(start, node.value)
                    envs[ast.unparse(target)] = start or result_env(node.value)
            elif isinstance(node, ast.For):
                table = table_of(node.iter)
                for inner in ast.walk(node) if table is not None else ():
                    if isinstance(inner, ast.Assign) and isinstance(
                        inner.targets[0], ast.Subscript
                    ):
                        env_of(inner.targets[0].value).take(table)
                    elif (
                        isinstance(inner, ast.Call)
                        and getattr(inner.func, "attr", None) == "setenv"
                    ):
                        envs.setdefault("os.environ", _Env(copy=True)).take(table)
            elif isinstance(node, ast.Delete):
                for target in node.targets:
                    if isinstance(target, ast.Subscript) and isinstance(
                        target.slice, ast.Constant
                    ):
                        if target.slice.value == _KEYRING_KEY:
                            env_of(target.value).unpin()
            elif isinstance(node.func, ast.Attribute) and id(node) not in handled:
                if node.func.attr == "update":
                    fill(env_of(node.func.value), node)
                elif node.func.attr == "pop" and node.args:
                    first = node.args[0]
                    if isinstance(first, ast.Constant) and first.value == _KEYRING_KEY:
                        env_of(node.func.value).unpin()
        # Unnamed literals: a returned dict, a ``dict(x, ...)`` passed on, ...
        for node in nodes:
            if id(node) in handled:
                continue
            start = source_env(node)
            if start is not None and not (
                isinstance(node, ast.Call) and not node.keywords and not node.args
            ):
                envs[id(node)] = start
                fill(start, node)
        for env in envs.values():
            if not env.table and "HOME" in env.keys and env.keys & _PROFILE_KEYS:
                pinned = env.pin == "null" or (
                    env.pin is None and not env.popped and pytest_file and env.copy
                )
                moves.append(
                    (
                        getattr(scope, "name", "<module>"),
                        getattr(scope, "lineno", 0),
                        pinned,
                    )
                )

        def mapping_at(value: ast.AST) -> _Env | None:
            """The mapping an ``env=`` or ``return`` expression hands on."""
            if isinstance(value, ast.Constant) and value.value is None:
                return _Env(copy=True)  # env=None inherits this environment
            if id(value) in envs:
                return envs[id(value)]
            if _is_environ(value):
                return _Env(copy=True)
            if isinstance(value, (ast.Name, ast.Attribute)):
                return envs.get(ast.unparse(value))
            start = source_env(value)
            if start is not None:
                fill(start, value)
                return start
            return result_env(value)

        for node in nodes:
            if isinstance(node, ast.Return) and node.value is not None:
                returns.setdefault(getattr(scope, "name", ""), []).append(
                    mapping_at(node.value)
                )
                continue
            if not isinstance(node, ast.Call):
                continue
            env_value = next((k.value for k in node.keywords if k.arg == "env"), None)
            argv = python_argv(node, values) if env_value is not None else None
            if argv is None:
                continue
            child, options = None, flags(argv)
            if options and options[-1] == "-c" and len(argv) > len(options) + 1:
                child = _python_source(resolve(argv[len(options) + 1], values))
            spawns.append(
                (
                    getattr(scope, "name", "<module>"),
                    env_value.lineno,
                    mapping_at(env_value),
                    _pins_in_process(child),
                )
            )

    def verdict(env: _Env | None, seen: tuple[str, ...] = ()) -> bool | None:
        if env is None:
            return None
        if env.pin is not None:
            return env.pin == "null"
        if env.popped:
            return False
        if env.copy:
            return True
        if env.helper is not None:
            found = returns.get(env.helper)
            if not found or env.helper in seen:
                return None
            each = [verdict(result, (*seen, env.helper)) for result in found]
            if any(result is False for result in each):
                return False
            return True if all(each) else None
        return None if env.unknown else False

    return moves, [
        (scope, line, True if in_process else verdict(env))
        for scope, line, env, in_process in sorted(spawns, key=lambda s: s[1])
    ]


_NULL = '"keyring.backends.null.Keyring"'


@pytest.mark.parametrize(
    ("source", "pytest_file", "pinned"),
    [
        # Scratch literal, unpinned / pinned / pinned by a module constant.
        ('def f(r):\n    return {"HOME": r, "TLDW_CONFIG_PATH": r}\n', False, [False]),
        (
            f'def f(r):\n    return {{"HOME": r, "XDG_DATA_HOME": r, "{_KEYRING_KEY}": {_NULL}}}\n',
            False,
            [True],
        ),
        (
            f'N = {_NULL}\ndef f(r):\n    return {{"HOME": r, "TLDW_TEST_MODE": "1", "{_KEYRING_KEY}": N}}\n',
            False,
            [True],
        ),
        # Only the null backend counts; another value or a pop does not.
        (
            f'def f(r):\n    return {{"HOME": r, "XDG_DATA_HOME": r, "{_KEYRING_KEY}": "keyring.backends.macOS.Keyring"}}\n',
            False,
            [False],
        ),
        (
            f'def f(r):\n    e = dict(os.environ)\n    e.update(HOME=r, TLDW_TEST_MODE="1")\n    e.pop("{_KEYRING_KEY}", None)\n',
            True,
            [False],
        ),
        # A copy of the running environment inherits the pin only under pytest.
        (
            'def f(r):\n    e = dict(os.environ)\n    e["HOME"] = r\n    e["TLDW_TEST_MODE"] = "1"\n',
            True,
            [True],
        ),
        (
            'def f(r):\n    e = dict(os.environ)\n    e["HOME"] = r\n    e["TLDW_TEST_MODE"] = "1"\n',
            False,
            [False],
        ),
        # An allowlist built from os.environ.items() is a fresh environment.
        (
            "def f(r):\n    e = {k: v for k, v in os.environ.items() if k in A}\n    e.update(HOME=r, TLDW_CONFIG_PATH=r)\n",
            True,
            [False],
        ),
        # A setenv elsewhere in the scope excuses no other mapping.
        (
            'def f(r, mp):\n    mp.setenv("X", "1")\n    return {"HOME": r, "TLDW_CONFIG_PATH": r}\n',
            True,
            [False],
        ),
        (
            'import os\nos.environ.update(HOME="h", XDG_CONFIG_HOME="c")\n',
            False,
            [False],
        ),
        (
            f'def f(e):\n    e.update(HOME="h", TLDW_CONFIG_PATH="c", {_KEYRING_KEY}={_NULL})\n',
            False,
            [True],
        ),
        # A path table counts for the mapping it is copied into.
        (
            f'def f(r):\n    p = {{"HOME": r, "XDG_DATA_HOME": r}}\n    e = {{"{_KEYRING_KEY}": {_NULL}}}\n    e.update({{k: str(v) for k, v in p.items()}})\n',
            False,
            [True],
        ),
        (
            'def f(r):\n    p = {"HOME": r, "XDG_DATA_HOME": r}\n    e = {"PATH": "x"}\n    e.update({k: str(v) for k, v in p.items()})\n',
            False,
            [False],
        ),
        # Rebinding a copy to a filtered copy or a helper's result starts afresh.
        (
            "def f(r):\n    e = dict(os.environ)\n    e = {k: v for k, v in e.items() if k in A}\n    e.update(HOME=r, TLDW_CONFIG_PATH=r)\n",
            True,
            [False],
        ),
        (
            "def f(r):\n    e = dict(os.environ)\n    e = allow(e)\n    e.update(HOME=r, TLDW_CONFIG_PATH=r)\n",
            True,
            [False],
        ),
        # A pop after a pin undoes it; a re-pin after the pop counts.
        (
            f'def f(r):\n    e = {{"HOME": r, "XDG_DATA_HOME": r, "{_KEYRING_KEY}": {_NULL}}}\n    e.pop("{_KEYRING_KEY}")\n',
            False,
            [False],
        ),
        (
            f'def f(r):\n    e = {{"HOME": r, "XDG_DATA_HOME": r}}\n    del e["{_KEYRING_KEY}"]\n    e["{_KEYRING_KEY}"] = {_NULL}\n',
            False,
            [True],
        ),
        ('def f(r):\n    return {"HOME": r, "PATH": r}\n', False, []),
    ],
)
def test_the_census_reads_each_way_an_environment_moves_home(
    source: str, pytest_file: bool, pinned: list[bool]
) -> None:
    """The negative control: unpinned moves are seen, pinned or inherited ones pass."""
    assert [
        move[2] for move in _profile_moves(source, pytest_file=pytest_file)
    ] == pinned


_RUN = "import subprocess, sys\nsubprocess.run("
_PIN = f'"{_KEYRING_KEY}": {_NULL}'
_CHILD_PIN = (
    "import keyring\nfrom keyring.backends.null import Keyring\n"
    "keyring.set_keyring(Keyring())\n"
)


@pytest.mark.parametrize(
    ("source", "pinned"),
    [
        # G1: a fresh environment, with no HOME at all, is flagged; a pin passes.
        (_RUN + '[sys.executable, "-c", s], env={"PATH": p})\n', [False]),
        (_RUN + f'[sys.executable, "-c", s], env={{"PATH": p, {_PIN}}})\n', [True]),
        (
            f'def f(p):\n    e = {{"PATH": p}}\n    e["{_KEYRING_KEY}"] = {_NULL}\n'
            '    subprocess.run([sys.executable, "-c", s], env=e)\n',
            [True],
        ),
        # A copy of the running environment, or no env, inherits this process's.
        (_RUN + '[sys.executable, "-c", s], env=dict(os.environ))\n', [True]),
        (_RUN + '[sys.executable, "-c", s], env={**os.environ, "X": "1"})\n', [True]),
        (_RUN + '[sys.executable, "-c", s], env=os.environ.copy())\n', [True]),
        (_RUN + '[sys.executable, "-c", s], env=None)\n', [True]),
        (
            f'def f():\n    e = os.environ.copy()\n    e.pop("{_KEYRING_KEY}")\n'
            "    subprocess.run([sys.executable], env=e)\n",
            [False],
        ),
        # A filtered copy is fresh, as for the HOME moves.
        (
            "def f():\n    e = {k: v for k, v in os.environ.items() if k != 'X'}\n"
            "    subprocess.run([sys.executable, 'x.py'], env=e)\n",
            [False],
        ),
        # ``e = {**e, ...}`` keeps what ``e`` was; a parameter is not judged.
        (
            "def f(e):\n    e = {**e, 'X': '1'}\n"
            "    subprocess.run([sys.executable], env=e)\n",
            [None],
        ),
        # Each way the argv starts Python; other programs are not counted.
        (
            'def f(p):\n    cmd = [sys.executable, "-c", s]\n'
            '    subprocess.Popen(cmd, env={"PATH": p})\n',
            [False],
        ),
        (
            _RUN + '[str(Path(sys.executable).resolve()), "x.py"], env={"PATH": p})\n',
            [False],
        ),
        (_RUN + '["python3", "x.py"], env={"PATH": p})\n', [False]),
        (_RUN + '[venv_python, "-m", "pytest"], env={"PATH": p})\n', [False]),
        (_RUN + '[venv_python, "-I", "-m", "pytest"], env={"PATH": p})\n', [False]),
        (
            "async def f(p):\n    await asyncio.create_subprocess_exec("
            'sys.executable, "-c", s, env={"PATH": p})\n',
            [False],
        ),
        (_RUN + '["git", "status"], env={"PATH": p})\n', []),
        # A same-module helper's result is judged by what it returns.
        (
            'def env():\n    return {"PATH": "x"}\n'
            "def f():\n    subprocess.run([sys.executable], env=env())\n",
            [False],
        ),
        (
            f'def env():\n    return {{"PATH": "x", {_PIN}}}\n'
            "def f():\n    e = env()\n    subprocess.run([sys.executable], env=e)\n",
            [True],
        ),
        (
            "def f():\n    subprocess.run([sys.executable], env=imported_env())\n",
            [None],
        ),
        # G5: a -c child that pins the null backend in-process before any app import.
        (
            f"CHILD = {_CHILD_PIN + 'import tldw_chatbook'!r}\n"
            + _RUN
            + '[sys.executable, "-c", CHILD, "mode"], env={"PATH": p})\n',
            [True],
        ),
        (
            f"CHILD = {'import tldw_chatbook' + chr(10) + _CHILD_PIN!r}\n"
            + _RUN
            + '[sys.executable, "-c", CHILD], env={"PATH": p})\n',
            [False],
        ),
        (
            f"CHILD = {_CHILD_PIN.replace('null', 'macOS')!r}\n"
            + _RUN
            + '[sys.executable, "-c", CHILD], env={"PATH": p})\n',
            [False],
        ),
        (  # the null class's name rebound to another backend before the call
            "CHILD = "
            + repr(
                _CHILD_PIN.replace(
                    "keyring.set",
                    "from keyring.backends.macOS import Keyring\nkeyring.set",
                )
            )
            + "\n"
            + _RUN
            + '[sys.executable, "-c", CHILD], env={"PATH": p})\n',
            [False],
        ),
        (  # options before -c
            f"CHILD = {_CHILD_PIN!r}\n"
            + _RUN
            + '[sys.executable, "-I", "-B", "-c", CHILD], env={"PATH": p})\n',
            [True],
        ),
        # An f-string child (its fields read as None) or a dedented constant.
        (
            _RUN
            + '[sys.executable, "-c", f'
            + repr(_CHILD_PIN + "x = {y!r}")
            + '], env={"PATH": p})\n',
            [True],
        ),
        (
            f"CHILD = textwrap.dedent({_CHILD_PIN!r})\n"
            + _RUN
            + '[sys.executable, "-c", CHILD], env={"PATH": p})\n',
            [True],
        ),
        # Only a -c source is read: a script path's pin is not seen.
        (_RUN + '[sys.executable, "child.py"], env={"PATH": p})\n', [False]),
    ],
)
def test_the_census_reads_each_way_a_python_child_gets_its_environment(
    source: str, pinned: list[bool | None]
) -> None:
    """The negative control for G1 and G5 (TASK-42520)."""
    assert [spawn[2] for spawn in _python_spawns(source)] == pinned


@functools.cache
def _census() -> tuple[int, list[str], set[tuple[str, str]], int, list[str]]:
    """(moves, unpinned moves, exemptions seen, Python spawns, unpinned spawns)."""
    moves, unpinned, exempt_seen = 0, [], set()
    spawns, unpinned_spawns = 0, []
    for tree in _CENSUS_TREES:
        for path in sorted((_REPO_ROOT / tree).rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            if not (_MENTIONS_HOME.search(source) or _PASSES_ENV.search(source)):
                continue
            relative = path.relative_to(_REPO_ROOT).as_posix()
            pytest_file = path.name.startswith("test_") or path.name == "conftest.py"
            home_moves, python_spawns = _scan(source, pytest_file=pytest_file)
            for scope, line, pinned in home_moves:
                moves += 1
                if (relative, scope) in _EXEMPT:
                    exempt_seen.add((relative, scope))
                elif not pinned:
                    unpinned.append(f"{relative}:{line} {scope}")
            for scope, line, pinned in python_spawns:
                spawns += 1
                if pinned is False:
                    unpinned_spawns.append(f"{relative}:{line} {scope}")
    return moves, unpinned, exempt_seen, spawns, unpinned_spawns


def test_every_fresh_environment_handed_to_a_python_child_pins_the_null_backend() -> (
    None
):
    """G1 (TASK-42520): a fresh ``env=`` drops the conftest's pin, HOME or not."""
    *_, spawns, unpinned = _census()

    assert spawns >= _MIN_PYTHON_SPAWNS, f"the census saw only {spawns} Python spawns"
    assert not unpinned, (
        "these start Python with a fresh environment that lacks "
        "PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring (the child "
        "resolves the OS store, and with no HOME the real home):\n"
        + "\n".join(unpinned)
    )


def test_every_environment_that_moves_home_pins_the_null_keyring_backend() -> None:
    moves, unpinned, exempt_seen, *_ = _census()

    assert moves >= _MIN_PROFILE_MOVES, f"the census saw only {moves} profile moves"
    assert not unpinned, (
        "these move HOME without PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring "
        "(without a conftest, keyring resolves the OS store):\n" + "\n".join(unpinned)
    )
    assert exempt_seen == set(_EXEMPT), (
        f"stale exemptions: {set(_EXEMPT) - exempt_seen}"
    )


def _load_script(relative: str):
    path = _REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(f"_keyring_pin_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_mount_profiler_pins_the_null_backend_before_any_app_import(
    tmp_path: Path,
) -> None:
    """Run by hand there is no conftest: the profile move must carry the pin."""
    from Tests.Performance.run_console_mount_profile import _configure_isolated_profile

    with mock.patch.dict(os.environ, {_KEYRING_KEY: "keyring.backends.fail.Keyring"}):
        _configure_isolated_profile(tmp_path)
        assert os.environ.get(_KEYRING_KEY) == NULL_BACKEND


def test_the_boot_budget_snapshot_script_pins_the_null_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _load_script("scripts/update_boot_budget_snapshots.py")
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    with mock.patch.dict(os.environ, {_KEYRING_KEY: "keyring.backends.fail.Keyring"}):
        script._isolate_config()
        assert os.environ.get(_KEYRING_KEY) == NULL_BACKEND


def test_the_notes_sync_tui_verifier_environment_pins_the_null_backend(
    tmp_path: Path,
) -> None:
    """Only the environment builder runs; the verifier itself never does."""
    verifier = _load_script("Helper_Scripts/verify_notes_files_sync_tui.py")

    environment, _config, _data = verifier.build_isolated_environment(tmp_path)

    assert environment.get(_KEYRING_KEY) == NULL_BACKEND


def test_the_task598_evidence_child_environment_pins_the_null_backend(
    tmp_path: Path,
) -> None:
    """G2 (TASK-42520): `.github/` scripts are in the census; this child runs
    the app's transcription with HOME moved. Only the builder runs."""
    evidence = _load_script(".github/scripts/task598_external_parakeet_evidence.py")
    host = {
        _KEYRING_KEY: "keyring.backends.fail.Keyring",
        "TASK598_TESTED_COMMIT": "0" * 40,
    }

    with mock.patch.dict(os.environ, host):
        environment = evidence._isolated_environment(tmp_path)

    assert environment.get(_KEYRING_KEY) == NULL_BACKEND


def test_the_task602_smoke_isolation_pins_the_null_backend_and_restores_it(
    tmp_path: Path,
) -> None:
    smoke = _load_script(".github/scripts/task602_platform_smoke.py")

    with mock.patch.dict(os.environ, {_KEYRING_KEY: "keyring.backends.fail.Keyring"}):
        previous = smoke._isolate_process(tmp_path)
        assert os.environ.get(_KEYRING_KEY) == NULL_BACKEND
        smoke._restore_process(previous)
        assert os.environ.get(_KEYRING_KEY) == "keyring.backends.fail.Keyring"

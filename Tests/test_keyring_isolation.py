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
import importlib.util
import os
import re
import tempfile
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
# conftest's pin, so only test files that copy or edit it are excused.

_REPO_ROOT = Path(__file__).resolve().parents[1]
_KEYRING_KEY = "PYTHON_KEYRING_BACKEND"
_CENSUS_TREES = ("Tests", "Helper_Scripts", "scripts")
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
#: An empty or broken scan must not pass (174 profile moves on 2026-11-24).
_MIN_PROFILE_MOVES = 120


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

    def __init__(self, copy: bool = False) -> None:
        self.keys: set[str] = set()
        self.pin: str | None = None  # "null", or "other" for any other value
        self.popped = False
        self.copy = copy  # a copy of the running environment, or os.environ itself
        self.table = False  # only a table of values copied into another mapping

    def derived(self) -> _Env:
        env = _Env(self.copy)
        env.keys, env.pin, env.popped = set(self.keys), self.pin, self.popped
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
    """(scope, line, pinned) for each environment a scope moves HOME in.

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

    def is_null(value: ast.AST) -> bool:
        if isinstance(value, ast.Name):
            return constants.get(value.id) == NULL_BACKEND
        return isinstance(value, ast.Constant) and value.value == NULL_BACKEND

    scopes = [tree] + [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    moves = []
    for scope in scopes:
        nodes = list(_own_nodes(scope))
        envs: dict[object, _Env] = {}
        handled: set[int] = set()

        def env_of(target: ast.AST) -> _Env:
            key = ast.unparse(target)
            if key not in envs:
                envs[key] = _Env(copy=_is_environ(target))
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
                    if key is None:
                        env.copy = env.copy or _is_environ(value)
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
                    return known.derived() if known else _Env()
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
                    start = source_env(node.value)
                    envs[ast.unparse(target)] = start or _Env()
                    if start is not None:
                        fill(start, node.value)
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
    return moves


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


def test_every_environment_that_moves_home_pins_the_null_keyring_backend() -> None:
    moves, unpinned, exempt_seen = 0, [], set()
    for tree in _CENSUS_TREES:
        for path in sorted((_REPO_ROOT / tree).rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            if not _MENTIONS_HOME.search(source):
                continue
            relative = path.relative_to(_REPO_ROOT).as_posix()
            pytest_file = path.name.startswith("test_") or path.name == "conftest.py"
            for scope, line, pinned in _profile_moves(source, pytest_file=pytest_file):
                moves += 1
                if (relative, scope) in _EXEMPT:
                    exempt_seen.add((relative, scope))
                elif not pinned:
                    unpinned.append(f"{relative}:{line} {scope}")

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

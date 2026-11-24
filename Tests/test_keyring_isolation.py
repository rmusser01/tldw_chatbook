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
}
#: An empty or broken scan must not pass (164 profile moves on 2026-11-24).
_MIN_PROFILE_MOVES = 120


def _is_environ(node: ast.AST) -> bool:
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


def _profile_moves(source: str, *, pytest_file: bool) -> list[tuple[str, int, bool]]:
    """(scope, line, pinned) for each function or module that moves HOME.

    Keys come from dict literals, ``update(...)``/``dict(...)`` keywords and
    ``x["KEY"] = ...``. A scope is pinned when it names the backend key, or,
    in a pytest file, when it copies or edits the running environment.
    """
    tree = ast.parse(source)
    scopes = [tree] + [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    moves = []
    for scope in scopes:
        keys: set[str] = set()
        names_key = inherits = False
        for node in _own_nodes(scope):
            if isinstance(node, ast.Constant) and node.value == _KEYRING_KEY:
                names_key = True
            elif isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if key is None:
                        inherits = inherits or _is_environ(value)
                    elif isinstance(key, ast.Constant) and isinstance(key.value, str):
                        keys.add(key.value)
            elif isinstance(node, ast.Call):
                func = node.func
                name = getattr(func, "attr", getattr(func, "id", None))
                if name in ("update", "dict"):
                    keys.update(kw.arg for kw in node.keywords if kw.arg)
                inherits = inherits or (
                    name == "setenv"
                    or (
                        name in ("update", "copy", "items")
                        and _is_environ(getattr(func, "value", None))
                    )
                    or (
                        name == "dict" and bool(node.args) and _is_environ(node.args[0])
                    )
                )
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Subscript) and isinstance(
                        target.slice, ast.Constant
                    ):
                        keys.add(str(target.slice.value))
                        inherits = inherits or _is_environ(target.value)
        if "HOME" in keys and keys & _PROFILE_KEYS:
            pinned = names_key or _KEYRING_KEY in keys or (pytest_file and inherits)
            moves.append(
                (
                    getattr(scope, "name", "<module>"),
                    getattr(scope, "lineno", 0),
                    pinned,
                )
            )
    return moves


@pytest.mark.parametrize(
    ("source", "pytest_file", "pinned"),
    [
        ('def f(r):\n    return {"HOME": r, "TLDW_CONFIG_PATH": r}\n', False, [False]),
        (
            'def f(r):\n    return {"HOME": r, "XDG_DATA_HOME": r,\n'
            '        "PYTHON_KEYRING_BACKEND": "keyring.backends.null.Keyring"}\n',
            False,
            [True],
        ),
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
        (
            'import os\nos.environ.update(HOME="h", XDG_CONFIG_HOME="c")\n',
            False,
            [False],
        ),
        (
            'def f(e):\n    e.update(HOME="h", TLDW_CONFIG_PATH="c", PYTHON_KEYRING_BACKEND="n")\n',
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

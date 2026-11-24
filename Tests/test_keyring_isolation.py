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
from collections import Counter
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
#: Exact (path, scope) pairs whose Python children are excused, with reasons.
_SPAWN_EXEMPT = {
    (
        "Tests/CI/test_plugin_qualification_workflow.py",
        "test_plugin_workflow_preserves_owned_loopback_and_asyncio",
    ): (
        "the allowlist is the workflow's own, read at run time; it holds HOME, "
        "and the test pins the backend"
    ),
    (
        "Tests/Canvas/test_startup_deferral.py",
        "test_native_app_does_not_import_served_control_transport",
    ): (
        "pops only the keys of CONTROL_ENV, a module dict the census does not "
        "evaluate (all CHATBOOK_CANVAS_CONTROL_*); _isolated_environment sets "
        "HOME and the pin"
    ),
    ("Tests/Library/test_rag_answer_first_query_latency.py", "_run"): (
        "pinned; HOME is dropped on purpose, and both child sources (_SEED, "
        "_MEASURE) set HOME and the XDG paths in-process before any app import"
    ),
}
#: An empty or broken scan must not pass. Measured on 2026-11-24 (TASK-42520):
#: 179 profile moves (176 in the first three trees, 2 in `.github/`, 1 in
#: `Packaging/`) and 294 recognised Python spawns passing ``env=`` (290 in
#: `Tests/`, 2 in `scripts/`, 1 each in `.github/` and `Packaging/`).
_MIN_PROFILE_MOVES = 120
_MIN_PYTHON_SPAWNS = 200
#: Spawns whose environment the census cannot read (a parameter, another
#: module's helper, a closure, a tuple unpacking), as (path, scope): count,
#: pinned exactly on 2026-11-24 (TASK-42520). Their builders were spot-checked
#: for the pin and HOME. A new one, or one that moves, fails until someone
#: checks it and updates this table.
_UNJUDGED_SPAWNS = {
    ("Packaging/windows/portable/wezterm/build.py", "build"): 1,
    ("Tests/App/test_construct_runtime_imports.py", "_run_construct_probe"): 1,
    ("Tests/Backup_Recovery/run_platform_product.py", "_run_pytest_phase"): 1,
    ("Tests/Backup_Recovery/test_complete_roundtrip.py", "_run_profile_child"): 1,
    (
        "Tests/Backup_Recovery/test_detached_layout.py",
        "test_restored_canonical_cache_created_by_real_getter_is_narrowly_excluded",
    ): 1,
    ("Tests/Backup_Recovery/test_isolated_pair_recovery.py", "_child"): 1,
    (
        "Tests/Backup_Recovery/test_native_credential_runner.py",
        "test_native_workflow_saved_script_imports_checkout_without_pythonpath",
    ): 1,
    (
        "Tests/Backup_Recovery/test_native_multivolume_replacement.py",
        "test_actual_two_device_interruption_recovers",
    ): 2,
    (
        "Tests/CI/test_task598_external_parakeet_evidence.py",
        "test_supervisor_removes_owned_native_temp_after_forced_timeout",
    ): 1,
    ("Tests/Canvas/browser/test_canvas_process_restart.py", "start"): 1,
    ("Tests/Character_Chat/test_visual_identity_pil_lazy.py", "_run_phase"): 1,
    (
        "Tests/Notes/test_file_notes_git_push_service.py",
        "test_network_context_child_scratch_directories_reject_writes",
    ): 1,
    ("Tests/Packaging/test_canvas_gateway_distribution.py", "canvas_gateway_wheel"): 1,
    (
        "Tests/Packaging/test_canvas_gateway_distribution.py",
        "test_canvas_gateway_loads_packaged_runtime_from_wheel",
    ): 1,
    ("Tests/Packaging/test_mcp_unified_distribution.py", "_exercise_server"): 1,
    (
        "Tests/Packaging/test_pixel_migu_distribution.py",
        "test_installed_pixel_migu_fresh_profile",
    ): 1,
    ("Tests/Packaging/test_private_sqlite_helper_distribution.py", "_run_helper"): 1,
    (
        "Tests/Packaging/test_private_sqlite_helper_distribution.py",
        "installed_helper",
    ): 2,
    (
        "Tests/Packaging/test_private_sqlite_helper_distribution.py",
        "test_installed_entry_preserves_argv_and_original_parent_checks",
    ): 1,
    ("Tests/Packaging/test_terminal_distribution.py", "terminal_wheel"): 1,
    (
        "Tests/ProductionApp/test_backup_restore_end_to_end.py",
        "_run_windows_console",
    ): 1,
    ("Tests/ProductionApp/test_native_credential_recovery.py", "_transfer"): 1,
    ("Tests/test_config_library_defaults.py", "read_profile"): 1,
    ("Tests/test_tiktoken_vendored_cache.py", "_run_source_child"): 1,
}


def _is_environ(node: ast.AST | None, names: frozenset[str] = frozenset()) -> bool:
    """``os.environ``, or a bare name bound by ``from os import environ``."""
    if isinstance(node, ast.Attribute):
        value = node.value
        return (
            node.attr == "environ" and isinstance(value, ast.Name) and value.id == "os"
        )
    return isinstance(node, ast.Name) and node.id in names


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
        self.removed: set[str] = set()  # HOME or the pin, popped or deleted
        #: HOME or the pin present without being set here: kept from the running
        #: environment by a key filter, or brought in by a ``**base``.
        self.carried: set[str] = set()

    def derived(self) -> _Env:
        env = _Env(self.copy, self.unknown)
        env.keys, env.pin, env.popped = set(self.keys), self.pin, self.popped
        env.helper, env.removed = self.helper, set(self.removed)
        env.carried = set(self.carried)
        return env

    def remove(self, key: str) -> None:
        """A ``pop``/``del`` of HOME or the pin; for the pin, it undoes any pin,
        and any inherited one."""
        self.keys.discard(key)
        self.carried.discard(key)
        self.removed.add(key)
        if key == _KEYRING_KEY:
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

    TASK-42520 (G1): a fresh environment drops the conftest's null pin, and
    without HOME the child's ``Path.home()`` falls back to the password
    database: the account's home, which is all
    ``anthropic_subscription._home_is_account_home()`` checks before the
    Keychain read. So every mapping passed as ``env=`` to a recognised Python
    child must set the null pin AND HOME, or be a copy of the running
    environment (``{**os.environ}``, ``dict(os.environ)``,
    ``os.environ.copy()``, ``os.environ``, ``None``), which carries both. A
    ``pop``/``del`` of either undoes a copy. A comprehension or loop that
    copies from the running environment by key (``.items()`` with a key
    filter, or by name from ``os.environ[n]``) carries what its filter
    provably keeps (see ``keeps``).

    Recognised children: an argv (a list, a name bound to one, ``list(...)``
    or ``tuple(...)`` of one, a same-module function's returned list, a
    starred first element resolving to one) whose first element is
    ``sys.executable`` (also through ``str``/``os.fspath``/
    ``Path(...).resolve()``), a ``python*`` string, or a variable whose name
    contains ``python`` or is ``interpreter``; or one with ``-m`` among its
    leading options; or ``sh``/``bash -c`` with an f-string that interpolates
    ``sys.executable``; including through ``asyncio.to_thread`` and
    ``create_subprocess_exec``. An argv held in a parameter is not
    recognised: for example in the generic runners ``run_child_with_watchdog``,
    ``supervise``, ``_run_helper``, ``_spawn_owned_process`` and ``_run_child``.
    A ``-c`` child whose source pins the null backend in-process
    (``keyring.set_keyring(Keyring())`` with ``Keyring`` from
    ``keyring.backends.null``) before any ``tldw_chatbook`` import has the
    pin (G5), but still needs HOME.

    ``pinned`` is None for a mapping the scan cannot read: a parameter,
    another module's helper, a method's result, a closure's variable, a name
    bound by tuple unpacking. A same-module function's result is judged by
    what that function returns. Limits: the verdict is the mapping's state at
    the end of its scope, not at the call.
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
    denylist built from ``os.environ.items()`` is a fresh environment, which
    carries the pin only when its key filter provably keeps it. A table
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

    environ_names = frozenset(
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "os"
        for alias in node.names
        if alias.name == "environ"
    )

    def is_environ(node: ast.AST | None) -> bool:
        return _is_environ(node, environ_names)

    #: Module names a ``global`` statement or a module-level ``+=`` may rebind.
    rebound = {
        name
        for node in ast.walk(tree)
        if isinstance(node, ast.Global)
        for name in node.names
    } | {
        node.target.id
        for node in tree.body
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name)
    }
    bindings = [
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    ]
    rebound |= {name for name in bindings if bindings.count(name) > 1}

    def literal(node: ast.AST):
        """An immutable literal: a string, a tuple display, ``frozenset(...)``."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Tuple):
            return tuple(literal(item) for item in node.elts)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "frozenset"
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], (ast.Tuple, ast.List, ast.Set))
        ):
            return tuple(literal(item) for item in node.args[0].elts)
        raise ValueError(ast.unparse(node))

    def is_null(value: ast.AST) -> bool:
        if isinstance(value, ast.Name):
            return constants.get(value.id) == NULL_BACKEND
        return isinstance(value, ast.Constant) and value.value == NULL_BACKEND

    def resolve(node: ast.AST, values: dict[str, ast.AST]) -> ast.AST:
        if isinstance(node, ast.Name):
            return values.get(node.id) or module_values.get(node.id) or node
        return node

    class Everything:
        """``os.environ`` in a filter: every name may be there, as in a copy."""

        def __contains__(self, item: object) -> bool:
            return True

    def evaluate(
        node: ast.AST,
        name: str | None = None,
        key: str | None = None,
        scope: tuple[set[str], dict[str, ast.AST]] = (set(), {}),
    ):
        """A filter expression's value with ``name`` bound to ``key``; raises
        ValueError (or KeyError/TypeError) when it cannot be known.

        Only comparisons, ``and``/``or``/``not``, constants and displays,
        string methods, ``re.search``/``match``/``fullmatch`` with a constant
        pattern and ``... in os.environ`` (true, as for a copy) are evaluated.
        Another name counts only when the scope binds it to a string constant
        or, unbound in the scope (``scope`` is its bound names and values),
        when the module binds it once to a string, a tuple or ``frozenset(...)``
        that nothing can mutate or rebind."""
        local_names, local_values = scope

        def value(node: ast.AST):
            if is_environ(node):
                return Everything()
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, ast.Name):
                if node.id == name:
                    return key
                if node.id in local_names:
                    bound = local_values.get(node.id)
                    if isinstance(bound, ast.Constant) and isinstance(bound.value, str):
                        return bound.value
                    raise ValueError(node.id)
                if node.id in rebound:
                    raise ValueError(node.id)
                return literal(module_values[node.id])
            if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
                return tuple(value(item) for item in node.elts)
            if isinstance(node, ast.BoolOp):
                found = [bool(value(item)) for item in node.values]
                return all(found) if isinstance(node.op, ast.And) else any(found)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                return not value(node.operand)
            if isinstance(node, ast.Compare) and len(node.ops) == 1:
                left, right = value(node.left), value(node.comparators[0])
                return {
                    ast.Eq: lambda: left == right,
                    ast.NotEq: lambda: left != right,
                    ast.In: lambda: left in right,
                    ast.NotIn: lambda: left not in right,
                }[type(node.ops[0])]()
            if isinstance(node, ast.Call) and not node.keywords:
                func, args = node.func, [value(arg) for arg in node.args]
                if isinstance(func, ast.Name) and func.id in ("frozenset", "set"):
                    return tuple(args[0]) if args else ()
                if (
                    ast.unparse(func) in ("re.search", "re.match", "re.fullmatch")
                    and len(args) == 2
                    and all(isinstance(arg, str) for arg in args)
                ):
                    return getattr(re, func.attr)(*args) is not None
                if isinstance(func, ast.Attribute) and func.attr in (
                    "startswith",
                    "endswith",
                    "upper",
                    "lower",
                    "casefold",
                ):
                    target = value(func.value)
                    if isinstance(target, str):
                        return getattr(target, func.attr)(*args)
            raise ValueError(ast.unparse(node))

        return value(node)

    def keeps(
        conditions: list[ast.AST],
        name: str,
        key: str,
        names: ast.AST | None = None,
        scope: tuple[set[str], dict[str, ast.AST]] = (set(), {}),
    ) -> bool:
        """Whether a filter over the running environment provably keeps
        ``key``: the ``if`` of ``{name: v for name, v in os.environ.items() if
        ...}``, or of a copy by name (``for name in names: if ...: e[name] =
        os.environ[name]``), whose ``names`` must then hold ``key``. Anything
        ``evaluate`` cannot know counts as dropping ``key``."""
        try:
            if names is not None and key not in evaluate(names, scope=scope):
                return False
            return all(bool(evaluate(test, name, key, scope)) for test in conditions)
        except (ValueError, KeyError, TypeError):
            return False

    def never_reaches(
        conditions: list[ast.AST],
        name: str,
        key: str,
        names: ast.AST,
        scope: tuple[set[str], dict[str, ast.AST]],
    ) -> bool:
        """Whether ``for name in names: [if ...:]`` provably never runs its body
        with ``key``: ``names`` lacks it, or a guard is false for it."""
        for test, bind in ((names, False), *((test, True) for test in conditions)):
            try:
                found = (
                    evaluate(test, name, key, scope)
                    if bind
                    else evaluate(test, scope=scope)
                )
            except (ValueError, KeyError, TypeError):
                continue
            if (not bind and key not in found) or (bind and not found):
                return True
        return False

    def is_python(node: ast.AST, values: dict[str, ast.AST]) -> bool:
        for _ in range(4):  # str(Path(sys.executable).resolve()) and the like
            if ast.unparse(node) == "sys.executable":
                return True
            # An interpreter held in a variable: ``python``, ``venv_python``, ...
            if isinstance(node, (ast.Name, ast.Attribute)):
                name = getattr(node, "id", None) or node.attr
                if "python" in name.lower() or name == "interpreter":
                    return True
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

    def execs_python(argv: list) -> bool:
        """``["/bin/sh", "-c", f"... exec {sys.executable} ..."]``."""
        head, options = argv[0], flags(argv)
        if not (
            isinstance(head, ast.Constant)
            and isinstance(head.value, str)
            and Path(head.value).name in ("sh", "bash", "zsh", "dash")
            and options[-1:] == ["-c"]
            and len(argv) > len(options) + 1
        ):
            return False
        command = argv[len(options) + 1]
        return isinstance(command, ast.JoinedStr) and any(
            isinstance(part, ast.FormattedValue)
            and ast.unparse(part.value) == "sys.executable"
            for part in command.values
        )

    def argv_of(node: ast.AST, values: dict[str, ast.AST], depth: int = 0):
        """An argv expression's elements and the names to read them with:
        a list, a name bound to one, ``list(...)``/``tuple(...)`` of one, a
        same-module function's returned list, a starred first element."""
        node = resolve(node, values)
        if depth > 6:
            return None
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in ("list", "tuple") and len(node.args) == 1:
                return argv_of(node.args[0], values, depth + 1)
            for returned, helper_values in argv_returns.get(node.func.id, ()):
                found = argv_of(returned, helper_values, depth + 1)
                if found:
                    return found
            return None
        if not isinstance(node, (ast.List, ast.Tuple)) or not node.elts:
            return None
        elts = list(node.elts)
        if isinstance(elts[0], ast.Starred):
            head = argv_of(elts[0].value, values, depth + 1)
            return (head[0] + elts[1:], head[1]) if head else None
        return elts, values

    def python_argv(call: ast.Call, values: dict[str, ast.AST]):
        """The argv of the Python child ``call`` starts, and the names to read
        it with, or None."""
        args = (
            call.args[1:]
            if getattr(call.func, "attr", None) == "to_thread"
            else call.args
        )
        if not args:
            return None
        argv, argv_values = argv_of(args[0], values) or (list(args), values)
        if "-m" in flags(argv) or is_python(argv[0], argv_values) or execs_python(argv):
            return argv, argv_values
        return None

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
    candidates = []  # (scope, line, mapping, call, names) for each ``env=`` call
    returns: dict[str, list[_Env | None]] = {}
    argv_returns: dict[str, list[tuple[ast.AST, dict[str, ast.AST]]]] = {}
    for scope in scopes:
        nodes = list(_own_nodes(scope))
        envs: dict[object, _Env] = {}
        handled: set[int] = set()
        values: dict[str, ast.AST] = {}
        arguments = getattr(scope, "args", None)
        local = (  # the names this scope binds, for ``keeps``
            {
                node.id
                for node in nodes
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
            }
            | {
                arg.arg
                for arg in (
                    [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
                    + [arg for arg in (arguments.vararg, arguments.kwarg) if arg]
                    if arguments
                    else []
                )
            },
            values,
        )

        def env_of(target: ast.AST) -> _Env:
            key = ast.unparse(target)
            if key not in envs:
                environ = is_environ(target)
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
            if is_environ(func.value):
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
                        base = known_env(value) or _Env(unknown=True)
                        env.copy = env.copy or base.copy
                        env.unknown = env.unknown or base.unknown
                        env.helper = env.helper or base.helper
                        env.popped = env.popped or base.popped
                        env.pin = base.pin or env.pin
                        env.removed |= base.removed
                        env.carried |= base.carried | (base.keys & {"HOME"})
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

        def comp_env(comp: ast.DictComp) -> _Env:
            """A comprehension is fresh; one that filters ``os.environ.items()``
            by key, or copies names from it, carries HOME and the pin when it
            provably keeps them."""
            env, generator = _Env(), comp.generators[0]
            target, iterable = generator.target, generator.iter
            if (
                len(comp.generators) == 1
                and isinstance(target, ast.Name)
                and ast.unparse(comp.key) == target.id
                and isinstance(comp.value, ast.Subscript)
                and is_environ(comp.value.value)
                and ast.unparse(comp.value.slice) == target.id
            ):
                env.carried = {
                    key
                    for key in ("HOME", _KEYRING_KEY)
                    if keeps(generator.ifs, target.id, key, iterable, local)
                }
            elif (
                len(comp.generators) == 1
                and isinstance(target, ast.Tuple)
                and len(target.elts) == 2
                and ast.unparse(comp.key) == ast.unparse(target.elts[0])
                and ast.unparse(comp.value) == ast.unparse(target.elts[1])
                and isinstance(target.elts[0], ast.Name)
                and getattr(getattr(iterable, "func", None), "attr", None) == "items"
                and is_environ(iterable.func.value)
            ):
                name = target.elts[0].id
                env.carried = {
                    key
                    for key in ("HOME", _KEYRING_KEY)
                    if keeps(generator.ifs, name, key, scope=local)
                }
            return env

        def copied_by_name(loop: ast.For) -> None:
            """``for n in names: [if ...:] e[n] = os.environ[n]`` carries into
            ``e`` whichever of HOME and the pin it provably copies. The body
            must be that assignment alone, or one ``if`` with no ``else`` whose
            body is that assignment alone: a ``continue`` or ``break`` could
            skip it."""
            if not isinstance(loop.target, ast.Name) or loop.orelse:
                return
            if len(loop.body) != 1:
                return
            name, statement = loop.target.id, loop.body[0]
            conditions = []
            if isinstance(statement, ast.If):
                if statement.orelse or len(statement.body) != 1:
                    return
                conditions, statement = [statement.test], statement.body[0]
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Subscript)
                and ast.unparse(statement.targets[0].slice) == name
                and isinstance(statement.value, ast.Subscript)
                and is_environ(statement.value.value)
                and ast.unparse(statement.value.slice) == name
            ):
                env_of(statement.targets[0].value).carried |= {
                    key
                    for key in ("HOME", _KEYRING_KEY)
                    if keeps(conditions, name, key, loop.iter, local)
                }

        #: A removal directly in a ``for`` loop's body, or in one ``if`` there:
        #: id(removal) -> (loop, the guards it sits under).
        loop_removals: dict[int, tuple[ast.For, list[ast.AST]]] = {}
        for loop in nodes:
            if isinstance(loop, ast.For) and isinstance(loop.target, ast.Name):
                for statement in loop.body:
                    guarded = isinstance(statement, ast.If)
                    for inner in statement.body if guarded else [statement]:
                        found = inner.value if isinstance(inner, ast.Expr) else inner
                        loop_removals[id(found)] = (
                            loop,
                            [statement.test] if guarded else [],
                        )

        def removed(mapping: ast.AST, key: ast.AST, node: ast.AST) -> None:
            """A ``pop``/``del``: a constant key drops itself if it is HOME or
            the pin. Another key may be either, so it drops both, unless it is
            the variable of a loop that provably never reaches one of them."""
            if isinstance(key, ast.Constant):
                if key.value in (_KEYRING_KEY, "HOME"):
                    env_of(mapping).remove(key.value)
                return
            loop, guards = loop_removals.get(id(node), (None, []))
            for dropped in ("HOME", _KEYRING_KEY):
                if not (
                    loop is not None
                    and isinstance(key, ast.Name)
                    and key.id == loop.target.id
                    and never_reaches(guards, key.id, dropped, loop.iter, local)
                ):
                    env_of(mapping).remove(dropped)

        def known_env(node: ast.AST) -> _Env | None:
            """The mapping an expression names, when this scope can read it."""
            if is_environ(node):
                return _Env(copy=True)
            if isinstance(node, ast.DictComp):
                return comp_env(node)
            return envs.get(ast.unparse(node))

        def result_env(value: ast.AST) -> _Env:
            """A mapping that does not copy one: a comprehension is fresh, a
            same-module function's result is that function's, the rest unknown."""
            if isinstance(value, ast.DictComp):
                return comp_env(value)
            env = _Env(unknown=True)
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
                    known = known_env(base)
                    return known.derived() if known else _Env(unknown=True)
                if isinstance(func, ast.Attribute) and func.attr == "copy":
                    if is_environ(func.value):
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
                copied_by_name(node)
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
                    if isinstance(target, ast.Subscript):
                        removed(target.value, target.slice, node)
            elif isinstance(node.func, ast.Attribute) and id(node) not in handled:
                if node.func.attr == "update":
                    fill(env_of(node.func.value), node)
                elif node.func.attr == "pop" and node.args:
                    removed(node.func.value, node.args[0], node)
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
                    env.pin is None
                    and not env.popped
                    and pytest_file
                    and (env.copy or _KEYRING_KEY in env.carried)
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
            if is_environ(value):
                return _Env(copy=True)
            if isinstance(value, (ast.Name, ast.Attribute)):
                return envs.get(ast.unparse(value))
            start = source_env(value)
            if start is not None:
                fill(start, value)
                return start
            return result_env(value)

        name = getattr(scope, "name", "<module>")
        for node in nodes:
            if isinstance(node, ast.Return) and node.value is not None:
                returns.setdefault(name, []).append(mapping_at(node.value))
                argv_returns.setdefault(name, []).append((node.value, values))
            elif isinstance(node, ast.Call):
                env_value = next(
                    (k.value for k in node.keywords if k.arg == "env"), None
                )
                if env_value is not None:
                    mapping = mapping_at(env_value)
                    candidates.append((name, env_value.lineno, mapping, node, values))

    def combined(found: list[bool | None]) -> bool | None:
        if any(item is False for item in found):
            return False
        return True if found and all(found) else None

    def parts(env: _Env | None, seen: tuple[str, ...] = ()):
        """(pinned, has HOME) for a mapping, each None when it cannot be read."""
        if env is None:
            return None, None
        if env.helper is not None:
            found = returns.get(env.helper) if env.helper not in seen else None
            each = [parts(result, (*seen, env.helper)) for result in found or ()]
            base = (combined([p for p, _ in each]), combined([h for _, h in each]))
        elif env.copy:
            base = (True, True)
        else:
            base = (None, None) if env.unknown else (False, False)
        if env.pin is not None:
            pin = env.pin == "null"
        elif env.popped:
            pin = False
        else:
            pin = True if _KEYRING_KEY in env.carried else base[0]
        if "HOME" in env.keys or "HOME" in env.carried:
            home = True
        else:
            home = False if "HOME" in env.removed else base[1]
        return pin, home

    spawns = []
    for name, line, mapping, call, values in candidates:
        found = python_argv(call, values)
        if found is None:
            continue
        argv, argv_values = found
        child, options = None, flags(argv)
        if options[-1:] == ["-c"] and len(argv) > len(options) + 1:
            child = _python_source(resolve(argv[len(options) + 1], argv_values))
        pin, home = parts(mapping)
        if _pins_in_process(child):
            pin = True
        spawns.append((name, line, False if False in (pin, home) else pin and home))
    return moves, sorted(spawns, key=lambda spawn: spawn[1])


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
_HOME = '"HOME": h'
_FRESH = 'env={"PATH": p}'
_HOMED = 'env={"PATH": p, "HOME": h}'
_CHILD_PIN = (
    "import keyring\nfrom keyring.backends.null import Keyring\n"
    "keyring.set_keyring(Keyring())\n"
)


def _filtered(condition: str) -> str:
    """A spawn whose environment filters ``os.environ.items()`` by key."""
    return (
        f"def f():\n    e = {{k: v for k, v in os.environ.items() if {condition}}}\n"
        "    subprocess.run([sys.executable, 'x.py'], env=e)\n"
    )


@pytest.mark.parametrize(
    ("source", "pinned"),
    [
        # G1: a fresh environment needs the null pin AND HOME. Without HOME the
        # child's Path.home() is the account's, which opens the Keychain read.
        (_RUN + f'[sys.executable, "-c", s], {_FRESH})\n', [False]),
        (_RUN + f'[sys.executable, "-c", s], env={{"PATH": p, {_PIN}}})\n', [False]),
        (_RUN + f'[sys.executable, "-c", s], {_HOMED})\n', [False]),
        (_RUN + f'[sys.executable, "-c", s], env={{{_HOME}, {_PIN}}})\n', [True]),
        (
            f'def f(p, h):\n    e = {{"PATH": p, "HOME": h}}\n    e["{_KEYRING_KEY}"] = {_NULL}\n'
            '    subprocess.run([sys.executable, "-c", s], env=e)\n',
            [True],
        ),
        # A copy of the running environment, or no env, inherits both.
        (_RUN + '[sys.executable, "-c", s], env=dict(os.environ))\n', [True]),
        (_RUN + '[sys.executable, "-c", s], env={**os.environ, "X": "1"})\n', [True]),
        (_RUN + '[sys.executable, "-c", s], env=os.environ.copy())\n', [True]),
        (_RUN + '[sys.executable, "-c", s], env=None)\n', [True]),
        # ... unless it pops one of them.
        (
            f'def f():\n    e = os.environ.copy()\n    e.pop("{_KEYRING_KEY}")\n'
            "    subprocess.run([sys.executable], env=e)\n",
            [False],
        ),
        (
            'def f():\n    e = os.environ.copy()\n    del e["HOME"]\n'
            "    subprocess.run([sys.executable], env=e)\n",
            [False],
        ),
        # A key filter over os.environ.items() carries what it provably keeps.
        (_filtered("k != 'X'"), [True]),
        (_filtered("k in ('PATH', 'HOME')"), [False]),
        (_filtered("not k.startswith(('PYTHON', 'X'))"), [False]),
        (
            f'A = frozenset({{"PATH", "HOME", "{_KEYRING_KEY}"}})\n'
            + _filtered("k in A"),
            [True],
        ),
        (_filtered("k in allowed()"), [False]),
        # ... and so does a copy by name, in a comprehension or a loop.
        (
            f"def f():\n    e = {{n: os.environ[n] for n in ('PATH', 'HOME', '{_KEYRING_KEY}')"
            " if n in os.environ}\n    subprocess.run([sys.executable], env=e)\n",
            [True],
        ),
        (
            "def f():\n    e = {n: os.environ[n] for n in ('PATH', 'HOME')}\n"
            "    subprocess.run([sys.executable], env=e)\n",
            [False],
        ),
        (
            f"def f():\n    e = {{'PATH': 'x'}}\n    for n in ('HOME', '{_KEYRING_KEY}'):\n"
            "        if n in os.environ:\n            e[n] = os.environ[n]\n"
            "    subprocess.run([sys.executable], env=e)\n",
            [True],
        ),
        (
            f"def f():\n    e = {{'PATH': 'x'}}\n    for n in ('HOME', '{_KEYRING_KEY}'):\n"
            "        if n != 'HOME':\n            e[n] = os.environ[n]\n"
            "    subprocess.run([sys.executable], env=e)\n",
            [False],
        ),
        # Review round 2: what the evaluator must not claim (each was True).
        (  # a scope name shadowing the module constant
            f'A = ("PATH", "HOME", "{_KEYRING_KEY}")\n'
            + _filtered("k in A").replace(
                "def f():\n", "def f():\n    A = ('PATH',)\n"
            ),
            [False],
        ),
        (  # a parameter shadowing it
            f'A = ("PATH", "HOME", "{_KEYRING_KEY}")\n'
            + _filtered("k in A").replace("def f():", "def f(A):"),
            [False],
        ),
        (  # a module constant a function rebinds
            f'A = ("PATH", "HOME", "{_KEYRING_KEY}")\n'
            "def g():\n    global A\n    A = ('PATH',)\n" + _filtered("k in A"),
            [False],
        ),
        (  # a module list mutated after binding
            f'A = ["PATH", "HOME", "{_KEYRING_KEY}"]\nA.remove("HOME")\n'
            + _filtered("k in A"),
            [False],
        ),
        (  # a continue that skips the copy
            f"def f():\n    e = {{'PATH': 'x'}}\n    for n in ('HOME', '{_KEYRING_KEY}'):\n"
            "        if n == 'HOME':\n            continue\n        e[n] = os.environ[n]\n"
            "    subprocess.run([sys.executable], env=e)\n",
            [False],
        ),
        (  # a break before it
            f"def f():\n    e = {{'PATH': 'x'}}\n    for n in ('HOME', '{_KEYRING_KEY}'):\n"
            "        e[n] = os.environ[n]\n        break\n"
            "    subprocess.run([sys.executable], env=e)\n",
            [False],
        ),
        (  # another object's .environ is not the running environment
            "def f(s):\n    e = {k: v for k, v in s.environ.items() if k != 'X'}\n"
            "    subprocess.run([sys.executable], env=e)\n",
            [False],
        ),
        (  # ``from os import environ`` is
            "from os import environ\n"
            "def f():\n    subprocess.run([sys.executable], env=dict(environ))\n",
            [True],
        ),
        (  # a pop by a loop variable that reaches HOME
            "def f():\n    e = os.environ.copy()\n    for n in ('HOME',):\n"
            "        e.pop(n, None)\n    subprocess.run([sys.executable], env=e)\n",
            [False],
        ),
        (  # a del by a key the scan cannot read
            "def f(k):\n    e = os.environ.copy()\n    del e[k]\n"
            "    subprocess.run([sys.executable], env=e)\n",
            [False],
        ),
        (  # a guard that matches the pin's name
            "def f():\n    e = os.environ.copy()\n    for n in tuple(e):\n"
            "        if re.search('KEY|TOKEN', n):\n            del e[n]\n"
            "    subprocess.run([sys.executable], env=e)\n",
            [False],
        ),
        # ... while removals that provably miss both keep the copy whole.
        (
            "def f():\n    e = os.environ.copy()\n    for n in ('PYTEST_ADDOPTS',):\n"
            "        e.pop(n, None)\n    subprocess.run([sys.executable], env=e)\n",
            [True],
        ),
        (
            "def f():\n    e = os.environ.copy()\n    for n in tuple(e):\n"
            "        if n.startswith('COV_CORE_'):\n            e.pop(n)\n"
            "    subprocess.run([sys.executable], env=e)\n",
            [True],
        ),
        # ``e = {**e, ...}`` keeps what ``e`` was; a parameter is not judged.
        (
            "def f(e):\n    e = {**e, 'X': '1'}\n"
            "    subprocess.run([sys.executable], env=e)\n",
            [None],
        ),
        # Each argv shape that starts Python; other programs are not counted.
        (
            'def f(p):\n    cmd = [sys.executable, "-c", s]\n'
            f"    subprocess.Popen(cmd, {_FRESH})\n",
            [False],
        ),
        (_RUN + f'[str(Path(sys.executable).resolve()), "x.py"], {_FRESH})\n', [False]),
        (_RUN + f'["python3", "x.py"], {_FRESH})\n', [False]),
        (_RUN + f'[tool, "-m", "pytest"], {_FRESH})\n', [False]),
        (_RUN + f'[tool, "-I", "-m", "pytest"], {_FRESH})\n', [False]),
        (_RUN + f'[str(python), "-I", "-c", s], {_FRESH})\n', [False]),
        (_RUN + f'[str(interpreter), "x.py"], {_FRESH})\n', [False]),
        (_RUN + f'[runtime.python_executable, "x.py"], {_FRESH})\n', [False]),
        (
            "def child_command():\n    return [sys.executable, '-I', 'x.py']\n"
            f"def f(p):\n    subprocess.run(child_command(), {_FRESH})\n",
            [False],
        ),
        (
            "def f(p):\n    cmd = [sys.executable, '-m', 'child']\n"
            f"    subprocess.run([*cmd, 'go'], {_FRESH})\n",
            [False],
        ),
        (
            "def f(p):\n    cmd = (sys.executable, 'x.py')\n"
            f"    subprocess.Popen(list(cmd), {_FRESH})\n",
            [False],
        ),
        (
            _RUN
            + f'["/bin/sh", "-c", f"umask 002; exec {{sys.executable}} -"], {_FRESH})\n',
            [False],
        ),
        (_RUN + f'["/bin/sh", "-c", f"ls {{d}}"], {_FRESH})\n', []),
        (
            "async def f(p):\n    await asyncio.create_subprocess_exec("
            f'sys.executable, "-c", s, {_FRESH})\n',
            [False],
        ),
        (
            "async def f(p):\n    await asyncio.to_thread("
            f'subprocess.run, [sys.executable, "x.py"], {_FRESH})\n',
            [False],
        ),
        (_RUN + f'["git", "status"], {_FRESH})\n', []),
        # A same-module helper's result is judged by what it returns.
        (
            'def env():\n    return {"PATH": "x"}\n'
            "def f():\n    subprocess.run([sys.executable], env=env())\n",
            [False],
        ),
        (
            f'def env(h):\n    return {{"PATH": "x", {_HOME}, {_PIN}}}\n'
            "def f():\n    e = env(h)\n    subprocess.run([sys.executable], env=e)\n",
            [True],
        ),
        (
            "def f():\n    subprocess.run([sys.executable], env=imported_env())\n",
            [None],
        ),
        # G5: a -c child that pins the null backend in-process before any app
        # import stands for the env pin, never for HOME.
        (
            f"CHILD = {_CHILD_PIN + 'import tldw_chatbook'!r}\n"
            + _RUN
            + f'[sys.executable, "-c", CHILD, "mode"], {_HOMED})\n',
            [True],
        ),
        (
            f"CHILD = {_CHILD_PIN!r}\n"
            + _RUN
            + f'[sys.executable, "-c", CHILD], {_FRESH})\n',
            [False],
        ),
        (
            f"CHILD = {'import tldw_chatbook' + chr(10) + _CHILD_PIN!r}\n"
            + _RUN
            + f'[sys.executable, "-c", CHILD], {_HOMED})\n',
            [False],
        ),
        (
            f"CHILD = {_CHILD_PIN.replace('null', 'macOS')!r}\n"
            + _RUN
            + f'[sys.executable, "-c", CHILD], {_HOMED})\n',
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
            + f'[sys.executable, "-c", CHILD], {_HOMED})\n',
            [False],
        ),
        (  # options before -c
            f"CHILD = {_CHILD_PIN!r}\n"
            + _RUN
            + f'[sys.executable, "-I", "-B", "-c", CHILD], {_HOMED})\n',
            [True],
        ),
        # An f-string child (its fields read as None) or a dedented constant.
        (
            _RUN
            + '[sys.executable, "-c", f'
            + repr(_CHILD_PIN + "x = {y!r}")
            + f"], {_HOMED})\n",
            [True],
        ),
        (
            f"CHILD = textwrap.dedent({_CHILD_PIN!r})\n"
            + _RUN
            + f'[sys.executable, "-c", CHILD], {_HOMED})\n',
            [True],
        ),
        # Only a -c source is read: a script path's pin is not seen.
        (_RUN + f'[sys.executable, "child.py"], {_HOMED})\n', [False]),
    ],
)
def test_the_census_reads_each_recognised_python_child_and_its_environment(
    source: str, pinned: list[bool | None]
) -> None:
    """The negative controls for G1 and G5 (TASK-42520)."""
    assert [spawn[2] for spawn in _python_spawns(source)] == pinned


@functools.cache
def _census() -> tuple:
    """(moves, unpinned moves, exemptions seen, Python spawns, failing spawns,
    spawn exemptions seen, spawns not judged)."""
    moves, unpinned, exempt_seen = 0, [], set()
    spawns, unpinned_spawns, spawn_exempt_seen, unjudged = 0, [], set(), []
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
                if (relative, scope) in _SPAWN_EXEMPT:
                    spawn_exempt_seen.add((relative, scope))
                elif pinned is False:
                    unpinned_spawns.append(f"{relative}:{line} {scope}")
                elif pinned is None:
                    unjudged.append(f"{relative}:{line} {scope}")
    return (
        moves,
        unpinned,
        exempt_seen,
        spawns,
        unpinned_spawns,
        spawn_exempt_seen,
        unjudged,
    )


def test_each_recognised_python_child_with_a_fresh_environment_gets_the_pin_and_a_home() -> (
    None
):
    """G1 (TASK-42520): a fresh ``env=`` drops the conftest's pin, and without
    HOME the child's ``Path.home()`` is the account's, which is all the
    Keychain read checks. Only the argv shapes `_python_spawns` names count."""
    *_, spawns, unpinned, exempt_seen, unjudged = _census()

    assert spawns >= _MIN_PYTHON_SPAWNS, f"the census saw only {spawns} Python spawns"
    assert not unpinned, (
        "these start Python with a fresh environment that lacks "
        "PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring (the child resolves "
        "the OS store) or HOME (the child reaches the account's home):\n"
        + "\n".join(unpinned)
    )
    assert exempt_seen == set(_SPAWN_EXEMPT), (
        f"stale exemptions: {set(_SPAWN_EXEMPT) - exempt_seen}"
    )
    found = Counter((site.split(":")[0], site.split(" ", 1)[1]) for site in unjudged)
    assert found == Counter(_UNJUDGED_SPAWNS), (
        "the Python spawns whose environment the census cannot read changed. "
        "Check that each new one pins the null backend and sets HOME, then "
        f"update _UNJUDGED_SPAWNS.\nnew: {found - Counter(_UNJUDGED_SPAWNS)}\n"
        f"gone: {Counter(_UNJUDGED_SPAWNS) - found}"
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

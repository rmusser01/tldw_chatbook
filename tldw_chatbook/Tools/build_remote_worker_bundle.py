"""Generate the single-file remote worker bundle (Phase 1d, Task 8).

``python -m tldw_chatbook.Tools.build_remote_worker_bundle`` regenerates
``Tools/remote_worker_bundle.py`` — a COMMITTED artifact — from the real
dependency modules of the pinned workspace worker. The bundle is what a
remote host executes: one flat stdlib-only module, targeting Python 3.10+,
whose ``main(stream)`` entry receives the buffered stdin AFTER the remote
bootstrap has consumed the bundle's own bytes (Task 9's harness positions
the stream).

Design decisions the drift guard pins:

* **Explicit module list, never package-``__init__`` traversal** (Task 4
  ledger carry): the root ``tldw_chatbook/__init__`` and
  ``tldw_chatbook/Tools/__init__`` do app-level env/loguru/tiktoken/tool-
  executor work that must never reach a remote host. ``BUNDLE_MODULES``
  below is the frozen true closure of
  ``tldw_chatbook.Tools.workspace_tool_worker``; the builder re-derives
  that closure in a fresh isolated interpreter on every run and refuses
  to build if the two disagree, so a newly added dependency fails the
  build instead of silently missing from the bundle.
* **Extract, not copy-paste**: each section is produced from
  ``inspect.getsource`` of the imported module, transformed via ``ast``,
  and unparsed — the shipped code IS the real module's code.
* **Flat namespace**: cross-module ``tldw_chatbook.*`` imports are
  dropped (the concatenation order provides those names); a build-time
  collision check refuses duplicate top-level names so ordering can
  never silently shadow anything.
* **Non-stdlib lazy imports become loud raises**: the closure's modules
  keep parent-side lazy imports (loguru, ``..Metrics``,
  ``..config``, portalocker, Skills/RAG resolvers) — none of which exist
  on a bare remote interpreter. The builder rewrites every remaining
  import statement whose root is outside ``sys.stdlib_module_names``
  into ``raise ImportError("... is not available inside the remote
  worker bundle")``. Source-level ``try/except ImportError`` guards
  (installed in Phase 0/1d: telemetry no-ops, the loguru no-op logger,
  the fcntl lock fallback) catch that raise and degrade gracefully on
  paths the bundle actually executes; any dormant path that would have
  needed the real dependency fails loudly instead of misbehaving.
* **IO adapter is bundle-only** (spec's stdout-contamination rule): the
  local worker emits bare frames because its pipe has no noise source;
  the remote bundle prefixes every response with ``RESPONSE_MAGIC``
  (exactly 16 bytes, defined once, in the adapter below).

The builder itself is stdlib-only; it runs at commit time in the parent
environment. Output is deterministic (no timestamps, stable section
order), which is what the byte-equality drift guard in
``Tests/Tools/test_remote_worker_bundle.py`` relies on.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any

#: The bundle's target interpreter floor (spec): a remote host may run a
#: bare system python3 as old as 3.10. CI + the drift suite enforce this
#: by parsing the artifact under a real 3.10.
REMOTE_PYTHON_FLOOR = (3, 10)

#: Frozen true import closure of ``workspace_tool_worker``, excluding
#: package ``__init__`` modules, in dependency order (dependencies first,
#: so every dropped cross-module import finds its name already bound).
BUNDLE_MODULES: tuple[str, ...] = (
    "tldw_chatbook.Utils.filesystem_identity",
    "tldw_chatbook.Utils.path_validation",
    "tldw_chatbook.Utils.sensitive_paths",
    "tldw_chatbook.Tools.workspace_wire_decode",
    "tldw_chatbook.Tools.local_tool_impls",
    "tldw_chatbook.Tools.workspace_root_pin",
    "tldw_chatbook.Tools.git_tool_impls",
    "tldw_chatbook.Tools.patch_tool_impls",
    "tldw_chatbook.Tools.workspace_tool_dispatch",
    "tldw_chatbook.Tools.workspace_tool_worker",
)

#: Data-only module whose contents the bundle embeds verbatim: the worker
#: does not import it, so it is not part of the closure above — the
#: builder renders its data into the artifact (see ``_denylist_section``).
REMOTE_DENYLIST_MODULE = "tldw_chatbook.Tools.remote_sensitive_paths"

_WORKER_MODULE = "tldw_chatbook.Tools.workspace_tool_worker"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_BUNDLE_PATH = Path(__file__).resolve().parent / "remote_worker_bundle.py"

_BUNDLE_DOCSTRING = '''"""Single-file remote workspace worker — GENERATED, COMMITTED ARTIFACT.

Produced by ``python -m tldw_chatbook.Tools.build_remote_worker_bundle``
from the real dependency modules listed in its ``BUNDLE_MODULES``. Do not
edit by hand: regenerate instead, or the drift guard
(``Tests/Tools/test_remote_worker_bundle.py``) fails on the next run.

Contract:

* Runs on a bare remote interpreter, Python {floor} or newer, with ONLY
  the standard library available — at import time AND at runtime. Every
  import root in this file is stdlib; the builder rewrote the parent's
  lazy third-party imports into loud ``ImportError`` raises that the
  source-level guards degrade around.
* ``main(stream)`` is the entry: ``stream`` is the buffered stdin
  positioned AFTER the bootstrap consumed this bundle's own bytes.
  Loaders that ``exec`` this file must register the executing namespace
  in ``sys.modules`` first — the closure's ``dataclass(slots=True)``
  classes resolve their defining module through it. The fixed remote
  bootstrap needs no such registration: it execs this file inside the
  interpreter's own ``__main__`` namespace, and the artifact's FINAL
  line — the ``BUNDLE_SHA256`` assignment — triggers
  ``_enter_worker_exchange`` (defined above, inside the stamped region),
  which runs the one exchange and propagates the worker exit code.
* Every response frame is emitted as ``RESPONSE_MAGIC + <json frame>``;
  use ``split_magic(raw)`` to strip before parsing. The LOCAL worker
  does not add this prefix — its pipe has no noise source.
* The ``ping`` operation (Task 9) is dispatched BEFORE the root pin: it
  captures the full root-to-``/`` directory identity chain, canonical
  path, remote python version, and ``BUNDLE_SHA256`` so a first-contact
  caller can build every other operation's pinned request.
* ``BUNDLE_SHA256`` (the final line) is the SHA-256 of this file's
  bytes ABOVE its own assignment line — a full-file digest is not
  self-embeddable (the stamp would change its own input). Every
  executable byte, including the bootstrap entry logic, lives in that
  stamped region; only the assignment's own line falls outside it (and
  under the bootstrap the entry helper's ``SystemExit`` fires from that
  line, so nothing after it could ever run). Derive the same value from
  the artifact with
  ``build_remote_worker_bundle.expected_bundle_stamp``.
* ``arm_watchdog`` is an unimplemented seam (Task 12); the bundle arms
  nothing yet.
* ``REMOTE_SENSITIVE_PATHS`` (embedded from
  ``Tools/remote_sensitive_paths.py``) are remote-home-relative paths the
  worker must never touch; enforcement wiring lands with the remote
  binding tasks.
"""'''


class BundleBuildError(RuntimeError):
    """Raised when the bundle cannot be built from the current sources."""


def _is_package_init(module: Any) -> bool:
    file = getattr(module, "__file__", None)
    return not file or str(file).endswith("__init__.py")


def _worker_closure_modules() -> set[str]:
    """Derive the worker's real dependency closure by AST traversal.

    Walks module-level absolute ``tldw_chatbook.*`` imports breadth-first,
    starting at the worker, and NEVER traverses into package ``__init__``
    modules: importing ``tldw_chatbook.Tools.workspace_tool_worker``
    transitively executes the root and ``Tools``/``Utils`` package inits
    (which pull ``tool_executor``/``tiktoken_runtime``), but those chains
    are exactly what ruling 2 excludes — no real dependency module
    references their names, so the bundle must not ship them.

    Function-level imports are deliberately not followed: the bundle's
    lazy third-party imports are rewritten to loud raises, not shipped.
    """
    visited: set[str] = set()
    frontier = [_WORKER_MODULE]
    while frontier:
        name = frontier.pop()
        if name in visited:
            continue
        module = importlib.import_module(name)
        visited.add(name)
        if _is_package_init(module):
            # Packages are imported as a side effect of their children;
            # their own eager imports are app-side, not worker deps.
            continue
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        for node in tree.body:
            targets: list[str] = []
            if isinstance(node, ast.Import):
                targets = [
                    alias.name
                    for alias in node.names
                    if alias.name.split(".")[0] == "tldw_chatbook"
                ]
            elif (
                isinstance(node, ast.ImportFrom)
                and node.level == 0
                and (node.module or "").split(".")[0] == "tldw_chatbook"
            ):
                targets = [node.module or ""]
                for alias in node.names:
                    # ``from package import submodule`` names a real module.
                    candidate = f"{node.module}.{alias.name}"
                    try:
                        spec = importlib.util.find_spec(candidate)
                    except (ImportError, ValueError, AttributeError):
                        spec = None
                    if spec is not None and spec.origin not in (None, "namespace"):
                        targets.append(candidate)
            for target in targets:
                if target in visited:
                    continue
                target_module = importlib.import_module(target)
                if not _is_package_init(target_module):
                    frontier.append(target)
    return visited


def _assert_frozen_list_matches_closure() -> None:
    """Refuse to build when the true closure and ``BUNDLE_MODULES`` differ."""
    closure = _worker_closure_modules()
    frozen = set(BUNDLE_MODULES)
    if closure != frozen:
        missing = sorted(closure - frozen)
        extra = sorted(frozen - closure)
        raise BundleBuildError(
            "BUNDLE_MODULES has drifted from the worker's real import "
            "closure (new modules must be audited for bundle safety, then "
            f"added to the frozen list): missing from list: {missing}; "
            f"listed but not imported: {extra}"
        )


def _is_main_guard(node: ast.stmt) -> bool:
    if not isinstance(node, ast.If):
        return False
    test = node.test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


def _is_tldw_import_from(node: ast.stmt) -> bool:
    """Whether ``node`` is a module-level ``from tldw_chatbook...`` import."""
    if isinstance(node, ast.ImportFrom):
        if node.level > 0:
            return False  # a module-level relative import would not even import
        return bool(node.module) and node.module.split(".")[0] == "tldw_chatbook"
    return False


def _split_tldw_aliases(node: ast.Import) -> tuple[list[ast.alias], list[ast.alias]]:
    """Partition one ``import a, b`` node into (kept, dropped) aliases.

    ``import os, tldw_chatbook.x`` must lose only its tldw alias — the
    stdlib alias keeps its import instead of the whole node being dropped.
    """
    kept = [
        alias for alias in node.names if alias.name.split(".")[0] != "tldw_chatbook"
    ]
    dropped = [alias for alias in node.names if alias.name.split(".")[0] == "tldw_chatbook"]
    return kept, dropped


def _is_all_assignment(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "__all__"
    )


def _is_future_import(node: ast.stmt) -> bool:
    return isinstance(node, ast.ImportFrom) and node.module == "__future__"


class _NonStdlibImportRewriter(ast.NodeTransformer):
    """Rewrite every non-stdlib import into a loud ``ImportError`` raise.

    Runs AFTER module-level tldw imports were dropped, so anything it
    finds is a function-level lazy import. The message names the lost
    dependency; source-level ``try/except ImportError`` guards around the
    rewritten statement degrade gracefully, everything else fails loudly
    on first execution instead of silently misbehaving.
    """

    def _raise_stub(self, description: str) -> ast.Raise:
        message = (
            f"{description} is not available inside the remote worker bundle"
        )
        return ast.Raise(
            exc=ast.Call(
                func=ast.Name(id="ImportError", ctx=ast.Load()),
                args=[ast.Constant(value=message)],
                keywords=[],
            ),
            cause=None,
        )

    def visit_Import(self, node: ast.Import) -> Any:
        roots = [alias.name for alias in node.names]
        non_stdlib = [
            root for root in roots if root.split(".")[0] not in sys.stdlib_module_names
        ]
        if not non_stdlib:
            return node
        stdlib_aliases = [
            alias
            for alias in node.names
            if alias.name.split(".")[0] in sys.stdlib_module_names
        ]
        statements: list[ast.stmt] = []
        if stdlib_aliases:
            statements.append(ast.Import(names=stdlib_aliases))
        statements.extend(self._raise_stub(name) for name in non_stdlib)
        return statements

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        target = "." * node.level + (node.module or "")
        if target.split(".")[0] in sys.stdlib_module_names and node.level == 0:
            return node
        names = ", ".join(alias.name for alias in node.names)
        return self._raise_stub(f"'{target}' (importing {names})")


def _module_section(module_name: str) -> tuple[str, set[str]]:
    """Render one module as a flat bundle section plus its top-level names."""
    module = importlib.import_module(module_name)
    source = inspect.getsource(module)
    tree = ast.parse(source, filename=inspect.getsourcefile(module))

    body: list[ast.stmt] = []
    for node in tree.body:
        if (
            _is_future_import(node)
            or _is_tldw_import_from(node)
            or _is_all_assignment(node)
            or _is_main_guard(node)
        ):
            continue
        if isinstance(node, ast.Import):
            kept, dropped = _split_tldw_aliases(node)
            if dropped:
                if kept:
                    body.append(ast.Import(names=kept))
                continue
        if module_name == _WORKER_MODULE and isinstance(node, ast.FunctionDef):
            if node.name == "main":
                # The bundle's IO adapter below defines the real entry
                # ``main(stream)``; the local no-arg ``main`` must not
                # survive to be shadowed by it.
                argument_names = [
                    *(arg.arg for arg in node.args.posonlyargs),
                    *(arg.arg for arg in node.args.args),
                    *(arg.arg for arg in node.args.kwonlyargs),
                ]
                if argument_names or node.args.vararg or node.args.kwarg:
                    raise BundleBuildError(
                        "workspace_tool_worker.main grew parameters; the "
                        "builder's drop rule for it must be re-reviewed"
                    )
                continue
        body.append(node)

    transformed = _NonStdlibImportRewriter().visit(
        ast.Module(body=body, type_ignores=[])
    )

    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            target = (
                "." * node.level + (node.module or "")
                if isinstance(node, ast.ImportFrom)
                else node.names[0].name
            )
            root = target.split(".")[0]
            if root not in sys.stdlib_module_names:
                raise BundleBuildError(
                    f"module-level import of '{target}' in {module_name} is "
                    "not stdlib; the closure gate should have caught this"
                )

    top_level_names: set[str] = set()
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            top_level_names.add(node.name)
        elif isinstance(node, ast.Assign):
            top_level_names.update(
                target.id
                for target in node.targets
                if isinstance(target, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            top_level_names.add(node.target.id)

    rendered = ast.unparse(transformed)
    header = (
        f"# {'=' * 75}\n"
        f"# Section: {module_name} (extracted from the live module by the\n"
        f"# builder; regenerate rather than editing)\n"
        f"# {'=' * 75}"
    )
    return f"{header}\n{rendered}\n", top_level_names


def _assert_no_top_level_collisions(sections: dict[str, set[str]]) -> None:
    owners: dict[str, str] = {}
    for module_name in BUNDLE_MODULES:
        for name in sections[module_name]:
            previous = owners.get(name)
            if previous is not None and previous != module_name:
                raise BundleBuildError(
                    f"top-level name collision across bundle sections: "
                    f"{name!r} defined in both {previous} and {module_name}; "
                    "resolve the collision at the source before rebuilding"
                )
            owners[name] = module_name


_IO_ADAPTER_SOURCE = '''

# ---------------------------------------------------------------------------
# Bundle IO adapter (builder-emitted; the LOCAL worker has no counterpart)
# ---------------------------------------------------------------------------

#: The response magic: exactly 16 bytes (controller ruling; the earlier
#: 15-byte literal was a miscount). Prefixed to every response frame this
#: bundle emits so stdout noise on a shared remote channel can never be
#: mistaken for a response frame. The LOCAL pinned worker does NOT add
#: this prefix: its pipe has no noise source. This is the single
#: definition site; transport tests (Task 11) mirror the literal.
RESPONSE_MAGIC = b"TLDW-REMOTE-0001"


def split_magic(raw: bytes) -> tuple[bool, bytes]:
    """Strip ``RESPONSE_MAGIC`` from one raw response line before parsing.

    Args:
        raw: One captured response line, magic-prefixed or not.

    Returns:
        ``(had_magic, stripped)`` — ``had_magic`` is ``False`` (and
        ``stripped`` is ``raw`` unchanged) when the prefix is absent.
    """
    if raw.startswith(RESPONSE_MAGIC):
        return True, raw[len(RESPONSE_MAGIC) :]
    return False, raw


class _MagicPrefixStdout:
    """Minimal stdout shim prefixing every write with ``RESPONSE_MAGIC``."""

    __slots__ = ("_stream",)

    def __init__(self, stream: Any) -> None:
        self._stream = stream

    def write(self, data: bytes) -> int:
        return self._stream.write(RESPONSE_MAGIC + data)

    def flush(self) -> None:
        return self._stream.flush()


def main(stream: Any, *, bundle_sha256: str = "") -> int:
    """Run one isolated protocol exchange over an already-positioned stdin.

    ``stream`` is the buffered stdin AFTER the remote bootstrap consumed
    this bundle's own bytes (Task 9's harness positions it). Responses
    are written to the process stdout buffer with ``RESPONSE_MAGIC``
    prefixed to each frame; the process exit code follows the local
    worker's contract (0 success, 2 refused/failed). ``bundle_sha256``
    defaults to the empty string for direct/in-process callers; the
    bootstrap entry path always supplies the artifact stamp.
    """
    return run_workspace_worker(
        stream,
        _MagicPrefixStdout(sys.stdout.buffer),
        sys.stderr.buffer,
        bundle_sha256=bundle_sha256,
    )


def _enter_worker_exchange(stamp: str) -> str:
    """Bootstrap entry seam — run the exchange when exec'd as ``__main__``.

    The builder emits the artifact's FINAL line as
    ``BUNDLE_SHA256 = _enter_worker_exchange("<digest>")``. Under the
    fixed remote bootstrap this module executes inside the interpreter's
    own ``__main__`` namespace, so evaluating that assignment fires the
    one exchange (and the ``SystemExit`` aborts module execution before
    anything uncovered could follow the line). In-process loaders run
    under their own module name and simply get the stamp bound.

    Defined ABOVE the stamp assignment on purpose: the entry logic must
    sit inside the region ``BUNDLE_SHA256`` attests, so a rewritten tail
    cannot launch divergent code while echoing a matching stamp.
    """
    if __name__ == "__main__":
        raise SystemExit(main(sys.stdin.buffer, bundle_sha256=stamp))
    return stamp


# ---------------------------------------------------------------------------
# Watchdog seam (Task 12 arms this — the bundle arms NOTHING yet)
# ---------------------------------------------------------------------------

#: Live registry of temp-file paths created by in-flight operations, for
#: the watchdog's cleanup sweep. Data-plane only in this task.
TEMP_REGISTRY: list[str] = []


def register_temp(path: str) -> None:
    """Record one temp-file path for the (Task 12) cleanup sweep."""
    if path not in TEMP_REGISTRY:
        TEMP_REGISTRY.append(path)


def unregister_temp(path: str) -> None:
    """Drop one temp-file path (its operation completed or cleaned up)."""
    if path in TEMP_REGISTRY:
        TEMP_REGISTRY.remove(path)


def arm_watchdog(budget_seconds: int, temp_registry: list[str]) -> None:
    """Arm the hard-timeout watchdog (Timer + ``signal.alarm`` default action).

    NOT IMPLEMENTED YET — Task 12 wires the real arming (budget + 2s
    alarm, process-group kill, temp-registry sweep). The seam ships now
    so the builder, bundle and Task 9 harness agree on its shape.
    """
    raise NotImplementedError(
        "arm_watchdog is implemented by the watchdog task (Task 12); the "
        "remote worker bundle does not arm anything yet"
    )
'''


def _denylist_section() -> str:
    denylist = importlib.import_module(REMOTE_DENYLIST_MODULE)
    paths: tuple[str, ...] = denylist.REMOTE_SENSITIVE_PATHS
    assert paths, f"{REMOTE_DENYLIST_MODULE} shipped an empty denylist"

    entries = "".join(f"    {entry!r},\n" for entry in paths)
    return (
        "\n\n"
        "# ---------------------------------------------------------------------------\n"
        "# Remote denylist (embedded from Tools/remote_sensitive_paths.py)\n"
        "# ---------------------------------------------------------------------------\n"
        "#: Remote-home-relative paths the worker must never read, list, or\n"
        "#: write, regardless of the pinned root. Enforcement wiring lands with\n"
        "#: the remote binding tasks; this task ships the data and its embed.\n"
        f"REMOTE_SENSITIVE_PATHS: tuple[str, ...] = (\n{entries})\n"
    )


#: The stamp assignment line the artifact ENDS with (minus the digest):
#: ``BUNDLE_SHA256 = _enter_worker_exchange("<digest>")`` — the final
#: line, and the only byte range outside the stamp's own coverage.
#: ``expected_bundle_stamp`` locates it from the RIGHT so the hashed
#: prefix rule is machine-derivable by any consumer.
_BUNDLE_STAMP_ASSIGNMENT = "BUNDLE_SHA256 = _enter_worker_exchange("


def expected_bundle_stamp(data: bytes) -> str:
    """Return the artifact stamp derivable from committed bundle bytes.

    The stamp is the SHA-256 of the artifact's bytes ABOVE the (sole)
    stamp-assignment line — the artifact's final line. Everything
    executable, including the bootstrap entry helper, lives inside that
    hashed region; a full-file digest is not self-embeddable, because
    the stamp would change its own input. The loopback harness, the
    transport, and the tests all derive the expected value through this
    single rule so a tampered or divergent bundle cannot silently agree
    with the caller.

    Args:
        data: The bundle artifact's exact bytes.

    Returns:
        The lowercase hex digest the bundle's ``ping`` must echo.

    Raises:
        ValueError: If the assignment line is absent (not a stamped
            bundle artifact).
    """
    marker = b"\n" + _BUNDLE_STAMP_ASSIGNMENT.encode("utf-8")
    index = data.rfind(marker)
    if index == -1:
        raise ValueError("bundle artifact lacks a BUNDLE_SHA256 stamp line")
    return hashlib.sha256(data[: index + 1]).hexdigest()


_BUNDLE_TAIL_TEMPLATE = '''

# ---------------------------------------------------------------------------
# Bundle identity stamp (builder-emitted; ping echoes this value)
# ---------------------------------------------------------------------------
#: SHA-256 of this file's bytes ABOVE this assignment line — which is the
#: file's FINAL line, so every executable byte (bootstrap entry included)
#: is covered; only this assignment's own line falls outside the digest.
#: A full-file digest is not self-embeddable (the stamp would change its
#: own input); derive the same value from the artifact with
#: ``build_remote_worker_bundle.expected_bundle_stamp``. The remote
#: worker's ``ping`` echoes it so callers can confirm which bundle the
#: remote actually executed.
BUNDLE_SHA256 = _enter_worker_exchange("{stamp}")
'''


def build_bundle_text() -> str:
    """Build the complete bundle text deterministically from live sources."""
    _assert_frozen_list_matches_closure()

    sections: list[str] = []
    top_level: dict[str, set[str]] = {}
    for module_name in BUNDLE_MODULES:
        section, names = _module_section(module_name)
        sections.append(section)
        top_level[module_name] = names
    _assert_no_top_level_collisions(top_level)

    floor = f"{REMOTE_PYTHON_FLOOR[0]}.{REMOTE_PYTHON_FLOOR[1]}"
    # Stage 1: assemble everything above the stamp with a placeholder
    # digest of identical length, so the stamp's own bytes cannot move
    # the boundary it is computed from.
    head = "\n".join(
        [
            _BUNDLE_DOCSTRING.format(floor=floor),
            "from __future__ import annotations",
            "",
            "\n\n".join(sections).rstrip("\n"),
            _IO_ADAPTER_SOURCE,
            _denylist_section(),
        ]
    ).rstrip("\n") + "\n"
    placeholder = "0" * 64
    unstamped = head + _BUNDLE_TAIL_TEMPLATE.format(stamp=placeholder)
    # Stage 2: hash the bytes above the assignment line, then substitute
    # — the placeholder and digest are both 64 characters, so the
    # substitution cannot shift the hashed prefix (which now includes
    # the bootstrap entry helper; only the assignment's own line stays
    # outside the digest).
    stamp = expected_bundle_stamp(unstamped.encode("utf-8"))
    return unstamped.replace(
        f'{_BUNDLE_STAMP_ASSIGNMENT}"{placeholder}"',
        f'{_BUNDLE_STAMP_ASSIGNMENT}"{stamp}"',
        1,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI: regenerate (default) or ``--check`` the committed bundle artifact."""
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate tldw_chatbook/Tools/remote_worker_bundle.py from the "
            "worker's real dependency modules."
        )
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed artifact matches a rebuild; exit 1 on drift",
    )
    arguments = parser.parse_args(argv)

    text = build_bundle_text()
    if arguments.check:
        if not _BUNDLE_PATH.exists():
            print(f"missing artifact: {_BUNDLE_PATH}", file=sys.stderr)
            return 1
        committed = _BUNDLE_PATH.read_text(encoding="utf-8")
        if committed != text:
            print(
                f"artifact drifted from sources: {_BUNDLE_PATH}\n"
                "regenerate with: "
                "python -m tldw_chatbook.Tools.build_remote_worker_bundle",
                file=sys.stderr,
            )
            return 1
        print(f"ok: {_BUNDLE_PATH} matches a fresh build")
        return 0

    _BUNDLE_PATH.write_text(text, encoding="utf-8")
    print(f"wrote {_BUNDLE_PATH} ({len(text)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

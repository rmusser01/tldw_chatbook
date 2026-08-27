"""Import-closure guards for the Research_Workspace facade (TASK-23023).

One import statement put 8 of this repo's modules on the boot path for a
single integer constant:

* ``Library/library_ingest_jobs.py`` (an ``app.py`` module-scope
  dependency) imports the stdlib-only validator
  ``Research_Workspace.source_operations.validate_source_operation_id``
  *through the package*, so the package ``__init__`` ran -- and it eagerly
  re-exported the whole tree: controller, overlay/layout stores, both
  adapters, quick notes.
* ``server_adapter`` in turn imported ``tldw_api.notes_workspace_schemas``
  (782 LOC, 26 pydantic models, 20.6 ms self) to read one integer:
  ``MAX_WORKSPACE_SOURCE_OWNER_ROWS``.

Measured interleaved x3 on 2026-08-27: that one statement cost 273-286 ms /
143 own modules standalone (eager) vs 78-81 ms / 4 own modules (lazy);
``import tldw_chatbook.app`` dropped 658 -> 650 own modules. Same class as
the TASK-21102/21107 facade leaks; the package ``__init__`` is now a PEP 562
lazy facade and the bound constants live in the stdlib-only
``tldw_api/notes_workspace_limits.py``.

The mount/construct leg is covered too (the TASK-21731 lesson: an
import-weight guard cannot see a cost that moved to screen mount):
``Tests/Performance/test_ui_ready_module_census.py`` pins
controller/layout_state/overlay_store/notes_workspace_schemas absent at
``_ui_ready``, and the deferred-state test below drives the real screen
from exactly the state boot leaves behind. ``local_adapter``/
``server_adapter``/``quick_notes`` are deliberately NOT pinned absent
anywhere: ``TldwCli.__init__``'s ``_wire_research_source_association``
legitimately builds readiness adapters at construction, so they leave the
IMPORT closure but stay on the construct leg (without the pydantic module,
which is the heavy part).

Subprocess-isolated for the same reason as the sibling closure guards:
``sys.modules`` is process-global, so an earlier test that legitimately
imported the Research Workspace tree would turn an in-process check into a
false pass/fail.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]

#: Modules the lazy facade takes off the boot import path. `tldw_api.
#: exceptions` also left the measured closure but is NOT pinned: it is a
#: general-purpose module any boot feature may legitimately import.
DEFERRED_MODULES = (
    "tldw_chatbook.Research_Workspace.controller",
    "tldw_chatbook.Research_Workspace.layout_state",
    "tldw_chatbook.Research_Workspace.overlay_store",
    "tldw_chatbook.Research_Workspace.local_adapter",
    "tldw_chatbook.Research_Workspace.server_adapter",
    "tldw_chatbook.Research_Workspace.quick_notes",
    "tldw_chatbook.tldw_api.notes_workspace_schemas",
)

#: Boot-path members that must STAY in the closure (anti-vacuity: this guard
#: is about what the facade drags in, not about whether it is reached).
#: `app.py` imports source_association/source_operation_store/paste_staging/
#: source_readiness at module scope; `library_ingest_jobs` imports
#: source_operations, which imports contracts.
EXPECTED_BOOT_MEMBERS = (
    "tldw_chatbook.Research_Workspace",
    "tldw_chatbook.Research_Workspace.contracts",
    "tldw_chatbook.Research_Workspace.source_operations",
    "tldw_chatbook.Research_Workspace.source_operation_store",
    "tldw_chatbook.Research_Workspace.source_association",
    "tldw_chatbook.Research_Workspace.paste_staging",
    "tldw_chatbook.Research_Workspace.source_readiness",
)

#: Scratch profile for the full-boot tests, mirroring
#: `test_ui_ready_module_census.py`: wizard completed, splash disabled, a
#: valid-SHAPED nonsense key so Console boots configured. Nothing dials out.
_PROBE_CONFIG_TOML = """\
[first_run]
setup_completed = true

[splash_screen]
enabled = false

[api_settings.openai]
api_key = "sk-abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKL"
"""


def _run_isolated_python(
    tmp_path: Path, code: str, *, full_profile: bool = False
) -> subprocess.CompletedProcess[str]:
    """Run a Python snippet in a fresh interpreter with isolated config/data dirs.

    Args:
        tmp_path: Per-test scratch directory for the subprocess's HOME/XDG so
            the app import can never read or write the live user config.
        code: The Python source to execute with ``python -c``.
        full_profile: When True, also write the boot-ready ``config.toml``
            and pin the screen pre-importer off -- required by the tests that
            boot the app to ``_ui_ready`` (the pre-importer's daemon thread
            imports every screen module and would race the deferred-state
            assertions nondeterministically).

    Returns:
        The completed process (never raises on nonzero exit).
    """
    data_home = tmp_path / "data"
    config_home = tmp_path / "config"
    home = tmp_path / "home"
    for path in (data_home, config_home, home):
        path.mkdir(parents=True, exist_ok=True)

    env = {
        **os.environ,
        "TLDW_TEST_MODE": "1",
        "XDG_DATA_HOME": str(data_home),
        "XDG_CONFIG_HOME": str(config_home),
        "HOME": str(home),
        "USERPROFILE": str(home),
        "PYTHONPATH": str(REPO_ROOT),
    }
    env.pop("PYTEST_CURRENT_TEST", None)
    env.pop("TLDW_CONFIG_PATH", None)
    if full_profile:
        (config_home / "config.toml").write_text(_PROBE_CONFIG_TOML)
        env["TLDW_CONFIG_PATH"] = str(config_home / "config.toml")
        env["TLDW_SCREEN_PREIMPORT"] = "0"

    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=300,
    )


_RESIDENT_HELPER = """
import sys

DEFERRED_MODULES = {deferred!r}


def resident():
    return [m for m in DEFERRED_MODULES if sys.modules.get(m) is not None]
""".format(deferred=DEFERRED_MODULES)


_APP_IMPORT_SNIPPET = _RESIDENT_HELPER + """
EXPECTED_BOOT_MEMBERS = {expected!r}

import tldw_chatbook.app  # noqa: F401

loaded = resident()
assert not loaded, f"deferred modules resident after app import: {{loaded}}"

for expected in EXPECTED_BOOT_MEMBERS:
    assert expected in sys.modules, f"expected closure member missing: {{expected}}"

# The validator the whole chain exists for still answers from the deferred
# state, without waking anything up.
from tldw_chatbook.Library.library_ingest_jobs import validate_source_operation_id

validate_source_operation_id("rsop-0123456789abcdef0123456789abcdef")
assert not resident(), f"validating an id woke the deferred tree: {{resident()}}"
print("APP_CLOSURE_OK")
""".format(expected=EXPECTED_BOOT_MEMBERS)


def test_app_import_does_not_execute_research_workspace_eager_reexports(
    tmp_path: Path,
) -> None:
    """`import tldw_chatbook.app` resolves none of the deferred facade members.

    The regression this pins measured 658 own modules at app import against
    the 660 budget, 8 of them from this one facade (2026-08-27); the lazy
    facade returns the count to 650.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's HOME/XDG.
    """
    result = _run_isolated_python(tmp_path, _APP_IMPORT_SNIPPET)
    assert result.returncode == 0, (
        "import tldw_chatbook.app must not execute the Research_Workspace "
        f"eager re-exports:\nstdout={result.stdout}\nstderr={result.stderr[-4000:]}"
    )
    assert "APP_CLOSURE_OK" in result.stdout


_FACADE_CONTRACT_SNIPPET = """
import importlib
import sys

import tldw_chatbook.Research_Workspace as rw

# Importing the bare package must resolve ZERO submodules.
resident = [
    m
    for m in sys.modules
    if m.startswith("tldw_chatbook.Research_Workspace.")
    and sys.modules[m] is not None
]
assert not resident, f"package import pulled submodules: {resident}"

# The map and __all__ must agree, or a name would be advertised that the
# facade cannot serve (or served that tooling cannot see).
assert sorted(rw.__all__) == sorted(rw._SUBMODULE_BY_NAME), (
    set(rw.__all__) ^ set(rw._SUBMODULE_BY_NAME)
)

# Every advertised name resolves to the IDENTICAL object its submodule owns,
# and is cached so the second lookup skips __getattr__.
for name in rw.__all__:
    value = getattr(rw, name)
    owner = importlib.import_module(
        "tldw_chatbook.Research_Workspace." + rw._SUBMODULE_BY_NAME[name]
    )
    assert value is getattr(owner, name), name
    assert name in vars(rw), f"{name} not cached on the package module"

# Unknown names fail the way a module attribute miss always has.
try:
    rw.NoSuchExport
except AttributeError:
    pass
else:
    raise AssertionError("unknown attribute did not raise AttributeError")

assert set(rw.__all__) <= set(dir(rw))
print("FACADE_CONTRACT_OK")
"""


def test_facade_names_resolve_lazily_to_the_owning_submodules(
    tmp_path: Path,
) -> None:
    """The PEP 562 facade serves every ``__all__`` name, lazily and identically.

    Absence is not the deliverable -- a deleted re-export would satisfy a
    pure closure guard. This proves the bare package import is free AND that
    all 31 advertised names still resolve to the exact objects their
    submodules own (a typo in ``_SUBMODULE_BY_NAME`` fails here).

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's HOME/XDG.
    """
    result = _run_isolated_python(tmp_path, _FACADE_CONTRACT_SNIPPET)
    assert result.returncode == 0, (
        f"facade contract failed:\nstdout={result.stdout}\n"
        f"stderr={result.stderr[-4000:]}"
    )
    assert "FACADE_CONTRACT_OK" in result.stdout


_SERVER_ADAPTER_SNIPPET = """
import sys

import tldw_chatbook.Research_Workspace.server_adapter as server_adapter

assert "tldw_chatbook.tldw_api.notes_workspace_schemas" not in sys.modules, (
    "server_adapter still imports the 26-model pydantic schema module"
)
assert server_adapter.MAX_WORKSPACE_SOURCE_OWNER_ROWS == 10_100

# Single-sourced: the schema module re-imports the bounds from the limits
# module, so the value the adapter enforces IS the object the schema fields
# validate against -- no copy that can drift.
from tldw_chatbook.tldw_api import notes_workspace_limits, notes_workspace_schemas

assert (
    notes_workspace_schemas.MAX_WORKSPACE_SOURCE_OWNER_ROWS
    is notes_workspace_limits.MAX_WORKSPACE_SOURCE_OWNER_ROWS
)
assert (
    notes_workspace_schemas.MAX_WORKSPACE_SOURCE_ROWS
    is notes_workspace_limits.MAX_WORKSPACE_SOURCE_ROWS
)
assert (
    server_adapter.MAX_WORKSPACE_SOURCE_OWNER_ROWS
    is notes_workspace_limits.MAX_WORKSPACE_SOURCE_OWNER_ROWS
)
print("SERVER_ADAPTER_LIMITS_OK")
"""


def test_server_adapter_reads_the_owner_rows_bound_without_the_schema_module(
    tmp_path: Path,
) -> None:
    """The adapter's bound comes from the stdlib-only limits module.

    ``chunking_engine_version.py`` (TASK-21102) and ``search_modes.py``
    (TASK-21731) are the exemplars: a pure value lifted out of the heavy
    module that owned it, re-imported by that module so no second copy can
    drift. Both properties are checked.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's HOME/XDG.
    """
    result = _run_isolated_python(tmp_path, _SERVER_ADAPTER_SNIPPET)
    assert result.returncode == 0, (
        f"server_adapter limits contract failed:\nstdout={result.stdout}\n"
        f"stderr={result.stderr[-4000:]}"
    )
    assert "SERVER_ADAPTER_LIMITS_OK" in result.stdout


_DEFERRED_SCREEN_SNIPPET = _RESIDENT_HELPER + """
import asyncio

#: Screen-leg members that must stay absent through _ui_ready. The adapter/
#: quick_notes members are excluded on purpose: app CONSTRUCTION legitimately
#: resolves them for the readiness coordinator (see the module docstring of
#: this test file).
ABSENT_AT_READY = (
    "tldw_chatbook.Research_Workspace.controller",
    "tldw_chatbook.Research_Workspace.layout_state",
    "tldw_chatbook.Research_Workspace.overlay_store",
    "tldw_chatbook.tldw_api.notes_workspace_schemas",
    "tldw_chatbook.UI.Screens.research_workspace_screen",
)


async def main() -> None:
    import tldw_chatbook.app

    assert not resident(), f"resident after app import: {resident()}"

    app = tldw_chatbook.app.TldwCli()
    async with app.run_test(size=(120, 40)):
        while not getattr(app, "_ui_ready", False):
            await asyncio.sleep(0.005)

        on_leg = [m for m in ABSENT_AT_READY if sys.modules.get(m) is not None]
        assert not on_leg, f"deferred members resident at _ui_ready: {on_leg}"

        # Drive the route exactly as the nav bar does, from the deferred
        # state: the route module import triggers the facade __getattr__.
        from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen

        await app.handle_screen_navigation(NavigateToScreen("research_workspace"))
        await asyncio.sleep(0)

        from tldw_chatbook.Research_Workspace.controller import (
            ResearchWorkspaceController,
        )
        from tldw_chatbook.UI.Screens.research_workspace_screen import (
            ResearchWorkspaceScreen,
        )

        screen = app.screen
        assert isinstance(screen, ResearchWorkspaceScreen), screen
        controller = getattr(screen, "_controller", None) or getattr(
            screen, "controller", None
        )
        assert isinstance(controller, ResearchWorkspaceController), controller
        assert sys.modules.get("tldw_chatbook.Research_Workspace.overlay_store"), (
            "navigation did not resolve the deferred overlay store"
        )
    print("DEFERRED_SCREEN_OK")


asyncio.run(main())
"""


@pytest.mark.integration
def test_research_workspace_screen_mounts_from_the_deferred_state(
    tmp_path: Path,
) -> None:
    """The real screen still opens once its imports are deferred.

    Boots the actual app headless to ``_ui_ready`` (deferred members proven
    absent at that moment -- the TASK-21731 "moved to mount" trap), then
    navigates to the Research Workspace route: the route module import
    resolves ``ResearchWorkspaceController`` etc. through the lazy facade,
    the screen mounts, and its controller is the real class.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's profile.
    """
    result = _run_isolated_python(tmp_path, _DEFERRED_SCREEN_SNIPPET, full_profile=True)
    assert result.returncode == 0, (
        "Research Workspace screen failed from the deferred state:\n"
        f"stdout={result.stdout[-2000:]}\nstderr={result.stderr[-4000:]}"
    )
    assert "DEFERRED_SCREEN_OK" in result.stdout


_BROKEN_SUBMODULE_SNIPPET = """
import asyncio
import sys

BLOCKED = "tldw_chatbook.Research_Workspace.controller"


class _BreakController:
    \"\"\"Meta-path finder simulating an install with one broken submodule.\"\"\"

    def find_spec(self, name, path=None, target=None):
        if name == BLOCKED:
            raise ImportError(f"simulated broken install for {name}")
        return None


sys.meta_path.insert(0, _BreakController())


async def main() -> None:
    # Boot must survive: the whole point of the lazy facade is that a broken
    # screen-side submodule no longer takes `import tldw_chatbook.app` down.
    import tldw_chatbook.app

    app = tldw_chatbook.app.TldwCli()
    async with app.run_test(size=(120, 40)):
        while not getattr(app, "_ui_ready", False):
            await asyncio.sleep(0.005)

        from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
        from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

        assert isinstance(app.screen, ChatScreen), app.screen

        # First use: the deferred import failure lands here, not at boot.
        await app.handle_screen_navigation(NavigateToScreen("research_workspace"))
        await asyncio.sleep(0)

        # Legible, not fatal: the app is alive, the user stayed on the
        # current screen, and was TOLD -- not left staring at an unchanged
        # screen with a stuck nav highlight (the task-2720 defect).
        assert isinstance(app.screen, ChatScreen), app.screen
        messages = [n.message for n in app._notifications]
        assert any("Couldn't open" in m for m in messages), messages

        # ...and the rest of the app still navigates.
        await app.handle_screen_navigation(NavigateToScreen("library"))
        await asyncio.sleep(0)
        from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

        assert isinstance(app.screen, LibraryScreen), app.screen
    print("BROKEN_SUBMODULE_LEGIBLE_OK")


asyncio.run(main())
"""


@pytest.mark.integration
def test_broken_deferred_submodule_fails_legibly_at_first_navigation(
    tmp_path: Path,
) -> None:
    """Moving an import moves its failure -- this pins where that failure lands.

    With ``Research_Workspace.controller`` unimportable, the eager facade
    used to kill ``import tldw_chatbook.app`` outright (a dead app). Now the
    app boots; the failure surfaces at first navigation as a user-visible
    "Couldn't open" notification while the current screen stays mounted and
    every other route keeps working. Runs in a subprocess with a meta-path
    blocker because ``sys.modules`` is process-global.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's profile.
    """
    result = _run_isolated_python(
        tmp_path, _BROKEN_SUBMODULE_SNIPPET, full_profile=True
    )
    assert result.returncode == 0, (
        "broken-submodule first-navigation contract failed:\n"
        f"stdout={result.stdout[-2000:]}\nstderr={result.stderr[-4000:]}"
    )
    assert "BROKEN_SUBMODULE_LEGIBLE_OK" in result.stdout

"""Import-closure guard: booting the app must not execute Persona_Visual or PIL (TASK-21103).

``import tldw_chatbook.app`` used to execute 93% of ``Persona_Visual`` (6,633
LOC) and put ``PIL.Image``/``PIL._imaging`` on the boot path -- measured at
1.276 s of a 3.10 s cold app import (41%) in the 2026-08-22 holistic perf
review. Three chains carried it:

1. ``app.py`` -> ``Chat/console_runtime.py`` -> ``Persona_Buddy.console_adapter``
   -> the eager ``Persona_Buddy/__init__`` -> ``controller.py`` ->
   ``Persona_Visual.repository``/``runtime`` -> the eager
   ``Persona_Visual/__init__`` (authoring/assets/importer, each
   ``from PIL import Image``), plus the eager ``from .Persona_Buddy import``
   in ``app.py`` itself;
2. ``app.py`` importing ``Widgets/Chat_Widgets/chat_message_enhanced``
   eagerly (module-level PIL + the ``textual_image`` package);
3. ``app.py`` -> ``UI/image_gen_command_provider`` ->
   ``UI/Screens/image_gen_demo_screen`` -> ``Image_Generation.worker`` ->
   ``Image_Generation.request_validation`` (module-level PIL).

The fix makes ``Persona_Buddy/__init__`` and ``Persona_Visual/__init__``
PEP-562 lazy facades (importing ``Persona_Buddy.console_adapter`` or any
``Persona_Visual`` submodule no longer executes the tree), builds the buddy
controller lazily on the app, and converts chains 2 and 3 to function-local
imports.

TASK-21200 added a fourth chain and a second guard. The Actor Packs
activation feature (``ac1037732``, ``a98f3c14d``) landed
``app.py`` -> ``Actor_Packs/__init__`` -> ``Actor_Packs.activation`` ->
``Persona_Visual.repository`` + ``Character_Chat.visual_identity`` (PIL),
re-introducing the whole regression while this file was already on disk --
the guard existed at their merge but CI was not yet enforcing checks. Note
that ``app.py`` imports ``Actor_Packs.activation``/``export``/``importer``
*directly*, so a lazy package ``__init__`` would not have helped: those three
modules must be heavy-free themselves.
``test_actor_pack_modules_do_not_execute_persona_visual_or_pil`` pins that
stronger, source-level property, and both guards now report the offending
import chain rather than only a module list.

Subprocess-isolated for the same reason as
``test_chunking_import_closure.py`` (TASK-21102), whose pattern this file
follows: ``sys.modules`` is process-global, so an earlier test in the
session that legitimately imported Persona_Visual would false-fail (or a
pre-imported app would false-pass) an in-process check.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_isolated_python(tmp_path: Path, code: str) -> subprocess.CompletedProcess[str]:
    """Run a Python snippet in a fresh interpreter with isolated config/data dirs.

    Args:
        tmp_path: Per-test scratch directory for the subprocess's HOME/XDG so
            the app import can never read or write the live user config.
        code: The Python source to execute with ``python -c``.

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
        "PYTHONPATH": str(REPO_ROOT),
    }
    env.pop("PYTEST_CURRENT_TEST", None)
    env.pop("TLDW_CONFIG_PATH", None)

    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )


# Installed *before* the import under test so a failure names the chain that
# pulled the heavy module in, not just the fact that it is resident. Reading
# `-X importtime` output instead is a known trap: its indentation nests by
# completion order, and misreading it sent TASK-21200's first diagnosis at the
# wrong module.
_IMPORT_CHAIN_TRACER = '''
import sys

_import_parent = {}


class _ChainTracer:
    """Record which module's body triggered each import.

    ``find_spec`` runs while the importing module is still on the stack, and
    importlib executes a module body in a frame whose ``co_name`` is
    ``<module>``. The nearest such frame outside this finder is therefore the
    true importer. Returns ``None`` so the real finders still resolve.
    """

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _import_parent:
            _import_parent[fullname] = self._importing_module()
        return None

    @staticmethod
    def _importing_module():
        depth = 2  # 0 = this frame, 1 = find_spec, 2 = importlib internals
        while True:
            try:
                frame = sys._getframe(depth)
            except ValueError:
                return "<root>"
            if frame.f_code.co_name == "<module>":
                name = frame.f_globals.get("__name__")
                if name and not name.startswith("importlib"):
                    return name
            depth += 1


def _import_chain(module):
    seen = {module}
    parts = [module]
    while True:
        parent = _import_parent.get(parts[-1])
        if parent is None or parent in seen:
            break
        seen.add(parent)
        parts.append(parent)
    return " -> ".join(reversed(parts))


FORBIDDEN_PREFIXES = (
    "PIL",
    "rich_pixels",
    "textual_image",
    "tldw_chatbook.Persona_Visual",
    "tldw_chatbook.Persona_Buddy.controller",
    "tldw_chatbook.Persona_Buddy.rendering",
)


def resident_heavy_modules():
    return sorted(
        m for m in sys.modules
        if any(m == p or m.startswith(p + ".") for p in FORBIDDEN_PREFIXES)
        and sys.modules[m] is not None
    )


def describe_leak(resident, what):
    lines = [
        f"{len(resident)} heavy module(s) executed by {what}.",
        "Each offending import chain (importer -> imported):",
    ]
    shown = set()
    for module in resident:
        chain = _import_chain(module)
        if chain in shown:
            continue
        shown.add(chain)
        lines.append(f"  {chain}")
    lines.append(
        "Fix by deferring the import (function-local, or TYPE_CHECKING for "
        "annotations) at the LAST tldw_chatbook module in the chain above -- "
        "see the TASK-21200 notes in tldw_chatbook/Actor_Packs/activation.py."
    )
    return "\\n".join(lines)


sys.meta_path.insert(0, _ChainTracer())
'''

_BUDDY_CLOSURE_SNIPPET = _IMPORT_CHAIN_TRACER + """
import tldw_chatbook.app  # noqa: F401

resident = resident_heavy_modules()
assert not resident, describe_leak(resident, "import tldw_chatbook.app")

# The stdlib-only console adapter seam is ALLOWED to stay import-time (it is
# what Chat/console_runtime.py needs at module scope); its parent package
# init must now be lazy, which the assertions above prove (importing
# console_adapter executed the whole controller chain before the fix).
assert "tldw_chatbook.Persona_Buddy.console_adapter" in sys.modules, (
    "console_adapter left the app import closure; this guard no longer "
    "exercises the lazy Persona_Buddy package init"
)

# Anti-vacuity: the converted entry-point modules must still be part of the
# app's import closure. If one of them leaves the closure entirely, this
# guard would otherwise pass without testing the conversion at all.
for expected in (
    "tldw_chatbook.Chat.console_runtime",
    "tldw_chatbook.UI.image_gen_command_provider",
):
    assert expected in sys.modules, f"expected closure member missing: {expected}"

print("PERSONA_BUDDY_CLOSURE_OK")
"""


def test_app_import_does_not_execute_persona_visual_or_pil(tmp_path: Path) -> None:
    """No Persona_Visual, buddy controller/rendering, PIL, rich_pixels or
    textual_image module is resident after ``import tldw_chatbook.app``.

    Regression guard for the TASK-21103 defect: before the fix, the eager
    ``Persona_Buddy`` import chain (and two independent chains through
    ``chat_message_enhanced`` and ``image_gen_command_provider``) executed
    PIL and most of Persona_Visual during ``import tldw_chatbook.app``, so
    this subprocess failed on the residency assertion.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's HOME/XDG.
    """
    result = _run_isolated_python(tmp_path, _BUDDY_CLOSURE_SNIPPET)
    assert result.returncode == 0, (
        "import tldw_chatbook.app must not execute Persona_Visual or PIL:\n"
        f"stdout={result.stdout}\nstderr={result.stderr[-4000:]}"
    )
    assert "PERSONA_BUDDY_CLOSURE_OK" in result.stdout


_LAZY_FACADE_SNIPPET = """
import sys

# Importing the stdlib-only adapter/preferences seams must not execute the
# controller chain (this is the exact import Chat/console_runtime.py performs
# at module scope).
from tldw_chatbook.Persona_Buddy.console_adapter import PersonaBuddyConsoleAdapter  # noqa: F401,E501
from tldw_chatbook.Persona_Buddy.preferences import parse_persona_buddy_preferences  # noqa: F401,E501

# Importing the Persona_Visual constants module (what Persona_Buddy/rendering
# needs) must not execute the rest of the Persona_Visual tree.
from tldw_chatbook.Persona_Visual.contracts import MAX_ASSET_DIMENSION

for forbidden in (
    "tldw_chatbook.Persona_Buddy.controller",
    "tldw_chatbook.Persona_Buddy.rendering",
    "tldw_chatbook.Persona_Visual.assets",
    "tldw_chatbook.Persona_Visual.authoring",
    "tldw_chatbook.Persona_Visual.importer",
    "tldw_chatbook.Persona_Visual.runtime",
    "PIL",
):
    assert forbidden not in sys.modules, (
        f"lazy package init leaked {forbidden}"
    )

# The facades must still serve their public names (PEP 562), and the served
# object must be the submodule's own -- one source of truth, no drifted copy.
import tldw_chatbook.Persona_Buddy as buddy_pkg
import tldw_chatbook.Persona_Visual as visual_pkg
from tldw_chatbook.Persona_Buddy.controller import PersonaBuddyController

assert buddy_pkg.PersonaBuddyController is PersonaBuddyController
assert visual_pkg.MAX_ASSET_DIMENSION is MAX_ASSET_DIMENSION
assert type(MAX_ASSET_DIMENSION) is int

print("PERSONA_LAZY_FACADE_OK")
"""


def test_persona_package_inits_are_lazy_and_single_sourced(tmp_path: Path) -> None:
    """The two package facades import tree-free and still serve their exports.

    Two properties in one subprocess: importing the stdlib-only seams
    (``console_adapter``, ``preferences``, ``Persona_Visual.contracts``) must
    not execute the controller/PIL tree, and the package-level names must
    still resolve to the same objects the submodules define.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's HOME/XDG.
    """
    result = _run_isolated_python(tmp_path, _LAZY_FACADE_SNIPPET)
    assert result.returncode == 0, (
        f"lazy persona package facade check failed:\nstdout={result.stdout}\n"
        f"stderr={result.stderr[-4000:]}"
    )
    assert "PERSONA_LAZY_FACADE_OK" in result.stdout


# The three Actor_Packs modules `app.py` imports at module scope. Naming them
# (and their heavy dependencies) explicitly is the point: a count-only guard
# tells the next author that something regressed, not what to defer.
_ACTOR_PACK_SOURCE_SNIPPET = _IMPORT_CHAIN_TRACER + """
import sys

# app.py imports these three DIRECTLY (not via the package facade), so each
# must be heavy-free on its own. Their heavy dependencies, which must all be
# function-local or TYPE_CHECKING-only:
#   activation -> Persona_Visual.repository, Character_Chat.visual_identity
#   export     -> Persona_Visual.assets/.repository, Character_Chat.visual_identity
#   importer   -> Persona_Visual.repository/.validation,
#                 Character_Chat.visual_identity
# `Character_Chat.visual_identity` is the PIL carrier (module-level
# `from PIL import Image`); `Persona_Visual.*` is forbidden outright.
import tldw_chatbook.Actor_Packs.activation as activation
import tldw_chatbook.Actor_Packs.export as export
import tldw_chatbook.Actor_Packs.importer as importer

resident = resident_heavy_modules()
assert not resident, describe_leak(
    resident, "importing Actor_Packs activation/export/importer"
)

# Anti-vacuity: the modules must really have executed and still expose the
# services app.py binds, so this guard cannot pass on a failed/no-op import.
assert activation.ActorPackActivationService is not None
assert export.ActorPackExportService is not None
assert importer.ActorPackImportService is not None
assert "tldw_chatbook.Character_Chat.visual_identity" not in sys.modules

print("ACTOR_PACK_SOURCE_CLOSURE_OK")
"""


def test_actor_pack_modules_do_not_execute_persona_visual_or_pil(
    tmp_path: Path,
) -> None:
    """Actor_Packs activation/export/importer import without PIL or Persona_Visual.

    Stronger and more local than the app-closure guard above: it pins the
    property at the source, so any future module-scope consumer of these three
    modules inherits it. Regression guard for TASK-21200, where the Actor Packs
    activation feature put ``Persona_Visual.repository`` and the PIL-importing
    ``Character_Chat.visual_identity`` at module scope in all three.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's HOME/XDG.
    """
    result = _run_isolated_python(tmp_path, _ACTOR_PACK_SOURCE_SNIPPET)
    assert result.returncode == 0, (
        "Actor_Packs modules must not execute Persona_Visual or PIL at import "
        f"time:\nstdout={result.stdout}\nstderr={result.stderr[-4000:]}"
    )
    assert "ACTOR_PACK_SOURCE_CLOSURE_OK" in result.stdout

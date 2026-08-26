"""Construct-time runtime-import allowlist (TASK-22222, finding 22222).

The import guards (`test_app_import_weight.py`, the Packaging closure
guards) all measure `import tldw_chatbook.app`. A function-level import
inside `TldwCli.__init__` is invisible to every one of them: the 2026-08-24
holistic perf review found `_wire_character_persona_services` re-importing
`Persona_Visual.*` at construct -- harmless that day, but a module boundary
crossed with no guard watching, and the class of change that let the boot
regress ~11% while all import guards stayed green.

This guard constructs the real `TldwCli()` in a subprocess against a
scratch profile, diffs this repo's `sys.modules` around the construction
call, and pins the newly-imported set against an allowlist. The pinned
property: **a new function-level import on the construction path is a
reviewed decision** -- add the row here, with the owning feature named, in
the same commit. Shrinking the set (deferring one of these to first feature
use) is a free win: membership is `observed <= allowlist`, so removals
never fail.

The census is of the WARM (second) construct: the probe constructs once to
create/migrate the profile, then measures a fresh interpreter's construct
against it -- the recurring user experience. The FRESH construct
additionally pulls the entire `Chunking` engine (~34 modules) through the
media-DB v6->v7 migration, which is legitimate one-time work (same
rationale as `test_ui_ready_module_census.py`); the warm census asserts
that family stays OFF the recurring path.

Warm-construct inventory (measured 2026-08-25, this branch, base dev
f0e896122; 13 modules, stable across consecutive runs): the
Persona_Visual/visual_identity wiring, Persona_Buddy preferences,
Scheduling migration modules, the Subscriptions startup reconcile, and the
Video_Generation store family. See `ALLOWED_CONSTRUCT_IMPORTS`.

Raising: when this fails, the message names the unlisted modules. Prefer
deferring the import to first feature use (the `Persona_Buddy.controller`
property pattern in app.py); if construction genuinely needs it, add the
row and the cause in the same commit and check whether the module drags a
heavy third-party closure (that cost lands on EVERY boot).

Documented blind spots (what a module diff cannot see):

* Residency, not time: work added inside an already-imported module's
  functions is invisible; only a construct-time probe sees it.
* Only `tldw_chatbook.*` is diffed: a construct-time import of a
  third-party package through an already-imported repo module (e.g. PIL via
  `Character_Chat.visual_identity`, finding 22217) shows up here only as
  its repo-side entry point. Third-party growth rides those rows unseen.
* Construction only: imports on the mount/first-paint legs are
  `test_ui_ready_module_census.py`'s territory, and `TLDW_TEST_MODE=1` is
  set (like every boot guard), so paths gated off test mode are not
  exercised.
* Warm construct only: fresh-profile/migration-time imports are a
  documented non-goal (the 22200 post-upgrade family), except that the
  `Chunking` absence assert below keeps the known-heaviest migration
  family from leaking onto the warm path.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

#: This repo's modules that `TldwCli()` construction may newly import on a
#: warm profile. Adding a row is the reviewed decision this guard forces.
ALLOWED_CONSTRUCT_IMPORTS: frozenset[str] = frozenset(
    {
        # Persona wiring (_wire_character_persona_services; the review's
        # original sighting -- reviewed and accepted as-is for now).
        "tldw_chatbook.Character_Chat.visual_identity",
        "tldw_chatbook.Persona_Buddy.preferences",
        "tldw_chatbook.Persona_Visual",
        "tldw_chatbook.Persona_Visual.contracts",
        "tldw_chatbook.Persona_Visual.repository",
        "tldw_chatbook.Persona_Visual.validation",
        # Scheduling DB migration registry (imported even when no migration
        # runs).
        "tldw_chatbook.Scheduling.db.migrations.v1_to_v2",
        "tldw_chatbook.Scheduling.db.migrations.v2_to_v3",
        # Subscriptions startup reconcile.
        "tldw_chatbook.Subscriptions.startup_reconcile",
        # Video generation store wiring.
        "tldw_chatbook.Video_Generation",
        "tldw_chatbook.Video_Generation.config",
        "tldw_chatbook.Video_Generation.video_formats",
        "tldw_chatbook.Video_Generation.video_store",
    }
)

#: Known-heavy family that belongs to FRESH-profile migration only; if it
#: appears on a warm construct, a migration (or its import) leaked onto the
#: recurring boot path.
FORBIDDEN_ON_WARM_CONSTRUCT_PREFIXES = ("tldw_chatbook.Chunking",)

_PROBE_SCRIPT = """
import sys

import tldw_chatbook.app  # the closure the import guards already police

before = {m for m in sys.modules if m.startswith("tldw_chatbook")}
app = tldw_chatbook.app.TldwCli()
print("CONSTRUCTED", flush=True)
after = {m for m in sys.modules if m.startswith("tldw_chatbook")}
for m in sorted(after - before):
    print("NEW:" + m, flush=True)
"""


def _run_construct_probe(env: dict[str, str]) -> list[str]:
    result = subprocess.run(
        [sys.executable, "-c", _PROBE_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, (
        f"app construct failed (rc={result.returncode}):\n{result.stderr[-4000:]}"
    )
    # Anti-vacuity: the sentinel proves construction ran to completion in
    # the probe; without it an early crash could report an empty diff.
    assert "CONSTRUCTED" in result.stdout, (
        f"construct sentinel missing:\n{result.stdout[-2000:]}"
    )
    return [
        line[len("NEW:") :]
        for line in result.stdout.splitlines()
        if line.startswith("NEW:")
    ]


@pytest.mark.integration
def test_construct_time_runtime_imports_stay_within_the_allowlist(
    tmp_path: Path,
) -> None:
    """`TldwCli()` on a warm profile imports only reviewed repo modules.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's profile.
    """
    home = tmp_path / "home"
    data = tmp_path / "data"
    config_dir = tmp_path / "config"
    for directory in (home, data, config_dir):
        directory.mkdir(mode=0o700)

    env = dict(os.environ)
    env.update(
        {
            "HOME": str(home),
            "USERPROFILE": str(home),
            "XDG_DATA_HOME": str(data),
            "XDG_CONFIG_HOME": str(config_dir),
            "TLDW_CONFIG_PATH": str(config_dir / "config.toml"),
            "TLDW_TEST_MODE": "1",
        }
    )
    env.pop("PYTEST_CURRENT_TEST", None)

    # First construct creates/migrates the profile (this is where the fresh
    # Chunking migration legitimately runs); the second, in a fresh
    # interpreter, is the recurring warm case the pin is against.
    _run_construct_probe(env)
    new_modules = _run_construct_probe(env)

    # Anti-vacuity: a warm construct that imports NOTHING would mean the
    # diff seam broke (or every row below was deferred at once -- in which
    # case retire this guard deliberately, don't let it idle green).
    assert new_modules, (
        "construct-time import census recorded no new modules at all; the "
        "probe's diff seam is not measuring construction."
    )

    unlisted = sorted(set(new_modules) - ALLOWED_CONSTRUCT_IMPORTS)
    assert not unlisted, (
        f"TldwCli.__init__ newly imported repo modules that are not on the "
        f"reviewed allowlist: {unlisted}. Function-level imports here are "
        "invisible to every import guard and their cost lands on every "
        "boot. Prefer deferring to first feature use; otherwise add the "
        "row with the owning feature named, in the same commit."
    )

    leaked = sorted(
        m
        for m in new_modules
        if any(
            m == p or m.startswith(p + ".")
            for p in FORBIDDEN_ON_WARM_CONSTRUCT_PREFIXES
        )
    )
    assert not leaked, (
        f"migration-only families imported on a WARM construct: {leaked}. "
        "The Chunking engine belongs to fresh-profile migration "
        "(media-DB v6->v7) only -- something moved it onto the recurring "
        "boot path."
    )

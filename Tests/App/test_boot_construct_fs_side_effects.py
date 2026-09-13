"""Boot census: construct-time filesystem side effects (task-22216).

TASK-21105's guard (test_boot_no_feature_db_files.py) asserts six feature
DB *filenames* never appear after ``TldwCli()`` construction. PR #1998
reintroduced the class that guard cannot see: ``ActorPackImportService``'s
``__init__`` ran ``sweep_staging()`` — ``secure_private_directory(...,
create=True)`` (a per-component ownership/mode walk) plus an ``os.scandir``
sweep of staging candidates — synchronously inside ``TldwCli.__init__``,
every boot, before the event loop exists. It creates a *directory*, not a
DB file, so the filename census stayed green.

This guard boots the real app in a subprocess against a scratch profile
with counters installed at the importer seam and asserts:

- zero ``secure_private_directory`` calls resolve through
  ``tldw_chatbook.Actor_Packs.importer`` during construction;
- zero ``os.scandir`` calls target the actor-pack staging directory
  during construction;
- the staging directory (``actor_pack_imports``) does not exist after
  construction — the sweep belongs to the deferred startup worker or the
  first import-feature use, never the construction path.

Honest blind spots (this guard is a fixed-list tripwire, not a tracer):

- It names ONE side-effect class instance: the actor-pack staging sweep
  seam plus one directory name. A new construct-time filesystem side
  effect introduced through any other seam (or a renamed staging root) is
  invisible until a row is added here.
- The ``secure_private_directory`` counter counts only calls resolved
  through the importer module's namespace; another module calling its own
  direct import of the helper is not attributed.
- The ``os.scandir`` counter matches on the staging directory name in the
  path argument; fd-based (``dir_fd``) or renamed-root traversal escapes.
- Construction only: side effects deferred to mount/first-paint hooks are
  out of scope (deliberately — that is where deferred work SHOULD run),
  and the boot runs with ``TLDW_TEST_MODE=1`` like every boot guard, so a
  code path gated off test mode is not exercised.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

STAGING_DIRECTORY_NAME = "actor_pack_imports"

_BOOT_SCRIPT = """
import os
import sys
from pathlib import Path

import tldw_chatbook.Actor_Packs.importer as importer

counts = {"secure_private_directory": 0, "staging_scandir": 0}

_real_secure = importer.secure_private_directory

def _counting_secure(path, *args, **kwargs):
    counts["secure_private_directory"] += 1
    return _real_secure(path, *args, **kwargs)

importer.secure_private_directory = _counting_secure

_real_scandir = os.scandir

def _counting_scandir(path=".", *args, **kwargs):
    try:
        rendered = os.fspath(path)
    except TypeError:
        rendered = ""
    if isinstance(rendered, bytes):
        rendered = rendered.decode(errors="replace")
    if "actor_pack_imports" in str(rendered):
        counts["staging_scandir"] += 1
    return _real_scandir(path, *args, **kwargs)

os.scandir = _counting_scandir
try:
    from tldw_chatbook.app import TldwCli

    app = TldwCli()
finally:
    os.scandir = _real_scandir
    importer.secure_private_directory = _real_secure

print("COUNT secure_private_directory", counts["secure_private_directory"])
print("COUNT staging_scandir", counts["staging_scandir"])

scratch = Path(sys.argv[1])
for path in sorted(scratch.rglob("*")):
    if path.is_dir():
        print("DIR", path.name)
    elif path.is_file():
        print("FILE", path.name)
"""


@pytest.mark.integration
def test_construction_performs_no_staging_sweep_filesystem_io(
    tmp_path: Path,
) -> None:
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

    result = subprocess.run(
        [sys.executable, "-c", _BOOT_SCRIPT, str(tmp_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, (
        f"app boot failed (rc={result.returncode}):\n{result.stderr[-4000:]}"
    )

    lines = result.stdout.split("\n")
    counts = {
        parts[1]: int(parts[2])
        for parts in (line.split() for line in lines)
        if len(parts) == 3 and parts[0] == "COUNT"
    }
    files = {
        parts[1] for parts in (line.split() for line in lines) if parts[:1] == ["FILE"]
    }
    directories = {
        parts[1] for parts in (line.split() for line in lines) if parts[:1] == ["DIR"]
    }

    # Guard against a silent no-op boot: construction MUST have created the
    # core profile stores, or this test is not measuring a real boot.
    assert "tldw_chatbook_ChaChaNotes.db" in files, (
        f"boot census looks empty/degenerate: {sorted(files)}"
    )

    assert counts.get("secure_private_directory") == 0, (
        "TldwCli.__init__ reached secure_private_directory through the "
        "actor-pack importer — the staging sweep is back on the "
        f"construction path (task-22216 regression): {counts}"
    )
    assert counts.get("staging_scandir") == 0, (
        "TldwCli.__init__ scandir'd the actor-pack staging directory — the "
        f"staging sweep is back on the construction path: {counts}"
    )
    assert STAGING_DIRECTORY_NAME not in directories, (
        f"{STAGING_DIRECTORY_NAME}/ was created during a boot that never "
        f"touched the import feature: {sorted(directories)}"
    )

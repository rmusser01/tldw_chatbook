"""Boot census: feature databases must not exist after app construction.

TASK-21105: seven feature databases used to be created and schema'd
synchronously inside ``TldwCli.__init__`` for features a never-user does
not touch. Six of them (research, writing, kanban, notifications, and the
event_state/sync_state server-parity stores) are now open-on-first-use.

This test boots the real app -- ``TldwCli()`` construction, no mount --
in a subprocess against a scratch profile and asserts none of the six
files (or their WAL sidecars) exist, while core stores that ARE part of
construction (ChaChaNotes) do exist, proving the boot actually ran.

The notes_sync_state store is deliberately NOT asserted absent here: its
lifecycle is being gated separately (TASK-21112). It is, however, subject
to the allowlist census below -- if its gating ever puts it on the
construction path, that lands here as a reviewed row, not silently.

TASK-22222 extension -- the fixed list's blind spot, stated honestly: the
six-name absence list can only catch the six regressions it names. A
SEVENTH feature store, added eagerly to ``TldwCli.__init__`` under any
other filename, passed this guard silently for months by design. The
second test below closes the *database* half of that gap: it censuses
every ``*.db`` file in the scratch profile after construction against an
allowlist of the stores construction is KNOWN to create, so a new
construct-time database is a reviewed decision (add the row, name the
feature, same commit) rather than a silent boot cost.

Blind spots that remain even with the census (be honest about them):

* Databases only. Non-DB side effects -- directories, staging sweeps,
  lock/marker files, config writes -- are invisible here; the staging-sweep
  class has its own guard (``test_boot_construct_fs_side_effects.py``,
  task-22216), which is itself a named-seam tripwire, not a tracer.
* The census walks the scratch profile tree. A store opened OUTSIDE the
  profile (absolute path bug, ``/tmp``, cwd) escapes the walk entirely.
* Construction only, ``TLDW_TEST_MODE=1`` (like every boot guard): stores
  created on mount, on timers, or on paths gated off test mode are out of
  scope -- deliberately, that is where deferred work SHOULD run.
* An allowlisted store that grows heavier schema/migration work inside
  construction is invisible: this census counts files, not cost.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

LAZY_FEATURE_DB_FILENAMES = (
    "tldw_chatbook_research.db",
    "tldw_chatbook_writing.db",
    "tldw_chatbook_kanban.db",
    "tldw_chatbook_notifications.db",
    "tldw_chatbook_event_state.db",
    "tldw_chatbook_sync_state.db",
)

#: TASK-22222: every ``*.db`` file construction may create in the profile.
#: Measured 2026-08-25 (fresh scratch profile, base dev f0e896122). Adding
#: a row is the reviewed decision the census exists to force -- name the
#: feature that now opens a store during ``TldwCli.__init__`` and why it
#: cannot be open-on-first-use (TASK-21105 is the precedent for deferring).
CONSTRUCT_TIME_DB_FILENAMES = frozenset(
    {
        "evals.db",
        "tldw_chatbook_ChaChaNotes.db",
        "tldw_chatbook_library_collections.db",
        "tldw_chatbook_media_v2.db",
        "tldw_chatbook_prompts.db",
        "tldw_chatbook_scheduled_tasks.db",
        "tldw_chatbook_subscriptions.db",
        "tldw_chatbook_workspaces.db",
    }
)

#: SQLite sidecar suffixes, stripped so the census pins base stores, not
#: journal modes.
_SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")

_BOOT_SCRIPT = """
import sys
from pathlib import Path

from tldw_chatbook.app import TldwCli

app = TldwCli()

scratch = Path(sys.argv[1])
for path in sorted(scratch.rglob("*")):
    if path.is_file():
        print(path.name)
"""


def _boot_and_list_files(tmp_path: Path) -> set[str]:
    """Construct ``TldwCli()`` in a subprocess; return created file names.

    Args:
        tmp_path: Isolated scratch directory the boot's profile lives in.

    Returns:
        Every file name found under the scratch tree after construction.
    """
    home = tmp_path / "home"
    data = tmp_path / "data"
    config_dir = tmp_path / "config"
    for directory in (home, data, config_dir):
        directory.mkdir(mode=0o700, exist_ok=True)

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

    created = set(result.stdout.split())
    # Guard against a silent no-op boot: construction MUST have created the
    # core profile stores, or this test is not measuring a real boot.
    assert "tldw_chatbook_ChaChaNotes.db" in created, (
        f"boot census looks empty/degenerate: {sorted(created)}"
    )
    return created


@pytest.mark.integration
def test_boot_without_feature_use_creates_no_feature_db_files(
    tmp_path: Path,
) -> None:
    created = _boot_and_list_files(tmp_path)

    for filename in LAZY_FEATURE_DB_FILENAMES:
        offenders = sorted(
            name
            for name in created
            if name == filename or name.startswith(filename + "-")
        )
        assert offenders == [], (
            f"{filename} (or a WAL sidecar) was created during a boot that "
            f"never touched the feature: {offenders}"
        )


@pytest.mark.integration
def test_boot_creates_only_allowlisted_db_files(tmp_path: Path) -> None:
    """Every ``*.db`` file construction creates is a reviewed allowlist row.

    The six-name test above can only catch the six regressions it names;
    this census catches the seventh store under ANY name (see the module
    docstring for the incident and the remaining blind spots).

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's profile.
    """
    created = _boot_and_list_files(tmp_path)

    db_stores = set()
    for name in created:
        base = name
        for suffix in _SQLITE_SIDECAR_SUFFIXES:
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        if base.endswith(".db"):
            db_stores.add(base)

    unlisted = sorted(db_stores - CONSTRUCT_TIME_DB_FILENAMES)
    assert not unlisted, (
        f"TldwCli() construction created database stores that are not on "
        f"the reviewed allowlist: {unlisted}. A construct-time store costs "
        "every boot (open + schema + migration checks) whether or not the "
        "feature is ever used -- prefer open-on-first-use (TASK-21105's "
        "precedent); if construction genuinely needs it, add the row with "
        "the owning feature named, in the same commit."
    )

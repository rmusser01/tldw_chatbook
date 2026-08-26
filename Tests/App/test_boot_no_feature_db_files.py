"""Boot census: feature databases must not exist after app construction.

TASK-21105: seven feature databases used to be created and schema'd
synchronously inside ``TldwCli.__init__`` for features a never-user does
not touch. Six of them (research, writing, kanban, notifications, and the
event_state/sync_state server-parity stores) are now open-on-first-use.

This test boots the real app -- ``TldwCli()`` construction, no mount --
in a subprocess against a scratch profile and asserts none of the six
files (or their WAL sidecars) exist, while core stores that ARE part of
construction (ChaChaNotes) do exist, proving the boot actually ran.

The notes_sync_state store is deliberately NOT asserted here: its
lifecycle is being gated separately (TASK-21112).
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


@pytest.mark.integration
def test_boot_without_feature_use_creates_no_feature_db_files(
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

    created = set(result.stdout.split())
    # Guard against a silent no-op boot: construction MUST have created the
    # core profile stores, or this test is not measuring a real boot.
    assert "tldw_chatbook_ChaChaNotes.db" in created, (
        f"boot census looks empty/degenerate: {sorted(created)}"
    )

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

# Tests/App/test_umask002_fallback_root_boot.py
"""
Two-boot regression for task-32901, reproducing a fresh Ubuntu 24.04 box
with umask 002: ~/.local/share lands group-writable (0775), so the ADR-127
fallback data root engages on the first boot.

Boot one used to create ~/.tldw_cli-data/default_user/chat_dicts with a
umask-derived mkdir (0775); boot two then refused it through the config
participant admission layer ("Could not create chat dictionaries folder ...
shared_writable_parent"), and ActorPackImportService rejected its own
profile root because the fallback root is a dotted base directory,
crashing TldwCli.__init__. After the fix: chat_dicts is created 0700 via
the private lifecycle (and an existing 0775 instance is hardened), and the
actor-pack roots validate under the fallback root.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

_BOOT_SCRIPT = """
import stat
import sys
from pathlib import Path

from tldw_chatbook.config import get_user_data_dir, load_settings

load_settings()
data_root = get_user_data_dir()
chat_dicts = data_root / "chat_dicts"
assert chat_dicts.exists(), chat_dicts
mode = stat.S_IMODE(chat_dicts.stat().st_mode)
print(f"CHAT_DICTS_MODE={oct(mode)}")

from tldw_chatbook.Actor_Packs.importer import _absolute_path

_absolute_path(data_root, allow_hidden=True)  # profile root wiring path
print("ACTOR_PACK_ROOT_OK")
"""


def _boot(home: Path) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "HOME": str(home),
        "TLDW_TEST_MODE": "1",
        "PYTHONPATH": str(REPO_ROOT),
    }
    env.pop("TLDW_CONFIG_PATH", None)
    env.pop("PYTEST_CURRENT_TEST", None)
    return subprocess.run(
        ["/bin/sh", "-c", f"umask 002; exec {sys.executable} -"],
        input=_BOOT_SCRIPT,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )


@pytest.mark.skipif(os.name != "posix", reason="POSIX umask/mode contract")
def test_two_boots_under_umask_002_with_fallback_root(tmp_path: Path) -> None:
    pytest.importorskip("portalocker", reason="config import needs hard deps")
    home = tmp_path / "home"
    share = home / ".local" / "share"
    share.mkdir(parents=True)
    share.chmod(0o775)  # group-writable -> ADR-127 fallback engages

    first = _boot(home)
    assert first.returncode == 0, first.stderr
    # Boot one already creates the folder hardened; a permissive mode here
    # is exactly the bug (it would poison boot two's admission check).
    assert "CHAT_DICTS_MODE=0o700" in first.stdout, first.stdout
    assert "ACTOR_PACK_ROOT_OK" in first.stdout

    # Simulate the poisoned pre-fix state: a pre-existing 0775 instance is
    # refused by the admission pre-check, but the boot must SURVIVE and name
    # the exact one-command repair instead of failing cryptically.
    chat_dicts = home / ".tldw_cli-data" / "default_user" / "chat_dicts"
    assert chat_dicts.exists()
    chat_dicts.chmod(0o775)
    second = _boot(home)
    assert second.returncode == 0, second.stderr
    assert "CHAT_DICTS_MODE=0o775" in second.stdout  # degraded but alive
    assert "ACTOR_PACK_ROOT_OK" in second.stdout
    assert f"chmod g-w,o-w -- {chat_dicts}" in second.stderr

    # Follow the printed repair, boot three is fully clean.
    chat_dicts.chmod(0o700)
    third = _boot(home)
    assert third.returncode == 0, third.stderr
    assert "CHAT_DICTS_MODE=0o700" in third.stdout
    assert "ACTOR_PACK_ROOT_OK" in third.stdout
    assert "Could not create chat dictionaries" not in third.stderr

"""A reboot that renumbers the volume must not lock the app out (TASK-34200).

macOS can give a volume a new ``st_dev`` after a reboot. The admission registry
pinned ``inode:<dev>:<ino>``, so after the owner's machine rebooted on
2026-09-28 every start failed with "Recovery required: recovery_scope_uncertain"
-- same file, same inode, same path, device 16777234 -> 16777230. Identity is
now path + inode; a registry recorded in the old format keeps matching.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.Backup_Recovery.control_records import admission_authority


def _registry(root: Path) -> Path:
    return root / "admission" / "registry.json"


def _marker(root: Path) -> Path:
    return root / "unbound-owner"


def _record_as_before_the_fix(root: Path, device: int) -> None:
    """Rewrite the stored inode token in the old ``inode:<dev>:<ino>`` form."""
    path = _registry(root)
    data = json.loads(path.read_text())
    entry = data["entries"]["bootstrap.unbound"]
    inode = os.stat(_marker(root)).st_ino
    entry["historical"] = sorted(
        [t for t in entry["historical"] if not t.startswith("inode:")]
        + [f"inode:{device}:{inode}"]
    )
    path.write_text(json.dumps(data))
    os.chmod(path, 0o600)


def test_a_registry_recorded_before_a_renumbering_still_admits(tmp_path: Path) -> None:
    """The owner's case: same file and inode, a different device number.

    Args:
        tmp_path: Holds the bootstrap root whose registry is rewritten.
    """
    root = tmp_path / "recovery-bootstrap"
    admission_authority(root)
    _record_as_before_the_fix(root, device=os.stat(root).st_dev + 4)
    admission_authority(root)  # raised RecoveryRequired before TASK-34200


def test_a_replaced_marker_is_still_refused(tmp_path: Path) -> None:
    """The fence still holds: a new file at the same path has a new inode.

    Args:
        tmp_path: Holds the bootstrap root whose marker is replaced.
    """
    root = tmp_path / "recovery-bootstrap"
    admission_authority(root)
    marker = _marker(root)
    replacement = root / "replacement"
    replacement.write_bytes(marker.read_bytes())
    os.chmod(replacement, 0o600)
    os.replace(replacement, marker)  # created before the old one is gone
    with pytest.raises(RecoveryRequired):
        admission_authority(root)


def test_new_registrations_record_a_device_free_token(tmp_path: Path) -> None:
    """New registries never pin the device again.

    Args:
        tmp_path: Holds the freshly registered bootstrap root.
    """
    root = tmp_path / "recovery-bootstrap"
    admission_authority(root)
    historical = json.loads(_registry(root).read_text())["entries"]["bootstrap.unbound"]["historical"]
    assert [t for t in historical if t.startswith("inode:")] == [
        f"inode:{os.stat(_marker(root)).st_ino}"
    ]


def test_identity_view_reads_both_token_formats() -> None:
    """Old ``inode:<dev>:<ino>`` and new ``inode:<ino>`` compare equal; paths untouched."""
    assert bootstrap.identity_view({"inode:16777234:9", "inode:9", "path:/x"}) == {
        "inode:9",
        "path:/x",
    }
    # Malformed legacy tokens are not normalized into a match (Qodo #2994).
    assert bootstrap.identity_view({"inode:invalid:9", "inode:5:x"}) == {
        "inode:invalid:9",
        "inode:5:x",
    }


def test_a_malformed_stored_token_cannot_admit_the_marker(tmp_path: Path) -> None:
    """A non-numeric device field is rejected, not normalized (Qodo #2994).

    Args:
        tmp_path: Holds the bootstrap root whose registry is corrupted.
    """
    root = tmp_path / "recovery-bootstrap"
    admission_authority(root)
    path = _registry(root)
    data = json.loads(path.read_text())
    entry = data["entries"]["bootstrap.unbound"]
    inode = os.stat(_marker(root)).st_ino
    entry["historical"] = sorted(
        [t for t in entry["historical"] if not t.startswith("inode:")] + [f"inode:invalid:{inode}"]
    )
    path.write_text(json.dumps(data))
    os.chmod(path, 0o600)
    with pytest.raises(RecoveryRequired):
        admission_authority(root)


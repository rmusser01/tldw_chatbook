"""Native mount admission rejects detached or incomplete Linux mount records."""

from pathlib import Path

import pytest

from tldw_chatbook.Backup_Recovery.native_platform import _linux_mount_type


@pytest.mark.parametrize(
    "descriptor,mounts",
    [
        ("", ""),
        ("mnt_id:\n", ""),
        ("mnt_id: 17\n", "18 1 0:1 / / rw - ext4 /dev/a rw"),
        ("mnt_id: 17\n", "17 1 incomplete"),
        ("mnt_id: 17\n", "17 1 - "),
    ],
)
def test_unavailable_mount_record_is_operational_refusal(
    monkeypatch, descriptor, mounts
):
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path: (
            descriptor if str(path).startswith("/proc/self/fdinfo/") else mounts
        ),
    )
    with pytest.raises(OSError, match="native_mount_identity_unavailable"):
        _linux_mount_type(9)


def test_mount_type_is_bound_to_descriptor_mount_id(monkeypatch):
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path: (
            "mnt_id: 17\n"
            if str(path).startswith("/proc/self/fdinfo/")
            else "16 1 0:1 / / rw - ext2 /dev/a rw\n17 1 0:2 / /home rw - ext4 /dev/b rw"
        ),
    )
    assert _linux_mount_type(9) == "ext4"

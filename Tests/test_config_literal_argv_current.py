"""Current selected config writer preserves literal executable argv values."""

import tomllib

import pytest

from Tests.Backup_Recovery.test_config_participant_lifetimes import source as _source
from Tests.Backup_Recovery.test_participant_lifetimes import (
    local_root,  # noqa: F401 - exact source fixture dependency
)

source = _source


@pytest.mark.parametrize(
    "command",
    [
        r"C:\Users\fixture\python.exe",
        r"C:\hostedtoolcache\windows\Python\3.12.10\x64\python.exe",
    ],
    ids=["ordinary", "literal-x64"],
)
def test_current_private_writer_literal_argv_roundtrip(source, command):
    document = {
        "general": {"users_name": "literal-argv-control"},
        "hooks": {
            "enabled": True,
            "hook": [{"id": "literal-argv", "command": [command, "-c", "pass"]}],
        },
    }
    loaded = source.replace_cli_config(document)
    parsed = tomllib.loads(source.read_cli_config_serialized())
    assert parsed == document
    assert loaded["hooks"] == document["hooks"]
    assert source.load_settings()["hooks"] == document["hooks"]


def test_missing_source_export_preserves_literal_argv(source):
    document = {"hooks": {"command": [r"C:\x64\python.exe", r"--literal=\x41"]}}
    source.get_cli_config_path().unlink()
    exported = source.export_cli_config_snapshot(document, timestamp="literal_argv")
    assert tomllib.loads(exported.read_text()) == document

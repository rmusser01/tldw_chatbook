"""The audit must reject unsafe setup before importing the app."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "ingest_lifecycle_native", Path(__file__).with_name("native_check.py")
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def write_profile(root, data, media):
    (root / "config.toml").write_text(
        f'[paths]\ndata_dir = "{data}"\n'
        f'[database]\nUSER_DB_BASE_DIR = "{root / "data/db"}"\n'
        f'media_db_path = "{media}"\n'
    )


def test_private_profile_is_accepted(tmp_path):
    (tmp_path / "data/db").mkdir(parents=True)
    write_profile(tmp_path, tmp_path / "data", tmp_path / "data/db/media.db")
    runner.validate_profile(tmp_path)


def test_missing_profile_is_rejected_without_creating_it(tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        runner.validate_profile(tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("escape", ["data", "database", "symlink", "missing_directory"])
def test_profile_escape_or_missing_parent_is_rejected(tmp_path, escape):
    root = tmp_path / "profile"
    (root / "data/db").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    data, media = root / "data", root / "data/db/media.db"
    if escape == "data":
        data = outside
    elif escape == "database":
        media = outside / "media.db"
    elif escape == "symlink":
        (root / "alias").symlink_to(outside, target_is_directory=True)
        media = root / "alias/media.db"
    else:
        data = root / "missing"
    write_profile(root, data, media)
    with pytest.raises(ValueError):
        runner.validate_profile(root)
    assert not list(outside.iterdir())

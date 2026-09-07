from __future__ import annotations

import os
from pathlib import Path

import pytest

from tldw_chatbook.Utils.private_paths import PrivatePathError
from tldw_chatbook.Widgets.Tamagotchi import SQLiteStorage


def _pet_state(name: str = "Pixel") -> dict[str, object]:
    return {
        "name": name,
        "happiness": 75,
        "hunger": 25,
        "energy": 80,
        "health": 100,
        "age": 3,
        "personality": "balanced",
        "is_alive": True,
        "total_interactions": 4,
        "sprite_theme": "emoji",
        "favorite_snack": "ramen",
    }


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode contract")
def test_sqlite_storage_file_crud_uses_private_database(tmp_path: Path) -> None:
    storage = SQLiteStorage(str(tmp_path / "pets.db"))
    try:
        assert storage.save("pixel", _pet_state())
        loaded = storage.load("pixel")
        assert loaded is not None
        assert loaded["name"] == "Pixel"
        assert loaded["favorite_snack"] == "ramen"
        assert storage.list_pets() == ["pixel"]
        assert storage.get_statistics()["total_pets"] == 1
        assert storage.delete("pixel")
        assert storage.load("pixel") is None
        assert (tmp_path / "pets.db").stat().st_mode & 0o777 == 0o600
    finally:
        storage.close()


def test_sqlite_storage_path_memory_persists_across_crud() -> None:
    storage = SQLiteStorage(Path(":memory:"))
    try:
        assert storage.save("pixel", _pet_state())
        assert storage.load("pixel")["name"] == "Pixel"
        assert storage.list_pets() == ["pixel"]
        assert storage.get_statistics()["total_pets"] == 1
        assert storage.delete("pixel")
    finally:
        storage.close()


def test_sqlite_storage_missing_parent_fails_before_creation(tmp_path: Path) -> None:
    database = tmp_path / "missing" / "pets.db"

    with pytest.raises((FileNotFoundError, PrivatePathError)):
        SQLiteStorage(str(database))

    assert not database.parent.exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX trust contract")
def test_sqlite_storage_unsafe_parent_fails_closed(tmp_path: Path) -> None:
    unsafe_parent = tmp_path / "unsafe"
    unsafe_parent.mkdir(mode=0o700)
    unsafe_parent.chmod(0o777)
    try:
        with pytest.raises(PrivatePathError):
            SQLiteStorage(str(unsafe_parent / "pets.db"))
    finally:
        os.chmod(unsafe_parent, 0o700)

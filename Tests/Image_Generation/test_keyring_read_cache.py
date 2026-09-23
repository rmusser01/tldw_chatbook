"""TASK-32924: Settings' Image/Video Gen panels resolve secrets in compose().

compose() runs on the UI loop on every open/save/revert/Test and read the OS
keyring once per backend lacking an env/config key -- up to 7 SecretService
D-Bus round trips per compose on Linux.
"""

from tldw_chatbook.Image_Generation.config import _TABLES
from tldw_chatbook.Media_Generation import config_machinery


def test_repeated_keyring_lookups_share_one_round_trip(monkeypatch):
    clock = [500.0]
    reads = []
    monkeypatch.setattr(config_machinery.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(config_machinery, "_KEYRING_READS", {})
    monkeypatch.setattr(
        config_machinery.keyring,
        "get_password",
        lambda service, backend: reads.append(backend) or "kr-" + backend,
    )

    for _ in range(5):
        assert config_machinery.keyring_get("novita", _TABLES) == "kr-novita"
    assert reads == ["novita"]

    clock[0] += 11  # a key added outside the app appears after the window
    config_machinery.keyring_get("novita", _TABLES)
    assert reads == ["novita", "novita"]


def test_failed_lookups_are_shared_too(monkeypatch):
    reads = []

    def locked(service, backend):
        reads.append(backend)
        raise RuntimeError("keyring locked")

    monkeypatch.setattr(config_machinery, "_KEYRING_READS", {})
    monkeypatch.setattr(config_machinery.keyring, "get_password", locked)
    for _ in range(3):
        assert config_machinery.keyring_get("fal", _TABLES) is None
    assert reads == ["fal"]

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


def test_slow_lookups_are_cached_from_when_they_returned(monkeypatch):
    """Qodo review on #2820: expiry started before a blocking lookup, so a
    lookup slower than the window was stored already expired."""
    clock = [500.0]
    reads = []
    monkeypatch.setattr(config_machinery.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(config_machinery, "_KEYRING_READS", {})

    def slow(service, backend):
        reads.append(backend)
        clock[0] += 15  # longer than the 10 s window
        raise RuntimeError("keyring locked")

    monkeypatch.setattr(config_machinery.keyring, "get_password", slow)
    config_machinery.keyring_get("fal", _TABLES)
    config_machinery.keyring_get("fal", _TABLES)
    assert reads == ["fal"]


def test_repeated_settings_compose_config_loads_share_keyring_reads(monkeypatch):
    """Qodo review on #2820 (integration): the Image Gen panel's compose()
    calls get_image_generation_config(reload=True) on every open/save/revert/
    Test. Repeating that real load must not repeat the keyring reads."""
    from tldw_chatbook.Image_Generation import config as image_config

    reads = []
    monkeypatch.setattr(config_machinery, "_KEYRING_READS", {})
    monkeypatch.setattr(
        config_machinery.keyring,
        "get_password",
        lambda service, backend: reads.append(backend),
    )
    monkeypatch.setattr(image_config, "_read_image_generation_toml", dict)
    image_config.get_image_generation_config(reload=True)
    first = len(reads)
    for _ in range(4):
        image_config.get_image_generation_config(reload=True)
    assert first > 0, "fixture must reach the keyring (no env/config keys)"
    assert len(reads) == first

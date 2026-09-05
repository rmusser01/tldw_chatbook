from __future__ import annotations

import tomllib

import pytest
from pydantic import ValidationError

from tldw_chatbook import config
from tldw_chatbook.LLM_Management.snapshot_settings import (
    SnapshotPreferences,
    load_snapshot_preferences,
    save_snapshot_preferences,
)


@pytest.mark.parametrize("bad", [0, 1001, True, 1.5, "10"])
def test_keep_count_rejects_non_integer_or_out_of_range(bad: object) -> None:
    """A coerced or out-of-range retention value must never reach persistence."""
    with pytest.raises(ValidationError):
        SnapshotPreferences(keep_count=bad)


def test_snapshot_defaults_are_opt_in_and_keep_ten() -> None:
    """A caller that omits both preferences receives the approved safe defaults."""
    assert SnapshotPreferences().model_dump() == {"enabled": False, "keep_count": 10}


def test_snapshot_expected_pair_rejects_update_at_locked_mutation(monkeypatch):
    from tldw_chatbook.LLM_Management import snapshot_settings as preferences

    before = SnapshotPreferences()
    newer = SnapshotPreferences(enabled=True, keep_count=12)
    desired = SnapshotPreferences(enabled=False, keep_count=20)
    assert save_snapshot_preferences(before)

    def interleaved(values, **kwargs):
        assert config.apply_settings_mutation_to_cli_config(
            {"llamacpp_snapshots": newer.model_dump()}
        ).fully_applied
        return config.apply_settings_mutation_to_cli_config(values, **kwargs)

    monkeypatch.setattr(
        preferences, "apply_settings_mutation_to_cli_config", interleaved
    )
    with pytest.raises(preferences.SnapshotPreferencesConflict):
        save_snapshot_preferences(desired, expected=before)
    assert load_snapshot_preferences() == newer


def test_snapshot_expected_default_pair_and_noop_succeed():
    before = SnapshotPreferences()
    config.apply_settings_mutation_to_cli_config(
        {}, delete_keys={"llamacpp_snapshots": ["enabled", "keep_count"]}
    )
    assert save_snapshot_preferences(before, expected=before)
    assert save_snapshot_preferences(before, expected=before)
    desired = SnapshotPreferences(enabled=True, keep_count=1000)
    assert save_snapshot_preferences(desired, expected=before)
    assert load_snapshot_preferences() == desired


def test_shipping_config_template_contains_the_snapshot_defaults() -> None:
    """A fresh profile must persist the same defaults exposed by the model."""
    template = tomllib.loads(config.CONFIG_TOML_CONTENT)

    assert template["llamacpp_snapshots"] == {
        "enabled": False,
        "keep_count": 10,
    }


def test_load_preferences_uses_real_isolated_config_owner(
    tmp_path, monkeypatch
) -> None:
    """Loading a missing profile creates and reads the shipping config section."""
    config_path = tmp_path / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    assert load_snapshot_preferences() == SnapshotPreferences()
    persisted = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert persisted["llamacpp_snapshots"] == {
        "enabled": False,
        "keep_count": 10,
    }


def test_changed_count_round_trips_through_real_config_owner(
    tmp_path, monkeypatch
) -> None:
    """The batch persistence seam must publish a changed effective keep count."""
    config_path = tmp_path / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    value = SnapshotPreferences(enabled=True, keep_count=37)

    assert save_snapshot_preferences(value) is True
    assert load_snapshot_preferences() == value


def test_failed_save_leaves_disk_and_effective_preferences_unchanged(
    tmp_path, monkeypatch
) -> None:
    """A failed atomic save must not claim or expose a new preference value."""
    config_path = tmp_path / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    before = SnapshotPreferences(enabled=False, keep_count=8)
    assert save_snapshot_preferences(before) is True
    before_bytes = config_path.read_bytes()

    monkeypatch.setattr(
        "tldw_chatbook.LLM_Management.snapshot_settings.save_settings_to_cli_config",
        lambda _sections: False,
    )

    assert (
        save_snapshot_preferences(SnapshotPreferences(enabled=True, keep_count=2))
        is False
    )
    assert config_path.read_bytes() == before_bytes
    assert load_snapshot_preferences() == before


@pytest.mark.parametrize(
    "body",
    [
        '[llamacpp_snapshots]\nenabled = "yes"\nkeep_count = 10\n',
        "[llamacpp_snapshots]\nenabled = false\nkeep_count = 0\n",
        '[llamacpp_snapshots]\nenabled = false\nkeep_count = "10"\n',
    ],
)
def test_malformed_config_is_a_visible_validation_failure(
    body: str, tmp_path, monkeypatch
) -> None:
    """Malformed persisted values must fail closed instead of enabling actions."""
    config_path = tmp_path / "config.toml"
    config_path.write_text(body, encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    with pytest.raises(ValidationError):
        load_snapshot_preferences()

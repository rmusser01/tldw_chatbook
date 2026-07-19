import tomllib
from types import SimpleNamespace

import tldw_chatbook.app as app_module
from tldw_chatbook.app import LibraryIngestQueueMixin
from tldw_chatbook.config import CONFIG_TOML_CONTENT


def test_heavy_lane_default_when_unset(monkeypatch):
    monkeypatch.setattr(app_module, "get_cli_setting", lambda *a, **k: None)
    assert LibraryIngestQueueMixin._ingest_heavy_lane_max_workers(SimpleNamespace()) == 1


def test_heavy_lane_uses_configured_value(monkeypatch):
    monkeypatch.setattr(app_module, "get_cli_setting", lambda *a, **k: 2)
    assert LibraryIngestQueueMixin._ingest_heavy_lane_max_workers(SimpleNamespace()) == 2


def test_heavy_lane_clamps_non_positive_to_one(monkeypatch):
    monkeypatch.setattr(app_module, "get_cli_setting", lambda *a, **k: 0)
    assert LibraryIngestQueueMixin._ingest_heavy_lane_max_workers(SimpleNamespace()) == 1


def test_config_template_valid_toml_and_heavy_lane_key_commented():
    parsed = tomllib.loads(CONFIG_TOML_CONTENT)   # must not raise
    # The key is documented as a COMMENT, so a fresh template parse must not
    # set it -- keeping the runtime default (1) in force.
    assert "ingest_heavy_lane_max_workers" not in parsed.get("library", {})
    assert "ingest_heavy_lane_max_workers" in CONFIG_TOML_CONTENT  # documented (commented)

"""TASK-658: quick_ingest() ignored [database] media_db_path.

``get_cli_setting("database", {})`` passes a non-string in the *key* slot for
a section name with no dot, and config.py returns the default before reading
anything. The configured path was silently discarded.

Note on patch targets: ``quick_ingest`` does
``from ..config import get_cli_setting`` *inside* the function body, so the
name is re-resolved from ``tldw_chatbook.config`` on every call -- patching
``tldw_chatbook.Local_Ingestion.local_file_ingestion.get_cli_setting`` (the
module attribute) has no effect, since that local import always shadows it.
The real seam is ``tldw_chatbook.config.get_cli_setting``. ``MediaDatabase``
and ``ingest_local_file`` *are* module-level names in
``local_file_ingestion.py`` (import and same-module def respectively), so
patching them on the ``lfi`` module works as expected.
"""

import pytest


def test_configured_media_db_path_is_honored(tmp_path, monkeypatch):
    import tldw_chatbook.Local_Ingestion.local_file_ingestion as lfi
    import tldw_chatbook.config as config_module

    configured = tmp_path / "configured_media.db"

    seen = {}

    class _FakeMediaDatabase:
        def __init__(self, db_path, client_id):
            seen["db_path"] = db_path

        def close_connection(self):
            pass

    monkeypatch.setattr(lfi, "MediaDatabase", _FakeMediaDatabase)
    monkeypatch.setattr(lfi, "ingest_local_file", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(
        config_module,
        "get_cli_setting",
        lambda section, key=None, default=None: (
            str(configured) if (section, key) == ("database", "media_db_path") else default
        ),
    )

    lfi.quick_ingest(tmp_path / "some_file.txt")
    assert seen["db_path"] == str(configured)


def test_fallback_applies_only_when_the_key_is_absent(tmp_path, monkeypatch):
    import tldw_chatbook.Local_Ingestion.local_file_ingestion as lfi
    import tldw_chatbook.config as config_module

    seen = {}

    class _FakeMediaDatabase:
        def __init__(self, db_path, client_id):
            seen["db_path"] = db_path

        def close_connection(self):
            pass

    monkeypatch.setattr(lfi, "MediaDatabase", _FakeMediaDatabase)
    monkeypatch.setattr(lfi, "ingest_local_file", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(
        config_module,
        "get_cli_setting",
        lambda section, key=None, default=None: default,
    )

    lfi.quick_ingest(tmp_path / "some_file.txt")
    assert "tldw_cli_media_v2.db" in seen["db_path"]

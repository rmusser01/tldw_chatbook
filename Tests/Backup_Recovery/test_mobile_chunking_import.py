"""Pure Chunking imports stay inert; runtime exports retain native identities."""

import importlib
import json
import os
import pwd
import sys
import tempfile
from pathlib import Path

import pytest

_EXPORTS = {
    **dict.fromkeys(
        (
            "Chunker",
            "improved_chunking_process",
            "chunk_for_embedding",
            "process_document_with_metadata",
            "DEFAULT_CHUNK_OPTIONS",
            "ENGINE_VERSION",
            "ChunkingError",
            "InvalidChunkingMethodError",
            "InvalidInputError",
            "LanguageDetectionError",
        ),
        "Chunk_Lib",
    ),
    **dict.fromkeys(
        (
            "LanguageChunkerFactory",
            "ChineseChunker",
            "JapaneseChunker",
            "DefaultChunker",
        ),
        "language_chunkers",
    ),
    **dict.fromkeys(("TokenBasedChunker", "create_token_chunker"), "token_chunker"),
}


def test_package_exports_are_the_installed_defining_objects():
    from tldw_chatbook import Chunking

    assert set(Chunking.__all__) == set(_EXPORTS) and len(Chunking.__all__) == 16
    assert set(_EXPORTS) <= set(dir(Chunking))
    for name, module in _EXPORTS.items():
        assert getattr(Chunking, name) is getattr(
            importlib.import_module("tldw_chatbook.Chunking." + module), name
        )


def test_unknown_names_and_native_submodule_loading():
    from tldw_chatbook import Chunking
    from tldw_chatbook.Chunking import _template_conversion, template_runtime

    assert _template_conversion is importlib.import_module(
        "tldw_chatbook.Chunking._template_conversion"
    )
    assert template_runtime is importlib.import_module(
        "tldw_chatbook.Chunking.template_runtime"
    )
    for name in (
        "not_an_export",
        "ChunkingTemplate",
        "ChunkingTemplateManager",
        "get_template_manager",
    ):
        with pytest.raises(AttributeError):
            getattr(Chunking, name)


def test_runtime_default_stays_frozen_after_real_module_import(monkeypatch, tmp_path):
    from Tests.Backup_Recovery.config_test_support import select_config_source
    from tldw_chatbook.Chunking import Chunk_Lib, token_chunker
    from tldw_chatbook.Chunking._shims import config as chunking_config

    selector = tmp_path / "config.toml"
    selector.write_text('[general]\nusers_name="shared"\n')
    selector.chmod(0o600)
    select_config_source(monkeypatch, selector, vars(chunking_config))

    baked = Chunk_Lib.DEFAULT_CHUNK_OPTIONS["summarize_system_prompt"]
    monkeypatch.setattr(token_chunker, "get_safe_import", lambda name: None)
    monkeypatch.setattr(
        Chunk_Lib, "get_internal_prompt", lambda *args, **kwargs: "LATER RESOLVER VALUE"
    )
    captured = []

    def dispatch(payload):
        captured.append(payload)
        return "summary"

    Chunk_Lib.Chunker().chunk_text(
        "Sentence one. Sentence two. Sentence three. " * 20,
        method="rolling_summarize",
        llm_call_function=dispatch,
        llm_api_config={},
    )
    assert captured
    joined = " ".join(str(row.get("system_message", "")) for row in captured)
    assert baked in joined and "LATER RESOLVER VALUE" not in joined


def cold_native_owner(result_path):
    """Standalone fresh-process proof; no pytest conftest or home override."""
    from Tests import network_guard, real_profile_guard

    entry = {key: os.environ.get(key) for key in ("HOME", "USERPROFILE")}
    uid_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    assert entry == {"HOME": str(uid_home), "USERPROFILE": None}
    real_profile_guard.install()
    network_guard.install()
    assert Path(os.environ["TLDW_TEST_REAL_HOME"]) == uid_home
    assert (
        str(uid_home / ".config/tldw_cli").lower()
        in real_profile_guard.protected_roots()
    )
    assert "tldw_chatbook.config" not in sys.modules
    root = Path(
        tempfile.mkdtemp(
            prefix="native-media-", dir=Path(os.environ["TMPDIR"]).resolve()
        )
    ).resolve()
    selector = root / "config.toml"
    selector.write_text('[general]\nusers_name="shared"\n')
    selector.chmod(0o600)
    os.environ["TLDW_CONFIG_PATH"] = str(selector)
    from tldw_chatbook.Backup_Recovery import bootstrap
    from tldw_chatbook.Backup_Recovery import storage_admission as storage

    bootstrap.default_bootstrap_root = lambda: root / "bootstrap"
    report = {
        "scope": "standalone cold pure converter and native Media owner",
        "root": str(root),
        "entry_home": entry,
    }
    owner = None

    def refuse_config(event, args):
        if event == "import" and args[0] == "tldw_chatbook.config":
            raise RuntimeError("cold_owner_imported_configuration")

    sys.addaudithook(refuse_config)
    try:
        from tldw_chatbook.Chunking._template_conversion import convert_template_row
        from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase

        converted = convert_template_row(
            {"name": "Words", "template_json": {"base_method": "words"}}
        )
        assert json.loads(converted["template_json"])["chunking"]["method"] == "words"
        assert not converted["deleted"]
        assert "tldw_chatbook.config" not in sys.modules and not storage._startups
        owner = MediaDatabase(root / "media.db", "cold-native-media")
        assert owner.db_path == root / "media.db"
        with owner.transaction() as connection:
            assert (
                connection.execute(
                    "SELECT COUNT(*) FROM ChunkingTemplates WHERE is_builtin = 1 AND deleted = 0"
                ).fetchone()[0]
                == 6
            )
        assert "tldw_chatbook.config" not in sys.modules and not storage._startups
        owner.close()
        owner = None
        assert (
            not storage._live_leases
            and not storage._operations
            and not storage._raw_operations
        )
        assert (
            not storage._holds and not storage._retiring_holds and not storage._startups
        )
        report["completed"] = True
    finally:
        if owner is not None:
            owner.close()
        report.update(
            profile_guard=real_profile_guard.take_violations(),
            network_guard=network_guard.blocked_attempts(),
            exit_home={key: os.environ.get(key) for key in entry},
            config_loaded="tldw_chatbook.config" in sys.modules,
            startups=len(storage._startups),
        )
        result_path.write_text(json.dumps(report, indent=2))
    assert report["entry_home"] == report["exit_home"]
    assert not report["profile_guard"] and not report["network_guard"]


if __name__ == "__main__":
    result = Path(sys.argv[1])
    assert result.is_absolute() and result.parent.is_dir() and not result.exists()
    cold_native_owner(result)

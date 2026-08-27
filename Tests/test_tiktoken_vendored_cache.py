"""The shipped tiktoken tables form a closed, immutable offline cache."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "tldw_chatbook" / "assets" / "tiktoken_cache"
BASE = "https://openaipublic.blob.core.windows.net"

TABLES = {
    "gpt2 vocab": {
        "encoding": "gpt2",
        "url": f"{BASE}/gpt-2/encodings/main/vocab.bpe",
        "cache_key": "6d1cbeee0f20b3d9449abfede4726ed8212e3aee",
        "sha256": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    },
    "gpt2 encoder": {
        "encoding": "gpt2",
        "url": f"{BASE}/gpt-2/encodings/main/encoder.json",
        "cache_key": "6c7ea1a7e38e3a7f062df639a5b80947f075ffe6",
        "sha256": "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
    },
    "r50k_base": {
        "encoding": "r50k_base",
        "url": f"{BASE}/encodings/r50k_base.tiktoken",
        "cache_key": "0ea1e91bbb3a60f729a8dc8f777fd2fc07cd8df4",
        "sha256": "306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930",
    },
    "p50k_base": {
        "encoding": "p50k_base",
        "url": f"{BASE}/encodings/p50k_base.tiktoken",
        "cache_key": "ec7223a39ce59f226a68acc30dc1af2788490e15",
        "sha256": "94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069",
    },
    "cl100k_base": {
        "encoding": "cl100k_base",
        "url": f"{BASE}/encodings/cl100k_base.tiktoken",
        "cache_key": "9b5ad71b2ce5302211f9c61530b329a4922fc6a4",
        "sha256": "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
    },
    "o200k_base": {
        "encoding": "o200k_base",
        "url": f"{BASE}/encodings/o200k_base.tiktoken",
        "cache_key": "fb374d419588a4632f3f557e76b4b70aebbca790",
        "sha256": "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
    },
}
EXPECTED_ENTRIES = {
    *(entry["cache_key"] for entry in TABLES.values()),
    "manifest.json",
    "LICENSE.txt",
    "NOTICE.txt",
}


def _runtime_module() -> Any:
    return importlib.import_module("tldw_chatbook.Utils.tiktoken_runtime")


def _run_source_child(script: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    child_env = env.copy()
    child_env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=child_env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_runtime_cache_has_exact_reviewed_inventory_and_hashes() -> None:
    assert {path.name for path in CACHE_DIR.iterdir()} == EXPECTED_ENTRIES
    for label, entry in TABLES.items():
        assert hashlib.sha1(entry["url"].encode()).hexdigest() == entry["cache_key"]  # nosec B324
        data = (CACHE_DIR / entry["cache_key"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"], label


def test_manifest_records_reviewed_runtime_and_redistribution_contract() -> None:
    manifest = json.loads((CACHE_DIR / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["tiktoken_version"] == "0.14.0"
    assert manifest["constructor_module"] == "tiktoken_ext.openai_public"
    assert manifest["constructor_path"] == "tiktoken_ext/openai_public.py"
    assert manifest["read_file_cached_signature"] == (
        "read_file_cached(blobpath: str, expected_hash: str | None = None) -> bytes"
    )
    assert manifest["cache_key_algorithm"] == "sha1(source_url UTF-8 bytes)"
    assert set(manifest["model_to_encoding_coverage"].values()) == {
        "gpt2",
        "cl100k_base",
        "p50k_base",
        "r50k_base",
        "o200k_base",
    }
    for model in (
        "gpt-5.6-terra",
        "gpt-5.6-sol",
        "gpt-5.6-luna",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
    ):
        assert manifest["model_to_encoding_coverage"][model] == "o200k_base"
    assert manifest["license"]["spdx"] == "MIT"
    assert manifest["license"]["source"] == (
        "tiktoken-0.14.0.dist-info/licenses/LICENSE"
    )
    assert manifest["license"]["clarification"] == (
        "https://github.com/openai/tiktoken/issues/92#issuecomment-1497875652"
    )
    assert manifest["update_procedure"]
    assert {entry["url"]: entry for entry in manifest["files"]} == {
        entry["url"]: entry for entry in TABLES.values()
    }

    notice = (CACHE_DIR / "NOTICE.txt").read_text(encoding="utf-8")
    for required in (
        "tiktoken 0.14.0",
        "tiktoken_ext/openai_public.py",
        "read_file_cached(blobpath: str, expected_hash: str | None = None)",
        "sha1(source_url UTF-8 bytes)",
        "https://github.com/openai/tiktoken/issues/92#issuecomment-1497875652",
        "Update procedure",
    ):
        assert required in notice
    for entry in TABLES.values():
        assert entry["encoding"] in notice
        assert entry["url"] in notice
        assert entry["cache_key"] in notice
        assert entry["sha256"] in notice


def test_package_import_installs_guard_and_loads_every_supported_encoding() -> None:
    env = os.environ.copy()
    env.pop("TIKTOKEN_CACHE_DIR", None)
    env.pop("DATA_GYM_CACHE_DIR", None)
    result = _run_source_child(
        """
from pathlib import Path
import os

assert "TIKTOKEN_CACHE_DIR" not in os.environ
assert "DATA_GYM_CACHE_DIR" not in os.environ
import tldw_chatbook
import tiktoken
import tiktoken.load
import tiktoken.registry
from tldw_chatbook.Utils import tiktoken_runtime

def fail_fetch(*_args, **_kwargs):
    raise AssertionError("tiktoken attempted an upstream fetch")

tiktoken.load.read_file = fail_fetch
assert Path(os.environ["TIKTOKEN_CACHE_DIR"]).resolve() == Path(
    os.environ["EXPECTED_CACHE_DIR"]
).resolve()
assert tiktoken.load.read_file_cached is tiktoken_runtime._read_bundled_file
tiktoken_runtime.install_tiktoken_runtime()
assert tiktoken.load.read_file_cached is tiktoken_runtime._read_bundled_file

tiktoken.registry.ENCODINGS.clear()
for encoding_name in (
    "gpt2",
    "r50k_base",
    "p50k_base",
    "cl100k_base",
    "o200k_base",
):
    assert tiktoken.get_encoding(encoding_name).encode("hello world")
""",
        {**env, "EXPECTED_CACHE_DIR": str(CACHE_DIR)},
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("override_key", "override_value"),
    [
        ("TIKTOKEN_CACHE_DIR", "/tmp/task-2526 explicit cache !"),
        ("DATA_GYM_CACHE_DIR", "/tmp/task-2526 legacy cache !"),
    ],
)
def test_preimport_cache_override_is_unchanged_and_keeps_upstream_reader(
    override_key: str,
    override_value: str,
) -> None:
    env = os.environ.copy()
    env.pop("TIKTOKEN_CACHE_DIR", None)
    env.pop("DATA_GYM_CACHE_DIR", None)
    env[override_key] = override_value
    result = _run_source_child(
        """
import os
import sys

key = os.environ["OVERRIDE_KEY"]
value = os.environ["OVERRIDE_VALUE"]
import tldw_chatbook

assert os.environ[key] == value
assert not any(name == "tiktoken" or name.startswith("tiktoken.") for name in sys.modules)

import tiktoken.load

assert tiktoken.load.read_file_cached.__module__ == "tiktoken.load"
""",
        {**env, "OVERRIDE_KEY": override_key, "OVERRIDE_VALUE": override_value},
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_package_import_without_tiktoken_preserves_character_fallback() -> None:
    env = os.environ.copy()
    env.pop("TIKTOKEN_CACHE_DIR", None)
    env.pop("DATA_GYM_CACHE_DIR", None)
    result = _run_source_child(
        """
import importlib.abc
import sys

class BlockTiktoken(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "tiktoken" or fullname.startswith("tiktoken."):
            raise ModuleNotFoundError("blocked tiktoken", name=fullname)
        return None

sys.meta_path.insert(0, BlockTiktoken())
import tldw_chatbook
from tldw_chatbook.Utils import token_counter

assert token_counter.TIKTOKEN_AVAILABLE is False
assert token_counter.estimate_tokens("abcdefghij", provider="openai") == 3
""",
        env,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture
def isolated_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[Any, dict[str, str]]:
    runtime = _runtime_module()
    entry = dict(TABLES["cl100k_base"])
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"files": [entry]}), encoding="utf-8")
    monkeypatch.setattr(runtime, "_ASSET_DIR", tmp_path)
    monkeypatch.setattr(runtime, "_MANIFEST_PATH", manifest_path)
    runtime._manifest_by_url.cache_clear()

    import tiktoken.load

    def fail_fetch(*_args: object, **_kwargs: object) -> bytes:
        raise AssertionError("bundled reader delegated to tiktoken.read_file")

    monkeypatch.setattr(tiktoken.load, "read_file", fail_fetch)
    yield runtime, entry
    runtime._manifest_by_url.cache_clear()


def test_bundled_reader_normalizes_missing_asset(isolated_bundle: tuple[Any, dict[str, str]]) -> None:
    runtime, entry = isolated_bundle
    with pytest.raises(runtime.BundledTiktokenAssetError, match="missing|read"):
        runtime._read_bundled_file(entry["url"], entry["sha256"])


def test_bundled_reader_rejects_corrupt_asset(
    isolated_bundle: tuple[Any, dict[str, str]],
) -> None:
    runtime, entry = isolated_bundle
    (runtime._ASSET_DIR / entry["cache_key"]).write_bytes(b"corrupt")
    with pytest.raises(runtime.BundledTiktokenAssetError, match="hash"):
        runtime._read_bundled_file(entry["url"], entry["sha256"])


def test_bundled_reader_rejects_expected_hash_mismatch(
    isolated_bundle: tuple[Any, dict[str, str]],
) -> None:
    runtime, entry = isolated_bundle
    with pytest.raises(runtime.BundledTiktokenAssetError, match="expected hash"):
        runtime._read_bundled_file(entry["url"], "0" * 64)


def test_bundled_reader_rejects_manifest_cache_key_mismatch(
    isolated_bundle: tuple[Any, dict[str, str]],
) -> None:
    runtime, entry = isolated_bundle
    manifest = json.loads(runtime._MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["files"][0]["cache_key"] = "0" * 40
    runtime._MANIFEST_PATH.write_text(json.dumps(manifest), encoding="utf-8")
    runtime._manifest_by_url.cache_clear()
    with pytest.raises(runtime.BundledTiktokenAssetError, match="cache key"):
        runtime._read_bundled_file(entry["url"], entry["sha256"])


def test_bundled_reader_rejects_unmanifested_url(
    isolated_bundle: tuple[Any, dict[str, str]],
) -> None:
    runtime, _entry = isolated_bundle
    with pytest.raises(runtime.BundledTiktokenAssetError, match="not in the manifest"):
        runtime._read_bundled_file(f"{BASE}/encodings/unreviewed.tiktoken", "0" * 64)


def test_bundled_reader_normalizes_manifest_lookup_failure(
    isolated_bundle: tuple[Any, dict[str, str]],
) -> None:
    runtime, entry = isolated_bundle
    runtime._MANIFEST_PATH.write_text("not json", encoding="utf-8")
    runtime._manifest_by_url.cache_clear()
    with pytest.raises(runtime.BundledTiktokenAssetError, match="manifest"):
        runtime._read_bundled_file(entry["url"], entry["sha256"])


def test_install_rejects_an_unreviewed_upstream_reader_signature() -> None:
    env = os.environ.copy()
    env.pop("DATA_GYM_CACHE_DIR", None)
    env["TIKTOKEN_CACHE_DIR"] = "bypass-during-package-import"
    result = _run_source_child(
        """
import os
import tiktoken.load
import tldw_chatbook
from tldw_chatbook.Utils.tiktoken_runtime import install_tiktoken_runtime

del os.environ["TIKTOKEN_CACHE_DIR"]
unreviewed_readers = (
    lambda blobpath: b"missing expected_hash",
    lambda blobpath, expected_hash: b"expected_hash is unexpectedly required",
)
for reader in unreviewed_readers:
    tiktoken.load.read_file_cached = reader
    try:
        install_tiktoken_runtime()
    except RuntimeError as error:
        assert "read_file_cached" in str(error)
    else:
        raise AssertionError("unreviewed tiktoken seam was accepted")
    assert "TIKTOKEN_CACHE_DIR" not in os.environ
""",
        env,
    )
    assert result.returncode == 0, result.stdout + result.stderr

"""Public native QA exports retain evidence without publishing host paths."""

import hashlib
import json
import runpy
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXPORTER = REPO / "Docs/superpowers/qa/export_receipts.py"


def test_receipt_paths_are_normalized_without_changing_hashes_or_urls(tmp_path):
    normalize = runpy.run_path(str(EXPORTER))["normalize_receipt"]
    home = tmp_path / "reviewer"
    main = home / "projects/chatbook"
    repo = main / ".worktrees/current"
    original = json.dumps(
        {
            "repo": str(repo),
            "module": str(repo / "tldw_chatbook/app.py"),
            "other_checkout": str(main / ".worktrees/previous/.venv/bin/python"),
            "config": str(home / ".config/tldw_cli/config.toml"),
            "profile": "/private/tmp/native-review/evidence/result.json",
            "temporary": "/private/var/folders/ab/host-specific/T/test/config.toml",
            "pytest": "/tmp/pytest-of-reviewer/pytest-1/profile/config.toml",
            "hash": "a" * 64,
            "url": "https://github.com/example/chatbook/pull/1",
        }
    )
    result = json.loads(normalize(original, repo=repo, home=home))
    assert result == {
        "repo": "<repo>",
        "module": "<repo>/tldw_chatbook/app.py",
        "other_checkout": "<repo>/.venv/bin/python",
        "config": "<home>/.config/tldw_cli/config.toml",
        "profile": "<tmp>/native-review/evidence/result.json",
        "temporary": "<tmp>/test/config.toml",
        "pytest": "<tmp>/pytest-of-user/pytest-1/profile/config.toml",
        "hash": "a" * 64,
        "url": "https://github.com/example/chatbook/pull/1",
    }
    assert normalize(
        normalize(original, repo=repo, home=home), repo=repo, home=home
    ) == normalize(original, repo=repo, home=home)


def test_export_preserves_raw_receipt_and_records_published_hash(tmp_path):
    raw = tmp_path / "raw.json"
    original = json.dumps(
        {"module": str(REPO / "tldw_chatbook/app.py"), "passed": True}
    )
    raw.write_text(original)
    destination = tmp_path / "published"
    subprocess.run(
        [sys.executable, str(EXPORTER), "--destination", str(destination), str(raw)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert raw.read_text() == original
    exported = destination / raw.name
    assert json.loads(exported.read_text()) == {
        "module": "<repo>/tldw_chatbook/app.py",
        "passed": True,
    }
    manifest = json.loads((destination / "publication-manifest.json").read_text())
    assert manifest[0]["source_sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
    assert (
        manifest[0]["export_sha256"]
        == hashlib.sha256(exported.read_bytes()).hexdigest()
    )
    assert manifest[0]["export"] == raw.name


def test_export_refuses_overwriting_raw_evidence(tmp_path):
    raw = tmp_path / "raw.json"
    raw.write_text('{"passed": true}')
    result = subprocess.run(
        [sys.executable, str(EXPORTER), "--destination", str(tmp_path), str(raw)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert raw.read_text() == '{"passed": true}'
    assert not (tmp_path / "publication-manifest.json").exists()


def test_export_validates_all_inputs_before_writing(tmp_path):
    good = tmp_path / "good.txt"
    bad = tmp_path / "bad.bin"
    good.write_text("valid receipt")
    bad.write_bytes(b"\xff\xfe")
    destination = tmp_path / "published"
    result = subprocess.run(
        [
            sys.executable,
            str(EXPORTER),
            "--destination",
            str(destination),
            str(good),
            str(bad),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert not destination.exists()


def test_export_rejects_duplicate_basenames(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    for directory in (first, second):
        (directory / "receipt.json").write_text("{}")
    destination = tmp_path / "published"
    result = subprocess.run(
        [
            sys.executable,
            str(EXPORTER),
            "--destination",
            str(destination),
            str(first / "receipt.json"),
            str(second / "receipt.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "receipt names must be unique" in result.stderr
    assert not destination.exists()


def test_export_refuses_existing_destination_aliases(tmp_path):
    raw = tmp_path / "receipt.json"
    raw.write_text('{"original": true}')
    other = tmp_path / "other.json"
    other.write_text('{"other": true}')
    destination = tmp_path / "published"
    destination.mkdir()
    (destination / raw.name).symlink_to(other)
    result = subprocess.run(
        [sys.executable, str(EXPORTER), "--destination", str(destination), str(raw)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert other.read_text() == '{"other": true}'

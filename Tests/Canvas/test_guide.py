"""Bounded packaged guides and examples under shipped runtime admission."""

import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from markdown_it import MarkdownIt

from tldw_chatbook.Canvas import guide
from tldw_chatbook.Canvas.authoring import canvas_authoring_guide
from tldw_chatbook.Canvas.compiler import compile_canvas_document
from tldw_chatbook.Canvas.profiles import load_profile_snapshot


TOPICS = {
    "basics": ("guides/basics.md", "canvas-v1", 1),
    "controls": ("guides/controls.md", "canvas-v1", 1),
    "mermaid": ("static/mermaid-authoring.txt", "canvas-v2-mermaid-1", 2),
    "repair": ("guides/repair.md", None, 0),
}


def test_closed_packaged_topic_inventory():
    assert guide.CANVAS_GUIDE_PATHS == {
        topic: path for topic, (path, _, _) in TOPICS.items()
    }
    assert guide.MAX_CANVAS_GUIDE_RESULT_BYTES == 12 * 1024
    project = Path(__file__).resolve().parents[2] / "pyproject.toml"
    package_data = tomllib.loads(project.read_text(encoding="utf-8"))["tool"][
        "setuptools"
    ]["package-data"]["tldw_chatbook.Canvas"]
    for path, _, _ in TOPICS.values():
        assert path in package_data


@pytest.mark.parametrize("topic", TOPICS)
def test_complete_topic_examples_compile_with_shipped_admission(topic):
    body = guide.read_canvas_guide(topic)
    result = json.dumps({"status": "ok", "topic": topic, "guide": body})
    assert len(result.encode("utf-8")) < guide.MAX_CANVAS_GUIDE_RESULT_BYTES
    examples = [
        token.content
        for token in MarkdownIt().parse(body)
        if token.type == "fence" and token.info == "html"
    ]
    _, profile, count = TOPICS[topic]
    assert len(examples) == count
    if not examples:
        return
    snapshot = load_profile_snapshot()
    assert any(
        record.profile_id == profile and record.executable
        for record in snapshot.profiles
    )
    assert profile in body
    if topic == "mermaid":
        assert body == canvas_authoring_guide(snapshot, profile)
    for example in examples:
        assert example.lower().startswith("<!doctype html>")
        assert "</html>" in example
        plan = compile_canvas_document(
            example, runtime_profile=profile, snapshot=snapshot
        )
        assert plan.runtime_profile == profile
        assert not plan.compatibility_issues
        assert bool(plan.scripts) == (topic == "controls")
        if topic == "mermaid":
            assert len(plan.diagrams) == 1


class StringSubclass(str):
    pass


@pytest.mark.parametrize(
    "topic",
    [
        None,
        1,
        True,
        [],
        {},
        b"basics",
        StringSubclass("basics"),
        "",
        "unknown",
        "BASICS",
        " basics",
        "../basics",
        "guides/basics.md",
        "/tmp/basics.md",
        "guides\\basics.md",
    ],
)
def test_invalid_topics_refuse_before_resource_lookup(topic, monkeypatch):
    def unexpected_lookup(package):
        pytest.fail("invalid topic reached package resources")

    monkeypatch.setattr(guide, "files", unexpected_lookup)
    with pytest.raises(ValueError, match="^invalid guide topic$"):
        guide.read_canvas_guide(topic)


@pytest.mark.parametrize("topic", TOPICS)
@pytest.mark.parametrize(
    ("payload", "error", "message"),
    [
        (None, FileNotFoundError, None),
        (b"", ValueError, "^guide is empty$"),
        (b" \n\t", ValueError, "^guide is empty$"),
        (b"valid prefix\xff", UnicodeDecodeError, None),
        (b"a" * (12 * 1024 + 1), ValueError, "^guide is oversized$"),
        ("é".encode("utf-8") * (6 * 1024 + 1), ValueError, "^guide is oversized$"),
    ],
)
def test_invalid_resources_refuse_without_partial_output(
    topic, payload, error, message, tmp_path, monkeypatch
):
    resource = tmp_path / TOPICS[topic][0]
    resource.parent.mkdir(parents=True)
    if payload is not None:
        resource.write_bytes(payload)
    monkeypatch.setattr(guide, "files", lambda package: tmp_path)
    with pytest.raises(error, match=message):
        guide.read_canvas_guide(topic)


def test_reader_preserves_exact_utf8_at_raw_byte_ceiling(tmp_path, monkeypatch):
    resource = tmp_path / "guides/basics.md"
    resource.parent.mkdir()
    body = "é" * (6 * 1024 - 1) + "\r\n"
    resource.write_bytes(body.encode("utf-8"))
    monkeypatch.setattr(guide, "files", lambda package: tmp_path)
    assert guide.read_canvas_guide("basics") == body


def test_module_import_does_not_read_guide_resources():
    probe = """
import sys

def reject_guide_read(event, args):
    if event == "open":
        path = str(args[0]).replace("\\\\", "/")
        if "/guides/" in path or path.endswith("mermaid-authoring.txt"):
            raise AssertionError("guide resource read on import: " + path)

sys.addaudithook(reject_guide_read)
import tldw_chatbook.Canvas.guide
"""
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr

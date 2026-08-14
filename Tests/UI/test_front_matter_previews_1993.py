"""TASK-1993: markdown previews consume YAML front matter instead of rendering it."""

import tldw_chatbook.Utils.markdown_parsing as markdown_parsing
from tldw_chatbook.Utils.markdown_parsing import front_matter_parser_factory

FRONT_MATTERED = "---\ntags:\n  - unsloth\nlicense: apache-2.0\n---\n# Title\n\nBody.\n"


def test_factory_parser_consumes_front_matter():
    factory = front_matter_parser_factory()
    assert factory is not None, "mdit-py-plugins installed in the dev venv"
    parser = factory()
    tokens = parser.parse(FRONT_MATTERED)
    assert tokens[0].type == "front_matter"
    # The document proper still parses normally.
    assert any(t.type == "heading_open" for t in tokens)
    # And gfm-like table support survives the plugin chain.
    table_tokens = parser.parse("| a | b |\n|---|---|\n| 1 | 2 |\n")
    assert any(t.type == "table_open" for t in table_tokens)


def test_factory_degrades_to_none_without_dependency(monkeypatch):
    monkeypatch.setattr(markdown_parsing, "check_dependency", lambda *a, **k: False)
    assert front_matter_parser_factory() is None

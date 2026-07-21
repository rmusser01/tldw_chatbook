import pytest

from tldw_chatbook.Internal_Prompts.catalog import CATALOG, PromptSpec, register
from tldw_chatbook.Internal_Prompts.resolver import (
    get_internal_prompt,
    render_internal_prompt,
    safe_substitute,
)


@pytest.fixture
def demo_spec():
    spec = register(
        PromptSpec(
            id="demo.greeting",
            subsystem="demo",
            title="Greeting",
            description="Test prompt.",
            used_in="tests",
            default="Hello {name}, JSON stays: {\"k\": 1}",
            required_placeholders=("name",),
            legacy_config_path="Prompts.demo_legacy",
        )
    )
    yield spec
    CATALOG.pop("demo.greeting", None)


# --- safe_substitute -------------------------------------------------------

def test_substitute_replaces_declared_tokens_only():
    out = safe_substitute("A {x} B {y} C {z}", x=1, y="two")
    assert out == "A 1 B two C {z}"


def test_substitute_leaves_json_and_ollama_braces_alone():
    text = 'Return {"score": 0.5} and {{ .Prompt }} end {q}'
    assert safe_substitute(text, q="Q") == 'Return {"score": 0.5} and {{ .Prompt }} end Q'


def test_substitute_never_raises_on_stray_braces():
    assert safe_substitute("{unclosed {weird}} {q}", q="ok") == "{unclosed {weird}} ok"


# --- precedence ------------------------------------------------------------

def test_default_when_no_config(demo_spec, scratch_config):
    assert get_internal_prompt("demo.greeting") == demo_spec.default


def test_override_table_wins(demo_spec, scratch_config):
    scratch_config(
        '[internal_prompts.demo.greeting]\ntext = "Hi {name}!"\nbaseline = "abc"\n'
    )
    assert get_internal_prompt("demo.greeting") == "Hi {name}!"


def test_override_plain_string_accepted(demo_spec, scratch_config):
    scratch_config('[internal_prompts.demo]\ngreeting = "Yo {name}"\n')
    assert get_internal_prompt("demo.greeting") == "Yo {name}"


def test_empty_override_means_no_override(demo_spec, scratch_config):
    scratch_config('[internal_prompts.demo]\ngreeting = ""\n')
    assert get_internal_prompt("demo.greeting") == demo_spec.default


def test_invalid_override_falls_back_to_default(demo_spec, scratch_config):
    scratch_config('[internal_prompts.demo]\ngreeting = "no placeholder here"\n')
    assert get_internal_prompt("demo.greeting") == demo_spec.default


def test_override_beats_legacy(demo_spec, scratch_config):
    scratch_config(
        '[internal_prompts.demo]\ngreeting = "Override {name}"\n'
        '[Prompts]\ndemo_legacy = "Legacy {name}"\n'
    )
    assert get_internal_prompt("demo.greeting") == "Override {name}"


# --- legacy tier -----------------------------------------------------------

def test_customized_legacy_honored(demo_spec, scratch_config):
    scratch_config('[Prompts]\ndemo_legacy = "Legacy {name}"\n')
    assert get_internal_prompt("demo.greeting") == "Legacy {name}"


def test_legacy_equal_to_shipped_stub_ignored(demo_spec, scratch_config, monkeypatch):
    from tldw_chatbook import config as config_mod

    monkeypatch.setitem(
        config_mod.DEFAULT_CONFIG_FROM_TOML.setdefault("Prompts", {}),
        "demo_legacy",
        "the shipped stub {name}",
    )
    scratch_config('[Prompts]\ndemo_legacy = "the shipped stub {name}"\n')
    assert get_internal_prompt("demo.greeting") == demo_spec.default


def test_invalid_legacy_falls_back(demo_spec, scratch_config):
    scratch_config('[Prompts]\ndemo_legacy = "customized but tokenless"\n')
    assert get_internal_prompt("demo.greeting") == demo_spec.default


# --- misc ------------------------------------------------------------------

def test_unknown_id_raises_keyerror(scratch_config):
    with pytest.raises(KeyError):
        get_internal_prompt("nope.nothing")


def test_render(demo_spec, scratch_config):
    out = render_internal_prompt("demo.greeting", name="Ada")
    assert out == 'Hello Ada, JSON stays: {"k": 1}'


def test_warn_once(demo_spec, scratch_config, caplog):
    scratch_config('[internal_prompts.demo]\ngreeting = "tokenless"\n')
    get_internal_prompt("demo.greeting")
    get_internal_prompt("demo.greeting")
    warnings = [r for r in caplog.records if "demo.greeting" in r.getMessage()]
    assert len(warnings) == 1

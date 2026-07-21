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


def test_substitute_value_containing_token_not_reexpanded():
    # Single-pass: a value that looks like another token must survive as-is,
    # regardless of kwarg order.
    assert safe_substitute("{a} {b}", b="{a}", a="X") == "X {a}"
    assert safe_substitute("{a} {b}", a="X", b="{a}") == "X {a}"


def test_substitute_handles_overlapping_key_names():
    # A full token includes its closing brace, so a name that is a prefix of
    # another name cannot corrupt it, regardless of kwarg order.
    assert safe_substitute("{a} {ab}", a="X", ab="Y") == "X Y"
    assert safe_substitute("{ab} {a}", ab="Y", a="X") == "Y X"
    assert (
        safe_substitute("{query} vs {query_list}", query="Q", query_list="QL")
        == "Q vs QL"
    )


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

    # Direct indexing on purpose: [Prompts] always exists in shipped defaults;
    # if that ever changes this fails loud (KeyError) instead of setdefault
    # silently leaking an empty section past monkeypatch's undo.
    monkeypatch.setitem(
        config_mod.DEFAULT_CONFIG_FROM_TOML["Prompts"],
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


def test_render_missing_required_value_warns_once_never_raises(
    demo_spec, scratch_config, caplog
):
    # Caller forgot name= entirely: token survives in output (never raises),
    # and a once-per-prompt warning fires so the bug is visible in logs.
    out = render_internal_prompt("demo.greeting")
    assert "{name}" in out
    render_internal_prompt("demo.greeting")
    warnings = [
        r
        for r in caplog.records
        if "without required placeholder value" in r.getMessage()
        and "demo.greeting" in r.getMessage()
    ]
    assert len(warnings) == 1

import pytest

from tldw_chatbook.Internal_Prompts.catalog import CATALOG, PromptSpec, register


def _spec(**overrides):
    base = dict(
        id="demo.example",
        subsystem="demo",
        title="Example",
        description="An example prompt.",
        used_in="tests",
        default="Hello {name}",
        required_placeholders=("name",),
    )
    base.update(overrides)
    return PromptSpec(**base)


def test_register_adds_to_catalog():
    spec = _spec()
    try:
        assert register(spec) is spec
        assert CATALOG["demo.example"] is spec
    finally:
        CATALOG.pop("demo.example", None)


def test_register_rejects_duplicate_id():
    spec = _spec()
    try:
        register(spec)
        with pytest.raises(ValueError, match="Duplicate"):
            register(_spec())
    finally:
        CATALOG.pop("demo.example", None)


def test_register_rejects_id_subsystem_mismatch():
    with pytest.raises(ValueError, match="subsystem"):
        register(_spec(id="other.example"))


def test_spec_is_frozen():
    spec = _spec()
    with pytest.raises(Exception):
        spec.default = "changed"


def test_spec_defaults():
    spec = _spec()
    assert spec.optional_placeholders == ()
    assert spec.contract_note is None
    assert spec.legacy_config_path is None
    assert spec.applies == "live"

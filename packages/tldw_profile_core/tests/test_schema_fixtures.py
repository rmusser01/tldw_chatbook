import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from tldw_profile_core import (
    ProfileManifest,
    ProfileProposal,
    ProfileRecord,
    ProfileScope,
    export_json_schema,
)

ROOT = Path(__file__).parents[1]
SOURCE_ROOT = ROOT / "src" / "tldw_profile_core"
MODELS = {
    "ProfileManifest": ProfileManifest,
    "ProfileScope": ProfileScope,
    "ProfileRecord": ProfileRecord,
    "ProfileProposal": ProfileProposal,
}


def fixtures():
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((ROOT / "fixtures" / "v1").glob("*.json"))
    ]


def test_combined_schema_has_one_root_defs_namespace_and_all_refs_resolve():
    schema = json.loads(
        (ROOT / "schemas" / "personal-context-v1.json").read_text(encoding="utf-8")
    )
    assert "$defs" in schema
    assert not any(
        "$defs" in branch for branch in schema.get("anyOf", schema.get("oneOf", []))
    )
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    for fixture in fixtures():
        errors = list(validator.iter_errors(fixture["data"]))
        assert (not errors) is fixture["valid"], (fixture, errors)


def test_schema_conditionals_require_their_selector_fields():
    schema = json.loads(
        (ROOT / "schemas" / "personal-context-v1.json").read_text(encoding="utf-8")
    )
    for model_name in ("ProfileRecord", "ProfileProposal"):
        for conditional in schema["$defs"][model_name]["allOf"]:
            selector = conditional["if"]
            assert set(selector["properties"]) <= set(selector["required"])


def test_fixtures_dispatch_to_models_with_matching_results():
    required = {
        "manifest",
        "scope",
        "active_record",
        "deleted_tombstone",
        "proposal",
        "unknown_field_version",
        "working_context_expiry",
        "missing_proposal_base",
        "deleted_with_content",
        "working_context_missing_expiry",
        "kind_payload_mismatch",
        "update_missing_content",
        "create_missing_content",
    }
    loaded = fixtures()
    assert {fixture["case"] for fixture in loaded} == required
    for fixture in loaded:
        model = MODELS[fixture["model"]]
        if fixture["valid"]:
            model.model_validate(fixture["data"])
        else:
            with pytest.raises(ValidationError):
                model.model_validate(fixture["data"])


def test_schema_and_fixture_resource_copies_byte_match_and_regenerate_deterministically(
    tmp_path,
):
    package_schema = ROOT / "schemas" / "personal-context-v1.json"
    source_schema = SOURCE_ROOT / "schemas" / "personal-context-v1.json"
    assert package_schema.read_bytes() == source_schema.read_bytes()
    generated = tmp_path / "personal-context-v1.json"
    export_json_schema(generated)
    assert generated.read_bytes() == package_schema.read_bytes()
    for fixture in (ROOT / "fixtures" / "v1").glob("*.json"):
        assert (
            fixture.read_bytes()
            == (SOURCE_ROOT / "fixtures" / "v1" / fixture.name).read_bytes()
        )

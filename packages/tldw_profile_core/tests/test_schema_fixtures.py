import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from tldw_profile_core import (
    PROFILE_DIALECT_ID,
    PROFILE_SCHEMA_ID,
    PROFILE_SEMANTIC_KEYWORD,
    PROFILE_SEMANTIC_RULES,
    PROFILE_SEMANTIC_VOCABULARY_ID,
    ProfileManifest,
    ProfileProposal,
    ProfileRecord,
    ProfileScope,
    export_json_schema,
    export_profile_meta_schema,
    validate_profile_semantics,
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


def schema_resource(name="personal-context-v1.json"):
    return json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8"))


def combined_errors(validator, data):
    errors = list(validator.iter_errors(data))
    if errors:
        return errors
    try:
        validate_profile_semantics(data)
    except ValueError as error:
        return [error]
    return []


def test_combined_schema_has_one_root_defs_namespace_and_all_refs_resolve():
    schema = schema_resource()
    assert "$defs" in schema
    assert not any(
        "$defs" in branch for branch in schema.get("anyOf", schema.get("oneOf", []))
    )
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    for fixture in fixtures():
        errors = list(validator.iter_errors(fixture["data"]))
        assert (not errors) is fixture.get("structurally_valid", fixture["valid"]), (
            fixture,
            errors,
        )


def test_custom_dialect_requires_the_versioned_semantic_vocabulary():
    schema = schema_resource()
    meta_schema = schema_resource("personal-context-v1-meta.json")
    assert schema["$id"] == PROFILE_SCHEMA_ID
    assert schema["$schema"] == PROFILE_DIALECT_ID
    assert "structural" in schema["$comment"]
    assert schema[PROFILE_SEMANTIC_KEYWORD] == PROFILE_SEMANTIC_RULES
    assert meta_schema["$id"] == PROFILE_DIALECT_ID
    assert meta_schema["$vocabulary"][PROFILE_SEMANTIC_VOCABULARY_ID] is True
    Draft202012Validator.check_schema(meta_schema)
    Draft202012Validator(meta_schema).validate(schema)


def test_combined_validator_enforces_structural_and_semantic_fixture_results():
    validator = Draft202012Validator(schema_resource())
    for fixture in fixtures():
        errors = combined_errors(validator, fixture["data"])
        assert (not errors) is fixture["valid"], (fixture, errors)


def test_schema_conditionals_require_their_selector_fields():
    schema = schema_resource()
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
        "pending_89_day_expiry",
        "proposal_identity_mismatch",
        "proposal_base_mismatch",
        "multibyte_canonical_overflow",
        "pending_deleted_proposed_record",
        "jcs_cross_runtime_conformance",
        "invalid_manifest_timestamp",
        "naive_scope_timestamp",
        "manifest_timestamp_order",
        "scope_timestamp_order",
        "record_timestamp_order",
        "record_expiry_order",
        "resolved_proposal_timestamp_order",
        "submillisecond_timestamp",
        "unsafe_manifest_integer",
        "proposed_record_timestamp_order",
        "numeric_epoch_timestamp",
        "space_datetime_syntax",
        "lowercase_datetime_syntax",
        "four_digit_fraction_timestamp",
        "string_manifest_counter",
        "boolean_manifest_counter",
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
    for name, exporter in {
        "personal-context-v1.json": export_json_schema,
        "personal-context-v1-meta.json": export_profile_meta_schema,
    }.items():
        package_schema = ROOT / "schemas" / name
        source_schema = SOURCE_ROOT / "schemas" / name
        assert package_schema.read_bytes() == source_schema.read_bytes()
        generated = tmp_path / name
        exporter(generated)
        assert generated.read_bytes() == package_schema.read_bytes()
    for fixture in (ROOT / "fixtures" / "v1").glob("*.json"):
        assert (
            fixture.read_bytes()
            == (SOURCE_ROOT / "fixtures" / "v1" / fixture.name).read_bytes()
        )

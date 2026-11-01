import importlib
import importlib.util
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError
from tldw_profile_core.canonical import canonical_bytes

ROOT = Path(__file__).parents[1]
FIXTURE = Path("fixtures/evidence_binding/v1/01-unicode-message.json")
RESOURCE = "tldw_profile_core/" + FIXTURE.as_posix()
FIXED_SHA = "dae4dd8a13189b7bf86559f7adaab2d6992fe3040a1d255b2c212dc7b15801fb"
IDENTITIES = (
    "binding_id",
    "authority_id",
    "governance_scope_id",
    "source_container_id",
    "source_object_id",
    "source_version_id",
)


class CustomText(str):
    pass


class CustomInteger(int):
    pass


class CustomDatetime(datetime):
    pass


def api():
    assert importlib.util.find_spec("tldw_profile_core.evidence_binding") is not None, (
        "binding API is missing"
    )
    module = importlib.import_module("tldw_profile_core.evidence_binding")
    return (
        module.OwnerVersionEvidenceBinding,
        module.owner_version_evidence_binding_digest,
    )


def fixture():
    return json.loads((ROOT / FIXTURE).read_text(encoding="utf-8"))


def data():
    return fixture()["data"]


def test_fixed_canonical_identity_and_json_round_trip():
    model, digest = api()
    binding = model(**data())
    assert canonical_bytes(binding) == fixture()["canonical_utf8"].encode("utf-8")
    assert digest(binding) == FIXED_SHA == fixture()["binding_sha256"]
    assert digest(model.model_validate_json(binding.model_dump_json())) == FIXED_SHA
    assert binding.captured_at == datetime(2026, 9, 26, 16, tzinfo=UTC)


@pytest.mark.parametrize("field", tuple(fixture()["data"]))
def test_every_serialized_field_is_required(field):
    model, _ = api()
    fields = data()
    del fields[field]
    with pytest.raises(ValidationError):
        model.model_validate(fields)
    with pytest.raises(ValidationError):
        model.model_validate_json(json.dumps(fields))


@pytest.mark.parametrize("field", IDENTITIES)
@pytest.mark.parametrize(
    "value", ["", "   ", "a" * 129, "a\x00b", "a\u200db", "\ud800", 1, b"id"]
)
def test_identity_bounds_are_enforced_on_each_identity(field, value):
    model, _ = api()
    fields = data() | {field: value}
    with pytest.raises(ValidationError):
        model.model_validate(fields)
    if type(value) is not bytes:
        with pytest.raises(ValidationError):
            model.model_validate_json(json.dumps(fields))


@pytest.mark.parametrize("field", IDENTITIES)
def test_maximum_unicode_identity_and_preserved_spelling(field):
    model, digest = api()
    original = model(**data())
    assert digest(model(**(data() | {field: "👋" * 128}))) != digest(original)
    first = model(**(data() | {field: " café "}))
    assert getattr(first, field) == " café "
    assert digest(first) != digest(model(**(data() | {field: " cafe\u0301 "})))


BAD_VALUES = [
    ("component_version", True),
    ("component_version", 1.0),
    ("component_version", "1"),
    ("component_version", 2),
    ("authority_kind", "other"),
    ("source_kind", "note"),
    ("version_kind", "captured_representation"),
    ("representation_id", "rendered_text"),
    ("offset_unit", "utf16"),
    ("source_role", "assistant"),
    ("span_start", True),
    ("span_start", 7.0),
    ("span_start", "7"),
    ("span_start", -1),
    ("span_start", 12),
    ("span_end", 6),
    ("span_end", 2**53),
    ("span_end", None),
    ("representation_sha256", "A" * 64),
    ("representation_sha256", "0" * 63),
    ("span_sha256", "g" * 64),
    ("span_sha256", "0" * 64 + "\n"),
    ("captured_at", "2026-09-26T16:00:00"),
    ("captured_at", "2026-09-26 16:00:00Z"),
    ("captured_at", "2026-02-30T16:00:00Z"),
    ("captured_at", "2026-09-26T16:00:00.0001Z"),
    ("captured_at", 1),
    ("captured_at", None),
]


@pytest.mark.parametrize(
    "field,value", BAD_VALUES, ids=[f"bad-{i}" for i in range(len(BAD_VALUES))]
)
def test_invalid_scalars_and_spans_reject_in_python_and_json(field, value):
    model, _ = api()
    fields = data() | {field: value}
    with pytest.raises(ValidationError):
        model.model_validate(fields)
    with pytest.raises(ValidationError):
        model.model_validate_json(json.dumps(fields))


@pytest.mark.parametrize("field", tuple(fixture()["data"]))
def test_no_field_accepts_null(field):
    model, _ = api()
    with pytest.raises(ValidationError):
        model.model_validate(data() | {field: None})


@pytest.mark.parametrize(
    "field",
    [
        "binding_id",
        "authority_kind",
        "source_kind",
        "version_kind",
        "representation_id",
        "representation_sha256",
        "offset_unit",
        "span_sha256",
        "source_role",
        "captured_at",
    ],
)
def test_custom_string_scalars_are_not_coerced(field):
    model, _ = api()
    with pytest.raises(ValidationError):
        model.model_validate(data() | {field: CustomText(data()[field])})


@pytest.mark.parametrize("field", ["component_version", "span_start", "span_end"])
def test_custom_integer_scalars_are_not_coerced(field):
    model, _ = api()
    with pytest.raises(ValidationError):
        model.model_validate(data() | {field: CustomInteger(data()[field])})


def test_capture_time_uses_existing_portable_constraints():
    model, digest = api()
    assert (
        digest(model(**(data() | {"captured_at": "2026-09-26T17:00:00+01:00"})))
        == FIXED_SHA
    )
    for value in (
        datetime(2026, 9, 26, 16),  # noqa: DTZ001 - deliberate invalid-input control
        datetime(2026, 9, 26, 16, microsecond=1, tzinfo=UTC),
        CustomDatetime(2026, 9, 26, 16, tzinfo=UTC),
    ):
        with pytest.raises(ValidationError):
            model.model_validate(data() | {"captured_at": value})


MUTATIONS = {
    "binding_id": "synthetic-binding-02",
    "authority_kind": "authenticated_tenant",
    "authority_id": "synthetic-authority-02",
    "governance_scope_id": "synthetic-scope-02",
    "source_container_id": "synthetic-conversation-02",
    "source_object_id": "synthetic-message-02",
    "source_version_id": "synthetic-revision-02",
    "representation_sha256": "0" * 64,
    "span_start": 8,
    "span_end": 12,
    "span_sha256": "0" * 64,
    "source_role": "quoted_material",
    "captured_at": "2026-09-26T16:00:00.001Z",
}


@pytest.mark.parametrize("field,value", list(MUTATIONS.items()))
def test_complete_identity_changes_when_one_bound_value_changes(field, value):
    model, digest = api()
    assert digest(model(**(data() | {field: value}))) != FIXED_SHA


@pytest.mark.parametrize("index", [0, 2**53 - 1])
def test_empty_spans_and_portable_integer_boundary_are_structural_only(index):
    model, digest = api()
    binding = model(**(data() | {"span_start": index, "span_end": index}))
    assert binding.span_start == binding.span_end == index
    assert digest(binding) != FIXED_SHA


@pytest.mark.parametrize(
    "updates",
    [
        {"span_start": True},
        {"source_object_id": 42},
        {"unexpected": "synthetic-only"},
        {"span_end": 0},
    ],
)
def test_digest_and_instance_admission_revalidate_unsafe_copies(updates):
    model, digest = api()
    copied = model(**data()).model_copy(update=updates)
    with pytest.raises(ValidationError):
        digest(copied)
    with pytest.raises(ValidationError):
        model.model_validate(copied)


def test_digest_revalidates_constructed_data_and_accepts_valid_constructed_values():
    model, digest = api()
    assert digest(model.model_construct(**data())) == FIXED_SHA
    with pytest.raises(ValidationError):
        digest(model.model_construct(**(data() | {"binding_id": " "})))
    missing = data()
    del missing["binding_id"]
    with pytest.raises(ValidationError):
        digest(model.model_construct(**missing))


def test_digest_rejects_unsupported_argument_types_and_subclasses():
    model, digest = api()

    class DerivedBinding(model):
        pass

    for argument in (data(), None, "source-text", DerivedBinding(**data())):
        with pytest.raises(TypeError):
            digest(argument)


def test_frozen_fields_and_automatic_diagnostics_hide_inputs():
    model, _ = api()
    binding = model(**data())
    with pytest.raises(ValidationError):
        binding.source_object_id = "changed"
    assert all(
        value not in repr(binding) + str(binding)
        for field, value in data().items()
        if field in IDENTITIES
    )
    malformed = "synthetic-private-" + "a" * 129
    with pytest.raises(ValidationError) as error:
        model.model_validate(data() | {"binding_id": malformed})
    assert malformed not in str(error.value)
    assert error.value.errors()[0]["input"] == malformed


def test_unknown_fields_are_rejected_and_schema_requires_complete_identity():
    model, _ = api()
    for key in ("native", "excerpt_ref", "source_path", "binding_digest"):
        with pytest.raises(ValidationError):
            model.model_validate(data() | {key: "synthetic-only"})
    schema = model.model_json_schema()
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(data())
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(data())


def test_component_fixture_parity():
    assert (ROOT / FIXTURE).read_bytes() == (
        ROOT / "src/tldw_profile_core" / FIXTURE
    ).read_bytes()


def test_offline_wheel_contains_its_own_binding_and_fixture(tmp_path):
    project = tmp_path / "project"
    shutil.copytree(
        ROOT,
        project,
        ignore=shutil.ignore_patterns(
            "build", "dist", "*.egg-info", "__pycache__", ".pytest_cache"
        ),
    )
    wheels = tmp_path / "wheels"
    built = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--no-index",
            "--no-cache-dir",
            "--wheel-dir",
            str(wheels),
            str(project),
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    wheel = next(wheels.glob("tldw_profile_core-*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        assert RESOURCE in archive.namelist(), (
            "component fixture missing from built wheel"
        )
        assert archive.read(RESOURCE) == (ROOT / FIXTURE).read_bytes()
        assert any(
            name.endswith(".data/data/" + FIXTURE.as_posix())
            for name in archive.namelist()
        )
    script = """import importlib.resources, json, sys
sys.path.insert(0, sys.argv[1])
import tldw_profile_core
from tldw_profile_core.evidence_binding import OwnerVersionEvidenceBinding, owner_version_evidence_binding_digest
assert tldw_profile_core.__file__.startswith(sys.argv[1])
resource = importlib.resources.files('tldw_profile_core').joinpath('fixtures/evidence_binding/v1/01-unicode-message.json')
fixture = json.loads(resource.read_text(encoding='utf-8'))
assert owner_version_evidence_binding_digest(OwnerVersionEvidenceBinding(**fixture['data'])) == fixture['binding_sha256']
"""
    loaded = subprocess.run(
        [sys.executable, "-I", "-c", script, str(wheel)],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert loaded.returncode == 0, loaded.stdout + loaded.stderr

# Personal Context owner-version evidence binding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for native inline execution in the existing isolated worktree. The user's native execution preference persists. Execute task-by-task with checkbox tracking; one bounded fresh review follows the complete task.

**Goal:** Provide a bounded immutable exact-binding component and complete identity digest without source resolution or V1 changes.

**Architecture:** One explicit shared-core module validates 18 required scalars and canonicalizes a freshly validated snapshot before hashing. Mirrored synthetic fixtures qualify exact bytes; an offline-built wheel qualifies both code and resource packaging. Native and server owners still supply authorization, original text, current version and lifecycle fences.

**Tech Stack:** Existing shared package Python >=3.11, native .venv Python 3.12.11, Pydantic, rfc8785, hashlib/unicodedata, pytest/jsonschema, setuptools/wheel and Ruff; no dependency or package-version change.

**Spec:** [Reviewed component design](../specs/2026-09-26-personal-context-owner-version-evidence-binding-design.md).

**Task:** [TASK-25907.14](../../../backlog/tasks/task-25907.14%20-%20Add-bounded-owner-version-evidence-bindings-to-shared-profile-core.md).

**ADR required:** no new ADR; direct scoped implementation of accepted exact-binding convention.
**ADR path:** [ADR-185](../../../backlog/decisions/185-versioned-profile-evidence-and-temporal-claims.md).
**Reason:** Data-only structure/identity and existing packaging, with no new repository, runtime consumer or authority boundary.

## Global Constraints

- Exactly 18 required fields, no omitted defaults, nulls, excerpt handles or stored binding_digest. Component format 1 is independent of SERIALIZED_SCHEMA_VERSION=1.
- IDs: 1–128 codepoints and <=512 strict UTF-8 bytes; nonblank, no Cc/Cf, no trimming/normalization. Strict built-in scalars reject bool/float/subclass coercion.
- Integers satisfy 0 <= span_start <= span_end <= 2**53 - 1; no source-text length check is claimed. Empty spans are identity metadata, not support.
- Digests are exactly 64 lowercase ASCII hex characters. Complete binding digest is SHA-256 of existing RFC8785 canonical bytes, including capture time and binding ID.
- Datetime admission uses existing portable timezone/millisecond semantics after strict scalar checks; equivalent instants have identical bytes.
- model_validate(instance) always revalidates. Digest admission revalidates a detached complete stored dictionary before serialization, retaining unknown copied keys for rejection.
- Suppress automatic field repr and validation input strings; structured errors remain sensitive. No logging, source text, path, URL, capability or resolver.
- Duplicate raw JSON keys follow the existing parser's last-key-wins behavior. The component hashes validated values; future strict wire/transport parsing must reject duplicates.
- Leave V1 root exports, canonical-object union, schemas, fixtures, parser, SERIALIZED_SCHEMA_VERSION, dependencies/version and runtime consumers unchanged.
- No source lookup, profile persistence, grants, provider/server/app access, network, full application sweep, background work, push, PR or merge.
- Work only in /Users/macbook-dev/.codex/worktrees/personal-context-memory-baseline/tldw_chatbook on codex/personal-context-memory-baseline, with PYTHONPATH=.:packages/tldw_profile_core/src.
- Preserve independent TASK-25907.10 and roadmap suffix starting ## Follow-up: generated-answer effectiveness byte-for-byte; stage only owned paths and the owned roadmap prefix.

---

### Task 1: Bounded component, complete identity and packaging

**Files:**
- Create: packages/tldw_profile_core/src/tldw_profile_core/evidence_binding.py
- Test: packages/tldw_profile_core/tests/test_evidence_binding.py
- Create identical: packages/tldw_profile_core/fixtures/evidence_binding/v1/01-unicode-message.json and packages/tldw_profile_core/src/tldw_profile_core/fixtures/evidence_binding/v1/01-unicode-message.json
- Modify only fixture inclusion: packages/tldw_profile_core/pyproject.toml
- Create: backlog/docs/personal-context-owner-version-evidence-binding.md and Docs/superpowers/reviews/2026-09-26-personal-context-owner-version-evidence-binding-review.md
- Update: this plan, reviewed specification, TASK-25907.14 and owned roadmap prefix.

**Interfaces:**
- Consumes: FrozenModel; PortableDateTime; I_JSON_MAX_INTEGER; canonical_bytes(value: BaseModel) -> bytes from the existing shared package.
- Produces: explicit module OwnerVersionEvidenceBinding with the spec's 18 fields; owner_version_evidence_binding_digest(binding: OwnerVersionEvidenceBinding) -> str.

- [x] **Step 1: Preserve baseline and record reviewed scope.** Confirm linked worktree and branch; record HEAD as task BASE and independent byte hashes. Run the five affected existing core modules under native Python and fresh temporary roots.

```bash
PYTHONPATH=.:packages/tldw_profile_core/src .venv/bin/python -m pytest packages/tldw_profile_core/tests/test_evidence.py packages/tldw_profile_core/tests/test_canonical.py packages/tldw_profile_core/tests/test_models.py packages/tldw_profile_core/tests/test_schema_fixtures.py packages/tldw_profile_core/tests/test_interview.py -q --basetemp=/private/tmp/memory-binding-baseline-20260926 --junitxml=/private/tmp/memory-binding-baseline-20260926.xml
```

Expected: 181 distinct existing core cases pass; no profile/service import. Backlog task goes In Progress and receives this plan before implementation.

- [x] **Step 2: Write the complete tests and synthetic fixture below, then observe RED.** Tests use a lazy assertion for the missing API, avoiding collection ImportError. The fixed digest and canonical bytes are independent literals. Run the new module; expect assertion failures because binding API is missing. Run the offline wheel case separately; expect the missing packaged fixture assertion. Record exact failures and native receipts, not an expected failure count.

```bash
PYTHONPATH=.:packages/tldw_profile_core/src .venv/bin/python -m pytest packages/tldw_profile_core/tests/test_evidence_binding.py -q -k 'not offline_wheel' --basetemp=/private/tmp/memory-binding-red-20260926 --junitxml=/private/tmp/memory-binding-red-20260926.xml
PYTHONPATH=.:packages/tldw_profile_core/src .venv/bin/python -m pytest packages/tldw_profile_core/tests/test_evidence_binding.py::test_offline_wheel_contains_its_own_binding_and_fixture -q --basetemp=/private/tmp/memory-binding-wheel-red-20260926 --junitxml=/private/tmp/memory-binding-wheel-red-20260926.xml
```

Expected: first command fails from API assertions with fixture parity as a valid control; second fails because the new resource is absent from the built wheel. Build uses --no-index, --no-deps and --no-build-isolation in a temporary copy.

- [x] **Step 3: Implement the complete module below and add fixture packaging entries.** Use normal inherited Pydantic APIs; no new parser or generalized resolver. Add fixtures/evidence_binding/v1/*.json to tldw_profile_core package-data, and add a [tool.setuptools.data-files."fixtures/evidence_binding/v1"]-equivalent TOML key/value containing the top-level 01-unicode-message.json path. Keep all old entries byte-for-byte.

The actual TOML addition is a quoted key inside the existing tool.setuptools.data-files table:

```toml
"fixtures/evidence_binding/v1" = ["fixtures/evidence_binding/v1/01-unicode-message.json"]
```

- [x] **Step 4: Run GREEN, static checks and V1 preservation.** Run the new module plus the same five existing modules, with fresh temporary roots/JUnit. Run whole-file Ruff/format on only the two new Python files, fixing formatting without changing the contract. Verify package imports come from this worktree and all V1 files/old test bytes match BASE.

```bash
PYTHONPATH=.:packages/tldw_profile_core/src .venv/bin/python -m pytest packages/tldw_profile_core/tests/test_evidence_binding.py packages/tldw_profile_core/tests/test_evidence.py packages/tldw_profile_core/tests/test_canonical.py packages/tldw_profile_core/tests/test_models.py packages/tldw_profile_core/tests/test_schema_fixtures.py packages/tldw_profile_core/tests/test_interview.py -q --basetemp=/private/tmp/memory-binding-green-20260926 --junitxml=/private/tmp/memory-binding-green-20260926.xml
.venv/bin/python -m ruff check packages/tldw_profile_core/src/tldw_profile_core/evidence_binding.py packages/tldw_profile_core/tests/test_evidence_binding.py
.venv/bin/python -m ruff format --check packages/tldw_profile_core/src/tldw_profile_core/evidence_binding.py packages/tldw_profile_core/tests/test_evidence_binding.py
```

Expected: all targeted cases pass, including actual wheel resource/module loading; zero Ruff/format findings. Counts are reported from JUnit, never guessed or added across repeated runs.

- [x] **Step 5: Write API docs and execution review, then request one bounded read-only final review.** API documentation must show explicit import, fixture example, strict error meanings, unsafe-copy revalidation and raw-JSON/structured-error caveats. Record baseline, RED, GREEN and wheel receipts, scoped ADR/compatibility checks, and review attribution. Reviewer checks this task's full delta against BASE, not previously completed memory tasks; no implementation delegation or provider calls.

- [x] **Step 6: Resolve material findings and finish locally.** Give each material code fix a failing regression and targeted recovery. Check all eight Backlog criteria only after evidence exists, add concise Implementation Notes, update owned roadmap prefix, verify links/task IDs/foreign hashes and stage explicit paths. Commit locally as feat(memory): add bounded owner-version evidence bindings. Keep branch/worktree; do not push/merge/PR or archive it.

## Complete test module

```python
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
```

## Complete shared-core module

```python
"""Bounded asserted evidence identity; no source access or profile admission."""

from datetime import datetime
from hashlib import sha256
from typing import Annotated, Literal
from unicodedata import category

from pydantic import AfterValidator, BeforeValidator, ConfigDict, Field, model_validator

from .canonical import I_JSON_MAX_INTEGER, PortableDateTime, canonical_bytes
from .payloads import FrozenModel


def _builtin_string(value: object) -> str:
    if type(value) is not str:
        raise ValueError("value must be a built-in string")
    return value


def _builtin_integer(value: object) -> int:
    if type(value) is not int:
        raise ValueError("value must be a built-in integer")
    return value


def _opaque_identity(value: str) -> str:
    if not value.strip():
        raise ValueError("identity must not be blank")
    if len(value.encode("utf-8", errors="strict")) > 512:
        raise ValueError("identity must not exceed 512 UTF-8 bytes")
    if any(category(character) in {"Cc", "Cf"} for character in value):
        raise ValueError("identity must not contain control or format characters")
    return value


def _capture_scalar(value: object) -> datetime | str:
    if type(value) not in (datetime, str):
        raise ValueError("capture time must be a built-in datetime or string")
    return value


Identity = Annotated[
    str,
    BeforeValidator(_builtin_string),
    Field(min_length=1, max_length=128),
    AfterValidator(_opaque_identity),
]
Digest = Annotated[
    str,
    BeforeValidator(_builtin_string),
    Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"),
]
Offset = Annotated[
    int, BeforeValidator(_builtin_integer), Field(ge=0, le=I_JSON_MAX_INTEGER)
]
CaptureTime = Annotated[PortableDateTime, BeforeValidator(_capture_scalar)]


class OwnerVersionEvidenceBinding(FrozenModel):
    """Immutable structural assertions about one exact message representation."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        hide_input_in_errors=True,
    )

    component_version: Annotated[Literal[1], BeforeValidator(_builtin_integer)] = Field(
        repr=False
    )
    binding_id: Identity = Field(repr=False)
    authority_kind: Annotated[
        Literal["local_profile", "authenticated_tenant"],
        BeforeValidator(_builtin_string),
    ] = Field(repr=False)
    authority_id: Identity = Field(repr=False)
    governance_scope_id: Identity = Field(repr=False)
    source_kind: Annotated[
        Literal["conversation_message"], BeforeValidator(_builtin_string)
    ] = Field(repr=False)
    source_container_id: Identity = Field(repr=False)
    source_object_id: Identity = Field(repr=False)
    version_kind: Annotated[
        Literal["owner_immutable"], BeforeValidator(_builtin_string)
    ] = Field(repr=False)
    source_version_id: Identity = Field(repr=False)
    representation_id: Annotated[
        Literal["message_content_text_v1"], BeforeValidator(_builtin_string)
    ] = Field(repr=False)
    representation_sha256: Digest = Field(repr=False)
    offset_unit: Annotated[
        Literal["unicode_codepoint"], BeforeValidator(_builtin_string)
    ] = Field(repr=False)
    span_start: Offset = Field(repr=False)
    span_end: Offset = Field(repr=False)
    span_sha256: Digest = Field(repr=False)
    source_role: Annotated[
        Literal[
            "direct_user_message",
            "quoted_material",
            "attachment",
            "tool_result",
            "imported_material",
        ],
        BeforeValidator(_builtin_string),
    ] = Field(repr=False)
    captured_at: CaptureTime = Field(repr=False)

    @model_validator(mode="after")
    def _ordered_span(self) -> "OwnerVersionEvidenceBinding":
        if self.span_start > self.span_end:
            raise ValueError("span_start must not exceed span_end")
        return self


def owner_version_evidence_binding_digest(binding: OwnerVersionEvidenceBinding) -> str:
    """Hash a freshly validated snapshot of complete binding field values.

    Args:
        binding: Exact component-class instance; assertions remain untrusted.

    Returns:
        Lowercase SHA-256 of the validated component's canonical UTF-8 bytes.

    Raises:
        TypeError: The argument is not the exact component class.
        pydantic.ValidationError: Stored field values or keys are malformed.
    """
    if type(binding) is not OwnerVersionEvidenceBinding:
        raise TypeError("binding must be an exact OwnerVersionEvidenceBinding instance")
    validated = OwnerVersionEvidenceBinding.model_validate(dict(vars(binding)))
    return sha256(canonical_bytes(validated)).hexdigest()
```

## Mirrored synthetic fixture

```json
{
  "case": "owner_version_unicode_message",
  "source_text": "Hi 👋 — café",
  "data": {
    "authority_id": "synthetic-authority-01",
    "authority_kind": "local_profile",
    "binding_id": "synthetic-binding-01",
    "captured_at": "2026-09-26T16:00:00.000Z",
    "component_version": 1,
    "governance_scope_id": "synthetic-scope-01",
    "offset_unit": "unicode_codepoint",
    "representation_id": "message_content_text_v1",
    "representation_sha256": "bdd355f021e0095bbd9ce9da729cfd0b9556b01bc01c5b59461ebc4e12e55bef",
    "source_container_id": "synthetic-conversation-01",
    "source_kind": "conversation_message",
    "source_object_id": "synthetic-message-01",
    "source_role": "direct_user_message",
    "source_version_id": "synthetic-revision-01",
    "span_end": 11,
    "span_sha256": "850f7dc43910ff890f8879c0ed26fe697c93a067ad93a7d50f466a7028a9bf4e",
    "span_start": 7,
    "version_kind": "owner_immutable"
  },
  "canonical_utf8": "{\"authority_id\":\"synthetic-authority-01\",\"authority_kind\":\"local_profile\",\"binding_id\":\"synthetic-binding-01\",\"captured_at\":\"2026-09-26T16:00:00.000Z\",\"component_version\":1,\"governance_scope_id\":\"synthetic-scope-01\",\"offset_unit\":\"unicode_codepoint\",\"representation_id\":\"message_content_text_v1\",\"representation_sha256\":\"bdd355f021e0095bbd9ce9da729cfd0b9556b01bc01c5b59461ebc4e12e55bef\",\"source_container_id\":\"synthetic-conversation-01\",\"source_kind\":\"conversation_message\",\"source_object_id\":\"synthetic-message-01\",\"source_role\":\"direct_user_message\",\"source_version_id\":\"synthetic-revision-01\",\"span_end\":11,\"span_sha256\":\"850f7dc43910ff890f8879c0ed26fe697c93a067ad93a7d50f466a7028a9bf4e\",\"span_start\":7,\"version_kind\":\"owner_immutable\"}",
  "binding_sha256": "dae4dd8a13189b7bf86559f7adaab2d6992fe3040a1d255b2c212dc7b15801fb"
}
```

## Review Focus

- Exact snapshot revalidation: unsafe copying, missing/unknown stored keys, instance/subclass admission and serialization performed only after validation.
- Python/JSON scalar parity, portable offset edges, whole/source/span digest distinctions, datetime UTC normalization and immutable field/error diagnostics.
- Canonical fixture coverage of all 18 fields; new resource and module actually included/loaded from the offline wheel; V1 fixture/schema/export bytes unaffected.
- Digest/data structure remains untrusted: no source grant, native tag, source read, historical-text promise, semantic support, approval or automatic use.
- Source policy cannot be inferred from metadata; future transport must reject duplicate JSON keys and govern even identity-bearing metadata. No runtime adapter is silently supplied.

## Plan self-review

Spec coverage maps strict admission and complete identity to Steps 2–4, actual packaging to Steps 2–4, documentation/limits/review to Step 5 and tracker/compatibility preservation to Step 6. Interfaces match the complete code below. Review fixed the fixture inclusion omission and distinguished parser/diagnostic limitations from runtime authority. One self-contained implementation task; no setup-only subtasks, placeholder code or dependency changes.

## Native execution progress

Baseline: 181 passes. RED: 156 missing-API failures, zero errors and one fixture-parity control; offline wheel RED proves the missing resource. GREEN: 339 distinct passes (158 new, 181 existing), including wheel imports/resources. Whole-file Ruff and format passed after import ordering, explicit subprocess check=False and a documented deliberately naive datetime rejection control. Added --no-cache-dir to keep wheel build caches inside the temporary run. Receipts use /private/tmp/memory-binding-{baseline,red,wheel-red,green}-20260926.xml; no repeated run is an additional case. Independent final review found no Critical/Important/Minor issues. Task criteria and closeout are recorded in the final tracker/receipt; the branch/worktree stay local.

Final native task run: 339 distinct passes in 1.47s, /private/tmp/memory-binding-task-done-20260926.xml. The bounded fresh reviewer independently probed all 18 fields, fixed digests and malformed-copy rejection; it did not rerun the 339 tests or wheel build. No review findings or implementation rulings remain.

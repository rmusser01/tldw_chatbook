"""Separate structural/semantic and actual installed-wheel conformance."""

import importlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

ROOT = Path(__file__).parents[1]
RESOURCES = sorted((ROOT / "fixtures/v2").glob("*.json"))
SCHEMA = "personal-context-v2.json"
META = "personal-context-v2-meta.json"
RULES = {
    "canonicalization": "rfc8785-v1",
    "canonicalDateTime": "utc-milliseconds-v1",
    "iJsonMaxSafeInteger": 9007199254740991,
    "canonicalPayloadMaxUtf8Bytes": 16384,
    "canonicalClaimMaxUtf8Bytes": 16384,
    "canonicalRecordMaxUtf8Bytes": 65536,
    "canonicalProposalMaxUtf8Bytes": 98304,
    "canonicalManifestMaxUtf8Bytes": 16384,
    "pendingProposalExpiryDays": 90,
    "aggregateRules": "profile-aggregates-v2",
    "claimProjection": "profile-claim-v2",
    "bindingProjection": "owner-version-evidence-binding-v1",
    "attributionRules": "claim-attribution-v2",
    "temporalRelations": "profile-relations-v2",
    "disclosureRules": "profile-disclosure-ceiling-v2",
    "manifestRequirements": "profile-context-requirements-v2",
}


def schema():
    path = ROOT / "schemas" / SCHEMA
    assert path.exists(), "V2 schema missing"
    return json.loads(path.read_text())


@pytest.mark.parametrize("path", RESOURCES, ids=lambda p: p.stem)
def test_structural_and_semantic_fixed_cases(path):
    f = json.loads(path.read_text())
    s = schema()
    Draft202012Validator.check_schema(s)
    assert (not list(Draft202012Validator(s).iter_errors(f["data"]))) is f[
        "structurally_valid"
    ]
    c = importlib.import_module("tldw_profile_core.v2_contract")
    if f["valid"]:
        c.validate_v2_object(f["data"])
    else:
        with pytest.raises(ValueError, match="^invalid V2 profile object$"):
            c.validate_v2_object(f["data"])
    assert (
        path.read_bytes()
        == (ROOT / "src/tldw_profile_core/fixtures/v2" / path.name).read_bytes()
    )


def test_required_dialect_and_export_reproduction(tmp_path):
    s = schema()
    meta = json.loads((ROOT / "schemas" / META).read_text())
    c = importlib.import_module("tldw_profile_core.v2_contract")
    assert s["$id"] == "urn:tldw:profile-core:schema:personal-context:2"
    assert s["$schema"] == meta["$id"] == "urn:tldw:profile-core:json-schema:dialect:2"
    assert s["x-tldw-profile-semantics"] == RULES
    assert (
        meta["$vocabulary"]["urn:tldw:profile-core:json-schema:vocabulary:semantic:2"]
        is True
    )
    Draft202012Validator.check_schema(meta)
    # Avoid resolving external Draft references: validate the closed rule declaration directly.
    rule_schema = meta["properties"]["x-tldw-profile-semantics"]
    v = Draft202012Validator(rule_schema)
    v.validate(RULES)
    for key in RULES:
        with pytest.raises(ValidationError):
            v.validate({k: val for k, val in RULES.items() if k != key})
        with pytest.raises(ValidationError):
            v.validate(RULES | {key: "wrong"})
    for name, exporter in [
        (SCHEMA, c.export_v2_json_schema),
        (META, c.export_v2_meta_schema),
    ]:
        out = tmp_path / name
        exporter(out)
        assert out.read_bytes() == (ROOT / "schemas" / name).read_bytes()
        assert (
            out.read_bytes()
            == (ROOT / "src/tldw_profile_core/schemas" / name).read_bytes()
        )
    assert all("$defs" not in branch for branch in s["anyOf"])


@pytest.mark.parametrize(
    "model_name", ["ProfileRecordV2", "ProfileProposalV2", "ProfileManifestV2"]
)
def test_structural_closed_aggregate_fields(model_name):
    s = schema()
    d = s["$defs"][model_name]
    assert d["additionalProperties"] is False
    assert set(d["required"]) == set(d["properties"])


def test_structural_denied_tombstone_and_receipt_shapes():
    v = Draft202012Validator(schema())
    tomb = json.loads((ROOT / "fixtures/v2/04-deleted-record.json").read_text())["data"]
    active = json.loads((ROOT / "fixtures/v2/02-active-record.json").read_text())[
        "data"
    ]
    receipt = json.loads((ROOT / "fixtures/v2/07-resolved-proposal.json").read_text())[
        "data"
    ]
    assert list(v.iter_errors(tomb | {"provenance": active["provenance"]}))
    assert list(v.iter_errors(receipt | {"provenance": active["provenance"]}))
    assert list(v.iter_errors(active | {"no_expiry": 1}))
    assert list(v.iter_errors(active | {"schema_version": True}))


@pytest.mark.parametrize("digest", ["A" * 64, "g" * 64])
@pytest.mark.parametrize(
    "path",
    [
        ("claim", "claim_digest"),
        ("claim", "approval_receipt", "claim_digest"),
        ("claim", "support_assessments", 0, "claim_digest"),
        ("claim", "support_assessments", 0, "binding_digest"),
        ("claim", "evidence_bindings", 0, "representation_sha256"),
        ("claim", "evidence_bindings", 0, "span_sha256"),
    ],
)
def test_schema_and_reference_validator_reject_malformed_digests(path, digest):
    value = json.loads((ROOT / "fixtures/v2/02-active-record.json").read_text())["data"]
    target = value
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = digest

    with pytest.raises(ValidationError):
        Draft202012Validator(schema()).validate(value)
    c = importlib.import_module("tldw_profile_core.v2_contract")
    with pytest.raises(ValueError, match="^invalid V2 profile object$"):
        c.validate_v2_object(value)


def test_real_offline_installed_wheel_conformance(tmp_path):
    schema()
    project = tmp_path / "project"
    shutil.copytree(
        ROOT,
        project,
        ignore=shutil.ignore_patterns(
            "build", "dist", "*.egg-info", "__pycache__", ".pytest_cache", ".ruff_cache"
        ),
    )
    wheels = tmp_path / "wheels"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "--isolated",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--no-index",
            "--no-cache-dir",
            "--wheel-dir",
            str(wheels),
            str(project),
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wheel = next(wheels.glob("*.whl"))
    resources = [
        p.relative_to(ROOT)
        for folder in (
            ROOT / "fixtures/v2",
            ROOT / "schemas",
            ROOT / "fixtures/v1",
            ROOT / "fixtures/evidence_binding/v1",
            ROOT / "fixtures/claim_meaning/v2",
        )
        for p in sorted(folder.glob("*.json"))
    ]
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata = next(n for n in names if n.endswith(".dist-info/METADATA"))
        assert "Requires-Python: >=3.12" in archive.read(metadata).decode().splitlines()
        for relative in resources:
            assert (
                archive.read("tldw_profile_core/" + relative.as_posix())
                == (ROOT / relative).read_bytes()
            )
            distributed = [
                n for n in names if n.endswith(".data/data/" + relative.as_posix())
            ]
            assert len(distributed) == 1
            assert archive.read(distributed[0]) == (ROOT / relative).read_bytes()
    installed = tmp_path / "installed"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "--isolated",
            "install",
            "--no-deps",
            "--no-index",
            "--no-cache-dir",
            "--target",
            str(installed),
            str(wheel),
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    script = """import sys,pathlib,json,importlib.resources
sys.path.insert(0,sys.argv[1])
import tldw_profile_core as v1
from tldw_profile_core import v2_contract as c, v2_models as m
from jsonschema import Draft202012Validator, ValidationError
root=pathlib.Path(sys.argv[1]).resolve()
for mod in (v1,c,m):assert pathlib.Path(mod.__file__).resolve().is_relative_to(root)
assert v1.SERIALIZED_SCHEMA_VERSION==1 and not hasattr(v1,'ProfileRecordV2')
package=importlib.resources.files('tldw_profile_core')
s=json.loads(package.joinpath('schemas/personal-context-v2.json').read_text())
for path in sorted((root/'fixtures/v2').glob('*.json')):
 f=json.loads(path.read_text());assert path.read_bytes()==package.joinpath('fixtures/v2/'+path.name).read_bytes()
 Draft202012Validator(s).validate(f['data'])
 if f['valid']:
  value=c.validate_v2_json(json.dumps(f['data']))
  assert c.canonical_v2_bytes(value).decode()==f['canonical_utf8']
  assert c.v2_object_digest(value)==f['sha256']
  assert c.v2_integrity_tag(value,bytes(range(32)))==f['integrity_tag']
 else:
  try:c.validate_v2_object(f['data'])
  except ValueError:pass
  else:raise AssertionError('invalid fixture accepted')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(installed)],
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_actual_dialect_validates_export_without_nested_declaration_errors():
    from jsonschema_specifications import REGISTRY
    from referencing import Resource

    meta = json.loads((ROOT / "schemas" / META).read_text())
    registry = REGISTRY.with_resource(meta["$id"], Resource.from_contents(meta))
    validator = Draft202012Validator(meta, registry=registry)
    exported = schema()
    errors = list(validator.iter_errors(exported))
    assert not errors, [(tuple(e.path), e.message) for e in errors[:5]]
    missing = {k: v for k, v in exported.items() if k != "x-tldw-profile-semantics"}
    assert list(validator.iter_errors(missing)), (
        "declared V2 root must require the semantic map"
    )

"""Inactive aggregate conformance: no application/source/provider use."""

import copy
import importlib
import importlib.util
import json
from hashlib import sha256
from pathlib import Path

import pytest
from pydantic import BaseModel, TypeAdapter

ROOT = Path(__file__).parents[1]
FIXTURES = sorted((ROOT / "fixtures/v2").glob("*.json"))


def api():
    assert importlib.util.find_spec("tldw_profile_core.v2_models"), (
        "V2 aggregates missing"
    )
    return importlib.import_module("tldw_profile_core.v2_models")


def contract():
    assert importlib.util.find_spec("tldw_profile_core.v2_contract"), (
        "V2 contract missing"
    )
    return importlib.import_module("tldw_profile_core.v2_contract")


def fixture(name="02-active-record"):
    return json.loads((ROOT / "fixtures/v2" / (name + ".json")).read_text())


def data(name="02-active-record"):
    return fixture(name)["data"]


def set_path(value, path, replacement):
    target = value
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = replacement
    return value


def rehash(record):
    claim = record["claim"]
    projection = {
        "projection": "profile-claim-v2",
        **{
            k: record[k]
            for k in ("profile_id", "record_id", "scope_id", "kind", "payload")
        },
        **{k: claim[k] for k in ("claim_basis", "temporal_validity", "relations")},
    }
    digest = sha256(
        json.dumps(
            projection, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()
    claim["claim_digest"] = digest
    for item in claim["support_assessments"]:
        item["claim_digest"] = digest
    if claim["approval_receipt"]:
        claim["approval_receipt"]["claim_digest"] = digest
    return record


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: p.stem)
def test_fixed_aggregate_cases(path):
    f = json.loads(path.read_text())
    model = getattr(api(), f["model"])
    if not f["valid"]:
        with pytest.raises((ValueError, TypeError)):
            model.model_validate(f["data"])
        return
    value = model.model_validate(f["data"])
    c = contract()
    assert c.canonical_v2_bytes(value).decode() == f["canonical_utf8"]
    assert c.v2_object_digest(value) == f["sha256"]
    assert c.v2_integrity_tag(value, bytes(range(32))) == f["integrity_tag"]
    assert "p1" not in repr(value) and "concise" not in repr(value)
    assert type(c.validate_v2_json(json.dumps(f["data"]))) is model
    assert type(TypeAdapter(model).validate_python(f["data"])) is model


@pytest.mark.parametrize(
    "name", ["01-manifest", "02-active-record", "05-create-proposal"]
)
def test_every_required_key(name):
    value = data(name)
    model = getattr(api(), fixture(name)["model"])
    for key in value:
        stripped = {k: v for k, v in value.items() if k != key}
        with pytest.raises((ValueError, TypeError)):
            model.model_validate(stripped)


@pytest.mark.parametrize("value", [True, False, "2", 1, 3, None])
def test_reject_envelope_versions(value):
    with pytest.raises((ValueError, TypeError)):
        api().ProfileRecordV2.model_validate(data() | {"schema_version": value})


def test_numeric_integer_envelope_and_deny_default():
    d = data()
    d["schema_version"] = 2.0
    d["controls"].pop("model_disclosure")
    model = api().ProfileRecordV2.model_validate(d)
    assert model.schema_version == 2
    assert model.controls.model_disclosure.kind == "deny"
    assert (
        '"model_disclosure":{"kind":"deny"}'
        in contract().canonical_v2_bytes(model).decode()
    )


MUTATIONS = [
    (["profile_id"], ""),
    (["record_id"], "x\u200b"),
    (["scope_id"], "x\x00"),
    (["version_id"], "a" * 129),
    (["profile_id"], "\ud800"),
    (["no_expiry"], 1),
    (["no_expiry"], "false"),
    (["schema_version"], True),
    (["created_at"], "2026-09-26T18:00:00.000Z"),
    (["updated_at"], "2026-09-26T16:00:00.0001Z"),
    (["created_at"], 0),
    (["created_at"], "2026-09-26 15:00:00Z"),
    (["controls", "model_disclosure"], None),
    (["controls", "model_disclosure"], {"kind": "unknown"}),
    (["controls", "model_disclosure"], {"kind": "deny", "purposes": ["conversation"]}),
    (
        ["controls", "model_disclosure"],
        {"kind": "on_device_only", "purposes": ["summary", "conversation"]},
    ),
    (
        ["controls", "model_disclosure"],
        {"kind": "on_device_only", "purposes": ["conversation", "conversation"]},
    ),
    (["controls", "model_disclosure"], {"kind": "on_device_only", "purposes": []}),
    (
        ["controls", "model_disclosure"],
        {"kind": "reviewed_destinations", "audiences": []},
    ),
    (
        ["controls", "model_disclosure"],
        {
            "kind": "reviewed_destinations",
            "audiences": [{"audience_handle": "*", "purposes": ["conversation"]}],
        },
    ),
    (["controls", "sync_mode"], "anything"),
    (["claim", "claim_digest"], "A" * 64),
    (["claim", "approval_receipt", "record_version_id"], "v0"),
    (["claim", "approval_receipt", "acted_at"], "2026-09-26T14:00:00.000Z"),
    (["claim", "approval_receipt", "acted_at"], "2026-09-26T18:00:00.000Z"),
    (["claim", "support_assessments", 0, "binding_id"], "foreign"),
    (["claim", "support_assessments", 0, "binding_digest"], "0" * 64),
    (["claim", "support_assessments", 0, "claim_digest"], "0" * 64),
    (["claim", "support_assessments", 0, "origin"], None),
    (["claim", "support_assessments", 0, "origin", "method"], "a" * 65),
    (["claim", "support_assessments", 0, "origin", "method"], "method\n"),
    (["claim", "support_assessments", 0, "origin", "method"], "api_key=abcdefg"),
    (["claim", "support_assessments", 0, "assessed_at"], None),
    (["claim", "support_assessments", 0, "status"], "not_assessed"),
    (["claim", "evidence_bindings", 0, "component_version"], 1.0),
    (["claim", "evidence_bindings", 0, "span_start"], 7.0),
    (["claim", "evidence_bindings", 0, "captured_at"], "2026-09-26T18:00:00.000Z"),
    (["expires_at"], "2026-09-27T17:00:00.000Z"),
    (["provenance"], None),
    (["claim"], None),
    (["payload"], None),
]


@pytest.mark.parametrize(
    "path,replacement", MUTATIONS, ids=[f"mutation-{i}" for i in range(len(MUTATIONS))]
)
def test_record_rejections(path, replacement):
    with pytest.raises((ValueError, TypeError)):
        api().ProfileRecordV2.model_validate(set_path(data(), path, replacement))


@pytest.mark.parametrize("field", ["evidence_bindings", "support_assessments"])
def test_duplicate_claim_arrays(field):
    d = data()
    d["claim"][field] *= 2
    with pytest.raises((ValueError, TypeError)):
        api().ProfileRecordV2.model_validate(d)


@pytest.mark.parametrize("version", [True, "0", -1, 2**53, 0.5, float("inf")])
def test_manifest_counter_rejections(version):
    with pytest.raises((ValueError, TypeError)):
        api().ProfileManifestV2.model_validate(
            data("01-manifest") | {"evidence_retirement_epoch": version}
        )


@pytest.mark.parametrize("state", ["accepted", "rejected", "superseded", "expired"])
def test_resolved_receipt_states(state):
    d = data("07-resolved-proposal") | {"state": state}
    assert api().ProfileProposalV2.model_validate(d).state == state
    with pytest.raises((ValueError, TypeError)):
        api().ProfileProposalV2.model_validate(d | {"provenance": data()["provenance"]})


@pytest.mark.parametrize(
    "path,replacement",
    [
        (["created_at"], "2026-09-26T16:00:00.000Z"),
        (["scope_id"], "other"),
        (
            ["proposed_record", "claim", "approval_receipt"],
            data()["claim"]["approval_receipt"],
        ),
        (["proposed_record", "parent_version_id"], "v0"),
        (["proposed_record", "state"], "archived"),
        (["target_record_id"], "r1"),
        (["provenance"], None),
    ],
    ids=[f"proposal-{i}" for i in range(7)],
)
def test_proposal_rejections(path, replacement):
    with pytest.raises((ValueError, TypeError)):
        api().ProfileProposalV2.model_validate(
            set_path(data("05-create-proposal"), path, replacement)
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("payload", data()["payload"]),
        ("provenance", data()["provenance"]),
        ("claim", data()["claim"]),
        ("no_expiry", True),
    ],
)
def test_deleted_has_no_content(field, value):
    with pytest.raises((ValueError, TypeError)):
        api().ProfileRecordV2.model_validate(data("04-deleted-record") | {field: value})


@pytest.mark.parametrize("hold", ["legacy_review", "privacy_review"])
def test_legacy_requires_unknown_held_deny_and_no_attribution(hold):
    d = data()
    c = d["claim"]
    c.update(
        claim_basis="legacy_unknown",
        temporal_validity={"kind": "unknown"},
        relations=[],
        evidence_bindings=[],
        support_assessments=[],
        approval_receipt=None,
        confidence_estimate=None,
        salience_hint=None,
        hold_reason=hold,
    )
    if hold == "privacy_review":
        d["provenance"].update(
            source="privacy_retirement",
            actor="system",
            reason_code="privacy_retirement",
            derived_from_record_id=None,
        )
    rehash(d)
    api().ProfileRecordV2.model_validate(d)
    broken = copy.deepcopy(d)
    broken["claim"]["hold_reason"] = None
    with pytest.raises((ValueError, TypeError)):
        api().ProfileRecordV2.model_validate(broken)


@pytest.mark.parametrize(
    "field",
    ["approval_receipt", "confidence_estimate", "salience_hint", "support_assessments"],
)
def test_privacy_successor_cannot_copy_attribution(field):
    d = data()
    d["claim"].update(
        approval_receipt=None,
        confidence_estimate=None,
        salience_hint=None,
        support_assessments=[],
        hold_reason="privacy_review",
    )
    d["provenance"].update(
        source="privacy_retirement",
        actor="system",
        reason_code="privacy_retirement",
        derived_from_record_id=None,
    )
    api().ProfileRecordV2.model_validate(d)
    replacements = {
        "approval_receipt": data()["claim"]["approval_receipt"],
        "confidence_estimate": {
            "value": 0.5,
            "origin": data()["claim"]["support_assessments"][0]["origin"],
            "assessed_at": d["updated_at"],
            "claim_digest": d["claim"]["claim_digest"],
        },
        "salience_hint": {
            "priority": "high",
            "origin": data()["claim"]["support_assessments"][0]["origin"],
            "chosen_at": d["updated_at"],
            "record_version_id": "v1",
        },
        "support_assessments": data()["claim"]["support_assessments"],
    }
    d["claim"][field] = replacements[field]
    with pytest.raises((ValueError, TypeError)):
        api().ProfileRecordV2.model_validate(d)


def test_bound_source_and_self_edge_local_invariants():
    d = data()
    d["claim"]["approval_receipt"] = None
    d["claim"]["temporal_validity"] = {
        "kind": "interval",
        "valid_from": d["created_at"],
        "valid_until": None,
        "basis": {"kind": "bound_source", "binding_ids": ["foreign"]},
    }
    rehash(d)
    with pytest.raises((ValueError, TypeError)):
        api().ProfileRecordV2.model_validate(d)
    d["claim"]["temporal_validity"]["basis"]["binding_ids"] = [
        d["claim"]["evidence_bindings"][0]["binding_id"]
    ]
    rehash(d)
    api().ProfileRecordV2.model_validate(d)
    d["claim"]["relations"] = [
        {
            "edge_id": "e1",
            "kind": "correction_of",
            "target_record_id": "r1",
            "target_version_id": "v1",
            "effect": {"kind": "all_target_validity"},
        }
    ]
    rehash(d)
    with pytest.raises((ValueError, TypeError)):
        api().ProfileRecordV2.model_validate(d)
    d["claim"]["relations"][0]["target_version_id"] = "v0"
    rehash(d)
    api().ProfileRecordV2.model_validate(d)


@pytest.mark.parametrize("which", ["confidence", "salience"])
def test_optional_attribution_binds_current_meaning_and_version(which):
    d = data()
    origin = d["claim"]["support_assessments"][0]["origin"]
    if which == "confidence":
        field = "confidence_estimate"
        value = {
            "value": 1,
            "origin": origin,
            "assessed_at": d["updated_at"],
            "claim_digest": d["claim"]["claim_digest"],
        }
        key = "claim_digest"
        bad = "0" * 64
    else:
        field = "salience_hint"
        value = {
            "priority": "high",
            "origin": origin,
            "chosen_at": d["updated_at"],
            "record_version_id": "v1",
        }
        key = "record_version_id"
        bad = "v0"
    d["claim"][field] = value
    api().ProfileRecordV2.model_validate(d)
    value[key] = bad
    with pytest.raises((ValueError, TypeError)):
        api().ProfileRecordV2.model_validate(d)


@pytest.mark.parametrize(
    "name", ["01-manifest", "02-active-record", "05-create-proposal"]
)
def test_duplicate_aware_and_compiled_json_paths(name):
    m = getattr(api(), fixture(name)["model"])
    raw = json.dumps(data(name))
    duplicate = raw[:-1] + ',"profile_id":"other"}'
    with pytest.raises((ValueError, TypeError)):
        m.model_validate_json(duplicate)
    with pytest.raises(ValueError, match="^invalid V2 profile JSON$"):
        contract().validate_v2_json(duplicate)
    with pytest.raises(TypeError):
        TypeAdapter(m).validate_json(raw)

    class Wrapper(BaseModel):
        item: m

    with pytest.raises(TypeError):
        Wrapper.model_validate_json('{"item":' + raw + "}")
    nested = raw.replace('"kind": "deny"', '"kind": "deny", "kind": "deny"')
    if nested != raw:
        with pytest.raises((ValueError, TypeError)):
            contract().validate_v2_json(nested)


@pytest.mark.parametrize(
    "name", ["01-manifest", "02-active-record", "05-create-proposal"]
)
def test_every_serializer_revalidates_unsafe_root_state(name):
    model = getattr(api(), fixture(name)["model"]).model_validate(data(name))
    bad = model.model_copy(update={"private_marker": "private-content"})
    c = contract()
    for helper, args in [
        (c.canonical_v2_bytes, ()),
        (c.v2_object_digest, ()),
        (c.v2_integrity_tag, (bytes(range(32)),)),
    ]:
        with pytest.raises(ValueError, match="^invalid V2 profile object$") as exc:
            helper(bad, *args)
        assert "private-content" not in str(exc.value)

    class FilteringDict(dict):
        def keys(self):
            raise AssertionError("callback must not run")

        def items(self):
            raise AssertionError("callback must not run")

    object.__setattr__(model, "__dict__", FilteringDict(vars(model)))
    with pytest.raises(ValueError, match="^invalid V2 profile object$"):
        c.canonical_v2_bytes(model)


@pytest.mark.parametrize("replacement", [True, 1.0, "1"])
def test_binding_canonical_boundary_preserves_strict_integer(replacement):
    model = api().ProfileRecordV2.model_validate(data())
    binding = model.claim.evidence_bindings[0].model_copy(
        update={"component_version": replacement}
    )
    broken = model.model_copy(
        update={
            "claim": model.claim.model_copy(update={"evidence_bindings": (binding,)})
        }
    )
    with pytest.raises(ValueError, match="^invalid V2 profile object$"):
        contract().canonical_v2_bytes(broken)


@pytest.mark.parametrize("key", [b"", b"x" * 31, b"x" * 33, "x" * 32, bytearray(32)])
def test_integrity_keys_are_exact_32_bytes(key):
    model = api().ProfileRecordV2.model_validate(data())
    with pytest.raises((ValueError, TypeError)):
        contract().v2_integrity_tag(model, key)


def test_payload_and_claim_byte_limits_after_defaults():
    d = data()
    d["payload"]["value"] = "😀" * 4096
    d["claim"]["approval_receipt"] = None
    d["claim"]["support_assessments"] = []
    rehash(d)
    with pytest.raises((ValueError, TypeError)):
        api().ProfileRecordV2.model_validate(d)
    d = data()
    d["semantic_key"] = {"namespace": "x" * 16384, "subject": "y" * 16384}
    d["payload"]["value"] = "v" * 15800
    d["claim"]["evidence_bindings"] = []
    d["claim"]["support_assessments"] = []
    d["claim"]["approval_receipt"] = None
    rehash(d)
    assert (
        len(contract().canonical_v2_bytes(api().ProfileRecordV2.model_validate(d)))
        < 65536
    )


def test_unknown_nonfinite_subclass_and_ambiguous_dispatch():
    c = contract()

    class Strange(str):
        pass

    for value in [
        data() | {"record_id": Strange("r1")},
        data() | {"revision": 0},
        {"schema_version": 1},
        float("nan"),
    ]:
        with pytest.raises(ValueError, match="^invalid V2 profile object$"):
            c.validate_v2_object(value)
    for raw in ['{"x":NaN}', "{", '{"x":Infinity}', b"\xff"]:
        with pytest.raises(ValueError, match="^invalid V2 profile JSON$"):
            c.validate_v2_json(raw)


@pytest.mark.parametrize(
    "value",
    [True, "0.5", -0.01, 1.01, float("inf"), float("nan")],
    ids=["bool", "string", "negative", "over-one", "infinity", "nan"],
)
def test_confidence_number_rejections(value):
    d = data()
    d["claim"]["confidence_estimate"] = {
        "value": value,
        "origin": d["claim"]["support_assessments"][0]["origin"],
        "assessed_at": d["updated_at"],
        "claim_digest": d["claim"]["claim_digest"],
    }
    with pytest.raises((ValueError, TypeError)):
        api().ProfileRecordV2.model_validate(d)


@pytest.mark.parametrize(
    "status", ["not_assessed", "insufficient", "supports", "contradicts"]
)
def test_assessment_status_and_empty_span(status):
    d = data()
    a = d["claim"]["support_assessments"][0]
    a["status"] = status
    if status == "not_assessed":
        a.update(origin=None, assessed_at=None)
    api().ProfileRecordV2.model_validate(d)
    binding = d["claim"]["evidence_bindings"][0]
    binding["span_end"] = binding["span_start"]
    a["binding_digest"] = sha256(
        json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if status in ("supports", "contradicts"):
        with pytest.raises((ValueError, TypeError)):
            api().ProfileRecordV2.model_validate(d)
    else:
        api().ProfileRecordV2.model_validate(d)


@pytest.mark.parametrize(
    "policy",
    [
        {
            "kind": "on_device_only",
            "purposes": ["conversation", "embedding", "interview", "summary"],
        },
        {
            "kind": "reviewed_destinations",
            "audiences": [
                {"audience_handle": "a", "purposes": ["conversation"]},
                {"audience_handle": "b", "purposes": ["summary"]},
            ],
        },
    ],
)
def test_policy_pairs_are_preserved_without_enrollment(policy):
    d = data()
    d["controls"]["model_disclosure"] = policy
    model = api().ProfileRecordV2.model_validate(d)
    output = json.loads(contract().canonical_v2_bytes(model))
    assert output["controls"]["model_disclosure"] == policy
    assert output["controls"]["sync_mode"] == "device_only"


@pytest.mark.parametrize("mode", ["expired", "no-expiry"])
def test_working_context_expiry_decisions(mode):
    d = data()
    d["kind"] = "working_context"
    d["payload"] = {
        "kind": "working_context",
        "schema_version": 1,
        "subject": "session",
        "value": "brief",
    }
    d["no_expiry"] = mode == "no-expiry"
    d["expires_at"] = None if d["no_expiry"] else "2026-09-27T17:00:00.000Z"
    rehash(d)
    api().ProfileRecordV2.model_validate(d)
    d["no_expiry"] = not d["no_expiry"]
    with pytest.raises((ValueError, TypeError)):
        api().ProfileRecordV2.model_validate(d)


def test_claim_exact_utf8_ceiling_and_one_byte_over():
    d = data()
    claim = d["claim"]
    claim.update(support_assessments=[], approval_receipt=None)
    template = copy.deepcopy(claim["evidence_bindings"][0])
    fields = [
        "authority_id",
        "governance_scope_id",
        "source_container_id",
        "source_object_id",
        "source_version_id",
    ]
    claim["evidence_bindings"] = [
        template | {"binding_id": f"b{i}", **{f: "a" * 128 for f in fields}}
        for i in range(8)
    ]
    raw = lambda: json.dumps(
        claim, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    needed = 16384 - len(raw())
    assert needed > 0
    for b in claim["evidence_bindings"]:
        for f in fields:
            if needed <= 0:
                break
            extra = min(384, needed)
            threes, rest = divmod(extra, 3)
            b[f] = (
                "😀" * threes
                + (["", "é", "€"][rest])
                + "a" * (128 - threes - bool(rest))
            )
            needed -= extra
    assert needed == 0 and len(raw()) == 16384
    api().ProfileRecordV2.model_validate(d)
    b = claim["evidence_bindings"][-1]
    f = fields[-1]
    assert "a" in b[f]
    b[f] = b[f].replace("a", "é", 1)
    assert len(raw()) == 16385
    with pytest.raises((ValueError, TypeError)):
        api().ProfileRecordV2.model_validate(d)


def test_record_exact_utf8_ceiling_and_one_byte_over():
    d = data()
    d["claim"].update(
        evidence_bindings=[], support_assessments=[], approval_receipt=None
    )
    d["payload"]["value"] = "v" * 15800
    d["semantic_key"] = {"namespace": "a" * 16384, "subject": "a" * 16384}
    rehash(d)
    raw = lambda: json.dumps(
        d, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    needed = 65536 - len(raw())
    assert 0 < needed < 3 * 16384
    threes, rest = divmod(needed, 3)
    d["semantic_key"]["namespace"] = (
        "😀" * threes + (["", "é", "€"][rest]) + "a" * (16384 - threes - bool(rest))
    )
    assert len(raw()) == 65536
    api().ProfileRecordV2.model_validate(d)
    d["semantic_key"]["subject"] = "é" + "a" * 16383
    assert len(raw()) == 65537
    with pytest.raises((ValueError, TypeError)):
        api().ProfileRecordV2.model_validate(d)


def test_payload_defaults_materialize_before_digest_check():
    d = data()
    d["payload"].pop("schema_version")
    d["payload"].pop("kind")
    model = api().ProfileRecordV2.model_validate(d)
    assert contract().canonical_v2_bytes(model).decode() == fixture()["canonical_utf8"]


@pytest.mark.parametrize("layer", ["claim", "binding", "origin", "payload"])
def test_nested_filtered_model_state_is_rejected(layer):
    model = api().ProfileRecordV2.model_validate(data())
    target = {
        "claim": model.claim,
        "binding": model.claim.evidence_bindings[0],
        "origin": model.claim.support_assessments[0].origin,
        "payload": model.payload,
    }[layer]

    class State(dict):
        def keys(self):
            raise AssertionError("must not invoke raw-state callbacks")

    object.__setattr__(target, "__dict__", State(vars(target)))
    with pytest.raises(ValueError, match="^invalid V2 profile object$"):
        contract().canonical_v2_bytes(model)


def test_unknown_discriminator_cannot_echo_private_input():
    d = data()
    d["controls"]["model_disclosure"] = {"kind": "private-sensitive-marker"}
    with pytest.raises((ValueError, TypeError)) as exc:
        api().ProfileRecordV2.model_validate(d)
    assert "private-sensitive-marker" not in str(exc.value)


@pytest.mark.parametrize(
    "name", ["01-manifest", "02-active-record", "05-create-proposal"]
)
def test_literal_subclasses_and_nested_unknown_keys_cannot_invoke_callbacks(name):
    d = data(name)

    class Text(str):
        def __hash__(self):
            raise AssertionError("subclass hashing callback")

        def __eq__(self, other):
            raise AssertionError("subclass equality callback")

    d["profile_id"] = Text("p1")
    with pytest.raises(ValueError, match="^invalid V2 profile object$"):
        contract().validate_v2_object(d)


@pytest.mark.parametrize("operation", ["archive", "promote"])
def test_pending_noncontent_operations(operation):
    d = data("06-update-proposal") | {"operation": operation, "proposed_record": None}
    assert api().ProfileProposalV2.model_validate(d).operation == operation


@pytest.mark.parametrize(
    "method",
    [{}, [], None, 123, 0.5, True],
    ids=["object", "array", "null", "integer", "float", "boolean"],
)
def test_malformed_attribution_method_has_generic_public_error(method):
    d = data()
    d["claim"]["support_assessments"][0]["origin"]["method"] = method
    with pytest.raises(ValueError, match="^invalid V2 profile object$"):
        contract().validate_v2_object(d)


@pytest.mark.parametrize("entry", ["model", "adapter", "json"])
@pytest.mark.parametrize("value", [0, 1, 0.0, 1.0, "false", "true"])
def test_per_call_strict_override_never_coerces_expiry_bool(entry, value):
    d = data("04-deleted-record")
    d["no_expiry"] = value
    m = api().ProfileRecordV2
    with pytest.raises((ValueError, TypeError)):
        if entry == "model":
            m.model_validate(d, strict=False)
        elif entry == "adapter":
            TypeAdapter(m).validate_python(d, strict=False)
        else:
            m.model_validate_json(json.dumps(d), strict=False)


@pytest.mark.parametrize("entry", ["model", "adapter", "json"])
@pytest.mark.parametrize("policy", ["ignore", "allow"])
@pytest.mark.parametrize(
    "path",
    [
        ("actor_id",),
        ("payload", "actor_id"),
        ("semantic_key", "actor_id"),
        ("claim", "evidence_bindings", 0, "actor_id"),
        ("claim", "temporal_validity", "record_id"),
        ("claim", "temporal_validity", "basis", "record_id"),
        ("claim", "support_assessments", 0, "origin", "purge_generation"),
        ("controls", "model_disclosure", "record_id"),
    ],
    ids=[
        "root",
        "payload",
        "semantic-key",
        "binding",
        "validity",
        "basis",
        "origin",
        "policy",
    ],
)
def test_per_call_extra_override_never_discards_or_retains_wrong_shape(
    entry, policy, path
):
    d = data()
    d["semantic_key"] = {"namespace": "a", "subject": "b"}
    target = d
    for k in path[:-1]:
        target = target[k]
    target[path[-1]] = "private-sensitive-marker"
    m = api().ProfileRecordV2
    with pytest.raises((ValueError, TypeError)):
        if entry == "model":
            m.model_validate(d, extra=policy)
        elif entry == "adapter":
            TypeAdapter(m).validate_python(d, extra=policy)
        else:
            m.model_validate_json(json.dumps(d), extra=policy)


@pytest.mark.parametrize("entry", ["model", "adapter", "json"])
def test_extra_override_cannot_hide_fields_in_defaulted_payload(entry):
    d = data()
    d["payload"].pop("kind")
    d["payload"].pop("schema_version")
    d["payload"]["actor_id"] = "private-sensitive-marker"
    m = api().ProfileRecordV2
    with pytest.raises((ValueError, TypeError)):
        if entry == "model":
            m.model_validate(d, extra="ignore")
        elif entry == "adapter":
            TypeAdapter(m).validate_python(d, extra="ignore")
        else:
            m.model_validate_json(json.dumps(d), extra="ignore")

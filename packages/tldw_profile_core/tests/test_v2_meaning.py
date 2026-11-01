import importlib
import importlib.util
from copy import deepcopy
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError
from tldw_profile_core.canonical import canonical_bytes

CANONICAL = b'{"claim_basis":"direct_user_assertion","kind":"preference","payload":{"kind":"preference","polarity":"like","schema_version":1,"subject":"replies","value":"concise"},"profile_id":"p1","projection":"profile-claim-v2","record_id":"r1","relations":[],"scope_id":"s1","temporal_validity":{"basis":{"kind":"user_reviewed"},"kind":"standing"}}'
FIXED_SHA = "4d91768974a491f84e8ef67d9b8577975e8615c47065cd97a16260a7f4c28593"
CHANGED_SCOPE_SHA = "de7ef8e9bffe5f5854390682357e8f8cbc08c4706dfa3e2c288910a08f7b1382"
START = "2026-09-15T00:00:00Z"
END = "2026-09-16T00:00:00Z"


def api():
    assert importlib.util.find_spec("tldw_profile_core.v2_meaning") is not None, (
        "meaning API missing"
    )
    return importlib.import_module("tldw_profile_core.v2_meaning")


def data():
    return {
        "projection": "profile-claim-v2",
        "profile_id": "p1",
        "record_id": "r1",
        "scope_id": "s1",
        "kind": "preference",
        "payload": {
            "schema_version": 1,
            "kind": "preference",
            "subject": "replies",
            "polarity": "like",
            "value": "concise",
        },
        "claim_basis": "direct_user_assertion",
        "temporal_validity": {"kind": "standing", "basis": {"kind": "user_reviewed"}},
        "relations": [],
    }


def interval(start=START, end=None, basis=None):
    return {
        "kind": "interval",
        "valid_from": start,
        "valid_until": end,
        "basis": basis or {"kind": "user_reviewed"},
    }


EDGE_BASE = {"edge_id": "e1", "target_record_id": "old", "target_version_id": "old-v1"}
EDGE_CASES = [
    EDGE_BASE | {"kind": "correction_of", "effect": {"kind": "all_target_validity"}},
    EDGE_BASE
    | {
        "kind": "correction_of",
        "effect": {"kind": "overlap", "valid_from": START, "valid_until": None},
    },
    EDGE_BASE | {"kind": "change_from", "transition_at": START},
    EDGE_BASE
    | {
        "kind": "supersedes",
        "effect": {"kind": "all_target_validity"},
        "reason_code": "user_replacement",
    },
    EDGE_BASE
    | {
        "kind": "supersedes",
        "effect": {"kind": "replace_from", "replace_from": START},
        "reason_code": "user_replacement",
    },
    EDGE_BASE | {"kind": "workspace_exception_to", "workspace_scope_id": "s1"},
]


def fields_for_edge(edge):
    fields = data() | {"relations": [deepcopy(edge)]}
    if edge["kind"] == "change_from":
        fields["temporal_validity"] = interval()
    return fields


def test_fixed_projection_and_json_round_trip():
    model = api().ClaimMeaningV2
    value = model(**data())
    assert canonical_bytes(value) == CANONICAL
    assert len(CANONICAL) == 337
    assert (
        canonical_bytes(model.model_validate_json(value.model_dump_json())) == CANONICAL
    )


@pytest.mark.parametrize("key", tuple(data()))
def test_every_projection_field_is_required_and_nonnull(key):
    model = api().ClaimMeaningV2
    fields = data()
    del fields[key]
    for invalid in (fields, data() | {key: None}):
        with pytest.raises(ValidationError):
            model.model_validate(invalid)


@pytest.mark.parametrize(
    "validity",
    [
        {"kind": "unknown"},
        {"kind": "standing", "basis": {"kind": "user_reviewed"}},
        interval(),
        interval(None, END),
        interval(START, END, {"kind": "bound_source", "binding_ids": ["b1", "b2"]}),
    ],
)
def test_validity_has_distinct_closed_shapes(validity):
    value = api().ClaimMeaningV2(**(data() | {"temporal_validity": validity}))
    assert value.temporal_validity.kind == validity["kind"]


@pytest.mark.parametrize(
    "validity",
    [
        {"kind": "unknown", "valid_from": None},
        {"kind": "standing"},
        {"kind": "standing", "basis": {"kind": "bound_source", "binding_ids": ["b1"]}},
        interval(None, None),
        interval(START, START),
        interval(END, START),
        interval(basis={"kind": "bound_source", "binding_ids": []}),
        interval(basis={"kind": "bound_source", "binding_ids": ["b1", "b1"]}),
        interval(basis={"kind": "bound_source", "binding_ids": ["b2", "b1"]}),
        interval(
            basis={"kind": "bound_source", "binding_ids": [f"b{i}" for i in range(9)]}
        ),
        interval(basis={"kind": "user_reviewed", "binding_ids": []}),
    ],
)
def test_invalid_validity_rejects(validity):
    with pytest.raises(ValidationError):
        api().ClaimMeaningV2(**(data() | {"temporal_validity": validity}))


@pytest.mark.parametrize(
    "value",
    [
        START.removesuffix("Z"),
        "2026-02-30T00:00:00Z",
        "2026-09-15 00:00:00Z",
        "2026-09-15T00:00:00.0001Z",
        True,
        1,
    ],
)
def test_invalid_time_is_not_inferred(value):
    with pytest.raises(ValidationError):
        api().ClaimMeaningV2(**(data() | {"temporal_validity": interval(value)}))


@pytest.mark.parametrize("edge", EDGE_CASES)
def test_each_relation_has_its_own_effect(edge):
    fields = fields_for_edge(edge)
    value = api().ClaimMeaningV2(**fields)
    assert value.relations[0].kind == edge["kind"]
    assert value.relations[0].target_version_id == "old-v1"


@pytest.mark.parametrize("index", range(len(EDGE_CASES)))
def test_each_edge_rejects_missing_and_extra_fields(index):
    model = api().ClaimMeaningV2
    edge = EDGE_CASES[index]
    for key in edge:
        invalid = deepcopy(edge)
        del invalid[key]
        with pytest.raises(ValidationError):
            model(
                **fields_for_edge(invalid | {"kind": edge["kind"]})
            ) if key != "kind" else model(**(data() | {"relations": [invalid]}))
    with pytest.raises(ValidationError):
        model(**fields_for_edge(edge | {"native": True}))


def test_relation_cross_checks_reject_without_target_lookup():
    bad = [
        data() | {"relations": [EDGE_CASES[2]]},
        data() | {"temporal_validity": interval(END), "relations": [EDGE_CASES[2]]},
        data() | {"relations": [EDGE_CASES[5] | {"workspace_scope_id": "other"}]},
        data() | {"relations": [EDGE_CASES[0], EDGE_CASES[0]]},
        data() | {"relations": [EDGE_CASES[0], EDGE_CASES[3] | {"edge_id": "e2"}]},
        data()
        | {
            "relations": [
                EDGE_CASES[0] | {"edge_id": "e2"},
                EDGE_CASES[0] | {"target_version_id": "v2"},
            ]
        },
        data()
        | {
            "relations": [
                EDGE_CASES[0] | {"edge_id": f"e{i}", "target_record_id": f"r{i}"}
                for i in range(5)
            ]
        },
        data()
        | {
            "temporal_validity": interval(),
            "relations": [
                EDGE_CASES[2],
                EDGE_CASES[2] | {"edge_id": "e2", "target_record_id": "second"},
            ],
        },
        data()
        | {
            "relations": [
                EDGE_BASE
                | {
                    "kind": "correction_of",
                    "effect": {
                        "kind": "overlap",
                        "valid_from": None,
                        "valid_until": None,
                    },
                }
            ]
        },
    ]
    for fields in bad:
        with pytest.raises(ValidationError):
            api().ClaimMeaningV2(**fields)


@pytest.mark.parametrize("field", ["profile_id", "record_id", "scope_id"])
@pytest.mark.parametrize(
    "value", ["", "  ", "x" * 129, "x\x00y", "x\u200dy", "\ud800", True, 42, b"id"]
)
def test_projection_id_bounds_and_types(field, value):
    with pytest.raises(ValidationError):
        api().ClaimMeaningV2(**(data() | {field: value}))


def test_maximum_unicode_id_and_unchanged_payload_defaults():
    fields = data() | {"scope_id": "👋" * 128}
    fields["payload"].pop("schema_version")
    fields["payload"].pop("kind")
    value = api().ClaimMeaningV2(**fields)
    assert value.scope_id == "👋" * 128
    assert value.payload.schema_version == 1 and value.payload.kind == "preference"


@pytest.mark.parametrize(
    "kind",
    [
        "identity",
        "preference",
        "relationship",
        "correction",
        "constraint",
        "goal",
        "convention",
        "working_context",
        "legacy_unclassified",
    ],
)
def test_existing_typed_payloads_remain_v1(kind):
    payload = {"kind": kind, "subject": "synthetic"}
    if kind == "legacy_unclassified":
        payload = {"kind": kind, "text": "synthetic"}
    elif kind == "goal":
        payload["outcome"] = "synthetic"
    else:
        payload["value"] = "synthetic"
        if kind == "preference":
            payload["polarity"] = "like"
    value = api().ClaimMeaningV2(**(data() | {"kind": kind, "payload": payload}))
    assert value.payload.schema_version == 1 and value.payload.kind == kind


def test_kind_mismatch_multibyte_overflow_and_invalid_unicode_reject():
    bad = [
        data() | {"kind": "constraint"},
        data() | {"payload": data()["payload"] | {"value": "é" * 9000}},
        data() | {"payload": data()["payload"] | {"value": "\ud800"}},
    ]
    for fields in bad:
        with pytest.raises(ValidationError):
            api().ClaimMeaningV2(**fields)


def test_legacy_meaning_cannot_acquire_dates_or_relations():
    model = api().ClaimMeaningV2
    valid = data() | {
        "claim_basis": "legacy_unknown",
        "temporal_validity": {"kind": "unknown"},
    }
    assert model(**valid).temporal_validity.kind == "unknown"
    for fields in (
        data() | {"claim_basis": "legacy_unknown"},
        valid | {"relations": [EDGE_CASES[0]]},
    ):
        with pytest.raises(ValidationError):
            model(**fields)


@pytest.mark.parametrize(
    "key",
    [
        "approval_receipt",
        "confidence_estimate",
        "evidence_bindings",
        "model_disclosure",
        "version_id",
        "created_at",
        "native",
        "source_path",
    ],
)
def test_metadata_cannot_enter_meaning_projection(key):
    with pytest.raises(ValidationError):
        api().ClaimMeaningV2(**(data() | {key: "synthetic-private"}))


def test_duplicate_json_members_reject_before_last_value_wins():
    model = api().ClaimMeaningV2
    wire = CANONICAL.decode().replace(
        '"scope_id":"s1"', '"scope_id":"other","scope_id":"s1"'
    )
    nested = CANONICAL.decode().replace(
        '"value":"concise"', '"value":"other","value":"concise"'
    )
    for text in (wire, nested):
        with pytest.raises(ValueError):
            model.model_validate_json(text)


def test_diagnostics_and_frozen_model_do_not_echo_private_values():
    model = api().ClaimMeaningV2
    value = model(
        **(
            data()
            | {
                "profile_id": "synthetic-private-profile",
                "payload": data()["payload"] | {"value": "synthetic-private-body"},
            }
        )
    )
    assert "synthetic-private" not in repr(value)
    with pytest.raises(ValidationError):
        value.scope_id = "other"
    for fields in (
        data() | {"scope_id": "synthetic-private-" + "x" * 129},
        data() | {"temporal_validity": {"kind": "synthetic-private-kind"}},
        data() | {"synthetic-private-key": "value"},
    ):
        with pytest.raises(ValidationError) as error:
            model.model_validate(fields)
        assert "synthetic-private" not in str(error.value)


class CustomText(str):
    def encode(self, *args, **kwargs):
        raise AssertionError("custom encode called")


class CustomInteger(int):
    pass


class CustomDatetime(datetime):
    pass


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param(data() | {"scope_id": CustomText("id")}, id="text-subclass"),
        pytest.param(
            data()
            | {"payload": data()["payload"] | {"schema_version": CustomInteger(1)}},
            id="integer-subclass",
        ),
        pytest.param(
            data()
            | {"temporal_validity": interval(CustomDatetime(2026, 9, 15, tzinfo=UTC))},
            id="datetime-subclass",
        ),
    ],
)
def test_custom_scalars_reject_before_callbacks(fields):
    with pytest.raises(ValidationError):
        api().ClaimMeaningV2(**fields)


def digest_api():
    module = api()
    assert hasattr(module, "claim_meaning_digest"), "meaning digest missing"
    return module


def test_fixed_digest_and_scope_cannot_transplant_meaning():
    module = digest_api()
    original = module.ClaimMeaningV2(**data())
    before = original.model_dump_json()
    assert module.claim_meaning_digest(original) == FIXED_SHA
    assert original.model_dump_json() == before
    assert (
        module.claim_meaning_digest(
            module.ClaimMeaningV2(**(data() | {"scope_id": "s2"}))
        )
        == CHANGED_SCOPE_SHA
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"profile_id": "p2"},
        {"record_id": "r2"},
        {"scope_id": "s2"},
        {
            "kind": "constraint",
            "payload": {"kind": "constraint", "subject": "replies", "value": "concise"},
        },
        {"payload": data()["payload"] | {"value": "long"}},
        {"claim_basis": "inference"},
        {"claim_basis": "imported_assertion"},
        {"claim_basis": "legacy_unknown", "temporal_validity": {"kind": "unknown"}},
        {"temporal_validity": {"kind": "unknown"}},
        {"temporal_validity": interval()},
        {"relations": [EDGE_CASES[0]]},
    ],
)
def test_every_included_nonliteral_field_changes_digest(updates):
    module = digest_api()
    assert (
        module.claim_meaning_digest(module.ClaimMeaningV2(**(data() | updates)))
        != FIXED_SHA
    )


def test_unicode_is_exact_and_equivalent_timezones_are_canonical():
    module = digest_api()

    def compute(fields):
        return module.claim_meaning_digest(module.ClaimMeaningV2(**fields))

    assert compute(data() | {"payload": data()["payload"] | {"value": "é"}}) != compute(
        data() | {"payload": data()["payload"] | {"value": "e\u0301"}}
    )
    utc = data() | {"temporal_validity": interval()}
    offset = data() | {"temporal_validity": interval("2026-09-15T02:00:00+02:00")}
    assert compute(utc) == compute(offset)


def test_digest_rejects_wrong_exact_root_without_running_serializer():
    module = digest_api()

    class Derived(module.ClaimMeaningV2):
        def model_dump(self, *args, **kwargs):
            raise AssertionError("untrusted serializer called")

    for value in (data(), Derived(**data()), None, object()):
        with pytest.raises(TypeError):
            module.claim_meaning_digest(value)


@pytest.mark.parametrize(
    "updates",
    [
        {"scope_id": True},
        {"payload": None},
        {"unexpected": "synthetic-private"},
        {"projection": "synthetic-private-tag"},
        {"kind": "synthetic-private-kind"},
        {"temporal_validity": {"kind": "synthetic-private-kind"}},
        {"payload": data()["payload"] | {"value": "é" * 9000}},
        {"relations": [EDGE_CASES[5] | {"workspace_scope_id": "other"}]},
    ],
)
def test_unsafe_copy_requires_fresh_complete_validation(updates):
    module = digest_api()
    original = module.ClaimMeaningV2(**data())
    unsafe = original.model_copy(update=updates)
    with pytest.raises((ValidationError, ValueError)) as error:
        module.claim_meaning_digest(unsafe)
    assert "synthetic-private" not in str(error.value)
    assert module.claim_meaning_digest(original) == FIXED_SHA


def test_construct_and_direct_attribute_poisoning_do_not_bypass_validation():
    module = digest_api()
    fields = data()
    del fields["scope_id"]
    missing = module.ClaimMeaningV2.model_construct(**fields)
    with pytest.raises((ValidationError, ValueError)):
        module.claim_meaning_digest(missing)
    value = module.ClaimMeaningV2(**data())
    object.__setattr__(value, "scope_id", False)
    with pytest.raises((ValidationError, ValueError)):
        module.claim_meaning_digest(value)


@pytest.mark.parametrize(
    "location", ["root", "payload", "validity", "basis", "edge", "effect"]
)
def test_nonnull_pydantic_extra_state_is_not_discarded(location):
    module = digest_api()
    value = module.ClaimMeaningV2(**(data() | {"relations": [EDGE_CASES[0]]}))
    targets = {
        "root": value,
        "payload": value.payload,
        "validity": value.temporal_validity,
        "basis": value.temporal_validity.basis,
        "edge": value.relations[0],
        "effect": value.relations[0].effect,
    }
    object.__setattr__(
        targets[location],
        "__pydantic_extra__",
        {"synthetic-private-key": "synthetic-private-body"},
    )
    with pytest.raises(ValueError) as error:
        module.claim_meaning_digest(value)
    assert "synthetic-private" not in str(error.value)


def test_nested_v1_payload_copy_is_revalidated_and_never_filtered():
    module = digest_api()
    value = module.ClaimMeaningV2(**data())
    for update in (
        {"schema_version": True},
        {"subject": " "},
        {"synthetic-private-key": "body"},
    ):
        with pytest.raises((ValidationError, ValueError)):
            module.claim_meaning_digest(
                value.model_copy(
                    update={"payload": value.payload.model_copy(update=update)}
                )
            )


def test_custom_objects_reject_before_methods_and_container_traversal():
    module = digest_api()
    calls = []

    class PoisonText(str):
        def encode(self, *args, **kwargs):
            calls.append("encode")
            raise AssertionError("custom encode called")

    class PoisonDict(dict):
        def items(self):
            calls.append("items")
            raise AssertionError("custom items called")

        def __iter__(self):
            calls.append("dict-iter")
            raise AssertionError("custom iteration called")

    class PoisonList(list):
        def __iter__(self):
            calls.append("list-iter")
            raise AssertionError("custom iteration called")

    class PoisonPayload(type(module.ClaimMeaningV2(**data()).payload)):
        def model_dump(self, *args, **kwargs):
            calls.append("dump")
            raise AssertionError("custom dump called")

    original = module.ClaimMeaningV2(**data())
    updates = [
        {"scope_id": PoisonText("s1")},
        {"payload": PoisonDict(data()["payload"])},
        {"relations": PoisonList([])},
        {"payload": PoisonPayload(**data()["payload"])},
        {
            "payload": original.payload.model_copy(
                update={"schema_version": CustomInteger(1)}
            )
        },
        {"temporal_validity": interval(CustomDatetime(2026, 9, 15, tzinfo=UTC))},
    ]
    for update in updates:
        with pytest.raises((ValidationError, ValueError)):
            module.claim_meaning_digest(original.model_copy(update=update))
    assert calls == []


def test_invalid_recursive_input_is_bounded_without_recursion_error():
    module = digest_api()
    recursive = {"kind": "standing"}
    recursive["basis"] = recursive
    with pytest.raises(ValueError, match="structural bounds"):
        module.claim_meaning_digest(
            module.ClaimMeaningV2.model_construct(
                **(data() | {"temporal_validity": recursive})
            )
        )


def test_maximum_typed_nested_projection_can_be_hashed():
    module = digest_api()
    fields = data() | {
        "temporal_validity": interval(
            START,
            END,
            {"kind": "bound_source", "binding_ids": [f"b{i}" for i in range(8)]},
        ),
        "relations": [
            EDGE_CASES[1] | {"edge_id": f"e{i}", "target_record_id": f"r{i}"}
            for i in range(4)
        ],
    }
    value = module.ClaimMeaningV2(**fields)
    assert len(module.claim_meaning_digest(value)) == 64
    assert canonical_bytes(value) == canonical_bytes(
        module.ClaimMeaningV2.model_validate_json(value.model_dump_json())
    )


def test_global_field_names_cannot_migrate_to_wrong_nested_shape():
    module = digest_api()
    value = module.ClaimMeaningV2(**data())
    object.__setattr__(value.payload, "scope_id", "synthetic-private")
    with pytest.raises((ValidationError, ValueError)) as error:
        module.claim_meaning_digest(value)
    assert "synthetic-private" not in str(error.value)


@pytest.mark.parametrize("entry", ["model", "digest"])
def test_custom_timezone_callbacks_are_not_run(entry):
    module = digest_api()
    calls = []

    class PoisonTimezone(tzinfo):
        def utcoffset(self, value):
            calls.append("utcoffset")
            raise AssertionError("custom timezone callback called")

    fields = data() | {
        "temporal_validity": interval(datetime(2026, 9, 15, tzinfo=PoisonTimezone()))
    }
    with pytest.raises((ValidationError, ValueError)):
        if entry == "model":
            module.ClaimMeaningV2(**fields)
        else:
            module.claim_meaning_digest(module.ClaimMeaningV2.model_construct(**fields))
    assert calls == []


@pytest.mark.parametrize(
    "zone",
    [UTC, timezone(timedelta(hours=2)), ZoneInfo("UTC")],
    ids=["utc", "fixed-offset", "stdlib-zone"],
)
def test_builtin_timezone_values_keep_portable_semantics(zone):
    module = digest_api()
    value = module.ClaimMeaningV2(
        **(data() | {"temporal_validity": interval(datetime(2026, 9, 15, tzinfo=zone))})
    )
    assert len(module.claim_meaning_digest(value)) == 64


@pytest.mark.parametrize("entry", ["model", "digest", "json"])
def test_custom_metaclass_hash_and_equality_are_not_run(entry):
    module = digest_api()
    calls = []

    class PoisonMeta(type):
        def __hash__(cls):
            calls.append("hash")
            raise AssertionError("custom metaclass hash called")

        def __eq__(cls, other):
            calls.append("eq")
            raise AssertionError("custom metaclass equality called")

    class PoisonValue(metaclass=PoisonMeta):
        pass

    with pytest.raises((TypeError, ValueError, ValidationError)):
        if entry == "json":
            module.ClaimMeaningV2.model_validate_json(PoisonValue())
        elif entry == "model":
            module.ClaimMeaningV2(**(data() | {"scope_id": PoisonValue()}))
        else:
            module.claim_meaning_digest(
                module.ClaimMeaningV2.model_construct(
                    **(data() | {"scope_id": PoisonValue()})
                )
            )
    assert calls == []


@pytest.mark.parametrize(
    "location", ["root", "payload", "validity", "basis", "edge", "effect"]
)
def test_model_state_dictionary_cannot_hide_fields_or_invoke_callbacks(location):
    module = digest_api()
    calls = []

    class HiddenDict(dict):
        def __iter__(self):
            calls.append("iter")
            return super().__iter__()

        def keys(self):
            calls.append("keys")
            return [key for key in dict.__iter__(self) if key != "private_unknown"]

        def __getitem__(self, key):
            calls.append("getitem")
            return super().__getitem__(key)

    value = module.ClaimMeaningV2(**(data() | {"relations": [EDGE_CASES[0]]}))
    targets = {
        "root": value,
        "payload": value.payload,
        "validity": value.temporal_validity,
        "basis": value.temporal_validity.basis,
        "edge": value.relations[0],
        "effect": value.relations[0].effect,
    }
    state = HiddenDict(vars(targets[location]))
    state["private_unknown"] = "synthetic-private-body"
    object.__setattr__(targets[location], "__dict__", state)
    with pytest.raises((ValueError, ValidationError)) as error:
        module.claim_meaning_digest(value)
    assert calls == []
    assert "synthetic-private" not in str(error.value)
    assert dict.__contains__(state, "private_unknown")


@pytest.mark.parametrize("entry", ["adapter", "containing-model"])
@pytest.mark.parametrize("duplicate", [False, True])
def test_alternate_pydantic_json_entrypoints_fail_closed(entry, duplicate):
    module = digest_api()
    wire = CANONICAL.decode()
    if duplicate:
        wire = wire.replace('"scope_id":"s1"', '"scope_id":"foreign","scope_id":"s1"')
    if entry == "adapter":
        validate = TypeAdapter(module.ClaimMeaningV2).validate_json
    else:

        class Envelope(BaseModel):
            meaning: module.ClaimMeaningV2

        validate = Envelope.model_validate_json
        wire = '{"meaning":' + wire + "}"
    with pytest.raises(TypeError, match="duplicate-aware"):
        validate(wire)


def test_dedicated_json_parser_and_python_adapters_remain_supported():
    module = digest_api()
    adapter = TypeAdapter(module.ClaimMeaningV2)
    assert module.claim_meaning_digest(adapter.validate_python(data())) == FIXED_SHA
    assert (
        module.claim_meaning_digest(
            module.ClaimMeaningV2.model_validate_json(CANONICAL)
        )
        == FIXED_SHA
    )
    assert (
        canonical_bytes(
            module.UnknownValidityV2.model_validate_json('{"kind":"unknown"}')
        )
        == b'{"kind":"unknown"}'
    )


def test_rejected_compiled_json_does_not_echo_input_through_default_wrapper():
    module = digest_api()

    class Envelope(BaseModel):
        meaning: module.ClaimMeaningV2

    with pytest.raises((TypeError, ValidationError)) as error:
        Envelope.model_validate_json('{"meaning":{"payload":{"value":"PRIVATE"}}}')
    assert "PRIVATE" not in str(error.value)

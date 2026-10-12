"""Native ingress must retain exact fields and use explicit version validation."""

import json
from datetime import UTC, datetime

import pytest
from tldw_profile_core import ProfileManifest, ProfileScope, ScopeKind, canonical_bytes

from Tests.Personal_Context.native_barrier_helpers import CORE_ROOT, v2_fixture


@pytest.fixture
def v1_objects(record_factory, proposal_factory):
    now = datetime(2026, 8, 29, 12, tzinfo=UTC)
    return {
        "manifest": ProfileManifest(
            profile_id="native-profile",
            revision=0,
            purge_generation=0,
            created_at=now,
            updated_at=now,
            current_version_id="m1",
        ),
        "scope": ProfileScope(
            profile_id="native-profile",
            scope_id="scope-global",
            kind=ScopeKind.GLOBAL,
            version_id="s1",
            created_at=now,
            updated_at=now,
        ),
        "record": record_factory("native-profile"),
        "proposal": proposal_factory("native-profile"),
    }


@pytest.mark.parametrize("kind", ["manifest", "scope", "record", "proposal"])
def test_v1_native_canonical_bytes_are_unchanged(kind, v1_objects):
    from tldw_chatbook.Personal_Context.native_codec import (
        decode_native_profile,
        native_v1_bytes,
    )

    original = v1_objects[kind]
    fixed = canonical_bytes(original)
    decoded = decode_native_profile(kind, fixed)
    assert decoded.schema_version == 1
    assert decoded.value == original
    assert decoded.canonical == fixed == native_v1_bytes(kind, original)
    assert native_v1_bytes(kind, json.loads(fixed)) == fixed
    body = json.loads(fixed)
    del body["schema_version"]
    assert decode_native_profile(kind, json.dumps(body)).canonical == fixed
    body["schema_version"] = 1.0
    assert decode_native_profile(kind, json.dumps(body)).canonical == fixed


@pytest.mark.parametrize("path", sorted((CORE_ROOT / "fixtures/v2").glob("*.json")))
def test_explicit_v2_fixed_vectors(path):
    from tldw_chatbook.Personal_Context.native_codec import (
        NativeProfileDecodeError,
        decode_native_profile,
    )

    fixed = json.loads(path.read_text())
    kind = (
        "manifest"
        if "profile_id" in fixed["data"] and "current_version_id" in fixed["data"]
        else "proposal"
        if "proposal_id" in fixed["data"]
        else "record"
    )
    raw = json.dumps(fixed["data"])
    if not fixed["valid"]:
        with pytest.raises(
            NativeProfileDecodeError, match="^personal_context_payload_invalid$"
        ):
            decode_native_profile(kind, raw)
        return
    decoded = decode_native_profile(kind, raw)
    assert decoded.schema_version == 2
    assert decoded.canonical == fixed["canonical_utf8"].encode()
    assert fixed["data"]["profile_id"] not in repr(decoded)
    fixed["data"]["schema_version"] = 2.0
    assert (
        decode_native_profile(kind, json.dumps(fixed["data"])).canonical
        == decoded.canonical
    )


@pytest.mark.parametrize(
    "version", [True, False, "1", "2", 0, 3, float("nan"), float("inf")]
)
def test_schema_coercion_and_unknown_versions_deny(version, v1_objects):
    from tldw_chatbook.Personal_Context.native_codec import (
        NativeProfileDecodeError,
        decode_native_profile,
    )

    body = json.loads(canonical_bytes(v1_objects["manifest"]))
    body["schema_version"] = version
    with pytest.raises(
        NativeProfileDecodeError, match="^personal_context_payload_invalid$"
    ):
        decode_native_profile("manifest", json.dumps(body))


@pytest.mark.parametrize(
    "raw",
    [
        b"\xff",
        "{",
        "[]",
        '{"schema_version":1,"schema_version":2}',
        '{"x":{"canary":1,"canary":2}}',
        '{"x":NaN}',
        '{"x":1e999}',
        '{"x":' + "[" * 25 + "1" + "]" * 25 + "}",
        " " * 262145,
        json.dumps({"x": [None] * 4097}),
    ],
)
def test_malformed_duplicate_and_bounded_input_deny(raw):
    from tldw_chatbook.Personal_Context.native_codec import (
        NativeProfileDecodeError,
        decode_native_profile,
    )

    with pytest.raises(NativeProfileDecodeError) as error:
        decode_native_profile("manifest", raw)
    assert str(error.value) == "personal_context_payload_invalid"
    assert "canary" not in repr(error.value)


@pytest.mark.parametrize("kind", ["scope", "record", "proposal", "unknown"])
def test_wrong_v2_kind_never_downgrades(kind):
    from tldw_chatbook.Personal_Context.native_codec import (
        NativeProfileDecodeError,
        decode_native_profile,
    )

    with pytest.raises(NativeProfileDecodeError):
        decode_native_profile(kind, json.dumps(v2_fixture("01-manifest")["data"]))


def test_v2_cannot_enter_native_v1_serializer():
    from tldw_profile_core.v2_contract import validate_v2_object

    from tldw_chatbook.Personal_Context.native_codec import (
        NativeProfileDecodeError,
        native_v1_bytes,
    )

    body = v2_fixture("01-manifest")["data"]
    for incoming in (body, validate_v2_object(body)):
        with pytest.raises(NativeProfileDecodeError):
            native_v1_bytes("manifest", incoming)
    del body["schema_version"]
    with pytest.raises(NativeProfileDecodeError):
        native_v1_bytes("manifest", body)


@pytest.mark.parametrize(
    "location", ["root", "payload", "controls", "provenance", "semantic_key"]
)
def test_constructed_unknown_fields_cannot_disappear(location, v1_objects):
    from tldw_chatbook.Personal_Context.native_codec import (
        NativeProfileDecodeError,
        native_v1_bytes,
    )

    record = v1_objects["record"]
    value = record if location == "root" else getattr(record, location)
    fields = dict(value.__dict__)
    fields["v2_canary"] = "private-canary"
    tampered = type(value).model_construct(**fields)
    # extra=forbid model_construct drops extras; bypass it to reproduce stored state.
    object.__setattr__(tampered, "__dict__", fields)
    if location != "root":
        tampered = record.model_copy(update={location: tampered})
    with pytest.raises(NativeProfileDecodeError):
        native_v1_bytes("record", tampered)


@pytest.mark.parametrize("bad", [0, 1, "false", "true"])
def test_no_expiry_never_coerces(bad, v1_objects):
    from tldw_chatbook.Personal_Context.native_codec import (
        NativeProfileDecodeError,
        native_v1_bytes,
    )

    body = json.loads(canonical_bytes(v1_objects["record"]))
    body["no_expiry"] = bad
    with pytest.raises(NativeProfileDecodeError):
        native_v1_bytes("record", body)


def test_custom_models_containers_and_raw_state_never_run_callbacks(v1_objects):
    from tldw_chatbook.Personal_Context.native_codec import (
        NativeProfileDecodeError,
        decode_native_profile,
        native_v1_bytes,
    )

    calls = []

    class HostileDict(dict):
        def items(self):
            calls.append("items")
            raise AssertionError("callback")

        def keys(self):
            calls.append("keys")
            return []

    class HostileManifest(ProfileManifest):
        def model_dump(self, **kwargs):
            calls.append("dump")
            return {}

    original = v1_objects["manifest"]
    incoming = [
        HostileDict(json.loads(canonical_bytes(original))),
        HostileManifest.model_construct(**original.__dict__),
    ]
    tampered = original.model_copy()
    object.__setattr__(tampered, "__dict__", HostileDict(original.__dict__))
    incoming.append(tampered)
    for value in incoming:
        with pytest.raises(NativeProfileDecodeError):
            native_v1_bytes("manifest", value)

    class HostileStr(str):
        def encode(self, *args, **kwargs):
            calls.append("encode")
            raise AssertionError("callback")

    with pytest.raises(NativeProfileDecodeError):
        decode_native_profile(
            "manifest", HostileStr(canonical_bytes(original).decode())
        )
    assert calls == []


def test_maximum_valid_v1_unicode_key_and_provenance_fit_native_limit(v1_objects):
    from tldw_profile_core import ProfileRecord

    from tldw_chatbook.Personal_Context.native_codec import (
        decode_native_profile,
        native_v1_bytes,
    )

    body = json.loads(canonical_bytes(v1_objects["record"]))
    body["semantic_key"] = {"namespace": "🦉" * 16384, "subject": "🦉" * 16384}
    body["provenance"]["source_references"] = ["🦉" * 128] * 32
    body["provenance"]["source_hashes"] = ["a" * 64] * 32
    valid = ProfileRecord.model_validate(body)
    fixed = canonical_bytes(valid)
    assert 131072 < len(fixed) < 262144
    assert native_v1_bytes("record", valid) == fixed
    assert decode_native_profile("record", fixed).canonical == fixed


def test_native_detachment_rejects_nested_type_and_datetime_callbacks(v1_objects):
    from datetime import timedelta, tzinfo

    from tldw_chatbook.Personal_Context.native_codec import (
        NativeProfileDecodeError,
        native_v1_bytes,
    )

    calls = []

    class CallbackMeta(type):
        def __hash__(cls):
            calls.append("class-hash")
            return id(cls)

        def __eq__(cls, other):
            calls.append("class-equality")
            return False

    class CallbackValue(metaclass=CallbackMeta):
        pass

    class CallbackZone(tzinfo):
        def utcoffset(self, dt):
            calls.append("offset")
            return timedelta(0)

        def dst(self, dt):
            return timedelta(0)

    record = v1_objects["record"].model_copy(update={"payload": CallbackValue()})
    with pytest.raises(NativeProfileDecodeError):
        native_v1_bytes("record", record)
    assert calls == []
    timestamp = datetime(2026, 8, 29, 12, tzinfo=CallbackZone())
    manifest = v1_objects["manifest"].model_copy(
        update={"created_at": timestamp, "updated_at": timestamp}
    )
    with pytest.raises(NativeProfileDecodeError):
        native_v1_bytes("manifest", manifest)
    assert calls == []


def test_missing_v2_library_component_is_generic_and_v1_still_works(
    monkeypatch, v1_objects
):
    import sys

    from tldw_chatbook.Personal_Context.native_codec import (
        NativeProfileDecodeError,
        decode_native_profile,
    )

    monkeypatch.setitem(sys.modules, "tldw_profile_core.v2_contract", None)
    assert (
        decode_native_profile(
            "manifest", canonical_bytes(v1_objects["manifest"])
        ).schema_version
        == 1
    )
    with pytest.raises(
        NativeProfileDecodeError, match="^personal_context_payload_invalid$"
    ):
        decode_native_profile("manifest", json.dumps(v2_fixture("01-manifest")["data"]))


def test_raw_type_rejection_never_invokes_metaclass_equality():
    from tldw_chatbook.Personal_Context.native_codec import (
        NativeProfileDecodeError,
        decode_native_profile,
    )

    calls = []

    class CallbackMeta(type):
        def __eq__(cls, other):
            calls.append("equality")
            raise AssertionError("raw-type-canary")

    class CallbackRaw(metaclass=CallbackMeta):
        pass

    with pytest.raises(NativeProfileDecodeError):
        decode_native_profile("manifest", CallbackRaw())
    assert calls == []

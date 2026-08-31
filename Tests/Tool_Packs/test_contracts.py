from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json

import pytest

from tldw_chatbook.MCP.hub_tool_catalog import HubTool
from tldw_chatbook.MCP.permission_store import definition_hash
from tldw_chatbook.Tool_Packs.contracts import (
    MAX_JSON_DEPTH,
    MAX_JSON_NODES,
    MAX_JSON_STRING_BYTES,
    MAX_PROFILE_BYTES,
    PortableFallback,
    PortableToolRule,
    ToolPackDocument,
    ToolPackError,
    ToolPackManifest,
    ToolProfilePayload,
    canonical_json_bytes,
    portable_contract_sha256,
    strict_json_object,
    validate_tool_pack_document,
    validate_tool_pack_manifest,
    validate_tool_profile_payload,
)


_ZERO_HASH = "0" * 64
_OMIT = object()


def _fallback(
    *, authority: str = "mcp", server_key: str = "*", state: str = "ask"
) -> dict[str, object]:
    return {"authority": authority, "server_key": server_key, "state": state}


def _rule(
    *,
    authority: str = "mcp",
    server_key: str = "local:docs",
    tool_name: str = "search",
    state: str = "allow",
    contract_sha256: object = _ZERO_HASH,
) -> dict[str, object]:
    result: dict[str, object] = {
        "authority": authority,
        "server_key": server_key,
        "tool_name": tool_name,
        "state": state,
    }
    if contract_sha256 is not _OMIT:
        result["contract_sha256"] = contract_sha256
    return result


def _profile(*, fallbacks: list[dict[str, object]] | None = None, tools=None):
    return {
        "schema": "tldw.tool-profile/v1",
        "fallbacks": fallbacks
        if fallbacks is not None
        else [
            _fallback(authority="builtin", server_key="agent:builtin"),
            _fallback(),
            _fallback(server_key="local:docs"),
        ],
        "tools": [_rule()] if tools is None else tools,
    }


def _manifest(profile_bytes: bytes) -> dict[str, object]:
    body: dict[str, object] = {
        "schema": "tldw.tool-pack/v1",
        "producer": {"name": "tldw_chatbook", "version": "1.0.0"},
        "required_features": [],
        "profile": {
            "suggested_id": "research-tools",
            "display_name": "Research tools",
            "payload": "profile/profile.json",
        },
        "files": [
            {
                "path": "profile/profile.json",
                "size": len(profile_bytes),
                "sha256": hashlib.sha256(profile_bytes).hexdigest(),
            }
        ],
    }
    preimage = (
        b"tldw.tool-pack/v1\0"
        + canonical_json_bytes(body)
        + b"\0"
        + profile_bytes
    )
    body["content_digest"] = hashlib.sha256(preimage).hexdigest()
    return body


def _hub(**changes: object) -> HubTool:
    values = {
        "server_key": "local:docs",
        "server_label": "Docs",
        "source": "local",
        "name": "search",
        "description": "Search\r\ndocuments.",
        "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
        "tags": ("network", "mutates"),
        "stale": False,
        "executable": True,
    }
    values.update(changes)
    return HubTool(**values)  # type: ignore[arg-type]


def test_tool_pack_error_has_only_the_stable_public_code() -> None:
    error = ToolPackError("import", "payload_invalid")

    assert error.operation == "import"
    assert error.category == "payload_invalid"
    assert str(error) == "tool_pack.import.payload_invalid"


@pytest.mark.parametrize(
    ("operation", "category", "expected"),
    [
        ("import", "schema_unsupported", "tool_pack.import.schema_unsupported"),
        ("import", "not_a_category", "tool_pack.import.payload_invalid"),
        ("export", "schema_unsupported", "tool_pack.export.profile_invalid"),
        ("export", "too_large", "tool_pack.export.too_large"),
        ("bind", "payload_invalid", "tool_pack.bind.confirmation_invalid"),
        ("remove", "payload_invalid", "tool_pack.remove.non_removable"),
        ("unknown", "payload_invalid", "tool_pack.import.payload_invalid"),
    ],
)
def test_tool_pack_error_never_constructs_a_pair_outside_the_stable_table(
    operation: str, category: str, expected: str
) -> None:
    assert str(ToolPackError(operation, category)) == expected


@pytest.mark.parametrize(
    "raw",
    [
        b'{"a":1,"a":2}',
        b'{"a":NaN}',
        b'{"a":Infinity}',
        b'{"a":-Infinity}',
        b"\xff",
        '{"a":"\\ud800"}'.encode(),
        b"[]",
    ],
)
def test_strict_json_rejects_noncanonical_inputs_with_supplied_category(raw: bytes) -> None:
    with pytest.raises(ToolPackError) as caught:
        strict_json_object(raw, category="payload_invalid", max_bytes=1024)

    assert caught.value.operation == "import"
    assert caught.value.category == "payload_invalid"
    assert str(caught.value) == "tool_pack.import.payload_invalid"


def test_strict_json_rejects_bytes_before_decoding() -> None:
    with pytest.raises(ToolPackError, match=r"^tool_pack\.import\.too_large$"):
        strict_json_object(b'{"a":1}', category="payload_invalid", max_bytes=6)


def test_strict_json_accepts_the_exact_byte_node_and_depth_boundaries() -> None:
    raw = b'{"a":1}'
    assert strict_json_object(raw, category="payload_invalid", max_bytes=len(raw)) == {
        "a": 1
    }

    exact_depth: object = None
    for _ in range(MAX_JSON_DEPTH - 2):
        exact_depth = [exact_depth]
    depth_raw = json.dumps({"root": exact_depth}).encode()
    assert strict_json_object(
        depth_raw, category="payload_invalid", max_bytes=len(depth_raw)
    )["root"] is not None

    exact_nodes = {str(index): None for index in range(MAX_JSON_NODES - 1)}
    node_raw = json.dumps(exact_nodes).encode()
    assert len(
        strict_json_object(
            node_raw, category="payload_invalid", max_bytes=len(node_raw)
        )
    ) == MAX_JSON_NODES - 1


def test_strict_json_rejects_one_byte_over_the_string_limit() -> None:
    raw = b'{"value":"' + b"x" * (MAX_JSON_STRING_BYTES + 1) + b'"}'

    with pytest.raises(ToolPackError) as caught:
        strict_json_object(raw, category="payload_invalid", max_bytes=len(raw))
    assert caught.value.category == "payload_invalid"


def test_strict_json_rejects_depth_and_node_boundary_overages() -> None:
    deep: object = 0
    for _ in range(MAX_JSON_DEPTH + 1):
        deep = [deep]
    many = {str(index): None for index in range(MAX_JSON_NODES)}

    for raw in (json.dumps({"root": deep}).encode(), json.dumps(many).encode()):
        with pytest.raises(ToolPackError) as caught:
            strict_json_object(raw, category="payload_invalid", max_bytes=len(raw))
        assert caught.value.category == "payload_invalid"


def test_strict_json_preserves_distinct_case_sensitive_schema_keys() -> None:
    assert strict_json_object(
        b'{"properties":{"Name":{},"name":{}}}',
        category="payload_invalid",
        max_bytes=1024,
    ) == {"properties": {"Name": {}, "name": {}}}


def test_canonical_json_normalizes_strings_keys_and_line_termination() -> None:
    assert canonical_json_bytes({"z": "e\u0301", "a": [True, None, 2]}) == (
        '{"a":[true,null,2],"z":"é"}\n'.encode()
    )


@pytest.mark.parametrize(
    "value",
    [
        {"a": float("nan")},
        {"a": float("inf")},
        {"a": object()},
        {1: "integer key"},
        ("tuple",),
        {"é": 1, "e\u0301": 2},
    ],
)
def test_canonical_json_rejects_unsupported_or_normalized_collision_values(value) -> None:
    with pytest.raises(ToolPackError, match=r"^tool_pack\.import\.payload_invalid$"):
        canonical_json_bytes(value)


def test_canonical_json_rejects_a_recursive_container() -> None:
    recursive: list[object] = []
    recursive.append(recursive)

    with pytest.raises(ToolPackError, match=r"^tool_pack\.import\.payload_invalid$"):
        canonical_json_bytes(recursive)


def test_canonical_json_enforces_exact_and_one_over_depth() -> None:
    exact: object = None
    for _ in range(MAX_JSON_DEPTH - 1):
        exact = [exact]
    assert canonical_json_bytes(exact).endswith(b"\n")

    one_over = [exact]
    with pytest.raises(ToolPackError, match=r"^tool_pack\.import\.payload_invalid$"):
        canonical_json_bytes(one_over)


def test_canonical_json_enforces_exact_and_one_over_node_count() -> None:
    exact = [None] * (MAX_JSON_NODES - 1)
    assert canonical_json_bytes(exact).startswith(b"[")

    one_over = exact + [None]
    with pytest.raises(ToolPackError, match=r"^tool_pack\.import\.payload_invalid$"):
        canonical_json_bytes(one_over)


def test_canonical_json_enforces_exact_and_one_over_string_bytes() -> None:
    assert canonical_json_bytes("x" * MAX_JSON_STRING_BYTES).endswith(b'"\n')

    with pytest.raises(ToolPackError, match=r"^tool_pack\.import\.payload_invalid$"):
        canonical_json_bytes("x" * (MAX_JSON_STRING_BYTES + 1))


def test_dataclass_instances_are_immutable() -> None:
    fallback = PortableFallback.from_dict(_fallback())

    with pytest.raises(FrozenInstanceError):
        fallback.state = "deny"  # type: ignore[misc]


@pytest.mark.parametrize("kind", ["fallback", "rule", "profile", "manifest"])
@pytest.mark.parametrize("mutation", ["missing", "unknown"])
def test_contract_objects_reject_missing_and_unknown_fields(kind: str, mutation: str) -> None:
    profile_bytes = canonical_json_bytes(_profile())
    values = {
        "fallback": _fallback(),
        "rule": _rule(),
        "profile": _profile(),
        "manifest": _manifest(profile_bytes),
    }
    factories = {
        "fallback": PortableFallback.from_dict,
        "rule": PortableToolRule.from_dict,
        "profile": ToolProfilePayload.from_dict,
        "manifest": ToolPackManifest.from_dict,
    }
    raw = values[kind]
    if mutation == "missing":
        raw.pop(next(iter(raw)))
    else:
        raw["unexpected"] = True

    with pytest.raises(ToolPackError):
        factories[kind](raw)


@pytest.mark.parametrize(
    "suggested_id",
    ["", "Upper", "-leading", "default", "ws-private", "x" * 129, "é"],
)
def test_manifest_rejects_invalid_or_reserved_suggested_ids(suggested_id: str) -> None:
    profile_bytes = canonical_json_bytes(_profile())
    manifest = _manifest(profile_bytes)
    manifest["profile"]["suggested_id"] = suggested_id  # type: ignore[index]

    with pytest.raises(ToolPackError) as caught:
        ToolPackManifest.from_dict(manifest)
    assert caught.value.category == "manifest_invalid"


def test_manifest_rejects_non_nfc_and_overlong_identity_strings() -> None:
    profile_bytes = canonical_json_bytes(_profile())
    for field, value in (("name", "e\u0301"), ("version", "x" * 129)):
        manifest = _manifest(profile_bytes)
        manifest["producer"][field] = value  # type: ignore[index]
        with pytest.raises(ToolPackError):
            ToolPackManifest.from_dict(manifest)


@pytest.mark.parametrize("state", ["ALLOW", "prompt", "", True, 1])
def test_rule_rejects_invalid_or_non_string_states(state: object) -> None:
    with pytest.raises(ToolPackError):
        PortableToolRule.from_dict(_rule(state=state))  # type: ignore[arg-type]


def test_rule_requires_hash_for_allow_and_ask() -> None:
    for state in ("allow", "ask"):
        for contract_sha256 in (_OMIT, None):
            with pytest.raises(ToolPackError):
                PortableToolRule.from_dict(
                    _rule(state=state, contract_sha256=contract_sha256)
                )


def test_deny_rule_accepts_an_omitted_fingerprint_and_omits_it_on_round_trip() -> None:
    raw = _rule(state="deny", contract_sha256=_OMIT)
    deny = PortableToolRule.from_dict(raw)

    assert deny.contract_sha256 is None
    assert deny.to_dict() == raw
    assert "contract_sha256" not in deny.to_dict()


def test_deny_rule_rejects_null_but_accepts_a_valid_present_fingerprint() -> None:
    with pytest.raises(ToolPackError):
        PortableToolRule.from_dict(_rule(state="deny", contract_sha256=None))

    deny = PortableToolRule.from_dict(_rule(state="deny"))
    assert deny.to_dict()["contract_sha256"] == _ZERO_HASH


def test_fallback_rejects_allow_and_authority_server_mismatch() -> None:
    bad = [
        _fallback(state="allow"),
        _fallback(authority="builtin", server_key="local:docs"),
        _fallback(authority="mcp", server_key="agent:builtin"),
    ]

    for raw in bad:
        with pytest.raises(ToolPackError):
            PortableFallback.from_dict(raw)


def test_profile_requires_sorted_complete_fallbacks_and_sorted_tools() -> None:
    unsorted_fallbacks = [
        _fallback(),
        _fallback(authority="builtin", server_key="agent:builtin"),
        _fallback(server_key="local:docs"),
    ]
    unsorted_tools = [
        _rule(tool_name="z"),
        _rule(tool_name="a"),
    ]

    with pytest.raises(ToolPackError):
        ToolProfilePayload.from_dict(_profile(fallbacks=unsorted_fallbacks))
    with pytest.raises(ToolPackError):
        ToolProfilePayload.from_dict(_profile(tools=unsorted_tools))
    with pytest.raises(ToolPackError):
        ToolProfilePayload.from_dict(_profile(fallbacks=[_fallback()]))


def test_profile_rejects_exact_and_casefold_identity_collisions() -> None:
    cases = [
        [_rule(), _rule()],
        [_rule(server_key="local:docs"), _rule(server_key="LOCAL:DOCS")],
        [_rule(tool_name="Straße"), _rule(tool_name="STRASSE")],
    ]

    for tools in cases:
        with pytest.raises(ToolPackError) as caught:
            ToolProfilePayload.from_dict(_profile(tools=tools))
        assert caught.value.category == "identity_duplicate"


def test_profile_rejects_non_nfc_and_overlong_tool_identities() -> None:
    for tool_name in ("e\u0301", "雪" * 171):
        with pytest.raises(ToolPackError):
            ToolProfilePayload.from_dict(_profile(tools=[_rule(tool_name=tool_name)]))


def test_manifest_rejects_bool_or_float_for_payload_size() -> None:
    profile_bytes = canonical_json_bytes(_profile())
    for invalid in (True, 1.0):
        manifest = _manifest(profile_bytes)
        manifest["files"][0]["size"] = invalid  # type: ignore[index]
        with pytest.raises(ToolPackError):
            ToolPackManifest.from_dict(manifest)


def test_manifest_collapses_hostile_tree_errors_to_the_manifest_category() -> None:
    profile_bytes = canonical_json_bytes(_profile())
    manifest = _manifest(profile_bytes)
    manifest["producer"]["name"] = manifest  # type: ignore[index]

    with pytest.raises(ToolPackError) as caught:
        ToolPackManifest.from_dict(manifest)
    assert str(caught.value) == "tool_pack.import.manifest_invalid"


def test_public_factories_detach_from_mutable_input_aliases() -> None:
    raw_profile = _profile()
    profile = ToolProfilePayload.from_dict(raw_profile)
    profile_before = profile.to_dict()
    raw_profile["fallbacks"][0]["state"] = "deny"
    raw_profile["tools"][0]["tool_name"] = "changed"
    assert profile.to_dict() == profile_before

    profile_bytes = canonical_json_bytes(profile_before)
    raw_manifest = _manifest(profile_bytes)
    manifest = ToolPackManifest.from_dict(raw_manifest)
    manifest_before = manifest.to_dict()
    raw_manifest["producer"]["name"] = "changed"
    raw_manifest["files"][0]["sha256"] = "f" * 64
    assert manifest.to_dict() == manifest_before


def test_direct_public_construction_cannot_retain_mutable_aliases() -> None:
    features: list[str] = []
    with pytest.raises(ToolPackError, match=r"^tool_pack\.import\.manifest_invalid$"):
        ToolPackManifest(
            "tldw.tool-pack/v1",
            "tldw_chatbook",
            "1.0.0",
            features,  # type: ignore[arg-type]
            "research-tools",
            "Research tools",
            "profile/profile.json",
            10,
            _ZERO_HASH,
            _ZERO_HASH,
        )

    manifest_alias: dict[str, object] = {}
    profile_alias: dict[str, object] = {}
    with pytest.raises(ToolPackError, match=r"^tool_pack\.import\.payload_invalid$"):
        ToolPackDocument(manifest_alias, profile_alias)  # type: ignore[arg-type]


def test_all_direct_public_contract_constructors_reject_invalid_state() -> None:
    with pytest.raises(ToolPackError):
        PortableFallback("mcp", "*", "allow")  # type: ignore[arg-type]
    with pytest.raises(ToolPackError):
        PortableToolRule("mcp", "local:docs", "search", "allow", None)
    with pytest.raises(ToolPackError):
        ToolProfilePayload("tldw.tool-profile/v1", [], [])  # type: ignore[arg-type]

    valid_profile = ToolProfilePayload.from_dict(_profile())
    profile_bytes = canonical_json_bytes(valid_profile.to_dict())
    valid_manifest = ToolPackManifest.from_dict(_manifest(profile_bytes))
    with pytest.raises(ToolPackError):
        ToolPackManifest(
            valid_manifest.schema,
            valid_manifest.producer_name,
            valid_manifest.producer_version,
            valid_manifest.required_features,
            "default",
            valid_manifest.display_name,
            valid_manifest.payload_path,
            valid_manifest.payload_size,
            valid_manifest.payload_sha256,
            valid_manifest.content_digest,
        )
    assert ToolPackDocument(valid_manifest, valid_profile).manifest is valid_manifest


def test_direct_document_construction_rejects_mismatched_manifest_and_profile() -> None:
    first_raw = _profile()
    first_bytes = canonical_json_bytes(first_raw)
    manifest = ToolPackManifest.from_dict(_manifest(first_bytes))
    changed_profile = ToolProfilePayload.from_dict(
        _profile(tools=[_rule(state="ask")])
    )

    with pytest.raises(ToolPackError, match=r"^tool_pack\.import\.manifest_invalid$"):
        ToolPackDocument(manifest, changed_profile)


@pytest.mark.parametrize(
    ("operation", "mutator", "expected_category"),
    [
        ("import", lambda raw: raw.__setitem__("schema", "future"), "schema_unsupported"),
        ("export", lambda raw: raw.__setitem__("schema", "future"), "profile_invalid"),
        (
            "import",
            lambda raw: raw.__setitem__("required_features", ["future"]),
            "feature_unsupported",
        ),
        (
            "export",
            lambda raw: raw.__setitem__("required_features", ["future"]),
            "profile_invalid",
        ),
    ],
)
def test_manifest_validation_maps_semantics_to_the_operation_error_table(
    operation: str, mutator, expected_category: str
) -> None:
    profile_bytes = canonical_json_bytes(_profile())
    raw = _manifest(profile_bytes)
    mutator(raw)

    with pytest.raises(ToolPackError) as caught:
        ToolPackManifest.from_dict(raw, operation=operation)
    assert caught.value.operation == operation
    assert caught.value.category == expected_category


def test_export_canonical_and_profile_validation_never_leak_import_or_contract_codes() -> None:
    with pytest.raises(ToolPackError) as canonical_error:
        canonical_json_bytes(object(), operation="export")
    assert str(canonical_error.value) == "tool_pack.export.profile_invalid"

    raw = _profile()
    raw["tools"][0]["state"] = "invalid"
    with pytest.raises(ToolPackError) as profile_error:
        ToolProfilePayload.from_dict(raw, operation="export")
    assert str(profile_error.value) == "tool_pack.export.profile_invalid"


def test_every_public_export_validator_uses_only_export_error_categories() -> None:
    calls = (
        lambda: PortableFallback.from_dict({}, operation="export"),
        lambda: PortableToolRule.from_dict({}, operation="export"),
        lambda: ToolProfilePayload.from_dict({}, operation="export"),
        lambda: ToolPackManifest.from_dict({}, operation="export"),
        lambda: ToolPackDocument.from_dicts(
            {}, {}, profile_bytes=b"{}\n", operation="export"
        ),
        lambda: strict_json_object(
            b"[]", category="payload_invalid", max_bytes=2, operation="export"
        ),
        lambda: canonical_json_bytes(object(), operation="export"),
        lambda: portable_contract_sha256(_hub(name="e\u0301"), operation="export"),
        lambda: validate_tool_pack_manifest({}, operation="export"),
        lambda: validate_tool_profile_payload({}, operation="export"),
        lambda: validate_tool_pack_document(
            {}, {}, profile_bytes=b"{}\n", operation="export"
        ),
    )

    for call in calls:
        with pytest.raises(ToolPackError) as caught:
            call()
        assert str(caught.value) == "tool_pack.export.profile_invalid"


def test_document_validates_exact_payload_size_hash_digest_and_canonical_bytes() -> None:
    profile = _profile()
    profile_bytes = canonical_json_bytes(profile)
    manifest = _manifest(profile_bytes)
    document = ToolPackDocument.from_dicts(manifest, profile, profile_bytes=profile_bytes)

    assert document.profile.schema == "tldw.tool-profile/v1"
    assert document.manifest.suggested_id == "research-tools"

    for damaged in (profile_bytes + b" ", profile_bytes.replace(b"\n", b"\r\n")):
        with pytest.raises(ToolPackError):
            ToolPackDocument.from_dicts(manifest, profile, profile_bytes=damaged)


def test_document_rejects_profile_payload_over_limit_before_schema_work() -> None:
    profile = _profile()
    profile_bytes = b"{" + b" " * MAX_PROFILE_BYTES + b"}"

    with pytest.raises(ToolPackError, match=r"^tool_pack\.import\.too_large$"):
        ToolPackDocument.from_dicts(_manifest(b"{}\n"), profile, profile_bytes=profile_bytes)


def test_document_independently_rejects_payload_sha256_mismatch() -> None:
    profile = _profile()
    profile_bytes = canonical_json_bytes(profile)
    manifest = _manifest(profile_bytes)
    manifest["files"][0]["sha256"] = "f" * 64
    body = dict(manifest)
    body.pop("content_digest")
    manifest["content_digest"] = hashlib.sha256(
        b"tldw.tool-pack/v1\0"
        + canonical_json_bytes(body)
        + b"\0"
        + profile_bytes
    ).hexdigest()

    with pytest.raises(ToolPackError, match=r"^tool_pack\.import\.manifest_invalid$"):
        ToolPackDocument.from_dicts(manifest, profile, profile_bytes=profile_bytes)


def test_document_independently_rejects_content_digest_mismatch() -> None:
    profile = _profile()
    profile_bytes = canonical_json_bytes(profile)
    manifest = _manifest(profile_bytes)
    manifest["content_digest"] = "f" * 64

    with pytest.raises(ToolPackError, match=r"^tool_pack\.import\.manifest_invalid$"):
        ToolPackDocument.from_dicts(manifest, profile, profile_bytes=profile_bytes)


def test_portable_fingerprint_normalizes_description_and_policy_tags() -> None:
    base = _hub(tags=("network", "mutates"))
    reordered = _hub(description="Search\ndocuments.", tags=("mutates", "network", "network"))

    assert portable_contract_sha256(base) == portable_contract_sha256(reordered)
    assert portable_contract_sha256(base) != portable_contract_sha256(
        _hub(tags=("network",))
    )
    assert portable_contract_sha256(base) != definition_hash(
        base.description, base.input_schema
    )


def test_portable_fingerprint_rejects_non_nfc_identity_and_invalid_schema_tree() -> None:
    for tool in (_hub(name="e\u0301"), _hub(input_schema={"bad": object()})):
        with pytest.raises(ToolPackError):
            portable_contract_sha256(tool)


def test_portable_fingerprint_equals_the_explicit_canonical_preimage() -> None:
    tool = _hub(description="Cafe\u0301\r\ndocuments.")
    expected_preimage = (
        '{"description":"Café\\ndocuments.","input_schema":{"properties":{"q":'
        '{"type":"string"}},"type":"object"},"policy_risk_tags":'
        '["mutates","network"],"tool_name":"search"}\n'
    ).encode()

    assert portable_contract_sha256(tool) == hashlib.sha256(expected_preimage).hexdigest()


def test_portable_fingerprint_includes_each_contract_field_independently() -> None:
    base = _hub(description="Search\ndocuments.")
    base_hash = portable_contract_sha256(base)

    assert portable_contract_sha256(_hub(name="lookup")) != base_hash
    assert portable_contract_sha256(_hub(description="Search\nother documents.")) != base_hash
    assert portable_contract_sha256(_hub(input_schema={"type": "object"})) != base_hash
    assert portable_contract_sha256(_hub(tags=("network",))) != base_hash


def test_portable_fingerprint_excludes_authority_and_server_identity() -> None:
    base = _hub(description="Search\ndocuments.")

    assert portable_contract_sha256(
        _hub(
            server_key="builtin:tldw_chatbook",
            server_label="Built in",
            source="builtin",
            description="Search\ndocuments.",
        )
    ) == portable_contract_sha256(base)

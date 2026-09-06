from __future__ import annotations


def test_library_unavailable_inspection_and_browse_payloads_are_distinct() -> None:
    """Catch exact inspection or complete browse being serialized as repair."""

    assert hasattr(navigation, "LibraryUnavailableConversationInspection")
    assert hasattr(navigation, "LibraryUnavailableConversationsBrowse")
    unresolved = UnresolvedConversationKey("Authority-A", "Conversation-X")
    return_target = RoleplayReturnTarget.console_context_character()
    inspection = navigation.LibraryUnavailableConversationInspection(
        unresolved=unresolved,
        return_target=return_target,
    )
    browse = navigation.LibraryUnavailableConversationsBrowse(
        selected=unresolved,
        return_target=return_target,
    )

    inspection_payload = navigation.serialize_library_unavailable_inspection(inspection)
    browse_payload = navigation.serialize_library_unavailable_browse(browse)

    assert (
        navigation.deserialize_library_unavailable_inspection(inspection_payload)
        == inspection
    )
    assert navigation.deserialize_library_unavailable_browse(browse_payload) == browse
    assert inspection_payload == {
        "version": 1,
        "source": "local",
        "data_authority_id": "Authority-A",
        "unresolved": {
            "version": 1,
            "tag": "unresolved_conversation",
            "data_authority_id": "Authority-A",
            "conversation_id": "Conversation-X",
        },
        "return_target": {
            "screen_id": "chat",
            "focus_id": "console-context-character",
        },
    }
    assert browse_payload == {
        "version": 1,
        "source": "local",
        "data_authority_id": "Authority-A",
        "selected": inspection_payload["unresolved"],
        "return_target": inspection_payload["return_target"],
    }


import pytest

from tldw_chatbook.Character_Chat.character_conversation_navigation import (
    ResolvedLocalCharacterKey,
    UnresolvedConversationKey,
)
from tldw_chatbook.UI.Navigation import character_conversation_navigation as navigation
from tldw_chatbook.UI.Navigation.character_conversation_navigation import (
    LibraryCharacterRepairContext,
    RoleplayCharacterConversationLink,
    RoleplayReturnTarget,
    deserialize_library_character_repair_context,
    deserialize_roleplay_character_conversation_link,
    serialize_library_character_repair_context,
    serialize_roleplay_character_conversation_link,
)


@pytest.mark.parametrize("kind", ["inspection", "browse"])
@pytest.mark.parametrize(
    "invalid",
    [
        "boolean_version",
        "nested_boolean_version",
        "extra",
        "authority",
        "resolved",
        "return_extra",
    ],
)
def test_unavailable_routes_preserve_strict_lazy_wire_boundary(kind, invalid):
    key = UnresolvedConversationKey("Authority-A", "Conversation-X")
    anchor = RoleplayReturnTarget.console_context_character()
    if kind == "inspection":
        payload = navigation.serialize_library_unavailable_inspection(
            navigation.LibraryUnavailableConversationInspection(key, anchor)
        )
        deserialize = navigation.deserialize_library_unavailable_inspection
        field = "unresolved"
    else:
        payload = navigation.serialize_library_unavailable_browse(
            navigation.LibraryUnavailableConversationsBrowse(key, anchor)
        )
        deserialize = navigation.deserialize_library_unavailable_browse
        field = "selected"
    if invalid == "boolean_version":
        payload["version"] = True
    elif invalid == "nested_boolean_version":
        payload[field]["version"] = True
    elif invalid == "extra":
        payload["unexpected"] = "value"
    elif invalid == "authority":
        payload["data_authority_id"] = "Authority-B"
    elif invalid == "resolved":
        payload[field]["tag"] = "resolved_local_character"
    else:
        payload["return_target"]["unexpected"] = "value"
    with pytest.raises(ValueError):
        deserialize(payload)


def test_roleplay_payload_round_trip_preserves_exact_typed_key_and_safe_fields() -> (
    None
):
    link = RoleplayCharacterConversationLink(
        character=ResolvedLocalCharacterKey("Authority-A", 7),
        conversation_id="Conversation-X",
        query="silver rain",
        data_revision=12,
        return_target=RoleplayReturnTarget.console_context_character(),
    )

    payload = serialize_roleplay_character_conversation_link(link)

    assert deserialize_roleplay_character_conversation_link(payload) == link
    assert payload == {
        "version": 1,
        "source": "local",
        "character": {
            "version": 1,
            "tag": "resolved_local_character",
            "data_authority_id": "Authority-A",
            "character_id": 7,
        },
        "conversation_id": "Conversation-X",
        "query": "silver rain",
        "data_revision": 12,
        "return_target": {
            "screen_id": "chat",
            "focus_id": "console-context-character",
        },
    }
    assert not ({"prompt", "transcript", "credential", "card", "path"} & payload.keys())


def test_library_repair_payload_round_trip_preserves_exact_unresolved_key() -> None:
    context = LibraryCharacterRepairContext(
        unresolved=UnresolvedConversationKey("Authority-A", "Conversation-X"),
        expected_conversation_version=4,
        historical_display_snapshot="Historical Ada",
        return_target=RoleplayReturnTarget.personas_conversations(),
    )

    payload = serialize_library_character_repair_context(context)

    assert deserialize_library_character_repair_context(payload) == context
    assert payload["unresolved"]["tag"] == "unresolved_conversation"
    assert not (
        {"prompt", "transcript", "credential", "card_body", "filesystem_path"}
        & payload.keys()
    )


@pytest.mark.parametrize("version", [0, 2, "1", None])
def test_payloads_reject_unknown_versions(version: object) -> None:
    payload = serialize_roleplay_character_conversation_link(
        RoleplayCharacterConversationLink(ResolvedLocalCharacterKey("authority", 1))
    )
    payload["version"] = version

    with pytest.raises(ValueError, match="version"):
        deserialize_roleplay_character_conversation_link(payload)


@pytest.mark.parametrize("conversation_id", ["", " ", "x" * 257])
def test_roleplay_payload_rejects_blank_or_overlong_conversation_ids(
    conversation_id: str,
) -> None:
    payload = serialize_roleplay_character_conversation_link(
        RoleplayCharacterConversationLink(ResolvedLocalCharacterKey("authority", 1))
    )
    payload["conversation_id"] = conversation_id

    with pytest.raises(ValueError, match="conversation_id"):
        deserialize_roleplay_character_conversation_link(payload)


def test_payloads_reject_nonlocal_sources_and_mismatched_authority_components() -> None:
    roleplay = serialize_roleplay_character_conversation_link(
        RoleplayCharacterConversationLink(ResolvedLocalCharacterKey("authority", 1))
    )
    roleplay["source"] = "server"
    with pytest.raises(ValueError, match="local"):
        deserialize_roleplay_character_conversation_link(roleplay)

    repair = serialize_library_character_repair_context(
        LibraryCharacterRepairContext(
            unresolved=UnresolvedConversationKey("authority", "conversation"),
            expected_conversation_version=1,
            historical_display_snapshot="Historical",
            return_target=RoleplayReturnTarget.personas_filter(),
        )
    )
    repair["data_authority_id"] = "other-authority"
    with pytest.raises(ValueError, match="authority"):
        deserialize_library_character_repair_context(repair)


@pytest.mark.parametrize(
    "focus_id", ["", " ", "#already-prefixed", "bad focus", "x" * 129, "../path"]
)
def test_return_target_rejects_invalid_focus_ids(focus_id: str) -> None:
    with pytest.raises(ValueError, match="focus_id"):
        RoleplayReturnTarget("chat", focus_id)


@pytest.mark.parametrize(
    ("payload_name", "nested_field"),
    (("roleplay", "character"), ("repair", "unresolved")),
)
def test_payloads_reject_unexpected_nested_identity_fields(
    payload_name: str, nested_field: str
) -> None:
    """A nested identity parser must not accept authority-adjacent data."""

    roleplay = serialize_roleplay_character_conversation_link(
        RoleplayCharacterConversationLink(ResolvedLocalCharacterKey("authority", 1))
    )
    repair = serialize_library_character_repair_context(
        LibraryCharacterRepairContext(
            unresolved=UnresolvedConversationKey("authority", "conversation"),
            expected_conversation_version=1,
            historical_display_snapshot="Historical",
            return_target=RoleplayReturnTarget.personas_conversations(),
        )
    )
    payload = roleplay if payload_name == "roleplay" else repair
    payload[nested_field]["unexpected"] = "must-not-be-accepted"

    parser = (
        deserialize_roleplay_character_conversation_link
        if payload_name == "roleplay"
        else deserialize_library_character_repair_context
    )
    with pytest.raises(ValueError, match="extra_forbidden"):
        parser(payload)


@pytest.mark.parametrize(
    ("screen_id", "focus_id"),
    (("settings", "arbitrary-focus"), ("chat", "personas-filter")),
)
def test_return_target_rejects_routes_outside_closed_character_flow(
    screen_id: str, focus_id: str
) -> None:
    """Character navigation cannot smuggle an arbitrary app destination."""

    with pytest.raises(ValueError, match="return target"):
        RoleplayReturnTarget(screen_id, focus_id)


def test_exact_conversation_link_requires_captured_query_revision() -> None:
    """Resume payloads bind the immutable list snapshot they came from."""

    with pytest.raises(ValueError, match="data_revision"):
        RoleplayCharacterConversationLink(
            ResolvedLocalCharacterKey("authority", 1),
            conversation_id="conversation",
        )


@pytest.mark.parametrize("version", [True, 1.0])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("kind", ["roleplay", "repair"])
def test_wire_versions_reject_integer_coercion(kind, nested, version) -> None:
    if kind == "roleplay":
        payload = serialize_roleplay_character_conversation_link(
            RoleplayCharacterConversationLink(ResolvedLocalCharacterKey("authority", 1))
        )
        parse = deserialize_roleplay_character_conversation_link
        identity = "character"
    else:
        payload = serialize_library_character_repair_context(
            LibraryCharacterRepairContext(
                UnresolvedConversationKey("authority", "conversation"),
                1,
                "Historical",
                RoleplayReturnTarget.personas_filter(),
            )
        )
        parse = deserialize_library_character_repair_context
        identity = "unresolved"
    (payload[identity] if nested else payload)["version"] = version
    with pytest.raises(ValueError, match="version"):
        parse(payload)


@pytest.mark.parametrize("kind", ("roleplay", "repair"))
def test_strict_wire_requires_every_field_and_rejects_extra_fields_at_each_level(kind):
    from copy import deepcopy

    if kind == "roleplay":
        payload = serialize_roleplay_character_conversation_link(
            RoleplayCharacterConversationLink(
                ResolvedLocalCharacterKey("authority", 1),
                "conversation",
                "",
                1,
                RoleplayReturnTarget.personas_filter(),
            )
        )
        parse = deserialize_roleplay_character_conversation_link
        identity = "character"
    else:
        payload = serialize_library_character_repair_context(
            LibraryCharacterRepairContext(
                UnresolvedConversationKey("authority", "conversation"),
                1,
                "Historical",
                RoleplayReturnTarget.personas_filter(),
            )
        )
        parse = deserialize_library_character_repair_context
        identity = "unresolved"
    for scope in (None, identity, "return_target"):
        fields = payload if scope is None else payload[scope]
        for field in (*fields, "unexpected"):
            changed = deepcopy(payload)
            destination = changed if scope is None else changed[scope]
            if field == "unexpected":
                destination[field] = "private-payload-must-not-appear"
            else:
                destination.pop(field)
            with pytest.raises(ValueError) as caught:
                parse(changed)
            assert "private-payload-must-not-appear" not in str(caught.value)


@pytest.mark.parametrize(
    "field,value",
    (
        ("character_id", True),
        ("character_id", "1"),
        ("character_id", 1.0),
        ("character_id", 2**63),
        ("data_authority_id", " authority"),
        ("data_authority_id", "é" * 129),
    ),
)
def test_strict_wire_rejects_identity_coercion_and_noncanonical_bounds(field, value):
    payload = serialize_roleplay_character_conversation_link(
        RoleplayCharacterConversationLink(ResolvedLocalCharacterKey("authority", 1))
    )
    payload["character"][field] = value
    with pytest.raises(ValueError):
        deserialize_roleplay_character_conversation_link(payload)

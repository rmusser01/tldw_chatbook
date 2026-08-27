from dataclasses import FrozenInstanceError, fields, replace
from typing import get_args

import pytest

from tldw_chatbook.Chat import console_chat_fork
from tldw_chatbook.Chat.console_chat_fork import (
    CONSOLE_FORK_FINGERPRINT_JSON_MAX_BYTES,
    CONSOLE_FORK_TITLE_MAX_LENGTH,
    ConsoleChatForkSnapshot,
    ConsoleForkCitationState,
    ConsoleForkCitationLink,
    ConsoleForkConfigurationSnapshot,
    ConsoleForkDurability,
    ConsoleForkEligibility,
    ConsoleForkFence,
    ConsoleForkImageSelectionFence,
    ConsoleForkLineageFence,
    ConsoleForkProjectedMessage,
    default_fork_title,
    normalize_fork_title,
)
from tldw_chatbook.Chat.console_context_policy import ConsoleContextPolicyOverrides
from tldw_chatbook.Chat.console_library_policy import (
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleLibraryPolicyCandidate,
)
from tldw_chatbook.Chat.console_project_instructions import (
    ProjectInstructionControlState,
    sanitize_fork_project_instruction_state,
)
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_speech_preferences import ConsoleSpeechPreferences


def _configuration_snapshot() -> ConsoleForkConfigurationSnapshot:
    return ConsoleForkConfigurationSnapshot(
        workspace_id="global",
        settings=ConsoleSessionSettings(provider="openai", model="gpt-test"),
        rag_scope=None,
        context_policy_overrides=ConsoleContextPolicyOverrides(),
        library_policy=ConsoleLibraryPolicyCandidate(
            auto_retrieve=ConsoleAutoRetrieve.NEVER,
            assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
        ),
        runtime_backend="chat",
        assistant_kind=None,
        assistant_id=None,
        assistant_authority_id=None,
        persona_memory_mode=None,
        character_id=None,
        character_name=None,
        user_display_name_override=None,
        character_system_template=None,
        speech_preferences=ConsoleSpeechPreferences(),
        project_instruction_state=ProjectInstructionControlState.new_session(),
    )


def test_fork_titles_use_the_approved_defaults_and_bound() -> None:
    assert default_fork_title("Research notes") == "Forked from Research notes"
    assert default_fork_title("") == "Untitled chat — fork"
    assert len(normalize_fork_title("x" * 100)) == 60


def test_fork_title_rejects_blank_normalized_text() -> None:
    with pytest.raises(ValueError, match="blank"):
        normalize_fork_title(" \n\t ")


def test_fork_title_reuses_the_console_title_deriver(monkeypatch) -> None:
    calls: list[tuple[str, int]] = []

    def fake_derive(draft: str, *, max_length: int) -> str:
        calls.append((draft, max_length))
        return "canonical title"

    monkeypatch.setattr(
        console_chat_fork,
        "derive_console_session_title",
        fake_derive,
    )

    assert normalize_fork_title("  proposed title  ") == "canonical title"
    assert calls == [("  proposed title  ", CONSOLE_FORK_TITLE_MAX_LENGTH)]


def test_fork_project_instruction_state_clears_only_source_notice_authority() -> None:
    source = ProjectInstructionControlState(
        project_instructions_enabled=True,
        working_folder_binding_id="binding-1",
        working_folder_locator_fingerprint="locator-fingerprint",
        project_instruction_notice_key="source-consent",
    )

    assert sanitize_fork_project_instruction_state(source) == replace(
        source,
        project_instruction_notice_key=None,
    )


def test_fork_records_are_frozen_slotted_contracts() -> None:
    record_types = (
        ConsoleForkEligibility,
        ConsoleForkLineageFence,
        ConsoleForkImageSelectionFence,
        ConsoleForkFence,
        ConsoleForkProjectedMessage,
        ConsoleForkConfigurationSnapshot,
        ConsoleForkCitationLink,
        ConsoleChatForkSnapshot,
    )

    assert all(record_type.__dataclass_params__.frozen for record_type in record_types)
    assert all(hasattr(record_type, "__slots__") for record_type in record_types)

    eligibility = ConsoleForkEligibility(True)
    with pytest.raises(FrozenInstanceError):
        eligibility.eligible = False  # type: ignore[misc]


@pytest.mark.parametrize(
    ("record_type", "expected_fields"),
    (
        (ConsoleForkEligibility, ("eligible", "reason")),
        (
            ConsoleForkLineageFence,
            (
                "native_message_id",
                "persisted_message_id",
                "native_parent_id",
                "role",
                "status",
                "visible_content",
                "visible_variant_id",
                "sibling_identity",
                "persisted_revision",
                "attachment_fingerprint",
            ),
        ),
        (
            ConsoleForkImageSelectionFence,
            (
                "native_message_id",
                "selected_position",
                "browse_revision",
                "attachment_meta_fingerprint",
            ),
        ),
        (
            ConsoleForkFence,
            (
                "source_session_id",
                "source_conversation_id",
                "source_conversation_version",
                "source_durability",
                "source_title",
                "source_configuration_fingerprint",
                "boundary_message_id",
                "lineage",
                "image_selections",
            ),
        ),
        (
            ConsoleForkProjectedMessage,
            (
                "native_message_id",
                "persisted_message_id",
                "native_parent_id",
                "persisted_parent_id",
                "turn_id",
                "role",
                "status",
                "content",
            ),
        ),
        (
            ConsoleForkConfigurationSnapshot,
            (
                "workspace_id",
                "settings",
                "rag_scope",
                "context_policy_overrides",
                "library_policy",
                "runtime_backend",
                "assistant_kind",
                "assistant_id",
                "assistant_authority_id",
                "persona_memory_mode",
                "character_id",
                "character_name",
                "user_display_name_override",
                "character_system_template",
                "speech_preferences",
                "project_instruction_state",
            ),
        ),
        (
            ConsoleForkCitationLink,
            ("source_persisted_message_id", "source_revision", "state"),
        ),
        (
            ConsoleChatForkSnapshot,
            (
                "fork_session_id",
                "fork_conversation_id",
                "title",
                "source_session_id",
                "source_conversation_id",
                "source_boundary_persisted_message_id",
                "durable",
                "messages",
                "configuration",
                "citation_links",
            ),
        ),
    ),
)
def test_fork_public_record_fields_are_exact(record_type, expected_fields) -> None:
    assert tuple(field.name for field in fields(record_type)) == expected_fields


def test_fork_literal_domains_are_exact() -> None:
    assert get_args(ConsoleForkDurability) == (
        "temporary",
        "durable",
        "unsaved_persistable",
    )
    assert get_args(ConsoleForkCitationState) == (
        "active_required",
        "unavailable",
        "none",
    )


def test_fork_fingerprint_is_canonical_and_domain_separated() -> None:
    configuration = _configuration_snapshot()
    image_selection = ConsoleForkImageSelectionFence(
        native_message_id="message-1",
        selected_position=0,
        browse_revision=1,
        attachment_meta_fingerprint="sha256:attachment",
    )

    first = console_chat_fork.fingerprint_console_fork_configuration(configuration)
    repeated = console_chat_fork.fingerprint_console_fork_configuration(
        replace(configuration)
    )
    other_domain = console_chat_fork.fingerprint_console_fork_image_selection(
        image_selection
    )

    assert first == repeated
    assert first != other_domain
    assert len(first) == 64


def test_private_fork_hash_uses_canonical_json_and_purpose_domains() -> None:
    fingerprint = console_chat_fork._fingerprint_console_fork_payload

    assert fingerprint("test", {"alpha": 1, "beta": 2}) == fingerprint(
        "test", {"beta": 2, "alpha": 1}
    )
    assert fingerprint("test", {"alpha": 1}) != fingerprint(
        "other", {"alpha": 1}
    )


def test_fork_fingerprint_rejects_unbounded_allowlisted_payload() -> None:
    with pytest.raises(ValueError, match="bounded"):
        console_chat_fork.fingerprint_console_fork_configuration(
            replace(
                _configuration_snapshot(),
                workspace_id="x" * CONSOLE_FORK_FINGERPRINT_JSON_MAX_BYTES,
            )
        )


@pytest.mark.parametrize(
    "forbidden_payload",
    (
        {"scratch_owner": "source-session"},
        {"permission_state": {"fs_write": "allow"}},
        {"raw_path": "/private/tmp/source-scratch"},
    ),
)
def test_fork_fingerprint_public_flow_rejects_non_allowlisted_payloads(
    forbidden_payload,
) -> None:
    with pytest.raises(TypeError, match="ConsoleForkConfigurationSnapshot"):
        console_chat_fork.fingerprint_console_fork_configuration(forbidden_payload)


def test_fork_fingerprint_rejects_source_notice_authority() -> None:
    source_authority = replace(
        _configuration_snapshot(),
        project_instruction_state=replace(
            ProjectInstructionControlState.new_session(),
            project_instruction_notice_key="source-consent",
        ),
    )

    with pytest.raises(ValueError, match="notice authority"):
        console_chat_fork.fingerprint_console_fork_configuration(source_authority)


def test_generic_fork_payload_fingerprint_is_not_public() -> None:
    assert not hasattr(console_chat_fork, "fingerprint_console_fork_payload")

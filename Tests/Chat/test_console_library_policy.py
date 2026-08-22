"""Contracts for device-local Console Library policy values."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tldw_chatbook.Chat.console_library_policy import (
    AUTOMATIC_LIBRARY_SOURCE_TYPES,
    ConsoleAssistantLibraryAccess,
    ConsoleAutoRetrieve,
    ConsoleConversationLibraryPolicy,
    ConsoleLibraryMigrationSeed,
    ConsoleLibraryPolicyDefaults,
    ConsoleLibraryPolicyHolder,
    ConsoleLibraryPolicySnapshot,
    normalize_policy_read,
)


def test_new_session_defaults_are_never_and_blocked():
    """Changing a safe shipped default must not grant Library authority."""
    defaults = ConsoleLibraryPolicyDefaults(
        auto_retrieve=ConsoleAutoRetrieve.NEVER,
        assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
    )

    assert defaults.auto_retrieve is ConsoleAutoRetrieve.NEVER
    assert defaults.assistant_access is ConsoleAssistantLibraryAccess.BLOCKED
    assert AUTOMATIC_LIBRARY_SOURCE_TYPES == ("notes", "media", "conversations")


@pytest.mark.parametrize("raw_value", ["true", 1, None])
def test_migration_seed_rejects_values_that_are_not_bools(raw_value):
    """A caller cannot accidentally turn permissive config values into a seed."""
    with pytest.raises(TypeError, match="auto_retrieve_on_send must be a bool"):
        ConsoleLibraryMigrationSeed(auto_retrieve_on_send=raw_value)


def test_policy_read_keeps_a_valid_durable_policy_and_revision():
    """Replacing a durable row with its safe fallback would revoke a chosen policy."""
    policy = ConsoleConversationLibraryPolicy(
        conversation_id="conversation-1",
        auto_retrieve=ConsoleAutoRetrieve.AUTOMATIC,
        assistant_access=ConsoleAssistantLibraryAccess.ALLOWED,
        policy_revision=7,
        updated_at="2026-08-22T12:00:00Z",
    )

    result = normalize_policy_read(policy)

    assert result.durable_policy is policy
    assert result.snapshot == ConsoleLibraryPolicySnapshot(
        auto_retrieve=ConsoleAutoRetrieve.AUTOMATIC,
        assistant_access=ConsoleAssistantLibraryAccess.ALLOWED,
        policy_revision=7,
        source="durable",
    )


@pytest.mark.parametrize(
    "invalid_field",
    [
        {"conversation_id": 7},
        {"auto_retrieve": "automatic"},
        {"assistant_access": "allowed"},
        {"policy_revision": True},
        {"updated_at": 7},
    ],
)
def test_policy_read_rejects_malformed_durable_policy_dataclass(invalid_field):
    """Runtime-invalid dataclass fields must not become durable authority."""
    policy = replace(
        ConsoleConversationLibraryPolicy(
            conversation_id="conversation-1",
            auto_retrieve=ConsoleAutoRetrieve.AUTOMATIC,
            assistant_access=ConsoleAssistantLibraryAccess.ALLOWED,
            policy_revision=7,
            updated_at="2026-08-22T12:00:00Z",
        ),
        **invalid_field,
    )

    result = normalize_policy_read(policy)

    assert result.durable_policy is None
    assert result.snapshot.source == "unavailable"
    assert result.snapshot.error_code == "corrupt_policy"
    assert result.snapshot.auto_retrieve is ConsoleAutoRetrieve.NEVER
    assert result.snapshot.assistant_access is ConsoleAssistantLibraryAccess.BLOCKED


def test_policy_read_uses_safe_missing_snapshot_when_no_row_exists():
    """A synced conversation without local policy must never inherit permission."""
    result = normalize_policy_read(None)

    assert result.durable_policy is None
    assert result.snapshot == ConsoleLibraryPolicySnapshot(
        auto_retrieve=ConsoleAutoRetrieve.NEVER,
        assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
        policy_revision=None,
        source="missing",
    )


def test_policy_read_marks_corrupt_values_unavailable_without_granting_access():
    """Malformed persistence data must not appear as an Allowed durable policy."""
    result = normalize_policy_read({"policy_revision": "not-an-int"})

    assert result.durable_policy is None
    assert result.snapshot.source == "unavailable"
    assert result.snapshot.error_code == "corrupt_policy"
    assert result.snapshot.auto_retrieve is ConsoleAutoRetrieve.NEVER
    assert result.snapshot.assistant_access is ConsoleAssistantLibraryAccess.BLOCKED


def test_policy_read_hides_exception_text_behind_a_bounded_error_code():
    """A database exception cannot leak into policy/UI state or grant access."""
    result = normalize_policy_read(RuntimeError("database secret: /private/path"))

    assert result.durable_policy is None
    assert result.snapshot.source == "unavailable"
    assert result.snapshot.error_code == "policy_read_error"
    assert "private" not in result.snapshot.error_code
    assert result.snapshot.assistant_access is ConsoleAssistantLibraryAccess.BLOCKED


def test_holder_retains_explicitly_staged_policy_state():
    """A save coordinator can distinguish an explicit user choice from loading."""
    snapshot = ConsoleLibraryPolicySnapshot(
        auto_retrieve=ConsoleAutoRetrieve.AUTOMATIC,
        assistant_access=ConsoleAssistantLibraryAccess.ALLOWED,
        policy_revision=None,
        source="temporary",
    )

    holder = ConsoleLibraryPolicyHolder(
        snapshot=snapshot,
        explicitly_staged=True,
        save_pending=True,
    )

    assert holder.snapshot is snapshot
    assert holder.explicitly_staged is True
    assert holder.save_pending is True

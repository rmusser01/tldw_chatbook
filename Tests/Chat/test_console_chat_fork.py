from dataclasses import FrozenInstanceError, replace

import pytest

from tldw_chatbook.Chat import console_chat_fork
from tldw_chatbook.Chat.console_chat_fork import (
    CONSOLE_FORK_FINGERPRINT_JSON_MAX_BYTES,
    CONSOLE_FORK_TITLE_MAX_LENGTH,
    ConsoleChatForkSnapshot,
    ConsoleForkCitationLink,
    ConsoleForkConfigurationSnapshot,
    ConsoleForkEligibility,
    ConsoleForkFence,
    ConsoleForkImageSelectionFence,
    ConsoleForkLineageFence,
    ConsoleForkProjectedMessage,
    default_fork_title,
    fingerprint_console_fork_payload,
    normalize_fork_title,
)
from tldw_chatbook.Chat.console_project_instructions import (
    ProjectInstructionControlState,
    sanitize_fork_project_instruction_state,
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


def test_fork_fingerprint_is_canonical_and_domain_separated() -> None:
    first = fingerprint_console_fork_payload(
        "configuration",
        {"workspace": "global", "settings": {"model": "m", "stream": True}},
    )
    reordered = fingerprint_console_fork_payload(
        "configuration",
        {"settings": {"stream": True, "model": "m"}, "workspace": "global"},
    )
    other_domain = fingerprint_console_fork_payload(
        "image-selection",
        {"workspace": "global", "settings": {"model": "m", "stream": True}},
    )

    assert first == reordered
    assert first != other_domain
    assert len(first) == 64


def test_fork_fingerprint_rejects_unbounded_or_non_json_payloads() -> None:
    with pytest.raises(ValueError, match="bounded"):
        fingerprint_console_fork_payload(
            "configuration",
            "x" * CONSOLE_FORK_FINGERPRINT_JSON_MAX_BYTES,
        )
    with pytest.raises(TypeError, match="canonical JSON"):
        fingerprint_console_fork_payload("configuration", object())

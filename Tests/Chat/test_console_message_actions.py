from dataclasses import replace

import pytest

from tldw_chatbook.Chat import console_message_actions as message_actions
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleActivityPresentation,
    ConsoleChatMessage,
    ConsoleMessageRole,
    ConsoleVariantSet,
)
from tldw_chatbook.Chat.console_chat_fork import ConsoleForkEligibility
from tldw_chatbook.Chat.console_message_actions import (
    ConsoleMessageActionService,
    action_row_guide,
)
from tldw_chatbook.Video_Generation.video_metadata import VideoGenerationMetadata


def test_assistant_message_actions_include_required_order():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="answer")

    actions = service.available_actions(message)

    assert [action.label for action in actions] == [
        "Copy",
        "🔊",
        "Edit",
        "Save as...",
        "Fork",
        "♻",
        "--->",
        "Feedback",
        "🗑",
    ]


def test_failed_user_row_offers_no_retry_action():
    """TASK-457(a): retry regenerates a failed ASSISTANT response; a failed USER
    row (the send-blocked optimistic echo) has nothing to regenerate, so it must
    not offer 'retry' — a failed ASSISTANT row still does."""
    service = ConsoleMessageActionService()

    failed_user = ConsoleChatMessage(
        role=ConsoleMessageRole.USER, content="hello", status="failed"
    )
    assert "retry" not in [
        action.action_id for action in service.available_actions(failed_user)
    ]

    failed_assistant = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT, content="", status="failed"
    )
    assert "retry" in [
        action.action_id for action in service.available_actions(failed_assistant)
    ]


def test_streaming_assistant_message_shows_completed_actions_disabled_with_reasons():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="partial",
        status="streaming",
    )

    actions = service.available_actions(message)

    assert [action.label for action in actions] == [
        "Copy",
        "Edit",
        "Save as...",
        "Fork",
        "♻",
        "--->",
        "Feedback",
        "🗑",
    ]
    assert all(action.enabled is False for action in actions)
    assert all(action.disabled_reason for action in actions)
    assert all(
        "finish" in action.disabled_reason.lower() or "WIP" in action.disabled_reason
        for action in actions
    )


def test_pending_assistant_message_shows_completed_actions_disabled_with_reasons():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="",
        status="pending",
    )

    actions = service.available_actions(message)

    assert [action.label for action in actions] == [
        "Copy",
        "Edit",
        "Save as...",
        "Fork",
        "♻",
        "--->",
        "Feedback",
        "🗑",
    ]
    assert all(action.enabled is False for action in actions)
    assert all(action.disabled_reason for action in actions)


def test_unavailable_save_destinations_carry_honest_default_reason():
    service = ConsoleMessageActionService(available_save_destinations={"Chatbook"})
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="answer")

    destinations = service.save_as_destinations(message)

    note = next(
        destination for destination in destinations if destination.label == "Note"
    )
    assert note.available is False
    assert note.reason == "Save as Note is not available in this session."
    assert "WIP" not in note.reason


def test_unavailable_save_destinations_use_provided_specific_reason():
    service = ConsoleMessageActionService(
        available_save_destinations={"Note", "Media", "Prompt"},
        unavailable_save_reasons={
            "Chatbook": "Only assistant responses can be saved as Chatbook artifacts.",
        },
    )
    message = ConsoleChatMessage(role=ConsoleMessageRole.USER, content="question")

    destinations = service.save_as_destinations(message)

    chatbook = next(
        destination for destination in destinations if destination.label == "Chatbook"
    )
    assert chatbook.available is False
    assert (
        chatbook.reason
        == "Only assistant responses can be saved as Chatbook artifacts."
    )
    assert [d.label for d in destinations if d.available] == ["Note", "Media", "Prompt"]


def test_available_save_destinations_have_no_reason():
    service = ConsoleMessageActionService(
        available_save_destinations={"Chatbook", "Note", "Media", "Prompt"},
    )
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="answer")

    destinations = service.save_as_destinations(message)

    assert all(destination.available for destination in destinations)
    assert all(destination.reason == "" for destination in destinations)


def test_action_labels_fit_compact_terminal_width_budget():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="answer")

    labels = service.plain_action_labels(message)

    assert " ".join(labels) == "Copy 🔊 Edit Fork ♻ ---> More…"
    assert len(" ".join(labels)) <= 48


def test_variant_action_labels_use_symbolic_navigation():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="first",
        id="m1",
        sibling_index=1,
        sibling_count=2,
    )

    actions = service.available_actions(message)

    assert [action.label for action in actions] == [
        "Copy",
        "🔊",
        "Edit",
        "Save as...",
        "<",
        ">",
        "Fork",
        "♻",
        "--->",
        "Feedback",
        "🗑",
    ]


@pytest.mark.parametrize(
    ("sibling_index", "sibling_count", "previous_enabled", "next_enabled"),
    [
        (0, 3, False, True),
        (1, 3, True, True),
        (2, 3, True, False),
    ],
)
def test_variant_nav_actions_gate_on_sibling_position(
    sibling_index: int,
    sibling_count: int,
    previous_enabled: bool,
    next_enabled: bool,
):
    """TASK-7: `<`/`>` enable state follows the sibling position, not the
    retired ``ConsoleVariantSet`` selection index."""
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="reply",
        id="m1",
        sibling_index=sibling_index,
        sibling_count=sibling_count,
    )

    actions = {
        action.action_id: action for action in service.available_actions(message)
    }

    assert actions["variant-previous"].enabled is previous_enabled
    assert actions["variant-next"].enabled is next_enabled


def test_variant_nav_actions_absent_for_linear_single_child_message():
    """TASK-7: gate is now ``sibling_count > 1``, not ``variants is not None``
    -- a linear (unforked) message offers no `<`/`>` at all."""
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT, content="reply", id="m1"
    )

    action_ids = [action.action_id for action in service.available_actions(message)]

    assert "variant-previous" not in action_ids
    assert "variant-next" not in action_ids


def test_variant_action_labels_fit_compact_terminal_width_budget():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="first",
        id="m1",
        sibling_index=1,
        sibling_count=2,
    )

    labels = service.plain_action_labels(message)

    assert " ".join(labels) == "Copy 🔊 Edit Fork ♻ ---> More…"
    assert len(" ".join(labels)) <= 52


def test_failed_action_labels_include_retry_inside_terminal_width_budget():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="failed",
        status="failed",
    )

    labels = service.plain_action_labels(message)

    assert " ".join(labels) == "Copy Edit Fork Retry ---> More…"
    assert len(" ".join(labels)) <= 52


def test_copy_action_returns_clipboard_text():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="answer")

    result = service.dispatch("copy", message)

    assert result.status == "completed"
    assert result.clipboard_text == "answer"


def test_delete_action_returns_completed_result():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="answer")

    result = service.dispatch("delete", message)

    assert result.status == "completed"
    assert result.visible_copy == "Deleted message from transcript."
    assert result.target_message_id == message.id


@pytest.mark.parametrize(
    ("action_id", "expected_feedback"),
    [("feedback-up", "up"), ("feedback-down", "down")],
)
def test_feedback_actions_return_completed_result(
    action_id: str, expected_feedback: str
):
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="answer")

    result = service.dispatch(action_id, message)

    assert result.status == "completed"
    assert result.visible_copy == f"Marked message feedback: {expected_feedback}."
    assert result.target_message_id == message.id
    assert result.target_content == expected_feedback


def test_edit_action_requests_modal_with_current_message_content():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="answer")

    result = service.dispatch("edit", message)

    assert result.status == "edit_requested"
    assert result.visible_copy == "Opened Edit Message."
    assert result.target_message_id == message.id
    assert result.target_content == "answer"


def test_unimplemented_actions_return_wip_reason():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="answer")

    result = service.dispatch("save-later", message)

    assert result.status == "wip"
    assert "WIP" in result.visible_copy


def test_regression_no_generation_kwargs_matches_text_sibling_gating():
    """Regression guard: callers that don't pass the new generation kwargs
    (every existing call site as of this task) must see byte-identical
    behavior to before -- pinned against a real text-sibling case. Also pins
    the TASK-1 speak action landing in its one new legacy spot (right after
    Copy) without disturbing anything else in this set."""
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="reply",
        id="m1",
        sibling_index=1,
        sibling_count=3,
    )

    actions = service.available_actions(message)

    assert [action.label for action in actions] == [
        "Copy",
        "🔊",
        "Edit",
        "Save as...",
        "<",
        ">",
        "Fork",
        "♻",
        "--->",
        "Feedback",
        "🗑",
    ]
    by_id = {action.action_id: action for action in actions}
    assert by_id["variant-previous"].enabled is True
    assert by_id["variant-next"].enabled is True
    assert "keep" not in by_id


def test_generation_variant_nav_hidden_at_count_one():
    """A single-variant generation message offers no `<`/`>`/keep at all."""
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="[image] x")

    action_ids = [
        action.action_id
        for action in service.available_actions(
            message, generation_variant_count=1, generation_browsed_index=0
        )
    ]

    assert "variant-previous" not in action_ids
    assert "variant-next" not in action_ids
    assert "keep" not in action_ids


def test_generation_variant_nav_visible_at_count_two():
    """Two-plus variants show `<`/`>`, gated by the GENERATION browsed index
    -- not by the message's (absent) text-sibling fields."""
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="[image] x")

    actions = {
        action.action_id: action
        for action in service.available_actions(
            message, generation_variant_count=2, generation_browsed_index=0
        )
    }

    assert "variant-previous" in actions
    assert "variant-next" in actions
    assert actions["variant-previous"].enabled is False
    assert actions["variant-next"].enabled is True


@pytest.mark.parametrize(
    ("browsed_index", "variant_count", "previous_enabled", "next_enabled"),
    [
        (0, 3, False, True),
        (1, 3, True, True),
        (2, 3, True, False),
    ],
)
def test_generation_variant_nav_boundary_enables(
    browsed_index: int,
    variant_count: int,
    previous_enabled: bool,
    next_enabled: bool,
):
    """Boundary-enable mirrors the text-sibling check exactly, but keyed off
    the generation browsed index/count instead of sibling_index/count."""
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="[image] x")

    actions = {
        action.action_id: action
        for action in service.available_actions(
            message,
            generation_variant_count=variant_count,
            generation_browsed_index=browsed_index,
        )
    }

    assert actions["variant-previous"].enabled is previous_enabled
    assert actions["variant-next"].enabled is next_enabled


@pytest.mark.parametrize("browsed_index", [0, 1, 2])
def test_keep_action_only_visible_when_browsed_away_from_canonical(browsed_index: int):
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="[image] x")

    action_ids = [
        action.action_id
        for action in service.available_actions(
            message, generation_variant_count=3, generation_browsed_index=browsed_index
        )
    ]

    assert ("keep" in action_ids) is (browsed_index != 0)


def test_generation_message_ignores_text_sibling_fields():
    """A generation message that (hypothetically) also carries stale
    text-sibling fields must still be gated by the generation kwargs --
    generation-variant gating takes precedence (spec §5.1/§7)."""
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="[image] x",
        sibling_index=0,
        sibling_count=1,  # would hide <> under the old sibling-only gate
    )

    actions = {
        action.action_id: action
        for action in service.available_actions(
            message, generation_variant_count=4, generation_browsed_index=2
        )
    }

    assert "variant-previous" in actions
    assert "variant-next" in actions
    assert actions["variant-previous"].enabled is True
    assert actions["variant-next"].enabled is True
    assert "keep" in actions


def test_generation_message_precedence_over_conflicting_sibling_state():
    """task-558: a stronger precedence pin than
    ``test_generation_message_ignores_text_sibling_fields`` above -- that
    test's ``sibling_count=1`` wouldn't trigger the old sibling-only
    ``elif message.sibling_count > 1`` branch anyway, so it can't actually
    distinguish "generation gating won" from "the old branch just didn't
    fire". Here the stale sibling fields (``sibling_index=2``,
    ``sibling_count=3``) would produce the OPPOSITE previous/next enabled
    states from the generation kwargs if sibling-count gating won instead
    of generation gating: sibling gating would enable previous (``2 > 0``)
    and disable next (``2 < 3 - 1`` is False); the generation kwargs
    (``generation_browsed_index=0`` of ``generation_variant_count=3``)
    disable previous and enable next.
    """
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="[image] x",
        sibling_index=2,
        sibling_count=3,
    )

    actions = {
        action.action_id: action
        for action in service.available_actions(
            message, generation_variant_count=3, generation_browsed_index=0
        )
    }

    assert "variant-previous" in actions
    assert "variant-next" in actions
    assert actions["variant-previous"].enabled is False
    assert actions["variant-next"].enabled is True


def test_generation_regenerate_stays_visible_and_enabled():
    """Regenerate (`♻`) stays visible on a generation message, still gated
    only by assistant-role as today (spec §7)."""
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="[image] x")

    actions = {
        action.action_id: action
        for action in service.available_actions(
            message, generation_variant_count=3, generation_browsed_index=1
        )
    }

    assert actions["regenerate"].enabled is True


def test_keep_action_dispatch_returns_completed_result():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT, content="[image] x", id="m1"
    )

    result = service.dispatch("keep", message)

    assert result.status == "completed"
    assert result.target_message_id == "m1"


def test_continue_action_targets_selected_variant_content():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT, content="first", id="m1"
    )
    message.variants = ConsoleVariantSet.from_contents(
        turn_id="turn-1",
        contents=["first", "second"],
        selected_index=1,
    )

    result = service.dispatch("continue", message)

    assert result.status == "continue_requested"
    assert result.target_message_id == "m1"
    assert result.target_content == "second"


# --- TASK-1: speak (TTS) action ------------------------------------------


def test_speak_action_present_for_completed_assistant_text():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="hello there",
    )

    action_ids = [action.action_id for action in service.available_actions(message)]

    assert "speak" in action_ids


@pytest.mark.parametrize(
    ("speech_state", "action_id", "enabled", "status_label"),
    [
        ("generating", "speak-stop", False, "Generating"),
        ("playing", "speak-stop", True, "Playing"),
        ("stopped", "speak", True, "Stopped"),
        ("failed", "speak", True, "Failed"),
    ],
)
def test_completed_assistant_header_has_canonical_speech_presentation(
    speech_state, action_id, enabled, status_label
):
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="hello there",
        status="complete",
        id="m1",
    )

    presentation = message_actions.resolve_console_header_speech(message, speech_state)

    assert presentation.action is not None
    assert presentation.action.action_id == action_id
    assert presentation.action.enabled is enabled
    assert presentation.status_label == status_label


def test_idle_header_never_hosts_speech_action():
    """Idle Speak lives in the selected action row, never the header."""
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="hello there",
        status="complete",
        id="m1",
    )

    for selected in (False, True):
        presentation = message_actions.resolve_console_header_speech(
            message, "idle", selected=selected
        )
        assert presentation.action is None
        assert presentation.status_label == ""


def test_playback_lifecycle_stays_visible_when_deselected():
    """Generating/playing controls must remain reachable after deselection."""
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="hello there",
        status="complete",
        id="m1",
    )

    for state, action_id in (("generating", "speak-stop"), ("playing", "speak-stop")):
        presentation = message_actions.resolve_console_header_speech(
            message, state, selected=False
        )
        assert presentation.action is not None
        assert presentation.action.action_id == action_id


def test_selected_action_row_includes_speak_and_swaps_to_stop():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="hello there",
        status="complete",
        id="m1",
    )

    def ids(speaking_message_id):
        return [
            action.action_id
            for action in service.selected_row_actions(
                message, speaking_message_id=speaking_message_id
            )
        ]

    idle_ids = ids(None)
    assert "speak" in idle_ids
    assert "speak-stop" not in idle_ids

    speaking_ids = ids("m1")
    assert "speak" not in speaking_ids
    assert "speak-stop" in speaking_ids
    assert set(speaking_ids) == set(id for id in idle_ids if id != "speak") | {
        "speak-stop"
    }

    other_ids = ids("some-other-message")
    assert "speak" in other_ids
    assert "speak-stop" not in other_ids


@pytest.mark.parametrize(
    "message",
    [
        ConsoleChatMessage(role=ConsoleMessageRole.USER, content="hello", id="u1"),
        ConsoleChatMessage(
            role=ConsoleMessageRole.ASSISTANT,
            content="partial",
            status="streaming",
            id="a1",
        ),
        ConsoleChatMessage(
            role=ConsoleMessageRole.ASSISTANT,
            content="   ",
            status="complete",
            id="a2",
        ),
    ],
)
def test_ineligible_message_header_has_no_speech_presentation(message):
    presentation = message_actions.resolve_console_header_speech(message, "idle")

    assert presentation.action is None
    assert presentation.status_label == ""


@pytest.mark.parametrize(
    "role",
    [
        ConsoleMessageRole.USER,
        ConsoleMessageRole.SYSTEM,
        ConsoleMessageRole.TOOL,
    ],
)
def test_speak_action_absent_for_non_assistant_text(role: ConsoleMessageRole):
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(role=role, content="hello there")

    action_ids = [action.action_id for action in service.available_actions(message)]

    assert "speak" not in action_ids


@pytest.mark.parametrize("status", ["pending", "streaming", "stopped", "failed"])
def test_speak_action_absent_for_incomplete_assistant_status(status):
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="partial answer",
        status=status,
    )

    action_ids = [action.action_id for action in service.available_actions(message)]

    assert "speak" not in action_ids


def test_speak_action_present_for_generation_card_marker_text():
    """A completed assistant generation card remains trusted assistant text."""
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT, content="[image] a red dragon"
    )

    action_ids = [
        action.action_id
        for action in service.available_actions(
            message, generation_variant_count=1, generation_browsed_index=0
        )
    ]

    assert "speak" in action_ids


def test_speak_action_absent_for_empty_content_message():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT, content="", status="pending"
    )

    action_ids = [action.action_id for action in service.available_actions(message)]

    assert "speak" not in action_ids


def test_speak_action_absent_for_whitespace_only_content_message():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(role=ConsoleMessageRole.USER, content="   ")

    action_ids = [action.action_id for action in service.available_actions(message)]

    assert "speak" not in action_ids


def test_speak_action_absent_for_failed_assistant_message():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT, content="partial answer", status="failed"
    )

    action_ids = [action.action_id for action in service.available_actions(message)]

    assert "speak" not in action_ids


def test_speak_action_absent_for_failed_user_message():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.USER, content="hello", status="failed"
    )

    action_ids = [action.action_id for action in service.available_actions(message)]

    assert "speak" not in action_ids


def test_speak_action_dispatch_returns_completed_result_with_message_content_and_id():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT, content="answer", id="m1"
    )

    result = service.dispatch("speak", message)

    assert result.status == "completed"
    assert result.target_message_id == "m1"
    assert result.target_content == "answer"


# --- task-559 unit 2: speak -> stop toggle --------------------------------


def test_speak_action_swaps_to_stop_when_message_is_speaking():
    """While THIS message is the one driving Console TTS, the row's 🔊
    speak action swaps to a ⏹ stop action in the same slot -- mirrors the
    generation card's browsed-index-driven action swap (Keep)."""
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT, content="answer", id="m1"
    )

    actions = service.available_actions(message, speaking_message_id="m1")

    action_ids = [action.action_id for action in actions]
    assert "speak" not in action_ids
    assert "speak-stop" in action_ids
    stop_action = next(a for a in actions if a.action_id == "speak-stop")
    assert stop_action.label == "⏹"
    assert stop_action.enabled is True
    # Row order is otherwise unchanged -- stop lands exactly where speak was.
    assert [a.action_id for a in actions] == [
        "copy",
        "speak-stop",
        "edit",
        "save-as",
        "fork",
        "regenerate",
        "continue",
        "feedback",
        "delete",
    ]


def test_speak_action_unaffected_when_a_different_message_is_speaking():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT, content="answer", id="m1"
    )

    actions = service.available_actions(message, speaking_message_id="other-message")

    action_ids = [action.action_id for action in actions]
    assert "speak" in action_ids
    assert "speak-stop" not in action_ids


def test_speak_action_unaffected_when_nothing_is_speaking():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT, content="answer", id="m1"
    )

    actions = service.available_actions(message, speaking_message_id=None)

    action_ids = [action.action_id for action in actions]
    assert "speak" in action_ids
    assert "speak-stop" not in action_ids


def test_speak_stop_absent_when_speak_itself_would_be_absent():
    """A failed message never shows speak -- so it must never show speak-stop
    either, even if (implausibly) it's the tracked speaking id."""
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="partial answer",
        status="failed",
        id="m1",
    )

    actions = service.available_actions(message, speaking_message_id="m1")

    action_ids = [action.action_id for action in actions]
    assert "speak" not in action_ids
    assert "speak-stop" not in action_ids


def test_speak_stop_action_dispatch_returns_completed_result():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT, content="answer", id="m1"
    )

    result = service.dispatch("speak-stop", message)

    assert result.status == "completed"
    assert result.target_message_id == "m1"


def test_original_attempt_action_is_explicit_and_precedes_regenerate():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="Repaired answer [S1]",
        id="assistant-repaired",
    )

    default_actions = service.available_actions(message)
    explicit_false_actions = service.available_actions(
        message,
        original_attempt_available=False,
    )
    available_actions = service.available_actions(
        message,
        original_attempt_available=True,
    )
    available_ids = [action.action_id for action in available_actions]

    assert all(
        action.action_id != "view-original-attempt" for action in default_actions
    )
    assert explicit_false_actions == default_actions
    assert available_ids.count("view-original-attempt") == 1
    assert available_ids.index("view-original-attempt") < available_ids.index(
        "regenerate"
    )
    assert "View original attempt" not in service.plain_action_labels(message)
    assert "View original attempt" not in service.plain_action_row(message)


@pytest.mark.parametrize(
    "message",
    (
        ConsoleChatMessage(
            role=ConsoleMessageRole.USER,
            content="question",
            id="user-message",
        ),
        ConsoleChatMessage(
            role=ConsoleMessageRole.SYSTEM,
            content="notice",
            id="system-message",
        ),
        ConsoleChatMessage(
            role=ConsoleMessageRole.ASSISTANT,
            content="failed",
            id="failed-assistant",
            status="failed",
        ),
    ),
)
def test_original_attempt_action_omits_ineligible_messages(message):
    actions = ConsoleMessageActionService().available_actions(
        message,
        original_attempt_available=True,
    )

    assert all(action.action_id != "view-original-attempt" for action in actions)


def test_original_attempt_dispatch_returns_only_safe_target():
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="Repaired answer [S1]",
        id="assistant-repaired",
    )

    result = ConsoleMessageActionService().dispatch(
        "view-original-attempt",
        message,
    )

    assert result.status == "completed"
    assert result.target_message_id == message.id
    assert result.target_content is None
    assert result.clipboard_text is None
    assert message.content not in result.visible_copy


def test_save_image_is_disabled_with_a_reason_in_a_temporary_chat():
    """The message-action row's Save Image writes a file -- blocked when
    temporary, and still enabled otherwise (the control)."""
    from tldw_chatbook.Chat.console_ephemeral import blocked_reason

    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="a picture",
        image_data=b"\x89PNG-bytes",
        image_mime_type="image/png",
    )

    blocked_actions = {
        action.action_id: action
        for action in ConsoleMessageActionService().available_actions(
            message, ephemeral=True
        )
    }
    save_image = blocked_actions["save-image"]
    assert save_image.enabled is False
    assert save_image.disabled_reason == blocked_reason("save-image", ephemeral=True)

    normal_actions = {
        action.action_id: action
        for action in ConsoleMessageActionService().available_actions(message)
    }
    assert normal_actions["save-image"].enabled is True
    assert normal_actions["save-image"].disabled_reason == ""


def _tool_output_action(message):
    return {
        action.action_id: action
        for action in ConsoleMessageActionService().available_actions(message)
    }.get("tool-output")


def test_diff_only_tool_marker_offers_expansion_labeled_diff():
    """TASK-1366: a file-write marker whose stripped result fit the preview
    (tool_output_full is None -- the common case) must still offer the
    expansion affordance, labeled for what it actually opens: the diff."""
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.TOOL,
        content="write_file → /tmp/a.py",
        tool_diff=("/tmp/a.py", "old\n", "new\n"),
    )

    action = _tool_output_action(message)

    assert action is not None, "diff-only marker must offer expansion"
    assert action.label == "Diff"


def test_full_output_tool_marker_keeps_full_output_label():
    """TASK-1860 copy is unchanged for a marker with hidden full text."""
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.TOOL,
        content="calculator → 42",
        tool_output_full="the whole untruncated result",
    )

    action = _tool_output_action(message)

    assert action is not None
    assert action.label == "Full output"


def test_tool_marker_with_full_output_and_diff_keeps_full_output_label():
    """Expansion shows both the full text and the diff; name the text."""
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.TOOL,
        content="write_file → /tmp/a.py",
        tool_output_full="the whole untruncated result",
        tool_diff=("/tmp/a.py", "old\n", "new\n"),
    )

    action = _tool_output_action(message)

    assert action is not None
    assert action.label == "Full output"


def test_plain_tool_marker_offers_no_expansion():
    """No hidden text and no diff: no dead affordance (TASK-1843 rule)."""
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.TOOL,
        content="calculator → 42",
    )

    assert _tool_output_action(message) is None


# --- task-2154.14 (DS-01): the action-row legend names glyphs in words ----


def test_action_row_guide_names_every_glyph_in_a_standard_row():
    """The legend under a selected row must decode each glyph-only button."""
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="answer")

    guide = action_row_guide(service.available_actions(message))

    assert guide == (
        "Guide: j/k select · c Copy · 🔊 Speak · e Edit · f Fork · r ♻ Regenerate · "
        "---> Continue · 👍/👎 Rate · 🗑 Delete · Esc clear"
    )


def test_action_row_guide_mirrors_the_rows_own_actions():
    """A row without Speak must not name a 🔊 the user cannot see."""
    service = ConsoleMessageActionService()
    user_message = ConsoleChatMessage(role=ConsoleMessageRole.USER, content="hi")

    guide = action_row_guide(service.available_actions(user_message))

    assert "🔊 Speak" not in guide
    assert "🗑 Delete" in guide
    assert guide.startswith("Guide: j/k select · ")
    assert guide.endswith(" · Esc clear")


def test_action_row_guide_follows_the_speak_stop_swap():
    """While the row shows ⏹, the legend says Stop speech -- not Speak."""
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT, content="answer", id="m1"
    )

    guide = action_row_guide(
        service.available_actions(message, speaking_message_id="m1")
    )

    assert "⏹ Stop speech" in guide
    assert "🔊 Speak" not in guide


def test_action_row_guide_names_variant_navigation_when_present():
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="answer",
        sibling_count=2,
        sibling_index=0,
    )

    guide = action_row_guide(service.available_actions(message))

    assert "</> Variants" in guide


def test_action_row_guide_without_glyph_actions_keeps_the_key_frame():
    guide = action_row_guide([])

    assert guide == "Guide: j/k select · Esc clear"


def test_plain_action_guide_matches_the_plain_action_rows_inputs():
    """Exports use the un-keyworded call, so the legend matches that row."""
    service = ConsoleMessageActionService()
    message = ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="answer")

    assert service.plain_action_guide(message) == action_row_guide(
        service.selected_row_actions(message)
    )


@pytest.mark.parametrize(
    "role",
    (ConsoleMessageRole.USER, ConsoleMessageRole.ASSISTANT),
)
def test_complete_conversation_rows_place_fork_immediately_before_regenerate(
    role: ConsoleMessageRole,
) -> None:
    message = ConsoleChatMessage(role=role, content="stable", id="message-1")

    action_ids = [
        action.action_id
        for action in ConsoleMessageActionService().available_actions(
            message,
            fork_eligibility=ConsoleForkEligibility(True),
        )
    ]

    assert "fork" in action_ids
    assert action_ids.index("fork") + 1 == action_ids.index("regenerate")


@pytest.mark.parametrize("status", ("stopped", "failed"))
def test_nonempty_partial_assistant_rows_can_fork(status: str) -> None:
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="partial answer",
        status=status,
        id="assistant-partial",
    )

    fork = next(
        action
        for action in ConsoleMessageActionService().available_actions(
            message,
            fork_eligibility=ConsoleForkEligibility(True),
        )
        if action.action_id == "fork"
    )

    assert fork.enabled is True
    assert fork.disabled_reason == ""


@pytest.mark.parametrize(
    ("status", "content"),
    (
        ("pending", ""),
        ("streaming", "partial"),
        ("discarded", "discarded response"),
        ("failed", "   "),
    ),
)
def test_unstable_assistant_rows_expose_a_fork_disabled_reason(
    status: str,
    content: str,
) -> None:
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content=content,
        status=status,
        id="assistant-unstable",
    )

    fork = next(
        action
        for action in ConsoleMessageActionService().available_actions(
            message,
            fork_eligibility=ConsoleForkEligibility(True),
        )
        if action.action_id == "fork"
    )

    assert fork.enabled is False
    assert fork.disabled_reason


def test_store_derived_fork_reason_controls_durable_eligibility() -> None:
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="saved boundary",
        persisted_message_id="persisted-boundary",
        id="assistant-1",
    )
    unsaved_prefix = ConsoleForkEligibility(
        False,
        "This message has not been saved yet. Try Fork again after it is saved.",
    )

    denied = next(
        action
        for action in ConsoleMessageActionService().available_actions(
            message,
            fork_eligibility=unsaved_prefix,
        )
        if action.action_id == "fork"
    )
    allowed_without_presentation_id = next(
        action
        for action in ConsoleMessageActionService().available_actions(
            replace(message, persisted_message_id=None),
            fork_eligibility=ConsoleForkEligibility(True),
        )
        if action.action_id == "fork"
    )

    assert denied.enabled is False
    assert denied.disabled_reason == unsaved_prefix.reason
    assert allowed_without_presentation_id.enabled is True


def test_action_groups_separate_primary_overflow_and_media_actions() -> None:
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="[image] answer",
        image_data=b"png",
        image_mime_type="image/png",
        id="assistant-image",
    )

    groups = ConsoleMessageActionService().action_groups(
        message,
        generation_variant_count=2,
        generation_browsed_index=0,
        fork_eligibility=ConsoleForkEligibility(True),
    )

    assert [action.action_id for action in groups.primary] == [
        "copy",
        "speak",
        "edit",
        "fork",
        "regenerate",
        "continue",
        "more",
    ]
    assert [action.label for action in groups.overflow] == [
        "Save as…",
        "Helpful",
        "Not helpful",
        "Delete",
    ]
    assert [action.action_id for action in groups.media] == [
        "variant-previous",
        "variant-next",
        "toggle-image-view",
        "save-image",
    ]
    assert not (
        {action.action_id for action in groups.primary}
        & {action.action_id for action in groups.media}
    )
    assert not (
        {action.action_id for action in groups.overflow}
        & {action.action_id for action in groups.media}
    )


@pytest.mark.parametrize(
    ("message", "expected_primary"),
    (
        (
            ConsoleChatMessage(
                role=ConsoleMessageRole.USER,
                content="question",
                id="user-complete",
            ),
            ("copy", "edit", "fork", "regenerate", "continue", "more"),
        ),
        (
            ConsoleChatMessage(
                role=ConsoleMessageRole.ASSISTANT,
                content="partial answer",
                status="stopped",
                id="assistant-stopped",
            ),
            ("copy", "edit", "fork", "regenerate", "continue", "more"),
        ),
    ),
)
def test_user_and_stopped_assistant_action_groups_are_exact(
    message,
    expected_primary,
) -> None:
    groups = ConsoleMessageActionService().action_groups(
        message,
        fork_eligibility=ConsoleForkEligibility(True),
    )

    assert tuple(action.action_id for action in groups.primary) == expected_primary
    assert tuple(action.action_id for action in groups.overflow) == (
        "save-as",
        "feedback-up",
        "feedback-down",
        "delete",
    )
    assert groups.media == ()


def test_video_actions_are_an_exact_separate_media_group() -> None:
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="generated video",
        video_metadata=VideoGenerationMetadata(
            name="clip",
            prompt="waves",
            backend="local",
        ),
        id="assistant-video",
    )

    groups = ConsoleMessageActionService().action_groups(
        message,
        video_file_available=True,
        fork_eligibility=ConsoleForkEligibility(True),
    )

    assert tuple(action.action_id for action in groups.primary) == (
        "copy",
        "speak",
        "edit",
        "fork",
        "regenerate",
        "continue",
        "more",
    )
    assert tuple(action.action_id for action in groups.overflow) == (
        "save-as",
        "feedback-up",
        "feedback-down",
        "delete",
    )
    assert tuple(action.action_id for action in groups.media) == (
        "video-play",
        "video-save-copy",
    )


def test_action_groups_preserve_the_speak_stop_slot() -> None:
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="answer",
        id="speaking-message",
    )

    groups = ConsoleMessageActionService().action_groups(
        message,
        speaking_message_id=message.id,
        fork_eligibility=ConsoleForkEligibility(True),
    )

    assert [action.action_id for action in groups.primary][:3] == [
        "copy",
        "speak-stop",
        "edit",
    ]


@pytest.mark.parametrize(
    "message",
    (
        ConsoleChatMessage(
            role=ConsoleMessageRole.TOOL,
            content="tool preview",
            tool_output_full="full tool output",
            id="tool-row",
        ),
        ConsoleChatMessage(
            role=ConsoleMessageRole.TOOL,
            content="working",
            activity_presentation=ConsoleActivityPresentation(
                "thinking", "Thinking", "done"
            ),
            id="activity-row",
        ),
    ),
)
def test_tool_and_activity_rows_never_expose_fork_or_more(message) -> None:
    groups = ConsoleMessageActionService().action_groups(
        message,
        fork_eligibility=ConsoleForkEligibility(True),
    )
    action_ids = {
        action.action_id
        for group in (groups.primary, groups.overflow, groups.media)
        for action in group
    }

    assert "fork" not in action_ids
    assert "more" not in action_ids
    if message.tool_output_full:
        assert [action.action_id for action in groups.primary] == ["tool-output"]


def test_assistant_activity_row_keeps_only_its_specialized_action() -> None:
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="review complete",
        change_review_run_id="review-1",
        activity_presentation=ConsoleActivityPresentation(
            "changes", "Changes", "done"
        ),
        id="assistant-activity-row",
    )

    groups = ConsoleMessageActionService().action_groups(
        message,
        fork_eligibility=ConsoleForkEligibility(True),
    )

    assert tuple(action.action_id for action in groups.primary) == ("review-changes",)
    assert groups.overflow == ()
    assert groups.media == ()


def test_fork_dispatch_requests_the_exact_message() -> None:
    message = ConsoleChatMessage(
        role=ConsoleMessageRole.ASSISTANT,
        content="answer",
        id="fork-boundary",
    )

    result = ConsoleMessageActionService().dispatch("fork", message)

    assert result.status == "fork_requested"
    assert result.target_message_id == "fork-boundary"

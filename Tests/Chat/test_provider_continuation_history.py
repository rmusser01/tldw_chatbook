"""Atomic provider-continuation history grouping and budgeting."""

from __future__ import annotations

import copy
import dataclasses

import pytest

from tldw_chatbook.Chat.console_history_budget import (
    ProviderContinuationSidecar,
    count_provider_continuation_tokens,
    provider_continuation_owner_groups,
)
from tldw_chatbook.Chat.provider_continuation import (
    ContinuationConflictError,
    ContinuationRestoreTarget,
    parse_provider_continuation_json,
)


def _checkpoint(
    *,
    call_id: str = "call_1",
    include_prior_round: bool = False,
):
    rounds = []
    if include_prior_round:
        rounds.append(
            {
                "assistant_content": "earlier answer",
                "reasoning_blocks": ["EARLIER-PRIVATE-REASONING"],
                "calls": [
                    {
                        "call_id": f"{call_id}_prior",
                        "name": "lookup",
                        "arguments": "{}",
                        "state": "completed",
                        "result": "earlier result",
                    }
                ],
            }
        )
    rounds.append(
        {
            "assistant_content": "visible answer",
            "reasoning_blocks": ["PRIVATE-REASONING-CANARY"],
            "calls": [
                {
                    "call_id": call_id,
                    "name": "lookup",
                    "arguments": '{"query":"PRIVATE-ARGUMENT-CANARY"}',
                    "state": "completed",
                    "result": "PRIVATE-RESULT-CANARY",
                }
            ],
        }
    )
    return parse_provider_continuation_json(
        {
            "schema_version": 1,
            "checkpoint_revision": 1,
            "provider": "deepseek",
            "protocol": "responses",
            "model": "deepseek-v4-flash",
            "api_base_url": "https://api.deepseek.com/v1",
            "state": "complete",
            "rounds": rounds,
        }
    )


def _target() -> ContinuationRestoreTarget:
    return ContinuationRestoreTarget(
        provider="deepseek",
        protocol="responses",
        model="deepseek-v4-flash",
        api_base_url="https://api.deepseek.com/v1",
    )


def test_grouping_validates_restore_and_preserves_active_owner_order() -> None:
    checkpoint = _checkpoint(include_prior_round=True)
    messages = [
        {"id": "u1", "role": "user", "content": "first"},
        {
            "id": "a1",
            "role": "assistant",
            "content": "visible answer",
            "provider_continuation": checkpoint,
        },
        {"id": "u2", "role": "user", "content": "current"},
    ]
    before = copy.deepcopy(messages)

    groups = provider_continuation_owner_groups(messages, target=_target())

    assert [group.owner_message_id for group in groups] == ["a1"]
    assert groups[0].rounds == checkpoint.rounds
    assert [round_.calls[0].call_id for round_ in groups[0].rounds] == [
        "call_1_prior",
        "call_1",
    ]
    assert messages == before
    assert all("tool_calls" not in vars(round_) for round_ in groups[0].rounds)

    for field, value in (
        ("provider", "zai"),
        ("protocol", "chat_completions"),
        ("model", "different-model"),
        ("api_base_url", "https://api.deepseek.com/v2"),
    ):
        assert (
            provider_continuation_owner_groups(
                messages,
                target=dataclasses.replace(_target(), **{field: value}),
            )
            == ()
        )


def test_active_mismatched_owner_still_fails_closed() -> None:
    active = parse_provider_continuation_json(
        {
            "schema_version": 1,
            "checkpoint_revision": 1,
            "provider": "deepseek",
            "protocol": "responses",
            "model": "deepseek-v4-flash",
            "api_base_url": "https://api.deepseek.com/v1",
            "state": "active",
            "rounds": [
                {
                    "assistant_content": "",
                    "reasoning_blocks": [],
                    "calls": [
                        {
                            "call_id": "call_active",
                            "name": "lookup",
                            "arguments": "{}",
                            "state": "pending",
                        }
                    ],
                }
            ],
        }
    )

    with pytest.raises(ContinuationConflictError, match="restore target mismatch"):
        provider_continuation_owner_groups(
            (ProviderContinuationSidecar("a1", active),),
            target=dataclasses.replace(_target(), provider="moonshot"),
        )


def test_unrelated_provider_receives_no_private_owner_group() -> None:
    messages = [
        {"id": "u1", "role": "user", "content": "first"},
        {
            "id": "a1",
            "role": "assistant",
            "content": "visible answer",
            "provider_continuation": _checkpoint(),
        },
        {"id": "u2", "role": "user", "content": "current"},
    ]

    assert (
        provider_continuation_owner_groups(
            messages,
            target=ContinuationRestoreTarget(
                provider="openai",
                protocol="chat_completions",
                model="gpt-5.6",
                api_base_url="https://api.openai.com/v1",
            ),
        )
        == ()
    )


def test_private_round_fields_are_counted_without_provider_wire_rows() -> None:
    group = provider_continuation_owner_groups(
        [
            {
                "id": "a1",
                "role": "assistant",
                "content": "visible answer",
                "provider_continuation": _checkpoint(),
            }
        ],
        target=_target(),
    )[0]
    counted_rows = []

    def capture_count(messages, _model):
        counted_rows.extend(messages)
        return sum(len(str(message.get("content", ""))) for message in messages)

    private_tokens = count_provider_continuation_tokens(
        group, model="m", count_fn=capture_count
    )

    counted = str(counted_rows[0]["content"])
    assert private_tokens == len(counted)
    assert "PRIVATE-REASONING-CANARY" in counted
    assert "PRIVATE-ARGUMENT-CANARY" in counted
    assert "PRIVATE-RESULT-CANARY" in counted
    assert "tool_calls" not in counted_rows[0]


def test_groups_only_selected_active_branch_and_skips_deleted_or_malformed_owners() -> (
    None
):
    selected = [
        {"id": "u1", "role": "user", "content": "question"},
        {
            "id": "a-selected",
            "role": "assistant",
            "content": "visible answer",
            "provider_continuation": _checkpoint(call_id="selected_call"),
        },
        {
            "id": "a-deleted",
            "role": "assistant",
            "content": "deleted sibling",
            "deleted": True,
            "provider_continuation": _checkpoint(call_id="deleted_call"),
        },
        {
            "id": "a-malformed",
            "role": "assistant",
            "content": "legacy visible",
            "provider_continuation": {"schema_version": 999},
        },
    ]

    groups = provider_continuation_owner_groups(selected, target=_target())

    assert [group.owner_message_id for group in groups] == ["a-selected"]


@pytest.mark.parametrize(
    ("deleted", "included"),
    [(True, False), (1, False), (False, True), (0, True), ("1", True), (2, True)],
)
def test_deleted_owner_accepts_only_exact_sqlite_boolean_encodings(
    deleted, included
) -> None:
    groups = provider_continuation_owner_groups(
        [
            {
                "id": "a1",
                "role": "assistant",
                "content": "visible",
                "deleted": deleted,
                "provider_continuation": _checkpoint(),
            }
        ],
        target=_target(),
    )

    assert bool(groups) is included

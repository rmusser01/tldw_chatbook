"""Incoming continuation ownership cannot disappear during historical merging."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_completed_replay import completed_group
from tldw_chatbook.Chat.console_agent_bridge import _StreamingModelAdapter
from tldw_chatbook.Chat.console_history_budget import ProviderContinuationSidecar
from tldw_chatbook.Chat.console_prepared_request import CONTINUATION_OWNER_KEY
from tldw_chatbook.Chat.provider_continuation import ContinuationRestoreTarget

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]


class PreparedReached(Exception):
    pass


def adapter_case(with_history):
    group = completed_group()
    prepared = []

    class Gateway:
        def prepare_chat_request(self, resolution, semantic, **kwargs):
            groups = (
                (
                    tuple(
                        g
                        for unit in semantic.compactable
                        for g in unit.continuation_groups
                    )
                    + semantic.active_continuation_groups
                )
                if hasattr(semantic, "compactable")
                else ()
            )
            prepared.append(groups)
            raise PreparedReached

    adapter = _StreamingModelAdapter(
        store=SimpleNamespace(session_id_for_message=lambda _: "session"),
        provider_gateway=Gateway(),
        resolution=SimpleNamespace(provider="custom-hosted", model="synthetic"),
        assistant_message_id="current",
        should_cancel=lambda: False,
        loop=asyncio.get_running_loop(),
        native_tools=True,
        continuation_sidecar=(ProviderContinuationSidecar("owner", group.checkpoint),)
        if with_history
        else (),
        continuation_target=ContinuationRestoreTarget(
            provider="custom-hosted",
            protocol="chat_completions",
            model="synthetic",
            api_base_url=group.checkpoint.api_base_url,
            selected_provider=group.checkpoint.selected_provider,
        ),
        continuation_owner_key=CONTINUATION_OWNER_KEY,
    )
    return adapter, group, prepared


@pytest.mark.parametrize("with_history", [False, True])
@pytest.mark.parametrize("invalid", ["orphan", "duplicate_group", "duplicate_owner"])
async def test_supplied_group_owner_must_be_unique_before_merge(with_history, invalid):
    adapter, group, prepared = adapter_case(with_history)
    row = {
        "role": "assistant",
        "content": "Final answer",
        CONTINUATION_OWNER_KEY: "current",
    }
    groups = (replace(group, owner_message_id="current"),)
    rows = [{"role": "user", "content": "new"}, row]
    if invalid == "orphan":
        groups = (replace(group, owner_message_id="orphan"),)
    elif invalid == "duplicate_group":
        groups *= 2
    else:
        rows.append(dict(row))
    with pytest.raises(
        ValueError,
        match="Continuation owner IDs must be unique|Every continuation group must attach",
    ):
        await asyncio.to_thread(
            adapter.chat_call, messages_payload=rows, continuation_groups=groups
        )
    assert prepared == []


@pytest.mark.parametrize("with_history", [False, True])
async def test_unselected_constructor_history_does_not_hide_valid_current_group(
    with_history,
):
    adapter, group, prepared = adapter_case(with_history)
    current = replace(group, owner_message_id="current")
    with pytest.raises(PreparedReached):
        await asyncio.to_thread(
            adapter.chat_call,
            messages_payload=[
                {"role": "user", "content": "new"},
                {
                    "role": "assistant",
                    "content": "Final answer",
                    CONTINUATION_OWNER_KEY: "current",
                },
            ],
            continuation_groups=(current,),
        )
    assert prepared == [(current,)]


async def test_exact_current_group_takes_precedence_over_same_owner_history():
    adapter, group, prepared = adapter_case(True)
    current = replace(
        group, checkpoint=replace(group.checkpoint, checkpoint_revision=4)
    )
    with pytest.raises(PreparedReached):
        await asyncio.to_thread(
            adapter.chat_call,
            messages_payload=[
                {"role": "user", "content": "new"},
                {
                    "role": "assistant",
                    "content": "Final answer",
                    CONTINUATION_OWNER_KEY: "owner",
                },
            ],
            continuation_groups=(current,),
        )
    assert prepared == [(current,)]


async def test_unselected_constructor_history_without_current_group_is_omitted():
    adapter, _group, prepared = adapter_case(True)
    with pytest.raises(PreparedReached):
        await asyncio.to_thread(
            adapter.chat_call,
            messages_payload=[{"role": "user", "content": "new"}],
            continuation_groups=(),
        )
    assert prepared == [()]

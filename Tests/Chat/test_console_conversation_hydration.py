"""One hydration policy, two callers (task-15860 Task 6).

`ChatScreen._resume_console_workspace_conversation` used to be the ONLY
code that could turn a persisted conversation into a Console session, and
it interleaved that policy with a screen's own work. A wake fired at
launch needs a session for a conversation nobody has opened, so the
session-producing half moved to
`Chat/console_conversation_hydration.py` -- a pure refactor, pinned here
in both directions:

* **characterization** -- the screen's `_console_messages_from_conversation
  _tree` seam (eight test files call it by name) still produces the same
  flattened tree, branches, parenthood and dropped-empty-row transparency;
* **equivalence** -- for one fixture conversation, the session a LAUNCH
  hydrates headlessly and the session the SCREEN resumes agree field for
  field on everything that reaches a provider payload.

The equivalence test is the one that matters: it is what stops the launch
path quietly drifting into a second, worse resume policy -- which is the
failure the plan named ("rather than duplicating it").
"""

from __future__ import annotations

import asyncio
import json

import pytest

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_fleet_wake_wiring import _attach_real_dbs
from Tests.UI.test_console_native_chat_flow import (
    StaticConversationTreeService,
    _configure_native_ready_console,
)
from tldw_chatbook.Chat.console_conversation_hydration import (
    ConsoleGenerationSettingsHydration,
    console_messages_from_conversation_tree,
    hydrate_console_generation_settings,
    hydrate_console_session,
    load_console_conversation_tree,
)
from tldw_chatbook.Chat.console_generation_settings_metadata import (
    CONSOLE_GENERATION_SETTINGS_METADATA_KEY,
    ConsoleGenerationSettingsReadStatus,
    ConsoleGenerationSettingsSnapshot,
    merge_console_generation_settings,
)
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    build_console_settings_readiness,
    default_console_session_settings,
)
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen


CONVERSATION_ID = "conv-fixture"

#: Deliberately awkward: two branches off one root, a truly-empty node in
#: the middle of a branch (its child must re-parent through it), a system
#: prompt, roleplay metadata and a pinned prefill.
FIXTURE_TREE = {
    "conversation": {
        "id": CONVERSATION_ID,
        "title": "Fixture conversation",
        "system_prompt": "  you are a careful assistant\n",
        "workspace_id": "ws-fixture",
        "runtime_backend": "local",
        "assistant_kind": "generic",
        "assistant_id": "console",
        "metadata": {
            "console_roleplay_context": {
                "version": 1,
                "user_name_override": "Robert",
                "character_system_template": "You are {{char}}.",
            },
            "pinned_response_prefill": "Certainly,",
        },
    },
    "root_threads": [
        {
            "id": "m1",
            "sender": "user",
            "content": "first user message",
            "children": [
                {
                    "id": "m2",
                    "sender": "assistant",
                    "content": "branch A reply",
                    "children": [
                        {
                            "id": "m3-empty",
                            "sender": "assistant",
                            "content": "",
                            "children": [
                                {
                                    "id": "m4",
                                    "sender": "user",
                                    "content": "after an empty row",
                                    "children": [],
                                }
                            ],
                        }
                    ],
                },
                {
                    "id": "m5",
                    "sender": "assistant",
                    "content": "branch B reply",
                    "children": [],
                },
            ],
        }
    ],
}


def _fixture_app(tmp_path):
    app = _build_test_app()
    _attach_real_dbs(app, tmp_path)
    _configure_native_ready_console(app)
    app.app_config.setdefault("console", {})["agent_runtime"] = False
    if app.chachanotes_db.get_conversation_by_id(CONVERSATION_ID) is None:
        app.chachanotes_db.add_conversation(
            {"id": CONVERSATION_ID, "title": "Fixture conversation"}
        )
    app.chat_conversation_scope_service = StaticConversationTreeService(
        {CONVERSATION_ID: FIXTURE_TREE}
    )
    return app


def _message_shape(store, session_id):
    return [
        (
            m.role.value,
            m.content,
            m.persisted_message_id,
            m.parent_message_id,
        )
        for m in store.messages_for_session(session_id)
    ]


def test_the_screen_tree_walk_still_flattens_every_branch(tmp_path):
    """Characterization: the screen seam eight test files call by name.

    Asserts the three properties the walk owns -- ALL branches (not the
    latest), parenthood taken from NESTING, and an empty row dropped but
    transparent to its children -- so a refactor that quietly kept only the
    active path cannot pass.
    """
    app = _fixture_app(tmp_path)
    screen = ChatScreen(app)
    messages = screen._console_messages_from_conversation_tree(FIXTURE_TREE)

    shape = [(m.persisted_message_id, m.parent_message_id) for m in messages]
    assert shape == [
        ("m1", None),
        ("m2", "m1"),
        ("m4", "m2"),
        ("m5", "m1"),
    ], (
        "the walk must keep BOTH branches, take parenthood from nesting, and "
        f"re-parent through the dropped empty row; got {shape}"
    )
    assert [m.role.value for m in messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]

    # `ConsoleChatMessage.id` is a per-instance uuid, so compare the fields
    # the walk actually decides rather than object identity.
    def _walk_shape(built):
        return [
            (m.role.value, m.content, m.persisted_message_id, m.parent_message_id)
            for m in built
        ]

    assert _walk_shape(messages) == _walk_shape(
        console_messages_from_conversation_tree(FIXTURE_TREE, db=app.chachanotes_db)
    ), "the screen seam and the module function disagree"


@pytest.mark.asyncio
async def test_a_launch_hydrated_session_matches_a_screen_resumed_one(tmp_path):
    """The equivalence pin: both callers, one fixture, identical sessions.

    Everything that can reach a provider payload is compared -- the whole
    message tree with its parenthood, the durable identity fields, the
    roleplay overlay, and the full settings snapshot.

    The one field that legitimately differs is named rather than skipped:
    the session's own `id` is a fresh uuid per session.
    """
    screen_app = _fixture_app(tmp_path)
    screen = ChatScreen(screen_app)
    screen._provider_readiness_app_config = lambda: screen_app.app_config
    async with screen_app.run_test(size=(160, 48)) as pilot:
        await screen_app.push_screen(screen)
        screen_app._initial_screen_pushed = True
        await pilot.pause()
        resumed = await screen._workspace._resume_console_workspace_conversation(
            CONVERSATION_ID
        )
        assert resumed is True, f"the screen resume did not succeed: {resumed!r}"
        screen_store = screen._console_chat_store
        screen_session = next(
            s
            for s in screen_store.sessions()
            if s.persisted_conversation_id == CONVERSATION_ID
        )
        screen_shape = _message_shape(screen_store, screen_session.id)

    # A second app, no screen at all: the launch shape.
    launch_app = _fixture_app(tmp_path)
    launch_store = launch_app.console_runtime.ensure_chat_store()
    tree = await load_console_conversation_tree(launch_app, CONVERSATION_ID)
    assert tree is not None
    conversation = tree["conversation"]
    hydration = hydrate_console_generation_settings(
        launch_app.app_config,
        conversation,
    )
    launch_session = await hydrate_console_session(
        app=launch_app,
        store=launch_store,
        conversation_id=CONVERSATION_ID,
        tree=tree,
        settings=hydration.settings,
        generation_durable_snapshot=hydration.durable_snapshot,
        generation_metadata_status=hydration.metadata_status,
    )

    assert _message_shape(launch_store, launch_session.id) == screen_shape, (
        "the launch-hydrated transcript differs from the screen-resumed one"
    )
    for field in (
        "title",
        "workspace_id",
        "persisted_conversation_id",
        "runtime_backend",
        "assistant_kind",
        "assistant_id",
        "assistant_authority_id",
        "character_id",
        "user_display_name_override",
        "character_system_template",
    ):
        assert getattr(launch_session, field) == getattr(screen_session, field), (
            f"session field {field!r} differs: "
            f"launch={getattr(launch_session, field)!r} "
            f"screen={getattr(screen_session, field)!r}"
        )
    assert launch_session.settings == screen_session.settings, (
        "the settings snapshots differ:\n"
        f"launch={launch_session.settings!r}\nscreen={screen_session.settings!r}"
    )
    assert launch_session.settings is not None
    assert launch_session.settings.system_prompt == (
        "  you are a careful assistant\n"
    ), (
        "the saved system prompt must be restored VERBATIM -- the comparison "
        "above is worthless if both sides restored nothing"
    )
    assert launch_session.user_display_name_override == "Robert", (
        "the roleplay overlay never reached either session, so comparing them "
        "proved nothing"
    )
    assert launch_session.id != screen_session.id, (
        "session ids are per-session uuids; equal ids would mean the two "
        "stores are the same object and this test is not comparing two callers"
    )


@pytest.mark.asyncio
async def test_workspace_resume_rejects_legacy_plain_settings_accessor(tmp_path):
    """Workspace resume requires the typed generation hydration contract."""
    app = _fixture_app(tmp_path)
    screen = ChatScreen(app)
    screen._workspace._session_settings_for_resume_accessor = (
        lambda _conversation: ConsoleSessionSettings(
            provider="openai",
            model="legacy-plain-settings",
        )
    )

    async with app.run_test(size=(160, 48)) as pilot:
        await app.push_screen(screen)
        app._initial_screen_pushed = True
        await pilot.pause()
        with pytest.raises(TypeError, match="ConsoleGenerationSettingsHydration"):
            await screen._workspace._resume_console_workspace_conversation(
                CONVERSATION_ID
            )


@pytest.mark.asyncio
async def test_production_hydration_never_activates_placeholder_authority(
    tmp_path, monkeypatch
):
    app = _fixture_app(tmp_path)
    store = app.console_runtime.ensure_chat_store()
    prior = store.create_session(title="Prior")
    observed = []
    original_hydrate = store.hydrate_session_library_policy

    async def observe_before_activation(session_id):
        observed.append((store.active_session_id, session_id))
        await asyncio.sleep(0)
        observed.append((store.active_session_id, session_id))
        return await original_hydrate(session_id)

    monkeypatch.setattr(
        store, "hydrate_session_library_policy", observe_before_activation
    )
    session = await hydrate_console_session(
        app=app,
        store=store,
        conversation_id=CONVERSATION_ID,
        tree=FIXTURE_TREE,
        settings=default_console_session_settings(app.app_config),
        target_scope_type="global",
    )

    assert observed == [(prior.id, session.id), (prior.id, session.id)]
    assert session.library_policy_hydrated is True
    assert store.active_session_id == session.id


def _saved_generation_snapshot() -> ConsoleGenerationSettingsSnapshot:
    return ConsoleGenerationSettingsSnapshot(
        provider="anthropic",
        model="claude-saved",
        temperature=0.2,
        top_p=0.6,
        min_p=0.05,
        top_k=25,
        max_tokens=3072,
        seed=17,
        presence_penalty=0.3,
        frequency_penalty=-0.3,
        reasoning_effort="high",
        reasoning_summary="detailed",
        verbosity="low",
        thinking_effort="medium",
        thinking_budget_tokens=8192,
        streaming=False,
    )


def test_screen_resume_hydration_ignores_active_session_provider_and_model() -> None:
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {"provider": "openai", "model": "global-model"},
        "api_settings": {
            "anthropic": {
                "api_key": "test-key",
                "api_url": "https://anthropic-current.example/v1",
            },
            "local_llamacpp": {
                "api_url": "http://127.0.0.1:9099",
                "model": "active-model",
            },
        },
    }
    screen = ChatScreen(app)
    store = screen._ensure_console_chat_store()
    store.create_session(
        settings=ConsoleSessionSettings(
            provider="local_llamacpp",
            model="active-model",
            temperature=1.4,
        )
    )
    snapshot = _saved_generation_snapshot()
    conversation = {
        "metadata": merge_console_generation_settings({}, snapshot),
        "system_prompt": "Saved system prompt",
    }

    hydration = screen._session._console_session_settings_for_resume(conversation)

    assert isinstance(hydration, ConsoleGenerationSettingsHydration)
    assert hydration.settings.provider == "anthropic"
    assert hydration.settings.model == "claude-saved"
    assert hydration.settings.temperature == 0.2
    assert hydration.settings.base_url == "https://anthropic-current.example/v1"
    assert hydration.settings.system_prompt == "Saved system prompt"
    assert hydration.durable_snapshot == snapshot
    assert hydration.metadata_status is ConsoleGenerationSettingsReadStatus.VALID


@pytest.mark.asyncio
async def test_resume_session_threads_generation_hydration_into_store(tmp_path) -> None:
    app = _fixture_app(tmp_path)
    store = app.console_runtime.ensure_chat_store()
    snapshot = _saved_generation_snapshot()
    conversation = dict(FIXTURE_TREE["conversation"])
    conversation["metadata"] = merge_console_generation_settings(
        conversation["metadata"], snapshot
    )
    tree = dict(FIXTURE_TREE)
    tree["conversation"] = conversation
    hydration = hydrate_console_generation_settings(app.app_config, conversation)

    session = await hydrate_console_session(
        app=app,
        store=store,
        conversation_id=CONVERSATION_ID,
        tree=tree,
        settings=hydration.settings,
        generation_durable_snapshot=hydration.durable_snapshot,
        generation_metadata_status=hydration.metadata_status,
    )

    assert session.settings == hydration.settings
    assert session.generation_durable_snapshot == snapshot
    assert (
        session.generation_metadata_status
        is ConsoleGenerationSettingsReadStatus.VALID
    )


def test_resume_keeps_missing_catalog_custom_model_and_unconfigured_provider() -> None:
    snapshot = ConsoleGenerationSettingsSnapshot(
        provider="anthropic",
        model="vendor/missing:custom-model",
        temperature=0.41,
        top_p=0.82,
        min_p=None,
        top_k=None,
        max_tokens=None,
        seed=None,
        presence_penalty=None,
        frequency_penalty=None,
        reasoning_effort=None,
        reasoning_summary=None,
        verbosity=None,
        thinking_effort=None,
        thinking_budget_tokens=None,
        streaming=True,
    )
    conversation = {
        "metadata": merge_console_generation_settings({}, snapshot),
    }
    app_config = {
        "chat_defaults": {"provider": "openai", "model": "global-model"},
        "api_settings": {"openai": {"api_key": "test-key"}},
    }

    hydration = hydrate_console_generation_settings(app_config, conversation)
    readiness = build_console_settings_readiness(
        hydration.settings,
        app_config=app_config,
        environ={},
    )

    assert hydration.settings.provider == "anthropic"
    assert hydration.settings.model == "vendor/missing:custom-model"
    assert readiness.native_send_supported is False


def test_generation_hydration_enforces_durable_snapshot_status_invariant() -> None:
    settings = ConsoleSessionSettings(provider="openai")
    snapshot = _saved_generation_snapshot()

    with pytest.raises(ValueError):
        ConsoleGenerationSettingsHydration(
            settings=settings,
            durable_snapshot=None,
            metadata_status=ConsoleGenerationSettingsReadStatus.VALID,
        )
    with pytest.raises(ValueError):
        ConsoleGenerationSettingsHydration(
            settings=settings,
            durable_snapshot=snapshot,
            metadata_status=ConsoleGenerationSettingsReadStatus.INVALID,
        )

    assert (
        ConsoleGenerationSettingsHydration(
            settings=settings,
            durable_snapshot=snapshot,
            metadata_status=ConsoleGenerationSettingsReadStatus.VALID,
        ).durable_snapshot
        == snapshot
    )
    assert (
        ConsoleGenerationSettingsHydration(
            settings=settings,
            durable_snapshot=None,
            metadata_status=ConsoleGenerationSettingsReadStatus.ABSENT,
        ).durable_snapshot
        is None
    )


def test_generation_hydration_rebases_saved_target_and_keeps_existing_owners() -> None:
    snapshot = _saved_generation_snapshot()
    metadata = merge_console_generation_settings(
        {"pinned_response_prefill": "Certainly,", "sibling": {"keep": True}},
        snapshot,
    )
    conversation = {
        "system_prompt": "  keep exact formatting\n",
        "metadata": json.dumps(metadata, sort_keys=True),
    }
    app_config = {
        "chat_defaults": {
            "provider": "openai",
            "model": "global-model",
            "temperature": 1.7,
        },
        "api_settings": {
            "anthropic": {
                "api_url": "https://current-config.example/v1",
                "model": "provider-fallback",
                "model_defaults": {
                    "claude-saved": {
                        "temperature": 0.9,
                        "top_p": 0.95,
                    }
                },
            }
        },
    }
    before_metadata = conversation["metadata"]

    hydration = hydrate_console_generation_settings(app_config, conversation)

    assert hydration.metadata_status is ConsoleGenerationSettingsReadStatus.VALID
    assert hydration.durable_snapshot == snapshot
    assert hydration.settings.provider == snapshot.provider
    assert hydration.settings.model == snapshot.model
    assert hydration.settings.source == "user"
    for field in (
        "temperature",
        "top_p",
        "min_p",
        "top_k",
        "max_tokens",
        "seed",
        "presence_penalty",
        "frequency_penalty",
        "reasoning_effort",
        "reasoning_summary",
        "verbosity",
        "thinking_effort",
        "thinking_budget_tokens",
        "streaming",
    ):
        assert getattr(hydration.settings, field) == getattr(snapshot, field)
    assert hydration.settings.base_url == "https://current-config.example/v1"
    assert hydration.settings.system_prompt == "  keep exact formatting\n"
    assert hydration.settings.pinned_prefill == "Certainly,"
    assert conversation["metadata"] == before_metadata


@pytest.mark.parametrize(
    ("owned", "status"),
    [
        ({"version": 1}, ConsoleGenerationSettingsReadStatus.INVALID),
        (
            {"version": 2, "future_field": "keep"},
            ConsoleGenerationSettingsReadStatus.UNSUPPORTED_VERSION,
        ),
    ],
)
def test_invalid_or_future_generation_metadata_falls_back_without_rewriting(
    owned, status
) -> None:
    metadata = {
        CONSOLE_GENERATION_SETTINGS_METADATA_KEY: owned,
        "pinned_response_prefill": "Pinned",
        "sibling": "untouched",
    }
    conversation = {
        "system_prompt": "Current system owner",
        "metadata": json.dumps(metadata, sort_keys=True),
    }
    app_config = {
        "chat_defaults": {
            "provider": "openai",
            "model": "current-model",
            "temperature": 0.45,
            "streaming": False,
        },
        "api_settings": {"openai": {"api_url": "https://current-openai.example/v1"}},
    }
    before = conversation["metadata"]

    hydration = hydrate_console_generation_settings(app_config, conversation)

    assert hydration.metadata_status is status
    assert hydration.durable_snapshot is None
    assert hydration.settings.provider == "openai"
    assert hydration.settings.model == "current-model"
    assert hydration.settings.temperature == 0.45
    assert hydration.settings.streaming is False
    assert hydration.settings.base_url == "https://current-openai.example/v1"
    assert hydration.settings.system_prompt == "Current system owner"
    assert hydration.settings.pinned_prefill == "Pinned"
    assert conversation["metadata"] == before

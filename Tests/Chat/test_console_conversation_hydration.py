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
from dataclasses import replace

import pytest

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_fleet_wake_wiring import _attach_real_dbs
from Tests.UI.test_console_native_chat_flow import (
    StaticConversationTreeService,
    _configure_native_ready_console,
)
from tldw_chatbook.Chat.console_conversation_hydration import (
    apply_resume_settings_overrides,
    console_messages_from_conversation_tree,
    hydrate_console_session,
    load_console_conversation_tree,
)
from tldw_chatbook.Chat.console_session_settings import (
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
        "system_prompt": "  You are Alraune.\n",
        "workspace_id": "ws-fixture",
        "runtime_backend": "local",
        "assistant_kind": "character",
        "assistant_id": "7",
        "assistant_authority_id": "local-authority",
        "character_id": 7,
        "metadata": {
            "console_roleplay_context": {
                "version": 2,
                "user_name_override": "Robert",
                "character_system_template": "  You are {{char}}.\n",
                "character_name_snapshot": "Alraune",
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
    launch_session = await hydrate_console_session(
        app=launch_app,
        store=launch_store,
        conversation_id=CONVERSATION_ID,
        tree=tree,
        settings=apply_resume_settings_overrides(
            default_console_session_settings(launch_app.app_config), conversation
        ),
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
        "character_name",
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
        "  You are Alraune.\n"
    ), (
        "the saved system prompt must be restored VERBATIM -- the comparison "
        "above is worthless if both sides restored nothing"
    )
    assert launch_session.user_display_name_override == "Robert", (
        "the roleplay overlay never reached either session, so comparing them "
        "proved nothing"
    )
    assert launch_session.character_name == "Alraune"
    assert launch_session.settings.character_label == "Alraune"
    assert launch_session.id != screen_session.id, (
        "session ids are per-session uuids; equal ids would mean the two "
        "stores are the same object and this test is not comparing two callers"
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

    monkeypatch.setattr(store, "hydrate_session_library_policy", observe_before_activation)
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure_boundary",
    ("hydrate_session_library_policy", "reconcile_pending_workspace_projection"),
)
async def test_hydration_rollback_is_atomic_across_policy_boundaries(
    failure_boundary,
    monkeypatch,
):
    store = _build_test_app().console_runtime.ensure_chat_store()
    prior_settings = default_console_session_settings({})
    prior = store.create_session(title="Prior", settings=prior_settings)
    store.set_session_draft(prior.id, "draft stays exact")

    async def fail_after_restore(_session_id):
        raise RuntimeError(f"failed {failure_boundary}")

    monkeypatch.setattr(store, failure_boundary, fail_after_restore)
    app = type(
        "HydrationApp",
        (),
        {
            "chachanotes_db": type(
                "HydrationDB",
                (),
                {"get_conversation_active_leaf": lambda _self, _target: None},
            )()
        },
    )()
    tree = {
        "conversation": {"id": "rollback-target", "title": "Rollback target"},
        "root_threads": [],
    }

    with pytest.raises(RuntimeError, match=f"failed {failure_boundary}"):
        await hydrate_console_session(
            app=app,
            store=store,
            conversation_id="rollback-target",
            tree=tree,
            settings=prior_settings,
        )

    assert store.active_session_id == prior.id
    assert store.sessions() == [prior]
    assert prior.settings is prior_settings
    assert prior.draft == "draft stays exact"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure_boundary",
    ("hydrate_session_library_policy", "reconcile_pending_workspace_projection"),
)
async def test_hydration_cancellation_rolls_back_then_propagates(
    failure_boundary,
    monkeypatch,
):
    store = _build_test_app().console_runtime.ensure_chat_store()
    prior_settings = default_console_session_settings({})
    prior = store.create_session(title="Prior", settings=prior_settings)
    store.set_session_draft(prior.id, "draft stays exact")

    async def cancel_after_restore(_session_id):
        raise asyncio.CancelledError

    monkeypatch.setattr(store, failure_boundary, cancel_after_restore)
    app = type(
        "HydrationApp",
        (),
        {
            "chachanotes_db": type(
                "HydrationDB",
                (),
                {"get_conversation_active_leaf": lambda _self, _target: None},
            )()
        },
    )()
    tree = {
        "conversation": {"id": "cancel-target", "title": "Cancel target"},
        "root_threads": [],
    }

    with pytest.raises(asyncio.CancelledError):
        await hydrate_console_session(
            app=app,
            store=store,
            conversation_id="cancel-target",
            tree=tree,
            settings=prior_settings,
        )

    assert store.active_session_id == prior.id
    assert store.sessions() == [prior]
    assert prior.settings is prior_settings
    assert prior.draft == "draft stays exact"


@pytest.mark.asyncio
async def test_hydration_restores_v2_local_character_snapshot_for_future_projections(
    tmp_path,
):
    """A saved name, not a mutable character card, owns resumed identity."""
    app = _fixture_app(tmp_path)
    store = app.console_runtime.ensure_chat_store()
    tree = {
        "conversation": {
            "id": "v2-character",
            "title": "Saved Alraune",
            "system_prompt": "Saved prompt for Alraune.",
            "runtime_backend": "local",
            "assistant_kind": "character",
            "assistant_id": "7",
            "assistant_authority_id": "local-authority",
            "character_id": 7,
            "metadata": {
                "console_roleplay_context": {
                    "version": 2,
                    "user_name_override": "Captain Rowan",
                    "character_system_template": "{{char}} speaks with {{user}}.",
                    "character_name_snapshot": "Alraune",
                }
            },
        },
        "root_threads": [],
    }

    session = await hydrate_console_session(
        app=app,
        store=store,
        conversation_id="v2-character",
        tree=tree,
        settings=replace(
            apply_resume_settings_overrides(
                default_console_session_settings(app.app_config), tree["conversation"]
            ),
            character_label="Renamed current card",
        ),
    )

    assert session.character_name == "Alraune"
    assert session.settings is not None
    assert session.settings.character_label == "Alraune"
    assert session.settings.system_prompt == "Saved prompt for Alraune."
    store._materialize_roleplay_projections_live(
        session.id, global_default="User"
    )
    assert session.settings.system_prompt == "Alraune speaks with Captain Rowan."
    assert "Renamed current card" not in session.settings.system_prompt


@pytest.mark.asyncio
async def test_hydration_keeps_v1_roleplay_without_unsaved_character_identity(tmp_path):
    """Legacy templates survive, but v1 never guesses a current card name."""
    app = _fixture_app(tmp_path)
    store = app.console_runtime.ensure_chat_store()
    tree = {
        "conversation": {
            "id": "v1-character",
            "title": "Legacy character",
            "system_prompt": "Saved legacy prompt.",
            "runtime_backend": "local",
            "assistant_kind": "character",
            "assistant_id": "7",
            "character_id": 7,
            "metadata": {
                "console_roleplay_context": {
                    "version": 1,
                    "user_name_override": "Captain Rowan",
                    "character_system_template": "{{char}} speaks with {{user}}.",
                }
            },
        },
        "root_threads": [],
    }

    session = await hydrate_console_session(
        app=app,
        store=store,
        conversation_id="v1-character",
        tree=tree,
        settings=replace(
            apply_resume_settings_overrides(
                default_console_session_settings(app.app_config), tree["conversation"]
            ),
            character_label="Renamed current card",
        ),
    )

    assert session.character_name is None
    assert session.settings is not None
    assert session.settings.character_label == ""
    assert session.settings.system_prompt == "Saved legacy prompt."
    assert session.user_display_name_override == "Captain Rowan"
    assert session.character_system_template == "{{char}} speaks with {{user}}."


@pytest.mark.asyncio
async def test_hydration_keeps_generic_sessions_without_character_identity(tmp_path):
    """A snapshot never grants a generic conversation character authority."""
    app = _fixture_app(tmp_path)
    store = app.console_runtime.ensure_chat_store()
    tree = {
        "conversation": {
            "id": "generic-session",
            "title": "Generic session",
            "system_prompt": "Saved generic prompt.",
            "runtime_backend": "local",
            "assistant_kind": "generic",
            "assistant_id": "console",
            "metadata": {
                "console_roleplay_context": {
                    "version": 2,
                    "character_name_snapshot": "Alraune",
                }
            },
        },
        "root_threads": [],
    }

    session = await hydrate_console_session(
        app=app,
        store=store,
        conversation_id="generic-session",
        tree=tree,
        settings=replace(
            apply_resume_settings_overrides(
                default_console_session_settings(app.app_config), tree["conversation"]
            ),
            character_label="Inherited label",
        ),
    )

    assert session.assistant_kind == "generic"
    assert session.character_name is None
    assert session.settings is not None
    assert session.settings.character_label == ""

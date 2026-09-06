"""Combined settings must not expose a half-published fork configuration."""

import pytest

from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_context_policy import (
    ConsoleContextPolicyOverrides,
    ContextCompactionMode,
)
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_settings_apply import (
    ConsoleSettingsAction,
    ConsoleSettingsDraftState,
    ConsoleSettingsSubmission,
    ConsoleSettingsSurface,
)


@pytest.mark.parametrize("publication_fails", (False, True))
def test_combined_settings_reject_forks_until_policy_publication_finishes(
    monkeypatch: pytest.MonkeyPatch, publication_fails: bool
) -> None:
    store = ConsoleChatStore()
    source = store.create_session(
        settings=ConsoleSessionSettings(provider="openai", model="old-model")
    )
    boundary = store.append_message(
        source.id, role=ConsoleMessageRole.USER, content="Fork this question"
    )
    other = store.create_session(
        settings=ConsoleSessionSettings(provider="openai", model="other-model")
    )
    other_boundary = store.append_message(
        other.id, role=ConsoleMessageRole.USER, content="Independent question"
    )
    assert store.fork_eligibility(boundary.id).eligible
    before = store.issue_fork_fence(boundary.id)
    policy = ConsoleContextPolicyOverrides(compaction_mode=ContextCompactionMode.OFF)
    submission = ConsoleSettingsSubmission(
        submission_id="combined-fork-publication",
        action=ConsoleSettingsAction.APPLY_TO_CHAT,
        surface=ConsoleSettingsSurface.FULL_SETTINGS,
        origin=store.capture_console_settings_origin(source.id),
        draft=ConsoleSettingsDraftState(
            settings=ConsoleSessionSettings(provider="openai", model="new-model"),
            context_policy_overrides=policy,
            field_drafts=(),
            model_drafts=(),
            endpoint_draft=None,
        ),
        user_display_name_override=None,
        default_field_mask=frozenset(),
    )
    observed = []
    publish_policy = store._replace_session_context_policy_live

    def observe_partial_publication(session, overrides):
        # Keep both real setters; probe exactly after the nested settings guard
        # has exited and before the new policy reaches the live session.
        assert session.settings.model == "new-model"
        assert session.context_policy_overrides == ConsoleContextPolicyOverrides()
        eligibility = store.fork_eligibility(boundary.id)
        try:
            store.issue_fork_fence(boundary.id)
        except ValueError as exc:
            rejection = str(exc)
        else:
            rejection = ""
        observed.append((eligibility.eligible, eligibility.reason, rejection))
        assert store.fork_eligibility(other_boundary.id).eligible
        store.issue_fork_fence(other_boundary.id)
        if publication_fails:
            raise RuntimeError("policy publication failed")
        publish_policy(session, overrides)

    monkeypatch.setattr(
        store, "_replace_session_context_policy_live", observe_partial_publication
    )
    if publication_fails:
        with pytest.raises(RuntimeError, match="policy publication failed"):
            store.commit_console_settings_live(submission)
    else:
        commit = store.commit_console_settings_live(submission)
        assert commit.settings.model == "new-model"
        assert commit.context_policy_overrides == policy
        assert source.context_policy_overrides == policy

    assert len(observed) == 1
    eligible, reason, rejection = observed[0]
    assert not eligible, "fork admitted between settings and policy publication"
    assert "changing" in reason.lower()
    assert "changing" in rejection.lower()
    assert store._fork_source_transitions == {}
    assert store.fork_eligibility(boundary.id).eligible
    store.issue_fork_fence(boundary.id)
    assert not store.validate_fork_fence(before)


@pytest.mark.parametrize(
    "route",
    ("publish_first_persisted_conversation", "rebind_persisted_conversation"),
)
@pytest.mark.parametrize("publication_fails", (False, True))
def test_binding_publication_rejects_forks_and_releases_on_every_exit(
    monkeypatch: pytest.MonkeyPatch, route: str, publication_fails: bool
) -> None:
    store = ConsoleChatStore()
    source = store.create_session(
        settings=ConsoleSessionSettings(provider="openai", model="source-model")
    )
    boundary = store.append_message(
        source.id, role=ConsoleMessageRole.USER, content="Source question"
    )
    other = store.create_session(
        settings=ConsoleSessionSettings(provider="openai", model="other-model")
    )
    other_boundary = store.append_message(
        other.id, role=ConsoleMessageRole.USER, content="Other question"
    )
    assert store.fork_eligibility(boundary.id).eligible
    store.issue_fork_fence(boundary.id)
    revision = source.conversation_binding_revision
    lookup = store._session_or_raise
    observed = []

    def observe_before_publication(session_id):
        probe.setattr(store, "_session_or_raise", lookup)
        session = lookup(session_id)
        assert session is source
        assert session.persisted_conversation_id is None
        eligibility = store.fork_eligibility(boundary.id)
        try:
            store.issue_fork_fence(boundary.id)
        except ValueError as exc:
            rejection = str(exc)
        else:
            rejection = ""
        observed.append((eligibility.eligible, eligibility.reason, rejection))
        assert store.fork_eligibility(other_boundary.id).eligible
        store.issue_fork_fence(other_boundary.id)
        if publication_fails:
            raise RuntimeError("binding publication failed")
        return session

    with monkeypatch.context() as probe:
        # Probe before the first identity write: after it, missing durable
        # message IDs alone would make this unsaved source ineligible and
        # conceal an absent publication guard.
        probe.setattr(store, "_session_or_raise", observe_before_publication)
        if publication_fails:
            with pytest.raises(RuntimeError, match="binding publication failed"):
                getattr(store, route)(source.id, "conversation-a")
        else:
            assert getattr(store, route)(source.id, "conversation-a") is source

    assert len(observed) == 1
    eligible, reason, rejection = observed[0]
    assert not eligible, "fork admitted during conversation binding publication"
    assert "changing" in reason.lower()
    assert "changing" in rejection.lower()
    assert store._fork_source_transitions == {}
    assert source.persisted_conversation_id == (
        None if publication_fails else "conversation-a"
    )
    assert source.conversation_binding_revision == revision + int(
        not publication_fails and route == "rebind_persisted_conversation"
    )
    if not publication_fails:
        store.rebind_persisted_conversation(source.id, None)
    assert store.fork_eligibility(boundary.id).eligible
    store.issue_fork_fence(boundary.id)


@pytest.mark.parametrize(
    "route",
    ("publish_first_persisted_conversation", "rebind_persisted_conversation"),
)
def test_invalid_binding_publication_does_not_leak_fork_ownership(route: str) -> None:
    store = ConsoleChatStore()
    source = store.create_session(
        settings=ConsoleSessionSettings(provider="openai", model="source-model")
    )
    boundary = store.append_message(
        source.id, role=ConsoleMessageRole.USER, content="Source question"
    )
    before = store.issue_fork_fence(boundary.id)

    with pytest.raises(ValueError, match="conversation_id"):
        getattr(store, route)(source.id, "")
    with pytest.raises(KeyError):
        getattr(store, route)("missing-session", "conversation-a")

    assert store._fork_source_transitions == {}
    assert source.persisted_conversation_id is None
    assert store.validate_fork_fence(before)

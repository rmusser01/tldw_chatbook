from dataclasses import FrozenInstanceError, replace
import inspect

import pytest

import tldw_chatbook.Chat.console_settings_apply as settings_apply
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_context_policy import ConsoleContextPolicyOverrides
from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
from tldw_chatbook.Chat.console_settings_apply import (
    FULL_MODEL_DEFAULT_FIELDS,
    QUICK_MODEL_DEFAULT_FIELDS,
    ConsoleEndpointDraft,
    ConsoleSettingsAction,
    ConsoleSettingsCommittedSubmission,
    ConsoleSettingsDraftState,
    ConsoleSettingsFieldDraft,
    ConsoleSettingsFieldProvenance,
    ConsoleSettingsLiveCommit,
    ConsoleSettingsOrigin,
    ConsoleSettingsSubmission,
    ConsoleSettingsTransfer,
    remember_model_draft,
    validate_console_settings_origin,
)


_COMMON_MODEL_FIELDS = frozenset(
    {
        "temperature",
        "top_p",
        "min_p",
        "top_k",
        "max_tokens",
        "seed",
        "presence_penalty",
        "frequency_penalty",
        "streaming",
    }
)
_ANTHROPIC_MODEL_FIELDS = _COMMON_MODEL_FIELDS | frozenset(
    {"thinking_effort", "thinking_budget_tokens"}
)


def _field(
    name: str,
    value: object | None,
    *,
    profile_override: object | None = None,
    provenance: ConsoleSettingsFieldProvenance = (
        ConsoleSettingsFieldProvenance.INHERITED
    ),
    dirty: bool = False,
) -> ConsoleSettingsFieldDraft:
    return ConsoleSettingsFieldDraft(
        name=name,
        effective_value=value,
        profile_override=profile_override,
        provenance=provenance,
        dirty=dirty,
    )


def _state(
    settings: ConsoleSessionSettings,
    *field_drafts: ConsoleSettingsFieldDraft,
    endpoint_draft: ConsoleEndpointDraft | None = None,
) -> ConsoleSettingsDraftState:
    return ConsoleSettingsDraftState(
        settings=settings,
        context_policy_overrides=ConsoleContextPolicyOverrides(),
        field_drafts=tuple(field_drafts),
        model_drafts=(),
        endpoint_draft=endpoint_draft,
    )


def _rebase(
    state: ConsoleSettingsDraftState,
    *,
    provider: str,
    model: str | None,
    app_config: dict[str, object],
    exposed_fields: frozenset[str] = FULL_MODEL_DEFAULT_FIELDS,
) -> ConsoleSettingsDraftState:
    return ConsoleChatController.rebase_console_settings_draft(
        object(),
        state,
        provider=provider,
        model=model,
        app_config=app_config,
        exposed_fields=exposed_fields,
    )


def test_origin_stores_the_exact_open_session_identity_and_is_immutable() -> None:
    origin = ConsoleSettingsOrigin(
        session_id="session-opened",
        persisted_conversation_id="conversation-opened",
        conversation_binding_revision=7,
    )

    assert origin.session_id == "session-opened"
    assert origin.persisted_conversation_id == "conversation-opened"
    assert origin.conversation_binding_revision == 7
    assert not hasattr(origin, "__dict__")
    with pytest.raises(FrozenInstanceError):
        origin.session_id = "another-session"


@pytest.mark.parametrize(
    (
        "origin_conversation_id",
        "origin_revision",
        "live_session_id",
        "live_conversation_id",
        "live_revision",
        "expected",
    ),
    (
        ("conversation-a", 3, "session-a", "conversation-a", 3, True),
        (None, 3, "session-a", None, 3, True),
        (None, 3, "session-a", "conversation-first", 3, True),
        ("conversation-a", 3, "session-a", "conversation-b", 3, False),
        (None, 3, "session-a", "conversation-rebound", 4, False),
        ("conversation-a", 3, None, None, 3, False),
        ("conversation-a", 3, "session-b", "conversation-a", 3, False),
    ),
)
def test_origin_validation_rejects_closed_or_rebound_sessions(
    origin_conversation_id: str | None,
    origin_revision: int,
    live_session_id: str | None,
    live_conversation_id: str | None,
    live_revision: int,
    expected: bool,
) -> None:
    origin = ConsoleSettingsOrigin(
        session_id="session-a",
        persisted_conversation_id=origin_conversation_id,
        conversation_binding_revision=origin_revision,
    )

    assert (
        validate_console_settings_origin(
            origin,
            live_session_id=live_session_id,
            live_persisted_conversation_id=live_conversation_id,
            live_conversation_binding_revision=live_revision,
        )
        is expected
    )


def test_default_profile_masks_are_exact_and_exclude_other_owners() -> None:
    assert QUICK_MODEL_DEFAULT_FIELDS == frozenset({"temperature", "streaming"})
    assert FULL_MODEL_DEFAULT_FIELDS == frozenset(
        {
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
        }
    )
    assert {
        "compaction_mode",
        "system_prompt",
        "base_url",
    }.isdisjoint(QUICK_MODEL_DEFAULT_FIELDS | FULL_MODEL_DEFAULT_FIELDS)


def test_default_profile_field_draft_separates_effective_and_override_values() -> None:
    inherited = _field(
        "temperature",
        0.42,
        profile_override=None,
        provenance=ConsoleSettingsFieldProvenance.INHERITED,
    )

    assert inherited.effective_value == 0.42
    assert inherited.profile_override is None
    assert not hasattr(inherited, "__dict__")
    with pytest.raises(FrozenInstanceError):
        inherited.dirty = True


def test_contract_value_objects_are_frozen_slotted_and_transfer_is_not_submission() -> (
    None
):
    origin = ConsoleSettingsOrigin("session-a", None, 0)
    draft = _state(ConsoleSessionSettings(provider="openai", model="gpt-test"))
    submission = ConsoleSettingsSubmission(
        submission_id="submission-a",
        action=ConsoleSettingsAction.APPLY_TO_CHAT,
        origin=origin,
        draft=draft,
        user_display_name_override=None,
        default_field_mask=frozenset(),
    )
    live_commit = ConsoleSettingsLiveCommit(
        submission_id="submission-a",
        session_id="session-a",
        persisted_conversation_id=None,
        conversation_binding_revision=0,
        generation_revision=1,
        context_policy_revision=1,
        settings=draft.settings,
        context_policy_overrides=draft.context_policy_overrides,
    )
    committed = ConsoleSettingsCommittedSubmission(submission, live_commit)
    transfer = ConsoleSettingsTransfer(origin, draft)

    assert committed.submission is submission
    assert committed.live_commit is live_commit
    assert not isinstance(transfer, ConsoleSettingsSubmission)
    assert not hasattr(submission, "__dict__")
    assert not hasattr(live_commit, "__dict__")
    assert not hasattr(committed, "__dict__")
    assert not hasattr(transfer, "__dict__")
    assert "textual" not in inspect.getsource(settings_apply).lower()


def test_rebase_uses_target_defaults_keeps_supported_dirty_fields_and_clears_others() -> (
    None
):
    source = _state(
        ConsoleSessionSettings(
            provider="openai",
            model="gpt-source",
            base_url="https://source.example.test/v1",
            temperature=0.21,
            top_p=0.31,
            reasoning_summary="detailed",
            thinking_effort="high",
            character_label="Keep me",
            system_prompt="Keep this prompt",
            pinned_prefill="Keep this prefill",
        ),
        _field(
            "temperature",
            0.21,
            profile_override=0.21,
            provenance=ConsoleSettingsFieldProvenance.EXPLICIT,
            dirty=True,
        ),
        _field("top_p", 0.31),
        _field(
            "reasoning_summary",
            "detailed",
            profile_override="detailed",
            provenance=ConsoleSettingsFieldProvenance.EXPLICIT,
            dirty=True,
        ),
        _field(
            "thinking_effort",
            "high",
            profile_override="high",
            provenance=ConsoleSettingsFieldProvenance.EXPLICIT,
            dirty=True,
        ),
        endpoint_draft=ConsoleEndpointDraft(
            value="https://source.example.test/v1",
            bound_provider_config_key="openai",
            dirty=True,
            checked=True,
        ),
    )
    app_config = {
        "chat_defaults": {"temperature": 0.55, "top_p": 0.65},
        "api_settings": {
            "anthropic": {
                "api_url": "https://target.example.test/v1",
                "model_defaults": {
                    "claude-target": {
                        "top_p": 0.81,
                        "thinking_effort": "low",
                    }
                },
            }
        },
    }

    rebased = _rebase(
        source,
        provider="Anthropic",
        model="claude-target",
        app_config=app_config,
    )
    fields = {field.name: field for field in rebased.field_drafts}

    assert rebased.settings.provider == "anthropic"
    assert rebased.settings.model == "claude-target"
    assert rebased.settings.base_url == "https://target.example.test/v1"
    assert rebased.settings.temperature == 0.21
    assert rebased.settings.top_p == 0.81
    assert rebased.settings.reasoning_summary is None
    assert rebased.settings.thinking_effort == "high"
    assert rebased.settings.character_label == "Keep me"
    assert rebased.settings.system_prompt == "Keep this prompt"
    assert rebased.settings.pinned_prefill == "Keep this prefill"
    assert frozenset(fields) == _ANTHROPIC_MODEL_FIELDS
    assert fields["temperature"].effective_value == 0.21
    assert fields["temperature"].provenance is (ConsoleSettingsFieldProvenance.CARRIED)
    assert fields["temperature"].dirty is True
    assert fields["top_p"].effective_value == 0.81
    assert fields["top_p"].profile_override == 0.81
    assert fields["top_p"].dirty is False
    assert fields["thinking_effort"].effective_value == "high"
    assert fields["thinking_effort"].dirty is True
    assert "reasoning_summary" not in fields
    assert rebased.endpoint_draft == ConsoleEndpointDraft(
        value="https://target.example.test/v1",
        bound_provider_config_key="anthropic",
        dirty=False,
        checked=False,
    )


def test_rebase_quick_materializes_inherited_profile_values() -> None:
    source = _state(
        ConsoleSessionSettings(provider="openai", model="source"),
        _field("temperature", 0.7),
        _field("streaming", True),
    )

    rebased = _rebase(
        source,
        provider="openai",
        model="target",
        app_config={"chat_defaults": {"temperature": 0.44, "streaming": False}},
        exposed_fields=QUICK_MODEL_DEFAULT_FIELDS,
    )
    fields = {field.name: field for field in rebased.field_drafts}

    assert frozenset(fields) == QUICK_MODEL_DEFAULT_FIELDS
    assert fields["temperature"].effective_value == 0.44
    assert fields["temperature"].profile_override == 0.44
    assert fields["temperature"].provenance is (
        ConsoleSettingsFieldProvenance.INHERITED
    )
    assert fields["streaming"].effective_value is False
    assert fields["streaming"].profile_override is False


def test_rebase_quick_materializes_dirty_profile_values_and_rejects_none() -> None:
    source = _state(
        ConsoleSessionSettings(provider="openai", model="target", temperature=0.7),
        _field("temperature", 0.7, dirty=True),
        _field("streaming", None, dirty=True),
    )

    rebased = _rebase(
        source,
        provider="openai",
        model="target",
        app_config={"chat_defaults": {"temperature": 0.44, "streaming": False}},
        exposed_fields=QUICK_MODEL_DEFAULT_FIELDS,
    )
    fields = {field.name: field for field in rebased.field_drafts}

    assert fields["temperature"].effective_value == 0.7
    assert fields["temperature"].profile_override == 0.7
    assert fields["temperature"].dirty is True
    assert fields["streaming"].effective_value is False
    assert fields["streaming"].profile_override is False
    assert fields["streaming"].dirty is False


def test_rebase_full_keeps_inherited_profile_controls_blank() -> None:
    source = _state(
        ConsoleSessionSettings(provider="openai", model="source"),
        *(_field(name, None) for name in FULL_MODEL_DEFAULT_FIELDS),
    )

    rebased = _rebase(
        source,
        provider="openai",
        model="target",
        app_config={"chat_defaults": {"temperature": 0.44, "streaming": False}},
    )
    fields = {field.name: field for field in rebased.field_drafts}

    assert fields["temperature"].effective_value == 0.44
    assert fields["temperature"].profile_override is None
    assert fields["streaming"].effective_value is False
    assert fields["streaming"].profile_override is None


@pytest.mark.parametrize(
    (
        "field_name",
        "profile_value",
        "fallback_value",
        "expected_effective",
        "expected_override",
        "expected_provenance",
    ),
    (
        (
            "temperature",
            "",
            0.44,
            0.44,
            None,
            ConsoleSettingsFieldProvenance.INHERITED,
        ),
        (
            "reasoning_effort",
            None,
            "low",
            "low",
            None,
            ConsoleSettingsFieldProvenance.INHERITED,
        ),
        (
            "streaming",
            False,
            True,
            False,
            False,
            ConsoleSettingsFieldProvenance.EXPLICIT,
        ),
        (
            "top_k",
            "0",
            7,
            0,
            0,
            ConsoleSettingsFieldProvenance.EXPLICIT,
        ),
        (
            "temperature",
            "0",
            0.44,
            0.0,
            0.0,
            ConsoleSettingsFieldProvenance.EXPLICIT,
        ),
    ),
)
def test_rebase_normalizes_exact_profile_overrides(
    field_name: str,
    profile_value: object,
    fallback_value: object,
    expected_effective: object,
    expected_override: object,
    expected_provenance: ConsoleSettingsFieldProvenance,
) -> None:
    source = _state(
        ConsoleSessionSettings(provider="openai", model="source-model"),
    )
    app_config = {
        "chat_defaults": {field_name: fallback_value},
        "api_settings": {
            "openai": {"model_defaults": {"target-model": {field_name: profile_value}}}
        },
    }

    rebased = _rebase(
        source,
        provider="openai",
        model="target-model",
        app_config=app_config,
    )
    field = {draft.name: draft for draft in rebased.field_drafts}[field_name]

    assert field.effective_value == expected_effective
    assert field.profile_override == expected_override
    assert field.provenance is expected_provenance


@pytest.mark.parametrize(
    ("provider", "target_model", "field_name", "supported"),
    (
        ("moonshot", "kimi-k2.5", "reasoning_effort", True),
        ("moonshot", "moonshot-v1-8k", "reasoning_effort", False),
        ("zai", "glm-5.2", "reasoning_effort", True),
        ("zai", "glm-4", "reasoning_effort", False),
        ("vllm", "local-model", "reasoning_effort", True),
        ("vllm", "local-model", "thinking_budget_tokens", False),
        ("llama_cpp", "local-model", "reasoning_effort", True),
        ("llama_cpp", "local-model", "thinking_budget_tokens", True),
        ("local_mlx_lm", "local-model", "reasoning_effort", True),
        ("local_mlx_lm", "local-model", "thinking_budget_tokens", False),
        ("aphrodite", "local-model", "reasoning_effort", False),
    ),
)
def test_rebase_uses_model_and_local_wire_capabilities(
    provider: str,
    target_model: str,
    field_name: str,
    supported: bool,
) -> None:
    value = 2048 if field_name.endswith("tokens") else "high"
    source = _state(
        replace(
            ConsoleSessionSettings(provider=provider, model="source-model"),
            **{field_name: value},
        ),
        _field(
            field_name,
            value,
            profile_override=value,
            provenance=ConsoleSettingsFieldProvenance.EXPLICIT,
            dirty=True,
        ),
    )

    rebased = _rebase(
        source,
        provider=provider,
        model=target_model,
        app_config={},
    )
    field_names = {field.name for field in rebased.field_drafts}

    assert (field_name in field_names) is supported
    assert (getattr(rebased.settings, field_name) == value) is supported


def test_rebase_quick_ignores_even_provider_bound_dirty_endpoint() -> None:
    source = _state(
        ConsoleSessionSettings(
            provider="openai",
            model="target",
            base_url="https://draft.example.test/v1",
        ),
        _field("temperature", 0.7),
        _field("streaming", True),
        endpoint_draft=ConsoleEndpointDraft(
            value="https://draft.example.test/v1",
            bound_provider_config_key="openai",
            dirty=True,
            checked=True,
        ),
    )

    rebased = _rebase(
        source,
        provider="openai",
        model="target",
        app_config={
            "api_settings": {
                "openai": {"api_url": "https://configured.example.test/v1"}
            }
        },
        exposed_fields=QUICK_MODEL_DEFAULT_FIELDS,
    )

    assert rebased.settings.base_url == "https://configured.example.test/v1"
    assert rebased.endpoint_draft == ConsoleEndpointDraft(
        value="https://configured.example.test/v1",
        bound_provider_config_key="openai",
        dirty=False,
        checked=False,
    )


def test_rebase_keyed_a_b_a_drafts_restore_deliberate_a_edits() -> None:
    state_a = _state(
        ConsoleSessionSettings(
            provider="openai",
            model="vendor/model:a",
            base_url="https://a-edited.example.test/v1",
            temperature=0.13,
        ),
        _field(
            "temperature",
            0.13,
            profile_override=0.13,
            provenance=ConsoleSettingsFieldProvenance.EXPLICIT,
            dirty=True,
        ),
        _field("streaming", True),
        endpoint_draft=ConsoleEndpointDraft(
            value="https://a-edited.example.test/v1",
            bound_provider_config_key="openai",
            dirty=True,
            checked=True,
        ),
    )
    remembered_a = remember_model_draft(state_a)
    state_b = _rebase(
        remembered_a,
        provider="anthropic",
        model="claude:b",
        app_config={
            "api_settings": {"anthropic": {"api_url": "https://b.example.test/v1"}}
        },
        exposed_fields=FULL_MODEL_DEFAULT_FIELDS,
    )
    remembered_b = remember_model_draft(
        replace(
            state_b,
            settings=replace(state_b.settings, temperature=0.27),
            field_drafts=tuple(
                replace(
                    field,
                    effective_value=0.27,
                    profile_override=0.27,
                    provenance=ConsoleSettingsFieldProvenance.EXPLICIT,
                    dirty=True,
                )
                if field.name == "temperature"
                else field
                for field in state_b.field_drafts
            ),
        )
    )

    restored_a = _rebase(
        remembered_b,
        provider="openai",
        model="vendor/model:a",
        app_config={
            "api_settings": {"openai": {"api_url": "https://a.example.test/v1"}}
        },
        exposed_fields=FULL_MODEL_DEFAULT_FIELDS,
    )
    restored_fields = {field.name: field for field in restored_a.field_drafts}

    assert [(draft.provider, draft.model) for draft in restored_a.model_drafts] == [
        ("openai", "vendor/model:a"),
        ("anthropic", "claude:b"),
    ]
    assert restored_a.settings.temperature == 0.13
    assert restored_fields["temperature"].dirty is True
    assert restored_fields["temperature"].provenance is (
        ConsoleSettingsFieldProvenance.EXPLICIT
    )
    assert restored_a.settings.base_url == "https://a-edited.example.test/v1"
    assert restored_a.endpoint_draft == ConsoleEndpointDraft(
        value="https://a-edited.example.test/v1",
        bound_provider_config_key="openai",
        dirty=True,
        checked=True,
    )


def test_rebase_exact_key_restores_provenance_exactly_as_remembered() -> None:
    remembered_a = remember_model_draft(
        _state(
            ConsoleSessionSettings(provider="openai", model="model-a"),
            _field(
                "temperature",
                0.11,
                profile_override=0.11,
                provenance=ConsoleSettingsFieldProvenance.EXPLICIT,
                dirty=True,
            ),
            _field(
                "top_p",
                0.22,
                provenance=ConsoleSettingsFieldProvenance.INHERITED,
                dirty=True,
            ),
            _field(
                "streaming",
                False,
                profile_override=False,
                provenance=ConsoleSettingsFieldProvenance.CARRIED,
                dirty=True,
            ),
        )
    )
    state_b = _rebase(
        remembered_a,
        provider="anthropic",
        model="model-b",
        app_config={},
    )

    restored_a = _rebase(
        state_b,
        provider="openai",
        model="model-a",
        app_config={},
    )
    restored_provenance = {
        field.name: field.provenance for field in restored_a.field_drafts
    }

    assert restored_provenance == {
        "temperature": ConsoleSettingsFieldProvenance.EXPLICIT,
        "top_p": ConsoleSettingsFieldProvenance.INHERITED,
        "streaming": ConsoleSettingsFieldProvenance.CARRIED,
        "min_p": ConsoleSettingsFieldProvenance.INHERITED,
        "top_k": ConsoleSettingsFieldProvenance.INHERITED,
        "max_tokens": ConsoleSettingsFieldProvenance.INHERITED,
        "seed": ConsoleSettingsFieldProvenance.INHERITED,
        "presence_penalty": ConsoleSettingsFieldProvenance.INHERITED,
        "frequency_penalty": ConsoleSettingsFieldProvenance.INHERITED,
        "reasoning_effort": ConsoleSettingsFieldProvenance.INHERITED,
        "reasoning_summary": ConsoleSettingsFieldProvenance.INHERITED,
        "verbosity": ConsoleSettingsFieldProvenance.INHERITED,
    }

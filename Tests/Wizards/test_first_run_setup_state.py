"""Unit tests for the pure first-run setup wizard state module."""

import json
import pickle
from collections.abc import Iterator, Mapping
from copy import copy, deepcopy
from dataclasses import FrozenInstanceError, asdict, fields, replace

import pytest

from tldw_chatbook.UI.Wizards import first_run_setup_state as setup_state
from tldw_chatbook.UI.Wizards.first_run_setup_state import (
    any_provider_configured,
    coerce_wizard_flag,
    should_offer_wizard,
    should_show_resume_toast,
)


def _config(api_settings=None, first_run=None):
    cfg = {}
    if api_settings is not None:
        cfg["api_settings"] = api_settings
    if first_run is not None:
        cfg["first_run"] = first_run
    return cfg


class TestSetupProgress:
    def test_progress_states_derive_from_active_track(self):
        builder = getattr(setup_state, "build_setup_progress", None)
        assert callable(builder), "setup progress must expose a pure projection"

        rows = builder(("welcome", "provider", "model", "summary"), 1)

        assert [row.step_id for row in rows] == [
            "welcome",
            "provider",
            "model",
            "summary",
        ]
        assert [row.title for row in rows] == [
            "Welcome",
            "Provider",
            "Model",
            "Summary",
        ]
        assert [row.state for row in rows] == [
            "complete",
            "active",
            "upcoming",
            "upcoming",
        ]

    def test_progress_items_are_frozen_and_slotted(self):
        item_type = getattr(setup_state, "SetupProgressItem", None)
        assert item_type is not None, "setup progress must expose its row contract"
        item = item_type(step_id="welcome", title="Welcome", state="active")

        assert item.__slots__ == ("step_id", "title", "state")
        with pytest.raises(FrozenInstanceError):
            item.state = "complete"

    def test_progress_projection_rejects_unknown_steps_and_clamps_position(self):
        builder = getattr(setup_state, "build_setup_progress", None)
        assert callable(builder), "setup progress must expose a pure projection"

        with pytest.raises(ValueError, match="unknown setup step"):
            builder(("welcome", "not-a-step"), 0)
        assert [row.state for row in builder(("welcome", "summary"), -9)] == [
            "active",
            "upcoming",
        ]
        assert [row.state for row in builder(("welcome", "summary"), 99)] == [
            "complete",
            "active",
        ]


class TestCoerceWizardFlag:
    def test_truthy_values(self):
        assert coerce_wizard_flag(True) is True
        assert coerce_wizard_flag("true") is True
        assert coerce_wizard_flag(1) is True

    def test_falsy_and_garbage_values(self):
        assert coerce_wizard_flag(False) is False
        assert coerce_wizard_flag(None) is False
        assert coerce_wizard_flag("nope") is False
        assert coerce_wizard_flag({}) is False


class TestAnyProviderConfigured:
    def test_empty_config_is_unconfigured(self):
        assert any_provider_configured(_config(), {}) is False

    def test_placeholder_key_does_not_count(self):
        cfg = _config(api_settings={"openai": {"api_key": "<API_KEY_HERE>"}})
        assert any_provider_configured(cfg, {}) is False

    def test_real_inline_key_counts(self):
        cfg = _config(api_settings={"openai": {"api_key": "sk-real"}})
        assert any_provider_configured(cfg, {}) is True

    def test_env_var_present_counts(self):
        cfg = _config(api_settings={"openai": {"api_key_env_var": "OPENAI_API_KEY"}})
        assert any_provider_configured(cfg, {"OPENAI_API_KEY": "sk-x"}) is True

    def test_env_var_declared_but_unset_does_not_count(self):
        cfg = _config(api_settings={"openai": {"api_key_env_var": "OPENAI_API_KEY"}})
        assert any_provider_configured(cfg, {}) is False

    def test_template_default_endpoint_urls_do_not_count(self):
        """UAT regression: the shipped config.toml template (config.py's
        CONFIG_TOML_CONTENT) pre-populates ~12 [api_settings.*] blocks with
        default endpoint URLs (llama.cpp http://localhost:8080, Ollama,
        vLLM, HuggingFace router, etc.) on every fresh install -- none of
        them entered by the user. Counting an endpoint URL as "configured"
        therefore made a truly fresh install look configured and the wizard
        never auto-offered in the real app (confirmed live via tmux UAT).
        An endpoint alone (no real key) must NOT count."""
        cfg = _config(api_settings={"llama_cpp": {"api_url": "http://127.0.0.1:8080"}})
        assert any_provider_configured(cfg, {}) is False


class TestShouldOfferWizard:
    def test_fresh_config_offers(self):
        assert should_offer_wizard(_config(), {}) is True

    def test_configured_provider_blocks_offer(self):
        cfg = _config(api_settings={"openai": {"api_key": "sk-real"}})
        assert should_offer_wizard(cfg, {}) is False

    def test_completed_blocks_offer(self):
        cfg = _config(first_run={"setup_completed": True})
        assert should_offer_wizard(cfg, {}) is False

    def test_started_but_not_completed_blocks_reoffer(self):
        cfg = _config(first_run={"setup_started": True})
        assert should_offer_wizard(cfg, {}) is False


class TestShouldShowResumeToast:
    def test_started_not_completed_shows_toast(self):
        cfg = _config(first_run={"setup_started": True})
        assert should_show_resume_toast(cfg, {}) is True

    def test_completed_never_shows_toast(self):
        cfg = _config(first_run={"setup_started": True, "setup_completed": True})
        assert should_show_resume_toast(cfg, {}) is False

    def test_never_started_never_shows_toast(self):
        assert should_show_resume_toast(_config(), {}) is False


def _resume_draft_config(**overrides):
    first_run = {
        "setup_started": True,
        "setup_completed": False,
        "draft_version": 1,
        "draft_track": "quick",
        "active_step_id": "model",
        "draft_values": {
            "welcome": {"track": "quick"},
            "provider": {
                "provider_key": "openai",
                "provider_value": "openai",
            },
        },
        "resume_attempted": False,
    }
    first_run.update(overrides)
    return {"first_run": first_run}


class TestSetupResumeDraft:
    def test_resume_draft_accepts_allowlisted_non_secret_values(self):
        from tldw_chatbook.UI.Wizards.first_run_setup_state import (
            SETUP_DRAFT_VERSION,
            read_setup_draft,
        )

        draft = read_setup_draft(_resume_draft_config())

        assert draft is not None
        assert draft.version == SETUP_DRAFT_VERSION == 1
        assert draft.track == "quick"
        assert draft.active_step_id == "model"
        assert draft.resume_attempted is False
        assert draft.values["provider"] == {
            "provider_key": "openai",
            "provider_value": "openai",
        }

    @pytest.mark.parametrize(
        "secret_field",
        [
            "api_key",
            "API-Key",
            "saved_credential",
            "masterPassword",
            "access_token",
            "client_secret",
        ],
    )
    def test_resume_draft_rejects_secret_shaped_fields(self, secret_field):
        from tldw_chatbook.UI.Wizards.first_run_setup_state import read_setup_draft

        config = _resume_draft_config(
            draft_values={"provider": {secret_field: "must-not-survive"}}
        )
        assert read_setup_draft(config) is None

    @pytest.mark.parametrize(
        "overrides",
        [
            {"draft_version": 2},
            {"draft_track": "expert"},
            {"active_step_id": "voice-from-the-future"},
            {"draft_values": {"unknown-step": {}}},
            {"draft_values": {"model": {"unknown_field": "value"}}},
        ],
    )
    def test_resume_draft_rejects_unknown_version_track_step_or_field(self, overrides):
        from tldw_chatbook.UI.Wizards.first_run_setup_state import read_setup_draft

        assert read_setup_draft(_resume_draft_config(**overrides)) is None

    def test_resume_draft_rejects_oversized_serialized_data(self):
        from tldw_chatbook.UI.Wizards.first_run_setup_state import read_setup_draft

        config = _resume_draft_config(
            draft_values={"model": {"model_id": "x" * (16 * 1024)}}
        )
        assert read_setup_draft(config) is None

    def test_resume_draft_rejects_more_than_64_fields(self):
        from tldw_chatbook.UI.Wizards.first_run_setup_state import read_setup_draft

        class RepeatedRecognizedFields(Mapping[str, object]):
            def __getitem__(self, key: str) -> object:
                if key != "model_id":
                    raise KeyError(key)
                return "model"

            def __iter__(self) -> Iterator[str]:
                return iter(("model_id",))

            def __len__(self) -> int:
                return 65

            def items(self):
                return iter(("model_id", f"model-{index}") for index in range(65))

        config = _resume_draft_config(
            draft_values={"model": RepeatedRecognizedFields()}
        )
        assert read_setup_draft(config) is None

    def test_resume_draft_scans_nested_secret_keys_before_rejecting_mapping_value(self):
        from tldw_chatbook.UI.Wizards.first_run_setup_state import read_setup_draft

        class TrackedNestedSecrets(Mapping[str, object]):
            scanned = False

            def __getitem__(self, key: str) -> object:
                if key == "profile":
                    return {"client_secret": "must-not-survive"}
                raise KeyError(key)

            def __iter__(self) -> Iterator[str]:
                return iter(("profile",))

            def __len__(self) -> int:
                return 1

            def items(self):
                self.scanned = True
                return iter((("profile", {"client_secret": "must-not-survive"}),))

        nested = TrackedNestedSecrets()
        config = _resume_draft_config(draft_values={"model": {"model_id": nested}})

        assert read_setup_draft(config) is None
        assert nested.scanned is True

    def test_resume_draft_rejects_non_json_value_under_recognized_field(self):
        from tldw_chatbook.UI.Wizards.first_run_setup_state import read_setup_draft

        config = _resume_draft_config(draft_values={"model": {"model_id": object()}})

        assert read_setup_draft(config) is None

    def test_isolated_draft_mutation_owns_only_first_run(self):
        from tldw_chatbook.UI.Wizards.first_run_setup_state import (
            SetupDraft,
            build_setup_draft_mutation,
        )

        settings, delete_keys = build_setup_draft_mutation(
            SetupDraft(
                1,
                "quick",
                "model",
                {"provider": {"provider_value": "openai"}},
                False,
            )
        )

        assert set(settings) == {"first_run"}
        assert set(settings["first_run"]) == {
            "draft_version",
            "draft_track",
            "active_step_id",
            "draft_values",
            "resume_attempted",
        }
        assert delete_keys == {}
        assert not ({"chat_defaults", "api_settings", "app_tts"} & set(settings))

    def test_none_draft_mutation_returns_exact_draft_deletes(self):
        from tldw_chatbook.UI.Wizards.first_run_setup_state import (
            SETUP_DRAFT_KEYS,
            build_setup_draft_mutation,
        )

        settings, delete_keys = build_setup_draft_mutation(None)

        assert settings == {}
        assert delete_keys == {"first_run": SETUP_DRAFT_KEYS}
        assert "setup_started" not in SETUP_DRAFT_KEYS
        assert "setup_completed" not in SETUP_DRAFT_KEYS

    def test_checkpoint_sanitizes_credentials_transport_and_non_scalar_values(self):
        from tldw_chatbook.UI.Wizards.first_run_setup_state import (
            setup_draft_checkpoint,
        )

        draft = setup_draft_checkpoint(
            track="quick",
            active_step_id="model",
            values={
                "welcome": {"track": "quick"},
                "provider": {
                    "provider_key": "openai",
                    "provider_value": "openai",
                    "api_key": "never-copy",
                    "entered_key": True,
                    "api_url": "https://user:secret@example.test/v1?token=hidden",
                },
                "tools": {"enabled_gates": ["write_file"]},
            },
        )

        assert draft.values == {
            "welcome": {"track": "quick"},
            "provider": {
                "provider_key": "openai",
                "provider_value": "openai",
            },
        }
        assert "never-copy" not in repr(draft)
        assert "example.test" not in repr(draft)

    def test_checkpoint_excludes_typed_provider_credential_record(self):
        credential_type = getattr(setup_state, "ProviderCredentialDraft", None)
        assert credential_type is not None
        credential = credential_type("draft", "checkpoint-secret", 7)

        draft = setup_state.setup_draft_checkpoint(
            track="quick",
            active_step_id="model",
            values={
                "provider": {
                    "provider_key": "custom",
                    "provider_value": "custom",
                    "credential": credential,
                }
            },
        )

        assert draft.values["provider"] == {
            "provider_key": "custom",
            "provider_value": "custom",
        }
        assert "checkpoint-secret" not in repr(draft)
        with pytest.raises(TypeError):
            setup_state.build_setup_draft_mutation(
                setup_state.FirstRunProviderDraft(
                    "custom",
                    "https://example.test/v1/chat/completions",
                    credential,
                )
            )

    def test_resume_parsing_does_not_mutate_active_configuration(self):
        from tldw_chatbook.UI.Wizards.first_run_setup_state import read_setup_draft

        config = _resume_draft_config()
        config.update(
            {
                "chat_defaults": {"provider": "anthropic", "model": "claude"},
                "api_settings": {"anthropic": {"api_key": "active-secret"}},
                "app_tts": {"provider": "local", "voice": "active-voice"},
            }
        )
        before = deepcopy(config)

        assert read_setup_draft(config) is not None
        assert config == before


class TestSetupRecoveryAction:
    def test_offer_for_pristine_first_run(self):
        from tldw_chatbook.UI.Wizards.first_run_setup_state import setup_recovery_action

        assert setup_recovery_action({}, {}) == "offer"

    def test_prompt_for_unfinished_unattempted_draft(self):
        from tldw_chatbook.UI.Wizards.first_run_setup_state import setup_recovery_action

        assert setup_recovery_action(_resume_draft_config(), {}) == "prompt"

    def test_home_for_uncleared_resume_attempt(self):
        from tldw_chatbook.UI.Wizards.first_run_setup_state import setup_recovery_action

        assert (
            setup_recovery_action(_resume_draft_config(resume_attempted=True), {})
            == "home"
        )

    @pytest.mark.parametrize(
        "config",
        [
            {"first_run": {"setup_started": True}},
            _resume_draft_config(draft_version=99),
            {"first_run": {"setup_completed": True}},
            {"api_settings": {"openai": {"api_key": "configured"}}},
        ],
    )
    def test_none_without_a_valid_unfinished_draft(self, config):
        from tldw_chatbook.UI.Wizards.first_run_setup_state import setup_recovery_action

        assert setup_recovery_action(config, {}) == "none"


from tldw_chatbook.UI.Wizards.first_run_setup_state import (
    STEP_APPEARANCE,
    STEP_MODEL,
    STEP_NOTES,
    STEP_PROTECT,
    STEP_PROVIDER,
    STEP_RAG,
    STEP_SPEECH,
    STEP_SUMMARY,
    STEP_TOOLS,
    STEP_VOICE,
    STEP_WELCOME,
    TRACK_FULL,
    TRACK_QUICK,
    active_step_ids,
    build_appearance_commit,
    build_model_commit,
    build_provider_commit,
    build_rag_commit,
    build_tools_commit,
    build_wizard_state_commit,
    commit_sections_allowed,
    invalidate_model_for_provider_change,
    stored_plaintext_key_present,
    tools_commit_delta,
)


class TestActiveStepIds:
    def test_full_track_without_key(self):
        """Voice follows Model; Speech remains after RAG in the Full track."""
        assert active_step_ids(TRACK_FULL, key_entered=False) == (
            STEP_WELCOME,
            STEP_PROVIDER,
            STEP_MODEL,
            STEP_VOICE,
            STEP_RAG,
            STEP_SPEECH,
            STEP_TOOLS,
            STEP_NOTES,
            STEP_APPEARANCE,
            STEP_SUMMARY,
        )

    def test_full_track_with_key(self):
        assert active_step_ids(TRACK_FULL, key_entered=True) == (
            STEP_WELCOME,
            STEP_PROVIDER,
            STEP_MODEL,
            STEP_VOICE,
            STEP_RAG,
            STEP_SPEECH,
            STEP_TOOLS,
            STEP_NOTES,
            STEP_APPEARANCE,
            STEP_PROTECT,
            STEP_SUMMARY,
        )

    def test_quick_track(self):
        assert active_step_ids(TRACK_QUICK, key_entered=False) == (
            STEP_WELCOME,
            STEP_PROVIDER,
            STEP_MODEL,
            STEP_VOICE,
            STEP_SUMMARY,
        )

    def test_quick_track_with_key(self):
        assert active_step_ids(TRACK_QUICK, key_entered=True) == (
            STEP_WELCOME,
            STEP_PROVIDER,
            STEP_MODEL,
            STEP_VOICE,
            STEP_PROTECT,
            STEP_SUMMARY,
        )


class TestCommitBuilders:
    def test_provider_commit_cloud(self):
        commit = build_provider_commit(
            provider_key="openai", api_key="sk-x", api_url=None
        )
        assert commit == {"api_settings.openai": {"api_key": "sk-x"}}

    def test_provider_commit_local(self):
        commit = build_provider_commit(
            provider_key="llama_cpp", api_key=None, api_url="http://127.0.0.1:8080"
        )
        assert commit == {
            "api_settings.llama_cpp": {"api_url": "http://127.0.0.1:8080"}
        }

    def test_provider_commit_env_key_writes_nothing_secret(self):
        commit = build_provider_commit(
            provider_key="openai", api_key=None, api_url=None
        )
        assert commit == {}

    def test_model_commit(self):
        commit = build_model_commit(provider_value="OpenAI", model_id="gpt-5.6-terra")
        assert commit == {
            "chat_defaults": {"provider": "OpenAI", "model": "gpt-5.6-terra"}
        }

    def test_tools_commit_only_gate_keys(self):
        commit = build_tools_commit(
            gate_values={"read_file_enabled": True, "write_file_enabled": False}
        )
        assert commit == {
            "tools": {"read_file_enabled": True, "write_file_enabled": False}
        }

    def test_appearance_commit_with_splash(self):
        commit = build_appearance_commit(
            default_theme="textual-dark", splash_card="matrix"
        )
        assert commit == {
            "general": {"default_theme": "textual-dark"},
            "splash_screen": {"card_selection": "matrix"},
        }

    def test_appearance_commit_without_splash(self):
        commit = build_appearance_commit(default_theme="textual-dark", splash_card=None)
        assert commit == {"general": {"default_theme": "textual-dark"}}

    def test_appearance_commit_omits_theme_when_falsy(self):
        """Bug-2b: a falsy default_theme (delta-aware "unchanged") must omit
        the general section entirely, not write an empty/None theme."""
        commit = build_appearance_commit(default_theme=None, splash_card="matrix")
        assert commit == {"splash_screen": {"card_selection": "matrix"}}

    def test_appearance_commit_reset_splash_to_random(self):
        """Bug-2c: explicit reset-to-random writes card_selection="random"
        even though splash_card itself has no truthy value to signal it."""
        commit = build_appearance_commit(
            default_theme=None, splash_card=None, reset_splash_to_random=True
        )
        assert commit == {"splash_screen": {"card_selection": "random"}}

    def test_appearance_commit_specific_card_wins_over_reset_flag(self):
        commit = build_appearance_commit(
            default_theme=None, splash_card="matrix", reset_splash_to_random=True
        )
        assert commit == {"splash_screen": {"card_selection": "matrix"}}

    def test_appearance_commit_nothing_changed_is_empty(self):
        commit = build_appearance_commit(default_theme=None, splash_card=None)
        assert commit == {}

    def test_rag_commit(self):
        commit = build_rag_commit(default_model_id="e5-small-v2")
        assert commit == {"embedding_config": {"default_model_id": "e5-small-v2"}}

    def test_state_commit(self):
        assert build_wizard_state_commit(started=True) == {
            "first_run": {"setup_started": True}
        }
        assert build_wizard_state_commit(completed=True) == {
            "first_run": {"setup_completed": True}
        }


def _first_run_provider_draft(
    *,
    provider="custom",
    endpoint="https://example.test/v1/chat/completions",
    source="none",
    value="",
    revision=0,
):
    credential_type = getattr(setup_state, "ProviderCredentialDraft", None)
    draft_type = getattr(setup_state, "FirstRunProviderDraft", None)
    assert credential_type is not None
    assert draft_type is not None
    return draft_type(
        provider,
        endpoint,
        credential_type(source, value, revision),
    )


class TestFirstRunProviderContracts:
    def test_exact_typed_contracts_reject_subclasses_without_callbacks(self):
        callbacks = []

        class HostileStr(str):
            def __hash__(self):
                callbacks.append("hash")
                raise AssertionError("must not hash hostile text")

            def __eq__(self, other):
                callbacks.append("eq")
                raise AssertionError("must not compare hostile text")

            def strip(self, *args, **kwargs):
                callbacks.append("strip")
                raise AssertionError("must not normalize hostile text")

        class HostileInt(int):
            def __index__(self):
                callbacks.append("index")
                raise AssertionError("must not coerce hostile integer")

        credential_type = setup_state.ProviderCredentialDraft
        draft_type = setup_state.FirstRunProviderDraft
        key_type = setup_state.FirstRunModelDiscoveryKey

        for operation in (
            lambda: credential_type(HostileStr("draft"), "secret", 1),
            lambda: credential_type("draft", HostileStr("secret"), 1),
            lambda: credential_type("draft", "secret", HostileInt(1)),
            lambda: draft_type(
                HostileStr("custom"),
                "https://example.test/v1/chat/completions",
                credential_type("none", "", 0),
            ),
            lambda: key_type(
                provider_key="custom",
                connection_identity=(
                    "custom",
                    "https://example.test/v1/chat/completions",
                ),
                credential_source=HostileStr("draft"),
                credential_revision=1,
            ),
        ):
            with pytest.raises(ValueError):
                operation()
        assert callbacks == []

    def test_builder_rejects_hostile_mapping_without_callbacks_or_secret_errors(self):
        callbacks = []

        class HostileConfig(dict):
            def get(self, key, default=None):
                callbacks.append(key)
                raise AssertionError("must not inspect hostile config")

        class HostileModel(str):
            def strip(self, *args, **kwargs):
                callbacks.append("model-strip")
                raise AssertionError("must not normalize hostile model")

        secret = "hostile-builder-secret"
        with pytest.raises(TypeError) as exc_info:
            setup_state.build_first_run_provider_commit(
                _first_run_provider_draft(source="draft", value=secret),
                "model",
                HostileConfig(),
            )

        assert callbacks == []
        assert secret not in str(exc_info.value)
        assert len(str(exc_info.value)) < 160

        with pytest.raises(ValueError):
            setup_state.build_first_run_provider_commit(
                _first_run_provider_draft(),
                HostileModel("model"),
                {},
            )
        assert callbacks == []

        class HostileProviderTables(dict):
            def items(self):
                callbacks.append("provider-items")
                raise AssertionError("must not scan hostile provider tables")

        with pytest.raises(TypeError):
            setup_state.build_first_run_provider_commit(
                _first_run_provider_draft(),
                "model",
                {"api_settings": HostileProviderTables()},
            )
        assert callbacks == []

    def test_credential_value_is_memory_only_repr_compare_and_asdict_safe(self):
        credential_type = getattr(setup_state, "ProviderCredentialDraft", None)
        assert credential_type is not None
        first = credential_type("draft", "first-secret", 3)
        second = credential_type("draft", "second-secret", 3)

        assert first == second
        assert not hasattr(first, "value")
        assert not hasattr(first, "_value")
        assert "first-secret" not in repr(first)
        assert asdict(first) == {"source": "draft", "revision": 3}
        serialized = json.dumps(asdict(first), sort_keys=True)
        assert "first-secret" not in serialized
        with pytest.raises(TypeError, match="memory-only"):
            pickle.dumps(first)
        assert first.__slots__ == ("source", "revision")
        with pytest.raises(FrozenInstanceError):
            first.revision = 4

    def test_credential_cannot_be_reconstructed_copied_pickled_or_subclassed(self):
        secret = "sealed-credential-secret"
        credential = setup_state.ProviderCredentialDraft("draft", secret, 3)

        operations = (
            lambda: replace(credential),
            lambda: replace(credential, revision=4),
            lambda: copy(credential),
            lambda: deepcopy(credential),
            lambda: pickle.dumps(credential),
        )
        for operation in operations:
            with pytest.raises((TypeError, ValueError)) as exc_info:
                operation()
            assert secret not in str(exc_info.value)
            assert len(str(exc_info.value)) < 160

        with pytest.raises(TypeError):

            class CredentialSubclass(setup_state.ProviderCredentialDraft):
                pass

        assert [item.name for item in fields(credential)] == ["source", "revision"]
        assert asdict(credential) == {"source": "draft", "revision": 3}
        assert secret not in repr(credential)

    @pytest.mark.parametrize(
        ("source", "value", "revision"),
        [
            ("saved", "", 0),
            ("stored", "must-not-carry-a-secret", 0),
            ("draft", "secret", -1),
            ("draft", "secret", True),
            ("draft", "secret", 2**63),
            ("draft", "x" * 8193, 0),
            ("environment", "not an env var", 0),
        ],
    )
    def test_credential_rejects_invalid_source_value_or_revision(
        self, source, value, revision
    ):
        credential_type = getattr(setup_state, "ProviderCredentialDraft", None)
        assert credential_type is not None
        with pytest.raises(ValueError):
            credential_type(source, value, revision)

    def test_provider_draft_repr_and_asdict_are_secret_free(self):
        draft = _first_run_provider_draft(
            source="draft", value="provider-draft-secret", revision=4
        )

        assert "provider-draft-secret" not in repr(draft)
        serialized = json.dumps(asdict(draft), sort_keys=True)
        assert "provider-draft-secret" not in serialized
        assert asdict(draft)["credential"] == {
            "source": "draft",
            "revision": 4,
        }

    @pytest.mark.parametrize(
        ("provider", "endpoint", "credential"),
        [
            ("", "https://example.test/v1", None),
            ("custom", "x" * 4097, None),
            ("custom", "https://example.test/\n", None),
            ("custom", "https://example.test/v1", object()),
        ],
    )
    def test_provider_draft_rejects_invalid_types_and_bounds(
        self, provider, endpoint, credential
    ):
        credential_type = getattr(setup_state, "ProviderCredentialDraft", None)
        draft_type = getattr(setup_state, "FirstRunProviderDraft", None)
        assert credential_type is not None
        assert draft_type is not None
        effective_credential = credential or credential_type("none", "", 0)
        with pytest.raises(ValueError):
            draft_type(provider, endpoint, effective_credential)

    def test_discovery_key_separates_endpoints_and_never_carries_value(self):
        builder = getattr(setup_state, "build_first_run_model_discovery_key", None)
        assert callable(builder)
        first = builder(
            _first_run_provider_draft(
                endpoint="http://127.0.0.1:8080",
                source="draft",
                value="discovery-secret",
                revision=8,
            )
        )
        equivalent = builder(
            _first_run_provider_draft(
                endpoint="http://127.0.0.1:8080/v1/chat/completions",
                source="draft",
                value="different-secret",
                revision=8,
            )
        )
        second = builder(
            _first_run_provider_draft(
                endpoint="http://127.0.0.1:8081",
                source="draft",
                value="discovery-secret",
                revision=8,
            )
        )

        assert first == equivalent
        assert first != second
        assert first.provider_key == "custom"
        assert first.credential_source == "draft"
        assert first.credential_revision == 8
        assert "secret" not in repr(first).lower()
        assert set(asdict(first)) == {
            "provider_key",
            "connection_identity",
            "credential_source",
            "credential_revision",
        }

    @pytest.mark.parametrize(
        ("provider", "settings", "expected_discovery", "expected_identity"),
        [
            (
                "moonshot",
                {"api_region": "china"},
                "https://api.moonshot.cn/v1/chat/completions",
                "https://api.moonshot.cn/v1/chat/completions",
            ),
            (
                "moonshot",
                {"api_region": "global"},
                "https://api.moonshot.ai/v1/chat/completions",
                "https://api.moonshot.ai/v1/chat/completions",
            ),
            (
                "huggingface",
                {"use_router_url_format": "true"},
                "https://router.huggingface.co/v1/chat/completions",
                "https://router.huggingface.co/v1/chat/completions",
            ),
            (
                "huggingface",
                {"use_router_url_format": "false"},
                "https://api-inference.huggingface.co/v1/chat/completions",
                "https://api-inference.huggingface.co/v1/chat/completions",
            ),
        ],
    )
    def test_settings_aware_builtin_endpoint_is_pinned_separately_from_editable_value(
        self,
        provider,
        settings,
        expected_discovery,
        expected_identity,
    ):
        resolved = setup_state.resolve_first_run_provider_draft(
            _first_run_provider_draft(provider=provider, endpoint=""),
            {"api_settings": {provider: settings}},
        )

        key = setup_state.build_first_run_model_discovery_key(resolved)

        assert resolved.endpoint == ""
        assert resolved.discovery_endpoint == expected_discovery
        assert key.connection_identity == (provider, expected_identity)
        mutation = setup_state.build_first_run_provider_commit(
            resolved,
            "model-id",
            {"api_settings": {provider: settings}},
        )
        provider_values = mutation.section_values[f"api_settings.{provider}"]
        assert not any(
            endpoint_key in provider_values
            for endpoint_key in ("api_url", "api_base_url", "base_url", "endpoint")
        )

    def test_settings_change_replaces_builtin_discovery_identity_without_editing_url(
        self,
    ):
        draft = _first_run_provider_draft(provider="moonshot", endpoint="")
        china = setup_state.resolve_first_run_provider_draft(
            draft,
            {"api_settings": {"moonshot": {"api_region": "china"}}},
        )
        default = setup_state.resolve_first_run_provider_draft(
            china,
            {"api_settings": {"moonshot": {"api_region": "global"}}},
        )

        assert china.endpoint == default.endpoint == ""
        assert china.discovery_endpoint != default.discovery_endpoint
        assert (
            setup_state.build_first_run_model_discovery_key(china)
            != setup_state.build_first_run_model_discovery_key(default)
        )

    def test_static_openai_builtin_discovery_identity_remains_supported(self):
        draft = setup_state.resolve_first_run_provider_draft(
            _first_run_provider_draft(provider="openai", endpoint=""),
            {"api_settings": {"openai": {}}},
        )

        assert draft.endpoint == ""
        assert draft.discovery_endpoint == (
            "https://api.openai.com/v1/chat/completions"
        )
        assert setup_state.build_first_run_model_discovery_key(
            draft
        ).connection_identity == (
            "openai",
            "https://api.openai.com/v1/chat/completions",
        )

    @pytest.mark.parametrize(
        "endpoint",
        [
            "",
            "ftp://example.test/v1",
            "https://user:secret@example.test/v1",
            "https://example.test/v1?token=secret",
            "https://example.test/" + ("x" * 4096),
        ],
    )
    def test_discovery_key_rejects_noncanonical_or_oversized_endpoint(self, endpoint):
        builder = getattr(setup_state, "build_first_run_model_discovery_key", None)
        assert callable(builder)
        with pytest.raises(ValueError):
            builder(_first_run_provider_draft(endpoint=endpoint))

    @pytest.mark.parametrize(
        "overrides",
        [
            {"provider_key": "Custom"},
            {
                "connection_identity": [
                    "custom",
                    "https://example.test/v1/chat/completions",
                ]
            },
            {"connection_identity": ("custom", "https://example.test/v1/models")},
            {"credential_source": "saved"},
            {"credential_revision": True},
            {"credential_revision": 2**63},
        ],
    )
    def test_discovery_key_constructor_revalidates_types_bounds_and_canonicality(
        self, overrides
    ):
        key_type = getattr(setup_state, "FirstRunModelDiscoveryKey", None)
        assert key_type is not None
        values = {
            "provider_key": "custom",
            "connection_identity": (
                "custom",
                "https://example.test/v1/chat/completions",
            ),
            "credential_source": "draft",
            "credential_revision": 1,
        }
        values.update(overrides)
        with pytest.raises(ValueError):
            key_type(**values)

    def test_llama_full_chat_url_commit_persists_root_defaults_and_confirmation(self):
        builder = getattr(setup_state, "build_first_run_provider_commit", None)
        assert callable(builder)
        mutation = builder(
            _first_run_provider_draft(
                provider="llama_cpp",
                endpoint="http://127.0.0.1:8080/v1/chat/completions",
            ),
            "local-model",
            {},
        )

        assert dict(mutation.section_values["api_settings.llama_cpp"]) == {
            "model": "local-model",
            "api_url": "http://127.0.0.1:8080",
            "credential_source": "none",
        }
        assert dict(mutation.section_values["chat_defaults"]) == {
            "provider": "llama_cpp",
            "model": "local-model",
        }
        assert dict(mutation.section_values["provider_setup.confirmed"]) == {
            "llama_cpp": True
        }

    def test_effective_llama_draft_uses_same_root_identity_as_commit(self):
        resolver = getattr(setup_state, "resolve_first_run_provider_draft", None)
        assert callable(resolver)

        resolved = resolver(
            _first_run_provider_draft(
                provider="llama_cpp",
                endpoint="http://127.0.0.1:8080/v1/chat/completions",
            ),
            {},
        )

        assert resolved.endpoint == "http://127.0.0.1:8080"

    @pytest.mark.parametrize(
        "endpoint",
        [
            "https://example.test/proxy",
            "https://example.test/proxy/v1",
            "https://example.test/proxy/v1/models",
            "https://example.test/proxy/v1/chat/completions",
        ],
    )
    def test_custom_endpoint_forms_commit_the_effective_chat_url(self, endpoint):
        builder = getattr(setup_state, "build_first_run_provider_commit", None)
        assert callable(builder)
        mutation = builder(
            _first_run_provider_draft(endpoint=endpoint),
            "custom-model",
            {},
        )

        assert mutation.section_values["api_settings.custom"]["api_url"] == (
            "https://example.test/proxy/v1/chat/completions"
        )

    def test_existing_custom_endpoint_alias_is_preserved(self):
        builder = getattr(setup_state, "build_first_run_provider_commit", None)
        assert callable(builder)
        mutation = builder(
            _first_run_provider_draft(
                endpoint="https://example.test/proxy/v1/chat/completions"
            ),
            "custom-model",
            {"api_settings": {"custom": {"base_url": "https://old.test/v1"}}},
        )

        values = mutation.section_values["api_settings.custom"]
        assert values["base_url"] == "https://example.test/proxy/v1"
        assert "api_url" not in values

    @pytest.mark.parametrize(
        ("provider", "configured_key", "configured_endpoint", "expected_endpoint"),
        [
            (
                "custom",
                "api_url",
                "https://custom.test/proxy/v1/chat/completions",
                "https://custom.test/proxy/v1/chat/completions",
            ),
            (
                "custom",
                "api_base_url",
                "https://custom.test/proxy/v1",
                "https://custom.test/proxy/v1",
            ),
            (
                "custom",
                "api_base",
                "https://custom.test/proxy/v1",
                "https://custom.test/proxy/v1",
            ),
            (
                "custom",
                "base_url",
                "https://custom.test/proxy/v1",
                "https://custom.test/proxy/v1",
            ),
            (
                "custom",
                "endpoint",
                "https://custom.test/proxy/v1/chat/completions",
                "https://custom.test/proxy/v1/chat/completions",
            ),
            (
                "llama_cpp",
                "api_url",
                "http://127.0.0.1:8080/v1/chat/completions",
                "http://127.0.0.1:8080",
            ),
        ],
    )
    def test_blank_endpoint_preserves_valid_owned_endpoint_alias(
        self,
        provider,
        configured_key,
        configured_endpoint,
        expected_endpoint,
    ):
        mutation = setup_state.build_first_run_provider_commit(
            _first_run_provider_draft(provider=provider, endpoint=""),
            "selected-model",
            {
                "api_settings": {
                    provider: {configured_key: configured_endpoint},
                }
            },
        )

        provider_values = mutation.section_values[f"api_settings.{provider}"]
        assert provider_values[configured_key] == expected_endpoint
        assert configured_key not in mutation.delete_keys.get(
            f"api_settings.{provider}", ()
        )
        assert mutation.section_values["provider_setup.confirmed"][provider] is True

    @pytest.mark.parametrize("provider", ["custom", "llama_cpp"])
    def test_blank_endpoint_rejects_endpoint_required_provider_without_existing_value(
        self, provider
    ):
        with pytest.raises(ValueError, match="endpoint"):
            setup_state.build_first_run_provider_commit(
                _first_run_provider_draft(provider=provider, endpoint=""),
                "selected-model",
                {"api_settings": {provider: {}}},
            )

    @pytest.mark.parametrize("provider", ["custom", "llama_cpp"])
    def test_blank_endpoint_rejects_malformed_existing_owned_value(self, provider):
        with pytest.raises(ValueError, match="endpoint"):
            setup_state.build_first_run_provider_commit(
                _first_run_provider_draft(provider=provider, endpoint=""),
                "selected-model",
                {"api_settings": {provider: {"api_url": "not a valid endpoint"}}},
            )

    def test_custom_openai_alias_uses_shared_owner_and_rejects_malformed_table(self):
        with pytest.raises(TypeError, match="Provider configuration"):
            setup_state.build_first_run_provider_commit(
                _first_run_provider_draft(
                    provider="custom",
                    endpoint="https://custom.test/v1/chat/completions",
                ),
                "selected-model",
                {"api_settings": {"Custom OpenAI": []}},
            )

    def test_custom_openai_alias_preserves_its_owned_table_name(self):
        mutation = setup_state.build_first_run_provider_commit(
            _first_run_provider_draft(
                provider="Custom OpenAI",
                endpoint="https://custom.test/v1/chat/completions",
            ),
            "selected-model",
            {"api_settings": {"Custom OpenAI": {"base_url": "https://old.test/v1"}}},
        )

        assert mutation.section_values["api_settings.Custom OpenAI"]["base_url"] == (
            "https://custom.test/v1"
        )
        assert mutation.section_values["chat_defaults"] == {
            "provider": "custom",
            "model": "selected-model",
        }

    @pytest.mark.parametrize(
        ("provider", "canonical"),
        [
            ("OpenAI", "openai"),
            ("Anthropic", "anthropic"),
            ("MistralAI", "mistralai"),
        ],
    )
    def test_persistence_aliases_normalize_endpoints_at_first_run_boundary(
        self, provider, canonical
    ):
        mutation = setup_state.build_first_run_provider_commit(
            _first_run_provider_draft(
                provider=provider,
                endpoint="https://provider.example.test/v1",
            ),
            "selected-model",
            {},
        )

        assert mutation.section_values["chat_defaults"]["provider"] == canonical

    @pytest.mark.parametrize("provider", ["OpenAI", "Anthropic", "MistralAI"])
    def test_malformed_alias_owned_table_is_rejected_before_endpoint_mutation(
        self, provider
    ):
        app_config = {"api_settings": {provider: []}}
        before = deepcopy(app_config)

        with pytest.raises(TypeError, match="Provider configuration"):
            setup_state.build_first_run_provider_commit(
                _first_run_provider_draft(
                    provider=provider,
                    endpoint="https://provider.example.test/v1",
                ),
                "selected-model",
                app_config,
            )

        assert app_config == before

    def test_alias_owner_after_bounded_scan_fails_closed_without_mutation(self):
        api_settings = {f"unknown_{index}": {} for index in range(256)}
        api_settings["OpenAI"] = []
        app_config = {"api_settings": api_settings}
        before = deepcopy(app_config)

        with pytest.raises(ValueError, match="too large"):
            setup_state.build_first_run_provider_commit(
                _first_run_provider_draft(
                    provider="openai",
                    endpoint="https://provider.example.test/v1",
                ),
                "selected-model",
                app_config,
            )

        assert app_config == before

    @pytest.mark.parametrize(
        ("source", "value", "expected_source", "set_key", "delete_key"),
        [
            ("draft", "typed-secret", "stored", "api_key", "api_key_env_var"),
            (
                "environment",
                "CUSTOM_API_KEY",
                "environment",
                "api_key_env_var",
                "api_key",
            ),
        ],
    )
    def test_commit_maps_credential_source_and_revision_without_identity_value(
        self, source, value, expected_source, set_key, delete_key
    ):
        builder = getattr(setup_state, "build_first_run_provider_commit", None)
        assert callable(builder)
        mutation = builder(
            _first_run_provider_draft(source=source, value=value, revision=11),
            "custom-model",
            {},
        )

        provider_values = mutation.section_values["api_settings.custom"]
        assert provider_values[set_key] == value
        assert delete_key in mutation.delete_keys["api_settings.custom"]
        assert mutation.semantic_identity.credential_source == expected_source
        assert mutation.semantic_identity.credential_revision == 11
        assert value not in repr(mutation)

    def test_none_source_preserves_an_existing_stored_credential(self):
        builder = getattr(setup_state, "build_first_run_provider_commit", None)
        assert callable(builder)
        mutation = builder(
            _first_run_provider_draft(source="none", value="", revision=5),
            "custom-model",
            {"api_settings": {"custom": {"api_key": "existing-secret"}}},
        )

        assert mutation.section_values["api_settings.custom"]["api_key"] == (
            "existing-secret"
        )
        assert mutation.semantic_identity.credential_source == "stored"
        assert "existing-secret" not in repr(mutation)

    def test_none_source_does_not_activate_unset_environment_declaration(self):
        mutation = setup_state.build_first_run_provider_commit(
            _first_run_provider_draft(source="none", value="", revision=6),
            "custom-model",
            {
                "api_settings": {
                    "custom": {
                        "api_url": "https://example.test/v1/chat/completions",
                        "api_key_env_var": "CUSTOM_API_KEY",
                    }
                }
            },
        )

        provider_values = mutation.section_values["api_settings.custom"]
        assert provider_values["credential_source"] == "none"
        assert "api_key_env_var" in mutation.delete_keys["api_settings.custom"]
        assert mutation.semantic_identity.credential_source == "none"

    def test_none_source_prefers_valid_inline_over_unset_environment_declaration(self):
        mutation = setup_state.build_first_run_provider_commit(
            _first_run_provider_draft(source="none", value="", revision=7),
            "custom-model",
            {
                "api_settings": {
                    "custom": {
                        "api_url": "https://example.test/v1/chat/completions",
                        "api_key": "legacy-inline-secret",
                        "api_key_env_var": "CUSTOM_API_KEY",
                    }
                }
            },
        )

        provider_values = mutation.section_values["api_settings.custom"]
        assert provider_values["api_key"] == "legacy-inline-secret"
        assert "api_key" not in mutation.delete_keys.get("api_settings.custom", ())
        assert "api_key_env_var" in mutation.delete_keys["api_settings.custom"]
        assert mutation.semantic_identity.credential_source == "stored"

    @pytest.mark.parametrize(
        "placeholder", ("<API_KEY_HERE>", "YOUR_KEY", "your_key", "your-api-key")
    )
    def test_none_source_prefers_valid_custom_environment_over_placeholder_inline(
        self, placeholder, monkeypatch
    ):
        monkeypatch.setenv("PRIVATE_CUSTOM_KEY", "active-environment-key")
        mutation = setup_state.build_first_run_provider_commit(
            _first_run_provider_draft(source="none", value="", revision=8),
            "custom-model",
            {
                "api_settings": {
                    "custom": {
                        "api_url": "https://example.test/v1/chat/completions",
                        "api_key": placeholder,
                        "api_key_env_var": "PRIVATE_CUSTOM_KEY",
                    }
                }
            },
        )

        provider_values = mutation.section_values["api_settings.custom"]
        assert provider_values["api_key_env_var"] == "PRIVATE_CUSTOM_KEY"
        assert provider_values["credential_source"] == "environment"
        assert "api_key" in mutation.delete_keys["api_settings.custom"]
        assert mutation.semantic_identity.credential_source == "environment"

    @pytest.mark.parametrize(
        "placeholder", ("<API_KEY_HERE>", "YOUR_KEY", "your_key", "your-api-key")
    )
    def test_replace_rejects_every_canonical_placeholder(self, placeholder):
        with pytest.raises(ValueError, match="Credential value is invalid"):
            setup_state.build_first_run_provider_commit(
                _first_run_provider_draft(
                    source="draft", value=placeholder, revision=9
                ),
                "custom-model",
                {},
            )

    @pytest.mark.parametrize(
        ("model_id", "app_config"),
        [
            ("", {}),
            (False, {}),
            ("x" * 121, {}),
            ("bad\nmodel", {}),
            ("model", {"api_settings": "malformed"}),
            ("model", {"api_settings": {"custom": "malformed"}}),
        ],
    )
    def test_commit_rejects_invalid_model_or_malformed_config(
        self, model_id, app_config
    ):
        builder = getattr(setup_state, "build_first_run_provider_commit", None)
        assert callable(builder)
        with pytest.raises((TypeError, ValueError)):
            builder(_first_run_provider_draft(), model_id, app_config)


class TestFirstRunSummaryActions:
    @pytest.mark.parametrize(
        ("provider_configured", "model_configured", "primary"),
        [
            (True, True, "start_chatting"),
            (True, False, "review_provider"),
            (False, True, "review_provider"),
            (False, False, "review_provider"),
        ],
    )
    def test_action_hierarchy_is_complete_and_unique(
        self, provider_configured, model_configured, primary
    ):
        builder = getattr(setup_state, "build_first_run_summary_actions", None)
        assert callable(builder)
        actions = builder(
            provider_configured=provider_configured,
            model_configured=model_configured,
        )

        assert actions == (primary, "explore_home", "review_settings")
        assert len(actions) == len(set(actions)) == 3

    def test_action_hierarchy_rejects_truthy_non_booleans(self):
        builder = getattr(setup_state, "build_first_run_summary_actions", None)
        assert callable(builder)
        with pytest.raises(ValueError):
            builder(provider_configured=1, model_configured=True)


class TestToolsCommitDelta:
    def test_no_changes_yields_empty_delta(self):
        assert (
            tools_commit_delta(
                gate_values={"read_file_enabled": False, "write_file_enabled": False},
                current_gates={},
            )
            == {}
        )

    def test_absent_current_key_defaults_to_false(self):
        """A gate never persisted before is treated as effectively False --
        matching BuiltinToolProvider's own get_cli_setting(..., False) gate
        check -- so turning it on is reported, unchanged is not."""
        delta = tools_commit_delta(
            gate_values={"read_file_enabled": True, "write_file_enabled": False},
            current_gates={},
        )
        assert delta == {"read_file_enabled": True}

    def test_on_to_off_transition_is_reported(self):
        delta = tools_commit_delta(
            gate_values={"read_file_enabled": False},
            current_gates={"read_file_enabled": True},
        )
        assert delta == {"read_file_enabled": False}

    def test_unchanged_on_gate_is_not_reported(self):
        delta = tools_commit_delta(
            gate_values={"read_file_enabled": True},
            current_gates={"read_file_enabled": True},
        )
        assert delta == {}


class TestDependencyInvalidation:
    def test_provider_change_clears_stale_model(self):
        commit = build_provider_commit(
            provider_key="anthropic", api_key="sk-a", api_url=None
        )
        merged = invalidate_model_for_provider_change(
            commit, previous_provider_value="OpenAI", new_provider_value="Anthropic"
        )
        assert merged["chat_defaults"] == {"provider": "Anthropic", "model": ""}

    def test_same_provider_leaves_model_alone(self):
        commit = build_provider_commit(
            provider_key="openai", api_key="sk-x", api_url=None
        )
        merged = invalidate_model_for_provider_change(
            commit, previous_provider_value="OpenAI", new_provider_value="OpenAI"
        )
        assert "chat_defaults" not in merged

    def test_empty_previous_still_writes_on_first_selection(self):
        """Bug-3: previous_provider_value="" (the persisted-fallback value on
        an empty/fresh config) must be treated as differing from any real
        new provider -- the old ``if previous_provider_value and ...``
        truthiness check silently skipped this because "" is falsy, which is
        exactly why a first-ever provider selection never synced
        chat_defaults.provider."""
        commit = build_provider_commit(
            provider_key="openai", api_key="sk-x", api_url=None
        )
        merged = invalidate_model_for_provider_change(
            commit, previous_provider_value="", new_provider_value="OpenAI"
        )
        assert merged["chat_defaults"] == {"provider": "OpenAI", "model": ""}

    def test_none_previous_is_treated_like_empty(self):
        """None (no information at all) must behave the same as "" above."""
        commit = build_provider_commit(
            provider_key="openai", api_key="sk-x", api_url=None
        )
        merged = invalidate_model_for_provider_change(
            commit, previous_provider_value=None, new_provider_value="OpenAI"
        )
        assert merged["chat_defaults"] == {"provider": "OpenAI", "model": ""}


class TestSectionAllowlist:
    def test_all_builders_stay_in_allowlist(self):
        from tldw_chatbook.UI.Wizards.first_run_speech_step_state import (
            build_speech_transcription_commit,
        )

        commits = [
            build_provider_commit(provider_key="openai", api_key="sk", api_url=None),
            build_model_commit(provider_value="OpenAI", model_id="m"),
            build_rag_commit(default_model_id="e5-small-v2"),
            build_tools_commit(gate_values={"read_file_enabled": True}),
            build_appearance_commit(default_theme="t", splash_card="c"),
            build_wizard_state_commit(started=True),
            build_speech_transcription_commit(
                provider_id="parakeet-onnx",
                model_id="nemo-parakeet-tdt-0.6b-v2",
                language="en",
                precision="int8",
            ),
        ]
        for commit in commits:
            assert commit_sections_allowed(commit), commit

    def test_foreign_section_rejected(self):
        assert commit_sections_allowed({"database": {"x": 1}}) is False


from tldw_chatbook.UI.Wizards.first_run_setup_state import (
    build_summary_rows,
    provider_summary_configured,
    read_provider_secret_presence,
    read_wizard_prefill,
)


class TestSecretPresence:
    def test_inline_key_is_configured_without_value(self):
        cfg = {"api_settings": {"openai": {"api_key": "sk-secret"}}}
        presence = read_provider_secret_presence(cfg, {}, provider_key="openai")
        assert presence.configured is True
        assert "sk-secret" not in repr(presence)

    def test_env_var_reported(self):
        cfg = {"api_settings": {"openai": {"api_key_env_var": "OPENAI_API_KEY"}}}
        presence = read_provider_secret_presence(
            cfg, {"OPENAI_API_KEY": "sk-x"}, provider_key="openai"
        )
        assert presence.env_var == "OPENAI_API_KEY"
        assert presence.env_var_set is True
        assert presence.env_var_declared is True
        assert presence.configured is True

    @pytest.mark.parametrize(
        "placeholder", ("", "<API_KEY_HERE>", "YOUR_KEY", "your_key", "your-api-key")
    )
    def test_canonical_placeholders_never_count_as_inline_or_environment(
        self, placeholder
    ):
        cfg = {
            "api_settings": {
                "custom": {
                    "api_key": placeholder,
                    "api_key_env_var": "PRIVATE_CUSTOM_KEY",
                }
            }
        }
        presence = read_provider_secret_presence(
            cfg, {"PRIVATE_CUSTOM_KEY": placeholder}, provider_key="custom"
        )

        assert presence.inline_configured is False
        assert presence.env_var_set is False
        assert presence.configured is False

    def test_unset_declared_env_var_remains_owned_configuration(self):
        cfg = {"api_settings": {"openai": {"api_key_env_var": "OPENAI_API_KEY"}}}
        presence = read_provider_secret_presence(cfg, {}, provider_key="openai")

        assert presence.env_var == "OPENAI_API_KEY"
        assert presence.env_var_set is False
        assert presence.env_var_declared is True

    def test_alias_owned_unset_env_var_remains_declared(self):
        cfg = {
            "api_settings": {"Custom OpenAI": {"api_key_env_var": "PRIVATE_CUSTOM_KEY"}}
        }

        presence = read_provider_secret_presence(cfg, {}, provider_key="custom")

        assert presence.env_var == "PRIVATE_CUSTOM_KEY"
        assert presence.env_var_set is False
        assert presence.env_var_declared is True

    def test_unconfigured(self):
        presence = read_provider_secret_presence({}, {}, provider_key="openai")
        assert presence.configured is False

    def test_conventional_env_var_detected_without_explicit_config(self):
        """No api_key_env_var persisted yet -- falls back to <PROVIDER>_API_KEY.

        Task-6 finding: app_config may not yet carry the packaged default's
        api_key_env_var entries (e.g. before first write to disk, or a
        pared-down test double), which is exactly the state a first-run
        wizard most plausibly sees. Without this fallback, an already
        env-exported key could never be detected as "found in your
        environment" here even though Chat's own readiness check
        (provider_readiness.get_provider_readiness) would find it.
        """
        presence = read_provider_secret_presence(
            {}, {"OPENAI_API_KEY": "sk-x"}, provider_key="openai"
        )
        assert presence.env_var == "OPENAI_API_KEY"
        assert presence.env_var_set is True
        assert presence.configured is True

    def test_conventional_env_var_fallback_does_not_leak_when_unset(self):
        """The fallback resolves a name but must not fabricate presence."""
        presence = read_provider_secret_presence({}, {}, provider_key="openai")
        assert presence.env_var == "OPENAI_API_KEY"
        assert presence.env_var_set is False
        assert presence.configured is False

    def test_no_fallback_for_keyless_providers(self):
        """llama_cpp needs no key; the fallback must not invent one."""
        presence = read_provider_secret_presence(
            {}, {"LLAMA_CPP_API_KEY": "unused"}, provider_key="llama_cpp"
        )
        assert presence.env_var is None
        assert presence.env_var_set is False
        assert presence.configured is False

    def test_explicit_none_hides_saved_and_environment_presence(self):
        presence = read_provider_secret_presence(
            {
                "api_settings": {
                    "custom": {
                        "credential_source": "none",
                        "api_key": "saved-presence-canary",
                        "api_key_env_var": "CUSTOM_API_KEY",
                    }
                }
            },
            {"CUSTOM_API_KEY": "environment-presence-canary"},
            provider_key="custom",
        )

        assert presence.configured is False
        assert presence.inline_configured is False
        assert presence.env_var_set is False

    def test_declared_but_unset_environment_is_not_an_active_credential_source(self):
        presence = read_provider_secret_presence(
            {
                "api_settings": {
                    "custom": {
                        "credential_source": "environment",
                        "api_key_env_var": "CUSTOM_API_KEY",
                    }
                }
            },
            {},
            provider_key="custom",
        )

        assert presence.env_var == "CUSTOM_API_KEY"
        assert presence.env_var_declared is True
        assert presence.env_var_set is False
        assert presence.configured is False


class TestWizardPrefill:
    def test_reads_current_values(self):
        cfg = {
            "chat_defaults": {"provider": "Anthropic", "model": "claude-opus-5"},
            "general": {"default_theme": "textual-light"},
            "tools": {"read_file_enabled": True},
            "splash_screen": {"card_selection": "matrix"},
        }
        prefill = read_wizard_prefill(cfg)
        assert prefill.provider_value == "Anthropic"
        assert prefill.model_id == "claude-opus-5"
        assert prefill.default_theme == "textual-light"
        assert ("read_file_enabled", True) in prefill.tool_gates
        assert prefill.card_selection == "matrix"

    def test_empty_config_yields_empty_strings(self):
        prefill = read_wizard_prefill({})
        assert prefill.provider_value == ""
        assert prefill.model_id == ""
        assert prefill.card_selection == ""


class TestStoredPlaintextKeyPresent:
    """Bug-4 truth table: active_step_ids' Protect-keys gate must be
    config-derived, not just "was a secret typed this run"."""

    def test_inline_key_no_encryption_is_true(self):
        cfg = {"api_settings": {"openai": {"api_key": "sk-real"}}}
        assert stored_plaintext_key_present(cfg) is True

    def test_encryption_enabled_is_false(self):
        cfg = {
            "api_settings": {"openai": {"api_key": "sk-real"}},
            "encryption": {"enabled": True},
        }
        assert stored_plaintext_key_present(cfg) is False

    def test_env_var_only_is_false(self):
        """A key that lives only in the environment is not a stored
        plaintext secret -- there is nothing on disk to protect."""
        cfg = {"api_settings": {"openai": {"api_key_env_var": "OPENAI_API_KEY"}}}
        assert stored_plaintext_key_present(cfg) is False

    def test_placeholder_key_is_false(self):
        cfg = {"api_settings": {"openai": {"api_key": "<API_KEY_HERE>"}}}
        assert stored_plaintext_key_present(cfg) is False

    def test_empty_config_is_false(self):
        assert stored_plaintext_key_present({}) is False


from tldw_chatbook.UI.Wizards.first_run_setup_state import curated_models_for_provider


class TestCuratedModelsForProvider:
    """Task-7 finding: ProviderStep persists chat_defaults.provider as the RAW
    provider_key (e.g. "openai", "llama_cpp"), matching chat_screen's
    detected-server path, while the curated [providers] table in config.toml
    is keyed by human display names (e.g. "OpenAI"). The fallback lookup must
    bridge both key forms rather than silently returning [] on a case/format
    mismatch."""

    def test_direct_match(self):
        catalog = {"OpenAI": ["gpt-a", "gpt-b"]}
        assert curated_models_for_provider(catalog, "OpenAI") == ["gpt-a", "gpt-b"]

    def test_bridges_raw_key_to_display_name(self):
        catalog = {"OpenAI": ["gpt-a", "gpt-b"]}
        assert curated_models_for_provider(catalog, "openai") == ["gpt-a", "gpt-b"]

    def test_bridges_raw_key_form_llama_cpp(self):
        catalog = {"Llama Cpp": ["local-model-1"]}
        assert curated_models_for_provider(catalog, "llama_cpp") == ["local-model-1"]

    def test_no_match_returns_empty(self):
        catalog = {"OpenAI": ["gpt-a"]}
        assert curated_models_for_provider(catalog, "anthropic") == []

    def test_empty_provider_value_returns_empty(self):
        assert curated_models_for_provider({"OpenAI": ["gpt-a"]}, "") == []


class TestSummaryRows:
    def test_rows_reflect_persisted_state(self):
        cfg = {
            "api_settings": {"openai": {"api_key": "sk-x"}},
            "chat_defaults": {"provider": "OpenAI", "model": "gpt-5.6-terra"},
            "encryption": {"enabled": True},
        }
        rows = {
            row.label: row
            for row in build_summary_rows(cfg, {}, rag_deps_installed=False)
        }
        assert rows["Provider"].ok is True
        assert rows["Default model"].ok is True
        assert rows["RAG"].ok is False
        assert "not installed" in rows["RAG"].detail
        assert rows["Key encryption"].ok is True

    def test_empty_config_all_missing(self):
        rows = build_summary_rows({}, {}, rag_deps_installed=True)
        by_label = {row.label: row for row in rows}
        assert by_label["Provider"].ok is False
        assert by_label["Tools"].ok is False  # no gates on => off is reported honestly

    def test_provider_row_checks_for_one_click_local_server_commit(self):
        """F2 regression: the wizard's own one-click "Use this server" path
        (ProviderStep._on_use_detected -> build_provider_commit) commits an
        api_url with NO api_key at all -- any_provider_configured (the
        auto-offer gate) deliberately never counts that. Reusing it verbatim
        for the SUMMARY row made the wizard's own one-click commit render
        "no credentials or endpoint" immediately after the user finished
        that exact flow."""
        cfg = {
            "api_settings": {"llama_cpp": {"api_url": "http://127.0.0.1:8080"}},
            "chat_defaults": {"provider": "llama_cpp", "model": ""},
            "first_run": {"setup_started": True},
        }
        rows = {
            row.label: row
            for row in build_summary_rows(cfg, {}, rag_deps_installed=False)
        }
        assert rows["Provider"].ok is True

    def test_provider_row_stays_unconfigured_for_pristine_template_shape(self):
        """The other half of F2: a synthetic dict shaped like the SHIPPED
        template's defaults (chat_defaults.provider="OpenAI", no first_run
        section at all, no api_key anywhere) must still show unconfigured --
        the fix must not resurrect the exact "poisoned by the always-present
        default" bug any_provider_configured itself was written to avoid.
        See TestFreshTemplateSummaryRow in test_first_run_setup_integration.py
        for the same claim against the REAL generated template file."""
        cfg = {
            "api_settings": {"openai": {"api_key_env_var": "OPENAI_API_KEY"}},
            "chat_defaults": {"provider": "OpenAI", "model": "gpt-5.6-terra"},
        }
        rows = {
            row.label: row
            for row in build_summary_rows(cfg, {}, rag_deps_installed=False)
        }
        assert rows["Provider"].ok is False

    def test_provider_row_endpoint_ignored_without_wizard_involvement(self):
        """A bare endpoint sitting in api_settings, matching chat_defaults.provider,
        does NOT count on its own -- only combined with the wizard's own
        setup_started/setup_completed flag (see provider_summary_configured's
        docstring for why that flag is the chosen "the wizard did this"
        signal)."""
        cfg = {
            "api_settings": {"llama_cpp": {"api_url": "http://127.0.0.1:8080"}},
            "chat_defaults": {"provider": "llama_cpp", "model": ""},
        }
        rows = {
            row.label: row
            for row in build_summary_rows(cfg, {}, rag_deps_installed=False)
        }
        assert rows["Provider"].ok is False

    def test_provider_row_endpoint_for_a_different_provider_does_not_leak(self):
        """Cross-provider contamination guard: the shipped template ships
        default local-server endpoints for MANY providers at once (llama_cpp,
        ollama, vllm, ...). A user who selects a totally different provider
        (Anthropic) and enters no key must not have some OTHER, untouched
        provider's leftover endpoint make this row falsely read ✓."""
        cfg = {
            "api_settings": {
                "anthropic": {"api_key_env_var": "ANTHROPIC_API_KEY"},
                "llama_cpp": {"api_url": "http://127.0.0.1:8080"},
            },
            "chat_defaults": {"provider": "anthropic", "model": ""},
            "first_run": {"setup_started": True},
        }
        rows = {
            row.label: row
            for row in build_summary_rows(cfg, {}, rag_deps_installed=False)
        }
        assert rows["Provider"].ok is False


class TestSpeechSummaryRow:
    """TASK-1301 AC#6: Summary reports PERSISTED transcription config plus
    installed readiness -- never transient widget state. build_summary_rows
    stays pure, so "installed readiness" is a plain bool the caller (real
    SummaryStep) resolves off-loop and passes in, exactly like
    rag_deps_installed."""

    def test_never_configured_by_the_wizard_is_default(self):
        from tldw_chatbook.UI.Wizards.first_run_setup_state import ROW_DEFAULT

        rows = {
            r.label: r for r in build_summary_rows({}, {}, rag_deps_installed=False)
        }
        assert rows["Speech transcription"].state == ROW_DEFAULT

    def test_shipped_template_defaults_do_not_count_as_configured(self):
        """The [transcription] template ships default_provider="faster-whisper"
        (or a platform MLX provider) and default_model="distil-large-v3" on
        every fresh install -- neither is ever blank. Reading "configured"
        off model_id alone would resurrect the exact template-poisoning bug
        any_provider_configured's own docstring warns about."""
        from tldw_chatbook.UI.Wizards.first_run_setup_state import ROW_DEFAULT

        cfg = {
            "transcription": {
                "default_provider": "faster-whisper",
                "default_model": "distil-large-v3",
                "default_language": "en",
            }
        }
        rows = {
            r.label: r for r in build_summary_rows(cfg, {}, rag_deps_installed=False)
        }
        assert rows["Speech transcription"].state == ROW_DEFAULT

    def test_configured_and_installed_is_configured(self):
        from tldw_chatbook.UI.Wizards.first_run_setup_state import ROW_CONFIGURED

        cfg = {
            "transcription": {
                "default_provider": "parakeet-onnx",
                "default_model": "nemo-parakeet-tdt-0.6b-v2",
                "default_language": "en",
            }
        }
        rows = {
            r.label: r
            for r in build_summary_rows(
                cfg,
                {},
                rag_deps_installed=False,
                speech_installed=True,
                speech_runtime_installed=True,
            )
        }
        row = rows["Speech transcription"]
        assert row.state == ROW_CONFIGURED
        assert "nemo-parakeet-tdt-0.6b-v2" in row.detail
        assert "en" in row.detail

    def test_configured_but_not_installed_is_attention(self):
        """AC#5/#6: persisted config surviving a later deletion of the
        artifact must be flagged, not silently reported as ready."""
        from tldw_chatbook.UI.Wizards.first_run_setup_state import ROW_ATTENTION

        cfg = {
            "transcription": {
                "default_provider": "parakeet-onnx",
                "default_model": "nemo-parakeet-tdt-0.6b-v2",
                "default_language": "en",
            }
        }
        rows = {
            r.label: r
            for r in build_summary_rows(
                cfg,
                {},
                rag_deps_installed=False,
                speech_installed=False,
                speech_runtime_installed=True,
            )
        }
        row = rows["Speech transcription"]
        assert row.state == ROW_ATTENTION
        # Review Important 1: "Settings" does not have a speech/model
        # category -- the real destination is the Lab nav destination's
        # Models screen (Lab -> Models -> Installed).
        assert "Lab" in row.detail
        assert "Settings" not in row.detail
        assert "not installed" in row.detail

    def test_configured_and_installed_but_runtime_missing_is_attention(self):
        """Important 4 residual (re-review): install the extra, complete
        setup (persisted + artifact installed), then remove the extra --
        the step now says "runtime not installed" while, without this fix,
        Summary would still claim configured/ready in the SAME run. Summary
        readiness must agree with the step's own runtime probe."""
        from tldw_chatbook.UI.Wizards.first_run_setup_state import ROW_ATTENTION

        cfg = {
            "transcription": {
                "default_provider": "parakeet-onnx",
                "default_model": "nemo-parakeet-tdt-0.6b-v2",
                "default_language": "en",
            }
        }
        rows = {
            r.label: r
            for r in build_summary_rows(
                cfg,
                {},
                rag_deps_installed=False,
                speech_installed=True,
                speech_runtime_installed=False,
            )
        }
        row = rows["Speech transcription"]
        assert row.state == ROW_ATTENTION
        assert "runtime" in row.detail.lower()
        assert "Lab" in row.detail

    def test_runtime_missing_takes_priority_over_not_installed(self):
        """Both problems at once must still produce ONE honest row, not a
        row that claims "not installed" when the deeper problem (the
        runtime that would run it is absent) is what the user needs to
        fix first."""
        from tldw_chatbook.UI.Wizards.first_run_setup_state import ROW_ATTENTION

        cfg = {
            "transcription": {
                "default_provider": "parakeet-onnx",
                "default_model": "nemo-parakeet-tdt-0.6b-v2",
                "default_language": "en",
            }
        }
        rows = {
            r.label: r
            for r in build_summary_rows(
                cfg,
                {},
                rag_deps_installed=False,
                speech_installed=False,
                speech_runtime_installed=False,
            )
        }
        row = rows["Speech transcription"]
        assert row.state == ROW_ATTENTION
        assert "runtime" in row.detail.lower()


class TestProviderSummaryConfigured:
    """Unit coverage for the pure helper directly, isolated from the rest of
    build_summary_rows' row-building."""

    def test_inline_key_counts_same_as_the_offer_gate(self):
        cfg = {"api_settings": {"openai": {"api_key": "sk-real"}}}
        assert provider_summary_configured(cfg, {}) is True

    def test_env_var_counts_same_as_the_offer_gate(self):
        cfg = {"api_settings": {"openai": {"api_key_env_var": "OPENAI_API_KEY"}}}
        assert provider_summary_configured(cfg, {"OPENAI_API_KEY": "sk-x"}) is True

    def test_one_click_endpoint_commit_counts(self):
        cfg = {
            "api_settings": {"llama_cpp": {"api_url": "http://127.0.0.1:8080"}},
            "chat_defaults": {"provider": "llama_cpp"},
            "first_run": {"setup_started": True},
        }
        assert provider_summary_configured(cfg, {}) is True

    def test_completed_flag_also_counts(self):
        cfg = {
            "api_settings": {"llama_cpp": {"api_url": "http://127.0.0.1:8080"}},
            "chat_defaults": {"provider": "llama_cpp"},
            "first_run": {"setup_completed": True},
        }
        assert provider_summary_configured(cfg, {}) is True

    def test_endpoint_without_either_wizard_flag_does_not_count(self):
        cfg = {
            "api_settings": {"llama_cpp": {"api_url": "http://127.0.0.1:8080"}},
            "chat_defaults": {"provider": "llama_cpp"},
        }
        assert provider_summary_configured(cfg, {}) is False

    def test_pristine_config_does_not_count(self):
        assert provider_summary_configured({}, {}) is False


class TestSummaryThreeState:
    """TASK-1504: matrix distinguishes configured / default / attention."""

    def test_untouched_defaults_are_not_claimed_as_configured(self):
        """Template values render as – default, never as ✓ configured."""
        from tldw_chatbook.UI.Wizards.first_run_setup_state import (
            ROW_DEFAULT,
            build_summary_rows,
        )

        rows = {
            r.label: r
            for r in build_summary_rows(
                {"general": {"default_theme": "textual-dark"}},
                {},
                rag_deps_installed=False,
            )
        }
        assert rows["Theme"].state == ROW_DEFAULT
        assert rows["Theme"].glyph == "–"
        assert rows["Tools"].state == ROW_DEFAULT
        assert rows["RAG"].state == ROW_DEFAULT  # optional, not an error

    def test_plaintext_keys_flag_encryption_as_attention(self):
        """Unencrypted stored keys make the encryption row a ✗ call to action."""
        from tldw_chatbook.UI.Wizards.first_run_setup_state import (
            ROW_ATTENTION,
            build_summary_rows,
        )

        cfg = {"api_settings": {"openai": {"api_key": "wizard-test-key-tri"}}}
        rows = {
            r.label: r for r in build_summary_rows(cfg, {}, rag_deps_installed=False)
        }
        assert rows["Key encryption"].state == ROW_ATTENTION
        assert rows["Key encryption"].glyph == "✗"

    def test_provider_without_model_is_attention_but_default_when_unconfigured(self):
        """A half-finished provider setup flags the model row; a pristine config does not."""
        from tldw_chatbook.UI.Wizards.first_run_setup_state import (
            ROW_ATTENTION,
            ROW_DEFAULT,
            build_summary_rows,
        )

        unconfigured = {
            r.label: r for r in build_summary_rows({}, {}, rag_deps_installed=False)
        }
        assert unconfigured["Default model"].state == ROW_DEFAULT
        cfg = {
            "api_settings": {"openai": {"api_key": "wizard-test-key-tri"}},
            "first_run": {"setup_started": True},
        }
        configured = {
            r.label: r for r in build_summary_rows(cfg, {}, rag_deps_installed=False)
        }
        assert configured["Default model"].state == ROW_ATTENTION

    def test_custom_theme_earns_configured(self):
        """Only a user-changed theme earns the ✓ configured state."""
        from tldw_chatbook.UI.Wizards.first_run_setup_state import (
            ROW_CONFIGURED,
            build_summary_rows,
        )

        rows = {
            r.label: r
            for r in build_summary_rows(
                {"general": {"default_theme": "nord"}}, {}, rag_deps_installed=False
            )
        }
        assert rows["Theme"].state == ROW_CONFIGURED


class TestRerunModelPrefill:
    """TASK-1374: prefill fires when the session provider matches persisted."""

    def test_same_provider_returns_persisted_model(self):
        """Re-running with the persisted provider surfaces the saved model."""
        from tldw_chatbook.UI.Wizards.first_run_setup_state import rerun_model_prefill

        cfg = {"chat_defaults": {"provider": "anthropic", "model": "claude-opus-5"}}
        assert rerun_model_prefill(cfg, provider_value="anthropic") == "claude-opus-5"

    def test_provider_forms_are_normalized(self):
        """Display-cased template value matches the raw key the wizard writes."""
        from tldw_chatbook.UI.Wizards.first_run_setup_state import rerun_model_prefill

        cfg = {"chat_defaults": {"provider": "OpenAI", "model": "gpt-5.6-terra"}}
        assert rerun_model_prefill(cfg, provider_value="openai") == "gpt-5.6-terra"

    def test_changed_provider_returns_empty(self):
        """A genuinely different provider must not inherit the old model."""
        from tldw_chatbook.UI.Wizards.first_run_setup_state import rerun_model_prefill

        cfg = {"chat_defaults": {"provider": "openai", "model": "gpt-5.6-terra"}}
        assert rerun_model_prefill(cfg, provider_value="anthropic") == ""

    def test_empty_inputs_return_empty(self):
        """No provider context or pristine config yields no prefill."""
        from tldw_chatbook.UI.Wizards.first_run_setup_state import rerun_model_prefill

        assert rerun_model_prefill({}, provider_value="openai") == ""
        assert (
            rerun_model_prefill(
                {"chat_defaults": {"provider": "openai", "model": "m"}},
                provider_value="",
            )
            == ""
        )


class TestSummaryTemplateHonesty:
    """task-2724: a Ctrl+N-only wizard walk (nothing chosen, no credentials)
    must not earn ✓ rows from template defaults merged in at config load.
    Observed live: '✗ Provider' directly above '✓ Default model — gpt-5.6-terra'
    and '✓ RAG — embedding model: e5-small-v2' under a header saying 'RAG off'."""

    _WALKED_TEMPLATE_CFG = {
        "api_settings": {"openai": {"api_key_env_var": "OPENAI_API_KEY"}},
        "chat_defaults": {"provider": "OpenAI", "model": "gpt-5.6-terra"},
        "embedding_config": {"default_model_id": "e5-small-v2"},
        "first_run": {"setup_started": True, "setup_completed": True},
    }

    def test_skipped_walkthrough_earns_no_configured_rows(self):
        rows = build_summary_rows(
            self._WALKED_TEMPLATE_CFG, {}, rag_deps_installed=True
        )
        configured = [row.label for row in rows if row.ok]
        assert configured == [], (
            f"template defaults claimed as user choices: {configured}"
        )

    def test_template_model_without_provider_reads_as_default(self):
        from tldw_chatbook.UI.Wizards.first_run_setup_state import ROW_DEFAULT

        rows = {
            r.label: r
            for r in build_summary_rows(
                self._WALKED_TEMPLATE_CFG, {}, rag_deps_installed=True
            )
        }
        assert rows["Default model"].state == ROW_DEFAULT
        assert rows["Default model"].detail == "not selected"

    def test_template_rag_model_reads_as_off_by_default(self):
        from tldw_chatbook.UI.Wizards.first_run_setup_state import ROW_DEFAULT

        rows = {
            r.label: r
            for r in build_summary_rows(
                self._WALKED_TEMPLATE_CFG, {}, rag_deps_installed=True
            )
        }
        assert rows["RAG"].state == ROW_DEFAULT
        assert "off by default" in rows["RAG"].detail

    def test_typed_model_without_provider_is_named_but_not_checked(self):
        cfg = dict(self._WALKED_TEMPLATE_CFG)
        cfg["chat_defaults"] = {"provider": "OpenAI", "model": "my-local-model"}
        rows = {
            r.label: r for r in build_summary_rows(cfg, {}, rag_deps_installed=True)
        }
        row = rows["Default model"]
        assert row.ok is False
        assert "my-local-model" in row.detail
        assert "provider" in row.detail

    def test_user_selected_rag_model_still_earns_configured(self):
        cfg = dict(self._WALKED_TEMPLATE_CFG)
        cfg["embedding_config"] = {"default_model_id": "bge-large-en"}
        rows = {
            r.label: r for r in build_summary_rows(cfg, {}, rag_deps_installed=True)
        }
        assert rows["RAG"].ok is True
        assert "bge-large-en" in rows["RAG"].detail


class TestProviderTrustChain:
    """TASK-21143 (UAT S-1/M-2/N-7): probe outcomes drive tracker, gate,
    and summary — "configured" must never masquerade as "working"."""

    def test_classify_discovery_failure(self):
        classify = setup_state.classify_discovery_failure
        assert classify("available", "") == setup_state.PROVIDER_PROBE_NONE
        assert (
            classify("listing_unavailable", "")
            == setup_state.PROVIDER_PROBE_NONE
        )
        assert (
            classify("connection_failed", "authentication")
            == setup_state.PROVIDER_PROBE_AUTH
        )
        for category in ("connection error", "request failed", "timeout", ""):
            assert (
                classify("connection_failed", category)
                == setup_state.PROVIDER_PROBE_CONNECTION
            )

    def test_summary_actions_flip_on_probe_failure(self):
        primary, secondary, tertiary = setup_state.build_first_run_summary_actions(
            provider_configured=True,
            model_configured=True,
            provider_probe_failed=True,
        )
        assert primary == "review_provider"
        assert (secondary, tertiary) == ("explore_home", "review_settings")
        # The saved-and-working case keeps its happy primary.
        primary, _, _ = setup_state.build_first_run_summary_actions(
            provider_configured=True,
            model_configured=True,
            provider_probe_failed=False,
        )
        assert primary == "start_chatting"

    def test_summary_actions_reject_non_bool_probe_state(self):
        with pytest.raises(ValueError):
            setup_state.build_first_run_summary_actions(
                provider_configured=True,
                model_configured=True,
                provider_probe_failed="yes",
            )

    def test_probe_failure_overlays_only_configured_provider_row(self):
        rows = (
            setup_state.SummaryRow(
                label="Provider", state=setup_state.ROW_CONFIGURED, detail=""
            ),
            setup_state.SummaryRow(
                label="Default model",
                state=setup_state.ROW_CONFIGURED,
                detail="m",
            ),
        )
        out = setup_state.apply_probe_failure_to_summary_rows(
            rows, setup_state.PROVIDER_PROBE_AUTH
        )
        assert out[0].state == setup_state.ROW_ATTENTION
        assert "authentication" in out[0].detail
        assert out[1] == rows[1]
        # No failure — untouched (identity, not equality, is fine too).
        assert (
            setup_state.apply_probe_failure_to_summary_rows(rows, "") == rows
        )
        # An unconfigured Provider row keeps its own, more specific message.
        unconfigured = (
            setup_state.SummaryRow(
                label="Provider",
                state=setup_state.ROW_ATTENTION,
                detail="no credentials or saved endpoint",
            ),
        )
        kept = setup_state.apply_probe_failure_to_summary_rows(
            unconfigured, setup_state.PROVIDER_PROBE_CONNECTION
        )
        assert kept == unconfigured

    def test_progress_attention_downgrades_only_completed_steps(self):
        items = setup_state.build_setup_progress(
            (
                setup_state.STEP_WELCOME,
                setup_state.STEP_PROVIDER,
                setup_state.STEP_MODEL,
                setup_state.STEP_SUMMARY,
            ),
            3,
            attention_ids=frozenset(
                {setup_state.STEP_PROVIDER, setup_state.STEP_MODEL}
            ),
        )
        assert [item.state for item in items] == [
            "complete",
            "attention",
            "attention",
            "active",
        ]
        # The active step never downgrades, even if flagged.
        items = setup_state.build_setup_progress(
            (setup_state.STEP_WELCOME, setup_state.STEP_PROVIDER),
            1,
            attention_ids=frozenset({setup_state.STEP_PROVIDER}),
        )
        assert [item.state for item in items] == ["complete", "active"]


class TestEnvKeyFirstRunNotice:
    """TASK-21147 (UAT E-1): name the env vars that silenced the wizard
    offer, exactly once, and only on genuinely fresh env-key installs."""

    def _config(self, first_run=None, api_key=None, env_var="OPENAI_API_KEY"):
        cfg = {
            "api_settings": {
                "openai": {"api_key_env_var": env_var}
                | ({"api_key": api_key} if api_key else {})
            }
        }
        if first_run is not None:
            cfg["first_run"] = first_run
        return cfg

    def test_fresh_install_with_env_key_names_the_var(self):
        names = setup_state.env_keys_that_silenced_first_run(
            self._config(), {"OPENAI_API_KEY": "sk-live-abc123456789"}
        )
        assert names == ("OPENAI_API_KEY",)

    def test_no_env_value_means_no_notice(self):
        assert (
            setup_state.env_keys_that_silenced_first_run(self._config(), {})
            == ()
        )

    def test_inline_config_key_means_no_notice(self):
        names = setup_state.env_keys_that_silenced_first_run(
            self._config(api_key="sk-live-abc123456789"),
            {"OPENAI_API_KEY": "sk-live-abc123456789"},
        )
        assert names == ()

    @pytest.mark.parametrize(
        "flags",
        [
            {"setup_started": True},
            {"setup_completed": True},
            {setup_state.ENV_KEY_NOTICE_KEY: True},
        ],
    )
    def test_any_recorded_state_suppresses_the_notice(self, flags):
        names = setup_state.env_keys_that_silenced_first_run(
            self._config(first_run=flags),
            {"OPENAI_API_KEY": "sk-live-abc123456789"},
        )
        assert names == ()

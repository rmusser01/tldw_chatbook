"""Unit tests for the pure first-run setup wizard state module."""

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

    def test_local_endpoint_url_counts(self):
        cfg = _config(api_settings={"llama_cpp": {"api_url": "http://127.0.0.1:8080"}})
        assert any_provider_configured(cfg, {}) is True


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


from tldw_chatbook.UI.Wizards.first_run_setup_state import (
    STEP_APPEARANCE,
    STEP_MODEL,
    STEP_NOTES,
    STEP_PROTECT,
    STEP_PROVIDER,
    STEP_RAG,
    STEP_SUMMARY,
    STEP_TOOLS,
    STEP_WELCOME,
    TRACK_FULL,
    TRACK_QUICK,
    active_step_ids,
    build_appearance_commit,
    build_model_commit,
    build_notes_commit,
    build_provider_commit,
    build_rag_commit,
    build_tools_commit,
    build_wizard_state_commit,
    commit_sections_allowed,
    invalidate_model_for_provider_change,
)


class TestActiveStepIds:
    def test_full_track_without_key(self):
        assert active_step_ids(TRACK_FULL, key_entered=False) == (
            STEP_WELCOME, STEP_PROVIDER, STEP_MODEL, STEP_RAG, STEP_TOOLS,
            STEP_NOTES, STEP_APPEARANCE, STEP_SUMMARY,
        )

    def test_full_track_with_key_includes_protect(self):
        assert STEP_PROTECT in active_step_ids(TRACK_FULL, key_entered=True)

    def test_quick_track(self):
        assert active_step_ids(TRACK_QUICK, key_entered=False) == (
            STEP_WELCOME, STEP_PROVIDER, STEP_MODEL, STEP_SUMMARY,
        )

    def test_quick_track_with_key(self):
        assert active_step_ids(TRACK_QUICK, key_entered=True) == (
            STEP_WELCOME, STEP_PROVIDER, STEP_MODEL, STEP_PROTECT, STEP_SUMMARY,
        )


class TestCommitBuilders:
    def test_provider_commit_cloud(self):
        commit = build_provider_commit(provider_key="openai", api_key="sk-x", api_url=None)
        assert commit == {"api_settings.openai": {"api_key": "sk-x"}}

    def test_provider_commit_local(self):
        commit = build_provider_commit(
            provider_key="llama_cpp", api_key=None, api_url="http://127.0.0.1:8080"
        )
        assert commit == {"api_settings.llama_cpp": {"api_url": "http://127.0.0.1:8080"}}

    def test_provider_commit_env_key_writes_nothing_secret(self):
        commit = build_provider_commit(provider_key="openai", api_key=None, api_url=None)
        assert commit == {}

    def test_model_commit(self):
        commit = build_model_commit(provider_value="OpenAI", model_id="gpt-5.6-terra")
        assert commit == {"chat_defaults": {"provider": "OpenAI", "model": "gpt-5.6-terra"}}

    def test_tools_commit_only_gate_keys(self):
        commit = build_tools_commit(gate_values={"read_file_enabled": True, "write_file_enabled": False})
        assert commit == {"tools": {"read_file_enabled": True, "write_file_enabled": False}}

    def test_notes_commit(self):
        commit = build_notes_commit(sync_directory="~/Notes", auto_sync_enabled=True)
        assert commit == {"notes": {"sync_directory": "~/Notes", "auto_sync_enabled": True}}

    def test_appearance_commit_with_splash(self):
        commit = build_appearance_commit(default_theme="textual-dark", splash_card="matrix")
        assert commit == {
            "general": {"default_theme": "textual-dark"},
            "splash_screen": {"card_selection": "matrix"},
        }

    def test_appearance_commit_without_splash(self):
        commit = build_appearance_commit(default_theme="textual-dark", splash_card=None)
        assert commit == {"general": {"default_theme": "textual-dark"}}

    def test_rag_commit(self):
        commit = build_rag_commit(default_model_id="e5-small-v2")
        assert commit == {"embedding_config": {"default_model_id": "e5-small-v2"}}

    def test_state_commit(self):
        assert build_wizard_state_commit(started=True) == {"first_run": {"setup_started": True}}
        assert build_wizard_state_commit(completed=True) == {"first_run": {"setup_completed": True}}


class TestDependencyInvalidation:
    def test_provider_change_clears_stale_model(self):
        commit = build_provider_commit(provider_key="anthropic", api_key="sk-a", api_url=None)
        merged = invalidate_model_for_provider_change(
            commit, previous_provider_value="OpenAI", new_provider_value="Anthropic"
        )
        assert merged["chat_defaults"] == {"provider": "Anthropic", "model": ""}

    def test_same_provider_leaves_model_alone(self):
        commit = build_provider_commit(provider_key="openai", api_key="sk-x", api_url=None)
        merged = invalidate_model_for_provider_change(
            commit, previous_provider_value="OpenAI", new_provider_value="OpenAI"
        )
        assert "chat_defaults" not in merged


class TestSectionAllowlist:
    def test_all_builders_stay_in_allowlist(self):
        commits = [
            build_provider_commit(provider_key="openai", api_key="sk", api_url=None),
            build_model_commit(provider_value="OpenAI", model_id="m"),
            build_rag_commit(default_model_id="e5-small-v2"),
            build_tools_commit(gate_values={"read_file_enabled": True}),
            build_notes_commit(sync_directory="~/n", auto_sync_enabled=False),
            build_appearance_commit(default_theme="t", splash_card="c"),
            build_wizard_state_commit(started=True),
        ]
        for commit in commits:
            assert commit_sections_allowed(commit), commit

    def test_foreign_section_rejected(self):
        assert commit_sections_allowed({"database": {"x": 1}}) is False


from tldw_chatbook.UI.Wizards.first_run_setup_state import (
    build_summary_rows,
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
        assert presence.configured is True

    def test_unconfigured(self):
        presence = read_provider_secret_presence({}, {}, provider_key="openai")
        assert presence.configured is False


class TestWizardPrefill:
    def test_reads_current_values(self):
        cfg = {
            "chat_defaults": {"provider": "Anthropic", "model": "claude-opus-5"},
            "notes": {"sync_directory": "~/N", "auto_sync_enabled": True},
            "general": {"default_theme": "textual-light"},
            "tools": {"read_file_enabled": True},
        }
        prefill = read_wizard_prefill(cfg)
        assert prefill.provider_value == "Anthropic"
        assert prefill.model_id == "claude-opus-5"
        assert prefill.sync_directory == "~/N"
        assert prefill.auto_sync_enabled is True
        assert prefill.default_theme == "textual-light"
        assert ("read_file_enabled", True) in prefill.tool_gates

    def test_empty_config_yields_empty_strings(self):
        prefill = read_wizard_prefill({})
        assert prefill.provider_value == ""
        assert prefill.model_id == ""


class TestSummaryRows:
    def test_rows_reflect_persisted_state(self):
        cfg = {
            "api_settings": {"openai": {"api_key": "sk-x"}},
            "chat_defaults": {"provider": "OpenAI", "model": "gpt-5.6-terra"},
            "encryption": {"enabled": True},
        }
        rows = {row.label: row for row in build_summary_rows(cfg, {}, rag_deps_installed=False)}
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

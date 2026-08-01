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
    stored_plaintext_key_present,
    tools_commit_delta,
)


class TestActiveStepIds:
    def test_full_track_without_key(self):
        """TASK-1301: the Speech step joins the FULL track only, right after
        RAG (also model-setup-shaped) and before Tools."""
        assert active_step_ids(TRACK_FULL, key_entered=False) == (
            STEP_WELCOME, STEP_PROVIDER, STEP_MODEL, STEP_RAG, STEP_SPEECH,
            STEP_TOOLS, STEP_NOTES, STEP_APPEARANCE, STEP_SUMMARY,
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

    def test_notes_commit_disable_only_omits_directory(self):
        """No sync_directory passed -> not present in the commit at all, so
        a merge-only writer never clobbers the persisted directory with an
        empty string when only the enabled flag flips off."""
        commit = build_notes_commit(auto_sync_enabled=False)
        assert commit == {"notes": {"auto_sync_enabled": False}}
        assert "sync_directory" not in commit["notes"]

    def test_appearance_commit_with_splash(self):
        commit = build_appearance_commit(default_theme="textual-dark", splash_card="matrix")
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
        assert build_wizard_state_commit(started=True) == {"first_run": {"setup_started": True}}
        assert build_wizard_state_commit(completed=True) == {"first_run": {"setup_completed": True}}


class TestToolsCommitDelta:
    def test_no_changes_yields_empty_delta(self):
        assert tools_commit_delta(
            gate_values={"read_file_enabled": False, "write_file_enabled": False},
            current_gates={},
        ) == {}

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

    def test_empty_previous_still_writes_on_first_selection(self):
        """Bug-3: previous_provider_value="" (the persisted-fallback value on
        an empty/fresh config) must be treated as differing from any real
        new provider -- the old ``if previous_provider_value and ...``
        truthiness check silently skipped this because "" is falsy, which is
        exactly why a first-ever provider selection never synced
        chat_defaults.provider."""
        commit = build_provider_commit(provider_key="openai", api_key="sk-x", api_url=None)
        merged = invalidate_model_for_provider_change(
            commit, previous_provider_value="", new_provider_value="OpenAI"
        )
        assert merged["chat_defaults"] == {"provider": "OpenAI", "model": ""}

    def test_none_previous_is_treated_like_empty(self):
        """None (no information at all) must behave the same as "" above."""
        commit = build_provider_commit(provider_key="openai", api_key="sk-x", api_url=None)
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
            build_notes_commit(sync_directory="~/n", auto_sync_enabled=False),
            build_appearance_commit(default_theme="t", splash_card="c"),
            build_wizard_state_commit(started=True),
            build_speech_transcription_commit(
                provider_id="parakeet-onnx",
                model_id="nemo-parakeet-tdt-0.6b-v2",
                language="en",
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
        assert presence.configured is True

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


class TestWizardPrefill:
    def test_reads_current_values(self):
        cfg = {
            "chat_defaults": {"provider": "Anthropic", "model": "claude-opus-5"},
            "notes": {"sync_directory": "~/N", "auto_sync_enabled": True},
            "general": {"default_theme": "textual-light"},
            "tools": {"read_file_enabled": True},
            "splash_screen": {"card_selection": "matrix"},
        }
        prefill = read_wizard_prefill(cfg)
        assert prefill.provider_value == "Anthropic"
        assert prefill.model_id == "claude-opus-5"
        assert prefill.sync_directory == "~/N"
        assert prefill.auto_sync_enabled is True
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
        rows = {row.label: row for row in build_summary_rows(cfg, {}, rag_deps_installed=False)}
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
        rows = {row.label: row for row in build_summary_rows(cfg, {}, rag_deps_installed=False)}
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
        rows = {row.label: row for row in build_summary_rows(cfg, {}, rag_deps_installed=False)}
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
        rows = {row.label: row for row in build_summary_rows(cfg, {}, rag_deps_installed=False)}
        assert rows["Provider"].ok is False


class TestSpeechSummaryRow:
    """TASK-1301 AC#6: Summary reports PERSISTED transcription config plus
    installed readiness -- never transient widget state. build_summary_rows
    stays pure, so "installed readiness" is a plain bool the caller (real
    SummaryStep) resolves off-loop and passes in, exactly like
    rag_deps_installed."""

    def test_never_configured_by_the_wizard_is_default(self):
        from tldw_chatbook.UI.Wizards.first_run_setup_state import ROW_DEFAULT

        rows = {r.label: r for r in build_summary_rows({}, {}, rag_deps_installed=False)}
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
        rows = {r.label: r for r in build_summary_rows(cfg, {}, rag_deps_installed=False)}
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
                cfg, {}, rag_deps_installed=False, speech_installed=True
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
                cfg, {}, rag_deps_installed=False, speech_installed=False
            )
        }
        row = rows["Speech transcription"]
        assert row.state == ROW_ATTENTION
        # Review Important 1: "Settings" does not have a speech/model
        # category -- the real destination is the Lab nav destination's
        # Models screen (Lab -> Models -> Installed).
        assert "Lab" in row.detail
        assert "Settings" not in row.detail


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

        rows = {r.label: r for r in build_summary_rows(
            {"general": {"default_theme": "textual-dark"}}, {}, rag_deps_installed=False
        )}
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
        rows = {r.label: r for r in build_summary_rows(cfg, {}, rag_deps_installed=False)}
        assert rows["Key encryption"].state == ROW_ATTENTION
        assert rows["Key encryption"].glyph == "✗"

    def test_provider_without_model_is_attention_but_default_when_unconfigured(self):
        """A half-finished provider setup flags the model row; a pristine config does not."""
        from tldw_chatbook.UI.Wizards.first_run_setup_state import (
            ROW_ATTENTION,
            ROW_DEFAULT,
            build_summary_rows,
        )

        unconfigured = {r.label: r for r in build_summary_rows({}, {}, rag_deps_installed=False)}
        assert unconfigured["Default model"].state == ROW_DEFAULT
        cfg = {
            "api_settings": {"openai": {"api_key": "wizard-test-key-tri"}},
            "first_run": {"setup_started": True},
        }
        configured = {r.label: r for r in build_summary_rows(cfg, {}, rag_deps_installed=False)}
        assert configured["Default model"].state == ROW_ATTENTION

    def test_custom_theme_earns_configured(self):
        """Only a user-changed theme earns the ✓ configured state."""
        from tldw_chatbook.UI.Wizards.first_run_setup_state import (
            ROW_CONFIGURED,
            build_summary_rows,
        )

        rows = {r.label: r for r in build_summary_rows(
            {"general": {"default_theme": "nord"}}, {}, rag_deps_installed=False
        )}
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
        assert rerun_model_prefill(
            {"chat_defaults": {"provider": "openai", "model": "m"}}, provider_value=""
        ) == ""

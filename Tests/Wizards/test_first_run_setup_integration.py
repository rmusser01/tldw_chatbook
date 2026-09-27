"""Integration tests: wizard commit plans against a real TOML config file."""

import asyncio
import json
import threading
import tomllib
from copy import deepcopy
from types import SimpleNamespace

import pytest

from Tests.private_profile import private_profile_test
from tldw_chatbook.Chat.provider_readiness import get_provider_readiness
from tldw_chatbook.config import ConfigMutationResult
from tldw_chatbook.UI.Wizards import first_run_setup_state as wizard_state
from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import SetupWizardContainer


@pytest.fixture()
def temp_config(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    from tldw_chatbook import config as config_module

    config_module.load_cli_config_and_ensure_existence(force_reload=True)
    yield config_path
    monkeypatch.delenv("TLDW_CONFIG_PATH", raising=False)
    config_module.load_cli_config_and_ensure_existence(force_reload=True)


def _reload():
    from tldw_chatbook.config import load_cli_config_and_ensure_existence

    return load_cli_config_and_ensure_existence(force_reload=True)


def _write(section_values):
    from tldw_chatbook.config import save_settings_to_cli_config

    assert wizard_state.commit_sections_allowed(section_values), section_values
    assert save_settings_to_cli_config(section_values) is True


class TestCommitRoundTrip:
    def test_provider_and_model_commits_land_in_toml(self, temp_config):
        _write(
            wizard_state.build_provider_commit(
                provider_key="openai", api_key="wizard-test-key-alpha", api_url=None
            )
        )
        _write(
            wizard_state.build_model_commit(
                provider_value="OpenAI", model_id="gpt-5.6-terra"
            )
        )
        config = _reload()
        assert config["api_settings"]["openai"]["api_key"] == "wizard-test-key-alpha"
        assert config["chat_defaults"]["provider"] == "OpenAI"
        assert config["chat_defaults"]["model"] == "gpt-5.6-terra"

    def test_atomic_first_run_provider_commit_lands_as_one_mutation(self, temp_config):
        from tldw_chatbook.Chat.provider_setup_persistence import (
            persist_provider_setup,
        )

        mutation = wizard_state.build_first_run_provider_commit(
            wizard_state.FirstRunProviderDraft(
                provider="llama_cpp",
                endpoint="http://127.0.0.1:8080/v1/chat/completions",
                credential=wizard_state.ProviderCredentialDraft("none", "", 0),
            ),
            "local-model",
            _reload(),
        )

        result = persist_provider_setup(mutation)

        assert result.fully_applied is True
        config = _reload()
        assert config["api_settings"]["llama_cpp"]["api_url"] == (
            "http://127.0.0.1:8080"
        )
        assert config["chat_defaults"]["provider"] == "llama_cpp"
        assert config["chat_defaults"]["model"] == "local-model"
        assert config["provider_setup"]["confirmed"]["llama_cpp"] is True

    def test_wizard_state_flags_land_and_gate_offers(self, temp_config):
        _write(wizard_state.build_wizard_state_commit(started=True))
        config = _reload()
        assert wizard_state.should_offer_wizard(config, {}) is False
        assert wizard_state.should_show_resume_toast(config, {}) is True
        _write(wizard_state.build_wizard_state_commit(completed=True))
        config = _reload()
        assert wizard_state.should_show_resume_toast(config, {}) is False

    def test_rerun_prefill_round_trip_without_secret_leak(self, temp_config):
        _write(
            wizard_state.build_provider_commit(
                provider_key="openai", api_key="wizard-test-key-beta", api_url=None
            )
        )
        _write(
            wizard_state.build_model_commit(
                provider_value="OpenAI", model_id="gpt-5.6-terra"
            )
        )
        config = _reload()
        prefill = wizard_state.read_wizard_prefill(config)
        assert prefill.provider_value == "OpenAI"
        assert "wizard-test-key-beta" not in repr(prefill)
        presence = wizard_state.read_provider_secret_presence(
            config, {}, provider_key="openai"
        )
        assert presence.configured is True
        assert "wizard-test-key-beta" not in repr(presence)

    def test_upgrader_config_never_auto_offers(self, temp_config):
        _write({"api_settings.anthropic": {"api_key": "wizard-test-key-gamma"}})
        config = _reload()
        assert wizard_state.should_offer_wizard(config, {}) is False

    def test_summary_rows_match_persisted_state(self, temp_config):
        config = _reload()
        rows = {
            r.label: r
            for r in wizard_state.build_summary_rows(
                config, {}, rag_deps_installed=False
            )
        }
        assert rows["Notes folder sync"].ok is False
        assert rows["Notes folder sync"].detail == "set up later in Library"


class TestFreshTemplateOfferGuard:
    """UAT regression pin (root cause of the live-app bug): every other test
    in this module builds its ``app_config`` from scratch as a Python dict.
    The shipped ``config.toml`` template (``config.py``'s
    ``CONFIG_TOML_CONTENT``) additionally pre-populates ~12
    ``[api_settings.*]`` blocks with default endpoint URLs (llama.cpp
    ``http://localhost:8080``, Ollama, vLLM, the HuggingFace router, etc.)
    that no synthetic-dict test ever reproduced. Loading the REAL generated
    template via ``temp_config``/``load_cli_config_and_ensure_existence`` is
    the only way to catch a regression where those default endpoints get
    miscounted as "configured" and the wizard silently never auto-offers."""

    def test_fresh_template_offers_wizard(self, temp_config):
        config = _reload()
        assert wizard_state.should_offer_wizard(config, {}) is True

    def test_template_with_one_real_inline_key_does_not_offer(self, temp_config):
        _write(
            wizard_state.build_provider_commit(
                provider_key="openai", api_key="wizard-test-key-epsilon", api_url=None
            )
        )
        config = _reload()
        assert wizard_state.should_offer_wizard(config, {}) is False


class TestFreshTemplateSummaryRow:
    """F2 regression, same "use the REAL generated template" rationale as
    TestFreshTemplateOfferGuard above: the shipped config.toml pre-populates
    chat_defaults.provider="OpenAI" and ~12 [api_settings.*] default
    endpoints, none of them entered by the user. The Summary step's Provider
    row must still read unconfigured against that untouched template, and
    must flip to configured once the wizard's own one-click local-server
    commit (endpoint only, no key) actually lands."""

    def test_pristine_template_provider_row_is_unconfigured(self, temp_config):
        config = _reload()
        rows = {
            r.label: r
            for r in wizard_state.build_summary_rows(
                config, {}, rag_deps_installed=False
            )
        }
        assert rows["Provider"].ok is False

    def test_one_click_local_server_commit_provider_row_is_configured(
        self, temp_config
    ):
        # Mirrors the wizard's own on-mount behavior (FirstRunSetupWizard.on_mount
        # -> _persist_started_flag) plus ProviderStep.commit()'s one-click,
        # no-api-key path (build_provider_commit(api_key=None, api_url=...)
        # followed by invalidate_model_for_provider_change writing
        # chat_defaults.provider).
        _write(wizard_state.build_wizard_state_commit(started=True))
        _write({"api_settings.llama_cpp": {"api_url": "http://127.0.0.1:8080"}})
        _write({"chat_defaults": {"provider": "llama_cpp", "model": ""}})
        config = _reload()
        rows = {
            r.label: r
            for r in wizard_state.build_summary_rows(
                config, {}, rag_deps_installed=False
            )
        }
        assert rows["Provider"].ok is True


class TestLoadSettingsProjectsFirstRun:
    """UAT regression pin (F-E): ``app.py`` repoints ``self.app_config`` at
    ``load_settings()`` (a differently-shaped, hand-curated projection of the
    raw TOML), not at ``load_cli_config_and_ensure_existence()`` directly.
    ``load_settings`` builds its return dict section-by-section (see
    ``config.py``'s ``config_dict = {...}`` literal) and, before this fix,
    never listed ``first_run`` among the sections it passes through -- every
    other section the wizard depends on (``chat_defaults``, ``notes``,
    ``console``, ...) IS listed. Every other test in this module reads back
    via ``load_cli_config_and_ensure_existence`` (the raw loader), which does
    carry ``first_run`` -- masking this exact gap. In the live app, the
    dropped section meant ``should_offer_wizard``/``should_show_resume_toast``
    never saw the persisted flags, so the wizard re-offered on every launch
    even after a real completion."""

    def test_completed_flag_survives_the_load_settings_projection(self, temp_config):
        from tldw_chatbook.config import load_settings

        _write(wizard_state.build_wizard_state_commit(completed=True))
        settings = load_settings(force_reload=True)
        assert settings["first_run"]["setup_completed"] is True
        assert wizard_state.should_offer_wizard(settings, {}) is False

    def test_started_only_flag_still_gates_offer_and_shows_resume_toast(
        self, temp_config
    ):
        from tldw_chatbook.config import load_settings

        _write(wizard_state.build_wizard_state_commit(started=True))
        settings = load_settings(force_reload=True)
        assert settings["first_run"]["setup_started"] is True
        assert wizard_state.should_offer_wizard(settings, {}) is False
        assert wizard_state.should_show_resume_toast(settings, {}) is True


class TestEncryptionAtRest:
    def test_enable_encryption_encrypts_stored_key(self, temp_config):
        from tldw_chatbook.config import enable_config_encryption

        _write(
            wizard_state.build_provider_commit(
                provider_key="openai", api_key="wizard-test-key-delta", api_url=None
            )
        )
        assert enable_config_encryption("integration-test-password") is True
        raw = temp_config.read_text()
        assert "wizard-test-key-delta" not in raw
        assert "enc:" in raw or "password_verifier" in raw


def _typed_provider_draft(
    *,
    provider="custom",
    endpoint="https://example.test/v1/chat/completions",
    source="draft",
    value="integration-secret",
    revision=1,
):
    return wizard_state.FirstRunProviderDraft(
        provider=provider,
        endpoint=endpoint,
        credential=wizard_state.ProviderCredentialDraft(source, value, revision),
    )


def _assert_locked_precondition(precondition) -> None:
    from tldw_chatbook.config import get_atomic_config_snapshot

    assert callable(precondition)
    assert precondition(get_atomic_config_snapshot())


@pytest.mark.usefixtures("temp_config")
class TestWizardAtomicProviderHandoff:
    @pytest.mark.asyncio
    async def test_concurrent_identical_model_commits_share_one_writer(
        self, monkeypatch
    ):
        container = SetupWizardContainer(SimpleNamespace(app_config={}))
        started = threading.Event()
        release = threading.Event()
        call_count = 0

        def writer(
            section_values, *, delete_keys=None, locked_snapshot_precondition=None
        ):
            _assert_locked_precondition(locked_snapshot_precondition)
            nonlocal call_count
            call_count += 1
            started.set()
            assert release.wait(2)
            return ConfigMutationResult(True, True, None)

        monkeypatch.setattr(
            "tldw_chatbook.Chat.provider_setup_persistence."
            "apply_settings_mutation_to_cli_config",
            writer,
        )
        container.stage_provider_setup(_typed_provider_draft())

        first = asyncio.create_task(
            container.commit_staged_provider_setup("custom-model")
        )
        assert await asyncio.to_thread(started.wait, 2)
        second = asyncio.create_task(
            container.commit_staged_provider_setup("custom-model")
        )
        await asyncio.sleep(0)
        release.set()

        assert await asyncio.gather(first, second) == [True, True]
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_restaging_before_writer_starts_invalidates_old_operation(
        self, monkeypatch
    ):
        container = SetupWizardContainer(SimpleNamespace(app_config={}))
        written_models = []

        def writer(
            section_values, *, delete_keys=None, locked_snapshot_precondition=None
        ):
            _assert_locked_precondition(locked_snapshot_precondition)
            written_models.append(section_values["chat_defaults"]["model"])
            return ConfigMutationResult(True, True, None)

        monkeypatch.setattr(
            "tldw_chatbook.Chat.provider_setup_persistence."
            "apply_settings_mutation_to_cli_config",
            writer,
        )
        container.stage_provider_setup(_typed_provider_draft(revision=1))
        stale = asyncio.create_task(
            container.commit_staged_provider_setup("stale-model")
        )
        await asyncio.sleep(0)

        assert container.stage_provider_setup(
            _typed_provider_draft(
                endpoint="https://replacement.test/v1/chat/completions",
                revision=2,
            )
        )
        assert await stale is False
        assert written_models == []

        assert await container.commit_staged_provider_setup("current-model") is True
        assert written_models == ["current-model"]

    @pytest.mark.asyncio
    async def test_newer_model_lease_before_writer_starts_supersedes_old_model(
        self, monkeypatch
    ):
        container = SetupWizardContainer(SimpleNamespace(app_config={}))
        written_models = []

        def writer(
            section_values, *, delete_keys=None, locked_snapshot_precondition=None
        ):
            _assert_locked_precondition(locked_snapshot_precondition)
            written_models.append(section_values["chat_defaults"]["model"])
            return ConfigMutationResult(True, True, None)

        monkeypatch.setattr(
            "tldw_chatbook.Chat.provider_setup_persistence."
            "apply_settings_mutation_to_cli_config",
            writer,
        )
        container.stage_provider_setup(_typed_provider_draft())
        stale = asyncio.create_task(
            container.commit_staged_provider_setup("stale-model")
        )
        await asyncio.sleep(0)
        current = asyncio.create_task(
            container.commit_staged_provider_setup("current-model")
        )

        assert await asyncio.gather(stale, current) == [False, True]
        assert written_models == ["current-model"]

    @pytest.mark.asyncio
    async def test_restaging_is_rejected_while_atomic_writer_is_unavoidable(
        self, monkeypatch
    ):
        container = SetupWizardContainer(SimpleNamespace(app_config={}))
        started = threading.Event()
        release = threading.Event()

        def writer(
            section_values, *, delete_keys=None, locked_snapshot_precondition=None
        ):
            _assert_locked_precondition(locked_snapshot_precondition)
            started.set()
            assert release.wait(2)
            return ConfigMutationResult(True, True, None)

        monkeypatch.setattr(
            "tldw_chatbook.Chat.provider_setup_persistence."
            "apply_settings_mutation_to_cli_config",
            writer,
        )
        original = _typed_provider_draft()
        container.stage_provider_setup(original)
        commit = asyncio.create_task(
            container.commit_staged_provider_setup("custom-model")
        )
        assert await asyncio.to_thread(started.wait, 2)

        assert (
            container.stage_provider_setup(_typed_provider_draft(revision=2)) is False
        )
        assert container.staged_provider_draft is original
        release.set()

        assert await commit is True

    @pytest.mark.asyncio
    async def test_writer_exception_releases_operation_for_retry(self, monkeypatch):
        container = SetupWizardContainer(SimpleNamespace(app_config={}))
        call_count = 0

        def writer(
            section_values, *, delete_keys=None, locked_snapshot_precondition=None
        ):
            _assert_locked_precondition(locked_snapshot_precondition)
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("private writer detail")
            return ConfigMutationResult(True, True, None)

        monkeypatch.setattr(
            "tldw_chatbook.Chat.provider_setup_persistence."
            "apply_settings_mutation_to_cli_config",
            writer,
        )
        container.stage_provider_setup(_typed_provider_draft())

        assert await container.commit_staged_provider_setup("custom-model") is False
        assert await container.commit_staged_provider_setup("custom-model") is True
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_caller_cancellation_does_not_duplicate_inflight_write(
        self, monkeypatch
    ):
        container = SetupWizardContainer(SimpleNamespace(app_config={}))
        started = threading.Event()
        release = threading.Event()
        call_count = 0

        def writer(
            section_values, *, delete_keys=None, locked_snapshot_precondition=None
        ):
            _assert_locked_precondition(locked_snapshot_precondition)
            nonlocal call_count
            call_count += 1
            started.set()
            assert release.wait(2)
            return ConfigMutationResult(True, True, None)

        monkeypatch.setattr(
            "tldw_chatbook.Chat.provider_setup_persistence."
            "apply_settings_mutation_to_cli_config",
            writer,
        )
        container.stage_provider_setup(_typed_provider_draft())
        caller = asyncio.create_task(
            container.commit_staged_provider_setup("custom-model")
        )
        assert await asyncio.to_thread(started.wait, 2)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        follower = asyncio.create_task(
            container.commit_staged_provider_setup("custom-model")
        )
        release.set()

        assert await follower is True
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_model_continue_calls_atomic_writer_once_and_mirrors_full_success(
        self, monkeypatch
    ):
        from tldw_chatbook import config as config_module
        from tldw_chatbook.Chat import provider_setup_persistence as persistence_module

        app_config = {"unrelated": {"keep": True}}
        container = SetupWizardContainer(SimpleNamespace(app_config=app_config))
        calls = []
        real_writer = config_module.apply_settings_mutation_to_cli_config

        def writer(
            section_values, *, delete_keys=None, locked_snapshot_precondition=None
        ):
            calls.append((section_values, delete_keys))
            return real_writer(
                section_values,
                delete_keys=delete_keys,
                locked_snapshot_precondition=locked_snapshot_precondition,
            )

        monkeypatch.setattr(
            persistence_module,
            "apply_settings_mutation_to_cli_config",
            writer,
        )

        assert container.stage_provider_setup(_typed_provider_draft()) is True
        assert app_config == {"unrelated": {"keep": True}}
        assert "not been saved" in container.finish_later_message()

        committed = await container.commit_staged_provider_setup("custom-model")

        assert committed is True
        assert len(calls) == 1
        sections, deletes = calls[0]
        assert set(sections) == {
            "api_settings.custom",
            "chat_defaults",
            "provider_setup.confirmed",
        }
        assert sections["chat_defaults"] == {
            "provider": "custom",
            "model": "custom-model",
        }
        assert sections["api_settings.custom"]["credential_source"] == "stored"
        assert deletes["api_settings.custom"] == ("api_key_env_var",)
        assert app_config["unrelated"] == {"keep": True}
        assert app_config["chat_defaults"] == {
            "provider": "custom",
            "model": "custom-model",
        }
        assert app_config["provider_setup"]["confirmed"]["custom"] is True
        assert container.provider_setup_committed is True
        assert container.committed_provider_model == "custom-model"
        assert "provider and model are saved" in container.finish_later_message()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "result",
        [
            ConfigMutationResult(False, False, None),
            ConfigMutationResult(False, False, "before_replace"),
            ConfigMutationResult(True, False, "cache_reload"),
        ],
    )
    async def test_atomic_failure_never_partially_updates_app_config(
        self, monkeypatch, result
    ):
        app_config = {
            "chat_defaults": {"provider": "openai", "model": "old-model"},
            "api_settings": {"openai": {"api_key": "old-secret"}},
        }
        before = deepcopy(app_config)
        container = SetupWizardContainer(SimpleNamespace(app_config=app_config))
        calls = []

        def writer(
            section_values, *, delete_keys=None, locked_snapshot_precondition=None
        ):
            _assert_locked_precondition(locked_snapshot_precondition)
            calls.append((section_values, delete_keys))
            return result

        monkeypatch.setattr(
            "tldw_chatbook.Chat.provider_setup_persistence."
            "apply_settings_mutation_to_cli_config",
            writer,
        )
        original = _typed_provider_draft()
        container.stage_provider_setup(original)

        committed = await container.commit_staged_provider_setup("custom-model")

        assert committed is False
        assert len(calls) == 1
        assert app_config == before
        assert container.provider_setup_committed is False
        assert container.committed_provider_model == ""
        assert container.staged_provider_draft is original

    @pytest.mark.asyncio
    async def test_atomic_retry_uses_one_writer_per_attempt_and_no_early_mirror(
        self, monkeypatch
    ):
        app_config = {"chat_defaults": {"provider": "openai", "model": "old-model"}}
        before = deepcopy(app_config)
        container = SetupWizardContainer(SimpleNamespace(app_config=app_config))
        results = iter(
            (
                ConfigMutationResult(False, False, "before_replace"),
                ConfigMutationResult(True, True, None),
            )
        )
        call_count = 0

        def writer(
            section_values, *, delete_keys=None, locked_snapshot_precondition=None
        ):
            _assert_locked_precondition(locked_snapshot_precondition)
            nonlocal call_count
            call_count += 1
            return next(results)

        monkeypatch.setattr(
            "tldw_chatbook.Chat.provider_setup_persistence."
            "apply_settings_mutation_to_cli_config",
            writer,
        )
        container.stage_provider_setup(_typed_provider_draft())

        assert await container.commit_staged_provider_setup("custom-model") is False
        assert app_config == before
        assert await container.commit_staged_provider_setup("custom-model") is True

        assert call_count == 2
        assert app_config["chat_defaults"] == {
            "provider": "custom",
            "model": "custom-model",
        }

    @pytest.mark.asyncio
    async def test_repeated_success_for_same_staged_pair_does_not_write_twice(
        self, monkeypatch
    ):
        from tldw_chatbook import config as config_module
        from tldw_chatbook.Chat import provider_setup_persistence as persistence_module

        container = SetupWizardContainer(SimpleNamespace(app_config={}))
        call_count = 0
        real_writer = config_module.apply_settings_mutation_to_cli_config

        def writer(
            section_values, *, delete_keys=None, locked_snapshot_precondition=None
        ):
            nonlocal call_count
            call_count += 1
            return real_writer(
                section_values,
                delete_keys=delete_keys,
                locked_snapshot_precondition=locked_snapshot_precondition,
            )

        monkeypatch.setattr(
            persistence_module,
            "apply_settings_mutation_to_cli_config",
            writer,
        )
        container.stage_provider_setup(_typed_provider_draft())

        assert await container.commit_staged_provider_setup("custom-model") is True
        assert await container.commit_staged_provider_setup("custom-model") is True
        assert call_count == 1

        assert await container.commit_staged_provider_setup("different-model") is True
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_provider_change_invalidates_the_committed_staged_model(
        self, monkeypatch
    ):
        container = SetupWizardContainer(SimpleNamespace(app_config={}))
        def writer(
            section_values, *, delete_keys=None, locked_snapshot_precondition=None
        ):
            del section_values, delete_keys
            _assert_locked_precondition(locked_snapshot_precondition)
            return ConfigMutationResult(True, True, None)

        monkeypatch.setattr(
            "tldw_chatbook.Chat.provider_setup_persistence."
            "apply_settings_mutation_to_cli_config",
            writer,
        )
        container.stage_provider_setup(_typed_provider_draft())
        assert await container.commit_staged_provider_setup("custom-model") is True

        container.stage_provider_setup(
            _typed_provider_draft(
                provider="llama_cpp",
                endpoint="http://127.0.0.1:8080",
                source="none",
                value="",
                revision=2,
            )
        )

        assert container.provider_setup_committed is False
        assert container.committed_provider_model == ""

    @pytest.mark.asyncio
    async def test_uncommitted_provider_checkpoint_resumes_at_provider_without_secret(
        self,
    ):
        app_config = {}
        container = SetupWizardContainer(SimpleNamespace(app_config=app_config))
        captured = []

        async def commit_config(settings, *, delete_keys=None, after_write=None):
            captured.append((settings, delete_keys))
            return True

        container.commit_config = commit_config
        container.wizard_data = {
            "welcome": {"track": "quick"},
            "provider": {
                "provider_key": "custom",
                "provider_value": "custom",
            },
        }
        container.stage_provider_setup(
            _typed_provider_draft(value="checkpoint-integration-secret")
        )

        assert await container.persist_setup_checkpoint("model") is True

        checkpoint = captured[0][0]["first_run"]
        assert checkpoint["active_step_id"] == "provider"
        serialized = json.dumps(checkpoint, sort_keys=True)
        assert "checkpoint-integration-secret" not in serialized
        assert "credential" not in serialized
        assert "example.test" not in serialized

    @pytest.mark.asyncio
    async def test_explicit_clear_deletes_both_credential_forms_after_model_commit(
        self, temp_config, monkeypatch
    ):
        monkeypatch.setenv("CUSTOM_API_KEY", "ambient-clear-canary")
        _write(
            {
                "api_settings.custom": {
                    "api_url": "https://example.test/v1/chat/completions",
                    "api_key": "old-inline-secret",
                    "api_key_env_var": "OLD_CUSTOM_KEY",
                }
            }
        )
        app_config = _reload()
        container = SetupWizardContainer(SimpleNamespace(app_config=app_config))
        container.stage_provider_setup(
            _typed_provider_draft(source="draft", value="", revision=2)
        )

        assert await container.commit_staged_provider_setup("custom-model") is True

        persisted = tomllib.loads(temp_config.read_text())["api_settings"]["custom"]
        assert persisted["api_url"] == "https://example.test/v1/chat/completions"
        assert persisted["credential_source"] == "none"
        assert "api_key" not in persisted
        assert "api_key_env_var" not in persisted
        readiness = get_provider_readiness(
            "custom",
            _reload(),
            environ={"CUSTOM_API_KEY": "ambient-clear-canary"},
        )
        assert readiness.ready is True
        assert readiness.api_key is None
        assert readiness.api_key_source is None

    @pytest.mark.asyncio
    async def test_unchanged_inline_key_outranks_unset_declaration_after_atomic_commit(
        self, temp_config, monkeypatch
    ):
        monkeypatch.delenv("CUSTOM_API_KEY", raising=False)
        _write(
            {
                "api_settings.custom": {
                    "api_url": "https://example.test/v1/chat/completions",
                    "api_key": "active-inline-secret",
                    "api_key_env_var": "CUSTOM_API_KEY",
                }
            }
        )
        app_config = _reload()
        container = SetupWizardContainer(SimpleNamespace(app_config=app_config))
        container.stage_provider_setup(
            _typed_provider_draft(source="none", value="", revision=2)
        )

        assert await container.commit_staged_provider_setup("custom-model") is True

        persisted = tomllib.loads(temp_config.read_text())["api_settings"]["custom"]
        assert persisted["api_key"] == "active-inline-secret"
        assert "api_key_env_var" not in persisted
        readiness = get_provider_readiness("custom", _reload(), environ={})
        assert readiness.ready is True
        assert readiness.api_key_source == "config:api_settings.custom.api_key"

    @pytest.mark.asyncio
    async def test_explicit_environment_source_replaces_inline_key_atomically(
        self, temp_config, monkeypatch
    ):
        monkeypatch.setenv("CUSTOM_API_KEY", "environment-secret")
        _write(
            {
                "api_settings.custom": {
                    "api_url": "https://example.test/v1/chat/completions",
                    "api_key": "old-inline-secret",
                    "api_key_env_var": "CUSTOM_API_KEY",
                }
            }
        )
        app_config = _reload()
        container = SetupWizardContainer(SimpleNamespace(app_config=app_config))
        container.stage_provider_setup(
            _typed_provider_draft(
                source="environment", value="CUSTOM_API_KEY", revision=3
            )
        )

        assert await container.commit_staged_provider_setup("custom-model") is True

        persisted = tomllib.loads(temp_config.read_text())["api_settings"]["custom"]
        assert persisted["api_key_env_var"] == "CUSTOM_API_KEY"
        assert persisted["credential_source"] == "environment"
        assert "api_key" not in persisted
        readiness = get_provider_readiness(
            "custom", _reload(), environ={"CUSTOM_API_KEY": "environment-secret"}
        )
        assert readiness.ready is True
        assert readiness.api_key_source == "env:CUSTOM_API_KEY"

    @pytest.mark.asyncio
    async def test_explicit_keyless_can_be_replaced_after_reload(
        self, temp_config, monkeypatch
    ):
        monkeypatch.setenv("CUSTOM_API_KEY", "ambient-replacement-canary")
        _write(
            {
                "api_settings.custom": {
                    "api_url": "https://example.test/v1/chat/completions",
                    "credential_source": "none",
                }
            }
        )
        replacement = SetupWizardContainer(
            SimpleNamespace(app_config=_reload())
        )
        replacement.stage_provider_setup(
            _typed_provider_draft(
                source="draft",
                value="new-inline-replacement-key",
                revision=4,
            )
        )

        assert await replacement.commit_staged_provider_setup("custom-model") is True

        persisted = tomllib.loads(temp_config.read_text())["api_settings"]["custom"]
        assert persisted["credential_source"] == "stored"
        assert persisted["api_key"] == "new-inline-replacement-key"
        assert "api_key_env_var" not in persisted
        readiness = get_provider_readiness(
            "custom",
            _reload(),
            environ={"CUSTOM_API_KEY": "ambient-replacement-canary"},
        )
        assert readiness.api_key == "new-inline-replacement-key"
        assert readiness.api_key_source == "config:api_settings.custom.api_key"


# -- TASK-33001.5 (D1): every setup exit reaches an untouched Console chat ----
#
# The shipped template default is OpenAI. With an OpenAI key in the
# environment it reads Ready, which is the state whose early return used to
# keep setup's saved provider out of an open, untouched Console chat: only
# "Start chatting" staged a handoff, and Home, Library, Notes and Skip left the
# chat on the template. Each case runs the wizard's own commit and exit code
# against a fresh private profile (real TOML, no mocked config writer).


def _console_on_template_defaults(monkeypatch):
    from unittest.mock import MagicMock

    from tldw_chatbook import config as config_module
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_session_settings import (
        build_console_settings_readiness,
    )
    from tldw_chatbook.UI.Navigation.pending_handoff_store import (
        PendingHandoffStore,
    )
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-template-ready-key")
    config_module.load_cli_config_and_ensure_existence(force_reload=True)
    boot = config_module.load_settings(force_reload=True)
    app_instance = MagicMock(
        app_config=boot,
        pending_handoffs=PendingHandoffStore(),
        console_new_chat_default_generation=0,
    )
    console = ChatScreen(app_instance)
    console._console_chat_store = ConsoleChatStore()
    app_instance.screen_stack = [console]
    template = console._session._ensure_active_console_session_settings()
    assert template.provider.lower() == "openai"
    assert build_console_settings_readiness(
        template, app_config=boot
    ).native_send_supported, "the template chat must read Ready"
    return app_instance, console


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exit_route", ["chat", "home", "library", "library_notes"]
)
@private_profile_test
async def test_saving_setup_exit_moves_an_untouched_console_chat_to_setup_choice(
    exit_route, monkeypatch, request
):
    """AC#3: Start chatting, Home, Library and Notes all land setup's pair."""
    from unittest.mock import MagicMock

    from tldw_chatbook.Chat.provider_readiness import provider_config_key
    from tldw_chatbook.Constants import TAB_CHAT, TAB_HOME, TAB_LIBRARY
    from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel
    from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
        EXIT_ROUTE_LIBRARY_NOTES,
    )

    route = {
        "chat": TAB_CHAT,
        "home": TAB_HOME,
        "library": TAB_LIBRARY,
        "library_notes": EXIT_ROUTE_LIBRARY_NOTES,
    }[exit_route]
    app_instance, console = _console_on_template_defaults(monkeypatch)
    container = SetupWizardContainer(app_instance)
    container._dismiss_screen = MagicMock()
    assert container.stage_provider_setup(
        _typed_provider_draft(
            provider="llama_cpp",
            endpoint="http://127.0.0.1:8080",
            source="none",
            value="",
        )
    )
    assert await container.commit_staged_provider_setup("setup-model.gguf")

    await container._finalize(route)

    container._dismiss_screen.assert_called_once_with(
        {"completed": True, "exit_route": route}
    )
    if route == TAB_CHAT:
        # Console consumes the staged first-chat target when it is shown; that
        # handoff path is pinned elsewhere and unchanged (AC#3).
        assert console._session.consume_pending_console_first_chat_intent(
            defer_presentation=True
        )
    else:
        assert (
            app_instance.pending_handoffs.claim(HandoffChannel.CONSOLE_FIRST_CHAT)
            is None
        ), "only Start chatting stages the first-chat handoff"
    shown = console._session._ensure_active_console_session_settings()
    assert (provider_config_key(shown.provider), shown.model) == (
        "llama_cpp",
        "setup-model.gguf",
    )


@pytest.mark.asyncio
@private_profile_test
async def test_start_chatting_lands_setup_choice_when_a_config_write_beats_the_handoff(
    monkeypatch, request
):
    """AC#3 in the order the live app runs on a fresh profile.

    Start chatting stages the first-chat handoff at the config generation its
    commit published. Console's first mount then writes the missing rail
    scope to the config file (``_ensure_console_rail_scope_seed``) before
    ``on_mount`` consumes the handoff, so the generation fence releases it
    and the handoff's own target is never created (tmux capture, 2026-09-26,
    on BASE and HEAD alike). The untouched chat must still show setup's pair.
    """
    from unittest.mock import MagicMock

    from tldw_chatbook.Chat.provider_readiness import provider_config_key
    from tldw_chatbook.config import save_setting_to_cli_config
    from tldw_chatbook.Constants import TAB_CHAT
    from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel

    app_instance, console = _console_on_template_defaults(monkeypatch)
    container = SetupWizardContainer(app_instance)
    container._dismiss_screen = MagicMock()
    assert container.stage_provider_setup(
        _typed_provider_draft(
            provider="llama_cpp",
            endpoint="http://127.0.0.1:8080",
            source="none",
            value="",
        )
    )
    assert await container.commit_staged_provider_setup("setup-model.gguf")
    await container._finalize(TAB_CHAT)
    claim = app_instance.pending_handoffs.claim(HandoffChannel.CONSOLE_FIRST_CHAT)
    assert claim is not None
    intent = claim.value
    assert app_instance.pending_handoffs.release(claim)

    assert save_setting_to_cli_config(
        "console.rail_state",
        "console_rail_state:global:shared-layout-v1",
        {"left_open": True},
    )

    assert not console._session.consume_pending_console_first_chat_intent(
        defer_presentation=True
    )
    store = console._session._ensure_console_chat_store()
    assert all(session.id != intent.session_id for session in store.sessions())
    shown = console._session._ensure_active_console_session_settings()
    assert (provider_config_key(shown.provider), shown.model) == (
        "llama_cpp",
        "setup-model.gguf",
    )


@pytest.mark.asyncio
@private_profile_test
async def test_skip_leaves_an_untouched_console_chat_on_the_saved_defaults(
    monkeypatch, request
):
    """AC#4: Skip saves no pair; the chat holds what a new blank chat gets."""
    from unittest.mock import MagicMock

    from tldw_chatbook import config as config_module
    from tldw_chatbook.Chat.console_session_settings import (
        blank_console_session_settings,
    )
    from tldw_chatbook.UI.Screens.settings_config_adapter import (
        SettingsConfigAdapter,
    )

    app_instance, console = _console_on_template_defaults(monkeypatch)
    # The saved defaults moved on after this chat was created (the Settings
    # exit's Providers & Models save), then setup is skipped.
    adapter = SettingsConfigAdapter()
    assert adapter.save_values(
        "api_settings.llama_cpp", {"api_url": "http://127.0.0.1:8080"}
    )
    assert adapter.save_values(
        "chat_defaults", {"provider": "llama_cpp", "model": "settings-model.gguf"}
    )
    saved_pair = dict(config_module.load_settings()["chat_defaults"])
    container = SetupWizardContainer(app_instance)
    container._dismiss_screen = MagicMock()

    await container._skip_entirely()

    container._dismiss_screen.assert_called_once_with(
        {"completed": True, "exit_route": None}
    )
    config = config_module.load_settings()
    assert config["first_run"]["setup_completed"] is True
    assert config["chat_defaults"] == saved_pair, "Skip must save no provider/model"
    shown = console._session._ensure_active_console_session_settings()
    assert shown == blank_console_session_settings(config)
    assert (shown.provider, shown.model) == ("llama_cpp", "settings-model.gguf")

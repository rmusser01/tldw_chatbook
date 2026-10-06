"""TASK-34100.13: the Full track's Search step does what it says.

fulltrack-01: the old RAG step wrote ``[embedding_config] default_model_id``,
which the RAG pipeline never reads, and the Summary then showed "✓ RAG".
The step now has one real control -- whether new Console conversations
search the Library on every send (``[chat_defaults]
rag_auto_retrieve_on_send``, the key Console and Settings use) -- and shows
the embedding model RAG actually resolves, read-only.

fulltrack-11: without the extras the step is no longer a dead page of pip
instructions. coverage-13: the shipped template's ``[embedding_config.models]``
table parses into complete rows.
"""

from __future__ import annotations

import tomllib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, RadioButton, RadioSet, Static

from tldw_chatbook.UI.Wizards import first_run_rag_state as rag_state
from tldw_chatbook.UI.Wizards import first_run_setup_state as wizard_state
from tldw_chatbook.UI.Wizards.BaseWizard import WizardStepConfig
from tldw_chatbook.UI.Wizards.first_run_rag_step import RagStep

LOCAL_MODEL = rag_state.EmbeddingModelFacts(name="all-MiniLM-L6-v2", cloud=False)
CLOUD_MODEL = rag_state.EmbeddingModelFacts(
    name="openai/text-embedding-3-small", cloud=True
)


class _StepHost(App):
    def __init__(self, step):
        super().__init__()
        self._step = step

    def compose(self) -> ComposeResult:
        yield self._step


def _rag_step(app_config, *, deps=True, model=LOCAL_MODEL):
    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config=app_config),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = RagStep(
        wizard=wizard,
        config=WizardStepConfig(id="rag", title="Search", step_number=5),
        deps_installed=lambda: deps,
        resolve_model=lambda: model,
    )
    return wizard, step


def _rendered(step) -> str:
    return "\n".join(str(static.render()) for static in step.query(Static))


async def _settle(pilot, step) -> None:
    for _ in range(20):
        await pilot.pause(0.02)
        if "checking" not in _rendered(step).lower():
            return


# -- AC#1: what the step writes ---------------------------------------------


@pytest.mark.asyncio
async def test_untouched_search_step_writes_nothing() -> None:
    wizard, step = _rag_step({"chat_defaults": {"rag_auto_retrieve_on_send": False}})
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await _settle(pilot, step)
        ok, error = await step.commit()
    assert (ok, error) == (True, "")
    wizard.commit_config.assert_not_awaited()


@pytest.mark.asyncio
async def test_choosing_automatic_writes_only_the_console_auto_retrieve_flag() -> None:
    wizard, step = _rag_step({"chat_defaults": {"rag_auto_retrieve_on_send": False}})
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await _settle(pilot, step)
        step.query_one("#setup-rag-auto-retrieve-automatic", RadioButton).value = True
        await pilot.pause()
        ok, error = await step.commit()
    assert (ok, error) == (True, "")
    committed = wizard.commit_config.await_args.args[0]
    assert committed == {"chat_defaults": {"rag_auto_retrieve_on_send": True}}
    assert step.get_step_data() == {"auto_retrieve": True}


@pytest.mark.asyncio
async def test_rerun_prefills_automatic_and_turning_it_off_writes_false() -> None:
    wizard, step = _rag_step({"chat_defaults": {"rag_auto_retrieve_on_send": True}})
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await _settle(pilot, step)
        assert step.query_one("#setup-rag-auto-retrieve-automatic", RadioButton).value
        step.query_one("#setup-rag-auto-retrieve-on-request", RadioButton).value = True
        await pilot.pause()
        await step.commit()
    committed = wizard.commit_config.await_args.args[0]
    assert committed == {"chat_defaults": {"rag_auto_retrieve_on_send": False}}


@pytest.mark.asyncio
async def test_step_shows_the_resolved_model_read_only_and_names_settings() -> None:
    _wizard, step = _rag_step(
        {"embedding_config": {"models": {"e5-small-v2": {}}, "default_model_id": "x"}}
    )
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await _settle(pilot, step)
        body = _rendered(step)
        # No embedding-model picker: the legacy table is not rendered at all.
        assert not step.query("#setup-rag-model-choice")
        assert "e5-small-v2" not in body
    assert "all-MiniLM-L6-v2" in body
    assert "on this computer" in body
    assert "Settings ▸ Domain Defaults ▸ RAG" in body
    assert "leaves your device" not in body  # a local model sends nothing


@pytest.mark.asyncio
async def test_a_cloud_embedding_model_carries_settings_privacy_sentence() -> None:
    from tldw_chatbook.UI.Screens.settings_library_rag_defaults import (
        LIBRARY_CLOUD_MODEL_PRIVACY_COPY,
    )

    _wizard, step = _rag_step({}, model=CLOUD_MODEL)
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await _settle(pilot, step)
        body = _rendered(step)
    assert "openai/text-embedding-3-small" in body
    assert LIBRARY_CLOUD_MODEL_PRIVACY_COPY in body


def test_settings_privacy_sentence_is_the_one_settings_renders() -> None:
    from tldw_chatbook.UI.Screens.settings_library_rag_defaults import (
        CONSOLE_DIRECT_LIBRARY_TOOLS_COPY,
        LIBRARY_CLOUD_MODEL_PRIVACY_COPY,
    )

    assert LIBRARY_CLOUD_MODEL_PRIVACY_COPY in CONSOLE_DIRECT_LIBRARY_TOOLS_COPY


def test_wizard_no_longer_owns_the_legacy_embedding_section() -> None:
    assert "embedding_config" not in wizard_state.WIZARD_OWNED_SECTIONS
    assert not hasattr(wizard_state, "build_rag_commit")


# -- AC#1: the Summary row comes from the flag plus the resolved model --------


def _rows(app_config, **kwargs):
    return {
        row.label: row
        for row in wizard_state.build_summary_rows(
            app_config, {}, rag_deps_installed=True, **kwargs
        )
    }


def test_summary_search_row_reads_the_flag_and_resolved_model_not_the_legacy_key() -> None:
    config = {
        "chat_defaults": {"rag_auto_retrieve_on_send": False},
        "embedding_config": {"default_model_id": "openai-text-embedding-3-small"},
    }
    rows = _rows(config, rag_model=LOCAL_MODEL)
    assert "RAG" not in rows
    row = rows["Search"]
    assert row.state == wizard_state.ROW_DEFAULT
    assert row.detail == "only when you ask · all-MiniLM-L6-v2 (on this computer)"
    assert "openai-text-embedding-3-small" not in row.detail


def test_summary_search_row_ticks_only_for_automatic_retrieval() -> None:
    config = {"chat_defaults": {"rag_auto_retrieve_on_send": True}}
    row = _rows(config, rag_model=CLOUD_MODEL)["Search"]
    assert row.state == wizard_state.ROW_CONFIGURED
    assert row.detail == (
        "automatic when you chat · openai/text-embedding-3-small (OpenAI API)"
    )


def test_an_openai_model_on_a_loopback_server_is_local() -> None:
    facts = rag_state.describe_embedding_model(
        "openai/nomic", base_url="http://127.0.0.1:8080/v1"
    )
    assert facts.cloud is False
    assert rag_state.embedding_location_phrase(facts) == "a server on this computer"


# -- AC#5: the extras are missing ---------------------------------------------


@pytest.mark.asyncio
async def test_missing_extras_keep_the_control_and_demote_the_package_name() -> None:
    wizard, step = _rag_step({}, deps=False)
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await _settle(pilot, step)
        body = _rendered(step)
        choice = step.query_one("#setup-rag-auto-retrieve", RadioSet)
        assert choice.display
        on_request = step.query_one("#setup-rag-auto-retrieve-on-request", RadioButton)
        assert on_request.value and not on_request.disabled
        copy = step.query_one("#setup-rag-copy-extras", Button)
        assert copy.display
        ok, _ = await step.commit()
    assert ok
    wizard.commit_config.assert_not_awaited()
    assert rag_state.RAG_EXTRAS_PACKAGE in body
    assert "pip" not in body.lower()
    assert "Settings ▸ Domain Defaults ▸ RAG" in body


@pytest.mark.asyncio
async def test_copy_action_copies_the_extras_package_name() -> None:
    _wizard, step = _rag_step({}, deps=False)
    app = _StepHost(step)
    copied: list[str] = []
    async with app.run_test(size=(120, 40)) as pilot:
        await _settle(pilot, step)
        app.copy_to_clipboard = copied.append  # type: ignore[method-assign]
        step.query_one("#setup-rag-copy-extras", Button).press()
        await pilot.pause()
    assert copied == [rag_state.RAG_EXTRAS_PACKAGE]


# -- AC#6: the shipped template ------------------------------------------------


def _rendered_template() -> dict:
    from tldw_chatbook import config as config_module

    return tomllib.loads(config_module.CONFIG_TOML_CONTENT)


def test_template_embedding_models_are_complete_rows() -> None:
    models = _rendered_template()["embedding_config"]["models"]
    assert "bge-small-en-v1.5" in models and "bge-base-en-v1.5" in models
    for model_id, entry in models.items():
        assert isinstance(entry, dict), model_id
        assert entry.get("provider"), model_id
        assert entry.get("model_name_or_path"), model_id


def test_template_openai_rows_have_real_dimensions_and_no_key_placeholder() -> None:
    models = _rendered_template()["embedding_config"]["models"]
    assert models["openai-text-embedding-3-small"]["dimension"] == 1536
    assert models["openai-text-embedding-3-large"]["dimension"] == 3072
    for entry in models.values():
        assert "api_key" not in entry
    # The two editable samples ship commented out, not as live choices.
    assert "my-local-nomic-model" not in models
    assert "local-llama-cpp-embeddings" not in models


def test_listable_embedding_models_hide_incomplete_and_sample_rows() -> None:
    from tldw_chatbook.config import listable_embedding_models

    models = {
        "bge-small-en-v1": {"5": {"provider": "huggingface"}},
        "my-local-nomic-model": {"provider": "openai", "model_name_or_path": "n"},
        "e5-small-v2": {"provider": "huggingface", "model_name_or_path": "e5"},
    }
    assert list(listable_embedding_models(models)) == ["e5-small-v2"]

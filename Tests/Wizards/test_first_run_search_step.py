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


DISTINCT_MODEL = "openai/zz-review-only-embedder"


def _patch_active_rag_config(monkeypatch, model=DISTINCT_MODEL, base_url=None):
    """Make the real RAG profile resolver return a model no default could."""
    from tldw_chatbook.RAG_Search.simplified import active_config

    calls = []

    def resolve():
        calls.append(1)
        return SimpleNamespace(
            embedding=SimpleNamespace(model=model, base_url=base_url)
        )

    monkeypatch.setattr(active_config, "resolve_active_rag_config", resolve)
    return calls


def test_the_resolver_reads_the_active_rag_profile(monkeypatch) -> None:
    """review-tests F3: the one link to ``resolve_active_rag_config()``.

    Every other test injects the model; a resolver that returned a fixed
    default without reading the profile passed them all.
    """
    calls = _patch_active_rag_config(monkeypatch)
    facts = rag_state.resolve_active_embedding_model()
    assert calls == [1]
    assert facts == rag_state.EmbeddingModelFacts(name=DISTINCT_MODEL, cloud=True)


def test_an_unresolvable_rag_profile_names_settings_instead(monkeypatch) -> None:
    from tldw_chatbook.RAG_Search.simplified import active_config

    def broken():
        raise RuntimeError("profile unreadable")

    monkeypatch.setattr(active_config, "resolve_active_rag_config", broken)
    assert rag_state.resolve_active_embedding_model() is None
    assert rag_state.embedding_model_line(None) == (
        f"Embedding model: see {rag_state.SEARCH_SETTINGS_PATH}."
    )


@pytest.mark.asyncio
async def test_a_step_built_without_a_resolver_shows_the_profiles_model(
    monkeypatch,
) -> None:
    """The wizard builds RagStep without ``resolve_model``: the default path."""
    _patch_active_rag_config(monkeypatch)
    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
    )
    step = RagStep(
        wizard=wizard,
        config=WizardStepConfig(id="rag", title="Search", step_number=5),
        deps_installed=lambda: True,
    )
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await _settle(pilot, step)
        line = str(step.query_one("#setup-rag-model", Static).render())
    assert DISTINCT_MODEL in line
    assert "OpenAI API" in line


@pytest.mark.asyncio
async def test_the_summary_search_row_is_repainted_with_the_resolved_model() -> None:
    """review-tests F3: the Summary paints first, then once the model resolves."""
    from tldw_chatbook.UI.Wizards.first_run_summary_step import SummaryStep

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
        wizard_data={"welcome": {"track": "full"}},
    )
    facts = rag_state.EmbeddingModelFacts(name=DISTINCT_MODEL, cloud=True)
    step = SummaryStep(
        wizard=wizard,
        config=WizardStepConfig(id="summary", title="Summary", step_number=10),
        load_config=lambda: {"chat_defaults": {"rag_auto_retrieve_on_send": True}},
        rag_deps_installed=lambda: True,
        speech_installed=lambda: False,
        speech_runtime_installed=lambda: False,
        rag_model=lambda: facts,
    )
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        rows = ""
        for _ in range(40):
            await pilot.pause(0.05)
            rows = str(step.query_one("#setup-summary-rows", Static).render())
            if DISTINCT_MODEL in rows:
                break
    search = next(line for line in rows.splitlines() if " Search — " in line)
    assert search.endswith(
        f"Search — automatic when you chat · {DISTINCT_MODEL} (OpenAI API)"
    )


@pytest.mark.asyncio
async def test_a_summary_whose_rows_are_pruned_mid_load_paints_nothing(monkeypatch) -> None:
    """Review TE-2 (TASK-34100.13): under load the mounted-wizard tests failed
    at run_test exit with NoMatches('#setup-exit-chat') raised by the
    Summary's row paint. App teardown prunes a step's children before the
    step's own worker is cancelled, so ``is_attached`` is still true while
    ``_paint_rows`` queries widgets that are gone; the branch's extra
    executor hop (the installed-engines probe) widened that window. A paint
    over a removed subtree is now a no-op, as the Quick note's already is."""
    import threading

    from tldw_chatbook.UI.Wizards.first_run_summary_step import SummaryStep
    from tldw_chatbook.Utils import local_stt_providers

    entered, release = threading.Event(), threading.Event()

    def slow_probe():
        entered.set()
        release.wait(5)
        return ()

    monkeypatch.setattr(local_stt_providers, "installed_local_providers", slow_probe)
    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
        wizard_data={"welcome": {"track": "full"}},
    )
    step = SummaryStep(
        wizard=wizard,
        config=WizardStepConfig(id="summary", title="Summary", step_number=10),
        load_config=lambda: {},
        rag_deps_installed=lambda: True,
        speech_installed=lambda: False,
        speech_runtime_installed=lambda: False,
        rag_model=lambda: None,
    )
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        step.on_show()
        for _ in range(100):
            if entered.is_set():
                break
            await pilot.pause(0.02)
        assert entered.is_set(), "the Summary never started loading its rows"
        await step.remove_children()
        assert step.is_attached
        release.set()
        worker = step._render_worker
        for _ in range(100):
            if worker is None or worker.is_finished:
                break
            await pilot.pause(0.02)
        error = worker.error if worker is not None else None
    assert error is None, f"the row paint raised {error!r}"


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
async def test_a_hidden_search_step_resolves_no_embedding_model(monkeypatch) -> None:
    """Review TE-2 (TASK-34100.13): the Search step resolved RAG's embedding
    model from on_mount, so every wizard -- the Quick track included, which
    never shows the step -- imported the RAG config chain (~0.5 s cold) on an
    executor thread during its first render. That GIL-bound import starved
    the loop: on a cold process the reflexive double Escape (TASK-2314)
    reached the confirm dialog after its 0.5 s grace and dismissed it, every
    time. The model is now resolved when the step is first shown."""
    from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import (
        FirstRunSetupWizard,
        SetupWizardContainer,
    )

    calls: list[int] = []

    def spy():
        calls.append(1)
        return LOCAL_MODEL

    monkeypatch.setattr(rag_state, "resolve_active_embedding_model", spy)
    app_instance = MagicMock()
    app_instance.app_config = {}
    wizard = FirstRunSetupWizard(app_instance)

    class _WizardHost(App):
        def compose(self) -> ComposeResult:
            yield from ()

        async def on_mount(self) -> None:
            self.push_screen(wizard)

    async with _WizardHost().run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.5)
        while_hidden = len(calls)
        container = wizard.query_one(SetupWizardContainer)
        container.select_track(wizard_state.TRACK_FULL)
        search = next(step for step in container.steps if isinstance(step, RagStep))
        container.show_step(container.steps.index(search))
        for _ in range(100):
            if calls:
                break
            await pilot.pause(0.02)
        container.show_step(0)
        await pilot.pause()
        container.show_step(container.steps.index(search))
        await pilot.pause(0.2)
    assert while_hidden == 0
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_commit_reads_the_pressed_radio_without_a_changed_event() -> None:
    """review-tests F6 (the F-A pattern, ported to the new radio): a press
    whose ``RadioSet.Changed`` never reached the step is still what Next
    saves."""
    wizard, step = _rag_step({"chat_defaults": {"rag_auto_retrieve_on_send": False}})
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await _settle(pilot, step)
        radio = step.query_one("#setup-rag-auto-retrieve", RadioSet)
        automatic = step.query_one("#setup-rag-auto-retrieve-automatic", RadioButton)
        on_request = step.query_one("#setup-rag-auto-retrieve-on-request", RadioButton)
        with radio.prevent(RadioButton.Changed, RadioSet.Changed):
            on_request.value = False
            automatic.value = True
        radio._pressed_button = automatic
        await pilot.pause()
        assert step.auto_retrieve is False  # the step never heard about it
        ok, _ = await step.commit()
    assert ok
    wizard.commit_config.assert_awaited_once_with(
        {"chat_defaults": {"rag_auto_retrieve_on_send": True}}
    )


def test_an_openai_model_on_a_loopback_server_is_local() -> None:
    facts = rag_state.describe_embedding_model(
        "openai/nomic", base_url="http://127.0.0.1:8080/v1"
    )
    assert facts.cloud is False
    assert rag_state.embedding_location_phrase(facts) == "a server on this computer"


# -- AC#5: the extras are missing ---------------------------------------------
# Owner ruling 2026-10-20: the auto-retrieve control is shown as UNAVAILABLE
# and says why (Console's automatic search needs the extras); the extras
# package is a secondary line with a copy action, no pip, the real Settings
# path. A keyword fallback for Console is TASK-34100.35, not this task.


def _children_ids(step) -> list[str]:
    """The ids of the Search step's top-level rows, top to bottom."""
    body = step.query_one(".setup-rag")
    return [child.id or "" for child in body.children]


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
async def test_without_the_extras_automatic_search_reads_unavailable_and_says_why() -> None:
    _wizard, step = _rag_step({}, deps=False)
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await _settle(pilot, step)
        automatic = step.query_one("#setup-rag-auto-retrieve-automatic", RadioButton)
        label, disabled = str(automatic.label), automatic.disabled
        rows = _children_ids(step)
        why = str(step.query_one("#setup-rag-unavailable", Static).render())
    assert disabled
    assert label == "Search my Library automatically — unavailable"
    # The reason sits directly under the control it explains.
    assert rows[rows.index("setup-rag-auto-retrieve") + 1] == "setup-rag-unavailable"
    assert why.startswith("Unavailable: Console's automatic search needs the")
    assert "optional search extras, which aren't installed" in why
    assert "Library ▸ Search / RAG" in why


@pytest.mark.asyncio
async def test_without_the_extras_the_package_is_a_secondary_line_after_the_reason() -> None:
    _wizard, step = _rag_step({}, deps=False)
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await _settle(pilot, step)
        rows = _children_ids(step)
        package = str(step.query_one("#setup-rag-extras-package", Static).render())
        hint = str(step.query_one("#setup-rag-extras", Static).render())
        body = _rendered(step)
    assert (
        rows.index("setup-rag-unavailable")
        < rows.index("setup-rag-extras-row")
        < rows.index("setup-rag-extras")
    )
    assert package == rag_state.RAG_EXTRAS_PACKAGE
    assert "with the tool you installed chatbook with" in hint
    assert "restart chatbook" in hint
    assert "Automatic retrieval in Settings ▸ Domain Defaults ▸ RAG" in hint
    assert "pip" not in body.lower()


@pytest.mark.asyncio
async def test_without_the_extras_a_saved_automatic_reads_unavailable_and_can_be_turned_off() -> None:
    """A re-run where automatic search was saved before the extras went away:
    each Console send stops at "Library search failed" (the search runs in
    rag mode, which is blocked without the runtime)."""
    wizard, step = _rag_step(
        {"chat_defaults": {"rag_auto_retrieve_on_send": True}}, deps=False
    )
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await _settle(pilot, step)
        automatic = step.query_one("#setup-rag-auto-retrieve-automatic", RadioButton)
        pressed, label = automatic.value, str(automatic.label)
        why = str(step.query_one("#setup-rag-unavailable", Static).render())
        download_shown = step.query_one("#setup-rag-download", Static).display
        step.query_one("#setup-rag-auto-retrieve-on-request", RadioButton).value = True
        await pilot.pause()
        ok, _ = await step.commit()
    assert pressed and label == "Search my Library automatically — unavailable"
    assert why.startswith("Unavailable: automatic search is on, but")
    assert '"Library search failed"' in why
    assert '"only when I ask"' in why
    assert not download_shown  # nothing is downloaded without the extras
    assert ok
    wizard.commit_config.assert_awaited_once_with(
        {"chat_defaults": {"rag_auto_retrieve_on_send": False}}
    )


@pytest.mark.asyncio
async def test_the_unavailable_reason_follows_a_saved_change_back_to_the_step() -> None:
    """Review R8 (TASK-34100.13): the reason line was chosen once, at compose.
    After Next saved "only when I ask", Back showed the step still saying
    automatic search is on and every message stops at "Library search failed"."""
    from tldw_chatbook.UI.Wizards import first_run_rag_state as rag_state

    _wizard, step = _rag_step(
        {"chat_defaults": {"rag_auto_retrieve_on_send": True}}, deps=False
    )
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await _settle(pilot, step)
        step.query_one("#setup-rag-auto-retrieve-on-request", RadioButton).value = True
        await pilot.pause()
        ok, _ = await step.commit()
        await pilot.pause()
        why = str(step.query_one("#setup-rag-unavailable", Static).render())
    assert ok
    assert why == rag_state.search_unavailable_copy(saved_automatic=False)
    assert '"Library search failed"' not in why


@pytest.mark.asyncio
async def test_with_the_extras_automatic_search_is_offered_plainly() -> None:
    _wizard, step = _rag_step({}, deps=True)
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await _settle(pilot, step)
        automatic = step.query_one("#setup-rag-auto-retrieve-automatic", RadioButton)
        label, disabled = str(automatic.label), automatic.disabled
        rows = _children_ids(step)
    assert (label, disabled) == (rag_state.SEARCH_AUTOMATIC_LABEL, False)
    assert "setup-rag-unavailable" not in rows
    assert "setup-rag-extras-row" not in rows


def test_summary_search_row_says_automatic_search_is_unavailable_without_the_extras() -> None:
    def row(auto):
        config = {"chat_defaults": {"rag_auto_retrieve_on_send": auto}}
        rows = wizard_state.build_summary_rows(
            config, {}, rag_deps_installed=False, rag_model=LOCAL_MODEL
        )
        return next(r for r in rows if r.label == "Search")

    off, on = row(False), row(True)
    assert off.state == wizard_state.ROW_DEFAULT
    assert off.detail == (
        "only when you ask · automatic search unavailable: the optional "
        "search extras aren't installed"
    )
    # A saved "automatic" that cannot run is not a tick.
    assert on.state == wizard_state.ROW_ATTENTION
    assert on.detail == (
        "automatic when you chat — unavailable: the optional search extras "
        "aren't installed"
    )


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


@pytest.mark.asyncio
async def test_choosing_automatic_says_the_first_search_may_download_a_local_model() -> None:
    """Live: the first automatic search on a fresh profile fetched the model."""
    _wizard, step = _rag_step({})
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await _settle(pilot, step)
        note = step.query_one("#setup-rag-download", Static)
        assert not note.display  # "only when I ask" fetches nothing
        step.query_one("#setup-rag-auto-retrieve-automatic", RadioButton).value = True
        await pilot.pause()
        assert note.display
        text = str(note.render())
    assert "all-MiniLM-L6-v2" in text and "downloads it" in text
    # Review G13-R1-02: a download longer than Console's 60 s start-up budget
    # pauses the send (library_retrieval_timeout) with Retry -- not only a wait.
    assert '"Library search timed out"' in text and "Retry" in text


def test_a_cloud_model_has_no_download_note() -> None:
    assert rag_state.automatic_download_note(CLOUD_MODEL) == ""


# -- tier-2 review S21, restored (TASK-34100.13 review TEF-4) ----------------
#
# The embedding model id comes from the user's active RAG profile, and the
# step and the Summary paint it into Statics. Textual parses a Static's text
# as markup unless it is built with markup=False: "[b]" silently vanishes and
# "[/x]" raises MarkupError (inside a worker here, so the line just stays on
# "checking…"). The old S21 test covered the retired picker; these cover the
# read-only line, the download note and the Summary row that replaced it.

MARKUP_SHAPED_IDS = ("embed-[b]-v1", "embed-[/x]-v2")


@pytest.mark.asyncio
@pytest.mark.parametrize("model_id", MARKUP_SHAPED_IDS)
async def test_a_markup_shaped_model_id_renders_literally_on_the_step(
    model_id,
) -> None:
    model = rag_state.EmbeddingModelFacts(name=model_id, cloud=False)
    _wizard, step = _rag_step({}, model=model)
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        await _settle(pilot, step)
        line = str(step.query_one("#setup-rag-model", Static).render())
        step.query_one("#setup-rag-auto-retrieve-automatic", RadioButton).value = True
        await pilot.pause()
        note = step.query_one("#setup-rag-download", Static)
        assert note.display
        note_text = str(note.render())
    assert f"Embedding model: {model_id} · " in line
    assert f"If {model_id} isn't on this computer yet" in note_text


@pytest.mark.asyncio
@pytest.mark.parametrize("model_id", MARKUP_SHAPED_IDS)
async def test_a_markup_shaped_model_id_renders_literally_in_the_summary_row(
    model_id,
) -> None:
    from tldw_chatbook.UI.Wizards.first_run_summary_step import SummaryStep

    wizard = SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        commit_config=AsyncMock(return_value=True),
        rerun=False,
        wizard_data={"welcome": {"track": "full"}},
    )
    facts = rag_state.EmbeddingModelFacts(name=model_id, cloud=False)
    step = SummaryStep(
        wizard=wizard,
        config=WizardStepConfig(id="summary", title="Summary", step_number=10),
        load_config=lambda: {"chat_defaults": {"rag_auto_retrieve_on_send": True}},
        rag_deps_installed=lambda: True,
        speech_installed=lambda: False,
        speech_runtime_installed=lambda: False,
        rag_model=lambda: facts,
    )
    async with _StepHost(step).run_test(size=(120, 40)) as pilot:
        rows = ""
        for _ in range(40):
            await pilot.pause(0.05)
            rows = str(step.query_one("#setup-summary-rows", Static).render())
            if "(on this computer)" in rows:
                break
    search = next(line for line in rows.splitlines() if " Search — " in line)
    assert search.endswith(
        f"Search — automatic when you chat · {model_id} (on this computer)"
    )

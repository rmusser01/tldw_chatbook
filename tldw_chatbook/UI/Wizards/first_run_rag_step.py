"""The first-run wizard's Search step (step id ``rag``).

Moved whole out of ``FirstRunSetupWizard.py`` (TASK-34100.1), so its ``@on``
handlers and workers stay registered on the class and fixes to this step have
room under the size ratchet. ``FirstRunSetupWizard`` still exports the class.
Patch this module, not the wizard, to replace what the step calls.

TASK-34100.13 (fulltrack-01, fulltrack-11): the step used to pick an
``[embedding_config]`` model the RAG pipeline never reads. It now offers the
one setting that changes what happens when you chat -- whether new Console
conversations search the Library on every send -- and shows the embedding
model RAG actually resolves, read-only. The facts and copy live in
``first_run_rag_state``.
"""

from __future__ import annotations

import asyncio
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
)

from textual import on
from textual.app import ComposeResult
from textual.containers import (
    Horizontal,
    Vertical,
)
from textual.css.query import NoMatches
from textual.widgets import (
    Button,
    Label,
    RadioButton,
    RadioSet,
    Static,
)

from tldw_chatbook.UI.Wizards import first_run_rag_state as rag_state
from tldw_chatbook.UI.Wizards.first_run_setup_widgets import (
    SetupRadioButton,
    SetupRadioSet,
    SetupStep,
)
from tldw_chatbook.UI.Wizards.first_run_step_guard import run_wizard_worker


class RagStep(SetupStep):
    """Search: when new chats search the Library, and the model RAG uses."""

    def __init__(
        self,
        wizard=None,
        config=None,
        *,
        deps_installed: Optional[Callable[[], bool]] = None,
        resolve_model: Optional[
            Callable[[], Optional[rag_state.EmbeddingModelFacts]]
        ] = None,
        **kwargs: Any,
    ):
        super().__init__(wizard=wizard, config=config, **kwargs)
        if deps_installed is None:
            from tldw_chatbook.Utils.optional_deps import embeddings_rag_deps_installed

            deps_installed = embeddings_rag_deps_installed
        self._deps_installed = deps_installed
        self._resolve_model = resolve_model or rag_state.resolve_active_embedding_model
        self._model_worker = None
        app_config = getattr(self.wizard.app_instance, "app_config", {}) or {}
        self._current_auto_retrieve = rag_state.read_auto_retrieve(app_config)
        self.auto_retrieve: bool = self._current_auto_retrieve

    def compose_step(self) -> ComposeResult:
        deps = bool(self._deps_installed())
        with Vertical(classes="setup-rag"):
            yield Static(rag_state.SEARCH_TITLE, classes="setup-title")
            yield Static(rag_state.SEARCH_SUBTITLE, classes="setup-subtitle")
            yield Label(rag_state.SEARCH_CHOICE_LABEL, classes="setup-field-label")
            with SetupRadioSet(
                id="setup-rag-auto-retrieve",
                classes="setup-choice-list setup-choice-fit",
            ):
                yield SetupRadioButton(
                    rag_state.SEARCH_ON_REQUEST_LABEL,
                    value=not self.auto_retrieve,
                    id="setup-rag-auto-retrieve-on-request",
                )
                yield SetupRadioButton(
                    rag_state.SEARCH_AUTOMATIC_LABEL,
                    value=self.auto_retrieve,
                    id="setup-rag-auto-retrieve-automatic",
                    # Without the extras every automatic search is blocked
                    # and each send would pause; a saved "automatic" stays
                    # selectable so a re-run can still turn it off.
                    disabled=not deps and not self.auto_retrieve,
                )
            yield Static(
                "Embedding model: checking…",
                id="setup-rag-model",
                classes="setup-subtitle",
                markup=False,
            )
            privacy = Static(
                "", id="setup-rag-privacy", classes="setup-step-note", markup=False
            )
            privacy.display = False
            yield privacy
            download = Static(
                "", id="setup-rag-download", classes="setup-step-note", markup=False
            )
            download.display = False
            yield download
            if not deps:
                yield Static(
                    rag_state.SEARCH_EXTRAS_MISSING_COPY,
                    id="setup-rag-extras",
                    classes="setup-step-note",
                    markup=False,
                )
                with Horizontal(classes="setup-rag-extras-row"):
                    yield Static(
                        rag_state.RAG_EXTRAS_PACKAGE,
                        id="setup-rag-extras-package",
                        markup=False,
                    )
                    yield Button(
                        "Copy package name",
                        id="setup-rag-copy-extras",
                        classes="setup-tertiary-button",
                    )

    def on_mount(self) -> None:
        self._model_worker = run_wizard_worker(
            self,
            self._show_resolved_model(),
            exclusive=True,
            group="setup-rag-model",
        )

    async def _show_resolved_model(self) -> None:
        facts = await asyncio.get_running_loop().run_in_executor(
            None, self._resolve_model
        )
        # The step can leave the DOM while the resolver runs (TASK-32892).
        try:
            line = self.query_one("#setup-rag-model", Static)
            privacy = self.query_one("#setup-rag-privacy", Static)
        except NoMatches:
            return
        line.update(rag_state.embedding_model_line(facts))
        if facts is not None and facts.cloud:
            privacy.update(rag_state.cloud_privacy_copy())
            privacy.display = True
        else:
            privacy.display = False
        self._facts = facts
        self._show_download_note()

    _facts: Optional[rag_state.EmbeddingModelFacts] = None

    def _show_download_note(self) -> None:
        """Under "automatically": what the first automatic search can fetch."""
        try:
            note = self.query_one("#setup-rag-download", Static)
        except NoMatches:
            return
        text = rag_state.automatic_download_note(self._facts)
        note.update(text)
        note.display = bool(text) and self._pressed_is_automatic()

    @on(RadioSet.Changed, "#setup-rag-auto-retrieve")
    def _on_auto_retrieve(self, event: RadioSet.Changed) -> None:
        event.stop()
        self.auto_retrieve = self._pressed_is_automatic()
        self._show_download_note()

    @on(Button.Pressed, "#setup-rag-copy-extras")
    def _on_copy_extras(self, event: Button.Pressed) -> None:
        event.stop()
        self.app.copy_to_clipboard(rag_state.RAG_EXTRAS_PACKAGE)
        # OSC 52 gives no delivery receipt (lessons-clipboard.md), so the
        # name stays on screen, selectable, whatever the terminal did.
        self.notify(
            f"Copied {rag_state.RAG_EXTRAS_PACKAGE} (if your terminal allows it).",
            markup=False,
        )

    def _pressed_is_automatic(self) -> bool:
        """F-A pattern: read the PRESSED radio, never only the Changed echo."""
        try:
            choice = self.query_one("#setup-rag-auto-retrieve", RadioSet)
        except Exception:
            return self.auto_retrieve
        pressed = choice.pressed_button
        if pressed is None or pressed not in choice.query(RadioButton):
            return self.auto_retrieve
        return pressed.id == "setup-rag-auto-retrieve-automatic"

    async def commit(self) -> tuple[bool, str]:
        chosen = self._pressed_is_automatic()
        mutation = rag_state.build_search_commit(
            auto_retrieve=chosen, current=self._current_auto_retrieve
        )
        if not mutation:
            return True, ""
        ok = await self.wizard.commit_config(mutation)
        if not ok:
            return False, "Saving the search setting failed."
        self.auto_retrieve = chosen
        self._current_auto_retrieve = chosen
        return True, ""

    def get_step_data(self) -> Dict[str, Any]:
        return {"auto_retrieve": self._pressed_is_automatic()}

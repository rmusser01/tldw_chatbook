"""The first-run wizard's RAG step.

Moved whole out of ``FirstRunSetupWizard.py`` (TASK-34100.1), so its ``@on``
handlers and workers stay registered on the class and fixes to this step have
room under the size ratchet. ``FirstRunSetupWizard`` still exports the class.
Patch this module, not the wizard, to replace what the step calls.
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
)

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import (
    RadioSet,
    Static,
)

from tldw_chatbook.Utils.input_validation import escape_markup
from tldw_chatbook.UI.Wizards.first_run_setup_widgets import (
    SetupRadioButton,
    SetupRadioSet,
    SetupStep,
    _radio_model_id,
)


class RagStep(SetupStep):
    """RAG/embeddings: report dep status; pick a default embedding model."""

    def __init__(self, wizard=None, config=None, *, deps_installed=None, **kwargs):
        super().__init__(wizard=wizard, config=config, **kwargs)
        if deps_installed is None:
            from tldw_chatbook.Utils.optional_deps import embeddings_rag_deps_installed

            deps_installed = embeddings_rag_deps_installed
        self._deps_installed = deps_installed
        self.selected_embedding_model: str = ""

    def compose_step(self) -> ComposeResult:
        with Vertical(classes="setup-rag"):
            yield Static("Search & RAG", classes="setup-title")
            yield Static("", id="setup-rag-status", classes="setup-subtitle")
            with SetupRadioSet(id="setup-rag-model-choice", classes="setup-choice-list"):
                for model_id in self._embedding_model_ids():
                    # The id comes from the user's `[embedding_config] models`
                    # table, so it must not be handed to a markup parser: a
                    # `[dim]` segment is silently deleted from the label (and
                    # so from what is read back into config), and a `[/]` one
                    # raises MarkupError out of compose(). The raw id rides
                    # `name`, as AppearanceStep rides `_theme_name`, so the
                    # committed value never depends on the rendering
                    # (tier-2 review S21 P3).
                    yield SetupRadioButton(escape_markup(model_id), name=model_id)

    def _embedding_model_ids(self) -> list[str]:
        app_config = getattr(self.wizard.app_instance, "app_config", {}) or {}
        embedding_config = app_config.get("embedding_config", {})
        models = (
            embedding_config.get("models", {})
            if isinstance(embedding_config, dict)
            else {}
        )
        return sorted(models) if isinstance(models, dict) else []

    def on_mount(self) -> None:
        status = self.query_one("#setup-rag-status", Static)
        if self._deps_installed():
            status.update(
                "Embedding dependencies are installed. Pick a default model, or skip."
            )
        else:
            status.update(
                # Static.update() treats [..] as Rich markup by default, so the
                # extras-package brackets must be escaped or "[embeddings_rag]"
                # silently vanishes from the rendered text instead of showing.
                # TASK-1502: quoted plainly — backticks are markdown idiom and
                # render literally in a TUI.
                "RAG lets the assistant search your own documents. Its "
                "optional dependencies aren't installed — install with: pip "
                'install "tldw_chatbook\\[embeddings_rag]" — then revisit '
                "Settings ▸ RAG. Skipping for now is fine."
            )
            try:
                # TASK-1502: hide the model list outright — a wall of disabled
                # options under a "not installed" message reads as breakage
                # and adds nothing the user can act on.
                self.query_one("#setup-rag-model-choice", RadioSet).display = False
            except Exception:
                pass

    @on(RadioSet.Changed, "#setup-rag-model-choice")
    def _on_model(self, event: RadioSet.Changed) -> None:
        self.selected_embedding_model = _radio_model_id(event.pressed)

    def _effective_embedding_model(self) -> str:
        """F-A fix: same pressed-radio fallback as ProviderStep/ModelStep."""
        if self.selected_embedding_model:
            return self.selected_embedding_model
        try:
            pressed = self.query_one("#setup-rag-model-choice", RadioSet).pressed_button
        except Exception:
            return ""
        return _radio_model_id(pressed) if pressed is not None else ""

    async def commit(self) -> tuple[bool, str]:
        from tldw_chatbook.UI.Wizards.first_run_setup_state import build_rag_commit

        model_id = self._effective_embedding_model()
        if not (self._deps_installed() and model_id):
            return True, ""
        ok = await self.wizard.commit_config(
            build_rag_commit(default_model_id=model_id)
        )
        if ok:
            self.selected_embedding_model = model_id
        return (True, "") if ok else (False, "Saving the embedding model failed.")

    def get_step_data(self) -> Dict[str, Any]:
        return {"embedding_model": self.selected_embedding_model}

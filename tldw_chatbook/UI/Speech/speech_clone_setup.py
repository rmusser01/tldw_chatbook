"""Focused, path-free presentation for one Guided audio.cpp clone setup."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.css.query import NoMatches
from textual.widgets import Button, Static, TextArea

from tldw_chatbook.TTS import AudioCppCloneSetupProjection
from tldw_chatbook.TTS.profile_reference_types import MAX_REFERENCE_TEXT_CHARACTERS
from .audio_cpp_runtime_card import AudioCppCloneDraftState


class SpeechCloneSetup(Vertical):
    """Render clone inputs while leaving reference ownership with the pane."""

    DEFAULT_CSS = """
    SpeechCloneSetup {
        height: auto;
    }
    """

    def __init__(
        self,
        projection: AudioCppCloneSetupProjection,
        *,
        draft_state: AudioCppCloneDraftState = "missing",
        transcript: str = "",
        id: str | None = None,
    ) -> None:
        if type(projection) is not AudioCppCloneSetupProjection:
            raise TypeError("clone setup projection must be exact")
        super().__init__(id=id)
        self.projection = projection
        self.draft_state = draft_state
        self.transcript = transcript

    def compose(self) -> ComposeResult:
        """Compose the bounded clone-reference controls and guidance."""

        projection = self.projection
        yield Static(
            f"Voice setup · {projection.family_label}",
            id="speech-clone-setup-title",
            classes="speech-section-head",
            markup=False,
        )
        yield Static(
            f"{projection.recipe_label} requires a matching spoken reference WAV.",
            id="speech-clone-recipe-guidance",
            classes="speech-clone-guidance",
            markup=False,
        )
        yield Static(
            "Reference audio and transcript remain local plaintext while in use. "
            "Filesystem permissions are not encryption; deletion is best effort.",
            id="speech-clone-privacy",
            classes="speech-clone-privacy",
            markup=False,
        )
        with Horizontal(classes="speech-clone-reference-row"):
            yield Static(
                "No reference WAV selected.",
                id="speech-clone-reference-status",
                classes="speech-source-status",
                markup=False,
            )
            yield Button(
                "Choose reference WAV",
                id="speech-clone-reference-choose",
                classes="workbench-action",
                compact=True,
            )
            yield Button(
                "Clear",
                id="speech-clone-reference-clear",
                classes="workbench-action",
                compact=True,
                disabled=True,
            )
        transcript = TextArea(
            self.transcript,
            id="speech-clone-reference-text",
            classes="speech-clone-transcript",
            placeholder="Type the exact words spoken in the reference WAV…",
        )
        transcript.show_line_numbers = False
        yield transcript
        yield Static(
            self._transcript_guidance(self.transcript),
            id="speech-clone-transcript-guidance",
            classes="speech-clone-guidance",
            markup=False,
        )
        yield Static(
            self._draft_copy(self.draft_state),
            id="speech-clone-error",
            classes="speech-clone-error",
            markup=False,
        )
        with Horizontal(classes="speech-clone-actions"):
            yield Button(
                "Use an existing Voice Profile",
                id="speech-clone-use-profile",
                classes="workbench-action",
                compact=True,
            )

    @staticmethod
    def _transcript_guidance(transcript: str) -> str:
        remaining = max(0, MAX_REFERENCE_TEXT_CHARACTERS - len(transcript))
        return (
            "Exact transcript required · "
            f"{remaining:,} of {MAX_REFERENCE_TEXT_CHARACTERS:,} characters remaining"
        )

    @staticmethod
    def _draft_copy(draft_state: AudioCppCloneDraftState) -> str:
        return {
            "missing": "Choose a WAV and provide its exact transcript.",
            "processing": "Validating the reference WAV…",
            "invalid": "Correct the highlighted reference field and try again.",
            "ready": "Reference ready for this model.",
        }[draft_state]

    def apply_draft_state(
        self,
        draft_state: AudioCppCloneDraftState,
        *,
        source_selected: bool,
        error_copy: str | None = None,
    ) -> None:
        """Update path-free validation presentation without recomposition."""

        self.draft_state = draft_state
        try:
            status = self.query_one("#speech-clone-reference-status", Static)
            status.update(
                "Reference WAV selected — local plaintext."
                if source_selected
                else "No reference WAV selected."
            )
            clear = self.query_one("#speech-clone-reference-clear", Button)
            clear.disabled = not source_selected
            self.query_one("#speech-clone-error", Static).update(
                error_copy or self._draft_copy(draft_state)
            )
        except NoMatches:
            return

    def apply_projection(self, projection: AudioCppCloneSetupProjection) -> None:
        """Replace only path-free recipe guidance in the mounted component."""

        if type(projection) is not AudioCppCloneSetupProjection:
            raise TypeError("clone setup projection must be exact")
        self.projection = projection
        try:
            self.query_one("#speech-clone-setup-title", Static).update(
                f"Voice setup · {projection.family_label}"
            )
            self.query_one("#speech-clone-recipe-guidance", Static).update(
                f"{projection.recipe_label} requires a matching spoken reference WAV."
            )
        except NoMatches:
            return

    def update_transcript_guidance(self, transcript: str) -> None:
        """Refresh the bounded character guidance for the visible editor."""

        self.transcript = transcript
        try:
            self.query_one("#speech-clone-transcript-guidance", Static).update(
                self._transcript_guidance(transcript)
            )
        except NoMatches:
            return


__all__ = ["SpeechCloneSetup"]

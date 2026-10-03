"""The first-run wizard's Speech step.

Moved whole out of ``FirstRunSetupWizard.py`` (TASK-34100.1), so its ``@on``
handlers and workers stay registered on the class and fixes to this step have
room under the size ratchet. ``FirstRunSetupWizard`` still exports the class.
Patch this module, not the wizard, to replace what the step calls.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Mapping,
    Optional,
)

from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.widget import Widget
from textual.widgets import (
    Button,
    Label,
    RadioButton,
    RadioSet,
    Static,
)
from textual.worker import (
    Worker,
    get_current_worker,
)

from tldw_chatbook.Local_Ingestion.parakeet_v2_artifact import (
    PARAKEET_PRECISIONS,
    active_managed_parakeet_dir,
    parakeet_descriptor,
    parakeet_v2_managed_service,
    parakeet_reference,
    parakeet_vad_descriptor,
    parakeet_vad_reference,
    run_parakeet_preflight,
    run_parakeet_provision,
    run_parakeet_vad_preflight,
    run_parakeet_vad_provision,
)
from tldw_chatbook.STT.parakeet_external import (
    ExternalParakeetVerificationError,
    format_external_parakeet_recovery,
)
from tldw_chatbook.STT.parakeet_sources import (
    ParakeetSourceError,
    ParakeetSourceErrorCode,
    ParakeetSourceKey,
    PreparedExternalSelection,
)
from tldw_chatbook.STT.transcribe_cpp_config import (
    configure_model_path as configure_transcribe_cpp_model_path,
    is_gguf_file,
)
from tldw_chatbook.Third_Party.textual_fspicker import (
    FileOpen,
    Filters,
    SelectDirectory,
)
from tldw_chatbook.UI.Screens.model_browser_state import install_failure_message
from tldw_chatbook.UI.Screens.model_installed_view import lifecycle_failure_message
from tldw_chatbook.UI.Wizards import first_run_speech_step_state as speech_state
from tldw_chatbook.UI.Wizards.BaseWizard import WizardStepConfig
from tldw_chatbook.Widgets.ModelArtifacts import (
    ActivationRequested,
    DeletionRequested,
    InstallProgressed,
    ModelActivationControls,
    ModelInstallModal,
    ModelInstallProgress,
    make_progress_callback,
)
from tldw_chatbook.Widgets.delete_confirmation_dialog import DeleteConfirmationDialog
from tldw_chatbook.UI.Wizards.first_run_setup_widgets import (
    SetupRadioButton,
    SetupRadioSet,
    SetupStep,
)
from tldw_chatbook.UI.Wizards.first_run_step_guard import wizard_work

if TYPE_CHECKING:
    from tldw_chatbook.UI.Wizards.FirstRunSetupWizard import SetupWizardContainer


class SpeechSetupStep(SetupStep):
    """Optional speech setup for exact managed Parakeet ONNX artifacts.

    TASK-1301: reuses the TASK-596 shared model-artifact controls
    (ModelInstallModal, ModelInstallProgress, ModelActivationControls) and
    the TASK-595 ModelArtifactService via the SAME
    Local_Ingestion.parakeet_v2_artifact convenience wrappers LibraryScreen's
    own Parakeet install surface already uses -- no duplicate artifact or
    network logic (AC#4). Language/precision options are enumerated from the
    canonical STT policy/catalog (first_run_speech_step_state, backed by
    tldw_chatbook.STT.routing) and gated to the exact managed Parakeet model
    and precision combinations (AC#2).

    Runtime gate (review Important 4): the `onnx-asr` extra is optional --
    missing it means a downloaded Parakeet artifact could never actually run.
    Gated exactly like RagStep gates on ``embeddings_rag_deps_installed()``:
    when the extra is absent, the install action stays visible for orientation
    but disabled so no unusable download can start.

    Persistence gate (AC#5 / review Important 3): commit() re-verifies --
    off the event loop -- that the exact selected artifact is active,
    AND requires that the user actually engaged this step THIS run
    (installed or activated it here) before writing anything to
    [transcription]. An artifact that merely happens to be active from an
    earlier session (e.g. installed via the Library screen) is not enough
    on its own -- a re-run that just presses Next through this step leaves
    whatever is already persisted completely untouched, however different
    (``remote-whisper``, ``default_language="auto"``, ...). The step also
    shows what is currently persisted before the user acts (the AC#5
    "prefill" clause) via ``first_run_speech_step_state.speech_prefill_status``.

    Skip and failures never trap the user (AC#6): Next/commit never blocks
    on install state, and a failed download still refreshes the step's own
    installed-state read so it never gets stuck showing a stale
    "installing…" affordance. A broken/not-ready installed item still shows
    ModelActivationControls(ready=False) so Delete (recovery) stays
    reachable (review Important 5).
    """

    def __init__(
        self,
        wizard: Optional["SetupWizardContainer"] = None,
        config: Optional[WizardStepConfig] = None,
        *,
        service_factory: Optional[Callable[[], Any]] = None,
        runtime_installed: Optional[Callable[[], bool]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(wizard=wizard, config=config, **kwargs)
        self._service_factory = service_factory or parakeet_v2_managed_service
        if runtime_installed is None:
            from tldw_chatbook.Utils.optional_deps import parakeet_onnx_deps_installed

            runtime_installed = parakeet_onnx_deps_installed
        self._runtime_installed = runtime_installed
        recommended = speech_state.recommended_speech_selection()
        self._selected_language = recommended.language
        self._selected_precision = recommended.precision
        self._service: Any = None
        self._loading = False
        self._loaded = False
        self._reload_after_load = False
        self._load_error: Optional[str] = None
        self._installed_item: Any = None
        self._operation: Optional[str] = None
        self._pending_report: Any = None
        self._progress: Any = None
        self._external_selection_generation = 0
        self._external_selection_token: tuple[int, int] | None = None
        self._external_scope_ids: dict[tuple[int, int], str] = {}
        self._external_selection_worker: Worker | None = None
        self._external_busy = False
        self._external_status = ""
        self._pending_external_selection: PreparedExternalSelection | None = None
        self._external_commit_handoff: asyncio.Task[bool] | None = None
        self._external_commit_detached = False
        self._external_commit_pending = False
        # Review Important 3: set only by a SUCCESSFUL install/activation
        # made THROUGH THIS STEP during this run -- see commit()'s use via
        # first_run_speech_step_state.should_persist_speech_config.
        self._acted_this_run = False
        app_config = getattr(self.wizard.app_instance, "app_config", {}) or {}
        transcription = (
            app_config.get("transcription", {})
            if isinstance(app_config, Mapping)
            else {}
        )
        prefill = speech_state.read_speech_prefill(app_config)
        if prefill.provider_id == speech_state.routing_policy().parakeet_provider_id:
            selected = speech_state.resolve_speech_selection(
                selected_language=prefill.language,
                selected_precision=prefill.precision or "int8",
                curated_selections=self._curated_selections(),
            )
            if selected is not None and selected.model_id == prefill.model_id:
                self._selected_language = selected.language
                self._selected_precision = selected.precision
        direct_config = (
            transcription.get("transcribe_cpp", {})
            if isinstance(transcription, Mapping)
            else {}
        )
        self._transcribe_cpp_configured = bool(
            direct_config.get("model_path")
            if isinstance(direct_config, Mapping)
            else False
        )

    def compose_step(self) -> ComposeResult:
        """Render the title/prefill/status/action block, then the language
        and precision catalogs.

        Entirely pure/I/O-free: it builds from ``self`` bookkeeping already
        in memory and the canonical Parakeet routing policy/precision values.
        Any I/O (checking installed state) is deferred to ``on_show``.

        Returns:
            The composed widgets: title, optional prefill line, status line
            plus its action control, an optional "use as default" affordance,
            then the language and precision ``RadioSet`` catalogs (see the
            Review Important 2 note below for the ordering rationale).
        """
        # Review Important 2: the primary action must be visible at the
        # wizard's own tested 120x40 budget -- title/subtitle/prefill/
        # status/action come FIRST (typically <=6 rows); the informational
        # language/precision catalog (up to 27 disabled rows, already
        # capped by ".setup-choice-list") comes after, reachable by
        # scrolling like any other step's overflow content.
        with Vertical(classes="setup-speech"):
            yield Static("Speech transcription (optional)", classes="setup-title")
            yield Static(
                f"Selected: {self._model_label()} — on-device "
                "speech-to-text for dictation. Optional; Next skips it.",
                classes="setup-subtitle",
            )
            prefill_text = self._prefill_status_text()
            if prefill_text:
                # NEW-1 (review): persisted values (and, below, the runtime-
                # missing message's bracketed extras names) may contain
                # literal "[...]" -- markup=False, same fix already applied
                # to "#setup-summary-rows" for the identical trap.
                yield Static(
                    prefill_text,
                    id="setup-speech-prefill",
                    classes="setup-subtitle",
                    markup=False,
                )
            status_text, action_widget = self._status_and_action()
            yield Static(
                status_text,
                id="setup-speech-status",
                classes="setup-subtitle",
                markup=False,
            )
            progress = ModelInstallProgress(
                self._progress, id="setup-speech-install-progress"
            )
            progress.display = (
                self._operation == "install" and self._progress is not None
            )
            yield progress
            yield Button(
                "Use model from disk…",
                id="setup-speech-use-from-disk",
                variant=(
                    "primary"
                    if action_widget is None
                    or getattr(action_widget, "disabled", False)
                    else "default"
                ),
                disabled=self._external_busy or self._lifecycle_pending,
            )
            external_status = Static(
                self._external_status,
                id="setup-speech-external-status",
                classes="setup-subtitle",
                markup=False,
            )
            external_status.display = bool(self._external_status)
            yield external_status
            if self._external_busy:
                yield Button(
                    "Cancel external setup",
                    id="setup-speech-cancel-external",
                    variant="default",
                )
            if action_widget is not None:
                yield action_widget
            if self._use_as_default_offer():
                # Review NEW-2: installed + active + configured elsewhere is
                # the one state where neither "install" nor "activate" is a
                # real action -- offer the affordance the prefill sentence
                # actually promises instead of leaving it undeliverable.
                yield Button(
                    f"Use {self._model_label()} as my default",
                    id="setup-speech-use-as-default",
                    variant="primary",
                )
            yield Static(
                "Existing local transcribe.cpp GGUF configured."
                if self._transcribe_cpp_configured
                else "No existing local transcribe.cpp GGUF configured.",
                id="setup-speech-transcribe-cpp-status",
                classes="setup-subtitle",
                markup=False,
            )
            yield Button(
                "Choose another GGUF…"
                if self._transcribe_cpp_configured
                else "Use an existing transcribe.cpp GGUF…",
                id="setup-speech-choose-transcribe-cpp-gguf",
                disabled=self._external_commit_pending,
            )
            yield Label("Language", classes="setup-field-label")
            with SetupRadioSet(
                id="setup-speech-language-choice", classes="setup-choice-list"
            ):
                for option in speech_state.speech_language_options(
                    curated_model_ids=self._curated_model_ids()
                ):
                    label = option.display_name + (
                        " (recommended)"
                        if option.code == "en"
                        else " — not yet available for managed install"
                        if not option.selectable
                        else ""
                    )
                    yield SetupRadioButton(
                        label,
                        id=f"setup-speech-language-{option.code}",
                        value=option.selectable
                        and option.code == self._selected_language,
                        disabled=not option.selectable or self._lifecycle_pending,
                    )
            yield Label("Precision", classes="setup-field-label")
            with SetupRadioSet(
                id="setup-speech-precision-choice", classes="setup-choice-list"
            ):
                for option in speech_state.speech_precision_options(
                    model_id=self._selection().model_id,
                    curated_selections=self._curated_selections(),
                ):
                    label = option.display_name + (
                        " (recommended)"
                        if option.value == "int8"
                        else " — not yet available for managed install"
                        if not option.selectable
                        else ""
                    )
                    # Minor 8: pre-press ONLY the one recommended option --
                    # "selectable" alone would pre-press every selectable
                    # precision the moment a second one is ever curated.
                    yield SetupRadioButton(
                        label,
                        id=f"setup-speech-precision-{option.value}",
                        value=(
                            option.selectable
                            and option.value == self._selected_precision
                        ),
                        disabled=not option.selectable or self._lifecycle_pending,
                    )

    # -- pure, I/O-free helpers ------------------------------------------
    @staticmethod
    def _curated_model_ids() -> frozenset[str]:
        policy = speech_state.routing_policy()
        return frozenset({policy.parakeet_v2_model_id, policy.parakeet_v3_model_id})

    @staticmethod
    def _curated_selections() -> frozenset[tuple[str, str]]:
        return frozenset(
            (model_id, precision)
            for model_id in SpeechSetupStep._curated_model_ids()
            for precision in PARAKEET_PRECISIONS
        )

    def _selection(self) -> speech_state.SpeechSelection:
        selection = speech_state.resolve_speech_selection(
            selected_language=self._selected_language,
            selected_precision=self._selected_precision,
            curated_selections=self._curated_selections(),
        )
        # The stored selection is initialized from, and only changed through,
        # selectable radios. This guard keeps a later registry change skip-safe.
        return selection or speech_state.recommended_speech_selection()

    @property
    def _reference(self) -> Any:
        selection = self._selection()
        return parakeet_reference(selection.model_id, selection.precision)

    def _model_label(self) -> str:
        selection = self._selection()
        descriptor = parakeet_descriptor(selection.model_id, selection.precision)
        policy = speech_state.routing_policy()
        version = "v2" if descriptor.model_id == policy.parakeet_v2_model_id else "v3"
        language = speech_state.LANGUAGE_DISPLAY_NAMES.get(
            selection.language, selection.language
        )
        return f"Parakeet {version} ({language}, {descriptor.precision.upper()})"

    # -- review finding 2: read the PRESSED radio, never a hardcoded default --
    _LANGUAGE_RADIO_ID_PREFIX = "setup-speech-language-"
    _PRECISION_RADIO_ID_PREFIX = "setup-speech-precision-"

    @on(RadioSet.Changed, "#setup-speech-language-choice")
    def _on_speech_language_changed(self, event: RadioSet.Changed) -> None:
        if self._lifecycle_pending or event.pressed is None:
            return
        button_id = event.pressed.id or ""
        language = button_id.removeprefix(self._LANGUAGE_RADIO_ID_PREFIX)
        self._set_exact_selection(language, self._selected_precision)

    @on(RadioSet.Changed, "#setup-speech-precision-choice")
    def _on_speech_precision_changed(self, event: RadioSet.Changed) -> None:
        if self._lifecycle_pending or event.pressed is None:
            return
        button_id = event.pressed.id or ""
        precision = button_id.removeprefix(self._PRECISION_RADIO_ID_PREFIX)
        self._set_exact_selection(self._selected_language, precision)

    def _set_exact_selection(self, language: str, precision: str) -> None:
        if self._external_commit_pending:
            return
        selection = speech_state.resolve_speech_selection(
            selected_language=language,
            selected_precision=precision,
            curated_selections=self._curated_selections(),
        )
        if selection is None:
            return
        if (
            selection.language == self._selected_language
            and selection.precision == self._selected_precision
        ):
            return
        self._discard_external_selection()
        self._selected_language = selection.language
        self._selected_precision = selection.precision
        self._installed_item = None
        self._loaded = False
        self._load_error = None
        self._pending_report = None
        self._ensure_loaded(force=True)

    def _effective_language(self) -> str:
        """The code of the currently pressed language radio, or "" for none.

        Mirrors ``ModelSetupStep._live_pressed_radio``'s guard: reading
        ``RadioSet.pressed_button`` unguarded can resurrect a stale press
        left over from before a ``recompose``, so membership in the set's
        CURRENT children is required too. "" (never pressed / step
        unmounted, e.g. ``commit()`` called before ``on_show()``) is a
        valid, skip-safe result -- ``resolve_speech_selection`` falls back
        to the recommended default for it.
        """
        return (
            self._pressed_radio_code(
                "#setup-speech-language-choice", self._LANGUAGE_RADIO_ID_PREFIX
            )
            or self._selected_language
        )

    def _effective_precision(self) -> str:
        """The value of the currently pressed precision radio, or "" for none."""
        return (
            self._pressed_radio_code(
                "#setup-speech-precision-choice", self._PRECISION_RADIO_ID_PREFIX
            )
            or self._selected_precision
        )

    def _pressed_radio_code(self, selector: str, id_prefix: str) -> str:
        try:
            radio_set = self.query_one(selector, RadioSet)
        except Exception:
            return ""
        pressed = radio_set.pressed_button
        if pressed is None or pressed not in radio_set.query(RadioButton):
            return ""
        button_id = pressed.id or ""
        if not button_id.startswith(id_prefix):
            return ""
        return button_id[len(id_prefix) :]

    def _prefill(self) -> Any:
        app_config = getattr(self.wizard.app_instance, "app_config", {}) or {}
        return speech_state.read_speech_prefill(app_config)

    def _installed_active(self) -> bool:
        item = self._installed_item
        return bool(item is not None and item.active)

    def _prefill_status_text(self) -> str:
        """AC#5's "re-run prefills" clause: show what is already persisted.

        Reads ``self.wizard.app_instance.app_config`` directly -- the same
        in-memory, already-loaded dict other steps read synchronously in
        compose_step() (e.g. RagStep._embedding_model_ids()) -- so this
        needs no worker. ``installed_active``/``acted_this_run`` make the
        copy state-aware (review NEW-2): the "installing or activating"
        promise is only shown when one of those is still a real action.
        """
        return speech_state.speech_prefill_status(
            self._prefill(),
            installed_active=self._installed_active(),
            acted_this_run=self._acted_this_run,
            runtime_installed=self._runtime_installed(),
            selected_label=self._model_label(),
        )

    def _use_as_default_offer(self) -> bool:
        """Review NEW-2: offer the real affordance the prefill sentence
        promises -- installed AND active (so neither Install nor Activate
        is available), a DIFFERENT provider is currently persisted, the
        runtime can actually run Parakeet, and the user has not already
        opted in this run (once acted, commit() already persists on Next).
        """
        if self._acted_this_run:
            return False
        if not self._runtime_installed():
            return False
        if not self._installed_active():
            return False
        prefill = self._prefill()
        return prefill.provider_id != speech_state.routing_policy().parakeet_provider_id

    @property
    def _lifecycle_pending(self) -> bool:
        """Review NEW-3: a forced reload in flight must ALSO disable the
        install/activation controls, not just an explicit operation --
        otherwise a just-deleted (or just-installed) artifact's stale
        ``_installed_item`` briefly re-renders with enabled controls before
        the reload's own callback replaces it (InstalledView's own pending
        computation includes its loading flag for the identical reason).
        """
        return (
            self._operation is not None
            or self._loading
            or self._external_busy
            or self._external_commit_pending
        )

    def _status_and_action(self) -> tuple[str, Optional[Widget]]:
        # Review Important 4: gate BEFORE the installed-state load so a
        # minimal install sees the real reason immediately. The action stays
        # visible for orientation but cannot start an unusable download.
        if not self._runtime_installed():
            return (
                'The "onnx-asr" runtime is not installed, so a downloaded '
                "model could not run. Install the extras package "
                '"tldw_chatbook[transcription_parakeet_onnx]" (or run '
                "pip install 'onnx-asr[cpu]==0.12.0'), then revisit this "
                "step. Skipping is safe — set this up later from Lab ▸ "
                "Models.",
                Button(
                    "Review and install…",
                    id="setup-speech-install",
                    variant="primary",
                    disabled=True,
                ),
            )
        if not self._loaded:
            if self._load_error:
                return self._load_error, Button("Retry", id="setup-speech-retry")
            return "Checking installed models…", None
        item = self._installed_item
        if item is None:
            return "Not installed.", Button(
                "Review and install…",
                id="setup-speech-install",
                variant="primary",
                disabled=self._lifecycle_pending,
            )
        if item.error is not None or not item.ready:
            # Review Important 5: reuse the SAME 596 control instead of a
            # dead end -- ready=False already keeps Delete enabled (the
            # only real recovery path) while disabling Activate.
            return (
                "This model needs attention — delete it below and install "
                "again, or manage it from Lab ▸ Models ▸ Installed.",
                ModelActivationControls(
                    self._reference,
                    active=item.active,
                    ready=item.ready,
                    pending=self._lifecycle_pending,
                ),
            )
        status = (
            "Installed and active." if item.active else "Installed, not yet active."
        )
        return status, ModelActivationControls(
            self._reference,
            active=item.active,
            ready=item.ready,
            pending=self._lifecycle_pending,
        )

    # -- user-owned external roots ---------------------------------------
    def _source_service(self) -> Any:
        """Return the app-owned source service shared with Lab and Library."""

        return self.wizard.app_instance._ensure_parakeet_source_service()

    def _source_key(self) -> ParakeetSourceKey:
        selection = self._selection()
        return ParakeetSourceKey.from_values(selection.model_id, selection.precision)

    def _next_external_token(self) -> tuple[int, int]:
        """Fence picker and worker callbacks to one exact mounted generation."""

        prior = self._external_selection_token
        if prior is not None:
            self._release_external_scope(prior)
        worker = self._external_selection_worker
        if worker is not None and not worker.is_finished:
            worker.cancel()
        self._external_selection_generation += 1
        token = (self._external_selection_generation, id(self))
        self._external_selection_token = token
        self._external_scope_ids[token] = f"setup-speech-{token[1]}-{token[0]}"
        self._external_selection_worker = None
        self._pending_external_selection = None
        return token

    def _owns_external_token(self, token: tuple[int, int]) -> bool:
        return (
            token == self._external_selection_token
            and token[1] == id(self)
            and self.is_attached
        )

    def _release_external_scope(self, token: tuple[int, int]) -> None:
        scope_id = self._external_scope_ids.pop(token, None)
        if scope_id is None:
            return
        service = getattr(
            self.wizard.app_instance,
            "_parakeet_source_service",
            None,
        )
        if service is not None:
            service.release_scope(scope_id)

    def _discard_external_selection(self) -> None:
        """Cancel pending external work without changing persisted source state."""

        token = self._external_selection_token
        handoff_active = (
            self._external_commit_handoff is not None
            and not self._external_commit_handoff.done()
        )
        worker = self._external_selection_worker
        if worker is not None and not worker.is_finished:
            worker.cancel()
        self._external_selection_generation += 1
        self._external_selection_token = None
        self._external_selection_worker = None
        self._external_busy = False
        self._external_status = ""
        self._pending_external_selection = None
        if handoff_active:
            self._external_commit_detached = True
        elif token is not None:
            self._release_external_scope(token)

    def _set_external_status(
        self,
        text: str,
        *,
        busy: bool | None = None,
    ) -> None:
        self._external_status = text
        if busy is not None:
            self._external_busy = busy
        self.refresh(recompose=True)

    @on(Button.Pressed, "#setup-speech-use-from-disk")
    def _use_external_pressed(self) -> None:
        if self._lifecycle_pending:
            return
        token = self._next_external_token()
        key = self._source_key()
        self.app.push_screen(
            SelectDirectory(
                str(Path.home()),
                title=f"Choose {key.model_id} {key.precision.upper()} directory",
            ),
            lambda selected: self._external_directory_selected(
                token,
                key,
                selected,
            ),
        )

    @on(Button.Pressed, "#setup-speech-cancel-external")
    def _cancel_external_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self._discard_external_selection()
        self._set_external_status(
            "External setup cancelled. The prior source is unchanged.",
            busy=False,
        )
        self.call_after_refresh(self._focus_external_disk_action)

    def _focus_external_disk_action(self) -> None:
        """Keep Enter on the external action after Cancel recomposes the step."""

        try:
            self.query_one("#setup-speech-use-from-disk", Button).focus()
        except NoMatches:
            pass

    def _external_directory_selected(
        self,
        token: tuple[int, int],
        key: ParakeetSourceKey,
        selected: Path | None,
    ) -> None:
        if not self._owns_external_token(token):
            self._release_external_scope(token)
            return
        if selected is None:
            self._discard_external_selection()
            return
        scope_id = self._external_scope_ids.get(token)
        if scope_id is None:
            return
        self._set_external_status("Verifying model files…", busy=True)
        self._external_selection_worker = self._verify_external_source(
            token,
            key,
            Path(selected),
            scope_id,
        )

    @wizard_work(
        thread=True,
        group="setup-speech-external-verify",
        exclusive=True,
        description="Verify external Parakeet source",
    )
    def _verify_external_source(
        self,
        token: tuple[int, int],
        key: ParakeetSourceKey,
        directory: Path,
        scope_id: str,
    ) -> None:
        """Hash one exact external root outside the Textual event loop."""

        worker = get_current_worker()

        def cancelled() -> bool:
            return worker.is_cancelled

        def progress(done: int, total: int) -> None:
            self.app.call_from_thread(
                self._apply_external_hash_progress,
                token,
                done,
                total,
            )

        try:
            prepared = self._source_service().prepare_external(
                key,
                directory,
                owner=("scope", scope_id),
                cancelled=cancelled,
                progress=progress,
            )
        except ExternalParakeetVerificationError as exc:
            message, is_error = format_external_parakeet_recovery(exc.code)
            if is_error:
                logger.warning(
                    "External Parakeet verification failed; error_type={}",
                    type(exc).__name__,
                )
            self.app.call_from_thread(
                self._apply_external_verification_result,
                token,
                None,
                message,
                is_error,
            )
            return
        except Exception as exc:
            logger.warning(
                "External Parakeet verification failed; error_type={}",
                type(exc).__name__,
            )
            self.app.call_from_thread(
                self._apply_external_verification_result,
                token,
                None,
                "The selected model could not be verified. Choose the directory again.",
                True,
            )
            return
        self.app.call_from_thread(
            self._apply_external_verification_result,
            token,
            prepared,
            None,
            False,
        )

    def _apply_external_hash_progress(
        self,
        token: tuple[int, int],
        done: int,
        total: int,
    ) -> None:
        if self._owns_external_token(token):
            self._set_external_status(
                f"Verifying model files · {done:,} / {total:,} bytes",
                busy=True,
            )

    def _apply_external_verification_result(
        self,
        token: tuple[int, int],
        prepared: PreparedExternalSelection | None,
        error: str | None,
        error_is_failure: bool = True,
    ) -> None:
        if not self._owns_external_token(token):
            self._release_external_scope(token)
            return
        self._external_selection_worker = None
        if error is not None or prepared is None:
            self._release_external_scope(token)
            message = error or "The selected model could not be verified."
            self._set_external_status(
                message,
                busy=False,
            )
            self.notify(
                message,
                severity="error" if error_is_failure else "information",
            )
            return
        self._set_external_status(
            "Checking the managed VAD dependency…",
            busy=True,
        )
        self._external_selection_worker = self._prepare_external_readiness(
            token,
            prepared,
        )

    @wizard_work(
        thread=True,
        group="setup-speech-external-ready",
        exclusive=True,
        description="Prepare external Parakeet configuration",
    )
    def _prepare_external_readiness(
        self,
        token: tuple[int, int],
        prepared: PreparedExternalSelection,
    ) -> None:
        """Recheck root/VAD and prepare a write-free config patch off-loop."""

        worker = get_current_worker()
        if worker.is_cancelled:
            return
        try:
            self._source_service().prepare_config_commit(prepared)
        except ParakeetSourceError as exc:
            outcome = (
                "vad"
                if exc.code is ParakeetSourceErrorCode.VAD_UNAVAILABLE
                else "error"
            )
        except Exception as exc:
            logger.warning(
                "External Parakeet readiness failed; error_type={}",
                type(exc).__name__,
            )
            outcome = "error"
        else:
            outcome = "ready"
        self.app.call_from_thread(
            self._apply_external_readiness_result,
            token,
            prepared,
            outcome,
        )

    def _apply_external_readiness_result(
        self,
        token: tuple[int, int],
        prepared: PreparedExternalSelection,
        outcome: str,
    ) -> None:
        if not self._owns_external_token(token):
            self._release_external_scope(token)
            return
        self._external_selection_worker = None
        if outcome == "vad":
            self._set_external_status(
                "Preparing the managed VAD dependency…",
                busy=True,
            )
            self._external_selection_worker = self._preflight_external_vad(
                token,
                prepared,
            )
            return
        if outcome != "ready":
            self._release_external_scope(token)
            message = (
                "The external source could not be prepared. "
                "The prior source is unchanged."
            )
            self._set_external_status(message, busy=False)
            self.notify(message, severity="error")
            return
        self._pending_external_selection = prepared
        message = (
            "External model verified. Continue to save."
            if self._runtime_installed()
            else "Runtime required"
        )
        self._set_external_status(message, busy=False)

    @wizard_work(
        thread=True,
        group="setup-speech-external-vad-preflight",
        exclusive=True,
        description="Check managed VAD dependency",
    )
    def _preflight_external_vad(
        self,
        token: tuple[int, int],
        prepared: PreparedExternalSelection,
    ) -> None:
        try:
            report = asyncio.run(run_parakeet_vad_preflight())
        except Exception as exc:
            logger.warning(
                "Managed VAD preflight failed; error_type={}",
                type(exc).__name__,
            )
            self.app.call_from_thread(
                self._apply_external_vad_preflight_result,
                token,
                prepared,
                None,
                "The managed VAD dependency could not be prepared.",
            )
            return
        self.app.call_from_thread(
            self._apply_external_vad_preflight_result,
            token,
            prepared,
            report,
            None,
        )

    def _apply_external_vad_preflight_result(
        self,
        token: tuple[int, int],
        prepared: PreparedExternalSelection,
        report: Any,
        error: str | None,
    ) -> None:
        if not self._owns_external_token(token):
            self._release_external_scope(token)
            return
        self._external_selection_worker = None
        vad_reference = parakeet_vad_reference()
        vad_source_url = parakeet_vad_descriptor().source_url
        if (
            error is not None
            or report is None
            or report.root != vad_reference
            or not report.entries
            or any(
                entry.ref != vad_reference or entry.source_url != vad_source_url
                for entry in report.entries
            )
        ):
            self._release_external_scope(token)
            message = error or "The managed VAD plan changed. Choose the model again."
            self._set_external_status(message, busy=False)
            self.notify(message, severity="error")
            return
        self.app.push_screen(
            ModelInstallModal(report, model_label="Silero VAD dependency"),
            lambda confirmed: self._confirm_external_vad(
                bool(confirmed),
                token,
                prepared,
                report,
            ),
        )

    def _confirm_external_vad(
        self,
        confirmed: bool,
        token: tuple[int, int],
        prepared: PreparedExternalSelection,
        report: Any,
    ) -> None:
        if not self._owns_external_token(token):
            return
        if not confirmed:
            self._discard_external_selection()
            self._set_external_status(
                "VAD install cancelled. The prior source is unchanged.",
                busy=False,
            )
            return
        self._set_external_status(
            "Installing the managed VAD dependency…",
            busy=True,
        )
        self._external_selection_worker = self._provision_external_vad(
            token,
            prepared,
            report,
        )

    @wizard_work(
        group="setup-speech-external-vad-install",
        exclusive=True,
        description="Install managed VAD dependency",
    )
    async def _provision_external_vad(
        self,
        token: tuple[int, int],
        prepared: PreparedExternalSelection,
        report: Any,
    ) -> None:
        def progress(event: Any) -> None:
            self._apply_external_vad_progress(
                token,
                event.bytes_done,
                event.bytes_total,
            )

        try:
            await run_parakeet_vad_provision(report, progress=progress)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Managed VAD installation failed; error_type={}",
                type(exc).__name__,
            )
            self._apply_external_vad_provision_result(
                token,
                prepared,
                "The managed VAD dependency could not be installed.",
            )
            return
        self._apply_external_vad_provision_result(
            token,
            prepared,
            None,
        )

    def _apply_external_vad_progress(
        self,
        token: tuple[int, int],
        done: int,
        total: int,
    ) -> None:
        if self._owns_external_token(token):
            self._set_external_status(
                f"Installing managed VAD dependency · {done:,} / {total:,} bytes",
                busy=True,
            )

    def _apply_external_vad_provision_result(
        self,
        token: tuple[int, int],
        prepared: PreparedExternalSelection,
        error: str | None,
    ) -> None:
        if not self._owns_external_token(token):
            self._release_external_scope(token)
            return
        self._external_selection_worker = None
        if error is not None:
            self._release_external_scope(token)
            self._set_external_status(error, busy=False)
            self.notify(error, severity="error")
            return
        self._set_external_status(
            "Rechecking model files and managed VAD…",
            busy=True,
        )
        self._external_selection_worker = self._prepare_external_readiness(
            token,
            prepared,
        )

    def on_unmount(self) -> None:
        self._discard_external_selection()

    # -- lazy installed-state load ----------------------------------------
    def on_show(self) -> None:
        """Trigger the lazy installed-state read for the selected artifact.

        ``compose_step`` renders synchronously from in-memory state only;
        this is where the step first asks the artifact service (via
        ``_ensure_loaded`` -> ``_load_installed_state``, an exclusive
        background worker) whether the selected managed artifact already
        exists, so the status line can move past "Checking installed
        models…". Idempotent: a step already re-shown (rerun navigation)
        does not re-trigger a redundant load (see ``_ensure_loaded``).

        Returns:
            None.
        """
        super().on_show()
        self._ensure_loaded()

    def _ensure_loaded(self, *, force: bool = False) -> None:
        # Minor 11: a forced reload requested while a load is already in
        # flight must not be silently dropped -- remember it and honor it
        # when the in-flight load's own callback runs (InstalledView's own
        # _reload_after_load pattern).
        if self._loading:
            if force:
                self._reload_after_load = True
            return
        if self._loaded and not force:
            return
        self._loading = True
        self._load_error = None
        self.refresh(recompose=True)
        self._load_installed_state()

    def _service_for_worker(self) -> Any:
        if self._service is None:
            self._service = self._service_factory()
        return self._service

    @wizard_work(thread=True, group="setup-speech-load", exclusive=True)
    def _load_installed_state(self) -> None:
        try:
            service = self._service_for_worker()
            item = next(
                (
                    candidate
                    for candidate in service.list_installed()
                    if candidate.descriptor is not None
                    and candidate.descriptor.reference == self._reference
                ),
                None,
            )
        except Exception:
            logger.opt(exception=True).error(
                "Speech setup step could not read installed models"
            )
            self.app.call_from_thread(
                self._apply_installed_state,
                None,
                "Could not check installed speech models.",
            )
            return
        self.app.call_from_thread(self._apply_installed_state, item, None)

    def _apply_installed_state(self, item: Any, error: Optional[str]) -> None:
        self._installed_item = item
        self._loading = False
        self._loaded = error is None
        self._load_error = error
        reload_after_load = self._reload_after_load
        self._reload_after_load = False
        if reload_after_load:
            self._ensure_loaded(force=True)
        else:
            self.refresh(recompose=True)

    # -- install: preflight -> consent modal -> provision ------------------
    @on(Button.Pressed, "#setup-speech-install")
    def _install_pressed(self) -> None:
        if self._lifecycle_pending:  # review NEW-3: also refuse during a reload
            return
        self._operation = "install"
        self.refresh(recompose=True)
        self._preflight_install()

    @on(Button.Pressed, "#setup-speech-retry")
    def _retry_pressed(self) -> None:
        self._ensure_loaded(force=True)

    @on(Button.Pressed, "#setup-speech-use-as-default")
    def _use_as_default_pressed(self) -> None:
        """Review NEW-2: make the affordance real. Sets the SAME
        ``_acted_this_run`` flag install/activate success sets -- nothing
        is written to disk here (matches every other step: only commit()
        on Next writes), but the pending choice is now genuine, and the
        prefill sentence updates to say so."""
        if self._lifecycle_pending or not self._use_as_default_offer():
            return
        self._acted_this_run = True
        self.notify(
            f"{self._model_label()} will become your default when you continue.",
            severity="information",
        )
        self.refresh(recompose=True)

    @on(Button.Pressed, "#setup-speech-choose-transcribe-cpp-gguf")
    def _choose_transcribe_cpp_gguf_pressed(self) -> None:
        """Open a GGUF-only picker for optional direct-local transcription."""

        if self._external_commit_pending:
            return

        async def picker_callback(selected_path: Path | None) -> None:
            if selected_path is not None and not self._external_commit_pending:
                self._configure_transcribe_cpp_gguf(selected_path)

        self.app.push_screen(
            FileOpen(
                location=Path.home(),
                title="Choose transcribe.cpp GGUF",
                filters=Filters(("GGUF models", is_gguf_file)),
            ),
            picker_callback,
        )

    @wizard_work(
        thread=True,
        group="setup-speech-transcribe-cpp-gguf",
        exclusive=True,
    )
    def _configure_transcribe_cpp_gguf(self, selected_path: Path) -> None:
        """Admit and persist a selected GGUF off the Textual event loop."""
        try:
            configure_transcribe_cpp_model_path(selected_path)
        except Exception:
            self.app.call_from_thread(
                self._apply_transcribe_cpp_gguf_result,
                False,
            )
            return
        self.app.call_from_thread(
            self._apply_transcribe_cpp_gguf_result,
            True,
        )

    def _apply_transcribe_cpp_gguf_result(self, configured: bool) -> None:
        """Apply a path-free direct-local GGUF configuration result."""
        if not configured:
            self.notify(
                "That GGUF cannot be used by transcribe.cpp. Choose another GGUF.",
                severity="warning",
            )
            return
        self._transcribe_cpp_configured = True
        self.notify(
            "Local GGUF configured for transcribe.cpp.",
            severity="information",
        )
        self.refresh(recompose=True)

    @wizard_work(
        thread=True, group="setup-speech-install", exclusive=True
    )
    def _preflight_install(self) -> None:
        import asyncio

        selection = self._selection()
        try:
            report = asyncio.run(  # policy-exception: worker-thread loop
                run_parakeet_preflight(selection.model_id, selection.precision)
            )
        except Exception as exc:
            logger.opt(exception=True).error("Speech transcription preflight failed")
            self.app.call_from_thread(
                self._apply_preflight_result,
                None,
                install_failure_message(exc, model_label=self._model_label()),
            )
            return
        self.app.call_from_thread(self._apply_preflight_result, report, None)

    def _apply_preflight_result(self, report: Any, error: Optional[str]) -> None:
        if error is not None or report is None:
            self._operation = None
            self.notify(error or "Speech model preflight failed.", severity="error")
            self.refresh(recompose=True)
            return
        self._pending_report = report
        self.app.push_screen(
            ModelInstallModal(
                report,
                model_label=self._model_label(),
                container_id="setup-speech-install-modal",
                confirm_id="setup-speech-install-confirm",
                cancel_id="setup-speech-install-cancel",
            ),
            self._confirm_install,
        )

    def _confirm_install(self, confirmed: bool) -> None:
        if self._external_commit_pending:
            return
        if not confirmed:
            self._pending_report = None
            self._operation = None
            self.refresh(recompose=True)
            return
        self._provision_install()

    @wizard_work(
        thread=True, group="setup-speech-install", exclusive=True
    )
    def _provision_install(self) -> None:
        import asyncio

        report = self._pending_report
        if report is None:
            self.app.call_from_thread(
                self._apply_provision_result,
                "No install plan is available; review the model again.",
            )
            return
        try:
            selection = self._selection()
            asyncio.run(  # policy-exception: worker-thread loop
                run_parakeet_provision(
                    selection.model_id,
                    selection.precision,
                    report,
                    progress=make_progress_callback(self.post_message),
                )
            )
        except Exception as exc:
            logger.opt(exception=True).error("Speech model installation failed")
            self.app.call_from_thread(
                self._apply_provision_result,
                install_failure_message(exc, model_label=self._model_label()),
            )
            return
        try:
            self._source_service().prefer_managed(self._source_key())
        except Exception as exc:
            logger.warning(
                "Speech model source preference failed after installation; "
                "error_type={}",
                type(exc).__name__,
            )
            self.app.call_from_thread(
                self._apply_provision_result,
                None,
                "Speech model installed and activated, but its source "
                "preference could not be saved. Activate it again to retry "
                "the preference.",
            )
            return
        self.app.call_from_thread(self._apply_provision_result, None)

    @on(InstallProgressed)
    def _install_progressed(self, event: InstallProgressed) -> None:
        event.stop()  # Minor 9: LibraryScreen's equivalent handler stops it too
        self._progress = event.progress
        try:
            progress = self.query_one(
                "#setup-speech-install-progress", ModelInstallProgress
            )
        except NoMatches:
            self.refresh(recompose=True)
            return
        progress.display = True
        progress.update_progress(event.progress)

    def _apply_provision_result(
        self,
        error: Optional[str],
        preference_error: Optional[str] = None,
    ) -> None:
        self._pending_report = None
        self._operation = None
        self._progress = None
        if error is not None:
            self.notify(error, severity="error")
        else:
            # Review Important 3: a SUCCESSFUL install made through this
            # step this run is the engagement commit()'s no-clobber gate
            # requires -- see should_persist_speech_config.
            self._acted_this_run = True
            self._discard_external_selection()
            self.notify(
                preference_error or "Speech model installed and activated.",
                severity="warning" if preference_error else "information",
            )
        # AC#6: failures never trap -- always refresh installed state so the
        # step reflects reality (and drops the disabled "installing…" affordance)
        # whether provisioning succeeded or failed.
        self._ensure_loaded(force=True)

    # -- activation / deletion: the 596 controls, reused verbatim ----------
    @on(ActivationRequested)
    def _activation_requested(self, event: ActivationRequested) -> None:
        event.stop()
        if self._lifecycle_pending:  # review NEW-3: also refuse during a reload
            return
        self._operation = "activate"
        self.refresh(recompose=True)
        self._activate_model()

    @wizard_work(
        thread=True, group="setup-speech-lifecycle", exclusive=True
    )
    def _activate_model(self) -> None:
        try:
            self._service_for_worker().activate(self._reference)
        except Exception as exc:
            logger.opt(exception=True).error("Speech model activation failed")
            self.app.call_from_thread(
                self._apply_lifecycle_result,
                lifecycle_failure_message(exc, operation="activation"),
            )
            return
        try:
            self._source_service().prefer_managed(self._source_key())
        except Exception as exc:
            logger.warning(
                "Speech model source preference failed after activation; error_type={}",
                type(exc).__name__,
            )
            self.app.call_from_thread(
                self._apply_lifecycle_result,
                None,
                "Speech model activated, but its source preference could not "
                "be saved. Activate it again to retry the preference.",
            )
            return
        self.app.call_from_thread(self._apply_lifecycle_result, None)

    @on(DeletionRequested)
    def _deletion_requested(self, event: DeletionRequested) -> None:
        event.stop()
        if self._lifecycle_pending:  # review NEW-3: also refuse during a reload
            return
        self.app.push_screen(
            DeleteConfirmationDialog(
                item_type="Model",
                item_name=self._model_label(),
                additional_warning=(
                    "The managed model files will be removed from this device."
                ),
                permanent=True,
            ),
            self._confirm_deletion,
        )

    def _confirm_deletion(self, confirmed: bool) -> None:
        if not confirmed or self._lifecycle_pending:
            return
        self._operation = "delete"
        self.refresh(recompose=True)
        self._delete_model()

    @wizard_work(
        thread=True, group="setup-speech-lifecycle", exclusive=True
    )
    def _delete_model(self) -> None:
        try:
            self._service_for_worker().delete(self._reference)
        except Exception as exc:
            logger.opt(exception=True).error("Speech model deletion failed")
            self.app.call_from_thread(
                self._apply_lifecycle_result,
                lifecycle_failure_message(exc, operation="deletion"),
            )
            return
        self.app.call_from_thread(self._apply_lifecycle_result, None)

    def _apply_lifecycle_result(
        self,
        error: Optional[str],
        preference_error: Optional[str] = None,
    ) -> None:
        # Capture BEFORE clearing: only a successful ACTIVATE counts as
        # engagement (review Important 3) -- deleting is not "opting in",
        # and the artifact will not be active afterwards anyway, so
        # commit()'s active-check already keeps that case skip-safe.
        operation = self._operation
        self._operation = None
        if error is not None:
            self.notify(error, severity="error")
        else:
            if operation == "activate":
                self._acted_this_run = True
                self._discard_external_selection()
            self.notify(
                preference_error or "Speech model updated.",
                severity="warning" if preference_error else "information",
            )
        self._ensure_loaded(force=True)

    # -- persistence gate (AC#5 / review Important 3 & 4) -------------------
    async def commit(self) -> tuple[bool, str]:
        """Persist ``[transcription]`` defaults, but only when it is safe to.

        A verified external selection is the exception to the managed-runtime
        gate: its source and speech defaults are written atomically even when
        the optional runtime is absent, so setup can finish before installation.

        For managed selections, writes nothing -- returns the
        ok-but-skip result ``(True, "")`` -- unless ALL of the following
        hold, each freshly re-verified rather than trusted from stale
        widget state:

        * the ``onnx-asr`` runtime extra is importable (Important 4) --
          persisting a provider the runtime cannot execute is worse than no
          config change at all;
        * the exact selected Parakeet artifact is verified ACTIVE right now
          (``_check_active``, run off the event loop in an executor, never
          the possibly-stale ``self._installed_item``);
        * the user engaged this step THIS wizard run -- installed,
          activated, or used "use as default" (``self._acted_this_run``) --
          see ``first_run_speech_step_state.should_persist_speech_config``.

        That last condition is Important 3's core no-clobber guarantee: an
        artifact that merely happens to be active from an earlier session
        (for example, installed via the Library screen) is not, on its
        own, reason to overwrite whatever is already configured in
        ``[transcription]`` (``remote-whisper``, ``default_language="auto"``,
        ...) just because the user pressed Next through a re-run without
        touching this step.

        When it does write, provider/model/language/precision come
        from ``first_run_speech_step_state.resolve_speech_selection`` --
        the PRESSED language/precision radios (read via
        ``_effective_language``/``_effective_precision``), never a
        hardcoded recommendation (review finding 2; see that function's
        docstring for the fallback rules).

        Returns:
            ``(True, "")`` when nothing needed writing, or the write
            succeeded; ``(False, <message>)`` when work is still pending or
            preparing, writing, or accepting an external selection failed.
        """
        if self._external_busy:
            return False, "Wait for external model verification to finish."
        if self._pending_external_selection is not None:
            return await self._commit_external_selection()

        # Important 4: never persist a provider the runtime cannot execute,
        # even if somehow both active and acted (belt-and-suspenders; the
        # UI-side gate in _status_and_action is the primary defense --
        # mirrors RagStep's own commit() re-check of deps_installed()).
        if not self._runtime_installed():
            return True, ""
        selection = speech_state.resolve_speech_selection(
            selected_language=self._effective_language(),
            selected_precision=self._effective_precision(),
            curated_selections=self._curated_selections(),
        )
        if selection is None:
            return True, ""
        active_dir = await asyncio.get_running_loop().run_in_executor(
            None, self._check_active, selection
        )
        if not speech_state.should_persist_speech_config(
            active=active_dir is not None, acted_this_run=self._acted_this_run
        ):
            # Skip-safe: nothing verified active, OR the user never engaged
            # this step this run -- either way, leave [transcription] byte-
            # identical to whatever is already persisted (review Important 3).
            return True, ""
        ok = await self.wizard.commit_config(
            speech_state.build_speech_transcription_commit(
                provider_id=selection.provider_id,
                model_id=selection.model_id,
                language=selection.language,
                precision=selection.precision,
            )
        )
        return (
            (True, "")
            if ok
            else (False, "Saving the speech transcription choice failed.")
        )

    async def _commit_external_selection(self) -> tuple[bool, str]:
        """Atomically write speech defaults plus one prepared external source."""

        prepared = self._pending_external_selection
        if prepared is None:
            return True, ""
        selection = speech_state.resolve_speech_selection(
            selected_language=self._effective_language(),
            selected_precision=self._effective_precision(),
            curated_selections=self._curated_selections(),
        )
        if selection is None:
            return False, "The selected speech model is no longer available."
        try:
            expected_key = ParakeetSourceKey.from_values(
                selection.model_id,
                selection.precision,
            )
        except ValueError:
            return False, "The selected speech model cannot use an external directory."
        if prepared.key is not expected_key:
            return (
                False,
                "The external model selection changed. Choose the directory again.",
            )

        loop = asyncio.get_running_loop()
        service = self._source_service()
        token = self._external_selection_token
        self._external_commit_detached = False
        self._external_commit_pending = True
        self.refresh(recompose=True)
        try:
            try:
                source_commit = await loop.run_in_executor(
                    None,
                    service.prepare_config_commit,
                    prepared,
                )
            except Exception as exc:
                logger.warning(
                    "External Parakeet config preparation failed; error_type={}",
                    type(exc).__name__,
                )
                return (
                    False,
                    "The external model or managed VAD changed. Choose the directory again.",
                )
            if token is not None and token != self._external_selection_token:
                self._release_external_scope(token)
                return False, "The external model selection changed. Choose it again."

            patch = speech_state.speech_config_patch(selection, source_commit)
            handoff = asyncio.create_task(
                self.wizard.commit_config(
                    patch,
                    after_write=lambda: service.accept_committed(source_commit),
                )
            )
            self._external_commit_handoff = handoff
            cancelled = False

            async def settle(operation: asyncio.Future[Any]) -> Any:
                nonlocal cancelled
                while True:
                    try:
                        return await asyncio.shield(operation)
                    except asyncio.CancelledError:
                        if operation.cancelled():
                            raise
                        cancelled = True
                        self._external_commit_detached = True
                        task = asyncio.current_task()
                        if task is not None:
                            task.uncancel()

            try:
                ok = await settle(handoff)
            except Exception as handoff_error:
                logger.warning(
                    "External Parakeet commit handoff failed; error_type={}",
                    type(handoff_error).__name__,
                )
                retry = loop.run_in_executor(
                    None,
                    service.accept_committed,
                    source_commit,
                )
                try:
                    await settle(retry)
                except Exception as retry_error:
                    logger.warning(
                        "External Parakeet commit reconciliation failed; error_type={}",
                        type(retry_error).__name__,
                    )
                    message = (
                        "The external source was saved, but it could not be activated "
                        "in this session. Restart the app, then retry setup."
                    )
                    self._external_status = message
                    return False, message
                ok = True

            if not ok:
                return False, "Saving the speech transcription choice failed."

            if token is not None:
                self._release_external_scope(token)
            if self._external_selection_token == token:
                self._external_selection_token = None
            self._pending_external_selection = None
            if cancelled:
                raise asyncio.CancelledError
            self._external_status = (
                "External source ready."
                if self._runtime_installed()
                else "Runtime required"
            )
            return True, ""
        finally:
            self._external_commit_handoff = None
            self._external_commit_pending = False
            if self.is_attached:
                self.refresh(recompose=True)

    def _check_active(self, selection: speech_state.SpeechSelection) -> Any:
        try:
            return active_managed_parakeet_dir(
                selection.model_id,
                selection.precision,
                service=self._service_for_worker(),
            )
        except Exception:
            logger.opt(exception=True).error("Speech setup active-check failed")
            return None

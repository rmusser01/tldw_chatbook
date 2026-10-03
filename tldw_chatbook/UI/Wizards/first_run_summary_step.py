"""The first-run wizard's Summary step.

Moved whole out of ``FirstRunSetupWizard.py`` (TASK-34100.1), so its ``@on``
handlers and workers stay registered on the class and fixes to this step have
room under the size ratchet. ``FirstRunSetupWizard`` still exports the class.
Patch this module, not the wizard, to replace what the step calls.
"""

from __future__ import annotations

import os
from typing import (
    Any,
    Dict,
    Optional,
)

from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.containers import (
    Horizontal,
    Vertical,
)
from textual.css.query import NoMatches
from textual.widget import Widget
from textual.widgets import (
    Button,
    Checkbox,
    Static,
)

from tldw_chatbook.Local_Ingestion.parakeet_v2_artifact import (
    active_managed_parakeet_dir,
)
from tldw_chatbook.Model_Artifacts.store import managed_model_artifact_root
from tldw_chatbook.UI.Wizards import first_run_speech_step_state as speech_state
from tldw_chatbook.UI.Wizards import first_run_setup_state as wizard_state
from tldw_chatbook.UI.Wizards.first_run_setup_widgets import (
    SetupCheckbox,
    SetupStep,
)
from tldw_chatbook.UI.Wizards.first_run_step_guard import run_wizard_worker
from tldw_chatbook.UI.Wizards.first_run_speech_step import SpeechSetupStep


#: task-32140: the Summary step's "Write your first note" exit. Not a real
#: tab id -- app.py's _continue_first_run_wizard_result rewrites it to
#: TAB_LIBRARY with the LIBRARY_NAV_CONTEXT_NOTES_CREATE context, the same
#: sentinel-then-rewrite shape TAB_LIBRARY itself uses for "Add your first
#: document" (task-32072), just one level further since two different
#: Library destinations both need to travel as one wizard exit_route.
EXIT_ROUTE_LIBRARY_NOTES = "library_notes"


class SummaryStep(SetupStep):
    """Read-back ✓/✗ matrix plus mode-dependent exits.

    Always re-reads the persisted config (never step memory) so the summary
    reflects what actually landed on disk, not what the in-memory steps
    think they committed.
    """

    def __init__(
        self,
        wizard=None,
        config=None,
        *,
        load_config=None,
        rag_deps_installed=None,
        speech_installed=None,
        speech_runtime_installed=None,
        **kwargs,
    ):
        super().__init__(wizard=wizard, config=config, **kwargs)
        self._load_config = load_config
        self._rag_deps_installed = rag_deps_installed
        # TASK-1301 AC#6: same injectable-callable shape as rag_deps_installed
        # -- defaults to a real, off-loop-safe check of the configured exact
        # managed Parakeet artifact's installed/active state.
        self._speech_installed = speech_installed
        # Review Important 4 residual: same shape again -- defaults to the
        # real onnx-asr runtime probe so Summary agrees with the Speech
        # step's own runtime gate instead of only checking files-on-disk.
        self._speech_runtime_installed = speech_runtime_installed
        self.exit_route: Optional[str] = None
        self.provider_model_complete = False
        self._render_worker = None

    def compose_step(self) -> ComposeResult:
        """Build the read-back matrix and the Summary's exit actions.

        Returns:
            The scrolling read-back body (title, per-track defaults note,
            summary rows, model-catalog consent checkbox, footer, and the
            post-setup interview checkbox) followed by the docked exit
            actions row (provider setup, add a document, write a note,
            explore Home, review settings).
        """
        with Vertical(classes="setup-summary"):
            yield Static("Setup summary", classes="setup-title")
            yield Static("", id="setup-summary-defaults-note", classes="setup-subtitle")
            # markup=False: row labels/details come from persisted config data
            # (embedding model ids, notes directories, ...) which may contain
            # literal "[...]" -- Static.update() otherwise parses that as Rich
            # markup and silently drops it from the rendered text.
            yield Static("", id="setup-summary-rows", markup=False)
            # TASK-21146 (UAT H-1): the online model-list consent belongs in
            # setup, not as a surprise modal the moment "Start chatting"
            # lands in Console. Default OFF (deny-by-default, same privacy
            # posture as the modal); shown only while no consent answer is
            # recorded (see _render_rows), so re-runs never re-ask. The
            # answer persists on completion (commit) via the exact
            # [model_catalog] contract _handle_model_catalog_consent writes.
            yield SetupCheckbox(
                "Keep model lists fresh — checks your configured providers "
                "online at startup",
                id="setup-summary-model-catalog-consent",
                classes="hidden",
            )
            yield Static(
                "", id="setup-summary-footer", classes="setup-subtitle", markup=False
            )
            yield Checkbox(
                "Get to know you after setup",
                False,
                id="setup-profile-interview-offer",
                compact=True,
            )
        # The exit actions are a DIRECT child of the step (the .setup-step
        # scroll container), not of the scrolling .setup-summary Vertical:
        # Textual docks position against the container's visible frame and
        # never scroll with content, which is what keeps the wizard's final
        # CTAs on screen no matter how tall the read-back matrix gets
        # (TASK-1495 AC #3 -- full-track content previously pushed them
        # below the fold at 120x40).
        # task-32140 review: five full-label buttons no longer fit in one
        # non-wrapping row at either supported wizard size (80x24 or
        # 120x40 -- Textual Horizontal never wraps). Two docked rows keep
        # every action on screen instead of running it off the right edge.
        with Vertical(classes="setup-summary-actions"):
            with Horizontal(classes="setup-summary-actions-row"):
                yield Button(
                    "Review provider setup", id="setup-exit-chat", variant="primary"
                )
                # task-32072: the Summary never said where content lives,
                # so a finished setup handed the user no way to put a
                # file anywhere.
                yield Button("Add your first document", id="setup-exit-library")
            with Horizontal(classes="setup-summary-actions-row"):
                # task-32140: a local-first user who came for notes was
                # told the only thing they could do needed an API key.
                yield Button(
                    "Write your first note", id="setup-exit-library-notes"
                )
                yield Button("Explore Home", id="setup-exit-home")
                yield Button("Review settings", id="setup-exit-settings")

    def on_show(self) -> None:
        super().on_show()
        track = (
            (self.wizard.wizard_data or {})
            .get(wizard_state.STEP_WELCOME, {})
            .get("track")
        )
        if track == wizard_state.TRACK_QUICK:
            self.query_one("#setup-summary-defaults-note", Static).update(
                "Left at recommended defaults: tools off, RAG off, default theme, "
                "notes sync off — each lives in Settings when you want it."
            )
        if self._render_worker is not None and self._render_worker.is_running:
            return
        self._render_worker = run_wizard_worker(
            self,
            self._render_rows(),
            exclusive=True,
            group="setup-summary-load",
        )

    async def _render_rows(self) -> None:
        import asyncio

        load = self._load_config
        if load is None:
            from tldw_chatbook.config import load_cli_config_and_ensure_existence

            def load():
                return load_cli_config_and_ensure_existence(force_reload=True)

        config = await asyncio.get_running_loop().run_in_executor(None, load)

        deps = self._rag_deps_installed
        if deps is None:
            from tldw_chatbook.Utils.optional_deps import embeddings_rag_deps_installed

            deps = embeddings_rag_deps_installed
        speech_installed_check = self._speech_installed
        if speech_installed_check is None:
            prefill = speech_state.read_speech_prefill(config)
            selection = speech_state.recommended_speech_selection()
            if (
                prefill.provider_id
                == speech_state.routing_policy().parakeet_provider_id
            ):
                resolved = speech_state.resolve_speech_selection(
                    selected_language=prefill.language,
                    selected_precision=prefill.precision or "int8",
                    curated_selections=SpeechSetupStep._curated_selections(),
                )
                if resolved is not None and resolved.model_id == prefill.model_id:
                    selection = resolved

            def speech_installed_check() -> bool:
                # Minor 12: a Quick-track user (who never saw the Speech
                # step) reaching Summary must not cause the managed
                # artifact store's directories to be created on disk --
                # constructing a real ModelArtifactService (what
                # active_managed_parakeet_dir() does internally) mkdirs
                # unconditionally. A read-only existence check first means
                # "nothing was ever installed by anyone" costs zero
                # filesystem writes; only go on to the real check once
                # something has legitimately created the root already.
                if not managed_model_artifact_root().exists():
                    return False
                return (
                    active_managed_parakeet_dir(
                        selection.model_id,
                        selection.precision,
                    )
                    is not None
                )

        speech_runtime_check = self._speech_runtime_installed
        if speech_runtime_check is None:
            from tldw_chatbook.Utils.optional_deps import parakeet_onnx_deps_installed

            speech_runtime_check = parakeet_onnx_deps_installed

        speech_installed = await asyncio.get_running_loop().run_in_executor(
            None, speech_installed_check
        )
        speech_runtime_installed = await asyncio.get_running_loop().run_in_executor(
            None, speech_runtime_check
        )
        # TASK-32892: the wizard can be dismissed or advanced while the
        # await above is in flight, and a post-await `query_one` raising
        # out of a worker whose `exit_on_error` defaults to True exits the
        # whole app mid-setup. Recheck, and the launch sites pass
        # exit_on_error=False for everything this recheck cannot see.
        # Qodo review of PR #2799: `is_attached`, not `is_mounted` -- see
        # `ProtectKeysStep._apply_password_worker` for why the latter is inert.
        if not self.is_attached:
            return
        from tldw_chatbook.UI.Wizards.first_run_setup_state import build_summary_rows

        # TASK-21146 (UAT H-1): offer the model-list consent only while no
        # answer is recorded — a rerun after any answer never re-asks.
        try:
            from tldw_chatbook.LLM_Provider_Catalog.model_catalog_settings import (
                load_model_catalog_settings,
            )

            consent_recorded = load_model_catalog_settings(
                config
            ).refresh_consent_recorded
        except Exception:
            consent_recorded = True  # fail closed: never re-ask on a bad read
        try:
            consent_box = self.query_one(
                "#setup-summary-model-catalog-consent", Checkbox
            )
            consent_box.set_class(consent_recorded, "hidden")
            self._model_catalog_consent_offered = not consent_recorded
        except NoMatches:
            self._model_catalog_consent_offered = False

        rows = build_summary_rows(
            config,
            dict(os.environ),
            rag_deps_installed=deps(),
            speech_installed=speech_installed,
            speech_runtime_installed=speech_runtime_installed,
        )
        row_states = {row.label: row.state for row in rows}
        from tldw_chatbook.UI.Wizards.first_run_setup_state import (
            ROW_CONFIGURED,
            apply_probe_failure_to_summary_rows,
            build_first_run_summary_actions,
        )

        # TASK-21143 (UAT S-1): build_summary_rows reads the config file,
        # where a saved-but-rejected key is indistinguishable from a working
        # one — the exact incident where the summary said "✓ Provider"
        # minutes after the probe got a 401. Overlay what the wizard's own
        # probe learned, and let it flip the primary action to
        # review_provider (the affordance that already existed for the
        # never-saved case).
        probe_failure = ""
        try:
            probe_failure = self.wizard.provider_probe_failure()
        except Exception:
            logger.debug("Summary probe-failure lookup skipped", exc_info=True)
        rows = apply_probe_failure_to_summary_rows(rows, probe_failure)

        primary, _, _ = build_first_run_summary_actions(
            provider_configured=row_states.get("Provider") == ROW_CONFIGURED,
            model_configured=row_states.get("Default model") == ROW_CONFIGURED,
            provider_probe_failed=bool(probe_failure),
        )
        self.provider_model_complete = primary == "start_chatting"
        primary_button = self.query_one("#setup-exit-chat", Button)
        primary_button.label = (
            "Start chatting"
            if self.provider_model_complete
            else "Review provider setup"
        )
        primary_button.tooltip = (
            "Open Console with this provider and model."
            if self.provider_model_complete
            else "Return to Provider and finish the connection and model setup."
        )
        # Static.update() parses "[...]" as Rich markup by default, so any
        # bracketed literal in a label/detail (e.g. a package extra name)
        # must be escaped or it silently vanishes from the rendered text.
        lines = [
            f"{row.glyph} {row.label}" + (f" — {row.detail}" if row.detail else "")
            for row in rows
        ]
        # TASK-1266: steps dropped by the compose-crash policy get a reasoned
        # row — the matrix must reflect that an area was never presented, not
        # silently omit it.
        failed_titles = []
        try:
            failed_titles = self.wizard.compose_failed_steps()
        except Exception:
            logger.debug("compose_failed_steps unavailable", exc_info=True)
        lines.extend(
            f"✗ {title} — step couldn't be shown (skipped); configure in Settings"
            for title in failed_titles
        )
        self.query_one("#setup-summary-rows", Static).update("\n".join(lines))
        from tldw_chatbook.config import get_cli_config_path

        # F-D fix: resolving the path and updating the widget were one bare
        # try/except Exception: pass -- ANY failure in either half (a
        # get_cli_config_path() error, or the query_one below) left the
        # footer exactly as compose() first rendered it (""), so the label
        # itself never even appeared, and any real failure vanished with no
        # trace. Resolve the path in its own guarded step with a visible
        # fallback string, so the footer's "Config file:" line always shows
        # SOMETHING and a genuine resolution failure is at least logged
        # instead of silently producing an empty-looking row.
        try:
            config_path_text = str(get_cli_config_path())
        except Exception:
            logger.warning(
                "Summary footer could not resolve the config path", exc_info=True
            )
            config_path_text = "(unknown — see Settings ▸ Diagnostics)"
        try:
            self.query_one("#setup-summary-footer", Static).update(
                "Config file: "
                f"{wizard_state.middle_truncate_path(config_path_text, max(40, (self.size.width or 120) - 18))}\n"
                "Re-run setup any time: Settings ▸ Diagnostics ▸ Run setup wizard."
            )
        except Exception:
            logger.debug("Summary footer widget unavailable to update", exc_info=True)

    @on(Button.Pressed, "#setup-exit-chat")
    def _exit_chat(self) -> None:
        if not self.provider_model_complete:
            self.wizard.review_provider_setup()
            return
        from tldw_chatbook.Constants import TAB_CHAT

        self._finish(TAB_CHAT)

    @on(Button.Pressed, "#setup-exit-library")
    def _exit_library(self) -> None:
        """Finish setup on Library's Import canvas (task-32072)."""
        from tldw_chatbook.Constants import TAB_LIBRARY

        self._finish(TAB_LIBRARY)

    @on(Button.Pressed, "#setup-exit-library-notes")
    def _exit_library_notes(self) -> None:
        """Finish setup on Library's New note view (task-32140)."""
        self._finish(EXIT_ROUTE_LIBRARY_NOTES)

    @on(Button.Pressed, "#setup-exit-home")
    def _exit_home(self) -> None:
        from tldw_chatbook.Constants import TAB_HOME

        self._finish(TAB_HOME)

    @on(Button.Pressed, "#setup-exit-settings")
    def _exit_settings(self) -> None:
        self.wizard.open_provider_settings()

    def _finish(self, exit_route: Optional[str]) -> None:
        self.exit_route = exit_route
        # Deviation from the task brief: the brief calls
        # self.wizard.handle_next() directly, but SetupWizardContainer's
        # handle_next is the @on(Button.Pressed, "#wizard-next") override
        # documented above -- it takes the Button.Pressed event and calls
        # event.prevent_default() on it (required so the base class's own
        # handle_next() doesn't ALSO fire per Textual's whole-MRO @on
        # dispatch; see that method's docstring/comment). Calling
        # handle_next() with no event, or with None, would raise on
        # event.prevent_default(). advance_programmatically() is the
        # extracted body (guard + worker dispatch) with no event
        # dependency, used by both the real button handler and this
        # programmatic exit path, so the dispatch semantics for the actual
        # Next button are unchanged.
        self.wizard.advance_programmatically()

    def preferred_focus(self) -> Optional[Widget]:
        """Land on the primary exit so Enter finishes setup (TASK-21146).

        Without this, whichever widget the async row-render reveals first
        (the consent checkbox) would race the step-change focus fix.
        """
        try:
            return self.query_one("#setup-exit-chat", Button)
        except NoMatches:
            return None

    async def commit(self) -> tuple[bool, str]:
        """Persist the model-list consent answer, when it was offered.

        TASK-21146 (UAT H-1): mirrors _handle_model_catalog_consent's exact
        [model_catalog] contract — the answer is recorded either way, and
        an unchecked box (the default) also disables auto refresh, so the
        Console modal never fires after a completed wizard while the
        skip-the-wizard path keeps the existing consent flow untouched.
        """
        if not getattr(self, "_model_catalog_consent_offered", False):
            return True, ""
        try:
            allowed = self.query_one(
                "#setup-summary-model-catalog-consent", Checkbox
            ).value
        except NoMatches:
            return True, ""
        section: Dict[str, Any] = {"refresh_consent_recorded": True}
        if not allowed:
            section["auto_refresh_enabled"] = False
        ok = await self.wizard.commit_config({"model_catalog": section})
        if not ok:
            return False, "Saving the model-list preference failed."
        if allowed:
            # TASK-21150 item (a): match the Console modal's allow path,
            # which refreshes immediately. Recording consent alone left the
            # first session on stale lists with no modal left to trigger
            # the fetch. Failure here is non-fatal: the answer is already
            # saved and the next launch refreshes on schedule.
            try:
                self.wizard.request_model_catalog_refresh()
            except Exception:
                logger.debug(
                    "Model catalog refresh request skipped", exc_info=True
                )
        return True, ""

    def get_step_data(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"exit_route": self.exit_route}
        if self.query_one("#setup-profile-interview-offer", Checkbox).value:
            result["offer_profile_interview"] = True
        return result

"""The first-run wizard's Model step.

Moved whole out of ``FirstRunSetupWizard.py`` (TASK-34100.1), so its ``@on``
handlers and workers stay registered on the class and fixes to this step have
room under the size ratchet. ``FirstRunSetupWizard`` still exports the class.
Patch this module, not the wizard, to replace what the step calls.
"""

from __future__ import annotations

from functools import partial
from typing import (
    Any,
    Dict,
    Literal,
    Mapping,
    Optional,
)

from loguru import logger
from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.widgets import (
    Button,
    Input,
    Label,
    RadioButton,
    RadioSet,
    Static,
)

from tldw_chatbook.UI.Wizards import first_run_model_discovery as model_discovery
from tldw_chatbook.UI.Wizards import first_run_setup_state as wizard_state
from tldw_chatbook.UI.Wizards.first_run_model_discovery import (
    _first_run_discovery_staged_settings,
    _handed_off_failure_category,
    _legacy_model_ids,
    _model_discovery_ui_outcome,
)
from tldw_chatbook.UI.Wizards.first_run_provider_step import ProviderStep
from tldw_chatbook.UI.Wizards.first_run_setup_widgets import (
    SetupRadioButton,
    SetupRadioSet,
    SetupStep,
)


# How many discovered models the radio picker renders. The full set stays on
# the step as _discovered_model_ids; the adjacent "Or enter a model name"
# input is the escape hatch for anything past this bound.
_PICKER_MODEL_LIMIT = 20


class ModelStep(SetupStep):
    """Pick a default model for the chosen provider.

    Model discovery tries the injectable scope service first (an 8s guard
    keeps a hanging/slow provider from blocking Next), then falls back to
    the curated ``[providers]`` table from config.toml. Whichever provider
    key form ProviderStep handed us (raw key or display name; see
    ``ProviderStep._display_value_for``), the curated lookup bridges both
    forms via ``first_run_setup_state.curated_models_for_provider`` so a
    case/format mismatch never silently empties the list.
    """

    def __init__(
        self,
        wizard=None,
        config=None,
        *,
        discover_models=None,
        provider_draft: wizard_state.FirstRunProviderDraft | None = None,
        **kwargs,
    ):
        super().__init__(wizard=wizard, config=config, **kwargs)
        if provider_draft is not None and (
            type(provider_draft) is not wizard_state.FirstRunProviderDraft
        ):
            raise TypeError("Model discovery requires FirstRunProviderDraft.")
        self._discover_models = discover_models
        self._explicit_provider_draft = provider_draft
        # The complete id set the last handoff produced. The picker renders
        # a bounded slice of it (see _PICKER_MODEL_LIMIT), so without this
        # the step keeps no record of what it actually received and a trim
        # introduced in the handoff is invisible from the outside.
        self._discovered_model_ids: tuple[str, ...] = ()
        self._shown_for_provider: Optional[str] = None
        self._shown_for_discovery_key: wizard_state.FirstRunModelDiscoveryKey | None = (
            None
        )
        self._selection_discovery_key: wizard_state.FirstRunModelDiscoveryKey | None = (
            None
        )
        self._rendered_discovery_key: wizard_state.FirstRunModelDiscoveryKey | None = (
            None
        )
        self._selection_config_precondition: object | None = None
        self._manual_decision_active = False
        self.selected_model_id: str = ""
        # Bug-5: tracks whether selected_model_id's current value came from
        # the free-text custom Input (as opposed to the RadioSet) -- lets
        # clearing that Input fall back to any active radio selection
        # instead of leaving a stale custom value in place.
        self._model_id_from_custom_input: bool = False
        self._model_load_generation = 0
        # TASK-21143 (UAT S-1/M-2): the classified outcome of the discovery
        # probe rendered for _rendered_discovery_key ("", "authentication",
        # "connection"). Read via current_probe_failure(), which returns ""
        # whenever the rendered key no longer matches the live identity —
        # the same staleness discipline the rest of this step uses.
        self._rendered_probe_failure: str = ""

    def current_probe_failure(self) -> str:
        """The failed-probe classification for the CURRENT provider identity.

        Returns:
            "" when the probe succeeded, never ran, or belongs to a
            superseded identity; otherwise "authentication" or
            "connection" (wizard_state.PROVIDER_PROBE_*).
        """
        if self._rendered_discovery_key is None:
            return ""
        try:
            current_key = self._current_discovery_key()
        except Exception:
            return ""
        if current_key != self._rendered_discovery_key:
            return ""
        return self._rendered_probe_failure

    def confirm_before_advance(self) -> Optional[str]:
        """UAT M-2: a known-failed probe must not be Next-ed past silently."""

        failure = self.current_probe_failure()
        if failure == wizard_state.PROVIDER_PROBE_AUTH:
            return (
                "The API key failed an authentication check, so this model "
                "setup is unverified. Continue anyway?"
            )
        if failure == wizard_state.PROVIDER_PROBE_CONNECTION:
            return (
                "The server couldn't be reached, so this model setup is "
                "unverified. Continue anyway?"
            )
        return None

    def invalidate_credential_bound_selection(self) -> None:
        """Drop model state derived under a credential that has rotated."""

        self.invalidate_discovery_bound_selection()

    def invalidate_discovery_bound_selection(self) -> None:
        """Drop a selection whose exact provider discovery identity changed."""

        self._model_load_generation += 1
        self._shown_for_discovery_key = None
        self._selection_discovery_key = None
        self._rendered_discovery_key = None
        self._rendered_probe_failure = ""
        self._selection_config_precondition = None
        self._manual_decision_active = False
        self.selected_model_id = ""
        self._model_id_from_custom_input = False
        if not self.is_mounted:
            return
        try:
            custom = self.query_one("#setup-model-custom", Input)
            with custom.prevent(Input.Changed):
                custom.value = ""
            self._clear_model_radio_selection()
        except Exception:
            return
        if self.is_active:
            self.on_show()

    def compose_step(self) -> ComposeResult:
        with Vertical(classes="setup-model"):
            yield Static("Pick a default model", classes="setup-title")
            yield Static("", id="setup-model-provider-line", classes="setup-subtitle")
            with SetupRadioSet(id="setup-model-choice", classes="setup-choice-list"):
                # disabled=True: an un-disabled placeholder is a real,
                # toggleable RadioButton -- pressing Enter/Space while it is
                # the only/highlighted option (e.g. an impatient user, or
                # discovery that never resolves) would fire RadioSet.Changed
                # and commit the literal placeholder text as the model id
                # (see _on_model_chosen). Same reasoning applies to the two
                # other placeholders this step ever mounts, below.
                yield SetupRadioButton(
                    "(loading models…)", id="setup-model-loading", disabled=True
                )
            yield Label("Or enter a model name", classes="setup-field-label")
            yield Input(id="setup-model-custom", placeholder="model-id")
            yield Button(
                "Retry", id="setup-model-retry", variant="default", classes="hidden"
            )

    def _current_provider(self) -> tuple[str, str]:
        provider_draft = self._current_provider_draft()
        if provider_draft is not None:
            return provider_draft.provider, provider_draft.provider
        data = (self.wizard.wizard_data or {}).get(wizard_state.STEP_PROVIDER, {})
        provider_key = str(data.get("provider_key", ""))
        provider_value = str(data.get("provider_value", ""))
        if provider_key:
            return provider_key, provider_value
        return "", ""

    def _current_discovery_key(
        self,
    ) -> wizard_state.FirstRunModelDiscoveryKey | None:
        provider_draft = self._current_provider_draft()
        if provider_draft is None:
            return None
        try:
            return wizard_state.build_first_run_model_discovery_key(provider_draft)
        except ValueError:
            return None

    def _current_provider_draft(
        self,
    ) -> wizard_state.FirstRunProviderDraft | None:
        provider_draft = getattr(self.wizard, "staged_provider_draft", None)
        if type(provider_draft) is wizard_state.FirstRunProviderDraft:
            return provider_draft
        return self._explicit_provider_draft

    def on_show(self) -> None:
        super().on_show()
        self._model_load_generation += 1
        load_generation = self._model_load_generation
        provider_key, provider_value = self._current_provider()
        discovery_key = self._current_discovery_key()
        exact_key_changed = (
            discovery_key is not None and discovery_key != self._shown_for_discovery_key
        )
        provider_changed = provider_key != self._shown_for_provider
        if exact_key_changed or (discovery_key is None and provider_changed):
            # UI half of dependency invalidation: the config half (clearing
            # chat_defaults.model) already happened in ProviderStep.commit()
            # via invalidate_model_for_provider_change. This just keeps the
            # step's own in-memory selection from surviving a Back-and-switch.
            #
            # TASK-1374: re-run prefill from a genuinely reachable condition.
            # The old guard keyed on wizard_data lacking a provider entry --
            # unreachable, since _advance() always records one before Model
            # can be shown. The real re-run signal is the session provider
            # MATCHING the persisted chat_defaults.provider: same provider ->
            # surface the saved model; changed provider -> blank (the config
            # half of that invalidation already happened in ProviderStep).
            first_identity = (
                self._shown_for_provider is None
                and self._shown_for_discovery_key is None
            )
            prefill_model_id = (
                wizard_state.rerun_model_prefill(
                    getattr(self.wizard.app_instance, "app_config", {}) or {},
                    provider_value=provider_value,
                )
                if first_identity
                else ""
            )
            self.selected_model_id = prefill_model_id
            self._model_id_from_custom_input = False
            self._shown_for_provider = provider_key
            self._shown_for_discovery_key = discovery_key
            self._selection_discovery_key = discovery_key if prefill_model_id else None
            self._selection_config_precondition = (
                self._config_precondition_for_discovery(discovery_key)
                if prefill_model_id
                else None
            )
            self._manual_decision_active = False
            try:
                self.query_one("#setup-model-custom", Input).value = prefill_model_id
            except Exception:
                pass
        try:
            # TASK-1503: display-case the provider in user copy — raw keys
            # like "anthropic"/"llama_cpp" are internals, not UI language.
            from tldw_chatbook.Chat.provider_catalog import provider_display_name

            display = provider_display_name(provider_key) if provider_key else ""
            self.query_one("#setup-model-provider-line", Static).update(
                f"Models for {display or 'your provider'}."
            )
        except Exception:
            pass
        live_radios = tuple(self.query("#setup-model-choice RadioButton"))
        rendered_is_current = (
            discovery_key is not None
            and discovery_key == self._rendered_discovery_key
            and any(getattr(button, "_model_id", "") for button in live_radios)
        )
        if rendered_is_current:
            self._restore_model_radio_selection(discovery_key)
        if provider_key and not rendered_is_current:
            self.run_worker(
                partial(
                    self._load_models,
                    provider_key,
                    provider_value,
                    discovery_key,
                    load_generation,
                ),
                exclusive=True,
                group="setup-model-load",
            )
        elif not provider_key:
            # F-F fix: with no provider chosen yet there is nothing to
            # discover against, so the old code simply skipped this branch
            # and left the initial "(loading models…)" RadioButton in place
            # forever -- a permanently-stuck loading indicator for a state
            # that was never actually loading. Replace it with copy that
            # tells the user what to do instead.
            self.run_worker(
                partial(
                    self._render_models,
                    [],
                    no_provider=True,
                    discovery_key=discovery_key,
                    load_generation=load_generation,
                ),
                exclusive=True,
                group="setup-model-load",
            )

    def _cancel_model_discovery(self) -> None:
        self._model_load_generation += 1
        if self.is_attached:
            self.workers.cancel_group(self, "setup-model-load")

    def on_hide(self) -> None:
        super().on_hide()
        self._cancel_model_discovery()
        owner = getattr(self.wizard, "_first_run_provider_discovery_owner", None)
        if isinstance(owner, ProviderStep):
            owner.cancel_selected_discovery_handoff()
        self._explicit_provider_draft = None

    def on_unmount(self) -> None:
        self._cancel_model_discovery()
        owner = getattr(self.wizard, "_first_run_provider_discovery_owner", None)
        if isinstance(owner, ProviderStep):
            owner.cancel_selected_discovery_handoff()
        self._explicit_provider_draft = None
        self._selection_config_precondition = None
        self._manual_decision_active = False

    async def _load_models(
        self,
        provider_key: str,
        provider_value: str,
        discovery_key: wizard_state.FirstRunModelDiscoveryKey | None = None,
        load_generation: int | None = None,
    ) -> None:
        import asyncio

        models: list[str] = []
        discovery_state = "available"
        failure_category = ""
        await self._render_models(
            [],
            discovery_state="loading",
            discovery_key=discovery_key,
            load_generation=load_generation,
        )
        provider_draft = self._current_provider_draft()
        if provider_draft is None or discovery_key is None:
            await self._render_models(
                [],
                discovery_key=discovery_key,
                load_generation=load_generation,
            )
            return
        discover = self._discover_models
        owner = getattr(self.wizard, "_first_run_provider_discovery_owner", None)
        handed_off = getattr(self.wizard, "_first_run_selected_provider_models", {})
        handed_outcomes = getattr(
            self.wizard, "_first_run_selected_provider_outcomes", {}
        )
        if isinstance(handed_outcomes, Mapping) and discovery_key in handed_outcomes:
            models, discovery_state, failure_category = _model_discovery_ui_outcome(
                handed_outcomes[discovery_key]
            )
            discover = None
        elif isinstance(handed_off, Mapping) and discovery_key in handed_off:
            models = list(handed_off[discovery_key])
            if (
                not models
                and isinstance(owner, ProviderStep)
                and owner._selected_discovery_key == discovery_key
                and owner._selected_discovery_state == "failed"
            ):
                discovery_state = "connection_failed"
                failure_category = _handed_off_failure_category(owner, discovery_key)
            discover = None
        elif (
            isinstance(owner, ProviderStep)
            and owner.is_mounted
            and owner.app is self.app
        ):
            try:
                selected_outcome = await asyncio.wait_for(
                    owner._outcome_from_selected_discovery(provider_key, discovery_key),
                    timeout=model_discovery.MODEL_DISCOVERY_TIMEOUT_SECONDS,
                )
            except TimeoutError:
                owner.cancel_selected_discovery_handoff()
                selected_outcome = None
                discovery_state = "connection_failed"
                failure_category = "timeout"
            except Exception:
                selected_outcome = None
            if selected_outcome is not None:
                models, discovery_state, failure_category = _model_discovery_ui_outcome(
                    selected_outcome
                )
            elif (
                owner._selected_discovery_key == discovery_key
                and owner._selected_discovery_state == "failed"
            ):
                discovery_state = "connection_failed"
                failure_category = _handed_off_failure_category(owner, discovery_key)
            # ProviderStep owns setup network work for this selection. If the
            # user advances before it finishes, use curated fallback rather
            # than issuing the same provider catalog request from ModelStep.
            discover = None
        elif isinstance(owner, ProviderStep):
            discover = None
        if isinstance(owner, ProviderStep):
            evidence = owner._test_evidence_for_discovery_key(discovery_key)
            if evidence is not None:
                if evidence.endpoint == "model_listing_unavailable":
                    discovery_state = "listing_unavailable"
                    models = []
                elif evidence.endpoint == "unreachable":
                    discovery_state = "connection_failed"
                    failure_category = (
                        evidence.category or "connection_error"
                    ).replace("_", " ")
                    models = []
                elif evidence.endpoint == "reachable" and not models:
                    models = list(evidence.model_ids)
        if discover is None:
            service = (
                None
                if isinstance(owner, ProviderStep)
                else getattr(
                    self.wizard.app_instance,
                    "llm_provider_catalog_scope_service",
                    None,
                )
            )
            if service is not None:

                async def discover(*, provider=provider_key, svc=service, **_identity):
                    return await svc.discover_models(
                        mode="local",
                        provider=provider,
                        staged_settings=_first_run_discovery_staged_settings(
                            provider_draft, discovery_key
                        ),
                        use_shared_cache=False,
                    )

        if discover is not None:
            try:
                result = await asyncio.wait_for(
                    discover(
                        provider=provider_key,
                        endpoint=discovery_key.connection_identity[1],
                        credential_source=discovery_key.credential_source,
                        credential_revision=discovery_key.credential_revision,
                    ),
                    timeout=model_discovery.MODEL_DISCOVERY_TIMEOUT_SECONDS,
                )
                try:
                    models, discovery_state, failure_category = (
                        _model_discovery_ui_outcome(result)
                    )
                except ValueError:
                    models = list(_legacy_model_ids(result))
            except TimeoutError:
                discovery_state = "connection_failed"
                failure_category = "timeout"
            except Exception:
                discovery_state = "connection_failed"
                failure_category = "connection error"
                logger.debug("Wizard model discovery failed", exc_info=True)
        if not models and discovery_state == "available":
            from tldw_chatbook.config import get_cli_providers_and_models

            models = wizard_state.curated_models_for_provider(
                get_cli_providers_and_models(), provider_value
            )
        self._discovered_model_ids = tuple(models)
        await self._render_models(
            models[:_PICKER_MODEL_LIMIT],
            discovery_state=discovery_state,
            failure_category=failure_category,
            discovery_key=discovery_key,
            load_generation=load_generation,
        )

    async def _render_models(
        self,
        models: list[str],
        *,
        no_provider: bool = False,
        discovery_state: str = "available",
        failure_category: str = "",
        discovery_key: wizard_state.FirstRunModelDiscoveryKey | None = None,
        load_generation: int | None = None,
    ) -> None:
        if (
            load_generation is not None
            and load_generation != self._model_load_generation
        ):
            return
        if discovery_key != self._current_discovery_key():
            return
        try:
            radio_set = self.query_one("#setup-model-choice", RadioSet)
        except Exception:
            return
        # Textual does not clear these owner pointers when children are
        # removed. Reset them before rebuilding so a restored selection can
        # only reference one of the newly mounted rows.
        radio_set._pressed_button = None
        radio_set._selected = None
        # remove_children()/mount() are message-queue operations -- both
        # return awaitables that must be awaited before the DOM change is
        # actually applied. Without awaiting the removal, a second call (e.g.
        # a provider switch that fires before the first discovery settles)
        # can try to mount fresh "setup-model-option-N" ids while the stale
        # ones are still present, raising DuplicateIds.
        await radio_set.remove_children()
        if (
            load_generation is not None
            and load_generation != self._model_load_generation
        ):
            return
        if discovery_key != self._current_discovery_key():
            return
        if models:
            # TASK-1503: the first entry (curated-default / top discovery hit)
            # carries a "recommended" tag in its LABEL only; the clean model
            # id lives on the button as `_model_id` so selection and commits
            # never round-trip display decoration into config.
            def _button(index: int, model_id: str) -> SetupRadioButton:
                label = f"{model_id}   (recommended)" if index == 0 else model_id
                button = SetupRadioButton(label, id=f"setup-model-option-{index}")
                button._model_id = model_id
                button._discovery_key = discovery_key
                return button

            await radio_set.mount_all(
                _button(index, model_id) for index, model_id in enumerate(models)
            )
            selected = (
                self.selected_model_id
                if not self._model_id_from_custom_input
                and self._selection_discovery_key == discovery_key
                else ""
            )
            if selected:
                self._restore_model_radio_selection(discovery_key)
        elif discovery_state == "listing_unavailable":
            await radio_set.mount(
                SetupRadioButton(
                    "Model listing unavailable; enter the model ID used by this endpoint.",
                    id="setup-model-listing-unavailable",
                    disabled=True,
                )
            )
        elif discovery_state == "connection_failed":
            category = failure_category or "connection error"
            # TASK-21143 (UAT M-1/M-4): auth failures point at the fix (the
            # key lives one step Back — Retry cannot succeed there);
            # connection failures name the server the user has to start.
            probe_failure = wizard_state.classify_discovery_failure(
                discovery_state, category
            )
            if probe_failure == wizard_state.PROVIDER_PROBE_AUTH:
                failed_text = (
                    "Authentication failed — this API key was rejected. Go "
                    "Back to fix it, or enter a model ID below."
                )
            else:
                provider_key = getattr(discovery_key, "provider_key", "")
                endpoint = ""
                identity = getattr(discovery_key, "connection_identity", ())
                if len(identity) > 1 and identity[1]:
                    endpoint = str(identity[1])
                at_endpoint = f" at {endpoint}" if endpoint else ""
                if provider_key in ("ollama", "local_ollama"):
                    failed_text = (
                        f"Ollama isn't running{at_endpoint}. Start it "
                        "(ollama serve), then Retry — or enter a model ID "
                        "below."
                    )
                elif provider_key in ("llama_cpp", "local_llamacpp"):
                    failed_text = (
                        f"The llama.cpp server isn't reachable{at_endpoint}. "
                        "Start it, then Retry — or enter a model ID below."
                    )
                else:
                    failed_text = (
                        f"Couldn't reach the server ({category}). Check it's "
                        "running, then Retry — or enter a model ID below."
                    )
            await radio_set.mount(
                SetupRadioButton(
                    failed_text,
                    id="setup-model-connection-failed",
                    disabled=True,
                )
            )
        elif discovery_state == "loading":
            await radio_set.mount(
                SetupRadioButton(
                    "(loading models…)",
                    id="setup-model-loading",
                    disabled=True,
                )
            )
        elif no_provider:
            await radio_set.mount(
                SetupRadioButton(
                    "Pick a provider first — or type a model name below",
                    id="setup-model-no-provider",
                    disabled=True,
                )
            )
        else:
            await radio_set.mount(
                SetupRadioButton(
                    "(no models found — enter one below)",
                    id="setup-model-empty",
                    disabled=True,
                )
            )
        if not self.is_attached:
            return
        # TASK-21143: record the classified outcome for the trust chain
        # (tracker "!", Model-step confirm gate, Summary override) before
        # the retry-button lookup's early return can skip it.
        self._rendered_probe_failure = wizard_state.classify_discovery_failure(
            discovery_state, failure_category or "connection error"
        )
        try:
            retry = self.query_one("#setup-model-retry", Button)
            # Retry stays for connection failures (start the server, retry);
            # it is hidden for auth failures — retrying cannot fix a
            # rejected key, the fix lives one step Back (UAT M-1).
            retry.set_class(
                discovery_state != "connection_failed"
                or self._rendered_probe_failure
                == wizard_state.PROVIDER_PROBE_AUTH,
                "hidden",
            )
        except NoMatches:
            return
        self._rendered_discovery_key = discovery_key

    @on(Button.Pressed, "#setup-model-retry")
    def _retry_model_discovery(self, event: Button.Pressed) -> None:
        event.stop()
        if not self.is_active or self._current_discovery_key() is None:
            return
        event.button.add_class("hidden")
        self._rendered_discovery_key = None
        self._manual_decision_active = False
        if self._model_id_from_custom_input:
            self._selection_config_precondition = None
        owner = getattr(self.wizard, "_first_run_provider_discovery_owner", None)
        provider_draft = self._current_provider_draft()
        if isinstance(owner, ProviderStep) and provider_draft is not None:
            owner._begin_selected_provider_discovery(
                provider_draft,
                sync_live_credential=False,
            )
        self.on_show()

    @on(RadioSet.Changed, "#setup-model-choice")
    def _on_model_chosen(self, event: RadioSet.Changed) -> None:
        if event.pressed is not None:
            self.set_selected_model_from_button(event.pressed)

    def set_selected_model_from_button(self, button: RadioButton) -> None:
        """Select via a radio row, reading the clean id, not the label.

        TASK-1503: labels may carry display decoration ("(recommended)");
        the undecorated model id is stored on the button as ``_model_id``.

        Args:
            button: The pressed radio row. Only its clean ``_model_id``
                attribute can supply a model id; status labels are ignored.
        """
        model_id = getattr(button, "_model_id", None)
        if not isinstance(model_id, str) or not model_id:
            return
        try:
            custom_input = self.query_one("#setup-model-custom", Input)
            with custom_input.prevent(Input.Changed):
                custom_input.value = ""
        except Exception:
            pass
        self.set_selected_model(
            model_id,
            discovery_key=getattr(button, "_discovery_key", None),
        )

    def _clear_model_radio_selection(self) -> None:
        """Clear Textual's radio value and owner pointer without event races."""

        try:
            radio_set = self.query_one("#setup-model-choice", RadioSet)
        except Exception:
            return
        pressed = radio_set.pressed_button
        if pressed is None:
            return
        with (
            radio_set.prevent(RadioButton.Changed),
            pressed.prevent(RadioButton.Changed),
        ):
            radio_set._pressed_button = None
            pressed.value = False

    def _restore_model_radio_selection(
        self,
        discovery_key: wizard_state.FirstRunModelDiscoveryKey | None,
    ) -> None:
        """Point RadioSet state only at the live row for this selection."""

        if (
            self._model_id_from_custom_input
            or self._selection_discovery_key != discovery_key
        ):
            return
        try:
            radio_set = self.query_one("#setup-model-choice", RadioSet)
        except Exception:
            return
        buttons = list(radio_set.query(RadioButton))
        selected = next(
            (
                button
                for button in buttons
                if getattr(button, "_model_id", "") == self.selected_model_id
            ),
            None,
        )
        with radio_set.prevent(RadioButton.Changed):
            for button in buttons:
                button.value = button is selected
        radio_set._pressed_button = selected
        radio_set._selected = buttons.index(selected) if selected is not None else None

    @on(Input.Changed, "#setup-model-custom")
    def _on_custom_model(self, event: Input.Changed) -> None:
        """Bug-5 fix: clearing the custom Input must clear the selection too.

        The old handler only ever ASSIGNED on a non-empty value, so
        clearing a previously-typed custom model left ``selected_model_id``
        stuck at the last typed value -- a "skip-safe" commit would then
        silently persist a model the input no longer shows. On empty, fall
        back to whatever radio button is currently pressed (or "" if none),
        rather than just blanking unconditionally.
        """
        previous_model = self.selected_model_id
        value = event.value.strip()
        if value:
            current_key = self._current_discovery_key()
            self._clear_model_radio_selection()
            self.selected_model_id = value
            self._model_id_from_custom_input = True
            if (
                not self._manual_decision_active
                or self._selection_discovery_key != current_key
            ):
                self._manual_decision_active = True
                self._selection_config_precondition = (
                    self._capture_current_config_precondition()
                )
            self._selection_discovery_key = current_key
        elif self._model_id_from_custom_input:
            self._model_id_from_custom_input = False
            self._manual_decision_active = False
            pressed = self._live_pressed_radio()
            # TASK-1503: clean id, never the (possibly decorated) label.
            self.selected_model_id = (
                str(getattr(pressed, "_model_id", pressed.label))
                if pressed is not None
                else ""
            )
            self._selection_discovery_key = (
                self._current_discovery_key() if self.selected_model_id else None
            )
            self._selection_config_precondition = (
                self._config_precondition_for_discovery(self._selection_discovery_key)
                if self.selected_model_id
                else None
            )
        if self.selected_model_id != previous_model:
            self._notify_provider_model_changed()

    def set_selected_model(
        self,
        model_id: str,
        *,
        discovery_key: wizard_state.FirstRunModelDiscoveryKey | None = None,
    ) -> None:
        changed = model_id != self.selected_model_id
        self.selected_model_id = model_id
        self._model_id_from_custom_input = False
        self._manual_decision_active = False
        current_key = self._current_discovery_key()
        self._selection_discovery_key = (
            discovery_key
            if model_id and discovery_key == current_key
            else current_key
            if model_id
            else None
        )
        self._selection_config_precondition = (
            self._config_precondition_for_discovery(self._selection_discovery_key)
            if model_id
            else None
        )
        if changed:
            self._notify_provider_model_changed(
                model_id=model_id,
                discovery_key=discovery_key,
            )

    def _notify_provider_model_changed(
        self,
        *,
        model_id: str = "",
        discovery_key: wizard_state.FirstRunModelDiscoveryKey | None = None,
    ) -> None:
        invalidate_save = getattr(
            self.wizard, "invalidate_provider_write_expectation", None
        )
        if callable(invalidate_save):
            invalidate_save()
        owner = getattr(self.wizard, "_first_run_provider_discovery_owner", None)
        if isinstance(owner, ProviderStep):
            owner._model_semantics_changed(
                model_id=model_id,
                discovery_key=discovery_key,
            )

    def _config_precondition_for_discovery(
        self,
        discovery_key: wizard_state.FirstRunModelDiscoveryKey | None,
    ) -> object | None:
        preconditions = getattr(
            self.wizard, "_first_run_provider_config_preconditions", {}
        )
        if discovery_key is None or not isinstance(preconditions, Mapping):
            return None
        return preconditions.get(discovery_key)

    def _capture_current_config_precondition(self) -> object | None:
        discovery_key = self._current_discovery_key()
        capture = getattr(self.wizard, "capture_provider_config_precondition", None)
        if discovery_key is None or not callable(capture):
            return None
        return capture(discovery_key)

    def _live_pressed_radio(self) -> Optional[RadioButton]:
        """F1 fix: read ``#setup-model-choice``'s ``pressed_button``, but only
        if it is still one of the RadioSet's *current* children.

        Textual's ``RadioSet._pressed_button`` (``textual/widgets/_radio_set.py``)
        is a plain instance attribute; ``remove_children()`` prunes DOM
        children but never touches it. ``_render_models`` calls
        ``remove_children()``/``mount_all()`` on every provider switch to
        swap in the new provider's models, so a RadioButton pressed under
        the OLD provider stays referenced by ``_pressed_button`` -- now
        pointing at a detached, no-longer-mounted widget -- until the user
        presses something in the NEW list. Reading ``pressed_button``
        unguarded after a provider switch (Back -> switch provider -> Next)
        therefore resurrects the previous provider's model id even though
        nothing in the currently-visible list was ever pressed. Guarding
        with membership in ``radio_set.query(RadioButton)`` (the set's
        live, currently-mounted children) closes that window without
        reaching into ``_pressed_button`` from application code.
        """
        try:
            radio_set = self.query_one("#setup-model-choice", RadioSet)
        except Exception:
            return None
        pressed = radio_set.pressed_button
        if pressed is None or pressed not in radio_set.query(RadioButton):
            return None
        discovery_key = self._current_discovery_key()
        if discovery_key is not None and (
            getattr(pressed, "_discovery_key", None) != discovery_key
        ):
            return None
        return pressed

    def _effective_model_id(self) -> str:
        """F-A fix: fall back to the RadioSet's own ``pressed_button`` when
        this step's own bookkeeping (``selected_model_id``, updated only by
        ``_on_model_chosen``/``_on_custom_model``) has nothing -- same
        reasoning as ``ProviderStep._effective_provider_key``. The three
        placeholder rows this step ever mounts (loading / no-provider /
        no-models-found) are all ``disabled=True`` and so can never actually
        become ``pressed_button``.

        F1 fix: the fallback goes through ``_live_pressed_radio()`` rather
        than reading ``pressed_button`` directly, so a stale press left over
        from a provider switch (see ``_live_pressed_radio``'s docstring)
        cannot resurrect the previous provider's model at commit time.
        """
        discovery_key = self._current_discovery_key()
        if self.selected_model_id and (
            discovery_key is None or self._selection_discovery_key == discovery_key
        ):
            return self.selected_model_id
        pressed = self._live_pressed_radio()
        if pressed is None:
            return ""
        # TASK-1503: read the clean id, never the (possibly decorated) label.
        return str(getattr(pressed, "_model_id", pressed.label))

    async def commit(self) -> tuple[bool, str]:
        _, provider_value = self._current_provider()
        model_id = self._effective_model_id()
        if not (provider_value and model_id):
            return True, ""  # skip-safe
        commit_staged = getattr(self.wizard, "commit_staged_provider_setup", None)
        if not callable(commit_staged):
            return False, "Return to Provider and review the connection."
        selection_key = self._selection_discovery_key
        if selection_key is None:
            pressed = self._live_pressed_radio()
            selection_key = getattr(pressed, "_discovery_key", None)
        provenance: Literal["discovered", "manual"] = (
            "manual" if self._model_id_from_custom_input else "discovered"
        )
        can_validate = getattr(
            self.wizard,
            "can_validate_committed_provider_setup",
            None,
        )
        if (
            callable(getattr(self.wizard, "capture_provider_config_precondition", None))
            and self._selection_config_precondition is None
            and not (
                callable(can_validate)
                and can_validate(
                    model_id,
                    discovery_key=selection_key,
                    model_provenance=provenance,
                )
            )
        ):
            return (
                False,
                (
                    "Connection settings changed. Refresh models or re-enter the "
                    "model ID."
                ),
            )
        commit_kwargs = {
            "discovery_key": selection_key,
            "model_provenance": provenance,
        }
        if self._selection_config_precondition is not None:
            commit_kwargs["config_precondition"] = self._selection_config_precondition
        ok = await commit_staged(model_id, **commit_kwargs)
        if ok:
            self.selected_model_id = model_id
            self._selection_discovery_key = self._current_discovery_key()
            if provenance == "manual":
                self._manual_decision_active = False
                self._selection_config_precondition = None
        elif (
            getattr(
                getattr(self.wizard, "_provider_last_config_result", None),
                "conflict_reason",
                None,
            )
            == "identity_changed"
            or selection_key != self._current_discovery_key()
        ):
            return (
                False,
                "Connection settings changed. Models were refreshed; select a "
                "model again or re-enter its ID.",
            )
        return (
            (True, "") if ok else (False, "Saving the provider and model setup failed.")
        )

    def get_step_data(self) -> Dict[str, Any]:
        return {"model_id": self._effective_model_id()}

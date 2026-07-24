"""Ephemeral preview-chat controller for the Personas workbench screen.

The preview conversation is intentionally in-memory: history lives here,
transcript rendering lives in ``PersonasPreviewPane``, and provider calls go
through the same Console gateway boundary without writing to local storage.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Mapping

from loguru import logger
from textual.css.query import QueryError

from ...Character_Chat.Character_Chat_Lib import replace_placeholders
from ...Chat.console_chat_models import ConsoleProviderSelection
from ...Chat.console_provider_gateway import ConsoleProviderGateway
from ...Chat.console_session_settings import build_default_console_session_settings
from ...Chat.provider_catalog import PROVIDER_DISPLAY_NAMES
from ...Widgets.Persona_Widgets.personas_character_editor_widget import (
    PersonasCharacterEditorWidget,
)
from ...Widgets.Persona_Widgets.personas_preview_pane import PersonasPreviewPane
from ..Navigation.main_navigation import NavigateToScreen
from ..Screens.settings_config_models import SettingsCategoryId

if TYPE_CHECKING:
    from ..Screens.personas_screen import PersonasScreen


logger = logger.bind(module="PersonasPreviewController")

# Cap on the preview transcript text staged into a Console handoff body.
PREVIEW_HANDOFF_TRANSCRIPT_CHAR_LIMIT = 6000


class PersonasPreviewController:
    """Owns preview chat state, provider calls, and Console handoff."""

    def __init__(self, screen: "PersonasScreen") -> None:
        self.screen = screen
        self.history: list[dict[str, str]] = []
        # Monotonic generation invalidates in-flight replies when Reset clears
        # the same selection without changing the selected entity key.
        self.generation: int = 0
        # Character id whose greeting last seeded the preview.
        self.seeded_for: str | None = None
        self.gateway: ConsoleProviderGateway | None = None
        # Provider key the Configure deep-link targets in Settings. This is
        # deliberately the CHARACTER-configured provider (or the chat default
        # when no character provider is set) - the provider a user would make
        # ready so replies stop falling back - even after a fallback send has
        # repainted the readout text with the provider that actually answered.
        self._readout_nav_provider: str = ""
        # Processed greetings for the selected character: index 0 is the
        # primary first_message, the rest are alternate_greetings (task-438).
        self._greetings: list[str] = []
        self._current_greeting_index: int = 0

    def invalidate(self) -> None:
        """Clear history and cancel in-flight preview workers."""
        self.generation += 1
        self.screen.workers.cancel_group(self.screen, "personas-preview")
        self.history.clear()

    async def reset(self, greeting: str, *, seeded_for: str | None = None) -> None:
        """Clear preview state and reseed the preview transcript.

        Args:
            greeting: Character greeting to render as the transcript seed.
            seeded_for: Optional selected character id represented by the seed.
        """
        self.invalidate()
        self.seeded_for = None
        try:
            await self.screen.query_one(PersonasPreviewPane).seed_greeting(greeting)
        except QueryError:
            # Tolerate calls that race screen teardown.
            pass
        self.seeded_for = seeded_for
        if seeded_for is None:
            self._greetings = []
            try:
                self.screen.query_one(PersonasPreviewPane).set_greetings([])
            except QueryError:
                pass
        self.refresh_provider_readout()

    def _load_greetings(self, record: dict[str, Any], name: str) -> str:
        """Store the processed greeting list, populate the selector, return the primary."""
        raw = [str(record.get("first_message") or "")]
        raw += [
            str(g)
            for g in (record.get("alternate_greetings") or [])
            if isinstance(g, str)
        ]
        self._greetings = [replace_placeholders(g, name, "User") for g in raw]
        self._current_greeting_index = 0
        try:
            self.screen.query_one(PersonasPreviewPane).set_greetings(self._greetings)
        except QueryError:
            pass
        return self._greetings[0] if self._greetings else ""

    async def reset_for_character(
        self,
        *,
        character_id: str,
        character_name: str,
        record: dict[str, Any] | None,
    ) -> None:
        """Seed the preview from a selected character when its full card is cached.

        Args:
            character_id: Selected character identifier.
            character_name: Display name used for greeting placeholders.
            record: Full character record, or ``None`` while the worker loads it.
        """
        if record is None:
            await self.reset("")
            return
        greeting = self._load_greetings(record, character_name)
        await self.reset(greeting, seeded_for=character_id)

    async def restore_conversation(
        self, *, greeting: str, history: list[dict], seeded_for: str | None
    ) -> None:
        """Rebuild the preview (greeting + turns) from saved state (task-434).

        Sets ``seeded_for`` before the first ``await``: ``invalidate`` cancels
        only the preview worker group, so the character-load worker's
        ``handle_character_loaded`` may still fire -- and must hit the
        seeded-for guard (``:159``) rather than erase the restored turns.

        Args:
            greeting: Saved greeting text to reseed the preview pane with.
            history: Saved user/assistant turns to replay into the pane.
            seeded_for: Entity id the saved preview was seeded for, normalized
                to ``str(seeded_for)`` (or ``None`` if falsy).

        Returns:
            None.
        """
        self.invalidate()
        self.seeded_for = str(seeded_for) if seeded_for else None
        try:
            pane = self.screen.query_one(PersonasPreviewPane)
        except QueryError:
            return
        await pane.seed_greeting(greeting)
        for message in history:
            role = message.get("role")
            content = str(message.get("content") or "")
            if role == "user":
                pane.append_user(content)
            elif role == "assistant":
                pane.append_reply(content)
        self.history = [dict(m) for m in history]
        self.refresh_provider_readout()

    async def close_gateway(self) -> None:
        """Release the preview gateway's HTTP client if it was created."""
        gateway = self.gateway
        self.gateway = None
        if gateway is None:
            return
        try:
            await gateway.aclose()
        except Exception:
            logger.bind(gateway_type=type(gateway).__name__).opt(
                exception=True
            ).warning("Could not close the preview provider gateway.")

    @staticmethod
    def _provider_label(provider_key: str) -> str:
        """Human-readable display name for a provider config key."""
        key = str(provider_key or "").strip()
        if not key:
            return ""
        return PROVIDER_DISPLAY_NAMES.get(
            key.lower(), key.replace("_", " ").replace("-", " ").title()
        )

    def provider_readout(self) -> tuple[str, str]:
        """Compute the pre-send provider readout from current config.

        The preview send path (``_run_reply``) tries ``character_defaults``
        first and falls back to ``chat_defaults`` when that provider is not
        ready (task-425). The readout mirrors that intent from config alone —
        no readiness probe — and is recomputed on every seed so config changes
        are reflected on the next selection.

        Returns:
            ``(readout_text, nav_provider)`` where ``nav_provider`` is the
            provider key the Configure deep-link should preselect.
        """
        raw_config = getattr(self.screen.app_instance, "app_config", {}) or {}
        config = raw_config if isinstance(raw_config, Mapping) else {}
        char_selection = self._selection_from_defaults(
            config, "character_defaults"
        )
        chat_selection = self._selection_from_defaults(config, "chat_defaults")
        char_provider = char_selection.provider
        char_model = self._selection_model(char_selection)
        chat_provider = chat_selection.provider
        chat_model = self._selection_model(chat_selection)
        if not char_provider:
            if chat_provider:
                text = (
                    f"Provider: {self._provider_label(chat_provider)} / "
                    f"{chat_model or 'default model'} (Console default)"
                )
                return text, chat_provider
            return "Provider: none configured - use Configure", ""
        text = (
            f"Provider: {self._provider_label(char_provider)} / "
            f"{char_model or 'default model'}"
        )
        # Note the fallback target only when it is a distinct provider. A
        # same-provider/different-model chat default is an approximation gap we
        # accept: this readout is config-only (no readiness probe), and after a
        # real send the readout is repainted with what actually resolved.
        if chat_provider and chat_provider.lower() != char_provider.lower():
            text += (
                " - Console default if unavailable: "
                f"{self._provider_label(chat_provider)}"
            )
        return text, char_provider

    def refresh_provider_readout(self) -> None:
        """Repaint the preview provider readout from current config."""
        text, nav_provider = self.provider_readout()
        self._readout_nav_provider = nav_provider
        try:
            self.screen.query_one(PersonasPreviewPane).set_provider_readout(text)
        except QueryError:
            pass

    def open_provider_settings(self) -> None:
        """Deep-link to Settings > Providers & Models for the readout provider."""
        context: dict[str, Any] = {"category": SettingsCategoryId.PROVIDERS_MODELS}
        if self._readout_nav_provider:
            context["provider"] = self._readout_nav_provider
        self.screen.post_message(NavigateToScreen("settings", context))

    async def handle_character_loaded(
        self, *, character_id: str, card_data: dict[str, Any] | None
    ) -> None:
        """Seed the preview greeting once the character load worker delivers.

        Args:
            character_id: Loaded character id.
            card_data: Full character card data from the load worker.
        """
        screen = self.screen
        character_id = str(character_id)
        if (
            screen.state.active_mode != "characters"
            or screen.state.selected_entity_kind != "character"
            or str(screen.state.selected_entity_id or "") != character_id
        ):
            return
        try:
            pane = screen.query_one(PersonasPreviewPane)
        except QueryError:
            return
        record = dict(card_data or {})
        name = str(record.get("name") or screen.state.selected_entity_name or "")
        greeting = self._load_greetings(record, name)
        # The speaker label is set once at selection (_select_character, from the
        # selection's display name) and must NOT be re-set here: set_speakers only
        # relabels FUTURE lines, so changing it on a same-character reload that
        # preserves the transcript would leave the existing lines under the old
        # prefix (mixed/stale prefixes — task-437 review).
        # Same-character reloads after an edit should update the reset seed,
        # not erase an in-progress preview conversation.
        if self.seeded_for == character_id and pane.transcript_text():
            pane.refresh_greeting_seed(greeting)
            self.refresh_provider_readout()
            return
        self.invalidate()
        await pane.seed_greeting(greeting)
        self.seeded_for = character_id
        self.refresh_provider_readout()

    def ensure_gateway(self) -> ConsoleProviderGateway:
        """Return the lazily-created Console provider gateway.

        Returns:
            Gateway configured from the app's current config snapshot.
        """
        if self.gateway is None:
            self.gateway = ConsoleProviderGateway(
                config_provider=lambda: (
                    getattr(self.screen.app_instance, "app_config", {}) or {}
                ),
            )
        return self.gateway

    def system_prompt(self) -> str:
        """Build a draft-aware system prompt for the selected persona source.

        Returns:
            Prompt text assembled from editor or selected-record fields.
        """
        screen = self.screen
        record: dict[str, Any] = {}
        if screen.state.active_mode == "characters":
            if screen._edit_mode in ("edit", "create"):
                try:
                    record = (
                        screen.query_one(
                            PersonasCharacterEditorWidget
                        ).get_character_data()
                        or {}
                    )
                except Exception:
                    logger.bind(
                        active_mode=screen.state.active_mode,
                        edit_mode=screen._edit_mode,
                        selection_kind=screen.state.selected_entity_kind,
                        selection_id=str(screen.state.selected_entity_id or ""),
                    ).opt(exception=True).warning(
                        "Could not collect editor data for the preview."
                    )
                    record = {}
            else:
                record = (
                    screen._full_character_record(
                        str(screen.state.selected_entity_id or "")
                    )
                    or {}
                )
        elif screen.state.active_mode == "personas":
            record = screen._profile_record(screen.state.selected_entity_id) or {}
        parts = [
            str(record.get(key) or "").strip()
            for key in ("system_prompt", "personality", "description", "scenario")
        ]
        prompt = "\n".join(part for part in parts if part)
        return prompt or "Stay in character."

    def handle_reply_requested(self, user_message: str) -> None:
        """Append the user turn and schedule one provider reply.

        Args:
            user_message: Message entered in the preview composer.
        """
        self.history.append({"role": "user", "content": user_message})
        self.screen.run_worker(
            self._run_reply(),
            exclusive=True,
            group="personas-preview",
        )

    def handle_reset(self) -> None:
        """Clear provider-facing preview state after the pane resets itself."""
        self.invalidate()

    async def handle_greeting_selected(self, index: int) -> None:
        """Re-seed the preview from the chosen greeting (AC#1); Reset then returns to it (AC#2)."""
        if index == self._current_greeting_index or not (
            0 <= index < len(self._greetings)
        ):
            return
        self._current_greeting_index = index
        self.invalidate()
        try:
            await self.screen.query_one(PersonasPreviewPane).seed_greeting(
                self._greetings[index]
            )
        except QueryError:
            pass

    def open_in_console(self) -> None:
        """Stage the current preview transcript as a Console handoff."""
        screen = self.screen
        transcript = screen.query_one(PersonasPreviewPane).transcript_text()
        truncated = len(transcript) > PREVIEW_HANDOFF_TRANSCRIPT_CHAR_LIMIT
        staged = screen._stage_handoff(
            item_type="preview-conversation",
            title="Personas preview conversation",
            body=transcript[:PREVIEW_HANDOFF_TRANSCRIPT_CHAR_LIMIT],
            body_truncated=truncated,
            suggested_prompt="Continue this conversation in character.",
        )
        if staged:
            screen._notify("Preview conversation staged in Console.", "information")

    def _pop_orphaned_user_turn(self) -> None:
        """Drop a trailing unanswered user entry from provider history only."""
        if self.history and self.history[-1].get("role") == "user":
            self.history.pop()

    @staticmethod
    def _selection_model(selection: ConsoleProviderSelection) -> str:
        """Return the effective model label from a provider selection."""
        return selection.explicit_model or selection.configured_model or ""

    def _reply_log_context(
        self,
        *,
        selection: ConsoleProviderSelection,
        selection_key: tuple[str | None, Any],
        generation: int,
        attempt: str,
        resolution: Any | None = None,
    ) -> dict[str, Any]:
        """Build safe context fields for preview provider exception logs."""
        selected_kind, selected_id = selection_key
        context: dict[str, Any] = {
            "operation": "personas_preview_reply",
            "provider": selection.provider,
            "model": self._selection_model(selection),
            "selection_kind": selected_kind or "",
            "selection_id": str(selected_id or ""),
            "generation": generation,
            "attempt": attempt,
            "streaming": selection.streaming,
        }
        if resolution is not None:
            context.update(
                resolved_provider=str(getattr(resolution, "provider", "") or ""),
                resolved_model=str(getattr(resolution, "model", "") or ""),
            )
        return context

    @staticmethod
    def _selection_from_defaults(
        config: Mapping[str, Any], defaults_key: str
    ) -> ConsoleProviderSelection:
        """Build a provider selection from one configured defaults section.

        Args:
            config: Current application configuration snapshot.
            defaults_key: Defaults section to resolve through Console settings.

        Returns:
            Provider selection carrying effective endpoint and generation fields.
        """
        raw_defaults = config.get(defaults_key, {})
        defaults = raw_defaults if isinstance(raw_defaults, Mapping) else {}
        provider = str(defaults.get("provider") or "").strip()
        explicit_model = str(defaults.get("model") or "").strip() or None

        settings_config: Mapping[str, Any] = config
        if defaults_key != "chat_defaults":
            # The Console builder owns endpoint, transport, and generation
            # precedence (ADR-006). Present character defaults through that
            # established input slot so the preview honors character-specific
            # sampling without duplicating config parsing.
            section_config = dict(config)
            section_config["chat_defaults"] = defaults
            settings_config = section_config
        settings = build_default_console_session_settings(
            settings_config,
            provider=provider,
            model=explicit_model,
        )
        return ConsoleProviderSelection(
            provider=settings.provider,
            base_url=settings.base_url,
            explicit_model=explicit_model,
            configured_model=None if explicit_model else settings.model,
            temperature=settings.temperature,
            top_p=settings.top_p,
            min_p=settings.min_p,
            top_k=settings.top_k,
            max_tokens=settings.max_tokens,
            seed=settings.seed,
            presence_penalty=settings.presence_penalty,
            frequency_penalty=settings.frequency_penalty,
            reasoning_effort=settings.reasoning_effort,
            reasoning_summary=settings.reasoning_summary,
            verbosity=settings.verbosity,
            thinking_effort=settings.thinking_effort,
            thinking_budget_tokens=settings.thinking_budget_tokens,
            streaming=settings.streaming,
        )

    async def _resolve_selection_with_fallback(
        self,
        gateway: ConsoleProviderGateway,
        *,
        selection_key: tuple[str | None, Any],
        generation: int,
    ) -> tuple[ConsoleProviderSelection, Any, str | None]:
        """Resolve ``character_defaults``, falling back to ``chat_defaults``.

        First-run configs carry the shipped ``[character_defaults]`` template
        (Anthropic) verbatim, so an unready character provider usually means
        the section was never a user choice — the user's working Console
        provider is the honest default (task-425). A usable character
        provider always wins.

        Args:
            gateway: Console provider gateway used for readiness resolution.
            selection_key: Selected entity identity for safe log context.
            generation: Preview generation used to reject stale replies.

        Returns:
            ``(selection, resolution, fallback_provider)`` where
            ``fallback_provider`` is the chat-defaults provider name when the
            fallback was used, else ``None``. ``resolution`` may be unready;
            the caller surfaces its copy.
        """
        raw_config = getattr(self.screen.app_instance, "app_config", {}) or {}
        config = raw_config if isinstance(raw_config, Mapping) else {}
        selection = self._selection_from_defaults(config, "character_defaults")
        try:
            resolution = await gateway.resolve_for_send(selection)
        except Exception:
            logger.bind(
                **self._reply_log_context(
                    selection=selection,
                    selection_key=selection_key,
                    generation=generation,
                    attempt="resolve",
                )
            ).opt(exception=True).error("Preview provider resolution failed.")
            raise
        if resolution.ready:
            return selection, resolution, None

        fallback = self._selection_from_defaults(config, "chat_defaults")
        chat_provider = fallback.provider
        same_target = (
            chat_provider.lower() == (selection.provider or "").lower()
            and self._selection_model(fallback) == self._selection_model(selection)
        )
        if chat_provider and not same_target:
            try:
                fallback_resolution = await gateway.resolve_for_send(fallback)
            except Exception:
                logger.bind(
                    **self._reply_log_context(
                        selection=fallback,
                        selection_key=selection_key,
                        generation=generation,
                        attempt="resolve",
                    )
                ).opt(exception=True).error("Preview provider resolution failed.")
                raise
            if fallback_resolution.ready:
                logger.bind(
                    character_provider=selection.provider,
                    fallback_provider=chat_provider,
                ).info(
                    "Character provider not ready; preview using chat_defaults."
                )
                return fallback, fallback_resolution, chat_provider
            # Both providers are unready. Surface the chat_defaults blocker: it
            # is the provider the guided-flow user actually configured, so its
            # copy is more actionable than the shipped character default's.
            return fallback, fallback_resolution, None
        return selection, resolution, None

    async def _run_reply(self) -> None:
        """Resolve the configured provider and stream one preview reply."""
        screen = self.screen
        pane = screen.query_one(PersonasPreviewPane)
        gateway = self.ensure_gateway()
        selection_key = (
            screen.state.selected_entity_kind,
            screen.state.selected_entity_id,
        )
        generation = self.generation

        def _stale() -> bool:
            return (
                not screen.is_mounted
                or generation != self.generation
                or (screen.state.selected_entity_kind, screen.state.selected_entity_id)
                != selection_key
            )

        try:
            (
                selection,
                resolution,
                fallback_provider,
            ) = await self._resolve_selection_with_fallback(
                gateway,
                selection_key=selection_key,
                generation=generation,
            )
        except Exception:
            if not _stale():
                self._pop_orphaned_user_turn()
                pane.set_status("Provider error - try again or configure in Settings")
            return
        if not resolution.ready:
            if not _stale():
                self._pop_orphaned_user_turn()
                copy = resolution.visible_copy or "Provider unavailable."
                pane.set_status(
                    f"{copy} Configure a provider in Settings: Providers & Models."
                )
            return
        if fallback_provider:
            pane.set_status(f"Running via Console default: {fallback_provider}")
        else:
            pane.set_status("Running")
        # Repaint the readout with the provider/model that actually resolved,
        # so it reflects reality (incl. a fallback) rather than config intent.
        resolved_readout = (
            f"Provider: {self._provider_label(resolution.provider)} / "
            f"{resolution.model or 'default model'}"
        )
        if fallback_provider:
            resolved_readout += " (Console default)"
        pane.set_provider_readout(resolved_readout)
        await pane.discard_partial_reply()
        history: list[dict[str, str]] = []
        for entry in self.history:
            if history and entry["role"] == "user" and history[-1]["role"] == "user":
                history[-1] = {
                    "role": "user",
                    "content": f"{history[-1]['content']}\n{entry['content']}",
                }
            else:
                history.append(dict(entry))
        messages = [{"role": "system", "content": self.system_prompt()}] + history

        async def _consume(res: Any) -> str | None:
            """Render one provider stream into the pane; ``None`` means stale."""
            consumed = ""
            opened = False
            async for chunk in gateway.stream_chat(res, messages):
                if _stale():
                    await pane.discard_partial_reply()
                    return None
                consumed += chunk
                if chunk:
                    if not opened:
                        pane.begin_reply()
                        opened = True
                    pane.append_reply_chunk(chunk)
            return consumed

        try:
            reply = await _consume(resolution)
        except Exception:
            logger.bind(
                **self._reply_log_context(
                    selection=selection,
                    selection_key=selection_key,
                    generation=generation,
                    attempt="streaming",
                    resolution=resolution,
                )
            ).opt(exception=True).error(
                "Preview provider call failed; retrying without streaming."
            )
            if _stale():
                await pane.discard_partial_reply()
                return
            await pane.discard_partial_reply()
            pane.set_status("Retrying without streaming...")
            retry_selection = dataclasses.replace(selection, streaming=False)
            retry_resolution = None
            try:
                retry_resolution = await gateway.resolve_for_send(retry_selection)
                if not retry_resolution.ready:
                    raise RuntimeError(
                        retry_resolution.visible_copy or "provider unavailable"
                    )
                reply = await _consume(retry_resolution)
            except Exception:
                logger.bind(
                    **self._reply_log_context(
                        selection=retry_selection,
                        selection_key=selection_key,
                        generation=generation,
                        attempt="non_streaming",
                        resolution=retry_resolution,
                    )
                ).opt(exception=True).error("Preview non-streaming retry failed.")
                if not _stale():
                    self._pop_orphaned_user_turn()
                    await pane.discard_partial_reply()
                    pane.set_status(
                        "Provider error - try again or configure in Settings"
                    )
                return
        if reply is None:
            return
        if _stale():
            await pane.discard_partial_reply()
            return
        if reply:
            self.history.append({"role": "assistant", "content": reply})
            pane.finalize_reply()
            if fallback_provider:
                pane.set_status(f"Ready - via Console default: {fallback_provider}")
            else:
                pane.set_status("Ready")
        else:
            pane.set_status("No reply received")

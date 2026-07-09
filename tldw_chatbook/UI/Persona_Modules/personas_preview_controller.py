"""Ephemeral preview-chat controller for the Personas workbench screen.

The preview conversation is intentionally in-memory: history lives here,
transcript rendering lives in ``PersonasPreviewPane``, and provider calls go
through the same Console gateway boundary without writing to local storage.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from loguru import logger
from textual.css.query import QueryError

from ...Character_Chat.Character_Chat_Lib import replace_placeholders
from ...Chat.console_chat_models import ConsoleProviderSelection
from ...Chat.console_provider_gateway import ConsoleProviderGateway
from ...Widgets.Persona_Widgets.personas_character_editor_widget import (
    PersonasCharacterEditorWidget,
)
from ...Widgets.Persona_Widgets.personas_preview_pane import PersonasPreviewPane

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
        greeting = replace_placeholders(
            str(record.get("first_message") or ""), character_name, "User"
        )
        await self.reset(greeting, seeded_for=character_id)

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
        greeting = replace_placeholders(
            str(record.get("first_message") or ""), name, "User"
        )
        # Same-character reloads after an edit should update the reset seed,
        # not erase an in-progress preview conversation.
        if self.seeded_for == character_id and pane.transcript_text():
            pane.refresh_greeting_seed(greeting)
            return
        self.invalidate()
        await pane.seed_greeting(greeting)
        self.seeded_for = character_id

    def ensure_gateway(self) -> ConsoleProviderGateway:
        """Return the lazily-created Console provider gateway.

        Returns:
            Gateway configured from the app's current config snapshot.
        """
        if self.gateway is None:
            self.gateway = ConsoleProviderGateway(
                config_provider=lambda: getattr(
                    self.screen.app_instance, "app_config", {}
                )
                or {},
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
                        screen.query_one(PersonasCharacterEditorWidget).get_character_data()
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
                record = screen._full_character_record(
                    str(screen.state.selected_entity_id or "")
                ) or {}
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

    async def _run_reply(self) -> None:
        """Resolve the configured provider and stream one preview reply."""
        screen = self.screen
        pane = screen.query_one(PersonasPreviewPane)
        config = getattr(screen.app_instance, "app_config", {}) or {}
        defaults = config.get("character_defaults", {}) or {}
        provider = str(defaults.get("provider") or "")
        model = str(defaults.get("model") or "")
        selection = ConsoleProviderSelection(provider=provider, explicit_model=model or None)
        gateway = self.ensure_gateway()
        selection_key = (screen.state.selected_entity_kind, screen.state.selected_entity_id)
        generation = self.generation

        def _stale() -> bool:
            return (
                not screen.is_mounted
                or generation != self.generation
                or (screen.state.selected_entity_kind, screen.state.selected_entity_id)
                != selection_key
            )

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
            if not _stale():
                self._pop_orphaned_user_turn()
                pane.set_status("Provider error - try again or configure in Settings")
            return
        if not resolution.ready:
            if not _stale():
                self._pop_orphaned_user_turn()
                pane.set_status(
                    resolution.visible_copy or "Provider unavailable - configure in Settings"
                )
            return
        pane.set_status("Running")
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
            pane.set_status("Ready")
        else:
            pane.set_status("No reply received")

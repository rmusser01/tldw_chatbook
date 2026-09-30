"""Canonical Console persistence protocol, re-exported by console_chat_store."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence  # noqa: UP035

from .citation_trace_models import SealedCitationWrite
from .console_context_policy import ConsoleContextPolicyOverrides
from .console_dispatch_checkpoint import (
    ConsoleDispatchCheckpoint,
    ConsoleDurableTurnAcceptance,
)
from .console_library_policy import ConsoleLibraryPolicyCandidate
from .console_session_endpoint_policy import ConsoleEndpointAdoptionReceipt
from .console_session_settings import ConsoleSessionSettings
from .console_speech_preferences import ConsoleSpeechPreferences
from .library_activity import LibraryActivityContribution
from .thinking_blocks import ThinkingHistoryPolicy


class ConsoleChatPersistence(Protocol):
    """Persistence surface used by Console without importing DB dependencies."""

    #: Raw DB handle backing this persistence adapter, or ``None`` when the
    #: adapter has none (e.g. a test fake, or a future persistence shape
    #: with no single underlying database). ``persist_session_if_needed``
    #: reaches through this seam -- rather than an undeclared ``getattr``
    #: probe -- to flush a session-held RAG retrieval scope
    #: (``SessionScopeHolder``) at first persistence (PR #747 review: a
    #: conforming adapter that structurally satisfied this Protocol without
    #: declaring ``.db`` made the flush silently no-op, losing the user's
    #: pre-persistence scope selection with no diagnostic). Declaring it
    #: here makes the seam an explicit, checkable part of the contract.
    db: Any | None

    def thinking_round_trip_version(self) -> int:
        """Return the exact supported durable thinking envelope version."""

    def commit_durable_turn(
        self,
        *,
        acceptance: ConsoleDurableTurnAcceptance,
        policy_candidate: ConsoleLibraryPolicyCandidate,
        conversation_kwargs: Mapping[str, object],
        context_policy_overrides: ConsoleContextPolicyOverrides | None = None,
    ) -> ConsoleDispatchCheckpoint:
        """Atomically create/validate and accept one durable Console turn."""

    def persist_console_library_activity(
        self,
        *,
        conversation_id: str,
        contribution: LibraryActivityContribution,
        message_ids: Mapping[str, str],
    ) -> None:
        """Persist one activity batch in a caller-owned transaction."""

    def create_conversation(self, **kwargs) -> str:
        """Create a persisted conversation and return its ID."""

    def get_console_fork_citation_state(
        self,
        message_id: str,
        revision: int,
        source_body: str,
        target_body: str,
    ) -> str:
        """Confirm one durable source message's citation ownership state."""

    def create_message(
        self,
        *,
        conversation_id: str,
        sender: str,
        content: str,
        image_data: bytes | None,
        image_mime_type: str | None,
        message_id: str | None = None,
        parent_message_id: str | None = None,
        feedback: str | None = None,
        attachments: Sequence[Mapping[str, Any]] | None = None,
        citation_write: SealedCitationWrite | None = None,
        usage_json: str | None = None,
        metadata_json: str | None = None,
        thinking_blocks_json: str | None = None,
        provider_continuation_json: str | None = None,
        assistant_generation_state: str | None = None,
    ) -> str:
        """Create a persisted message and return its ID.

        ``attachments``, when given, covers ALL positions (0..N-1) and is
        authoritative over the scalar ``image_data``/``image_mime_type``
        kwargs; ``None`` leaves the pre-split legacy behavior unchanged.
        Optional: fakes used in tests may omit this parameter entirely.

        ``citation_write``, when present, is committed atomically with the
        message by citation-aware adapters. Narrow test fakes may omit this
        optional parameter entirely.

        ``usage_json`` (Console cost ticker), when present, is the
        message's normalized provider-usage JSON. Optional: narrow test
        fakes may omit this parameter entirely -- the store only passes it
        to adapters that declare it (see ``_persistence_accepts_kwarg``).

        ``metadata_json`` (task-2364), when present, is the message's
        structured metadata JSON (engine provenance, interrupted flag,
        transcript status). Same optionality and same declare-to-receive
        rule as ``usage_json``.

        The three assistant-generation fields are optional for narrow fakes,
        but production adapters receive them in the same create transaction.
        """

    def update_message_content(
        self,
        *,
        message_id: str,
        content: str,
        image_data: bytes | None,
        image_mime_type: str | None,
        parent_message_id: str | None = None,
        feedback: str | None = None,
        update_parent: bool = False,
        update_feedback: bool = False,
        attachments: Sequence[Mapping[str, Any]] | None = None,
        usage_json: str | None = None,
        metadata_json: str | None = None,
        expected_version: int | None = None,
        preserve_provider_continuation: bool = False,
        clear_generation_provenance: bool = False,
    ) -> bool:
        """Update persisted message content.

        ``attachments`` follows the same split-addressing contract as
        ``create_message``; ``None`` (the Console store's edit path always
        passes this) leaves attachments untouched. Optional: fakes used in
        tests may omit this parameter entirely.

        ``usage_json`` (Console cost ticker), when present, overwrites the
        row's normalized provider-usage JSON. Optional: narrow test fakes
        may omit this parameter entirely -- the store only passes it to
        adapters that declare it, and only when usage is actually known,
        so a content-only update never clobbers an existing value with
        ``None``.

        ``metadata_json`` (task-2364) follows the identical contract for
        the structured metadata column.
        """

    def read_canonical_generation_projection(
        self, message_id: str
    ) -> Mapping[str, Any] | None:
        """Read the canonical versioned fields for a body-only projection."""

    def read_canonical_generation_projection_bundle(
        self, message_id: str
    ) -> Mapping[str, Any] | None:
        """Read one row and every generation sidecar from one DB snapshot."""

    def replace_assistant_generation_projection(
        self,
        *,
        message_id: str,
        content: str,
        thinking_blocks_json: str | None,
        provider_continuation_json: str | None,
        assistant_generation_state: str | None,
        usage_json: str | None,
        expected_version: int | None = None,
    ) -> int:
        """Atomically replace one selected assistant generation."""

    def update_message_usage(self, *, message_id: str, usage_json: str) -> bool:
        """Persist normalized usage as a version-neutral, local-only write.

        Unlike ``update_message_content``'s optional ``usage_json`` kwarg
        (which rides a content update and legitimately bumps the row's
        version), this method exists SOLELY for a usage-only flush against
        an already-terminal message -- the Stop-path case described on
        ``ConsoleChatStore.set_message_usage``. It must not advance
        ``version``/``last_modified`` (the ``messages_sync_update`` trigger
        watches those columns, not just content, so bumping them on a
        usage-only write would enqueue a ``sync_log`` row whose payload can
        never carry ``usage_json`` -- pure cross-device churn for a column
        that is local-only by design).

        Entirely optional: this whole method, not just a kwarg, may be
        absent. The store probes for it with ``hasattr``/``callable``
        (same philosophy as ``_persistence_accepts_kwarg``) and falls back
        to the ordinary content-carrying update path when it is not
        present, so narrow test fakes written before this method existed
        keep working unchanged.
        """

    def update_message_metadata(self, *, message_id: str, metadata_json: str) -> bool:
        """Persist structured metadata as a version-neutral, local-only write.

        The task-2364 sibling of ``update_message_usage`` above, with the
        identical contract: metadata-only flush against an already-persisted
        row, no ``version``/``last_modified`` bump (the
        ``messages_sync_update`` trigger watches those, and no sync payload
        can ever carry ``metadata_json``), and entirely optional -- the
        store probes for it and falls back to the content-carrying update
        path when an adapter does not provide it.
        """

    def append_message_exchanges(
        self, *, message_id: str, rows: Sequence[Mapping[str, Any]]
    ) -> bool:
        """Upsert captured provider exchanges for a message (local-only).

        The Conversation Inspector sibling of ``update_message_usage`` --
        each row carries its own ``run_tag``/``seq`` identity, so this is an
        upsert rather than a single-column write. Entirely optional, probed
        the same hasattr+callable way as ``update_message_usage``: a
        persistence adapter that does not implement it simply never
        receives an exchange flush (``ConsoleChatStore._persist_exchanges_
        only`` bails silently rather than falling back to the content path
        -- captures have no content-carrying fallback to ride).
        """

    def get_message_version(self, message_id: str) -> int | None:
        """Return the current positive durable row version, if trustworthy.

        Args:
            message_id: Persisted Chat message identifier.

        Returns:
            The exact positive integer row version, or ``None`` when the row
            cannot provide a trustworthy version fence.
        """

    def get_console_fork_source_message(
        self, message_id: str
    ) -> tuple[int, str] | None:
        """Return one exact persisted source revision/body pair for a fork fence."""

    def get_console_fork_active_leaf(self, conversation_id: str) -> str | None:
        """Return the canonical durable active leaf used by a fork fence."""

    def get_conversation_version(self, conversation_id: str) -> int | None:
        """Return the current positive durable conversation row version."""

    def get_conversation_speech_preferences(
        self, conversation_id: str
    ) -> ConsoleSpeechPreferences:
        """Return fail-closed reply-speech preferences for one conversation."""

    def update_conversation_speech_preferences(
        self,
        *,
        conversation_id: str,
        preferences: ConsoleSpeechPreferences,
        expected_version: int,
    ) -> bool:
        """Optimistically merge reply-speech preferences into metadata."""

    def update_conversation_system_prompt(
        self,
        *,
        conversation_id: str,
        system_prompt: str | None,
    ) -> bool:
        """Persist a changed system prompt for an already-saved conversation."""

    def update_conversation_thinking_history_policy(
        self,
        *,
        conversation_id: str,
        policy: ThinkingHistoryPolicy,
    ) -> bool:
        """Persist one conversation-owned optional thinking replay policy."""

    def update_conversation_roleplay_context(
        self,
        *,
        conversation_id: str,
        user_name_override: str | None,
        character_system_template: str | None,
        character_name_snapshot: str | None = None,
        persona_system_template: str | None = None,
    ) -> bool:
        """Persist Console-owned roleplay identity context for a conversation.

        Args:
            conversation_id: Durable conversation identifier.
            user_name_override: Optional saved user display-name override.
            character_system_template: Optional saved character prompt template.
            character_name_snapshot: Optional historical character display name.
            persona_system_template: Optional saved persona prompt template.

        Returns:
            True when the roleplay context was persisted.
        """

    def update_conversation_pinned_prefill(
        self,
        *,
        conversation_id: str,
        pinned_prefill: str | None,
    ) -> bool:
        """Set or clear the pinned response prefill on a conversation."""

    def update_conversation_console_session_settings(
        self,
        *,
        conversation_id: str,
        settings: ConsoleSessionSettings,
    ) -> bool:
        """Persist the latest complete Console settings snapshot."""

    def adopt_console_session_endpoint_settings(
        self,
        *,
        conversation_id: str,
        settings: ConsoleSessionSettings,
    ) -> ConsoleEndpointAdoptionReceipt:
        """Persist endpoint-safe settings and return an exact rollback receipt."""

    def rollback_console_session_endpoint_adoption(
        self,
        *,
        receipt: ConsoleEndpointAdoptionReceipt,
    ) -> bool:
        """Restore pre-adoption metadata while the receipt still owns the row."""

    def update_conversation_title(
        self,
        *,
        conversation_id: str,
        title: str,
    ) -> bool:
        """Persist a changed title for an already-saved conversation.

        Args:
            conversation_id: Durable Chat conversation identifier.
            title: New conversation title (already validated non-blank).

        Returns:
            True when the update was applied; False when refused (e.g. an
            optimistic-lock version check failed).
        """

    def get_conversation_console_project_context(
        self, *, conversation_id: str
    ) -> str | None:
        """Return versioned local project-context JSON when available."""

    def set_conversation_console_project_context(
        self,
        *,
        conversation_id: str,
        project_context_json: str | None,
    ) -> None:
        """Write local project-context JSON without synchronized metadata."""

    def get_attachments_for_messages(
        self, message_ids: Sequence[str]
    ) -> dict[str, list[dict[str, Any]]]:
        """Batch-fetch extra (position >= 1) attachments for messages.

        Optional: not all persistence fakes implement this. Callers should
        probe with ``getattr(persistence, "get_attachments_for_messages", None)``
        before invoking it (see Task 5).
        """

    def append_message_attachment(
        self,
        message_id: str,
        *,
        data: bytes,
        mime_type: str,
        display_name: str = "",
        generation_metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Append one new image variant to a message, in place (no rewrite).

        Optional: not all persistence fakes implement this. Callers should
        probe with ``getattr(persistence, "append_message_attachment", None)``
        before invoking it -- the narrow, additive counterpart to
        ``update_message_content(attachments=...)`` used by
        ``ConsoleChatStore.append_generation_variant``.
        """

    def keep_message_attachment(self, message_id: str, position: int) -> None:
        """Promote a stored variant to be the message's canonical image.

        Optional: not all persistence fakes implement this. Callers should
        probe with ``getattr(persistence, "keep_message_attachment", None)``
        before invoking it -- a targeted position swap, used by
        ``ConsoleChatStore.keep_generation_variant`` instead of the
        full-list ``update_message_content(attachments=...)`` rewrite (which
        would NULL any in-memory byte-less variant it re-sends).
        """

    def get_generation_metadata_for_messages(
        self, message_ids: Sequence[str]
    ) -> dict[str, list[dict[str, Any]]]:
        """Batch-fetch generation-metadata sidecar rows for messages.

        Optional: not all persistence fakes implement this. Callers should
        probe with
        ``getattr(persistence, "get_generation_metadata_for_messages", None)``
        before invoking it -- feeds
        ``ConsoleChatStore.hydrate_generation_metadata`` at conversation
        load.
        """

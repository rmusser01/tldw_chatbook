"""Console-owned linearized activation of exact character conversations."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, TypeVar

from loguru import logger

from tldw_chatbook.Character_Chat.character_conversation_navigation import (
    LocalCharacterConversationTarget,
)

ConsoleStateT = TypeVar("ConsoleStateT")


@dataclass(frozen=True)
class CharacterConversationActivationRequest:
    """Immutable query identity carried from selection through Console commit."""

    target: LocalCharacterConversationTarget
    data_authority_id: str
    data_revision: int

    def __post_init__(self) -> None:
        if not isinstance(self.target, LocalCharacterConversationTarget):
            raise TypeError("target must be a LocalCharacterConversationTarget")
        if self.data_authority_id != self.target.character.data_authority_id:
            raise ValueError("activation authority must match the target authority")
        if (
            isinstance(self.data_revision, bool)
            or not isinstance(self.data_revision, int)
            or self.data_revision < 0
        ):
            raise ValueError("data_revision must be a non-negative integer")


class ConsoleActivationPhase(StrEnum):
    """Presentation phase for an exact Console activation."""

    IDLE = "idle"
    OPENING_CANCELLABLE = "opening_cancellable"
    COMMITTING = "committing"
    FAILURE_VISIBLE = "failure_visible"


class ConsoleActivationResultKind(StrEnum):
    """Closed outcome set returned to every activation caller."""

    OPENED = "opened"
    CANCELLED_PRECOMMIT = "cancelled_precommit"
    NOT_FOUND = "not_found"
    DATA_PROFILE_CHANGED = "data_profile_changed"
    CHARACTER_UNAVAILABLE = "character_unavailable"
    FAILED = "failed"


@dataclass(frozen=True)
class ConsoleConversationActivationResult:
    """Outcome of one exact target activation attempt."""

    kind: ConsoleActivationResultKind
    target: LocalCharacterConversationTarget
    commit_started: bool


@dataclass(frozen=True)
class ConsoleActivationCommit:
    """Console hydration outcome plus the exact runtime owned by this attempt."""

    opened: bool
    owned_runtime_token: object | None = None


async def _maybe_await(value):
    return await value if inspect.isawaitable(value) else value


class ConsoleConversationActivationCoordinator(Generic[ConsoleStateT]):
    """Single-flight activation coordinator with a pre-commit cancel point.

    The Console supplies its incumbent state capture, exact revalidation,
    hydration/open, rollback, and visible-destination checks. The coordinator
    owns ordering: revalidation and cancellation finish before the commit
    acknowledgement, and every post-commit failure restores the captured state.
    """

    def __init__(
        self,
        *,
        capture_state: Callable[[], ConsoleStateT],
        revalidate: Callable[
            [LocalCharacterConversationTarget | CharacterConversationActivationRequest],
            ConsoleActivationResultKind
            | None
            | Awaitable[ConsoleActivationResultKind | None],
        ],
        open_target: Callable[
            [LocalCharacterConversationTarget | CharacterConversationActivationRequest],
            object | Awaitable[object],
        ],
        rollback_opened_target: Callable[[object], None | Awaitable[None]],
        restore_state: Callable[[ConsoleStateT], None | Awaitable[None]],
        exact_target_visible: Callable[
            [LocalCharacterConversationTarget | CharacterConversationActivationRequest],
            bool | Awaitable[bool],
        ],
        mutation_lock: asyncio.Lock | None = None,
    ) -> None:
        self._capture_state = capture_state
        self._revalidate = revalidate
        self._open_target = open_target
        self._rollback_opened_target = rollback_opened_target
        self._restore_state = restore_state
        self._exact_target_visible = exact_target_visible
        self._mutation_lock = mutation_lock or asyncio.Lock()
        self._admission_lock = asyncio.Lock()
        self._active_request: (
            LocalCharacterConversationTarget
            | CharacterConversationActivationRequest
            | None
        ) = None
        self._active_attempt: (
            asyncio.Task[ConsoleConversationActivationResult] | None
        ) = None
        self._active_commit_event: asyncio.Event | None = None
        self.phase = ConsoleActivationPhase.IDLE

    async def activate(
        self,
        target: LocalCharacterConversationTarget
        | CharacterConversationActivationRequest,
        cancellation: asyncio.Event | None = None,
    ) -> ConsoleConversationActivationResult:
        """Activate ``target`` or join its existing exact single-flight attempt."""

        if not isinstance(
            target,
            (LocalCharacterConversationTarget, CharacterConversationActivationRequest),
        ):
            raise TypeError("activation requires a typed target or request")
        while True:
            async with self._admission_lock:
                attempt = self._active_attempt
                if attempt is None:
                    commit_event = asyncio.Event()
                    attempt = asyncio.create_task(
                        self._run_serialized_attempt(
                            target, cancellation or asyncio.Event(), commit_event
                        )
                    )
                    self._active_request = target
                    self._active_attempt = attempt
                    self._active_commit_event = commit_event
                    attempt.add_done_callback(self._release)
                    join = True
                else:
                    join = self._active_request == target
            if join:
                return await asyncio.shield(attempt)
            # A different target waits outside admission.  It may then claim
            # the one global Console mutation lane; its revalidation is fresh.
            await asyncio.shield(attempt)

    async def _run_serialized_attempt(
        self,
        target: LocalCharacterConversationTarget
        | CharacterConversationActivationRequest,
        cancellation: asyncio.Event,
        commit_event: asyncio.Event,
    ) -> ConsoleConversationActivationResult:
        """Hold the app-owned Console mutation lane for the whole attempt."""

        async with self._mutation_lock:
            return await self._run_attempt(target, cancellation, commit_event)

    async def wait_until_commit_started(
        self,
        target: LocalCharacterConversationTarget
        | CharacterConversationActivationRequest,
    ) -> None:
        """Wait until the current attempt crosses its commit linearization point."""

        event = self._active_commit_event if self._active_request == target else None
        if event is None:
            await asyncio.sleep(0)
            event = (
                self._active_commit_event if self._active_request == target else None
            )
        if event is None:
            raise RuntimeError("target has no activation attempt")
        await event.wait()

    def _release(
        self, attempt: asyncio.Task[ConsoleConversationActivationResult]
    ) -> None:
        if self._active_attempt is attempt:
            self._active_request = None
            self._active_attempt = None
            self._active_commit_event = None

    async def _run_attempt(
        self,
        target: LocalCharacterConversationTarget
        | CharacterConversationActivationRequest,
        cancellation: asyncio.Event,
        commit_event: asyncio.Event,
    ) -> ConsoleConversationActivationResult:
        prior_state = self._capture_state()
        self.phase = ConsoleActivationPhase.OPENING_CANCELLABLE
        commit_started = False
        opened_token: object | None = None
        try:
            revalidation = await _maybe_await(self._revalidate(target))
            if revalidation is not None:
                if revalidation not in {
                    ConsoleActivationResultKind.NOT_FOUND,
                    ConsoleActivationResultKind.DATA_PROFILE_CHANGED,
                    ConsoleActivationResultKind.CHARACTER_UNAVAILABLE,
                }:
                    revalidation = ConsoleActivationResultKind.FAILED
                self.phase = ConsoleActivationPhase.FAILURE_VISIBLE
                return ConsoleConversationActivationResult(
                    revalidation,
                    target.target
                    if isinstance(target, CharacterConversationActivationRequest)
                    else target,
                    False,
                )
            if cancellation.is_set():
                self.phase = ConsoleActivationPhase.IDLE
                return ConsoleConversationActivationResult(
                    ConsoleActivationResultKind.CANCELLED_PRECOMMIT,
                    target.target
                    if isinstance(target, CharacterConversationActivationRequest)
                    else target,
                    False,
                )

            commit_started = True
            self.phase = ConsoleActivationPhase.COMMITTING
            commit_event.set()
            opened = await _maybe_await(self._open_target(target))
            if isinstance(opened, ConsoleActivationCommit):
                opened_token = opened.owned_runtime_token
                opened_ok = opened.opened
            else:
                opened_token = opened if opened else None
                opened_ok = bool(opened)
            visible = (
                bool(await _maybe_await(self._exact_target_visible(target)))
                if opened_ok
                else False
            )
            if opened_ok and visible:
                self.phase = ConsoleActivationPhase.IDLE
                return ConsoleConversationActivationResult(
                    ConsoleActivationResultKind.OPENED,
                    target.target
                    if isinstance(target, CharacterConversationActivationRequest)
                    else target,
                    True,
                )
        except asyncio.CancelledError:
            if not commit_started:
                self.phase = ConsoleActivationPhase.IDLE
                return ConsoleConversationActivationResult(
                    ConsoleActivationResultKind.CANCELLED_PRECOMMIT,
                    target.target
                    if isinstance(target, CharacterConversationActivationRequest)
                    else target,
                    False,
                )
            raise
        except Exception:  # noqa: BLE001 - boundary converts adapters to typed failure
            logger.bind(
                operation_id=id(commit_event),
                target_type="local_character_conversation",
                stage="open_target",
            ).opt(exception=True).warning(
                "Character-conversation activation failed after validation"
            )

        if commit_started:
            if opened_token is not None:
                try:
                    await _maybe_await(self._rollback_opened_target(opened_token))
                except Exception:  # noqa: BLE001 - continue restoring prior UI
                    logger.bind(
                        operation_id=id(commit_event),
                        target_type="local_character_conversation",
                        stage="remove_owned_runtime",
                        runtime_token=id(opened_token),
                    ).opt(exception=True).error(
                        "Could not remove owned Console session after failed activation"
                    )
            try:
                await _maybe_await(self._restore_state(prior_state))
            except Exception:  # noqa: BLE001 - rollback is best-effort at boundary
                logger.bind(
                    operation_id=id(commit_event),
                    target_type="local_character_conversation",
                    stage="restore_prior_runtime",
                ).opt(exception=True).error(
                    "Could not restore prior Console session after failed activation"
                )
        self.phase = ConsoleActivationPhase.FAILURE_VISIBLE
        return ConsoleConversationActivationResult(
            ConsoleActivationResultKind.FAILED,
            target.target
            if isinstance(target, CharacterConversationActivationRequest)
            else target,
            commit_started,
        )

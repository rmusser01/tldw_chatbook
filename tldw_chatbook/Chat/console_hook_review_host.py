"""Initial hook-review methods on the existing resident interrupt host.

This mixin has no constructor or separate state. InterruptRoundHost owns its
lock, registries, FIFO, presentation tokens and pending-decision accounting.
"""

from __future__ import annotations

import asyncio
import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from loguru import logger

from tldw_chatbook.Chat.console_chat_models import CONSOLE_PENDING_HOOK_REVIEW_KIND
from tldw_chatbook.Chat.console_hook_review import (
    ConsoleHookReviewProjection,
    HookReviewResult,
)

if TYPE_CHECKING:
    from tldw_chatbook.Agents.hook_permissions import (
        HookPermissions,
        HookReviewSnapshot,
    )


@dataclass(slots=True, eq=False)
class _HookReviewOperation:
    """Exact native custody, reserved before task scheduling."""

    review_id: str
    generation: int
    session_id: str
    purpose: Literal["approve", "revoke", "disable", "recover", "reset", "verify"]
    owner: HookPermissions = field(repr=False)
    expected: HookReviewSnapshot = field(repr=False)
    presentation_token: object = field(repr=False)
    retired: asyncio.Future[None] = field(repr=False)
    task: asyncio.Task | None = field(default=None, repr=False)


class InitialHookReviewMixin:
    """Named hook-review lifecycle on one shared InterruptRoundHost instance."""

    def _hook_review_current_locked(self, state) -> bool:
        """Fail closed on displaced source/session, including accessor errors."""
        if state.get("settled") or state.get("revoked"):
            return False
        try:
            session = state["session_ref"]()
            store = self.read_controller_store()
            return bool(
                session is not None
                and store is state["store"]
                and any(row is session for row in store.sessions())
                and session.conversation_binding_revision == state["binding_revision"]
                and session.ephemeral == state["ephemeral"]
                and (
                    self.read_hook_review_shutdown is None
                    or not self.read_hook_review_shutdown(state["session_id"])
                )
                and (
                    self.read_hook_review_owner is None
                    or self.read_hook_review_owner() is state["owner"]
                )
            )
        except Exception:
            return False

    def _hook_review_head_locked(self, state) -> bool:
        payloads = self._pending_decision_payloads_locked(state["session_id"])
        return bool(payloads and payloads[0].get("_decision_id") == state["review_id"])

    def _hook_review_project_current(self) -> None:
        callback = self.read_controller_project_pending_decision_for_active_session()
        if callable(callback):
            callback()
        self._publish_console_attention_change()

    def _hook_review_changed(self, state) -> None:
        state["loop"].call_soon_threadsafe(self._hook_review_project_current)

    @staticmethod
    def _hook_review_complete_future(future, value) -> None:
        if not future.done():
            future.set_result(value)

    def _cancel_hook_review_exact(self, review_id: str, generation: int) -> bool:
        with self.lock:
            state = self.registries[CONSOLE_PENDING_HOOK_REVIEW_KIND].get(review_id)
            if (
                state is None
                or state["generation"] != generation
                or state.get("settled")
            ):
                return False
            state["settled"] = True
            state["revoked"] = True
            state["result"] = HookReviewResult("cancel")
            state["terminal_reason"] = "cancel"
            if state["operation"] is not None:
                self.payloads[CONSOLE_PENDING_HOOK_REVIEW_KIND][review_id]["phase"] = (
                    "finishing"
                )
        state["loop"].call_soon_threadsafe(
            self._hook_review_complete_future, state["answer"], state["result"]
        )
        self.retire_hook_review(review_id, generation)
        self._hook_review_changed(state)
        return True

    def _hook_review_accounting_changed(self, session_id, review_id, pending) -> None:
        """Only disposable attention callbacks run outside the resident lock."""
        try:
            sink = self.read_controller__buddy_sink()
            if sink is not None:
                sink.approval_round(session_id, review_id, pending=pending)
            self.read_controller__advance_lifecycle_revision()(session_id)
            self._publish_console_attention_change()
        except Exception:
            logger.debug("Hook review attention callback failed.")

    def begin_hook_review(
        self,
        session_id: str,
        review_id: str,
        generation: int,
        snapshot: HookReviewSnapshot,
        *,
        waiting_for_send: bool,
        owner: HookPermissions,
        loop: asyncio.AbstractEventLoop,
    ) -> asyncio.Future[HookReviewResult]:
        """Publish an initial async review without a thread parked for input."""
        store = self.read_controller_store()
        session = next((row for row in store.sessions() if row.id == session_id), None)
        if session is None:
            raise RuntimeError("Hook review session is unavailable.")
        answer = loop.create_future()
        state = {
            "review_id": review_id,
            "session_id": session_id,
            "generation": generation,
            "owner": owner,
            "snapshot": snapshot,
            "waiting_for_send": waiting_for_send,
            "loop": loop,
            "answer": answer,
            "store": store,
            "session_ref": weakref.ref(session),
            "binding_revision": session.conversation_binding_revision,
            "ephemeral": session.ephemeral,
            "operation": None,
            "presentation_token": None,
            "attachment_generation": None,
            "settled": False,
            "revoked": False,
            "terminal_reason": None,
            "remaining_active_seconds": None,
            "active_since": None,
            "decision_type": CONSOLE_PENDING_HOOK_REVIEW_KIND,
            "decision_id": review_id,
        }
        with self.lock:
            registry = self.registries[CONSOLE_PENDING_HOOK_REVIEW_KIND]
            if review_id in registry:
                raise RuntimeError("Hook review is already registered.")
            if not self._hook_review_current_locked(state):
                raise RuntimeError("Hook review source or session changed.")
            order = self.read_controller__pending_decision_order() + 1
            self.write_controller__pending_decision_order(order)
            state["decision_order"] = order
            registry[review_id] = state
            self.payloads[CONSOLE_PENDING_HOOK_REVIEW_KIND][review_id] = {
                "session_id": session_id,
                "generation": generation,
                "snapshot": snapshot,
                "waiting_for_send": waiting_for_send,
                "busy": False,
                "_decision_type": CONSOLE_PENDING_HOOK_REVIEW_KIND,
                "_decision_id": review_id,
                "_decision_order": order,
            }
            self.read_controller__pending_approvals().setdefault(session_id, set()).add(
                review_id
            )
            self.read_controller__pending_round_kinds().setdefault(session_id, {})[
                review_id
            ] = CONSOLE_PENDING_HOOK_REVIEW_KIND
        self._hook_review_accounting_changed(session_id, review_id, True)
        self._hook_review_changed(state)
        return answer

    def claim_hook_review_presentation(
        self,
        review_id: str,
        generation: int,
        attachment_generation: int,
        *,
        replace_existing: bool = False,
    ) -> ConsoleHookReviewProjection | None:
        """Claim only the selected session; displaced exact sources terminate."""
        displaced = False
        with self.lock:
            state = self.registries[CONSOLE_PENDING_HOOK_REVIEW_KIND].get(review_id)
            if state is None or state["generation"] != generation:
                return None
            if not self._hook_review_current_locked(state):
                displaced = not state.get("settled")
            elif state["store"].active_session_id == state[
                "session_id"
            ] and self._hook_review_head_locked(state):
                token = state["presentation_token"]
                if (
                    replace_existing
                    or token is None
                    or state["attachment_generation"] != attachment_generation
                ):
                    if token is not None:
                        self._decision_views.pop(token, None)
                    token = object()
                    state["presentation_token"] = token
                    state["attachment_generation"] = attachment_generation
                    self._decision_views[token] = (
                        state["session_id"],
                        frozenset((CONSOLE_PENDING_HOOK_REVIEW_KIND,)),
                        (
                            state["session_ref"],
                            state["binding_revision"],
                            state["ephemeral"],
                            review_id,
                        ),
                    )
                    self.decision_view_revision += 1
                return ConsoleHookReviewProjection(
                    review_id=review_id,
                    session_id=state["session_id"],
                    generation=generation,
                    snapshot=state["snapshot"],
                    waiting_for_send=state["waiting_for_send"],
                    presentation_token=token,
                    attachment_generation=attachment_generation,
                    busy=state["operation"] is not None,
                )
        if displaced:
            self._cancel_hook_review_exact(review_id, generation)
        return None

    def release_hook_review_presentation(
        self, review_id: str, generation: int, token: object
    ) -> bool:
        with self.lock:
            state = self.registries[CONSOLE_PENDING_HOOK_REVIEW_KIND].get(review_id)
            if (
                state is None
                or state["generation"] != generation
                or state["presentation_token"] is not token
            ):
                return False
            state["presentation_token"] = None
            state["attachment_generation"] = None
            self._decision_views.pop(token, None)
            self.decision_view_revision += 1
        self._hook_review_changed(state)
        return True

    def release_hook_review_attachment(self, attachment_generation: int) -> None:
        with self.lock:
            released = []
            for state in self.registries[CONSOLE_PENDING_HOOK_REVIEW_KIND].values():
                if state["attachment_generation"] == attachment_generation:
                    token = state["presentation_token"]
                    if token is not None:
                        self._decision_views.pop(token, None)
                    state["presentation_token"] = None
                    state["attachment_generation"] = None
                    released.append(state)
            if released:
                self.decision_view_revision += 1
        for state in released:
            self._hook_review_changed(state)

    def begin_hook_review_operation(
        self, review_id: str, generation: int, token: object, purpose: str
    ) -> _HookReviewOperation | None:
        """Reserve exact physical custody before scheduling any task."""
        if purpose not in {
            "approve",
            "revoke",
            "disable",
            "recover",
            "reset",
            "verify",
        }:
            raise ValueError("Unknown hook review operation.")
        displaced = False
        with self.lock:
            state = self.registries[CONSOLE_PENDING_HOOK_REVIEW_KIND].get(review_id)
            if (
                state is None
                or state["generation"] != generation
                or state.get("settled")
            ):
                return None
            if not self._hook_review_current_locked(state):
                displaced = not state.get("settled")
            elif (
                token is not None
                and state["presentation_token"] is token
                and state["operation"] is None
                and state["store"].active_session_id == state["session_id"]
                and self._hook_review_head_locked(state)
            ):
                operation = _HookReviewOperation(
                    review_id=review_id,
                    generation=generation,
                    session_id=state["session_id"],
                    purpose=purpose,
                    owner=state["owner"],
                    expected=state["snapshot"],
                    presentation_token=token,
                    retired=state["loop"].create_future(),
                )
                state["operation"] = operation
                self.payloads[CONSOLE_PENDING_HOOK_REVIEW_KIND][review_id]["busy"] = (
                    True
                )
            else:
                return None
        if displaced:
            self._cancel_hook_review_exact(review_id, generation)
            return None
        self._hook_review_changed(state)
        return operation

    def finish_hook_review_operation(
        self,
        operation: _HookReviewOperation,
        snapshot: HookReviewSnapshot | None,
        *,
        error: BaseException | None = None,
    ) -> bool:
        """Retire physical work; stale presentations can never produce Ready."""
        changed = False
        terminal = None
        notify_state = None
        with self.lock:
            state = self.registries[CONSOLE_PENDING_HOOK_REVIEW_KIND].get(
                operation.review_id
            )
            if (
                state is not None
                and state["generation"] == operation.generation
                and state["operation"] is operation
            ):
                notify_state = state
                state["operation"] = None
                payload = self.payloads[CONSOLE_PENDING_HOOK_REVIEW_KIND][
                    operation.review_id
                ]
                payload["busy"] = False
                if state.get("settled"):
                    terminal = state.get("result")
                elif not self._hook_review_current_locked(state) or (
                    snapshot is not None
                    and (
                        snapshot.config.config_path
                        != state["snapshot"].config.config_path
                        or snapshot.store_path != state["snapshot"].store_path
                    )
                ):
                    terminal = HookReviewResult("cancel")
                else:
                    if snapshot is not None:
                        state["snapshot"] = snapshot
                        payload["snapshot"] = snapshot
                    if (
                        operation.purpose == "verify"
                        and error is None
                        and snapshot is not None
                        and snapshot.ready
                        and state["presentation_token"] is operation.presentation_token
                        and state["store"].active_session_id == state["session_id"]
                        and self._hook_review_head_locked(state)
                    ):
                        terminal = HookReviewResult("ready", snapshot)
                    changed = True
                if terminal is not None and not state.get("settled"):
                    state["settled"] = True
                    state["result"] = terminal
                    state["terminal_reason"] = terminal.kind
        operation.retired.get_loop().call_soon_threadsafe(
            self._hook_review_complete_future, operation.retired, None
        )
        if notify_state is not None:
            if terminal is not None:
                notify_state["loop"].call_soon_threadsafe(
                    self._hook_review_complete_future, notify_state["answer"], terminal
                )
                self.retire_hook_review(operation.review_id, operation.generation)
            self._hook_review_changed(notify_state)
        return changed

    def resolve_hook_review(
        self,
        review_id: str,
        generation: int,
        result: HookReviewResult,
        *,
        presentation_token: object | None = None,
    ) -> bool:
        """Resolve Cancel/Settings only; Ready requires a verified operation."""
        if result.kind not in {"cancel", "settings"}:
            return False
        with self.lock:
            state = self.registries[CONSOLE_PENDING_HOOK_REVIEW_KIND].get(review_id)
            if (
                state is None
                or state["generation"] != generation
                or state.get("settled")
                or presentation_token is None
                or state["presentation_token"] is not presentation_token
                or not self._hook_review_current_locked(state)
                or not self._hook_review_head_locked(state)
            ):
                return False
            state["settled"] = True
            state["result"] = result
            state["terminal_reason"] = result.kind
            if state["operation"] is not None:
                self.payloads[CONSOLE_PENDING_HOOK_REVIEW_KIND][review_id]["phase"] = (
                    "finishing"
                )
        state["loop"].call_soon_threadsafe(
            self._hook_review_complete_future, state["answer"], result
        )
        self.retire_hook_review(review_id, generation)
        self._hook_review_changed(state)
        return True

    def cancel_hook_reviews(self, session_id: str | None = None) -> None:
        """Seal logical requests; keep issued work until actual retirement."""
        with self.lock:
            states = [
                state
                for state in self.registries[CONSOLE_PENDING_HOOK_REVIEW_KIND].values()
                if session_id is None or state["session_id"] == session_id
            ]
            for state in states:
                if not state.get("settled"):
                    state["settled"] = True
                    state["revoked"] = True
                    state["result"] = HookReviewResult("cancel")
                    state["terminal_reason"] = "cancel"
                    if state["operation"] is not None:
                        self.payloads[CONSOLE_PENDING_HOOK_REVIEW_KIND][
                            state["review_id"]
                        ]["phase"] = "finishing"
        for state in states:
            state["loop"].call_soon_threadsafe(
                self._hook_review_complete_future, state["answer"], state["result"]
            )
            self.retire_hook_review(state["review_id"], state["generation"])
            self._hook_review_changed(state)

    def hook_review_retirements(
        self, session_id: str | None = None
    ) -> tuple[asyncio.Future[None], ...]:
        with self.lock:
            return tuple(
                state["operation"].retired
                for state in self.registries[CONSOLE_PENDING_HOOK_REVIEW_KIND].values()
                if state["operation"] is not None
                and (session_id is None or state["session_id"] == session_id)
            )

    def retire_hook_review(self, review_id: str, generation: int) -> bool:
        """Remove the exact terminal record without depending on a waiter."""
        with self.lock:
            registry = self.registries[CONSOLE_PENDING_HOOK_REVIEW_KIND]
            state = registry.get(review_id)
            if (
                state is None
                or state["generation"] != generation
                or not state.get("settled")
                or state["operation"] is not None
            ):
                return False
            token = state["presentation_token"]
            if token is not None:
                self._decision_views.pop(token, None)
            registry.pop(review_id)
            self.payloads[CONSOLE_PENDING_HOOK_REVIEW_KIND].pop(review_id, None)
            self.read_controller__announced_pending_decision_ids().discard(review_id)
            session_id = state["session_id"]
            rounds = self.read_controller__pending_approvals().get(session_id)
            if rounds is not None:
                rounds.discard(review_id)
                if not rounds:
                    self.read_controller__pending_approvals().pop(session_id, None)
            kinds = self.read_controller__pending_round_kinds().get(session_id)
            if kinds is not None:
                kinds.pop(review_id, None)
                if not kinds:
                    self.read_controller__pending_round_kinds().pop(session_id, None)
        self._hook_review_accounting_changed(state["session_id"], review_id, False)
        self._hook_review_changed(state)
        return True

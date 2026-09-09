"""Content-free adapters from trusted Console lifecycle state to Buddy leases."""

from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass
from threading import RLock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .controller import PersonaBuddyController, PersonaBuddyLeaseToken


_SOURCES = frozenset(
    {"console-run", "approval", "tool", "wake", "voice", "explicit", "authored"}
)
_BUILTIN_STATES = frozenset(
    {
        "idle",
        "listening",
        "thinking",
        "speaking",
        "approval_needed",
        "tool_running",
        "wake_armed",
        "offline",
        "error",
    }
)
_CUSTOM_STATE_PATTERN = re.compile(r"[a-z][a-z0-9_.:-]{0,95}\Z")
_OWNER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_CUSTOM_STATE_SOURCES = frozenset({"explicit", "authored"})
_UNSAFE_STATE_PREFIXES = ("env:", "file:", "ftp:", "http:", "https:", "proc:", "ssh:")
_UNSAFE_STATE_MARKERS = (
    "access_token",
    "api_key",
    "apikey",
    "auth_token",
    "authorization",
    "bearer_token",
    "client_secret",
    "password",
    "passwd",
    "private_key",
    "refresh_token",
    "secret",
    "secret_key",
)
_RUN_STATES = {
    "validating": "thinking",
    "checking": "thinking",
    "checking_citations": "thinking",
    "retrying": "thinking",
    "streaming": "speaking",
    "failed": "error",
}
_RUN_RELEASE_STATES = frozenset({"idle", "completed", "stopped", "blocked"})
_VOICE_STATES = {
    "live": "listening",
    "listening": "listening",
    "thinking": "thinking",
    "speaking": "speaking",
    "connecting": "offline",
    "reconnecting": "offline",
}
_TOOL_START = "tool_call"
_TOOL_RELEASE = frozenset({"tool_result", "error", "cancelled", "terminal"})


@dataclass(frozen=True, slots=True)
class BuddyLifecycleEvent:
    """One trusted operational state transition with no user/model content."""

    source: str
    owner: str
    state: str
    terminal: bool = False
    expires_at: float | None = None

    def __post_init__(self) -> None:
        if self.source not in _SOURCES:
            raise ValueError("persona_buddy_console_source_invalid")
        if type(self.owner) is not str or _OWNER_PATTERN.fullmatch(self.owner) is None:
            raise ValueError("persona_buddy_console_owner_invalid")
        if type(self.state) is not str or not self.state:
            raise ValueError("persona_buddy_console_state_invalid")
        if type(self.terminal) is not bool:
            raise ValueError("persona_buddy_console_terminal_invalid")
        if self.expires_at is not None and (
            type(self.expires_at) is not float or not math.isfinite(self.expires_at)
        ):
            raise ValueError("persona_buddy_console_expiry_invalid")
        if _CUSTOM_STATE_PATTERN.fullmatch(self.state) is None:
            raise ValueError("persona_buddy_console_state_invalid")
        if self.state.startswith(_UNSAFE_STATE_PREFIXES):
            raise ValueError("persona_buddy_console_state_invalid")
        compact_state = re.sub(r"[._:-]+", "_", self.state)
        if any(marker in compact_state for marker in _UNSAFE_STATE_MARKERS):
            raise ValueError("persona_buddy_console_state_invalid")
        if (
            self.state not in _BUILTIN_STATES
            and self.source not in _CUSTOM_STATE_SOURCES
        ):
            raise ValueError("persona_buddy_console_state_invalid")
        if self.source == "explicit" and not self.terminal and self.expires_at is None:
            raise ValueError("persona_buddy_console_expiry_invalid")


class PersonaBuddyConsoleAdapter:
    """Thread-safe source/owner ledger for trusted Console producers."""

    def __init__(self, controller: PersonaBuddyController | None) -> None:
        self._controller = controller
        self._lock = RLock()
        self._clock = time.monotonic
        self._disposed = False
        self._scope_sessions: frozenset[str] | None = None
        self._scope_conversations: frozenset[str] | None = None
        self._tokens: dict[tuple[str, str], PersonaBuddyLeaseToken] = {}
        self._run_generation: dict[str, int] = {}
        self._run_owners: dict[str, str] = {}
        self._approval_owners: dict[tuple[str, str], str] = {}
        self._tool_owners: dict[tuple[str, int], str] = {}
        self._tool_sessions: dict[tuple[str, int], str | None] = {}
        self._wake_owners: dict[tuple[str, str], str] = {}
        self._voice_owners: dict[tuple[str, int], str] = {}
        self._voice_generation: dict[str, int] = {}
        self._next_voice_generation = 0

    def bind_controller(self, controller: PersonaBuddyController | None) -> None:
        """Bind the app controller once it exists during startup ordering."""
        with self._lock:
            if self._disposed or controller is None:
                return
            if self._controller is None:
                self._controller = controller
            elif self._controller is not controller:
                raise RuntimeError("persona_buddy_console_controller_replacement")

    def set_scope(
        self, *, session_ids: frozenset[str], conversation_ids: frozenset[str]
    ) -> bool:
        """Replace permitted owners, retiring leases from the previous scope.

        Returns whether scope changed. Callers replay current trusted state only
        after a change; repeated reconciliation does not restart animations.
        """
        if not isinstance(session_ids, frozenset) or not isinstance(
            conversation_ids, frozenset
        ):
            raise TypeError("buddy_scope_invalid")
        if any(
            type(value) is not str or not value
            for value in session_ids | conversation_ids
        ):
            raise ValueError("buddy_scope_invalid")
        with self._lock:
            if self._disposed:
                return False
            if (session_ids, conversation_ids) == (
                self._scope_sessions,
                self._scope_conversations,
            ):
                return False
            # Scope affects presentation, not accepted run identity. Keep leases
            # for retained members, and keep the run ledger so a reattached
            # projection accepts that run's original terminal callback.
            retained = {
                ("console-run", owner)
                for session, owner in self._run_owners.items()
                if session in session_ids
            }
            retained.update(
                ("approval", owner)
                for (session, _), owner in self._approval_owners.items()
                if session in session_ids
            )
            retained.update(
                ("tool", owner)
                for key, owner in self._tool_owners.items()
                if self._tool_sessions.get(key) in session_ids
            )
            retained.update(
                ("wake", owner)
                for (conversation, _), owner in self._wake_owners.items()
                if conversation in conversation_ids
            )
            retained.update(
                ("voice", owner)
                for (session, _), owner in self._voice_owners.items()
                if session in session_ids
            )
            for key, token in tuple(self._tokens.items()):
                if key not in retained:
                    if self._controller is not None:
                        self._controller.release_state(token=token)
                    self._tokens.pop(key, None)
            self._approval_owners = {
                key: owner
                for key, owner in self._approval_owners.items()
                if key[0] in session_ids
            }
            self._tool_owners = {
                key: owner
                for key, owner in self._tool_owners.items()
                if ("tool", owner) in retained
            }
            self._tool_sessions = {
                key: session
                for key, session in self._tool_sessions.items()
                if key in self._tool_owners
            }
            self._wake_owners = {
                key: owner
                for key, owner in self._wake_owners.items()
                if key[0] in conversation_ids
            }
            self._voice_owners = {
                key: owner
                for key, owner in self._voice_owners.items()
                if key[0] in session_ids
            }
            self._voice_generation = {
                session: generation
                for session, generation in self._voice_generation.items()
                if session in session_ids
            }
            self._scope_sessions = session_ids
            self._scope_conversations = conversation_ids
            return True

    def _session_allowed(self, session_id: str | None) -> bool:
        return self._scope_sessions is None or session_id in self._scope_sessions

    @staticmethod
    def _owner(kind: str, *parts: object) -> str:
        digest = hashlib.sha256()
        for part in parts:
            encoded = str(part).encode("utf-8", errors="strict")
            digest.update(len(encoded).to_bytes(4, "big"))
            digest.update(encoded)
        return f"{kind}:{digest.hexdigest()[:32]}"

    def publish(self, event: BuddyLifecycleEvent) -> bool:
        """Acquire, replace, or release one exact content-free event owner."""

        key = (event.source, event.owner)
        with self._lock:
            if self._disposed:
                return False
            self._prune_expired_locked()
            controller = self._controller
            if controller is None:
                return False
            if (
                event.source == "explicit"
                and event.expires_at is not None
                and event.expires_at <= self._clock()
            ):
                return False
            if event.terminal:
                token = self._tokens.pop(key, None)
                self._drop_owner_refs_locked(event.source, event.owner)
                return token is not None and controller.release_state(token=token)
            token = controller.acquire_state(
                source=event.source,
                owner=event.owner,
                state=event.state,
                expires_at=event.expires_at,
            )
            self._tokens[key] = token
            return True

    def _release(self, source: str, owner: str) -> bool:
        return self.publish(
            BuddyLifecycleEvent(
                source=source,
                owner=owner,
                state="idle",
                terminal=True,
            )
        )

    def run_state(
        self,
        session_id: str,
        status: object,
        *,
        run_owner: str | None = None,
        replay: bool = False,
    ) -> str | None:
        """Map run state; scope replay preserves an already accepted owner."""

        normalized = str(getattr(status, "value", status)).strip().lower()
        with self._lock:
            if self._disposed or self._controller is None:
                return None
            owner = self._run_owners.get(session_id)
            if normalized == "validating" and not (replay and owner is not None):
                if owner is not None:
                    self._release("console-run", owner)
                generation = self._run_generation.get(session_id, 0) + 1
                self._run_generation[session_id] = generation
                owner = self._owner("run", session_id, generation)
                self._run_owners[session_id] = owner
            elif run_owner is not None:
                if owner != run_owner:
                    return None
                owner = run_owner
            elif owner is None and normalized in _RUN_STATES:
                generation = self._run_generation.get(session_id, 0) + 1
                self._run_generation[session_id] = generation
                owner = self._owner("run", session_id, generation)
                self._run_owners[session_id] = owner
            if normalized in _RUN_RELEASE_STATES:
                if owner is not None:
                    self._release("console-run", owner)
                    self._run_owners.pop(session_id, None)
                return None
            state = _RUN_STATES.get(normalized)
            if state is None or owner is None or not self._session_allowed(session_id):
                return owner
            self.publish(
                BuddyLifecycleEvent(
                    source="console-run",
                    owner=owner,
                    state=state,
                )
            )
            return owner

    def approval_round(
        self, session_id: str, round_id: str, *, pending: bool
    ) -> str | None:
        """Acquire or settle one exact approval round."""

        key = (session_id, round_id)
        with self._lock:
            if (
                self._disposed
                or self._controller is None
                or not self._session_allowed(session_id)
            ):
                return None
            owner = self._approval_owners.get(key)
            if not pending:
                if owner is not None:
                    self._release("approval", owner)
                    self._approval_owners.pop(key, None)
                return None
            if owner is None:
                owner = self._owner("approval", session_id, round_id)
                self._approval_owners[key] = owner
            self.publish(
                BuddyLifecycleEvent(
                    source="approval",
                    owner=owner,
                    state="approval_needed",
                )
            )
            return owner

    def tool_step(
        self, run_id: str, sequence: int, kind: str, *, session_id: str | None = None
    ) -> str | None:
        """Pair a tool-call start with its exact result or run cleanup."""

        key = (run_id, sequence)
        with self._lock:
            if (
                self._disposed
                or self._controller is None
                or not self._session_allowed(session_id)
            ):
                return None
            if kind == _TOOL_START:
                self._tool_sessions[key] = session_id
                owner = self._tool_owners.get(key)
                if owner is None:
                    owner = self._owner("tool", run_id, sequence)
                    self._tool_owners[key] = owner
                self.publish(
                    BuddyLifecycleEvent(
                        source="tool",
                        owner=owner,
                        state="tool_running",
                    )
                )
                return owner
            if kind in _TOOL_RELEASE:
                owner = self._tool_owners.pop(key, None)
                self._tool_sessions.pop(key, None)
                if owner is not None:
                    self._release("tool", owner)
                return None
            return self._tool_owners.get(key)

    def release_run(self, run_id: str) -> None:
        """Release every still-live tool owner for one terminal run."""

        with self._lock:
            if self._disposed:
                return
            for key, owner in tuple(self._tool_owners.items()):
                if key[0] == run_id:
                    self._release("tool", owner)
                    self._tool_owners.pop(key, None)
                    self._tool_sessions.pop(key, None)

    def wake(self, conversation_id: str, run_id: str, *, active: bool) -> str | None:
        """Mirror one pending or delivering fleet-wake membership."""

        key = (conversation_id, run_id)
        with self._lock:
            if (
                self._disposed
                or self._controller is None
                or (
                    self._scope_conversations is not None
                    and conversation_id not in self._scope_conversations
                )
            ):
                return None
            owner = self._wake_owners.get(key)
            if not active:
                if owner is not None:
                    self._release("wake", owner)
                    self._wake_owners.pop(key, None)
                return None
            if owner is None:
                owner = self._owner("wake", conversation_id, run_id)
                self._wake_owners[key] = owner
            self.publish(
                BuddyLifecycleEvent(
                    source="wake",
                    owner=owner,
                    state="wake_armed",
                )
            )
            return owner

    def clear_wakes(self, conversation_id: str | None = None) -> None:
        """Release settled wake owners, optionally for one conversation."""

        with self._lock:
            if self._disposed:
                return
            for key, owner in tuple(self._wake_owners.items()):
                if conversation_id is None or key[0] == conversation_id:
                    self._release("wake", owner)
                    self._wake_owners.pop(key, None)

    def voice_state(self, session_id: str, generation: int, state: str) -> str | None:
        """Publish one exact realtime-loop generation and fence replacements."""

        key = (session_id, generation)
        with self._lock:
            if (
                self._disposed
                or self._controller is None
                or not self._session_allowed(session_id)
            ):
                return None
            current_generation = self._voice_generation.get(session_id)
            if current_generation is not None and generation < current_generation:
                return None
            if current_generation is not None and generation > current_generation:
                self.release_voice(session_id, current_generation)
            self._voice_generation[session_id] = generation
            owner = self._voice_owners.get(key)
            if state == "idle":
                self.release_voice(session_id, generation)
                return None
            mapped = _VOICE_STATES.get(state)
            if mapped is None:
                return owner
            if owner is None:
                owner = self._owner("voice", session_id, generation)
                self._voice_owners[key] = owner
            self.publish(BuddyLifecycleEvent(source="voice", owner=owner, state=mapped))
            return owner

    def next_voice_generation(self, session_id: str) -> int | None:
        """Mint one app-wide monotonic realtime-loop incarnation."""

        del session_id  # the token is intentionally app-wide, not screen-local
        with self._lock:
            if self._disposed:
                return None
            self._next_voice_generation += 1
            return self._next_voice_generation

    def release_voice(self, session_id: str, generation: int) -> None:
        """Release only the named realtime-loop generation."""

        key = (session_id, generation)
        with self._lock:
            if self._disposed:
                return
            owner = self._voice_owners.pop(key, None)
            if owner is not None:
                self._release("voice", owner)
            if self._voice_generation.get(session_id) == generation:
                self._voice_generation.pop(session_id, None)

    def release_session(
        self, session_id: str, *, sources: Iterable[str] | None = None
    ) -> None:
        """Release exact session-owned run, approval, and voice leases."""

        allowed = set(sources or {"console-run", "approval", "voice"})
        with self._lock:
            if self._disposed:
                return
            if "console-run" in allowed:
                owner = self._run_owners.pop(session_id, None)
                if owner is not None:
                    self._release("console-run", owner)
            if "approval" in allowed:
                for key, owner in tuple(self._approval_owners.items()):
                    if key[0] == session_id:
                        self._release("approval", owner)
                        self._approval_owners.pop(key, None)
            if "voice" in allowed:
                generation = self._voice_generation.get(session_id)
                if generation is not None:
                    self.release_voice(session_id, generation)

    def release_all(self) -> None:
        """Release every owner while keeping this adapter reusable."""

        with self._lock:
            if self._disposed:
                return
            self._prune_expired_locked()
            self._release_all_locked()

    def dispose(self) -> None:
        """Terminally fence producers before releasing every exact token."""

        with self._lock:
            if self._disposed:
                return
            self._disposed = True
            self._release_all_locked()
            self._run_generation.clear()
            self._next_voice_generation = 0

    def _release_all_locked(self) -> None:
        """Release and clear all live owner maps. Caller holds ``_lock``."""

        controller = self._controller
        for token in tuple(self._tokens.values()):
            if controller is not None:
                controller.release_state(token=token)
        self._tokens.clear()
        self._run_owners.clear()
        self._approval_owners.clear()
        self._tool_owners.clear()
        self._tool_sessions.clear()
        self._wake_owners.clear()
        self._voice_owners.clear()
        self._voice_generation.clear()

    def _prune_expired_locked(self) -> None:
        """Drop expired timed owners from token and source bookkeeping."""

        now = self._clock()
        controller = self._controller
        for key, token in tuple(self._tokens.items()):
            if token.expires_at is None or token.expires_at > now:
                continue
            self._tokens.pop(key, None)
            self._drop_owner_refs_locked(token.source, token.owner)
            if controller is not None:
                controller.release_state(token=token)

    def _drop_owner_refs_locked(self, source: str, owner: str) -> None:
        """Remove one exact released owner from its source index."""

        if source == "console-run":
            indexed = self._run_owners
        elif source == "approval":
            indexed = self._approval_owners
        elif source == "tool":
            indexed = self._tool_owners
        elif source == "wake":
            indexed = self._wake_owners
        elif source == "voice":
            indexed = self._voice_owners
        else:
            return
        for key, candidate in tuple(indexed.items()):
            if candidate != owner:
                continue
            indexed.pop(key, None)
            if source == "voice" and self._voice_generation.get(key[0]) == key[1]:
                self._voice_generation.pop(key[0], None)

    def active_owner_count(self, source: str | None = None) -> int:
        """Return a content-free owner count for focused lifecycle tests."""

        with self._lock:
            self._prune_expired_locked()
            if source is None:
                return len(self._tokens)
            return sum(1 for token_source, _ in self._tokens if token_source == source)

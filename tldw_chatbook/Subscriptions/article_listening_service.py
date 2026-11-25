"""App-owned resumable transport for local saved Watchlists articles."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from pathlib import Path
import re
from threading import Event
from typing import TYPE_CHECKING, Literal
from functools import wraps
import time
from uuid import uuid4

from .article_listening_lease import ListeningBusy
from .article_listening_models import ListeningCheckpoint, ListeningConflict
from .article_listening_store import ArticleListeningStore

if TYPE_CHECKING:
    from tldw_chatbook.TTS.audio_output_ownership import (
        AudioOutputIntent,
        AudioOutputLease,
    )
    from tldw_chatbook.TTS.controlled_wav_playback import ControlledWavPlayback
    from .article_listening_models import (
        ListeningAuthority,
        ListeningCandidate,
        ListeningEntry,
    )
    from .article_listening_store import _TerminalOwner


@dataclass(frozen=True, slots=True)
class ListeningRevision:
    """Opaque command target scoped to one service, policy and durable session."""

    service: str
    epoch: int
    incarnation: str | None
    revision: int
    entry_id: str | None


@dataclass(frozen=True, slots=True)
class ListeningSnapshot:
    """Read-only transport projection. Titles are private display values."""

    phase: str = "empty"
    revision: ListeningRevision | None = None
    current_title: str | None = field(default=None, repr=False)
    suspended_title: str | None = field(default=None, repr=False)
    queue_position: int | None = None
    queue_count: int = 0
    playable_count: int = 0
    unavailable_count: int = 0
    stored_feed_text: bool | None = None
    is_excerpt: bool | None = None
    elapsed_frames: int | None = None
    duration_frames: int | None = None
    sample_rate: int | None = None
    completed_chunks: int | None = None
    total_chunks: int | None = None
    error: str | None = None
    actions: frozenset[str] = frozenset()
    unavailable_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BriefListeningPreview:
    playable_count: int
    unavailable_reasons: tuple[str, ...]
    action: Literal["play", "resume", "resume_foreground", "play_again", "replace"]


@dataclass(slots=True)
class _Admission:
    ticket: AudioOutputIntent | None = None
    cancelled: bool = False
    selected: bool = False

    def withdraw(self):
        if not self.selected:
            self.cancelled = True
            if self.ticket is not None:
                self.ticket.cancel()


def _serialized_command(method):
    """Transfer a selected command to its owner while allowing its observer to leave."""

    @wraps(method)
    async def command(self, *args, **kwargs):
        await self._command.acquire()

        async def perform():
            try:
                return await method(self, *args, **kwargs)
            finally:
                self._command.release()

        return await asyncio.shield(self._retain(perform()))

    return command


@dataclass(eq=False, repr=False)
class _Attempt:
    entry: ListeningEntry
    authority: ListeningAuthority
    terminal_owner: _TerminalOwner
    ticket: AudioOutputIntent
    epoch: int
    run: ControlledWavPlayback | None = None
    lease: AudioOutputLease | None = None
    task: asyncio.Task | None = None
    transition: str | None = None
    candidate: ListeningCandidate | None = None
    ended: bool = False
    advance: bool = False
    revoked: bool = False
    preparing_output: bool = False
    ambiguous_readiness_pin: bool = False
    next_checkpoint: float = 0
    rejection: str | None = None


_REASONS = frozenset(
    """
    article_audio_busy article_audio_corrupt article_audio_database_unavailable article_audio_disk_full
    article_audio_effects_unqualified article_audio_limit article_audio_missing article_audio_retired
    article_audio_storage_unavailable article_audio_unavailable article_conversion_unavailable
    article_kokoro_engine_unbound article_native_budget_unavailable article_retirement_unconfirmed
    article_speed_unavailable article_chunk_limit article_text_limit controlled_prepare_failed
    explicit_regeneration_required invalid_article_audio listening_article_limit listening_busy
    listening_authority_changed listening_brief_limit listening_brief_unavailable listening_database_changed
    listening_entry_changed listening_entry_unavailable listening_generation_changed listening_revision_changed
    listening_source_changed listening_voice_changed listening_voice_unbound listening_voice_unavailable
    local_listening_authority_required no_playable_articles pin_retirement_unconfirmed player_birth_unconfirmed
    player_containment_unavailable player_failed player_retirement_unconfirmed player_start_failed
    player_start_refused previous_player_unconfirmed progress_unavailable qualified_ffplay_unavailable
    resource_retirement_unconfirmed terminal_progress_unconfirmed unsupported_audio_route unsupported_platform
    startup_deadline_exceeded checkpoint_unconfirmed
""".split()
)


def _reason(error):
    if isinstance(error, (ValueError, ListeningBusy)) and len(error.args) == 1:
        value = error.args[0]
        if type(value) is str and value in _REASONS:
            return value
    return "listening_unavailable"


class ArticleListeningService:
    """Coordinate the session, synthesis operation and exact output attempt."""

    def __init__(
        self,
        store: ArticleListeningStore,
        profile_data_root: Path,
        *,
        freeze_voice,
        authorize,
        output=None,
        prepare_output=None,
        acquire_output=None,
        speech_services=None,
    ) -> None:
        self.store = store
        self.root = profile_data_root
        self._freeze_voice = freeze_voice
        self._authorize = authorize
        self._session = None
        self._phase = "empty"
        self._error = None
        self._closed = False
        self._paused_for_maintenance = False
        self._output = output
        self._prepare_output = prepare_output
        self._acquire_output = acquire_output
        self._speech_services = speech_services
        self._audio = None
        self._active = None
        self._holder = None
        self._holder_task = None
        self._holder_stop = None
        self._command = asyncio.Lock()
        self._write = asyncio.Lock()
        self._tasks = set()
        self._chunk_progress = (None, None)
        self._duration = None
        self._epoch = 0
        self._service_id = uuid4().hex
        self._admission = 0
        self._ownership_error = False
        self._retiring_task = None
        self._closing_task = None
        self._visible = True
        self._restored = False
        self._adopt_on_resume = True

    def _retain(self, coroutine):
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)

        def finished(task):
            self._tasks.discard(task)
            if not task.cancelled():
                task.exception()

        task.add_done_callback(finished)
        return task

    @property
    def _current(self):
        s = self._session
        return (
            (s.foreground or (s.queue[s.cursor] if s.cursor is not None else None))
            if s
            else None
        )

    async def _take_holder(self):
        if self._holder is not None:
            return
        loop = asyncio.get_running_loop()
        ready = loop.create_future()
        stop = Event()

        def hold():
            try:
                from .article_audio_assets import ArticleAudioAssets

                ArticleAudioAssets(self.store, profile_data_root=self.root).reconcile()
                with self.store.session_lease(self.root) as holder:
                    loop.call_soon_threadsafe(ready.set_result, holder)
                    stop.wait()
            except Exception as error:
                if not ready.done():
                    loop.call_soon_threadsafe(ready.set_exception, error)
                else:
                    raise

        task = asyncio.create_task(asyncio.to_thread(hold))
        try:
            self._holder = await asyncio.shield(ready)
        except BaseException:
            stop.set()
            await asyncio.shield(task)
            raise
        self._holder_task, self._holder_stop = task, stop
        self._ownership_error = False

    def _track_audio_result(self, s, a, updated, audio):
        entry = (
            (updated.foreground or updated.queue[updated.cursor]) if updated else None
        )
        if (
            updated is None
            or updated.incarnation != s.incarnation
            or (
                updated.playback_generation != s.playback_generation
                or entry.entry_id != a.entry.entry_id
                or entry.artifact != audio.generation
                or entry.item_id != audio.source_item_id
                or entry.bookmark is None
                or entry.bookmark.sha256 != audio.wav.sha256
            )
        ):
            raise ListeningConflict("listening_entry_changed")
        self._session = updated
        self._duration = audio.wav.content_frames
        a.entry = entry
        return entry

    async def _release_holder(self):
        if self._holder_task:
            self._holder_stop.set()
            await asyncio.shield(self._holder_task)
            self._holder_task = self._holder = self._holder_stop = None

    def _eligible(self, attempt):
        try:
            return (
                self._active is attempt
                and not attempt.revoked
                and not self._closed
                and not self._paused_for_maintenance
                and attempt.ticket.is_current()
                and self._authorize() == attempt.authority
            )
        except Exception:
            return False

    def _accept(self, expected_revision, *, session_wide=False):
        current = self._revision()
        return (
            not self._closed
            and not self._paused_for_maintenance
            and not (self._retiring_task and not self._retiring_task.done())
            and (
                expected_revision is None
                or isinstance(expected_revision, ListeningRevision)
                and expected_revision.service == current.service
                and expected_revision.epoch == current.epoch
                and expected_revision.incarnation == current.incarnation
                and (session_wide or expected_revision.entry_id == current.entry_id)
            )
        )

    def _revision(self):
        s = self._session
        return ListeningRevision(
            self._service_id,
            self._epoch,
            s.incarnation if s else None,
            s.revision if s else 0,
            self._current.entry_id if self._current else None,
        )

    def snapshot(self) -> ListeningSnapshot:
        """Return display state without I/O or provider access."""
        s = self._session
        revision = self._revision()
        if not self._visible:
            return ListeningSnapshot(
                phase="unavailable",
                revision=revision,
                error="local_listening_authority_required",
            )
        if s is None:
            return ListeningSnapshot(
                phase=self._phase, revision=revision, error=self._error
            )
        queued = s.queue[s.cursor] if s.cursor is not None else None
        current = s.foreground or queued
        playable = sum(entry.item_id is not None for entry in s.queue)
        point = current.bookmark if current is not None else None
        phase = self._phase
        if self._ownership_error or self._closed or self._paused_for_maintenance:
            actions = ()
        elif current is None or (current.item_id is None and not playable):
            actions = ("clear",)
        elif phase == "completed":
            actions = ("play_again", "clear")
        elif phase in ("preparing", "playing"):
            actions = ("pause", "stop", "skip", "clear")
        elif phase == "retiring":
            actions = ("stop",)
        else:
            actions = ("stop", "skip", "clear")
            if current.availability == "missing":
                actions += ("regenerate",)
            elif self._error:
                actions += ("retry",)
            else:
                actions += ("resume",)
        frame = point.frame if point else None
        if phase == "playing" and self._active and self._active.run:
            obs = self._active.run.position()
            if (
                obs
                and self._active.lease
                and obs.generation == self._active.lease.generation
            ):
                frame = obs.frame
        return ListeningSnapshot(
            self._phase,
            revision,
            current.title if current else None,
            queued.title if queued and s.foreground else None,
            s.cursor + 1 if s.cursor is not None else None,
            len(s.queue),
            playable,
            len(s.queue) - playable,
            stored_feed_text=current.stored_feed_text if current else None,
            elapsed_frames=frame,
            duration_frames=self._duration,
            sample_rate=point.sample_rate if point else None,
            completed_chunks=self._chunk_progress[0],
            total_chunks=self._chunk_progress[1],
            error=self._error,
            actions=frozenset(actions),
            unavailable_reasons=tuple(
                e.unavailable_reason
                for e in (*s.queue, *((s.foreground,) if s.foreground else ()))
                if e.unavailable_reason
            ),
        )

    async def restore(self) -> None:
        """Read the installed session without acquiring speech or starting output."""
        if self._closed:
            return
        async with self._command:
            self._restored = True
            try:
                self._authorize()
                self._visible = True
                if callable(self.store):
                    self.store = self.store()
                if self._active is not None:
                    return
                self._session = await asyncio.to_thread(self.store.load_session)
                self._phase = self._session.transport if self._session else "empty"
                self._error = None
                self._ownership_error = False
                if self._session is not None:
                    await self._take_holder()
                    self._session = await asyncio.to_thread(self.store.load_session)
                    self._phase = self._session.transport if self._session else "empty"
            except Exception as error:
                self._error = _reason(error)
                self._ownership_error = True
                self._visible = self._error != "local_listening_authority_required"

    async def _checkpoint(
        self,
        *,
        cursor,
        foreground="keep",
        resume=False,
        transport="paused",
        bookmarks=(),
    ):
        s = self._session
        update = ListeningCheckpoint(
            s.authority,
            s.incarnation,
            s.playback_generation,
            cursor,
            resume,
            transport,
            bookmarks,
            foreground,
        )
        self._session = await asyncio.to_thread(
            self.store.checkpoint,
            update,
            expected_revision=s.revision,
            lease=self._holder,
            playback_owner=self._active.terminal_owner if self._active else None,
        )

    async def _terminal(self, attempt, result):
        """Commit both roles before this run's exact output lease can release."""
        from tldw_chatbook.TTS.controlled_wav_playback import CompletionKind

        async with self._write:
            if self._active is not attempt or attempt.ended:
                return
            if result is not None and (
                attempt.lease is None or result.generation != attempt.lease.generation
            ):
                raise ListeningConflict("listening_generation_changed")
            s = self._session
            current = self._current
            if current is None or current.entry_id != attempt.entry.entry_id:
                raise ListeningConflict("listening_entry_changed")
            point = result.bookmark if result is not None else None
            try:
                await self._apply_terminal(attempt, result, s, current, point)
            except ListeningConflict as error:
                if attempt.transition == "stop" or self._session is not s:
                    raise
                # Live admission can expire while the original process is reaping.
                # Only the original held session may retain its last observation.
                self._session = await asyncio.to_thread(
                    self.store.pause_held, s, point, owner=attempt.terminal_owner
                )
                attempt.advance = False
                attempt.rejection = self._error = _reason(error)
            delayed_retirement = self._phase == "retiring"
            self._phase = self._session.transport
            if self._current is None or self._current.entry_id != current.entry_id:
                self._duration = None
            if result and result.kind in (
                CompletionKind.UNCERTAIN,
                CompletionKind.FAILED,
            ):
                self._error = _reason(ValueError(result.reason))
            attempt.ended = True
            self.store.finish_playback(attempt.terminal_owner)
            if delayed_retirement:
                self._active = None

    async def _advance_queue(self, s, marks=()):
        after = len(s.queue) if s.cursor is None else s.cursor + 1
        next_slot = next(
            (i for i in range(after, len(s.queue)) if s.queue[i].item_id is not None),
            None,
        )
        await self._checkpoint(
            cursor=s.cursor if next_slot is None else next_slot,
            transport="completed" if next_slot is None else "paused",
            bookmarks=marks,
        )
        return next_slot is not None

    async def _apply_terminal(self, attempt, result, s, current, point):
        from tldw_chatbook.TTS.controlled_wav_playback import CompletionKind

        marks = ((current.entry_id, point),) if point else ()
        transition = attempt.transition
        if transition == "replace":
            self._session = await asyncio.to_thread(
                self.store.replace_session,
                attempt.candidate,
                expected_revision=s.revision,
                lease=self._holder,
            )
        elif transition == "interrupt":
            candidate = attempt.candidate
            if (
                candidate.image != attempt.terminal_owner.image
                or candidate.authority != s.authority
                or (candidate.observed_incarnation, candidate.observed_revision)
                != (s.incarnation, s.revision)
            ):
                raise ListeningConflict("listening_entry_authority_changed")
            await self._checkpoint(
                cursor=s.cursor,
                foreground=candidate.foreground,
                resume=s.resume_queue
                if s.foreground
                else bool(s.queue and self._phase == "playing"),
                bookmarks=marks,
            )
        elif transition == "skip" or (
            transition is None
            and result is not None
            and result.kind is CompletionKind.NATURAL
            and self._eligible(attempt)
        ):
            if s.foreground and s.queue:
                await self._checkpoint(
                    cursor=s.cursor, foreground="remove", bookmarks=marks
                )
                attempt.advance = s.resume_queue
            else:
                attempt.advance = await self._advance_queue(s, marks)
        else:
            self._session = await asyncio.to_thread(
                self.store.pause_held, s, point, owner=attempt.terminal_owner
            )

    async def _retire_output(self, attempt):
        """Retain even an unstarted player's pins and exact grant until settlement."""
        try:
            if await attempt.run.retire():
                if attempt.lease is not None:
                    # A grant returned after cancellation never reached run.start.
                    # A started run has already invoked its before-release callback.
                    if not attempt.ended:
                        await self._terminal(attempt, None)
                    attempt.lease.release()
                return True
        except Exception:
            pass
        if attempt.lease is not None:
            attempt.lease.block()
        if self._active is attempt:
            self._phase, self._error = "retiring", "player_retirement_unconfirmed"
        return False

    async def _halt(self, transition="stop", candidate=None):
        attempt = self._active
        if attempt is None:
            return
        attempt.transition = transition
        attempt.candidate = candidate
        attempt.revoked = True
        attempt.ticket.cancel()
        committing = self._audio is not None and not self._audio.revoke()
        if (
            not attempt.run
            and attempt.task
            and (committing or attempt.preparing_output)
        ):
            # A selected SQLite commit must publish its result, and an in-flight
            # output preparation must hand back its pins before this owner leaves.
            await asyncio.shield(attempt.task)
        if attempt.ambiguous_readiness_pin:
            raise ListeningBusy("pin_retirement_unconfirmed")
        if attempt.run and not attempt.ended:
            if not await self._retire_output(attempt):
                raise ListeningBusy("player_retirement_unconfirmed")
        if not attempt.ended:
            await self._terminal(attempt, None)
        self._active = None
        if attempt.rejection:
            raise ListeningConflict(attempt.rejection)

    async def _launch(self, *, regenerate=False, ticket=None):
        if self._active is not None or self._current is None:
            return
        if self._current.item_id is None:
            s = self._session
            slot = next(
                (i for i in range(s.cursor or 0, len(s.queue)) if s.queue[i].item_id),
                None,
            )
            if s.foreground is not None or slot is None:
                self._error = "no_playable_articles"
                return
            await self._checkpoint(cursor=slot)
        s = self._session
        entry, terminal_owner = await asyncio.to_thread(
            self.store.open_playback, s, self._current.entry_id, lease=self._holder
        )
        if self._authorize() != s.authority:
            raise ListeningConflict("listening_authority_changed")
        owner = self._output() if callable(self._output) else self._output
        ticket = ticket or owner.begin_intent()
        attempt = self._active = _Attempt(
            entry, s.authority, terminal_owner, ticket, self._epoch
        )
        self._phase, self._error, self._duration = "preparing", None, None
        self._chunk_progress = (None, None)
        attempt.task = self._retain(self._play(attempt, s, regenerate))

    async def _check_output_readiness(self):
        from tldw_chatbook.TTS.controlled_wav_profile import controlled_wav_readiness

        a = self._active
        if a is None or not self._eligible(a):
            return "article_audio_retired"
        a.preparing_output = True
        try:
            reason = await controlled_wav_readiness(custody=self._holder)
            a.ambiguous_readiness_pin = reason == "pin_retirement_unconfirmed"
            return reason
        finally:
            a.preparing_output = False

    async def _play(self, a, s, regenerate):
        from tldw_chatbook.TTS.controlled_wav_playback import (
            CompletionKind,
            ControlledPreparationUnconfirmed,
            ControlledWavPlayback,
        )
        from .article_listening_audio import ArticleListeningAudio

        try:
            if not self._eligible(a):
                return
            if self._audio is None:
                self._audio = ArticleListeningAudio(
                    self.store,
                    self.root,
                    tts_service=None,
                    readiness=self._check_output_readiness,
                    lease=self._holder,
                    speech_services=self._speech_services,
                )

            def progress(done, total):
                if self._eligible(a):
                    self._chunk_progress = done, total

            self._audio.progress = progress
            audio = await self._audio.prepare(
                a.entry.entry_id,
                expected_revision=s.revision,
                regenerate=regenerate,
                permission=lambda: self._eligible(a),
            )
            if self._active is not a or a.ended:
                return
            async with self._write:
                updated = self._audio.attached_session or s
                if self._active is not a or a.ended:
                    return
                entry = self._track_audio_result(s, a, updated, audio)
                if not self._eligible(a):
                    return
            before = lambda result: self._terminal(a, result)
            a.preparing_output = True
            if self._prepare_output:
                a.run = await self._prepare_output(
                    audio.wav, entry.bookmark, before_release=before
                )
            else:
                await asyncio.to_thread(self._holder.recover_predecessor)
                a.run = await ControlledWavPlayback.prepare(
                    audio.wav,
                    entry.bookmark,
                    before_release=before,
                    custody=self._holder,
                )
            if a.run.already_at_end:
                await self._terminal(a, None)
                self._error = "article_at_end"
                return
            if not self._eligible(a):
                await self._retire_output(a)
                return
            a.lease = await self._acquire_output(
                a.ticket, a.run, lambda: self._eligible(a)
            )
            if a.lease is None or not self._eligible(a):
                await self._retire_output(a)
                return
            await a.run.start(a.lease)
            if self._eligible(a) and not a.ended:
                self._phase = "playing"
            a.next_checkpoint = time.monotonic() + 1
            next_write = time.monotonic() + 1
            completion = a.run.completion
            while not completion.done():
                await asyncio.wait(
                    (completion,), timeout=max(0, next_write - time.monotonic())
                )
                if completion.done():
                    break
                if not self._eligible(a):
                    await self._retire_output(a)
                    break
                await self._save_progress(a, time.monotonic())
                next_write = time.monotonic() + 1
            result = completion.result()
            if result.kind is CompletionKind.UNCERTAIN and not a.ended:
                self._phase = "retiring"
                self._error = "player_retirement_unconfirmed"
                return
            if not a.ended:
                await self._terminal(a, result)
        except Exception as error:
            if self._active is a:
                self._error = _reason(error)
                if isinstance(error, ControlledPreparationUnconfirmed):
                    a.run = error.run
                if a.ambiguous_readiness_pin or isinstance(
                    error, ControlledPreparationUnconfirmed
                ):
                    self._phase = "retiring"
                    return
                if self._error in (
                    "article_audio_missing",
                    "article_audio_corrupt",
                ) and self._eligible(a):
                    updated = await asyncio.to_thread(self.store.load_session)

                    def missing(entry):
                        return (
                            replace(entry, availability="missing")
                            if entry and entry.artifact == a.entry.artifact
                            else entry
                        )

                    expected = replace(
                        s,
                        revision=s.revision + 1,
                        queue=tuple(map(missing, s.queue)),
                        foreground=missing(s.foreground),
                    )
                    if updated == expected:
                        self._session = updated
                if a.run and not a.ended:
                    if not await self._retire_output(a):
                        return
        finally:
            if self._active is a and self._phase != "retiring" and not a.transition:
                if not a.ended:
                    self._phase = "paused"
                    self.store.finish_playback(a.terminal_owner)
                self._active = None
                if a.advance and not a.revoked:
                    self._retain(self._continue(a, self._session))

    async def _continue(self, a, session):
        async with self._command:
            if (
                not self._closed
                and not self._paused_for_maintenance
                and self._epoch == a.epoch
                and a.ticket.is_current()
                and self._active is None
                and self._session is session
                and self._authorize() == a.authority
            ):
                await self._launch()

    async def _save_progress(self, a, now):
        async with self._write:
            if (
                not self._eligible(a)
                or a.ended
                or self._phase != "playing"
                or now < a.next_checkpoint
            ):
                return
            obs = a.run.position()
            point = self._current.bookmark
            if (
                obs is not None
                and obs.generation == a.lease.generation
                and obs.frame > point.frame
            ):
                await self._checkpoint(
                    cursor=self._session.cursor,
                    resume=self._session.resume_queue,
                    transport="playing",
                    bookmarks=((a.entry.entry_id, replace(point, frame=obs.frame)),),
                )
                a.next_checkpoint = now + 1

    async def listen(
        self,
        item_id: str,
        *,
        runtime_backend: str,
        expected_revision: ListeningRevision | None = None,
    ) -> None:
        """Start a local article after validating its source namespace."""
        if (
            runtime_backend != "local"
            or type(item_id) is not str
            or not re.fullmatch(r"local:watchlist_item:[1-9][0-9]{0,18}", item_id)
        ):
            self._error = "local_listening_authority_required"
            return
        local_id = int(item_id.rsplit(":", 1)[-1])
        await self._request_admission(
            local_id, brief=False, expected_revision=expected_revision
        )

    async def play_brief(
        self,
        briefing_id: int,
        *,
        runtime_backend: str,
        replace: bool = False,
        expected_revision: ListeningRevision | None = None,
    ) -> None:
        if (
            runtime_backend != "local"
            or type(briefing_id) is not int
            or not 0 < briefing_id < 2**63
        ):
            self._error = "local_listening_authority_required"
            return
        await self._request_admission(
            briefing_id,
            brief=True,
            replace_session=replace,
            expected_revision=expected_revision,
        )

    async def _request_admission(self, ident, **options):
        request = _Admission()
        try:
            await asyncio.shield(self._retain(self._admit(request, ident, **options)))
        except asyncio.CancelledError:
            request.withdraw()
            raise

    async def _prepare_candidate(self, ident, brief, authority, voice):
        from .article_speech import CONVERTER, prepare_saved_text

        candidate = await asyncio.to_thread(
            self.store.prepare_brief if brief else self.store.prepare_item,
            ident,
            authority=authority,
            voice=voice,
            converter=CONVERTER,
            prepare_text=prepare_saved_text,
        )
        if not any(e.item_id for e in candidate.queue) and (
            not candidate.foreground or candidate.foreground.item_id is None
        ):
            raise ListeningConflict("no_playable_articles")
        return candidate

    async def preview_brief(
        self, briefing_id: int, *, runtime_backend: str
    ) -> BriefListeningPreview:
        """Return counts for the confirmation display without installing the queue."""
        if not self._accept(None):
            raise ListeningBusy("listening_busy")
        return await asyncio.shield(
            self._retain(self._preview(briefing_id, runtime_backend))
        )

    async def _preview(self, briefing_id, runtime_backend):
        if (
            runtime_backend != "local"
            or type(briefing_id) is not int
            or not 0 < briefing_id < 2**63
        ):
            raise ListeningConflict("local_listening_authority_required")
        authority = self._authorize()
        revision = self._revision()
        if callable(self.store):
            self.store = self.store()
        from .article_speech import CONVERTER, prepare_saved_text

        installed = self._session or await asyncio.to_thread(self.store.load_session)
        if installed and installed.brief_id == briefing_id:
            if self._authorize() != authority or not self._accept(
                revision, session_wide=True
            ):
                raise ListeningConflict("listening_authority_changed")
            if installed.transport == "completed":
                action = "play_again"
            elif installed.foreground:
                action = "resume_foreground"
            else:
                action = "resume"
            reasons = tuple(
                e.unavailable_reason for e in installed.queue if e.unavailable_reason
            )
            return BriefListeningPreview(
                len(installed.queue) - len(reasons), reasons, action
            )
        voice = installed.voice if installed else await self._freeze_voice()
        candidate = await asyncio.to_thread(
            self.store.prepare_brief,
            briefing_id,
            authority=authority,
            voice=voice,
            converter=CONVERTER,
            prepare_text=prepare_saved_text,
        )
        if (
            not self._accept(revision, session_wide=True)
            or self._authorize() != authority
        ):
            raise ListeningConflict("listening_authority_changed")
        reasons = tuple(
            e.unavailable_reason for e in candidate.queue if e.unavailable_reason
        )
        return BriefListeningPreview(
            len(candidate.queue) - len(reasons),
            reasons,
            "replace" if installed else "play",
        )

    async def _admit(
        self, request, ident, *, brief, expected_revision, replace_session=False
    ):
        if not self._accept(expected_revision):
            return
        ticket = None
        epoch = None
        try:
            authority = self._authorize()
            async with self._command:
                if request.cancelled or not self._accept(expected_revision):
                    return
                if callable(self.store):
                    self.store = self.store()
                await self._take_holder()
                s = self._session
                if s is None:
                    s = self._session = await asyncio.to_thread(self.store.load_session)
                if request.cancelled:
                    if s is None:
                        await self._release_holder()
                    return
                if brief and s and s.brief_id == ident:
                    return
                if brief and s and not replace_session:
                    return
                if (
                    not brief
                    and self._active
                    and self._current
                    and self._current.item_id == ident
                ):
                    return
                self._admission += 1
                epoch = self._epoch, self._admission
                owner = self._output() if callable(self._output) else self._output
                ticket = request.ticket = owner.begin_intent(tentative=True)
            # The previous owner can play, pause or stop while this request is prepared.
            voice = await self._freeze_voice()
            if (
                request.cancelled
                or epoch != (self._epoch, self._admission)
                or self._authorize() != authority
            ):
                return
            candidate = await self._prepare_candidate(ident, brief, authority, voice)
            async with self._command:
                if (
                    epoch != (self._epoch, self._admission)
                    or not self._accept(None)
                    or self._authorize() != authority
                ):
                    return
                async with self._write:
                    current = self._session
                    if current and candidate.observed_revision != current.revision:
                        if s is None or current.incarnation != s.incarnation:
                            raise ListeningConflict("listening_revision_changed")
                        candidate = await self._prepare_candidate(
                            ident, brief, authority, voice
                        )
                    if (
                        epoch != (self._epoch, self._admission)
                        or self._authorize() != authority
                        or not ticket.promote()
                    ):
                        return
                    request.selected = True
                    if self._active:
                        self._active.revoked = True
                if (
                    not brief
                    and current
                    and current.queue
                    and current.transport != "completed"
                ):
                    if self._active:
                        await self._halt("interrupt", candidate)
                    else:
                        if current.authority != authority:
                            current = self._session = await asyncio.to_thread(
                                self.store.adopt_session, current, lease=self._holder
                            )
                            candidate = await self._prepare_candidate(
                                ident, brief, authority, voice
                            )
                        await self._checkpoint(
                            cursor=current.cursor,
                            foreground=candidate.foreground,
                            resume=current.resume_queue
                            if current.foreground
                            else False,
                        )
                elif self._active:
                    await self._halt("replace", candidate)
                else:
                    self._session = await asyncio.to_thread(
                        self.store.replace_session,
                        candidate,
                        expected_revision=current.revision if current else None,
                        lease=self._holder,
                    )
                if epoch == (self._epoch, self._admission) and ticket.is_current():
                    self._epoch += 1
                    await self._launch(ticket=ticket)
                    ticket = None
        except Exception as error:
            if not request.cancelled and (epoch is None or epoch[1] == self._admission):
                self._error = _reason(error)
        finally:
            if ticket is not None:
                ticket.cancel()
            if epoch is not None and self._session is None:
                async with self._command:
                    if epoch[1] == self._admission and self._session is None:
                        await self._release_holder()

    async def resume(
        self, *, expected_revision: ListeningRevision | None = None
    ) -> None:
        await self._resume(False, False, expected_revision)

    async def retry(
        self,
        *,
        regenerate: bool = False,
        expected_revision: ListeningRevision | None = None,
    ) -> None:
        await self._resume(False, regenerate, expected_revision)

    async def play_again(
        self, *, expected_revision: ListeningRevision | None = None
    ) -> None:
        await self._resume(True, False, expected_revision)

    @_serialized_command
    async def _resume(self, replay, regenerate, revision):
        if not self._accept(revision) or self._active or not self._session:
            return
        try:
            authority = self._authorize()
            await self._take_holder()
            if self._session.transport == "completed" and not replay:
                return
            if self._adopt_on_resume or self._session.authority != authority or replay:
                self._session = await asyncio.to_thread(
                    self.store.adopt_session,
                    self._session,
                    lease=self._holder,
                    replay=replay,
                )
                self._adopt_on_resume = False
            self._epoch += 1
            await self._launch(regenerate=regenerate)
        except Exception as error:
            self._error = _reason(error)

    async def pause(
        self, *, expected_revision: ListeningRevision | None = None
    ) -> None:
        await self.stop(expected_revision=expected_revision)

    @_serialized_command
    async def stop(self, *, expected_revision: ListeningRevision | None = None) -> None:
        if self._accept(expected_revision, session_wide=True):
            self._epoch += 1
            await self._halt()

    @_serialized_command
    async def skip(self, *, expected_revision: ListeningRevision | None = None) -> None:
        if not self._accept(expected_revision) or not self._session:
            return
        self._epoch += 1
        self._authorize()
        await self._take_holder()
        if self._active:
            a = self._active
            await self._halt("skip")
            if a.advance:
                await self._launch()
        else:
            s = self._session
            if s.foreground and s.queue:
                await self._checkpoint(cursor=s.cursor, foreground="remove")
                if s.resume_queue:
                    await self._launch()
            elif await self._advance_queue(s):
                await self._launch()
            else:
                self._phase = "completed"

    @_serialized_command
    async def clear(
        self, *, expected_revision: ListeningRevision | None = None
    ) -> None:
        if not self._accept(expected_revision, session_wide=True) or not self._session:
            return
        self._epoch += 1
        self._authorize()
        await self._take_holder()
        await self._halt()
        await asyncio.to_thread(
            self.store.clear,
            expected_revision=self._session.revision,
            lease=self._holder,
        )
        self._session, self._phase, self._error = None, "empty", None
        if self._audio:
            await self._audio.settle()
            self._audio = None
        await self._release_holder()

    async def close(self) -> None:
        """Close listening intake for this app instance."""
        self.begin_close()
        if self._closing_task is None:
            self._closing_task = asyncio.create_task(self._close())
        await asyncio.shield(self._closing_task)

    async def _close(self):
        if self._retiring_task:
            await asyncio.shield(self._retiring_task)
        await self._quiesce()

    def begin_close(self) -> None:
        self._closed = True
        self.revoke()

    def revoke(self) -> None:
        """Synchronously withdraw eligibility before cleanup can yield."""
        self._epoch += 1
        if self._active:
            self._active.revoked = True
            self._active.ticket.cancel()
            if self._active.lease:
                self._active.lease.revoke()
        if self._audio:
            self._audio.revoke()

    async def _quiesce(self):
        async with self._command:
            await self._halt()
        if self._audio:
            await self._audio.settle()
            self._audio = None
        pending = [
            t
            for t in self._tasks
            if t not in (asyncio.current_task(), self._retiring_task)
        ]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        await self._release_holder()

    def policy_changed(self) -> None:
        """Fence old callbacks in the runtime policy commit's owner-loop turn."""
        self.revoke()
        self._adopt_on_resume = True
        try:
            self._authorize()
            self._visible = True
        except Exception:
            self._visible = False
        if (
            self._holder_task is not None
            or self._active is not None
            or self._command.locked()
            or self._restored
        ):
            if self._retiring_task is None or self._retiring_task.done():
                self._retiring_task = self._retain(self._settle_policy())

    async def _settle_policy(self):
        await self._quiesce()
        if (
            self._restored
            and self._visible
            and not self._closed
            and not self._paused_for_maintenance
        ):
            await self.restore()

    def _maintenance_close_admission(self):
        self._paused_for_maintenance = True
        self.policy_changed()

    async def _maintenance_drain(self, deadline):
        task = self._retiring_task
        if task:
            done, _ = await asyncio.wait(
                (task,), timeout=max(0, deadline - time.monotonic())
            )
            if not done:
                return False
            task.result()
        return self._holder is None

    def _maintenance_resume(self):
        self._paused_for_maintenance = False

    async def delete_source(self, source_id, *, database=None):
        return await self._delete("source", source_id, database=database)

    async def delete_article(self, item_id):
        return await self._delete("item", item_id)

    async def delete_brief(self, brief_id):
        return await self._delete("brief", brief_id)

    async def delete_watchlist(self, watchlist_id, bundle):
        return await self._delete("watchlist", watchlist_id, bundle, database=bundle.db)

    @_serialized_command
    async def _delete(self, kind, ident, bundle=None, *, database=None):
        if not self._accept(None):
            raise ListeningBusy("listening_busy")
        authority = self._authorize()
        if callable(self.store):
            self.store = self.store()
        if database is not None and database is not self.store.db:
            raise ListeningConflict("listening_database_changed")
        await self._take_holder()
        db = self.store.db
        if self._session is None:
            self._session = await asyncio.to_thread(self.store.load_session)
        s = self._session
        entries = (*s.queue, *((s.foreground,) if s.foreground else ())) if s else ()
        affected = any(
            (kind == "source" and e.source_id == ident)
            or (kind == "item" and e.item_id == ident)
            for e in entries
        )
        if s and s.brief_id is not None:
            if kind == "brief":
                affected = s.brief_id == ident
            elif kind == "watchlist":

                def owns_brief():
                    with db.artifact_read_snapshot() as c:
                        return (
                            c.execute(
                                "SELECT 1 FROM briefings WHERE id=? AND watchlist_id=?",
                                (s.brief_id, ident),
                            ).fetchone()
                            is not None
                        )

                affected = await asyncio.to_thread(owns_brief)
        if affected:
            await self._halt()
            if self._audio:
                await self._audio.settle()
        method = (
            bundle.delete
            if bundle
            else {
                "source": db.delete_subscription,
                "item": db.delete_article,
                "brief": db.delete_brief,
            }[kind]
        )

        def remove():
            # Acquire the SQLite write lock before selecting a destructive
            # action. An epoch change before publication rolls it back.
            with db.destructive_transaction(lease=self._holder):
                if self._authorize() != authority:
                    raise ListeningConflict("listening_authority_changed")
                result = method(ident, lease=self._holder)
                if self._authorize() != authority:
                    raise ListeningConflict("listening_authority_changed")
                return result

        result = await asyncio.to_thread(remove)
        if affected:
            self._session = await asyncio.to_thread(self.store.load_session)
            self._phase = self._session.transport if self._session else "empty"
            from .article_audio_assets import ArticleAudioAssets

            assets = ArticleAudioAssets(
                self.store, profile_data_root=self.root, lease=self._holder
            )
            await asyncio.to_thread(assets.reconcile)
        if self._session is None:
            await self._release_holder()
        return result

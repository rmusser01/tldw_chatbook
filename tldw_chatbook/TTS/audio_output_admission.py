"""UI-facing admission and explicit confirmation of external playback."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from loguru import logger

from tldw_chatbook.TTS.audio_output_ownership import (
    AudioOutputIntent,
    AudioOutputLease,
    OutputBlocked,
    OutputClosed,
    OutputHandoffRequired,
    OutputSuperseded,
)
from tldw_chatbook.Widgets.confirmation_dialog import (
    PROMPT_WATCH_INTERVAL_SECONDS,
    ConfirmationDialog,
)


def retain_retired_handoff_file(app: Any, release: Callable[[], None]) -> None:
    """Retain a locally retired, delegated file until its deadline or app exit."""
    from tldw_chatbook.TTS.audio_player import HAND_OFF_RETENTION_SECONDS

    if vars(app).get("_audio_output_closed"):
        release()
        return
    pending = vars(app).setdefault("_retired_handoff_files", set())
    loop = asyncio.get_running_loop()
    deadline = loop.time() + HAND_OFF_RETENTION_SECONDS

    def finish() -> None:
        if finish in pending:
            pending.remove(finish)
            timer.cancel()
            release()

    def expire() -> None:
        nonlocal timer
        if loop.time() < deadline:
            timer = loop.call_at(deadline, expire)
        else:
            finish()

    pending.add(finish)
    timer = loop.call_at(deadline, expire)


class _PlaybackConfirmation(ConfirmationDialog):
    def __init__(
        self,
        owner_app: Any,
        ticket: AudioOutputIntent,
        epoch: object,
        eligible: Callable[[], bool],
    ) -> None:
        super().__init__(
            title="External playback",
            message="Stop playback in the external player, then confirm to play audio here.",
            confirm_label="Playback stopped",
            cancel_label="Cancel",
        )
        self.owner_app = owner_app
        self.epoch = epoch
        self.eligible = eligible
        self.ticket = ticket
        self.answer = asyncio.get_running_loop().create_future()

    def current(self) -> bool:
        return (
            vars(self.owner_app).get("_audio_output_prompt") is self
            and self.owner_app.audio_output.handoff_epoch is self.epoch
            and self.eligible()
        )

    def resolve(self, answer: bool | None) -> None:
        if not self.answer.done():
            self.answer.set_result(answer is True)

    def withdraw(self) -> None:
        self.resolve(False)
        if self.is_mounted and self.app.screen is self:
            self.dismiss(False)

    def _check_request(self) -> None:
        if self.answer.done() or not self.current():
            self.withdraw()

    def on_mount(self) -> None:
        self.set_interval(PROMPT_WATCH_INTERVAL_SECONDS, self._check_request)

    def on_screen_resume(self) -> None:
        self.call_after_refresh(self._check_request)

    def on_unmount(self) -> None:
        self.resolve(self.result)


def is_audio_confirmation_for(app: Any, ticket: AudioOutputIntent) -> bool:
    """Identify the active confirmation covering this pending playback request."""
    prompt = vars(app).get("_audio_output_prompt")
    return (
        isinstance(prompt, _PlaybackConfirmation)
        and prompt.ticket is ticket
        and ticket.is_current()
        and app.audio_output.handoff_epoch is prompt.epoch
        and app.screen is prompt
    )


async def acquire_audio_output(
    app: Any,
    ticket: AudioOutputIntent,
    owner: str,
    run: Any,
    *,
    current: Callable[[], bool],
    external: bool = False,
) -> AudioOutputLease | None:
    """Admit this prepared run, asking about prior external playback if needed.

    Call from a worker or independent task so the app can dispatch modal input.
    The caller owns unstarted run cleanup and checks authority again at start.
    """

    def eligible() -> bool:
        try:
            return ticket.is_current() and current()
        except Exception as error:
            logger.debug(
                "Audio request owner unavailable type={}", type(error).__name__
            )
            return False

    if not eligible():
        return None
    host = app.audio_output
    previous = vars(app).get("_audio_output_prompt")
    if previous is not None:
        previous.withdraw()
    while eligible():
        try:
            lease = await host.acquire(
                owner, run.retire, intent=ticket, external=external
            )
        except (OutputClosed, OutputSuperseded):
            return None
        except OutputBlocked as error:
            app.notify(
                f"Cannot start audio until {error.owner} stops.", severity="warning"
            )
            return None
        except OutputHandoffRequired as error:
            epoch = error.epoch
        else:
            if eligible() and lease.authorizes(run.retire):
                return lease
            await host.stop(lease)
            return None

        if not eligible() or host.handoff_epoch is not epoch:
            return None

        prompt = _PlaybackConfirmation(app, ticket, epoch, eligible)
        app._audio_output_prompt = prompt
        try:
            await app.push_screen(prompt, callback=prompt.resolve)
            if not prompt.current():
                return None
            accepted = await prompt.answer
            if (
                not accepted
                or not prompt.current()
                or not host.acknowledge_handoff(ticket, epoch)
            ):
                return None
        except Exception as error:
            logger.warning("Audio confirmation failed type={}", type(error).__name__)
            return None
        finally:
            if vars(app).get("_audio_output_prompt") is prompt:
                app._audio_output_prompt = None
            prompt.withdraw()
    return None

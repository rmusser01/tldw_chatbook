"""Runs character probe conversations.

Every provider call goes through ``asyncio.to_thread``: the app's chat gateway
(``Chat_Functions.chat_api_call``) is a plain synchronous ``def``, and calling
it from the event loop would block the whole TUI. Conversations run
concurrently under the bench's ``concurrency`` setting; turns WITHIN a
conversation are strictly sequential, because turn N needs turn N-1's reply.

Cancelling stops SCHEDULING further turns and conversations. It cannot abort a
turn already in flight -- ``to_thread`` survives task cancellation -- so an
in-flight provider call always runs to completion and is recorded. A
conversation whose first turn has not yet started when cancellation lands
still appears in the result list, marked with an empty ``turns`` tuple and a
"Cancelled" ``error``, rather than being silently dropped: the grid's shape
(one entry per card x probe x target x sample) stays predictable regardless
of when a run was stopped.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Mapping, Optional, Sequence

from loguru import logger

from .models import (
    CardSnapshot,
    CharacterProbeConfig,
    Conversation,
    ConversationTurn,
    ProbeSet,
)
from .prompt import build_messages
from .targets import ResolvedTarget, resolve_targets

#: The injected provider callable. Synchronous by contract -- see the module
#: docstring for why it must never be awaited directly.
#: Shape: ``chat_fn(messages, model, temperature, max_tokens, seed) -> str``.
ChatCallable = Callable[..., str]

#: Optional progress callback, fired as ``progress(done, total)`` once per
#: conversation that finishes (successfully, with an error, or cancelled).
ProgressCallback = Callable[[int, int], None]


def _sample_seed(bench_seed: Optional[int], sample_index: int) -> Optional[int]:
    """The seed for one sample of a cell.

    A non-negative bench seed is offset by the sample index, which is the
    whole point of seeding a multi-sample run: one fixed seed would return
    N identical answers, tripling review volume for zero information, so a
    seeded run must be reproducible AND have samples that genuinely differ.

    A NEGATIVE seed is a sentinel, not an arithmetic value: llama.cpp reads
    it as "pick a random seed", and ``storage.load_character_bench``
    deliberately accepts one for that reason. Offsetting it is worse than
    useless -- ``-1`` becomes ``-1, 0, 1, ...``, so sample 0 is randomly
    seeded while every later sample gets a *deterministic* seed the user
    never chose. The offset exists precisely so samples differ; applied to
    the random sentinel it does the exact opposite, quietly collapsing the
    variance the samples were taken to reveal, and the run still looks like
    it worked. A negative seed is therefore passed through unchanged for
    every sample.

    ``0`` is NOT a sentinel: it is a real, explicitly-chosen seed (the same
    falsy-but-real case ``storage._stored_int_field`` exists for), so it
    offsets normally.

    Args:
        bench_seed: ``CharacterProbeConfig.seed``.
        sample_index: Zero-based sample number within its cell.

    Returns:
        Optional[int]: ``None`` when unseeded, the seed unchanged when it is
        the negative random sentinel, otherwise ``bench_seed +
        sample_index``.
    """
    if bench_seed is None:
        return None
    if bench_seed < 0:
        return bench_seed
    return bench_seed + sample_index


class CancelToken:
    """Cancels a whole run; see the module docstring for what that means."""

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        """Request that no further turns or conversations be scheduled."""
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        """Whether :meth:`cancel` has been called."""
        return self._cancelled


class CharacterProbeRunner:
    """Runs a bench's full grid of conversations.

    The grid is every combination of the resolved ``cards``, the probes in
    ``probe_set``, the requested ``targets``, and ``config.samples_per_cell``
    -- one :class:`~tldw_chatbook.Evals.character_probe.models.Conversation`
    per combination.
    """

    def __init__(
        self, chat_fn: ChatCallable, cancel_token: Optional[CancelToken] = None
    ) -> None:
        """Create a runner bound to one provider callable.

        Args:
            chat_fn: The synchronous provider callable; see :data:`ChatCallable`.
                Always dispatched through ``asyncio.to_thread``, never awaited
                or called directly on the event loop.
            cancel_token: Shared cancellation flag. A fresh, never-cancelled
                :class:`CancelToken` is created when omitted, so a caller who
                does not need cancellation can ignore this entirely.
        """
        self._chat = chat_fn
        self._cancel = cancel_token or CancelToken()

    async def _run_conversation(
        self,
        card: CardSnapshot,
        probe_index: int,
        turns: Sequence[str],
        target: ResolvedTarget,
        sample_index: int,
        config: CharacterProbeConfig,
    ) -> Conversation:
        """Run one card/probe/target/sample cell's turns, in order.

        ``target`` arrives already resolved (see :mod:`.targets`): its
        steering has been read out of the row's ``config`` JSON, which is
        the only place an ``eval_models`` row ever keeps it.
        """
        steering = target.steering
        seed = _sample_seed(config.seed, sample_index)
        collected: list[ConversationTurn] = []
        replies: list[str] = []
        error = ""
        for turn_index, user_turn in enumerate(turns):
            if self._cancel.is_cancelled:
                error = "Cancelled before this turn ran."
                break
            messages = build_messages(card, steering, turns, replies)
            try:
                reply = await asyncio.to_thread(
                    self._chat,
                    messages=messages,
                    model=target.model_id,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    seed=seed,
                )
            except Exception as exc:  # noqa: BLE001 -- a provider failure ends only this conversation
                error = f"Turn {turn_index + 1} failed: {exc}"
                break
            reply_text = str(reply or "")
            replies.append(reply_text)
            collected.append(ConversationTurn(user=user_turn, reply=reply_text))
        return Conversation(
            card_id=card.id,
            probe_index=probe_index,
            sample_index=sample_index,
            target_id=target.id,
            turns=tuple(collected),
            error=error,
        )

    async def run(
        self,
        cards: Sequence[CardSnapshot],
        probe_set: ProbeSet,
        targets: Sequence[Mapping[str, Any]],
        config: CharacterProbeConfig,
        progress: Optional[ProgressCallback] = None,
    ) -> list[Conversation]:
        """Run every (card x probe x target x sample) conversation.

        Args:
            cards: Snapshotted cards, already resolved.
            probe_set: The scripts to run.
            targets: ``eval_models`` rows, as ``EvalsDB.get_model``/
                ``list_models`` return them. Each is validated and its
                steering read out of its ``config`` JSON by
                :func:`~tldw_chatbook.Evals.character_probe.targets.resolve_targets`
                BEFORE the first provider call, so a malformed or
                prefix-steered row costs nothing rather than surfacing
                mid-grid. That steering composes ahead of the card's own
                system prompt.
            config: The bench, supplying concurrency, samples, seed, and
                sampling parameters.
            progress: Optional ``(done, total)`` callback fired once per
                conversation as it finishes, regardless of outcome. A
                callback that RAISES is logged and swallowed: an observer
                must never be able to destroy the run it is watching (see
                ``_guarded`` below).

        Returns:
            list[Conversation]: Every conversation, including failed,
            partial, and cancelled ones -- a failed or cancelled cell is
            still evidence and stays reviewable. Length is always
            ``len(cards) * len(probe_set.probes) * len(targets) *
            config.samples_per_cell``.

        Raises:
            ValueError: Propagated from ``resolve_targets`` for an empty
                target list, duplicate target ids, or any row that is
                malformed or prefix-steered.
        """
        resolved_targets = resolve_targets(targets)
        jobs = [
            (card, probe_index, probe.turns, target, sample_index)
            for card in cards
            for probe_index, probe in enumerate(probe_set.probes)
            for target in resolved_targets
            for sample_index in range(config.samples_per_cell)
        ]
        semaphore = asyncio.Semaphore(config.concurrency)
        done = 0
        total = len(jobs)

        async def _guarded(job) -> Conversation:
            nonlocal done
            card, probe_index, turns, target, sample_index = job
            async with semaphore:
                conversation = await self._run_conversation(
                    card, probe_index, turns, target, sample_index, config
                )
            done += 1
            if progress is not None:
                try:
                    progress(done, total)
                except Exception:  # noqa: BLE001 -- see below
                    # A progress callback is an OBSERVER. Left unguarded it
                    # runs inside an asyncio.gather child, so one raising
                    # callback propagates out of gather and discards every
                    # conversation the run already completed -- hours of
                    # real provider calls thrown away because a progress
                    # bar's widget was unmounted. A run's output must
                    # survive its observer.
                    logger.exception(
                        "character probe progress callback failed at "
                        f"{done}/{total}; continuing the run"
                    )
            return conversation

        return list(await asyncio.gather(*(_guarded(job) for job in jobs)))

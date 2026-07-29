# tldw_chatbook/Agents/run_log_eviction.py
"""Evict older run-log-backed rounds from the SEND payload (TASK-1272, Phase 3).

Phase 1 (PR #1066) is additive: it writes a lossless run log but never
shrinks what is sent to the provider, so long-horizon runs and
small-context local models still overflow the model window exactly as
before. This module is the "1:1 PRO-LONG" mechanism the design spec
(Docs/superpowers/specs/2026-07-27-agent-programmatic-run-memory-design.md
§10) describes: keep recent rounds in context verbatim, replace older
ones with a short pointer, and let the agent retrieve anything it
actually needs via ``search_run_log``.

Applied at the SEND seam only (``agent_service.AgentService._make_call_
model``'s inner ``call_model`` closure), immediately before the provider
call. ``agent_runtime.run_agent_loop``'s own ``messages`` list is never
touched by this module -- that list backs cycle detection, retries, and
step accounting, none of which may see a shorter history than the run
actually produced. Only what is SENT shrinks.

Pure with respect to I/O: the only external call is
``Chat.console_history_budget.bound_messages_to_window``, itself pure.
Window lookup, safety margin, reply reservation, and leading-system-prefix
preservation are solved there and are reused, not reimplemented -- the
only addition this module makes is a protocol-aware turn-boundary
predicate (see ``_make_round_boundary``) and the synthetic placeholder
that replaces whatever got dropped.

**Live-verified model-dependence (2026-07-28), read before enabling this
in production:** the payload bound here is correct and provably bounded --
that part is verified and not in question. Whether bounding it also lets a
run COMPLETE more work than sending the full history depends on the model.
PRO-LONG's premise is that the agent compensates for evicted rounds by
actively querying ``search_run_log``; a capable (frontier-class) model does
this reliably. A 26B local model on the fence protocol, live-tested against
a 10-file sequential task, did not: with eviction on, its payload correctly
plateaued under the declared window (eviction doing exactly its job), but
the run ended ``status=stuck`` with an empty answer -- the model re-read
files it had already read instead of searching the log for them, until the
loop's cycle detector killed the run. The SAME task with eviction off
overflowed the window but answered correctly. See the design spec
(Docs/superpowers/specs/2026-07-27-agent-programmatic-run-memory-design.md
§10) for the full numbers and reasoning. This is precisely why
``RUN_LOG_EVICT_ENABLED_KEY`` defaults to off: enabling it is a bet on the
configured model's ability to recover from the log, not a strict
improvement to turn on universally -- and the irony is that small-context
local models, one of this feature's own motivating cases, are the class of
model least likely to win that bet.
"""

from __future__ import annotations

from typing import Any, Callable

from loguru import logger

from tldw_chatbook.Chat.console_history_budget import (
    DEFAULT_RESPONSE_RESERVATION,
    bound_messages_to_window,
)

from .agent_models import FENCE_TOOL_RESULT_PREFIX, SEARCH_RUN_LOG_TOOL_NAME

#: `[agents]` config key (see `run_log._setting`'s env-var/TOML/default
#: tiering). Off by default (task-1272 AC #5): existing runs must stay
#: byte-identical to today's payload until a user opts in.
#:
#: MODEL-DEPENDENT, live-verified 2026-07-28 (see this module's docstring
#: and the design spec §10 for the full evidence): this suits a model
#: capable of actively using `search_run_log` to recover evicted context --
#: verified against frontier-class models per the design spec's cited
#: paper. A weaker model (live-tested: a 26B local model, fence protocol)
#: may not query the log reliably when its recent turns are trimmed, and
#: will instead re-attempt work it already completed, potentially running
#: the loop into its cycle detector and a `stuck` status with no answer --
#: worse than simply overflowing the window would have been. This is WHY
#: the default is off: turning it on should be a deliberate choice for a
#: model known to search its log, not a blanket "trim more, always
#: better" setting.
RUN_LOG_EVICT_ENABLED_KEY = "run_log_evict_enabled"

#: `[agents]` config key for the minimum-recent-rounds floor (live-verified
#: 2026-07-28 follow-up). See `DEFAULT_MIN_RECENT_ROUNDS` for the default
#: and its rationale.
RUN_LOG_EVICT_MIN_RECENT_ROUNDS_KEY = "run_log_evict_min_recent_rounds"

#: Never trim below this many of the most recent complete rounds,
#: regardless of the token budget (`bound_messages_to_window`'s
#: `min_recent_turns`). Without a floor, "keep whatever fits" can
#: degenerate at a tight window to keeping only the in-flight round -- an
#: agent then cannot see the handful of steps it just took and repeats
#: them, which live-verified as a fixed-point payload (eviction removing
#: new rounds as fast as they are added) ending in the cycle detector
#: firing and the run going `stuck`. 4 is chosen to comfortably cover a
#: short, linear multi-step task -- e.g. the live reproduction (read four
#: files, one round each, then answer) -- entirely in view at once: with a
#: floor of 4, all four read rounds stay visible together right up to the
#: final answering turn, instead of the oldest of them aging out mid-task.
#: Smaller (2-3) risked keeping the SAME class of bug on a task just one
#: round longer than the floor; larger meaningfully narrows how much
#: eviction can ever save on a small-context model, which is this phase's
#: whole point. Configurable because that trade-off is genuinely
#: workload-dependent.
DEFAULT_MIN_RECENT_ROUNDS = 4


def coerce_min_recent_rounds(value: object) -> int:
    """Defensively coerce a configured floor to a non-negative int.

    Called by ``agent_service._make_call_model`` on the raw value
    ``run_log._setting(RUN_LOG_EVICT_MIN_RECENT_ROUNDS_KEY, ...)`` returns
    (a string from an env-var override, whatever TOML parsed it to, or
    already the int default) before it reaches
    ``bound_history_for_send``/``bound_messages_to_window``, mirroring the
    defensive-coercion pattern ``run_log._coerce_positive_int`` already
    establishes for the other numeric run-log settings.

    Args:
        value: The raw configured value.

    Returns:
        ``value`` as a non-negative int. ``0`` is a valid, deliberate
        choice (opts out of the floor, keeping only the current-turn
        guarantee ``bound_messages_to_window`` already provides
        unconditionally) and is passed through, not treated as invalid.
        Anything non-numeric or negative falls back to
        ``DEFAULT_MIN_RECENT_ROUNDS``, logged at warning.
    """
    try:
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            raise TypeError(f"unsupported type {type(value).__name__}")
        parsed = int(value)
    except (TypeError, ValueError):
        logger.warning(
            f"run log: invalid {RUN_LOG_EVICT_MIN_RECENT_ROUNDS_KEY}={value!r}; "
            f"using default {DEFAULT_MIN_RECENT_ROUNDS}"
        )
        return DEFAULT_MIN_RECENT_ROUNDS
    if parsed < 0:
        logger.warning(
            f"run log: negative {RUN_LOG_EVICT_MIN_RECENT_ROUNDS_KEY}={parsed}; "
            f"using default {DEFAULT_MIN_RECENT_ROUNDS}"
        )
        return DEFAULT_MIN_RECENT_ROUNDS
    return parsed


def _is_fence_tool_result(message: dict[str, Any]) -> bool:
    """Whether ``message`` is a fence-protocol tool-result continuation row.

    Matches the EXACT prefix ``agent_runtime._append_tool_result`` writes
    for the fence protocol (``FENCE_TOOL_RESULT_PREFIX``, shared via
    ``agent_models`` so the two can never drift apart), rather than a
    re-typed copy of the string.
    """
    content = message.get("content")
    return isinstance(content, str) and content.startswith(FENCE_TOOL_RESULT_PREFIX)


def _make_round_boundary(*, native: bool) -> Callable[[dict[str, Any]], bool]:
    """Build the round-boundary predicate for ``bound_messages_to_window``.

    A "round" here is one model turn plus whatever it produced: a plain
    final answer, or a tool call and its result. Every assistant-authored
    message starts a new round (``run_agent_loop`` never emits two
    assistant messages for the same round); everything that follows until
    the next one is a continuation of it -- a native ``role="tool"``
    reply, or a fence-protocol ``role="user"`` tool-result row.

    ``console_history_budget._group_turns``'s DEFAULT rule (``role ==
    "user"`` starts a boundary) is Console's own turn concept and is
    wrong for an agent run's payload in two ways this predicate corrects:

    1. It never treats an assistant reply as a boundary at all, which
       would collapse an entire run's growth into one undroppable
       "current turn" and defeat this module's purpose outright.
    2. For the fence protocol specifically, it misreads every tool-result
       row as a NEW boundary (``agent_runtime._append_tool_result``'s
       fence branch appends one as ``role="user"``), splitting it from
       the assistant round it answers. An eviction built on that
       misreading could drop the tool call while keeping its result, or
       the reverse -- exactly the orphaned-pair failure task-1272 warns
       against.

    Native tool-results (``role="tool"``) never collide with either rule,
    so this predicate's fence-specific check is inert for native runs;
    ``native`` is still threaded through explicitly (rather than relying
    on that as an accident of role naming) so a native run can never, even
    in principle, have a genuine ``role="user"`` row misread as a
    continuation.

    Args:
        native: Whether this run's provider uses the native tool-call
            protocol (``agent_service._make_call_model``'s ``native``,
            already resolved at the call site).

    Returns:
        A predicate suitable for ``bound_messages_to_window``'s
        ``is_turn_boundary``: ``True`` for a row that starts a new round,
        ``False`` for one that continues the previous round.
    """

    def boundary(message: dict[str, Any]) -> bool:
        role = message.get("role")
        if role == "assistant":
            return True
        if role != "user":
            return False
        if native:
            return True
        return not _is_fence_tool_result(message)

    return boundary


def _pinned_prefix_len(payload: list[dict[str, Any]]) -> int:
    """How many leading rows `bound_messages_to_window(pin_first_user=True)`
    pins: the leading system rows plus, ambiguity-free, everything up to and
    including the first `role == "user"` row after them (the task
    instruction -- see `bound_history_for_send`'s `pin_first_user=True`).

    Mirrors that function's own prefix computation on the ORIGINAL payload
    rather than reading it back off the result, so the synthetic note can be
    spliced at the exact boundary; not a reimplementation of the trimming
    itself, only of the few lines that locate this one insertion point.
    """
    index = 0
    while index < len(payload) and payload[index].get("role") == "system":
        index += 1
    for i in range(index, len(payload)):
        if payload[i].get("role") == "user":
            return i + 1
    return index


def _synthetic_note(dropped_rounds: int) -> dict[str, str]:
    """Build the placeholder telling the model what was evicted.

    Deliberately does not name specific run-log record numbers: the loop
    does not track which record(s) back which evicted round, and a guessed
    number would be followed with false confidence -- worse than no
    pointer at all. A round count plus the tool name is what this seam can
    honestly promise (task-1272 requirement #4).

    Sent as a ``role="user"`` row rather than ``role="system"``: a system
    row appearing anywhere but first in the array trips some local chat
    templates (a common assumption is "at most one system message, and it
    is the first"), which is precisely the audience the fence protocol --
    and therefore this eviction path -- serves. Bracketed wording marks it
    as machine-generated so the model does not mistake it for something
    the human said.

    Args:
        dropped_rounds: Number of whole rounds removed from this send.

    Returns:
        A single message dict to splice in where the dropped rounds were.
    """
    round_word = "round" if dropped_rounds == 1 else "rounds"
    return {
        "role": "user",
        "content": (
            f"[Context note: {dropped_rounds} earlier {round_word} of this "
            f"run were omitted here to stay within the model's context "
            f"window. Nothing was lost -- this run's complete history is "
            f"recorded in full. Call {SEARCH_RUN_LOG_TOOL_NAME} if you "
            f"need something from before this point instead of "
            f"re-deriving or guessing it.]"
        ),
    }


def bound_history_for_send(
    payload: list[dict[str, Any]],
    *,
    model: str,
    provider: str,
    native: bool,
    enabled: bool,
    response_reservation: int = DEFAULT_RESPONSE_RESERVATION,
    window: int | None = None,
    count_fn: Callable[[list[dict[str, Any]], str], int] | None = None,
    min_recent_rounds: int = DEFAULT_MIN_RECENT_ROUNDS,
) -> list[dict[str, Any]]:
    """Bound one turn's SEND payload to the model window, run-log-aware.

    Called at the provider-call seam, immediately before ``chat_call``,
    with the FULL prospective payload (leading system message plus
    history) -- never ``run_agent_loop``'s own ``messages`` list, and
    never anything that gets fed back into the loop.

    Args:
        payload: The full provider payload for this turn (system message
            first, then history). Never mutated.
        model: Model name, for window lookup and token counting.
        provider: Provider/endpoint name, for window lookup fallback.
        native: Whether this run uses the native tool-call protocol.
            Selects the round-boundary rule (see ``_make_round_boundary``).
        enabled: The fully-resolved gate for this run: the run log is
            active AND ``search_run_log`` was actually offered to this
            agent (``log_active`` in ``agent_service._run_one``) AND the
            ``run_log_evict_enabled`` config flag is on. This function
            does not re-derive any part of that gate -- callers MUST pass
            ``False`` unless every condition holds (task-1272 requirement
            #1: never evict when there is no log to recover from).
        response_reservation: Tokens reserved for the reply.
        window: Explicit context window override (tests); ``None`` uses
            the normal token_counter lookup.
        count_fn: Injectable token counter (tests); ``None`` uses the real
            one.
        min_recent_rounds: Minimum number of most recent complete rounds
            that must always survive, regardless of budget (forwarded to
            ``bound_messages_to_window``'s ``min_recent_turns``; see
            ``DEFAULT_MIN_RECENT_ROUNDS`` for the default and its
            rationale). Already coerced/validated by the caller
            (``agent_service._make_call_model``, via ``run_log._setting``)
            -- this function trusts it as-is, same as every other
            already-resolved parameter here.

    Returns:
        ``payload`` unchanged (same object) when ``enabled`` is ``False``,
        when nothing needed dropping, or when trimming itself failed;
        otherwise a NEW list with the oldest whole rounds removed and a
        synthetic note in their place. Never raises: any failure degrades
        to sending the full history for this turn, logged at warning --
        eviction is a context optimisation, never load-bearing for the
        run's correctness (task-1272: "must never raise into an agent
        run").
    """
    if not enabled:
        return payload
    try:
        bound = bound_messages_to_window(
            payload,
            model=model,
            provider=provider,
            response_reservation=response_reservation,
            window=window,
            count_fn=count_fn,
            is_turn_boundary=_make_round_boundary(native=native),
            # Live-verified 2026-07-28: without this, the task instruction
            # -- the payload's only REAL role="user" row -- sits in the
            # middle of history like any other droppable round and a tight
            # enough window silently evicted it, leaving the agent with no
            # memory of what it was asked to do (it then narrated about
            # its own log instead of finishing). See the parameter's
            # docstring in console_history_budget.py.
            pin_first_user=True,
            # Live-verified follow-up, same day: without a floor, a tight
            # enough window can keep only the in-flight round, so the agent
            # can no longer see the handful of steps it just took and
            # repeats them -- see `DEFAULT_MIN_RECENT_ROUNDS`.
            min_recent_turns=min_recent_rounds,
        )
        if not bound.dropped_turns:
            return payload
        # Splice the note in right where the dropped rounds were: after the
        # PINNED prefix (recomputed here on the ORIGINAL payload -- a small
        # scan mirroring `bound_messages_to_window`'s own `pin_first_user`
        # logic, not a re-implementation of the trimmer itself -- because
        # that prefix is preserved verbatim, so its length is identical in
        # `bound.messages`).
        result = list(bound.messages)
        result.insert(_pinned_prefix_len(payload), _synthetic_note(bound.dropped_turns))
        return result
    except Exception:  # noqa: BLE001 -- eviction must never abort a run
        logger.opt(exception=True).warning(
            "run-log eviction failed for this turn; sending full history"
        )
        return payload

"""Console hands-free controller.

Extracted out of `ChatScreen` (wave-2 console decomposition, task 1): the
V3 pipeline hands-free conversation loop -- speak, it sends, the reply is
spoken, speak again -- plus the two-engine coordination points shared with
the V4 realtime loop added by PR #1350. `Chat/console_hands_free.py`
(`HandsFreeController`, the headless FSM) and `Chat/reply_sentence_
sequencer.py` (`SentenceSequencer`, the speech splitter) are pure/headless;
this module is their thin Console-screen wiring, matching `dictation.py`'s
own "controller, a plain object that owns state and behaviour with no
region of its own" shape
(`Docs/superpowers/specs/2026-08-02-screen-decomposition-design.md`).

TWO-ENGINE BOUNDARY (the load-bearing design decision this file makes):

`_enter_console_hands_free_loop` is an engine FORK, not a pipeline-only
entry point -- PR #1350 (realtime voice V4) added a second engine,
`ChatScreen._console_realtime` (+ `_console_realtime_close_worker`),
mutually exclusive with this module's own `_console_hands_free` pipeline
session by construction (the fork picks exactly one per loop entry).
`ConsoleHandsFreeController` owns the FORK and both cross-engine action
entry points (`action_toggle_console_hands_free`/`action_exit_console_
hands_free`, moved here as-is, plus `check_action`'s availability check,
moved as `console_hands_free_exit_available`) because all three read BOTH
engines -- that is coordination logic, not pipeline-only logic. "Esc from
any point in the loop" is a promise the docs make about hands-free, not
about one engine's implementation of it, so `action_exit_console_hands_
free` exits both unconditionally.

The realtime engine's own state and ~60-method implementation
(`ConsoleRealtimeSession` and every `_console_realtime_*`/`_on_console_
realtime_*` method) deliberately did NOT move here and stay on `ChatScreen`
exactly as PR #1350 wrote them -- dragging that whole stack across the
boundary was never this task's job, only the fork and the two action entry
points were. This module reaches the realtime engine through exactly two
named constructor callables, `realtime_session_accessor` and
`enter_realtime_loop` (see `__init__`'s docstring, binding kind 3) -- never
a bare screen handle. The reverse direction (realtime code on `ChatScreen`
calling into the PIPELINE engine moved here, e.g. `_console_realtime_
fallback_to_pipeline`'s loud fallback) is ordinary screen-calls-its-own-
controller traffic: `ChatScreen` holds `self._hands_free` the same way it
holds `self._dictation`/`self._workspace`, and calls its methods directly
by their ORIGINAL private names -- no injection needed for that direction,
matching the established convention throughout this decomposition.

Moved verbatim (byte-for-byte, mechanically diffable against the pre-move
source):

- The module-level vocabulary this loop's own wiring needs:
  `CONSOLE_HANDS_FREE_DEGRADED_MESSAGE`, `ConsoleHandsFreeSession` (the
  per-loop-entry dataclass), and `CONSOLE_REALTIME_FORCED_UNCONFIGURED_
  MESSAGE` (realtime-labeled but consumed ONLY by the fork below -- it
  stayed textually inside `ChatScreen`'s realtime-constants block since
  wave 1 for no reason beyond neighboring code; the fork is the only
  reader, so it moved with its one caller rather than staying behind for a
  circular re-import).
- Every `ChatScreen` method matching `*hands_free*` whose body was
  pipeline-shaped: the engine fork itself, plus all 26 methods between
  `_enter_console_hands_free_pipeline_loop` and `_on_console_hands_free_
  sequencer_drained` in the pre-move source's "Hands-free conversation
  loop" section, plus `_deliver_console_hands_free_capture_ended` (a
  dictation-cluster-adjacent async helper the pre-move source placed near
  `ChatScreen.on_unmount` rather than in that section, but whose own
  docstring names it "Console hands-free" throughout) and `_console_
  pipeline_hands_free_blocker` (the fallback-availability check the
  realtime engine's own loud-fallback method calls). 28 methods, ~730
  lines -- both counts confirmed against the pre-move source. Plus the two
  action entry points and the `check_action` branch described above.
- `_CONSOLE_HANDS_FREE_CAPTURE_ENDED_WAIT_SECONDS`, the class constant
  `_deliver_console_hands_free_capture_ended` is bound on.

Three kinds of binding, same rule as `dictation.py`'s own (see that
module's `ConsoleDictationController.__init__` docstring for the full
rationale; restated briefly here):

1. **Framework services** (`run_worker`, `set_interval`, `is_mounted`)
   live-read from `screen` via `@property` on every access.
2. **App-level dependencies that are not this controller's own state** are
   NAMED keyword-only constructor callables (`composer_accessor`, `chat_
   store_accessor`, dictation's state/actions, `run_pending_voice_action`,
   and the two realtime callables above) -- each exposed back to the 28
   moved bodies as a thin property under the SAME original name the
   pre-move source used, so none of those bodies needed an internal edit.
   A handful of these (`_console_dictation_state`, `_console_dictation_
   origin_session_id`, `_console_pending_voice_action`) were bare-attribute
   reads/writes in the original, not calls -- their properties therefore
   invoke the stored callable internally and return/accept the VALUE,
   unlike `_console_composer_or_none` and friends, whose properties return
   the callable itself because the original bodies already called it with
   `()`. `_console_pending_voice_action` is write-only (the 28 bodies never
   read it, only `_console_hands_free_request_stop_and_send` writes it) --
   its getter raises rather than silently returning a stale/wrong value.
3. `app_instance` is the one plain-attribute exception, for the same
   reason `dictation.py` snapshots it: `notify()`/`post_message()` calls
   read it as a bare attribute in the pre-move source, never through a
   call that could go stale.

There is no more "kind 2, disclosed temporary exception" bucket here, and
`dictation.py`'s no longer has one either (see that module's docstring):
wave 1 left FOUR of dictation's reach-backs (`_console_hands_free` and its
`_console_hands_free_vad_degraded` sibling, `_enter_console_hands_free_
loop`, `_console_hands_free_force_immediate_send`, `_deliver_console_
hands_free_capture_ended`) as live properties straight through `screen`,
explicitly disclosed as temporary because hands-free had no controller of
its own yet to hand a named dependency to. It does now: `ChatScreen.
__init__` wires all four (plus `_console_realtime_adopt_transcript` and
`_run_pending_console_voice_action`, which were the same shape for the
same reason even though their targets are screen-owned, not moved here) as
named callables pointed at THIS controller, exactly like every other
app-level dependency in this decomposition. That exception is over.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import asyncio
import threading
import time
from typing import Any, TYPE_CHECKING

from loguru import logger

from ...Chat import console_voice_input
from ...Chat.console_voice_input import (
    acoustic_barge_in_enabled,
    handsfree_send_delay_seconds,
    resolve_handsfree_engine,
    realtime_enabled,
)
from ...Chat.console_chat_models import ConsoleMessageRole
from ...Chat.console_hands_free import (
    CloseCapture,
    CountdownTick,
    ExitLoop,
    HandsFreeController,
    HandsFreeIntent,
    ModeChanged,
    OpenCapture,
    RequestStopAndSend,
    SilenceSpeech,
    SuppressReplySpeech,
)
from ...Chat.reply_sentence_sequencer import SentenceSequencer
from ...Widgets.Console import ConsoleComposerBar

if TYPE_CHECKING:
    from ...TTS.profile_types import CharacterRef
    from ...Widgets.Console.console_control_bar import (
        ConsoleAutoSpeakResumeRequested,
        ConsoleAutoSpeakRetryRequested,
    )
    from ...Widgets.Console.console_speech_controls import ConsoleAutoSpeakChanged
    from ..Screens.chat_screen import ChatScreen

logger = logger.bind(module="ChatScreen")


#: Task 5 (VAD-degraded honesty carrier): shown once per hands-free ENTRY
#: (not once per app run, unlike `VAD_UNAVAILABLE_MESSAGE` -- entering the
#: loop is the moment this limitation actually starts to matter) when
#: `webrtcvad` is unavailable. Reuses `VAD_UNAVAILABLE_MESSAGE`'s own
#: framing (see `console_voice_input.VoiceVadUnavailable`'s docstring):
#: without it, the silence gate that drives auto-send/barge-in never fires.
CONSOLE_HANDS_FREE_DEGRADED_MESSAGE = (
    "Hands-free is degraded: voice-activity detection (webrtcvad) is not "
    "installed, so it cannot auto-send on a pause or hear a spoken barge-in. "
    'Use the mic button, "Console, stop.", or Esc/ctrl+shift+h to end a turn.'
)


#: Realtime-labeled but consumed only by the engine fork below -- see the
#: module docstring's "moved verbatim" section for why this lives here now
#: instead of in `ChatScreen`'s own realtime-constants block.
CONSOLE_REALTIME_FORCED_UNCONFIGURED_MESSAGE = (
    "Hands-free is set to the realtime engine, but [realtime] enabled is "
    'false. Turn it on in config, or set dictation.handsfree_engine to "auto" '
    'or "pipeline".'
)


@dataclass
class ConsoleHandsFreeSession:
    """Everything the hands-free conversation loop needs while it runs.

    Constructed once per loop entry (`ChatScreen._enter_console_hands_free_
    loop`) and torn down on `ExitLoop` (`ChatScreen._teardown_console_hands_
    free_loop`) -- never reused across loop entries, unlike the one-shot
    dictation session, so a fresh `HandsFreeController`/`SentenceSequencer`
    pair with clean state is guaranteed for every "hands free" invocation.

    Attributes:
        controller: The headless FSM driving the loop.
        sequencer: The headless sentence-boundary speech sequencer, reused
            across every reply in this loop (`begin_reply()` resets its
            per-reply state -- see that method's docstring).
        tick_timer: The `set_interval(0.1, ...)` handle driving both
            `controller.tick(now)` and the chip repaint. Stopped on
            teardown.
        reply_id: The outstanding reply's assistant-message id, or None
            when no reply is outstanding. Claimed by the first delta/
            completion tap call that passes the reply-identity guard (see
            `pending_session_id`/`pending_existing_assistant_ids` below and
            `_console_hands_free_try_claim_reply`'s docstring) while
            `controller.state == "awaiting_reply"`.
        toast_shown_for_reply: Policy state for `speak_utterance`'s `quiet`
            parameter -- at most one failure toast per reply; every
            subsequent utterance in the same reply passes `quiet=True`
            once this is True. Reset in `_begin_console_hands_free_reply`.
        countdown_remaining: The most recent `CountdownTick.remaining`,
            painted into the chip by the 0.1 s tick (see `_repaint_console_
            hands_free_chip`). Meaningless outside `controller.state ==
            "countdown"`.
        pending_session_id: The Console session `RequestStopAndSend`
            recorded this send as going into (`_console_dictation_origin_
            session_id` at that moment -- the SAME value the existing V2
            wrong-session-refusal already uses, not `store.active_session_
            id` re-read later, which a tab switch could have moved on from
            by dispatch time). `None` until the first send. Part of the
            reply-identity guard (task-5 review B1/M7).
        pending_existing_assistant_ids: A snapshot of every assistant
            message id already present in `pending_session_id` at the
            moment `RequestStopAndSend` fired -- taken BEFORE the send
            creates its own new assistant row, so any id in this set can
            only be a PRE-EXISTING (therefore stale/foreign) reply, never
            the new one. Part of the reply-identity guard (task-5 review
            B1).
    """

    controller: HandsFreeController
    sequencer: SentenceSequencer
    tick_timer: Any = None
    reply_id: str | None = None
    toast_shown_for_reply: bool = False
    countdown_remaining: float = 0.0
    pending_session_id: str | None = None
    pending_existing_assistant_ids: frozenset[str] = frozenset()


class ConsoleHandsFreeController:
    """Owns the Console shell's V3 pipeline hands-free loop, plus the
    engine-fork/action coordination shared with the V4 realtime loop.

    Wave 2's first controller extraction to sit alongside a still-on-screen
    sibling engine (see the module docstring's two-engine boundary
    section). `ChatScreen` constructs exactly one of these, in `__init__`,
    and keeps a `self._hands_free` reference plus thin delegations for the
    two Textual `action_*` entry points and the `check_action` branch (all
    three call straight through to this controller -- see their own
    docstrings on `ChatScreen`).
    """

    #: Bound on how long `_deliver_console_hands_free_capture_ended` waits
    #: for a limit-triggered stop to actually reach `idle` before giving up
    #: -- generous relative to `dictation.stop_join_timeout_seconds`'s own
    #: 30s default (that timeout is the transcription-thread join `_stop_
    #: console_dictation` itself is bounded by; this only needs to exceed
    #: it, not match it exactly). Giving up here is a safe failure mode: no
    #: `on_capture_ended` is delivered at all, so the FSM's own bookkeeping
    #: is never told something that did not (yet) happen, and the user can
    #: still exit manually (Esc/mic/ctrl+shift+h/spoken "stop").
    _CONSOLE_HANDS_FREE_CAPTURE_ENDED_WAIT_SECONDS: float = 40.0

    def __init__(
        self,
        screen: "ChatScreen",
        *,
        app_instance: Any,
        composer_accessor: Callable[[], ConsoleComposerBar | None],
        chat_store_accessor: Callable[[], Any],
        dictation_state_accessor: Callable[[], str],
        dictation_origin_session_id_accessor: Callable[[], str | None],
        set_pending_voice_action: Callable[[str | None], None],
        request_dictation_start: Callable[[], None],
        request_dictation_stop: Callable[[], None],
        run_pending_voice_action: Callable[[str | None], Any],
        realtime_session_accessor: Callable[[], Any],
        enter_realtime_loop: Callable[..., None],
        request_auto_speak_enabled: Callable[[bool], None],
        request_auto_speak_resume: Callable[[], None],
        request_auto_speak_retry: Callable[[], None],
        sync_auto_speak_controls: Callable[[bool, bool, bool], None],
        sync_hands_free_state: Callable[[bool], None],
    ) -> None:
        """Build the controller and bind everything its moved bodies need.

        Every moved method body below is a byte-for-byte copy of the
        pre-extraction `ChatScreen` method (see the module docstring's
        "moved verbatim" section) -- no internal line was edited to
        retarget a call or an attribute. That is possible because this
        constructor binds every name those bodies reference that is not
        this controller's own state, under the SAME name the original
        method used. See the module docstring for the three binding kinds.

        Args:
            screen: The Console screen. Used ONLY for the framework
                services in binding kind 1 (`run_worker`/`set_interval`/
                `is_mounted`). No `query_one` traffic -- this controller
                owns no DOM of its own, so there is no region boundary for
                it to cross.
            app_instance: For `notify()` and posting TTS events -- unchanged
                from how the pre-extraction methods used `self.app_
                instance`. Snapshotted as a plain attribute; see the module
                docstring's binding-kind-3 paragraph for why that is
                correct here (every pre-move use was a bare-attribute read,
                never a call that could go stale).
            composer_accessor: `ChatScreen._console_composer_or_none`,
                late-binding lambda -- same rationale as `dictation.py`'s
                identical parameter.
            chat_store_accessor: `ChatScreen._ensure_console_chat_store`,
                same rationale.
            dictation_state_accessor: Reads `ChatScreen._console_dictation_
                state` (itself a proxy onto `ConsoleDictationController`'s
                own state) at call time. The property exposing this below
                CALLS the accessor and returns the value, unlike `_console_
                composer_or_none`'s property -- the pre-move bodies read
                `self._console_dictation_state` bare, never with `()`.
            dictation_origin_session_id_accessor: Same shape, for
                `ChatScreen._console_dictation_origin_session_id`. Read-only
                -- only `_console_hands_free_request_stop_and_send` reads
                it, and it never writes it.
            set_pending_voice_action: Writes `ChatScreen._console_pending_
                voice_action` (itself a proxy onto dictation's own state).
                Setter-only: `_console_hands_free_request_stop_and_send` is
                the ONLY pre-move body that touches this attribute, and it
                only ever assigns to it, never reads it -- the property
                below is therefore write-only too (its getter raises),
                rather than silently exposing a read no moved body needs.
            request_dictation_start: `ChatScreen._request_console_dictation_
                start`, itself already a one-line delegation onto
                `ConsoleDictationController`.
            request_dictation_stop: `ChatScreen._request_console_dictation_
                stop`, same shape.
            run_pending_voice_action: `ChatScreen._run_pending_console_
                voice_action` -- general screen-orchestration (chat store,
                send button, tab creation, TTS read-back), shared with
                `ConsoleDictationController`'s OWN identically-named
                dependency; both point at the same screen method.
            realtime_session_accessor: Reads `ChatScreen._console_realtime`
                -- the ONLY way this controller ever touches the realtime
                engine's live session. See the module docstring's
                two-engine boundary section.
            enter_realtime_loop: `ChatScreen._enter_console_realtime_loop`,
                the realtime engine's own entry point -- called only from
                the fork (`_enter_console_hands_free_loop`), which picks
                exactly one engine per loop entry.
            request_auto_speak_enabled: Late-bound coordinator enable request.
            request_auto_speak_resume: Late-bound coordinator resume request.
            request_auto_speak_retry: Late-bound coordinator retry request.
            sync_auto_speak_controls: Presentation-only auto-speak state edge.
            sync_hands_free_state: Presentation-only Hands-free state edge.
        """
        self._screen = screen
        self.app_instance = app_instance
        self._composer_accessor = composer_accessor
        self._chat_store_accessor = chat_store_accessor
        self._dictation_state_accessor = dictation_state_accessor
        self._dictation_origin_session_id_accessor = (
            dictation_origin_session_id_accessor
        )
        self._set_pending_voice_action_fn = set_pending_voice_action
        self._request_dictation_start_fn = request_dictation_start
        self._request_dictation_stop_fn = request_dictation_stop
        self._run_pending_voice_action_fn = run_pending_voice_action
        self._realtime_session_accessor = realtime_session_accessor
        self._enter_realtime_loop_fn = enter_realtime_loop
        self._request_auto_speak_enabled_fn = request_auto_speak_enabled
        self._request_auto_speak_resume_fn = request_auto_speak_resume
        self._request_auto_speak_retry_fn = request_auto_speak_retry
        self._sync_auto_speak_controls_fn = sync_auto_speak_controls
        self._sync_hands_free_state_fn = sync_hands_free_state

        # The pipeline engine's own state, moved verbatim from
        # `ChatScreen.__init__`.
        self._console_hands_free: ConsoleHandsFreeSession | None = None
        #: True once `_install_console_hands_free_store_tap` has wrapped the
        #: store's `append_stream_chunk`/`mark_message_*` methods. The store
        #: itself is a lazily-created singleton for this screen instance
        #: (`_ensure_console_chat_store`), so this only ever needs doing once.
        self._console_hands_free_store_tap_installed = False
        #: `(store, {seam: (had_own_attr, original)}, {seam: wrapper})` while
        #: this screen's tap is installed -- see
        #: `uninstall_console_hands_free_store_tap`. The store OUTLIVES the
        #: screen since task-15860, so the tap has to come back off.
        self._console_hands_free_store_tap_undo: Any | None = None
        #: Set once (per app run) by a `VoiceVadUnavailable` event, via
        #: dictation's injected `set_hands_free_vad_degraded` callable. Read
        #: by `_enter_console_hands_free_pipeline_loop`, which shows a
        #: dedicated warning naming exactly what auto-send/barge-in cannot
        #: do in degraded mode, and by `_console_pipeline_hands_free_
        #: blocker`, the realtime engine's own loud-fallback check.
        self._console_hands_free_vad_degraded = False

    def _sync_hands_free_switch(self, active: bool) -> None:
        """Mirror hands-free session state through the presentation edge.

        task-18911 (fix 2): the Switch is the soft-keyboard-only user's
        entry/exit for the mode; every session lifecycle change repaints it
        so it never disagrees with reality. No-op when the header controls
        are not mounted (pre-mount, mid-teardown).
        """
        self._sync_hands_free_state_fn(active)

    async def _resolve_console_auto_speak_destination(
        self,
        assistant_kind: str | None,
        character_ref: "CharacterRef | None",
    ) -> Any:
        """Resolve the same effective TTS authority used by synthesis."""
        ensure_handler = getattr(self.app_instance, "_ensure_tts_handler", None)
        if not callable(ensure_handler):
            return None
        handler = await ensure_handler()
        resolver = getattr(handler, "resolve_console_speech_destination", None)
        if not callable(resolver):
            return None
        try:
            return await resolver(assistant_kind, character_ref)
        except Exception:
            logger.opt(exception=True).warning(
                "Failed to resolve the Console auto-speak destination."
            )
            return None

    def _sync_console_auto_speak_controls(
        self,
        enabled: bool,
        paused: bool,
        retry_available: bool = False,
    ) -> None:
        """Push authoritative state through the presentation-only edge."""
        self._sync_auto_speak_controls_fn(enabled, paused, retry_available)

    def on_console_auto_speak_changed(
        self,
        event: "ConsoleAutoSpeakChanged",
    ) -> None:
        """Request the durable per-conversation auto-speak state.

        Args:
            event: Change event carrying the requested enabled state.
        """
        self._request_auto_speak_enabled_fn(event.enabled)

    def on_console_auto_speak_resume_requested(
        self,
        event: "ConsoleAutoSpeakResumeRequested",
    ) -> None:
        """Resume future automatic speech after a failure.

        Args:
            event: Resume request from the Console speech controls.
        """
        self._request_auto_speak_resume_fn()

    def on_console_auto_speak_retry_requested(
        self,
        event: "ConsoleAutoSpeakRetryRequested",
    ) -> None:
        """Retry the failed automatic reply.

        Args:
            event: Retry request from the Console speech controls.
        """
        self._request_auto_speak_retry_fn()

    @property
    def is_mounted(self) -> bool:
        """Whether the Console screen is currently mounted.

        A live check: mount state changes over the controller's life, so
        this has to re-read `screen.is_mounted` on every access.
        """
        return self._screen.is_mounted

    @property
    def run_worker(self) -> Any:
        """`Screen.run_worker`, bound. See `__init__`'s docstring for why
        this is a property rather than a value snapshotted once."""
        return self._screen.run_worker

    @property
    def set_interval(self) -> Any:
        """`Screen.set_interval`, bound. See `__init__`'s docstring."""
        return self._screen.set_interval

    @property
    def _console_composer_or_none(self) -> Any:
        """The injected `composer_accessor`. Kept under this name so the
        moved method bodies below still call `self._console_composer_or_
        none()` unchanged. See `__init__`'s docstring (binding kind 3)."""
        return self._composer_accessor

    @property
    def _ensure_console_chat_store(self) -> Any:
        """The injected `chat_store_accessor`. See `_console_composer_or_
        none`'s docstring immediately above."""
        return self._chat_store_accessor

    @property
    def _console_dictation_state(self) -> str:
        """Calls the injected `dictation_state_accessor` and returns the
        value -- the pre-move bodies read this bare, never with `()`."""
        return self._dictation_state_accessor()

    @property
    def _console_dictation_origin_session_id(self) -> str | None:
        """Calls the injected `dictation_origin_session_id_accessor`."""
        return self._dictation_origin_session_id_accessor()

    @property
    def _console_pending_voice_action(self) -> str | None:
        """Write-only: see `__init__`'s docstring for
        `set_pending_voice_action`.

        Raises `RuntimeError`, deliberately NOT `AttributeError`: this is a
        property, and `hasattr()`/`getattr(obj, name, default)` swallow
        `AttributeError` specifically. A defensive
        `getattr(self._hands_free, "_console_pending_voice_action", None)`
        would then read None forever regardless of the real value, with no
        error ever surfacing. `RuntimeError` propagates instead.
        """
        raise RuntimeError(
            "_console_pending_voice_action is write-only on ConsoleHandsFreeController"
        )

    @_console_pending_voice_action.setter
    def _console_pending_voice_action(self, value: str | None) -> None:
        self._set_pending_voice_action_fn(value)

    @property
    def _request_console_dictation_start(self) -> Any:
        """The injected `request_dictation_start`. See `_console_composer_
        or_none`'s docstring."""
        return self._request_dictation_start_fn

    @property
    def _request_console_dictation_stop(self) -> Any:
        """The injected `request_dictation_stop`. See `_console_composer_
        or_none`'s docstring."""
        return self._request_dictation_stop_fn

    @property
    def _run_pending_console_voice_action(self) -> Any:
        """The injected `run_pending_voice_action`. See `_console_composer_
        or_none`'s docstring."""
        return self._run_pending_voice_action_fn

    @property
    def _console_realtime(self) -> Any:
        """Calls the injected `realtime_session_accessor`. The realtime
        engine's live session, or None -- owned by
        `ConsoleRealtimeController.session`; see the module docstring's
        two-engine boundary section."""
        return self._realtime_session_accessor()

    @property
    def _enter_console_realtime_loop(self) -> Any:
        """The injected `enter_realtime_loop`. See `_console_composer_or_
        none`'s docstring."""
        return self._enter_realtime_loop_fn

    def action_toggle_console_hands_free(self) -> None:
        """`ctrl+shift+h`: enter the hands-free loop, or exit it if already running.

        Both engines exit through their own controller's `on_exit_request()`
        -- the toggle never tears state down directly, so the exit runs the
        same reasoned `ExitLoop` path every other exit route uses.
        """
        if self._console_hands_free is not None:
            self._console_hands_free.controller.on_exit_request()
            return
        if self._console_realtime is not None:
            self._console_realtime.controller.on_exit_request()
            return
        owner = getattr(self.app_instance, "meeting_session_owner", None)
        if owner is not None and getattr(owner, "is_active", False):
            self.app_instance.notify(
                "Meeting in progress: stop it in Meetings before using hands-free.",
                severity="warning",
            )
            return
        self._enter_console_hands_free_loop(
            capture_live=self._console_dictation_state == "recording"
        )

    def _enter_console_hands_free_loop(self, *, capture_live: bool) -> None:
        """Pick the hands-free engine, then start that engine's loop.

        The fork (V4 task 5, rule 1) is deliberately the ONLY place engine
        selection happens, and it happens once per loop entry -- never
        mid-loop. A loop that is already running keeps the engine it was
        started with: `resolve_handsfree_engine()` reads live config, and
        re-resolving it on a re-entry (a spoken "hands free" mid-loop, say)
        could otherwise hand a running V3 loop's re-entry to the realtime
        engine and leave two loops fighting over the microphone.

        `"realtime"` selected while `[realtime] enabled` is false is a
        FORCED-but-unconfigured selection, not a mistake to paper over:
        `resolve_handsfree_engine()`'s docstring is explicit that it never
        silently downgrades an explicit `dictation.handsfree_engine =
        "realtime"` to the pipeline, and that being honest about it is the
        caller's job. So it is refused here, loudly, rather than starting
        an engine the user disabled or a fallback they did not ask for.

        Args:
            capture_live: True when an existing one-shot dictation capture
                is already open and should be adopted as the loop's first
                turn; forwarded verbatim to whichever engine is selected.
        """
        if self._console_hands_free is not None:
            self._enter_console_hands_free_pipeline_loop(capture_live=capture_live)
            return
        if self._console_realtime is not None:
            # `RealtimeLoopController.enter()` is itself idempotent, so a
            # stray re-entry has nothing to re-confirm here (unlike V3,
            # whose `enter()` carries capture bookkeeping).
            return
        if resolve_handsfree_engine() == "realtime":
            if not realtime_enabled():
                self.app_instance.notify(
                    CONSOLE_REALTIME_FORCED_UNCONFIGURED_MESSAGE, severity="warning"
                )
                return
            self._enter_console_realtime_loop(capture_live=capture_live)
            return
        self._enter_console_hands_free_pipeline_loop(capture_live=capture_live)

    def _enter_console_hands_free_pipeline_loop(self, *, capture_live: bool) -> None:
        """Start (or re-confirm) the V3 pipeline hands-free loop.

        Reached from the engine fork above, and directly from the realtime
        engine's loud fallback (`_console_realtime_fallback_to_pipeline`),
        which must NOT re-run the fork -- it would resolve straight back to
        the realtime engine that just failed.

        Args:
            capture_live: True when an existing one-shot dictation capture
                is already open and should be adopted as the loop's first
                turn (spoken "hands free" mid-capture, or the key binding
                pressed while already recording); False opens a fresh
                capture (the key binding pressed from idle). Ignored on
                re-entry -- `HandsFreeController.enter()`'s own re-entry
                semantics trust its own `capture_open` bookkeeping instead
                of a possibly-stale argument (see that method's docstring).
        """
        existing = self._console_hands_free
        if existing is not None:
            existing.controller.enter(capture_live=capture_live)
            return
        if self._console_hands_free_vad_degraded:
            self.app_instance.notify(
                CONSOLE_HANDS_FREE_DEGRADED_MESSAGE, severity="warning"
            )
        controller = HandsFreeController(
            emit=self._handle_console_hands_free_intent,
            send_delay_seconds=handsfree_send_delay_seconds(),
            acoustic_barge_in=acoustic_barge_in_enabled(),
        )
        sequencer = SentenceSequencer(
            speak=self._dispatch_console_hands_free_speak,
            stop_speech=self._stop_console_hands_free_speech,
        )
        session = ConsoleHandsFreeSession(controller=controller, sequencer=sequencer)
        sequencer.on_drained = self._on_console_hands_free_sequencer_drained
        self._console_hands_free = session
        self._sync_hands_free_switch(True)
        self._install_console_hands_free_store_tap()
        session.tick_timer = self.set_interval(0.1, self._tick_console_hands_free)
        controller.enter(capture_live=capture_live)

    def _teardown_console_hands_free_loop(self) -> None:
        """Drop the loop session and repaint the chip back to normal.

        Only ever called from `_console_hands_free_exit_loop` (`ExitLoop`'s
        handler), after that method has already silenced any reply audio
        and closed the capture -- this just stops the tick timer and clears
        the composer's borrowed hands-free chip state.
        """
        session = self._console_hands_free
        if session is None:
            return
        if session.tick_timer is not None:
            session.tick_timer.stop()
        self._console_hands_free = None
        self._sync_hands_free_switch(False)
        composer = self._console_composer_or_none()
        if composer is not None:
            # Repaints over whatever hands-free's own `set_voice_status`
            # calls last left on screen (`countdown`/`awaiting-reply`/
            # `speaking` are not lifecycle states `sync_dictation_state`
            # knows, so only a fresh call with the REAL current one-shot
            # state clears them).
            composer.sync_dictation_state(self._console_dictation_state)

    def _tick_console_hands_free(self) -> None:
        """`set_interval(0.1, ...)`: the controller's only clock input."""
        session = self._console_hands_free
        if session is None:
            return
        session.controller.tick(time.monotonic())
        self._repaint_console_hands_free_chip()

    def _handle_console_hands_free_intent(self, intent: HandsFreeIntent) -> None:
        """Route one `HandsFreeIntent`, emitted synchronously by the
        controller, to the wiring machinery that acts on it."""
        if isinstance(intent, RequestStopAndSend):
            self._console_hands_free_request_stop_and_send()
        elif isinstance(intent, (SilenceSpeech, SuppressReplySpeech)):
            self._console_hands_free_silence_speech()
        elif isinstance(intent, OpenCapture):
            self._console_hands_free_open_capture()
        elif isinstance(intent, CloseCapture):
            self._console_hands_free_close_capture()
        elif isinstance(intent, CountdownTick):
            self._console_hands_free_countdown_tick(intent.remaining)
        elif isinstance(intent, ModeChanged):
            self._console_hands_free_mode_changed(intent.state)
        elif isinstance(intent, ExitLoop):
            self._console_hands_free_exit_loop()

    def _console_hands_free_request_stop_and_send(self) -> None:
        """`RequestStopAndSend`: drive the existing V2 pending-send seam.

        Queues the send exactly like a spoken "Console, send." does
        (`_console_pending_voice_action = "send"`), then either stops the
        still-open capture -- the common case; `_stop_console_dictation`'s
        own success tail runs `_run_pending_console_voice_action`, which
        dispatches the queued send once the transcript has actually landed
        -- or, if the capture has ALREADY ended by the time this intent
        lands (a service-side capture limit reached `on_capture_ended`
        before this ran, so `_console_dictation_state` is already back at
        `idle`), dispatches the queued send directly, since there is
        nothing left to stop. There is no second send path either way --
        both branches ultimately run `_run_pending_console_voice_action`,
        the same method a spoken "send" already uses.

        Task-5 review B1/M7: records `pending_session_id` (from
        `_console_dictation_origin_session_id` -- the SAME value V2's own
        wrong-session-refusal already uses, NOT a fresh `store.active_
        session_id` read, which a tab switch could have moved on from by
        the time a deferred idle-branch dispatch actually runs) and a
        snapshot of every assistant message id already in that session
        (`pending_existing_assistant_ids`), both consumed by the
        reply-identity guard (`_console_hands_free_try_claim_reply`) to
        decide which later delta/completion tap call is really THIS send's
        reply.
        """
        session = self._console_hands_free
        sending_session_id = self._console_dictation_origin_session_id
        if session is not None:
            session.pending_session_id = sending_session_id
            session.pending_existing_assistant_ids = (
                self._console_hands_free_assistant_ids(sending_session_id)
            )
        self._console_pending_voice_action = "send"
        if self._console_dictation_state == "recording":
            self._request_console_dictation_stop()
            return
        if self._console_dictation_state == "idle":
            self.run_worker(
                self._run_pending_console_voice_action(sending_session_id),
                exclusive=True,
                group="console-hands-free-send",
                exit_on_error=False,
            )
        # else ("starting"/"transcribing"): a stop is already in flight for
        # this same capture; its own tail will pick up the queued action.

    def _console_hands_free_assistant_ids(
        self, session_id: str | None
    ) -> frozenset[str]:
        """Return every assistant message id currently in `session_id`.

        Used to snapshot "pre-existing" reply ids before a send, so the
        reply-identity guard can tell a brand-new reply from a stale one
        that already existed (task-5 review B1).
        """
        if not session_id:
            return frozenset()
        store = self._ensure_console_chat_store()
        try:
            messages = store.messages_for_session(session_id)
        except KeyError:
            return frozenset()
        return frozenset(m.id for m in messages if m.role == "assistant")

    def _console_hands_free_force_immediate_send(self) -> None:
        """Spoken "send" mid-loop: drive the SAME countdown-expiry path
        `RequestStopAndSend` uses, collapsed to (near) zero wall time
        (task-5 review I3).

        Only the controller's OWN public inputs are used here (`on_voice_
        final()` then two `tick()` calls) -- no reach into its private
        `_begin_awaiting_reply()`/`_send_delay_seconds` -- so this is
        exactly the path a real countdown expiry takes, just compressed:
        from `listening`, `on_voice_final()` arms the countdown; from
        `countdown` already (a prior segment's own final already armed
        one -- e.g. "hello there" dictated, then "Console, send." spoken
        immediately after, before that countdown would have expired on its
        own), the existing arming is reused as-is rather than re-armed.
        Either way, the first `tick()` adopts/re-confirms `now` as the
        anchor (elapsed 0 relative to a same-call anchor, never expires on
        its own); the second, with `now` pushed far enough into the future
        that `remaining` clamps to 0 regardless of the configured
        `dictation.handsfree_send_delay_seconds`, expires it -- which is
        what actually emits `RequestStopAndSend` and moves the controller
        into `awaiting_reply`, so the reply that follows is genuinely
        spoken instead of streaming into a `listening` loop that silently
        drops it. A no-op in every other state (`awaiting_reply`/
        `speaking`/`idle`) -- nothing to send yet, or a send is already
        outstanding.
        """
        session = self._console_hands_free
        if session is None:
            return
        controller = session.controller
        if controller.state == "listening":
            controller.on_voice_final()
        elif controller.state != "countdown":
            return
        now = time.monotonic()
        controller.tick(now)
        controller.tick(now + 3600.0)

    def _console_hands_free_silence_speech(self) -> None:
        """`SilenceSpeech`/`SuppressReplySpeech`: stop any playing speech and
        flush the sequencer.

        Two things, unconditionally: (1) the both-ways TTS stop routine
        (`TTSPlaybackEvent(action="stop")`) fires directly here too, not
        only through `flush()`'s conditional `stop_speech()` -- a `_speak_
        status` ack ("Sent.", "Discarded.", ...) bypasses the sequencer
        entirely, so without this a barge-in mid-ack would leave it
        playing (task-5 review M9); (2) `SentenceSequencer.flush()` clears
        the queue, calls `stop_speech()` (redundant with (1) when
        something is in flight -- harmless, matches the existing "post a
        bare stop before every capture-open" idiom this file already
        uses) exactly iff an utterance is in flight, and latches
        suppression so nothing from this reply speaks again.
        `SilenceSpeech` (a barge-in mid-`speaking`, or re-`enter()`
        catching a still-speaking reply) typically has something in
        flight to stop; `SuppressReplySpeech` (a keypress during
        `awaiting_reply`, or `on_reply_failed()`'s recovery) typically
        does not -- the suppression latch still needs setting either way,
        which is why both intents route here.
        """
        session = self._console_hands_free
        if session is None:
            return
        from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
            TTSPlaybackEvent,
        )

        self.app_instance.post_message(TTSPlaybackEvent(action="stop"))
        session.sequencer.flush()

    def _console_hands_free_open_capture(self) -> None:
        """`OpenCapture`: idempotent-safe via `_request_console_dictation_
        start`'s own guard -- a no-op unless `_console_dictation_state`
        is genuinely `idle`."""
        if self._console_dictation_state == "idle":
            self._request_console_dictation_start()

    def _console_hands_free_close_capture(self) -> None:
        """`CloseCapture`: idempotent-safe via `_request_console_dictation_
        stop`'s own guard -- a no-op unless genuinely `recording`."""
        if self._console_dictation_state == "recording":
            self._request_console_dictation_stop()

    def _console_hands_free_countdown_tick(self, remaining: float) -> None:
        """`CountdownTick`: record the remaining seconds for the chip."""
        session = self._console_hands_free
        if session is None:
            return
        session.countdown_remaining = remaining

    def _console_hands_free_mode_changed(self, state: str) -> None:
        """`ModeChanged`: reset per-reply state on entering `awaiting_reply`,
        seed the countdown's first-paint value on entering `countdown`,
        then repaint the chip for whatever state this is."""
        if state == "awaiting_reply":
            self._begin_console_hands_free_reply()
        elif state == "countdown":
            session = self._console_hands_free
            if session is not None:
                # Task-5 final review I1: `ModeChanged("countdown")` fires
                # from `_transition`, itself called from `on_voice_final()`
                # -- BEFORE the first real `CountdownTick` (which only
                # arrives on the next `tick()` call) ever writes `session.
                # countdown_remaining`. Without this, the repaint below
                # would show the dataclass default `0.0` on turn 1 ("sending
                # in 0.0s…", reading as "sending NOW") or the PREVIOUS
                # countdown's last value on later turns. Seeded from the
                # controller's own configured delay -- exactly what the
                # first tick would compute anyway (`remaining = send_delay
                # - 0`).
                session.countdown_remaining = session.controller._send_delay_seconds
        self._repaint_console_hands_free_chip()

    def _begin_console_hands_free_reply(self) -> None:
        """Reset per-reply state at `ModeChanged("awaiting_reply")`.

        `SentenceSequencer.begin_reply()` is REQUIRED before feeding a
        second (or later) reply's deltas on this reused sequencer instance
        -- without it, the suppression latch/fence/buffer state from the
        PRIOR reply survives, and `on_drained` never fires again, so the
        loop never reopens the microphone (see that method's docstring).
        Also clears `reply_id` (a fresh reply claims a fresh id -- see
        `_on_console_hands_free_delta`) and the per-reply toast policy.
        """
        session = self._console_hands_free
        if session is None:
            return
        session.sequencer.begin_reply()
        session.reply_id = None
        session.toast_shown_for_reply = False

    def _console_hands_free_exit_loop(self) -> None:
        """`ExitLoop`: the controller deliberately does NOT emit
        `SilenceSpeech`/`CloseCapture` alongside this intent (see
        `HandsFreeController._exit`'s callers) -- this handler performs
        both itself, in that order, before tearing the session down."""
        self._console_hands_free_silence_speech()
        self._console_hands_free_close_capture()
        self._teardown_console_hands_free_loop()

    def _repaint_console_hands_free_chip(self) -> None:
        """Paint the hands-free loop's mode into the composer's voice chip.

        `listening` RESTORES the ordinary dictation chip rather than being
        left untouched (task-5 final review I1): the ordinary pipeline
        (`VoicePartial`/`VoiceFinal`/the elapsed ticker) does paint an
        accurate "recording" chip for it on its OWN -- but only the very
        first time, before this loop has ever borrowed the chip for
        `countdown`/`awaiting_reply`/`speaking`. By the time control
        returns HERE to `listening` (a cancelled countdown, a barge-in, a
        drained reply), the chip is showing whatever borrowed text was
        painted last, and nothing else was going to overwrite it -- a
        cancelled countdown kept reading "sending in 0.0s…" for the
        user's entire next utterance (PROBE A). `composer.sync_dictation_
        state(...)`, re-applied with the composer's OWN currently-tracked
        partial/elapsed/segment-transcribing values (not this loop's),
        is the same idiom `_teardown_console_hands_free_loop` already uses
        to clear a borrowed chip on exit -- safe to call redundantly (the
        widget's own `entering_recording`/`state_changed` guards no-op
        when nothing actually changed), so calling it on every 0.1s tick
        while `listening` is cheap. The other three states either close
        the mic (default mode) or otherwise have nothing else painting
        the chip, so they are driven directly through `ConsoleComposerBar.
        set_voice_status`, which -- unlike `set_voice_partial`/`sync_
        dictation_state` -- is not gated on the one-shot dictation
        lifecycle state, so it keeps painting correctly even once
        `_console_dictation_state` has already reached `idle`.
        """
        session = self._console_hands_free
        if session is None:
            return
        composer = self._console_composer_or_none()
        if composer is None:
            return
        state = session.controller.state
        if state == "listening":
            composer.sync_dictation_state(self._console_dictation_state)
            return
        if state == "countdown":
            composer.set_voice_status(
                "countdown",
                message=(
                    f"hands-free · sending in {session.countdown_remaining:.1f}s…"
                ),
            )
        elif state == "awaiting_reply":
            composer.set_voice_status(
                "awaiting-reply", message="hands-free · thinking…"
            )
        elif state == "speaking":
            composer.set_voice_status("speaking", message="hands-free · speaking")

    def _install_console_hands_free_store_tap(self) -> None:
        """Wrap the store's creation/delta/completion seams, once, for this
        screen's life.

        `Chat/console_agent_bridge.py`'s streaming adapter (and
        `ConsoleChatController`'s own non-agent streaming path) both call
        `store.append_message`/`store.append_stream_chunk`/`store.mark_
        message_complete`/`store.mark_message_failed`/`store.mark_message_
        stopped` directly -- there is no existing observer/subscription
        mechanism on the store, so this wraps the bound methods on the
        store itself (a lazily-created singleton for this screen instance
        -- `_ensure_console_chat_store` only ever builds one). Read-only:
        every wrapper calls the original method FIRST and returns its
        result unchanged; the tap only observes. Idempotent -- installed at
        most once per screen instance, and stays installed across loop
        exit/re-entry. task-15860: it is now REMOVED at `on_unmount`
        (`uninstall_console_hands_free_store_tap`), because the store
        outlives the screen and an un-removed tap would both strand a dead
        screen and re-wrap once per Console visit. `append_message` is the
        EARLIEST of the five seams -- it fires the instant the assistant
        row is created, before any streaming (task-5 final review I3).
        """
        if self._console_hands_free_store_tap_installed:
            return
        store = self._ensure_console_chat_store()
        original_append_message = store.append_message
        original_append = store.append_stream_chunk
        original_complete = store.mark_message_complete
        original_failed = store.mark_message_failed
        original_stopped = store.mark_message_stopped

        def _append_message(*args: Any, **kwargs: Any):
            result = original_append_message(*args, **kwargs)
            # Task-5 final review I3: the EARLIEST observable "generation
            # truly begins" signal -- filtered to ASSISTANT rows here,
            # before the marshal, so a user/system append (far more
            # frequent) never pays a cross-thread round trip for nothing.
            # See `_on_console_hands_free_assistant_row_created`.
            if result.role is ConsoleMessageRole.ASSISTANT:
                self._console_hands_free_marshal(
                    self._on_console_hands_free_assistant_row_created,
                    result.id,
                )
            return result

        def _append_stream_chunk(message_id: str, chunk: str):
            result = original_append(message_id, chunk)
            self._console_hands_free_marshal(
                self._on_console_hands_free_delta, message_id, chunk
            )
            return result

        def _mark_message_complete(message_id: str):
            result = original_complete(message_id)
            self._console_hands_free_marshal(
                self._on_console_hands_free_terminal, message_id, False
            )
            return result

        def _mark_message_failed(message_id: str):
            result = original_failed(message_id)
            self._console_hands_free_marshal(
                self._on_console_hands_free_terminal, message_id, True
            )
            return result

        def _mark_message_stopped(message_id: str):
            result = original_stopped(message_id)
            self._console_hands_free_marshal(
                self._on_console_hands_free_terminal, message_id, True
            )
            return result

        wrappers = {
            "append_message": _append_message,
            "append_stream_chunk": _append_stream_chunk,
            "mark_message_complete": _mark_message_complete,
            "mark_message_failed": _mark_message_failed,
            "mark_message_stopped": _mark_message_stopped,
        }
        # task-15860: remember what to put back. The store is app-owned and
        # now OUTLIVES this screen, so a tap left installed would (a) keep a
        # dead screen alive through five closures and (b) be wrapped AGAIN
        # by the next visit's controller -- one nesting level per Console
        # visit, forever. `on_unmount` calls
        # `uninstall_console_hands_free_store_tap`.
        self._console_hands_free_store_tap_undo = (
            store,
            {name: (name in store.__dict__, getattr(store, name)) for name in wrappers},
            wrappers,
        )
        for name, wrapper in wrappers.items():
            setattr(store, name, wrapper)
        self._console_hands_free_store_tap_installed = True

    def uninstall_console_hands_free_store_tap(self) -> None:
        """Restore the five store seams this screen wrapped, if it wrapped them.

        Idempotent, and conservative: a seam that no longer holds THIS
        screen's wrapper (something else re-wrapped it afterwards) is left
        exactly as it is rather than clobbered.
        """
        undo = getattr(self, "_console_hands_free_store_tap_undo", None)
        self._console_hands_free_store_tap_undo = None
        self._console_hands_free_store_tap_installed = False
        if undo is None:
            return
        store, originals, wrappers = undo
        for name, wrapper in wrappers.items():
            if store.__dict__.get(name) is not wrapper:
                continue
            had_own, original = originals[name]
            if had_own:
                setattr(store, name, original)
            else:
                try:
                    delattr(store, name)
                except AttributeError:  # pragma: no cover - already gone
                    pass

    def _console_hands_free_marshal(
        self, callback: Callable[..., None], *args: Any
    ) -> None:
        """Run `callback(*args)` on the UI thread (task-5 review I1).

        The tap wraps store methods reachable from TWO contexts: the
        async, on-the-app-loop direct-provider send path, and -- the
        DEFAULT production path, `[console] agent_runtime` on by default
        -- a WORKER THREAD running its own event loop
        (`ConsoleChatController._run_agent_reply` ->
        `asyncio.to_thread(bridge.run_reply)` ->
        `console_agent_bridge.py`'s `_StreamingModelAdapter.chat_call` ->
        `store.append_stream_chunk`/etc, all on that thread). Everything
        downstream of this tap touches widgets (`ConsoleComposerBar.set_
        voice_status` via the chip repaint) or calls `run_worker`/`post_
        message` (which themselves require -- or at least assume -- the
        app's own thread), so every call must land back there. `self.app`
        itself is UNSAFE to read off-thread (it resolves via a context
        var with no active app on a bare worker thread -- `NoActiveAppError`)
        -- `self.app_instance` (the stored `TldwCli` reference, plain
        attribute access, safe from any thread) is used for both the
        thread-identity check and the marshal call.

        Task-5 review round 2, D1: the tap is installed once and never
        uninstalled (see `_install_console_hands_free_store_tap`), so
        EVERY streamed chunk of EVERY message pays for this call, hands-
        free running or not -- the fast path below (bail out before the
        thread-identity check, let alone a real `call_from_thread` round
        trip, when the loop is not running) is what keeps the common
        case (hands-free never entered this session) cheap: measured
        ~60us/chunk down to near-zero. Also: `App.call_from_thread` raises
        `RuntimeError("App is not running")` when the app has no running
        event loop (e.g. the standard test harness, where `app_instance`
        is a `TldwCli` that was never `run()`) -- wrapped in `except
        Exception` so a hands-free plumbing/timing issue can NEVER escape
        into `store.append_message`/`append_stream_chunk`/`mark_message_
        *`, which every reply -- hands-free or not -- streams through.

        Task-5 final review I2: the UI-thread branch used to call
        `callback(*args)` bare -- the "NEVER escape" claim two paragraphs
        up was therefore false on the (also supported, non-agent-runtime)
        direct-provider path, which calls this tap from the UI thread
        directly. Both branches now share the identical guarantee.
        """
        if self._console_hands_free is None:
            return
        if threading.get_ident() == self.app_instance._thread_id:
            try:
                callback(*args)
            except Exception:
                logger.opt(exception=True).warning(
                    "Console hands-free: tap callback failed on the UI "
                    "thread; dropping this callback"
                )
            return
        try:
            self.app_instance.call_from_thread(callback, *args)
        except Exception:
            logger.opt(exception=True).warning(
                "Console hands-free: tap marshal failed off-thread; "
                "dropping this callback"
            )

    def _console_hands_free_try_claim_reply(
        self, session: "ConsoleHandsFreeSession", message_id: str
    ) -> bool:
        """Claim `message_id` as the outstanding reply's id, if eligible.

        REPLY IDENTITY (binding carrier, task-5 review B1): eligible means
        ALL of:
          (a) `controller.state == "awaiting_reply"` -- nothing has been
              claimed yet, and a reply is genuinely outstanding;
          (b) `message_id` belongs to the SAME session `RequestStopAndSend`
              recorded this send as going into (`session.pending_session_
              id`) -- rules out a concurrently-streaming BACKGROUND
              session's reply (parallel per-session runs are a supported
              feature, `console_chat_controller.py`'s `send_refusal_copy`
              gates on `max_parallel_runs`, not "only one run app-wide");
          (c) `message_id` was NOT already present in that session before
              this send (`session.pending_existing_assistant_ids`) -- rules
              out a STALE reply in the SAME session: a keyboard barge-in
              during `awaiting_reply` suppresses speech but never cancels
              generation, so that reply's message id already exists (and
              keeps streaming) by the time the NEXT turn's send fires; without
              this check the next turn would claim the OLD reply's id (it is
              the first one still streaming) and speak its leftover sentences
              into the new turn -- reopening exactly the hazard task-3's
              `on_reply_started` docstring warns about
              (`console_hands_free.py`'s `_reply_abandoned_by_watchdog`
              framing is the same class of "a suppressed reply must not
              resurrect" issue, one layer up).

        There is still no independent ground truth beyond "new, in the
        right session" -- there is no earlier synchronous "reply started,
        here is its id" signal reachable without touching
        `console_chat_controller.py` (out of this task's file list) -- but
        that pair of checks is exactly what the brief's reply-identity
        constraint requires: a wrong id can no longer win the slot.
        """
        if session.controller.state != "awaiting_reply":
            return False
        if session.pending_session_id is None:
            return False
        store = self._ensure_console_chat_store()
        try:
            owner_session_id = store.session_id_for_message(message_id)
        except KeyError:
            return False
        if owner_session_id != session.pending_session_id:
            return False
        if message_id in session.pending_existing_assistant_ids:
            return False
        session.reply_id = message_id
        return True

    def _on_console_hands_free_assistant_row_created(self, message_id: str) -> None:
        """`store.append_message`'s ASSISTANT-role tap (task-5 final review
        I3): the EARLIEST observable "generation truly begins" signal.

        Before this, the only `on_reply_started()` call sites were the
        first streamed delta and the terminal tap -- both downstream of
        the model actually producing VISIBLE output. On the DEFAULT
        agent-runtime path, `console_agent_bridge.py`'s streaming adapter
        only forwards fence-gated PRIMARY-turn output, so a run that does
        tool round-trips first (or opens with a fenced code block the
        sequencer skips by design, or is a sealed/non-streaming turn)
        could blow the `awaiting_reply` watchdog's `AWAITING_REPLY_
        DEADLINE_SECONDS` before a single visible token arrived -- the
        FSM's own docstring calls that "routine," not exceptional, and
        promises the watchdog does not punish it. `store.append_message`
        creates the assistant row ONCE, synchronously, before any
        streaming (agent or not) begins -- reusing it here makes the
        watchdog measure what its docstring actually says it measures
        (send -> `on_reply_started()`), not send -> first visible token.

        Reuses the SAME reply-identity guard the delta/completion taps
        use (`_console_hands_free_try_claim_reply`): a claim made here
        for the WRONG session or a pre-existing id is refused exactly
        like a claim from a delta would be, so this cannot weaken the B1
        guarantee -- a non-assistant append never reaches here at all
        (filtered in the wrapper, before the marshal), and an assistant
        append for an unrelated/background session's OWN reply fails the
        SAME session+novelty checks a delta for it would.
        """
        session = self._console_hands_free
        if session is None or session.reply_id is not None:
            return
        if self._console_hands_free_try_claim_reply(session, message_id):
            session.controller.on_reply_started()

    def _on_console_hands_free_delta(self, message_id: str, chunk: str) -> None:
        """Delta tap: feed one streamed chunk into the loop's sentence sequencer.

        UI-thread only -- always reached via `_console_hands_free_marshal`
        (task-5 review I1). The FIRST delta that passes `_console_hands_
        free_try_claim_reply` (see its docstring for the full reply-identity
        contract) claims `session.reply_id` for this turn, and doubles as
        the earliest available `on_reply_started()` signal. Every later
        delta is fed ONLY when its `message_id` matches `session.reply_id`;
        anything else is dropped before it can reach the sequencer.
        """
        session = self._console_hands_free
        if session is None:
            return
        if session.reply_id is None:
            if not self._console_hands_free_try_claim_reply(session, message_id):
                return
            session.controller.on_reply_started()
        elif message_id != session.reply_id:
            return
        session.sequencer.feed(chunk)

    def _on_console_hands_free_terminal(self, message_id: str, failed: bool) -> None:
        """Completion tap: `mark_message_complete`/`mark_message_failed`/
        `mark_message_stopped`.

        UI-thread only -- always reached via `_console_hands_free_marshal`
        (task-5 review I1; `failed` is positional here, not keyword-only,
        for that same call site's benefit -- `call_from_thread`/direct
        dispatch both pass it positionally). Claims `session.reply_id` via
        `_console_hands_free_try_claim_reply` the same way the delta tap
        does when it has not already been claimed -- a reply that streams
        ZERO chunks (a zero-speakable reply, or a failure before any
        content arrived) still needs its completion recognized, or the
        loop hangs in `awaiting_reply` until the 30s watchdog gives up on
        it. Dropped when `message_id` does not match the outstanding reply.
        """
        session = self._console_hands_free
        if session is None:
            return
        if session.reply_id is None:
            if not self._console_hands_free_try_claim_reply(session, message_id):
                return
        elif session.reply_id != message_id:
            return
        if failed:
            session.sequencer.flush()
            session.controller.on_reply_failed()
            return
        session.controller.on_reply_started()
        session.sequencer.reply_completed()
        session.controller.on_reply_finished()

    def _dispatch_console_hands_free_speak(self, text: str) -> None:
        """`SentenceSequencer`'s `speak` callable: dispatch one utterance.

        Synchronous (the sequencer's contract) -- schedules the actual
        async `speak_utterance` call as a worker. Also this wiring's only
        call site for `HandsFreeController.on_first_utterance()`: safe on
        EVERY dispatch (idempotent -- a no-op outside `awaiting_reply`, see
        that method's docstring), so no separate first-utterance flag is
        needed here.
        """
        session = self._console_hands_free
        if session is None:
            return
        session.controller.on_first_utterance()
        token = session.sequencer.current_utterance_token
        self.run_worker(
            self._speak_console_hands_free_utterance(text, token),
            exclusive=False,
            group="console-hands-free-speech",
            exit_on_error=False,
        )

    async def _speak_console_hands_free_utterance(
        self, text: str, token: int | None
    ) -> None:
        """Speak one utterance via the cooldown-free `speak_utterance` entry.

        `token` is `session.sequencer.current_utterance_token`, captured
        synchronously at dispatch time (binding carrier: production callers
        MUST thread it through into `utterance_finished(ok, token=...)` --
        see that method's docstring). `quiet` implements the "at most one
        failure toast per reply" policy: the first failed utterance in a
        reply shows its toast and latches `toast_shown_for_reply`; every
        later utterance in the SAME reply then passes `quiet=True` and only
        logs.
        """
        session = self._console_hands_free
        if session is None:
            return
        handler = await self.app_instance._ensure_tts_handler()
        if handler is None:
            session.sequencer.utterance_finished(False, token=token)
            return
        quiet = session.toast_shown_for_reply

        def _on_finished(ok: bool) -> None:
            current = self._console_hands_free
            if current is not session:
                # A different loop entry (or none at all) owns the screen's
                # hands-free state now; this utterance's own sequencer/token
                # bookkeeping is no longer live to report back into.
                return
            if not ok:
                session.toast_shown_for_reply = True
            session.sequencer.utterance_finished(ok, token=token)

        await handler.speak_utterance(text, on_finished=_on_finished, quiet=quiet)

    def _stop_console_hands_free_speech(self) -> None:
        """`SentenceSequencer`'s `stop_speech` callable: the existing
        both-ways stop routine (silences BOTH the streaming sink and the
        legacy player) -- see `_request_console_dictation_start`'s
        identical use for the mic/speaker exclusion invariant."""
        from tldw_chatbook.Event_Handlers.TTS_Events.tts_events import (
            TTSPlaybackEvent,
        )

        self.app_instance.post_message(TTSPlaybackEvent(action="stop"))

    def _on_console_hands_free_sequencer_drained(self) -> None:
        """`SentenceSequencer.on_drained`: nothing left queued or in flight."""
        session = self._console_hands_free
        if session is None:
            return
        session.controller.on_sequencer_drained()

    async def _deliver_console_hands_free_capture_ended(
        self, scheduled_for: "ConsoleHandsFreeSession", had_segments: bool
    ) -> None:
        """Wait for a limit-triggered stop to actually release the
        microphone, THEN deliver `on_capture_ended` (task-5 review B2).

        Polls `_console_dictation_state` rather than hooking `_stop_
        console_dictation`'s own internals directly, so this stays a
        self-contained addition next to the ONE call site that needs it
        instead of threading a new "was this a limit stop" flag through
        that already-delicate method.

        Task-5 review round 2, D4: `scheduled_for` is the session object
        captured by the CALLER at schedule time (while the limit-hit
        capture was still live), not re-read from `self._console_hands_
        free` after the wait. Re-reading was wrong: if the user exits and
        re-enters hands-free during the (up to 40s) wait, `self._console_
        hands_free` now points at a brand-new session/controller for a
        DIFFERENT loop -- delivering this stale capture-ended to it would
        silently burn the new loop's own one-time reopen ceiling for an
        ending that has nothing to do with it. Delivered only when the
        CURRENT session is still identically the one this was scheduled
        for.
        """
        deadline = (
            time.monotonic() + self._CONSOLE_HANDS_FREE_CAPTURE_ENDED_WAIT_SECONDS
        )
        while self._console_dictation_state != "idle":
            if not self.is_mounted or time.monotonic() >= deadline:
                logger.warning(
                    "Console hands-free: capture-ended delivery gave up "
                    "waiting for the limit-triggered stop to reach idle"
                )
                return
            await asyncio.sleep(0.05)
        if self._console_hands_free is not scheduled_for:
            return
        scheduled_for.controller.on_capture_ended(
            had_segments=had_segments, limit_hit=True
        )

    def _console_pipeline_hands_free_blocker(self) -> str | None:
        """Why the V3 pipeline loop is not a usable fallback, or None.

        Two blockers, both already defined elsewhere in this screen: a
        degraded VAD (the pipeline loop cannot auto-send at all -- see
        `_console_hands_free_vad_degraded`), and dictation being
        unavailable outright (no capture backend or no speech provider).

        `Availability.remedy` is included when there is one (final review
        M7): this string ends up in the toast that reports BOTH engines
        failing, which is the only place the user is told what to install
        -- dropping it left them with a diagnosis and no fix.
        """
        if self._console_hands_free_vad_degraded:
            return "voice-activity detection is unavailable, so auto-send cannot work"
        try:
            availability = console_voice_input.probe()
        except Exception:  # noqa: BLE001 - a probe crash is not a refusal
            logger.opt(exception=True).debug(
                "Console realtime: dictation availability probe crashed"
            )
            return None
        if not availability.ok:
            reason = availability.reason or "dictation is unavailable"
            remedy = str(availability.remedy or "").strip()
            return f"{reason} {remedy}".strip() if remedy else reason
        return None

    def action_exit_console_hands_free(self) -> None:
        """Priority Esc: exit the hands-free loop from any point (task-5
        review I2) -- see `check_action`'s gate and the `BINDINGS` entry's
        docstring-comment for why this needs to be `priority=True` rather
        than relying on `on_key`'s own (bubbling-order) branch alone.

        Covers BOTH engines (V4 task 5): "Esc from any point in the loop"
        is a promise the docs make about hands-free, not about one
        engine's implementation of it.
        """
        hands_free = self._console_hands_free
        if hands_free is not None:
            hands_free.controller.on_exit_request()
        realtime = self._console_realtime
        if realtime is not None:
            realtime.controller.on_exit_request()

    def console_hands_free_exit_available(self) -> bool:
        """`ChatScreen.check_action`'s `exit_console_hands_free` branch,
        moved: whether EITHER engine is running, so the priority-Esc
        binding (`action_exit_console_hands_free` above) only lights up
        while there is something for it to exit."""
        return (
            self._console_hands_free is not None or self._console_realtime is not None
        )

    def teardown(self) -> None:
        """Abandon the pipeline loop's timer during screen unmount.

        Moved out of `ChatScreen.on_unmount` (wave-2 console decomposition,
        task 1), where this was two inline statements; `on_unmount` now
        calls this as one line, mirroring `ConsoleDictationController.
        teardown`'s own precedent. Direct timer stop + state drop, not the
        full `ExitLoop` intent path: unmount is abandon teardown (V2-style,
        per the design doc's error-handling section), not a graceful exit --
        no further TTS/dictation calls are safe to issue against a screen
        that is being torn down.
        """
        hands_free = self._console_hands_free
        if hands_free is not None and hands_free.tick_timer is not None:
            hands_free.tick_timer.stop()
        self._console_hands_free = None

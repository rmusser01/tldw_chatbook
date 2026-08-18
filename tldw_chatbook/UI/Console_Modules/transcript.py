"""Console shell transcript region — the main column and its transcript surface.

Extracted verbatim out of ``ChatScreen.compose_content`` (wave-3 console
decomposition, task 2): the subtree that used to be built inline as

    main_column = Vertical(id="console-main-column")
    with main_column:
        transcript_region = self._frame_console_region(
            Vertical(id="console-transcript-region", classes="console-region"),
            top=False,
        )
        with transcript_region:
            yield self._ensure_console_session_surface()

Ids and nesting inside this subtree are preserved exactly, and this class
reuses ``id="console-main-column"`` as its own root, so it sits in the DOM
exactly where the old plain ``Vertical`` sat — the same "reuse the outer id
as the region root" shape ``ConsoleLeftRail``/``ConsoleInspectorRail`` used
in wave 1. ``#console-main-column`` is pinned by the painted-geometry
baseline (``Tests/UI/test_console_shell_regions.py``) at all three sizes, so
that placement is not a stylistic preference.

**Naming**: the DOM here spells two things, an outer column and an inner
framed region, and the whole column exists to hold the transcript — there is
no other content in it. So the class is named for what the column is FOR
(``ConsoleTranscriptRegion``) while the ids stay exactly as they are, the
same judgement call ``ConsoleInspectorRail`` made when the plan's placeholder
name disagreed with the DOM.

**The sizing stays on the screen.** ``width``/``min_width``/``min_height``
are still assigned by ``compose_content`` on the constructed instance, not in
here — they are facts about this widget's place among its ``#console-
workspace-grid`` siblings (``3fr`` left rail, ``13fr`` main column, ``4fr``
inspector), not about the transcript. Both rails are wired the same way.

**What did NOT move, and why.** By the rule wave 1 settled (a body that
reaches beyond the region's own DOM stays on the screen; screen-side
``query_one`` into a region's ids crosses the compound-widget boundary
transparently), these all stay on ``ChatScreen`` and keep working unmodified:

- ``_sync_native_console_transcript`` (124 lines) — the transcript's render
  pump. Every DOM write it makes is region-local, but its body also reads
  and MUTATES three other clusters' state (``_console_original_attempt_
  previews`` — owned by ``ConsoleMessageController`` since task 1;
  ``_console_image_preparing`` — the image-generation cluster;
  ``_console_citation_counts`` — the citation cluster), dispatches the
  ``console-image-prep`` worker, and calls ``_sync_console_transcript_
  guidance`` (which reaches ``#console-setup-modal``, outside this region).
  A region that mutated three sibling clusters' mutable state through
  accessor callables would be a controller wearing a region's clothes.
- ``_native_console_transcript_fingerprint`` — pure, no DOM, sole caller is
  the method above.
- ``_start_console_transcript_sync_timer`` / ``_stop_...`` — named for the
  transcript, but the 0.2 s tick pumps ``_sync_native_console_chat_ui`` (tab
  glyphs, rails, chips), and the timer handle is screen-lifecycle state
  (``on_unmount`` stops it).
- ``_sync_console_transcript_guidance`` and ``_sync_console_native_session_
  tabs`` — both query ``#console-session-surface`` (inside this region) but
  also reach the setup modal / the store / the session+workspace
  controllers.
- ``_ensure_console_session_surface`` — owns ``ChatScreen.console_session_
  surface``, a public attribute other screen code reads (the fleet
  coach-mark path). This region takes a zero-arg builder instead; see
  ``__init__``.
- ``_selected_console_message_inspector_rows``, ``_console_change_review_
  run_id``, ``_sync_console_pending_delete_confirmation``, ``on_key``,
  ``handle_console_citation_sources`` — each reads this region's transcript
  but exists to serve another surface (Inspector rows, the change-review
  screen, the message controller's pending-delete state, screen-wide key
  routing, the citation modal).
- ``_clear_native_console_message_selection`` — the near-miss worth
  recording. Its first four lines are pure region-local DOM, and it reads
  like the third sibling of the two reading-state methods below; its
  remaining three clear ``_pending_console_delete_message_id`` (a
  ``ConsoleMessageController`` proxy since task 1), reset the screen's
  ``_last_native_transcript_refresh_key`` render gate, and call the
  guidance sync. Verdict flipped from "moves" to "stays" only once the
  whole body was read, which is the argument for reading whole bodies.

**No ``@on`` handlers moved.** Every event that originates inside this
region and is handled today (``.console-transcript-citation-sources``,
``#console-fleet-coachmark-dismiss``, the session tab buttons in
``on_button_pressed``) has a body that reaches beyond it. This region stops
nothing, so all of them keep bubbling to the screen exactly as before —
matching ``ConsoleInspectorRail``, which likewise owns pixels and zero
handlers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.css.query import QueryError
from textual.widget import Widget

from ...Widgets.Console import ConsoleTranscript
from .frame import frame_console_region


@dataclass(frozen=True)
class _ConsoleTranscriptReadingState:
    """The transcript's semantic reading position across a layout change.

    Moved verbatim from ``chat_screen.py`` (which imports it back from here
    for ``_finish_console_composer_layout_change``'s signature — that
    annotation is evaluated at class-creation time, so the name is a real
    runtime dependency there, not a typing-only one).
    """

    anchored: bool
    scroll_y: float
    selected_message_id: str | None


class ConsoleTranscriptRegion(Vertical):
    """The Console shell's main column: the framed transcript surface.

    Composes nothing of its own beyond the two containers the screen used to
    build inline; the transcript itself lives inside ``ConsoleSessionSurface``
    (``Widgets/Console/console_session_surface.py``), an existing leaf widget
    this region places rather than absorbs — a region is a one-place
    composition of leaf widgets, not a relocation of them (DESIGN.md §7,
    "The One Home Rule"). Nothing here reaches ``app_instance``.
    """

    def __init__(
        self,
        *,
        session_surface_builder: Callable[[], Widget],
        **kwargs,
    ) -> None:
        """Create the main column around a session-surface builder.

        Args:
            session_surface_builder: Zero-arg callable that returns the
                Console session surface to mount inside
                ``#console-transcript-region``. A builder, not a pre-built
                instance, is the point — though for a reason narrower than
                ``ConsoleLeftRail``'s avatar, and worth stating precisely so
                nobody over-trusts it: the screen MEMOISES this widget on
                ``ChatScreen.console_session_surface``, so a builder and a
                stored instance would re-yield the very same object, and the
                builder buys no protection against re-yielding a widget
                Textual has already removed. What it does buy is the
                re-sync: ``_ensure_console_session_surface`` re-applies the
                current background-effect settings on every call, which a
                stored instance would silently skip. Late-binding at CALL
                time, matching ``ConsoleDictationController``'s constructor
                rule (see ``dictation.py``'s module docstring).
            kwargs: Forwarded to ``Vertical``.
        """
        super().__init__(id="console-main-column", **kwargs)
        self._session_surface_builder = session_surface_builder

    def compose(self) -> ComposeResult:
        """Compose the framed transcript region and the session surface.

        Returns:
            The framed ``#console-transcript-region`` container holding the
            Console session surface. ``top=False`` is deliberate: the control
            bar directly above already paints that edge, so the transcript
            reads as continuous with it instead of drawing a doubled rule.
        """
        transcript_region = frame_console_region(
            Vertical(id="console-transcript-region", classes="console-region"),
            top=False,
            # TASK-17651: the workspace grid's own bottom border is the
            # bottom stack's single separator; the region ends flush.
            bottom=False,
        )
        with transcript_region:
            yield self._session_surface_builder()

    def _transcript_or_none(self) -> ConsoleTranscript | None:
        """Return the mounted transcript widget, or ``None`` when absent.

        Every method below opened with this same try/query/except before the
        move; it is factored out here rather than repeated four times.

        Returns:
            The ``#console-native-transcript`` widget inside this region, or
            ``None`` before it is mounted (or after it is torn down).
        """
        try:
            return self.query_one("#console-native-transcript", ConsoleTranscript)
        except QueryError:
            return None

    def capture_reading_state(self) -> _ConsoleTranscriptReadingState | None:
        """Capture the semantic reading position before composer layout changes.

        Moved verbatim from ``ChatScreen._capture_console_transcript_reading_
        state``.

        Returns:
            The transcript's tail-follow state, scroll offset and selected
            message id, or ``None`` when no transcript is mounted.
        """
        transcript = self._transcript_or_none()
        if transcript is None:
            return None
        return _ConsoleTranscriptReadingState(
            anchored=bool(
                transcript.is_anchored
                and not getattr(transcript, "_anchor_released", False)
            ),
            scroll_y=float(transcript.scroll_y),
            selected_message_id=transcript.selected_message_id,
        )

    def restore_reading_state(
        self,
        state: _ConsoleTranscriptReadingState | None,
    ) -> None:
        """Restore the transcript anchor, offset, and selected message.

        Moved verbatim from ``ChatScreen._restore_console_transcript_reading_
        state``.

        Args:
            state: A previously captured reading state; ``None`` is a no-op,
                which is what callers pass when the capture found no
                transcript.
        """
        if state is None:
            return
        transcript = self._transcript_or_none()
        if transcript is None:
            return
        revealed = False
        if state.selected_message_id is not None:
            # TASK-15455: assigning the id directly bypasses `select_message`,
            # so a selection captured before the transcript re-windowed (a
            # session switch between capture and restore) could name a message
            # with no mounted row. Inert whenever the id is already mounted.
            revealed = transcript.reveal_message(state.selected_message_id)
        transcript.selected_message_id = state.selected_message_id

        def _apply_reading_position() -> None:
            if state.anchored:
                transcript.anchor()
                return
            transcript.release_anchor()
            transcript.scroll_to(
                y=min(state.scroll_y, float(transcript.max_scroll_y)),
                animate=False,
            )

        if revealed:
            # Read order matters: the offset is clamped against
            # `max_scroll_y`, which only grows once the revealed rows are
            # mounted. Applying it first silently drops the reader at the
            # pre-reveal maximum instead of where they were.
            transcript.call_later(transcript.refresh_messages)
            transcript.call_after_refresh(_apply_reading_position)
            return
        _apply_reading_position()

    def note_follow_intent(self) -> None:
        """Stamp a programmatic jump-to-tail intent on the transcript (TASK-336).

        Moved verbatim from ``ChatScreen._note_console_follow_intent``,
        including its finding: this stays singular/view-only on purpose.
        Unlike the screen's per-session stash maps, it never carries
        session-owned DATA across a send's lifetime — it is a one-shot
        directive consumed by whichever session's transcript happens to be
        ``#console-native-transcript`` (a single widget instance reflecting
        the ACTIVE session) at the next render. A background session's send
        stamping this while a different session is viewed just requests an
        extra, harmless tail-follow on whatever the transcript renders next;
        there is no cross-session data to leak or clobber.
        """
        transcript = self._transcript_or_none()
        if transcript is None:
            return
        transcript.note_follow_intent()

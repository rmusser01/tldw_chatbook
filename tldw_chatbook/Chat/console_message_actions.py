"""Pure selected-message action contracts for the native Console transcript."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Literal

from markdown_it import MarkdownIt

from tldw_chatbook.Canvas.compiler import CanvasCompileError, compile_canvas_document
from tldw_chatbook.Chat.console_chat_fork import ConsoleForkEligibility
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_ephemeral import blocked_reason

ConsoleActionStatus = Literal[
    "completed",
    "wip",
    "blocked",
    "continue_requested",
    "edit_requested",
    "fork_requested",
    "canvas_open_requested",
    "canvas_repair_requested",
]
ConsoleSpeechPresentationState = Literal[
    "idle",
    "generating",
    "playing",
    "stopped",
    "failed",
]


@dataclass(frozen=True)
class ConsoleMessageAction:
    """One visible action in the selected-message action row."""

    action_id: str
    label: str
    enabled: bool = True
    disabled_reason: str = ""


@dataclass(frozen=True, slots=True)
class ConsoleCanvasHtmlBlock:
    """One parsed assistant HTML fence eligible for a Canvas action."""

    index: int
    identity: str
    html: str
    compatible: bool
    compatibility_codes: tuple[str, ...] = ()


def assistant_canvas_html_blocks(
    message: ConsoleChatMessage,
) -> tuple[ConsoleCanvasHtmlBlock, ...]:
    """Parse Canvas-eligible HTML fences without inspecting rendered Markdown."""

    if (
        message.role is not ConsoleMessageRole.ASSISTANT
        or message.status != "complete"
    ):
        return ()
    blocks: list[ConsoleCanvasHtmlBlock] = []
    for token in MarkdownIt("commonmark").parse(message.content):
        language = (
            token.info.strip().split(maxsplit=1)[0].casefold() if token.info else ""
        )
        if token.type != "fence" or language != "html":
            continue
        index = len(blocks)
        codes: tuple[str, ...] = ()
        compatible = True
        try:
            plan = compile_canvas_document(token.content)
            codes = tuple(issue.code for issue in plan.compatibility_issues)
        except CanvasCompileError as exc:
            codes = tuple(issue.code for issue in exc.issues)
            compatible = False
        blocks.append(
            ConsoleCanvasHtmlBlock(
                index=index,
                identity=f"{message.id}:canvas-html:{index}",
                html=token.content,
                compatible=compatible,
                compatibility_codes=codes,
            )
        )
    return tuple(blocks)


def canvas_block_origin_turn_id(
    message: ConsoleChatMessage,
    block_index: int,
) -> str:
    """Return a restart-stable, source-free identity for one Canvas import.

    A hydrated persisted turn identity is preferred when the message owns one.
    Ordinary persisted assistant rows currently do not, so their persisted
    message id plus parsed block position is the deterministic fallback.
    Hashing keeps the storage-facing identifier bounded and prevents either
    identity from leaking into incidental diagnostics. Temporary messages
    remain scoped to their in-memory turn/message identity and session lifecycle.
    """

    if message.persisted_message_id is not None:
        stable_owner = message.trace_turn_id or message.persisted_message_id
        digest = sha256(
            f"{stable_owner}\0{block_index}".encode("utf-8")
        ).hexdigest()
        return f"canvas-import-{digest}"
    return message.turn_id or message.trace_turn_id or message.id


def resolve_canvas_html_block(
    message: ConsoleChatMessage, reference: ConsoleCanvasBlockReference
) -> ConsoleCanvasHtmlBlock | None:
    """Resolve one exact parsed block at the immediate trusted consumer seam."""

    if message.id != reference.message_id:
        return None
    return next(
        (
            block
            for block in assistant_canvas_html_blocks(message)
            if block.index == reference.block_index
            and block.identity == reference.identity
        ),
        None,
    )


@dataclass(frozen=True, slots=True)
class ConsoleMessageActionGroups:
    """Stable direct, overflow, and media action groups for one row."""

    primary: tuple[ConsoleMessageAction, ...]
    overflow: tuple[ConsoleMessageAction, ...]
    media: tuple[ConsoleMessageAction, ...]


@dataclass(frozen=True)
class ConsoleHeaderSpeechPresentation:
    """Visible speech action and bounded lifecycle copy for one message header."""

    action: ConsoleMessageAction | None
    status_label: str = ""


def _speech_visible(message: ConsoleChatMessage) -> bool:
    """Return whether a message may expose trusted Manual Speak."""
    return (
        message.role is ConsoleMessageRole.ASSISTANT
        and message.status == "complete"
        and bool(message.content.strip())
    )


def resolve_console_header_speech(
    message: ConsoleChatMessage,
    state: ConsoleSpeechPresentationState,
    *,
    selected: bool = False,
) -> ConsoleHeaderSpeechPresentation:
    """Resolve the header speech presentation for a Console message.

    The header never hosts the idle Speak action: speak lives in the
    selected-message action row with the other per-message options. The
    header shows only active-playback lifecycle status (generating/playing)
    and its terminal states (stopped/failed), which must stay visible even
    when the message is deselected so playback remains controllable.
    """
    if not _speech_visible(message):
        return ConsoleHeaderSpeechPresentation(action=None)
    if state == "idle":
        return ConsoleHeaderSpeechPresentation(action=None)
    if state == "generating":
        return ConsoleHeaderSpeechPresentation(
            action=ConsoleMessageAction(
                "speak-stop",
                "⏹",
                enabled=False,
                disabled_reason="Speech audio is being generated.",
            ),
            status_label="Generating",
        )
    if state == "playing":
        return ConsoleHeaderSpeechPresentation(
            action=ConsoleMessageAction("speak-stop", "⏹"),
            status_label="Playing",
        )
    if state == "stopped":
        return ConsoleHeaderSpeechPresentation(
            action=ConsoleMessageAction("speak", "🔊"),
            status_label="Stopped",
        )
    if state == "failed":
        return ConsoleHeaderSpeechPresentation(
            action=ConsoleMessageAction("speak", "🔊"),
            status_label="Failed",
        )
    return ConsoleHeaderSpeechPresentation(action=ConsoleMessageAction("speak", "🔊"))


@dataclass(frozen=True)
class ConsoleCanvasBlockReference:
    """Source-free identity resolved only by the immediate Console consumer."""

    message_id: str
    block_index: int
    identity: str
    create_new: bool


@dataclass(frozen=True)
class ConsoleActionResult:
    """Result of dispatching a Console selected-message action."""

    action_id: str
    status: ConsoleActionStatus
    visible_copy: str
    clipboard_text: str | None = None
    target_message_id: str | None = None
    target_content: str | None = None
    target_invocation_id: str | None = None
    canvas_block_ref: ConsoleCanvasBlockReference | None = None


@dataclass(frozen=True)
class ConsoleSaveDestination:
    """One Save as destination shown in the Console save modal."""

    label: str
    available: bool
    reason: str = ""


#: task-2154.14 (DS-01): legend segments for the row under a selected
#: message, naming each glyph-only button in words so the meaning is on
#: screen instead of behind a tooltip. Text-labeled buttons (Save as...,
#: Full output, Review, Try, keep) already name themselves and are omitted.
#: The key hints (c/e/r) mirror ConsoleTranscript.BINDINGS.
ACTION_GUIDE_SEGMENTS: tuple[tuple[str, str], ...] = (
    ("copy", "c Copy"),
    ("speak", "🔊 Speak"),
    ("speak-stop", "⏹ Stop speech"),
    ("edit", "e Edit"),
    ("fork", "f Fork"),
    ("regenerate", "r ♻ Regenerate"),
    ("continue", "---> Continue"),
    ("feedback", "👍/👎 Rate"),
    ("delete", "🗑 Delete"),
    ("variant-previous", "</> Variants"),
    ("video-play", "▶ Play"),
    ("video-save-copy", "Save copy"),
)


def action_row_guide(actions: list[ConsoleMessageAction]) -> str:
    """Build the always-visible legend for a selected message's action row.

    The legend is derived from the row's ACTUAL actions, in button order, so
    a row without Speak never names a 🔊 the user cannot see and the
    speak-stop swap reads "⏹ Stop speech" instead of pointing at a 🔊 that
    is not there. Only glyph-only buttons need naming (DS-01); the key
    hints and j/k/Esc framing come from task-362's static guide, which this
    replaces.

    Args:
        actions: The row's actions as returned by
            ``ConsoleMessageActionService.available_actions`` (``feedback``
            still grouped as one entry).

    Returns:
        The one-line guide, e.g. ``Guide: j/k select · c Copy · 🔊 Speak ·
        e Edit · r ♻ Regenerate · ---> Continue · 👍/👎 Rate · 🗑 Delete ·
        Esc clear``.
    """
    segments_by_id = dict(ACTION_GUIDE_SEGMENTS)
    parts: list[str] = []
    for action in actions:
        segment = segments_by_id.get(action.action_id)
        if segment is not None and segment not in parts:
            parts.append(segment)
    if not parts:
        return "Guide: j/k select · Esc clear"
    return f"Guide: j/k select · {' · '.join(parts)} · Esc clear"


class ConsoleMessageActionService:
    """Resolve and dispatch safe Console selected-message actions."""

    FEEDBACK_PLAIN_LABELS: tuple[str, str] = ("👍", "👎")

    _COMPLETED_ACTIONS: tuple[tuple[str, str], ...] = (
        ("copy", "Copy"),
        ("speak", "🔊"),
        ("edit", "Edit"),
        ("save-as", "Save as..."),
        ("fork", "Fork"),
        ("regenerate", "♻"),
        ("continue", "--->"),
        ("feedback", "Feedback"),
        ("delete", "🗑"),
    )
    #: TASK-1860: reveals the FULL tool result behind a truncated marker.
    #: Offered only for a TOOL marker that actually carries more than its
    #: `content` shows -- an expand control that opens an identical view is
    #: the same dead affordance TASK-1843 removed from the Inspector.
    _TOOL_OUTPUT_ACTIONS: tuple[tuple[str, str], ...] = (
        ("tool-output", "Full output"),
    )
    #: TASK-1366: a diff-carrying marker whose stripped result FIT the
    #: preview has no fuller text to show -- expansion reveals the inline
    #: diff row instead, so the affordance says what it opens.
    _TOOL_DIFF_ACTIONS: tuple[tuple[str, str], ...] = (("tool-output", "Diff"),)
    #: TASK-1972: offered only on a change-summary row (one carrying the
    #: run id it reviews). Opens the Change Review screen for THAT turn.
    _REVIEW_CHANGES_ACTIONS: tuple[tuple[str, str], ...] = (
        ("review-changes", "Review"),
    )
    _VARIANT_NAV_ACTIONS: tuple[tuple[str, str], ...] = (
        ("variant-previous", "<"),
        ("variant-next", ">"),
    )
    _KEEP_ACTION: tuple[tuple[str, str], ...] = (("keep", "keep"),)
    _SPEAK_STOP_ACTION: tuple[str, str] = ("speak-stop", "⏹")
    _IMAGE_VIEW_ACTIONS: tuple[tuple[str, str], ...] = (("toggle-image-view", "View"),)
    _SAVE_IMAGE_ACTIONS: tuple[tuple[str, str], ...] = (("save-image", "Save Image"),)
    #: task-3401.5: offered only on a video-generation message (one carrying
    #: video_metadata). Play opens the ephemeral file with the OS player;
    #: Save copies it out of the ephemeral store -- the only byte escape
    #: hatch, and always an explicit user act (ADR-044).
    _VIDEO_ACTIONS: tuple[tuple[str, str], ...] = (
        ("video-play", "▶"),
        ("video-save-copy", "Save"),
    )
    #: Enablement reason shared by both video actions when the file is gone.
    _VIDEO_FILE_MISSING_REASON = (
        "The ephemeral video file is gone — regenerate to recreate it."
    )
    _QUARANTINED_FEEDBACK_REASON = (
        "Reload the canonical generation before recording feedback."
    )
    _VIEW_ORIGINAL_ATTEMPT_ACTION: tuple[tuple[str, str], ...] = (
        ("view-original-attempt", "View original attempt"),
    )
    _PRIMARY_ACTION_IDS = frozenset(
        {
            "copy",
            "speak",
            "speak-stop",
            "edit",
            "fork",
            "regenerate",
            "retry",
            "continue",
        }
    )
    _SPECIALIZED_ACTION_IDS = frozenset(
        {"raw-cli-stop", "tool-output", "review-changes"}
    )
    _MEDIA_ACTION_IDS = frozenset(
        {
            "variant-previous",
            "variant-next",
            "keep",
            "toggle-image-view",
            "save-image",
            "video-play",
            "video-save-copy",
        }
    )

    @staticmethod
    def _has_tool_output(message: ConsoleChatMessage) -> bool:
        """Whether this row hides tool output its `content` does not show."""
        if message.role is not ConsoleMessageRole.TOOL:
            return False
        # Deliberately NOT "is the full text absent from content": an
        # EXPANDED row does contain it, and that must not remove the control
        # that collapses it again. Whether there is more to show is settled
        # once, when the marker is built.
        # TASK-1366: a diff-carrying marker always has more to show -- the
        # inline diff row -- even when the stripped result was short enough
        # that `tool_output_full` is None (the common case for file writes).
        return bool(message.tool_output_full) or message.tool_diff is not None

    @staticmethod
    def _has_image(message: ConsoleChatMessage) -> bool:
        return message.image_data is not None or bool(message.image_mime_type)

    @staticmethod
    def _speak_visible(message: ConsoleChatMessage) -> bool:
        """Offer speech only for trusted completed assistant text."""
        return _speech_visible(message)

    def __init__(
        self,
        *,
        available_save_destinations: set[str] | None = None,
        unavailable_save_reasons: dict[str, str] | None = None,
    ) -> None:
        self.available_save_destinations = set(available_save_destinations or ())
        self.unavailable_save_reasons = dict(unavailable_save_reasons or {})

    @classmethod
    def _base_actions_with(
        cls, inserted: tuple[tuple[str, str], ...]
    ) -> list[tuple[str, str]]:
        """Return the base row with extras before the Fork/Regenerate pair."""
        actions: list[tuple[str, str]] = []
        for action_id, label in cls._COMPLETED_ACTIONS:
            if action_id == "fork":
                actions.extend(inserted)
            actions.append((action_id, label))
        return actions

    def available_actions(
        self,
        message: ConsoleChatMessage,
        *,
        generation_variant_count: int = 0,
        generation_browsed_index: int = 0,
        speaking_message_id: str | None = None,
        original_attempt_available: bool = False,
        ephemeral: bool = False,
        video_file_available: bool = False,
        fork_eligibility: ConsoleForkEligibility = ConsoleForkEligibility(True),
    ) -> list[ConsoleMessageAction]:
        """Return canonical selected-message actions for a transcript message.

        Args:
            message: Transcript message to resolve actions for.
            generation_variant_count: Number of image-generation variants
                carried by this message (0 for a non-generation message).
                When > 0 this gates `<`/`>`/Keep INSTEAD of the text-sibling
                ``sibling_count``/``sibling_index`` fields on ``message`` --
                an image-variant set and a text-sibling set are mutually
                exclusive shapes (spec §5.1). Defaults to 0 so existing
                callers that don't pass these kwargs see byte-identical
                behavior.
            generation_browsed_index: Currently browsed variant index for a
                generation message (ignored when ``generation_variant_count``
                is 0).
            speaking_message_id: id of the Console message currently driving
                TTS playback, if any (task-559 unit 2). When it matches
                ``message.id`` the row's 🔊 speak action swaps to a ⏹
                speak-stop action in the same slot -- mirrors how the
                generation card's browsed index swaps in "Keep". Defaults to
                ``None`` so existing callers see byte-identical behavior.
            original_attempt_available: Whether this completed assistant has
                a current-session original-attempt preview. Defaults to false;
                plain/export helpers intentionally never pass it.
            ephemeral: Whether the active session is temporary, which blocks
                the row actions that would write a derived artifact to disk
                (currently just Save Image).
            fork_eligibility: Store-derived active-prefix durability result.
                Message-local settled/content checks remain presentation-only;
                this service never infers persisted lineage from message fields.
        """
        if not isinstance(fork_eligibility, ConsoleForkEligibility):
            raise TypeError("fork_eligibility must be ConsoleForkEligibility")
        raw_cli = message.raw_cli_presentation
        if raw_cli is not None:
            actions: list[ConsoleMessageAction] = []
            if raw_cli.lifecycle_state in {"starting", "running"}:
                actions.append(ConsoleMessageAction("raw-cli-stop", "Stop"))
            elif raw_cli.lifecycle_state == "stopping":
                actions.append(
                    ConsoleMessageAction(
                        "raw-cli-stop",
                        "Stopping…",
                        enabled=False,
                        disabled_reason="Raw CLI cancellation is already in progress.",
                    )
                )
            if self._has_tool_output(message):
                actions.append(ConsoleMessageAction("tool-output", "Full output"))
            return actions
        disabled_reason = self._disabled_reason(message)
        is_generation_message = generation_variant_count > 0
        completed_actions = list(self._COMPLETED_ACTIONS)
        extra_actions: list[tuple[str, str]] = []
        if (
            original_attempt_available
            and message.status == "complete"
            and self._is_assistant_message(message)
        ):
            extra_actions.extend(self._VIEW_ORIGINAL_ATTEMPT_ACTION)
        if is_generation_message:
            if generation_variant_count > 1:
                extra_actions.extend(self._VARIANT_NAV_ACTIONS)
            if generation_browsed_index != 0:
                extra_actions.extend(self._KEEP_ACTION)
        elif message.sibling_count > 1:
            extra_actions.extend(self._VARIANT_NAV_ACTIONS)
        if extra_actions:
            completed_actions = self._base_actions_with(tuple(extra_actions))
        if self._has_tool_output(message):
            # Diff-only marker (no fuller TEXT behind the preview): the
            # expand control opens the inline diff row, so label it that
            # way. Full-output and full-output+diff markers keep the
            # TASK-1860 copy.
            if message.tool_output_full:
                completed_actions = completed_actions + list(self._TOOL_OUTPUT_ACTIONS)
            else:
                completed_actions = completed_actions + list(self._TOOL_DIFF_ACTIONS)
        if getattr(message, "change_review_run_id", None):
            completed_actions = completed_actions + list(self._REVIEW_CHANGES_ACTIONS)
        if self._has_image(message):
            completed_actions = (
                completed_actions
                + list(self._IMAGE_VIEW_ACTIONS)
                + list(self._SAVE_IMAGE_ACTIONS)
            )
        if getattr(message, "video_metadata", None) is not None:
            completed_actions = completed_actions + list(self._VIDEO_ACTIONS)
        for block in assistant_canvas_html_blocks(message):
            completed_actions.extend(
                (
                    (f"canvas-open-{block.index}", "Open in Canvas"),
                    (f"canvas-open-new-{block.index}", "Open as new"),
                )
            )
        if not self._is_forkable_row(message):
            completed_actions = [
                (action_id, label)
                for action_id, label in completed_actions
                if action_id != "fork"
            ]
        if not self._speak_visible(message):
            completed_actions = [
                (action_id, label)
                for action_id, label in completed_actions
                if action_id != "speak"
            ]
        elif speaking_message_id == message.id:
            completed_actions = [
                (self._SPEAK_STOP_ACTION[0], self._SPEAK_STOP_ACTION[1])
                if action_id == "speak"
                else (action_id, label)
                for action_id, label in completed_actions
            ]
        if message.status == "failed" and self._is_assistant_message(message):
            # Retry regenerates a failed ASSISTANT response. A failed USER row —
            # e.g. the TASK-457(a) optimistic echo rejected before any provider
            # send — has nothing to regenerate, so it must not offer retry (the
            # user re-sends from the composer instead). Speak and Continue are
            # also absent: a failed response is retried in place rather than
            # read aloud or extended as a new assistant turn.
            completed_actions = [
                ("retry", "Retry") if action_id == "regenerate" else (action_id, label)
                for action_id, label in completed_actions
                if action_id not in {"speak", "continue"}
            ]
        return [
            ConsoleMessageAction(
                action_id=action_id,
                label=label,
                enabled=disabled_reason == ""
                and self._action_enabled(
                    action_id,
                    message,
                    generation_variant_count=generation_variant_count,
                    generation_browsed_index=generation_browsed_index,
                    ephemeral=ephemeral,
                    video_file_available=video_file_available,
                    fork_eligibility=fork_eligibility,
                ),
                disabled_reason=disabled_reason
                or self._action_disabled_reason(
                    action_id,
                    message,
                    generation_variant_count=generation_variant_count,
                    generation_browsed_index=generation_browsed_index,
                    ephemeral=ephemeral,
                    video_file_available=video_file_available,
                    fork_eligibility=fork_eligibility,
                ),
            )
            for action_id, label in completed_actions
        ]

    def plain_action_labels(self, message: ConsoleChatMessage) -> list[str]:
        """Return terminal-width labels for a message action row."""
        return self.expand_plain_action_labels(self.selected_row_actions(message))

    def action_groups(
        self,
        message: ConsoleChatMessage,
        *,
        generation_variant_count: int = 0,
        generation_browsed_index: int = 0,
        speaking_message_id: str | None = None,
        original_attempt_available: bool = False,
        ephemeral: bool = False,
        video_file_available: bool = False,
        fork_eligibility: ConsoleForkEligibility = ConsoleForkEligibility(True),
    ) -> ConsoleMessageActionGroups:
        """Resolve the row once, then split direct, overflow, and media actions.

        Args:
            message: Transcript message to resolve.
            generation_variant_count: Generated-image variant count.
            generation_browsed_index: Selected generated-image position.
            speaking_message_id: Message currently driving speech playback.
            original_attempt_available: Whether a safe original preview exists.
            ephemeral: Whether disk-writing media actions must be blocked.
            video_file_available: Whether the ephemeral video bytes still exist.
            fork_eligibility: Store-derived active-prefix durability result.

        Returns:
            Immutable primary, overflow, and media action tuples.
        """

        actions = tuple(
            self.available_actions(
                message,
                generation_variant_count=generation_variant_count,
                generation_browsed_index=generation_browsed_index,
                speaking_message_id=speaking_message_id,
                original_attempt_available=original_attempt_available,
                ephemeral=ephemeral,
                video_file_available=video_file_available,
                fork_eligibility=fork_eligibility,
            )
        )
        if not self._is_forkable_row(message):
            return ConsoleMessageActionGroups(
                primary=tuple(
                    action
                    for action in actions
                    if action.action_id in self._SPECIALIZED_ACTION_IDS
                ),
                overflow=(),
                media=(),
            )
        overflow = self._overflow_actions(actions)
        primary_ids = self._PRIMARY_ACTION_IDS
        if generation_variant_count == 0:
            primary_ids = primary_ids | {
                "variant-previous",
                "variant-next",
                "toggle-image-view",
                "save-image",
            }
        primary = tuple(action for action in actions if action.action_id in primary_ids)
        if overflow:
            overflow_enabled = any(action.enabled for action in overflow)
            primary += (
                ConsoleMessageAction(
                    "more",
                    "More…",
                    enabled=overflow_enabled,
                    disabled_reason=(
                        "" if overflow_enabled else overflow[0].disabled_reason
                    ),
                ),
            )
        media = tuple(
            action
            for action in actions
            if action.action_id in self._MEDIA_ACTION_IDS
            and (
                action.action_id
                not in {
                    "variant-previous",
                    "variant-next",
                    "keep",
                    "toggle-image-view",
                    "save-image",
                }
                or generation_variant_count > 0
            )
        )
        return ConsoleMessageActionGroups(
            primary=primary,
            overflow=overflow,
            media=media,
        )

    @staticmethod
    def _overflow_actions(
        actions: tuple[ConsoleMessageAction, ...],
    ) -> tuple[ConsoleMessageAction, ...]:
        overflow: list[ConsoleMessageAction] = []
        for action in actions:
            if action.action_id == "save-as":
                overflow.append(
                    ConsoleMessageAction(
                        "save-as",
                        "Save as…",
                        action.enabled,
                        action.disabled_reason,
                    )
                )
            elif action.action_id == "view-original-attempt":
                overflow.append(action)
            elif action.action_id == "feedback":
                overflow.extend(
                    (
                        ConsoleMessageAction(
                            "feedback-up",
                            "Helpful",
                            action.enabled,
                            action.disabled_reason,
                        ),
                        ConsoleMessageAction(
                            "feedback-down",
                            "Not helpful",
                            action.enabled,
                            action.disabled_reason,
                        ),
                    )
                )
            elif action.action_id == "delete":
                overflow.append(
                    ConsoleMessageAction(
                        "delete",
                        "Delete",
                        action.enabled,
                        action.disabled_reason,
                    )
                )
            elif action.action_id.startswith("canvas-open"):
                overflow.append(action)
        return tuple(overflow)

    def selected_row_actions(
        self,
        message: ConsoleChatMessage,
        *,
        generation_variant_count: int = 0,
        generation_browsed_index: int = 0,
        speaking_message_id: str | None = None,
        original_attempt_available: bool = False,
        ephemeral: bool = False,
        video_file_available: bool = False,
        fork_eligibility: ConsoleForkEligibility = ConsoleForkEligibility(True),
    ) -> list[ConsoleMessageAction]:
        """Return the selected-message action row, including Speak/Stop.

        Speak is a per-message option like copy/edit: it renders in the
        action row of the SELECTED message and swaps to speak-stop while
        that message is the active TTS speaking message. The header keeps
        only active-playback lifecycle status (generating/playing/stopped/
        failed) so playback stays controllable after deselection.
        """
        return list(
            self.action_groups(
                message,
                generation_variant_count=generation_variant_count,
                generation_browsed_index=generation_browsed_index,
                speaking_message_id=speaking_message_id,
                original_attempt_available=original_attempt_available,
                ephemeral=ephemeral,
                video_file_available=video_file_available,
                fork_eligibility=fork_eligibility,
            ).primary
        )

    def plain_action_row(self, message: ConsoleChatMessage) -> str:
        """Return a terminal-readable action row for plain transcript exports."""
        return " ".join(self.plain_action_labels(message))

    def plain_action_guide(self, message: ConsoleChatMessage) -> str:
        """Return the action-row legend for plain transcript exports.

        Same un-keyworded ``available_actions`` call as ``plain_action_row``,
        so an export's legend names exactly the glyphs its action row shows.
        """
        return action_row_guide(self.selected_row_actions(message))

    @classmethod
    def expand_plain_action_labels(
        cls, actions: list[ConsoleMessageAction]
    ) -> list[str]:
        """Expand grouped UI actions into the labels shown in plain text."""
        labels: list[str] = []
        for action in actions:
            if action.action_id == "feedback":
                labels.extend(cls.FEEDBACK_PLAIN_LABELS)
            else:
                labels.append(action.label)
        return labels

    def save_as_destinations(
        self, message: ConsoleChatMessage
    ) -> list[ConsoleSaveDestination]:
        """Return Save as destinations, including explicit unavailable entries."""
        _ = message
        labels = ("Chatbook", "Note", "Media", "Prompt")
        destinations: list[ConsoleSaveDestination] = []
        for label in labels:
            available = label in self.available_save_destinations
            reason = ""
            if not available:
                reason = (
                    self.unavailable_save_reasons.get(label)
                    or f"Save as {label} is not available in this session."
                )
            destinations.append(
                ConsoleSaveDestination(label=label, available=available, reason=reason)
            )
        return destinations

    def dispatch(
        self, action_id: str, message: ConsoleChatMessage
    ) -> ConsoleActionResult:
        """Dispatch a pure action result without touching UI or persistence."""
        if action_id == "raw-cli-stop":
            raw_cli = message.raw_cli_presentation
            if raw_cli is None or raw_cli.lifecycle_state not in {
                "starting",
                "running",
            }:
                return ConsoleActionResult(
                    action_id=action_id,
                    status="blocked",
                    visible_copy="Raw CLI command is no longer running.",
                )
            return ConsoleActionResult(
                action_id=action_id,
                status="completed",
                visible_copy="Stopping raw CLI command…",
                target_message_id=message.id,
                target_invocation_id=raw_cli.invocation_id,
            )
        if message.status in {"pending", "streaming"}:
            return ConsoleActionResult(
                action_id=action_id,
                status="blocked",
                visible_copy=self._disabled_reason(message),
            )
        if action_id.startswith("canvas-open-"):
            try:
                block_index = int(action_id.rsplit("-", 1)[1])
            except (ValueError, IndexError):
                block_index = -1
            blocks = assistant_canvas_html_blocks(message)
            block = next((item for item in blocks if item.index == block_index), None)
            if block is None:
                return ConsoleActionResult(
                    action_id=action_id,
                    status="blocked",
                    visible_copy="That HTML block is no longer available.",
                    target_message_id=message.id,
                )
            if not block.compatible:
                codes = ", ".join(block.compatibility_codes) or "unsupported input"
                return ConsoleActionResult(
                    action_id=action_id,
                    status="canvas_repair_requested",
                    visible_copy="Prepared a Canvas compatibility repair request.",
                    target_message_id=message.id,
                    target_content=(
                        "Please rewrite HTML block "
                        f"{block.index + 1} from your previous response as one "
                        "self-contained Canvas V1 HTML document with inline CSS and "
                        f"JavaScript only. Resolve these compatibility issues: {codes}."
                    ),
                    target_invocation_id=block.identity,
                )
            create_new = action_id.startswith("canvas-open-new-")
            return ConsoleActionResult(
                action_id=action_id,
                status="canvas_open_requested",
                visible_copy=(
                    "Opening HTML as a new Canvas."
                    if create_new
                    else "Opening HTML in Canvas."
                ),
                target_message_id=message.id,
                canvas_block_ref=ConsoleCanvasBlockReference(
                    message_id=message.id,
                    block_index=block.index,
                    identity=block.identity,
                    create_new=create_new,
                ),
            )
        if (
            action_id in {"feedback-up", "feedback-down"}
            and message.generation_projection_quarantined
        ):
            return ConsoleActionResult(
                action_id=action_id,
                status="blocked",
                visible_copy=self._QUARANTINED_FEEDBACK_REASON,
                target_message_id=message.id,
            )
        if action_id == "copy":
            return ConsoleActionResult(
                action_id=action_id,
                status="completed",
                visible_copy="Copied message to clipboard.",
                clipboard_text=message.content,
            )
        if action_id == "view-original-attempt":
            return ConsoleActionResult(
                action_id=action_id,
                status="completed",
                visible_copy="Toggled original attempt preview.",
                target_message_id=message.id,
            )
        if action_id == "speak":
            return ConsoleActionResult(
                action_id=action_id,
                status="completed",
                visible_copy="Speaking message.",
                target_message_id=message.id,
                target_content=message.content,
            )
        if action_id == "speak-stop":
            # task-559 unit 2: stop is safe to request unconditionally --
            # the app-level TTSPlaybackEvent(action="stop") handler already
            # no-ops when nothing is playing/cached for this message id.
            return ConsoleActionResult(
                action_id=action_id,
                status="completed",
                visible_copy="Stopped speaking.",
                target_message_id=message.id,
            )
        if action_id == "retry" and message.status == "failed":
            return ConsoleActionResult(
                action_id=action_id,
                status="completed",
                visible_copy="Retrying failed response.",
            )
        if action_id == "edit":
            target_content = (
                message.variants.current.content
                if message.variants is not None
                else message.content
            )
            return ConsoleActionResult(
                action_id=action_id,
                status="edit_requested",
                visible_copy="Opened Edit Message.",
                target_message_id=message.id,
                target_content=target_content,
            )
        if action_id == "fork":
            return ConsoleActionResult(
                action_id=action_id,
                status="fork_requested",
                visible_copy="Opened Fork chat.",
                target_message_id=message.id,
            )
        if action_id in {"feedback-up", "feedback-down"}:
            feedback = "up" if action_id == "feedback-up" else "down"
            return ConsoleActionResult(
                action_id=action_id,
                status="completed",
                visible_copy=f"Marked message feedback: {feedback}.",
                target_message_id=message.id,
                target_content=feedback,
            )
        if action_id == "delete":
            return ConsoleActionResult(
                action_id=action_id,
                status="completed",
                visible_copy="Deleted message from transcript.",
                target_message_id=message.id,
            )
        if action_id in {"variant-previous", "variant-next"}:
            return ConsoleActionResult(
                action_id=action_id,
                status="completed",
                visible_copy="Selected response variant.",
            )
        if action_id == "keep":
            return ConsoleActionResult(
                action_id=action_id,
                status="completed",
                visible_copy="Kept this variant as the message's canonical image.",
                target_message_id=message.id,
            )
        if (
            action_id == "regenerate"
            and not ConsoleMessageActionService._is_assistant_message(message)
        ):
            return ConsoleActionResult(
                action_id=action_id,
                status="blocked",
                visible_copy="Only assistant messages can be regenerated.",
            )
        if (
            action_id == "continue"
            and message.status == "failed"
            and ConsoleMessageActionService._is_assistant_message(message)
        ):
            return ConsoleActionResult(
                action_id=action_id,
                status="blocked",
                visible_copy="Retry the failed response instead.",
            )
        if action_id == "continue":
            target_content = (
                message.variants.current.content
                if message.variants is not None
                else message.content
            )
            return ConsoleActionResult(
                action_id=action_id,
                status="continue_requested",
                visible_copy="Continuing from selected message.",
                target_message_id=message.id,
                target_content=target_content,
            )
        if action_id == "toggle-image-view":
            return ConsoleActionResult(
                action_id=action_id,
                status="completed",
                visible_copy="Toggled image view.",
                target_message_id=message.id,
            )
        if action_id == "save-image":
            return ConsoleActionResult(
                action_id=action_id,
                status="completed",
                visible_copy="Saving image to disk.",
                target_message_id=message.id,
            )
        if action_id == "video-play":
            return ConsoleActionResult(
                action_id=action_id,
                status="completed",
                visible_copy="Opening video with the system player.",
                target_message_id=message.id,
            )
        if action_id == "video-save-copy":
            return ConsoleActionResult(
                action_id=action_id,
                status="completed",
                visible_copy="Saving a copy of the video to disk.",
                target_message_id=message.id,
            )
        return ConsoleActionResult(
            action_id=action_id,
            status="wip",
            visible_copy=f"WIP: {action_id} is not wired yet.",
        )

    @staticmethod
    def _disabled_reason(message: ConsoleChatMessage) -> str:
        if message.status in {"pending", "streaming"}:
            return "Wait for response to finish before using message actions."
        return ""

    @staticmethod
    def _variant_action_enabled(
        action_id: str,
        message: ConsoleChatMessage,
        *,
        generation_variant_count: int = 0,
        generation_browsed_index: int = 0,
    ) -> bool:
        if generation_variant_count > 0:
            # Generation-variant boundary check takes precedence over the
            # text-sibling fields for these two ids (spec §7) -- a
            # generation message never carries text siblings.
            if action_id == "variant-previous":
                return generation_browsed_index > 0
            if action_id == "variant-next":
                return generation_browsed_index < generation_variant_count - 1
            return True
        if action_id == "variant-previous":
            return message.sibling_index > 0
        if action_id == "variant-next":
            return message.sibling_index < message.sibling_count - 1
        return True

    @staticmethod
    def _action_enabled(
        action_id: str,
        message: ConsoleChatMessage,
        *,
        generation_variant_count: int = 0,
        generation_browsed_index: int = 0,
        ephemeral: bool = False,
        video_file_available: bool = False,
        fork_eligibility: ConsoleForkEligibility = ConsoleForkEligibility(True),
    ) -> bool:
        if action_id == "regenerate":
            return ConsoleMessageActionService._is_assistant_message(message)
        if action_id == "fork":
            return not ConsoleMessageActionService._fork_disabled_reason(
                message,
                fork_eligibility,
            )
        if action_id in {"feedback", "feedback-up", "feedback-down"}:
            return not message.generation_projection_quarantined
        if action_id == "save-image":
            return blocked_reason("save-image", ephemeral=ephemeral) is None
        if action_id in {"video-play", "video-save-copy"}:
            return video_file_available
        return ConsoleMessageActionService._variant_action_enabled(
            action_id,
            message,
            generation_variant_count=generation_variant_count,
            generation_browsed_index=generation_browsed_index,
        )

    @staticmethod
    def _action_disabled_reason(
        action_id: str,
        message: ConsoleChatMessage,
        *,
        generation_variant_count: int = 0,
        generation_browsed_index: int = 0,
        ephemeral: bool = False,
        video_file_available: bool = False,
        fork_eligibility: ConsoleForkEligibility = ConsoleForkEligibility(True),
    ) -> str:
        if (
            action_id == "regenerate"
            and not ConsoleMessageActionService._is_assistant_message(message)
        ):
            return "Only assistant messages can be regenerated."
        if action_id == "fork":
            return ConsoleMessageActionService._fork_disabled_reason(
                message,
                fork_eligibility,
            )
        if (
            action_id in {"feedback", "feedback-up", "feedback-down"}
            and message.generation_projection_quarantined
        ):
            return ConsoleMessageActionService._QUARANTINED_FEEDBACK_REASON
        if action_id == "save-image":
            return blocked_reason("save-image", ephemeral=ephemeral) or ""
        if action_id in {"video-play", "video-save-copy"} and not video_file_available:
            return ConsoleMessageActionService._VIDEO_FILE_MISSING_REASON
        if action_id in {
            "variant-previous",
            "variant-next",
        } and not ConsoleMessageActionService._variant_action_enabled(
            action_id,
            message,
            generation_variant_count=generation_variant_count,
            generation_browsed_index=generation_browsed_index,
        ):
            return "No response variant in that direction."
        return ""

    @staticmethod
    def _is_assistant_message(message: ConsoleChatMessage) -> bool:
        role = getattr(message.role, "value", message.role)
        return str(role).lower() == ConsoleMessageRole.ASSISTANT.value

    @staticmethod
    def _is_forkable_row(message: ConsoleChatMessage) -> bool:
        role = getattr(message.role, "value", message.role)
        return (
            str(role).lower()
            in {ConsoleMessageRole.USER.value, ConsoleMessageRole.ASSISTANT.value}
            and message.activity_presentation is None
        )

    @staticmethod
    def _fork_disabled_reason(
        message: ConsoleChatMessage,
        eligibility: ConsoleForkEligibility,
    ) -> str:
        if message.status in {"pending", "streaming"}:
            return "Wait for this message to finish before forking."
        if message.status == "discarded":
            return "Discarded messages cannot be forked."
        if message.status in {"stopped", "failed"} and not message.content.strip():
            return "This partial response has no content to fork."
        if (
            message.status != "complete"
            and not ConsoleMessageActionService._is_assistant_message(message)
        ):
            return "Only complete user messages can be forked."
        if not eligibility.eligible:
            return eligibility.reason or "This message cannot be forked."
        return ""

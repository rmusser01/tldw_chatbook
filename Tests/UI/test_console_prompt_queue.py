from __future__ import annotations

import inspect
from dataclasses import fields as dataclass_fields
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
from textual.app import ComposeResult
from textual.widgets import Button, TextArea

from Tests.private_profile import private_profile_test

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from Tests.UI.test_console_dictation import _mounted_console, _ready_host
from tldw_chatbook.Chat.attachment_core import PendingAttachment
from tldw_chatbook.Chat.console_chat_models import ConsoleControllerActivity
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_prompt_queue import (
    MAX_CONSOLE_QUEUE_ENTRIES,
    ConsolePromptQueueRegistry,
    PromptQueueMode,
    PromptQueuePauseReason,
    PromptQueueReservation,
    PromptQueueSnapshot,
    QueueMutationStatus,
    make_prompt_preview,
)
from tldw_chatbook.Chat.console_runtime import (
    ConsoleRuntime,
    ConsoleTurnRecoveryEntry,
)
from tldw_chatbook.UI.Console_Modules.prompt_queue import (
    RECOVERY_TURN_PREVIEW_CELLS,
    ConsolePromptDispatchStatus,
    ConsolePromptQueuePresentation,
    ConsolePromptQueueRegion,
    ConsolePromptQueueUIController,
    derive_prompt_queue_presentation,
)
from tldw_chatbook.Widgets.Console import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.console_prompt_queue_modal import (
    ConsolePromptQueueModal,
)
from tldw_chatbook.Widgets.Console.console_session_surface import (
    ConsoleSessionSurface,
)


def _activity(
    session_id: str = "session-a",
    *,
    preparing: bool = False,
    accepted: bool = False,
    occupies: bool | None = None,
    count: int = 0,
    paused: bool = False,
) -> ConsoleControllerActivity:
    return ConsoleControllerActivity(
        session_id=session_id,
        occupies_slot=preparing or accepted if occupies is None else occupies,
        preparing_before_acceptance=preparing,
        accepted_live_turn=accepted,
        needs_approval=False,
        queued_count=count,
        queue_paused=paused,
        terminal_notification_eligible=False,
    )


def _registry_with_chain() -> ConsolePromptQueueRegistry:
    registry = ConsolePromptQueueRegistry()
    snapshot = registry.snapshot("session-a")
    registry.begin_chain(
        "session-a", context_epoch=1, expected_revision=snapshot.revision
    )
    return registry


def test_presentation_uses_exact_send_queue_boundaries() -> None:
    registry = ConsolePromptQueueRegistry()
    empty = registry.snapshot("session-a")

    preparing = derive_prompt_queue_presentation(empty, _activity(preparing=True))
    assert preparing.send_label == "Preparing..."
    assert preparing.send_enabled is False

    handoff = derive_prompt_queue_presentation(empty, _activity(occupies=True))
    assert handoff.send_label == "Preparing..."
    assert handoff.send_enabled is False

    chain = _registry_with_chain()
    chained = chain.snapshot("session-a")
    queue = derive_prompt_queue_presentation(chained, _activity(accepted=True))
    assert queue.send_label == "Queue"
    assert queue.send_enabled is True

    for index in range(MAX_CONSOLE_QUEUE_ENTRIES):
        chained = chain.admit(
            "session-a",
            text=f"prompt {index}",
            expected_revision=chained.revision,
        ).snapshot
    full = derive_prompt_queue_presentation(
        chained,
        _activity(accepted=True, count=MAX_CONSOLE_QUEUE_ENTRIES),
    )
    assert full.send_label == "Queue full"
    assert full.send_enabled is False


def test_refusing_presentation_copy_names_the_wait_the_run_is_in() -> None:
    """TASK-33620.4: a refusing presentation's tooltip is also the composer's
    reason-strip copy, so it must be true of the run it describes.

    Only a prompt-chain turn is ever queue-accepted. Regenerate / continue
    (and an agent wake) never create a chain, so they occupy the slot
    WITHOUT preparing-before-acceptance for their whole stream -- the queue
    never opens behind them, and saying it will is the review's finding.
    """
    registry = ConsolePromptQueueRegistry()
    empty = registry.snapshot("session-a")

    preparing = derive_prompt_queue_presentation(empty, _activity(preparing=True))
    assert preparing.send_tooltip == "Queue opens once this turn is accepted"

    chainless = derive_prompt_queue_presentation(empty, _activity(occupies=True))
    assert chainless.send_enabled is False
    assert chainless.send_tooltip == "Wait for the current run to finish"

    chain = _registry_with_chain()
    chained = chain.snapshot("session-a")
    for index in range(MAX_CONSOLE_QUEUE_ENTRIES):
        chained = chain.admit(
            "session-a",
            text=f"prompt {index}",
            expected_revision=chained.revision,
        ).snapshot
    full = derive_prompt_queue_presentation(
        chained,
        _activity(accepted=True, count=MAX_CONSOLE_QUEUE_ENTRIES),
    )
    assert full.send_tooltip == "Queue full — manage it to make room"


def test_background_session_label_exposes_count_only() -> None:
    label = ConsoleSessionSurface._tab_label("Session", queued_count=3)

    assert label == "Q3 Session"


@pytest.mark.parametrize(
    ("reason", "failed_turn_preview", "state", "label", "action"),
    [
        # TASK-33621.19 AC#2: "Turn failed" only for a real failed turn,
        # and it names that turn.
        (
            PromptQueuePauseReason.FAILED,
            "Answer with the word BRAVO.",
            'Turn failed: "Answer with the word BRAVO."',
            "Retry",
            "retry-failed",
        ),
        # AC#3: a FAILED pause with no failed message offers Resume, never
        # a Retry that can only refuse.
        (PromptQueuePauseReason.FAILED, None, "Paused", "Resume", "toggle-pause"),
        (PromptQueuePauseReason.MANUAL, None, "Paused", "Resume", "toggle-pause"),
        (
            PromptQueuePauseReason.STOPPED,
            None,
            "Turn stopped",
            "Resume next",
            "resume-next",
        ),
        (
            PromptQueuePauseReason.CONTEXT_CHANGED,
            None,
            "Context changed",
            "Review",
            "review",
        ),
        (
            PromptQueuePauseReason.DISPATCH_REFUSED,
            None,
            "Start refused",
            "Try again",
            "toggle-pause",
        ),
    ],
)
def test_paused_shelf_exposes_state_specific_primary_action(
    reason: PromptQueuePauseReason,
    failed_turn_preview: str | None,
    state: str,
    label: str,
    action: str,
) -> None:
    registry = _registry_with_chain()
    snapshot = registry.admit(
        "session-a",
        text="waiting",
        expected_revision=registry.snapshot("session-a").revision,
    ).snapshot
    snapshot = registry.pause(
        "session-a", reason=reason, expected_revision=snapshot.revision
    ).snapshot

    presentation = derive_prompt_queue_presentation(
        snapshot,
        _activity(count=1, paused=True),
        failed_turn_preview=failed_turn_preview,
    )

    assert presentation.state_label == state
    assert presentation.pause_label == label
    assert presentation.primary_action == action


class _RegionApp(ConsolidatedCSSApp):
    def compose(self) -> ComposeResult:
        yield ConsolePromptQueueRegion(id="queue")


class _RecoveryRegionApp(ConsolidatedCSSApp):
    def __init__(self, actions: list[tuple[str, int, str]]) -> None:
        super().__init__()
        self.actions = actions

    def compose(self) -> ComposeResult:
        yield ConsolePromptQueueRegion(
            id="queue",
            on_primary_requested=lambda session_id, revision, action: (
                self.actions.append((session_id, revision, action))
            ),
        )


@pytest.mark.asyncio
async def test_region_is_revision_guarded_and_hides_preview_when_collapsed() -> None:
    registry = _registry_with_chain()
    snapshot = registry.admit(
        "session-a",
        text="safe preview",
        expected_revision=registry.snapshot("session-a").revision,
    ).snapshot
    presentation = derive_prompt_queue_presentation(
        snapshot, _activity(accepted=True, count=1)
    )

    app = _RegionApp()
    async with app.run_test(size=(80, 24)) as pilot:
        region = app.query_one("#queue", ConsolePromptQueueRegion)
        assert region.sync_presentation("session-a", presentation) is True
        await pilot.pause()
        assert region.sync_presentation("session-a", presentation) is False
        assert region.has_class("-visible")
        assert region.query_one("#console-prompt-queue-summary").renderable == (
            "Queue 1/10 · Draining"
        )

        collapsed = derive_prompt_queue_presentation(
            snapshot,
            _activity(accepted=True, count=1),
            composer_collapsed=True,
        )
        region.sync_presentation("session-a", collapsed)
        assert not region.has_class("-visible")


@pytest.mark.asyncio
async def test_recovery_shelf_buttons_pin_the_displayed_turn_id() -> None:
    registry = ConsolePromptQueueRegistry()
    presentation = derive_prompt_queue_presentation(
        registry.snapshot("session-a"),
        _activity(),
        turn_recovery_id="turn-a",
    )
    actions: list[tuple[str, int, str]] = []
    app = _RecoveryRegionApp(actions)

    async with app.run_test(size=(100, 24)) as pilot:
        region = app.query_one("#queue", ConsolePromptQueueRegion)
        region.sync_presentation("session-a", presentation)
        await pilot.pause()
        await pilot.click("#console-prompt-queue-manage")
        await pilot.click("#console-prompt-queue-pause")

    assert actions == [
        ("session-a", 0, "turn-recovery:restore:turn-a"),
        ("session-a", 0, "turn-recovery:discard:turn-a"),
    ]


class _PaintedRegionApp(ConsolidatedCSSApp):
    """The shelf under every app-tier sheet production loads.

    The app bundle's ``Button { border: none }`` is what lets a one-row
    Button paint its label at all; without it the harness measures nothing.
    """

    CSS_PATH = [str(path) for path in APP_STYLESHEETS]

    def compose(self) -> ComposeResult:
        yield ConsolePromptQueueRegion(id="queue")


def _shelf_snapshot(
    mode: PromptQueueMode,
    reason: PromptQueuePauseReason | None = None,
    *,
    count: int = 1,
) -> PromptQueueSnapshot:
    return PromptQueueSnapshot(
        session_id="session-a",
        revision=1,
        entries=(),
        waiting_count=count,
        claimed_count=0,
        total_count=count,
        mode=mode,
        pause_reason=reason,
        reservation=PromptQueueReservation.HELD,
        expected_context_epoch=1,
        closing=False,
    )


def _every_shelf_presentation(derive=derive_prompt_queue_presentation):
    """The shelf states ``presentation_for`` can project, by button label.

    Built through the production projection with the arguments
    ``ConsolePromptQueueUIController.presentation_for`` passes, so the labels
    come from production, not from a copy kept here. Only a new
    ``PromptQueuePauseReason`` is picked up automatically. A new
    ``PromptQueueMode`` or a new projection argument needs a row here, and
    ``test_painted_label_walk_covers_every_shelf_projection_input`` fails
    until it has one. 'Starting...' and the bare 'Turn failed' are not walked:
    their buttons carry the same labels as states that are. The failed turn's
    name uses the production preview budget at its full width, behind a full
    queue: that summary is the shelf's longest, and the one most likely to
    push a button off the row.
    """

    paused = PromptQueueMode.PAUSED
    failed_name = make_prompt_preview(
        "Summarise the attached quarterly report in five bullets, then rank "
        "each risk by impact",
        cell_budget=RECOVERY_TURN_PREVIEW_CELLS,
    )
    yield (
        "draining",
        derive(
            _shelf_snapshot(PromptQueueMode.DRAINING), _activity(accepted=True, count=1)
        ),
    )
    yield (
        "pause-after-turn",
        derive(
            _shelf_snapshot(PromptQueueMode.PAUSE_AFTER_TURN),
            _activity(accepted=True, count=1),
        ),
    )
    for reason in PromptQueuePauseReason:
        failed = reason is PromptQueuePauseReason.FAILED
        # The failed turn's row is also painted at a full queue: 'Queue 10/10'
        # is one cell wider than 'Queue 1/10', on the longest summary.
        count = MAX_CONSOLE_QUEUE_ENTRIES if failed else 1
        yield (
            f"paused-{reason.value}",
            derive(
                _shelf_snapshot(paused, reason, count=count),
                _activity(count=count, paused=True),
                failed_turn_preview=failed_name if failed else None,
            ),
        )
    yield (
        "response-recovery-blocked",
        derive(
            _shelf_snapshot(paused, PromptQueuePauseReason.MANUAL),
            _activity(count=1, paused=True),
            dispatch_recovery_blocked=True,
        ),
    )
    yield (
        "unsent-turn-recovery",
        derive(
            _shelf_snapshot(PromptQueueMode.DRAINING, count=0),
            _activity(),
            turn_recovery_id="turn-a",
        ),
    )
    # TASK-33621.2: a refused send states the controller's reason. The
    # second is fitted to the budget left behind a full queue's prefix.
    for count, refusal in (
        (0, "Last send is blocked; resolve it first."),
        (
            MAX_CONSOLE_QUEUE_ENTRIES,
            "Another send is still preparing for this conversation.",
        ),
    ):
        yield (
            f"unsent-turn-refused-queue-{count}",
            derive(
                _shelf_snapshot(PromptQueueMode.DRAINING, count=count),
                _activity(count=count),
                turn_recovery_id="turn-a",
                turn_recovery_reason=refusal,
            ),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "shelf_width",
    # The shelf spans the whole Console shell, rails open or not, so its
    # width is the terminal's. 40/44/46/47 are terminals narrower than the
    # supported 80 columns: they straddle the 45-47 cells the widest label
    # pair needed beside a 20-cell summary. Then both sides of the shelf's
    # own 92-cell narrow threshold, then the widths the mounted Console
    # gives the shelf at 80/100/120/160/235 terminal columns.
    [40, 44, 46, 47, 60, 80, 91, 92, 100, 120, 160, 235],
)
async def test_every_shelf_button_paints_its_whole_label(shelf_width) -> None:
    """TASK-33625.4: the controls on the queue shelf never clip their label.

    The Manage slot was a fixed 8 cells; padding 0 1 plus Button's own
    line-pad 1 left 4 cells, so it painted 'Mana' and 'Rest'. The Pause slot's
    fixed 15 left 11, so 'Keep draining' wrapped and painted only 'Keep'.
    The oracle is the painted strip, not the label value or the widget width:
    each label must be in the button's own rendered line AND in the screen's
    composited row inside the button's columns, which also catches a button
    pushed past the shelf's right edge.
    """

    app = _PaintedRegionApp()
    async with app.run_test(size=(shelf_width, 10)) as pilot:
        region = app.query_one("#queue", ConsolePromptQueueRegion)
        manage = region.query_one("#console-prompt-queue-manage", Button)
        pause = region.query_one("#console-prompt-queue-pause", Button)
        for name, presentation in _every_shelf_presentation():
            assert region.sync_presentation("session-a", presentation)
            await pilot.pause()
            assert region.display and region.region.height == 1, name
            row = app.screen._compositor.render_strips()[region.region.y].text
            for button in (manage, pause):
                label = button.label.plain
                assert label, name
                own_line = button.render_line(0).text
                assert own_line.strip() == label, (
                    f"{name} at {shelf_width}: {button.id} painted "
                    f"{own_line!r} for label {label!r} in "
                    f"{button.region.width} cells"
                )
                assert button.region.right <= region.region.right, (
                    f"{name} at {shelf_width}: {button.id} ends at "
                    f"{button.region.right}, past the shelf's "
                    f"{region.region.right}; row {row!r}"
                )
                assert label in row[button.region.x : button.region.right], (
                    f"{name} at {shelf_width}: {label!r} is not painted in "
                    f"{button.id}'s columns; row {row!r}"
                )
            assert manage.region.right <= pause.region.x, name


def test_painted_label_walk_covers_every_shelf_projection_input() -> None:
    """TASK-33625.5: the painted-label walk reaches every state the shelf can.

    The walk above is the proof that no shelf state pushes a button off the
    row, so it only holds if the walk reaches every state. The shelf once had
    a dispatch-recovery mode, entered only through the projection's
    ``dispatch_recovery=`` argument. Production never passed that argument
    and the walk never exercised it. Fed the real 'Response delivery status is
    unknown on the source device.' copy, it pushed Discard to cell 102 on a
    92-101 cell shelf. So every queue mode, and every projection argument
    given a non-default value, must appear in the walk. Every parameter after
    the snapshot and the activity counts, whatever its kind: a new input
    added before the ``*`` must be walked too.
    """

    parameters = list(
        inspect.signature(derive_prompt_queue_presentation).parameters.values()
    )
    assert [parameter.name for parameter in parameters[:2]] == [
        "snapshot",
        "activity",
    ]
    defaults = {parameter.name: parameter.default for parameter in parameters[2:]}
    walked_inputs: set[str] = set()
    walked_modes: set[PromptQueueMode] = set()

    def recording_derive(snapshot, activity, **kwargs):
        walked_modes.add(snapshot.mode)
        walked_inputs.update(
            name for name, value in kwargs.items() if value != defaults[name]
        )
        return derive_prompt_queue_presentation(snapshot, activity, **kwargs)

    assert list(_every_shelf_presentation(derive=recording_derive))
    # composer_collapsed only hides the shelf, so it has no button to paint;
    # test_region_is_revision_guarded_and_hides_preview_when_collapsed pins it.
    unwalked = sorted(set(defaults) - walked_inputs - {"composer_collapsed"})
    assert not unwalked, (
        "the shelf projection accepts state inputs the painted-label walk "
        f"never exercises: {unwalked}"
    )
    assert walked_modes == set(PromptQueueMode)


def test_queue_shelf_has_no_dispatch_recovery_mode() -> None:
    """Pin that the queue shelf has no dispatch-recovery mode (TASK-33625.5).

    Response recovery is the #console-dispatch-recovery card's job alone. The
    shelf's copy of it (Retry anyway / Retry response / Unavailable) was
    unreachable in production, so it is gone. Pin that the projection can
    neither take a recovery state nor hand the shelf recovery actions. The
    shelf's own blocked state stays: see
    test_queue_presentation_cannot_offer_resume_while_dispatch_recovery_blocks.
    """

    parameters = inspect.signature(derive_prompt_queue_presentation).parameters
    assert "dispatch_recovery" not in parameters
    presentation_fields = {
        field.name for field in dataclass_fields(ConsolePromptQueuePresentation)
    }
    assert "recovery_actions" not in presentation_fields


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (100, 30), (120, 40), (160, 40), (235, 52)])
@private_profile_test
async def test_mounted_shelf_and_neighboring_composer_fit_terminal(
    request, size
) -> None:
    _app, host = _ready_host()
    async with host.run_test(size=size) as pilot:
        console = await _mounted_console(host, pilot)
        controller = console._ensure_console_chat_controller()
        session_id = controller.store.active_session_id
        snapshot = controller.prompt_queue_registry.snapshot(session_id)
        snapshot = controller.prompt_queue_registry.begin_chain(
            session_id,
            context_epoch=controller.store.conversation_context_epoch(session_id),
            expected_revision=snapshot.revision,
        ).snapshot
        controller.prompt_queue_registry.admit(
            session_id,
            text="geometry-safe queued prompt",
            expected_revision=snapshot.revision,
        )
        controller.prompt_queue_coordinator.publish_registry_change(session_id)
        await console._sync_native_console_chat_ui()
        await pilot.pause()

        region = console.query_one("#console-prompt-queue", ConsolePromptQueueRegion)
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        manage = region.query_one("#console-prompt-queue-manage", Button)
        pause = region.query_one("#console-prompt-queue-pause", Button)
        send = composer.query_one("#console-send-message", Button)

        assert region.display
        assert region.region.height == 1
        # task-17661: ALL transient strips sit at the top of the control
        # deck, above the status line — the shelf's nearest lower neighbor
        # in the default (chips-above) placement is the status row, and the
        # composer keeps its quiet gap below the chips. DOM-order geometry,
        # exact in any harness.
        chips = console.query_one("#console-status-chips")
        assert region.region.y + region.region.height == chips.region.y
        assert (
            chips.region.y + chips.region.height
            <= composer.region.y - composer.styles.margin.top
        )
        assert manage.region.right <= region.region.right
        assert pause.region.right <= region.region.right
        assert manage.region.right <= pause.region.x
        assert send.label.plain == "Queue"
        assert send.region.right <= composer.region.right
        # TASK-33625.4: the painted label, not the widget's width. A fixed
        # 8-cell Manage painted 'Mana' on the real Console at every width.
        row = console._compositor.render_strips()[region.region.y].text
        for button, label in ((manage, "Manage"), (pause, "Pause")):
            assert button.render_line(0).text.strip() == label, (
                f"{size}: shelf {region.region.width} cells, {button.id} "
                f"painted {button.render_line(0).text!r}"
            )
            assert label in row[button.region.x : button.region.right], row


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("size", "narrow"),
    [((80, 24), True), ((100, 30), False), ((160, 40), False)],
)
@private_profile_test
async def test_mounted_shelf_naming_a_failed_turn_keeps_retry_on_screen(
    request, size, narrow
) -> None:
    """TASK-33621.19 review: the named 'Turn failed: "..."' summary is the
    shelf's longest label; it truncates instead of pushing Retry off-screen.

    The 80-column case is the narrow shelf, and its label is genuinely wider
    than the space left beside the buttons, so the painted line must end in
    an ellipsis. The wide cases paint the whole label.
    """

    from rich.cells import cell_len

    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole

    _app, host = _ready_host()
    async with host.run_test(size=size) as pilot:
        console = await _mounted_console(host, pilot)
        controller = console._ensure_console_chat_controller()
        store = controller.store
        session_id = store.active_session_id
        failed_prompt = "Summarise the attached quarterly report in five bullets"
        store.append_message(
            session_id, role=ConsoleMessageRole.USER, content=failed_prompt
        )
        reply = store.append_message(
            session_id, role=ConsoleMessageRole.ASSISTANT, content=""
        )
        store.mark_message_failed(reply.id)
        registry = controller.prompt_queue_registry
        snapshot = registry.begin_chain(
            session_id,
            context_epoch=store.conversation_context_epoch(session_id),
            expected_revision=registry.snapshot(session_id).revision,
        ).snapshot
        snapshot = registry.admit(
            session_id,
            text="the next waiting prompt",
            expected_revision=snapshot.revision,
        ).snapshot
        registry.pause(
            session_id,
            reason=PromptQueuePauseReason.FAILED,
            expected_revision=snapshot.revision,
        )
        controller.prompt_queue_coordinator.publish_registry_change(session_id)
        await console._sync_native_console_chat_ui()
        await pilot.pause()

        region = console.query_one("#console-prompt-queue", ConsolePromptQueueRegion)
        summary = region.query_one("#console-prompt-queue-summary")
        manage = region.query_one("#console-prompt-queue-manage", Button)
        retry = region.query_one("#console-prompt-queue-pause", Button)

        assert str(retry.label) == "Retry"
        label = str(summary.render())
        assert "Turn failed" in label
        assert summary.region.right <= manage.region.x
        assert manage.region.right <= retry.region.x
        assert retry.region.right <= region.region.right
        assert "Retry" in retry.render_line(0).text

        painted = summary.render_line(0).text.rstrip()
        assert region.has_class("-narrow") is narrow
        if narrow:
            assert not region.query_one("#console-prompt-queue-preview").display
            assert cell_len(label) > summary.region.width
            assert painted.endswith("…")
            assert label.startswith(painted[:-1])
            assert "Turn failed" in painted
        else:
            assert painted == label


@pytest.mark.asyncio
@private_profile_test
async def test_navigation_confirmation_is_pure_and_preserves_manager_edit(
    request,
) -> None:
    _app, host = _ready_host()
    async with host.run_test(size=(100, 30)) as pilot:
        console = await _mounted_console(host, pilot)
        controller = console._ensure_console_chat_controller()
        session_id = controller.store.active_session_id
        snapshot = controller.prompt_queue_registry.snapshot(session_id)
        snapshot = controller.prompt_queue_registry.begin_chain(
            session_id,
            context_epoch=controller.store.conversation_context_epoch(session_id),
            expected_revision=snapshot.revision,
        ).snapshot
        snapshot = controller.prompt_queue_registry.admit(
            session_id,
            text="keep this private edit",
            expected_revision=snapshot.revision,
        ).snapshot
        controller.prompt_queue_registry.pause(
            session_id,
            reason=PromptQueuePauseReason.MANUAL,
            expected_revision=snapshot.revision,
        )
        controller.prompt_queue_coordinator.publish_registry_change(session_id)
        await console._sync_native_console_chat_ui()
        await pilot.pause()
        await pilot.click("#console-prompt-queue-manage")
        await pilot.pause()
        assert isinstance(host.screen_stack[-1], ConsolePromptQueueModal)
        await pilot.click("#console-prompt-queue-edit")
        edit = host.screen_stack[-1].query_one(
            "#console-prompt-queue-edit-input", TextArea
        )
        edit.text = "unsaved manager edit"
        edit.focus()

        assert await console.confirm_navigation() is True
        await pilot.pause()

        assert isinstance(host.screen_stack[-1], ConsolePromptQueueModal)
        assert edit.text == "unsaved manager edit"
        assert edit.has_focus


@pytest.mark.asyncio
@private_profile_test
async def test_full_console_manager_mounts_entry_children_before_live_list_insert(
    request,
) -> None:
    """Opening Manage must not race child mounts against an unattached row."""

    _app, host = _ready_host()
    async with host.run_test(size=(100, 30)) as pilot:
        console = await _mounted_console(host, pilot)
        controller = console._ensure_console_chat_controller()
        session_id = controller.store.active_session_id
        snapshot = controller.prompt_queue_registry.snapshot(session_id)
        snapshot = controller.prompt_queue_registry.begin_chain(
            session_id,
            context_epoch=controller.store.conversation_context_epoch(session_id),
            expected_revision=snapshot.revision,
        ).snapshot
        controller.prompt_queue_registry.admit(
            session_id,
            text="mounted before children",
            expected_revision=snapshot.revision,
        )
        controller.prompt_queue_coordinator.publish_registry_change(session_id)
        await console._sync_native_console_chat_ui()

        console.query_one("#console-prompt-queue-manage", Button).press()
        await pilot.pause()

        modal = host.screen_stack[-1]
        assert isinstance(modal, ConsolePromptQueueModal)
        entry_buttons = list(modal.query(".console-prompt-queue-entry-select"))
        assert len(entry_buttons) == 1
        assert entry_buttons[0].is_mounted
        assert entry_buttons[0].parent is not None
        assert entry_buttons[0].parent.is_mounted


class _FakeChatController:
    def __init__(self, *, accepted: bool, preparing: bool = False) -> None:
        self.prompt_queue_registry = (
            _registry_with_chain() if accepted else ConsolePromptQueueRegistry()
        )
        self.store = SimpleNamespace(
            active_session_id="session-a",
            conversation_context_epoch=lambda _session_id: 8,
        )
        self.prompt_queue_coordinator = SimpleNamespace(
            dispatch_recovery_blocks_queue=lambda _session_id: False
        )
        self._accepted = accepted
        self._preparing = preparing

    def activity_for(self, session_id: str) -> ConsoleControllerActivity:
        snapshot = self.prompt_queue_registry.snapshot(session_id)
        return _activity(
            session_id,
            preparing=self._preparing,
            accepted=self._accepted,
            count=snapshot.total_count,
        )

    async def queue_prompt(
        self, session_id: str, *, text: str, expected_revision: int, configuration=None
    ):
        return self.prompt_queue_registry.admit(
            session_id, text=text, expected_revision=expected_revision
        )

    def edit_queued_prompt(
        self,
        session_id: str,
        *,
        entry_id: str,
        text: str,
        expected_revision: int,
        configuration=None,
    ):
        return self.prompt_queue_registry.edit(
            session_id,
            entry_id=entry_id,
            text=text,
            expected_revision=expected_revision,
        )

    def send_refusal_copy(self, _session_id: str) -> str:
        return ""


def _ui_controller(
    fake: _FakeChatController,
    calls: dict[str, Any],
    *,
    edit_refusal=lambda _text: "",
    turn_recovery_ids=None,
    restore_turn_recovery=None,
    discard_turn_recovery=None,
    load_recovered_turn=None,
) -> ConsolePromptQueueUIController:
    async def append_system(text: str) -> None:
        calls["system"].append(text)

    async def sync_ui() -> None:
        calls["sync"].append(True)

    kwargs = {}
    kwargs.update(
        turn_recovery_ids=turn_recovery_ids or (lambda _session_id: ()),
        restore_turn_recovery=restore_turn_recovery or (lambda _turn_id: None),
        discard_turn_recovery=discard_turn_recovery or (lambda _turn_id: False),
        load_recovered_turn=load_recovered_turn or (lambda _session_id: None),
    )
    return ConsolePromptQueueUIController(
        chat_controller_accessor=lambda: fake,
        capture_configuration=lambda _: None,
        ensure_active_session=lambda: None,
        blocked_reason_accessor=lambda: "",
        setup_blocked_reason_accessor=lambda: "",
        append_system_message=append_system,
        notify=lambda text, severity: calls["notified"].append((text, severity)),
        focus_composer=lambda: calls["focused"].append(True),
        note_follow_intent=lambda: calls["follow"].append(True),
        launch_chain=lambda draft, session_id: (
            calls["staged"].append((draft, session_id)) or "turn-a"
        ),
        commit_captured_draft=lambda session_id, stash: calls["committed"].append(
            (session_id, stash)
        ),
        commit_queued_draft=lambda session_id, stash: calls["queued"].append(
            (session_id, stash)
        ),
        edit_refusal=edit_refusal,
        sync_ui=sync_ui,
        **kwargs,
    )


def _calls() -> dict[str, Any]:
    return {
        "system": [],
        "sync": [],
        "notified": [],
        "focused": [],
        "staged": [],
        "queued": [],
        "committed": [],
        "inflight": {},
        "follow": [],
    }


def _runtime_with_two_recoveries():
    store = ConsoleChatStore()
    store.create_session(
        session_id="session-a", title="Recovery", workspace_id="global"
    )
    first_attachment = PendingAttachment(
        "/private/first-secret.png",
        "first-secret.png",
        "image",
        "attachment",
        data=b"first-secret-bytes",
    )
    second_attachment = PendingAttachment(
        "/private/second-secret.png",
        "second-secret.png",
        "image",
        "attachment",
        data=b"second-secret-bytes",
    )
    runtime = ConsoleRuntime(SimpleNamespace())
    runtime.set_chat_store(store)
    entries = (
        ConsoleTurnRecoveryEntry(
            turn_id="turn-a",
            session_id="session-a",
            draft="first secret draft",
            attachments=(first_attachment,),
            insertion_order=1,
        ),
        ConsoleTurnRecoveryEntry(
            turn_id="turn-b",
            session_id="session-a",
            draft="second secret draft",
            attachments=(second_attachment,),
            insertion_order=2,
        ),
    )
    runtime._turn_recoveries.update((entry.turn_id, entry) for entry in entries)
    runtime._recovery_turns_by_session["session-a"] = [
        entry.turn_id for entry in entries
    ]
    return runtime, store, entries


def _recovery_ui_controller(runtime, fake, calls, loaded):
    return _ui_controller(
        fake,
        calls,
        turn_recovery_ids=lambda session_id: tuple(
            entry.turn_id for entry in runtime.recoveries_for_session(session_id)
        ),
        restore_turn_recovery=runtime.restore_turn_recovery,
        discard_turn_recovery=runtime.discard_turn_recovery,
        load_recovered_turn=lambda session_id: loaded.append(session_id),
    )


def test_fresh_controller_projects_oldest_recovery_without_secret_body() -> None:
    runtime, store, _entries = _runtime_with_two_recoveries()
    fake = _FakeChatController(accepted=False)
    fake.store = store
    calls = _calls()

    fresh = _recovery_ui_controller(runtime, fake, calls, [])
    presentation = fresh.presentation_for("session-a")
    rendered = repr(presentation)

    assert presentation.count == 0
    assert presentation.shelf_visible
    assert presentation.state_label == "Unsent turn needs attention"
    assert presentation.primary_action == "turn-recovery"
    assert presentation.turn_recovery_id == "turn-a"
    for secret in (
        "turn-a",
        "first secret draft",
        "first-secret.png",
        "/private/first-secret.png",
        "first-secret-bytes",
    ):
        assert secret not in rendered


@pytest.mark.asyncio
async def test_restore_restages_exact_attachments_then_reveals_next_recovery() -> None:
    runtime, store, entries = _runtime_with_two_recoveries()
    suffix = PendingAttachment(
        "/later.png", "later.png", "image", "attachment", data=b"later"
    )
    store.add_pending_attachment("session-a", suffix)
    fake = _FakeChatController(accepted=False)
    fake.store = store
    calls = _calls()
    loaded: list[str] = []
    controller = _recovery_ui_controller(runtime, fake, calls, loaded)

    await controller.handle_primary_intent(
        "session-a",
        action="turn-recovery:restore:turn-a",
        expected_revision=0,
    )

    assert store.session_draft("session-a") == "first secret draft"
    assert store.pending_attachments("session-a") == [
        entries[0].attachments[0],
        suffix,
    ]
    assert store.pending_attachments("session-a")[0] is entries[0].attachments[0]
    assert loaded == ["session-a"]
    assert calls["focused"] == [True]
    assert controller.presentation_for("session-a").turn_recovery_id == "turn-b"


@pytest.mark.asyncio
async def test_discard_releases_exact_oldest_and_reveals_next_recovery() -> None:
    runtime, store, _entries = _runtime_with_two_recoveries()
    fake = _FakeChatController(accepted=False)
    fake.store = store
    calls = _calls()
    controller = _recovery_ui_controller(runtime, fake, calls, [])

    await controller.handle_primary_intent(
        "session-a",
        action="turn-recovery:discard:turn-a",
        expected_revision=0,
    )

    assert [entry.turn_id for entry in runtime.recoveries_for_session("session-a")] == [
        "turn-b"
    ]
    assert controller.presentation_for("session-a").turn_recovery_id == "turn-b"
    assert store.session_draft("session-a") == ""


@pytest.mark.asyncio
async def test_stale_recovery_action_is_a_warning_and_does_not_touch_next() -> None:
    runtime, store, entries = _runtime_with_two_recoveries()
    fake = _FakeChatController(accepted=False)
    fake.store = store
    calls = _calls()
    controller = _recovery_ui_controller(runtime, fake, calls, [])
    assert runtime.discard_turn_recovery("turn-a")

    await controller.handle_primary_intent(
        "session-a",
        action="turn-recovery:restore:turn-a",
        expected_revision=0,
    )

    assert runtime.recoveries_for_session("session-a") == (entries[1],)
    assert store.session_draft("session-a") == ""
    assert store.pending_attachments("session-a") == []
    assert calls["notified"] == [
        ("That unsent turn is no longer available.", "warning")
    ]


@pytest.mark.asyncio
async def test_ambiguous_restore_warns_without_changing_recovery_or_store() -> None:
    runtime, store, entries = _runtime_with_two_recoveries()
    existing = PendingAttachment(
        "/existing.png",
        "existing.png",
        "image",
        "attachment",
        data=b"existing",
    )
    store.set_session_draft("session-a", "new live draft")
    store.add_pending_attachment("session-a", existing)
    fake = _FakeChatController(accepted=False)
    fake.store = store
    calls = _calls()
    controller = _recovery_ui_controller(runtime, fake, calls, [])

    await controller.handle_primary_intent(
        "session-a",
        action="turn-recovery:restore:turn-a",
        expected_revision=0,
    )

    assert runtime.recoveries_for_session("session-a") == entries
    assert store.session_draft("session-a") == "new live draft"
    assert store.pending_attachments("session-a") == [existing]
    assert calls["focused"] == []
    assert calls["notified"] == [
        ("That unsent turn could not be restored safely.", "warning")
    ]


@pytest.mark.asyncio
async def test_restore_for_closed_session_warns_and_keeps_exact_recovery() -> None:
    runtime, store, entries = _runtime_with_two_recoveries()
    store.close_session("session-a")
    fake = _FakeChatController(accepted=False)
    fake.store = store
    calls = _calls()
    controller = _recovery_ui_controller(runtime, fake, calls, [])

    await controller.handle_primary_intent(
        "session-a",
        action="turn-recovery:restore:turn-a",
        expected_revision=0,
    )

    assert runtime.recoveries_for_session("session-a") == entries
    assert calls["focused"] == []
    assert calls["notified"] == [
        ("That unsent turn could not be restored safely.", "warning")
    ]


@pytest.mark.asyncio
async def test_dispatch_admits_exact_text_behind_accepted_turn() -> None:
    fake = _FakeChatController(accepted=True)
    calls = _calls()
    controller = _ui_controller(fake, calls)

    outcome = await controller.dispatch("  exact text\n", stash=None)

    assert outcome.status is ConsolePromptDispatchStatus.QUEUED
    snapshot = fake.prompt_queue_registry.snapshot("session-a")
    entry = snapshot.entries[0]
    body = fake.prompt_queue_registry.read_waiting_text(
        "session-a", entry_id=entry.entry_id, expected_revision=snapshot.revision
    )
    assert body.text == "  exact text\n"
    assert calls["staged"] == []
    assert calls["queued"] == [("session-a", None)]


@pytest.mark.asyncio
@pytest.mark.parametrize("admission", ("busy", "race", "edit"))
@private_profile_test
async def test_wired_queue_admission_freezes_view_source_filter(
    request, monkeypatch, admission
) -> None:
    from Tests.Chat.test_console_turn_execution_context import _authority, _destination
    from Tests.Chat.test_console_turn_preparation import _preparation_values
    from tldw_chatbook.Chat.console_library_policy import ConsoleAutoRetrieve
    from tldw_chatbook.Chat.console_turn_context import ConsoleTurnExecutionContext
    from tldw_chatbook.Chat.console_turn_preparation import (
        ConsoleTurnPreparation,
        ConsoleTurnPreparationState,
    )

    _app, host = _ready_host()
    async with host.run_test(size=(100, 30)) as pilot:
        console = await _mounted_console(host, pilot)
        controller = console._ensure_console_chat_controller()
        store = controller.store
        session_id = store.active_session_id
        selected = ["notes"]
        monkeypatch.setattr(
            console._session, "_rag_source_types_accessor", lambda: tuple(selected)
        )
        snapshot = controller.prompt_queue_registry.snapshot(session_id)
        armed = controller.prompt_queue_registry.begin_chain(
            session_id,
            context_epoch=store.conversation_context_epoch(session_id),
            expected_revision=snapshot.revision,
        )
        assert armed.applied
        activity_calls = 0

        def activity(owner):
            nonlocal activity_calls
            activity_calls += 1
            return _activity(owner, accepted=admission != "race" or activity_calls > 1)

        monkeypatch.setattr(controller, "activity_for", activity)
        if admission == "race":
            # The initial UI activity snapshot is stale while admission becomes
            # accepted between that snapshot and the exact launch boundary.
            monkeypatch.setattr(controller, "send_refusal_copy", lambda _: "")
        # Use the production wiring and builder, not an injected capture double.
        if admission == "edit":
            queued = await controller.queue_prompt(
                session_id, text="original", expected_revision=armed.snapshot.revision
            )
            outcome = console._prompt_queue.edit_waiting(
                session_id,
                queued.snapshot.entries[0].entry_id,
                text="edited",
                expected_revision=queued.snapshot.revision,
            )
            assert outcome.applied
        else:
            outcome = await console._prompt_queue.dispatch(
                "queued", session_id=session_id
            )
            assert outcome.status is ConsolePromptDispatchStatus.QUEUED
        request = (
            controller.prompt_queue_registry._states[session_id]
            .waiting[0]
            .custody_request
        )
        assert request.configuration.rag_defaults["source_types"] == ("notes",)
        selected[:] = ["media", "conversations"]
        runtime = console._console_runtime()
        assert runtime.detach_view(console, runtime._attached_generation)
        assert runtime.view is None
        authority = _authority()
        authority = replace(
            authority,
            policy=replace(
                authority.policy, auto_retrieve=ConsoleAutoRetrieve.AUTOMATIC
            ),
        )
        context = ConsoleTurnExecutionContext(
            request.configuration, authority, _destination()
        )
        assert controller._frozen_rag_source_types(context) == ("notes",)
        values = _preparation_values(session_id=session_id, execution_context=context)
        values.update(
            executed_draft=request.draft,
            transient_user_message_id=None,
            attachment_ids=(),
            evidence_ids=(),
            prefill_id=None,
        )
        preparation = ConsoleTurnPreparation(**values)
        assert store.begin_preparation(preparation) is preparation
        requests = []

        async def search(query, source_types, mode, **kwargs):
            requests.append((query, source_types, mode, kwargs))
            return {"results": []}

        monkeypatch.setattr(
            controller.app, "library_rag_search_service", SimpleNamespace(search=search)
        )
        result = await controller.prepare_library_for_turn(preparation.preparation_id)
        assert result.state is ConsoleTurnPreparationState.READY
        assert requests[0][0] == request.draft
        assert requests[0][1] == ("notes",)


@pytest.mark.asyncio
@pytest.mark.parametrize("admission", ("busy", "race", "edit"))
@private_profile_test
async def test_wired_queue_rejects_wrong_owner_before_draft_or_queue_mutation(
    request, monkeypatch, admission
) -> None:
    _app, host = _ready_host()
    async with host.run_test(size=(100, 30)) as pilot:
        console = await _mounted_console(host, pilot)
        controller = console._ensure_console_chat_controller()
        store = controller.store
        session_id = store.active_session_id
        other = store.create_session(ephemeral=True)
        wrong = controller.resolve_runtime_turn_configuration_snapshot(other.id)
        store.switch_session(session_id)
        monkeypatch.setattr(
            console._session, "_build_console_turn_execution_context", lambda _: wrong
        )
        snapshot = controller.prompt_queue_registry.snapshot(session_id)
        armed = controller.prompt_queue_registry.begin_chain(
            session_id,
            context_epoch=store.conversation_context_epoch(session_id),
            expected_revision=snapshot.revision,
        )
        activity_calls = 0

        def activity(owner):
            nonlocal activity_calls
            activity_calls += 1
            return _activity(owner, accepted=admission != "race" or activity_calls > 1)

        monkeypatch.setattr(controller, "activity_for", activity)
        if admission == "race":
            monkeypatch.setattr(controller, "send_refusal_copy", lambda _: "")
        if admission == "edit":
            queued = await controller.queue_prompt(
                session_id, text="original", expected_revision=armed.snapshot.revision
            )
        before = controller.prompt_queue_registry.snapshot(session_id)
        committed = []
        monkeypatch.setattr(
            console._prompt_queue,
            "_commit_queued_draft",
            lambda *args: committed.append(args),
        )
        monkeypatch.setattr(
            console._prompt_queue,
            "_commit_captured_draft",
            lambda *args: committed.append(args),
        )
        if admission == "edit":
            result = console._prompt_queue.edit_waiting(
                session_id,
                queued.snapshot.entries[0].entry_id,
                text="replacement",
                expected_revision=before.revision,
            )
            assert result.status is QueueMutationStatus.INVALID
        else:
            result = await console._prompt_queue.dispatch(
                "keep draft", session_id=session_id
            )
            assert result.status is ConsolePromptDispatchStatus.REFUSED
        assert "different session" in result.detail
        assert controller.prompt_queue_registry.snapshot(session_id) == before
        assert committed == []


@pytest.mark.asyncio
async def test_dispatch_stages_one_manual_chain_when_queue_does_not_own_work() -> None:
    fake = _FakeChatController(accepted=False)
    calls = _calls()
    controller = _ui_controller(fake, calls)

    outcome = await controller.dispatch("send now", stash=None)

    assert outcome.status is ConsolePromptDispatchStatus.SENT
    assert calls["staged"] == [("send now", "session-a")]
    assert calls["committed"] == [("session-a", None)]
    assert fake.prompt_queue_registry.snapshot("session-a").total_count == 0


@pytest.mark.asyncio
async def test_runtime_custody_succeeds_before_composer_revision_is_committed() -> None:
    fake = _FakeChatController(accepted=False)
    calls = _calls()
    stash = object()
    events: list[str] = []

    controller = _ui_controller(fake, calls)
    controller._launch_chain = lambda draft, session_id: (
        events.append("custody") or "turn-a"
    )
    controller._commit_captured_draft = lambda session_id, captured: events.append(
        f"composer-commit:{session_id}"
    )

    outcome = await controller.dispatch("send now", stash=stash)

    assert outcome.status is ConsolePromptDispatchStatus.SENT
    assert events == ["custody", "composer-commit:session-a"]


@pytest.mark.asyncio
async def test_dispatch_uses_explicit_owning_session_instead_of_active_session() -> (
    None
):
    fake = _FakeChatController(accepted=False)
    calls = _calls()
    controller = _ui_controller(fake, calls)

    outcome = await controller.dispatch(
        "belongs to b", session_id="session-b", stash=None
    )

    assert outcome.status is ConsolePromptDispatchStatus.SENT
    assert calls["staged"] == [("belongs to b", "session-b")]
    assert calls["committed"] == [("session-b", None)]


@pytest.mark.asyncio
async def test_synchronous_custody_refusal_leaves_composer_revision_untouched() -> None:
    fake = _FakeChatController(accepted=False)
    calls = _calls()
    stash = object()
    staged_attachments = [object()]
    controller = _ui_controller(fake, calls)

    def refuse(_draft: str, _session_id: str) -> str:
        raise RuntimeError("custody refused")

    controller._launch_chain = refuse

    outcome = await controller.dispatch("keep me", stash=stash)

    assert outcome.status is ConsolePromptDispatchStatus.REFUSED
    assert calls["committed"] == []
    assert staged_attachments == staged_attachments


@pytest.mark.asyncio
async def test_dispatch_keeps_captured_stash_when_queue_is_full() -> None:
    fake = _FakeChatController(accepted=True)
    snapshot = fake.prompt_queue_registry.snapshot("session-a")
    for index in range(MAX_CONSOLE_QUEUE_ENTRIES):
        snapshot = fake.prompt_queue_registry.admit(
            "session-a",
            text=f"queued {index}",
            expected_revision=snapshot.revision,
        ).snapshot
    calls = _calls()
    controller = _ui_controller(fake, calls)
    stash = object()

    outcome = await controller.dispatch("must survive", stash=stash)

    assert outcome.status is ConsolePromptDispatchStatus.REFUSED
    assert calls["queued"] == []
    assert calls["staged"] == []
    assert outcome.detail == "Queue full (10/10). Manage or remove an item."


@pytest.mark.asyncio
async def test_pre_acceptance_race_keeps_captured_stash_instead_of_launching() -> None:
    fake = _FakeChatController(accepted=False)
    calls = _calls()
    controller = _ui_controller(fake, calls)
    stash = object()
    activity_calls = 0

    def activity_for(session_id: str) -> ConsoleControllerActivity:
        nonlocal activity_calls
        activity_calls += 1
        return _activity(session_id, preparing=activity_calls > 1)

    fake.activity_for = activity_for  # type: ignore[method-assign]
    outcome = await controller.dispatch("race-safe", stash=stash)

    assert outcome.status is ConsolePromptDispatchStatus.REFUSED
    assert calls["staged"] == []
    assert "Preparing" in outcome.detail


@pytest.mark.asyncio
async def test_finished_chain_boundary_reroutes_to_one_normal_send() -> None:
    fake = _FakeChatController(accepted=False)
    calls = _calls()
    controller = _ui_controller(fake, calls)
    activity_calls = 0

    def activity_for(session_id: str) -> ConsoleControllerActivity:
        nonlocal activity_calls
        activity_calls += 1
        return _activity(session_id, accepted=activity_calls == 1)

    fake.activity_for = activity_for  # type: ignore[method-assign]
    outcome = await controller.dispatch("boundary", stash=None)

    assert outcome.status is ConsolePromptDispatchStatus.SENT
    assert calls["staged"] == [("boundary", "session-a")]
    assert fake.prompt_queue_registry.snapshot("session-a").total_count == 0


def test_manager_edit_refuses_recognized_slash_command_without_mutation() -> None:
    fake = _FakeChatController(accepted=True)
    snapshot = fake.prompt_queue_registry.snapshot("session-a")
    snapshot = fake.prompt_queue_registry.admit(
        "session-a", text="ordinary", expected_revision=snapshot.revision
    ).snapshot
    entry_id = snapshot.entries[0].entry_id
    calls = _calls()
    controller = _ui_controller(
        fake,
        calls,
        edit_refusal=lambda text: (
            "Slash commands cannot be queued." if text == "/help" else ""
        ),
    )

    result = controller.edit_waiting(
        "session-a",
        entry_id,
        text="/help",
        expected_revision=snapshot.revision,
    )

    assert result.status is QueueMutationStatus.INVALID
    after = fake.prompt_queue_registry.read_waiting_text(
        "session-a",
        entry_id=entry_id,
        expected_revision=snapshot.revision,
    )
    assert after.text == "ordinary"


@pytest.mark.asyncio
async def test_use_current_context_rejects_an_epoch_that_changed_after_review() -> None:
    fake = _FakeChatController(accepted=True)
    calls = _calls()
    controller = _ui_controller(fake, calls)
    snapshot = fake.prompt_queue_registry.snapshot("session-a")

    result = await controller.recover(
        "session-a",
        action="use-current-context",
        expected_revision=snapshot.revision,
        reviewed_context_epoch=7,
    )

    assert result.status is QueueMutationStatus.INVALID
    assert "changed since review" in result.detail


@pytest.mark.asyncio
@private_profile_test
async def test_dirty_queue_edit_vetoes_navigation_and_preserves_text(request) -> None:
    """TASK-31701: the one lossy Console navigation is guarded again.

    A queue-manager edit whose text diverges from the queued entry vetoes
    `flush_pending_work` (the seam app.py consults before dismissing
    overlays), preserving the modal, the typed text, and focus. Saving
    the edit clears the veto -- navigation is lossless again.
    """
    _app, host = _ready_host()
    async with host.run_test(size=(100, 30)) as pilot:
        console = await _mounted_console(host, pilot)
        controller = console._ensure_console_chat_controller()
        session_id = controller.store.active_session_id
        snapshot = controller.prompt_queue_registry.snapshot(session_id)
        snapshot = controller.prompt_queue_registry.begin_chain(
            session_id,
            context_epoch=controller.store.conversation_context_epoch(session_id),
            expected_revision=snapshot.revision,
        ).snapshot
        controller.prompt_queue_registry.admit(
            session_id,
            text="original queued text",
            expected_revision=snapshot.revision,
        )
        controller.prompt_queue_coordinator.publish_registry_change(session_id)
        await console._sync_native_console_chat_ui()
        await pilot.pause()
        await pilot.click("#console-prompt-queue-manage")
        await pilot.pause()
        modal = host.screen_stack[-1]
        assert isinstance(modal, ConsolePromptQueueModal)

        # No edit open: nothing to protect.
        assert console.flush_pending_work() is True

        await pilot.click("#console-prompt-queue-edit")
        edit = modal.query_one("#console-prompt-queue-edit-input", TextArea)
        # Edit open but text unchanged: still lossless, still allowed.
        assert console.flush_pending_work() is True

        edit.text = "edited but not yet saved"
        assert console.flush_pending_work() is False, (
            "a dirty edit must veto -- the navigation seam would dismiss "
            "the modal and discard the typed text"
        )
        assert isinstance(host.screen_stack[-1], ConsolePromptQueueModal), (
            "the veto must not disturb the open manager"
        )
        assert edit.text == "edited but not yet saved"

        # Saving resolves the veto.
        await pilot.click("#console-prompt-queue-save")
        await pilot.pause()
        assert console.flush_pending_work() is True, (
            "a saved edit loses nothing; navigation must be allowed again"
        )


def _bare_modal_for_dirty_check(
    *, editing: str | None, edit_text: str, baseline: str | None, read_result
):
    """A ConsolePromptQueueModal with only the dirty-check's seams stubbed.

    Args:
        editing: Value for ``_editing_entry_id``.
        edit_text: Text the stubbed edit TextArea reports.
        baseline: Value for ``_editing_baseline_text``.
        read_result: What the stubbed controller's ``read_waiting_text``
            returns.

    Returns:
        The bare modal instance.
    """
    modal = ConsolePromptQueueModal.__new__(ConsolePromptQueueModal)
    modal._editing_entry_id = editing
    modal._editing_baseline_text = baseline
    modal._revision = 7
    modal.session_id = "session-1"
    modal._queue_controller = SimpleNamespace(
        read_waiting_text=lambda *args, **kwargs: read_result
    )
    modal.query_one = lambda selector, widget_type=None: SimpleNamespace(text=edit_text)
    return modal


def test_has_unsaved_edit_unit_states() -> None:
    """Isolated decision table for the dirty check (Qodo #2425).

    The race states (STALE_REVISION / LOCKED / NOT_FOUND) must judge
    against the edit-time baseline: the save path refuses them too, so
    the veto is the user's only copy of the modified text.
    """
    from tldw_chatbook.Chat.console_prompt_queue import QueueMutationStatus

    applied = SimpleNamespace(status=QueueMutationStatus.APPLIED, text="queued")

    # No edit open: never dirty.
    assert (
        _bare_modal_for_dirty_check(
            editing=None, edit_text="anything", baseline=None, read_result=applied
        ).has_unsaved_edit()
        is False
    )
    # Edit open, text matches the current queue: clean.
    assert (
        _bare_modal_for_dirty_check(
            editing="e1", edit_text="queued", baseline="queued", read_result=applied
        ).has_unsaved_edit()
        is False
    )
    # Edit open, text diverges from the current queue: dirty.
    assert (
        _bare_modal_for_dirty_check(
            editing="e1", edit_text="typed", baseline="queued", read_result=applied
        ).has_unsaved_edit()
        is True
    )
    # Race states: current unreadable -> judge against the baseline.
    for status in (
        QueueMutationStatus.STALE_REVISION,
        QueueMutationStatus.LOCKED,
        QueueMutationStatus.NOT_FOUND,
    ):
        raced = SimpleNamespace(status=status, text=None)
        assert (
            _bare_modal_for_dirty_check(
                editing="e1", edit_text="typed", baseline="queued", read_result=raced
            ).has_unsaved_edit()
            is True
        ), f"{status}: modified text must stay protected through the race"
        assert (
            _bare_modal_for_dirty_check(
                editing="e1", edit_text="queued", baseline="queued", read_result=raced
            ).has_unsaved_edit()
            is False
        ), f"{status}: an unmodified edit protects nothing"
    # Defensive: no baseline at all -> any typed text is protected.
    raced = SimpleNamespace(status=QueueMutationStatus.NOT_FOUND, text=None)
    assert (
        _bare_modal_for_dirty_check(
            editing="e1", edit_text="typed", baseline=None, read_result=raced
        ).has_unsaved_edit()
        is True
    )
    assert (
        _bare_modal_for_dirty_check(
            editing="e1", edit_text="", baseline=None, read_result=raced
        ).has_unsaved_edit()
        is False
    )


def test_flush_pending_work_unit_stack_walk() -> None:
    """Isolated contract for the screen hook (Qodo #2425).

    Vetoes (with one warning notification) when any stacked screen
    reports an unsaved edit; allows when none does or none provides the
    probe.
    """
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    class _App:
        def __init__(self, stack):
            self.screen_stack = stack

    class _Host(ChatScreen):
        # Textual's `app` is a read-only property; the bare unit needs a
        # stubbed stack, so the subclass overrides the lookup.
        _stub_app = None

        @property
        def app(self):
            return self._stub_app

    screen = _Host.__new__(_Host)
    notes: list[str] = []
    screen.notify = lambda message, **kwargs: notes.append(str(message))

    plain = SimpleNamespace()  # no has_unsaved_edit at all
    clean = SimpleNamespace(has_unsaved_edit=lambda: False)
    dirty = SimpleNamespace(has_unsaved_edit=lambda: True)

    screen._stub_app = _App([plain, clean])
    assert ChatScreen.flush_pending_work(screen) is True
    assert notes == []

    screen._stub_app = _App([plain, clean, dirty])
    assert ChatScreen.flush_pending_work(screen) is False
    assert len(notes) == 1 and "Unsaved queue edit" in notes[0]


# ---------------------------------------------------------------------------
# TASK-33621.19 (GAP1-04 / GAP5-09): the paused shelf offered Retry for a
# FAILED pause that had no failed turn behind it, and that Retry refused with
# "No matching stopped or failed turn is available." The retry target is now
# the exact turn that paused the queue -- the newest assistant turn, because a
# paused queue gates every other generation in its session -- and the shelf
# names it. With no such turn the shelf offers Resume instead.
# ---------------------------------------------------------------------------


def _store_with_turns(*turns: tuple[str, str]) -> tuple[ConsoleChatStore, list[str]]:
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole

    store = ConsoleChatStore()
    store.create_session(session_id="session-a", title="Queue owner", ephemeral=True)
    assistant_ids: list[str] = []
    for prompt, outcome in turns:
        store.append_message("session-a", role=ConsoleMessageRole.USER, content=prompt)
        # An empty assistant row is a pending generation; content is final.
        reply = store.append_message(
            "session-a",
            role=ConsoleMessageRole.ASSISTANT,
            content="reply" if outcome == "complete" else "",
        )
        if outcome == "failed":
            store.mark_message_failed(reply.id)
        elif outcome == "stopped":
            store.mark_message_stopped(reply.id)
        assistant_ids.append(reply.id)
    return store, assistant_ids


def _paused_fake(store: ConsoleChatStore, reason: PromptQueuePauseReason):
    fake = _FakeChatController(accepted=True)
    fake.store = store
    registry = fake.prompt_queue_registry
    snapshot = registry.snapshot("session-a")
    for text in ("CHARLIE", "DELTA"):
        snapshot = registry.admit(
            "session-a", text=text, expected_revision=snapshot.revision
        ).snapshot
    registry.pause("session-a", reason=reason, expected_revision=snapshot.revision)
    fake.retried: list[tuple[str, str]] = []

    async def retry_failed_queue_turn(message_id: str):
        fake.retried.append(("failed", message_id))
        return QueueMutationResultStub.applied(registry.snapshot("session-a"))

    async def retry_stopped_queue_turn(message_id: str):
        fake.retried.append(("stopped", message_id))
        return QueueMutationResultStub.applied(registry.snapshot("session-a"))

    fake.retry_failed_queue_turn = retry_failed_queue_turn
    fake.retry_stopped_queue_turn = retry_stopped_queue_turn
    return fake


class QueueMutationResultStub:
    @staticmethod
    def applied(snapshot):
        from tldw_chatbook.Chat.console_prompt_queue import PromptQueueMutationResult

        return PromptQueueMutationResult(QueueMutationStatus.APPLIED, snapshot)


def test_failed_pause_names_the_failed_turn_and_retry_targets_it() -> None:
    store, assistant_ids = _store_with_turns(
        ("Answer with the word ALPHA.", "complete"),
        ("Answer with the word BRAVO.", "failed"),
    )
    fake = _paused_fake(store, PromptQueuePauseReason.FAILED)
    controller = _ui_controller(fake, _calls())

    turn = controller.recovery_turn("session-a", action="retry-failed")
    presentation = controller.presentation_for("session-a")

    assert turn is not None
    assert turn.message_id == assistant_ids[-1]
    assert turn.preview == "Answer with the word BRAVO."
    assert presentation.state_label == 'Turn failed: "Answer with the word BRAVO."'
    assert presentation.pause_label == "Retry"
    assert presentation.primary_action == "retry-failed"
    assert presentation.next_preview == "CHARLIE"


@pytest.mark.asyncio
async def test_retry_reruns_exactly_the_named_failed_turn() -> None:
    store, assistant_ids = _store_with_turns(
        ("Answer with the word ALPHA.", "failed"),
        ("Answer with the word BRAVO.", "failed"),
    )
    fake = _paused_fake(store, PromptQueuePauseReason.FAILED)
    calls = _calls()
    controller = _ui_controller(fake, calls)
    presentation = controller.presentation_for("session-a")

    await controller.handle_primary_intent(
        "session-a",
        action=presentation.primary_action,
        expected_revision=presentation.revision,
    )

    assert fake.retried == [("failed", assistant_ids[-1])]
    assert calls["notified"] == []


@pytest.mark.asyncio
async def test_failed_pause_without_a_failed_turn_routes_the_press_to_resume() -> None:
    # The false pause from the report: every turn succeeded, and an OLDER
    # failure elsewhere in the conversation is not what paused this queue.
    # This proves only the label and the routing: resume_prompt_queue is a
    # stub here. That the real Resume drains, or lands on Context changed
    # without raising, is proven against the real controller in
    # Tests/Chat/test_console_prompt_queue_coordinator.py
    # (test_real_paused_queue_without_failed_turn_offers_resume_that_drains,
    # test_resume_after_*_does_not_raise).
    store, _assistant_ids = _store_with_turns(
        ("An older question.", "failed"),
        ("Answer with the word ALPHA.", "complete"),
        ("Answer with the word BRAVO.", "complete"),
    )
    fake = _paused_fake(store, PromptQueuePauseReason.FAILED)
    resumed: list[str] = []

    async def resume_prompt_queue(session_id: str):
        resumed.append(session_id)
        return QueueMutationResultStub.applied(
            fake.prompt_queue_registry.snapshot(session_id)
        )

    fake.resume_prompt_queue = resume_prompt_queue
    calls = _calls()
    controller = _ui_controller(fake, calls)

    assert controller.recovery_turn("session-a", action="retry-failed") is None
    presentation = controller.presentation_for("session-a")
    assert "failed" not in presentation.state_label.lower()
    assert presentation.state_label == "Paused"
    assert presentation.pause_label == "Resume"
    assert presentation.primary_action == "toggle-pause"

    await controller.handle_primary_intent(
        "session-a",
        action=presentation.primary_action,
        expected_revision=presentation.revision,
    )

    assert resumed == ["session-a"]
    assert calls["notified"] == []
    assert fake.retried == []


def test_stopped_pause_retry_target_is_the_stopped_turn_only() -> None:
    store, assistant_ids = _store_with_turns(
        ("Answer with the word ALPHA.", "stopped"),
    )
    fake = _paused_fake(store, PromptQueuePauseReason.STOPPED)
    controller = _ui_controller(fake, _calls())

    stopped = controller.recovery_turn("session-a", action="retry-stopped")
    assert stopped is not None and stopped.message_id == assistant_ids[-1]
    assert controller.recovery_turn("session-a", action="retry-failed") is None


class _HeldQueueGateway:
    """Ready local destination whose Nth reply streams only once released."""

    def __init__(self) -> None:
        import asyncio

        self.started = [asyncio.Event() for _ in range(5)]
        self.release = [asyncio.Event() for _ in range(5)]
        self.user_turns: list[str] = []

    async def resolve_for_send(self, selection):
        from Tests.console_provider_doubles import with_destination
        from tldw_chatbook.Chat.console_provider_gateway import (
            ConsoleProviderResolution,
        )

        return with_destination(
            ConsoleProviderResolution(
                provider=selection.provider,
                base_url=selection.base_url or "",
                model=(
                    selection.explicit_model
                    or selection.configured_model
                    or "test-model"
                ),
                ready=True,
                readiness_key="llama_cpp",
                execution_key="llama_cpp",
            )
        )

    async def stream_chat(self, _resolution, messages, **_kwargs):
        call = len(self.user_turns)
        self.user_turns.append(
            next(
                message["content"]
                for message in reversed(messages)
                if message.get("role") == "user"
            )
        )
        self.started[call].set()
        await self.release[call].wait()
        yield f"reply-{call + 1}"


async def _wait_until(pilot, predicate, *, timeout: float = 45.0) -> None:
    # Generous: a mounted Console under a loaded xdist worker can take tens
    # of seconds to stream one reply; the deadline only bounds a hang.
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await pilot.pause(0.02)
    assert predicate()


@pytest.mark.asyncio
@private_profile_test
async def test_shelf_pause_after_shelf_resume_lets_the_queued_turn_finish(
    request,
) -> None:
    """TASK-33621.19 review: a shelf Resume drains whole turns in its worker.

    Pressing the shelf's own Pause while that queued turn is generating
    means "pause after this turn". It used to start a second worker in the
    same exclusive group, which cancelled the drain -- killing the turn in
    flight and falling back to a paused queue.
    """

    import asyncio

    from Tests.UI.app_factory import attach_chachanotes_db
    from tldw_chatbook.Chat.chat_conversation_service import (
        ChatConversationService,
    )
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.console_prompt_queue import PromptQueueMode

    app, host = _ready_host()
    # A real (in-memory) conversation DB: the persisted durable path is the
    # one whose queued turns the reported sessions ran. The runtime checks
    # the conversation's archive state before each queued send.
    app.local_chat_conversation_service = ChatConversationService(
        attach_chachanotes_db(app)
    )
    async with host.run_test(size=(120, 30)) as pilot:
        console = await _mounted_console(host, pilot)
        controller = console._ensure_console_chat_controller()
        gateway = _HeldQueueGateway()
        controller.provider_gateway = gateway
        controller._agent_runtime_enabled = False
        session_id = controller.store.active_session_id
        registry = controller.prompt_queue_registry
        shelf = console.query_one("#console-prompt-queue", ConsolePromptQueueRegion)
        shelf_button = shelf.query_one("#console-prompt-queue-pause", Button)

        def shelf_is_current(label: str) -> bool:
            # The shelf pins each press to the revision it last painted, and
            # a UI sync that is already in flight coalesces a new request:
            # press only once the painted revision is the live one. Presses
            # go through Button.press() -- the shelf's own on_button_pressed
            # -- because a coordinate click can land on a toast under load.
            painted = shelf._presentation
            return (
                painted is not None
                and painted.revision == registry.snapshot(session_id).revision
                and str(shelf_button.label) == label
            )

        async def shelf_shows(label: str) -> None:
            import time

            deadline = time.monotonic() + 45.0
            while time.monotonic() < deadline:
                await console._sync_native_console_chat_ui()
                if shelf_is_current(label):
                    return
                await pilot.pause(0.05)
            assert shelf_is_current(label), (shelf._presentation, label)

        owner = asyncio.create_task(
            controller.run_prompt_chain("owner turn", session_id=session_id)
        )
        try:
            await _wait_until(pilot, gateway.started[0].is_set)
            snapshot = registry.snapshot(session_id)
            for text in ("first queued", "second queued", "third queued"):
                admitted = await controller.queue_prompt(
                    session_id, text=text, expected_revision=snapshot.revision
                )
                assert admitted.applied, admitted
                snapshot = admitted.snapshot
            # Pause after the owner turn, from the shelf, so the queue holds
            # three prompts behind a completed turn.
            await shelf_shows("Pause")
            shelf_button.press()
            await _wait_until(
                pilot,
                lambda: (
                    registry.snapshot(session_id).mode
                    is PromptQueueMode.PAUSE_AFTER_TURN
                ),
            )
            gateway.release[0].set()
            await asyncio.wait_for(owner, timeout=45)
            assert registry.snapshot(session_id).pause_reason is (
                PromptQueuePauseReason.MANUAL
            )

            # Shelf Resume: the drain now runs inside the shelf's worker.
            await shelf_shows("Resume")
            shelf_button.press()
            await _wait_until(pilot, gateway.started[1].is_set)
            # Shelf Pause while that queued turn generates.
            await shelf_shows("Pause")
            shelf_button.press()
            await _wait_until(
                pilot,
                lambda: (
                    registry.snapshot(session_id).mode
                    is PromptQueueMode.PAUSE_AFTER_TURN
                ),
            )
            gateway.release[1].set()
            await _wait_until(
                pilot,
                lambda: registry.snapshot(session_id).mode is PromptQueueMode.PAUSED,
            )
        finally:
            for release in gateway.release:
                release.set()
            if not owner.done():
                owner.cancel()

        final = registry.snapshot(session_id)
        assert final.pause_reason is PromptQueuePauseReason.MANUAL
        assert final.waiting_count == 2
        assert gateway.user_turns == ["owner turn", "first queued"]
        replies = [
            message
            for message in controller.store.messages_for_session(session_id)
            if message.role is ConsoleMessageRole.ASSISTANT
        ]
        assert [message.status for message in replies] == ["complete", "complete"]

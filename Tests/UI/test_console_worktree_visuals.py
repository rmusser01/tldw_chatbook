"""Explicit bounded native visual evidence; run separately from functional gates."""

import pytest
from textual.widgets import Button

from Tests.UI.test_console_worktree_recovery import CardApp, payload
from tldw_chatbook.UI.Screens.chat_screen_state import TaskResumeState
from tldw_chatbook.Widgets.Chat_Widgets.chat_task_cards import ChatTaskCards


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 40), (60, 28)])
async def test_native_card_and_recovery_visual_packet(size):
    from pathlib import Path

    from tldw_chatbook.Chat.console_worktree_recovery import RecoveryPage
    from tldw_chatbook.Widgets.Chat_Widgets.worktree_recovery_dialog import (
        WorktreeRecoveryDialog,
    )

    class StyledApp(CardApp):
        CSS_PATH = str(
            Path(__file__).resolve().parents[2]
            / "tldw_chatbook/css/tldw_cli_modular.tcss"
        )

    app = StyledApp()
    async with app.run_test(size=size) as pilot:
        state = TaskResumeState(
            pending_worktree_merge={
                **payload(),
                "source": "/workspace/agent-edits/" + "long-folder/" * 6,
                "diffstat": "\n".join(f" file-{i}.py | 12 ++++++" for i in range(30)),
            }
        )
        app.query_one(ChatTaskCards).sync_state(state)
        await pilot.pause()
        button = app.query_one(".worktree-allow", Button)
        assert button.region.bottom <= size[1]
        assert button.region.width > 0
        evidence = (
            Path(__file__).resolve().parents[2]
            / ".superpowers/sdd/2026-09-12-agent-worktree-console-recovery/visual-evidence"
        )
        evidence.mkdir(exist_ok=True)
        (evidence / f"card-{size[0]}.svg").write_text(app.export_screenshot())
        page = RecoveryPage(
            rows=tuple(
                {
                    "run_id": f"child-{i}",
                    "child_path": "/workspace/agent-edits/" + "long-folder/" * 5,
                    "writer_state": writer,
                    "mutation_state": mutation,
                    "run_status": "done",
                }
                for i, (writer, mutation) in enumerate(
                    [
                        ("drained", "unresolved"),
                        ("held", "unresolved"),
                        ("drained", "applying"),
                        ("drained", "discarded_cleanup_pending"),
                    ]
                )
            ),
            conversation_id="Conversation: Fix import ordering",
            repository="/workspace/chatbook",
        )
        app.push_screen(WorktreeRecoveryDialog(page))
        await pilot.pause()
        close = app.screen.query_one("#worktree-close", Button)
        assert close.region.bottom <= size[1]
        assert close.region.width > 0
        (evidence / f"list-{size[0]}.svg").write_text(app.export_screenshot())
        await app.pop_screen()


@pytest.mark.asyncio
async def test_final_narrow_discard_and_resolved_packet():
    from pathlib import Path

    from textual.containers import VerticalScroll

    from tldw_chatbook.Chat.console_worktree_recovery import RecoveryPage
    from tldw_chatbook.Widgets.Chat_Widgets.worktree_recovery_dialog import (
        WorktreeRecoveryDialog,
    )

    class StyledApp(CardApp):
        CSS_PATH = str(
            Path(__file__).resolve().parents[2]
            / "tldw_chatbook/css/tldw_cli_modular.tcss"
        )

    app = StyledApp()
    async with app.run_test(size=(60, 28)) as pilot:
        app.query_one(ChatTaskCards).sync_state(
            TaskResumeState(
                pending_worktree_merge={
                    **payload(),
                    "action": "discard",
                    "source": "/workspace/agent-edits/long-folder/retained-checkout",
                    "diffstat": " a.py | 2 ++",
                }
            )
        )
        await pilot.pause()
        evidence = (
            Path(__file__).resolve().parents[2]
            / ".superpowers/sdd/2026-09-12-agent-worktree-console-recovery/visual-evidence"
        )
        (evidence / "discard-60.svg").write_text(app.export_screenshot())
        rows = tuple(
            {
                "run_id": f"child-{i}",
                "child_path": "/workspace/agent-edits/" + "long-folder/" * 5,
                "writer_state": "drained",
                "mutation_state": state,
                "run_status": "done",
            }
            for i, state in enumerate(
                ("unresolved", "applied", "merged", "discarded_cleanup_pending")
            )
        )
        app.push_screen(
            WorktreeRecoveryDialog(
                RecoveryPage(
                    rows=rows,
                    conversation_id="Conversation: Fix import ordering",
                    repository="/workspace/chatbook",
                )
            )
        )
        await pilot.pause()
        app.screen.query_one("#worktree-recovery-rows", VerticalScroll).scroll_end(
            animate=False
        )
        await pilot.pause()
        (evidence / "resolved-60.svg").write_text(app.export_screenshot())
        assert app.screen.query_one("#worktree-close", Button).region.bottom <= 28
        await app.pop_screen()

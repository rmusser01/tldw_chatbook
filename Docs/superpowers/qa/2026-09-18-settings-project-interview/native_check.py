"""TASK-32775 native workspace creation and fixed project-context interview.

Usage: native_check.py PRIVATE_PROFILE TMUX_SOCKET SESSION
All workspace, Persona and context writes stay in the private profile.
"""

import asyncio
import hashlib
import json
import os
import runpy
import secrets
import subprocess
import sys
import traceback
from pathlib import Path


def main():
    root = Path(sys.argv[1]).resolve()
    socket, session = sys.argv[2:4]
    (root / "launch.json").write_text(json.dumps({"pid": os.getpid()}))
    qa = Path(__file__).resolve().parents[1]
    runpy.run_path(str(qa / "2026-09-16-ingest-lifecycle/native_check.py"))[
        "validate_profile"
    ](root)
    evidence = root / "evidence"
    evidence.mkdir(exist_ok=False)
    os.environ.update(
        HOME=str(root / "home"),
        USERPROFILE=str(root / "home"),
        TLDW_CONFIG_PATH=str(root / "config.toml"),
        XDG_DATA_HOME=str(root / "data"),
        XDG_CONFIG_HOME=str(root / "config"),
        PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
    )
    for key in (
        "NO_COLOR",
        "OPENAI_API_KEY",
        "LLAMA_CPP_API_KEY",
        "SEARX_URL",
        "SERPER_API_KEY",
    ):
        os.environ.pop(key, None)

    from textual.css.query import NoMatches, QueryError
    from textual_image._terminal import probe_terminal

    from tldw_chatbook.app import TldwCli

    probe_terminal()
    app = TldwCli()
    source = Path(__file__).resolve().parents[4]
    result = {
        "pid": os.getpid(),
        "cells": [],
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_hashes": {
            p: hashlib.sha256((source / p).read_bytes()).hexdigest()
            for p in (
                "tldw_chatbook/UI/Screens/settings_screen.py",
                "tldw_chatbook/UI/Screens/profile_interview_screen.py",
                "tldw_chatbook/Widgets/Settings_Widgets/personal_context_review_modal.py",
                "tldw_chatbook/Personal_Context/interview_coordinator.py",
                "tldw_chatbook/css/components/_profile_interview.tcss",
                "tldw_chatbook/css/tldw_cli_modular.tcss",
                "tldw_chatbook/css/features/_settings.tcss",
                "tldw_chatbook/css/core/_variables.tcss",
                "tldw_chatbook/css/components/_dialogs.tcss",
                "tldw_chatbook/css/widget_defaults_scoped.tcss",
                "tldw_chatbook/css/widget_defaults_self.tcss",
                "tldw_chatbook/css/screen_agentic_settings.tcss",
                "tldw_chatbook/Workspaces/registry_service.py",
                "tldw_chatbook/config.py",
            )
        },
    }

    def record():
        (evidence / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    async def tmux(*args):
        return await asyncio.to_thread(
            subprocess.run,
            ["/opt/homebrew/bin/tmux", "-L", socket, *args],
            check=True,
            text=True,
            capture_output=True,
        )

    async def journey(pilot):
        async def wait_for(predicate, label):
            result["waiting_for"] = label
            record()
            deadline = asyncio.get_running_loop().time() + 35
            while asyncio.get_running_loop().time() < deadline:
                try:
                    if predicate():
                        await pilot.pause()
                        return
                except (NoMatches, QueryError):
                    pass
                await pilot.pause(0.03)
            raise AssertionError(label)

        try:
            assert app._instance_lock_status.acquired
            assert type(app._driver).__name__ == "LinuxDriver"
            assert sys.stdout.isatty() and sys.stderr.isatty()
            assert app.console.file.isatty()
            result.update(driver="LinuxDriver", tty_streams=True, lock_acquired=True)
            await wait_for(lambda: getattr(app, "_ui_ready", False), "Startup")
            await pilot.press("f4")
            await wait_for(
                lambda: type(app.screen).__name__ == "SettingsScreen", "Settings"
            )
            handles = {
                key: value
                for key, value in vars(app).items()
                if key.endswith("_db") and value is not None
            }
            assert handles
            result["live_database_attributes"] = sorted(handles)

            async def category(name, value):
                await pilot.press("escape", "/", *name, "enter")
                await wait_for(lambda: app.screen.active_category == value, name)

            async def focus(selector):
                control = app.screen.query_one(selector)
                control.focus()
                await pilot.pause()
                await pilot.wait_for_scheduled_animations()
                await pilot.pause()
                if isinstance(control, Button):
                    await wait_for(
                        lambda: not control.has_class("-active"), "Button ready"
                    )
                region, clip = app.screen._compositor.visible_widgets[control]
                assert region.intersection(clip) == region
                return control

            async def edit(selector, value):
                field = await focus(selector)
                await pilot.press("home", "shift+end", "backspace", *value)
                await wait_for(lambda: field.value == value, "Field edit")
                return field

            async def capture(stem):
                await wait_for(lambda: not app.screen.query("Toast"), "Notices clear")
                app.save_screenshot(stem + ".svg", path=str(evidence))
                pane = await tmux("capture-pane", "-p", "-t", session)
                (evidence / (stem + ".txt")).write_text(pane.stdout)

            from textual.widgets import Button, Checkbox, Input

            from tldw_chatbook.Personal_Context.bootstrap import (
                bootstrap_personal_context_service,
            )
            from tldw_chatbook.Personal_Context.key_protector import (
                PassphraseProfileKeyProtector,
            )
            from tldw_chatbook.Personal_Context.paths import (
                get_personal_context_db_path,
            )
            from tldw_chatbook.UI.Screens.profile_interview_screen import (
                ProfileInterviewScreen,
            )
            from tldw_chatbook.Widgets.Settings_Widgets.personal_context_review_modal import (
                PersonalContextReviewModal,
            )

            # Real encrypted persistence through the production protector and service.
            # Private fixture password is process-local, never a user's credential.
            phrase = secrets.token_urlsafe(32)
            protector = PassphraseProfileKeyProtector(
                root / "profile-key.json", lambda: phrase
            )
            service = bootstrap_personal_context_service(key_protector=protector)
            app._personal_context_service = service
            # Cold auto-provisioning currently fails while the Tool Profile guard
            # is deferred (preserved run001). Qualify interview behavior after
            # explicit first-use initialization; the cold-path defect stays open.
            await category("Tool Profiles", "tool-profiles")
            await wait_for(
                lambda: app.tool_pack_service is not None, "Tool Profiles ready"
            )
            registry = app.workspace_registry_service
            result["fixture_setup"] = (
                "Real workspace registry, auto-provisioned Persona, fixed local interview coordinator, memory-only draft fallback, encrypted Personal Context repository with private passphrase protector. No provider requests."
            )

            async def settled():
                await wait_for(
                    lambda: (
                        not getattr(app.screen, "_category_pane_swap_pending", False)
                    ),
                    "Pane settled",
                )
                await pilot.wait_for_scheduled_animations()
                await pilot.pause()

            def paint():
                return "\n".join(
                    strip.text for strip in app.screen._compositor.render_strips()
                )

            async def press(selector, key="enter"):
                await focus(selector)
                await pilot.press(key)
                await pilot.pause()

            async def loaded():
                await wait_for(
                    lambda: (
                        isinstance(app.screen, ProfileInterviewScreen)
                        and not app.screen._busy
                        and app.screen._session is not None
                    ),
                    "Interview loaded",
                )
                await settled()

            async def create(name):
                await category("Workspaces", "workspaces")
                await settled()
                await press("#settings-workspace-create")
                await wait_for(
                    lambda: type(app.screen).__name__ == "WorkspaceCreateModal",
                    "Create dialog",
                )
                await edit("#workspace-create-name", name)
                await press("#workspace-create-profile-interview", "space")
                assert app.screen.query_one(
                    "#workspace-create-profile-interview", Checkbox
                ).value
                await press("#workspace-create-confirm")
                await loaded()
                workspace = next(
                    w for w in registry.list_workspaces() if w.name == name
                )
                assert workspace.assistant_defaults is not None
                persona = app.local_character_persona_service.get_persona_profile(
                    workspace.assistant_defaults.assistant_id
                )
                assert persona is not None
                assert workspace.assistant_defaults.persona_memory_mode == "read_only"
                scope = app.screen._session.scope_id
                assert (
                    service.list_workspace_bindings()[scope]["local_workspace_id"]
                    == workspace.workspace_id
                )
                return workspace, scope

            for theme in ("textual-dark", "textual-light"):
                for size in ((170, 48), (80, 24)):
                    stem = f"{theme}-{size[0]}x{size[1]}"
                    app.theme = theme
                    await tmux(
                        "resize-window",
                        "-t",
                        session,
                        "-x",
                        str(size[0]),
                        "-y",
                        str(size[1]),
                    )
                    await wait_for(
                        lambda size=size: (
                            app.size.width == size[0] and app.size.height == size[1]
                        ),
                        "Terminal resized",
                    )
                    cancelled, cancel_scope = await create("Cancel " + stem)
                    await edit(
                        "#profile-interview-answer", "Unsubmitted project detail"
                    )
                    await pilot.press("escape")
                    await wait_for(
                        lambda: (
                            type(app.screen).__name__ == "ProfileInterviewCancelModal"
                        ),
                        "Cancel disclosure",
                    )
                    await capture(stem + "-cancel")
                    await press("#profile-interview-cancel-discard")
                    await wait_for(
                        lambda: type(app.screen).__name__ == "SettingsScreen",
                        "Cancelled interview returned",
                    )
                    assert registry.get_workspace(cancelled.workspace_id) is not None
                    assert service.list_records(scope_ids=(cancel_scope,)) == ()

                    workspace, scope = await create("Review " + stem)
                    interview = app.screen
                    coordinator = interview._coordinator
                    original_answer = coordinator.answer
                    failed = False

                    def fail_once(session_id, answer, original_answer=original_answer):
                        nonlocal failed
                        if not failed:
                            failed = True
                            raise RuntimeError("injected private draft write failure")
                        return original_answer(session_id, answer)

                    coordinator.answer = fail_once
                    first = "Accessible reader for " + stem
                    await edit("#profile-interview-answer", first)
                    assert "What is the main outcome for this workspace?" in paint()
                    assert first in paint()
                    await capture(stem + "-answer")
                    await pilot.press("tab", "enter")
                    await wait_for(
                        lambda interview=interview: not interview._busy,
                        "Rejected answer",
                    )
                    assert (
                        interview.query_one("#profile-interview-answer", Input).value
                        == first
                    )
                    await focus("#profile-interview-answer")
                    assert first in paint()
                    await capture(stem + "-retry")
                    await pilot.press("tab", "enter")
                    await wait_for(
                        lambda interview=interview: (
                            not interview._busy and len(interview._session.turns) == 1
                        ),
                        "Accepted first answer",
                    )
                    assert (
                        interview.query_one("#profile-interview-answer", Input).value
                        == ""
                    )
                    second = "Readers who use keyboard navigation"
                    await edit("#profile-interview-answer", second)
                    await pilot.press("tab", "enter")
                    await wait_for(
                        lambda interview=interview: (
                            not interview._busy and len(interview._session.turns) == 2
                        ),
                        "Accepted second answer",
                    )
                    await press("#profile-interview-finish")
                    await wait_for(
                        lambda: isinstance(app.screen, PersonalContextReviewModal),
                        "Final review",
                    )
                    assert len(app.screen._diff.changes) == 2
                    assert service.list_records(scope_ids=(scope,)) == ()
                    reviewed = "Reviewed reader for " + stem
                    await edit("#personal-context-review-value-0", reviewed)
                    await press("#personal-context-review-apply-0")
                    await wait_for(
                        lambda reviewed=reviewed: (
                            not app.screen._busy
                            and app.screen._diff.changes[
                                0
                            ].change.proposed_payload.outcome
                            == reviewed
                        ),
                        "Review edit applied",
                    )

                    def unavailable_resume(_session_id):
                        raise RuntimeError("injected temporary draft read failure")

                    coordinator.resume = unavailable_resume
                    await pilot.press("escape")
                    await loaded()
                    assert app.screen is interview
                    button = interview.query_one("#profile-interview-review", Button)
                    assert button.display and not button.disabled
                    await focus("#profile-interview-review")
                    await capture(stem + "-review-return")
                    await pilot.press("enter")
                    await wait_for(
                        lambda: isinstance(app.screen, PersonalContextReviewModal),
                        "Review reopened",
                    )
                    review = app.screen
                    await focus("#personal-context-review-value-0")
                    selected_value = review.query_one(
                        "#personal-context-review-value-0", Input
                    ).value
                    assert selected_value == reviewed
                    assert selected_value in paint()
                    await press("#personal-context-review-select-1", "space")
                    assert len(review.selected_change_ids) == 1
                    wanted = review._diff.changes[0].change.proposed_payload
                    await focus("#personal-context-review-save-only")
                    await capture(stem + "-review")
                    await pilot.press("enter")
                    await wait_for(
                        lambda: type(app.screen).__name__ == "SettingsScreen",
                        "Context saved and returned",
                    )
                    records = service.list_records(scope_ids=(scope,))
                    assert len(records) == 1 and records[0].payload == wanted
                    assert not service.status().runtime_enabled
                    assert (
                        registry.get_workspace(
                            workspace.workspace_id
                        ).assistant_defaults
                        == workspace.assistant_defaults
                    )
                    result["cells"].append(
                        {
                            "theme": theme,
                            "size": size,
                            "workspace_id": workspace.workspace_id,
                            "cancelled_workspace_id": cancelled.workspace_id,
                            "scope_id": scope,
                            "record_id": records[0].record_id,
                            "auto_persona_created": True,
                            "cancel_preserves_workspace": True,
                            "cancel_creates_no_context": True,
                            "failed_answer_retained": True,
                            "retry_accepted_once": True,
                            "review_return_restored": True,
                            "applied_edit_preserved_without_resume": True,
                            "only_selected_context_saved": True,
                            "runtime_disabled": True,
                        }
                    )
                    record()
            # Reopen the encrypted repository through its real protector and service.
            reopened = bootstrap_personal_context_service(
                db_path=get_personal_context_db_path(), key_protector=protector
            )
            for cell in result["cells"]:
                records = reopened.list_records(scope_ids=(cell["scope_id"],))
                assert len(records) == 1 and records[0].record_id == cell["record_id"]
            result["reopened_encrypted_records"] = 4
            result["passed"] = True
        except Exception:  # noqa: BLE001 - retain failed-run diagnostics
            result.update(
                passed=False,
                error=traceback.format_exc(),
                focused_id=getattr(app.focused, "id", None),
                focused_region=str(getattr(app.focused, "region", None)),
            )
            app.save_screenshot("failed-state.svg", path=str(evidence))
        finally:
            record()
            if type(app.screen).__name__ == "PersonalContextReviewModal":
                await pilot.press("escape")
                await pilot.pause()
            if type(app.screen).__name__ == "ProfileInterviewScreen":
                await pilot.press("escape")
                await pilot.pause()
                if type(app.screen).__name__ == "ProfileInterviewCancelModal":
                    app.screen.query_one("#profile-interview-cancel-discard").focus()
                    await pilot.press("enter")
                    await pilot.pause()
            await tmux("send-keys", "-t", session, "C-q")

    app.run(auto_pilot=journey, size=(170, 48))
    result.update(app_run_returned=True)
    record()
    raise SystemExit(0 if result.get("passed") else 1)


if __name__ == "__main__":
    main()

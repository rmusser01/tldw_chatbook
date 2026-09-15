"""Exercise Prompt Use in Console in an isolated native TldwCli profile."""

import asyncio
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

root = Path(sys.argv[1]).resolve()
socket, session = sys.argv[2:4]
os.environ["TLDW_CONFIG_PATH"] = str(root / "config.toml")
os.environ["XDG_DATA_HOME"] = str(root / "data")
os.environ["XDG_CONFIG_HOME"] = str(root / "config")

from textual import events
from textual.css.query import NoMatches, QueryError
from textual.screen import ModalScreen
from textual.widgets import Input, TextArea

from tldw_chatbook.app import TldwCli
from tldw_chatbook.Widgets.Console import ConsoleComposerBar
from tldw_chatbook.Widgets.Console.prompt_variables_dialog import PromptVariablesDialog

app = TldwCli()
result = {"pid": os.getpid(), "steps": []}


def record():
    (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")


def painted(widget):
    region = widget.region
    strips = list(app.screen._compositor.render_strips())
    return "\n".join(
        strips[y].crop(region.x, region.right).text
        for y in range(max(0, region.y), min(region.bottom, len(strips)))
    )


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
        deadline = asyncio.get_running_loop().time() + 25
        while asyncio.get_running_loop().time() < deadline:
            try:
                if predicate():
                    await pilot.pause()
                    return
            except (NoMatches, QueryError):
                pass
            await pilot.pause(0.03)
        raise AssertionError(label)

    async def activate(surface, selector, label):
        target = surface.query_one(selector)
        target.focus()
        await wait_for(
            lambda: surface.focused is target and label in painted(target),
            f"readable {selector}",
        )
        await wait_for(lambda: not target.has_class("-active"), "prior press finished")
        await pilot.press("enter")

    async def capture(name):
        await wait_for(lambda: not app.query("Toast"), "transient notices expire")
        app.save_screenshot(name, path=str(root))

    async def console_ready():
        await wait_for(
            lambda: (
                type(app.screen).__name__ == "ChatScreen"
                and bool(app.screen.query("#console-native-composer"))
            ),
            "Console ready",
        )
        screen = app.screen
        await wait_for(
            lambda: app.console_prompt_target_projection() is not None,
            "published target",
        )
        return screen, screen.query_one("#console-native-composer", ConsoleComposerBar)

    async def library_ready():
        await pilot.press("ctrl+3")
        await wait_for(
            lambda: type(app.screen).__name__ == "LibraryScreen", "Library ready"
        )
        return app.screen

    async def open_saved_prompt(screen, name, system, user):
        # Seed a saved source; this probe reviews insertion, not authoring.
        prompt_id, _, _ = app.prompts_db.add_prompt(
            name=name, author="", details="", system_prompt=system, user_prompt=user
        )
        await screen._select_library_rail_row("browse-prompts")
        selector = f"#library-prompt-row-{prompt_id}"
        await wait_for(lambda: bool(screen.query(selector)), "saved Prompt row")
        await activate(screen, selector, name)
        await wait_for(
            lambda: (
                screen._prompts_state.selected_prompt_id == prompt_id
                and bool(screen.query("#library-prompt-user"))
                and screen.query_one("#library-prompt-user", TextArea).text == user
            ),
            "saved source loaded",
        )
        await activate(screen, "#library-prompt-mode-basic", "Basic")
        await wait_for(
            lambda: screen.query_one("#library-prompt-user", TextArea).text == user,
            "source retained in Basic",
        )
        return prompt_id

    def value_input(dialog, name):
        index = next(i for i, v in enumerate(dialog._plan.variables) if v.name == name)
        return dialog.query_one(f"#prompt-variable-value-{index}", Input)

    async def variables_ready():
        await wait_for(
            lambda: (
                isinstance(app.screen, PromptVariablesDialog)
                and bool(app.screen.query("#prompt-variables-cancel"))
            ),
            "variables ready",
        )
        return app.screen

    async def wait_draft(composer, expected):
        await wait_for(
            lambda: composer.draft_text() == expected, "exact appended draft"
        )
        await wait_for(
            lambda: (
                app.screen.focused is not None
                and (
                    app.screen.focused in composer.ancestors_with_self
                    or composer in app.screen.focused.ancestors_with_self
                )
            ),
            "composer focus",
        )

    try:
        assert app._instance_lock_status.acquired
        result["exclusive_profile"] = True
        await wait_for(lambda: getattr(app, "_ui_ready", False), "app ready")
        result["driver"] = type(app._driver).__name__
        assert result["driver"] == "LinuxDriver"
        await pilot.press("ctrl+2")
        console, composer = await console_ready()
        for width, height, theme in (
            (170, 48, "textual-dark"),
            (80, 24, "textual-light"),
        ):
            await tmux(
                "resize-window", "-t", session, "-x", str(width), "-y", str(height)
            )
            await wait_for(
                lambda width=width, height=height: tuple(app.size) == (width, height),
                "native size",
            )
            app.theme = theme
            store = console._ensure_console_chat_store()
            session_id = store.active_session_id
            before_system = store.session_settings(session_id).system_prompt
            composer.clear_draft()
            composer.insert_text("Existing native draft")
            screen = await library_ready()
            direct_name = f"Native direct {os.getpid()} {width}"
            direct_id = await open_saved_prompt(
                screen, direct_name, "", "Keep {{literal}}."
            )
            await activate(screen, "#library-prompt-insert-console", "Use in Console")
            console, composer = await console_ready()
            expected = "Existing native draft\nKeep {literal}."
            await wait_draft(composer, expected)
            assert store.session_settings(session_id).system_prompt == before_system
            await capture(f"direct-{width}.svg")
            screen = await library_ready()
            variable_name = f"Native variables {os.getpid()} {width}"
            variable_id = await open_saved_prompt(
                screen, variable_name, "Use {tone}.", "Hello {name}; literal {{name}}."
            )
            await activate(screen, "#library-prompt-insert-console", "Use in Console")
            dialog = await variables_ready()
            value_input(dialog, "name").value = "discard me"
            await pilot.press("escape")
            await wait_for(lambda screen=screen: app.screen is screen, "dialog Cancel")
            assert composer.draft_text() == expected
            assert store.session_settings(session_id).system_prompt == before_system
            await activate(screen, "#library-prompt-insert-console", "Use in Console")
            dialog = await variables_ready()
            assert value_input(dialog, "name").value == ""
            value_input(dialog, "name").value = "Zoë {other}"
            checkbox = dialog.query_one("#prompt-variables-apply-system")
            assert not checkbox.value
            checkbox.focus()
            await pilot.press("space")
            await wait_for(
                lambda dialog=dialog: len(dialog._plan.variables) == 2,
                "System authorized",
            )
            value_input(dialog, "tone").value = "calm"
            assert "R" not in painted(checkbox)
            await capture(f"authorized-{width}.svg")
            await activate(dialog, "#prompt-variables-apply", "Apply")
            console, composer = await console_ready()
            expected += "\nHello Zoë {other}; literal {name}."
            await wait_draft(composer, expected)
            assert store.session_settings(session_id).system_prompt == "Use calm."
            await wait_for(
                lambda console=console: (
                    str(console.query_one("#console-system-prompt-chip").render())
                    == "System Prompt: set"
                ),
                "System status chip updated",
            )
            await capture(f"applied-{width}.svg")
            screen = await library_ready()
            await activate(screen, "#library-prompt-insert-console", "Use in Console")
            dialog = await variables_ready()
            assert not dialog.query_one("#prompt-variables-apply-system").value
            await activate(
                dialog, "#prompt-variables-original", "Use original placeholders"
            )
            console, composer = await console_ready()
            expected += "\nHello {name}; literal {{name}}."
            await wait_draft(composer, expected)
            assert store.session_settings(session_id).system_prompt == "Use calm."
            screen = await library_ready()
            await pilot.press("ctrl+2")
            console, composer = await console_ready()
            assert composer.draft_text() == expected
            result["steps"].append(
                {
                    "size": [width, height],
                    "theme": theme,
                    "direct_prompt_id": direct_id,
                    "direct_prompt_name": direct_name,
                    "variable_prompt_id": variable_id,
                    "variable_prompt_name": variable_name,
                    "direct_escaped_text_appended": True,
                    "cancel_changes_nothing": True,
                    "values_reset_on_reopen": True,
                    "system_requires_opt_in": True,
                    "checkbox_has_no_clipped_label": True,
                    "system_status_chip_set": True,
                    "rendered_user_text_appended": True,
                    "original_source_appended": True,
                    "navigation_does_not_duplicate_handoff": True,
                    "final_draft": expected,
                    "final_system": "Use calm.",
                }
            )
            record()
        result["passed"] = True
        result["normal_quit_requested"] = True
        record()
        app.post_message(events.Key("ctrl+q", "\x11"))
    except Exception:  # noqa: BLE001 - Record failed native assertions before cleanup.
        result["passed"] = False
        result["error"] = traceback.format_exc()
        app.save_screenshot("failed-state.svg", path=str(root))
        record()
        if isinstance(app.screen, ModalScreen):
            await pilot.press("escape")
        app.post_message(events.Key("ctrl+q", "\x11"))


app.run(auto_pilot=journey, size=(170, 48))
result["app_run_returned"] = True
record()
raise SystemExit(0 if result.get("passed") else 1)

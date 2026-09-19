"""Exercise read-only Skills Files navigation in an isolated native TldwCli profile."""

import asyncio
import hashlib
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

os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.null.Keyring"
os.environ.pop("NO_COLOR", None)

from textual import events
from textual.css.query import NoMatches, QueryError
from textual.screen import ModalScreen
from textual.widgets import Input, Static, TextArea

from tldw_chatbook.app import TldwCli

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
        deadline = asyncio.get_running_loop().time() + 20
        while asyncio.get_running_loop().time() < deadline:
            try:
                if predicate():
                    await pilot.pause()
                    return
            except (NoMatches, QueryError):
                pass
            await pilot.pause(0.03)
        raise AssertionError(f"{label}: focus={app.screen.focused!r}")

    async def activate(selector, label):
        await wait_for(lambda: bool(app.screen.query(selector)), selector)

        def focus_current():
            target = app.screen.query_one(selector)
            target.focus()
            return target.has_focus and target.region.width > 0

        await wait_for(focus_current, f"focus {selector}")
        target = app.screen.query_one(selector)
        if label:
            await wait_for(lambda: label in painted(target), f"readable {selector}")
        await wait_for(lambda: not target.has_class("-active"), "action feedback")
        await pilot.press("enter")
        await pilot.pause()

    async def expect_focus(selector, label):
        await wait_for(
            lambda: (
                app.screen.focused is app.screen.query_one(selector)
                and label in painted(app.screen.focused)
            ),
            f"natural focus {selector}",
        )

    async def capture(name):
        await wait_for(lambda: not app.query("Toast"), "notices expired")
        app.save_screenshot(name, path=str(root))

    try:
        assert app._instance_lock_status.acquired
        result["exclusive_profile"] = True
        await wait_for(lambda: getattr(app, "_ui_ready", False), "app ready")
        assert type(app._driver).__name__ == "LinuxDriver"
        result["driver"] = type(app._driver).__name__
        await pilot.press("ctrl+3")
        await wait_for(lambda: type(app.screen).__name__ == "LibraryScreen", "Library")
        screen = app.screen
        local = app.local_skills_service
        long_path = "references/" + "a" * 75 + "/" + "b" * 85 + ".md"
        supporting = {
            "references/00-guide.md": "café\n",
            "references/01-empty.txt": "",
            long_path: "Long path contents.\n",
            **{f"references/guide-{i:02}.md": "Read this guide.\n" for i in range(60)},
            "references/zz-last.md": "Final supporting file.\n",
        }
        await local.create_skill(
            name="files-bundle",
            content="---\nname: files-bundle\ndescription: Original description\n---\nOriginal body.",
            supporting_files=supporting,
        )
        await local.create_skill(
            name="files-empty",
            content="---\nname: files-empty\ndescription: Empty inventory\n---\nBody.",
        )
        bundle = local.skills_dir / "files-bundle"
        binary = bundle / "assets/logo.bin"
        binary.parent.mkdir()
        binary.write_bytes(b"\x00\xff\x10\x20\x30")

        def hashes():
            return {
                str(path.relative_to(bundle)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in bundle.rglob("*")
                if path.is_file()
            }

        result["skills_dir"] = str(local.skills_dir)
        result["original_hashes"] = hashes()
        trust = app.local_skill_trust_service
        result["original_trust"] = trust.status_for_skill("files-bundle").trust_status

        async def open_items():
            shell = screen.query_one("#library-skills-reader-shell")
            await wait_for(lambda: shell.effective_layout.reader_width > 0, "layout")
            if not shell.effective_layout.items_open:
                await activate("#library-skills-items-grip", None)

        for width, height, theme in (
            (170, 48, "textual-dark"),
            (80, 24, "textual-light"),
        ):
            await tmux(
                "resize-window", "-t", session, "-x", str(width), "-y", str(height)
            )
            await wait_for(
                lambda width=width, height=height: tuple(app.size) == (width, height),
                "terminal resize",
            )
            app.theme = theme
            await screen._select_library_rail_row("browse-skills")
            await wait_for(
                lambda: bool(screen.query("#library-skills-items-grip")), "Skills"
            )
            await open_items()
            await activate("#library-skill-row-files-empty", "files-empty")
            await activate("#library-skill-mode-files", "Files")
            await wait_for(
                lambda: bool(screen.query("#library-skill-supporting")), "empty Files"
            )
            await expect_focus("#library-skill-mode-files", "Files")
            assert "No supporting files." in painted(
                screen.query_one("#library-skill-supporting")
            )
            await capture(f"empty-{width}.svg")

            await open_items()
            await activate("#library-skill-row-files-bundle", "files-bundle")
            await activate("#library-skill-mode-files", "Files")
            await wait_for(
                lambda: bool(screen.query("#library-skill-supporting")), "bundle Files"
            )
            await expect_focus("#library-skill-mode-files", "Files")
            region = screen.query_one("#library-skill-files-region")
            assert not region.query(Input) and not region.query(TextArea)
            copy = str(screen.query_one("#library-skill-supporting", Static).renderable)
            assert "assets/logo.bin — 5 bytes (binary)" in copy
            assert "references/00-guide.md (6 bytes)" in copy
            assert "references/01-empty.txt (0 bytes)" in copy
            assert long_path in copy and "SKILL.md" not in copy
            await capture(f"inventory-{width}.svg")
            await pilot.press("end")
            await wait_for(
                lambda: (
                    "references/zz-last.md"
                    in painted(screen.query_one("#library-skill-supporting"))
                ),
                "last file visible after End",
            )
            await capture(f"last-file-{width}.svg")
            await pilot.press("home")
            await expect_focus("#library-skill-mode-files", "Files")
            await activate("#library-skill-mode-edit", "Edit")
            await wait_for(lambda: bool(screen.query("#library-skill-body")), "editor")
            screen.query_one(
                "#library-skill-body", TextArea
            ).text = "Keep this unsaved draft."
            screen.query_one(
                "#library-skill-description", Input
            ).value = "Unsaved description"
            await pilot.pause()
            assert screen._skills_state.dirty
            await activate("#library-skill-mode-files", "Files")
            await wait_for(
                lambda: bool(screen.query("#library-skill-files-region")), "dirty Files"
            )
            await expect_focus("#library-skill-mode-files", "Files")
            assert screen._skills_state.dirty
            await activate("#library-skill-mode-edit", "Edit")
            await wait_for(
                lambda: bool(screen.query("#library-skill-body")), "return editor"
            )
            assert (
                screen.query_one("#library-skill-body", TextArea).text
                == "Keep this unsaved draft."
            )
            assert (
                screen.query_one("#library-skill-description", Input).value
                == "Unsaved description"
            )
            assert screen._skills_state.dirty
            await activate("#library-skill-discard", "Discard changes")
            await expect_focus("#library-skill-row-files-bundle", "files-bundle")
            assert hashes() == result["original_hashes"]
            assert (
                trust.status_for_skill("files-bundle").trust_status
                == result["original_trust"]
            )
            result["steps"].append(
                {
                    "size": [width, height],
                    "theme": theme,
                    "empty_inventory_readable": True,
                    "nested_text_binary_and_byte_sizes": True,
                    "keyboard_end_reaches_last_file": True,
                    "keyboard_home_restores_files_tab": True,
                    "draft_survives_files_round_trip": True,
                    "discard_returns_to_bundle_row": True,
                    "file_hashes_and_trust_unchanged": True,
                }
            )
            record()
        result["passed"] = True
        result["normal_quit_requested"] = True
        record()
        app.post_message(events.Key("ctrl+q", "\x11"))
    except Exception:  # noqa: BLE001 - preserve evidence before normal shutdown
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

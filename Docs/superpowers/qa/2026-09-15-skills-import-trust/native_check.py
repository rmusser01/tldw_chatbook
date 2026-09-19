"""Exercise Skills import and exact trust review in an isolated native TldwCli profile."""

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
from textual.widgets import Input, Static

from tldw_chatbook.app import TldwCli
from tldw_chatbook.UI.Library_Modules.skill_import_choice_modal import (
    SkillImportChoiceModal,
)
from tldw_chatbook.UI.Screens.skills_screen import (
    SkillTrustBootstrapModal,
    SkillTrustPassphraseModal,
)

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
        result["driver"] = type(app._driver).__name__
        assert result["driver"] == "LinuxDriver"
        await pilot.press("ctrl+3")
        await wait_for(lambda: type(app.screen).__name__ == "LibraryScreen", "Library")
        screen = app.screen
        local = app.local_skills_service
        app.theme = "textual-dark"
        trust = app.local_skill_trust_service
        result["skills_dir"] = str(local.skills_dir)
        result["trust_dir"] = str(trust.trust_store.store_dir)
        assert trust.reduced_rollback_protection
        result["private_file_marker"] = True
        await local.create_skill(
            name="native-baseline",
            content="---\nname: native-baseline\ndescription: Synthetic baseline\n---\nBaseline only.\n",
        )
        await screen._select_library_rail_row("browse-skills")
        await wait_for(
            lambda: bool(screen.query("#library-skills-items-grip")), "Skills"
        )
        shell = screen.query_one("#library-skills-reader-shell")
        await wait_for(
            lambda shell=shell: shell.effective_layout.reader_width > 0, "Skills layout"
        )
        if not shell.effective_layout.items_open:
            await activate("#library-skills-items-grip", None)
        await activate("#library-skills-trust-action", "Set up skill trust")
        await wait_for(
            lambda: isinstance(app.screen, SkillTrustBootstrapModal), "bootstrap"
        )
        app.screen.query_one("#skill-trust-bootstrap-input", Input).value = ""
        app.screen.query_one("#skill-trust-bootstrap-confirm-input", Input).value = ""
        await expect_focus("#skill-trust-bootstrap-input", "New trust passphrase")
        await pilot.press("enter")
        await wait_for(
            lambda: (
                str(
                    app.screen.query_one(
                        "#skill-trust-bootstrap-error", Static
                    ).renderable
                )
                == "Passphrase cannot be blank."
            ),
            "blank bootstrap error",
        )
        await capture("bootstrap-170.svg")
        await tmux("resize-window", "-t", session, "-x", "80", "-y", "24")
        await wait_for(lambda: tuple(app.size) == (80, 24), "compact bootstrap")
        app.theme = "textual-light"
        app.screen.query_one(
            "#skill-trust-bootstrap-input", Input
        ).value = "synthetic-qa-passphrase"
        app.screen.query_one(
            "#skill-trust-bootstrap-confirm-input", Input
        ).value = "mismatch"
        await pilot.press("enter")
        await wait_for(
            lambda: (
                str(
                    app.screen.query_one(
                        "#skill-trust-bootstrap-error", Static
                    ).renderable
                )
                == "Passphrases do not match."
            ),
            "mismatch bootstrap error",
        )
        await capture("bootstrap-80.svg")
        app.screen.query_one(
            "#skill-trust-bootstrap-confirm-input", Input
        ).value = "synthetic-qa-passphrase"
        await activate("#skill-trust-bootstrap-submit", "Submit")
        await wait_for(
            lambda: (
                app.screen is screen
                and not trust.status_for_skill("native-baseline").trust_blocked
            ),
            "bootstrapped",
        )
        result["bootstrap_blank_and_mismatch"] = True
        for width, height, theme in (
            (170, 48, "textual-dark"),
            (80, 24, "textual-light"),
        ):
            await tmux(
                "resize-window", "-t", session, "-x", str(width), "-y", str(height)
            )
            await wait_for(
                lambda width=width, height=height: tuple(app.size) == (width, height),
                "size",
            )
            app.theme = theme
            source = root / f"package-{width}"
            name = f"zeta-{width}"
            for candidate in (f"alpha-{width}", name):
                folder = source / candidate
                folder.mkdir(parents=True)
                (folder / "SKILL.md").write_text(
                    f"---\nname: {candidate}\ndescription: Imported café\n---\nBody {candidate} [literal].\n"
                )
            await screen._select_library_rail_row("browse-skills")
            await wait_for(
                lambda: bool(screen.query("#library-skills-items-grip")), "list"
            )
            shell = screen.query_one("#library-skills-reader-shell")
            await wait_for(
                lambda shell=shell: shell.effective_layout.reader_width > 0, "layout"
            )
            if not shell.effective_layout.items_open:
                await activate("#library-skills-items-grip", None)
            if screen._skills_state.view == "list":
                await activate("#library-skills-sort", "sort")
                await activate("#library-skills-sort-status", "Status")
            assert screen._skills_state.sort == "status"
            await activate("#library-skills-import", "Import skill")
            await wait_for(
                lambda: bool(screen.query("#library-skills-import-path")), "path"
            )
            screen.query_one("#library-skills-import-path", Input).value = str(
                root / "missing"
            )
            await pilot.pause()
            await activate("#library-skills-import-run", "Import")
            await wait_for(
                lambda: (
                    not screen._library_skills_import_in_flight
                    and bool(screen._library_skills_import_status)
                ),
                "invalid path",
            )
            await expect_focus("#library-skills-import-path", "missing")
            await capture(f"import-error-{width}.svg")
            assert not screen._library_skills_import_review_name
            screen.query_one("#library-skills-import-path", Input).value = str(source)
            await pilot.pause()
            for cancel in (True, False):
                await activate("#library-skills-import-run", "Import")
                await wait_for(
                    lambda: isinstance(app.screen, SkillImportChoiceModal), "choice"
                )
                if cancel:
                    await capture(f"choice-{width}.svg")
                    await activate("#skill-import-choice-cancel", "Cancel")
                    await wait_for(
                        lambda: (
                            app.screen is screen
                            and not screen._library_skills_import_in_flight
                        ),
                        "choice Cancel",
                    )
                    await wait_for(
                        lambda: (
                            screen.focused
                            is screen.query_one("#library-skills-import-path")
                        ),
                        "Cancel focus",
                    )
                    assert not (local.skills_dir / name).exists()
                else:
                    await pilot.press("down")
                    assert (
                        app.screen.query_one("#skill-import-choice-list").highlighted
                        == 1
                    )
                    await activate("#skill-import-choice-import", "Import skill")
            await wait_for(
                lambda name=name: (
                    app.screen is screen
                    and screen._library_skills_import_review_name == name
                    and not screen._library_skills_import_in_flight
                ),
                "imported",
            )
            await expect_focus("#library-skills-import-review", f'Review "{name}"')
            assert trust.status_for_skill(name).trust_blocked
            assert not (local.skills_dir / f"alpha-{width}").exists()
            await activate("#library-skills-import-review", f'Review "{name}"')
            await wait_for(
                lambda: bool(screen.query("#library-skill-trust-review")), "trust panel"
            )
            await expect_focus("#library-skill-trust-review", "Review changes")
            assert screen._skills_state.reader_mode == "trust"
            await activate("#library-skill-trust-review", "Review changes")
            await wait_for(
                lambda: screen._skills_state.active_review is not None,
                "captured review",
            )
            assert f"Body {name} [literal]." in str(
                screen.query_one(
                    "#library-skill-trust-review-content", Static
                ).renderable
            )
            await expect_focus("#library-skill-trust-review", "Review changes")
            await capture(f"review-{width}.svg")
            skill_path = local.skills_dir / name / "SKILL.md"
            skill_path.write_text(skill_path.read_text() + "Newer on disk.\n")
            for stale in (True, False):
                await activate("#library-skill-trust-approve", "Approve")
                await wait_for(
                    lambda: isinstance(app.screen, SkillTrustPassphraseModal),
                    "approve passphrase",
                )
                await pilot.press("enter")
                assert (
                    str(
                        app.screen.query_one(
                            "#skill-trust-passphrase-error", Static
                        ).renderable
                    )
                    == "Passphrase cannot be blank."
                )
                if stale:
                    await capture(f"passphrase-{width}.svg")
                app.screen.query_one(
                    "#skill-trust-passphrase-input", Input
                ).value = "synthetic-qa-passphrase"
                await pilot.press("enter")
                await wait_for(
                    lambda: (
                        app.screen is screen
                        and screen._skills_state.active_review is None
                    ),
                    "approval outcome",
                )
                if stale:
                    assert trust.status_for_skill(name).trust_blocked
                    await activate("#library-skill-trust-review", "Review changes")
                    await wait_for(
                        lambda: screen._skills_state.active_review is not None,
                        "fresh review",
                    )
                    assert "Newer on disk." in str(
                        screen.query_one(
                            "#library-skill-trust-review-content", Static
                        ).renderable
                    )
                else:
                    await wait_for(
                        lambda: not screen._skills_state.editor_state.trust_blocked,
                        "approved UI",
                    )
                    assert not trust.status_for_skill(name).trust_blocked
            await expect_focus("#library-skill-mode-trust", "Trust")
            await capture(f"approved-{width}.svg")
            result["steps"].append(
                {
                    "size": [width, height],
                    "theme": theme,
                    "name": name,
                    "invalid_path_recovered": True,
                    "candidate_cancel_preserved_draft": True,
                    "only_selected_candidate_imported": True,
                    "import_remained_blocked": True,
                    "captured_literal_content": True,
                    "stale_approval_blocked": True,
                    "fresh_snapshot_approved": True,
                    "approved_sha256": hashlib.sha256(
                        skill_path.read_bytes()
                    ).hexdigest(),
                    "final_focus": screen.focused.id if screen.focused else None,
                }
            )
            record()
        result["passed"] = True
        result["normal_quit_requested"] = True
        record()
        app.post_message(events.Key("ctrl+q", "\x11"))
    except Exception:  # noqa: BLE001 - record failures before normal native shutdown
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

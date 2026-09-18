"""Read-only bundle inventory and draft-preserving Files navigation."""

from __future__ import annotations

import asyncio
import hashlib

import pytest
from textual.widgets import Input, Static, TextArea

from Tests.Skills.test_skills_library_flow import (
    _real_uninitialized_trust_service,
    _wire_empty_non_skill_services,
)
from Tests.UI.test_library_prompt_collection_journeys import _focus
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)
from Tests.UI.test_library_skill_editor_journeys import _activate
from Tests.UI.test_library_skills_canvas import _build_test_app
from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
from tldw_chatbook.Skills_Interop.skills_scope_service import SkillsScopeService


def _painted(host, widget=None) -> str:
    strips = list(host.screen._compositor.render_strips())
    if widget is None:
        return "\n".join(strip.text for strip in strips)
    region = widget.region.intersection(widget.parent.parent.content_region)
    return "\n".join(
        strips[y].crop(region.x, region.right).text
        for y in range(max(0, region.y), min(region.bottom, len(strips)))
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_files_inventory_keyboard_journey_preserves_bundle_and_draft(
    tmp_path, monkeypatch, size, theme
):
    """Catch clipped inventories, inaccessible mode returns and draft/file loss."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    app = _build_test_app()
    app.library_new_profile_admission = True
    _wire_empty_non_skill_services(app)
    trust = _real_uninitialized_trust_service(tmp_path)
    local = LocalSkillsService(store_dir=tmp_path, trust_service=trust)
    app.local_skills_service = local
    app.local_skill_trust_service = trust
    app.skills_scope_service = SkillsScopeService(
        local_service=local, server_service=None
    )
    content = (
        "---\nname: bundle\ndescription: Original description\n---\nOriginal body."
    )
    long_path = "references/" + "a" * 75 + "/" + "b" * 85 + ".md"
    supporting = {
        "references/00-guide.md": "café\n",
        "references/01-empty.txt": "",
        long_path: "Long path contents.\n",
        **{f"references/guide-{i:02}.md": "Read this guide.\n" for i in range(60)},
        "references/zz-last.md": "Final supporting file.\n",
    }
    await local.create_skill(
        name="bundle", content=content, supporting_files=supporting
    )
    await local.create_skill(
        name="empty",
        content="---\nname: empty\ndescription: No supporting files\n---\nBody.",
    )
    binary = local.skills_dir / "bundle/assets/logo.bin"
    binary.parent.mkdir()
    binary.write_bytes(b"\x00\xff\x10\x20\x30")
    await asyncio.to_thread(trust.bootstrap_trust, "synthetic-files-passphrase")
    trust_dir = trust.trust_store.store_dir
    original_trust = {
        str(path.relative_to(trust_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in trust_dir.rglob("*")
        if path.is_file()
    }
    assert trust.status_for_skill("bundle").trust_status == "trusted"
    bundle = local.skills_dir / "bundle"
    original = {
        str(path.relative_to(bundle)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in bundle.rglob("*")
        if path.is_file()
    }
    host = LibraryProductionCSSHarness(app)
    host.theme = theme
    app.notify = host.notify
    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await screen._select_library_rail_row("browse-skills")
        await _wait_for_selector(screen, pilot, "#library-skills-items-grip")
        shell = screen.query_one("#library-skills-reader-shell")
        await _wait_for_condition(
            pilot,
            lambda: shell.effective_layout.reader_width > 0,
            message="Skills layout did not settle",
        )
        if not shell.effective_layout.items_open:
            await _activate(screen, host, pilot, "#library-skills-items-grip", "--->")
        await _activate(screen, host, pilot, "#library-skill-row-empty", "empty")
        await _activate(screen, host, pilot, "#library-skill-mode-files", "Files")
        await _wait_for_selector(screen, pilot, "#library-skill-files-region")
        await _focus(screen, host, pilot, "#library-skill-mode-files", "Files")
        assert "No supporting files." in _painted(host)

        if not shell.effective_layout.items_open:
            await _activate(screen, host, pilot, "#library-skills-items-grip", "--->")
        await _activate(screen, host, pilot, "#library-skill-row-bundle", "bundle")
        await _activate(screen, host, pilot, "#library-skill-mode-files", "Files")
        await _wait_for_selector(screen, pilot, "#library-skill-files-region")
        await _focus(screen, host, pilot, "#library-skill-mode-files", "Files")
        region = screen.query_one("#library-skill-files-region")
        assert not region.query(Input) and not region.query(TextArea)
        inventory = str(
            screen.query_one("#library-skill-supporting", Static).renderable
        )
        assert "assets/logo.bin — 5 bytes (binary)" in inventory
        assert "references/00-guide.md (6 bytes)" in inventory
        assert "references/01-empty.txt (0 bytes)" in inventory
        assert long_path in inventory
        assert "SKILL.md" not in inventory
        assert "café" not in inventory
        work = screen.query_one("#library-skill-work-pane")
        assert work.max_scroll_y > 0

        # Reach real content below the fold using only the reader's keyboard path.
        await pilot.press("end")
        await _wait_for_condition(
            pilot,
            lambda: "references/zz-last.md" in _painted(host),
            message="Keyboard End did not reveal the last supporting file",
        )
        await pilot.press("home")
        await _focus(screen, host, pilot, "#library-skill-mode-files", "Files")
        assert "assets/logo.bin" in _painted(host)
        # Every long path character must remain available, including its suffix.
        supporting_widget = screen.query_one("#library-skill-supporting")
        frames = [_painted(host, supporting_widget)]
        for _ in range(5):
            await pilot.press("down")
            frames.append(_painted(host, supporting_widget))
        assert long_path in "".join("".join(frame.split()) for frame in frames)

        await pilot.press("home")
        await _activate(screen, host, pilot, "#library-skill-mode-edit", "Edit")
        body = await _wait_for_selector(screen, pilot, "#library-skill-body")
        body.text = "Keep this unsaved draft."
        screen.query_one(
            "#library-skill-description", Input
        ).value = "Unsaved description"
        await pilot.pause()
        assert screen._skills_state.dirty
        await _focus(screen, host, pilot, "#library-skill-mode-edit", "Edit")
        await pilot.press("tab", "tab")
        await _focus(screen, host, pilot, "#library-skill-mode-files", "Files")
        await pilot.press("enter")
        await _wait_for_selector(screen, pilot, "#library-skill-files-region")
        await _focus(screen, host, pilot, "#library-skill-mode-files", "Files")
        assert screen._skills_state.dirty
        assert "references/00-guide.md (6 bytes)" in str(
            screen.query_one("#library-skill-supporting", Static).renderable
        )
        await pilot.press("shift+tab", "shift+tab")
        await _focus(screen, host, pilot, "#library-skill-mode-edit", "Edit")
        await pilot.press("enter")
        await _wait_for_selector(screen, pilot, "#library-skill-body")
        assert (
            screen.query_one("#library-skill-body", TextArea).text
            == "Keep this unsaved draft."
        )
        assert (
            screen.query_one("#library-skill-description", Input).value
            == "Unsaved description"
        )
        assert screen._skills_state.dirty

    after = {
        str(path.relative_to(bundle)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in bundle.rglob("*")
        if path.is_file()
    }
    assert after == original
    assert trust.status_for_skill("bundle").trust_status == "trusted"
    assert {
        str(path.relative_to(trust_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in trust_dir.rglob("*")
        if path.is_file()
    } == original_trust

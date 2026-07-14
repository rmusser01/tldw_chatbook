"""Real-service tests for the Library skills Import row (Task 5 of the
Skills sub-project).

Mirrors ``Tests/Skills/test_skills_library_flow.py``'s real-service
posture: a real ``LocalSkillsService``/``SkillsScopeService`` wired onto a
real ``LibraryScreen`` via ``App.run_test()`` -- no hand-rolled fakes for
the service layer.

Per the sub-project's compat directive, this suite imports REAL SKILL.md
files copied from the ``obra/superpowers`` skillset (see
``Tests/fixtures/superpowers_skills/README.md`` for provenance) through
the actual Library Import row -- not synthetic content -- so any
incompatibility between what that real-world skillset actually writes and
what ``local_skills_service``/``skills_schemas`` accepts surfaces here
honestly rather than being assumed away. A handful of additional,
CLEARLY synthetic edge cases (name too long, oversized content, a nested
reference subfolder) are constructed inline with ``tmp_path`` rather than
committed as fixture files, to keep the fixtures directory small.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.widgets import Button, Input, Static

from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
from tldw_chatbook.Skills_Interop.skills_scope_service import SkillsScopeService
from tldw_chatbook.tldw_api.skills_schemas import SKILL_NAME_PATTERN

from Tests.Skills.test_skills_library_flow import (
    _real_trust_service,
    _wire_empty_non_skill_services,
)
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _wait_for_library_shell,
)
from Tests.UI.test_screen_navigation import _build_test_app


FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "superpowers_skills"


def _real_skills_scope_service_with_trust(tmp_path):
    """Build a real ``LocalSkillsService``/``SkillsScopeService`` pair backed
    by a real (unlocked, NOT bootstrapped) trust service.

    Unlike ``test_skills_library_flow._real_skills_scope_service`` (which
    defaults to ``allow_untrusted_without_trust_service=True`` -- a compat
    mode that reports EVERY skill as already trusted, making trust
    irrelevant), this suite is specifically about asserting imported
    skills land TRUST-PENDING, so it always wires a real trust service.
    Never bootstrapped: a freshly imported skill in a trust store that has
    never been bootstrapped at all reports ``trust_uninitialized``/
    ``trust_blocked=True`` -- itself a genuine "needs review before use"
    state, not a synthetic stand-in for one.
    """
    trust_service = _real_trust_service(tmp_path)
    local_service = LocalSkillsService(store_dir=tmp_path, trust_service=trust_service)
    service = SkillsScopeService(local_service=local_service, server_service=None)
    return local_service, service


# The real superpowers skills copied into the fixtures dir (see its
# README.md for provenance) -- five distinct skills, one of which
# (requesting-code-review) has a real supporting reference file.
REAL_FIXTURE_SKILLS = (
    "executing-plans",
    "requesting-code-review",
    "using-superpowers",
    "verification-before-completion",
    "writing-plans",
)


async def _open_skills_import_row(screen, pilot) -> None:
    """Open the Skills rail row, then the inline Import row below its toolbar."""
    screen.query_one("#library-row-browse-skills", Button).press()
    await pilot.pause()
    await pilot.pause()
    screen.query_one("#library-skills-import", Button).press()
    await pilot.pause()


async def _run_skills_import_via_ui(screen, pilot, path: Path, *, attempts: int = 150) -> str:
    """Type ``path`` into the Import row and press Import, returning the
    outcome line once it changes from whatever it showed before this call.

    Waiting for a CHANGE (not just "non-empty") matters here: the Import
    row's outcome ``Static`` is never cleared between successive imports
    (only "Cancel" clears it), so blindly waiting for non-empty text would
    read a STALE outcome left over from a previous import in the same
    Import-row session.
    """
    previous = str(screen.query_one("#library-skills-import-status", Static).renderable)
    screen.query_one("#library-skills-import-path", Input).value = str(path)
    await pilot.pause()
    screen.query_one("#library-skills-import-run", Button).press()
    await pilot.pause()
    status_text = previous
    for _ in range(attempts):
        status_text = str(screen.query_one("#library-skills-import-status", Static).renderable)
        if status_text != previous:
            return status_text
        await pilot.pause(0.02)
    return status_text


@pytest.mark.asyncio
async def test_import_real_superpowers_skills_lands_trust_pending(tmp_path):
    """Import every real superpowers fixture skill through the actual
    Library Import row (rail row -> Import… -> path -> Import), then
    assert each one lands TRUST-PENDING (blocked) per the spec, with the
    exact outcome-line copy the brief pins, and that its persisted
    name/description validate against ``skills_schemas``'s own
    constraints (the real names are lowercase-hyphenated and well under
    the 64-char limit; the real descriptions are well under 1000 chars).
    """
    local_service, service = _real_skills_scope_service_with_trust(tmp_path)
    app = _build_test_app()
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skills_import_row(screen, pilot)

        for name in REAL_FIXTURE_SKILLS:
            status = await _run_skills_import_via_ui(screen, pilot, FIXTURES_DIR / name)
            assert status == "1 imported · re-review it in the trust panel", (
                f"unexpected outcome importing {name!r}: {status!r}"
            )

    context = await service.get_context(mode="local")
    blocked_names = {item["name"] for item in context["blocked_skills"]}
    assert blocked_names == set(REAL_FIXTURE_SKILLS)
    # TRUST-PENDING means blocked, not merely absent from the trusted list:
    # every imported skill must be blocked, none available yet.
    assert context["available_skills"] == []

    for name in REAL_FIXTURE_SKILLS:
        record = await local_service.get_skill(name)
        assert record["trust_blocked"] is True
        assert SKILL_NAME_PATTERN.match(record["name"]), record["name"]
        assert record["description"], f"{name} has no description"
        assert len(record["description"]) <= 1000


@pytest.mark.asyncio
async def test_import_skill_via_skill_md_file_path_derives_name_from_parent_directory(tmp_path):
    """Pointing the Import row at the ``SKILL.md`` FILE itself (not its
    parent directory) must resolve to the SAME correct skill name as
    pointing it at the directory -- the incompatibility this guards
    against: every real skill package uses the literal filename
    ``SKILL.md`` for every skill, so naively deriving the name from that
    file's own basename (as a generic ``import_skill_file(filename=...)``
    call would) produces the same wrong name ("skill") for every import
    regardless of which skill it actually is. ``_run_library_skills_import``
    must use the PARENT DIRECTORY's name instead for this exact shape.
    """
    local_service, service = _real_skills_scope_service_with_trust(tmp_path)
    app = _build_test_app()
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skills_import_row(screen, pilot)

        skill_md_path = FIXTURES_DIR / "verification-before-completion" / "SKILL.md"
        status = await _run_skills_import_via_ui(screen, pilot, skill_md_path)
        assert status == "1 imported · re-review it in the trust panel"

    record = await local_service.get_skill("verification-before-completion")
    assert record["name"] == "verification-before-completion"
    assert record["trust_blocked"] is True
    with pytest.raises(Exception):
        await local_service.get_skill("skill")


@pytest.mark.asyncio
async def test_import_skill_with_supporting_reference_file_threads_it_through(tmp_path):
    """``requesting-code-review`` has one real flat sibling file
    (``code-reviewer.md``) alongside its ``SKILL.md`` -- importing it by
    directory path must carry that sibling through as a supporting file
    with its exact real content, not silently drop it.
    """
    local_service, service = _real_skills_scope_service_with_trust(tmp_path)
    app = _build_test_app()
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service
    host = LibraryHarness(app)

    skill_dir = FIXTURES_DIR / "requesting-code-review"
    real_supporting_content = (skill_dir / "code-reviewer.md").read_text(encoding="utf-8")

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skills_import_row(screen, pilot)

        status = await _run_skills_import_via_ui(screen, pilot, skill_dir)
        assert status == "1 imported · re-review it in the trust panel"

    record = await local_service.get_skill("requesting-code-review")
    assert record["supporting_files"] == {"code-reviewer.md": real_supporting_content}


@pytest.mark.asyncio
async def test_import_skill_with_extra_frontmatter_fields_applies_recognized_and_drops_unknown(
    tmp_path,
):
    """``executing-plans-with-metadata`` is a synthetic fixture (real
    executing-plans description/body, augmented frontmatter -- see the
    fixtures README) that exercises frontmatter fields the real
    superpowers skillset never actually uses on its own
    (argument_hint/allowed-tools/license/compatibility/model/context/
    metadata). Recognized fields must be applied; the two UNRECOGNIZED
    fields it also carries (``priority``, ``tags``) must be silently
    dropped -- ``local_skills_service``'s own frontmatter parser filters
    to a fixed known-fields allowlist with no rejection/warning path, so
    this is the actual (not hypothetical) incompatibility surface between
    this schema and an arbitrary external skill spec.
    """
    local_service, service = _real_skills_scope_service_with_trust(tmp_path)
    app = _build_test_app()
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skills_import_row(screen, pilot)

        status = await _run_skills_import_via_ui(
            screen, pilot, FIXTURES_DIR / "executing-plans-with-metadata",
        )
        assert status == "1 imported · re-review it in the trust panel"

    record = await local_service.get_skill("executing-plans-with-metadata")
    assert record["argument_hint"] == "plan file path"
    assert record["allowed_tools"] == ["Read", "Write", "Bash"]
    assert record["model"] == "claude-sonnet-5"
    assert record["context"] == "fork"
    assert record["license"] == "MIT"
    assert record["compatibility"] == "Claude Code, Codex CLI"
    assert record["metadata"] == {
        "origin": "superpowers",
        "upstream_skill": "executing-plans",
    }
    assert record["validation_status"] == "valid"
    # The unrecognized fields never reach the persisted record at all --
    # not present, not surfaced as a validation error either. (``version``
    # is deliberately NOT checked here: it collides with the skill's own
    # legitimate revision-counter field, always present regardless of
    # frontmatter content.)
    assert "priority" not in record
    assert "tags" not in record
    assert not any("priority" in error or "tags" in error for error in record["validation_errors"])


@pytest.mark.asyncio
async def test_reimporting_the_same_skill_name_is_skipped_not_duplicated(tmp_path):
    """Importing the same real skill twice must skip the second attempt
    (never silently overwrite, matching the prompts import's own
    duplicate-name posture) and report which name collided.
    """
    local_service, service = _real_skills_scope_service_with_trust(tmp_path)
    app = _build_test_app()
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service
    host = LibraryHarness(app)

    skill_dir = FIXTURES_DIR / "executing-plans"

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skills_import_row(screen, pilot)

        first_status = await _run_skills_import_via_ui(screen, pilot, skill_dir)
        assert first_status == "1 imported · re-review it in the trust panel"

        second_status = await _run_skills_import_via_ui(screen, pilot, skill_dir)
        assert second_status == 'Skipped — a skill named "executing-plans" already exists.'

    record = await local_service.get_skill("executing-plans")
    assert record["version"] == 1


@pytest.mark.asyncio
async def test_import_row_reports_missing_skill_md_and_unknown_path_gracefully(tmp_path):
    """A folder with no ``SKILL.md`` and a path that does not exist at all
    both surface a specific, honest outcome line -- never a crash, never a
    silent no-op that leaves the user guessing.
    """
    local_service, service = _real_skills_scope_service_with_trust(tmp_path)
    app = _build_test_app()
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service
    host = LibraryHarness(app)

    empty_folder = tmp_path / "not-a-skill"
    empty_folder.mkdir()
    missing_path = tmp_path / "does-not-exist"

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skills_import_row(screen, pilot)

        status = await _run_skills_import_via_ui(screen, pilot, empty_folder)
        assert status == "No SKILL.md found in that folder."

        status = await _run_skills_import_via_ui(screen, pilot, missing_path)
        assert status == "Could not find that file or folder."

        loose_file = tmp_path / "notes.txt"
        loose_file.write_text("not a skill", encoding="utf-8")
        status = await _run_skills_import_via_ui(screen, pilot, loose_file)
        assert status == "Unsupported file type."

    context = await service.get_context(mode="local")
    assert context["available_skills"] == []
    assert context["blocked_skills"] == []


@pytest.mark.asyncio
async def test_import_row_rejects_name_too_long_without_partial_state(tmp_path):
    """A skill whose DIRECTORY name exceeds the 64-character
    ``skills_schemas`` limit must fail cleanly (a real incompatibility a
    misnamed import folder could trigger) -- reported as a specific
    failure, and nothing partially written (``import_skill`` validates
    before any disk write, so a failed import must leave the skill store
    completely untouched).
    """
    local_service, service = _real_skills_scope_service_with_trust(tmp_path)
    app = _build_test_app()
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service
    host = LibraryHarness(app)

    long_name = "a" + "-oversized-skill-name" * 4  # well over 64 characters
    assert len(long_name) > 64
    skill_dir = tmp_path / "oversized" / long_name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {long_name}\ndescription: Too long to import.\n---\n\nBody.\n",
        encoding="utf-8",
    )

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skills_import_row(screen, pilot)

        status = await _run_skills_import_via_ui(screen, pilot, skill_dir)
        assert status == "Could not import that skill."

    context = await service.get_context(mode="local")
    assert context["available_skills"] == []
    assert context["blocked_skills"] == []


@pytest.mark.asyncio
async def test_import_row_rejects_oversized_content_without_partial_state(tmp_path):
    """A ``SKILL.md`` whose content exceeds ``skills_schemas``'s 500,000
    character limit must fail cleanly rather than crash or silently
    truncate -- constructed inline (not a committed fixture) to keep the
    fixtures directory small.
    """
    local_service, service = _real_skills_scope_service_with_trust(tmp_path)
    app = _build_test_app()
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service
    host = LibraryHarness(app)

    skill_dir = tmp_path / "oversized-content"
    skill_dir.mkdir()
    oversized_body = "x" * 600_000
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: oversized-content\ndescription: Body is too large.\n---\n\n{oversized_body}\n",
        encoding="utf-8",
    )

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skills_import_row(screen, pilot)

        status = await _run_skills_import_via_ui(screen, pilot, skill_dir)
        assert status == "Could not import that skill."

    context = await service.get_context(mode="local")
    assert context["available_skills"] == []
    assert context["blocked_skills"] == []


@pytest.mark.asyncio
async def test_import_row_skips_nested_reference_subfolder_without_failing_import(tmp_path):
    """A skill directory with a NESTED reference subfolder (the real
    ``using-superpowers`` skill's own ``references/`` layout -- not
    copied into the fixtures dir to keep it small, reproduced here
    structurally) must still import successfully, but the nested file is
    NOT carried through as a supporting file: ``local_skills_service``'s
    supporting-file name pattern has no path-separator support, so
    recursing into subdirectories would either be silently flattened
    (name collisions) or rejected outright. Skipping nested paths keeps
    this import path's own limitation explicit and non-crashing rather
    than surprising.
    """
    local_service, service = _real_skills_scope_service_with_trust(tmp_path)
    app = _build_test_app()
    _wire_empty_non_skill_services(app)
    app.skills_scope_service = service
    host = LibraryHarness(app)

    skill_dir = tmp_path / "nested-refs-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: nested-refs-skill\ndescription: Has a nested references folder.\n---\n\nBody.\n",
        encoding="utf-8",
    )
    references_dir = skill_dir / "references"
    references_dir.mkdir()
    (references_dir / "note.md").write_text("A nested reference file.", encoding="utf-8")

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_skills_import_row(screen, pilot)

        status = await _run_skills_import_via_ui(screen, pilot, skill_dir)
        assert status == "1 imported · re-review it in the trust panel"

    record = await local_service.get_skill("nested-refs-skill")
    assert record["trust_blocked"] is True
    supporting_files = record.get("supporting_files") or {}
    assert "references/note.md" not in supporting_files
    assert "note.md" not in supporting_files

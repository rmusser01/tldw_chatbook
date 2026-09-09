"""Obsidian-vault behaviour of the one-time Database Notes import (task-32129)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tldw_chatbook.Library.library_note_import_state import _effect_summary
from tldw_chatbook.Notes.note_import_discovery import discover_import_sources
from tldw_chatbook.Notes.note_import_parsers import _wikilinks, parse_import_sources
from tldw_chatbook.Notes.note_import_plan_models import (
    ImportAction,
    ImportBounds,
    ImportClassification,
    ParsedNotePayload,
    rewrite_wikilinks,
)
from tldw_chatbook.Notes.note_import_planner import classify_import_batch


def _bounds() -> ImportBounds:
    return ImportBounds(
        max_files=200,
        max_file_bytes=100_000,
        max_total_bytes=1_000_000,
        max_depth=6,
        max_reason_length=200,
        max_entries=1_000,
    )


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_vault(root: Path) -> Path:
    """Build a small Obsidian vault fixture with each critique-8 trait.

    Traits: an `.obsidian/` config tree, an Obsidian `.trash/`, a `Templates/`
    folder whose note title is a `{{date}}` placeholder, YAML frontmatter with a
    title, tags in both YAML shapes, aliases, and wikilinks (plain, aliased,
    folder-qualified, embedded, and unresolvable).
    """
    _write(root, ".obsidian/app.json", json.dumps({"livePreview": True}))
    _write(root, ".obsidian/plugins/dataview/data.json", json.dumps({"x": 1}))
    _write(root, ".trash/Old idea.md", "# Old idea\n\nDeleted inside Obsidian.\n")
    _write(
        root,
        "Templates/Daily.md",
        "---\ntags: [daily]\ncreated: {{date}}\n---\n# {{date:YYYY-MM-DD}}\n\n- \n",
    )
    _write(
        root,
        "README.md",
        "# My vault\n\nStart at [[Projects/Library review]].\n",
    )
    _write(
        root,
        "Projects/Library review.md",
        "---\n"
        "title: Library review\n"
        "aliases: [notes-review, lib-review]\n"
        "tags: [project, ux]\n"
        "status: in-progress\n"
        "---\n"
        "\n"
        "# Ignored heading\n"
        "\n"
        "See [[Daily/2026-09-07|yesterday]], [[README]] and [[Missing]].\n"
        "![[attachments/diagram.png]]\n",
    )
    _write(
        root,
        "Daily/2026-09-07.md",
        "---\ntags:\n  - daily\n  - meeting\n---\n# 2026-09-07\n\nBody.\n",
    )
    _write(root, "Journal/Templated.md", "# {{date:YYYY-MM-DD}}\n\nBody.\n")
    _write(
        root,
        "Journal/Properties only.md",
        '---\ntitle: Properties only\ntags: [meta]\nsource: "[[README]]"\n---\n',
    )
    _write(
        root,
        "Journal/Snippets.md",
        "# Snippets\n\n"
        "Write `[[README]]` to link a note.\n\n"
        "```python\nx = \"[[README]]\"\n```\n\n"
        "A real link: [[README]].\n",
    )
    return root


@pytest.fixture
def vault(tmp_path: Path) -> Path:
    return _build_vault(tmp_path / "vault")


def _payloads(
    root: Path,
    *,
    obsidian_mode: bool = True,
) -> dict[str, ParsedNotePayload]:
    bounds = _bounds()
    discovery = discover_import_sources(
        [root],
        bounds,
        obsidian_mode=obsidian_mode,
    )
    batch = parse_import_sources(discovery, bounds, obsidian_mode=obsidian_mode)
    return {
        source.candidate.source.display_path: source.payloads[0]
        for source in batch.parsed
    }


def test_vault_root_is_detected_and_its_private_folders_are_skipped(
    vault: Path,
) -> None:
    """`.obsidian/`, `.trash/` and `Templates/` are skipped with plain reasons."""
    discovery = discover_import_sources([vault], _bounds(), obsidian_mode=True)

    assert discovery.vault_detected is True
    assert {skip.display_path: skip.reason_code for skip in discovery.skips} == {
        "vault/.obsidian": "obsidian_config",
        "vault/.trash": "obsidian_trash",
        "vault/Templates": "obsidian_template",
    }
    for skip in discovery.skips:
        assert "Obsidian" in skip.user_message
    admitted = {candidate.source.display_path for candidate in discovery.candidates}
    assert not any(
        path.startswith(("vault/.obsidian", "vault/.trash", "vault/Templates"))
        for path in admitted
    )
    assert discovery.failures == ()


def test_obsidian_mode_off_keeps_the_previous_discovery_behaviour(
    vault: Path,
) -> None:
    """Turning the toggle off restores the pre-task walk over every folder."""
    discovery = discover_import_sources([vault], _bounds(), obsidian_mode=False)

    admitted = {candidate.source.display_path for candidate in discovery.candidates}
    assert discovery.vault_detected is True
    assert discovery.skips == ()
    assert "vault/.obsidian/app.json" in admitted
    assert "vault/Templates/Daily.md" in admitted


def test_a_plain_folder_is_never_treated_as_a_vault(tmp_path: Path) -> None:
    """Without `.obsidian/` a folder named Templates still imports."""
    root = tmp_path / "Notes"
    _write(root, "Templates/Daily.md", "# Daily\n\nBody.\n")

    discovery = discover_import_sources([root], _bounds(), obsidian_mode=True)

    assert discovery.vault_detected is False
    assert discovery.skips == ()
    assert "Notes/Templates/Daily.md" in {
        candidate.source.display_path for candidate in discovery.candidates
    }


def test_skipped_entries_are_classified_as_skipped_not_failed(vault: Path) -> None:
    """A skipped vault folder reaches the review as Skipped with its reason."""
    bounds = _bounds()
    discovery = discover_import_sources([vault], bounds, obsidian_mode=True)

    batch = parse_import_sources(discovery, bounds, obsidian_mode=True)

    skipped = {
        issue.display_path: issue
        for issue in batch.issues
        if issue.classification is ImportClassification.SKIPPED
    }
    assert set(skipped) == {"vault/.obsidian", "vault/.trash", "vault/Templates"}
    assert not [
        issue
        for issue in batch.issues
        if issue.classification is ImportClassification.FAILED
    ]
    assert "Obsidian configuration" in skipped["vault/.obsidian"].user_message


def test_the_review_shows_each_skip_with_its_own_reason(vault: Path) -> None:
    """A skipped vault folder keeps its reason all the way into the preview."""
    bounds = _bounds()
    discovery = discover_import_sources([vault], bounds, obsidian_mode=True)
    batch = parse_import_sources(discovery, bounds, obsidian_mode=True)

    plan = classify_import_batch(batch, bounds)

    skipped = {
        item.source.display_path: item
        for item in plan.items
        if item.classification is ImportClassification.SKIPPED
    }
    assert set(skipped) == {"vault/.obsidian", "vault/.trash", "vault/Templates"}
    assert skipped["vault/Templates"].reason.startswith("Obsidian template")
    assert all(
        item.selected_action is ImportAction.SKIP
        and item.allowed_actions == (ImportAction.SKIP,)
        for item in skipped.values()
    )


def test_the_review_shows_the_resulting_title_keywords_and_links(vault: Path) -> None:
    """The review row states what the import will produce, before it runs."""
    bounds = _bounds()
    discovery = discover_import_sources([vault], bounds, obsidian_mode=True)
    plan = classify_import_batch(
        parse_import_sources(discovery, bounds, obsidian_mode=True), bounds
    )
    item = next(
        entry
        for entry in plan.items
        if entry.source.display_path == "vault/Projects/Library review.md"
    )

    summary = _effect_summary(item)

    assert "Library review" in summary
    assert "keywords project, ux, alias: notes-review, alias: lib-review" in summary
    assert "3 links" in summary


def test_frontmatter_supplies_the_title_keywords_and_aliases(vault: Path) -> None:
    """Frontmatter title wins over the first heading and tags become keywords.

    An alias is an alternate *name*, not a tag, so it keeps an ``alias:``
    prefix and stays tellable apart in Info (task-32178).
    """
    payload = _payloads(vault)["vault/Projects/Library review.md"]

    assert payload.title == "Library review"
    assert payload.keywords == (
        "project",
        "ux",
        "alias: notes-review",
        "alias: lib-review",
    )
    assert not payload.content.startswith("---")
    assert payload.content.startswith("# Ignored heading")


def test_block_sequence_tags_also_become_keywords(vault: Path) -> None:
    """`tags:` written as a YAML block sequence parses like the inline form."""
    payload = _payloads(vault)["vault/Daily/2026-09-07.md"]

    assert payload.keywords == ("daily", "meeting")
    assert payload.title == "2026-09-07"
    assert payload.content.startswith("# 2026-09-07")


def test_a_template_placeholder_never_becomes_a_note_title(vault: Path) -> None:
    """A `{{date:…}}` heading falls back to the file stem."""
    payload = _payloads(vault)["vault/Journal/Templated.md"]

    assert payload.title == "Templated"


def test_a_note_without_frontmatter_keeps_the_heading_rule(vault: Path) -> None:
    """Today's first-`# `-heading title rule is unchanged."""
    payload = _payloads(vault)["vault/README.md"]

    assert payload.title == "My vault"
    assert payload.wikilinks == ("Projects/Library review",)


def test_wikilink_targets_are_recorded_without_embeds(vault: Path) -> None:
    """Aliased and folder-qualified links are recorded; embeds are not."""
    payload = _payloads(vault)["vault/Projects/Library review.md"]

    assert payload.wikilinks == ("Daily/2026-09-07", "README", "Missing")


def test_obsidian_mode_off_leaves_frontmatter_and_links_untouched(
    vault: Path,
) -> None:
    """With the toggle off the parser behaves exactly as it did before."""
    payload = _payloads(vault, obsidian_mode=False)["vault/Projects/Library review.md"]

    assert payload.title == "Ignored heading"
    assert payload.keywords == ()
    assert payload.wikilinks == ()
    assert payload.content.startswith("---")


def test_a_frontmatter_only_note_still_imports_with_its_metadata(
    vault: Path,
) -> None:
    """An Obsidian Properties-only file must not become a failure."""
    payload = _payloads(vault)["vault/Journal/Properties only.md"]

    assert payload.title == "Properties only"
    assert payload.keywords == ("meta",)
    assert payload.content.startswith("---")


def test_retained_frontmatter_metadata_is_never_a_link(vault: Path) -> None:
    """A `[[target]]` inside retained Properties-only YAML is metadata, not a link."""
    payload = _payloads(vault)["vault/Journal/Properties only.md"]

    assert payload.wikilinks == ()
    assert rewrite_wikilinks(payload, {"readme": "note-id-1"}).content == (
        payload.content
    )


def test_a_link_label_cannot_widen_a_neighbouring_link() -> None:
    """A trailing backslash in an alias is escaped, not left to eat the `]`."""
    payload = ParsedNotePayload(
        title="Escapes",
        content="[[README|see\\]] then [docs](http://example.com).",
        wikilinks=("README",),
    )

    rewritten = rewrite_wikilinks(payload, {"readme": "note-id-1"})

    assert rewritten.content == (
        "[see\\\\](note://note-id-1) then [docs](http://example.com)."
    )


def test_links_inside_code_spans_are_not_recorded(vault: Path) -> None:
    """A `[[Target]]` in a fence or backticks is sample text, not a link."""
    payload = _payloads(vault)["vault/Journal/Snippets.md"]

    assert payload.wikilinks == ("README",)


def test_links_inside_code_spans_are_never_rewritten(vault: Path) -> None:
    """The rewrite leaves fenced and inline code exactly as written."""
    payload = _payloads(vault)["vault/Journal/Snippets.md"]

    rewritten = rewrite_wikilinks(payload, {"readme": "note-id-1"})

    assert "`[[README]]`" in rewritten.content
    assert 'x = "[[README]]"' in rewritten.content
    assert "A real link: [README](note://note-id-1)." in rewritten.content


def test_an_unclosed_fence_keeps_the_rest_of_the_note_as_code() -> None:
    """An opener with no closer is code to end of file, so nothing after it is a link.

    PR #2549 review, finding 8: the scanner recognised a fenced block only
    when it found the closing delimiter, so a `[[Target]]` after an
    unfinished ``` opener was recorded as a link and rewritten into literal
    sample text.
    """
    content = "```\nSee [[README]] for the setup.\n"

    assert _wikilinks(content) == ()

    payload = ParsedNotePayload(title="Snippet", content=content, wikilinks=())
    assert rewrite_wikilinks(payload, {"readme": "note-id-1"}).content == content


def test_a_closed_fence_still_ends_at_its_closer() -> None:
    """The unclosed-fence rule does not swallow links after a finished block."""
    content = "```\n[[README]]\n```\n\nA real link: [[README]]."

    assert _wikilinks(content) == ("README",)

    payload = ParsedNotePayload(
        title="Snippet", content=content, wikilinks=("README",)
    )
    rewritten = rewrite_wikilinks(payload, {"readme": "note-id-1"})

    assert "```\n[[README]]\n```" in rewritten.content
    assert "A real link: [README](note://note-id-1)." in rewritten.content


def test_a_template_placeholder_is_rejected_with_the_toggle_off(
    vault: Path,
) -> None:
    """The placeholder-title rule is unconditional, not an Obsidian-mode extra."""
    payload = _payloads(vault, obsidian_mode=False)["vault/Journal/Templated.md"]

    assert payload.title == "Templated"

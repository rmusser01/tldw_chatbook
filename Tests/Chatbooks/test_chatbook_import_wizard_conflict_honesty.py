# test_chatbook_import_wizard_conflict_honesty.py
# Description: Regression coverage for task-19861 (the conflict step claimed
#              "no conflicts detected" that nothing had checked)
"""
`ChatbookImportWizard`'s step 3 used to ship two permanently-frozen claims:

1. an unconditional ``Static`` reading "✅ No conflicts detected - all
   content is new", rendered for every user on every import, with no
   conflict check anywhere in the wizard or the importer; and
2. a "Potential Conflicts Detected:" panel (``#conflict-container``) that
   carried the ``hidden`` class in ``compose()`` with no code path anywhere
   that could reveal it.

So the step asserted a computed-sounding fact it never computed, on the
screen where the user picks the strategy that fact should inform. These
tests pin the honest behaviour: both elements are driven by
``ChatbookImporter.scan_conflicts``, a real read-only check of the
destination databases using the same lookup keys the import itself uses.

The born-red tests mount the step with a destination pre-loaded with every
item in the chatbook; at this task's base the unconditional Static rendered
"no conflicts" over exactly that destination.

``pytestmark = bootstrap_profile``: the scan constructs real Chatbook
databases through the config-participant admission chain, which fails
closed with ``RecoveryRequired`` under the per-test config redirect (see
Tests/conftest.py's note on the same trap for the media-DB v6->v7
migration).
"""

import ast
import inspect
import shutil
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from tldw_chatbook.Chatbooks.chatbook_creator import ChatbookCreator
from tldw_chatbook.Chatbooks.chatbook_importer import ChatbookImporter
from tldw_chatbook.Chatbooks.chatbook_models import ContentType
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.UI.Wizards import ChatbookImportWizard as wizard_module
from tldw_chatbook.UI.Wizards.BaseWizard import WizardStepConfig
from tldw_chatbook.UI.Wizards.ChatbookImportWizard import ConflictResolutionStep

pytestmark = pytest.mark.bootstrap_profile

#: The exact copy the unconditional Static shipped at base. Any test that
#: needs "the no-conflicts claim is absent" matches on this phrase, so
#: rewording the claim instead of earning it cannot satisfy the guard.
NO_CONFLICTS_CLAIM = "No conflicts detected"


class _StepHost(App):
    """Mount a single wizard step, the idiom used by Tests/Wizards."""

    def __init__(self, step):
        super().__init__()
        self._step = step

    def compose(self) -> ComposeResult:
        yield self._step


def _fake_wizard(**wizard_data) -> SimpleNamespace:
    return SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        wizard_data=dict(wizard_data),
        initial_execution_mode="local",
        can_go_back=True,
    )


def _make_db_paths(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    return {
        "ChaChaNotes": str(root / "ChaChaNotes.db"),
        "Media": str(root / "Client_Media_DB.db"),
        "Prompts": str(root / "Prompts_DB.db"),
    }


@pytest.fixture
def chatbook_zip(tmp_path, chachanotes_template_db) -> Path:
    """Export a chatbook holding one of each strategy-governed content type."""
    source_paths = _make_db_paths(tmp_path / "source")
    shutil.copyfile(chachanotes_template_db, source_paths["ChaChaNotes"])
    chacha = CharactersRAGDB(source_paths["ChaChaNotes"], "test_client")

    char_id = chacha.add_character_card(
        {
            "name": "Field Guide",
            "description": "A helpful guide",
            "personality": "Precise",
            "scenario": "",
            "greeting_message": "Hello!",
            "example_messages": "",
            "version": 1,
        }
    )
    conv_id = chacha.add_conversation(
        {
            "title": "Trip planning",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "root_id": "conv_root",
            "character_id": char_id,
        }
    )
    chacha.add_message(
        {"conversation_id": conv_id, "sender": "user", "content": "Where to?"}
    )
    note_id = chacha.add_note(title="Packing list", content="Boots, map, water")

    media_db = MediaDatabase(source_paths["Media"], "test_client")
    media_db.add_media_with_keywords(
        url="https://example.com/trail",
        title="Trail Video",
        media_type="video",
        content="A transcript of the trail video.",
        keywords=["trail"],
        prompt="Summarize",
        analysis_content="A trail video",
        transcription_model="whisper",
    )
    media_row = media_db.get_media_by_url("https://example.com/trail")
    assert media_row, "the media fixture must produce a media row to export"
    media_ids = [str(media_row["id"])]

    export_path = tmp_path / "bundle.zip"
    success, message, _deps = ChatbookCreator(db_paths=source_paths).create_chatbook(
        name="Trip Bundle",
        description="Round-trip fixture",
        content_selections={
            ContentType.CONVERSATION: [str(conv_id)],
            ContentType.NOTE: [str(note_id)],
            ContentType.CHARACTER: [str(char_id)],
            ContentType.MEDIA: media_ids,
        },
        output_path=export_path,
        author="Tests",
        include_media=True,
    )
    assert success is True, message
    return export_path


@pytest.fixture
def destination_paths(tmp_path, chachanotes_template_db) -> dict:
    paths = _make_db_paths(tmp_path / "destination")
    shutil.copyfile(chachanotes_template_db, paths["ChaChaNotes"])
    CharactersRAGDB(paths["ChaChaNotes"], "test_client")
    MediaDatabase(paths["Media"], "test_client")
    return paths


def _import_once(chatbook_zip: Path, destination_paths: dict) -> None:
    """Pre-load the destination with every item in the chatbook."""
    from tldw_chatbook.Chatbooks.chatbook_importer import ImportStatus
    from tldw_chatbook.Chatbooks.conflict_resolver import ConflictResolution

    ChatbookImporter(destination_paths).import_chatbook(
        chatbook_path=chatbook_zip,
        conflict_resolution=ConflictResolution.SKIP,
        prefix_imported=True,
        import_media=True,
        import_status=ImportStatus(),
    )


async def _mount_and_scan(
    chatbook_zip: Path,
    destination_paths: dict,
    monkeypatch,
    *,
    with_manifest: bool = True,
):
    """Mount the real conflict step, run its scan, and return its state.

    Returns ``(step, no_conflicts_visible, no_conflicts_text,
    conflict_container_visible, conflict_lines)``.
    """
    manifest, error = ChatbookImporter(destination_paths).preview_chatbook(chatbook_zip)
    assert error is None, error

    preview_data = {"manifest": manifest} if with_manifest else {}
    step = ConflictResolutionStep(
        wizard=_fake_wizard(
            **{
                "file-selection": {"file_path": str(chatbook_zip)},
                "preview-validation": preview_data,
            }
        ),
        config=WizardStepConfig(
            id="conflict-resolution",
            title="Conflict Resolution",
            step_number=3,
        ),
    )
    monkeypatch.setattr(
        wizard_module, "get_chatbook_database_paths", lambda: destination_paths
    )

    async with _StepHost(step).run_test(size=(120, 60)) as pilot:
        await pilot.pause()
        if getattr(step, "conflict_worker", None) is None:
            # `on_show` did not fire for this host; drive the same entry point.
            await step._scan_conflicts()
            await pilot.pause()
        no_conflicts = step.query_one("#no-conflicts", Static)
        container = step.query_one("#conflict-container")
        lines = [
            str(item.renderable) for item in step.query_one("#conflict-list").children
        ]
        # Visibility is the ``hidden`` class, not the ``display`` property:
        # compose()/workers toggle the class, and Textual keeps ``display``
        # True for a class-hidden widget.
        state = (
            step,
            not no_conflicts.has_class("hidden"),
            str(no_conflicts.renderable),
            not container.has_class("hidden"),
            lines,
        )
    return state


# ---------------------------------------------------------------------------
# The born-red cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preloaded_destination_is_never_told_all_content_is_new(
    chatbook_zip, destination_paths, monkeypatch
):
    """AC#1/AC#2, born red at base.

    Every item in the chatbook is already present in the destination; at
    this task's base the unconditional Static still said "✅ No conflicts
    detected - all content is new" over exactly this state.
    """
    _import_once(chatbook_zip, destination_paths)

    (
        _step,
        no_conflicts_visible,
        no_conflicts_text,
        container_visible,
        lines,
    ) = await _mount_and_scan(chatbook_zip, destination_paths, monkeypatch)

    assert not no_conflicts_visible, no_conflicts_text
    assert NO_CONFLICTS_CLAIM not in no_conflicts_text, no_conflicts_text
    assert container_visible, "the findings panel must be revealed"
    assert lines, "the destination holds all four items; each must be named"
    joined = "\n".join(lines)
    for title in ("Trip planning", "Packing list", "Field Guide"):
        assert title in joined, joined
    assert "Trail Video" in joined, joined
    # The lines state WHAT already exists, not just that something does.
    assert "[Imported] Trip planning" in joined, joined
    assert "https://example.com/trail" in joined, joined


@pytest.mark.asyncio
async def test_empty_destination_earns_the_no_conflicts_claim(
    chatbook_zip, destination_paths, monkeypatch
):
    """AC#5: the message shown for a genuinely empty destination is true of
    it -- and it is not shown before the check that earns it has run."""
    (
        _step,
        no_conflicts_visible,
        no_conflicts_text,
        container_visible,
        lines,
    ) = await _mount_and_scan(chatbook_zip, destination_paths, monkeypatch)

    # The scan ran and found nothing: the claim is now earned.
    assert no_conflicts_visible, no_conflicts_text
    assert NO_CONFLICTS_CLAIM in no_conflicts_text, no_conflicts_text
    assert not container_visible
    assert lines == []

    # And it says what it checked, so "no conflicts" cannot read as broader
    # than the check was.
    for word in ("conversation", "note", "character", "media"):
        assert word in no_conflicts_text.lower(), no_conflicts_text


@pytest.mark.asyncio
async def test_the_claim_does_not_exist_before_any_check_has_run(
    chatbook_zip, destination_paths, monkeypatch
):
    """AC#1, the pre-scan half: freshly composed, before the scan completes,
    the step must not already assert an outcome.

    At base the Static rendered "✅ No conflicts detected - all content is
    new" from ``compose()`` alone. This is also the mutation check for the
    family: restoring the unconditional Static makes this test red again.
    """
    manifest, error = ChatbookImporter(destination_paths).preview_chatbook(chatbook_zip)
    assert error is None, error

    step = ConflictResolutionStep(
        wizard=_fake_wizard(
            **{
                "file-selection": {"file_path": str(chatbook_zip)},
                "preview-validation": {"manifest": manifest},
            }
        ),
        config=WizardStepConfig(
            id="conflict-resolution", title="Conflict Resolution", step_number=3
        ),
    )

    async def _no_scan(*_args, **_kwargs):
        """Stand in for the scan so only compose()'s shipped state is seen."""

    step._scan_conflicts = _no_scan  # type: ignore[method-assign]
    async with _StepHost(step).run_test(size=(120, 60)) as pilot:
        await pilot.pause()
        no_conflicts = step.query_one("#no-conflicts", Static)
        container = step.query_one("#conflict-container")
        assert no_conflicts.has_class("hidden"), str(no_conflicts.renderable)
        assert container.has_class("hidden")


@pytest.mark.asyncio
async def test_no_manifest_means_no_claim_either_way(
    chatbook_zip, destination_paths, monkeypatch
):
    """When no scan can run (no validated manifest yet), the step says
    nothing about conflicts in either direction."""
    (
        _step,
        no_conflicts_visible,
        _text,
        container_visible,
        lines,
    ) = await _mount_and_scan(
        chatbook_zip, destination_paths, monkeypatch, with_manifest=False
    )

    assert not no_conflicts_visible
    assert not container_visible
    assert lines == []


# ---------------------------------------------------------------------------
# The scan itself
# ---------------------------------------------------------------------------


def test_scan_names_every_preloaded_item(chatbook_zip, destination_paths):
    """AC#2: the scan reports each colliding item, keyed the way the import
    path keys conflicts (names for conversations/notes/characters, URL for
    media, both prefix variants for names)."""
    _import_once(chatbook_zip, destination_paths)

    findings, error = ChatbookImporter(destination_paths).scan_conflicts(chatbook_zip)

    assert error is None, error
    assert findings is not None
    by_type = {}
    for finding in findings:
        by_type.setdefault(finding.content_type, []).append(finding)
    assert set(by_type) == {
        ContentType.CONVERSATION,
        ContentType.NOTE,
        ContentType.CHARACTER,
        ContentType.MEDIA,
    }, sorted(str(key) for key in by_type)
    # The prefixed variant is what the default options would collide with
    # (the fixture imported with prefix_imported=True).
    assert by_type[ContentType.CONVERSATION][0].existing_name == (
        "[Imported] Trip planning"
    )
    assert by_type[ContentType.NOTE][0].existing_name == "[Imported] Packing list"
    assert by_type[ContentType.CHARACTER][0].existing_name == "[Imported] Field Guide"
    assert by_type[ContentType.MEDIA][0].existing_name == "https://example.com/trail"


def test_scan_finds_the_unprefixed_variant_too(chatbook_zip, destination_paths):
    """A destination holding the PLAIN names collides with the import when
    the user turns the rename option off; the scan must see that as well."""
    chacha = CharactersRAGDB(destination_paths["ChaChaNotes"], "test_client")
    chacha.add_note(title="Packing list", content="already here, unprefixed")

    findings, error = ChatbookImporter(destination_paths).scan_conflicts(chatbook_zip)

    assert error is None, error
    assert any(
        finding.content_type == ContentType.NOTE
        and finding.title == "Packing list"
        and finding.existing_name == "Packing list"
        for finding in findings or []
    ), findings


def test_scan_of_an_empty_destination_returns_no_findings(
    chatbook_zip, destination_paths
):
    findings, error = ChatbookImporter(destination_paths).scan_conflicts(chatbook_zip)

    assert error is None, error
    assert findings == []


def test_scan_of_a_broken_archive_reports_an_error(tmp_path, destination_paths):
    not_a_zip = tmp_path / "broken.zip"
    not_a_zip.write_bytes(b"this is not a zip archive")

    findings, error = ChatbookImporter(destination_paths).scan_conflicts(not_a_zip)

    assert findings is None
    assert error is not None


def test_scan_does_not_write_to_the_destination(chatbook_zip, destination_paths):
    """The scan runs before the user has confirmed anything; it must not
    mutate the databases it is reporting on."""
    _import_once(chatbook_zip, destination_paths)
    db = CharactersRAGDB(destination_paths["ChaChaNotes"], "test_client")
    before = sorted((row["title"], row["content"]) for row in db.list_notes(limit=1000))
    db.close_connection()

    ChatbookImporter(destination_paths).scan_conflicts(chatbook_zip)

    db = CharactersRAGDB(destination_paths["ChaChaNotes"], "test_client")
    after = sorted((row["title"], row["content"]) for row in db.list_notes(limit=1000))
    db.close_connection()
    assert before == after


# ---------------------------------------------------------------------------
# The structural lock: neither element may go back to being unconditional
# ---------------------------------------------------------------------------


def test_no_conflicts_element_ships_hidden_and_the_panel_has_a_reveal_path():
    """AC#3: ``#no-conflicts`` must not be visible from ``compose()`` alone,
    and ``#conflict-container`` must have a code path that reveals it (it
    used to be permanently hidden with no code able to show it)."""
    source = Path(inspect.getfile(wizard_module)).read_text(encoding="utf-8")
    tree = ast.parse(source)

    step_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ConflictResolutionStep"
    )
    compose = next(
        node
        for node in step_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "compose"
    )
    compose_source = ast.get_source_segment(source, compose) or ""

    assert 'id="no-conflicts"' in compose_source
    assert '"no-conflicts hidden"' in compose_source, (
        "the no-conflicts message must ship hidden; an unconditional Static "
        "is exactly the defect task-19861 removed"
    )
    assert 'id="conflict-container"' in compose_source
    assert "conflict-preview hidden" in compose_source

    # Some code OUTSIDE compose must be able to reveal the panel: the class
    # references the container after compose (the reveal path the old code
    # never had).
    outside = []
    for node in step_class.body:
        if isinstance(node, ast.FunctionDef) and node.name != "compose":
            segment = ast.get_source_segment(source, node) or ""
            if "conflict-container" in segment:
                outside.append(node.name)
    assert outside, (
        "no method other than compose() references #conflict-container; the "
        "panel is permanently hidden again"
    )
    # The reveal path must remove the hidden class, not just mention the id.
    reveal_found = False
    for node in step_class.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for call in ast.walk(node):
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "remove_class"
                    and call.args
                    and isinstance(call.args[0], ast.Constant)
                    and call.args[0].value == "hidden"
                ):
                    reveal_found = True
    assert reveal_found, "no remove_class('hidden') call exists to reveal anything"

# test_chatbook_import_result_honesty.py
# Description: Regression coverage for task-19734 (the import wizard claimed
#              per-type imports, a success headline, and two options it never had)
"""
Three sibling defects of task-19550, all in ``ChatbookImportWizard``, all the
same shape: *the app asserts outcomes it did not produce.*

1. The per-type completion rows were painted from the MANIFEST
   (``manifest.total_conversations > 0`` and friends), so re-importing an
   already-imported chatbook under the default "skip existing" strategy
   showed four green ``✓ Imported ...`` rows and "Import Completed
   Successfully!" directly above a summary reading ``Imported: 0 /
   Skipped: N``. A whole-import-granularity fix would only relocate the lie
   to the partial case, so ``ImportStatus`` now carries per-type results and
   every row reports its own type's outcome.
2. ``preserve_timestamps`` and ``import_tags`` were collected into the import
   options and read by nothing at all.
3. "Merge with existing tags" (default ON) was passed as ``prefix_imported``,
   whose only effect is prepending ``"[Imported] "`` to item NAMES. It never
   touched a tag.

The end-to-end test below is the born-red one: it drives a real re-import of
a real chatbook through the real wizard step and fails at base on the four
``✓ Imported ...`` rows and the success headline.
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
from textual.widgets import Checkbox, Static

from tldw_chatbook.Chatbooks.chatbook_creator import ChatbookCreator
from tldw_chatbook.Chatbooks.chatbook_importer import (
    IMPORT_OUTCOME_EMPTY,
    IMPORT_OUTCOME_EXCLUDED,
    IMPORT_OUTCOME_FAILED,
    IMPORT_OUTCOME_IMPORTED,
    IMPORT_OUTCOME_NONE,
    IMPORT_OUTCOME_PARTIAL,
    IMPORT_OUTCOME_SKIPPED,
    ChatbookImporter,
    ImportStatus,
    ImportTypeResult,
)
from tldw_chatbook.Chatbooks.chatbook_models import (
    ChatbookManifest,
    ChatbookVersion,
    ContentType,
)
from tldw_chatbook.Chatbooks.conflict_resolver import ConflictResolution
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.UI.Wizards import ChatbookImportWizard as wizard_module
from tldw_chatbook.UI.Wizards.BaseWizard import WizardStepConfig
from tldw_chatbook.UI.Wizards.ChatbookImportWizard import (
    PER_TYPE_STATUS_ROWS,
    ImportOptionsStep,
    ImportProgressStep,
    describe_import_outcome,
    describe_type_result,
)

PER_TYPE_ROW_IDS = frozenset(row_id for row_id, _, _ in PER_TYPE_STATUS_ROWS)


class _StepHost(App):
    """Mount a single wizard step, the idiom used by Tests/Wizards."""

    def __init__(self, step):
        super().__init__()
        self._step = step

    def compose(self) -> ComposeResult:
        yield self._step


class _FakeWizard:
    """The slice of ``WizardContainer`` a mounted progress step touches."""

    def __init__(self, **wizard_data):
        self.app_instance = MagicMock(app_config={})
        self.wizard_data = dict(wizard_data)
        self.initial_execution_mode = "local"
        self.can_go_back = True
        self.next_button = SimpleNamespace(label="Next →", variant="primary")

    def query_one(self, selector, expect_type=None):
        assert selector == "#wizard-next", selector
        return self.next_button


def _options_wizard(**wizard_data) -> SimpleNamespace:
    return SimpleNamespace(
        app_instance=MagicMock(app_config={}),
        wizard_data=dict(wizard_data),
        initial_execution_mode="local",
        can_go_back=True,
    )


def _options_step() -> ImportOptionsStep:
    manifest = ChatbookManifest(
        version=ChatbookVersion.V1,
        name="Bundle",
        description="Bundle",
    )
    manifest.total_conversations = 1
    return ImportOptionsStep(
        wizard=_options_wizard(**{"preview-validation": {"manifest": manifest}}),
        config=WizardStepConfig(
            id="import-options",
            title="Import Options",
            step_number=4,
        ),
    )


def _status(
    *,
    total=0,
    successful=0,
    skipped=0,
    failed=0,
    by_type=None,
) -> ImportStatus:
    """Build an ``ImportStatus`` with the given aggregate (and per-type) counts."""
    status = ImportStatus()
    status.total_items = total
    status.successful_items = successful
    status.skipped_items = skipped
    status.failed_items = failed
    status.processed_items = successful + skipped + failed
    for content_type, counts in (by_type or {}).items():
        result = status.result_for(content_type)
        result.attempted = counts.get("attempted", 0)
        result.excluded = counts.get("excluded", 0)
        result.successful = counts.get("successful", 0)
        result.skipped = counts.get("skipped", 0)
        result.failed = counts.get("failed", 0)
    return status


# ---------------------------------------------------------------------------
# Fixtures: a real chatbook, and a destination that already contains it
# ---------------------------------------------------------------------------


def _make_db_paths(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    return {
        "ChaChaNotes": str(root / "ChaChaNotes.db"),
        "Media": str(root / "Client_Media_DB.db"),
        "Prompts": str(root / "Prompts_DB.db"),
    }


@pytest.fixture
def chatbook_zip(tmp_path, chachanotes_template_db) -> Path:
    """Export a chatbook holding one of each visible content type."""
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


def _import_once(chatbook_zip: Path, destination_paths: dict) -> ImportStatus:
    """Run one import with the wizard's own default options."""
    status = ImportStatus()
    ChatbookImporter(destination_paths).import_chatbook(
        chatbook_path=chatbook_zip,
        conflict_resolution=ConflictResolution.SKIP,
        prefix_imported=True,
        import_media=True,
        import_status=status,
    )
    return status


async def _run_progress_step(
    chatbook_zip: Path, destination_paths: dict, monkeypatch, **option_overrides
):
    """Mount the real progress step and let it run one real import."""
    manifest, error = ChatbookImporter(destination_paths).preview_chatbook(chatbook_zip)
    assert error is None, error

    options = {
        "execution_mode": "local",
        "import_media": True,
        "import_embeddings": False,
        "prefix_imported": True,
    }
    options.update(option_overrides)

    step = ImportProgressStep(
        wizard=_FakeWizard(
            **{
                "file-selection": {"file_path": str(chatbook_zip)},
                "preview-validation": {"manifest": manifest},
                "conflict-resolution": {"resolution_strategy": ConflictResolution.SKIP},
                "import-options": options,
            }
        ),
        config=WizardStepConfig(id="import-progress", title="Importing", step_number=5),
    )
    monkeypatch.setattr(
        wizard_module, "get_chatbook_database_paths", lambda: destination_paths
    )

    rows: dict = {}
    async with _StepHost(step).run_test(size=(120, 60)) as pilot:
        await pilot.pause()
        if step.import_worker is None:
            # `on_show` did not fire for this host; drive the same entry point.
            await step._import_chatbook()
        for _ in range(200):
            if step.is_complete:
                break
            await pilot.pause()
        assert step.is_complete, "the import step never reached its completion state"

        for row_id, _content_type, _noun in PER_TYPE_STATUS_ROWS:
            rows[row_id] = str(step.query_one(f"#{row_id}", Static).renderable)
        rows["status-finalize"] = str(
            step.query_one("#status-finalize", Static).renderable
        )
        headline = str(step.query_one("#progress-title", Static).renderable)
        banner = str(step.query_one("#completion-message", Static).renderable)
        stats = {
            stat: str(step.query_one(f"#{stat}", Static).renderable)
            for stat in ("stat-total", "stat-imported", "stat-skipped", "stat-failed")
        }
    return step, rows, headline, banner, stats


# ---------------------------------------------------------------------------
# (a) The born-red case: a fully-skipped re-import
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reimport_of_an_already_imported_chatbook_claims_no_imports(
    chatbook_zip, destination_paths, monkeypatch
):
    """AC#1/AC#2, end to end and born red.

    At this task's base the second run painted "✓ Imported conversations",
    "✓ Imported notes", "✓ Imported characters" and "✓ Imported media" -- all
    four off manifest counts -- under "✅ Import Completed Successfully!",
    while the summary panel underneath read Imported 0 / Skipped 4.
    """
    first = _import_once(chatbook_zip, destination_paths)
    assert first.successful_items > 0, "the fixture's first import must actually land"

    step, rows, headline, banner, stats = await _run_progress_step(
        chatbook_zip, destination_paths, monkeypatch
    )

    # The import genuinely imported nothing the second time round.
    assert step.import_status.successful_items == 0
    assert step.import_status.skipped_items == first.successful_items
    assert step.import_status.outcome == IMPORT_OUTCOME_SKIPPED

    per_type_rows = {row_id: rows[row_id] for row_id in PER_TYPE_ROW_IDS}
    assert not any("✓ Imported" in text for text in per_type_rows.values()), (
        per_type_rows
    )
    assert all("Skipped" in text for text in per_type_rows.values()), per_type_rows

    assert "Successfully" not in banner, banner
    assert "Nothing was imported" in banner, banner
    assert "Complete!" not in headline, headline

    # Headline and summary agree: zero imported.
    assert stats["stat-imported"] == "0"
    assert stats["stat-skipped"] == str(first.successful_items)


@pytest.mark.asyncio
async def test_first_import_still_reports_its_real_per_type_successes(
    chatbook_zip, destination_paths, monkeypatch
):
    """The honest positive case must keep working: a clean import ticks every
    type it actually wrote, with the count it wrote."""
    step, rows, headline, banner, stats = await _run_progress_step(
        chatbook_zip, destination_paths, monkeypatch
    )

    assert step.import_status.outcome == IMPORT_OUTCOME_IMPORTED
    assert rows["status-conversations"] == "✓ Imported 1 conversations"
    assert rows["status-notes"] == "✓ Imported 1 notes"
    assert rows["status-characters"] == "✓ Imported 1 characters"
    assert rows["status-media"] == "✓ Imported 1 media items"
    assert headline == "✅ Import Complete!"
    assert "Import completed" in banner
    assert stats["stat-imported"] == str(step.import_status.successful_items)


@pytest.mark.asyncio
async def test_media_left_out_by_the_user_is_never_reported_as_imported(
    chatbook_zip, destination_paths, monkeypatch
):
    """Turning "Import media files" off must not leave a row claiming media
    landed, nor an unexplained shortfall between Total and Imported."""
    step, rows, headline, banner, stats = await _run_progress_step(
        chatbook_zip, destination_paths, monkeypatch, import_media=False
    )

    assert "Imported" not in rows["status-media"], rows["status-media"]
    assert "were not imported" in rows["status-media"], rows["status-media"]
    assert step.import_status.outcome == IMPORT_OUTCOME_IMPORTED
    assert stats["stat-total"] == stats["stat-imported"]


# ---------------------------------------------------------------------------
# Per-type results: partial failure is distinguishable from success
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("counts", "expected_state", "expected_fragments"),
    [
        ({"attempted": 3, "successful": 3}, "completed", ["✓ Imported 3 notes"]),
        (
            {"attempted": 3, "successful": 1, "failed": 2},
            "warning",
            ["⚠ Imported 1 of 3 notes", "2 failed"],
        ),
        (
            {"attempted": 3, "successful": 2, "skipped": 1},
            "warning",
            ["⚠ Imported 2 of 3 notes", "1 skipped"],
        ),
        (
            {"attempted": 3, "skipped": 3},
            "warning",
            ["⊘ Skipped 3 notes", "already present"],
        ),
        ({"attempted": 3, "failed": 3}, "error", ["✗ Imported no notes", "3 failed"]),
        # An early return (missing DB path) records nothing at all: silence is
        # not success.
        (
            {"attempted": 3},
            "error",
            ["✗ Imported no notes", "3 unaccounted for"],
        ),
        ({}, "", ["No notes in this chatbook"]),
        ({"excluded": 4}, "", ["4 notes", "were not imported"]),
    ],
)
def test_each_type_row_reports_that_type_s_own_result(
    counts, expected_state, expected_fragments
):
    """AC#2: a type with zero successes never shows "✓ Imported ...", and a
    partially-failed type is distinguishable from a fully-successful one."""
    result = ImportTypeResult(ContentType.NOTE)
    for field, value in counts.items():
        setattr(result, field, value)

    state, text = describe_type_result(result, "notes")

    assert state == expected_state, (state, text)
    for fragment in expected_fragments:
        assert fragment in text, (fragment, text)
    if expected_state != "completed":
        assert "✓ Imported" not in text, text


def test_a_partially_failed_type_is_not_reported_like_a_successful_one():
    """The case a whole-import-granularity fix would have missed: notes landed,
    characters all failed, so the run 'succeeded' overall."""
    notes = ImportTypeResult(ContentType.NOTE)
    notes.attempted = notes.successful = 2
    characters = ImportTypeResult(ContentType.CHARACTER)
    characters.attempted = characters.failed = 2

    notes_state, notes_text = describe_type_result(notes, "notes")
    chars_state, chars_text = describe_type_result(characters, "characters")

    assert (notes_state, notes_text) == ("completed", "✓ Imported 2 notes")
    assert chars_state == "error"
    assert "✓ Imported" not in chars_text


# ---------------------------------------------------------------------------
# AC#3: the headline and the Imported/Skipped/Failed summary cannot disagree
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("successful", [0, 1, 3])
@pytest.mark.parametrize("skipped", [0, 1, 3])
@pytest.mark.parametrize("failed", [0, 1, 3])
def test_headline_never_contradicts_the_summary_counts(successful, skipped, failed):
    """AC#3 over every combination of imported / skipped / failed."""
    total = successful + skipped + failed
    status = _status(total=total, successful=successful, skipped=skipped, failed=failed)

    title, banner, state = describe_import_outcome(status)
    claims_success = "Successfully" in banner or "completed" in banner.lower()
    claims_nothing = "Nothing was imported" in banner

    if successful == 0:
        assert not claims_success, (banner, status.to_dict())
        assert claims_nothing, (banner, status.to_dict())
        assert state != "outcome-imported"
    else:
        assert not claims_nothing, (banner, status.to_dict())
        assert str(successful) in banner, banner

    if skipped or failed:
        assert state != "outcome-imported", (state, banner)
        assert "✅" not in title, (title, banner)
    if successful and not skipped and not failed:
        assert state == "outcome-imported", (state, banner)


def test_an_empty_chatbook_reports_nothing_rather_than_success():
    title, banner, state = describe_import_outcome(_status())

    assert state == "outcome-empty"
    assert "Nothing was imported" in banner
    assert "Successfully" not in banner
    assert "✅" not in title


# ---------------------------------------------------------------------------
# The importer's own return value and per-type bookkeeping
# ---------------------------------------------------------------------------


def test_an_all_skipped_import_says_in_words_that_nothing_was_imported(
    chatbook_zip, destination_paths
):
    """AC#1, caller-visible return value: ``success`` still means "no fatal
    error" (an all-skipped re-import is not an error), but the message it is
    returned with states that nothing was imported, and ``status.outcome``
    names it."""
    _import_once(chatbook_zip, destination_paths)

    status = ImportStatus()
    success, message = ChatbookImporter(destination_paths).import_chatbook(
        chatbook_path=chatbook_zip,
        conflict_resolution=ConflictResolution.SKIP,
        prefix_imported=True,
        import_status=status,
    )

    assert success is True
    assert "No items were imported" in message
    assert "Successfully imported" not in message
    assert status.outcome == IMPORT_OUTCOME_SKIPPED
    assert status.successful_items == 0


def test_per_type_results_sum_to_the_aggregate_counts(chatbook_zip, destination_paths):
    """The per-type ledger and the totals are two views of one run; a row that
    forgot to record its type would show up here."""
    status = _import_once(chatbook_zip, destination_paths)

    assert status.successful_items == sum(
        result.successful for result in status.by_type.values()
    )
    assert status.skipped_items == sum(
        result.skipped for result in status.by_type.values()
    )
    assert status.failed_items == sum(
        result.failed for result in status.by_type.values()
    )
    assert status.total_items == status.planned_items
    assert {content_type for content_type in status.by_type} == {
        ContentType.CONVERSATION,
        ContentType.NOTE,
        ContentType.CHARACTER,
        ContentType.MEDIA,
    }


def test_status_dict_carries_the_per_type_breakdown(chatbook_zip, destination_paths):
    status = _import_once(chatbook_zip, destination_paths)

    payload = status.to_dict()

    assert payload["outcome"] == IMPORT_OUTCOME_IMPORTED
    assert payload["by_type"]["note"]["successful"] == 1
    assert payload["by_type"]["note"]["outcome"] == IMPORT_OUTCOME_IMPORTED


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ({"attempted": 0}, IMPORT_OUTCOME_NONE),
        ({"attempted": 0, "excluded": 2}, IMPORT_OUTCOME_EXCLUDED),
        ({"attempted": 2, "successful": 2}, IMPORT_OUTCOME_IMPORTED),
        ({"attempted": 2, "successful": 1, "skipped": 1}, IMPORT_OUTCOME_PARTIAL),
        ({"attempted": 2, "skipped": 2}, IMPORT_OUTCOME_SKIPPED),
        ({"attempted": 2, "skipped": 1, "failed": 1}, IMPORT_OUTCOME_FAILED),
        ({"attempted": 2}, IMPORT_OUTCOME_FAILED),
    ],
)
def test_type_result_outcome_vocabulary(counts, expected):
    result = ImportTypeResult(ContentType.MEDIA)
    for field, value in counts.items():
        setattr(result, field, value)

    assert result.outcome == expected


def test_whole_import_outcome_is_empty_only_when_there_was_nothing_to_do():
    assert _status().outcome == IMPORT_OUTCOME_EMPTY
    assert _status(total=2, skipped=2).outcome == IMPORT_OUTCOME_SKIPPED
    assert _status(total=2, successful=2).outcome == IMPORT_OUTCOME_IMPORTED
    assert _status(total=2, successful=1, failed=1).outcome == IMPORT_OUTCOME_PARTIAL
    assert _status(total=2, failed=2).outcome == IMPORT_OUTCOME_FAILED
    # A run that recorded nothing at all for items it was asked to import is a
    # failure, not an empty chatbook.
    assert _status(total=2).outcome == IMPORT_OUTCOME_FAILED


# ---------------------------------------------------------------------------
# (b) Dead controls, (c) the mislabelled one
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_options_step_offers_no_control_that_nothing_reads():
    """AC#4: ``preserve_timestamps`` and ``import_tags`` were collected and
    consumed by nothing. Removed, not disabled -- a greyed box still reads as
    "handled"."""
    step = _options_step()
    async with _StepHost(step).run_test(size=(120, 60)) as pilot:
        await pilot.pause()

        labels = [str(checkbox.label).lower() for checkbox in step.query(Checkbox)]
        ids = [checkbox.id or "" for checkbox in step.query(Checkbox)]
        rendered = " ".join(
            str(static.renderable).lower() for static in step.query(Static)
        )
        data = step.get_step_data()

    assert labels, "the options step should still offer its real options"
    assert not any("preserve timestamps" in label for label in labels), labels
    assert "preserve-timestamps" not in ids, ids
    assert "preserve_timestamps" not in data, data
    assert "keep original creation and modification dates" not in rendered

    assert not any(label.strip() == "import tags" for label in labels), labels
    assert "import-tags" not in ids, ids
    assert "import_tags" not in data, data


@pytest.mark.asyncio
async def test_the_rename_option_is_labelled_as_a_rename_not_as_tag_merging():
    """AC#5: the box said "Merge with existing tags" (default ON) and was
    wired to ``prefix_imported``, which renames items and never reads a tag."""
    step = _options_step()
    async with _StepHost(step).run_test(size=(120, 60)) as pilot:
        await pilot.pause()

        labels = [str(checkbox.label) for checkbox in step.query(Checkbox)]
        ids = [checkbox.id or "" for checkbox in step.query(Checkbox)]
        rendered = " ".join(str(static.renderable) for static in step.query(Static))
        data = step.get_step_data()

    assert not any("merge with existing tags" in label.lower() for label in labels), (
        labels
    )
    assert "merge-tags" not in ids, ids
    assert "merge_tags" not in data, data
    assert "Combine imported tags with any existing tags" not in rendered

    assert "prefix-imported" in ids, ids
    prefix_label = next(
        label for label, widget_id in zip(labels, ids) if widget_id == "prefix-imported"
    )
    assert "[Imported]" in prefix_label, prefix_label
    assert "tag" not in prefix_label.lower(), prefix_label
    assert data["prefix_imported"] is True


def test_the_prefix_option_renames_items_and_nothing_else(
    chatbook_zip, destination_paths, tmp_path, chachanotes_template_db
):
    """The behaviour the label now promises, proven against the database: the
    option changes NAMES, which is exactly what ``prefix_imported`` does."""
    ChatbookImporter(destination_paths).import_chatbook(
        chatbook_path=chatbook_zip,
        conflict_resolution=ConflictResolution.SKIP,
        prefix_imported=True,
        import_media=True,
        import_status=ImportStatus(),
    )
    prefixed = CharactersRAGDB(destination_paths["ChaChaNotes"], "test_client")
    assert [note["title"] for note in prefixed.search_notes("Packing")] == [
        "[Imported] Packing list"
    ]
    assert [
        conversation["title"]
        for conversation in prefixed.search_conversations_by_title("Trip planning")
    ] == ["[Imported] Trip planning"]
    assert "[Imported] Field Guide" in [
        character["name"] for character in prefixed.list_character_cards()
    ]

    plain_paths = _make_db_paths(tmp_path / "destination-unprefixed")
    shutil.copyfile(chachanotes_template_db, plain_paths["ChaChaNotes"])
    CharactersRAGDB(plain_paths["ChaChaNotes"], "test_client")
    MediaDatabase(plain_paths["Media"], "test_client")
    ChatbookImporter(plain_paths).import_chatbook(
        chatbook_path=chatbook_zip,
        conflict_resolution=ConflictResolution.SKIP,
        prefix_imported=False,
        import_media=True,
        import_status=ImportStatus(),
    )
    plain = CharactersRAGDB(plain_paths["ChaChaNotes"], "test_client")
    assert [note["title"] for note in plain.search_notes("Packing")] == ["Packing list"]


def test_the_prefix_flag_only_ever_rewrites_a_name():
    """The label used to promise tag merging. Pin structurally that every
    ``prefix_imported`` branch in the importer rewrites a name or title and
    touches nothing else -- there is no tag write hiding behind the flag."""
    from tldw_chatbook.Chatbooks import chatbook_importer

    source = Path(inspect.getfile(chatbook_importer)).read_text(encoding="utf-8")
    tree = ast.parse(source)

    branches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "prefix_imported"
    ]
    assert len(branches) == 4, [node.lineno for node in branches]

    for branch in branches:
        assert branch.orelse == [], branch.lineno
        assert len(branch.body) == 1, ast.get_source_segment(source, branch)
        assignment = branch.body[0]
        assert isinstance(assignment, ast.Assign), ast.get_source_segment(
            source, branch
        )
        (target,) = assignment.targets
        assert isinstance(target, ast.Name), ast.get_source_segment(source, branch)
        assert target.id.endswith(("_name", "_title")), target.id


def test_the_wizard_passes_its_prefix_option_through_under_its_own_name():
    """The options key and the importer parameter now agree, so nobody has to
    remember that ``merge_tags`` secretly meant ``prefix_imported``."""
    source = inspect.getsource(ImportProgressStep._import_chatbook)

    assert "merge_tags" not in source, source
    assert source.count('options.get("prefix_imported", False)') == 2, source


# ---------------------------------------------------------------------------
# The structural lock: no per-type row may be painted from a manifest value
# ---------------------------------------------------------------------------


def _module_tree():
    source = Path(inspect.getfile(wizard_module)).read_text(encoding="utf-8")
    return source, ast.parse(source)


_MANIFEST_TOTAL_ATTRIBUTES = {
    "total_conversations",
    "total_notes",
    "total_characters",
    "total_media_items",
    "total_prompts",
    "total_kept_briefings",
}


def test_no_per_type_row_can_report_an_outcome_from_a_constant_state():
    """A per-type row may be pre-announced ("⟳ Importing notes...") but its
    OUTCOME must come from the results table, so no literal
    ``_update_status("status-notes", "completed", ...)`` may exist.

    This is the class-level lock task-19734 asks for: the bug was a row whose
    truth value came from the manifest, and the only way to reintroduce it is
    to hard-code an outcome state for one of these ids.
    """
    _source, tree = _module_tree()

    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "_update_status"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[1], ast.Constant)
        ):
            continue
        status_id, state = node.args[0].value, node.args[1].value
        if status_id in PER_TYPE_ROW_IDS and state not in ("active", ""):
            offenders.append((status_id, state, node.lineno))

    assert offenders == [], (
        "these per-type rows hard-code an outcome state instead of deriving it "
        f"from the import's results: {offenders}"
    )


def test_the_per_type_row_describer_cannot_see_a_manifest():
    """``describe_type_result`` takes a result and a noun. If a manifest value
    could reach it, a row could once again be true of the chatbook's contents
    and false of the import."""
    source, tree = _module_tree()

    describer = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "describe_type_result"
    )

    parameters = [argument.arg for argument in describer.args.args]
    assert parameters == ["result", "noun"], parameters
    assert describer.args.kwonlyargs == []
    assert describer.args.vararg is None and describer.args.kwarg is None

    body = ast.get_source_segment(source, describer) or ""
    assert "manifest" not in body.lower(), body
    referenced = {
        node.attr for node in ast.walk(describer) if isinstance(node, ast.Attribute)
    }
    assert not (referenced & _MANIFEST_TOTAL_ATTRIBUTES), referenced


def test_the_row_painter_reads_only_the_recorded_results():
    """``_paint_type_result_rows`` is the single producer of per-type outcome
    rows, and its only data source is ``self.import_status``."""
    source, tree = _module_tree()

    painter = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_paint_type_result_rows"
    )
    body = ast.get_source_segment(source, painter) or ""

    assert "import_status" in body
    assert "manifest" not in body.lower(), body
    referenced = {
        node.attr for node in ast.walk(painter) if isinstance(node, ast.Attribute)
    }
    assert not (referenced & _MANIFEST_TOTAL_ATTRIBUTES), referenced


def test_the_completion_banner_is_derived_and_never_shipped_pre_written():
    """The panel used to be composed with the literal
    "✅ Import Completed Successfully!" already in it, so the claim existed
    before any result did."""
    source, _tree = _module_tree()

    assert "Import Completed Successfully" not in source, (
        "the success banner must be produced by describe_import_outcome, not "
        "baked into compose()"
    )

    compose_source = inspect.getsource(ImportProgressStep.compose)
    assert 'id="completion-message"' in compose_source
    assert "✅" not in compose_source, compose_source

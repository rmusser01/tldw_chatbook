# test_chatbook_import_wizard_validation.py
# Description: Regression coverage for task-1870 fix-wave F2 (statistics-mismatch false alarm)
"""
`PreviewValidationStep._run_validation` (ChatbookImportWizard.py) compares
`len(manifest.content_items)` against a hand-summed "expected total" of the
manifest's per-type statistics, and shows a "Statistics mismatch" warning
when the two disagree. That sum omitted `total_kept_briefings` (this
feature's own content type) and `total_prompts` (pre-existing), so a
kept-only chatbook -- precisely the bundle task-1870 exists to produce --
always showed a false "Statistics mismatch" warning on its own happy path.

The arithmetic is exercised directly via the extracted
`_expected_content_total` staticmethod rather than by mounting the wizard
step: `_run_validation` itself calls `self.query_one(...)`, which needs a
running Textual App, while the mismatch check is pure arithmetic over the
manifest and does not need a live UI to verify.
"""

from tldw_chatbook.Chatbooks.chatbook_models import (
    ChatbookManifest,
    ChatbookVersion,
    ContentItem,
    ContentType,
)
from tldw_chatbook.UI.Wizards.ChatbookImportWizard import PreviewValidationStep


def _manifest(**totals) -> ChatbookManifest:
    manifest = ChatbookManifest(
        version=ChatbookVersion.V1,
        name="Test Bundle",
        description="Test",
    )
    for field, value in totals.items():
        setattr(manifest, field, value)
    return manifest


def test_kept_only_manifest_has_no_statistics_mismatch():
    """The happy path this task exists for: a chatbook containing only kept
    briefings must not trip the mismatch warning (task-1870 fix-wave F2)."""
    manifest = _manifest(total_kept_briefings=1)
    manifest.content_items = [
        ContentItem(id="1", type=ContentType.KEPT_BRIEFING, title="Kept"),
    ]

    expected_total = PreviewValidationStep._expected_content_total(manifest)

    assert expected_total == len(manifest.content_items) == 1


def test_prompts_only_manifest_has_no_statistics_mismatch():
    """`total_prompts` was already omitted before this task; folding it into
    the same fix (per the whole-branch review) must not regress it."""
    manifest = _manifest(total_prompts=2)
    manifest.content_items = [
        ContentItem(id="1", type=ContentType.PROMPT, title="P1"),
        ContentItem(id="2", type=ContentType.PROMPT, title="P2"),
    ]

    expected_total = PreviewValidationStep._expected_content_total(manifest)

    assert expected_total == len(manifest.content_items) == 2


def test_mixed_manifest_across_every_tracked_content_type_matches():
    manifest = _manifest(
        total_conversations=1,
        total_notes=1,
        total_characters=1,
        total_media_items=1,
        total_prompts=1,
        total_kept_briefings=1,
    )
    manifest.content_items = [
        ContentItem(id=str(i), type=content_type, title=str(i))
        for i, content_type in enumerate(
            [
                ContentType.CONVERSATION,
                ContentType.NOTE,
                ContentType.CHARACTER,
                ContentType.MEDIA,
                ContentType.PROMPT,
                ContentType.KEPT_BRIEFING,
            ]
        )
    ]

    expected_total = PreviewValidationStep._expected_content_total(manifest)

    assert expected_total == len(manifest.content_items) == 6


def test_genuine_mismatch_is_still_detected():
    """The fix must not paper over a REAL mismatch -- only the false alarm
    caused by the two missing terms."""
    manifest = _manifest(total_kept_briefings=1)
    manifest.content_items = []  # manifest claims 1 kept briefing, lists none

    expected_total = PreviewValidationStep._expected_content_total(manifest)

    assert expected_total == 1
    assert len(manifest.content_items) == 0
    assert expected_total != len(manifest.content_items)

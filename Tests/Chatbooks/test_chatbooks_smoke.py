from pathlib import Path
import tempfile

import pytest

from tldw_chatbook.Chatbooks.chatbook_creator import ChatbookCreator
from tldw_chatbook.Chatbooks.chatbook_models import ContentType


@pytest.mark.unit
def test_chatbook_creator_minimal_archive_creation():
    # Minimal test: create an empty chatbook archive to verify packaging path
    creator = ChatbookCreator(db_paths={})

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test_chatbook.zip"
        success, message, info = creator.create_chatbook(
            name="Smoke Chatbook",
            description="Smoke test",
            content_selections={},  # No content
            output_path=out_path,
            author="Test",
            include_media=False,
            include_embeddings=False,
            tags=["smoke"],
            categories=["test"],
        )

        assert success, message
        assert out_path.exists() and out_path.stat().st_size > 0


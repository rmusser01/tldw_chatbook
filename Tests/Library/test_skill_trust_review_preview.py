from tldw_chatbook.Widgets.Library.library_skills_canvas import skill_trust_review_preview


def test_present_binary_shows_metadata_not_deleted():
    review = {
        "changed_files": ["assets/logo.png"],
        "current_files": {},   # binary absent from text view
        "current_fingerprints": [
            {"relative_path": "assets/logo.png", "file_type": "supporting_binary",
             "byte_length": 2048, "sha256": "deadbeef"},
        ],
    }
    out = skill_trust_review_preview(review)
    assert "assets/logo.png" in out
    assert "binary" in out.lower()
    assert "2048" in out
    assert "deadbeef"[:8] in out
    assert "deleted" not in out.lower()


def test_genuinely_deleted_still_shows_deleted():
    review = {
        "changed_files": ["gone.md"],
        "current_files": {},
        "current_fingerprints": [],   # not on disk at all
    }
    out = skill_trust_review_preview(review)
    assert "deleted" in out.lower()

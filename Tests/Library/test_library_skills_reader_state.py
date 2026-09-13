"""Pure presentation contracts for the Library Skills reader."""

from tldw_chatbook.Library.library_skills_state import (
    coerce_skill_reader_mode,
    skill_review_identity_line,
)


def test_skill_reader_defaults_to_overview_and_accepts_explicit_modes() -> None:
    assert coerce_skill_reader_mode(None) == "overview"
    assert coerce_skill_reader_mode("unknown") == "overview"
    assert coerce_skill_reader_mode("overview") == "overview"
    assert coerce_skill_reader_mode("edit") == "edit"
    assert coerce_skill_reader_mode("trust") == "trust"
    assert coerce_skill_reader_mode("files") == "files"


def test_skill_review_identity_names_exact_generation_and_fingerprint() -> None:
    digest = "a" * 64

    assert (
        skill_review_identity_line(
            {"manifest_generation": 12, "current_digest": digest}
        )
        == f"Reviewed files · trust generation 12 · sha256:{digest}"
    )


def test_skill_review_identity_is_absent_without_captured_service_truth() -> None:
    assert skill_review_identity_line(None) == ""
    assert skill_review_identity_line({"manifest_generation": 12}) == ""

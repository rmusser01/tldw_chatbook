"""The Skills trust-header predicate is shared, so the two paths agree (TASK-32804.11).

compose derived has_skills with a `trust_posture == "recovery_review"` disjunct;
sync_state's header-only in-place update derived it WITHOUT that disjunct, so a
posture settling to recovery_review during a routine refresh dropped the
recovery banner (and its only "Review restored skills" action). Both now call
skill_trust_header_has_skills.
"""

from tldw_chatbook.Library.library_skills_state import skill_trust_header_has_skills


def test_recovery_review_always_shows_the_header():
    # The exact regression: recovery_review must show the header even with no
    # rows and a non-fresh summary (the in-place path used to return False).
    assert skill_trust_header_has_skills(
        "recovery_review",
        source_summary_fresh=False,
        has_rows=False,
        has_title_count=False,
        blocked_total=0,
    ) is True


def test_non_recovery_needs_a_fresh_populated_summary():
    # Not fresh -> hidden regardless of rows.
    assert skill_trust_header_has_skills(
        "ready", source_summary_fresh=False, has_rows=True,
        has_title_count=True, blocked_total=9,
    ) is False
    # Fresh but empty -> hidden.
    assert skill_trust_header_has_skills(
        "ready", source_summary_fresh=True, has_rows=False,
        has_title_count=False, blocked_total=0,
    ) is False
    # Fresh with rows / title count / blocked -> shown.
    assert skill_trust_header_has_skills(
        "ready", source_summary_fresh=True, has_rows=True,
        has_title_count=False, blocked_total=0,
    ) is True
    assert skill_trust_header_has_skills(
        "ready", source_summary_fresh=True, has_rows=False,
        has_title_count=False, blocked_total=3,
    ) is True

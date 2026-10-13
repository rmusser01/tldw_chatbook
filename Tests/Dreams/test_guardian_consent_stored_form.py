# Tests/Dreams/test_guardian_consent_stored_form.py
"""The Guardian consent check and the profile refresh store topics one way.

A property test, kept out of ``test_cycle_service.py`` because the Console P0
gate runs named tests from that file in an environment without Hypothesis
(follow-up to #2921; the import made the whole file fail to collect there).
"""

import pytest
from hypothesis import HealthCheck, given
from hypothesis import settings as hyp_settings
from hypothesis import strategies as st

from tldw_chatbook.DB.Dreams_DB import DreamsDB


@pytest.fixture()
def dreams_db(tmp_path):
    database = DreamsDB(tmp_path / "dreams.sqlite", "test-client")
    yield database
    database.close_all()


@hyp_settings(
    max_examples=400,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    st.text(
        alphabet=st.sampled_from(
            "ab CD\u00e9e\u0301\u0958\u200b\u00a0\u2062\u2800\u3000I\u0130"
            "\u00df\u1e9e\u212a\u03a3\u03c2\ufb01"
        ),
        min_size=1,
        max_size=40,
    )
)
def test_the_stored_topic_helper_is_what_the_refresh_really_stores(dreams_db, label):
    """_stored_topic_text must give exactly the text the refresh STORES,
    read back from the database -- not a rebuild of the refresh's steps,
    which could not see the storing step drift from the helper (review of
    #3097). If it raises, the refresh must store nothing. The alphabet
    carries letters whose casefold differs from their lowercase (sharp s,
    final sigma, the fi ligature)."""
    from tldw_chatbook.Dreams.cycle_service import (
        _preferred_sources,
        _stored_form_signals,
        _stored_topic_text,
        _upsert_profile_signals,
    )
    from tldw_chatbook.Dreams.interest_profile import merge_signals

    with dreams_db.transaction() as conn:
        conn.execute("DELETE FROM dream_interest_profile")
    collected, dropped = _stored_form_signals(
        [
            (
                "guardian",
                [
                    {
                        "facet": "topic",
                        "text": label,
                        "weight": 0.5,
                        "source": "guardian",
                    }
                ],
            )
        ]
    )
    _upsert_profile_signals(
        dreams_db,
        merge_signals([rows for _origin, rows in collected]),
        _preferred_sources(collected),
        now_iso="2026-09-22T08:00:00+00:00",
        dropped=dropped,
    )
    stored = {row["text"] for row in dreams_db.list_profile()}

    try:
        expected = {_stored_topic_text(label)}
    except ValueError:
        expected = set()
    assert stored == expected

"""TASK-1091: the delete copy's grammar, and the trimming claim it came with.

The third Watchlists UAT reported two things. One was real, one was not, and the
difference is worth keeping:

* **Real.** With one source attached, the delete confirmation read *"Its 1
  source are not deleted. They stay in..."*. The noun was pluralised; the verb
  and pronoun were not.
* **Not real.** It also reported that names keep leading whitespace, citing a
  tree row that appeared indented. `WatchlistBundleService` strips on both
  create and rename and already rejects a whitespace-only name — asserted below,
  so the claim is not re-filed a third time. The indent came from Textual
  centring a `Button` label; short names sit further right than long ones. Fixed
  in CSS by left-aligning the tree labels.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions.watchlist_bundle_service import WatchlistBundleService
from tldw_chatbook.UI.Screens.watchlists_collections_screen import (
    watchlist_delete_consequence,
)

pytestmark = pytest.mark.unit


def _service(tmp_path) -> WatchlistBundleService:
    return WatchlistBundleService(SubscriptionsDB(tmp_path / "subs.db", "test"))


def test_delete_copy_reads_correctly_for_one_source():
    """AC#3: singular gets a singular verb and pronoun."""
    message = watchlist_delete_consequence(1)
    assert "1 source is not deleted" in message
    assert "It stays in Watchlists" in message
    assert "are not deleted" not in message
    assert "They stay" not in message


@pytest.mark.parametrize("count", [0, 2, 7])
def test_delete_copy_reads_correctly_for_other_counts(count):
    """The plural branch keeps the wording that was already good."""
    message = watchlist_delete_consequence(count)
    assert f"{count} sources are not deleted" in message
    assert "They stay in Watchlists" in message


def test_delete_copy_still_explains_the_consequence():
    """The value of this sentence is that it says what happens next.

    Both branches must keep the Unassigned explanation — that is the part a
    user actually needs, and it is easy to lose while fixing the grammar.
    """
    for count in (1, 3):
        message = watchlist_delete_consequence(count)
        assert "Unassigned" in message
        assert "another watchlist" in message


def test_names_are_already_trimmed_on_create_and_rename(tmp_path):
    """The UAT's trimming claim, checked rather than assumed.

    Asserted so the finding is not re-filed: this is the second UAT report in
    the programme that did not reproduce.
    """
    service = _service(tmp_path)
    created = service.create("  Daily  ")
    assert created["name"] == "Daily"

    renamed = service.rename(int(created["id"]), "  Renamed  ")
    assert renamed["name"] == "Renamed"


def test_a_whitespace_only_name_is_rejected(tmp_path):
    """AC#2, already satisfied: it raises rather than storing an unnameable row."""
    service = _service(tmp_path)
    with pytest.raises(ValueError):
        service.create("   ")
    created = service.create("Real")
    with pytest.raises(ValueError):
        service.rename(int(created["id"]), "   ")

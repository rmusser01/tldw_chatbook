"""Original dict descriptors do not prove an instance has an exact plain dict."""

import os

from Tests.Backup_Recovery import test_raw_related_source_preparation as sources
from Tests.Backup_Recovery import test_raw_stock_hold_pause as pause_controls
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.admission import Admission

configured_source = sources.configured_source
local_root = sources.local_root


def test_authority_dict_subclass_declines_grouping_without_foreign_membership(
    configured_source, local_root, monkeypatch
):
    source = configured_source
    startup = storage._startups[(os.getpid(), str(local_root))]
    authority = storage._holds[startup._key].authority
    assert type(authority) is Admission
    original_values = vars(authority)
    assert type(original_values) is dict  # noqa: E721 - exact baseline mapping.
    foreign_memberships, calls = [], []

    class ForeignDict(dict):
        def __contains__(self, key):
            foreign_memberships.append(key)
            return super().__contains__(key)

    before_leases = set(storage._live_leases)
    with monkeypatch.context() as patch:
        patch.setattr(authority, "__dict__", ForeignDict(original_values))
        assert type(vars(authority)) is ForeignDict
        # The class type, lookup and original __dict__ descriptor stay original.
        # Only the actual returned mapping is custom. Old dynamic native calls
        # must run twice; optional eligibility must never call its membership.
        with sources._original_scope_acquisitions(source) as observed:
            with pause_controls._scope_pause_calls(
                source, lambda state, hold: calls.append((state, hold))
            ):
                assert pause_controls._read(source) == "retirement"
        assert len(observed.entries) == 1
        operation, state, leases = observed.entries[0]
        assert len(leases) == 2 and state.holds[0] is state.holds[1]
        assert all(row[0] is state and row[1] is state.holds[0] for row in calls)
        assert len(calls) == 2
        assert foreign_memberships == []
        assert set(storage._live_leases) == before_leases
        assert operation not in storage._raw_operations
    assert vars(authority) is original_values

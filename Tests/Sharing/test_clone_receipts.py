"""Sharing retains Clone operation identity across reopening and lost responses."""

import pytest

from tldw_chatbook.Sharing_Interop.clone_receipts import SharingCloneReceipts


def test_clone_receipt_survives_reopen_and_is_qualified_to_the_server_account(tmp_path):
    root = tmp_path / "private"
    first = SharingCloneReceipts(root, "profile-a", "account-a")
    before = first.prepare(27)
    assert before.operation_id == ""

    restarted = SharingCloneReceipts(root, "profile-a", "account-a")
    assert restarted.prepare(27) == before
    operation = "d290f1ee-6c54-4b01-90e6-d701748f0851"
    restarted.remember(27, before.idempotency_key, operation)
    assert first.get(27).operation_id == operation
    assert first.list_receipts() == {27: first.get(27)}

    for profile, account in (("profile-b", "account-a"), ("profile-a", "account-b")):
        assert SharingCloneReceipts(root, profile, account).get(27) is None
    with pytest.raises(ValueError, match="different clone operation"):
        restarted.remember(
            27, before.idempotency_key, "45415d39-53e9-4e85-9e2b-e436361ec868"
        )


def test_clone_receipt_must_belong_to_the_prepared_operation(tmp_path):
    store = SharingCloneReceipts(tmp_path / "private", "profile", "account")
    first = store.prepare(27)
    second = store.prepare(28)
    with pytest.raises(ValueError, match="different clone request"):
        store.remember(
            27, second.idempotency_key, "d290f1ee-6c54-4b01-90e6-d701748f0851"
        )
    assert store.get(27) == first

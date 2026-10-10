"""Real host directory durability and descriptor retirement for Tool Packs."""

import errno
import os

import pytest

from Tests.private_profile import private_profile_test


@pytest.mark.asyncio
@private_profile_test
async def test_receipt_barrier_flushes_an_actual_directory(request, tmp_path):
    from tldw_chatbook.Tool_Packs.receipt_store import ToolPackReceiptStore

    assert tmp_path.is_dir()
    ToolPackReceiptStore._fsync_directory(tmp_path)
    store = ToolPackReceiptStore(tmp_path / "receipts")
    assert store.root.is_dir()


@pytest.mark.asyncio
@private_profile_test
async def test_receipt_barrier_propagates_failure_and_closes_descriptor(
    request, tmp_path, monkeypatch
):
    from tldw_chatbook.Tool_Packs.receipt_store import ToolPackReceiptStore

    observed = []
    failure = OSError(errno.EIO, "declared directory durability failure")

    def failed_barrier(descriptor):
        os.fstat(descriptor)
        observed.append(descriptor)
        raise failure

    with monkeypatch.context() as patch:
        if os.name == "nt":
            from tldw_chatbook.Utils import windows_files

            patch.setattr(windows_files, "flush_directory", failed_barrier)
        else:
            patch.setattr(os, "fsync", failed_barrier)
        with pytest.raises(OSError) as caught:
            ToolPackReceiptStore._fsync_directory(tmp_path)
    assert caught.value is failure
    assert len(observed) == 1
    with pytest.raises(OSError) as closed:
        os.fstat(observed[0])
    assert closed.value.errno == errno.EBADF


@pytest.mark.asyncio
@private_profile_test
async def test_receipt_publishes_private_native_file_and_reads_it(request, tmp_path):
    from Tests.Tool_Packs.test_receipt_store import _receipt_bytes
    from tldw_chatbook.Tool_Packs.receipt_store import ToolPackReceiptStore

    store = ToolPackReceiptStore(tmp_path / "receipts")
    data = _receipt_bytes()
    with store.reserve(len(data)) as reservation:
        handle = reservation.commit(data)
    verified = store.read(handle.receipt_id, expected_digest=handle.digest)
    assert verified.handle == handle
    assert verified.receipt.profile_id == "research"

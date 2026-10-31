"""Ineligible raw-pin evidence adds no metadata reads or ownership."""

from __future__ import annotations

# Imported fixture names intentionally match pytest parameters.
import errno
import os
import sys

import pytest

from Tests.Backup_Recovery.config_test_support import install_config_source
from Tests.Backup_Recovery.test_participant_lifetimes import (
    local_root as local_root,  # noqa: PLC0414 - exported pytest fixture.
)
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage


@pytest.mark.parametrize(
    "reuse_enabled",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                os.name != "nt", reason="Actual Windows eligibility"
            ),
        ),
    ],
)
def test_original_config_pin_skips_ineligible_evidence(
    tmp_path, monkeypatch, local_root, reuse_enabled
):
    data = tmp_path / "data"
    data.mkdir(mode=0o700)
    selector = tmp_path / "config.toml"
    selector.write_text(
        '[general]\nusers_name="raw-pin"\n[paths]\ndata_dir="' + data.as_posix() + '"\n'
    )
    selector.chmod(0o600)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(selector))
    source = install_config_source(monkeypatch)
    source.get_user_data_dir()
    with source.locked_hooks_config_snapshot():
        pass
    if not reuse_enabled:
        monkeypatch.setattr(storage, "_EVIDENCE_REUSE", False)
    before = (
        frozenset(raw._states),
        frozenset(storage._raw_operations),
        frozenset(storage._live_leases),
    )
    metadata = []
    pins = []
    closed = []
    previous = sys.getprofile()

    def observe(frame, event, value):
        if previous is not None:
            previous(frame, event, value)
        if event == "call" and frame.f_code is storage._path_evidence.__code__:
            parent = frame.f_back
            while parent is not None:
                if parent.f_code is raw._pin_parent.__code__:
                    if parent.f_locals["state"].source is source:
                        metadata.append(True)
                    break
                parent = parent.f_back
        if (
            event == "return"
            and frame.f_code is raw._pin_parent.__code__
            and frame.f_locals["state"].source is source
        ):
            pins.append((frame.f_locals["state"], value))
        if event == "return" and frame.f_code is raw._close_descriptor.__code__:
            state, fd = frame.f_locals["state"], frame.f_locals["fd"]
            if state.source is source and (state, fd) in pins:
                try:
                    os.fstat(fd)
                except OSError as error:
                    closed.append(error.errno == errno.EBADF)
                else:
                    closed.append(False)

    sys.setprofile(observe)
    try:
        with source.locked_hooks_config_snapshot() as snapshot:
            assert snapshot.config_path == selector
    finally:
        sys.setprofile(previous)
    assert pins and len(closed) == len(pins) and all(closed)
    assert all(not state.active and not state.descriptors for state, _ in pins)
    assert before == (
        frozenset(raw._states),
        frozenset(storage._raw_operations),
        frozenset(storage._live_leases),
    )
    assert not metadata, "raw pin read evidence that cannot be published"

"""PERF-07/08 (TASK-33266/33267): reused admission evidence never changes a verdict.

ADR-126's 2026-09-29 amendment lets ordinary storage admission reuse the allowed
result of an unmodified derivation while per-call stamps stay identical. The
oracle below applies each mutation in a catalog, then compares:

* ``acquire_storage`` with evidence reuse available, against
* ``acquire_storage`` with reuse switched off (the full derivation).

They must agree exactly: the same namespaces, or the same refusal reason. Some
writers run in a subprocess, so no in-process hook can be what notices them;
only the stamps can. The settle margin is patched to 0 so evidence recorded
milliseconds ago is reusable, the worst case for coarse timestamps.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from Tests.Backup_Recovery.test_bootstrap import local_scope  # noqa: F401
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.control_records import bind_profile

REPO = Path(__file__).resolve().parents[2]


def _verdict(path: Path) -> tuple[str, object]:
    """Acquire once and release; report what the caller would have observed."""
    try:
        with storage.acquire_storage(path) as lease:
            return ("allowed", lease.execution_context(path)[1])
    except bootstrap.RecoveryRequired as error:
        return ("refused", str(error))


def _in_subprocess(code: str, **values: str) -> None:
    """Run a protocol writer in another process, so no in-process hook fires."""
    script = "import json, sys\nvalues = json.loads(sys.argv[1])\n" + code
    completed = subprocess.run(
        [sys.executable, "-c", script, json.dumps(values)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO)},
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]


def _pending_from_subprocess(root, config, data, tmp_path):
    _in_subprocess(
        "from pathlib import Path\n"
        "from tldw_chatbook.Backup_Recovery.control_records import register_pending\n"
        "register_pending(Path(values['root']), 'op', ('profile',),"
        " Path(values['control']), (Path(values['config']),))\n",
        root=str(root), control=str(tmp_path / "control"), config=str(config),
    )


def _unrelated_pending_from_subprocess(root, config, data, tmp_path):
    other = tmp_path / "other.toml"
    other.write_text("other")
    elsewhere = tmp_path / "elsewhere-data"
    elsewhere.mkdir(mode=0o700)
    _in_subprocess(
        "from pathlib import Path\n"
        "from tldw_chatbook.Backup_Recovery.admission import Admission\n"
        "from tldw_chatbook.Backup_Recovery.control_records import register_pending\n"
        "Admission.open_existing(Path(values['root']) / 'admission')"
        ".register('elsewhere', (Path(values['elsewhere']),))\n"
        "register_pending(Path(values['root']), 'op2', ('elsewhere',),"
        " Path(values['control']), (Path(values['other']),))\n",
        root=str(root), control=str(tmp_path / "control2"),
        other=str(other), elsewhere=str(elsewhere),
    )


def _registry_remap_from_subprocess(root, config, data, tmp_path):
    moved = tmp_path / "moved"
    moved.mkdir(mode=0o700)
    _in_subprocess(
        "from pathlib import Path\n"
        "from tldw_chatbook.Backup_Recovery.admission import Admission\n"
        "Admission.open_existing(Path(values['root']) / 'admission')"
        ".register('extra', (Path(values['moved']),))\n",
        root=str(root), moved=str(moved),
    )


def _registry_intent(root, config, data, tmp_path):
    intent = root / "admission" / "registry.pending.json"
    intent.write_text("{}")
    intent.chmod(0o600)


def _profile_edited_in_place(root, config, data, tmp_path):
    record = next(root.glob("profile-*.json"))
    body = json.loads(record.read_text())
    body["roots"] = sorted([*body["roots"], str(tmp_path / "grafted")])
    with open(record, "r+", encoding="utf-8") as stream:  # same inode, new bytes
        stream.seek(0)
        stream.write(json.dumps(body))
        stream.truncate()


def _marker_replaced(root, config, data, tmp_path):
    marker = root / "unbound-owner"
    replacement = root / "unbound-owner.new"
    replacement.write_bytes(marker.read_bytes())
    replacement.chmod(0o600)
    os.replace(replacement, marker)


def _selector_edited(root, config, data, tmp_path):
    config.write_text("scope2")


def _ancestor_group_writable(root, config, data, tmp_path):
    tmp_path.chmod(0o777)


def _data_renamed_and_recreated(root, config, data, tmp_path):
    data.rename(tmp_path / "data.old")
    data.mkdir(mode=0o700)


def _data_swapped_for_symlink(root, config, data, tmp_path):
    target = tmp_path / "elsewhere"
    target.mkdir(mode=0o700)
    data.rename(tmp_path / "data.old")
    data.symlink_to(target, target_is_directory=True)


def _bootstrap_root_removed(root, config, data, tmp_path):
    shutil.rmtree(root)


def _nothing(root, config, data, tmp_path):
    pass


MUTATIONS = {
    "control-no-change": _nothing,
    "pending-record-from-subprocess": _pending_from_subprocess,
    "unrelated-pending-from-subprocess": _unrelated_pending_from_subprocess,
    "registry-replaced-from-subprocess": _registry_remap_from_subprocess,
    "registry-intent-file": _registry_intent,
    "profile-record-edited-in-place": _profile_edited_in_place,
    "enrollment-marker-replaced": _marker_replaced,
    "config-selector-edited": _selector_edited,
    "ancestor-made-group-writable": _ancestor_group_writable,
    "data-dir-renamed-and-recreated": _data_renamed_and_recreated,
    "data-dir-swapped-for-symlink": _data_swapped_for_symlink,
    "bootstrap-root-removed": _bootstrap_root_removed,
}

#: Mutations the full derivation is known to refuse. Pinned so the oracle can
#: never pass vacuously by having both sides allow everything.
REFUSED_BY_THE_DERIVATION = {
    "pending-record-from-subprocess",
    "registry-intent-file",
    "profile-record-edited-in-place",
    "enrollment-marker-replaced",
    "ancestor-made-group-writable",
}


@pytest.fixture
def reuse_switch(monkeypatch):
    """Toggle evidence reuse; a no-op until the reuse path exists."""
    monkeypatch.setattr(storage, "_EVIDENCE_SETTLE_NS", 0, raising=False)

    def switch(enabled: bool) -> None:
        monkeypatch.setattr(storage, "_EVIDENCE_REUSE", enabled, raising=False)

    return switch


@pytest.mark.parametrize("bound", [True, False], ids=["bound", "unbound"])
@pytest.mark.parametrize("mutation", sorted(MUTATIONS))
def test_reused_evidence_matches_the_full_derivation(
    local_scope, reuse_switch, tmp_path, mutation, bound  # noqa: F811
):
    root, config, data, _ = local_scope
    if mutation == "profile-record-edited-in-place" and not bound:
        pytest.skip("an unbound selection has no profile record to edit")
    if bound:
        bind_profile(root, config, ("profile",), root / "admission")
    target = data / "store.db"
    reuse_switch(True)
    # A live lease keeps the native hold -- and any evidence on it -- alive, as
    # the startup enrollment does in the running app.
    startup = storage.acquire_storage()
    try:
        for _ in range(2):  # record evidence, then serve from it
            assert _verdict(target)[0] == "allowed"
        try:
            MUTATIONS[mutation](root, config, data, tmp_path)
            reused = _verdict(target)
            reuse_switch(False)
            derived = _verdict(target)
        finally:
            tmp_path.chmod(0o700)
    finally:
        startup.close()

    assert reused == derived, f"{mutation}: reuse gave {reused}, derivation {derived}"
    if mutation in REFUSED_BY_THE_DERIVATION:
        assert derived[0] == "refused", f"{mutation} was expected to refuse: {derived}"

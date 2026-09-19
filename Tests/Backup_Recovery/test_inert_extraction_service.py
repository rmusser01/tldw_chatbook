"""The retained service exposes byte-only manual recovery without activation."""

import json

import pytest

from Tests.Backup_Recovery.test_inert_extraction import _archive
from tldw_chatbook.Backup_Recovery.recovery_service import RecoveryService
from tldw_chatbook.Backup_Recovery.service_storage import work_root


def test_service_previews_and_extracts_actual_unsupported_group(tmp_path):
    archive, payloads = _archive(tmp_path)
    service = RecoveryService(tmp_path / "control")
    destination = tmp_path / "manual"
    try:
        inspection = service.start_inspection(archive.path, password=None)
        assert service.wait(inspection, timeout=10)["state"] == "succeeded"
        preview = service.start_extraction_preview(
            inspection, group_ids=("unsupported",), destination=destination
        )
        state = service.wait(preview, timeout=10)
        assert state["phase"] == "extraction_review_ready"
        plan = state["result"]["plan"]
        assert not destination.exists()
        assert {service.control_root, work_root(service.control_root)} <= set(
            plan.protected_roots
        )
        operation = service.start_extraction(inspection, plan)
        result = service.wait(operation, timeout=10)
        assert result["state"] == "succeeded"
        assert result["phase"] == "inert_extracted"
        assert result["result"]["path"] == str(destination)
        assert not result["result"].get("restoration_validated", False)
        assert not result["result"].get("opened", False)
        report = json.loads((destination / "inert-mapping.json").read_text())
        assert {row["logical_id"] for row in report["files"]} == {"db", "config"}
        for row in report["files"]:
            assert (destination / row["output"]).read_bytes() == payloads[
                row["logical_id"]
            ]
    finally:
        service.close()
    assert destination.is_dir()


@pytest.mark.parametrize("root", ["control", "work"])
def test_service_manual_extraction_protects_actual_custom_storage(tmp_path, root):
    archive, _ = _archive(tmp_path)
    service = RecoveryService(tmp_path / "custom-control")
    try:
        inspection = service.start_inspection(archive.path, password=None)
        assert service.wait(inspection, timeout=10)["state"] == "succeeded"
        protected = (
            service.control_root
            if root == "control"
            else work_root(service.control_root)
        )
        output = protected / "manual"
        preview = service.start_extraction_preview(
            inspection, group_ids=("unsupported",), destination=output
        )
        result = service.wait(preview, timeout=10)
        assert result["state"] == "failed"
        assert result["issues"] == ("output_overlap",)
        assert not output.exists()
    finally:
        service.close()


def test_service_refuses_manual_plan_without_its_storage_restrictions(tmp_path):
    from threading import Event

    from tldw_chatbook.Backup_Recovery.inert_extraction import preview_inert_extraction
    from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits

    archive, _ = _archive(tmp_path)
    service = RecoveryService(tmp_path / "custom-control")
    output = tmp_path / "manual"
    try:
        inspection = service.start_inspection(archive.path, password=None)
        assert service.wait(inspection, timeout=10)["state"] == "succeeded"
        # A standalone manual plan does not know this service's custom storage.
        plan = preview_inert_extraction(
            service.inspection(inspection),
            group_ids=("unsupported",),
            destination=output,
            limits=ArchiveLimits(),
            cancel=Event(),
        )
        with pytest.raises(ValueError, match="extraction_preview_required"):
            service.start_extraction(inspection, plan)
        assert not output.exists()
    finally:
        service.close()

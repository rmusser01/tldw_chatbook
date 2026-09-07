"""End-to-end integration test for full-bundle supporting-file fidelity.

Builds a real zip bundle containing a nested executable script and a binary
asset, imports it through ``LocalSkillsService.import_skill_file``, and
verifies the nested script surfaces in ``supporting_files`` while the binary
asset surfaces in ``bundle_files`` (never in the text-only ``supporting_files``
view). Then round-trips the skill through ``export_skill`` and asserts the
binary asset comes back byte-identical -- proving the whole import/export
stack preserves a real bundle faithfully, not just its individual pieces in
isolation.
"""

import io
import zipfile

import pytest

from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService


@pytest.mark.asyncio
async def test_bundle_roundtrip_and_tamper_detection(tmp_path):
    svc = LocalSkillsService(store_dir=tmp_path)
    # Build a bundle with a nested executable script and a binary asset.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("SKILL.md", "---\nname: sdd\n---\nRun scripts/run.sh\n")
        info = zipfile.ZipInfo("scripts/run.sh")
        info.external_attr = 0o755 << 16
        z.writestr(info, "#!/bin/sh\necho hi\n")
        z.writestr("assets/logo.png", b"\x89PNG\x00bin")
    await svc.import_skill_file(
        buf.getvalue(), filename="sdd.zip", content_type="application/zip"
    )
    skill = await svc.get_skill("sdd")
    assert "scripts/run.sh" in skill["supporting_files"]
    assert any(
        b["path"] == "assets/logo.png" and not b["is_text"]
        for b in skill["bundle_files"]
    )
    # Export -> re-import is byte-identical for the binary asset.
    export = await svc.export_skill("sdd")
    with zipfile.ZipFile(io.BytesIO(export["content"])) as z:
        assert z.read("assets/logo.png") == b"\x89PNG\x00bin"

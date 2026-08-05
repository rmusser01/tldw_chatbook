"""task-584: files a skill script produces must survive, bounded and listed.

The scratch directory was deleted immediately after every run, so any artifact a
script produced was destroyed — blocking the whole class of extract/convert/
report skills. Output is now retained in a per-run directory the USER can open,
and the tool result carries a listing (never contents), with the oldest run
directories pruned so retention cannot grow without bound.
"""

import pytest

from tldw_chatbook.Skills_Interop import local_skills_service as svc_module


@pytest.mark.asyncio
async def test_produced_files_survive_the_run(script_service):
    """AC#2: the artifact is still there afterwards."""
    service, name = script_service
    (service._skill_dir(name) / "scripts" / "make.py").write_text(
        "open('report.txt','w').write('hello artifact')\n", encoding="utf-8"
    )
    service.trust_service.trust_current_skill(name)

    result = await service.run_skill_script(name, "scripts/make.py", [])
    assert result.exit_code == 0
    assert result.output_dir is not None
    produced = {f["name"]: f for f in result.output_files}
    assert "report.txt" in produced
    from pathlib import Path

    assert (Path(result.output_dir) / "report.txt").read_text() == "hello artifact"


@pytest.mark.asyncio
async def test_listing_reports_size_but_never_contents(script_service):
    """AC#6: the transcript must not be a dumping ground for file contents."""
    service, name = script_service
    (service._skill_dir(name) / "scripts" / "make.py").write_text(
        "open('data.txt','w').write('SECRETPAYLOAD')\n", encoding="utf-8"
    )
    service.trust_service.trust_current_skill(name)

    result = await service.run_skill_script(name, "scripts/make.py", [])
    entry = next(f for f in result.output_files if f["name"] == "data.txt")
    assert entry["size"] == len("SECRETPAYLOAD")
    assert "SECRETPAYLOAD" not in str(result.output_files)


@pytest.mark.asyncio
async def test_a_run_producing_nothing_lists_nothing(script_service):
    service, name = script_service
    result = await service.run_skill_script(name, "scripts/hello.py", [])
    assert result.exit_code == 0
    assert result.output_files == ()


@pytest.mark.asyncio
async def test_old_run_directories_are_pruned(script_service, monkeypatch):
    """AC#3: retention is bounded by run count."""
    monkeypatch.setattr(svc_module, "SCRIPT_OUTPUT_KEEP_RUNS", 3)
    service, name = script_service
    from pathlib import Path

    # Must PRODUCE something: a run with no artifacts keeps no directory.
    (service._skill_dir(name) / "scripts" / "make.py").write_text(
        "open('a.txt','w').write('x')\n", encoding="utf-8"
    )
    service.trust_service.trust_current_skill(name)
    for _ in range(6):
        result = await service.run_skill_script(name, "scripts/make.py", [])
    root = Path(result.output_dir).parent
    runs = [p for p in root.iterdir() if p.is_dir()]
    assert len(runs) <= 3, f"expected pruning to 3, found {len(runs)}"
    # The newest run must always survive its own pruning pass.
    assert Path(result.output_dir).exists()


@pytest.mark.asyncio
async def test_output_root_is_never_inside_a_skill_bundle(script_service):
    """AC#5: a run must not be able to write into its own trusted bundle."""
    service, name = script_service
    (service._skill_dir(name) / "scripts" / "make.py").write_text(
        "open('a.txt','w').write('x')\n", encoding="utf-8"
    )
    service.trust_service.trust_current_skill(name)
    result = await service.run_skill_script(name, "scripts/make.py", [])
    from pathlib import Path

    out = Path(result.output_dir).resolve()
    skills_root = Path(service._skill_dir(name)).resolve().parent
    assert skills_root not in out.parents and out != skills_root

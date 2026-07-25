"""Shared fixtures for `Tests/Skills/`.

Construction of `SkillTrustStore`/`SkillTrustService` follows the pattern
established in `test_skill_trust_service.py`: `SkillTrustStore` requires an
explicit `marker_store` (a file-backed marker store is fine for tests).
"""

import pytest

from tldw_chatbook.Skills_Interop.skill_trust_service import SkillTrustService
from tldw_chatbook.Skills_Interop.skill_trust_store import (
    FileSkillTrustGenerationMarkerStore,
    SkillTrustStore,
)


@pytest.fixture
def make_trust_service(tmp_path):
    """Return a factory that builds `SkillTrustService` instances sharing one store.

    Args:
        tmp_path: Pytest-provided temporary directory fixture.

    Returns:
        A zero-argument callable that constructs a new `SkillTrustService`
        bound to the same on-disk `skills_dir`/`trust_dir`, so repeated calls
        simulate a fresh process re-reading persisted state.
    """
    skills_dir = tmp_path / "skills"
    trust_dir = tmp_path / "trust"
    skills_dir.mkdir(exist_ok=True)
    trust_dir.mkdir(exist_ok=True)

    def _make() -> SkillTrustService:
        return SkillTrustService(
            skills_dir=skills_dir,
            trust_store=SkillTrustStore(
                store_dir=trust_dir,
                marker_store=FileSkillTrustGenerationMarkerStore(
                    trust_dir / "marker.json"
                ),
            ),
        )

    return _make


@pytest.fixture
def trust_service_with_skill(make_trust_service):
    """Return a `SkillTrustService` with one on-disk demo skill (with a script).

    Args:
        make_trust_service: Factory fixture for building trust-service instances.

    Returns:
        A `(service, skill_name)` tuple where `skill_name` names a skill
        directory containing a `SKILL.md` and a `scripts/hello.py`.
    """
    service = make_trust_service()
    name = "demo-skill"
    skill_dir = service.skills_dir / name
    (skill_dir / "scripts").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: demo\n---\nbody\n", encoding="utf-8"
    )
    (skill_dir / "scripts" / "hello.py").write_text(
        "print('hello')", encoding="utf-8"
    )
    return service, name

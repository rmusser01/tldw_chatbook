"""Pixel-migu must work from both published artifact paths on a fresh profile."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

from Tests.Packaging import test_installed_distribution as distribution_support
from Tests.Packaging.test_installed_distribution import (
    _install_wheel_path,
    _private_child_env,
    _read_only_installed_tree,
)

built_distributions = distribution_support.built_distributions
sdist_wheel = distribution_support.sdist_wheel

pytestmark = pytest.mark.integration
RESOURCE_ROOTS = (
    "tldw_chatbook/assets/characters/pixel_migu",
    "tldw_chatbook/assets/persona_visual/pixel_migu",
)

INSTALLED_PROBE = r"""
import json, os
from pathlib import Path
import tldw_chatbook
assert Path(tldw_chatbook.__file__).resolve().is_relative_to(Path(os.environ["EXPECTED_TARGET"]))
from tldw_chatbook.config import seed_builtin_content
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Character_Chat.visual_identity import resolve_visual_identity
from tldw_chatbook.Character_Chat.local_character_persona_service import LocalCharacterPersonaService
from tldw_chatbook.Actor_Packs.repository import ActorPackRepository
from tldw_chatbook.Actor_Packs.persona_coordinator import PersonaActorPackCoordinator
from tldw_chatbook.DB.VisualIdentity_DB import VisualIdentityRepository
from tldw_chatbook.Persona_Visual.repository import PersonaVisualRepository
from tldw_chatbook.Persona_Visual.runtime import resolve_active_persona_visual
from tldw_chatbook.Persona_Visual.builtin_pixel_migu import ensure_builtin_pixel_migu_buddy
root = Path(os.environ["PIXEL_MIGU_PROFILE"]).resolve()
root.mkdir(mode=0o700)
db = CharactersRAGDB(root / "actors.db", client_id="installed-pixel-migu")
seed_builtin_content(db)
service = LocalCharacterPersonaService(db, persona_store_path=root / "personas.json")
coordinator = PersonaActorPackCoordinator(ActorPackRepository(db), service)
persona = ensure_builtin_pixel_migu_buddy(service, coordinator, profile_root=root)
assert persona["name"] == "pixel-migu"
card_id = persona["character_card_id"]
assert db.get_character_card_by_id(card_id)["name"] == "pixel-migu"
character = VisualIdentityRepository(db).get_active_actor_pack("character", card_id)
assert len(character["assets"]) == 18
for asset in character["assets"]:
    key = asset["expression_key"]
    result = resolve_visual_identity(db, actor_kind="character", actor_id=card_id, requested_state="idle", manual_expression_key=key)
    assert result.resolved_expression_key == key and result.image_bytes
repo = PersonaVisualRepository(db)
graph = repo.get_active_persona_pack(persona["id"])
assert len(graph.assets) == 64
for state in ("idle", "wake_armed", "listening", "thinking", "speaking", "tool_running", "approval_needed", "error", "offline"):
    for reduced in (False, True):
        result = resolve_active_persona_visual(repo, persona["id"], root, state, reduced_motion=reduced)
        assert result.frames, (state, result)
        assert result.resolved_state == state, (state, result)
before = (root / "personas.json").read_bytes()
seed_builtin_content(db)
service = LocalCharacterPersonaService(db, persona_store_path=root / "personas.json")
coordinator = PersonaActorPackCoordinator(ActorPackRepository(db), service)
ensure_builtin_pixel_migu_buddy(service, coordinator, profile_root=root)
assert before == (root / "personas.json").read_bytes()
assert len(service.list_persona_profiles()) == 1
assert repo.get_active_persona_pack(persona["id"]).identity == graph.identity
db.close_connection()
print("pixel-migu-installed-ok: 18 expressions, 9 Buddy states, restart preserved")
"""


def test_pixel_migu_assets_are_present_and_identical_in_both_artifacts(
    built_distributions,
):
    root = Path(__file__).resolve().parents[2]
    expected = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for prefix in RESOURCE_ROOTS
        for path in (root / prefix).rglob("*")
        if path.is_file()
    }
    assert len([key for key in expected if "/expressions/" in key]) == 18
    assert (
        len(
            [
                key
                for key in expected
                if "/persona_visual/" in key and key.endswith(".png")
            ]
        )
        == 64
    )
    with (
        zipfile.ZipFile(built_distributions.wheel) as wheel,
        tarfile.open(built_distributions.sdist) as sdist,
    ):
        sdist_root = sdist.getnames()[0].split("/")[0]
        for name, data in expected.items():
            assert (
                hashlib.sha256(wheel.read(name)).digest()
                == hashlib.sha256(data).digest()
            )
            assert sdist.extractfile(f"{sdist_root}/{name}").read() == data


@pytest.mark.parametrize("artifact", ["wheel", "sdist-wheel"])
def test_installed_pixel_migu_fresh_profile(
    built_distributions, sdist_wheel, tmp_path, artifact
):
    target, state, run = (tmp_path / name for name in ("target", "state", "run"))
    state.mkdir()
    run.mkdir()
    wheel = built_distributions.wheel if artifact == "wheel" else sdist_wheel.wheel
    _install_wheel_path(wheel, target)
    env = _private_child_env(state, target, built_distributions.source_root)
    env["PIXEL_MIGU_PROFILE"] = str(state / "pixel-migu")
    with _read_only_installed_tree(target):
        completed = subprocess.run(
            [sys.executable, "-c", INSTALLED_PROBE],
            cwd=run,
            env=env,
            text=True,
            capture_output=True,
            timeout=120,
        )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "pixel-migu-installed-ok" in completed.stdout


@pytest.mark.parametrize("archive_kind", ["wheel", "sdist"])
@pytest.mark.parametrize(
    "missing",
    [
        "tldw_chatbook/assets/characters/pixel_migu/expressions/love.png",
        "tldw_chatbook/assets/persona_visual/pixel_migu/speaking-4.png",
    ],
)
def test_release_checker_rejects_missing_pixel_migu_art(
    built_distributions, tmp_path, archive_kind, missing
):
    dist_dir = distribution_support._dist_dir_without(
        built_distributions,
        tmp_path,
        drop_from_wheel=[missing] if archive_kind == "wheel" else (),
        drop_from_sdist=[missing] if archive_kind == "sdist" else (),
    )
    result = distribution_support._run_manifest_checker(
        built_distributions, dist_dir, tmp_path
    )
    assert result.returncode == 1
    assert missing in result.stdout + result.stderr

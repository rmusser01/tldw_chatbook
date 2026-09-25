"""Built-in skills source (TASK-32954, spec §3.5)."""

import asyncio
import hashlib

import pytest

from tldw_chatbook.Skills_Interop import builtin_skills as bs
from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService

NAME = "character-creator"


def _svc(tmp_path, disabled=frozenset()):
    return LocalSkillsService(
        store_dir=tmp_path,
        builtin_disabled_loader=lambda: disabled,
        allow_untrusted_without_trust_service=True,
    )


class _RaisingTrust:
    """Any trust read on a built-in path is a failure."""

    def status_for_skill(self, *_a, **_k):
        raise AssertionError("trust read for a built-in")

    def ensure_skill_trusted(self, *_a, **_k):
        raise AssertionError("trust read for a built-in")

    def verify_skill_content(self, *_a, **_k):
        raise AssertionError("trust read for a built-in")

    def trusted_file_paths(self, *_a, **_k):
        raise AssertionError("trust read for a built-in")

    def current_fingerprint_digest(self, *_a, **_k):
        raise AssertionError("trust read for a built-in")


def test_digest_pin_matches_shipped_files():
    assert bs.BUILTIN_SKILL_DIGESTS, "no built-ins pinned"
    for name, files in bs.BUILTIN_SKILL_DIGESTS.items():
        shipped = {
            p.relative_to(bs.builtin_skill_dir(name)).as_posix()
            for p in bs.builtin_skill_dir(name).rglob("*")
            if p.is_file()
        }
        assert shipped == set(files), f"re-pin {name}: file set changed"
        for rel, digest in files.items():
            data = (bs.builtin_skill_dir(name) / rel).read_bytes()
            assert hashlib.sha256(data).hexdigest() == digest, f"re-pin {name}/{rel}"
    assert bs.verify_builtin_skill(NAME) is None


def test_skill_text_names_the_registered_tools_and_no_hub_allow_advice():
    text = (bs.builtin_skill_dir(NAME) / "SKILL.md").read_text(encoding="utf-8")
    for tool in ("character_search", "character_get", "character_save"):
        assert f"`{tool}`" in text
    assert "## Permissions" not in text
    assert "MCP hub" not in text


def test_builtin_in_context_without_trust(tmp_path):
    svc = _svc(tmp_path)
    svc.trust_service = pytest.fail  # any trust call would explode
    ctx = asyncio.run(svc.get_context())
    names = [s["name"] for s in ctx["available_skills"]]
    assert NAME in names
    summary = next(s for s in ctx["available_skills"] if s["name"] == NAME)
    assert summary["trust_status"] == "builtin"
    assert summary["source"] == "builtin"
    assert summary["trust_blocked"] is False


def test_builtin_visible_even_when_trust_service_is_locked(tmp_path):
    # No trust service and no escape hatch: user skills would be blocked, a
    # built-in is not (it never consults trust).
    svc = LocalSkillsService(store_dir=tmp_path, builtin_disabled_loader=frozenset)
    ctx = asyncio.run(svc.get_context())
    assert NAME in [s["name"] for s in ctx["available_skills"]]


def test_builtins_off_without_a_loader(tmp_path):
    # Only the app (which honours [skills] disabled_builtins) opts in; side
    # services built without a loader never show a built-in.
    svc = LocalSkillsService(
        store_dir=tmp_path, allow_untrusted_without_trust_service=True
    )
    assert NAME not in svc._visible_records()
    assert asyncio.run(svc.seed_builtin_skills())["seeded"] == []


def test_disabled_builtin_hidden(tmp_path):
    ctx = asyncio.run(_svc(tmp_path, frozenset({NAME})).get_context())
    assert NAME not in [s["name"] for s in ctx["available_skills"]]
    assert NAME not in [s["name"] for s in ctx["blocked_skills"]]


def test_tampered_builtin_is_blocked(tmp_path, monkeypatch):
    monkeypatch.setitem(bs.BUILTIN_SKILL_DIGESTS, NAME, {"SKILL.md": "0" * 64})
    svc = _svc(tmp_path)
    ctx = asyncio.run(svc.get_context())
    blocked = {s["name"]: s for s in ctx["blocked_skills"]}
    assert blocked[NAME]["trust_reason_code"] == "builtin_modified"
    assert blocked[NAME]["trust_blocked"] is True
    with pytest.raises(Exception, match="builtin_modified|blocked|trust"):
        asyncio.run(svc.execute_skill(NAME))


def test_get_skill_reads_package_content(tmp_path):
    skill = asyncio.run(_svc(tmp_path).get_skill(NAME))
    assert "# Character Creator" in skill["content"] and skill["source"] == "builtin"


def test_listed_with_source_marker(tmp_path):
    listing = asyncio.run(_svc(tmp_path).list_skills())
    row = next(s for s in listing["skills"] if s["name"] == NAME)
    assert row["source"] == "builtin"


def test_read_skill_file_and_execute_need_no_trust(tmp_path):
    svc = _svc(tmp_path)
    svc.trust_service = _RaisingTrust()
    body = asyncio.run(svc.read_skill_file(NAME, "SKILL.md"))
    assert "# Character Creator" in body["content"]
    result = asyncio.run(svc.execute_skill(NAME, args="a pirate"))
    assert "character_save" in result["rendered_prompt"]


def test_update_and_delete_refused_and_package_untouched(tmp_path):
    svc = _svc(tmp_path)
    before = (bs.builtin_skill_dir(NAME) / "SKILL.md").read_bytes()
    with pytest.raises(ValueError, match="read-only"):
        asyncio.run(svc.delete_skill(NAME))
    with pytest.raises(ValueError, match="read-only"):
        asyncio.run(svc.update_skill(NAME, content="x"))
    assert (bs.builtin_skill_dir(NAME) / "SKILL.md").read_bytes() == before
    assert bs.verify_builtin_skill(NAME) is None


def test_customize_copies_and_user_copy_overrides(tmp_path):
    svc = _svc(tmp_path)
    result = asyncio.run(svc.seed_builtin_skills())
    assert NAME in result["seeded"]
    assert (tmp_path / "skills" / NAME / "SKILL.md").exists()
    assert asyncio.run(svc.seed_builtin_skills())["seeded"] == []  # overwrite=False
    skill = asyncio.run(svc.get_skill(NAME))
    assert skill.get("source") != "builtin"
    # The user copy is an ordinary skill: editable and deletable, and the
    # package stays untouched when it is deleted.
    before = (bs.builtin_skill_dir(NAME) / "SKILL.md").read_bytes()
    asyncio.run(svc.update_skill(NAME, content=skill["content"] + "\nMore.\n"))
    assert asyncio.run(svc.delete_skill(NAME)) is True
    assert (bs.builtin_skill_dir(NAME) / "SKILL.md").read_bytes() == before
    # Deleting the override brings the built-in back.
    assert asyncio.run(svc.get_skill(NAME))["source"] == "builtin"


def test_user_override_record_is_flagged(tmp_path):
    svc = _svc(tmp_path)
    asyncio.run(svc.seed_builtin_skills())
    records = svc._visible_records()
    assert records[NAME].get("overrides_builtin") is True
    assert records[NAME].get("source") != "builtin"
    row = next(
        s for s in asyncio.run(svc.list_skills())["skills"] if s["name"] == NAME
    )
    assert row.get("overrides_builtin") is True


def test_non_builtin_user_record_is_not_flagged(tmp_path):
    svc = _svc(tmp_path)
    asyncio.run(
        svc.create_skill(
            name="my-skill", content="---\nname: my-skill\ndescription: d\n---\nBody\n"
        )
    )
    assert "overrides_builtin" not in svc._visible_records()["my-skill"]


def test_disabled_builtins_from_config():
    assert bs.disabled_builtins_from_config({}) == frozenset()
    assert bs.disabled_builtins_from_config(
        {"skills": {"disabled_builtins": [NAME]}}
    ) == frozenset({NAME})
    # A malformed value never raises and never disables anything.
    assert bs.disabled_builtins_from_config(
        {"skills": {"disabled_builtins": NAME}}
    ) == frozenset()


def test_app_wires_loader_to_in_memory_config(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    try:
        import tldw_chatbook.app as app_module
    except RecoveryRequired:  # ADR-126 machine state; CI imports the app cleanly
        pytest.skip("app import blocked by RecoveryRequired on this machine")

    monkeypatch.setattr(app_module, "get_user_data_dir", lambda: tmp_path)
    app = SimpleNamespace(
        _local_skills_service=None,
        _skills_scope_service=None,
        _local_skills_stack_inputs=(None, None),
        app_config={"skills": {"disabled_builtins": [NAME]}},
    )
    app_module.TldwCli._build_local_skills_stack(app)
    loader = app._local_skills_service._builtin_disabled_loader
    assert loader() == frozenset({NAME})
    app.app_config = {"skills": {}}  # live dict, re-read on every call
    assert loader() == frozenset()

"""The Console per-send skill capture sees built-ins without a trust read."""

from types import SimpleNamespace

from tldw_chatbook.Chat.console_chat_controller import capture_skill_context_maximum
from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService


class _RaisingTrust:
    def status_for_skill(self, *_a, **_k):
        raise AssertionError("trust read on the send path")

    def current_fingerprint_digest(self, *_a, **_k):
        raise AssertionError("trust read on the send path")

    def ensure_skill_trusted(self, *_a, **_k):
        raise AssertionError("trust read on the send path")


def test_capture_includes_builtin_without_trust_read(tmp_path):
    local = LocalSkillsService(
        store_dir=tmp_path,
        trust_service=_RaisingTrust(),
        builtin_disabled_loader=frozenset,
    )
    captured = capture_skill_context_maximum(SimpleNamespace(local_skills_service=local))
    names = [item["name"] for item in captured["available_skills"]]
    assert "character-creator" in names
    assert "- character-creator" in captured["context_text"]


def test_capture_honours_disabled_builtin(tmp_path):
    local = LocalSkillsService(
        store_dir=tmp_path,
        trust_service=_RaisingTrust(),
        builtin_disabled_loader=lambda: frozenset({"character-creator"}),
    )
    captured = capture_skill_context_maximum(SimpleNamespace(local_skills_service=local))
    assert "character-creator" not in [i["name"] for i in captured["available_skills"]]

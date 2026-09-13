"""Exact lazy public exports; recovery declarations do not bootstrap runtime."""

from importlib import import_module

_EXPORTS = {
    "LocalSkillsService": (".local_skills_service", "LocalSkillsService"),
    "default_local_skills_store_dir": (
        ".local_skills_service",
        "default_local_skills_store_dir",
    ),
    "ServerSkillsService": (".server_skills_service", "ServerSkillsService"),
    "SkillTrustService": (".skill_trust_service", "SkillTrustService"),
    "SkillTrustBlockedError": (".skill_trust_models", "SkillTrustBlockedError"),
    "SkillTrustStatus": (".skill_trust_models", "SkillTrustStatus"),
    "SkillsBackend": (".skills_scope_service", "SkillsBackend"),
    "SkillsScopeService": (".skills_scope_service", "SkillsScopeService"),
}

__all__ = [
    "LocalSkillsService",
    "ServerSkillsService",
    "SkillTrustBlockedError",
    "SkillTrustService",
    "SkillTrustStatus",
    "SkillsBackend",
    "SkillsScopeService",
    "default_local_skills_store_dir",
]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module, __name__), symbol)
    globals()[name] = value
    return value

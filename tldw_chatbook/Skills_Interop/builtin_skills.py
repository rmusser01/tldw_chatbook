"""Built-in skills shipped as package assets (TASK-32954, spec §3.5).

Built-ins are read-only, never pass through the trust service, and are
integrity-checked against the SHA-256 pins below on every load. A user skill
of the same name overrides a built-in (see ``LocalSkillsService``).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

BUILTIN_SKILLS_DIR = Path(__file__).resolve().parents[1] / "assets" / "skills"
#: Re-pin after editing a shipped file:
#: python -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" <file>
BUILTIN_SKILL_DIGESTS: dict[str, dict[str, str]] = {
    "character-creator": {
        "SKILL.md": "9357136502468abbac7f08c7b576ad46950b6b0ba86fb3e2fc3cd842cde14418"
    },
}
_SCRIPT_SUFFIXES = frozenset(
    {".py", ".sh", ".bash", ".js", ".ts", ".bat", ".ps1", ".rb", ".pl"}
)


def builtin_skill_dir(name: str) -> Path:
    """Return the package directory of a built-in skill."""
    return BUILTIN_SKILLS_DIR / name


def verify_builtin_skill(name: str) -> str | None:
    """Check a built-in against its pinned file set and digests.

    Returns:
        None when intact, else a block reason code (``builtin_missing`` or
        ``builtin_modified``). A built-in carrying any script is refused.
    """
    root = builtin_skill_dir(name)
    pinned = BUILTIN_SKILL_DIGESTS.get(name)
    if pinned is None or not root.is_dir():
        return "builtin_missing"
    try:
        shipped = {
            p.relative_to(root).as_posix()
            for p in root.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        }
        if shipped != set(pinned) or any(
            Path(p).suffix.lower() in _SCRIPT_SUFFIXES for p in shipped
        ):
            return "builtin_modified"
        for rel, digest in pinned.items():
            if hashlib.sha256((root / rel).read_bytes()).hexdigest() != digest:
                return "builtin_modified"
    except OSError:
        return "builtin_missing"
    return None


def builtin_skill_records(disabled: frozenset[str]) -> dict[str, dict[str, Any]]:
    """Index-shaped stubs for enabled built-ins (front matter parsed by the service)."""
    return {
        name: {"name": name, "source": "builtin"}
        for name in BUILTIN_SKILL_DIGESTS
        if name not in disabled
    }


def disabled_builtins_from_config(config: Mapping[str, Any] | None) -> frozenset[str]:
    """Read ``[skills] disabled_builtins`` from an in-memory config dict.

    A malformed value disables nothing rather than raising.
    """
    skills = (config or {}).get("skills")
    value = skills.get("disabled_builtins") if isinstance(skills, Mapping) else None
    if not isinstance(value, (list, tuple)):
        return frozenset()
    return frozenset(item for item in value if isinstance(item, str))

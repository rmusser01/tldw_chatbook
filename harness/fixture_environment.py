"""Original private app environment with only the CI source root forwarded."""
import os
from pathlib import Path


def child_environment(root: Path, private: Path, config: Path) -> dict[str, str]:
    env = {
        key: os.environ[key]
        for key in ("PATH", "TERM", "COLORTERM", "LANG", "LC_ALL", "LC_CTYPE")
        if key in os.environ
    }
    env["FIRSTSEND_SOURCE"] = str(root)
    env.update(
        HOME=str(private / "home"),
        XDG_CONFIG_HOME=str(private / "config"),
        XDG_CACHE_HOME=str(private / "cache"),
        XDG_DATA_HOME=str(private / "data"),
        TLDW_CONFIG_PATH=str(config),
        PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        PYTHONPATH=os.pathsep.join(
            (str(root), str(root / "packages/tldw_profile_core/src"))
        ),
    )
    return env

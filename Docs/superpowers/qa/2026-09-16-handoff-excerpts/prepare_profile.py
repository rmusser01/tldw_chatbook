"""Create a fresh no-secret profile. Run with the project's Python interpreter."""

import hashlib
import json
import sys
from pathlib import Path


def main():
    root = Path(sys.argv[1]).resolve()
    root.mkdir(exist_ok=False)
    (root / "data/db").mkdir(parents=True)
    (root / "config").mkdir()
    config = """[general]
default_tab = "chat"
users_name = "default_user"
log_level = "INFO"
[first_run]
setup_completed = true
[splash_screen]
enabled = false
[appearance]
reduce_motion = true
[model_catalog]
use_models_dev = false
auto_refresh_enabled = false
refresh_consent_recorded = true
[chat_defaults]
provider = "llama_cpp"
model = "review-local-model"
[api_settings.llama_cpp]
api_url = "http://127.0.0.1:1"
model = "review-local-model"
[paths]
data_dir = "PROFILE/data"
[database]
USER_DB_BASE_DIR = "PROFILE/data/db"
""".replace("PROFILE", str(root))
    for stem in (
        "chachanotes",
        "prompts",
        "media",
        "research",
        "writing",
        "library_collections",
        "workspaces",
        "evals",
        "rag_indexing",
        "subscriptions",
    ):
        config += f'{stem}_db_path = "{root}/data/db/{stem}.db"\n'
    (root / "config.toml").write_text(config)
    files = [
        Path.home() / ".config/tldw_cli" / name
        for name in (
            "config.toml",
            "ui_state.toml",
            "runtime_policy.json",
        )
    ]
    hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        if path.exists()
        else None
        for path in files
    }
    (root / "default-before.json").write_text(json.dumps(hashes, indent=2) + "\n")


if __name__ == "__main__":
    main()

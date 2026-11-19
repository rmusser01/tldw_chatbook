#!/usr/bin/env python3
"""Write or repair the config.toml of an rp-harness profile. Stdlib only (Python >= 3.11).

  profile_config.py init <profile-dir> [--extra FILE]
      Write a FRESH isolated config (overwrites): [general] users_name = "rp_review",
      [paths] data_dir, [first_run] setup_started/setup_completed = true (no wizard),
      [model_catalog] auto_refresh_enabled = false (no consent modal), every [database]
      *_db_path (the 14 keys of Backup_Recovery/profile_paths.py:DATABASE_PATHS) and
      USER_DB_BASE_DIR, all inside <profile-dir>; then appends FILE (e.g. mock_llm.toml).
      User data dir = <profile-dir>/data/rp_review/.

  profile_config.py repath <profile-dir>
      Rewrite ONLY the path keys above in the existing config, keeping every other line
      (sections the app or a test added, e.g. [roleplay.reader], survive untouched).

  profile_config.py set <config.toml> <table> <key> <string-value>
      Set one string key, preserving the rest of the file.

  profile_config.py get <config.toml> <dotted.key>
      Print a value; exit 1 when absent.

Every rewrite is verified before it is written: the new file must parse, and its parsed
content must equal the old parsed content with exactly the requested keys changed.
Writes are atomic (temp file + rename) and mode 0600.
"""
from __future__ import annotations

import copy
import json
import os
import re
import sys
import tempfile
import tomllib

USER_FOLDER = "rp_review"
DB_FILES = {
    "chachanotes_db_path": "tldw_chatbook_ChaChaNotes.db",
    "prompts_db_path": "tldw_chatbook_prompts.db",
    "media_db_path": "tldw_chatbook_media_v2.db",
    "library_collections_db_path": "tldw_chatbook_library_collections.db",
    "library_ingest_jobs_db_path": "tldw_chatbook_library_ingest_jobs.db",
    "workspaces_db_path": "tldw_chatbook_workspaces.db",
    "subscriptions_db_path": "tldw_chatbook_subscriptions.db",
    "evals_db_path": "evals.db",
    "rag_indexing_db_path": "rag_indexing.db",
    "notifications_db_path": "tldw_chatbook_notifications.db",
    "research_db_path": "tldw_chatbook_research.db",
    "writing_db_path": "tldw_chatbook_writing.db",
    "scheduled_tasks_db_path": "tldw_chatbook_scheduled_tasks.db",
    "tts_profiles_db_path": "tldw_chatbook_tts_profiles.db",
}

HEADER_RE = re.compile(r"^\s*\[\s*([^\[\]]+?)\s*\]\s*(?:#.*)?$")
ARRAY_HEADER_RE = re.compile(r"^\s*\[\[")
KEY_RE = re.compile(r"""^\s*(?:"([^"]+)"|'([^']+)'|([A-Za-z0-9_\-]+))\s*=""")


def q(value: str) -> str:
    """TOML basic string (JSON escaping is a valid subset)."""
    return json.dumps(value)


def path_keys(profile: str) -> list[tuple[str, str, str]]:
    p = os.path.abspath(profile)
    u = os.path.join(p, "data", USER_FOLDER)
    keys = [("paths", "data_dir", os.path.join(p, "data"))]
    keys += [("database", k, os.path.join(u, f)) for k, f in DB_FILES.items()]
    keys.append(("database", "USER_DB_BASE_DIR", os.path.join(p, "data") + "/"))
    return keys


def norm_table(name: str) -> str:
    return ".".join(part.strip().strip('"').strip("'") for part in name.split("."))


def set_keys(text: str, updates: list[tuple[str, str, str]]) -> str:
    """Line-based edit: replace `key = ...` inside [table], else insert it at the end of
    that table, else append a new [table]. Verified against tomllib afterwards."""
    for table, key, value in updates:
        lines = text.splitlines(keepends=True)
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        cur: str | None = None
        last_in_table: int | None = None
        replaced = False
        for i, line in enumerate(lines):
            if ARRAY_HEADER_RE.match(line):
                cur = "[[array]]"
                continue
            m = HEADER_RE.match(line)
            if m:
                cur = norm_table(m.group(1))
                if cur == table:
                    last_in_table = i
                continue
            if cur != table:
                continue
            if line.strip():
                last_in_table = i
            km = KEY_RE.match(line)
            if km and (km.group(1) or km.group(2) or km.group(3)) == key:
                lines[i] = f"{key} = {q(value)}\n"
                replaced = True
        if not replaced:
            if last_in_table is not None:
                lines.insert(last_in_table + 1, f"{key} = {q(value)}\n")
            else:
                if lines and lines[-1].strip():
                    lines.append("\n")
                lines.append(f"[{table}]\n{key} = {q(value)}\n")
        text = "".join(lines)
    return text


def verified(old_text: str, new_text: str, updates: list[tuple[str, str, str]]) -> str:
    old = tomllib.loads(old_text) if old_text.strip() else {}
    expected = copy.deepcopy(old)
    for table, key, value in updates:
        node = expected
        for part in table.split("."):
            node = node.setdefault(part, {})
        node[key] = value
    new = tomllib.loads(new_text)
    if new != expected:
        sys.exit("profile_config: refusing to write - the edited config does not parse back to "
                 "the original plus the requested keys (multi-line strings or dotted keys?)")
    return new_text


def atomic_write(path: str, text: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=".config.", suffix=".tmp", dir=d)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def cmd_init(profile: str, extra: str | None) -> str:
    p = os.path.abspath(profile)
    for sub in ("home", "xdg_config", "xdg_data", "xdg_cache", "xdg_state", "data", "cwd"):
        os.makedirs(os.path.join(p, sub), exist_ok=True)
    for sub in ("", "home", "data"):
        os.chmod(os.path.join(p, sub) if sub else p, 0o700)
    out = ['[general]', f'users_name = {q(USER_FOLDER)}', '',
           '[paths]', f'data_dir = {q(os.path.join(p, "data"))}', '',
           '[first_run]', 'setup_started = true', 'setup_completed = true', '',
           '[model_catalog]', 'auto_refresh_enabled = false', '', '[database]']
    out += [f"{k} = {q(v)}" for t, k, v in path_keys(p) if t == "database"]
    text = "\n".join(out) + "\n"
    if extra:
        with open(extra) as fh:
            text += "\n" + fh.read().rstrip("\n") + "\n"
    tomllib.loads(text)  # must parse
    cfg = os.path.join(p, "config.toml")
    atomic_write(cfg, text)
    return cfg


def cmd_repath(profile: str) -> str:
    cfg = os.path.join(os.path.abspath(profile), "config.toml")
    with open(cfg) as fh:
        old = fh.read()
    updates = path_keys(profile)
    atomic_write(cfg, verified(old, set_keys(old, updates), updates))
    return cfg


def cmd_set(cfg: str, table: str, key: str, value: str) -> str:
    with open(cfg) as fh:
        old = fh.read()
    updates = [(norm_table(table), key, value)]
    atomic_write(cfg, verified(old, set_keys(old, updates), updates))
    return cfg


def cmd_get(cfg: str, dotted: str) -> int:
    with open(cfg, "rb") as fh:
        node = tomllib.load(fh)
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return 1
        node = node[part]
    print(node if not isinstance(node, (dict, list)) else json.dumps(node))
    return 0


def main(argv: list[str]) -> int:
    if len(argv) >= 3 and argv[1] == "init":
        extra = None
        if len(argv) == 5 and argv[3] == "--extra":
            extra = argv[4] or None
        elif len(argv) != 3:
            print(__doc__, file=sys.stderr)
            return 2
        print(cmd_init(argv[2], extra))
        return 0
    if len(argv) == 3 and argv[1] == "repath":
        print(cmd_repath(argv[2]))
        return 0
    if len(argv) == 6 and argv[1] == "set":
        print(cmd_set(argv[2], argv[3], argv[4], argv[5]))
        return 0
    if len(argv) == 4 and argv[1] == "get":
        return cmd_get(argv[2], argv[3])
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))

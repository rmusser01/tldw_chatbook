#!/usr/bin/env python3
"""Add 50 overflow personas ("Zz Overflow Persona NN") to probe the persona list cap
(the default list call returns 100). Gap-round helper.

Run ONLY through seedrun.sh against a run copy (or the volume master):
  seedrun.sh "$HARNESS_STATE/runs/<socket>" persona_more.py
harness_guard refuses anything outside HARNESS_STATE, the real profile, and the golden /
golden.preseed.bak / empty masters.
"""
import sys

sys.dont_write_bytecode = True  # never leave __pycache__ next to the harness
import harness_guard  # same directory as this script

PROFILE = harness_guard.require_profile(allow_runs=True, allow_masters=("volume",))

from tldw_chatbook import config  # noqa: E402

harness_guard.check_app_paths(config, PROFILE)
db = config.get_chachanotes_db_lazy()
assert db is not None, "chachanotes db failed to open"
from tldw_chatbook.Backup_Recovery.chat_source_participants import build_persona_service  # noqa: E402

svc = build_persona_service(db)
for i in range(50):
    svc.create_persona_profile(dict(name=f"Zz Overflow Persona {i+1:02d}", description="cap test",
                                    system_prompt="cap test", personality_traits="cap"))
print("total personas (limit 1000):", len(svc.list_persona_profiles(limit=1000)))
print("default list (limit 100):", len(svc.list_persona_profiles()))

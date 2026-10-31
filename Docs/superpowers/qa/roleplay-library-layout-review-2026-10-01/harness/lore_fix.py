#!/usr/bin/env python3
"""Idempotent lore top-up after scale_seed.py: ensure at least 43 lore books exist, attach
the "Grand Codex" book to Dungeon Master Vex, and print the codex probe entries (1, 100,
200, 250) plus the disabled count. Gap-round helper.

Run ONLY through seedrun.sh against a run copy or the volume master:
  seedrun.sh "$HARNESS_STATE/runs/<socket>" lore_fix.py
harness_guard refuses anything outside HARNESS_STATE, the real profile, and the golden /
golden.preseed.bak / empty masters.
"""
import random
import sys

sys.dont_write_bytecode = True  # never leave __pycache__ next to the harness
import harness_guard  # same directory as this script

PROFILE = harness_guard.require_profile(allow_runs=True, allow_masters=("volume",))

from tldw_chatbook import config  # noqa: E402

harness_guard.check_app_paths(config, PROFILE)
db = config.get_chachanotes_db_lazy()
assert db is not None, "chachanotes db failed to open"
from tldw_chatbook.Character_Chat.world_book_manager import WorldBookManager
m = WorldBookManager(db)
rng = random.Random(7)
books = m.list_world_books(True)
names = {b["name"] for b in books}
print("existing books:", len(books))
PLACES = ["Thornhold", "Brackenfell", "Saltmarsh Abbey", "Vellmoor", "Cinderreach", "Gloamwater", "Highspire",
          "Rookwood", "Ambergate", "Mistral Quay", "Ironvein", "Hollowmere", "Starfall Ridge", "Duskhaven",
          "Wyrmrest", "Ebonhurst", "Coldharbour", "Lanternwick", "Foxglove Vale", "Grimsby Deep"]
j = 100
while len(m.list_world_books(True)) < 43:
    nm = f"{rng.choice(['Aetheria — Region','Campaign','Gazetteer:','Faction:','Bestiary:'])} {PLACES[j % len(PLACES)]} {j}"
    j += 1
    if nm in names: continue
    names.add(nm)
    bid = m.create_world_book(name=nm, description=f"Scale book {j}", enabled=(j % 5 != 0), scan_depth=3, token_budget=500)
    for e in range(rng.randint(0, 20)):
        m.create_world_book_entry(bid, [f"k{j}-{e}"], f"Fact {e} about {nm}.", insertion_order=e)
big = next(b for b in m.list_world_books(True) if b["name"].startswith("Grand Codex"))
print("big:", big["id"], len(m.get_world_book_entries(big["id"])))
vex = db.execute_query("SELECT id FROM character_cards WHERE name='Dungeon Master Vex' AND deleted=0").fetchone()[0]
m.attach_world_book_to_character(big["id"], int(vex))
ents = m.get_world_book_entries(big["id"])
for idx in (0, 99, 199, 249):
    e = ents[idx]; print(idx + 1, e["keys"], len(e["content"]), e["enabled"])
print("disabled:", sum(1 for e in ents if not e["enabled"]))
print("total books:", len(m.list_world_books(True)))
print([b["name"] for b in m.list_world_books(True)])

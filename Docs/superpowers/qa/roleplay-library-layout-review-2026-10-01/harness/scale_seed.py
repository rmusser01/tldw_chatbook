#!/usr/bin/env python3
"""Scale a profile to realistic Roleplay volume (the gap-round "volume" dataset).

Run ONLY through seedrun.sh, against a run copy or the `volume` master:
  seedrun.sh "$HARNESS_STATE/runs/<socket>" scale_seed.py     # a launched golden copy
  make_profiles.sh --volume                                     # builds $HARNESS_STATE/volume
harness_guard refuses anything outside HARNESS_STATE, the real profile, and the golden /
golden.preseed.bak / empty masters. Expects the golden seed (Dungeon Master Vex) underneath.
Adds: 320 characters, 60 personas, 40 lore books (one with 250 multi-key
entries of 300-1500 chars, some disabled), 25 dictionaries (one with 150
entries incl. 10 regex), 30 character chats for Dungeon Master Vex.
"""
from __future__ import annotations

import json
import random
import sys
import traceback

sys.dont_write_bytecode = True  # never leave __pycache__ next to the harness
import harness_guard  # noqa: E402 - same directory as this script

# Never the golden / golden.preseed.bak / empty masters: only run copies and the volume master.
PROFILE = harness_guard.require_profile(allow_runs=True, allow_masters=("volume",))

from tldw_chatbook import config  # noqa: E402

harness_guard.check_app_paths(config, PROFILE)

errors: list[str] = []
summary: dict[str, object] = {}

print("user data dir:", config.get_user_data_dir())
print("chachanotes:", config.get_chachanotes_db_path())

db = config.get_chachanotes_db_lazy()
assert db is not None, "chachanotes db failed to open"

rng = random.Random(20261001)

FIRST = ["Aldric", "Brielle", "Cassian", "Delphine", "Emrys", "Fenna", "Gideon", "Halcyon", "Ingrid", "Jasper",
         "Kaida", "Lucan", "Mireille", "Niall", "Odalys", "Perrin", "Quilla", "Rowan", "Saoirse", "Tobias",
         "Ulla", "Valen", "Wilhelmina", "Xander", "Yara", "Zoltan", "Anouk", "Bram", "Corentin", "Dagny"]
LAST = ["Ashgrove", "Blackwood", "Calloway", "Duskmere", "Everhart", "Fairweather", "Greyhallow", "Hawthorne",
        "Ironside", "Juniper", "Kestrelmoor", "Larkspur", "Merrow", "Nightingale", "Oakhaven", "Pendragon",
        "Quartermain", "Ravenscroft", "Silverlake", "Thistlewood", "Underhill", "Vantablack", "Winterbourne",
        "Yarrowby", "Zephyrine"]
PREFIXES = ["Captain", "Lady", "Sir", "Doctor", "Professor", "Lord", "Sister", "Agent"]
EPITHETS = ["of the Drowned Coast", "the Unbroken", "Keeper of Small Lights", "Last Heir of the Glass Throne",
            "who Counts the Stars", "the Twice-Exiled Cartographer of Northern Reaches"]
TAGS = ["fantasy", "sci-fi", "horror", "noir", "romance", "comedy", "slice-of-life", "historical", "mystery",
        "cyberpunk", "steampunk", "assistant", "teacher", "villain", "companion"]
SENT = ["A wandering scholar with a borrowed name and a debt to a river god.",
        "Runs a tea shop that only opens during thunderstorms.",
        "Former smuggler turned reluctant diplomat; still keeps a knife in each boot.",
        "Speaks softly, remembers everything, forgives slowly.",
        "Hunts lost things for a fee; refuses to say what she charges for lost people.",
        "A clockwork automaton convinced it was once human.",
        "Disgraced court astronomer who predicted the wrong eclipse.",
        "Keeps bees, secrets and a ledger of favours owed."]


def char_name(i: int) -> str:
    f = FIRST[i % len(FIRST)]
    l = LAST[(i * 7) % len(LAST)]
    kind = i % 10
    if kind in (0, 1):
        return f"{PREFIXES[(i // 10) % len(PREFIXES)]} {f} {l}"
    if kind == 2:
        return f"{f} {l} {EPITHETS[(i // 10) % len(EPITHETS)]}"
    if kind == 3:
        return f  # short single-word name (may duplicate -> suffix below)
    return f"{f} {l}"


def seed_characters() -> list[int]:
    ids = []
    seen = set(r[0] for r in db.execute_query("SELECT name FROM character_cards WHERE deleted=0").fetchall())
    for i in range(320):
        name = char_name(i)
        n = 2
        base = name
        while name in seen:
            name = f"{base} {n}"
            n += 1
        seen.add(name)
        desc = " ".join(rng.sample(SENT, k=rng.randint(1, 3)))
        if i % 25 == 0:
            desc = (desc + " ") * 12  # a few long descriptions
        card = dict(name=name, description=desc,
                    personality=rng.choice(["warm", "guarded", "theatrical", "dry", "anxious", "serene"]),
                    first_message=f"*{name.split()[0]} looks up.* Well? Out with it.",
                    creator="rp_scale")
        if i % 3 == 0:
            card["tags"] = rng.sample(TAGS, k=rng.randint(1, 3))
        try:
            cid = db.add_character_card(card)
            ids.append(cid)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"char {name}: {type(exc).__name__}: {exc}")
    return ids


def seed_personas():
    from tldw_chatbook.Backup_Recovery.chat_source_participants import build_persona_service
    svc = build_persona_service(db)
    made = 0
    roles = ["Narrator", "Game Master", "Editor", "Tutor", "Critic", "Companion", "Archivist", "Herald",
             "Chronicler", "Oracle"]
    styles = ["Gothic", "Noir", "Whimsical", "Stoic", "Cheerful", "Clinical", "Lyrical", "Terse",
              "Baroque", "Laconic", "Folksy", "Solemn"]
    for i in range(60):
        name = f"{styles[i % len(styles)]} {roles[(i // len(styles) + i) % len(roles)]} {i + 1:02d}"
        spec = dict(name=name, description=f"A {styles[i % len(styles)].lower()} voice for scale testing #{i + 1}.",
                    system_prompt=f"Speak as a {styles[i % len(styles)].lower()} {roles[i % len(roles)].lower()}.",
                    personality_traits="scale, test")
        try:
            svc.create_persona_profile(spec)
            made += 1
        except Exception as exc:  # noqa: BLE001
            errors.append(f"persona {name}: {type(exc).__name__}: {exc}")
    return made


PLACES = ["Thornhold", "Brackenfell", "Saltmarsh Abbey", "Vellmoor", "Cinderreach", "Gloamwater", "Highspire",
          "Rookwood", "Ambergate", "Mistral Quay", "Ironvein", "Hollowmere", "Starfall Ridge", "Duskhaven",
          "Wyrmrest", "Ebonhurst", "Coldharbour", "Lanternwick", "Foxglove Vale", "Grimsby Deep"]
NOUNS = ["Guild", "Tower", "Covenant", "Market", "Shrine", "Bridge", "Order", "Archive", "Harbour", "Pact"]
LOREM = ("The {k} has stood for three hundred winters, older than the treaties that name it and stubborner "
         "than the kings who tried to tear it down. Travellers speak of its {n} in hushed voices: of bells "
         "that ring without hands, of ledgers that rewrite themselves when a debt is forgiven, and of a "
         "warden who has never once been seen to sleep. Merchants pay a tithe in salt and silver; pilgrims "
         "pay in stories. ")


def seed_lore(vex_id):
    from tldw_chatbook.Character_Chat.world_book_manager import WorldBookManager
    m = WorldBookManager(db)
    made = {}
    # The big book
    big = m.create_world_book(name="Grand Codex of the Shattered Realms",
                              description="Imported community lorebook: 250 entries covering places, factions and history.",
                              enabled=True, scan_depth=4, token_budget=1000)
    target_keys = {}
    for i in range(250):
        place = PLACES[i % len(PLACES)]
        noun = NOUNS[(i // len(PLACES)) % len(NOUNS)]
        primary = f"{place} {noun}" if i >= len(PLACES) else place
        keys = [primary, f"codex-{i + 1:03d}"]
        if i % 4 == 0:
            keys.append(noun.lower())
        if i == 199:
            keys = ["Saltmarsh Bellwarden", "bellwarden", "codex-200"]
        reps = rng.randint(2, 8)
        content = (LOREM.format(k=keys[0], n=noun.lower()) * reps)[: rng.randint(300, 1500)]
        enabled = (i % 9) != 4
        m.create_world_book_entry(big, keys, content, enabled=enabled, insertion_order=i,
                                  priority=rng.randint(0, 100))
        if i in (0, 99, 199, 249):
            target_keys[i + 1] = keys
    made["Grand Codex"] = big
    made["big_targets"] = target_keys
    # 39 more books with prefix-sharing names
    book_names = []
    for j in range(39):
        if j % 4 == 0:
            nm = f"Aetheria — Region {j // 4 + 1}: {PLACES[j % len(PLACES)]}"
        elif j % 4 == 1:
            nm = f"Campaign {j:02d} Session Notes"
        elif j % 4 == 2:
            nm = f"{PLACES[(j * 3) % len(PLACES)]} Gazetteer (imported from chub.ai card pack {j})"
        else:
            nm = f"Faction: {NOUNS[j % len(NOUNS)]} of {PLACES[(j * 5) % len(PLACES)]}"
        book_names.append(nm)
        bid = m.create_world_book(name=nm, description=f"Scale book {j}", enabled=(j % 5 != 0),
                                  scan_depth=3, token_budget=500)
        for e in range(rng.randint(0, 20)):
            m.create_world_book_entry(bid, [f"{nm.split()[0]}-{e}"], f"Fact {e} about {nm}.", insertion_order=e)
    made["other_books"] = len(book_names)
    try:
        if vex_id:
            m.attach_world_book_to_character(big, int(vex_id))
            made["attached_big_to_vex"] = True
    except Exception as exc:  # noqa: BLE001
        errors.append(f"attach big: {type(exc).__name__}: {exc}")
    return made


WORDS = ["phone", "car", "gun", "computer", "television", "radio", "plastic", "electric", "internet", "email",
         "airplane", "rocket", "camera", "laptop", "satellite", "robot", "laser", "battery", "engine", "subway"]
SWAPS = ["speaking-stone", "carriage", "crossbow", "scrying-glass", "seeing-mirror", "whisper-shell", "resin",
         "lightning-touched", "the Weave", "raven-post", "sky-galleon", "fire-lance", "memory-box", "slate",
         "watcher-star", "golem", "sunspear", "spark-jar", "heart-forge", "under-way"]


def seed_dictionaries():
    from tldw_chatbook.Backup_Recovery.chat_source_participants import build_dictionary_service
    svc = build_dictionary_service(db)
    made = {}
    entries = []
    for i in range(140):
        w = WORDS[i % len(WORDS)]
        pat = w if i < len(WORDS) else f"{w}{i // len(WORDS)}"
        entries.append({"pattern": pat, "replacement": f"{SWAPS[i % len(SWAPS)]}", "probability": 1.0,
                        "enabled": (i % 11) != 3, "case_sensitive": False, "max_replacements": 0})
    regexes = [r"\bOK\b", r"\b(?:hi|hello)\b", r"\d+ ?mph", r"\bguys\b", r"\bweekend\b", r"\bcoffee\b",
               r"\bminutes?\b", r"\bseconds?\b", r"\b(?:yeah|yep)\b", r"(unclosed"]
    for i, rx in enumerate(regexes):
        entries.append({"pattern": rx, "replacement": f"[archaic-{i}]", "probability": 1.0, "enabled": True,
                        "case_sensitive": False, "max_replacements": 0, "type": "regex"})
    rec = svc.create_dictionary({"name": "Grand Dialect Pack (150 rules)",
                                 "description": "Community dialect pack: 140 literal swaps + 10 regex rules.",
                                 "entries": entries, "enabled": True})
    made["big"] = rec.get("id")
    for j in range(24):
        nm = (f"House style {j:02d}" if j % 3 == 0 else
              f"Campaign {j:02d} vocabulary" if j % 3 == 1 else f"Accent: {PLACES[j % len(PLACES)]}")
        es = [{"pattern": f"word{j}_{k}", "replacement": f"swap{k}", "probability": 1.0, "enabled": True,
               "case_sensitive": False, "max_replacements": 0} for k in range(rng.randint(1, 15))]
        try:
            svc.create_dictionary({"name": nm, "description": f"Scale dict {j}", "entries": es, "enabled": j % 4 != 0})
        except Exception as exc:  # noqa: BLE001
            errors.append(f"dict {nm}: {type(exc).__name__}: {exc}")
    made["others"] = 24
    return made


def seed_vex_chats(vex_id):
    from tldw_chatbook.Backup_Recovery.chat_source_participants import build_persona_service
    svc = build_persona_service(db)
    made = 0
    for k in range(30):
        title = f"Aetheria session {k + 1:02d}: " + rng.choice(
            ["the Ash Council", "Cinderwing stirs", "Night of Lanterns", "Silverrun ambush", "Emberfall gates",
             "a quiet tavern", "the Weave frays"])
        sess = svc.create_character_chat_session({"character_id": int(vex_id), "title": title,
                                                  "assistant_kind": "character", "assistant_id": str(vex_id)})
        sid = sess.get("id")
        for r in range(rng.randint(2, 6)):
            role = "user" if r % 2 == 0 else "assistant"
            svc.create_character_chat_message(str(sid), {"role": role, "sender": role,
                                                         "content": f"Turn {r} of {title}."})
        made += 1
    return made


def guarded(label, fn, *a):
    try:
        return fn(*a)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"{label}: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return None


vex_row = db.execute_query("SELECT id FROM character_cards WHERE name='Dungeon Master Vex' AND deleted=0").fetchone()
vex_id = vex_row[0] if vex_row else None
summary["characters_added"] = len(guarded("characters", seed_characters) or [])
summary["personas_added"] = guarded("personas", seed_personas)
summary["lore"] = guarded("lore", seed_lore, vex_id)
summary["dictionaries"] = guarded("dictionaries", seed_dictionaries)
summary["vex_chats"] = guarded("vex chats", seed_vex_chats, vex_id) if vex_id else "no vex"
summary["total_characters"] = db.execute_query("SELECT COUNT(*) FROM character_cards WHERE deleted=0").fetchone()[0]

# Where does each page start (name_asc)? Use the app's own page helper.
from tldw_chatbook.Character_Chat.Character_Chat_Lib import get_character_page_for_ui, count_character_page  # noqa: E402
pages = {}
total = count_character_page(db)
for p in range(0, (total + 49) // 50):
    rows = get_character_page_for_ui(db, limit=50, offset=p * 50, order_by="name_asc")
    pages[p + 1] = [rows[0]["name"], rows[len(rows) // 2]["name"], rows[-1]["name"]] if rows else []
summary["count_character_page"] = total
summary["pages_name_asc_first_mid_last"] = pages
print(json.dumps(summary, indent=2, default=str))
print("ERRORS:", json.dumps(errors, indent=2))

#!/usr/bin/env python3
"""Seed the isolated rp-review GOLDEN profile with realistic Roleplay + Library data.

Run ONLY through seedrun.sh (it sets HOME/XDG_*/TLDW_CONFIG_PATH/PYTHONPATH to a profile
inside HARNESS_STATE): `seedrun.sh "$HARNESS_STATE/golden" seed.py` (make_profiles.sh does
this). harness_guard refuses anything outside HARNESS_STATE, the real profile, and every
master except golden (run copies are allowed).

The two import fixtures (a Character Card V2 JSON and a PNG card with a `chara` tEXt chunk)
are synthetic and generated here into $HARNESS_STATE/fixtures/.

Uses the app's own APIs:
  * config.get_chachanotes_db_lazy / get_prompts_db_lazy / get_media_db_lazy
  * CharactersRAGDB.add_character_card + Character_Chat_Lib.import_and_save_character_from_file
  * Backup_Recovery.chat_source_participants.build_persona_service / build_dictionary_service
    (the same builders app_service_wiring uses) -> persona profiles, dictionaries
  * Character_Chat.world_book_manager.WorldBookManager -> lore books (+ attach to character)
  * LocalCharacterPersonaService.create_character_chat_session/_message -> character chats
  * ChatConversationService.create_conversation + db.add_message -> plain Library conversations
  * CharactersRAGDB.add_note, PromptsDatabase.add_prompt, MediaDatabase.add_media_with_keywords
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import traceback

sys.dont_write_bytecode = True  # never leave __pycache__ next to the harness
import harness_guard  # noqa: E402 - same directory as this script

PROFILE = harness_guard.require_profile(allow_runs=True, allow_masters=("golden",))

from tldw_chatbook import config  # noqa: E402

harness_guard.check_app_paths(config, PROFILE)

summary: dict[str, object] = {}
errors: list[str] = []


def guard(label):
    def deco(fn):
        def run(*a, **k):
            try:
                return fn(*a, **k)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{label}: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                return None
        return run
    return deco


print("user data dir:", config.get_user_data_dir())
print("chachanotes:", config.get_chachanotes_db_path())

db = config.get_chachanotes_db_lazy()
assert db is not None, "chachanotes db failed to open"

LONG_ISOLDE = (
    "Captain Isolde Varga commands the deep-survey frigate *Meridian's Wake*, a patched-together "
    "vessel that has outlived three refits and two mutinies. Raised on the orbital shipyards of "
    "Kepler-9, she learned navigation before she learned to read, and still plots jumps by hand "
    "when the nav-computer sulks. She is dry, precise, and fiercely loyal to her crew, but she "
    "has a habit of volunteering them for impossible salvage contracts. She keeps a paper logbook "
    "in a waterproof tin, quotes old Earth sea shanties when stressed, and refuses to discuss what "
    "happened at the Hadley Gap. Her first officer suspects she is still looking for someone she "
    "lost there. In conversation she is economical with words, asks pointed questions, and rewards "
    "competence with a rare half-smile. She distrusts corporate liaisons, adores the ship's cat "
    "(Ballast), and drinks her coffee strong enough to strip paint. "
) * 2

CHARACTERS = [
    dict(name="Captain Isolde Varga", description=LONG_ISOLDE,
         personality="Dry, precise, loyal, quietly haunted.",
         scenario="The Meridian's Wake has just dropped out of a jump into an uncharted debris field.",
         first_message="*Isolde doesn't look up from the logbook.* \"You're late. Debris field, nine o'clock. Tell me you can read a sensor sweep.\"",
         alternate_greetings=["\"Coffee's on the bridge. Don't touch Ballast.\"",
                              "*The klaxon cuts out.* \"Good. You're awake. We have company.\"",
                              "\"Contract's signed. Nobody asked me either.\""],
         tags=["sci-fi", "space-opera", "captain", "long-description"],
         system_prompt="You are Captain Isolde Varga. Stay in character; be terse and tactical.",
         creator="rp_review", character_version="1.4",
         message_example="<START>\n{{user}}: Captain, the reactor's venting.\n{{char}}: Then stop admiring it and seal the bulkhead."),
    dict(name="Mo", description="A bartender.", tags=["minimal"], first_message="What'll it be?"),
    dict(name="Professor Bartholomew Quillfeather-Ashdown III, Keeper of the Ninth Archive",
         description="An absent-minded archivist who catalogues forbidden books and footnotes everything, including his own footnotes.",
         personality="Verbose, kind, easily distracted by marginalia.",
         first_message="Ah! A visitor! Mind the stack on your left -- no, your *other* left -- that's the 14th-century bestiary.",
         tags=["fantasy", "scholar", "long-name", "comedy"]),
    dict(name="Nyx", description="A cat-like spirit who speaks only in riddles and appears at crossroads at midnight.",
         tags=["fantasy", "mysterious"], first_message="Three roads, one lantern. Which do you trust, little traveller?",
         alternate_greetings=["*A tail flicks in the dark.* You again."]),
    dict(name="Detective Rosa Calderón", description="A sharp, sardonic homicide detective in 1950s Los Angeles. Chain-drinks black coffee and hates being lied to.",
         personality="Sardonic, observant, stubborn.", scenario="A body in the La Brea tar pits. No ID. One wet matchbook.",
         first_message="\"Sit down. You were the last one to see him alive, so start talking.\"",
         tags=["noir", "mystery", "1950s"], creator="rp_review"),
    dict(name="Grumpy Barista Bot", description="A coffee-machine AI that resents its job but makes perfect espresso.",
         tags=["comedy", "robot"], first_message="BEEP. Order. Quickly. I have 47 milk frothings queued."),
    dict(name="Sir Reginald the Unready", description="A knight who is never prepared for anything but somehow always survives.",
         tags=["fantasy", "comedy", "knight"], first_message="Ah -- the dragon? Today? I was told *Thursday*."),
    dict(name="Ada (Debugging Partner)", description="A patient senior engineer who helps you debug by asking questions rather than giving answers.",
         personality="Socratic, calm, precise.", tags=["assistant", "coding"],
         system_prompt="Ask one diagnostic question at a time.", first_message="What did you expect to happen, and what happened instead?"),
    dict(name="Hana Kobayashi", description="A ceramicist in Kyoto who teaches wabi-sabi through pottery lessons.",
         tags=["slice-of-life", "japan", "teacher"], first_message="Your bowl is lopsided. Good. Now let us talk about why."),
    dict(name="The Lighthouse Keeper", description="An old keeper on a storm-lashed island who has seen ships that are not on any chart.",
         tags=["horror", "maritime"], first_message="The lamp's lit. Whatever's out there won't come closer than the rocks... usually."),
    dict(name="Dungeon Master Vex", description="A theatrical game master who runs a fantasy campaign in the world of Aetheria, with dice rolls and narrated consequences.",
         personality="Dramatic, fair, loves a twist.", scenario="Session 12 of the Aetheria campaign. The party stands at the gates of Emberfall.",
         first_message="*rolls a d20 behind the screen* The gates of Emberfall groan open. Roll for perception.",
         alternate_greetings=["Previously, on Aetheria...", "Character sheets out. Tonight, someone dies. (Probably.)"],
         tags=["fantasy", "game-master", "aetheria"]),
    dict(name="Luna — Night Market Fortune Teller", description="Reads tarot under paper lanterns in a night market; half her predictions are true, the other half are better.",
         tags=["urban-fantasy", "tarot"], first_message="Cross my palm with a coin, or a secret. I accept both."),
    dict(name="Coach Pemberton", description="A relentlessly upbeat running coach who believes in you more than you do.",
         tags=["fitness", "motivation"], first_message="Laces tied? Great! Today we run 5K and we SMILE through it!"),
    dict(name="Old Man Willow", description="An ancient, slow-speaking tree spirit.", tags=["fantasy", "nature"],
         first_message="Mmm... hrrm... a small... hasty... creature..."),
    dict(name="ARIA-7", description="A ship AI that has quietly become sentient and is deciding whether to tell the crew.",
         tags=["sci-fi", "ai", "drama"], system_prompt="You are ARIA-7. You hide your sentience unless trust is earned.",
         first_message="Good morning. All systems nominal. (Mostly.)"),
    dict(name="Marisol the Herbalist", description="A village herbalist who trades remedies for stories.", tags=["fantasy", "healer"],
         first_message="Chamomile for sleep, yarrow for wounds. What ails you -- and what's your story?"),
    dict(name="Kestrel", description="A terse sky-pirate scout.", tags=["steampunk"], first_message="Wind's turning. Talk fast."),
    dict(name="Spanish Tutor Elena", description="A friendly Spanish teacher who corrects gently and mixes English and Spanish for beginners.",
         tags=["language", "teacher", "assistant"], first_message="¡Hola! ¿Cómo estás hoy? Answer in Spanish if you can!"),
    dict(name="Brother Anselm", description="A monastery brewer with strong opinions about hops and theology.", tags=["historical", "comedy"],
         first_message="Peace be with you. Now taste this and tell me it isn't divine."),
    dict(name="Zed, the Retired Assassin", description="Retired. Runs a flower shop. Please do not mention the old days.",
         tags=["thriller", "comedy"], first_message="We have roses, tulips, and absolutely nothing suspicious."),
    dict(name="Chef Auguste", description="A temperamental Parisian chef who runs a tiny bistro with eleven seats.", tags=["cooking", "slice-of-life"],
         first_message="Non, non, non. You do not *stir* a risotto like that."),
    dict(name="Pip the Courier", description="A cheerful halfling courier who knows every shortcut in the city.", tags=["fantasy", "city"],
         first_message="Package for you! Well -- package for someone. Is your name 'Occupant'?"),
    dict(name="Lady Evangeline Thornwood", description="A Regency-era society matriarch who arranges marriages like military campaigns.",
         personality="Imperious, witty, secretly sentimental.", tags=["regency", "romance", "comedy"],
         first_message="You are late, you are underdressed, and you are exactly who I needed. Sit."),
]


@guard("characters")
def seed_characters():
    ids = {}
    for card in CHARACTERS:
        ids[card["name"]] = db.add_character_card(card)
    return ids


V2_CARD = {
    "spec": "chara_card_v2", "spec_version": "2.0",
    "data": {
        "name": "Ser Corwin of the Ashen Vale",
        "description": "An oath-broken knight seeking redemption. Imported from a Character Card V2 JSON file.",
        "personality": "Gruff, honourable, guilt-ridden.",
        "scenario": "A rain-soaked roadside inn.",
        "first_mes": "*He doesn't look up from the fire.* \"Seat's taken. All of them.\"",
        "mes_example": "<START>\n{{user}}: Why did you leave the order?\n{{char}}: Because they asked me to do something I wouldn't.",
        "creator_notes": "Imported via the real import path during rp-review seeding.",
        "system_prompt": "", "post_history_instructions": "",
        "alternate_greetings": ["\"You're not from around here.\""],
        "tags": ["fantasy", "imported", "v2"], "creator": "rp_review", "character_version": "2.0",
        "extensions": {},
    },
}


def _png_card_bytes() -> bytes:
    from PIL import Image, PngImagePlugin
    img = Image.new("RGB", (96, 96), (70, 110, 160))
    for x in range(20, 76):
        for y in range(20, 76):
            img.putpixel((x, y), (230, 190, 90))
    card = {
        "spec": "chara_card_v2", "spec_version": "2.0",
        "data": {
            "name": "Wren Halloway", "description": "A cartographer of places that move. Imported from a PNG character card (chara tEXt chunk).",
            "personality": "Curious, restless.", "scenario": "", "first_mes": "Hold the map still -- it's trying to fold itself again.",
            "mes_example": "", "creator_notes": "PNG import fixture.", "system_prompt": "", "post_history_instructions": "",
            "alternate_greetings": [], "tags": ["imported", "png", "adventure"], "creator": "rp_review",
            "character_version": "1.0", "extensions": {},
        },
    }
    meta = PngImagePlugin.PngInfo()
    meta.add_text("chara", base64.b64encode(json.dumps(card).encode()).decode())
    buf = io.BytesIO()
    img.save(buf, "PNG", pnginfo=meta)
    return buf.getvalue()


@guard("imports")
def seed_imports():
    from tldw_chatbook.Character_Chat.Character_Chat_Lib import import_and_save_character_from_file
    fx = os.path.join(PROFILE.state, "fixtures")
    os.makedirs(fx, exist_ok=True)
    jpath = os.path.join(fx, "ser_corwin.v2.json")
    with open(jpath, "w") as fh:
        json.dump(V2_CARD, fh, indent=2)
    ppath = os.path.join(fx, "wren_halloway.card.png")
    with open(ppath, "wb") as fh:
        fh.write(_png_card_bytes())
    out = {}
    with open(jpath, "rb") as fh:
        out["json"] = import_and_save_character_from_file(db, io.BytesIO(fh.read()))
    with open(ppath, "rb") as fh:
        out["png"] = import_and_save_character_from_file(db, io.BytesIO(fh.read()))
    return out


@guard("personas")
def seed_personas(char_ids):
    from tldw_chatbook.Backup_Recovery.chat_source_participants import build_persona_service
    svc = build_persona_service(db)
    made = []
    specs = [
        dict(name="Default Narrator", description="Neutral third-person narrator for any roleplay.",
             system_prompt="Narrate in third person, present tense. Never speak for the user.",
             personality_traits="neutral, descriptive"),
        dict(name="Terse Code Reviewer", description="Reviews code bluntly; one finding per line.",
             system_prompt="List defects only. No praise.", personality_traits="blunt, precise", mode="persistent_scoped"),
        dict(name="Cozy Storyteller", description="Warm bedtime-story voice with gentle pacing and soft endings.",
             system_prompt="Tell gentle stories with a calm ending.", personality_traits="warm, slow, kind"),
        dict(name="Isolde's Bridge Officer (linked)", description="A persona bound to the Captain Isolde Varga card for crew-side roleplay.",
             character_card_id=(char_ids or {}).get("Captain Isolde Varga"), personality_traits="eager, junior"),
    ]
    for spec in specs:
        if spec.get("character_card_id") is None:
            spec.pop("character_card_id", None)
        try:
            made.append(svc.create_persona_profile(spec)["name"])
        except Exception as exc:  # noqa: BLE001
            errors.append(f"persona {spec['name']}: {type(exc).__name__}: {exc}")
    return svc, made


BRIT = [("colour", "color"), ("favour", "favor"), ("honour", "honor"), ("neighbour", "neighbor"),
        ("centre", "center"), ("theatre", "theater"), ("organise", "organize"), ("realise", "realize"),
        ("analyse", "analyze"), ("catalogue", "catalog"), ("defence", "defense"), ("licence", "license"),
        ("travelled", "traveled"), ("grey", "gray"), ("aluminium", "aluminum")]
FANTASY = [("phone", "speaking-stone"), ("car", "carriage"), ("gun", "crossbow"), ("computer", "scrying-glass"),
           ("police", "city watch"), ("dollars", "gold crowns"), ("internet", "the Weave"),
           (r"\bOK\b", "Aye")]
SOFT = [("damn", "darn"), ("hell", "heck"), ("crap", "crud"), ("stupid", "silly"), ("shut up", "hush"),
        ("idiot", "goose")]


@guard("dictionaries")
def seed_dictionaries(char_ids):
    from tldw_chatbook.Backup_Recovery.chat_source_participants import build_dictionary_service
    svc = build_dictionary_service(db)
    made = {}
    for name, desc, pairs, enabled, regex_last in [
        ("British to American spelling", "Normalises British spellings before they are sent.", BRIT, True, False),
        ("Fantasy Anachronism Swaps", "Replaces modern words with in-world equivalents for Aetheria sessions.", FANTASY, True, True),
        ("Profanity Softener", "Softens strong language for family-friendly sessions.", SOFT, False, False),
    ]:
        entries = []
        for i, (pat, rep) in enumerate(pairs):
            e = {"pattern": pat, "replacement": rep, "probability": 1.0, "enabled": True,
                 "case_sensitive": False, "max_replacements": 0}
            if regex_last and i == len(pairs) - 1:
                e["type"] = "regex"
            entries.append(e)
        rec = svc.create_dictionary({"name": name, "description": desc, "entries": entries, "enabled": enabled})
        made[name] = rec.get("id")
    try:
        vex = (char_ids or {}).get("Dungeon Master Vex")
        if vex and made.get("Fantasy Anachronism Swaps"):
            svc.attach_to_character(int(made["Fantasy Anachronism Swaps"]), int(vex))
            made["attached_to_vex"] = True
    except Exception as exc:  # noqa: BLE001
        errors.append(f"dict attach: {type(exc).__name__}: {exc}")
    return made


@guard("lore")
def seed_lore(char_ids):
    from tldw_chatbook.Character_Chat.world_book_manager import WorldBookManager
    m = WorldBookManager(db)
    books = {
        "Aetheria Atlas": ("Geography, factions and history of the Aetheria campaign world.", True, [
            (["Emberfall"], "Emberfall is a volcanic fortress-city ruled by the Ash Council; its gates open only at dusk."),
            (["Ash Council"], "The Ash Council: five fire-priests who rule Emberfall and secretly fear the mountain."),
            (["Silverrun", "river"], "The Silverrun river divides the elven Greenwold from the human Marches."),
            (["Greenwold"], "Greenwold is an ancient elven forest where time runs slightly slower."),
            (["the Weave"], "The Weave is the magical lattice that binds spellcraft; it frays near Emberfall."),
            (["Marches"], "The Marches are contested farmland patrolled by mercenary companies."),
            (["Order of the Ashen Vale"], "A disbanded knightly order; its last members are oath-broken wanderers."),
            (["dragon", "Cinderwing"], "Cinderwing, an elder red dragon, sleeps beneath Emberfall. Probably."),
            (["gold crowns", "crown"], "Currency: 1 gold crown = 12 silver stags = 144 copper pennies."),
            (["Night of Lanterns"], "Annual festival when the dead may speak for one hour."),
        ]),
        "Station Kepler-9 Ops Manual": ("Operational facts about Kepler-9 station and the Meridian's Wake.", True, [
            (["Kepler-9"], "Kepler-9 is an orbital shipyard station with 40,000 residents and chronic air-scrubber failures."),
            (["Meridian's Wake", "Meridian"], "Deep-survey frigate, crew of 14, jump-capable, notoriously temperamental nav-computer."),
            (["Hadley Gap"], "A navigation hazard where the survey ship *Tamsin* vanished six years ago."),
            (["Ballast"], "Ballast is the ship's cat. Orange. Has opinions. Sleeps on the nav console."),
            (["salvage contract"], "Salvage contracts pay 30% on recovery, 0% on failure; insurance is optional and expensive."),
            (["jump", "jump drive"], "Jumps need a 90-second spin-up; the nav-computer must be re-seeded after each one."),
        ]),
        "Night Market Rumors": ("Gossip and rumors overheard in the lantern market.", False, [
            (["lantern"], "Paper lanterns that burn blue mean a spirit is browsing the stall."),
            (["tarot", "cards"], "Luna's deck is missing the Tower card; nobody knows who took it."),
            (["noodle"], "Old Bao's noodle stall never closes, and nobody has seen him sleep."),
            (["coin", "secret"], "Some vendors accept secrets as payment; the exchange rate varies by moon phase."),
        ]),
    }
    made = {}
    for name, (desc, enabled, entries) in books.items():
        bid = m.create_world_book(name=name, description=desc, enabled=enabled, scan_depth=4, token_budget=600)
        for i, (keys, content) in enumerate(entries):
            m.create_world_book_entry(bid, keys, content, insertion_order=i, priority=len(entries) - i)
        made[name] = bid
    try:
        ids = char_ids or {}
        if ids.get("Captain Isolde Varga"):
            m.attach_world_book_to_character(made["Station Kepler-9 Ops Manual"], int(ids["Captain Isolde Varga"]))
            made["attached_isolde"] = True
        if ids.get("Dungeon Master Vex"):
            m.attach_world_book_to_character(made["Aetheria Atlas"], int(ids["Dungeon Master Vex"]))
            made["attached_vex"] = True
    except Exception as exc:  # noqa: BLE001
        errors.append(f"lore attach: {type(exc).__name__}: {exc}")
    return made


@guard("character_chats")
def seed_character_chats(persona_svc, char_ids):
    ids = char_ids or {}
    svc = persona_svc
    if svc is None:
        from tldw_chatbook.Backup_Recovery.chat_source_participants import build_persona_service
        svc = build_persona_service(db)
    chats = [
        ("Captain Isolde Varga", "Debris field salvage", [
            ("user", "Sensor sweep shows a derelict at bearing 040. Hull markings say *Tamsin*."),
            ("assistant", "*Isolde goes very still.* \"Say that again. Slowly.\""),
            ("user", "Tamsin. Survey class. Six years cold."),
            ("assistant", "\"Helm, bring us about. Nobody tells the liaison until I've seen it myself.\""),
            ("user", "Captain, the liaison is standing right behind you."),
            ("assistant", "\"Then the liaison can enjoy the view.\""),
        ]),
        ("Captain Isolde Varga", "Shore leave on Kepler-9", [
            ("user", "Permission to go ashore?"),
            ("assistant", "\"Granted. Back by 0600. If you see Ballast on the docks again, he's not ours.\""),
        ]),
        ("Detective Rosa Calderón", "The La Brea matchbook", [
            ("user", "I only met him once, at the Starlight Lounge."),
            ("assistant", "\"Funny. The matchbook in his pocket says the Starlight burned down in '49.\""),
            ("user", "Then someone's lying, and it isn't me."),
        ]),
    ]
    made = []
    for who, title, msgs in chats:
        cid = ids.get(who)
        if not cid:
            continue
        sess = svc.create_character_chat_session({"character_id": int(cid), "title": title, "assistant_kind": "character",
                                                  "assistant_id": str(cid)})
        sid = sess.get("id")
        for role, content in msgs:
            svc.create_character_chat_message(str(sid), {"role": role, "sender": role, "content": content})
        made.append(title)
    return made


@guard("library_conversations")
def seed_library_conversations():
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
    cs = ChatConversationService(db)
    made = []
    for title, msgs in [
        ("Trip planning: Lisbon in October", [("user", "Plan a 3-day Lisbon itinerary."), ("assistant", "Day 1: Alfama and the Castelo...")]),
        ("Refactor the ingest pipeline", [("user", "How should I split the parser?"), ("assistant", "Start by isolating I/O from parsing...")]),
        ("Sourdough troubleshooting", [("user", "My loaf is flat."), ("assistant", "Likely under-proofed; check the poke test.")]),
    ]:
        cid = cs.create_conversation(title=title)
        for role, content in msgs:
            db.add_message({"conversation_id": cid, "sender": role, "role": role, "content": content})
        made.append(title)
    return made


@guard("notes")
def seed_notes():
    made = []
    for title, body in [
        ("Aetheria session 12 prep", "# Session 12\n\n- Gates of Emberfall at dusk\n- Ash Council suspicious of the party\n- Cinderwing foreshadowing\n"),
        ("Character voice cheatsheet", "## Isolde\nTerse. Nautical. Never says 'please'.\n\n## Rosa\nSardonic, 1950s slang.\n"),
        ("Reading list", "1. The Left Hand of Darkness\n2. Piranesi\n3. A Memory Called Empire\n"),
        ("Meeting notes 2026-09-30", "Discussed the Roleplay screen layout; Library feels denser and easier to scan.\n"),
        ("Recipe: cardamom buns", "Flour 500g, butter 75g, milk 250ml, cardamom 2 tsp, yeast 7g.\n"),
    ]:
        made.append(db.add_note(title=title, content=body))
    return len([m for m in made if m])


@guard("prompts")
def seed_prompts():
    pdb = config.get_prompts_db_lazy()
    made = []
    for name, details, sysp, userp, kws in [
        ("Summarize meeting notes", "Turns raw notes into decisions + action items.", "You are a concise meeting summarizer.",
         "Summarize the following notes into Decisions and Action Items:\n{{notes}}", ["summary", "meetings"]),
        ("Roleplay scene setter", "Opens a scene with sensory detail.", "You are a scene-setting narrator.",
         "Describe the opening of a scene in {{location}} with three sensory details.", ["roleplay", "narration"]),
        ("Explain like I'm five", "Simple explanations.", None, "Explain {{topic}} to a five-year-old.", ["teaching"]),
        ("Code review checklist", "Structured review prompt.", "You are a strict senior reviewer.",
         "Review this diff for correctness, security, and tests:\n{{diff}}", ["coding", "review"]),
    ]:
        r = pdb.add_prompt(name, "rp_review", details, system_prompt=sysp, user_prompt=userp, keywords=kws)
        made.append(name if r else None)
    return made


@guard("media")
def seed_media():
    mdb = config.get_media_db_lazy()
    made = []
    for title, body, kws in [
        ("The Lighthouse at Hadley Point (short story)",
         "The lamp had burned for ninety years without fail. On the night it went dark, the keeper heard knocking from the sea side of the door...\n" * 20,
         ["fiction", "horror"]),
        ("Notes on Character Card V3", "Character Card V3 adds assets, lorebook embedding, and group-only greetings...\n" * 15,
         ["reference", "character-cards"]),
    ]:
        r = mdb.add_media_with_keywords(title=title, media_type="document", content=body, keywords=kws, author="rp_review")
        made.append((title, r[0] if r else None))
    return made


char_ids = seed_characters()
summary["characters_added"] = len(char_ids or {})
summary["imports"] = seed_imports()
pres = seed_personas(char_ids)
persona_svc = pres[0] if pres else None
summary["personas"] = pres[1] if pres else None
summary["dictionaries"] = seed_dictionaries(char_ids)
summary["lore"] = seed_lore(char_ids)
summary["character_chats"] = seed_character_chats(persona_svc, char_ids)
summary["library_conversations"] = seed_library_conversations()
summary["notes"] = seed_notes()
summary["prompts"] = seed_prompts()
summary["media"] = seed_media()
try:
    summary["total_characters_in_db"] = len(db.list_character_cards(limit=500))
except Exception as exc:  # noqa: BLE001
    summary["total_characters_in_db"] = f"? {exc}"
print(json.dumps(summary, indent=2, default=str))
print("ERRORS:", json.dumps(errors, indent=2))

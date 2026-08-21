---
id: TASK-19577
title: >-
  User Guide drift — wrong composer buttons, retired screens, inverted persona
  semantics, and seven unresolvable stamp SHAs
status: To Do
assignee: []
created_date: '2026-08-21 20:23'
labels:
  - documentation
  - user-guide
priority: medium
dependencies: []
---

## Description

Source: 2026-08-21 holistic review, Lane 6 (UX coherence / honesty) — its
systemic doc-drift finding. Every item below was **re-verified at this branch
base**, and several of the review's own numbers were wrong in the direction of
*understating* the problem; the corrected figures are used here.

**1. `console.md` describes three composer buttons that do not exist.**
`console.md:81-84` lists **Send, Mic, Attach, Save**, with Stop "between Send
and Mic". The real composer (`Widgets/Console/console_composer_bar.py:5061-5257`)
yields `Composer ▾`, `Menu`, draft, `Send`, `Dictate`, hidden `Stop`, hidden
`✕`.
- **Mic**: zero occurrences as a live label — the state map at
  `console_composer_bar.py:1792-1796` is `Dictate` / `Dictate…` / `Dictating`.
  (One stale in-code string survives at
  `UI/Console_Modules/dictation.py:1508`: "Limit reached — press Mic to
  continue." — that is a **product** string to fix, not a doc fix.)
- **Attach** and **Save**: moved into the composer Menu, per the comment at
  `console_composer_bar.py:5236-5241`.
- The page **contradicts itself**: `console.md:35-37` already says
  Save-as-Chatbook lives in the Menu, 46 lines before the stale list.
- Stamp integrity failure: `console.md:478-479` claims *"This page's layout
  tour re-verified against that build"* (dev @ `b6036515e`, 2026-08-18), but the
  relabel landed in `7dbbc401b` on **2026-08-07** — eleven days *before* the
  claimed re-verification. The stamp asserts a check that cannot have happened.

**2. The attachments page reprints the exact glyphs the code deliberately
replaced.** `console/attachments-images-voice.md` prints `●` (lines 26, 117,
121, 151, 152) and `◌` (120, 122, 152). `Chat/console_glyphs.py:26-44` replaced
both **specifically to remove collisions** — `GLYPH_VOICE_RECORDING = "◉"`
(because `●` is now agent-run-only) and `GLYPH_VOICE_WORKING = "◐"` (because
`◌` is now temporary-session-only). Live labels: `"◐ Transcribing…"`,
`"◐ Preparing microphone…"`, timer head `◉`. The page also reprints the retired
labels **Mic / Mic… / Rec ● / Text…**. Its only stamp predates the change.

**3. `library/search-and-rag.md:411-421` sends users to a retired screen.**
Its "Not the same screen as 'Search'" section describes a separate Search
screen with **Search / Saved / History / Maintenance** tabs and tells users to
go to Maintenance to backfill a semantic index. `screen_registry.py:203-211`
retires the route (`"search": "library"`), and the palette entry
(`app.py:1290-1291`) is a deep link **into Library**. `library.md:31-33`
already says so. The index-backfill advice points at a tab that does not exist.

**4. Roleplay naming drift — 15 hits across 7 files, including two that are
worse than naming.** `shell_destinations.py:84-91` retired the long form
("'Roleplay' is the one public name everywhere"), but
`"Roleplay & Chat Dictionaries"` still appears 15 times.
- **An impossible palette entry.** `roleplay-chat-dictionaries.md:23` tells
  users to type `"Tab Navigation: Switch to Roleplay & Chat Dictionaries"`.
  `app.py:1005` builds that string from the destination's `full_label`, which is
  **"Roleplay"** — the string the doc instructs users to type **cannot exist**.
- **An inversion of the app's persona semantics.**
  `roleplay-chat-dictionaries.md:55` prints, as a verbatim tooltip,
  `"Personas — assistant profiles for roleplay and chat"`. The real tooltip
  (`personas_screen.py:345`) is `"Personas — who you play in the chat."` The
  doc flips the persona from the **user's** side to the **AI's** side. Same
  inversion at `roleplay-chat-dictionaries.md:6-7` and
  `characters-and-personas.md:7`. The code's own copy corroborates the correct
  reading (`shell_destinations.py:88,90` — "user profiles", "user profile
  context"). This matters beyond wording: it is the standing project rule that
  `{{user}}` is the human and `{{char}}` is the character.
  (The macro table at `characters-and-personas.md:258-263` gets it right — the
  inversion is confined to the persona-concept prose.)

**5. `library.md` lists `research` as retired when it is a live route.**
`library.md:26-28` and `:305-307` name research among "the six retired
screens". `screen_registry.py:157-159` registers it as a real route, and
`:201-202` says explicitly: *"'research' is a REAL screen route again
(task-16322, ADR-068)… deliberately NOT an alias."*
**The review additionally claimed that screen "crashes on first use". That is
REPORTED-BUT-UNCONFIRMED and the evidence points the other way** — this
filing's verification ran `Tests/UI/test_research_screen.py` (**43 passed**) and
two headless mount probes, both of which succeeded. The probes used a stub
`app_instance`, so a crash depending on real `TldwCli` wiring would not have
been caught. **The first step for that sub-claim is to confirm it against the
real app; do not fix a crash that has not been reproduced.**

**6. `settings.md` never mentions Video Gen, while `console/video.md` sends
users there.** `settings.md` is 710 lines, stamped 2026-08-20, and
`grep -ci video` returns **0**. `console/video.md:11,41` says "Configure:
Settings → Video Gen." The category is real
(`settings_screen.py:1224-1225`), so the pointer is right and the settings page
is simply missing the section.

**7. Stamp integrity is weaker than the review reported.** Corrected figures:

| review claimed | actually measured |
|---|---|
| 138 stamp lines | **126** occurrences across 29 files |
| 31 with no SHA | **27** (24 substantive; 3 are template boilerplate) |
| 1 invalid SHA | **7 distinct unresolvable SHAs** |
| 8 of 12 branches gone | **20 cited; 12 gone from local *and* origin; 17 of 20 absent from origin** |

The worst offender, `0662e09f5`, is cited in **seven** pages (library.md:390,
import-and-export.md:661, media-and-conversations.md:415, notes.md:247,
prompts.md:419, search-and-rag.md:604, skills.md:277) and `git cat-file -t`
fails on it. Others: `d6b6a738f` (×3), `9f90e17b8` (×2), `0c24f50d9`,
`ec1ed811e`, `ee68f42ed`, plus `385afa95` in prose.
A stamp that names an object nobody can resolve is not evidence of
verification — it is the appearance of it, which is the same failure mode as
the fake backup checkbox in TASK-19550.

**8. `console/video.md` is the only non-stub page in the guide with no stamp at
all** (101 lines, linked from `console.md:16`), while `index.md:9-10` promises
every fully-written page carries one.

**9. An entire user-facing subsystem is undocumented.**
`tldw_chatbook/Persona_Visual/` (10 modules, ~234 KB) plus
`DB/VisualIdentity_DB.py` (30 KB) back **two user-facing panels** with
**12 buttons — 14 including the modal** (the review said 8):
- **"Visual Identity"** (`personas_visual_identity_pack_widget.py:127`), hosted
  in the Roleplay ▸ Characters editor — 6 buttons: Replace…, Generate, Generate
  All, Clear, Save, Cancel.
- **"Persona Visual"** (`personas_persona_visual_pack_widget.py:227`) — 6
  buttons: Replace…, Clear, Add Custom State, Import Pack…, Save Pack, Cancel
  Draft; plus `PersonaVisualCustomStateDialog` with Add State / Cancel.

Every relevant grep over `Docs/User_Guide/` returns **zero**: "Visual
Identity", "visual pack", "Generate All", "Import Pack", "Add Custom State".

## Acceptance Criteria

- [ ] `console.md` describes the composer that actually ships (Composer ▾,
      Menu, Send, Dictate, and the hidden Stop/✕), and no longer contradicts
      its own Menu description 46 lines earlier
- [ ] The stale `"press Mic to continue."` string at
      `UI/Console_Modules/dictation.py:1508` is fixed in the **product**
- [ ] `console/attachments-images-voice.md` prints the glyphs the code uses
      (`◉`, `◐`) and drops the retired Mic / Mic… / Rec ● / Text… labels
- [ ] `library/search-and-rag.md:411-421` no longer describes a retired Search
      screen; the semantic-index backfill instructions point at where that
      genuinely lives now
- [ ] "Roleplay & Chat Dictionaries" is replaced with "Roleplay" across all 15
      occurrences, and the impossible palette string is corrected
- [ ] The persona semantics inversion is corrected everywhere it appears — a
      persona is **who the user plays**, matching `personas_screen.py:345`
- [ ] `library.md` stops listing `research` as retired
- [ ] The "research screen crashes on first use" claim is **confirmed or
      refuted against the real app** and filed separately if real — it is not
      treated as fact in this task
- [ ] `settings.md` documents the Video Gen category that `console/video.md`
      sends users to
- [ ] Every "Verified against" stamp cites a SHA that resolves in this repo;
      the seven unresolvable SHAs are corrected or removed
- [ ] A check fails when a stamp cites an unresolvable object or a branch that
      no longer exists on origin — stamps must be verifiable, or they are
      decoration
- [ ] `console/video.md` carries a stamp, or is explicitly marked a stub
- [ ] The Persona_Visual subsystem is documented: both panels and all 12
      buttons, in a page linked from the guide index

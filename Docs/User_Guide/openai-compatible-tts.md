# Using OpenAI-compatible TTS servers — point speech at your own server

tldw_chatbook's OpenAI TTS provider can talk to **any server that speaks the
OpenAI speech API** (`POST /v1/audio/speech`), not just OpenAI itself. That
includes local, keyless engines such as **pocket-tts**, and self-hosted
gateways. When you point it at your own server, your server's model and
voice names are passed through exactly as you type them, and no API key is
required.

## What you need

- A running OpenAI-compatible TTS server and its address — for a local
  server, typically something like `http://127.0.0.1:8080`. Check the
  server's startup output or docs for the exact port, plus the model and
  voice names it accepts.
- tldw_chatbook with audio playback working (the Console speak button or
  the Speech Lab).

## Getting there

Press **F9**, or click **F9 Settings** in the nav bar (**Ctrl+P** →
"Tab Navigation: Switch to Settings" also works), then pick **Speech &
TTS** in the left
category rail (it sits in the **Core** group). The panel has three cards:
**Global defaults**, **Provider setup**, and **Configuration inspector**,
with a **Save** / **Revert** action row at the bottom.

## Point the OpenAI provider at your server

1. In **Provider setup**, set **Configure Provider** to **OpenAI**. (As the
   panel notes, this only chooses which provider's form you're editing — it
   does not change the Default TTS Provider yet.)
2. Set **Base URL** to your server's *full* speech endpoint — the server
   address followed by the API path. For a local server on port 8080:

   ```
   http://127.0.0.1:8080/v1/audio/speech
   ```

   The default value, `https://api.openai.com/v1/audio/speech`, is the
   official OpenAI endpoint; replace the whole thing.
3. **Credentials — usually skip this for local servers.** Keyless servers
   like pocket-tts need no credential: leave it unset and requests are sent
   with no `Authorization` header. If your server *does* require a key, use
   **Set credential**. Note that the `OPENAI_API_KEY` environment variable,
   if set, is also sent to a custom Base URL — unset it (or use a saved
   local credential) if you don't want that key going to your server.
   Your OpenAI **Organization ID**, if configured, is *never* sent to a
   custom Base URL — only to the official OpenAI endpoint.
4. In **Global defaults**:
   - **Default TTS Provider** → **OpenAI**
   - **Model policy** → **Exact**, and **Model value** → the model name
     your server expects (whatever its docs say — it is passed through
     unmodified).
   - **Voice policy** → **Exact**, and **Voice value** → one of your
     server's voice names (also passed through unmodified). Or choose
     **Server default** to let the server pick.
   - **Output format** and **Speed** as you like (check which formats your
     server supports; `mp3` and `wav` are the most widely implemented).
5. Press **Save**.

## Try it

- **Console:** select a completed assistant message and press its **🔊**
  action — you should hear your server's voice. Press **⏹** to stop.
- **Speech Lab:** open **Lab** in the nav bar, switch the mode chip to
  **Speech**, pick **🎤 TTS Playground**, enter text, and press
  **Generate**.

## Worked example: pocket-tts

pocket-tts is a small local TTS engine that exposes the OpenAI speech API
and needs no API key. With its server running (see its own docs for the
start command and port):

| Setting | Value |
|---------|-------|
| Configure Provider | OpenAI |
| Base URL | `http://127.0.0.1:<its-port>/v1/audio/speech` |
| Credential | leave unset |
| Default TTS Provider | OpenAI |
| Model policy / Model value | Exact / the model name from pocket-tts's docs |
| Voice policy / Voice value | Exact / a pocket-tts voice name (or Server default) |

## If you run an AllTalk server

AllTalk has its own first-class provider — use it instead of the OpenAI
form: **Provider setup** → **Configure Provider** → **AllTalk**, then set
**Server URL** (default `http://127.0.0.1:7851`) and **Default language**,
and set **Default TTS Provider** → **AllTalk**. No key is needed.

## App-wide default voice profile

Beyond pointing individual axes (Provider / Model / Voice) at your server,
**Global defaults** also has a **Default voice profile** selector at the
very top of the card — above Default TTS Provider, since it outranks those
fields. It lets you name one saved voice profile as the assistant's
app-wide default voice, used for Console speech whenever no
character-specific voice applies.

- **Options:** "None — use the fields below" (the previous behavior — the
  Default TTS Provider / Model / Voice fields underneath decide), plus one
  entry per voice profile you've saved (Lab ▸ Speech ▸ Voice Profiles or
  the Playground's "Save result as profile").
- **Live-linked.** Editing the chosen profile elsewhere (its provider,
  model, or voice) changes what speaks everywhere it's the default — you
  never need to reselect it here.
- **Precedence:** explicit request → a character's assigned voice → this
  default profile → the Model/Voice fields below → provider fallback. A
  character's own assigned voice always wins over the app-wide default;
  the default profile only applies when no character-specific voice does.
- **If the profile becomes unusable** (deleted, the profile store is
  unavailable, or the saved id is malformed), speech does **not** silently
  fall back — it refuses and offers a one-tap **"Use global voice?"**
  confirmation naming the default voice specifically (never
  mislabeled as a character voice, even on a message with no character
  context at all). Settings itself never silently clears a saved-but-broken
  selection: the picker keeps showing it (as "*id* (unavailable)") with an
  explanatory note, until you pick something else.
- **While the profile list is still loading** (e.g. right after opening
  Settings), the note under the selector reads "Loading voice profiles…"
  rather than a false "unavailable" — that only appears once the store has
  actually confirmed the saved profile is gone.
- **Deleting the profile that is the current app default warns you first**
  in the Voice Profiles library's delete confirmation, in addition to any
  character assignments it already reports.

## Per-character voices

Beyond the app-wide default above, a **character** can carry its own voice,
which outranks it. This is managed in two different places, deliberately
kept apart:

- **Create the profile** in the Speech Lab (**F7** ▸ **Speech** ▸ **🎤 TTS
  Playground**): generate a result, then **Save result as profile**. This
  works for all seven built-in providers, not just audio.cpp — pick
  whichever provider/model/voice you generated with.
- **Assign it to a character** in **Roleplay & Chat Dictionaries**
  (**Ctrl+5**) ▸ **Characters** ▸ open a character's card or editor ▸ the
  **Voice & Speech** block. Its picker's first option is **"Use global
  default"** (this app-wide default, or the plain axes below it if no
  default profile is set); every other option is one of your saved
  profiles. See [Characters & Personas ▸ Voice &
  Speech](roleplay-chat-dictionaries/characters-and-personas.md#voice--speech)
  for the full walkthrough of that block (Preview, Create, Edit/Repair,
  Remove).

**Precedence, restated:** explicit request → **the character's assigned
voice** → this app-wide default profile → the Model/Voice fields below it →
provider fallback. A character's own voice always wins, whether or not an
app-wide default profile is set.

**Legacy providers show "No catalog check."** audio.cpp is the only
provider tldw_chatbook can probe for a live model/voice catalog. For the
other six — OpenAI (including a custom Base URL like the one this page is
about), ElevenLabs, Kokoro, Chatterbox, Higgs, and AllTalk — a profile's
exact model and voice are used as-is, with nothing to verify them against.
So instead of an "Unverified" state that a Refresh could resolve, these
profiles show the honest, **permanent** state **"No catalog check"** (the
character picker's own option list shows it lowercased, "no catalog
check"). This is not an error: it means "there's no catalog to check
against," not "something is wrong." If the voice is actually misspelled or
missing on your server, you'll find out the same way as anywhere else on a
custom endpoint — the request fails at speak time.

## Quirks & troubleshooting

- **"Unable to connect to TTS service"** — the server isn't running at the
  Base URL, or the URL is wrong. Confirm you included the `/v1/audio/speech`
  path: requests go to the Base URL exactly as written.
- **Model or voice names only pass through for custom Base URLs.** Against
  the official OpenAI endpoint, a model outside `tts-1`/`tts-1-hd` falls
  back to `tts-1`, and a voice outside OpenAI's six falls back to `alloy`.
- **Wrong audio or errors after switching servers** — re-check **Model
  value** and **Voice value**; they must be names *your* server accepts,
  since tldw_chatbook no longer rewrites them for custom endpoints.
- The Base URL must be an absolute `http://` or `https://` URL without
  embedded credentials or a `#fragment` — anything else is rejected on
  save.

## Related settings & docs

- Console speak actions: [Console ▸ Attachments, images & voice](console/attachments-images-voice.md)
- The Speech Lab (playground, voice profiles, audiobooks): [Lab](lab.md) 🚧
- Per-character voice assignment, in full: [Characters & Personas ▸ Voice &
  Speech](roleplay-chat-dictionaries/characters-and-personas.md#voice--speech)

—
*Verified against dev @ 265dbd687 — 2026-08-04 (labels quoted from
`speech_tts_settings_panel.py`; keyless/passthrough behavior shipped in
TASK-2260, PR #1332). Verified against 0c24f50d9 — 2026-08-06 (voice
profiles slice 3, task 6 live check: default voice profile set and saved
through Settings ▸ Speech & TTS ▸ Global defaults; a real OpenAI Console
speak with no character active used the default profile's voice, not the
global axes voice; deleting that profile made speech refuse with the
"Use global voice?" dialog naming the default voice, not a character
voice; a fresh Settings open resolved the picker straight to the real
profile list with no false "unavailable"). Verified against dev @
7f23e0263 — 2026-08-06 (voice profiles slice 4: per-character voices
section added; labels cross-checked against
`stts_screen.py`'s `SPEECH_RAIL_SECTIONS` ("🗣️ Voice Profiles"),
`speech_playground_pane.py` / `STTS_Window.py` ("Save result as profile"),
and `stts_profile_library.py` / `personas_character_tts_widget.py` /
`personas_screen.py` for the "No catalog check" / "no catalog check"
copy — this page's own default-profile precedence bullet above was not
re-verified live in this pass). Re-verified live 2026-08-11 with the
TASK-15420 fix: a Console 🔊 speak with an exact custom model AND custom
voice name reached a mock OpenAI-compatible server at a custom Base URL
with both passed through unmodified (between roughly 2026-07-26 and this
fix, an internal model-route allowlist rejected any non-official model
name on the Console path before the request was sent, so this page's
Model-value instruction did not work there). Re-verified live 2026-08-11
with the TASK-15421 fix: the Speech Lab playground now seeds its Model and
Voice controls from your saved exact selection — a custom name appears as
"*your-name* (no catalog check)", stays selected, and Generate sends it
unmodified (previously the playground silently generated with
`tts-1`/`alloy` regardless of your saved settings).*

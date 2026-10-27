# First-Run Setup

> Verified against: fix/task-32959-summary-voice-line, 2026-09-27 (task-32959: the Summary gains a Voice line read back from the saved `[app_tts]` table). Previously: feat/wizard-omnivoice-tts, 2026-09-26 (task-32958: OmniVoice as a fourth Voice service).

On your first launch, chatbook offers a guided setup. It is entirely optional —
every step can be skipped (Next moves on without configuring it; on the
Provider step, Next with an empty key field, **Enter** in that field, and the
**Skip — connect later** button beside it all skip the step once), Escape
asks before closing, and anything you configure (or
don't) can be changed later in Settings. A skipped provider stays picked:
go Back, paste a key, and Next saves it for that provider. Provider and Model
then show as skipped in the progress tracker, and the Model step asks you to
pick a provider first.

If a step can't save what you entered, the reason appears just above the
navigation buttons — fix it and press Next again, or go Back. The same line
reports an unexpected error on a step: the step stays on screen, the keyboard
keeps working, and **Esc** (exit setup) and, after the first step, **← Back**
stay available, from the keys (**Ctrl+B**, **Ctrl+N**, **Enter**, **Esc**)
as well as the buttons. That includes a run you resumed from "Continue
setup?", and a step that fails while it opens: setup lands on that step with
the error line. If picking a provider fails this way, the provider you had
picked before stays selected, together with any key you typed for it; a key
is never carried over to a different provider. The highlight stays on the row
you tried, so the line names the provider that is still selected ("Couldn't
switch to Anthropic — OpenAI is still selected"), and it goes away once you
pick a provider that works. If you had not picked one yet, none is selected,
and Next goes on without a provider, as if you had skipped the step. A
failure after the new provider is already selected leaves it selected, and
the line then names no other provider.

While a Next is saving, **← Back**, **Next →** and **Skip setup** / **Exit
setup** are disabled and drawn dimmed. A Next that is still working after
about half a second also shows a line just above them that names the work:
"Preparing the Full setup…" after Welcome, "Saving the OpenAI connection…"
after Provider, "Saving the provider and model…" after Model, "Saving voice
settings…" after Voice (that save can take up to 30 seconds), "Finishing
setup…" after a Summary button, and "Saving *step* settings…" elsewhere. After
two seconds the line also counts the seconds ("Saving voice settings… 4 s").
It goes away as soon as the next step opens, setup closes, or the step
explains why it can't move on. A quick Next shows no line at all. When a step does refuse to move on (a missing API key, say), the keyboard
stays where it was, on **Next →** if you pressed it, so once you have fixed
the problem, Enter there (or Ctrl+N anywhere) tries again.
The search for local servers that the Provider step starts runs in the
background, so the screen keeps responding while it looks.

The model list that the Provider step fetches is kept for that provider, key
and address, so going Back to Provider and forward again reuses it. Only a
list that arrived is kept. After a check that failed (the server wasn't
running yet, say), setup asks the server again in the background the next
time it needs the list: when you go Back to Provider, when you press **Next →**
there, when you come Back to the Model step from a later step, and when you
press **Retry** on the Model step. A model list that arrives after a failed
**Test connection** replaces that test's result. A different key or address
always counts as a new check. A **Test connection** result is the app's, not
only setup's: Console, Home, Settings and Chat settings show it for the same
provider, address and key (a key you typed counts once setup saves it), so a
refused test reads Not ready in Console too.

A model you pick from the server's list becomes the default that setup saves,
but it is not added to the provider's saved model list, and the line under
**Pick a default model** says so after "Models for *Provider*.". To add one
later, use Settings ▸ Providers & Models ▸ Advanced ▸ Saved model list: after
**Discover models**, a model chosen in **Default model** joins the list when
**s** saves, and **Add to model list** adds the models you check. When the
server listed nothing and the step offers the provider's saved model list
instead, the line names only the provider.

When the Model step has no models to list, the list box shows one line of
plain text instead:

- "(loading models…)" while it asks the server.
- "Pick a provider first — or type a model name below" when no provider is chosen.
- "(no models found — enter one below)" when the server lists none.
- "Model listing unavailable; enter the model ID used by this endpoint." when the server has no model list, when it answers the model list with a 403 (that key may not list models, which says nothing about chatting, so the key is not marked as rejected), or when Anthropic uses your Claude subscription: its model list needs an API key, and the subscription is not one. The Provider step names the same cause, after its own model check and after **Test connection** alike: "Model listing unavailable; you can enter the model ID on the next step." After a 403 to a check that sent no key, both lines add where a key would go, since a server may answer a missing key that way: "If this server needs a key, add one under Authentication." on the Provider step, and "If this server needs a key, go Back to add one." on the Model step. A 403 to a check that sent a key (typed, saved or from an environment variable) adds neither, and neither does a Claude subscription.
- "Authentication failed — this API key was rejected. Go Back to fix it, or enter a model ID below." when the key is rejected.
- "*Provider* needs an API key and none was set. Go Back to add one, or enter a model ID below." when the server refused a check that carried no key (a llama.cpp server started with `--api-key`, say).

  These two show no **Retry**, because retrying cannot fix a key. For a
  provider that doesn't need a key (a local server, say), the Provider step
  names the same cause, after its own model check and after **Test
  connection** alike (the line starts with ✗): "*Provider* rejected this
  API key. Update it, or continue anyway." or "*Provider* needs an API key and
  none was set. Add one under Authentication, or continue anyway." (For a
  provider that needs a key, a rejected key keeps you on the Provider step
  instead; see *API keys and checks* below.) In both
  cases the section is
  titled "Authentication", not "Authentication (optional)", because a server
  that refuses a key, or asks for one, wants one. The title stays while that
  provider and address are selected, until a check of that address that sends
  no key gets through (the server was restarted without `--api-key`).

  Under a key you typed for such a provider, the line says what the last
  **Test connection** or model check of that key found: "Key staged — the server rejected it; update
  it above." after a 401, and "Key staged — the server did not refuse it." when
  the server answered. Before any check of that key, or when the check could
  not reach the server or got no usable model list (as after a 403 to **Test
  connection** or to the model check: that key may not list models, which
  says nothing about chatting), it reads "Key staged — it will be checked
  when you continue." (For a provider that needs a key, the line reads "Key
  staged. Press Check key to try it." and **Check key**'s answer is the
  step's status line.)
- "Couldn't reach the server (*reason*). Check it's running, then Retry — or enter a model ID below." when the check failed for another reason, such as a timeout. Ollama and llama.cpp name the server and its address instead: "Ollama isn't running at *address*. Start it (ollama serve), then Retry — or enter a model ID below." and "The llama.cpp server isn't reachable at *address*. Start it, then Retry — or enter a model ID below." These three show **Retry**.
- "*address* answered HTTP *code* — *what it means*. Retry, or enter a model ID below." when the server answered with an error (a 503 while llama.cpp still loads its model, say); "*address* answered, but its certificate isn't trusted — add your CA in Settings ▸ Network." for a certificate your computer doesn't trust; and "*address* speaks plain HTTP — go Back and use http:// instead of https://." for an `https://` address on a plain-HTTP server. These show **Retry** too, and none of them says the server couldn't be reached: it answered.

The line is a message, not an option: it has no radio marker, and the arrow
keys, Space and Enter cannot select it. Tab skips it (while models load it
holds the list's place), so focus goes to the **Or enter a model name** field
below, which always works.

The Provider step lists the same providers as Settings ▸ Providers & Models,
with a filter (matching names and IDs the way Settings' provider search does)
and a count line above the list. **Ready on this machine** comes first: each
local server setup found (see *Local servers* below) and each provider whose
key is already exported or saved ("Anthropic · key in ANTHROPIC_API_KEY").
Then Popular, Cloud, Local and Other; every row says whether it runs on
**This computer** or needs a **Cloud · API key**. Legacy aliases and the
"#2" custom slot sit behind **Show legacy entries (N)** at the end unless your
config already uses one; the filter still finds them, and Settings lists them
all. A filter that matches only a server setup found says so ("1 local
server matches “studio”"). Every row can be picked with the arrow keys and set up here, hosted
presets included (Together, Fireworks, NVIDIA NIM, BytePlus, DeepInfra and
the rest; see [Settings ▸ Inference clouds](settings.md#inference-clouds)).
Azure OpenAI, Databricks and Cloudflare Workers AI ask for their resource,
workspace or account URL above the key, with an example. Setup saves each
preset's key, model and documented base URL in its own
`[api_settings.<provider>]` table. **Custom Hosted** is not listed: it is the
internal engine route for custom endpoints, which you set up as **Custom
OpenAI-compatible**.

## Local servers

Setup looks for model servers on this computer as soon as the Provider step
opens: llama.cpp (127.0.0.1:8080 and :9099), Ollama (:11434), LM Studio
(:1234), Jan (:1337), an OpenAI-compatible KoboldCpp (:5001), and any local
address already in your config. Both `127.0.0.1` and `localhost` are checked,
and one server shows as one row. A server it finds is the first row of the
list, already highlighted — "llama.cpp on this computer · 127.0.0.1:9099 ·
3 models" — so **Enter** uses it. LM Studio, Jan and KoboldCpp are set up
through the **Custom OpenAI-compatible** row. Picking another provider never
forgets the server, and picking that engine's own row later fills in the
port that was found.

When nothing answers, the step says which addresses it checked, offers
**Check again**, and has a collapsed **Don't have a local server yet?** with
a pointer to this section, the set-it-up-now path (pick your engine, press
Next, and type the model you'll run), and **Use llama3.2 with Ollama — I'll
start it later**, which picks Ollama and starts the Model step with
`llama3.2` typed in (`ollama pull llama3.2` fetches it). While the step is
open and nothing was found, it quietly looks again every 15 seconds, so a
server you start meanwhile appears on its own. If the engine you picked
doesn't answer, the line under it names the address: "Nothing is answering
at http://localhost:11434 — start Ollama, then Check again."

A server that answers over https with a certificate your computer doesn't
trust (a self-signed server, or a company network that inspects secure
connections) is reported as exactly that — "The server answered, but its
certificate isn't trusted" — with a pointer to **Settings ▸ Network**, where
you can add your company's CA bundle. Setup never offers to turn
verification off. An `https://` address on a server that only speaks plain
HTTP is pointed at `http://` instead. Either way the server answered, so
neither the Model step nor the reminder you see after going Back says it
couldn't be reached.

## API keys and checks

Paste a key into the key field; a pasted key is cleaned of invisible
characters, wrapping quotes, a `Bearer ` or `NAME=` prefix and line breaks,
and the line under the field says what it removed and echoes only the length
and last four characters (cleared when you leave the field or press Next). A
typed key is cleaned the same way when you press Next. **Show** reveals
the key, and **Ctrl+U** clears the field. A key whose prefix belongs to
another provider (`sk-ant-`, `sk-or-`, `gsk_`, `AIza`, `sk-proj-`) offers to
switch to that provider; if you had already typed a key for that provider,
the line under the field says the moved key replaced it. With the key
already exported, the step offers
"Use OPENAI_API_KEY from your environment (found; not stored)" or **Store a
different key for this app**, which shows the key field; the line under it
says the variable must be set in the shell that starts chatbook. A saved key
is used instead of an exported one, and the step says so. If you move to
another provider after typing a key, a line under the list says that key
won't be saved unless you choose its provider. When setup saves your
provider, every other provider whose key is exported, and that your config
doesn't already decide, is saved as using that variable
(`credential_source = "environment"` with the variable's name). The key
itself is never written to `config.toml`.

**Check key** beside the key lists the provider's models with that key — the
same non-billable check Settings makes — and reads "✓ Key works — OpenAI
returned 137 models.", or "✗ OpenAI rejected this key.". A provider the
wizard has no model listing for (Azure OpenAI, Cloudflare Workers AI, MiniMax
and a few other hosted presets) is sent nothing, and the line says only that
its key isn't checked here (it is checked when you send your first message).
A provider whose model-list route answers that it isn't there (HTTP 404 or
405) can't be checked without a paid request, and the line says so. A
provider that answered without a usable list says what it answered instead:
that it won't list models for this key (HTTP 403, which never keeps you on
the step: the key may still chat), an error with its status code, or an
empty list. OpenRouter, Hugging Face and Novita list models
without a valid key, so that list can't confirm one. A rejected key (a 401)
keeps you on the Provider step, with the reason under the key field; for an
exported key the fix is **Store a different key for this app** (or another
provider), and **Skip — connect later** stays available. Any other failed
check marks Provider with **!** in the tracker as soon as it is known,
including one the Model step finds. The status line under the step uses ✓,
✗ and ! with matching colours, nothing is reported as an error before you
act, and **Test connection** (on local and custom rows) reports "✓ Connected
— llama.cpp returned 3 models. Pick one on the next step." or names the HTTP
status it got; a local server started with an API key that answers a keyless
test says it needs one. A refusal that names the endpoint field goes away as
soon as you edit that field.

## Keyboard

- **Enter** continues to the next step (from a choice list or a text field;
  in the API-key field it first checks a key you typed — the same check as
  **Check key** — and with the field empty it skips the provider step).
  **Ctrl+N** / **Ctrl+B** also move next/back, and **Escape** asks before
  leaving setup.
- **Arrow keys select** as they move through a choice list — what you land on
  is what you get; no extra keypress needed. Picking a provider that needs a
  key with **Enter**, **Space** or a click moves you into its key field.
- **Typing on the provider list filters it**: letters go to the "Filter
  providers — name or ID" field above the list while the list keeps the
  keyboard, **Backspace** edits the filter, and **Space** still picks.
- **Tab** moves from a step's content to **Next** first; the footer runs
  progress · Back · Next · Skip/Exit, left to right.

## The two tracks

- **Quick setup (recommended)** — connect one provider, pick a default model,
  optionally try a voice and protect your keys, done. Everything else stays at
  recommended defaults (tools off, RAG off, default theme, notes sync off).
  The step count never changes mid-run — the key-protection step is always
  shown, and simply says so when there is nothing to protect (the progress
  tracker then marks it – instead of ✓).
- **Full setup** — also walks through RAG/embeddings, built-in tools, notes
  sync, appearance, and key encryption.

## What each step does

| Step | What it configures | Where to change it later |
|---|---|---|
| Provider | API key or local server endpoint | Settings ▸ Providers & Models |
| Model | Default chat model | Settings ▸ Providers & Models |
| RAG | Embedding model (needs the `embeddings_rag` extras) | Settings ▸ RAG |
| Speech (full track) | Voice-input transcription language and precision | `[transcription]` in config.toml — no Settings category owns it yet |
| Tools | Built-in tool gates (all off by default) | Settings ▸ Tools & MCP (Ctrl+9) ▸ Servers ▸ built-in row ▸ **Tool gates**, or `[tools]` in config.toml |
| Notes sync | Folder + on/off toggle | [Library ▸ Notes](library/notes.md), the toolbar's Sync panel — not in Settings |
| Appearance | Theme and splash screen card | Settings ▸ Appearance |
| Voice | Spoken replies — "No voice for now" (the default), PocketTTS (its own local server), OpenAI (your OpenAI key), a Custom endpoint, or OmniVoice (local, installs its model here); sample + "Test and Hear" (endpoint, model, voice and format under Advanced) | Settings ▸ Speech & TTS |
| Protect keys | Config encryption (password at startup) | Settings ▸ Privacy & Security ▸ **Encryption**: Encrypt keys, Change password, Turn off encryption (each asks for the master password) |

The Tools step is the only place in setup that turns a tool on, and it says
so up front: "Everything is off by default. Tools that read or change your
files still show an approval card every time they run." Each row carries the
tool's plain-language name and one line about what it does — the read-class
ones (Read file, List directory, Find files, Search in files, Expand
document) add that they ask before running unless you approve a longer
scope, and the ones that write are marked with ⚠. Leaving every switch off
is a supported outcome: the summary then reads "gates off; the assistant
may still use chatbook's own tools (sub-agents, web search, notes, Watchlists)
when the model's context window has room. Turn gates on under Settings ▸
Tools & MCP ▸ Servers ▸ built-in row ▸ Tool gates", which is where the same
switches live after setup (**Ctrl+9** opens Settings ▸ Tools & MCP). The first half is there because
the switches are not the whole story: Console's assistant also gets the agent
runtime (sub-agents, skills), the local web and Watchlists tools
(`[console] local_tools_enabled`) and chatbook's built-in notes and character
tools, every call still subject to its MCP Ask/Allow/Off permission. The
app's internal `chat_with_llm` tool, which the in-process server cannot run,
is never offered. A self-hosted model whose context window the app has not
read yet gets a plain request with no tools until it can be sized.

The Voice step's first choice is **No voice for now**, and it is selected
unless a voice is already saved: Next then writes nothing, and the step says
so ("Nothing is saved. Set up a voice any time in Settings ▸ Speech & TTS.").
On a re-run the step starts from the voice you saved, for example "Current
voice: OpenAI · tts-1-hd · shimmer — unchanged unless you edit it.", and Next
leaves it exactly as it was unless you change something, test a sample, or
change **Use this voice when Chatbook reads replies aloud**. Choosing **No
voice for now** over a saved voice keeps that voice and says so ("Keeps your
current voice (OpenAI · tts-1-hd · shimmer); Next changes nothing. Replies are
read aloud only while Speak replies is on in Console."). When another provider
reads your replies (for example kokoro, set up in Settings), the step starts on
**No voice for now** and names it ("Current voice: kokoro — kept as it is").
A profile that still holds the PocketTTS address an earlier version of setup
wrote (`127.0.0.1:8765/v1/audio/speech`, which pocket-tts never serves) also
starts on **No voice for now**, with a line saying that voice can't speak; pick
a service to replace it. Resuming a setup that an earlier version left
unfinished doesn't bring that address back either.

A line under the service choice says whether it will work: "PocketTTS — not
running at 127.0.0.1:8000" (one quick connection check), "OpenAI — uses your
OpenAI key (key found)", and so on. That check only shows that something is
listening at the address, so when it is, the line says "a server is listening"
and leaves it to **Test and Hear** to confirm the server is PocketTTS (port
8000 is a common default for other local servers). It only connects to an IP
address or `localhost`; for a host name the line says **Test and Hear** checks
it, so typing an address never waits on a name lookup. **Test and Hear** plays a
short sample; Enter in **Sample text** runs it too (the hint line says so). A
failed test starts "Test failed —" and names the cause: the server isn't
running at that address, the key was rejected, there is no speech endpoint at
that host, it timed out, or the reply wasn't audio. A PocketTTS address on
another port counts as Custom, but its failures still name PocketTTS and say
how to start it there (`pocket-tts serve --port 8766`). While **Test and
Hear** can't run, the status line under it says why ("Type some sample text
above to test the voice.", or "To test, fix this under Advanced: …", for
example a PocketTTS address with a format other than `wav`). After a test,
focus goes back to **Test and Hear**, unless you moved it while the test ran.

PocketTTS, OpenAI and Custom share one OpenAI-compatible voice slot. While no
other provider reads your replies, a voice saved there *is* the one replies
use, so **Use this voice when Chatbook reads replies aloud** is ticked and
can't be unticked, and the line under it says so ("Replies will use this voice
— no other voice is set up.", or "This becomes the voice replies use — it
replaces OpenAI · tts-1-hd · shimmer."). When another provider reads replies,
the box is yours: a successful test ticks it, and unticked the service is
saved for later while replies keep using that provider ("Saved for later;
replies keep using kokoro."). When OmniVoice reads your replies, the step
starts on **OmniVoice** with the box ticked and locked ("Replies use OmniVoice
now — pick another service to change that."), and the other services name it
("Saved for later; replies keep using OmniVoice."). The
endpoint, authentication, model, voice and format sit under "Advanced" (the
**API key** option uses your OpenAI key, from the Provider step, this step,
Settings or `OPENAI_API_KEY`); Voice and Format are pickers with an "Other…"
choice, and editing one of these away from the selected service's own values
switches the service to **Custom** (your edits are kept if you switch away and
back).

**PocketTTS** is a separate local server, not part of Chatbook: install
`pocket-tts` (from PyPI) and start it with `pocket-tts serve`. Chatbook talks
to its own API at `http://127.0.0.1:8000/tts` (its default port; change the
endpoint under Advanced if you started it on another port). It returns WAV
audio only.

**OpenAI** needs an OpenAI API key. If none is found, paste one into the
masked **OpenAI API key** field that appears: it is kept in setup until you
press Next, then saved where Settings ▸ Speech & TTS keeps it, so **Protect
keys** can still encrypt it. You can also pick another service or choose "No
voice for now". **Leave setup and add key in Settings…** asks first — setup
picks up at Voice next time.

When Voice does save, the step reports the result itself and refuses to move
on if the save failed, so setup never raises a pop-up notification over a
later step's buttons. On terminals smaller than about 100×30 the wizard shows
a one-line nudge — everything still works, steps just scroll. Where one row is
too narrow for every service name (80 columns, for example), the Voice
service choice wraps to two rows so no name is cut off.

The **Protect keys** step reads your config file each time it opens and
says what it holds:

- **A key saved as plain text** — it names each provider whose key is saved
  ("Your OpenAI key is saved as plain text in config.toml…") and the file's
  full path (a long path wraps at its `/` separators rather than being
  shortened), then offers two buttons: **Encrypt with a password…** and
  **Keep as plain text** (which simply continues). It counts every secret
  that encrypting would encrypt: a key under a provider's settings, an older
  `[API] openai_api_key`-style entry, a custom endpoint's key (named by the
  endpoint), and any other saved secret, which it names by its config key.
  After encrypting, the step says "✓ Encryption enabled.", the button becomes
  **Change password…**, and the keyboard moves to **Next →**, so Enter
  continues to the summary. A failed attempt's error goes away when you try
  again, so a success is never shown beside an earlier failure. While the
  keys are being encrypted (or the password changed), the step's buttons are
  disabled and **Next →** waits with "Encryption is still running", so setup
  never records the step before the outcome.
- **Keys already encrypted** (for example when you run setup again) — "Your
  saved API keys are encrypted", with **Change password…**, which asks for the
  current and the new password. After a change the keyboard moves to
  **Next →** as well. Setup never sets a second password over encrypted keys.
  A key you typed on this run but haven't saved yet is called out here too,
  with **Back to Model**.
- **A key you typed but that isn't saved yet** (you skipped the Model step;
  the key is saved together with the model) — it says so and offers **Back
  to Model**. The keyboard stays on **Next →**, so Enter still moves forward;
  **Back to Model** is one Tab away. An older key for the same provider does
  not count as saving its replacement. Encrypting saved keys also leaves this
  warning visible until Model saves the new draft. Only the key typed for the provider you
  last chose counts: switching to a provider that needs no key drops it.
- **Keys only from environment variables** — it names them (for example
  `OPENAI_API_KEY`) and says config.toml holds no key to encrypt.
- **No key at all** — "Nothing to protect".
- **A saved key that can't be read** (encryption is off, but a value is still
  encrypted with an earlier master password) — it names up to three such
  values, providers' keys first, then "and N more", and says where to
  re-enter or clear them: Settings ▸ Providers & Models when every one is a
  provider's key, Settings ▸ Advanced Config otherwise. It offers no Encrypt button,
  because encrypting is refused until that value is fixed. The summary's Key
  encryption row says the same.
- **config.toml can't be read** — it says so and offers nothing, rather than
  guessing whether your keys are encrypted (if the file says encryption is
  on, it shows the encrypted state instead).

The progress tracker marks this step ✓ only when your keys are encrypted;
otherwise (nothing to protect, keys kept as plain text, a file it couldn't
read, or a saved key it can't read) it shows –. When you resume setup, the
step and the tracker follow what config.toml holds now, not what your saved
progress recorded: if you turned encryption off after leaving setup, Protect
shows the plain-text offer again.

Every state says where to change this later: Settings ▸ Privacy & Security ▸
**Encryption**.

If **Exit setup** cannot save where you are in setup, it offers **Exit
without saving progress**: settings you already saved stay saved, and setup
just won't resume from that step. **Skip setup** on the first step does the
same when it cannot record that setup is finished; exiting then means setup
opens again the next time chatbook starts.

Choosing **OmniVoice** shows a local panel instead of the endpoint fields: it
tells you if the `omnivoice_tts` engine is missing, installs the 1.1 GB model
through the same consent dialog as the model browser, and tests a sample on
this computer. Saving it as the default also saves a fixed voice seed, but the
default voice can still vary between replies — create a voice profile in Voice
Cloning for a consistent voice.

**Model downloads (Speech and OmniVoice).** Both installs open the same
review as Lab ▸ Models. It leads with what you are deciding: the download
size, where it installs and how much space is free, and each component's
licence (for Parakeet: "CC-BY-4.0 (speech model) · MIT (voice-activity
detector)"). Source, pinned revision hash, provenance and staging figures
are under **Details**. While a download runs, the progress line shows
how much has arrived, the speed and the time left (Speech also repeats it on
its own status line: "Downloading Parakeet v2 (English, INT8) — 65.0 of
632.8 MiB · 4.1 MiB/s · about 2 min left"), and **Cancel download** stops it
within a few seconds: the partial download is removed and the step says
"Download cancelled — nothing was saved." Only one model downloads at a
time, so starting the second install while the first is running waits in
line ("Waiting for the OmniVoice voice model download to finish…") and then
starts on its own; Cancel works while it waits, too, and leaving setup
cancels an install that is still waiting its turn. **Use model from disk…**
(Parakeet files you already have) also installs the voice-activity detector
those files need when it is not installed yet, and that download waits in
line the same way. If
Chatbook is closed or killed part-way through, the next review says "A
previous download stopped at 120.0 of 632.8 MiB", its button reads
**Resume download**, and the download continues from about that point
instead of starting over.
While the Speech download runs, its language, precision and GGUF choices
are disabled.

The final summary shows a ✓/✗ line per area, read back from what was actually
saved — and if the connection check failed while you were setting up (a
rejected API key, a server that needs a key you didn't set, an unreachable
local server), the summary says so instead
of showing a ✓, the progress tracker marks those steps with !, and moving
past the model step asks for an explicit "Continue anyway". The Voice line names
the voice replies use — service, model and voice (for example "✓ Voice —
OpenAI · tts-1-hd · shimmer") — or another provider that reads them, adding a
voice saved beside it ("kokoro (default voice); PocketTTS also saved"). It
reads "not set up (optional)" when no voice is saved, and flags the old
`127.0.0.1:8765` PocketTTS address as unable to speak instead of showing a ✓.

The Summary's exits are **Review provider setup**, **Add your first document**
(lands on Library's Import canvas — this is where your content lives),
**Write your first note** (lands on Library's New note view — no provider
needed), **Explore Home**, and **Review settings**.

Finishing with **Start chatting** opens Console on the provider and model you
just saved, in one chat tab, with no warning. The empty transcript says what
setup connected, for example "Setup complete — OpenAI · gpt-4.1-mini. Ready —
type a message to begin." If the saved default changed between setup and
Console opening, a single notice names the model Console is using instead.

The Summary also asks — once, default off — whether chatbook may check your
configured providers' model lists online at startup. Whatever you choose is
final until you change it in Settings; finishing setup never hands you a
separate consent pop-up afterwards. Local servers (see *Local servers*) are
auto-detected on localhost only; no probe traffic leaves your machine without
your action.

## Starting chatbook when your keys are encrypted

If you set a master password (Protect keys, or Settings ▸ Privacy &
Security ▸ Encryption), chatbook asks for it in the terminal before the app
opens. Both ways of starting it — `tldw-cli` and
`python -m tldw_chatbook.app` — use the same prompt:

```
Your saved API keys are encrypted. Enter the master password you set during setup.
Master password (leave empty if you forgot it):
```

- **A wrong password** prints "That password didn't match. Try again." and
  asks again. There is no limit on attempts.
- **The right password, but a key encrypted with a different one** (left
  behind by an older version that let you set a second password) prints
  "That password is right, but some saved keys were encrypted with a
  different password and can't be read with it." The app does not open with
  keys it cannot read. If you remember the earlier master password, enter
  it; otherwise leave the prompt empty to reset the keys.
- **An earlier password that still reads every saved key** is accepted even
  though it is not the latest one you set. chatbook says "That password
  reads every saved key, so it is your master password again", re-encrypts
  the keys under it and waits for Enter before opening the app. The later
  password stops working.
- **Forgot it?** Press Enter on an empty prompt. chatbook explains what a
  reset does and offers `[R]eset saved keys or [Q]uit`. **Reset** removes
  every encrypted value from config.toml and turns encryption off, then
  waits for Enter so you can read what happened, and opens the app. Chats,
  notes and documents are not touched; you re-enter your API keys in
  Settings ▸ Providers & Models. **Quit** leaves everything as it was. The
  choice is read from the terminal even when standard input is redirected.
- **Ctrl+C or Ctrl+D** at the prompt, or while the password is being
  checked, quits without changing anything.
- **config.toml says encryption is on but its password check is missing**:
  chatbook says so. If saved keys are encrypted, it asks for the master
  password: one that reads every key unlocks them and repairs the check, and
  an empty entry offers the reset. If nothing is encrypted, it offers the
  `[R]eset saved keys or [Q]uit` choice straight away.
- If chatbook cannot ask privately (no terminal attached), it opens a small
  recovery window that says so in plain words; relaunch from a terminal to try
  again. Backup & Restore stays one button away there but no longer opens on
  its own.
- **Browser sessions (`tldw-cli --serve`) cannot ask for the master
  password.** Each browser session shows that recovery window instead of
  prompting on the server's terminal. To serve chatbook in a browser, turn
  encryption off first (from a terminal launch, in Settings ▸ Privacy &
  Security ▸ Encryption). In that window, opening another profile from
  Backup & Restore says it needs a terminal instead of ending the session.

A saved key that is still encrypted is never treated as a usable key. If a
key cannot be decrypted, its provider shows as not ready, never ready. A
provider that needs a key reads as missing its key ("API key missing", and
the API key row's source word **missing** in Providers & Models). A local
provider whose saved key is still encrypted reads "Not ready · no key" too,
even though it would need no key otherwise. Either way, re-enter or clear the
key in Settings ▸ Providers & Models. A custom endpoint (a local server saved under **Custom endpoints**)
whose key is still encrypted reads "Not ready · no key" the same way;
re-enter or clear its key there. A custom endpoint's own key decides: a
still-encrypted key saved under the provider it is built on (for example
llama.cpp in Providers & Models) does not block it. While a local server's
key is still encrypted, chatbook also stops asking that server for its
context size. OpenAI and ElevenLabs text-to-speech, remote Whisper transcription,
image generation (SwarmUI included) and the tldw server connection never send
a still-encrypted value either: they use the next place a key can come from
(an environment variable, the keychain, another saved field) or send none.

## Setting up another machine

A second computer does not need setup again. There are three routes:
carry your config.toml, export the same environment keys, or restore a
backup. The first two carry settings only; to bring your chats, notes and
documents with them, also copy the data folder (below). A backup can carry
both, but Create backup does not yet work on every profile; see "Restore a
backup" below before relying on it.

### Carry your config.toml

Setup saves what it configures in one file, `config.toml`, by default
`~/.config/tldw_cli/config.toml`. Copy it to the new machine, then either
put it at the same path and start `tldw-cli` as usual, or keep it anywhere
and point chatbook at it:

```
tldw-cli --config /path/to/config.toml
python -m tldw_chatbook.app --config /path/to/config.toml
TLDW_CONFIG_PATH=/path/to/config.toml tldw-cli
```

`--config` and the `TLDW_CONFIG_PATH` environment variable do the same
thing: `--config` for one launch, the variable for every launch that sees
it. When both are set, `--config` wins. If the file does not exist yet,
chatbook creates a new config there and offers setup as on a first launch;
its folder must already exist. Before anything starts, `--config` refuses
a folder ("give the config.toml file inside it"), a file in a folder that
does not exist ("… does not exist; create that folder or check the path"),
a new file in a folder you cannot write to ("cannot create …"), and a file
or folder you cannot read ("cannot read …"). With either, Settings names
the file an override config.

The `[first_run]` table decides whether setup is offered:

| In the copied file | On the new machine |
|---|---|
| `setup_completed = true` | Setup is not offered; chatbook opens as it did on the old machine. |
| `setup_started = true`, without `setup_completed` | The setup was left unfinished: chatbook asks "Continue setup?" when the file still holds that run's progress, and otherwise says setup isn't finished. It never restarts setup by itself. |
| No `[first_run]` table | Setup is offered once, unless a provider is already configured (a key saved in the file, or a provider key in the environment). |

So copy the config of a finished setup. You can run setup again any time
from Settings ▸ Diagnostics ▸ Run Setup Wizard.

**Your keys travel with the file.** A copied config.toml carries every
provider API key saved in it, in plain text unless password encryption is on
(Protect keys, or Settings ▸ Privacy & Security ▸ Encryption). Handle the
copy like a password: don't commit it to a dotfiles repository, put it in a
shared or synced folder, or send it unencrypted. If encryption is on, the new
machine asks for the same master password before the app opens, through
both `tldw-cli` and `python -m tldw_chatbook.app`, including with
`--config` (see "Starting chatbook when your keys are encrypted" above).

What config.toml does not carry: your chats, notes, characters and documents
live in database files under `~/.local/share/tldw_cli/`, not in the config.
To bring them, quit chatbook on both machines and copy that whole folder,
every file in it, to the same place on the new machine. If chatbook has
already run there, move its folder aside first rather than copying into it;
the copy replaces anything chatbook saved there. A backup (below) carries
them too, once Create backup works on your profile. Paths written in the
file are used as written: `~/...` paths follow the new machine's home
folder, while an absolute path (a notes sync folder, a custom database
location) must exist there. `--config` and `TLDW_CONFIG_PATH` choose the
config file only; they do not move the data folder.

### Environment keys

chatbook never writes a key it read from an environment variable
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY` and so on) into config.toml. A setup
that used an exported key therefore carries no key in the file: export the
same variables on the new machine. When the file also holds a saved key for
the same provider, the saved key is used. On a machine with no config at
all, an exported provider key is enough for chatbook to skip the setup
offer; it says once which variable it found ("Found OPENAI_API_KEY — you're
ready to chat. Run setup any time: Settings ▸ Diagnostics ▸ Run setup
wizard.").

### Launch flags

| Flag | What it does |
|---|---|
| `--config PATH` | Uses this config.toml for this launch (the same as `TLDW_CONFIG_PATH`; the flag wins when both are set). |
| `--no-splash` | Skips the splash screen for this launch only. `[splash_screen] enabled` in config is not changed. A `--serve` browser session is started without it, so it still shows the splash when config enables it. |

Neither flag writes to config.toml, and no launch flag marks setup as
completed. `tldw-cli --help` lists both and ends with a note that names
`TLDW_CONFIG_PATH` and this section, with a link to this page. A recovery
profile opened from Backup & Restore selects its own config, so `--config`
cannot be combined with it.

### Restore a backup

To bring your data as well as your settings, create a backup on the old
machine (Settings ▸ Overview ▸ **Backup & Restore** ▸ **Create backup**). It
writes a `.tldw-backup.zip` file, or `.tldw-backup.zip.age` when encrypted.
Backup coverage is still growing: the Review lists each store, and one
marked `unsupported` (on some profiles that includes the config and the main
databases) is not in the archive. Until the Review shows what you need as
included, carry config.toml and the data folder as described above.

**Known problem:** on some profiles Create backup currently stops with
"Failed: capturing …" (ending in `backup_operation_failed` or
`admission_timeout`) and writes no file, even after a successful Review. If
that happens, carry config.toml and the data folder as described above
instead.

In the Create backup form, the line just above the buttons says why
**Create backup** is disabled: "Create backup unlocks after a successful
Review." until **Review** succeeds (and again after any change to the form),
or the review's own reason, such as a Partial backup that needs "Acknowledge
Partial archive…" ticked, or not enough free space (at the destination, or
in the temporary folder where the backup is staged). Pressing **Create
backup** uses up that review, so the line then reads "Backup started;
progress is shown at the top. Press Review to create another."

On the new machine, setup's Welcome step (and the "Continue setup?" dialog)
has **Restore a backup**. It opens "Restore from a backup" straight on the
Inspect / restore pane, with the cursor in the archive field and the format
named under it; **Inspect / restore** is the first of the actions above it.
Choose the archive and press **Inspect** (or Enter in the field); the
restore controls appear once the archive is verified. **Esc** returns to
setup with your choices intact.

- A config.toml there gets "That's a settings file, not a backup archive."
  with the `tldw-cli --config <this file>` command and this section's name.
  Carry it as described above instead.
- A folder gets "Choose the archive file, not a folder."
- Changing the archive path clears either message.

## Running it again

- **Settings ▸ Diagnostics ▸ Run Setup Wizard**, or
- Command palette: "Setup: Run setup wizard…"

On a re-run, current values are prefilled and stored API keys are shown only
as "configured" — never displayed.

*Verified against fix/library-crit8-polish-shell — 2026-09-08 (task-32072: the
Summary now offers "Add your first document", which finishes setup on Library's
Import canvas; the wizard previously never mentioned Library at all.)*

*Verified against fix/library-notes-onboarding — 2026-09-09 (task-32140: the
Summary offered no path into Notes for a local-first user without a
provider — every exit pointed at provider setup or the generic Import
canvas. Added "Write your first note", which finishes setup on Library's
New note view directly. The Console's post-setup "Get started" card gained
the matching "Write a note in Library" action, which needs no provider and
stays available for the whole time the card blocks the composer.)*

*Tools step verified against `fix/approval-wave-b-card` @ e7409210cc and
`fix/approval-wave-c-hub` @ a999fcf6e6 — 2026-09-10 (task-32290, against code
and tests, not a live screen): the step's own copy, the read-class "Asks
before running unless you approve a longer scope." descriptions, and the
summary's "all off; turn them on under MCP ▸ Servers ▸ Tool gates"
destination, which replaces this page's older "there is no Tools category"
pointer. Read-class wording updated 2026-09-11 (task-32290 part 2, Qodo
follow-up to task-32284/32289): the blurb's old closing sentence is gone,
replaced by the "approve a longer scope" wording quoted above, once the
wizard's copy moved onto `GateableTool.blurb` alongside the rest of the
tool row.*

*Verified against fix/library-notes-w3-wizard-toast — 2026-09-11 (task-32266:
the Voice step's save raised the global "Settings saved successfully!" toast,
which Textual docks bottom-right of the current screen — by the time the write
settled the wizard had advanced, so the toast landed over the Protect step's
buttons or the Summary's exit actions, "Write your first note" among them. The
wizard's save no longer announces itself; the step that made it still reports
every outcome in place.)*

*Verified against fix/library-notes-w4-console-handoff — 2026-09-14 (task-32555
AC#3, at 235x52): on the Provider step with OpenAI picked and no key, the key
field's hint ends "No key yet? Enter skips this step — you can add a provider
later in Settings." and Enter in the empty field moves from Step 2 of 6 to Step
3 of 6; the Summary then reads "✗ Provider — no credentials or saved endpoint"
(`wave4-caps/console-handoff/handoff-12a-wizard-key-hint`, `12-wizard-skip`,
`12b-wizard-summary`). Enter used to do nothing there — no probe, no advance,
no message. A provider that is already ready without a typed key (an exported
environment variable, a local server) is not cleared by that Enter; the hint
does not offer the skip in that state.*

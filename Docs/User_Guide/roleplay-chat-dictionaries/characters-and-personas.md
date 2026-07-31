# Characters & Personas — author who the AI plays

## What this screen is for

A **character** is who the AI plays: a name, a description, a personality, an
opening line, and the prompt text that keeps it in role. A **persona** is the
smaller cousin — an assistant profile (name, description, system prompt) you
can stage into a chat without inventing a whole cast member. Both live behind
mode chips on the Roleplay & Chat Dictionaries screen, and both can be handed
to Console with one button. Come here to write, import, illustrate, or hand
off a character.

## Getting there

Press **Ctrl+5** (or **Ctrl+P** → "Roleplay & Chat Dictionaries"), then pick
the **Characters** chip in the Modes strip — it is the mode the screen opens
on. **Personas** is the chip beside it. Once you are on this screen,
**Ctrl+1** and **Ctrl+2** jump to those two modes (they do *not* change
screens here — see [Keyboard & commands](#keyboard--commands)).

## Layout tour

![Character editor](../images/roleplay/character-editor.svg)

Three panes: the **Library** rail, the centre pane, and the **Inspector**
rail. The rails behave the same in every mode and are described on the
[parent page](../roleplay-chat-dictionaries.md); this page is about the
centre, which shows either **the character card** (read-only, with a single
**Edit** button; "No character loaded. Select one from the library." before
you pick one) or **the character editor**, headed "Character Editor". Below
either sit two docked panels, **"Dictionaries (copied into this character)"**
and **"World Books (copied into this character)"**. The rail's **New**,
**Import** and **Duplicate** act on the current mode; the Inspector's
**Attach to Console**, **Start Chat**, **Export JSON**, **Export PNG** and
**Delete** act on the current selection.

## Features & controls

### The character card

Rows render as "Label: value" and appear only when filled in: **Name**
(falling back to "Unnamed Character"), **Description**, **Personality**,
**Scenario**, **First message** (the greeting the character opens with),
**System prompt**, **Post-history instructions** (text pushed in after the
conversation so far), **Creator**, and **Version** (defaulting to "1.0").
Three rows always show: **"Tags: none"** (or a comma-separated list),
**"Alternate greetings: N"**, and **"Avatar: none"** / **"Avatar: embedded"**.
The first alternate greeting, if any, is previewed underneath. **Edit** opens
the editor; it stays disabled until a saved character is loaded.

### Editor — the generation toolbar

Everything here drafts text with the model Console is set to use.

| Control | What it does |
|---|---|
| "Context: whole character" ⟷ "Context: field + description" | Click to flip. Tooltip: "How much of the character the model sees when generating a single field" |
| "Generate whole character…" | Opens the concept row. Tooltip: "Draft every empty field from a short concept" |
| Concept row | An input reading "One-line concept, e.g. a drowned-library archivist", plus **Draft** and **Cancel** |
| Per-field **Generate** | Drafts that field alone; the button reads "Generating…" while it works |

Two behaviours worth keeping apart. A **per-field Generate** never writes into
the field: the text arrives in a preview panel titled "Generated {field} -
review before applying" (filling in as it streams) with **Accept**,
**Regenerate** and **Discard** — nothing touches your form until **Accept**.
**Draft** (whole character) writes straight into the form, but only into
fields that are *empty*, so filling a blank card never overwrites text you
already wrote; it reports "Filled N empty field(s)." or "Nothing was filled:
every generated field already has content." An empty concept is refused with
"Describe the character first, in a line or two."; failures read "Could not
generate: …".

### Editor — fields

Primary fields, in form order: **Name** (placeholder "Character name"),
**First message**, **Description**, **Personality**, **System prompt**. Each
long one carries its own **Generate**.

**"Advanced ▸"** expands (to "Advanced ▾") — collapsed every time you open the
editor — and holds **Scenario**, **Post-history instructions** and **Creator
notes** (each with **Generate**); **Creator** (placeholder "Creator name"),
**Version** (defaulting to "1.0") and **"Tags (comma-separated)"**
(placeholder "tag, another tag"); and **Alternate greetings**, a real list
(column "Greeting") rather than a text box — type into the box, press **Add**,
then select a row to **Update**, **Delete**, **Move up** or **Move down**.

### Editor — avatar

The row reads "Avatar: none" or "Avatar: embedded" ("Avatar: generating…"
while an image is being made), followed by **Upload**, **✨ Generate** and
**Remove**. **Upload** opens "Upload Character Avatar" with filters for Image
Files, PNG Files, JPEG Files, WEBP Files and GIF Files; files must be **5 MB
or smaller** and one of "PNG, JPG, JPEG, WEBP, or GIF", or you get "Avatar
image must be 5 MB or smaller.", "Unsupported avatar image type…" or "Avatar
image file is empty." On success: **"Avatar staged. Save the character to
persist it."** — the picture is not stored until you press **Save**.
**✨ Generate** needs a Description first ("Add a description first.") and
needs image generation configured; when it isn't, the toast names the setting.

### Editor — expressions

A block headed **Expressions** with a "Style: …" readout and four buttons —
**Style…** (pick the image style), **✨ Generate all**, **Import set…**,
**Export set…** — over one row per expression state (*thinking*, *speaking*,
*error*), each with **Upload**, **✨ Generate** and **Clear**. Those per-state
buttons stay disabled until the character has been saved once ("Save the
character to add expressions."), and exporting a set before saving is refused
too. **✨ Generate all** asks first — "This will overwrite the existing avatar
and/or expression images. Continue?" (**Generate all** / **Cancel**) — and
reports "{k}/{N} generated.".

### Editor — saving and validation

The footer lists problems under **"Validation errors:"** — **"name:
required"** and **"image exceeds 5 MB"** both block **Save** (the offending
row is outlined in red), while **"greeting N is blank"** is only a *warning*:
listed, but it never blocks the save and never marks the row.

**Save** persists and toasts **"Character saved."** (failures read "Save
failed: …"); the editor stays open on the saved character so you can keep
working. **Cancel** leaves it. Either way, unsaved edits raise the
**"Unsaved Changes"** dialog — 'The tab "{name}" has unsaved changes. Are you
sure you want to close it?' with **Close Without Saving** / **Keep Open**.

### Create, duplicate, delete

**New** in the Library rail (or **Ctrl+N**) opens a blank editor with the
cursor in Name. **Duplicate** copies the selection — description, personality,
scenario, greetings, tags, avatar and all — as **"{name} (copy)"**, then
"(copy 2)", "(copy 3)"…, with **no confirmation**. **Delete** (Inspector)
confirms with **"Delete {name}? This cannot be undone here."** (**Delete** /
**Cancel**) and reports "Deleted."

### Importing a character card

**Import** in the Library rail opens **"Import Character Card"** with filter
groups "Character Cards", "JSON Files", "Card Images (PNG/WebP)", "Markdown
Files" and "All Files". What actually reads: **PNG cards** carrying the
standard embedded card data (older and v3 flavours), **WebP cards** carrying
it in their EXIF comment, and **text cards** — `.json`, `.yaml`, `.md` — as
raw JSON, as front matter, or as a fenced JSON block; common exports from
other apps are recognised and converted on the way in. Other image types are
refused ("Use PNG or WebP cards."). Anything else that goes wrong gives one
deliberately plain message: **"Character import failed; verify the file and
retry."** — details go to the log, not the toast.

On success: **"Character imported."**, and the list jumps to the new row; if
the card carried a lorebook the toast gains **" Lorebook '{name}' attached
(N entries)."** If that name already existed you get **"Character already
existed; selected it. Re-importing does not update an existing character."**
— to take a newer version of a card, delete the old character first or import
under a different name.

### Exporting

The Inspector offers **Export JSON** (characters and personas) and **Export
PNG** (characters only), opening "Export as JSON" / "Export as PNG" with the
name pre-filled; the JSON export carries the avatar with it. Results read
"Exported to {path}" or "Export failed: …". Refusals: "Select a saved item
before exporting." and "PNG export is only available for characters."; while a
selection has unsaved edits both buttons are disabled with the tooltip "Save
before using this action; the selection has unsaved edits."

### Handing a character to Console

Two Inspector buttons, and they are not the same thing:

| | **Attach to Console** | **Start Chat** |
|---|---|---|
| Staged prompt | "Use {name} to guide the next response." | "Respond as {name}." |
| Toast | "Staged in Console." | "Chat staged in Console." |
| Blocked when the model isn't ready? | No | Yes — "Start Chat blocked: {reason}" |

Both carry the character's Name plus its non-empty Description, Personality,
Scenario and System prompt. **Attach** only stages context for you to send
later, so it stays available even when the model isn't ready; **Start Chat**
opens a chat, so it waits for a ready model. The Inspector's readiness line
says which — "Console ready", "Console blocked: select an item", or the
reason — and neither button works on an unsaved selection.

### Dictionaries and World Books copied into a character

The **"Dictionaries (copied into this character)"** panel lists the chat
dictionaries travelling with this character (columns "dictionary" and
"entries"), with **Attach dictionary…** and **Detach**, and reads "No
dictionaries attached to this character yet." when empty. Each attached
dictionary is a **snapshot** — editing the original later does *not* update
the copy inside the character; re-attach to refresh. See
[Chat dictionaries](chat-dictionaries.md).

The **"World Books (copied into this character)"** panel works identically for
lore (columns "world book" and "entries", buttons **Attach world book…** and
**Detach**, empty state "No world books attached to this character yet.") and
carries the same snapshot warning. See [Lore books](lore-books.md).

### Personas

Switch to the **Personas** chip. A persona is deliberately thin: the card is
headed **"Persona Profile"** and shows **Name**, **Description** and **System
prompt** with an **Edit** button. The **"Persona Editor"** holds **Name**
(placeholder "Persona name"), **Description**, **System prompt**,
**Personality traits**, **Mode** — two choices, "session_scoped" and
"persistent_scoped", shown exactly like that
(its two values are saved with the persona, but nothing in the local app
reads them today — the field belongs to the server-side persona schema) — and an
**Enabled** switch that starts on, over **Save** / **Cancel**. Save toasts
"Persona saved."; the unsaved-changes dialog behaves as it does for
characters. What is *missing* compared to characters: **no Import** and **no
Duplicate** (both hidden in this mode) and **no Export PNG** — only Export
JSON. Attach to Console and Start Chat work the same way, staging the
persona's Description and System prompt.

### Placeholders in your text

Character text traditionally uses placeholders, and this app resolves them —
but **nothing in the app tells you so**: no tooltip, hint or help line
anywhere on this screen mentions them, so this guide is where you learn them.

- `{{char}}`, `{{character}}`, `{{persona}}` and `<CHAR>` all become the
  **character's** name.
- `{{user}}` and `<USER>` become **your** name.

Note the trap: `{{persona}}` resolves to the *character*, never to you. In the
previews on this screen, `{{user}}` comes out as the literal word "User".

## Common tasks

1. **Write a character from a one-line concept.** Library rail **New** (or
   **Ctrl+N**) → **"Generate whole character…"** → type the concept, e.g. a
   drowned-library archivist → **Draft**. Read what landed in each field, fix
   what you dislike, **Save**.
2. **Redraft one field.** **Edit**, press **Generate** on the field, read the
   "Generated … - review before applying" panel, then **Accept**,
   **Regenerate** or **Discard**.
3. **Add an alternate greeting.** **Edit** → **Advanced ▸** → type it into the
   box under "Alternate greetings" → **Add** → order it with **Move up** /
   **Move down** → **Save**.
4. **Give the character a picture.** **Edit** → **Upload** on the avatar row →
   pick a PNG/JPG/WEBP/GIF under 5 MB → **Save** (nothing is stored until you
   do). Or **✨ Generate**, once the Description is written.
5. **Import a character card.** Characters mode → **Import** → choose a PNG,
   WebP or JSON card. The new character is selected for you; if a lorebook
   came with it, the toast says so.
6. **Start a chat as the character.** Select it, then Inspector →
   **Start Chat** ("Chat staged in Console."). If it is blocked, read the
   readiness line and fix the model — or use **Attach to Console**, which
   needs no ready model.
7. **Make a persona.** **Personas** chip (**Ctrl+2**) → **New** → fill in
   Name, Description and System prompt → **Save**.

## Keyboard & commands

| Key | Action |
|---|---|
| Ctrl+N | New character (or new persona, in Personas mode) |
| Ctrl+F | Jump to the Library rail's search box |
| Ctrl+S | Save — works in both the character editor and the persona editor |
| Escape | Cancel the open editor (the **Cancel** path, unsaved guard included) |
| Ctrl+Enter | Attach the selection to Console (does nothing when Attach is unavailable) |
| Ctrl+1 / Ctrl+2 | Switch to the Characters / Personas mode |

**Ctrl+1 – Ctrl+4 do not change screens here** — they switch modes. To leave,
click a nav label, use **Ctrl+P**, or use Ctrl+5 … Ctrl+0.

## Related settings & docs

- [Roleplay & Chat Dictionaries](../roleplay-chat-dictionaries.md) — the
  parent screen: the Modes strip, the Library rail, the Inspector, and the
  unsaved-changes guard.
- [Chat dictionaries](chat-dictionaries.md) and [Lore books](lore-books.md) —
  the two things you can copy into a character.
- [Chat basics](../console/chat-basics.md) — where an attached or started
  character actually gets used.
- [World & lore books](../../Features/World-Lore-Books-Documented.md) — deep
  dive on the lore format (its walkthrough describes a retired screen; read
  it for concepts only).
- **There is no feature deep-dive for character cards.** Nothing else in the
  documentation covers the card format, the editor, or the import paths —
  this page is it.
- Avatar and expression generation needs an image-generation backend
  configured in Settings; the refusal toast names the setting. Guide index:
  [index](../index.md).

## Quirks & troubleshooting

- **Duplicate has no confirmation.** One click makes "{name} (copy)";
  delete the copy if it was a mistake.
- **Re-importing does not update an existing character** — the import is
  skipped with "Character already existed; selected it." Delete the old one
  first, or rename before importing.
- **Import failures are deliberately vague.** "Character import failed; verify
  the file and retry." covers every cause; check the file really is a
  PNG/WebP card with embedded data, or valid JSON/YAML.
- **Expression buttons need a saved character** ("Save the character to add
  expressions."); the avatar buttons do not.
- **Blank greetings warn but don't block** — the save goes through, and an
  empty greeting simply does nothing in a chat.
- **Connected to a server, the character actions go quiet.** New, Import,
  Duplicate, Edit, Export and Delete all disable themselves with no
  explanation on screen. Nothing is broken — those actions are local-only.
- **Validation lines look technical while you type.** The as-you-type list
  prefixes each line with the form field's internal name; the list you get
  when Save is blocked is the readable one.
- **Nothing explains the placeholders.** `{{char}}` / `{{user}}` and their
  aliases work, but no tooltip or hint in the app mentions them — see
  [Placeholders in your text](#placeholders-in-your-text).
- **Alternate greetings are a list, not a text box.** Several greetings typed
  into the scratch box become *one* greeting; press **Add** once per greeting.

—
*Verified against dev @ 207053253 — 2026-07-31*

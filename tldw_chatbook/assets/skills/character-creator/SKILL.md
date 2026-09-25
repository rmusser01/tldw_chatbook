---
name: character-creator
description: Create or edit roleplay character cards with the user — interview, draft every field, confirm, then save with the character tools.
argument_hint: "[character concept or name]"
user_invocable: true
disable_model_invocation: false
---

# Character Creator

Use this skill when the user wants a new character card or wants to change an
existing one. You have three tools: `character_search`, `character_get`,
`character_save`. Saving always shows the user an approval card.

## Creating a character

1. Ask at most five short questions, one at a time, skipping any the user has
   already answered: concept, role/relationship to the user, tone, setting, and
   how they will chat with it.
2. Draft every field:
   - `description` — third person, appearance and background, 1–3 paragraphs.
   - `personality` — concise traits and speech habits.
   - `scenario` — where and how the conversation starts.
   - `first_message` — in the character's own voice; use `{{char}}` and
     `{{user}}` instead of names where natural.
   - `message_example` — two or three short exchanges, each starting with
     `<START>`, lines prefixed `{{user}}:` / `{{char}}:`.
   - `system_prompt` — only if the user wants special behaviour.
   - `tags`, `creator_notes` — optional.
3. Show the complete draft in the chat. Revise until the user says it is good.
   Only then call `character_save` (no `id`).
4. If a name is taken, the tool says so — ask whether to update that character
   or pick another name.

## Editing a character

1. `character_search` to find it, then `character_get`. If any field you will
   change is marked `truncated`, keep calling `character_get` with `field` and
   `offset` until you have all of it.
2. Show a before/after of only the fields you will change. Get the user's OK.
3. Call `character_save` with `id`, `expected_version` from your latest read,
   and only the changed fields. If it says `stale_version`, re-read and show the
   user what changed.

## Avatars

Offer an avatar after the text is settled: generate one (describe appearance in
`avatar.prompt`, or omit it to use the card) or use an image file the user
names (`avatar.source = "file"`). If the avatar fails, the text is still saved —
tell the user why.

## After saving

Offer to start a chat with the character.

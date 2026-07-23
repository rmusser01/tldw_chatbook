# Skills Runtime — Codex-Style `$`-Mention Invocation

**Status:** Approved design (2026-07-23).
**Program:** Skills-install program. Follows Spec 1 (trust foundation, PR #762) and
Spec 2 (full-bundle fidelity, PR #784), both merged. This spec migrates *user*
skill invocation in Console chat to the Codex convention; the sibling runtime
concern (fork reference-file reachability) is the next spec.

## Problem

Console skill invocation today is a whole-message slash command:

- The message must **start** with `/skill-name` (`_apply_skill_substitution`,
  `console_chat_controller.py:1187` checks `content.startswith(COMMAND_PREFIX)`);
  everything after the first token becomes `{{args}}`.
- On match, **inline** mode replaces the *entire* message with the rendered body;
  **fork** mode drops prior context and runs the body as a sub-agent.
- A skill mentioned mid-prose (`summarize this, then /style-guide it`) is not
  recognized at all, and there is no way to expand a skill *into* surrounding
  text — the invocation consumes the whole message.

Meanwhile the ecosystem has converged on a different convention.
[Codex](https://developers.openai.com/codex/skills) splits invocation into:
**`/skills`** (a browse/picker command) and **`$skill-name`** (a *mention*,
usable **anywhere in the prompt** — "Use `$brainstorming` to explore ideas…"),
plus implicit model-driven activation. The `$` sigil also solves the detection
problem an embedded `/` cannot: mid-prose `/` is everywhere (paths, dates,
URLs, `and/or`), while `$word` is rare and unambiguous.

## Decision (user-approved)

Adopt the **full Codex model**: `$skill-name` becomes THE user invocation;
`/skill-name` invocation is **hard-removed**; `/skills` remains as the browse
picker. Leading `$skill-name args…` keeps the `{{args}}` capability; embedded
mentions are argless.

## Design

### 1. Invocation model

**`$skill-name`** is the only user-typed skill invocation, with two
position-dependent forms:

| Form | Example | Behavior |
|------|---------|----------|
| **Leading** — message starts with `$skill-name` | `$pdf-processing extract the tables` | The arg-bearing whole-message form. Trailing text → `{{args}}`. **inline** mode replaces the message with the rendered body; **fork** mode takes over the turn (drops prior context except a leading system message) — exactly today's leading-command semantics, re-sigiled. |
| **Embedded** — mention anywhere else in the message | `summarize the draft, $style-guide it, then list open questions` | **Argless** expansion (`{{args}}` renders empty). The rendered **inline** body is spliced **at the mention's position**; all surrounding prose (before, after, between mentions) is preserved verbatim. Multiple embedded mentions each expand. |

The **sigil** disambiguates invocation (`$`) from commands (`/`); position on
`$` disambiguates args-bearing (leading) from argless (embedded). This matches
Codex's own usage: `$skill-installer linear` (leading, args) vs. "use
`$brainstorming` to explore" (embedded, argless).

### 2. Mention detection (the safety rules)

- A candidate token is `$` followed by the longest run of `[a-zA-Z0-9-]`.
  Trailing punctuation stays prose (`$style-guide.` → token `style-guide`).
- A token expands **only** on an **exact, case-insensitive** match to a known,
  **user-invocable** skill name. No prefix matching for mentions (the leading
  form may keep the resolver's unique-prefix behavior; embedded never guesses).
- Everything else stays literal text: `$5`, `$100`, `$PATH`, `$not-a-skill`.
  (Residual edge, accepted: a skill literally named `5` or `path` would make
  those tokens expand — the exact-match gate is the protection; an escape
  syntax for a literal `$known-skill-name` is out of scope.)
- **Fork skills cannot be embedded.** A fork skill takes over the turn; there
  is nothing to splice into. An embedded mention of a fork skill is left
  **literal** (no expansion, no error). A *leading* `$fork-skill args` runs
  fork as today.
- **Trust.** Every expansion re-verifies through `execute_skill` at payload
  build time (today's discipline, per mention). A **leading** mention of an
  untrusted/blocked skill refuses the send with `SKILL_UNTRUSTED_REFUSE`
  (unchanged semantics). An **embedded** untrusted mention is left literal and
  a system note names the skipped skill(s) — the surrounding prose is real
  user content and must not be lost to a refusal.

### 3. `/skill-name` invocation: hard remove

The direct `/skill-name` invocation is removed outright. A message leading
with `/pdf-processing …` is **no longer a skill invocation**: the skill claim
in the command grammar (`console_skill_resolver`'s `CommandParse` claiming) is
removed, and such text receives whatever treatment any other unknown
slash-leading text gets today (it is NOT intercepted, NOT redirected, and NOT
expanded). No transition shim, no redirect note — a clean break, per decision.

**`/skills` stays** as the registered browse command in its **bare** form
(list the available skills). Its `/skills <name> [args]` *run* form is removed
along with `/skill-name` (same hard-remove treatment); running a skill is
`$name`'s job now. The picker's "submit a skill" affordance (chat_screen's
skill-pick → submit path) composes a `$name` message instead of `/name`.

### 4. Autocomplete / composer surfaces

Typing **`$`** in the Console composer surfaces skill-name completion
(candidates = user-invocable skills), replacing the `/`-triggered skill
completion. `/`-completion continues to offer registered *commands* (including
`/skills`) but no longer offers skill names.

### 5. Where it lives (touched surfaces)

- `tldw_chatbook/Chat/console_skill_resolver.py` — resolver gains the mention
  scanner (token extraction + exact-match rule); the `/`-command
  `CommandParse` skill-claiming is removed.
- `tldw_chatbook/Chat/console_chat_controller.py` —
  `_apply_skill_substitution` (the render-fresh choke point on every send
  path: fresh, retry, continue, regenerate) reworked: parse leading `$` form
  (args → `{{args}}`, inline-replace / fork-takeover), then scan-and-splice
  embedded mentions; per-mention trust verification; untrusted-embedded system
  note.
- `tldw_chatbook/UI/Screens/chat_screen.py` — `/skills` command handler:
  bare-list kept, `<name> [args]` run form removed; skill-pick submit path
  composes `$name`; composer completion moves skill names to the `$` trigger.
- The **model-invokes-skill-as-tool** path (`console_agent_bridge`,
  `_BridgeSkillRunner`) is untouched — that is Codex's *implicit* invocation
  equivalent and already exists.

### 6. Rendering details

- Embedded splice uses the skill's **inline** rendered body (front-matter
  stripped, `{{args}}` → empty string), inserted in place of the `$token`
  text. No added markers or fences around the spliced body (the body is
  prompt-text, as today).
- Splicing happens on the **ephemeral provider payload** only — the stored
  transcript keeps the user's raw text with `$mentions`, exactly as the
  leading command does today (render-fresh at payload build, never persisted
  expansion).
- A message that is *only* a mention (`$style-guide`) is the leading form with
  empty args — inline-replace (equivalent outcome to a splice of the whole
  message).

## Testing strategy

- **Resolver/scanner unit:** token extraction (trailing punctuation, hyphens,
  `$5`/`$PATH`/unknown → literal); exact-match-only for embedded; leading form
  arg split preserved; case-insensitivity.
- **Substitution unit:** single embedded mention splices in place preserving
  surrounding text; multiple mentions all expand; embedded fork → literal;
  embedded untrusted → literal + system note (prose not lost); leading
  untrusted → refuse (unchanged); leading `$skill args` → `{{args}}`
  substitution with inline-replace AND fork-takeover; `{{args}}` empty for
  embedded.
- **Removal:** `/skill-name` no longer resolves as a skill (message passes
  through as ordinary text); `/skills` bare list unchanged; `/skills <name>`
  run form removed; skill-pick submit composes `$name`.
- **Completion:** `$` surfaces skill candidates; `/` no longer offers skill
  names but still offers `/skills`.
- **Regression:** existing tests that pin `/skill-name` invocation flip to the
  `$` forms (they encode the removed convention); dictionary/lorebook
  substitution ordering on the same payload path unaffected.

## Out of scope (later specs)

- Embedded args (a delimiter syntax such as `$name{args}`).
- Fork reference-file reachability (the `skill_file` scoped-read layer — next).
- Implicit model-driven activation changes (exists via the agent bridge).
- An escape syntax for a literal `$known-skill-name` in prose.
- Any change to trust, storage, or import subsystems.

## Migration / compatibility

Breaking by decision: users who typed `/skill-name` must switch to
`$skill-name` (the `/skills` picker now composes `$name`, aiding discovery).
No data migration; stored transcripts are untouched (raw text was always
persisted). Release note calls out the new convention and its Codex parity.

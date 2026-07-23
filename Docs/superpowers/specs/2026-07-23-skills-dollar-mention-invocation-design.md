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
- An **embedded** token expands **only** on an **exact, case-SENSITIVE** match
  to a known, **user-invocable** skill name. Skill names are guaranteed
  lowercase (`SKILL_NAME_PATTERN`), so requiring the exact lowercase token
  keeps the entire shell-variable class (`$PATH`, `$HOME`, `$USER`) literal
  *even when* a skill shares the name (skill `path` ≠ prose `$PATH`). No
  prefix matching for mentions. The **leading** form keeps the resolver's
  forgiving behavior (case-insensitive, unique-prefix) — it is an explicit
  command position.
- Everything else stays literal text: `$5`, `$100`, `$PATH`, `$not-a-skill`.
  (Residual edge, accepted: a skill literally named `5` would make `$5`
  expand — the exact-match gate is the protection; an escape syntax for a
  literal `$known-skill-name` is out of scope.)
- **Code spans are skipped.** Mentions inside markdown fenced blocks
  (``` ``` ```) and inline backtick spans stay literal — users routinely paste
  shell/code full of `$vars`, and a pasted `$build`/`$test` must never splice
  a skill body into their code. (Plain-prose pastes containing an exact
  lowercase skill token can still expand — accepted, made rare by the
  case-sensitive exact-match gate.)
- **No recursive expansion.** Splicing is a single pass over the user's
  original text; a `$mention` inside a spliced skill body is NOT expanded.
- **Fork skills cannot be embedded.** A fork skill takes over the turn; there
  is nothing to splice into. An embedded mention of a fork skill is left
  **literal** (no expansion, no error). A *leading* `$fork-skill args` runs
  fork as today. **Mode-detection mechanism:** `SkillCommandCandidate`
  carries only `name`/`description` (no execution mode), so the embedded scan
  calls `execute_skill` per mention — which it needs anyway for trust
  re-verification — and leaves the mention literal when the result's
  `execution_mode != "inline"` (the discarded fork render is side-effect-free).
- **Trust.** Every expansion re-verifies through `execute_skill` at payload
  build time (today's discipline, per mention). A **leading** mention of an
  untrusted/blocked skill refuses the send with `SKILL_UNTRUSTED_REFUSE`
  (unchanged semantics). An **embedded** untrusted mention is left literal and
  a system note names the skipped skill(s) — the surrounding prose is real
  user content and must not be lost to a refusal.

### 3. `/skill-name` invocation: hard remove

The direct `/skill-name` invocation is removed outright — at **both** layers
that claim it today (verified in code):

1. **Composer layer:** the fallback resolver
   (`console_skill_resolver.make_skill_fallback_resolver`, registered into
   `console_command_grammar`'s `ConsoleCommandRegistry`) claims `/skill-name`
   drafts as `KIND_FALLBACK` at parse time. This registration is removed.
2. **Controller layer:** `_apply_skill_substitution`'s leading-`/` branch
   (`content.startswith(COMMAND_PREFIX)` → `_split_skill_command_word` →
   `resolve_skill_command`) is replaced by the `$` parsing of §1/§2.

Post-removal UX (this is the grammar's existing unknown-command behavior, now
precisely stated): a draft leading with `/former-skill-name` parses as
`KIND_UNKNOWN` → the composer shows the standard **"Unknown command" hint**,
and a **second Enter (armed)** sends the draft as literal text. Feedback, not
silence — but no shim, no redirect note, per decision.

Accepted consequences (documented, not bugs):
- **Historical transcript turns** whose raw persisted text is an old
  `/skill-name args` command no longer re-expand on retry/continue/regenerate
  — they are sent as literal text (only `$` forms render-fresh now).
- The composer-level **pre-send blocked-skill hint** (the `KIND_UNKNOWN`
  blocked-match response that told a user a typed skill was needs-review
  before sending) goes away with the fallback resolver; an untrusted leading
  `$name` is still refused at payload build with `SKILL_UNTRUSTED_REFUSE`
  (a system row), which remains the authoritative gate.

**`/skills` stays** as the registered browse command in its **bare** form
(list the available skills). Its `/skills <name> [args]` *run* form is removed
along with `/skill-name` (same hard-remove treatment); running a skill is
`$name`'s job now. The picker's "submit a skill" affordance (chat_screen's
skill-pick → submit path) composes a `$name` message instead of `/name`.

### 4. Discovery surfaces

Verified: there is **no interactive skill-name autocomplete today** — the
composer's only skill-discovery surfaces are the `/skills` bare list and the
unknown-command hint. This spec therefore changes copy, not completion
machinery: the `/skills` transcript listing (`format_skills_list`) renders
**`$name`** rows, and any copy that teaches the `/name` form updates
likewise. An interactive `$`-triggered autocomplete popup is an explicit
**out-of-scope follow-up**, not part of this migration.

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
  composes `$name`; composer completion moves skill names to the `$` trigger;
  the fallback-resolver registration is removed.
- `console_skill_resolver.format_skills_list` — the `/skills` transcript
  listing renders `/{name}` rows today; they become `$name` rows (any other
  user-visible copy that teaches the `/name` form updates likewise).
- The **model-invokes-skill-as-tool** path (`console_agent_bridge`,
  `_BridgeSkillRunner`) is untouched — that is Codex's *implicit* invocation
  equivalent and already exists.

**Payload-pass ordering (verified, must be preserved):** all four send sites
run `_apply_skill_substitution` → `_apply_chat_dictionaries` →
`_apply_world_info`, so a spliced skill body is subject to the downstream
dictionary/world-info passes exactly like today's replaced message. The one
site that deliberately skips skill substitution ("may execute skills with
side effects") continues to skip it — the embedded scan lives inside
`_apply_skill_substitution`, so that site stays skill-free automatically.

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
  `$5`/`$100`/unknown → literal); embedded is exact-match **case-sensitive**
  (`$PATH` stays literal even with a skill named `path`; `$path` expands);
  leading form keeps case-insensitive/prefix resolution + arg split; code-span
  skip (mentions inside ``` fences and inline backticks stay literal); no
  recursion (a `$mention` inside a spliced body does not expand).
- **Substitution unit:** single embedded mention splices in place preserving
  surrounding text; multiple mentions all expand; embedded fork → literal
  (via `execution_mode` from the per-mention `execute_skill`); embedded
  untrusted → literal + system note (prose not lost); leading untrusted →
  refuse (unchanged); leading `$skill args` → `{{args}}` substitution with
  inline-replace AND fork-takeover; `{{args}}` empty for embedded.
- **Removal:** the fallback-resolver registration is gone — a `/former-skill`
  draft parses `KIND_UNKNOWN` (hint, then armed literal send); the
  controller's leading-`/` branch no longer expands (a historical raw
  `/name args` turn retried sends literally); `/skills` bare list unchanged;
  `/skills <name>` run form removed; skill-pick submit composes `$name`;
  `format_skills_list` rows render `$name`.
- **Completion:** `$` surfaces skill candidates; `/` no longer offers skill
  names but still offers `/skills`.
- **Regression:** existing tests that pin `/skill-name` invocation flip to the
  `$` forms (they encode the removed convention); the verified
  skills → dictionaries → world-info payload-pass ordering is pinned; the
  skip-skills send site stays skill-free.

## Out of scope (later specs)

- Embedded args (a delimiter syntax such as `$name{args}`).
- An interactive `$`-triggered skill-name autocomplete popup (none exists for
  `/` today either; copy-only discovery in this spec).
- Fork reference-file reachability (the `skill_file` scoped-read layer — next).
- Implicit model-driven activation changes (exists via the agent bridge).
- An escape syntax for a literal `$known-skill-name` in prose.
- Any change to trust, storage, or import subsystems.

## Migration / compatibility

Breaking by decision: users who typed `/skill-name` must switch to
`$skill-name` (the `/skills` picker now composes `$name`, aiding discovery).
No data migration; stored transcripts are untouched (raw text was always
persisted). Release note calls out the new convention and its Codex parity.

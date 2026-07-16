# Skills — Library home + Console/agent invocation — design

Date: 2026-07-14 (approved in-session by user)
Status: Approved design; implementation plan to follow.

Sub-project 2 of the agent-runtime program (task-200). Builds on the shipped vertical slice
(#620 prompts + slash grammar, #623 engine + ToolProvider seam, #629 agent-capable Console).
MCP (task-201, parallel session) is the third plug into the same ToolProvider socket.

## Purpose

Give skills a first-class management home in the Library (the standalone Skills tab is retired)
and make trusted skills usable from the agent-capable Console on **both** surfaces the SKILL.md
format encodes: users trigger them (`/skill-name [args]` / `/skills <name> [args]`) and the
agent discovers and calls them mid-run as tools. Both are thin adapters over shipped machinery.

## Decisions locked with the user

- **Both invocation surfaces, flag-gated per skill**: `user_invocable` gates the slash surface;
  `disable_model_invocation` gates the agent-tool surface.
- **Home: Library ▸ Skills** (Prompts precedent) — Browse rail row `Skills (N)`, Create ▸ New
  skill, Console-parity canvases; the `skills` top-nav tab retires behind a `skills → library`
  route alias.
- **Trust UX: refuse + point to Library.** Invoking an untrusted skill (new, unsigned, or
  modified since approval) from the Console produces an inline transcript error naming the skill
  and the reason, pointing to Library ▸ Skills — nothing executes. Approval (diff review +
  passphrase, the existing `skill_trust_*` flow) lives only in the Library.
- **Review corrections (design-review pass, all folded in):**
  1. Skills-as-tools route through the **spawn path**, never plain `invoke()` (budgets/cancel/DB
     lineage — see Architecture).
  2. **Render-vs-persist**: the raw command is what's stored and displayed; the rendered body
     substitutes into the provider payload for that turn only; retries re-render + re-trust-check.
  3. Skill `model` override **deferred** (ignored in v1; a marker notes when a skill declares one).
  4. `context: fork` = **clean-context primary run** (no conversation history), not a literal
     sub-agent; `inline` = history + rendered turn.
  5. Built-in commands and built-in tool names always shadow skills (registration order);
     the Library editor warns on shadowing names.

## Architecture — units

- `tldw_chatbook/Library/library_skills_state.py` — pure list/detail state builders (mirror
  `library_prompts_state.py`): rows carry name, description, trust glyph (✓ trusted / ⚠ needs
  review), and an invocation-flags line (`user · agent`, per the two flags). The list renders
  BOTH populations the scope service returns — trusted skills and `blocked_skills`
  (needs-review): blocked rows cannot invoke but open the detail with the trust panel primed
  (the review flow depends on them being visible).
- `tldw_chatbook/Widgets/Library/library_skills_canvas.py` — list + detail canvases. Detail =
  SKILL.md editor: frontmatter fields (name, description, argument_hint, allowed_tools,
  user_invocable, disable_model_invocation, context, model — model shown with a "not applied in
  v1" hint), body TextArea, supporting-files list (read-only names + sizes in v1; add/remove via
  import), and the **trust panel**: current trust state, what-changed diff, approve (passphrase
  modal — reuse `SkillTrustPassphraseModal`). Revoke is DEFERRED: the shipped trust
  service exposes approve + diff but no revoke/untrust primitive — editing a skill already
  returns it to needs-review, which covers the practical case. Explicit Save with the established
  save-outcome/conflict discipline; saving a trusted skill marks it needs-review (content hash
  changed) — the editor says so before saving.
- `tldw_chatbook/Chat/console_skill_resolver.py` — pure-ish resolver registered on
  `console_command_grammar`'s fallback hook (the seam #620 shipped with zero resolvers):
  bare `/skill-name [args]` resolves exact-ci → unique-prefix against user-invocable trusted
  skills; `/skills` becomes a REGISTERED command — `/skills <name> [args]` invokes,
  bare `/skills` lists available skills (name + one-line description) as a transcript system row.
  Unknown words still fall through to the existing unknown-command hint. Ambiguity opens the
  picker (ConsolePromptPickerModal pattern, skill mode).
- `Agents/tool_catalog.py` gains `SkillToolProvider`: `list_catalog()` = trusted,
  model-invocable skills (id `skill:<name>`, the skill name as the model-facing tool name,
  description as the one-liner); `load_schema()` = a single-string schema
  `{"args": {"type": "string", "description": <argument_hint or description>}}`;
  `invoke()` raises by design (see next bullet).
- **Per-run spawn wiring (review correction 1):** `agent_service` (or the Console bridge),
  when assembling a run's registry/deps, wraps each skill catalog entry in a run-scoped
  executor: an agent calling a skill tool routes through **that run's spawn machinery** —
  budget-counted (`max_subagents`), cancel-fan-out via `should_cancel`, child AgentRunsDB row
  with `parent_run_id`, result capped like any sub-agent. The pure loop and the ToolProvider
  protocol are unchanged; `SkillToolProvider.invoke()` existing only to satisfy the protocol
  and erroring loudly guards against future misuse outside a run context.
- Console integration in `chat_screen.py`/`console_chat_controller.py`: the slash surface
  renders via `skills_scope_service.execute_skill` (which already enforces trust +
  exact-content hash) and drives the agent turn with a narrowed `AgentConfig`.

## Invocation semantics

**Slash surface** (`/skill-name args` or `/skills name args`):
1. Resolve (user-invocable + trusted only). Untrusted → the refuse transcript row; nothing runs.
2. `execute_skill` renders the body (`{{args}}` substituted; args validated non-pathological
   before substitution).
3. The turn runs through the normal agent reply engine with a per-turn `AgentConfig`:
   `allowed_tools = skill.allowed_tools ∩ runtime builtins` (a skill can only narrow, never
   grant); `context: inline` → provider payload = conversation history + the rendered body as
   the turn; `context: fork` → provider payload = the rendered body only (clean context).
4. **Transcript + persistence carry the raw command** as the user message (compact, honest),
   plus a marker row naming the skill (`skill code-review → driving this turn`, TOOL-role,
   escaped). Subsequent turns' history therefore contains the raw command — an accepted,
   documented trade-off.
5. **The substitution rule (one rule for fresh sends AND retries):** when building a provider
   payload, if the turn's TRIGGERING user message — the final user message in the payload —
   parses as a resolvable skill command, it is rendered fresh at build time (re-resolve,
   re-trust-check, `{{args}}` re-substituted, `context` re-read so fork/inline survives
   retries). Earlier history messages are never substituted. The stored raw command is thus the
   durable record that drives re-rendering — no per-message skill metadata needed. A skill
   edited since approval (now untrusted) refuses at retry with the standard refuse copy.

**Agent-tool surface** (mid-run): the agent sees trusted model-invocable skills in its catalog
(subject to the run's `allowed_tools` and progressive disclosure), loads one, calls it with an
args string → the run-scoped executor renders the skill and runs it as a **sub-agent** (clean
context, the skill's narrowed tools minus spawn, result string back). Skills cannot call skills
in v1 (a skill's own run excludes skill tools — depth stays bounded).

## Security model

- **Trust gates twice**: catalog/resolve time (untrusted skills invisible to the agent and
  refused for users) AND execution time — `execute_skill`'s existing `_require_trusted_skill` +
  `_verify_exact_skill_content` hash re-verification runs on every render, so a skill modified
  between listing and invoking refuses (no rug-pulls). The `agent_service` permission gate
  additionally requires skill tools to be disclosed + allowed like any tool.
- `allowed_tools` **intersects** with the runtime's builtin set — never grants.
- **Run allowlist composition**: a normal Console run's `AgentConfig.allowed_tools` =
  builtins ∪ eligible skill names (trusted + model-invocable), computed fresh at run start —
  otherwise the disclosed-AND-allowed permission gate would refuse every skill tool. A
  skill-driven turn's intersection rule (`skill.allowed_tools ∩ builtins`) then naturally
  excludes other skills, consistent with no-skills-calling-skills.
- All skill-derived text (names, descriptions, hints, marker copy, list rows) markup-escaped
  where markup is enabled; raw where `markup=False` (the #629 discipline). Args length-capped
  before substitution.

## Catalog scale (lands WITH this work, not after)

Skills are the first realistic >8-tool catalog, so progressive disclosure
(`find_tools`/`load_tools`) goes live here — and the two deferred catalog fixes land as part of
Phase 2, with regression tests: (a) duplicate re-load desync (loop-side dedupe before slicing —
F1-b, plan-a-final-review); (b) `_owner_and_id` per-lookup re-listing (cache the tool_id →
provider map; cache scope is PER RUN — the catalog is listed fresh at run start, so skill CRUD
between runs is always picked up and no cross-run invalidation signal is needed —
task-227-adjacent TODO in tool_catalog.py).

## Error handling

- Empty skill store → `/skills` lists nothing with a quiet "create skills in Library ▸ Skills"
  row; the Library list shows the standard empty state.
- Render failures (missing skill dir, bad frontmatter) → honest transcript error naming the
  skill; the composer draft is preserved.
- A skill-tool sub-agent failing/stuck → the standard sub-agent error result feeds the parent
  (shipped semantics); the run never wedges.
- Import: per-file outcomes; imported skills arrive trust-pending with the review panel primed.

## Testing

- **Pure**: resolver (exact/prefix/ambiguous/untrusted/unknown fall-through, `/skills` listing),
  args substitution + caps, schema derivation, state builders incl. trust glyph + flags line,
  shadow-warning predicate.
- **Service (real trust store + skills dir)**: approve → resolvable; edit → refuses everywhere;
  hash re-verify between list and invoke; intersect semantics; fork vs inline payload shapes;
  run-scoped skill-tool execution produces a budget-counted, parent-linked, cancellable
  sub-agent run.
- **UI (real App + run_test)**: Library rail count, list/detail, trust panel approve/revoke via
  the passphrase modal, save-marks-needs-review warning, refuse copy in the Console, picker,
  marker rows, route alias.
- **Live gate (served, 2050×1240, real llama.cpp)**: `/skill-name` driving a real run (one
  `inline`, one `fork`); the refuse path on an edited skill; the agent discovering and calling
  a skill mid-run **with >8 skills seeded** so progressive disclosure actually engages; the
  catalog-fix regressions. STOP for user approval before PR (per program convention).

## Out of scope (named follow-ups)

Skill `model` override (deferred — needs a one-turn provider/model override path);
trust revoke/untrust (no shipped primitive; edit-marks-needs-review covers the practical case);
skills-calling-skills; supporting-file editing in the detail canvas (import-only in v1);
server-side skills backend surfaces beyond what `skills_scope_service` already routes;
skill marketplace/discovery.

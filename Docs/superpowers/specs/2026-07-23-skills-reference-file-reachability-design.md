# Skills Runtime — Reference-File Reachability (`skill_file` tool)

**Status:** Approved design (2026-07-23), mechanism hardened after two code-verified
runtime traces.
**Program:** Skills-install program. Follows Spec 2 (full-bundle fidelity, PR #784)
and the `$`-mention invocation migration (PR #801), both merged. This spec makes
the faithfully-stored bundles *usable*: an agent can read a skill's
`references/*` on demand.

## Problem

Spec 2 stores a skill's nested `references/`, `scripts/`, and `assets/`
faithfully, but nothing exposes them at runtime. `execute_skill` returns only the
rendered `SKILL.md` body; the Agents runtime's builtin tools are Calculator and
DateTime — **there is no file-read capability anywhere in the agent tool
catalog**. A skill body that says "see `references/api.md`" dead-ends: the
sub-agent that IS the skill cannot read its own bundle, and the main Console
agent cannot read the bundles of skills the user just `$`-mentioned. This spec
adds the first file capability to the agent runtime — which makes tight scoping
load-bearing, not a nicety.

## Decisions (user-approved)

- **Mechanism B**: a skill-scoped read tool (not path-in-prompt, not eager
  inlining) — preserves progressive disclosure and containment.
- **Scope**: model-invoked skill forks AND native-tools user turns whose payload
  was produced by `$skill` invocations. Plain sends (no tool loop) get nothing —
  no tool, no dangling "bundled files" promise.
- **Always granted for a skill's own bundle**: reading your own references is
  the progressive-disclosure baseline, NOT subject to the skill's frontmatter
  `allowed-tools` narrowing.
- **Per-read cap 100 KB** with an explicit truncation marker; text-only.

## Design

### 1. One read seam: `LocalSkillsService.read_skill_file`

`read_skill_file(skill_name: str, relative_path: str) -> dict` with
`{content: str, truncated: bool, size: int}`. Synchronous (the whole agent
runtime runs on a worker thread with no event loop). Per call, in order:

1. `_enforce("skills.read_file.launch.local")` (policy gate, mirroring
   `execute_skill`).
2. `_require_trusted_skill(skill_name)` — trust re-verified at READ time
   (sync; raises `SkillTrustBlockedError`). A skill revoked mid-run stops
   being readable immediately, mirroring the render-fresh discipline.
3. `validate_supporting_file_path(relative_path)` — traversal/bad-name/depth
   rejection (Spec 2's validator).
4. Containment + read via the existing `_read_text_preserving_newlines`
   discipline (resolve, symlink checks on dir and file,
   `get_safe_relative_path` containment, UTF-8 decode) — reused, not
   re-implemented. `SKILL.md` itself is readable (harmless; it is the body).
5. A binary file (null byte / decode failure) returns a clean refusal string
   ("binary file — N bytes; not readable as text"), never bytes, never raises.
6. Output capped at **100,000 characters**; `truncated: True` plus a trailing
   marker line when cut.

**Scope-service passthrough is LOCAL-ONLY**: `SkillsScopeService.read_skill_file`
uses the bespoke-dispatch pattern (like `count_skills`/`delete_skill`) and
rejects `mode="server"` with a clear domain error — never a raw
`AttributeError` from a missing server method. (All existing runtime skill
paths are already hardcoded `mode="local"`.)

### 2. `skill_file` is the FOURTH RUNTIME TOOL (verified mechanism)

Two traced facts force this shape: (a) the tool registry is **per-run and
shared** between the primary agent and its skill-children — a
provider-registered tool cannot distinguish callers and, on the service-owned
registry, would leak across runs; (b) provider-registered tools pass through
disclosure budgets (`initial_disclosure` direct-discloses only when the catalog
is ≤ 8 entries), which could silently hide a capability the rendered prompt
advertises. The runtime tools (`spawn_subagent`/`find_tools`/`load_tools`)
already solve both problems: pinned via `runtime_schemas` (active from step 1,
never disclosure-gated), dispatched by a name-branch in `run_agent_loop`,
authorized by their own logic rather than `config.allowed_tools` membership.

`skill_file(skill_name: str, path: str)` joins them:

- **Schema pinning:** `_run_one` appends the `skill_file` schema to its
  `runtime_schemas` whenever the run's bindings object (below) is non-empty —
  for the primary run AND for a skill-spawned child.
- **Dispatch:** a `skill_file` name-branch (runtime-tool pattern) invoking a
  per-run reader closure; NOT `ToolCatalogRegistry.invoke_by_name`, NOT a
  provider.
- **Authorization: the per-run `SkillFileBindings` object.** A small mutable
  holder created per `run_reply`/agent run: `{authorized: set[str],
  reader: <scope-service read_skill_file>}`.
  - Seeded with the TURN's `$skill` binding set (see §3).
  - `SkillRunner.run` **adds the spawned skill's name** before `spawn(...)`,
    so the sub-agent that IS the skill can always read its own bundle —
    independent of its frontmatter `allowed-tools` (which only ever narrows
    builtins and is untouched).
  - A `skill_file` call naming a skill outside `authorized` → clean
    `ToolResult(ok=False, ...)` refusal. Within one run, primary and children
    share the bindings union — acceptable: every authorized name is a
    trusted, user- or model-triggered skill in this same conversation, and
    the real security boundary is per-call trust re-verification +
    directory containment in §1, not caller identity.
- **Collision exclusion:** `skill_file` joins the composition-time
  collision-exclusion set (the `RUNTIME_TOOL_NAMES` mechanism that already
  keeps skills named `spawn_subagent`/`find_tools`/`load_tools` out of the
  catalog), so a skill literally named `skill_file` is never registered as a
  skill tool.

### 3. Turn bindings: recorded at substitution, threaded to the run

`_apply_skill_substitution` already executes every `$skill` that drives a turn.
It now ALSO returns the set of skill names that actually rendered (leading
resolved name; embedded successfully-spliced names — NOT blocked/fork-literal
mentions): the return widens from 3-tuple to 4-tuple
`(messages, refuse, notes, skill_bindings)` — the same widening idiom the notes
channel used, applied at the same 4 call sites, threaded through the single
shared funnel (`_stream_assistant_response` → `_run_agent_reply` →
`bridge.run_reply(turn_skill_bindings=...)`). `run_reply` seeds the
`SkillFileBindings` object; empty bindings + no skill spawn ⇒ the schema is
never pinned and the tool simply doesn't exist for the run.

Bindings are **per-turn** (a plain follow-up message gets no tool). The bridge's
per-conversation snapshot dicts are the named seam if accumulation is ever
wanted — explicitly out of scope now.

### 4. Discovery metadata travels with capability

`execute_skill` gains an additive `reference_files` field (relative path, size,
`is_text` — from Spec 2's `_read_bundle_manifest`). Only the two tool-injecting
consumers render it into the prompt — a compact block appended to the rendered
body: `Bundled files (readable via skill_file): references/api.md (2 KB), …`.
Plain sends ignore the field (no dangling promise). Binaries are listed with a
`(binary)` marker (listable, not readable).

### 5. Bounds

Per-read 100 KB cap (§1). Tool-call count rides the loop's existing per-run
budgets (no new budget). Reads return text as tool results only — never spliced
into user prose, no recursion concerns.

## Components & boundaries

| Unit | File | Responsibility |
|------|------|----------------|
| Read seam | `Skills_Interop/local_skills_service.py` | `read_skill_file` (policy → trust → validate → contained read → cap); reuses `_read_text_preserving_newlines` + `validate_supporting_file_path` + `_require_trusted_skill` |
| Scope passthrough | `Skills_Interop/skills_scope_service.py` | local-only bespoke dispatch; clean server-mode rejection |
| Bindings object + schema | `Agents/agent_models.py` (or sibling) | `SkillFileBindings`; `SKILL_FILE_TOOL_NAME`/schema constant |
| Runtime dispatch | `Agents/agent_runtime.py` + `Agents/agent_service.py` | pin schema into `runtime_schemas` when bindings non-empty; `skill_file` name-branch → reader closure; sync |
| Fork grant | `Chat/console_agent_bridge.py` (`_BridgeSkillRunner.run`) | add spawned skill's name to bindings before `spawn` |
| Turn bindings | `Chat/console_chat_controller.py` | 4-tuple widening; thread `turn_skill_bindings` through the funnel to `run_reply` |
| Collision exclusion | `Chat/console_agent_bridge.py` (`_non_colliding_skill_entries`) / `RUNTIME_TOOL_NAMES` | a skill named `skill_file` is never catalog-registered |
| Prompt metadata | `Skills_Interop/local_skills_service.py` (`execute_skill`) + the two injecting consumers | additive `reference_files`; "Bundled files" block rendered only where the tool exists |

## Testing strategy

- **Seam unit:** trusted read happy path (nested path); trust-revoked →
  `SkillTrustBlockedError`; traversal (`../`), absolute, bad segment → rejected;
  symlink → rejected; binary → clean refusal string; >100 KB → truncated flag +
  marker; unknown skill/file → clean error; scope service `mode="server"` →
  domain error.
- **Bindings/authorization unit:** call naming an unauthorized skill → refusal;
  authorized → content; empty bindings ⇒ schema not pinned (tool absent).
- **Fork path:** a skill child spawned via `SkillRunner` can read its OWN
  `references/x.md`; cannot read a non-bound other skill; grant independent of
  frontmatter `allowed-tools`.
- **Turn path:** leading `$skill` and embedded splices record bindings
  (blocked/fork-literal mentions do NOT); plain send → no tool; 4-tuple threaded
  at all 4 call sites (mechanical unpack updates).
- **Collision:** a skill named `skill_file` is not registered as a skill tool.
- **E2E:** fork skill whose body references `references/api.md` reads it on
  demand; the "Bundled files" block appears exactly where the tool exists.

## Out of scope (deliberate)

- Script **execution** (`scripts/*` running) — the trust-gated follow-on layer.
- Binary reads / asset serving; per-conversation binding accumulation (seam
  named: the bridge's per-conversation dicts); plain-send fallback listings;
  audit-trail entries for reads; remote/server skills; multimodal-turn bindings
  (multimodal payloads still skip substitution entirely, pre-existing).

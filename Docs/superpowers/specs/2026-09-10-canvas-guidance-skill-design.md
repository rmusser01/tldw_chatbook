# Offer-first Canvas guidance and companion skill

Date: 2026-09-10 (America/Los_Angeles)
Status: Written spec approved by the user on 2026-09-10
Baseline: locally available `origin/dev`, `3afa68f1b9`
Scope: design only; implementation planning follows written-spec approval.
Task: [TASK-32312](../../../backlog/tasks/task-32312%20-%20Design-offer-first-Canvas-guidance-and-companion-skill.md)
Decision: [ADR-149](../../../backlog/decisions/149-offer-first-canvas-guide-and-inline-skill.md)
Implementation plan: [Canvas guidance and skill](../plans/2026-09-10-canvas-guidance-skill-implementation.md)

## Purpose

Help the Console assistant recognize when Canvas would improve an answer, offer
that option before spending tokens on authoring, and produce compatible displays
after the user accepts. Reuse Chatbook's existing Canvas renderer, tools,
conversation ownership, and revision lifecycle.

The user approved guidance plus an optional skill, rather than a host-enforced
proposal/acceptance workflow. Compliance is model behavior, not a new security or
billing guarantee. The only new tool is a bounded, read-only documentation tool.

## Existing implementation and decisions

The initiating workspace predates Canvas. Implementation must start from a branch
containing the existing V1 and qualified V2 Mermaid delivery, not recreate them in
that older checkout. This design worktree is based on the baseline above.

- [Canvas tool provider](../../../tldw_chatbook/Agents/canvas_tool_provider.py)
  supplies `canvas_list`, `canvas_read`, `canvas_create`, and `canvas_update`.
- [Console bridge](../../../tldw_chatbook/Chat/console_agent_bridge.py) supplies a
  cheap discovery hint and constructs guidance for disclosed tools. Its
  `_BridgeSkillRunner` launches model-invoked skills as restricted child runs;
  those children do not inherit Canvas authority.
- [Canvas authoring helpers](../../../tldw_chatbook/Canvas/authoring.py) and the
  packaged Mermaid guide supply exact-profile guidance. Source acceptance does
  not prove browser preview success.
- User `$skill` mentions support inline expansion through the existing Console
  controller. Imported local skill trust is rechecked at use time.

Existing governance remains authoritative:

- [ADR-121](../../../backlog/decisions/121-local-versioned-canvas-artifacts-and-browser-sandbox.md):
  Canvas ownership, atomic turns, immutable revisions, sandbox, and browser delivery.
- [ADR-124](../../../backlog/decisions/124-canvas-mermaid-subset-and-immutable-runtime-profiles.md):
  Mermaid subset, exact profiles, qualification, and recovery.
- [ADR-009](../../../backlog/decisions/009-local-skill-trust-boundary.md): imported
  local skill trust and re-verification.

ADR required: yes
ADR path: `backlog/decisions/149-offer-first-canvas-guide-and-inline-skill.md`
Reason: Adds a public documentation-tool contract and records product-documentation
versus imported-skill ownership while preserving the existing Canvas boundaries.

## 1. Offer before proactive authoring

The discovery hint and loaded Canvas tool guidance both carry the same concise
rule. A model must see it before needing to load an authoring guide.

Offer Canvas selectively when a substantial visual, interaction, or iteratively
revised single-page result materially helps. Examples include adjustable
comparisons, calculators, small simulations, and diagrams. Ordinary prose, short
code, and simple tables usually remain in chat. Do not offer merely because a
request contains numbers or mentions a chart.

A proactive offer is one short sentence naming the proposed artifact and its
benefit, for example: “An interactive Canvas would let you adjust the assumptions
and compare outcomes. Want me to create one?” Then wait for the user's answer.
Do not load detailed guide examples, delegate authoring, or generate HTML,
JavaScript, SVG, or Mermaid source for that proposal before acceptance. Normal
work and concise chat answers can continue when they do not depend on the choice.

| User situation | Required behavior |
| --- | --- |
| No Canvas request, material benefit identified | Offer briefly and wait; no speculative artifact source. |
| User accepts the proposed Canvas | Author that artifact in the owning Console conversation. |
| User declines | Continue in chat; do not repeat the same offer unless the user changes direction. |
| Explicit request to create a Canvas | Treat the request as consent; do not ask again. |
| Requested edit to an existing Canvas | Read and revise it without another offer. |
| Bare `$canvas`, or mention without a concrete creation/edit request | Clarify what is wanted; skill activation itself is not consent to generate an artifact. |
| A different unsolicited artifact or substantial unrequested redesign | Make a new offer. |
| No answer or ambiguous response | Do not infer acceptance from elapsed time. |

Consent is artifact-scoped and represented by the conversation, without a new
database field, setting, or persisted approval object. Ordinary bounded corrections
of the accepted artifact are covered; consent is not a standing license to create
unrelated Canvases. If conversation context no longer establishes the requested
work, ask a short clarification rather than assuming consent.

## 2. Main-conversation guide tool

Add `canvas_guide` to the existing scoped `CanvasToolProvider`. Keep creation,
reads, and revisions in the owning Console run. Do not extend skill-child tool
inheritance or introduce a general skill-activation framework.

### Contract

- Tool identity: `canvas:canvas_guide`; public name: `canvas_guide`.
- Parameters: one required `topic` string, closed enum `basics`, `controls`,
  `mermaid`, or `repair`; reject additional properties. No filesystem path,
  source, URL, Canvas ID, or user-supplied document is accepted.
- Success: an ordinary successful `ToolResult` with a JSON object containing
  `status: "ok"`, the requested `topic`, and `guide` (Markdown text).
- Failure: the provider's existing bounded error envelope, with stable categories
  for invalid arguments, unavailable guide data, or unavailable Canvas authority.
  Never return partial/truncated guide text as a successful result.
- The full serialized successful JSON result is at most 12 KiB UTF-8 per topic.
  Validate the packaged resources and the final result against that limit.
- Documentation is lazy-loaded through `importlib.resources` from a fixed topic
  map in the installed Chatbook package. It works from an installed wheel,
  independently of repository files or imported user skills.

The tool returns documentation only: no model call, compiler invocation, preview
launch, selection change, revision staging, or host action. It requires no existing
Canvas selection. Availability uses the existing live Canvas provider registration,
run scope, policy, and kill switch; disabling Canvas must prevent stale invocations
as well as fresh discovery.

Add the name to the appropriate provider-owned reservation, registration,
discovery, policy, and tool-result handling surfaces. Preserve the existing
mutation classification for `canvas_create` and `canvas_update` only. Keep the
four artifact-tool names distinguishable from the guide name where current
code/tests mean “all four V1 tools.” Discovery hints describe only tools that the
run actually offers; a guide must not imply authority to create or update.

The model receives the guide body. Non-model Canvas record projections carry
only bounded topic/status/byte-count metadata for this operation, rather than
duplicating the guide and examples into transcript cards or operational logs.
Extend the existing closed projection contract for this new result shape; do not
route it through private artifact-source handling or make a guide call look like
a newly created Canvas card.

## 3. Companion skill and shared content

Ship an optional importable skill at `Docs/Examples/skills/canvas/SKILL.md`, named
`canvas`. The skill body is at most 4 KiB UTF-8 and contains workflow instructions,
not embedded example documents. Its native frontmatter uses `context: inline`,
`user_invocable: true`, and `disable_model_invocation: true`. Preserve the existing
leading-argument template (`{{args}}`) and embedded-mention behavior. Do not pin a
model or declare an `allowed_tools` field that changes the owning run's tool set.

Import and trust approval use the existing Library flow. Do not silently install,
auto-trust, activate at startup, or overwrite a user's skill with the same name.
Document where to obtain the example folder and how to import/trust it. Installed
users can obtain it from the documented repository skill directory; the built-in
guide resources must themselves be included in the application wheel.

The skill teaches selective offering, consent, guide discovery, profile-aware
authoring, reading before edits, and honest reporting of results. After consent,
it directs the assistant to discover/load `canvas_guide` and use the appropriate
topic, followed by the existing Canvas tools. It never tells the assistant to
invoke itself as a model tool or launch a subagent for Canvas authoring.

Detailed authoring examples live once, in the packaged topic guides consumed by
both the direct tool path and the skill-directed path. The imported skill does
not bundle a second runtime manual. A concise shared product policy supplies
the discovery/runtime offer-first text; the small skill necessarily restates the
behavior and is checked for consistency.

Reading packaged product documentation is not reading an installed local skill.
`canvas_guide` must never resolve a user skill directory, bypass skill trust, or
read arbitrary files. Missing/locked/untrusted `$canvas` follows normal skill
refusal behavior. A normal Canvas request can still use the product guide without
installing a skill; do not silently replace a refused skill invocation in that turn.

## 4. Authoring content and runtime compatibility

Load only the topics needed for the accepted request, and reuse guide text already
in the current context. Do not load all topics by default or repeatedly fetch the
same topic during an unchanged edit. A resumed context may reread missing guidance.
The change does not add all guide bodies to system prompts, skill listings, or
tool descriptions. Existing profile-specific guidance remains in place.

| Topic | Required contents |
| --- | --- |
| `basics` | Complete self-contained document shape; readable responsive layout; labels and keyboard access; simple passive SVG chart; create versus revise; unsupported browser assumptions. |
| `controls` | Small complete calculator/comparison example with bounded input/change handling, explicit element lookup, and visible output updates. |
| `mermaid` | Text-only `pre[data-canvas-diagram="mermaid"]` declarations; exact supported flow/sequence examples aligned with the existing packaged profile guide; escaping, limits, and excluded syntax. |
| `repair` | Read/current-parent update flow; conflict recovery; source acceptance versus preview status; bounded diagnostics; source-only/unavailable-profile preservation. |

Examples use Chatbook's complete HTML document format, not Codex Visualize
fragments. Use inline CSS, supported classic-script DOM/event APIs, and passive
SVG. Do not assume React, D3, Chart.js, CDNs, modules, native `window`, HTML canvas
drawing APIs, CSS custom properties, networking, storage, or arbitrary Mermaid.js
APIs exist. Interactive pages keep state in their allowed JavaScript realm.
`canvas.submit` and `canvas.download` remain requests for the existing confirmed
host actions, not automatic submission or download permissions.

Guides are instructional, not profile-admission authorities. Examples identify
the profile they require. The current tool/profile guidance and runtime checks
take precedence over generic examples, especially on historical updates. Do not
infer profile availability from a guide successfully loading. Preserve source
when a profile is unavailable; adaptation to a current profile follows the
existing explicit-new-Canvas workflow, never silent migration. This work changes
no pinned renderer bytes, profile manifests, grammar, runtime limits, or storage.

## 5. Data flow and recovery

Proactive path: cheap discovery hint → short offer → user accepts → main assistant
loads relevant guide topic(s) → existing create tool → existing turn settlement
and preview behavior. An explicit creation request skips only the offer.

Explicit skill path: user imports/trusts the example once → `$canvas` expands
inline in the current Console turn → the same consent and guide flow. A user
requesting `$canvas` with no task receives a short clarification.

Edit path: list if needed to identify the intended Canvas → read its current
complete selected source and revision ID → load missing relevant guidance →
submit a complete replacement with `expected_parent_revision_id`. On a conflict,
reread and adapt the requested edit instead of overwriting newer work or retrying
the same parent. Existing branch/historical-selection semantics remain intact.

Do not claim success merely because a tool returned a staged revision: existing
atomic turn settlement still determines what persists. Distinguish source saved,
preview pending, preview ready, and preview failed/source-only using only the
evidence available to the assistant. Do not invent a new status-query tool or
claim the assistant can see a browser diagnostic it has not received.

Use explicit tool diagnostics or a user-provided repair request. Existing browser
failures never trigger automatic model submissions. A concrete bounded correction
can continue within the accepted task; after one failed repair attempt, report the
remaining limitation and let the user choose whether to spend more on repair.
Do not generate speculative versions in a loop. Repeated revision conflicts also
require reporting the conflict rather than unbounded retries.

If Canvas tools are unavailable, explain briefly and continue with useful chat
content. Do not create external runnable HTML, fetch libraries, enable settings,
or suggest a different runtime as an automatic workaround. If only the guide is
unavailable, use already available authoritative runtime guidance when sufficient;
otherwise explain the limitation rather than inventing APIs.

## 6. Verification and acceptance evidence

Implementation planning must select focused tests around the changed tool provider,
Console discovery/guidance, projections, skill import/invocation, and packaged data.
Do not run a full repository suite without the user's opt-in.

1. Verify the agreed offer-first rule appears in discovery and loaded guidance,
   including explicit requests, requested edits, refusal, and lack of an answer.
   Verify new topic bodies are absent until requested; existing profile guidance
   need not be removed or loaded a second time.
2. Exercise the real provider/catalog path for valid topics, invalid/extra inputs,
   output byte limits, missing/oversized resources, disabled/revoked Canvas, and
   source-safe guide record projections. Guide reads must not stage revisions,
   change selection, emit Canvas cards, or start preview/model work.
3. Verify package inclusion and reading from a built wheel without a source tree.
   Validate the skill frontmatter, leading arguments, and embedded inline use;
   import/trust it through the real service and prove no child run is dispatched
   and existing tool authority is retained. Verify normal untrusted refusal.
4. Compile all complete guide examples with the real Canvas compiler and the
   intended qualified profiles. Run focused existing browser harnesses to prove
   the primary control interaction changes the display and supported Mermaid
   examples render. Record browser/runtime versions and distinguish compiler
   acceptance from browser execution. Inspect narrow and ordinary viewport
   usability without redesigning the Canvas shell.
5. Perform a small recorded model-behavior review using synthetic conversations:
   proactive offer/no pre-acceptance source, acceptance, refusal/no repeated offer,
   explicit create, requested update, bare skill invocation, and unavailable
   Canvas. Include one repair scenario. Record model/settings, supplied tools,
   observed calls/output, and available usage data. A scripted provider fixture
   verifies plumbing only; prompt assertions do not prove model compliance.
   Use an available authorized test provider and report unrun scenarios honestly
   if none is available. Do not claim guaranteed compliance or quantified token
   savings from these limited observations.

Use the repository's scoped static checks for changed Python and documentation
checks for the spec/skill. Update the Canvas user guide with the offer-first flow,
optional skill installation, and the new guide. Reuse existing Canvas harnesses;
this work does not requalify or widen the underlying runtime.

## 7. Scope and implementation handoff

Deliver one cohesive change: offer-first guidance, one scoped guide tool, shared
packaged topic resources, an optional inline skill, and focused evidence/docs.
No new UI, consent store, settings, skill runtime, database migration, Canvas
renderer, public network API, or dependency is required.

Rejected alternatives: skill-only delivery misses ordinary proactive use;
model-invoked authoring skills add an unnecessary child run without Canvas
authority; enforced proposal acceptance adds UI/state and was declined by the
user; blocking only create/update is too late to prevent generating tool arguments;
always injecting the full manual spends tokens before the user chooses.

The next stage is an implementation plan after written-spec approval. It must
link ADR-149 and ADR-121/124/009, inventory the exact provider
reservation/projection/discovery callers, include wheel packaging and targeted
tests, and preserve the existing immutable runtime assets. This document is the
behavior and contract specification, not authorization to begin implementation.

Independent [spec review](../reviews/2026-09-10-canvas-guidance-skill-spec-review.md)
approved the design without blocking or advisory findings. The user subsequently
approved the written spec on 2026-09-10, authorizing implementation planning.

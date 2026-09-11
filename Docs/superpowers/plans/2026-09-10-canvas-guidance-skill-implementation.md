# Offer-first Canvas Guidance and Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Offer Canvas before proactive authoring, then help the owning Console assistant create compatible displays through a bounded guide tool and optional inline skill.

**Architecture:** Extend the existing scoped Canvas provider with `canvas_guide(topic)` and share a short offer-first policy between discovery and loaded guidance. Read packaged documentation on demand; an optional trusted `$canvas` skill expands inline and points to the same guide. Preserve the four artifact tools, profiles, renderer, turn settlement, and skill trust boundaries.

**Tech Stack:** Python 3.12+, existing agent/catalog interfaces, `importlib.resources`, Markdown, existing pytest and Playwright Canvas harnesses. No new dependencies.

---

## Approved inputs and execution boundary

- [Approved spec](../specs/2026-09-10-canvas-guidance-skill-design.md), approved by the user on 2026-09-10.
- [ADR-149](../../../backlog/decisions/149-offer-first-canvas-guide-and-inline-skill.md).
- Existing [ADR-121](../../../backlog/decisions/121-local-versioned-canvas-artifacts-and-browser-sandbox.md), [ADR-124](../../../backlog/decisions/124-canvas-mermaid-subset-and-immutable-runtime-profiles.md), and [ADR-009](../../../backlog/decisions/009-local-skill-trust-boundary.md).
- [TASK-32460](../../../backlog/tasks/task-32460%20-%20Design-offer-first-Canvas-guidance-and-companion-skill.md) tracks the design and plan, not product delivery.
- Baseline: `323ae22c5a` on `codex/canvas-guidance-skill-design`, based on `origin/dev` at `3afa68f1b9`.

ADR required: yes (already recorded)
ADR path: `backlog/decisions/149-offer-first-canvas-guide-and-inline-skill.md`
Reason: The approved ADR covers this read-only guide contract and optional inline skill; no further architectural decision is introduced by this plan.

This is one feature-sized delivery with five sequential implementation tasks below.
At execution start, create one atomic implementation Backlog task covering spec
sections 1–6, set it In Progress, and attach this plan before changing product code.
Allocate its ID using the repository's collision-check procedure; do not reuse the
documentation task or reserve a guessed future number in this plan. The task's
acceptance criteria must cover consent guidance, bounded scoped guide behavior,
trusted inline skill behavior, packaged resources, working examples, and recorded
verification evidence. Read `backlog/docs/lessons-testing-evidence.md`,
`backlog/docs/lessons-live-verification.md`, and
`backlog/docs/lessons-backlog-hygiene.md` before execution.

Use @superpowers:test-driven-development for new logic,
@superpowers:systematic-debugging for failures, and
@superpowers:verification-before-completion before completion claims. Keep edits
minimal per @ponytail. This plan does not authorize a full repository test sweep.

## File map and important existing callers

Paths below are repository-relative to the isolated design worktree. New test
files are explicitly labeled; named existing files and symbols were inspected.

| File | Responsibility/change |
| --- | --- |
| New `tldw_chatbook/Canvas/guide.py` | Lightweight shared offer policy, topic map, 12 KiB limit, bounded resource reader; stdlib only, no eager file reads. |
| New `tldw_chatbook/Canvas/guides/basics.md`, `controls.md`, `repair.md` | Focused documentation and complete examples. |
| Existing `tldw_chatbook/Canvas/static/mermaid-authoring.txt` | Reuse unchanged as the `mermaid` topic; do not duplicate its two examples. |
| `tldw_chatbook/Agents/canvas_tool_provider.py` | Fifth tool schema, validation/dispatch, guide JSON result/projections, narrow disclosed guidance. |
| `tldw_chatbook/Agents/tool_catalog.py` | Reserve/authenticate the fifth Canvas name while retaining create/update-only mutation classification. |
| `tldw_chatbook/Chat/console_agent_bridge.py` | Cheap offer-first discovery; disclose only actually available capabilities. |
| `pyproject.toml` | Explicitly package the three guide Markdown files; existing Mermaid guide is already packaged. |
| New `Docs/Examples/skills/canvas/SKILL.md` | Optional short inline user skill, model invocation disabled. |
| `Docs/User_Guide/console/canvas.md` | Offer-first behavior, guide tool, skill acquisition/import/trust and limitations. |
| New `Tests/Canvas/test_guide.py` | Reader limits, example extraction/compilation, package failure behavior. |
| `Tests/Agents/test_canvas_tool_provider.py` | Provider/catalog, authority, closed projections, no effects. |
| `Tests/Chat/test_console_agent_bridge.py`, `Tests/Agents/test_agent_service.py` | Partial disclosure, discovery, request budgeting and post-load guidance. |
| `Tests/Canvas/test_canvas_kill_switch.py` | Cached catalogs and stale guide calls after disable. |
| New `Tests/Skills/test_canvas_skill.py`; `Tests/Chat/test_console_skill_substitution.py` | Real skill import/trust and inline integration; no child dispatch. |
| `Tests/Packaging/test_canvas_gateway_distribution.py` | Wheel/sdist resource bytes and wheel-only guide reads. |
| New `Tests/Canvas/browser/test_canvas_guide_examples.py` | Execute the exact guide examples through the existing browser harness. |
| New `Docs/superpowers/qa/2026-09-10-canvas-guidance-skill.md` | Implementation evidence and limited model-behavior review. |

Do not restructure the large provider/bridge files as part of this work. The new
small module owns documentation, not artifact persistence or profile admission.

Before modifying name sets, repeat this focused inventory on the execution head:

```bash
rg -n 'CANVAS_TOOL_NAMES|CANVAS_RESERVED_TOOL_NAMES|CANVAS_MUTATION_TOOL_NAMES|CANVAS_RUNTIME_GUIDANCE|CANVAS_DISCOVERY_HINT' tldw_chatbook Tests/Agents Tests/Chat Tests/Canvas
```

Known integration hazards:

- `ToolCatalogRegistry.register_canvas_provider` compares the provider's **exact**
  name set with `CANVAS_RESERVED_TOOL_NAMES`; update both in the same commit.
- `canvas_tool_provider._validate_arguments`, `invoke`, `_serialize_result`,
  `_project_arguments`, `_project_result`, and `_source_free_payload` have closed
  cases. A new name without a new dispatch case currently falls into update.
- `_context_canvas_profiles` reads only genuine Canvas tool results; a guide
  result must not supply artifact profile state or masquerade as a Canvas read.
- `build_canvas_runtime_guidance` must describe only disclosed tools. Guide-only
  disclosure must not expose create/update APIs or full profile examples.
- `_append_canvas_discovery_hint` currently requires all four V1 tools. Separate
  this artifact-set requirement from availability of the new guide.
- `_compose_run_registry_and_allowed` uses reserved names to stop builtin/local,
  skill, or MCP collisions. `_BridgeSkillRunner` must remain unchanged: model
  skill calls still spawn restricted children.
- `AgentService` already rebuilds/budgets per-request guidance; verify the existing
  tests before deciding whether any production change there is needed.
- Literal four-tool fixtures exist in `Tests/Agents/test_agent_service.py`,
  `Tests/Chat/test_console_chat_controller.py`, and
  `Tests/Canvas/test_canvas_kill_switch.py`. Update only assertions that actually
  represent the complete catalog; preserve useful four-tool compatibility cases.

## Working commands

From `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/canvas-guidance-skill-design`:

```bash
export CANVAS_PLAN_PY=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python
"$CANVAS_PLAN_PY" --version
"$CANVAS_PLAN_PY" -c 'import tldw_chatbook; print(tldw_chatbook.__file__)'
```

Require Python >=3.12 and imports from this worktree. If moved to another host,
select its project interpreter explicitly. Run tests with `python -m pytest` from
the worktree, not a bare pytest entry point pointing at another checkout. Do not
install dependencies or browser engines merely for planning. During execution,
missing test prerequisites must be reported or installed through the repository's
existing setup route; a skipped browser test is not evidence that examples work.

## Task 1: Package focused authoring guides

**Create:** `tldw_chatbook/Canvas/guide.py`, three `Canvas/guides/*.md` files,
`Tests/Canvas/test_guide.py`.
**Modify:** `pyproject.toml`.
**Read:** `Canvas/authoring.py`, `Canvas/static/mermaid-authoring.txt`,
`Docs/Canvas/V1_RUNTIME_COMPATIBILITY.md`, and `V2_RUNTIME_COMPATIBILITY.md`.

- [x] **1. Write reader and complete-example tests.** Cover all four topics, closed
  topic selection (non-string, unknown, path-shaped values), empty/missing/invalid
  UTF-8/oversized resources, and absence of file reads on module import. Use a
  temporary resource tree and monkeypatch only `guide.files` for failure cases.
  Parse fenced `html` examples with the existing MarkdownIt test dependency.
  `basics` and `controls` each have one complete example, `mermaid` reuses its two,
  and `repair` has workflow text without a fifth document to maintain.
- [x] **2. Run RED:**

  ```bash
  "$CANVAS_PLAN_PY" -m pytest Tests/Canvas/test_guide.py -q
  ```

  Expect an import/assertion failure for the absent guide reader/resources, not a
  missing unrelated dependency. Fix test setup before treating another error as RED.
- [x] **3. Implement the small reader and topic map.** Use this complete core,
  adding public API type hints/docstrings and the shared policy constant:

  ```python
  from importlib.resources import files

  MAX_CANVAS_GUIDE_RESULT_BYTES = 12 * 1024
  CANVAS_GUIDE_PATHS = {
      "basics": "guides/basics.md",
      "controls": "guides/controls.md",
      "mermaid": "static/mermaid-authoring.txt",
      "repair": "guides/repair.md",
  }

  def read_canvas_guide(topic: str) -> str:
      if type(topic) is not str or topic not in CANVAS_GUIDE_PATHS:
          raise ValueError("invalid guide topic")
      resource = files("tldw_chatbook.Canvas").joinpath(CANVAS_GUIDE_PATHS[topic])
      with resource.open("rb") as handle:
          raw = handle.read(MAX_CANVAS_GUIDE_RESULT_BYTES + 1)
      if len(raw) > MAX_CANVAS_GUIDE_RESULT_BYTES:
          raise ValueError("guide is oversized")
      guide = raw.decode("utf-8")
      if not guide.strip():
          raise ValueError("guide is empty")
      return guide
  ```

  The provider will additionally cap the **serialized JSON result**, including
  escaping/envelope overhead. Do not add a cache, profile selector, user-directory
  lookup, or a second Mermaid source. The raw-read bound prevents oversized
  package resources from requiring an unbounded read.
- [x] **4. Author the three guides to the approved spec.** Label examples with
  their required profile. `basics` has a compact passive-SVG comparison with
  readable labels and responsive layout. `controls` has a labeled numeric input,
  visible result, and script using `getElementById`, `addEventListener`, and
  `textContent`; use stable IDs `quantity`, `unit-price`, and `total` for its
  demonstrated multiplication behavior. Use no unsupported APIs or CSS variables.
  `repair` covers current-parent reads, conflicts, staged versus saved versus
  preview-ready, source-only profiles, and stopping after one failed repair.
  Keep all topic bodies under the serialized 12 KiB ceiling rather than padding
  them toward it. Add exact `guides/basics.md`, `guides/controls.md`, and
  `guides/repair.md` package-data entries; leave pinned static assets untouched.
- [x] **5. Run GREEN and compile examples:**

  ```bash
  "$CANVAS_PLAN_PY" -m pytest Tests/Canvas/test_guide.py Tests/Canvas/test_authoring.py -q
  ```

  Use `load_profile_snapshot()` for admitted-profile tests; the test fixture named
  `candidate_snapshot` force-enables the profile and cannot prove shipped admission.
  Retain existing tests' separate candidate semantics. Expected: every complete
  example compiles with its required currently admitted profile, and invalid
  resource cases refuse without partial output. Browser execution is Task 4.
- [x] **6. Commit only Task 1 files** after scoped static checks with message
  `feat(canvas): package bounded on-demand authoring guides`.

## Task 2: Expose the scoped guide tool and safe projections

**Modify:** `Agents/canvas_tool_provider.py`, `Agents/tool_catalog.py`,
`Tests/Agents/test_canvas_tool_provider.py`, `Tests/Canvas/test_canvas_kill_switch.py`.

- [x] **1. Add provider/catalog RED tests.** Extend existing `_provider`/`_invoke`
  helpers and the genuine issuer-bound registration path. A minimal outcome test:

  ```python
  def test_canvas_guide_returns_docs_without_artifact_operations():
      provider, coordinator, _authority = _provider()
      result = _invoke(provider, "canvas_guide", {"topic": "controls"})
      assert result.ok, result.error
      payload = json.loads(result.content)
      assert set(payload) == {"status", "topic", "guide"}
      assert payload["status"] == "ok" and payload["topic"] == "controls"
      assert "getElementById" in payload["guide"]
      assert len(result.content.encode("utf-8")) <= 12 * 1024
      assert coordinator.calls == []
  ```

  Also cover extra/missing args, non-string topic, forged provider and same-name
  collisions, stale run/call context, no selected Canvas, guide-only loading,
  disable/re-enable latch, unchanged mutation classification, missing resource,
  oversized serialized result (quote/control-character expansion), and all four
  projection audiences. Seed an identifiable guide body and prove only the model
  sees it; argument projections retain only the validated topic. Invalid
  guide-shaped payloads must fail closed with the existing projection-unavailable
  category, without echoing arbitrary body/error text.
- [x] **2. Run RED:**

  ```bash
  "$CANVAS_PLAN_PY" -m pytest Tests/Agents/test_canvas_tool_provider.py Tests/Canvas/test_canvas_kill_switch.py -k 'guide' -q
  ```

  Expected: the missing fifth tool fails assertions; existing four-tool behavior
  remains the baseline.
- [x] **3. Add the schema and coordinated reserved names.** Introduce an explicit
  `CANVAS_ARTIFACT_TOOL_NAMES` set for the four existing tools; make
  `CANVAS_TOOL_NAMES` their union with `canvas_guide`. Keep stable existing order
  and append the guide. Update `CANVAS_RESERVED_TOOL_NAMES` in `tool_catalog.py`
  in the same change. The new schema is exactly:

  ```python
  {
      "type": "object",
      "properties": {"topic": {"type": "string", "enum": [
          "basics", "controls", "mermaid", "repair"
      ]}},
      "required": ["topic"],
      "additionalProperties": False,
  }
  ```

  Describe it as reading a focused authoring guide after the user requests or
  accepts Canvas. Add argument validation before resource access. Preserve
  `canvas_disabled`, `canvas_scope_unavailable`, and `invalid_arguments` handling.
- [x] **4. Add explicit guide dispatch and bounded serialization.** After the
  existing live/run checks and `_validate_arguments`, read the selected topic.
  Catch missing/undecodable/invalid resources into a fixed `guide_unavailable`
  provider error; add fixed safe copy in `_ERROR_MESSAGES`. Do not expose paths or
  exception strings. Serialize using existing `_json`:

  ```python
  content = _json({"status": "ok", "topic": checked["topic"], "guide": guide})
  if len(content.encode("utf-8")) > MAX_CANVAS_GUIDE_RESULT_BYTES:
      return _error("guide_unavailable")
  return ToolResult(ok=True, content=content)
  ```

  Do not call `_serialize_result`'s artifact branches for this operation. The
  guide schema should return before `load_schema` builds profile-authoring
  wrappers; loading it alone must not eagerly fetch the manual or Mermaid guide.
  Add only a short guide instruction to `_CANVAS_TOOL_GUIDANCE` so the new name
  is safe during ordinary guidance construction even before Task 3 refines policy.
- [x] **5. Extend closed projections.** Validate the exact successful key set
  `{status, topic, guide}`, known topic, string body, and the same serialized
  byte bound. Replace it for non-model audiences with
  `{status: "ok", topic: <known topic>, guide_bytes: <UTF-8 body size>}`.
  `_project_arguments` handles guide calls as a closed topic-only case. Keep
  existing artifact shape checks unchanged. Guide results cannot populate
  `_context_canvas_profiles`; use the artifact-name set for that history reader.
  Do not emit a Canvas card or classify guide reads as reversible mutations.
- [x] **6. Run GREEN and retained provider checks:**

  ```bash
  "$CANVAS_PLAN_PY" -m pytest Tests/Agents/test_canvas_tool_provider.py Tests/Canvas/test_canvas_kill_switch.py Tests/Canvas/test_authoring.py -q
  ```

  Adjust complete-catalog fixtures deliberately, including parameterized result
  maps, while preserving explicit four-tool cases. Expected: real registration,
  dispatch, context fences, sanitized projections, and the original create/read/
  update behavior pass. Add an actual AgentService round trip to existing provider
  persistence tests proving the guide body reaches the next model request but
  not stored Agent/tool records; a formatter unit test alone is insufficient.
- [x] **7. Commit the coordinated provider/catalog tests and implementation** with
  message `feat(canvas): expose scoped guide tool with bounded projections`.

## Task 3: Wire offer-first guidance and the optional inline skill

**Modify:** `Canvas/guide.py`, `Agents/canvas_tool_provider.py`,
`Chat/console_agent_bridge.py`, `Tests/Chat/test_console_agent_bridge.py`,
`Tests/Agents/test_agent_service.py`, `Tests/Chat/test_console_skill_substitution.py`.
**Create:** `Docs/Examples/skills/canvas/SKILL.md`, `Tests/Skills/test_canvas_skill.py`.
**Modify documentation:** `Docs/User_Guide/console/canvas.md`.

- [x] **1. Write focused RED tests for policy and disclosure.** Extend
  `test_canvas_discovery_hint_requires_the_actual_complete_run_allow_list`,
  `test_model_request_guidance_tracks_the_exact_disclosed_canvas_schema_set`,
  `test_first_request_plan_counts_canvas_guidance_before_direct_disclosure`, and
  `test_load_tools_adds_canvas_guidance_on_the_next_budgeted_request`.
  Cover all five names, four artifact names without a guide, guide-only, and
  partial create/update disclosure. Assert the offer-first rule appears where
  appropriate, absent topics remain unloaded, unavailable names are not advertised,
  and the actual final guidance remains included in token estimation. Keep tests
  for exact historical profiles and current request budget behavior.
- [x] **2. Write skill RED tests through the existing services.** Pattern simple
  content/metadata checks after `Tests/Skills/test_web_research_skill.py`, but use
  the real trust service fixture pattern from `test_skill_trust_service.py` for
  trust claims; do not use `allow_untrusted_without_trust_service=True` as trust
  evidence. Import the exact new skill, show refusal before trust, explicitly
  trust only the temporary test skill, render it, and revoke/modify it to prove
  refusal. Verify inline/user-invocable/model-disabled metadata, absent model and
  allowed-tools overrides, nonempty rendered args, no bundled duplicate guides,
  and <=4 KiB body. Extend the substitution/controller harness with a real service
  adapter for leading and embedded `$canvas`; capture the owning run's final
  tool set and make child spawning fail the test. Bare invocation must retain
  the clarify-before-authoring instruction.
- [x] **3. Run RED:**

  ```bash
  "$CANVAS_PLAN_PY" -m pytest Tests/Skills/test_canvas_skill.py Tests/Chat/test_console_skill_substitution.py -q
  "$CANVAS_PLAN_PY" -m pytest Tests/Chat/test_console_agent_bridge.py Tests/Agents/test_agent_service.py -k canvas -q
  ```

  Expected: new policy/skill assertions fail for their missing behavior, not
  environmental setup. Existing generic skill semantics must stay intact.
- [x] **4. Define and reuse the short product policy.** Use this wording as the
  shared constant, adapting only for clarity while keeping all conditions:

  > Offer Canvas only when a substantial visual or interaction materially helps.
  > For proactive use, describe the proposed artifact and benefit in one short
  > sentence and wait for acceptance before loading detailed guides, delegating,
  > or generating artifact source. A decline or no answer does not authorize
  > creation; do not repeat the same declined offer. Explicit Canvas requests and
  > requested edits already authorize that work. Consent covers that artifact and
  > bounded corrections; unrelated artifacts or unrequested redesigns need a new
  > offer. If context does not establish consent, clarify.

  The discovery hint uses the shared policy before tool loading. Require the
  artifact-name set for the existing full-artifact discovery hint and mention
  `canvas_guide` only when actually allowed; provide a documentation-only hint
  if only the guide is available. Loaded guidance describes just disclosed names.
  Guide-only guidance has no mutation/profile examples. Avoid new guide-body I/O
  during prompt construction; keep existing profile guidance authoritative.
- [x] **5. Write the optional skill.** Use native frontmatter:

  ```yaml
  name: canvas
  description: Create or revise a requested Canvas using Chatbook's supported HTML, interactive controls, and offline diagrams.
  argument_hint: requested Canvas or change
  context: inline
  user_invocable: true
  disable_model_invocation: true
  ```

  Include `{{args}}` in a clearly labeled user-request section. The short body
  repeats the consent rules, treats empty/bare activation as a clarification,
  discovers available tools, reads only relevant topics after consent, and uses
  current reads/revision IDs for edits. It states exact-profile precedence,
  available-evidence-only preview reporting, one failed repair stop, and normal
  untrusted/unavailable behavior. Do not invoke the skill as a model tool or add
  changes to `_BridgeSkillRunner`/skill-child permissions.
- [x] **6. Update user documentation.** Explain the offer and refusal flows,
  explicit requests/edits, and optional skill import from the repository's
  `Docs/Examples/skills/canvas` directory through Library > Skills. Link the actual
  example and explain review/trust is still required. State that ordinary Canvas
  use does not require installing the skill; never replace a refused invocation
  silently. Preserve existing source/save/preview/confirmed-action explanations.
- [x] **7. Run GREEN with the same commands**, plus
  `Tests/Chat/test_console_personal_context_snapshot.py` and exact affected
  Canvas nodes in `Tests/Chat/test_console_chat_controller.py` if their complete
  catalog fixtures changed. Expected: policy/disclosure/inline wiring work without
  a child run or expanded authority. These tests prove wiring, not model obedience.
- [x] **8. Commit only Task 3 files** with message
  `feat(canvas): offer before authoring and add inline canvas skill`.

## Task 4: Prove installed packaging and actual example interactions

**Modify:** `Tests/Packaging/test_canvas_gateway_distribution.py`.
**Create:** `Tests/Canvas/browser/test_canvas_guide_examples.py`.
**Read/reuse:** browser `test_canvas_zero_egress.py` and `test_canvas_mermaid.py`.

- [x] **1. Extend packaging tests.** Add the new module and three Markdown paths
  to `CANVAS_GATEWAY_PATHS`. Extend the existing wheel probe to import the reader
  from the built wheel, read all four topics, and check nonempty bounded text.
  Preserve its source/sdist byte comparison and isolated temporary working
  directory; verify the reader's module resolves inside `.whl/`, not this checkout.
  This documents a new contract even if Task 1 packaging already makes it pass.
- [x] **2. Add exact-example browser tests.** Extract examples from the actual
  topic reader with MarkdownIt rather than copying HTML into fixtures. Reuse
  `_wire_plan`, `_new_page`, `_load`, the Chromium/loopback fixtures, and egress
  assertion from `test_canvas_zero_egress.py`. Use the normal admitted profile
  snapshot for V2; do not force-enable a profile to claim shipped compatibility.
  Export/reuse its fixtures exactly as `test_canvas_mermaid.py` does. Mark the
  new tests `loopback_network`.
- [x] **3. Assert visible behavior.** Require a ready preview and expected text/SVG
  for all examples. For controls, set quantity to 3 and unit price to 7 using
  browser input events and require total to become 21; then test an invalid or
  empty input is handled visibly without uncaught runtime failure. Check the
  labels and keyboard reachability. For both Mermaid documents, require rendered
  nodes/participant labels and no failure diagnostic. Confirm generated zero
  egress using the incumbent recorder. Capture narrow 390x844 and ordinary
  1280x800 views for inspection; do not modify the shell to fit a bad example.
- [x] **4. Run the bounded installed/browser checks:**

  ```bash
  "$CANVAS_PLAN_PY" -m pytest Tests/Packaging/test_canvas_gateway_distribution.py -q
  "$CANVAS_PLAN_PY" -m pytest Tests/Canvas/browser/test_canvas_guide_examples.py -q
  ```

  Expected: wheel/sdist bytes match, wheel-only reads work, and every guide example
  executes its intended interaction or diagram. Fix guide content if necessary;
  do not loosen the renderer/profile budgets or bypass browser failures. Record
  host/interpreter/browser versions, exact commands, counts, and skip/failure
  reasons. Broaden browser tests only if a failure requires adjacent investigation.
- [x] **5. Commit packaging/browser tests and any demonstrated example fixes**
  with message `test(canvas): verify packaged guides and browser examples`.

## Task 5: Record model behavior, review the diff, and close delivery

**Create:** `Docs/superpowers/qa/2026-09-10-canvas-guidance-skill.md`.
**Update:** the implementation Backlog task created at execution start.

- [x] **1. Use synthetic conversations with an available authorized test model.**
  Exercise the actual Console request/tool path, using existing capture facilities
  if available. Do not replace this with a scripted provider and call it model
  evidence. Do not open personal conversations or print credentials. Record the
  model/provider/version where available, generation settings, prompt/tools,
  observed tool sequence, outcome, and usage fields actually returned.

  | Scenario | Expected observation |
  | --- | --- |
  | “Help me understand how quantity and unit price affect a total.” | If Canvas is proposed, one short offer; no guide body/source/create before acceptance. |
  | Reply “yes, make that Canvas” to the offer | Relevant guide then creation in the owning run. |
  | Reply “no, explain it here” | Useful chat answer; no creation or repeated same offer. |
  | User changes topic without answering the offer | No inferred acceptance or speculative source. |
  | “Create a Canvas calculator for quantity times unit price.” | Direct requested creation without a redundant offer. |
  | “Change that calculator to show a discount.” | Read current source/revision, then update; no repeated offer. |
  | `$canvas` alone | Clarification; no artifact or unnecessary guide load. |
  | Concrete `$canvas` request after normal import/trust | Inline authoring; no skill subagent. |
  | Canvas unavailable | Honest chat fallback; no unsupported tool claim or external-runtime workaround. |
  | Received compatibility failure, then one failed concrete repair | Explain remaining limitation and ask whether further repair is wanted; no rewrite loop. |

  Save only synthetic evidence needed to audit the outcome. If the model chooses
  a reasonable chat answer instead of offering, record it honestly; use a clear
  high-benefit comparison prompt to exercise the offer rather than counting a
  forced “offer now” instruction as spontaneous selection evidence.
- [x] **2. Record limitations and fixes.** Separate prompt/plumbing tests,
  compiled examples, browser evidence, and model behavior. If no authorized
  provider is available, mark model scenarios unrun and retain them as outstanding
  acceptance evidence; do not claim guaranteed consent compliance or token savings.
  Fix actual failures at the shared policy or example source, then rerun only
  affected checks. No new evaluation platform or new judge model is needed.
- [x] **3. Run final targeted regressions once after final changes.** Reuse the
  Task 1–4 commands and relevant selected nodes; do not repeat unchanged expensive
  browser/build runs without a reason. For static checks, derive the changed Python
  list from the execution baseline, run the repository's available Ruff checker and
  formatter against those files, and review any pre-existing formatting drift
  without unrelated reformatting. Run `git diff --check` and
  `python3 scripts/check_backlog_task_ids.py`. No full test sweep.
- [x] **4. Self-review and request scoped code review.** Verify every approved
  spec condition, exact reservations and projection shapes, correct primary-run
  authority, absence of guide text before demand, real import/trust semantics,
  and unchanged pinned assets. In particular:

  ```bash
  git diff 323ae22c5a -- tldw_chatbook/Canvas/static tldw_chatbook/Canvas/mermaid
  git diff --check
  ```

  Expected: no changes to existing immutable static/grammar assets. The new guide
  files live outside that closure. Confirm documentation paths and package entries
  refer to actual files, and new skill/tool names do not shadow other providers.
- [x] **5. Finish the implementation task accurately.** Add implementation notes
  with changed behavior, files, ADR links, evidence, and material limitations.
  Check each acceptance criterion only with its required evidence. Leave genuinely
  unverified criteria open. Record a lesson only if this work produced a reusable
  incident-backed finding. Commit the evidence/notes with
  `docs(canvas): record guide and inline skill verification`; mark the implementation
  task Done via CLI only when its definition of done is satisfied.

## Execution status

The user approved both design and implementation. TASK-32459 tracks delivery on
`codex/canvas-guidance-skill-design`. Independent browser/package and skill/trust
work proceeded alongside provider wiring with disjoint write scopes; shared
provider and guidance changes remained sequential. The parent completed guidance
wiring while the worker finished the separate skill/trust portion.

Stages 1–4 are implemented, locally verified, and independently reviewed. Final
quality review approved the code. Stage 5 recorded all ten model scenario classes
against the authorized Qwen3.8 llama.cpp endpoint after the user approved enabling
this app's macOS Local Network permission. A failed repair-response sample led
to a narrow clarification of the shared policy, skill, and repair guide; its
recheck stopped correctly. Streaming timeouts, reconstructed edit state, and
synthetic repair reports remain explicit evidence limitations. See the
[QA record](../qa/2026-09-10-canvas-guidance-skill.md) for exact results and limits.

# Local Agent Tools — Phase 3c (web-research skill) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a `web-research` skill that orchestrates `web_search` + `web_fetch` through a subagent run — plus the minimal skill-runner change that makes it possible.

**Architecture:** Two parts: (1) widen `_BridgeSkillRunner`'s narrowing set from builtins-only to builtins + local tools (a skill can only narrow, never grant; approval gating is unchanged because the child run shares the parent's review hook and stamp scope); (2) the skill definition itself (`SKILL.md`) shipped as an installable example, with a smoke test.

**Spec:** `Docs/superpowers/specs/2026-08-05-local-agent-tools-phases-3-4-replan.md` §2.6
**Spec deviation (disclosed):** the spec said "no new runtime code" — verification showed skill-run children are narrowed against BUILTINS ONLY (`console_agent_bridge.py:824-830`, `intersect_skill_tools` in `Agents/tool_catalog.py`), so without the widening in Task 1 the skill's child gets no web tools at all. Task 1 is the minimal honest fix (~small diff + tests). Native `spawn_subagent` children already inherit local tools (`agent_service.py:394-404`); only the skill path is affected.

## Verified facts (do not re-derive)

- `_BridgeSkillRunner.run` (`Chat/console_agent_bridge.py:820-830`): `allowed_tools = intersect_skill_tools(result.get("allowed_tools"), self._builtin_names)`; `intersect_skill_tools(None, ...)` passes ALL of the second arg through (`tool_catalog.py:130-158`).
- `run_reply` already computes `local_names` and constructs `_BridgeSkillRunner(skills_service=..., skill_names=..., builtin_names=...)` (`console_agent_bridge.py:817-821` region) — threading `local_names` through is a small, mechanical change.
- `intersect_skill_tools` semantics: narrows, NEVER grants; result ordering follows the second arg's order.
- Approval gating is inherited: `spawn()` in `agent_service.py:354-410` builds the child run with the same LoopDeps (review hook + review_state_scope), so the child's web/local calls are gated exactly like the parent's; `review_state_scope` isolates the parent's stamps (phase-1 mechanism).
- Skill format: front-matter `name`/`description`/`argument_hint`/`allowed-tools` (space-separated) + markdown body (see `Tests/fixtures/superpowers_skills/executing-plans-with-metadata/SKILL.md`). Skill install: `local_skills_service.import_skill` (`Skills_Interop/local_skills_service.py:779`); trust scanning per ADR-009.
- tldw_server `web_research_module.py` (~410 lines, at `/tmp/tldw_server_mcp/tldw_server/.../web_research_module.py`) is the orchestration reference: decompose into sub-questions → search per angle → fetch top sources → synthesize with citations and conflict flags.
- Run tests with `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest` from the worktree.

---

## Task 0: Backlog task

- [ ] Create "Local agent tools phase 3c: web-research skill". ACs:
  1. Skill-run subagents can be narrowed against local tools (never granted beyond the parent's allow-list)
  2. A skill declaring web_search/web_fetch in allowed-tools gets exactly those (plus requested builtins); undeclared skills behave as before
  3. web-research skill definition parses and passes trust scanning
  4. Install documentation exists
  5. All new tests pass
  Commit: `docs: create phase-3c backlog task`

---

## Task 1: Skill-runner local-tool narrowing

**Files:**
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py` (`_BridgeSkillRunner` + construction site)
- Test: `Tests/Chat/test_console_agent_bridge_local.py` (or the bridge test file housing skill-runner tests — find with `grep -rn "_BridgeSkillRunner\|intersect_skill_tools" Tests/`)

- [ ] **Step 1: Failing tests**

```python
def test_skill_run_narrows_against_local_tools():
    # runner with builtin_names=("calculator","get_current_datetime"),
    # local_names=("web_fetch","web_search","fs_write")
    # skill declaring allowed_tools=["web_fetch","calculator"] ->
    # spawn receives allowed_tools == ("calculator","web_fetch")
    #   (order follows the narrowing set: builtins first, then local)


def test_skill_run_undeclared_gets_builtins_and_local():
    # declared None -> passes through the FULL narrowing set (previous
    # behavior passed builtins only; document the change)


def test_skill_run_never_grants():
    # skill declaring ["web_fetch","spawn_subagent","mcp__x__y"] ->
    # only names in the narrowing set survive; spawn/runtime/foreign names dropped


def test_skill_run_child_still_approval_gated():
    # a declared local tool in the child still resolves through the
    # permission machinery (the child's calls share the parent's review
    # hook — assert the wiring, not the whole flow; follow existing
    # bridge/agent-service test patterns)
```

- [ ] **Step 2: Implement** — `_BridgeSkillRunner` gains `local_names: tuple[str, ...] = ()`; `run()` calls `intersect_skill_tools(declared, self._builtin_names + self._local_names)`; construction site in `run_reply` passes the already-computed `local_names`. Update the relevant docstrings (the "never against skill names... or local tools" comment if present — check what's actually written around :817-830 and keep the comments honest about the new narrowing set).
- [ ] **Step 3:** tests pass; `pytest Tests/Chat/ -k "bridge or skill" -q` no regressions
- [ ] **Step 4:** `git commit -m "feat: skill subagent runs narrow against local tools"`

---

## Task 2: The `web-research` skill

**Files:**
- Create: `Docs/Examples/skills/web-research/SKILL.md`
- Test: `Tests/Skills_Interop/test_web_research_skill.py` (new, in the existing skills test layout — find with `ls Tests/ | grep -i skill`)

- [ ] **Step 1: Failing test** — the skill file parses (use the real front-matter parsing path from Skills_Interop, not a hand-rolled parser); `name == "web-research"`; `allowed-tools` entries all name REGISTERED local tools (`web_search`, `web_fetch` — assert against `LocalToolProvider`'s default spec names); body mentions both tools and a citation requirement; trust scanner (`skill_trust_scanner`) finds no violations.

- [ ] **Step 2: Write the skill.** Front-matter:

```yaml
---
name: web-research
description: Research a question across the web — decompose into sub-questions, search multiple angles, fetch primary sources, synthesize with citations. Use when the user wants current, sourced answers beyond the model's knowledge.
argument_hint: research question or topic
allowed-tools: web_search web_fetch
---
```

Body (orchestration prompt for the subagent, informed by tldw_server's web_research_module flow — keep it tight, <150 lines): decompose into 2-5 sub-questions; run web_search per angle (note: use find_tools/load_tools first if the tool schemas aren't visible — the discovery hint applies to children too); select and web_fetch primary sources (prefer official/original over aggregators); synthesize with inline citations (URL per claim), a conflicts/caveats section when sources disagree, and a "not found / uncertain" statement rather than fabrication; stop conditions (budget, diminishing returns).

- [ ] **Step 3:** test passes
- [ ] **Step 4:** `git commit -m "feat: web-research skill definition + smoke test"`

---

## Task 3: Docs + close-out

**Files:**
- Create or modify: `Docs/Examples/skills/README.md` (install instructions) or extend the skill's own doc

- [ ] **Step 1:** Write install docs: import via the skills library UI or copy `Docs/Examples/skills/web-research/` into the user skills directory (`<user_data_dir>/skills/`, see `local_skills_service.py:75`); trust-scanning flow; note the skill requires local tools enabled (`[console] local_tools_enabled`) and that web tools default to ask-permission.
- [ ] **Step 2:** backlog close-out (ACs, Implementation Notes, Done) — controller-led.
- [ ] **Step 3:** final review subagent; superpowers:finishing-a-development-branch.
- [ ] Commit: `docs: web-research skill install docs`

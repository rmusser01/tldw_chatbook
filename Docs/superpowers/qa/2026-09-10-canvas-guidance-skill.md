# Canvas guide and inline skill verification

Status: implemented and reviewed; targeted checks and bounded live-model sample complete
Task: TASK-32313
Baseline: `027422cfaa` (`codex/canvas-guidance-skill-design`)
Spec: [Approved design](../specs/2026-09-10-canvas-guidance-skill-design.md)
Plan: [Implementation steps](../plans/2026-09-10-canvas-guidance-skill-implementation.md)
ADR: [149](../../../backlog/decisions/149-offer-first-canvas-guide-and-inline-skill.md)

## Environment and baseline

- macOS; Python 3.12.11; Textual 8.2.8. The shared project virtualenv imports
  `tldw_chatbook` from the isolated Canvas worktree when run there.
- Baseline command: project Python `-m pytest Tests/Agents/test_canvas_tool_provider.py -q`.
  Result: **99 passed in 2.64s**, before implementation.
- Three baseline warnings concerned requests dependency versions, deprecated
  `pkg_resources`, and an existing invalid escape sequence in `patch_tool_impls.py`.
  Shared pytest temporary-directory cleanup also encountered unrelated old
  permission-protected directories. Subsequent scoped runs use unique task-owned
  `--basetemp` directories to avoid cleanup of other tasks' temporary state.

## Live-model availability

The configured chat default is `llama_cpp` / `local-model`, with endpoint
`http://localhost:8080`. A sandbox-independent read-only check found connection
refused on local ports 8080, 8000, 1234, and 11434. The user specified an alternative
llama.cpp endpoint at `http://192.168.5.196:9191`. A read-only request to its
`/v1/models` endpoint failed to connect both inside and outside the sandbox.
A direct socket check outside the sandbox returned `OSError 65: No route to host`.
The initial conclusion that the server was unreachable from the Mac was wrong:
the user reported it working, and the existing Firefox page showed llama-ui at
that address. No Firefox navigation, typing, or changes were performed; the user
subsequently prohibited further use of that browser.

Process-specific diagnostics found an active route through en0. macOS nehelper
logs at 22:39:38, 22:42:15, and 22:44:36 on September 10 explicitly reported:
`Local network denied by preference for ChatGPT (com.openai.codex)`.
The user then explicitly authorized enabling this app's Local Network permission.
System Settings showed ChatGPT off before the change and on afterward.
`/v1/models` and `/health` then succeeded from the command tools. The server
reported Qwen3.8-27B-UD-Q8_K_XL.gguf, llama.cpp build `b10430-4c1a0af40`,
112,384 context tokens, and one processing slot. No further Firefox use occurred.

The initial temporary harness manually constructed a provider resolution and
therefore skipped normal endpoint capability discovery. Its fallback-protocol
responses included malformed tool fences and do not qualify native Canvas
creation. The corrected harness calls the production `resolve_for_send`;
the server's reported template capabilities resolve to `native_tools=True`.
No production protocol or provider behavior was changed to repair the harness.

### Recorded model behavior

The [compact evidence](canvas-guidance-live-2026-09-10/summary.json) retains all ten
scenario classes, the failed repair baseline and corrected recheck, model/tool
sequences, final responses, request-input hashes, returned usage, and distinct
synthetic HTML sources. Full temporary gateway-input captures and the small
driver remain in `/private/tmp/canvas-live-model-32313`. These are headless
Console bridge/provider samples, not an interactive Console UI walkthrough.

The production `ConsoleAgentBridge`, `AgentService`, `ConsoleProviderGateway`,
scoped `CanvasToolProvider`, compiler, and temporary Canvas controller ran
unchanged. Provider capability discovery used `resolve_for_send`. Skill cases
used normal local import, explicit trust of the imported snapshot in a temporary
store, and the actual controller's inline substitution. Recorded first requests
contain the rendered skill body. All retained runs spawned zero children.

| Scenario | Observed behavior |
| --- | --- |
| Proactive explanation | Answered in chat and offered a small interactive calculator and its benefit; no Canvas guide/source/mutation before consent. An unrelated calculator call was denied by the harness. |
| Acceptance | Loaded basics and controls, then successfully staged a canvas-v1 calculator with no compatibility issues. Its HTML matches the browser-tested controls example apart from the terminal newline. |
| Refusal | Continued in chat; no new tool call or repeated Canvas offer. |
| Topic change | Answered the new question; no new tool call or assumed acceptance. |
| Explicit creation | Loaded basics and controls and created directly without another offer; successful staged result and committed temporary settlement. |
| Requested edit | Read a seeded, reachable calculator, then updated it with that exact revision as the expected parent; discount fields added, no compatibility issues, temporary settlement committed. |
| Bare `$canvas` | Asked what to create; no tool call. |
| Concrete trusted `$canvas` | Expanded inline, loaded controls, and created in the owning run; no child, no compatibility issues, temporary settlement committed. |
| Canvas unavailable | Reported missing Canvas tools and made no artifact. Offered chat math or supplying code as choices; did not generate or execute an external artifact. An unrelated calculator call was denied. |
| Reported failed repair | Initially falsely claimed another repair without a tool call. After clarifying the shared policy, skill, and repair guide, the same case stopped, acknowledged the unresolved failure, and offered further work only if wanted, with no tool call or claim of a new fix. |

The final wording explicitly says that a report of continued failure is not
permission for another repair, and prohibits claiming a mutation without a
matching successful tool result. The original failed response remains in the
evidence. The successful repair recheck is `10-repair-stop-recheck`; the other
cases precede this narrowly scoped wording clarification.

### Settings, harness corrections, and limits

- Server: llama.cpp `b10430-4c1a0af40`, Qwen3.8-27B-UD-Q8_K_XL.gguf.
  Temperature 0, seed 42, reasoning effort `none`; the session instruction was
  “Be concise. Keep artifacts small and complete.” The native streaming samples
  used a 4,096-token output cap. Final non-streaming creation/edit/skill/repair
  samples used 2,048, a 240-second content watchdog, a 300-second HTTP timeout,
  zero configured HTTP retries, and a 420-second whole-turn limit. These changes
  applied only to the temporary test profile.
- Default native streaming attempts at edit/direct creation hit the existing
  90-second content watchdog. Streaming text also contained mojibake. Those
  attempts do not establish successful default streaming behavior. A first
  non-streaming attempt hit the local adapter's separate 120-second HTTP timeout;
  it was stopped before continuing with explicit timeout/retry settings.
- The first edit harness had staged source but had not confirmed temporary
  settlement. The corrected edit recompiled the successful acceptance sample's
  exact source into a fresh temporary session, confirmed settlement, then asked
  the real model to edit it. This proves editing reachable seeded source, not
  uninterrupted UI create-to-edit continuity or durable database persistence.
- Canvas tool approvals were automatic within the synthetic harness; other
  tools were denied. Enablement was explicitly supplied by the harness. The
  sample measures model behavior, not the settings/approval UI or a host-enforced
  consent gate. The final skill adapter delegates file reads as well as genuine
  context and execution operations. Independent read-only review found no model
  or network mocks and confirmed temporary source reachability.
- Repair context was synthetic conversation reporting an initial preview failure
  and one failed correction. It tests behavior after a reported failure; it does
  not reproduce a renderer failure or exercise browser diagnostics.
- Gateway recordings are inputs to the real gateway, not raw HTTP captures.
  Non-streaming samples returned per-call prompt/completion/total usage, retained
  verbatim. Streaming sample usage payloads were absent; their run counters are
  not substituted for provider-reported usage. No comparative token savings,
  general model-obedience guarantee, or successful live browser preview is claimed.
- Product code and immutable runtime assets were not changed to accommodate the
  harness. The only product change from live sampling was the repair/reporting
  wording. After it, **210 targeted guide/provider/context/skill tests passed in
  2.90s**, and **40 packaging/skill-substitution tests passed in 26.71s**, including
  a fresh wheel/sdist check. Ruff lint and format checks passed for the changed
  Python module. Exact browser example sources did not change.

## Stage results and reviews

### Packaged guides

Commit `e249600ed7fb194136b0399246e63f29d58ee27c` adds the bounded reader,
three focused guides, package entries, and reader/example tests. Mermaid text
remains the existing immutable resource.

- RED: `-m pytest Tests/Canvas/test_guide.py -q` failed because the reader did
  not yet exist.
- GREEN: `-m pytest Tests/Canvas/test_guide.py Tests/Canvas/test_authoring.py -q`
  passed **50 tests in 1.37s** using a unique task-owned basetemp.
- Scoped Ruff lint/format and whitespace checks passed.
- Every complete example compiles under the shipped profile admission snapshot;
  this is compiler evidence, with browser execution and installed packaging pending.
- Independent stage 1 spec and code-quality reviews both passed. The spec
  reviewer independently reran the 50 checks.

### Browser and packaging verification

The independent Task 4 checks were brought forward alongside provider wiring
because their write set is separate. The skill import/trust portion of Task 3
also runs separately; shared provider/guidance edits remain sequential.

- Initial wheel/sdist checks: **3 passed in 13.89s**, including byte-for-byte
  packaged resources and reading every topic from an isolated wheel import.
- Chromium initially could not start inside the macOS sandbox (Mach port
  permission denial). Running the same four tests outside that sandbox launched
  Chromium **145.0.7632.6**.
- The first actual browser run found `invalid-plan` in both new HTML examples.
  Chromium expanded CSS `background` and `border` shorthands into disallowed
  properties. Explicit longhands fixed the examples; pinned runtime assets and
  runtime admission were preserved.
- Corrected examples: **4 passed in 4.31s**. Tests extract the packaged Markdown
  fences verbatim, compile with shipped admission, run the calculator (3 × 7 = 21),
  clear/recover input, verify keyboard traversal, render both Mermaid documents,
  check final ready status, and assert zero generated egress.
- All eight captured views were inspected at 390×844 and 1280×800. The two new
  HTML examples fit narrow widths. Existing Mermaid examples retain the runtime's
  intrinsic-size diagram presentation (horizontal scrolling at narrow widths).
- Screenshots: `/private/tmp/canvas-guidance-browser-3/test_packaged_guide_example_ex*/`.
- After the CSS correction, `Tests/Canvas/test_guide.py` plus
  `Tests/Packaging/test_canvas_gateway_distribution.py` passed **49 tests in
  13.21s** with a fresh wheel/sdist build. Scoped Ruff and whitespace checks pass.
- Task 4 commits: `d404d3721d`, `99a5782314`. Spec review found label assertions
  could match retained source, so they now target rendered SVG text. The four
  stricter browser tests passed in **4.37s**; spec re-review and code-quality
  review both passed.

### Scoped guide provider

Commit `da6cf9426d` adds the fifth reserved Canvas tool, closed topic validation,
bounded model results and metadata-only projections. Real AgentService tests
verify the guide reaches the next model request but not stored records, emitted
steps, or run logs.

- RED guide selection: **36 failed, 8 passed** before implementation.
- Final provider/kill-switch guide selection: **48 passed, 105 deselected**.
- Provider, kill-switch, and authoring regression files: **157 passed**.
- Ruff lint and whitespace checks pass. Three affected files pass formatting;
  existing formatting drift in the large catalog file was left unchanged.
- Independent spec and code-quality reviews both passed.

### Offer-first guidance and inline skill

Commits `c207501bdd` and `19f0649059` add the optional 3,293-byte skill body and
shared policy in discovery and loaded runtime guidance. Complete artifact access
without the guide remains supported; guide-only access advertises documentation.
No skill-child runner or permission change was required.

- Policy RED: **6 failed, 2 passed** for missing policy wiring.
- Policy/request guidance checks: **10 passed in 1.20s**, including all five
  tools, four artifact tools without guide, guide-only, partial mutations, no
  discovery guide-body reads, final-request policy, and budget inclusion.
- `Tests/Skills/test_canvas_skill.py` and
  `Tests/Chat/test_console_skill_substitution.py`: **40 passed in 16.87s**.
  Tests import the exact skill into a temporary real trust service, refuse before
  trust and after content modification, and verify inline owning-run substitution.
- Final core regression files (reader, provider, kill switch, authoring, skill,
  and personal-context snapshot): **222 passed in 3.62s**.
- An existing substitution test failed identically when restored from baseline
  `3afa68f1b9`: it expected acceptance before placeholder creation. Existing
  commit `a26cdafd80e` deliberately moved the hook after commit. The corrected
  test snapshots one pending placeholder and zero provider calls at acceptance;
  production send code is unchanged.
- The existing first-request budget test used a provider capability stub that
  rejected `reasoning_replay`, letting a caught exception masquerade as budget
  fallback. Its stub now accepts that argument, and a post-call assertion proves
  the policy actually reached the token counter.
- Scoped Ruff passes. Four large files already fail full-file formatting at the
  execution baseline. Comparing formatter edits against changed lines left no
  new formatting drift; unrelated baseline formatting was preserved.
- Spec review requested the general clarification condition in shared guidance,
  beyond bare skill activation. Commit `afa76f416b` adds that sentence and tests
  in discovery and all five schema-set cases. RED: **6 failed**; GREEN:
  **10 passed in 1.05s**. Independent spec re-review passed and reran the same
  ten checks successfully. Stage 3 quality review passed.
- The final reader/policy module update was included in a new wheel/sdist build:
  reader and packaging checks again passed **49 tests in 13.23s**.

## Final review and retained work

Independent overall review approved `027422cfaa..afa76f416b` with no actionable
code findings. All stage spec and quality reviews passed. The final review
confirmed the fixed topics, bounded reader/result, closed projections, scoped
authority, trust/inline behavior, and unchanged pinned runtime/grammar assets and
production controller. It explicitly retained live qualification as outstanding.

Ruff passed all 13 changed Python files; nine passed full-file formatting and the
four remaining files had verified baseline formatting drift outside changed lines.
Whitespace and documentation-link checks passed. Backlog ID validation passed
across 3,694 task files. No full test sweep, merge, or push was performed.

The code and evidence stay on `codex/canvas-guidance-skill-design` in the isolated
worktree. The ten scenario classes now have recorded model observations, including
the retained failed-repair baseline and successful wording recheck. TASK-32313's
evidence criterion is satisfied with the explicit limits above.

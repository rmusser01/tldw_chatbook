# Canvas guide and inline skill verification

Status: code implemented and reviewed; local verification passed; live-model qualification pending
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
A direct socket check outside the sandbox returned `OSError 65: No route to host`;
this establishes a host/network availability problem, not an HTTP/model response.
The user was asked to restore reachability or provide another reachable address.
Retry this authorized endpoint when the live checks are ready. No cloud model
was selected and no model generation calls have been made.

Model-behavior acceptance is pending; automated prompt and provider fixtures do
not substitute for this evidence. Code, package, and browser verification continue
independently.

The final `/v1/models` probe also failed to connect. All ten model scenarios in
Task 5 of the linked plan remain **unrun**. No generation request was dispatched;
model identity, model decisions, and response usage are unavailable. There is no
claim of guaranteed consent obedience or measured token savings. TASK-32313 must
remain In Progress with its model-evidence acceptance criterion unchecked until
the authorized server is reachable and the actual Console sample is recorded.

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
worktree. TASK-32313 has AC 1–4 checked; AC 5 remains unchecked for the ten unrun
live-model scenarios. Restore reachability of the authorized server, then run
the actual Console sample described in the plan before declaring the task Done.

# Native goal Console qualification — TASK-32120

Implementation base: `fcfd77f47024e641959594670ecb2e79901e9b2a`.
Accepted contract: [ADR-141](../../../../../backlog/decisions/141-native-console-goal-runs.md), preserving [ADR-134](../../../../../backlog/decisions/134-fleet-admission-and-automatic-work-budgets.md).

## Actual CLI result with a deterministic provider

[deterministic-cli.json](deterministic-cli.json) records the actual target argv,
exits **7 then 0**, stdout/stderr, five model-visible message payloads and runtime
lookup IDs across **two increments**. The test uses real SQLite, the real native
controller/bridge, approved local `fs_edit`, and real POSIX subprocess execution.
The recording provider chooses references from actual tool messages; it does not
read private coordinator observations to invent evidence. This fixture is not a
live language model.

The unchanged [check.py](check.py) receives the project path explicitly, checks
`fixture.txt`, and retains the same SHA256 before/after. The approved edit produces
[fixture.diff](fixture.diff); the inspected final [fixture.txt](fixture.txt) is
exactly `valid` plus a newline. Two generations and five model calls remain
charged. An external sentinel is unchanged. Local file tools are binding-confined;
this does **not** establish OS confinement of the trusted script. Qualification is
for the POSIX local-skill path, not every configurable executor.

## Actual configured local endpoint result

Endpoint: `http://127.0.0.1:9099/v1`, advertising `Qwen2.5-0.5B-Instruct`.
The application uses Chatbook's `llama_cpp` adapter and incumbent fenced-tool
protocol. This names the adapter, not the endpoint's serving binary. Parent
verified a local uvicorn process and a corroborating existing local-LLM UAT record;
the temporary server application source was no longer available for inspection.

Two initial harness attempts stopped **before network dispatch** at
`native_goal_required` (zero HTTP/model calls), revealing an incorrect native
function-calling gate. After the scoped fix, the actual endpoint was exercised:

| Actual run | Evidence | Result |
|---|---|---|
| Before handoff clarification | [HTTP trace](local-endpoint-before-prompt-fix.json) | 2 HTTP 200 replies, 2 increments/calls, no tools or evidence, malformed reports, paused/no_progress |
| Final, clarified handoff | [HTTP trace](local-endpoint-final.json) | 2 HTTP 200 replies, 2 increments/calls, no tools or evidence, malformed reports, paused/no_progress |

Both runs had finite caps: 2 iterations, 8 model calls, 50000 tokens, 1024 output
tokens/call, 60 elapsed seconds; each iteration 4 turns, 32 steps, 30 seconds.
The final request clearly permits intermediate fenced tool calls and requires
strict JSON only for the FINAL report after authorized work. Its second response
recommended completion without proof; completion was refused. Both runs left
`invalid\n` unchanged, an empty diff and an unchanged external sentinel.
`quality_success=false` is intentional and honest. This is negative evidence of
safe refusal, not certification of successful goal execution by this model.
No cloud fallback, model download, or further generation was used.

These JSON files are actual captured endpoint/controller results over synthetic
fixtures. Only temporary fixture-root paths were replaced by `<fixture-root>`;
raw artifact hashes identify the unnormalized temporary sources. No request
headers, secrets, user conversations, private application logs or user files
are included.

## Modal rendering and keyboard evidence

Final captures use the shipped consolidated stylesheet and real ChatScreen
harness, 80×24 and ordinary 160×44. The Console rail was explicitly opened and
its state/positive geometry asserted before mounting the modal. The dimmed
fresh-profile backdrop remains its setup view: these captures establish modal
layout/keyboard behavior, not operational rail rendering.

| Surface | 80×24 | 160×44 |
|---|---|---|
| Setup | [PNG](goal-setup-80x24.svg.png), [SVG](goal-setup-80x24.svg) | [PNG](goal-setup-160x44.svg.png), [SVG](goal-setup-160x44.svg) |
| Review | [PNG](goal-review-80x24.svg.png), [SVG](goal-review-80x24.svg) | [PNG](goal-review-160x44.svg.png), [SVG](goal-review-160x44.svg) |

SVG blank-line whitespace was removed for the repository whitespace gate; drawing
content is unchanged. First-pass captures were inspected, then one bounded confirmation verified
objective/criteria first, hidden empty sources, human-readable wait reasons,
fixed action rows, focus/Enter and Escape. The 80-column Review capture is
scrolled to fresh check output. Only implemented controls are added; the dimmed
footer belongs to the underlying Console. Quick Look rendered the exported SVGs
because installed cairosvg lacks libcairo; its white canvas padding is not TUI
layout. [Geometry output](modal-geometry.txt) records the final capture run.
Later identity-only fixes did not change the pictured layout; the same geometry
tests passed in [selected-review-controls.txt](selected-review-controls.txt).

## Targeted validation and baseline boundaries

Runs overlap; **do not add their counts**:

- Earlier broad targeted gate: 186 passed, 1 opt-in live skip, 2 failures. One
  fixture lacked the real change tracker; the other exposed a lost Pause wake
  during an awaited snapshot. The race was fixed by clearing the wake before the
  read, then checked with a controlled read/Pause interleaving.
- [Amended affected scope](targeted-amended.txt): **98 passed, 1 intentional live
  skip**. Includes setup/status/settings/navigation, dispatch/scheduling/runtime,
  actual CLI and unchanged screen ratchet. Untouched Change Review and Settings
  regression groups had passed in the earlier broad run.
- [Final selected-review/removal controls and navigation](selected-review-controls.txt):
  **10 passed** after clearing captured acceptance on selection/revision change.
- [Final setup controls](setup-controls.txt): **6 passed**, including double Start,
  actual runtime launch/Starting retry, off-loop trust reference and launch drain.
- Canonical batched Save/global Revert and mounted goal settings: 3 passed,
  355 deselected (separate earlier targeted subset).
- Privacy/migration/ownership/diagnostics: **97 passed, 1 historical-commit skip,
  3 proved baseline diagnostic failures**. Do not report this as an overall pass.

[Final inventory delta](goal-diagnostic-final-delta.json) proves the only remaining
manifest mismatch is the pre-goal TTS checksum, with no new goal-owner or sink
change. [Baseline label proof](goal-diagnostic-static-baseline-proof.json) shows
the other two failing assertions expect seven plus one removed bridge/fleet
labels identically absent at preserved baseline `77bc58dc17`. The diagnostic test
source is unchanged. Existing RequestsDependencyWarning and invalid-escape
SyntaxWarnings come from unchanged dependencies/source.

[Scoped Ruff](scoped-ruff.txt) and [format check](scoped-format.txt) pass for all
new/small affected Python owners. [Legacy lint comparison](legacy-lint-proof.json)
proves zero new diagnostic code/message tuples in the six large incumbent owners;
[changed-range formatting](legacy-format.txt) is AST-preserving and at a fixed
point. Their pre-existing lint/format debt was not blanket-rewritten. CSS build
completed with only a generated timestamp difference, which was reverted. The
ChatScreen ratchet remains unchanged and passes after the relevant review-provider
extraction. No full suite was run.

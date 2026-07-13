# Console ▸ prompts injection + /system — Phase 2 gate evidence (2026-07-12)

Branch: `claude/prompts-library-spec` (worktree off `origin/dev`, merge-base `993653d8`).
Spec: `Docs/superpowers/specs/2026-07-12-library-prompts-console-injection-design.md`.
Plan Phase 2 (Tasks 9–15).

Captured live from textual-serve (real app CSS, worktree code), playwright bundled chromium,
viewport 2050×1240, isolated profile with 7 real two-part prompts seeded via
`PromptsDatabase.add_prompt` into `…/default_user/tldw_chatbook_prompts.db`. Provider: real
llama.cpp @ `127.0.0.1:9099`, model `Qwen3.6-27B-Uncensored-…-Q8_K_P.gguf`.

## Captures

- `A-02-system-applied-rail-preview-2026-07-12.png` — after `/system Three word oracle`: the Model
  rail line flips from **`System: none`** to **`System: You must answer every question in exac…`**
  (truncated, in-place update — no recompose), the composer is **cleared** (the successful
  named-apply clears the command), and the context estimate jumps **0 → 23 tokens** (the system
  prompt is now counted — the Task-13 review's context-estimate minor, fixed in Task 14).
- `B-01-prompt-picker-ambiguous-2026-07-12.png` — `/prompt Summar` (ambiguous: two matches) opens
  the **Insert prompt** picker, search field pre-filled with `Summar`, both `Summarize …` prompts
  listed, first row focused (keyboard-first).
- `B-02-picker-selection-inserted-2026-07-12.png` — selecting a picker row inserts that prompt's
  **user part** into the composer with paste semantics (`Pasted Text: 54 Characters`).
- `C-01-prompt-short-inline-2026-07-12.png` — `/prompt Three word oracle` (exact match, short user
  part) inserts **inline**: composer shows `What is the capital of France?` as plain text (below the
  paste-collapse threshold).
- `C-02-prompt-collapsed-paste-2026-07-12.png` — `/prompt Code review checklist` (exact match,
  469-char user part) inserts as a **collapsed paste token**: `Pasted Text: 469 Characters`.
- `D-01-unknown-command-hint-2026-07-12.png` — `/frobnicate the widget` → transcript hint
  **`Unknown command /frobnicate — available: /prompt, /system. Press Enter again to send as text.`**
  with the draft preserved (Enter-again armed).
- `D-02-enter-again-sent-2026-07-12.png` — pressing Enter again sends the literal text as a normal
  **`User  /frobnicate the widget`** message (composer cleared, conversation titled from it) — the
  `/usr/bin/…`-style escape hatch works.
- `E-01-system-editor-modal-2026-07-12.png` — bare `/system` opens the **Edit system prompt** modal:
  scope line **`Applies to this session.`**, full TextArea, `Name` field, `Save to Library`, and
  `Clear · Cancel · Apply` actions.
- `F-01-system-empty-part-error-2026-07-12.png` — `/system User only snippet` (a prompt with an
  empty system part) → inline transcript error **`Prompt "User only snippet" has no system part.`**
  (byte-exact), rail still `System: none` (session unchanged), draft preserved.
- `smoke-01-initial-2026-07-12.png` — baseline Console with the new `System: none` rail line.

## Real send — the applied system prompt reaches the provider

`provider-request-with-system-message-2026-07-12.json` is the **actual provider request body**
captured off the wire (a logging proxy in front of llama.cpp) for a send made with
`Three word oracle`'s system part applied. Its `messages` array is:

```
[ {"role":"system","content":"You must answer every question in exactly three words. Never more, never fewer."},
  {"role":"user",  "content":"What is the capital of Japan?"} ]
```

The system message is present as a leading `system` role — proving `ConsoleSessionSettings.system_prompt`
is prepended into the real provider payload (Task 13 `_leading_system_message`). Behavioral
corroboration: with the system prompt applied, the model's own reasoning counts words
("Option 2: 'The capital is Tokyo.' (4 words) — Too long"); an identical send with **no** system
prompt produces a request with roles `['user']` only and no word-count reasoning.

Evidence method used: **request-capture** (strongest form) — the injected proxy captured the exact
JSON via the low-level `chat_api_call("llama_cpp", …, system_message=…)` path (the same handler the
Console send worker routes through). Note: the browser-driven UI send worker did not itself route
through the injected proxy within a served session (the Console resolves its endpoint independently
of the runtime proxy swap), so the on-the-wire capture was taken through the identical provider
call path headlessly rather than by screenshotting a completed streamed answer.

## Verification

- Broad sweep (`Tests/Chat Tests/Library` + the Console UI suites + composer/picker/system-prompt):
  **1844 passed / 1 load-flake** (`test_library_shell_search_history_row_reruns_query`, passes in
  isolation), plus 229 passed re-running the four Console suites the final-review fixes touched, and
  86 + 127 on the fix's covering suites.
- Whole-branch review (opus): **APPROVE WITH FIXES** — 1 Important (unescaped prompt description in
  the list-row label → crash on `[/]`; fixed `7943d8ba`), 2 deferred UX items folded in
  (`/system <name>` now clears the composer on success `7a310857`; stale-default refresh preserves
  an applied `/system` prompt `d3c04731`). Minor-rollup triaged: 0 block / 6 defer-to-backlog / 10
  drop. Full report: `.superpowers/sdd/final-review-report.md`.

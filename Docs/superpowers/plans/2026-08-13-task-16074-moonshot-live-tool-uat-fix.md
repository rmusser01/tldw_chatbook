# Moonshot Live Native-Tool UAT Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accept Moonshot Kimi K3's bounded `system_fingerprint`, terminal choice usage, and identical trailing usage SSE shapes so the real Console native-tool continuation UAT completes instead of surfacing a synthetic 502.

**Architecture:** Keep request construction, provider policy, AgentService, and Console unchanged. Correct the provider-neutral hosted streaming response allowlists, validate the optional fingerprint under the existing metadata cap, accept mapping-valued usage only on a terminal choice with no same-event top-level usage, and accept exactly one later top-level duplicate only when it is identical under type-strict canonical JSON equality. Prove the fix at both the pure parser and real Console-to-scripted-HTTP boundaries before the final paid UAT.

**Tech Stack:** Python 3.12, pytest, requests/SSE, Textual Console agent bridge, Loguru redaction tests, Backlog.md, GitHub CLI

---

## Scope And File Map

- Modify `tldw_chatbook/LLM_Calls/hosted_chat.py`: admit and bound the standard
  optional streaming `system_fingerprint` field and Moonshot's terminal
  choice-level usage mapping and identical trailing duplicate.
- Modify `Tests/LLM_Calls/test_hosted_chat.py`: pin accepted, null, malformed,
  oversized, ambiguous, misplaced, and unknown strict-parser behavior.
- Modify `Tests/Chat/test_kimi_zai_native_tools.py`: mirror Moonshot's live
  fingerprint and terminal choice usage in the existing joined Console
  HTTP/tool-continuation fixture while preserving Z.ai's wire shape.
- Test `Tests/Chat/test_live_moonshot_zai_api.py` without modifying or weakening
  its exact tool/result/final marker contract.
- Modify `backlog/tasks/task-16074 - Make-Moonshot-live-native-tool-continuation-pass.md`:
  record evidence, completed ACs, and Implementation Notes.
- Modify the TASK-16074 spec and this plan only for review corrections or final
  evidence links.

No new production module, dependency, configuration field, provider branch,
or persistent payload-capture hook is planned.

ADR required: no

ADR path: `backlog/decisions/063-hosted-provider-wire-and-durable-tool-continuation.md`

Reason: this is a response-compatibility correction inside ADR-063's accepted
neutral hosted wire boundary.

### Task 0: Commit The Reviewed Design And Plan

**Files:**
- Modify: `backlog/tasks/task-16074 - Make-Moonshot-live-native-tool-continuation-pass.md`
- Modify: `Docs/superpowers/specs/2026-08-13-task-16074-moonshot-live-tool-uat-fix-design.md`
- Create: `Docs/superpowers/plans/2026-08-13-task-16074-moonshot-live-tool-uat-fix.md`

- [ ] **Step 1: Stage only the reviewed documentation**

```bash
git add -- \
  'backlog/tasks/task-16074 - Make-Moonshot-live-native-tool-continuation-pass.md' \
  Docs/superpowers/specs/2026-08-13-task-16074-moonshot-live-tool-uat-fix-design.md \
  Docs/superpowers/plans/2026-08-13-task-16074-moonshot-live-tool-uat-fix.md
git diff --cached --check
```

Expected: only the task/spec/plan root-cause and review updates are staged.

- [ ] **Step 2: Commit the planning checkpoint**

```bash
git commit -m "docs(chat): plan Moonshot stream fingerprint fix"
git status --short
```

Expected: commit succeeds and the worktree is clean before TDD/rebase work.

### Task 1: Pin The Complete Fingerprint Contract RED

**Files:**
- Modify: `Tests/LLM_Calls/test_hosted_chat.py:520-660`
- Modify: `Tests/Chat/test_kimi_zai_native_tools.py:62-108`
- Test: `Tests/LLM_Calls/test_hosted_chat.py`
- Test: `Tests/Chat/test_kimi_zai_native_tools.py`

- [ ] **Step 1: Add accepted bounded and null stream regressions**

Add a complete synthetic stream whose first event includes the exact live
top-level shape and a bounded fingerprint:

```python
@pytest.mark.parametrize("fingerprint", ["fp_kimi_live", None])
def test_hosted_chat_stream_accepts_system_fingerprint(
    fingerprint: str | None,
) -> None:
    event = {
        "id": "chatcmpl_test",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "kimi-k3",
        "system_fingerprint": fingerprint,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": "done"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"total_tokens": 3},
    }
    stream = HostedChatStream(
        iter(
            [
                SSERecord(event=None, data=json.dumps(event)),
                SSERecord(event=None, data="[DONE]"),
            ]
        ),
        finish_policy=_POLICY,
    )

    assert list(stream) == [event]
    assert stream.terminal_turn.text == "done"
```

- [ ] **Step 2: Add malformed, oversized, and unknown-key RED cases**

Before production changes, add a parameterized event test for:

```python
[True, 1, "", "x" * (hosted_chat._MAX_METADATA_CHARS + 1)]
```

Each `system_fingerprint` must raise `HostedChatProtocolError`. Add a separate
event with `"unexpected_live_metadata": "value"` that must also raise. The
malformed/unknown tests are expected to pass on the old code because the old
allowlist rejects the field wholesale; their purpose is to constrain the
minimal GREEN implementation before it is written.

- [ ] **Step 3: Make the existing joined fixture reproduce the live field**

In `_tool_turn(provider)`, add
`"system_fingerprint": "fp_kimi_live"` to the first event only when
`provider == "moonshot"`. Preserve the existing tool deltas, reasoning,
terminal usage, continuation assertions, and Z.ai bytes.

- [ ] **Step 4: Run the accepted and joined tests and verify RED**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/LLM_Calls/test_hosted_chat.py::test_hosted_chat_stream_accepts_system_fingerprint \
  'Tests/Chat/test_kimi_zai_native_tools.py::test_console_runs_two_native_calls_with_private_continuation[moonshot]'
```

Expected: both accepted parser cases fail with `HostedChatProtocolError: Hosted
Chat stream event is malformed`. The joined test reaches an error outcome with
no calculator tool/checkpoint and normally stops at its earlier
`assert checkpoint is not None` assertion; it need not expose the parser
exception directly because Console intentionally maps it to safe provider-error
state.

- [ ] **Step 5: Run the fail-closed cases on the old code**

Run the new malformed/oversized/unknown tests by exact node or `-k
'system_fingerprint or unexpected_live_metadata'`.

Expected: all pass before production changes, proving the new GREEN cannot
simply broaden the event to arbitrary metadata.

- [ ] **Step 6: Confirm the accepted/joined RED is discriminating**

Temporarily remove only `system_fingerprint` from each test event and rerun.
Expected: both pass. Restore the field and confirm both are RED again. Do not
commit this temporary mutation.

### Task 2: Apply The Minimal Bounded Parser Fix

**Files:**
- Modify: `tldw_chatbook/LLM_Calls/hosted_chat.py:186-200`
- Modify: `Tests/LLM_Calls/test_hosted_chat.py:520-760`
- Test: `Tests/LLM_Calls/test_hosted_chat.py`
- Test: `Tests/Chat/test_kimi_zai_native_tools.py`

- [ ] **Step 1: Admit and validate the optional field**

Change only the streaming top-level allowlist and optional metadata check:

```python
if set(event) - {
    "id",
    "object",
    "created",
    "model",
    "system_fingerprint",
    "choices",
    "usage",
}:
    raise HostedChatProtocolError("Hosted Chat stream event is malformed.")
fingerprint = event.get("system_fingerprint")
if fingerprint is not None:
    _required_metadata(fingerprint, "system fingerprint")
```

Do not allow arbitrary extra keys, coerce non-strings, change provider builders,
or log/retain any new private state.

- [ ] **Step 2: Run every new contract case and verify GREEN**

Run the Task 1 accepted/joined command plus the malformed, oversized, and
unknown-key nodes.

Expected: every new case passes. The joined test completes its tool call and
continuation rather than merely suppressing the parser error.

- [ ] **Step 3: Run the complete neutral hosted parser file**

Run:

```bash
../../.venv/bin/python -m pytest -q Tests/LLM_Calls/test_hosted_chat.py
```

Expected: all tests pass; malformed, oversized, post-terminal, and resource
limit cases remain green.

- [ ] **Step 4: Commit the parser correction**

```bash
git add -- \
  tldw_chatbook/LLM_Calls/hosted_chat.py \
  Tests/LLM_Calls/test_hosted_chat.py \
  Tests/Chat/test_kimi_zai_native_tools.py
git commit -m "fix(llm): accept Moonshot stream fingerprints"
```

### Task 2A: Accept The UAT-Observed Terminal Choice Usage

**Files:**
- Modify: `tldw_chatbook/LLM_Calls/hosted_chat.py`
- Modify: `Tests/LLM_Calls/test_hosted_chat.py`
- Modify: `Tests/Chat/test_kimi_zai_native_tools.py`

- [ ] **Step 1: Pin accepted and fail-closed choice usage RED cases**

Add a terminal stream choice containing a mapping-valued `usage` field. Add
controls proving non-mapping usage, usage before a finish reason, simultaneous
top-level plus choice usage, and a later conflicting duplicate all raise
`HostedChatProtocolError`. Update only the Moonshot joined fixture to reproduce
terminal choice usage followed by the identical trailing top-level event;
preserve Z.ai's existing single trailing top-level usage event.

- [ ] **Step 2: Apply the minimal choice allowlist correction**

Admit `usage` in the strict choice allowlist, require it to be a mapping, reject
same-event dual placement, reuse the existing terminal-placement check, and
accept exactly one later top-level mapping only when it canonically equals the
recorded usage with JSON types preserved. Reject further duplicates and
top-level-only repeats. Do not coerce, merge, log, or persist provider usage
data.

- [ ] **Step 3: Verify GREEN and commit**

Run the exact choice-usage parser cases and Moonshot joined node, then both
touched test files and Task 3's related matrix. Ruff, formatting, compilation,
and diff checks must pass before committing the correction.

### Task 3: Run Focused Related Verification

**Files:**
- Test: `Tests/LLM_Calls/test_hosted_chat.py`
- Test: `Tests/LLM_Calls/test_moonshot.py`
- Test: `Tests/Chat/test_kimi_zai_provider_contract.py`
- Test: `Tests/Chat/test_kimi_zai_native_tools.py`
- Test: `Tests/Chat/test_console_provider_gateway.py`
- Test: `Tests/Agents/test_provider_continuation_runtime.py`
- Test: `Tests/Chat/test_sensitive_llm_logging.py`
- Test: `Tests/Chat/test_live_moonshot_zai_api.py`

- [ ] **Step 1: Run the provider and joined continuation matrix**

```bash
../../.venv/bin/python -m pytest -q \
  Tests/LLM_Calls/test_hosted_chat.py \
  Tests/LLM_Calls/test_moonshot.py \
  Tests/Chat/test_kimi_zai_provider_contract.py \
  Tests/Chat/test_kimi_zai_native_tools.py \
  Tests/Chat/test_live_moonshot_zai_api.py \
  Tests/Chat/test_console_provider_gateway.py \
  -k 'moonshot or hosted or live_gate or live_subprocess'
```

Expected: all selected tests pass; paid cases skip because the live opt-in/key
are absent. If loopback binding is sandbox-blocked, rerun only the affected
`allow_network` nodes with loopback permission.

- [ ] **Step 2: Run focused AgentService and privacy regressions**

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Agents/test_provider_continuation_runtime.py \
  Tests/Chat/test_sensitive_llm_logging.py \
  -k 'moonshot or hosted or continuation'
```

Expected: all selected tests pass with no credential/payload canary in output.

- [ ] **Step 3: Run targeted static checks**

```bash
../../.venv/bin/python -m ruff check \
  tldw_chatbook/LLM_Calls/hosted_chat.py \
  Tests/LLM_Calls/test_hosted_chat.py \
  Tests/Chat/test_kimi_zai_native_tools.py
../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/LLM_Calls/hosted_chat.py \
  Tests/LLM_Calls/test_hosted_chat.py \
  Tests/Chat/test_kimi_zai_native_tools.py
../../.venv/bin/python -m compileall -q tldw_chatbook/LLM_Calls/hosted_chat.py
git diff --check
```

Expected: all commands exit zero. Do not format unrelated legacy files.

- [ ] **Step 4: Mutation-prove the regression**

Temporarily remove `"system_fingerprint"` from the streaming allowlist.
Run the two exact Task 1 nodes and expect both to fail. Restore the production
line and rerun them GREEN. Confirm `git diff --check` afterward.

### Task 4: Rebase And Review The Final Code Candidate

**Files:**
- Review: `origin/dev...HEAD`
- Test: the Task 1 and Task 3 focused paths

- [ ] **Step 1: Fetch and inspect upstream overlap**

```bash
git fetch origin dev
git log --oneline HEAD..origin/dev -- \
  tldw_chatbook/LLM_Calls/hosted_chat.py \
  Tests/LLM_Calls/test_hosted_chat.py \
  Tests/Chat/test_kimi_zai_native_tools.py
```

Expected: any upstream overlap is understood before rewriting the branch.

- [ ] **Step 2: Rebase file-by-file**

```bash
git rebase origin/dev
```

Resolve conflicts one file at a time and stage only exact resolved paths—never
`git add -A`. Confirm no upstream task filename or unrelated file was reverted.

- [ ] **Step 3: Verify committed and working-tree whitespace**

```bash
git diff --check origin/dev...HEAD
git diff --check
```

Expected: both commands exit zero.

- [ ] **Step 4: Rerun the focused candidate gates**

Rerun the Task 1 exact regression nodes, both Task 3 pytest commands, Ruff,
format check, compileall, and the mutation proof on the rebased HEAD.

Expected: all focused checks are green and the mutation still fails for the
intended reason.

- [ ] **Step 5: Request correctness/privacy/YAGNI review**

Use `superpowers:requesting-code-review` against `origin/dev...HEAD`. Reviewers
must check the exact live evidence, bounded metadata type/size validation,
unknown-key rejection, joined reachability, privacy, and whether one neutral
allowlist entry is the smallest fix.

- [ ] **Step 6: Resolve every actionable finding before UAT**

For each finding, reproduce it with a focused failing test when behavior
changes, implement the smallest correction, rerun affected tests plus Task 1,
and commit the fix. Repeat Step 3 and the relevant Task 3 gates. Do not begin
paid UAT while any actionable review finding remains.

### Task 5: Run Paid UAT And Close Task Evidence

**Files:**
- Test: `Tests/Chat/test_live_moonshot_zai_api.py`
- Read only: ignored main-checkout `moonshot-api-key.txt`
- Modify: `backlog/tasks/task-16074 - Make-Moonshot-live-native-tool-continuation-pass.md`
- Modify: TASK-16074 spec and plan evidence only

- [ ] **Step 1: Prove the credential file is ignored and untracked**

```bash
git -C /Users/macbook-dev/Documents/GitHub/tldw_chatbook check-ignore moonshot-api-key.txt
git ls-files --error-unmatch moonshot-api-key.txt
```

Expected: the first command identifies the ignore rule; the second exits
nonzero because the credential is not tracked.

- [ ] **Step 2: Record the reviewed code SHA and run exactly the paid node**

Record `git rev-parse HEAD`, then load the key into the child environment
without printing it:

```bash
TLDW_LIVE_MOONSHOT=1 \
MOONSHOT_API_KEY="$(</Users/macbook-dev/Documents/GitHub/tldw_chatbook/moonshot-api-key.txt)" \
../../.venv/bin/python -m pytest -q \
  'Tests/Chat/test_live_moonshot_zai_api.py::test_live_hosted_text_and_native_tool[moonshot]'
```

Expected: `1 passed`. The harness proves one calculator call, exact
arguments/result, provider continuation, and the final marker on the reviewed,
rebased production SHA.

- [ ] **Step 3: Run a pre-closeout tracked-tree privacy search**

```bash
git grep -l -F \
  -f /Users/macbook-dev/Documents/GitHub/tldw_chatbook/moonshot-api-key.txt \
  -- .
```

Expected: exit 1 and no output. `-l` reports filenames only if a regression
exists; it never prints the credential or matching line. This is an early
guard; Step 7 repeats it after the evidence commit.

- [ ] **Step 4: Audit the reviewed code candidate diff**

```bash
git status --short
git diff --check origin/dev...HEAD
git diff --stat origin/dev...HEAD
```

Inspect `git diff origin/dev...HEAD` and verify no key file, raw live body,
complete request payload, diagnostic probe, or unrelated change is present.

- [ ] **Step 5: Complete task hygiene after review and UAT**

Check all four ACs and add concise Implementation Notes containing the root
cause, RED/GREEN commands, paid UAT SHA/result, privacy audit, ADR-063 reuse,
review result, and exact modified files. Set TASK-16074 to Done through Backlog
CLI and re-read `backlog task 16074 --plain` to ensure notes/checklists survived.

- [ ] **Step 6: Commit closeout documentation and verify committed whitespace**

```bash
git add -- \
  'backlog/tasks/task-16074 - Make-Moonshot-live-native-tool-continuation-pass.md' \
  Docs/superpowers/specs/2026-08-13-task-16074-moonshot-live-tool-uat-fix-design.md \
  Docs/superpowers/plans/2026-08-13-task-16074-moonshot-live-tool-uat-fix.md
git diff --cached --check
git commit -m "docs(chat): close Moonshot live tool UAT"
git diff --check origin/dev...HEAD
```

Expected: task status/ACs/notes are committed and the full committed range is
whitespace-clean. Documentation-only closeout does not invalidate the paid
production SHA.

- [ ] **Step 7: Repeat privacy and complete-diff audits after evidence commit**

```bash
git grep -l -F \
  -f /Users/macbook-dev/Documents/GitHub/tldw_chatbook/moonshot-api-key.txt \
  -- .
git diff --check origin/dev...HEAD
git status --short
```

Expected: `git grep` exits 1 with no filenames, the committed range is clean,
and the worktree is empty. Inspect the final `origin/dev...HEAD` diff so the
Implementation Notes and task evidence are included in the privacy/scope audit.

## Execution Evidence

- RED/GREEN covered all three live stream mismatches plus strict malformed,
  conflicting, repeated, misplaced, and JSON-type-distinct controls.
- Final focused gates: 109 touched-file tests, 77 provider/Console tests, and 60
  AgentService/privacy tests passed; Ruff, formatting, focused mypy, compileall,
  and diff checks passed.
- Two independent final reviews approved code and spec.
- Paid Moonshot UAT passed on reviewed/rebased code SHA `da2816853` (`1 passed`
  in 16.82s).

### Task 6: Open, Review, Merge, And Clean Up The Follow-Up PR

**Files:**
- Review: the pushed branch and GitHub PR against `dev`

- [ ] **Step 1: Push the settled branch**

```bash
git push --set-upstream origin codex/moonshot-live-tool-uat-fix
```

Expected: the remote feature branch points at the verified local HEAD.

- [ ] **Step 2: Open a ready PR to `dev`**

Create a ready PR titled `fix(llm): accept Moonshot stream fingerprints`.
Reference TASK-16074 and ADR-063; include focused RED/GREEN, static, review,
privacy, and paid-UAT evidence without any credential, prompt, raw response, or
payload content.

- [ ] **Step 3: Inspect required checks and all review threads**

Use `gh pr checks <number>` and `gh pr view <number> --comments` plus the GitHub
API for unresolved inline review threads. Wait for required checks to complete;
for external CI providers, report only their details URL.

- [ ] **Step 4: Diagnose and fix each failure/comment separately**

For every actionable failure or review comment: reproduce, add/adjust a focused
test where behavior changes, implement the smallest fix, rerun only affected
and Task 1 gates, run Ruff/diff checks, commit, and push. Then recheck CI and
threads rather than assuming the push resolved them.

- [ ] **Step 5: Re-run paid UAT and evidence gates after relevant changes**

If any post-UAT commit changes `hosted_chat.py`, Moonshot transport, Console or
AgentService behavior, or the live harness, rerun Task 5 Step 2 on the new HEAD
and update task/PR evidence. Commit that evidence update, repeat Task 5 Step 7,
push it, and wait for required CI/checks to rerun on the new SHA. Re-open the
review-thread audit before proceeding. Documentation-only changes do not
require another paid call, but still require the evidence commit, privacy/diff
audit, push, and CI recheck when they alter tracked files.

- [ ] **Step 6: Merge only after all gates are green**

Confirm `git status --short` is empty, required checks pass, actionable review
threads are resolved, the branch is mergeable with current `dev`, and the
latest relevant-code SHA has a passing paid UAT. Merge the PR into `dev` and
verify the resulting merge commit contains the parser fix and task closeout.

- [ ] **Step 7: Clean up after the verified merge**

Use `superpowers:finishing-a-development-branch` from the main checkout to
remove this worktree and delete the merged local/remote feature branch.
Preserve the ignored local key file unless the user explicitly asks to delete
it.

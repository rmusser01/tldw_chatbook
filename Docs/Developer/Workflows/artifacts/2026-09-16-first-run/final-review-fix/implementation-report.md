# Single final-review fix wave

Status: DONE.

Worktree: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/workflows-authoring-dev`
Branch: `codex/workflows-sequential-design`
Base: `766e2ca43222703a9b07fe67cbcb9dcf3a75811f`
Commits:
- `cdc7c57909a76cafc451757fb99f9aaa2f8719d4` — `fix(workflows): keep Open Note responsive during Note writes`
- `bcc8290dd2d883893c781a83a3e5cd0f13f27eb5` — `fix(workflows): clear busy status after Open Note retry`

Final HEAD: `bcc8290dd2d883893c781a83a3e5cd0f13f27eb5`.

## Implemented scope

Fixed the final review's sole Important finding: Open Note could block the app
loop on the existing Notes owner's mutex while a later Note worker held that
mutex and synchronously awaited its inner authority callback on the app loop.

`NotesInteropService.bound_notes_db` now accepts keyword-only `blocking=True`.
Its default worker contract is unchanged. Nonblocking contention raises
`BlockingIOError("note_destination_busy")` without entering, releasing someone
else's lock, manufacturing a database, or changing the route. A successful
acquire retains the exact cached-object identity check and releases in `finally`,
including when validation or the caller's body raises.

Open Note uses `blocking=False` and displays: “Local Notes is busy. Try Open Note
again when the current operation finishes.” Retry remains manual. Changed or
unavailable destinations retain the existing fail-closed message. Existing run,
Note, scope, user, owner, path, client, file-availability, and cache-identity checks
remain in place; navigation uses the same existing Library message and ID.
Successful validated navigation also clears and refreshes the transient status,
so returning to the retained Workflows screen does not show a stale busy warning.

Only these four source/test files changed:

- `tldw_chatbook/Notes/Notes_Library.py`
- `tldw_chatbook/UI/Screens/workflows_screen.py`
- `Tests/Workflows/test_local_steps.py`
- `Tests/UI/test_workflows_run.py`

ADR required: no new ADR.
ADR paths: `backlog/decisions/138-portable-workflow-definitions-and-local-execution.md`
and `backlog/decisions/125-lock-safe-private-sqlite-validation.md`.
Reason: the authorized optional mode is a small amendment within the existing
owner guard. No new owner, lock, worker, database, schema, helper, retry loop, or
infrastructure was introduced. No CSS/bundle/layout change.

Read the full fix brief and review, repository instructions, TASK-32691, governing
ADR sections, design-language constitution, and relevant testing-evidence lessons
(including controlled lock inversion, loop/worker waits, and pre-import profile
isolation). Applied systematic-debugging, test-driven-development, and
verification-before-completion skills; followed the supplied implementer
self-review/report contract with its full-suite clause overridden by the brief.

## Deterministic regression and limits of RED

The existing disposable-profile UI/session harness supplies real Notes owners,
real SQLite transactions, accepted review text, and a mocked HTTP transport.
The saved workflow has two sequential real Note steps. After review acceptance,
the first commits; a test-only wrapper around the *original* owner mutex holds
the second after acquisition but before its inner authority callback.

A queued `Button.Pressed` carries the confirmed first result into the real
screen handler. The wrapper records and rejects any contended blocking acquire
from the app-loop thread. The regression fails explicitly on that record. It
does not deliberately freeze the loop or measure how often the production race
occurs. It proves that the real handler requests the deadlocking operation at
the exact reachable boundary identified by the source trace, while retaining
the real mutex/owner and real Note effects. The worker's 30-second barrier
watchdog and unconditional `finally` release keep RED bounded and cleanup safe.
No arbitrary sleep, stress loop, child process, or generic test framework was
added for the regression.

GREEN verifies compositor-painted busy/retry status, an app-loop heartbeat,
real Quit/Stay controls, and responsive Cancel while the second worker remains
physically owned. Releasing after Cancel lets its real authority callback
observe Stop: it creates no second row, physically settles, and fresh Open Note
navigates to the exact first ID. A second parameter lets the second Note finish:
two distinct IDs retain the exact accepted text, an old queued first-result
event cannot navigate, and fresh Open Note opens the currently confirmed second
ID. No first-result history behavior is introduced.

Seven queued-event cases freshly reject changed scope, user, owner, DB path,
client, cache object, and absent cached route. Existing later-edit navigation
coverage remains. Guard tests cover contention without mutation/unlocking,
successful retry, default blocking worker behavior, and release on body/route
errors in both modes.

## Commands and raw evidence

All commands ran in the worktree above. Python and Ruff are the root venv's
`/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python` and `ruff`.
Artifacts are under
`Docs/Developer/Workflows/artifacts/2026-09-16-first-run/final-review-fix/`.

Before pytest, executed the existing standalone stdlib/third-party bootstrap:

```sh
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python Tests/Workflows/test_file_to_note_integration.py .superpowers/sdd/2026-09-16-workflows-first-run/final-fix-bootstrap
```

Output was the private bootstrap's absolute `config/config.toml` path (exit 0).
No app/core module was imported by this standalone preparation. Every pytest
invocation used this exact environment scrub/exec body, with its exact argument
list recorded below; stdout/stderr were piped through `tee` to the named raw
artifact with `set -o pipefail`:

```python
import os, re, sys
for name in tuple(os.environ):
    if re.search(r'KEY|TOKEN|PASSWORD|SECRET|CREDENTIAL|PROXY|CACHE|BASE_URL|API_URL|^HF_|^HUGGING|^TRANSFORMERS_|^TLDW_|^TASK6_|^WORKFLOW_UI_CAPTURES$', name):
        del os.environ[name]
os.environ.update(PYTHONPATH='.', TLDW_TEST_CONFIG_ROOT='/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/workflows-authoring-dev/.superpowers/sdd/2026-09-16-workflows-first-run/final-fix-bootstrap')
# os.execv(sys.executable, [sys.executable, *the exact arguments below])
```

RED arguments / `red.txt` (before production changes):

```text
-m pytest -o addopts= --basetemp=.superpowers/sdd/2026-09-16-workflows-first-run/final-fix-red-tmp -q Tests/UI/test_workflows_run.py Tests/Workflows/test_local_steps.py -k 'open_note_during_second or bound_context_nonblocking or bound_context_default_worker or bound_context_releases'
```

Exit 1: `5 failed, 1 passed, 125 deselected, 1 warning in 6.98s`.
Both UI cases failed with `AssertionError: Open Note requested a contended
blocking acquire on the app loop`. Three API cases failed with
`TypeError: NotesInteropService.bound_notes_db() got an unexpected keyword
argument 'blocking'`. The blocking-default control passed. No EPERM or unrelated
environment failure was classified as RED.

GREEN arguments / `green.txt`:

```text
-m pytest -o addopts= --basetemp=.superpowers/sdd/2026-09-16-workflows-first-run/final-fix-green-tmp -q Tests/UI/test_workflows_run.py Tests/Workflows/test_local_steps.py -k 'open_note_during_second or bound_context_nonblocking or bound_context_default_worker or bound_context_releases or queued_open_note_rechecks_destination'
```

Exit 0: `13 passed, 118 deselected, 1 warning in 23.82s`.

Covering arguments / `covering.txt`:

```text
-m pytest -o addopts= --basetemp=.superpowers/sdd/2026-09-16-workflows-first-run/final-fix-covering-tmp -q Tests/UI/test_workflows_run.py Tests/Workflows/test_local_steps.py Tests/Workflows/test_session_lifecycle.py Tests/Notes/test_notes_library_unit.py
```

Exit 0: `229 passed, 1 warning in 160.50s (0:02:40)`.

Separate ProductionApp arguments / `production-lifecycle.txt`:

```text
-m pytest -o addopts= --basetemp=.superpowers/sdd/2026-09-16-workflows-first-run/final-fix-production-tmp -q Tests/ProductionApp/test_workflows_session_lifecycle.py
```

Exit 0: `12 passed, 1 warning in 12.60s`.
Its dedicated `Tests/ProductionApp/conftest.py` owns its separate
collection-time private root. It is not co-collected with the UI/Notes selection.

## Static attribution and self-review

Ruff version: `0.16.6`. Same effective configuration for baseline and current.
The existing stdlib-only checker was reused without modification:

```sh
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python .superpowers/sdd/2026-09-16-workflows-first-run/check_static_delta.py --base 766e2ca43222703a9b07fe67cbcb9dcf3a75811f --ruff /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff tldw_chatbook/Notes/Notes_Library.py tldw_chatbook/UI/Screens/workflows_screen.py Tests/UI/test_workflows_run.py Tests/Workflows/test_local_steps.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check tldw_chatbook/Notes/Notes_Library.py tldw_chatbook/UI/Screens/workflows_screen.py Tests/UI/test_workflows_run.py Tests/Workflows/test_local_steps.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff format --check tldw_chatbook/Notes/Notes_Library.py tldw_chatbook/UI/Screens/workflows_screen.py Tests/UI/test_workflows_run.py Tests/Workflows/test_local_steps.py
git diff --check -- Tests/UI/test_workflows_run.py Tests/Workflows/test_local_steps.py tldw_chatbook/Notes/Notes_Library.py tldw_chatbook/UI/Screens/workflows_screen.py
```

`static-delta.json`: exit 0, zero introduced diagnostics or formatter edits.
Every retained finding maps to its complete unchanged baseline source span;
each formatter edit matches its exact baseline edit, not merely net counts.
Notes retains 115/115 lint findings and 14/14 formatter edits. The UI and both
test files have zero/zero findings and formatter edits.

`static-current-ruff.txt`: exit 1, `Found 115 errors.` (all legacy Notes debt).
`static-current-format.txt`: exit 1, `1 file would be reformatted, 3 files already
formatted` (only legacy Notes). Source whitespace check: exit 0, no output.
The three clean files also passed a separate `ruff check` (exit 0, `All checks
passed!`). Only the two test files were passed to the formatter; no blanket
Notes formatting occurred.

Self-review read the complete owned diff: only the existing mutex acquisition
and Open Note branch changed in production. All destination checks remain,
the default worker call sites and inner fresh-authority callback are untouched,
and exception release uses `finally` only after a successful acquire. Tests
exercise real persisted rows and actual navigation, including successful
controls; they do not merely assert mock calls or hide the stale-event path by
disabling a button. Initial self-review identified no additional issue; the
controller's subsequent stale-status finding and its correction are recorded
below, including a fresh behavioral RED/GREEN.

## Limits and handoff

- Requests emits the existing dependency-version warning concerning urllib3 /
  chardet / charset_normalizer. Passing tests are not claimed warning-free.
- No full sweep, live localhost:9099 request, real model call, host-profile import
  or repair, credential read, dependency installation, child agent, or reviewer.
- Mock HTTP is deliberate. This fix does not requalify model behavior, native
  terminal input/fonts, unrelated historical failures, or whole-repository health.
- Controller-owned review record, plan, task, guide, lessons, handoff/final-review
  evidence and unrelated TASK-32601 / `.uat-workflows-9NUT5t` are outside this
  change. They were not staged. This report is saved scratch, not staged.
- Only the four listed files and this wave's exact raw-artifact subtree are
  authorized for the commit. No push, PR, merge, or controller task-status change.

## Initial commit and state (before the scoped status follow-up)

Staged exactly the four source/test files plus these seven raw outputs:
`red.txt`, `green.txt`, `covering.txt`, `production-lifecycle.txt`,
`static-delta.json`, `static-current-ruff.txt`, `static-current-format.txt`, all
under the authorized `final-review-fix/` subtree. Verified the complete staged
path list before committing. No active Git hook was installed (sample files
only), so no review agent or model invocation was triggered.

The initial sandboxed `git add` was denied while creating the worktree's
`index.lock` in the root repository's Git metadata directory. The exact-path
staging operation then succeeded with sandbox escalation; no automatic approval
review rejection or outstanding permission blocker occurred.

```sh
git commit -m "fix(workflows): keep Open Note responsive during Note writes"
```

Exit 0:

```text
[codex/workflows-sequential-design cdc7c57909] fix(workflows): keep Open Note responsive during Note writes
 11 files changed, 3778 insertions(+), 4 deletions(-)
```

Post-commit `git rev-parse HEAD` returned `cdc7c57909a76cafc451757fb99f9aaa2f8719d4`.
`git diff --cached --name-only` returned no paths. `git diff HEAD --exit-code --`
with the four exact files and artifact subtree returned exit 0/no output.
`git status --short` retained only controller/unrelated work: review record,
guide, plan, lessons, TASK-32601, `.uat-workflows-9NUT5t/`, controller handoff
output and final branch review output. No owned source/artifact changes remain.

The controller's separate joined actual-app integration run is not duplicated,
counted as this implementer's evidence, or included in this commit. No correctness
concerns remain from this scoped fix; the disclosed warning, legacy static debt,
and verification limits still apply.

## Controller-requested status follow-up within the same wave

After the initial commit, the controller identified that `_error` was never
cleared on successful Open Note navigation. Source inspection confirmed that
`_show_status` prioritizes this persistent field. The existing multi-Note
regression now installs and retains the actual Workflows screen, completes its
successful retry/navigation, returns to that same screen, scrolls the status
into view, and checks the compositor-painted output for a stale busy warning.
Both cancellation and completion variants failed with the actual busy sentence
still painted. This is a visible stale-status regression, not merely a private
field assertion.

The authorized correction adds only two production lines in the same Open Note
handler: clear `_error` and refresh status after the captured destination guard
succeeds, before posting the existing navigation message. Contention and changed
destination failures retain their messages; no route/authority checks changed.
No new owner, UI structure, callback, or broader behavior was added.

Used the same environment scrub, exact private bootstrap, root venv, worktree,
and `set -o pipefail`/`tee` raw-output capture described above.

Follow-up RED / `retry-status-red.txt`:

```text
-m pytest -o addopts= --basetemp=.superpowers/sdd/2026-09-16-workflows-first-run/final-fix-retry-status-red-tmp -q Tests/UI/test_workflows_run.py -k 'open_note_during_second'
```

Exit 1: `2 failed, 55 deselected, 1 warning in 9.71s`.
Both failures: `AssertionError: assert 'busy' not in ...`, with painted output
containing `notes is busy. try open note again when the current operation finishes.`
Both had already verified successful exact-ID navigation before returning.

Follow-up GREEN / `retry-status-green.txt`:

```text
-m pytest -o addopts= --basetemp=.superpowers/sdd/2026-09-16-workflows-first-run/final-fix-retry-status-green-tmp -q Tests/UI/test_workflows_run.py -k 'open_note_during_second or queued_open_note_rechecks_destination'
```

Exit 0: `9 passed, 48 deselected, 1 warning in 23.80s`.
The same pre-existing Requests dependency warning remains.

Static checks on exactly the two changed files:

```sh
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff check tldw_chatbook/UI/Screens/workflows_screen.py Tests/UI/test_workflows_run.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/ruff format --check tldw_chatbook/UI/Screens/workflows_screen.py Tests/UI/test_workflows_run.py
git diff --check -- tldw_chatbook/UI/Screens/workflows_screen.py Tests/UI/test_workflows_run.py
```

All exit 0; outputs respectively `All checks passed!`, `2 files already
formatted`, and no output. Notes guard and local-step tests are unchanged since
their initial covering/static checks. Per the controller's explicit scope,
the 229-test selection and ProductionApp 12 were not repeated for these two
status-clear lines. Their earlier results remain historical covering evidence,
not claims of a post-follow-up rerun.

Self-review of the follow-up diff confirmed two production lines plus seven
lines extending the existing test. Staged exactly those two files and the two
new raw RED/GREEN outputs. The original four-file scope remains the complete
source/test scope of both commits.

```sh
git commit -m "fix(workflows): clear busy status after Open Note retry"
```

Exit 0:

```text
[codex/workflows-sequential-design bcc8290dd2] fix(workflows): clear busy status after Open Note retry
 4 files changed, 1982 insertions(+)
```

Final verification: HEAD is `bcc8290dd2d883893c781a83a3e5cd0f13f27eb5`, the index
is empty, and `git diff HEAD --exit-code --` against the four allowed files and
owned artifact subtree returns exit 0/no output. Controller and unrelated changes
remain unstaged, including `controller-postfix-integration.txt`. No push/PR/merge.

The controller separately reported its joined actual-app check as 2 passed /
1 live skip / 1 existing warning in 17.66s, with no localhost:9099 POST. This is
controller-owned evidence, not a test this implementer ran or counted. No
duplicate joined run was performed. Both commits and all owned raw evidence are
saved; no known correctness concerns remain within this fix wave's scope.

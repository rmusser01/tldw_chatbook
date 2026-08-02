# TASK-859 Subscription Security Configuration Consolidation Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use `superpowers:test-driven-development`
> for implementation and `superpowers:verification-before-completion` before
> any completion claim.

**Goal:** Remove the five unread `[subscriptions.security]` settings and the
unused subscription-local denylist while preserving the HTTP/HTTPS boundary and
proving that the real `[web_security]` switch controls subscription runtime
behavior.

**Architecture:** `tldw_chatbook.Utils.egress` remains the only production
owner of DNS, private-address, and cloud-metadata policy. The subscription
validator retains its HTTP/HTTPS allowlist because egress intentionally returns
early when its kill switch is disabled. Shipped configuration and documentation
name only controls that the runtime actually consumes.

**Tech stack:** Python 3.11+, TOML, `ast`, pytest, Ruff.

**Design:**
`Docs/superpowers/specs/2026-08-02-subscription-security-config-consolidation-design.md`

**Backlog:**
`backlog/tasks/task-859 - Delete-or-implement-the-unread-subscriptions.security-switches-and-stale-metadata-denylist.md`

**ADR required:** no

**ADR path:** N/A

**Reason:** The change removes misleading unused configuration and duplicate
dead policy data while preserving TASK-328's established egress, TLS, and XML
runtime boundaries. It does not introduce a new architectural decision.

---

## Task 0: Synchronize the approved plan with current `dev`

**Files:** None expected unless upstream conflicts require plan revalidation.

### Step 1: Fetch and rebase before implementation

From the repository's main worktree, fetch `origin/dev`; then rebase this
feature branch:

```bash
git fetch origin dev
git rebase origin/dev
git merge-base --is-ancestor origin/dev HEAD
```

Expected: the rebase succeeds and the ancestry check exits zero. Resolve any
conflict by preserving both current upstream behavior and this approved task
scope; never restore stale line-based assumptions from the original task.

### Step 2: Revalidate task relevance and baseline

Repeat the production grep for the five fake settings and duplicate constants,
then run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Utils/test_egress.py \
  Tests/Subscriptions/test_subscription_egress_wiring.py -q
```

Expected: the task remains relevant and the focused baseline is green. If
upstream already satisfies an acceptance criterion, update the plan before
writing tests rather than reintroducing removed code.

---

## Task 1: Add failing configuration and policy-ownership sentinels

**Files:**

- Create:
  `Tests/Subscriptions/test_subscription_security_config_contract.py`
- Read:
  `tldw_chatbook/config.py`
- Read:
  `tldw_chatbook/Subscriptions/security.py`
- Read:
  `tldw_chatbook/Utils/egress.py`

### Step 1: Write the shipped-configuration failure

Create a direct configuration test that parses the production template:

```python
import tomllib

from tldw_chatbook.config import CONFIG_TOML_CONTENT


def test_shipped_config_does_not_advertise_subscription_security_switches():
    shipped = tomllib.loads(CONFIG_TOML_CONTENT)

    assert "security" not in shipped["subscriptions"]
```

This must inspect the real template, not a copied fixture.

### Step 2: Write the metadata single-owner sentinel

In the same test module, retain the following fixed regression baseline:

```python
CANONICAL_METADATA_ENDPOINTS = {
    "169.254.169.254",
    "100.100.100.200",
    "fd00:ec2::254",
    "metadata.google.internal",
    "metadata.azure.com",
}
CANONICAL_EGRESS_PATH = Path("Utils/egress.py")
```

Assert every endpoint has exactly `{CANONICAL_EGRESS_PATH}` as its production
owner. Also derive additional current endpoints from the top-level
`Assign`/`AnnAssign` declarations of `_METADATA_IPS` and
`METADATA_HOSTNAMES` in `Utils/egress.py`, and apply the same single-owner check
to the union. Scanning AST string constants avoids false positives from
comments while still catching renamed duplicate tables. Collect the inventory
in a module-scoped fixture that parses every production Python file once,
reusing the already-parsed egress tree, and sort endpoints and paths so failure
diagnostics are deterministic.

### Step 3: Write the scheme-policy sentinel

AST-scan literal collections only under the production `tldw_chatbook/` package.
Fail when any list, tuple, set, or frozenset contains three or more of these
blocked-scheme markers:

```python
BLOCKED_SCHEME_MARKERS = {"file", "ftp", "gopher", "javascript", "data"}
```

Read the `SecurityValidator` class assignments from
`Subscriptions/security.py`'s AST without importing the `Subscriptions`
package, then assert the source contract explicitly:

```python
assert not validator_duplicate_attributes
assert validator_allowed_schemes == {"http", "https"}
```

The threshold allows ordinary isolated scheme strings elsewhere while catching
an authoritative-looking denylist. The retained HTTP/HTTPS allowlist is not a
denylist and is required independently of the SSRF kill switch. There are no
excluded production paths: any future legitimate duplicate must first add an
explicit consistency test and deliberately revise this sentinel. Sort scheme
names and violation locations in diagnostics so hash iteration cannot make
failure output unstable.

### Step 4: Run the new contract tests and verify RED

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Subscriptions/test_subscription_security_config_contract.py -q
```

Expected: failures showing that the shipped `security` table still exists and
that `Subscriptions/security.py` still owns the duplicate constants. Confirm
the failures are assertion failures for the intended contracts, not import or
test-construction errors.

### Step 5: Commit the red tests

```bash
git add Tests/Subscriptions/test_subscription_security_config_contract.py
git commit -m "test(task-859): guard subscription security policy ownership"
```

---

## Task 2: Add subscription runtime characterization tests

**Files:**

- Modify: `Tests/Subscriptions/test_subscription_egress_wiring.py`
- Modify: `Tests/Utils/test_egress.py`

### Step 1: Prove the real kill switch changes subscription behavior

Add a test that:

1. makes egress DNS resolve a hostname to `192.168.1.10`;
2. supplies `get_cli_setting("web_security", "enabled", ...)` from mutable test
   state;
3. asserts `SecurityValidator.validate_feed_url()` raises `SSRFError` while
   enabled; and
4. flips the state to disabled and asserts the same HTTP URL is normalized and
   returned.

This exercises the production subscription function and production egress
function together. It must not merely assert that a TOML value round-trips.

### Step 2: Pin the disabled-policy scheme boundary

Add a separate regression test with `[web_security].enabled = false` that
asserts `ftp://example.com/feed` still raises `SSRFError` with the existing
"not allowed" message. This is the security invariant caught during spec
review.

### Step 3: Preserve input error contracts

Add parameterized direct-function assertions that empty input and missing
scheme/host still raise `ValueError`, while an unsupported scheme raises
`SSRFError`.

### Step 4: Cover every canonical metadata endpoint behaviorally

Parameterize the existing egress metadata tests so both canonical hostnames and
all three canonical IP addresses are exercised:

```python
METADATA_HOSTS = ("metadata.google.internal", "metadata.azure.com")
METADATA_IPS = ("169.254.169.254", "100.100.100.200", "fd00:ec2::254")
```

Each must produce `allowed is False` and `reason == "metadata"`. Test IPv6 as a
bracketed URL literal. These are characterization tests and may already pass;
their purpose is to keep the canonical owner complete.

### Step 5: Run the behavior tests as characterization

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Subscriptions/test_subscription_egress_wiring.py \
  Tests/Utils/test_egress.py -q
```

Expected: all pass. These tests characterize the real surviving config and
scheme contracts before dead configuration and constants are removed. Task 1
already supplies the required red tests for the implementation.

### Step 6: Commit the characterization tests

```bash
git add Tests/Subscriptions/test_subscription_egress_wiring.py Tests/Utils/test_egress.py
git commit -m "test(task-859): pin subscription egress config behavior"
```

---

## Task 3: Remove the false shipped configuration

**Files:**

- Modify: `tldw_chatbook/config.py`
- Test: `Tests/Subscriptions/test_subscription_security_config_contract.py`

### Step 1: Delete only the false table

Remove this complete block from `CONFIG_TOML_CONTENT`:

```toml
# Security settings
[subscriptions.security]
enable_xxe_protection = true
enable_ssrf_protection = true
verify_ssl_certificates = true
max_redirects = 5
request_timeout = 30
```

Do not remove or rename `[web_security]`, the monitor-only optional
`ssl_verify` mapping input (which defaults verification on), or other
`[subscriptions]` settings in this task. `ssl_verify` is not a persisted
subscriptions DB field or UI control.

### Step 2: Prove the template still parses

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Subscriptions/test_subscription_security_config_contract.py::test_shipped_subscription_config_has_no_security_child_table \
  Tests/test_config_console_defaults.py -q
```

Expected: all selected tests pass. The still-red policy-ownership assertions
remain intentionally unselected until Task 4.

### Step 3: Commit the configuration removal

```bash
git add tldw_chatbook/config.py
git commit -m "fix(config): remove unread subscription security switches"
```

---

## Task 4: Consolidate subscription policy ownership

**Files:**

- Modify: `tldw_chatbook/Subscriptions/security.py`
- Test: `Tests/Subscriptions/test_subscription_egress_wiring.py`
- Test: `Tests/Subscriptions/test_subscription_security_config_contract.py`

### Step 1: Remove only dead duplicate policy data

Delete `SecurityValidator.BLOCKED_SCHEMES` and
`SecurityValidator.METADATA_ENDPOINTS`. Retain:

```python
ALLOWED_SCHEMES = {"http", "https"}
```

and its pre-egress validation so disabling SSRF never enables unsupported
subscription transport schemes.

Do not change `validate_feed_url()` parsing, normalization, delegation call
shape, or denial-reason mapping in this task; those behaviors are outside the
acceptance criteria.

### Step 2: Run focused tests and verify GREEN

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Subscriptions/test_subscription_security_config_contract.py \
  Tests/Subscriptions/test_subscription_egress_wiring.py \
  Tests/Utils/test_egress.py -q
```

Expected: all pass. Confirm the disabled-SSRF FTP regression and all canonical
metadata cases pass.

### Step 3: Commit the production consolidation

```bash
git add tldw_chatbook/Subscriptions/security.py
git commit -m "fix(subscriptions): consolidate URL policy ownership"
```

---

## Task 5: Correct subscription architecture documentation

**Files:**

- Modify: `tldw_chatbook/Subscriptions/SUB-Arch.md`

### Step 1: Replace the stale configuration example

Remove the `[subscriptions.security]` example. Show the real egress contract:

```toml
[web_security]
enabled = true
allowed_hosts = []
```

### Step 2: Document the other real owners accurately

State that:

- FeedMonitor and URLMonitor accept an optional `ssl_verify` mapping value and
  default verification on; it is not a persisted subscriptions DB field or UI
  control;
- redirect bounds are code-owned by shared guarded-fetch helpers;
- network feed monitor/scraper parser modules prefer optional `defusedxml`,
  with their existing module-specific standard-library fallback behavior;
- the active OPML import WatchlistOpmlService uses stdlib
  `xml.etree.ElementTree` directly;
- `SecurityValidator.validate_xml_content()` is not part of the active parsing
  path; and
- legacy `[subscriptions.security]` tables are ignored and safe to delete.

Do not claim the current XML fallback is unconditional protection.

### Step 3: Verify retired names are absent from shipped surfaces

Run:

```bash
rg -n "enable_xxe_protection|enable_ssrf_protection|verify_ssl_certificates|max_redirects|request_timeout|\[subscriptions\.security\]" \
  tldw_chatbook/config.py tldw_chatbook/Subscriptions/SUB-Arch.md
```

Expected: no matches. Unrelated `request_timeout` settings elsewhere are not in
scope and must not be mechanically removed.

### Step 4: Commit the documentation correction

```bash
git add tldw_chatbook/Subscriptions/SUB-Arch.md
git commit -m "docs(subscriptions): name active security controls"
```

---

## Task 6: Verify, review, and close TASK-859

**Files:**

- Modify:
  `backlog/tasks/task-859 - Delete-or-implement-the-unread-subscriptions.security-switches-and-stale-metadata-denylist.md`
- Modify:
  `Docs/superpowers/specs/2026-08-02-subscription-security-config-consolidation-design.md`
- Modify: this plan as steps complete
- Modify: `backlog/docs/lessons-testing-evidence.md`

### Step 1: Run the focused security/configuration matrix

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Subscriptions/test_subscription_security_config_contract.py \
  Tests/Subscriptions/test_subscription_egress_wiring.py \
  Tests/Utils/test_egress.py \
  Tests/test_config_console_defaults.py -q
```

Expected: all pass.

### Step 2: Run the broader subscription suite

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Subscriptions -q
```

Expected: all pass. If an unrelated baseline failure occurs, reproduce it on
the exact `origin/dev` base before attributing it.

### Step 3: Run the repository suite

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q
```

Expected: all pass. If the current development baseline is not green, record
the exact failures and verify unchanged failures against `origin/dev`; do not
claim a green full suite.

### Step 4: Run static and source checks

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check \
  tldw_chatbook/config.py \
  tldw_chatbook/Subscriptions/security.py \
  Tests/Subscriptions/test_subscription_egress_wiring.py \
  Tests/Subscriptions/test_subscription_security_config_contract.py
```

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check --ignore E402 \
  Tests/Utils/test_egress.py
```

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check \
  tldw_chatbook/Subscriptions/security.py \
  Tests/Subscriptions/test_subscription_security_config_contract.py
```

`origin/dev` was verified before implementation to already report that
`tldw_chatbook/config.py` and
`Tests/Subscriptions/test_subscription_egress_wiring.py` would be reformatted,
and full-file E402 findings in `Tests/Utils/test_egress.py` are verified
baseline debt. Do not reformat or mass-edit those files in this task; that
would create unrelated churn. Ruff lint (with E402 ignored only for the
baseline egress test file) and diff hygiene still cover the modified files.

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m compileall -q \
  tldw_chatbook/config.py tldw_chatbook/Subscriptions/security.py \
  Tests/Subscriptions/test_subscription_egress_wiring.py \
  Tests/Subscriptions/test_subscription_security_config_contract.py \
  Tests/Utils/test_egress.py
```

```bash
git diff --check
git diff --check origin/dev...HEAD
```

Expected: all pass.

### Step 5: Check security, performance, licence, and reusable lessons

Security is covered by the focused egress matrix and the disabled-policy FTP
regression. Run the installed-distribution contract that verifies project
licence metadata and packaged notices:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Packaging/test_installed_distribution.py::test_built_artifacts_match_distribution_contract -q
```

Inspect the task diff and record that no dependency, vendored code, licence
metadata, hot-path algorithm, I/O, or concurrency behavior changed. That is the
bounded performance/licence assessment for this deletion-only runtime change;
do not invent a meaningless benchmark.

This task did surface a reusable testing incident: removing a duplicate scheme
validator looked safe until review tested the SSRF-disabled state and found that
egress returns before scheme validation. Add a concise incident-based entry to
`backlog/docs/lessons-testing-evidence.md` explaining that consolidation tests
must exercise disabled/bypass modes before deleting an apparently duplicate
guard.

### Step 6: Perform code review and remedy findings

Use `superpowers:requesting-code-review`. Review the complete diff against
TASK-859 and this design, verify all findings before changing code, and rerun
the affected tests after every correction.

### Step 7: Complete Backlog and design hygiene

Only after verification:

1. check every TASK-859 acceptance criterion;
2. add concise Implementation Notes including behavior, files, tests, ADR check,
   review findings, and any baseline-attributed failures;
3. mark completed plan steps;
4. set the design status to `Implemented and verified`; and
5. set TASK-859 to `Done` through the Backlog CLI.

### Step 8: Run final completion verification

Use `superpowers:verification-before-completion`, rerun the focused matrix,
Ruff checks, compileall, the installed-distribution contract, and
`git diff --check`, then inspect `git status` and the complete branch diff before
making any completion claim. First prove the branch is based on the reviewed
development head:

```bash
git merge-base --is-ancestor origin/dev HEAD
git diff --check origin/dev...HEAD
```

### Step 9: Commit closeout documentation

```bash
git add \
  "backlog/tasks/task-859 - Delete-or-implement-the-unread-subscriptions.security-switches-and-stale-metadata-denylist.md" \
  backlog/docs/lessons-testing-evidence.md \
  Docs/superpowers/specs/2026-08-02-subscription-security-config-consolidation-design.md \
  Docs/superpowers/plans/2026-08-02-subscription-security-config-consolidation.md
git commit -m "docs(task-859): record pre-closeout verification"
```

## Pre-closeout verification evidence

Recorded 2026-08-02 before final whole-branch review. This is deliberately
pre-closeout evidence: TASK-859 remains In Progress, its acceptance criteria
remain unchecked, and no implementation notes or Done transition have been
added.

### Factual correction and scope deviation

The original task/design/plan overstated two existing implementation details.
TLS verification is not a persisted subscriptions DB `ssl_verify` column or UI
control: FeedMonitor and URLMonitor consume an optional mapping value and
default verification to enabled. XML handling is not one module-wide
optional-`defusedxml` policy: network feed monitor/scraper parser modules prefer
that optional dependency with module-specific stdlib fallbacks, while active
OPML import `WatchlistOpmlService` directly uses stdlib
`xml.etree.ElementTree`; `SecurityValidator.validate_xml_content()` has no
production callers. The task/design/plan now state these boundaries. This is a
documentation correction only; it neither expands scope into `ssl_verify`
persistence/UI work nor changes XML policy.

### A. Focused security/configuration matrix

Command:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Subscriptions/test_subscription_security_config_contract.py \
  Tests/Subscriptions/test_subscription_egress_wiring.py \
  Tests/Utils/test_egress.py \
  Tests/test_config_console_defaults.py -q
```

Result: **111 passed, 1 warning in 5.71s**. The warning was
`requests`' existing `RequestsDependencyWarning` for the installed urllib3 /
chardet-or-charset_normalizer combination.

### B. Broader subscription suite

Command:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Subscriptions -q
```

Result: **549 passed, 1 warning in 42.18s** (the same
`RequestsDependencyWarning`).

### C. Full repository suite

Command started:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q
```

Result: **interrupted and non-final** at 8% after approximately 12 minutes
because `origin/dev` advanced during the run and the branch became two commits
behind. The live output contained one unclassified `F` marker near 3%, but the
process was stopped before pytest emitted its failure node ID or error summary.
Do not treat this as a baseline failure or a full-suite result; rerun the full
suite after rebasing and capture any final node IDs before an origin/dev
comparison.

### D. Static and source checks

- Normal Ruff command for `config.py`, `Subscriptions/security.py`, the
  subscription wiring test, and the new contract test: **passed** (`All checks
  passed!`).
- `python -m ruff check --ignore E402 Tests/Utils/test_egress.py`: **passed**
  (`All checks passed!`). Full-file E402 findings remain verified baseline debt;
  no mass-formatting was performed.
- Ruff format check for `Subscriptions/security.py` and the new contract test:
  **failed**: `Tests/Subscriptions/test_subscription_security_config_contract.py`
  would be reformatted; `Subscriptions/security.py` was already formatted. This
  test file is outside this documentation closeout's ownership and needs
  controller follow-up before a final clean static claim.
- `compileall -q` for `config.py`, `Subscriptions/security.py`, both
  subscription test files, and `Tests/Utils/test_egress.py`: **passed**.
- `git diff --check` and `git diff --check origin/dev...HEAD`: **passed** at
  the time run.
- Targeted stale-key scan over `config.py` and `Subscriptions/SUB-Arch.md`:
  **no matches** (expected `rg` exit 1).

### E. Packaging and licence

Command:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Packaging/test_installed_distribution.py::test_built_artifacts_match_distribution_contract -q
```

Result: **1 passed in 11.52s**. This verifies the installed distribution's
licence metadata and packaged notices.

### F. Ancestry, branch surface, and bounded assessment

`git merge-base --is-ancestor origin/dev HEAD` returned **exit 1** after
`origin/dev` moved: the worktree status was ahead 14 and behind 2. Final
whole-branch review and closure are therefore blocked pending rebase onto the
current `origin/dev` and reruns of the necessary verification gates.

The recorded `git diff --name-only origin/dev...HEAD` surface contained only
the TASK-859 config/security/SUB-Arch production files, their three targeted
tests, and this task's plan/design/task docs. It contains no dependency, vendor,
or licence-metadata edits, and no hot-path algorithm, I/O, or concurrency
changes. No performance benchmark is warranted for that deletion-only scope;
the installed-distribution contract above is the licence/package evidence.

## Final closeout verification evidence

Recorded 2026-08-02 after the final rebase, sentinel corrections, and
whole-branch review. This section supersedes the pre-closeout blockers above;
the interrupted pre-rebase run and stale ancestry/format findings remain only
as historical evidence. Tasks 0 through 6 and all of their planned steps are
complete.

### A. Reviewed base and branch hygiene

The reviewed development base was
`5a7400801ef75e1f9b510d8ab22fa883ad8a597b`. At final verification the branch
was clean, 19 commits ahead, and not behind `origin/dev`.
`git merge-base --is-ancestor origin/dev HEAD`, branch diff/status checks,
`git diff --check`, the stale-key scan, and the source-only subscription import
isolation probe were green. The final diff remained limited to TASK-859's
config/security/documentation files, three targeted tests, and task evidence.

### B. Focused and subscription verification

The final focused matrix completed with **115 passed and 1
`RequestsDependencyWarning`**. The latest rebase also introduced unrelated
feed-server subscription tests. The first sandboxed `Tests/Subscriptions` run
therefore reported **33 socket-permission failures**; rerunning the same suite
with the required socket access outside the sandbox completed with **604 passed
and the same warning**.

Static verification was green: normal Ruff lint for the selected config,
security, wiring, and contract files; Ruff with only the inherited E402 rule
ignored for `Tests/Utils/test_egress.py`; selected Ruff format checks;
`compileall` for every changed Python file; and both working-tree and branch
`git diff --check` checks. The installed-distribution contract completed with
**1 passed**, covering the packaged licence metadata and notices.

### C. Full-suite result and exact-base comparison

The repository suite is **not claimed green**. It completed on the pre-rebase
feature branch pinned to base
`1ff1ee8a61aee628c7b0a48fefd917dff54c9b8a` with **27,214 passed, 203 skipped,
13 failed, 0 errors, and 124 warnings in 5h33m44s**. All 13 exact failing node
IDs were rerun in isolation on both that feature branch and an untouched
worktree at the pinned base; both produced the identical result: **6 failed and
7 passed**.

The six deterministic failures were the persistent-diagnostic inventory, LLM
destination-action census, unified-shell worker-count census, watchlists source
form `size0`, watchlists frequency control `size1`, and monochrome mosaic-color
contracts. The other seven nodes passed in isolation on both revisions and are
order-dependent full-suite flakes. No failure reproduced only on the TASK-859
branch, so the comparison found no branch-specific full-suite regression.

### D. Independent final review and bounded assessment

The final whole-branch reviewer independently reran the **115-test focused
matrix**, a **13-pass config source-template probe**, the **1-pass installed
distribution contract**, and static, ancestry, and branch-diff checks. All were
green and the reviewer reported no findings.

The security assessment is bounded by the focused egress matrix, including the
invariant that FTP remains rejected when egress policy is disabled. The diff
adds no dependency, vendored code, licence metadata, hot-path algorithm, I/O,
or concurrency behavior, so no performance benchmark is warranted. The
installed-distribution contract is the package/licence evidence. The incident
that exposed the disabled-egress scheme gap is recorded in
`backlog/docs/lessons-testing-evidence.md`: exercise disabled and bypass modes
before deleting an apparently duplicate guard.

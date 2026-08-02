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

In the same test module, parse every production `tldw_chatbook/**/*.py` file
with `ast`. For each exact canonical metadata string below, collect the relative
paths containing that string constant:

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
owner. Scanning AST string constants avoids false positives from comments while
still catching renamed duplicate tables.

### Step 3: Write the scheme-policy sentinel

AST-scan literal collections only under the production `tldw_chatbook/` package.
Fail when any list, tuple, set, or frozenset contains three or more of these
blocked-scheme markers:

```python
BLOCKED_SCHEME_MARKERS = {"file", "ftp", "gopher", "javascript", "data"}
```

Also assert the live subscription contract explicitly:

```python
assert "BLOCKED_SCHEMES" not in SecurityValidator.__dict__
assert "METADATA_ENDPOINTS" not in SecurityValidator.__dict__
assert SecurityValidator.ALLOWED_SCHEMES == {"http", "https"}
```

The threshold allows ordinary isolated scheme strings elsewhere while catching
an authoritative-looking denylist. The retained HTTP/HTTPS allowlist is not a
denylist and is required independently of the SSRF kill switch. There are no
excluded production paths: any future legitimate duplicate must first add an
explicit consistency test and deliberately revise this sentinel.

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

Do not remove or rename `[web_security]`, per-subscription `ssl_verify`, or
other `[subscriptions]` settings in this task.

### Step 2: Prove the template still parses

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Subscriptions/test_subscription_security_config_contract.py::test_shipped_config_does_not_advertise_subscription_security_switches \
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

- `ssl_verify` is per subscription;
- redirect bounds are code-owned by shared guarded-fetch helpers;
- production subscription parser modules prefer optional `defusedxml`, with
  their existing module-specific standard-library fallback behavior;
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
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check \
  tldw_chatbook/Subscriptions/security.py \
  Tests/Subscriptions/test_subscription_security_config_contract.py
```

`origin/dev` was verified before implementation to already report that
`tldw_chatbook/config.py` and
`Tests/Subscriptions/test_subscription_egress_wiring.py` would be reformatted.
Do not reformat either whole file in this task; that would create unrelated
churn. Ruff lint and diff hygiene still cover both modified files.

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m compileall -q \
  tldw_chatbook/config.py tldw_chatbook/Subscriptions/security.py \
  Tests/Subscriptions/test_subscription_egress_wiring.py \
  Tests/Subscriptions/test_subscription_security_config_contract.py
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
git commit -m "docs(task-859): record verified completion"
```

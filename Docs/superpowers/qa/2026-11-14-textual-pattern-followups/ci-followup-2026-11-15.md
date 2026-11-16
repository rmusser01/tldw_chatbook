# PR #3569 CI follow-up

The first CI run on rebased head
`374207ae0fd9a0371ed8b0068f53acce8b25e065` failed its persistent diagnostic
inventory check in preflight. The behavior lanes were skipped after that failure.
Both lint gates and the performance check passed; the other preflight guards
and both workflow-selector checks also passed.

Failure: [preflight job 132895830261](https://github.com/rmusser01/tldw_chatbook/actions/runs/43741623041/job/132895830261).
The inventory still assigned 73 diagnostic calls to SettingsScreen and none to
its category panels. The Speech migration moved two calls, leaving 71 in the
screen and two in `category_panels.py`; owner files therefore increase from 742
to 743. This was missed by the earlier targeted rebase checks.

## Statement and privacy review

The checker's prescribed `--statements` comparison uses the last inventory
commit, `09d8db39a51fa86b43b6c0498a5bfdc35d48a638`, as its lower bound. The two
removed statements match the two added ones:

- `logger.debug("Ignoring unsafe Speech Settings navigation context")`
- `logger.debug("Ignoring invalid Speech Settings navigation target")`

Both preserve their rejection conditions, DEBUG level and constant message.
Neither has interpolation, extra arguments, exception capture or user content.
The containing module changes the logger backend from standard-library logging
to Loguru. The existing `_forward_loguru_to_standard` bridge maps DEBUG to
standard-library DEBUG and retains source-path ownership. These calls create no
handler or persistent destination. The inventory classifies diagnostic owners
separately from configured sink topology; the backend transition is not a claim
that both logger objects are identical.

ADR required: no. This corrects derived ownership metadata for TASK-1378.3 under
existing ADR-303/304/161, without a new logging policy or production change.

## Validation

The ordinary local checker reproduced the exact two-row failure. After the
statement review, ordinary `--write` regenerated the inventory and a fresh
ordinary checker run passed: **743 owners, 8,886 diagnostic calls and 16 sink
files**. Independent artifact review confirmed that only the two expected owner
rows changed and all seven non-owner sections are identical, including sink
topology, path-privacy candidates, classification rules and schema.

Five existing focused fixture tests pass with zero failures/errors/skips:
case-sensitive owner ordering, position-independent call digests, content-change
detection, duplicate multiplicity and sink identity. These ran with `--noconftest`
and plugin autoload disabled; the only warnings concern the unused `asyncio_mode`
and `timeout` configuration options. No checker, test, production or workflow
code changed. No full suite was run.

The coordinator independently verified the raw evidence hashes and JUnit
outcomes. Artifact SHA-256:
`df83470f4b77fd6bb7988f8db83f83e4ae77f916a48611265a737f62cb03bcad`.
Green-check log SHA-256:
`48c3e84c9454f79743b5363ac8ae375b55b9deeb41b00c28c0dbccd1d3922d3e`.
Five-case JUnit SHA-256:
`61b9fa488bbe998e4cc472c68514df0e7c7bc55d1da6e8f202ac85d5c304a3cc`.

One earlier restricted-context scan was interrupted and is not counted as a
failure reproduction. The completed ordinary red/write/green runs used the
same checker without changed limits or acceptance rules. No sandbox-performance
cause is asserted.

The original CI failure remains retained; the passing local inventory check
does not itself establish that the previously skipped behavior lanes pass.

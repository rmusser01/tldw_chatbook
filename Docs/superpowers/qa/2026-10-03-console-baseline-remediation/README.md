# Console baseline failure remediation — verification record

The 21 unique failing nodes recorded by the original feature's four baseline logs now pass. Before this repair, 18 already passed and three failed. The remaining two token-budget doubles rejected current keyword parameters, and the fork barrier lost the real persistence boolean. Only those test doubles changed; production guards and observable assertions remain intact.

| Check | Result | Qualification |
|---|---|---|
| Recorded nodes before repair | 3 failed, 18 passed in 29.62s | All 21 historical nodes were selected. |
| Recorded nodes after repair | 21 passed in 32.18s | No warning summary. |
| Seven complete affected modules | 443 passed, 7 warnings in 348.15s | Six invalid-escape syntax warnings and FD growth of +751 (12 to 763, limit 200). |
| Committed-source equality, scoped Ruff, exact formatter ratchet, whitespace | Every exit 0 | Checks use commit `1c1a7e3566a01b2d0b963673272eae562025ef25`; both amended files have zero formatter debt. |

The 21-node selection overlaps the 443-test run; counts are not added. No skip, xfail, warning filter, resource threshold, production behavior, migration or dependency was changed. No full repository suite was requested or run. Descriptor growth remains unresolved; these results do not establish general resource cleanup.

The independent task review and final follow-up review are approved, with no Critical/Important findings or code fixes required. Controller static evidence records actual return codes `[0, 0, 0, 0]` for source equality, scoped Ruff, formatter ratchet and whitespace, resolving the task review's silent-output evidence concern. Publication is separately awaiting the PR base choice: the original local base branch does not exist on GitHub, and a direct PR to dev/main would include earlier local commits.

- [Task brief](task-1-brief.md), [implementation report](task-1-report.md), and [independent task review](task-1-review.md).
- [Final independent review](final-review.md), [controller static evidence](controller-static-evidence.json), and [rulings](rulings.md).
- [All 21 nodes](baseline-nodes.json) and [historical log provenance](baseline-node-provenance.json).
- [Pre-edit failures](verification-logs/recorded-red.log), [recorded nodes passing](verification-logs/recorded-green.log), and [complete module run](verification-logs/owners-green.log).
- Literal argv/environment are in `commands/`; complete output is in `verification-logs/`.
- [Formatter baseline](format-baseline.json), [execution ledger](execution-ledger.md), and [global constraints](global-constraints.md).
- [Source manifest](artifact-source-manifest.json) records hashes for each readable artifact and its byte-exact compressed source in `source-archives/`.

Commands retain their historical scratch paths. Readable copies trim trailing whitespace; compressed sources reproduce the original bytes. The original feature's [QA record](../2026-10-02-console-chat-starts/README.md) and historical failures remain preserved.

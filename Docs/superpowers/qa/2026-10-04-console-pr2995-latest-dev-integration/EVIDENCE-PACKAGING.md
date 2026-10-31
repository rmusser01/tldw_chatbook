# Qualification evidence packaging

[Download the complete evidence ZIP](qualification-evidence.zip). It retains every original file path and byte from the 2,118-file publication checkpoint. All entry hashes were verified before removing standalone copies. Markdown reports, review verdicts, ledgers, rulings and original manifests/audits remain available as regular files. [Packing receipt and entry index](evidence-packing.json) records every original hash and the ZIP hash.

The original manifest paths refer to those ZIP entries. Extract the archive into an empty directory to read the JSON maps, commands, logs, XML and verifier text. No dependencies or private profile/config/database files are bundled.

The PR previously changed 4,147 files; its first runtime file was number4,107, and PerfGuard was not scheduled. [GitHub documents limits on diff-based path filtering](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#git-diff-comparisons). Those observations are consistent with a filter limit; GitHub exposed no skip diagnostic. Lossless packaging keeps the source within normal review/diff enumeration while retaining all qualification evidence.

All 8,123 qualified source/test hashes stay exact. No workflow, job, test, permission or budget logic changed. All older QA directories and original frozen bytes remain preserved. Fresh current-head Qodo completion, required CI including both UI shards and PerfGuard, and latest dev ancestry still gate merge.

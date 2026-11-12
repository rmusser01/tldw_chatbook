# PR #2995 merged

[PR #2995](https://github.com/rmusser01/tldw_chatbook/pull/2995) merged into `dev` at 2026-10-05T10:52:49Z through normal repository protections. Merge commit `74557e202ac38c6d29510d0940a062ca7cc7f38b` joins the latest `dev`, `3146bbd8da2f30eaa97d72bd0ab860de4c40de75`, and the exact reviewed head, `3790437bdb686e40539cb82af64277b1f074f6b6`.

## Final gates

- PR Fast Lane, all three UI shards, and Derived passed on the reviewed head. PerfGuard's latency, boot, and storage checks executed successfully, as did Canvas reproduction from pinned public inputs. The earlier download failure and local diagnosis remain preserved.
- Qodo review `5412722170` completed for that head, with latest-commit confirmation `5992103347`. All review threads were resolved. Independent review traced the final reported routing omission to the static base schema: the live Console dynamically adds permitted routing arguments, and native and text serializers preserve them under ADR-147 and ADR-219. The [technical reply](https://github.com/rmusser01/tldw_chatbook/pull/2995#discussion_r4183161209) records the disposition. This source inspection does not claim a new live provider capture.
- Task 24 and Task 25 reviews approved the documentation and lossless compatibility changes for the 17 checkpoint fields. Evidence covers 111 targeted cases: 102 unchanged passes and nine corrected final passes. Original failures, aliases, and warning qualifications remain available. Verification stayed targeted; dependencies, caps, and authority gates were unchanged.

## Task records and preservation

Backlog task `34215.2` is Done, with all 11 acceptance criteria checked after the confirmed merge. Local bookkeeping commit `299d3dbf8051527acf2956fc73dbcd2280097acf` changes only the plan, task record, and testing lesson. Source and test bytes are unchanged. These records remain in the existing managed local checkout. Parent task `34215` was already Done, and its historical notes are intact.

This final archive contains the current feedback, independent routing review, source hash proof, executed CI steps, premerge gate snapshot, merge and parent receipts, local bookkeeping proof, and all 109 ordered rulings with the complete ledger. All 12,596 previously tracked QA files remain exact. The managed worktree, SDD ledger, private recovery evidence, and shared dirty checkout are preserved. The archive manifest, credential audit, and preservation receipt describe the evidence limits.

Automation closure is verified separately after this archive is committed. Task closure and this final archive are retained locally; there is no postmerge push or second PR.

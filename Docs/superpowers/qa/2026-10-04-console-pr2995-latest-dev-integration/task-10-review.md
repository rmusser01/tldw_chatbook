# Task 10 independent scoped review

Spec compliance: **Compliant**

Quality: **Approved**

No Critical, Important or Minor findings. No scoped source or test fix is required.

## Reviewed scope and source identity

Reviewed the exact eight pure placements and six retained store routes selected by `task-10-source-requirements.md` and `task-10-brief.md`. Read the frozen report, placement/freeze map, selected preflight report/map, immutable diff and named safe evidence. Governing ADRs are `backlog/decisions/092-console-chat-fork-copy-and-authority-boundary.md` and `backlog/decisions/219-console-chat-destinations-and-bounded-starts.md`; this implements their existing ownership boundaries and requires no new ADR.

BASE is `fe4eaf365c08705dcbe3edfa542fa0259d9b038e`; reviewed final is `1d55d8f89fd2d2b05dce65d3f011ece9eb1bf6c9`. The immutable 56,629-byte package `review-fe4eaf365c..1d55d8f89f.diff` has SHA256 `711fbfa9a558dca599921cac77f4037e66c30a487aeadcab1879620b28a18bdb` and independently matches the three-path Git diff with ten context lines. The complete changed-path list is exactly the two approved production paths and `Tests/Chat/test_console_chat_fork.py`. I used immutable Git source and static standard-library AST/hash/XML analysis. I did not execute product imports, tests, installs or subagents, and made no source, index or HEAD change.

## Spec and behavior checks

The placement map is complete:

| Original store route or stage responsibility | Pure owner and final lines |
| --- | --- |
| `_fork_message_state_is_eligible` | `console_chat_fork.py:1140`, `console_fork_message_state_is_eligible` |
| `_fork_visible_selection` | `console_chat_fork.py:1154`, `console_fork_visible_selection` |
| `_fork_attachment_fingerprint` | `console_chat_fork.py:1178`, `fingerprint_console_fork_attachments` |
| `_validate_fork_image_selections` | `console_chat_fork.py:1269`, `validate_console_fork_image_selections` |
| `_fork_video_fingerprint` | `console_chat_fork.py:1312`, `fingerprint_console_fork_video` |
| Stage message projection | `console_chat_fork.py:1346`, `project_console_fork_message` |
| Stage candidate comparison | `console_chat_fork.py:1473`, `console_fork_candidate_matches_fence` |
| `_validate_video_projection_tuple` | `console_chat_fork.py:1556`, `validate_console_fork_video_projection` |

All paths in that table are under `tldw_chatbook/Chat/`. The six store wrappers retain their original argument/return ASTs and static/class decorators at `console_chat_store.py:7010`, `7074`, `7081`, `7089`, `7098` and `7103`. Their forwarding documentation accurately describes the pure owner. The original all-or-nothing video responsibility docstring is preserved verbatim at its implementation owner. Independent comparison confirms all six original implementation bodies after removal of forwarding documentation/local imports and the explicit live fingerprint callback substitution. Original exact type checks, exception types/messages, JSON serialization, hash domains, tuple order and return behavior survive.

I independently expanded both stage calls into the final `_stage_fork_snapshot` and reconstructed the original whole method AST, including its docstring, signature and decorators. Its original ordering survives: initial exact fence/image-selection validation; destination validation and source lookup; UUID/turn/variant allocation; message projection followed by append, alias writers and predecessor advancement; configuration and citation capture; candidate construction and comparison; final exact fence/image-selection validation; combined refusal and return. `stage_fork_snapshot` retains `_fork_source_lock`. Mutable currentness, allocations, alias updates, destination/citation ownership, candidate construction and final authority remain in the store.

The complete named-definition map independently matches all 624 store functions/methods: exactly the selected seven definitions change and 617 remain identical. All 18 preexisting fork definitions retain their ASTs. The exact three-path Git change list and hashed carry map preserve the unmoved controller, draft/handoff/receipt/revision/authority/I1 owners and historical QA without reopening those features. No persistence writer or derived contract requires retargeting.

## Quality and boundary checks

The three fingerprint callbacks use invocation-time lambdas reading current `self`/`cls` routes. They preserve patches installed after callback creation. The two parametrized store probes and class-wrapper probe exercise that timing and assert the actual source or projected argument identity; they do not merely patch before staging.

Each of the eight helpers has no store receiver, attribute/subscript writer, await/yield or asynchronous function. Message/lineage inputs and alias/identity mappings are borrowed. Construction lists remain invocation-local and become immutable tuples/frozen projections. The added borrowed-input control supplies mapping proxies, checks exact source serialization before/after, retains attachment/generation/parameter/byte identity, checks alias equality and frozen-result assignment refusal. All original 4,114 test-source lines remain an exact byte prefix of the final 4,279-line file; existing assertions, exclusions and warning policies are retained.

The private pure-global image validator and selected-image fingerprint now resolve in the existing fork owner. This is the selected namespace change, documented by the preflight's bounded consumer/patch census. Store method/class routes remain live; arbitrary external dynamically constructed private-global patches are not established compatible.

Static imports introduce no new eager project-module identity. `math` moves to the existing fork owner; `MAX_ATTACHMENT_BYTES` joins its existing attachment import. Video dependencies remain function-local, and the guarded video type import follows `TraceForkBoundary`. The final commit changes only the order of those two TYPE_CHECKING imports. I independently removed that guard and compared complete runtime ASTs between the tested source tree and final commit: identical.

Actual source counts independently match the frozen map: store 22,655 → **22,338**, unchanged cap **22,344**, six lines of slack within the existing 50-line tolerance. Owner 1,135 → **1,584** with no preexisting cap row. Joint production source grows **132** lines. No budget increase, exception or assumed overall size reduction is present. The selected proposal and first NON-FIT artifacts remain hash-identical.

## Receipt and carry evidence

The safe manifest's 34 named evidence files, seven retained preflight artifacts, frozen report and freeze map all independently match their recorded hashes. It contains named receipts/logs/XML and source/AST/blob maps, with no profile, config body, database or cache artifact. The carry map names 20,394 production/test/historical-QA paths, including 12,337 historical QA paths; only the three approved paths differ.

The behavioral receipts pin stable before/after HEAD `fe4eaf365c...` and stable file hashes during the precommit runs. Those hashes independently match the subsequently committed tested source tree; the final owner differs solely by the proven guarded import reorder. XML confirms **280 passed**, zero failures/errors/skips: 241 fork cases, 32 public-boundary cases and seven lineage cases. The two store-only ratchets likewise have **2 passed**, zero failures/errors/skips. Both retained logs contain 24 unsuppressed post-summary PytestWarning entries about preexisting global pytest garbage-directory removal failures. Those warnings and their logs remain part of the evidence.

Final formatter, fatal Ruff and whitespace receipts pin the final source hashes, retain stable before/after source and have exit 0. Final unrestricted Ruff retains **exit 1** and **131 findings**. I independently compared baseline/final code-and-message multisets both jointly and within each file: identical, with no introduced or removed finding. The prior 122-finding source-only failure/reconstruction and the intermediate extra guarded-import I001 are retained; the report does not turn unrestricted lint into a clean pass.

The hashed derived carry map records unchanged diagnostic projections (store 93, fork zero; zero sink/path candidates) and no derived update. Independent source comparison confirms the semantic writer and unmapped owner boundaries remain in place. These carries preserve existing evidence; they are not a newly executed unrelated-subsystem qualification.

## Cannot verify from this diff

- Arbitrary external dynamically constructed patches of former store-private pure globals. The selected relocation explicitly carries this compatibility limit; no such external consumer is supplied.
- Fresh startup timing/runtime census or public navigation. The 1,033 ready and 557 preimport ceilings are retained; static import equivalence is not a new runtime measurement.
- Current-dev/controller integration, the controller's separate cap repair, external checks or publication readiness. Root owns those gates; this verdict approves only Task 10's frozen scoped change.

These are receipt/carry and scope limits, not additional Task 10 defects or requests to rerun the passing suites.

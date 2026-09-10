# Post-review TTS qualification

This package qualifies application code at `401fb7cb838eacad092eeea531b712d2a6400f5d`, rebased on `dev` at `41c4a5858f4323a7e52d4c7c07d8c84099493cf9`. The final commit adds evidence, backlog/lesson updates and the expanded AllTalk migration-to-resolution regression; application Python and validation script bytes remain those tested here. Earlier packages remain frozen pre-review evidence.

| Check | Result |
| --- | --- |
| [Post-rebase targeted tests](checks/qodo-post-rebase-targeted.log) | 405 passed, one explicit real-MPS skip, one inherited audioop warning. |
| [Final AllTalk store/resolver/provider tests](checks/qodo-alltalk-resolution-final.log) | 81 passed, one inherited warning; overlaps the earlier selection. |
| [Ruff disposition](checks/qodo-ruff-disposition.json) | No new diagnostics in maintained changed Python; 154 inherited diagnostics. Raw frozen QA source snapshots are outside that claim. |
| [Formatter](checks/qodo-format-final.log) | Four maintained runner/verifier and new regression modules pass. |
| [Diagnostic inventory](checks/qodo-post-rebase-diagnostic-check.log) | Reviewed inventory matches latest rebased source: 594 owners, 7,663 TASK-494 entries. |
| [Complete wheel identity](wheel/qodo-clean-complete-wheel-identity.json) | All 2,275 Python files match source, wheel and installation: 2,266 application and nine profile-core files. |
| [Serial playback matrix](wheel/qodo-clean-installed-wheel-matrix.json) | Five tuples pass real runtime/playback and joined cleanup. English CPU WAV, MPS WAV and ONNX MP3 yield 9/9 exact full transcripts and three real inference-overlap Stop/recovery controls. Japanese/Mandarin ONNX WAV each play two full clips, with content review required. |
| [Process release](wheel/qodo-clean-ownership-release.json) | All 23 recorded controller, parent, worker, explicit player and ASR PIDs are absent. |
| [User configuration](wheel/qodo-clean-user-config-after.json) | Hash unchanged; all five real ASR runs pass a profile-access audit with an intentional denied-open positive control. |

The [13 review dispositions](review-dispositions.json) record the repaired admission, dependency, documentation, frontend concurrency, AllTalk default and Settings catalog issues. Exact test selectors are in [commands](checks/commands.json). Red/green logs preserve failed attempts, including test-fixture corrections and the temporary local-model annotation error, without treating a later pass as proof that an earlier run passed.

The remaining AllTalk review concerned `female_01.wav`: it is intentionally still the *historical migration sentinel*. An untouched legacy default does not become an exact Studio override. New shipped defaults and inherited provider defaults use `alloy`; explicitly configured/requested IDs and a changed `narrator.wav` selection remain exact. Tests connect the store to effective selection and separately verify the actual submitted provider payload. This preserves user intent without inventing unsupported filename-to-speaker mappings.

The first post-review wheel reused an ignored setuptools build directory containing a Library module deleted by the rebase. File-set verification failed, but a controller was incorrectly launched anyway. [Exclusion evidence](excluded/qodo-stale-wheel-exclusion.json) and its original matrix/log are retained; per-run originals remain at the explicit task-local paths in that matrix. Those results are excluded from final source-matched qualification. After archiving the owned build directory, the clean wheel passed exact application identity before the [controller](wheel/controller.py.txt) admitted work. The nine bundled profile-core files were additionally compared without changing source, wheel or installation. The clean wheel digest is `8b72926288fd77da2300e5abe8cf5770be0f0e0969b973423d0827077349b927`.

Japanese content uncertainty and Mandarin Traditional/Simplified transcript differences remain unchanged; no universal pronunciation claim is made. This matrix exercises mounted Speech Lab and trusted Console delivery, not acoustic loopback or full-shell navigation. Native audio.cpp/PortAudio repair adoption, Higgs cancellation latency, native-language review and unavailable platform/cloud cases remain in the [workstream follow-up matrix](../README.md#follow-up-boundaries). Windows now explicitly tracks an equivalent guarded reader because the hardened verifier uses POSIX descriptor-relative no-follow opens.

[Package provenance](package-provenance.json) identifies original hashes and paths. Five run JSON records factor only duplicate source maps into one manifest; reconstruction is exact. Logs and remaining receipts are copied byte-for-byte. Models, audio, binaries and private profiles remain task-local, so the verifier cannot be rerun from this Git-only package without those original assets. Content receipts retain the original evidence/audio hashes and complete raw transcripts.

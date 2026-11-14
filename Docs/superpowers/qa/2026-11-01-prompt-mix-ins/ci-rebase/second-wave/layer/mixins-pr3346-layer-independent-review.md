# Independent layer-extraction review

What this change does: It gives DB reads a pure shared value/decoder dependency and leaves request composition in its existing owner. It moves application service draining out of kernel Utils while keeping shutdown behavior and error propagation intact.

No actionable findings. This is a read-only review of final_review's fourteen-path extraction against HEAD022, not a review of concurrent runtime/UI/performance edits.

Checked the connected callers and current source, including coherent Prompts DB snapshot transactions, batch limits, UUID/order handling, artifact-type refusal before literal decoding, existing v2 codec routing, shared class identity/re-exports, and app_lifecycle's real unmount delegation. Models retain JSON serialization and validation; there is no new operational authority or receipt format. All known direct callers of the moved private unmount owner now resolve through app_lifecycle. Watchdog state and termination handlers stay in Utils. The moved drain still resolves its app_lifecycle dependencies lazily at invocation, preserving cancellation and outer-finally TTS cleanup.

Independently recomputed AST identity against HEAD: all 20 operational declarations, all 12 artifact-codec declarations, and all 16 shutdown declarations are identical across old/new owners. Read the added import-isolation and malformed/mislabeled snapshot tests, and inspected the author's actual logs: 119 passed in 24.33s, followed by 3 export/compatibility controls in 0.49s. These tests were not rerun. The only reviewed size-pin edits shrink Utils 776 to 554 and add the exact extracted owner at 225; no growth waiver or layer-allowlist expansion.

Assumed load: the existing single-user Console with concurrent source callbacks and shutdown tasks; this extraction adds no work inside those paths. Performance import inventory is coordinated separately by the parent/performance owner.

Verdict: Ship this extraction. Not checked: hosted CI or newer-dev rebase; no full suite was run.

Reviewed file bytes (SHA256; the architecture file assessment covers only the two shutdown rows):

- `tldw_chatbook/Chat/console_prompt_mixin_models.py`: `56506626487984b0a4ecb918c38b270a65decfbeb3f72498e78aa00c054407b8`
- `tldw_chatbook/Chat/console_prompt_mixins.py`: `b7bb3bebe2daaa17034f8b229ac49eb6160ffdb0c4338a4a1c8d17b091856029`
- `tldw_chatbook/Prompt_Management/prompt_artifact_models.py`: `9c1a1798705b1fa25fd25a2f5fcd426040f685a181692239cbbe2cfef26c9e85`
- `tldw_chatbook/Prompt_Management/prompt_artifact_codec.py`: `52608d73efcf56b8800f1465201ded11c0c5741384368d62ae1fb41b4c32a11e`
- `tldw_chatbook/Prompt_Management/prompt_source_capabilities.py`: `8fc7eab1730a95916d1ea0d590ea833a651e6d7048b4f9590a91aa3e82af2794`
- `tldw_chatbook/DB/console_prompt_mixins.py`: `e29b370249449363fddefd0315215153f123b3870c3db739d121e4b7639881d8`
- `tldw_chatbook/DB/Prompts_DB.py`: `6c59032108f5a5513edb9f8aff50040983fb3798eb1d00b5418a61644abc7136`
- `tldw_chatbook/Utils/app_shutdown.py`: `a54534dc6103baf1c3433b81766de6e4a0d018b814557e77283236b30424e9ff`
- `tldw_chatbook/app_shutdown.py`: `b43f61888c2b3681eb36b90235fb022d9939086d0c9c3d4d2e95cb952ca0f3f9`
- `tldw_chatbook/app_lifecycle.py`: `937526133306fc924db7502a5595c697cb9a903391d9af06e4cee9f7a40c88e8`
- `Tests/Chat/test_console_prompt_mixin_values.py`: `8df3e8fa470c75d32734d3dd4dd1c738f698195c685b3a63194805a2ceb46290`
- `Tests/Prompt_Management/test_prompt_mixins.py`: `c38bb3e82c3cd29cbbee6d3bb95fdae36e330493eb0e205303fd62100699f0da`
- `Tests/TTS/test_tts_app_ownership.py`: `0e6f813a8b741c9cd74b6695e6c6dc638d133d74e12c42fcc55b9eebd210fa4f`
- `Tests/Architecture/test_module_size_ratchet.py`: `fd943580501d879a024b1619f0a6a5f1758d80ef24176e1f66d6284110f7379d`

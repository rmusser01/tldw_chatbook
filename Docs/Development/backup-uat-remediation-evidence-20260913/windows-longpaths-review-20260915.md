# Windows long-path checkout review

APPROVED: no actionable findings in the sole workflow delta against HEAD `2031d0c5261c2b4781aba40752f228da9d19b4eb`.

File: `.github/workflows/voice-aec-wheels.yml`
SHA256: `986e970a24aa2fbd7e38b971cec5532be4cb255cc6a1c012fa6dc682e39d7278`

The exact four-field new step is first in build-wheels, before its original pinned checkout: Windows-only condition, pwsh, and `git config --global core.longpaths true`. Independently parsed old/current YAML; removed only this exact step and the complete parsed documents are equal. The original checkout action, synthetic PR merge selection, persist-credentials=false, all matrix platforms, permissions, inputs, jobs, build/test commands and deadlines remain unchanged. git diff --check passes.

This enables Git for Windows builtin long-path support on the ephemeral hosted Windows runner. It does not alter filenames/content, omit the inherited task, check out feature HEAD instead of the merge, change ACL/native guards, create ancestry or merge the PR. No earlier prohibition on this ordinary Git setting was identified; previous excluded workarounds changed tested source or ancestry. The setting addresses the observed checkout-stage MAX_PATH refusal, not an executed Voice build failure.

Official implementation guidance: [Git for Windows long paths](https://gitforwindows.org/git-cannot-create-a-file-or-directory-with-a-long-path.html). Downstream tools can have independent path limits; fresh Windows genuine-merge checkout and unchanged vendor/wheel tests remain required before claiming successful qualification. No new apps/tests/network operations, repository edits, branches, commits or pushes were performed in this final review. Parent retains the stated wait for ongoing CI terminal evidence before commit/push.

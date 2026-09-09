---
id: TASK-32164
title: Qualify native shutdown under a working sanitizer toolchain
status: Done
assignee:
  - '@codex'
created_date: '2026-09-09 07:02'
updated_date: '2026-09-09 07:40'
labels: []
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The native shutdown repair passes real macOS CPU/Metal and transport checks, but the current macOS ASan binary deadlocks during dyld/malloc sanitizer initialization before main, including outside the sandbox. The available Linux ARM validation image has no C/C++ compiler. Preserve this explicit validation gap without treating startup timeouts as transport failures.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A pinned sanitizer toolchain reaches the HTTP regression entrypoint; macOS startup behavior is reproduced and resolved or documented against a working alternative toolchain.
- [x] #2 All nine active-request, streaming, idle/partial-client and exception shutdown cases pass with ASan and UBSan enabled, with compiler/runtime/source provenance and complete logs.
- [x] #3 Sanitizer findings, if any, are repaired with targeted regressions and all validation containers and child processes are joined.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A; existing ADR-023 ownership boundary remains unchanged.
Reason: Temporary sanitizer tooling and qualification only; no production dependency or system toolchain changes.

1. Preserve the bounded macOS ASan startup stack and the first Linux compiler-missing receipts; retain the unchanged native request-drain patch and exact source hashes.
2. Build a separate task-owned Linux ARM image from the existing pinned Python/Debian image with C/C++ build tools and ASan/UBSan runtimes. Bound provisioning and record immutable image, compiler, runtime and recipe provenance; touch no existing container.
3. Run the same nine transport-only cases once serially in a uniquely named network-none container with read-only source and a private evidence mount, after coordinating CPU/audio use. Bound compile/test deadlines and join every owned child/container.
4. Compare outcomes, preserve raw logs and any limitations, package compact QA receipts and verify hashes. Mark criteria and final task status only after actual passing evidence and cleanup checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Qualified the unchanged native HTTP request-drain repair under a working Linux ARM ASan/UBSan toolchain. A separate image pinned as sha256:596934820d0c869610526649be433cefcdce0b9b4abdbd8040fc6b2894b83bcd extends the supplied Debian 12 image with GCC/G++ 12.2.0 and sanitizer runtimes. No macOS toolchain, approved native runtime baseline or native patch changed.

The no-argument instrumented executable reached usage exit 2, then all nine targeted transport cases passed once with ASan/UBSan and no diagnostics. The two-CPU network-none container completed in 18.872 seconds, was removed, and preserved all 507 source-input hashes plus the native patch hash. All owned children were joined; no models, playback or preexisting containers were touched.

Preserved the macOS pre-main ASan initialization stack and initial Linux compiler-missing receipts. Linux is a qualified alternative; these results do not resolve macOS startup. The first image recipe's raw-SHA FROM lookup failed before provisioning; the successful recipe verified the existing local tag against the exact parent image and checked parent layers.

QA: Docs/QA/tts-macos-burndown-2026-09-09/native/sanitizer/linux-qualified-02/README.md. Exact compiler/runtime/image/build/case logs and hashes are included. Verified all 41 copied artifacts, 46 package-index hashes, 45 report links, JSON parsing, and task-local launcher syntax. No native code fix was needed, so no extra native regression was introduced. Added the concrete startup-control incident to backlog/docs/lessons-testing-evidence.md.

ADR required: no; temporary qualification tooling leaves the existing ADR-023 application/runtime boundary unchanged.
<!-- SECTION:NOTES:END -->

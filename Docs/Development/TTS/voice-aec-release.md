# Voice AEC companion release

`tldw-voice-aec` is the native WebRTC AEC3 companion for `tldw_chatbook`. The
application and companion always use the same version. The application
`speech_recording` extra exact-pins that companion version so an application release
cannot silently acquire a different native implementation.

The local 0.1.9.0 companion adds `DUPLEX_ABI_VERSION = 1` and
`NativeDuplexBridge` alongside `AecProcessor`. Device startup checks ABI and
callback ownership before opening sounddevice's native callback seam. Missing
or older components report native duplex unavailable; rebuild/install the
matching component and restart. No Python device-callback fallback is permitted.
The installed-wheel entry point exercises callback PCM/identity and AEC before
its existing corpus gate. Software ABI checks do not qualify a device or release;
the packaged qualification manifest remains unqualified.

The governing architecture is [ADR-098](../../../backlog/decisions/098-low-latency-speculative-duplex-voice-pipeline.md).
The reviewed WebRTC revision, closure, Abseil pin, license inventory, and declared
hash-pinned patch series are recorded under `native/voice_aec/vendor/webrtc/`.
A vendor update is a new security and legal review, not a routine dependency bump.

Windows Release wheels statically link the MSVC runtime in both the WebRTC and
binding targets, preserving the single-extension archive policy. Do not bundle
or silently exclude repair-discovered runtime DLLs. Keep native allocation
ownership inside the extension; debug CRT builds are not redistributable.
Static runtime security updates require rebuilding and requalifying the wheels
with an appropriately licensed, supported toolchain; they do not arrive through
an independently updated private DLL. See ADR-098 for the boundary rationale.

The pybind11 3.1.0 binding headers are independently pinned with
`pybind11==3.1.0`. The checked-in
`native/voice_aec/PYBIND11_LICENSE.txt` is byte-identical to the BSD-3-Clause license
from the authoritative
[`v3.1.0` source tag](https://github.com/pybind/pybind11/tree/v3.1.0). Its SHA-256 is
`83965b843b98f670d3a85bd041ed4b372c8ec50d7b4a5995a83ac697ba675dcb`.
Changing the pin requires a fresh compatibility build and legal review.

## Source-bound qualification order

Physical safety evidence is measured by the live production-path runner; manually
entered physical metrics are invalid. Changing any path in
`Packaging/speculative_voice_source_paths.txt` invalidates all earlier automated,
soak, and physical evidence. Regenerate it in this exact order:

1. Freeze and commit every source-listed file.
2. Compute one source digest from the clean worktree.
3. Regenerate the automated, duplex-soak, and cancellation-soak reports.
4. Run physical-device qualification from that same commit and digest.
5. Pass every physical report through the strict report consumer.
6. Build rollout authority only after the full platform/device matrix exists.

The operator procedure, privacy boundary, audible-sample warning, and exact project-venv
command are in
[speculative-voice-qualification.md](speculative-voice-qualification.md).

## Required release order

1. Bump `project.version` in the root and companion `pyproject.toml` files together.
   Update the exact `tldw-voice-aec==<app-version>` entry in `speech_recording`, then
   run `Packaging/check_voice_aec_version_sync.py`.
2. If upstream source changed, update the immutable commit and tree, re-vendor the
   complete manifest, and review `LICENSE`, `PATENTS`, `THIRD_PARTY_NOTICES.md`, the
   Abseil license, the Ooura FFT notice, the pybind11 license, hashes, and patch
   series. An incomplete or unreviewed notice blocks release. The companion uses PEP
   639 metadata: its
   `License-File` headers and dist-info license directory must contain the exact
   reviewed WebRTC license, patent notice, Ooura notice, pybind11 license, and
   third-party notices.
3. Merge only after the **Voice AEC wheels** workflow qualifies CPython 3.11–3.13 on
   macOS x86_64/arm64, Windows x86_64, and Linux x86_64/aarch64. Each repaired wheel is
   installed outside the checkout and runs the native binding and echo-corpus smoke.
   Pull requests receive only an unsigned
   `voice-aec-structural-only-<commit>` artifact. The PR path has no OIDC or
   attestation permission and cannot sign, produce, or upload the release-eligible
   artifact. Only a trusted push or manual run on `refs/heads/main` enters the separate
   signing job.
4. Retain the successful default-branch matrix workflow-run ID. The run produces one
   immutable `voice-aec-qualified-<commit>` artifact containing the exact wheels,
   sdist, `SHA256SUMS`, source-tree digest, SPDX SBOM, license inventory, notices, one
   SLSA provenance statement and keyless Sigstore bundle per distribution, and the
   GitHub OIDC provenance bundle. Its SPDX 2.3 SBOM embeds
   `LicenseRef-Ooura-FFT` in `hasExtractedLicensingInfos`; the extracted text must be
   byte-for-byte equal to the reviewed `vendor/webrtc/OOURA_LICENSE`. The 30-day
   Actions artifact is release transport, not durable release evidence. The wheel
   workflow never publishes.
5. Protect a tag named exactly `voice-aec-v<app-version>` and point it at the same
   commit as that successful matrix run. Start **Release Voice AEC** from that tag and
   supply the matrix workflow-run ID. Do not use a pull-request run or a run from a
   different commit.
6. The release workflow downloads the named artifact from that exact run. It verifies
   the repository and workflow identity, tag/source commit, source-tree digest,
   `SHA256SUMS`, closed artifact contents, SBOM coverage, license inventory,
   `THIRD_PARTY_NOTICES.md`, `PATENTS`, `PYBIND11_LICENSE.txt`, and every retained
   provenance statement. It
   uses the official GitHub attestation verifier and `sigstore verify identity` with
   the GitHub Actions OIDC issuer. It then recomputes the application voice-source
   digest, validates the checked-in automated/soak/physical reports, selects the exact
   repaired wheel matching each report's CPython ABI, regenerates both packaged rollout
   authority files, and requires byte-for-byte equality with the reviewed files. A
   clean application wheel install must also pass the runtime authority smoke before
   the protected publishing job becomes eligible. Any mismatch fails closed.
7. A required reviewer approves the protected `pypi-release` environment. Before
   Trusted Publishing, the protected job deterministically archives the entire exact
   qualified directory as
   `voice-aec-qualified-<source-sha>.tar.gz` on the already-verified
   `voice-aec-v<app-version>` GitHub Release. When the release is absent, the workflow
   passes the archive to `gh release create` so GitHub stages the asset on the release
   draft and publishes the release and asset atomically. On a rerun, an existing
   release must already
   contain the asset; the workflow never uploads to, deletes from, overwrites, or edits
   that release. It accepts the existing asset only when it is byte-identical. After
   either path, the workflow requires the release to have the exact protected tag, be
   non-draft, and report immutable before it re-downloads the asset and verifies its
   SHA-256. Any missing asset or mutable release state fails before Trusted Publishing.
   This durable release asset must be retained for the application release's support
   lifetime. PyPI Trusted Publishing then uploads the already-qualified wheel and sdist
   bytes with attestations enabled. The release workflow must never rebuild, repair, or
   rename a distribution: ZIP timestamps and platform repair output need not be
   deterministic, so only the qualified hashes identify releasable bytes.
8. After publication, the workflow installs `tldw-voice-aec==<app-version>` from PyPI
   on every supported Python/platform cell using `--only-binary`, `--require-hashes`,
   and the retained wheel hashes, then runs an installed native smoke test.
9. Only after every published-wheel installation succeeds may the application release
   whose `speech_recording` extra pins that exact version be published.

## Signing and retention policy

Every wheel and the sdist has a canonical in-toto/SLSA provenance statement binding
its SHA-256 digest to the source commit, source-tree digest, repository, and exact wheel
workflow identity. The statement uses the official GitHub Actions workflow build type
`https://actions.github.io/buildtypes/workflow/v1`, including the exact workflow
repository/ref/path, stable repository and owner IDs, event, hosted-runner identity,
resolved Git commit, and run/attempt invocation URL. The source-tree hash is retained
as the `tldw_source_tree_digest` extension. The statement is keyless-signed with
Sigstore using GitHub OIDC. The
official GitHub build-provenance attestation covers the distribution bytes as a second
independent verifier. Keep the SHA-specific GitHub Release evidence archive,
`SHA256SUMS`, SPDX SBOM, legal notices, per-file statements and Sigstore bundles,
GitHub attestation bundle, workflow run ID, and environment approval record for the
release's support lifetime. The expiring Actions artifact remains only a transport
copy.

The unprotected verification job receives only read permissions. The protected
`pypi-release` job receives `contents: write` solely to create or verify the immutable
evidence asset, `actions: read` for exact-run transport, and the OIDC permission needed
by Trusted Publishing. It has no long-lived signing key. GitHub Actions and release
actions are pinned by full commit SHA. The `pypi-release`
environment and `voice-aec-v*` tags must be protected by repository rules; changing
those rules requires a release-security review.

## Fail-closed and rollback policy

Missing platform wheels, failed native smoke tests, unexpected shared libraries,
version drift, missing SBOM subjects, altered hashes, extra artifact files, missing
`PATENTS` or other notices, an unverified Sigstore/GitHub OIDC identity, or a failed
Trusted Publishing step blocks the companion and application releases. Do not rebuild
locally to replace one failing file and do not use `skip-existing` to hide a partial
publish.

PyPI distributions are immutable. If only part of a version is published, treat that
version as unusable: do not publish the app version, diagnose the incident, bump both
projects to a new version, and qualify a completely new artifact. Never overwrite a
published file. Until every gate passes, the application capability check keeps the
legacy safe pipeline selected; missing AEC must degrade to half duplex rather than open
an unsuppressed microphone during assistant speech.

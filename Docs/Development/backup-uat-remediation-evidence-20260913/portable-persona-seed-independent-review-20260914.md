# Portable Persona seed — independent review

APPROVED for the frozen test-only seed correction, subject to separately recording the author's still-running complete 20-case result. No actionable source finding. No product edits or reviewer repository edits.

Verified test_created_persona_subtree_rollback.py SHA-256 fe009977d205c74d1428d3611c73e1e60ea7d023c9462b99db47e986bf0a5033. AST comparison against HEAD proves all module content outside _SEED_PERSONA is unchanged, including the real lifecycle, 19 negative cases, and all budgets/assertions.

The seed creates existing artwork in the native publisher's format: independently generated UUID pack/version directory tokens, canonical snapshot manifest and context from the existing validator, validated PNG length/hash and publisher asset-row projection, immutable manifest/assets paths, and a cleanup marker calculated from the actual native version-directory identity. The real PersonaVisualRepository activates the graph and reads it back, with manifest hash equality. This bypasses only the POSIX-specific filesystem publisher for fixture setup, not backup discovery, storage admission, replacement or rollback. It is not evidence of Windows Persona publication support.

Private output directories and files use existing create_private_directory/create_private_file and platform_files.os rather than POSIX descriptor assumptions. Each payload is flushed by the native file helper and read back exactly. The snapshot input remains disposable fixture data outside the profile. No fixture concurrent writer or authority bypass is introduced into the workflow under test.

Independent native seed probe passed 1 test in 5.31s: /private/tmp/uat-persona-portable-seed-independent.log. Probe /private/tmp/test_persona_portable_seed_review.py verifies owner-private native modes and ownership throughout the resulting Persona tree, regular-file/directory types with no symlinks, retained version-directory identity, valid marker shape, exact asset path, and active graph readback. This is macOS execution, not native Windows acceptance. Author's full 20-case run remains /private/tmp/uat-portable-persona-seed-green.log and was still active at review time.

## Runner additions

Approved root's two restart import-origin selection additions to run_platform_product.py: the exact already-executed native regression node is added to the full/support product list and support-diagnostic. Previously reviewed Console file addition remains. Diagnostic selection is now previous nine nodes plus 21 Console cases plus one import-origin case = 31. No prior selection removed, no key/mode or deadline change. The native regression itself passed in the earlier full 15-case restart run.

# Semantic foundation evidence

This bundle retains the original Task1 allocation/native receipts and Task2/3 provider, generation, query and retirement receipts before private-workspace cleanup. The ZIP preserves the original member bytes; the manifest records each SHA-256 and byte length. No model weights, temporary profile, source corpus or SQLite cache is included.

The standalone implementer and review reports are historical snapshots with their original source hashes, adverse runs and limitations. Task1 source commits were 42a75186 and 439f905f; Task2 source commits were 2160d5b3 and ef0af31c; Task3 source commits were 805701f9, 5ad64a28, 286be157 and 538faa10. Their final scoped review reports are retained alongside them. Later Task4 changes require their own evidence; these receipts do not certify later code merely because a test name is unchanged.

`quota-foundation-proposal.md` and `sqlite-runtime-probe.json` document the source-conditioned SQLite allocation certificate and its runtime/VFS/cooperating-writer limits. A sampled peak is not a universal quota proof.

To inspect one original receipt without extracting or changing the checkout, use Python's standard-library `zipfile.ZipFile(...).read(member_name)`. Member names are listed in `manifest.json`; child XML, parent logs and pre-run hashes remain distinct evidence. Existing warning/adverse results have not been rewritten or suppressed.

The directory's Git attributes preserve original bytes across checkout platforms. Verification must compare committed Git blobs and archive members with the manifest; a matching working copy alone is insufficient.

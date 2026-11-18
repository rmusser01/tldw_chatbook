# Evidence use

The main audit report is one directory above. Probe sources are suffixed `.py.txt`
so ordinary test discovery does not execute historical defect witnesses. This
replay recipe targets the recorded Windows host/account and project Python.
Check the pinned product source first; later repaired code need not reproduce
these historical failures. Keep existing privacy/network guards and 180-second
test limits. Never run against the user's default profile.

1. From the repository root, create a new, uniquely named directory such as
   `.test-tmp/book3283-replay-<unique>/character-bench`. Refuse an existing replay
   directory. Also create the launcher's required parent
   `.test-tmp/textual-37031-37034` if absent; do not reuse an existing run label.
2. Copy the selected domain's probe and conftest there, restoring `.py` suffixes.
   Copy `run-targeted-external-profile.py.txt` into the new replay directory as
   `run-targeted-external-profile.py`. Leave all archived witnesses unchanged.
3. **Character only:** in the copied probe, replace its one hardcoded
   `OUT = ROOT / '.test-tmp/textual-after3283/character-bench'` assignment with
   `OUT = Path(__file__).resolve().parent`. This confines observations/identity
   files to the new replay directory. Do not run the unmodified copy: it targets
   the original evidence directory. Research already writes beside its copied
   probe; Notes writes under its fresh pytest temporary directory.
4. Run the copied launcher with the project Python, a fresh unique run label,
   and the copied probe path as its selector. It creates the run receipt folder
   and native-private profile/TEMP beneath the recorded per-user
   `C:/Users/GDesktop-1/AppData/Local`. On another host, qualify an equivalent
   safe per-user parent before adapting only that copied launcher path; do not
   relax the guards or redirect to the default profile. The archived launcher
   is a receipt of the original environment, not a universal installer.

Raw XML and SVG bytes are stored losslessly as deterministic `.gz` files. This
preserves pytest's meaningful whitespace without introducing trailing-whitespace
diff failures. `archived-receipts.json` records compressed and decoded SHA256;
decode with Python `gzip.decompress` before XML parsing. The main manifest hashes
the actual repository representation with CRLF normalized to LF for text, and
keeps binary archives unchanged. Original report hashes refer to decoded bytes.

Domain investigation reports retain original names/private provenance. Use the
main evidence/archive mapping for portable locations; sources gain `.txt` and raw
XML/frames gain `.gz`. Original private manifests are retained in the worktree;
the root manifest is authoritative for this portable packet.

Raw pytest `.log` files remain private scratch and are excluded from the
repository packet under its existing ignore rule. The portable packet includes
the actual XML, exact observations, probe sources and receipt hashes needed to
assess the findings; the original private directories retain the raw logs.

Prototype Notes01/02 and Research attempt1 reports are exclusions or earlier
observations, not passing repair evidence. Notes02's copied source preserves an
unproven startup-cause comment; the main audit explicitly rejects that causal
claim. Only final source and actual reached-writer/owner observations qualify
the new tasks. No product fix is implemented by these probes.

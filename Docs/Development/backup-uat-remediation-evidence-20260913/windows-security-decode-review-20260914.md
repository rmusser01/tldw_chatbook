# Independent Windows security decode implementation review

Verdict: **Approved for the bounded native Windows experiment; no actionable finding.** This is not Windows acceptance or evidence of startup improvement. Reviewed only the three files listed in the attached exact-byte receipt against `9dff9dd1b2`.

## Native security and allocation boundary

Every `security(handle, is_directory)` still performs fresh `GetSecurityInfo` before any lookup. A failed acquisition raises immediately. Every successful acquisition reaches the existing `LocalFree` finally on hit, miss, decoder failure and uncached fallback. File identity, volume, reparse, namespace and native lease checks are untouched. The `_stat_handle` observation order and `fchmod` fresh owner check are unchanged.

Only complete freshly acquired self-relative descriptor bytes of length20–4096 enter the128-entry stdlib LRU. The key additionally includes the directory flag, explicitly captured current-user SID and native instance. The successful immutable `(uid, mode)` tuple is the only result retained; no pointers or handles escape. At most512KiB of descriptor payload is retained, plus bounded key/result overhead. Concurrent duplicate misses are permissible; exceptions do not populate the cache. This does not promise identical allocation-failure opportunities to repeatedly converting SIDs.

The added ABI declarations use DWORD/c_uint32 for descriptor length and BOOL/c_int32 plus pointer outputs for owner; the existing DACL declaration is correctly reused. Header control/length are read while the native allocation is live. Owner and DACL addresses from the copied self-relative buffer remain valid through synchronous GetAce/SID decoding because the local buffer is retained through return. Native owner/DACL BOOL results are checked. The passed SID, rather than a second mutable self.user_sid read, drives both ownership and ACL projection.

An independent AST comparison proves `_decode_security` is the former decoder body unchanged except `self.user_sid` becoming its explicit argument. Null/unknown/inherit-only ACL handling therefore retains the existing conservative semantics. Non-self-relative and oversized observations retain the original decoder path. New native extraction failures propagate without fallback to an older result.

## Tests and executable Windows instrumentation

Independent local result: **40 passed,19 Windows-only skipped in0.87s** across the new module and existing Windows-files module. Evidence: `/private/tmp/uat-security-decode-independent.log` and `.xml`. These portable tests cover fresh acquisition/free on hits, real key-byte changes at the ABI seam, user/directory separation, acquisition and decode errors, uncached boundaries, retention limit and conservative null/unknown ACE results.

Read the four new Windows-only tests in full. They use actual native file handles, public Everyone ACL grant/hardening, invalid-handle acquisition, path replacement identities and fresh security/count observations. The counter proxy delegates unchanged ctypes functions; dereferencing the SECURITY_DESCRIPTOR output with POINTER(c_void_p)[0] yields a hashable integer, verified independently. SID-string frees are distinguished from still-live descriptor allocations. Both proxies restore before the paired microcost sampling and native descriptor closes in finally. No obvious Windows-only executable error was found. Native invalid-handle/ACL mutation/free-count/identity assertions still require CI execution.

The timing case measures repeated warm descriptor decoding versus the original decoder with fresh GetSecurityInfo on both paths; it has no timing threshold. Its output supports only this microcost comparison, not cache-hit rate, total security CPU attribution or app startup wall time. Existing native tests remain selected.

The runner AST is identical apart from appending this one file to `_NATIVE_TESTS`; no product selections, assertions, deadlines or modes change. No repository edits or app runs were performed by this reviewer. Root owns static comparison and Windows qualification.

Exact SHA256 values are recorded in `/private/tmp/uat-security-decode-independent-hashes.json`.

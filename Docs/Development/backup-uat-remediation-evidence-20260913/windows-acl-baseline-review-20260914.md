# Independent ACL native baseline implementation review

APPROVED. Read-only scope: windows_acl_fixture.py and test_windows_acl_fixture.py uncommitted delta against 18f7321c0a. Exact SHA receipt: /private/tmp/uat-acl-native-baseline-independent-hashes.json. No actionable findings.

The setup setter reuses the original DACL pointer and original protection flags. Its successful return is immediately followed by a verification inside the final restoration lifetime. Baseline admission allows only unchanged owner/revision/ACL bytes and unchanged control or its 0x0400 addition; bit clearing, protection changes and all other fields fail before yielding. The temporary read descriptor is freed without invalidating the immutable copied baseline tuple. The original descriptor remains alive across the body and both setters, then is freed once by its owning context.

A setup setter error fails before body and frees the original descriptor. Setup read/validation failure still attempts final original-DACL restoration. No unestablished baseline can result in success. A final setter/read error retains the active setup/body exception as its context. On an established baseline, the final entire tuple comparison is exact: no masking of AUTO_INHERITED, no owner/SACL write, no changed product permission policy, no retry loop.

Independent complete helper suite: 22 passed (see /private/tmp/uat-acl-seed-independent.log). It covers current/unchanged initialization, every refused baseline field class, pre-existing bit loss, final-only bit gain/loss, setup read/secondary cleanup failures, original body exception identity and safe bounded error metadata. Existing native Windows evidence establishes why pre-body conversion is needed; successful Windows execution of this corrected fixture is still pending. No native Windows pass is claimed here.

Final test-only SIM117 adjustment reviewed: five nested contexts merged in identical pytest.raises-then-helper entry order and reverse exit order. Approval retained; test SHA bc0e5d029fd383e56d90d7d50ad2fb8fc13c3f91c15fa32b2cc4126a72955ad1. Helper unchanged. Author final run 22 passed in 0.70s, /private/tmp/uat-acl-seed-final.log.

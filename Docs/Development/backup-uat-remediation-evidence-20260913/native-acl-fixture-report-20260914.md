# Windows native ACL fixture correction

Scope: TASK-32562, test-only. No product source, deadlines, owner authority or runtime guards changed. Three owned files are frozen; hashes are in /private/tmp/uat-native-acl-final-hashes.json.

Native RED: all three Windows 9e5914 unsafe_parent cases fail at generated child line 67, the saved/re-exported icacls text-byte comparison. The preceding actual Everyone:R grant, native public-posture check, raw._check refusal, post-refusal public-posture assertion and icacls restore returned successfully. This is evidence of a fixture verification failure, not a product guard failure. The actual two export contents were not retained, so no claim is made about their specific text difference.

The candidate retains the original native descriptor through GetSecurityInfo on a pinned exact parent handle. It restores only the captured DACL through existing SetSecurityInfo, using the original protected/unprotected flag. Owner and SACL are not written. A subsequent native read must exactly match original owner SID, full descriptor control bits and revision, and all DACL bytes. Nothing normalizes inheritance, protection flags or ACE order. If the API does not restore that original state, the test fails. Descriptor buffers and handles are released on success or failure. Cleanup failures retain the original body error as exception context.

The fixture still makes the parent genuinely public with icacls, verifies native privacy before/public after/refusal/public after refusal/restored private, preserves parent device/inode/mode and all original config/control file device/inode/bytes, and forbids new selected files or temp files. POSIX chmod behavior and the 35-second native-child ceiling are unchanged. icacls text export/import is no longer used as a restoration oracle.

Verification:
- Supplied exact native Windows RED: /private/tmp/uat-windows-9e5914-support/junit-failures/{c1a17cad8f834db2,764e5f6ec54f3c04,861a23179eb92baf}.txt.
- New helper tests first fail due missing helper (6 cases, /private/tmp/uat-native-acl-helper-red.log); this is not a second native reproduction.
- Final local suite: 27 passed in 22.90 seconds (/private/tmp/uat-native-acl-final-green.log): all 19 sibling cases, including 15 guard cases, plus 8 helper cleanup/state-verification tests.
- Ruff clean (/private/tmp/uat-native-acl-ruff.json), git diff --check clean.
- Bandit: native helper and existing sibling test have no findings; new unit test has only its 11 expected pytest assertions (B101), zero other findings (/private/tmp/uat-native-acl-bandit.json). Assertions are the test contract, not product security enforcement; the helper's native equality refusal is an explicit conditional raise.

Native Windows acceptance remains pending the parent's runner. POSIX/local mock lifecycle tests do not establish that the Windows APIs preserved the descriptor. In particular, SetSecurityInfo's native control/inheritance behavior is checked rather than assumed. No Linux remote-log reads or other UAT fixture accesses occurred for this correction.

Official API references reviewed: [GetSecurityInfo](https://learn.microsoft.com/en-us/windows/win32/api/aclapi/nf-aclapi-getsecurityinfo), [SetSecurityInfo](https://learn.microsoft.com/en-us/windows/win32/api/aclapi/nf-aclapi-setsecurityinfo), [GetSecurityDescriptorControl](https://learn.microsoft.com/en-us/windows/win32/api/securitybaseapi/nf-securitybaseapi-getsecuritydescriptorcontrol). SetSecurityInfo is the documented filesystem API; SetKernelObjectSecurity was deliberately not used.

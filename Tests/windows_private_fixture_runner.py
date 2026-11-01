"""Run ordinary Windows fixture tests under a genuine user default owner.

The separate required elevated-custody job must use its original token. This
launcher changes only the test process's native default owner for newly created
objects, never production guards, existing ACLs, or the token user/privileges.
See https://learn.microsoft.com/windows/win32/secauthz/owner-of-a-new-object.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager


@contextmanager
def user_fixture_default_owner():
    """Set and restore the actual process default owner for ordinary fixtures."""
    if os.name != "nt":
        yield
        return
    if os.environ.get("TLDW_REQUIRE_ELEVATED_CUSTODY") == "1":
        raise RuntimeError("required elevated custody must retain its original token")

    import win32api
    import win32con
    import win32security

    token = win32security.OpenProcessToken(
        win32api.GetCurrentProcess(),
        win32con.TOKEN_QUERY | win32con.TOKEN_ADJUST_DEFAULT,
    )
    try:
        original = win32security.GetTokenInformation(token, win32security.TokenOwner)
        user = win32security.GetTokenInformation(token, win32security.TokenUser)[0]
        win32security.SetTokenInformation(token, win32security.TokenOwner, user)
        try:
            actual = win32security.GetTokenInformation(token, win32security.TokenOwner)
            if actual != user:
                raise RuntimeError(
                    "native fixture default owner did not become TokenUser"
                )
            yield
        finally:
            win32security.SetTokenInformation(token, win32security.TokenOwner, original)
            if (
                win32security.GetTokenInformation(token, win32security.TokenOwner)
                != original
            ):
                raise RuntimeError("native fixture default owner was not restored")
    finally:
        token.Close()


def main() -> int:
    """Run pytest after selecting real native fixture ownership."""
    with user_fixture_default_owner():
        import pytest

        return int(pytest.main(sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())

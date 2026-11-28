"""Rebind the reviewed disposable live launcher to the exact merged source."""
import hashlib
import importlib.util
import os
from pathlib import Path

original = Path(__file__).with_name("pr3023-current-live-launch-r2.py")
assert hashlib.sha256(original.read_bytes()).hexdigest() == "b30cc93e757927ec33151ec0cefcc81e36a8ddef5ef287fec8f0dd87d2327b94"
spec = importlib.util.spec_from_file_location("merged_live_launcher", original)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.ROOTS = {
    "baseline": (Path(os.environ["FIRSTSEND_SOURCE"]), "4155fb8298c63107cb0df99b79b963dbbd05e2a3"),
    "candidate": (Path(os.environ["FIRSTSEND_SOURCE"]), "13f7aaec1e14ed07921acd7c7c88d046e1069f07"),
}
module.__file__ = __file__
original_app = module.app

def checked_app(profile, role, *, check_only=False):
    result = original_app(profile, role, check_only=True)
    guard = module.load("complete_config_guard", "postsave-current31-config-guard-r4.py", "ce6ea09787f733a59f06c72267b6d86ea4453f7be0d9c8094c6ebc9b9e391f2d")
    guard.require_checked(profile)
    if check_only:
        return result
    return original_app(profile, role)

module.app = checked_app
if __name__ == "__main__":
    module.main()

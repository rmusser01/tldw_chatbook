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
    "baseline": (Path(os.environ["FIRSTSEND_SOURCE"]), "cf788f548edd2ba3eb865776bf8568b27219a447"),
    "candidate": (Path(os.environ["FIRSTSEND_SOURCE"]), "cf788f548edd2ba3eb865776bf8568b27219a447"),
}
module.__file__ = __file__
original_app = module.app

def checked_app(profile, role, *, check_only=False):
    result = original_app(profile, role, check_only=True)
    guard = module.load("complete_config_guard", "postsave-current31-config-guard-r4.py", "c8d5d50f1b881920ee8724b69c05877d24fe25283528ba540b34868a7d52b622")
    guard.require_checked(profile)
    if check_only:
        return result
    observer_class = module.load("postsave_observer", "postsave-observer.py", "50cb67cd8f9ea728609955eb00800edaef1f43311acfb26eb2ad61ccdf452b26").Observer
    observer = observer_class(module.ROOTS[role][0], profile / "postsave-spans.json")
    observer.start()
    try:
        return original_app(profile, role)
    finally:
        observer.close()

module.app = checked_app
if __name__ == "__main__":
    module.main()

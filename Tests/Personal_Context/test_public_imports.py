"""Personal Context recovery stays cold; explicit public API keeps its identities."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from Tests import real_profile_guard


REPO_ROOT = Path(__file__).resolve().parents[2]

# Frozen from the pre-fix public API, independently of the lazy dispatch table.
EXPECTED_EXPORTS = (
    ("EncryptedEnvelope", "crypto"),
    ("EnvelopeCipher", "crypto"),
    ("InMemoryProfileKeyProtector", "key_protector"),
    ("InterviewPassphraseKeyProtector", "key_protector"),
    ("InterviewCommitOutcomeUnknownError", "interview_coordinator"),
    ("InterviewCommitReceipt", "interview_coordinator"),
    ("InterviewDiff", "interview_diff"),
    ("InterviewDiffChange", "interview_diff"),
    ("InterviewDraftConflictError", "interview_draft_repository"),
    ("InterviewDraftExpiredError", "interview_draft_repository"),
    ("InterviewDraftRepository", "interview_draft_repository"),
    ("InterviewProviderError", "interview_provider"),
    ("InterviewProviderRequest", "interview_provider"),
    ("InterviewQuestionProvider", "interview_provider"),
    ("InterviewSession", "interview_coordinator"),
    ("KeyringProfileKeyProtector", "key_protector"),
    ("PassphraseProfileKeyProtector", "key_protector"),
    ("AgentAuthority", "runtime_policy"),
    ("AuthorizedProfileContextView", "service"),
    ("PersonalContextAuthorityError", "runtime_policy"),
    ("PersonalContextRepository", "repository"),
    ("PersonalContextSettingsSnapshot", "service"),
    ("PersonalContextService", "service"),
    ("ProfileContextRequest", "context_service"),
    ("ProfileContextService", "context_service"),
    ("ProfileContextSnapshot", "context_service"),
    ("ProfileConflictError", "service"),
    ("ProfileKeyCollisionError", "service"),
    ("ProfileKeyMaterial", "key_protector"),
    ("ProfileKeyProtector", "key_protector"),
    ("ProfileLockedError", "key_protector"),
    ("ProfileOperationalStatus", "service"),
    ("ProfileInterviewCoordinator", "interview_coordinator"),
    ("RecordMutation", "service"),
    ("SettingsScopeSnapshot", "service"),
    ("StoredInterviewDraft", "interview_draft_repository"),
    ("ConfiguredModelQuestionProvider", "interview_provider"),
    ("FixedQuestionProvider", "interview_provider"),
    ("build_interview_diff", "interview_diff"),
)

SCRIPT = r"""
import importlib
import json
import os
from pathlib import Path
import sys

from Tests import network_guard, real_profile_guard

network_guard.install()
real_profile_guard.install()
snapshot = json.loads(os.environ[real_profile_guard.PROFILE_SNAPSHOT_ENV])
assert os.environ[real_profile_guard.REAL_HOME_ENV] != os.environ["HOME"]
assert tuple(snapshot["roots"]) == real_profile_guard.protected_roots()
assert Path(snapshot["home"]) == real_profile_guard.resolved_real_home()
mode, encoded = sys.argv[1:]
expected = json.loads(encoded)
root = Path(os.environ["TLDW_TEST_CONFIG_ROOT"])
before = sorted(str(p.relative_to(root)) for p in root.rglob("*"))
assert "tldw_chatbook.config" not in sys.modules
assert "tldw_chatbook.Chat" not in sys.modules
assert "tldw_chatbook.Personal_Context" not in sys.modules

package = importlib.import_module("tldw_chatbook.Personal_Context")
names = [name for name, _ in expected]
assert package.__all__ == names
assert set(names) <= set(dir(package))

if mode == "recovery":
    recovery = importlib.import_module("tldw_chatbook.Personal_Context.recovery")
    owners = recovery.recovery_adapters()
    assert tuple(owner.owner_id for owner in owners) == ("personal_context.repository",)
    assert all(owner.schema_policy().schema_sql for owner in owners)

if mode in {"package", "recovery"}:
    assert "tldw_chatbook.config" not in sys.modules
    assert "tldw_chatbook.Chat" not in sys.modules
    assert "tldw_chatbook.Utils.token_counter" not in sys.modules
    assert "tldw_chatbook.Personal_Context.context_service" not in sys.modules
    assert "tldw_chatbook.Personal_Context.service" not in sys.modules
    assert "tldw_chatbook.Personal_Context.key_protector" not in sys.modules
    assert sorted(str(p.relative_to(root)) for p in root.rglob("*")) == before
else:
    assert mode == "exports"
    for name, module in expected:
        exported = getattr(package, name)
        canonical = getattr(
            importlib.import_module("tldw_chatbook.Personal_Context." + module),
            name,
        )
        assert exported is canonical
        assert getattr(package, name) is canonical
        assert vars(package)[name] is canonical
    namespace = {}
    exec("from tldw_chatbook.Personal_Context import *", namespace)
    assert set(namespace) - {"__builtins__"} == set(names)
    assert all(namespace[name] is getattr(package, name) for name in names)

try:
    getattr(package, "missing_public_export")
except AttributeError:
    pass
else:
    raise AssertionError("unknown public name did not raise AttributeError")
assert set(names) <= set(dir(package))
assert network_guard.blocked_attempts() == ()
assert real_profile_guard.take_violations() == []
print(json.dumps({"mode": mode, "exports": len(names), "qualified": True}))
"""


@pytest.mark.parametrize("mode", ("package", "recovery", "exports"))
def test_personal_context_public_import_contract(tmp_path: Path, mode: str) -> None:
    home, config, data, cache, temporary = (
        tmp_path / name for name in ("home", "config", "data", "cache", "tmp")
    )
    for directory in (home, config, data, cache, temporary):
        directory.mkdir(mode=0o700)
    guard_environment = real_profile_guard.child_environment()
    assert set(guard_environment) == {
        real_profile_guard.REAL_HOME_ENV,
        real_profile_guard.PROFILE_SNAPSHOT_ENV,
    }
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("TLDW_", "PYTEST_"))
    }
    env.update(guard_environment)
    env.update(
        HOME=str(home),
        USERPROFILE=str(home),
        XDG_CONFIG_HOME=str(config),
        XDG_DATA_HOME=str(data),
        XDG_CACHE_HOME=str(cache),
        TMPDIR=str(temporary),
        TEMP=str(temporary),
        TMP=str(temporary),
        TLDW_TEST_CONFIG_ROOT=str(tmp_path),
        PYTHONPATH=os.pathsep.join(
            (str(REPO_ROOT), str(REPO_ROOT / "packages/tldw_profile_core/src"))
        ),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
        HF_HUB_OFFLINE="1",
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", SCRIPT, mode, json.dumps(EXPECTED_EXPORTS)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == {
        "mode": mode,
        "exports": 39,
        "qualified": True,
    }

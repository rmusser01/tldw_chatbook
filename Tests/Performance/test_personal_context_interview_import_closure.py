"""Ordinary profile sends do not initialize the unrelated interview workflow."""

import json

import pytest

from Tests.Performance.test_app_import_weight import _run_isolated_python


@pytest.mark.parametrize("state", ["locked", "absent", "disabled", "ready"])
def test_ordinary_profile_composition_does_not_load_interviews(tmp_path, state):
    result = _run_isolated_python(
        tmp_path,
        f"""
from Tests import real_profile_guard
real_profile_guard.install()
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd() / "packages/tldw_profile_core/src"))
import keyring
from keyring.backends.null import Keyring
keyring.set_keyring(Keyring())
from tldw_chatbook.Personal_Context.bootstrap import bootstrap_personal_context_service
from tldw_chatbook.Personal_Context.key_protector import InMemoryProfileKeyProtector
from tldw_chatbook.Personal_Context.context_service import ProfileContextRequest, ProfileContextService
from tldw_chatbook.Chat.console_personal_context import compose_profile_tool_provider
state = {state!r}
service = bootstrap_personal_context_service(
    db_path={str(tmp_path / "profile.db")!r},
    key_protector=None if state == "locked" else InMemoryProfileKeyProtector(),
)
if state in ("disabled", "ready"):
    service.create_profile()
if state == "ready":
    service.set_runtime_enabled(True)
assert service.status().state.value == state, service.status()
snapshot = ProfileContextService(service).build_snapshot(ProfileContextRequest("hello", 4096))
assert snapshot.serialized_block == ""
provider = compose_profile_tool_provider(service, workspace_id=None, ephemeral=False,
    run_id="run", session_id="session", current_user_message=None, kill_switch=lambda: False)
assert (provider is not None) == (state == "ready")
loaded = sorted(name for name in sys.modules if name.startswith("tldw_chatbook.Personal_Context.interview_"))
assert not real_profile_guard.take_violations()
print(json.dumps({{"state": state, "interviews": loaded}}))
""",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload == {"state": state, "interviews": []}


@pytest.mark.parametrize("submodule_first", [False, True])
def test_interview_public_exports_keep_identity_and_discoverability(
    tmp_path, submodule_first
):
    result = _run_isolated_python(
        tmp_path,
        f"""
from Tests import real_profile_guard
real_profile_guard.install()
import importlib
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd() / "packages/tldw_profile_core/src"))
import tldw_chatbook.Personal_Context as package
exports = {{
    "interview_coordinator": ("InterviewCommitOutcomeUnknownError", "InterviewCommitReceipt", "InterviewSession", "ProfileInterviewCoordinator"),
    "interview_diff": ("InterviewDiff", "InterviewDiffChange", "build_interview_diff"),
    "interview_draft_repository": ("InterviewDraftConflictError", "InterviewDraftExpiredError", "InterviewDraftRepository", "StoredInterviewDraft"),
    "interview_provider": ("ConfiguredModelQuestionProvider", "FixedQuestionProvider", "InterviewProviderError", "InterviewProviderRequest", "InterviewQuestionProvider"),
}}
public = ("EncryptedEnvelope", "EnvelopeCipher", "InMemoryProfileKeyProtector", "InterviewPassphraseKeyProtector", "InterviewCommitOutcomeUnknownError", "InterviewCommitReceipt", "InterviewDiff", "InterviewDiffChange", "InterviewDraftConflictError", "InterviewDraftExpiredError", "InterviewDraftRepository", "InterviewProviderError", "InterviewProviderRequest", "InterviewQuestionProvider", "InterviewSession", "KeyringProfileKeyProtector", "PassphraseProfileKeyProtector", "AgentAuthority", "AuthorizedProfileContextView", "PersonalContextAuthorityError", "PersonalContextRepository", "PersonalContextSettingsSnapshot", "PersonalContextService", "ProfileContextRequest", "ProfileContextService", "ProfileContextSnapshot", "ProfileConflictError", "ProfileKeyCollisionError", "ProfileKeyMaterial", "ProfileKeyProtector", "ProfileLockedError", "ProfileOperationalStatus", "ProfileInterviewCoordinator", "RecordMutation", "SettingsScopeSnapshot", "StoredInterviewDraft", "ConfiguredModelQuestionProvider", "FixedQuestionProvider", "build_interview_diff")
assert tuple(package.__all__) == public
before = set(sys.modules)
assert set(public) | set(exports) <= set(dir(package))
assert set(sys.modules) == before
for module_name, symbols in exports.items():
    if {submodule_first!r}:
        module = importlib.import_module('.' + module_name, package.__name__)
    for name in symbols:
        value = getattr(package, name)
        module = importlib.import_module('.' + module_name, package.__name__)
        assert value is getattr(module, name) and getattr(package, name) is value
        assert vars(package)[name] is value
    assert getattr(package, module_name) is module
namespace = {{}}
exec("from tldw_chatbook.Personal_Context import *", namespace)
assert set(namespace) - {{"__builtins__"}} == set(public)
assert all(namespace[name] is getattr(package, name) for name in public)
try:
    getattr(package, "unknown_profile_export")
except AttributeError:
    pass
else:
    raise AssertionError("unknown attribute must raise AttributeError")
assert not real_profile_guard.take_violations()
""",
    )
    assert result.returncode == 0, result.stdout + result.stderr

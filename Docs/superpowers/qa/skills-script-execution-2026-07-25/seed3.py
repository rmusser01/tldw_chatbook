"""Seed trust EXACTLY the way app.py constructs it (keychain-first marker store)."""
import asyncio, os, shutil
os.environ["TLDW_CONFIG_PATH"] = os.environ["QA_SCRATCH"] + "/config.toml"
from tldw_chatbook.config import get_user_data_dir
from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
from tldw_chatbook.Skills_Interop.skill_trust_service import SkillTrustService
from tldw_chatbook.Skills_Interop.skill_trust_store import (
    SkillTrustStore, skill_trust_account_scope,
    build_default_skill_trust_key_cache, build_skill_trust_marker_store_with_fallback)

store = get_user_data_dir() / "skills"; trust_dir = store / "skills" ; trust_dir = store / "trust"
skills_dir = store / "skills"
scope = skill_trust_account_scope(trust_dir)
marker_store, reduced = build_skill_trust_marker_store_with_fallback(
    fallback_marker_path=trust_dir / "generation_marker.json", account_scope=scope)
print("marker store:", type(marker_store).__name__, "reduced_rollback:", reduced)
trust = SkillTrustService(
    skills_dir=skills_dir,
    trust_store=SkillTrustStore(store_dir=trust_dir, marker_store=marker_store),
    key_cache=build_default_skill_trust_key_cache(account_scope=scope),
    keyring_convenience_enabled=False,
    reduced_rollback_protection=reduced)

# Start clean: drop any prior manifest/marker/keys from earlier seeding attempts.
try:
    trust.reset_trust()
    print("reset_trust ok")
except Exception as e:
    print("reset_trust:", e)
for n in ("demo-runner", "grant-demo"):
    shutil.rmtree(skills_dir / n, ignore_errors=True)
(store / "tldw_chatbook_skills.json").unlink(missing_ok=True)

trust.bootstrap_trust("qa-579-passphrase")
trust.enable_keyring_convenience()
svc = LocalSkillsService(store_dir=store, trust_service=trust)

async def main():
    await svc.create_skill(name="demo-runner",
        content="---\nname: demo-runner\ndescription: QA579 greet skill\n---\nRun scripts/hello.py when asked to greet.\n",
        supporting_files={"scripts/hello.py": "print('QA579 HELLO FROM THE SCRIPT')\n"}, trust_approved=True)
    await svc.create_skill(name="grant-demo",
        content="---\nname: grant-demo\ndescription: QA579 grant skill\n---\nRun scripts/mark.py when asked to mark.\n",
        supporting_files={"scripts/mark.py": "print('QA579 MARK v1')\n"}, trust_approved=True)
    print("posture:", trust.trust_posture())
    for n in ("demo-runner", "grant-demo"):
        print(f"  {n}: {trust.status_for_skill(n).trust_status} granted={trust.script_execution_granted(n)}")
asyncio.run(main())

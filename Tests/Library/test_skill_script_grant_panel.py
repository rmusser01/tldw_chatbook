"""Grant visibility + revoke in the Library skills trust panel."""

from tldw_chatbook.Widgets.Library.library_skills_canvas import skill_script_grant_line


def test_line_states_when_scripts_may_run_without_asking():
    line = skill_script_grant_line(True)
    assert "without asking" in line.lower()


def test_line_states_when_every_run_is_confirmed():
    line = skill_script_grant_line(False)
    assert "confirm" in line.lower() or "asked" in line.lower()


def test_revoking_clears_the_grant(trust_service_with_skill):
    service, name = trust_service_with_skill
    service.grant_script_execution(name)
    assert service.script_execution_granted(name) is True
    service.revoke_script_execution(name)
    assert service.script_execution_granted(name) is False

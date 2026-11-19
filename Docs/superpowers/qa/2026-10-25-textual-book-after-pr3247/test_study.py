import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from textual.widgets import Button, Input, Static
from Tests.private_profile import private_profile_test
from Tests.UI.test_study_quiz_start_responsiveness import Host, Client, QUESTION, open_quizzes, start_button, settle, until, release_on_exit
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Study_Interop import QuizScopeService, LocalQuizService, ServerQuizService
from tldw_chatbook.UI.Screens.study_scope_models import StudyScopeContext, StudyScopeType

OUT = Path(__file__).parent

def receipt(name, data):
    (OUT / (name + '.json')).write_text(json.dumps(data, indent=2), encoding='utf-8')

class RecordedLocal(LocalQuizService):
    def __init__(self, db):
        super().__init__(db)
        self.submitted = []
    def submit_attempt(self, attempt_id, *, answers):
        self.submitted.append({'attempt_id': str(attempt_id), 'answers': answers})
        return super().submit_attempt(attempt_id, answers=answers)

class RecordedClient(Client):
    def __init__(self):
        super().__init__()
        self.submitted = []
    async def list_quizzes(self, **kwargs):
        return [{'id': 1, 'name': 'Global server quiz', 'total_questions': 1, 'workspace_id': None}, {'id': 2, 'name': 'Workspace B quiz', 'total_questions': 1, 'workspace_id': 'B'}]
    async def list_quiz_questions(self, quiz_id, **kwargs):
        q = dict(QUESTION)
        if quiz_id == 2:
            q.update(id=12, quiz_id=2, question_text='Workspace B question')
        return [q]
    async def submit_quiz_attempt(self, attempt_id, payload):
        self.submitted.append({'attempt_id': attempt_id, 'answers': payload.model_dump()})
        return {'id': attempt_id, 'quiz_id': 1, 'score': 1, 'total_possible': 1, 'questions': [dict(QUESTION)]}

@pytest.mark.asyncio
@pytest.mark.parametrize('transition', ['ordinary', 'runtime-local', 'workspace-B', 'runtime-away-back'])
@private_profile_test
async def test_pending_start_publication_observation(request, transition):
    client = RecordedClient()
    db = CharactersRAGDB(db_path=':memory:', client_id='pr3247-audit-study')
    db.create_quiz(name='Local only quiz')
    local = RecordedLocal(db)
    app = Host(client)
    app.instance.study_quiz_scope_service = QuizScopeService(local_service=local, server_service=ServerQuizService(client))
    try:
        async with app.run_test(size=(180, 80)) as pilot, release_on_exit(client):
            screen, window = await open_quizzes(app, pilot)
            original = window.query_one('#quiz-answer-input', Input)
            start_button(screen, 'shell').press()
            await asyncio.wait_for(client.entered.wait(), 5)
            held = set(app.workers)
            async def drain_new():
                await pilot.pause()
                for worker in list(app.workers):
                    if worker not in held:
                        await worker.wait()
                await pilot.pause()
            if transition in {'runtime-local', 'runtime-away-back'}:
                app.instance.current_runtime_backend = 'local'
                await screen.handle_runtime_backend_changed('local')
                await drain_new()
                assert window.quizzes_controller._current_mode() == 'local'
                assert window.quizzes_controller.current_attempt_id is None
                if transition == 'runtime-away-back':
                    app.instance.current_runtime_backend = 'server'
                    await screen.handle_runtime_backend_changed('server')
                    await drain_new()
            elif transition == 'workspace-B':
                await screen._apply_scope_context_and_refresh(StudyScopeContext(scope_type=StudyScopeType.WORKSPACE, workspace_id='B', workspace_name='Workspace B'), study_window=window)
                await drain_new()
                assert window.quizzes_controller._workspace_id() == 'B'
                assert window.quizzes_controller.current_attempt_id is None
            assert original.is_attached
            original.value = 'new-context draft'
            await pilot.pause()
            before = {'mode': window.quizzes_controller._current_mode(), 'scope': window.quizzes_controller._scope_arguments(), 'selected_quiz': window.quizzes_controller.selected_quiz_label(), 'draft': original.value, 'attempt_id': window.quizzes_controller.current_attempt_id, 'attached': original.is_attached}
            client.release.set()
            await settle(app, pilot)
            after = {'mode': window.quizzes_controller._current_mode(), 'scope': window.quizzes_controller._scope_arguments(), 'draft': original.value, 'attempt_id': window.quizzes_controller.current_attempt_id, 'question': str(window.query_one('#quiz-attempt-question', Static).render()), 'status': str(window.query_one('#quiz-attempt-status', Static).render()), 'selected_quiz': window.quizzes_controller.selected_quiz_label()}
            original.value = '4'
            await pilot.pause()
            submit = window.query_one('#submit-quiz-answer-button', Button)
            assert not submit.disabled
            submit.press()
            await settle(app, pilot)
            record = {'transition': transition, 'before_release': before, 'after_release': after, 'local_submissions': local.submitted, 'server_submissions': client.submitted, 'notices': app.notices, 'final_status': str(window.query_one('#quiz-attempt-status', Static).render()), 'app_exception': repr(app._exception), 'server_starts': client.calls, 'cancelled': client.cancelled}
            receipt('publication-' + transition, record)
            assert app._exception is None
            assert client.calls == 1 and client.cancelled == 0
            assert after['attempt_id'] == '21' and after['question'] == 'Two plus two'
            assert after['draft'] == ''
            if transition == 'runtime-local':
                assert len(local.submitted) == 1 and not client.submitted
                assert 'Failed to submit quiz attempt.' in app.notices
            else:
                assert len(client.submitted) == 1 and not local.submitted
    finally:
        client.release.set()
        db.close()

@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['request', 'policy'])
@private_profile_test
async def test_shell_resume_after_failed_start_observation(request, failure):
    client = RecordedClient()
    client.release.set()
    client.fail_once = failure == 'request'
    app = Host(client)
    if failure == 'policy':
        app.instance.require_ui_action_allowed = lambda **kwargs: SimpleNamespace(allowed=False)
    async with app.run_test(size=(180,80)) as pilot, release_on_exit(client):
        screen, window = await open_quizzes(app, pilot)
        resume = screen.query_one('#study-resume-last', Button)
        assert resume.disabled and screen.current_study_session is None
        start_button(screen, 'shell').press()
        await settle(app, pilot)
        assert window.quizzes_controller.current_attempt_id is None
        screen.query_one('#view-dashboard-btn', Button).press()
        await settle(app, pilot)
        record = {'failure': failure, 'session': screen.current_study_session, 'is_studying': screen.is_studying, 'resume_label': str(resume.label), 'resume_enabled': not resume.disabled, 'notices': app.notices, 'calls': client.calls, 'app_exception': repr(app._exception)}
        assert not resume.disabled
        resume.press()
        await settle(app, pilot)
        record['section_after_resume'] = screen.current_section
        record['attempt_after_resume'] = window.quizzes_controller.current_attempt_id
        receipt('resume-' + failure, record)
        assert screen.current_section == 'quizzes'
        assert window.quizzes_controller.current_attempt_id is None
        assert app._exception is None

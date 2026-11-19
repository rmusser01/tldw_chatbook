"""Private read-only current-run attach observations, never setup authority."""
import asyncio
import json
from pathlib import Path
import time

# Exact marker order in sealed ca080 helper, including original owned worker.
PHASES=('attach_entered','ui_ready_wait_before','ui_ready_wait_after','navigate_chat_before','navigate_chat_after','ensure_controller_before','ensure_controller_after','gateway_peek_before','gateway_peek_after','fixture_directory_before','fixture_directory_after','registry_peek_before','registry_peek_after','workspace_owned_call_before','workspace_owned_worker_entered','create_workspace_before','create_workspace_after','add_folder_binding_before','add_folder_binding_after','set_workspace_before','set_workspace_after','workspace_owned_worker_returned','workspace_owned_call_after','new_session_before','new_session_after','service_peek_before','service_peek_after','global_ask_before','global_ask_after','bridge_observer_before','bridge_observer_after','attach_snapshot_before','attach_snapshot_after')


def read_owned(run,name,limit):
    path=run/name
    if path.resolve()!=run.resolve()/name:raise RuntimeError('Actual_attach_file_outside_owned_run')
    try:
        with path.open('rb') as stream:data=stream.read(limit+1)
    except FileNotFoundError:return None
    if len(data)>limit:raise RuntimeError('Actual_attach_file_size_bound')
    return data


def read_attach_snapshot(run,identity,request_before_ns,request_after_ns,previous):
    control_raw=read_owned(run,'product-control.json',4096)
    if control_raw is None:raise RuntimeError('Actual_attach_current_child_identity_missing')
    control=json.loads(control_raw)
    if any(control.get(key)!=identity[key] for key in ('pid','source_root','nonce','port','provider_port')):raise RuntimeError('Actual_attach_current_child_identity_changed')
    job_raw=read_owned(run,'attach-job.json',65536)
    raw=read_owned(run,'attach-stages.jsonl',65536)
    job_again=read_owned(run,'attach-job.json',65536)
    now_ns=time.perf_counter_ns()  # After reads, not before a concurrent append.
    phases=[];partial=False
    if raw is not None:
        pieces=raw.split(b'\n');tail=pieces.pop();partial=bool(tail)
        if len(tail)>4096 or len(pieces)>len(PHASES):raise RuntimeError('Actual_attach_phase_record_bound')
        for index,line in enumerate(pieces):
            if not line or len(line)>4096:raise RuntimeError('Actual_attach_phase_line_invalid')
            row=json.loads(line)
            if type(row) is not dict or set(row)!={'stage','timestamp_ns','pid','thread_id'}:raise RuntimeError('Actual_attach_phase_schema_invalid')
            if row['stage']!=PHASES[index] or type(row['pid']) is not int or row['pid']!=identity['pid'] or type(row['thread_id']) is not int or row['thread_id']<=0:raise RuntimeError('Actual_attach_phase_owner_or_order_invalid')
            stamp=row['timestamp_ns']
            if type(stamp) is not int or not request_before_ns<=stamp<=now_ns or (phases and stamp<phases[-1]['timestamp_ns']):raise RuntimeError('Actual_attach_phase_timestamp_invalid')
            phases.append(row)
    if len(phases)<len(previous) or phases[:len(previous)]!=previous:raise RuntimeError('Actual_attach_phase_history_changed')
    if job_raw!=job_again:return {'status':'pending_observation','phases':phases,'observation':'job_bytes_changed_during_read'}
    if job_raw is None:return {'status':'pending_observation','phases':phases,'observation':'incomplete_phase' if partial else ('complete_rows' if phases else 'missing_phase_and_job')}
    try:job=json.loads(job_raw)
    except (json.JSONDecodeError,UnicodeDecodeError):
        # Original write_text is not atomic/framed: malformed versus partial
        # cannot be proved here. Neither releases readiness; retain a miss and
        # let the unchanged phase/total deadline reject persistent unreadability.
        return {'status':'pending_observation','phases':phases,'observation':'job_json_incomplete_or_malformed'}
    if type(job) is not dict or job.get('status') not in {'completed','failed'}:raise RuntimeError('Actual_attach_job_status_invalid')
    started=job.get('started_ns');finished=job.get('finished_ns')
    if type(started) is not int or not request_before_ns<=started<=request_after_ns or type(finished) is not int or not started<=finished<=now_ns:raise RuntimeError('Actual_attach_job_identity_or_time_invalid')
    if partial:raise RuntimeError('Actual_attach_finished_job_has_incomplete_phase')
    if phases and (phases[0]['timestamp_ns']<started or finished<phases[-1]['timestamp_ns']):raise RuntimeError('Actual_attach_job_precedes_phase')
    if job['status']=='completed':
        if len(phases)!=len(PHASES) or type(job.get('result')) is not dict:raise RuntimeError('Actual_attach_completed_without_full_current_phases')
    elif type(job.get('error_type')) is not str:raise RuntimeError('Actual_attach_failed_job_schema_invalid')
    return {**job,'phases':phases,'observation':'complete_current_job'}


async def observe_attach(run,identity,request_before_ns,request_after_ns,result):
    started=last_progress=time.monotonic();last_phase=None;previous=[];polls=[]
    result['attach_job_polls']=polls
    while True:
        poll_started=time.monotonic()
        phase_age=poll_started-last_progress;total_age=poll_started-started
        if phase_age>=60 or total_age>=300:
            result['attach_job_failure']={'status':'observation_deadline','last_known_phase':last_phase,'phase_age_s':phase_age,'total_age_s':total_age}
            raise TimeoutError('Actual_attach_phase_or_total_bound_exceeded')
        answer=read_attach_snapshot(run,identity,request_before_ns,request_after_ns,previous)
        # Preserve V13 late-response fence against PRIOR progress, even if an
        # actual file read is delayed or a completed/new-phase record appears.
        observed_at=time.monotonic();phase_age=observed_at-last_progress;total_age=observed_at-started
        if phase_age>=60 or total_age>=300:
            result['attach_job_failure']={'status':'late_passive_observation','reply_status':answer['status'],'last_known_phase':last_phase,'phase_age_s':phase_age,'total_age_s':total_age}
            raise TimeoutError('Actual_attach_phase_or_total_bound_exceeded')
        phases=answer['phases'];previous=phases
        phase=(phases[-1]['stage'],phases[-1]['timestamp_ns']) if phases else None
        # Only an actual new complete phase resets progress. File mtime,
        # repeated reads, missing files and incomplete tails never do.
        if phase!=last_phase:last_progress=observed_at;last_phase=phase
        polls.append({'python_ns':time.perf_counter_ns(),'status':answer['status'],'observation':answer['observation'],'phases':phases,'duration_s':observed_at-poll_started})
        if len(polls)>2048:raise RuntimeError('Actual_attach_poll_evidence_bound')
        if answer['status']=='completed':return answer['result']
        if answer['status']=='failed':
            result['attach_job_failure']=answer
            raise RuntimeError('Actual_attach_job_failed')
        await asyncio.sleep(.2)

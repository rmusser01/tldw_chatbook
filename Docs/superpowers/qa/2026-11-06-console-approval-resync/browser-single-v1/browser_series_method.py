"""Browser transport bounds; genuine frame reviews remain independent evidence."""
import hashlib,json,math,statistics
from pathlib import Path
from native_series_method import runtime_round_limit,planned_outcomes,nearest_rank_p95
from browser_presented_frames import duration_upper_ms


def authorized_rounds(selection,run):
    if selection.get('diagnostic_stack') is not False:raise RuntimeError('Browser_C_diagnostics_forbidden')
    limit=runtime_round_limit(selection,selection.get('method_authorization'))
    if limit==1:return limit
    source=Path(selection['source_root']).resolve();auth=selection['method_authorization'];freeze=Path(selection['freeze_path']);raw=freeze.read_bytes();frozen=json.loads(raw)
    if (frozen.get('qualification_authorized') is not True or frozen.get('revision')!=selection.get('revision') or Path(frozen.get('source_root','')).resolve()!=source
        or auth.get('revision')!=selection.get('revision') or auth.get('freeze_sha256')!=hashlib.sha256(raw).hexdigest() or Path(auth.get('source_root','')).resolve()!=source
        or auth.get('runtime_sha256')!=selection['fixture_hashes']):raise RuntimeError('Browser_qualification_source_freeze_binding_missing')
    for name,digest in selection['fixture_hashes'].items():
        path=(Path(run)/name).resolve()
        if not path.is_relative_to(Path(run).resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise RuntimeError('Browser_runtime_bytes_changed')
    for key in ('feedback_review','terminal_review'):
        ref=auth.get(key)
        if type(ref) is not dict:raise RuntimeError('Actual_browser_reference_review_missing')
        path=Path(ref['path']);data=path.read_bytes()
        if hashlib.sha256(data).hexdigest()!=ref['sha256']:raise RuntimeError('Browser_reference_review_changed')
        review=json.loads(data)
        if review.get('observed') is not True or not review.get('reviewer'):raise RuntimeError('Browser_reference_not_observed')
        image=Path(review['source_png'])
        if hashlib.sha256(image.read_bytes()).hexdigest()!=review['png_sha256']:raise RuntimeError('Browser_reference_image_changed')
    return limit


def eligible(frame):
    return (frame.get('presented_us') is not None and frame.get('rejection','missing') is None
            and frame.get('identity_matches')==frame.get('eligible_matches')==1)


def select_observed_frame(run,frames,review,*,correlation,state):
    if (review.get('observed') is not True or not review.get('reviewer') or review.get('correlation')!=correlation
            or review.get('state')!=state):raise RuntimeError('Browser_actual_frame_review_missing_or_stale')
    if state=='terminal':
        if review.get('visible_nonce_prefix')!=correlation[:12] or review.get('terminal_UI_verified') is not True:raise RuntimeError('Browser_terminal_generation_not_observed')
    elif review.get('full_nonce_verified') is not True:raise RuntimeError('Browser_full_nonce_not_observed')
    candidates=[f for f in frames if f.get('image')==review.get('image') and f.get('image_sha256')==review.get('image_sha256')]
    if len(candidates)!=1 or not eligible(candidates[0]):raise RuntimeError('Browser_reviewed_frame_not_unique_presented')
    frame=candidates[0];path=(Path(run)/frame['image']).resolve()
    if not path.is_relative_to(Path(run).resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=frame['image_sha256']:raise RuntimeError('Browser_actual_frame_bytes_changed')
    return frame


def _clock_interval(interval):
    lo,hi=interval['lower_ms'],interval['upper_ms']
    if type(lo) not in (int,float) or type(hi) not in (int,float) or not math.isfinite(lo) or not math.isfinite(hi) or lo>hi:raise ValueError('Invalid_clock_interval')
    return lo,hi


def server_duration(frame,server_start_ns,calibration):
    if not eligible(frame) or type(server_start_ns) is not int or server_start_ns<0:raise ValueError('Missing_actual_boundary')
    trace=calibration['envelope']['trace_page'];server=calibration['envelope']['server_page'];tl,th=_clock_interval(trace);sl,sh=_clock_interval(server)
    lower=frame['presented_us']/1000-th+sl-server_start_ns/1e6
    upper=duration_upper_ms(presented_trace_us=frame['presented_us'],server_start_ns=server_start_ns,trace_page=trace,server_page=server)
    if not math.isfinite(upper) or upper<0:raise ValueError('Inconsistent_actual_boundary_order')
    return {'lower_ms':lower,'upper_ms':upper,'clock_error_width_ms':(th-tl)+(sh-sl)}


def browser_duration(frame,event,calibration):
    if not eligible(frame) or event.get('trusted') is not True:raise ValueError('Missing_trusted_browser_input')
    start=event.get('timestamp_ms')
    if type(start) not in (int,float) or not math.isfinite(start):raise ValueError('Invalid_browser_input_clock')
    lo,hi=_clock_interval(calibration['envelope']['trace_page']);lower=frame['presented_us']/1000-hi-start;upper=frame['presented_us']/1000-lo-start
    if upper<0:raise ValueError('Browser_input_after_feedback')
    return {'lower_ms':lower,'upper_ms':upper,'clock_error_width_ms':hi-lo}


def stage_times(rows,correlation):
    owned=[r for r in rows if r.get('correlation')==correlation];out={}
    for name in ('request_complete','local_checks_complete','card_commit_enter','resolver_entered','worker_released'):
        found=[r for r in owned if r.get('stage')==name]
        if len(found)!=1 or type(found[0].get('timestamp_ns')) is not int:raise ValueError('Missing_or_ambiguous_owner_stage:'+name)
        out[name]=found[0]['timestamp_ns']
    for kind in ('MouseDown','MouseUp'):
        found=[r['timestamp_ns'] for r in owned if r.get('stage')=='app_input_prequeue' and r.get('kind')==kind]
        if len(found)!=1 or type(found[0]) is not int:raise ValueError('Missing_or_ambiguous_app_input:'+kind)
        out[kind]=found[0]
    if not (out['request_complete']<=out['local_checks_complete']<=out['MouseDown']<=out['MouseUp']<=out['card_commit_enter']<=out['resolver_entered']<=out['worker_released']):raise ValueError('Actual_owner_stage_order_inconsistent')
    return out


def sample_metrics(rows,correlation,actionable,feedback,card_clock,feedback_clock,browser_input):
    stamps=stage_times(rows,correlation)
    return {'local_feedback':server_duration(feedback,stamps['MouseDown'],feedback_clock),'local_actionable':server_duration(actionable,stamps['local_checks_complete'],card_clock),
            'transport_inclusive_feedback':browser_duration(feedback,browser_input,feedback_clock),'app_MouseDown_to_MouseUp_ms':(stamps['MouseUp']-stamps['MouseDown'])/1e6,
            'committing_MouseUp_to_handler_ms':(stamps['card_commit_enter']-stamps['MouseUp'])/1e6,'request_complete_to_local_checks_ms':(stamps['local_checks_complete']-stamps['request_complete'])/1e6,
            'prior_presented_output_to_card':None,'real_provider_generation':None,'real_model_continuation':None}


def summarize(records,planned):
    all_rows=planned_outcomes(records,planned);warm=[r for r in all_rows if r['round_index']>0];cold=all_rows[:1]
    valid=[r for r in warm if r.get('status')=='complete' and r.get('metrics') is not None]
    distributions={}
    for field in ('local_feedback','local_actionable','transport_inclusive_feedback'):
        values=[r['metrics'][field]['upper_ms'] for r in valid]
        distributions[field]={'sample_count':len(values),'planned_warm':len(warm),'missing_warm':len(warm)-len(values),'median_ms':statistics.median(values) if values else None,'p95_ms':nearest_rank_p95(values) if len(values)==len(warm) and values else None,'maximum_ms':max(values) if values else None}
    return {'qualified':False,'product_samples':0,'qualification_pending_full_source_worker_effect_closure':True,'planned':planned,'all_planned_outcomes':all_rows,'first_use':cold,'distributions':distributions,'method_boundaries_complete':planned==41 and len(valid)==40}


def terminal_pair(run,frames,reviews,*,correlation,ready_server_ns,calibration):
    if type(reviews) is not list or len(reviews)!=2:raise RuntimeError('Two_actual_terminal_reviews_required')
    selected=[select_observed_frame(run,frames,r,correlation=correlation,state='terminal') for r in reviews]
    if len({(f['source_id'],f['frame_sequence']) for f in selected})!=2 or not selected[0]['presented_us']<selected[1]['presented_us']:raise RuntimeError('Terminal_presentations_not_distinct_and_ordered')
    for frame in selected:
        interval=server_duration(frame,ready_server_ns,calibration)
        if interval['lower_ms']<0:raise RuntimeError('Terminal_presentation_not_after_actual_ready_owner_boundary')
    return selected


def validate_clock_series(calibrations):
    if type(calibrations) is not list or len(calibrations) not in (2,82):raise ValueError('Validation2_or_qualification82_trace_segments_required')
    pid=calibrations[0].get('page_pid')
    if type(pid) is not int or any(c.get('page_pid')!=pid for c in calibrations):raise ValueError('Actual_segment_renderer_changed')
    anchors={};segments=[]
    for clock in ('trace_page','server_page'):
        first=calibrations[0]['intervals']['before'][clock];last=calibrations[-1]['intervals']['after'][clock]
        if first.get('samples')!=20 or last.get('samples')!=20:raise ValueError('Process20_before_after_anchors_missing')
        fl,fh=_clock_interval(first);ll,lh=_clock_interval(last);anchors[clock]={'lower_ms':min(fl,ll),'upper_ms':max(fh,lh),'before_after_center_shift_ms':((ll+lh)-(fl+fh))/2}
    for index,c in enumerate(calibrations):
        if type(c.get('round_index')) is not int or c['round_index']!=index//2 or c.get('role')!=('actionable' if index%2==0 else 'feedback'):raise ValueError('Actual_segment_round_role_order_mismatch')
        for phase in ('before','after'):
            expected=20 if (index==0 and phase=='before') or (index==len(calibrations)-1 and phase=='after') else 1
            for clock in ('trace_page','server_page'):
                if c['intervals'][phase][clock].get('samples')!=expected:raise ValueError('Unexpected_segment_clock_sample_count')
                lo,hi=_clock_interval(c['intervals'][phase][clock]);anchor=anchors[clock]
                if max(lo,anchor['lower_ms'])>min(hi,anchor['upper_ms']):raise ValueError('Actual_endpoint_clock_origin_or_drift_mismatch')
        for clock in ('trace_page','server_page'):
            lo,hi=_clock_interval(c['envelope'][clock]);anchor=anchors[clock]
            if max(lo,anchor['lower_ms'])>min(hi,anchor['upper_ms']):raise ValueError('Actual_segment_clock_origin_or_drift_mismatch')
        segments.append({'segment_index':index,'clock_envelope':c['envelope'],'known_clock_error_upper_ms':c['known_clock_error_upper_ms']})
    return {'renderer_pid':pid,'segments':len(calibrations),'process_anchors':anchors,'actual_segment_mappings':segments,'assumed_shared_trace_origin':False,'overhead_subtracted':False}


def validate_capture_limits(path):
    path=Path(path)
    if path.stat().st_size>64*1024*1024:raise ValueError('Persisted_trace_JSON_64MiB_representation_cap')
    data=json.loads(path.read_text(encoding='utf-8'))
    if data.get('tracing_complete',{}).get('dataLossOccurred') is not False:raise ValueError('Actual_trace_loss_or_missing_completion')
    if len(data['traceEvents'])>131072:raise ValueError('Trace_decoded_row_budget_8MiB_divided_by64')
    shots=[r for r in data['traceEvents'] if r.get('name')=='Screenshot']
    if len(shots)>50:raise ValueError('Actual_segment_50_screenshot_overflow_invalid')
    for shot in shots:
        snapshot=shot.get('args',{}).get('snapshot')
        if type(snapshot) is not str or len(snapshot)>4*((1024*1024+2)//3):raise ValueError('Actual_JPEG_snapshot_1MiB_cap_or_missing')
    return {'Chrome_trace_buffer_KiB':8192,'screenshot_count':len(shots),'screenshot_cap':50,'trace_rows':len(data['traceEvents']),'trace_row_cap':131072,'persisted_JSON_bytes':path.stat().st_size,'JSON_representation_cap_bytes':64*1024*1024,'all_overflow_retained_invalid':True}


def validate_capture_series(paths):
    if type(paths) is not list or len(paths) not in (2,82):raise ValueError('Exact2_or82_capture_segments_required')
    rounds=len(paths)//2;reports=[validate_capture_limits(p) for p in paths]
    for index in range(rounds):
        if sum(r['screenshot_count'] for r in reports[index*2:index*2+2])>64:raise ValueError('Combined_round64_screenshot_overflow_invalid')
    total_rows=sum(r['trace_rows'] for r in reports);total_bytes=sum(r['persisted_JSON_bytes'] for r in reports);total_shots=sum(r['screenshot_count'] for r in reports)
    limits={'segments':rounds*2,'screenshots':rounds*64,'trace_rows':rounds*2*131072,'JSON_bytes':rounds*2*64*1024*1024}
    if total_rows>limits['trace_rows'] or total_bytes>limits['JSON_bytes'] or total_shots>limits['screenshots']:raise ValueError('Actual_whole_process_trace_budget_exceeded')
    return {'rounds':rounds,'segments':len(paths),'actual_trace_rows':total_rows,'actual_JSON_bytes':total_bytes,'actual_screenshots':total_shots,'limits':limits,'Chrome_ring_KiB_per_segment':8192,'overflow_retained_invalid':True}

"""QA-only strict actual-presentation joins and explicit clock intervals."""
from collections import defaultdict
import json


def presented_frames(events, *, page_pid):
    """Return joined page screenshot frames, retaining every rejected screenshot."""
    active = defaultdict(list)
    pipelines = []
    unmatched = 0
    for event in sorted(events, key=lambda item: item.get('ts', 0)):
        if event.get('name') != 'PipelineReporter':
            continue
        key = (event['pid'], json.dumps(event.get('id2'), sort_keys=True))
        if event['ph'] == 'b':
            active[key].append(event)
        elif event['ph'] == 'e':
            if not active[key]:
                unmatched += 1
                continue
            begin = active[key].pop()
            pipelines.append({'pid':begin['pid'], 'begin_us':begin['ts'], 'presented_us':event['ts'], **begin.get('args',{}).get('frame_reporter',{})})
    presented = {event['ts'] for event in events if event.get('name') == 'FramePresented'}
    displayed = {event['ts'] for event in events if event.get('name') == 'Display::FrameDisplayed'}
    output = []
    for event in events:
        if event.get('name') != 'Screenshot':
            continue
        args = event['args']
        matches = [item for item in pipelines if item['pid'] == page_pid and item.get('frame_source') == args.get('source_id') and item.get('frame_sequence') == args.get('frame_sequence')]
        eligible = [item for item in matches if item.get('state') == 'STATE_PRESENTED_ALL' and item.get('has_missing_content') is False and item.get('checkerboarded_needs_raster') is False and item.get('checkerboarded_needs_record') is False and item['presented_us'] in presented and item['presented_us'] in displayed]
        valid = len(matches) == len(eligible) == 1
        output.append({'source_id':args.get('source_id'), 'frame_sequence':args.get('frame_sequence'), 'identity_matches':len(matches), 'eligible_matches':len(eligible), 'presented_us':eligible[0]['presented_us'] if valid else None, 'snapshot':args.get('snapshot'), 'rejection':None if valid else 'missing_ambiguous_or_incomplete_presentation'})
    return output, {'unmatched_ends':unmatched, 'open_reporters':sum(map(len,active.values()))}


def offset_interval(samples, *, quantum_ms=0.1):
    """Intersect observed remote timestamps with local before/after brackets."""
    assert samples
    lows = [sample['remote_ms'] - sample['after_ms'] - quantum_ms for sample in samples]
    highs = [sample['remote_ms'] - sample['before_ms'] + quantum_ms for sample in samples]
    lower, upper = max(lows), min(highs)
    assert lower <= upper, 'clock_brackets_inconsistent'
    return {'lower_ms':lower, 'upper_ms':upper, 'width_ms':upper-lower, 'samples':len(samples), 'quantum_allowance_ms':quantum_ms}


def duration_upper_ms(*, presented_trace_us, server_start_ns, trace_page, server_page):
    """Conservative cross-clock duration, never assuming a shared clock origin."""
    return presented_trace_us / 1000 - trace_page['lower_ms'] + server_page['upper_ms'] - server_start_ns / 1e6


def self_check(path):
    """Controls mutate actual synthetic trace joins, retaining semantic scope."""
    from copy import deepcopy
    from pathlib import Path
    rows = json.loads(Path(path).read_text(encoding='utf-8'))['traceEvents']
    pid = next(row['pid'] for row in rows if row.get('name') == 'qa_gesture')
    frames, diagnostics = presented_frames(rows, page_pid=pid)
    assert sum(frame['presented_us'] is not None for frame in frames) == 4
    assert frames[0]['presented_us'] is None  # actual blank/foreign frame
    assert not diagnostics['unmatched_ends'] and not diagnostics['open_reporters']
    wrong_owner, _ = presented_frames(rows, page_pid=-1)
    assert all(frame['presented_us'] is None for frame in wrong_owner)
    missing = [row for row in rows if row.get('name') != 'FramePresented']
    assert all(frame['presented_us'] is None for frame in presented_frames(missing,page_pid=pid)[0])
    partial = deepcopy(rows)
    for row in partial:
        if row.get('name') == 'PipelineReporter' and row['pid'] == pid and row['ph'] == 'b':
            row['args']['frame_reporter']['state'] = 'STATE_PRESENTED_PARTIAL'
    assert all(frame['presented_us'] is None for frame in presented_frames(partial,page_pid=pid)[0])
    duplicated = deepcopy(rows)
    wanted = frames[1]['frame_sequence']
    begin = next(row for row in rows if row.get('name') == 'PipelineReporter' and row['pid'] == pid and row['ph'] == 'b' and row.get('args',{}).get('frame_reporter',{}).get('frame_sequence') == wanted)
    end = next(row for row in rows if row.get('name') == 'PipelineReporter' and row['pid'] == pid and row['ph'] == 'e' and row.get('id2') == begin.get('id2') and row['ts'] == frames[1]['presented_us'])
    for row in (begin,end):
        clone = deepcopy(row)
        clone['id2'] = {'local':'0xqa-duplicate'}
        duplicated.append(clone)
    result = presented_frames(duplicated,page_pid=pid)[0]
    assert next(frame for frame in result if frame['frame_sequence'] == wanted)['presented_us'] is None
    interval = offset_interval([{'remote_ms':110,'before_ms':9,'after_ms':11}],quantum_ms=0)
    assert interval['lower_ms'] == 99 and interval['upper_ms'] == 101
    assert duration_upper_ms(presented_trace_us=200_000,server_start_ns=130_000_000,trace_page={'lower_ms':99},server_page={'upper_ms':100}) == 71
    return {'actual_joins':4,'foreign_initial_frame_rejected':True,'wrong_owner_rejected':True,'missing_actual_presentation_rejected':True,'partial_presentation_rejected':True,'ambiguous_identity_rejected':True,'clock_upper_bound_control':True}

if __name__ == '__main__':
    import sys
    print(json.dumps(self_check(sys.argv[1])))

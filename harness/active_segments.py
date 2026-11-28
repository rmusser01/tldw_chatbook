from collections import defaultdict
import hashlib
import json
from pathlib import Path

def segments(rows):
    stacks = defaultdict(list)
    complete, clipped = ([], [])
    for row in sorted(rows, key=lambda r: r['ns']):
        key = tuple((row[k] for k in ('thread', 'task', 'file', 'line', 'function')))
        stack = stacks[key]
        event = row['event']
        if event == 'start':
            stack.append({'entry': row, 'active': row, 'parts': []})
        elif event in ('resume', 'yield', 'return', 'unwind'):
            if not stack:
                clipped.append(row)
                continue
            activation = stack[-1]
            if event == 'resume':
                assert activation['active'] is None
                activation['active'] = row
                continue
            start = activation['active']
            assert start is not None
            assert row['ns'] >= start['ns'] and row['cpu_ns'] >= start['cpu_ns']
            activation['parts'].append({'start_ns': start['ns'], 'end_ns': row['ns'], 'wall_ms': (row['ns'] - start['ns']) / 1000000.0, 'thread_cpu_ms': (row['cpu_ns'] - start['cpu_ns']) / 1000000.0})
            activation['active'] = None
            if event != 'yield':
                stack.pop()
                entry = activation['entry']
                parts = activation['parts']
                complete.append({k: entry[k] for k in ('function', 'file', 'thread', 'task')} | {'start_ns': entry['ns'], 'end_ns': row['ns'], 'inclusive_wall_ms': (row['ns'] - entry['ns']) / 1000000.0, 'active_wall_ms': sum((p['wall_ms'] for p in parts)), 'active_thread_cpu_ms': sum((p['thread_cpu_ms'] for p in parts)), 'segments': parts, 'outcome': event})
    return (complete, clipped, [a['entry'] for stack in stacks.values() for a in stack])

def self_check():

    def row(event, ns, cpu, task=1, function='f'):
        return dict(event=event, ns=ns, cpu_ns=cpu, thread=1, task=task, file='probe.py', line=1, function=function)
    rows = [row('start', 0, 0), row('yield', 10, 5), row('start', 20, 10, task=2), row('return', 40, 30, task=2), row('resume', 100, 90), row('return', 110, 95)]
    completed, clipped, pending = segments(rows)
    assert not clipped and (not pending)
    f = next((x for x in completed if x['task'] == 1))
    assert f['inclusive_wall_ms'] == 110 / 1000000.0 and f['active_wall_ms'] == 20 / 1000000.0 and (f['active_thread_cpu_ms'] == 10 / 1000000.0)
    completed, clipped, pending = segments([row('start', 1, 1), row('start', 2, 2), row('unwind', 3, 3), row('return', 4, 4), row('start', 5, 5)])
    assert len(completed) == 2 and (not clipped) and (len(pending) == 1)

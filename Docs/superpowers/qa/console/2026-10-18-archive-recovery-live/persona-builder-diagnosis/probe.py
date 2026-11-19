"""Read-only source/artifact inspection plus actual pure builder A/B; no app/DB."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import traceback

ROOT = Path(__file__).parent.resolve()
WORKTREE = Path('/Users/macbook-dev/.codex/worktrees/agent-archive-recovery/tldw_chatbook')
LIVE = Path('/private/tmp/archive-recovery-live-20261018/direct-persona')
BASE = '75b671c001e84bf2713c380e30de3e587bcc8ae0'
ARTIFACTS = ['config.toml', 'persona-control.json', 'blocked-initial.json', 'blocked-checkpoints.json', 'data/task7_live/tldw_cli_app.log']
def hashes():
    return {name: hashlib.sha256((LIVE/name).read_bytes()).hexdigest() for name in ARTIFACTS}
before = hashes()
# Imports may read settings; select a disposable byte-for-byte read copy only.
shutil.copyfile(LIVE/'config.toml', ROOT/'config-read-copy.toml')
os.environ['TLDW_CONFIG_PATH'] = str(ROOT/'config-read-copy.toml')
os.environ['TMPDIR'] = str(ROOT)
os.environ['PYTHON_KEYRING_BACKEND'] = 'keyring.backends.null.Keyring'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.dont_write_bytecode = True
network_attempts = []
def deny_network(*args, **kwargs):
    network_attempts.append('forbidden network attempt')
    raise RuntimeError('offline builder diagnostic forbids network')
socket.socket.connect = deny_network
socket.socket.connect_ex = deny_network
socket.create_connection = deny_network
socket.getaddrinfo = deny_network

def audit(event, args):
    if event == 'sqlite3.connect':
        raise RuntimeError('offline builder diagnostic forbids any DB connection')
    if event == 'open':
        path, mode, flags = args
        write = any(c in (mode or '') for c in 'wax+') or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        if write and isinstance(path, (str, bytes, os.PathLike)):
            target = Path(os.fsdecode(path)).resolve()
            if not target.is_relative_to(ROOT):
                raise RuntimeError(f'offline builder diagnostic forbids external writes: {target}')
sys.addaudithook(audit)
sys.path.insert(0, str(WORKTREE))
from tldw_chatbook.Chat.console_prepared_request import build_console_request
from tldw_chatbook.Chat.console_trace_models import FrozenTracePolicy
from tldw_chatbook.Chat.console_trace_provenance import (
    ConsoleRequestRoute,
    ConsoleTraceCaptureMode,
    ProviderArtifactTraceProvenance,
    SavedRevisionTraceProvenance,
    TraceProvenanceSource,
    request_route_provenance,
)

persona = json.loads((LIVE/'persona-control.json').read_text())['persona']
blocked = json.loads((LIVE/'blocked-initial.json').read_text())
user = next(row for row in blocked['source_messages'] if row['role'] == 'user')
revision = blocked['user_revisions'][0]
assert revision['source_message_id'] == user['id']
assert revision['source_conversation_id'] == user['conversation_id']
policy = FrozenTracePolicy('d0cd65e4-777d-49cb-a933-a8fbdf5e5746', 'offline-proof-version', False, None)
saved = SavedRevisionTraceProvenance(revision['revision_id'])
user_row = {'role': 'user', 'content': user['content']}
results = {}
for label, messages, descriptors in [
    ('nonempty_persona_system', [{'role':'system','content':persona['system_prompt']}, user_row], (ProviderArtifactTraceProvenance(TraceProvenanceSource.ACTIVE_REQUEST, policy), saved)),
    ('empty_persona_system_omitted', [user_row], (saved,)),
]:
    try:
        request = build_console_request(
            messages,
            capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
            capture_policy=policy,
            message_provenance=descriptors,
            memory_provenance=(), mandatory_provenance=(), tool_provenance=(),
            metadata_provenance=(request_route_provenance(ConsoleRequestRoute.FRESH),),
        )
        results[label] = {'outcome':'built', 'system_count':len(request.system), 'active_request_count':len(request.active_request)}
    except Exception as exc:
        results[label] = {'outcome':'raised', 'exception_type':type(exc).__name__, 'exception_message':str(exc), 'traceback':traceback.format_exc()}
    print(label, json.dumps(results[label], sort_keys=True))

# Compare the exact offending producer expression and validator against the base.
def source_at(revision, relative):
    if revision == 'HEAD':
        return (WORKTREE/relative).read_text()
    return subprocess.check_output(['git','show',f'{revision}:{relative}'], cwd=WORKTREE, text=True)
def named(tree, name):
    return next(node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name)
producer_path = 'tldw_chatbook/Chat/console_chat_controller.py'
validator_path = 'tldw_chatbook/Chat/console_trace_provenance.py'
baseline = {}
for revision_name in (BASE, 'HEAD'):
    producer_src = source_at(revision_name, producer_path)
    producer_tree = ast.parse(producer_src)
    producer = named(producer_tree, '_build_durable_trace_request')
    assignment = next(node for node in ast.walk(producer) if isinstance(node, ast.Assign) and any(isinstance(t,ast.Name) and t.id == 'descriptors' for t in node.targets))
    validator_src = source_at(revision_name, validator_path)
    validator_tree = ast.parse(validator_src)
    validator = named(validator_tree, 'ConsoleRequestProvenance')
    baseline[revision_name] = {
        'producer_descriptors_line': assignment.lineno,
        'producer_descriptors_source': ast.get_source_segment(producer_src, assignment),
        'producer_descriptors_ast_sha256': hashlib.sha256(ast.dump(assignment,include_attributes=False).encode()).hexdigest(),
        'validator_class_ast_sha256': hashlib.sha256(ast.dump(validator,include_attributes=False).encode()).hexdigest(),
    }
preexisting = all(baseline[BASE][key] == baseline['HEAD'][key] for key in ['producer_descriptors_ast_sha256','validator_class_ast_sha256'])
after = hashes()
result = {'results':results, 'baseline':baseline, 'preexisting_at_base75b':preexisting, 'network_attempts':network_attempts, 'live_artifacts_sha256_before':before, 'live_artifacts_sha256_after':after, 'live_artifacts_unchanged':before==after, 'actual_db_connections':0, 'limits':'Pure request builder reproduction with exact retained Persona text/current saved user revision, not a recovered live exception traceback. No app/controller constructed or native send performed. Frozen policy opaque ID/version synthetic; PII settings match retained live config.'}
(ROOT/'result.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
assert results['nonempty_persona_system']['exception_type'] == 'TraceProvenanceAlignmentError'
assert results['nonempty_persona_system']['exception_message'] == 'trace provenance category mismatch: system'
assert results['empty_persona_system_omitted']['outcome'] == 'built'
assert preexisting and before == after and not network_attempts
print('PREEXISTING_AT_BASE',preexisting)
print('LIVE_ARTIFACTS_UNCHANGED',before == after)
print('NETWORK_ATTEMPTS',len(network_attempts))

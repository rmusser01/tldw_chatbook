"""Exact full configuration checks; no ignored keys or application cache."""
from pathlib import Path
import copy
import hashlib
import json
import tomllib

REFERENCE = Path(__file__).with_name('postsave-current31-config-reference-r4.json')
REFERENCE_SHA256 = '8303e343bbd3c59b4008bffe0476e0114280b47c9425316df05f4ae3bb07d073'

def normalized(value, root):
    if isinstance(value, str): return value.replace(str(root), '<owned-profile>')
    if isinstance(value, list): return [normalized(v, root) for v in value]
    if isinstance(value, dict): return {k: normalized(v, root) for k, v in value.items()}
    return value

def reference():
    assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest() == REFERENCE_SHA256
    return json.loads(REFERENCE.read_text())

def require(values, profile, phase):
    assert phase in ('checked_configuration', 'configuration')
    assert normalized(values, profile) == reference()[phase], 'complete checked configuration differs from frozen reference'

def require_checked(profile):
    from tldw_chatbook.config import load_cli_config_and_ensure_existence
    require(load_cli_config_and_ensure_existence(), profile, 'checked_configuration')

def require_settled(profile):
    require(tomllib.loads((profile/'config.toml').read_text()), profile, 'configuration')

def self_check():
    data = reference()
    for phase in ('checked_configuration', 'configuration'):
        require(data[phase], Path('<owned-profile>'), phase)
        relocated = json.loads(json.dumps(data[phase]).replace('<owned-profile>', '/private/tmp/exact-profile-control'))
        require(relocated, Path('/private/tmp/exact-profile-control'), phase)
        for target in ('selected', 'unselected', 'extra'):
            bad = copy.deepcopy(data[phase])
            if target == 'selected': bad['api_settings']['deepseek']['temperature'] = 0.6
            elif target == 'unselected': bad['providers']['vLLM'] = ['unexpected-reference-control']
            else: bad['unexpected_reference_key'] = True
            try: require(bad, Path('<owned-profile>'), phase)
            except AssertionError: pass
            else: raise AssertionError('complete configuration drift was admitted')
    bad = copy.deepcopy(data['configuration'])
    bad['console']['onboarding']['first_send_completed'] = False
    try: require(bad, Path('<owned-profile>'), 'configuration')
    except AssertionError: pass
    else: raise AssertionError('post-send lifecycle effect was ignored')

if __name__ == '__main__':
    self_check()
    print(json.dumps({'full_phase_maps':2, 'private_path_positive_controls':2, 'drift_refusals':7}))

"""Read-only b110 Python blob overlay; never checks out or edits repository files."""
from importlib import abc,machinery
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

REPO=Path('/Users/macbook-dev/.codex/worktrees/prompt-mix-ins/tldw_chatbook')
ROOT=Path('/private/tmp/mixins-5a8-ui26-diagnosis')
BASE='b1108874cab28d23f660aa22c11c863466a84a8e'
HEAD='5a8f23eb9dfabeda25d46ed168d8ac675f483217'


def prepare():
    names=subprocess.check_output(['git','diff','--name-only',BASE,HEAD,'--','tldw_chatbook'],cwd=REPO,text=True).splitlines()
    entries={}
    for name in names:
        if not name.endswith('.py'):
            continue
        data=subprocess.check_output(['git','show',f'{BASE}:{name}'],cwd=REPO)
        p=ROOT/'b110-python-blobs'/name
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(data)
        module=name[:-3].replace('/','.')
        if module.endswith('.__init__'):
            module=module[:-9]
        entries[module]={'relative_path':name,'blob_path':str(p),'sha256':hashlib.sha256(data).hexdigest()}
    manifest={'base':BASE,'head':HEAD,'scope':'All production Python files differing between b110 and5a8; unchanged production Python and current CSS/resources remain on read-only repository. Exact test body unchanged; only fixture cleanup-hold release added since dev; b110 fixture already identical to current.','modules':entries}
    (ROOT/'b110-overlay-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


class BlobLoader(machinery.SourceFileLoader):
    def __init__(self,name,origin,blob_path):
        super().__init__(name,origin)
        self.blob_path=blob_path
    def get_code(self,fullname):
        return compile(Path(self.blob_path).read_bytes(),self.path,'exec',dont_inherit=True)
    def get_source(self,fullname):
        return Path(self.blob_path).read_text()


class Finder(abc.MetaPathFinder):
    def __init__(self,manifest):
        self.modules=manifest['modules']
    def find_spec(self,fullname,path=None,target=None):
        value=self.modules.get(fullname)
        if value is None:
            return None
        spec=machinery.PathFinder.find_spec(fullname,path)
        assert spec is not None and spec.origin,fullname
        spec.loader=BlobLoader(fullname,spec.origin,value['blob_path'])
        return spec


if __name__=='__main__':
    if '--prepare' in sys.argv:
        print(json.dumps({'modules':len(prepare()['modules'])}))
        raise SystemExit(0)
    manifest=json.loads((ROOT/'b110-overlay-manifest.json').read_text())
    sys.meta_path.insert(0,Finder(manifest))
    sys.path.insert(0,str(REPO))
    import pytest
    plugins=[]
    if '--observe' in sys.argv:
        import profile_observer
        plugins=[profile_observer.Observer()]
    raise SystemExit(pytest.main(['-q','-x','-p','no:cacheprovider','--tb=short','Tests/UI/test_response_rules_end_to_end.py::test_learn_repair_reopen_and_exclude_through_console[ancestry-size1]'],plugins=plugins))

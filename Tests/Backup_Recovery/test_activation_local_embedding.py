"""Explicit local model review uses real offline HF loading, never cache guesses."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

_TINY_MODEL = r"""
import os
from pathlib import Path
from Tests.network_guard import install, blocked_attempts
install()
selector=Path(os.environ['TLDW_CONFIG_PATH']);base=selector.parent.parent;data=base/'data'
selector.write_text('[general]\nusers_name="test"\n[paths]\ndata_dir="'+str(data)+'"\n[rag.service]\nfirst_run_import_done=true\n');selector.chmod(0o600)
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import BertConfig,BertModel,PreTrainedTokenizerFast
model=data/'local-bert';model.mkdir(mode=0o700)
vocab={'[PAD]':0,'[UNK]':1,'[CLS]':2,'[SEP]':3,'[MASK]':4,'local':5,'model':6,'content':7}
tokens=Tokenizer(WordLevel(vocab,unk_token='[UNK]'));tokens.pre_tokenizer=Whitespace()
tokenizer=PreTrainedTokenizerFast(tokenizer_object=tokens,unk_token='[UNK]',pad_token='[PAD]',model_max_length=64)
tokenizer.save_pretrained(model)
torch.manual_seed(7)
tiny=BertModel(BertConfig(vocab_size=len(vocab),hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=24,max_position_embeddings=64))
tiny.save_pretrained(model);del tiny,tokenizer
"""

_LOADER = (
    _TINY_MODEL
    + r"""
import sys
from tldw_chatbook.Embeddings import Embeddings_Lib as library
library._ensure_transformers()
original_tokenizer=library.AutoTokenizer.from_pretrained
original_model=library.AutoModel.from_pretrained
calls=[]
def tokenizer_load(*args,**kwargs):
 assert kwargs.get('local_files_only') is True,'tokenizer can download'
 assert kwargs.get('trust_remote_code') is False
 calls.append('tokenizer');return original_tokenizer(*args,**kwargs)
def model_load(*args,**kwargs):
 assert kwargs.get('local_files_only') is True,'model can download'
 assert kwargs.get('trust_remote_code') is False
 calls.append('model');return original_model(*args,**kwargs)
library.AutoTokenizer.from_pretrained=tokenizer_load
library.AutoModel.from_pretrained=model_load
if sys.argv[1]=='fallback':
 original_to=BertModel.to
 failed=[]
 def to(self,*args,**kwargs):
  if not failed:
   failed.append(True);raise RuntimeError('Cannot copy out of meta tensor')
  return original_to(self,*args,**kwargs)
 BertModel.to=to
cfg=library.HFModelCfg(model_name_or_path=str(model),device='cpu',local_files_only=True)
embedder=library._HuggingFaceEmbedder(cfg)
vectors=embedder.embed(['local model content'])
assert vectors.shape==(1,16)
assert calls.count('model')==(2 if sys.argv[1]=='fallback' else 1)
embedder.close()
assert not blocked_attempts()
print('retired and reopened')
"""
)


@pytest.mark.parametrize("route", ["load", "fallback"])
def test_actual_hf_loaders_keep_local_only(tmp_path, route):
    _run(tmp_path, route, "local", script=_LOADER)


_REVIEW = (
    _TINY_MODEL
    + r"""
import sys, asyncio, time
from tldw_chatbook.Backup_Recovery import bootstrap,storage_admission as storage
from tldw_chatbook.Backup_Recovery.activation import ActivationStore,bind_activation
from tldw_chatbook.Backup_Recovery.control_records import admission_authority,register_pending
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
from tldw_chatbook.RAG_Search import model_recovery as recovery
from tldw_chatbook.RAG_Search.activation import RAGActivationRequired,require_local_model_construction
from tldw_chatbook.RAG_Search.simplified.embeddings_wrapper import EmbeddingsServiceWrapper
from tldw_chatbook.Embeddings.Embeddings_Lib import HFModelCfg,_HuggingFaceEmbedder
root=bootstrap.default_bootstrap_root()
startup=storage._startups.pop((os.getpid(),str(root)),None)
if startup is not None:startup.close()
authority=admission_authority(root);authority.register('profile',(selector.parent,data))
control=base/'operation';control.mkdir(mode=0o700)
owners=('config','rag.definitions','rag.projections','db.rag_indexing','models.artifacts')
register_pending(root,'restore',('profile',),control,(selector,))
with authority.maintenance(('profile',),3) as session:
 bind_activation(root,'restore',selector,'generation',owners,session=session)
(root/('pending-'+bootstrap._key('restore')+'.json')).unlink()
activation=ActivationStore(control/'activation')
config=RAGConfig.from_dict({'embedding':{'model':str(model),'device':'cpu'},'vector_store':{'persist_directory':str(data/'vectors')}})
def denied(call):
 try:call()
 except (OSError,ValueError,RuntimeError):return
 raise AssertionError('unreviewed model effect accepted')
denied(lambda:require_local_model_construction(config))
review=recovery.preview_local_embedding(config)
assert not activation.allowed('generation','models.artifacts')
recovery.approve_local_embedding(config,review.fingerprint)
assert activation.allowed('generation','models.artifacts')
assert not activation.allowed('generation','config')
denied(lambda:EmbeddingsServiceWrapper(str(model),device='cpu'))
for owner in owners:
 if owner!='models.artifacts':activation.approve('generation',owner)
require_local_model_construction(config)
wrapper=EmbeddingsServiceWrapper(str(model),device='cpu')
if sys.argv[1]!='cancel_load':assert wrapper.create_embedding('local model content').shape==(16,)
if sys.argv[1]=='changed':
 original=(model/'config.json').read_bytes();(model/'config.json').write_bytes(original+b' ')
 denied(lambda:wrapper.create_embedding('local model content'))
 denied(lambda:recovery.approve_local_embedding(config,review.fingerprint))
 wrapper.close()
 fresh=recovery.preview_local_embedding(config);assert fresh.fingerprint!=review.fingerprint
 recovery.approve_local_embedding(config,fresh.fingerprint)
 wrapper=EmbeddingsServiceWrapper(str(model),device='cpu')
 assert wrapper.create_embedding('local model content').shape==(16,)
if sys.argv[1] in ('cancel_load','cancel_encode'):
 import threading
 from tldw_chatbook.Embeddings import Embeddings_Lib as library
 library._ensure_transformers()
 entered=threading.Event();release=threading.Event()
 if sys.argv[1]=='cancel_load':
  original=library.AutoModel.from_pretrained
  def blocked(*args,**kwargs):
   entered.set();assert release.wait(10);return original(*args,**kwargs)
  library.AutoModel.from_pretrained=blocked
 else:
  original=BertModel.forward
  def blocked(self,*args,**kwargs):
   entered.set();assert release.wait(10);return original(self,*args,**kwargs)
  BertModel.forward=blocked
 async def cancel_native():
  task=asyncio.create_task(wrapper.create_embedding_async('local model content'))
  for _ in range(500):
   if entered.is_set():break
   await asyncio.sleep(.01)
  assert entered.is_set(),'real native boundary not reached'
  recovery.participant._maintenance_close_admission()
  drain=asyncio.create_task(recovery.participant._maintenance_drain(time.monotonic()+5))
  task.cancel();await asyncio.sleep(.1)
  assert not task.done(),'cancellation abandoned native effect'
  assert not drain.done(),'maintenance passed live native effect'
  assert recovery.participant._borrowers
  release.set()
  try:await task
  except asyncio.CancelledError:pass
  else:raise AssertionError('cancellation not propagated')
  assert await drain
  assert wrapper.factory is None
  await recovery.participant._maintenance_resume()
 asyncio.run(cancel_native())
 assert wrapper.create_embedding('local model content').shape==(16,)
if sys.argv[1]=='receipt':
 receipt=next(activation._generation('generation').glob('local-embedding-*.json'))
 receipt.chmod(0o644)
 denied(lambda:require_local_model_construction(config))
 receipt.chmod(0o600)
 payload=receipt.read_text();receipt.write_text(payload.replace('"version":1','"version":2'))
 denied(lambda:require_local_model_construction(config))
 receipt.write_text(payload)
if sys.argv[1]=='settings':
 config.embedding.device='auto'
 denied(lambda:require_local_model_construction(config))
 denied(lambda:recovery.approve_local_embedding(config,review.fingerprint))
if sys.argv[1]=='alias':
 (model/'linked.json').symlink_to(model/'config.json')
 denied(lambda:recovery.preview_local_embedding(config))
 (model/'linked.json').unlink()
if sys.argv[1]=='bare':
 wrapper.close()
 bare=_HuggingFaceEmbedder(HFModelCfg(model_name_or_path=str(model),device='cpu',local_files_only=True))
 assert bare.embed(['local model content']).shape==(1,16)
 recovery.participant._maintenance_close_admission()
 assert not asyncio.run(recovery.participant._maintenance_drain(time.monotonic()+1))
 bare.close()
 assert asyncio.run(recovery.participant._maintenance_drain(time.monotonic()+1))
 asyncio.run(recovery.participant._maintenance_resume())
if sys.argv[1]=='retire':
 recovery.participant._maintenance_close_admission()
 assert asyncio.run(recovery.participant._maintenance_drain(time.monotonic()+5))
 assert wrapper.factory is None
 asyncio.run(recovery.participant._maintenance_resume())
 assert wrapper.create_embedding('local model content').shape==(16,)
wrapper.close()
assert not recovery.participant._borrowers
assert not blocked_attempts()
print('retired and reopened')
"""
)


@pytest.mark.parametrize(
    "route",
    [
        "approved",
        "changed",
        "retire",
        "cancel_load",
        "cancel_encode",
        "receipt",
        "settings",
        "alias",
        "bare",
    ],
)
def test_explicit_local_model_review_and_retirement(tmp_path, route):
    _run(tmp_path, route, "local", script=_REVIEW)


_ORDINARY = (
    _TINY_MODEL
    + r"""
import asyncio,time,sys
from tldw_chatbook.RAG_Search.simplified.embeddings_wrapper import EmbeddingsServiceWrapper
from tldw_chatbook.RAG_Search import model_recovery as recovery
wrapper=EmbeddingsServiceWrapper(str(model),device='cpu')
assert wrapper.create_embedding('local model content').shape==(16,)
assert recovery.participant._borrowers
recovery.participant._maintenance_close_admission()
if sys.argv[1]=='close_failure':
 record=next(iter(wrapper.factory._cache.values()));original=record['close']
 def failed():raise RuntimeError('actual close failed')
 record['close']=failed
 try:asyncio.run(recovery.participant._maintenance_drain(time.monotonic()+3))
 except RuntimeError as error:assert str(error)=='actual close failed'
 else:raise AssertionError('failed close was accepted')
 assert recovery.participant._borrowers
 record['close']=original
assert asyncio.run(recovery.participant._maintenance_drain(time.monotonic()+3))
assert not recovery.participant._borrowers
asyncio.run(recovery.participant._maintenance_resume())
assert wrapper.create_embedding('local model content').shape==(16,)
wrapper.close()
assert not blocked_attempts()
print('retired and reopened')
"""
)


@pytest.mark.parametrize("route", ["ordinary", "close_failure"])
def test_ordinary_local_hf_native_lifetime(tmp_path, route):
    _run(tmp_path, route, "local", script=_ORDINARY)


_ESCAPE = _REVIEW.replace(
    "review=recovery.preview_local_embedding(config)",
    r"""
import json
if sys.argv[1]=='tokenizer':
 outside=data/'outside-tokenizer.json';outside.write_bytes((model/'tokenizer.json').read_bytes())
 path=model/'tokenizer_config.json';payload=json.loads(path.read_text());payload['tokenizer_file']=str(outside);path.write_text(json.dumps(payload))
else:
 path=model/'model.safetensors.index.json'
 path.write_text(json.dumps({'metadata':{},'weight_map':{'embeddings.word_embeddings.weight':'../outside.safetensors'}}))
denied(lambda:recovery.preview_local_embedding(config))
assert not activation.allowed('generation','models.artifacts')
print('retired and reopened');raise SystemExit(0)
review=recovery.preview_local_embedding(config)
""",
    1,
)


@pytest.mark.parametrize("route", ["tokenizer", "shard"])
def test_review_refuses_model_file_references_outside_closure(tmp_path, route):
    _run(tmp_path, route, "local", script=_ESCAPE)


_DURABLE = _REVIEW.replace(
    "assert not recovery.participant._borrowers\nassert not blocked_attempts()",
    r"""
assert not recovery.participant._borrowers
if sys.argv[1]=='generation':
 control2=base/'operation2';control2.mkdir(mode=0o700)
 register_pending(root,'again',('profile',),control2,(selector,))
 with authority.maintenance(('profile',),3) as session:
  bind_activation(root,'again',selector,'generation2',owners,session=session)
 (root/('pending-'+bootstrap._key('again')+'.json')).unlink()
 second=ActivationStore(control2/'activation')
 for owner in owners:second.approve('generation2',owner)
 denied(lambda:require_local_model_construction(config))
 fresh=recovery.preview_local_embedding(config)
 assert fresh.fingerprint!=review.fingerprint
 recovery.approve_local_embedding(config,fresh.fingerprint)
 wrapper=EmbeddingsServiceWrapper(str(model),device='cpu')
 assert wrapper.create_embedding('local model content').shape==(16,)
 wrapper.close()
assert not blocked_attempts()
""",
    1,
)


@pytest.mark.parametrize("route", ["generation"])
def test_local_model_receipt_is_durable_and_generation_bound(tmp_path, route):
    _run(tmp_path, route, "local", script=_DURABLE)


_FACTORY = _REVIEW.replace(
    "wrapper=EmbeddingsServiceWrapper(str(model),device='cpu')",
    """
from tldw_chatbook.RAG_Search.simplified.rag_factory import create_rag_service
service=create_rag_service(config=config)
assert service.embeddings.get_embedding_dimension()==16
assert service.embeddings.create_embedding('local model content').shape==(16,)
service.close()
wrapper=EmbeddingsServiceWrapper(str(model),device='cpu')
""",
    1,
)


def test_actual_shared_rag_factory_uses_reviewed_local_hf_model(tmp_path):
    _run(tmp_path, "factory", "local", script=_FACTORY)


_ISOLATED_CHILD = r"""
import os,sys
from pathlib import Path
from Tests.network_guard import install,blocked_attempts
install()
from tldw_chatbook.Backup_Recovery.isolated_restore import select_profile,_launch_descriptor
profile,control,mode=sys.argv[1:];control=Path(control)
select_profile(profile,control)
entry=_launch_descriptor(profile,control)
from tldw_chatbook.RAG_Search.simplified.config import RAGConfig
from tldw_chatbook.RAG_Search.simplified.embeddings_wrapper import EmbeddingsServiceWrapper
from tldw_chatbook.RAG_Search import model_recovery as recovery
from tldw_chatbook.Backup_Recovery.activation import ActivationStore
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook import config as installed_config
assert installed_config.CLI_APP_CLIENT_ID==entry.installation_id
model=Path(entry.data)/'local-bert'
config=RAGConfig.from_dict({'embedding':{'model':str(model),'device':'cpu'}})
if mode=='approve':
 review=recovery.preview_local_embedding(config)
 recovery.approve_local_embedding(config,review.fingerprint)
 _,profiles=bootstrap._records(bootstrap.default_bootstrap_root())
 witness=next(p['activation'] for p in profiles if p['selector']==entry.config)
 store=ActivationStore(Path(witness['store_root']))
 for owner in ('config','rag.definitions','rag.projections','db.rag_indexing'):
  store.approve(witness['generation'],owner)
wrapper=EmbeddingsServiceWrapper(str(model),device='cpu')
assert wrapper.create_embedding('local model content').shape==(16,)
wrapper.close()
assert not blocked_attempts()
print('verified isolated local model')
"""

_ISOLATED = (
    _TINY_MODEL
    + r"""
import json,shutil,subprocess,sys
from threading import Event
from Tests.Backup_Recovery.test_restore_plan import sealed
from tldw_chatbook.Backup_Recovery.isolated_restore import restore_isolated
from tldw_chatbook.Backup_Recovery.restore_plan import plan_restore
from tldw_chatbook.Backup_Recovery.profile_catalog import ProfileCatalog
source=base/'archive-source';source.mkdir(mode=0o700)
def config_manifest(doc):
 doc['owners'][0]['owner_id']='config'
 doc['files'][0].update(owner_id='config',logical_id='profile:profile:config',relative_path='config.toml')
 doc['dependency_groups'][0]['members']=['profile:profile:config']
archive=sealed(source,mutate=config_manifest,data=b'[general]\nusers_name="original"\n[rag.service]\nfirst_run_import_done=true\n'.replace(b'\\n',b'\n'))
dest=base/'destinations';dest.mkdir(mode=0o700)
plan=plan_restore(archive,mode='isolated',destinations={'root':dest/'config','profile:profile:paths.data_dir':dest/'data'},target=None,profile_names={'profile':'recovered'})
control=base/'isolated-control'
profile=restore_isolated(archive,plan,control,Event())
selected,data=ProfileCatalog(control).resolve(profile)
shutil.copytree(model,data/'local-bert')
for mode in ('approve','reopen'):
 result=subprocess.run([sys.executable,'-c',CHILD,profile,str(control),mode],env=os.environ.copy(),capture_output=True,text=True,timeout=25)
 assert result.returncode==0,result.stderr[-5000:]
 assert 'verified isolated local model' in result.stdout
assert not blocked_attempts()
print('retired and reopened')
"""
)


def test_finalized_isolated_generation_reopens_local_model_receipt(tmp_path):
    _run(
        tmp_path,
        "isolated",
        "local",
        script="CHILD=" + repr(_ISOLATED_CHILD) + "\n" + _ISOLATED,
        timeout=60,
    )


_COPIED = _REVIEW.replace(
    "if sys.argv[1]=='changed':",
    r"""
async def copied_context():
 with recovery.participant.operation():
  recovery.participant._maintenance_close_admission()
  async def unrelated():
   wrapper.create_embedding('local model content')
  try:
   await asyncio.create_task(unrelated())
  except (OSError,ValueError,RuntimeError):pass
  else:raise AssertionError('copied task acquired model admission')
 await recovery.participant._maintenance_resume()
asyncio.run(copied_context())
if sys.argv[1]=='changed':
""",
    1,
)


def test_copied_context_cannot_admit_unrelated_model_work(tmp_path):
    _run(tmp_path, "copied", "local", script=_COPIED)


_CLOSE_RACE = (
    _TINY_MODEL
    + r"""
import threading,time
from tldw_chatbook.Embeddings.Embeddings_Lib import HFModelCfg,_HuggingFaceEmbedder
from tldw_chatbook.RAG_Search.model_recovery import participant
bare=_HuggingFaceEmbedder(HFModelCfg(model_name_or_path=str(model),device='cpu',local_files_only=True))
entered=threading.Event();release=threading.Event();closing=threading.Event();errors=[]
original=BertModel.forward
def blocked(self,*args,**kwargs):
 entered.set();assert release.wait(10);return original(self,*args,**kwargs)
BertModel.forward=blocked
def encode():
 try:assert bare.embed(['local model content']).shape==(1,16)
 except BaseException as error:errors.append(error)
def close():
 closing.set()
 try:bare.close()
 except BaseException as error:errors.append(error)
worker=threading.Thread(target=encode);worker.start();assert entered.wait(10)
closer=threading.Thread(target=close);closer.start();assert closing.wait(2)
time.sleep(.1)
assert closer.is_alive() and bare in participant._borrowers
release.set();worker.join(10);closer.join(10)
assert not worker.is_alive() and not closer.is_alive() and not errors,errors
assert bare not in participant._borrowers
assert not blocked_attempts()
print('retired and reopened')
"""
)


def test_actual_bare_close_waits_for_its_native_encoder(tmp_path):
    _run(tmp_path, "close", "local", script=_CLOSE_RACE)


def test_actual_runtime_settles_local_hf_before_core_pause_and_reloads(tmp_path):
    from Tests.Backup_Recovery.test_runtime_startup_handoff import _SCRIPT as runtime

    setup = r"""
    from tldw_chatbook.RAG_Search.model_recovery import participant as model_lifetime
    from tldw_chatbook.RAG_Search.simplified.embeddings_wrapper import EmbeddingsServiceWrapper
    wrapper=EmbeddingsServiceWrapper(str(model),device='cpu')
    assert wrapper.create_embedding('local model content').shape==(16,)
    native_entered=threading.Event();native_release=threading.Event()
    original_forward=BertModel.forward
    def blocked_forward(self,*args,**kwargs):
        native_entered.set();assert native_release.wait(10)
        return original_forward(self,*args,**kwargs)
    BertModel.forward=blocked_forward
    accepted=asyncio.create_task(wrapper.create_embedding_async('local model content'))
    while not native_entered.is_set():
        if accepted.done():accepted.result()
        await asyncio.sleep(.001)
    async def finish_native():
        while not model_lifetime._closed:await asyncio.sleep(.001)
        assert storage._pause is None
        assert wrapper.factory is not None
        native_release.set()
        assert (await accepted).shape==(16,)
    finishing=asyncio.create_task(finish_native())
"""
    script = _TINY_MODEL + runtime.replace(
        "    runtime = RuntimeMaintenance(app)",
        setup + "\n    runtime = RuntimeMaintenance(app)",
    ).replace(
        "        runtime.retire_local_caches()",
        "        await finishing\n        assert wrapper.factory is None\n        assert not model_lifetime._borrowers\n        runtime.retire_local_caches()",
    ).replace(
        "        assert not errors\n",
        "        BertModel.forward=original_forward\n        assert wrapper.create_embedding('local model content').shape==(16,)\n        wrapper.close()\n        assert not errors\n",
    )
    _run(tmp_path, "local-model", "resume", script=script, timeout=60)

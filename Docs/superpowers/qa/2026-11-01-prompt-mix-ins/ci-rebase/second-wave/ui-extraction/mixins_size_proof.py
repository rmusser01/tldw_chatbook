import ast,copy,json
from pathlib import Path
r=Path('/Users/macbook-dev/.codex/worktrees/prompt-mix-ins/tldw_chatbook/tldw_chatbook');b=Path('/private/tmp/mixins-size-before-ba1')
def tree(root,f):return ast.parse((root/f).read_text())
def defs(t):return {n.name:n for n in ast.walk(t) if isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef))}
results=[]
def check(sf,tf,names,body=False,renames=None):
 old=defs(tree(b,sf));new=defs(tree(r,tf))
 for name in names:
  target=(renames or {}).get(name,name);a=copy.deepcopy(old[name]);z=copy.deepcopy(new[target])
  if body:a,z=a.body,z.body
  else:
   if isinstance(a,(ast.FunctionDef,ast.AsyncFunctionDef)):
    a.decorator_list=[d for d in a.decorator_list if not(isinstance(d,ast.Name) and d.id=='staticmethod')]
  same=ast.dump(ast.Module(body=a,type_ignores=[]))==ast.dump(ast.Module(body=z,type_ignores=[])) if body else ast.dump(a)==ast.dump(z)
  results.append(dict(source=sf,target=tf,name=name,identical=same,scope='body' if body else 'declaration'))
  assert same,(sf,tf,name)
screen='UI/Screens/personas_screen.py'
models=['_CharacterTTSControlSnapshot','_ActorPackExportUIAuthority','_CharacterTTSAuthorityUnavailable','_CharacterTTSMixedSnapshot','_CharacterVisualIdentityLoadSnapshot','_VisualIdentityPreviewSnapshot','_VisualIdentityAuthorSnapshot','_VisualIdentityAuthoringState','_PersonaVisualAuthorSnapshot','_PersonaSharedVisualIdentityLoadSnapshot','_PersonaSharedVisualIdentityAuthorSnapshot','_PersonaSharedVisualIdentityAuthoringState','_PersonaVisualAuthoringState','_CharacterSaveAuthority','_ActorPackPortraitChoice','_ActorPackCreateSession']
check(screen,'Widgets/Persona_Widgets/personas_state.py',models)
check(screen,'UI/Persona_Modules/personas_preview_coordinator.py',['_DrainedTaskResult','_drain_async','_drain_to_thread','_join_task','_persona_visual_ui_lifetime','_visual_identity_ui_lifetime'])
check(screen,'Widgets/Persona_Widgets/personas_character_tts_widget.py',['_character_tts_availability_map','_character_tts_presentation_from_snapshot','_character_tts_profile_tokens'])
check(screen,'UI/Persona_Modules/character_avatar_helpers.py',['_unbound_persona_shared_visual_identity_metadata','_visual_identity_pack_metadata','_visual_identity_graph_matches_snapshot','_visual_identity_resolution_identity','_visual_identity_direction','_canonical_visual_identity_assets','_actor_pack_portrait_name'])
check(screen,'UI/Persona_Modules/character_avatar_helpers.py',['_eligible_actor_pack_portraits'],body=True)
frame='UI/Persona_Modules/roleplay_frame_state.py';window='UI/Persona_Modules/roleplay_list_window_state.py';old=defs(tree(b,frame));new=defs(tree(r,window));moved=[name for name,n in old.items() if name in new and n.col_offset==0]
check(frame,window,moved)
# Exact original item-list body, except the one local parent class import needed after splitting files.
pane='Widgets/Persona_Widgets/personas_library_pane.py';widget='Widgets/Persona_Widgets/roleplay_item_list.py'
check(pane,widget,['LibraryRow','option_id_for','_prompt_content'])
a=defs(tree(b,pane))['RoleplayItemList'];z=defs(tree(r,widget))['RoleplayItemList'];lookup=next(n for n in z.body if isinstance(n,ast.FunctionDef) and n.name=='_pane');assert isinstance(lookup.body[0],ast.ImportFrom);lookup.body.pop(0)
assert ast.dump(a)==ast.dump(z);results.append(dict(name='RoleplayItemList',identical=True,scope='declaration minus added late parent import'))
class Rename(ast.NodeTransformer):
 def __init__(self,m):self.m=m
 def visit_Name(self,n):
  if n.id in self.m:n.id=self.m[n.id]
  return n
for sf,tf,name,newname,mapping in [
 ('UI/Console_Modules/session.py','UI/Console_Modules/session_prompt_seed.py','_character_session_prompt_seed','_character_session_prompt_seed',{'seed_type':'CharacterSessionPromptSeed'}),
 ('UI/Console_Modules/session.py','UI/Console_Modules/session_prompt_seed.py','_persona_session_prompt_seed','_persona_session_prompt_seed',{'seed_type':'PersonaSessionPromptSeed'}),
 ('UI/Console_Modules/workspace.py','Workspaces/display_state.py','_normalized_console_workspace_id','normalized_console_workspace_id',{'global_workspace_id':'CONSOLE_GLOBAL_WORKSPACE_ID'}),
 ('Widgets/Console/console_transcript.py','Widgets/Console/console_selection_geometry.py','_body_wrap_table','build_body_wrap_table',{'content_type':'Content'})]:
 a=defs(tree(b,sf))[name];z=Rename(mapping).visit(defs(tree(r,tf))[newname]);assert ast.dump(ast.Module(body=a.body,type_ignores=[]))==ast.dump(ast.Module(body=z.body,type_ignores=[])),name
 results.append(dict(name=name,identical=True,scope='body with explicit dependency identifier mapping',mapping=mapping))
# Item constants retain literal values, including mutable singleton dicts reexported by reference.
old=tree(b,frame);new=tree(r,window)
def values(t):
 out={}
 for n in t.body:
  if isinstance(n,ast.Assign):
   for target in n.targets:
    if isinstance(target,ast.Name):out[target.id]=n.value
  elif isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name):out[n.target.id]=n.value
 return out
a,z=values(old),values(new);keys=sorted(set(a)&set(z));assert all(ast.dump(a[k])==ast.dump(z[k]) for k in keys)
print('Verified moved declarations/bodies:',len(results));print('Verified shared literal/value declarations:',len(keys));print('All matched with only documented explicit-dependency normalization and late parent import.')
Path('/private/tmp/mixins-size-body-proof.json').write_text(json.dumps({'definitions':results,'constants':keys},indent=2))

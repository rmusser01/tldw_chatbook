exec(open('/private/tmp/pr2995-task6/prove_integration.py').read().split('a,b,c=')[0])
proof=json.loads((O/'task-6-preservation.json').read_text());out=[]
for row in proof['owned_python']:
 if row['strict_ast_equal']:continue
 p=row['path']; fs=[]
 for ref,tag in [(old,'old'),(base,'base'),(dev,'dev')]:
  f=pathlib.Path('/private/tmp/pr2995-task6')/('canonical-'+p.replace('/','--')+'.'+tag);f.write_text(ast.unparse(ast.parse(blob(ref,p)))+'\n');fs.append(str(f))
 merged=subprocess.run(['git','merge-file','-p',*fs],capture_output=True);out.append({'path':p,'merge_exit':merged.returncode,'exact_combined_ast':merged.returncode==0 and asts(merged.stdout)==asts(blob('HEAD',p))})
(O/'task-6-semantic-combination.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

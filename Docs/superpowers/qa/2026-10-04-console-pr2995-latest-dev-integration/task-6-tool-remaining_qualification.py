exec(open('/private/tmp/pr2995-task6/qualify.py').read().split('for label,args in groups:')[0])
for label,args in groups[5:]:
 print('START '+label,flush=True);p=subprocess.run([py,run,label,*args]);print('END '+label+' exit='+str(p.returncode),flush=True)

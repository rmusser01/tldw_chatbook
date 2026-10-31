import subprocess
argv=["ps","-axo","pid,args"]
r=subprocess.run(argv,capture_output=True,text=True)
assert r.returncode==0,r.stderr
owned=[line for line in r.stdout.splitlines() if "-m pytest" in line and "--basetemp=/private/tmp/console-finalfix-" in line]
print("Process snapshot argv:",repr(argv),"returncode:",r.returncode)
print("Exact finalfix pytest basetemp matches:",len(owned))
for line in owned: print(line)
assert not owned,"An exact owned final-fix pytest process remains active."

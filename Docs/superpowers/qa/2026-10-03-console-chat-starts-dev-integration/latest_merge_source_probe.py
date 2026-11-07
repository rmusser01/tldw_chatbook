from pathlib import Path
import difflib,hashlib,json,subprocess
head="149adb53f1d10b090e31a5f458e8c2ffcd3ca458"
feature="df81b3e65c2c2a0b221c85962658acc96fc567d2"
dev="9b28ce1479efed6bca687cfbb33d261322f83abe"
scratch=Path(__file__).resolve().parent
calls=[]
def git(*args):
    argv=["git",*args];result=subprocess.run(argv,capture_output=True,text=True)
    calls.append({"argv":argv,"cwd":str(Path.cwd()),"returncode":result.returncode,"output":result.stdout+result.stderr})
    assert result.returncode==0,calls[-1]
    return result.stdout
assert git("rev-parse","HEAD").strip()==head
assert git("show","--format=%P","--no-patch",head).strip().split()==[feature,dev]
assert not git("status","--porcelain")
path="tldw_chatbook/UI/Screens/chat_screen.py"
before=git("show",feature+":"+path);after=git("show",head+":"+path)
line="                spend.console_rate_limit_line(provider_key),\n"
assert line not in before
assert after.replace(line,"",1)==before
lesson="backlog/docs/lessons-testing-evidence.md"
old=git("show",feature+":"+lesson).splitlines(True)
new=git("show",head+":"+lesson).splitlines(True)
ops=difflib.SequenceMatcher(a=old,b=new,autojunk=False).get_opcodes()
assert all(tag in {"equal","insert"} for tag,*_ in ops)
paths=git("diff","--name-only",feature,head,"--","tldw_chatbook","Tests").splitlines()
hashes=[]
for p in paths:
    data=Path(p).read_bytes()
    assert data.decode()==git("show",head+":"+p)
    hashes.append({"path":p,"sha256":hashlib.sha256(data).hexdigest()})
(scratch/"latest-merge-source-detail.json").write_text(json.dumps({"head":head,"parents":[feature,dev],"screen_added_line":line,"lesson_opcodes":ops,"source_hashes":hashes,"calls":calls},indent=2))
print(f"Merge HEAD{head}: exact parents; sole chat_screen insertion is rate-limit argument; lesson merge preserves all prior lines; {len(hashes)} upstream source/test files match committed bytes; tracked clean.")

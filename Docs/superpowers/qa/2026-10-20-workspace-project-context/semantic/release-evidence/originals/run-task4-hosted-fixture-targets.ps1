param(
    [Parameter(Mandatory = $true)][string]$RunName,
    [Parameter(Mandatory = $true)][string[]]$Selectors
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../../..')).Path
$profile = Join-Path $PSScriptRoot "native-temp/$RunName"
$receipts = Join-Path $PSScriptRoot 'task2-receipts'
$pre = Join-Path $receipts "$RunName-pre.json"
$post = Join-Path $receipts "$RunName-post.json"
$raw = Join-Path $receipts "$RunName-parent.log"
$xml = Join-Path $receipts "$RunName-parent.xml"
foreach ($path in @($profile, $pre, $post, $raw, $xml)) {
    if (Test-Path -LiteralPath $path) { throw "fixture_run_name_already_exists: $path" }
}
$python = 'C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'core_runtime_unavailable' }
Set-Location -LiteralPath $repo
$hashPaths = @(
    'Tests/Workspaces/test_project_index_service.py',
    'tldw_chatbook/Workspaces/project_index/runtime.py',
    'tldw_chatbook/Utils/private_paths.py',
    '.github/workflows/derived-artifacts.yml'
) + @($Selectors | ForEach-Object { ($_ -split '::', 2)[0] } | Sort-Object -Unique)
function Get-Hashes {
    @($hashPaths | Sort-Object -Unique | ForEach-Object {
        @{ path = $_; sha256 = (Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash }
    })
}
$before = Get-Hashes
New-Item -ItemType Directory -Path $profile | Out-Null
$env:PYTHONPATH = (Join-Path $repo 'packages/tldw_profile_core/src')
$env:TEMP = $profile
$env:TMP = $profile
$env:TMPDIR = $profile
$env:TLDW_TEST_CONFIG_ROOT = $profile
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
Remove-Item Env:TLDW_TEST_OFFLINE_EMBEDDING_MODEL -ErrorAction SilentlyContinue
$env:TLDW_TEST_SELECTORS_JSON = ConvertTo-Json -InputObject @($Selectors) -Compress
$env:TLDW_PARENT_XML = $xml
$script = 'import os,json,sqlite3,importlib.util,sys; from tldw_chatbook.Workspaces.project_index.allocation import _SOURCE_ID; c=sqlite3.connect(":memory:"); source=c.execute("SELECT sqlite_source_id()").fetchone()[0]; options={row[0] for row in c.execute("PRAGMA compile_options")}; print("PYTHON",sys.version,sys.executable,"SQLite",sqlite3.sqlite_version,source,sorted(options),"sentence_transformers_present",importlib.util.find_spec("sentence_transformers") is not None,flush=True); assert sqlite3.sqlite_version=="3.49.1" and source==_SOURCE_ID and {"TEMP_STORE=1","MAX_PAGE_SIZE=65536","THREADSAFE=1"}<=options and importlib.util.find_spec("sentence_transformers") is None; import pytest; raise SystemExit(pytest.main([*json.loads(os.environ["TLDW_TEST_SELECTORS_JSON"]),"-q","--timeout=180","--tb=short","--durations=0","--no-showlocals","--junitxml="+os.environ["TLDW_PARENT_XML"]]))'
@{run=$RunName;cwd=$repo;argv=@($python,'-c',$script);selectors=$Selectors;hashes=$before;model_selected=$false} |
    ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $pre -Encoding utf8
& $python -c $script 2>&1 | Tee-Object -FilePath $raw
$code = $LASTEXITCODE
Get-ChildItem -LiteralPath $profile -Recurse -File -Include pytest.log,pytest.xml -ErrorAction SilentlyContinue | ForEach-Object {
    $case = $_.Directory.Parent.Name
    $suffix = if ($_.Name -eq 'pytest.log') { 'child.log' } else { 'child.xml' }
    Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $receipts "$RunName-$case-$suffix")
}
$after = Get-Hashes
$equal = (($before | ConvertTo-Json -Depth 6 -Compress) -ceq ($after | ConvertTo-Json -Depth 6 -Compress))
@{run=$RunName;exit_code=$code;hashes=$after;source_unchanged=$equal;parent_log=$raw;parent_xml=$xml} |
    ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $post -Encoding utf8
if ($code -ne 0 -or -not $equal) { exit 1 }

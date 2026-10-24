param(
    [Parameter(Mandatory = $true)][string]$RunName,
    [Parameter(Mandatory = $true)][string[]]$Selectors
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../../..')).Path
$profile = Join-Path $PSScriptRoot "native-temp/$RunName"
$receipts = Join-Path $PSScriptRoot 'task2-receipts'
$manifest = Join-Path $receipts "$RunName-pre.json"
$raw = Join-Path $receipts "$RunName-parent.log"
if ((Test-Path -LiteralPath $profile) -or (Test-Path -LiteralPath $manifest) -or (Test-Path -LiteralPath $raw)) {
    throw 'targeted_run_name_already_exists'
}
$shared = 'C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Lib/site-packages'
$python = Join-Path $PSScriptRoot 'qualification-venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $shared) -or -not (Test-Path -LiteralPath $python)) {
    throw 'qualification_runtime_unavailable'
}
New-Item -ItemType Directory -Path $profile | Out-Null
Set-Location -LiteralPath $repo
$env:TLDW_TEST_SHARED_SITE = $shared
$env:PYTHONPATH = (@(
    (Join-Path $repo 'packages/tldw_profile_core/src'),
    $shared,
    (Join-Path $shared 'win32'),
    (Join-Path $shared 'win32/lib'),
    (Join-Path $shared 'pythonwin'),
    (Join-Path $shared 'pywin32_system32')
) -join ';')
$env:TEMP = $profile
$env:TMP = $profile
$env:TMPDIR = $profile
$env:TLDW_TEST_CONFIG_ROOT = $profile
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_CACHE = 'C:/Users/GDesktop-1/.cache/huggingface/hub'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:TLDW_TEST_OFFLINE_EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
$env:TLDW_TEST_SELECTORS_JSON = ConvertTo-Json -InputObject @($Selectors) -Compress
$sourcePaths = @(
    'tldw_chatbook/Embeddings/Embeddings_Lib.py',
    'tldw_chatbook/RAG_Search/projection_scope.py',
    'tldw_chatbook/RAG_Search/simplified/embeddings_wrapper.py',
    'tldw_chatbook/Workspaces/project_index/semantic.py',
    'tldw_chatbook/Workspaces/project_index/service.py'
) + @($Selectors | ForEach-Object { ($_ -split '::', 2)[0] } | Sort-Object -Unique)
$hashes = @($sourcePaths | Sort-Object -Unique | Where-Object { Test-Path -LiteralPath $_ } | ForEach-Object {
    @{ path = $_; sha256 = (Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash }
})
$script = 'import os,site,json; site.addsitedir(os.environ["TLDW_TEST_SHARED_SITE"]); import pytest; raise SystemExit(pytest.main([*json.loads(os.environ["TLDW_TEST_SELECTORS_JSON"]),"-q","--tb=short","--show-capture=no","--no-showlocals"]))'
@{
    cwd = $repo
    argv = @($python, '-c', $script)
    selectors = $Selectors
    shared_site = $shared
    hf_hub_cache = $env:HF_HUB_CACHE
    offline_model = $env:TLDW_TEST_OFFLINE_EMBEDDING_MODEL
    hashes = $hashes
} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifest -Encoding utf8
& $python -c $script 2>&1 | Tee-Object -FilePath $raw
$code = $LASTEXITCODE
Get-ChildItem -LiteralPath $profile -Recurse -File -Include pytest.log,pytest.xml -ErrorAction SilentlyContinue | ForEach-Object {
    $case = $_.Directory.Parent.Name
    $suffix = if ($_.Name -eq 'pytest.log') { 'child.log' } else { 'child.xml' }
    Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $receipts "$RunName-$case-$suffix")
}
exit $code

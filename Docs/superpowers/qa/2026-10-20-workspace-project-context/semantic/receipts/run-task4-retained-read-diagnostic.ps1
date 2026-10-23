param([Parameter(Mandatory = $true)][string]$RunName)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../../..')).Path
$stable = (Resolve-Path -LiteralPath (Join-Path $repo '.superpowers/k5-scale/stable-profile')).Path
$tempRoot = Join-Path $PSScriptRoot "native-temp/$RunName"
$manifest = Join-Path $PSScriptRoot "$RunName-pre.json"
$raw = Join-Path $PSScriptRoot "$RunName-raw.log"
$python = Join-Path $PSScriptRoot 'qualification-venv/Scripts/python.exe'
$script = Join-Path $PSScriptRoot 'task4_retained_read_diagnostic.py'
$shared = 'C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Lib/site-packages'
if ((Test-Path -LiteralPath $tempRoot) -or (Test-Path -LiteralPath $manifest) -or (Test-Path -LiteralPath $raw)) { throw 'read_diagnostic_run_name_already_exists' }
foreach ($path in @($stable, $python, $script, $shared)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "read_diagnostic_setup_missing: $path" }
}
New-Item -ItemType Directory -Path $tempRoot | Out-Null
Set-Location -LiteralPath $repo
$env:PYTHONPATH = (@(
    (Join-Path $repo 'packages/tldw_profile_core/src'),
    $shared,
    (Join-Path $shared 'win32'),
    (Join-Path $shared 'win32/lib'),
    (Join-Path $shared 'pythonwin'),
    (Join-Path $shared 'pywin32_system32')
) -join ';')
$env:TLDW_TEST_SHARED_SITE = $shared
$env:TLDW_TEST_CONFIG_ROOT = $stable
$env:TEMP = $tempRoot
$env:TMP = $tempRoot
$env:TMPDIR = $tempRoot
$env:PYTHONIOENCODING = 'utf-8'
$paths = @(
    '.superpowers/sdd/2026-10-20-workspace-project-semantic-index/task4_retained_read_diagnostic.py',
    'tldw_chatbook/Workspaces/project_index/service.py',
    'tldw_chatbook/Workspaces/project_index/allocation.py',
    'tldw_chatbook/DB/Workspace_Project_Index_DB.py'
)
$hashes = @($paths | ForEach-Object { @{ path = $_; sha256 = (Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash } })
@{ cwd = $repo; argv = @($python, '-c', 'site.addsitedir(shared);runpy.run_path(script,run_name="__main__")'); runner = $script; profile = $stable; temp = $tempRoot; hashes = $hashes } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifest -Encoding utf8
$bootstrap = 'import os,site,runpy; site.addsitedir(os.environ["TLDW_TEST_SHARED_SITE"]); runpy.run_path(os.path.join(os.getcwd(), ".superpowers/sdd/2026-10-20-workspace-project-semantic-index/task4_retained_read_diagnostic.py"), run_name="__main__")'
& $python -c $bootstrap 2>&1 | Tee-Object -FilePath $raw
exit $LASTEXITCODE

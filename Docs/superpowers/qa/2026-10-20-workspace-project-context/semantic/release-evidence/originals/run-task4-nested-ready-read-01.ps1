$ErrorActionPreference = 'Stop'
$runName = 'task4-nested-ready-read-01'
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../../..')).Path
$stable = (Resolve-Path -LiteralPath (Join-Path $repo '.superpowers/k5-scale/stable-profile')).Path
$launch = Get-Content -LiteralPath (Join-Path $repo '.superpowers/k5-scale/nested-launch.json') -Raw | ConvertFrom-Json
$group = [string]$launch.cache
$python = Join-Path $PSScriptRoot 'qualification-venv/Scripts/python.exe'
$script = Join-Path $PSScriptRoot 'task4_nested_ready_read_01.py'
$shared = 'C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Lib/site-packages'
$tempRoot = Join-Path $PSScriptRoot "native-temp/$runName"
$pre = Join-Path $PSScriptRoot "$runName-pre.json"
$post = Join-Path $PSScriptRoot "$runName-post.json"
$stdout = Join-Path $PSScriptRoot "$runName-stdout.log"
$stderr = Join-Path $PSScriptRoot "$runName-stderr.log"
$childResult = Join-Path $PSScriptRoot "$runName-child.json"
$resultPath = Join-Path $PSScriptRoot "$runName-result.json"
foreach ($path in @($python, $script, $shared, $group, $stable)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "read_setup_missing: $path" }
}
foreach ($path in @($tempRoot, $pre, $post, $stdout, $stderr, $childResult, $resultPath)) {
    if (Test-Path -LiteralPath $path) { throw "read_run_name_already_exists: $path" }
}
function Get-GroupInventory {
    param([string]$Root)
    $files = @(Get-ChildItem -LiteralPath $Root -File -Recurse -Force | Sort-Object FullName)
    return @($files | ForEach-Object {
        [ordered]@{
            relative = [IO.Path]::GetRelativePath($Root, $_.FullName)
            bytes = $_.Length
            sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
        }
    })
}
function Get-SourceHashes {
    $paths = @(
        'tldw_chatbook/Workspaces/project_index/service.py',
        'tldw_chatbook/DB/Workspace_Project_Index_DB.py',
        'tldw_chatbook/Workspaces/project_index/runtime.py',
        'tldw_chatbook/Workspaces/project_index/models.py',
        '.superpowers/sdd/2026-10-20-workspace-project-semantic-index/task4_nested_ready_read_01.py',
        '.superpowers/sdd/2026-10-20-workspace-project-semantic-index/run-task4-nested-ready-read-01.ps1'
    )
    return @($paths | ForEach-Object { [ordered]@{ path = $_; sha256 = (Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash } })
}
Set-Location -LiteralPath $repo
$before = [ordered]@{
    run = $runName
    group = $group
    inventory = @(Get-GroupInventory -Root $group)
    source_hashes = @(Get-SourceHashes)
    started_utc = [DateTime]::UtcNow.ToString('o')
}
$before | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath $pre -Encoding utf8
New-Item -ItemType Directory -Path $tempRoot | Out-Null
$env:PYTHONPATH = (@(
    $repo, (Join-Path $repo 'packages/tldw_profile_core/src'), $shared,
    (Join-Path $shared 'win32'), (Join-Path $shared 'win32/lib'),
    (Join-Path $shared 'pythonwin'), (Join-Path $shared 'pywin32_system32')
) -join ';')
$env:TLDW_TEST_SHARED_SITE = $shared
$env:TLDW_TEST_CONFIG_ROOT = $stable
$env:TEMP = $tempRoot
$env:TMP = $tempRoot
$env:TMPDIR = $tempRoot
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$bootstrap = 'import os,site,runpy; site.addsitedir(os.environ["TLDW_TEST_SHARED_SITE"]); runpy.run_path(os.path.join(os.getcwd(), ".superpowers/sdd/2026-10-20-workspace-project-semantic-index/task4_nested_ready_read_01.py"), run_name="__main__")'
$process = Start-Process -FilePath $python -ArgumentList @('-c', ('"' + $bootstrap.Replace('"', '\"') + '"')) -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
$startUtc = $process.StartTime.ToUniversalTime().ToString('o')
$retired = $process.WaitForExit(180000)
$forced = $false
if (-not $retired) {
    $current = Get-CimInstance Win32_Process -Filter "ProcessId=$($process.Id)"
    $samePath = $current -and ([string]$current.ExecutablePath).Equals($python, [StringComparison]::OrdinalIgnoreCase)
    $sameStart = $current -and (([Management.ManagementDateTimeConverter]::ToDateTime($current.CreationDate)).ToUniversalTime().ToString('o') -eq $startUtc)
    if (-not ($samePath -and $sameStart)) { throw 'timeout_process_identity_uncertain_refuse_kill' }
    Stop-Process -Id $process.Id -Force
    $process.WaitForExit(10000) | Out-Null
    $forced = $true
}
$after = [ordered]@{
    run = $runName
    group = $group
    inventory = @(Get-GroupInventory -Root $group)
    source_hashes = @(Get-SourceHashes)
    finished_utc = [DateTime]::UtcNow.ToString('o')
}
$after | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath $post -Encoding utf8
$beforeInventory = ConvertTo-Json -InputObject $before.inventory -Depth 5 -Compress
$afterInventory = ConvertTo-Json -InputObject $after.inventory -Depth 5 -Compress
$beforeSources = ConvertTo-Json -InputObject $before.source_hashes -Depth 5 -Compress
$afterSources = ConvertTo-Json -InputObject $after.source_hashes -Depth 5 -Compress
$result = [ordered]@{
    run = $runName
    pid = $process.Id
    process_started_utc = $startUtc
    exit_code = $process.ExitCode
    forced = $forced
    cache_group_unchanged = ($beforeInventory -ceq $afterInventory)
    source_unchanged = ($beforeSources -ceq $afterSources)
    child_result_present = (Test-Path -LiteralPath $childResult)
    before = $pre
    after = $post
    stdout = $stdout
    stderr = $stderr
    child_result = $childResult
}
$result | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $resultPath -Encoding utf8
$result | ConvertTo-Json -Depth 5
if ($forced -or $process.ExitCode -ne 0 -or -not $result.cache_group_unchanged -or -not $result.source_unchanged -or -not $result.child_result_present) { exit 1 }

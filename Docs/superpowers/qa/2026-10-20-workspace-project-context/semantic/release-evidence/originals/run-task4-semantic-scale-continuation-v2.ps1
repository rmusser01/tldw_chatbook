param([Parameter(Mandatory = $true)][string]$RunName)
$ErrorActionPreference = 'Stop'

if ($RunName -notmatch '^task4-semantic-scale-continuation-[a-z0-9-]+$') {
    throw 'invalid_scale_run_name'
}
$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../../..')).Path
$stable = (Resolve-Path -LiteralPath (Join-Path $repo '.superpowers/k5-scale/stable-profile')).Path
$python = Join-Path $PSScriptRoot 'qualification-venv/Scripts/python.exe'
$script = Join-Path $PSScriptRoot 'task4_semantic_scale_continuation_v2.py'
$shared = 'C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Lib/site-packages'
$tempRoot = Join-Path $PSScriptRoot "native-temp/$RunName"
$pre = Join-Path $PSScriptRoot "$RunName-pre.json"
$post = Join-Path $PSScriptRoot "$RunName-post.json"
$result = Join-Path $PSScriptRoot "$RunName-result.json"
$stdout = Join-Path $PSScriptRoot "$RunName-stdout.log"
$stderr = Join-Path $PSScriptRoot "$RunName-stderr.log"
$wholeSeconds = 9600

# This launcher is prepared only. The parent must separately authorize its run
# and execute the established exact native-temp trust preflight before launch.
if (Get-ChildItem -LiteralPath $PSScriptRoot -Filter "$RunName*" | Select-Object -First 1) {
    throw 'scale_run_name_already_exists'
}
if (Test-Path -LiteralPath $tempRoot) { throw 'scale_native_temp_already_exists' }
foreach ($path in @($stable, $python, $script, $shared)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "scale_setup_missing: $path" }
}
$paths = @(
    '.superpowers/sdd/2026-10-20-workspace-project-semantic-index/task4_semantic_scale_continuation_v2.py',
    '.superpowers/sdd/2026-10-20-workspace-project-semantic-index/run-task4-semantic-scale-continuation-v2.ps1',
    '.superpowers/sdd/2026-10-20-workspace-project-semantic-index/task4-semantic-scale-final-01-nested-report.json',
    '.superpowers/sdd/2026-10-20-workspace-project-semantic-index/task4-nested-ready-read-01-child.json',
    'tldw_chatbook/Embeddings/Embeddings_Lib.py',
    'tldw_chatbook/RAG_Search/projection_scope.py',
    'tldw_chatbook/RAG_Search/simplified/embeddings_wrapper.py',
    'tldw_chatbook/Workspaces/project_index/semantic.py',
    'tldw_chatbook/Workspaces/project_index/service.py',
    'tldw_chatbook/Workspaces/project_index/runtime.py',
    'tldw_chatbook/DB/Workspace_Project_Index_DB.py',
    'tldw_chatbook/Widgets/Settings_Widgets/project_index_binding_card.py',
    'Tests/Agents/test_project_index_tool.py',
    'Tests/Workspaces/test_project_index_service.py',
    'Tests/Workspaces/test_project_semantic_app_quality.py'
)
Set-Location -LiteralPath $repo
foreach ($path in $paths) {
    if (-not (Test-Path -LiteralPath $path)) { throw "scale_source_missing: $path" }
}
function Get-SourceHashes {
    @($paths | ForEach-Object {
        @{ path = $_; sha256 = (Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash }
    })
}
function Get-CommandHash([string]$CommandLine) {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        [BitConverter]::ToString(
            $sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($CommandLine))
        ).Replace('-', '')
    }
    finally { $sha.Dispose() }
}
function Get-TreeSnapshot([int]$RootPid) {
    $all = @(Get-CimInstance Win32_Process)
    $rootProcess = $all | Where-Object ProcessId -eq $RootPid | Select-Object -First 1
    if ($null -eq $rootProcess) { return @() }
    $seen = @{}
    $createdByPid = @{}
    $queue = [System.Collections.Generic.Queue[int]]::new()
    $queue.Enqueue($RootPid)
    while ($queue.Count -gt 0) {
        $current = $queue.Dequeue()
        $process = $all | Where-Object ProcessId -eq $current | Select-Object -First 1
        if ($null -eq $process -or $null -eq $process.CreationDate -or
            [string]::IsNullOrWhiteSpace($process.ExecutablePath) -or
            [string]::IsNullOrWhiteSpace($process.CommandLine)) {
            throw 'scale_tree_identity_incomplete'
        }
        $created = $process.CreationDate.ToUniversalTime()
        if ($current -ne $RootPid) {
            $parent = $seen[[int]$process.ParentProcessId]
            if ($null -eq $parent -or $created -lt $createdByPid[[int]$process.ParentProcessId]) {
                throw 'scale_tree_parent_identity_unverified'
            }
        }
        $createdByPid[$current] = $created
        $seen[$current] = [pscustomobject]@{
            pid = [int]$process.ProcessId
            parent_pid = [int]$process.ParentProcessId
            created_utc = $created.ToString('o')
            executable_path = [string]$process.ExecutablePath
            command_sha256 = Get-CommandHash ([string]$process.CommandLine)
        }
        foreach ($childProcess in ($all | Where-Object ParentProcessId -eq $current)) {
            $queue.Enqueue([int]$childProcess.ProcessId)
        }
    }
    @($seen.Values | Sort-Object pid)
}
$before = Get-SourceHashes
New-Item -ItemType Directory -Path $tempRoot | Out-Null
$env:PYTHONPATH = (@(
    $repo,
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
$env:HF_HUB_CACHE = 'C:/Users/GDesktop-1/.cache/huggingface/hub'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:TLDW_TEST_OFFLINE_EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
$env:TLDW_SCALE_RUN_NAME = $RunName
@{
    cwd = $repo
    argv = @($python, $script)
    profile = $stable
    temp = $tempRoot
    model = $env:TLDW_TEST_OFFLINE_EMBEDDING_MODEL
    whole_process_bound_seconds = $wholeSeconds
    per_shape_build_bound_seconds = 3600
    hashes = $before
} | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath $pre -Encoding utf8

$timedOut = $false
$forcedExit = $false
$exitCode = $null
$pidValue = $null
$startedAt = $null
$launchIdentity = $null
$launchRecord = $null
$timeoutTreeBefore = @()
$timeoutTreeRechecked = @()
$timeoutTreeSurvivors = @()
$failure = $null
try {
    $child = Start-Process -FilePath $python -ArgumentList @($script) -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $pidValue = $child.Id
    $launchIdentity = Get-CimInstance Win32_Process -Filter "ProcessId = $pidValue"
    if ($null -ne $launchIdentity) {
        if ($launchIdentity.ExecutablePath -ine $python -or
            $launchIdentity.CommandLine -notlike '*task4_semantic_scale_continuation_v2.py*' -or
            [math]::Abs(($launchIdentity.CreationDate.ToUniversalTime() - $child.StartTime.ToUniversalTime()).TotalSeconds) -gt 1) {
            throw 'scale_launch_identity_unverified'
        }
        $startedAt = $launchIdentity.CreationDate.ToUniversalTime().ToString('o')
        $launchRecord = @{
            pid = $pidValue
            created_utc = $startedAt
            executable_path = [string]$launchIdentity.ExecutablePath
            command_sha256 = Get-CommandHash ([string]$launchIdentity.CommandLine)
        }
    }
    else {
        $startedAt = $child.StartTime.ToUniversalTime().ToString('o')
    }
    if (-not $child.WaitForExit($wholeSeconds * 1000)) {
        $timedOut = $true
        $actual = Get-CimInstance Win32_Process -Filter "ProcessId = $pidValue"
        if ($null -eq $launchIdentity -or $null -eq $actual -or
            $actual.ExecutablePath -ine $python -or
            $actual.CommandLine -notlike '*task4_semantic_scale_continuation_v2.py*' -or
            $actual.CreationDate.ToUniversalTime().ToString('o') -cne $startedAt -or
            (Get-CommandHash ([string]$actual.CommandLine)) -cne
                (Get-CommandHash ([string]$launchIdentity.CommandLine))) {
            throw 'scale_timeout_pid_identity_unverified'
        }
        $timeoutTreeBefore = @(Get-TreeSnapshot $pidValue)
        $timeoutTreeRechecked = @(Get-TreeSnapshot $pidValue)
        $rootRechecked = $timeoutTreeRechecked | Where-Object pid -eq $pidValue | Select-Object -First 1
        if ($timeoutTreeBefore.Count -eq 0 -or
            ($timeoutTreeBefore | ConvertTo-Json -Depth 5 -Compress) -cne
                ($timeoutTreeRechecked | ConvertTo-Json -Depth 5 -Compress) -or
            $null -eq $rootRechecked -or $rootRechecked.created_utc -cne $startedAt -or
            $rootRechecked.command_sha256 -cne $launchRecord.command_sha256) {
            throw 'scale_timeout_tree_identity_unverified'
        }
        # Exact twice-inspected task-owned PID tree only. Forced exit is adverse.
        & taskkill.exe /PID $pidValue /T /F | Out-Null
        if ($LASTEXITCODE -ne 0) { throw 'scale_timeout_tree_abort_failed' }
        $forcedExit = $true
        $child.WaitForExit(30000) | Out-Null
        $timeoutTreeSurvivors = @($timeoutTreeRechecked | Where-Object {
            $live = Get-CimInstance Win32_Process -Filter "ProcessId = $($_.pid)"
            $null -ne $live -and $null -ne $live.CreationDate -and
                $live.CreationDate.ToUniversalTime().ToString('o') -ceq $_.created_utc
        })
    }
    $exitCode = $child.ExitCode
}
catch {
    $failure = $_.Exception.GetType().Name
    throw
}
finally {
    $after = Get-SourceHashes
    $same = (($before | ConvertTo-Json -Depth 6 -Compress) -ceq ($after | ConvertTo-Json -Depth 6 -Compress))
    @{
        run = $RunName
        hashes = $after
        same_source_and_harness = $same
    } | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath $post -Encoding utf8
    @{
        run = $RunName
        child_pid = $pidValue
        child_started_utc = $startedAt
        child_launch_identity = $launchRecord
        child_exit_code = $exitCode
        whole_process_timed_out = $timedOut
        forced_exit = $forcedExit
        timeout_tree_before = $timeoutTreeBefore
        timeout_tree_rechecked = $timeoutTreeRechecked
        timeout_tree_survivors = $timeoutTreeSurvivors
        launcher_failure_class = $failure
        source_and_harness_unchanged = $same
        stdout = $stdout
        stderr = $stderr
        origins = (Join-Path $PSScriptRoot "$RunName-origins.json")
        events = (Join-Path $PSScriptRoot "$RunName-events.jsonl")
        nested = (Join-Path $PSScriptRoot "$RunName-nested-report.json")
        wide = (Join-Path $PSScriptRoot "$RunName-wide-report.json")
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $result -Encoding utf8
}
if ($timedOut -or $forcedExit -or $exitCode -ne 0 -or -not $same) {
    throw 'scale_qualification_adverse_or_source_changed'
}

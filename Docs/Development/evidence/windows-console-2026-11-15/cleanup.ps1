$ErrorActionPreference = 'Stop'
$evidenceRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)
$profileParent = [System.IO.Path]::GetFullPath('C:/Users/GDesktop-1')
$roots = @{}
foreach ($directory in @($evidenceRoot, (Join-Path $evidenceRoot 'validation'))) {
    foreach ($file in Get-ChildItem -LiteralPath $directory -File -Filter '*.json') {
        $record = Get-Content -LiteralPath $file.FullName -Raw | ConvertFrom-Json
        foreach ($field in @('fixture_root', 'temporary_profile')) {
            $value = $record.$field
            if ($value -isnot [string] -or -not $value) { continue }
            $target = [System.IO.Path]::GetFullPath($value)
            if ([System.IO.Path]::GetDirectoryName($target) -ne $profileParent -or
                [System.IO.Path]::GetFileName($target) -notmatch '^\.tmp-(pr3166-final|windows-perf)-[a-f0-9]{12}$') {
                throw "Unexpected receipt-owned root: $target"
            }
            if (-not $roots.ContainsKey($target)) { $roots[$target] = @() }
            $roots[$target] += $file.Name
        }
    }
}
if (-not $roots.Count) { throw 'No receipt-owned roots found' }
$processes = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'"
foreach ($process in $processes) {
    $command = [string]$process.CommandLine
    if ($command -match 'windows-console-2026-11-15[\\/](probe|run_probe|run_tests)\.py') {
        throw "Owned profiling/test process is still active: $($process.ProcessId)"
    }
    foreach ($target in $roots.Keys) {
        if ($command.IndexOf($target, [StringComparison]::OrdinalIgnoreCase) -ge 0) {
            throw "A process still names an owned fixture: $($process.ProcessId)"
        }
    }
}
# Check every final absolute root before starting any recursive deletion.
foreach ($target in $roots.Keys) {
    if (Test-Path -LiteralPath $target) {
        $item = Get-Item -LiteralPath $target -Force
        if (-not $item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw "Fixture root itself is not an ordinary directory: $target"
        }
        if ($item.FullName -ne $target) { throw "Resolved fixture changed: $target" }
    }
}
$results = @()
foreach ($target in ($roots.Keys | Sort-Object)) {
    $status = 'missing'
    $detail = $null
    try {
        if (Test-Path -LiteralPath $target) {
            # Pytest may create current-directory links. Remove only their
            # checked in-root entries, without recursion or following targets.
            $links = @(Get-ChildItem -LiteralPath $target -Recurse -Force |
                Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint })
            foreach ($link in $links) {
                $entry = [System.IO.Path]::GetFullPath($link.FullName)
                if (-not $entry.StartsWith($target + '\', [StringComparison]::OrdinalIgnoreCase)) {
                    throw "Out-of-root link entry: $entry"
                }
                Remove-Item -LiteralPath $entry -Force
            }
            $remaining = @(Get-ChildItem -LiteralPath $target -Recurse -Force |
                Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint })
            if ($remaining.Count) { throw 'Fixture still contains a reparse entry' }
            Remove-Item -LiteralPath $target -Recurse
            if (Test-Path -LiteralPath $target) { throw 'Fixture remains after removal' }
            $status = 'removed'
        }
    } catch {
        $status = 'failed'
        $detail = $_.Exception.Message
    }
    $results += [pscustomobject][ordered]@{root=$target; receipts=$roots[$target]; status=$status; detail=$detail}
}
$report = [ordered]@{
    scope='Only exact current-task receipt-owned native fixture roots; real profile untouched'
    active_owned_processes=0
    roots=$results
}
$report | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $evidenceRoot 'cleanup.json') -Encoding utf8
$results | Group-Object status | Select-Object Name,Count
if (@($results | Where-Object status -eq 'failed').Count) { exit 1 }

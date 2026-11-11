param([Parameter(Mandatory = $true)][ValidatePattern('^[a-z0-9-]+$')][string]$RunName)
$ErrorActionPreference = 'Stop'
$privateTarget = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$expectedTarget = 'C:\Users\GDesktop-1\.codex\worktrees\workspace-project-guidance\tldw_tui\.superpowers\sdd\2026-10-20-workspace-project-semantic-index'
if ($privateTarget -ne $expectedTarget) { throw 'unexpected_private_test_target' }
$beforePath = Join-Path $privateTarget ($RunName + '-permission-before.json')
$afterPath = Join-Path $privateTarget ($RunName + '-permission-after.json')
if ((Test-Path -LiteralPath $beforePath) -or (Test-Path -LiteralPath $afterPath)) { throw 'permission_receipt_exists' }
$directoryAcl = Get-Acl -LiteralPath $privateTarget
$expectedSids = @('S-1-5-21-2888389458-149452071-4026681343-1004','S-1-5-21-3778582364-158406648-3898098024-3276331917')
$removeRules = @($directoryAcl.Access | Where-Object {
    $ruleSid = $_.IdentityReference.Translate([System.Security.Principal.SecurityIdentifier]).Value
    (-not $_.IsInherited) -and ($ruleSid -in $expectedSids) -and ([int]$_.FileSystemRights -eq 1245631) -and ([int]$_.InheritanceFlags -eq 3) -and ([int]$_.PropagationFlags -eq 0) -and ($_.AccessControlType -eq [System.Security.AccessControl.AccessControlType]::Allow)
})
if ($removeRules.Count -notin @(0,2)) { throw 'unexpected_private_test_acl_shape' }
@{path=$privateTarget;sddl=$directoryAcl.Sddl;matching_grants=$removeRules.Count;insertion_actor='unestablished'} | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $beforePath -Encoding utf8
foreach ($ruleEntry in $removeRules) { [void]$directoryAcl.RemoveAccessRuleSpecific($ruleEntry) }
if ($removeRules.Count) { Set-Acl -LiteralPath $privateTarget -AclObject $directoryAcl }
@{path=$privateTarget;sddl=(Get-Acl -LiteralPath $privateTarget).Sddl;removed_only_exact_grants=$removeRules.Count} | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $afterPath -Encoding utf8
$repoPath = (Resolve-Path -LiteralPath (Join-Path $privateTarget '../../..')).Path
Push-Location -LiteralPath $repoPath
try {
    $verifyCode = 'import pathlib,site; site.addsitedir("C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Lib/site-packages"); from tldw_chatbook.Utils.private_paths import verify_trusted_directory; sdd=pathlib.Path(".superpowers/sdd/2026-10-20-workspace-project-semantic-index").absolute(); rows=[(p,verify_trusted_directory(p,allow_shared_sticky=False)) for p in (sdd,sdd/"native-temp")]; [print(str(p),r.status.value,r.reason) for p,r in rows]'
    & '.venv/Scripts/python.exe' -c $verifyCode
    if ($LASTEXITCODE -ne 0) { throw 'private_test_trust_verification_failed' }
} finally { Pop-Location }

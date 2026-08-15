[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$ServerBinary,
    [Parameter(Mandatory = $true)][string]$TextPackageRoot,
    [Parameter(Mandatory = $true)][string]$ClonePackageRoot,
    [Parameter(Mandatory = $true)][string]$CloneReferenceWav,
    [Parameter(Mandatory = $true)][string]$CloneReferenceText,
    [Parameter(Mandatory = $true)][string]$TextRecipeId,
    [Parameter(Mandatory = $true)][int]$TextRecipeRevision,
    [Parameter(Mandatory = $true)][string]$TextPackageVariant,
    [Parameter(Mandatory = $true)][string]$TextModelId,
    [Parameter(Mandatory = $true)][string]$CloneRecipeId,
    [Parameter(Mandatory = $true)][int]$CloneRecipeRevision,
    [Parameter(Mandatory = $true)][string]$ClonePackageVariant,
    [Parameter(Mandatory = $true)][string]$CloneModelId,
    [Parameter(Mandatory = $true)][string]$CloneArtifactId,
    [Parameter(Mandatory = $true)][string]$CloneArtifactRevision,
    [Parameter(Mandatory = $true)][string]$CloneArtifactVariant,
    [Parameter(Mandatory = $true)][string]$EvidenceOutput,
    [string]$PythonCommand = "python"
)

$ErrorActionPreference = "Stop"
$privateRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("tldw-audio-cpp-uat-" + [guid]::NewGuid().ToString("N"))
$savedEnvironment = @{}
$environmentNames = @("HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "TLDW_CONFIG_PATH")

try {
    New-Item -ItemType Directory -Path $privateRoot | Out-Null
    foreach ($name in $environmentNames) {
        $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
    }
    $env:HOME = Join-Path $privateRoot "home"
    $env:XDG_CONFIG_HOME = Join-Path $privateRoot "config"
    $env:XDG_DATA_HOME = Join-Path $privateRoot "data"
    $env:TLDW_CONFIG_PATH = Join-Path $env:XDG_CONFIG_HOME "config.toml"
    $runtimeRoot = Join-Path $privateRoot "runtime"
    New-Item -ItemType Directory -Path $env:HOME, $env:XDG_CONFIG_HOME, $env:XDG_DATA_HOME, $runtimeRoot | Out-Null

    $arguments = @(
        "scripts/uat_audio_cpp_windows.py",
        "--server-binary", $ServerBinary,
        "--text-package-root", $TextPackageRoot,
        "--clone-package-root", $ClonePackageRoot,
        "--clone-reference-wav", $CloneReferenceWav,
        "--clone-reference-text", $CloneReferenceText,
        "--text-recipe-id", $TextRecipeId,
        "--text-recipe-revision", $TextRecipeRevision,
        "--text-package-variant", $TextPackageVariant,
        "--text-model-id", $TextModelId,
        "--clone-recipe-id", $CloneRecipeId,
        "--clone-recipe-revision", $CloneRecipeRevision,
        "--clone-package-variant", $ClonePackageVariant,
        "--clone-model-id", $CloneModelId,
        "--clone-artifact-id", $CloneArtifactId,
        "--clone-artifact-revision", $CloneArtifactRevision,
        "--clone-artifact-variant", $CloneArtifactVariant,
        "--runtime-root", $runtimeRoot,
        "--result-file", $EvidenceOutput
    )
    & $PythonCommand @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "The objective Windows audio.cpp UAT did not pass."
    }

    $player = New-Object System.Media.SoundPlayer
    foreach ($sample in @("text.wav", "clone.wav")) {
        $player.SoundLocation = Join-Path $runtimeRoot $sample
        $player.PlaySync()
    }
    $heard = Read-Host "Were both generated samples audible and intelligible? Type yes or no"
    $finalArguments = $arguments + @("--audible", $(if ($heard -eq "yes") { "yes" } else { "no" }))
    & $PythonCommand @finalArguments
    exit $LASTEXITCODE
}
finally {
    foreach ($name in $environmentNames) {
        [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], "Process")
    }
    if (Test-Path -LiteralPath $privateRoot) {
        Remove-Item -LiteralPath $privateRoot -Recurse -Force
    }
}

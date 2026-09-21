param(
    [Parameter(Mandatory = $true)]
    [string]$Slot,
    [string]$Version = "1.1.2",
    [string]$ShadowVersion = "challenger-0.2.0"
)

$ErrorActionPreference = "Stop"
$repository = Split-Path -Parent $PSScriptRoot
$logDirectory = Join-Path $repository "logs"
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
$logPath = Join-Path $logDirectory "scheduled-slots.log"

Push-Location $repository
try {
    $timestamp = (Get-Date).ToUniversalTime().ToString("o")
    Add-Content -LiteralPath $logPath -Value "$timestamp START $Slot"
    $preflightArguments = @(
        "compose", "run", "--rm", "nfl-bets", "pilot", "preflight",
        "--slot", $Slot, "--version", $Version
    )
    if ($ShadowVersion) {
        $preflightArguments += @("--shadow-version", $ShadowVersion)
    }
    & docker @preflightArguments *>> $logPath
    if ($LASTEXITCODE -ne 0) {
        throw "Scheduled slot $Slot failed zero-credit preflight"
    }
    $dockerArguments = @(
        "compose", "run", "--rm", "nfl-bets", "pilot", "capture",
        "--slot", $Slot, "--version", $Version
    )
    if ($ShadowVersion) {
        $dockerArguments += @("--shadow-version", $ShadowVersion)
    }
    & docker @dockerArguments *>> $logPath
    if ($LASTEXITCODE -ne 0) {
        throw "Scheduled slot $Slot failed with exit code $LASTEXITCODE"
    }
    $timestamp = (Get-Date).ToUniversalTime().ToString("o")
    Add-Content -LiteralPath $logPath -Value "$timestamp COMPLETE $Slot"
}
catch {
    $timestamp = (Get-Date).ToUniversalTime().ToString("o")
    Add-Content -LiteralPath $logPath -Value "$timestamp FAILED $Slot $($_.Exception.Message)"
    throw
}
finally {
    Pop-Location
}

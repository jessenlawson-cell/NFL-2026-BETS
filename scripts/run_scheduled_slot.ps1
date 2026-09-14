param(
    [Parameter(Mandatory = $true)]
    [string]$Slot
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
    docker compose run --rm nfl-bets pilot capture --slot $Slot *>> $logPath
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

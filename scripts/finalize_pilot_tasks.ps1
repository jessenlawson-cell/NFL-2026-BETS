param(
    [Parameter(Mandatory = $true)]
    [string]$PilotWeekBucket,
    [string]$Version = "1.1.2",
    [string]$ShadowVersion = "challenger-0.2.0"
)

$ErrorActionPreference = "Stop"
$repository = Split-Path -Parent $PSScriptRoot
$reportPath = Join-Path $repository "reports\manual_pilot_$PilotWeekBucket.json"
Push-Location $repository
try {
    docker compose run --rm nfl-bets pilot status --week-bucket $PilotWeekBucket `
        --version $Version --shadow-version $ShadowVersion
    if ($LASTEXITCODE -ne 0) {
        throw "Pilot verification failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

$pilot = Get-Content -Raw -LiteralPath $reportPath | ConvertFrom-Json
if ($pilot.status -ne "PASSED" -or $pilot.passed_slots -ne 16) {
    Write-Output "Pilot $PilotWeekBucket is incomplete; recurring tasks were not installed."
    exit 0
}
& (Join-Path $PSScriptRoot "install_scheduled_tasks.ps1") `
    -PilotWeekBucket $PilotWeekBucket -Version $Version -ShadowVersion $ShadowVersion

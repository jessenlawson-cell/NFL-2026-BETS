param(
    [string]$Version = "challenger-0.2.0"
)

$ErrorActionPreference = "Stop"
$repository = Split-Path -Parent $PSScriptRoot
$artifact = Join-Path $repository "artifacts\models\$Version\candidate.joblib"
$policy = Join-Path $repository "manifests\prospective_policy_$Version.json"
if ((Test-Path -LiteralPath $artifact) -and (Test-Path -LiteralPath $policy)) {
    Write-Output "$Version is already frozen."
    exit 0
}
if ((Test-Path -LiteralPath $artifact) -or (Test-Path -LiteralPath $policy)) {
    throw "Partial $Version freeze detected; refusing to overwrite it."
}

& (Join-Path $PSScriptRoot "run_feature_refresh.ps1") -CompletedWeek 2

Push-Location $repository
try {
    docker compose run --rm nfl-bets challenger train --version $Version --through-week 2
    if ($LASTEXITCODE -ne 0) {
        throw "Challenger freeze failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

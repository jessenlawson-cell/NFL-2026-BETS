param(
    [Parameter(Mandatory = $true)]
    [ValidateRange(2, 7)]
    [int]$CompletedWeek
)

$ErrorActionPreference = "Stop"
$repository = Split-Path -Parent $PSScriptRoot
$logDirectory = Join-Path $repository "logs"
New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
$logPath = Join-Path $logDirectory "feature-refresh.log"

function Invoke-NflBets {
    param([string[]]$Arguments)
    & docker compose run --rm nfl-bets @Arguments *>> $logPath
    if ($LASTEXITCODE -ne 0) {
        throw "nfl-bets $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
    }
}

Push-Location $repository
try {
    Invoke-NflBets -Arguments @("data", "sync", "--through", "2026")
    $manifestPath = Join-Path $repository "manifests\nflverse_sync.latest.json"
    $asOf = (Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json).retrieved_at_utc
    Invoke-NflBets -Arguments @("v11", "features", "build", "--as-of", $asOf)
    Invoke-NflBets -Arguments @(
        "challenger", "features", "build", "--as-of", $asOf,
        "--through-week", $CompletedWeek
    )
    Invoke-NflBets -Arguments @("validate")
}
finally {
    Pop-Location
}

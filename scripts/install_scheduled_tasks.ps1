param(
    [Parameter(Mandatory = $true)]
    [string]$PilotWeekBucket
)

$ErrorActionPreference = "Stop"
$repository = Split-Path -Parent $PSScriptRoot
$pilotReport = Join-Path $repository "reports\manual_pilot_$PilotWeekBucket.json"
if (-not (Test-Path -LiteralPath $pilotReport)) {
    throw "The required manual-pilot report does not exist: $pilotReport"
}
$pilot = Get-Content -Raw -LiteralPath $pilotReport | ConvertFrom-Json
if ($pilot.status -ne "PASSED" -or $pilot.passed_slots -ne 16) {
    throw "Scheduled tasks cannot be installed until all 16 manual pilot slots pass."
}
$timeZone = Get-TimeZone
if ($timeZone.Id -ne "Eastern Standard Time") {
    throw "Windows must use the Eastern Standard Time zone for America/Toronto scheduling."
}

$slots = @(
    @{ Name = "sunday_open_2000"; Day = "Sunday"; Time = "20:00" },
    @{ Name = "sunday_open_2330"; Day = "Sunday"; Time = "23:30" },
    @{ Name = "wednesday_0900"; Day = "Wednesday"; Time = "09:00" },
    @{ Name = "wednesday_1700"; Day = "Wednesday"; Time = "17:00" },
    @{ Name = "thursday_1200"; Day = "Thursday"; Time = "12:00" },
    @{ Name = "thursday_1930"; Day = "Thursday"; Time = "19:30" },
    @{ Name = "friday_1700"; Day = "Friday"; Time = "17:00" },
    @{ Name = "saturday_0900"; Day = "Saturday"; Time = "09:00" },
    @{ Name = "saturday_1700"; Day = "Saturday"; Time = "17:00" },
    @{ Name = "saturday_2300"; Day = "Saturday"; Time = "23:00" },
    @{ Name = "sunday_0845"; Day = "Sunday"; Time = "08:45" },
    @{ Name = "sunday_1245"; Day = "Sunday"; Time = "12:45" },
    @{ Name = "sunday_1545"; Day = "Sunday"; Time = "15:45" },
    @{ Name = "sunday_1945"; Day = "Sunday"; Time = "19:45" },
    @{ Name = "monday_1200"; Day = "Monday"; Time = "12:00" },
    @{ Name = "monday_1945"; Day = "Monday"; Time = "19:45" }
)

$runner = Join-Path $PSScriptRoot "run_scheduled_slot.ps1"
foreach ($slot in $slots) {
    $taskName = "NFL-BETS-$($slot.Name)"
    $arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$runner`" -Slot $($slot.Name)"
    $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arguments
    $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $slot.Day -At $slot.Time
    $settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew
    Register-ScheduledTask `
        -TaskName $taskName `
        -Description "NFL observer capture; no retries and PASS-only predictions." `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Force | Out-Null
}

Write-Output "Installed 16 NFL observer tasks after verified manual pilot $PilotWeekBucket."

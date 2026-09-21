param(
    [string]$PilotWeekBucket = "2026-09-22",
    [string]$Version = "1.1.2",
    [string]$ShadowVersion = "challenger-0.2.0",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
if ((Get-TimeZone).Id -ne "Eastern Standard Time") {
    throw "Windows must use Eastern Standard Time for America/Toronto scheduling."
}
$start = [datetime]::ParseExact(
    $PilotWeekBucket,
    "yyyy-MM-dd",
    [Globalization.CultureInfo]::InvariantCulture
)
if ($start.DayOfWeek -ne [DayOfWeek]::Tuesday) {
    throw "PilotWeekBucket must be a Tuesday."
}

$repository = Split-Path -Parent $PSScriptRoot
$captureRunner = Join-Path $PSScriptRoot "run_scheduled_slot.ps1"
$freezeRunner = Join-Path $PSScriptRoot "freeze_challenger.ps1"
$finalizer = Join-Path $PSScriptRoot "finalize_pilot_tasks.ps1"
$slots = @(
    @{ Name = "wednesday_0900"; Offset = 1; Time = "09:00" },
    @{ Name = "wednesday_1700"; Offset = 1; Time = "17:00" },
    @{ Name = "thursday_1200"; Offset = 2; Time = "12:00" },
    @{ Name = "thursday_1930"; Offset = 2; Time = "19:30" },
    @{ Name = "friday_1700"; Offset = 3; Time = "17:00" },
    @{ Name = "saturday_0900"; Offset = 4; Time = "09:00" },
    @{ Name = "saturday_1700"; Offset = 4; Time = "17:00" },
    @{ Name = "saturday_2300"; Offset = 4; Time = "23:00" },
    @{ Name = "sunday_0845"; Offset = 5; Time = "08:45" },
    @{ Name = "sunday_1245"; Offset = 5; Time = "12:45" },
    @{ Name = "sunday_1545"; Offset = 5; Time = "15:45" },
    @{ Name = "sunday_1945"; Offset = 5; Time = "19:45" },
    @{ Name = "sunday_open_2000"; Offset = 5; Time = "20:00" },
    @{ Name = "sunday_open_2330"; Offset = 5; Time = "23:30" },
    @{ Name = "monday_1200"; Offset = 6; Time = "12:00" },
    @{ Name = "monday_1945"; Offset = 6; Time = "19:45" }
)
$tasks = @(
    @{ Kind = "PREPARE"; Name = "prepare_1200"; At = $start.AddHours(12) },
    @{ Kind = "PREPARE"; Name = "prepare_1800"; At = $start.AddHours(18) }
)
foreach ($slot in $slots) {
    $time = [TimeSpan]::Parse($slot.Time)
    $tasks += @{
        Kind = "CAPTURE"
        Name = $slot.Name
        At = $start.AddDays($slot.Offset).Date.Add($time)
    }
}
$tasks += @{
    Kind = "FINALIZE"
    Name = "finalize_2015"
    At = $start.AddDays(6).Date.Add([TimeSpan]::Parse("20:15"))
}

$preview = $tasks | Sort-Object At | ForEach-Object {
    [pscustomobject]@{
        task_name = "NFL-BETS-PILOT-$($start.ToString('yyyyMMdd'))-$($_.Name)"
        kind = $_.Kind
        slot = if ($_.Kind -eq "CAPTURE") { $_.Name } else { $null }
        at = $_.At.ToString("yyyy-MM-ddTHH:mm:ss")
    }
}
if ($DryRun) {
    $preview | ConvertTo-Json -Depth 3
    exit 0
}

$principal = New-ScheduledTaskPrincipal `
    -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) `
    -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 2)
foreach ($task in $tasks) {
    $taskName = "NFL-BETS-PILOT-$($start.ToString('yyyyMMdd'))-$($task.Name)"
    if ($task.Kind -eq "CAPTURE") {
        $arguments = (
            "-NoProfile -ExecutionPolicy Bypass -File `"$captureRunner`" " +
            "-Slot $($task.Name) -Version $Version -ShadowVersion $ShadowVersion"
        )
    }
    elseif ($task.Kind -eq "PREPARE") {
        $arguments = (
            "-NoProfile -ExecutionPolicy Bypass -File `"$freezeRunner`" " +
            "-Version $ShadowVersion"
        )
    }
    else {
        $arguments = (
            "-NoProfile -ExecutionPolicy Bypass -File `"$finalizer`" " +
            "-PilotWeekBucket $PilotWeekBucket -Version $Version " +
            "-ShadowVersion $ShadowVersion"
        )
    }
    Register-ScheduledTask -TaskName $taskName `
        -Description "NFL observer pilot; one execution, no automatic retry, PASS-only." `
        -Action (New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arguments) `
        -Trigger (New-ScheduledTaskTrigger -Once -At $task.At) `
        -Settings $settings -Principal $principal -Force | Out-Null
}
$preview | ConvertTo-Json -Depth 3

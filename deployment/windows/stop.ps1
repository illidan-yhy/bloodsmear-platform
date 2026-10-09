$ErrorActionPreference = 'Stop'
$expectedPython = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '.venv\Scripts\python.exe'))
$targets = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object {
    $_.ExecutablePath -and [string]::Equals($_.ExecutablePath, $expectedPython, [StringComparison]::OrdinalIgnoreCase) -and
    $_.CommandLine -match '-m\s+bloodsmear\.cli\s+serve(?:\s|$)'
}
if (-not $targets) { Write-Host 'No service belonging to this deployment directory was found.'; exit 0 }
Write-Host 'For graceful shutdown, press Ctrl+C in the startup window. This stop script terminates this deployment process; an interrupted batch resumes on next startup.'
foreach ($targetProcess in $targets) { Stop-Process -Id $targetProcess.ProcessId }

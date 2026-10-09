[CmdletBinding()]
param([string]$PythonExe = '', [switch]$Offline, [switch]$CheckPythonOnly)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$root = $PSScriptRoot
if (-not (Test-Path -LiteralPath (Join-Path $root 'src'))) { throw 'Run this script from the extracted deployment package.' }
& (Join-Path $root 'verify-package.ps1')
$probeScript = Join-Path $root 'python_probe.py'
function Invoke-PythonProbe {
    param([string]$Executable, [string[]]$ProbeArguments)
    # Expected discovery failures must not become terminating NativeCommandError.
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $probeOutput = @(& $Executable @ProbeArguments 2>$null)
        $probeExitCode = $LASTEXITCODE
        return [PSCustomObject]@{ ExitCode = $probeExitCode; Value = ($probeOutput | Select-Object -Last 1) }
    } finally {
        $ErrorActionPreference = $previousPreference
    }
}
if (-not $PythonExe) {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        $probe = Invoke-PythonProbe -Executable $launcher.Source -ProbeArguments @('-3.11', $probeScript, 'executable')
        if ($probe.ExitCode -eq 0) { $PythonExe = $probe.Value }
    }
}
if (-not $PythonExe) {
    $candidate = Get-Command python -ErrorAction SilentlyContinue
    if ($candidate) { $PythonExe = $candidate.Source }
}
if ($PythonExe) {
    $probe = Invoke-PythonProbe -Executable $PythonExe -ProbeArguments @($probeScript, 'version')
    if ($probe.ExitCode -ne 0 -or $probe.Value -ne '3.11') { $PythonExe = '' }
}
if (-not $PythonExe) {
    $installer = Join-Path $root 'installers/python-3.11.9-amd64.exe'
    if (-not (Test-Path -LiteralPath $installer)) { throw 'Python 3.11 x64 not found. Install Python 3.11 or use -PythonExe C:\path\python.exe.' }
    Write-Host 'Installing Python 3.11 for the current user. No system PATH change.'
    $target = Join-Path $root 'runtime\python'
    $installArguments = @('/quiet','InstallAllUsers=0','PrependPath=0','Include_launcher=0','Include_test=0','Include_pip=1',('TargetDir="' + $target + '"'))
    $process = Start-Process -FilePath $installer -ArgumentList $installArguments -Wait -PassThru -WindowStyle Hidden
    if ($process.ExitCode -notin @(0,3010)) { throw "Python installer failed: $($process.ExitCode)" }
    $PythonExe = Join-Path $target 'python.exe'
}
$probe = Invoke-PythonProbe -Executable $PythonExe -ProbeArguments @($probeScript, 'validate')
if ($probe.ExitCode -ne 0) { throw 'Python version/architecture validation failed. Python 3.11 x64 is required.' }
Write-Host $probe.Value
if ($CheckPythonOnly) { return }
$venvPython = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $venvPython)) {
    & $PythonExe -m venv (Join-Path $root '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}
if ($Offline) {
    $wheels = Join-Path $root 'wheelhouse'
    if (-not (Test-Path -LiteralPath $wheels)) { throw 'This package has no offline wheelhouse. Use online installation.' }
    & $venvPython -m pip install --no-index --find-links $wheels -r (Join-Path $root 'requirements.lock')
} else {
    & $venvPython -m pip install --timeout 180 --retries 5 --only-binary=:all: -r (Join-Path $root 'requirements.lock')
}
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Save the terminal error for troubleshooting.' }
if (-not (Test-Path -LiteralPath '.env')) { Copy-Item -LiteralPath 'config.env.example' -Destination '.env' }
foreach ($folder in @('outputs','data','logs')) { New-Item -ItemType Directory -Force -Path (Join-Path $root $folder) | Out-Null }
$env:PYTHONPATH = Join-Path $root 'src'
& $venvPython 'deployment_check.py' --require-gpu --output 'outputs/environment-check.json'
if ($LASTEXITCODE -ne 0) { throw 'Installation finished, but the GPU check failed. Check driver and VC++ runtime; see deployment manual.' }
Write-Host 'Installation and GPU smoke test passed. Run 02_start.cmd and open http://127.0.0.1:8000.'

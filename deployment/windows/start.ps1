$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run 01_install.cmd first.' }
if (-not (Test-Path -LiteralPath '.env')) { Copy-Item -LiteralPath 'config.env.example' -Destination '.env' }
$env:PYTHONPATH = Join-Path $PSScriptRoot 'src'
Write-Host 'Open http://127.0.0.1:8000 after startup. Keep this window open; Ctrl+C stops the service.'
& $python -m bloodsmear.cli serve
exit $LASTEXITCODE

$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force -Path 'outputs' | Out-Null
$outputPath = Join-Path $PSScriptRoot 'outputs\computer-info.txt'
$os = Get-CimInstance Win32_OperatingSystem
$computer = Get-CimInstance Win32_ComputerSystem
$cpu = Get-CimInstance Win32_Processor
$gpu = Get-CimInstance Win32_VideoController
$lines = @('Environment collection (no patient data)', "Time: $(Get-Date -Format o)", "OS: $($os.Caption) $($os.Version) $($os.OSArchitecture)", "RAM GB: $([math]::Round($computer.TotalPhysicalMemory / 1GB, 1))", "CPU: $($cpu.Name)", "GPU: $($gpu.Name)", "Windows display driver: $($gpu.DriverVersion)")
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { $lines += & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader }
Get-PSDrive -PSProvider FileSystem | ForEach-Object { $lines += "Disk $($_.Name) free GB: $([math]::Round($_.Free / 1GB, 1))" }
$lines | Set-Content -LiteralPath $outputPath -Encoding UTF8
$lines | ForEach-Object { Write-Host $_ }
Write-Host "Saved: $outputPath"

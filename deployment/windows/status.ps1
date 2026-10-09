param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
try {
    $result = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health/ready" -TimeoutSec 10
    $result | Format-List
    if ($result.provider -ne 'CUDAExecutionProvider') { Write-Warning 'The service is not using CUDA.' }
} catch { Write-Host "Service is unavailable on port $Port. Check startup window and logs/app.log."; exit 1 }

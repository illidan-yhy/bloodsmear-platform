[CmdletBinding()]
param(
    [ValidateRange(1, 65535)][int]$Port = 8000,
    [string]$LogDir = ''
)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)

# Support both a packaged installation and invocation from the source checkout.
$statusProjectRoot = $PSScriptRoot
if (-not (Test-Path -LiteralPath (Join-Path $statusProjectRoot 'src'))) {
    $statusSourceRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
    if (Test-Path -LiteralPath (Join-Path $statusSourceRoot 'src')) { $statusProjectRoot = $statusSourceRoot }
}

function Get-StatusSetting([string]$Name, [string]$DefaultValue) {
    $settingValue = [Environment]::GetEnvironmentVariable($Name)
    if (-not [string]::IsNullOrWhiteSpace($settingValue)) { return $settingValue }
    $resolvedValue = $DefaultValue
    $settingFile = Join-Path $statusProjectRoot '.env'
    if (Test-Path -LiteralPath $settingFile) {
        foreach ($settingLine in (Get-Content -LiteralPath $settingFile -Encoding UTF8)) {
            if ($settingLine -match ('^\s*(?:export\s+)?' + [regex]::Escape($Name) + '\s*=\s*(.*?)\s*$')) {
                $settingValue = $Matches[1].Trim()
                if ($settingValue -match '^"((?:\\"|[^"])*)"\s*(?:#.*)?$') {
                    $resolvedValue = $Matches[1].Replace('\"', '"').Replace('\\', '\')
                } elseif ($settingValue -match "^'((?:\\'|[^'])*)'\s*(?:#.*)?$") {
                    $resolvedValue = $Matches[1].Replace("\'", "'").Replace('\\', '\')
                } else {
                    $resolvedValue = ($settingValue -replace '\s+#.*$', '').Trim()
                }
            }
        }
    }
    return $resolvedValue
}

function Read-StartupFailure {
    try {
        $recordPath = Join-Path $LogDir 'startup-status.json'
        if (-not (Test-Path -LiteralPath $recordPath)) { return $null }
        $record = Get-Content -Raw -LiteralPath $recordPath -Encoding UTF8 | ConvertFrom-Json
        $recordPort = 0
        $recordTime = [DateTimeOffset]::MinValue
        if (-not [int]::TryParse([string]$record.port, [ref]$recordPort) -or $recordPort -ne $Port) { return $null }
        if (-not [DateTimeOffset]::TryParse([string]$record.updated_at, [ref]$recordTime)) { return $null }
        if ($record.status -ne 'failed') { return $null }
        return $record
    } catch { return $null }
}

function Get-HttpErrorCode($Failure) {
    $bodies = @([string]$Failure.ErrorDetails.Message)
    foreach ($body in $bodies) {
        try {
            $payload = $body | ConvertFrom-Json
            if ([string]$payload.error.code -match '^[A-Z][A-Z0-9_]{0,63}$') { return [string]$payload.error.code }
        } catch { }
    }
    try {
        $response = $Failure.Exception.Response
        if ($response.PSObject.Methods.Name -contains 'GetResponseStream') {
            $reader = [IO.StreamReader]::new($response.GetResponseStream(), [Text.Encoding]::UTF8)
            try { $body = $reader.ReadToEnd() } finally { $reader.Dispose() }
        } else {
            $body = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
        }
        $payload = $body | ConvertFrom-Json
        if ([string]$payload.error.code -match '^[A-Z][A-Z0-9_]{0,63}$') { return [string]$payload.error.code }
    } catch { }
    return $null
}

function Show-GpuHint {
    Write-Host '请检查 NVIDIA 显卡驱动、VC++ x64 运行库及 GPU 部署依赖；修复后重新启动服务。'
}

try {
    if (-not $PSBoundParameters.ContainsKey('Port')) {
        $configuredPort = Get-StatusSetting 'BLOODSMEAR_PORT' '8000'
        $parsedPort = 0
        if (-not [int]::TryParse($configuredPort, [ref]$parsedPort) -or $parsedPort -lt 1 -or $parsedPort -gt 65535) { throw 'Invalid port configuration' }
        $Port = $parsedPort
    }
    if ([string]::IsNullOrWhiteSpace($LogDir)) { $LogDir = Get-StatusSetting 'BLOODSMEAR_LOG_DIR' 'logs' }
    if (-not [IO.Path]::IsPathRooted($LogDir)) { $LogDir = Join-Path $statusProjectRoot $LogDir }
} catch {
    Write-Host '无法读取状态检查配置，请检查 .env 中的端口和日志目录设置。'
    exit 3
}

$restoreStatusProxy = $false
$originalStatusProxy = $null
try {
    # The local readiness check must not depend on a browser/VPN proxy.
    $requestOptions = @{ Uri = "http://127.0.0.1:$Port/health/ready"; TimeoutSec = 10 }
    if ((Get-Command Invoke-RestMethod).Parameters.ContainsKey('NoProxy')) { $requestOptions.NoProxy = $true }
    else {
        $originalStatusProxy = [Net.WebRequest]::DefaultWebProxy
        $restoreStatusProxy = $true
        [Net.WebRequest]::DefaultWebProxy = $null
    }
    $result = Invoke-RestMethod @requestOptions
} catch {
    $failure = $_
    if ($null -ne $failure.Exception.Response) {
        $httpStatus = [int]$failure.Exception.Response.StatusCode
        $errorCode = Get-HttpErrorCode $failure
        if ($errorCode -eq 'GPU_UNAVAILABLE') {
            Write-Host "服务已运行，但 GPU 不可用（HTTP $httpStatus，GPU_UNAVAILABLE）。"
            Show-GpuHint
            exit 2
        }
        $codeSuffix = if ($errorCode) { "，错误代码：$errorCode" } else { '' }
        Write-Host "服务已响应，但未就绪（HTTP $httpStatus$codeSuffix）。请检查启动窗口和日志。"
        exit 3
    }
    $startupFailure = Read-StartupFailure
    if ($startupFailure -and $startupFailure.error_code -eq 'GPU_UNAVAILABLE') {
        Write-Host "服务未运行或端口 $Port 不可达；记录显示最近一次启动因 GPU 不可用失败（GPU_UNAVAILABLE）。"
        Write-Host "记录时间：$($startupFailure.updated_at)"
        Show-GpuHint
        exit 2
    }
    if ($startupFailure -and [string]$startupFailure.error_code -match '^[A-Z][A-Z0-9_]{0,63}$') {
        Write-Host "服务未运行或端口 $Port 不可达；最近一次启动失败（错误代码：$($startupFailure.error_code)）。请检查启动窗口和日志。"
        exit 3
    }
    Write-Host "服务未运行或端口 $Port 不可达，请先启动服务并核对端口；如启动失败，请查看启动窗口和日志。"
    exit 1
} finally {
    if ($restoreStatusProxy) { [Net.WebRequest]::DefaultWebProxy = $originalStatusProxy }
}

if ($result.status -eq 'ready' -and $result.provider -eq 'CUDAExecutionProvider') {
    Write-Host "服务已运行，GPU 正常（CUDAExecutionProvider，端口 $Port）。"
    exit 0
}
if ($result.status -eq 'ready' -and $result.provider -eq 'CPUExecutionProvider') {
    Write-Host "服务已运行，当前为 CPU 模式（未使用 GPU，端口 $Port）。"
    exit 0
}
Write-Host '服务已响应，但状态信息不符合预期，请检查启动窗口和日志。'
exit 3

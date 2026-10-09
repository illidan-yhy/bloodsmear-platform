$ErrorActionPreference = 'Stop'
$rootPath = [IO.Path]::GetFullPath($PSScriptRoot)
$manifestPath = Join-Path $rootPath 'PACKAGE_MANIFEST.json'
if (-not (Test-Path -LiteralPath $manifestPath)) { throw 'Package manifest is missing. Extract the entire ZIP before installation.' }
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($entry in $manifest.files.PSObject.Properties) {
    $fullPath = [IO.Path]::GetFullPath((Join-Path $rootPath $entry.Name))
    if (-not $fullPath.StartsWith($rootPath.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Invalid package path.' }
    if (-not (Test-Path -LiteralPath $fullPath -PathType Leaf)) { throw "Missing package file: $($entry.Name)" }
    $actual = (Get-FileHash -LiteralPath $fullPath -Algorithm SHA256).Hash
    if ($actual -ne $entry.Value) { throw "Package checksum mismatch: $($entry.Name)" }
}
Write-Host "Package checksum verification passed: $($manifest.name)"
